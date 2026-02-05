// SPDX-License-Identifier: MIT OR Apache-2.0
//! Core vault operations: store, retrieve, rotate, delete, grant, revoke secrets.

use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use graph_engine::{Direction, GraphEngine, PropertyValue};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{
    access::AccessController,
    attenuation::AttenuationPolicy,
    audit::{AuditLog, AuditOperation},
    encryption::Cipher,
    key::{MasterKey, SALT_SIZE},
    obfuscation::{self, Obfuscator},
    rate_limit::{Operation, RateLimiter},
    signing::EdgeSigner,
    ttl::GrantTTLTracker,
    Permission, Result, VaultConfig, VaultError, VersionInfo,
};

/// Secure secret storage with graph-based access control.
pub struct Vault {
    pub(crate) store: TensorStore,
    pub graph: std::sync::Arc<GraphEngine>,
    pub(crate) cipher: Cipher,
    pub(crate) obfuscator: Obfuscator,
    pub(crate) edge_signer: EdgeSigner,
    pub(crate) attenuation: AttenuationPolicy,
    pub(crate) ttl_tracker: GrantTTLTracker,
    rate_limiter: Option<RateLimiter>,
    pub(crate) max_versions: usize,
    config: VaultConfig,
}

impl Vault {
    /// Storage key prefix for vault secrets (obfuscated).
    const PREFIX: &'static str = "_vk:";
    /// Node ID for the root entity with universal access.
    pub const ROOT: &'static str = "node:root";
    /// Edge type for access grants.
    const ACCESS_EDGE: &'static str = "VAULT_ACCESS";
    /// Storage key for persisted salt.
    const SALT_KEY: &'static str = "_vault:salt";

    /// Create a new vault with the given master key.
    ///
    /// If `config.salt` is `None`, the vault will:
    /// 1. Try to load a previously persisted salt from storage
    /// 2. If none exists, generate a cryptographically random salt and persist it
    ///
    /// This ensures consistent key derivation across vault reopens while
    /// preventing hardcoded salt vulnerabilities.
    pub fn new(
        master_key: &[u8],
        graph: std::sync::Arc<GraphEngine>,
        store: TensorStore,
        config: VaultConfig,
    ) -> Result<Self> {
        // Determine the salt to use
        let derived = if config.salt.is_some() {
            // Explicit salt provided - use it directly
            let (key, _) = MasterKey::derive(master_key, &config)?;
            key
        } else if let Some(persisted_salt) = Self::load_salt(&store) {
            // Use persisted salt for consistency
            MasterKey::derive_with_salt(master_key, &persisted_salt, &config)?
        } else {
            // Generate new random salt and persist it
            let (key, new_salt) = MasterKey::derive(master_key, &config)?;
            Self::save_salt(&store, new_salt)?;
            key
        };

        let obfuscator = Obfuscator::new(&derived);
        let edge_signer = EdgeSigner::new(&derived);
        let cipher = Cipher::new(&derived);

        let saved_config = config.clone();
        let rate_limiter = config.rate_limit.map(RateLimiter::new);
        let max_versions = config.max_versions.max(1); // At least 1 version
        let attenuation = config.attenuation;

        let ttl_tracker = GrantTTLTracker::load(&store).unwrap_or_default();

        let vault = Self {
            store,
            graph,
            cipher,
            obfuscator,
            edge_signer,
            attenuation,
            ttl_tracker,
            rate_limiter,
            max_versions,
            config: saved_config,
        };

        vault.ensure_root_exists()?;
        vault.cleanup_expired_grants();
        Ok(vault)
    }

    fn load_salt(store: &TensorStore) -> Option<[u8; SALT_SIZE]> {
        store.get(Self::SALT_KEY).ok().and_then(|data| {
            if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("_salt") {
                if bytes.len() == SALT_SIZE {
                    let mut salt = [0u8; SALT_SIZE];
                    salt.copy_from_slice(bytes);
                    return Some(salt);
                }
            }
            None
        })
    }

    fn save_salt(store: &TensorStore, salt: [u8; SALT_SIZE]) -> Result<()> {
        let mut data = TensorData::new();
        data.set(
            "_salt",
            TensorValue::Scalar(ScalarValue::Bytes(salt.to_vec())),
        );
        store
            .put(Self::SALT_KEY, data)
            .map_err(|e| VaultError::StorageError(e.to_string()))
    }

    /// Create vault from NEUMANN_VAULT_KEY environment variable.
    pub fn from_env(graph: std::sync::Arc<GraphEngine>, store: TensorStore) -> Result<Self> {
        let key = std::env::var("NEUMANN_VAULT_KEY")
            .map_err(|_| VaultError::KeyDerivationError("NEUMANN_VAULT_KEY not set".to_string()))?;

        let decoded = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &key)
            .map_err(|e| {
                VaultError::KeyDerivationError(format!("Invalid base64 in NEUMANN_VAULT_KEY: {e}"))
            })?;

        Self::new(&decoded, graph, store, VaultConfig::default())
    }

    fn ensure_root_exists(&self) -> Result<()> {
        // Ensure root node exists in graph
        let _ = self.get_or_create_entity_node(Self::ROOT);

        // Also store in TensorStore for backwards compatibility
        if !self.store.exists(Self::ROOT) {
            let mut root = TensorData::new();
            root.set(
                "_type",
                TensorValue::Scalar(ScalarValue::String("vault_root".into())),
            );
            root.set(
                "_label",
                TensorValue::Scalar(ScalarValue::String("root".into())),
            );
            self.store
                .put(Self::ROOT, root)
                .map_err(|e| VaultError::StorageError(e.to_string()))?;
        }
        Ok(())
    }

    // ========== Node-based Graph API Helpers ==========

    /// Find a graph node by entity key without creating it.
    fn find_entity_node(&self, entity_key: &str) -> Option<u64> {
        self.graph
            .find_nodes_by_property("entity_key", &PropertyValue::String(entity_key.to_string()))
            .ok()
            .and_then(|nodes| nodes.first().map(|n| n.id))
    }

    /// Get or create a graph node for an entity key.
    pub(crate) fn get_or_create_entity_node(&self, entity_key: &str) -> u64 {
        if let Ok(nodes) = self
            .graph
            .find_nodes_by_property("entity_key", &PropertyValue::String(entity_key.to_string()))
        {
            if let Some(node) = nodes.first() {
                return node.id;
            }
        }

        let mut props = HashMap::new();
        props.insert(
            "entity_key".to_string(),
            PropertyValue::String(entity_key.to_string()),
        );

        self.graph.create_node("VaultEntity", props).unwrap_or(0)
    }

    /// Add a signed edge between two entity keys.
    ///
    /// For VAULT_ACCESS edges, also stores a `vault_capacity` property
    /// encoding the permission level (1=Read, 2=Write, 3=Admin) for
    /// bottleneck flow calculations.
    fn add_entity_graph_edge(&self, from_key: &str, to_key: &str, edge_type: &str) -> Result<u64> {
        let from_node = self.get_or_create_entity_node(from_key);
        let to_node = self.get_or_create_entity_node(to_key);
        let timestamp = Self::current_timestamp();

        let signature = self
            .edge_signer
            .sign_edge(from_key, to_key, edge_type, timestamp);

        let mut props = HashMap::new();
        props.insert("vault_sig".to_string(), PropertyValue::Bytes(signature));
        props.insert("vault_sig_ts".to_string(), PropertyValue::Int(timestamp));

        // Store capacity for bottleneck flow on VAULT_ACCESS edges
        if let Some(perm) = Permission::from_edge_type(edge_type) {
            props.insert(
                "vault_capacity".to_string(),
                PropertyValue::Int(perm.to_level()),
            );
        }

        self.graph
            .create_edge(from_node, to_node, edge_type, props, true)
            .map_err(|e| VaultError::GraphError(e.to_string()))
    }

    /// Get outgoing edges for an entity, returning (edge_id, target_key, edge_type).
    fn get_entity_outgoing_edges(&self, entity_key: &str) -> Vec<(u64, String, String)> {
        let node_id = self.get_or_create_entity_node(entity_key);
        let mut result = Vec::new();

        if let Ok(edges) = self.graph.edges_of(node_id, Direction::Outgoing) {
            for edge in edges {
                let target_id = if edge.from == node_id {
                    edge.to
                } else {
                    edge.from
                };
                if let Ok(target_node) = self.graph.get_node(target_id) {
                    if let Some(PropertyValue::String(key)) =
                        target_node.properties.get("entity_key")
                    {
                        result.push((edge.id, key.clone(), edge.edge_type.clone()));
                    }
                }
            }
        }

        result
    }

    /// Delete an edge by ID.
    fn delete_graph_edge(&self, edge_id: u64) -> Result<()> {
        self.graph
            .delete_edge(edge_id)
            .map_err(|e| VaultError::GraphError(e.to_string()))
    }

    fn vault_key(&self, secret_key: &str) -> String {
        let obfuscated = self.obfuscator.obfuscate_key(secret_key);
        format!("{}{}", Self::PREFIX, obfuscated)
    }

    fn blob_key(&self, secret_key: &str, nonce: &[u8]) -> String {
        self.obfuscator.generate_storage_id(secret_key, nonce)
    }

    pub(crate) fn secret_node_key(&self, secret_key: &str) -> String {
        let obfuscated = self.obfuscator.obfuscate_key(secret_key);
        format!("vault_secret:{obfuscated}")
    }

    fn current_timestamp() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    /// Store a secret.
    ///
    /// For existing secrets, this adds a new version. Old versions are kept
    /// up to `max_versions` limit.
    pub fn set(&self, requester: &str, key: &str, value: &str) -> Result<()> {
        // Rate limit check
        self.check_rate_limit(requester, Operation::Set)?;

        // For new secrets, only root can create. For existing, need Write permission.
        let secret_node = self.secret_node_key(key);
        let vault_storage_key = self.vault_key(key);
        let is_update = self.store.exists(&secret_node);

        if is_update {
            self.check_access_with_permission(requester, key, Permission::Write)?;
        } else if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can create new secrets".to_string(),
            ));
        }

        // Pad plaintext to hide length, then encrypt
        let padded = obfuscation::pad_plaintext(value.as_bytes())?;
        let (ciphertext, nonce) = self.cipher.encrypt(&padded)?;
        let timestamp = Self::current_timestamp();

        // Store ciphertext in separate blob for pointer indirection
        let blob_storage_key = self.blob_key(key, &nonce);
        let mut blob_tensor = TensorData::new();
        blob_tensor.set("_data", TensorValue::Scalar(ScalarValue::Bytes(ciphertext)));
        blob_tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
        );
        blob_tensor.set("_ts", TensorValue::Scalar(ScalarValue::Int(timestamp)));
        self.store
            .put(&blob_storage_key, blob_tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        // Get or create metadata tensor
        let mut tensor = if is_update {
            self.store
                .get(&vault_storage_key)
                .map_err(|_| VaultError::NotFound(key.to_string()))?
        } else {
            // Encrypt the original key name for list() to work
            let (encrypted_key, key_nonce) = self.cipher.encrypt(key.as_bytes())?;

            // Encrypt metadata using AEAD
            let creator_bytes = requester.as_bytes();
            let obfuscated_creator = self.obfuscator.encrypt_metadata(creator_bytes)?;
            let timestamp_bytes = timestamp.to_le_bytes();
            let obfuscated_timestamp = self.obfuscator.encrypt_metadata(&timestamp_bytes)?;

            let mut t = TensorData::new();
            t.set(
                "_key_enc",
                TensorValue::Scalar(ScalarValue::Bytes(encrypted_key)),
            );
            t.set(
                "_key_nonce",
                TensorValue::Scalar(ScalarValue::Bytes(key_nonce.to_vec())),
            );
            t.set(
                "_creator_obf",
                TensorValue::Scalar(ScalarValue::Bytes(obfuscated_creator)),
            );
            t.set(
                "_created_obf",
                TensorValue::Scalar(ScalarValue::Bytes(obfuscated_timestamp)),
            );
            t
        };

        // Update version info
        let mut versions = Self::get_version_blobs(&tensor);
        versions.push(blob_storage_key.clone());

        // Prune old versions if exceeding max
        while versions.len() > self.max_versions {
            if let Some(old_blob) = versions.first() {
                self.store.delete(old_blob).ok();
            }
            versions.remove(0);
        }

        // Update tensor with new version info
        tensor.set("_blob", TensorValue::Pointer(blob_storage_key));
        tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
        );
        tensor.set("_versions", TensorValue::Pointers(versions));
        tensor.set("_version_count", TensorValue::Scalar(ScalarValue::Int(1)));

        self.store
            .put(vault_storage_key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        // Create secret node for access control if it doesn't exist
        if !is_update {
            let mut node = TensorData::new();
            node.set(
                "_type",
                TensorValue::Scalar(ScalarValue::String("vault_secret".into())),
            );
            node.set(
                "_secret_key",
                TensorValue::Scalar(ScalarValue::String(key.to_string())),
            );
            self.store
                .put(&secret_node, node)
                .map_err(|e| VaultError::StorageError(e.to_string()))?;

            // Root always has Admin access to secrets it creates
            let edge_type = format!("{}{}", Self::ACCESS_EDGE, Permission::Admin.edge_suffix());
            self.add_entity_graph_edge(Self::ROOT, &secret_node, &edge_type)?;
        }

        // Log audit
        self.log_operation(requester, key, &AuditOperation::Set);

        Ok(())
    }

    fn get_version_blobs(tensor: &TensorData) -> Vec<String> {
        match tensor.get("_versions") {
            Some(TensorValue::Pointers(v)) => v.clone(),
            _ => {
                // Backward compatibility: single blob is version 1
                if let Some(TensorValue::Pointer(blob)) = tensor.get("_blob") {
                    vec![blob.clone()]
                } else {
                    vec![]
                }
            },
        }
    }

    /// Retrieve a secret (requires graph path from requester to secret).
    pub fn get(&self, requester: &str, key: &str) -> Result<String> {
        // Opportunistic cleanup of expired grants
        self.cleanup_expired_grants();

        // Rate limit check
        self.check_rate_limit(requester, Operation::Get)?;

        self.check_access(requester, key)?;

        let tensor = self
            .store
            .get(&self.vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Follow pointer indirection to get ciphertext blob
        let blob_key = match tensor.get("_blob") {
            Some(TensorValue::Pointer(p)) => p.clone(),
            _ => return Err(VaultError::NotFound(key.to_string())),
        };

        let blob_tensor = self
            .store
            .get(&blob_key)
            .map_err(|_| VaultError::CryptoError("Blob not found".to_string()))?;

        let ciphertext = match blob_tensor.get("_data") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return Err(VaultError::CryptoError("Blob data missing".to_string())),
        };

        let nonce = match tensor.get("_nonce") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return Err(VaultError::CryptoError("Missing nonce".to_string())),
        };

        // Decrypt and unpad
        let padded = self.cipher.decrypt(&ciphertext, &nonce)?;
        let plaintext = obfuscation::unpad_plaintext(&padded)?;

        // Log audit
        self.log_operation(requester, key, &AuditOperation::Get);

        String::from_utf8(plaintext)
            .map_err(|e| VaultError::CryptoError(format!("Invalid UTF-8: {e}")))
    }

    /// Retrieve a specific version of a secret.
    ///
    /// Version numbers are 1-based. Version 1 is the oldest kept version.
    pub fn get_version(&self, requester: &str, key: &str, version: u32) -> Result<String> {
        self.check_access(requester, key)?;

        let tensor = self
            .store
            .get(&self.vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        let versions = Self::get_version_blobs(&tensor);
        let idx = (version as usize).saturating_sub(1);

        if idx >= versions.len() {
            return Err(VaultError::NotFound(format!(
                "version {version} not found for '{key}'"
            )));
        }

        let blob_key = &versions[idx];
        let blob_tensor = self
            .store
            .get(blob_key)
            .map_err(|_| VaultError::CryptoError("Blob not found".to_string()))?;

        let ciphertext = match blob_tensor.get("_data") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return Err(VaultError::CryptoError("Blob data missing".to_string())),
        };

        let nonce = match blob_tensor.get("_nonce") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => {
                // Fallback to main tensor nonce for backward compatibility
                match tensor.get("_nonce") {
                    Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
                    _ => return Err(VaultError::CryptoError("Missing nonce".to_string())),
                }
            },
        };

        let padded = self.cipher.decrypt(&ciphertext, &nonce)?;
        let plaintext = obfuscation::unpad_plaintext(&padded)?;

        String::from_utf8(plaintext)
            .map_err(|e| VaultError::CryptoError(format!("Invalid UTF-8: {e}")))
    }

    /// List all versions of a secret with metadata.
    pub fn list_versions(&self, requester: &str, key: &str) -> Result<Vec<VersionInfo>> {
        self.check_access(requester, key)?;

        let tensor = self
            .store
            .get(&self.vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        let versions = Self::get_version_blobs(&tensor);
        let mut infos = Vec::with_capacity(versions.len());

        for (idx, blob_key) in versions.iter().enumerate() {
            let created_at = if let Ok(blob) = self.store.get(blob_key) {
                match blob.get("_ts") {
                    Some(TensorValue::Scalar(ScalarValue::Int(ts))) => *ts,
                    _ => 0,
                }
            } else {
                0
            };

            infos.push(VersionInfo {
                version: (idx + 1) as u32,
                created_at,
            });
        }

        Ok(infos)
    }

    /// Get the current version number.
    pub fn current_version(&self, requester: &str, key: &str) -> Result<u32> {
        self.check_access(requester, key)?;

        let tensor = self
            .store
            .get(&self.vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        let versions = Self::get_version_blobs(&tensor);
        Ok(versions.len() as u32)
    }

    /// Rollback to a previous version (requires Write permission).
    ///
    /// This creates a new version with the content from the specified version.
    pub fn rollback(&self, requester: &str, key: &str, version: u32) -> Result<()> {
        let value = self.get_version(requester, key, version)?;
        self.set(requester, key, &value)
    }

    /// Grant access to a secret with Admin permission (backward compatible).
    pub fn grant(&self, requester: &str, entity: &str, key: &str) -> Result<()> {
        self.grant_with_permission(requester, entity, key, Permission::Admin)
    }

    /// Grant access to a secret with a specific permission level.
    pub fn grant_with_permission(
        &self,
        requester: &str,
        entity: &str,
        key: &str,
        level: Permission,
    ) -> Result<()> {
        // Rate limit check
        self.check_rate_limit(requester, Operation::Grant)?;

        // Requester needs Admin permission to grant access
        self.check_access_with_permission(requester, key, Permission::Admin)?;

        let secret_node = self.secret_node_key(key);
        if !self.store.exists(&secret_node) {
            return Err(VaultError::NotFound(key.to_string()));
        }

        // Create edge with permission level encoded in type
        let edge_type = format!("{}{}", Self::ACCESS_EDGE, level.edge_suffix());
        self.add_entity_graph_edge(entity, &secret_node, &edge_type)?;

        // Log audit
        self.log_operation(
            requester,
            key,
            &AuditOperation::Grant {
                to: entity.to_string(),
                permission: format!("{level:?}").to_lowercase(),
            },
        );

        Ok(())
    }

    /// Grant access to a secret with a TTL (time-to-live).
    ///
    /// The grant will automatically expire after the specified duration.
    /// Expired grants are cleaned up opportunistically during vault operations.
    pub fn grant_with_ttl(
        &self,
        requester: &str,
        entity: &str,
        key: &str,
        level: Permission,
        ttl: Duration,
    ) -> Result<()> {
        // First, create the grant normally
        self.grant_with_permission(requester, entity, key, level)?;

        // Then register with TTL tracker and persist
        self.ttl_tracker.add(entity, key, ttl);
        self.ttl_tracker.persist(&self.store).ok();

        Ok(())
    }

    /// Revoke access to a secret (requires Admin permission).
    pub fn revoke(&self, requester: &str, entity: &str, key: &str) -> Result<()> {
        // Only those with Admin permission can revoke access
        self.check_access_with_permission(requester, key, Permission::Admin)?;

        let secret_node = self.secret_node_key(key);

        // Find and delete the access edge (handles all permission levels)
        for (edge_id, to, edge_type) in self.get_entity_outgoing_edges(entity) {
            if to == secret_node && edge_type.starts_with(Self::ACCESS_EDGE) {
                self.delete_graph_edge(edge_id)?;
            }
        }

        // Remove from TTL tracker if present and persist
        if self.ttl_tracker.remove(entity, key) {
            self.ttl_tracker.persist(&self.store).ok();
        }

        // Log audit
        self.log_operation(
            requester,
            key,
            &AuditOperation::Revoke {
                from: entity.to_string(),
            },
        );

        Ok(())
    }

    /// Clean up expired grants.
    ///
    /// Returns the number of grants that were revoked.
    pub fn cleanup_expired_grants(&self) -> usize {
        let expired = self.ttl_tracker.get_expired();
        let mut revoked = 0;

        for (entity, key) in expired {
            let secret_node = self.secret_node_key(&key);

            for (edge_id, to, edge_type) in self.get_entity_outgoing_edges(&entity) {
                if to == secret_node
                    && edge_type.starts_with(Self::ACCESS_EDGE)
                    && self.delete_graph_edge(edge_id).is_ok()
                {
                    revoked += 1;
                }
            }
        }

        if revoked > 0 {
            self.ttl_tracker.persist(&self.store).ok();
        }

        revoked
    }

    /// Delete a secret (requires Admin permission).
    ///
    /// This deletes all versions of the secret.
    pub fn delete(&self, requester: &str, key: &str) -> Result<()> {
        self.check_access_with_permission(requester, key, Permission::Admin)?;

        let vault_storage_key = self.vault_key(key);

        // Delete all version blobs
        if let Ok(tensor) = self.store.get(&vault_storage_key) {
            for blob_key in Self::get_version_blobs(&tensor) {
                self.store.delete(&blob_key).ok();
            }
        }

        self.store
            .delete(&vault_storage_key)
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Clean up the secret node from TensorStore
        let secret_node = self.secret_node_key(key);
        self.store.delete(&secret_node).ok();

        // Explicitly remove access edges and clean up TTL entries, then delete node
        if let Some(node_id) = self.find_entity_node(&secret_node) {
            if let Ok(edges) = self.graph.edges_of(node_id, Direction::Incoming) {
                for edge in edges {
                    if edge.edge_type.starts_with(Self::ACCESS_EDGE) {
                        if let Ok(from_node) = self.graph.get_node(edge.from) {
                            if let Some(PropertyValue::String(entity_key)) =
                                from_node.properties.get("entity_key")
                            {
                                self.ttl_tracker.remove(entity_key, key);
                            }
                        }
                    }
                    self.delete_graph_edge(edge.id).ok();
                }
            }
            self.graph.delete_node(node_id).ok();
            self.ttl_tracker.persist(&self.store).ok();
        }

        // Log audit
        self.log_operation(requester, key, &AuditOperation::Delete);

        Ok(())
    }

    /// Rotate a secret value (requires Write permission).
    ///
    /// Creates a new version. Previous versions are kept up to `max_versions`.
    pub fn rotate(&self, requester: &str, key: &str, new_value: &str) -> Result<()> {
        self.check_access_with_permission(requester, key, Permission::Write)?;

        let vault_storage_key = self.vault_key(key);

        if !self.store.exists(&vault_storage_key) {
            return Err(VaultError::NotFound(key.to_string()));
        }

        // Pad and encrypt new value
        let padded = obfuscation::pad_plaintext(new_value.as_bytes())?;
        let (ciphertext, nonce) = self.cipher.encrypt(&padded)?;
        let timestamp = Self::current_timestamp();

        // Store new ciphertext blob with version info
        let new_blob_key = self.blob_key(key, &nonce);
        let mut blob_tensor = TensorData::new();
        blob_tensor.set("_data", TensorValue::Scalar(ScalarValue::Bytes(ciphertext)));
        blob_tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
        );
        blob_tensor.set("_ts", TensorValue::Scalar(ScalarValue::Int(timestamp)));
        self.store
            .put(&new_blob_key, blob_tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        // Get current tensor
        let mut tensor = self
            .store
            .get(&vault_storage_key)
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Update version list
        let mut versions = Self::get_version_blobs(&tensor);
        versions.push(new_blob_key.clone());

        // Prune old versions if exceeding max
        while versions.len() > self.max_versions {
            if let Some(old_blob) = versions.first() {
                self.store.delete(old_blob).ok();
            }
            versions.remove(0);
        }

        // Update metadata with new blob pointer and nonce
        tensor.set("_blob", TensorValue::Pointer(new_blob_key));
        tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
        );
        tensor.set("_versions", TensorValue::Pointers(versions));

        // Encrypt rotation metadata using AEAD
        let rotator_obf = self.obfuscator.encrypt_metadata(requester.as_bytes())?;
        let timestamp_obf = self.obfuscator.encrypt_metadata(&timestamp.to_le_bytes())?;

        tensor.set(
            "_rotator_obf",
            TensorValue::Scalar(ScalarValue::Bytes(rotator_obf)),
        );
        tensor.set(
            "_rotated_obf",
            TensorValue::Scalar(ScalarValue::Bytes(timestamp_obf)),
        );

        self.store
            .put(vault_storage_key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        // Log audit
        self.log_operation(requester, key, &AuditOperation::Rotate);

        Ok(())
    }

    /// List secret keys matching a pattern.
    pub fn list(&self, requester: &str, pattern: &str) -> Result<Vec<String>> {
        // Rate limit check
        self.check_rate_limit(requester, Operation::List)?;

        let all_keys = self.store.scan(Self::PREFIX);

        let mut accessible = Vec::new();
        for vault_key in all_keys {
            if let Ok(tensor) = self.store.get(&vault_key) {
                let original_key = self.decrypt_key_name(&tensor);
                if let Some(key) = original_key {
                    if Self::matches_pattern(&key, pattern) && self.has_access(requester, &key) {
                        accessible.push(key);
                    }
                }
            }
        }

        // Log audit (using pattern as key)
        self.log_operation(requester, pattern, &AuditOperation::List);

        Ok(accessible)
    }

    fn decrypt_key_name(&self, tensor: &TensorData) -> Option<String> {
        let encrypted_key = match tensor.get("_key_enc") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return None,
        };
        let key_nonce = match tensor.get("_key_nonce") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return None,
        };

        self.cipher
            .decrypt(&encrypted_key, &key_nonce)
            .ok()
            .and_then(|bytes| String::from_utf8(bytes).ok())
    }

    /// Create a scoped vault view for a specific entity.
    pub fn scope(&self, entity: &str) -> crate::scoped::ScopedVault<'_> {
        crate::scoped::ScopedVault::new(self, entity)
    }

    /// Create a namespaced vault view for multi-tenant isolation.
    pub fn namespace(
        &self,
        namespace: &str,
        identity: &str,
    ) -> crate::namespaced::NamespacedVault<'_> {
        crate::namespaced::NamespacedVault::new(self, namespace, identity)
    }

    /// Access the underlying graph engine (for testing/benchmarking).
    pub fn graph(&self) -> &std::sync::Arc<GraphEngine> {
        &self.graph
    }

    fn check_access(&self, requester: &str, key: &str) -> Result<()> {
        self.check_access_with_permission(requester, key, Permission::Read)
    }

    fn check_access_with_permission(
        &self,
        requester: &str,
        key: &str,
        required: Permission,
    ) -> Result<()> {
        if requester == Self::ROOT {
            return Ok(());
        }

        let secret_node = self.secret_node_key(key);

        if AccessController::check_path_with_permission_verified(
            &self.graph,
            requester,
            &secret_node,
            required,
            &self.edge_signer,
            &self.attenuation,
        ) {
            Ok(())
        } else if AccessController::check_path(&self.graph, requester, &secret_node) {
            Err(VaultError::InsufficientPermission(format!(
                "{requester} has access but not {required} permission on '{key}'"
            )))
        } else {
            Err(VaultError::AccessDenied(format!(
                "No path from {requester} to secret '{key}'"
            )))
        }
    }

    fn has_access(&self, requester: &str, key: &str) -> bool {
        if requester == Self::ROOT {
            return true;
        }

        let secret_node = self.secret_node_key(key);
        AccessController::get_permission_level_verified(
            &self.graph,
            requester,
            &secret_node,
            &self.edge_signer,
            &self.attenuation,
        )
        .is_some()
    }

    fn check_rate_limit(&self, requester: &str, op: Operation) -> Result<()> {
        if requester == Self::ROOT {
            return Ok(());
        }

        if let Some(limiter) = &self.rate_limiter {
            limiter
                .check_and_record(requester, op)
                .map_err(VaultError::RateLimited)
        } else {
            Ok(())
        }
    }

    /// Get the permission level for a requester on a secret.
    pub fn get_permission(&self, requester: &str, key: &str) -> Option<Permission> {
        if requester == Self::ROOT {
            return Some(Permission::Admin);
        }

        let secret_node = self.secret_node_key(key);
        AccessController::get_permission_level_verified(
            &self.graph,
            requester,
            &secret_node,
            &self.edge_signer,
            &self.attenuation,
        )
    }

    fn log_operation(&self, requester: &str, key: &str, operation: &AuditOperation) {
        let obfuscated_key = self.obfuscator.obfuscate_key(key);
        let audit_log = AuditLog::new(&self.store);
        audit_log.record(requester, &obfuscated_key, operation);
    }

    /// Query audit entries for a specific secret.
    pub fn audit_log(&self, key: &str) -> Vec<crate::audit::AuditEntry> {
        let obfuscated_key = self.obfuscator.obfuscate_key(key);
        let audit = AuditLog::new(&self.store);
        audit.by_secret(&obfuscated_key)
    }

    /// Query audit entries by entity (who performed operations).
    pub fn audit_by_entity(&self, entity: &str) -> Vec<crate::audit::AuditEntry> {
        let audit = AuditLog::new(&self.store);
        audit.by_entity(entity)
    }

    /// Query audit entries since a timestamp (unix milliseconds).
    pub fn audit_since(&self, since_millis: i64) -> Vec<crate::audit::AuditEntry> {
        let audit = AuditLog::new(&self.store);
        audit.since(since_millis)
    }

    /// Get recent audit entries.
    pub fn audit_recent(&self, limit: usize) -> Vec<crate::audit::AuditEntry> {
        let audit = AuditLog::new(&self.store);
        audit.recent(limit)
    }

    fn matches_pattern(key: &str, pattern: &str) -> bool {
        if pattern.is_empty() || pattern == "*" {
            return true;
        }

        if let Some(prefix) = pattern.strip_suffix('*') {
            key.starts_with(prefix)
        } else {
            key == pattern
        }
    }

    /// Re-encrypt all secrets and re-sign all edges under new key material
    /// derived from a new password. Returns the number of secrets re-encrypted.
    pub fn rotate_master_key(&mut self, new_password: &[u8]) -> Result<usize> {
        let vault_keys = self.store.scan(Self::PREFIX);
        let decrypted_secrets = self.decrypt_all_secrets(&vault_keys)?;
        let secrets_count = decrypted_secrets.len();
        let edge_records = self.collect_edge_topology();

        // Derive new key material
        let mut new_salt = [0u8; SALT_SIZE];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut new_salt);
        let new_master = MasterKey::derive_with_salt(new_password, &new_salt, &self.config)?;
        let new_cipher = Cipher::new(&new_master);
        let new_obfuscator = Obfuscator::new(&new_master);
        let new_signer = EdgeSigner::new(&new_master);

        // Delete old storage entries
        self.delete_old_entries(&vault_keys);

        // Re-encrypt under new keys
        let old_to_new_node_map =
            self.reencrypt_secrets(&decrypted_secrets, &new_cipher, &new_obfuscator)?;

        // Re-sign edges
        self.resign_edges(&edge_records, &new_signer, &old_to_new_node_map);

        // Swap key material and persist
        Self::save_salt(&self.store, new_salt)?;
        self.cipher = new_cipher;
        self.obfuscator = new_obfuscator;
        self.edge_signer = new_signer;
        self.ttl_tracker.persist(&self.store).ok();

        let audit_log = AuditLog::new(&self.store);
        audit_log.record(
            Self::ROOT,
            "master_key",
            &AuditOperation::RotateMasterKey { secrets_count },
        );

        Ok(secrets_count)
    }

    fn decrypt_all_secrets(&self, vault_keys: &[String]) -> Result<Vec<DecryptedSecret>> {
        let mut secrets = Vec::new();
        for vault_key in vault_keys {
            let tensor = self
                .store
                .get(vault_key)
                .map_err(|e| VaultError::StorageError(e.to_string()))?;

            let Some(name) = self.decrypt_key_name(&tensor) else {
                continue;
            };

            let creator = match tensor.get("_creator_obf") {
                Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => {
                    self.obfuscator.decrypt_metadata(b)?
                },
                _ => Vec::new(),
            };
            let created = match tensor.get("_created_obf") {
                Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => {
                    self.obfuscator.decrypt_metadata(b)?
                },
                _ => Vec::new(),
            };

            let versions = self.decrypt_versions(&tensor)?;
            secrets.push(DecryptedSecret {
                name,
                versions,
                creator,
                created,
            });
        }
        Ok(secrets)
    }

    fn decrypt_versions(&self, tensor: &TensorData) -> Result<Vec<(String, i64)>> {
        let mut versions = Vec::new();
        for blob_key in Self::get_version_blobs(tensor) {
            let blob = self
                .store
                .get(&blob_key)
                .map_err(|e| VaultError::StorageError(e.to_string()))?;

            let (
                Some(TensorValue::Scalar(ScalarValue::Bytes(ciphertext))),
                Some(TensorValue::Scalar(ScalarValue::Bytes(nonce))),
            ) = (blob.get("_data"), blob.get("_nonce"))
            else {
                continue;
            };

            let timestamp = match blob.get("_ts") {
                Some(TensorValue::Scalar(ScalarValue::Int(ts))) => *ts,
                _ => 0,
            };

            let padded = self.cipher.decrypt(ciphertext, nonce)?;
            let plaintext = obfuscation::unpad_plaintext(&padded)?;
            let value = String::from_utf8(plaintext)
                .map_err(|e| VaultError::CryptoError(format!("Invalid UTF-8: {e}")))?;
            versions.push((value, timestamp));
        }
        Ok(versions)
    }

    fn collect_edge_topology(&self) -> Vec<EdgeRecord> {
        let mut records = Vec::new();
        let edge_types = [
            "VAULT_ACCESS_READ",
            "VAULT_ACCESS_WRITE",
            "VAULT_ACCESS_ADMIN",
            "VAULT_ACCESS",
        ];

        for edge_type in &edge_types {
            let Ok(edges) = self.graph.find_edges_by_type(edge_type) else {
                continue;
            };
            for edge in edges {
                let from_key = self.node_entity_key(edge.from);
                let to_key = self.node_entity_key(edge.to);
                if let (Some(from), Some(to)) = (from_key, to_key) {
                    let has_sig = edge.properties.contains_key("vault_sig");
                    let capacity = match edge.properties.get("vault_capacity") {
                        Some(PropertyValue::Int(c)) => Some(*c),
                        _ => None,
                    };
                    records.push(EdgeRecord {
                        edge_id: edge.id,
                        from_key: from,
                        to_key: to,
                        edge_type: edge.edge_type.clone(),
                        capacity,
                        has_sig,
                    });
                }
            }
        }
        records
    }

    fn node_entity_key(&self, node_id: u64) -> Option<String> {
        self.graph
            .get_node(node_id)
            .ok()
            .and_then(|n| match n.properties.get("entity_key") {
                Some(PropertyValue::String(s)) => Some(s.clone()),
                _ => None,
            })
    }

    fn delete_old_entries(&self, vault_keys: &[String]) {
        for vault_key in vault_keys {
            if let Ok(t) = self.store.get(vault_key) {
                for blob_key in Self::get_version_blobs(&t) {
                    self.store.delete(&blob_key).ok();
                }
            }
            self.store.delete(vault_key).ok();
        }
        for key in &self.store.scan("vault_secret:") {
            self.store.delete(key).ok();
        }
    }

    fn reencrypt_secrets(
        &self,
        secrets: &[DecryptedSecret],
        new_cipher: &Cipher,
        new_obfuscator: &Obfuscator,
    ) -> Result<HashMap<String, String>> {
        let mut old_to_new = HashMap::new();

        for secret in secrets {
            let new_vault_key = format!(
                "{}{}",
                Self::PREFIX,
                new_obfuscator.obfuscate_key(&secret.name)
            );
            let new_secret_node = format!(
                "vault_secret:{}",
                new_obfuscator.obfuscate_key(&secret.name)
            );
            let old_secret_node = format!(
                "vault_secret:{}",
                self.obfuscator.obfuscate_key(&secret.name)
            );

            old_to_new.insert(old_secret_node.clone(), new_secret_node.clone());

            let (version_keys, latest_blob, latest_nonce) =
                self.reencrypt_versions(secret, new_cipher, new_obfuscator)?;

            self.store_rotated_metadata(
                secret,
                &new_vault_key,
                &new_secret_node,
                new_cipher,
                new_obfuscator,
                &version_keys,
                &latest_blob,
                &latest_nonce,
            )?;

            // Update graph node entity_key property
            if let Some(node_id) = self.find_entity_node(&old_secret_node) {
                let mut props = HashMap::new();
                props.insert(
                    "entity_key".to_string(),
                    PropertyValue::String(new_secret_node),
                );
                self.graph.update_node(node_id, None, props).ok();
            }
        }
        Ok(old_to_new)
    }

    fn reencrypt_versions(
        &self,
        secret: &DecryptedSecret,
        new_cipher: &Cipher,
        new_obfuscator: &Obfuscator,
    ) -> Result<(Vec<String>, String, Vec<u8>)> {
        let mut keys = Vec::new();
        let mut latest_blob = String::new();
        let mut latest_nonce = Vec::new();

        for (plaintext, timestamp) in &secret.versions {
            let padded = obfuscation::pad_plaintext(plaintext.as_bytes())?;
            let (ciphertext, nonce) = new_cipher.encrypt(&padded)?;
            let blob_key = new_obfuscator.generate_storage_id(&secret.name, &nonce);

            let mut blob = TensorData::new();
            blob.set("_data", TensorValue::Scalar(ScalarValue::Bytes(ciphertext)));
            blob.set(
                "_nonce",
                TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
            );
            blob.set("_ts", TensorValue::Scalar(ScalarValue::Int(*timestamp)));
            self.store
                .put(&blob_key, blob)
                .map_err(|e| VaultError::StorageError(e.to_string()))?;

            latest_blob.clone_from(&blob_key);
            latest_nonce = nonce.to_vec();
            keys.push(blob_key);
        }
        Ok((keys, latest_blob, latest_nonce))
    }

    #[allow(clippy::too_many_arguments)]
    fn store_rotated_metadata(
        &self,
        secret: &DecryptedSecret,
        new_vault_key: &str,
        new_secret_node: &str,
        new_cipher: &Cipher,
        new_obfuscator: &Obfuscator,
        version_keys: &[String],
        latest_blob: &str,
        latest_nonce: &[u8],
    ) -> Result<()> {
        let (encrypted_name, key_nonce) = new_cipher.encrypt(secret.name.as_bytes())?;
        let new_creator_obf = new_obfuscator.encrypt_metadata(&secret.creator)?;
        let new_created_obf = new_obfuscator.encrypt_metadata(&secret.created)?;

        let mut tensor = TensorData::new();
        tensor.set(
            "_key_enc",
            TensorValue::Scalar(ScalarValue::Bytes(encrypted_name)),
        );
        tensor.set(
            "_key_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(key_nonce.to_vec())),
        );
        tensor.set(
            "_creator_obf",
            TensorValue::Scalar(ScalarValue::Bytes(new_creator_obf)),
        );
        tensor.set(
            "_created_obf",
            TensorValue::Scalar(ScalarValue::Bytes(new_created_obf)),
        );
        tensor.set("_blob", TensorValue::Pointer(latest_blob.to_string()));
        tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(latest_nonce.to_vec())),
        );
        tensor.set("_versions", TensorValue::Pointers(version_keys.to_vec()));
        tensor.set("_version_count", TensorValue::Scalar(ScalarValue::Int(1)));
        self.store
            .put(new_vault_key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        let mut node = TensorData::new();
        node.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("vault_secret".into())),
        );
        node.set(
            "_secret_key",
            TensorValue::Scalar(ScalarValue::String(secret.name.clone())),
        );
        self.store
            .put(new_secret_node, node)
            .map_err(|e| VaultError::StorageError(e.to_string()))
    }

    fn resign_edges(
        &self,
        records: &[EdgeRecord],
        new_signer: &EdgeSigner,
        old_to_new: &HashMap<String, String>,
    ) {
        let timestamp = Self::current_timestamp();
        for record in records {
            if !record.has_sig {
                continue;
            }

            let from_key = old_to_new.get(&record.from_key).unwrap_or(&record.from_key);
            let to_key = old_to_new.get(&record.to_key).unwrap_or(&record.to_key);

            let new_sig = new_signer.sign_edge(from_key, to_key, &record.edge_type, timestamp);

            let mut props = HashMap::new();
            props.insert("vault_sig".to_string(), PropertyValue::Bytes(new_sig));
            props.insert("vault_sig_ts".to_string(), PropertyValue::Int(timestamp));
            if let Some(cap) = record.capacity {
                props.insert("vault_capacity".to_string(), PropertyValue::Int(cap));
            }

            self.graph.update_edge(record.edge_id, props).ok();
        }
    }
}

struct DecryptedSecret {
    name: String,
    versions: Vec<(String, i64)>,
    creator: Vec<u8>,
    created: Vec<u8>,
}

struct EdgeRecord {
    edge_id: u64,
    from_key: String,
    to_key: String,
    edge_type: String,
    capacity: Option<i64>,
    has_sig: bool,
}
