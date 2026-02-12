// SPDX-License-Identifier: MIT OR Apache-2.0
//! Core vault operations: store, retrieve, rotate, delete, grant, revoke secrets.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use dashmap::DashMap;
use zeroize::Zeroizing;

use graph_engine::{Direction, GraphEngine, PropertyValue};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{
    access::AccessController,
    anomaly::{AnomalyMonitor, AnomalyThresholds},
    attenuation::AttenuationPolicy,
    audit::{AuditLog, AuditOperation},
    delegation::{DelegationManager, DelegationRecord},
    dependency::ImpactReport,
    dynamic::{DynamicSecretMetadata, SecretTemplate},
    encryption::Cipher,
    engine::EngineRegistry,
    key::{MasterKey, SALT_SIZE},
    obfuscation::{self, Obfuscator},
    pitr::VaultSnapshot,
    pki::{CertInfo, CertificateRequest, PkiEngine},
    policy::{PolicyManager, PolicyMatch, PolicyTemplate},
    quota::{QuotaManager, ResourceQuota, ResourceUsage},
    rate_limit::{Operation, RateLimiter},
    rotation::{PendingRotation, RotationGenerator, RotationPolicy, RotationPolicyManager},
    seal::SealGuard,
    shamir::KeyShare,
    signing::EdgeSigner,
    sync_target::SyncManager,
    ttl::GrantTTLTracker,
    wrapping::WrappingToken,
    Permission, Result, VaultConfig, VaultError, VaultEvent, VersionInfo,
};

/// Per-entry outcome from a batch set operation.
#[derive(Debug, Clone)]
pub struct BatchSetResult {
    pub succeeded: usize,
    pub failed: Vec<(String, VaultError)>,
}

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
    /// Per-secret locks to serialize metadata read-modify-write in set/rotate/delete.
    version_locks: DashMap<String, Arc<Mutex<()>>>,
    /// Derived audit integrity key for HMAC + AEAD on audit entries.
    audit_key: Zeroizing<[u8; 32]>,
    /// Transit cipher for encrypt-as-a-service operations.
    transit_cipher: Cipher,
    /// Anomaly detection monitor for agent behavior.
    anomaly_monitor: AnomalyMonitor,
    /// Delegation relationship manager.
    delegation_manager: DelegationManager,
    /// Seal state: when sealed, all crypto operations are blocked.
    seal_guard: SealGuard,
    /// Per-namespace resource quotas.
    quota_manager: QuotaManager,
    /// Policy-based access control templates.
    policy_manager: PolicyManager,
    /// Automated rotation policies.
    rotation_manager: RotationPolicyManager,
    /// Pluggable secret engine registry.
    engine_registry: EngineRegistry,
    /// Secret sync manager for pushing to external targets.
    sync_manager: SyncManager,
    /// Snapshot encryption cipher (derived from snapshot subkey).
    snapshot_cipher: Cipher,
    /// Event handler for monitoring and alerting.
    event_handler: Arc<dyn crate::VaultEventHandler>,
    /// Maximum secret value size in bytes.
    max_value_size: usize,
}

impl Vault {
    /// Storage key prefix for vault secrets (obfuscated).
    pub(crate) const PREFIX: &'static str = "_vk:";
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
        let transit_cipher = Cipher::from_raw_key(derived.transit_key());
        let audit_key = Zeroizing::new(derived.audit_key());

        let saved_config = config.clone();
        let rate_limiter = config.rate_limit.map(RateLimiter::new);
        let max_versions = config.max_versions.max(1); // At least 1 version
        let attenuation = config.attenuation;

        let ttl_tracker = GrantTTLTracker::load(&store).unwrap_or_default();

        let anomaly_monitor = config.anomaly_thresholds.as_ref().map_or_else(
            || AnomalyMonitor::new(AnomalyThresholds::default()),
            |t| AnomalyMonitor::load(&store, t.clone()),
        );
        let delegation_manager =
            DelegationManager::load(&store, config.max_delegation_depth.unwrap_or(3));

        let quota_manager = QuotaManager::load(&store);
        let policy_manager = PolicyManager::load(&store);
        let rotation_manager = RotationPolicyManager::load(&store);
        let engine_registry = EngineRegistry::new();
        let snapshot_cipher = Cipher::from_raw_key(derived.snapshot_key());
        let sync_manager = SyncManager::with_cipher(Cipher::from_raw_key(derived.sync_key()));
        sync_manager.load_subscriptions(&store);
        let seal_guard = SealGuard::from_store(&store);

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
            version_locks: DashMap::new(),
            audit_key,
            transit_cipher,
            anomaly_monitor,
            delegation_manager,
            seal_guard,
            quota_manager,
            policy_manager,
            rotation_manager,
            engine_registry,
            sync_manager,
            snapshot_cipher,
            event_handler: config
                .event_handler
                .clone()
                .unwrap_or_else(|| Arc::new(crate::NoopEventHandler)),
            max_value_size: config.max_value_size,
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
    pub(crate) fn find_entity_node(&self, entity_key: &str) -> Option<u64> {
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
        let Some(node_id) = self.find_entity_node(entity_key) else {
            return Vec::new();
        };
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

    fn extract_namespace(key: &str) -> &str {
        key.find('/').map_or("default", |idx| &key[..idx])
    }

    fn version_lock(&self, vault_key: &str) -> Arc<Mutex<()>> {
        self.version_locks
            .entry(vault_key.to_string())
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    }

    fn store_blob(
        &self,
        blob_key: &str,
        ciphertext: Vec<u8>,
        nonce: &[u8],
        timestamp: i64,
    ) -> Result<()> {
        let mut blob_tensor = TensorData::new();
        blob_tensor.set("_data", TensorValue::Scalar(ScalarValue::Bytes(ciphertext)));
        blob_tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
        );
        blob_tensor.set("_ts", TensorValue::Scalar(ScalarValue::Int(timestamp)));
        self.store
            .put(blob_key, blob_tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))
    }

    fn emit_cleanup_error(&self, context: &str, error: &dyn std::fmt::Display) {
        self.event_handler.on_event(&VaultEvent::CleanupError {
            context: context.into(),
            error: error.to_string(),
        });
    }

    fn emit_poison_recovery(&self, context: &str) {
        self.event_handler.on_event(&VaultEvent::PoisonRecovery {
            context: context.into(),
        });
    }

    pub(crate) fn secret_node_key(&self, secret_key: &str) -> String {
        let obfuscated = self.obfuscator.obfuscate_key(secret_key);
        format!("vault_secret:{obfuscated}")
    }

    pub(crate) fn audit_key(&self) -> &[u8; 32] {
        &self.audit_key
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
        self.set_inner(requester, key, value, None)
    }

    /// Store a secret with a time-to-live.
    ///
    /// After the TTL elapses, `get()` returns `SecretExpired`. The data stays
    /// in storage until overwritten or deleted. Use `clear_expiration()` to
    /// remove the deadline.
    pub fn set_with_ttl(
        &self,
        requester: &str,
        key: &str,
        value: &str,
        ttl: Duration,
    ) -> Result<()> {
        self.set_inner(requester, key, value, Some(ttl))
    }

    fn set_inner(
        &self,
        requester: &str,
        key: &str,
        value: &str,
        ttl: Option<Duration>,
    ) -> Result<()> {
        self.seal_guard.check_sealed()?;

        // Rate limit check
        self.check_rate_limit(requester, Operation::Set)?;

        // Enforce maximum value size
        if value.len() > self.max_value_size {
            return Err(VaultError::InvalidKey(format!(
                "value exceeds maximum size of {} bytes",
                self.max_value_size
            )));
        }

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

        // Quota check
        let namespace = Self::extract_namespace(key);
        self.quota_manager
            .check_quota(namespace, u64::from(!is_update), value.len() as u64)?;

        // Pad plaintext to hide length, then encrypt with AAD
        let padded = obfuscation::pad_plaintext(value.as_bytes())?;
        let (ciphertext, nonce) = self
            .cipher
            .encrypt_with_aad(&padded, vault_storage_key.as_bytes())?;
        let timestamp = Self::current_timestamp();

        // Store ciphertext in separate blob for pointer indirection
        let blob_storage_key = self.blob_key(key, &nonce);
        self.store_blob(&blob_storage_key, ciphertext, &nonce, timestamp)?;

        // Serialize metadata read-modify-write per secret
        let lock = self.version_lock(&vault_storage_key);
        let _guard = lock.lock().unwrap_or_else(|e| {
            self.emit_poison_recovery("set_inner: version metadata lock");
            e.into_inner()
        });

        // Get or create metadata tensor
        let mut tensor = if is_update {
            self.store
                .get(&vault_storage_key)
                .map_err(|_| VaultError::NotFound(key.to_string()))?
        } else {
            self.create_secret_metadata(key, requester, timestamp)?
        };

        // Update version info
        let mut versions = Self::get_version_blobs(&tensor);
        versions.push(blob_storage_key.clone());

        // Prune old versions if exceeding max
        while versions.len() > self.max_versions {
            if let Some(old_blob) = versions.first() {
                if let Err(e) = self.store.delete(old_blob) {
                    self.emit_cleanup_error("set_inner: old version prune", &e);
                }
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

        // Handle TTL: set or clear expiration
        if let Some(ttl) = ttl {
            let expires_at = Self::current_timestamp() + ttl.as_secs() as i64;
            tensor.set(
                "_expires_at",
                TensorValue::Scalar(ScalarValue::Int(expires_at)),
            );
        } else if is_update {
            // Regular set() on existing secret clears any expiration
            tensor.remove("_expires_at");
        }

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

        // Record quota usage
        let ns = Self::extract_namespace(key);
        if is_update {
            self.quota_manager.record_operation(&self.store, ns);
        } else {
            self.quota_manager
                .record_secret_added(&self.store, ns, value.len() as u64);
        }

        // Log audit
        self.log_operation(requester, key, &AuditOperation::Set);

        Ok(())
    }

    fn create_secret_metadata(
        &self,
        key: &str,
        requester: &str,
        timestamp: i64,
    ) -> Result<TensorData> {
        let (encrypted_key, key_nonce) =
            self.cipher.encrypt_with_aad(key.as_bytes(), b"key_name")?;

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
        Ok(t)
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
        self.seal_guard.check_sealed()?;

        // Opportunistic cleanup of expired grants
        self.cleanup_expired_grants();

        // Rate limit check
        self.check_rate_limit(requester, Operation::Get)?;

        self.check_access(requester, key)?;

        let tensor = self
            .store
            .get(&self.vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Check expiry
        if let Some(TensorValue::Scalar(ScalarValue::Int(expires_at))) = tensor.get("_expires_at") {
            if Self::current_timestamp() >= *expires_at {
                return Err(VaultError::SecretExpired(key.to_string()));
            }
        }

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

        // Decrypt and unpad (AAD bound to storage key)
        let vault_storage_key = self.vault_key(key);
        let padded =
            self.cipher
                .decrypt_with_aad(&ciphertext, &nonce, vault_storage_key.as_bytes())?;
        let plaintext = obfuscation::unpad_plaintext(&padded)?;

        // Track legacy ciphertext (no version byte tag)
        if ciphertext.first() != Some(&0x02) {
            self.log_operation(requester, key, &AuditOperation::LegacyDecrypt);
            self.event_handler.on_event(&VaultEvent::LegacyDecrypt {
                entity: requester.to_string(),
                key: key.to_string(),
            });
        }

        // Log audit
        self.log_operation(requester, key, &AuditOperation::Get);

        String::from_utf8(plaintext)
            .map_err(|e| VaultError::CryptoError(format!("Invalid UTF-8: {e}")))
    }

    /// Retrieve a specific version of a secret.
    ///
    /// Version numbers are 1-based. Version 1 is the oldest kept version.
    pub fn get_version(&self, requester: &str, key: &str, version: u32) -> Result<String> {
        self.seal_guard.check_sealed()?;
        self.check_access(requester, key)?;

        let tensor = self
            .store
            .get(&self.vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Check expiry
        if let Some(TensorValue::Scalar(ScalarValue::Int(expires_at))) = tensor.get("_expires_at") {
            if Self::current_timestamp() >= *expires_at {
                return Err(VaultError::SecretExpired(key.to_string()));
            }
        }

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

        let vault_storage_key = self.vault_key(key);
        let padded =
            self.cipher
                .decrypt_with_aad(&ciphertext, &nonce, vault_storage_key.as_bytes())?;
        let plaintext = obfuscation::unpad_plaintext(&padded)?;

        // Track legacy ciphertext (no version byte tag)
        if ciphertext.first() != Some(&0x02) {
            self.log_operation(requester, key, &AuditOperation::LegacyDecrypt);
            self.event_handler.on_event(&VaultEvent::LegacyDecrypt {
                entity: requester.to_string(),
                key: key.to_string(),
            });
        }

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
        self.seal_guard.check_sealed()?;
        let value = self.get_version(requester, key, version)?;
        self.set(requester, key, &value)
    }

    /// Grant access to a secret with Admin permission (backward compatible).
    pub fn grant(&self, requester: &str, entity: &str, key: &str) -> Result<()> {
        self.seal_guard.check_sealed()?;
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
        self.seal_guard.check_sealed()?;

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
    ///
    /// Note: the graph edge, TTL tracker update, and persist are separate
    /// operations. If the process crashes between them, `cleanup_expired_grants()`
    /// at next startup will reconcile.
    pub fn grant_with_ttl(
        &self,
        requester: &str,
        entity: &str,
        key: &str,
        level: Permission,
        ttl: Duration,
    ) -> Result<()> {
        self.seal_guard.check_sealed()?;
        // First, create the grant normally
        self.grant_with_permission(requester, entity, key, level)?;

        // Then register with TTL tracker and persist
        self.ttl_tracker.add(entity, key, ttl);
        self.ttl_tracker.persist(&self.store).ok();

        Ok(())
    }

    /// Revoke access to a secret (requires Admin permission).
    ///
    /// The graph edge deletion, TTL tracker cleanup, and persist are separate.
    /// If the process crashes mid-revoke, `cleanup_expired_grants()` at next
    /// startup will reconcile stale grants.
    pub fn revoke(&self, requester: &str, entity: &str, key: &str) -> Result<()> {
        self.seal_guard.check_sealed()?;
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
        self.seal_guard.check_sealed()?;
        self.check_access_with_permission(requester, key, Permission::Admin)?;

        let vault_storage_key = self.vault_key(key);

        // Serialize with concurrent set/rotate on the same secret
        let lock = self.version_lock(&vault_storage_key);
        let _guard = lock.lock().unwrap_or_else(|e| {
            self.emit_poison_recovery("delete: version metadata lock");
            e.into_inner()
        });

        // Delete all version blobs
        if let Ok(tensor) = self.store.get(&vault_storage_key) {
            for blob_key in Self::get_version_blobs(&tensor) {
                if let Err(e) = self.store.delete(&blob_key) {
                    self.emit_cleanup_error("delete: version blob cleanup", &e);
                }
            }
        }

        self.store
            .delete(&vault_storage_key)
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Clean up the secret node from TensorStore
        let secret_node = self.secret_node_key(key);
        if let Err(e) = self.store.delete(&secret_node) {
            self.emit_cleanup_error("delete: secret node cleanup", &e);
        }

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
                    if let Err(e) = self.delete_graph_edge(edge.id) {
                        self.emit_cleanup_error("delete: graph edge cleanup", &e);
                    }
                }
            }
            if let Err(e) = self.graph.delete_node(node_id) {
                self.emit_cleanup_error("delete: graph node cleanup", &e);
            }
            if let Err(e) = self.ttl_tracker.persist(&self.store) {
                self.emit_cleanup_error("delete: TTL persist", &e);
            }
        }

        // Log audit
        self.log_operation(requester, key, &AuditOperation::Delete);

        // Record quota usage
        let ns = Self::extract_namespace(key);
        self.quota_manager.record_secret_removed(&self.store, ns, 0);

        Ok(())
    }

    /// Rotate a secret value (requires Write permission).
    ///
    /// Creates a new version. Previous versions are kept up to `max_versions`.
    pub fn rotate(&self, requester: &str, key: &str, new_value: &str) -> Result<()> {
        self.seal_guard.check_sealed()?;
        self.check_access_with_permission(requester, key, Permission::Write)?;

        let vault_storage_key = self.vault_key(key);

        if !self.store.exists(&vault_storage_key) {
            return Err(VaultError::NotFound(key.to_string()));
        }

        // Pad and encrypt new value with AAD
        let padded = obfuscation::pad_plaintext(new_value.as_bytes())?;
        let (ciphertext, nonce) = self
            .cipher
            .encrypt_with_aad(&padded, vault_storage_key.as_bytes())?;
        let timestamp = Self::current_timestamp();

        // Store new ciphertext blob with version info
        let new_blob_key = self.blob_key(key, &nonce);
        self.store_blob(&new_blob_key, ciphertext, &nonce, timestamp)?;

        // Serialize metadata read-modify-write per secret
        let lock = self.version_lock(&vault_storage_key);
        let _guard = lock.lock().unwrap_or_else(|e| {
            self.emit_poison_recovery("rotate: version metadata lock");
            e.into_inner()
        });

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
                if let Err(e) = self.store.delete(old_blob) {
                    self.emit_cleanup_error("rotate: old version prune", &e);
                }
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
        self.quota_manager
            .record_operation(&self.store, Self::extract_namespace(key));

        Ok(())
    }

    /// List secret keys matching a pattern.
    ///
    /// For exact-match patterns (no wildcards), uses O(1) direct key lookup
    /// instead of scanning all secrets.
    pub fn list(&self, requester: &str, pattern: &str) -> Result<Vec<String>> {
        self.seal_guard.check_sealed()?;

        // Opportunistic cleanup of expired grants
        self.cleanup_expired_grants();

        // Rate limit check
        self.check_rate_limit(requester, Operation::List)?;

        let accessible = if !pattern.is_empty() && !pattern.contains('*') {
            // Exact match: O(1) lookup via obfuscated key
            let vault_storage_key = self.vault_key(pattern);
            if self.store.exists(&vault_storage_key) && self.has_access(requester, pattern) {
                // Check expiry for exact match
                let expired = self
                    .store
                    .get(&vault_storage_key)
                    .ok()
                    .and_then(|t| Self::is_expired(&t))
                    .unwrap_or(false);
                if expired {
                    vec![]
                } else {
                    vec![pattern.to_string()]
                }
            } else {
                vec![]
            }
        } else {
            // Wildcard/prefix/empty: full scan
            let all_keys = self.store.scan(Self::PREFIX);
            let mut result = Vec::new();
            for vault_key in all_keys {
                if let Ok(tensor) = self.store.get(&vault_key) {
                    // Skip expired secrets
                    if Self::is_expired(&tensor).unwrap_or(false) {
                        continue;
                    }
                    let original_key = self.decrypt_key_name(&tensor);
                    if let Some(key) = original_key {
                        if Self::matches_pattern(&key, pattern) && self.has_access(requester, &key)
                        {
                            result.push(key);
                        }
                    }
                }
            }
            result
        };

        // Log audit (using pattern as key)
        self.log_operation(requester, pattern, &AuditOperation::List);

        Ok(accessible)
    }

    pub(crate) fn decrypt_key_name(&self, tensor: &TensorData) -> Option<String> {
        let encrypted_key = match tensor.get("_key_enc") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return None,
        };
        let key_nonce = match tensor.get("_key_nonce") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return None,
        };

        self.cipher
            .decrypt_with_aad(&encrypted_key, &key_nonce, b"key_name")
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

    // ========== Graph Intelligence API ==========

    pub fn explain_access(
        &self,
        entity: &str,
        secret: &str,
    ) -> crate::graph_intel::AccessExplanation {
        crate::graph_intel::explain_access(self, entity, secret)
    }

    pub fn blast_radius(&self, entity: &str) -> crate::graph_intel::BlastRadius {
        crate::graph_intel::blast_radius(self, entity)
    }

    pub fn simulate_grant(
        &self,
        entity: &str,
        secret: &str,
        permission: Permission,
    ) -> crate::graph_intel::SimulationResult {
        crate::graph_intel::simulate_grant(self, entity, secret, permission)
    }

    pub fn security_audit(&self) -> crate::graph_intel::SecurityAuditReport {
        crate::graph_intel::security_audit(self)
    }

    pub fn find_critical_entities(&self) -> Vec<crate::graph_intel::CriticalEntity> {
        crate::graph_intel::find_critical_entities(self)
    }

    pub fn privilege_analysis(&self) -> crate::graph_intel::PrivilegeAnalysisReport {
        crate::graph_intel::privilege_analysis(self)
    }

    pub fn delegation_anomaly_scores(&self) -> Vec<crate::graph_intel::DelegationAnomalyScore> {
        crate::graph_intel::delegation_anomaly_scores(self)
    }

    pub fn infer_roles(&self) -> crate::graph_intel::RoleInferenceResult {
        crate::graph_intel::infer_roles(self)
    }

    pub fn trust_transitivity(&self) -> crate::graph_intel::TrustTransitivityReport {
        crate::graph_intel::trust_transitivity(self)
    }

    pub fn risk_propagation(&self) -> crate::graph_intel::RiskPropagationReport {
        crate::graph_intel::risk_propagation(self)
    }

    // ========== Behavior Embeddings & Anomaly Detection ==========

    pub fn compute_behavior_embeddings(
        &self,
        config: crate::graph_intel::BehaviorEmbeddingConfig,
    ) -> Vec<crate::graph_intel::NodeEmbedding> {
        crate::graph_intel::compute_behavior_embeddings(self, config)
    }

    pub fn detect_geometric_anomalies(
        &self,
        k: usize,
        threshold_multiplier: f64,
    ) -> crate::graph_intel::GeometricAnomalyReport {
        let embeddings = self
            .compute_behavior_embeddings(crate::graph_intel::BehaviorEmbeddingConfig::default());
        crate::graph_intel::detect_geometric_anomalies(&embeddings, k, threshold_multiplier)
    }

    pub fn cluster_entities(&self) -> crate::graph_intel::ClusteringResult {
        crate::graph_intel::cluster_entities(self)
    }

    // ========== Developer Experience API ==========

    /// List secrets with pagination support.
    pub fn list_paginated(
        &self,
        requester: &str,
        pattern: &str,
        offset: usize,
        limit: usize,
    ) -> Result<crate::PagedSecrets> {
        let all = self.list(requester, pattern)?;
        let total = all.len();

        if limit == 0 {
            return Ok(crate::PagedSecrets {
                secrets: all,
                offset: 0,
                limit: 0,
                total,
                has_more: false,
            });
        }

        let start = offset.min(total);
        let end = (start + limit).min(total);
        let secrets = all[start..end].to_vec();
        let has_more = end < total;

        Ok(crate::PagedSecrets {
            secrets,
            offset,
            limit,
            total,
            has_more,
        })
    }

    /// List secrets with metadata summaries.
    #[allow(clippy::cast_precision_loss)]
    pub fn list_with_metadata(
        &self,
        requester: &str,
        pattern: &str,
    ) -> Result<Vec<crate::SecretSummary>> {
        let keys = self.list(requester, pattern)?;
        let mut summaries = Vec::with_capacity(keys.len());

        for key in keys {
            let vault_key = self.vault_key(&key);
            let version_count = if let Ok(tensor) = self.store.get(&vault_key) {
                Self::get_version_blobs(&tensor).len() as u32
            } else {
                0
            };

            // Get created_at from first version's timestamp
            let created_at = self
                .list_versions(requester, &key)
                .ok()
                .and_then(|versions| versions.first().map(|v| v.created_at))
                .unwrap_or(0);

            // Get last_accessed from audit log
            let last_accessed = self.audit_log(&key).ok().and_then(|entries| {
                entries
                    .iter()
                    .filter(|e| matches!(e.operation, AuditOperation::Get))
                    .max_by_key(|e| e.timestamp)
                    .map(|e| e.timestamp)
            });

            // Count entities with access to this secret
            let entity_count = self.count_entity_access(&key);

            summaries.push(crate::SecretSummary {
                key,
                version_count,
                created_at,
                last_accessed,
                entity_count,
            });
        }

        Ok(summaries)
    }

    /// Count entities that have access to a given secret.
    fn count_entity_access(&self, key: &str) -> usize {
        let secret_node_key = self.secret_node_key(key);
        if let Some(node_id) = self.find_entity_node(&secret_node_key) {
            self.graph
                .edges_of(node_id, graph_engine::Direction::Incoming)
                .unwrap_or_default()
                .iter()
                .filter(|e| e.edge_type.starts_with("VAULT_ACCESS"))
                .count()
        } else {
            0
        }
    }

    /// Compare two versions of a secret side-by-side.
    pub fn diff_versions(
        &self,
        requester: &str,
        key: &str,
        version_a: u32,
        version_b: u32,
    ) -> Result<crate::VersionDiff> {
        let value_a = self.get_version(requester, key, version_a)?;
        let value_b = self.get_version(requester, key, version_b)?;

        let versions = self.list_versions(requester, key)?;
        let timestamp_a = versions
            .get(version_a.saturating_sub(1) as usize)
            .map_or(0, |v| v.created_at);
        let timestamp_b = versions
            .get(version_b.saturating_sub(1) as usize)
            .map_or(0, |v| v.created_at);

        self.log_operation(
            requester,
            key,
            &AuditOperation::DiffVersions {
                version_a,
                version_b,
            },
        );

        Ok(crate::VersionDiff {
            key: key.to_string(),
            version_a,
            version_b,
            value_a,
            value_b,
            timestamp_a,
            timestamp_b,
        })
    }

    /// Get a chronological changelog for a secret.
    pub fn changelog(&self, requester: &str, key: &str) -> Result<Vec<crate::ChangelogEntry>> {
        self.check_access(requester, key)?;

        let mut entries = Vec::new();

        // Version history entries
        if let Ok(versions) = self.list_versions(requester, key) {
            for v in &versions {
                entries.push(crate::ChangelogEntry {
                    version: Some(v.version),
                    operation: if v.version == 1 {
                        "set".to_string()
                    } else {
                        "rotate".to_string()
                    },
                    entity: String::new(),
                    timestamp: v.created_at,
                });
            }
        }

        // Audit log entries
        if let Ok(audit_entries) = self.audit_log(key) {
            for entry in &audit_entries {
                entries.push(crate::ChangelogEntry {
                    version: None,
                    operation: format!("{:?}", entry.operation)
                        .split(['(', '{', ' '])
                        .next()
                        .unwrap_or("unknown")
                        .to_lowercase(),
                    entity: entry.entity.clone(),
                    timestamp: entry.timestamp,
                });
            }
        }

        entries.sort_by_key(|e| e.timestamp);
        Ok(entries)
    }

    // ========== Similarity & Duplication ==========

    /// Compute metadata features for a secret.
    #[allow(clippy::cast_precision_loss)]
    fn compute_secret_features(
        &self,
        key: &str,
        requester: &str,
    ) -> Option<crate::similarity::SecretFeatures> {
        let vault_key = self.vault_key(key);
        let tensor = self.store.get(&vault_key).ok()?;
        let versions = Self::get_version_blobs(&tensor);

        // Get first version timestamp for age
        let created_at = versions
            .first()
            .and_then(|blob_key| {
                self.store
                    .get(blob_key)
                    .ok()
                    .and_then(|blob| match blob.get("_ts") {
                        Some(TensorValue::Scalar(ScalarValue::Int(ts))) => Some(*ts),
                        _ => None,
                    })
            })
            .unwrap_or(0);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let age_days = ((now - created_at) as f32) / 86_400_000.0;

        // Access frequency from audit log
        let access_count = self
            .audit_log(key)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|e| matches!(e.operation, AuditOperation::Get))
                    .count()
            })
            .unwrap_or(0);

        // Days since last rotation
        let last_rotation = versions
            .last()
            .and_then(|blob_key| {
                self.store
                    .get(blob_key)
                    .ok()
                    .and_then(|blob| match blob.get("_ts") {
                        Some(TensorValue::Scalar(ScalarValue::Int(ts))) => Some(*ts),
                        _ => None,
                    })
            })
            .unwrap_or(created_at);
        let rotation_days = ((now - last_rotation) as f32) / 86_400_000.0;

        let entity_count = self.count_entity_access(key);

        // Permission entropy
        let _ = requester; // Used for access checks upstream
        let perm_entropy = (entity_count as f32).ln_1p();

        Some(crate::similarity::SecretFeatures {
            key: key.to_string(),
            creation_age_days: age_days.max(0.0),
            version_count: versions.len() as f32,
            access_frequency: access_count as f32,
            days_since_rotation: rotation_days.max(0.0),
            entity_count: entity_count as f32,
            permission_entropy: perm_entropy,
        })
    }

    /// Find secrets with similar operational metadata.
    pub fn find_similar(
        &self,
        requester: &str,
        key: &str,
        k: usize,
    ) -> Result<Vec<crate::similarity::SimilarSecret>> {
        self.check_access(requester, key)?;

        let keys = self.list(requester, "*")?;
        let index = crate::similarity::SimilarityIndex::new();

        for secret_key in &keys {
            if let Some(features) = self.compute_secret_features(secret_key, requester) {
                index.insert(secret_key, features.to_embedding());
            }
        }

        let query_features = self
            .compute_secret_features(key, requester)
            .ok_or_else(|| VaultError::NotFound(key.to_string()))?;

        let results = index.search(&query_features.to_embedding(), k + 1);

        // Filter out the query key itself
        let filtered: Vec<_> = results
            .into_iter()
            .filter(|r| r.key != key)
            .take(k)
            .collect();

        self.log_operation(requester, key, &AuditOperation::FindSimilar { k });

        Ok(filtered)
    }

    /// Check for potential duplicate secrets by high metadata similarity.
    pub fn check_duplication(
        &self,
        requester: &str,
        threshold: f32,
    ) -> Result<Vec<(String, String, f32)>> {
        let keys = self.list(requester, "*")?;
        let index = crate::similarity::SimilarityIndex::new();

        for secret_key in &keys {
            if let Some(features) = self.compute_secret_features(secret_key, requester) {
                index.insert(secret_key, features.to_embedding());
            }
        }

        let mut duplicates = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for secret_key in &keys {
            if let Some(features) = self.compute_secret_features(secret_key, requester) {
                let results = index.search(&features.to_embedding(), 2);
                for similar in results {
                    if similar.key != *secret_key && similar.similarity >= threshold {
                        let pair_key = if *secret_key < similar.key {
                            format!("{}:{}", secret_key, similar.key)
                        } else {
                            format!("{}:{}", similar.key, secret_key)
                        };
                        if seen.insert(pair_key) {
                            duplicates.push((
                                secret_key.clone(),
                                similar.key.clone(),
                                similar.similarity,
                            ));
                        }
                    }
                }
            }
        }

        Ok(duplicates)
    }

    // ========== Template Management ==========

    /// Save a secret generation template.
    pub fn save_template(
        &self,
        requester: &str,
        name: &str,
        template: SecretTemplate,
    ) -> Result<()> {
        self.seal_guard.check_sealed()?;
        self.check_rate_limit(requester, Operation::List)?;

        // Load templates from store to get current state
        let mgr = crate::template_store::TemplateManager::load(&self.store);
        mgr.save(&self.store, name, template, requester)?;

        self.log_operation(
            requester,
            name,
            &AuditOperation::SaveTemplate {
                template_name: name.to_string(),
            },
        );
        Ok(())
    }

    /// Get a saved template by name.
    pub fn get_template(&self, name: &str) -> Option<crate::template_store::StoredTemplate> {
        let mgr = crate::template_store::TemplateManager::load(&self.store);
        mgr.get(name)
    }

    /// List all saved template names.
    pub fn list_templates(&self) -> Vec<String> {
        let mgr = crate::template_store::TemplateManager::load(&self.store);
        mgr.list()
    }

    /// Delete a saved template.
    pub fn delete_template(&self, requester: &str, name: &str) -> Result<()> {
        self.seal_guard.check_sealed()?;

        let mgr = crate::template_store::TemplateManager::load(&self.store);
        mgr.delete(&self.store, name)?;

        self.log_operation(
            requester,
            name,
            &AuditOperation::DeleteTemplate {
                template_name: name.to_string(),
            },
        );
        Ok(())
    }

    // ========== Access Topology ==========

    /// Build an access topology matrix from the current vault state.
    pub fn build_access_topology(
        &self,
        config: crate::topology::TopologyConfig,
    ) -> Result<crate::topology::AccessTopology> {
        crate::topology::AccessTopology::from_vault(self, config)
    }

    /// Analyze policy redundancy and identify mergeable policy groups.
    pub fn analyze_policy_redundancy(&self) -> Result<crate::topology::PolicyRedundancyReport> {
        crate::topology::analyze_policy_redundancy(self)
    }

    fn log_operation(&self, requester: &str, key: &str, operation: &AuditOperation) {
        self.log_operation_with_context(requester, key, operation, None);
    }

    pub(crate) fn log_operation_with_context(
        &self,
        requester: &str,
        key: &str,
        operation: &AuditOperation,
        context: Option<&crate::audit::AuditContext>,
    ) {
        let obfuscated_key = self.obfuscator.obfuscate_key(key);
        let audit_log = AuditLog::new(&self.store, Some(*self.audit_key));
        audit_log.record_with_context(requester, &obfuscated_key, operation, context);

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let events = self
            .anomaly_monitor
            .check(requester, &obfuscated_key, operation, now_ms);
        for event in events {
            self.event_handler.on_event(&VaultEvent::Anomaly(event));
        }
    }

    /// Query audit entries for a specific secret.
    pub fn audit_log(&self, key: &str) -> Result<Vec<crate::audit::AuditEntry>> {
        self.seal_guard.check_sealed()?;
        let obfuscated_key = self.obfuscator.obfuscate_key(key);
        let audit = AuditLog::new(&self.store, Some(*self.audit_key));
        Ok(audit.by_secret(&obfuscated_key))
    }

    /// Query audit entries by entity (who performed operations).
    pub fn audit_by_entity(&self, entity: &str) -> Result<Vec<crate::audit::AuditEntry>> {
        self.seal_guard.check_sealed()?;
        let audit = AuditLog::new(&self.store, Some(*self.audit_key));
        Ok(audit.by_entity(entity))
    }

    /// Query audit entries since a timestamp (unix milliseconds).
    pub fn audit_since(&self, since_millis: i64) -> Result<Vec<crate::audit::AuditEntry>> {
        self.seal_guard.check_sealed()?;
        let audit = AuditLog::new(&self.store, Some(*self.audit_key));
        Ok(audit.since(since_millis))
    }

    /// Get recent audit entries.
    pub fn audit_recent(&self, limit: usize) -> Result<Vec<crate::audit::AuditEntry>> {
        self.seal_guard.check_sealed()?;
        let audit = AuditLog::new(&self.store, Some(*self.audit_key));
        Ok(audit.recent(limit))
    }

    /// Delegate access from parent to child agent.
    ///
    /// The parent must have at least the requested permission on each secret.
    /// The child receives `min(parent permission, requested permission)` per secret.
    pub fn delegate(
        &self,
        parent: &str,
        child: &str,
        secrets: &[&str],
        permission: Permission,
        ttl: Option<Duration>,
    ) -> Result<DelegationRecord> {
        // Verify parent has sufficient permission on every secret
        for &secret in secrets {
            let parent_perm = self.get_permission(parent, secret).ok_or_else(|| {
                VaultError::AccessDenied(format!("{parent} has no access to '{secret}'"))
            })?;
            if !parent_perm.allows(permission) {
                return Err(VaultError::InsufficientPermission(format!(
                    "{parent} has {parent_perm} but tried to delegate {permission} on '{secret}'"
                )));
            }
        }

        // Compute effective ceiling = min across all secrets of min(parent perm, requested)
        let effective = secrets.iter().fold(permission, |acc, &secret| {
            let parent_perm = self
                .get_permission(parent, secret)
                .unwrap_or(Permission::Read);
            let effective_for_secret = if parent_perm.to_level() < acc.to_level() {
                parent_perm
            } else {
                acc
            };
            if effective_for_secret.to_level() < acc.to_level() {
                effective_for_secret
            } else {
                acc
            }
        });

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        let ttl_ms = ttl.map(|d| d.as_millis() as i64);
        let secret_names: Vec<String> = secrets.iter().map(|s| (*s).to_string()).collect();

        let record = self.delegation_manager.register(
            parent,
            child,
            secret_names,
            effective,
            ttl_ms,
            now_ms,
        )?;

        // Create graph edges for each secret
        for &secret in secrets {
            let secret_node = self.secret_node_key(secret);
            let edge_type = format!("{}{}", Self::ACCESS_EDGE, effective.edge_suffix());
            self.add_entity_graph_edge(child, &secret_node, &edge_type)?;
        }

        // Register TTLs if applicable
        if let Some(duration) = ttl {
            for &secret in secrets {
                self.ttl_tracker.add(child, secret, duration);
            }
            self.ttl_tracker.persist(&self.store).ok();
        }

        self.delegation_manager.persist(&self.store);

        // Audit
        for &secret in secrets {
            self.log_operation(
                parent,
                secret,
                &AuditOperation::Grant {
                    to: child.to_string(),
                    permission: format!("{effective:?}").to_lowercase(),
                },
            );
        }

        Ok(record)
    }

    /// Revoke a delegation from parent to child, removing graph edges.
    pub fn revoke_delegation(&self, parent: &str, child: &str) -> Result<Vec<String>> {
        let record = self
            .delegation_manager
            .revoke(parent, child)
            .ok_or_else(|| {
                VaultError::NotFound(format!("no delegation from {parent} to {child}"))
            })?;

        let mut revoked_secrets = Vec::new();
        for secret in &record.secrets {
            let secret_node = self.secret_node_key(secret);
            for (edge_id, to, edge_type) in self.get_entity_outgoing_edges(child) {
                if to == secret_node && edge_type.starts_with(Self::ACCESS_EDGE) {
                    self.delete_graph_edge(edge_id)?;
                }
            }
            self.ttl_tracker.remove(child, secret);
            revoked_secrets.push(secret.clone());
        }

        self.delegation_manager.persist(&self.store);
        self.ttl_tracker.persist(&self.store).ok();

        for secret in &revoked_secrets {
            self.log_operation(
                parent,
                secret,
                &AuditOperation::Revoke {
                    from: child.to_string(),
                },
            );
        }

        Ok(revoked_secrets)
    }

    /// Revoke a delegation and all transitive sub-delegations.
    pub fn revoke_delegation_cascading(
        &self,
        parent: &str,
        child: &str,
    ) -> Result<Vec<DelegationRecord>> {
        let revoked = self.delegation_manager.revoke_cascading(parent, child);

        for record in &revoked {
            for secret in &record.secrets {
                let secret_node = self.secret_node_key(secret);
                for (edge_id, to, edge_type) in self.get_entity_outgoing_edges(&record.child) {
                    if to == secret_node && edge_type.starts_with(Self::ACCESS_EDGE) {
                        self.delete_graph_edge(edge_id).ok();
                    }
                }
                self.ttl_tracker.remove(&record.child, secret);
            }
        }

        self.delegation_manager.persist(&self.store);
        self.ttl_tracker.persist(&self.store).ok();
        Ok(revoked)
    }

    /// Access the anomaly monitor.
    pub fn anomaly_monitor(&self) -> &AnomalyMonitor {
        &self.anomaly_monitor
    }

    /// Access the delegation manager.
    pub fn delegation_manager(&self) -> &DelegationManager {
        &self.delegation_manager
    }

    /// Persist anomaly profiles to storage.
    pub fn persist_anomaly_profiles(&self) {
        self.anomaly_monitor.persist(&self.store);
    }

    /// Remove the expiration deadline from a secret (requires Admin permission).
    pub fn clear_expiration(&self, requester: &str, key: &str) -> Result<()> {
        self.check_access_with_permission(requester, key, Permission::Admin)?;

        let vault_storage_key = self.vault_key(key);
        let lock = self.version_lock(&vault_storage_key);
        let _guard = lock.lock().unwrap_or_else(|e| {
            self.emit_poison_recovery("clear_expiration: version metadata lock");
            e.into_inner()
        });

        let mut tensor = self
            .store
            .get(&vault_storage_key)
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        tensor.remove("_expires_at");

        self.store
            .put(vault_storage_key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))
    }

    /// Get the expiration timestamp of a secret, if set.
    ///
    /// Returns `None` if the secret has no expiration.
    pub fn get_expiration(&self, requester: &str, key: &str) -> Result<Option<i64>> {
        self.check_access(requester, key)?;

        let tensor = self
            .store
            .get(&self.vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        Ok(match tensor.get("_expires_at") {
            Some(TensorValue::Scalar(ScalarValue::Int(ts))) => Some(*ts),
            _ => None,
        })
    }

    /// Encrypt arbitrary data using the vault's transit key.
    ///
    /// The caller must have access to the named secret. The transit cipher is
    /// a separate HKDF subkey, so compromising transit ciphertext does not
    /// compromise stored secrets.
    pub fn encrypt_for(&self, requester: &str, key: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
        self.seal_guard.check_sealed()?;
        self.check_access(requester, key)?;

        let (ciphertext, nonce) = self
            .transit_cipher
            .encrypt_with_aad(plaintext, key.as_bytes())?;
        let mut sealed = Vec::with_capacity(crate::NONCE_SIZE + ciphertext.len());
        sealed.extend_from_slice(&nonce);
        sealed.extend_from_slice(&ciphertext);

        self.log_operation(requester, key, &AuditOperation::TransitEncrypt);
        Ok(sealed)
    }

    /// Decrypt data that was encrypted with `encrypt_for()`.
    pub fn decrypt_as(&self, requester: &str, key: &str, sealed: &[u8]) -> Result<Vec<u8>> {
        self.seal_guard.check_sealed()?;
        self.check_access(requester, key)?;

        if sealed.len() <= crate::NONCE_SIZE {
            return Err(VaultError::CryptoError("sealed data too short".to_string()));
        }

        let nonce = &sealed[..crate::NONCE_SIZE];
        let ciphertext = &sealed[crate::NONCE_SIZE..];

        let plaintext = self
            .transit_cipher
            .decrypt_with_aad(ciphertext, nonce, key.as_bytes())?;

        self.log_operation(requester, key, &AuditOperation::TransitDecrypt);
        Ok(plaintext)
    }

    /// Bypass graph-based access control for a one-time emergency read.
    ///
    /// Creates a temporary Read grant with the given duration, heavily audited
    /// with the justification string. Rate-limited to prevent abuse.
    pub fn emergency_access(
        &self,
        requester: &str,
        key: &str,
        justification: &str,
        duration: Duration,
    ) -> Result<String> {
        self.seal_guard.check_sealed()?;

        // Emergency access requires rate limiting to be enabled
        if self.rate_limiter.is_none() {
            return Err(VaultError::AccessDenied(
                "emergency access requires rate limiting to be enabled".to_string(),
            ));
        }

        // Rate limit check (very restrictive)
        self.check_rate_limit(requester, Operation::BreakGlass)?;

        // Verify secret exists
        let vault_storage_key = self.vault_key(key);
        let tensor = self
            .store
            .get(&vault_storage_key)
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Check expiry
        if let Some(TensorValue::Scalar(ScalarValue::Int(expires_at))) = tensor.get("_expires_at") {
            if Self::current_timestamp() >= *expires_at {
                return Err(VaultError::SecretExpired(key.to_string()));
            }
        }

        // Decrypt the current version
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

        let padded =
            self.cipher
                .decrypt_with_aad(&ciphertext, &nonce, vault_storage_key.as_bytes())?;
        let plaintext = obfuscation::unpad_plaintext(&padded)?;

        // Create temporary Read grant
        let secret_node = self.secret_node_key(key);
        let edge_type = format!("{}{}", Self::ACCESS_EDGE, Permission::Read.edge_suffix());
        self.add_entity_graph_edge(requester, &secret_node, &edge_type)?;
        self.ttl_tracker.add(requester, key, duration);
        self.ttl_tracker.persist(&self.store).ok();

        // Audit with justification
        self.log_operation(
            requester,
            key,
            &AuditOperation::BreakGlass {
                justification: justification.to_string(),
                duration_secs: duration.as_secs(),
            },
        );

        String::from_utf8(plaintext)
            .map_err(|e| VaultError::CryptoError(format!("Invalid UTF-8: {e}")))
    }

    /// Retrieve multiple secrets in a single operation.
    ///
    /// Each key is checked independently; failures are captured per-key.
    pub fn batch_get(
        &self,
        requester: &str,
        keys: &[&str],
    ) -> Result<Vec<(String, Result<String>)>> {
        self.seal_guard.check_sealed()?;

        // Single rate limit check for the batch
        self.check_rate_limit(requester, Operation::Get)?;

        self.cleanup_expired_grants();

        let mut results = Vec::with_capacity(keys.len());
        for &key in keys {
            let result = self.batch_get_single(requester, key);
            results.push((key.to_string(), result));
        }

        self.log_operation(
            requester,
            "*",
            &AuditOperation::BatchGet { count: keys.len() },
        );

        Ok(results)
    }

    fn batch_get_single(&self, requester: &str, key: &str) -> Result<String> {
        self.check_access(requester, key)?;

        let tensor = self
            .store
            .get(&self.vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Check expiry
        if let Some(TensorValue::Scalar(ScalarValue::Int(expires_at))) = tensor.get("_expires_at") {
            if Self::current_timestamp() >= *expires_at {
                return Err(VaultError::SecretExpired(key.to_string()));
            }
        }

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

        let vault_storage_key = self.vault_key(key);
        let padded =
            self.cipher
                .decrypt_with_aad(&ciphertext, &nonce, vault_storage_key.as_bytes())?;
        let plaintext = obfuscation::unpad_plaintext(&padded)?;

        String::from_utf8(plaintext)
            .map_err(|e| VaultError::CryptoError(format!("Invalid UTF-8: {e}")))
    }

    /// Store multiple secrets. Entries are written individually; partial failure
    /// is possible.
    ///
    /// Locks are acquired in sorted key order to prevent deadlocks. Each entry
    /// follows the same permission rules as `set()`.
    pub fn batch_set(&self, requester: &str, entries: &[(&str, &str)]) -> Result<()> {
        self.seal_guard.check_sealed()?;

        if entries.is_empty() {
            return Ok(());
        }

        // Single rate limit check for the batch
        self.check_rate_limit(requester, Operation::Set)?;

        // Sort by vault_key for deadlock prevention
        let mut sorted: Vec<_> = entries.to_vec();
        sorted.sort_by(|a, b| self.vault_key(a.0).cmp(&self.vault_key(b.0)));

        for (key, value) in &sorted {
            self.set_inner(requester, key, value, None)?;
        }

        self.log_operation(
            requester,
            "*",
            &AuditOperation::BatchSet {
                count: entries.len(),
            },
        );

        Ok(())
    }

    /// Detailed batch set result showing per-entry outcomes.
    pub fn batch_set_detailed(
        &self,
        requester: &str,
        entries: &[(&str, &str)],
    ) -> Result<BatchSetResult> {
        self.seal_guard.check_sealed()?;

        if entries.is_empty() {
            return Ok(BatchSetResult {
                succeeded: 0,
                failed: Vec::new(),
            });
        }

        self.check_rate_limit(requester, Operation::Set)?;

        let mut sorted: Vec<_> = entries.to_vec();
        sorted.sort_by(|a, b| self.vault_key(a.0).cmp(&self.vault_key(b.0)));

        let mut succeeded = 0;
        let mut failed = Vec::new();
        for (key, value) in &sorted {
            match self.set_inner(requester, key, value, None) {
                Ok(()) => succeeded += 1,
                Err(e) => failed.push(((*key).to_string(), e)),
            }
        }

        self.log_operation(
            requester,
            "*",
            &AuditOperation::BatchSet {
                count: entries.len(),
            },
        );

        Ok(BatchSetResult { succeeded, failed })
    }

    fn is_expired(tensor: &TensorData) -> Option<bool> {
        if let Some(TensorValue::Scalar(ScalarValue::Int(expires_at))) = tensor.get("_expires_at") {
            Some(Self::current_timestamp() >= *expires_at)
        } else {
            None
        }
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

    // ========== Wrapping (Cubbyhole) ==========

    /// Wrap a secret behind a single-use token.
    pub fn wrap_secret(&self, requester: &str, key: &str, ttl_ms: i64) -> Result<String> {
        self.seal_guard.check_sealed()?;
        self.check_rate_limit(requester, Operation::Wrap)?;
        self.check_access(requester, key)?;

        let value = self.get(requester, key)?;
        let (token, _) = crate::wrapping::wrap_secret(&self.store, &self.cipher, &value, ttl_ms)?;

        self.log_operation(requester, key, &AuditOperation::Wrap);
        Ok(token)
    }

    /// Unwrap a secret -- single use, deletes after read.
    pub fn unwrap_secret(&self, token: &str) -> Result<String> {
        self.seal_guard.check_sealed()?;
        let value = crate::wrapping::unwrap_secret(&self.store, &self.cipher, token)?;
        self.log_operation("anonymous", token, &AuditOperation::Unwrap);
        Ok(value)
    }

    /// Get info about a wrapping token.
    pub fn wrapping_token_info(&self, token: &str) -> Option<WrappingToken> {
        crate::wrapping::wrapping_token_info(&self.store, token)
    }

    // ========== Dependencies ==========

    /// Add a dependency: child_key depends on parent_key.
    pub fn add_dependency(&self, requester: &str, parent_key: &str, child_key: &str) -> Result<()> {
        self.seal_guard.check_sealed()?;
        self.check_access_with_permission(requester, parent_key, Permission::Admin)?;
        self.check_access_with_permission(requester, child_key, Permission::Admin)?;

        let parent_node = self.secret_node_key(parent_key);
        let child_node = self.secret_node_key(child_key);
        let ts = Self::current_timestamp();
        crate::dependency::add_dependency(&self.graph, &parent_node, &child_node, ts)?;

        self.log_operation(requester, parent_key, &AuditOperation::AddDependency);
        Ok(())
    }

    /// Remove a dependency between two secrets.
    pub fn remove_dependency(
        &self,
        requester: &str,
        parent_key: &str,
        child_key: &str,
    ) -> Result<()> {
        self.seal_guard.check_sealed()?;
        self.check_access_with_permission(requester, parent_key, Permission::Admin)?;

        let parent_node = self.secret_node_key(parent_key);
        let child_node = self.secret_node_key(child_key);
        crate::dependency::remove_dependency(&self.graph, &parent_node, &child_node)?;

        self.log_operation(requester, parent_key, &AuditOperation::RemoveDependency);
        Ok(())
    }

    /// Get secrets that depend on the given secret (children).
    pub fn get_dependencies(&self, requester: &str, key: &str) -> Result<Vec<String>> {
        self.check_access(requester, key)?;
        let node = self.secret_node_key(key);
        Ok(crate::dependency::get_dependencies(&self.graph, &node))
    }

    /// Get secrets the given secret depends on (parents).
    pub fn get_dependents(&self, requester: &str, key: &str) -> Result<Vec<String>> {
        self.check_access(requester, key)?;
        let node = self.secret_node_key(key);
        Ok(crate::dependency::get_dependents(&self.graph, &node))
    }

    /// Transitive impact analysis for a secret.
    pub fn impact_analysis(&self, requester: &str, key: &str) -> Result<ImpactReport> {
        self.check_access(requester, key)?;
        let node = self.secret_node_key(key);
        let report = crate::dependency::impact_analysis(&self.graph, &node);
        self.log_operation(requester, key, &AuditOperation::ImpactAnalysis);
        Ok(report)
    }

    /// Add a weighted dependency: child_key depends on parent_key with a weight.
    pub fn add_weighted_dependency(
        &self,
        requester: &str,
        parent_key: &str,
        child_key: &str,
        weight: crate::dependency::DependencyWeight,
        description: Option<&str>,
    ) -> Result<()> {
        self.seal_guard.check_sealed()?;
        self.check_access_with_permission(requester, parent_key, Permission::Admin)?;
        self.check_access_with_permission(requester, child_key, Permission::Admin)?;

        let parent_node = self.secret_node_key(parent_key);
        let child_node = self.secret_node_key(child_key);
        let ts = Self::current_timestamp();
        crate::dependency::add_weighted_dependency(
            &self.graph,
            &parent_node,
            &child_node,
            weight,
            description,
            ts,
        )?;

        self.log_operation(requester, parent_key, &AuditOperation::AddDependency);
        Ok(())
    }

    /// Weighted transitive impact analysis for a secret.
    pub fn weighted_impact_analysis(
        &self,
        requester: &str,
        key: &str,
    ) -> Result<crate::dependency::WeightedImpactReport> {
        self.check_access(requester, key)?;
        let node = self.secret_node_key(key);
        let report = crate::dependency::weighted_impact_analysis(&self.graph, &node);
        self.log_operation(requester, key, &AuditOperation::WeightedImpactAnalysis);
        Ok(report)
    }

    /// Generate a prioritized rotation plan for a secret's dependencies.
    pub fn rotation_plan(
        &self,
        requester: &str,
        key: &str,
    ) -> Result<crate::dependency::RotationPlan> {
        self.check_access(requester, key)?;
        let node = self.secret_node_key(key);
        let plan = crate::dependency::rotation_plan(&self.graph, &node);
        self.log_operation(requester, key, &AuditOperation::RotationPlan);
        Ok(plan)
    }

    /// Compute heat-kernel trust diffusion across the access graph.
    pub fn heat_kernel_trust(
        &self,
        config: crate::heat_kernel::HeatKernelConfig,
    ) -> crate::heat_kernel::HeatKernelTrustReport {
        let report = crate::heat_kernel::heat_kernel_trust(self, config.clone());
        self.log_operation(
            Self::ROOT,
            "_system",
            &AuditOperation::HeatKernelTrust {
                diffusion_time: config.diffusion_time,
            },
        );
        report
    }

    /// Build a 3D access tensor from audit log data.
    pub fn build_access_tensor(
        &self,
        config: crate::access_tensor::AccessTensorConfig,
    ) -> Result<crate::access_tensor::AccessTensor> {
        let num_buckets = config.num_buckets;
        let tensor = crate::access_tensor::AccessTensor::from_vault(self, config)?;
        self.log_operation(
            Self::ROOT,
            "_system",
            &AuditOperation::BuildAccessTensor { num_buckets },
        );
        Ok(tensor)
    }

    /// Analyze temporal access patterns via TT decomposition and drift detection.
    pub fn analyze_temporal_patterns(
        &self,
        tensor_config: crate::access_tensor::AccessTensorConfig,
        analysis_config: crate::temporal_analysis::TemporalAnalysisConfig,
    ) -> Result<crate::temporal_analysis::TemporalAnalysisReport> {
        let tensor = crate::access_tensor::AccessTensor::from_vault(self, tensor_config)?;
        let report = crate::temporal_analysis::analyze_temporal_patterns(&tensor, analysis_config);
        self.log_operation(
            Self::ROOT,
            "_system",
            &AuditOperation::AnalyzeTemporalPatterns,
        );
        Ok(report)
    }

    // ========== Quotas ==========

    /// Set a resource quota for a namespace (root only).
    pub fn set_quota(&self, requester: &str, namespace: &str, quota: ResourceQuota) -> Result<()> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can set quotas".to_string(),
            ));
        }
        self.quota_manager
            .set_quota(&self.store, namespace, quota)?;
        self.log_operation(requester, namespace, &AuditOperation::SetQuota);
        Ok(())
    }

    /// Get the quota for a namespace.
    pub fn get_quota(&self, namespace: &str) -> Option<ResourceQuota> {
        self.quota_manager.get_quota(namespace)
    }

    /// Get current usage for a namespace.
    pub fn get_usage(&self, namespace: &str) -> ResourceUsage {
        self.quota_manager.get_usage(namespace)
    }

    /// Remove a quota for a namespace (root only).
    pub fn remove_quota(&self, requester: &str, namespace: &str) -> Result<()> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can remove quotas".to_string(),
            ));
        }
        self.quota_manager.remove_quota(&self.store, namespace)?;
        self.log_operation(requester, namespace, &AuditOperation::RemoveQuota);
        Ok(())
    }

    // ========== Dynamic Secrets ==========

    /// Generate a dynamic (short-lived) secret.
    pub fn generate_dynamic_secret(
        &self,
        requester: &str,
        template: &SecretTemplate,
        ttl_ms: i64,
        one_time: bool,
    ) -> Result<(String, String)> {
        self.seal_guard.check_sealed()?;
        self.check_rate_limit(requester, Operation::DynamicGenerate)?;

        let value = crate::dynamic::generate_from_template(template);
        let id = crate::dynamic::new_secret_id();
        let now = Self::current_timestamp() * 1000; // Convert to millis
        let expires = now + ttl_ms;

        crate::dynamic::store_metadata(&self.store, &id, template, requester, expires, one_time)?;

        // Store the actual value encrypted via normal vault set
        self.set(requester, &id, &value)?;

        self.log_operation(requester, &id, &AuditOperation::DynamicGenerate);
        Ok((id, value))
    }

    /// Get a dynamic secret by ID.
    pub fn get_dynamic_secret(&self, requester: &str, secret_id: &str) -> Result<String> {
        self.seal_guard.check_sealed()?;

        if let Some(meta) = crate::dynamic::get_metadata(&self.store, secret_id) {
            if meta.consumed {
                return Err(VaultError::NotFound(format!(
                    "dynamic secret consumed: {secret_id}"
                )));
            }
            // Explicit metadata-level expiration check
            let now_ms = Self::current_timestamp() * 1000;
            if meta.expires_at_ms > 0 && now_ms >= meta.expires_at_ms {
                return Err(VaultError::SecretExpired(format!(
                    "dynamic secret expired: {secret_id}"
                )));
            }
            // Get value FIRST, then mark consumed (only on success)
            let value = self.get(requester, secret_id)?;
            if meta.one_time {
                crate::dynamic::mark_consumed(&self.store, secret_id)?;
            }
            return Ok(value);
        }

        self.get(requester, secret_id)
    }

    /// List dynamic secret metadata (root only).
    pub fn list_dynamic_secrets(&self, requester: &str) -> Result<Vec<DynamicSecretMetadata>> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can list dynamic secrets".to_string(),
            ));
        }
        Ok(crate::dynamic::list_metadata(&self.store))
    }

    /// Revoke a dynamic secret (root or original requester only).
    pub fn revoke_dynamic_secret(&self, requester: &str, secret_id: &str) -> Result<()> {
        self.seal_guard.check_sealed()?;
        if requester != Self::ROOT {
            if let Some(meta) = crate::dynamic::get_metadata(&self.store, secret_id) {
                if meta.requester != requester {
                    return Err(VaultError::AccessDenied(
                        "only root or the original requester can revoke dynamic secrets"
                            .to_string(),
                    ));
                }
            }
        }
        crate::dynamic::revoke_metadata(&self.store, secret_id);
        self.delete(requester, secret_id).ok(); // Best effort delete
        Ok(())
    }

    // ========== Seal/Unseal ==========

    /// Seal the vault, blocking all crypto operations.
    ///
    /// Zeroizes all in-memory key material after persisting the seal state.
    pub fn seal(&mut self) -> Result<()> {
        self.seal_guard.seal();
        self.seal_guard.persist(&self.store);
        // Audit BEFORE zeroizing so audit key is still live for the seal event
        self.log_operation(Self::ROOT, "vault", &AuditOperation::Seal);
        // Zeroize all key material
        self.cipher = Cipher::from_raw_key([0u8; 32]);
        self.transit_cipher = Cipher::from_raw_key([0u8; 32]);
        self.snapshot_cipher = Cipher::from_raw_key([0u8; 32]);
        self.obfuscator = Obfuscator::from_zeroed();
        self.edge_signer = EdgeSigner::from_zeroed();
        *self.audit_key = [0u8; 32];
        Ok(())
    }

    /// Unseal the vault with the master password.
    pub fn unseal(&mut self, master_password: &[u8]) -> Result<()> {
        // Re-derive keys
        let derived = if self.config.salt.is_some() {
            let (key, _) = MasterKey::derive(master_password, &self.config)?;
            key
        } else if let Some(persisted_salt) = Self::load_salt(&self.store) {
            MasterKey::derive_with_salt(master_password, &persisted_salt, &self.config)?
        } else {
            return Err(VaultError::KeyDerivationError("no salt found".to_string()));
        };

        self.cipher = Cipher::new(&derived);
        self.obfuscator = Obfuscator::new(&derived);
        self.edge_signer = EdgeSigner::new(&derived);
        self.transit_cipher = Cipher::from_raw_key(derived.transit_key());
        self.audit_key = Zeroizing::new(derived.audit_key());
        self.snapshot_cipher = Cipher::from_raw_key(derived.snapshot_key());
        self.sync_manager
            .update_cipher(Cipher::from_raw_key(derived.sync_key()));

        self.seal_guard.unseal();
        self.seal_guard.persist(&self.store);
        self.log_operation(Self::ROOT, "vault", &AuditOperation::Unseal);
        Ok(())
    }

    /// Unseal the vault using Shamir key shares.
    pub fn unseal_with_shares(&mut self, shares: &[KeyShare]) -> Result<()> {
        let master_key = crate::shamir::reconstruct_master_key(shares)?;

        self.cipher = Cipher::new(&master_key);
        self.obfuscator = Obfuscator::new(&master_key);
        self.edge_signer = EdgeSigner::new(&master_key);
        self.transit_cipher = Cipher::from_raw_key(master_key.transit_key());
        self.audit_key = Zeroizing::new(master_key.audit_key());
        self.snapshot_cipher = Cipher::from_raw_key(master_key.snapshot_key());
        self.sync_manager
            .update_cipher(Cipher::from_raw_key(master_key.sync_key()));

        self.seal_guard.unseal();
        self.seal_guard.persist(&self.store);
        self.log_operation(Self::ROOT, "vault", &AuditOperation::Unseal);
        Ok(())
    }

    /// Check if the vault is sealed.
    pub fn is_sealed(&self) -> bool {
        self.seal_guard.is_sealed()
    }

    // ========== Policy Templates ==========

    /// Add a policy template (root only).
    pub fn add_policy(&self, requester: &str, template: PolicyTemplate) -> Result<()> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can add policies".to_string(),
            ));
        }
        self.policy_manager.add_policy(&self.store, template)?;
        self.log_operation(requester, "policy", &AuditOperation::AddPolicy);
        Ok(())
    }

    /// Remove a policy template (root only).
    pub fn remove_policy(&self, requester: &str, name: &str) -> Result<()> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can remove policies".to_string(),
            ));
        }
        self.policy_manager.remove_policy(&self.store, name)?;
        self.log_operation(requester, name, &AuditOperation::RemovePolicy);
        Ok(())
    }

    /// List all policy templates.
    pub fn list_policies(&self) -> Vec<PolicyTemplate> {
        self.policy_manager.list_policies()
    }

    /// Get a specific policy template.
    pub fn get_policy(&self, name: &str) -> Option<PolicyTemplate> {
        self.policy_manager.get_policy(name)
    }

    /// Evaluate all policies for an entity.
    pub fn evaluate_policies(&self, entity: &str) -> Vec<PolicyMatch> {
        self.policy_manager.evaluate_policies(entity)
    }

    // ========== Point-in-Time Recovery ==========

    /// Create a snapshot of the current vault state.
    pub fn create_snapshot(&self, requester: &str, label: &str) -> Result<VaultSnapshot> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can create snapshots".to_string(),
            ));
        }
        let snap = crate::pitr::create_snapshot(
            &self.store,
            label,
            Some(&self.snapshot_cipher),
            Some(self.audit_key.as_ref()),
        )?;
        self.log_operation(requester, &snap.id, &AuditOperation::CreateSnapshot);
        Ok(snap)
    }

    /// Restore vault state from a snapshot.
    pub fn restore_snapshot(&self, requester: &str, snapshot_id: &str) -> Result<usize> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can restore snapshots".to_string(),
            ));
        }
        let count = crate::pitr::restore_snapshot(
            &self.store,
            snapshot_id,
            Some(&self.snapshot_cipher),
            Some(self.audit_key.as_ref()),
        )?;
        self.log_operation(requester, snapshot_id, &AuditOperation::RestoreSnapshot);
        Ok(count)
    }

    /// List all snapshots.
    pub fn list_snapshots(&self) -> Vec<VaultSnapshot> {
        crate::pitr::list_snapshots(&self.store)
    }

    /// Delete a snapshot.
    pub fn delete_snapshot(&self, requester: &str, snapshot_id: &str) -> Result<()> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can delete snapshots".to_string(),
            ));
        }
        crate::pitr::delete_snapshot(&self.store, snapshot_id);
        Ok(())
    }

    // ========== Rotation Policies ==========

    /// Set an automated rotation policy for a secret (Admin required).
    pub fn set_rotation_policy(
        &self,
        requester: &str,
        key: &str,
        policy: RotationPolicy,
    ) -> Result<()> {
        self.check_access_with_permission(requester, key, Permission::Admin)?;
        let obf = self.obfuscator.obfuscate_key(key);
        self.rotation_manager
            .set_policy(&self.store, &obf, policy)?;
        self.log_operation(requester, key, &AuditOperation::AutoRotate);
        Ok(())
    }

    /// Get the rotation policy for a secret.
    pub fn get_rotation_policy(&self, _requester: &str, key: &str) -> Option<RotationPolicy> {
        let obf = self.obfuscator.obfuscate_key(key);
        self.rotation_manager.get_policy(&obf)
    }

    /// Remove a rotation policy.
    pub fn remove_rotation_policy(&self, requester: &str, key: &str) -> Result<()> {
        self.check_access_with_permission(requester, key, Permission::Admin)?;
        let obf = self.obfuscator.obfuscate_key(key);
        self.rotation_manager.remove_policy(&self.store, &obf);
        Ok(())
    }

    /// Check for pending rotations.
    pub fn check_pending_rotations(&self) -> Vec<PendingRotation> {
        self.rotation_manager.check_pending()
    }

    /// Execute a rotation: generate a new value and rotate the secret.
    pub fn execute_rotation(&self, requester: &str, key: &str) -> Result<String> {
        self.seal_guard.check_sealed()?;
        self.check_access_with_permission(requester, key, Permission::Write)?;

        let obf = self.obfuscator.obfuscate_key(key);
        let policy = self
            .rotation_manager
            .get_policy(&obf)
            .ok_or_else(|| VaultError::NotFound(format!("no rotation policy for '{key}'")))?;

        let new_value = match &policy.generator {
            RotationGenerator::None => {
                return Err(VaultError::InvalidKey(
                    "rotation policy has no generator; rotate manually".to_string(),
                ));
            },
            RotationGenerator::Password(config) => {
                crate::dynamic::generate_from_template(&SecretTemplate::Password(config.clone()))
            },
            RotationGenerator::Token(config) => {
                crate::dynamic::generate_from_template(&SecretTemplate::Token(config.clone()))
            },
        };

        self.rotate(requester, key, &new_value)?;
        self.rotation_manager.mark_rotated(&self.store, &obf)?;
        self.log_operation(requester, key, &AuditOperation::AutoRotate);
        Ok(new_value)
    }

    /// List all rotation policies.
    pub fn list_rotation_policies(&self) -> Vec<RotationPolicy> {
        self.rotation_manager.list_policies()
    }

    // ========== Engine Registry ==========

    /// Register a secret engine plugin.
    pub fn register_engine(&self, engine: Box<dyn crate::engine::SecretEngine>) -> Result<()> {
        self.engine_registry.register(engine)
    }

    /// Unregister a secret engine.
    pub fn unregister_engine(&self, name: &str) -> Result<()> {
        self.engine_registry.unregister(name)
    }

    /// List registered engines.
    pub fn list_engines(&self) -> Vec<String> {
        self.engine_registry.list_engines()
    }

    /// Generate a secret using a named engine.
    pub fn engine_generate(
        &self,
        requester: &str,
        engine_name: &str,
        params: &serde_json::Value,
    ) -> Result<String> {
        self.seal_guard.check_sealed()?;
        let value = self.engine_registry.generate(engine_name, params)?;
        self.log_operation(requester, engine_name, &AuditOperation::EngineGenerate);
        Ok(value)
    }

    /// Revoke a secret via a named engine.
    pub fn engine_revoke(&self, requester: &str, engine_name: &str, secret_id: &str) -> Result<()> {
        self.engine_registry.revoke(engine_name, secret_id)?;
        self.log_operation(requester, engine_name, &AuditOperation::EngineRevoke);
        Ok(())
    }

    // ========== PKI ==========

    /// Initialize the PKI engine (root only).
    pub fn init_pki(&self, requester: &str) -> Result<()> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can initialize PKI".to_string(),
            ));
        }
        PkiEngine::init_ca(&self.store)?;
        Ok(())
    }

    /// Issue a certificate (root only).
    pub fn issue_certificate(
        &self,
        requester: &str,
        request: &CertificateRequest,
        ttl: Duration,
    ) -> Result<(String, Vec<u8>)> {
        self.seal_guard.check_sealed()?;
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can issue certificates".to_string(),
            ));
        }
        let result = PkiEngine::issue_certificate(&self.store, request, ttl)?;
        self.log_operation(requester, &result.0, &AuditOperation::IssueCertificate);
        Ok(result)
    }

    /// Revoke a certificate (root only).
    pub fn revoke_certificate(&self, requester: &str, serial: &str) -> Result<()> {
        self.seal_guard.check_sealed()?;
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can revoke certificates".to_string(),
            ));
        }
        PkiEngine::revoke_certificate(&self.store, serial)?;
        self.log_operation(requester, serial, &AuditOperation::RevokeCertificate);
        Ok(())
    }

    /// List all certificates (root only).
    pub fn list_certificates(&self, requester: &str) -> Result<Vec<CertInfo>> {
        if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can list certificates".to_string(),
            ));
        }
        Ok(PkiEngine::list_certificates(&self.store))
    }

    /// Get the CA certificate PEM.
    pub fn get_ca_certificate(&self, _requester: &str) -> Result<Vec<u8>> {
        PkiEngine::get_ca_certificate(&self.store)
    }

    /// Get the certificate revocation list.
    pub fn get_revocation_list(&self) -> Result<crate::pki::RevocationList> {
        PkiEngine::get_revocation_list(&self.store)
    }

    /// Check if a certificate is revoked by serial number.
    pub fn is_certificate_revoked(&self, serial: &str) -> bool {
        PkiEngine::is_revoked(&self.store, serial)
    }

    // ========== Sync ==========

    /// Register a sync target.
    pub fn register_sync_target(
        &self,
        target: Box<dyn crate::sync_target::SyncTarget>,
    ) -> Result<()> {
        self.sync_manager.register_target(target)
    }

    /// Subscribe a secret to a sync target.
    pub fn subscribe_sync(&self, requester: &str, key: &str, target_name: &str) -> Result<()> {
        self.check_access(requester, key)?;
        self.sync_manager.subscribe(&self.store, key, target_name)?;
        self.log_operation(requester, key, &AuditOperation::SyncSubscribe);
        Ok(())
    }

    /// Unsubscribe a secret from a sync target.
    pub fn unsubscribe_sync(&self, requester: &str, key: &str, target_name: &str) -> Result<()> {
        self.check_access(requester, key)?;
        self.sync_manager.unsubscribe(&self.store, key, target_name)
    }

    /// Manually trigger a sync for a secret key.
    pub fn trigger_sync(&self, requester: &str, key: &str) -> Result<usize> {
        self.check_access(requester, key)?;
        let value = self.get(requester, key)?;
        let count = self.sync_manager.trigger_sync(key, &value)?;
        self.log_operation(requester, key, &AuditOperation::SyncPush);
        Ok(count)
    }

    /// List sync targets.
    pub fn list_sync_targets(&self) -> Vec<String> {
        self.sync_manager.list_targets()
    }

    /// Health check all sync targets.
    pub fn sync_health(&self) -> Vec<(String, bool)> {
        self.sync_manager.health_check()
    }

    /// Comprehensive vault health and status information.
    pub fn vault_status(&self) -> crate::VaultStatus {
        crate::VaultStatus {
            sealed: self.seal_guard.is_sealed(),
            total_secrets: self.store.scan(Self::PREFIX).len(),
            sync_health: self.sync_manager.health_check(),
            pending_rotations: self.rotation_manager.check_pending().len(),
            snapshot_count: crate::pitr::list_snapshots(&self.store).len(),
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

        // Re-encrypt under new keys FIRST (both old and new coexist temporarily)
        let old_to_new_node_map =
            self.reencrypt_secrets(&decrypted_secrets, &new_cipher, &new_obfuscator)?;

        // Re-sign edges
        self.resign_edges(&edge_records, &new_signer, &old_to_new_node_map);

        // Delete old entries AFTER successful re-encryption
        let new_secret_nodes: std::collections::HashSet<&str> =
            old_to_new_node_map.values().map(String::as_str).collect();
        self.delete_old_entries(&vault_keys, &new_secret_nodes);

        // Swap key material and persist
        Self::save_salt(&self.store, new_salt)?;
        self.cipher = new_cipher;
        self.obfuscator = new_obfuscator;
        self.edge_signer = new_signer;
        self.audit_key = Zeroizing::new(new_master.audit_key());
        self.transit_cipher = Cipher::from_raw_key(new_master.transit_key());
        self.snapshot_cipher = Cipher::from_raw_key(new_master.snapshot_key());
        self.sync_manager
            .update_cipher(Cipher::from_raw_key(new_master.sync_key()));
        self.ttl_tracker.persist(&self.store).ok();

        let audit_log = AuditLog::new(&self.store, Some(*self.audit_key));
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

            let expires_at = match tensor.get("_expires_at") {
                Some(TensorValue::Scalar(ScalarValue::Int(ts))) => Some(*ts),
                _ => None,
            };

            let versions = self.decrypt_versions(&tensor, vault_key.as_bytes())?;
            secrets.push(DecryptedSecret {
                name,
                versions,
                creator,
                created,
                expires_at,
            });
        }
        Ok(secrets)
    }

    fn decrypt_versions(&self, tensor: &TensorData, aad: &[u8]) -> Result<Vec<(String, i64)>> {
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

            let padded = self.cipher.decrypt_with_aad(ciphertext, nonce, aad)?;
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

    pub(crate) fn node_entity_key(&self, node_id: u64) -> Option<String> {
        self.graph
            .get_node(node_id)
            .ok()
            .and_then(|n| match n.properties.get("entity_key") {
                Some(PropertyValue::String(s)) => Some(s.clone()),
                _ => None,
            })
    }

    fn delete_old_entries(
        &self,
        vault_keys: &[String],
        preserve_nodes: &std::collections::HashSet<&str>,
    ) {
        for vault_key in vault_keys {
            if let Ok(t) = self.store.get(vault_key) {
                for blob_key in Self::get_version_blobs(&t) {
                    self.store.delete(&blob_key).ok();
                }
            }
            self.store.delete(vault_key).ok();
        }
        for key in &self.store.scan("vault_secret:") {
            if !preserve_nodes.contains(key.as_str()) {
                self.store.delete(key).ok();
            }
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
                self.reencrypt_versions(secret, new_cipher, new_obfuscator, &new_vault_key)?;

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
                self.graph.update_node(node_id, None, props).map_err(|e| {
                    VaultError::GraphError(format!("rotation: node update failed: {e}"))
                })?;
            }
        }
        Ok(old_to_new)
    }

    fn reencrypt_versions(
        &self,
        secret: &DecryptedSecret,
        new_cipher: &Cipher,
        new_obfuscator: &Obfuscator,
        new_vault_key: &str,
    ) -> Result<(Vec<String>, String, Vec<u8>)> {
        let mut keys = Vec::new();
        let mut latest_blob = String::new();
        let mut latest_nonce = Vec::new();

        for (plaintext, timestamp) in &secret.versions {
            let padded = obfuscation::pad_plaintext(plaintext.as_bytes())?;
            let (ciphertext, nonce) =
                new_cipher.encrypt_with_aad(&padded, new_vault_key.as_bytes())?;
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
        let (encrypted_name, key_nonce) =
            new_cipher.encrypt_with_aad(secret.name.as_bytes(), b"key_name")?;
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

        // Preserve expiration from old secret
        if let Some(expires_at) = secret.expires_at {
            tensor.set(
                "_expires_at",
                TensorValue::Scalar(ScalarValue::Int(expires_at)),
            );
        }

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
    expires_at: Option<i64>,
}

struct EdgeRecord {
    edge_id: u64,
    from_key: String,
    to_key: String,
    edge_type: String,
    capacity: Option<i64>,
    has_sig: bool,
}
