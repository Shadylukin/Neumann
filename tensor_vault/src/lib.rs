//! Tensor Vault: Secure secret storage with graph-based access control.
//!
//! Secrets are encrypted at rest using AES-256-GCM. Access is controlled
//! by graph topology - a requester must have a path to the secret node.
//!
//! Security features:
//! - AES-256-GCM authenticated encryption
//! - Argon2id key derivation (GPU/ASIC resistant)
//! - Key obfuscation via HMAC (hides secret names in storage)
//! - Metadata encryption (hides creator, timestamps)
//! - Length padding (hides plaintext size)
//! - Pointer indirection (hides storage patterns)
//! - Graph-based access control (topological authorization)

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::format_collect)]

mod access;
mod audit;
mod encryption;
mod key;
mod obfuscation;
mod rate_limit;
mod ttl;

pub use access::AccessController;
pub use audit::{AuditEntry, AuditLog, AuditOperation};
pub use encryption::{Cipher, NONCE_SIZE};
pub use key::{MasterKey, KEY_SIZE};
pub use obfuscation::{Obfuscator, PaddingSize};
pub use rate_limit::{Operation, RateLimitConfig, RateLimiter};
pub use ttl::GrantTTLTracker;

use graph_engine::GraphEngine;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

/// Permission levels for vault access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Permission {
    /// Read-only access: can get(), list()
    #[default]
    Read,
    /// Write access: can get(), list(), set(), rotate()
    Write,
    /// Admin access: can get(), list(), set(), rotate(), delete(), grant(), revoke()
    Admin,
}

impl Permission {
    /// Check if this permission level allows the given operation.
    pub fn allows(&self, required: Self) -> bool {
        matches!(
            (self, required),
            (Self::Admin, _) | (Self::Write, Self::Read | Self::Write) | (Self::Read, Self::Read)
        )
    }

    /// Get edge type suffix for this permission level.
    fn edge_suffix(self) -> &'static str {
        match self {
            Self::Read => "_READ",
            Self::Write => "_WRITE",
            Self::Admin => "_ADMIN",
        }
    }

    /// Parse from edge type.
    fn from_edge_type(edge_type: &str) -> Option<Self> {
        if edge_type.ends_with("_READ") {
            Some(Self::Read)
        } else if edge_type.ends_with("_WRITE") {
            Some(Self::Write)
        } else if edge_type.ends_with("_ADMIN") {
            Some(Self::Admin)
        } else if edge_type == "VAULT_ACCESS" {
            // Backward compatibility: old grants are Admin
            Some(Self::Admin)
        } else {
            None
        }
    }
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "READ"),
            Self::Write => write!(f, "WRITE"),
            Self::Admin => write!(f, "ADMIN"),
        }
    }
}

impl std::str::FromStr for Permission {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "READ" | "R" => Ok(Self::Read),
            "WRITE" | "W" => Ok(Self::Write),
            "ADMIN" | "A" => Ok(Self::Admin),
            _ => Err(format!("Invalid permission: {s}")),
        }
    }
}

/// Error types for vault operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VaultError {
    /// Access denied - no path from requester to secret.
    AccessDenied(String),
    /// Secret not found.
    NotFound(String),
    /// Encryption/decryption failed.
    CryptoError(String),
    /// Key derivation failed.
    KeyDerivationError(String),
    /// Storage error from `TensorStore`.
    StorageError(String),
    /// Graph error from path verification.
    GraphError(String),
    /// Invalid secret key format.
    InvalidKey(String),
    /// Insufficient permissions for operation.
    InsufficientPermission(String),
    /// Rate limit exceeded.
    RateLimited(String),
}

impl std::fmt::Display for VaultError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AccessDenied(msg) => write!(f, "access denied: {msg}"),
            Self::NotFound(key) => write!(f, "secret not found: {key}"),
            Self::CryptoError(msg) => write!(f, "crypto error: {msg}"),
            Self::KeyDerivationError(msg) => write!(f, "key derivation error: {msg}"),
            Self::StorageError(msg) => write!(f, "storage error: {msg}"),
            Self::GraphError(msg) => write!(f, "graph error: {msg}"),
            Self::InvalidKey(msg) => write!(f, "invalid key: {msg}"),
            Self::InsufficientPermission(msg) => write!(f, "insufficient permission: {msg}"),
            Self::RateLimited(msg) => write!(f, "rate limited: {msg}"),
        }
    }
}

impl std::error::Error for VaultError {}

pub type Result<T> = std::result::Result<T, VaultError>;

/// Configuration for the vault.
#[derive(Debug, Clone)]
pub struct VaultConfig {
    /// Salt for key derivation (randomly generated if not provided).
    pub salt: Option<[u8; 16]>,
    /// Memory cost for Argon2id (in KiB, default: 65536 = 64MB).
    pub argon2_memory_cost: u32,
    /// Time cost for Argon2id (iterations, default: 3).
    pub argon2_time_cost: u32,
    /// Parallelism for Argon2id (default: 4).
    pub argon2_parallelism: u32,
    /// Rate limiting configuration (None = disabled).
    pub rate_limit: Option<RateLimitConfig>,
    /// Maximum number of versions to keep per secret (default: 5).
    pub max_versions: usize,
}

impl Default for VaultConfig {
    fn default() -> Self {
        Self {
            salt: None,
            argon2_memory_cost: 65536,
            argon2_time_cost: 3,
            argon2_parallelism: 4,
            rate_limit: None,
            max_versions: 5,
        }
    }
}

impl VaultConfig {
    /// Create a config with rate limiting enabled.
    #[must_use]
    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit = Some(config);
        self
    }

    /// Create a config with custom max versions.
    #[must_use]
    pub fn with_max_versions(mut self, max: usize) -> Self {
        self.max_versions = max;
        self
    }
}

/// Information about a secret version.
#[derive(Debug, Clone)]
pub struct VersionInfo {
    /// Version number (1-based).
    pub version: u32,
    /// Unix timestamp in milliseconds when the version was created.
    pub created_at: i64,
}

/// Secure secret storage with graph-based access control.
pub struct Vault {
    store: TensorStore,
    pub graph: Arc<GraphEngine>,
    cipher: Cipher,
    obfuscator: Obfuscator,
    ttl_tracker: GrantTTLTracker,
    rate_limiter: Option<RateLimiter>,
    max_versions: usize,
}

impl Vault {
    /// Storage key prefix for vault secrets (obfuscated).
    const PREFIX: &'static str = "_vk:";
    /// Node ID for the root entity with universal access.
    pub const ROOT: &'static str = "node:root";
    /// Edge type for access grants.
    const ACCESS_EDGE: &'static str = "VAULT_ACCESS";

    /// Create a new vault with the given master key.
    pub fn new(
        master_key: &[u8],
        graph: Arc<GraphEngine>,
        store: TensorStore,
        config: VaultConfig,
    ) -> Result<Self> {
        let derived = MasterKey::derive(master_key, &config)?;
        let obfuscator = Obfuscator::new(&derived);
        let cipher = Cipher::new(derived);

        let rate_limiter = config.rate_limit.map(RateLimiter::new);
        let max_versions = config.max_versions.max(1); // At least 1 version

        let vault = Self {
            store,
            graph,
            cipher,
            obfuscator,
            ttl_tracker: GrantTTLTracker::new(),
            rate_limiter,
            max_versions,
        };

        vault.ensure_root_exists()?;
        Ok(vault)
    }

    /// Create vault from NEUMANN_VAULT_KEY environment variable.
    pub fn from_env(graph: Arc<GraphEngine>, store: TensorStore) -> Result<Self> {
        let key = std::env::var("NEUMANN_VAULT_KEY")
            .map_err(|_| VaultError::KeyDerivationError("NEUMANN_VAULT_KEY not set".to_string()))?;

        let decoded = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &key)
            .map_err(|e| {
                VaultError::KeyDerivationError(format!("Invalid base64 in NEUMANN_VAULT_KEY: {e}"))
            })?;

        Self::new(&decoded, graph, store, VaultConfig::default())
    }

    fn ensure_root_exists(&self) -> Result<()> {
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

    fn vault_key(&self, secret_key: &str) -> String {
        // Obfuscate the key using HMAC to hide the original key name
        let obfuscated = self.obfuscator.obfuscate_key(secret_key);
        format!("{}{}", Self::PREFIX, obfuscated)
    }

    fn blob_key(&self, secret_key: &str, nonce: &[u8]) -> String {
        // Generate storage ID for the ciphertext blob (pointer indirection)
        self.obfuscator.generate_storage_id(secret_key, nonce)
    }

    fn secret_node_key(secret_key: &str) -> String {
        format!("vault_secret:{secret_key}")
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
        let secret_node = Self::secret_node_key(key);
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
        let padded = obfuscation::pad_plaintext(value.as_bytes());
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

            // Obfuscate metadata
            let creator_bytes = requester.as_bytes();
            let obfuscated_creator = self.obfuscator.obfuscate_metadata(creator_bytes);
            let timestamp_bytes = timestamp.to_le_bytes();
            let obfuscated_timestamp = self.obfuscator.obfuscate_metadata(&timestamp_bytes);

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
                let _ = self.store.delete(old_blob);
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
        tensor.set(
            "_version_count",
            TensorValue::Scalar(ScalarValue::Int(1)), // Will be set correctly by get_version_blobs
        );

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
            self.graph
                .add_entity_edge(Self::ROOT, &secret_node, &edge_type)
                .map_err(|e| VaultError::GraphError(e.to_string()))?;
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
        // Get the old version value
        let value = self.get_version(requester, key, version)?;

        // Create new version with the old value
        // Note: This will create a new version, preserving history
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

        let secret_node = Self::secret_node_key(key);
        if !self.store.exists(&secret_node) {
            return Err(VaultError::NotFound(key.to_string()));
        }

        // Create edge with permission level encoded in type
        let edge_type = format!("{}{}", Self::ACCESS_EDGE, level.edge_suffix());
        self.graph
            .add_entity_edge(entity, &secret_node, &edge_type)
            .map_err(|e| VaultError::GraphError(e.to_string()))?;

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

        // Then register with TTL tracker
        self.ttl_tracker.add(entity, key, ttl);

        Ok(())
    }

    /// Revoke access to a secret (requires Admin permission).
    pub fn revoke(&self, requester: &str, entity: &str, key: &str) -> Result<()> {
        // Only those with Admin permission can revoke access
        self.check_access_with_permission(requester, key, Permission::Admin)?;

        let secret_node = Self::secret_node_key(key);

        // Find and delete the access edge (handles all permission levels)
        if let Ok(edges) = self.graph.get_entity_outgoing(entity) {
            for edge_key in edges {
                if let Ok((_, to, edge_type, _)) = self.graph.get_entity_edge(&edge_key) {
                    if to == secret_node && edge_type.starts_with(Self::ACCESS_EDGE) {
                        self.graph
                            .delete_entity_edge(&edge_key)
                            .map_err(|e| VaultError::GraphError(e.to_string()))?;
                    }
                }
            }
        }

        // Remove from TTL tracker if present
        self.ttl_tracker.remove(entity, key);

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
    /// This is called opportunistically during vault operations.
    pub fn cleanup_expired_grants(&self) -> usize {
        let expired = self.ttl_tracker.get_expired();
        let mut revoked = 0;

        for (entity, key) in expired {
            let secret_node = Self::secret_node_key(&key);

            // Find and delete the access edge
            if let Ok(edges) = self.graph.get_entity_outgoing(&entity) {
                for edge_key in edges {
                    if let Ok((_, to, edge_type, _)) = self.graph.get_entity_edge(&edge_key) {
                        if to == secret_node
                            && edge_type.starts_with(Self::ACCESS_EDGE)
                            && self.graph.delete_entity_edge(&edge_key).is_ok()
                        {
                            revoked += 1;
                        }
                    }
                }
            }
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
                let _ = self.store.delete(&blob_key);
            }
        }

        self.store
            .delete(&vault_storage_key)
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Also clean up the secret node
        let secret_node = Self::secret_node_key(key);
        let _ = self.store.delete(&secret_node);

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
        let padded = obfuscation::pad_plaintext(new_value.as_bytes());
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
                let _ = self.store.delete(old_blob);
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

        // Obfuscate rotation metadata
        let rotator_obf = self.obfuscator.obfuscate_metadata(requester.as_bytes());
        let timestamp_obf = self.obfuscator.obfuscate_metadata(&timestamp.to_le_bytes());

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
            // Decrypt the original key name from stored metadata
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
    pub fn scope(&self, entity: &str) -> ScopedVault<'_> {
        ScopedVault {
            vault: self,
            entity: entity.to_string(),
        }
    }

    /// Create a namespaced vault view for multi-tenant isolation.
    ///
    /// All keys accessed through the returned view are prefixed with the namespace.
    pub fn namespace(&self, namespace: &str, identity: &str) -> NamespacedVault<'_> {
        NamespacedVault::new(self, namespace, identity)
    }

    /// Access the underlying graph engine (for testing/benchmarking).
    pub fn graph(&self) -> &Arc<GraphEngine> {
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
        // Root always has Admin access
        if requester == Self::ROOT {
            return Ok(());
        }

        let secret_node = Self::secret_node_key(key);

        if AccessController::check_path_with_permission(
            &self.graph,
            requester,
            &secret_node,
            required,
        ) {
            Ok(())
        } else {
            // Determine if it's "no access" or "insufficient permission"
            if AccessController::check_path(&self.graph, requester, &secret_node) {
                Err(VaultError::InsufficientPermission(format!(
                    "{requester} has access but not {required} permission on '{key}'"
                )))
            } else {
                Err(VaultError::AccessDenied(format!(
                    "No path from {requester} to secret '{key}'"
                )))
            }
        }
    }

    fn has_access(&self, requester: &str, key: &str) -> bool {
        // Root always has access
        if requester == Self::ROOT {
            return true;
        }

        // Check if path exists from requester to secret node
        let secret_node = Self::secret_node_key(key);

        AccessController::check_path(&self.graph, requester, &secret_node)
    }

    fn check_rate_limit(&self, requester: &str, op: Operation) -> Result<()> {
        // Root is not rate limited
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

        let secret_node = Self::secret_node_key(key);
        AccessController::get_permission_level(&self.graph, requester, &secret_node)
    }

    fn log_operation(&self, requester: &str, key: &str, operation: &AuditOperation) {
        let audit_log = AuditLog::new(&self.store);
        audit_log.record(requester, key, operation);
    }

    /// Query audit entries for a specific secret.
    pub fn audit_log(&self, key: &str) -> Vec<AuditEntry> {
        let audit = AuditLog::new(&self.store);
        audit.by_secret(key)
    }

    /// Query audit entries by entity (who performed operations).
    pub fn audit_by_entity(&self, entity: &str) -> Vec<AuditEntry> {
        let audit = AuditLog::new(&self.store);
        audit.by_entity(entity)
    }

    /// Query audit entries since a timestamp (unix milliseconds).
    pub fn audit_since(&self, since_millis: i64) -> Vec<AuditEntry> {
        let audit = AuditLog::new(&self.store);
        audit.since(since_millis)
    }

    /// Get recent audit entries.
    pub fn audit_recent(&self, limit: usize) -> Vec<AuditEntry> {
        let audit = AuditLog::new(&self.store);
        audit.recent(limit)
    }

    fn matches_pattern(key: &str, pattern: &str) -> bool {
        if pattern.is_empty() || pattern == "*" {
            return true;
        }

        // Simple glob matching: only support trailing *
        if let Some(prefix) = pattern.strip_suffix('*') {
            key.starts_with(prefix)
        } else {
            key == pattern
        }
    }
}

/// A scoped view of the vault for a specific entity.
pub struct ScopedVault<'a> {
    vault: &'a Vault,
    entity: String,
}

impl ScopedVault<'_> {
    pub fn set(&self, key: &str, value: &str) -> Result<()> {
        self.vault.set(&self.entity, key, value)
    }

    pub fn get(&self, key: &str) -> Result<String> {
        self.vault.get(&self.entity, key)
    }

    pub fn delete(&self, key: &str) -> Result<()> {
        self.vault.delete(&self.entity, key)
    }

    pub fn list(&self, pattern: &str) -> Result<Vec<String>> {
        self.vault.list(&self.entity, pattern)
    }
}

/// A namespaced view of the vault that prefixes all keys with a namespace.
///
/// Provides isolation between different tenants or agent contexts.
pub struct NamespacedVault<'a> {
    vault: &'a Vault,
    namespace: String,
    identity: String,
}

impl<'a> NamespacedVault<'a> {
    /// Create a new namespaced vault view.
    pub fn new(vault: &'a Vault, namespace: &str, identity: &str) -> Self {
        Self {
            vault,
            namespace: namespace.to_string(),
            identity: identity.to_string(),
        }
    }

    fn prefixed_key(&self, key: &str) -> String {
        format!("{}:{}", self.namespace, key)
    }

    fn strip_prefix<'b>(&self, key: &'b str) -> Option<&'b str> {
        let prefix = format!("{}:", self.namespace);
        key.strip_prefix(&prefix)
    }

    /// Store a secret in the namespace.
    pub fn set(&self, key: &str, value: &str) -> Result<()> {
        let prefixed = self.prefixed_key(key);
        self.vault.set(&self.identity, &prefixed, value)
    }

    /// Retrieve a secret from the namespace.
    pub fn get(&self, key: &str) -> Result<String> {
        let prefixed = self.prefixed_key(key);
        self.vault.get(&self.identity, &prefixed)
    }

    /// Delete a secret from the namespace.
    pub fn delete(&self, key: &str) -> Result<()> {
        let prefixed = self.prefixed_key(key);
        self.vault.delete(&self.identity, &prefixed)
    }

    /// Rotate a secret in the namespace.
    pub fn rotate(&self, key: &str, new_value: &str) -> Result<()> {
        let prefixed = self.prefixed_key(key);
        self.vault.rotate(&self.identity, &prefixed, new_value)
    }

    /// List secrets in the namespace matching a pattern.
    ///
    /// Only returns keys within this namespace, with the namespace prefix stripped.
    pub fn list(&self, pattern: &str) -> Result<Vec<String>> {
        // Prefix the pattern with namespace
        let ns_pattern = format!("{}:{}", self.namespace, pattern);
        let keys = self.vault.list(&self.identity, &ns_pattern)?;

        // Strip namespace prefix from results
        Ok(keys
            .into_iter()
            .filter_map(|k| self.strip_prefix(&k).map(String::from))
            .collect())
    }

    /// Grant access to a secret in this namespace.
    pub fn grant(&self, entity: &str, key: &str, level: Permission) -> Result<()> {
        let prefixed = self.prefixed_key(key);
        self.vault
            .grant_with_permission(&self.identity, entity, &prefixed, level)
    }

    /// Revoke access to a secret in this namespace.
    pub fn revoke(&self, entity: &str, key: &str) -> Result<()> {
        let prefixed = self.prefixed_key(key);
        self.vault.revoke(&self.identity, entity, &prefixed)
    }

    /// Get the namespace name.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Get the identity.
    pub fn identity(&self) -> &str {
        &self.identity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap()
    }

    #[test]
    fn test_set_get_roundtrip() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "sk-secret123").unwrap();
        let value = vault.get(Vault::ROOT, "api_key").unwrap();

        assert_eq!(value, "sk-secret123");
    }

    #[test]
    fn test_access_denied_without_path() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_root_always_has_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        let value = vault.get(Vault::ROOT, "secret").unwrap();

        assert_eq!(value, "value");
    }

    #[test]
    fn test_grant_enables_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "sk-123").unwrap();

        // Alice can't access yet
        assert!(vault.get("user:alice", "api_key").is_err());

        // Grant access
        vault.grant(Vault::ROOT, "user:alice", "api_key").unwrap();

        // Now Alice can access
        let value = vault.get("user:alice", "api_key").unwrap();
        assert_eq!(value, "sk-123");
    }

    #[test]
    fn test_revoke_removes_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "sk-123").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "api_key").unwrap();

        // Alice can access
        assert!(vault.get("user:alice", "api_key").is_ok());

        // Revoke access
        vault.revoke(Vault::ROOT, "user:alice", "api_key").unwrap();

        // Alice can no longer access
        assert!(vault.get("user:alice", "api_key").is_err());
    }

    #[test]
    fn test_rotate_updates_value() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "old_value").unwrap();
        vault.rotate(Vault::ROOT, "api_key", "new_value").unwrap();

        let value = vault.get(Vault::ROOT, "api_key").unwrap();
        assert_eq!(value, "new_value");
    }

    #[test]
    fn test_delete_removes_secret() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "value").unwrap();
        vault.delete(Vault::ROOT, "api_key").unwrap();

        let result = vault.get(Vault::ROOT, "api_key");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_list_filters_by_pattern() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "openai:key", "sk-1").unwrap();
        vault.set(Vault::ROOT, "openai:org", "org-1").unwrap();
        vault.set(Vault::ROOT, "github:token", "ghp-1").unwrap();

        let openai_keys = vault.list(Vault::ROOT, "openai:*").unwrap();
        assert_eq!(openai_keys.len(), 2);
        assert!(openai_keys.contains(&"openai:key".to_string()));
        assert!(openai_keys.contains(&"openai:org".to_string()));

        let all_keys = vault.list(Vault::ROOT, "*").unwrap();
        assert_eq!(all_keys.len(), 3);
    }

    #[test]
    fn test_list_filters_by_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "public:key", "value1").unwrap();
        vault.set(Vault::ROOT, "private:key", "value2").unwrap();

        vault
            .grant(Vault::ROOT, "user:alice", "public:key")
            .unwrap();

        // Alice can only see public:key
        let alice_keys = vault.list("user:alice", "*").unwrap();
        assert_eq!(alice_keys.len(), 1);
        assert!(alice_keys.contains(&"public:key".to_string()));

        // Root can see all
        let root_keys = vault.list(Vault::ROOT, "*").unwrap();
        assert_eq!(root_keys.len(), 2);
    }

    #[test]
    fn test_scoped_vault() {
        let vault = create_test_vault();

        // Create a secret and grant access to alice
        vault.set(Vault::ROOT, "shared:key", "value").unwrap();
        vault
            .grant(Vault::ROOT, "user:alice", "shared:key")
            .unwrap();

        let scoped = vault.scope("user:alice");
        let value = scoped.get("shared:key").unwrap();
        assert_eq!(value, "value");
    }

    #[test]
    fn test_non_root_cannot_create_secrets() {
        let vault = create_test_vault();

        let result = vault.set("user:alice", "new_secret", "value");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_transitive_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Create a chain: alice -> team -> secret
        vault
            .graph
            .add_entity_edge("user:alice", "team:devs", "MEMBER")
            .unwrap();
        vault
            .graph
            .add_entity_edge("team:devs", "vault_secret:secret", "VAULT_ACCESS")
            .unwrap();

        // Alice can access via transitive path
        let value = vault.get("user:alice", "secret").unwrap();
        assert_eq!(value, "value");
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let vault = Arc::new(create_test_vault());

        vault.set(Vault::ROOT, "shared", "initial").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "shared").unwrap();
        vault.grant(Vault::ROOT, "user:bob", "shared").unwrap();

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let vault = Arc::clone(&vault);
                let user = if i % 2 == 0 { "user:alice" } else { "user:bob" };
                thread::spawn(move || {
                    for _ in 0..100 {
                        let _ = vault.get(user, "shared");
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_vault_error_display() {
        let errors = [
            (
                VaultError::AccessDenied("test".to_string()),
                "access denied: test",
            ),
            (
                VaultError::NotFound("key".to_string()),
                "secret not found: key",
            ),
            (
                VaultError::CryptoError("bad".to_string()),
                "crypto error: bad",
            ),
            (
                VaultError::KeyDerivationError("fail".to_string()),
                "key derivation error: fail",
            ),
            (
                VaultError::StorageError("io".to_string()),
                "storage error: io",
            ),
            (
                VaultError::GraphError("node".to_string()),
                "graph error: node",
            ),
            (
                VaultError::InvalidKey("empty".to_string()),
                "invalid key: empty",
            ),
        ];

        for (err, expected) in errors {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn test_scoped_vault_set_and_delete() {
        let vault = create_test_vault();

        // Grant alice permission to access secrets she creates
        let scoped = vault.scope(Vault::ROOT);

        // Root creates and the scoped vault works
        scoped.set("scoped:key", "scoped_value").unwrap();
        let value = scoped.get("scoped:key").unwrap();
        assert_eq!(value, "scoped_value");

        // Delete via scoped vault
        scoped.delete("scoped:key").unwrap();
        assert!(scoped.get("scoped:key").is_err());
    }

    #[test]
    fn test_scoped_vault_list() {
        let vault = create_test_vault();
        let scoped = vault.scope(Vault::ROOT);

        scoped.set("list:a", "1").unwrap();
        scoped.set("list:b", "2").unwrap();

        let keys = scoped.list("list:*").unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_list_exact_match() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "exact_key", "value1").unwrap();
        vault.set(Vault::ROOT, "exact_key_extra", "value2").unwrap();

        // Exact match (no glob)
        let keys = vault.list(Vault::ROOT, "exact_key").unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], "exact_key");
    }

    #[test]
    fn test_get_not_found() {
        let vault = create_test_vault();
        let result = vault.get(Vault::ROOT, "nonexistent");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_delete_not_found() {
        let vault = create_test_vault();
        let result = vault.delete(Vault::ROOT, "nonexistent");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_rotate_not_found() {
        let vault = create_test_vault();
        let result = vault.rotate(Vault::ROOT, "nonexistent", "new_value");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_grant_on_nonexistent_secret() {
        let vault = create_test_vault();
        // Grant on non-existent secret should fail
        let result = vault.grant(Vault::ROOT, "user:alice", "nonexistent");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_overwrite_existing_secret() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "key", "value1").unwrap();
        vault.set(Vault::ROOT, "key", "value2").unwrap();

        let value = vault.get(Vault::ROOT, "key").unwrap();
        assert_eq!(value, "value2");
    }

    #[test]
    fn test_empty_pattern_returns_all() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "a", "1").unwrap();
        vault.set(Vault::ROOT, "b", "2").unwrap();

        let keys = vault.list(Vault::ROOT, "").unwrap();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_from_env_with_valid_key() {
        use base64::Engine;

        // Set up a valid 32-byte key as base64
        let key_bytes = b"01234567890123456789012345678901";
        let encoded = base64::engine::general_purpose::STANDARD.encode(key_bytes);
        std::env::set_var("NEUMANN_VAULT_KEY", &encoded);

        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let result = Vault::from_env(graph, store);

        // Restore env
        std::env::remove_var("NEUMANN_VAULT_KEY");

        assert!(result.is_ok());
    }

    #[test]
    fn test_from_env_missing_key() {
        std::env::remove_var("NEUMANN_VAULT_KEY");

        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let result = Vault::from_env(graph, store);

        assert!(matches!(result, Err(VaultError::KeyDerivationError(_))));
    }

    #[test]
    fn test_from_env_invalid_base64() {
        std::env::set_var("NEUMANN_VAULT_KEY", "not-valid-base64!!!");

        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let result = Vault::from_env(graph, store);

        std::env::remove_var("NEUMANN_VAULT_KEY");

        assert!(matches!(result, Err(VaultError::KeyDerivationError(_))));
    }

    #[test]
    fn test_overwrite_without_access_denied() {
        let vault = create_test_vault();

        // Root creates secret
        vault.set(Vault::ROOT, "protected", "initial").unwrap();

        // Alice tries to overwrite without access
        let result = vault.set("user:alice", "protected", "hacked");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_overwrite_with_granted_access() {
        let vault = create_test_vault();

        // Root creates secret and grants alice access
        vault.set(Vault::ROOT, "shared", "initial").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "shared").unwrap();

        // Alice can now overwrite
        vault.set("user:alice", "shared", "updated").unwrap();

        let value = vault.get(Vault::ROOT, "shared").unwrap();
        assert_eq!(value, "updated");
    }

    #[test]
    fn test_short_key_still_works() {
        // Argon2id can derive a key from any length input
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());

        // Short key works - Argon2id derives a proper key
        let result = Vault::new(
            b"short",
            graph.clone(),
            store.clone(),
            VaultConfig::default(),
        );
        assert!(result.is_ok());

        // Even empty key works with Argon2id
        let store2 = TensorStore::new();
        let result = Vault::new(b"", graph, store2, VaultConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_revoke_on_nonexistent_secret() {
        let vault = create_test_vault();

        // Non-root user trying to revoke on non-existent secret fails with AccessDenied
        // because check_access is called first and alice has no path to the secret
        let result = vault.revoke("user:alice", "user:bob", "nonexistent");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        // Root can revoke on non-existent secret (succeeds but does nothing)
        let result = vault.revoke(Vault::ROOT, "user:alice", "nonexistent");
        assert!(result.is_ok());
    }

    // === Security feature tests ===

    #[test]
    fn test_storage_key_is_obfuscated() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "secret").unwrap();

        // The original key should NOT appear in storage keys
        let all_keys = vault.store.scan("_vk:");
        for key in &all_keys {
            assert!(!key.contains("api_key"), "Key should be obfuscated");
        }
        // But we should have at least one vault entry
        assert_eq!(all_keys.len(), 1);
    }

    #[test]
    fn test_pointer_indirection_creates_blob() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "my_secret", "value").unwrap();

        // Check that blob storage was created with _vs: prefix
        let blob_keys = vault.store.scan("_vs:");
        assert_eq!(blob_keys.len(), 1, "Should have one blob");

        // The blob should contain encrypted data
        let blob = vault.store.get(&blob_keys[0]).unwrap();
        assert!(blob.get("_data").is_some(), "Blob should have _data field");
    }

    #[test]
    fn test_metadata_is_obfuscated() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Get the metadata tensor
        let all_keys = vault.store.scan("_vk:");
        let tensor = vault.store.get(&all_keys[0]).unwrap();

        // Creator should be obfuscated, not plaintext
        assert!(
            tensor.get("_creator_obf").is_some(),
            "Should have obfuscated creator"
        );
        assert!(
            tensor.get("_created_by").is_none(),
            "Should NOT have plaintext creator"
        );

        // Timestamp should be obfuscated
        assert!(
            tensor.get("_created_obf").is_some(),
            "Should have obfuscated timestamp"
        );
        assert!(
            tensor.get("_created_at").is_none(),
            "Should NOT have plaintext timestamp"
        );
    }

    #[test]
    fn test_encrypted_key_name_stored_for_listing() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "list_test_key", "value").unwrap();

        // The encrypted key name should be stored for list() to work
        let all_keys = vault.store.scan("_vk:");
        let tensor = vault.store.get(&all_keys[0]).unwrap();

        assert!(
            tensor.get("_key_enc").is_some(),
            "Should have encrypted key"
        );
        assert!(tensor.get("_key_nonce").is_some(), "Should have key nonce");

        // Verify list() can still find it
        let found = vault.list(Vault::ROOT, "list_test_*").unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], "list_test_key");
    }

    #[test]
    fn test_delete_removes_blob() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "temp", "data").unwrap();

        // Verify blob exists
        let blobs_before = vault.store.scan("_vs:");
        assert_eq!(blobs_before.len(), 1);

        vault.delete(Vault::ROOT, "temp").unwrap();

        // Blob should be deleted
        let blobs_after = vault.store.scan("_vs:");
        assert_eq!(blobs_after.len(), 0, "Blob should be deleted");
    }

    #[test]
    fn test_rotate_creates_new_version() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "rotatable", "old_value").unwrap();

        let blobs_before = vault.store.scan("_vs:");
        assert_eq!(blobs_before.len(), 1);

        vault.rotate(Vault::ROOT, "rotatable", "new_value").unwrap();

        // Old blob kept as version, new blob created
        let blobs_after = vault.store.scan("_vs:");
        assert_eq!(blobs_after.len(), 2, "Should have 2 versions");

        // Value should be updated
        let value = vault.get(Vault::ROOT, "rotatable").unwrap();
        assert_eq!(value, "new_value");

        // Old value should still be accessible via version 1
        let old_value = vault.get_version(Vault::ROOT, "rotatable", 1).unwrap();
        assert_eq!(old_value, "old_value");
    }

    #[test]
    fn test_padding_hides_length() {
        // Short values should be padded to same size
        let short1 = obfuscation::pad_plaintext(b"a");
        let short2 = obfuscation::pad_plaintext(b"abc");
        let short3 = obfuscation::pad_plaintext(b"hello world!");

        // All short values pad to Small (256 bytes)
        assert_eq!(short1.len(), 256);
        assert_eq!(short2.len(), 256);
        assert_eq!(short3.len(), 256);

        // Unpad should recover original
        assert_eq!(obfuscation::unpad_plaintext(&short1).unwrap(), b"a");
        assert_eq!(obfuscation::unpad_plaintext(&short2).unwrap(), b"abc");
    }

    // === Permission Level Tests ===

    #[test]
    fn test_permission_enum_allows() {
        // Read allows only Read
        assert!(Permission::Read.allows(Permission::Read));
        assert!(!Permission::Read.allows(Permission::Write));
        assert!(!Permission::Read.allows(Permission::Admin));

        // Write allows Read and Write
        assert!(Permission::Write.allows(Permission::Read));
        assert!(Permission::Write.allows(Permission::Write));
        assert!(!Permission::Write.allows(Permission::Admin));

        // Admin allows all
        assert!(Permission::Admin.allows(Permission::Read));
        assert!(Permission::Admin.allows(Permission::Write));
        assert!(Permission::Admin.allows(Permission::Admin));
    }

    #[test]
    fn test_permission_from_str() {
        assert_eq!("READ".parse::<Permission>().unwrap(), Permission::Read);
        assert_eq!("read".parse::<Permission>().unwrap(), Permission::Read);
        assert_eq!("R".parse::<Permission>().unwrap(), Permission::Read);

        assert_eq!("WRITE".parse::<Permission>().unwrap(), Permission::Write);
        assert_eq!("W".parse::<Permission>().unwrap(), Permission::Write);

        assert_eq!("ADMIN".parse::<Permission>().unwrap(), Permission::Admin);
        assert_eq!("A".parse::<Permission>().unwrap(), Permission::Admin);

        assert!("INVALID".parse::<Permission>().is_err());
    }

    #[test]
    fn test_permission_display() {
        assert_eq!(format!("{}", Permission::Read), "READ");
        assert_eq!(format!("{}", Permission::Write), "WRITE");
        assert_eq!(format!("{}", Permission::Admin), "ADMIN");
    }

    #[test]
    fn test_grant_with_read_permission() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:reader", "secret", Permission::Read)
            .unwrap();

        // Reader can get
        assert!(vault.get("user:reader", "secret").is_ok());

        // Reader can list
        assert!(vault.list("user:reader", "*").is_ok());

        // Reader cannot set (requires Write)
        let result = vault.set("user:reader", "secret", "new_value");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));

        // Reader cannot delete (requires Admin)
        let result = vault.delete("user:reader", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_grant_with_write_permission() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:writer", "secret", Permission::Write)
            .unwrap();

        // Writer can get
        assert!(vault.get("user:writer", "secret").is_ok());

        // Writer can set
        vault.set("user:writer", "secret", "updated").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "updated");

        // Writer can rotate
        vault.rotate("user:writer", "secret", "rotated").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "rotated");

        // Writer cannot delete (requires Admin)
        let result = vault.delete("user:writer", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));

        // Writer cannot grant (requires Admin)
        let result = vault.grant("user:writer", "user:other", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_grant_with_admin_permission() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:admin", "secret", Permission::Admin)
            .unwrap();

        // Admin can get
        assert!(vault.get("user:admin", "secret").is_ok());

        // Admin can set
        vault.set("user:admin", "secret", "admin_update").unwrap();

        // Admin can grant
        vault
            .grant_with_permission("user:admin", "user:reader", "secret", Permission::Read)
            .unwrap();
        assert!(vault.get("user:reader", "secret").is_ok());

        // Admin can revoke
        vault.revoke("user:admin", "user:reader", "secret").unwrap();
        assert!(vault.get("user:reader", "secret").is_err());

        // Admin can delete
        vault.delete("user:admin", "secret").unwrap();
        assert!(vault.get("user:admin", "secret").is_err());
    }

    #[test]
    fn test_get_permission() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Root always has Admin
        assert_eq!(
            vault.get_permission(Vault::ROOT, "secret"),
            Some(Permission::Admin)
        );

        // Grant different levels
        vault
            .grant_with_permission(Vault::ROOT, "user:reader", "secret", Permission::Read)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:writer", "secret", Permission::Write)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:admin", "secret", Permission::Admin)
            .unwrap();

        assert_eq!(
            vault.get_permission("user:reader", "secret"),
            Some(Permission::Read)
        );
        assert_eq!(
            vault.get_permission("user:writer", "secret"),
            Some(Permission::Write)
        );
        assert_eq!(
            vault.get_permission("user:admin", "secret"),
            Some(Permission::Admin)
        );

        // No access
        assert_eq!(vault.get_permission("user:nobody", "secret"), None);
    }

    #[test]
    fn test_insufficient_permission_error_display() {
        let err = VaultError::InsufficientPermission("test message".to_string());
        assert_eq!(err.to_string(), "insufficient permission: test message");
    }

    #[test]
    fn test_read_user_cannot_rotate() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:reader", "secret", Permission::Read)
            .unwrap();

        let result = vault.rotate("user:reader", "secret", "new_value");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_write_user_cannot_revoke() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:writer", "secret", Permission::Write)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:other", "secret", Permission::Read)
            .unwrap();

        // Writer cannot revoke other's access
        let result = vault.revoke("user:writer", "user:other", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_transitive_permission_minimum() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Create chain: alice -> team (MEMBER) -> secret (VAULT_ACCESS_READ)
        vault
            .graph
            .add_entity_edge("user:alice", "team:devs", "MEMBER")
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Read)
            .unwrap();

        // Alice gets Read through the chain
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Read)
        );

        // Alice can get but not set
        assert!(vault.get("user:alice", "secret").is_ok());
        let result = vault.set("user:alice", "secret", "hack");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_backward_compat_old_vault_access_edge() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Manually create old-style VAULT_ACCESS edge (simulating pre-upgrade data)
        vault
            .graph
            .add_entity_edge("user:legacy", "vault_secret:secret", "VAULT_ACCESS")
            .unwrap();

        // Old-style edge should grant Admin access
        assert_eq!(
            vault.get_permission("user:legacy", "secret"),
            Some(Permission::Admin)
        );

        // Legacy user can do Admin operations
        assert!(vault.get("user:legacy", "secret").is_ok());
        assert!(vault.set("user:legacy", "secret", "updated").is_ok());
        assert!(vault.grant("user:legacy", "user:other", "secret").is_ok());
    }

    // === TTL Grant Tests ===

    #[test]
    fn test_grant_with_ttl_creates_grant() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        // Alice should have access
        assert!(vault.get("user:alice", "secret").is_ok());
    }

    #[test]
    fn test_grant_with_ttl_expires() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Grant with 0 TTL (immediately expired)
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        // Small delay to ensure expiration
        std::thread::sleep(Duration::from_millis(10));

        // Next get() call should trigger cleanup
        // Alice's grant is expired, so access should be denied
        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_cleanup_expired_grants() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Grant with 0 TTL (immediately expired)
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        // Small delay
        std::thread::sleep(Duration::from_millis(10));

        // Explicit cleanup
        let revoked = vault.cleanup_expired_grants();
        assert_eq!(revoked, 1);

        // Alice should no longer have access
        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_revoke_removes_from_ttl_tracker() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Grant with TTL
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        // Revoke manually
        vault.revoke(Vault::ROOT, "user:alice", "secret").unwrap();

        // Cleanup should find nothing (already revoked)
        let revoked = vault.cleanup_expired_grants();
        assert_eq!(revoked, 0);
    }

    #[test]
    fn test_multiple_ttl_grants() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Grant alice with 0 TTL (expired)
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        // Grant bob with long TTL (not expired)
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:bob",
                "secret",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));

        // Cleanup should only revoke alice
        let revoked = vault.cleanup_expired_grants();
        assert_eq!(revoked, 1);

        // Alice denied, Bob allowed
        assert!(matches!(
            vault.get("user:alice", "secret"),
            Err(VaultError::AccessDenied(_))
        ));
        assert!(vault.get("user:bob", "secret").is_ok());
    }

    // === Rate Limiting Tests ===

    fn create_rate_limited_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_rate_limit(RateLimitConfig {
            max_gets: 3,
            max_lists: 2,
            max_sets: 2,
            max_grants: 2,
            window: Duration::from_secs(60),
        });
        Vault::new(b"test_password", graph, store, config).unwrap()
    }

    #[test]
    fn test_rate_limit_get() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // First 3 gets should succeed
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.get("user:alice", "secret").is_ok());

        // 4th should be rate limited
        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_list() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // First 2 lists should succeed
        assert!(vault.list("user:alice", "*").is_ok());
        assert!(vault.list("user:alice", "*").is_ok());

        // 3rd should be rate limited
        let result = vault.list("user:alice", "*");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_set() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:writer", "secret", Permission::Write)
            .unwrap();

        // First 2 sets should succeed
        assert!(vault.set("user:writer", "secret", "v1").is_ok());
        assert!(vault.set("user:writer", "secret", "v2").is_ok());

        // 3rd should be rate limited
        let result = vault.set("user:writer", "secret", "v3");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_grant() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:admin", "secret").unwrap();

        // First 2 grants should succeed
        assert!(vault.grant("user:admin", "user:a", "secret").is_ok());
        assert!(vault.grant("user:admin", "user:b", "secret").is_ok());

        // 3rd should be rate limited
        let result = vault.grant("user:admin", "user:c", "secret");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_root_exempt() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Root is exempt from rate limiting
        for _ in 0..10 {
            assert!(vault.get(Vault::ROOT, "secret").is_ok());
        }
    }

    #[test]
    fn test_rate_limit_separate_entities() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        vault.grant(Vault::ROOT, "user:bob", "secret").unwrap();

        // Alice uses her quota
        for _ in 0..3 {
            assert!(vault.get("user:alice", "secret").is_ok());
        }
        assert!(vault.get("user:alice", "secret").is_err());

        // Bob still has his quota
        for _ in 0..3 {
            assert!(vault.get("user:bob", "secret").is_ok());
        }
        assert!(vault.get("user:bob", "secret").is_err());
    }

    #[test]
    fn test_rate_limited_error_display() {
        let err = VaultError::RateLimited("test message".to_string());
        assert_eq!(err.to_string(), "rate limited: test message");
    }

    #[test]
    fn test_vault_without_rate_limit() {
        let vault = create_test_vault(); // Uses default config with no rate limit

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // No rate limiting - should work for many calls
        for _ in 0..100 {
            assert!(vault.get("user:alice", "secret").is_ok());
        }
    }

    // === Namespace Isolation Tests ===

    #[test]
    fn test_namespaced_vault_set_get() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("api_key", "secret123").unwrap();
        let value = ns.get("api_key").unwrap();

        assert_eq!(value, "secret123");
    }

    #[test]
    fn test_namespaced_vault_key_prefixing() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("api_key", "secret123").unwrap();

        // The actual key in the vault should be prefixed
        // Root can access via full prefixed key
        let value = vault.get(Vault::ROOT, "team:alpha:api_key").unwrap();
        assert_eq!(value, "secret123");

        // But the short key should NOT work at root level
        let result = vault.get(Vault::ROOT, "api_key");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_namespaced_vault_isolation() {
        let vault = create_test_vault();

        // Two namespaces, same key name
        let ns_alpha = vault.namespace("team:alpha", Vault::ROOT);
        let ns_beta = vault.namespace("team:beta", Vault::ROOT);

        ns_alpha.set("api_key", "alpha_secret").unwrap();
        ns_beta.set("api_key", "beta_secret").unwrap();

        // Each namespace sees its own value
        assert_eq!(ns_alpha.get("api_key").unwrap(), "alpha_secret");
        assert_eq!(ns_beta.get("api_key").unwrap(), "beta_secret");

        // Root sees both with full keys
        assert_eq!(
            vault.get(Vault::ROOT, "team:alpha:api_key").unwrap(),
            "alpha_secret"
        );
        assert_eq!(
            vault.get(Vault::ROOT, "team:beta:api_key").unwrap(),
            "beta_secret"
        );
    }

    #[test]
    fn test_namespaced_vault_list() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("key1", "v1").unwrap();
        ns.set("key2", "v2").unwrap();

        // Set a key outside the namespace
        vault.set(Vault::ROOT, "other_key", "other").unwrap();

        // List within namespace only returns namespace keys (without prefix)
        let keys = ns.list("*").unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"key1".to_string()));
        assert!(keys.contains(&"key2".to_string()));
        assert!(!keys.contains(&"other_key".to_string()));
    }

    #[test]
    fn test_namespaced_vault_list_pattern() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("openai:key", "sk-1").unwrap();
        ns.set("openai:org", "org-1").unwrap();
        ns.set("github:token", "ghp-1").unwrap();

        // Pattern matching within namespace
        let openai_keys = ns.list("openai:*").unwrap();
        assert_eq!(openai_keys.len(), 2);
        assert!(openai_keys.contains(&"openai:key".to_string()));
        assert!(openai_keys.contains(&"openai:org".to_string()));
    }

    #[test]
    fn test_namespaced_vault_delete() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("api_key", "secret").unwrap();
        ns.delete("api_key").unwrap();

        let result = ns.get("api_key");
        assert!(matches!(result, Err(VaultError::NotFound(_))));

        // Also gone at vault level
        let result = vault.get(Vault::ROOT, "team:alpha:api_key");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_namespaced_vault_rotate() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", Vault::ROOT);

        ns.set("api_key", "old_value").unwrap();
        ns.rotate("api_key", "new_value").unwrap();

        assert_eq!(ns.get("api_key").unwrap(), "new_value");
    }

    #[test]
    fn test_namespaced_vault_grant_revoke() {
        let vault = create_test_vault();

        // Root creates a namespaced secret
        let ns_root = vault.namespace("team:alpha", Vault::ROOT);
        ns_root.set("api_key", "secret").unwrap();

        // Grant alice Read access to the namespaced secret
        ns_root
            .grant("user:alice", "api_key", Permission::Read)
            .unwrap();

        // Alice can access via namespace view
        let ns_alice = vault.namespace("team:alpha", "user:alice");
        assert_eq!(ns_alice.get("api_key").unwrap(), "secret");

        // Revoke access
        ns_root.revoke("user:alice", "api_key").unwrap();

        // Alice can no longer access
        let result = ns_alice.get("api_key");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_namespaced_vault_permission_enforcement() {
        let vault = create_test_vault();

        // Root creates a namespaced secret
        let ns_root = vault.namespace("team:alpha", Vault::ROOT);
        ns_root.set("api_key", "secret").unwrap();

        // Grant alice Read-only access
        ns_root
            .grant("user:alice", "api_key", Permission::Read)
            .unwrap();

        // Alice can read via namespace
        let ns_alice = vault.namespace("team:alpha", "user:alice");
        assert!(ns_alice.get("api_key").is_ok());

        // Alice cannot write via namespace
        let result = ns_alice.set("api_key", "hacked");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_namespaced_vault_accessors() {
        let vault = create_test_vault();
        let ns = vault.namespace("team:alpha", "user:alice");

        assert_eq!(ns.namespace(), "team:alpha");
        assert_eq!(ns.identity(), "user:alice");
    }

    #[test]
    fn test_namespaced_vault_cross_namespace_blocked() {
        let vault = create_test_vault();

        // Alice creates in namespace alpha
        let ns_alpha = vault.namespace("team:alpha", Vault::ROOT);
        ns_alpha.set("secret", "alpha_data").unwrap();

        // Grant alice access to alpha:secret
        ns_alpha
            .grant("user:alice", "secret", Permission::Read)
            .unwrap();

        // Alice tries to access via beta namespace - should fail
        let ns_beta = vault.namespace("team:beta", "user:alice");
        let result = ns_beta.get("secret");
        assert!(matches!(
            result,
            Err(VaultError::NotFound(_) | VaultError::AccessDenied(_))
        ));
    }

    #[test]
    fn test_namespace_helper_method() {
        let vault = create_test_vault();

        // Use the helper method
        let ns = vault.namespace("myns", Vault::ROOT);
        ns.set("key", "value").unwrap();

        assert_eq!(ns.get("key").unwrap(), "value");
    }

    // === Audit Query API Tests ===

    #[test]
    fn test_audit_log_records_get() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Clear any initial audit entries by checking before
        let before = vault.audit_log("secret").len();

        vault.get("user:alice", "secret").unwrap();

        let entries = vault.audit_log("secret");
        assert!(
            entries.len() > before,
            "Should have new audit entries after get"
        );

        // Find the get entry
        let get_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Get))
            .collect();
        assert!(!get_entries.is_empty(), "Should have Get audit entry");
    }

    #[test]
    fn test_audit_log_records_set() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "new_secret", "value").unwrap();

        let entries = vault.audit_log("new_secret");
        let set_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Set))
            .collect();
        assert!(!set_entries.is_empty(), "Should have Set audit entry");
    }

    #[test]
    fn test_audit_log_records_grant() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Read)
            .unwrap();

        let entries = vault.audit_log("secret");
        let grant_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Grant { .. }))
            .collect();
        assert!(!grant_entries.is_empty(), "Should have Grant audit entry");

        // Check grant details
        if let AuditOperation::Grant { to, permission } = &grant_entries[0].operation {
            assert_eq!(to, "user:alice");
            assert_eq!(permission, "read");
        }
    }

    #[test]
    fn test_audit_log_records_revoke() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        vault.revoke(Vault::ROOT, "user:alice", "secret").unwrap();

        let entries = vault.audit_log("secret");
        let revoke_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Revoke { .. }))
            .collect();
        assert!(!revoke_entries.is_empty(), "Should have Revoke audit entry");

        if let AuditOperation::Revoke { from } = &revoke_entries[0].operation {
            assert_eq!(from, "user:alice");
        }
    }

    #[test]
    fn test_audit_log_records_rotate() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "old").unwrap();
        vault.rotate(Vault::ROOT, "secret", "new").unwrap();

        let entries = vault.audit_log("secret");
        let rotate_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Rotate))
            .collect();
        assert!(!rotate_entries.is_empty(), "Should have Rotate audit entry");
    }

    #[test]
    fn test_audit_log_records_delete() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.delete(Vault::ROOT, "secret").unwrap();

        let entries = vault.audit_log("secret");
        let delete_entries: Vec<_> = entries
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::Delete))
            .collect();
        assert!(!delete_entries.is_empty(), "Should have Delete audit entry");
    }

    #[test]
    fn test_audit_by_entity() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "s1").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "s2").unwrap();

        vault.get("user:alice", "s1").unwrap();
        vault.get("user:alice", "s2").unwrap();

        let entries = vault.audit_by_entity("user:alice");
        assert!(
            entries.len() >= 2,
            "Should have at least 2 entries for alice"
        );
    }

    #[test]
    fn test_audit_recent() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();
        vault.set(Vault::ROOT, "s3", "v3").unwrap();

        let recent = vault.audit_recent(2);
        assert_eq!(recent.len(), 2, "Should return at most 2 entries");

        // Entries should be sorted by timestamp (most recent first)
        if recent.len() >= 2 {
            assert!(recent[0].timestamp >= recent[1].timestamp);
        }
    }

    #[test]
    fn test_audit_since() {
        let vault = create_test_vault();

        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        std::thread::sleep(std::time::Duration::from_millis(10));

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let entries = vault.audit_since(before);
        assert!(
            !entries.is_empty(),
            "Should have entries since the timestamp"
        );
    }

    // === Secret Versioning Tests ===

    #[test]
    fn test_version_on_set() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.set(Vault::ROOT, "secret", "v2").unwrap();

        // Current value should be v2
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "v2");

        // Should have 2 versions
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 2);
    }

    #[test]
    fn test_get_specific_version() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "version_1").unwrap();
        vault.set(Vault::ROOT, "secret", "version_2").unwrap();
        vault.set(Vault::ROOT, "secret", "version_3").unwrap();

        // Get each version
        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 1).unwrap(),
            "version_1"
        );
        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 2).unwrap(),
            "version_2"
        );
        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 3).unwrap(),
            "version_3"
        );
    }

    #[test]
    fn test_get_version_not_found() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();

        // Version 2 doesn't exist
        let result = vault.get_version(Vault::ROOT, "secret", 2);
        assert!(matches!(result, Err(VaultError::NotFound(_))));

        // Version 100 doesn't exist
        let result = vault.get_version(Vault::ROOT, "secret", 100);
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_list_versions() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        vault.set(Vault::ROOT, "secret", "v2").unwrap();

        let versions = vault.list_versions(Vault::ROOT, "secret").unwrap();
        assert_eq!(versions.len(), 2);

        assert_eq!(versions[0].version, 1);
        assert_eq!(versions[1].version, 2);

        // Version 2 should have a later timestamp
        assert!(versions[1].created_at >= versions[0].created_at);
    }

    #[test]
    fn test_rollback() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "original").unwrap();
        vault.set(Vault::ROOT, "secret", "modified").unwrap();

        // Current should be modified
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "modified");

        // Rollback to version 1
        vault.rollback(Vault::ROOT, "secret", 1).unwrap();

        // Current should now be original (but as a new version)
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "original");

        // Should now have 3 versions
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);
    }

    #[test]
    fn test_max_versions_pruning() {
        // Create vault with max 3 versions
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_max_versions(3);
        let vault = Vault::new(b"test", graph, store, config).unwrap();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.set(Vault::ROOT, "secret", "v2").unwrap();
        vault.set(Vault::ROOT, "secret", "v3").unwrap();
        vault.set(Vault::ROOT, "secret", "v4").unwrap();

        // Should only have 3 versions (v2, v3, v4)
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);

        // Version 1 should now be v2 (v1 was pruned)
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 1).unwrap(), "v2");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 2).unwrap(), "v3");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 3).unwrap(), "v4");
    }

    #[test]
    fn test_rotate_keeps_versions() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "initial").unwrap();
        vault.rotate(Vault::ROOT, "secret", "rotated").unwrap();

        // Should have 2 versions
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 2);

        // Can get both versions
        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 1).unwrap(),
            "initial"
        );
        assert_eq!(
            vault.get_version(Vault::ROOT, "secret", 2).unwrap(),
            "rotated"
        );
    }

    #[test]
    fn test_delete_removes_all_versions() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.set(Vault::ROOT, "secret", "v2").unwrap();
        vault.set(Vault::ROOT, "secret", "v3").unwrap();

        let blobs_before = vault.store.scan("_vs:");
        assert_eq!(blobs_before.len(), 3);

        vault.delete(Vault::ROOT, "secret").unwrap();

        // All blobs should be deleted
        let blobs_after = vault.store.scan("_vs:");
        assert_eq!(blobs_after.len(), 0);
    }

    #[test]
    fn test_version_access_control() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Alice has no access
        let result = vault.get_version("user:alice", "secret", 1);
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        // Grant access to alice
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Now alice can access
        assert_eq!(
            vault.get_version("user:alice", "secret", 1).unwrap(),
            "value"
        );
    }

    #[test]
    fn test_with_max_versions_config() {
        let config = VaultConfig::default().with_max_versions(10);
        assert_eq!(config.max_versions, 10);
    }

    #[test]
    fn test_version_info_timestamps() {
        let vault = create_test_vault();

        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let versions = vault.list_versions(Vault::ROOT, "secret").unwrap();
        assert_eq!(versions.len(), 1);
        assert!(versions[0].created_at >= before);
    }
}
