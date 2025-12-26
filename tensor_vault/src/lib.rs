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
mod encryption;
mod key;
mod obfuscation;

pub use access::AccessController;
pub use encryption::{Cipher, NONCE_SIZE};
pub use key::{MasterKey, KEY_SIZE};
pub use obfuscation::{Obfuscator, PaddingSize};

use graph_engine::GraphEngine;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

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
}

impl Default for VaultConfig {
    fn default() -> Self {
        Self {
            salt: None,
            argon2_memory_cost: 65536,
            argon2_time_cost: 3,
            argon2_parallelism: 4,
        }
    }
}

/// Secure secret storage with graph-based access control.
pub struct Vault {
    store: TensorStore,
    pub graph: Arc<GraphEngine>,
    cipher: Cipher,
    obfuscator: Obfuscator,
}

impl Vault {
    /// Storage key prefix for vault secrets (obfuscated).
    const PREFIX: &'static str = "_vk:";
    /// Node ID for the root entity with universal access.
    pub const ROOT: &'static str = "node:root";
    /// Edge type for access grants.
    const ACCESS_EDGE: &'static str = "VAULT_ACCESS";
    /// Edge type for audit log entries.
    const AUDIT_EDGE: &'static str = "ACCESSED";

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

        let vault = Self {
            store,
            graph,
            cipher,
            obfuscator,
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
    pub fn set(&self, requester: &str, key: &str, value: &str) -> Result<()> {
        // For new secrets, only root can create. For existing, need access.
        let secret_node = Self::secret_node_key(key);
        if self.store.exists(&secret_node) {
            self.check_access(requester, key)?;
        } else if requester != Self::ROOT {
            return Err(VaultError::AccessDenied(
                "only root can create new secrets".to_string(),
            ));
        }

        // Pad plaintext to hide length, then encrypt
        let padded = obfuscation::pad_plaintext(value.as_bytes());
        let (ciphertext, nonce) = self.cipher.encrypt(&padded)?;

        // Store ciphertext in separate blob for pointer indirection
        let blob_storage_key = self.blob_key(key, &nonce);
        let mut blob_tensor = TensorData::new();
        blob_tensor.set("_data", TensorValue::Scalar(ScalarValue::Bytes(ciphertext)));
        self.store
            .put(&blob_storage_key, blob_tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        // Encrypt the original key name for list() to work
        let (encrypted_key, key_nonce) = self.cipher.encrypt(key.as_bytes())?;

        // Obfuscate metadata
        let creator_bytes = requester.as_bytes();
        let obfuscated_creator = self.obfuscator.obfuscate_metadata(creator_bytes);
        let timestamp = Self::current_timestamp();
        let timestamp_bytes = timestamp.to_le_bytes();
        let obfuscated_timestamp = self.obfuscator.obfuscate_metadata(&timestamp_bytes);

        // Create metadata tensor with pointer to blob
        let mut tensor = TensorData::new();
        tensor.set("_blob", TensorValue::Pointer(blob_storage_key));
        tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
        );
        tensor.set(
            "_key_enc",
            TensorValue::Scalar(ScalarValue::Bytes(encrypted_key)),
        );
        tensor.set(
            "_key_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(key_nonce.to_vec())),
        );
        tensor.set(
            "_creator_obf",
            TensorValue::Scalar(ScalarValue::Bytes(obfuscated_creator)),
        );
        tensor.set(
            "_created_obf",
            TensorValue::Scalar(ScalarValue::Bytes(obfuscated_timestamp)),
        );

        self.store
            .put(self.vault_key(key), tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        // Create secret node for access control if it doesn't exist
        if !self.store.exists(&secret_node) {
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

            // Root always has access to secrets it creates
            self.graph
                .add_entity_edge(Self::ROOT, &secret_node, Self::ACCESS_EDGE)
                .map_err(|e| VaultError::GraphError(e.to_string()))?;
        }

        Ok(())
    }

    /// Retrieve a secret (requires graph path from requester to secret).
    pub fn get(&self, requester: &str, key: &str) -> Result<String> {
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

        // Log access (create audit edge)
        self.log_access(requester, key);

        String::from_utf8(plaintext)
            .map_err(|e| VaultError::CryptoError(format!("Invalid UTF-8: {e}")))
    }

    /// Grant access to a secret (create edge from entity to secret node).
    pub fn grant(&self, requester: &str, entity: &str, key: &str) -> Result<()> {
        // Only those with access can grant access
        self.check_access(requester, key)?;

        let secret_node = Self::secret_node_key(key);
        if !self.store.exists(&secret_node) {
            return Err(VaultError::NotFound(key.to_string()));
        }

        self.graph
            .add_entity_edge(entity, &secret_node, Self::ACCESS_EDGE)
            .map_err(|e| VaultError::GraphError(e.to_string()))?;

        Ok(())
    }

    /// Revoke access to a secret (delete edge).
    pub fn revoke(&self, requester: &str, entity: &str, key: &str) -> Result<()> {
        // Only those with access can revoke access
        self.check_access(requester, key)?;

        let secret_node = Self::secret_node_key(key);

        // Find and delete the access edge
        if let Ok(edges) = self.graph.get_entity_outgoing(entity) {
            for edge_key in edges {
                if let Ok((_, to, edge_type, _)) = self.graph.get_entity_edge(&edge_key) {
                    if to == secret_node && edge_type == Self::ACCESS_EDGE {
                        self.graph
                            .delete_entity_edge(&edge_key)
                            .map_err(|e| VaultError::GraphError(e.to_string()))?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Delete a secret.
    pub fn delete(&self, requester: &str, key: &str) -> Result<()> {
        self.check_access(requester, key)?;

        let vault_storage_key = self.vault_key(key);

        // Get the blob pointer before deleting metadata
        if let Ok(tensor) = self.store.get(&vault_storage_key) {
            if let Some(TensorValue::Pointer(blob_key)) = tensor.get("_blob") {
                // Delete the ciphertext blob
                let _ = self.store.delete(blob_key);
            }
        }

        self.store
            .delete(&vault_storage_key)
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Also clean up the secret node
        let secret_node = Self::secret_node_key(key);
        let _ = self.store.delete(&secret_node);

        Ok(())
    }

    /// Rotate a secret value.
    pub fn rotate(&self, requester: &str, key: &str, new_value: &str) -> Result<()> {
        self.check_access(requester, key)?;

        let vault_storage_key = self.vault_key(key);

        if !self.store.exists(&vault_storage_key) {
            return Err(VaultError::NotFound(key.to_string()));
        }

        // Pad and encrypt new value
        let padded = obfuscation::pad_plaintext(new_value.as_bytes());
        let (ciphertext, nonce) = self.cipher.encrypt(&padded)?;

        // Get current tensor
        let mut tensor = self
            .store
            .get(&vault_storage_key)
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Delete old blob
        if let Some(TensorValue::Pointer(old_blob_key)) = tensor.get("_blob") {
            let _ = self.store.delete(old_blob_key);
        }

        // Store new ciphertext blob
        let new_blob_key = self.blob_key(key, &nonce);
        let mut blob_tensor = TensorData::new();
        blob_tensor.set("_data", TensorValue::Scalar(ScalarValue::Bytes(ciphertext)));
        self.store
            .put(&new_blob_key, blob_tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        // Update metadata with new blob pointer and nonce
        tensor.set("_blob", TensorValue::Pointer(new_blob_key));
        tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
        );

        // Obfuscate rotation metadata
        let rotator_obf = self.obfuscator.obfuscate_metadata(requester.as_bytes());
        let timestamp = Self::current_timestamp();
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

        Ok(())
    }

    /// List secret keys matching a pattern.
    pub fn list(&self, requester: &str, pattern: &str) -> Result<Vec<String>> {
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

    /// Access the underlying graph engine (for testing/benchmarking).
    pub fn graph(&self) -> &Arc<GraphEngine> {
        &self.graph
    }

    fn check_access(&self, requester: &str, key: &str) -> Result<()> {
        if self.has_access(requester, key) {
            Ok(())
        } else {
            Err(VaultError::AccessDenied(format!(
                "No path from {requester} to secret '{key}'"
            )))
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

    fn log_access(&self, requester: &str, key: &str) {
        // Create permanent audit edge: secret -> ACCESSED -> requester
        // Direction is reversed so audit edges don't grant access
        let secret_node = Self::secret_node_key(key);
        let _ = self
            .graph
            .add_entity_edge(&secret_node, requester, Self::AUDIT_EDGE);
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
    fn test_rotate_creates_new_blob() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "rotatable", "old_value").unwrap();

        let blobs_before = vault.store.scan("_vs:");
        assert_eq!(blobs_before.len(), 1);

        vault.rotate(Vault::ROOT, "rotatable", "new_value").unwrap();

        // Old blob deleted, new blob created
        let blobs_after = vault.store.scan("_vs:");
        assert_eq!(blobs_after.len(), 1);

        // But it should be a different blob (different nonce)
        assert_ne!(blobs_before[0], blobs_after[0], "Should be a new blob");

        // Value should be updated
        let value = vault.get(Vault::ROOT, "rotatable").unwrap();
        assert_eq!(value, "new_value");
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
}
