//! Tensor Vault: Secure secret storage with graph-based access control.
//!
//! Secrets are encrypted at rest using AES-256-GCM. Access is controlled
//! by graph topology - a requester must have a path to the secret node.

mod access;
mod encryption;
mod key;

pub use access::AccessController;
pub use encryption::{Cipher, NONCE_SIZE};
pub use key::{MasterKey, KEY_SIZE};

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
    graph: Arc<GraphEngine>,
    cipher: Cipher,
}

impl Vault {
    /// Storage key prefix for vault secrets.
    const PREFIX: &'static str = "_vault:";
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
        let cipher = Cipher::new(derived);

        let vault = Self {
            store,
            graph,
            cipher,
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

    fn vault_key(secret_key: &str) -> String {
        format!("{}{}", Self::PREFIX, secret_key)
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

        let (ciphertext, nonce) = self.cipher.encrypt(value.as_bytes())?;

        let mut tensor = TensorData::new();
        tensor.set(
            "_ciphertext",
            TensorValue::Scalar(ScalarValue::Bytes(ciphertext)),
        );
        tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
        );
        tensor.set(
            "_created_by",
            TensorValue::Scalar(ScalarValue::String(requester.to_string())),
        );
        tensor.set(
            "_created_at",
            TensorValue::Scalar(ScalarValue::Int(Self::current_timestamp())),
        );

        self.store
            .put(Self::vault_key(key), tensor)
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
            .get(&Self::vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        let ciphertext = match tensor.get("_ciphertext") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return Err(VaultError::NotFound(key.to_string())),
        };

        let nonce = match tensor.get("_nonce") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return Err(VaultError::CryptoError("Missing nonce".to_string())),
        };

        let plaintext = self.cipher.decrypt(&ciphertext, &nonce)?;

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

        self.store
            .delete(&Self::vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        // Also clean up the secret node
        let secret_node = Self::secret_node_key(key);
        let _ = self.store.delete(&secret_node);

        Ok(())
    }

    /// Rotate a secret value.
    pub fn rotate(&self, requester: &str, key: &str, new_value: &str) -> Result<()> {
        self.check_access(requester, key)?;

        if !self.store.exists(&Self::vault_key(key)) {
            return Err(VaultError::NotFound(key.to_string()));
        }

        let (ciphertext, nonce) = self.cipher.encrypt(new_value.as_bytes())?;

        let mut tensor = self
            .store
            .get(&Self::vault_key(key))
            .map_err(|_| VaultError::NotFound(key.to_string()))?;

        tensor.set(
            "_ciphertext",
            TensorValue::Scalar(ScalarValue::Bytes(ciphertext)),
        );
        tensor.set(
            "_nonce",
            TensorValue::Scalar(ScalarValue::Bytes(nonce.to_vec())),
        );
        tensor.set(
            "_rotated_by",
            TensorValue::Scalar(ScalarValue::String(requester.to_string())),
        );
        tensor.set(
            "_rotated_at",
            TensorValue::Scalar(ScalarValue::Int(Self::current_timestamp())),
        );

        self.store
            .put(Self::vault_key(key), tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// List secret keys matching a pattern.
    pub fn list(&self, requester: &str, pattern: &str) -> Result<Vec<String>> {
        let all_keys = self.store.scan(Self::PREFIX);

        let mut accessible = Vec::new();
        for vault_key in all_keys {
            let key = vault_key.strip_prefix(Self::PREFIX).unwrap_or(&vault_key);

            if Self::matches_pattern(key, pattern) && self.has_access(requester, key) {
                accessible.push(key.to_string());
            }
        }

        Ok(accessible)
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
}
