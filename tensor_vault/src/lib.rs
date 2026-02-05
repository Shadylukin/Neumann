// SPDX-License-Identifier: MIT OR Apache-2.0
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
mod attenuation;
mod audit;
mod encryption;
mod key;
pub mod namespaced;
mod obfuscation;
mod rate_limit;
pub mod scoped;
mod signing;
mod ttl;
mod vault;

use serde::{Deserialize, Serialize};

pub use access::AccessController;
pub use attenuation::AttenuationPolicy;
pub use audit::{AuditEntry, AuditLog, AuditOperation};
pub use encryption::{Cipher, NONCE_SIZE};
pub use key::{MasterKey, KEY_SIZE, SALT_SIZE};
pub use obfuscation::{Obfuscator, PaddingSize};
pub use rate_limit::{Operation, RateLimitConfig, RateLimiter};
pub use ttl::GrantTTLTracker;
pub use vault::Vault;

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

    /// Numeric level for bottleneck flow (1=Read, 2=Write, 3=Admin).
    pub(crate) fn to_level(self) -> i64 {
        match self {
            Self::Read => 1,
            Self::Write => 2,
            Self::Admin => 3,
        }
    }

    /// Parse from numeric level.
    pub(crate) fn from_level(level: i64) -> Option<Self> {
        match level {
            1 => Some(Self::Read),
            2 => Some(Self::Write),
            3 => Some(Self::Admin),
            _ => None,
        }
    }

    /// Get edge type suffix for this permission level.
    pub(crate) fn edge_suffix(self) -> &'static str {
        match self {
            Self::Read => "_READ",
            Self::Write => "_WRITE",
            Self::Admin => "_ADMIN",
        }
    }

    /// Parse from edge type.
    pub(crate) fn from_edge_type(edge_type: &str) -> Option<Self> {
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
    /// Permission attenuation policy (default: attenuate with distance).
    pub attenuation: AttenuationPolicy,
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
            attenuation: AttenuationPolicy::default(),
        }
    }
}

impl VaultConfig {
    /// Create a config with an explicit salt.
    #[must_use]
    pub fn with_salt(mut self, salt: [u8; 16]) -> Self {
        self.salt = Some(salt);
        self
    }

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

    /// Create a config with a custom attenuation policy.
    #[must_use]
    pub fn with_attenuation(mut self, policy: AttenuationPolicy) -> Self {
        self.attenuation = policy;
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

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc, time::Duration};

    use graph_engine::{Direction, GraphEngine, PropertyValue};
    use serial_test::serial;
    use tensor_store::TensorStore;

    use super::*;

    fn create_test_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap()
    }

    /// Helper to add edges between entity keys using the node-based API.
    fn add_test_edge(graph: &GraphEngine, from_key: &str, to_key: &str, edge_type: &str) {
        let get_or_create = |key: &str| -> u64 {
            if let Ok(nodes) =
                graph.find_nodes_by_property("entity_key", &PropertyValue::String(key.to_string()))
            {
                if let Some(node) = nodes.first() {
                    return node.id;
                }
            }
            let mut props = HashMap::new();
            props.insert(
                "entity_key".to_string(),
                PropertyValue::String(key.to_string()),
            );
            graph.create_node("TestEntity", props).unwrap_or(0)
        };

        let from_node = get_or_create(from_key);
        let to_node = get_or_create(to_key);
        graph
            .create_edge(from_node, to_node, edge_type, HashMap::new(), true)
            .ok();
    }

    /// Check whether an entity node has any outgoing VAULT_ACCESS edges.
    fn has_access_edges(graph: &GraphEngine, entity_key: &str) -> bool {
        let nodes = graph
            .find_nodes_by_property("entity_key", &PropertyValue::String(entity_key.to_string()))
            .unwrap_or_default();
        let Some(node) = nodes.first() else {
            return false;
        };
        graph
            .edges_of(node.id, Direction::Outgoing)
            .unwrap_or_default()
            .iter()
            .any(|e| e.edge_type.starts_with("VAULT_ACCESS"))
    }

    /// Helper to get outgoing edges for an entity key.
    fn get_test_outgoing_edges(graph: &GraphEngine, entity_key: &str) -> Vec<(String, String)> {
        let find_node = |key: &str| -> Option<u64> {
            graph
                .find_nodes_by_property("entity_key", &PropertyValue::String(key.to_string()))
                .ok()
                .and_then(|nodes| nodes.first().map(|n| n.id))
        };

        let get_key = |id: u64| -> Option<String> {
            graph.get_node(id).ok().and_then(|node| {
                if let Some(PropertyValue::String(key)) = node.properties.get("entity_key") {
                    Some(key.clone())
                } else {
                    None
                }
            })
        };

        let Some(node_id) = find_node(entity_key) else {
            return Vec::new();
        };

        let mut result = Vec::new();
        if let Ok(edges) = graph.edges_of(node_id, Direction::Outgoing) {
            for edge in edges {
                let target_id = if edge.from == node_id {
                    edge.to
                } else {
                    edge.from
                };
                if let Some(target_key) = get_key(target_id) {
                    result.push((target_key, edge.edge_type.clone()));
                }
            }
        }
        result
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
    fn test_list_excludes_member_only_paths() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Alice has a MEMBER-only path to the secret node (no VAULT_ACCESS edge)
        // This should NOT make the secret visible in list()
        let secret_node = vault.secret_node_key("secret");
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        add_test_edge(&vault.graph, "team:devs", &secret_node, "MEMBER");

        let alice_keys = vault.list("user:alice", "*").unwrap();
        assert!(
            alice_keys.is_empty(),
            "MEMBER-only path should not leak secret names in list()"
        );
    }

    #[test]
    fn test_list_includes_vault_access_paths() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Alice has a proper VAULT_ACCESS path via a team
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Read)
            .unwrap();

        let alice_keys = vault.list("user:alice", "*").unwrap();
        assert_eq!(alice_keys.len(), 1);
        assert!(alice_keys.contains(&"secret".to_string()));
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
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault.grant(Vault::ROOT, "team:devs", "secret").unwrap();

        // Alice can access via transitive path
        let value = vault.get("user:alice", "secret").unwrap();
        assert_eq!(value, "value");
    }

    #[test]
    fn test_concurrent_access() {
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

        let scoped = vault.scope(Vault::ROOT);

        scoped.set("scoped:key", "scoped_value").unwrap();
        let value = scoped.get("scoped:key").unwrap();
        assert_eq!(value, "scoped_value");

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
    #[serial]
    fn test_from_env_with_valid_key() {
        use base64::Engine;

        let key_bytes = b"01234567890123456789012345678901";
        let encoded = base64::engine::general_purpose::STANDARD.encode(key_bytes);
        std::env::set_var("NEUMANN_VAULT_KEY", &encoded);

        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let result = Vault::from_env(graph, store);

        std::env::remove_var("NEUMANN_VAULT_KEY");

        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn test_from_env_missing_key() {
        std::env::remove_var("NEUMANN_VAULT_KEY");

        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let result = Vault::from_env(graph, store);

        assert!(matches!(result, Err(VaultError::KeyDerivationError(_))));
    }

    #[test]
    #[serial]
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

        vault.set(Vault::ROOT, "protected", "initial").unwrap();

        let result = vault.set("user:alice", "protected", "hacked");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_overwrite_with_granted_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "shared", "initial").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "shared").unwrap();

        vault.set("user:alice", "shared", "updated").unwrap();

        let value = vault.get(Vault::ROOT, "shared").unwrap();
        assert_eq!(value, "updated");
    }

    #[test]
    fn test_short_key_still_works() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());

        let result = Vault::new(b"short", graph.clone(), store, VaultConfig::default());
        assert!(result.is_ok());

        let store2 = TensorStore::new();
        let result = Vault::new(b"", graph, store2, VaultConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_revoke_on_nonexistent_secret() {
        let vault = create_test_vault();

        let result = vault.revoke("user:alice", "user:bob", "nonexistent");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        let result = vault.revoke(Vault::ROOT, "user:alice", "nonexistent");
        assert!(result.is_ok());
    }

    // === Security feature tests ===

    #[test]
    fn test_storage_key_is_obfuscated() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "api_key", "secret").unwrap();

        let all_keys = vault.store.scan("_vk:");
        for key in &all_keys {
            assert!(!key.contains("api_key"), "Key should be obfuscated");
        }
        assert_eq!(all_keys.len(), 1);
    }

    #[test]
    fn test_pointer_indirection_creates_blob() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "my_secret", "value").unwrap();

        let blob_keys = vault.store.scan("_vs:");
        assert_eq!(blob_keys.len(), 1, "Should have one blob");

        let blob = vault.store.get(&blob_keys[0]).unwrap();
        assert!(blob.get("_data").is_some(), "Blob should have _data field");
    }

    #[test]
    fn test_metadata_is_obfuscated() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let all_keys = vault.store.scan("_vk:");
        let tensor = vault.store.get(&all_keys[0]).unwrap();

        assert!(
            tensor.get("_creator_obf").is_some(),
            "Should have obfuscated creator"
        );
        assert!(
            tensor.get("_created_by").is_none(),
            "Should NOT have plaintext creator"
        );

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

        let all_keys = vault.store.scan("_vk:");
        let tensor = vault.store.get(&all_keys[0]).unwrap();

        assert!(
            tensor.get("_key_enc").is_some(),
            "Should have encrypted key"
        );
        assert!(tensor.get("_key_nonce").is_some(), "Should have key nonce");

        let found = vault.list(Vault::ROOT, "list_test_*").unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0], "list_test_key");
    }

    #[test]
    fn test_delete_removes_blob() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "temp", "data").unwrap();

        let blobs_before = vault.store.scan("_vs:");
        assert_eq!(blobs_before.len(), 1);

        vault.delete(Vault::ROOT, "temp").unwrap();

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

        let blobs_after = vault.store.scan("_vs:");
        assert_eq!(blobs_after.len(), 2, "Should have 2 versions");

        let value = vault.get(Vault::ROOT, "rotatable").unwrap();
        assert_eq!(value, "new_value");

        let old_value = vault.get_version(Vault::ROOT, "rotatable", 1).unwrap();
        assert_eq!(old_value, "old_value");
    }

    #[test]
    fn test_padding_hides_length() {
        let short1 = obfuscation::pad_plaintext(b"a").unwrap();
        let short2 = obfuscation::pad_plaintext(b"abc").unwrap();
        let short3 = obfuscation::pad_plaintext(b"hello world!").unwrap();

        assert_eq!(short1.len(), 256);
        assert_eq!(short2.len(), 256);
        assert_eq!(short3.len(), 256);

        assert_eq!(obfuscation::unpad_plaintext(&short1).unwrap(), b"a");
        assert_eq!(obfuscation::unpad_plaintext(&short2).unwrap(), b"abc");
    }

    // === Permission Level Tests ===

    #[test]
    fn test_permission_enum_allows() {
        assert!(Permission::Read.allows(Permission::Read));
        assert!(!Permission::Read.allows(Permission::Write));
        assert!(!Permission::Read.allows(Permission::Admin));

        assert!(Permission::Write.allows(Permission::Read));
        assert!(Permission::Write.allows(Permission::Write));
        assert!(!Permission::Write.allows(Permission::Admin));

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

        assert!(vault.get("user:reader", "secret").is_ok());
        assert!(vault.list("user:reader", "*").is_ok());

        let result = vault.set("user:reader", "secret", "new_value");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));

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

        assert!(vault.get("user:writer", "secret").is_ok());

        vault.set("user:writer", "secret", "updated").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "updated");

        vault.rotate("user:writer", "secret", "rotated").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "rotated");

        let result = vault.delete("user:writer", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));

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

        assert!(vault.get("user:admin", "secret").is_ok());
        vault.set("user:admin", "secret", "admin_update").unwrap();

        vault
            .grant_with_permission("user:admin", "user:reader", "secret", Permission::Read)
            .unwrap();
        assert!(vault.get("user:reader", "secret").is_ok());

        vault.revoke("user:admin", "user:reader", "secret").unwrap();
        assert!(vault.get("user:reader", "secret").is_err());

        vault.delete("user:admin", "secret").unwrap();
        assert!(vault.get("user:admin", "secret").is_err());
    }

    #[test]
    fn test_get_permission() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        assert_eq!(
            vault.get_permission(Vault::ROOT, "secret"),
            Some(Permission::Admin)
        );

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

        let result = vault.revoke("user:writer", "user:other", "secret");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_transitive_permission_minimum() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Read)
            .unwrap();

        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Read)
        );

        assert!(vault.get("user:alice", "secret").is_ok());
        let result = vault.set("user:alice", "secret", "hack");
        assert!(matches!(result, Err(VaultError::InsufficientPermission(_))));
    }

    #[test]
    fn test_secret_node_key_is_obfuscated() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "my_api_key", "value").unwrap();

        vault
            .grant(Vault::ROOT, "user:alice", "my_api_key")
            .unwrap();

        let edges = get_test_outgoing_edges(&vault.graph, "user:alice");
        assert!(!edges.is_empty(), "Expected grant to create edge");

        for (to, edge_type) in &edges {
            assert!(edge_type.starts_with("VAULT_ACCESS"));

            assert!(
                !to.contains("my_api_key"),
                "Secret name should be obfuscated in graph: found {to}"
            );
            assert!(
                to.starts_with("vault_secret:"),
                "Edge should point to vault secret node: {to}"
            );
        }

        assert!(vault.get("user:alice", "my_api_key").is_ok());
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

        assert!(vault.get("user:alice", "secret").is_ok());
    }

    #[test]
    fn test_grant_with_ttl_expires() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));

        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_cleanup_expired_grants() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));

        let revoked = vault.cleanup_expired_grants();
        assert_eq!(revoked, 1);

        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_revoke_removes_from_ttl_tracker() {
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

        vault.revoke(Vault::ROOT, "user:alice", "secret").unwrap();

        let revoked = vault.cleanup_expired_grants();
        assert_eq!(revoked, 0);
    }

    #[test]
    fn test_multiple_ttl_grants() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

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

        let revoked = vault.cleanup_expired_grants();
        assert_eq!(revoked, 1);

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

        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.get("user:alice", "secret").is_ok());

        let result = vault.get("user:alice", "secret");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_list() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        assert!(vault.list("user:alice", "*").is_ok());
        assert!(vault.list("user:alice", "*").is_ok());

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

        assert!(vault.set("user:writer", "secret", "v1").is_ok());
        assert!(vault.set("user:writer", "secret", "v2").is_ok());

        let result = vault.set("user:writer", "secret", "v3");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_grant() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:admin", "secret").unwrap();

        assert!(vault.grant("user:admin", "user:a", "secret").is_ok());
        assert!(vault.grant("user:admin", "user:b", "secret").is_ok());

        let result = vault.grant("user:admin", "user:c", "secret");
        assert!(matches!(result, Err(VaultError::RateLimited(_))));
    }

    #[test]
    fn test_rate_limit_root_exempt() {
        let vault = create_rate_limited_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

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

        for _ in 0..3 {
            assert!(vault.get("user:alice", "secret").is_ok());
        }
        assert!(vault.get("user:alice", "secret").is_err());

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
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

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

        let value = vault.get(Vault::ROOT, "team:alpha:api_key").unwrap();
        assert_eq!(value, "secret123");

        let result = vault.get(Vault::ROOT, "api_key");
        assert!(matches!(result, Err(VaultError::NotFound(_))));
    }

    #[test]
    fn test_namespaced_vault_isolation() {
        let vault = create_test_vault();

        let ns_alpha = vault.namespace("team:alpha", Vault::ROOT);
        let ns_beta = vault.namespace("team:beta", Vault::ROOT);

        ns_alpha.set("api_key", "alpha_secret").unwrap();
        ns_beta.set("api_key", "beta_secret").unwrap();

        assert_eq!(ns_alpha.get("api_key").unwrap(), "alpha_secret");
        assert_eq!(ns_beta.get("api_key").unwrap(), "beta_secret");

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

        vault.set(Vault::ROOT, "other_key", "other").unwrap();

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

        let ns_root = vault.namespace("team:alpha", Vault::ROOT);
        ns_root.set("api_key", "secret").unwrap();

        ns_root
            .grant("user:alice", "api_key", Permission::Read)
            .unwrap();

        let ns_alice = vault.namespace("team:alpha", "user:alice");
        assert_eq!(ns_alice.get("api_key").unwrap(), "secret");

        ns_root.revoke("user:alice", "api_key").unwrap();

        let result = ns_alice.get("api_key");
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));
    }

    #[test]
    fn test_namespaced_vault_permission_enforcement() {
        let vault = create_test_vault();

        let ns_root = vault.namespace("team:alpha", Vault::ROOT);
        ns_root.set("api_key", "secret").unwrap();

        ns_root
            .grant("user:alice", "api_key", Permission::Read)
            .unwrap();

        let ns_alice = vault.namespace("team:alpha", "user:alice");
        assert!(ns_alice.get("api_key").is_ok());

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

        let ns_alpha = vault.namespace("team:alpha", Vault::ROOT);
        ns_alpha.set("secret", "alpha_data").unwrap();

        ns_alpha
            .grant("user:alice", "secret", Permission::Read)
            .unwrap();

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

        let before = vault.audit_log("secret").len();

        vault.get("user:alice", "secret").unwrap();

        let entries = vault.audit_log("secret");
        assert!(
            entries.len() > before,
            "Should have new audit entries after get"
        );

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

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "v2");
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 2);
    }

    #[test]
    fn test_get_specific_version() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "version_1").unwrap();
        vault.set(Vault::ROOT, "secret", "version_2").unwrap();
        vault.set(Vault::ROOT, "secret", "version_3").unwrap();

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

        let result = vault.get_version(Vault::ROOT, "secret", 2);
        assert!(matches!(result, Err(VaultError::NotFound(_))));

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

        assert!(versions[1].created_at >= versions[0].created_at);
    }

    #[test]
    fn test_rollback() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "original").unwrap();
        vault.set(Vault::ROOT, "secret", "modified").unwrap();

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "modified");

        vault.rollback(Vault::ROOT, "secret", 1).unwrap();

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "original");
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);
    }

    #[test]
    fn test_max_versions_pruning() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_max_versions(3);
        let vault = Vault::new(b"test", graph, store, config).unwrap();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.set(Vault::ROOT, "secret", "v2").unwrap();
        vault.set(Vault::ROOT, "secret", "v3").unwrap();
        vault.set(Vault::ROOT, "secret", "v4").unwrap();

        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);

        assert_eq!(vault.get_version(Vault::ROOT, "secret", 1).unwrap(), "v2");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 2).unwrap(), "v3");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 3).unwrap(), "v4");
    }

    #[test]
    fn test_rotate_keeps_versions() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "initial").unwrap();
        vault.rotate(Vault::ROOT, "secret", "rotated").unwrap();

        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 2);

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

        let blobs_after = vault.store.scan("_vs:");
        assert_eq!(blobs_after.len(), 0);
    }

    #[test]
    fn test_version_access_control() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        let result = vault.get_version("user:alice", "secret", 1);
        assert!(matches!(result, Err(VaultError::AccessDenied(_))));

        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

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

    // === Signed Edge Tests ===

    #[test]
    fn test_signed_edge_grants_access() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Signed edges (created by grant) should work
        let value = vault.get("user:alice", "secret").unwrap();
        assert_eq!(value, "value");

        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );
    }

    #[test]
    fn test_tampered_edge_denied() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Verify access works before tampering
        assert!(vault.get("user:alice", "secret").is_ok());

        // Find the access edge and tamper with its signature
        let secret_node = vault.secret_node_key("secret");
        let alice_node_id = vault.get_or_create_entity_node("user:alice");
        let edges = vault
            .graph
            .edges_of(alice_node_id, Direction::Outgoing)
            .unwrap();

        for edge in edges {
            if edge.edge_type.starts_with("VAULT_ACCESS") {
                // Delete the edge and recreate with a tampered signature
                vault.graph.delete_edge(edge.id).unwrap();

                let target_node = vault.get_or_create_entity_node(&secret_node);
                let mut props = HashMap::new();
                props.insert(
                    "vault_sig".to_string(),
                    PropertyValue::Bytes(vec![0xDE, 0xAD, 0xBE, 0xEF]),
                );
                props.insert("vault_sig_ts".to_string(), PropertyValue::Int(12345));
                vault
                    .graph
                    .create_edge(alice_node_id, target_node, &edge.edge_type, props, true)
                    .unwrap();
            }
        }

        // Access should be denied with tampered signature
        let result = vault.get("user:alice", "secret");
        assert!(
            result.is_err(),
            "Tampered edge signature should deny access"
        );
    }

    #[test]
    fn test_unsigned_legacy_edge_accepted() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Manually create an unsigned VAULT_ACCESS edge (legacy/backward-compatible)
        let secret_node = vault.secret_node_key("secret");
        add_test_edge(
            &vault.graph,
            "user:legacy",
            &secret_node,
            "VAULT_ACCESS_READ",
        );

        // Legacy unsigned edges should still be accepted
        let value = vault.get("user:legacy", "secret").unwrap();
        assert_eq!(value, "value");

        assert_eq!(
            vault.get_permission("user:legacy", "secret"),
            Some(Permission::Read)
        );
    }

    #[test]
    fn test_tampered_edge_not_in_list() {
        let vault = create_test_vault();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Verify secret shows in list before tampering
        assert_eq!(vault.list("user:alice", "*").unwrap().len(), 1);

        // Tamper with the edge signature
        let secret_node = vault.secret_node_key("secret");
        let alice_node_id = vault.get_or_create_entity_node("user:alice");
        let edges = vault
            .graph
            .edges_of(alice_node_id, Direction::Outgoing)
            .unwrap();

        for edge in edges {
            if edge.edge_type.starts_with("VAULT_ACCESS") {
                vault.graph.delete_edge(edge.id).unwrap();

                let target_node = vault.get_or_create_entity_node(&secret_node);
                let mut props = HashMap::new();
                props.insert(
                    "vault_sig".to_string(),
                    PropertyValue::Bytes(vec![0xFF; 32]),
                );
                props.insert("vault_sig_ts".to_string(), PropertyValue::Int(99999));
                vault
                    .graph
                    .create_edge(alice_node_id, target_node, &edge.edge_type, props, true)
                    .unwrap();
            }
        }

        // Tampered edge should make secret invisible in list
        let keys = vault.list("user:alice", "*").unwrap();
        assert!(
            keys.is_empty(),
            "Tampered edge should hide secret from list()"
        );
    }

    // === Attenuation Integration Tests ===

    fn create_vault_with_attenuation(policy: AttenuationPolicy) -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_attenuation(policy);
        Vault::new(b"test_password", graph, store, config).unwrap()
    }

    #[test]
    fn test_attenuation_direct_admin_preserved() {
        let vault = create_test_vault(); // default policy: admin_limit=1

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Admin)
            .unwrap();

        // Direct (1 hop): Admin preserved
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );
    }

    #[test]
    fn test_attenuation_2hop_admin_attenuated_to_write() {
        let vault = create_test_vault(); // default: admin_limit=1, write_limit=2

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // alice -> team (MEMBER, 1 hop) -> secret (VAULT_ACCESS_ADMIN, +1 = 2 hops)
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Admin)
            .unwrap();

        // 2 hops: Admin attenuated to Write
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Write)
        );

        // Can read and write but not delete (Admin required)
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.set("user:alice", "secret", "updated").is_ok());
        assert!(matches!(
            vault.delete("user:alice", "secret"),
            Err(VaultError::InsufficientPermission(_))
        ));
    }

    #[test]
    fn test_attenuation_3hop_attenuated_to_read() {
        let vault = create_test_vault(); // default: admin_limit=1, write_limit=2

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // alice -> org (MEMBER) -> team (MEMBER) -> secret (VAULT_ACCESS_ADMIN)
        // depth: 0 -> 1 -> 2 -> VAULT_ACCESS at depth 3
        add_test_edge(&vault.graph, "user:alice", "org:eng", "MEMBER");
        add_test_edge(&vault.graph, "org:eng", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Admin)
            .unwrap();

        // 3 hops: Admin attenuated to Read
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Read)
        );

        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(matches!(
            vault.set("user:alice", "secret", "hack"),
            Err(VaultError::InsufficientPermission(_))
        ));
    }

    #[test]
    fn test_attenuation_beyond_horizon_denied() {
        // horizon=2 means BFS stops at depth 2
        let policy = AttenuationPolicy {
            admin_limit: 1,
            write_limit: 1,
            horizon: 2,
        };
        let vault = create_vault_with_attenuation(policy);

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // alice -> org -> team -> secret (3 MEMBER hops + VAULT_ACCESS = beyond horizon)
        add_test_edge(&vault.graph, "user:alice", "org:eng", "MEMBER");
        add_test_edge(&vault.graph, "org:eng", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Admin)
            .unwrap();

        // Beyond horizon: no access
        assert_eq!(vault.get_permission("user:alice", "secret"), None);
        assert!(vault.get("user:alice", "secret").is_err());
    }

    #[test]
    fn test_no_attenuation_policy_preserves_admin_at_depth() {
        let vault = create_vault_with_attenuation(AttenuationPolicy::none());

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Deep chain: alice -> g1 -> g2 -> g3 -> g4 -> secret
        add_test_edge(&vault.graph, "user:alice", "group:1", "MEMBER");
        add_test_edge(&vault.graph, "group:1", "group:2", "MEMBER");
        add_test_edge(&vault.graph, "group:2", "group:3", "MEMBER");
        add_test_edge(&vault.graph, "group:3", "group:4", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "group:4", "secret", Permission::Admin)
            .unwrap();

        // With no attenuation, Admin is preserved regardless of depth
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );
    }

    #[test]
    fn test_with_attenuation_config_builder() {
        let policy = AttenuationPolicy {
            admin_limit: 3,
            write_limit: 5,
            horizon: 20,
        };
        let config = VaultConfig::default().with_attenuation(policy);
        assert_eq!(config.attenuation.admin_limit, 3);
        assert_eq!(config.attenuation.write_limit, 5);
        assert_eq!(config.attenuation.horizon, 20);
    }

    // === Bottleneck Flow Tests ===

    #[test]
    fn test_bottleneck_restricts_permission() {
        // Grant Admin to a team lead, but team lead grants Read to member.
        // Member should get Read (bottleneck of the edge capacity).
        let vault = create_vault_with_attenuation(AttenuationPolicy::none());

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:lead", "secret", Permission::Admin)
            .unwrap();

        // Lead grants Read to member -- the edge capacity is Read
        vault
            .grant_with_permission("user:lead", "user:member", "secret", Permission::Read)
            .unwrap();

        // Member gets Read (capped by edge capacity)
        assert_eq!(
            vault.get_permission("user:member", "secret"),
            Some(Permission::Read)
        );
        assert!(vault.get("user:member", "secret").is_ok());
        assert!(matches!(
            vault.set("user:member", "secret", "hack"),
            Err(VaultError::InsufficientPermission(_))
        ));
    }

    #[test]
    fn test_bottleneck_multiple_paths_best_wins() {
        // Two paths with different capacities: best effective wins.
        let vault = create_vault_with_attenuation(AttenuationPolicy::none());

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Path 1: direct Read
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Read)
            .unwrap();

        // Path 2: via team with Write
        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Write)
            .unwrap();

        // Best of (Read, Write) = Write
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_bottleneck_with_attenuation() {
        // Capacity=Admin, but 2 hops attenuate Admin->Write. Bottleneck = min(Admin, Write) = Write.
        let vault = create_test_vault(); // default attenuation: admin_limit=1

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        add_test_edge(&vault.graph, "user:alice", "team:devs", "MEMBER");
        vault
            .grant_with_permission(Vault::ROOT, "team:devs", "secret", Permission::Admin)
            .unwrap();

        // 2 hops + Admin capacity: attenuation gives Write, capacity is Admin
        // effective = min(Write, Admin) = Write
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_permission_to_level_roundtrip() {
        assert_eq!(
            Permission::from_level(Permission::Read.to_level()),
            Some(Permission::Read)
        );
        assert_eq!(
            Permission::from_level(Permission::Write.to_level()),
            Some(Permission::Write)
        );
        assert_eq!(
            Permission::from_level(Permission::Admin.to_level()),
            Some(Permission::Admin)
        );
        assert_eq!(Permission::from_level(0), None);
        assert_eq!(Permission::from_level(4), None);
    }

    // === TTL Persistence Tests ===

    #[test]
    fn test_ttl_survives_vault_reopen() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_salt([42u8; 16]);

        // Create vault and grant with TTL
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store.clone(),
            config.clone(),
        )
        .unwrap();
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

        // Verify TTL is tracked
        assert!(!vault.ttl_tracker.is_empty());

        // Drop vault and reopen with same store
        drop(vault);
        let vault2 = Vault::new(b"test_password", graph, store, config).unwrap();

        // TTL should survive the reopen
        assert!(
            !vault2.ttl_tracker.is_empty(),
            "TTL grants should persist across vault reopens"
        );
    }

    // === Delete Graph Cleanup Tests ===

    #[test]
    fn test_delete_cleans_graph_edges() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store,
            VaultConfig::default(),
        )
        .unwrap();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        // Alice should have access
        assert!(vault.get("user:alice", "secret").is_ok());

        // Delete the secret
        vault.delete(Vault::ROOT, "secret").unwrap();

        // The secret node should be gone from the graph (use actual obfuscated key)
        let secret_node_key = vault.secret_node_key("secret");
        let found = graph
            .find_nodes_by_property("entity_key", &PropertyValue::String(secret_node_key))
            .unwrap_or_default();
        assert!(found.is_empty(), "Secret graph node should be deleted");

        // Alice's entity node should have no outgoing VAULT_ACCESS edges
        assert!(
            !has_access_edges(&graph, "user:alice"),
            "Alice should have no VAULT_ACCESS edges after secret deletion"
        );
    }

    #[test]
    fn test_delete_cleans_ttl_entries() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap();

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

        assert!(
            !vault.ttl_tracker.is_empty(),
            "TTL should be tracked after grant"
        );

        // Delete the secret -- TTL entries should be cleaned up
        vault.delete(Vault::ROOT, "secret").unwrap();

        assert!(
            vault.ttl_tracker.is_empty(),
            "TTL tracker should be empty after secret deletion"
        );
    }

    #[test]
    fn test_delete_cleans_edges_multiple_grantees() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store,
            VaultConfig::default(),
        )
        .unwrap();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        vault.grant(Vault::ROOT, "user:bob", "secret").unwrap();

        // Both should have access
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(vault.get("user:bob", "secret").is_ok());

        // Delete the secret
        vault.delete(Vault::ROOT, "secret").unwrap();

        // Both entities should have no VAULT_ACCESS edges
        for entity in &["user:alice", "user:bob"] {
            assert!(
                !has_access_edges(&graph, entity),
                "{entity} should have no VAULT_ACCESS edges after secret deletion"
            );
        }
    }

    #[test]
    fn test_delete_then_recreate_secret() {
        let vault = create_test_vault();

        // Create, grant, delete
        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        vault.delete(Vault::ROOT, "secret").unwrap();

        // Recreate the same secret
        vault.set(Vault::ROOT, "secret", "v2").unwrap();

        // Alice should NOT have access (old grant was cleaned up)
        assert!(
            vault.get("user:alice", "secret").is_err(),
            "Old grant should not carry over after delete + recreate"
        );

        // Root should still work
        let val = vault.get(Vault::ROOT, "secret").unwrap();
        assert_eq!(val, "v2");

        // Can grant again
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        assert_eq!(vault.get("user:alice", "secret").unwrap(), "v2");
    }

    // === Master Key Rotation Tests ===

    fn create_test_vault_mut() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(b"old_password", graph, store, VaultConfig::default()).unwrap()
    }

    #[test]
    fn test_rotate_master_key_basic() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "api_key", "sk-secret123").unwrap();
        vault.set(Vault::ROOT, "db_pass", "pg-hunter2").unwrap();

        let count = vault.rotate_master_key(b"new_password").unwrap();
        assert_eq!(count, 2);

        // Secrets accessible with new key material
        assert_eq!(vault.get(Vault::ROOT, "api_key").unwrap(), "sk-secret123");
        assert_eq!(vault.get(Vault::ROOT, "db_pass").unwrap(), "pg-hunter2");
    }

    #[test]
    fn test_rotate_master_key_preserves_access() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Read)
            .unwrap();

        assert!(vault.get("user:alice", "secret").is_ok());

        vault.rotate_master_key(b"new_password").unwrap();

        // Alice can still access after rotation
        assert_eq!(vault.get("user:alice", "secret").unwrap(), "value");
    }

    #[test]
    fn test_rotate_master_key_preserves_versions() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "v1").unwrap();
        vault.set(Vault::ROOT, "secret", "v2").unwrap();
        vault.set(Vault::ROOT, "secret", "v3").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "v3");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 1).unwrap(), "v1");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 2).unwrap(), "v2");
        assert_eq!(vault.get_version(Vault::ROOT, "secret", 3).unwrap(), "v3");
        assert_eq!(vault.current_version(Vault::ROOT, "secret").unwrap(), 3);
    }

    #[test]
    fn test_rotate_master_key_old_password_fails() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let mut vault = Vault::new(
            b"old_password",
            graph.clone(),
            store.clone(),
            VaultConfig::default(),
        )
        .unwrap();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.rotate_master_key(b"new_password").unwrap();

        // Opening with old password should produce different key material,
        // so decryption should fail
        let old_vault = Vault::new(b"old_password", graph, store, VaultConfig::default()).unwrap();
        let result = old_vault.get(Vault::ROOT, "secret");
        assert!(
            result.is_err(),
            "Old password should not decrypt after rotation"
        );
    }

    #[test]
    fn test_rotate_master_key_empty_vault() {
        let mut vault = create_test_vault_mut();

        let count = vault.rotate_master_key(b"new_password").unwrap();
        assert_eq!(count, 0);

        // Vault still works after rotating with no secrets
        vault.set(Vault::ROOT, "new_secret", "value").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "new_secret").unwrap(), "value");
    }

    #[test]
    fn test_rotate_master_key_list_works() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "alpha", "a").unwrap();
        vault.set(Vault::ROOT, "beta", "b").unwrap();
        vault.set(Vault::ROOT, "gamma", "c").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        let mut keys = vault.list(Vault::ROOT, "*").unwrap();
        keys.sort();
        assert_eq!(keys, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_rotate_master_key_edge_signatures_valid() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "secret", Permission::Admin)
            .unwrap();

        // Verify access works before rotation
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );

        vault.rotate_master_key(b"new_password").unwrap();

        // Signed edges should still verify with new key material
        assert_eq!(
            vault.get_permission("user:alice", "secret"),
            Some(Permission::Admin)
        );
        vault.set("user:alice", "secret", "updated").unwrap();
        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "updated");
    }

    #[test]
    fn test_rotate_master_key_ttl_preserved() {
        let mut vault = create_test_vault_mut();

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

        vault.rotate_master_key(b"new_password").unwrap();

        // TTL grant should still work after rotation
        assert!(vault.get("user:alice", "secret").is_ok());
        assert!(!vault.ttl_tracker.is_empty());
    }

    #[test]
    fn test_rotate_master_key_audit_recorded() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.rotate_master_key(b"new_password").unwrap();

        let recent = vault.audit_recent(10);
        let rotation_entries: Vec<_> = recent
            .iter()
            .filter(|e| matches!(e.operation, AuditOperation::RotateMasterKey { .. }))
            .collect();
        assert!(
            !rotation_entries.is_empty(),
            "Should have RotateMasterKey audit entry"
        );

        if let AuditOperation::RotateMasterKey { secrets_count } = &rotation_entries[0].operation {
            assert_eq!(*secrets_count, 1);
        }
    }

    #[test]
    fn test_rotate_master_key_multiple_grants() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "s1", "v1").unwrap();
        vault.set(Vault::ROOT, "s2", "v2").unwrap();

        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s1", Permission::Admin)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:bob", "s2", Permission::Write)
            .unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "s2", Permission::Read)
            .unwrap();

        vault.rotate_master_key(b"rotated").unwrap();

        assert_eq!(vault.get("user:alice", "s1").unwrap(), "v1");
        assert_eq!(vault.get("user:bob", "s2").unwrap(), "v2");
        assert_eq!(vault.get("user:alice", "s2").unwrap(), "v2");

        // Permission levels preserved
        assert_eq!(
            vault.get_permission("user:alice", "s1"),
            Some(Permission::Admin)
        );
        assert_eq!(
            vault.get_permission("user:bob", "s2"),
            Some(Permission::Write)
        );
    }

    #[test]
    fn test_rotate_master_key_new_vault_instance() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let mut vault = Vault::new(
            b"old_password",
            graph.clone(),
            store.clone(),
            VaultConfig::default(),
        )
        .unwrap();

        vault.set(Vault::ROOT, "secret", "my_value").unwrap();
        vault.rotate_master_key(b"new_password").unwrap();

        // Create a fresh vault instance with the new password
        let vault2 = Vault::new(b"new_password", graph, store, VaultConfig::default()).unwrap();
        assert_eq!(vault2.get(Vault::ROOT, "secret").unwrap(), "my_value");
    }

    #[test]
    fn test_rotate_master_key_unsigned_edges_preserved() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Create an unsigned legacy edge
        let secret_node = vault.secret_node_key("secret");
        add_test_edge(
            &vault.graph,
            "user:legacy",
            &secret_node,
            "VAULT_ACCESS_READ",
        );

        // Legacy user can access
        assert!(vault.get("user:legacy", "secret").is_ok());

        vault.rotate_master_key(b"new_password").unwrap();

        // Unsigned legacy edge should still work (not re-signed)
        assert!(vault.get("user:legacy", "secret").is_ok());
    }

    #[test]
    fn test_rotate_master_key_secrets_without_grants() {
        let mut vault = create_test_vault_mut();

        // Create secrets with no grants (only root access)
        vault.set(Vault::ROOT, "lonely_secret", "alone").unwrap();

        vault.rotate_master_key(b"new_password").unwrap();

        assert_eq!(vault.get(Vault::ROOT, "lonely_secret").unwrap(), "alone");
    }

    #[test]
    fn test_ttl_grant_cleaned_on_restart() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_salt([42u8; 16]);

        // Create vault, store a secret, grant with short TTL
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store.clone(),
            config.clone(),
        )
        .unwrap();
        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault
            .grant_with_ttl(
                Vault::ROOT,
                "user:alice",
                "secret",
                Permission::Read,
                Duration::from_secs(0),
            )
            .unwrap();

        // Wait for TTL to expire, then drop (simulating process exit without cleanup)
        std::thread::sleep(Duration::from_millis(10));
        drop(vault);

        // Reopen vault -- constructor should clean up the expired grant
        let vault2 = Vault::new(b"test_password", graph, store, config).unwrap();

        // The expired grant's graph edge should have been removed on startup
        let result = vault2.get("user:alice", "secret");
        assert!(
            matches!(result, Err(VaultError::AccessDenied(_))),
            "expired TTL grant should be revoked after restart"
        );
    }

    #[test]
    fn test_ttl_non_expired_survives_restart() {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let config = VaultConfig::default().with_salt([42u8; 16]);

        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store.clone(),
            config.clone(),
        )
        .unwrap();
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

        drop(vault);

        // Reopen vault -- non-expired grant should still work
        let vault2 = Vault::new(b"test_password", graph, store, config).unwrap();
        assert!(
            vault2.get("user:alice", "secret").is_ok(),
            "non-expired TTL grant should survive restart"
        );
    }

    #[test]
    fn test_rotate_master_key_double_rotation() {
        let mut vault = create_test_vault_mut();

        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

        vault.rotate_master_key(b"password_2").unwrap();
        vault.rotate_master_key(b"password_3").unwrap();

        assert_eq!(vault.get(Vault::ROOT, "secret").unwrap(), "value");
        assert_eq!(vault.get("user:alice", "secret").unwrap(), "value");
    }
}
