// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Namespaced vault view for multi-tenant isolation.

use std::time::Duration;

use crate::{vault::Vault, Permission, Result};

/// A namespaced view of the vault that prefixes all keys with a namespace.
///
/// Provides isolation between different tenants or agent contexts.
pub struct NamespacedVault<'a> {
    vault: &'a Vault,
    namespace: String,
    identity: String,
}

impl<'a> NamespacedVault<'a> {
    pub(crate) fn new(vault: &'a Vault, namespace: &str, identity: &str) -> Self {
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
        let ns_pattern = format!("{}:{}", self.namespace, pattern);
        let keys = self.vault.list(&self.identity, &ns_pattern)?;

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

    /// Store a secret with a TTL in the namespace.
    pub fn set_with_ttl(&self, key: &str, value: &str, ttl: Duration) -> Result<()> {
        let prefixed = self.prefixed_key(key);
        self.vault
            .set_with_ttl(&self.identity, &prefixed, value, ttl)
    }

    /// Remove expiration from a secret in the namespace.
    pub fn clear_expiration(&self, key: &str) -> Result<()> {
        let prefixed = self.prefixed_key(key);
        self.vault.clear_expiration(&self.identity, &prefixed)
    }

    /// Get the expiration timestamp of a secret in the namespace.
    pub fn get_expiration(&self, key: &str) -> Result<Option<i64>> {
        let prefixed = self.prefixed_key(key);
        self.vault.get_expiration(&self.identity, &prefixed)
    }

    /// Encrypt data using the transit key, scoped to a secret in the namespace.
    pub fn encrypt_for(&self, key: &str, plaintext: &[u8]) -> Result<Vec<u8>> {
        let prefixed = self.prefixed_key(key);
        self.vault.encrypt_for(&self.identity, &prefixed, plaintext)
    }

    /// Decrypt transit-encrypted data, scoped to a secret in the namespace.
    pub fn decrypt_as(&self, key: &str, sealed: &[u8]) -> Result<Vec<u8>> {
        let prefixed = self.prefixed_key(key);
        self.vault.decrypt_as(&self.identity, &prefixed, sealed)
    }

    /// Emergency access to a secret in the namespace.
    pub fn emergency_access(
        &self,
        key: &str,
        justification: &str,
        duration: Duration,
    ) -> Result<String> {
        let prefixed = self.prefixed_key(key);
        self.vault
            .emergency_access(&self.identity, &prefixed, justification, duration)
    }

    /// Batch-get multiple secrets in the namespace.
    pub fn batch_get(&self, keys: &[&str]) -> Result<Vec<(String, Result<String>)>> {
        let prefixed_keys: Vec<String> = keys.iter().map(|k| self.prefixed_key(k)).collect();
        let prefixed_refs: Vec<&str> = prefixed_keys.iter().map(String::as_str).collect();
        let results = self.vault.batch_get(&self.identity, &prefixed_refs)?;

        Ok(results
            .into_iter()
            .map(|(k, v)| {
                let stripped = self
                    .strip_prefix(&k)
                    .map_or_else(|| k.clone(), String::from);
                (stripped, v)
            })
            .collect())
    }

    /// Batch-set multiple secrets in the namespace.
    pub fn batch_set(&self, entries: &[(&str, &str)]) -> Result<()> {
        let prefixed: Vec<(String, String)> = entries
            .iter()
            .map(|(k, v)| (self.prefixed_key(k), (*v).to_string()))
            .collect();
        let refs: Vec<(&str, &str)> = prefixed
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        self.vault.batch_set(&self.identity, &refs)
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
    use std::sync::Arc;
    use std::time::Duration;

    use graph_engine::GraphEngine;
    use tensor_store::TensorStore;

    use crate::vault::Vault;
    use crate::{RateLimitConfig, VaultConfig};

    fn create_test_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(b"test_password", graph, store, VaultConfig::default()).unwrap()
    }

    #[test]
    fn test_namespaced_clear_expiration() {
        let vault = create_test_vault();
        let ns = vault.namespace("prod", Vault::ROOT);

        ns.set_with_ttl("ephemeral", "temp_value", Duration::from_secs(3600))
            .unwrap();

        // Confirm expiration is set
        let exp = ns.get_expiration("ephemeral").unwrap();
        assert!(exp.is_some());

        // Clear expiration
        ns.clear_expiration("ephemeral").unwrap();

        // Confirm expiration is removed
        let exp_after = ns.get_expiration("ephemeral").unwrap();
        assert!(exp_after.is_none());

        // Secret is still readable
        assert_eq!(ns.get("ephemeral").unwrap(), "temp_value");
    }

    #[test]
    fn test_namespaced_emergency_access() {
        let config = VaultConfig::default().with_rate_limit(RateLimitConfig {
            max_gets: 100,
            max_lists: 100,
            max_sets: 100,
            max_grants: 100,
            max_break_glass: 10,
            max_wraps: 100,
            max_generates: 100,
            window: Duration::from_secs(60),
        });
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(b"test_password", graph, store, config).unwrap();

        // Root sets a secret in the namespace
        let ns_root = vault.namespace("secure", Vault::ROOT);
        ns_root.set("classified", "top_secret_data").unwrap();

        // user:bob has no access through normal channels
        let ns_bob = vault.namespace("secure", "user:bob");
        assert!(ns_bob.get("classified").is_err());

        // Emergency break-glass access
        let value = ns_bob
            .emergency_access(
                "classified",
                "Production incident P1-4321",
                Duration::from_secs(60),
            )
            .unwrap();
        assert_eq!(value, "top_secret_data");
    }

    #[test]
    fn test_namespaced_batch_set() {
        let vault = create_test_vault();
        let ns = vault.namespace("env", Vault::ROOT);

        let entries = vec![
            ("DB_HOST", "localhost"),
            ("DB_PORT", "5432"),
            ("DB_NAME", "mydb"),
        ];
        ns.batch_set(&entries).unwrap();

        assert_eq!(ns.get("DB_HOST").unwrap(), "localhost");
        assert_eq!(ns.get("DB_PORT").unwrap(), "5432");
        assert_eq!(ns.get("DB_NAME").unwrap(), "mydb");
    }
}
