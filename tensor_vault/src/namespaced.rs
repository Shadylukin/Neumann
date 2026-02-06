// SPDX-License-Identifier: MIT OR Apache-2.0
//! Namespaced vault view for multi-tenant isolation.

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

    /// Get the namespace name.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Get the identity.
    pub fn identity(&self) -> &str {
        &self.identity
    }
}
