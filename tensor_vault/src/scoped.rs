// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Scoped vault view for a specific entity.

use std::time::Duration;

use crate::audit::AuditEntry;
use crate::vault::Vault;
use crate::{Permission, Result, VersionInfo};

/// A scoped view of the vault for a specific entity.
///
/// All operations are automatically performed as the scoped entity,
/// removing the need to pass the requester on every call.
pub struct ScopedVault<'a> {
    vault: &'a Vault,
    entity: String,
}

impl<'a> ScopedVault<'a> {
    pub(crate) fn new(vault: &'a Vault, entity: &str) -> Self {
        Self {
            vault,
            entity: entity.to_string(),
        }
    }

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

    pub fn rotate(&self, key: &str, new_value: &str) -> Result<()> {
        self.vault.rotate(&self.entity, key, new_value)
    }

    pub fn get_version(&self, key: &str, version: u32) -> Result<String> {
        self.vault.get_version(&self.entity, key, version)
    }

    pub fn list_versions(&self, key: &str) -> Result<Vec<VersionInfo>> {
        self.vault.list_versions(&self.entity, key)
    }

    pub fn current_version(&self, key: &str) -> Result<u32> {
        self.vault.current_version(&self.entity, key)
    }

    pub fn rollback(&self, key: &str, version: u32) -> Result<()> {
        self.vault.rollback(&self.entity, key, version)
    }

    pub fn grant_with_permission(&self, entity: &str, key: &str, level: Permission) -> Result<()> {
        self.vault
            .grant_with_permission(&self.entity, entity, key, level)
    }

    pub fn grant_with_ttl(
        &self,
        entity: &str,
        key: &str,
        level: Permission,
        ttl: Duration,
    ) -> Result<()> {
        self.vault
            .grant_with_ttl(&self.entity, entity, key, level, ttl)
    }

    pub fn revoke(&self, entity: &str, key: &str) -> Result<()> {
        self.vault.revoke(&self.entity, entity, key)
    }

    pub fn audit_log(&self, key: &str) -> Vec<AuditEntry> {
        self.vault.audit_log(key)
    }

    pub fn get_permission(&self, key: &str) -> Option<Permission> {
        self.vault.get_permission(&self.entity, key)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graph_engine::GraphEngine;
    use tensor_store::TensorStore;

    use super::*;
    use crate::VaultConfig;

    fn create_scoped_vault_env() -> (Vault, Arc<GraphEngine>) {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        let vault = Vault::new(
            b"test_password",
            graph.clone(),
            store,
            VaultConfig::default(),
        )
        .unwrap();
        (vault, graph)
    }

    #[test]
    fn test_scoped_set_and_get() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "db/password", "initial").unwrap();
        vault
            .grant(Vault::ROOT, "user:alice", "db/password")
            .unwrap();

        let scoped = vault.scope("user:alice");
        assert_eq!(scoped.get("db/password").unwrap(), "initial");
    }

    #[test]
    fn test_scoped_rotate() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "db/password", "initial").unwrap();
        vault
            .grant(Vault::ROOT, "user:alice", "db/password")
            .unwrap();

        let scoped = vault.scope("user:alice");
        scoped.rotate("db/password", "rotated").unwrap();
        assert_eq!(scoped.get("db/password").unwrap(), "rotated");
    }

    #[test]
    fn test_scoped_get_version() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "v1").unwrap();
        vault.set(Vault::ROOT, "key", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        assert_eq!(scoped.get_version("key", 1).unwrap(), "v1");
        assert_eq!(scoped.get_version("key", 2).unwrap(), "v2");
    }

    #[test]
    fn test_scoped_list_versions() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "v1").unwrap();
        vault.set(Vault::ROOT, "key", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        let versions = scoped.list_versions("key").unwrap();
        assert_eq!(versions.len(), 2);
    }

    #[test]
    fn test_scoped_current_version() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "v1").unwrap();
        vault.set(Vault::ROOT, "key", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        assert_eq!(scoped.current_version("key").unwrap(), 2);
    }

    #[test]
    fn test_scoped_rollback() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "v1").unwrap();
        vault.set(Vault::ROOT, "key", "v2").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        scoped.rollback("key", 1).unwrap();
        assert_eq!(scoped.get("key").unwrap(), "v1");
    }

    #[test]
    fn test_scoped_grant_with_permission() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        scoped
            .grant_with_permission("user:bob", "key", Permission::Read)
            .unwrap();

        assert_eq!(vault.get("user:bob", "key").unwrap(), "value");
    }

    #[test]
    fn test_scoped_grant_with_ttl() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        scoped
            .grant_with_ttl(
                "user:bob",
                "key",
                Permission::Read,
                Duration::from_secs(3600),
            )
            .unwrap();

        assert_eq!(vault.get("user:bob", "key").unwrap(), "value");
    }

    #[test]
    fn test_scoped_revoke() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();
        vault.grant(Vault::ROOT, "user:bob", "key").unwrap();

        let scoped = vault.scope("user:alice");
        scoped.revoke("user:bob", "key").unwrap();

        assert!(vault.get("user:bob", "key").is_err());
    }

    #[test]
    fn test_scoped_audit_log() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "key").unwrap();

        let scoped = vault.scope("user:alice");
        let _ = scoped.get("key").unwrap();

        let log = scoped.audit_log("key");
        assert!(!log.is_empty());
    }

    #[test]
    fn test_scoped_get_permission() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();
        vault
            .grant_with_permission(Vault::ROOT, "user:alice", "key", Permission::Write)
            .unwrap();

        let scoped = vault.scope("user:alice");
        assert_eq!(scoped.get_permission("key"), Some(Permission::Write));
    }

    #[test]
    fn test_scoped_no_access() {
        let (vault, _graph) = create_scoped_vault_env();
        vault.set(Vault::ROOT, "key", "value").unwrap();

        let scoped = vault.scope("user:nobody");
        assert!(scoped.get("key").is_err());
        assert_eq!(scoped.get_permission("key"), None);
    }
}
