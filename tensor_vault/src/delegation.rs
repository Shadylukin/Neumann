// SPDX-License-Identifier: MIT OR Apache-2.0
//! Agent delegation protocol for permission-preserving sub-agent access.
//!
//! Allows agents to delegate their own permissions (or a subset) to child
//! agents, with depth limits and cascading revocation. Composes naturally
//! with the attenuation system: each delegation hop adds graph distance,
//! so permissions decay with chain depth.

use std::collections::VecDeque;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorStore, TensorValue};

use crate::{Permission, Result, VaultError};

/// Storage key prefix for persisted delegation records.
const DELEGATION_PREFIX: &str = "_vdel:";

/// A recorded delegation from parent to child agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationRecord {
    /// The delegating (parent) agent.
    pub parent: String,
    /// The receiving (child) agent.
    pub child: String,
    /// Original (plaintext) secret names -- needed for graph edge management.
    pub secrets: Vec<String>,
    /// Maximum permission the child can have (ceiling).
    pub max_permission: Permission,
    /// Optional TTL in milliseconds.
    pub ttl_ms: Option<i64>,
    /// When this delegation was created (unix ms).
    pub created_at_ms: i64,
    /// How many delegation hops from root to this child.
    pub delegation_depth: u32,
}

/// Manages delegation relationships between agents.
pub struct DelegationManager {
    records: DashMap<String, DelegationRecord>,
    max_depth: u32,
}

impl DelegationManager {
    pub fn new(max_depth: u32) -> Self {
        Self {
            records: DashMap::new(),
            max_depth,
        }
    }

    /// Load persisted delegation records from the store.
    pub fn load(store: &TensorStore, max_depth: u32) -> Self {
        let mgr = Self::new(max_depth);
        for key in store.scan(DELEGATION_PREFIX) {
            if let Ok(data) = store.get(&key) {
                if let Some(TensorValue::Scalar(ScalarValue::String(json))) = data.get("_record") {
                    if let Ok(record) = serde_json::from_str::<DelegationRecord>(json) {
                        let record_key = key
                            .strip_prefix(DELEGATION_PREFIX)
                            .unwrap_or(&key)
                            .to_string();
                        mgr.records.insert(record_key, record);
                    }
                }
            }
        }
        mgr
    }

    /// Persist all delegation records to the store.
    pub fn persist(&self, store: &TensorStore) {
        for key in store.scan(DELEGATION_PREFIX) {
            store.delete(&key).ok();
        }
        for entry in &self.records {
            let storage_key = format!("{DELEGATION_PREFIX}{}", entry.key());
            if let Ok(json) = serde_json::to_string(entry.value()) {
                let mut data = tensor_store::TensorData::new();
                data.set("_record", TensorValue::Scalar(ScalarValue::String(json)));
                store.put(&storage_key, data).ok();
            }
        }
    }

    /// Register a new delegation from parent to child.
    pub fn register(
        &self,
        parent: &str,
        child: &str,
        secrets: Vec<String>,
        max_permission: Permission,
        ttl_ms: Option<i64>,
        now_ms: i64,
    ) -> Result<DelegationRecord> {
        // Reject self-delegation
        if parent == child {
            return Err(VaultError::GraphError(
                "cannot delegate to self".to_string(),
            ));
        }

        // Reject cycles: child must not be an ancestor of parent
        if self.is_ancestor(child, parent) {
            return Err(VaultError::GraphError(
                "delegation would create a cycle".to_string(),
            ));
        }

        // Check depth limit
        let parent_depth = self.delegation_depth(parent);
        let child_depth = parent_depth + 1;
        if child_depth > self.max_depth {
            return Err(VaultError::GraphError(format!(
                "delegation depth {child_depth} exceeds maximum {}",
                self.max_depth
            )));
        }

        let record = DelegationRecord {
            parent: parent.to_string(),
            child: child.to_string(),
            secrets,
            max_permission,
            ttl_ms,
            created_at_ms: now_ms,
            delegation_depth: child_depth,
        };

        let key = Self::record_key(parent, child);
        self.records.insert(key, record.clone());
        Ok(record)
    }

    /// Revoke a single delegation from parent to child.
    pub fn revoke(&self, parent: &str, child: &str) -> Option<DelegationRecord> {
        let key = Self::record_key(parent, child);
        self.records.remove(&key).map(|(_, v)| v)
    }

    /// Revoke a delegation and all transitive delegations (BFS).
    pub fn revoke_cascading(&self, parent: &str, child: &str) -> Vec<DelegationRecord> {
        let mut revoked = Vec::new();

        // Start by revoking the direct delegation
        if let Some(record) = self.revoke(parent, child) {
            revoked.push(record);
        }

        // BFS through descendants
        let mut queue = VecDeque::new();
        queue.push_back(child.to_string());

        while let Some(current) = queue.pop_front() {
            let children = self.children_of(&current);
            for c in children {
                if let Some(record) = self.revoke(&current, &c) {
                    revoked.push(record);
                    queue.push_back(c);
                }
            }
        }

        revoked
    }

    /// Get direct children of an entity.
    pub fn children_of(&self, parent: &str) -> Vec<String> {
        let prefix = format!("{parent}:");
        self.records
            .iter()
            .filter(|e| e.key().starts_with(&prefix))
            .map(|e| e.value().child.clone())
            .collect()
    }

    /// Get all transitive descendants of an entity (BFS).
    pub fn descendants_of(&self, entity: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(entity.to_string());

        while let Some(current) = queue.pop_front() {
            for child in self.children_of(&current) {
                if !result.contains(&child) {
                    result.push(child.clone());
                    queue.push_back(child);
                }
            }
        }
        result
    }

    /// Look up a specific delegation record.
    pub fn get_delegation(&self, parent: &str, child: &str) -> Option<DelegationRecord> {
        let key = Self::record_key(parent, child);
        self.records.get(&key).map(|e| e.value().clone())
    }

    /// Walk the delegation chain upward to compute depth.
    pub fn delegation_depth(&self, entity: &str) -> u32 {
        // Find any record where this entity is the child
        for entry in &self.records {
            if entry.value().child == entity {
                return entry.value().delegation_depth;
            }
        }
        0
    }

    /// Remove expired delegations. Returns the count removed.
    pub fn cleanup_expired(&self, now_ms: i64) -> usize {
        let mut expired_keys = Vec::new();
        for entry in &self.records {
            if let Some(ttl) = entry.value().ttl_ms {
                if entry.value().created_at_ms + ttl <= now_ms {
                    expired_keys.push(entry.key().clone());
                }
            }
        }
        let count = expired_keys.len();
        for key in expired_keys {
            self.records.remove(&key);
        }
        count
    }

    fn record_key(parent: &str, child: &str) -> String {
        format!("{parent}:{child}")
    }

    /// Check if `ancestor` is an ancestor of `entity` in the delegation chain.
    fn is_ancestor(&self, ancestor: &str, entity: &str) -> bool {
        // Walk upward from entity
        let mut current = entity.to_string();
        let mut visited = std::collections::HashSet::new();
        visited.insert(current.clone());

        loop {
            let mut found_parent = None;
            for entry in &self.records {
                if entry.value().child == current {
                    found_parent = Some(entry.value().parent.clone());
                    break;
                }
            }
            match found_parent {
                Some(parent) if parent == ancestor => return true,
                Some(parent) => {
                    if !visited.insert(parent.clone()) {
                        return false; // loop guard
                    }
                    current = parent;
                },
                None => return false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_delegation() {
        let mgr = DelegationManager::new(3);
        let record = mgr
            .register(
                "agent:root",
                "agent:child",
                vec!["secret1".into()],
                Permission::Write,
                None,
                1000,
            )
            .unwrap();
        assert_eq!(record.parent, "agent:root");
        assert_eq!(record.child, "agent:child");
        assert_eq!(record.delegation_depth, 1);
    }

    #[test]
    fn test_depth_limit_exceeded() {
        let mgr = DelegationManager::new(2);
        mgr.register("a", "b", vec![], Permission::Read, None, 1000)
            .unwrap();
        mgr.register("b", "c", vec![], Permission::Read, None, 1000)
            .unwrap();
        let result = mgr.register("c", "d", vec![], Permission::Read, None, 1000);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("depth"));
    }

    #[test]
    fn test_self_delegation_rejected() {
        let mgr = DelegationManager::new(3);
        let result = mgr.register("agent:a", "agent:a", vec![], Permission::Read, None, 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("self"));
    }

    #[test]
    fn test_cycle_prevention() {
        let mgr = DelegationManager::new(5);
        mgr.register("a", "b", vec![], Permission::Read, None, 1000)
            .unwrap();
        mgr.register("b", "c", vec![], Permission::Read, None, 1000)
            .unwrap();
        // c -> a would create a cycle: a -> b -> c -> a
        let result = mgr.register("c", "a", vec![], Permission::Read, None, 1000);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cycle"));
    }

    #[test]
    fn test_permission_ceiling() {
        let mgr = DelegationManager::new(3);
        let record = mgr
            .register(
                "parent",
                "child",
                vec!["s1".into()],
                Permission::Read,
                None,
                1000,
            )
            .unwrap();
        assert_eq!(record.max_permission, Permission::Read);
    }

    #[test]
    fn test_revoke_delegation() {
        let mgr = DelegationManager::new(3);
        mgr.register("a", "b", vec!["s1".into()], Permission::Write, None, 1000)
            .unwrap();
        let revoked = mgr.revoke("a", "b");
        assert!(revoked.is_some());
        assert!(mgr.get_delegation("a", "b").is_none());
    }

    #[test]
    fn test_revoke_cascading() {
        let mgr = DelegationManager::new(5);
        mgr.register("a", "b", vec!["s1".into()], Permission::Write, None, 1000)
            .unwrap();
        mgr.register("b", "c", vec!["s1".into()], Permission::Read, None, 1000)
            .unwrap();
        mgr.register("c", "d", vec!["s1".into()], Permission::Read, None, 1000)
            .unwrap();

        let revoked = mgr.revoke_cascading("a", "b");
        assert_eq!(revoked.len(), 3);
        assert!(mgr.get_delegation("a", "b").is_none());
        assert!(mgr.get_delegation("b", "c").is_none());
        assert!(mgr.get_delegation("c", "d").is_none());
    }

    #[test]
    fn test_descendants_of() {
        let mgr = DelegationManager::new(5);
        mgr.register("a", "b", vec![], Permission::Read, None, 1000)
            .unwrap();
        mgr.register("b", "c", vec![], Permission::Read, None, 1000)
            .unwrap();
        mgr.register("b", "d", vec![], Permission::Read, None, 1000)
            .unwrap();

        let mut desc = mgr.descendants_of("a");
        desc.sort();
        assert_eq!(desc, vec!["b", "c", "d"]);
    }

    #[test]
    fn test_children_of() {
        let mgr = DelegationManager::new(5);
        mgr.register("a", "b", vec![], Permission::Read, None, 1000)
            .unwrap();
        mgr.register("a", "c", vec![], Permission::Read, None, 1000)
            .unwrap();
        mgr.register("b", "d", vec![], Permission::Read, None, 1000)
            .unwrap();

        let mut children = mgr.children_of("a");
        children.sort();
        assert_eq!(children, vec!["b", "c"]);
    }

    #[test]
    fn test_delegation_with_ttl() {
        let mgr = DelegationManager::new(3);
        let record = mgr
            .register("a", "b", vec![], Permission::Read, Some(5000), 1000)
            .unwrap();
        assert_eq!(record.ttl_ms, Some(5000));
    }

    #[test]
    fn test_persist_and_load() {
        let store = TensorStore::new();
        let mgr = DelegationManager::new(3);
        mgr.register("a", "b", vec!["s1".into()], Permission::Write, None, 1000)
            .unwrap();
        mgr.persist(&store);

        let loaded = DelegationManager::load(&store, 3);
        let record = loaded.get_delegation("a", "b").unwrap();
        assert_eq!(record.parent, "a");
        assert_eq!(record.child, "b");
        assert_eq!(record.secrets, vec!["s1"]);
    }

    #[test]
    fn test_cleanup_expired() {
        let mgr = DelegationManager::new(3);
        mgr.register("a", "b", vec![], Permission::Read, Some(1000), 1000)
            .unwrap();
        mgr.register("a", "c", vec![], Permission::Read, Some(5000), 1000)
            .unwrap();
        mgr.register("a", "d", vec![], Permission::Read, None, 1000)
            .unwrap();

        // At t=2000, only a->b should be expired (created 1000 + ttl 1000 = 2000)
        let count = mgr.cleanup_expired(2000);
        assert_eq!(count, 1);
        assert!(mgr.get_delegation("a", "b").is_none());
        assert!(mgr.get_delegation("a", "c").is_some());
        assert!(mgr.get_delegation("a", "d").is_some());
    }

    #[test]
    fn test_depth_calculation() {
        let mgr = DelegationManager::new(5);
        mgr.register("a", "b", vec![], Permission::Read, None, 1000)
            .unwrap();
        mgr.register("b", "c", vec![], Permission::Read, None, 1000)
            .unwrap();

        assert_eq!(mgr.delegation_depth("a"), 0);
        assert_eq!(mgr.delegation_depth("b"), 1);
        assert_eq!(mgr.delegation_depth("c"), 2);
    }
}
