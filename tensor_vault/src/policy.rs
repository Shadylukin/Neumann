// SPDX-License-Identifier: MIT OR Apache-2.0
//! Attribute-based policy templates for declarative access control.

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{Permission, Result, VaultError};

/// Storage prefix for policy templates.
const POLICY_PREFIX: &str = "_vpol:";

/// A policy template that grants access based on entity/secret patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyTemplate {
    /// Unique name for this policy.
    pub name: String,
    /// Glob pattern to match entity names (e.g. "team:engineering/*").
    pub match_pattern: String,
    /// Glob pattern to match secret keys (e.g. "staging/*").
    pub secret_pattern: String,
    /// Permission level to grant.
    pub permission: Permission,
    /// Optional TTL in milliseconds for the granted access.
    pub ttl_ms: Option<i64>,
}

/// Result of evaluating policies for an entity.
#[derive(Debug, Clone)]
pub struct PolicyMatch {
    /// Name of the matching policy.
    pub policy_name: String,
    /// Secret pattern from the policy.
    pub secret_pattern: String,
    /// Permission level from the policy.
    pub permission: Permission,
    /// Optional TTL from the policy.
    pub ttl_ms: Option<i64>,
}

/// Thread-safe policy manager backed by persistent storage.
pub struct PolicyManager {
    policies: DashMap<String, PolicyTemplate>,
}

impl Default for PolicyManager {
    fn default() -> Self {
        Self {
            policies: DashMap::new(),
        }
    }
}

impl PolicyManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load policies from storage.
    pub fn load(store: &TensorStore) -> Self {
        let manager = Self::new();
        for key in store.scan(POLICY_PREFIX) {
            if let Some(name) = key.strip_prefix(POLICY_PREFIX) {
                if let Ok(tensor) = store.get(&key) {
                    if let Some(policy) = deserialize_policy(name, &tensor) {
                        manager.policies.insert(name.to_string(), policy);
                    }
                }
            }
        }
        manager
    }

    /// Add or update a policy template.
    pub fn add_policy(&self, store: &TensorStore, template: PolicyTemplate) -> Result<()> {
        let key = format!("{POLICY_PREFIX}{}", template.name);
        let tensor = serialize_policy(&template);
        store
            .put(&key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;
        self.policies.insert(template.name.clone(), template);
        Ok(())
    }

    /// Remove a policy template.
    pub fn remove_policy(&self, store: &TensorStore, name: &str) -> Result<()> {
        let key = format!("{POLICY_PREFIX}{name}");
        store.delete(&key).ok();
        self.policies.remove(name);
        Ok(())
    }

    /// List all policy templates.
    pub fn list_policies(&self) -> Vec<PolicyTemplate> {
        self.policies.iter().map(|e| e.value().clone()).collect()
    }

    /// Get a specific policy by name.
    pub fn get_policy(&self, name: &str) -> Option<PolicyTemplate> {
        self.policies.get(name).map(|p| p.clone())
    }

    /// Evaluate all policies for a given entity, returning matching policies.
    pub fn evaluate_policies(&self, entity: &str) -> Vec<PolicyMatch> {
        let mut matches = Vec::new();
        for entry in &self.policies {
            let policy = entry.value();
            if glob_match(&policy.match_pattern, entity) {
                matches.push(PolicyMatch {
                    policy_name: policy.name.clone(),
                    secret_pattern: policy.secret_pattern.clone(),
                    permission: policy.permission,
                    ttl_ms: policy.ttl_ms,
                });
            }
        }
        matches
    }

    /// Check if any policy grants the entity access to a specific secret.
    pub fn check_policy_access(
        &self,
        entity: &str,
        secret_key: &str,
    ) -> Option<(Permission, Option<i64>)> {
        let mut best: Option<(Permission, Option<i64>)> = None;

        for entry in &self.policies {
            let policy = entry.value();
            if glob_match(&policy.match_pattern, entity)
                && glob_match(&policy.secret_pattern, secret_key)
            {
                match best {
                    None => best = Some((policy.permission, policy.ttl_ms)),
                    Some((current_perm, _)) => {
                        if policy.permission.allows(current_perm)
                            && !current_perm.allows(policy.permission)
                        {
                            best = Some((policy.permission, policy.ttl_ms));
                        }
                    },
                }
            }
        }

        best
    }
}

/// Simple glob matching supporting `*` and `?` wildcards.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    glob_match_inner(&pattern_chars, &text_chars)
}

fn glob_match_inner(pattern: &[char], text: &[char]) -> bool {
    let mut pi = 0;
    let mut ti = 0;
    let mut saved_pattern_idx = usize::MAX;
    let mut saved_text_idx = 0;

    while ti < text.len() {
        if pi < pattern.len() && (pattern[pi] == '?' || pattern[pi] == text[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pattern.len() && pattern[pi] == '*' {
            saved_pattern_idx = pi;
            saved_text_idx = ti;
            pi += 1;
        } else if saved_pattern_idx != usize::MAX {
            pi = saved_pattern_idx + 1;
            saved_text_idx += 1;
            ti = saved_text_idx;
        } else {
            return false;
        }
    }

    while pi < pattern.len() && pattern[pi] == '*' {
        pi += 1;
    }

    pi == pattern.len()
}

fn serialize_policy(policy: &PolicyTemplate) -> TensorData {
    let mut t = TensorData::new();
    t.set(
        "_name",
        TensorValue::Scalar(ScalarValue::String(policy.name.clone())),
    );
    t.set(
        "_match",
        TensorValue::Scalar(ScalarValue::String(policy.match_pattern.clone())),
    );
    t.set(
        "_secret",
        TensorValue::Scalar(ScalarValue::String(policy.secret_pattern.clone())),
    );
    t.set(
        "_permission",
        TensorValue::Scalar(ScalarValue::String(policy.permission.to_string())),
    );
    if let Some(ttl) = policy.ttl_ms {
        t.set("_ttl", TensorValue::Scalar(ScalarValue::Int(ttl)));
    }
    t
}

fn deserialize_policy(name: &str, tensor: &TensorData) -> Option<PolicyTemplate> {
    let match_pattern = match tensor.get("_match") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => return None,
    };
    let secret_pattern = match tensor.get("_secret") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => return None,
    };
    let permission_str = match tensor.get("_permission") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => return None,
    };
    let permission = permission_str.parse::<Permission>().ok()?;
    let ttl_ms = match tensor.get("_ttl") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => Some(*v),
        _ => None,
    };

    Some(PolicyTemplate {
        name: name.to_string(),
        match_pattern,
        secret_pattern,
        permission,
        ttl_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_exact_match() {
        assert!(glob_match("hello", "hello"));
        assert!(!glob_match("hello", "world"));
    }

    #[test]
    fn test_glob_star() {
        assert!(glob_match("team:*", "team:engineering"));
        assert!(glob_match("team:*", "team:"));
        assert!(!glob_match("team:*", "user:alice"));
    }

    #[test]
    fn test_glob_question_mark() {
        assert!(glob_match("user:?lice", "user:alice"));
        assert!(!glob_match("user:?lice", "user:bob"));
    }

    #[test]
    fn test_glob_complex() {
        assert!(glob_match("*/staging/*", "app/staging/db"));
        assert!(glob_match("*/*", "a/b"));
        assert!(!glob_match("*/*", "abc"));
    }

    #[test]
    fn test_add_and_list_policies() {
        let store = TensorStore::new();
        let manager = PolicyManager::new();

        let policy = PolicyTemplate {
            name: "eng-staging".to_string(),
            match_pattern: "team:engineering/*".to_string(),
            secret_pattern: "staging/*".to_string(),
            permission: Permission::Read,
            ttl_ms: None,
        };

        manager.add_policy(&store, policy).unwrap();
        let policies = manager.list_policies();
        assert_eq!(policies.len(), 1);
        assert_eq!(policies[0].name, "eng-staging");
    }

    #[test]
    fn test_remove_policy() {
        let store = TensorStore::new();
        let manager = PolicyManager::new();

        let policy = PolicyTemplate {
            name: "test".to_string(),
            match_pattern: "*".to_string(),
            secret_pattern: "*".to_string(),
            permission: Permission::Read,
            ttl_ms: None,
        };

        manager.add_policy(&store, policy).unwrap();
        assert!(manager.get_policy("test").is_some());

        manager.remove_policy(&store, "test").unwrap();
        assert!(manager.get_policy("test").is_none());
    }

    #[test]
    fn test_evaluate_policies() {
        let store = TensorStore::new();
        let manager = PolicyManager::new();

        manager
            .add_policy(
                &store,
                PolicyTemplate {
                    name: "eng".to_string(),
                    match_pattern: "team:engineering/*".to_string(),
                    secret_pattern: "staging/*".to_string(),
                    permission: Permission::Read,
                    ttl_ms: None,
                },
            )
            .unwrap();

        let matches = manager.evaluate_policies("team:engineering/alice");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].policy_name, "eng");

        let no_match = manager.evaluate_policies("team:sales/bob");
        assert!(no_match.is_empty());
    }

    #[test]
    fn test_check_policy_access() {
        let store = TensorStore::new();
        let manager = PolicyManager::new();

        manager
            .add_policy(
                &store,
                PolicyTemplate {
                    name: "dev-read".to_string(),
                    match_pattern: "dev:*".to_string(),
                    secret_pattern: "config/*".to_string(),
                    permission: Permission::Read,
                    ttl_ms: Some(3600_000),
                },
            )
            .unwrap();

        let access = manager.check_policy_access("dev:alice", "config/db");
        assert!(access.is_some());
        let (perm, ttl) = access.unwrap();
        assert_eq!(perm, Permission::Read);
        assert_eq!(ttl, Some(3600_000));

        let no_access = manager.check_policy_access("dev:alice", "production/db");
        assert!(no_access.is_none());
    }

    #[test]
    fn test_highest_permission_wins() {
        let store = TensorStore::new();
        let manager = PolicyManager::new();

        manager
            .add_policy(
                &store,
                PolicyTemplate {
                    name: "read".to_string(),
                    match_pattern: "user:*".to_string(),
                    secret_pattern: "shared/*".to_string(),
                    permission: Permission::Read,
                    ttl_ms: None,
                },
            )
            .unwrap();

        manager
            .add_policy(
                &store,
                PolicyTemplate {
                    name: "write".to_string(),
                    match_pattern: "user:*".to_string(),
                    secret_pattern: "shared/*".to_string(),
                    permission: Permission::Write,
                    ttl_ms: None,
                },
            )
            .unwrap();

        let access = manager.check_policy_access("user:alice", "shared/key");
        assert!(access.is_some());
        let (perm, _) = access.unwrap();
        assert_eq!(perm, Permission::Write);
    }

    #[test]
    fn test_persistence() {
        let store = TensorStore::new();
        let manager = PolicyManager::new();

        manager
            .add_policy(
                &store,
                PolicyTemplate {
                    name: "persist".to_string(),
                    match_pattern: "user:*".to_string(),
                    secret_pattern: "key/*".to_string(),
                    permission: Permission::Admin,
                    ttl_ms: None,
                },
            )
            .unwrap();

        let loaded = PolicyManager::load(&store);
        let p = loaded.get_policy("persist").unwrap();
        assert_eq!(p.permission, Permission::Admin);
    }

    #[test]
    fn test_get_nonexistent_policy() {
        let manager = PolicyManager::new();
        assert!(manager.get_policy("nope").is_none());
    }

    #[test]
    fn test_policy_with_ttl() {
        let store = TensorStore::new();
        let manager = PolicyManager::new();

        manager
            .add_policy(
                &store,
                PolicyTemplate {
                    name: "ttl-policy".to_string(),
                    match_pattern: "*".to_string(),
                    secret_pattern: "*".to_string(),
                    permission: Permission::Read,
                    ttl_ms: Some(60_000),
                },
            )
            .unwrap();

        let p = manager.get_policy("ttl-policy").unwrap();
        assert_eq!(p.ttl_ms, Some(60_000));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let template = PolicyTemplate {
            name: "test".to_string(),
            match_pattern: "user:*".to_string(),
            secret_pattern: "secret/*".to_string(),
            permission: Permission::Write,
            ttl_ms: Some(1000),
        };
        let json = serde_json::to_string(&template).unwrap();
        let deser: PolicyTemplate = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.name, "test");
        assert_eq!(deser.permission, Permission::Write);
    }

    #[test]
    fn test_empty_policy_list() {
        let manager = PolicyManager::new();
        assert!(manager.list_policies().is_empty());
    }

    #[test]
    fn test_update_existing_policy() {
        let store = TensorStore::new();
        let manager = PolicyManager::new();

        manager
            .add_policy(
                &store,
                PolicyTemplate {
                    name: "p1".to_string(),
                    match_pattern: "a:*".to_string(),
                    secret_pattern: "b:*".to_string(),
                    permission: Permission::Read,
                    ttl_ms: None,
                },
            )
            .unwrap();

        // Update with same name
        manager
            .add_policy(
                &store,
                PolicyTemplate {
                    name: "p1".to_string(),
                    match_pattern: "c:*".to_string(),
                    secret_pattern: "d:*".to_string(),
                    permission: Permission::Admin,
                    ttl_ms: None,
                },
            )
            .unwrap();

        let p = manager.get_policy("p1").unwrap();
        assert_eq!(p.match_pattern, "c:*");
        assert_eq!(p.permission, Permission::Admin);
    }
}
