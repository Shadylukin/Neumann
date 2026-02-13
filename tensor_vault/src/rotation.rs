// SPDX-License-Identifier: MIT OR Apache-2.0
//! Automated rotation policy management.

use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{
    dynamic::{PasswordConfig, TokenConfig},
    Result, VaultError,
};

/// Storage prefix for rotation policies.
const ROT_PREFIX: &str = "_vrot:";

/// A declarative rotation policy for a secret.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationPolicy {
    /// The secret key this policy applies to.
    pub secret_key: String,
    /// Rotation interval in milliseconds.
    pub interval_ms: i64,
    /// When the secret was last rotated (unix millis).
    pub last_rotated_ms: i64,
    /// How to generate the new value on rotation.
    pub generator: RotationGenerator,
    /// How far in advance to warn about pending rotation (millis).
    pub notify_before_ms: i64,
}

/// How to generate a new secret value on rotation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationGenerator {
    /// No auto-generation; manual rotation required.
    None,
    /// Auto-generate a password.
    Password(PasswordConfig),
    /// Auto-generate a token.
    Token(TokenConfig),
}

/// A rotation that is due or overdue.
#[derive(Debug, Clone)]
pub struct PendingRotation {
    /// The secret key that needs rotation.
    pub secret_key: String,
    /// How many milliseconds overdue (0 = due now, negative = upcoming).
    pub overdue_ms: i64,
    /// The generator to use.
    pub generator: RotationGenerator,
}

/// Thread-safe rotation policy manager.
pub struct RotationPolicyManager {
    policies: DashMap<String, RotationPolicy>,
}

impl Default for RotationPolicyManager {
    fn default() -> Self {
        Self {
            policies: DashMap::new(),
        }
    }
}

impl RotationPolicyManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load policies from storage.
    pub fn load(store: &TensorStore) -> Self {
        let manager = Self::new();
        for key in store.scan(ROT_PREFIX) {
            if let Some(secret_key) = key.strip_prefix(ROT_PREFIX) {
                if let Ok(tensor) = store.get(&key) {
                    if let Some(policy) = deserialize_policy(secret_key, &tensor) {
                        manager.policies.insert(secret_key.to_string(), policy);
                    }
                }
            }
        }
        manager
    }

    /// Set a rotation policy for a secret.
    pub fn set_policy(
        &self,
        store: &TensorStore,
        obfuscated_key: &str,
        policy: RotationPolicy,
    ) -> Result<()> {
        let key = format!("{ROT_PREFIX}{obfuscated_key}");
        let tensor = serialize_policy(&policy);
        store
            .put(&key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;
        self.policies.insert(obfuscated_key.to_string(), policy);
        Ok(())
    }

    /// Get the rotation policy for a secret.
    pub fn get_policy(&self, obfuscated_key: &str) -> Option<RotationPolicy> {
        self.policies.get(obfuscated_key).map(|p| p.clone())
    }

    /// Remove a rotation policy.
    pub fn remove_policy(&self, store: &TensorStore, obfuscated_key: &str) {
        let key = format!("{ROT_PREFIX}{obfuscated_key}");
        store.delete(&key).ok();
        self.policies.remove(obfuscated_key);
    }

    /// List all rotation policies.
    pub fn list_policies(&self) -> Vec<RotationPolicy> {
        self.policies.iter().map(|e| e.value().clone()).collect()
    }

    /// Check for pending (due or overdue) rotations.
    pub fn check_pending(&self) -> Vec<PendingRotation> {
        let now = now_ms();
        let mut pending = Vec::new();

        for entry in &self.policies {
            let policy = entry.value();
            let due_at = policy.last_rotated_ms + policy.interval_ms;
            let warn_at = due_at - policy.notify_before_ms;

            if now >= warn_at {
                pending.push(PendingRotation {
                    secret_key: policy.secret_key.clone(),
                    overdue_ms: now - due_at,
                    generator: policy.generator.clone(),
                });
            }
        }

        pending
    }

    /// Update the last_rotated_ms timestamp for a policy.
    pub fn mark_rotated(&self, store: &TensorStore, obfuscated_key: &str) -> Result<()> {
        if let Some(mut entry) = self.policies.get_mut(obfuscated_key) {
            entry.last_rotated_ms = now_ms();
            let key = format!("{ROT_PREFIX}{obfuscated_key}");
            let tensor = serialize_policy(&entry);
            store
                .put(&key, tensor)
                .map_err(|e| VaultError::StorageError(e.to_string()))?;
        }
        Ok(())
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn serialize_policy(policy: &RotationPolicy) -> TensorData {
    let mut t = TensorData::new();
    t.set(
        "_secret_key",
        TensorValue::Scalar(ScalarValue::String(policy.secret_key.clone())),
    );
    t.set(
        "_interval",
        TensorValue::Scalar(ScalarValue::Int(policy.interval_ms)),
    );
    t.set(
        "_last_rotated",
        TensorValue::Scalar(ScalarValue::Int(policy.last_rotated_ms)),
    );
    t.set(
        "_notify_before",
        TensorValue::Scalar(ScalarValue::Int(policy.notify_before_ms)),
    );

    let gen_json = serde_json::to_string(&policy.generator).unwrap_or_default();
    t.set(
        "_generator",
        TensorValue::Scalar(ScalarValue::String(gen_json)),
    );
    t
}

fn deserialize_policy(secret_key: &str, tensor: &TensorData) -> Option<RotationPolicy> {
    let sk = match tensor.get("_secret_key") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
        _ => secret_key.to_string(),
    };
    let interval_ms = match tensor.get("_interval") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
        _ => return None,
    };
    let last_rotated_ms = match tensor.get("_last_rotated") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
        _ => 0,
    };
    let notify_before_ms = match tensor.get("_notify_before") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
        _ => 0,
    };
    let generator = match tensor.get("_generator") {
        Some(TensorValue::Scalar(ScalarValue::String(s))) => {
            serde_json::from_str(s).unwrap_or(RotationGenerator::None)
        },
        _ => RotationGenerator::None,
    };

    Some(RotationPolicy {
        secret_key: sk,
        interval_ms,
        last_rotated_ms,
        generator,
        notify_before_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get_policy() {
        let store = TensorStore::new();
        let manager = RotationPolicyManager::new();

        let policy = RotationPolicy {
            secret_key: "db/password".to_string(),
            interval_ms: 86_400_000,
            last_rotated_ms: now_ms(),
            generator: RotationGenerator::None,
            notify_before_ms: 3_600_000,
        };

        manager.set_policy(&store, "obf_key", policy).unwrap();
        let got = manager.get_policy("obf_key").unwrap();
        assert_eq!(got.secret_key, "db/password");
        assert_eq!(got.interval_ms, 86_400_000);
    }

    #[test]
    fn test_remove_policy() {
        let store = TensorStore::new();
        let manager = RotationPolicyManager::new();

        let policy = RotationPolicy {
            secret_key: "key".to_string(),
            interval_ms: 1000,
            last_rotated_ms: 0,
            generator: RotationGenerator::None,
            notify_before_ms: 0,
        };

        manager.set_policy(&store, "k1", policy).unwrap();
        manager.remove_policy(&store, "k1");
        assert!(manager.get_policy("k1").is_none());
    }

    #[test]
    fn test_list_policies() {
        let store = TensorStore::new();
        let manager = RotationPolicyManager::new();

        for i in 0..3 {
            let policy = RotationPolicy {
                secret_key: format!("key_{i}"),
                interval_ms: 1000,
                last_rotated_ms: 0,
                generator: RotationGenerator::None,
                notify_before_ms: 0,
            };
            manager
                .set_policy(&store, &format!("k{i}"), policy)
                .unwrap();
        }

        assert_eq!(manager.list_policies().len(), 3);
    }

    #[test]
    fn test_check_pending_overdue() {
        let store = TensorStore::new();
        let manager = RotationPolicyManager::new();

        let policy = RotationPolicy {
            secret_key: "overdue".to_string(),
            interval_ms: 1,     // 1ms interval
            last_rotated_ms: 0, // Long ago
            generator: RotationGenerator::None,
            notify_before_ms: 0,
        };
        manager.set_policy(&store, "k", policy).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(5));
        let pending = manager.check_pending();
        assert_eq!(pending.len(), 1);
        assert!(pending[0].overdue_ms > 0);
    }

    #[test]
    fn test_check_pending_not_due() {
        let store = TensorStore::new();
        let manager = RotationPolicyManager::new();

        let policy = RotationPolicy {
            secret_key: "future".to_string(),
            interval_ms: 86_400_000, // 1 day
            last_rotated_ms: now_ms(),
            generator: RotationGenerator::None,
            notify_before_ms: 0,
        };
        manager.set_policy(&store, "k", policy).unwrap();

        let pending = manager.check_pending();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_mark_rotated() {
        let store = TensorStore::new();
        let manager = RotationPolicyManager::new();

        let policy = RotationPolicy {
            secret_key: "key".to_string(),
            interval_ms: 1000,
            last_rotated_ms: 0,
            generator: RotationGenerator::None,
            notify_before_ms: 0,
        };
        manager.set_policy(&store, "k", policy).unwrap();

        let before = now_ms();
        manager.mark_rotated(&store, "k").unwrap();

        let updated = manager.get_policy("k").unwrap();
        assert!(updated.last_rotated_ms >= before);
    }

    #[test]
    fn test_password_generator() {
        let policy = RotationPolicy {
            secret_key: "key".to_string(),
            interval_ms: 1000,
            last_rotated_ms: 0,
            generator: RotationGenerator::Password(PasswordConfig::default()),
            notify_before_ms: 0,
        };
        assert!(matches!(policy.generator, RotationGenerator::Password(_)));
    }

    #[test]
    fn test_token_generator() {
        let policy = RotationPolicy {
            secret_key: "key".to_string(),
            interval_ms: 1000,
            last_rotated_ms: 0,
            generator: RotationGenerator::Token(TokenConfig::default()),
            notify_before_ms: 0,
        };
        assert!(matches!(policy.generator, RotationGenerator::Token(_)));
    }

    #[test]
    fn test_persistence() {
        let store = TensorStore::new();
        let manager = RotationPolicyManager::new();

        let policy = RotationPolicy {
            secret_key: "persisted".to_string(),
            interval_ms: 5000,
            last_rotated_ms: 12345,
            generator: RotationGenerator::None,
            notify_before_ms: 1000,
        };
        manager.set_policy(&store, "pk", policy).unwrap();

        let loaded = RotationPolicyManager::load(&store);
        let p = loaded.get_policy("pk").unwrap();
        assert_eq!(p.secret_key, "persisted");
        assert_eq!(p.interval_ms, 5000);
    }

    #[test]
    fn test_notify_before() {
        let store = TensorStore::new();
        let manager = RotationPolicyManager::new();

        // Rotation due in 1 hour but notify_before is 2 hours
        let policy = RotationPolicy {
            secret_key: "early_warn".to_string(),
            interval_ms: 3_600_000,
            last_rotated_ms: now_ms(),
            generator: RotationGenerator::None,
            notify_before_ms: 7_200_000, // 2 hours before
        };
        manager.set_policy(&store, "ew", policy).unwrap();

        let pending = manager.check_pending();
        assert_eq!(pending.len(), 1); // Should appear as pending due to early warning
    }

    #[test]
    fn test_serialization_roundtrip() {
        let policy = RotationPolicy {
            secret_key: "test".to_string(),
            interval_ms: 1000,
            last_rotated_ms: 500,
            generator: RotationGenerator::Password(PasswordConfig::default()),
            notify_before_ms: 100,
        };
        let json = serde_json::to_string(&policy).unwrap();
        let deser: RotationPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.secret_key, "test");
    }

    #[test]
    fn test_get_nonexistent_policy() {
        let manager = RotationPolicyManager::new();
        assert!(manager.get_policy("nope").is_none());
    }
}
