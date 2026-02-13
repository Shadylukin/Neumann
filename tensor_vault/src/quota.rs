// SPDX-License-Identifier: MIT OR Apache-2.0
//! Per-namespace resource quotas and usage tracking.

use std::time::{SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{Result, VaultError};

/// Storage prefix for quota configs.
const QUOTA_PREFIX: &str = "_vquota:";
/// Storage prefix for usage tracking.
const USAGE_PREFIX: &str = "_vusage:";

/// Resource limits for a namespace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuota {
    /// Maximum number of secrets in this namespace.
    pub max_secrets: u64,
    /// Maximum total storage in bytes.
    pub max_storage_bytes: u64,
    /// Maximum operations per hour.
    pub max_ops_per_hour: u64,
}

/// Current resource usage for a namespace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// Current number of secrets.
    pub secret_count: u64,
    /// Current storage bytes used.
    pub storage_bytes: u64,
    /// Operations performed in the current hour.
    pub ops_this_hour: u64,
    /// When the hourly counter was last reset (unix millis).
    pub last_reset_ms: i64,
}

/// Thread-safe quota manager backed by persistent storage.
pub struct QuotaManager {
    quotas: DashMap<String, ResourceQuota>,
    usage: DashMap<String, ResourceUsage>,
}

impl Default for QuotaManager {
    fn default() -> Self {
        Self {
            quotas: DashMap::new(),
            usage: DashMap::new(),
        }
    }
}

impl QuotaManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load quotas and usage from storage.
    pub fn load(store: &TensorStore) -> Self {
        let manager = Self::new();

        // Load quotas
        for key in store.scan(QUOTA_PREFIX) {
            if let Some(ns) = key.strip_prefix(QUOTA_PREFIX) {
                if let Ok(tensor) = store.get(&key) {
                    if let Some(quota) = deserialize_quota(&tensor) {
                        manager.quotas.insert(ns.to_string(), quota);
                    }
                }
            }
        }

        // Load usage
        for key in store.scan(USAGE_PREFIX) {
            if let Some(ns) = key.strip_prefix(USAGE_PREFIX) {
                if let Ok(tensor) = store.get(&key) {
                    let usage = deserialize_usage(&tensor);
                    manager.usage.insert(ns.to_string(), usage);
                }
            }
        }

        manager
    }

    /// Set a quota for a namespace.
    pub fn set_quota(
        &self,
        store: &TensorStore,
        namespace: &str,
        quota: ResourceQuota,
    ) -> Result<()> {
        let key = format!("{QUOTA_PREFIX}{namespace}");
        let tensor = serialize_quota(&quota);
        store
            .put(&key, tensor)
            .map_err(|e| VaultError::StorageError(e.to_string()))?;
        self.quotas.insert(namespace.to_string(), quota);
        Ok(())
    }

    /// Get the quota for a namespace.
    pub fn get_quota(&self, namespace: &str) -> Option<ResourceQuota> {
        self.quotas.get(namespace).map(|q| q.clone())
    }

    /// Get current usage for a namespace.
    pub fn get_usage(&self, namespace: &str) -> ResourceUsage {
        self.usage
            .get(namespace)
            .map(|u| u.clone())
            .unwrap_or_default()
    }

    /// Remove a quota for a namespace.
    pub fn remove_quota(&self, store: &TensorStore, namespace: &str) -> Result<()> {
        let key = format!("{QUOTA_PREFIX}{namespace}");
        store.delete(&key).ok();
        self.quotas.remove(namespace);
        Ok(())
    }

    /// Check if an operation would exceed the quota. Returns Ok(()) if allowed.
    pub fn check_quota(
        &self,
        namespace: &str,
        additional_secrets: u64,
        additional_bytes: u64,
    ) -> Result<()> {
        let Some(quota) = self.quotas.get(namespace) else {
            return Ok(()); // No quota set = unlimited
        };

        let mut usage = self
            .usage
            .get(namespace)
            .map(|u| u.clone())
            .unwrap_or_default();

        // Auto-reset hourly counter
        let now = now_ms();
        let one_hour_ms = 3_600_000;
        if now - usage.last_reset_ms >= one_hour_ms {
            usage.ops_this_hour = 0;
            usage.last_reset_ms = now;
        }

        if usage.secret_count + additional_secrets > quota.max_secrets {
            return Err(VaultError::QuotaExceeded(format!(
                "namespace '{namespace}': would exceed max secrets ({} + {} > {})",
                usage.secret_count, additional_secrets, quota.max_secrets
            )));
        }

        if usage.storage_bytes + additional_bytes > quota.max_storage_bytes {
            return Err(VaultError::QuotaExceeded(format!(
                "namespace '{namespace}': would exceed max storage",
            )));
        }

        if usage.ops_this_hour + 1 > quota.max_ops_per_hour {
            return Err(VaultError::QuotaExceeded(format!(
                "namespace '{namespace}': exceeded ops per hour ({})",
                quota.max_ops_per_hour
            )));
        }

        Ok(())
    }

    /// Record a secret being added to a namespace.
    pub fn record_secret_added(&self, store: &TensorStore, namespace: &str, bytes: u64) {
        let mut usage = self
            .usage
            .entry(namespace.to_string())
            .or_insert_with(|| ResourceUsage {
                last_reset_ms: now_ms(),
                ..Default::default()
            });

        let now = now_ms();
        let one_hour_ms = 3_600_000;
        if now - usage.last_reset_ms >= one_hour_ms {
            usage.ops_this_hour = 0;
            usage.last_reset_ms = now;
        }

        usage.secret_count += 1;
        usage.storage_bytes += bytes;
        usage.ops_this_hour += 1;

        let snapshot = usage.clone();
        drop(usage);
        Self::persist_usage(store, namespace, &snapshot);
    }

    /// Record a secret being removed from a namespace.
    pub fn record_secret_removed(&self, store: &TensorStore, namespace: &str, bytes: u64) {
        let mut usage = self.usage.entry(namespace.to_string()).or_default();

        usage.secret_count = usage.secret_count.saturating_sub(1);
        usage.storage_bytes = usage.storage_bytes.saturating_sub(bytes);
        usage.ops_this_hour += 1;

        let snapshot = usage.clone();
        drop(usage);
        Self::persist_usage(store, namespace, &snapshot);
    }

    /// Record an operation.
    pub fn record_operation(&self, store: &TensorStore, namespace: &str) {
        let mut usage = self
            .usage
            .entry(namespace.to_string())
            .or_insert_with(|| ResourceUsage {
                last_reset_ms: now_ms(),
                ..Default::default()
            });

        let now = now_ms();
        let one_hour_ms = 3_600_000;
        if now - usage.last_reset_ms >= one_hour_ms {
            usage.ops_this_hour = 0;
            usage.last_reset_ms = now;
        }
        usage.ops_this_hour += 1;

        let snapshot = usage.clone();
        drop(usage);
        Self::persist_usage(store, namespace, &snapshot);
    }

    fn persist_usage(store: &TensorStore, namespace: &str, usage: &ResourceUsage) {
        let key = format!("{USAGE_PREFIX}{namespace}");
        let tensor = serialize_usage(usage);
        let _ = store.put(&key, tensor);
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn i64_from_u64(val: u64) -> i64 {
    i64::try_from(val).unwrap_or(i64::MAX)
}

fn serialize_quota(quota: &ResourceQuota) -> TensorData {
    let mut t = TensorData::new();
    t.set(
        "_max_secrets",
        TensorValue::Scalar(ScalarValue::Int(i64_from_u64(quota.max_secrets))),
    );
    t.set(
        "_max_storage",
        TensorValue::Scalar(ScalarValue::Int(i64_from_u64(quota.max_storage_bytes))),
    );
    t.set(
        "_max_ops",
        TensorValue::Scalar(ScalarValue::Int(i64_from_u64(quota.max_ops_per_hour))),
    );
    t
}

fn deserialize_quota(tensor: &TensorData) -> Option<ResourceQuota> {
    let max_secrets = match tensor.get("_max_secrets") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => u64::try_from(*v).unwrap_or(0),
        _ => return None,
    };
    let max_storage_bytes = match tensor.get("_max_storage") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => u64::try_from(*v).unwrap_or(0),
        _ => return None,
    };
    let max_ops_per_hour = match tensor.get("_max_ops") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => u64::try_from(*v).unwrap_or(0),
        _ => return None,
    };
    Some(ResourceQuota {
        max_secrets,
        max_storage_bytes,
        max_ops_per_hour,
    })
}

fn serialize_usage(usage: &ResourceUsage) -> TensorData {
    let mut t = TensorData::new();
    t.set(
        "_secrets",
        TensorValue::Scalar(ScalarValue::Int(i64_from_u64(usage.secret_count))),
    );
    t.set(
        "_bytes",
        TensorValue::Scalar(ScalarValue::Int(i64_from_u64(usage.storage_bytes))),
    );
    t.set(
        "_ops",
        TensorValue::Scalar(ScalarValue::Int(i64_from_u64(usage.ops_this_hour))),
    );
    t.set(
        "_reset",
        TensorValue::Scalar(ScalarValue::Int(usage.last_reset_ms)),
    );
    t
}

fn deserialize_usage(tensor: &TensorData) -> ResourceUsage {
    let secret_count = match tensor.get("_secrets") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => u64::try_from(*v).unwrap_or(0),
        _ => 0,
    };
    let storage_bytes = match tensor.get("_bytes") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => u64::try_from(*v).unwrap_or(0),
        _ => 0,
    };
    let ops_this_hour = match tensor.get("_ops") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => u64::try_from(*v).unwrap_or(0),
        _ => 0,
    };
    let last_reset_ms = match tensor.get("_reset") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => *v,
        _ => 0,
    };
    ResourceUsage {
        secret_count,
        storage_bytes,
        ops_this_hour,
        last_reset_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get_quota() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        let quota = ResourceQuota {
            max_secrets: 100,
            max_storage_bytes: 1_000_000,
            max_ops_per_hour: 1000,
        };

        manager.set_quota(&store, "team-a", quota.clone()).unwrap();
        let got = manager.get_quota("team-a").unwrap();
        assert_eq!(got.max_secrets, 100);
        assert_eq!(got.max_storage_bytes, 1_000_000);
    }

    #[test]
    fn test_get_quota_nonexistent() {
        let manager = QuotaManager::new();
        assert!(manager.get_quota("nonexistent").is_none());
    }

    #[test]
    fn test_quota_exceeded_secrets() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        let quota = ResourceQuota {
            max_secrets: 2,
            max_storage_bytes: 1_000_000,
            max_ops_per_hour: 1000,
        };
        manager.set_quota(&store, "ns", quota).unwrap();
        manager.record_secret_added(&store, "ns", 100);
        manager.record_secret_added(&store, "ns", 100);

        let result = manager.check_quota("ns", 1, 0);
        assert!(matches!(result, Err(VaultError::QuotaExceeded(_))));
    }

    #[test]
    fn test_quota_not_exceeded() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        let quota = ResourceQuota {
            max_secrets: 10,
            max_storage_bytes: 1_000_000,
            max_ops_per_hour: 1000,
        };
        manager.set_quota(&store, "ns", quota).unwrap();
        manager.record_secret_added(&store, "ns", 100);

        assert!(manager.check_quota("ns", 1, 0).is_ok());
    }

    #[test]
    fn test_no_quota_means_unlimited() {
        let manager = QuotaManager::new();
        assert!(manager.check_quota("any", 1000, 1_000_000).is_ok());
    }

    #[test]
    fn test_remove_quota() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        let quota = ResourceQuota {
            max_secrets: 1,
            max_storage_bytes: 100,
            max_ops_per_hour: 10,
        };
        manager.set_quota(&store, "ns", quota).unwrap();
        assert!(manager.get_quota("ns").is_some());

        manager.remove_quota(&store, "ns").unwrap();
        assert!(manager.get_quota("ns").is_none());
    }

    #[test]
    fn test_usage_tracking() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        manager.record_secret_added(&store, "ns", 500);
        manager.record_secret_added(&store, "ns", 300);

        let usage = manager.get_usage("ns");
        assert_eq!(usage.secret_count, 2);
        assert_eq!(usage.storage_bytes, 800);
    }

    #[test]
    fn test_secret_removed_tracking() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        manager.record_secret_added(&store, "ns", 500);
        manager.record_secret_added(&store, "ns", 300);
        manager.record_secret_removed(&store, "ns", 300);

        let usage = manager.get_usage("ns");
        assert_eq!(usage.secret_count, 1);
        assert_eq!(usage.storage_bytes, 500);
    }

    #[test]
    fn test_default_usage() {
        let manager = QuotaManager::new();
        let usage = manager.get_usage("nonexistent");
        assert_eq!(usage.secret_count, 0);
        assert_eq!(usage.storage_bytes, 0);
        assert_eq!(usage.ops_this_hour, 0);
    }

    #[test]
    fn test_quota_persistence() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        let quota = ResourceQuota {
            max_secrets: 50,
            max_storage_bytes: 500_000,
            max_ops_per_hour: 500,
        };
        manager.set_quota(&store, "persistent", quota).unwrap();
        manager.record_secret_added(&store, "persistent", 1000);

        // Reload from store
        let loaded = QuotaManager::load(&store);
        let q = loaded.get_quota("persistent").unwrap();
        assert_eq!(q.max_secrets, 50);

        let u = loaded.get_usage("persistent");
        assert_eq!(u.secret_count, 1);
    }

    #[test]
    fn test_ops_per_hour_exceeded() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        let quota = ResourceQuota {
            max_secrets: 100,
            max_storage_bytes: 1_000_000,
            max_ops_per_hour: 2,
        };
        manager.set_quota(&store, "ns", quota).unwrap();
        manager.record_operation(&store, "ns");
        manager.record_operation(&store, "ns");

        let result = manager.check_quota("ns", 0, 0);
        assert!(matches!(result, Err(VaultError::QuotaExceeded(_))));
    }

    #[test]
    fn test_storage_exceeded() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        let quota = ResourceQuota {
            max_secrets: 100,
            max_storage_bytes: 1000,
            max_ops_per_hour: 1000,
        };
        manager.set_quota(&store, "ns", quota).unwrap();
        manager.record_secret_added(&store, "ns", 900);

        let result = manager.check_quota("ns", 0, 200);
        assert!(matches!(result, Err(VaultError::QuotaExceeded(_))));
    }

    #[test]
    fn test_multiple_namespaces() {
        let store = TensorStore::new();
        let manager = QuotaManager::new();

        let q1 = ResourceQuota {
            max_secrets: 10,
            max_storage_bytes: 10_000,
            max_ops_per_hour: 100,
        };
        let q2 = ResourceQuota {
            max_secrets: 5,
            max_storage_bytes: 5_000,
            max_ops_per_hour: 50,
        };

        manager.set_quota(&store, "ns1", q1).unwrap();
        manager.set_quota(&store, "ns2", q2).unwrap();
        manager.record_secret_added(&store, "ns1", 100);

        assert_eq!(manager.get_usage("ns1").secret_count, 1);
        assert_eq!(manager.get_usage("ns2").secret_count, 0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let quota = ResourceQuota {
            max_secrets: 42,
            max_storage_bytes: 12345,
            max_ops_per_hour: 99,
        };
        let json = serde_json::to_string(&quota).unwrap();
        let deserialized: ResourceQuota = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.max_secrets, 42);
    }
}
