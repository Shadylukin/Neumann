// SPDX-License-Identifier: MIT OR Apache-2.0
//! 3D access tensor built from audit log data.
//!
//! Represents access patterns as a tensor H[entity, secret, time_bucket]
//! for temporal analysis, seasonal pattern extraction, and drift detection.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::audit::AuditLog;
use crate::vault::Vault;
use crate::Result;

/// Configuration for building an access tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessTensorConfig {
    /// Time bucket granularity in milliseconds (default: 3,600,000 = 1 hour).
    pub bucket_size_ms: i64,
    /// Number of time buckets (default: 168 = 1 week of hourly buckets).
    pub num_buckets: usize,
    /// Start time in unix milliseconds. If `None`, computed from `num_buckets` ago.
    pub start_time_ms: Option<i64>,
    /// Filter to specific operation types. `None` means all operations.
    pub operations: Option<Vec<String>>,
}

impl Default for AccessTensorConfig {
    fn default() -> Self {
        Self {
            bucket_size_ms: 3_600_000,
            num_buckets: 168,
            start_time_ms: None,
            operations: None,
        }
    }
}

/// 3D access tensor: entities x secrets x time_buckets.
pub struct AccessTensor {
    pub(crate) entity_index: HashMap<String, usize>,
    pub(crate) secret_index: HashMap<String, usize>,
    pub(crate) data: Vec<f32>,
    pub(crate) dimensions: (usize, usize, usize),
    pub(crate) config: AccessTensorConfig,
}

/// Per-entity access behavior profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityAccessProfile {
    /// Entity identifier.
    pub entity: String,
    /// Mean access rate across all time buckets.
    pub mean_rate: f64,
    /// Standard deviation of access rate.
    pub rate_stddev: f64,
    /// Time bucket with the most accesses.
    pub peak_bucket: usize,
    /// Shannon entropy of access distribution.
    pub entropy: f64,
    /// Total number of accesses.
    pub total_accesses: u64,
}

/// Per-secret access pattern profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecretAccessProfile {
    /// Secret key.
    pub secret: String,
    /// Number of distinct entities that accessed this secret.
    pub unique_accessors: usize,
    /// Time bucket with the most accesses.
    pub peak_bucket: usize,
    /// Burstiness: (max_bucket / mean_bucket) - 1. Zero if uniform.
    pub burstiness: f64,
}

impl AccessTensor {
    /// Build an access tensor from the vault's audit log.
    pub fn from_vault(vault: &Vault, config: AccessTensorConfig) -> Result<Self> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let start = config
            .start_time_ms
            .unwrap_or_else(|| now_ms - config.bucket_size_ms * config.num_buckets as i64);

        let audit = AuditLog::new(&vault.store, Some(*vault.audit_key()));
        let entries = audit.since(start);

        // Collect unique entities and secrets
        let mut entity_set: Vec<String> = Vec::new();
        let mut secret_set: Vec<String> = Vec::new();
        let mut entity_map: HashMap<String, usize> = HashMap::new();
        let mut secret_map: HashMap<String, usize> = HashMap::new();

        for entry in &entries {
            if let Some(ref ops) = config.operations {
                let op_str = format!("{:?}", entry.operation);
                if !ops.iter().any(|o| op_str.contains(o)) {
                    continue;
                }
            }
            if !entity_map.contains_key(&entry.entity) {
                let idx = entity_set.len();
                entity_set.push(entry.entity.clone());
                entity_map.insert(entry.entity.clone(), idx);
            }
            if !secret_map.contains_key(&entry.secret_key) {
                let idx = secret_set.len();
                secret_set.push(entry.secret_key.clone());
                secret_map.insert(entry.secret_key.clone(), idx);
            }
        }

        let n_entities = entity_set.len();
        let n_secrets = secret_set.len();
        let n_buckets = config.num_buckets;
        let total = n_entities * n_secrets * n_buckets;

        let mut data = vec![0.0_f32; total];

        for entry in &entries {
            if let Some(ref ops) = config.operations {
                let op_str = format!("{:?}", entry.operation);
                if !ops.iter().any(|o| op_str.contains(o)) {
                    continue;
                }
            }
            let Some(&eidx) = entity_map.get(&entry.entity) else {
                continue;
            };
            let Some(&sidx) = secret_map.get(&entry.secret_key) else {
                continue;
            };
            #[allow(clippy::cast_sign_loss)] // bucket index is guaranteed non-negative
            let bucket = ((entry.timestamp - start) / config.bucket_size_ms) as usize;
            if bucket < n_buckets {
                let idx = eidx * n_secrets * n_buckets + sidx * n_buckets + bucket;
                data[idx] += 1.0;
            }
        }

        Ok(Self {
            entity_index: entity_map,
            secret_index: secret_map,
            data,
            dimensions: (n_entities, n_secrets, n_buckets),
            config,
        })
    }

    /// Get access count for a specific (entity, secret, bucket) triple.
    pub fn get(&self, entity: &str, secret: &str, bucket: usize) -> f32 {
        let Some(&eidx) = self.entity_index.get(entity) else {
            return 0.0;
        };
        let Some(&sidx) = self.secret_index.get(secret) else {
            return 0.0;
        };
        let (_, n_secrets, n_buckets) = self.dimensions;
        if bucket >= n_buckets {
            return 0.0;
        }
        self.data[eidx * n_secrets * n_buckets + sidx * n_buckets + bucket]
    }

    /// Get time series for a specific entity-secret pair.
    pub fn time_series(&self, entity: &str, secret: &str) -> Vec<f32> {
        let Some(&eidx) = self.entity_index.get(entity) else {
            return Vec::new();
        };
        let Some(&sidx) = self.secret_index.get(secret) else {
            return Vec::new();
        };
        let (_, n_secrets, n_buckets) = self.dimensions;
        let start = eidx * n_secrets * n_buckets + sidx * n_buckets;
        self.data[start..start + n_buckets].to_vec()
    }

    /// Get the full access vector for an entity (all secrets, all buckets).
    pub fn entity_vector(&self, entity: &str) -> Vec<f32> {
        let Some(&eidx) = self.entity_index.get(entity) else {
            return Vec::new();
        };
        let (_, n_secrets, n_buckets) = self.dimensions;
        let len = n_secrets * n_buckets;
        let start = eidx * len;
        self.data[start..start + len].to_vec()
    }

    /// Get the full access vector for a secret (all entities, all buckets).
    pub fn secret_vector(&self, secret: &str) -> Vec<f32> {
        let Some(&sidx) = self.secret_index.get(secret) else {
            return Vec::new();
        };
        let (n_entities, n_secrets, n_buckets) = self.dimensions;
        let mut vec = Vec::with_capacity(n_entities * n_buckets);
        for eidx in 0..n_entities {
            let start = eidx * n_secrets * n_buckets + sidx * n_buckets;
            vec.extend_from_slice(&self.data[start..start + n_buckets]);
        }
        vec
    }

    /// Compute per-entity access profiles.
    pub fn entity_profiles(&self) -> Vec<EntityAccessProfile> {
        let (_, n_secrets, n_buckets) = self.dimensions;
        let mut profiles = Vec::new();

        for (entity, &eidx) in &self.entity_index {
            let len = n_secrets * n_buckets;
            let start = eidx * len;
            let slice = &self.data[start..start + len];

            // Per-bucket totals
            let mut bucket_totals = vec![0.0_f64; n_buckets];
            for sidx in 0..n_secrets {
                for b in 0..n_buckets {
                    bucket_totals[b] += f64::from(slice[sidx * n_buckets + b]);
                }
            }

            let total: f64 = bucket_totals.iter().sum();
            #[allow(clippy::cast_precision_loss)] // bucket count will never exceed 2^52
            let n = n_buckets as f64;
            let mean = total / n;

            let variance = bucket_totals
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / n;
            let stddev = variance.sqrt();

            let peak_bucket = bucket_totals
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);

            // Shannon entropy
            let entropy = if total > 0.0 {
                bucket_totals
                    .iter()
                    .filter(|&&v| v > 0.0)
                    .map(|v| {
                        let p = v / total;
                        -p * p.ln()
                    })
                    .sum()
            } else {
                0.0
            };

            #[allow(clippy::cast_sign_loss)]
            let total_accesses = total as u64;

            profiles.push(EntityAccessProfile {
                entity: entity.clone(),
                mean_rate: mean,
                rate_stddev: stddev,
                peak_bucket,
                entropy,
                total_accesses,
            });
        }

        profiles.sort_by(|a, b| b.total_accesses.cmp(&a.total_accesses));
        profiles
    }

    /// Compute per-secret access profiles.
    pub fn secret_profiles(&self) -> Vec<SecretAccessProfile> {
        let (n_entities, n_secrets, n_buckets) = self.dimensions;
        let mut profiles = Vec::new();

        for (secret, &sidx) in &self.secret_index {
            let mut bucket_totals = vec![0.0_f64; n_buckets];
            let mut accessor_count = 0_usize;

            for eidx in 0..n_entities {
                let start = eidx * n_secrets * n_buckets + sidx * n_buckets;
                let slice = &self.data[start..start + n_buckets];
                let entity_total: f32 = slice.iter().sum();
                if entity_total > 0.0 {
                    accessor_count += 1;
                }
                for (b, val) in slice.iter().enumerate() {
                    bucket_totals[b] += f64::from(*val);
                }
            }

            let total: f64 = bucket_totals.iter().sum();
            #[allow(clippy::cast_precision_loss)] // bucket count will never exceed 2^52
            let n = n_buckets as f64;
            let mean = total / n;
            let max_bucket = bucket_totals
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            let burstiness = if mean > 0.0 {
                (max_bucket / mean) - 1.0
            } else {
                0.0
            };

            let peak_bucket = bucket_totals
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(i, _)| i);

            profiles.push(SecretAccessProfile {
                secret: secret.clone(),
                unique_accessors: accessor_count,
                peak_bucket,
                burstiness,
            });
        }

        profiles.sort_by(|a, b| b.unique_accessors.cmp(&a.unique_accessors));
        profiles
    }

    /// Raw tensor data as a flat slice.
    pub fn raw_data(&self) -> &[f32] {
        &self.data
    }

    /// Tensor dimensions: (entities, secrets, time_buckets).
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.dimensions
    }

    /// List of entity names in the tensor.
    pub fn entities(&self) -> Vec<String> {
        let mut result: Vec<(String, usize)> = self
            .entity_index
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        result.sort_by_key(|(_, idx)| *idx);
        result.into_iter().map(|(k, _)| k).collect()
    }

    /// List of secret names in the tensor.
    pub fn secrets(&self) -> Vec<String> {
        let mut result: Vec<(String, usize)> = self
            .secret_index
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        result.sort_by_key(|(_, idx)| *idx);
        result.into_iter().map(|(k, _)| k).collect()
    }

    /// Access tensor config.
    pub fn config(&self) -> &AccessTensorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graph_engine::GraphEngine;
    use tensor_store::TensorStore;

    use super::*;
    use crate::VaultConfig;

    fn create_test_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(
            b"test_password",
            graph.clone(),
            store,
            VaultConfig::default(),
        )
        .unwrap()
    }

    fn record_audit(vault: &Vault, entity: &str, secret: &str) {
        let audit = AuditLog::new(&vault.store, Some(*vault.audit_key()));
        audit.record(entity, secret, &crate::audit::AuditOperation::Get);
    }

    #[test]
    fn test_tensor_empty_vault() {
        let vault = create_test_vault();
        let config = AccessTensorConfig {
            num_buckets: 10,
            ..AccessTensorConfig::default()
        };
        let tensor = AccessTensor::from_vault(&vault, config).unwrap();
        assert_eq!(tensor.dimensions(), (0, 0, 10));
        assert!(tensor.raw_data().is_empty());
    }

    #[test]
    fn test_tensor_single_entry() {
        let vault = create_test_vault();
        record_audit(&vault, "user:alice", "db/password");

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        // Use num_buckets = 11 so entries at ~now fall in bucket 10 (valid 0-10)
        let config = AccessTensorConfig {
            bucket_size_ms: 3_600_000,
            num_buckets: 11,
            start_time_ms: Some(now_ms - 3_600_000 * 10),
            operations: None,
        };
        let tensor = AccessTensor::from_vault(&vault, config).unwrap();
        assert_eq!(tensor.dimensions().0, 1); // 1 entity
        assert_eq!(tensor.dimensions().1, 1); // 1 secret

        // The entry should be somewhere in the tensor
        let total: f32 = tensor.raw_data().iter().sum();
        assert!((total - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tensor_multiple_buckets() {
        let vault = create_test_vault();

        // Record multiple entries
        for _ in 0..5 {
            record_audit(&vault, "user:alice", "db/password");
        }
        record_audit(&vault, "user:bob", "api/key");

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let config = AccessTensorConfig {
            bucket_size_ms: 3_600_000,
            num_buckets: 25,
            start_time_ms: Some(now_ms - 3_600_000 * 24),
            operations: None,
        };
        let tensor = AccessTensor::from_vault(&vault, config).unwrap();
        assert_eq!(tensor.dimensions().0, 2); // alice, bob
        assert_eq!(tensor.dimensions().1, 2); // db/password, api/key
    }

    #[test]
    fn test_tensor_entity_vector() {
        let vault = create_test_vault();
        record_audit(&vault, "user:alice", "secret1");
        record_audit(&vault, "user:alice", "secret2");

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let config = AccessTensorConfig {
            bucket_size_ms: 3_600_000,
            num_buckets: 11,
            start_time_ms: Some(now_ms - 3_600_000 * 10),
            operations: None,
        };
        let tensor = AccessTensor::from_vault(&vault, config).unwrap();
        let vec = tensor.entity_vector("user:alice");
        let total: f32 = vec.iter().sum();
        assert!((total - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tensor_time_series() {
        let vault = create_test_vault();
        record_audit(&vault, "user:alice", "secret1");

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let config = AccessTensorConfig {
            bucket_size_ms: 3_600_000,
            num_buckets: 11,
            start_time_ms: Some(now_ms - 3_600_000 * 10),
            operations: None,
        };
        let tensor = AccessTensor::from_vault(&vault, config).unwrap();
        let ts = tensor.time_series("user:alice", "secret1");
        assert_eq!(ts.len(), 11);
        let total: f32 = ts.iter().sum();
        assert!((total - 1.0).abs() < f32::EPSILON);

        // Nonexistent returns empty
        let empty = tensor.time_series("user:nobody", "secret1");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_tensor_entity_profiles() {
        let vault = create_test_vault();
        for _ in 0..3 {
            record_audit(&vault, "user:alice", "s1");
        }
        record_audit(&vault, "user:bob", "s1");

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let config = AccessTensorConfig {
            bucket_size_ms: 3_600_000,
            num_buckets: 11,
            start_time_ms: Some(now_ms - 3_600_000 * 10),
            operations: None,
        };
        let tensor = AccessTensor::from_vault(&vault, config).unwrap();
        let profiles = tensor.entity_profiles();
        assert_eq!(profiles.len(), 2);

        // Alice has more accesses, should be first
        assert_eq!(profiles[0].entity, "user:alice");
        assert_eq!(profiles[0].total_accesses, 3);
        assert_eq!(profiles[1].total_accesses, 1);
    }

    #[test]
    fn test_tensor_secret_profiles() {
        let vault = create_test_vault();
        record_audit(&vault, "user:alice", "popular");
        record_audit(&vault, "user:bob", "popular");
        record_audit(&vault, "user:alice", "private");

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let config = AccessTensorConfig {
            bucket_size_ms: 3_600_000,
            num_buckets: 11,
            start_time_ms: Some(now_ms - 3_600_000 * 10),
            operations: None,
        };
        let tensor = AccessTensor::from_vault(&vault, config).unwrap();
        let profiles = tensor.secret_profiles();
        assert_eq!(profiles.len(), 2);

        // "popular" has 2 unique accessors
        let popular = profiles.iter().find(|p| p.secret == "popular").unwrap();
        assert_eq!(popular.unique_accessors, 2);
    }

    #[test]
    fn test_tensor_dimensions() {
        let vault = create_test_vault();
        record_audit(&vault, "e1", "s1");
        record_audit(&vault, "e2", "s2");
        record_audit(&vault, "e3", "s3");

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let config = AccessTensorConfig {
            bucket_size_ms: 3_600_000,
            num_buckets: 25,
            start_time_ms: Some(now_ms - 3_600_000 * 24),
            operations: None,
        };
        let tensor = AccessTensor::from_vault(&vault, config).unwrap();
        let (entities, secrets, buckets) = tensor.dimensions();
        assert_eq!(entities, 3);
        assert_eq!(secrets, 3);
        assert_eq!(buckets, 25);
        assert_eq!(tensor.raw_data().len(), 3 * 3 * 25);
    }

    /// Helper to build a small tensor with known data for getter tests.
    fn make_tensor_with_two_entities() -> AccessTensor {
        let vault = create_test_vault();
        record_audit(&vault, "user:alice", "s1");
        record_audit(&vault, "user:alice", "s2");
        record_audit(&vault, "user:bob", "s1");

        #[allow(clippy::cast_precision_loss)]
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        let config = AccessTensorConfig {
            bucket_size_ms: 3_600_000,
            num_buckets: 11,
            start_time_ms: Some(now_ms - 3_600_000 * 10),
            operations: None,
        };
        AccessTensor::from_vault(&vault, config).unwrap()
    }

    #[test]
    fn test_tensor_get_missing_entity() {
        let tensor = make_tensor_with_two_entities();
        assert!((tensor.get("nonexistent", "s1", 0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tensor_get_missing_secret() {
        let tensor = make_tensor_with_two_entities();
        assert!((tensor.get("user:alice", "nonexistent", 0) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tensor_get_out_of_bounds_bucket() {
        let tensor = make_tensor_with_two_entities();
        assert!((tensor.get("user:alice", "s1", 9999) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tensor_time_series_missing_secret() {
        let tensor = make_tensor_with_two_entities();
        let ts = tensor.time_series("user:alice", "nonexistent");
        assert!(ts.is_empty());
    }

    #[test]
    fn test_tensor_entity_vector_missing() {
        let tensor = make_tensor_with_two_entities();
        let vec = tensor.entity_vector("nonexistent");
        assert!(vec.is_empty());
    }

    #[test]
    fn test_tensor_secret_vector() {
        let tensor = make_tensor_with_two_entities();
        // s1 was accessed by both alice and bob, so secret_vector should
        // contain data from both entities (n_entities * n_buckets values).
        let vec = tensor.secret_vector("s1");
        let (n_entities, _, n_buckets) = tensor.dimensions();
        assert_eq!(vec.len(), n_entities * n_buckets);
        // Total accesses for s1: alice(1) + bob(1) = 2
        let total: f32 = vec.iter().sum();
        assert!((total - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tensor_secret_vector_missing() {
        let tensor = make_tensor_with_two_entities();
        let vec = tensor.secret_vector("nonexistent");
        assert!(vec.is_empty());
    }

    #[test]
    fn test_tensor_operations_filter() {
        let vault = create_test_vault();
        // Record different operation types
        let audit = AuditLog::new(&vault.store, Some(*vault.audit_key()));
        audit.record("user:alice", "s1", &crate::audit::AuditOperation::Get);
        audit.record("user:alice", "s1", &crate::audit::AuditOperation::Set);
        audit.record("user:bob", "s2", &crate::audit::AuditOperation::Get);
        audit.record("user:bob", "s2", &crate::audit::AuditOperation::Delete);

        #[allow(clippy::cast_precision_loss)]
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64;

        // Only include "Get" operations
        let config = AccessTensorConfig {
            bucket_size_ms: 3_600_000,
            num_buckets: 11,
            start_time_ms: Some(now_ms - 3_600_000 * 10),
            operations: Some(vec!["Get".to_string()]),
        };
        let tensor = AccessTensor::from_vault(&vault, config).unwrap();

        // Only Get ops should appear: alice->s1(Get) and bob->s2(Get)
        let total: f32 = tensor.raw_data().iter().sum();
        assert!((total - 2.0).abs() < f32::EPSILON);
        assert_eq!(tensor.dimensions().0, 2); // alice, bob
        assert_eq!(tensor.dimensions().1, 2); // s1, s2
    }
}
