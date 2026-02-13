// SPDX-License-Identifier: MIT OR Apache-2.0
//! Anomaly detection for vault agent behavior.
//!
//! Monitors per-agent access patterns and emits events when unusual
//! activity is detected: first-time secret access, frequency spikes,
//! bulk operations, and dormant agent resumption.

use std::collections::{HashMap, HashSet};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, TensorStore, TensorValue};

use crate::audit::AuditOperation;

/// Storage key prefix for persisted agent profiles.
const PROFILE_PREFIX: &str = "_vap:";

/// Configurable thresholds for anomaly detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyThresholds {
    /// Operations per window that trigger a frequency spike (default 50).
    pub frequency_spike_limit: u64,
    /// Sliding window size in milliseconds (default 60_000 = 1 minute).
    pub frequency_window_ms: i64,
    /// Operations in a burst that trigger a bulk-operation event (default 10).
    pub bulk_operation_threshold: u64,
    /// Milliseconds of inactivity before resumption is flagged (default 86_400_000 = 24h).
    pub inactive_threshold_ms: i64,
}

impl Default for AnomalyThresholds {
    fn default() -> Self {
        Self {
            frequency_spike_limit: 50,
            frequency_window_ms: 60_000,
            bulk_operation_threshold: 10,
            inactive_threshold_ms: 86_400_000,
        }
    }
}

/// Per-agent behavioral state tracked by the monitor.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentProfile {
    /// Obfuscated keys this agent has accessed.
    pub known_secrets: HashSet<String>,
    /// Per-secret access count (obfuscated keys).
    pub access_counts: HashMap<String, u64>,
    /// Timestamp (ms) of the most recent operation.
    pub last_activity_ms: i64,
    /// Total lifetime operations.
    pub total_ops: u64,
    /// Recent timestamps for sliding-window frequency analysis.
    pub recent_timestamps: Vec<i64>,
}

/// An anomalous event detected by the monitor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnomalyEvent {
    /// Agent is accessing a secret for the first time.
    FirstSecretAccess {
        /// Entity performing the access.
        entity: String,
        /// Obfuscated key of the secret being accessed.
        secret_key: String,
    },
    /// Agent operation rate exceeds the configured limit.
    FrequencySpike {
        /// Entity whose rate spiked.
        entity: String,
        /// Number of operations within the sliding window.
        ops_in_window: u64,
        /// Configured spike limit that was exceeded.
        threshold: u64,
    },
    /// Agent performed a burst of operations.
    BulkOperation {
        /// Entity performing the burst.
        entity: String,
        /// Number of accesses to the same secret.
        operation_count: u64,
        /// Configured bulk threshold that was reached.
        threshold: u64,
    },
    /// A previously dormant agent has resumed activity.
    InactiveAgentResumed {
        /// Entity that resumed activity.
        entity: String,
        /// Duration of inactivity in milliseconds before resumption.
        inactive_duration_ms: i64,
    },
}

/// Thread-safe anomaly monitor that tracks per-agent behavior.
pub struct AnomalyMonitor {
    profiles: DashMap<String, AgentProfile>,
    thresholds: AnomalyThresholds,
}

impl AnomalyMonitor {
    /// Create a new anomaly monitor with the given thresholds.
    pub fn new(thresholds: AnomalyThresholds) -> Self {
        Self {
            profiles: DashMap::new(),
            thresholds,
        }
    }

    /// Load persisted agent profiles from the store.
    pub fn load(store: &TensorStore, thresholds: AnomalyThresholds) -> Self {
        let monitor = Self::new(thresholds);
        for key in store.scan(PROFILE_PREFIX) {
            if let Ok(data) = store.get(&key) {
                if let Some(TensorValue::Scalar(ScalarValue::String(json))) = data.get("_profile") {
                    if let Ok(profile) = serde_json::from_str::<AgentProfile>(json) {
                        let entity = key.strip_prefix(PROFILE_PREFIX).unwrap_or(&key);
                        monitor.profiles.insert(entity.to_string(), profile);
                    }
                }
            }
        }
        monitor
    }

    /// Persist all agent profiles to the store.
    pub fn persist(&self, store: &TensorStore) {
        // Remove old entries first
        for key in store.scan(PROFILE_PREFIX) {
            store.delete(&key).ok();
        }
        for entry in &self.profiles {
            let storage_key = format!("{PROFILE_PREFIX}{}", entry.key());
            if let Ok(json) = serde_json::to_string(entry.value()) {
                let mut data = tensor_store::TensorData::new();
                data.set("_profile", TensorValue::Scalar(ScalarValue::String(json)));
                store.put(&storage_key, data).ok();
            }
        }
    }

    /// Core detection: check an operation and return any anomaly events.
    pub fn check(
        &self,
        entity: &str,
        obfuscated_key: &str,
        operation: &AuditOperation,
        now_ms: i64,
    ) -> Vec<AnomalyEvent> {
        let mut events = Vec::new();
        let mut profile = self.profiles.entry(entity.to_string()).or_default().clone();

        // 1. Inactive agent resumed
        if profile.last_activity_ms > 0 {
            let gap = now_ms - profile.last_activity_ms;
            if gap > self.thresholds.inactive_threshold_ms {
                events.push(AnomalyEvent::InactiveAgentResumed {
                    entity: entity.to_string(),
                    inactive_duration_ms: gap,
                });
            }
        }

        // 2. First secret access (skip for List operations)
        if !matches!(operation, AuditOperation::List)
            && !profile.known_secrets.contains(obfuscated_key)
        {
            events.push(AnomalyEvent::FirstSecretAccess {
                entity: entity.to_string(),
                secret_key: obfuscated_key.to_string(),
            });
            profile.known_secrets.insert(obfuscated_key.to_string());
        }

        // 3. Sliding window frequency check
        let window_start = now_ms - self.thresholds.frequency_window_ms;
        profile.recent_timestamps.retain(|&ts| ts >= window_start);
        profile.recent_timestamps.push(now_ms);

        let ops_in_window = profile.recent_timestamps.len() as u64;
        if ops_in_window > self.thresholds.frequency_spike_limit {
            events.push(AnomalyEvent::FrequencySpike {
                entity: entity.to_string(),
                ops_in_window,
                threshold: self.thresholds.frequency_spike_limit,
            });
        }

        // 4. Bulk operation check (per-secret burst)
        let count = profile
            .access_counts
            .entry(obfuscated_key.to_string())
            .or_insert(0);
        *count += 1;
        if *count == self.thresholds.bulk_operation_threshold {
            events.push(AnomalyEvent::BulkOperation {
                entity: entity.to_string(),
                operation_count: *count,
                threshold: self.thresholds.bulk_operation_threshold,
            });
        }

        // Update counters
        profile.last_activity_ms = now_ms;
        profile.total_ops += 1;
        self.profiles.insert(entity.to_string(), profile);

        events
    }

    /// Get the profile for a specific entity.
    pub fn get_profile(&self, entity: &str) -> Option<AgentProfile> {
        self.profiles.get(entity).map(|p| p.clone())
    }

    /// List all monitored entity names.
    pub fn monitored_entities(&self) -> Vec<String> {
        self.profiles.iter().map(|e| e.key().clone()).collect()
    }

    /// Reset (remove) a specific entity's profile.
    pub fn reset_profile(&self, entity: &str) {
        self.profiles.remove(entity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_thresholds() -> AnomalyThresholds {
        AnomalyThresholds {
            frequency_spike_limit: 5,
            frequency_window_ms: 1000,
            bulk_operation_threshold: 3,
            inactive_threshold_ms: 5000,
        }
    }

    #[test]
    fn test_first_secret_access_detected() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        let events = monitor.check("agent:a", "secret_x", &AuditOperation::Get, 1000);
        assert!(events.iter().any(|e| matches!(
            e,
            AnomalyEvent::FirstSecretAccess { entity, secret_key }
            if entity == "agent:a" && secret_key == "secret_x"
        )));
    }

    #[test]
    fn test_repeat_access_no_event() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        monitor.check("agent:a", "secret_x", &AuditOperation::Get, 1000);
        let events = monitor.check("agent:a", "secret_x", &AuditOperation::Get, 1001);
        assert!(!events
            .iter()
            .any(|e| matches!(e, AnomalyEvent::FirstSecretAccess { .. })));
    }

    #[test]
    fn test_frequency_spike_detected() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        // Threshold is 5 ops in 1000ms window
        for i in 0..5 {
            monitor.check("agent:a", &format!("s{i}"), &AuditOperation::Get, 100 + i);
        }
        let events = monitor.check("agent:a", "s5", &AuditOperation::Get, 105);
        assert!(events
            .iter()
            .any(|e| matches!(e, AnomalyEvent::FrequencySpike { .. })));
    }

    #[test]
    fn test_below_threshold_no_spike() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        // Only 3 ops -- well below the limit of 5
        for i in 0..3 {
            let events = monitor.check("agent:a", &format!("s{i}"), &AuditOperation::Get, i);
            assert!(!events
                .iter()
                .any(|e| matches!(e, AnomalyEvent::FrequencySpike { .. })));
        }
    }

    #[test]
    fn test_inactive_agent_resumed() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        monitor.check("agent:a", "s1", &AuditOperation::Get, 1000);
        // Gap of 10_000ms > threshold of 5000ms
        let events = monitor.check("agent:a", "s1", &AuditOperation::Get, 11_000);
        assert!(events.iter().any(|e| matches!(
            e,
            AnomalyEvent::InactiveAgentResumed { inactive_duration_ms, .. }
            if *inactive_duration_ms == 10_000
        )));
    }

    #[test]
    fn test_first_op_no_inactive_event() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        let events = monitor.check("agent:a", "s1", &AuditOperation::Get, 1000);
        assert!(!events
            .iter()
            .any(|e| matches!(e, AnomalyEvent::InactiveAgentResumed { .. })));
    }

    #[test]
    fn test_list_op_no_first_secret_event() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        let events = monitor.check("agent:a", "pattern", &AuditOperation::List, 1000);
        assert!(!events
            .iter()
            .any(|e| matches!(e, AnomalyEvent::FirstSecretAccess { .. })));
    }

    #[test]
    fn test_concurrent_agents() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        let events_a = monitor.check("agent:a", "s1", &AuditOperation::Get, 1000);
        let events_b = monitor.check("agent:b", "s1", &AuditOperation::Get, 1000);
        // Both should see FirstSecretAccess for their own profile
        assert!(events_a.iter().any(
            |e| matches!(e, AnomalyEvent::FirstSecretAccess { entity, .. } if entity == "agent:a")
        ));
        assert!(events_b.iter().any(
            |e| matches!(e, AnomalyEvent::FirstSecretAccess { entity, .. } if entity == "agent:b")
        ));
    }

    #[test]
    fn test_persist_and_load() {
        let store = TensorStore::new();
        let monitor = AnomalyMonitor::new(default_thresholds());
        monitor.check("agent:a", "s1", &AuditOperation::Get, 1000);
        monitor.check("agent:b", "s2", &AuditOperation::Set, 2000);
        monitor.persist(&store);

        let loaded = AnomalyMonitor::load(&store, default_thresholds());
        let profile_a = loaded.get_profile("agent:a").unwrap();
        assert_eq!(profile_a.total_ops, 1);
        assert!(profile_a.known_secrets.contains("s1"));

        let profile_b = loaded.get_profile("agent:b").unwrap();
        assert_eq!(profile_b.total_ops, 1);
        assert!(profile_b.known_secrets.contains("s2"));
    }

    #[test]
    fn test_reset_profile() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        monitor.check("agent:a", "s1", &AuditOperation::Get, 1000);
        assert!(monitor.get_profile("agent:a").is_some());

        monitor.reset_profile("agent:a");
        assert!(monitor.get_profile("agent:a").is_none());
    }

    #[test]
    fn test_unknown_agent_none() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        assert!(monitor.get_profile("nonexistent").is_none());
    }

    #[test]
    fn test_monitored_entities() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        monitor.check("agent:a", "s1", &AuditOperation::Get, 1000);
        monitor.check("agent:b", "s1", &AuditOperation::Get, 1000);

        let mut entities = monitor.monitored_entities();
        entities.sort();
        assert_eq!(entities, vec!["agent:a", "agent:b"]);
    }

    #[test]
    fn test_sliding_window_pruning() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        // Window is 1000ms. Insert ops at t=0..4
        for i in 0..4 {
            monitor.check("agent:a", &format!("s{i}"), &AuditOperation::Get, i);
        }
        // Now at t=2000, old entries should be pruned
        monitor.check("agent:a", "s_late", &AuditOperation::Get, 2000);
        let profile = monitor.get_profile("agent:a").unwrap();
        // Only the t=2000 entry should remain (window_start = 2000 - 1000 = 1000)
        assert_eq!(profile.recent_timestamps.len(), 1);
        assert_eq!(profile.recent_timestamps[0], 2000);
    }

    #[test]
    fn test_bulk_operation_check() {
        let monitor = AnomalyMonitor::new(default_thresholds());
        // Threshold is 3, accessing the same secret repeatedly
        let e1 = monitor.check("agent:a", "s1", &AuditOperation::Get, 1000);
        assert!(!e1
            .iter()
            .any(|e| matches!(e, AnomalyEvent::BulkOperation { .. })));

        let e2 = monitor.check("agent:a", "s1", &AuditOperation::Get, 1001);
        assert!(!e2
            .iter()
            .any(|e| matches!(e, AnomalyEvent::BulkOperation { .. })));

        let e3 = monitor.check("agent:a", "s1", &AuditOperation::Get, 1002);
        assert!(e3.iter().any(|e| matches!(
            e,
            AnomalyEvent::BulkOperation { operation_count, threshold, .. }
            if *operation_count == 3 && *threshold == 3
        )));
    }
}
