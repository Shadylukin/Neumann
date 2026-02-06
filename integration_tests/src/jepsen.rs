// SPDX-License-Identifier: MIT OR Apache-2.0
//! Jepsen-style distributed systems test harness.
//!
//! Combines chaos injection with linearizability checking to verify that
//! a distributed system maintains correctness under fault conditions.
//! The harness records a history of operations, applies nemesis actions
//! (faults) according to a schedule, and then checks whether the observed
//! history is linearizable with respect to a sequential register model.

use std::time::Duration;

use crate::chaos::{ChaosCluster, ChaosConfig};
use crate::linearizability::{
    HistoryRecorder, LinearizabilityChecker, LinearizabilityResult, OpType, RegisterModel, Value,
};

// ---------------------------------------------------------------------------
// Nemesis types
// ---------------------------------------------------------------------------

/// Configuration for nemesis (fault injection) during Jepsen tests.
#[derive(Debug, Clone)]
pub enum NemesisAction {
    /// Crash a random node.
    RandomCrash,
    /// Partition network into majority/minority.
    MajorityPartition,
    /// Inject clock drift on a random node.
    ClockDrift { drift_ms: i64 },
    /// Set asymmetric link quality.
    LinkDegradation { drop_rate: f32 },
    /// Heal all faults.
    HealAll,
}

/// Schedule of nemesis actions for a test.
#[derive(Debug, Clone)]
pub struct NemesisSchedule {
    /// Actions to execute in order.
    pub actions: Vec<(Duration, NemesisAction)>,
    /// Whether to heal all faults at the end.
    pub heal_at_end: bool,
}

impl NemesisSchedule {
    #[must_use]
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
            heal_at_end: true,
        }
    }

    #[must_use]
    pub fn add(mut self, delay: Duration, action: NemesisAction) -> Self {
        self.actions.push((delay, action));
        self
    }

    #[must_use]
    pub fn with_heal_at_end(mut self, heal: bool) -> Self {
        self.heal_at_end = heal;
        self
    }

    /// Create a simple partition-then-heal schedule.
    #[must_use]
    pub fn partition_heal(partition_duration: Duration) -> Self {
        Self::new()
            .add(Duration::ZERO, NemesisAction::MajorityPartition)
            .add(partition_duration, NemesisAction::HealAll)
    }

    /// Create a crash-recover schedule.
    #[must_use]
    pub fn crash_recover(crash_duration: Duration) -> Self {
        Self::new()
            .add(Duration::ZERO, NemesisAction::RandomCrash)
            .add(crash_duration, NemesisAction::HealAll)
    }
}

impl Default for NemesisSchedule {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Jepsen result
// ---------------------------------------------------------------------------

/// Result of a Jepsen-style test.
#[derive(Debug)]
pub struct JepsenResult {
    pub duration: Duration,
    pub operation_count: usize,
    pub linearizability: LinearizabilityResult,
    pub nemesis_actions_applied: usize,
    pub total_faults_injected: u64,
}

impl JepsenResult {
    /// Returns true if the recorded history is linearizable.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.linearizability == LinearizabilityResult::Ok
    }
}

// ---------------------------------------------------------------------------
// Jepsen harness
// ---------------------------------------------------------------------------

/// Jepsen-style test harness combining chaos cluster with linearizability checking.
pub struct JepsenHarness {
    cluster: ChaosCluster,
    history: HistoryRecorder,
    nemesis: NemesisSchedule,
    checker: LinearizabilityChecker<RegisterModel>,
    /// Index of the next nemesis action to apply.
    nemesis_cursor: usize,
    /// Number of nemesis actions already applied.
    nemesis_applied: usize,
}

impl JepsenHarness {
    #[must_use]
    pub fn new(node_count: usize, nemesis: NemesisSchedule) -> Self {
        Self::with_chaos_config(node_count, ChaosConfig::default(), nemesis)
    }

    #[must_use]
    pub fn with_chaos_config(
        node_count: usize,
        config: ChaosConfig,
        nemesis: NemesisSchedule,
    ) -> Self {
        Self {
            cluster: ChaosCluster::new(node_count, config),
            history: HistoryRecorder::new(),
            nemesis,
            checker: LinearizabilityChecker::new(RegisterModel),
            nemesis_cursor: 0,
            nemesis_applied: 0,
        }
    }

    /// Record a write operation and return its operation id.
    pub fn record_write(&mut self, client_id: u64, key: String, value: Value) -> u64 {
        self.history.invoke(client_id, OpType::Write, key, value)
    }

    /// Record a read operation and return its operation id.
    pub fn record_read(&mut self, client_id: u64, key: String) -> u64 {
        self.history
            .invoke(client_id, OpType::Read, key, Value::None)
    }

    /// Complete an operation with its result.
    pub fn complete_op(&mut self, op_id: u64, result: Value) {
        self.history.complete(op_id, result);
    }

    /// Apply pending nemesis actions whose delay has elapsed.
    ///
    /// Returns the number of actions applied during this call.
    pub fn apply_nemesis(&mut self, elapsed: Duration) -> usize {
        let mut applied = 0;

        while self.nemesis_cursor < self.nemesis.actions.len() {
            let (delay, _) = &self.nemesis.actions[self.nemesis_cursor];
            if elapsed < *delay {
                break;
            }

            let action = self.nemesis.actions[self.nemesis_cursor].1.clone();
            self.apply_action(&action);
            self.nemesis_cursor += 1;
            self.nemesis_applied += 1;
            applied += 1;
        }

        applied
    }

    fn apply_action(&mut self, action: &NemesisAction) {
        match action {
            NemesisAction::RandomCrash => {
                self.cluster.crash_node(0);
            },
            NemesisAction::MajorityPartition => {
                let _ = self.cluster.create_majority_partition();
            },
            NemesisAction::ClockDrift { drift_ms } => {
                self.cluster.inject_clock_drift(0, *drift_ms);
            },
            NemesisAction::LinkDegradation { drop_rate } => {
                self.cluster.set_link_quality(0, 1, *drop_rate);
            },
            NemesisAction::HealAll => {
                let count = self.cluster.node_count();
                for i in 0..count {
                    self.cluster.recover_node(i);
                }
                self.cluster.heal_majority_partition();
            },
        }
    }

    /// Get a reference to the cluster for direct manipulation.
    #[must_use]
    pub fn cluster(&self) -> &ChaosCluster {
        &self.cluster
    }

    /// Get a mutable reference to the cluster.
    pub fn cluster_mut(&mut self) -> &mut ChaosCluster {
        &mut self.cluster
    }

    /// Check linearizability of recorded history and return results.
    #[must_use]
    pub fn check(self) -> JepsenResult {
        let duration = self
            .history
            .operations()
            .last()
            .map_or(Duration::ZERO, |op| {
                op.complete_time
                    .map_or(Duration::ZERO, |ct| ct.duration_since(op.invoke_time))
            });
        let operation_count = self.history.len();
        let linearizability = self.checker.check(self.history.operations());

        let total_faults = self.cluster.total_dropped_messages()
            + self.cluster.total_reordered_messages()
            + self.cluster.total_corrupted_messages();

        JepsenResult {
            duration,
            operation_count,
            linearizability,
            nemesis_actions_applied: self.nemesis_applied,
            total_faults_injected: total_faults,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nemesis_schedule_new() {
        let schedule = NemesisSchedule::new();
        assert!(schedule.actions.is_empty());
        assert!(schedule.heal_at_end);
    }

    #[test]
    fn test_nemesis_schedule_add() {
        let schedule = NemesisSchedule::new()
            .add(Duration::from_secs(1), NemesisAction::RandomCrash)
            .add(Duration::from_secs(5), NemesisAction::HealAll);

        assert_eq!(schedule.actions.len(), 2);
        assert_eq!(schedule.actions[0].0, Duration::from_secs(1));
        assert_eq!(schedule.actions[1].0, Duration::from_secs(5));
    }

    #[test]
    fn test_nemesis_schedule_partition_heal() {
        let schedule = NemesisSchedule::partition_heal(Duration::from_secs(10));

        assert_eq!(schedule.actions.len(), 2);
        assert_eq!(schedule.actions[0].0, Duration::ZERO);
        assert!(matches!(
            schedule.actions[0].1,
            NemesisAction::MajorityPartition
        ));
        assert_eq!(schedule.actions[1].0, Duration::from_secs(10));
        assert!(matches!(schedule.actions[1].1, NemesisAction::HealAll));
        assert!(schedule.heal_at_end);
    }

    #[test]
    fn test_nemesis_schedule_crash_recover() {
        let schedule = NemesisSchedule::crash_recover(Duration::from_secs(5));

        assert_eq!(schedule.actions.len(), 2);
        assert!(matches!(schedule.actions[0].1, NemesisAction::RandomCrash));
        assert!(matches!(schedule.actions[1].1, NemesisAction::HealAll));
        assert_eq!(schedule.actions[1].0, Duration::from_secs(5));
    }

    #[test]
    fn test_jepsen_harness_creation() {
        let nemesis = NemesisSchedule::new();
        let harness = JepsenHarness::new(3, nemesis);

        assert_eq!(harness.cluster().node_count(), 3);
        assert!(harness.history.is_empty());
    }

    #[test]
    fn test_jepsen_harness_record_ops() {
        let nemesis = NemesisSchedule::new();
        let mut harness = JepsenHarness::new(3, nemesis);

        let w = harness.record_write(1, "x".to_string(), Value::Int(42));
        assert_eq!(w, 0);

        let r = harness.record_read(2, "x".to_string());
        assert_eq!(r, 1);

        harness.complete_op(w, Value::Int(42));
        harness.complete_op(r, Value::Int(42));

        assert_eq!(harness.history.len(), 2);
    }

    #[test]
    fn test_jepsen_harness_check_empty() {
        let nemesis = NemesisSchedule::new();
        let harness = JepsenHarness::new(3, nemesis);

        let result = harness.check();
        assert!(result.is_valid());
        assert_eq!(result.operation_count, 0);
        assert_eq!(result.nemesis_actions_applied, 0);
        assert_eq!(result.total_faults_injected, 0);
    }

    #[test]
    fn test_jepsen_result_is_valid() {
        let valid = JepsenResult {
            duration: Duration::from_secs(1),
            operation_count: 10,
            linearizability: LinearizabilityResult::Ok,
            nemesis_actions_applied: 2,
            total_faults_injected: 5,
        };
        assert!(valid.is_valid());

        let invalid = JepsenResult {
            duration: Duration::from_secs(1),
            operation_count: 10,
            linearizability: LinearizabilityResult::Violation("stale read on key 'x'".to_string()),
            nemesis_actions_applied: 2,
            total_faults_injected: 5,
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_nemesis_action_debug() {
        let crash = NemesisAction::RandomCrash;
        let debug = format!("{crash:?}");
        assert_eq!(debug, "RandomCrash");

        let partition = NemesisAction::MajorityPartition;
        assert_eq!(format!("{partition:?}"), "MajorityPartition");

        let drift = NemesisAction::ClockDrift { drift_ms: 500 };
        let drift_debug = format!("{drift:?}");
        assert!(drift_debug.contains("500"));

        let degradation = NemesisAction::LinkDegradation { drop_rate: 0.25 };
        let degradation_debug = format!("{degradation:?}");
        assert!(degradation_debug.contains("0.25"));

        let heal = NemesisAction::HealAll;
        assert_eq!(format!("{heal:?}"), "HealAll");
    }
}
