// SPDX-License-Identifier: MIT OR Apache-2.0
//! Jepsen-style tests for Raft consensus scenarios.
//!
//! Uses the JepsenHarness to record operation histories with nemesis
//! (fault) injection and verifies linearizability of the results.

use std::time::Duration;

use integration_tests::chaos::ChaosConfig;
use integration_tests::jepsen::{JepsenHarness, NemesisAction, NemesisSchedule};
use integration_tests::linearizability::{LinearizabilityResult, Value};

#[test]
fn test_jepsen_no_faults_linearizable() {
    let nemesis = NemesisSchedule::new();
    let mut harness = JepsenHarness::new(3, nemesis);

    // Write then read
    let w = harness.record_write(1, "x".to_string(), Value::Int(42));
    harness.complete_op(w, Value::None);

    let r = harness.record_read(2, "x".to_string());
    harness.complete_op(r, Value::Int(42));

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.operation_count, 2);
    assert_eq!(result.nemesis_actions_applied, 0);
}

#[test]
fn test_jepsen_partition_heal_schedule() {
    let nemesis = NemesisSchedule::partition_heal(Duration::from_millis(100));
    let mut harness = JepsenHarness::new(5, nemesis);

    // Apply nemesis at time 0 (partition)
    let applied = harness.apply_nemesis(Duration::ZERO);
    assert_eq!(applied, 1);

    // Verify partition was created
    let t0 = harness.cluster().transport(0).unwrap();
    assert!(t0.is_partitioned(&"node-2".to_string()));

    // Record operations in majority partition
    let w = harness.record_write(3, "key".to_string(), Value::Int(100));
    harness.complete_op(w, Value::None);

    let r = harness.record_read(4, "key".to_string());
    harness.complete_op(r, Value::Int(100));

    // Apply heal at time > 100ms
    let applied = harness.apply_nemesis(Duration::from_millis(200));
    assert_eq!(applied, 1);

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.nemesis_actions_applied, 2);
}

#[test]
fn test_jepsen_crash_recover_schedule() {
    let nemesis = NemesisSchedule::crash_recover(Duration::from_millis(50));
    let mut harness = JepsenHarness::new(3, nemesis);

    // Apply crash
    let applied = harness.apply_nemesis(Duration::ZERO);
    assert_eq!(applied, 1);

    // Node 0 should be crashed
    assert!(harness.cluster().is_node_crashed(0));

    // Operations on surviving nodes
    let w = harness.record_write(2, "data".to_string(), Value::Str("hello".to_string()));
    harness.complete_op(w, Value::None);

    // Recover
    let applied = harness.apply_nemesis(Duration::from_millis(100));
    assert_eq!(applied, 1);

    assert!(!harness.cluster().is_node_crashed(0));

    let result = harness.check();
    assert!(result.is_valid());
}

#[test]
fn test_jepsen_clock_drift_nemesis() {
    let nemesis = NemesisSchedule::new()
        .add(Duration::ZERO, NemesisAction::ClockDrift { drift_ms: 5000 })
        .add(Duration::from_millis(200), NemesisAction::HealAll);

    let mut harness = JepsenHarness::new(3, nemesis);

    harness.apply_nemesis(Duration::ZERO);

    // Clock drift should be set on node 0
    let clock = harness.cluster().clock(0).unwrap();
    assert_eq!(clock.drift_offset(), 5000);

    // Writes and reads are still logically linearizable
    let w = harness.record_write(1, "t".to_string(), Value::Int(1));
    harness.complete_op(w, Value::None);
    let r = harness.record_read(2, "t".to_string());
    harness.complete_op(r, Value::Int(1));

    harness.apply_nemesis(Duration::from_millis(300));

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.nemesis_actions_applied, 2);
}

#[test]
fn test_jepsen_link_degradation_nemesis() {
    let nemesis = NemesisSchedule::new().add(
        Duration::ZERO,
        NemesisAction::LinkDegradation { drop_rate: 0.5 },
    );

    let mut harness = JepsenHarness::new(3, nemesis);
    harness.apply_nemesis(Duration::ZERO);

    let w = harness.record_write(1, "k".to_string(), Value::Int(99));
    harness.complete_op(w, Value::None);

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.nemesis_actions_applied, 1);
}

#[test]
fn test_jepsen_multiple_writes_single_key() {
    let nemesis = NemesisSchedule::partition_heal(Duration::from_millis(50));
    let mut harness = JepsenHarness::new(3, nemesis);

    // Multiple sequential writes
    for i in 0..5 {
        let w = harness.record_write(1, "counter".to_string(), Value::Int(i));
        harness.complete_op(w, Value::None);
    }

    // Read final value
    let r = harness.record_read(2, "counter".to_string());
    harness.complete_op(r, Value::Int(4));

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.operation_count, 6);
}

#[test]
fn test_jepsen_with_chaos_config() {
    let config = ChaosConfig::mild();
    let nemesis = NemesisSchedule::new();
    let mut harness = JepsenHarness::with_chaos_config(3, config, nemesis);

    let w = harness.record_write(1, "x".to_string(), Value::Int(1));
    harness.complete_op(w, Value::None);

    let result = harness.check();
    assert!(result.is_valid());
}

#[test]
fn test_jepsen_harness_cluster_access() {
    let nemesis = NemesisSchedule::new();
    let mut harness = JepsenHarness::new(5, nemesis);

    assert_eq!(harness.cluster().node_count(), 5);

    // Crash via cluster_mut
    harness.cluster_mut().crash_node(0);
    assert!(harness.cluster().is_node_crashed(0));
    assert_eq!(harness.cluster().active_node_count(), 4);
}

#[test]
fn test_jepsen_stale_read_detected() {
    let nemesis = NemesisSchedule::new();
    let mut harness = JepsenHarness::new(3, nemesis);

    // Write x=1, write x=2, read x=1 (stale)
    let w1 = harness.record_write(1, "x".to_string(), Value::Int(1));
    harness.complete_op(w1, Value::None);

    let w2 = harness.record_write(1, "x".to_string(), Value::Int(2));
    harness.complete_op(w2, Value::None);

    let r = harness.record_read(2, "x".to_string());
    harness.complete_op(r, Value::Int(1));

    let result = harness.check();
    assert!(!result.is_valid());
    assert!(matches!(
        result.linearizability,
        LinearizabilityResult::Violation(_)
    ));
}

#[test]
fn test_jepsen_combined_faults() {
    let nemesis = NemesisSchedule::new()
        .add(Duration::ZERO, NemesisAction::RandomCrash)
        .add(
            Duration::from_millis(50),
            NemesisAction::ClockDrift { drift_ms: 1000 },
        )
        .add(
            Duration::from_millis(100),
            NemesisAction::LinkDegradation { drop_rate: 0.3 },
        )
        .add(Duration::from_millis(200), NemesisAction::HealAll);

    let mut harness = JepsenHarness::new(5, nemesis);

    // Apply all actions progressively
    harness.apply_nemesis(Duration::ZERO);
    harness.apply_nemesis(Duration::from_millis(75));
    harness.apply_nemesis(Duration::from_millis(150));

    // Record operations between fault injections
    let w = harness.record_write(3, "k".to_string(), Value::Int(7));
    harness.complete_op(w, Value::None);

    harness.apply_nemesis(Duration::from_millis(250));

    let r = harness.record_read(4, "k".to_string());
    harness.complete_op(r, Value::Int(7));

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.nemesis_actions_applied, 4);
}
