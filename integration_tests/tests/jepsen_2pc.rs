// SPDX-License-Identifier: MIT OR Apache-2.0
//! Jepsen-style tests for Two-Phase Commit scenarios.
//!
//! Simulates distributed transaction scenarios with fault injection
//! and checks that the observed operation history is linearizable.

use std::time::Duration;

use integration_tests::chaos::ChaosConfig;
use integration_tests::jepsen::{JepsenHarness, NemesisAction, NemesisSchedule};
use integration_tests::linearizability::Value;

#[test]
fn test_2pc_no_faults_all_participants_agree() {
    let nemesis = NemesisSchedule::new();
    let mut harness = JepsenHarness::new(3, nemesis);

    // Coordinator writes prepare to all participants
    let prepare = harness.record_write(1, "tx:1:prepare".to_string(), Value::Int(1));
    harness.complete_op(prepare, Value::None);

    // All participants vote yes
    let vote1 = harness.record_write(2, "tx:1:vote:1".to_string(), Value::Int(1));
    harness.complete_op(vote1, Value::None);
    let vote2 = harness.record_write(3, "tx:1:vote:2".to_string(), Value::Int(1));
    harness.complete_op(vote2, Value::None);

    // Coordinator commits
    let commit = harness.record_write(1, "tx:1:commit".to_string(), Value::Int(1));
    harness.complete_op(commit, Value::None);

    // Read committed state
    let r = harness.record_read(2, "tx:1:commit".to_string());
    harness.complete_op(r, Value::Int(1));

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.operation_count, 5);
}

#[test]
fn test_2pc_coordinator_crash_during_prepare() {
    let nemesis = NemesisSchedule::new()
        .add(Duration::from_millis(50), NemesisAction::RandomCrash)
        .add(Duration::from_millis(200), NemesisAction::HealAll);

    let mut harness = JepsenHarness::new(3, nemesis);

    // Coordinator starts prepare
    let prepare = harness.record_write(1, "tx:2:prepare".to_string(), Value::Int(1));
    harness.complete_op(prepare, Value::None);

    // Crash coordinator
    harness.apply_nemesis(Duration::from_millis(100));

    // Participants can still record their votes even though coordinator crashed
    let vote = harness.record_write(2, "tx:2:vote:1".to_string(), Value::Int(1));
    harness.complete_op(vote, Value::None);

    // Recover
    harness.apply_nemesis(Duration::from_millis(300));

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.nemesis_actions_applied, 2);
}

#[test]
fn test_2pc_participant_crash_vote_abort() {
    let nemesis = NemesisSchedule::new();
    let mut harness = JepsenHarness::new(4, nemesis);

    // Prepare
    let prepare = harness.record_write(1, "tx:3:prepare".to_string(), Value::Int(1));
    harness.complete_op(prepare, Value::None);

    // Crash a participant
    harness.cluster_mut().crash_node(2);

    // Remaining participants vote
    let vote1 = harness.record_write(2, "tx:3:vote:1".to_string(), Value::Int(1));
    harness.complete_op(vote1, Value::None);
    let vote3 = harness.record_write(4, "tx:3:vote:3".to_string(), Value::Int(1));
    harness.complete_op(vote3, Value::None);

    // Missing vote from crashed participant -> abort
    let abort = harness.record_write(1, "tx:3:abort".to_string(), Value::Int(1));
    harness.complete_op(abort, Value::None);

    let r = harness.record_read(2, "tx:3:abort".to_string());
    harness.complete_op(r, Value::Int(1));

    let result = harness.check();
    assert!(result.is_valid());
}

#[test]
fn test_2pc_network_partition_during_commit() {
    let nemesis = NemesisSchedule::new()
        .add(Duration::ZERO, NemesisAction::MajorityPartition)
        .add(Duration::from_millis(150), NemesisAction::HealAll);

    let mut harness = JepsenHarness::new(5, nemesis);

    // Prepare and commit before partition
    let prepare = harness.record_write(1, "tx:4:prepare".to_string(), Value::Int(1));
    harness.complete_op(prepare, Value::None);

    // Apply partition
    harness.apply_nemesis(Duration::ZERO);

    // Majority side commits
    let commit = harness.record_write(3, "tx:4:commit".to_string(), Value::Int(1));
    harness.complete_op(commit, Value::None);

    // Heal
    harness.apply_nemesis(Duration::from_millis(200));

    // After healing, all nodes should see committed state
    let r = harness.record_read(1, "tx:4:commit".to_string());
    harness.complete_op(r, Value::Int(1));

    let result = harness.check();
    assert!(result.is_valid());
}

#[test]
fn test_2pc_concurrent_transactions_different_keys() {
    let nemesis = NemesisSchedule::new();
    let mut harness = JepsenHarness::new(3, nemesis);

    // Transaction A writes key "a"
    let wa = harness.record_write(1, "a".to_string(), Value::Int(10));
    harness.complete_op(wa, Value::None);

    // Transaction B writes key "b" concurrently
    let wb = harness.record_write(2, "b".to_string(), Value::Int(20));
    harness.complete_op(wb, Value::None);

    // Both reads should succeed independently
    let ra = harness.record_read(1, "a".to_string());
    harness.complete_op(ra, Value::Int(10));
    let rb = harness.record_read(2, "b".to_string());
    harness.complete_op(rb, Value::Int(20));

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.operation_count, 4);
}

#[test]
fn test_2pc_clock_drift_does_not_break_linearizability() {
    let nemesis = NemesisSchedule::new().add(
        Duration::ZERO,
        NemesisAction::ClockDrift { drift_ms: 10_000 },
    );

    let mut harness = JepsenHarness::new(3, nemesis);
    harness.apply_nemesis(Duration::ZERO);

    // Even with extreme clock drift, the logical operation order is preserved
    let w = harness.record_write(1, "tx_clock".to_string(), Value::Int(42));
    harness.complete_op(w, Value::None);

    let r = harness.record_read(2, "tx_clock".to_string());
    harness.complete_op(r, Value::Int(42));

    let result = harness.check();
    assert!(result.is_valid());
}

#[test]
fn test_2pc_link_degradation_with_operations() {
    let nemesis = NemesisSchedule::new().add(
        Duration::ZERO,
        NemesisAction::LinkDegradation { drop_rate: 0.8 },
    );

    let mut harness = JepsenHarness::new(3, nemesis);
    harness.apply_nemesis(Duration::ZERO);

    // Write multiple values
    for i in 0..3 {
        let w = harness.record_write(1, format!("key{i}"), Value::Int(i));
        harness.complete_op(w, Value::None);
    }

    // Read them back
    for i in 0..3 {
        let r = harness.record_read(2, format!("key{i}"));
        harness.complete_op(r, Value::Int(i));
    }

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.operation_count, 6);
}

#[test]
fn test_2pc_result_reports_faults() {
    let config = ChaosConfig::moderate();
    let nemesis = NemesisSchedule::new();
    let harness = JepsenHarness::with_chaos_config(3, config, nemesis);

    let result = harness.check();
    assert!(result.is_valid());
    assert_eq!(result.operation_count, 0);
    // No actual messages sent so no faults counted
    assert_eq!(result.total_faults_injected, 0);
}

#[test]
fn test_2pc_multiple_transactions_same_key_sequential() {
    let nemesis = NemesisSchedule::new();
    let mut harness = JepsenHarness::new(3, nemesis);

    // Transaction 1: write x=1
    let w1 = harness.record_write(1, "x".to_string(), Value::Int(1));
    harness.complete_op(w1, Value::None);

    // Transaction 2: write x=2
    let w2 = harness.record_write(1, "x".to_string(), Value::Int(2));
    harness.complete_op(w2, Value::None);

    // Transaction 3: write x=3
    let w3 = harness.record_write(1, "x".to_string(), Value::Int(3));
    harness.complete_op(w3, Value::None);

    // Read should return latest
    let r = harness.record_read(2, "x".to_string());
    harness.complete_op(r, Value::Int(3));

    let result = harness.check();
    assert!(result.is_valid());
}

#[test]
fn test_jepsen_nemesis_cursor_progression() {
    let nemesis = NemesisSchedule::new()
        .add(Duration::from_millis(10), NemesisAction::RandomCrash)
        .add(Duration::from_millis(100), NemesisAction::HealAll)
        .add(Duration::from_millis(200), NemesisAction::MajorityPartition)
        .add(Duration::from_millis(300), NemesisAction::HealAll);

    let mut harness = JepsenHarness::new(5, nemesis);

    // At time 0, nothing should happen (first action at 10ms)
    let applied = harness.apply_nemesis(Duration::ZERO);
    assert_eq!(applied, 0);

    // At 50ms, crash should have happened
    let applied = harness.apply_nemesis(Duration::from_millis(50));
    assert_eq!(applied, 1);

    // At 150ms, heal should happen
    let applied = harness.apply_nemesis(Duration::from_millis(150));
    assert_eq!(applied, 1);

    // At 350ms, both remaining actions should happen
    let applied = harness.apply_nemesis(Duration::from_millis(350));
    assert_eq!(applied, 2);

    // No more actions
    let applied = harness.apply_nemesis(Duration::from_millis(500));
    assert_eq!(applied, 0);
}
