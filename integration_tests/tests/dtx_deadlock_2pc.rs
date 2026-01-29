// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for deadlock detection in 2PC flow.
//!
//! Tests deadlock detection during prepare phase, cross-shard deadlocks,
//! victim selection policies, and cycle detection with many transactions.

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager},
    distributed_tx::{DistributedTxCoordinator, PrepareRequest, PrepareVote},
    DeadlockDetector, DeadlockDetectorConfig, VictimSelectionPolicy, WaitForGraph,
};
use tensor_store::SparseVector;

fn create_test_coordinator() -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::with_consensus(consensus)
}

#[test]
fn test_2pc_deadlock_detection_during_prepare() {
    let coordinator = create_test_coordinator();

    // TX 100 acquires key_a
    coordinator
        .lock_manager()
        .try_lock(100, &["key_a".to_string()])
        .expect("lock should succeed");

    // TX 200 acquires key_b
    coordinator
        .lock_manager()
        .try_lock(200, &["key_b".to_string()])
        .expect("lock should succeed");

    // TX 100 tries to prepare with key_b (held by TX 200)
    let request1 = PrepareRequest {
        tx_id: 100,
        coordinator: "node1".to_string(),
        operations: vec![tensor_chain::Transaction::Put {
            key: "key_b".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
        timeout_ms: 5000,
    };
    let vote1 = coordinator.handle_prepare(request1);
    assert!(
        matches!(vote1, PrepareVote::Conflict { .. }),
        "TX 100 should conflict"
    );

    // TX 200 tries to prepare with key_a (held by TX 100)
    let request2 = PrepareRequest {
        tx_id: 200,
        coordinator: "node1".to_string(),
        operations: vec![tensor_chain::Transaction::Put {
            key: "key_a".to_string(),
            data: vec![2],
        }],
        delta_embedding: SparseVector::from_dense(&[0.0, 1.0]),
        timeout_ms: 5000,
    };
    let vote2 = coordinator.handle_prepare(request2);
    assert!(
        matches!(vote2, PrepareVote::Conflict { .. }),
        "TX 200 should conflict"
    );

    // Now the wait graph should have a cycle: 100 -> 200 -> 100
    let cycles = coordinator.wait_graph().detect_cycles();
    assert!(!cycles.is_empty(), "should detect deadlock cycle");
}

#[test]
fn test_2pc_deadlock_cross_shard() {
    // Simulate cross-shard deadlock with two coordinators
    let coordinator1 = create_test_coordinator();
    let coordinator2 = create_test_coordinator();
    let shared_wait_graph = Arc::new(WaitForGraph::new());

    // Shard 1: TX 100 holds key_a
    coordinator1
        .lock_manager()
        .try_lock(100, &["key_a".to_string()])
        .expect("lock should succeed");

    // Shard 2: TX 200 holds key_b
    coordinator2
        .lock_manager()
        .try_lock(200, &["key_b".to_string()])
        .expect("lock should succeed");

    // Cross-shard conflict: TX 100 wants key_b on shard 2
    let result = coordinator2.lock_manager().try_lock_with_wait_tracking(
        100,
        &["key_b".to_string()],
        &shared_wait_graph,
        None,
    );
    assert!(result.is_err(), "TX 100 should fail to acquire key_b");

    // Cross-shard conflict: TX 200 wants key_a on shard 1
    let result = coordinator1.lock_manager().try_lock_with_wait_tracking(
        200,
        &["key_a".to_string()],
        &shared_wait_graph,
        None,
    );
    assert!(result.is_err(), "TX 200 should fail to acquire key_a");

    // Shared wait graph should detect the cross-shard cycle
    let cycles = shared_wait_graph.detect_cycles();
    assert!(!cycles.is_empty(), "should detect cross-shard deadlock");
}

#[test]
fn test_2pc_deadlock_victim_selection_policy() {
    let config = DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::Youngest);
    let detector = DeadlockDetector::new(config);
    let graph = detector.graph();

    // TX 100 starts waiting first (older)
    graph.add_wait(100, 200, None);
    thread::sleep(Duration::from_millis(20));
    // TX 200 starts waiting later (younger)
    graph.add_wait(200, 100, None);

    let deadlocks = detector.detect();
    assert!(!deadlocks.is_empty(), "should detect deadlock");

    // Youngest policy: TX 200 should be the victim
    assert_eq!(
        deadlocks[0].victim_tx_id, 200,
        "youngest transaction should be victim"
    );
}

#[test]
fn test_2pc_cycle_detection_with_many_transactions() {
    let detector = DeadlockDetector::with_defaults();
    let graph = detector.graph();

    // Create a chain of 50 transactions that forms a cycle
    for i in 1..=50u64 {
        let next = if i == 50 { 1 } else { i + 1 };
        graph.add_wait(i, next, None);
    }

    let start = std::time::Instant::now();
    let deadlocks = detector.detect();
    let elapsed = start.elapsed();

    // Should complete quickly even with many transactions
    assert!(
        elapsed < Duration::from_millis(100),
        "detection took too long: {:?}",
        elapsed
    );

    // Should detect the cycle
    assert!(!deadlocks.is_empty(), "should detect cycle");
}

#[test]
fn test_2pc_victim_selection_lowest_priority() {
    let config =
        DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::LowestPriority);
    let detector = DeadlockDetector::new(config);
    let graph = detector.graph();

    // TX 100 has high priority (low value)
    graph.add_wait(100, 200, Some(1));
    // TX 200 has low priority (high value) - should be victim
    graph.add_wait(200, 100, Some(100));

    let deadlocks = detector.detect();
    assert!(!deadlocks.is_empty(), "should detect deadlock");

    // Lowest priority policy: TX 200 should be victim (has priority 100)
    assert_eq!(
        deadlocks[0].victim_tx_id, 200,
        "lowest priority transaction should be victim"
    );
}

#[test]
fn test_2pc_no_false_positive_after_release() {
    let coordinator = create_test_coordinator();

    // TX 100 acquires and releases lock
    let handle = coordinator
        .lock_manager()
        .try_lock(100, &["key_a".to_string()])
        .expect("lock should succeed");

    // TX 200 tries and fails (creates wait edge)
    let result = coordinator.lock_manager().try_lock_with_wait_tracking(
        200,
        &["key_a".to_string()],
        coordinator.wait_graph(),
        None,
    );
    assert!(result.is_err(), "TX 200 should fail");

    // TX 100 releases with wait cleanup
    coordinator
        .lock_manager()
        .release_by_handle_with_wait_cleanup(handle, coordinator.wait_graph());

    // TX 200 should now be able to acquire
    let result = coordinator.lock_manager().try_lock_with_wait_tracking(
        200,
        &["key_a".to_string()],
        coordinator.wait_graph(),
        None,
    );
    assert!(result.is_ok(), "TX 200 should succeed after release");

    // No cycles should exist
    let cycles = coordinator.wait_graph().detect_cycles();
    assert!(cycles.is_empty(), "no cycles should exist");
}

#[test]
fn test_2pc_multiple_independent_deadlocks() {
    let detector = DeadlockDetector::with_defaults();
    let graph = detector.graph();

    // Deadlock 1: TX 1 <-> TX 2
    graph.add_wait(1, 2, None);
    graph.add_wait(2, 1, None);

    // Deadlock 2: TX 3 <-> TX 4 (independent)
    graph.add_wait(3, 4, None);
    graph.add_wait(4, 3, None);

    let deadlocks = detector.detect();

    // Should detect both independent deadlocks
    assert!(
        deadlocks.len() >= 2,
        "should detect multiple independent deadlocks"
    );

    // Collect all transactions involved in detected deadlocks
    let involved: HashSet<_> = deadlocks.iter().flat_map(|dl| dl.cycle.iter()).collect();
    assert!(
        involved.contains(&1) || involved.contains(&2),
        "should include deadlock 1"
    );
    assert!(
        involved.contains(&3) || involved.contains(&4),
        "should include deadlock 2"
    );
}

#[test]
fn test_2pc_deadlock_stats_tracking() {
    let detector = DeadlockDetector::with_defaults();
    let graph = detector.graph();

    // Create a cycle
    graph.add_wait(1, 2, None);
    graph.add_wait(2, 1, None);

    // Run detection multiple times
    for _ in 0..3 {
        let _ = detector.detect();
    }

    let stats = detector.stats().snapshot();
    assert!(stats.detection_cycles >= 3, "should track detection cycles");
    assert!(
        stats.deadlocks_detected >= 3,
        "should track detected deadlocks"
    );
}

#[test]
fn test_2pc_would_create_cycle_prevention() {
    let graph = WaitForGraph::new();

    // Create a chain: TX1 -> TX2 -> TX3
    graph.add_wait(1, 2, None);
    graph.add_wait(2, 3, None);

    // Check: adding TX3 -> TX1 would create a cycle
    assert!(
        graph.would_create_cycle(3, 1),
        "should detect potential cycle"
    );

    // Check: adding TX4 -> TX1 would NOT create a cycle
    assert!(
        !graph.would_create_cycle(4, 1),
        "should not falsely detect cycle"
    );

    // Self-loop always creates cycle
    assert!(
        graph.would_create_cycle(1, 1),
        "self-wait should create cycle"
    );
}
