// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for deadlock detection in 2PC.
//!
//! Tests the wait-for graph, cycle detection, and victim selection
//! in distributed transaction scenarios.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tensor_chain::{
    DeadlockDetector, DeadlockDetectorConfig, LockManager, VictimSelectionPolicy, WaitForGraph,
};

/// Create a LockManager for testing.
fn test_lock_manager() -> LockManager {
    LockManager::new()
}

#[test]
fn test_two_transaction_deadlock() {
    let detector = DeadlockDetector::with_defaults();
    let graph = detector.graph();

    // TX1 holds key_a, waiting for key_b (held by TX2)
    // TX2 holds key_b, waiting for key_a (held by TX1)
    graph.add_wait(1, 2, None); // TX1 waiting for TX2
    graph.add_wait(2, 1, None); // TX2 waiting for TX1

    let deadlocks = detector.detect();

    assert!(!deadlocks.is_empty(), "should detect deadlock");
    let dl = &deadlocks[0];
    assert!(dl.cycle.contains(&1) || dl.cycle.contains(&2));
    assert!(dl.victim_tx_id == 1 || dl.victim_tx_id == 2);
}

#[test]
fn test_three_transaction_cycle() {
    let detector = DeadlockDetector::with_defaults();
    let graph = detector.graph();

    // TX1 -> TX2 -> TX3 -> TX1
    graph.add_wait(1, 2, None);
    graph.add_wait(2, 3, None);
    graph.add_wait(3, 1, None);

    let deadlocks = detector.detect();

    assert!(!deadlocks.is_empty(), "should detect 3-node cycle");
}

#[test]
fn test_victim_selection_youngest() {
    let config = DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::Youngest);
    let detector = DeadlockDetector::new(config);
    let graph = detector.graph();

    // TX1 starts waiting first
    graph.add_wait(1, 2, None);
    thread::sleep(Duration::from_millis(20));
    // TX2 starts waiting later (younger)
    graph.add_wait(2, 1, None);

    let deadlocks = detector.detect();

    assert!(!deadlocks.is_empty());
    // TX2 should be victim (youngest - most recent wait start)
    assert_eq!(deadlocks[0].victim_tx_id, 2);
}

#[test]
fn test_victim_selection_oldest() {
    let config = DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::Oldest);
    let detector = DeadlockDetector::new(config);
    let graph = detector.graph();

    // TX1 starts waiting first (oldest)
    graph.add_wait(1, 2, None);
    thread::sleep(Duration::from_millis(20));
    // TX2 starts waiting later
    graph.add_wait(2, 1, None);

    let deadlocks = detector.detect();

    assert!(!deadlocks.is_empty());
    // TX1 should be victim (oldest - earliest wait start)
    assert_eq!(deadlocks[0].victim_tx_id, 1);
}

#[test]
fn test_victim_selection_priority() {
    let config =
        DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::LowestPriority);
    let detector = DeadlockDetector::new(config);
    let graph = detector.graph();

    // TX1 has high priority (low value)
    graph.add_wait(1, 2, Some(1));
    // TX2 has low priority (high value) - should be victim
    graph.add_wait(2, 1, Some(100));

    let deadlocks = detector.detect();

    assert!(!deadlocks.is_empty());
    // TX2 has lowest priority, should be victim
    assert_eq!(deadlocks[0].victim_tx_id, 2);
}

#[test]
fn test_victim_selection_most_locks() {
    let config = DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::MostLocks);
    let mut detector = DeadlockDetector::new(config);

    detector.graph().add_wait(1, 2, None);
    detector.graph().add_wait(2, 1, None);

    // TX1 holds 2 locks, TX2 holds 10 locks
    detector.set_lock_count_fn(|tx_id| if tx_id == 1 { 2 } else { 10 });

    let deadlocks = detector.detect();

    assert!(!deadlocks.is_empty());
    // TX2 holds more locks, should be victim
    assert_eq!(deadlocks[0].victim_tx_id, 2);
}

#[test]
fn test_no_false_positive_sequential_locks() {
    let lock_manager = test_lock_manager();
    let graph = WaitForGraph::new();

    // TX1 acquires key_a
    let handle1 = lock_manager.try_lock(1, &["key_a".to_string()]).unwrap();

    // TX1 releases key_a
    lock_manager.release_by_handle(handle1);

    // TX2 acquires key_a (now available)
    let result = lock_manager.try_lock(2, &["key_a".to_string()]);
    assert!(result.is_ok());

    // No cycle should exist
    let detector = DeadlockDetector::with_defaults();
    assert!(detector.detect().is_empty());
    assert!(graph.detect_cycles().is_empty());
}

#[test]
fn test_lock_manager_with_wait_tracking() {
    let lock_manager = test_lock_manager();
    let graph = WaitForGraph::new();

    // TX1 acquires key_a
    let result = lock_manager.try_lock_with_wait_tracking(1, &["key_a".to_string()], &graph, None);
    assert!(result.is_ok());

    // TX2 tries to acquire key_a - should fail and update wait graph
    let result = lock_manager.try_lock_with_wait_tracking(2, &["key_a".to_string()], &graph, None);
    assert!(result.is_err());

    let wait_info = result.unwrap_err();
    assert_eq!(wait_info.blocking_tx_id, 1);
    assert!(wait_info.conflicting_keys.contains(&"key_a".to_string()));

    // Wait graph should show TX2 waiting for TX1
    assert!(graph.waiting_for(2).contains(&1));
}

#[test]
fn test_lock_manager_keys_for_transaction() {
    let lock_manager = test_lock_manager();

    // TX1 acquires multiple keys
    lock_manager
        .try_lock(
            1,
            &[
                "key_a".to_string(),
                "key_b".to_string(),
                "key_c".to_string(),
            ],
        )
        .unwrap();

    let keys = lock_manager.keys_for_transaction(1);
    assert_eq!(keys.len(), 3);
    assert!(keys.contains(&"key_a".to_string()));
    assert!(keys.contains(&"key_b".to_string()));
    assert!(keys.contains(&"key_c".to_string()));
}

#[test]
fn test_lock_manager_lock_count_for_transaction() {
    let lock_manager = test_lock_manager();

    // TX1 acquires 3 keys
    lock_manager
        .try_lock(
            1,
            &[
                "key_a".to_string(),
                "key_b".to_string(),
                "key_c".to_string(),
            ],
        )
        .unwrap();

    assert_eq!(lock_manager.lock_count_for_transaction(1), 3);
    assert_eq!(lock_manager.lock_count_for_transaction(2), 0); // TX2 has no locks
}

#[test]
fn test_wait_graph_cleanup_on_commit() {
    let lock_manager = test_lock_manager();
    let graph = WaitForGraph::new();

    // TX1 acquires key_a
    lock_manager.try_lock(1, &["key_a".to_string()]).unwrap();

    // TX2 tries to acquire key_a - fails and adds wait edge
    let _ = lock_manager.try_lock_with_wait_tracking(2, &["key_a".to_string()], &graph, None);
    assert!(graph.waiting_for(2).contains(&1));

    // TX1 commits (releases locks)
    lock_manager.release(1);

    // Simulate TX2 retry after TX1 commits - should succeed
    let result = lock_manager.try_lock_with_wait_tracking(2, &["key_a".to_string()], &graph, None);
    assert!(result.is_ok());

    // Wait graph should be cleaned up
    assert!(graph.waiting_for(2).is_empty());
}

#[test]
fn test_deadlock_with_lock_manager() {
    let lock_manager = test_lock_manager();
    let graph = Arc::new(WaitForGraph::new());

    // TX1 acquires key_a
    lock_manager.try_lock(1, &["key_a".to_string()]).unwrap();
    // TX2 acquires key_b
    lock_manager.try_lock(2, &["key_b".to_string()]).unwrap();

    // TX1 tries to acquire key_b (held by TX2)
    let result = lock_manager.try_lock_with_wait_tracking(1, &["key_b".to_string()], &graph, None);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().blocking_tx_id, 2);

    // TX2 tries to acquire key_a (held by TX1) - DEADLOCK
    let result = lock_manager.try_lock_with_wait_tracking(2, &["key_a".to_string()], &graph, None);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().blocking_tx_id, 1);

    // Detect the cycle
    let _detector = DeadlockDetector::with_defaults();
    // Use the same graph
    let cycles = graph.detect_cycles();
    assert!(!cycles.is_empty(), "should detect deadlock cycle");
}

#[test]
fn test_detection_performance_100_transactions() {
    let detector = DeadlockDetector::with_defaults();
    let graph = detector.graph();

    // Create a chain of 100 transactions (no cycle)
    for i in 1..100u64 {
        graph.add_wait(i, i + 1, None);
    }

    let start = std::time::Instant::now();
    let deadlocks = detector.detect();
    let elapsed = start.elapsed();

    // Should complete in < 100ms
    assert!(
        elapsed < Duration::from_millis(100),
        "detection took too long: {:?}",
        elapsed
    );
    assert!(deadlocks.is_empty(), "no cycle in chain");
}

#[test]
fn test_detection_performance_100_transactions_with_cycle() {
    let detector = DeadlockDetector::with_defaults();
    let graph = detector.graph();

    // Create a chain of 100 transactions with a cycle at the end
    for i in 1..100u64 {
        graph.add_wait(i, i + 1, None);
    }
    // Close the cycle
    graph.add_wait(100, 1, None);

    let start = std::time::Instant::now();
    let deadlocks = detector.detect();
    let elapsed = start.elapsed();

    // Should complete in < 100ms
    assert!(
        elapsed < Duration::from_millis(100),
        "detection took too long: {:?}",
        elapsed
    );
    assert!(!deadlocks.is_empty(), "should detect cycle");
}

#[test]
fn test_deadlock_stats() {
    let detector = DeadlockDetector::with_defaults();
    let graph = detector.graph();

    // Create a cycle
    graph.add_wait(1, 2, None);
    graph.add_wait(2, 1, None);

    // Run detection
    let _ = detector.detect();

    let stats = detector.stats().snapshot();
    assert!(stats.detection_cycles > 0);
    assert!(stats.deadlocks_detected > 0);
}

#[test]
fn test_multiple_independent_cycles() {
    let detector = DeadlockDetector::with_defaults();
    let graph = detector.graph();

    // Cycle 1: TX1 <-> TX2
    graph.add_wait(1, 2, None);
    graph.add_wait(2, 1, None);

    // Cycle 2: TX3 <-> TX4 (independent)
    graph.add_wait(3, 4, None);
    graph.add_wait(4, 3, None);

    let deadlocks = detector.detect();

    // Should detect both cycles
    assert!(deadlocks.len() >= 2, "should detect multiple cycles");
}

#[test]
fn test_wait_graph_remove_transaction_cleans_edges() {
    let graph = WaitForGraph::new();

    // TX1 -> TX2 -> TX3
    graph.add_wait(1, 2, None);
    graph.add_wait(2, 3, None);
    // TX4 -> TX2
    graph.add_wait(4, 2, None);

    assert_eq!(graph.waiting_on(2).len(), 2); // TX1 and TX4 waiting for TX2

    // Remove TX2
    graph.remove_transaction(2);

    // TX2 should be completely gone
    assert!(graph.waiting_for(2).is_empty());
    assert!(graph.waiting_on(2).is_empty());

    // TX1 and TX4 should no longer be waiting for TX2
    assert!(!graph.waiting_for(1).contains(&2));
    assert!(!graph.waiting_for(4).contains(&2));
}

#[test]
fn test_would_create_cycle_prevention() {
    let graph = WaitForGraph::new();

    // TX1 -> TX2 -> TX3
    graph.add_wait(1, 2, None);
    graph.add_wait(2, 3, None);

    // Check if adding TX3 -> TX1 would create a cycle
    assert!(
        graph.would_create_cycle(3, 1),
        "should detect potential cycle"
    );

    // Adding TX4 -> TX1 should not create a cycle
    assert!(!graph.would_create_cycle(4, 1), "should not create cycle");
}

#[test]
fn test_concurrent_graph_operations() {
    let graph = Arc::new(WaitForGraph::new());
    let mut handles = vec![];

    // Spawn threads adding edges
    for i in 0..20u64 {
        let g = Arc::clone(&graph);
        handles.push(thread::spawn(move || {
            g.add_wait(i, (i + 1) % 20, Some(i as u32));
        }));
    }

    // Spawn threads doing cycle detection
    for _ in 0..5 {
        let g = Arc::clone(&graph);
        handles.push(thread::spawn(move || {
            let _ = g.detect_cycles();
        }));
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Graph should be consistent
    assert!(graph.edge_count() <= 20);
}

#[test]
fn test_disabled_detector() {
    let config = DeadlockDetectorConfig::disabled();
    let detector = DeadlockDetector::new(config);

    // Create a cycle that would be detected
    detector.graph().add_wait(1, 2, None);
    detector.graph().add_wait(2, 1, None);

    // Detection should return empty when disabled
    let deadlocks = detector.detect();
    assert!(deadlocks.is_empty());
}

#[test]
fn test_auto_abort_configuration() {
    let config = DeadlockDetectorConfig::default().without_auto_abort();
    assert!(!config.auto_abort_victim);

    let config = DeadlockDetectorConfig::default();
    assert!(config.auto_abort_victim);
}

#[test]
fn test_max_cycle_length_filter() {
    let config = DeadlockDetectorConfig::default().with_max_cycle_length(3);
    let detector = DeadlockDetector::new(config);
    let graph = detector.graph();

    // Create a 5-node cycle (exceeds max of 3)
    graph.add_wait(1, 2, None);
    graph.add_wait(2, 3, None);
    graph.add_wait(3, 4, None);
    graph.add_wait(4, 5, None);
    graph.add_wait(5, 1, None);

    let deadlocks = detector.detect();

    // Long cycles should be filtered
    for dl in &deadlocks {
        assert!(dl.cycle.len() <= 3, "cycle too long: {:?}", dl.cycle);
    }
}
