//! Integration tests for distributed transaction concurrency safety.
//!
//! Tests the fixes for TOCTOU race conditions:
//! - Max concurrent transaction limit enforcement
//! - Wait-for graph atomicity with lock acquisition

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use tensor_chain::{
    deadlock::WaitForGraph, ConsensusConfig, ConsensusManager, DistributedTxConfig,
    DistributedTxCoordinator, LockManager,
};

fn create_coordinator_with_limit(max_concurrent: usize) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(
        consensus,
        DistributedTxConfig {
            max_concurrent,
            ..Default::default()
        },
    )
}

#[test]
fn test_max_concurrent_enforced_under_contention() {
    let max = 10;
    let coordinator = Arc::new(create_coordinator_with_limit(max));
    let success_count = Arc::new(AtomicUsize::new(0));
    let failure_count = Arc::new(AtomicUsize::new(0));

    // 50 threads all try to begin transactions simultaneously
    let handles: Vec<_> = (0..50)
        .map(|_| {
            let coord = Arc::clone(&coordinator);
            let successes = Arc::clone(&success_count);
            let failures = Arc::clone(&failure_count);
            thread::spawn(move || {
                match coord.begin("node".to_string(), vec![0]) {
                    Ok(_) => successes.fetch_add(1, Ordering::Relaxed),
                    Err(_) => failures.fetch_add(1, Ordering::Relaxed),
                };
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let successes = success_count.load(Ordering::Relaxed);
    let failures = failure_count.load(Ordering::Relaxed);

    assert_eq!(successes, max, "Exactly max_concurrent should succeed");
    assert_eq!(failures, 40, "Rest should fail");
    assert_eq!(coordinator.pending_count(), max);
}

#[test]
fn test_wait_graph_consistency_under_lock_contention() {
    let lock_manager = Arc::new(LockManager::new());
    let wait_graph = Arc::new(WaitForGraph::new());

    // TX1 holds the contested lock
    lock_manager
        .try_lock(1, &["contested_key".to_string()])
        .unwrap();

    let contenders = 20;
    let wait_edges = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (2..contenders + 2)
        .map(|tx_id| {
            let lm = Arc::clone(&lock_manager);
            let wg = Arc::clone(&wait_graph);
            let edges = Arc::clone(&wait_edges);
            thread::spawn(move || {
                // All will fail to acquire
                let _ = lm.try_lock_with_wait_tracking(
                    tx_id,
                    &["contested_key".to_string()],
                    &wg,
                    None,
                );
                // Check our wait edge was added
                if wg.waiting_for(tx_id).contains(&1) {
                    edges.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // All contenders should have proper wait edges
    assert_eq!(wait_edges.load(Ordering::Relaxed), contenders as usize);
}

#[test]
fn test_lock_manager_active_locks_via_accessor() {
    let coordinator = create_coordinator_with_limit(100);

    // Create a transaction and acquire locks via coordinator
    let tx = coordinator.begin("node".to_string(), vec![0]).unwrap();
    coordinator
        .lock_manager()
        .try_lock(tx.tx_id, &["key_a".to_string()])
        .unwrap();

    assert_eq!(coordinator.lock_manager().active_lock_count(), 1);

    // Release via abort
    coordinator.abort(tx.tx_id, "test cleanup").unwrap();
    assert_eq!(coordinator.pending_count(), 0);
}

#[test]
fn test_multiple_blockers_all_added_to_wait_graph() {
    let lock_manager = LockManager::new();
    let wait_graph = WaitForGraph::new();

    // TX1 and TX2 hold different keys
    lock_manager.try_lock(1, &["key_a".to_string()]).unwrap();
    lock_manager.try_lock(2, &["key_b".to_string()]).unwrap();

    // TX3 tries to acquire both keys - should record both blockers
    let result = lock_manager.try_lock_with_wait_tracking(
        3,
        &["key_a".to_string(), "key_b".to_string()],
        &wait_graph,
        None,
    );

    assert!(result.is_err());
    let waiting_for = wait_graph.waiting_for(3);
    // TX3 should be waiting for both TX1 and TX2
    assert!(
        waiting_for.contains(&1) && waiting_for.contains(&2),
        "TX3 should be waiting for both TX1 and TX2, got: {:?}",
        waiting_for
    );
}

#[test]
fn test_wait_graph_removed_on_successful_lock() {
    let lock_manager = LockManager::new();
    let wait_graph = WaitForGraph::new();

    // Manually add a wait edge for TX2 (simulating previous conflict)
    wait_graph.add_wait(2, 99, None);
    assert!(wait_graph.waiting_for(2).contains(&99));

    // TX2 successfully acquires lock (no conflict)
    let result =
        lock_manager.try_lock_with_wait_tracking(2, &["free_key".to_string()], &wait_graph, None);

    assert!(result.is_ok());
    // Wait edge should be removed after successful acquisition
    assert!(
        wait_graph.waiting_for(2).is_empty(),
        "Wait edges should be cleared after successful lock"
    );
}

#[test]
fn test_concurrent_begin_max_limit_strictly_enforced() {
    // Test that even under high contention, we never exceed the limit
    let max = 5;
    let coordinator = Arc::new(create_coordinator_with_limit(max));
    let success_count = Arc::new(AtomicUsize::new(0));

    // 100 threads all try to begin transactions simultaneously
    let handles: Vec<_> = (0..100)
        .map(|_| {
            let coord = Arc::clone(&coordinator);
            let counter = Arc::clone(&success_count);
            thread::spawn(move || {
                if coord.begin("node".to_string(), vec![0]).is_ok() {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Exactly max should succeed, never more
    assert_eq!(
        success_count.load(Ordering::Relaxed),
        max,
        "Should have exactly max_concurrent successful begins"
    );

    // Pending count should match
    assert_eq!(
        coordinator.pending_count(),
        max,
        "Pending count should equal max_concurrent"
    );
}

#[test]
fn test_lock_manager_wait_tracking_stress() {
    // Stress test for wait-for graph consistency
    let lock_manager = Arc::new(LockManager::new());
    let wait_graph = Arc::new(WaitForGraph::new());

    // TX1 holds the lock
    lock_manager.try_lock(1, &["key".to_string()]).unwrap();

    let num_contenders = 50;
    let successful_waits = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (2..=num_contenders + 1)
        .map(|tx_id| {
            let lm = Arc::clone(&lock_manager);
            let wg = Arc::clone(&wait_graph);
            let counter = Arc::clone(&successful_waits);
            thread::spawn(move || {
                let result =
                    lm.try_lock_with_wait_tracking(tx_id as u64, &["key".to_string()], &wg, None);
                if result.is_err() && wg.waiting_for(tx_id as u64).contains(&1) {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // All contenders should have proper wait edges
    assert_eq!(
        successful_waits.load(Ordering::Relaxed),
        num_contenders,
        "All contenders should be waiting for TX1"
    );
}
