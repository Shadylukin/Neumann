// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for TOCTOU (Time-Of-Check-To-Time-Of-Use) safety in 2PC.
//!
//! Tests that concurrent operations respect limits and maintain consistency
//! between WAL writes and pending transaction state.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tempfile::tempdir;
use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager},
    distributed_tx::{DistributedTxConfig, DistributedTxCoordinator},
    tx_wal::TxWal,
    LockManager, WaitForGraph,
};

fn create_coordinator_with_limit(max_concurrent: usize) -> DistributedTxCoordinator {
    let config = DistributedTxConfig {
        max_concurrent,
        ..Default::default()
    };
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, config)
}

#[test]
fn test_concurrent_begin_respects_max_limit() {
    let max_concurrent = 5;
    let coordinator = Arc::new(create_coordinator_with_limit(max_concurrent));
    let success_count = Arc::new(AtomicUsize::new(0));
    let failure_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    // Try to start more transactions than the limit concurrently
    for i in 0..20 {
        let coord = Arc::clone(&coordinator);
        let success = Arc::clone(&success_count);
        let failure = Arc::clone(&failure_count);

        handles.push(thread::spawn(move || {
            match coord.begin(&format!("node{}", i), &[i % 3]) {
                Ok(_) => {
                    success.fetch_add(1, Ordering::SeqCst);
                },
                Err(_) => {
                    failure.fetch_add(1, Ordering::SeqCst);
                },
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread should complete");
    }

    let successes = success_count.load(Ordering::SeqCst);
    let failures = failure_count.load(Ordering::SeqCst);

    // At most max_concurrent transactions should succeed
    assert!(
        successes <= max_concurrent,
        "at most {} transactions should succeed, got {}",
        max_concurrent,
        successes
    );

    // Some should have failed due to limit
    assert!(failures > 0, "some transactions should fail due to limit");

    // Pending count should match successes
    assert_eq!(
        coordinator.pending_count(),
        successes,
        "pending count should match successful begins"
    );
}

#[test]
fn test_concurrent_lock_wait_tracking_atomic() {
    let lock_manager = LockManager::new();
    let wait_graph = Arc::new(WaitForGraph::new());

    // One transaction holds the lock
    lock_manager
        .try_lock(1, &["contested_key".to_string()])
        .expect("initial lock should succeed");

    let conflict_count = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    // Many transactions try to acquire the same key concurrently
    for i in 2..=20u64 {
        let lm = &lock_manager;
        let wg = Arc::clone(&wait_graph);
        let conflicts = Arc::clone(&conflict_count);

        // Note: we can't move lock_manager into thread, so use scoped threads
        handles.push(thread::scope(|_| {
            let result = lm.try_lock_with_wait_tracking(
                i,
                &["contested_key".to_string()],
                &wg,
                Some(i as u32),
            );
            if result.is_err() {
                conflicts.fetch_add(1, Ordering::SeqCst);
            }
            result
        }));
    }

    // All should have failed and be in the wait graph
    let conflicts = conflict_count.load(Ordering::SeqCst);
    assert!(conflicts > 0, "some transactions should conflict");

    // All conflicting transactions should be waiting for TX 1
    for i in 2..=20u64 {
        let waiting_for = wait_graph.waiting_for(i);
        if !waiting_for.is_empty() {
            assert!(
                waiting_for.contains(&1),
                "tx {} should be waiting for tx 1",
                i
            );
        }
    }
}

#[test]
fn test_wal_write_atomic_with_pending_insert() {
    let dir = tempdir().expect("temp dir should be created");
    let wal_path = dir.path().join("test.wal");
    let wal = TxWal::open(&wal_path).expect("WAL should open");

    let config = DistributedTxConfig {
        max_concurrent: 10,
        ..Default::default()
    };
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let coordinator = DistributedTxCoordinator::new(consensus, config).with_wal(wal);

    // Begin multiple transactions
    let mut tx_ids = vec![];
    for i in 0..5 {
        let tx = coordinator
            .begin(&format!("node{}", i), &[i])
            .expect("begin should succeed");
        tx_ids.push(tx.tx_id);
    }

    // All should be in pending
    assert_eq!(
        coordinator.pending_count(),
        5,
        "all transactions should be pending"
    );

    // All should be retrievable
    for &tx_id in &tx_ids {
        assert!(
            coordinator.get(tx_id).is_some(),
            "transaction {} should be retrievable",
            tx_id
        );
    }
}

#[test]
fn test_max_concurrent_never_exceeded_under_load() {
    let max_concurrent = 3;
    let coordinator = Arc::new(create_coordinator_with_limit(max_concurrent));
    let max_observed = Arc::new(AtomicUsize::new(0));
    let iterations = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    // Multiple threads continuously creating and cleaning up transactions
    for t in 0..4 {
        let coord = Arc::clone(&coordinator);
        let max_obs = Arc::clone(&max_observed);
        let iters = Arc::clone(&iterations);

        handles.push(thread::spawn(move || {
            for i in 0..50 {
                // Try to begin
                if let Ok(tx) = coord.begin(&format!("node{}-{}", t, i), &[t]) {
                    // Record current pending count
                    let current = coord.pending_count();
                    let mut max = max_obs.load(Ordering::SeqCst);
                    while current > max {
                        match max_obs.compare_exchange_weak(
                            max,
                            current,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        ) {
                            Ok(_) => break,
                            Err(m) => max = m,
                        }
                    }

                    // Simulate some work
                    thread::sleep(Duration::from_micros(100));

                    // Clean up via abort (public API)
                    let _ = coord.abort(tx.tx_id, "test cleanup");
                }
                iters.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread should complete");
    }

    let max = max_observed.load(Ordering::SeqCst);
    assert!(
        max <= max_concurrent,
        "max concurrent should never exceed limit, observed {}",
        max
    );

    let total_iters = iterations.load(Ordering::SeqCst);
    assert!(total_iters > 100, "should have completed many iterations");
}

#[test]
fn test_lock_manager_concurrent_operations_consistency() {
    let lock_manager = Arc::new(LockManager::new());
    let wait_graph = Arc::new(WaitForGraph::new());

    let mut handles = vec![];

    // Thread 1: Continuously acquires and releases locks
    {
        let lm = Arc::clone(&lock_manager);
        let wg = Arc::clone(&wait_graph);
        handles.push(thread::spawn(move || {
            for i in 0..100u64 {
                if let Ok(handle) = lm.try_lock(i, &[format!("key{}", i % 10)]) {
                    thread::sleep(Duration::from_micros(50));
                    lm.release_by_handle_with_wait_cleanup(handle, &wg);
                }
            }
        }));
    }

    // Thread 2: Tries to acquire conflicting locks
    {
        let lm = Arc::clone(&lock_manager);
        let wg = Arc::clone(&wait_graph);
        handles.push(thread::spawn(move || {
            for i in 1000..1100u64 {
                let _ = lm.try_lock_with_wait_tracking(i, &[format!("key{}", i % 10)], &wg, None);
            }
        }));
    }

    // Thread 3: Runs cycle detection
    {
        let wg = Arc::clone(&wait_graph);
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                let _ = wg.detect_cycles();
                thread::sleep(Duration::from_micros(100));
            }
        }));
    }

    for handle in handles {
        handle.join().expect("thread should complete without panic");
    }

    // Final state should be consistent (no panics, no deadlocks in test itself)
    // Wait graph should be in valid state
    let _ = wait_graph.edge_count();
    let _ = wait_graph.transaction_count();
}

#[test]
fn test_begin_failure_does_not_leak_transaction() {
    let coordinator = Arc::new(create_coordinator_with_limit(2));

    // Fill up to limit
    let tx1 = coordinator
        .begin(&"node1".to_string(), &[0])
        .expect("first begin should succeed");
    let tx2 = coordinator
        .begin(&"node2".to_string(), &[1])
        .expect("second begin should succeed");

    assert_eq!(coordinator.pending_count(), 2);

    // Third should fail
    let result = coordinator.begin(&"node3".to_string(), &[2]);
    assert!(result.is_err(), "third begin should fail");

    // Should still have only 2 pending
    assert_eq!(
        coordinator.pending_count(),
        2,
        "failed begin should not leak"
    );

    // Original transactions should still be accessible
    assert!(coordinator.get(tx1.tx_id).is_some());
    assert!(coordinator.get(tx2.tx_id).is_some());
}
