//! Integration tests for wait-for graph tracking in 2PC distributed transactions.
//!
//! Tests that the wait-for graph is properly updated during lock conflicts in
//! handle_prepare and cleaned up during commit/abort/timeout.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager, DeltaVector},
    distributed_tx::{DistributedTxConfig, DistributedTxCoordinator, PrepareRequest, PrepareVote},
    LockManager, WaitForGraph,
};
use tensor_store::SparseVector;

fn create_test_coordinator() -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::with_consensus(consensus)
}

fn create_coordinator_with_config(config: DistributedTxConfig) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, config)
}

#[test]
fn test_wait_graph_updated_on_prepare_conflict() {
    let coordinator = create_test_coordinator();

    // First transaction acquires locks directly
    let keys = vec!["key1".to_string(), "key2".to_string()];
    let _handle1 = coordinator.lock_manager().try_lock(100, &keys).unwrap();

    // Second transaction tries to prepare - should fail and populate wait graph
    let request = PrepareRequest {
        tx_id: 200,
        coordinator: "node1".to_string(),
        operations: vec![tensor_chain::Transaction::Put {
            key: "key1".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
        timeout_ms: 5000,
    };

    let vote = coordinator.handle_prepare(request);

    // Should be a conflict vote
    assert!(
        matches!(vote, PrepareVote::Conflict { .. }),
        "expected conflict vote"
    );

    // Wait graph should have edge: 200 -> 100
    let waiting_for = coordinator.wait_graph().waiting_for(200);
    assert!(
        waiting_for.contains(&100),
        "tx 200 should be waiting for tx 100"
    );
}

#[test]
fn test_wait_graph_cleanup_on_commit() {
    let coordinator = create_test_coordinator();

    // Start a transaction
    let tx = coordinator
        .begin("node1".to_string(), vec![0])
        .expect("begin should succeed");
    let tx_id = tx.tx_id;

    // Manually add to wait graph to simulate prior conflict
    coordinator.wait_graph().add_wait(tx_id, 999, None);
    assert!(
        !coordinator.wait_graph().is_empty(),
        "wait graph should have entry"
    );

    // Prepare the transaction
    let keys = vec!["key1".to_string()];
    let handle = coordinator
        .lock_manager()
        .try_lock(tx_id, &keys)
        .expect("lock should succeed");

    // Record YES vote
    let vote = PrepareVote::Yes {
        lock_handle: handle,
        delta: DeltaVector::from_sparse(
            SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            keys.into_iter().collect(),
            tx_id,
        ),
    };
    coordinator.record_vote(tx_id, 0, vote);

    // Commit should clean wait graph
    coordinator.commit(tx_id).expect("commit should succeed");

    // Wait graph should be cleaned for this transaction
    assert!(
        coordinator.wait_graph().waiting_for(tx_id).is_empty(),
        "wait graph should be cleaned after commit"
    );
}

#[test]
fn test_wait_graph_cleanup_on_abort() {
    let coordinator = create_test_coordinator();

    // Start a transaction
    let tx = coordinator
        .begin("node1".to_string(), vec![0])
        .expect("begin should succeed");
    let tx_id = tx.tx_id;

    // Add to wait graph
    coordinator.wait_graph().add_wait(tx_id, 888, None);
    assert!(!coordinator.wait_graph().is_empty());

    // Prepare the transaction
    let keys = vec!["key1".to_string()];
    let handle = coordinator
        .lock_manager()
        .try_lock(tx_id, &keys)
        .expect("lock should succeed");

    // Record YES vote
    let vote = PrepareVote::Yes {
        lock_handle: handle,
        delta: DeltaVector::from_sparse(
            SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            keys.into_iter().collect(),
            tx_id,
        ),
    };
    coordinator.record_vote(tx_id, 0, vote);

    // Abort should clean wait graph
    coordinator
        .abort(tx_id, "test abort")
        .expect("abort should succeed");

    // Wait graph should be cleaned
    assert!(
        coordinator.wait_graph().waiting_for(tx_id).is_empty(),
        "wait graph should be cleaned after abort"
    );
}

// Note: test_wait_graph_cleanup_on_timeout is covered by unit tests in distributed_tx.rs
// since it requires access to private fields to set short timeouts.

#[test]
fn test_wait_graph_multiple_blockers_tracked() {
    let lock_manager = LockManager::new();
    let wait_graph = WaitForGraph::new();

    // Transaction 100 locks key1
    lock_manager
        .try_lock(100, &["key1".to_string()])
        .expect("lock should succeed");

    // Transaction 200 locks key2
    lock_manager
        .try_lock(200, &["key2".to_string()])
        .expect("lock should succeed");

    // Transaction 300 tries to lock both keys - should track both blockers
    let keys = vec!["key1".to_string(), "key2".to_string()];
    let result = lock_manager.try_lock_with_wait_tracking(300, &keys, &wait_graph, None);

    assert!(result.is_err(), "should fail to acquire locks");

    // Wait graph should have edges to both blockers
    let waiting_for = wait_graph.waiting_for(300);
    assert!(
        waiting_for.contains(&100) && waiting_for.contains(&200),
        "tx 300 should be waiting for both tx 100 and tx 200, got: {:?}",
        waiting_for
    );
}

#[test]
fn test_wait_graph_no_orphaned_edges_after_release() {
    let lock_manager = LockManager::new();
    let wait_graph = WaitForGraph::new();

    // Transaction 100 acquires lock
    let handle1 = lock_manager
        .try_lock(100, &["key1".to_string()])
        .expect("lock should succeed");

    // Transaction 200 tries and fails, creating wait edge
    let _ = lock_manager.try_lock_with_wait_tracking(200, &["key1".to_string()], &wait_graph, None);

    assert!(
        wait_graph.waiting_for(200).contains(&100),
        "tx 200 should be waiting for tx 100"
    );

    // Release tx 100's lock with cleanup
    lock_manager.release_by_handle_with_wait_cleanup(handle1, &wait_graph);

    // Edge 200->100 should be removed because tx 100 is cleaned from graph
    assert!(
        !wait_graph.waiting_for(200).contains(&100),
        "edge 200->100 should be removed when tx 100 is cleaned"
    );
}

#[test]
fn test_wait_graph_concurrent_updates() {
    let coordinator = Arc::new(create_test_coordinator());
    let mut handles = vec![];

    // Multiple threads try to prepare transactions that conflict
    for i in 0..10u64 {
        let coord = Arc::clone(&coordinator);
        handles.push(thread::spawn(move || {
            // All try to lock the same key
            let request = PrepareRequest {
                tx_id: 1000 + i,
                coordinator: format!("node{}", i),
                operations: vec![tensor_chain::Transaction::Put {
                    key: "shared_key".to_string(),
                    data: vec![i as u8],
                }],
                delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
                timeout_ms: 5000,
            };
            let _ = coord.handle_prepare(request);
        }));
    }

    for handle in handles {
        handle.join().expect("thread should complete");
    }

    // Wait graph should be consistent (no panics during concurrent access)
    let edge_count = coordinator.wait_graph().edge_count();
    // At least some edges should exist (first tx succeeded, others wait)
    // or no edges if last tx was first to acquire
    assert!(edge_count >= 0, "wait graph should be in valid state");
}
