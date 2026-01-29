// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for cross-shard distributed transactions with 2PC.
//!
//! Tests the two-phase commit protocol with delta-based conflict detection.

use std::{sync::atomic::Ordering, time::Duration};

use tensor_chain::{
    ConsensusConfig, ConsensusManager, DeltaVector, DistributedTransaction, DistributedTxConfig,
    DistributedTxCoordinator, DistributedTxStats, LockManager, PrepareRequest, PrepareVote,
    Transaction, TxParticipant, TxPhase,
};
use tensor_store::SparseVector;

// ============================================================================
// Helper Functions
// ============================================================================

fn create_coordinator() -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::with_consensus(consensus)
}

fn create_coordinator_with_config(config: DistributedTxConfig) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, config)
}

fn create_delta(embedding: Vec<f32>, keys: Vec<&str>, tx_id: u64) -> DeltaVector {
    DeltaVector::new(
        embedding,
        keys.into_iter().map(|s| s.to_string()).collect(),
        tx_id,
    )
}

// ============================================================================
// Distributed Transaction Creation Tests
// ============================================================================

#[test]
fn test_distributed_tx_basic_creation() {
    let tx = DistributedTransaction::new("coordinator".to_string(), vec![0, 1, 2]);

    assert_eq!(tx.coordinator, "coordinator");
    assert_eq!(tx.participants, vec![0, 1, 2]);
    assert_eq!(tx.phase, TxPhase::Preparing);
    assert!(tx.operations.is_empty());
    assert!(tx.deltas.is_empty());
    assert!(tx.votes.is_empty());
}

#[test]
fn test_distributed_tx_add_operations_per_shard() {
    let mut tx = DistributedTransaction::new("coord".to_string(), vec![0, 1]);

    // Add operations for shard 0
    tx.add_operations(
        0,
        vec![
            Transaction::Put {
                key: "users:1".to_string(),
                data: vec![1, 2, 3],
            },
            Transaction::Put {
                key: "users:2".to_string(),
                data: vec![4, 5, 6],
            },
        ],
    );

    // Add operations for shard 1
    tx.add_operations(
        1,
        vec![Transaction::Put {
            key: "orders:100".to_string(),
            data: vec![7, 8, 9],
        }],
    );

    assert_eq!(tx.operations.len(), 2);
    assert_eq!(tx.operations.get(&0).unwrap().len(), 2);
    assert_eq!(tx.operations.get(&1).unwrap().len(), 1);
}

#[test]
fn test_distributed_tx_delta_tracking() {
    let mut tx = DistributedTransaction::new("coord".to_string(), vec![0, 1, 2]);

    // Add deltas from each shard
    let delta0 = create_delta(vec![1.0, 0.0, 0.0], vec!["key1"], 1);
    let delta1 = create_delta(vec![0.0, 1.0, 0.0], vec!["key2"], 1);
    let delta2 = create_delta(vec![0.0, 0.0, 1.0], vec!["key3"], 1);

    tx.add_delta(0, delta0);
    tx.add_delta(1, delta1);
    tx.add_delta(2, delta2);

    // Verify affected keys
    let affected = tx.affected_keys();
    assert!(affected.contains("key1"));
    assert!(affected.contains("key2"));
    assert!(affected.contains("key3"));
    assert_eq!(affected.len(), 3);

    // Verify merged delta
    let merged = tx.merged_delta().unwrap();
    assert_eq!(merged.to_dense(3), vec![1.0, 1.0, 1.0]);
}

// ============================================================================
// Lock Manager Tests
// ============================================================================

#[test]
fn test_lock_manager_acquire_release() {
    let lock_manager = LockManager::new();

    // Acquire locks
    let keys = vec!["account:1".to_string(), "account:2".to_string()];
    let handle = lock_manager.try_lock(100, &keys).unwrap();

    assert!(lock_manager.is_locked("account:1"));
    assert!(lock_manager.is_locked("account:2"));
    assert!(!lock_manager.is_locked("account:3"));

    assert_eq!(lock_manager.lock_holder("account:1"), Some(100));
    assert_eq!(lock_manager.lock_holder("account:2"), Some(100));

    // Release locks
    lock_manager.release_by_handle(handle);

    assert!(!lock_manager.is_locked("account:1"));
    assert!(!lock_manager.is_locked("account:2"));
}

#[test]
fn test_lock_manager_conflict_detection() {
    let lock_manager = LockManager::new();

    // First transaction acquires lock
    let tx1_keys = vec!["shared_resource".to_string()];
    lock_manager.try_lock(1, &tx1_keys).unwrap();

    // Second transaction tries to lock same key
    let tx2_keys = vec!["shared_resource".to_string()];
    let result = lock_manager.try_lock(2, &tx2_keys);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), 1); // Conflicting tx_id
}

#[test]
fn test_lock_manager_same_tx_relock() {
    let lock_manager = LockManager::new();

    // Same transaction can lock additional keys
    let keys1 = vec!["key1".to_string()];
    lock_manager.try_lock(1, &keys1).unwrap();

    // Same tx can lock more keys
    let keys2 = vec!["key2".to_string()];
    let result = lock_manager.try_lock(1, &keys2);
    assert!(result.is_ok());

    // But same tx can also re-lock existing keys (idempotent)
    let keys3 = vec!["key1".to_string()];
    let result = lock_manager.try_lock(1, &keys3);
    assert!(result.is_ok());
}

#[test]
fn test_lock_manager_release_by_tx() {
    let lock_manager = LockManager::new();

    let keys = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    lock_manager.try_lock(42, &keys).unwrap();

    assert_eq!(lock_manager.active_lock_count(), 3);

    // Release all locks for this transaction
    lock_manager.release(42);

    assert_eq!(lock_manager.active_lock_count(), 0);
}

// ============================================================================
// Coordinator 2PC Flow Tests
// ============================================================================

#[test]
fn test_coordinator_begin_transaction() {
    let coordinator = create_coordinator();

    let tx = coordinator
        .begin("node1".to_string(), vec![0, 1, 2])
        .unwrap();

    assert_eq!(tx.participants, vec![0, 1, 2]);
    assert_eq!(coordinator.pending_count(), 1);
    assert_eq!(coordinator.stats().started.load(Ordering::Relaxed), 1);
}

#[test]
fn test_coordinator_max_concurrent_limit() {
    let config = DistributedTxConfig {
        max_concurrent: 3,
        ..Default::default()
    };
    let coordinator = create_coordinator_with_config(config);

    // Create max transactions
    coordinator.begin("node1".to_string(), vec![0]).unwrap();
    coordinator.begin("node1".to_string(), vec![1]).unwrap();
    coordinator.begin("node1".to_string(), vec![2]).unwrap();

    // Next one should fail
    let result = coordinator.begin("node1".to_string(), vec![3]);
    assert!(result.is_err());
}

#[test]
fn test_coordinator_prepare_vote_yes() {
    let coordinator = create_coordinator();

    let request = PrepareRequest {
        tx_id: 1,
        coordinator: "coord".to_string(),
        operations: vec![Transaction::Put {
            key: "test_key".to_string(),
            data: vec![1, 2, 3],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0, 0.0]),
        timeout_ms: 5000,
    };

    let vote = coordinator.handle_prepare(request);

    match vote {
        PrepareVote::Yes { lock_handle, delta } => {
            assert!(lock_handle > 0);
            assert!(delta.affected_keys.contains("test_key"));
        },
        _ => panic!("Expected Yes vote"),
    }
}

#[test]
fn test_coordinator_full_commit_flow() {
    let coordinator = create_coordinator();

    // Begin transaction
    let tx = coordinator.begin("coord".to_string(), vec![0, 1]).unwrap();
    let tx_id = tx.tx_id;

    // Simulate votes from shards (orthogonal deltas)
    let delta0 = create_delta(vec![1.0, 0.0, 0.0, 0.0], vec!["shard0_key"], tx_id);
    let delta1 = create_delta(vec![0.0, 1.0, 0.0, 0.0], vec!["shard1_key"], tx_id);

    // Record first vote
    let phase = coordinator.record_vote(
        tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );
    assert!(phase.is_none()); // Not all voted yet

    // Record second vote
    let phase = coordinator.record_vote(
        tx_id,
        1,
        PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );
    assert_eq!(phase, Some(TxPhase::Prepared));

    // Commit
    coordinator.commit(tx_id).unwrap();

    assert_eq!(coordinator.pending_count(), 0);
    assert_eq!(coordinator.stats().committed.load(Ordering::Relaxed), 1);
}

#[test]
fn test_coordinator_abort_on_no_vote() {
    let coordinator = create_coordinator();

    let tx = coordinator.begin("coord".to_string(), vec![0, 1]).unwrap();
    let tx_id = tx.tx_id;

    let delta = create_delta(vec![1.0, 0.0], vec!["key1"], tx_id);

    // First shard votes yes
    coordinator.record_vote(
        tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta,
        },
    );

    // Second shard votes no
    let phase = coordinator.record_vote(
        tx_id,
        1,
        PrepareVote::No {
            reason: "resource unavailable".to_string(),
        },
    );

    assert_eq!(phase, Some(TxPhase::Aborting));
}

#[test]
fn test_coordinator_conflict_vote() {
    let coordinator = create_coordinator();

    let tx = coordinator.begin("coord".to_string(), vec![0, 1]).unwrap();
    let tx_id = tx.tx_id;

    let delta = create_delta(vec![1.0, 0.0], vec!["key1"], tx_id);

    // First shard votes yes
    coordinator.record_vote(
        tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta,
        },
    );

    // Second shard detects conflict
    let phase = coordinator.record_vote(
        tx_id,
        1,
        PrepareVote::Conflict {
            similarity: 0.95,
            conflicting_tx: 999,
        },
    );

    assert_eq!(phase, Some(TxPhase::Aborting));
    assert_eq!(coordinator.stats().conflicts.load(Ordering::Relaxed), 1);
}

#[test]
fn test_coordinator_cross_shard_conflict_detection() {
    let config = DistributedTxConfig {
        orthogonal_threshold: 0.1,
        ..Default::default()
    };
    let coordinator = create_coordinator_with_config(config);

    let tx = coordinator.begin("coord".to_string(), vec![0, 1]).unwrap();
    let tx_id = tx.tx_id;

    // Both shards touch the same key with high similarity deltas
    let delta0 = create_delta(vec![1.0, 0.1, 0.0], vec!["shared_key"], tx_id);
    let delta1 = create_delta(vec![0.9, 0.2, 0.0], vec!["shared_key"], tx_id);

    coordinator.record_vote(
        tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );
    let phase = coordinator.record_vote(
        tx_id,
        1,
        PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );

    // Should abort due to cross-shard conflict on same key
    assert_eq!(phase, Some(TxPhase::Aborting));
}

#[test]
fn test_coordinator_orthogonal_merge_success() {
    let config = DistributedTxConfig {
        orthogonal_threshold: 0.1,
        ..Default::default()
    };
    let coordinator = create_coordinator_with_config(config);

    let tx = coordinator.begin("coord".to_string(), vec![0, 1]).unwrap();
    let tx_id = tx.tx_id;

    // Orthogonal deltas on different keys
    let delta0 = create_delta(vec![1.0, 0.0, 0.0], vec!["key_a"], tx_id);
    let delta1 = create_delta(vec![0.0, 1.0, 0.0], vec!["key_b"], tx_id);

    coordinator.record_vote(
        tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );
    let phase = coordinator.record_vote(
        tx_id,
        1,
        PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );

    assert_eq!(phase, Some(TxPhase::Prepared));
    assert_eq!(
        coordinator
            .stats()
            .orthogonal_merges
            .load(Ordering::Relaxed),
        1
    );
}

// ============================================================================
// Participant Tests
// ============================================================================

#[test]
fn test_participant_prepare_commit() {
    let participant = TxParticipant::new_in_memory();

    let request = PrepareRequest {
        tx_id: 42,
        coordinator: "coord".to_string(),
        operations: vec![Transaction::Put {
            key: "local_data".to_string(),
            data: vec![1, 2, 3],
        }],
        delta_embedding: SparseVector::from_dense(&[0.5, 0.5, 0.0]),
        timeout_ms: 5000,
    };

    // Prepare
    let vote = participant.prepare(request);
    assert!(matches!(vote, PrepareVote::Yes { .. }));
    assert_eq!(participant.prepared_count(), 1);

    // Commit
    let response = participant.commit(42);
    assert!(response.success);
    assert_eq!(participant.prepared_count(), 0);
}

#[test]
fn test_participant_prepare_abort() {
    let participant = TxParticipant::new_in_memory();

    let request = PrepareRequest {
        tx_id: 42,
        coordinator: "coord".to_string(),
        operations: vec![Transaction::Put {
            key: "local_data".to_string(),
            data: vec![1, 2, 3],
        }],
        delta_embedding: SparseVector::from_dense(&[0.5, 0.5, 0.0]),
        timeout_ms: 5000,
    };

    participant.prepare(request);
    assert_eq!(participant.prepared_count(), 1);

    // Abort
    let response = participant.abort(42);
    assert!(response.success);
    assert_eq!(participant.prepared_count(), 0);
}

#[test]
fn test_participant_conflict_on_locked_key() {
    let participant = TxParticipant::new_in_memory();

    // First prepare
    let request1 = PrepareRequest {
        tx_id: 1,
        coordinator: "coord".to_string(),
        operations: vec![Transaction::Put {
            key: "contested_key".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
        timeout_ms: 5000,
    };

    let vote1 = participant.prepare(request1);
    assert!(matches!(vote1, PrepareVote::Yes { .. }));

    // Second prepare on same key
    let request2 = PrepareRequest {
        tx_id: 2,
        coordinator: "coord".to_string(),
        operations: vec![Transaction::Put {
            key: "contested_key".to_string(),
            data: vec![2],
        }],
        delta_embedding: SparseVector::from_dense(&[0.0, 1.0]),
        timeout_ms: 5000,
    };

    let vote2 = participant.prepare(request2);
    match vote2 {
        PrepareVote::Conflict { conflicting_tx, .. } => {
            assert_eq!(conflicting_tx, 1);
        },
        _ => panic!("Expected Conflict vote"),
    }
}

#[test]
fn test_participant_stale_cleanup() {
    let participant = TxParticipant::new_in_memory();

    let request = PrepareRequest {
        tx_id: 1,
        coordinator: "coord".to_string(),
        operations: vec![Transaction::Put {
            key: "key".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0]),
        timeout_ms: 5000,
    };

    participant.prepare(request);
    assert_eq!(participant.prepared_count(), 1);

    // Cleanup with zero timeout should remove immediately prepared txs
    let stale = participant.cleanup_stale(Duration::from_secs(0));
    assert_eq!(stale.len(), 1);
    assert_eq!(stale[0], 1);
    assert_eq!(participant.prepared_count(), 0);
}

// ============================================================================
// Statistics Tests
// ============================================================================

#[test]
fn test_distributed_tx_stats() {
    let stats = DistributedTxStats::default();

    stats.started.fetch_add(100, Ordering::Relaxed);
    stats.committed.fetch_add(80, Ordering::Relaxed);
    stats.aborted.fetch_add(15, Ordering::Relaxed);
    stats.conflicts.fetch_add(10, Ordering::Relaxed);
    stats.orthogonal_merges.fetch_add(5, Ordering::Relaxed);

    // Commit rate: 80/100 = 0.8
    assert!((stats.commit_rate() - 0.8).abs() < 0.001);

    // Conflict rate: 10/100 = 0.1
    assert!((stats.conflict_rate() - 0.1).abs() < 0.001);
}

// ============================================================================
// Timeout and Cleanup Tests
// ============================================================================

#[test]
fn test_coordinator_timeout_cleanup() {
    // Create transaction with custom short timeout
    let mut tx = DistributedTransaction::new("coord".to_string(), vec![0]);
    tx.timeout_ms = 1; // 1ms timeout

    // Sleep to ensure timeout
    std::thread::sleep(Duration::from_millis(5));

    assert!(tx.is_timed_out());
}

#[test]
fn test_distributed_tx_phase_transitions() {
    let tx = DistributedTransaction::new("coord".to_string(), vec![0, 1]);
    assert_eq!(tx.phase, TxPhase::Preparing);

    // Verify default phase
    assert_eq!(TxPhase::default(), TxPhase::Preparing);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_single_shard_transaction() {
    let coordinator = create_coordinator();

    let tx = coordinator.begin("coord".to_string(), vec![0]).unwrap();
    let tx_id = tx.tx_id;

    let delta = create_delta(vec![1.0, 0.0], vec!["single_key"], tx_id);

    let phase = coordinator.record_vote(
        tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta,
        },
    );
    assert_eq!(phase, Some(TxPhase::Prepared));

    coordinator.commit(tx_id).unwrap();
    assert_eq!(coordinator.stats().committed.load(Ordering::Relaxed), 1);
}

#[test]
fn test_empty_delta_transaction() {
    let tx = DistributedTransaction::new("coord".to_string(), vec![0]);

    // No deltas added
    assert!(tx.merged_delta().is_none());
    assert!(tx.affected_keys().is_empty());
}

#[test]
fn test_commit_not_prepared_fails() {
    let coordinator = create_coordinator();

    let tx = coordinator.begin("coord".to_string(), vec![0]).unwrap();
    let tx_id = tx.tx_id;

    // Try to commit without voting
    let result = coordinator.commit(tx_id);
    assert!(result.is_err());
}

#[test]
fn test_abort_releases_locks() {
    let coordinator = create_coordinator();

    let tx = coordinator.begin("coord".to_string(), vec![0]).unwrap();
    let tx_id = tx.tx_id;

    // Simulate prepare and record the vote properly
    let request = PrepareRequest {
        tx_id,
        coordinator: "coord".to_string(),
        operations: vec![Transaction::Put {
            key: "locked_key".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
        timeout_ms: 5000,
    };

    // Handle prepare returns the vote with lock handle
    let vote = coordinator.handle_prepare(request);
    let (lock_handle, delta) = match vote {
        PrepareVote::Yes { lock_handle, delta } => (lock_handle, delta),
        _ => panic!("Expected Yes vote"),
    };

    // Key should be locked after prepare
    assert!(coordinator.lock_manager().is_locked("locked_key"));

    // Record the vote so abort can release the locks
    coordinator.record_vote(tx_id, 0, PrepareVote::Yes { lock_handle, delta });

    // Abort should release the lock
    coordinator.abort(tx_id, "test abort").unwrap();

    // Key should no longer be locked
    assert!(!coordinator.lock_manager().is_locked("locked_key"));
}
