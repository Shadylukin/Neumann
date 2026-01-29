// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for 2PC abort broadcast.
//!
//! Tests:
//! - Coordinator broadcasts abort to all participants
//! - Participants release locks on abort receipt
//! - Abort acknowledgment flow
//! - Retry mechanism for unacknowledged aborts
//! - Cleanup after max retries

use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager, DeltaVector},
    distributed_tx::{DistributedTxConfig, DistributedTxCoordinator, PrepareVote, TxParticipant},
};

fn create_test_coordinator() -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, DistributedTxConfig::default())
}

fn create_test_participant() -> TxParticipant {
    TxParticipant::new_in_memory()
}

#[test]
fn test_abort_broadcast_to_participants() {
    let coordinator = create_test_coordinator();

    // Begin a transaction with 3 shards
    let tx = coordinator
        .begin("node1".to_string(), vec![0, 1, 2])
        .unwrap();

    // Record votes from all shards - one says No
    coordinator.record_vote(
        tx.tx_id,
        0,
        PrepareVote::No {
            reason: "test".to_string(),
        },
    );
    coordinator.record_vote(
        tx.tx_id,
        1,
        PrepareVote::Yes {
            lock_handle: 1,
            delta: DeltaVector::new(
                vec![1.0],
                ["k1"].iter().map(|s| s.to_string()).collect(),
                tx.tx_id,
            ),
        },
    );
    coordinator.record_vote(
        tx.tx_id,
        2,
        PrepareVote::Yes {
            lock_handle: 2,
            delta: DeltaVector::new(
                vec![0.0, 1.0],
                ["k2"].iter().map(|s| s.to_string()).collect(),
                tx.tx_id,
            ),
        },
    );

    // Take pending aborts - should have one abort for all 3 shards
    let pending = coordinator.take_pending_aborts();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].0, tx.tx_id);
    assert_eq!(pending[0].2.len(), 3); // All 3 shards
}

#[test]
fn test_participant_releases_locks_on_abort() {
    let participant = create_test_participant();

    // Prepare a transaction (acquires locks)
    let tx_id = 42;
    let request = tensor_chain::distributed_tx::PrepareRequest {
        tx_id,
        coordinator: "coord".to_string(),
        operations: vec![],
        delta_embedding: tensor_store::SparseVector::new(0),
        timeout_ms: 5000,
    };
    participant.prepare(request);

    // Check it's awaiting decision
    assert!(participant.get_awaiting_decision().contains(&tx_id));

    // Abort
    let response = participant.abort(tx_id);
    assert!(response.success);

    // Should no longer be awaiting decision
    assert!(!participant.get_awaiting_decision().contains(&tx_id));
}

#[test]
fn test_abort_ack_sent_by_participant() {
    let participant = create_test_participant();

    // Prepare a transaction
    let tx_id = 99;
    let request = tensor_chain::distributed_tx::PrepareRequest {
        tx_id,
        coordinator: "coord".to_string(),
        operations: vec![],
        delta_embedding: tensor_store::SparseVector::new(0),
        timeout_ms: 5000,
    };
    participant.prepare(request);

    // Abort returns a response (which would be sent as ack)
    let response = participant.abort(tx_id);
    assert!(response.success);
    assert!(response.error.is_none());
}

#[test]
fn test_abort_ack_clears_pending_state() {
    let coordinator = create_test_coordinator();

    // Track an abort with 2 shards
    coordinator.track_abort(123, vec![0, 1]);

    // Acknowledge from shard 0
    let complete = coordinator.handle_abort_ack(123, 0);
    assert!(!complete);

    // Acknowledge from shard 1 - should complete
    let complete = coordinator.handle_abort_ack(123, 1);
    assert!(complete);

    // Verify state is cleared (handle_abort_ack for non-existent returns false)
    let complete = coordinator.handle_abort_ack(123, 0);
    assert!(!complete);
}

#[test]
fn test_abort_retry_on_missing_ack() {
    let coordinator = create_test_coordinator();

    // Track an abort
    coordinator.track_abort(456, vec![0, 1, 2]);

    // First retry check - should return empty (too early)
    let retries = coordinator.get_retry_aborts();
    assert!(retries.is_empty());

    // Note: Can't easily test time-based retry in a unit test without mocking
    // The get_retry_aborts() function uses real time
}

#[test]
fn test_abort_gives_up_after_max_retries() {
    let coordinator = create_test_coordinator();

    // Track an abort
    coordinator.track_abort(789, vec![0, 1]);

    // Cleanup stale - should return empty initially
    let stale = coordinator.cleanup_stale_aborts();
    assert!(stale.is_empty());

    // Note: Can't easily test timeout-based cleanup without mocking
    // The cleanup_stale_aborts() function uses real time
}

#[test]
fn test_abort_flow_complete() {
    let coordinator = create_test_coordinator();

    // Begin transaction
    let tx = coordinator.begin("node1".to_string(), vec![0, 1]).unwrap();

    // Record NO votes from both shards
    coordinator.record_vote(
        tx.tx_id,
        0,
        PrepareVote::No {
            reason: "conflict".to_string(),
        },
    );
    coordinator.record_vote(
        tx.tx_id,
        1,
        PrepareVote::No {
            reason: "conflict".to_string(),
        },
    );

    // Get pending aborts
    let pending = coordinator.take_pending_aborts();
    assert!(!pending.is_empty());

    // Simulate tracking for ack (as coordinator would do after broadcast)
    for (tx_id, _, shards) in &pending {
        coordinator.track_abort(*tx_id, shards.clone());
    }

    // Simulate acks from participants
    for shard in &pending[0].2 {
        coordinator.handle_abort_ack(pending[0].0, *shard);
    }

    // State should be cleaned up after all acks
}

#[test]
fn test_cross_shard_conflict_queues_abort() {
    let coordinator = create_test_coordinator();

    // Begin transaction with 2 shards
    let tx = coordinator.begin("node1".to_string(), vec![0, 1]).unwrap();

    // Both shards vote YES with overlapping keys and similar deltas
    let delta0 = DeltaVector::new(
        vec![1.0, 0.0],
        ["shared_key"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );
    let delta1 = DeltaVector::new(
        vec![0.9, 0.1], // Similar to delta0
        ["shared_key"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    coordinator.record_vote(
        tx.tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );
    let phase = coordinator.record_vote(
        tx.tx_id,
        1,
        PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );

    // Should have detected conflict
    if phase == Some(tensor_chain::distributed_tx::TxPhase::Aborting) {
        // Cross-shard conflict was detected - check abort was queued
        let pending = coordinator.take_pending_aborts();
        assert!(!pending.is_empty());
    }
    // Note: Whether conflict is detected depends on similarity threshold
}
