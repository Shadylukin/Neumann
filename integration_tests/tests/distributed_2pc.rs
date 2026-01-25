//! Integration tests for distributed 2PC transactions.
//!
//! Tests the two-phase commit protocol across multiple participants:
//! - Prepare phase (acquire locks, compute deltas)
//! - Commit/abort phase
//! - Conflict detection
//! - TxHandler message handling

use std::sync::Arc;

use tensor_chain::{
    block::Transaction,
    consensus::{ConsensusConfig, ConsensusManager},
    DistributedTxConfig, DistributedTxCoordinator, Message, MessageHandler, PrepareRequest,
    TxAbortMsg, TxAckMsg, TxCommitMsg, TxHandler, TxParticipant, TxPhase, TxPrepareMsg, TxVote,
};
use tensor_store::SparseVector;

fn create_test_coordinator() -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::with_consensus(consensus)
}

#[test]
fn test_2pc_single_shard_commit() {
    let coordinator = create_test_coordinator();

    // Begin distributed transaction
    let tx = coordinator
        .begin("coordinator".to_string(), vec![0])
        .expect("Failed to begin transaction");

    // Prepare request to shard 0
    let request = PrepareRequest {
        tx_id: tx.tx_id,
        coordinator: "coordinator".to_string(),
        operations: vec![Transaction::Put {
            key: "user:1".to_string(),
            data: vec![1, 2, 3],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
        timeout_ms: 5000,
    };

    // Simulate participant response
    let vote = coordinator.handle_prepare(request);
    assert!(matches!(vote, tensor_chain::PrepareVote::Yes { .. }));

    // Record vote
    let phase = coordinator.record_vote(tx.tx_id, 0, vote);
    assert_eq!(phase, Some(TxPhase::Prepared));

    // Commit
    coordinator.commit(tx.tx_id).unwrap();
    assert_eq!(coordinator.pending_count(), 0);
}

#[test]
fn test_2pc_multi_shard_all_yes() {
    let coordinator = create_test_coordinator();

    let tx = coordinator
        .begin("coordinator".to_string(), vec![0, 1, 2])
        .expect("Failed to begin transaction");

    // All shards vote yes with orthogonal deltas
    for shard in 0..3 {
        let request = PrepareRequest {
            tx_id: tx.tx_id,
            coordinator: "coordinator".to_string(),
            operations: vec![Transaction::Put {
                key: format!("shard{}:key", shard),
                data: vec![shard as u8],
            }],
            delta_embedding: {
                let mut v = vec![0.0; 3];
                v[shard] = 1.0;
                SparseVector::from_dense(&v)
            },
            timeout_ms: 5000,
        };

        let vote = coordinator.handle_prepare(request);
        coordinator.record_vote(tx.tx_id, shard, vote);
    }

    // Should be prepared after all votes
    let tx_state = coordinator.get(tx.tx_id).unwrap();
    assert_eq!(tx_state.phase, TxPhase::Prepared);

    // Commit
    coordinator.commit(tx.tx_id).unwrap();
    assert_eq!(
        coordinator
            .stats
            .committed
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn test_2pc_conflict_detected() {
    let coordinator = create_test_coordinator();

    // First transaction locks key
    let tx1 = coordinator
        .begin("coordinator".to_string(), vec![0])
        .expect("Failed to begin tx1");

    let request1 = PrepareRequest {
        tx_id: tx1.tx_id,
        coordinator: "coordinator".to_string(),
        operations: vec![Transaction::Put {
            key: "shared_key".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
        timeout_ms: 5000,
    };

    let vote1 = coordinator.handle_prepare(request1);
    assert!(matches!(vote1, tensor_chain::PrepareVote::Yes { .. }));

    // Second transaction tries same key - should conflict
    let request2 = PrepareRequest {
        tx_id: tx1.tx_id + 1, // Different tx
        coordinator: "coordinator".to_string(),
        operations: vec![Transaction::Put {
            key: "shared_key".to_string(),
            data: vec![2],
        }],
        delta_embedding: SparseVector::from_dense(&[0.0, 1.0]),
        timeout_ms: 5000,
    };

    let vote2 = coordinator.handle_prepare(request2);
    assert!(matches!(vote2, tensor_chain::PrepareVote::Conflict { .. }));
}

#[test]
fn test_2pc_abort_on_no_vote() {
    let coordinator = create_test_coordinator();

    let tx = coordinator
        .begin("coordinator".to_string(), vec![0, 1])
        .expect("Failed to begin transaction");

    // Shard 0 votes yes
    let request0 = PrepareRequest {
        tx_id: tx.tx_id,
        coordinator: "coordinator".to_string(),
        operations: vec![Transaction::Put {
            key: "key0".to_string(),
            data: vec![0],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
        timeout_ms: 5000,
    };
    let vote0 = coordinator.handle_prepare(request0);
    coordinator.record_vote(tx.tx_id, 0, vote0);

    // Shard 1 votes no
    let phase = coordinator.record_vote(
        tx.tx_id,
        1,
        tensor_chain::PrepareVote::No {
            reason: "resource unavailable".to_string(),
        },
    );

    assert_eq!(phase, Some(TxPhase::Aborting));
}

#[test]
fn test_tx_handler_prepare_commit_flow() {
    let participant = Arc::new(TxParticipant::new_in_memory());
    let handler = TxHandler::new(participant.clone());

    // Prepare message
    let prepare_msg = Message::TxPrepare(TxPrepareMsg {
        tx_id: 1,
        coordinator: "coordinator".to_string(),
        shard_id: 0,
        operations: vec![Transaction::Put {
            key: "test_key".to_string(),
            data: vec![1, 2, 3],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
        timeout_ms: 5000,
    });

    let response = handler.handle(&"coordinator".to_string(), &prepare_msg);
    assert!(response.is_some());

    if let Some(Message::TxPrepareResponse(resp)) = response {
        assert_eq!(resp.tx_id, 1);
        assert!(matches!(resp.vote, TxVote::Yes { .. }));

        // Participant should have prepared transaction
        assert_eq!(participant.prepared_count(), 1);

        // Commit
        let commit_msg = Message::TxCommit(TxCommitMsg {
            tx_id: 1,
            shards: vec![0],
        });

        let ack = handler.handle(&"coordinator".to_string(), &commit_msg);
        assert!(ack.is_some());

        if let Some(Message::TxAck(ack)) = ack {
            assert!(ack.success);
        }

        // Participant should have no prepared transactions after commit
        assert_eq!(participant.prepared_count(), 0);
    }
}

#[test]
fn test_tx_handler_prepare_abort_flow() {
    let participant = Arc::new(TxParticipant::new_in_memory());
    let handler = TxHandler::new(participant.clone());

    // Prepare
    let prepare_msg = Message::TxPrepare(TxPrepareMsg {
        tx_id: 1,
        coordinator: "coordinator".to_string(),
        shard_id: 0,
        operations: vec![Transaction::Put {
            key: "test_key".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0]),
        timeout_ms: 5000,
    });

    handler.handle(&"coordinator".to_string(), &prepare_msg);
    assert_eq!(participant.prepared_count(), 1);

    // Abort
    let abort_msg = Message::TxAbort(TxAbortMsg {
        tx_id: 1,
        reason: "conflict detected".to_string(),
        shards: vec![0],
    });

    let ack = handler.handle(&"coordinator".to_string(), &abort_msg);
    assert!(ack.is_some());

    if let Some(Message::TxAck(TxAckMsg { success, .. })) = ack {
        assert!(success);
    }

    // Participant should have no prepared transactions after abort
    assert_eq!(participant.prepared_count(), 0);
}

#[test]
fn test_tx_participant_lock_conflict() {
    let participant = Arc::new(TxParticipant::new_in_memory());
    let handler = TxHandler::new(participant.clone());

    // First prepare succeeds
    let prepare1 = Message::TxPrepare(TxPrepareMsg {
        tx_id: 1,
        coordinator: "coordinator".to_string(),
        shard_id: 0,
        operations: vec![Transaction::Put {
            key: "shared_key".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0]),
        timeout_ms: 5000,
    });

    let response1 = handler.handle(&"coordinator".to_string(), &prepare1);
    if let Some(Message::TxPrepareResponse(resp)) = response1 {
        assert!(matches!(resp.vote, TxVote::Yes { .. }));
    }

    // Second prepare on same key should conflict
    let prepare2 = Message::TxPrepare(TxPrepareMsg {
        tx_id: 2,
        coordinator: "coordinator".to_string(),
        shard_id: 0,
        operations: vec![Transaction::Put {
            key: "shared_key".to_string(),
            data: vec![2],
        }],
        delta_embedding: SparseVector::from_dense(&[0.0, 1.0]),
        timeout_ms: 5000,
    });

    let response2 = handler.handle(&"coordinator".to_string(), &prepare2);
    if let Some(Message::TxPrepareResponse(resp)) = response2 {
        assert!(matches!(resp.vote, TxVote::Conflict { .. }));
    }
}

#[test]
fn test_2pc_cross_shard_conflict_detection() {
    let config = DistributedTxConfig {
        orthogonal_threshold: 0.5, // More sensitive
        ..Default::default()
    };
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let coordinator = DistributedTxCoordinator::new(consensus, config);

    let tx = coordinator
        .begin("coordinator".to_string(), vec![0, 1])
        .expect("Failed to begin transaction");

    // Shard 0: Similar delta + overlapping key
    let delta0 = tensor_chain::consensus::DeltaVector::new(
        vec![1.0, 0.0, 0.0],
        ["shared_key"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    // Shard 1: Similar delta + same overlapping key
    let delta1 = tensor_chain::consensus::DeltaVector::new(
        vec![0.9, 0.1, 0.0], // Similar to delta0
        ["shared_key"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    // Record votes
    coordinator.record_vote(
        tx.tx_id,
        0,
        tensor_chain::PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );

    let phase = coordinator.record_vote(
        tx.tx_id,
        1,
        tensor_chain::PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );

    // Should detect conflict due to similar deltas on same key
    assert_eq!(phase, Some(TxPhase::Aborting));
}

#[test]
fn test_2pc_orthogonal_merges() {
    let coordinator = create_test_coordinator();

    let tx = coordinator
        .begin("coordinator".to_string(), vec![0, 1])
        .expect("Failed to begin transaction");

    // Shard 0: Orthogonal delta + different key
    let delta0 = tensor_chain::consensus::DeltaVector::new(
        vec![1.0, 0.0, 0.0],
        ["key_a"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    // Shard 1: Orthogonal delta + different key
    let delta1 = tensor_chain::consensus::DeltaVector::new(
        vec![0.0, 1.0, 0.0], // Orthogonal to delta0
        ["key_b"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    // Record votes
    coordinator.record_vote(
        tx.tx_id,
        0,
        tensor_chain::PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );

    let phase = coordinator.record_vote(
        tx.tx_id,
        1,
        tensor_chain::PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );

    // Should succeed - orthogonal deltas can merge
    assert_eq!(phase, Some(TxPhase::Prepared));

    // Verify orthogonal merge was counted
    coordinator.commit(tx.tx_id).unwrap();
    assert!(
        coordinator
            .stats
            .orthogonal_merges
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0
    );
}

#[test]
fn test_2pc_stats_tracking() {
    let coordinator = create_test_coordinator();

    // Create and commit a transaction
    let tx = coordinator
        .begin("coordinator".to_string(), vec![0])
        .unwrap();
    let request = PrepareRequest {
        tx_id: tx.tx_id,
        coordinator: "coordinator".to_string(),
        operations: vec![Transaction::Put {
            key: "key".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0]),
        timeout_ms: 5000,
    };
    let vote = coordinator.handle_prepare(request);
    coordinator.record_vote(tx.tx_id, 0, vote);
    coordinator.commit(tx.tx_id).unwrap();

    // Create and abort a transaction
    let tx2 = coordinator
        .begin("coordinator".to_string(), vec![0])
        .unwrap();
    coordinator.abort(tx2.tx_id, "test abort").unwrap();

    let stats = coordinator.stats();
    assert!(stats.started.load(std::sync::atomic::Ordering::Relaxed) >= 2);
    assert!(stats.committed.load(std::sync::atomic::Ordering::Relaxed) >= 1);
    assert!(stats.aborted.load(std::sync::atomic::Ordering::Relaxed) >= 1);
}
