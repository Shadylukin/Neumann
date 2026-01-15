//! Integration tests for distributed transaction crash recovery.
//!
//! Tests the persistence and recovery of 2PC coordinator and participant state:
//! - Coordinator crash after collecting YES votes
//! - Coordinator crash during commit phase
//! - Participant crash after prepare
//! - Multi-node crash recovery

use std::collections::HashSet;
use std::time::Duration;

use tensor_chain::{
    block::Transaction,
    consensus::{ConsensusConfig, ConsensusManager, DeltaVector},
    CoordinatorState, DistributedTxConfig, DistributedTxCoordinator, ParticipantState,
    PrepareRequest, PrepareVote, SerializableLockState, TxParticipant, TxPhase,
};
use tensor_store::{SparseVector, TensorStore};

fn create_coordinator_with_config(config: DistributedTxConfig) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, config)
}

fn create_coordinator() -> DistributedTxCoordinator {
    create_coordinator_with_config(DistributedTxConfig::default())
}

// ============= Coordinator Crash Recovery Tests =============

#[test]
fn test_coordinator_crash_after_all_yes_votes_recovery() {
    let store = TensorStore::new();
    let node_id = "coord1";

    // Phase 1: Set up coordinator with pending transaction that has all YES votes
    let tx_id = {
        let coordinator = create_coordinator();
        let tx = coordinator
            .begin("coord1".to_string(), vec![0, 1])
            .expect("Failed to begin transaction");

        // Handle prepare and record YES votes from both shards
        for shard in 0..2 {
            let request = PrepareRequest {
                tx_id: tx.tx_id,
                coordinator: "coord1".to_string(),
                operations: vec![Transaction::Put {
                    key: format!("shard{}:key", shard),
                    data: vec![shard as u8],
                }],
                delta_embedding: {
                    let mut v = vec![0.0; 2];
                    v[shard] = 1.0;
                    SparseVector::from_dense(&v)
                },
                timeout_ms: 5000,
            };
            let vote = coordinator.handle_prepare(request);
            coordinator.record_vote(tx.tx_id, shard, vote);
        }

        // Verify we're in Prepared state
        let pending_tx = coordinator.get(tx.tx_id).unwrap();
        assert_eq!(pending_tx.phase, TxPhase::Prepared);

        // Save state before "crash"
        coordinator.save_to_store(node_id, &store).unwrap();

        tx.tx_id
    };
    // coordinator is dropped here (simulating crash)

    // Phase 2: Load coordinator from store and recover
    let consensus2 = ConsensusManager::new(ConsensusConfig::default());
    let restored = DistributedTxCoordinator::load_from_store(
        node_id,
        &store,
        consensus2,
        DistributedTxConfig::default(),
    )
    .unwrap();

    // Verify pending transaction was restored
    assert_eq!(restored.pending_count(), 1);
    assert!(restored.get(tx_id).is_some());

    // Run recovery
    let stats = restored.recover();
    assert_eq!(stats.pending_commit, 1);

    // Verify transaction moved to Committing
    let decisions = restored.get_pending_decisions();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].1, TxPhase::Committing);

    // Complete the commit
    restored.complete_commit(tx_id).unwrap();
    assert_eq!(restored.pending_count(), 0);
}

#[test]
fn test_coordinator_crash_during_commit_recovery() {
    use tensor_store::{ScalarValue, TensorData, TensorValue};

    let store = TensorStore::new();
    let node_id = "coord1";

    // Phase 1: Create a coordinator state with transaction in Committing phase
    let tx_id = {
        let coordinator = create_coordinator();
        let tx = coordinator
            .begin("coord1".to_string(), vec![0])
            .expect("Failed to begin transaction");

        // Handle prepare
        let request = PrepareRequest {
            tx_id: tx.tx_id,
            coordinator: "coord1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        };
        let vote = coordinator.handle_prepare(request);
        coordinator.record_vote(tx.tx_id, 0, vote);

        // Get current state and modify it
        let mut state = coordinator.to_state();
        if let Some(pending_tx) = state.pending.get_mut(&tx.tx_id) {
            pending_tx.phase = TxPhase::Committing;
        }

        // Save the modified state directly
        let bytes = bincode::serialize(&state).unwrap();
        let mut data = TensorData::new();
        data.set("state", TensorValue::Scalar(ScalarValue::Bytes(bytes)));
        store
            .put(&format!("_dtx:coordinator:{}:state", node_id), data)
            .unwrap();

        tx.tx_id
    };

    // Phase 2: Recover
    let consensus2 = ConsensusManager::new(ConsensusConfig::default());
    let restored = DistributedTxCoordinator::load_from_store(
        node_id,
        &store,
        consensus2,
        DistributedTxConfig::default(),
    )
    .unwrap();

    let stats = restored.recover();
    assert_eq!(stats.pending_commit, 1);

    // Verify we can complete the commit
    restored.complete_commit(tx_id).unwrap();
    assert_eq!(restored.pending_count(), 0);
}

#[test]
fn test_coordinator_crash_with_timeout_recovery() {
    use tensor_store::{ScalarValue, TensorData, TensorValue};

    let store = TensorStore::new();
    let node_id = "coord1";

    // Phase 1: Create transaction with very short timeout
    let _tx_id = {
        let coordinator = create_coordinator();
        let tx = coordinator
            .begin("coord1".to_string(), vec![0])
            .expect("Failed to begin transaction");

        // Get current state and modify it to appear timed out
        let mut state = coordinator.to_state();
        if let Some(pending_tx) = state.pending.get_mut(&tx.tx_id) {
            pending_tx.timeout_ms = 1;
            pending_tx.started_at = pending_tx.started_at.saturating_sub(10_000);
        }

        // Save the modified state directly
        let bytes = bincode::serialize(&state).unwrap();
        let mut data = TensorData::new();
        data.set("state", TensorValue::Scalar(ScalarValue::Bytes(bytes)));
        store
            .put(&format!("_dtx:coordinator:{}:state", node_id), data)
            .unwrap();

        tx.tx_id
    };

    // Phase 2: Recover
    let consensus2 = ConsensusManager::new(ConsensusConfig::default());
    let restored = DistributedTxCoordinator::load_from_store(
        node_id,
        &store,
        consensus2,
        DistributedTxConfig::default(),
    )
    .unwrap();

    let stats = restored.recover();
    assert_eq!(stats.timed_out, 1);

    // Verify transaction marked for abort
    let decisions = restored.get_pending_decisions();
    assert_eq!(decisions.len(), 1);
    assert_eq!(decisions[0].1, TxPhase::Aborting);
}

// ============= Participant Crash Recovery Tests =============

#[test]
fn test_participant_crash_after_prepare_recovery() {
    let store = TensorStore::new();
    let node_id = "part1";
    let shard_id = 0;

    // Phase 1: Prepare transaction on participant
    let tx_id = {
        let participant = TxParticipant::new();

        let request = PrepareRequest {
            tx_id: 42,
            coordinator: "coord1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1, 2, 3],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.5]),
            timeout_ms: 5000,
        };

        let vote = participant.prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));

        // Verify locks are held
        assert!(participant.locks.is_locked("key1"));

        // Save state
        participant
            .save_to_store(node_id, shard_id, &store)
            .unwrap();

        42
    };

    // Phase 2: Load participant and verify state recovered
    let restored = TxParticipant::load_from_store(node_id, shard_id, &store);

    assert_eq!(restored.prepared_count(), 1);
    assert!(restored.locks.is_locked("key1"));

    // Verify we can still commit
    let response = restored.commit(tx_id);
    assert!(response.success);
    assert!(!restored.locks.is_locked("key1"));
}

#[test]
fn test_participant_crash_expired_presumed_abort() {
    let store = TensorStore::new();
    let node_id = "part1";
    let shard_id = 0;

    // Phase 1: Create a prepared transaction that will be old
    {
        let participant = TxParticipant::new();

        let request = PrepareRequest {
            tx_id: 100,
            coordinator: "coord1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        participant.prepare(request);

        // Manually make it look old by modifying the prepared_at timestamp
        {
            let mut prepared = participant.prepared.write();
            if let Some(tx) = prepared.get_mut(&100) {
                tx.prepared_at_ms = tx.prepared_at_ms.saturating_sub(60_000); // 1 minute ago
            }
        }

        participant
            .save_to_store(node_id, shard_id, &store)
            .unwrap();
    }

    // Phase 2: Load and recover with short timeout
    let restored = TxParticipant::load_from_store(node_id, shard_id, &store);

    // Verify the old prepared transaction exists
    assert_eq!(restored.prepared_count(), 1);

    // Recovery with 30 second timeout should release the expired transaction
    let awaiting = restored.recover(Duration::from_secs(30));

    // Transaction was expired, so nothing awaiting
    assert!(awaiting.is_empty());
    assert_eq!(restored.prepared_count(), 0);
}

// ============= Multi-Node Recovery Tests =============

#[test]
fn test_coordinator_and_participant_crash_recovery() {
    let store = TensorStore::new();

    // Set up coordinator and participants
    let tx_id = {
        let coordinator = create_coordinator();
        let participant0 = TxParticipant::new();
        let participant1 = TxParticipant::new();

        let tx = coordinator
            .begin("coord1".to_string(), vec![0, 1])
            .expect("Failed to begin transaction");

        // Prepare on both participants
        for (shard, participant) in [(0, &participant0), (1, &participant1)] {
            let request = PrepareRequest {
                tx_id: tx.tx_id,
                coordinator: "coord1".to_string(),
                operations: vec![Transaction::Put {
                    key: format!("shard{}:key", shard),
                    data: vec![shard as u8],
                }],
                delta_embedding: {
                    let mut v = vec![0.0; 2];
                    v[shard as usize] = 1.0;
                    SparseVector::from_dense(&v)
                },
                timeout_ms: 5000,
            };
            let vote = participant.prepare(request);
            coordinator.record_vote(tx.tx_id, shard, vote);
        }

        // Save all state
        coordinator.save_to_store("coord1", &store).unwrap();
        participant0.save_to_store("coord1", 0, &store).unwrap();
        participant1.save_to_store("coord1", 1, &store).unwrap();

        tx.tx_id
    };

    // Restore all components
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let coord_restored = DistributedTxCoordinator::load_from_store(
        "coord1",
        &store,
        consensus,
        DistributedTxConfig::default(),
    )
    .unwrap();
    let part0_restored = TxParticipant::load_from_store("coord1", 0, &store);
    let part1_restored = TxParticipant::load_from_store("coord1", 1, &store);

    // Verify all state recovered
    assert_eq!(coord_restored.pending_count(), 1);
    assert_eq!(part0_restored.prepared_count(), 1);
    assert_eq!(part1_restored.prepared_count(), 1);

    // Recover coordinator
    let stats = coord_restored.recover();
    assert_eq!(stats.pending_commit, 1);

    // Complete commit
    coord_restored.complete_commit(tx_id).unwrap();
    part0_restored.commit(tx_id);
    part1_restored.commit(tx_id);

    assert_eq!(coord_restored.pending_count(), 0);
    assert_eq!(part0_restored.prepared_count(), 0);
    assert_eq!(part1_restored.prepared_count(), 0);
}

#[test]
fn test_multiple_transactions_crash_recovery() {
    let store = TensorStore::new();
    let node_id = "coord1";

    // Create multiple transactions in different states
    let _tx_ids: Vec<u64> = {
        let coordinator = create_coordinator();

        let mut ids = Vec::new();

        // Transaction 1: Preparing
        let tx1 = coordinator
            .begin("coord1".to_string(), vec![0])
            .expect("Failed to begin transaction");
        ids.push(tx1.tx_id);

        // Transaction 2: Prepared with all YES votes
        let tx2 = coordinator
            .begin("coord1".to_string(), vec![0])
            .expect("Failed to begin transaction");
        let request2 = PrepareRequest {
            tx_id: tx2.tx_id,
            coordinator: "coord1".to_string(),
            operations: vec![Transaction::Put {
                key: "key2".to_string(),
                data: vec![2],
            }],
            delta_embedding: SparseVector::from_dense(&[0.0, 1.0]),
            timeout_ms: 5000,
        };
        let vote2 = coordinator.handle_prepare(request2);
        coordinator.record_vote(tx2.tx_id, 0, vote2);
        ids.push(tx2.tx_id);

        coordinator.save_to_store(node_id, &store).unwrap();
        ids
    };

    // Recover
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let restored = DistributedTxCoordinator::load_from_store(
        node_id,
        &store,
        consensus,
        DistributedTxConfig::default(),
    )
    .unwrap();

    assert_eq!(restored.pending_count(), 2);

    let stats = restored.recover();

    // tx1 is still preparing, tx2 ready to commit
    assert_eq!(stats.pending_prepare, 1);
    assert_eq!(stats.pending_commit, 1);
}

#[test]
fn test_persistence_survives_multiple_restarts() {
    let store = TensorStore::new();
    let node_id = "coord1";
    let shard_id = 0;

    // First session - create and prepare
    let tx_id = {
        let coordinator = create_coordinator();
        let participant = TxParticipant::new();

        let tx = coordinator
            .begin("coord1".to_string(), vec![0])
            .expect("Failed to begin transaction");

        let request = PrepareRequest {
            tx_id: tx.tx_id,
            coordinator: "coord1".to_string(),
            operations: vec![Transaction::Put {
                key: "persistent_key".to_string(),
                data: vec![42],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        };
        let vote = participant.prepare(request.clone());
        coordinator.record_vote(tx.tx_id, 0, vote);

        coordinator.save_to_store(node_id, &store).unwrap();
        participant
            .save_to_store(node_id, shard_id, &store)
            .unwrap();

        tx.tx_id
    };

    // Second session - just load and save again
    {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::load_from_store(
            node_id,
            &store,
            consensus,
            DistributedTxConfig::default(),
        )
        .unwrap();
        let participant = TxParticipant::load_from_store(node_id, shard_id, &store);

        assert_eq!(coordinator.pending_count(), 1);
        assert_eq!(participant.prepared_count(), 1);

        coordinator.save_to_store(node_id, &store).unwrap();
        participant
            .save_to_store(node_id, shard_id, &store)
            .unwrap();
    }

    // Third session - verify state survived multiple restarts
    {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::load_from_store(
            node_id,
            &store,
            consensus,
            DistributedTxConfig::default(),
        )
        .unwrap();
        let participant = TxParticipant::load_from_store(node_id, shard_id, &store);

        assert_eq!(coordinator.pending_count(), 1);
        assert!(coordinator.get(tx_id).is_some());
        assert_eq!(participant.prepared_count(), 1);
        assert!(participant.locks.is_locked("persistent_key"));

        // Finally commit
        coordinator.recover();
        coordinator.complete_commit(tx_id).unwrap();
        participant.commit(tx_id);

        assert_eq!(coordinator.pending_count(), 0);
        assert_eq!(participant.prepared_count(), 0);
    }
}

// ============= State Serialization Tests =============

#[test]
fn test_coordinator_state_bincode_roundtrip() {
    use std::collections::HashMap;

    let mut pending = HashMap::new();
    let tx = tensor_chain::DistributedTransaction::new("node1".to_string(), vec![0, 1, 2]);
    pending.insert(tx.tx_id, tx);

    let state = CoordinatorState {
        pending,
        lock_state: SerializableLockState {
            locks: HashMap::new(),
            tx_locks: HashMap::new(),
            default_timeout_ms: 30000,
        },
    };

    let bytes = bincode::serialize(&state).unwrap();
    let restored: CoordinatorState = bincode::deserialize(&bytes).unwrap();

    assert_eq!(restored.pending.len(), 1);
    assert_eq!(restored.lock_state.default_timeout_ms, 30000);
}

#[test]
fn test_participant_state_bincode_roundtrip() {
    use std::collections::HashMap;

    let mut prepared = HashMap::new();
    prepared.insert(
        1u64,
        tensor_chain::PreparedTx {
            tx_id: 1,
            lock_handle: 100,
            operations: vec![Transaction::Put {
                key: "key".to_string(),
                data: vec![1, 2, 3],
            }],
            delta: DeltaVector::new(vec![1.0], HashSet::new(), 1),
            prepared_at_ms: 1000,
        },
    );

    let state = ParticipantState {
        prepared,
        lock_state: SerializableLockState {
            locks: HashMap::new(),
            tx_locks: HashMap::new(),
            default_timeout_ms: 30000,
        },
    };

    let bytes = bincode::serialize(&state).unwrap();
    let restored: ParticipantState = bincode::deserialize(&bytes).unwrap();

    assert_eq!(restored.prepared.len(), 1);
    assert!(restored.prepared.contains_key(&1));
}
