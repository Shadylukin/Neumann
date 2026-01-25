//! Integration tests for distributed transaction state cleanup error handling.

use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager},
    DistributedTxConfig, DistributedTxCoordinator, TxParticipant,
};
use tensor_store::TensorStore;

#[test]
fn test_coordinator_clear_state_success() {
    let store = TensorStore::new();
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let coordinator = DistributedTxCoordinator::new(consensus, DistributedTxConfig::default());

    // Save state
    coordinator.save_to_store("node1", &store).unwrap();

    // Clear should succeed and return Ok
    let result = DistributedTxCoordinator::clear_persisted_state("node1", &store);
    assert!(result.is_ok());

    // Verify state is gone - load should create fresh
    let consensus2 = ConsensusManager::new(ConsensusConfig::default());
    let restored = DistributedTxCoordinator::load_from_store(
        "node1",
        &store,
        consensus2,
        DistributedTxConfig::default(),
    )
    .unwrap();
    assert_eq!(restored.pending_count(), 0);
}

#[test]
fn test_participant_clear_state_success() {
    let store = TensorStore::new();
    let participant = TxParticipant::new_in_memory();

    // Save state
    participant.save_to_store("node1", 0, &store).unwrap();

    // Clear should succeed and return Ok
    let result = TxParticipant::clear_persisted_state("node1", 0, &store);
    assert!(result.is_ok());

    // Verify state is gone - load should create fresh
    let restored = TxParticipant::load_from_store("node1", 0, &store);
    assert_eq!(restored.prepared_count(), 0);
}

#[test]
fn test_clear_nonexistent_state_succeeds() {
    let store = TensorStore::new();

    // Clearing non-existent state should succeed (idempotent)
    let coord_result = DistributedTxCoordinator::clear_persisted_state("nonexistent", &store);
    assert!(coord_result.is_ok());

    let part_result = TxParticipant::clear_persisted_state("nonexistent", 99, &store);
    assert!(part_result.is_ok());
}
