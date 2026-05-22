// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuzz target for distributed transaction state cleanup operations.
//!
//! Tests that clear_persisted_state handles various node IDs and shard IDs
//! without panicking and returns proper Results.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager},
    DistributedTxConfig, DistributedTxCoordinator, TxParticipant,
};
use tensor_store::TensorStore;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    node_id: String,
    shard_id: usize,
    save_first: bool,
}

fuzz_target!(|input: FuzzInput| {
    // Skip empty node IDs (invalid)
    if input.node_id.is_empty() || input.node_id.len() > 256 {
        return;
    }

    let store = TensorStore::new();

    // Test coordinator clear
    if input.save_first {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, DistributedTxConfig::default());
        let _ = coordinator.save_to_store(&input.node_id, &store);
    }

    // Should not panic, should return Result
    let _ = DistributedTxCoordinator::clear_persisted_state(&input.node_id, &store);

    // Test participant clear
    if input.save_first {
        let participant = TxParticipant::new_in_memory();
        let _ = participant.save_to_store(&input.node_id, input.shard_id, &store);
    }

    // Should not panic, should return Result
    let _ = TxParticipant::clear_persisted_state(&input.node_id, input.shard_id, &store);
});
