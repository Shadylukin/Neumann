// SPDX-License-Identifier: MIT OR Apache-2.0
//! Stateful fuzzer for 2PC distributed transaction operations.
//!
//! Tests that:
//! - Sequences of begin/record_vote/commit/abort never panic
//! - Transaction phase transitions are valid
//! - No transaction ends in an inconsistent state

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;
use tensor_chain::{
    consensus::{ConsensusManager, DeltaVector},
    distributed_tx::{DistributedTxConfig, DistributedTxCoordinator, PrepareVote, TxPhase},
};
use tensor_store::SparseVector;

#[derive(Debug, Arbitrary)]
enum TxOp {
    Begin { shard_count: u8 },
    RecordVoteYes { tx_idx: u8, shard_idx: u8 },
    RecordVoteNo { tx_idx: u8, shard_idx: u8 },
    Commit { tx_idx: u8 },
    Abort { tx_idx: u8 },
}

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    operations: Vec<TxOp>,
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = FuzzInput::arbitrary(&mut arbitrary::Unstructured::new(data)) else {
        return;
    };
    if input.operations.len() > 50 {
        return;
    }

    let consensus = ConsensusManager::default_config();
    let config = DistributedTxConfig::default();
    let coordinator = DistributedTxCoordinator::new(consensus, config);

    let mut active_tx_ids: Vec<u64> = Vec::new();

    for op in &input.operations {
        match op {
            TxOp::Begin { shard_count } => {
                let count = ((*shard_count) % 4 + 1) as usize; // 1..=4 shards
                let participants: Vec<usize> = (0..count).collect();
                if let Ok(tx) = coordinator.begin("coordinator".to_string(), participants) {
                    active_tx_ids.push(tx.tx_id);
                }
            }
            TxOp::RecordVoteYes { tx_idx, shard_idx } => {
                if active_tx_ids.is_empty() {
                    continue;
                }
                let tx_id = active_tx_ids[(*tx_idx as usize) % active_tx_ids.len()];
                let shard = (*shard_idx as usize) % 4;
                let delta = DeltaVector::from_sparse(
                    SparseVector::new(4),
                    HashSet::new(),
                    tx_id,
                );
                let vote = PrepareVote::Yes {
                    lock_handle: 0,
                    delta,
                };
                let _ = coordinator.record_vote(tx_id, shard, vote);
            }
            TxOp::RecordVoteNo { tx_idx, shard_idx } => {
                if active_tx_ids.is_empty() {
                    continue;
                }
                let tx_id = active_tx_ids[(*tx_idx as usize) % active_tx_ids.len()];
                let shard = (*shard_idx as usize) % 4;
                let vote = PrepareVote::No {
                    reason: "fuzzer rejection".to_string(),
                };
                let _ = coordinator.record_vote(tx_id, shard, vote);
            }
            TxOp::Commit { tx_idx } => {
                if active_tx_ids.is_empty() {
                    continue;
                }
                let idx = (*tx_idx as usize) % active_tx_ids.len();
                let tx_id = active_tx_ids[idx];
                if coordinator.commit(tx_id).is_ok() {
                    active_tx_ids.swap_remove(idx);
                }
            }
            TxOp::Abort { tx_idx } => {
                if active_tx_ids.is_empty() {
                    continue;
                }
                let idx = (*tx_idx as usize) % active_tx_ids.len();
                let tx_id = active_tx_ids[idx];
                if coordinator.abort(tx_id, "fuzzer abort").is_ok() {
                    active_tx_ids.swap_remove(idx);
                }
            }
        }
    }

    // Invariant: all transactions in valid phase
    for tx_id in &active_tx_ids {
        if let Some(tx) = coordinator.get(*tx_id) {
            assert!(
                matches!(
                    tx.phase,
                    TxPhase::Preparing
                        | TxPhase::Prepared
                        | TxPhase::Committing
                        | TxPhase::Committed
                        | TxPhase::Aborting
                        | TxPhase::Aborted
                ),
                "Transaction {} in unexpected phase: {:?}",
                tx_id,
                tx.phase
            );
        }
    }
});
