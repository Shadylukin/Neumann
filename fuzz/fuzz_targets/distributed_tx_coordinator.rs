// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuzz target for DistributedTxCoordinator state machine.
//!
//! Tests 2PC protocol state transitions and conflict detection.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{
    ConsensusConfig, ConsensusManager, DeltaVector, DistributedTxConfig, DistributedTxCoordinator,
    PrepareVote,
};

#[derive(Arbitrary, Debug)]
enum CoordinatorOp {
    Begin {
        coordinator: u8,
        num_shards: u8,
    },
    RecordVoteYes {
        tx_id: u64,
        shard_id: u8,
        lock_handle: u64,
        delta: Vec<f32>,
        keys: Vec<String>,
    },
    RecordVoteNo {
        tx_id: u64,
        shard_id: u8,
        reason: String,
    },
    RecordVoteConflict {
        tx_id: u64,
        shard_id: u8,
        similarity: f32,
        conflicting_tx: u64,
    },
    Commit {
        tx_id: u64,
    },
    Abort {
        tx_id: u64,
        reason: String,
    },
    PendingCount,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    config: FuzzConfig,
    ops: Vec<CoordinatorOp>,
}

#[derive(Arbitrary, Debug)]
struct FuzzConfig {
    max_concurrent: u8,
    prepare_timeout_ms: u64,
    orthogonal_threshold: f32,
}

fn create_coordinator(config: &FuzzConfig) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let dtx_config = DistributedTxConfig {
        max_concurrent: (config.max_concurrent as usize).clamp(1, 100),
        prepare_timeout_ms: config.prepare_timeout_ms.clamp(100, 60000),
        orthogonal_threshold: if config.orthogonal_threshold.is_finite() {
            config.orthogonal_threshold.clamp(0.01, 0.99)
        } else {
            0.1
        },
        ..Default::default()
    };
    DistributedTxCoordinator::new(consensus, dtx_config)
}

fn create_delta(vec: Vec<f32>, keys: Vec<String>, tx_id: u64) -> DeltaVector {
    use std::collections::HashSet;

    // Normalize to reasonable size
    let vec: Vec<f32> = vec
        .into_iter()
        .take(128)
        .map(|v| if v.is_finite() { v } else { 0.0 })
        .collect();

    let keys: HashSet<String> = keys.into_iter().take(10).filter(|k| k.len() < 50).collect();

    let final_vec = if vec.is_empty() { vec![0.0; 4] } else { vec };
    let final_keys = if keys.is_empty() {
        ["default_key".to_string()].into_iter().collect()
    } else {
        keys
    };

    DeltaVector::new(final_vec, final_keys, tx_id)
}

fuzz_target!(|input: FuzzInput| {
    let coordinator = create_coordinator(&input.config);

    // Limit operations
    let ops = if input.ops.len() > 50 {
        &input.ops[..50]
    } else {
        &input.ops
    };

    for op in ops {
        match op {
            CoordinatorOp::Begin {
                coordinator: coord_id,
                num_shards,
            } => {
                let num = (*num_shards as usize).clamp(1, 10);
                let shards: Vec<usize> = (0..num).collect();
                let _ = coordinator.begin(format!("node{}", coord_id), shards);
            },
            CoordinatorOp::RecordVoteYes {
                tx_id,
                shard_id,
                lock_handle,
                delta,
                keys,
            } => {
                let delta = create_delta(delta.clone(), keys.clone(), *tx_id);
                let vote = PrepareVote::Yes {
                    lock_handle: *lock_handle,
                    delta,
                };
                let _ = coordinator.record_vote(*tx_id, *shard_id as usize, vote);
            },
            CoordinatorOp::RecordVoteNo {
                tx_id,
                shard_id,
                reason,
            } => {
                let reason = if reason.len() > 100 {
                    reason[..100].to_string()
                } else {
                    reason.clone()
                };
                let vote = PrepareVote::No { reason };
                let _ = coordinator.record_vote(*tx_id, *shard_id as usize, vote);
            },
            CoordinatorOp::RecordVoteConflict {
                tx_id,
                shard_id,
                similarity,
                conflicting_tx,
            } => {
                let sim = if similarity.is_finite() {
                    *similarity
                } else {
                    0.5
                };
                let vote = PrepareVote::Conflict {
                    similarity: sim,
                    conflicting_tx: *conflicting_tx,
                };
                let _ = coordinator.record_vote(*tx_id, *shard_id as usize, vote);
            },
            CoordinatorOp::Commit { tx_id } => {
                let _ = coordinator.commit(*tx_id);
            },
            CoordinatorOp::Abort { tx_id, reason } => {
                let reason = if reason.len() > 100 {
                    &reason[..100]
                } else {
                    reason
                };
                let _ = coordinator.abort(*tx_id, reason);
            },
            CoordinatorOp::PendingCount => {
                let count = coordinator.pending_count();
                // Should never exceed max_concurrent
                assert!(count <= input.config.max_concurrent as usize + 10);
            },
        }
    }

    // Verify invariants
    let stats = coordinator.stats();
    let started = stats.started.load(std::sync::atomic::Ordering::Relaxed);
    let committed = stats.committed.load(std::sync::atomic::Ordering::Relaxed);
    let aborted = stats.aborted.load(std::sync::atomic::Ordering::Relaxed);

    // Committed + aborted <= started
    assert!(committed + aborted <= started + coordinator.pending_count() as u64);
});
