// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

//! Fuzz test for 2PC flow with deadlock detection.
//!
//! This fuzz target tests complete 2PC flows including transaction lifecycle,
//! deadlock detection, and victim selection.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;
use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager, DeltaVector},
    DistributedTxConfig, DistributedTxCoordinator, DeadlockDetector, DeadlockDetectorConfig,
    VictimSelectionPolicy, PrepareVote, PrepareRequest, Transaction,
};
use tensor_store::SparseVector;

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
enum FuzzOp {
    BeginTransaction {
        node_idx: u8,
        participant_count: u8,
    },
    PrepareOnShard {
        tx_id_mod: u8,
        key_count: u8,
        key_base: u8,
    },
    RecordVote {
        tx_id_mod: u8,
        shard_mod: u8,
        is_yes: bool,
    },
    Commit {
        tx_id_mod: u8,
    },
    Abort {
        tx_id_mod: u8,
    },
    RunDeadlockDetection,
    CleanupTimeouts,
    AddDirectWaitEdge {
        waiter_mod: u8,
        holder_mod: u8,
        priority: Option<u8>,
    },
    RemoveWaitEdges {
        tx_id_mod: u8,
    },
    CheckForCycles,
}

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
struct FuzzInput {
    policy: FuzzVictimPolicy,
    operations: Vec<FuzzOp>,
}

#[derive(Debug, Arbitrary, Clone, Copy)]
enum FuzzVictimPolicy {
    Youngest,
    Oldest,
    LowestPriority,
}

impl From<FuzzVictimPolicy> for VictimSelectionPolicy {
    fn from(p: FuzzVictimPolicy) -> Self {
        match p {
            FuzzVictimPolicy::Youngest => VictimSelectionPolicy::Youngest,
            FuzzVictimPolicy::Oldest => VictimSelectionPolicy::Oldest,
            FuzzVictimPolicy::LowestPriority => VictimSelectionPolicy::LowestPriority,
        }
    }
}

fn create_test_coordinator() -> DistributedTxCoordinator {
    let config = DistributedTxConfig {
        max_concurrent: 50,
        ..Default::default()
    };
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, config)
}

fuzz_target!(|input: FuzzInput| {
    // Limit operation count
    if input.operations.len() > 150 {
        return;
    }

    let coordinator = create_test_coordinator();
    let detector_config = DeadlockDetectorConfig::default().with_policy(input.policy.into());
    let detector = DeadlockDetector::new(detector_config);

    let mut active_tx_ids: HashSet<u64> = HashSet::new();
    let mut committed_tx_ids: HashSet<u64> = HashSet::new();
    let mut aborted_tx_ids: HashSet<u64> = HashSet::new();

    for op in &input.operations {
        match op {
            FuzzOp::BeginTransaction {
                node_idx,
                participant_count,
            } => {
                let node = format!("node{}", node_idx % 5);
                let count = (*participant_count as usize % 5).max(1);
                let participants: Vec<usize> = (0..count).collect();

                if let Ok(tx) = coordinator.begin(&node, &participants) {
                    active_tx_ids.insert(tx.tx_id);
                }
            },

            FuzzOp::PrepareOnShard {
                tx_id_mod,
                key_count,
                key_base,
            } => {
                let tx_id = (*tx_id_mod as u64) % 100 + 1;
                let count = (*key_count as usize % 5).max(1);
                let base = *key_base as usize;

                let operations: Vec<Transaction> = (0..count)
                    .map(|i| Transaction::Put {
                        key: format!("key_{}", (base + i) % 50),
                        data: vec![i as u8],
                    })
                    .collect();

                let request = PrepareRequest {
                    tx_id,
                    coordinator: "node0".to_string(),
                    operations,
                    delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
                    timeout_ms: 5000,
                };

                let _ = coordinator.handle_prepare(&request);
            },

            FuzzOp::RecordVote {
                tx_id_mod,
                shard_mod,
                is_yes,
            } => {
                let tx_id = (*tx_id_mod as u64) % 100 + 1;
                let shard = (*shard_mod as usize) % 10;

                if active_tx_ids.contains(&tx_id) || coordinator.get(tx_id).is_some() {
                    let vote = if *is_yes {
                        PrepareVote::Yes {
                            lock_handle: tx_id * 1000 + shard as u64,
                            delta: DeltaVector::from_sparse(
                                SparseVector::from_dense(&[1.0, 0.0, 0.0]),
                                std::iter::once(format!("key_{}", shard)).collect(),
                                tx_id,
                            ),
                        }
                    } else {
                        PrepareVote::Conflict {
                            similarity: 0.9,
                            conflicting_tx: tx_id + 1,
                        }
                    };
                    let _ = coordinator.record_vote(tx_id, shard, vote);
                }
            },

            FuzzOp::Commit { tx_id_mod } => {
                let tx_id = (*tx_id_mod as u64) % 100 + 1;
                if coordinator.commit(tx_id).is_ok() {
                    active_tx_ids.remove(&tx_id);
                    committed_tx_ids.insert(tx_id);
                }
            },

            FuzzOp::Abort { tx_id_mod } => {
                let tx_id = (*tx_id_mod as u64) % 100 + 1;
                if coordinator.abort(tx_id, "fuzz abort").is_ok() {
                    active_tx_ids.remove(&tx_id);
                    aborted_tx_ids.insert(tx_id);
                }
            },

            FuzzOp::RunDeadlockDetection => {
                let deadlocks = detector.detect();
                for dl in &deadlocks {
                    // Invariant: victim should be in the cycle
                    assert!(
                        dl.cycle.contains(&dl.victim_tx_id),
                        "victim {} should be in cycle {:?}",
                        dl.victim_tx_id,
                        dl.cycle
                    );
                }
            },

            FuzzOp::CleanupTimeouts => {
                let timed_out = coordinator.cleanup_timeouts();
                for tx_id in timed_out {
                    active_tx_ids.remove(&tx_id);
                }
            },

            FuzzOp::AddDirectWaitEdge {
                waiter_mod,
                holder_mod,
                priority,
            } => {
                let waiter = (*waiter_mod as u64) % 100 + 1;
                let holder = (*holder_mod as u64) % 100 + 1;
                let prio = priority.map(|p| p as u32);
                detector.graph().add_wait(waiter, holder, prio);
            },

            FuzzOp::RemoveWaitEdges { tx_id_mod } => {
                let tx_id = (*tx_id_mod as u64) % 100 + 1;
                detector.graph().remove_transaction(tx_id);
            },

            FuzzOp::CheckForCycles => {
                let cycles = coordinator.wait_graph().detect_cycles();
                // Cycles should be valid (non-empty if detected)
                for cycle in &cycles {
                    assert!(!cycle.is_empty(), "detected cycle should not be empty");
                    assert!(cycle.len() >= 2, "cycle should have at least 2 nodes");
                }
            },
        }
    }

    // Final invariants
    // 1. No transaction should be both committed and aborted
    let overlap: HashSet<_> = committed_tx_ids.intersection(&aborted_tx_ids).collect();
    assert!(
        overlap.is_empty(),
        "transactions cannot be both committed and aborted: {:?}",
        overlap
    );

    // 2. Pending count should be consistent
    let pending = coordinator.pending_count();
    assert!(
        pending <= 50,
        "pending count {} exceeds max_concurrent",
        pending
    );

    // 3. Wait graph should be in valid state
    let edge_count = coordinator.wait_graph().edge_count();
    assert!(edge_count <= 10000, "edge count too large: {}", edge_count);
});
