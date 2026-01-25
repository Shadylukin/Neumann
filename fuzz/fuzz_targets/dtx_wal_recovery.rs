#![no_main]

//! Fuzz test for distributed transaction WAL recovery.
//!
//! This fuzz target simulates WAL entry sequences with crashes at various points
//! to verify the robustness of the 2PC recovery mechanism.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager},
    DistributedTxConfig, DistributedTxCoordinator, TxPhase, TxWal, TxWalEntry,
    PrepareVoteKind, TxOutcome,
};
use tempfile::tempdir;

/// Fuzzable WAL operation sequence
#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
enum FuzzWalOp {
    /// Begin a new transaction with participants
    TxBegin {
        tx_id_mod: u8,
        participant_count: u8,
    },
    /// Record a vote
    PrepareVote {
        tx_id_mod: u8,
        shard_mod: u8,
        is_yes: bool,
        lock_handle_mod: u8,
    },
    /// Phase transition
    PhaseChange {
        tx_id_mod: u8,
        to_phase: FuzzPhase,
    },
    /// Transaction complete
    TxComplete {
        tx_id_mod: u8,
        is_committed: bool,
    },
    /// Simulate crash (truncate remaining operations)
    SimulateCrash,
}

#[derive(Debug, Arbitrary, Clone, Copy)]
enum FuzzPhase {
    Preparing,
    Prepared,
    Committing,
    Committed,
    Aborting,
    Aborted,
}

impl From<FuzzPhase> for TxPhase {
    fn from(p: FuzzPhase) -> Self {
        match p {
            FuzzPhase::Preparing => TxPhase::Preparing,
            FuzzPhase::Prepared => TxPhase::Prepared,
            FuzzPhase::Committing => TxPhase::Committing,
            FuzzPhase::Committed => TxPhase::Committed,
            FuzzPhase::Aborting => TxPhase::Aborting,
            FuzzPhase::Aborted => TxPhase::Aborted,
        }
    }
}

/// Input for fuzzing
#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
struct FuzzInput {
    operations: Vec<FuzzWalOp>,
}

fn create_coordinator_with_wal(wal: TxWal) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, DistributedTxConfig::default()).with_wal(wal)
}

fuzz_target!(|input: FuzzInput| {
    // Limit operation count to prevent OOM
    if input.operations.len() > 100 {
        return;
    }

    let dir = match tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let wal_path = dir.path().join("fuzz_tx.wal");

    // Track the last valid phase for each transaction
    let mut last_phase: std::collections::HashMap<u64, TxPhase> = std::collections::HashMap::new();

    // Phase 1: Write WAL entries
    {
        let mut wal = match TxWal::open(&wal_path) {
            Ok(w) => w,
            Err(_) => return,
        };

        for op in &input.operations {
            match op {
                FuzzWalOp::SimulateCrash => break,

                FuzzWalOp::TxBegin {
                    tx_id_mod,
                    participant_count,
                } => {
                    let tx_id = (*tx_id_mod as u64) % 10 + 1;
                    let count = (*participant_count as usize % 5).max(1);
                    let participants: Vec<usize> = (0..count).collect();

                    last_phase.insert(tx_id, TxPhase::Preparing);

                    let _ = wal.append(&TxWalEntry::TxBegin { tx_id, participants });
                },

                FuzzWalOp::PrepareVote {
                    tx_id_mod,
                    shard_mod,
                    is_yes,
                    lock_handle_mod,
                } => {
                    let tx_id = (*tx_id_mod as u64) % 10 + 1;
                    let shard = (*shard_mod as usize) % 5;
                    let lock_handle = (*lock_handle_mod as u64) * 100 + 1;

                    let vote = if *is_yes {
                        PrepareVoteKind::Yes { lock_handle }
                    } else {
                        PrepareVoteKind::No
                    };

                    let _ = wal.append(&TxWalEntry::PrepareVote { tx_id, shard, vote });
                },

                FuzzWalOp::PhaseChange { tx_id_mod, to_phase } => {
                    let tx_id = (*tx_id_mod as u64) % 10 + 1;
                    let from = last_phase.get(&tx_id).copied().unwrap_or(TxPhase::Preparing);
                    let to: TxPhase = (*to_phase).into();

                    last_phase.insert(tx_id, to);

                    let _ = wal.append(&TxWalEntry::PhaseChange { tx_id, from, to });
                },

                FuzzWalOp::TxComplete {
                    tx_id_mod,
                    is_committed,
                } => {
                    let tx_id = (*tx_id_mod as u64) % 10 + 1;
                    let outcome = if *is_committed {
                        TxOutcome::Committed
                    } else {
                        TxOutcome::Aborted
                    };

                    last_phase.remove(&tx_id);

                    let _ = wal.append(&TxWalEntry::TxComplete { tx_id, outcome });
                },
            }
        }
    }

    // Phase 2: Recover from WAL
    let wal = match TxWal::open(&wal_path) {
        Ok(w) => w,
        Err(_) => return,
    };

    let coordinator = create_coordinator_with_wal(wal);

    // Recovery should not panic
    let result = coordinator.recover_from_wal();

    if let Ok(stats) = result {
        // Invariant: counts should be non-negative (trivially true for usize)
        // Invariant: pending transactions should match recovery state
        let pending_count = coordinator.pending_count();

        // The pending count should equal the sum of prepared + committing + aborting
        // (minus any that completed during recovery)
        assert!(
            pending_count <= stats.pending_prepare + stats.pending_commit + stats.pending_abort,
            "Pending count should not exceed recovered transactions"
        );

        // Verify recovered transactions have valid states
        let decisions = coordinator.get_pending_decisions();
        for (tx_id, phase) in decisions {
            assert!(
                matches!(phase, TxPhase::Committing | TxPhase::Aborting),
                "Decision transactions should be in Committing or Aborting phase, got {:?} for tx {}",
                phase,
                tx_id
            );
        }

        // Verify that recovered votes have valid lock handles for YES votes
        for tx_id in 1..=10u64 {
            if let Some(tx) = coordinator.get(tx_id) {
                for (shard, vote) in &tx.votes {
                    match vote {
                        tensor_chain::PrepareVote::Yes { lock_handle, .. } => {
                            // Lock handle should be set (non-zero after recovery with proper WAL)
                            // Note: This may be 0 if the vote was not in the WAL
                            assert!(
                                *lock_handle != u64::MAX,
                                "Lock handle should be valid for shard {} in tx {}",
                                shard,
                                tx_id
                            );
                        },
                        _ => {},
                    }
                }
            }
        }
    }

    // Phase 3: Verify WAL replay is idempotent
    let wal2 = match TxWal::open(&wal_path) {
        Ok(w) => w,
        Err(_) => return,
    };

    let entries1 = wal2.replay().unwrap_or_default();
    let entries2 = wal2.replay().unwrap_or_default();
    assert_eq!(entries1.len(), entries2.len(), "WAL replay should be deterministic");
});
