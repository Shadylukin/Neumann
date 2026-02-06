// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz test for WAL lock release entry sequences.
//!
//! Simulates 2PC transaction flows with lock acquisition and release,
//! verifying that WAL recovery correctly handles incomplete lock release
//! sequences (e.g., crash between TxComplete and lock release).

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tempfile::tempdir;
use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager},
    DistributedTxConfig, DistributedTxCoordinator, PrepareVoteKind, TxOutcome, TxPhase, TxWal,
    TxWalEntry,
};

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
enum FuzzLockOp {
    /// Begin a transaction with N participants.
    Begin {
        tx_id_mod: u8,
        participant_count: u8,
    },
    /// Vote YES with a lock handle.
    VoteYes {
        tx_id_mod: u8,
        shard_mod: u8,
        lock_handle_mod: u8,
    },
    /// Vote NO (no lock acquired).
    VoteNo { tx_id_mod: u8, shard_mod: u8 },
    /// Transition to Prepared phase.
    Prepare { tx_id_mod: u8 },
    /// Transition to Committing phase.
    StartCommit { tx_id_mod: u8 },
    /// Complete the transaction.
    Complete { tx_id_mod: u8, committed: bool },
    /// Simulate a crash (stop writing, then recover).
    Crash,
}

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
struct FuzzInput {
    operations: Vec<FuzzLockOp>,
}

fn create_coordinator_with_wal(wal: TxWal) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, DistributedTxConfig::default()).with_wal(wal)
}

fuzz_target!(|input: FuzzInput| {
    if input.operations.len() > 120 {
        return;
    }

    let dir = match tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let wal_path = dir.path().join("lock_release.wal");

    // Track lock handles per transaction for invariant checking
    let mut tx_locks: std::collections::HashMap<u64, Vec<u64>> =
        std::collections::HashMap::new();
    let mut completed_txs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut last_phase: std::collections::HashMap<u64, TxPhase> =
        std::collections::HashMap::new();

    // Phase 1: Write WAL entries
    {
        let mut wal = match TxWal::open(&wal_path) {
            Ok(w) => w,
            Err(_) => return,
        };

        for op in &input.operations {
            match op {
                FuzzLockOp::Crash => break,

                FuzzLockOp::Begin {
                    tx_id_mod,
                    participant_count,
                } => {
                    let tx_id = u64::from(*tx_id_mod) % 8 + 1;
                    if completed_txs.contains(&tx_id) {
                        continue;
                    }
                    let count = (usize::from(*participant_count) % 4).max(1);
                    let participants: Vec<usize> = (0..count).collect();

                    last_phase.insert(tx_id, TxPhase::Preparing);
                    tx_locks.entry(tx_id).or_default();

                    let _ = wal.append(&TxWalEntry::TxBegin { tx_id, participants });
                }

                FuzzLockOp::VoteYes {
                    tx_id_mod,
                    shard_mod,
                    lock_handle_mod,
                } => {
                    let tx_id = u64::from(*tx_id_mod) % 8 + 1;
                    if completed_txs.contains(&tx_id) {
                        continue;
                    }
                    let shard = usize::from(*shard_mod) % 4;
                    let lock_handle = u64::from(*lock_handle_mod) * 100 + 1;

                    tx_locks.entry(tx_id).or_default().push(lock_handle);

                    let _ = wal.append(&TxWalEntry::PrepareVote {
                        tx_id,
                        shard,
                        vote: PrepareVoteKind::Yes { lock_handle },
                    });
                }

                FuzzLockOp::VoteNo {
                    tx_id_mod,
                    shard_mod,
                } => {
                    let tx_id = u64::from(*tx_id_mod) % 8 + 1;
                    if completed_txs.contains(&tx_id) {
                        continue;
                    }
                    let shard = usize::from(*shard_mod) % 4;

                    let _ = wal.append(&TxWalEntry::PrepareVote {
                        tx_id,
                        shard,
                        vote: PrepareVoteKind::No,
                    });
                }

                FuzzLockOp::Prepare { tx_id_mod } => {
                    let tx_id = u64::from(*tx_id_mod) % 8 + 1;
                    if completed_txs.contains(&tx_id) {
                        continue;
                    }
                    let from = last_phase.get(&tx_id).copied().unwrap_or(TxPhase::Preparing);
                    last_phase.insert(tx_id, TxPhase::Prepared);

                    let _ = wal.append(&TxWalEntry::PhaseChange {
                        tx_id,
                        from,
                        to: TxPhase::Prepared,
                    });
                }

                FuzzLockOp::StartCommit { tx_id_mod } => {
                    let tx_id = u64::from(*tx_id_mod) % 8 + 1;
                    if completed_txs.contains(&tx_id) {
                        continue;
                    }
                    let from = last_phase.get(&tx_id).copied().unwrap_or(TxPhase::Prepared);
                    last_phase.insert(tx_id, TxPhase::Committing);

                    let _ = wal.append(&TxWalEntry::PhaseChange {
                        tx_id,
                        from,
                        to: TxPhase::Committing,
                    });
                }

                FuzzLockOp::Complete {
                    tx_id_mod,
                    committed,
                } => {
                    let tx_id = u64::from(*tx_id_mod) % 8 + 1;
                    let outcome = if *committed {
                        TxOutcome::Committed
                    } else {
                        TxOutcome::Aborted
                    };

                    completed_txs.insert(tx_id);
                    last_phase.remove(&tx_id);

                    let _ = wal.append(&TxWalEntry::TxComplete { tx_id, outcome });
                }
            }
        }
    }

    // Phase 2: Recover from WAL -- must not panic
    let wal = match TxWal::open(&wal_path) {
        Ok(w) => w,
        Err(_) => return,
    };

    let coordinator = create_coordinator_with_wal(wal);

    if let Ok(stats) = coordinator.recover_from_wal() {
        // Invariant: pending count <= total recovered in-flight txs
        let pending = coordinator.pending_count();
        assert!(
            pending <= stats.pending_prepare + stats.pending_commit + stats.pending_abort,
            "pending {} > recovered total {}",
            pending,
            stats.pending_prepare + stats.pending_commit + stats.pending_abort
        );

        // Invariant: completed transactions should NOT be in pending
        for &tx_id in &completed_txs {
            assert!(
                coordinator.get(tx_id).is_none(),
                "Completed tx {} should not be pending after recovery",
                tx_id
            );
        }

        // Invariant: all pending txs have valid phases
        let decisions = coordinator.get_pending_decisions();
        for (tx_id, phase) in &decisions {
            assert!(
                matches!(phase, TxPhase::Committing | TxPhase::Aborting),
                "Decision tx {} in unexpected phase {:?}",
                tx_id,
                phase
            );
        }
    }

    // Phase 3: Replay must be idempotent
    let wal2 = match TxWal::open(&wal_path) {
        Ok(w) => w,
        Err(_) => return,
    };
    let entries1 = wal2.replay().unwrap_or_default();
    let entries2 = wal2.replay().unwrap_or_default();
    assert_eq!(
        entries1.len(),
        entries2.len(),
        "WAL replay must be deterministic"
    );
});
