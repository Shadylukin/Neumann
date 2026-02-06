// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for 2PC crash recovery with incomplete lock releases.
//!
//! Simulates scenarios where the coordinator crashes between committing a
//! transaction and releasing its locks, verifying that WAL recovery correctly
//! handles orphaned lock state.

use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager},
    DistributedTxConfig, DistributedTxCoordinator, PrepareVoteKind, TxOutcome, TxPhase, TxWal,
    TxWalEntry,
};

fn create_coordinator_with_wal(wal: TxWal) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, DistributedTxConfig::default()).with_wal(wal)
}

#[test]
fn test_wal_recovery_committed_tx_not_pending() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Write a complete transaction lifecycle
    {
        let mut wal = TxWal::open(&wal_path).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 100,
            participants: vec![0, 1],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 100,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 1 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 100,
            shard: 1,
            vote: PrepareVoteKind::Yes { lock_handle: 2 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 100,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 100,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })
        .unwrap();
        wal.append(&TxWalEntry::TxComplete {
            tx_id: 100,
            outcome: TxOutcome::Committed,
        })
        .unwrap();
    }

    // Recover -- completed tx should not appear in pending
    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    assert_eq!(
        coordinator.pending_count(),
        0,
        "Committed tx should not be pending"
    );
    assert!(
        coordinator.get(100).is_none(),
        "Completed tx should not be in pending map"
    );
    // Recovery only restores incomplete txs; completed txs are simply pruned from replay state
    assert_eq!(
        stats.pending_prepare + stats.pending_commit + stats.pending_abort,
        0,
        "Fully committed tx should produce no pending work"
    );
}

#[test]
fn test_wal_recovery_crash_during_prepare() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Simulate crash during prepare: Begin + 1 vote, no completion
    {
        let mut wal = TxWal::open(&wal_path).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 200,
            participants: vec![0, 1],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 200,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 10 },
        })
        .unwrap();
        // Crash here -- no TxComplete
    }

    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    // TX 200 never reached Prepared phase (still in Preparing), so recovery
    // treats it as an incomplete prepare that can be safely aborted on timeout.
    // It is NOT restored to the pending map since no decision was reached.
    assert_eq!(
        stats.pending_prepare + stats.pending_commit + stats.pending_abort,
        0,
        "Preparing-phase tx is not restored (no decision reached)"
    );
    assert_eq!(coordinator.pending_count(), 0);
}

#[test]
fn test_wal_recovery_crash_during_commit() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Simulate crash after entering Committing phase but before TxComplete
    {
        let mut wal = TxWal::open(&wal_path).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 300,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 300,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 20 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 300,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 300,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })
        .unwrap();
        // Crash here -- committed decision made but not completed
    }

    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    // Should recover with a pending commit decision
    assert!(
        stats.pending_commit > 0,
        "Should have pending commit decision"
    );

    let decisions = coordinator.get_pending_decisions();
    let commit_decisions: Vec<_> = decisions
        .iter()
        .filter(|(_, phase)| matches!(phase, TxPhase::Committing))
        .collect();
    assert!(
        !commit_decisions.is_empty(),
        "Should have at least one Committing decision to complete"
    );
}

#[test]
fn test_wal_recovery_crash_during_abort() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Simulate crash after entering Aborting phase
    {
        let mut wal = TxWal::open(&wal_path).unwrap();
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 400,
            participants: vec![0, 1],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 400,
            shard: 0,
            vote: PrepareVoteKind::No,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 400,
            from: TxPhase::Preparing,
            to: TxPhase::Aborting,
        })
        .unwrap();
        // Crash -- abort started but not completed
    }

    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    assert!(
        stats.pending_abort > 0,
        "Should have pending abort decision"
    );

    let decisions = coordinator.get_pending_decisions();
    let abort_decisions: Vec<_> = decisions
        .iter()
        .filter(|(_, phase)| matches!(phase, TxPhase::Aborting))
        .collect();
    assert!(
        !abort_decisions.is_empty(),
        "Should have at least one Aborting decision to complete"
    );
}

#[test]
fn test_wal_recovery_multiple_transactions_mixed() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    {
        let mut wal = TxWal::open(&wal_path).unwrap();

        // TX 500: fully committed
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 500,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 500,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 50 },
        })
        .unwrap();
        wal.append(&TxWalEntry::TxComplete {
            tx_id: 500,
            outcome: TxOutcome::Committed,
        })
        .unwrap();

        // TX 501: still in prepare (incomplete)
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 501,
            participants: vec![0, 1],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 501,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 51 },
        })
        .unwrap();

        // TX 502: fully aborted
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 502,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::TxComplete {
            tx_id: 502,
            outcome: TxOutcome::Aborted,
        })
        .unwrap();

        // TX 503: in committing phase (needs completion)
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 503,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 503,
            from: TxPhase::Preparing,
            to: TxPhase::Committing,
        })
        .unwrap();
    }

    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    // TX 500 and 502 completed, should not be pending
    assert!(
        coordinator.get(500).is_none(),
        "TX 500 completed, should not be pending"
    );
    assert!(
        coordinator.get(502).is_none(),
        "TX 502 completed, should not be pending"
    );

    // TX 501 is in Preparing (no Prepared phase change), so recovery drops it.
    // TX 503 is in Committing, so recovery restores it.
    assert_eq!(stats.pending_commit, 1, "TX 503 should be pending commit");
    assert_eq!(stats.pending_prepare, 0, "TX 501 never reached Prepared");
    assert_eq!(
        coordinator.pending_count(),
        1,
        "Only TX 503 should be pending"
    );
}

#[test]
fn test_wal_replay_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    {
        let mut wal = TxWal::open(&wal_path).unwrap();
        for tx_id in 1..=5 {
            wal.append(&TxWalEntry::TxBegin {
                tx_id,
                participants: vec![0],
            })
            .unwrap();
            wal.append(&TxWalEntry::PrepareVote {
                tx_id,
                shard: 0,
                vote: PrepareVoteKind::Yes {
                    lock_handle: tx_id * 10,
                },
            })
            .unwrap();
            wal.append(&TxWalEntry::TxComplete {
                tx_id,
                outcome: TxOutcome::Committed,
            })
            .unwrap();
        }
    }

    let wal = TxWal::open(&wal_path).unwrap();
    let entries1 = wal.replay().unwrap();
    let entries2 = wal.replay().unwrap();

    assert_eq!(
        entries1.len(),
        entries2.len(),
        "Replay must be deterministic"
    );
    assert_eq!(
        entries1.len(),
        15,
        "Should have 15 entries (3 per tx * 5 txs)"
    );
}
