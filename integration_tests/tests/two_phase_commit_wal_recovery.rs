// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for 2PC WAL recovery fixes.
//!
//! Tests the specific fixes for:
//! - Issue 1: TxComplete logged before lock release
//! - Issue 2: Lock handles persisted in PrepareVoteKind::Yes
//! - Issue 3: Votes logged to WAL in record_vote()

use tempfile::tempdir;
use tensor_chain::{
    block::Transaction,
    consensus::{ConsensusConfig, ConsensusManager},
    DistributedTxConfig, DistributedTxCoordinator, PrepareRequest, PrepareVote, PrepareVoteKind,
    TxPhase, TxWal, TxWalEntry,
};
use tensor_store::SparseVector;

fn create_coordinator_with_wal(wal: TxWal) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, DistributedTxConfig::default()).with_wal(wal)
}

// ============= Issue 1: TxComplete logged before lock release =============

#[test]
fn test_coordinator_crash_after_txcomplete_logged_releases_locks_once() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    // Phase 1: Create a WAL with a transaction that has TxComplete logged
    // This simulates a crash AFTER TxComplete was logged but BEFORE locks were released
    {
        let mut wal = TxWal::open(&wal_path).unwrap();

        // Begin transaction
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 42,
            participants: vec![0, 1],
        })
        .unwrap();

        // Record YES votes with lock handles
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 42,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 100 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 42,
            shard: 1,
            vote: PrepareVoteKind::Yes { lock_handle: 101 },
        })
        .unwrap();

        // Phase change to Prepared
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 42,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();

        // Phase change to Committing
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 42,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })
        .unwrap();

        // TxComplete logged - after this, recovery should NOT try to release locks again
        wal.append(&TxWalEntry::TxComplete {
            tx_id: 42,
            outcome: tensor_chain::TxOutcome::Committed,
        })
        .unwrap();
    }

    // Phase 2: Recover from WAL
    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    // Transaction should be removed (TxComplete was logged)
    // Recovery should NOT try to release locks again
    assert_eq!(
        stats.pending_commit, 0,
        "Transaction should be completed, not pending"
    );
    assert_eq!(
        stats.pending_prepare, 0,
        "No transactions should be pending"
    );
    assert_eq!(
        coordinator.pending_count(),
        0,
        "No pending transactions after recovery"
    );
}

#[test]
fn test_coordinator_crash_before_txcomplete_retries_commit() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    // Phase 1: Create a WAL with a transaction in Committing phase but NO TxComplete
    {
        let mut wal = TxWal::open(&wal_path).unwrap();

        wal.append(&TxWalEntry::TxBegin {
            tx_id: 42,
            participants: vec![0],
        })
        .unwrap();

        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 42,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 100 },
        })
        .unwrap();

        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 42,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();

        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 42,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })
        .unwrap();

        // No TxComplete - crash happened before it was logged
    }

    // Phase 2: Recover from WAL
    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    // Transaction should be marked for commit retry
    assert_eq!(
        stats.pending_commit, 1,
        "Transaction should need commit completion"
    );
    assert_eq!(coordinator.pending_count(), 1, "One pending transaction");

    // Complete the commit
    coordinator.complete_commit(42).unwrap();
    assert_eq!(
        coordinator.pending_count(),
        0,
        "No pending transactions after completion"
    );
}

// ============= Issue 2: Lock handles persisted in PrepareVoteKind::Yes =============

#[test]
fn test_participant_vote_persisted_with_lock_handle() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    let lock_handle_shard0 = 12345u64;
    let lock_handle_shard1 = 67890u64;

    // Phase 1: Create WAL with votes containing lock handles
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
            vote: PrepareVoteKind::Yes {
                lock_handle: lock_handle_shard0,
            },
        })
        .unwrap();

        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 100,
            shard: 1,
            vote: PrepareVoteKind::Yes {
                lock_handle: lock_handle_shard1,
            },
        })
        .unwrap();

        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 100,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();
    }

    // Phase 2: Recover from WAL and verify lock handles are restored
    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    assert_eq!(stats.pending_prepare, 1, "One prepared transaction");

    // Get the recovered transaction and verify lock handles
    let tx = coordinator.get(100).expect("Transaction should exist");
    assert_eq!(tx.votes.len(), 2, "Should have 2 votes");

    for (shard, vote) in &tx.votes {
        match vote {
            PrepareVote::Yes { lock_handle, .. } => {
                if *shard == 0 {
                    assert_eq!(
                        *lock_handle, lock_handle_shard0,
                        "Lock handle for shard 0 should be preserved"
                    );
                } else if *shard == 1 {
                    assert_eq!(
                        *lock_handle, lock_handle_shard1,
                        "Lock handle for shard 1 should be preserved"
                    );
                }
            },
            _ => panic!("Expected Yes vote"),
        }
    }
}

#[test]
fn test_recovery_from_committing_with_lock_handles() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    let expected_lock_handle = 99999u64;

    // Phase 1: Create WAL with transaction in Committing phase
    {
        let mut wal = TxWal::open(&wal_path).unwrap();

        wal.append(&TxWalEntry::TxBegin {
            tx_id: 200,
            participants: vec![0],
        })
        .unwrap();

        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 200,
            shard: 0,
            vote: PrepareVoteKind::Yes {
                lock_handle: expected_lock_handle,
            },
        })
        .unwrap();

        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 200,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();

        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 200,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })
        .unwrap();
    }

    // Phase 2: Recover and verify lock handle is available for release
    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    assert_eq!(stats.pending_commit, 1, "One transaction pending commit");

    let tx = coordinator.get(200).expect("Transaction should exist");
    if let Some(PrepareVote::Yes { lock_handle, .. }) = tx.votes.get(&0) {
        assert_eq!(
            *lock_handle, expected_lock_handle,
            "Lock handle should be preserved during committing recovery"
        );
    } else {
        panic!("Expected Yes vote with lock handle");
    }

    // Complete commit should work with preserved lock handle
    coordinator.complete_commit(200).unwrap();
    assert_eq!(coordinator.pending_count(), 0);
}

// ============= Issue 3: Votes logged to WAL in record_vote() =============

#[test]
fn test_record_vote_logs_to_wal() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    let tx_id;
    {
        let wal = TxWal::open(&wal_path).unwrap();
        let coordinator = create_coordinator_with_wal(wal);

        // Begin a transaction
        let tx = coordinator
            .begin("coord1".to_string(), vec![0])
            .expect("Failed to begin transaction");
        tx_id = tx.tx_id;

        // Create a prepare request
        let request = PrepareRequest {
            tx_id,
            coordinator: "coord1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1, 2, 3],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        };

        // Handle prepare to get a vote
        let vote = coordinator.handle_prepare(request);

        // Record the vote - this should log to WAL
        coordinator.record_vote(tx_id, 0, vote);
    }

    // Verify WAL contains the vote entry
    let wal = TxWal::open(&wal_path).unwrap();
    let entries = wal.replay().unwrap();

    // Find the PrepareVote entry
    let vote_entries: Vec<_> = entries
        .iter()
        .filter(|e| matches!(e, TxWalEntry::PrepareVote { .. }))
        .collect();

    assert!(
        !vote_entries.is_empty(),
        "WAL should contain PrepareVote entry"
    );

    if let TxWalEntry::PrepareVote {
        tx_id: logged_tx_id,
        shard,
        vote,
    } = &vote_entries[0]
    {
        assert_eq!(
            *logged_tx_id, tx_id,
            "Vote should be for correct transaction"
        );
        assert_eq!(*shard, 0, "Vote should be for shard 0");
        assert!(
            matches!(vote, PrepareVoteKind::Yes { lock_handle } if *lock_handle > 0),
            "Vote should be Yes with valid lock handle"
        );
    }
}

#[test]
fn test_vote_persisted_survives_crash() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    let tx_id;
    let lock_handle;

    // Phase 1: Record vote and "crash" (drop coordinator)
    {
        let wal = TxWal::open(&wal_path).unwrap();
        let coordinator = create_coordinator_with_wal(wal);

        let tx = coordinator
            .begin("coord1".to_string(), vec![0])
            .expect("Failed to begin transaction");
        tx_id = tx.tx_id;

        let request = PrepareRequest {
            tx_id,
            coordinator: "coord1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        };

        let vote = coordinator.handle_prepare(request);

        // Extract lock handle from vote
        lock_handle = match &vote {
            PrepareVote::Yes { lock_handle, .. } => *lock_handle,
            _ => panic!("Expected Yes vote"),
        };

        // Record the vote
        coordinator.record_vote(tx_id, 0, vote);

        // Coordinator is dropped here (simulating crash)
    }

    // Phase 2: Recover and verify vote is preserved
    {
        let wal = TxWal::open(&wal_path).unwrap();
        let coordinator = create_coordinator_with_wal(wal);
        let stats = coordinator.recover_from_wal().unwrap();

        // Transaction should be recovered with vote
        assert_eq!(stats.pending_prepare, 1, "Transaction should be prepared");

        let tx = coordinator.get(tx_id).expect("Transaction should exist");
        assert_eq!(tx.votes.len(), 1, "Should have 1 vote");

        if let Some(PrepareVote::Yes {
            lock_handle: recovered_lock_handle,
            ..
        }) = tx.votes.get(&0)
        {
            assert_eq!(
                *recovered_lock_handle, lock_handle,
                "Lock handle should be preserved across crash"
            );
        } else {
            panic!("Expected Yes vote after recovery");
        }
    }
}

// ============= Concurrent crash recovery tests =============

#[test]
fn test_concurrent_crash_recovery_no_deadlock() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    // Create WAL with multiple transactions in various states
    {
        let mut wal = TxWal::open(&wal_path).unwrap();

        // Transaction 1: Prepared
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 1,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 1,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 100 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 1,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();

        // Transaction 2: Committing
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 2,
            participants: vec![1],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 2,
            shard: 1,
            vote: PrepareVoteKind::Yes { lock_handle: 200 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 2,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 2,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })
        .unwrap();

        // Transaction 3: Aborting
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 3,
            participants: vec![2],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 3,
            shard: 2,
            vote: PrepareVoteKind::No,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 3,
            from: TxPhase::Preparing,
            to: TxPhase::Aborting,
        })
        .unwrap();

        // Transaction 4: Completed (should be ignored)
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 4,
            participants: vec![3],
        })
        .unwrap();
        wal.append(&TxWalEntry::TxComplete {
            tx_id: 4,
            outcome: tensor_chain::TxOutcome::Committed,
        })
        .unwrap();
    }

    // Recover all transactions
    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    // Verify correct recovery counts
    assert_eq!(stats.pending_prepare, 1, "One prepared transaction (tx 1)");
    assert_eq!(stats.pending_commit, 1, "One committing transaction (tx 2)");
    assert_eq!(stats.pending_abort, 1, "One aborting transaction (tx 3)");

    // Transaction 4 should not appear (it has TxComplete)
    assert_eq!(coordinator.pending_count(), 3, "3 transactions pending");
    assert!(
        coordinator.get(4).is_none(),
        "Completed transaction should not be pending"
    );

    // Complete all pending transactions without deadlock
    coordinator.complete_commit(2).unwrap();
    coordinator.complete_abort(3).unwrap();
    coordinator.abort(1, "test cleanup").unwrap();

    assert_eq!(coordinator.pending_count(), 0, "All transactions resolved");
}

#[test]
fn test_no_vote_entry_recovers_as_preparing() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    // Create WAL with transaction that has no votes logged
    {
        let mut wal = TxWal::open(&wal_path).unwrap();

        wal.append(&TxWalEntry::TxBegin {
            tx_id: 500,
            participants: vec![0, 1],
        })
        .unwrap();
        // No votes recorded - transaction was in preparing phase when crash occurred
    }

    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    // Transaction should not be in any of the prepared/committing/aborting lists
    // since it never reached those phases
    assert_eq!(stats.pending_prepare, 0);
    assert_eq!(stats.pending_commit, 0);
    assert_eq!(stats.pending_abort, 0);
}

#[test]
fn test_mixed_vote_types_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    // Create WAL with transaction having mixed Yes and No votes
    {
        let mut wal = TxWal::open(&wal_path).unwrap();

        wal.append(&TxWalEntry::TxBegin {
            tx_id: 600,
            participants: vec![0, 1, 2],
        })
        .unwrap();

        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 600,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 111 },
        })
        .unwrap();

        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 600,
            shard: 1,
            vote: PrepareVoteKind::No,
        })
        .unwrap();

        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 600,
            shard: 2,
            vote: PrepareVoteKind::Yes { lock_handle: 333 },
        })
        .unwrap();

        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 600,
            from: TxPhase::Preparing,
            to: TxPhase::Aborting,
        })
        .unwrap();
    }

    let wal = TxWal::open(&wal_path).unwrap();
    let coordinator = create_coordinator_with_wal(wal);
    let stats = coordinator.recover_from_wal().unwrap();

    assert_eq!(stats.pending_abort, 1, "Transaction should be aborting");

    let tx = coordinator.get(600).expect("Transaction should exist");
    assert_eq!(tx.votes.len(), 3, "Should have 3 votes recovered");

    // Verify vote types
    assert!(matches!(
        tx.votes.get(&0),
        Some(PrepareVote::Yes {
            lock_handle: 111,
            ..
        })
    ));
    assert!(matches!(tx.votes.get(&1), Some(PrepareVote::No { .. })));
    assert!(matches!(
        tx.votes.get(&2),
        Some(PrepareVote::Yes {
            lock_handle: 333,
            ..
        })
    ));
}
