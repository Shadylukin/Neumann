//! Integration tests for atomic snapshot and WAL operations.
//!
//! These tests verify that:
//! 1. WAL truncate operations are atomic (no temp files left behind)
//! 2. Snapshot operations survive simulated restarts
//! 3. Concurrent operations on atomic_io are safe

use std::fs;
use std::sync::Arc;
use std::thread;

use tempfile::tempdir;
use tensor_chain::raft_wal::{RaftWal, RaftWalEntry, WalConfig};
use tensor_chain::tx_wal::{TxOutcome, TxWal, TxWalEntry};
use tensor_chain::{atomic_truncate, atomic_write, AtomicWriter};

#[test]
fn test_wal_truncate_atomic_no_temp_files() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("raft.wal");

    // Create WAL and add entries
    let mut wal = RaftWal::open(&wal_path).unwrap();
    for i in 0..10 {
        wal.append(&RaftWalEntry::TermChange { new_term: i })
            .unwrap();
    }
    assert_eq!(wal.entry_count(), 10);

    // Truncate
    wal.truncate().unwrap();
    assert_eq!(wal.entry_count(), 0);

    // Verify no temp files remain
    for entry in fs::read_dir(dir.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        assert!(
            !name.contains(".tmp."),
            "Temp file found after truncate: {}",
            name
        );
    }

    // Verify WAL is empty
    let entries = wal.replay().unwrap();
    assert!(entries.is_empty());
}

#[test]
fn test_tx_wal_truncate_atomic_no_temp_files() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("tx.wal");

    // Create WAL and add entries
    let mut wal = TxWal::open(&wal_path).unwrap();
    for i in 0..5 {
        wal.append(&TxWalEntry::TxBegin {
            tx_id: i,
            participants: vec![0, 1],
        })
        .unwrap();
    }
    assert_eq!(wal.entry_count(), 5);

    // Truncate
    wal.truncate().unwrap();
    assert_eq!(wal.entry_count(), 0);

    // Verify no temp files remain
    for entry in fs::read_dir(dir.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        assert!(
            !name.contains(".tmp."),
            "Temp file found after truncate: {}",
            name
        );
    }
}

#[test]
fn test_snapshot_persistence_survives_restart() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("restart.wal");

    // First "session" - create WAL and add entries
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        wal.append(&RaftWalEntry::TermChange { new_term: 1 })
            .unwrap();
        wal.append(&RaftWalEntry::VoteCast {
            term: 1,
            candidate_id: "node1".to_string(),
        })
        .unwrap();
        wal.append(&RaftWalEntry::TermAndVote {
            term: 2,
            voted_for: Some("node2".to_string()),
        })
        .unwrap();
        // WAL dropped here (simulates shutdown)
    }

    // Second "session" - verify data persisted
    {
        let wal = RaftWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();

        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], RaftWalEntry::TermChange { new_term: 1 });
        assert_eq!(
            entries[1],
            RaftWalEntry::VoteCast {
                term: 1,
                candidate_id: "node1".to_string()
            }
        );
        assert_eq!(
            entries[2],
            RaftWalEntry::TermAndVote {
                term: 2,
                voted_for: Some("node2".to_string())
            }
        );
    }
}

#[test]
fn test_concurrent_snapshot_operations() {
    let dir = tempdir().unwrap();
    let dir_path = Arc::new(dir.path().to_path_buf());

    // Spawn multiple threads doing atomic operations
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let dir = Arc::clone(&dir_path);
            thread::spawn(move || {
                let path = dir.join(format!("concurrent_{}.dat", i));

                // Write data atomically
                atomic_write(&path, format!("data_{}", i).as_bytes()).unwrap();

                // Verify the write
                let content = fs::read(&path).unwrap();
                assert_eq!(content, format!("data_{}", i).as_bytes());

                // Truncate atomically
                atomic_truncate(&path).unwrap();

                // Verify truncation
                let content = fs::read(&path).unwrap();
                assert!(content.is_empty());
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify all files exist and are empty
    for i in 0..10 {
        let path = dir_path.join(format!("concurrent_{}.dat", i));
        assert!(path.exists());
        let content = fs::read(&path).unwrap();
        assert!(content.is_empty());
    }

    // Verify no temp files remain
    for entry in fs::read_dir(&*dir_path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        assert!(!name.contains(".tmp."), "Temp file found: {}", name);
    }
}

#[test]
fn test_wal_truncate_then_append() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("truncate_append.wal");

    let mut wal = RaftWal::open(&wal_path).unwrap();

    // Add entries
    wal.append(&RaftWalEntry::TermChange { new_term: 1 })
        .unwrap();
    wal.append(&RaftWalEntry::TermChange { new_term: 2 })
        .unwrap();
    assert_eq!(wal.entry_count(), 2);

    // Truncate
    wal.truncate().unwrap();
    assert_eq!(wal.entry_count(), 0);

    // Append new entries after truncate
    wal.append(&RaftWalEntry::TermChange { new_term: 100 })
        .unwrap();
    wal.append(&RaftWalEntry::TermChange { new_term: 101 })
        .unwrap();
    assert_eq!(wal.entry_count(), 2);

    // Verify new entries
    let entries = wal.replay().unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0], RaftWalEntry::TermChange { new_term: 100 });
    assert_eq!(entries[1], RaftWalEntry::TermChange { new_term: 101 });
}

#[test]
fn test_atomic_writer_commit_abort_cycle() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("writer_test.dat");

    // First write and commit
    {
        let mut writer = AtomicWriter::new(&path).unwrap();
        use std::io::Write;
        writer.write_all(b"committed data").unwrap();
        writer.commit().unwrap();
    }

    assert!(path.exists());
    assert_eq!(fs::read(&path).unwrap(), b"committed data");

    // Second write and abort
    {
        let mut writer = AtomicWriter::new(&path).unwrap();
        use std::io::Write;
        writer.write_all(b"aborted data").unwrap();
        writer.abort();
    }

    // Original data should still be there
    assert_eq!(fs::read(&path).unwrap(), b"committed data");

    // Third write via drop (implicit abort)
    {
        let mut writer = AtomicWriter::new(&path).unwrap();
        use std::io::Write;
        writer.write_all(b"dropped data").unwrap();
        // Dropped without commit
    }

    // Original data should still be there
    assert_eq!(fs::read(&path).unwrap(), b"committed data");
}

#[test]
fn test_tx_wal_complete_lifecycle() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("lifecycle.wal");

    // Create WAL
    let mut wal = TxWal::open(&wal_path).unwrap();

    // Simulate a complete transaction lifecycle
    wal.append(&TxWalEntry::TxBegin {
        tx_id: 1,
        participants: vec![0, 1, 2],
    })
    .unwrap();

    wal.append(&TxWalEntry::PrepareVote {
        tx_id: 1,
        shard: 0,
        vote: tensor_chain::tx_wal::PrepareVoteKind::Yes { lock_handle: 100 },
    })
    .unwrap();

    wal.append(&TxWalEntry::PrepareVote {
        tx_id: 1,
        shard: 1,
        vote: tensor_chain::tx_wal::PrepareVoteKind::Yes { lock_handle: 101 },
    })
    .unwrap();

    wal.append(&TxWalEntry::PrepareVote {
        tx_id: 1,
        shard: 2,
        vote: tensor_chain::tx_wal::PrepareVoteKind::Yes { lock_handle: 102 },
    })
    .unwrap();

    wal.append(&TxWalEntry::PhaseChange {
        tx_id: 1,
        from: tensor_chain::TxPhase::Preparing,
        to: tensor_chain::TxPhase::Committing,
    })
    .unwrap();

    wal.append(&TxWalEntry::TxComplete {
        tx_id: 1,
        outcome: TxOutcome::Committed,
    })
    .unwrap();

    // Truncate after transaction complete
    wal.truncate().unwrap();

    // Verify clean slate
    let entries = wal.replay().unwrap();
    assert!(entries.is_empty());

    // Start new transaction
    wal.append(&TxWalEntry::TxBegin {
        tx_id: 2,
        participants: vec![0],
    })
    .unwrap();

    let entries = wal.replay().unwrap();
    assert_eq!(entries.len(), 1);
}

#[test]
fn test_wal_with_rotation_and_truncate() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("rotate_truncate.wal");

    let config = WalConfig {
        max_size_bytes: 200, // Force rotation
        auto_rotate: true,
        max_rotated_files: 2,
        ..Default::default()
    };

    let mut wal = RaftWal::open_with_config(&wal_path, config).unwrap();

    // Write entries to trigger rotation
    for i in 0..20 {
        wal.append(&RaftWalEntry::TermChange { new_term: i })
            .unwrap();
    }

    // Rotated files should exist
    let rotated1 = dir.path().join("rotate_truncate.wal.1");
    assert!(rotated1.exists());

    // Truncate main WAL
    wal.truncate().unwrap();

    // Main WAL should be empty
    let entries = wal.replay().unwrap();
    assert!(entries.is_empty());

    // No temp files
    for entry in fs::read_dir(dir.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        assert!(!name.contains(".tmp."), "Temp file found: {}", name);
    }
}
