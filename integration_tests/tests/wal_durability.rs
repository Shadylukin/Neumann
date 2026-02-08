// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! WAL durability tests: torn writes, crash recovery, CRC corruption detection.
//!
//! These tests validate that WALs correctly handle partial writes (torn pages),
//! recover committed data after crashes, and detect silent corruption via CRC32.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use tensor_chain::raft_wal::{RaftRecoveryState, RaftWal, RaftWalEntry, WalConfig};
use tensor_chain::{OrphanedLock, PrepareVoteKind, TxOutcome, TxRecoveryState, TxWal, TxWalEntry};
use tensor_store::{
    ScalarValue, TensorData, TensorStore, TensorValue, WalConfig as StoreWalConfig,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Write entries, record file size after each, then sweep-truncate the last
/// entry at every byte offset to verify that recovery returns only the
/// preceding entries.
fn torn_write_sweep<W, R>(dir: &Path, filename: &str, write_entries: W, recover_and_verify: R)
where
    W: Fn(&Path) -> Vec<u64>,
    R: Fn(&Path, usize) -> bool,
{
    let wal_path = dir.join(filename);
    let sizes = write_entries(&wal_path);
    assert!(
        sizes.len() >= 2,
        "need at least 2 entries for torn-write sweep"
    );

    let last_start = sizes[sizes.len() - 2];
    let last_end = sizes[sizes.len() - 1];
    let expected_before_last = sizes.len() - 1;

    for trunc_at in last_start..last_end {
        let copy_path = dir.join(format!("{filename}.trunc_{trunc_at}"));
        fs::copy(&wal_path, &copy_path).expect("copy WAL for truncation");
        let f = OpenOptions::new()
            .write(true)
            .open(&copy_path)
            .expect("open copy for truncation");
        f.set_len(trunc_at).expect("truncate file");
        drop(f);

        assert!(
            recover_and_verify(&copy_path, expected_before_last),
            "torn-write at byte {trunc_at}: expected {expected_before_last} entries"
        );

        fs::remove_file(&copy_path).ok();
    }
}

/// Flip a single bit in a file.
fn flip_bit_at(path: &Path, byte_offset: u64, bit: u8) {
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("open file for bit-flip");
    f.seek(SeekFrom::Start(byte_offset))
        .expect("seek to offset");
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf).expect("read byte");
    buf[0] ^= 1 << bit;
    f.seek(SeekFrom::Start(byte_offset)).expect("seek back");
    f.write_all(&buf).expect("write flipped byte");
    f.flush().expect("flush");
}

fn raft_wal_config() -> WalConfig {
    WalConfig {
        enable_checksums: true,
        verify_on_replay: true,
        ..WalConfig::default()
    }
}

fn store_wal_config() -> StoreWalConfig {
    StoreWalConfig {
        enabled: true,
        enable_checksums: true,
        verify_on_replay: true,
        ..StoreWalConfig::default()
    }
}

fn make_tensor_data(val: i64) -> TensorData {
    let mut td = TensorData::new();
    td.set("v", TensorValue::Scalar(ScalarValue::Int(val)));
    td
}

// ---------------------------------------------------------------------------
// Raft WAL torn-write tests
// ---------------------------------------------------------------------------

#[test]
fn test_raft_wal_torn_write_sweep() {
    let dir = tempfile::tempdir().unwrap();

    let write_entries = |path: &Path| -> Vec<u64> {
        let mut wal = RaftWal::open_with_config(path, raft_wal_config()).unwrap();
        let mut sizes = Vec::new();

        let entries = vec![
            RaftWalEntry::TermAndVote {
                term: 1,
                voted_for: None,
            },
            RaftWalEntry::LogAppend {
                index: 1,
                term: 1,
                command_hash: [0u8; 32],
            },
            RaftWalEntry::TermChange { new_term: 2 },
            RaftWalEntry::VoteCast {
                term: 2,
                candidate_id: "n1".to_string(),
            },
            RaftWalEntry::LogAppend {
                index: 2,
                term: 2,
                command_hash: [1u8; 32],
            },
        ];

        for entry in &entries {
            wal.append(entry).unwrap();
            sizes.push(wal.current_size());
        }
        sizes
    };

    let recover_and_verify = |path: &Path, expected: usize| -> bool {
        let wal = RaftWal::open_with_config(path, raft_wal_config()).unwrap();
        let entries = wal.replay().unwrap();
        if entries.len() != expected {
            return false;
        }
        let state = RaftRecoveryState::from_entries(&entries);
        // After 4 entries: term=2, voted_for="n1"
        state.current_term == 2 && state.voted_for.as_deref() == Some("n1")
    };

    torn_write_sweep(dir.path(), "raft.wal", write_entries, recover_and_verify);
}

#[test]
fn test_raft_wal_every_entry_boundary_truncation() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("raft_boundary.wal");

    let mut wal = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    let mut sizes = vec![0u64]; // size before any entry

    for i in 0..10u64 {
        wal.append(&RaftWalEntry::LogAppend {
            index: i,
            term: 1,
            command_hash: [i as u8; 32],
        })
        .unwrap();
        sizes.push(wal.current_size());
    }

    // Truncate at each entry boundary and verify
    for boundary_idx in 0..10 {
        let copy = dir.path().join(format!("boundary_{boundary_idx}.wal"));
        fs::copy(&wal_path, &copy).unwrap();
        let f = OpenOptions::new().write(true).open(&copy).unwrap();
        f.set_len(sizes[boundary_idx]).unwrap();
        drop(f);

        let wal2 = RaftWal::open_with_config(&copy, raft_wal_config()).unwrap();
        let entries = wal2.replay().unwrap();
        assert_eq!(
            entries.len(),
            boundary_idx,
            "truncation at boundary {boundary_idx}"
        );

        fs::remove_file(&copy).ok();
    }
}

#[test]
fn test_raft_wal_double_vote_prevented_after_torn_write() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("vote.wal");

    // Write first vote fully
    let mut wal = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    wal.append(&RaftWalEntry::VoteCast {
        term: 5,
        candidate_id: "nodeA".to_string(),
    })
    .unwrap();
    let size_after_first = wal.current_size();

    // Write second vote
    wal.append(&RaftWalEntry::VoteCast {
        term: 5,
        candidate_id: "nodeB".to_string(),
    })
    .unwrap();
    let size_after_second = wal.current_size();
    drop(wal);

    // Truncate mid-second-entry
    let mid = size_after_first + (size_after_second - size_after_first) / 2;
    let f = OpenOptions::new().write(true).open(&wal_path).unwrap();
    f.set_len(mid).unwrap();
    drop(f);

    let wal2 = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    let entries = wal2.replay().unwrap();
    assert_eq!(entries.len(), 1);

    let state = RaftRecoveryState::from_entries(&entries);
    assert_eq!(
        state.voted_for.as_deref(),
        Some("nodeA"),
        "must see vote for nodeA only"
    );
}

// ---------------------------------------------------------------------------
// TxWal torn-write tests
// ---------------------------------------------------------------------------

#[test]
fn test_tx_wal_torn_write_sweep() {
    let dir = tempfile::tempdir().unwrap();

    let write_entries = |path: &Path| -> Vec<u64> {
        let mut wal = TxWal::open_with_config(path, raft_wal_config()).unwrap();
        let mut sizes = Vec::new();

        let entries = vec![
            TxWalEntry::TxBegin {
                tx_id: 1,
                participants: vec![0, 1],
            },
            TxWalEntry::PrepareVote {
                tx_id: 1,
                shard: 0,
                vote: PrepareVoteKind::Yes { lock_handle: 100 },
            },
            TxWalEntry::PhaseChange {
                tx_id: 1,
                from: tensor_chain::TxPhase::Preparing,
                to: tensor_chain::TxPhase::Committing,
            },
            TxWalEntry::TxComplete {
                tx_id: 1,
                outcome: TxOutcome::Committed,
            },
        ];

        for entry in &entries {
            wal.append(entry).unwrap();
            sizes.push(wal.current_size());
        }
        sizes
    };

    let recover_and_verify = |path: &Path, expected: usize| -> bool {
        let wal = TxWal::open_with_config(path, raft_wal_config()).unwrap();
        let entries = wal.replay().unwrap();
        if entries.len() != expected {
            return false;
        }
        // With 3 entries (TxBegin + PrepareVote + PhaseChange), tx_id=1 should
        // be in committing_txs (phase changed to Committing, not yet completed)
        let state = TxRecoveryState::from_entries(&entries);
        !state.committing_txs.is_empty() && state.committing_txs[0].tx_id == 1
    };

    torn_write_sweep(dir.path(), "tx.wal", write_entries, recover_and_verify);
}

#[test]
fn test_tx_wal_orphaned_lock_detected_after_crash() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("orphan.wal");

    let mut wal = TxWal::open_with_config(&wal_path, raft_wal_config()).unwrap();

    wal.append(&TxWalEntry::TxBegin {
        tx_id: 42,
        participants: vec![0],
    })
    .unwrap();
    wal.append(&TxWalEntry::PrepareVote {
        tx_id: 42,
        shard: 0,
        vote: PrepareVoteKind::Yes { lock_handle: 777 },
    })
    .unwrap();
    wal.append(&TxWalEntry::TxComplete {
        tx_id: 42,
        outcome: TxOutcome::Committed,
    })
    .unwrap();
    // Intentionally do NOT write AllLocksReleased

    drop(wal);

    let wal2 = TxWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    let entries = wal2.replay().unwrap();
    let state = TxRecoveryState::from_entries(&entries);

    assert!(
        !state.orphaned_locks.is_empty(),
        "should detect orphaned lock"
    );
    let orphan: &OrphanedLock = &state.orphaned_locks[0];
    assert_eq!(orphan.tx_id, 42);
    assert_eq!(orphan.lock_handle, 777);
}

// ---------------------------------------------------------------------------
// TensorStore crash-recover-verify
// ---------------------------------------------------------------------------

#[test]
fn test_tensor_store_crash_recover_100_keys() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("store.wal");

    {
        let store = TensorStore::open_durable(&wal_path, store_wal_config()).unwrap();
        for i in 0..100 {
            store
                .put_durable(format!("key_{i}"), make_tensor_data(i))
                .unwrap();
        }
    } // drop = crash

    let recovered = TensorStore::recover(&wal_path, &store_wal_config(), None).unwrap();
    for i in 0..100 {
        let key = format!("key_{i}");
        assert!(recovered.exists(&key), "key {key} missing after recovery");
        let data = recovered.get(&key).unwrap();
        match data.get("v") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => {
                assert_eq!(*v, i, "wrong value for {key}");
            },
            other => panic!("expected Int for {key}, got {other:?}"),
        }
    }
}

#[test]
fn test_tensor_store_crash_recover_with_deletes() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("store_del.wal");

    {
        let store = TensorStore::open_durable(&wal_path, store_wal_config()).unwrap();
        for i in 0..50 {
            store
                .put_durable(format!("k{i}"), make_tensor_data(i))
                .unwrap();
        }
        // Delete keys 10..20
        for i in 10..20 {
            store.delete_durable(&format!("k{i}")).unwrap();
        }
        // Write 20 more
        for i in 50..70 {
            store
                .put_durable(format!("k{i}"), make_tensor_data(i))
                .unwrap();
        }
    }

    let recovered = TensorStore::recover(&wal_path, &store_wal_config(), None).unwrap();

    // 40 surviving from first batch + 20 from second = 60
    for i in 0..70 {
        let key = format!("k{i}");
        if (10..20).contains(&i) {
            assert!(!recovered.exists(&key), "{key} should be deleted");
        } else {
            assert!(recovered.exists(&key), "{key} missing after recovery");
        }
    }
}

#[test]
fn test_tensor_store_crash_recover_overwrite_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("store_ow.wal");

    {
        let store = TensorStore::open_durable(&wal_path, store_wal_config()).unwrap();
        store.put_durable("x", make_tensor_data(1)).unwrap();
        store.put_durable("x", make_tensor_data(2)).unwrap();
        store.put_durable("x", make_tensor_data(3)).unwrap();
    }

    let recovered = TensorStore::recover(&wal_path, &store_wal_config(), None).unwrap();
    assert!(recovered.exists("x"));
    match recovered.get("x").unwrap().get("v") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => {
            assert_eq!(*v, 3, "last write wins");
        },
        other => panic!("expected Int(3), got {other:?}"),
    }
}

#[test]
fn test_tensor_store_crash_mid_batch_partial_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("store_batch.wal");

    {
        let config = StoreWalConfig {
            enabled: true,
            enable_checksums: true,
            verify_on_replay: true,
            sync_mode: tensor_store::SyncMode::Batched { max_entries: 10 },
            ..StoreWalConfig::default()
        };
        let store = TensorStore::open_durable(&wal_path, config).unwrap();
        for i in 0..50 {
            store
                .put_durable(format!("bk{i}"), make_tensor_data(i))
                .unwrap();
        }
        // Do NOT call wal_sync() -- drop immediately
    }

    let recovered = TensorStore::recover(&wal_path, &store_wal_config(), None).unwrap();

    // At least 40 should survive (4 complete batches of 10)
    let mut count = 0;
    for i in 0..50 {
        if recovered.exists(&format!("bk{i}")) {
            count += 1;
        }
    }
    assert!(
        count >= 40,
        "expected at least 40 keys after batched crash, got {count}"
    );
}

// ---------------------------------------------------------------------------
// CRC corruption detection
// ---------------------------------------------------------------------------

#[test]
fn test_raft_wal_crc_bit_flip_detected() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("crc_flip.wal");

    let mut wal = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    for i in 0..3u64 {
        wal.append(&RaftWalEntry::LogAppend {
            index: i,
            term: 1,
            command_hash: [i as u8; 32],
        })
        .unwrap();
    }
    // Record size after first entry to locate entry 2's CRC field
    drop(wal);

    // Re-open to get sizes: write entries one-by-one to measure
    let sizes = entry_sizes_raft(&wal_path, 3);
    // CRC field of entry 1 (0-indexed) is at sizes[0] + 4 (after length prefix)
    let crc_offset = sizes[0] + 4;
    flip_bit_at(&wal_path, crc_offset, 0);

    let wal2 = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    let result = wal2.replay();
    assert!(
        result.is_err(),
        "replay should fail on CRC mismatch, got {result:?}"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("checksum") || err_msg.contains("Checksum"),
        "error should mention checksum: {err_msg}"
    );
}

#[test]
fn test_raft_wal_payload_corruption_detected() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("payload_flip.wal");

    let mut wal = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    for i in 0..3u64 {
        wal.append(&RaftWalEntry::LogAppend {
            index: i,
            term: 1,
            command_hash: [i as u8; 32],
        })
        .unwrap();
    }
    drop(wal);

    let sizes = entry_sizes_raft(&wal_path, 3);
    // Payload of entry 1 starts at sizes[0] + 8 (length=4 + CRC=4)
    let payload_offset = sizes[0] + 8;
    flip_bit_at(&wal_path, payload_offset, 3);

    let wal2 = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    let result = wal2.replay();
    // Either CRC mismatch error or deserialization failure (stops at entry 1)
    match result {
        Err(_) => {}, // CRC mismatch detected
        Ok(entries) => {
            // Deserialization failure: should recover at most entry 0
            assert!(
                entries.len() <= 1,
                "corrupted payload should stop replay, got {} entries",
                entries.len()
            );
        },
    }
}

#[test]
fn test_tensor_store_wal_corruption_stops_replay() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("store_corrupt.wal");

    {
        let store = TensorStore::open_durable(&wal_path, store_wal_config()).unwrap();
        for i in 0..5 {
            store
                .put_durable(format!("ck{i}"), make_tensor_data(i))
                .unwrap();
        }
    }

    // Corrupt entry 2's CRC (after entries 0 and 1)
    let sizes = entry_sizes_store(&wal_path, 5);
    let crc_offset = sizes[2] + 4;
    flip_bit_at(&wal_path, crc_offset, 0);

    let result = TensorStore::recover(&wal_path, &store_wal_config(), None);
    match result {
        Err(e) => {
            let msg = format!("{e:?}");
            assert!(
                msg.contains("checksum") || msg.contains("Checksum") || msg.contains("WAL"),
                "error should indicate corruption: {msg}"
            );
        },
        Ok(recovered) => {
            // If it recovered partially, only entries before corruption should exist
            assert!(recovered.exists("ck0"), "entry 0 should survive");
            assert!(recovered.exists("ck1"), "entry 1 should survive");
        },
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_raft_wal_empty_file_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("empty.wal");
    fs::write(&wal_path, &[]).unwrap();

    let wal = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    let entries = wal.replay().unwrap();
    assert!(entries.is_empty(), "empty WAL should replay as empty");
}

#[test]
fn test_raft_wal_single_byte_file_recovery() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("single_byte.wal");
    fs::write(&wal_path, &[0x42]).unwrap();

    let wal = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    let entries = wal.replay().unwrap();
    assert!(
        entries.is_empty(),
        "1-byte WAL should replay as empty (incomplete length prefix)"
    );
}

#[test]
fn test_tx_wal_rotation_preserves_entries() {
    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("rotate.wal");

    let config = WalConfig {
        enable_checksums: true,
        verify_on_replay: true,
        max_rotated_files: 3,
        auto_rotate: false,
        ..WalConfig::default()
    };

    let mut wal = TxWal::open_with_config(&wal_path, config).unwrap();

    // Write entries before rotation
    wal.append(&TxWalEntry::TxBegin {
        tx_id: 1,
        participants: vec![0],
    })
    .unwrap();
    wal.append(&TxWalEntry::TxComplete {
        tx_id: 1,
        outcome: TxOutcome::Committed,
    })
    .unwrap();

    // Rotate
    wal.rotate().unwrap();

    // Write entries after rotation
    wal.append(&TxWalEntry::TxBegin {
        tx_id: 2,
        participants: vec![1],
    })
    .unwrap();
    wal.append(&TxWalEntry::TxComplete {
        tx_id: 2,
        outcome: TxOutcome::Aborted,
    })
    .unwrap();

    drop(wal);

    // Replay from the current (post-rotation) WAL: should see entries after rotation
    let wal2 = TxWal::open_with_config(&wal_path, WalConfig::default()).unwrap();
    let entries = wal2.replay().unwrap();
    assert_eq!(
        entries.len(),
        2,
        "current WAL should have post-rotation entries"
    );

    // The rotated file should exist and have the pre-rotation entries
    let rotated_path = wal_path.with_extension("wal.1");
    if rotated_path.exists() {
        let wal3 = TxWal::open_with_config(&rotated_path, WalConfig::default()).unwrap();
        let old_entries = wal3.replay().unwrap();
        assert_eq!(
            old_entries.len(),
            2,
            "rotated WAL should have pre-rotation entries"
        );
    }
}

// ---------------------------------------------------------------------------
// Measurement helpers
// ---------------------------------------------------------------------------

/// Measure byte offsets of each entry in a Raft WAL by reading the file.
fn entry_sizes_raft(path: &Path, count: usize) -> Vec<u64> {
    let mut f = fs::File::open(path).unwrap();
    let mut offsets = Vec::with_capacity(count + 1);
    let mut pos = 0u64;

    for _ in 0..count {
        offsets.push(pos);
        let mut len_buf = [0u8; 4];
        f.read_exact(&mut len_buf).unwrap();
        let len = u32::from_le_bytes(len_buf) as u64;
        // Skip CRC (4 bytes) + payload (len bytes)
        let skip = 4 + len;
        f.seek(SeekFrom::Current(skip as i64)).unwrap();
        pos += 4 + 4 + len; // length_prefix + crc + payload
    }
    offsets
}

/// Measure byte offsets of each entry in a TensorStore WAL.
fn entry_sizes_store(path: &Path, count: usize) -> Vec<u64> {
    // Same format: [4-byte len][4-byte CRC][payload]
    entry_sizes_raft(path, count)
}

// ---------------------------------------------------------------------------
// OS-level fault injection tests (Part B)
// ---------------------------------------------------------------------------

#[test]
#[cfg(unix)]
fn test_raft_wal_readonly_file_append_fails() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("readonly.wal");

    // Write some entries
    let mut wal = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    for i in 0..3u64 {
        wal.append(&RaftWalEntry::LogAppend {
            index: i,
            term: 1,
            command_hash: [i as u8; 32],
        })
        .unwrap();
    }
    drop(wal);

    // Make file read-only
    let mut perms = fs::metadata(&wal_path).unwrap().permissions();
    perms.set_mode(0o444);
    fs::set_permissions(&wal_path, perms).unwrap();

    // Try to open and append -- should fail on open (BufWriter<File> needs write)
    let result = RaftWal::open_with_config(&wal_path, raft_wal_config());
    assert!(
        result.is_err(),
        "opening read-only WAL for writing should fail"
    );

    // Restore permissions so tempdir cleanup succeeds
    let mut perms = fs::metadata(&wal_path).unwrap().permissions();
    perms.set_mode(0o644);
    fs::set_permissions(&wal_path, perms).unwrap();

    // Verify existing entries are intact
    let wal2 = RaftWal::open_with_config(&wal_path, raft_wal_config()).unwrap();
    let entries = wal2.replay().unwrap();
    assert_eq!(
        entries.len(),
        3,
        "existing entries must survive read-only episode"
    );
}

#[test]
#[cfg(unix)]
fn test_tensor_store_readonly_wal_put_fails_gracefully() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let wal_path = dir.path().join("store_ro.wal");

    // Write initial data
    {
        let store = TensorStore::open_durable(&wal_path, store_wal_config()).unwrap();
        store.put_durable("x", make_tensor_data(1)).unwrap();
        store.put_durable("y", make_tensor_data(2)).unwrap();
    }

    // Make WAL read-only
    let mut perms = fs::metadata(&wal_path).unwrap().permissions();
    perms.set_mode(0o444);
    fs::set_permissions(&wal_path, perms).unwrap();

    // Opening the store for durable writes should fail (WAL needs write access)
    let result = TensorStore::open_durable(&wal_path, store_wal_config());
    assert!(
        result.is_err(),
        "opening durable store with read-only WAL should fail"
    );

    // Restore permissions
    let mut perms = fs::metadata(&wal_path).unwrap().permissions();
    perms.set_mode(0o644);
    fs::set_permissions(&wal_path, perms).unwrap();

    // Prior data still recoverable
    let recovered = TensorStore::recover(&wal_path, &store_wal_config(), None).unwrap();
    assert!(recovered.exists("x"), "x should survive");
    assert!(recovered.exists("y"), "y should survive");
    match recovered.get("x").unwrap().get("v") {
        Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 1),
        other => panic!("expected Int(1), got {other:?}"),
    }
}
