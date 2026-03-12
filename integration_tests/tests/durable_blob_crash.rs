// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for DurableBlobLog crash recovery.
//!
//! Tests crash recovery at each fsync boundary:
//! - After WAL PREPARE fsync
//! - After data segment fsync
//! - After WAL COMMIT fsync
//! - After footer fsync
//! - After WAL SEAL fsync
//!
//! The core invariant being tested:
//! - Ack to caller only AFTER WAL COMMIT fsync
//! - Only COMMIT records are visible (reads, recovery, GC)
//! - PREPARE without COMMIT = invisible, GC candidate

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use tempfile::tempdir;
use tensor_store::{DurableBlobLog, DurableBlobLogConfig, DurableChunkHash};

fn test_config(dir: &Path) -> DurableBlobLogConfig {
    DurableBlobLogConfig {
        segment_dir: dir.to_path_buf(),
        segment_size: 4096,  // Small for testing
        enable_fsync: false, // We'll manually control fsync behavior
        cache_size: 100,
    }
}

// ============= Basic Recovery Tests =============

#[test]
fn test_recovery_with_empty_wal() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    // First open should create empty log
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        assert_eq!(log.chunk_count(), 0);
    }

    // Second open should work with empty WAL
    {
        let log = DurableBlobLog::open(config).unwrap();
        assert_eq!(log.chunk_count(), 0);
    }
}

#[test]
fn test_recovery_after_single_commit() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let hash;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        hash = log.append(b"test data for recovery").unwrap();
        log.sync().unwrap();
    }

    // Reopen and verify
    {
        let log = DurableBlobLog::open(config).unwrap();
        assert_eq!(log.chunk_count(), 1);
        let data = log.get(&hash).unwrap();
        assert_eq!(data, b"test data for recovery");
    }
}

#[test]
fn test_recovery_with_multiple_commits() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let mut hashes = Vec::new();
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        for i in 0..10 {
            let hash = log.append(format!("chunk {i}").as_bytes()).unwrap();
            hashes.push(hash);
        }
        log.sync().unwrap();
    }

    // Reopen and verify all chunks
    {
        let log = DurableBlobLog::open(config).unwrap();
        assert_eq!(log.chunk_count(), 10);
        for (i, hash) in hashes.iter().enumerate() {
            let data = log.get(hash).unwrap();
            assert_eq!(data, format!("chunk {i}").as_bytes());
        }
    }
}

#[test]
fn test_recovery_with_tombstones() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let hash1;
    let hash2;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        hash1 = log.append(b"to keep").unwrap();
        hash2 = log.append(b"to delete").unwrap();
        log.delete(&hash2).unwrap();
        log.sync().unwrap();
    }

    // Reopen and verify tombstone is preserved
    {
        let log = DurableBlobLog::open(config).unwrap();
        assert!(log.contains(&hash1));
        assert!(!log.contains(&hash2));

        let data = log.get(&hash1).unwrap();
        assert_eq!(data, b"to keep");

        let result = log.get(&hash2);
        assert!(result.is_err());
    }
}

// ============= Crash at PREPARE Tests =============

#[test]
fn test_crash_after_prepare_only() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    // Simulate crash after PREPARE is logged but before COMMIT
    // by writing only a PREPARE record to WAL
    {
        let wal_path = dir.path().join("blob.wal");
        fs::create_dir_all(dir.path()).unwrap();

        // Write a PREPARE record manually
        let hash = DurableChunkHash::from_data(b"orphan data");
        let prepare_record = tensor_store::durable_blob_log::BlobWalRecord::Prepare {
            hash,
            segment_id: 1,
            offset: 0,
            len: 11,
        };
        let bytes = bitcode::serialize(&prepare_record).unwrap();
        let len = bytes.len() as u32;
        let crc = crc32fast::hash(&bytes);

        let mut file = File::create(&wal_path).unwrap();
        file.write_all(&len.to_le_bytes()).unwrap();
        file.write_all(&crc.to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
    }

    // Recovery should see PREPARE without COMMIT as GC candidate
    {
        let log = DurableBlobLog::open(config).unwrap();
        assert_eq!(log.chunk_count(), 0, "PREPARE-only should not be visible");

        let gc_candidates = log.gc_candidates();
        assert_eq!(gc_candidates.len(), 1, "Should have one GC candidate");

        // The orphan should not be retrievable
        let hash = DurableChunkHash::from_data(b"orphan data");
        assert!(!log.contains(&hash));
    }
}

#[test]
fn test_crash_after_prepare_data_written_no_commit() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    // Simulate: PREPARE logged, data written to segment, but COMMIT not logged
    {
        fs::create_dir_all(dir.path()).unwrap();

        // Write data to segment file
        let segment_path = dir.path().join("segment_00000001.bin");
        let data = b"orphan chunk data";
        let mut segment_file = File::create(&segment_path).unwrap();
        segment_file.write_all(data).unwrap();
        segment_file.sync_all().unwrap();

        // Write PREPARE to WAL (but no COMMIT)
        let wal_path = dir.path().join("blob.wal");
        let hash = DurableChunkHash::from_data(data);
        let prepare_record = tensor_store::durable_blob_log::BlobWalRecord::Prepare {
            hash,
            segment_id: 1,
            offset: 0,
            len: data.len() as u32,
        };
        let bytes = bitcode::serialize(&prepare_record).unwrap();
        let len = bytes.len() as u32;
        let crc = crc32fast::hash(&bytes);

        let mut wal_file = File::create(&wal_path).unwrap();
        wal_file.write_all(&len.to_le_bytes()).unwrap();
        wal_file.write_all(&crc.to_le_bytes()).unwrap();
        wal_file.write_all(&bytes).unwrap();
        wal_file.sync_all().unwrap();
    }

    // Recovery: data exists on disk but no COMMIT, so invisible
    {
        let log = DurableBlobLog::open(config).unwrap();
        let hash = DurableChunkHash::from_data(b"orphan chunk data");
        assert!(
            !log.contains(&hash),
            "Data without COMMIT should be invisible"
        );
        assert_eq!(log.gc_candidates().len(), 1, "Should be GC candidate");
    }
}

// ============= Crash at COMMIT Tests =============

#[test]
fn test_full_commit_cycle_recovery() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let hash;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        hash = log.append(b"fully committed").unwrap();
        // Don't call sync - simulate crash after in-memory ack
    }

    // Recovery should find the committed chunk
    // (because append() calls sync_wal after COMMIT)
    {
        let log = DurableBlobLog::open(config).unwrap();
        assert!(log.contains(&hash), "Committed chunk should survive");
        let data = log.get(&hash).unwrap();
        assert_eq!(data, b"fully committed");
    }
}

// ============= WAL Corruption Tests =============

#[test]
fn test_recovery_with_truncated_wal() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    // Write valid data first
    let hash;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        hash = log.append(b"before corruption").unwrap();
        log.sync().unwrap();
    }

    // Append garbage to WAL
    {
        let wal_path = dir.path().join("blob.wal");
        let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
        file.write_all(&[0xFF, 0xFE, 0xFD, 0xFC]).unwrap();
        file.sync_all().unwrap();
    }

    // Recovery should skip corrupted tail and preserve valid data
    {
        let log = DurableBlobLog::open(config).unwrap();
        assert!(log.contains(&hash), "Valid data should survive corruption");
        let data = log.get(&hash).unwrap();
        assert_eq!(data, b"before corruption");
    }
}

#[test]
fn test_recovery_with_crc_mismatch() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    // Write valid data first
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        let _hash = log.append(b"valid data").unwrap();
        log.sync().unwrap();
    }

    // Corrupt a byte in the last record (flip CRC)
    {
        let wal_path = dir.path().join("blob.wal");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&wal_path)
            .unwrap();

        // Corrupt last byte (part of CRC or payload)
        file.seek(SeekFrom::End(-1)).unwrap();
        let mut byte = [0u8];
        file.read_exact(&mut byte).unwrap();
        byte[0] ^= 0xFF; // Flip all bits
        file.seek(SeekFrom::End(-1)).unwrap();
        file.write_all(&byte).unwrap();
        file.sync_all().unwrap();
    }

    // Recovery should skip the corrupted record
    // (This might mean the last commit is lost, depending on what was corrupted)
    // Just verify recovery doesn't panic
    {
        let _log = DurableBlobLog::open(config).unwrap();
        // Recovery completes without panic - test passes
    }
}

// ============= Segment Sealing Crash Tests =============

#[test]
fn test_recovery_after_segment_seal() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.segment_size = 256; // Very small to trigger sealing

    let hashes: Vec<DurableChunkHash>;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        hashes = (0..20)
            .map(|i| {
                let data = format!("chunk data with padding {i:04}");
                log.append(data.as_bytes()).unwrap()
            })
            .collect();
        log.sync().unwrap();
    }

    // Verify segment was sealed
    let segment_count;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        segment_count = log.segment_count();
        assert!(
            segment_count >= 2,
            "Should have sealed at least one segment"
        );

        // All chunks should be recoverable
        for (i, hash) in hashes.iter().enumerate() {
            let data = log.get(hash).unwrap();
            assert_eq!(data, format!("chunk data with padding {i:04}").as_bytes());
        }
    }
}

#[test]
fn test_crash_during_segment_seal() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    // Write some data that gets committed
    let committed_hash;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        committed_hash = log.append(b"committed before seal crash").unwrap();
        log.sync().unwrap();
    }

    // Simulate partial SEAL by writing SEAL record without valid footer
    {
        let wal_path = dir.path().join("blob.wal");
        let seal_record = tensor_store::durable_blob_log::BlobWalRecord::Seal {
            segment_id: 1,
            footer_offset: 12345, // Bogus offset
        };
        let bytes = bitcode::serialize(&seal_record).unwrap();
        let len = bytes.len() as u32;
        let crc = crc32fast::hash(&bytes);

        let mut file = OpenOptions::new().append(true).open(&wal_path).unwrap();
        file.write_all(&len.to_le_bytes()).unwrap();
        file.write_all(&crc.to_le_bytes()).unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
    }

    // Recovery should handle invalid SEAL gracefully
    {
        let log = DurableBlobLog::open(config).unwrap();
        // The committed chunk should still be accessible
        assert!(log.contains(&committed_hash));
        let data = log.get(&committed_hash).unwrap();
        assert_eq!(data, b"committed before seal crash");
    }
}

// ============= Deduplication Across Recovery Tests =============

#[test]
fn test_deduplication_preserved_across_recovery() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let data = b"deduplicated content";
    let hash;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        hash = log.append(data).unwrap();
        let hash2 = log.append(data).unwrap();
        assert_eq!(hash, hash2);
        assert_eq!(log.chunk_count(), 1);
        log.sync().unwrap();
    }

    // After recovery, dedup should still work
    {
        let log = DurableBlobLog::open(config).unwrap();
        let hash3 = log.append(data).unwrap();
        assert_eq!(hash, hash3);
        assert_eq!(log.chunk_count(), 1); // Still just one chunk
    }
}

// ============= Large Data Recovery Tests =============

#[test]
fn test_recovery_with_large_chunks() {
    let dir = tempdir().unwrap();
    let mut config = test_config(dir.path());
    config.segment_size = 1024 * 1024; // 1MB segments

    let large_data: Vec<u8> = (0..65536).map(|i| (i % 256) as u8).collect();
    let hash;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        hash = log.append(&large_data).unwrap();
        log.sync().unwrap();
    }

    // Recovery should preserve large chunks correctly
    {
        let log = DurableBlobLog::open(config).unwrap();
        let recovered = log.get(&hash).unwrap();
        assert_eq!(recovered.len(), large_data.len());
        assert_eq!(recovered, large_data);
    }
}

// ============= Mixed Operations Recovery Tests =============

#[test]
fn test_recovery_with_mixed_operations() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    let hash1;
    let hash2;
    let hash3;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();

        // Add three chunks
        hash1 = log.append(b"chunk one").unwrap();
        hash2 = log.append(b"chunk two").unwrap();
        hash3 = log.append(b"chunk three").unwrap();

        // Delete middle chunk
        log.delete(&hash2).unwrap();

        // Add duplicate of first
        let hash1_dup = log.append(b"chunk one").unwrap();
        assert_eq!(hash1, hash1_dup);

        log.sync().unwrap();
    }

    // Verify state after recovery
    {
        let log = DurableBlobLog::open(config).unwrap();

        assert!(log.contains(&hash1), "chunk one should exist");
        assert!(!log.contains(&hash2), "chunk two should be deleted");
        assert!(log.contains(&hash3), "chunk three should exist");

        assert_eq!(log.get(&hash1).unwrap(), b"chunk one");
        assert!(log.get(&hash2).is_err());
        assert_eq!(log.get(&hash3).unwrap(), b"chunk three");
    }
}

// ============= Concurrent Recovery Tests =============

#[test]
fn test_recovery_with_many_chunks() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    const NUM_CHUNKS: usize = 100;
    let hashes: Vec<DurableChunkHash>;
    {
        let log = DurableBlobLog::open(config.clone()).unwrap();
        hashes = (0..NUM_CHUNKS)
            .map(|i| {
                let data = format!("chunk number {i:06}");
                log.append(data.as_bytes()).unwrap()
            })
            .collect();
        log.sync().unwrap();
    }

    // Verify all chunks after recovery
    {
        let log = DurableBlobLog::open(config).unwrap();
        assert_eq!(log.chunk_count(), NUM_CHUNKS as u64);

        for (i, hash) in hashes.iter().enumerate() {
            let data = log.get(hash).unwrap();
            assert_eq!(data, format!("chunk number {i:06}").as_bytes());
        }
    }
}

// ============= Invariant Tests =============

#[test]
fn test_durability_invariant_visibility() {
    // Core invariant: Only COMMIT records are visible
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    {
        let log = DurableBlobLog::open(config.clone()).unwrap();

        // Write and commit
        let committed = log.append(b"committed").unwrap();

        // Verify committed is visible immediately
        assert!(log.contains(&committed));

        // Write something that we then delete before "crash"
        let deleted = log.append(b"to delete").unwrap();
        log.delete(&deleted).unwrap();

        // Deleted should not be visible even before "crash"
        assert!(!log.contains(&deleted));

        log.sync().unwrap();
    }

    // After "crash" and recovery
    {
        let log = DurableBlobLog::open(config).unwrap();

        // Committed should still be visible
        let committed = DurableChunkHash::from_data(b"committed");
        assert!(log.contains(&committed));

        // Deleted should still not be visible
        let deleted = DurableChunkHash::from_data(b"to delete");
        assert!(!log.contains(&deleted));
    }
}

#[test]
fn test_gc_candidates_are_prepare_without_commit() {
    let dir = tempdir().unwrap();
    let config = test_config(dir.path());

    // Manually create WAL with orphaned PREPARE records
    {
        fs::create_dir_all(dir.path()).unwrap();
        let wal_path = dir.path().join("blob.wal");

        // Write PREPARE for chunk A (no COMMIT)
        let hash_a = DurableChunkHash::from_data(b"orphan A");
        let prepare_a = tensor_store::durable_blob_log::BlobWalRecord::Prepare {
            hash: hash_a,
            segment_id: 1,
            offset: 0,
            len: 8,
        };

        // Write PREPARE + COMMIT for chunk B (complete)
        let hash_b = DurableChunkHash::from_data(b"committed B");
        let prepare_b = tensor_store::durable_blob_log::BlobWalRecord::Prepare {
            hash: hash_b,
            segment_id: 1,
            offset: 100,
            len: 11,
        };
        let commit_b = tensor_store::durable_blob_log::BlobWalRecord::Commit {
            hash: hash_b,
            segment_id: 1,
            offset: 100,
            len: 11,
            crc32: crc32fast::hash(b"committed B"),
        };

        // Write PREPARE for chunk C (no COMMIT)
        let hash_c = DurableChunkHash::from_data(b"orphan C");
        let prepare_c = tensor_store::durable_blob_log::BlobWalRecord::Prepare {
            hash: hash_c,
            segment_id: 1,
            offset: 200,
            len: 8,
        };

        let mut file = File::create(&wal_path).unwrap();

        for record in &[prepare_a, prepare_b, commit_b, prepare_c] {
            let bytes = bitcode::serialize(record).unwrap();
            let len = bytes.len() as u32;
            let crc = crc32fast::hash(&bytes);
            file.write_all(&len.to_le_bytes()).unwrap();
            file.write_all(&crc.to_le_bytes()).unwrap();
            file.write_all(&bytes).unwrap();
        }
        file.sync_all().unwrap();

        // Also write segment file with committed B's data
        let segment_path = dir.path().join("segment_00000001.bin");
        let mut segment = File::create(&segment_path).unwrap();
        segment.seek(SeekFrom::Start(100)).unwrap();
        segment.write_all(b"committed B").unwrap();
        segment.sync_all().unwrap();
    }

    // Recovery should identify orphans as GC candidates
    {
        let log = DurableBlobLog::open(config).unwrap();

        let gc = log.gc_candidates();
        assert_eq!(gc.len(), 2, "Should have two GC candidates (A and C)");

        let hash_a = DurableChunkHash::from_data(b"orphan A");
        let hash_b = DurableChunkHash::from_data(b"committed B");
        let hash_c = DurableChunkHash::from_data(b"orphan C");

        assert!(gc.contains(&hash_a), "A should be GC candidate");
        assert!(!gc.contains(&hash_b), "B should NOT be GC candidate");
        assert!(gc.contains(&hash_c), "C should be GC candidate");

        // Only B should be visible
        assert!(!log.contains(&hash_a));
        assert!(log.contains(&hash_b));
        assert!(!log.contains(&hash_c));
    }
}
