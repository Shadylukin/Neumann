// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::path::PathBuf;
use tensor_chain::{
    snapshot_buffer::{SnapshotBuffer, SnapshotBufferConfig},
    snapshot_streaming::{deserialize_entries, serialize_entries, SnapshotReader},
    Block, BlockHeader, LogEntry,
};

#[derive(Arbitrary, Debug)]
struct BufferInput {
    test_case: TestCase,
}

#[derive(Arbitrary, Debug)]
enum TestCase {
    /// Test buffer write-then-read roundtrip
    BufferRoundtrip {
        chunks: Vec<Vec<u8>>,
        max_memory: u16,
    },
    /// Test random chunk access
    ChunkAccess {
        data: Vec<u8>,
        offsets: Vec<u16>,
        max_memory: u16,
    },
    /// Test streaming writer/reader roundtrip
    StreamingRoundtrip { entry_count: u8, term: u8 },
    /// Test hash consistency
    HashConsistency {
        chunks: Vec<Vec<u8>>,
        max_memory: u16,
    },
    /// Test spill threshold behavior
    SpillThreshold {
        chunk_sizes: Vec<u16>,
        max_memory: u16,
    },
    /// Test deserialize with arbitrary bytes
    DeserializeArbitrary { data: Vec<u8> },
}

fn temp_dir() -> PathBuf {
    std::env::temp_dir().join("snapshot_buffer_fuzz")
}

fn make_config(max_memory: u16) -> SnapshotBufferConfig {
    // Clamp to reasonable range (1KB - 1MB)
    let max_mem = ((max_memory as usize) * 64).clamp(1024, 1024 * 1024);
    SnapshotBufferConfig {
        max_memory_bytes: max_mem,
        temp_dir: temp_dir(),
        initial_file_capacity: 4096,
    }
}

fn create_log_entry(index: u64, term: u64) -> LogEntry {
    let header = BlockHeader::new(index, [0u8; 32], [0u8; 32], [0u8; 32], "fuzz".to_string());
    let block = Block::new(header, vec![]);
    LogEntry::new(term, index, block)
}

fuzz_target!(|input: BufferInput| {
    match input.test_case {
        TestCase::BufferRoundtrip { chunks, max_memory } => {
            let config = make_config(max_memory);

            // Limit total data size
            let total_size: usize = chunks.iter().map(|c| c.len()).sum();
            if total_size > 10 * 1024 * 1024 {
                return;
            }

            let mut buffer = match SnapshotBuffer::new(config) {
                Ok(b) => b,
                Err(_) => return,
            };

            // Write all chunks
            let mut expected_data = Vec::new();
            for chunk in &chunks {
                if buffer.write(chunk).is_err() {
                    return;
                }
                expected_data.extend_from_slice(chunk);
            }

            if buffer.finalize().is_err() {
                return;
            }

            // Verify length
            assert_eq!(buffer.total_len() as usize, expected_data.len());

            // Verify content via as_bytes
            if let Ok(actual) = buffer.as_bytes() {
                assert_eq!(actual, &expected_data[..]);
            }

            // Cleanup
            let _ = buffer.cleanup();
        },

        TestCase::ChunkAccess {
            data,
            offsets,
            max_memory,
        } => {
            if data.is_empty() || data.len() > 1024 * 1024 {
                return;
            }

            let config = make_config(max_memory);

            let mut buffer = match SnapshotBuffer::new(config) {
                Ok(b) => b,
                Err(_) => return,
            };

            if buffer.write(&data).is_err() {
                return;
            }

            if buffer.finalize().is_err() {
                return;
            }

            // Test random chunk access
            for offset in offsets.iter().take(10) {
                let offset = (*offset as u64) % buffer.total_len();
                let len = ((buffer.total_len() - offset) as usize).min(100);

                if let Ok(slice) = buffer.as_slice(offset, len) {
                    // Verify it matches expected data
                    assert_eq!(slice, &data[offset as usize..offset as usize + len]);
                }
            }

            let _ = buffer.cleanup();
        },

        TestCase::StreamingRoundtrip { entry_count, term } => {
            let count = (entry_count as usize).min(100);
            if count == 0 {
                return;
            }

            let term = term.max(1) as u64;

            // Create entries
            let entries: Vec<LogEntry> = (1..=count as u64)
                .map(|i| create_log_entry(i, term))
                .collect();

            // Serialize using streaming writer
            let buffer = match serialize_entries(&entries, make_config(1000)) {
                Ok(b) => b,
                Err(_) => return,
            };

            // Verify hash is non-zero
            let hash = buffer.hash();
            if count > 0 {
                assert_ne!(hash, [0u8; 32]);
            }

            // Deserialize and verify
            if let Ok(bytes) = buffer.as_bytes() {
                if let Ok(restored) = deserialize_entries(bytes) {
                    assert_eq!(restored.len(), count);
                    for (i, entry) in restored.iter().enumerate() {
                        assert_eq!(entry.index, (i + 1) as u64);
                        assert_eq!(entry.term, term);
                    }
                }
            }
        },

        TestCase::HashConsistency { chunks, max_memory } => {
            let config = make_config(max_memory);

            let total_size: usize = chunks.iter().map(|c| c.len()).sum();
            if total_size > 1024 * 1024 {
                return;
            }

            let mut buffer = match SnapshotBuffer::new(config) {
                Ok(b) => b,
                Err(_) => return,
            };

            for chunk in &chunks {
                if buffer.write(chunk).is_err() {
                    return;
                }
            }

            if buffer.finalize().is_err() {
                return;
            }

            // Get streaming hash
            let streaming_hash = buffer.hash();

            // Compute manual hash from full data
            if let Ok(bytes) = buffer.as_bytes() {
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(bytes);
                let manual_hash: [u8; 32] = hasher.finalize().into();

                assert_eq!(streaming_hash, manual_hash, "hash mismatch");
            }

            let _ = buffer.cleanup();
        },

        TestCase::SpillThreshold {
            chunk_sizes,
            max_memory,
        } => {
            let config = make_config(max_memory);
            let max_mem = config.max_memory_bytes;

            let mut buffer = match SnapshotBuffer::new(config) {
                Ok(b) => b,
                Err(_) => return,
            };

            let mut total_written = 0usize;
            for size in chunk_sizes.iter().take(50) {
                let size = (*size as usize).min(max_mem);
                let chunk = vec![0u8; size];

                if buffer.write(&chunk).is_err() {
                    break;
                }
                total_written += size;

                // Buffer should correctly track length
                assert_eq!(buffer.total_len() as usize, total_written);
            }

            let _ = buffer.finalize();
            let _ = buffer.cleanup();
        },

        TestCase::DeserializeArbitrary { data } => {
            if data.len() > 1024 * 1024 {
                return;
            }

            // Try to deserialize arbitrary data - should not panic
            let _ = deserialize_entries(&data);

            // Also try creating a reader from arbitrary data
            let config = make_config(1000);
            if let Ok(mut buffer) = SnapshotBuffer::new(config) {
                if buffer.write(&data).is_ok() && buffer.finalize().is_ok() {
                    // Try to create a reader - may fail but shouldn't panic
                    if let Ok(reader) = SnapshotReader::new(&buffer) {
                        // Try to iterate - may fail but shouldn't panic
                        for _ in reader.take(10) {}
                    }
                }
                let _ = buffer.cleanup();
            }
        },
    }
});
