// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for memory-efficient Raft snapshot streaming.
//!
//! Tests the SnapshotBuffer and streaming serialization/deserialization:
//! - Memory-bounded snapshot creation with file-backed overflow
//! - Streaming snapshot transfer with zero-copy chunk access
//! - Large snapshot handling without OOM
//! - Concurrent snapshot transfers
//! - Backwards compatibility with legacy format

use std::sync::Arc;

use tensor_chain::{
    snapshot_streaming::{deserialize_entries, serialize_entries, SnapshotReader, SnapshotWriter},
    Block, BlockHeader, LogEntry, MemoryTransport, RaftConfig, RaftNode, SnapshotBufferConfig,
    SnapshotMetadata,
};

fn create_test_block(height: u64, proposer: &str) -> Block {
    let header = BlockHeader::new(
        height,
        [0u8; 32],
        [0u8; 32],
        [0u8; 32],
        proposer.to_string(),
    );
    Block::new(header, vec![])
}

fn create_test_log_entry(index: u64, term: u64) -> LogEntry {
    LogEntry::new(term, index, create_test_block(index, "test_proposer"))
}

fn create_node_with_config(id: &str, peers: Vec<String>, config: RaftConfig) -> Arc<RaftNode> {
    let transport = Arc::new(MemoryTransport::new(id.to_string()));
    Arc::new(RaftNode::new(id.to_string(), peers, transport, config))
}

fn test_buffer_config() -> SnapshotBufferConfig {
    SnapshotBufferConfig {
        max_memory_bytes: 1024 * 1024, // 1MB for testing
        temp_dir: std::env::temp_dir().join("raft_snapshot_streaming_tests"),
        initial_file_capacity: 4096,
    }
}

#[test]
fn test_streaming_snapshot_roundtrip() {
    // Create entries
    let entries: Vec<LogEntry> = (1..=100).map(|i| create_test_log_entry(i, 1)).collect();

    // Serialize using streaming writer
    let buffer = serialize_entries(&entries, test_buffer_config()).unwrap();

    // Verify hash is consistent
    let hash = buffer.hash();
    assert_ne!(hash, [0u8; 32]);

    // Deserialize using streaming reader
    let bytes = buffer.as_bytes().unwrap();
    let restored = deserialize_entries(bytes).unwrap();

    assert_eq!(restored.len(), 100);
    for (i, entry) in restored.iter().enumerate() {
        assert_eq!(entry.index, (i + 1) as u64);
        assert_eq!(entry.term, 1);
    }
}

#[test]
fn test_streaming_snapshot_large() {
    // Create a larger snapshot (1000 entries)
    let entries: Vec<LogEntry> = (1..=1000)
        .map(|i| create_test_log_entry(i, (i / 100) + 1))
        .collect();

    let buffer = serialize_entries(&entries, test_buffer_config()).unwrap();

    // Read using streaming reader
    let reader = SnapshotReader::new(&buffer).unwrap();
    assert_eq!(reader.entry_count(), 1000);

    let restored: Vec<LogEntry> = reader.collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(restored.len(), 1000);
}

#[test]
fn test_streaming_partial_read() {
    let entries: Vec<LogEntry> = (1..=100).map(|i| create_test_log_entry(i, 1)).collect();

    let buffer = serialize_entries(&entries, test_buffer_config()).unwrap();
    let mut reader = SnapshotReader::new(&buffer).unwrap();

    // Read only first 10 entries
    for i in 1..=10 {
        let entry = reader.read_entry().unwrap().unwrap();
        assert_eq!(entry.index, i);
    }

    assert_eq!(reader.entries_read(), 10);
    assert_eq!(reader.remaining(), 90);
}

#[test]
fn test_buffer_spill_to_file() {
    // Create config with very small memory limit to force file spillage
    let config = SnapshotBufferConfig {
        max_memory_bytes: 1024, // 1KB - will definitely spill
        temp_dir: std::env::temp_dir().join("raft_snapshot_spill_test"),
        initial_file_capacity: 4096,
    };

    // Create entries that exceed memory limit
    let entries: Vec<LogEntry> = (1..=50).map(|i| create_test_log_entry(i, 1)).collect();

    let buffer = serialize_entries(&entries, config).unwrap();

    // Verify we can still read the data
    let bytes = buffer.as_bytes().unwrap();
    let restored = deserialize_entries(bytes).unwrap();

    assert_eq!(restored.len(), 50);
}

#[test]
fn test_zero_copy_chunk_access() {
    let config = RaftConfig {
        snapshot_chunk_size: 256, // Small chunks for testing
        snapshot_max_memory: 1024 * 1024,
        ..Default::default()
    };
    let node = create_node_with_config("leader", vec![], config);

    // Create a snapshot buffer with test data
    let entries: Vec<LogEntry> = (1..=20).map(|i| create_test_log_entry(i, 1)).collect();

    let buffer = serialize_entries(&entries, test_buffer_config()).unwrap();
    let total_len = buffer.total_len();

    // Use the streaming chunk iterator
    let mut chunk_count = 0;
    let mut total_bytes = 0u64;

    for chunk_result in node.snapshot_chunk_iter(&buffer) {
        let (offset, data, is_last) = chunk_result.unwrap();
        assert_eq!(offset, total_bytes);
        total_bytes += data.len() as u64;
        chunk_count += 1;

        if is_last {
            assert_eq!(total_bytes, total_len);
        }
    }

    assert!(chunk_count > 0);
    assert_eq!(total_bytes, total_len);
}

#[test]
fn test_get_snapshot_chunk_streaming() {
    let config = RaftConfig {
        snapshot_chunk_size: 100,
        ..Default::default()
    };
    let node = create_node_with_config("test", vec![], config);

    let entries: Vec<LogEntry> = (1..=10).map(|i| create_test_log_entry(i, 1)).collect();

    let buffer = serialize_entries(&entries, test_buffer_config()).unwrap();

    // Get first chunk
    let (chunk, is_last) = node.get_snapshot_chunk_streaming(&buffer, 0).unwrap();
    assert!(!chunk.is_empty());
    assert!(chunk.len() <= 100);

    // If buffer is small enough, first chunk might be last
    if buffer.total_len() <= 100 {
        assert!(is_last);
    } else {
        assert!(!is_last);
    }
}

#[test]
fn test_streaming_create_snapshot() {
    let config = RaftConfig {
        snapshot_threshold: 5,
        snapshot_max_memory: 1024 * 1024,
        ..Default::default()
    };
    // Single-node cluster to avoid quorum issues during propose
    let node = create_node_with_config("leader", vec![], config);

    // Make node a leader so it can propose blocks
    node.become_leader();

    // Propose blocks to create log entries
    for i in 1..=10 {
        let block = create_test_block(i, "leader");
        node.propose(block).unwrap();
    }

    // Set finalized height
    node.set_finalized_height(10);

    // Create streaming snapshot
    let (metadata, buffer) = node.create_snapshot_streaming().unwrap();

    assert_eq!(metadata.last_included_index, 10);
    assert!(buffer.total_len() > 0);

    // Verify we can read the entries back
    let reader = SnapshotReader::new(&buffer).unwrap();
    let entries: Vec<LogEntry> = reader.collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(entries.len(), 10);
}

#[test]
fn test_install_snapshot_streaming() {
    let config = RaftConfig {
        snapshot_max_memory: 1024 * 1024,
        ..Default::default()
    };
    let follower = create_node_with_config("follower", vec!["leader".to_string()], config);

    // Create snapshot entries
    let entries: Vec<LogEntry> = (1..=5).map(|i| create_test_log_entry(i, 1)).collect();

    let buffer = serialize_entries(&entries, test_buffer_config()).unwrap();
    let hash = buffer.hash();

    let metadata =
        SnapshotMetadata::new(5, 1, hash, vec!["leader".to_string()], buffer.total_len());

    // Install using streaming method
    let result = follower.install_snapshot_streaming(metadata, &buffer);
    assert!(result.is_ok());

    // Verify state was updated
    assert_eq!(follower.finalized_height(), 5);
}

#[test]
fn test_backwards_compatibility_legacy_format() {
    // Create legacy format snapshot (bincode serialized Vec<LogEntry>)
    let entries: Vec<LogEntry> = (1..=10).map(|i| create_test_log_entry(i, 1)).collect();

    let legacy_data = bitcode::serialize(&entries).unwrap();

    // deserialize_entries should handle legacy format
    let restored = deserialize_entries(&legacy_data).unwrap();
    assert_eq!(restored.len(), 10);

    for (i, entry) in restored.iter().enumerate() {
        assert_eq!(entry.index, (i + 1) as u64);
    }
}

#[test]
fn test_snapshot_buffer_cleanup() {
    let config = SnapshotBufferConfig {
        max_memory_bytes: 512, // Force file spillage
        temp_dir: std::env::temp_dir().join("raft_snapshot_cleanup_test"),
        initial_file_capacity: 4096,
    };

    let temp_dir = config.temp_dir.clone();
    std::fs::create_dir_all(&temp_dir).ok();

    // Create and fill buffer
    {
        let entries: Vec<LogEntry> = (1..=20).map(|i| create_test_log_entry(i, 1)).collect();

        let mut buffer = serialize_entries(&entries, config).unwrap();

        // Buffer should be using file mode
        assert!(buffer.total_len() > 512);

        // Cleanup should remove temp files
        buffer.cleanup().unwrap();
    }

    // Temp directory may still exist but our buffer's temp file should be gone
    // (We can't easily verify this without checking specific file patterns)
}

#[test]
fn test_concurrent_snapshot_readers() {
    use std::thread;

    let entries: Vec<LogEntry> = (1..=100).map(|i| create_test_log_entry(i, 1)).collect();

    let buffer = Arc::new(serialize_entries(&entries, test_buffer_config()).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let buf = Arc::clone(&buffer);
            thread::spawn(move || {
                let reader = SnapshotReader::new(&buf).unwrap();
                let restored: Vec<LogEntry> = reader.collect::<Result<Vec<_>, _>>().unwrap();
                assert_eq!(restored.len(), 100);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_streaming_hash_consistency() {
    let entries: Vec<LogEntry> = (1..=50).map(|i| create_test_log_entry(i, 1)).collect();

    // Create buffer and get hash
    let buffer = serialize_entries(&entries, test_buffer_config()).unwrap();
    let hash1 = buffer.hash();

    // Get bytes and compute hash manually
    use sha2::{Digest, Sha256};
    let bytes = buffer.as_bytes().unwrap();
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let hash2: [u8; 32] = hasher.finalize().into();

    assert_eq!(hash1, hash2);
}

#[test]
fn test_empty_snapshot() {
    let config = test_buffer_config();
    let writer = SnapshotWriter::new(config).unwrap();

    assert_eq!(writer.entry_count(), 0);

    let buffer = writer.finish().unwrap();
    let reader = SnapshotReader::new(&buffer).unwrap();

    assert_eq!(reader.entry_count(), 0);
    assert!(reader.collect::<Vec<_>>().is_empty());
}

#[test]
fn test_receive_snapshot_with_buffer() {
    let config = RaftConfig {
        snapshot_max_memory: 1024 * 1024,
        ..Default::default()
    };
    let follower = create_node_with_config("follower", vec![], config);

    // Simulate receiving chunks
    let entries: Vec<LogEntry> = (1..=5).map(|i| create_test_log_entry(i, 1)).collect();

    let src_buffer = serialize_entries(&entries, test_buffer_config()).unwrap();
    let bytes = src_buffer.as_bytes().unwrap();
    let total_size = bytes.len() as u64;

    // Split into chunks and receive them
    let chunk_size = 64;
    let mut offset = 0u64;

    while offset < total_size {
        let end = ((offset as usize) + chunk_size).min(bytes.len());
        let chunk = &bytes[offset as usize..end];
        let is_last = end >= bytes.len();

        let result = follower.receive_snapshot_chunk(offset, chunk, total_size, is_last);
        assert!(result.is_ok());

        if is_last {
            assert!(result.unwrap());
        }

        offset = end as u64;
    }

    // Get accumulated buffer
    let received_buffer = follower.take_pending_snapshot_buffer();
    assert!(received_buffer.is_some());

    let received_buffer = received_buffer.unwrap();
    let received_bytes = received_buffer.as_bytes().unwrap();
    assert_eq!(received_bytes, bytes);
}
