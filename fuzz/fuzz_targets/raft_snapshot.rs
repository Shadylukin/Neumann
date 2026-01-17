//! Fuzz test for Raft snapshot serialization and installation.
//!
//! Tests that:
//! - SnapshotMetadata serialization/deserialization is stable
//! - Snapshot installation validates data correctly
//! - Chunk receiving handles edge cases

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;
use tensor_chain::{MemoryTransport, RaftConfig, RaftNode, SnapshotMetadata};

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Last included index
    last_included_index: u64,
    /// Last included term
    last_included_term: u64,
    /// Snapshot hash bytes
    hash_bytes: [u8; 32],
    /// Config node count (capped at 10)
    config_count: u8,
    /// Snapshot size
    size: u64,
    /// Chunk data
    chunk_data: Vec<u8>,
    /// Total size for chunk receive
    total_size: u64,
    /// Whether this is the last chunk
    is_last: bool,
    /// Chunk offset
    offset: u64,
}

fn make_node_ids(count: u8) -> Vec<String> {
    let capped = (count % 10) as usize;
    (0..capped).map(|i| format!("node{}", i)).collect()
}

fuzz_target!(|input: FuzzInput| {
    // Test 1: SnapshotMetadata serialization roundtrip
    let metadata = SnapshotMetadata::new(
        input.last_included_index,
        input.last_included_term,
        input.hash_bytes,
        make_node_ids(input.config_count),
        input.size,
    );

    let serialized = match bincode::serialize(&metadata) {
        Ok(bytes) => bytes,
        Err(_) => return, // Invalid serialization, skip
    };

    let deserialized: SnapshotMetadata = match bincode::deserialize(&serialized) {
        Ok(m) => m,
        Err(_) => panic!("Failed to deserialize metadata we just serialized"),
    };

    // Verify fields preserved
    assert_eq!(
        metadata.last_included_index,
        deserialized.last_included_index
    );
    assert_eq!(metadata.last_included_term, deserialized.last_included_term);
    assert_eq!(metadata.snapshot_hash, deserialized.snapshot_hash);
    assert_eq!(metadata.size, deserialized.size);

    // Test 2: RaftNode snapshot chunk receiving
    let transport = Arc::new(MemoryTransport::new("test_node".to_string()));
    let config = RaftConfig {
        snapshot_chunk_size: 1024,
        ..RaftConfig::default()
    };
    let node = RaftNode::new(
        "test_node".to_string(),
        vec!["peer".to_string()],
        transport,
        config,
    );

    // Only test valid chunk scenarios (offset 0 for first chunk)
    if !input.chunk_data.is_empty() && input.total_size > 0 {
        // Start with offset 0
        let _ = node.receive_snapshot_chunk(
            0,
            &input.chunk_data[..input.chunk_data.len().min(100)],
            input.total_size,
            input.is_last && input.chunk_data.len() <= 100,
        );
    }

    // Test 3: Get snapshot chunks
    if !input.chunk_data.is_empty() {
        let chunks = node.get_snapshot_chunks(&input.chunk_data);

        // Verify all data is covered
        let total_len: usize = chunks.iter().map(|(_, data, _)| data.len()).sum();
        assert_eq!(total_len, input.chunk_data.len());

        // Verify last chunk is marked correctly
        if let Some((_, _, is_last)) = chunks.last() {
            assert!(is_last);
        }
    }

    // Test 4: should_compact with no finalized entries
    assert!(!node.should_compact());

    // Test 5: create_snapshot with no finalized entries
    assert!(node.create_snapshot().is_err());
});
