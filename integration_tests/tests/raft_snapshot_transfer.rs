// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for Raft snapshot transfer in tensor_chain.
//!
//! Tests snapshot creation, transfer, and installation including:
//! - Snapshot creation from finalized log entries
//! - Chunked snapshot transfer between nodes
//! - Snapshot installation on lagging followers
//! - Log compaction after snapshot

use std::sync::Arc;

use sha2::{Digest, Sha256};
use tensor_chain::{
    Block, BlockHeader, LogEntry, MemoryTransport, RaftConfig, RaftNode, SnapshotMetadata,
};

fn compute_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

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

fn create_test_log_entry(index: u64) -> LogEntry {
    LogEntry::new(1, index, create_test_block(index, "test_proposer"))
}

fn create_node_with_config(id: &str, peers: Vec<String>, config: RaftConfig) -> Arc<RaftNode> {
    let transport = Arc::new(MemoryTransport::new(id.to_string()));
    Arc::new(RaftNode::new(id.to_string(), peers, transport, config))
}

#[test]
fn test_snapshot_creation_and_chunks() {
    let config = RaftConfig {
        snapshot_threshold: 5,
        snapshot_trailing_logs: 2,
        snapshot_chunk_size: 100, // Small chunks for testing
        ..Default::default()
    };
    let node = create_node_with_config("leader", vec!["follower".to_string()], config);

    // Snapshot creation requires finalized entries (which we don't have yet)
    let result = node.create_snapshot();
    assert!(result.is_err()); // No finalized entries yet

    // Test chunking on arbitrary data
    let data = vec![0u8; 250];
    let chunks = node.get_snapshot_chunks(&data);

    // With 100 byte chunks, 250 bytes should give 3 chunks
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].1.len(), 100);
    assert_eq!(chunks[1].1.len(), 100);
    assert_eq!(chunks[2].1.len(), 50);
    assert!(chunks[2].2); // Last chunk
}

#[test]
fn test_snapshot_metadata_roundtrip() {
    let original = SnapshotMetadata::new(
        100,
        5,
        [42u8; 32],
        vec![
            "node1".to_string(),
            "node2".to_string(),
            "node3".to_string(),
        ],
        1024 * 1024,
    );

    // Serialize and deserialize
    let bytes = bitcode::serialize(&original).unwrap();
    let restored: SnapshotMetadata = bitcode::deserialize(&bytes).unwrap();

    assert_eq!(restored.last_included_index, 100);
    assert_eq!(restored.last_included_term, 5);
    assert_eq!(restored.snapshot_hash, [42u8; 32]);
    assert_eq!(restored.config.len(), 3);
    assert_eq!(restored.size, 1024 * 1024);
}

#[test]
fn test_snapshot_chunk_receive_flow() {
    let config = RaftConfig::default();
    let node = create_node_with_config("follower", vec![], config);

    // Simulate receiving chunks
    let original_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let total_size = original_data.len() as u64;

    // Receive first chunk
    let result = node.receive_snapshot_chunk(0, &original_data[0..4], total_size, false);
    assert!(result.is_ok());
    assert!(!result.unwrap());

    // Receive second chunk
    let result = node.receive_snapshot_chunk(4, &original_data[4..8], total_size, false);
    assert!(result.is_ok());
    assert!(!result.unwrap());

    // Receive final chunk
    let result = node.receive_snapshot_chunk(8, &original_data[8..10], total_size, true);
    assert!(result.is_ok());
    assert!(result.unwrap()); // Complete!

    // Get the accumulated data
    let received = node.take_pending_snapshot_data();
    assert_eq!(received, original_data);
}

#[test]
fn test_snapshot_message_routing() {
    // Create leader and follower with connected transports
    let transport_leader = Arc::new(MemoryTransport::new("leader".to_string()));
    let transport_follower = Arc::new(MemoryTransport::new("follower".to_string()));

    // Connect transports
    transport_leader.connect_to("follower".to_string(), transport_follower.sender());
    transport_follower.connect_to("leader".to_string(), transport_leader.sender());

    let config = RaftConfig {
        snapshot_chunk_size: 1024,
        ..Default::default()
    };

    let _leader = Arc::new(RaftNode::new(
        "leader".to_string(),
        vec!["follower".to_string()],
        transport_leader,
        config.clone(),
    ));

    let _follower = Arc::new(RaftNode::new(
        "follower".to_string(),
        vec!["leader".to_string()],
        transport_follower,
        config,
    ));

    // Basic test that nodes can be created with connected transports
    // Full message routing tests are in unit tests
}

#[test]
fn test_install_snapshot_updates_state() {
    let config = RaftConfig::default();
    let node = create_node_with_config("follower", vec!["leader".to_string()], config);

    // Create snapshot data with log entries
    let entries: Vec<LogEntry> = (1..=10).map(create_test_log_entry).collect();
    let data = bitcode::serialize(&entries).unwrap();

    // Note: last_included_term must match the term in the log entries (which is 1)
    let snapshot_hash = compute_hash(&data);
    let metadata = SnapshotMetadata::new(
        10, // last_included_index
        1,  // last_included_term (matches log entry term)
        snapshot_hash,
        vec!["leader".to_string(), "follower".to_string()],
        data.len() as u64,
    );

    // Install the snapshot
    let result = node.install_snapshot(metadata.clone(), &data);
    assert!(result.is_ok());

    // Verify state was updated
    assert_eq!(node.finalized_height(), 10);
    assert!(node.get_snapshot_metadata().is_some());

    let snapshot_meta = node.get_snapshot_metadata().unwrap();
    assert_eq!(snapshot_meta.last_included_index, 10);
    assert_eq!(snapshot_meta.last_included_term, 1);
}

#[test]
fn test_snapshot_install_validates_data() {
    let config = RaftConfig::default();
    let node = create_node_with_config("follower", vec![], config);

    // Empty snapshot data
    let empty_entries: Vec<LogEntry> = vec![];
    let empty_data = bitcode::serialize(&empty_entries).unwrap();
    let metadata = SnapshotMetadata::new(5, 1, [0u8; 32], vec![], empty_data.len() as u64);

    let result = node.install_snapshot(metadata, &empty_data);
    assert!(result.is_err());

    // Mismatched index
    let entries: Vec<LogEntry> = (1..=3).map(create_test_log_entry).collect();
    let data = bitcode::serialize(&entries).unwrap();
    let wrong_metadata = SnapshotMetadata::new(
        10, // Claims index 10, but data only has up to 3
        1,
        [0u8; 32],
        vec![],
        data.len() as u64,
    );

    let result = node.install_snapshot(wrong_metadata, &data);
    assert!(result.is_err());
}

#[test]
fn test_needs_snapshot_detection() {
    let config = RaftConfig::default();
    let transport = Arc::new(MemoryTransport::new("leader".to_string()));
    let node = Arc::new(RaftNode::new(
        "leader".to_string(),
        vec!["follower".to_string()],
        transport,
        config,
    ));

    // Initially, no snapshot needed (no snapshot exists)
    assert!(!node.needs_snapshot_for_follower(&"follower".to_string()));
}

#[test]
fn test_config_snapshot_defaults() {
    let config = RaftConfig::default();

    assert_eq!(config.snapshot_threshold, 10_000);
    assert_eq!(config.snapshot_trailing_logs, 100);
    assert_eq!(config.snapshot_chunk_size, 1024 * 1024);
}

#[test]
fn test_should_compact_logic() {
    let config = RaftConfig {
        snapshot_threshold: 10,
        ..Default::default()
    };
    let node = create_node_with_config("node", vec![], config);

    // Without enough entries, should not compact
    assert!(!node.should_compact());
}
