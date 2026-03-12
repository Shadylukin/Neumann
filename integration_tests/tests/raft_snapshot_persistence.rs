// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for Raft snapshot persistence.
//!
//! Tests:
//! - Snapshot hash validation on receipt
//! - Follower persists received snapshot
//! - Startup recovers from persisted snapshot
//! - Corrupted snapshot rejected on startup
//! - Full snapshot roundtrip integrity

use std::sync::Arc;

use sha2::{Digest, Sha256};
use tensor_chain::{
    Block, BlockHeader, LogEntry, MemoryTransport, RaftConfig, RaftNode, SnapshotMetadata,
};
use tensor_store::TensorStore;

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

fn create_log_entry(index: u64, term: u64) -> LogEntry {
    let block = create_test_block(index, "proposer");
    LogEntry::new(term, index, block)
}

fn compute_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

#[test]
fn test_snapshot_hash_computed_correctly() {
    let store = TensorStore::new();
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let mut config = RaftConfig::default();
    config.snapshot_threshold = 5;

    let node = Arc::new(RaftNode::with_store(
        "node1".to_string(),
        vec![],
        transport,
        config,
        &store,
    ));

    // Make leader and add entries
    node.become_leader();
    for i in 1..=5 {
        let block = create_test_block(i, "node1");
        node.propose(block).unwrap();
    }
    node.set_finalized_height(5);

    // Create snapshot
    let (metadata, data) = node.create_snapshot().unwrap();

    // Verify hash matches data
    let expected_hash = compute_hash(&data);
    assert_eq!(metadata.snapshot_hash, expected_hash);
    assert_ne!(metadata.snapshot_hash, [0u8; 32]);
}

#[test]
fn test_follower_persists_snapshot() {
    let store = TensorStore::new();
    let transport = Arc::new(MemoryTransport::new("follower".to_string()));

    let node = RaftNode::with_store(
        "follower".to_string(),
        vec!["leader".to_string()],
        transport,
        RaftConfig::default(),
        &store,
    );

    // Create valid snapshot data
    let entries: Vec<LogEntry> = (1..=10).map(|i| create_log_entry(i, 1)).collect();
    let data = bitcode::serialize(&entries).unwrap();
    let snapshot_hash = compute_hash(&data);

    let metadata = SnapshotMetadata::new(
        10,
        1,
        snapshot_hash,
        vec!["follower".to_string(), "leader".to_string()],
        data.len() as u64,
    );

    // Install snapshot (simulating receipt from leader)
    node.install_snapshot(metadata.clone(), &data).unwrap();

    // Save to store (as follower would do)
    node.save_snapshot(&metadata, &data, &store).unwrap();

    // Verify persisted
    let (loaded_meta, loaded_data) =
        RaftNode::load_snapshot("follower", &store).expect("Should load snapshot");
    assert_eq!(loaded_meta.last_included_index, 10);
    assert_eq!(loaded_data, data);
}

#[test]
fn test_startup_loads_valid_snapshot() {
    let store = TensorStore::new();

    // Create and persist a valid snapshot
    let entries: Vec<LogEntry> = (1..=5).map(|i| create_log_entry(i, 1)).collect();
    let data = bitcode::serialize(&entries).unwrap();
    let snapshot_hash = compute_hash(&data);

    let metadata = SnapshotMetadata::new(
        5,
        1,
        snapshot_hash,
        vec!["node1".to_string()],
        data.len() as u64,
    );

    // Save snapshot directly to store
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let node1 = RaftNode::with_store(
        "node1".to_string(),
        vec![],
        transport1,
        RaftConfig::default(),
        &store,
    );
    node1.save_snapshot(&metadata, &data, &store).unwrap();
    drop(node1);

    // Create new node - should load snapshot
    let transport2 = Arc::new(MemoryTransport::new("node1".to_string()));
    let node2 = RaftNode::with_store(
        "node1".to_string(),
        vec![],
        transport2,
        RaftConfig::default(),
        &store,
    );

    let loaded = node2.get_snapshot_metadata().expect("Should have snapshot");
    assert_eq!(loaded.last_included_index, 5);
    assert_eq!(loaded.last_included_term, 1);
}

#[test]
fn test_startup_rejects_corrupted_snapshot() {
    let store = TensorStore::new();

    // Create and persist a valid snapshot
    let entries: Vec<LogEntry> = (1..=5).map(|i| create_log_entry(i, 1)).collect();
    let data = bitcode::serialize(&entries).unwrap();
    let snapshot_hash = compute_hash(&data);

    let metadata = SnapshotMetadata::new(
        5,
        1,
        snapshot_hash,
        vec!["node1".to_string()],
        data.len() as u64,
    );

    // Save snapshot
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let node1 = RaftNode::with_store(
        "node1".to_string(),
        vec![],
        transport1,
        RaftConfig::default(),
        &store,
    );
    node1.save_snapshot(&metadata, &data, &store).unwrap();
    drop(node1);

    // Corrupt the data in store
    let corrupted_data = vec![0u8; data.len()];
    let mut snap_data = tensor_store::TensorData::new();
    snap_data.set(
        "data",
        tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Bytes(corrupted_data)),
    );
    store.put("_raft:snapshot:data:node1", snap_data).unwrap();

    // Create new node - should reject corrupted snapshot
    let transport2 = Arc::new(MemoryTransport::new("node1".to_string()));
    let node2 = RaftNode::with_store(
        "node1".to_string(),
        vec![],
        transport2,
        RaftConfig::default(),
        &store,
    );

    // Corrupted snapshot should be rejected
    assert!(node2.get_snapshot_metadata().is_none());
}

#[test]
fn test_snapshot_roundtrip_integrity() {
    let store = TensorStore::new();

    // Create leader with entries (single-node cluster for propose to work without quorum issues)
    let transport1 = Arc::new(MemoryTransport::new("leader".to_string()));
    let mut config = RaftConfig::default();
    config.snapshot_threshold = 5;

    let leader = Arc::new(RaftNode::with_store(
        "leader".to_string(),
        vec![],
        transport1,
        config.clone(),
        &store,
    ));

    leader.become_leader();
    for i in 1..=10 {
        let block = create_test_block(i, "leader");
        leader.propose(block).unwrap();
    }
    leader.set_finalized_height(10);

    // Create snapshot
    let (metadata, data) = leader.create_snapshot().unwrap();

    // Save snapshot (as leader does during compaction)
    leader.save_snapshot(&metadata, &data, &store).unwrap();

    // Simulate follower receiving and persisting
    let follower_store = TensorStore::new();
    let transport2 = Arc::new(MemoryTransport::new("follower".to_string()));
    let follower = RaftNode::with_store(
        "follower".to_string(),
        vec!["leader".to_string()],
        transport2,
        config.clone(),
        &follower_store,
    );

    // Install and persist
    follower.install_snapshot(metadata.clone(), &data).unwrap();
    follower
        .save_snapshot(&metadata, &data, &follower_store)
        .unwrap();
    drop(follower);

    // Restart follower - should recover
    let transport3 = Arc::new(MemoryTransport::new("follower".to_string()));
    let follower2 = RaftNode::with_store(
        "follower".to_string(),
        vec!["leader".to_string()],
        transport3,
        config,
        &follower_store,
    );

    let restored = follower2.get_snapshot_metadata().expect("Should restore");
    assert_eq!(restored.last_included_index, metadata.last_included_index);
    assert_eq!(restored.last_included_term, metadata.last_included_term);
    assert_eq!(restored.snapshot_hash, metadata.snapshot_hash);
}
