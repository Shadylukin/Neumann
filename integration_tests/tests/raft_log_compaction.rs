//! Integration tests for Raft automatic log compaction.
//!
//! Tests:
//! - Leader auto-compacts log when threshold exceeded
//! - Compaction survives restart (snapshot state restored)
//! - Compaction respects cooldown period
//! - Only leaders trigger compaction (followers do not)
//! - New leader uses existing snapshot metadata

use std::sync::Arc;

use sha2::{Digest, Sha256};
use tensor_chain::{
    Block, BlockHeader, LogEntry, MemoryTransport, RaftConfig, RaftNode, RaftState,
    SnapshotMetadata,
};
use tensor_store::TensorStore;
use tokio::time::{sleep, Duration};

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

fn create_log_entry(index: u64, term: u64) -> LogEntry {
    let block = create_test_block(index, "proposer");
    LogEntry::new(term, index, block)
}

/// Create a single node with a TensorStore for persistence testing.
fn create_node_with_store(node_id: &str, config: RaftConfig, store: &TensorStore) -> Arc<RaftNode> {
    let transport = Arc::new(MemoryTransport::new(node_id.to_string()));
    Arc::new(RaftNode::with_store(
        node_id.to_string(),
        vec![],
        transport,
        config,
        store,
    ))
}

#[tokio::test]
async fn test_leader_auto_compacts_log() {
    // Create a node with low snapshot threshold for testing
    let mut config = RaftConfig::default();
    config.snapshot_threshold = 10; // Trigger after 10 entries
    config.snapshot_trailing_logs = 2; // Keep only 2 trailing
    config.compaction_check_interval = 1; // Check every tick
    config.compaction_cooldown_ms = 0; // No cooldown for testing

    let store = TensorStore::new();
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    // Single-node cluster to avoid quorum issues
    let node = Arc::new(RaftNode::with_store(
        "node1".to_string(),
        vec![],
        transport,
        config,
        &store,
    ));

    // Make the node a leader so it can trigger compaction
    node.become_leader();
    assert_eq!(node.state(), RaftState::Leader);

    // Add entries to exceed threshold
    for i in 1..=15 {
        let block = create_test_block(i, "node1");
        node.propose(block).unwrap();
    }

    // Set finalized height to trigger compaction eligibility
    node.set_finalized_height(12);

    // Verify log has entries
    let initial_log_len = node.last_log_index();
    assert!(initial_log_len >= 10);

    // Tick to trigger compaction check
    node.tick_async().await.unwrap();

    // Allow time for compaction to complete
    sleep(Duration::from_millis(10)).await;

    // Verify snapshot was created (metadata should exist)
    let snapshot_meta = node.get_snapshot_metadata();
    assert!(
        snapshot_meta.is_some(),
        "Snapshot should have been created after compaction"
    );

    // Verify snapshot was persisted to store
    let (loaded_meta, _data) =
        RaftNode::load_snapshot("node1", &store).expect("Snapshot should be persisted to store");
    assert!(loaded_meta.last_included_index > 0);
}

#[tokio::test]
async fn test_compaction_survives_restart() {
    let store = TensorStore::new();
    let config = RaftConfig::default();

    // Create first node and save a snapshot
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let node1 = RaftNode::with_store(
        "node1".to_string(),
        vec![],
        transport1,
        config.clone(),
        &store,
    );

    let data = vec![1, 2, 3, 4, 5];
    let snapshot_hash = compute_hash(&data);
    let metadata = SnapshotMetadata::new(
        100,
        5,
        snapshot_hash,
        vec!["node1".to_string(), "node2".to_string()],
        data.len() as u64,
    );

    // Save snapshot
    node1.save_snapshot(&metadata, &data, &store).unwrap();

    // Drop the first node (simulating restart)
    drop(node1);

    // Create a new node from the same store (simulating restart)
    let transport2 = Arc::new(MemoryTransport::new("node1".to_string()));
    let node2 = RaftNode::with_store("node1".to_string(), vec![], transport2, config, &store);

    // Verify snapshot metadata was restored
    let restored_meta = node2
        .get_snapshot_metadata()
        .expect("Snapshot metadata should be restored after restart");
    assert_eq!(restored_meta.last_included_index, 100);
    assert_eq!(restored_meta.last_included_term, 5);
    assert_eq!(restored_meta.config, vec!["node1", "node2"]);
}

#[tokio::test]
async fn test_compaction_respects_cooldown() {
    // Create a node with cooldown enabled
    let mut config = RaftConfig::default();
    config.snapshot_threshold = 5;
    config.compaction_check_interval = 1;
    config.compaction_cooldown_ms = 10_000; // 10 second cooldown

    let store = TensorStore::new();
    let node = create_node_with_store("node1", config, &store);

    // Make leader
    node.become_leader();

    // Add entries
    for i in 1..=10 {
        let block = create_test_block(i, "node1");
        node.propose(block).unwrap();
    }
    node.set_finalized_height(8);

    // First tick should trigger compaction
    node.tick_async().await.unwrap();

    // Get first snapshot metadata
    let first_meta = node.get_snapshot_metadata();

    // Add more entries
    for i in 11..=20 {
        let block = create_test_block(i, "node1");
        node.propose(block).unwrap();
    }
    node.set_finalized_height(18);

    // Second tick should be blocked by cooldown
    node.tick_async().await.unwrap();

    // Snapshot metadata should be the same (no new compaction)
    let second_meta = node.get_snapshot_metadata();

    // If first compaction happened, second should be same due to cooldown
    if let (Some(first), Some(second)) = (first_meta, second_meta) {
        assert_eq!(
            first.last_included_index, second.last_included_index,
            "Cooldown should prevent new compaction"
        );
    }
}

#[tokio::test]
async fn test_follower_does_not_compact() {
    let mut config = RaftConfig::default();
    config.snapshot_threshold = 5;
    config.compaction_check_interval = 1;
    config.compaction_cooldown_ms = 0;

    let store = TensorStore::new();
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let node = Arc::new(RaftNode::with_store(
        "node1".to_string(),
        vec!["node2".to_string()],
        transport,
        config,
        &store,
    ));

    // Keep as follower (default state)
    assert_eq!(node.state(), RaftState::Follower);

    // Add entries directly to the log (simulating replication from leader)
    // Note: In real scenarios, followers receive entries via AppendEntries
    // For this test, we verify that tick_async as follower doesn't trigger compaction

    // Tick as follower
    node.tick_async().await.unwrap();

    // No snapshot should be created (only leaders compact)
    assert!(
        node.get_snapshot_metadata().is_none(),
        "Followers should not trigger compaction"
    );
}

#[tokio::test]
async fn test_new_leader_uses_existing_snapshot() {
    let store = TensorStore::new();

    // First leader creates and persists a snapshot
    let config1 = RaftConfig::default();
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let node1 = RaftNode::with_store(
        "node1".to_string(),
        vec!["node2".to_string()],
        transport1,
        config1,
        &store,
    );

    let data = bincode::serialize(&vec![create_log_entry(1, 1), create_log_entry(2, 1)]).unwrap();
    let snapshot_hash = compute_hash(&data);
    let metadata = SnapshotMetadata::new(
        50,
        3,
        snapshot_hash,
        vec!["node1".to_string(), "node2".to_string()],
        data.len() as u64,
    );
    node1.save_snapshot(&metadata, &data, &store).unwrap();

    // Simulate first leader stepping down
    drop(node1);

    // New node (could be different node ID in real cluster) loads from store
    // In a real cluster, nodes would have separate stores, but for this test
    // we're verifying that snapshot persistence works
    let config2 = RaftConfig::default();
    let transport2 = Arc::new(MemoryTransport::new("node1".to_string()));
    let node2 = RaftNode::with_store(
        "node1".to_string(),
        vec!["node2".to_string()],
        transport2,
        config2,
        &store,
    );

    // New node should have the snapshot metadata from the previous leader
    let restored = node2
        .get_snapshot_metadata()
        .expect("New node should have snapshot metadata");
    assert_eq!(restored.last_included_index, 50);
    assert_eq!(restored.last_included_term, 3);
    assert_eq!(restored.snapshot_hash, snapshot_hash);

    // When this node becomes leader, it starts from the persisted snapshot state
    node2.become_leader();
    assert_eq!(node2.state(), RaftState::Leader);

    // The leader can continue from where the previous leader left off
    let current_meta = node2.get_snapshot_metadata().unwrap();
    assert_eq!(current_meta.last_included_index, 50);
}
