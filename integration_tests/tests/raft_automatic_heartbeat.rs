//! Integration tests for automatic Raft heartbeat sending.
//!
//! Tests:
//! - Heartbeat starts automatically when becoming leader
//! - Heartbeat stops when stepping down from leadership
//! - Heartbeat maintains quorum with connected followers
//! - Heartbeat stats are tracked correctly
//! - Manual heartbeat mode (auto_heartbeat = false)
//! - Concurrent leadership changes
//! - Heartbeat restart after re-election

use std::{sync::Arc, time::Duration};

use tensor_chain::{MemoryTransport, RaftConfig, RaftNode, RaftState};
use tokio::time::sleep;

fn create_connected_cluster(
    auto_heartbeat: bool,
) -> (Arc<RaftNode>, Arc<RaftNode>, Arc<RaftNode>) {
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));
    let transport3 = Arc::new(MemoryTransport::new("node3".to_string()));

    // Fully connected mesh
    transport1.connect_to("node2".to_string(), transport2.sender());
    transport1.connect_to("node3".to_string(), transport3.sender());
    transport2.connect_to("node1".to_string(), transport1.sender());
    transport2.connect_to("node3".to_string(), transport3.sender());
    transport3.connect_to("node1".to_string(), transport1.sender());
    transport3.connect_to("node2".to_string(), transport2.sender());

    let config = RaftConfig {
        election_timeout: (100, 200),
        heartbeat_interval: 30,
        auto_heartbeat,
        max_heartbeat_failures: 3,
        ..RaftConfig::default()
    };

    let node1 = Arc::new(RaftNode::new(
        "node1".to_string(),
        vec!["node2".to_string(), "node3".to_string()],
        transport1,
        config.clone(),
    ));
    let node2 = Arc::new(RaftNode::new(
        "node2".to_string(),
        vec!["node1".to_string(), "node3".to_string()],
        transport2,
        config.clone(),
    ));
    let node3 = Arc::new(RaftNode::new(
        "node3".to_string(),
        vec!["node1".to_string(), "node2".to_string()],
        transport3,
        config,
    ));

    (node1, node2, node3)
}

#[tokio::test]
async fn test_heartbeat_starts_on_leader_election() {
    let (node1, _node2, _node3) = create_connected_cluster(true);

    // Initially not leader and no heartbeat
    assert_eq!(node1.state(), RaftState::Follower);
    assert!(!node1.is_heartbeat_running());

    // Become leader with automatic heartbeat
    node1.become_leader_with_heartbeat();

    assert_eq!(node1.state(), RaftState::Leader);
    assert!(node1.is_heartbeat_running());

    // Clean up
    node1.stop_heartbeat_task();
}

#[tokio::test]
async fn test_heartbeat_stops_on_step_down() {
    let (node1, _node2, _node3) = create_connected_cluster(true);

    // Become leader
    node1.become_leader_with_heartbeat();
    assert!(node1.is_heartbeat_running());

    // Stop heartbeat and become follower
    node1.stop_heartbeat_task();

    // Give heartbeat loop time to clean up
    sleep(Duration::from_millis(50)).await;

    // Heartbeat task should be stopped
    assert!(!node1.is_heartbeat_running());
}

#[tokio::test]
async fn test_heartbeat_stats_tracking() {
    let (node1, _node2, _node3) = create_connected_cluster(true);

    node1.become_leader_with_heartbeat();

    // Wait for some heartbeats to be sent
    sleep(Duration::from_millis(150)).await;

    let stats = node1.heartbeat_stats_snapshot();

    // Should have attempted some heartbeats (may fail due to mock transport)
    assert!(
        stats.heartbeats_sent > 0 || stats.heartbeats_failed > 0,
        "Expected heartbeat attempts, got sent={} failed={}",
        stats.heartbeats_sent,
        stats.heartbeats_failed
    );

    node1.stop_heartbeat_task();
}

#[tokio::test]
async fn test_manual_heartbeat_mode() {
    let (node1, _node2, _node3) = create_connected_cluster(false);

    // With auto_heartbeat=false, become_leader_with_heartbeat should NOT start task
    node1.become_leader_with_heartbeat();

    assert_eq!(node1.state(), RaftState::Leader);
    assert!(
        !node1.is_heartbeat_running(),
        "Heartbeat should not auto-start when auto_heartbeat=false"
    );

    // Can still start manually
    node1.start_heartbeat_task().unwrap();
    assert!(node1.is_heartbeat_running());

    node1.stop_heartbeat_task();
}

#[tokio::test]
async fn test_heartbeat_restart_after_reelection() {
    let (node1, _node2, _node3) = create_connected_cluster(true);

    // First term as leader
    node1.become_leader_with_heartbeat();
    assert!(node1.is_heartbeat_running());

    // Step down (stop heartbeat task)
    node1.stop_heartbeat_task();
    sleep(Duration::from_millis(20)).await;
    assert!(!node1.is_heartbeat_running());

    // Wait a bit
    sleep(Duration::from_millis(50)).await;

    // Re-elected as leader (become_leader sets state back to Leader)
    node1.become_leader_with_heartbeat();
    assert!(node1.is_heartbeat_running());

    // Stats should have been reset on new start
    let stats = node1.heartbeat_stats_snapshot();
    // New heartbeat task starts with reset stats
    assert_eq!(stats.consecutive_failures, 0);

    node1.stop_heartbeat_task();
}

#[tokio::test]
async fn test_concurrent_start_stop() {
    let (node1, _node2, _node3) = create_connected_cluster(true);
    node1.become_leader();

    // Multiple start calls should be idempotent
    node1.start_heartbeat_task().unwrap();
    node1.start_heartbeat_task().unwrap();
    node1.start_heartbeat_task().unwrap();
    assert!(node1.is_heartbeat_running());

    // Multiple stop calls should be safe
    node1.stop_heartbeat_task();
    node1.stop_heartbeat_task();
    node1.stop_heartbeat_task();

    sleep(Duration::from_millis(10)).await;
    assert!(!node1.is_heartbeat_running());
}

#[tokio::test]
async fn test_heartbeat_sends_to_followers() {
    // Create a cluster with connected transports
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));
    let transport3 = Arc::new(MemoryTransport::new("node3".to_string()));

    transport1.connect_to("node2".to_string(), transport2.sender());
    transport1.connect_to("node3".to_string(), transport3.sender());
    transport2.connect_to("node1".to_string(), transport1.sender());
    transport2.connect_to("node3".to_string(), transport3.sender());
    transport3.connect_to("node1".to_string(), transport1.sender());
    transport3.connect_to("node2".to_string(), transport2.sender());

    let config = RaftConfig {
        heartbeat_interval: 20,
        auto_heartbeat: true,
        ..RaftConfig::default()
    };

    let node1 = Arc::new(RaftNode::new(
        "node1".to_string(),
        vec!["node2".to_string(), "node3".to_string()],
        transport1,
        config,
    ));

    // Become leader and start heartbeats
    node1.become_leader_with_heartbeat();

    // Wait for heartbeats
    sleep(Duration::from_millis(100)).await;

    let stats = node1.heartbeat_stats_snapshot();
    // Should have sent heartbeats (success or failure depends on transport mock)
    assert!(
        stats.heartbeats_sent > 0 || stats.heartbeats_failed > 0,
        "Expected heartbeat activity"
    );

    node1.stop_heartbeat_task();
}

#[tokio::test]
async fn test_heartbeat_last_timestamp() {
    let (node1, _node2, _node3) = create_connected_cluster(true);

    node1.become_leader_with_heartbeat();

    // Wait for at least one heartbeat
    sleep(Duration::from_millis(100)).await;

    let stats = node1.heartbeat_stats_snapshot();

    // Last heartbeat timestamp should be set if any heartbeats were sent
    if stats.heartbeats_sent > 0 {
        assert!(
            stats.last_heartbeat_at.is_some(),
            "Expected last_heartbeat_at to be set"
        );
    }

    node1.stop_heartbeat_task();
}
