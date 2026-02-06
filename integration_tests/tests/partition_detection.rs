// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for partition detection and metrics observability.
//!
//! Tests the new partition detection features:
//! - PartitionStatus tracking (QuorumReachable, QuorumLost, Stalemate)
//! - MembershipStats health check metrics
//! - ChainMetrics aggregation during partitions
//! - QuorumTracker in RaftNode

use std::{
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use tensor_chain::{
    membership::{ClusterConfig, HealthConfig, LocalNodeConfig},
    ChainMetrics, MembershipManager, MembershipStats, MemoryTransport, PartitionStatus,
    QuorumTracker, RaftConfig, RaftNode, RaftStats,
};

fn create_test_cluster_config(node_id: &str, peers: &[(&str, &str)]) -> ClusterConfig {
    let mut config = ClusterConfig::new(
        "test-cluster",
        LocalNodeConfig {
            node_id: node_id.to_string(),
            bind_address: "127.0.0.1:9100".parse().unwrap(),
        },
    );

    for (peer_id, addr) in peers {
        config = config.with_peer(*peer_id, addr.parse().unwrap());
    }

    config.with_health(HealthConfig {
        ping_interval_ms: 50,
        failure_threshold: 2,
        ping_timeout_ms: 100,
        startup_grace_ms: 0, // No grace period for tests
    })
}

// ============================================================================
// PartitionStatus Tests
// ============================================================================

#[tokio::test]
async fn test_partition_status_quorum_reachable() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config(
        "node1",
        &[("node2", "127.0.0.1:9101"), ("node3", "127.0.0.1:9102")],
    );

    let manager = MembershipManager::new(config, transport);

    // Mark all peers healthy - should have quorum (3 nodes, 2 needed)
    manager.mark_healthy(&"node2".to_string());
    manager.mark_healthy(&"node3".to_string());

    let status = manager.partition_status();
    assert_eq!(status, PartitionStatus::QuorumReachable);
    assert!(manager.is_safe_to_write());
}

#[tokio::test]
async fn test_partition_status_quorum_lost() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config(
        "node1",
        &[
            ("node2", "127.0.0.1:9101"),
            ("node3", "127.0.0.1:9102"),
            ("node4", "127.0.0.1:9103"),
            ("node5", "127.0.0.1:9104"),
        ],
    );

    let manager = MembershipManager::new(config, transport);

    // Mark only 1 peer healthy - not enough for quorum (5 nodes, need 3)
    manager.mark_healthy(&"node2".to_string());
    manager.mark_failed(&"node3".to_string());
    manager.mark_failed(&"node4".to_string());
    manager.mark_failed(&"node5".to_string());

    // With local node + node2 = 2 healthy, need 3 for quorum
    let status = manager.partition_status();
    assert_eq!(status, PartitionStatus::QuorumLost);
    assert!(!manager.is_safe_to_write());
}

#[tokio::test]
async fn test_partition_status_stalemate() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    // 4 node cluster for stalemate
    let config = create_test_cluster_config(
        "node1",
        &[
            ("node2", "127.0.0.1:9101"),
            ("node3", "127.0.0.1:9102"),
            ("node4", "127.0.0.1:9103"),
        ],
    );

    let manager = MembershipManager::new(config, transport);

    // Mark 1 peer healthy, 2 failed -> 2 healthy (local + node2), 2 failed
    // This is a 50/50 split - stalemate
    manager.mark_healthy(&"node2".to_string());
    manager.mark_failed(&"node3".to_string());
    manager.mark_failed(&"node4".to_string());

    let status = manager.partition_status();
    assert_eq!(status, PartitionStatus::Stalemate);
    assert!(!manager.is_safe_to_write());
}

#[tokio::test]
async fn test_partition_status_unknown_during_grace() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let mut config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);
    // Set a long grace period
    config = config.with_health(HealthConfig {
        ping_interval_ms: 50,
        failure_threshold: 2,
        ping_timeout_ms: 100,
        startup_grace_ms: 60000, // 60 seconds
    });

    let manager = MembershipManager::new(config, transport);

    // During grace period, status should be Unknown
    let status = manager.partition_status();
    assert_eq!(status, PartitionStatus::Unknown);
}

#[tokio::test]
async fn test_partition_status_in_cluster_view() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config(
        "node1",
        &[("node2", "127.0.0.1:9101"), ("node3", "127.0.0.1:9102")],
    );

    let manager = MembershipManager::new(config, transport);

    // Initially all unknown
    let view = manager.view();
    // Status might be Unknown during initial period
    assert!(
        view.partition_status == PartitionStatus::Unknown
            || view.partition_status == PartitionStatus::QuorumLost
    );

    // Mark all healthy
    manager.mark_healthy(&"node2".to_string());
    manager.mark_healthy(&"node3".to_string());

    let view = manager.view();
    assert_eq!(view.partition_status, PartitionStatus::QuorumReachable);
}

// ============================================================================
// MembershipStats Tests
// ============================================================================

#[tokio::test]
async fn test_membership_stats_health_checks() {
    let stats = MembershipStats::new();

    // Record some health checks
    stats.health_checks.fetch_add(10, Ordering::Relaxed);
    stats.health_check_failures.fetch_add(2, Ordering::Relaxed);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.health_checks, 10);
    assert_eq!(snapshot.health_check_failures, 2);
}

#[tokio::test]
async fn test_membership_stats_partition_events() {
    let stats = MembershipStats::new();

    stats.partition_events.fetch_add(1, Ordering::Relaxed);
    stats.quorum_lost_events.fetch_add(1, Ordering::Relaxed);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.partition_events, 1);
    assert_eq!(snapshot.quorum_lost_events, 1);
}

#[tokio::test]
async fn test_membership_stats_ping_timing() {
    let stats = MembershipStats::new();

    // Record some ping timings
    stats.ping_timing.record(100);
    stats.ping_timing.record(200);
    stats.ping_timing.record(150);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.ping_timing.count, 3);
    assert_eq!(snapshot.ping_timing.total_us, 450);
    assert_eq!(snapshot.ping_timing.min_us, 100);
    assert_eq!(snapshot.ping_timing.max_us, 200);
}

// ============================================================================
// QuorumTracker Tests
// ============================================================================

#[tokio::test]
async fn test_quorum_tracker_success_failure() {
    let tracker = QuorumTracker::new(Duration::from_secs(5), 3);

    // Record successes
    tracker.record_success(&"node2".to_string());
    tracker.record_success(&"node3".to_string());

    assert!(tracker.is_reachable(&"node2".to_string()));
    assert!(tracker.is_reachable(&"node3".to_string()));

    // Record failures for node2
    tracker.record_failure(&"node2".to_string());
    tracker.record_failure(&"node2".to_string());
    tracker.record_failure(&"node2".to_string());

    // After 3 failures, node2 should be unreachable
    assert!(!tracker.is_reachable(&"node2".to_string()));
    assert!(tracker.is_reachable(&"node3".to_string()));
}

#[tokio::test]
async fn test_quorum_tracker_has_quorum_3_nodes() {
    let tracker = QuorumTracker::new(Duration::from_secs(5), 3);

    // No responses yet - no quorum (only self = 1)
    assert!(!tracker.has_quorum(2)); // 3 nodes total, need 2

    // 1 reachable peer + self = 2 >= 2 quorum
    tracker.record_success(&"node2".to_string());
    assert!(tracker.has_quorum(2));
}

#[tokio::test]
async fn test_quorum_tracker_has_quorum_5_nodes() {
    let tracker = QuorumTracker::new(Duration::from_secs(5), 3);

    // 5 nodes = 4 peers + self
    // Quorum = 3

    // Self + 1 peer = 2, not enough
    tracker.record_success(&"node2".to_string());
    assert!(!tracker.has_quorum(4)); // Need 3, have 2

    // Self + 2 peers = 3, exactly quorum
    tracker.record_success(&"node3".to_string());
    assert!(tracker.has_quorum(4));

    // Self + 3 peers = 4, above quorum
    tracker.record_success(&"node4".to_string());
    assert!(tracker.has_quorum(4));
}

#[tokio::test]
async fn test_quorum_tracker_reachable_count() {
    let tracker = QuorumTracker::new(Duration::from_secs(5), 3);

    tracker.record_success(&"node2".to_string());
    tracker.record_success(&"node3".to_string());

    assert_eq!(tracker.reachable_count(), 2);

    // Fail node2
    for _ in 0..3 {
        tracker.record_failure(&"node2".to_string());
    }

    assert_eq!(tracker.reachable_count(), 1);
}

#[tokio::test]
async fn test_quorum_tracker_recovery() {
    let tracker = QuorumTracker::new(Duration::from_secs(5), 3);

    // Mark node2 as failed
    tracker.record_success(&"node2".to_string());
    for _ in 0..3 {
        tracker.record_failure(&"node2".to_string());
    }
    assert!(!tracker.is_reachable(&"node2".to_string()));

    // Successful response should reset failure count and make it reachable
    tracker.record_success(&"node2".to_string());
    assert!(tracker.is_reachable(&"node2".to_string()));
}

// ============================================================================
// RaftStats Tests
// ============================================================================

#[tokio::test]
async fn test_raft_stats_heartbeat_tracking() {
    let stats = RaftStats::new();

    stats.heartbeat_successes.fetch_add(10, Ordering::Relaxed);
    stats.heartbeat_failures.fetch_add(2, Ordering::Relaxed);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.heartbeat_successes, 10);
    assert_eq!(snapshot.heartbeat_failures, 2);
}

#[tokio::test]
async fn test_raft_stats_quorum_events() {
    let stats = RaftStats::new();

    stats.quorum_checks.fetch_add(100, Ordering::Relaxed);
    stats.quorum_lost_events.fetch_add(1, Ordering::Relaxed);
    stats.leader_step_downs.fetch_add(1, Ordering::Relaxed);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.quorum_checks, 100);
    assert_eq!(snapshot.quorum_lost_events, 1);
    assert_eq!(snapshot.leader_step_downs, 1);
}

#[tokio::test]
async fn test_raft_stats_timing() {
    let stats = RaftStats::new();

    // Record election timing
    stats.election_timing.record(5000); // 5ms
    stats.election_timing.record(7000); // 7ms

    // Record heartbeat timing
    stats.heartbeat_timing.record(100);
    stats.heartbeat_timing.record(200);

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.election_timing.count, 2);
    assert_eq!(snapshot.heartbeat_timing.count, 2);
}

// ============================================================================
// ChainMetrics Integration Tests
// ============================================================================

#[tokio::test]
async fn test_chain_metrics_aggregation() {
    let metrics = ChainMetrics::new();

    // Add some stats
    metrics
        .raft
        .heartbeat_successes
        .fetch_add(100, Ordering::Relaxed);
    metrics
        .raft
        .heartbeat_failures
        .fetch_add(5, Ordering::Relaxed);
    metrics.dtx.started.fetch_add(50, Ordering::Relaxed);
    metrics.dtx.committed.fetch_add(45, Ordering::Relaxed);
    metrics
        .membership
        .health_checks
        .fetch_add(200, Ordering::Relaxed);
    metrics
        .membership
        .health_check_failures
        .fetch_add(10, Ordering::Relaxed);

    let snapshot = metrics.snapshot();

    // Verify aggregated values
    assert_eq!(snapshot.raft.heartbeat_successes, 100);
    assert_eq!(snapshot.raft.heartbeat_failures, 5);
    assert_eq!(snapshot.dtx.started, 50);
    assert_eq!(snapshot.dtx.committed, 45);
    assert_eq!(snapshot.membership.health_checks, 200);
    assert_eq!(snapshot.membership.health_check_failures, 10);
}

#[tokio::test]
async fn test_chain_metrics_is_cluster_healthy() {
    let metrics = ChainMetrics::new();

    // Healthy cluster metrics
    metrics
        .raft
        .heartbeat_successes
        .fetch_add(100, Ordering::Relaxed);
    metrics
        .raft
        .heartbeat_failures
        .fetch_add(5, Ordering::Relaxed); // 95% success
    metrics
        .membership
        .health_checks
        .fetch_add(100, Ordering::Relaxed);
    metrics
        .membership
        .health_check_failures
        .fetch_add(5, Ordering::Relaxed); // 95% success

    let snapshot = metrics.snapshot();
    assert!(snapshot.is_cluster_healthy());

    // Add quorum lost event - should be unhealthy
    metrics
        .raft
        .quorum_lost_events
        .fetch_add(1, Ordering::Relaxed);
    let snapshot = metrics.snapshot();
    assert!(!snapshot.is_cluster_healthy());
}

#[tokio::test]
async fn test_chain_metrics_heartbeat_success_rate() {
    let metrics = ChainMetrics::new();

    metrics
        .raft
        .heartbeat_successes
        .fetch_add(80, Ordering::Relaxed);
    metrics
        .raft
        .heartbeat_failures
        .fetch_add(20, Ordering::Relaxed);

    let snapshot = metrics.snapshot();
    let rate = snapshot.heartbeat_success_rate();

    assert!((rate - 0.8).abs() < 0.001); // 80%
}

#[tokio::test]
async fn test_chain_metrics_tx_commit_rate() {
    let metrics = ChainMetrics::new();

    metrics.dtx.started.fetch_add(100, Ordering::Relaxed);
    metrics.dtx.committed.fetch_add(90, Ordering::Relaxed);

    let snapshot = metrics.snapshot();
    let rate = snapshot.tx_commit_rate();

    assert!((rate - 0.9).abs() < 0.001); // 90%
}

// ============================================================================
// RaftNode Integration with QuorumTracker Tests
// ============================================================================

fn create_3_node_cluster() -> (Arc<RaftNode>, Arc<RaftNode>, Arc<RaftNode>) {
    let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let t2 = Arc::new(MemoryTransport::new("node2".to_string()));
    let t3 = Arc::new(MemoryTransport::new("node3".to_string()));

    t1.connect_to("node2".to_string(), t2.sender());
    t1.connect_to("node3".to_string(), t3.sender());
    t2.connect_to("node1".to_string(), t1.sender());
    t2.connect_to("node3".to_string(), t3.sender());
    t3.connect_to("node1".to_string(), t1.sender());
    t3.connect_to("node2".to_string(), t2.sender());

    let config = RaftConfig {
        election_timeout: (50, 100),
        heartbeat_interval: 25,
        ..RaftConfig::default()
    };

    let node1 = Arc::new(RaftNode::new(
        "node1".to_string(),
        vec!["node2".to_string(), "node3".to_string()],
        t1,
        config.clone(),
    ));
    let node2 = Arc::new(RaftNode::new(
        "node2".to_string(),
        vec!["node1".to_string(), "node3".to_string()],
        t2,
        config.clone(),
    ));
    let node3 = Arc::new(RaftNode::new(
        "node3".to_string(),
        vec!["node1".to_string(), "node2".to_string()],
        t3,
        config,
    ));

    (node1, node2, node3)
}

#[tokio::test]
async fn test_raft_node_stats_accessor() {
    let (node1, _node2, _node3) = create_3_node_cluster();

    // Access stats through node
    let stats = node1.stats();
    let snapshot = stats.snapshot();

    // Should start at zero
    assert_eq!(snapshot.heartbeat_successes, 0);
    assert_eq!(snapshot.heartbeat_failures, 0);
}

#[tokio::test]
async fn test_raft_node_quorum_tracker_accessor() {
    let (node1, _node2, _node3) = create_3_node_cluster();

    // Access quorum tracker through node
    let tracker = node1.quorum_tracker();

    // Initially no peers should be reachable (no heartbeats sent yet)
    assert_eq!(tracker.reachable_count(), 0);
}

// ============================================================================
// Partition Scenario Tests
// ============================================================================

#[tokio::test]
async fn test_membership_partition_detection_flow() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config(
        "node1",
        &[("node2", "127.0.0.1:9101"), ("node3", "127.0.0.1:9102")],
    );

    let manager = MembershipManager::new(config, transport);

    // Initially mark all healthy
    manager.mark_healthy(&"node2".to_string());
    manager.mark_healthy(&"node3".to_string());

    assert_eq!(manager.partition_status(), PartitionStatus::QuorumReachable);
    assert!(manager.is_safe_to_write());

    // Simulate partition - lose both peers
    manager.mark_failed(&"node2".to_string());
    manager.mark_failed(&"node3".to_string());

    // With only local node healthy (1 out of 3), quorum lost
    assert_eq!(manager.partition_status(), PartitionStatus::QuorumLost);
    assert!(!manager.is_safe_to_write());

    // Partition heals - one peer recovers
    manager.mark_healthy(&"node2".to_string());

    // Now 2 out of 3 healthy - quorum restored
    assert_eq!(manager.partition_status(), PartitionStatus::QuorumReachable);
    assert!(manager.is_safe_to_write());
}

#[tokio::test]
async fn test_metrics_track_partition_events() {
    let metrics = ChainMetrics::new();

    // Simulate normal operation
    for _ in 0..100 {
        metrics
            .raft
            .heartbeat_successes
            .fetch_add(1, Ordering::Relaxed);
    }

    // Simulate partition
    for _ in 0..10 {
        metrics
            .raft
            .heartbeat_failures
            .fetch_add(1, Ordering::Relaxed);
    }
    metrics
        .raft
        .quorum_lost_events
        .fetch_add(1, Ordering::Relaxed);
    metrics
        .membership
        .partition_events
        .fetch_add(1, Ordering::Relaxed);

    let snapshot = metrics.snapshot();

    // Verify metrics captured the partition
    assert_eq!(snapshot.raft.heartbeat_successes, 100);
    assert_eq!(snapshot.raft.heartbeat_failures, 10);
    assert_eq!(snapshot.raft.quorum_lost_events, 1);
    assert_eq!(snapshot.membership.partition_events, 1);

    // Cluster should be unhealthy due to quorum lost
    assert!(!snapshot.is_cluster_healthy());
}
