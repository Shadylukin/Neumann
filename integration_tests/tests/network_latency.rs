//! Integration tests for network latency simulation.
//!
//! Tests distributed system behavior under various latency conditions:
//! - Raft consensus with high latency
//! - Leadership election timing
//! - Heartbeat failures due to latency
//! - Per-peer latency differences (asymmetric networks)

use std::{sync::Arc, time::Duration};

use tensor_chain::{MemoryTransport, RaftConfig, RaftNode, RaftState};
use tokio::time::{sleep, Instant};

fn create_cluster_with_latency(
    latency_ms: Option<(u64, u64)>,
) -> (Arc<RaftNode>, Arc<RaftNode>, Arc<RaftNode>) {
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));
    let transport3 = Arc::new(MemoryTransport::new("node3".to_string()));

    // Set latency on all transports
    if let Some(latency) = latency_ms {
        transport1.set_latency(Some(latency));
        transport2.set_latency(Some(latency));
        transport3.set_latency(Some(latency));
    }

    // Fully connected mesh
    transport1.connect_to("node2".to_string(), transport2.sender());
    transport1.connect_to("node3".to_string(), transport3.sender());
    transport2.connect_to("node1".to_string(), transport1.sender());
    transport2.connect_to("node3".to_string(), transport3.sender());
    transport3.connect_to("node1".to_string(), transport1.sender());
    transport3.connect_to("node2".to_string(), transport2.sender());

    let config = RaftConfig {
        election_timeout: (150, 300),
        heartbeat_interval: 50,
        auto_heartbeat: true,
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
async fn test_raft_with_no_latency() {
    let (node1, _node2, _node3) = create_cluster_with_latency(None);

    let start = Instant::now();
    node1.become_leader_with_heartbeat();
    let elapsed = start.elapsed();

    assert_eq!(node1.state(), RaftState::Leader);
    // Should be very fast without latency
    assert!(
        elapsed < Duration::from_millis(10),
        "Expected <10ms, got {:?}",
        elapsed
    );

    node1.stop_heartbeat_task();
}

#[tokio::test]
async fn test_raft_with_low_latency() {
    // 5-10ms latency (typical LAN)
    let (node1, _node2, _node3) = create_cluster_with_latency(Some((5, 10)));

    node1.become_leader_with_heartbeat();
    assert_eq!(node1.state(), RaftState::Leader);

    // Wait for some heartbeats (each will have latency)
    sleep(Duration::from_millis(200)).await;

    let stats = node1.heartbeat_stats_snapshot();
    // Should have sent some heartbeats despite latency
    assert!(
        stats.heartbeats_sent > 0 || stats.heartbeats_failed > 0,
        "Expected heartbeat activity"
    );

    node1.stop_heartbeat_task();
}

#[tokio::test]
async fn test_raft_with_high_latency() {
    // 50-100ms latency (cross-region)
    let (node1, _node2, _node3) = create_cluster_with_latency(Some((50, 100)));

    node1.become_leader_with_heartbeat();
    assert_eq!(node1.state(), RaftState::Leader);

    // Wait for heartbeats - with high latency they'll be slow
    sleep(Duration::from_millis(500)).await;

    let stats = node1.heartbeat_stats_snapshot();
    // With 50-100ms latency and 50ms heartbeat interval, heartbeats will be slow
    // but should still attempt some
    assert!(
        stats.heartbeats_sent > 0 || stats.heartbeats_failed > 0,
        "Expected heartbeat activity even with high latency"
    );

    node1.stop_heartbeat_task();
}

#[tokio::test]
async fn test_message_timing_with_latency() {
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));

    // Set 20ms fixed latency
    transport1.set_latency(Some((20, 20)));

    transport1.connect_to("node2".to_string(), transport2.sender());

    // Measure send time
    let start = Instant::now();
    let msg = tensor_chain::Message::RequestVote(tensor_chain::RequestVote {
        term: 1,
        candidate_id: "node1".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: tensor_store::SparseVector::new(128),
    });

    use tensor_chain::Transport;
    let _ = transport1.send(&"node2".to_string(), msg).await;
    let elapsed = start.elapsed();

    // Should take at least 20ms due to latency
    assert!(
        elapsed >= Duration::from_millis(18), // Allow small margin
        "Expected >=18ms with 20ms latency, got {:?}",
        elapsed
    );
}

#[tokio::test]
async fn test_per_peer_latency() {
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));
    let transport3 = Arc::new(MemoryTransport::new("node3".to_string()));

    // Global latency: 10ms
    transport1.set_latency(Some((10, 10)));
    // Override for node3: 50ms (simulating farther node)
    transport1.set_peer_latency(&"node3".to_string(), (50, 50));

    transport1.connect_to("node2".to_string(), transport2.sender());
    transport1.connect_to("node3".to_string(), transport3.sender());

    let msg = tensor_chain::Message::RequestVote(tensor_chain::RequestVote {
        term: 1,
        candidate_id: "node1".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: tensor_store::SparseVector::new(128),
    });

    use tensor_chain::Transport;

    // Send to node2 - should use global 10ms latency
    let start = Instant::now();
    let _ = transport1.send(&"node2".to_string(), msg.clone()).await;
    let elapsed_node2 = start.elapsed();

    // Send to node3 - should use peer-specific 50ms latency
    let start = Instant::now();
    let _ = transport1.send(&"node3".to_string(), msg).await;
    let elapsed_node3 = start.elapsed();

    assert!(
        elapsed_node2 < elapsed_node3,
        "node2 ({:?}) should be faster than node3 ({:?})",
        elapsed_node2,
        elapsed_node3
    );
    assert!(
        elapsed_node3 >= Duration::from_millis(48),
        "node3 should have ~50ms latency, got {:?}",
        elapsed_node3
    );
}

#[tokio::test]
async fn test_latency_can_be_cleared() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));

    transport.connect_to("node2".to_string(), transport2.sender());

    // Set high latency
    transport.set_latency(Some((100, 100)));

    let msg = tensor_chain::Message::RequestVote(tensor_chain::RequestVote {
        term: 1,
        candidate_id: "node1".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: tensor_store::SparseVector::new(128),
    });

    use tensor_chain::Transport;

    // First send with latency
    let start = Instant::now();
    let _ = transport.send(&"node2".to_string(), msg.clone()).await;
    let with_latency = start.elapsed();

    // Clear latency
    transport.set_latency(None);

    // Send without latency
    let start = Instant::now();
    let _ = transport.send(&"node2".to_string(), msg).await;
    let without_latency = start.elapsed();

    assert!(
        with_latency > without_latency * 10,
        "With latency ({:?}) should be much slower than without ({:?})",
        with_latency,
        without_latency
    );
}

#[tokio::test]
async fn test_asymmetric_network_latency() {
    // Simulate asymmetric network where node1->node2 is fast but node2->node1 is slow
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));

    // node1 -> node2: 5ms
    transport1.set_latency(Some((5, 5)));
    // node2 -> node1: 50ms (asymmetric - maybe congested return path)
    transport2.set_latency(Some((50, 50)));

    transport1.connect_to("node2".to_string(), transport2.sender());
    transport2.connect_to("node1".to_string(), transport1.sender());

    let msg = tensor_chain::Message::RequestVote(tensor_chain::RequestVote {
        term: 1,
        candidate_id: "test".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: tensor_store::SparseVector::new(128),
    });

    use tensor_chain::Transport;

    // node1 -> node2 should be fast
    let start = Instant::now();
    let _ = transport1.send(&"node2".to_string(), msg.clone()).await;
    let forward = start.elapsed();

    // node2 -> node1 should be slow
    let start = Instant::now();
    let _ = transport2.send(&"node1".to_string(), msg).await;
    let backward = start.elapsed();

    assert!(
        backward > forward * 5,
        "Backward ({:?}) should be much slower than forward ({:?})",
        backward,
        forward
    );
}
