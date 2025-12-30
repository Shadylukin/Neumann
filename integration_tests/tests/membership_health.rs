//! Integration tests for cluster membership health tracking.
//!
//! Tests:
//! - Health check success/failure detection
//! - Node failure detection via health checks
//! - Health state transitions and callbacks
//! - Membership with Raft voting integration

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tensor_chain::{
    MemoryTransport, Message, MembershipCallback, MembershipManager, NodeHealth,
    RaftConfig, RaftNode,
};
use tensor_chain::membership::{ClusterConfig, ClusterView, HealthConfig, LocalNodeConfig};
use tensor_chain::network::RequestVote;

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

struct TestCallback {
    health_changes: Mutex<Vec<(String, NodeHealth, NodeHealth)>>,
    view_changes: AtomicUsize,
}

impl TestCallback {
    fn new() -> Self {
        Self {
            health_changes: Mutex::new(Vec::new()),
            view_changes: AtomicUsize::new(0),
        }
    }

    fn health_change_count(&self) -> usize {
        self.health_changes.lock().len()
    }

    fn view_change_count(&self) -> usize {
        self.view_changes.load(Ordering::SeqCst)
    }

    fn last_change(&self) -> Option<(String, NodeHealth, NodeHealth)> {
        self.health_changes.lock().last().cloned()
    }
}

impl MembershipCallback for TestCallback {
    fn on_health_change(
        &self,
        node_id: &String,
        old_health: NodeHealth,
        new_health: NodeHealth,
    ) {
        self.health_changes
            .lock()
            .push((node_id.clone(), old_health, new_health));
    }

    fn on_view_change(&self, _view: &ClusterView) {
        self.view_changes.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn test_membership_manager_creation() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);

    let manager = MembershipManager::new(config, transport);

    assert_eq!(manager.local_id(), "node1");
    assert_eq!(manager.cluster_id(), "test-cluster");
    assert_eq!(manager.peer_ids().len(), 1);
}

#[tokio::test]
async fn test_membership_initial_state_unknown() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);

    let manager = MembershipManager::new(config, transport);

    // All nodes start as Unknown
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Unknown);
}

#[tokio::test]
async fn test_membership_mark_healthy() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);

    let manager = MembershipManager::new(config, transport);
    let callback = Arc::new(TestCallback::new());
    manager.register_callback(callback.clone());

    // Mark node as healthy
    manager.mark_healthy(&"node2".to_string());

    // Verify transition
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Healthy);
    assert_eq!(status.consecutive_failures, 0);

    // Callback should have been invoked
    assert_eq!(callback.health_change_count(), 1);
    let change = callback.last_change().unwrap();
    assert_eq!(change.0, "node2");
    assert_eq!(change.1, NodeHealth::Unknown);
    assert_eq!(change.2, NodeHealth::Healthy);
}

#[tokio::test]
async fn test_membership_mark_failed() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);

    let manager = MembershipManager::new(config, transport);
    let callback = Arc::new(TestCallback::new());
    manager.register_callback(callback.clone());

    // Mark node as failed
    manager.mark_failed(&"node2".to_string());

    // Verify transition
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Failed);

    // Callback should have been invoked
    assert_eq!(callback.health_change_count(), 1);
    let change = callback.last_change().unwrap();
    assert_eq!(change.0, "node2");
    assert_eq!(change.1, NodeHealth::Unknown);
    assert_eq!(change.2, NodeHealth::Failed);
}

#[tokio::test]
async fn test_membership_detects_node_failure() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[
        ("node2", "127.0.0.1:9101"),
        ("node3", "127.0.0.1:9102"),
    ]);

    let manager = MembershipManager::new(config, transport);
    let callback = Arc::new(TestCallback::new());
    manager.register_callback(callback.clone());

    // Initially mark nodes as healthy
    manager.mark_healthy(&"node2".to_string());
    manager.mark_healthy(&"node3".to_string());

    let view = manager.view();
    assert_eq!(view.healthy_count(), 2);

    // Simulate node2 failure
    manager.mark_failed(&"node2".to_string());

    let view = manager.view();
    assert_eq!(view.healthy_count(), 1);
    assert!(!view.is_healthy(&"node2".to_string()));
    assert!(view.is_healthy(&"node3".to_string()));

    // Verify failure was detected
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Failed);

    // Callbacks should have been invoked
    assert!(callback.health_change_count() >= 1);
}

#[tokio::test]
async fn test_membership_recovery_on_rejoin() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);

    let manager = MembershipManager::new(config, transport);
    let callback = Arc::new(TestCallback::new());
    manager.register_callback(callback.clone());

    // First mark as failed
    manager.mark_failed(&"node2".to_string());
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Failed);

    // Then mark as healthy (simulating rejoin)
    manager.mark_healthy(&"node2".to_string());
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Healthy);
    assert_eq!(status.consecutive_failures, 0);

    // Callbacks should have been invoked for both transitions
    let changes = callback.health_changes.lock();
    assert!(changes.len() >= 2);
}

#[tokio::test]
async fn test_membership_run_and_shutdown() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);

    let manager = Arc::new(MembershipManager::new(config, transport));

    // Start the run loop in a background task
    let manager_clone = manager.clone();
    let run_handle = tokio::spawn(async move {
        manager_clone.run().await
    });

    // Let it run for a bit
    tokio::time::sleep(Duration::from_millis(100)).await;

    assert!(manager.is_running());

    // Shutdown
    manager.shutdown();

    // run() should complete
    let result = tokio::time::timeout(Duration::from_secs(1), run_handle).await;
    assert!(result.is_ok());
    assert!(!manager.is_running());
}

#[tokio::test]
async fn test_membership_cluster_view_generation() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[
        ("node2", "127.0.0.1:9101"),
        ("node3", "127.0.0.1:9102"),
    ]);

    let manager = MembershipManager::new(config, transport);

    let initial_gen = manager.view().generation;

    // Mark a node failed - should increment generation
    manager.mark_failed(&"node2".to_string());
    let gen_after_fail = manager.view().generation;
    assert!(gen_after_fail > initial_gen);

    // Mark a node healthy - should increment generation again
    manager.mark_healthy(&"node2".to_string());
    let gen_after_recover = manager.view().generation;
    assert!(gen_after_recover > gen_after_fail);
}

#[tokio::test]
async fn test_raft_membership_integration_unhealthy_candidate() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[
        ("node2", "127.0.0.1:9101"),
        ("node3", "127.0.0.1:9102"),
    ]);

    let membership = Arc::new(MembershipManager::new(config, transport.clone()));

    // Mark node2 as failed
    membership.mark_failed(&"node2".to_string());

    // Create Raft node with membership manager
    let raft = RaftNode::with_membership(
        "node1".to_string(),
        vec!["node2".to_string(), "node3".to_string()],
        transport,
        RaftConfig::default(),
        membership,
    );

    // RequestVote from unhealthy node2 should be rejected
    let rv = RequestVote {
        term: 1,
        candidate_id: "node2".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: vec![],
    };

    let response = raft.handle_message(&"node2".to_string(), &Message::RequestVote(rv));

    if let Some(Message::RequestVoteResponse(rvr)) = response {
        assert!(!rvr.vote_granted, "Should not vote for unhealthy candidate");
    } else {
        panic!("Expected RequestVoteResponse");
    }
}

#[tokio::test]
async fn test_raft_membership_integration_healthy_candidate() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[
        ("node2", "127.0.0.1:9101"),
    ]);

    let membership = Arc::new(MembershipManager::new(config, transport.clone()));

    // Mark node2 as healthy
    membership.mark_healthy(&"node2".to_string());

    // Create Raft node with membership manager
    let raft = RaftNode::with_membership(
        "node1".to_string(),
        vec!["node2".to_string()],
        transport,
        RaftConfig::default(),
        membership,
    );

    // RequestVote from healthy node2 should be accepted
    let rv = RequestVote {
        term: 1,
        candidate_id: "node2".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: vec![],
    };

    let response = raft.handle_message(&"node2".to_string(), &Message::RequestVote(rv));

    if let Some(Message::RequestVoteResponse(rvr)) = response {
        assert!(rvr.vote_granted, "Should vote for healthy candidate");
    } else {
        panic!("Expected RequestVoteResponse");
    }
}

#[tokio::test]
async fn test_membership_health_state_transitions() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);

    let manager = MembershipManager::new(config, transport);

    // Initial state is Unknown
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Unknown);

    // Transition to Healthy
    manager.mark_healthy(&"node2".to_string());
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Healthy);

    // Transition to Failed
    manager.mark_failed(&"node2".to_string());
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Failed);

    // Back to Healthy (recovery)
    manager.mark_healthy(&"node2".to_string());
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Healthy);
}

#[tokio::test]
async fn test_membership_no_duplicate_transitions() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);

    let manager = MembershipManager::new(config, transport);
    let callback = Arc::new(TestCallback::new());
    manager.register_callback(callback.clone());

    // Mark as failed twice
    manager.mark_failed(&"node2".to_string());
    let count_after_first = callback.health_change_count();

    manager.mark_failed(&"node2".to_string());
    let count_after_second = callback.health_change_count();

    // Should not generate duplicate callback
    assert_eq!(count_after_first, count_after_second);
}

#[tokio::test]
async fn test_membership_multiple_peers() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[
        ("node2", "127.0.0.1:9101"),
        ("node3", "127.0.0.1:9102"),
        ("node4", "127.0.0.1:9103"),
    ]);

    let manager = MembershipManager::new(config, transport);

    assert_eq!(manager.peer_ids().len(), 3);

    // Mark all healthy
    manager.mark_healthy(&"node2".to_string());
    manager.mark_healthy(&"node3".to_string());
    manager.mark_healthy(&"node4".to_string());

    let view = manager.view();
    assert_eq!(view.total_count(), 3);
    assert_eq!(view.healthy_count(), 3);

    // Fail one node
    manager.mark_failed(&"node3".to_string());

    let view = manager.view();
    assert_eq!(view.healthy_count(), 2);
    assert_eq!(view.failed_nodes.len(), 1);
    assert!(view.failed_nodes.contains(&"node3".to_string()));
}

#[tokio::test]
async fn test_membership_cluster_view_contents() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[
        ("node2", "127.0.0.1:9101"),
        ("node3", "127.0.0.1:9102"),
    ]);

    let manager = MembershipManager::new(config, transport);

    // Set different states
    manager.mark_healthy(&"node2".to_string());
    manager.mark_failed(&"node3".to_string());

    let view = manager.view();

    // Check view contents
    assert_eq!(view.total_count(), 2);
    assert_eq!(view.healthy_count(), 1);
    assert!(view.is_healthy(&"node2".to_string()));
    assert!(!view.is_healthy(&"node3".to_string()));
    assert_eq!(view.healthy_nodes.len(), 1);
    assert_eq!(view.failed_nodes.len(), 1);
}

#[tokio::test]
async fn test_membership_node_status_not_found() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = create_test_cluster_config("node1", &[("node2", "127.0.0.1:9101")]);

    let manager = MembershipManager::new(config, transport);

    // Non-existent node should return None
    let status = manager.node_status(&"nonexistent".to_string());
    assert!(status.is_none());
}
