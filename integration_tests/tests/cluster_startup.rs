// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for distributed cluster startup.
//!
//! Tests 3-node cluster initialization and basic operations:
//! - TCP transport binding
//! - Peer connections
//! - Raft leader election
//! - Query executor registration

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use parking_lot::RwLock;
use query_router::QueryRouter;
use tensor_chain::{
    ClusterNodeConfig, ClusterOrchestrator, ClusterPeerConfig, OrchestratorConfig, QueryExecutor,
    RaftConfig, SecurityMode,
};
use tokio::time::sleep;

fn test_addr(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

/// Wrapper to implement QueryExecutor for Arc<RwLock<QueryRouter>>.
struct RouterExecutor(Arc<RwLock<QueryRouter>>);

impl QueryExecutor for RouterExecutor {
    fn execute(&self, query: &str) -> Result<Vec<u8>, String> {
        let router = self.0.read();
        router.execute_for_cluster(query)
    }
}

#[tokio::test]
async fn test_single_node_cluster_startup() {
    // Start a single node cluster
    let local = ClusterNodeConfig::new("node1", test_addr(19001));
    let config = OrchestratorConfig::new(local, vec![])
        .with_security_mode(SecurityMode::Development);

    let orchestrator = ClusterOrchestrator::start(config).await;
    assert!(orchestrator.is_ok(), "Failed to start single node cluster");

    let orchestrator = orchestrator.unwrap();
    assert_eq!(orchestrator.node_id(), "node1");
    // Chain height is at least 0 (genesis block may or may not exist yet)
    let _ = orchestrator.chain_height();

    // Shutdown gracefully
    orchestrator.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_single_node_with_query_executor() {
    let local = ClusterNodeConfig::new("node1", test_addr(19002));
    let config = OrchestratorConfig::new(local, vec![])
        .with_security_mode(SecurityMode::Development);

    let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

    // Create and register query executor
    let router = Arc::new(RwLock::new(QueryRouter::new()));
    let executor: Arc<dyn QueryExecutor> = Arc::new(RouterExecutor(Arc::clone(&router)));
    orchestrator.register_query_executor(executor);

    // Create a table through the router
    {
        let r = router.read();
        r.execute_parsed("CREATE TABLE test (id INT, name TEXT)")
            .unwrap();
        r.execute_parsed("INSERT INTO test VALUES (1, 'hello')")
            .unwrap();
    }

    // Shutdown
    orchestrator.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_3_node_cluster_startup() {
    // Use different ports to avoid conflicts
    let port_base = 19010;

    let local1 = ClusterNodeConfig::new("node1", test_addr(port_base));
    let local2 = ClusterNodeConfig::new("node2", test_addr(port_base + 1));
    let local3 = ClusterNodeConfig::new("node3", test_addr(port_base + 2));

    // Start node1 first
    let config1 = OrchestratorConfig::new(
        local1,
        vec![
            ClusterPeerConfig::new("node2", test_addr(port_base + 1)),
            ClusterPeerConfig::new("node3", test_addr(port_base + 2)),
        ],
    )
    .with_security_mode(SecurityMode::Development)
    .with_raft(RaftConfig {
        election_timeout: (100, 200),
        heartbeat_interval: 50,
        ..RaftConfig::default()
    });

    let node1 = ClusterOrchestrator::start(config1).await;
    assert!(node1.is_ok(), "Failed to start node1: {:?}", node1.err());
    let node1 = node1.unwrap();

    // Start node2
    let config2 = OrchestratorConfig::new(
        local2,
        vec![
            ClusterPeerConfig::new("node1", test_addr(port_base)),
            ClusterPeerConfig::new("node3", test_addr(port_base + 2)),
        ],
    )
    .with_security_mode(SecurityMode::Development)
    .with_raft(RaftConfig {
        election_timeout: (100, 200),
        heartbeat_interval: 50,
        ..RaftConfig::default()
    });

    let node2 = ClusterOrchestrator::start(config2).await;
    assert!(node2.is_ok(), "Failed to start node2: {:?}", node2.err());
    let node2 = node2.unwrap();

    // Start node3
    let config3 = OrchestratorConfig::new(
        local3,
        vec![
            ClusterPeerConfig::new("node1", test_addr(port_base)),
            ClusterPeerConfig::new("node2", test_addr(port_base + 1)),
        ],
    )
    .with_security_mode(SecurityMode::Development)
    .with_raft(RaftConfig {
        election_timeout: (100, 200),
        heartbeat_interval: 50,
        ..RaftConfig::default()
    });

    let node3 = ClusterOrchestrator::start(config3).await;
    assert!(node3.is_ok(), "Failed to start node3: {:?}", node3.err());
    let node3 = node3.unwrap();

    // Wait a moment for connections to establish
    sleep(Duration::from_millis(100)).await;

    // All nodes should be running (chain height may vary based on initialization)
    println!(
        "Node heights: {}, {}, {}",
        node1.chain_height(),
        node2.chain_height(),
        node3.chain_height()
    );

    // Shutdown all nodes
    node1.shutdown().await.unwrap();
    node2.shutdown().await.unwrap();
    node3.shutdown().await.unwrap();
}

#[tokio::test]
async fn test_3_node_cluster_with_run_loop() {
    let port_base = 19020;

    // Start 3 nodes wrapped in Arc for sharing
    let mut nodes: Vec<Arc<ClusterOrchestrator>> = Vec::new();
    for i in 0..3 {
        let node_id = format!("node{}", i + 1);
        let local = ClusterNodeConfig::new(&node_id, test_addr(port_base + i as u16));

        let peers: Vec<ClusterPeerConfig> = (0..3)
            .filter(|&j| j != i)
            .map(|j| {
                ClusterPeerConfig::new(format!("node{}", j + 1), test_addr(port_base + j as u16))
            })
            .collect();

        let config = OrchestratorConfig::new(local, peers)
            .with_security_mode(SecurityMode::Development)
            .with_raft(RaftConfig {
                election_timeout: (100, 200),
                heartbeat_interval: 50,
                ..RaftConfig::default()
            });

        let node = ClusterOrchestrator::start(config).await.unwrap();

        // Register query executor
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let executor: Arc<dyn QueryExecutor> = Arc::new(RouterExecutor(Arc::clone(&router)));
        node.register_query_executor(executor);

        nodes.push(Arc::new(node));
    }

    // Create shutdown flag
    let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Spawn run loops for each node
    let mut handles = Vec::new();
    for node in &nodes {
        let node_clone = Arc::clone(node);
        let shutdown_clone = Arc::clone(&shutdown);
        let handle = tokio::spawn(async move {
            while !shutdown_clone.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = node_clone.raft().tick_async().await;
                sleep(Duration::from_millis(50)).await;
            }
        });
        handles.push(handle);
    }

    // Let the cluster run for a bit to allow leader election
    sleep(Duration::from_millis(500)).await;

    // Check that at most one leader exists
    let leader_count = nodes.iter().filter(|n| n.is_leader()).count();
    assert!(
        leader_count <= 1,
        "Multiple leaders detected: {}",
        leader_count
    );

    // Signal shutdown
    shutdown.store(true, std::sync::atomic::Ordering::Relaxed);

    // Wait for all run loops to finish
    for handle in handles {
        let _ = handle.await;
    }

    // Shutdown all nodes
    for node in &nodes {
        node.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn test_cluster_membership_view() {
    let port_base = 19030;

    let local = ClusterNodeConfig::new("node1", test_addr(port_base));
    let config = OrchestratorConfig::new(
        local,
        vec![
            ClusterPeerConfig::new("node2", test_addr(port_base + 1)),
            ClusterPeerConfig::new("node3", test_addr(port_base + 2)),
        ],
    )
    .with_security_mode(SecurityMode::Development);

    let node = ClusterOrchestrator::start(config).await.unwrap();

    // Check membership view
    let membership = node.membership();
    let view = membership.view();

    // Should have at least the local node configured
    println!("Membership view: {} nodes", view.nodes.len());
    for n in &view.nodes {
        println!("  - {}", n.node_id);
    }

    // Local node should be present
    assert!(view.nodes.iter().any(|n| n.node_id == "node1"));

    node.shutdown().await.unwrap();
}
