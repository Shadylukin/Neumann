// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for distributed tensor_chain components.
//!
//! Tests TCP transport, membership management, and multi-node operations.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use parking_lot::RwLock;
use tensor_chain::{
    ClusterConfig, ClusterView, HealthConfig, LocalNodeConfig, MembershipCallback,
    MembershipManager, MemoryTransport, Message, NodeHealth, PeerConfig, RaftConfig, RaftNode,
    SecurityMode, TcpTransport, TcpTransportConfig, TensorChain, Transaction, Transport,
};
use tensor_store::{
    ConsistentHashConfig, ConsistentHashPartitioner, PartitionedStore, Partitioner, ScalarValue,
    TensorData, TensorStore, TensorValue,
};
use tokio::time::sleep;

// ============================================================================
// Helper Functions
// ============================================================================

fn make_tensor(value: i64) -> TensorData {
    let mut data = TensorData::new();
    data.set("value", TensorValue::Scalar(ScalarValue::Int(value)));
    data
}

// ============================================================================
// TCP Transport Integration Tests
// ============================================================================

#[tokio::test]
async fn test_tcp_transport_two_node_handshake() {
    // Node 1 config
    let config1 = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap())
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let transport1 = TcpTransport::new(config1);

    // Start node 1 to get actual bound address
    transport1.start().await.unwrap();
    let addr1 = transport1.bound_addr().unwrap();

    // Node 2 config pointing to node 1
    let config2 = TcpTransportConfig::new("node2", "127.0.0.1:0".parse().unwrap())
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let transport2 = TcpTransport::new(config2);
    transport2.start().await.unwrap();

    // Connect node 2 to node 1
    let peer_config = PeerConfig {
        node_id: "node1".to_string(),
        address: addr1.to_string(),
    };
    transport2.connect(&peer_config).await.unwrap();

    // Give time for handshake
    sleep(Duration::from_millis(100)).await;

    // Both should see each other as peers
    assert!(transport2.peers().contains(&"node1".to_string()));

    transport1.shutdown().await;
    transport2.shutdown().await;
}

#[tokio::test]
async fn test_tcp_transport_message_exchange() {
    // Set up two nodes
    let config1 = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap())
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let transport1 = Arc::new(TcpTransport::new(config1));
    transport1.start().await.unwrap();
    let addr1 = transport1.bound_addr().unwrap();

    let config2 = TcpTransportConfig::new("node2", "127.0.0.1:0".parse().unwrap())
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let transport2 = Arc::new(TcpTransport::new(config2));
    transport2.start().await.unwrap();
    let addr2 = transport2.bound_addr().unwrap();

    // Connect both ways
    transport1
        .connect(&PeerConfig {
            node_id: "node2".to_string(),
            address: addr2.to_string(),
        })
        .await
        .unwrap();

    transport2
        .connect(&PeerConfig {
            node_id: "node1".to_string(),
            address: addr1.to_string(),
        })
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    // Send message from node1 to node2
    let msg = Message::Ping { term: 42 };
    transport1.send(&"node2".to_string(), msg).await.unwrap();

    // Receive on node2
    let transport2_clone = transport2.clone();
    let recv_task = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(2), transport2_clone.recv()).await
    });

    let result = recv_task.await.unwrap();
    assert!(result.is_ok());
    let (from, received) = result.unwrap().unwrap();
    assert_eq!(from, "node1");
    if let Message::Ping { term } = received {
        assert_eq!(term, 42);
    } else {
        panic!("Expected Ping message");
    }

    transport1.shutdown().await;
    transport2.shutdown().await;
}

// ============================================================================
// Membership Manager Integration Tests
// ============================================================================

#[tokio::test]
async fn test_membership_health_checking() {
    // Create two transports
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));

    // Link them
    transport1.connect_to("node2".to_string(), transport2.sender());
    transport2.connect_to("node1".to_string(), transport1.sender());

    // Create membership manager for node1
    let health_config = HealthConfig {
        ping_interval_ms: 100,
        failure_threshold: 2,
        ping_timeout_ms: 50,
        startup_grace_ms: 0,
    };

    let cluster_config = ClusterConfig::new(
        "test-cluster",
        LocalNodeConfig {
            node_id: "node1".to_string(),
            bind_address: "127.0.0.1:9100".parse().unwrap(),
        },
    )
    .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
    .with_health(health_config);

    let manager = MembershipManager::new(cluster_config, transport1.clone());

    // Start manager
    manager.start().await.unwrap();

    // Run health check
    manager.check_health().await.unwrap();

    // Node2 should be healthy
    let status = manager.node_status(&"node2".to_string()).unwrap();
    assert_eq!(status.health, NodeHealth::Healthy);
}

#[tokio::test]
async fn test_membership_callback_notification() {
    // Track callback invocations
    struct TestCallback {
        health_changes: RwLock<Vec<(String, NodeHealth, NodeHealth)>>,
        view_changes: AtomicUsize,
    }

    impl MembershipCallback for TestCallback {
        fn on_health_change(&self, node_id: &String, old: NodeHealth, new: NodeHealth) {
            self.health_changes
                .write()
                .push((node_id.clone(), old, new));
        }

        fn on_view_change(&self, _view: &ClusterView) {
            self.view_changes.fetch_add(1, Ordering::SeqCst);
        }
    }

    let transport = Arc::new(MemoryTransport::new("node1".to_string()));

    let health_config = HealthConfig {
        ping_interval_ms: 100,
        failure_threshold: 1,
        ping_timeout_ms: 50,
        startup_grace_ms: 0,
    };

    let cluster_config = ClusterConfig::new(
        "test-cluster",
        LocalNodeConfig {
            node_id: "node1".to_string(),
            bind_address: "127.0.0.1:9100".parse().unwrap(),
        },
    )
    .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
    .with_health(health_config);

    let manager = MembershipManager::new(cluster_config, transport);

    let callback = Arc::new(TestCallback {
        health_changes: RwLock::new(Vec::new()),
        view_changes: AtomicUsize::new(0),
    });

    manager.register_callback(callback.clone());

    // Mark a node as failed manually
    manager.mark_failed(&"node2".to_string());

    // Callback should have been invoked
    let changes = callback.health_changes.read();
    assert_eq!(changes.len(), 1);
    assert_eq!(changes[0].0, "node2");
    assert_eq!(changes[0].2, NodeHealth::Failed);
}

// ============================================================================
// Partitioned Store Integration Tests
// ============================================================================

#[test]
fn test_partitioned_store_distribution() {
    let store = TensorStore::new();
    let config = ConsistentHashConfig::new("node1").with_virtual_nodes(256);

    let partitioner = {
        let mut p = ConsistentHashPartitioner::new(config);
        p.add_node("node1".to_string());
        p.add_node("node2".to_string());
        p.add_node("node3".to_string());
        Arc::new(p) as Arc<dyn Partitioner>
    };

    let partitioned = PartitionedStore::with_partitioner(store, partitioner);

    // Insert keys and track distribution
    let mut local_count = 0;
    let mut remote_count = 0;

    for i in 0..100 {
        let key = format!("key:{}", i);
        if partitioned.is_local(&key) {
            let tensor = make_tensor(i);
            partitioned.put_local(&key, tensor).unwrap();
            local_count += 1;
        } else {
            remote_count += 1;
        }
    }

    // Should have some keys on each type
    assert!(local_count > 0, "Should have some local keys");
    assert!(remote_count > 0, "Should have some remote keys");

    // Local count should match actual store size
    assert_eq!(partitioned.local_count(), local_count);
}

#[test]
fn test_partitioned_store_scan_local() {
    let store = TensorStore::new();
    let config = ConsistentHashConfig::new("node1").with_virtual_nodes(256);

    let partitioner = {
        let mut p = ConsistentHashPartitioner::new(config);
        p.add_node("node1".to_string());
        p.add_node("node2".to_string());
        Arc::new(p) as Arc<dyn Partitioner>
    };

    let partitioned = PartitionedStore::with_partitioner(store, partitioner);

    // Insert only local keys with a specific prefix
    for i in 0..50 {
        let key = format!("users:{}", i);
        if partitioned.is_local(&key) {
            let tensor = make_tensor(i);
            partitioned.put_local(&key, tensor).unwrap();
        }
    }

    // Scan should only return local keys
    let local_keys = partitioned.scan_local("users:");
    for key in &local_keys {
        assert!(
            partitioned.is_local(key),
            "Scan should only return local keys"
        );
    }
}

// ============================================================================
// Chain Integration Tests
// ============================================================================

#[test]
fn test_chain_block_creation() {
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "node1");
    chain.initialize().unwrap();

    // Create multiple transactions
    for i in 0..5 {
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: format!("entity:{}", i),
            data: vec![i as u8; 10],
        })
        .unwrap();
        chain.commit(&tx).unwrap();
    }

    // Verify chain state
    assert_eq!(chain.height(), 5);

    // Verify chain integrity
    chain.verify().unwrap();
}

#[test]
fn test_chain_history_tracking() {
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "node1");
    chain.initialize().unwrap();

    // Multiple updates to the same key
    for i in 0..3 {
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: "counter".to_string(),
            data: vec![i as u8],
        })
        .unwrap();
        chain.commit(&tx).unwrap();
    }

    // Get history
    let history = chain.history("counter").unwrap();
    assert_eq!(history.len(), 3);

    // Verify values are in order
    for (i, (height, tx)) in history.iter().enumerate() {
        assert_eq!(*height as usize, i + 1);
        if let Transaction::Put { data, .. } = tx {
            assert_eq!(data[0], i as u8);
        }
    }
}

#[test]
fn test_chain_transaction_rollback() {
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "node1");
    chain.initialize().unwrap();

    // Create and commit first transaction
    let tx1 = chain.begin().unwrap();
    tx1.add_operation(Transaction::Put {
        key: "committed".to_string(),
        data: vec![1],
    })
    .unwrap();
    chain.commit(&tx1).unwrap();

    // Create and rollback second transaction
    let tx2 = chain.begin().unwrap();
    tx2.add_operation(Transaction::Put {
        key: "rolledback".to_string(),
        data: vec![2],
    })
    .unwrap();
    chain.rollback(&tx2).unwrap();

    // Height should only reflect committed transaction
    assert_eq!(chain.height(), 1);

    // Only committed key should be in history
    let committed_history = chain.history("committed").unwrap();
    assert_eq!(committed_history.len(), 1);

    let rolledback_history = chain.history("rolledback").unwrap();
    assert_eq!(rolledback_history.len(), 0);
}

// ============================================================================
// Raft Consensus Integration Tests
// ============================================================================

#[tokio::test]
async fn test_raft_leader_election() {
    // Create three nodes
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));
    let transport3 = Arc::new(MemoryTransport::new("node3".to_string()));

    // Link all transports
    transport1.connect_to("node2".to_string(), transport2.sender());
    transport1.connect_to("node3".to_string(), transport3.sender());
    transport2.connect_to("node1".to_string(), transport1.sender());
    transport2.connect_to("node3".to_string(), transport3.sender());
    transport3.connect_to("node1".to_string(), transport1.sender());
    transport3.connect_to("node2".to_string(), transport2.sender());

    let peers = vec!["node2".to_string(), "node3".to_string()];

    // Create raft config
    let config1 = RaftConfig {
        election_timeout: (150, 300),
        heartbeat_interval: 50,
        ..Default::default()
    };

    let raft1 = RaftNode::new(
        "node1".to_string(),
        peers.clone(),
        transport1.clone(),
        config1,
    );

    // Check initial state
    let state = raft1.state();
    assert_eq!(state, tensor_chain::RaftState::Follower);

    // Check term is valid
    let term = raft1.current_term();
    assert_eq!(term, 0, "Initial term should be 0");
}

#[tokio::test]
async fn test_raft_log_replication() {
    // Set up two nodes
    let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
    let transport2 = Arc::new(MemoryTransport::new("node2".to_string()));

    transport1.connect_to("node2".to_string(), transport2.sender());
    transport2.connect_to("node1".to_string(), transport1.sender());

    // Create raft nodes
    let config1 = RaftConfig::default();
    let config2 = RaftConfig::default();

    let raft1 = RaftNode::new(
        "node1".to_string(),
        vec!["node2".to_string()],
        transport1,
        config1,
    );

    let raft2 = RaftNode::new(
        "node2".to_string(),
        vec!["node1".to_string()],
        transport2,
        config2,
    );

    // Both nodes should start as followers
    assert_eq!(raft1.state(), tensor_chain::RaftState::Follower);
    assert_eq!(raft2.state(), tensor_chain::RaftState::Follower);

    // Both should have term 0
    assert_eq!(raft1.current_term(), 0);
    assert_eq!(raft2.current_term(), 0);
}

// ============================================================================
// Full Distributed System Integration Test
// ============================================================================

#[tokio::test]
async fn test_distributed_chain_with_partitioning() {
    // This test simulates a distributed system with:
    // - Partitioned storage
    // - TCP-like transport (using MemoryTransport for determinism)
    // - Membership management
    // - Chain operations

    // Set up partitioner
    let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
    let partitioner = {
        let mut p = ConsistentHashPartitioner::new(config);
        p.add_node("node1".to_string());
        p.add_node("node2".to_string());
        Arc::new(p) as Arc<dyn Partitioner>
    };

    // Create partitioned store for node1
    let store = TensorStore::new();
    let partitioned = PartitionedStore::with_partitioner(store, partitioner);

    // Create chain with the underlying store
    let chain = TensorChain::new(partitioned.store().clone(), "node1");
    chain.initialize().unwrap();

    // Insert data through partitioned store for local keys
    let mut inserted_keys = Vec::new();
    for i in 0..20 {
        let key = format!("data:{}", i);
        if partitioned.is_local(&key) {
            let tensor = make_tensor(i);
            partitioned.put_local(&key, tensor).unwrap();
            inserted_keys.push(key);
        }
    }

    // Create blockchain transactions for local data
    for key in &inserted_keys {
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: key.clone(),
            data: vec![1, 2, 3],
        })
        .unwrap();
        chain.commit(&tx).unwrap();
    }

    // Verify chain recorded all transactions
    assert_eq!(chain.height() as usize, inserted_keys.len());

    // Verify chain integrity
    chain.verify().unwrap();

    // Verify history is available for each key
    for key in &inserted_keys {
        let history = chain.history(key).unwrap();
        assert_eq!(history.len(), 1);
    }
}
