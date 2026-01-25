//! Integration tests for the gossip-based membership protocol.
//!
//! Tests:
//! - LWW CRDT merge semantics
//! - Gossip message propagation
//! - Suspicion and failure detection
//! - Membership convergence

use std::sync::Arc;

use tensor_chain::{
    GossipConfig, GossipMembershipManager, GossipMessage, GossipNodeState, LWWMembershipState,
    MemoryTransport, NodeHealth,
};

#[test]
fn test_lww_crdt_merge_concurrent_updates() {
    let mut state1 = LWWMembershipState::new();
    let mut state2 = LWWMembershipState::new();

    // Node1 sees nodeA as healthy at t=1
    state1.merge(&[GossipNodeState::new(
        "nodeA".to_string(),
        NodeHealth::Healthy,
        1,
        0,
    )]);

    // Node2 sees nodeA as failed at t=2 (later)
    state2.merge(&[GossipNodeState::new(
        "nodeA".to_string(),
        NodeHealth::Failed,
        2,
        0,
    )]);

    // Node1 receives node2's state - should update to failed (t=2 > t=1)
    let changed = state1.merge(&[GossipNodeState::new(
        "nodeA".to_string(),
        NodeHealth::Failed,
        2,
        0,
    )]);

    assert_eq!(changed.len(), 1);
    assert_eq!(changed[0], "nodeA");
    assert_eq!(
        state1.get(&"nodeA".to_string()).unwrap().health,
        NodeHealth::Failed
    );
}

#[test]
fn test_lww_crdt_incarnation_supersedes_timestamp() {
    let mut state = LWWMembershipState::new();

    // Old update with high timestamp
    state.merge(&[GossipNodeState::new(
        "nodeA".to_string(),
        NodeHealth::Healthy,
        100,
        0,
    )]);

    // New update with lower timestamp but higher incarnation
    let changed = state.merge(&[GossipNodeState::new(
        "nodeA".to_string(),
        NodeHealth::Failed,
        50,
        1, // Higher incarnation
    )]);

    assert_eq!(changed.len(), 1);
    assert_eq!(
        state.get(&"nodeA".to_string()).unwrap().health,
        NodeHealth::Failed
    );
    assert_eq!(state.get(&"nodeA".to_string()).unwrap().incarnation, 1);
}

#[test]
fn test_gossip_message_roundtrip() {
    let messages = vec![
        GossipMessage::Sync {
            sender: "node1".to_string(),
            states: vec![
                GossipNodeState::new("node2".to_string(), NodeHealth::Healthy, 1, 0),
                GossipNodeState::new("node3".to_string(), NodeHealth::Degraded, 2, 1),
            ],
            sender_time: 100,
        },
        GossipMessage::Suspect {
            reporter: "node1".to_string(),
            suspect: "node2".to_string(),
            incarnation: 5,
        },
        GossipMessage::Alive {
            node_id: "node2".to_string(),
            incarnation: 6,
        },
        GossipMessage::PingReq {
            origin: "node1".to_string(),
            target: "node3".to_string(),
            sequence: 42,
        },
        GossipMessage::PingAck {
            origin: "node2".to_string(),
            target: "node3".to_string(),
            sequence: 42,
            success: true,
        },
    ];

    for msg in messages {
        let serialized = bitcode::serialize(&msg).unwrap();
        let deserialized: GossipMessage = bitcode::deserialize(&serialized).unwrap();
        assert_eq!(msg, deserialized);
    }
}

#[test]
fn test_gossip_manager_peer_tracking() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = GossipConfig {
        fanout: 2,
        gossip_interval_ms: 100,
        suspicion_timeout_ms: 1000,
        max_states_per_message: 10,
        geometric_routing: false,
        indirect_ping_count: 2,
        indirect_ping_timeout_ms: 200,
        require_signatures: false,
        max_message_age_ms: 300_000,
    };

    let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

    // Add peers
    manager.add_peer("node2".to_string());
    manager.add_peer("node3".to_string());
    manager.add_peer("node4".to_string());

    // Should have 4 nodes (self + 3 peers)
    assert_eq!(manager.node_count(), 4);

    // Self should be healthy
    let self_state = manager.node_state(&"node1".to_string()).unwrap();
    assert_eq!(self_state.health, NodeHealth::Healthy);

    // Peers should be unknown initially
    let peer_state = manager.node_state(&"node2".to_string()).unwrap();
    assert_eq!(peer_state.health, NodeHealth::Unknown);
}

#[test]
fn test_gossip_handle_sync_updates_state() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = GossipConfig::default();

    let manager = GossipMembershipManager::new("node1".to_string(), config, transport);
    manager.add_peer("node2".to_string());

    // Simulate receiving sync from node2
    let sync_msg = GossipMessage::Sync {
        sender: "node2".to_string(),
        states: vec![
            GossipNodeState::new("node2".to_string(), NodeHealth::Healthy, 10, 0),
            GossipNodeState::new("node3".to_string(), NodeHealth::Healthy, 10, 0),
        ],
        sender_time: 10,
    };

    manager.handle_gossip(sync_msg);

    // node2 should now be healthy
    assert_eq!(
        manager.node_state(&"node2".to_string()).unwrap().health,
        NodeHealth::Healthy
    );

    // node3 should be added and healthy
    assert_eq!(manager.node_count(), 3);
    assert_eq!(
        manager.node_state(&"node3".to_string()).unwrap().health,
        NodeHealth::Healthy
    );
}

#[test]
fn test_gossip_suspicion_and_refutation() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = GossipConfig {
        suspicion_timeout_ms: 10000, // Long timeout for testing
        ..GossipConfig::default()
    };

    let manager = GossipMembershipManager::new("node1".to_string(), config, transport);
    manager.add_peer("node2".to_string());

    // First mark node2 as healthy with incarnation 0
    manager.handle_gossip(GossipMessage::Sync {
        sender: "node2".to_string(),
        states: vec![GossipNodeState::new(
            "node2".to_string(),
            NodeHealth::Healthy,
            1,
            0,
        )],
        sender_time: 1,
    });

    assert_eq!(
        manager.node_state(&"node2".to_string()).unwrap().health,
        NodeHealth::Healthy
    );

    // Suspect node2
    manager.handle_gossip(GossipMessage::Suspect {
        reporter: "node3".to_string(),
        suspect: "node2".to_string(),
        incarnation: 0,
    });

    // Node2 should be degraded (suspected)
    assert_eq!(
        manager.node_state(&"node2".to_string()).unwrap().health,
        NodeHealth::Degraded
    );

    // Node2 refutes with higher incarnation
    manager.handle_gossip(GossipMessage::Alive {
        node_id: "node2".to_string(),
        incarnation: 1,
    });

    // Node2 should be healthy again
    assert_eq!(
        manager.node_state(&"node2".to_string()).unwrap().health,
        NodeHealth::Healthy
    );
    assert_eq!(
        manager
            .node_state(&"node2".to_string())
            .unwrap()
            .incarnation,
        1
    );
}

#[test]
fn test_gossip_states_for_gossip_limits_and_orders() {
    let mut state = LWWMembershipState::new();

    // Add 10 nodes with different timestamps
    for i in 0..10 {
        state.merge(&[GossipNodeState::new(
            format!("node{}", i),
            NodeHealth::Healthy,
            i as u64,
            0,
        )]);
    }

    // Request only 5 states
    let gossip_states = state.states_for_gossip(5);
    assert_eq!(gossip_states.len(), 5);

    // Should be ordered by timestamp descending (most recent first)
    for i in 1..gossip_states.len() {
        assert!(gossip_states[i - 1].timestamp >= gossip_states[i].timestamp);
    }
}

#[test]
fn test_gossip_health_counts() {
    let mut state = LWWMembershipState::new();

    state.merge(&[
        GossipNodeState::new("healthy1".to_string(), NodeHealth::Healthy, 1, 0),
        GossipNodeState::new("healthy2".to_string(), NodeHealth::Healthy, 2, 0),
        GossipNodeState::new("healthy3".to_string(), NodeHealth::Healthy, 3, 0),
        GossipNodeState::new("degraded1".to_string(), NodeHealth::Degraded, 4, 0),
        GossipNodeState::new("degraded2".to_string(), NodeHealth::Degraded, 5, 0),
        GossipNodeState::new("failed1".to_string(), NodeHealth::Failed, 6, 0),
    ]);

    let (healthy, degraded, failed) = state.count_by_health();
    assert_eq!(healthy, 3);
    assert_eq!(degraded, 2);
    assert_eq!(failed, 1);
}

#[test]
fn test_gossip_lamport_time_sync() {
    let mut state1 = LWWMembershipState::new();
    let mut state2 = LWWMembershipState::new();

    // State1 advances to time 5
    for _ in 0..5 {
        state1.tick();
    }
    assert_eq!(state1.lamport_time(), 5);

    // State2 receives message with time 5
    state2.sync_time(5);
    // Should be max(0, 5) + 1 = 6
    assert_eq!(state2.lamport_time(), 6);

    // State2 advances further
    state2.tick();
    assert_eq!(state2.lamport_time(), 7);

    // State1 syncs with state2's time
    state1.sync_time(7);
    // Should be max(5, 7) + 1 = 8
    assert_eq!(state1.lamport_time(), 8);
}

#[tokio::test]
async fn test_gossip_round_count_increments() {
    let transport = Arc::new(MemoryTransport::new("node1".to_string()));
    let config = GossipConfig::default();

    let manager = GossipMembershipManager::new("node1".to_string(), config, transport);
    manager.add_peer("node2".to_string());

    assert_eq!(manager.round_count(), 0);

    manager.gossip_round().await.unwrap();
    assert_eq!(manager.round_count(), 1);

    manager.gossip_round().await.unwrap();
    assert_eq!(manager.round_count(), 2);
}

#[test]
fn test_gossip_node_state_supersedes_logic() {
    // Same incarnation, higher timestamp wins
    let state1 = GossipNodeState::new("node".to_string(), NodeHealth::Healthy, 1, 0);
    let state2 = GossipNodeState::new("node".to_string(), NodeHealth::Failed, 2, 0);
    assert!(state2.supersedes(&state1));
    assert!(!state1.supersedes(&state2));

    // Higher incarnation always wins (even with lower timestamp)
    let state3 = GossipNodeState::new("node".to_string(), NodeHealth::Healthy, 1, 1);
    assert!(state3.supersedes(&state1));
    assert!(state3.supersedes(&state2));

    // Equal values don't supersede
    let state4 = GossipNodeState::new("node".to_string(), NodeHealth::Healthy, 1, 0);
    assert!(!state1.supersedes(&state4));
    assert!(!state4.supersedes(&state1));
}
