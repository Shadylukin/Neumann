//! Integration tests for Raft dynamic membership via joint consensus.
//!
//! Tests cluster membership changes including:
//! - Adding learners to a cluster
//! - Promoting learners to voters
//! - Removing nodes from a cluster
//! - Joint consensus for safe transitions
//! - Membership surviving leader changes

use std::sync::Arc;

use tensor_chain::{MemoryTransport, RaftConfig, RaftMembershipConfig, RaftNode};

/// Create a connected N-node cluster with memory transport.
fn create_cluster(node_ids: &[&str]) -> Vec<Arc<RaftNode>> {
    let node_ids: Vec<String> = node_ids.iter().map(|s| s.to_string()).collect();
    let transports: Vec<Arc<MemoryTransport>> = node_ids
        .iter()
        .map(|id| Arc::new(MemoryTransport::new(id.clone())))
        .collect();

    // Connect all transports to each other
    for i in 0..transports.len() {
        for j in 0..transports.len() {
            if i != j {
                transports[i].connect_to(node_ids[j].clone(), transports[j].sender());
            }
        }
    }

    // Shorter timeouts for tests
    let config = RaftConfig {
        election_timeout: (50, 100),
        heartbeat_interval: 25,
        ..RaftConfig::default()
    };

    node_ids
        .iter()
        .zip(transports.into_iter())
        .map(|(id, transport)| {
            let peers: Vec<String> = node_ids.iter().filter(|p| *p != id).cloned().collect();
            Arc::new(RaftNode::new(id.clone(), peers, transport, config.clone()))
        })
        .collect()
}

#[tokio::test]
async fn test_membership_config_initial_state() {
    // Create a 3-node cluster
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    // All nodes should have same initial membership
    for node in &nodes {
        let config = node.membership_config();
        assert_eq!(config.voters.len(), 3);
        assert!(config.voters.contains(&"node1".to_string()));
        assert!(config.voters.contains(&"node2".to_string()));
        assert!(config.voters.contains(&"node3".to_string()));
        assert!(config.learners.is_empty());
        assert!(config.joint.is_none());
    }
}

#[tokio::test]
async fn test_add_learner_requires_leader() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    // Try to add learner when not leader - should fail
    let result = nodes[0].add_learner("node4".to_string());
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not the leader"));
}

#[tokio::test]
async fn test_add_learner_as_leader() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    // Make node1 the leader
    nodes[0].become_leader();
    assert!(nodes[0].is_leader());

    // Add a learner
    let result = nodes[0].add_learner("node4".to_string());
    assert!(result.is_ok());

    // Verify learner was added
    let config = nodes[0].membership_config();
    assert!(config.learners.contains(&"node4".to_string()));
    assert!(!config.voters.contains(&"node4".to_string()));
}

#[tokio::test]
async fn test_cannot_add_existing_node_as_learner() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    nodes[0].become_leader();

    // Try to add existing voter as learner
    let result = nodes[0].add_learner("node2".to_string());
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("already in cluster"));
}

#[tokio::test]
async fn test_promote_learner_requires_leader() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    // Not a leader - should fail
    let result = nodes[0].promote_learner(&"node4".to_string());
    assert!(result.is_err());
}

#[tokio::test]
async fn test_promote_learner_as_leader() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    nodes[0].become_leader();

    // First add as learner
    nodes[0].add_learner("node4".to_string()).unwrap();

    // Now promote
    let result = nodes[0].promote_learner(&"node4".to_string());
    assert!(result.is_ok());

    // Verify promoted
    let config = nodes[0].membership_config();
    assert!(config.voters.contains(&"node4".to_string()));
    assert!(!config.learners.contains(&"node4".to_string()));
}

#[tokio::test]
async fn test_remove_node_requires_leader() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    let result = nodes[0].remove_node(&"node2".to_string());
    assert!(result.is_err());
}

#[tokio::test]
async fn test_remove_node_as_leader() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    nodes[0].become_leader();

    // Remove node2
    let result = nodes[0].remove_node(&"node2".to_string());
    assert!(result.is_ok());

    // Verify removed
    let config = nodes[0].membership_config();
    assert!(!config.voters.contains(&"node2".to_string()));
    assert_eq!(config.voters.len(), 2);
}

#[tokio::test]
async fn test_leader_cannot_remove_self() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    nodes[0].become_leader();

    // Try to remove self
    let result = nodes[0].remove_node(&"node1".to_string());
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Cannot remove self"));
}

#[tokio::test]
async fn test_membership_replication_targets_includes_learners() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    nodes[0].become_leader();
    nodes[0].add_learner("node4".to_string()).unwrap();

    let targets = nodes[0].replication_targets();
    assert!(targets.contains(&"node2".to_string()));
    assert!(targets.contains(&"node3".to_string()));
    assert!(targets.contains(&"node4".to_string())); // Learner included
}

#[tokio::test]
async fn test_membership_has_quorum_simple() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    // For 3 nodes, quorum is 2
    let mut votes = std::collections::HashSet::new();
    votes.insert("node1".to_string());

    // 1 vote - not quorum
    assert!(!nodes[0].has_quorum(&votes));

    votes.insert("node2".to_string());
    // 2 votes - quorum
    assert!(nodes[0].has_quorum(&votes));
}

#[tokio::test]
async fn test_membership_not_in_joint_consensus_initially() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    assert!(!nodes[0].in_joint_consensus());
    assert!(!nodes[1].in_joint_consensus());
    assert!(!nodes[2].in_joint_consensus());
}

#[tokio::test]
async fn test_learner_caught_up_initially_false() {
    let nodes = create_cluster(&["node1", "node2", "node3"]);

    nodes[0].become_leader();
    nodes[0].add_learner("node4".to_string()).unwrap();

    // New learner not caught up yet
    assert!(!nodes[0].is_learner_caught_up(&"node4".to_string()));
}

#[tokio::test]
async fn test_config_change_serialization() {
    use tensor_chain::ConfigChange;

    let changes = vec![
        ConfigChange::AddLearner {
            node_id: "new_node".to_string(),
        },
        ConfigChange::PromoteLearner {
            node_id: "learner".to_string(),
        },
        ConfigChange::RemoveNode {
            node_id: "old_node".to_string(),
        },
        ConfigChange::JointChange {
            additions: vec!["n4".to_string()],
            removals: vec!["n1".to_string()],
        },
    ];

    for change in changes {
        let bytes = bitcode::serialize(&change).unwrap();
        let decoded: ConfigChange = bitcode::deserialize(&bytes).unwrap();
        assert_eq!(decoded, change);
    }
}

#[tokio::test]
async fn test_raft_membership_config_serialization() {
    let mut config = RaftMembershipConfig::new(vec!["n1".to_string(), "n2".to_string()]);
    config.add_learner("learner1".to_string());
    config.config_index = 42;

    let bytes = bitcode::serialize(&config).unwrap();
    let decoded: RaftMembershipConfig = bitcode::deserialize(&bytes).unwrap();

    assert_eq!(decoded.voters, config.voters);
    assert_eq!(decoded.learners, config.learners);
    assert_eq!(decoded.config_index, 42);
}

#[tokio::test]
async fn test_snapshot_includes_membership() {
    use tensor_chain::SnapshotMetadata;

    let membership = RaftMembershipConfig::new(vec!["n1".to_string(), "n2".to_string()]);

    let metadata = SnapshotMetadata::with_membership(100, 5, [1u8; 32], membership.clone(), 1024);

    // Verify membership is stored
    assert_eq!(metadata.membership.voters.len(), 2);
    assert!(metadata.membership.voters.contains(&"n1".to_string()));
    assert!(metadata.membership.voters.contains(&"n2".to_string()));

    // Verify backward compatibility - config field also populated
    assert_eq!(metadata.config.len(), 2);
}

#[tokio::test]
async fn test_log_entry_config_change() {
    use tensor_chain::{ConfigChange, LogEntry};

    let entry = LogEntry::config(
        2,
        10,
        ConfigChange::AddLearner {
            node_id: "new_node".to_string(),
        },
    );

    assert!(entry.is_config_change());
    assert_eq!(entry.term, 2);
    assert_eq!(entry.index, 10);

    // Verify serialization
    let bytes = bitcode::serialize(&entry).unwrap();
    let decoded: LogEntry = bitcode::deserialize(&bytes).unwrap();
    assert!(decoded.is_config_change());
}
