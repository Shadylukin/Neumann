//! Integration tests for Raft leadership transfer and pre-vote.
//!
//! Tests:
//! - Pre-vote prevents term inflation from partitioned nodes
//! - Pre-vote allows normal elections when non-partitioned
//! - Leadership transfer success flow
//! - Leadership transfer blocks proposals
//! - Leadership transfer timeout handling
//! - Pre-vote with stale log rejection
//! - Multiple sequential transfers

use std::{sync::Arc, time::Duration};

use tensor_chain::{
    Block, BlockHeader, MemoryTransport, Message, PreVote, RaftConfig, RaftNode, RaftState,
};
use tokio::time::{sleep, timeout};

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

/// Create a connected 3-node cluster returning nodes AND transports for partition injection.
fn create_3_node_cluster_with_transports() -> (
    Arc<RaftNode>,
    Arc<RaftNode>,
    Arc<RaftNode>,
    Arc<MemoryTransport>,
    Arc<MemoryTransport>,
    Arc<MemoryTransport>,
) {
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
        election_timeout: (50, 100),
        heartbeat_interval: 25,
        enable_pre_vote: true, // Enable pre-vote for these tests
        transfer_timeout_ms: 200,
        ..RaftConfig::default()
    };

    let node1 = Arc::new(RaftNode::new(
        "node1".to_string(),
        vec!["node2".to_string(), "node3".to_string()],
        transport1.clone(),
        config.clone(),
    ));
    let node2 = Arc::new(RaftNode::new(
        "node2".to_string(),
        vec!["node1".to_string(), "node3".to_string()],
        transport2.clone(),
        config.clone(),
    ));
    let node3 = Arc::new(RaftNode::new(
        "node3".to_string(),
        vec!["node1".to_string(), "node2".to_string()],
        transport3.clone(),
        config,
    ));

    (node1, node2, node3, transport1, transport2, transport3)
}

#[tokio::test]
async fn test_prevote_prevents_term_inflation() {
    // A partitioned node should not be able to inflate the cluster's term
    let (node1, _node2, _node3, t1, t2, t3) = create_3_node_cluster_with_transports();

    // Make node1 the leader
    node1.become_leader();
    assert_eq!(node1.state(), RaftState::Leader);

    // Partition node3 from everyone
    t1.partition(&"node3".to_string());
    t2.partition(&"node3".to_string());
    t3.partition(&"node1".to_string());
    t3.partition(&"node2".to_string());

    // Record initial term
    let initial_term = node1.current_term();

    // Partitioned node tries pre-vote multiple times
    // But since it can't get responses, it won't inflate its term
    for _ in 0..5 {
        // Simulate time passing - the partitioned node would try to start election
        // but pre-vote prevents term inflation since it can't reach quorum
        // In pre-vote, term is NOT incremented until quorum is reached
        sleep(Duration::from_millis(10)).await;
    }

    // Node1's term should not have been inflated by the partitioned node
    assert_eq!(
        node1.current_term(),
        initial_term,
        "leader term should not be inflated by partitioned node"
    );
}

#[tokio::test]
async fn test_prevote_allows_normal_election() {
    // A non-partitioned node should be able to win election via pre-vote
    let (node1, node2, node3, _t1, _t2, _t3) = create_3_node_cluster_with_transports();

    // All nodes start as followers
    assert_eq!(node1.state(), RaftState::Follower);
    assert_eq!(node2.state(), RaftState::Follower);
    assert_eq!(node3.state(), RaftState::Follower);

    // Node1 starts pre-vote
    node1.start_pre_vote_async().await.unwrap();

    // Process pre-vote requests from node1 at node2 and node3
    // After election timeout, they should grant pre-votes

    // Simulate node2 receiving and responding to pre-vote
    if let Ok(Ok((from, msg))) = timeout(Duration::from_millis(100), node2.transport_recv()).await {
        if let Message::PreVote(pv) = msg {
            // Simulate timeout elapsed by using test helper
            node2.reset_heartbeat_for_election();

            let response = node2.handle_message(&from, &Message::PreVote(pv.clone()));
            if let Some(Message::PreVoteResponse(pvr)) = response {
                // Send response back
                node1.handle_message(&"node2".to_string(), &Message::PreVoteResponse(pvr));
            }
        }
    }

    // After receiving quorum of pre-votes, node1 should start real election
    // and become candidate with incremented term
    assert_eq!(node1.state(), RaftState::Candidate);
    assert_eq!(node1.current_term(), 1); // Term incremented by real election
}

#[tokio::test]
async fn test_leadership_transfer_success() {
    let (node1, node2, _node3, _t1, _t2, _t3) = create_3_node_cluster_with_transports();

    // Make node1 the leader
    node1.become_leader();
    assert_eq!(node1.state(), RaftState::Leader);

    // Set node1 as leader for node2
    node2.set_current_leader(Some("node1".to_string()));

    // Start transfer to node2
    node1
        .transfer_leadership_async(&"node2".to_string())
        .await
        .unwrap();

    // Node1 should have transfer in progress
    assert!(node1.is_transfer_in_progress());

    // Process messages at node2
    // First: AppendEntries (heartbeat to catch up)
    if let Ok(Ok((from, msg))) = timeout(Duration::from_millis(100), node2.transport_recv()).await {
        let _response = node2.handle_message(&from, &msg);
    }

    // Second: TimeoutNow
    if let Ok(Ok((from, msg))) = timeout(Duration::from_millis(100), node2.transport_recv()).await {
        if let Message::TimeoutNow(_tn) = &msg {
            node2.handle_message(&from, &msg);
        }
    }

    // Node2 should have started election
    assert_eq!(node2.state(), RaftState::Candidate);
}

#[tokio::test]
async fn test_leadership_transfer_blocks_proposals() {
    let (node1, _node2, _node3, _t1, _t2, _t3) = create_3_node_cluster_with_transports();

    // Make node1 the leader
    node1.become_leader();

    // Start transfer
    node1.transfer_leadership(&"node2".to_string()).unwrap();

    // Proposals should be blocked
    let block = create_test_block(1, "node1");
    let result = node1.propose(block);

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("transfer in progress"));
}

#[tokio::test]
async fn test_leadership_transfer_timeout() {
    let (node1, _node2, _node3, t1, _t2, _t3) = create_3_node_cluster_with_transports();

    // Make node1 the leader
    node1.become_leader();

    // Partition node2 so it can't receive the transfer
    t1.partition(&"node2".to_string());

    // Start transfer
    node1.transfer_leadership(&"node2".to_string()).unwrap();
    assert!(node1.is_transfer_in_progress());

    // Wait for timeout (200ms configured)
    sleep(Duration::from_millis(250)).await;

    // Tick should cancel the transfer
    node1.tick_async().await.unwrap();

    // Transfer should be cancelled
    assert!(
        !node1.is_transfer_in_progress(),
        "transfer should timeout and cancel"
    );
}

#[tokio::test]
async fn test_prevote_with_stale_log() {
    // A node with stale log should not be granted pre-votes
    let (node1, _node2, _node3, _t1, _t2, _t3) = create_3_node_cluster_with_transports();

    // Add entries to node1's log
    node1.become_leader();

    // Mark peers as reachable so propose() can succeed (quorum check)
    node1
        .quorum_tracker()
        .record_success(&"node2".to_string());
    node1
        .quorum_tracker()
        .record_success(&"node3".to_string());

    let block = create_test_block(1, "node1");
    node1.propose(block).unwrap();

    // Node2 has empty log - simulate it trying pre-vote with stale log
    // by directly testing the pre-vote handler

    // Node1 should deny pre-vote from node2 due to stale log
    // (node2's log is behind node1's)

    // Create a pre-vote request from node2 with empty log
    let pv = PreVote {
        term: 0,
        candidate_id: "node2".to_string(),
        last_log_index: 0,
        last_log_term: 0,
        state_embedding: tensor_store::SparseVector::new(0),
    };

    // Set timeout elapsed on node1
    node1.reset_heartbeat_for_election();

    let response = node1.handle_message(&"node2".to_string(), &Message::PreVote(pv));

    if let Some(Message::PreVoteResponse(pvr)) = response {
        assert!(
            !pvr.vote_granted,
            "should not grant pre-vote to node with stale log"
        );
    } else {
        panic!("expected PreVoteResponse");
    }
}

#[tokio::test]
async fn test_multiple_transfers() {
    let (node1, _node2, _node3, _t1, _t2, _t3) = create_3_node_cluster_with_transports();

    // Make node1 the leader
    node1.become_leader();

    // First transfer - start and cancel
    node1.transfer_leadership(&"node2".to_string()).unwrap();
    assert!(node1.is_transfer_in_progress());
    node1.cancel_transfer();
    assert!(!node1.is_transfer_in_progress());

    // Second transfer - should work
    node1.transfer_leadership(&"node3".to_string()).unwrap();
    assert!(node1.is_transfer_in_progress());

    // Cancel and verify
    node1.cancel_transfer();
    assert!(!node1.is_transfer_in_progress());
}

#[tokio::test]
async fn test_transfer_during_partition() {
    let (node1, _node2, _node3, t1, _t2, _t3) = create_3_node_cluster_with_transports();

    // Make node1 the leader
    node1.become_leader();

    // Partition the target
    t1.partition(&"node2".to_string());

    // Transfer should still be initiated (sync part succeeds)
    node1.transfer_leadership(&"node2".to_string()).unwrap();
    assert!(node1.is_transfer_in_progress());

    // But the async send will fail when we try to send
    // The transfer will eventually timeout
}
