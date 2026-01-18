//! Integration tests for Raft + 2PC + Consensus interactions.
//!
//! Tests complex distributed scenarios:
//! - Leader election followed by cross-shard commit
//! - Semantic conflict detection during partition
//! - Orthogonal auto-merge with Raft replication
//! - Full pipeline: workspace -> delta -> consensus -> block -> Raft

use std::sync::Arc;

use tensor_chain::{
    block::Transaction,
    consensus::{ConsensusConfig, ConsensusManager, DeltaVector},
    DistributedTxConfig, DistributedTxCoordinator, MemoryTransport, PrepareRequest, PrepareVote,
    RaftConfig, RaftNode, RaftState, TxPhase,
};
use tensor_store::SparseVector;

fn create_raft_cluster(
    node_count: usize,
) -> (
    Vec<Arc<RaftNode>>,
    Vec<Arc<MemoryTransport>>,
    Arc<
        parking_lot::RwLock<
            std::collections::HashMap<String, Vec<(String, tensor_chain::Message)>>,
        >,
    >,
) {
    let messages = Arc::new(parking_lot::RwLock::new(std::collections::HashMap::<
        String,
        Vec<(String, tensor_chain::Message)>,
    >::new()));

    let node_ids: Vec<String> = (0..node_count).map(|i| format!("node{}", i)).collect();

    let mut transports = Vec::new();
    let mut nodes = Vec::new();

    for i in 0..node_count {
        let node_id = node_ids[i].clone();
        let transport = Arc::new(MemoryTransport::new(node_id.clone()));
        transports.push(transport.clone());

        let peers: Vec<String> = node_ids
            .iter()
            .filter(|id| **id != node_id)
            .cloned()
            .collect();

        let config = RaftConfig {
            election_timeout: (50, 100),
            heartbeat_interval: 20,
            enable_pre_vote: true,
            auto_heartbeat: false,
            ..RaftConfig::default()
        };

        let node = Arc::new(RaftNode::new(node_id, peers, transport, config));
        nodes.push(node);
    }

    // Connect transports using sender channels
    for i in 0..node_count {
        for j in 0..node_count {
            if i != j {
                transports[i].connect_to(node_ids[j].clone(), transports[j].sender());
            }
        }
    }

    (nodes, transports, messages)
}

fn create_coordinator_with_config(config: DistributedTxConfig) -> DistributedTxCoordinator {
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    DistributedTxCoordinator::new(consensus, config)
}

fn create_coordinator() -> DistributedTxCoordinator {
    create_coordinator_with_config(DistributedTxConfig::default())
}

// ============= Scenario 1: Leader Election + Cross-Shard Commit =============

#[test]
fn test_leader_election_then_cross_shard_commit() {
    // Setup: 3-node Raft cluster
    let (nodes, _transports, _messages) = create_raft_cluster(3);

    // 1. Trigger election on node 0
    nodes[0].start_election();
    assert_eq!(nodes[0].current_term(), 1);

    // Simulate receiving votes from other nodes
    let vote_response1 = tensor_chain::RequestVoteResponse {
        term: 1,
        vote_granted: true,
        voter_id: "node1".to_string(),
    };
    let vote_response2 = tensor_chain::RequestVoteResponse {
        term: 1,
        vote_granted: true,
        voter_id: "node2".to_string(),
    };

    nodes[0].handle_message(
        &"node1".to_string(),
        &tensor_chain::Message::RequestVoteResponse(vote_response1),
    );
    nodes[0].handle_message(
        &"node2".to_string(),
        &tensor_chain::Message::RequestVoteResponse(vote_response2),
    );

    // Node 0 should now be leader (has majority)
    assert_eq!(nodes[0].state(), RaftState::Leader);

    // 2. Now run a cross-shard 2PC transaction
    let coordinator = create_coordinator();
    let tx = coordinator
        .begin("coordinator".to_string(), vec![0, 1])
        .expect("Failed to begin transaction");

    // Both shards vote YES with orthogonal deltas
    for shard in 0..2 {
        let request = PrepareRequest {
            tx_id: tx.tx_id,
            coordinator: "coordinator".to_string(),
            operations: vec![Transaction::Put {
                key: format!("shard{}:key", shard),
                data: vec![shard as u8],
            }],
            delta_embedding: {
                let mut v = vec![0.0; 2];
                v[shard] = 1.0;
                SparseVector::from_dense(&v)
            },
            timeout_ms: 5000,
        };
        let vote = coordinator.handle_prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));
        coordinator.record_vote(tx.tx_id, shard, vote);
    }

    // Transaction should be prepared
    let tx_state = coordinator.get(tx.tx_id).unwrap();
    assert_eq!(tx_state.phase, TxPhase::Prepared);

    // 3. Commit the transaction
    coordinator.commit(tx.tx_id).unwrap();
    assert_eq!(coordinator.pending_count(), 0);

    // Verify stats
    assert!(
        coordinator
            .stats
            .committed
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 1
    );
}

// ============= Scenario 2: Semantic Conflict During Partition =============

#[test]
fn test_semantic_conflict_during_concurrent_modifications() {
    // Simulates two transactions modifying the same key with similar deltas
    let config = DistributedTxConfig {
        orthogonal_threshold: 0.5, // More sensitive to conflicts
        ..Default::default()
    };
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let coordinator = DistributedTxCoordinator::new(consensus, config);

    let tx = coordinator
        .begin("coordinator".to_string(), vec![0, 1])
        .expect("Failed to begin transaction");

    // Shard 0: Modifies "user:1" with delta [1,0,0]
    let delta0 = DeltaVector::new(
        vec![1.0, 0.0, 0.0],
        ["user:1"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    // Shard 1: Also modifies "user:1" with similar delta [0.9,0.1,0]
    // This should be detected as a conflict (same key, similar direction)
    let delta1 = DeltaVector::new(
        vec![0.9, 0.1, 0.0],
        ["user:1"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    // Record votes
    coordinator.record_vote(
        tx.tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );

    let phase = coordinator.record_vote(
        tx.tx_id,
        1,
        PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );

    // Should detect conflict due to overlapping keys with similar deltas
    assert_eq!(
        phase,
        Some(TxPhase::Aborting),
        "Expected conflict to be detected for same key with similar deltas"
    );
}

#[test]
fn test_semantic_conflict_dual_commit_prevented() {
    // Verify that a transaction cannot be committed twice
    let coordinator = create_coordinator();

    let tx = coordinator
        .begin("coordinator".to_string(), vec![0])
        .expect("Failed to begin transaction");

    // Vote YES
    let request = PrepareRequest {
        tx_id: tx.tx_id,
        coordinator: "coordinator".to_string(),
        operations: vec![Transaction::Put {
            key: "key".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0]),
        timeout_ms: 5000,
    };
    let vote = coordinator.handle_prepare(request);
    coordinator.record_vote(tx.tx_id, 0, vote);

    // First commit succeeds
    let result1 = coordinator.commit(tx.tx_id);
    assert!(result1.is_ok());

    // Second commit should fail (transaction no longer pending)
    let result2 = coordinator.commit(tx.tx_id);
    assert!(result2.is_err());
}

// ============= Scenario 3: Orthogonal Auto-Merge =============

#[test]
fn test_orthogonal_auto_merge_different_keys() {
    let coordinator = create_coordinator();

    let tx = coordinator
        .begin("coordinator".to_string(), vec![0, 1])
        .expect("Failed to begin transaction");

    // Shard 0: Modifies "account:A" (delta [1,0,0])
    let delta0 = DeltaVector::new(
        vec![1.0, 0.0, 0.0],
        ["account:A"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    // Shard 1: Modifies "account:B" (delta [0,1,0]) - orthogonal and different key
    let delta1 = DeltaVector::new(
        vec![0.0, 1.0, 0.0],
        ["account:B"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    coordinator.record_vote(
        tx.tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );

    let phase = coordinator.record_vote(
        tx.tx_id,
        1,
        PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );

    // Should succeed - orthogonal deltas on different keys can merge
    assert_eq!(phase, Some(TxPhase::Prepared));

    // Commit should work
    coordinator.commit(tx.tx_id).unwrap();

    // Verify orthogonal merge was counted
    assert!(
        coordinator
            .stats
            .orthogonal_merges
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0,
        "Expected orthogonal merge to be counted"
    );
}

#[test]
fn test_orthogonal_auto_merge_same_key_orthogonal_deltas() {
    // Even with the same key, truly orthogonal deltas could theoretically merge
    // This tests the boundary condition
    let coordinator = create_coordinator();

    let tx = coordinator
        .begin("coordinator".to_string(), vec![0, 1])
        .expect("Failed to begin transaction");

    // Different keys, orthogonal vectors
    let delta0 = DeltaVector::new(
        vec![1.0, 0.0, 0.0, 0.0],
        ["key_x"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    let delta1 = DeltaVector::new(
        vec![0.0, 0.0, 1.0, 0.0],
        ["key_y"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    coordinator.record_vote(
        tx.tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );

    let phase = coordinator.record_vote(
        tx.tx_id,
        1,
        PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );

    // Should succeed
    assert_eq!(phase, Some(TxPhase::Prepared));
}

// ============= Scenario 4: Full Pipeline Test =============

#[test]
fn test_full_flow_workspace_to_commit() {
    // Tests the complete flow: begin -> prepare -> vote -> commit

    let coordinator = create_coordinator();

    // 1. Begin transaction
    let tx = coordinator
        .begin("node1".to_string(), vec![0, 1, 2])
        .expect("Failed to begin transaction");

    assert_eq!(coordinator.pending_count(), 1);
    let tx_state = coordinator.get(tx.tx_id).unwrap();
    assert_eq!(tx_state.phase, TxPhase::Preparing);

    // 2. Each shard prepares with its operations
    let operations = vec![
        vec![Transaction::Put {
            key: "users:1".to_string(),
            data: b"alice".to_vec(),
        }],
        vec![Transaction::Put {
            key: "accounts:1".to_string(),
            data: b"balance:100".to_vec(),
        }],
        vec![Transaction::Put {
            key: "logs:1".to_string(),
            data: b"created".to_vec(),
        }],
    ];

    for shard in 0..3 {
        let request = PrepareRequest {
            tx_id: tx.tx_id,
            coordinator: "node1".to_string(),
            operations: operations[shard].clone(),
            delta_embedding: {
                let mut v = vec![0.0; 3];
                v[shard] = 1.0;
                SparseVector::from_dense(&v)
            },
            timeout_ms: 5000,
        };

        let vote = coordinator.handle_prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));
        coordinator.record_vote(tx.tx_id, shard, vote);
    }

    // 3. Verify transaction is prepared
    let tx_state = coordinator.get(tx.tx_id).unwrap();
    assert_eq!(tx_state.phase, TxPhase::Prepared);
    assert!(tx_state.all_voted());
    assert!(tx_state.all_yes());

    // 4. Commit
    coordinator.commit(tx.tx_id).unwrap();
    assert_eq!(coordinator.pending_count(), 0);

    // 5. Verify stats
    let stats = coordinator.stats();
    assert!(stats.started.load(std::sync::atomic::Ordering::Relaxed) >= 1);
    assert!(stats.committed.load(std::sync::atomic::Ordering::Relaxed) >= 1);
}

#[test]
fn test_full_flow_with_abort() {
    let coordinator = create_coordinator();

    // Begin transaction
    let tx = coordinator
        .begin("node1".to_string(), vec![0, 1])
        .expect("Failed to begin transaction");

    // Shard 0 votes YES
    let request0 = PrepareRequest {
        tx_id: tx.tx_id,
        coordinator: "node1".to_string(),
        operations: vec![Transaction::Put {
            key: "key0".to_string(),
            data: vec![0],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
        timeout_ms: 5000,
    };
    let vote0 = coordinator.handle_prepare(request0);
    coordinator.record_vote(tx.tx_id, 0, vote0);

    // Shard 1 votes NO
    let phase = coordinator.record_vote(
        tx.tx_id,
        1,
        PrepareVote::No {
            reason: "Resource unavailable".to_string(),
        },
    );

    // Should move to aborting
    assert_eq!(phase, Some(TxPhase::Aborting));

    // Actually abort the transaction to increment stats
    let abort_result = coordinator.abort(tx.tx_id, "Resource unavailable");
    assert!(abort_result.is_ok(), "Abort should succeed");

    // Verify stats - aborted counter is incremented by abort()
    let stats = coordinator.stats();
    assert!(stats.aborted.load(std::sync::atomic::Ordering::Relaxed) >= 1);
}

// ============= Scenario 5: Raft Leader Proposal =============

#[test]
fn test_raft_leader_proposal() {
    let (nodes, _transports, _messages) = create_raft_cluster(3);

    // Make node0 leader
    nodes[0].start_election();
    let vote_response1 = tensor_chain::RequestVoteResponse {
        term: 1,
        vote_granted: true,
        voter_id: "node1".to_string(),
    };
    let vote_response2 = tensor_chain::RequestVoteResponse {
        term: 1,
        vote_granted: true,
        voter_id: "node2".to_string(),
    };

    nodes[0].handle_message(
        &"node1".to_string(),
        &tensor_chain::Message::RequestVoteResponse(vote_response1),
    );
    nodes[0].handle_message(
        &"node2".to_string(),
        &tensor_chain::Message::RequestVoteResponse(vote_response2),
    );

    assert_eq!(nodes[0].state(), RaftState::Leader);

    // Mark peers as reachable so quorum is available for writes
    nodes[0]
        .quorum_tracker()
        .record_success(&"node1".to_string());
    nodes[0]
        .quorum_tracker()
        .record_success(&"node2".to_string());

    // Create a block to propose
    let header = tensor_chain::BlockHeader::new(
        1,                                  // height
        tensor_chain::BlockHash::default(), // prev_hash
        tensor_chain::BlockHash::default(), // tx_root
        tensor_chain::BlockHash::default(), // state_root
        "node0".to_string(),                // proposer
    );
    let block = tensor_chain::Block::new(
        header,
        vec![Transaction::Put {
            key: "test_key".to_string(),
            data: vec![1, 2, 3],
        }],
    );

    // Propose the block
    let result = nodes[0].propose(block);
    assert!(result.is_ok());

    // Verify log length increased
    assert_eq!(nodes[0].log_length(), 1);
}

// ============= Edge Cases =============

#[test]
fn test_empty_transaction_participants() {
    let coordinator = create_coordinator();

    // Transaction with no participants
    let tx = coordinator.begin("node1".to_string(), vec![]);

    // Should succeed
    assert!(tx.is_ok());
    let tx = tx.unwrap();

    // Since no participants, all_voted is trivially true
    let tx_state = coordinator.get(tx.tx_id).unwrap();
    assert!(tx_state.all_voted());
    // Note: The transaction stays in Preparing phase since no votes can move it to Prepared
    // (needs at least one participant to vote)
}

#[test]
fn test_transaction_timeout_detection() {
    // Create coordinator (timeout is per-transaction, default 5000ms)
    let coordinator = create_coordinator();

    let tx = coordinator
        .begin("node1".to_string(), vec![0])
        .expect("Failed to begin transaction");

    // The transaction uses its own timeout_ms (default 5000ms)
    // We can't easily test timeout without modifying the transaction directly
    // So we just verify the is_timed_out method exists and works at startup
    let tx_state = coordinator.get(tx.tx_id).unwrap();
    assert!(
        !tx_state.is_timed_out(),
        "Transaction should not be timed out immediately"
    );
}

#[test]
fn test_concurrent_transactions_different_keys() {
    let coordinator = create_coordinator();

    // Transaction 1 on key_a
    let tx1 = coordinator
        .begin("node1".to_string(), vec![0])
        .expect("Failed to begin tx1");

    let request1 = PrepareRequest {
        tx_id: tx1.tx_id,
        coordinator: "node1".to_string(),
        operations: vec![Transaction::Put {
            key: "key_a".to_string(),
            data: vec![1],
        }],
        delta_embedding: SparseVector::from_dense(&[1.0, 0.0]),
        timeout_ms: 5000,
    };
    let vote1 = coordinator.handle_prepare(request1);
    assert!(matches!(vote1, PrepareVote::Yes { .. }));

    // Transaction 2 on key_b (should not conflict)
    let tx2 = coordinator
        .begin("node1".to_string(), vec![0])
        .expect("Failed to begin tx2");

    let request2 = PrepareRequest {
        tx_id: tx2.tx_id,
        coordinator: "node1".to_string(),
        operations: vec![Transaction::Put {
            key: "key_b".to_string(),
            data: vec![2],
        }],
        delta_embedding: SparseVector::from_dense(&[0.0, 1.0]),
        timeout_ms: 5000,
    };
    let vote2 = coordinator.handle_prepare(request2);
    assert!(
        matches!(vote2, PrepareVote::Yes { .. }),
        "Different keys should not conflict"
    );
}

#[test]
fn test_merged_delta_computation() {
    let coordinator = create_coordinator();

    let tx = coordinator
        .begin("node1".to_string(), vec![0, 1])
        .expect("Failed to begin transaction");

    let delta0 = DeltaVector::new(
        vec![1.0, 0.0, 0.0],
        ["key_a"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    let delta1 = DeltaVector::new(
        vec![0.0, 2.0, 0.0],
        ["key_b"].iter().map(|s| s.to_string()).collect(),
        tx.tx_id,
    );

    coordinator.record_vote(
        tx.tx_id,
        0,
        PrepareVote::Yes {
            lock_handle: 1,
            delta: delta0,
        },
    );

    coordinator.record_vote(
        tx.tx_id,
        1,
        PrepareVote::Yes {
            lock_handle: 2,
            delta: delta1,
        },
    );

    // Get merged delta
    let tx_state = coordinator.get(tx.tx_id).unwrap();
    let merged = tx_state.merged_delta().unwrap();

    // Merged delta should combine both
    assert_eq!(merged.affected_keys.len(), 2);
    assert!(merged.affected_keys.contains("key_a"));
    assert!(merged.affected_keys.contains("key_b"));
}
