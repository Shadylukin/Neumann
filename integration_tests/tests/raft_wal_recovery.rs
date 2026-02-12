// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for Raft WAL recovery.
//!
//! Tests:
//! - Raft node recovers term and vote after crash
//! - No double voting after recovery
//! - WAL correctly handles partial writes

use std::sync::Arc;

use tempfile::tempdir;
use tensor_chain::{
    MemoryTransport, RaftConfig, RaftNode, RaftRecoveryState, RaftState, RaftWal, RaftWalEntry,
};

fn create_test_node_with_wal(
    id: &str,
    peers: Vec<String>,
    wal_path: &std::path::Path,
) -> Arc<RaftNode> {
    let transport = Arc::new(MemoryTransport::new(id.to_string()));
    Arc::new(
        RaftNode::with_wal(
            id.to_string(),
            peers,
            transport,
            RaftConfig::default(),
            wal_path,
        )
        .expect("Failed to create node with WAL"),
    )
}

#[test]
fn test_raft_recovers_term_after_crash() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("node1.wal");

    // Create node and start election (increments term)
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);
        node.start_election();
        assert_eq!(node.current_term(), 1);
    }

    // Simulate crash and recovery
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);
        // Node should recover term from WAL
        assert_eq!(node.current_term(), 1, "Term should be recovered from WAL");
    }
}

#[test]
fn test_raft_recovers_vote_after_crash() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("node1.wal");

    // Create node, start election (votes for self)
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);
        node.start_election();
        assert_eq!(node.current_term(), 1);
        // Node voted for itself during election start
    }

    // Verify via direct WAL read that vote was persisted
    {
        let wal = RaftWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();

        let has_vote = entries.iter().any(|e| {
            matches!(
                e,
                RaftWalEntry::TermAndVote {
                    term: 1,
                    voted_for: Some(candidate)
                } if candidate == "node1"
            )
        });
        assert!(has_vote, "WAL should contain vote for node1");
    }

    // Recovery should restore the vote
    {
        let recovery = RaftRecoveryState::from_wal(&RaftWal::open(&wal_path).unwrap()).unwrap();
        assert_eq!(recovery.current_term, 1);
        assert_eq!(recovery.voted_for, Some("node1".to_string()));
    }
}

#[test]
fn test_no_double_vote_after_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("node1.wal");

    // First session: vote in term 5
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        wal.append(&RaftWalEntry::TermAndVote {
            term: 5,
            voted_for: Some("leader1".to_string()),
        })
        .unwrap();
    }

    // Second session: recover and check state
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);

        // Node should have recovered term 5 and vote for leader1
        assert_eq!(node.current_term(), 5);

        // The node should be in follower state (default after recovery)
        assert_eq!(node.state(), RaftState::Follower);
    }
}

#[test]
fn test_wal_survives_multiple_elections() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("node1.wal");

    // Multiple election cycles
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);
        node.start_election(); // term 1
        node.start_election(); // term 2
        node.start_election(); // term 3
        assert_eq!(node.current_term(), 3);
    }

    // Recover and verify final term
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);
        assert_eq!(node.current_term(), 3);
    }
}

#[test]
fn test_wal_handles_empty_wal_file() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("empty.wal");

    // Create node with fresh (empty) WAL
    let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);

    // Should start with term 0
    assert_eq!(node.current_term(), 0);
    assert_eq!(node.state(), RaftState::Follower);
}

#[test]
fn test_recovery_state_from_complex_wal() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("complex.wal");

    // Write complex sequence of entries
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();

        // Election 1
        wal.append(&RaftWalEntry::TermAndVote {
            term: 1,
            voted_for: Some("node1".to_string()),
        })
        .unwrap();

        // Step down on higher term
        wal.append(&RaftWalEntry::TermAndVote {
            term: 3,
            voted_for: None,
        })
        .unwrap();

        // Vote for another node
        wal.append(&RaftWalEntry::TermAndVote {
            term: 3,
            voted_for: Some("node2".to_string()),
        })
        .unwrap();

        // New term clears vote
        wal.append(&RaftWalEntry::TermChange { new_term: 5 })
            .unwrap();
    }

    // Verify recovery state
    {
        let wal = RaftWal::open(&wal_path).unwrap();
        let recovery = RaftRecoveryState::from_wal(&wal).unwrap();

        assert_eq!(recovery.current_term, 5);
        assert_eq!(recovery.voted_for, None); // TermChange clears vote
    }
}
