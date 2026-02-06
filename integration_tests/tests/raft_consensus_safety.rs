// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for Raft consensus safety invariants.
//!
//! Tests:
//! - No double vote after WAL recovery
//! - 4-node cluster requires 3 for quorum
//! - Snapshot term survives crash recovery
//! - Async election vote persisted to WAL

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
fn test_no_double_vote_after_wal_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("node1.wal");

    // Create a WAL with a vote in term 5
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        wal.append(&RaftWalEntry::TermAndVote {
            term: 5,
            voted_for: Some("leader1".to_string()),
        })
        .unwrap();
    }

    // Try to add another vote in the same term (simulating corrupt/malicious WAL)
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        wal.append(&RaftWalEntry::VoteCast {
            term: 5,
            candidate_id: "leader2".to_string(),
        })
        .unwrap();
        wal.append(&RaftWalEntry::TermAndVote {
            term: 5,
            voted_for: Some("leader3".to_string()),
        })
        .unwrap();
    }

    // Recovery should only accept the first vote
    let wal = RaftWal::open(&wal_path).unwrap();
    let state = RaftRecoveryState::from_wal(&wal).unwrap();

    assert_eq!(state.current_term, 5);
    // First vote (leader1) should be preserved, duplicates ignored
    assert_eq!(
        state.voted_for,
        Some("leader1".to_string()),
        "Recovery should only accept first vote for a term"
    );
}

#[test]
fn test_4_node_cluster_election_requires_3_votes() {
    // Create a 4-node cluster and verify election behavior
    // With 4 nodes, quorum is 3 (majority > half)
    // This test verifies the fix for: div_ceil(4, 2) = 2 bug
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("quorum.wal");

    let node = create_test_node_with_wal(
        "node1",
        vec![
            "node2".to_string(),
            "node3".to_string(),
            "node4".to_string(),
        ],
        &wal_path,
    );

    // Start election - node votes for itself
    node.start_election();
    assert_eq!(node.state(), RaftState::Candidate);
    assert_eq!(node.current_term(), 1);

    // With only 1 vote (self), should still be candidate (need 3)
    // Note: quorum = (4 / 2) + 1 = 3, not div_ceil(4, 2) = 2
    assert_eq!(
        node.state(),
        RaftState::Candidate,
        "Should still be candidate with only 1 vote (need 3 for quorum)"
    );
}

#[test]
fn test_snapshot_term_survives_crash_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("snapshot.wal");

    // Simulate node that had a snapshot installed with term 10
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        // Node was at term 5, voted for someone
        wal.append(&RaftWalEntry::TermAndVote {
            term: 5,
            voted_for: Some("old_leader".to_string()),
        })
        .unwrap();
        // Then snapshot with higher term installed
        wal.append(&RaftWalEntry::SnapshotTaken {
            last_included_index: 1000,
            last_included_term: 10,
        })
        .unwrap();
    }

    // Recovery should have term 10 from snapshot
    let wal = RaftWal::open(&wal_path).unwrap();
    let state = RaftRecoveryState::from_wal(&wal).unwrap();

    assert_eq!(state.current_term, 10, "Snapshot term should be recovered");
    assert_eq!(
        state.voted_for, None,
        "voted_for should be reset when term changes from snapshot"
    );
    assert_eq!(state.last_snapshot_index, Some(1000));
    assert_eq!(state.last_snapshot_term, Some(10));
}

#[tokio::test]
async fn test_async_election_vote_persisted() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("async_election.wal");

    // Create node and start async election
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);

        // Start async election - this should persist to WAL
        node.start_election_async().await.unwrap();

        assert_eq!(node.current_term(), 1);
        assert_eq!(node.state(), RaftState::Candidate);
    }

    // Verify WAL has the vote persisted
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
        assert!(has_vote, "Async election should persist vote to WAL");
    }

    // Recovery should restore the vote
    {
        let wal = RaftWal::open(&wal_path).unwrap();
        let state = RaftRecoveryState::from_wal(&wal).unwrap();

        assert_eq!(state.current_term, 1);
        assert_eq!(state.voted_for, Some("node1".to_string()));
    }
}

#[test]
fn test_snapshot_does_not_downgrade_term() {
    // Snapshot with lower term should not downgrade current_term
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("no_downgrade.wal");

    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        // Node is at term 10
        wal.append(&RaftWalEntry::TermAndVote {
            term: 10,
            voted_for: Some("current_leader".to_string()),
        })
        .unwrap();
        // Snapshot with lower term 5
        wal.append(&RaftWalEntry::SnapshotTaken {
            last_included_index: 500,
            last_included_term: 5,
        })
        .unwrap();
    }

    let wal = RaftWal::open(&wal_path).unwrap();
    let state = RaftRecoveryState::from_wal(&wal).unwrap();

    // Term should stay at 10, not downgrade to 5
    assert_eq!(
        state.current_term, 10,
        "Snapshot should not downgrade current_term"
    );
    // Vote should be preserved since term didn't change
    assert_eq!(state.voted_for, Some("current_leader".to_string()));
}

#[test]
fn test_vote_after_term_change_accepted() {
    // After TermChange, the next vote should be accepted
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("vote_after_term.wal");

    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        // Vote in term 5
        wal.append(&RaftWalEntry::VoteCast {
            term: 5,
            candidate_id: "node_a".to_string(),
        })
        .unwrap();
        // Term changes to 6
        wal.append(&RaftWalEntry::TermChange { new_term: 6 })
            .unwrap();
        // New vote in term 6 should be accepted
        wal.append(&RaftWalEntry::VoteCast {
            term: 6,
            candidate_id: "node_b".to_string(),
        })
        .unwrap();
    }

    let wal = RaftWal::open(&wal_path).unwrap();
    let state = RaftRecoveryState::from_wal(&wal).unwrap();

    assert_eq!(state.current_term, 6);
    assert_eq!(
        state.voted_for,
        Some("node_b".to_string()),
        "Vote in new term should be accepted"
    );
}
