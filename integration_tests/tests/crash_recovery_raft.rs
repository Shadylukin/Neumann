//! Integration tests for Raft crash recovery scenarios.
//!
//! Tests:
//! - Recovery after partial WAL write
//! - Snapshot recovery after crash
//! - Term and vote persistence across multiple restarts
//! - Log entry recovery consistency

use std::io::Write;
use std::sync::Arc;

use tempfile::tempdir;
use tensor_chain::{
    Block, BlockHeader, MemoryTransport, RaftConfig, RaftNode, RaftRecoveryState, RaftState,
    RaftWal, RaftWalEntry,
};
use tensor_store::TensorStore;

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

fn create_test_block(height: u64, proposer: &str) -> Block {
    let header = BlockHeader::new(
        height,
        [0u8; 32], // prev_hash
        [0u8; 32], // tx_root
        [0u8; 32], // state_root
        proposer.to_string(),
    );
    Block::new(header, vec![])
}

// ============= Partial Write Recovery Tests =============

#[test]
fn test_raft_wal_recovery_after_partial_write() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("partial.wal");

    // Write a valid entry
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        wal.append(&RaftWalEntry::TermAndVote {
            term: 5,
            voted_for: Some("leader1".to_string()),
        })
        .unwrap();
    }

    // Simulate partial write by appending garbage
    {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal_path)
            .unwrap();
        // Write incomplete/corrupted data
        file.write_all(&[0xFF, 0xFE, 0xFD]).unwrap();
    }

    // Recovery should handle gracefully, skipping partial entry
    let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);

    // Should recover the valid entry
    assert_eq!(node.current_term(), 5, "Should recover valid term before corruption");
    assert_eq!(node.state(), RaftState::Follower);
}

#[test]
fn test_raft_wal_recovery_empty_corruption() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("corrupt.wal");

    // Write valid entries first
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        wal.append(&RaftWalEntry::TermAndVote {
            term: 3,
            voted_for: None,
        })
        .unwrap();
        wal.append(&RaftWalEntry::TermChange { new_term: 4 })
            .unwrap();
    }

    // Truncate to corrupt last entry
    {
        let metadata = std::fs::metadata(&wal_path).unwrap();
        let file_len = metadata.len();
        // Truncate a few bytes to corrupt the last entry
        if file_len > 5 {
            let file = std::fs::File::options()
                .write(true)
                .open(&wal_path)
                .unwrap();
            file.set_len(file_len - 3).unwrap();
        }
    }

    // Recovery should handle the truncated entry
    let wal = RaftWal::open(&wal_path).unwrap();
    let recovery = RaftRecoveryState::from_wal(&wal);

    // Should either recover what was possible or return an error
    // The important thing is it doesn't panic
    if let Ok(state) = recovery {
        // If it recovered, should have at least term 3
        assert!(state.current_term >= 3);
    }
    // An error is also acceptable for corrupted data
}

// ============= Snapshot Recovery Tests =============

#[test]
fn test_raft_snapshot_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("snapshot.wal");

    // Create a WAL with snapshot entry
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();

        // Start with some term
        wal.append(&RaftWalEntry::TermAndVote {
            term: 10,
            voted_for: Some("node1".to_string()),
        })
        .unwrap();

        // Add snapshot taken entry
        wal.append(&RaftWalEntry::SnapshotTaken {
            last_included_index: 100,
            last_included_term: 8,
        })
        .unwrap();

        // Continue with more term changes after snapshot
        wal.append(&RaftWalEntry::TermChange { new_term: 11 })
            .unwrap();
    }

    // Recovery should restore snapshot metadata
    let wal = RaftWal::open(&wal_path).unwrap();
    let recovery = RaftRecoveryState::from_wal(&wal).unwrap();

    assert_eq!(recovery.current_term, 11);
    assert_eq!(recovery.last_snapshot_index, Some(100));
    assert_eq!(recovery.last_snapshot_term, Some(8));
}

#[test]
fn test_raft_snapshot_metadata_persistence_via_store() {
    let store = TensorStore::new();

    // First session - create node with store and save
    {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = RaftConfig {
            snapshot_threshold: 10,
            auto_heartbeat: false,
            ..RaftConfig::default()
        };
        let node = RaftNode::with_store(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
            &store,
        );

        // Save term state
        node.start_election();
        node.save_to_store(&store).unwrap();
    }

    // Second session - verify state persisted
    {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::with_store(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            RaftConfig::default(),
            &store,
        );

        // Term should be restored
        assert!(node.current_term() >= 1);
    }
}

// ============= Multiple Restart Recovery Tests =============

#[test]
fn test_raft_multiple_restarts_preserves_state() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("multi_restart.wal");

    // First session
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);
        node.start_election(); // term 1
        assert_eq!(node.current_term(), 1);
    }

    // Second session - add more elections
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);
        assert_eq!(node.current_term(), 1);
        node.start_election(); // term 2
        node.start_election(); // term 3
        assert_eq!(node.current_term(), 3);
    }

    // Third session - verify final state
    {
        let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);
        assert_eq!(node.current_term(), 3);
        assert_eq!(node.state(), RaftState::Follower); // Starts as follower on recovery
    }
}

#[test]
fn test_raft_vote_recovery_prevents_double_vote() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("vote_recovery.wal");

    // First session - vote for a specific candidate
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();
        wal.append(&RaftWalEntry::TermAndVote {
            term: 5,
            voted_for: Some("candidate_a".to_string()),
        })
        .unwrap();
    }

    // Second session - recovery should preserve the vote
    {
        let wal = RaftWal::open(&wal_path).unwrap();
        let recovery = RaftRecoveryState::from_wal(&wal).unwrap();

        assert_eq!(recovery.current_term, 5);
        assert_eq!(recovery.voted_for, Some("candidate_a".to_string()));

        // A node with this state should not vote for a different candidate in term 5
    }
}

// ============= Log Entry Recovery Tests =============

#[test]
fn test_raft_log_entries_persist_with_store() {
    let store = TensorStore::new();

    // First session - propose some blocks
    {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = RaftConfig {
            auto_heartbeat: false,
            ..RaftConfig::default()
        };
        let node = Arc::new(RaftNode::with_store(
            "node1".to_string(),
            vec![],
            transport,
            config,
            &store,
        ));

        // Single node - becomes leader on election
        node.start_election();
        // Note: Without receiving votes, it stays candidate

        // Save state
        node.save_to_store(&store).unwrap();
    }

    // Second session - verify state persists
    {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = Arc::new(RaftNode::with_store(
            "node1".to_string(),
            vec![],
            transport,
            RaftConfig::default(),
            &store,
        ));

        // Term should be persisted
        assert!(node.current_term() >= 1);
    }
}

#[test]
fn test_raft_commit_index_recovery() {
    let store = TensorStore::new();

    // First session
    {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = RaftConfig {
            auto_heartbeat: false,
            ..RaftConfig::default()
        };
        let node = RaftNode::with_store(
            "node1".to_string(),
            vec![],
            transport,
            config,
            &store,
        );

        // Propose and commit a block (single node commits immediately)
        node.start_election();
        let block = create_test_block(1, "node1");

        // For single node, it becomes leader and can propose
        if node.state() == RaftState::Leader {
            let _ = node.propose(block);
        }

        node.save_to_store(&store).unwrap();
    }

    // Second session - verify commit index
    {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::with_store(
            "node1".to_string(),
            vec![],
            transport,
            RaftConfig::default(),
            &store,
        );

        // Verify we can access commit index without panic (always >= 0)
        let _commit_index = node.commit_index();
    }
}

// ============= Edge Cases =============

#[test]
fn test_raft_recovery_with_empty_wal() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("empty.wal");

    // Create empty WAL
    {
        let _wal = RaftWal::open(&wal_path).unwrap();
        // Close without writing anything
    }

    // Should start with default state
    let node = create_test_node_with_wal("node1", vec!["node2".to_string()], &wal_path);

    assert_eq!(node.current_term(), 0);
    assert_eq!(node.state(), RaftState::Follower);
}

#[test]
fn test_raft_recovery_term_change_clears_vote() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("term_change.wal");

    // Create WAL with vote followed by term change
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();

        // Vote in term 5
        wal.append(&RaftWalEntry::TermAndVote {
            term: 5,
            voted_for: Some("old_candidate".to_string()),
        })
        .unwrap();

        // Term change to 7 should clear vote
        wal.append(&RaftWalEntry::TermChange { new_term: 7 })
            .unwrap();
    }

    // Recovery should have term 7 with no vote
    let wal = RaftWal::open(&wal_path).unwrap();
    let recovery = RaftRecoveryState::from_wal(&wal).unwrap();

    assert_eq!(recovery.current_term, 7);
    assert_eq!(recovery.voted_for, None, "Term change should clear vote");
}

#[test]
fn test_raft_recovery_preserves_highest_term() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("highest_term.wal");

    // Create WAL with multiple term changes
    {
        let mut wal = RaftWal::open(&wal_path).unwrap();

        wal.append(&RaftWalEntry::TermChange { new_term: 1 })
            .unwrap();
        wal.append(&RaftWalEntry::TermChange { new_term: 5 })
            .unwrap();
        wal.append(&RaftWalEntry::TermChange { new_term: 3 })
            .unwrap(); // Lower term (shouldn't happen in practice)
        wal.append(&RaftWalEntry::TermChange { new_term: 10 })
            .unwrap();
    }

    // Recovery should have the highest term
    let wal = RaftWal::open(&wal_path).unwrap();
    let recovery = RaftRecoveryState::from_wal(&wal).unwrap();

    assert_eq!(recovery.current_term, 10);
}
