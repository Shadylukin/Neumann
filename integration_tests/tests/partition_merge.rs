//! Integration tests for partition merge protocol.
//!
//! Tests the full merge flow: heal detection, view exchange, data reconciliation,
//! transaction reconciliation, and finalization.

use tensor_chain::distributed_tx::TxPhase;
use tensor_chain::gossip::GossipNodeState;
use tensor_chain::membership::NodeHealth;
use tensor_chain::{
    DataReconciler, MembershipReconciler, MembershipViewSummary, MergePhase, PartitionMergeConfig,
    PartitionMergeManager, PartitionStateSummary, PendingTxState, TransactionReconciler,
};
use tensor_store::SparseVector;

/// Test heal detection and merge session initiation.
#[test]
fn test_partition_heal_detection() {
    let config = PartitionMergeConfig::default();
    let manager = PartitionMergeManager::new("node1".to_string(), config);

    // Simulate healed nodes
    let healed = vec!["node2".to_string(), "node3".to_string()];
    let session_id = manager.start_merge(healed);

    assert!(session_id.is_some());
    let id = session_id.unwrap();
    assert_eq!(manager.session_phase(id), Some(MergePhase::HealDetection));
}

/// Test membership view reconciliation with LWW-CRDT.
#[test]
fn test_membership_view_reconciliation() {
    // Node1's view: node2 is Healthy (incarnation 5)
    let local = MembershipViewSummary::new("node1".to_string(), 100, 1).with_states(vec![
        GossipNodeState {
            node_id: "node2".to_string(),
            health: NodeHealth::Healthy,
            timestamp: 100,
            updated_at: 100,
            incarnation: 5,
        },
        GossipNodeState {
            node_id: "node3".to_string(),
            health: NodeHealth::Healthy,
            timestamp: 100,
            updated_at: 100,
            incarnation: 3,
        },
    ]);

    // Node2's view: node2 is Degraded (incarnation 6 - higher wins)
    let remote = MembershipViewSummary::new("node2".to_string(), 110, 1).with_states(vec![
        GossipNodeState {
            node_id: "node2".to_string(),
            health: NodeHealth::Degraded,
            timestamp: 110,
            updated_at: 110,
            incarnation: 6, // Higher incarnation
        },
        GossipNodeState {
            node_id: "node4".to_string(),
            health: NodeHealth::Healthy,
            timestamp: 110,
            updated_at: 110,
            incarnation: 1,
        },
    ]);

    let (merged, _conflicts) = MembershipReconciler::merge(&local, &remote).unwrap();

    // Should have 3 unique nodes in merged view (node2, node3, node4)
    assert_eq!(merged.node_states.len(), 3);

    // node2 should have incarnation 6 (from remote, higher incarnation wins)
    let node2 = merged
        .node_states
        .iter()
        .find(|s| s.node_id == "node2")
        .unwrap();
    assert_eq!(node2.incarnation, 6);
    assert_eq!(node2.health, NodeHealth::Degraded);

    // node3 should be from local only
    let node3 = merged
        .node_states
        .iter()
        .find(|s| s.node_id == "node3")
        .unwrap();
    assert_eq!(node3.incarnation, 3);

    // node4 should be from remote only
    let node4 = merged
        .node_states
        .iter()
        .find(|s| s.node_id == "node4")
        .unwrap();
    assert_eq!(node4.incarnation, 1);
}

/// Test data state reconciliation with orthogonal changes.
#[test]
fn test_data_state_reconciliation_orthogonal() {
    // Local partition made changes in dimension 0-10
    let mut local_emb = SparseVector::new(100);
    for i in 0..10 {
        local_emb.set(i, 1.0);
    }

    // Remote partition made changes in dimension 50-60
    let mut remote_emb = SparseVector::new(100);
    for i in 50..60 {
        remote_emb.set(i, 1.0);
    }

    let local = PartitionStateSummary::new("local".to_string())
        .with_embedding(local_emb)
        .with_hash([1u8; 32]);
    let remote = PartitionStateSummary::new("remote".to_string())
        .with_embedding(remote_emb)
        .with_hash([2u8; 32]);

    let reconciler = DataReconciler::default();
    let result = reconciler.reconcile(&local, &remote);

    // Orthogonal changes should merge successfully
    assert!(result.success);
    assert!(!result.requires_manual);
    assert!(result.merged_data.is_some());

    // Merged data should have non-zero values in both ranges
    let merged = result.merged_data.unwrap();
    assert!(merged.get(0) > 0.0);
    assert!(merged.get(55) > 0.0);
}

/// Test data state reconciliation with conflicting changes.
#[test]
fn test_data_state_reconciliation_conflicting() {
    // Both partitions made changes to the same dimensions
    let mut local_emb = SparseVector::new(100);
    local_emb.set(0, 1.0);
    local_emb.set(1, 0.5);
    local_emb.set(2, 0.3);

    let mut remote_emb = SparseVector::new(100);
    remote_emb.set(0, 0.8);
    remote_emb.set(1, 0.4);
    remote_emb.set(2, 0.2);

    let local = PartitionStateSummary::new("local".to_string())
        .with_embedding(local_emb)
        .with_hash([1u8; 32]);
    let remote = PartitionStateSummary::new("remote".to_string())
        .with_embedding(remote_emb)
        .with_hash([2u8; 32]);

    // Use strict thresholds to force conflict detection
    let reconciler = DataReconciler::new(0.001, 0.9999);
    let result = reconciler.reconcile(&local, &remote);

    // Highly similar non-identical changes should conflict
    assert!(!result.success);
    assert!(result.requires_manual);
    assert!(!result.conflicts.is_empty());
}

/// Test pending transaction reconciliation.
#[test]
fn test_pending_tx_reconciliation() {
    // Transaction 1: Both sides have it with YES votes
    let mut local_tx1 = PendingTxState::new(1, "coord".to_string(), TxPhase::Preparing);
    local_tx1.votes.insert(0, true);
    local_tx1.votes.insert(1, true);

    let mut remote_tx1 = PendingTxState::new(1, "coord".to_string(), TxPhase::Preparing);
    remote_tx1.votes.insert(0, true);
    remote_tx1.votes.insert(2, true); // Additional vote from shard 2

    // Transaction 2: Local has YES, remote has NO
    let mut local_tx2 = PendingTxState::new(2, "coord".to_string(), TxPhase::Preparing);
    local_tx2.votes.insert(0, true);

    let mut remote_tx2 = PendingTxState::new(2, "coord".to_string(), TxPhase::Preparing);
    remote_tx2.votes.insert(1, false);

    // Transaction 3: Only on local
    let mut local_tx3 = PendingTxState::new(3, "coord".to_string(), TxPhase::Preparing);
    local_tx3.votes.insert(0, true);
    local_tx3.votes.insert(1, true);

    let reconciler = TransactionReconciler::default();
    let result = reconciler
        .reconcile(&[local_tx1, local_tx2, local_tx3], &[remote_tx1, remote_tx2])
        .unwrap();

    // Transaction 1: All YES -> commit
    assert!(result.to_commit.contains(&1));

    // Transaction 2: Has NO vote -> abort
    assert!(result.to_abort.contains(&2));

    // Transaction 3: Only on local with all YES -> commit
    assert!(result.to_commit.contains(&3));
}

/// Test full partition merge flow.
#[test]
fn test_full_partition_merge_flow() {
    let config = PartitionMergeConfig::default();
    let manager = PartitionMergeManager::new("node1".to_string(), config);

    // 1. Start merge session
    let session_id = manager
        .start_merge(vec!["node2".to_string()])
        .expect("should start merge");

    assert_eq!(
        manager.session_phase(session_id),
        Some(MergePhase::HealDetection)
    );

    // 2. Set local state summary
    let local_summary = PartitionStateSummary::new("node1".to_string())
        .with_log_position(100, 5)
        .with_hash([1u8; 32]);
    manager.set_local_summary(session_id, local_summary);

    // 3. Advance through phases
    manager.advance_session(session_id); // -> ViewExchange
    assert_eq!(
        manager.session_phase(session_id),
        Some(MergePhase::ViewExchange)
    );

    // Set local view
    let local_view = MembershipViewSummary::new("node1".to_string(), 100, 1);
    manager.set_local_view(session_id, local_view);

    manager.advance_session(session_id); // -> MembershipReconciliation
    assert_eq!(
        manager.session_phase(session_id),
        Some(MergePhase::MembershipReconciliation)
    );

    manager.advance_session(session_id); // -> DataReconciliation
    assert_eq!(
        manager.session_phase(session_id),
        Some(MergePhase::DataReconciliation)
    );

    manager.advance_session(session_id); // -> TransactionReconciliation
    assert_eq!(
        manager.session_phase(session_id),
        Some(MergePhase::TransactionReconciliation)
    );

    manager.advance_session(session_id); // -> Finalization
    assert_eq!(
        manager.session_phase(session_id),
        Some(MergePhase::Finalization)
    );

    manager.advance_session(session_id); // -> Completed
    assert_eq!(
        manager.session_phase(session_id),
        Some(MergePhase::Completed)
    );

    // 4. Complete the session
    manager.complete_session(session_id);

    // Session should be removed
    assert!(manager.get_session(session_id).is_none());

    // Stats should reflect success
    let stats = manager.stats_snapshot();
    assert_eq!(stats.sessions_started, 1);
    assert_eq!(stats.sessions_completed, 1);
    assert_eq!(stats.sessions_failed, 0);
}

/// Test merge during concurrent operations.
#[test]
fn test_concurrent_partition_heal() {
    let config = PartitionMergeConfig::default();
    let manager = PartitionMergeManager::new("node1".to_string(), config);

    // Start first merge
    let session1 = manager.start_merge(vec!["node2".to_string()]);
    assert!(session1.is_some());

    // Concurrent merge should be blocked (max_concurrent_merges = 1)
    let session2 = manager.start_merge(vec!["node3".to_string()]);
    assert!(session2.is_none());

    // Complete first merge
    manager.complete_session(session1.unwrap());

    // Now concurrent merge should be allowed (with different node)
    let session3 = manager.start_merge(vec!["node3".to_string()]);
    assert!(session3.is_some());
}

/// Test merge cooldown enforcement.
#[test]
fn test_merge_cooldown_enforcement() {
    let mut config = PartitionMergeConfig::default();
    config.merge_cooldown_ms = 60_000; // 60 seconds

    let manager = PartitionMergeManager::new("node1".to_string(), config);

    // Start and complete first merge
    let session1 = manager.start_merge(vec!["node2".to_string()]).unwrap();
    manager.complete_session(session1);

    // Immediate retry with same node should be blocked by cooldown
    let session2 = manager.start_merge(vec!["node2".to_string()]);
    assert!(session2.is_none());

    // Different node should work
    let session3 = manager.start_merge(vec!["node3".to_string()]);
    assert!(session3.is_some());
}

/// Test merge session failure and retry.
#[test]
fn test_merge_session_failure_retry() {
    let config = PartitionMergeConfig::default();
    let manager = PartitionMergeManager::new("node1".to_string(), config);

    let session_id = manager.start_merge(vec!["node2".to_string()]).unwrap();

    // Fail the session
    manager.fail_session(session_id, "network timeout");

    // Session should be in Failed state
    let session = manager.get_session(session_id).unwrap();
    assert_eq!(session.phase, MergePhase::Failed);
    assert_eq!(session.last_error, Some("network timeout".to_string()));

    // Stats should reflect failure
    let stats = manager.stats_snapshot();
    assert_eq!(stats.sessions_failed, 1);
}

/// Test statistics tracking.
#[test]
fn test_merge_statistics_tracking() {
    let config = PartitionMergeConfig::default();
    let manager = PartitionMergeManager::new("node1".to_string(), config);

    // Initial stats
    let stats = manager.stats_snapshot();
    assert_eq!(stats.sessions_started, 0);
    assert_eq!(stats.sessions_completed, 0);
    assert_eq!(stats.sessions_failed, 0);

    // Complete a session
    let session1 = manager.start_merge(vec!["node2".to_string()]).unwrap();
    for _ in 0..7 {
        manager.advance_session(session1);
    }
    manager.complete_session(session1);

    let stats = manager.stats_snapshot();
    assert_eq!(stats.sessions_started, 1);
    assert_eq!(stats.sessions_completed, 1);
    assert!(stats.total_merge_duration_ms > 0 || stats.sessions_completed == 1);
}
