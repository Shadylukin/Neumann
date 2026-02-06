// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for simultaneous partition merge initiation.
//!
//! Validates that when both sides of a partition detect healing and attempt
//! to initiate merge sessions concurrently, the system handles deduplication,
//! generation tracking, and repartition detection correctly.

use tensor_chain::{MergePhase, PartitionMergeConfig, PartitionMergeManager};

fn default_config() -> PartitionMergeConfig {
    PartitionMergeConfig {
        merge_cooldown_ms: 0, // disable cooldown for testing
        max_concurrent_merges: 5,
        ..PartitionMergeConfig::default()
    }
}

#[test]
fn test_both_sides_initiate_merge_unique_sessions() {
    let config = default_config();

    // Two nodes detect the same heal event and both try to start merge
    let manager_a = PartitionMergeManager::new("node_a".to_string(), config.clone());
    let manager_b = PartitionMergeManager::new("node_b".to_string(), config);

    // Both managers start merge sessions for the same healed node set
    let session_a = manager_a.start_merge(vec!["node_b".to_string()]);
    let session_b = manager_b.start_merge(vec!["node_a".to_string()]);

    // Both should succeed (each manager tracks its own sessions)
    assert!(session_a.is_some(), "Manager A should start a session");
    assert!(session_b.is_some(), "Manager B should start a session");

    // Session IDs are independent per manager
    let sid_a = session_a.unwrap();
    let sid_b = session_b.unwrap();

    // Both start in HealDetection phase
    assert_eq!(
        manager_a.session_phase(sid_a),
        Some(MergePhase::HealDetection)
    );
    assert_eq!(
        manager_b.session_phase(sid_b),
        Some(MergePhase::HealDetection)
    );
}

#[test]
fn test_concurrent_session_limit() {
    let config = PartitionMergeConfig {
        merge_cooldown_ms: 0,
        max_concurrent_merges: 2,
        ..PartitionMergeConfig::default()
    };

    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    let s1 = manager.start_merge(vec!["peer_1".to_string()]);
    let s2 = manager.start_merge(vec!["peer_2".to_string()]);
    let s3 = manager.start_merge(vec!["peer_3".to_string()]);

    assert!(s1.is_some(), "First session should start");
    assert!(s2.is_some(), "Second session should start");
    assert!(s3.is_none(), "Third session should be blocked by limit");

    assert_eq!(manager.active_session_count(), 2);
}

#[test]
fn test_repartition_during_merge_aborts_session() {
    let config = default_config();
    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    // Start a merge session
    let sid = manager.start_merge(vec!["node_b".to_string()]).unwrap();
    assert_eq!(manager.session_phase(sid), Some(MergePhase::HealDetection));

    // Simulate a new partition event during the merge
    manager.notify_partition_event();

    // Repartition should be detected
    assert!(
        manager.is_repartitioned(sid),
        "Should detect repartition during active merge"
    );

    // Advancing the session should fail it
    let phase = manager.advance_session(sid);
    assert_eq!(
        phase,
        Some(MergePhase::Failed),
        "Session should fail on repartition detection"
    );
}

#[test]
fn test_no_repartition_without_event() {
    let config = default_config();
    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    let sid = manager.start_merge(vec!["node_b".to_string()]).unwrap();

    // No partition events -- should not be repartitioned
    assert!(
        !manager.is_repartitioned(sid),
        "No partition event, should not be repartitioned"
    );

    // Session should advance normally
    let phase = manager.advance_session(sid);
    assert_eq!(phase, Some(MergePhase::ViewExchange));

    let phase = manager.advance_session(sid);
    assert_eq!(phase, Some(MergePhase::MembershipReconciliation));
}

#[test]
fn test_session_phases_advance_in_order() {
    let config = default_config();
    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    let sid = manager.start_merge(vec!["node_b".to_string()]).unwrap();

    let expected_phases = [
        MergePhase::ViewExchange,
        MergePhase::MembershipReconciliation,
        MergePhase::DataReconciliation,
        MergePhase::TransactionReconciliation,
        MergePhase::Finalization,
        MergePhase::Completed,
    ];

    for expected in &expected_phases {
        let phase = manager.advance_session(sid).unwrap();
        assert_eq!(phase, *expected, "Phase should advance to {expected:?}");
    }
}

#[test]
fn test_complete_session_removes_it() {
    let config = default_config();
    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    let sid = manager.start_merge(vec!["node_b".to_string()]).unwrap();
    assert_eq!(manager.active_session_count(), 1);

    manager.complete_session(sid);

    // Session should be removed
    assert_eq!(manager.active_session_count(), 0);
    assert_eq!(
        manager.session_phase(sid),
        None,
        "Completed session should not exist"
    );
}

#[test]
fn test_fail_session_keeps_in_map() {
    let config = default_config();
    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    let sid = manager.start_merge(vec!["node_b".to_string()]).unwrap();
    manager.fail_session(sid, "test failure");

    // Failed session stays in map (unlike completed which is removed)
    assert_eq!(
        manager.session_phase(sid),
        Some(MergePhase::Failed),
        "Failed session should remain with Failed phase"
    );
}

#[test]
fn test_multiple_partition_events_increments_generation() {
    let config = default_config();
    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    let sid = manager.start_merge(vec!["node_b".to_string()]).unwrap();

    // Multiple partition events
    manager.notify_partition_event();
    manager.notify_partition_event();
    manager.notify_partition_event();

    // Still detected as repartitioned
    assert!(manager.is_repartitioned(sid));

    // Advance should fail
    assert_eq!(manager.advance_session(sid), Some(MergePhase::Failed));
}

#[test]
fn test_stats_track_sessions() {
    let config = default_config();
    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    let sid1 = manager.start_merge(vec!["peer_1".to_string()]).unwrap();
    let sid2 = manager.start_merge(vec!["peer_2".to_string()]).unwrap();

    manager.complete_session(sid1);
    manager.fail_session(sid2, "test");

    let stats = manager.stats_snapshot();
    assert_eq!(stats.sessions_started, 2);
    assert_eq!(stats.sessions_completed, 1);
    assert_eq!(stats.sessions_failed, 1);
}

#[test]
fn test_repartition_aborted_metric() {
    let config = default_config();
    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    let sid = manager.start_merge(vec!["node_b".to_string()]).unwrap();

    // Trigger repartition and advance to cause abort
    manager.notify_partition_event();
    let _ = manager.advance_session(sid);

    let stats = manager.stats_snapshot();
    assert!(
        stats.merge_aborted_repartition > 0,
        "Should track repartition abort metric"
    );
}

#[test]
fn test_cooldown_blocks_immediate_retry() {
    let config = PartitionMergeConfig {
        merge_cooldown_ms: 60_000, // 60 second cooldown
        max_concurrent_merges: 5,
        ..PartitionMergeConfig::default()
    };

    let manager = PartitionMergeManager::new("node_a".to_string(), config);

    // First merge with peer_1 should work
    let s1 = manager.start_merge(vec!["peer_1".to_string()]);
    assert!(s1.is_some());

    // Immediate retry with same peer should be blocked by cooldown
    let s2 = manager.start_merge(vec!["peer_1".to_string()]);
    assert!(
        s2.is_none(),
        "Cooldown should block immediate retry with same peer"
    );

    // Different peer should still work
    let s3 = manager.start_merge(vec!["peer_2".to_string()]);
    assert!(s3.is_some(), "Different peer should not be blocked");
}
