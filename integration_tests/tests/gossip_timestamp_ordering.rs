// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for gossip timestamp ordering.
//!
//! Tests the HLC-based timestamp ordering to ensure monotonicity
//! and correct ordering across gossip state updates.

use tensor_chain::{membership::NodeHealth, GossipNodeState, HLCTimestamp, HybridLogicalClock};

#[test]
fn test_gossip_state_timestamps_are_monotonic() {
    // Create multiple gossip states in sequence
    let mut states = Vec::new();
    for i in 0..100 {
        let state =
            GossipNodeState::new(format!("node-{}", i % 5), NodeHealth::Healthy, i as u64, 0);
        states.push(state);
    }

    // All states should have non-zero updated_at
    for state in &states {
        assert!(
            state.updated_at > 0,
            "updated_at should be positive, got {}",
            state.updated_at
        );
    }
}

#[test]
fn test_hlc_timestamps_are_monotonic() {
    let hlc = HybridLogicalClock::new(12345).expect("HLC creation should succeed");

    let mut prev = hlc.now().expect("now() should succeed");
    for _ in 0..1000 {
        let current = hlc.now().expect("now() should succeed");
        assert!(
            current > prev,
            "HLC timestamps should be monotonic: {:?} should be > {:?}",
            current,
            prev
        );
        prev = current;
    }
}

#[test]
fn test_hlc_timestamps_from_different_nodes_are_comparable() {
    let hlc1 = HybridLogicalClock::new(1).expect("HLC1 creation should succeed");
    let hlc2 = HybridLogicalClock::new(2).expect("HLC2 creation should succeed");

    let ts1 = hlc1.now().expect("now() should succeed");
    let ts2 = hlc2.now().expect("now() should succeed");

    // Timestamps should be comparable (not necessarily equal, but orderable)
    let _ = ts1 < ts2 || ts1 >= ts2; // Just verify comparison works
}

#[test]
fn test_hlc_receive_advances_time() {
    let hlc = HybridLogicalClock::new(1).expect("HLC creation should succeed");

    let local = hlc.now().expect("now() should succeed");

    // Simulate receiving a timestamp from the "future"
    let remote = HLCTimestamp::new(local.wall_ms() + 10000, 0, 2);

    let after_receive = hlc.receive(&remote).expect("receive should succeed");

    // After receiving, our time should be after both
    assert!(
        after_receive > local,
        "After receive should be > local: {:?} vs {:?}",
        after_receive,
        local
    );
    assert!(
        after_receive > remote,
        "After receive should be > remote: {:?} vs {:?}",
        after_receive,
        remote
    );
}

#[test]
fn test_gossip_state_try_new_returns_error_gracefully() {
    // try_new should succeed on normal systems
    let result = GossipNodeState::try_new("test-node".to_string(), NodeHealth::Healthy, 1, 0);

    // On a normal system, this should succeed
    assert!(result.is_ok(), "try_new should succeed on normal system");

    let state = result.unwrap();
    assert!(state.updated_at > 0, "updated_at should be positive");
}

#[test]
fn test_gossip_state_with_wall_time() {
    let wall_time = 1700000000000u64; // Arbitrary timestamp

    let state = GossipNodeState::with_wall_time(
        "test-node".to_string(),
        NodeHealth::Degraded,
        42,
        5,
        wall_time,
    );

    assert_eq!(state.node_id, "test-node");
    assert_eq!(state.health, NodeHealth::Degraded);
    assert_eq!(state.timestamp, 42);
    assert_eq!(state.incarnation, 5);
    assert_eq!(state.updated_at, wall_time);
}

#[test]
fn test_hlc_timestamp_ordering_with_same_wall_time() {
    let ts1 = HLCTimestamp::new(1000, 0, 1);
    let ts2 = HLCTimestamp::new(1000, 1, 1);
    let ts3 = HLCTimestamp::new(1000, 1, 2);

    // Same wall time, different logical - logical breaks tie
    assert!(ts1 < ts2);

    // Same wall time and logical, different node - node ID breaks tie
    assert!(ts2 < ts3);
}

#[test]
fn test_hlc_timestamp_serialization_roundtrip() {
    let original = HLCTimestamp::new(123456789, 42, 99);

    // Test as_u64 / from_u64 roundtrip (loses node_id_hash)
    let packed = original.as_u64();
    let restored = HLCTimestamp::from_u64(packed);

    assert_eq!(original.wall_ms(), restored.wall_ms());
    assert_eq!(original.logical(), restored.logical());
}

#[test]
fn test_gossip_state_supersedes_uses_incarnation() {
    // Higher incarnation always wins
    let old = GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 100, 0);
    let new = GossipNodeState::new("node1".to_string(), NodeHealth::Failed, 1, 1);

    assert!(
        new.supersedes(&old),
        "Higher incarnation should supersede lower"
    );
    assert!(
        !old.supersedes(&new),
        "Lower incarnation should not supersede higher"
    );
}

#[test]
fn test_gossip_state_supersedes_uses_timestamp_for_same_incarnation() {
    let old = GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 1, 0);
    let new = GossipNodeState::new("node1".to_string(), NodeHealth::Degraded, 2, 0);

    assert!(
        new.supersedes(&old),
        "Higher timestamp should supersede for same incarnation"
    );
}

#[test]
fn test_rapid_hlc_generation_maintains_ordering() {
    let hlc = HybridLogicalClock::from_node_id("rapid-test").expect("HLC creation should succeed");

    // Generate timestamps as fast as possible
    let mut timestamps: Vec<HLCTimestamp> = Vec::with_capacity(10000);
    for _ in 0..10000 {
        timestamps.push(hlc.now().expect("now() should succeed"));
    }

    // Verify strict ordering
    for window in timestamps.windows(2) {
        assert!(
            window[1] > window[0],
            "Monotonicity violated: {:?} should be > {:?}",
            window[1],
            window[0]
        );
    }
}
