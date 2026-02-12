// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for SequenceTracker bounded growth and cleanup.
//!
//! Tests that the SequenceTracker prevents unbounded memory growth
//! through max_entries limits and TTL-based cleanup.

use tensor_chain::{SequenceTracker, SequenceTrackerConfig};

// ============================================================================
// Max Entries Bounds Tests
// ============================================================================

#[test]
fn test_sequence_tracker_bounded_growth() {
    let config = SequenceTrackerConfig::default()
        .with_max_entries(100)
        .with_cleanup_interval(10000); // High interval to avoid auto-cleanup

    let tracker = SequenceTracker::with_config(config);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Fill up the tracker to capacity
    for i in 0..100 {
        let sender = format!("sender_{}", i);
        let result = tracker.check_and_record(&sender, 1, now_ms);
        assert!(result.is_ok(), "Sender {} should be accepted", i);
    }

    assert_eq!(tracker.len(), 100);

    // Next new sender should be rejected
    let result = tracker.check_and_record(&"overflow_sender".to_string(), 1, now_ms);
    assert!(result.is_err(), "Overflow sender should be rejected");
    assert!(result.unwrap_err().to_string().contains("at capacity"));

    // Tracker size should still be 100
    assert_eq!(tracker.len(), 100);
}

#[test]
fn test_sequence_tracker_existing_sender_not_rejected() {
    let config = SequenceTrackerConfig::default()
        .with_max_entries(10)
        .with_cleanup_interval(10000);

    let tracker = SequenceTracker::with_config(config);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Fill up the tracker
    for i in 0..10 {
        let sender = format!("sender_{}", i);
        tracker.check_and_record(&sender, 1, now_ms).unwrap();
    }

    assert_eq!(tracker.len(), 10);

    // Existing senders should still be able to update
    for i in 0..10 {
        let sender = format!("sender_{}", i);
        let result = tracker.check_and_record(&sender, 2, now_ms);
        assert!(result.is_ok(), "Existing sender {} should update", i);
    }

    // Size should remain the same
    assert_eq!(tracker.len(), 10);
}

// ============================================================================
// Multiple Attack Patterns
// ============================================================================

#[test]
fn test_sequence_tracker_many_unique_senders_attack() {
    // Simulates an attacker trying to exhaust memory with many unique senders
    let config = SequenceTrackerConfig::default()
        .with_max_entries(50)
        .with_cleanup_interval(10000);

    let tracker = SequenceTracker::with_config(config);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let mut accepted = 0;
    let mut rejected = 0;

    // Try to add many more senders than limit
    for i in 0..1000 {
        let sender = format!("attacker_{}", i);
        match tracker.check_and_record(&sender, 1, now_ms) {
            Ok(_) => accepted += 1,
            Err(_) => rejected += 1,
        }
    }

    // Should accept exactly 50, reject the rest
    assert_eq!(accepted, 50);
    assert_eq!(rejected, 950);
    assert_eq!(tracker.len(), 50);
}

#[test]
fn test_sequence_tracker_interleaved_updates() {
    // Tests that existing senders can update while new senders are rejected
    let config = SequenceTrackerConfig::default()
        .with_max_entries(5)
        .with_cleanup_interval(10000);

    let tracker = SequenceTracker::with_config(config);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Add 5 legitimate senders
    for i in 0..5 {
        let sender = format!("legit_{}", i);
        tracker.check_and_record(&sender, 1, now_ms).unwrap();
    }

    // Interleave: update existing, try new, update existing
    for round in 0..10 {
        // Update existing senders
        for i in 0..5 {
            let sender = format!("legit_{}", i);
            let seq = (round * 5 + i + 2) as u64;
            tracker.check_and_record(&sender, seq, now_ms).unwrap();
        }

        // Try new sender (should fail)
        let new_sender = format!("new_{}", round);
        let result = tracker.check_and_record(&new_sender, 1, now_ms);
        assert!(result.is_err());
    }

    // Only the original 5 should be tracked
    assert_eq!(tracker.len(), 5);
}

// ============================================================================
// Len and IsEmpty Tests
// ============================================================================

#[test]
fn test_sequence_tracker_len_is_empty() {
    let tracker = SequenceTracker::new();

    assert!(tracker.is_empty());
    assert_eq!(tracker.len(), 0);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    tracker
        .check_and_record(&"sender1".to_string(), 1, now_ms)
        .unwrap();

    assert!(!tracker.is_empty());
    assert_eq!(tracker.len(), 1);

    tracker
        .check_and_record(&"sender2".to_string(), 1, now_ms)
        .unwrap();

    assert_eq!(tracker.len(), 2);

    tracker.clear();

    assert!(tracker.is_empty());
    assert_eq!(tracker.len(), 0);
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_sequence_tracker_config_builder() {
    let config = SequenceTrackerConfig::default()
        .with_max_age_ms(60_000)
        .with_max_entries(500)
        .with_cleanup_interval(50);

    assert_eq!(config.max_age_ms, 60_000);
    assert_eq!(config.max_entries, 500);
    assert_eq!(config.cleanup_interval, 50);
}

#[test]
fn test_sequence_tracker_default_config() {
    let config = SequenceTrackerConfig::default();

    assert_eq!(config.max_age_ms, 5 * 60 * 1000); // 5 minutes
    assert_eq!(config.max_entries, 10_000);
    assert_eq!(config.cleanup_interval, 100);
}

#[test]
fn test_sequence_tracker_with_custom_config() {
    let config = SequenceTrackerConfig::default()
        .with_max_entries(25)
        .with_max_age_ms(1000);

    let tracker = SequenceTracker::with_config(config);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Fill to capacity
    for i in 0..25 {
        tracker
            .check_and_record(&format!("s{}", i), 1, now_ms)
            .unwrap();
    }

    // Next should fail
    let result = tracker.check_and_record(&"overflow".to_string(), 1, now_ms);
    assert!(result.is_err());
}
