// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_chain::{SequenceTracker, SequenceTrackerConfig};

fuzz_target!(|data: Vec<(String, u64)>| {
    // Create a tracker with a small limit to test bounds
    let config = SequenceTrackerConfig::default()
        .with_max_entries(100)
        .with_cleanup_interval(50);

    let tracker = SequenceTracker::with_config(config);

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    // Try to add many senders
    for (sender, sequence) in data.iter() {
        let _ = tracker.check_and_record(sender, *sequence, now_ms);
    }

    // Invariant: tracker should never exceed max_entries
    assert!(
        tracker.len() <= 100,
        "Tracker exceeded max_entries: {} > 100",
        tracker.len()
    );

    // Invariant: len() and is_empty() should be consistent
    assert_eq!(tracker.is_empty(), tracker.len() == 0);
});
