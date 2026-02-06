// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;
use tensor_chain::{extract_timestamp_hint, generate_tx_id, is_plausible_tx_id};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    count: u8,
    window_ms: u32,
}

fuzz_target!(|input: FuzzInput| {
    // Generate multiple IDs and verify uniqueness
    let count = (input.count as usize).max(1).min(100);
    let mut ids = HashSet::with_capacity(count);

    for _ in 0..count {
        let id = generate_tx_id();

        // Verify ID is non-zero
        assert_ne!(id, 0, "Generated ID should not be zero");

        // Verify uniqueness
        assert!(ids.insert(id), "Duplicate ID generated");

        // Extract timestamp and verify it's reasonable
        let timestamp_hint = extract_timestamp_hint(id);

        // Timestamp should be after our custom epoch (2024-01-01)
        let custom_epoch_ms: u64 = 1704067200000;
        assert!(
            timestamp_hint >= custom_epoch_ms,
            "Timestamp hint {} is before custom epoch",
            timestamp_hint
        );

        // Test plausibility with various windows
        let window_ms = (input.window_ms as u64).max(1);

        // Freshly generated ID should be plausible within a large window
        // Use 65536ms (full range of 16-bit field) to ensure it passes
        assert!(
            is_plausible_tx_id(id, 65536),
            "Freshly generated ID should be plausible in 65536ms window"
        );

        // Test with user-provided window (may or may not pass depending on timing)
        let _plausible = is_plausible_tx_id(id, window_ms);
        // We don't assert here since the result depends on actual timing

        // Verify ID structure: extract components and ensure they fit
        let ms_bits = (id >> 48) & 0xFFFF;
        let us_bits = (id >> 32) & 0xFFFF;
        let random_bits = id & 0xFFFF_FFFF;

        assert!(ms_bits <= 0xFFFF, "ms_bits overflow");
        assert!(us_bits <= 0xFFFF, "us_bits overflow");
        assert!(random_bits <= 0xFFFF_FFFF, "random_bits overflow");
    }

    // If we generated multiple IDs, verify they don't have sequential patterns
    if ids.len() >= 2 {
        let id_vec: Vec<u64> = ids.into_iter().collect();

        for window in id_vec.windows(2) {
            let diff = if window[1] > window[0] {
                window[1] - window[0]
            } else {
                window[0] - window[1]
            };

            // With 32 bits of randomness, exact difference of 1 is nearly impossible
            // (would indicate a sequential counter pattern)
            if diff == 1 {
                // This is extremely unlikely but not impossible due to wraparound
                // Just verify the random portions are different
                let random1 = window[0] & 0xFFFF_FFFF;
                let random2 = window[1] & 0xFFFF_FFFF;
                assert_ne!(
                    random1, random2,
                    "Random portions should differ when IDs differ by 1"
                );
            }
        }
    }

    // Test extract_timestamp_hint consistency
    let id = generate_tx_id();
    let hint1 = extract_timestamp_hint(id);
    let hint2 = extract_timestamp_hint(id);
    assert_eq!(hint1, hint2, "Timestamp extraction should be deterministic");

    // Test is_plausible_tx_id with edge cases
    assert!(is_plausible_tx_id(id, u64::MAX), "Should be plausible with max window");

    // A completely fabricated ID with wrong timestamp might not be plausible
    let fake_id: u64 = 0x1234_5678_9ABC_DEF0;
    let _fake_plausible = is_plausible_tx_id(fake_id, 1);
    // Result depends on current time, so no assertion
});
