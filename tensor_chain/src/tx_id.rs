// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Cryptographically secure transaction ID generation with time ordering.
//!
//! # Overview
//!
//! This module generates 64-bit transaction IDs that combine time-based ordering with
//! cryptographic unpredictability. The design ensures:
//!
//! - **Approximate time ordering**: IDs generated close in time sort close together
//! - **Unpredictability**: 32 bits of CSPRNG randomness prevents ID guessing
//! - **Uniqueness**: Microsecond precision + overflow counter + randomness prevents collisions
//! - **Compactness**: Fits in a single `u64` for efficient storage and transmission
//!
//! # ID Structure
//!
//! ```text
//! 63      48 47      32 31                              0
//! +----------+----------+--------------------------------+
//! | ms (16b) | us+ovf   |       random (32 bits)         |
//! +----------+----------+--------------------------------+
//! ```
//!
//! | Bits | Field | Description |
//! |------|-------|-------------|
//! | 63-48 | Milliseconds | Time since custom epoch, mod 65536 (~65 seconds cycle) |
//! | 47-32 | Microseconds + Overflow | Sub-millisecond precision + collision counter |
//! | 31-0 | Random | CSPRNG output from `OsRng` |
//!
//! # Custom Epoch
//!
//! The module uses a custom epoch of 2024-01-01 00:00:00 UTC instead of the Unix epoch.
//! This provides more bits of useful timestamp information within the 16-bit millisecond
//! field, extending the practical range of time ordering.
//!
//! # Collision Handling
//!
//! When multiple IDs are generated within the same millisecond:
//!
//! 1. The overflow counter increments atomically
//! 2. This is added to the microsecond field
//! 3. Combined with 32 bits of randomness, collisions are essentially impossible
//!
//! With 32 bits of randomness, the birthday bound suggests ~65,000 IDs before a 50%
//! collision probability in the random portion alone. Combined with the timestamp
//! and overflow counter, practical collision probability is negligible.
//!
//! # Usage
//!
//! ## Generating IDs
//!
//! ```rust
//! use tensor_chain::tx_id::generate_tx_id;
//!
//! // Generate unique transaction ID
//! let tx_id = generate_tx_id();
//!
//! // IDs are sortable (approximately by time)
//! let tx_id2 = generate_tx_id();
//! // tx_id < tx_id2 with high probability if generated in sequence
//! ```
//!
//! ## Extracting Timestamp Hints
//!
//! ```rust
//! use tensor_chain::tx_id::{generate_tx_id, extract_timestamp_hint};
//!
//! let tx_id = generate_tx_id();
//!
//! // Get approximate creation time (milliseconds since Unix epoch)
//! let hint = extract_timestamp_hint(tx_id);
//! ```
//!
//! ## Validating Plausibility
//!
//! ```rust
//! use tensor_chain::tx_id::{generate_tx_id, is_plausible_tx_id};
//!
//! let tx_id = generate_tx_id();
//!
//! // Check if ID was plausibly generated recently
//! let window_ms = 10_000;  // 10 second window
//! if is_plausible_tx_id(tx_id, window_ms) {
//!     // ID timestamp is within acceptable range
//! } else {
//!     // ID may be invalid, stale, or replayed
//! }
//! ```
//!
//! # Security Properties
//!
//! ## Unpredictability
//!
//! With 32 bits of CSPRNG randomness, an attacker cannot predict future IDs even if
//! they know the exact timestamp. This prevents:
//! - Pre-computing valid transaction IDs
//! - Guessing IDs for unauthorized access
//! - Timing attacks based on sequential patterns
//!
//! ## Replay Detection
//!
//! The [`is_plausible_tx_id`] function enables detection of obviously replayed or
//! forged IDs by checking if the timestamp component falls within an acceptable
//! window of the current time.
//!
//! ## Thread Safety
//!
//! The overflow counter uses atomic operations, making [`generate_tx_id`] safe to
//! call concurrently from multiple threads without locks.
//!
//! # Limitations
//!
//! - **Time ordering is approximate**: The 16-bit millisecond field wraps every ~65 seconds
//! - **Timestamp extraction is imprecise**: Only 16 bits of millisecond precision
//! - **Clock skew**: IDs from different machines may not be perfectly ordered
//!
//! # See Also
//!
//! - [`crate::distributed_tx`]: 2PC coordinator using transaction IDs
//! - [`crate::hlc`]: Hybrid logical clocks for cross-node ordering

use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::RngCore;

/// Custom epoch: 2024-01-01 00:00:00 UTC in milliseconds since UNIX epoch.
/// Using a custom epoch extends the useful range of the 16-bit millisecond field.
const CUSTOM_EPOCH_MS: u64 = 1_704_067_200_000;

/// Overflow counter for same-millisecond collisions.
static OVERFLOW_COUNTER: AtomicU16 = AtomicU16::new(0);

/// Last timestamp seen (milliseconds since custom epoch).
static LAST_TIMESTAMP: AtomicU64 = AtomicU64::new(0);

/// Generate a cryptographically secure transaction ID.
///
/// The ID combines:
/// - Timestamp (32 bits total: 16 bits ms, 16 bits us + overflow)
/// - Random (32 bits from `OsRng`)
///
/// This prevents ID prediction while maintaining approximate time ordering.
#[must_use]
pub fn generate_tx_id() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    #[allow(clippy::cast_possible_truncation)]
    let ms = now.as_millis() as u64;
    #[allow(clippy::cast_possible_truncation)]
    let us = (now.as_micros() % 1000) as u16;

    // Handle same-millisecond collisions with overflow counter
    let current_ts = ms.wrapping_sub(CUSTOM_EPOCH_MS);
    let last_ts = LAST_TIMESTAMP.swap(current_ts, Ordering::Relaxed);

    let overflow = if current_ts == last_ts {
        OVERFLOW_COUNTER.fetch_add(1, Ordering::Relaxed)
    } else {
        OVERFLOW_COUNTER.store(0, Ordering::Relaxed);
        0
    };

    // Get 32 bits of cryptographic randomness
    let random_bits = u64::from(rand::rng().next_u32());

    // Assemble the ID:
    // - Bits 63-48: milliseconds (mod 65536)
    // - Bits 47-32: microseconds + overflow
    // - Bits 31-0: random
    let ms_bits = (current_ts & 0xFFFF) << 48;
    let us_bits = u64::from(us.wrapping_add(overflow)) << 32;

    ms_bits | us_bits | random_bits
}

/// Extract the approximate timestamp hint from a transaction ID.
///
/// Returns the estimated milliseconds since UNIX epoch when the ID was generated.
/// This is only an approximation due to the 16-bit truncation of the millisecond field.
#[must_use]
pub const fn extract_timestamp_hint(tx_id: u64) -> u64 {
    let ms_component = (tx_id >> 48) & 0xFFFF;
    CUSTOM_EPOCH_MS + ms_component
}

/// Check if a transaction ID is plausible given a time window.
///
/// Returns true if the ID's timestamp component falls within `window_ms`
/// milliseconds of the current time. This can be used to detect obviously
/// invalid or replayed transaction IDs.
#[must_use]
pub fn is_plausible_tx_id(tx_id: u64, window_ms: u64) -> bool {
    #[allow(clippy::cast_possible_truncation)]
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let current_ms_component = (now.wrapping_sub(CUSTOM_EPOCH_MS)) & 0xFFFF;
    let id_ms_component = (tx_id >> 48) & 0xFFFF;

    // Handle wraparound in the 16-bit field
    let diff = if id_ms_component > current_ms_component {
        // Could be future or wrapped around
        let forward_diff = id_ms_component - current_ms_component;
        let backward_diff = (0x10000 - id_ms_component) + current_ms_component;
        forward_diff.min(backward_diff)
    } else {
        current_ms_component - id_ms_component
    };

    diff <= window_ms
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::thread;

    #[test]
    fn test_uniqueness() {
        let mut ids = HashSet::new();
        for _ in 0..10_000 {
            let id = generate_tx_id();
            assert!(ids.insert(id), "Duplicate ID generated: {}", id);
        }
    }

    #[test]
    fn test_unpredictability() {
        // Generate two consecutive IDs
        let id1 = generate_tx_id();
        let id2 = generate_tx_id();

        // The random portions (lower 32 bits) should differ significantly
        let random1 = id1 & 0xFFFF_FFFF;
        let random2 = id2 & 0xFFFF_FFFF;

        // With 32 bits of randomness, the chance of collision is 1 in 4 billion
        assert_ne!(random1, random2, "Random portions should differ");
    }

    #[test]
    fn test_timestamp_extraction() {
        let before = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let id = generate_tx_id();
        let extracted = extract_timestamp_hint(id);

        let after = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // The extracted timestamp should be close to the actual time
        // (within the 65536ms window of the 16-bit field)
        let before_component = (before.wrapping_sub(CUSTOM_EPOCH_MS)) & 0xFFFF;
        let after_component = (after.wrapping_sub(CUSTOM_EPOCH_MS)) & 0xFFFF;
        let extracted_component = (extracted.wrapping_sub(CUSTOM_EPOCH_MS)) & 0xFFFF;

        // Allow for wraparound
        let in_range = if before_component <= after_component {
            extracted_component >= before_component && extracted_component <= after_component
        } else {
            // Wraparound case
            extracted_component >= before_component || extracted_component <= after_component
        };

        assert!(
            in_range,
            "Extracted timestamp {} not in range [{}, {}]",
            extracted_component, before_component, after_component
        );
    }

    #[test]
    fn test_concurrent_generation() {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                thread::spawn(|| {
                    let mut ids = Vec::with_capacity(1000);
                    for _ in 0..1000 {
                        ids.push(generate_tx_id());
                    }
                    ids
                })
            })
            .collect();

        let mut all_ids = HashSet::new();
        for handle in handles {
            let ids = handle.join().unwrap();
            for id in ids {
                assert!(
                    all_ids.insert(id),
                    "Duplicate ID in concurrent test: {}",
                    id
                );
            }
        }

        assert_eq!(all_ids.len(), 8000);
    }

    #[test]
    fn test_no_sequential_pattern() {
        let ids: Vec<u64> = (0..100).map(|_| generate_tx_id()).collect();

        // Check that no consecutive IDs differ by exactly 1 (would indicate sequential pattern)
        for window in ids.windows(2) {
            let diff = window[1].abs_diff(window[0]);
            // With 32 bits of randomness, difference of exactly 1 is astronomically unlikely
            assert_ne!(
                diff, 1,
                "Sequential pattern detected: {} and {}",
                window[0], window[1]
            );
        }
    }

    #[test]
    fn test_plausibility_check_valid() {
        let id = generate_tx_id();
        // ID just generated should be plausible within a 10 second window
        assert!(is_plausible_tx_id(id, 10_000));
    }

    #[test]
    fn test_plausibility_check_with_small_window() {
        let id = generate_tx_id();
        // Wait a bit and check with a very small window
        thread::sleep(std::time::Duration::from_millis(10));
        // Should still be plausible within 100ms
        assert!(is_plausible_tx_id(id, 100));
    }

    #[test]
    fn test_custom_epoch() {
        // Verify our custom epoch is reasonable (should be in the past)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        assert!(CUSTOM_EPOCH_MS < now, "Custom epoch should be in the past");

        // Should be 2024-01-01 00:00:00 UTC
        // This is approximately 1704067200000 ms since UNIX epoch
        assert_eq!(
            CUSTOM_EPOCH_MS, 1704067200000,
            "Custom epoch should be 2024-01-01"
        );
    }

    #[test]
    fn test_id_structure() {
        let id = generate_tx_id();

        // Extract components
        let ms_bits = (id >> 48) & 0xFFFF;
        let us_bits = (id >> 32) & 0xFFFF;
        let random_bits = id & 0xFFFF_FFFF;

        // ms_bits should be less than 65536 (16 bits)
        assert!(ms_bits < 0x10000);
        // us_bits should be less than 65536 (16 bits)
        assert!(us_bits < 0x10000);
        // random_bits should be less than 2^32 (32 bits)
        assert!(random_bits < 0x1_0000_0000);
    }

    #[test]
    fn test_overflow_counter() {
        // Generate many IDs rapidly to trigger overflow counter
        let mut ids = HashSet::new();
        for _ in 0..1000 {
            let id = generate_tx_id();
            assert!(ids.insert(id), "Duplicate ID with overflow: {}", id);
        }
    }

    #[test]
    fn test_id_nonzero() {
        // All generated IDs should be non-zero due to randomness
        for _ in 0..100 {
            let id = generate_tx_id();
            assert_ne!(id, 0, "Generated ID should not be zero");
        }
    }

    #[test]
    fn test_extract_timestamp_hint_consistency() {
        let id = generate_tx_id();
        let hint1 = extract_timestamp_hint(id);
        let hint2 = extract_timestamp_hint(id);

        // Same ID should always give same hint
        assert_eq!(hint1, hint2);
    }

    #[test]
    fn test_bits_distribution() {
        // Generate IDs and verify randomness distribution
        let ids: Vec<u64> = (0..1000).map(|_| generate_tx_id()).collect();

        // Count set bits in the random portion across all IDs
        let mut bit_counts = [0u32; 32];
        for id in &ids {
            let random = id & 0xFFFF_FFFF;
            for (i, count) in bit_counts.iter_mut().enumerate() {
                if (random >> i) & 1 == 1 {
                    *count += 1;
                }
            }
        }

        // Each bit should be set roughly 50% of the time (with some variance)
        // For 1000 samples, we expect ~500 with reasonable variance
        for (i, &count) in bit_counts.iter().enumerate() {
            assert!(
                count > 300 && count < 700,
                "Bit {} has suspicious distribution: {} set out of 1000",
                i,
                count
            );
        }
    }

    #[test]
    fn test_is_plausible_tx_id_wraparound() {
        // Construct an ID where the ms component is far ahead of current time
        // in the 16-bit field, triggering the wraparound branch
        #[allow(clippy::cast_possible_truncation)]
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let current_ms_component = (now.wrapping_sub(CUSTOM_EPOCH_MS)) & 0xFFFF;

        // Place the ID ms component just ahead of current, within window
        let id_ms = (current_ms_component + 5) & 0xFFFF;
        let tx_id = id_ms << 48;
        assert!(is_plausible_tx_id(tx_id, 100));

        // Place the ID ms component far ahead (near wraparound), outside window
        let id_ms_far = (current_ms_component.wrapping_add(0x8000)) & 0xFFFF;
        let tx_id_far = id_ms_far << 48;
        assert!(!is_plausible_tx_id(tx_id_far, 100));
    }
}
