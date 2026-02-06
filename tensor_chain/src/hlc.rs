// SPDX-License-Identifier: MIT OR Apache-2.0
//! Hybrid Logical Clock (HLC) for distributed timestamp ordering.
//!
//! Provides monotonically increasing timestamps that combine wall clock
//! time with logical counters to ensure ordering even when system time
//! goes backwards or fails.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::ChainError;

/// A hybrid logical clock timestamp.
///
/// Combines wall clock time with a logical counter and node identifier
/// to provide globally unique, monotonically increasing timestamps.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct HLCTimestamp {
    /// Wall clock time in milliseconds since UNIX epoch.
    wall_ms: u64,
    /// Logical counter for ordering within the same millisecond.
    logical: u64,
    /// Hash of the node ID for tie-breaking.
    node_id_hash: u32,
}

impl HLCTimestamp {
    #[must_use]
    pub fn new(wall_ms: u64, logical: u64, node_id_hash: u32) -> Self {
        Self {
            wall_ms,
            logical,
            node_id_hash,
        }
    }

    /// Create a timestamp from a raw u64 value.
    ///
    /// Uses upper 48 bits for `wall_ms`, lower 16 for logical.
    /// Node ID hash is set to 0.
    #[must_use]
    pub fn from_u64(value: u64) -> Self {
        Self {
            wall_ms: value >> 16,
            logical: value & 0xFFFF,
            node_id_hash: 0,
        }
    }

    /// Convert to a packed u64 representation.
    ///
    /// Uses upper 48 bits for `wall_ms`, lower 16 for logical.
    /// Note: this loses the `node_id_hash` for compact storage.
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        (self.wall_ms << 16) | (self.logical & 0xFFFF)
    }

    /// Get the wall clock component in milliseconds.
    #[must_use]
    pub fn wall_ms(&self) -> u64 {
        self.wall_ms
    }

    /// Get the logical counter component.
    #[must_use]
    pub fn logical(&self) -> u64 {
        self.logical
    }

    /// Get the node ID hash component.
    #[must_use]
    pub fn node_id_hash(&self) -> u32 {
        self.node_id_hash
    }

    /// Check if this timestamp is strictly before another.
    #[must_use]
    pub fn is_before(&self, other: &Self) -> bool {
        self < other
    }

    /// Check if this timestamp is strictly after another.
    #[must_use]
    pub fn is_after(&self, other: &Self) -> bool {
        self > other
    }
}

/// A Hybrid Logical Clock for generating monotonic timestamps.
///
/// This implementation uses a monotonic clock (Instant) anchored to an
/// initial wall clock reading to avoid issues with system time going
/// backwards or being unavailable.
#[derive(Debug)]
pub struct HybridLogicalClock {
    /// Last recorded wall time in milliseconds.
    last_wall_ms: AtomicU64,
    /// Logical counter for ordering within the same millisecond.
    logical: AtomicU64,
    /// Hash of the node ID for tie-breaking.
    node_id_hash: u32,
    /// Monotonic clock start time.
    monotonic_start: Instant,
    /// Wall clock time at start in milliseconds.
    wall_start_ms: u64,
    /// Simulated clock drift offset in milliseconds (for testing).
    drift_offset_ms: AtomicI64,
}

impl HybridLogicalClock {
    /// Create a new HLC for the given node ID.
    ///
    /// # Errors
    /// Returns an error if the system time is before the UNIX epoch.
    pub fn new(node_id: u64) -> Result<Self, ChainError> {
        #[allow(clippy::cast_possible_truncation)]
        let wall_start_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| ChainError::ClockError(format!("system time before epoch: {e}")))?
            .as_millis() as u64;

        #[allow(clippy::cast_possible_truncation)]
        let node_id_hash = (node_id & 0xFFFF_FFFF) as u32;

        Ok(Self {
            last_wall_ms: AtomicU64::new(wall_start_ms),
            logical: AtomicU64::new(0),
            node_id_hash,
            monotonic_start: Instant::now(),
            wall_start_ms,
            drift_offset_ms: AtomicI64::new(0),
        })
    }

    /// Create an HLC from a string node ID.
    ///
    /// # Errors
    /// Returns an error if the system time is before the UNIX epoch.
    pub fn from_node_id(node_id: &str) -> Result<Self, ChainError> {
        // Use a simple hash of the node ID string
        let hash = node_id.bytes().fold(0u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(u64::from(b))
        });
        Self::new(hash)
    }

    /// Compute the current wall time with drift applied.
    fn wall_with_drift(&self) -> u64 {
        #[allow(clippy::cast_possible_truncation)]
        let elapsed_ms = self.monotonic_start.elapsed().as_millis() as u64;
        let base_wall = self.wall_start_ms.saturating_add(elapsed_ms);
        let drift = self.drift_offset_ms.load(Ordering::SeqCst);
        if drift >= 0 {
            base_wall.saturating_add(drift.unsigned_abs())
        } else {
            base_wall.saturating_sub(drift.unsigned_abs())
        }
    }

    /// Get the current timestamp.
    ///
    /// This method is infallible after construction because it uses
    /// the monotonic clock anchored to the initial wall clock reading.
    ///
    /// # Errors
    /// This method currently always succeeds after construction.
    pub fn now(&self) -> Result<HLCTimestamp, ChainError> {
        let current_wall = self.wall_with_drift();

        let last = self.last_wall_ms.load(Ordering::SeqCst);

        if current_wall > last {
            // Wall clock advanced - reset logical counter
            self.last_wall_ms.store(current_wall, Ordering::SeqCst);
            self.logical.store(0, Ordering::SeqCst);
            Ok(HLCTimestamp {
                wall_ms: current_wall,
                logical: 0,
                node_id_hash: self.node_id_hash,
            })
        } else {
            // Same or earlier wall time - increment logical counter
            // Use saturating_add to prevent overflow at u64::MAX
            let prev = self.logical.fetch_add(1, Ordering::SeqCst);
            let logical = prev.saturating_add(1);
            Ok(HLCTimestamp {
                wall_ms: last,
                logical,
                node_id_hash: self.node_id_hash,
            })
        }
    }

    /// Update the clock based on a received timestamp.
    ///
    /// Returns a new timestamp that is guaranteed to be after both
    /// the local time and the received timestamp.
    ///
    /// # Errors
    /// This method currently always succeeds after construction.
    pub fn receive(&self, received: &HLCTimestamp) -> Result<HLCTimestamp, ChainError> {
        let current_wall = self.wall_with_drift();

        let last = self.last_wall_ms.load(Ordering::SeqCst);
        let max_wall = current_wall.max(last).max(received.wall_ms);

        if max_wall > last {
            self.last_wall_ms.store(max_wall, Ordering::SeqCst);
        }

        // Determine the new logical counter (use saturating_add to prevent overflow)
        let new_logical = if max_wall == last && max_wall == received.wall_ms {
            // All three equal - take max logical + 1
            let local_logical = self.logical.load(Ordering::SeqCst);
            local_logical.max(received.logical).saturating_add(1)
        } else if max_wall == last {
            // Local is ahead - increment local logical
            self.logical.load(Ordering::SeqCst).saturating_add(1)
        } else if max_wall == received.wall_ms {
            // Received is ahead - use received logical + 1
            received.logical.saturating_add(1)
        } else {
            // Current wall is ahead - reset logical
            0
        };

        self.logical.store(new_logical, Ordering::SeqCst);

        Ok(HLCTimestamp {
            wall_ms: max_wall,
            logical: new_logical,
            node_id_hash: self.node_id_hash,
        })
    }

    /// Get the node ID hash.
    #[must_use]
    pub fn node_id_hash(&self) -> u32 {
        self.node_id_hash
    }

    /// Get the current wall time estimate in milliseconds.
    #[must_use]
    pub fn estimated_wall_ms(&self) -> u64 {
        self.wall_with_drift()
    }

    /// Set the simulated clock drift offset in milliseconds.
    ///
    /// Positive values make the clock appear ahead; negative values make it
    /// appear behind. The HLC monotonicity guarantee is preserved regardless
    /// of drift direction.
    pub fn set_drift_offset(&self, offset_ms: i64) {
        self.drift_offset_ms.store(offset_ms, Ordering::SeqCst);
    }

    /// Simulate a clock jump (e.g. NTP correction) by adding to the current
    /// drift offset. Multiple jumps accumulate.
    pub fn inject_clock_jump(&self, jump_ms: i64) {
        self.drift_offset_ms.fetch_add(jump_ms, Ordering::SeqCst);
    }

    /// Get the current drift offset in milliseconds.
    #[must_use]
    pub fn drift_offset(&self) -> i64 {
        self.drift_offset_ms.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlc_timestamp_new() {
        let ts = HLCTimestamp::new(1000, 5, 42);
        assert_eq!(ts.wall_ms(), 1000);
        assert_eq!(ts.logical(), 5);
        assert_eq!(ts.node_id_hash(), 42);
    }

    #[test]
    fn test_hlc_timestamp_ordering() {
        let ts1 = HLCTimestamp::new(1000, 0, 1);
        let ts2 = HLCTimestamp::new(1000, 1, 1);
        let ts3 = HLCTimestamp::new(1001, 0, 1);

        assert!(ts1 < ts2);
        assert!(ts2 < ts3);
        assert!(ts1 < ts3);
        assert!(ts1.is_before(&ts2));
        assert!(ts3.is_after(&ts2));
    }

    #[test]
    fn test_hlc_timestamp_as_u64() {
        let ts = HLCTimestamp::new(0x1234, 0x0056, 99);
        let packed = ts.as_u64();
        // wall_ms << 16 | logical
        assert_eq!(packed, (0x1234u64 << 16) | 0x0056);
    }

    #[test]
    fn test_hlc_timestamp_from_u64() {
        let packed = (0x5678u64 << 16) | 0x00AB;
        let ts = HLCTimestamp::from_u64(packed);
        assert_eq!(ts.wall_ms(), 0x5678);
        assert_eq!(ts.logical(), 0x00AB);
        assert_eq!(ts.node_id_hash(), 0);
    }

    #[test]
    fn test_hlc_timestamp_roundtrip() {
        let original = HLCTimestamp::new(123456789, 42, 0);
        let packed = original.as_u64();
        let restored = HLCTimestamp::from_u64(packed);
        assert_eq!(original.wall_ms(), restored.wall_ms());
        assert_eq!(original.logical(), restored.logical());
    }

    #[test]
    fn test_hlc_timestamp_default() {
        let ts = HLCTimestamp::default();
        assert_eq!(ts.wall_ms(), 0);
        assert_eq!(ts.logical(), 0);
        assert_eq!(ts.node_id_hash(), 0);
    }

    #[test]
    fn test_hlc_new() {
        let hlc = HybridLogicalClock::new(12345).unwrap();
        assert_eq!(hlc.node_id_hash(), 12345);
    }

    #[test]
    fn test_hlc_from_node_id() {
        let hlc = HybridLogicalClock::from_node_id("node-1").unwrap();
        // Just verify it creates successfully
        assert!(hlc.node_id_hash() > 0 || hlc.node_id_hash() == 0);
    }

    #[test]
    fn test_hlc_now_monotonic() {
        let hlc = HybridLogicalClock::new(1).unwrap();

        let ts1 = hlc.now().unwrap();
        let ts2 = hlc.now().unwrap();
        let ts3 = hlc.now().unwrap();

        // Each timestamp should be strictly greater than the previous
        assert!(ts2 > ts1, "ts2 {:?} should be > ts1 {:?}", ts2, ts1);
        assert!(ts3 > ts2, "ts3 {:?} should be > ts2 {:?}", ts3, ts2);
    }

    #[test]
    fn test_hlc_receive_advances() {
        let hlc = HybridLogicalClock::new(1).unwrap();

        // Get local time
        let local = hlc.now().unwrap();

        // Receive a timestamp from the future
        let remote = HLCTimestamp::new(local.wall_ms() + 1000, 0, 2);
        let after_receive = hlc.receive(&remote).unwrap();

        // Should be after both local and remote
        assert!(
            after_receive > local,
            "after_receive {:?} should be > local {:?}",
            after_receive,
            local
        );
        assert!(
            after_receive > remote,
            "after_receive {:?} should be > remote {:?}",
            after_receive,
            remote
        );
    }

    #[test]
    fn test_hlc_receive_same_wall_time() {
        let hlc = HybridLogicalClock::new(1).unwrap();

        let local = hlc.now().unwrap();

        // Receive a timestamp with the same wall time but higher logical
        let remote = HLCTimestamp::new(local.wall_ms(), local.logical() + 10, 2);
        let after_receive = hlc.receive(&remote).unwrap();

        assert!(after_receive > remote);
    }

    #[test]
    fn test_hlc_estimated_wall_ms() {
        let hlc = HybridLogicalClock::new(1).unwrap();
        let estimated = hlc.estimated_wall_ms();
        // Should be a reasonable value (after year 2020)
        assert!(estimated > 1577836800000); // Jan 1, 2020
    }

    #[test]
    fn test_hlc_timestamp_serialization() {
        let ts = HLCTimestamp::new(12345678, 99, 42);
        let serialized = bitcode::serialize(&ts).unwrap();
        let deserialized: HLCTimestamp = bitcode::deserialize(&serialized).unwrap();
        assert_eq!(ts, deserialized);
    }

    #[test]
    fn test_hlc_timestamp_hash() {
        use std::collections::HashSet;

        let ts1 = HLCTimestamp::new(1000, 1, 1);
        let ts2 = HLCTimestamp::new(1000, 1, 1);
        let ts3 = HLCTimestamp::new(1000, 2, 1);

        let mut set = HashSet::new();
        set.insert(ts1);
        set.insert(ts2); // Should not add duplicate
        set.insert(ts3);

        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_hlc_rapid_calls() {
        let hlc = HybridLogicalClock::new(1).unwrap();
        let mut prev = hlc.now().unwrap();

        // Generate 1000 timestamps rapidly
        for _ in 0..1000 {
            let current = hlc.now().unwrap();
            assert!(current > prev, "Monotonicity violated");
            prev = current;
        }
    }

    #[test]
    fn test_hlc_timestamp_debug() {
        let ts = HLCTimestamp::new(1000, 5, 42);
        let debug = format!("{:?}", ts);
        assert!(debug.contains("HLCTimestamp"));
        assert!(debug.contains("1000"));
    }
}
