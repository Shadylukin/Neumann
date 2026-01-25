//! Centralized metrics infrastructure for tensor_chain observability.
//!
//! Provides thread-safe timing statistics using atomics, following the same
//! patterns as existing stats structures (FastPathStats, DistributedTxStats).

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-safe timing statistics using atomics.
///
/// Records count, total duration, min, and max for a category of operations.
/// All operations use `Ordering::Relaxed` for performance.
#[derive(Debug)]
pub struct TimingStats {
    count: AtomicU64,
    total_us: AtomicU64,
    min_us: AtomicU64,
    max_us: AtomicU64,
}

impl Default for TimingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingStats {
    pub fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_us: AtomicU64::new(0),
            min_us: AtomicU64::new(u64::MAX),
            max_us: AtomicU64::new(0),
        }
    }

    /// Record a duration in microseconds.
    pub fn record(&self, duration_us: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(duration_us, Ordering::Relaxed);
        self.min_us.fetch_min(duration_us, Ordering::Relaxed);
        self.max_us.fetch_max(duration_us, Ordering::Relaxed);
    }

    /// Record a duration from a `std::time::Duration`.
    pub fn record_duration(&self, duration: std::time::Duration) {
        self.record(duration.as_micros() as u64);
    }

    /// Get the count of recorded operations.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the total duration in microseconds.
    pub fn total_us(&self) -> u64 {
        self.total_us.load(Ordering::Relaxed)
    }

    /// Get the average duration in microseconds.
    pub fn avg_us(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            self.total_us.load(Ordering::Relaxed) as f64 / count as f64
        }
    }

    /// Get the minimum duration in microseconds.
    pub fn min_us(&self) -> u64 {
        let min = self.min_us.load(Ordering::Relaxed);
        if min == u64::MAX {
            0
        } else {
            min
        }
    }

    /// Get the maximum duration in microseconds.
    pub fn max_us(&self) -> u64 {
        self.max_us.load(Ordering::Relaxed)
    }

    /// Take a point-in-time snapshot of the statistics.
    pub fn snapshot(&self) -> TimingSnapshot {
        let count = self.count.load(Ordering::Relaxed);
        let total = self.total_us.load(Ordering::Relaxed);
        let min = self.min_us.load(Ordering::Relaxed);
        let max = self.max_us.load(Ordering::Relaxed);
        TimingSnapshot {
            count,
            total_us: total,
            min_us: if min == u64::MAX { 0 } else { min },
            max_us: max,
            avg_us: if count == 0 {
                0.0
            } else {
                total as f64 / count as f64
            },
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_us.store(0, Ordering::Relaxed);
        self.min_us.store(u64::MAX, Ordering::Relaxed);
        self.max_us.store(0, Ordering::Relaxed);
    }
}

/// Point-in-time snapshot of timing statistics.
///
/// Serializable for export to monitoring systems.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TimingSnapshot {
    pub count: u64,
    pub total_us: u64,
    pub min_us: u64,
    pub max_us: u64,
    pub avg_us: f64,
}

impl TimingSnapshot {
    /// Check if any operations were recorded.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the average duration in milliseconds.
    pub fn avg_ms(&self) -> f64 {
        self.avg_us / 1000.0
    }

    /// Get the minimum duration in milliseconds.
    pub fn min_ms(&self) -> f64 {
        self.min_us as f64 / 1000.0
    }

    /// Get the maximum duration in milliseconds.
    pub fn max_ms(&self) -> f64 {
        self.max_us as f64 / 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_timing_stats_new() {
        let stats = TimingStats::new();
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.total_us(), 0);
        assert_eq!(stats.min_us(), 0);
        assert_eq!(stats.max_us(), 0);
        assert_eq!(stats.avg_us(), 0.0);
    }

    #[test]
    fn test_timing_stats_record_single() {
        let stats = TimingStats::new();
        stats.record(100);

        assert_eq!(stats.count(), 1);
        assert_eq!(stats.total_us(), 100);
        assert_eq!(stats.min_us(), 100);
        assert_eq!(stats.max_us(), 100);
        assert_eq!(stats.avg_us(), 100.0);
    }

    #[test]
    fn test_timing_stats_record_multiple() {
        let stats = TimingStats::new();
        stats.record(100);
        stats.record(200);
        stats.record(300);

        assert_eq!(stats.count(), 3);
        assert_eq!(stats.total_us(), 600);
        assert_eq!(stats.min_us(), 100);
        assert_eq!(stats.max_us(), 300);
        assert_eq!(stats.avg_us(), 200.0);
    }

    #[test]
    fn test_timing_stats_record_duration() {
        let stats = TimingStats::new();
        stats.record_duration(std::time::Duration::from_micros(500));

        assert_eq!(stats.count(), 1);
        assert_eq!(stats.total_us(), 500);
    }

    #[test]
    fn test_timing_stats_min_max_extremes() {
        let stats = TimingStats::new();
        stats.record(1);
        stats.record(1_000_000);

        assert_eq!(stats.min_us(), 1);
        assert_eq!(stats.max_us(), 1_000_000);
    }

    #[test]
    fn test_timing_stats_snapshot() {
        let stats = TimingStats::new();
        stats.record(100);
        stats.record(200);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.count, 2);
        assert_eq!(snapshot.total_us, 300);
        assert_eq!(snapshot.min_us, 100);
        assert_eq!(snapshot.max_us, 200);
        assert_eq!(snapshot.avg_us, 150.0);
    }

    #[test]
    fn test_timing_stats_snapshot_empty() {
        let stats = TimingStats::new();
        let snapshot = stats.snapshot();

        assert_eq!(snapshot.count, 0);
        assert_eq!(snapshot.total_us, 0);
        assert_eq!(snapshot.min_us, 0);
        assert_eq!(snapshot.max_us, 0);
        assert_eq!(snapshot.avg_us, 0.0);
        assert!(snapshot.is_empty());
    }

    #[test]
    fn test_timing_stats_reset() {
        let stats = TimingStats::new();
        stats.record(100);
        stats.record(200);

        stats.reset();

        assert_eq!(stats.count(), 0);
        assert_eq!(stats.total_us(), 0);
        assert_eq!(stats.min_us(), 0);
        assert_eq!(stats.max_us(), 0);
    }

    #[test]
    fn test_timing_stats_concurrent_access() {
        let stats = std::sync::Arc::new(TimingStats::new());
        let mut handles = vec![];

        for i in 0..10 {
            let stats_clone = stats.clone();
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    stats_clone.record((i * 100 + j) as u64);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(stats.count(), 1000);
    }

    #[test]
    fn test_timing_snapshot_conversions() {
        let snapshot = TimingSnapshot {
            count: 10,
            total_us: 5000,
            min_us: 100,
            max_us: 1000,
            avg_us: 500.0,
        };

        assert_eq!(snapshot.avg_ms(), 0.5);
        assert_eq!(snapshot.min_ms(), 0.1);
        assert_eq!(snapshot.max_ms(), 1.0);
        assert!(!snapshot.is_empty());
    }

    #[test]
    fn test_timing_snapshot_serialization() {
        let snapshot = TimingSnapshot {
            count: 5,
            total_us: 1000,
            min_us: 100,
            max_us: 300,
            avg_us: 200.0,
        };

        let bytes = bitcode::serialize(&snapshot).unwrap();
        let restored: TimingSnapshot = bitcode::deserialize(&bytes).unwrap();

        assert_eq!(snapshot, restored);
    }

    #[test]
    fn test_timing_snapshot_default() {
        let snapshot = TimingSnapshot::default();
        assert_eq!(snapshot.count, 0);
        assert_eq!(snapshot.total_us, 0);
        assert_eq!(snapshot.min_us, 0);
        assert_eq!(snapshot.max_us, 0);
        assert_eq!(snapshot.avg_us, 0.0);
        assert!(snapshot.is_empty());
    }
}
