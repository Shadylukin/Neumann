//! Memory instrumentation for tracking shard and node access patterns.
//!
//! Provides low-overhead tracking to identify hot vs cold data regions,
//! enabling intelligent memory tiering decisions.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Default shard count matching DashMap's internal sharding.
pub const DEFAULT_SHARD_COUNT: usize = 16;

/// Per-shard access statistics.
pub struct ShardStats {
    reads: AtomicU64,
    writes: AtomicU64,
    last_access_ms: AtomicU64,
}

impl ShardStats {
    fn new() -> Self {
        Self {
            reads: AtomicU64::new(0),
            writes: AtomicU64::new(0),
            last_access_ms: AtomicU64::new(0),
        }
    }

    #[inline]
    fn record_read(&self) {
        self.reads.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    fn record_write(&self, epoch_ms: u64) {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.last_access_ms.store(epoch_ms, Ordering::Relaxed);
    }

    fn snapshot(&self, shard_id: usize) -> ShardStatsSnapshot {
        ShardStatsSnapshot {
            shard_id,
            reads: self.reads.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
            last_access_ms: self.last_access_ms.load(Ordering::Relaxed),
        }
    }
}

/// Tracks access patterns across TensorStore shards.
pub struct ShardAccessTracker {
    shards: Box<[ShardStats]>,
    shard_count: usize,
    start_time: Instant,
    sample_rate: u32,
    sample_counter: AtomicU64,
}

impl ShardAccessTracker {
    /// Create a new tracker with specified shard count and sampling rate.
    /// Sample rate of 1 means track every access; 100 means track 1 in 100.
    pub fn new(shard_count: usize, sample_rate: u32) -> Self {
        let shards: Vec<ShardStats> = (0..shard_count).map(|_| ShardStats::new()).collect();
        Self {
            shards: shards.into_boxed_slice(),
            shard_count,
            start_time: Instant::now(),
            sample_rate: sample_rate.max(1),
            sample_counter: AtomicU64::new(0),
        }
    }

    /// Create with default settings (16 shards, 1:100 sampling).
    pub fn with_defaults() -> Self {
        Self::new(DEFAULT_SHARD_COUNT, 100)
    }

    /// Record a read access for the given shard (sampled).
    /// Shard IDs are mapped to tracker shards using modulo.
    #[inline]
    pub fn record_read(&self, shard_id: usize) {
        if self.should_sample() {
            let idx = shard_id % self.shard_count;
            self.shards[idx].record_read();
        }
    }

    /// Record a write access for the given shard (sampled).
    /// Shard IDs are mapped to tracker shards using modulo.
    #[inline]
    pub fn record_write(&self, shard_id: usize) {
        if self.should_sample() {
            let idx = shard_id % self.shard_count;
            let epoch_ms = self.start_time.elapsed().as_millis() as u64;
            self.shards[idx].record_write(epoch_ms);
        }
    }

    #[inline]
    fn should_sample(&self) -> bool {
        if self.sample_rate == 1 {
            return true;
        }
        let count = self.sample_counter.fetch_add(1, Ordering::Relaxed);
        count.is_multiple_of(self.sample_rate as u64)
    }

    /// Get shards that haven't been accessed within the threshold (in ms).
    pub fn cold_shards(&self, threshold_ms: u64) -> Vec<usize> {
        let now = self.start_time.elapsed().as_millis() as u64;
        self.shards
            .iter()
            .enumerate()
            .filter(|(_, stats)| {
                let last = stats.last_access_ms.load(Ordering::Relaxed);
                last == 0 || now.saturating_sub(last) > threshold_ms
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Get shards sorted by access count (hottest first).
    pub fn hot_shards(&self, limit: usize) -> Vec<(usize, u64)> {
        let mut shard_counts: Vec<(usize, u64)> = self
            .shards
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let total = s.reads.load(Ordering::Relaxed) + s.writes.load(Ordering::Relaxed);
                (i, total)
            })
            .collect();
        shard_counts.sort_by(|a, b| b.1.cmp(&a.1));
        shard_counts.truncate(limit);
        shard_counts
    }

    /// Create a point-in-time snapshot.
    pub fn snapshot(&self) -> ShardAccessSnapshot {
        let shard_stats: Vec<ShardStatsSnapshot> = self
            .shards
            .iter()
            .enumerate()
            .map(|(i, s)| s.snapshot(i))
            .collect();

        let hot_shards = self.hot_shards(5);

        ShardAccessSnapshot {
            shard_stats,
            hot_shards,
            timestamp_ms: self.start_time.elapsed().as_millis() as u64,
            sample_rate: self.sample_rate,
        }
    }

    /// Total read count across all shards (scaled by sample rate).
    pub fn total_reads(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.reads.load(Ordering::Relaxed))
            .sum::<u64>()
            * self.sample_rate as u64
    }

    /// Total write count across all shards (scaled by sample rate).
    pub fn total_writes(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.writes.load(Ordering::Relaxed))
            .sum::<u64>()
            * self.sample_rate as u64
    }

    /// Get the shard count.
    pub fn shard_count(&self) -> usize {
        self.shard_count
    }
}

impl Default for ShardAccessTracker {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// HNSW-specific access statistics.
pub struct HNSWAccessStats {
    entry_point_accesses: AtomicU64,
    layer0_traversals: AtomicU64,
    upper_layer_traversals: AtomicU64,
    total_searches: AtomicU64,
    distance_calculations: AtomicU64,
    start_time: Instant,
}

impl HNSWAccessStats {
    pub fn new() -> Self {
        Self {
            entry_point_accesses: AtomicU64::new(0),
            layer0_traversals: AtomicU64::new(0),
            upper_layer_traversals: AtomicU64::new(0),
            total_searches: AtomicU64::new(0),
            distance_calculations: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    #[inline]
    pub fn record_search(&self) {
        self.total_searches.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn record_entry_point_access(&self) {
        self.entry_point_accesses.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn record_layer_traversal(&self, layer: usize) {
        if layer == 0 {
            self.layer0_traversals.fetch_add(1, Ordering::Relaxed);
        } else {
            self.upper_layer_traversals.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn record_distance_calculations(&self, count: u64) {
        self.distance_calculations
            .fetch_add(count, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> HNSWStatsSnapshot {
        let total_searches = self.total_searches.load(Ordering::Relaxed);
        let distance_calcs = self.distance_calculations.load(Ordering::Relaxed);

        HNSWStatsSnapshot {
            entry_point_accesses: self.entry_point_accesses.load(Ordering::Relaxed),
            layer0_traversals: self.layer0_traversals.load(Ordering::Relaxed),
            upper_layer_traversals: self.upper_layer_traversals.load(Ordering::Relaxed),
            total_searches,
            distance_calculations: distance_calcs,
            avg_distances_per_search: if total_searches > 0 {
                distance_calcs as f64 / total_searches as f64
            } else {
                0.0
            },
            uptime_ms: self.start_time.elapsed().as_millis() as u64,
        }
    }

    pub fn total_searches(&self) -> u64 {
        self.total_searches.load(Ordering::Relaxed)
    }

    pub fn distance_calculations(&self) -> u64 {
        self.distance_calculations.load(Ordering::Relaxed)
    }
}

impl Default for HNSWAccessStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-time snapshot of per-shard statistics.
#[derive(Debug, Clone)]
pub struct ShardStatsSnapshot {
    pub shard_id: usize,
    pub reads: u64,
    pub writes: u64,
    pub last_access_ms: u64,
}

impl ShardStatsSnapshot {
    pub fn total_accesses(&self) -> u64 {
        self.reads + self.writes
    }
}

/// Point-in-time snapshot of shard access patterns.
#[derive(Debug, Clone)]
pub struct ShardAccessSnapshot {
    pub shard_stats: Vec<ShardStatsSnapshot>,
    pub hot_shards: Vec<(usize, u64)>,
    pub timestamp_ms: u64,
    pub sample_rate: u32,
}

impl ShardAccessSnapshot {
    pub fn total_reads(&self) -> u64 {
        self.shard_stats.iter().map(|s| s.reads).sum::<u64>() * self.sample_rate as u64
    }

    pub fn total_writes(&self) -> u64 {
        self.shard_stats.iter().map(|s| s.writes).sum::<u64>() * self.sample_rate as u64
    }

    /// Access distribution as percentages per shard.
    pub fn distribution(&self) -> Vec<(usize, f64)> {
        let total: u64 = self.shard_stats.iter().map(|s| s.total_accesses()).sum();
        if total == 0 {
            return vec![];
        }
        self.shard_stats
            .iter()
            .map(|s| (s.shard_id, s.total_accesses() as f64 / total as f64 * 100.0))
            .collect()
    }
}

/// Point-in-time snapshot of HNSW access patterns.
#[derive(Debug, Clone)]
pub struct HNSWStatsSnapshot {
    pub entry_point_accesses: u64,
    pub layer0_traversals: u64,
    pub upper_layer_traversals: u64,
    pub total_searches: u64,
    pub distance_calculations: u64,
    pub avg_distances_per_search: f64,
    pub uptime_ms: u64,
}

impl HNSWStatsSnapshot {
    /// Layer 0 accounts for most work; this ratio indicates traversal efficiency.
    pub fn layer0_ratio(&self) -> f64 {
        let total = self.layer0_traversals + self.upper_layer_traversals;
        if total == 0 {
            0.0
        } else {
            self.layer0_traversals as f64 / total as f64
        }
    }

    /// Searches per second.
    pub fn searches_per_second(&self) -> f64 {
        if self.uptime_ms == 0 {
            0.0
        } else {
            self.total_searches as f64 / (self.uptime_ms as f64 / 1000.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_shard_tracker_new() {
        let tracker = ShardAccessTracker::new(8, 1);
        assert_eq!(tracker.shard_count(), 8);
    }

    #[test]
    fn test_shard_tracker_defaults() {
        let tracker = ShardAccessTracker::with_defaults();
        assert_eq!(tracker.shard_count(), DEFAULT_SHARD_COUNT);
    }

    #[test]
    fn test_record_read_write() {
        let tracker = ShardAccessTracker::new(4, 1); // No sampling

        tracker.record_read(0);
        tracker.record_read(0);
        tracker.record_write(1);

        let snapshot = tracker.snapshot();
        assert_eq!(snapshot.shard_stats[0].reads, 2);
        assert_eq!(snapshot.shard_stats[1].writes, 1);
    }

    #[test]
    fn test_sampling() {
        let tracker = ShardAccessTracker::new(4, 10); // 1 in 10

        for _ in 0..100 {
            tracker.record_read(0);
        }

        // Should have recorded ~10 (1 in 10)
        let snapshot = tracker.snapshot();
        assert!(snapshot.shard_stats[0].reads >= 8 && snapshot.shard_stats[0].reads <= 12);
    }

    #[test]
    fn test_cold_shards() {
        let tracker = ShardAccessTracker::new(4, 1);

        // Access shards 0 and 2
        tracker.record_write(0);
        tracker.record_write(2);

        // Shards 1 and 3 should be cold (never accessed)
        let cold = tracker.cold_shards(1000);
        assert!(cold.contains(&1));
        assert!(cold.contains(&3));
    }

    #[test]
    fn test_hot_shards() {
        let tracker = ShardAccessTracker::new(4, 1);

        // Make shard 2 the hottest
        for _ in 0..100 {
            tracker.record_read(2);
        }
        for _ in 0..50 {
            tracker.record_read(0);
        }

        let hot = tracker.hot_shards(2);
        assert_eq!(hot[0].0, 2); // Shard 2 is hottest
        assert_eq!(hot[1].0, 0); // Shard 0 is second
    }

    #[test]
    fn test_total_reads_writes() {
        let tracker = ShardAccessTracker::new(4, 1);

        tracker.record_read(0);
        tracker.record_read(1);
        tracker.record_write(2);

        assert_eq!(tracker.total_reads(), 2);
        assert_eq!(tracker.total_writes(), 1);
    }

    #[test]
    fn test_out_of_bounds_shard_uses_modulo() {
        let tracker = ShardAccessTracker::new(4, 1);

        // Shard 100 maps to shard 0 (100 % 4 = 0)
        tracker.record_read(100);
        tracker.record_write(100);

        // Access should be recorded in shard 0
        assert_eq!(tracker.total_reads(), 1);
        assert_eq!(tracker.total_writes(), 1);
    }

    #[test]
    fn test_hnsw_stats_new() {
        let stats = HNSWAccessStats::new();
        assert_eq!(stats.total_searches(), 0);
    }

    #[test]
    fn test_hnsw_record_search() {
        let stats = HNSWAccessStats::new();

        stats.record_search();
        stats.record_search();

        assert_eq!(stats.total_searches(), 2);
    }

    #[test]
    fn test_hnsw_layer_traversals() {
        let stats = HNSWAccessStats::new();

        stats.record_layer_traversal(0);
        stats.record_layer_traversal(0);
        stats.record_layer_traversal(1);
        stats.record_layer_traversal(2);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.layer0_traversals, 2);
        assert_eq!(snapshot.upper_layer_traversals, 2);
    }

    #[test]
    fn test_hnsw_distance_calculations() {
        let stats = HNSWAccessStats::new();

        stats.record_search();
        stats.record_distance_calculations(50);
        stats.record_search();
        stats.record_distance_calculations(30);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.distance_calculations, 80);
        assert!((snapshot.avg_distances_per_search - 40.0).abs() < 0.01);
    }

    #[test]
    fn test_hnsw_layer0_ratio() {
        let stats = HNSWAccessStats::new();

        stats.record_layer_traversal(0);
        stats.record_layer_traversal(0);
        stats.record_layer_traversal(0);
        stats.record_layer_traversal(1);

        let snapshot = stats.snapshot();
        assert!((snapshot.layer0_ratio() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_distribution() {
        let tracker = ShardAccessTracker::new(4, 1);

        tracker.record_read(0);
        tracker.record_read(0);
        tracker.record_read(1);
        tracker.record_read(1);

        let snapshot = tracker.snapshot();
        let dist = snapshot.distribution();

        // Shards 0 and 1 should each have 50%
        let shard0 = dist.iter().find(|(id, _)| *id == 0).unwrap().1;
        let shard1 = dist.iter().find(|(id, _)| *id == 1).unwrap().1;
        assert!((shard0 - 50.0).abs() < 0.01);
        assert!((shard1 - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_distribution() {
        let tracker = ShardAccessTracker::new(4, 1);
        let snapshot = tracker.snapshot();
        assert!(snapshot.distribution().is_empty());
    }

    #[test]
    fn test_shard_stats_snapshot_total() {
        let snapshot = ShardStatsSnapshot {
            shard_id: 0,
            reads: 10,
            writes: 5,
            last_access_ms: 100,
        };
        assert_eq!(snapshot.total_accesses(), 15);
    }

    #[test]
    fn test_concurrent_access() {
        let tracker = std::sync::Arc::new(ShardAccessTracker::new(4, 1));

        let handles: Vec<_> = (0..4)
            .map(|shard_id| {
                let tracker = tracker.clone();
                thread::spawn(move || {
                    for _ in 0..1000 {
                        tracker.record_read(shard_id);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tracker.total_reads(), 4000);
    }

    #[test]
    fn test_default_trait() {
        let tracker = ShardAccessTracker::default();
        assert_eq!(tracker.shard_count(), DEFAULT_SHARD_COUNT);

        let stats = HNSWAccessStats::default();
        assert_eq!(stats.total_searches(), 0);
    }

    #[test]
    fn test_hnsw_entry_point() {
        let stats = HNSWAccessStats::new();

        stats.record_entry_point_access();
        stats.record_entry_point_access();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.entry_point_accesses, 2);
    }

    #[test]
    fn test_searches_per_second() {
        let stats = HNSWAccessStats::new();

        for _ in 0..100 {
            stats.record_search();
        }

        // Sleep briefly to get non-zero uptime
        thread::sleep(std::time::Duration::from_millis(10));

        let snapshot = stats.snapshot();
        assert!(snapshot.searches_per_second() > 0.0);
    }

    #[test]
    fn test_snapshot_total_reads_with_sampling() {
        let tracker = ShardAccessTracker::new(4, 10); // 1:10 sampling

        // Record 100 reads, expect ~10 tracked
        for _ in 0..100 {
            tracker.record_read(0);
        }

        let snapshot = tracker.snapshot();
        // total_reads scales back up by sample_rate
        let total = snapshot.total_reads();
        assert!(total >= 80 && total <= 120);
    }
}
