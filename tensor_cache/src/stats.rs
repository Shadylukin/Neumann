use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheLayer {
    Exact,
    Semantic,
    Embedding,
}

impl fmt::Display for CacheLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact => write!(f, "exact"),
            Self::Semantic => write!(f, "semantic"),
            Self::Embedding => write!(f, "embedding"),
        }
    }
}

/// Thread-safe cache statistics with atomic counters.
pub struct CacheStats {
    exact_hits: AtomicU64,
    exact_misses: AtomicU64,
    semantic_hits: AtomicU64,
    semantic_misses: AtomicU64,
    embedding_hits: AtomicU64,
    embedding_misses: AtomicU64,

    tokens_saved_in: AtomicU64,
    tokens_saved_out: AtomicU64,
    cost_saved_microdollars: AtomicU64,

    evictions: AtomicU64,
    expirations: AtomicU64,

    exact_size: AtomicUsize,
    semantic_size: AtomicUsize,
    embedding_size: AtomicUsize,

    start_time: Instant,
}

impl CacheStats {
    #[must_use]
    pub fn new() -> Self {
        Self {
            exact_hits: AtomicU64::new(0),
            exact_misses: AtomicU64::new(0),
            semantic_hits: AtomicU64::new(0),
            semantic_misses: AtomicU64::new(0),
            embedding_hits: AtomicU64::new(0),
            embedding_misses: AtomicU64::new(0),
            tokens_saved_in: AtomicU64::new(0),
            tokens_saved_out: AtomicU64::new(0),
            cost_saved_microdollars: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            expirations: AtomicU64::new(0),
            exact_size: AtomicUsize::new(0),
            semantic_size: AtomicUsize::new(0),
            embedding_size: AtomicUsize::new(0),
            start_time: Instant::now(),
        }
    }

    pub fn record_hit(&self, layer: CacheLayer) {
        match layer {
            CacheLayer::Exact => self.exact_hits.fetch_add(1, Ordering::Relaxed),
            CacheLayer::Semantic => self.semantic_hits.fetch_add(1, Ordering::Relaxed),
            CacheLayer::Embedding => self.embedding_hits.fetch_add(1, Ordering::Relaxed),
        };
    }

    pub fn record_miss(&self, layer: CacheLayer) {
        match layer {
            CacheLayer::Exact => self.exact_misses.fetch_add(1, Ordering::Relaxed),
            CacheLayer::Semantic => self.semantic_misses.fetch_add(1, Ordering::Relaxed),
            CacheLayer::Embedding => self.embedding_misses.fetch_add(1, Ordering::Relaxed),
        };
    }

    pub fn record_tokens_saved(&self, input: usize, output: usize) {
        self.tokens_saved_in
            .fetch_add(input as u64, Ordering::Relaxed);
        self.tokens_saved_out
            .fetch_add(output as u64, Ordering::Relaxed);
    }

    pub fn record_cost_saved(&self, microdollars: u64) {
        self.cost_saved_microdollars
            .fetch_add(microdollars, Ordering::Relaxed);
    }

    pub fn record_eviction(&self, count: usize) {
        self.evictions.fetch_add(count as u64, Ordering::Relaxed);
    }

    pub fn record_expiration(&self, count: usize) {
        self.expirations.fetch_add(count as u64, Ordering::Relaxed);
    }

    pub fn set_size(&self, layer: CacheLayer, size: usize) {
        match layer {
            CacheLayer::Exact => self.exact_size.store(size, Ordering::Relaxed),
            CacheLayer::Semantic => self.semantic_size.store(size, Ordering::Relaxed),
            CacheLayer::Embedding => self.embedding_size.store(size, Ordering::Relaxed),
        }
    }

    pub fn increment_size(&self, layer: CacheLayer) {
        match layer {
            CacheLayer::Exact => self.exact_size.fetch_add(1, Ordering::Relaxed),
            CacheLayer::Semantic => self.semantic_size.fetch_add(1, Ordering::Relaxed),
            CacheLayer::Embedding => self.embedding_size.fetch_add(1, Ordering::Relaxed),
        };
    }

    pub fn decrement_size(&self, layer: CacheLayer) {
        match layer {
            CacheLayer::Exact => {
                self.exact_size
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                        Some(v.saturating_sub(1))
                    })
                    .ok();
            },
            CacheLayer::Semantic => {
                self.semantic_size
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                        Some(v.saturating_sub(1))
                    })
                    .ok();
            },
            CacheLayer::Embedding => {
                self.embedding_size
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                        Some(v.saturating_sub(1))
                    })
                    .ok();
            },
        }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self, layer: CacheLayer) -> f64 {
        let (hits, misses) = match layer {
            CacheLayer::Exact => (
                self.exact_hits.load(Ordering::Relaxed),
                self.exact_misses.load(Ordering::Relaxed),
            ),
            CacheLayer::Semantic => (
                self.semantic_hits.load(Ordering::Relaxed),
                self.semantic_misses.load(Ordering::Relaxed),
            ),
            CacheLayer::Embedding => (
                self.embedding_hits.load(Ordering::Relaxed),
                self.embedding_misses.load(Ordering::Relaxed),
            ),
        };

        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.exact_size.load(Ordering::Relaxed)
            + self.semantic_size.load(Ordering::Relaxed)
            + self.embedding_size.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn size(&self, layer: CacheLayer) -> usize {
        match layer {
            CacheLayer::Exact => self.exact_size.load(Ordering::Relaxed),
            CacheLayer::Semantic => self.semantic_size.load(Ordering::Relaxed),
            CacheLayer::Embedding => self.embedding_size.load(Ordering::Relaxed),
        }
    }

    #[must_use]
    pub fn tokens_saved(&self) -> (u64, u64) {
        (
            self.tokens_saved_in.load(Ordering::Relaxed),
            self.tokens_saved_out.load(Ordering::Relaxed),
        )
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cost_saved_dollars(&self) -> f64 {
        self.cost_saved_microdollars.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    #[must_use]
    pub fn evictions(&self) -> u64 {
        self.evictions.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn expirations(&self) -> u64 {
        self.expirations.load(Ordering::Relaxed)
    }

    #[must_use]
    pub fn hits(&self, layer: CacheLayer) -> u64 {
        match layer {
            CacheLayer::Exact => self.exact_hits.load(Ordering::Relaxed),
            CacheLayer::Semantic => self.semantic_hits.load(Ordering::Relaxed),
            CacheLayer::Embedding => self.embedding_hits.load(Ordering::Relaxed),
        }
    }

    #[must_use]
    pub fn misses(&self, layer: CacheLayer) -> u64 {
        match layer {
            CacheLayer::Exact => self.exact_misses.load(Ordering::Relaxed),
            CacheLayer::Semantic => self.semantic_misses.load(Ordering::Relaxed),
            CacheLayer::Embedding => self.embedding_misses.load(Ordering::Relaxed),
        }
    }

    #[must_use]
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    #[must_use]
    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            exact_hits: self.exact_hits.load(Ordering::Relaxed),
            exact_misses: self.exact_misses.load(Ordering::Relaxed),
            semantic_hits: self.semantic_hits.load(Ordering::Relaxed),
            semantic_misses: self.semantic_misses.load(Ordering::Relaxed),
            embedding_hits: self.embedding_hits.load(Ordering::Relaxed),
            embedding_misses: self.embedding_misses.load(Ordering::Relaxed),
            tokens_saved_in: self.tokens_saved_in.load(Ordering::Relaxed),
            tokens_saved_out: self.tokens_saved_out.load(Ordering::Relaxed),
            cost_saved_dollars: self.cost_saved_dollars(),
            evictions: self.evictions.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
            exact_size: self.exact_size.load(Ordering::Relaxed),
            semantic_size: self.semantic_size.load(Ordering::Relaxed),
            embedding_size: self.embedding_size.load(Ordering::Relaxed),
            uptime_secs: self.uptime_secs(),
        }
    }
}

impl Default for CacheStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Point-in-time snapshot of cache statistics for reporting.
#[derive(Debug, Clone)]
pub struct StatsSnapshot {
    pub exact_hits: u64,
    pub exact_misses: u64,
    pub semantic_hits: u64,
    pub semantic_misses: u64,
    pub embedding_hits: u64,
    pub embedding_misses: u64,
    pub tokens_saved_in: u64,
    pub tokens_saved_out: u64,
    pub cost_saved_dollars: f64,
    pub evictions: u64,
    pub expirations: u64,
    pub exact_size: usize,
    pub semantic_size: usize,
    pub embedding_size: usize,
    pub uptime_secs: u64,
}

impl StatsSnapshot {
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self, layer: CacheLayer) -> f64 {
        let (hits, misses) = match layer {
            CacheLayer::Exact => (self.exact_hits, self.exact_misses),
            CacheLayer::Semantic => (self.semantic_hits, self.semantic_misses),
            CacheLayer::Embedding => (self.embedding_hits, self.embedding_misses),
        };
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    #[must_use]
    pub const fn total_entries(&self) -> usize {
        self.exact_size + self.semantic_size + self.embedding_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_hit_miss() {
        let stats = CacheStats::new();

        stats.record_hit(CacheLayer::Exact);
        stats.record_hit(CacheLayer::Exact);
        stats.record_miss(CacheLayer::Exact);

        assert_eq!(stats.hits(CacheLayer::Exact), 2);
        assert_eq!(stats.misses(CacheLayer::Exact), 1);
        assert!((stats.hit_rate(CacheLayer::Exact) - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_tokens_saved() {
        let stats = CacheStats::new();

        stats.record_tokens_saved(100, 50);
        stats.record_tokens_saved(200, 100);

        let (in_tokens, out_tokens) = stats.tokens_saved();
        assert_eq!(in_tokens, 300);
        assert_eq!(out_tokens, 150);
    }

    #[test]
    fn test_cost_saved() {
        let stats = CacheStats::new();

        stats.record_cost_saved(1_500_000);
        stats.record_cost_saved(500_000);

        assert!((stats.cost_saved_dollars() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_size_tracking() {
        let stats = CacheStats::new();

        stats.increment_size(CacheLayer::Exact);
        stats.increment_size(CacheLayer::Exact);
        stats.increment_size(CacheLayer::Semantic);

        assert_eq!(stats.size(CacheLayer::Exact), 2);
        assert_eq!(stats.size(CacheLayer::Semantic), 1);
        assert_eq!(stats.total_entries(), 3);

        stats.decrement_size(CacheLayer::Exact);
        assert_eq!(stats.size(CacheLayer::Exact), 1);
    }

    #[test]
    fn test_snapshot() {
        let stats = CacheStats::new();

        stats.record_hit(CacheLayer::Exact);
        stats.record_miss(CacheLayer::Semantic);
        stats.increment_size(CacheLayer::Embedding);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.exact_hits, 1);
        assert_eq!(snapshot.semantic_misses, 1);
        assert_eq!(snapshot.embedding_size, 1);
    }

    #[test]
    fn test_empty_hit_rate() {
        let stats = CacheStats::new();
        assert!((stats.hit_rate(CacheLayer::Exact) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_eviction_tracking() {
        let stats = CacheStats::new();

        stats.record_eviction(10);
        stats.record_expiration(5);

        assert_eq!(stats.evictions(), 10);
        assert_eq!(stats.expirations(), 5);
    }

    #[test]
    fn test_set_size() {
        let stats = CacheStats::new();

        stats.set_size(CacheLayer::Exact, 100);
        stats.set_size(CacheLayer::Semantic, 50);
        stats.set_size(CacheLayer::Embedding, 25);

        assert_eq!(stats.size(CacheLayer::Exact), 100);
        assert_eq!(stats.size(CacheLayer::Semantic), 50);
        assert_eq!(stats.size(CacheLayer::Embedding), 25);
        assert_eq!(stats.total_entries(), 175);
    }

    #[test]
    fn test_decrement_size_underflow() {
        let stats = CacheStats::new();

        stats.decrement_size(CacheLayer::Exact);
        stats.decrement_size(CacheLayer::Semantic);
        stats.decrement_size(CacheLayer::Embedding);

        assert_eq!(stats.size(CacheLayer::Exact), 0);
        assert_eq!(stats.size(CacheLayer::Semantic), 0);
        assert_eq!(stats.size(CacheLayer::Embedding), 0);
    }

    #[test]
    fn test_snapshot_hit_rate() {
        let stats = CacheStats::new();

        stats.record_hit(CacheLayer::Exact);
        stats.record_hit(CacheLayer::Exact);
        stats.record_miss(CacheLayer::Exact);
        stats.record_hit(CacheLayer::Semantic);
        stats.record_miss(CacheLayer::Semantic);
        stats.record_miss(CacheLayer::Semantic);

        let snapshot = stats.snapshot();

        let exact_rate = snapshot.hit_rate(CacheLayer::Exact);
        assert!((exact_rate - 0.666).abs() < 0.01);

        let semantic_rate = snapshot.hit_rate(CacheLayer::Semantic);
        assert!((semantic_rate - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_empty_hit_rate() {
        let stats = CacheStats::new();
        let snapshot = stats.snapshot();

        assert_eq!(snapshot.hit_rate(CacheLayer::Exact), 0.0);
        assert_eq!(snapshot.hit_rate(CacheLayer::Semantic), 0.0);
        assert_eq!(snapshot.hit_rate(CacheLayer::Embedding), 0.0);
    }

    #[test]
    fn test_snapshot_total_entries() {
        let stats = CacheStats::new();

        stats.set_size(CacheLayer::Exact, 10);
        stats.set_size(CacheLayer::Semantic, 20);
        stats.set_size(CacheLayer::Embedding, 30);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.total_entries(), 60);
    }

    #[test]
    fn test_default_trait() {
        let stats = CacheStats::default();
        assert_eq!(stats.evictions(), 0);
        assert_eq!(stats.expirations(), 0);
        assert_eq!(stats.total_entries(), 0);
    }

    #[test]
    fn test_hits_by_layer() {
        let stats = CacheStats::new();

        stats.record_hit(CacheLayer::Exact);
        stats.record_hit(CacheLayer::Semantic);
        stats.record_hit(CacheLayer::Semantic);
        stats.record_hit(CacheLayer::Embedding);
        stats.record_hit(CacheLayer::Embedding);
        stats.record_hit(CacheLayer::Embedding);

        assert_eq!(stats.hits(CacheLayer::Exact), 1);
        assert_eq!(stats.hits(CacheLayer::Semantic), 2);
        assert_eq!(stats.hits(CacheLayer::Embedding), 3);
    }

    #[test]
    fn test_misses_by_layer() {
        let stats = CacheStats::new();

        stats.record_miss(CacheLayer::Exact);
        stats.record_miss(CacheLayer::Semantic);
        stats.record_miss(CacheLayer::Semantic);
        stats.record_miss(CacheLayer::Embedding);
        stats.record_miss(CacheLayer::Embedding);
        stats.record_miss(CacheLayer::Embedding);

        assert_eq!(stats.misses(CacheLayer::Exact), 1);
        assert_eq!(stats.misses(CacheLayer::Semantic), 2);
        assert_eq!(stats.misses(CacheLayer::Embedding), 3);
    }

    #[test]
    fn test_uptime_secs() {
        let stats = CacheStats::new();
        assert!(stats.uptime_secs() < 2);
    }

    #[test]
    fn test_snapshot_uptime() {
        let stats = CacheStats::new();
        let snapshot = stats.snapshot();
        assert!(snapshot.uptime_secs < 2);
    }

    #[test]
    fn test_snapshot_clone() {
        let stats = CacheStats::new();
        stats.record_hit(CacheLayer::Exact);
        stats.record_cost_saved(1000);

        let snapshot1 = stats.snapshot();
        let snapshot2 = snapshot1.clone();

        assert_eq!(snapshot1.exact_hits, snapshot2.exact_hits);
        assert_eq!(snapshot1.cost_saved_dollars, snapshot2.cost_saved_dollars);
    }

    #[test]
    fn test_cache_layer_equality() {
        assert_eq!(CacheLayer::Exact, CacheLayer::Exact);
        assert_ne!(CacheLayer::Exact, CacheLayer::Semantic);
        assert_ne!(CacheLayer::Semantic, CacheLayer::Embedding);
    }

    #[test]
    fn test_cache_layer_copy() {
        let layer = CacheLayer::Exact;
        let copied = layer;
        assert_eq!(layer, copied);
    }

    #[test]
    fn test_hit_rate_by_layer() {
        let stats = CacheStats::new();

        stats.record_hit(CacheLayer::Semantic);
        stats.record_miss(CacheLayer::Semantic);
        stats.record_miss(CacheLayer::Semantic);
        stats.record_miss(CacheLayer::Semantic);

        stats.record_hit(CacheLayer::Embedding);

        assert!((stats.hit_rate(CacheLayer::Semantic) - 0.25).abs() < 0.01);
        assert!((stats.hit_rate(CacheLayer::Embedding) - 1.0).abs() < 0.01);
    }
}
