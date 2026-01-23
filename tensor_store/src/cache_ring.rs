//! Fixed-size cache ring with configurable eviction strategies.
//!
//! `CacheRing` provides a fixed-capacity cache that never resizes, eliminating
//! resize stalls. Eviction is handled via LRU, LFU, or Hybrid scoring.
//!
//! # Design Philosophy
//!
//! - Fixed size: capacity is set at creation and never changes
//! - No resize stalls: all operations are O(1) amortized
//! - Pluggable eviction: LRU, LFU, `CostBased`, or Hybrid strategies
//! - Thread-safe: uses `parking_lot` for low-contention access

use std::{
    collections::BTreeMap,
    hash::{Hash, Hasher},
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use fxhash::FxHasher;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Eviction strategy for cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Least Recently Used - evicts entries that haven't been accessed recently.
    LRU,
    /// Least Frequently Used - evicts entries with lowest access count.
    LFU,
    /// Cost-based - evicts entries with lowest cost savings per byte.
    CostBased,
    /// Hybrid strategy combining LRU, LFU, and cost factors.
    Hybrid {
        lru_weight: u8,
        lfu_weight: u8,
        cost_weight: u8,
    },
}

impl Default for EvictionStrategy {
    fn default() -> Self {
        Self::Hybrid {
            lru_weight: 40,
            lfu_weight: 30,
            cost_weight: 30,
        }
    }
}

/// Cache entry metadata.
#[derive(Debug, Clone)]
struct CacheEntry<V> {
    key_hash: u64,
    key: String,
    value: V,
    last_access: Instant,
    access_count: u64,
    cost: f64,
    size_bytes: usize,
}

/// Eviction scorer calculates priority for eviction.
pub struct EvictionScorer {
    strategy: EvictionStrategy,
}

impl EvictionScorer {
    #[must_use]
    pub const fn new(strategy: EvictionStrategy) -> Self {
        Self { strategy }
    }

    /// Calculate eviction score. Lower scores are evicted first.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn score(
        &self,
        last_access_secs: f64,
        access_count: u64,
        cost: f64,
        size_bytes: usize,
    ) -> f64 {
        match self.strategy {
            EvictionStrategy::LRU => -last_access_secs,
            EvictionStrategy::LFU => access_count as f64,
            EvictionStrategy::CostBased => {
                if size_bytes == 0 {
                    0.0
                } else {
                    cost / size_bytes as f64
                }
            },
            EvictionStrategy::Hybrid {
                lru_weight,
                lfu_weight,
                cost_weight,
            } => {
                let total = f64::from(lru_weight) + f64::from(lfu_weight) + f64::from(cost_weight);
                let recency_w = f64::from(lru_weight) / total;
                let frequency_w = f64::from(lfu_weight) / total;
                let cost_w = f64::from(cost_weight) / total;

                let age_minutes = last_access_secs / 60.0;
                let recency_score = 1.0 / (1.0 + age_minutes);
                let frequency_score = (1.0 + access_count as f64).log2();
                let cost_score = cost;

                cost_score.mul_add(
                    cost_w,
                    recency_score.mul_add(recency_w, frequency_score * frequency_w),
                )
            },
        }
    }
}

/// Fixed-size cache with eviction.
///
/// # Thread Safety
///
/// Uses `parking_lot::RwLock` for concurrent access.
pub struct CacheRing<V> {
    slots: RwLock<Vec<Option<CacheEntry<V>>>>,
    index: RwLock<BTreeMap<u64, usize>>,
    capacity: usize,
    count: AtomicU64,
    strategy: EvictionStrategy,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<V: Clone> CacheRing<V> {
    /// Create a new cache ring with the specified capacity and eviction strategy.
    #[must_use]
    pub fn new(capacity: usize, strategy: EvictionStrategy) -> Self {
        let slots: Vec<Option<CacheEntry<V>>> = (0..capacity).map(|_| None).collect();

        Self {
            slots: RwLock::new(slots),
            index: RwLock::new(BTreeMap::new()),
            capacity,
            count: AtomicU64::new(0),
            strategy,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Create with default LRU strategy.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(capacity, EvictionStrategy::LRU)
    }

    fn hash_key(key: &str) -> u64 {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Get a value from the cache.
    pub fn get(&self, key: &str) -> Option<V> {
        let key_hash = Self::hash_key(key);

        let index = self.index.read();
        if let Some(&slot_idx) = index.get(&key_hash) {
            drop(index);

            let mut slots = self.slots.write();
            if let Some(ref mut entry) = slots[slot_idx] {
                if entry.key == key {
                    entry.last_access = Instant::now();
                    entry.access_count += 1;
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return Some(entry.value.clone());
                }
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Put a value into the cache.
    pub fn put(&self, key: &str, value: V, cost: f64, size_bytes: usize) {
        let key_hash = Self::hash_key(key);

        // Check if key already exists and update in place
        {
            let existing_slot = self.index.read().get(&key_hash).copied();
            if let Some(slot_idx) = existing_slot {
                let mut slots = self.slots.write();
                if let Some(ref mut entry) = slots[slot_idx] {
                    if entry.key == key {
                        entry.value = value;
                        entry.last_access = Instant::now();
                        entry.access_count += 1;
                        entry.cost = cost;
                        entry.size_bytes = size_bytes;
                        return;
                    }
                }
            }
        }

        // Find a slot: either empty or evict lowest-scored
        let slot_idx = self.find_slot_for_insert();

        let mut slots = self.slots.write();
        let mut index = self.index.write();

        // Remove old entry from index if slot was occupied
        if let Some(ref old_entry) = slots[slot_idx] {
            index.remove(&old_entry.key_hash);
        } else {
            self.count.fetch_add(1, Ordering::Relaxed);
        }

        // Insert new entry
        slots[slot_idx] = Some(CacheEntry {
            key_hash,
            key: key.to_string(),
            value,
            last_access: Instant::now(),
            access_count: 1,
            cost,
            size_bytes,
        });

        index.insert(key_hash, slot_idx);
        drop(index);
        drop(slots);
    }

    fn find_slot_for_insert(&self) -> usize {
        let slots = self.slots.read();
        let scorer = EvictionScorer::new(self.strategy);
        let now = Instant::now();

        let mut best_slot = 0;
        let mut best_score = f64::MAX;

        for (idx, slot) in slots.iter().enumerate() {
            match slot {
                None => {
                    drop(slots);
                    return idx; // Empty slot, use immediately
                },
                Some(entry) => {
                    let age_secs = now.duration_since(entry.last_access).as_secs_f64();
                    let score =
                        scorer.score(age_secs, entry.access_count, entry.cost, entry.size_bytes);
                    if score < best_score {
                        best_score = score;
                        best_slot = idx;
                    }
                },
            }
        }
        drop(slots);

        best_slot
    }

    /// Delete a value from the cache.
    pub fn delete(&self, key: &str) -> bool {
        let key_hash = Self::hash_key(key);

        let Some(slot_idx) = self.index.write().remove(&key_hash) else {
            return false;
        };

        let mut slots = self.slots.write();
        if let Some(ref entry) = slots[slot_idx] {
            if entry.key == key {
                slots[slot_idx] = None;
                drop(slots);
                self.count.fetch_sub(1, Ordering::Relaxed);
                return true;
            }
        }
        drop(slots);

        false
    }

    /// Check if a key exists in the cache.
    pub fn contains(&self, key: &str) -> bool {
        let key_hash = Self::hash_key(key);

        let Some(&slot_idx) = self.index.read().get(&key_hash) else {
            return false;
        };

        let slots = self.slots.read();
        let result = slots[slot_idx]
            .as_ref()
            .is_some_and(|entry| entry.key == key);
        drop(slots);

        result
    }

    /// Get the number of entries in the cache.
    #[allow(clippy::cast_possible_truncation)] // Count bounded by capacity (usize)
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed) as usize
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the cache capacity.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear all entries from the cache.
    pub fn clear(&self) {
        let mut slots = self.slots.write();
        let mut index = self.index.write();

        for slot in slots.iter_mut() {
            *slot = None;
        }
        index.clear();
        drop(slots);
        drop(index);
        self.count.store(0, Ordering::Relaxed);
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Evict the lowest-scored entries from the cache.
    pub fn evict(&self, count: usize) -> usize {
        if count == 0 {
            return 0;
        }

        let scorer = EvictionScorer::new(self.strategy);
        let now = Instant::now();

        // Collect candidates with scores
        let mut candidates: Vec<(usize, f64)> = {
            let slots = self.slots.read();
            slots
                .iter()
                .enumerate()
                .filter_map(|(idx, slot)| {
                    slot.as_ref().map(|entry| {
                        let age_secs = now.duration_since(entry.last_access).as_secs_f64();
                        let score = scorer.score(
                            age_secs,
                            entry.access_count,
                            entry.cost,
                            entry.size_bytes,
                        );
                        (idx, score)
                    })
                })
                .collect()
        };

        // Sort by score (lowest first for eviction)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(count);

        // Evict the candidates
        let mut evicted = 0;
        let mut slots = self.slots.write();
        let mut index = self.index.write();

        for (slot_idx, _) in candidates {
            if let Some(ref entry) = slots[slot_idx] {
                index.remove(&entry.key_hash);
                slots[slot_idx] = None;
                self.count.fetch_sub(1, Ordering::Relaxed);
                evicted += 1;
            }
        }
        drop(slots);
        drop(index);

        evicted
    }

    /// Scan keys matching a prefix.
    pub fn scan_prefix(&self, prefix: &str) -> Vec<String> {
        let slots = self.slots.read();
        slots
            .iter()
            .filter_map(|slot| slot.as_ref())
            .filter(|entry| entry.key.starts_with(prefix))
            .map(|entry| entry.key.clone())
            .collect()
    }

    /// Get cache statistics.
    #[allow(clippy::cast_precision_loss)] // Acceptable for statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;

        CacheStats {
            hits,
            misses,
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
            entries: self.len(),
            capacity: self.capacity,
        }
    }

    /// Create a snapshot of the cache for serialization.
    pub fn snapshot(&self) -> CacheRingSnapshot<V> {
        let slots = self.slots.read();
        let entries: Vec<CacheEntrySnapshot<V>> = slots
            .iter()
            .filter_map(|slot| {
                slot.as_ref().map(|e| CacheEntrySnapshot {
                    key: e.key.clone(),
                    value: e.value.clone(),
                    access_count: e.access_count,
                    cost: e.cost,
                    size_bytes: e.size_bytes,
                })
            })
            .collect();
        drop(slots);

        CacheRingSnapshot {
            entries,
            capacity: self.capacity,
            strategy: self.strategy,
        }
    }

    /// Restore cache from a snapshot.
    #[must_use]
    pub fn restore(snapshot: CacheRingSnapshot<V>) -> Self {
        let cache = Self::new(snapshot.capacity, snapshot.strategy);

        for entry in snapshot.entries {
            cache.put(&entry.key, entry.value, entry.cost, entry.size_bytes);
        }

        cache
    }
}

impl<V: Clone> Default for CacheRing<V> {
    fn default() -> Self {
        Self::with_capacity(1000)
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub entries: usize,
    pub capacity: usize,
}

/// Serializable cache entry snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntrySnapshot<V> {
    pub key: String,
    pub value: V,
    pub access_count: u64,
    pub cost: f64,
    pub size_bytes: usize,
}

/// Serializable cache snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheRingSnapshot<V> {
    pub entries: Vec<CacheEntrySnapshot<V>>,
    pub capacity: usize,
    pub strategy: EvictionStrategy,
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        thread,
        time::{Duration, Instant},
    };

    use super::*;

    #[test]
    fn test_new() {
        let cache: CacheRing<String> = CacheRing::new(100, EvictionStrategy::LRU);
        assert_eq!(cache.capacity(), 100);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_with_capacity() {
        let cache: CacheRing<i32> = CacheRing::with_capacity(50);
        assert_eq!(cache.capacity(), 50);
    }

    #[test]
    fn test_default() {
        let cache: CacheRing<i32> = CacheRing::default();
        assert_eq!(cache.capacity(), 1000);
    }

    #[test]
    fn test_put_get() {
        let cache: CacheRing<String> = CacheRing::with_capacity(10);

        cache.put("key1", "value1".to_string(), 1.0, 100);
        cache.put("key2", "value2".to_string(), 1.0, 100);

        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        assert_eq!(cache.get("key2"), Some("value2".to_string()));
        assert_eq!(cache.get("key3"), None);
    }

    #[test]
    fn test_update() {
        let cache: CacheRing<String> = CacheRing::with_capacity(10);

        cache.put("key1", "value1".to_string(), 1.0, 100);
        cache.put("key1", "value2".to_string(), 2.0, 200);

        assert_eq!(cache.get("key1"), Some("value2".to_string()));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_delete() {
        let cache: CacheRing<String> = CacheRing::with_capacity(10);

        cache.put("key1", "value1".to_string(), 1.0, 100);
        assert!(cache.contains("key1"));

        assert!(cache.delete("key1"));
        assert!(!cache.contains("key1"));
        assert!(!cache.delete("key1")); // Already deleted
    }

    #[test]
    fn test_contains() {
        let cache: CacheRing<i32> = CacheRing::with_capacity(10);

        cache.put("key1", 42, 1.0, 8);
        assert!(cache.contains("key1"));
        assert!(!cache.contains("key2"));
    }

    #[test]
    fn test_clear() {
        let cache: CacheRing<i32> = CacheRing::with_capacity(10);

        for i in 0..5 {
            cache.put(&format!("key{}", i), i, 1.0, 8);
        }

        assert_eq!(cache.len(), 5);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lru_eviction() {
        let cache: CacheRing<i32> = CacheRing::new(3, EvictionStrategy::LRU);

        cache.put("a", 1, 1.0, 8);
        thread::sleep(Duration::from_millis(10));
        cache.put("b", 2, 1.0, 8);
        thread::sleep(Duration::from_millis(10));
        cache.put("c", 3, 1.0, 8);

        // Access "a" to make it most recently used
        let _ = cache.get("a");
        thread::sleep(Duration::from_millis(10));

        // This should evict "b" (least recently used)
        cache.put("d", 4, 1.0, 8);

        assert!(cache.contains("a"));
        assert!(!cache.contains("b")); // Evicted
        assert!(cache.contains("c"));
        assert!(cache.contains("d"));
    }

    #[test]
    fn test_lfu_eviction() {
        let cache: CacheRing<i32> = CacheRing::new(3, EvictionStrategy::LFU);

        cache.put("a", 1, 1.0, 8);
        cache.put("b", 2, 1.0, 8);
        cache.put("c", 3, 1.0, 8);

        // Access "a" multiple times
        for _ in 0..5 {
            let _ = cache.get("a");
        }
        // Access "c" a few times
        for _ in 0..3 {
            let _ = cache.get("c");
        }
        // "b" has lowest access count

        // This should evict "b" (least frequently used)
        cache.put("d", 4, 1.0, 8);

        assert!(cache.contains("a"));
        assert!(!cache.contains("b")); // Evicted
        assert!(cache.contains("c"));
        assert!(cache.contains("d"));
    }

    #[test]
    fn test_cost_based_eviction() {
        let cache: CacheRing<i32> = CacheRing::new(3, EvictionStrategy::CostBased);

        // High cost per byte
        cache.put("expensive", 1, 10.0, 10);
        // Low cost per byte
        cache.put("cheap", 2, 1.0, 1000);
        // Medium
        cache.put("medium", 3, 5.0, 100);

        // This should evict "cheap" (lowest cost/byte ratio)
        cache.put("new", 4, 1.0, 8);

        assert!(cache.contains("expensive"));
        assert!(!cache.contains("cheap")); // Evicted
        assert!(cache.contains("medium"));
        assert!(cache.contains("new"));
    }

    #[test]
    fn test_hybrid_eviction() {
        let cache: CacheRing<i32> = CacheRing::new(
            3,
            EvictionStrategy::Hybrid {
                lru_weight: 40,
                lfu_weight: 30,
                cost_weight: 30,
            },
        );

        cache.put("a", 1, 1.0, 8);
        cache.put("b", 2, 1.0, 8);
        cache.put("c", 3, 1.0, 8);

        // New entry triggers eviction
        cache.put("d", 4, 1.0, 8);

        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_stats() {
        let cache: CacheRing<i32> = CacheRing::with_capacity(10);

        cache.put("key1", 1, 1.0, 8);
        cache.put("key2", 2, 1.0, 8);

        let _ = cache.get("key1"); // Hit
        let _ = cache.get("key1"); // Hit
        let _ = cache.get("key3"); // Miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert!((stats.hit_rate - 0.666).abs() < 0.01);
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.capacity, 10);
    }

    #[test]
    fn test_snapshot_restore() {
        let cache: CacheRing<String> = CacheRing::with_capacity(10);

        cache.put("key1", "value1".to_string(), 1.0, 100);
        cache.put("key2", "value2".to_string(), 2.0, 200);

        let snapshot = cache.snapshot();
        let restored = CacheRing::restore(snapshot);

        assert_eq!(restored.get("key1"), Some("value1".to_string()));
        assert_eq!(restored.get("key2"), Some("value2".to_string()));
    }

    #[test]
    fn test_concurrent_reads_writes() {
        let cache = Arc::new(CacheRing::<i32>::with_capacity(100));

        let mut handles = vec![];

        // Writers
        for t in 0..4 {
            let c = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    c.put(&format!("key-{}-{}", t, i), i, 1.0, 8);
                }
            }));
        }

        // Readers
        for _ in 0..4 {
            let c = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let _ = c.get("key-0-0");
                    let _ = c.len();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Cache should have entries (may have evicted some due to capacity)
        assert!(cache.len() <= 100);
    }

    #[test]
    fn test_no_resize_stall() {
        let cache = CacheRing::<i32>::with_capacity(100);
        let count = 10_000;
        let mut max_op_time = Duration::ZERO;

        for i in 0..count {
            let start = Instant::now();
            cache.put(&format!("key{}", i), i, 1.0, 8);
            let elapsed = start.elapsed();
            if elapsed > max_op_time {
                max_op_time = elapsed;
            }
        }

        // Fixed capacity means no resize stalls
        assert!(
            max_op_time.as_millis() < 100,
            "Max operation time {:?} exceeded 100ms threshold",
            max_op_time
        );
    }

    #[test]
    fn test_eviction_scorer_lru() {
        let scorer = EvictionScorer::new(EvictionStrategy::LRU);

        let old_score = scorer.score(3600.0, 100, 0.5, 1000);
        let new_score = scorer.score(60.0, 100, 0.5, 1000);

        assert!(old_score < new_score);
    }

    #[test]
    fn test_eviction_scorer_lfu() {
        let scorer = EvictionScorer::new(EvictionStrategy::LFU);

        let low_freq = scorer.score(60.0, 1, 0.5, 1000);
        let high_freq = scorer.score(60.0, 100, 0.5, 1000);

        assert!(low_freq < high_freq);
    }

    #[test]
    fn test_eviction_scorer_cost() {
        let scorer = EvictionScorer::new(EvictionStrategy::CostBased);

        let low_value = scorer.score(60.0, 10, 0.001, 10000);
        let high_value = scorer.score(60.0, 10, 0.1, 100);

        assert!(low_value < high_value);
    }

    #[test]
    fn test_eviction_scorer_cost_zero_size() {
        let scorer = EvictionScorer::new(EvictionStrategy::CostBased);
        assert_eq!(scorer.score(60.0, 10, 0.5, 0), 0.0);
    }

    #[test]
    fn test_eviction_scorer_hybrid() {
        let scorer = EvictionScorer::new(EvictionStrategy::Hybrid {
            lru_weight: 40,
            lfu_weight: 30,
            cost_weight: 30,
        });

        let bad = scorer.score(3600.0, 1, 0.001, 1000);
        let good = scorer.score(60.0, 100, 0.1, 100);

        assert!(bad < good);
    }

    #[test]
    fn test_eviction_strategy_default() {
        let strategy = EvictionStrategy::default();
        assert!(matches!(strategy, EvictionStrategy::Hybrid { .. }));
    }

    #[test]
    fn test_cache_stats_fields() {
        let stats = CacheStats {
            hits: 10,
            misses: 5,
            hit_rate: 0.666,
            entries: 100,
            capacity: 1000,
        };

        assert_eq!(stats.hits, 10);
        assert_eq!(stats.misses, 5);
    }

    #[test]
    fn test_delete_nonexistent() {
        let cache: CacheRing<i32> = CacheRing::with_capacity(10);
        assert!(!cache.delete("nonexistent"));
    }

    #[test]
    fn test_get_miss() {
        let cache: CacheRing<i32> = CacheRing::with_capacity(10);
        assert!(cache.get("nonexistent").is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_evict_explicit_lru() {
        let cache: CacheRing<i32> = CacheRing::new(10, EvictionStrategy::LRU);
        for i in 0..5 {
            cache.put(&format!("key{i}"), i, 1.0, 8);
            thread::sleep(Duration::from_millis(5));
        }
        let evicted = cache.evict(2);
        assert_eq!(evicted, 2);
        assert_eq!(cache.len(), 3);
        // Oldest keys should be evicted
        assert!(!cache.contains("key0"));
        assert!(!cache.contains("key1"));
    }

    #[test]
    fn test_evict_explicit_lfu() {
        let cache: CacheRing<i32> = CacheRing::new(10, EvictionStrategy::LFU);
        cache.put("frequent", 1, 1.0, 8);
        cache.put("rare", 2, 1.0, 8);
        // Access frequent multiple times
        for _ in 0..5 {
            let _ = cache.get("frequent");
        }
        let evicted = cache.evict(1);
        assert_eq!(evicted, 1);
        assert!(cache.contains("frequent"));
        assert!(!cache.contains("rare"));
    }

    #[test]
    fn test_evict_explicit_cost_based() {
        let cache: CacheRing<i32> = CacheRing::new(10, EvictionStrategy::CostBased);
        cache.put("high_cost", 1, 100.0, 8);
        cache.put("low_cost", 2, 1.0, 8);
        let evicted = cache.evict(1);
        assert_eq!(evicted, 1);
        assert!(cache.contains("high_cost")); // High cost kept
    }

    #[test]
    fn test_evict_zero_count() {
        let cache: CacheRing<i32> = CacheRing::new(10, EvictionStrategy::LRU);
        cache.put("key", 1, 1.0, 8);
        let evicted = cache.evict(0);
        assert_eq!(evicted, 0);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_stats_zero_requests() {
        let cache: CacheRing<i32> = CacheRing::new(10, EvictionStrategy::LRU);
        let stats = cache.stats();
        assert_eq!(stats.hit_rate, 0.0);
    }

    #[test]
    fn test_scan_prefix() {
        let cache: CacheRing<i32> = CacheRing::with_capacity(20);
        cache.put("user:1", 1, 1.0, 8);
        cache.put("user:2", 2, 1.0, 8);
        cache.put("item:1", 10, 1.0, 8);
        cache.put("user:3", 3, 1.0, 8);

        let user_keys = cache.scan_prefix("user:");
        assert_eq!(user_keys.len(), 3);
        assert!(user_keys.iter().all(|k| k.starts_with("user:")));

        let item_keys = cache.scan_prefix("item:");
        assert_eq!(item_keys.len(), 1);
    }

    #[test]
    fn test_put_update_existing() {
        let cache: CacheRing<i32> = CacheRing::with_capacity(10);
        cache.put("key", 1, 1.0, 8);
        assert_eq!(cache.get("key"), Some(1));

        // Update the same key with new value
        cache.put("key", 2, 2.0, 16);
        assert_eq!(cache.get("key"), Some(2));

        // Should still have only one entry
        assert_eq!(cache.len(), 1);
    }
}
