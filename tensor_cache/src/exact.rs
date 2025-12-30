//! Exact match cache layer with O(1) lookup.

#![allow(dead_code)]

use crate::config::{CacheConfig, EvictionStrategy};
use crate::error::{CacheError, Result};
use crate::eviction::EvictionScorer;
use crate::stats::{CacheLayer, CacheStats};
use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Entry in the exact cache.
#[derive(Debug, Clone)]
pub struct ExactEntry {
    /// The cached response.
    pub response: String,
    /// Number of input tokens.
    pub input_tokens: usize,
    /// Number of output tokens.
    pub output_tokens: usize,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry expires.
    pub expires_at: Instant,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Last access time.
    pub last_accessed: Instant,
    /// Model used for the response.
    pub model: String,
}

impl ExactEntry {
    /// Check if this entry has expired.
    pub fn is_expired(&self) -> bool {
        Instant::now() >= self.expires_at
    }

    /// Record an access to this entry.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }
}

/// Generate a cache key from prompt, model, and parameters.
#[allow(dead_code)]
pub fn generate_key(prompt: &str, model: &str, params_hash: u64) -> String {
    let mut hasher = DefaultHasher::new();
    prompt.hash(&mut hasher);
    model.hash(&mut hasher);
    params_hash.hash(&mut hasher);
    format!("_cache:exact:{:016x}", hasher.finish())
}

/// Generate a cache key from prompt only.
pub fn generate_prompt_key(prompt: &str) -> String {
    let mut hasher = DefaultHasher::new();
    prompt.hash(&mut hasher);
    format!("_cache:exact:{:016x}", hasher.finish())
}

/// Exact match cache with O(1) lookup.
pub struct ExactCache {
    entries: DashMap<String, ExactEntry>,
    capacity: usize,
    stats: Arc<CacheStats>,
    eviction_strategy: EvictionStrategy,
    input_cost_per_1k: f64,
    output_cost_per_1k: f64,
}

impl ExactCache {
    /// Create a new exact cache.
    pub fn new(config: &CacheConfig, stats: Arc<CacheStats>) -> Self {
        Self {
            entries: DashMap::with_capacity(config.exact_capacity),
            capacity: config.exact_capacity,
            stats,
            eviction_strategy: config.eviction_strategy,
            input_cost_per_1k: config.input_cost_per_1k,
            output_cost_per_1k: config.output_cost_per_1k,
        }
    }

    /// Look up an entry by key.
    pub fn get(&self, key: &str) -> Option<ExactEntry> {
        if let Some(mut entry) = self.entries.get_mut(key) {
            if entry.is_expired() {
                drop(entry);
                self.entries.remove(key);
                self.stats.decrement_size(CacheLayer::Exact);
                self.stats.record_expiration(1);
                self.stats.record_miss(CacheLayer::Exact);
                return None;
            }
            entry.record_access();
            self.stats.record_hit(CacheLayer::Exact);
            Some(entry.clone())
        } else {
            self.stats.record_miss(CacheLayer::Exact);
            None
        }
    }

    /// Insert an entry into the cache.
    pub fn insert(
        &self,
        key: String,
        response: String,
        input_tokens: usize,
        output_tokens: usize,
        model: String,
        ttl: Duration,
    ) -> Result<()> {
        if self.entries.len() >= self.capacity {
            return Err(CacheError::CacheFull {
                current: self.entries.len(),
                capacity: self.capacity,
            });
        }

        let now = Instant::now();
        let entry = ExactEntry {
            response,
            input_tokens,
            output_tokens,
            created_at: now,
            expires_at: now + ttl,
            access_count: 0,
            last_accessed: now,
            model,
        };

        let is_new = self.entries.insert(key, entry).is_none();
        if is_new {
            self.stats.increment_size(CacheLayer::Exact);
        }
        Ok(())
    }

    /// Remove an entry from the cache.
    pub fn remove(&self, key: &str) -> Option<ExactEntry> {
        if let Some((_, entry)) = self.entries.remove(key) {
            self.stats.decrement_size(CacheLayer::Exact);
            Some(entry)
        } else {
            None
        }
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Get the current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        let count = self.entries.len();
        self.entries.clear();
        self.stats.set_size(CacheLayer::Exact, 0);
        self.stats.record_eviction(count);
    }

    /// Remove all expired entries. Returns count of removed entries.
    pub fn cleanup_expired(&self) -> usize {
        let mut expired_keys = Vec::new();

        for entry in self.entries.iter() {
            if entry.is_expired() {
                expired_keys.push(entry.key().clone());
            }
        }

        for key in &expired_keys {
            self.entries.remove(key);
        }

        let count = expired_keys.len();
        if count > 0 {
            self.stats.set_size(CacheLayer::Exact, self.entries.len());
            self.stats.record_expiration(count);
        }

        count
    }

    /// Get candidates for eviction, sorted by score (lowest first).
    pub fn eviction_candidates(&self, limit: usize) -> Vec<(String, f64)> {
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .map(|entry| {
                let key = entry.key().clone();
                let score = self.eviction_score(entry.value());
                (key, score)
            })
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);
        candidates
    }

    /// Calculate eviction score for an entry (lower = more likely to evict).
    fn eviction_score(&self, entry: &ExactEntry) -> f64 {
        let scorer = EvictionScorer::new(self.eviction_strategy);
        let last_access_secs = entry.last_accessed.elapsed().as_secs_f64();
        let cost_per_hit = (entry.input_tokens as f64 * self.input_cost_per_1k
            + entry.output_tokens as f64 * self.output_cost_per_1k)
            / 1000.0;
        let size_bytes = entry.response.len();

        scorer.score(
            last_access_secs,
            entry.access_count,
            cost_per_hit,
            size_bytes,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cache() -> ExactCache {
        let config = CacheConfig::default();
        let stats = Arc::new(CacheStats::new());
        ExactCache::new(&config, stats)
    }

    #[test]
    fn test_insert_and_get() {
        let cache = create_test_cache();

        cache
            .insert(
                "key1".into(),
                "response1".into(),
                10,
                20,
                "gpt-4".into(),
                Duration::from_secs(60),
            )
            .unwrap();

        let entry = cache.get("key1").unwrap();
        assert_eq!(entry.response, "response1");
        assert_eq!(entry.input_tokens, 10);
        assert_eq!(entry.output_tokens, 20);
    }

    #[test]
    fn test_get_nonexistent() {
        let cache = create_test_cache();
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_remove() {
        let cache = create_test_cache();

        cache
            .insert(
                "key1".into(),
                "response1".into(),
                10,
                20,
                "gpt-4".into(),
                Duration::from_secs(60),
            )
            .unwrap();

        let removed = cache.remove("key1");
        assert!(removed.is_some());
        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_expired_entry() {
        let cache = create_test_cache();

        cache
            .insert(
                "key1".into(),
                "response1".into(),
                10,
                20,
                "gpt-4".into(),
                Duration::from_millis(1),
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));

        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_access_count() {
        let cache = create_test_cache();

        cache
            .insert(
                "key1".into(),
                "response1".into(),
                10,
                20,
                "gpt-4".into(),
                Duration::from_secs(60),
            )
            .unwrap();

        cache.get("key1");
        cache.get("key1");
        let entry = cache.get("key1").unwrap();

        assert_eq!(entry.access_count, 3);
    }

    #[test]
    fn test_generate_key() {
        let key1 = generate_key("prompt", "gpt-4", 123);
        let key2 = generate_key("prompt", "gpt-4", 123);
        let key3 = generate_key("prompt", "gpt-4", 456);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert!(key1.starts_with("_cache:exact:"));
    }

    #[test]
    fn test_capacity_limit() {
        let mut config = CacheConfig::default();
        config.exact_capacity = 2;
        let stats = Arc::new(CacheStats::new());
        let cache = ExactCache::new(&config, stats);

        cache
            .insert(
                "k1".into(),
                "r1".into(),
                1,
                1,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();
        cache
            .insert(
                "k2".into(),
                "r2".into(),
                1,
                1,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();

        let result = cache.insert(
            "k3".into(),
            "r3".into(),
            1,
            1,
            "m".into(),
            Duration::from_secs(60),
        );
        assert!(matches!(result, Err(CacheError::CacheFull { .. })));
    }

    #[test]
    fn test_clear() {
        let cache = create_test_cache();

        cache
            .insert(
                "k1".into(),
                "r1".into(),
                1,
                1,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();
        cache
            .insert(
                "k2".into(),
                "r2".into(),
                1,
                1,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();

        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_eviction_uses_lru_strategy() {
        let mut config = CacheConfig::default();
        config.eviction_strategy = EvictionStrategy::LRU;
        let stats = Arc::new(CacheStats::new());
        let cache = ExactCache::new(&config, stats);

        // Insert two entries
        cache
            .insert(
                "old".into(),
                "r".into(),
                10,
                10,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(50));
        cache
            .insert(
                "new".into(),
                "r".into(),
                10,
                10,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();

        // With LRU, older entries should have lower scores (evicted first)
        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // First candidate should be "old" (lower score = older = evict first)
        assert_eq!(candidates[0].0, "old");
    }

    #[test]
    fn test_eviction_uses_lfu_strategy() {
        let mut config = CacheConfig::default();
        config.eviction_strategy = EvictionStrategy::LFU;
        let stats = Arc::new(CacheStats::new());
        let cache = ExactCache::new(&config, stats);

        cache
            .insert(
                "rarely".into(),
                "r".into(),
                10,
                10,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();
        cache
            .insert(
                "often".into(),
                "r".into(),
                10,
                10,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();

        // Access "often" multiple times
        cache.get("often");
        cache.get("often");
        cache.get("often");

        // With LFU, less frequently accessed should have lower scores
        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // First candidate should be "rarely" (lower frequency = evict first)
        assert_eq!(candidates[0].0, "rarely");
    }

    #[test]
    fn test_eviction_uses_hybrid_strategy() {
        let mut config = CacheConfig::default();
        config.eviction_strategy = EvictionStrategy::Hybrid {
            lru_weight: 40,
            lfu_weight: 30,
            cost_weight: 30,
        };
        let stats = Arc::new(CacheStats::new());
        let cache = ExactCache::new(&config, stats);

        // Insert entries with different characteristics
        cache
            .insert(
                "low_value".into(),
                "x".into(),
                1,
                1,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();
        cache
            .insert(
                "high_value".into(),
                "x".repeat(100),
                100,
                100,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();

        // Access high_value more
        cache.get("high_value");
        cache.get("high_value");

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // low_value should be evicted first (less value)
        assert_eq!(candidates[0].0, "low_value");
    }

    #[test]
    fn test_eviction_uses_cost_strategy() {
        let mut config = CacheConfig::default();
        config.eviction_strategy = EvictionStrategy::CostBased;
        config.input_cost_per_1k = 0.01;
        config.output_cost_per_1k = 0.03;
        let stats = Arc::new(CacheStats::new());
        let cache = ExactCache::new(&config, stats);

        // Low cost entry (few tokens)
        cache
            .insert(
                "cheap".into(),
                "x".into(),
                1,
                1,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();
        // High cost entry (many tokens)
        cache
            .insert(
                "expensive".into(),
                "y".into(),
                1000,
                1000,
                "m".into(),
                Duration::from_secs(60),
            )
            .unwrap();

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // Cheap entry should be evicted first (lower cost savings)
        assert_eq!(candidates[0].0, "cheap");
    }
}
