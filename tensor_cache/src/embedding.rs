//! Embedding cache layer with O(1) lookup.

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

/// Entry in the embedding cache.
#[derive(Debug, Clone)]
pub struct EmbeddingEntry {
    /// The cached embedding vector.
    pub embedding: Vec<f32>,
    /// Model used to generate this embedding.
    pub model: String,
    /// Dimension of the embedding.
    pub dimension: usize,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry expires (optional, embeddings often don't expire).
    pub expires_at: Option<Instant>,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Last access time.
    pub last_accessed: Instant,
}

impl EmbeddingEntry {
    /// Check if this entry has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Instant::now() >= expires_at
        } else {
            false
        }
    }

    /// Record an access to this entry.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }
}

/// Generate a cache key from source and content hash.
pub fn generate_key(source: &str, content_hash: u64) -> String {
    format!("_cache:emb:{}:{:016x}", source, content_hash)
}

/// Compute a hash of the content.
pub fn hash_content(content: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

/// Embedding cache with O(1) lookup.
pub struct EmbeddingCache {
    entries: DashMap<String, EmbeddingEntry>,
    capacity: usize,
    stats: Arc<CacheStats>,
    eviction_strategy: EvictionStrategy,
}

impl EmbeddingCache {
    /// Create a new embedding cache.
    pub fn new(config: &CacheConfig, stats: Arc<CacheStats>) -> Self {
        Self {
            entries: DashMap::with_capacity(config.embedding_capacity),
            capacity: config.embedding_capacity,
            stats,
            eviction_strategy: config.eviction_strategy,
        }
    }

    /// Look up an embedding by key.
    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        if let Some(mut entry) = self.entries.get_mut(key) {
            if entry.is_expired() {
                drop(entry);
                self.entries.remove(key);
                self.stats.decrement_size(CacheLayer::Embedding);
                self.stats.record_expiration(1);
                self.stats.record_miss(CacheLayer::Embedding);
                return None;
            }
            entry.record_access();
            self.stats.record_hit(CacheLayer::Embedding);
            Some(entry.embedding.clone())
        } else {
            self.stats.record_miss(CacheLayer::Embedding);
            None
        }
    }

    /// Look up by source and content.
    pub fn get_by_content(&self, source: &str, content: &str) -> Option<Vec<f32>> {
        let key = generate_key(source, hash_content(content));
        self.get(&key)
    }

    /// Insert an embedding into the cache.
    pub fn insert(
        &self,
        key: String,
        embedding: Vec<f32>,
        model: String,
        ttl: Option<Duration>,
    ) -> Result<()> {
        if self.entries.len() >= self.capacity {
            return Err(CacheError::CacheFull {
                current: self.entries.len(),
                capacity: self.capacity,
            });
        }

        let now = Instant::now();
        let dimension = embedding.len();

        let entry = EmbeddingEntry {
            embedding,
            model,
            dimension,
            created_at: now,
            expires_at: ttl.map(|d| now + d),
            access_count: 0,
            last_accessed: now,
        };

        let is_new = self.entries.insert(key, entry).is_none();
        if is_new {
            self.stats.increment_size(CacheLayer::Embedding);
        }
        Ok(())
    }

    /// Insert by source and content.
    pub fn insert_by_content(
        &self,
        source: &str,
        content: &str,
        embedding: Vec<f32>,
        model: String,
        ttl: Option<Duration>,
    ) -> Result<()> {
        let key = generate_key(source, hash_content(content));
        self.insert(key, embedding, model, ttl)
    }

    /// Get or compute an embedding.
    ///
    /// If the embedding is cached, returns it. Otherwise, calls the compute
    /// function and caches the result.
    pub fn get_or_compute<F>(
        &self,
        source: &str,
        content: &str,
        model: &str,
        ttl: Option<Duration>,
        compute: F,
    ) -> Result<Vec<f32>>
    where
        F: FnOnce() -> Result<Vec<f32>>,
    {
        let key = generate_key(source, hash_content(content));

        if let Some(embedding) = self.get(&key) {
            return Ok(embedding);
        }

        let embedding = compute()?;
        self.insert(key, embedding.clone(), model.to_string(), ttl)?;
        Ok(embedding)
    }

    /// Remove an entry from the cache.
    pub fn remove(&self, key: &str) -> Option<EmbeddingEntry> {
        if let Some((_, entry)) = self.entries.remove(key) {
            self.stats.decrement_size(CacheLayer::Embedding);
            Some(entry)
        } else {
            None
        }
    }

    /// Invalidate all embeddings for a source.
    pub fn invalidate_source(&self, source: &str) -> usize {
        let prefix = format!("_cache:emb:{}:", source);
        let mut to_remove = Vec::new();

        for entry in self.entries.iter() {
            if entry.key().starts_with(&prefix) {
                to_remove.push(entry.key().clone());
            }
        }

        for key in &to_remove {
            self.remove(key);
        }

        to_remove.len()
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
        self.stats.set_size(CacheLayer::Embedding, 0);
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
            self.stats
                .set_size(CacheLayer::Embedding, self.entries.len());
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

    /// Calculate eviction score for an entry.
    fn eviction_score(&self, entry: &EmbeddingEntry) -> f64 {
        let scorer = EvictionScorer::new(self.eviction_strategy);
        let last_access_secs = entry.last_accessed.elapsed().as_secs_f64();
        // Embeddings don't have token costs, but we can use size as proxy
        let size_bytes = entry.embedding.len() * std::mem::size_of::<f32>();

        scorer.score(last_access_secs, entry.access_count, 0.0, size_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cache() -> EmbeddingCache {
        let config = CacheConfig::default();
        let stats = Arc::new(CacheStats::new());
        EmbeddingCache::new(&config, stats)
    }

    #[test]
    fn test_insert_and_get() {
        let cache = create_test_cache();
        let embedding = vec![0.1, 0.2, 0.3];

        cache
            .insert(
                "key1".into(),
                embedding.clone(),
                "text-embedding-3-small".into(),
                None,
            )
            .unwrap();

        let retrieved = cache.get("key1").unwrap();
        assert_eq!(retrieved, embedding);
    }

    #[test]
    fn test_get_by_content() {
        let cache = create_test_cache();
        let embedding = vec![0.1, 0.2, 0.3];

        cache
            .insert_by_content(
                "doc",
                "Hello world",
                embedding.clone(),
                "model".into(),
                None,
            )
            .unwrap();

        let retrieved = cache.get_by_content("doc", "Hello world").unwrap();
        assert_eq!(retrieved, embedding);
    }

    #[test]
    fn test_get_or_compute_cached() {
        let cache = create_test_cache();
        let embedding = vec![0.1, 0.2, 0.3];

        cache
            .insert_by_content("doc", "content", embedding.clone(), "model".into(), None)
            .unwrap();

        let mut compute_called = false;
        let result = cache
            .get_or_compute("doc", "content", "model", None, || {
                compute_called = true;
                Ok(vec![0.4, 0.5, 0.6])
            })
            .unwrap();

        assert!(!compute_called);
        assert_eq!(result, embedding);
    }

    #[test]
    fn test_get_or_compute_miss() {
        let cache = create_test_cache();

        let mut compute_called = false;
        let result = cache
            .get_or_compute("doc", "content", "model", None, || {
                compute_called = true;
                Ok(vec![0.1, 0.2, 0.3])
            })
            .unwrap();

        assert!(compute_called);
        assert_eq!(result, vec![0.1, 0.2, 0.3]);

        // Should be cached now
        let result2 = cache.get_by_content("doc", "content").unwrap();
        assert_eq!(result2, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_invalidate_source() {
        let cache = create_test_cache();

        cache
            .insert_by_content("doc1", "a", vec![0.1], "m".into(), None)
            .unwrap();
        cache
            .insert_by_content("doc1", "b", vec![0.2], "m".into(), None)
            .unwrap();
        cache
            .insert_by_content("doc2", "c", vec![0.3], "m".into(), None)
            .unwrap();

        assert_eq!(cache.len(), 3);
        let removed = cache.invalidate_source("doc1");
        assert_eq!(removed, 2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_expired_entry() {
        let cache = create_test_cache();

        cache
            .insert(
                "key1".into(),
                vec![0.1],
                "m".into(),
                Some(Duration::from_millis(1)),
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));
        assert!(cache.get("key1").is_none());
    }

    #[test]
    fn test_no_expiry() {
        let cache = create_test_cache();

        cache
            .insert("key1".into(), vec![0.1], "m".into(), None)
            .unwrap();

        // Entry without TTL should not expire
        let entry = cache.entries.get("key1").unwrap();
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_generate_key() {
        let key1 = generate_key("doc", 123);
        let key2 = generate_key("doc", 123);
        let key3 = generate_key("doc", 456);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert!(key1.starts_with("_cache:emb:doc:"));
    }

    #[test]
    fn test_hash_content() {
        let h1 = hash_content("hello");
        let h2 = hash_content("hello");
        let h3 = hash_content("world");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_remove() {
        let cache = create_test_cache();
        let embedding = vec![0.1, 0.2, 0.3];

        cache
            .insert("key1".into(), embedding.clone(), "m".into(), None)
            .unwrap();

        assert!(cache.contains("key1"));
        let removed = cache.remove("key1");
        assert!(removed.is_some());
        assert!(!cache.contains("key1"));
    }

    #[test]
    fn test_remove_nonexistent() {
        let cache = create_test_cache();
        let removed = cache.remove("nonexistent");
        assert!(removed.is_none());
    }

    #[test]
    fn test_clear() {
        let cache = create_test_cache();

        cache
            .insert("key1".into(), vec![0.1], "m".into(), None)
            .unwrap();
        cache
            .insert("key2".into(), vec![0.2], "m".into(), None)
            .unwrap();

        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cleanup_expired() {
        let cache = create_test_cache();

        cache
            .insert(
                "key1".into(),
                vec![0.1],
                "m".into(),
                Some(Duration::from_millis(1)),
            )
            .unwrap();
        cache
            .insert("key2".into(), vec![0.2], "m".into(), None)
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));
        let cleaned = cache.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cleanup_expired_empty() {
        let cache = create_test_cache();
        let cleaned = cache.cleanup_expired();
        assert_eq!(cleaned, 0);
    }

    #[test]
    fn test_eviction_candidates() {
        let cache = create_test_cache();

        cache
            .insert("key1".into(), vec![0.1], "m".into(), None)
            .unwrap();
        cache
            .insert("key2".into(), vec![0.2], "m".into(), None)
            .unwrap();

        // Access key1 to increase its score
        cache.get("key1");

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // key2 should have lower score (less frequently accessed)
        assert_eq!(candidates[0].0, "key2");
    }

    #[test]
    fn test_eviction_candidates_limit() {
        let cache = create_test_cache();

        for i in 0..10 {
            cache
                .insert(format!("key{}", i), vec![0.1], "m".into(), None)
                .unwrap();
        }

        let candidates = cache.eviction_candidates(3);
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_contains() {
        let cache = create_test_cache();

        assert!(!cache.contains("key1"));
        cache
            .insert("key1".into(), vec![0.1], "m".into(), None)
            .unwrap();
        assert!(cache.contains("key1"));
    }

    #[test]
    fn test_cache_full_error() {
        let mut config = CacheConfig::default();
        config.embedding_capacity = 2;
        let stats = Arc::new(CacheStats::new());
        let cache = EmbeddingCache::new(&config, stats);

        cache
            .insert("key1".into(), vec![0.1], "m".into(), None)
            .unwrap();
        cache
            .insert("key2".into(), vec![0.2], "m".into(), None)
            .unwrap();

        let result = cache.insert("key3".into(), vec![0.3], "m".into(), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_entry_record_access() {
        let cache = create_test_cache();
        cache
            .insert("key1".into(), vec![0.1], "m".into(), None)
            .unwrap();

        // Access multiple times
        cache.get("key1");
        cache.get("key1");

        let entry = cache.entries.get("key1").unwrap();
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_is_empty() {
        let cache = create_test_cache();
        assert!(cache.is_empty());

        cache
            .insert("key1".into(), vec![0.1], "m".into(), None)
            .unwrap();
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_entry_dimension() {
        let cache = create_test_cache();
        cache
            .insert("key1".into(), vec![0.1, 0.2, 0.3], "m".into(), None)
            .unwrap();

        let entry = cache.entries.get("key1").unwrap();
        assert_eq!(entry.dimension, 3);
    }

    #[test]
    fn test_get_or_compute_error() {
        let cache = create_test_cache();

        let result = cache.get_or_compute("doc", "content", "model", None, || {
            Err(CacheError::NotFound("test error".into()))
        });

        assert!(result.is_err());
    }

    #[test]
    fn test_eviction_uses_lru_strategy() {
        let mut config = CacheConfig::default();
        config.eviction_strategy = EvictionStrategy::LRU;
        let stats = Arc::new(CacheStats::new());
        let cache = EmbeddingCache::new(&config, stats);

        cache
            .insert("old".into(), vec![0.1, 0.2], "m".into(), None)
            .unwrap();
        std::thread::sleep(Duration::from_millis(50));
        cache
            .insert("new".into(), vec![0.3, 0.4], "m".into(), None)
            .unwrap();

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // Older entry should be first (lower score)
        assert_eq!(candidates[0].0, "old");
    }

    #[test]
    fn test_eviction_uses_lfu_strategy() {
        let mut config = CacheConfig::default();
        config.eviction_strategy = EvictionStrategy::LFU;
        let stats = Arc::new(CacheStats::new());
        let cache = EmbeddingCache::new(&config, stats);

        cache
            .insert("rarely".into(), vec![0.1, 0.2], "m".into(), None)
            .unwrap();
        cache
            .insert("often".into(), vec![0.3, 0.4], "m".into(), None)
            .unwrap();

        // Access "often" multiple times
        cache.get("often");
        cache.get("often");
        cache.get("often");

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // Less frequently accessed should have lower score
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
        let cache = EmbeddingCache::new(&config, stats);

        // Small embedding
        cache
            .insert("small".into(), vec![0.1], "m".into(), None)
            .unwrap();
        // Large embedding
        cache
            .insert("large".into(), vec![0.1; 100], "m".into(), None)
            .unwrap();

        // Access large more
        cache.get("large");
        cache.get("large");

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // Small (less valuable) should have lower score
        assert_eq!(candidates[0].0, "small");
    }
}
