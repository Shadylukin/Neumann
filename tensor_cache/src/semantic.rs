//! Semantic similarity cache layer with O(log n) lookup via HNSW.

#![allow(dead_code)]

use crate::config::{CacheConfig, EvictionStrategy};
use crate::error::{CacheError, Result};
use crate::eviction::EvictionScorer;
use crate::index::CacheIndex;
use crate::stats::{CacheLayer, CacheStats};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tensor_store::{EmbeddingStorage, HNSWMemoryStats, SparseVector};
use uuid::Uuid;

/// Entry in the semantic cache.
#[derive(Debug, Clone)]
pub struct SemanticEntry {
    /// Unique identifier for this entry.
    pub id: String,
    /// The original query text.
    pub query: String,
    /// The cached response.
    pub response: String,
    /// Number of input tokens.
    pub input_tokens: usize,
    /// Number of output tokens.
    pub output_tokens: usize,
    /// Model used for the response.
    pub model: String,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry expires.
    pub expires_at: Instant,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Last access time.
    pub last_accessed: Instant,
    /// Version tag for invalidation.
    pub version: Option<String>,
}

impl SemanticEntry {
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

/// Result of a semantic cache lookup.
#[derive(Debug, Clone)]
pub struct SemanticHit {
    /// The cached response.
    pub response: String,
    /// The original query that produced this response.
    pub query: String,
    /// Similarity between the lookup query and the cached query.
    pub similarity: f32,
    /// Number of input tokens saved.
    pub input_tokens: usize,
    /// Number of output tokens saved.
    pub output_tokens: usize,
    /// Model used.
    pub model: String,
}

/// Semantic cache with HNSW-based similarity search.
pub struct SemanticCache {
    /// HNSW index for semantic search.
    index: CacheIndex,
    /// Entry storage.
    entries: DashMap<String, SemanticEntry>,
    /// Capacity limit.
    capacity: usize,
    /// Default similarity threshold.
    threshold: f32,
    /// Statistics tracker.
    stats: Arc<CacheStats>,
    /// Eviction strategy.
    eviction_strategy: EvictionStrategy,
    /// Cost per 1000 input tokens.
    input_cost_per_1k: f64,
    /// Cost per 1000 output tokens.
    output_cost_per_1k: f64,
}

impl SemanticCache {
    /// Create a new semantic cache.
    pub fn new(config: &CacheConfig, stats: Arc<CacheStats>) -> Self {
        Self {
            index: CacheIndex::new(config.embedding_dim),
            entries: DashMap::with_capacity(config.semantic_capacity),
            capacity: config.semantic_capacity,
            threshold: config.semantic_threshold,
            stats,
            eviction_strategy: config.eviction_strategy,
            input_cost_per_1k: config.input_cost_per_1k,
            output_cost_per_1k: config.output_cost_per_1k,
        }
    }

    /// Look up a semantically similar entry.
    pub fn get(&self, embedding: &[f32], threshold: Option<f32>) -> Option<SemanticHit> {
        let threshold = threshold.unwrap_or(self.threshold);

        match self.index.search(embedding, 1, threshold) {
            Ok(results) if !results.is_empty() => {
                let result = &results[0];
                if let Some(mut entry) = self.entries.get_mut(&result.key) {
                    if entry.is_expired() {
                        drop(entry);
                        self.remove(&result.key);
                        self.stats.record_miss(CacheLayer::Semantic);
                        return None;
                    }
                    entry.record_access();
                    self.stats.record_hit(CacheLayer::Semantic);
                    Some(SemanticHit {
                        response: entry.response.clone(),
                        query: entry.query.clone(),
                        similarity: result.similarity,
                        input_tokens: entry.input_tokens,
                        output_tokens: entry.output_tokens,
                        model: entry.model.clone(),
                    })
                } else {
                    self.stats.record_miss(CacheLayer::Semantic);
                    None
                }
            },
            _ => {
                self.stats.record_miss(CacheLayer::Semantic);
                None
            },
        }
    }

    /// Insert an entry into the semantic cache.
    pub fn insert(
        &self,
        query: String,
        embedding: &[f32],
        response: String,
        input_tokens: usize,
        output_tokens: usize,
        model: String,
        ttl: Duration,
        version: Option<String>,
    ) -> Result<String> {
        if self.entries.len() >= self.capacity {
            return Err(CacheError::CacheFull {
                current: self.entries.len(),
                capacity: self.capacity,
            });
        }

        let id = format!("_cache:sem:{}", Uuid::new_v4());
        let now = Instant::now();

        let entry = SemanticEntry {
            id: id.clone(),
            query,
            response,
            input_tokens,
            output_tokens,
            model,
            created_at: now,
            expires_at: now + ttl,
            access_count: 0,
            last_accessed: now,
            version,
        };

        // Insert into HNSW index
        self.index.insert(&id, embedding)?;

        // Insert into entry storage
        let is_new = self.entries.insert(id.clone(), entry).is_none();
        if is_new {
            self.stats.increment_size(CacheLayer::Semantic);
        }

        Ok(id)
    }

    /// Look up a semantically similar entry using a sparse query.
    pub fn get_sparse(
        &self,
        embedding: &SparseVector,
        threshold: Option<f32>,
    ) -> Option<SemanticHit> {
        let threshold = threshold.unwrap_or(self.threshold);

        match self.index.search_sparse(embedding, 1, threshold) {
            Ok(results) if !results.is_empty() => {
                let result = &results[0];
                if let Some(mut entry) = self.entries.get_mut(&result.key) {
                    if entry.is_expired() {
                        drop(entry);
                        self.remove(&result.key);
                        self.stats.record_miss(CacheLayer::Semantic);
                        return None;
                    }
                    entry.record_access();
                    self.stats.record_hit(CacheLayer::Semantic);
                    Some(SemanticHit {
                        response: entry.response.clone(),
                        query: entry.query.clone(),
                        similarity: result.similarity,
                        input_tokens: entry.input_tokens,
                        output_tokens: entry.output_tokens,
                        model: entry.model.clone(),
                    })
                } else {
                    self.stats.record_miss(CacheLayer::Semantic);
                    None
                }
            },
            _ => {
                self.stats.record_miss(CacheLayer::Semantic);
                None
            },
        }
    }

    /// Insert an entry with a sparse embedding.
    pub fn insert_sparse(
        &self,
        query: String,
        embedding: &SparseVector,
        response: String,
        input_tokens: usize,
        output_tokens: usize,
        model: String,
        ttl: Duration,
        version: Option<String>,
    ) -> Result<String> {
        if self.entries.len() >= self.capacity {
            return Err(CacheError::CacheFull {
                current: self.entries.len(),
                capacity: self.capacity,
            });
        }

        let id = format!("_cache:sem:{}", Uuid::new_v4());
        let now = Instant::now();

        let entry = SemanticEntry {
            id: id.clone(),
            query,
            response,
            input_tokens,
            output_tokens,
            model,
            created_at: now,
            expires_at: now + ttl,
            access_count: 0,
            last_accessed: now,
            version,
        };

        // Insert into HNSW index
        self.index.insert_sparse(&id, embedding)?;

        // Insert into entry storage
        let is_new = self.entries.insert(id.clone(), entry).is_none();
        if is_new {
            self.stats.increment_size(CacheLayer::Semantic);
        }

        Ok(id)
    }

    /// Insert with automatic sparse/dense selection based on sparsity.
    pub fn insert_auto(
        &self,
        query: String,
        embedding: &[f32],
        sparsity_threshold: f32,
        response: String,
        input_tokens: usize,
        output_tokens: usize,
        model: String,
        ttl: Duration,
        version: Option<String>,
    ) -> Result<String> {
        if self.entries.len() >= self.capacity {
            return Err(CacheError::CacheFull {
                current: self.entries.len(),
                capacity: self.capacity,
            });
        }

        let id = format!("_cache:sem:{}", Uuid::new_v4());
        let now = Instant::now();

        let entry = SemanticEntry {
            id: id.clone(),
            query,
            response,
            input_tokens,
            output_tokens,
            model,
            created_at: now,
            expires_at: now + ttl,
            access_count: 0,
            last_accessed: now,
            version,
        };

        // Insert into HNSW index with auto-sparsification
        self.index.insert_auto(&id, embedding, sparsity_threshold)?;

        // Insert into entry storage
        let is_new = self.entries.insert(id.clone(), entry).is_none();
        if is_new {
            self.stats.increment_size(CacheLayer::Semantic);
        }

        Ok(id)
    }

    /// Get memory statistics for the HNSW index.
    pub fn memory_stats(&self) -> HNSWMemoryStats {
        self.index.memory_stats()
    }

    /// Get the embedding storage for a cache entry.
    pub fn get_embedding(&self, id: &str) -> Option<EmbeddingStorage> {
        self.index.get_embedding(id)
    }

    /// Remove an entry by ID.
    pub fn remove(&self, id: &str) -> Option<SemanticEntry> {
        self.index.remove(id);
        if let Some((_, entry)) = self.entries.remove(id) {
            self.stats.decrement_size(CacheLayer::Semantic);
            Some(entry)
        } else {
            None
        }
    }

    /// Check if an ID exists.
    pub fn contains(&self, id: &str) -> bool {
        self.entries.contains_key(id)
    }

    /// Get the current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries (thread-safe).
    pub fn clear(&self) {
        let count = self.entries.len();
        self.entries.clear();
        self.index.clear();
        self.stats.set_size(CacheLayer::Semantic, 0);
        self.stats.record_eviction(count);
    }

    /// Remove all expired entries. Returns count of removed entries.
    pub fn cleanup_expired(&self) -> usize {
        let mut expired_ids = Vec::new();

        for entry in self.entries.iter() {
            if entry.is_expired() {
                expired_ids.push(entry.key().clone());
            }
        }

        for id in &expired_ids {
            self.remove(id);
        }

        let count = expired_ids.len();
        if count > 0 {
            self.stats.record_expiration(count);
        }

        count
    }

    /// Invalidate entries by version.
    pub fn invalidate_version(&self, version: &str) -> usize {
        let mut to_remove = Vec::new();

        for entry in self.entries.iter() {
            if entry.version.as_deref() == Some(version) {
                to_remove.push(entry.key().clone());
            }
        }

        for id in &to_remove {
            self.remove(id);
        }

        to_remove.len()
    }

    /// Get candidates for eviction, sorted by score (lowest first).
    pub fn eviction_candidates(&self, limit: usize) -> Vec<(String, f64)> {
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .map(|entry| {
                let id = entry.key().clone();
                let score = self.eviction_score(entry.value());
                (id, score)
            })
            .collect();

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(limit);
        candidates
    }

    /// Calculate eviction score for an entry.
    fn eviction_score(&self, entry: &SemanticEntry) -> f64 {
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

    fn create_test_cache() -> SemanticCache {
        let mut config = CacheConfig::default();
        config.embedding_dim = 3;
        let stats = Arc::new(CacheStats::new());
        SemanticCache::new(&config, stats)
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag == 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / mag).collect()
        }
    }

    #[test]
    fn test_insert_and_get() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .insert(
                "What is 2+2?".into(),
                &embedding,
                "4".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        let hit = cache.get(&embedding, None).unwrap();
        assert_eq!(hit.response, "4");
        assert_eq!(hit.query, "What is 2+2?");
        assert!(hit.similarity > 0.99);
    }

    #[test]
    fn test_semantic_similarity() {
        let cache = create_test_cache();

        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.9, 0.1, 0.0]); // Similar

        cache
            .insert(
                "query1".into(),
                &v1,
                "response1".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Should find similar entry with low threshold
        let hit = cache.get(&v2, Some(0.8)).unwrap();
        assert_eq!(hit.response, "response1");
    }

    #[test]
    fn test_threshold_miss() {
        let cache = create_test_cache();

        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]); // Orthogonal

        cache
            .insert(
                "query1".into(),
                &v1,
                "response1".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // High threshold should miss orthogonal vector
        assert!(cache.get(&v2, Some(0.9)).is_none());
    }

    #[test]
    fn test_remove() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        let id = cache
            .insert(
                "query".into(),
                &embedding,
                "response".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        assert!(cache.contains(&id));
        cache.remove(&id);
        assert!(!cache.contains(&id));
        assert!(cache.get(&embedding, None).is_none());
    }

    #[test]
    fn test_version_invalidation() {
        let cache = create_test_cache();

        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]);

        cache
            .insert(
                "q1".into(),
                &v1,
                "r1".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                Some("v1".into()),
            )
            .unwrap();

        cache
            .insert(
                "q2".into(),
                &v2,
                "r2".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                Some("v2".into()),
            )
            .unwrap();

        assert_eq!(cache.len(), 2);
        let removed = cache.invalidate_version("v1");
        assert_eq!(removed, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_expired_entry() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .insert(
                "query".into(),
                &embedding,
                "response".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_millis(1),
                None,
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));
        assert!(cache.get(&embedding, None).is_none());
    }

    #[test]
    fn test_clear() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .insert(
                "query".into(),
                &embedding,
                "response".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cleanup_expired() {
        let cache = create_test_cache();
        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]);

        cache
            .insert(
                "q1".into(),
                &v1,
                "r1".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_millis(1),
                None,
            )
            .unwrap();
        cache
            .insert(
                "q2".into(),
                &v2,
                "r2".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));
        let cleaned = cache.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cleanup_expired_none() {
        let cache = create_test_cache();
        let cleaned = cache.cleanup_expired();
        assert_eq!(cleaned, 0);
    }

    #[test]
    fn test_eviction_candidates() {
        let cache = create_test_cache();
        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]);

        cache
            .insert(
                "q1".into(),
                &v1,
                "r1".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();
        cache
            .insert(
                "q2".into(),
                &v2,
                "r2".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Access the first entry to increase its score
        cache.get(&v1, None);

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_eviction_candidates_limit() {
        let cache = create_test_cache();

        for i in 0..5 {
            let angle = i as f32 * 0.5;
            let v = normalize(&[angle.cos(), angle.sin(), 0.0]);
            cache
                .insert(
                    format!("q{}", i),
                    &v,
                    format!("r{}", i),
                    10,
                    5,
                    "gpt-4".into(),
                    Duration::from_secs(60),
                    None,
                )
                .unwrap();
        }

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_is_empty() {
        let cache = create_test_cache();
        assert!(cache.is_empty());

        let embedding = normalize(&[1.0, 0.0, 0.0]);
        cache
            .insert(
                "query".into(),
                &embedding,
                "response".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_full_error() {
        let mut config = CacheConfig::default();
        config.semantic_capacity = 2;
        config.embedding_dim = 3;
        let stats = Arc::new(CacheStats::new());
        let cache = SemanticCache::new(&config, stats);

        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]);
        let v3 = normalize(&[0.0, 0.0, 1.0]);

        cache
            .insert(
                "q1".into(),
                &v1,
                "r1".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();
        cache
            .insert(
                "q2".into(),
                &v2,
                "r2".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        let result = cache.insert(
            "q3".into(),
            &v3,
            "r3".into(),
            10,
            5,
            "gpt-4".into(),
            Duration::from_secs(60),
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_entry_record_access() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        let id = cache
            .insert(
                "query".into(),
                &embedding,
                "response".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Access multiple times
        cache.get(&embedding, None);
        cache.get(&embedding, None);

        let entry = cache.entries.get(&id).unwrap();
        assert_eq!(entry.access_count, 2);
    }

    #[test]
    fn test_semantic_hit_fields() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .insert(
                "What is 2+2?".into(),
                &embedding,
                "4".into(),
                100,
                50,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        let hit = cache.get(&embedding, None).unwrap();
        assert_eq!(hit.response, "4");
        assert_eq!(hit.query, "What is 2+2?");
        assert_eq!(hit.input_tokens, 100);
        assert_eq!(hit.output_tokens, 50);
        assert_eq!(hit.model, "gpt-4");
        assert!(hit.similarity > 0.99);
    }

    #[test]
    fn test_entry_is_expired() {
        let now = Instant::now();
        let entry = SemanticEntry {
            id: "test".into(),
            query: "q".into(),
            response: "r".into(),
            input_tokens: 10,
            output_tokens: 5,
            model: "m".into(),
            created_at: now,
            expires_at: now - Duration::from_secs(1), // Already expired
            access_count: 0,
            last_accessed: now,
            version: None,
        };
        assert!(entry.is_expired());
    }

    #[test]
    fn test_entry_not_expired() {
        let now = Instant::now();
        let entry = SemanticEntry {
            id: "test".into(),
            query: "q".into(),
            response: "r".into(),
            input_tokens: 10,
            output_tokens: 5,
            model: "m".into(),
            created_at: now,
            expires_at: now + Duration::from_secs(60),
            access_count: 0,
            last_accessed: now,
            version: None,
        };
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_get_miss_when_entry_not_in_dashmap() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        // No entries, should miss
        assert!(cache.get(&embedding, None).is_none());
    }

    #[test]
    fn test_insert_sparse_and_get() {
        let cache = create_test_cache();
        let dense = normalize(&[1.0, 0.0, 0.0]);
        let sparse = SparseVector::from_dense(&dense);

        cache
            .insert_sparse(
                "What is 2+2?".into(),
                &sparse,
                "4".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Search with sparse query
        let hit = cache.get_sparse(&sparse, None).unwrap();
        assert_eq!(hit.response, "4");
        assert_eq!(hit.query, "What is 2+2?");
        assert!(hit.similarity > 0.99);
    }

    #[test]
    fn test_insert_auto_selects_sparse() {
        // Use 10-dim for better sparsity control
        let mut config = CacheConfig::default();
        config.embedding_dim = 10;
        let stats = Arc::new(CacheStats::new());
        let cache = SemanticCache::new(&config, stats);

        // 80% zeros - should be stored as sparse
        let embedding = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];

        let id = cache
            .insert_auto(
                "sparse query".into(),
                &embedding,
                0.7,
                "response".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Verify it's stored as sparse
        let stored = cache.get_embedding(&id);
        assert!(matches!(stored, Some(EmbeddingStorage::Sparse(_))));
    }

    #[test]
    fn test_insert_auto_selects_dense() {
        let cache = create_test_cache();

        // No zeros - should be stored as dense
        let embedding = normalize(&[0.5, 0.5, 0.5]);

        let id = cache
            .insert_auto(
                "dense query".into(),
                &embedding,
                0.7,
                "response".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Verify it's stored as dense
        let stored = cache.get_embedding(&id);
        assert!(matches!(stored, Some(EmbeddingStorage::Dense(_))));
    }

    #[test]
    fn test_memory_stats() {
        let cache = create_test_cache();

        let dense = normalize(&[1.0, 0.0, 0.0]);
        cache
            .insert(
                "q1".into(),
                &dense,
                "r1".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        let stats = cache.memory_stats();
        assert_eq!(stats.dense_count, 1);
        assert!(stats.embedding_bytes > 0);
    }

    #[test]
    fn test_sparse_cache_full() {
        let mut config = CacheConfig::default();
        config.semantic_capacity = 1;
        config.embedding_dim = 3;
        let stats = Arc::new(CacheStats::new());
        let cache = SemanticCache::new(&config, stats);

        let sparse = SparseVector::from_dense(&[1.0, 0.0, 0.0]);

        cache
            .insert_sparse(
                "q1".into(),
                &sparse,
                "r1".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Second insert should fail
        let result = cache.insert_sparse(
            "q2".into(),
            &sparse,
            "r2".into(),
            10,
            5,
            "gpt-4".into(),
            Duration::from_secs(60),
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_mixed_dense_sparse_retrieval() {
        let cache = create_test_cache();

        // Insert dense
        let dense = normalize(&[1.0, 0.0, 0.0]);
        cache
            .insert(
                "dense".into(),
                &dense,
                "dense response".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Insert sparse (similar vector)
        let sparse = SparseVector::from_dense(&normalize(&[0.9, 0.1, 0.0]));
        cache
            .insert_sparse(
                "sparse".into(),
                &sparse,
                "sparse response".into(),
                10,
                5,
                "gpt-4".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Dense query should find both
        let hit = cache.get(&dense, Some(0.5)).unwrap();
        // Should find the most similar (itself)
        assert_eq!(hit.response, "dense response");
    }

    #[test]
    fn test_eviction_uses_lru_strategy() {
        let mut config = CacheConfig::default();
        config.embedding_dim = 3;
        config.eviction_strategy = EvictionStrategy::LRU;
        let stats = Arc::new(CacheStats::new());
        let cache = SemanticCache::new(&config, stats);

        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]);

        cache
            .insert(
                "old".into(),
                &v1,
                "r".into(),
                10,
                10,
                "m".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();
        std::thread::sleep(Duration::from_millis(50));
        cache
            .insert(
                "new".into(),
                &v2,
                "r".into(),
                10,
                10,
                "m".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // Older entry should be first (lower score)
        // Note: We check the score relationship, not the key order since IDs are UUIDs
        assert!(candidates[0].1 < candidates[1].1);
    }

    #[test]
    fn test_eviction_uses_lfu_strategy() {
        let mut config = CacheConfig::default();
        config.embedding_dim = 3;
        config.eviction_strategy = EvictionStrategy::LFU;
        let stats = Arc::new(CacheStats::new());
        let cache = SemanticCache::new(&config, stats);

        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]);

        cache
            .insert(
                "rarely".into(),
                &v1,
                "r".into(),
                10,
                10,
                "m".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();
        cache
            .insert(
                "often".into(),
                &v2,
                "r".into(),
                10,
                10,
                "m".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Access v2 entry multiple times
        cache.get(&v2, None);
        cache.get(&v2, None);
        cache.get(&v2, None);

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // Less frequently accessed should have lower score
        assert!(candidates[0].1 < candidates[1].1);
    }

    #[test]
    fn test_eviction_uses_hybrid_strategy() {
        let mut config = CacheConfig::default();
        config.embedding_dim = 3;
        config.eviction_strategy = EvictionStrategy::Hybrid {
            lru_weight: 40,
            lfu_weight: 30,
            cost_weight: 30,
        };
        config.input_cost_per_1k = 0.01;
        config.output_cost_per_1k = 0.03;
        let stats = Arc::new(CacheStats::new());
        let cache = SemanticCache::new(&config, stats);

        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]);

        // Low value entry
        cache
            .insert(
                "low".into(),
                &v1,
                "x".into(),
                1,
                1,
                "m".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();
        // High value entry
        cache
            .insert(
                "high".into(),
                &v2,
                "x".repeat(100),
                100,
                100,
                "m".into(),
                Duration::from_secs(60),
                None,
            )
            .unwrap();

        // Access high value more
        cache.get(&v2, None);
        cache.get(&v2, None);

        let candidates = cache.eviction_candidates(2);
        assert_eq!(candidates.len(), 2);
        // Lower value entry should have lower score
        assert!(candidates[0].1 < candidates[1].1);
    }
}
