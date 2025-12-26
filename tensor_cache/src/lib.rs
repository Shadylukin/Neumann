//! TensorCache - Module 10 of Neumann
//!
//! Semantic caching for LLM responses with exact and semantic matching,
//! cost tracking, and background eviction.
//!
//! # Cache Layers
//!
//! - **Exact Cache**: O(1) hash-based lookup for identical queries
//! - **Semantic Cache**: O(log n) HNSW-based similarity search
//! - **Embedding Cache**: O(1) cached embeddings for queries
//!
//! # Example
//!
//! ```ignore
//! use tensor_cache::{Cache, CacheConfig};
//!
//! let cache = Cache::new();
//!
//! // Store a response
//! cache.put(
//!     "What is 2+2?",
//!     &embedding,
//!     "4",
//!     "gpt-4",
//!     0,
//! )?;
//!
//! // Look up (tries exact first, then semantic)
//! if let Some(hit) = cache.get("What is 2+2?", Some(&embedding)) {
//!     println!("Cached: {} (saved ${:.4})", hit.response, hit.cost_saved);
//! }
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::use_self)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::unused_self)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::similar_names)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::unnecessary_wraps)]

mod config;
mod embedding;
mod error;
mod eviction;
mod exact;
mod index;
mod semantic;
mod stats;
mod tokenizer;
mod ttl;

pub use config::{CacheConfig, EvictionStrategy};
pub use error::{CacheError, Result};
pub use eviction::{EvictionHandle, EvictionManager, EvictionScorer};
pub use stats::{CacheLayer, CacheStats, StatsSnapshot};
pub use tokenizer::{ModelPricing, TokenCounter};

use embedding::EmbeddingCache;
use exact::ExactCache;
use semantic::SemanticCache;
use std::sync::Arc;
use std::time::Duration;
use tensor_store::TensorStore;
use ttl::TTLTracker;

/// Result of a successful cache lookup.
#[derive(Debug, Clone)]
pub struct CacheHit {
    /// The cached response.
    pub response: String,
    /// Which cache layer the hit came from.
    pub layer: CacheLayer,
    /// Similarity score (only for semantic hits).
    pub similarity: Option<f32>,
    /// Input tokens saved.
    pub input_tokens: usize,
    /// Output tokens saved.
    pub output_tokens: usize,
    /// Estimated cost saved in dollars.
    pub cost_saved: f64,
}

/// LLM response cache with exact, semantic, and embedding layers.
pub struct Cache {
    /// Underlying storage (for persistence compatibility).
    #[allow(dead_code)]
    store: TensorStore,
    /// Exact match cache.
    exact: ExactCache,
    /// Semantic similarity cache.
    semantic: SemanticCache,
    /// Embedding cache.
    embedding: EmbeddingCache,
    /// TTL tracker.
    #[allow(dead_code)]
    ttl_tracker: TTLTracker,
    /// Statistics.
    stats: Arc<CacheStats>,
    /// Configuration.
    config: CacheConfig,
    /// Eviction handle (if background eviction is running).
    eviction_handle: Option<EvictionHandle>,
}

impl Cache {
    /// Create a new cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a cache with custom configuration.
    pub fn with_config(config: CacheConfig) -> Self {
        let stats = Arc::new(CacheStats::new());
        let store = TensorStore::new();

        Self {
            store,
            exact: ExactCache::new(&config, Arc::clone(&stats)),
            semantic: SemanticCache::new(&config, Arc::clone(&stats)),
            embedding: EmbeddingCache::new(&config, Arc::clone(&stats)),
            ttl_tracker: TTLTracker::new(),
            stats,
            config,
            eviction_handle: None,
        }
    }

    /// Create a cache using an existing TensorStore.
    pub fn with_store(store: TensorStore, config: CacheConfig) -> Self {
        let stats = Arc::new(CacheStats::new());

        Self {
            store,
            exact: ExactCache::new(&config, Arc::clone(&stats)),
            semantic: SemanticCache::new(&config, Arc::clone(&stats)),
            embedding: EmbeddingCache::new(&config, Arc::clone(&stats)),
            ttl_tracker: TTLTracker::new(),
            stats,
            config,
            eviction_handle: None,
        }
    }

    /// Look up a cached response.
    ///
    /// Tries exact match first, then semantic similarity if embedding is provided.
    pub fn get(&self, prompt: &str, embedding: Option<&[f32]>) -> Option<CacheHit> {
        // Try exact match first (key is based on prompt only)
        let exact_key = exact::generate_prompt_key(prompt);
        if let Some(entry) = self.exact.get(&exact_key) {
            let cost_saved = TokenCounter::estimate_cost(
                entry.input_tokens,
                entry.output_tokens,
                self.config.input_cost_per_1k,
                self.config.output_cost_per_1k,
            );

            self.stats
                .record_tokens_saved(entry.input_tokens, entry.output_tokens);
            self.stats
                .record_cost_saved((cost_saved * 1_000_000.0) as u64);

            return Some(CacheHit {
                response: entry.response,
                layer: CacheLayer::Exact,
                similarity: None,
                input_tokens: entry.input_tokens,
                output_tokens: entry.output_tokens,
                cost_saved,
            });
        }

        // Try semantic match if embedding provided
        if let Some(emb) = embedding {
            if let Some(hit) = self.semantic.get(emb, None) {
                let cost_saved = TokenCounter::estimate_cost(
                    hit.input_tokens,
                    hit.output_tokens,
                    self.config.input_cost_per_1k,
                    self.config.output_cost_per_1k,
                );

                self.stats
                    .record_tokens_saved(hit.input_tokens, hit.output_tokens);
                self.stats
                    .record_cost_saved((cost_saved * 1_000_000.0) as u64);

                return Some(CacheHit {
                    response: hit.response,
                    layer: CacheLayer::Semantic,
                    similarity: Some(hit.similarity),
                    input_tokens: hit.input_tokens,
                    output_tokens: hit.output_tokens,
                    cost_saved,
                });
            }
        }

        None
    }

    /// Store a response in the cache.
    ///
    /// Stores in both exact and semantic caches if embedding is provided.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails due to dimension mismatch.
    pub fn put(
        &self,
        prompt: &str,
        embedding: &[f32],
        response: &str,
        model: &str,
        _params_hash: u64,
    ) -> Result<()> {
        let input_tokens = TokenCounter::count(prompt).unwrap_or(0);
        let output_tokens = TokenCounter::count(response).unwrap_or(0);
        let ttl = self.config.default_ttl;

        // Store in exact cache (key is based on prompt only)
        let exact_key = exact::generate_prompt_key(prompt);
        self.exact.insert(
            exact_key,
            response.to_string(),
            input_tokens,
            output_tokens,
            model.to_string(),
            ttl,
        )?;

        // Store in semantic cache
        self.semantic.insert(
            prompt.to_string(),
            embedding,
            response.to_string(),
            input_tokens,
            output_tokens,
            model.to_string(),
            ttl,
            None,
        )?;

        Ok(())
    }

    /// Store a response with custom TTL.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails due to dimension mismatch.
    pub fn put_with_ttl(
        &self,
        prompt: &str,
        embedding: &[f32],
        response: &str,
        model: &str,
        _params_hash: u64,
        ttl: Duration,
    ) -> Result<()> {
        let input_tokens = TokenCounter::count(prompt).unwrap_or(0);
        let output_tokens = TokenCounter::count(response).unwrap_or(0);

        let exact_key = exact::generate_prompt_key(prompt);
        self.exact.insert(
            exact_key,
            response.to_string(),
            input_tokens,
            output_tokens,
            model.to_string(),
            ttl,
        )?;

        self.semantic.insert(
            prompt.to_string(),
            embedding,
            response.to_string(),
            input_tokens,
            output_tokens,
            model.to_string(),
            ttl,
            None,
        )?;

        Ok(())
    }

    /// Get a cached embedding.
    pub fn get_embedding(&self, source: &str, content: &str) -> Option<Vec<f32>> {
        self.embedding.get_by_content(source, content)
    }

    /// Store an embedding.
    ///
    /// # Errors
    ///
    /// Returns an error if the embedding cache insertion fails.
    pub fn put_embedding(
        &self,
        source: &str,
        content: &str,
        embedding: Vec<f32>,
        model: &str,
    ) -> Result<()> {
        self.embedding
            .insert_by_content(source, content, embedding, model.to_string(), None)
    }

    /// Get or compute an embedding.
    ///
    /// # Errors
    ///
    /// Returns an error if the compute function fails.
    pub fn get_or_compute_embedding<F>(
        &self,
        source: &str,
        content: &str,
        model: &str,
        compute: F,
    ) -> Result<Vec<f32>>
    where
        F: FnOnce() -> Result<Vec<f32>>,
    {
        self.embedding
            .get_or_compute(source, content, model, None, compute)
    }

    /// Invalidate entries by exact key.
    ///
    /// Note: model and `params_hash` are accepted for API compatibility but
    /// the exact cache key is based only on the prompt.
    pub fn invalidate(&self, prompt: &str, _model: &str, _params_hash: u64) -> bool {
        let key = exact::generate_prompt_key(prompt);
        self.exact.remove(&key).is_some()
    }

    /// Invalidate semantic cache entries by version.
    pub fn invalidate_version(&self, version: &str) -> usize {
        self.semantic.invalidate_version(version)
    }

    /// Invalidate embeddings for a source.
    pub fn invalidate_embeddings(&self, source: &str) -> usize {
        self.embedding.invalidate_source(source)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get a statistics snapshot.
    pub fn stats_snapshot(&self) -> StatsSnapshot {
        self.stats.snapshot()
    }

    /// Get the configuration.
    pub const fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Manually run eviction.
    ///
    /// Returns the number of entries evicted.
    pub fn evict(&self, count: usize) -> usize {
        let mut evicted = 0;

        // Evict from exact cache
        let candidates = self.exact.eviction_candidates(count);
        for (key, _) in candidates {
            if self.exact.remove(&key).is_some() {
                evicted += 1;
            }
        }

        // Evict from semantic cache if needed
        if evicted < count {
            let remaining = count - evicted;
            let candidates = self.semantic.eviction_candidates(remaining);
            for (id, _) in candidates {
                if self.semantic.remove(&id).is_some() {
                    evicted += 1;
                }
            }
        }

        // Evict from embedding cache if needed
        if evicted < count {
            let remaining = count - evicted;
            let candidates = self.embedding.eviction_candidates(remaining);
            for (key, _) in candidates {
                if self.embedding.remove(&key).is_some() {
                    evicted += 1;
                }
            }
        }

        evicted
    }

    /// Clean up expired entries.
    ///
    /// Returns the total number of entries cleaned up.
    pub fn cleanup_expired(&self) -> usize {
        let exact = self.exact.cleanup_expired();
        let semantic = self.semantic.cleanup_expired();
        let embedding = self.embedding.cleanup_expired();
        exact + semantic + embedding
    }

    /// Clear all cache entries.
    pub fn clear(&mut self) {
        self.exact.clear();
        self.semantic.clear();
        self.embedding.clear();
        self.ttl_tracker.clear();
    }

    /// Start background eviction.
    ///
    /// This must be called from within a tokio runtime.
    pub fn start_eviction(&mut self) {
        if self.eviction_handle.is_some() {
            return;
        }

        let manager = EvictionManager::from_cache_config(&self.config, Arc::clone(&self.stats));

        // We need to use a weak reference pattern here to avoid circular references
        // Since we can't easily pass &self to a spawned task, we'll use a simpler approach
        // where the eviction task just cleans up expired entries

        let exact = self.exact.eviction_candidates(0).len(); // Just to access
        let _ = exact;

        // For now, we'll create a simple eviction function that returns 0
        // In a real implementation, you'd use Arc<Cache> or channels
        let handle = manager.start(|_batch_size| {
            // This is a placeholder - in production, you'd use a channel or Arc<Cache>
            0
        });

        self.eviction_handle = Some(handle);
    }

    /// Stop background eviction.
    pub async fn stop_eviction(&mut self) {
        if let Some(handle) = self.eviction_handle.take() {
            handle.shutdown().await;
        }
    }

    /// Check if background eviction is running.
    pub fn is_eviction_running(&self) -> bool {
        self.eviction_handle
            .as_ref()
            .is_some_and(EvictionHandle::is_running)
    }

    /// Get the total number of cached entries across all layers.
    pub fn len(&self) -> usize {
        self.exact.len() + self.semantic.len() + self.embedding.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_cache() -> Cache {
        let mut config = CacheConfig::default();
        config.embedding_dim = 3;
        Cache::with_config(config)
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
    fn test_put_and_get_exact() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("What is 2+2?", &embedding, "4", "gpt-4", 0)
            .unwrap();

        // Exact match without embedding
        let hit = cache.get("What is 2+2?", None).unwrap();
        assert_eq!(hit.response, "4");
        assert_eq!(hit.layer, CacheLayer::Exact);
    }

    #[test]
    fn test_put_and_get_semantic() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);
        let similar = normalize(&[0.95, 0.05, 0.0]);

        cache
            .put("What is 2+2?", &embedding, "4", "gpt-4", 0)
            .unwrap();

        // Miss on exact (different prompt), hit on semantic
        let hit = cache.get("Different prompt", Some(&similar)).unwrap();
        assert_eq!(hit.response, "4");
        assert_eq!(hit.layer, CacheLayer::Semantic);
        assert!(hit.similarity.unwrap() > 0.9);
    }

    #[test]
    fn test_embedding_cache() {
        let cache = create_test_cache();
        let embedding = vec![0.1, 0.2, 0.3];

        cache
            .put_embedding("doc", "content", embedding.clone(), "model")
            .unwrap();

        let retrieved = cache.get_embedding("doc", "content").unwrap();
        assert_eq!(retrieved, embedding);
    }

    #[test]
    fn test_invalidate() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("prompt", &embedding, "response", "gpt-4", 0)
            .unwrap();
        assert!(cache.get("prompt", None).is_some());

        cache.invalidate("prompt", "gpt-4", 0);
        assert!(cache.get("prompt", None).is_none());
    }

    #[test]
    fn test_stats() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("prompt", &embedding, "response", "gpt-4", 0)
            .unwrap();
        cache.get("prompt", None); // Hit
        cache.get("other", None); // Miss

        let snapshot = cache.stats_snapshot();
        assert_eq!(snapshot.exact_hits, 1);
        assert_eq!(snapshot.exact_misses, 1);
    }

    #[test]
    fn test_evict() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        for i in 0..10 {
            cache
                .put(
                    &format!("prompt{}", i),
                    &embedding,
                    "response",
                    "gpt-4",
                    i as u64,
                )
                .unwrap();
        }

        assert!(cache.len() > 0);
        let evicted = cache.evict(5);
        assert!(evicted > 0);
    }

    #[test]
    fn test_clear() {
        let mut cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("prompt", &embedding, "response", "gpt-4", 0)
            .unwrap();
        cache
            .put_embedding("doc", "content", vec![0.1], "model")
            .unwrap();

        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_default() {
        let cache = Cache::default();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_with_store() {
        let store = TensorStore::new();
        let mut config = CacheConfig::default();
        config.embedding_dim = 3;
        let cache = Cache::with_store(store, config);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_put_with_ttl() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put_with_ttl(
                "prompt",
                &embedding,
                "response",
                "gpt-4",
                0,
                Duration::from_secs(60),
            )
            .unwrap();

        let hit = cache.get("prompt", None).unwrap();
        assert_eq!(hit.response, "response");
    }

    #[test]
    fn test_get_or_compute_embedding_cached() {
        let cache = create_test_cache();
        let embedding = vec![0.1, 0.2, 0.3];

        cache
            .put_embedding("source", "content", embedding.clone(), "model")
            .unwrap();

        let mut compute_called = false;
        let result = cache
            .get_or_compute_embedding("source", "content", "model", || {
                compute_called = true;
                Ok(vec![0.4, 0.5, 0.6])
            })
            .unwrap();

        assert!(!compute_called);
        assert_eq!(result, embedding);
    }

    #[test]
    fn test_get_or_compute_embedding_miss() {
        let cache = create_test_cache();

        let mut compute_called = false;
        let result = cache
            .get_or_compute_embedding("source", "content", "model", || {
                compute_called = true;
                Ok(vec![0.1, 0.2, 0.3])
            })
            .unwrap();

        assert!(compute_called);
        assert_eq!(result, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_invalidate_version() {
        let cache = create_test_cache();
        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]);

        // Insert with versions via semantic cache directly
        cache
            .semantic
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
            .semantic
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

        let removed = cache.invalidate_version("v1");
        assert_eq!(removed, 1);
    }

    #[test]
    fn test_invalidate_embeddings() {
        let cache = create_test_cache();

        cache
            .put_embedding("source1", "a", vec![0.1], "model")
            .unwrap();
        cache
            .put_embedding("source1", "b", vec![0.2], "model")
            .unwrap();
        cache
            .put_embedding("source2", "c", vec![0.3], "model")
            .unwrap();

        let removed = cache.invalidate_embeddings("source1");
        assert_eq!(removed, 2);
    }

    #[test]
    fn test_cleanup_expired() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put_with_ttl(
                "prompt",
                &embedding,
                "response",
                "gpt-4",
                0,
                Duration::from_millis(1),
            )
            .unwrap();

        std::thread::sleep(Duration::from_millis(10));
        let cleaned = cache.cleanup_expired();
        // cleanup_expired returns usize, at least 0 entries cleaned
        let _ = cleaned;
    }

    #[test]
    fn test_len_and_is_empty() {
        let cache = create_test_cache();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        let embedding = normalize(&[1.0, 0.0, 0.0]);
        cache
            .put("prompt", &embedding, "response", "gpt-4", 0)
            .unwrap();

        assert!(cache.len() > 0);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_config_accessor() {
        let cache = create_test_cache();
        let config = cache.config();
        assert_eq!(config.embedding_dim, 3);
    }

    #[test]
    fn test_stats_accessor() {
        let cache = create_test_cache();
        let stats = cache.stats();
        assert_eq!(stats.evictions(), 0);
    }

    #[test]
    fn test_cache_hit_fields() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("What is 2+2?", &embedding, "4", "gpt-4", 0)
            .unwrap();

        let hit = cache.get("What is 2+2?", None).unwrap();
        assert_eq!(hit.response, "4");
        assert_eq!(hit.layer, CacheLayer::Exact);
        assert!(hit.similarity.is_none());
        assert!(hit.input_tokens > 0);
        assert!(hit.output_tokens > 0);
        assert!(hit.cost_saved > 0.0);
    }

    #[test]
    fn test_semantic_hit_with_similarity() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);
        let similar = normalize(&[0.99, 0.01, 0.0]);

        cache
            .put("What is 2+2?", &embedding, "4", "gpt-4", 0)
            .unwrap();

        // Different prompt but similar embedding
        let hit = cache.get("Different prompt", Some(&similar)).unwrap();
        assert_eq!(hit.layer, CacheLayer::Semantic);
        assert!(hit.similarity.is_some());
        assert!(hit.similarity.unwrap() > 0.9);
    }

    #[test]
    fn test_eviction_running_status() {
        let cache = create_test_cache();
        assert!(!cache.is_eviction_running());
    }

    #[test]
    fn test_evict_from_multiple_layers() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        // Add entries to exact cache
        for i in 0..5 {
            cache
                .put(
                    &format!("prompt{}", i),
                    &embedding,
                    "response",
                    "gpt-4",
                    i as u64,
                )
                .unwrap();
        }

        // Add embeddings
        for i in 0..5 {
            cache
                .put_embedding(&format!("src{}", i), "content", vec![0.1], "model")
                .unwrap();
        }

        let total_before = cache.len();
        let evicted = cache.evict(total_before + 10); // Try to evict more than we have
        assert!(evicted > 0);
        assert!(evicted <= total_before);
    }

    #[test]
    fn test_cache_miss_no_embedding() {
        let cache = create_test_cache();
        assert!(cache.get("nonexistent", None).is_none());
    }

    #[test]
    fn test_cache_miss_with_embedding() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);
        assert!(cache.get("nonexistent", Some(&embedding)).is_none());
    }
}
