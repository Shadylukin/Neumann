//! Tensor-native LLM response cache - Module 10 of Neumann
//!
//! Tensor-native semantic caching for LLM responses with exact and semantic matching,
//! cost tracking, and background eviction.
//!
//! # Architecture
//!
//! Uses `TensorStore` as its backing store, aligning with the tensor-native
//! paradigm used by `tensor_vault` and `tensor_blob`. Cache entries are stored as
//! `TensorData` with standardized field prefixes.
//!
//! # Cache Layers
//!
//! - **Exact Cache**: O(1) hash-based lookup for identical queries
//! - **Semantic Cache**: O(log n) HNSW-based similarity search
//! - **Embedding Cache**: O(1) cached embeddings for queries
//!
//! # Example
//!
//! ```
//! use tensor_cache::{Cache, CacheConfig};
//!
//! // Configure cache with 3-dimensional embeddings
//! let mut config = CacheConfig::default();
//! config.embedding_dim = 3;
//! let cache = Cache::with_config(config);
//!
//! // Store a response
//! let embedding = vec![0.1, 0.2, 0.3];
//! cache.put("What is 2+2?", &embedding, "4", "gpt-4", None).unwrap();
//!
//! // Look up (tries exact first, then semantic)
//! if let Some(hit) = cache.get("What is 2+2?", Some(&embedding)) {
//!     println!("Cached: {}", hit.response);
//! }
//! ```

#![forbid(unsafe_code)]

mod config;
mod error;
mod eviction;
mod index;
mod stats;
mod tokenizer;

pub use config::{CacheConfig, EvictionStrategy};
pub use error::{CacheError, Result};
pub use eviction::{EvictionManager, EvictionScorer};
pub use stats::{CacheLayer, CacheStats, StatsSnapshot};
pub use tokenizer::{ModelPricing, TokenCounter};

// Re-export geometric types from tensor_store for convenience
pub use tensor_store::{DistanceMetric, SparseVector};

use index::CacheIndex;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

/// Parameters for building a cache entry.
struct EntryParams<'a> {
    layer: CacheLayer,
    response: &'a str,
    embedding: Option<&'a [f32]>,
    input_tokens: usize,
    output_tokens: usize,
    model: &'a str,
    created_at: i64,
    expires_at: i64,
    version: Option<&'a str>,
}

/// Key prefixes for cache entries in `TensorStore`.
mod prefixes {
    pub const EXACT: &str = "_cache:exact:";
    pub const SEMANTIC: &str = "_cache:sem:";
    pub const EMBEDDING: &str = "_cache:emb:";
}

/// Field names for cache entry `TensorData`.
mod fields {
    pub const RESPONSE: &str = "_response";
    pub const EMBEDDING: &str = "_embedding";
    pub const EMBEDDING_DIM: &str = "_embedding_dim";
    pub const INPUT_TOKENS: &str = "_input_tokens";
    pub const OUTPUT_TOKENS: &str = "_output_tokens";
    pub const MODEL: &str = "_model";
    pub const LAYER: &str = "_layer";
    pub const CREATED_AT: &str = "_created_at";
    pub const EXPIRES_AT: &str = "_expires_at";
    pub const ACCESS_COUNT: &str = "_access_count";
    pub const LAST_ACCESS: &str = "_last_access";
    pub const VERSION: &str = "_version";
    pub const SOURCE: &str = "_source";
    pub const CONTENT_HASH: &str = "_content_hash";
}

/// Result of a successful cache lookup.
#[derive(Debug, Clone)]
pub struct CacheHit {
    pub response: String,
    pub layer: CacheLayer,
    pub similarity: Option<f32>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cost_saved: f64,
    pub metric_used: Option<DistanceMetric>,
}

/// LLM response cache with tensor-native storage.
///
/// Uses `TensorStore` as the unified backing store, following the same
/// pattern as `tensor_vault` and `tensor_blob`.
pub struct Cache {
    store: TensorStore,
    index: CacheIndex,
    stats: Arc<CacheStats>,
    config: CacheConfig,
}

impl Cache {
    /// Create a new cache with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default())
    }

    /// Create a cache with custom configuration.
    #[must_use]
    pub fn with_config(config: CacheConfig) -> Self {
        let stats = Arc::new(CacheStats::new());
        let store = TensorStore::new();
        let index = CacheIndex::new(config.embedding_dim, config.distance_metric.clone());

        Self {
            store,
            index,
            stats,
            config,
        }
    }

    /// Create a cache with a shared `TensorStore` (for integration with other engines).
    #[must_use]
    pub fn with_store(store: TensorStore, config: CacheConfig) -> Self {
        let stats = Arc::new(CacheStats::new());
        let index = CacheIndex::new(config.embedding_dim, config.distance_metric.clone());

        Self {
            store,
            index,
            stats,
            config,
        }
    }

    /// Look up a cached response.
    ///
    /// Tries exact match first, then semantic similarity if embedding is provided.
    pub fn get(&self, prompt: &str, embedding: Option<&[f32]>) -> Option<CacheHit> {
        self.get_with_metric(prompt, embedding, None)
    }

    /// Look up a cached response with a specific distance metric.
    pub fn get_with_metric(
        &self,
        prompt: &str,
        embedding: Option<&[f32]>,
        metric: Option<&DistanceMetric>,
    ) -> Option<CacheHit> {
        // Try exact match first
        if let Some(hit) = self.try_exact_match(prompt) {
            return Some(hit);
        }

        // Record exact miss
        self.stats.record_miss(CacheLayer::Exact);

        // Try semantic match if embedding provided
        if let Some(emb) = embedding {
            if let Some(hit) = self.try_semantic_match(emb, metric) {
                return Some(hit);
            }
            self.stats.record_miss(CacheLayer::Semantic);
        }

        None
    }

    fn try_exact_match(&self, prompt: &str) -> Option<CacheHit> {
        let key = Self::exact_key(prompt);
        let data = self.store.get(&key).ok()?;

        if Self::is_expired(&data) {
            return None;
        }

        // Access tracking handled by CacheRing internally
        self.stats.record_hit(CacheLayer::Exact);

        let response = Self::get_string_field(&data, fields::RESPONSE)?;
        let input_tokens = Self::get_usize_field(&data, fields::INPUT_TOKENS);
        let output_tokens = Self::get_usize_field(&data, fields::OUTPUT_TOKENS);

        self.stats.record_tokens_saved(input_tokens, output_tokens);

        Some(CacheHit {
            response,
            layer: CacheLayer::Exact,
            similarity: None,
            input_tokens,
            output_tokens,
            cost_saved: 0.0, // Computed lazily if needed
            metric_used: None,
        })
    }

    fn try_semantic_match(
        &self,
        embedding: &[f32],
        metric: Option<&DistanceMetric>,
    ) -> Option<CacheHit> {
        let threshold = self.config.semantic_threshold;

        let selected_metric = metric.map_or_else(|| self.select_metric(embedding), Clone::clone);

        let results = self
            .index
            .search_with_metric(embedding, 1, threshold, &selected_metric)
            .ok()?;
        let result = results.into_iter().next()?;
        let sem_key = result.key;
        let similarity = result.similarity;

        let data = self.store.get(&sem_key).ok()?;

        if Self::is_expired(&data) {
            return None;
        }

        // Access tracking handled by CacheRing internally
        self.stats.record_hit(CacheLayer::Semantic);

        let response = Self::get_string_field(&data, fields::RESPONSE)?;
        let input_tokens = Self::get_usize_field(&data, fields::INPUT_TOKENS);
        let output_tokens = Self::get_usize_field(&data, fields::OUTPUT_TOKENS);

        self.stats.record_tokens_saved(input_tokens, output_tokens);

        Some(CacheHit {
            response,
            layer: CacheLayer::Semantic,
            similarity: Some(similarity),
            input_tokens,
            output_tokens,
            cost_saved: 0.0, // Computed lazily if needed
            metric_used: Some(selected_metric),
        })
    }

    /// Select distance metric based on embedding sparsity.
    fn select_metric(&self, embedding: &[f32]) -> DistanceMetric {
        if !self.config.auto_select_metric {
            return self.config.distance_metric.clone();
        }

        let sparse = SparseVector::from_dense(embedding);
        if sparse.sparsity() >= self.config.sparsity_metric_threshold {
            DistanceMetric::Jaccard
        } else {
            self.config.distance_metric.clone()
        }
    }

    /// Store a response in the cache.
    ///
    /// # Errors
    ///
    /// Returns an error if insertion fails due to dimension mismatch or capacity.
    pub fn put(
        &self,
        prompt: &str,
        embedding: &[f32],
        response: &str,
        model: &str,
        ttl: Option<Duration>,
    ) -> Result<()> {
        let input_tokens = TokenCounter::count(prompt);
        let output_tokens = TokenCounter::count(response);
        let ttl = ttl.unwrap_or(self.config.default_ttl);
        let now = Self::now_millis();
        let expires_at = now + Self::i64_from_u128(ttl.as_millis());

        let exact_count = self.stats.size(CacheLayer::Exact);
        if exact_count >= self.config.exact_capacity {
            return Err(CacheError::CacheFull {
                current: exact_count,
                capacity: self.config.exact_capacity,
            });
        }

        let exact_key = Self::exact_key(prompt);
        let exact_data = Self::build_entry(&EntryParams {
            layer: CacheLayer::Exact,
            response,
            embedding: None,
            input_tokens,
            output_tokens,
            model,
            created_at: now,
            expires_at,
            version: None,
        });
        self.store
            .put(&exact_key, exact_data)
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
        self.stats.increment_size(CacheLayer::Exact);

        let sem_count = self.stats.size(CacheLayer::Semantic);
        if sem_count >= self.config.semantic_capacity {
            return Err(CacheError::CacheFull {
                current: sem_count,
                capacity: self.config.semantic_capacity,
            });
        }

        let sem_key = Self::semantic_key();
        let sem_data = Self::build_entry(&EntryParams {
            layer: CacheLayer::Semantic,
            response,
            embedding: Some(embedding),
            input_tokens,
            output_tokens,
            model,
            created_at: now,
            expires_at,
            version: None,
        });
        self.store
            .put(&sem_key, sem_data)
            .map_err(|e| CacheError::StorageError(e.to_string()))?;

        self.index.insert(&sem_key, embedding)?;
        self.stats.increment_size(CacheLayer::Semantic);

        Ok(())
    }

    /// Get a cached embedding.
    pub fn get_embedding(&self, source: &str, content: &str) -> Option<Vec<f32>> {
        let key = Self::embedding_key(source, content);
        let data = self.store.get(&key).ok()?;

        if Self::is_expired(&data) {
            return None;
        }

        Self::get_embedding_field(&data)
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
        let emb_count = self.stats.size(CacheLayer::Embedding);
        if emb_count >= self.config.embedding_capacity {
            return Err(CacheError::CacheFull {
                current: emb_count,
                capacity: self.config.embedding_capacity,
            });
        }

        let key = Self::embedding_key(source, content);
        let now = Self::now_millis();
        let expires_at = now + Self::i64_from_u128(self.config.default_ttl.as_millis());

        let mut data = TensorData::new();
        data.set(fields::LAYER, scalar_string("embedding"));
        data.set(fields::SOURCE, scalar_string(source));
        data.set(
            fields::CONTENT_HASH,
            TensorValue::Scalar(ScalarValue::Int(Self::i64_from_u64(Self::hash_content(
                content,
            )))),
        );
        data.set(
            fields::EMBEDDING_DIM,
            TensorValue::Scalar(ScalarValue::Int(Self::i64_from_usize(embedding.len()))),
        );
        data.set(fields::EMBEDDING, TensorValue::Vector(embedding));
        data.set(fields::MODEL, scalar_string(model));
        data.set(
            fields::CREATED_AT,
            TensorValue::Scalar(ScalarValue::Int(now)),
        );
        data.set(
            fields::EXPIRES_AT,
            TensorValue::Scalar(ScalarValue::Int(expires_at)),
        );
        data.set(
            fields::ACCESS_COUNT,
            TensorValue::Scalar(ScalarValue::Int(0)),
        );
        data.set(
            fields::LAST_ACCESS,
            TensorValue::Scalar(ScalarValue::Int(now)),
        );

        self.store
            .put(&key, data)
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
        self.stats.increment_size(CacheLayer::Embedding);

        Ok(())
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
        if let Some(emb) = self.get_embedding(source, content) {
            return Ok(emb);
        }

        let embedding = compute()?;
        self.put_embedding(source, content, embedding.clone(), model)?;
        Ok(embedding)
    }

    /// Simple key-value get for CLI interface.
    pub fn get_simple(&self, key: &str) -> Option<String> {
        let cache_key = Self::exact_key(key);
        let data = self.store.get(&cache_key).ok()?;

        if Self::is_expired(&data) {
            return None;
        }

        Self::get_string_field(&data, fields::RESPONSE)
    }

    /// Simple key-value put for CLI interface.
    pub fn put_simple(&self, key: &str, value: &str) {
        let cache_key = Self::exact_key(key);
        let now = Self::now_millis();
        let expires_at = now + Self::i64_from_u128(self.config.default_ttl.as_millis());

        let data = Self::build_entry(&EntryParams {
            layer: CacheLayer::Exact,
            response: value,
            embedding: None,
            input_tokens: 0,
            output_tokens: 0,
            model: "cli",
            created_at: now,
            expires_at,
            version: None,
        });

        let _ = self.store.put(&cache_key, data);
    }

    /// Invalidate entries by prompt.
    #[must_use]
    pub fn invalidate(&self, prompt: &str) -> bool {
        let key = Self::exact_key(prompt);
        if self.store.delete(&key).is_ok() {
            self.stats.decrement_size(CacheLayer::Exact);
            true
        } else {
            false
        }
    }

    /// Invalidate semantic cache entries by version.
    #[must_use]
    pub fn invalidate_version(&self, version: &str) -> usize {
        let keys = self.store.scan(prefixes::SEMANTIC);
        let mut removed = 0;

        for key in keys {
            if let Ok(data) = self.store.get(&key) {
                if let Some(v) = Self::get_string_field(&data, fields::VERSION) {
                    if v == version && self.store.delete(&key).is_ok() {
                        let _ = self.index.remove(&key);
                        self.stats.decrement_size(CacheLayer::Semantic);
                        removed += 1;
                    }
                }
            }
        }

        removed
    }

    /// Invalidate embeddings for a source.
    #[must_use]
    pub fn invalidate_embeddings(&self, source: &str) -> usize {
        let prefix = format!("{}{}:", prefixes::EMBEDDING, source);
        let keys = self.store.scan(&prefix);
        let mut removed = 0;

        for key in keys {
            if self.store.delete(&key).is_ok() {
                self.stats.decrement_size(CacheLayer::Embedding);
                removed += 1;
            }
        }

        removed
    }

    /// Get cache statistics.
    #[must_use]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get a statistics snapshot.
    #[must_use]
    pub fn stats_snapshot(&self) -> StatsSnapshot {
        self.stats.snapshot()
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Manually run eviction.
    ///
    /// Delegates to `CacheRing`'s efficient eviction which uses the configured
    /// eviction strategy (LRU, LFU, Cost, Hybrid) with O(n log n) sorting.
    #[must_use]
    pub fn evict(&self, count: usize) -> usize {
        let evicted = self.store.evict_cache(count);
        if evicted > 0 {
            self.stats.record_eviction(evicted);
        }
        evicted
    }

    /// Clean up expired entries.
    #[must_use]
    pub fn cleanup_expired(&self) -> usize {
        let mut cleaned = 0;

        // Clean exact cache
        for key in self.store.scan(prefixes::EXACT) {
            if let Ok(data) = self.store.get(&key) {
                if Self::is_expired(&data) && self.store.delete(&key).is_ok() {
                    self.stats.decrement_size(CacheLayer::Exact);
                    cleaned += 1;
                }
            }
        }

        // Clean semantic cache
        for key in self.store.scan(prefixes::SEMANTIC) {
            if let Ok(data) = self.store.get(&key) {
                if Self::is_expired(&data) && self.store.delete(&key).is_ok() {
                    let _ = self.index.remove(&key);
                    self.stats.decrement_size(CacheLayer::Semantic);
                    cleaned += 1;
                }
            }
        }

        // Clean embedding cache
        for key in self.store.scan(prefixes::EMBEDDING) {
            if let Ok(data) = self.store.get(&key) {
                if Self::is_expired(&data) && self.store.delete(&key).is_ok() {
                    self.stats.decrement_size(CacheLayer::Embedding);
                    cleaned += 1;
                }
            }
        }

        // Record all expirations at once
        if cleaned > 0 {
            self.stats.record_expiration(cleaned);
        }

        cleaned
    }

    /// Clear all cache entries.
    pub fn clear(&self) {
        // Delete all cache entries
        for key in self.store.scan(prefixes::EXACT) {
            let _ = self.store.delete(&key);
        }
        for key in self.store.scan(prefixes::SEMANTIC) {
            let _ = self.store.delete(&key);
        }
        for key in self.store.scan(prefixes::EMBEDDING) {
            let _ = self.store.delete(&key);
        }

        // Clear HNSW index
        self.index.clear();

        // Reset stats sizes
        self.stats.set_size(CacheLayer::Exact, 0);
        self.stats.set_size(CacheLayer::Semantic, 0);
        self.stats.set_size(CacheLayer::Embedding, 0);
    }

    /// Get the total number of cached entries across all layers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stats.total_entries()
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // --- Helper functions ---

    fn exact_key(prompt: &str) -> String {
        let hash = Self::hash_content(prompt);
        format!("{}{:016x}", prefixes::EXACT, hash)
    }

    fn semantic_key() -> String {
        format!("{}{}", prefixes::SEMANTIC, uuid::Uuid::new_v4())
    }

    fn embedding_key(source: &str, content: &str) -> String {
        let hash = Self::hash_content(content);
        format!("{}{}:{:016x}", prefixes::EMBEDDING, source, hash)
    }

    fn hash_content(content: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    fn now_millis() -> i64 {
        Self::i64_from_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
        )
    }

    fn is_expired(data: &TensorData) -> bool {
        let expires_at = Self::get_i64_field(data, fields::EXPIRES_AT);
        if expires_at == 0 {
            return false; // No expiration set
        }
        Self::now_millis() > expires_at
    }

    fn build_entry(params: &EntryParams<'_>) -> TensorData {
        let mut data = TensorData::new();

        let layer_str = match params.layer {
            CacheLayer::Exact => "exact",
            CacheLayer::Semantic => "semantic",
            CacheLayer::Embedding => "embedding",
        };
        data.set(fields::LAYER, scalar_string(layer_str));
        data.set(fields::RESPONSE, scalar_string(params.response));
        data.set(
            fields::INPUT_TOKENS,
            TensorValue::Scalar(ScalarValue::Int(Self::i64_from_usize(params.input_tokens))),
        );
        data.set(
            fields::OUTPUT_TOKENS,
            TensorValue::Scalar(ScalarValue::Int(Self::i64_from_usize(params.output_tokens))),
        );
        data.set(fields::MODEL, scalar_string(params.model));
        data.set(
            fields::CREATED_AT,
            TensorValue::Scalar(ScalarValue::Int(params.created_at)),
        );
        data.set(
            fields::EXPIRES_AT,
            TensorValue::Scalar(ScalarValue::Int(params.expires_at)),
        );
        data.set(
            fields::ACCESS_COUNT,
            TensorValue::Scalar(ScalarValue::Int(0)),
        );
        data.set(
            fields::LAST_ACCESS,
            TensorValue::Scalar(ScalarValue::Int(params.created_at)),
        );

        if let Some(emb) = params.embedding {
            data.set(
                fields::EMBEDDING_DIM,
                TensorValue::Scalar(ScalarValue::Int(Self::i64_from_usize(emb.len()))),
            );
            data.set(fields::EMBEDDING, TensorValue::Vector(emb.to_vec()));
        }

        if let Some(v) = params.version {
            data.set(fields::VERSION, scalar_string(v));
        }

        data
    }

    fn get_string_field(data: &TensorData, field: &str) -> Option<String> {
        match data.get(field) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => Some(s.clone()),
            _ => None,
        }
    }

    fn get_i64_field(data: &TensorData, field: &str) -> i64 {
        match data.get(field) {
            Some(TensorValue::Scalar(ScalarValue::Int(i))) => *i,
            _ => 0,
        }
    }

    fn get_usize_field(data: &TensorData, field: &str) -> usize {
        let val = Self::get_i64_field(data, field);
        usize::try_from(val).unwrap_or(0)
    }

    fn get_embedding_field(data: &TensorData) -> Option<Vec<f32>> {
        match data.get(fields::EMBEDDING) {
            Some(TensorValue::Vector(v)) => Some(v.clone()),
            Some(TensorValue::Sparse(s)) => Some(s.to_dense()),
            _ => None,
        }
    }

    fn i64_from_usize(val: usize) -> i64 {
        i64::try_from(val).unwrap_or(i64::MAX)
    }

    fn i64_from_u64(val: u64) -> i64 {
        i64::try_from(val).unwrap_or(i64::MAX)
    }

    fn i64_from_u128(val: u128) -> i64 {
        i64::try_from(val).unwrap_or(i64::MAX)
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

fn scalar_string(s: &str) -> TensorValue {
    TensorValue::Scalar(ScalarValue::String(s.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_store_scan_works() {
        let store = TensorStore::new();
        store.put("_test:a", TensorData::new()).unwrap();
        store.put("_test:b", TensorData::new()).unwrap();
        store.put("_other:c", TensorData::new()).unwrap();

        let keys = store.scan("_test:");
        assert_eq!(keys.len(), 2, "Expected 2 keys with prefix _test:");
    }

    #[test]
    fn test_cache_store_scan_works() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        // Put some entries
        cache
            .put("prompt1", &embedding, "response1", "gpt-4", None)
            .unwrap();
        cache
            .put("prompt2", &embedding, "response2", "gpt-4", None)
            .unwrap();

        // Check stats-based len
        assert_eq!(cache.len(), 4, "Expected 4 entries (2 exact + 2 semantic)");

        // Check store scan
        let exact_count = cache.store.scan(prefixes::EXACT).len();
        let semantic_count = cache.store.scan(prefixes::SEMANTIC).len();

        eprintln!("exact_count from scan: {}", exact_count);
        eprintln!("semantic_count from scan: {}", semantic_count);

        assert!(exact_count > 0, "Expected scan to find exact entries");
    }

    fn create_test_cache() -> Cache {
        let mut config = CacheConfig::default();
        config.embedding_dim = 3;
        Cache::with_config(config)
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        SparseVector::from_dense(v)
            .normalize()
            .map(|sv| sv.to_dense())
            .unwrap_or_else(|| v.to_vec())
    }

    #[test]
    fn test_put_and_get_exact() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("What is 2+2?", &embedding, "4", "gpt-4", None)
            .unwrap();

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
            .put("What is 2+2?", &embedding, "4", "gpt-4", None)
            .unwrap();

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
            .put("prompt", &embedding, "response", "gpt-4", None)
            .unwrap();
        assert!(cache.get("prompt", None).is_some());

        assert!(cache.invalidate("prompt"));
        assert!(cache.get("prompt", None).is_none());
    }

    #[test]
    fn test_stats() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("prompt", &embedding, "response", "gpt-4", None)
            .unwrap();
        cache.get("prompt", None);
        cache.get("other", None);

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
                    None,
                )
                .unwrap();
        }

        let initial_len = cache.len();
        assert!(initial_len > 0, "Cache should not be empty after puts");

        let evicted = cache.evict(5);
        // Eviction should reduce the cache size
        assert!(
            cache.len() < initial_len || evicted > 0,
            "Expected eviction to reduce size or return > 0"
        );
    }

    #[test]
    fn test_clear() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("prompt", &embedding, "response", "gpt-4", None)
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
    fn test_simple_get_put() {
        let cache = create_test_cache();
        cache.put_simple("key", "value");

        let result = cache.get_simple("key").unwrap();
        assert_eq!(result, "value");
    }

    #[test]
    fn test_get_or_compute_embedding() {
        let cache = create_test_cache();

        let result = cache
            .get_or_compute_embedding("source", "content", "model", || Ok(vec![0.1, 0.2, 0.3]))
            .unwrap();

        assert_eq!(result, vec![0.1, 0.2, 0.3]);

        // Second call should use cached value
        let mut compute_called = false;
        let result2 = cache
            .get_or_compute_embedding("source", "content", "model", || {
                compute_called = true;
                Ok(vec![0.4, 0.5, 0.6])
            })
            .unwrap();

        assert!(!compute_called);
        assert_eq!(result2, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_len_and_is_empty() {
        let cache = create_test_cache();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());

        let embedding = normalize(&[1.0, 0.0, 0.0]);
        cache
            .put("prompt", &embedding, "response", "gpt-4", None)
            .unwrap();

        assert!(cache.len() > 0);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_with_shared_store() {
        let store = TensorStore::new();
        let config = CacheConfig::default();
        let cache = Cache::with_store(store, config);

        assert!(cache.is_empty());
    }
}
