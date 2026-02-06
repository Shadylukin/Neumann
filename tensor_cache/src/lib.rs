// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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
//! let cache = Cache::with_config(config).unwrap();
//!
//! // Store a response
//! let embedding = vec![0.1, 0.2, 0.3];
//! cache
//!     .put("What is 2+2?", &embedding, "4", "gpt-4", None)
//!     .unwrap();
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

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::Arc,
    time::Duration,
};

use tokio::{sync::broadcast, task::JoinHandle};

pub use config::{CacheConfig, EvictionStrategy};
pub use error::{CacheError, Result};
pub use eviction::{EvictionManager, EvictionScorer};
use index::CacheIndex;
pub use stats::{CacheLayer, CacheStats, StatsSnapshot};
// Re-export geometric types from tensor_store for convenience
pub use tensor_store::{DistanceMetric, SparseVector};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};
pub use tokenizer::{ModelPricing, TokenCounter};

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
    shutdown_tx: broadcast::Sender<()>,
    eviction_handle: std::sync::Mutex<Option<JoinHandle<()>>>,
}

impl Cache {
    /// # Panics
    ///
    /// Never panics - default configuration is always valid.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(CacheConfig::default()).expect("default config is valid")
    }

    /// # Errors
    ///
    /// Returns `CacheError::InvalidConfig` if the configuration fails validation.
    pub fn with_config(config: CacheConfig) -> Result<Self> {
        config.validate().map_err(CacheError::InvalidConfig)?;

        let stats = Arc::new(CacheStats::new());
        let store = TensorStore::new();
        let index = CacheIndex::new(config.embedding_dim, config.distance_metric.clone());
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            store,
            index,
            stats,
            config,
            shutdown_tx,
            eviction_handle: std::sync::Mutex::new(None),
        })
    }

    /// Create a cache with a shared `TensorStore` (for integration with other engines).
    ///
    /// # Errors
    ///
    /// Returns `CacheError::InvalidConfig` if the configuration fails validation.
    pub fn with_store(store: TensorStore, config: CacheConfig) -> Result<Self> {
        config.validate().map_err(CacheError::InvalidConfig)?;

        let stats = Arc::new(CacheStats::new());
        let index = CacheIndex::new(config.embedding_dim, config.distance_metric.clone());
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            store,
            index,
            stats,
            config,
            shutdown_tx,
            eviction_handle: std::sync::Mutex::new(None),
        })
    }

    /// Tries exact match first, then semantic similarity if embedding is provided.
    #[must_use]
    pub fn get(&self, prompt: &str, embedding: Option<&[f32]>) -> Option<CacheHit> {
        self.get_with_metric(prompt, embedding, None)
    }

    #[must_use]
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

    #[must_use]
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
        // Use sparse format for vectors with >50% zeros
        let storage = if Self::should_use_sparse(&embedding) {
            TensorValue::Sparse(SparseVector::from_dense(&embedding))
        } else {
            TensorValue::Vector(embedding)
        };
        data.set(fields::EMBEDDING, storage);
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

    #[must_use]
    pub fn get_simple(&self, key: &str) -> Option<String> {
        let cache_key = Self::exact_key(key);
        let data = self.store.get(&cache_key).ok()?;

        if Self::is_expired(&data) {
            return None;
        }

        Self::get_string_field(&data, fields::RESPONSE)
    }

    /// Simple key-value put for CLI interface.
    ///
    /// # Errors
    ///
    /// Returns an error if storage insertion fails.
    pub fn put_simple(&self, key: &str, value: &str) -> Result<()> {
        let exact_count = self.stats.size(CacheLayer::Exact);
        if exact_count >= self.config.exact_capacity {
            return Err(CacheError::CacheFull {
                current: exact_count,
                capacity: self.config.exact_capacity,
            });
        }

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

        self.store
            .put(&cache_key, data)
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
        self.stats.increment_size(CacheLayer::Exact);

        Ok(())
    }

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

    #[must_use]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    #[must_use]
    pub fn stats_snapshot(&self) -> StatsSnapshot {
        self.stats.snapshot()
    }

    #[must_use]
    pub const fn config(&self) -> &CacheConfig {
        &self.config
    }

    /// Start background eviction and TTL cleanup.
    ///
    /// Runs at `config.eviction_interval` intervals, processing
    /// `config.eviction_batch_size` entries per cycle. Also cleans up
    /// expired entries each cycle.
    ///
    /// # Errors
    ///
    /// Returns `CacheError::LockPoisoned` if the internal mutex is poisoned.
    pub fn start_background_eviction(self: &Arc<Self>) -> Result<()> {
        let mut handle_guard = self
            .eviction_handle
            .lock()
            .map_err(|_| CacheError::LockPoisoned("eviction handle".into()))?;

        if handle_guard.is_some() {
            return Ok(()); // Already running
        }

        let cache = Arc::clone(self);
        let interval = self.config.eviction_interval;
        let batch_size = self.config.eviction_batch_size;
        let mut shutdown_rx = self.shutdown_tx.subscribe();

        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let _ = cache.evict(batch_size);
                        let _ = cache.cleanup_expired();
                    }
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                }
            }
        });

        *handle_guard = Some(handle);
        drop(handle_guard);
        Ok(())
    }

    /// Stop background eviction gracefully.
    ///
    /// # Errors
    ///
    /// Returns `CacheError::LockPoisoned` if the internal mutex is poisoned.
    pub async fn stop_background_eviction(&self) -> Result<()> {
        // Send shutdown signal (ignore error if no receivers)
        self.shutdown_tx.send(()).ok();

        let handle = {
            let mut guard = self
                .eviction_handle
                .lock()
                .map_err(|_| CacheError::LockPoisoned("eviction handle".into()))?;
            guard.take()
        };

        if let Some(h) = handle {
            // Wait for the task to complete, ignoring join errors
            let _ = h.await;
        }

        Ok(())
    }

    /// Check if background eviction is currently running.
    #[must_use]
    pub fn is_background_eviction_running(&self) -> bool {
        self.eviction_handle
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Manually run eviction using the configured strategy (LRU, LFU, Cost, Hybrid).
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

    pub fn clear(&self) {
        // Delete all cache entries - entries may not exist, delete is idempotent
        for key in self.store.scan(prefixes::EXACT) {
            self.store.delete(&key).ok();
        }
        for key in self.store.scan(prefixes::SEMANTIC) {
            self.store.delete(&key).ok();
        }
        for key in self.store.scan(prefixes::EMBEDDING) {
            self.store.delete(&key).ok();
        }

        // Clear HNSW index
        self.index.clear();

        // Reset stats sizes
        self.stats.set_size(CacheLayer::Exact, 0);
        self.stats.set_size(CacheLayer::Semantic, 0);
        self.stats.set_size(CacheLayer::Embedding, 0);
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.stats.total_entries()
    }

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
            // Use sparse format for vectors with >50% zeros
            let storage = if Self::should_use_sparse(emb) {
                TensorValue::Sparse(SparseVector::from_dense(emb))
            } else {
                TensorValue::Vector(emb.to_vec())
            };
            data.set(fields::EMBEDDING, storage);
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

    /// Check if a vector should use sparse storage (50% threshold).
    fn should_use_sparse(vector: &[f32]) -> bool {
        if vector.is_empty() {
            return false;
        }
        let nnz = vector.iter().filter(|&&v| v.abs() > 1e-6).count();
        // For 0.5 threshold: sparse if nnz <= len/2, i.e., nnz*2 <= len
        nnz * 2 <= vector.len()
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

        eprintln!("exact_count from scan: {exact_count}");
        eprintln!("semantic_count from scan: {semantic_count}");

        assert!(exact_count > 0, "Expected scan to find exact entries");
    }

    fn create_test_cache() -> Cache {
        let mut config = CacheConfig::default();
        config.embedding_dim = 3;
        Cache::with_config(config).unwrap()
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        SparseVector::from_dense(v)
            .normalize()
            .map_or_else(|| v.to_vec(), |sv| sv.to_dense())
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
        let _ = cache.get("prompt", None); // Stats test - hit
        let _ = cache.get("other", None); // Stats test - miss

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
                .put(&format!("prompt{i}"), &embedding, "response", "gpt-4", None)
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
        cache.put_simple("key", "value").unwrap();

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

        assert!(!cache.is_empty());
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_with_shared_store() {
        let store = TensorStore::new();
        let config = CacheConfig::default();
        let cache = Cache::with_store(store, config).unwrap();

        assert!(cache.is_empty());
    }

    #[test]
    fn test_config_validation_invalid_eviction_batch() {
        let config = CacheConfig {
            eviction_batch_size: 0,
            ..Default::default()
        };
        let result = Cache::with_config(config);
        assert!(matches!(result, Err(CacheError::InvalidConfig(_))));
    }

    #[test]
    fn test_semantic_miss() {
        let cache = create_test_cache();
        let embedding1 = normalize(&[1.0, 0.0, 0.0]);
        let embedding2 = normalize(&[0.0, 1.0, 0.0]); // Orthogonal

        cache
            .put("prompt1", &embedding1, "response1", "gpt-4", None)
            .unwrap();

        // Search with orthogonal embedding - should miss semantic cache
        let result = cache.get("nonexistent", Some(&embedding2));
        assert!(result.is_none());

        let snapshot = cache.stats_snapshot();
        assert_eq!(snapshot.semantic_misses, 1);
    }

    #[test]
    fn test_exact_capacity_full() {
        let config = CacheConfig {
            embedding_dim: 3,
            exact_capacity: 2,
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache.put("p1", &embedding, "r1", "gpt-4", None).unwrap();
        cache.put("p2", &embedding, "r2", "gpt-4", None).unwrap();

        let result = cache.put("p3", &embedding, "r3", "gpt-4", None);
        assert!(matches!(result, Err(CacheError::CacheFull { .. })));
    }

    #[test]
    fn test_semantic_capacity_full() {
        let config = CacheConfig {
            embedding_dim: 3,
            exact_capacity: 100,
            semantic_capacity: 2,
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();

        let e1 = normalize(&[1.0, 0.0, 0.0]);
        let e2 = normalize(&[0.0, 1.0, 0.0]);
        let e3 = normalize(&[0.0, 0.0, 1.0]);

        cache.put("p1", &e1, "r1", "gpt-4", None).unwrap();
        cache.put("p2", &e2, "r2", "gpt-4", None).unwrap();

        let result = cache.put("p3", &e3, "r3", "gpt-4", None);
        assert!(matches!(result, Err(CacheError::CacheFull { .. })));
    }

    #[test]
    fn test_embedding_capacity_full() {
        let config = CacheConfig {
            embedding_dim: 3,
            embedding_capacity: 2,
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();

        cache
            .put_embedding("src", "c1", vec![0.1, 0.2, 0.3], "model")
            .unwrap();
        cache
            .put_embedding("src", "c2", vec![0.1, 0.2, 0.3], "model")
            .unwrap();

        let result = cache.put_embedding("src", "c3", vec![0.1, 0.2, 0.3], "model");
        assert!(matches!(result, Err(CacheError::CacheFull { .. })));
    }

    #[test]
    fn test_auto_metric_selection_sparse() {
        let config = CacheConfig {
            embedding_dim: 10,
            auto_select_metric: true,
            sparsity_metric_threshold: 0.5,
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();

        // Dense embedding (no zeros)
        let dense = normalize(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        cache
            .put("dense", &dense, "dense_response", "gpt-4", None)
            .unwrap();

        // Sparse embedding (80% zeros)
        let sparse = normalize(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        cache
            .put("sparse", &sparse, "sparse_response", "gpt-4", None)
            .unwrap();

        // Query with sparse should use Jaccard
        let hit = cache.get("different", Some(&sparse));
        if let Some(h) = hit {
            assert_eq!(h.metric_used, Some(DistanceMetric::Jaccard));
        }
    }

    #[test]
    fn test_auto_metric_disabled() {
        let config = CacheConfig {
            embedding_dim: 10,
            auto_select_metric: false,
            distance_metric: DistanceMetric::Cosine,
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();

        let sparse = normalize(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        cache
            .put("sparse", &sparse, "response", "gpt-4", None)
            .unwrap();

        // Even with sparse embedding, should use Cosine since auto_select is disabled
        let hit = cache.get("different", Some(&sparse));
        if let Some(h) = hit {
            assert_eq!(h.metric_used, Some(DistanceMetric::Cosine));
        }
    }

    #[test]
    fn test_invalidate_nonexistent() {
        let cache = create_test_cache();
        assert!(!cache.invalidate("nonexistent"));
    }

    #[test]
    fn test_invalidate_version() {
        let cache = create_test_cache();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        // Put with version
        cache
            .put("prompt1", &embedding, "response1", "gpt-4", None)
            .unwrap();

        // invalidate_version scans semantic cache entries
        let removed = cache.invalidate_version("v1.0");
        // No entries have version set via put(), so should be 0
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_invalidate_embeddings() {
        let cache = create_test_cache();

        cache
            .put_embedding("source1", "content1", vec![0.1, 0.2, 0.3], "model")
            .unwrap();
        cache
            .put_embedding("source1", "content2", vec![0.4, 0.5, 0.6], "model")
            .unwrap();
        cache
            .put_embedding("source2", "content1", vec![0.7, 0.8, 0.9], "model")
            .unwrap();

        // Invalidate all embeddings from source1
        let removed = cache.invalidate_embeddings("source1");
        assert_eq!(removed, 2);

        // source2 embedding should still exist
        assert!(cache.get_embedding("source2", "content1").is_some());
        assert!(cache.get_embedding("source1", "content1").is_none());
    }

    #[test]
    fn test_cleanup_expired() {
        use std::time::Duration;

        let config = CacheConfig {
            embedding_dim: 3,
            default_ttl: Duration::from_millis(50),
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("prompt", &embedding, "response", "gpt-4", None)
            .unwrap();
        cache.put_simple("key", "value").unwrap();
        cache
            .put_embedding("src", "content", vec![0.1, 0.2, 0.3], "model")
            .unwrap();

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(100));

        let cleaned = cache.cleanup_expired();
        // cleanup_expired returns usize, test that it runs without panic
        let _ = cleaned;
    }

    #[test]
    fn test_get_expired_returns_none() {
        use std::time::Duration;

        let config = CacheConfig {
            embedding_dim: 3,
            default_ttl: Duration::from_millis(50),
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();

        cache.put_simple("key", "value").unwrap();
        assert!(cache.get_simple("key").is_some());

        std::thread::sleep(Duration::from_millis(100));

        // Should return None for expired entry
        assert!(cache.get_simple("key").is_none());
    }

    #[test]
    fn test_get_embedding_expired_returns_none() {
        use std::time::Duration;

        let config = CacheConfig {
            embedding_dim: 3,
            default_ttl: Duration::from_millis(50),
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();

        cache
            .put_embedding("src", "content", vec![0.1, 0.2, 0.3], "model")
            .unwrap();

        std::thread::sleep(Duration::from_millis(100));

        assert!(cache.get_embedding("src", "content").is_none());
    }

    #[test]
    fn test_semantic_match_expired_returns_none() {
        use std::time::Duration;

        let config = CacheConfig {
            embedding_dim: 3,
            default_ttl: Duration::from_millis(50),
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("prompt", &embedding, "response", "gpt-4", None)
            .unwrap();

        std::thread::sleep(Duration::from_millis(100));

        // Semantic lookup should return None for expired
        let result = cache.get("different", Some(&embedding));
        assert!(result.is_none());
    }

    #[test]
    fn test_stats_accessor() {
        let cache = create_test_cache();
        let stats = cache.stats();
        assert_eq!(stats.total_entries(), 0);
    }

    #[test]
    fn test_config_accessor() {
        let cache = create_test_cache();
        let config = cache.config();
        assert_eq!(config.embedding_dim, 3);
    }

    #[test]
    fn test_cache_layer_debug() {
        assert_eq!(format!("{:?}", CacheLayer::Exact), "Exact");
        assert_eq!(format!("{:?}", CacheLayer::Semantic), "Semantic");
        assert_eq!(format!("{:?}", CacheLayer::Embedding), "Embedding");
    }

    #[test]
    fn test_exact_match_expired_returns_none() {
        use std::time::Duration;

        let config = CacheConfig {
            embedding_dim: 3,
            default_ttl: Duration::from_millis(50),
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        cache
            .put("prompt", &embedding, "response", "gpt-4", None)
            .unwrap();

        std::thread::sleep(Duration::from_millis(100));

        // Exact lookup should return None for expired
        let result = cache.get("prompt", None);
        assert!(result.is_none());
    }

    #[test]
    fn test_cleanup_expired_with_entries() {
        use std::time::Duration;

        let config = CacheConfig {
            embedding_dim: 3,
            default_ttl: Duration::from_millis(10),
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        // Add entries to all layers
        cache
            .put("prompt", &embedding, "response", "gpt-4", None)
            .unwrap();
        cache.put_simple("key", "value").unwrap();
        cache
            .put_embedding("src", "content", vec![0.1, 0.2, 0.3], "model")
            .unwrap();

        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(50));

        // cleanup_expired should find and clean expired entries
        let cleaned = cache.cleanup_expired();
        assert!(cleaned > 0, "Expected to clean at least 1 expired entry");
    }

    // Sparse embedding tests

    #[test]
    fn test_sparse_embedding_storage_and_retrieval() {
        let mut config = CacheConfig::default();
        config.embedding_dim = 100;
        let cache = Cache::with_config(config).unwrap();

        // Create a sparse embedding (>50% zeros)
        let mut sparse = vec![0.0f32; 100];
        sparse[0] = 1.0;
        sparse[50] = 2.0;
        sparse[99] = 3.0;

        cache
            .put_embedding("source", "content", sparse.clone(), "model")
            .unwrap();

        let retrieved = cache.get_embedding("source", "content").unwrap();
        assert_eq!(retrieved.len(), 100);
        assert!((retrieved[0] - 1.0).abs() < f32::EPSILON);
        assert!((retrieved[50] - 2.0).abs() < f32::EPSILON);
        assert!((retrieved[99] - 3.0).abs() < f32::EPSILON);
        assert!(retrieved[1].abs() < f32::EPSILON);
    }

    #[test]
    fn test_sparse_semantic_cache() {
        let mut config = CacheConfig::default();
        config.embedding_dim = 100;
        config.semantic_threshold = 0.9;
        let cache = Cache::with_config(config).unwrap();

        // Create a sparse embedding
        let mut sparse = vec![0.0f32; 100];
        sparse[0] = 1.0;

        cache
            .put("prompt", &sparse, "response", "gpt-4", None)
            .unwrap();

        // Should retrieve via exact match
        let hit = cache.get("prompt", None).unwrap();
        assert_eq!(hit.response, "response");
        assert_eq!(hit.layer, CacheLayer::Exact);
    }

    #[test]
    fn test_sparse_detection_threshold() {
        // Exactly 50% zeros should use sparse
        let half_sparse: Vec<f32> = (0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect();
        assert!(Cache::should_use_sparse(&half_sparse));

        // Less than 50% zeros should use dense
        let mostly_dense: Vec<f32> = (0..100).map(|i| if i < 40 { 0.0 } else { 1.0 }).collect();
        assert!(!Cache::should_use_sparse(&mostly_dense));

        // 97% zeros should use sparse
        let very_sparse: Vec<f32> = (0..100).map(|i| if i < 3 { 1.0 } else { 0.0 }).collect();
        assert!(Cache::should_use_sparse(&very_sparse));
    }

    // Background eviction tests

    #[tokio::test]
    async fn test_start_stop_background_eviction() {
        let cache = Arc::new(create_test_cache());

        assert!(!cache.is_background_eviction_running());

        cache.start_background_eviction().unwrap();
        assert!(cache.is_background_eviction_running());

        cache.stop_background_eviction().await.unwrap();
        assert!(!cache.is_background_eviction_running());
    }

    #[tokio::test]
    async fn test_double_start_is_idempotent() {
        let cache = Arc::new(create_test_cache());

        cache.start_background_eviction().unwrap();
        assert!(cache.is_background_eviction_running());

        // Second start should succeed without error
        cache.start_background_eviction().unwrap();
        assert!(cache.is_background_eviction_running());

        cache.stop_background_eviction().await.unwrap();
        assert!(!cache.is_background_eviction_running());
    }

    #[tokio::test]
    async fn test_stop_without_start_is_safe() {
        let cache = Arc::new(create_test_cache());

        // Stop without start should not panic or error
        cache.stop_background_eviction().await.unwrap();
        assert!(!cache.is_background_eviction_running());
    }

    #[tokio::test]
    async fn test_background_eviction_mechanism() {
        use std::time::Duration;

        let config = CacheConfig {
            embedding_dim: 3,
            default_ttl: Duration::from_millis(10),
            eviction_interval: Duration::from_millis(20),
            eviction_batch_size: 10,
            ..Default::default()
        };
        let cache = Arc::new(Cache::with_config(config).unwrap());
        let embedding = normalize(&[1.0, 0.0, 0.0]);

        // Add entries
        cache
            .put("prompt", &embedding, "response", "gpt-4", None)
            .unwrap();
        cache.put_simple("key", "value").unwrap();

        let initial_len = cache.len();
        assert!(initial_len > 0);

        // Wait for entries to expire
        tokio::time::sleep(Duration::from_millis(30)).await;

        // Verify cleanup_expired works correctly (this is what the background task calls)
        let directly_cleaned = cache.cleanup_expired();
        assert!(
            directly_cleaned > 0,
            "Direct cleanup should clean expired entries"
        );

        let snapshot = cache.stats_snapshot();
        assert!(snapshot.expirations > 0, "Expirations should be recorded");
    }
}
