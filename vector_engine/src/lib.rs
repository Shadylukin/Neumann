//! Vector Engine - k-NN similarity search with HNSW support
//!
//! This crate provides embeddings storage and similarity search functionality
//! for the Neumann database system.
//!
//! # Features
//!
//! - **Dense and sparse vectors**: Automatic sparse detection for memory efficiency
//! - **Multiple distance metrics**: Cosine, Euclidean, dot product, and extended metrics
//! - **HNSW indexing**: Hierarchical Navigable Small World graphs for fast k-NN search
//! - **Batch operations**: Parallel processing for large embedding batches
//! - **Entity embeddings**: Associate embeddings with existing entities in TensorStore
//! - **Memory bounds**: Configurable limits for dimension and scan operations
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use vector_engine::{VectorEngine, VectorEngineConfig};
//!
//! let engine = VectorEngine::new();
//! engine.store_embedding("doc1", vec![0.1, 0.2, 0.3]).unwrap();
//! engine.store_embedding("doc2", vec![0.2, 0.3, 0.4]).unwrap();
//!
//! let results = engine.search_similar(&[0.15, 0.25, 0.35], 10).unwrap();
//! ```
//!
//! # Configuration
//!
//! Use `VectorEngineConfig` to tune behavior:
//! - `max_dimension`: Limit embedding dimensions for memory safety
//! - `max_keys_per_scan`: Bound memory usage on unbounded operations
//! - `sparse_threshold`: Control sparse vector detection (0.0-1.0)
//! - `parallel_threshold`: Control when to use parallel processing

// Pedantic lint configuration for vector_engine
#![allow(clippy::cast_possible_truncation)] // Truncation from usize to f32 for ratios is intentional
#![allow(clippy::cast_precision_loss)] // Precision loss in f32 casts is acceptable for ratios
#![allow(clippy::needless_pass_by_value)] // Vec ownership is intentional for API design
#![allow(clippy::missing_errors_doc)] // Error conditions are self-evident from Result types
#![allow(clippy::uninlined_format_args)] // Keep format strings readable
#![allow(clippy::similar_names)] // score vs store is clear in context
#![allow(clippy::doc_markdown)] // Flexibility in doc formatting
#![allow(clippy::must_use_candidate)] // Not all pure functions need #[must_use]
#![allow(clippy::missing_const_for_fn)] // Some functions can't be const due to dependencies
#![allow(missing_docs)] // Docs exist on public types; field-level docs not required

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tensor_store::{
    fields, hnsw::simd, SparseVector, TensorData, TensorStore, TensorStoreError, TensorValue,
};
use tracing::instrument;
// Re-export HNSW types from tensor_store for backward compatibility
pub use tensor_store::{HNSWConfig, HNSWIndex};

// Re-export distance metrics from tensor_store for extended metric support (9 variants + composite)
pub use tensor_store::{DistanceMetric as ExtendedDistanceMetric, GeometricConfig};

/// Error types for vector operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorError {
    /// The requested embedding was not found.
    NotFound(String),
    /// Vector dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Empty vector provided.
    EmptyVector,
    /// Invalid top_k value.
    InvalidTopK,
    /// Storage error from underlying tensor store.
    StorageError(String),
    /// Validation error during batch operation.
    BatchValidationError { index: usize, cause: String },
    /// Operation error during batch operation.
    BatchOperationError { index: usize, cause: String },
    /// Configuration error.
    ConfigurationError(String),
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(key) => write!(f, "Embedding not found: {key}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {expected}, got {got}")
            },
            Self::EmptyVector => write!(f, "Empty vector provided"),
            Self::InvalidTopK => write!(f, "Invalid top_k value (must be > 0)"),
            Self::StorageError(e) => write!(f, "Storage error: {e}"),
            Self::BatchValidationError { index, cause } => {
                write!(f, "Batch validation error at index {index}: {cause}")
            },
            Self::BatchOperationError { index, cause } => {
                write!(f, "Batch operation error at index {index}: {cause}")
            },
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {msg}"),
        }
    }
}

impl std::error::Error for VectorError {}

impl From<TensorStoreError> for VectorError {
    fn from(e: TensorStoreError) -> Self {
        Self::StorageError(e.to_string())
    }
}

/// Conversion from the simple `DistanceMetric` (3 variants) to the extended
/// `ExtendedDistanceMetric` (10 variants) for HNSW-based search.
impl From<DistanceMetric> for ExtendedDistanceMetric {
    fn from(metric: DistanceMetric) -> Self {
        match metric {
            DistanceMetric::Euclidean => Self::Euclidean,
            // Both Cosine and DotProduct map to Cosine (closest equivalent)
            DistanceMetric::Cosine | DistanceMetric::DotProduct => Self::Cosine,
        }
    }
}

pub type Result<T> = std::result::Result<T, VectorError>;

/// Result of a similarity search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// The key of the matching embedding.
    pub key: String,
    /// The similarity score (cosine similarity, range -1 to 1).
    pub score: f32,
}

impl SearchResult {
    #[must_use]
    pub const fn new(key: String, score: f32) -> Self {
        Self { key, score }
    }
}

/// Distance metric for similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity: measures angle between vectors (default).
    #[default]
    Cosine,
    /// Euclidean distance: L2 norm of difference.
    Euclidean,
    /// Dot product: inner product of vectors.
    DotProduct,
}

// ========== Configuration ==========

/// Configuration for the Vector Engine.
#[derive(Debug, Clone)]
pub struct VectorEngineConfig {
    /// Default dimension for embeddings (validated on first store if set).
    pub default_dimension: Option<usize>,
    /// Threshold for sparse vector detection (fraction of zeros).
    pub sparse_threshold: f32,
    /// Batch size threshold for parallel processing.
    pub parallel_threshold: usize,
    /// Default distance metric for search operations.
    pub default_metric: DistanceMetric,
    /// Maximum allowed embedding dimension for memory safety.
    pub max_dimension: Option<usize>,
    /// Maximum keys to return from unbounded scan operations.
    pub max_keys_per_scan: Option<usize>,
    /// Threshold for using parallel iteration in batch operations.
    pub batch_parallel_threshold: usize,
}

impl Default for VectorEngineConfig {
    fn default() -> Self {
        Self {
            default_dimension: None,
            sparse_threshold: 0.5,
            parallel_threshold: 5000,
            default_metric: DistanceMetric::Cosine,
            max_dimension: None,
            max_keys_per_scan: None,
            batch_parallel_threshold: 100,
        }
    }
}

impl VectorEngineConfig {
    #[must_use]
    pub const fn high_throughput() -> Self {
        Self {
            default_dimension: None,
            sparse_threshold: 0.5,
            parallel_threshold: 1000, // Lower threshold = more parallelism
            default_metric: DistanceMetric::Cosine,
            max_dimension: None,
            max_keys_per_scan: None,
            batch_parallel_threshold: 100,
        }
    }

    #[must_use]
    pub const fn low_memory() -> Self {
        Self {
            default_dimension: None,
            sparse_threshold: 0.3, // More aggressive sparse detection
            parallel_threshold: 5000,
            default_metric: DistanceMetric::Cosine,
            max_dimension: Some(4096),
            max_keys_per_scan: Some(10_000),
            batch_parallel_threshold: 100,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.sparse_threshold < 0.0 || self.sparse_threshold > 1.0 {
            return Err(VectorError::ConfigurationError(
                "sparse_threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.parallel_threshold == 0 {
            return Err(VectorError::ConfigurationError(
                "parallel_threshold must be greater than 0".to_string(),
            ));
        }
        if let Some(max_dim) = self.max_dimension {
            if max_dim == 0 {
                return Err(VectorError::ConfigurationError(
                    "max_dimension must be greater than 0".to_string(),
                ));
            }
        }
        if let Some(max_keys) = self.max_keys_per_scan {
            if max_keys == 0 {
                return Err(VectorError::ConfigurationError(
                    "max_keys_per_scan must be greater than 0".to_string(),
                ));
            }
        }
        if self.batch_parallel_threshold == 0 {
            return Err(VectorError::ConfigurationError(
                "batch_parallel_threshold must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

// ========== Batch Operations ==========

/// Input for batch embedding storage.
#[derive(Debug, Clone)]
pub struct EmbeddingInput {
    pub key: String,
    pub vector: Vec<f32>,
}

impl EmbeddingInput {
    #[must_use]
    pub fn new(key: impl Into<String>, vector: Vec<f32>) -> Self {
        Self {
            key: key.into(),
            vector,
        }
    }
}

/// Result of a batch operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchResult {
    pub stored_keys: Vec<String>,
    pub count: usize,
}

impl BatchResult {
    #[must_use]
    pub const fn new(stored_keys: Vec<String>) -> Self {
        let count = stored_keys.len();
        Self { stored_keys, count }
    }
}

// ========== Pagination ==========

/// Pagination parameters for list and search operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pagination {
    /// Number of items to skip.
    pub skip: usize,
    /// Maximum items to return.
    pub limit: Option<usize>,
    /// Whether to compute total count (may be expensive).
    pub count_total: bool,
}

impl Pagination {
    #[must_use]
    pub const fn new(skip: usize, limit: usize) -> Self {
        Self {
            skip,
            limit: Some(limit),
            count_total: false,
        }
    }

    #[must_use]
    pub const fn with_total(mut self) -> Self {
        self.count_total = true;
        self
    }

    #[must_use]
    pub const fn skip_only(skip: usize) -> Self {
        Self {
            skip,
            limit: None,
            count_total: false,
        }
    }
}

/// Result of a paginated query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagedResult<T> {
    pub items: Vec<T>,
    pub total_count: Option<usize>,
    pub has_more: bool,
}

impl<T> PagedResult<T> {
    #[must_use]
    pub const fn new(items: Vec<T>, total_count: Option<usize>, has_more: bool) -> Self {
        Self {
            items,
            total_count,
            has_more,
        }
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self {
            items: Vec::new(),
            total_count: Some(0),
            has_more: false,
        }
    }
}

/// Vector Engine for storing and searching embeddings.
///
/// Uses cosine similarity for nearest neighbor search with optional HNSW indexing.
///
/// # Thread Safety
///
/// `VectorEngine` is thread-safe and can be shared across threads via `Arc<VectorEngine>`.
/// Thread safety is inherited from `TensorStore`'s internal sharded `SlabRouter` which uses
/// fine-grained locking. Concurrent reads and writes to different keys proceed in parallel.
///
/// # Memory Considerations
///
/// Use `VectorEngineConfig::max_keys_per_scan` and `max_dimension` to bound memory usage
/// in production environments. For large datasets, prefer `list_keys_paginated()` over
/// `list_keys()` and use HNSW indexing for efficient k-NN search.
///
/// # Distance Metrics
///
/// - `DistanceMetric` (3 variants): Simple API for basic similarity search
///   - `Cosine`, `Euclidean`, `DotProduct`
/// - `ExtendedDistanceMetric` (10 variants): Full metric support for HNSW-based search
///   - Includes `Angular`, `Jaccard`, `Overlap`, `Manhattan`, `Geodesic`, `Composite`
///
/// Use `DistanceMetric` for `search_similar()` and `ExtendedDistanceMetric` for
/// `search_with_hnsw_and_metric()`.
pub struct VectorEngine {
    store: TensorStore,
    config: VectorEngineConfig,
}

impl VectorEngine {
    /// Create a new VectorEngine with a fresh TensorStore.
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: TensorStore::new(),
            config: VectorEngineConfig::default(),
        }
    }

    /// Create a VectorEngine using an existing TensorStore.
    #[must_use]
    pub fn with_store(store: TensorStore) -> Self {
        Self {
            store,
            config: VectorEngineConfig::default(),
        }
    }

    /// Create a VectorEngine with custom configuration.
    ///
    /// Validates the configuration before construction.
    pub fn with_config(config: VectorEngineConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            store: TensorStore::new(),
            config,
        })
    }

    /// Create a VectorEngine with existing store and custom configuration.
    ///
    /// Validates the configuration before construction.
    pub fn with_store_and_config(store: TensorStore, config: VectorEngineConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { store, config })
    }

    /// Get a reference to the underlying TensorStore.
    pub fn store(&self) -> &TensorStore {
        &self.store
    }

    /// Get a reference to the configuration.
    pub const fn config(&self) -> &VectorEngineConfig {
        &self.config
    }

    /// Key prefix for embeddings.
    fn embedding_key(key: &str) -> String {
        format!("emb:{}", key)
    }

    /// Key prefix for all embeddings (for scanning).
    fn embedding_prefix() -> &'static str {
        "emb:"
    }

    /// Store an embedding vector with the given key.
    ///
    /// Automatically uses sparse format based on configured sparse_threshold.
    /// Overwrites any existing embedding with the same key.
    #[instrument(skip(self, vector), fields(key = %key, vector_dim = vector.len()))]
    pub fn store_embedding(&self, key: &str, vector: Vec<f32>) -> Result<()> {
        if vector.is_empty() {
            return Err(VectorError::EmptyVector);
        }

        if let Some(max_dim) = self.config.max_dimension {
            if vector.len() > max_dim {
                return Err(VectorError::DimensionMismatch {
                    expected: max_dim,
                    got: vector.len(),
                });
            }
        }

        let storage_key = Self::embedding_key(key);
        let mut tensor = TensorData::new();

        // Detect sparse vectors and store in optimal format
        let storage = if self.should_use_sparse(&vector) {
            TensorValue::Sparse(SparseVector::from_dense(&vector))
        } else {
            TensorValue::Vector(vector)
        };
        tensor.set("vector", storage);

        self.store.put(storage_key, tensor)?;
        Ok(())
    }

    /// Check if a vector should use sparse storage based on config threshold.
    fn should_use_sparse(&self, vector: &[f32]) -> bool {
        Self::should_use_sparse_with_threshold(vector, self.config.sparse_threshold)
    }

    /// Check if a vector should use sparse storage with a given threshold.
    fn should_use_sparse_with_threshold(vector: &[f32], threshold: f32) -> bool {
        if vector.is_empty() {
            return false;
        }
        let nnz = vector.iter().filter(|&&v| v.abs() > 1e-6).count();
        let zero_ratio = 1.0 - (nnz as f32 / vector.len() as f32);
        zero_ratio >= threshold
    }

    /// Get an embedding by key.
    ///
    /// Returns the embedding as a dense vector regardless of storage format.
    #[instrument(skip(self), fields(key = %key))]
    pub fn get_embedding(&self, key: &str) -> Result<Vec<f32>> {
        let storage_key = Self::embedding_key(key);
        let tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(key.to_string()))?;

        match tensor.get("vector") {
            Some(TensorValue::Vector(v)) => Ok(v.clone()),
            Some(TensorValue::Sparse(s)) => Ok(s.to_dense()),
            _ => Err(VectorError::NotFound(key.to_string())),
        }
    }

    /// Delete an embedding by key.
    #[instrument(skip(self), fields(key = %key))]
    pub fn delete_embedding(&self, key: &str) -> Result<()> {
        let storage_key = Self::embedding_key(key);
        if !self.store.exists(&storage_key) {
            return Err(VectorError::NotFound(key.to_string()));
        }
        self.store.delete(&storage_key)?;
        Ok(())
    }

    /// Check if an embedding exists.
    pub fn exists(&self, key: &str) -> bool {
        let storage_key = Self::embedding_key(key);
        self.store.exists(&storage_key)
    }

    /// Get the count of stored embeddings.
    pub fn count(&self) -> usize {
        self.store.scan_count(Self::embedding_prefix())
    }

    /// Search for the top_k most similar embeddings to the query vector.
    ///
    /// Returns results sorted by similarity score (highest first).
    /// Uses cosine similarity with SIMD acceleration.
    /// Automatically uses parallel iteration for large datasets.
    #[instrument(skip(self, query), fields(query_dim = query.len(), top_k = top_k))]
    pub fn search_similar(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        if let Some(max_dim) = self.config.max_dimension {
            if query.len() > max_dim {
                return Err(VectorError::DimensionMismatch {
                    expected: max_dim,
                    got: query.len(),
                });
            }
        }

        // Pre-compute query magnitude for efficiency
        let query_magnitude = Self::magnitude(query);
        if query_magnitude == 0.0 {
            // Zero vector - return empty results
            return Ok(Vec::new());
        }

        let keys = self.store.scan(Self::embedding_prefix());

        // Use parallel iteration for large datasets, sequential for small
        let mut results: Vec<SearchResult> = if keys.len() >= self.config.parallel_threshold {
            Self::search_parallel(&self.store, &keys, query, query_magnitude)
        } else {
            Self::search_sequential(&self.store, &keys, query, query_magnitude)
        };

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top_k
        results.truncate(top_k);

        Ok(results)
    }

    /// Search for the top_k most similar embeddings using the specified distance metric.
    ///
    /// - Cosine: Higher scores are more similar (range -1 to 1)
    /// - DotProduct: Higher scores are more similar (unbounded)
    /// - Euclidean: Lower distances are more similar (converted to score via 1/(1+dist))
    pub fn search_similar_with_metric(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        let query_magnitude = Self::magnitude(query);
        // Zero-magnitude queries are invalid for cosine/dot product but valid for euclidean
        if query_magnitude == 0.0 && !matches!(metric, DistanceMetric::Euclidean) {
            return Ok(Vec::new());
        }

        let keys = self.store.scan(Self::embedding_prefix());

        let mut results: Vec<SearchResult> = if keys.len() >= self.config.parallel_threshold {
            Self::search_parallel_with_metric(&self.store, &keys, query, query_magnitude, metric)
        } else {
            Self::search_sequential_with_metric(&self.store, &keys, query, query_magnitude, metric)
        };

        // Sort by score descending (higher is better for all metrics after transformation)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(top_k);
        Ok(results)
    }

    /// Extract vector from TensorValue, handling both dense and sparse formats.
    fn extract_vector(value: &TensorValue) -> Option<Vec<f32>> {
        match value {
            TensorValue::Vector(v) => Some(v.clone()),
            TensorValue::Sparse(s) => Some(s.to_dense()),
            _ => None,
        }
    }

    /// Sequential similarity search (faster for small datasets)
    fn search_sequential(
        store: &TensorStore,
        keys: &[String],
        query: &[f32],
        query_magnitude: f32,
    ) -> Vec<SearchResult> {
        keys.iter()
            .filter_map(|storage_key| {
                let tensor = store.get(storage_key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get("vector")?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::cosine_similarity(query, &stored_vec, query_magnitude);
                let key = storage_key
                    .strip_prefix(Self::embedding_prefix())
                    .unwrap_or(storage_key)
                    .to_string();

                Some(SearchResult::new(key, score))
            })
            .collect()
    }

    /// Parallel similarity search (faster for large datasets)
    fn search_parallel(
        store: &TensorStore,
        keys: &[String],
        query: &[f32],
        query_magnitude: f32,
    ) -> Vec<SearchResult> {
        keys.par_iter()
            .filter_map(|storage_key| {
                let tensor = store.get(storage_key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get("vector")?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::cosine_similarity(query, &stored_vec, query_magnitude);
                let key = storage_key
                    .strip_prefix(Self::embedding_prefix())
                    .unwrap_or(storage_key)
                    .to_string();

                Some(SearchResult::new(key, score))
            })
            .collect()
    }

    /// Sequential similarity search with configurable metric
    fn search_sequential_with_metric(
        store: &TensorStore,
        keys: &[String],
        query: &[f32],
        query_magnitude: f32,
        metric: DistanceMetric,
    ) -> Vec<SearchResult> {
        keys.iter()
            .filter_map(|storage_key| {
                let tensor = store.get(storage_key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get("vector")?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::compute_score(query, &stored_vec, query_magnitude, metric);
                let key = storage_key
                    .strip_prefix(Self::embedding_prefix())
                    .unwrap_or(storage_key)
                    .to_string();

                Some(SearchResult::new(key, score))
            })
            .collect()
    }

    /// Parallel similarity search with configurable metric
    fn search_parallel_with_metric(
        store: &TensorStore,
        keys: &[String],
        query: &[f32],
        query_magnitude: f32,
        metric: DistanceMetric,
    ) -> Vec<SearchResult> {
        keys.par_iter()
            .filter_map(|storage_key| {
                let tensor = store.get(storage_key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get("vector")?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::compute_score(query, &stored_vec, query_magnitude, metric);
                let key = storage_key
                    .strip_prefix(Self::embedding_prefix())
                    .unwrap_or(storage_key)
                    .to_string();

                Some(SearchResult::new(key, score))
            })
            .collect()
    }

    /// Compute score based on the distance metric.
    fn compute_score(
        query: &[f32],
        stored: &[f32],
        query_magnitude: f32,
        metric: DistanceMetric,
    ) -> f32 {
        match metric {
            DistanceMetric::Cosine => Self::cosine_similarity(query, stored, query_magnitude),
            DistanceMetric::DotProduct => simd::dot_product(query, stored),
            DistanceMetric::Euclidean => {
                // Convert distance to similarity: 1 / (1 + distance)
                let dist = Self::euclidean_distance(query, stored);
                1.0 / (1.0 + dist)
            },
        }
    }

    /// Compute Euclidean distance between two vectors.
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        // Sum of squared differences
        let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        sum_sq.sqrt()
    }

    /// Compute cosine similarity between two vectors using SIMD.
    /// Optimized version that accepts pre-computed query magnitude.
    fn cosine_similarity(a: &[f32], b: &[f32], a_magnitude: f32) -> f32 {
        let dot_product = simd::dot_product(a, b);
        let b_magnitude = simd::magnitude(b);

        if b_magnitude == 0.0 {
            return 0.0;
        }

        dot_product / (a_magnitude * b_magnitude)
    }

    /// Compute the magnitude (L2 norm) of a vector using SIMD.
    fn magnitude(v: &[f32]) -> f32 {
        simd::magnitude(v)
    }

    /// Compute cosine similarity between two vectors (public helper).
    pub fn compute_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.is_empty() || b.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if a.len() != b.len() {
            return Err(VectorError::DimensionMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }

        let a_magnitude = Self::magnitude(a);
        if a_magnitude == 0.0 {
            return Ok(0.0);
        }

        Ok(Self::cosine_similarity(a, b, a_magnitude))
    }

    /// Get the dimension of stored embeddings (from first embedding found).
    pub fn dimension(&self) -> Option<usize> {
        let keys = self.store.scan(Self::embedding_prefix());
        for key in keys {
            if let Ok(tensor) = self.store.get(&key) {
                if let Some(vec) = tensor.get("vector").and_then(Self::extract_vector) {
                    return Some(vec.len());
                }
            }
        }
        None
    }

    /// List all embedding keys.
    pub fn list_keys(&self) -> Vec<String> {
        self.store
            .scan(Self::embedding_prefix())
            .into_iter()
            .filter_map(|k| k.strip_prefix(Self::embedding_prefix()).map(String::from))
            .collect()
    }

    /// List embedding keys with memory safety bounds.
    ///
    /// Uses `config.max_keys_per_scan` if set, otherwise returns all keys.
    /// Prefer this over `list_keys()` in production environments.
    pub fn list_keys_bounded(&self) -> Vec<String> {
        let limit = self.config.max_keys_per_scan.unwrap_or(usize::MAX);
        self.store
            .scan(Self::embedding_prefix())
            .into_iter()
            .take(limit)
            .filter_map(|k| k.strip_prefix(Self::embedding_prefix()).map(String::from))
            .collect()
    }

    /// Clear all embeddings.
    pub fn clear(&self) -> Result<()> {
        let keys = self.store.scan(Self::embedding_prefix());
        for key in keys {
            self.store.delete(&key)?;
        }
        Ok(())
    }

    /// Build an HNSW index from all stored embeddings.
    ///
    /// Returns a tuple of (index, key_mapping) where key_mapping maps node IDs to keys.
    /// Use this for fast approximate nearest neighbor search on large datasets.
    ///
    /// Validates dimension consistency across all vectors.
    ///
    /// # Example
    /// ```ignore
    /// let engine = VectorEngine::new();
    /// // ... store embeddings ...
    /// let (index, keys) = engine.build_hnsw_index(HNSWConfig::default())?;
    /// let results = index.search(&query, 10);
    /// for (node_id, score) in results {
    ///     println!("Key: {}, Score: {}", keys[node_id], score);
    /// }
    /// ```
    #[instrument(skip(self, config))]
    pub fn build_hnsw_index(&self, config: HNSWConfig) -> Result<(HNSWIndex, Vec<String>)> {
        let keys = self.list_keys();

        if keys.is_empty() {
            return Ok((HNSWIndex::with_config(config), Vec::new()));
        }

        // Validate first vector and establish expected dimension
        let first_key = &keys[0];
        let first_vector = self.get_embedding(first_key)?;
        let expected_dim = first_vector.len();

        // Check config dimension constraint
        if let Some(max_dim) = self.config.max_dimension {
            if expected_dim > max_dim {
                return Err(VectorError::DimensionMismatch {
                    expected: max_dim,
                    got: expected_dim,
                });
            }
        }

        let index = HNSWIndex::with_config(config);
        let mut key_mapping = Vec::with_capacity(keys.len());

        index.insert(first_vector);
        key_mapping.push(first_key.clone());

        for key in keys.into_iter().skip(1) {
            let vector = self.get_embedding(&key)?;

            // Validate dimension consistency
            if vector.len() != expected_dim {
                return Err(VectorError::DimensionMismatch {
                    expected: expected_dim,
                    got: vector.len(),
                });
            }

            index.insert(vector);
            key_mapping.push(key);
        }

        Ok((index, key_mapping))
    }

    /// Build an HNSW index with default configuration.
    pub fn build_hnsw_index_default(&self) -> Result<(HNSWIndex, Vec<String>)> {
        self.build_hnsw_index(HNSWConfig::default())
    }

    /// Estimate memory usage for building an HNSW index.
    ///
    /// Returns estimated bytes needed based on current embeddings.
    pub fn estimate_hnsw_memory(&self) -> Result<usize> {
        let count = self.count();
        if count == 0 {
            return Ok(0);
        }

        // Sample first embedding for dimension
        let keys = self.list_keys();
        let first = self.get_embedding(&keys[0])?;
        let dim = first.len();

        // Estimate: vectors + HNSW graph structure + key mapping
        // vectors: count * dim * 4 bytes (f32)
        // graph: ~count * M * 2 * 8 bytes (M=16 default, node ID refs)
        // keys: ~count * 32 bytes average key length
        let vector_bytes = count * dim * 4;
        let graph_bytes = count * 16 * 2 * 8;
        let key_bytes = count * 32;

        Ok(vector_bytes + graph_bytes + key_bytes)
    }

    /// This is a convenience method that converts node IDs back to SearchResults.
    pub fn search_with_hnsw(
        &self,
        index: &HNSWIndex,
        key_mapping: &[String],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        let results = index.search(query, top_k);
        Ok(results
            .into_iter()
            .filter_map(|(node_id, score)| {
                key_mapping.get(node_id).map(|key| SearchResult {
                    key: key.clone(),
                    score,
                })
            })
            .collect())
    }

    /// Search using HNSW for candidates, then re-rank with the specified metric.
    ///
    /// Fetches 2x candidates from HNSW, then re-ranks using the exact metric.
    /// Useful when you need more precision than HNSW's cosine-based search.
    pub fn search_with_hnsw_and_metric(
        &self,
        index: &HNSWIndex,
        key_mapping: &[String],
        query: &[f32],
        top_k: usize,
        metric: ExtendedDistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        // Fetch 2x candidates from HNSW
        let candidate_count = top_k.saturating_mul(2).max(10);
        let candidates = index.search(query, candidate_count);

        let query_sparse = SparseVector::from_dense(query);

        // Re-rank with exact metric
        let mut results: Vec<SearchResult> = candidates
            .iter()
            .filter_map(|(node_id, _)| {
                let key = key_mapping.get(*node_id)?;
                let vector = self.get_embedding(key).ok()?;
                let stored_sparse = SparseVector::from_dense(&vector);
                let raw_score = metric.compute(&query_sparse, &stored_sparse);
                let score = metric.to_similarity(raw_score);
                Some(SearchResult::new(key.clone(), score))
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(top_k);
        Ok(results)
    }

    // ========== Batch Operations ==========

    /// Store multiple embeddings in a single batch operation.
    ///
    /// Validates all inputs upfront before storing any.
    /// Uses parallel processing for batches larger than `config.batch_parallel_threshold`.
    #[instrument(skip(self, inputs), fields(count = inputs.len()))]
    pub fn batch_store_embeddings(&self, inputs: Vec<EmbeddingInput>) -> Result<BatchResult> {
        if inputs.is_empty() {
            return Ok(BatchResult::new(Vec::new()));
        }

        // Validate all inputs first
        for (i, input) in inputs.iter().enumerate() {
            if input.vector.is_empty() {
                return Err(VectorError::BatchValidationError {
                    index: i,
                    cause: "Empty vector provided".to_string(),
                });
            }
        }

        // Store embeddings (parallel for large batches)
        if inputs.len() >= self.config.batch_parallel_threshold {
            let results: Vec<Result<String>> = inputs
                .par_iter()
                .enumerate()
                .map(|(i, input)| {
                    self.store_embedding(&input.key, input.vector.clone())
                        .map(|()| input.key.clone())
                        .map_err(|e| VectorError::BatchOperationError {
                            index: i,
                            cause: e.to_string(),
                        })
                })
                .collect();

            // Collect results, failing on first error
            let mut stored_keys = Vec::with_capacity(inputs.len());
            for result in results {
                stored_keys.push(result?);
            }
            Ok(BatchResult::new(stored_keys))
        } else {
            let mut stored_keys = Vec::with_capacity(inputs.len());
            for (i, input) in inputs.iter().enumerate() {
                self.store_embedding(&input.key, input.vector.clone())
                    .map_err(|e| VectorError::BatchOperationError {
                        index: i,
                        cause: e.to_string(),
                    })?;
                stored_keys.push(input.key.clone());
            }
            Ok(BatchResult::new(stored_keys))
        }
    }

    /// Delete multiple embeddings in a single batch operation.
    ///
    /// Returns the count of successfully deleted embeddings.
    /// Silently skips keys that don't exist.
    #[instrument(skip(self, keys), fields(count = keys.len()))]
    pub fn batch_delete_embeddings(&self, keys: Vec<String>) -> Result<usize> {
        if keys.is_empty() {
            return Ok(0);
        }

        if keys.len() >= self.config.batch_parallel_threshold {
            let count: usize = keys
                .par_iter()
                .filter_map(|key| self.delete_embedding(key).ok())
                .count();
            Ok(count)
        } else {
            let mut count = 0;
            for key in &keys {
                if self.delete_embedding(key).is_ok() {
                    count += 1;
                }
            }
            Ok(count)
        }
    }

    // ========== Pagination ==========

    /// List embedding keys with pagination.
    pub fn list_keys_paginated(&self, pagination: Pagination) -> PagedResult<String> {
        let all_keys: Vec<String> = self
            .store
            .scan(Self::embedding_prefix())
            .into_iter()
            .filter_map(|k| k.strip_prefix(Self::embedding_prefix()).map(String::from))
            .collect();

        let total = all_keys.len();
        let total_count = if pagination.count_total {
            Some(total)
        } else {
            None
        };

        let skipped: Vec<String> = all_keys.into_iter().skip(pagination.skip).collect();

        let items: Vec<String> = match pagination.limit {
            Some(limit) => skipped.into_iter().take(limit).collect(),
            None => skipped,
        };

        let has_more = pagination.skip + items.len() < total;

        PagedResult::new(items, total_count, has_more)
    }

    /// Search for similar embeddings with pagination.
    pub fn search_similar_paginated(
        &self,
        query: &[f32],
        top_k: usize,
        pagination: Pagination,
    ) -> Result<PagedResult<SearchResult>> {
        // Get full results up to skip + limit
        let total_needed = pagination.skip + pagination.limit.unwrap_or(top_k);
        let results = self.search_similar(query, total_needed.min(top_k))?;

        let total_count = if pagination.count_total {
            Some(results.len())
        } else {
            None
        };

        let skipped: Vec<SearchResult> = results.into_iter().skip(pagination.skip).collect();

        let items: Vec<SearchResult> = match pagination.limit {
            Some(limit) => skipped.into_iter().take(limit).collect(),
            None => skipped,
        };

        let has_more = match (pagination.limit, total_count) {
            (Some(_), Some(total)) => pagination.skip + items.len() < total,
            _ => false,
        };

        Ok(PagedResult::new(items, total_count, has_more))
    }

    /// Search for similar entities with pagination.
    pub fn search_entities_paginated(
        &self,
        query: &[f32],
        top_k: usize,
        pagination: Pagination,
    ) -> Result<PagedResult<SearchResult>> {
        let total_needed = pagination.skip + pagination.limit.unwrap_or(top_k);
        let results = self.search_entities(query, total_needed.min(top_k))?;

        let total_count = if pagination.count_total {
            Some(results.len())
        } else {
            None
        };

        let skipped: Vec<SearchResult> = results.into_iter().skip(pagination.skip).collect();

        let items: Vec<SearchResult> = match pagination.limit {
            Some(limit) => skipped.into_iter().take(limit).collect(),
            None => skipped,
        };

        let has_more = match (pagination.limit, total_count) {
            (Some(_), Some(total)) => pagination.skip + items.len() < total,
            _ => false,
        };

        Ok(PagedResult::new(items, total_count, has_more))
    }

    // ========== Unified Entity Mode ==========
    // These methods work with entity keys directly (e.g., "user:1") and use the
    // _embedding field, enabling cross-engine queries on shared entities.

    /// Store embedding in an entity's _embedding field. Creates entity if needed.
    ///
    /// Automatically uses sparse format for vectors with >50% zeros.
    #[instrument(skip(self, vector), fields(key = %entity_key, vector_dim = vector.len()))]
    pub fn set_entity_embedding(&self, entity_key: &str, vector: Vec<f32>) -> Result<()> {
        if vector.is_empty() {
            return Err(VectorError::EmptyVector);
        }

        if let Some(max_dim) = self.config.max_dimension {
            if vector.len() > max_dim {
                return Err(VectorError::DimensionMismatch {
                    expected: max_dim,
                    got: vector.len(),
                });
            }
        }

        let mut tensor = self
            .store
            .get(entity_key)
            .unwrap_or_else(|_| TensorData::new());

        // Detect sparse vectors and store in optimal format
        let storage = if self.should_use_sparse(&vector) {
            TensorValue::Sparse(SparseVector::from_dense(&vector))
        } else {
            TensorValue::Vector(vector)
        };
        tensor.set(fields::EMBEDDING, storage);
        self.store.put(entity_key, tensor)?;
        Ok(())
    }

    /// Get embedding from an entity's _embedding field.
    ///
    /// Returns the embedding as a dense vector regardless of storage format.
    pub fn get_entity_embedding(&self, entity_key: &str) -> Result<Vec<f32>> {
        let tensor = self
            .store
            .get(entity_key)
            .map_err(|_| VectorError::NotFound(entity_key.to_string()))?;

        match tensor.get(fields::EMBEDDING) {
            Some(TensorValue::Vector(v)) => Ok(v.clone()),
            Some(TensorValue::Sparse(s)) => Ok(s.to_dense()),
            _ => Err(VectorError::NotFound(entity_key.to_string())),
        }
    }

    /// Check if an entity has an embedding.
    pub fn entity_has_embedding(&self, entity_key: &str) -> bool {
        self.store
            .get(entity_key)
            .map(|t| t.has(fields::EMBEDDING))
            .unwrap_or(false)
    }

    /// Remove embedding from an entity (keeps other entity data).
    pub fn remove_entity_embedding(&self, entity_key: &str) -> Result<()> {
        let mut tensor = self
            .store
            .get(entity_key)
            .map_err(|_| VectorError::NotFound(entity_key.to_string()))?;

        if tensor.remove(fields::EMBEDDING).is_none() {
            return Err(VectorError::NotFound(entity_key.to_string()));
        }

        self.store.put(entity_key, tensor)?;
        Ok(())
    }

    /// Search for similar entities using _embedding field.
    #[instrument(skip(self, query), fields(query_dim = query.len(), top_k = top_k))]
    pub fn search_entities(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        if let Some(max_dim) = self.config.max_dimension {
            if query.len() > max_dim {
                return Err(VectorError::DimensionMismatch {
                    expected: max_dim,
                    got: query.len(),
                });
            }
        }

        let query_magnitude = Self::magnitude(query);
        if query_magnitude == 0.0 {
            return Ok(Vec::new());
        }

        let keys = self.store.scan("");

        let mut results: Vec<SearchResult> = keys
            .iter()
            .filter_map(|key| {
                let tensor = self.store.get(key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get(fields::EMBEDDING)?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::cosine_similarity(query, &stored_vec, query_magnitude);
                Some(SearchResult::new(key.clone(), score))
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }

    /// Scan for all entity keys that have embeddings.
    pub fn scan_entities_with_embeddings(&self) -> Vec<String> {
        self.store
            .scan("")
            .into_iter()
            .filter(|key| self.entity_has_embedding(key))
            .collect()
    }

    /// Count entities with embeddings (unified mode).
    pub fn count_entities_with_embeddings(&self) -> usize {
        self.scan_entities_with_embeddings().len()
    }
}

impl Default for VectorEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vector(dim: usize, seed: usize) -> Vec<f32> {
        // Create vectors where each seed produces a unique direction
        // by using a prime-based hash-like function
        (0..dim)
            .map(|i| {
                let x = (seed * 31 + i * 17) as f32;
                (x * 0.0001).sin() * ((seed + i) as f32 * 0.001)
            })
            .collect()
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
    fn store_and_retrieve_embedding() {
        let engine = VectorEngine::new();
        let vector = vec![1.0, 2.0, 3.0];

        engine.store_embedding("test", vector.clone()).unwrap();

        let retrieved = engine.get_embedding("test").unwrap();
        assert_eq!(retrieved, vector);
    }

    #[test]
    fn store_overwrites_existing() {
        let engine = VectorEngine::new();

        engine.store_embedding("key", vec![1.0, 2.0]).unwrap();
        engine.store_embedding("key", vec![3.0, 4.0]).unwrap();

        let retrieved = engine.get_embedding("key").unwrap();
        assert_eq!(retrieved, vec![3.0, 4.0]);
    }

    #[test]
    fn delete_embedding() {
        let engine = VectorEngine::new();

        engine.store_embedding("key", vec![1.0, 2.0]).unwrap();
        assert!(engine.exists("key"));

        engine.delete_embedding("key").unwrap();
        assert!(!engine.exists("key"));
    }

    #[test]
    fn delete_nonexistent_returns_error() {
        let engine = VectorEngine::new();

        let result = engine.delete_embedding("nonexistent");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn get_nonexistent_returns_error() {
        let engine = VectorEngine::new();

        let result = engine.get_embedding("nonexistent");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn empty_vector_returns_error() {
        let engine = VectorEngine::new();

        let result = engine.store_embedding("key", vec![]);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn count_embeddings() {
        let engine = VectorEngine::new();

        assert_eq!(engine.count(), 0);

        engine.store_embedding("a", vec![1.0]).unwrap();
        engine.store_embedding("b", vec![2.0]).unwrap();
        engine.store_embedding("c", vec![3.0]).unwrap();

        assert_eq!(engine.count(), 3);
    }

    #[test]
    fn search_similar_basic() {
        let engine = VectorEngine::new();

        // Store some vectors
        engine.store_embedding("a", vec![1.0, 0.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![1.0, 1.0, 0.0]).unwrap();

        // Search for similar to [1, 0, 0]
        let results = engine.search_similar(&[1.0, 0.0, 0.0], 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be "a" with score 1.0 (exact match)
        assert_eq!(results[0].key, "a");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_similar_top_k() {
        let engine = VectorEngine::new();

        for i in 0..10 {
            engine
                .store_embedding(&format!("v{}", i), vec![i as f32, 0.0])
                .unwrap();
        }

        let results = engine.search_similar(&[5.0, 0.0], 3).unwrap();

        // Should return exactly 3 results
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_similar_fewer_than_k() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0]).unwrap();

        let results = engine.search_similar(&[1.0, 0.0], 10).unwrap();

        // Should return only 2 (all available)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_similar_empty_query_error() {
        let engine = VectorEngine::new();

        let result = engine.search_similar(&[], 5);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_similar_zero_top_k_error() {
        let engine = VectorEngine::new();

        let result = engine.search_similar(&[1.0, 0.0], 0);
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let score = VectorEngine::compute_similarity(&v, &v).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = VectorEngine::compute_similarity(&a, &b).unwrap();
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let score = VectorEngine::compute_similarity(&a, &b).unwrap();
        assert!((score - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_normalized_vectors() {
        // cos(45deg) = sqrt(2)/2 ≈ 0.707
        let a = normalize(&[1.0, 0.0]);
        let b = normalize(&[1.0, 1.0]);
        let score = VectorEngine::compute_similarity(&a, &b).unwrap();
        let expected = (2.0_f32).sqrt() / 2.0;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = VectorEngine::compute_similarity(&a, &b);
        assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let score = VectorEngine::compute_similarity(&a, &b).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn search_skips_dimension_mismatch() {
        let engine = VectorEngine::new();

        engine.store_embedding("2d", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("3d", vec![1.0, 0.0, 0.0]).unwrap();

        // Search with 2D query - should only match 2D vector
        let results = engine.search_similar(&[1.0, 0.0], 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "2d");
    }

    #[test]
    fn store_10000_vectors_search() {
        let engine = VectorEngine::new();
        let dim = 128;

        // Store 10,000 vectors
        for i in 0..10000 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        assert_eq!(engine.count(), 10000);

        // Search for similar to a known vector
        let query = create_test_vector(dim, 5000);
        let results = engine.search_similar(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // The most similar should be the exact match
        assert_eq!(results[0].key, "v5000");
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn high_dimensional_768() {
        let engine = VectorEngine::new();
        let dim = 768;

        // Store a few high-dimensional vectors
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        let query = create_test_vector(dim, 50);
        let results = engine.search_similar(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "v50");
    }

    #[test]
    fn high_dimensional_1536() {
        let engine = VectorEngine::new();
        let dim = 1536;

        // Store a few high-dimensional vectors (OpenAI embedding size)
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        let query = create_test_vector(dim, 75);
        let results = engine.search_similar(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].key, "v75");
    }

    #[test]
    fn similarity_scores_mathematically_correct() {
        let engine = VectorEngine::new();

        // Store vectors with known similarity relationships
        // Using normalized vectors for easier verification
        engine
            .store_embedding("unit_x", normalize(&[1.0, 0.0, 0.0]))
            .unwrap();
        engine
            .store_embedding("unit_y", normalize(&[0.0, 1.0, 0.0]))
            .unwrap();
        engine
            .store_embedding("unit_z", normalize(&[0.0, 0.0, 1.0]))
            .unwrap();
        engine
            .store_embedding("diag_xy", normalize(&[1.0, 1.0, 0.0]))
            .unwrap();
        engine
            .store_embedding("neg_x", normalize(&[-1.0, 0.0, 0.0]))
            .unwrap();

        let query = normalize(&[1.0, 0.0, 0.0]);
        let results = engine.search_similar(&query, 5).unwrap();

        // Verify each score mathematically
        for result in &results {
            match result.key.as_str() {
                "unit_x" => assert!((result.score - 1.0).abs() < 1e-6), // cos(0) = 1
                "unit_y" | "unit_z" => assert!(result.score.abs() < 1e-6), // cos(90) = 0
                "diag_xy" => {
                    let expected = (2.0_f32).sqrt() / 2.0; // cos(45) ≈ 0.707
                    assert!((result.score - expected).abs() < 1e-6);
                },
                "neg_x" => assert!((result.score - (-1.0)).abs() < 1e-6), // cos(180) = -1
                _ => panic!("Unexpected key: {}", result.key),
            }
        }
    }

    #[test]
    fn list_keys() {
        let engine = VectorEngine::new();

        engine.store_embedding("alpha", vec![1.0]).unwrap();
        engine.store_embedding("beta", vec![2.0]).unwrap();
        engine.store_embedding("gamma", vec![3.0]).unwrap();

        let mut keys = engine.list_keys();
        keys.sort();

        assert_eq!(keys, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn clear_all() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0]).unwrap();
        engine.store_embedding("b", vec![2.0]).unwrap();

        assert_eq!(engine.count(), 2);

        engine.clear().unwrap();

        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn dimension() {
        let engine = VectorEngine::new();

        assert_eq!(engine.dimension(), None);

        engine.store_embedding("test", vec![1.0, 2.0, 3.0]).unwrap();

        assert_eq!(engine.dimension(), Some(3));
    }

    #[test]
    fn default_trait() {
        let engine = VectorEngine::default();
        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn with_store_constructor() {
        let store = TensorStore::new();
        let engine = VectorEngine::with_store(store);
        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn error_display() {
        assert_eq!(
            format!("{}", VectorError::NotFound("test".into())),
            "Embedding not found: test"
        );
        assert_eq!(
            format!(
                "{}",
                VectorError::DimensionMismatch {
                    expected: 3,
                    got: 5
                }
            ),
            "Dimension mismatch: expected 3, got 5"
        );
        assert_eq!(
            format!("{}", VectorError::EmptyVector),
            "Empty vector provided"
        );
        assert_eq!(
            format!("{}", VectorError::InvalidTopK),
            "Invalid top_k value (must be > 0)"
        );
        assert_eq!(
            format!("{}", VectorError::StorageError("test".into())),
            "Storage error: test"
        );
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = VectorError::NotFound("test".into());
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    #[test]
    fn error_is_error_trait() {
        let error: Box<dyn std::error::Error> = Box::new(VectorError::EmptyVector);
        assert!(error.to_string().contains("Empty"));
    }

    #[test]
    fn search_result_clone_and_eq() {
        let r1 = SearchResult::new("key".into(), 0.5);
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }

    #[test]
    fn search_zero_query_vector() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();

        // Zero vector query
        let results = engine.search_similar(&[0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_no_embeddings() {
        let engine = VectorEngine::new();
        let results = engine.search_similar(&[1.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn exists_check() {
        let engine = VectorEngine::new();

        assert!(!engine.exists("key"));
        engine.store_embedding("key", vec![1.0]).unwrap();
        assert!(engine.exists("key"));
    }

    // HNSW tests

    #[test]
    fn hnsw_basic_insert_and_search() {
        let index = HNSWIndex::new();

        // Insert a few vectors
        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);
        index.insert(vec![0.0, 0.0, 1.0]);

        assert_eq!(index.len(), 3);

        // Search for similar to [1, 0, 0]
        let results = index.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);

        // First result should have highest similarity to [1, 0, 0]
        assert_eq!(results[0].0, 0); // node 0 is [1, 0, 0]
        assert!((results[0].1 - 1.0).abs() < 1e-6); // similarity ≈ 1.0
    }

    #[test]
    fn hnsw_empty_search() {
        let index = HNSWIndex::new();
        let results = index.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn hnsw_many_vectors() {
        let index = HNSWIndex::new();
        let dim = 64;

        // Insert 1000 vectors
        for i in 0..1000 {
            let vector = create_test_vector(dim, i);
            index.insert(vector);
        }

        assert_eq!(index.len(), 1000);

        // Search for a known vector
        let query = create_test_vector(dim, 500);
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
        // HNSW is approximate - the exact match should be in top results with high score
        // Find node 500 in results
        let exact_match = results.iter().find(|(id, _)| *id == 500);
        assert!(
            exact_match.is_some(),
            "Expected to find node 500 in top 10 results"
        );
        // The similarity should be very high (close to 1.0)
        assert!(exact_match.unwrap().1 > 0.99);
    }

    #[test]
    fn hnsw_high_recall() {
        let config = HNSWConfig::high_recall();
        let index = HNSWIndex::with_config(config);
        let dim = 32;

        // Insert vectors
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            index.insert(vector);
        }

        // High recall config should find exact match
        let query = create_test_vector(dim, 42);
        let results = index.search(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn hnsw_high_speed() {
        let config = HNSWConfig::high_speed();
        let index = HNSWIndex::with_config(config);
        let dim = 32;

        // Insert vectors
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            index.insert(vector);
        }

        assert_eq!(index.len(), 100);
    }

    #[test]
    fn hnsw_get_vector() {
        let index = HNSWIndex::new();

        let original = vec![1.0, 2.0, 3.0];
        let id = index.insert(original.clone());

        let retrieved = index.get_vector(id);
        assert_eq!(retrieved, Some(original));
    }

    #[test]
    fn hnsw_search_with_ef() {
        let index = HNSWIndex::new();
        let dim = 32;

        for i in 0..100 {
            index.insert(create_test_vector(dim, i));
        }

        let query = create_test_vector(dim, 50);

        // Higher ef should give better recall
        let results_low = index.search_with_ef(&query, 5, 10);
        let results_high = index.search_with_ef(&query, 5, 200);

        assert_eq!(results_low.len(), 5);
        assert_eq!(results_high.len(), 5);

        // Both should find the exact match in top 5
        assert!(results_high.iter().any(|(id, _)| *id == 50));
    }

    #[test]
    fn hnsw_default_trait() {
        let index = HNSWIndex::default();
        assert!(index.is_empty());
    }

    #[test]
    fn hnsw_config_default() {
        let config = HNSWConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m0, 32);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
    }

    #[test]
    fn engine_build_hnsw_index() {
        let engine = VectorEngine::new();
        let dim = 32;

        // Store some vectors
        for i in 0..50 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        // Build HNSW index
        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();

        assert_eq!(index.len(), 50);
        assert_eq!(key_mapping.len(), 50);
    }

    #[test]
    fn engine_search_with_hnsw() {
        let engine = VectorEngine::new();
        let dim = 32;

        // Store vectors
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            engine
                .store_embedding(&format!("vec{}", i), vector)
                .unwrap();
        }

        // Build index
        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();

        // Search
        let query = create_test_vector(dim, 42);
        let results = engine
            .search_with_hnsw(&index, &key_mapping, &query, 5)
            .unwrap();

        assert_eq!(results.len(), 5);
        // Should find the exact match
        assert!(results.iter().any(|r| r.key.contains("42")));
    }

    #[test]
    fn engine_hnsw_empty_query_error() {
        let engine = VectorEngine::new();
        let index = HNSWIndex::new();
        let key_mapping: Vec<String> = vec![];

        let result = engine.search_with_hnsw(&index, &key_mapping, &[], 5);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn engine_hnsw_zero_top_k_error() {
        let engine = VectorEngine::new();
        let index = HNSWIndex::new();
        let key_mapping: Vec<String> = vec![];

        let result = engine.search_with_hnsw(&index, &key_mapping, &[1.0], 0);
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    // Unified Entity Mode tests

    #[test]
    fn entity_embedding_store_and_retrieve() {
        let engine = VectorEngine::new();
        let vector = vec![1.0, 2.0, 3.0];

        engine
            .set_entity_embedding("user:1", vector.clone())
            .unwrap();

        let retrieved = engine.get_entity_embedding("user:1").unwrap();
        assert_eq!(retrieved, vector);
    }

    #[test]
    fn entity_embedding_preserves_other_fields() {
        let store = TensorStore::new();

        let mut data = TensorData::new();
        data.set(
            "name",
            TensorValue::Scalar(tensor_store::ScalarValue::String("Alice".into())),
        );
        store.put("user:1", data).unwrap();

        let engine = VectorEngine::with_store(store);
        engine
            .set_entity_embedding("user:1", vec![0.1, 0.2])
            .unwrap();

        let tensor = engine.store.get("user:1").unwrap();
        assert!(tensor.has("name"));
        assert!(tensor.has(fields::EMBEDDING));
    }

    #[test]
    fn entity_has_embedding_check() {
        let engine = VectorEngine::new();

        assert!(!engine.entity_has_embedding("user:1"));

        engine
            .set_entity_embedding("user:1", vec![1.0, 2.0])
            .unwrap();
        assert!(engine.entity_has_embedding("user:1"));
    }

    #[test]
    fn entity_embedding_remove() {
        let engine = VectorEngine::new();

        engine
            .set_entity_embedding("user:1", vec![1.0, 2.0])
            .unwrap();
        assert!(engine.entity_has_embedding("user:1"));

        engine.remove_entity_embedding("user:1").unwrap();
        assert!(!engine.entity_has_embedding("user:1"));
    }

    #[test]
    fn entity_embedding_remove_nonexistent_error() {
        let engine = VectorEngine::new();
        let result = engine.remove_entity_embedding("user:999");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn entity_embedding_get_nonexistent_error() {
        let engine = VectorEngine::new();
        let result = engine.get_entity_embedding("user:999");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn entity_embedding_empty_vector_error() {
        let engine = VectorEngine::new();
        let result = engine.set_entity_embedding("user:1", vec![]);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_entities_basic() {
        let engine = VectorEngine::new();

        engine
            .set_entity_embedding("user:1", vec![1.0, 0.0, 0.0])
            .unwrap();
        engine
            .set_entity_embedding("user:2", vec![0.0, 1.0, 0.0])
            .unwrap();
        engine
            .set_entity_embedding("user:3", vec![1.0, 1.0, 0.0])
            .unwrap();

        let results = engine.search_entities(&[1.0, 0.0, 0.0], 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "user:1");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_entities_filters_non_embeddings() {
        let store = TensorStore::new();

        let mut user1 = TensorData::new();
        user1.set(fields::EMBEDDING, TensorValue::Vector(vec![1.0, 0.0]));
        store.put("user:1", user1).unwrap();

        let mut user2 = TensorData::new();
        user2.set(
            "name",
            TensorValue::Scalar(tensor_store::ScalarValue::String("Bob".into())),
        );
        store.put("user:2", user2).unwrap();

        let engine = VectorEngine::with_store(store);
        let results = engine.search_entities(&[1.0, 0.0], 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "user:1");
    }

    #[test]
    fn scan_entities_with_embeddings() {
        let engine = VectorEngine::new();

        engine
            .set_entity_embedding("user:1", vec![1.0, 2.0])
            .unwrap();
        engine
            .set_entity_embedding("user:2", vec![3.0, 4.0])
            .unwrap();

        let keys = engine.scan_entities_with_embeddings();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn count_entities_with_embeddings() {
        let engine = VectorEngine::new();

        assert_eq!(engine.count_entities_with_embeddings(), 0);

        engine.set_entity_embedding("user:1", vec![1.0]).unwrap();
        engine.set_entity_embedding("user:2", vec![2.0]).unwrap();

        assert_eq!(engine.count_entities_with_embeddings(), 2);
    }

    #[test]
    fn search_entities_empty_query_error() {
        let engine = VectorEngine::new();
        let result = engine.search_entities(&[], 5);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_entities_zero_top_k_error() {
        let engine = VectorEngine::new();
        let result = engine.search_entities(&[1.0], 0);
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    #[test]
    fn search_entities_zero_query_returns_empty() {
        let engine = VectorEngine::new();
        engine
            .set_entity_embedding("user:1", vec![1.0, 0.0])
            .unwrap();

        let results = engine.search_entities(&[0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn search_with_metric_cosine() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.707, 0.707]).unwrap();
        engine.store_embedding("c", vec![0.0, 1.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[1.0, 0.0], 3, DistanceMetric::Cosine)
            .unwrap();

        assert_eq!(results.len(), 3);
        // Cosine: identical vector should have highest score
        assert_eq!(results[0].key, "a");
        assert!((results[0].score - 1.0).abs() < 0.01);
    }

    #[test]
    fn search_with_metric_dot_product() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![2.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![0.5, 0.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[1.0, 0.0], 3, DistanceMetric::DotProduct)
            .unwrap();

        assert_eq!(results.len(), 3);
        // Dot product: larger magnitude vector has higher score
        assert_eq!(results[0].key, "b");
        assert!((results[0].score - 2.0).abs() < 0.01);
    }

    #[test]
    fn search_with_metric_euclidean() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![2.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![10.0, 0.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[1.0, 0.0], 3, DistanceMetric::Euclidean)
            .unwrap();

        assert_eq!(results.len(), 3);
        // Euclidean: closest vector (distance 0) should have highest score
        assert_eq!(results[0].key, "a");
        // Score = 1 / (1 + 0) = 1.0
        assert!((results[0].score - 1.0).abs() < 0.01);
        // "b" is distance 1, score = 1 / (1 + 1) = 0.5
        assert_eq!(results[1].key, "b");
        assert!((results[1].score - 0.5).abs() < 0.01);
    }

    #[test]
    fn search_with_metric_empty_query_error() {
        let engine = VectorEngine::new();
        let result = engine.search_similar_with_metric(&[], 5, DistanceMetric::Cosine);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_with_metric_zero_top_k_error() {
        let engine = VectorEngine::new();
        let result = engine.search_similar_with_metric(&[1.0], 0, DistanceMetric::Cosine);
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    #[test]
    fn search_with_metric_zero_query() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[0.0, 0.0], 5, DistanceMetric::Cosine)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_with_metric_zero_query_euclidean() {
        let engine = VectorEngine::new();
        engine.store_embedding("origin", vec![0.0, 0.0]).unwrap();
        engine.store_embedding("unit", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("far", vec![10.0, 0.0]).unwrap();

        // Zero query should work for Euclidean (finds vectors closest to origin)
        let results = engine
            .search_similar_with_metric(&[0.0, 0.0], 3, DistanceMetric::Euclidean)
            .unwrap();

        assert_eq!(results.len(), 3);
        // Origin is closest to [0,0] (distance 0, score 1.0)
        assert_eq!(results[0].key, "origin");
        assert!((results[0].score - 1.0).abs() < 0.01);
        // Unit is next (distance 1, score 0.5)
        assert_eq!(results[1].key, "unit");
        assert!((results[1].score - 0.5).abs() < 0.01);
    }

    #[test]
    fn euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let dist = VectorEngine::euclidean_distance(&a, &a);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance_unit() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let dist = VectorEngine::euclidean_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance_pythagoras() {
        // 3-4-5 triangle
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = VectorEngine::euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    // Sparse vector tests

    #[test]
    fn sparse_vector_storage_and_retrieval() {
        let engine = VectorEngine::new();

        // Create a sparse vector (>50% zeros)
        let mut sparse = vec![0.0f32; 100];
        sparse[0] = 1.0;
        sparse[50] = 2.0;
        sparse[99] = 3.0;

        engine.store_embedding("sparse", sparse.clone()).unwrap();
        let retrieved = engine.get_embedding("sparse").unwrap();

        assert_eq!(retrieved.len(), sparse.len());
        assert_eq!(retrieved[0], 1.0);
        assert_eq!(retrieved[50], 2.0);
        assert_eq!(retrieved[99], 3.0);
    }

    #[test]
    fn sparse_vector_search() {
        let engine = VectorEngine::new();

        // Create sparse vectors with 97% zeros (3 non-zeros in 100 elements)
        let mut v1 = vec![0.0f32; 100];
        v1[0] = 1.0;
        v1[1] = 0.0;
        v1[2] = 0.0;

        let mut v2 = vec![0.0f32; 100];
        v2[0] = 0.707;
        v2[1] = 0.707;

        let mut v3 = vec![0.0f32; 100];
        v3[0] = 0.0;
        v3[1] = 1.0;

        engine.store_embedding("v1", v1).unwrap();
        engine.store_embedding("v2", v2).unwrap();
        engine.store_embedding("v3", v3).unwrap();

        // Query with [1, 0, 0, ...]
        let mut query = vec![0.0f32; 100];
        query[0] = 1.0;

        let results = engine.search_similar(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // v1 should be the best match
        assert_eq!(results[0].key, "v1");
        assert!((results[0].score - 1.0).abs() < 0.01);
    }

    #[test]
    fn sparse_entity_embedding() {
        let engine = VectorEngine::new();

        let mut sparse = vec![0.0f32; 100];
        sparse[10] = 5.0;
        sparse[20] = -3.0;

        engine
            .set_entity_embedding("entity:1", sparse.clone())
            .unwrap();
        let retrieved = engine.get_entity_embedding("entity:1").unwrap();

        assert_eq!(retrieved.len(), 100);
        assert_eq!(retrieved[10], 5.0);
        assert_eq!(retrieved[20], -3.0);
        assert_eq!(retrieved[0], 0.0);
    }

    #[test]
    fn sparse_detection_threshold() {
        // Exactly 50% zeros should use sparse at 0.5 threshold
        let half_sparse: Vec<f32> = (0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect();
        assert!(VectorEngine::should_use_sparse_with_threshold(
            &half_sparse,
            0.5
        ));

        // Less than 50% zeros should use dense at 0.5 threshold
        let mostly_dense: Vec<f32> = (0..100).map(|i| if i < 40 { 0.0 } else { 1.0 }).collect();
        assert!(!VectorEngine::should_use_sparse_with_threshold(
            &mostly_dense,
            0.5
        ));

        // 97% zeros should use sparse
        let very_sparse: Vec<f32> = (0..100).map(|i| if i < 3 { 1.0 } else { 0.0 }).collect();
        assert!(VectorEngine::should_use_sparse_with_threshold(
            &very_sparse,
            0.5
        ));
    }

    #[test]
    fn sparse_search_with_metric() {
        let engine = VectorEngine::new();

        // Sparse vectors
        let mut v1 = vec![0.0f32; 100];
        v1[0] = 1.0;

        let mut v2 = vec![0.0f32; 100];
        v2[0] = 2.0;

        engine.store_embedding("v1", v1).unwrap();
        engine.store_embedding("v2", v2).unwrap();

        let mut query = vec![0.0f32; 100];
        query[0] = 1.0;

        let results = engine
            .search_similar_with_metric(&query, 2, DistanceMetric::Euclidean)
            .unwrap();

        assert_eq!(results.len(), 2);
        // v1 is closest (distance 0)
        assert_eq!(results[0].key, "v1");
    }

    #[test]
    fn search_entities_with_sparse() {
        let engine = VectorEngine::new();

        // Create sparse entity embeddings
        let mut e1 = vec![0.0f32; 100];
        e1[0] = 1.0;

        let mut e2 = vec![0.0f32; 100];
        e2[1] = 1.0;

        engine.set_entity_embedding("user:1", e1).unwrap();
        engine.set_entity_embedding("user:2", e2).unwrap();

        let mut query = vec![0.0f32; 100];
        query[0] = 1.0;

        let results = engine.search_entities(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "user:1");
        assert!((results[0].score - 1.0).abs() < 0.01);
    }

    // ========== Configuration Tests ==========

    #[test]
    fn config_default() {
        let config = VectorEngineConfig::default();
        assert_eq!(config.default_dimension, None);
        assert!((config.sparse_threshold - 0.5).abs() < 1e-6);
        assert_eq!(config.parallel_threshold, 5000);
        assert_eq!(config.default_metric, DistanceMetric::Cosine);
    }

    #[test]
    fn config_high_throughput() {
        let config = VectorEngineConfig::high_throughput();
        assert_eq!(config.parallel_threshold, 1000);
    }

    #[test]
    fn config_low_memory() {
        let config = VectorEngineConfig::low_memory();
        assert!((config.sparse_threshold - 0.3).abs() < 1e-6);
    }

    #[test]
    fn config_validate_valid() {
        let config = VectorEngineConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validate_invalid_sparse_threshold() {
        let config = VectorEngineConfig {
            sparse_threshold: 1.5,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(VectorError::ConfigurationError(_))
        ));
    }

    #[test]
    fn config_validate_invalid_parallel_threshold() {
        let config = VectorEngineConfig {
            parallel_threshold: 0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(VectorError::ConfigurationError(_))
        ));
    }

    #[test]
    fn engine_with_config() {
        let config = VectorEngineConfig::high_throughput();
        let engine = VectorEngine::with_config(config).unwrap();
        assert_eq!(engine.config().parallel_threshold, 1000);
    }

    #[test]
    fn engine_with_store_and_config() {
        let store = TensorStore::new();
        let config = VectorEngineConfig::low_memory();
        let engine = VectorEngine::with_store_and_config(store, config).unwrap();
        assert!((engine.config().sparse_threshold - 0.3).abs() < 1e-6);
    }

    // ========== Batch Operations Tests ==========

    #[test]
    fn batch_store_embeddings_basic() {
        let engine = VectorEngine::new();
        let inputs = vec![
            EmbeddingInput::new("a", vec![1.0, 0.0]),
            EmbeddingInput::new("b", vec![0.0, 1.0]),
            EmbeddingInput::new("c", vec![1.0, 1.0]),
        ];

        let result = engine.batch_store_embeddings(inputs).unwrap();
        assert_eq!(result.count, 3);
        assert_eq!(result.stored_keys, vec!["a", "b", "c"]);
        assert_eq!(engine.count(), 3);
    }

    #[test]
    fn batch_store_embeddings_empty() {
        let engine = VectorEngine::new();
        let result = engine.batch_store_embeddings(vec![]).unwrap();
        assert_eq!(result.count, 0);
        assert!(result.stored_keys.is_empty());
    }

    #[test]
    fn batch_store_embeddings_validation_error() {
        let engine = VectorEngine::new();
        let inputs = vec![
            EmbeddingInput::new("a", vec![1.0, 0.0]),
            EmbeddingInput::new("b", vec![]), // Empty vector
        ];

        let result = engine.batch_store_embeddings(inputs);
        assert!(matches!(
            result,
            Err(VectorError::BatchValidationError { index: 1, .. })
        ));
    }

    #[test]
    fn batch_delete_embeddings_basic() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0]).unwrap();
        engine.store_embedding("b", vec![2.0]).unwrap();
        engine.store_embedding("c", vec![3.0]).unwrap();

        let count = engine
            .batch_delete_embeddings(vec!["a".to_string(), "b".to_string()])
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(engine.count(), 1);
        assert!(engine.exists("c"));
    }

    #[test]
    fn batch_delete_embeddings_empty() {
        let engine = VectorEngine::new();
        let count = engine.batch_delete_embeddings(vec![]).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn batch_delete_embeddings_nonexistent() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0]).unwrap();

        let count = engine
            .batch_delete_embeddings(vec!["a".to_string(), "nonexistent".to_string()])
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn embedding_input_new() {
        let input = EmbeddingInput::new("test", vec![1.0, 2.0]);
        assert_eq!(input.key, "test");
        assert_eq!(input.vector, vec![1.0, 2.0]);
    }

    #[test]
    fn batch_result_new() {
        let result = BatchResult::new(vec!["a".to_string(), "b".to_string()]);
        assert_eq!(result.count, 2);
        assert_eq!(result.stored_keys, vec!["a", "b"]);
    }

    // ========== Pagination Tests ==========

    #[test]
    fn pagination_new() {
        let p = Pagination::new(10, 20);
        assert_eq!(p.skip, 10);
        assert_eq!(p.limit, Some(20));
        assert!(!p.count_total);
    }

    #[test]
    fn pagination_with_total() {
        let p = Pagination::new(0, 10).with_total();
        assert!(p.count_total);
    }

    #[test]
    fn pagination_skip_only() {
        let p = Pagination::skip_only(5);
        assert_eq!(p.skip, 5);
        assert_eq!(p.limit, None);
    }

    #[test]
    fn list_keys_paginated_basic() {
        let engine = VectorEngine::new();
        for i in 0..10 {
            engine
                .store_embedding(&format!("v{:02}", i), vec![i as f32])
                .unwrap();
        }

        let result = engine.list_keys_paginated(Pagination::new(0, 3));
        assert_eq!(result.items.len(), 3);
        assert!(result.has_more);
        assert_eq!(result.total_count, None);
    }

    #[test]
    fn list_keys_paginated_with_total() {
        let engine = VectorEngine::new();
        for i in 0..10 {
            engine
                .store_embedding(&format!("v{:02}", i), vec![i as f32])
                .unwrap();
        }

        let result = engine.list_keys_paginated(Pagination::new(0, 5).with_total());
        assert_eq!(result.items.len(), 5);
        assert_eq!(result.total_count, Some(10));
        assert!(result.has_more);
    }

    #[test]
    fn list_keys_paginated_skip() {
        let engine = VectorEngine::new();
        for i in 0..10 {
            engine
                .store_embedding(&format!("v{:02}", i), vec![i as f32])
                .unwrap();
        }

        let result = engine.list_keys_paginated(Pagination::new(8, 5).with_total());
        assert_eq!(result.items.len(), 2);
        assert!(!result.has_more);
    }

    #[test]
    fn search_similar_paginated_basic() {
        let engine = VectorEngine::new();
        for i in 0..10 {
            engine
                .store_embedding(&format!("v{}", i), vec![i as f32, 0.0])
                .unwrap();
        }

        let result = engine
            .search_similar_paginated(&[5.0, 0.0], 10, Pagination::new(0, 3).with_total())
            .unwrap();

        assert_eq!(result.items.len(), 3);
    }

    #[test]
    fn search_entities_paginated_basic() {
        let engine = VectorEngine::new();
        for i in 0..5 {
            engine
                .set_entity_embedding(&format!("user:{}", i), vec![i as f32, 0.0])
                .unwrap();
        }

        let result = engine
            .search_entities_paginated(&[2.0, 0.0], 5, Pagination::new(0, 2).with_total())
            .unwrap();

        assert_eq!(result.items.len(), 2);
    }

    #[test]
    fn paged_result_empty() {
        let result: PagedResult<String> = PagedResult::empty();
        assert!(result.items.is_empty());
        assert_eq!(result.total_count, Some(0));
        assert!(!result.has_more);
    }

    // ========== Error Variants Tests ==========

    #[test]
    fn error_batch_validation_display() {
        let error = VectorError::BatchValidationError {
            index: 5,
            cause: "test error".to_string(),
        };
        assert_eq!(
            format!("{}", error),
            "Batch validation error at index 5: test error"
        );
    }

    #[test]
    fn error_batch_operation_display() {
        let error = VectorError::BatchOperationError {
            index: 3,
            cause: "op failed".to_string(),
        };
        assert_eq!(
            format!("{}", error),
            "Batch operation error at index 3: op failed"
        );
    }

    #[test]
    fn error_configuration_display() {
        let error = VectorError::ConfigurationError("bad config".to_string());
        assert_eq!(format!("{}", error), "Configuration error: bad config");
    }

    #[test]
    fn error_variants_clone_and_eq() {
        let e1 = VectorError::BatchValidationError {
            index: 1,
            cause: "test".to_string(),
        };
        let e2 = e1.clone();
        assert_eq!(e1, e2);

        let e3 = VectorError::ConfigurationError("test".to_string());
        let e4 = e3.clone();
        assert_eq!(e3, e4);
    }

    // ========== HNSW Metric Integration Tests ==========

    #[test]
    fn search_with_hnsw_and_metric_basic() {
        let engine = VectorEngine::new();
        let dim = 32;

        for i in 0..50 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();
        let query = create_test_vector(dim, 25);

        let results = engine
            .search_with_hnsw_and_metric(
                &index,
                &key_mapping,
                &query,
                5,
                ExtendedDistanceMetric::Cosine,
            )
            .unwrap();

        assert_eq!(results.len(), 5);
        // Should find the exact match
        assert!(results.iter().any(|r| r.key == "v25"));
    }

    #[test]
    fn search_with_hnsw_and_metric_euclidean() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![2.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![10.0, 0.0]).unwrap();

        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();

        let results = engine
            .search_with_hnsw_and_metric(
                &index,
                &key_mapping,
                &[1.0, 0.0],
                3,
                ExtendedDistanceMetric::Euclidean,
            )
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "a"); // Closest
    }

    #[test]
    fn search_with_hnsw_and_metric_empty_query() {
        let engine = VectorEngine::new();
        let index = HNSWIndex::new();
        let key_mapping: Vec<String> = vec![];

        let result = engine.search_with_hnsw_and_metric(
            &index,
            &key_mapping,
            &[],
            5,
            ExtendedDistanceMetric::Cosine,
        );
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_with_hnsw_and_metric_zero_top_k() {
        let engine = VectorEngine::new();
        let index = HNSWIndex::new();
        let key_mapping: Vec<String> = vec![];

        let result = engine.search_with_hnsw_and_metric(
            &index,
            &key_mapping,
            &[1.0],
            0,
            ExtendedDistanceMetric::Cosine,
        );
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    // ========== Concurrent Tests ==========

    #[test]
    fn test_concurrent_store_embedding_same_key() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::thread;

        let engine = Arc::new(VectorEngine::new());
        let success = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let eng = Arc::clone(&engine);
                let s = Arc::clone(&success);
                thread::spawn(move || {
                    // Each thread stores with the same key but different value
                    let vector = vec![i as f32, i as f32];
                    if eng.store_embedding("contested", vector).is_ok() {
                        s.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All stores should succeed (last-write-wins)
        assert_eq!(success.load(Ordering::SeqCst), 10);
        assert!(engine.exists("contested"));
    }

    #[test]
    fn test_concurrent_delete_embedding_same_key() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::thread;

        let engine = Arc::new(VectorEngine::new());
        engine.store_embedding("to_delete", vec![1.0, 2.0]).unwrap();

        let success = Arc::new(AtomicUsize::new(0));
        let error = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let eng = Arc::clone(&engine);
                let s = Arc::clone(&success);
                let e = Arc::clone(&error);
                thread::spawn(move || {
                    match eng.delete_embedding("to_delete") {
                        Ok(()) => {
                            s.fetch_add(1, Ordering::SeqCst);
                        },
                        Err(VectorError::NotFound(_)) => {
                            e.fetch_add(1, Ordering::SeqCst);
                        },
                        Err(err) => panic!("unexpected error: {err:?}"),
                    };
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Exactly 1 should succeed, 9 should fail with NotFound
        assert_eq!(success.load(Ordering::SeqCst), 1);
        assert_eq!(error.load(Ordering::SeqCst), 9);
    }

    #[test]
    fn test_concurrent_search_similar() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        use std::thread;

        let engine = Arc::new(VectorEngine::new());

        // Store some vectors
        for i in 0..100 {
            engine
                .store_embedding(&format!("v{}", i), vec![i as f32, 0.0])
                .unwrap();
        }

        // Verify data is stored before starting concurrent reads
        assert_eq!(engine.count(), 100);

        let success_count = Arc::new(AtomicUsize::new(0));

        // 20 threads searching concurrently
        let handles: Vec<_> = (0..20)
            .map(|t| {
                let eng = Arc::clone(&engine);
                let counter = Arc::clone(&success_count);
                thread::spawn(move || {
                    // Run multiple searches per thread
                    for _ in 0..5 {
                        let query = vec![t as f32, 0.0];
                        if let Ok(results) = eng.search_similar(&query, 5) {
                            if !results.is_empty() {
                                counter.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Most searches should succeed (some may transiently see partial state)
        let successes = success_count.load(Ordering::SeqCst);
        assert!(
            successes > 50,
            "Expected at least 50 successful searches, got {}",
            successes
        );
    }

    #[test]
    fn test_concurrent_store_and_search() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(VectorEngine::new());

        // Pre-populate some data
        for i in 0..50 {
            engine
                .store_embedding(&format!("init{}", i), vec![i as f32, 0.0])
                .unwrap();
        }

        // Mixed reads and writes
        let handles: Vec<_> = (0..20)
            .map(|t| {
                let eng = Arc::clone(&engine);
                thread::spawn(move || {
                    if t % 2 == 0 {
                        // Writer
                        for i in 0..10 {
                            let key = format!("t{}_v{}", t, i);
                            eng.store_embedding(&key, vec![t as f32, i as f32]).unwrap();
                        }
                    } else {
                        // Reader
                        for _ in 0..10 {
                            let query = vec![t as f32, 0.0];
                            let _ = eng.search_similar(&query, 5);
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify some data exists
        assert!(engine.count() >= 50);
    }

    #[test]
    fn test_concurrent_batch_operations() {
        use std::sync::Arc;
        use std::thread;

        let engine = Arc::new(VectorEngine::new());

        let handles: Vec<_> = (0..5)
            .map(|t| {
                let eng = Arc::clone(&engine);
                thread::spawn(move || {
                    let inputs: Vec<EmbeddingInput> = (0..20)
                        .map(|i| {
                            EmbeddingInput::new(format!("t{}_b{}", t, i), vec![t as f32, i as f32])
                        })
                        .collect();
                    eng.batch_store_embeddings(inputs).unwrap()
                })
            })
            .collect();

        let results: Vec<BatchResult> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Each batch should have stored 20 items
        for result in &results {
            assert_eq!(result.count, 20);
        }

        // Total should be 100
        assert_eq!(engine.count(), 100);
    }

    // ========== Extended Distance Metric Tests ==========

    #[test]
    fn extended_metric_re_export() {
        // Verify re-export works
        let metric = ExtendedDistanceMetric::Jaccard;
        assert!(metric.higher_is_better());

        let config = GeometricConfig::default();
        assert!((config.cosine_weight - 0.5).abs() < 1e-6);
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn sparse_with_custom_threshold() {
        let config = VectorEngineConfig {
            sparse_threshold: 0.8, // Very high threshold
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        // 70% zeros - should NOT use sparse with 0.8 threshold
        let mostly_zeros: Vec<f32> = (0..100).map(|i| if i < 30 { 1.0 } else { 0.0 }).collect();
        assert!(!engine.should_use_sparse(&mostly_zeros));

        // 90% zeros - should use sparse
        let very_sparse: Vec<f32> = (0..100).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect();
        assert!(engine.should_use_sparse(&very_sparse));
    }

    #[test]
    fn batch_store_large_batch() {
        let engine = VectorEngine::new();
        let inputs: Vec<EmbeddingInput> = (0..150)
            .map(|i| EmbeddingInput::new(format!("v{}", i), vec![i as f32, 0.0]))
            .collect();

        let result = engine.batch_store_embeddings(inputs).unwrap();
        assert_eq!(result.count, 150);
        assert_eq!(engine.count(), 150);
    }

    #[test]
    fn batch_delete_large_batch() {
        let engine = VectorEngine::new();
        for i in 0..150 {
            engine
                .store_embedding(&format!("v{}", i), vec![i as f32])
                .unwrap();
        }

        let keys: Vec<String> = (0..150).map(|i| format!("v{}", i)).collect();
        let count = engine.batch_delete_embeddings(keys).unwrap();
        assert_eq!(count, 150);
        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn pagination_empty_result() {
        let engine = VectorEngine::new();
        let result = engine.list_keys_paginated(Pagination::new(0, 10).with_total());
        assert!(result.items.is_empty());
        assert_eq!(result.total_count, Some(0));
        assert!(!result.has_more);
    }

    #[test]
    fn pagination_skip_past_end() {
        let engine = VectorEngine::new();
        for i in 0..5 {
            engine
                .store_embedding(&format!("v{}", i), vec![i as f32])
                .unwrap();
        }

        let result = engine.list_keys_paginated(Pagination::new(10, 5).with_total());
        assert!(result.items.is_empty());
        assert_eq!(result.total_count, Some(5));
        assert!(!result.has_more);
    }

    // ========== Distance Metric Conversion Tests ==========

    #[test]
    fn distance_metric_conversion_cosine() {
        let simple = DistanceMetric::Cosine;
        let extended: ExtendedDistanceMetric = simple.into();
        assert!(matches!(extended, ExtendedDistanceMetric::Cosine));
    }

    #[test]
    fn distance_metric_conversion_euclidean() {
        let simple = DistanceMetric::Euclidean;
        let extended: ExtendedDistanceMetric = simple.into();
        assert!(matches!(extended, ExtendedDistanceMetric::Euclidean));
    }

    #[test]
    fn distance_metric_conversion_dot_product() {
        let simple = DistanceMetric::DotProduct;
        let extended: ExtendedDistanceMetric = simple.into();
        // DotProduct maps to Cosine (closest equivalent)
        assert!(matches!(extended, ExtendedDistanceMetric::Cosine));
    }

    // ========== Extended Distance Metric Tests ==========

    #[test]
    fn search_with_hnsw_and_metric_angular() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.707, 0.707]).unwrap();
        engine.store_embedding("c", vec![0.0, 1.0]).unwrap();

        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();

        let results = engine
            .search_with_hnsw_and_metric(
                &index,
                &key_mapping,
                &[1.0, 0.0],
                3,
                ExtendedDistanceMetric::Angular,
            )
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "a");
    }

    #[test]
    fn search_with_hnsw_and_metric_jaccard() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![1.0, 0.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![0.0, 0.0, 1.0]).unwrap();

        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();

        let results = engine
            .search_with_hnsw_and_metric(
                &index,
                &key_mapping,
                &[1.0, 1.0, 0.0],
                3,
                ExtendedDistanceMetric::Jaccard,
            )
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "a");
    }

    #[test]
    fn search_with_hnsw_and_metric_overlap() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![1.0, 0.0, 0.0]).unwrap();

        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();

        let results = engine
            .search_with_hnsw_and_metric(
                &index,
                &key_mapping,
                &[1.0, 1.0, 0.0],
                2,
                ExtendedDistanceMetric::Overlap,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_with_hnsw_and_metric_manhattan() {
        let engine = VectorEngine::new();

        engine.store_embedding("origin", vec![0.0, 0.0]).unwrap();
        engine.store_embedding("one", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("two", vec![2.0, 0.0]).unwrap();

        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();

        let results = engine
            .search_with_hnsw_and_metric(
                &index,
                &key_mapping,
                &[0.0, 0.0],
                3,
                ExtendedDistanceMetric::Manhattan,
            )
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "origin");
    }

    // ========== Numeric Edge Cases ==========

    #[test]
    fn store_and_search_with_very_small_values() {
        let engine = VectorEngine::new();

        // Use values small enough to test but large enough to avoid underflow
        // when computing magnitude (squared sum)
        let tiny = vec![1e-18_f32, 1e-18, 1e-18];
        engine.store_embedding("tiny", tiny.clone()).unwrap();

        let results = engine.search_similar(&tiny, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "tiny");
    }

    #[test]
    fn store_and_search_with_large_values() {
        let engine = VectorEngine::new();

        let large = vec![1e30_f32, 1e30, 1e30];
        engine.store_embedding("large", large.clone()).unwrap();

        let results = engine.search_similar(&large, 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_handles_denormalized_floats() {
        let engine = VectorEngine::new();

        let denorm = vec![f32::MIN_POSITIVE / 2.0, 1.0, 0.0];
        engine.store_embedding("denorm", denorm.clone()).unwrap();

        let results = engine.search_similar(&denorm, 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn zero_vector_with_euclidean_metric() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 0.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[0.0, 0.0], 2, DistanceMetric::Euclidean)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "b");
    }

    // ========== Dimension Edge Cases ==========

    #[test]
    fn single_dimension_vector() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0]).unwrap();
        engine.store_embedding("b", vec![2.0]).unwrap();
        engine.store_embedding("c", vec![-1.0]).unwrap();

        let results = engine.search_similar(&[1.0], 3).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "a");
    }

    #[test]
    fn high_dimension_4096() {
        let engine = VectorEngine::new();
        let dim = 4096;

        let v1: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.001).sin()).collect();
        let v2: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.002).sin()).collect();

        engine.store_embedding("v1", v1.clone()).unwrap();
        engine.store_embedding("v2", v2).unwrap();

        let results = engine.search_similar(&v1, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "v1");
    }

    #[test]
    fn mismatched_dimensions_silently_skipped() {
        let engine = VectorEngine::new();

        engine.store_embedding("dim2", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("dim3", vec![1.0, 0.0, 0.0]).unwrap();
        engine
            .store_embedding("dim4", vec![1.0, 0.0, 0.0, 0.0])
            .unwrap();

        let results = engine.search_similar(&[1.0, 0.0], 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "dim2");
    }

    // ========== Config Validation Edge Cases ==========

    #[test]
    fn config_validate_negative_sparse_threshold() {
        let config = VectorEngineConfig {
            sparse_threshold: -0.1,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(VectorError::ConfigurationError(_))
        ));
    }

    #[test]
    fn config_presets_are_valid() {
        assert!(VectorEngineConfig::default().validate().is_ok());
        assert!(VectorEngineConfig::high_throughput().validate().is_ok());
        assert!(VectorEngineConfig::low_memory().validate().is_ok());
    }

    // ========== Scale Tests ==========

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored
    fn scale_100k_vector_search() {
        let engine = VectorEngine::new();
        let dim = 128;

        for i in 0..100_000 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        assert_eq!(engine.count(), 100_000);

        let query = create_test_vector(dim, 50_000);
        let results = engine.search_similar(&query, 10).unwrap();

        assert_eq!(results.len(), 10);
        assert_eq!(results[0].key, "v50000");
    }

    #[test]
    #[ignore] // Run with: cargo test --release -- --ignored
    fn scale_10k_batch_store() {
        let engine = VectorEngine::new();
        let dim = 64;

        let inputs: Vec<EmbeddingInput> = (0..10_000)
            .map(|i| EmbeddingInput::new(format!("v{}", i), create_test_vector(dim, i)))
            .collect();

        let result = engine.batch_store_embeddings(inputs).unwrap();
        assert_eq!(result.count, 10_000);
        assert_eq!(engine.count(), 10_000);
    }

    // ========== Additional Coverage Tests ==========

    #[test]
    fn paged_result_new_with_data() {
        let items = vec!["a".to_string(), "b".to_string()];
        let result = PagedResult::new(items.clone(), Some(10), true);
        assert_eq!(result.items, items);
        assert_eq!(result.total_count, Some(10));
        assert!(result.has_more);
    }

    #[test]
    fn pagination_default() {
        let p = Pagination::default();
        assert_eq!(p.skip, 0);
        assert_eq!(p.limit, None);
        assert!(!p.count_total);
    }

    #[test]
    fn sparse_detection_empty_vector() {
        assert!(!VectorEngine::should_use_sparse_with_threshold(&[], 0.5));
    }

    #[test]
    fn extract_vector_non_vector_type() {
        let value = TensorValue::Scalar(tensor_store::ScalarValue::Int(42));
        assert!(VectorEngine::extract_vector(&value).is_none());
    }

    #[test]
    fn store_accessor() {
        let engine = VectorEngine::new();
        engine.store_embedding("test", vec![1.0, 2.0]).unwrap();

        let store = engine.store();
        assert!(store.exists("emb:test"));
    }

    #[test]
    fn entity_embedding_no_embedding_field() {
        let engine = VectorEngine::new();
        let store = engine.store();

        let mut data = TensorData::new();
        data.set(
            "name",
            TensorValue::Scalar(tensor_store::ScalarValue::String("test".into())),
        );
        store.put("entity:1", data).unwrap();

        let result = engine.get_entity_embedding("entity:1");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn remove_entity_embedding_no_field() {
        let engine = VectorEngine::new();
        let store = engine.store();

        let mut data = TensorData::new();
        data.set(
            "name",
            TensorValue::Scalar(tensor_store::ScalarValue::String("test".into())),
        );
        store.put("entity:1", data).unwrap();

        let result = engine.remove_entity_embedding("entity:1");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    // ========== Production Hardening Tests ==========

    #[test]
    fn config_validate_invalid_max_dimension_zero() {
        let config = VectorEngineConfig {
            max_dimension: Some(0),
            ..Default::default()
        };
        let result = config.validate();
        assert!(matches!(
            result,
            Err(VectorError::ConfigurationError(msg)) if msg.contains("max_dimension")
        ));
    }

    #[test]
    fn config_validate_invalid_max_keys_per_scan_zero() {
        let config = VectorEngineConfig {
            max_keys_per_scan: Some(0),
            ..Default::default()
        };
        let result = config.validate();
        assert!(matches!(
            result,
            Err(VectorError::ConfigurationError(msg)) if msg.contains("max_keys_per_scan")
        ));
    }

    #[test]
    fn config_validate_invalid_batch_parallel_threshold_zero() {
        let config = VectorEngineConfig {
            batch_parallel_threshold: 0,
            ..Default::default()
        };
        let result = config.validate();
        assert!(matches!(
            result,
            Err(VectorError::ConfigurationError(msg)) if msg.contains("batch_parallel_threshold")
        ));
    }

    #[test]
    fn with_config_validates_and_returns_error() {
        let config = VectorEngineConfig {
            max_dimension: Some(0),
            ..Default::default()
        };
        let result = VectorEngine::with_config(config);
        assert!(matches!(result, Err(VectorError::ConfigurationError(_))));
    }

    #[test]
    fn with_store_and_config_validates_and_returns_error() {
        let store = TensorStore::new();
        let config = VectorEngineConfig {
            batch_parallel_threshold: 0,
            ..Default::default()
        };
        let result = VectorEngine::with_store_and_config(store, config);
        assert!(matches!(result, Err(VectorError::ConfigurationError(_))));
    }

    #[test]
    fn with_config_valid_succeeds() {
        let config = VectorEngineConfig {
            max_dimension: Some(1024),
            max_keys_per_scan: Some(1000),
            ..Default::default()
        };
        let result = VectorEngine::with_config(config);
        assert!(result.is_ok());
    }

    #[test]
    fn list_keys_bounded_respects_limit() {
        let config = VectorEngineConfig {
            max_keys_per_scan: Some(3),
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        for i in 0..10 {
            engine
                .store_embedding(&format!("v{}", i), vec![i as f32])
                .unwrap();
        }

        let keys = engine.list_keys_bounded();
        assert_eq!(keys.len(), 3);
    }

    #[test]
    fn list_keys_bounded_no_limit() {
        let engine = VectorEngine::new();

        for i in 0..10 {
            engine
                .store_embedding(&format!("v{}", i), vec![i as f32])
                .unwrap();
        }

        let keys = engine.list_keys_bounded();
        assert_eq!(keys.len(), 10);
    }

    #[test]
    fn search_similar_rejects_oversized_dimension() {
        let config = VectorEngineConfig {
            max_dimension: Some(10),
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        let oversized_query = vec![0.0; 20];
        let result = engine.search_similar(&oversized_query, 5);

        assert!(matches!(
            result,
            Err(VectorError::DimensionMismatch {
                expected: 10,
                got: 20
            })
        ));
    }

    #[test]
    fn store_embedding_rejects_oversized_dimension() {
        let config = VectorEngineConfig {
            max_dimension: Some(5),
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        let oversized = vec![0.0; 10];
        let result = engine.store_embedding("key", oversized);

        assert!(matches!(
            result,
            Err(VectorError::DimensionMismatch {
                expected: 5,
                got: 10
            })
        ));
    }

    #[test]
    fn set_entity_embedding_rejects_oversized_dimension() {
        let config = VectorEngineConfig {
            max_dimension: Some(5),
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        let oversized = vec![0.0; 10];
        let result = engine.set_entity_embedding("entity:1", oversized);

        assert!(matches!(
            result,
            Err(VectorError::DimensionMismatch {
                expected: 5,
                got: 10
            })
        ));
    }

    #[test]
    fn search_entities_rejects_oversized_dimension() {
        let config = VectorEngineConfig {
            max_dimension: Some(5),
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        let oversized_query = vec![0.0; 10];
        let result = engine.search_entities(&oversized_query, 5);

        assert!(matches!(
            result,
            Err(VectorError::DimensionMismatch {
                expected: 5,
                got: 10
            })
        ));
    }

    #[test]
    fn build_hnsw_index_validates_dimension_consistency() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 2.0, 3.0]).unwrap();
        engine.store_embedding("b", vec![4.0, 5.0, 6.0]).unwrap();

        let result = engine.build_hnsw_index_default();
        assert!(result.is_ok());

        let (index, keys) = result.unwrap();
        assert_eq!(index.len(), 2);
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn build_hnsw_index_empty_store() {
        let engine = VectorEngine::new();

        let result = engine.build_hnsw_index_default();
        assert!(result.is_ok());

        let (index, keys) = result.unwrap();
        assert!(index.is_empty());
        assert!(keys.is_empty());
    }

    #[test]
    fn build_hnsw_index_rejects_exceeding_max_dimension() {
        let config = VectorEngineConfig {
            max_dimension: Some(5),
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        // Store a valid small embedding first
        engine.store_embedding("a", vec![1.0, 2.0]).unwrap();

        // The dimension 2 is within max_dimension 5, so build should succeed
        let result = engine.build_hnsw_index_default();
        assert!(result.is_ok());
    }

    #[test]
    fn estimate_hnsw_memory_empty_store() {
        let engine = VectorEngine::new();
        let estimate = engine.estimate_hnsw_memory().unwrap();
        assert_eq!(estimate, 0);
    }

    #[test]
    fn estimate_hnsw_memory_calculation() {
        let engine = VectorEngine::new();

        // Store 10 embeddings of dimension 128
        for i in 0..10 {
            engine
                .store_embedding(&format!("v{}", i), vec![1.0; 128])
                .unwrap();
        }

        let estimate = engine.estimate_hnsw_memory().unwrap();

        // Expected: 10 * 128 * 4 (vectors) + 10 * 16 * 2 * 8 (graph) + 10 * 32 (keys)
        // = 5120 + 2560 + 320 = 8000
        let expected = 10 * 128 * 4 + 10 * 16 * 2 * 8 + 10 * 32;
        assert_eq!(estimate, expected);
    }

    #[test]
    fn batch_uses_config_parallel_threshold() {
        let config = VectorEngineConfig {
            batch_parallel_threshold: 5,
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        // Store 10 items (above threshold of 5)
        let inputs: Vec<EmbeddingInput> = (0..10)
            .map(|i| EmbeddingInput::new(format!("v{}", i), vec![i as f32, 0.0]))
            .collect();

        let result = engine.batch_store_embeddings(inputs).unwrap();
        assert_eq!(result.count, 10);
    }

    #[test]
    fn config_low_memory_has_bounds() {
        let config = VectorEngineConfig::low_memory();
        assert_eq!(config.max_dimension, Some(4096));
        assert_eq!(config.max_keys_per_scan, Some(10_000));
    }

    #[test]
    fn config_default_has_new_fields() {
        let config = VectorEngineConfig::default();
        assert_eq!(config.max_dimension, None);
        assert_eq!(config.max_keys_per_scan, None);
        assert_eq!(config.batch_parallel_threshold, 100);
    }

    #[test]
    fn config_high_throughput_has_new_fields() {
        let config = VectorEngineConfig::high_throughput();
        assert_eq!(config.max_dimension, None);
        assert_eq!(config.max_keys_per_scan, None);
        assert_eq!(config.batch_parallel_threshold, 100);
    }
}
