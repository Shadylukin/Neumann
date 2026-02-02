// SPDX-License-Identifier: MIT OR Apache-2.0
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
//!
//! # Distance Metrics
//!
//! Two metric enums are available:
//!
//! ## `DistanceMetric` (Simple API)
//! Use for basic similarity search via `search_similar_with_metric()`:
//! - `Cosine` - Best for normalized embeddings (text, images). Range: [-1, 1]
//! - `Euclidean` - Best for absolute distances. Range: [0, inf)
//! - `DotProduct` - Best for recommendation systems. Range: (-inf, inf)
//!
//! ## `ExtendedDistanceMetric` (HNSW API)
//! Use for HNSW index construction via `search_with_hnsw_and_metric()`:
//! - All `DistanceMetric` variants plus:
//! - `Angular` - Cosine converted to angular distance
//! - `Manhattan` - L1 norm, robust to outliers
//! - `Chebyshev` - L-infinity norm, max absolute difference
//! - `Jaccard` - Set similarity for binary/sparse vectors
//! - `Overlap` - Minimum overlap coefficient
//! - `Geodesic` - Spherical distance for geographic data
//! - `Composite` - Weighted combination of metrics
//!
//! ## When to Use Which
//! | Use Case | Recommended Metric |
//! |----------|-------------------|
//! | Text embeddings (OpenAI, etc.) | `Cosine` |
//! | Image feature vectors | `Cosine` or `Euclidean` |
//! | Sparse vectors (TF-IDF) | `Jaccard` or `Cosine` |
//! | Geographic coordinates | `Geodesic` |
//! | Recommendation scores | `DotProduct` |
//! | General purpose | `Cosine` (default) |

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
#![warn(missing_docs)]

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tensor_store::{
    fields, hnsw::simd, SparseVector, TensorData, TensorStore, TensorStoreError, TensorValue,
};
use tracing::instrument;
// Re-export HNSW types from tensor_store for backward compatibility
pub use tensor_store::{HNSWConfig, HNSWIndex, ScalarQuantizedVector};
// Re-export WAL config for durable storage
pub use tensor_store::WalConfig;

// Re-export distance metrics from tensor_store for extended metric support (9 variants + composite)
pub use tensor_store::{DistanceMetric as ExtendedDistanceMetric, GeometricConfig};

// Re-export new quantization and index types
pub use tensor_store::{
    BinaryThreshold, BinaryVector, IVFConfig, IVFIndex, IVFIndexState, IVFStorage, PQCodebook,
    PQConfig, PQVector,
};

/// Error types for vector operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorError {
    /// The requested embedding was not found.
    NotFound(String),
    /// Vector dimension mismatch.
    DimensionMismatch {
        /// The expected dimension.
        expected: usize,
        /// The actual dimension received.
        got: usize,
    },
    /// Empty vector provided.
    EmptyVector,
    /// Invalid top_k value.
    InvalidTopK,
    /// Storage error from underlying tensor store.
    StorageError(String),
    /// Validation error during batch operation.
    BatchValidationError {
        /// Index of the failing input.
        index: usize,
        /// Description of the validation failure.
        cause: String,
    },
    /// Operation error during batch operation.
    BatchOperationError {
        /// Index of the failing operation.
        index: usize,
        /// Description of the operation failure.
        cause: String,
    },
    /// Configuration error.
    ConfigurationError(String),
    /// Collection already exists.
    CollectionExists(String),
    /// Collection not found.
    CollectionNotFound(String),
    /// IO error during persistence operations.
    IoError(String),
    /// Serialization error during persistence operations.
    SerializationError(String),
    /// Search operation timed out.
    SearchTimeout {
        /// Operation that timed out.
        operation: String,
        /// Configured timeout in milliseconds.
        timeout_ms: u64,
    },
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
            Self::CollectionExists(name) => write!(f, "Collection already exists: {name}"),
            Self::CollectionNotFound(name) => write!(f, "Collection not found: {name}"),
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
            Self::SerializationError(msg) => write!(f, "Serialization error: {msg}"),
            Self::SearchTimeout {
                operation,
                timeout_ms,
            } => {
                write!(f, "search timeout: {operation} exceeded {timeout_ms}ms")
            },
        }
    }
}

impl std::error::Error for VectorError {}

impl From<TensorStoreError> for VectorError {
    fn from(e: TensorStoreError) -> Self {
        Self::StorageError(e.to_string())
    }
}

impl From<io::Error> for VectorError {
    fn from(e: io::Error) -> Self {
        Self::IoError(e.to_string())
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

/// A specialized Result type for vector operations.
pub type Result<T> = std::result::Result<T, VectorError>;

/// Monotonic deadline for search timeout checking.
#[derive(Debug, Clone, Copy)]
struct Deadline {
    deadline: Option<Instant>,
    timeout_ms: u64,
}

impl Deadline {
    fn from_duration(timeout: Option<Duration>) -> Self {
        Self {
            deadline: timeout.map(|d| Instant::now() + d),
            timeout_ms: timeout.map_or(0, |d| d.as_millis() as u64),
        }
    }

    #[cfg(test)]
    const fn never() -> Self {
        Self {
            deadline: None,
            timeout_ms: 0,
        }
    }

    #[inline]
    fn is_expired(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    const fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }
}

/// Result of a similarity search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// The key of the matching embedding.
    pub key: String,
    /// The similarity score (cosine similarity, range -1 to 1).
    pub score: f32,
}

impl SearchResult {
    #[allow(missing_docs)]
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

// ========== Filtered Search Types ==========

/// Filter condition for filtered similarity search.
///
/// Evaluates metadata fields attached to embeddings.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterCondition {
    /// Equality: field = value.
    Eq(String, FilterValue),
    /// Not equal: field != value.
    Ne(String, FilterValue),
    /// Less than: field < value.
    Lt(String, FilterValue),
    /// Less than or equal: field <= value.
    Le(String, FilterValue),
    /// Greater than: field > value.
    Gt(String, FilterValue),
    /// Greater than or equal: field >= value.
    Ge(String, FilterValue),
    /// Logical AND of two conditions.
    And(Box<Self>, Box<Self>),
    /// Logical OR of two conditions.
    Or(Box<Self>, Box<Self>),
    /// Always true (matches all).
    True,
    /// Field exists check.
    Exists(String),
    /// String contains substring.
    Contains(String, String),
    /// String starts with prefix.
    StartsWith(String, String),
    /// Value in list.
    In(String, Vec<FilterValue>),
}

impl FilterCondition {
    #[allow(missing_docs)]
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self::And(Box::new(self), Box::new(other))
    }

    #[allow(missing_docs)]
    #[must_use]
    pub fn or(self, other: Self) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }
}

/// Filter value for comparisons.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// String value.
    String(String),
    /// Boolean value.
    Bool(bool),
    /// Null value.
    Null,
}

impl From<i64> for FilterValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<f64> for FilterValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<String> for FilterValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for FilterValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

impl From<bool> for FilterValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

/// Strategy for filtered search execution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FilterStrategy {
    /// Automatically choose strategy based on estimated selectivity.
    #[default]
    Auto,
    /// Pre-filter: build subset of matching vectors, then search.
    /// Better for highly selective filters (< 10% match).
    PreFilter,
    /// Post-filter: search full index, then filter results.
    /// Better for non-selective filters (> 10% match).
    PostFilter,
}

/// Configuration for filtered search behavior.
#[derive(Debug, Clone)]
pub struct FilteredSearchConfig {
    /// Filter strategy selection.
    pub strategy: FilterStrategy,
    /// Selectivity threshold for auto strategy (0.0-1.0).
    /// Below this threshold, use pre-filter; above, use post-filter.
    pub selectivity_threshold: f32,
    /// Oversample factor for post-filter strategy.
    /// Fetches this many times more candidates before filtering.
    pub oversample_factor: usize,
}

impl Default for FilteredSearchConfig {
    fn default() -> Self {
        Self {
            strategy: FilterStrategy::Auto,
            selectivity_threshold: 0.1,
            oversample_factor: 3,
        }
    }
}

impl FilteredSearchConfig {
    /// Creates a config that always uses pre-filtering.
    #[must_use]
    pub const fn pre_filter() -> Self {
        Self {
            strategy: FilterStrategy::PreFilter,
            selectivity_threshold: 0.1,
            oversample_factor: 3,
        }
    }

    /// Creates a config that always uses post-filtering.
    #[must_use]
    pub const fn post_filter() -> Self {
        Self {
            strategy: FilterStrategy::PostFilter,
            selectivity_threshold: 0.1,
            oversample_factor: 3,
        }
    }

    /// Creates a config with custom oversample factor.
    #[must_use]
    pub const fn with_oversample(mut self, factor: usize) -> Self {
        self.oversample_factor = factor;
        self
    }
}

// ========== Collection Types ==========

/// Configuration for a vector collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorCollectionConfig {
    /// Expected embedding dimension (enforced on insert if set).
    pub dimension: Option<usize>,
    /// Default distance metric for this collection.
    pub distance_metric: DistanceMetric,
    /// Whether to auto-build HNSW index on insert threshold.
    pub auto_index: bool,
    /// Auto-index threshold (number of vectors before auto-building).
    pub auto_index_threshold: usize,
}

impl Default for VectorCollectionConfig {
    fn default() -> Self {
        Self {
            dimension: None,
            distance_metric: DistanceMetric::Cosine,
            auto_index: false,
            auto_index_threshold: 1000,
        }
    }
}

impl VectorCollectionConfig {
    /// Creates a collection config with specified dimension.
    #[must_use]
    pub const fn with_dimension(mut self, dim: usize) -> Self {
        self.dimension = Some(dim);
        self
    }

    /// Creates a collection config with specified distance metric.
    #[must_use]
    pub const fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Creates a collection config with auto-indexing enabled.
    #[must_use]
    pub const fn with_auto_index(mut self, threshold: usize) -> Self {
        self.auto_index = true;
        self.auto_index_threshold = threshold;
        self
    }
}

// ========== Index Persistence Types ==========

/// Persistent vector index format for saving to disk.
///
/// This captures the essential data needed to restore a collection:
/// vectors, their keys, and configuration. HNSW indices can be rebuilt
/// from this data since the vectors contain all necessary information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentVectorIndex {
    /// Collection this index belongs to ("default" for non-collection embeddings).
    pub collection: String,
    /// Collection configuration.
    pub config: VectorCollectionConfig,
    /// Vector data: (key, vector, optional metadata).
    pub vectors: Vec<VectorEntry>,
    /// Creation timestamp (Unix seconds).
    pub created_at: u64,
    /// Format version for future compatibility.
    pub version: u32,
}

/// A single vector entry in the persistent index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// The embedding key.
    pub key: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// Optional metadata (serialized as JSON-compatible map).
    pub metadata: Option<HashMap<String, MetadataValue>>,
}

/// Metadata value for serialization (simplified from TensorValue).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetadataValue {
    /// Null value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// String value.
    String(String),
}

impl MetadataValue {
    /// Convert from a `TensorValue` if possible.
    #[must_use]
    pub fn from_tensor_value(tv: &TensorValue) -> Option<Self> {
        use tensor_store::ScalarValue;
        match tv {
            TensorValue::Scalar(ScalarValue::Null) => Some(Self::Null),
            TensorValue::Scalar(ScalarValue::Bool(b)) => Some(Self::Bool(*b)),
            TensorValue::Scalar(ScalarValue::Int(i)) => Some(Self::Int(*i)),
            TensorValue::Scalar(ScalarValue::Float(f)) => Some(Self::Float(*f)),
            TensorValue::Scalar(ScalarValue::String(s)) => Some(Self::String(s.clone())),
            _ => None, // Vectors, sparse vectors, etc. not stored as metadata
        }
    }
}

impl From<MetadataValue> for TensorValue {
    fn from(mv: MetadataValue) -> Self {
        use tensor_store::ScalarValue;
        match mv {
            MetadataValue::Null => Self::Scalar(ScalarValue::Null),
            MetadataValue::Bool(b) => Self::Scalar(ScalarValue::Bool(b)),
            MetadataValue::Int(i) => Self::Scalar(ScalarValue::Int(i)),
            MetadataValue::Float(f) => Self::Scalar(ScalarValue::Float(f)),
            MetadataValue::String(s) => Self::Scalar(ScalarValue::String(s)),
        }
    }
}

impl PersistentVectorIndex {
    /// Current format version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new empty persistent index.
    #[must_use]
    pub fn new(collection: String, config: VectorCollectionConfig) -> Self {
        Self {
            collection,
            config,
            vectors: Vec::new(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            version: Self::CURRENT_VERSION,
        }
    }

    /// Add a vector entry.
    pub fn push(
        &mut self,
        key: String,
        vector: Vec<f32>,
        metadata: Option<HashMap<String, MetadataValue>>,
    ) {
        self.vectors.push(VectorEntry {
            key,
            vector,
            metadata,
        });
    }

    /// Number of vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
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
    /// Search operation timeout. `None` means no timeout (default).
    pub search_timeout: Option<Duration>,
    /// Maximum index file size in bytes (default: 100MB).
    pub max_index_file_bytes: Option<usize>,
    /// Maximum entries in a loaded index (default: 1M).
    pub max_index_entries: Option<usize>,
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
            search_timeout: None,
            max_index_file_bytes: Some(100 * 1024 * 1024), // 100MB default
            max_index_entries: Some(1_000_000),            // 1M entries
        }
    }
}

impl VectorEngineConfig {
    /// Creates a configuration optimized for high throughput workloads.
    ///
    /// Uses a lower parallel threshold (1000) for more aggressive parallelism.
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
            search_timeout: None,
            max_index_file_bytes: Some(100 * 1024 * 1024), // 100MB
            max_index_entries: Some(1_000_000),
        }
    }

    /// Creates a configuration optimized for low memory environments.
    ///
    /// Sets `max_dimension` to 4096 and `max_keys_per_scan` to 10,000.
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
            search_timeout: Some(Duration::from_secs(30)),
            max_index_file_bytes: Some(10 * 1024 * 1024), // 10MB for low memory
            max_index_entries: Some(100_000),             // 100K entries
        }
    }

    /// Validates the configuration parameters.
    ///
    /// Returns an error if any parameter is out of valid range.
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
        if let Some(max_bytes) = self.max_index_file_bytes {
            if max_bytes == 0 {
                return Err(VectorError::ConfigurationError(
                    "max_index_file_bytes must be greater than 0".to_string(),
                ));
            }
        }
        if let Some(max_entries) = self.max_index_entries {
            if max_entries == 0 {
                return Err(VectorError::ConfigurationError(
                    "max_index_entries must be greater than 0".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Set the default embedding dimension.
    #[must_use]
    pub const fn with_default_dimension(mut self, dim: usize) -> Self {
        self.default_dimension = Some(dim);
        self
    }

    /// Set the sparse vector detection threshold (0.0-1.0).
    #[must_use]
    pub const fn with_sparse_threshold(mut self, threshold: f32) -> Self {
        self.sparse_threshold = threshold;
        self
    }

    /// Set the parallel processing threshold.
    #[must_use]
    pub const fn with_parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Set the default distance metric.
    #[must_use]
    pub const fn with_default_metric(mut self, metric: DistanceMetric) -> Self {
        self.default_metric = metric;
        self
    }

    /// Set the maximum allowed embedding dimension.
    #[must_use]
    pub const fn with_max_dimension(mut self, max: usize) -> Self {
        self.max_dimension = Some(max);
        self
    }

    /// Set the maximum keys per scan operation.
    #[must_use]
    pub const fn with_max_keys_per_scan(mut self, max: usize) -> Self {
        self.max_keys_per_scan = Some(max);
        self
    }

    /// Set the batch parallel processing threshold.
    #[must_use]
    pub const fn with_batch_parallel_threshold(mut self, threshold: usize) -> Self {
        self.batch_parallel_threshold = threshold;
        self
    }

    /// Set the search operation timeout.
    #[must_use]
    pub const fn with_search_timeout(mut self, timeout: Duration) -> Self {
        self.search_timeout = Some(timeout);
        self
    }

    /// Set the maximum index file size in bytes.
    #[must_use]
    pub const fn with_max_index_file_bytes(mut self, max_bytes: usize) -> Self {
        self.max_index_file_bytes = Some(max_bytes);
        self
    }

    /// Set the maximum number of entries in a loaded index.
    #[must_use]
    pub const fn with_max_index_entries(mut self, max_entries: usize) -> Self {
        self.max_index_entries = Some(max_entries);
        self
    }
}

// ========== HNSW Build Options ==========

/// Storage strategy for HNSW index construction.
///
/// Controls how vectors are stored internally in the HNSW index.
/// Different strategies trade off memory usage vs search performance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HNSWStorageStrategy {
    /// Dense storage (default). Stores full f32 vectors.
    /// Best search quality, highest memory usage.
    #[default]
    Dense,
    /// Automatic sparse/dense detection based on sparsity threshold.
    /// Vectors with sparsity above the threshold use sparse storage.
    Auto,
    /// 8-bit scalar quantization (~4x memory reduction).
    /// Slight recall degradation but significant memory savings.
    Quantized,
}

/// Options for building an HNSW index.
///
/// Combines storage strategy with HNSW configuration parameters.
/// Use factory methods for common presets or builders for custom configuration.
#[derive(Debug, Clone)]
pub struct HNSWBuildOptions {
    /// Storage strategy for vectors in the index.
    pub storage: HNSWStorageStrategy,
    /// HNSW algorithm configuration.
    pub hnsw_config: HNSWConfig,
}

impl Default for HNSWBuildOptions {
    fn default() -> Self {
        Self {
            storage: HNSWStorageStrategy::Dense,
            hnsw_config: HNSWConfig::default(),
        }
    }
}

impl HNSWBuildOptions {
    /// Creates default options (Dense storage, default HNSW config).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates options optimized for low memory usage.
    ///
    /// Uses quantized storage (~4x memory reduction) with high-speed HNSW config.
    #[must_use]
    pub fn memory_optimized() -> Self {
        Self {
            storage: HNSWStorageStrategy::Quantized,
            hnsw_config: HNSWConfig::high_speed(),
        }
    }

    /// Creates options optimized for search recall quality.
    ///
    /// Uses dense storage with high-recall HNSW config.
    #[must_use]
    pub fn high_recall() -> Self {
        Self {
            storage: HNSWStorageStrategy::Dense,
            hnsw_config: HNSWConfig::high_recall(),
        }
    }

    /// Creates options optimized for sparse vectors.
    ///
    /// Uses auto-detection to store sparse vectors efficiently.
    #[must_use]
    pub fn sparse_optimized() -> Self {
        Self {
            storage: HNSWStorageStrategy::Auto,
            hnsw_config: HNSWConfig::default(),
        }
    }

    /// Sets the storage strategy.
    #[must_use]
    pub const fn with_storage(mut self, storage: HNSWStorageStrategy) -> Self {
        self.storage = storage;
        self
    }

    /// Sets the HNSW configuration.
    #[must_use]
    pub fn with_hnsw_config(mut self, config: HNSWConfig) -> Self {
        self.hnsw_config = config;
        self
    }

    /// Sets the sparsity threshold for Auto storage strategy.
    ///
    /// Vectors with sparsity (fraction of zeros) above this threshold
    /// will use sparse storage. Only affects `HNSWStorageStrategy::Auto`.
    #[must_use]
    pub fn with_sparsity_threshold(mut self, threshold: f32) -> Self {
        self.hnsw_config.sparsity_threshold = threshold;
        self
    }
}

// ========== IVF Build Options ==========

/// Options for building an IVF (Inverted File) index.
///
/// IVF partitions vectors into clusters for sublinear search.
/// Supports multiple storage formats within clusters.
#[derive(Debug, Clone, Default)]
pub struct IVFBuildOptions {
    /// IVF configuration (clusters, nprobe, storage format).
    pub config: IVFConfig,
}

impl IVFBuildOptions {
    /// Creates default IVF options (100 clusters, flat storage).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates IVF-Flat options with specified number of clusters.
    ///
    /// IVF-Flat stores full vectors in each cluster list.
    /// Best recall, highest memory usage.
    #[must_use]
    pub fn flat(num_clusters: usize) -> Self {
        Self {
            config: IVFConfig::flat(num_clusters),
        }
    }

    /// Creates IVF-PQ options with specified clusters and PQ config.
    ///
    /// IVF-PQ uses Product Quantization for extreme compression.
    /// Best memory efficiency, good recall.
    #[must_use]
    pub fn pq(num_clusters: usize, pq_config: PQConfig) -> Self {
        Self {
            config: IVFConfig::pq(num_clusters, pq_config),
        }
    }

    /// Creates IVF-Binary options with specified clusters.
    ///
    /// IVF-Binary uses binary quantization for fast Hamming distance.
    /// Fastest search, lower recall.
    #[must_use]
    pub fn binary(num_clusters: usize) -> Self {
        Self {
            config: IVFConfig::binary(num_clusters, BinaryThreshold::Sign),
        }
    }

    /// Sets the number of clusters to probe during search.
    #[must_use]
    pub const fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.config.nprobe = nprobe;
        self
    }

    /// Sets the number of clusters.
    #[must_use]
    pub const fn with_num_clusters(mut self, num_clusters: usize) -> Self {
        self.config.num_clusters = num_clusters;
        self
    }

    /// Sets the storage format.
    #[must_use]
    pub fn with_storage(mut self, storage: IVFStorage) -> Self {
        self.config.storage = storage;
        self
    }
}

// ========== Batch Operations ==========

/// Input for batch embedding storage.
#[derive(Debug, Clone)]
pub struct EmbeddingInput {
    /// The key to store the embedding under.
    pub key: String,
    /// The embedding vector.
    pub vector: Vec<f32>,
}

impl EmbeddingInput {
    /// Creates a new embedding input with the given key and vector.
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
    /// Keys that were successfully stored.
    pub stored_keys: Vec<String>,
    /// Number of embeddings stored.
    pub count: usize,
}

impl BatchResult {
    /// Creates a new batch result from a list of stored keys.
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
    /// Creates pagination with skip offset and limit.
    #[must_use]
    pub const fn new(skip: usize, limit: usize) -> Self {
        Self {
            skip,
            limit: Some(limit),
            count_total: false,
        }
    }

    /// Enables total count computation in the result.
    #[must_use]
    pub const fn with_total(mut self) -> Self {
        self.count_total = true;
        self
    }

    /// Creates pagination with only a skip offset (no limit).
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
    /// The items in this page.
    pub items: Vec<T>,
    /// Total count of items (if requested via `count_total`).
    pub total_count: Option<usize>,
    /// Whether more items exist beyond this page.
    pub has_more: bool,
}

impl<T> PagedResult<T> {
    /// Creates a new paged result with the given items, total count, and has_more flag.
    #[must_use]
    pub const fn new(items: Vec<T>, total_count: Option<usize>, has_more: bool) -> Self {
        Self {
            items,
            total_count,
            has_more,
        }
    }

    /// Creates an empty paged result with zero total count.
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
    /// Collection configurations (name -> config).
    collections: Arc<RwLock<HashMap<String, VectorCollectionConfig>>>,
    /// Lock to serialize delete operations and prevent TOCTOU races.
    delete_lock: RwLock<()>,
}

impl VectorEngine {
    /// Create a new VectorEngine with a fresh TensorStore.
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: TensorStore::new(),
            config: VectorEngineConfig::default(),
            collections: Arc::new(RwLock::new(HashMap::new())),
            delete_lock: RwLock::new(()),
        }
    }

    /// Create a VectorEngine using an existing TensorStore.
    #[must_use]
    pub fn with_store(store: TensorStore) -> Self {
        Self {
            store,
            config: VectorEngineConfig::default(),
            collections: Arc::new(RwLock::new(HashMap::new())),
            delete_lock: RwLock::new(()),
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
            collections: Arc::new(RwLock::new(HashMap::new())),
            delete_lock: RwLock::new(()),
        })
    }

    /// Create a VectorEngine with existing store and custom configuration.
    ///
    /// Validates the configuration before construction.
    pub fn with_store_and_config(store: TensorStore, config: VectorEngineConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            store,
            config,
            collections: Arc::new(RwLock::new(HashMap::new())),
            delete_lock: RwLock::new(()),
        })
    }

    /// Opens a durable vector engine with WAL at the given path.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the WAL file cannot be opened.
    pub fn open_durable<P: AsRef<Path>>(wal_path: P, wal_config: WalConfig) -> Result<Self> {
        Self::open_durable_with_config(wal_path, wal_config, VectorEngineConfig::default())
    }

    /// Opens a durable vector engine with WAL and custom config.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if the WAL file cannot be opened.
    pub fn open_durable_with_config<P: AsRef<Path>>(
        wal_path: P,
        wal_config: WalConfig,
        config: VectorEngineConfig,
    ) -> Result<Self> {
        config.validate()?;
        let store = TensorStore::open_durable(wal_path, wal_config)
            .map_err(|e| VectorError::StorageError(e.to_string()))?;
        Ok(Self {
            store,
            config,
            collections: Arc::new(RwLock::new(HashMap::new())),
            delete_lock: RwLock::new(()),
        })
    }

    /// Recovers a vector engine from WAL and optional snapshot.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if recovery fails.
    pub fn recover<P: AsRef<Path>>(
        wal_path: P,
        wal_config: &WalConfig,
        snapshot_path: Option<&Path>,
    ) -> Result<Self> {
        Self::recover_with_config(
            wal_path,
            wal_config,
            snapshot_path,
            VectorEngineConfig::default(),
        )
    }

    /// Recovers a vector engine from WAL with custom config.
    ///
    /// # Errors
    ///
    /// Returns `StorageError` if recovery fails.
    pub fn recover_with_config<P: AsRef<Path>>(
        wal_path: P,
        wal_config: &WalConfig,
        snapshot_path: Option<&Path>,
        config: VectorEngineConfig,
    ) -> Result<Self> {
        config.validate()?;
        let store = TensorStore::recover(wal_path, wal_config, snapshot_path)
            .map_err(|e| VectorError::StorageError(e.to_string()))?;
        Ok(Self {
            store,
            config,
            collections: Arc::new(RwLock::new(HashMap::new())),
            delete_lock: RwLock::new(()),
        })
    }

    /// Returns whether this engine is using durable storage.
    #[must_use]
    pub fn is_durable(&self) -> bool {
        self.store.has_wal()
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

    // ========== Collection Key Helpers ==========

    /// Key prefix for embeddings in a named collection.
    fn collection_embedding_key(collection: &str, key: &str) -> String {
        format!("coll:{}:emb:{}", collection, key)
    }

    /// Key prefix for all embeddings in a collection (for scanning).
    fn collection_embedding_prefix(collection: &str) -> String {
        format!("coll:{}:emb:", collection)
    }

    /// The default collection name.
    pub const DEFAULT_COLLECTION: &'static str = "default";

    // ========== Collection Management ==========

    /// Create a new collection with the given configuration.
    ///
    /// Returns an error if a collection with the same name already exists.
    #[instrument(skip(self, config), fields(collection = %name))]
    pub fn create_collection(&self, name: &str, config: VectorCollectionConfig) -> Result<()> {
        let mut collections = self.collections.write();
        if collections.contains_key(name) {
            return Err(VectorError::CollectionExists(name.to_string()));
        }
        collections.insert(name.to_string(), config);
        drop(collections);
        Ok(())
    }

    /// Delete a collection and all its embeddings.
    ///
    /// Returns an error if the collection does not exist.
    #[instrument(skip(self), fields(collection = %name))]
    pub fn delete_collection(&self, name: &str) -> Result<()> {
        {
            let mut collections = self.collections.write();
            if !collections.contains_key(name) {
                return Err(VectorError::CollectionNotFound(name.to_string()));
            }
            collections.remove(name);
        }

        // Delete all embeddings in the collection
        let prefix = Self::collection_embedding_prefix(name);
        let keys = self.store.scan(&prefix);
        for key in keys {
            let _ = self.store.delete(&key);
        }

        Ok(())
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    /// Get configuration for a collection.
    ///
    /// Returns None if the collection does not exist.
    pub fn get_collection_config(&self, name: &str) -> Option<VectorCollectionConfig> {
        self.collections.read().get(name).cloned()
    }

    /// Check if a collection exists.
    pub fn collection_exists(&self, name: &str) -> bool {
        self.collections.read().contains_key(name)
    }

    /// Count embeddings in a collection.
    pub fn collection_count(&self, name: &str) -> usize {
        let prefix = Self::collection_embedding_prefix(name);
        self.store.scan(&prefix).len()
    }

    // ========== Collection-Aware Storage ==========

    /// Store embedding in a specific collection.
    ///
    /// Creates the collection with default config if it doesn't exist.
    #[instrument(skip(self, vector), fields(collection = %collection, key = %key, vector_dim = vector.len()))]
    pub fn store_in_collection(&self, collection: &str, key: &str, vector: Vec<f32>) -> Result<()> {
        self.store_in_collection_with_metadata(collection, key, vector, HashMap::new())
    }

    /// Store embedding with metadata in a specific collection.
    #[instrument(skip(self, vector, metadata), fields(collection = %collection, key = %key, vector_dim = vector.len()))]
    pub fn store_in_collection_with_metadata(
        &self,
        collection: &str,
        key: &str,
        vector: Vec<f32>,
        metadata: HashMap<String, TensorValue>,
    ) -> Result<()> {
        if vector.is_empty() {
            return Err(VectorError::EmptyVector);
        }

        // Check collection config for dimension constraint
        let collection_config = self.collections.read().get(collection).cloned();
        if let Some(ref config) = collection_config {
            if let Some(expected_dim) = config.dimension {
                if vector.len() != expected_dim {
                    return Err(VectorError::DimensionMismatch {
                        expected: expected_dim,
                        got: vector.len(),
                    });
                }
            }
        }

        // Check global max_dimension
        if let Some(max_dim) = self.config.max_dimension {
            if vector.len() > max_dim {
                return Err(VectorError::DimensionMismatch {
                    expected: max_dim,
                    got: vector.len(),
                });
            }
        }

        let storage_key = Self::collection_embedding_key(collection, key);
        let mut tensor = TensorData::new();

        let storage = if self.should_use_sparse(&vector) {
            TensorValue::Sparse(SparseVector::from_dense(&vector))
        } else {
            TensorValue::Vector(vector)
        };
        tensor.set("vector", storage);

        // Store metadata with prefix
        for (field, value) in metadata {
            tensor.set(Self::metadata_field_key(&field), value);
        }

        self.store.put(storage_key, tensor)?;
        Ok(())
    }

    /// Get embedding from a specific collection.
    pub fn get_from_collection(&self, collection: &str, key: &str) -> Result<Vec<f32>> {
        let storage_key = Self::collection_embedding_key(collection, key);
        let tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(format!("{}:{}", collection, key)))?;

        let vector_value = tensor
            .get("vector")
            .ok_or_else(|| VectorError::NotFound(format!("{}:{}", collection, key)))?;

        Self::extract_vector(vector_value)
            .ok_or_else(|| VectorError::NotFound(format!("{}:{}", collection, key)))
    }

    /// Delete embedding from a specific collection.
    pub fn delete_from_collection(&self, collection: &str, key: &str) -> Result<()> {
        let storage_key = Self::collection_embedding_key(collection, key);
        if !self.store.exists(&storage_key) {
            return Err(VectorError::NotFound(format!("{}:{}", collection, key)));
        }
        self.store.delete(&storage_key)?;
        Ok(())
    }

    /// Check if embedding exists in a specific collection.
    pub fn exists_in_collection(&self, collection: &str, key: &str) -> bool {
        let storage_key = Self::collection_embedding_key(collection, key);
        self.store.exists(&storage_key)
    }

    /// List all keys in a collection.
    pub fn list_collection_keys(&self, collection: &str) -> Vec<String> {
        let prefix = Self::collection_embedding_prefix(collection);
        self.store
            .scan(&prefix)
            .into_iter()
            .filter_map(|k| k.strip_prefix(&prefix).map(ToString::to_string))
            .collect()
    }

    /// Get metadata for a vector in a specific collection.
    pub fn get_collection_metadata(
        &self,
        collection: &str,
        key: &str,
    ) -> Result<HashMap<String, TensorValue>> {
        let storage_key = Self::collection_embedding_key(collection, key);
        let tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(key.to_string()))?;

        let mut metadata = HashMap::new();
        for (field, value) in tensor.fields_iter() {
            if let Some(meta_field) = field.strip_prefix(Self::METADATA_PREFIX) {
                metadata.insert(meta_field.to_string(), value.clone());
            }
        }
        Ok(metadata)
    }

    // ========== Collection-Aware Search ==========

    /// Search within a specific collection.
    #[instrument(skip(self, query), fields(collection = %collection, query_dim = query.len(), top_k))]
    pub fn search_in_collection(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        let deadline = Deadline::from_duration(self.config.search_timeout);

        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        // Check collection config for dimension constraint
        let collection_config = self.collections.read().get(collection).cloned();
        if let Some(ref config) = collection_config {
            if let Some(expected_dim) = config.dimension {
                if query.len() != expected_dim {
                    return Err(VectorError::DimensionMismatch {
                        expected: expected_dim,
                        got: query.len(),
                    });
                }
            }
        }

        let prefix = Self::collection_embedding_prefix(collection);
        let query_magnitude = Self::magnitude(query);
        if query_magnitude == 0.0 {
            return Ok(Vec::new());
        }

        let keys: Vec<_> = self.store.scan(&prefix);

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_in_collection".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        let mut results: Vec<SearchResult> = keys
            .into_iter()
            .filter_map(|storage_key| {
                let key = storage_key.strip_prefix(&prefix)?;
                let tensor = self.store.get(&storage_key).ok()?;
                let vector_value = tensor.get("vector")?;
                let vector = Self::extract_vector(vector_value)?;

                if vector.len() != query.len() {
                    return None;
                }

                let score = Self::cosine_similarity(query, &vector, query_magnitude);
                Some(SearchResult::new(key.to_string(), score))
            })
            .collect();

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_in_collection".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }

    /// Search with filter in a specific collection.
    #[allow(clippy::too_many_lines)]
    #[instrument(skip(self, query, filter, config), fields(collection = %collection, query_dim = query.len(), top_k))]
    pub fn search_filtered_in_collection(
        &self,
        collection: &str,
        query: &[f32],
        top_k: usize,
        filter: &FilterCondition,
        config: Option<FilteredSearchConfig>,
    ) -> Result<Vec<SearchResult>> {
        let deadline = Deadline::from_duration(self.config.search_timeout);

        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        // Check collection config for dimension constraint
        let collection_config_opt = self.collections.read().get(collection).cloned();
        if let Some(ref coll_config) = collection_config_opt {
            if let Some(expected_dim) = coll_config.dimension {
                if query.len() != expected_dim {
                    return Err(VectorError::DimensionMismatch {
                        expected: expected_dim,
                        got: query.len(),
                    });
                }
            }
        }

        let prefix = Self::collection_embedding_prefix(collection);
        let query_magnitude = Self::magnitude(query);
        if query_magnitude == 0.0 {
            return Ok(Vec::new());
        }

        let filter_config = config.unwrap_or_default();
        let strategy = match filter_config.strategy {
            FilterStrategy::Auto => {
                // Estimate selectivity for collection
                let keys = self.store.scan(&prefix);
                let sample_size = 100.min(keys.len());
                if sample_size == 0 {
                    FilterStrategy::PostFilter
                } else {
                    let matches = keys
                        .iter()
                        .take(sample_size)
                        .filter(|k| {
                            self.store
                                .get(k)
                                .map(|t| Self::evaluate_filter(&t, filter))
                                .unwrap_or(false)
                        })
                        .count();
                    let selectivity = matches as f32 / sample_size as f32;
                    if selectivity < filter_config.selectivity_threshold {
                        FilterStrategy::PreFilter
                    } else {
                        FilterStrategy::PostFilter
                    }
                }
            },
            other => other,
        };

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_filtered_in_collection".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        let mut results: Vec<SearchResult> = match strategy {
            FilterStrategy::PreFilter | FilterStrategy::Auto => {
                // Pre-filter: filter first, then search
                self.store
                    .scan(&prefix)
                    .into_iter()
                    .filter_map(|storage_key| {
                        let tensor = self.store.get(&storage_key).ok()?;
                        if !Self::evaluate_filter(&tensor, filter) {
                            return None;
                        }
                        let key = storage_key.strip_prefix(&prefix)?;
                        let vector_value = tensor.get("vector")?;
                        let vector = Self::extract_vector(vector_value)?;
                        if vector.len() != query.len() {
                            return None;
                        }
                        let score = Self::cosine_similarity(query, &vector, query_magnitude);
                        Some(SearchResult::new(key.to_string(), score))
                    })
                    .collect()
            },
            FilterStrategy::PostFilter => {
                // Post-filter: search first with oversample, then filter
                let oversample_k = top_k
                    .saturating_mul(filter_config.oversample_factor)
                    .max(top_k);
                let candidates = self.search_in_collection(collection, query, oversample_k)?;
                candidates
                    .into_iter()
                    .filter(|r| {
                        let storage_key = Self::collection_embedding_key(collection, &r.key);
                        self.store
                            .get(&storage_key)
                            .map(|t| Self::evaluate_filter(&t, filter))
                            .unwrap_or(false)
                    })
                    .collect()
            },
        };

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_filtered_in_collection".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
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
        let _guard = self.delete_lock.write();

        let storage_key = Self::embedding_key(key);
        if !self.store.exists(&storage_key) {
            return Err(VectorError::NotFound(key.to_string()));
        }
        self.store.delete(&storage_key)?;
        Ok(())
    }

    /// Check if an embedding exists.
    #[instrument(skip(self), fields(key = %key))]
    pub fn exists(&self, key: &str) -> bool {
        let storage_key = Self::embedding_key(key);
        self.store.exists(&storage_key)
    }

    /// Get the count of stored embeddings.
    #[instrument(skip(self))]
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
        let deadline = Deadline::from_duration(self.config.search_timeout);

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

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_similar".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        // Use parallel iteration for large datasets, sequential for small
        let mut results: Vec<SearchResult> = if keys.len() >= self.config.parallel_threshold {
            Self::search_parallel(&self.store, &keys, query, query_magnitude)
        } else {
            Self::search_sequential(&self.store, &keys, query, query_magnitude)
        };

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_similar".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

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
    #[instrument(skip(self, query), fields(query_dim = query.len(), top_k, metric = ?metric))]
    pub fn search_similar_with_metric(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        let deadline = Deadline::from_duration(self.config.search_timeout);

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

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_similar_with_metric".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        let mut results: Vec<SearchResult> = if keys.len() >= self.config.parallel_threshold {
            Self::search_parallel_with_metric(&self.store, &keys, query, query_magnitude, metric)
        } else {
            Self::search_sequential_with_metric(&self.store, &keys, query, query_magnitude, metric)
        };

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_similar_with_metric".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

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

        if a_magnitude == 0.0 || b_magnitude == 0.0 {
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

    /// List embedding keys, respecting `config.max_keys_per_scan` if set.
    #[instrument(skip(self))]
    pub fn list_keys(&self) -> Vec<String> {
        self.list_keys_bounded()
    }

    /// List embedding keys with memory safety bounds.
    ///
    /// Uses `config.max_keys_per_scan` if set, otherwise returns all keys.
    /// Prefer this over `list_keys()` in production environments.
    #[instrument(skip(self))]
    pub fn list_keys_bounded(&self) -> Vec<String> {
        let limit = self.config.max_keys_per_scan.unwrap_or(usize::MAX);
        self.store
            .scan(Self::embedding_prefix())
            .into_iter()
            .take(limit)
            .filter_map(|k| k.strip_prefix(Self::embedding_prefix()).map(String::from))
            .collect()
    }

    /// Clear embeddings, respecting `max_keys_per_scan` if set.
    ///
    /// Returns the number of embeddings deleted. If `max_keys_per_scan` is set
    /// and there were more embeddings than the limit, call again until 0 is returned.
    #[instrument(skip(self))]
    pub fn clear(&self) -> Result<usize> {
        let max_keys = self.config.max_keys_per_scan.unwrap_or(usize::MAX);
        let keys: Vec<_> = self
            .store
            .scan(Self::embedding_prefix())
            .into_iter()
            .take(max_keys)
            .collect();

        let count = keys.len();
        for key in keys {
            self.store.delete(&key)?;
        }
        Ok(count)
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
        self.build_hnsw_index_with_options(HNSWBuildOptions {
            storage: HNSWStorageStrategy::Dense,
            hnsw_config: config,
        })
    }

    /// Build an HNSW index with configurable storage strategy.
    ///
    /// Returns a tuple of (index, key_mapping) where key_mapping maps node IDs to keys.
    /// Use `HNSWBuildOptions` to configure both storage strategy and HNSW parameters.
    ///
    /// # Storage Strategies
    ///
    /// - `Dense`: Full f32 vectors, best quality (default)
    /// - `Auto`: Automatic sparse/dense detection based on sparsity threshold
    /// - `Quantized`: 8-bit scalar quantization (~4x memory reduction)
    ///
    /// # Example
    /// ```ignore
    /// let engine = VectorEngine::new();
    /// // ... store embeddings ...
    ///
    /// // Memory-optimized index with quantization
    /// let (index, keys) = engine.build_hnsw_index_with_options(
    ///     HNSWBuildOptions::memory_optimized()
    /// )?;
    ///
    /// // High-recall index with dense storage
    /// let (index, keys) = engine.build_hnsw_index_with_options(
    ///     HNSWBuildOptions::high_recall()
    /// )?;
    ///
    /// // Custom configuration
    /// let (index, keys) = engine.build_hnsw_index_with_options(
    ///     HNSWBuildOptions::new()
    ///         .with_storage(HNSWStorageStrategy::Auto)
    ///         .with_sparsity_threshold(0.7)
    /// )?;
    /// ```
    #[instrument(skip(self, options))]
    pub fn build_hnsw_index_with_options(
        &self,
        options: HNSWBuildOptions,
    ) -> Result<(HNSWIndex, Vec<String>)> {
        let keys = self.list_keys();

        if keys.is_empty() {
            return Ok((HNSWIndex::with_config(options.hnsw_config), Vec::new()));
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

        let index = HNSWIndex::with_config(options.hnsw_config);
        let mut key_mapping = Vec::with_capacity(keys.len());

        insert_with_strategy(&index, first_vector, options.storage);
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

            insert_with_strategy(&index, vector, options.storage);
            key_mapping.push(key);
        }

        Ok((index, key_mapping))
    }

    /// Build an HNSW index with default configuration.
    #[instrument(skip(self))]
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
        let deadline = Deadline::from_duration(self.config.search_timeout);

        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        let results = index.search(query, top_k);

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_with_hnsw".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

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
        let deadline = Deadline::from_duration(self.config.search_timeout);

        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        // Fetch 2x candidates from HNSW
        let candidate_count = top_k.saturating_mul(2).max(10);
        let candidates = index.search(query, candidate_count);

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_with_hnsw_and_metric".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

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

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_with_hnsw_and_metric".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(top_k);
        Ok(results)
    }

    // ========== IVF Index Methods ==========

    /// Build an IVF (Inverted File) index from all stored embeddings.
    ///
    /// IVF partitions vectors into clusters using k-means, enabling sublinear
    /// search by only scanning the `nprobe` closest clusters.
    ///
    /// Returns the index and a mapping from IVF vector IDs to keys.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use vector_engine::{VectorEngine, IVFBuildOptions};
    ///
    /// let engine = VectorEngine::new();
    /// engine.store_embedding("doc1", vec![0.1, 0.2, 0.3, 0.4]).unwrap();
    /// engine.store_embedding("doc2", vec![0.5, 0.6, 0.7, 0.8]).unwrap();
    ///
    /// // Build IVF-Flat index with 10 clusters
    /// let (index, keys) = engine.build_ivf_index(IVFBuildOptions::flat(10)).unwrap();
    /// ```
    #[instrument(skip(self, options))]
    pub fn build_ivf_index(&self, options: IVFBuildOptions) -> Result<(IVFIndex, Vec<String>)> {
        let keys = self.list_keys();

        if keys.is_empty() {
            return Ok((IVFIndex::new(options.config), Vec::new()));
        }

        // Collect all vectors
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(keys.len());
        let mut expected_dim = None;

        for key in &keys {
            let vector = self.get_embedding(key)?;

            // Validate dimension consistency
            match expected_dim {
                None => expected_dim = Some(vector.len()),
                Some(dim) if dim != vector.len() => {
                    return Err(VectorError::DimensionMismatch {
                        expected: dim,
                        got: vector.len(),
                    });
                },
                _ => {},
            }

            vectors.push(vector);
        }

        // Check config dimension constraint
        if let Some(max_dim) = self.config.max_dimension {
            if let Some(dim) = expected_dim {
                if dim > max_dim {
                    return Err(VectorError::DimensionMismatch {
                        expected: max_dim,
                        got: dim,
                    });
                }
            }
        }

        // Convert to references for training
        let vector_refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        // Create and train the IVF index
        let mut index = IVFIndex::new(options.config);
        index.train(&vector_refs);

        // Add all vectors to the index
        for vector in &vectors {
            index.add(vector);
        }

        Ok((index, keys))
    }

    /// Build an IVF-Flat index with default settings.
    ///
    /// Uses 100 clusters and sqrt(100)=10 nprobe by default.
    #[instrument(skip(self))]
    pub fn build_ivf_index_default(&self) -> Result<(IVFIndex, Vec<String>)> {
        self.build_ivf_index(IVFBuildOptions::default())
    }

    /// Search using an IVF index.
    ///
    /// Searches the `nprobe` closest clusters and returns the top-k results.
    /// This is faster than brute force for large datasets.
    ///
    /// # Arguments
    ///
    /// * `index` - The IVF index to search
    /// * `key_mapping` - Mapping from IVF vector IDs to original keys
    /// * `query` - The query vector
    /// * `top_k` - Number of results to return
    #[instrument(skip(self, index, key_mapping, query))]
    pub fn search_with_ivf(
        &self,
        index: &IVFIndex,
        key_mapping: &[String],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        let deadline = Deadline::from_duration(self.config.search_timeout);

        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        let results = index.search(query, top_k);

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_with_ivf".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        Ok(results
            .into_iter()
            .filter_map(|(vector_id, distance)| {
                key_mapping.get(vector_id).map(|key| SearchResult {
                    key: key.clone(),
                    // IVF returns distances, convert to similarity
                    score: 1.0 / (1.0 + distance),
                })
            })
            .collect())
    }

    /// Search using an IVF index with custom nprobe.
    ///
    /// Higher nprobe values search more clusters, improving recall
    /// at the cost of speed.
    #[instrument(skip(self, index, key_mapping, query))]
    pub fn search_with_ivf_nprobe(
        &self,
        index: &IVFIndex,
        key_mapping: &[String],
        query: &[f32],
        top_k: usize,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>> {
        let deadline = Deadline::from_duration(self.config.search_timeout);

        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        let results = index.search_with_nprobe(query, top_k, nprobe);

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_with_ivf_nprobe".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        Ok(results
            .into_iter()
            .filter_map(|(vector_id, distance)| {
                key_mapping.get(vector_id).map(|key| SearchResult {
                    key: key.clone(),
                    score: 1.0 / (1.0 + distance),
                })
            })
            .collect())
    }

    /// Estimate memory usage for building an IVF index.
    ///
    /// Returns estimated bytes based on the storage format and cluster count.
    pub fn estimate_ivf_memory(&self, options: &IVFBuildOptions) -> Result<usize> {
        let count = self.count();
        if count == 0 {
            return Ok(0);
        }

        // Sample first embedding for dimension
        let keys = self.list_keys();
        let first = self.get_embedding(&keys[0])?;
        let dim = first.len();

        let num_clusters = options.config.num_clusters;

        // Centroids memory
        let centroid_bytes = num_clusters * dim * 4;

        // Vector storage depends on format
        let vector_bytes = match &options.config.storage {
            IVFStorage::Flat => count * dim * 4, // Full f32 vectors
            IVFStorage::PQ(pq_config) => {
                count * pq_config.num_subspaces // 1 byte per subspace
            },
            IVFStorage::Binary(_) => count * dim.div_ceil(64) * 8, // 1 bit per dim packed into u64
        };

        // Inverted list overhead (IDs, metadata)
        let list_overhead = count * 8; // Vector IDs

        Ok(centroid_bytes + vector_bytes + list_overhead)
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
        let _guard = self.delete_lock.write();

        let deleted = keys
            .into_iter()
            .filter(|key| {
                let storage_key = Self::embedding_key(key);
                if self.store.exists(&storage_key) {
                    self.store.delete(&storage_key).is_ok()
                } else {
                    false
                }
            })
            .count();

        Ok(deleted)
    }

    // ========== Pagination ==========

    /// List embedding keys with pagination.
    #[instrument(skip(self, pagination))]
    pub fn list_keys_paginated(&self, pagination: Pagination) -> PagedResult<String> {
        // Apply memory bound from config
        let max_scan = self.config.max_keys_per_scan.unwrap_or(usize::MAX);

        // Calculate fetch limit: skip + limit, bounded by config (saturating to prevent overflow)
        let fetch_limit = pagination
            .skip
            .saturating_add(pagination.limit.unwrap_or(max_scan))
            .min(max_scan);

        // Single pass: scan -> take -> filter -> skip -> take -> collect
        let items: Vec<String> = self
            .store
            .scan(Self::embedding_prefix())
            .into_iter()
            .take(fetch_limit)
            .filter_map(|k| k.strip_prefix(Self::embedding_prefix()).map(String::from))
            .skip(pagination.skip)
            .take(pagination.limit.unwrap_or(usize::MAX))
            .collect();

        // Get total count if requested (uses count() which is O(n) but no extra Vec)
        let total_count = if pagination.count_total {
            Some(self.count())
        } else {
            None
        };

        let has_more = total_count.map_or_else(
            || items.len() == pagination.limit.unwrap_or(0),
            |total| pagination.skip.saturating_add(items.len()) < total,
        );

        PagedResult::new(items, total_count, has_more)
    }

    /// Search for similar embeddings with pagination.
    #[instrument(skip(self, query, pagination), fields(query_dim = query.len(), top_k))]
    pub fn search_similar_paginated(
        &self,
        query: &[f32],
        top_k: usize,
        pagination: Pagination,
    ) -> Result<PagedResult<SearchResult>> {
        // Get full results up to skip + limit (saturating to prevent overflow)
        let total_needed = pagination
            .skip
            .saturating_add(pagination.limit.unwrap_or(top_k));
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
            (Some(_), Some(total)) => pagination.skip.saturating_add(items.len()) < total,
            _ => false,
        };

        Ok(PagedResult::new(items, total_count, has_more))
    }

    /// Search for similar entities with pagination.
    #[instrument(skip(self, query, pagination), fields(query_dim = query.len(), top_k))]
    pub fn search_entities_paginated(
        &self,
        query: &[f32],
        top_k: usize,
        pagination: Pagination,
    ) -> Result<PagedResult<SearchResult>> {
        // Saturating add to prevent overflow
        let total_needed = pagination
            .skip
            .saturating_add(pagination.limit.unwrap_or(top_k));
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
            (Some(_), Some(total)) => pagination.skip.saturating_add(items.len()) < total,
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
        let deadline = Deadline::from_duration(self.config.search_timeout);

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

        let max_scan = self.config.max_keys_per_scan.unwrap_or(usize::MAX);
        let keys: Vec<_> = self.store.scan("").into_iter().take(max_scan).collect();

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_entities".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

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

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_entities".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }

    /// Scan for all entity keys that have embeddings.
    ///
    /// Respects `max_keys_per_scan` if set.
    pub fn scan_entities_with_embeddings(&self) -> Vec<String> {
        let max_scan = self.config.max_keys_per_scan.unwrap_or(usize::MAX);
        self.store
            .scan("")
            .into_iter()
            .take(max_scan)
            .filter(|key| self.entity_has_embedding(key))
            .collect()
    }

    /// Count entities with embeddings (unified mode).
    pub fn count_entities_with_embeddings(&self) -> usize {
        self.scan_entities_with_embeddings().len()
    }

    // ========== Metadata Storage for Filtered Search ==========

    /// Metadata field prefix for embedding metadata.
    const METADATA_PREFIX: &'static str = "meta:";

    /// Convert a metadata field name to its storage key.
    fn metadata_field_key(field: &str) -> String {
        format!("{}{}", Self::METADATA_PREFIX, field)
    }

    /// Store embedding with associated metadata for filtered search.
    ///
    /// Metadata fields are stored alongside the vector in the same TensorData,
    /// using a "meta:" prefix to distinguish them from the vector field.
    ///
    /// # Example
    /// ```rust,ignore
    /// use vector_engine::VectorEngine;
    /// use tensor_store::TensorValue;
    /// use std::collections::HashMap;
    ///
    /// let engine = VectorEngine::new();
    /// let mut metadata = HashMap::new();
    /// metadata.insert("category".to_string(), TensorValue::from("electronics"));
    /// metadata.insert("price".to_string(), TensorValue::from(299.99_f64));
    ///
    /// engine.store_embedding_with_metadata("product1", vec![0.1, 0.2, 0.3], metadata).unwrap();
    /// ```
    #[instrument(skip(self, vector, metadata), fields(key = %key, vector_dim = vector.len()))]
    pub fn store_embedding_with_metadata(
        &self,
        key: &str,
        vector: Vec<f32>,
        metadata: HashMap<String, TensorValue>,
    ) -> Result<()> {
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

        // Store the vector
        let storage = if self.should_use_sparse(&vector) {
            TensorValue::Sparse(SparseVector::from_dense(&vector))
        } else {
            TensorValue::Vector(vector)
        };
        tensor.set("vector", storage);

        // Store metadata fields with prefix
        for (field, value) in metadata {
            tensor.set(Self::metadata_field_key(&field), value);
        }

        self.store.put(storage_key, tensor)?;
        Ok(())
    }

    /// Retrieve metadata fields for an embedding.
    ///
    /// Returns only the metadata fields, not the vector itself.
    /// Metadata field names are returned without the internal "meta:" prefix.
    #[instrument(skip(self), fields(key = %key))]
    pub fn get_metadata(&self, key: &str) -> Result<HashMap<String, TensorValue>> {
        let storage_key = Self::embedding_key(key);
        let tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(key.to_string()))?;

        let mut metadata = HashMap::new();
        for (field, value) in tensor.fields_iter() {
            if let Some(meta_field) = field.strip_prefix(Self::METADATA_PREFIX) {
                metadata.insert(meta_field.to_string(), value.clone());
            }
        }

        Ok(metadata)
    }

    /// Update metadata without touching the vector.
    ///
    /// Merges the provided metadata with existing metadata. To remove a field,
    /// use `remove_metadata_field()` instead.
    ///
    /// Returns an error if the embedding does not exist.
    #[instrument(skip(self, metadata), fields(key = %key))]
    pub fn update_metadata(&self, key: &str, metadata: HashMap<String, TensorValue>) -> Result<()> {
        let storage_key = Self::embedding_key(key);
        let mut tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(key.to_string()))?;

        // Verify the embedding exists
        if tensor.get("vector").is_none() {
            return Err(VectorError::NotFound(key.to_string()));
        }

        // Update metadata fields
        for (field, value) in metadata {
            tensor.set(Self::metadata_field_key(&field), value);
        }

        self.store.put(storage_key, tensor)?;
        Ok(())
    }

    /// Remove a specific metadata field from an embedding.
    #[instrument(skip(self), fields(key = %key, field = %field))]
    pub fn remove_metadata_field(&self, key: &str, field: &str) -> Result<()> {
        let storage_key = Self::embedding_key(key);
        let mut tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(key.to_string()))?;

        tensor.remove(&Self::metadata_field_key(field));
        self.store.put(storage_key, tensor)?;
        Ok(())
    }

    /// Check if an embedding has a specific metadata field.
    pub fn has_metadata_field(&self, key: &str, field: &str) -> bool {
        let storage_key = Self::embedding_key(key);
        self.store
            .get(&storage_key)
            .map(|t| t.has(&Self::metadata_field_key(field)))
            .unwrap_or(false)
    }

    /// Get a single metadata field value.
    pub fn get_metadata_field(&self, key: &str, field: &str) -> Result<Option<TensorValue>> {
        let storage_key = Self::embedding_key(key);
        let tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(key.to_string()))?;

        Ok(tensor.get(&Self::metadata_field_key(field)).cloned())
    }

    // ========== Filtered Search ==========

    /// Search for similar embeddings with filter predicate applied.
    ///
    /// The filter is applied to metadata fields. Supports hybrid pre/post-filter
    /// strategies based on estimated selectivity.
    ///
    /// # Strategy Selection
    ///
    /// - **Pre-filter**: Best when filter matches < 10% of embeddings.
    ///   Filters first, then searches the subset.
    /// - **Post-filter**: Best when filter matches > 10% of embeddings.
    ///   Searches with oversample, then filters results.
    /// - **Auto**: Estimates selectivity and chooses automatically.
    #[instrument(skip(self, query, filter, config), fields(query_dim = query.len(), top_k))]
    pub fn search_similar_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &FilterCondition,
        config: Option<FilteredSearchConfig>,
    ) -> Result<Vec<SearchResult>> {
        let deadline = Deadline::from_duration(self.config.search_timeout);

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

        let config = config.unwrap_or_default();

        // Determine strategy
        let strategy = match config.strategy {
            FilterStrategy::Auto => self.choose_filter_strategy(filter, &config),
            other => other,
        };

        if deadline.is_expired() {
            return Err(VectorError::SearchTimeout {
                operation: "search_similar_filtered".to_string(),
                timeout_ms: deadline.timeout_ms(),
            });
        }

        match strategy {
            FilterStrategy::PreFilter | FilterStrategy::Auto => {
                Ok(self.search_with_pre_filter(query, top_k, filter))
            },
            FilterStrategy::PostFilter => {
                self.search_with_post_filter(query, top_k, filter, &config)
            },
        }
    }

    /// Choose filter strategy based on estimated selectivity.
    fn choose_filter_strategy(
        &self,
        filter: &FilterCondition,
        config: &FilteredSearchConfig,
    ) -> FilterStrategy {
        // For True filter, always post-filter (nothing to filter)
        if matches!(filter, FilterCondition::True) {
            return FilterStrategy::PostFilter;
        }

        // Estimate selectivity by sampling
        let sample_size = 100.min(self.count());
        if sample_size == 0 {
            return FilterStrategy::PostFilter;
        }

        let keys = self.list_keys();
        let sample_keys: Vec<_> = keys.iter().take(sample_size).collect();
        let matches = sample_keys
            .iter()
            .filter(|k| self.evaluate_filter_for_key(k, filter))
            .count();

        let selectivity = matches as f32 / sample_size as f32;

        if selectivity < config.selectivity_threshold {
            FilterStrategy::PreFilter
        } else {
            FilterStrategy::PostFilter
        }
    }

    /// Pre-filter strategy: filter first, then search subset.
    fn search_with_pre_filter(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &FilterCondition,
    ) -> Vec<SearchResult> {
        let query_magnitude = Self::magnitude(query);
        if query_magnitude == 0.0 {
            return Vec::new();
        }

        // Get all matching keys
        let matching_keys: Vec<String> = self
            .list_keys()
            .into_iter()
            .filter(|key| self.evaluate_filter_for_key(key, filter))
            .collect();

        if matching_keys.is_empty() {
            return Vec::new();
        }

        // Search only matching embeddings
        let mut results: Vec<SearchResult> = matching_keys
            .iter()
            .filter_map(|key| {
                let vector = self.get_embedding(key).ok()?;
                if vector.len() != query.len() {
                    return None;
                }
                let score = Self::cosine_similarity(query, &vector, query_magnitude);
                Some(SearchResult::new(key.clone(), score))
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        results
    }

    /// Post-filter strategy: search with oversample, then filter.
    fn search_with_post_filter(
        &self,
        query: &[f32],
        top_k: usize,
        filter: &FilterCondition,
        config: &FilteredSearchConfig,
    ) -> Result<Vec<SearchResult>> {
        // Oversample to get more candidates
        let oversample_k = top_k.saturating_mul(config.oversample_factor).max(top_k);
        let candidates = self.search_similar(query, oversample_k)?;

        // Filter candidates
        let filtered: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|r| self.evaluate_filter_for_key(&r.key, filter))
            .take(top_k)
            .collect();

        Ok(filtered)
    }

    /// Evaluate a filter condition against an embedding's metadata.
    fn evaluate_filter_for_key(&self, key: &str, filter: &FilterCondition) -> bool {
        let storage_key = Self::embedding_key(key);
        let Ok(tensor) = self.store.get(&storage_key) else {
            return false;
        };

        Self::evaluate_filter(&tensor, filter)
    }

    /// Evaluate a filter condition against a TensorData.
    fn evaluate_filter(tensor: &TensorData, filter: &FilterCondition) -> bool {
        match filter {
            FilterCondition::True => true,
            FilterCondition::And(a, b) => {
                Self::evaluate_filter(tensor, a) && Self::evaluate_filter(tensor, b)
            },
            FilterCondition::Or(a, b) => {
                Self::evaluate_filter(tensor, a) || Self::evaluate_filter(tensor, b)
            },
            FilterCondition::Exists(field) => tensor.has(&Self::metadata_field_key(field)),
            FilterCondition::Eq(field, val) => {
                Self::compare_field(tensor, field, val, |ord| ord == std::cmp::Ordering::Equal)
            },
            FilterCondition::Ne(field, val) => {
                Self::compare_field(tensor, field, val, |ord| ord != std::cmp::Ordering::Equal)
            },
            FilterCondition::Lt(field, val) => {
                Self::compare_field(tensor, field, val, |ord| ord == std::cmp::Ordering::Less)
            },
            FilterCondition::Le(field, val) => {
                Self::compare_field(tensor, field, val, |ord| ord != std::cmp::Ordering::Greater)
            },
            FilterCondition::Gt(field, val) => {
                Self::compare_field(tensor, field, val, |ord| ord == std::cmp::Ordering::Greater)
            },
            FilterCondition::Ge(field, val) => {
                Self::compare_field(tensor, field, val, |ord| ord != std::cmp::Ordering::Less)
            },
            FilterCondition::Contains(field, substr) => {
                Self::string_contains(tensor, field, substr)
            },
            FilterCondition::StartsWith(field, prefix) => {
                Self::string_starts_with(tensor, field, prefix)
            },
            FilterCondition::In(field, values) => values.iter().any(|v| {
                Self::compare_field(tensor, field, v, |ord| ord == std::cmp::Ordering::Equal)
            }),
        }
    }

    /// Compare a metadata field with a filter value.
    fn compare_field<F>(tensor: &TensorData, field: &str, val: &FilterValue, cmp: F) -> bool
    where
        F: Fn(std::cmp::Ordering) -> bool,
    {
        let meta_key = Self::metadata_field_key(field);
        let Some(stored) = tensor.get(&meta_key) else {
            return false;
        };

        let ordering = Self::compare_tensor_value_to_filter(stored, val);
        ordering.is_some_and(cmp)
    }

    /// Compare TensorValue to FilterValue, returning ordering if compatible.
    fn compare_tensor_value_to_filter(
        tensor_val: &TensorValue,
        filter_val: &FilterValue,
    ) -> Option<std::cmp::Ordering> {
        use tensor_store::ScalarValue;

        match (tensor_val, filter_val) {
            (TensorValue::Scalar(ScalarValue::Int(a)), FilterValue::Int(b)) => Some(a.cmp(b)),
            (TensorValue::Scalar(ScalarValue::Float(a)), FilterValue::Float(b)) => a.partial_cmp(b),
            (TensorValue::Scalar(ScalarValue::Float(a)), FilterValue::Int(b)) => {
                a.partial_cmp(&(*b as f64))
            },
            (TensorValue::Scalar(ScalarValue::Int(a)), FilterValue::Float(b)) => {
                (*a as f64).partial_cmp(b)
            },
            (TensorValue::Scalar(ScalarValue::String(a)), FilterValue::String(b)) => Some(a.cmp(b)),
            (TensorValue::Scalar(ScalarValue::Bool(a)), FilterValue::Bool(b)) => Some(a.cmp(b)),
            (TensorValue::Scalar(ScalarValue::Null), FilterValue::Null) => {
                Some(std::cmp::Ordering::Equal)
            },
            _ => None, // Incompatible types
        }
    }

    /// Check if a string field contains a substring.
    fn string_contains(tensor: &TensorData, field: &str, substr: &str) -> bool {
        use tensor_store::ScalarValue;

        let meta_key = Self::metadata_field_key(field);
        match tensor.get(&meta_key) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.contains(substr),
            _ => false,
        }
    }

    /// Check if a string field starts with a prefix.
    fn string_starts_with(tensor: &TensorData, field: &str, prefix: &str) -> bool {
        use tensor_store::ScalarValue;

        let meta_key = Self::metadata_field_key(field);
        match tensor.get(&meta_key) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.starts_with(prefix),
            _ => false,
        }
    }

    /// Estimate selectivity of a filter (fraction of embeddings that match).
    ///
    /// Returns a value between 0.0 and 1.0. This is useful for query planning.
    pub fn estimate_filter_selectivity(&self, filter: &FilterCondition) -> f32 {
        let count = self.count();
        if count == 0 {
            return 0.0;
        }

        let sample_size = 100.min(count);
        let keys = self.list_keys();
        let sample_keys: Vec<_> = keys.iter().take(sample_size).collect();
        let matches = sample_keys
            .iter()
            .filter(|k| self.evaluate_filter_for_key(k, filter))
            .count();

        matches as f32 / sample_size as f32
    }

    /// Count embeddings matching a filter condition.
    pub fn count_matching(&self, filter: &FilterCondition) -> usize {
        self.list_keys()
            .into_iter()
            .filter(|key| self.evaluate_filter_for_key(key, filter))
            .count()
    }

    /// List keys matching a filter condition.
    pub fn list_keys_matching(&self, filter: &FilterCondition) -> Vec<String> {
        self.list_keys()
            .into_iter()
            .filter(|key| self.evaluate_filter_for_key(key, filter))
            .collect()
    }

    // ========== Index Persistence ==========

    /// Create a snapshot of a collection that can be saved to disk.
    ///
    /// Captures all vectors, keys, and metadata from the specified collection.
    /// For the default (non-collection) embeddings, use `DEFAULT_COLLECTION` as the name.
    #[instrument(skip(self), fields(collection = %collection))]
    pub fn snapshot_collection(&self, collection: &str) -> PersistentVectorIndex {
        let config = self.get_collection_config(collection).unwrap_or_default();
        let mut index = PersistentVectorIndex::new(collection.to_string(), config);

        let prefix = if collection == Self::DEFAULT_COLLECTION {
            Self::embedding_prefix().to_string()
        } else {
            Self::collection_embedding_prefix(collection)
        };

        for storage_key in self.store.scan(&prefix) {
            let key = if collection == Self::DEFAULT_COLLECTION {
                storage_key.strip_prefix(Self::embedding_prefix())
            } else {
                storage_key.strip_prefix(&prefix)
            };

            let Some(key) = key else { continue };
            let Ok(tensor) = self.store.get(&storage_key) else {
                continue;
            };

            // Extract vector
            let Some(vector) = tensor.get("vector").and_then(Self::extract_vector) else {
                continue;
            };

            // Extract metadata
            let mut metadata: HashMap<String, MetadataValue> = HashMap::new();
            for (field, value) in tensor.fields_iter() {
                if let Some(meta_key) = field.strip_prefix(Self::METADATA_PREFIX) {
                    if let Some(mv) = MetadataValue::from_tensor_value(value) {
                        metadata.insert(meta_key.to_string(), mv);
                    }
                }
            }

            let metadata_opt = if metadata.is_empty() {
                None
            } else {
                Some(metadata)
            };
            index.push(key.to_string(), vector, metadata_opt);
        }

        index
    }

    /// Save a collection's index to a file (JSON format).
    ///
    /// The file can be loaded later with `load_index()`.
    #[instrument(skip(self, path), fields(collection = %collection))]
    pub fn save_index(&self, collection: &str, path: impl AsRef<Path>) -> Result<()> {
        let index = self.snapshot_collection(collection);
        let json = serde_json::to_string_pretty(&index)
            .map_err(|e| VectorError::SerializationError(e.to_string()))?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Save a collection's index to a file in compact binary format.
    ///
    /// Uses bincode for efficient serialization. The file can be loaded
    /// with `load_index_binary()`.
    #[instrument(skip(self, path), fields(collection = %collection))]
    pub fn save_index_binary(&self, collection: &str, path: impl AsRef<Path>) -> Result<()> {
        let index = self.snapshot_collection(collection);
        let bytes = bincode::serialize(&index)
            .map_err(|e| VectorError::SerializationError(e.to_string()))?;
        fs::write(path, bytes)?;
        Ok(())
    }

    /// Load an index from a JSON file and restore vectors to a collection.
    ///
    /// Returns the collection name from the loaded index.
    /// If the collection already exists with vectors, they will be overwritten.
    #[instrument(skip(self, path))]
    pub fn load_index(&self, path: impl AsRef<Path>) -> Result<String> {
        let path = path.as_ref();

        // Check file size before reading
        if let Some(max_bytes) = self.config.max_index_file_bytes {
            let metadata = fs::metadata(path)?;
            if metadata.len() > max_bytes as u64 {
                return Err(VectorError::ConfigurationError(format!(
                    "index file size {} exceeds limit {}",
                    metadata.len(),
                    max_bytes
                )));
            }
        }

        let json = fs::read_to_string(path)?;
        let index: PersistentVectorIndex = serde_json::from_str(&json)
            .map_err(|e| VectorError::SerializationError(e.to_string()))?;

        // Check entry count after deserialization
        if let Some(max_entries) = self.config.max_index_entries {
            if index.vectors.len() > max_entries {
                return Err(VectorError::ConfigurationError(format!(
                    "index entry count {} exceeds limit {}",
                    index.vectors.len(),
                    max_entries
                )));
            }
        }

        self.restore_from_index(index)
    }

    /// Load an index from a binary file and restore vectors to a collection.
    ///
    /// Returns the collection name from the loaded index.
    #[instrument(skip(self, path))]
    pub fn load_index_binary(&self, path: impl AsRef<Path>) -> Result<String> {
        let path = path.as_ref();

        // Check file size before reading
        if let Some(max_bytes) = self.config.max_index_file_bytes {
            let metadata = fs::metadata(path)?;
            if metadata.len() > max_bytes as u64 {
                return Err(VectorError::ConfigurationError(format!(
                    "index file size {} exceeds limit {}",
                    metadata.len(),
                    max_bytes
                )));
            }
        }

        let bytes = fs::read(path)?;
        let index: PersistentVectorIndex = bincode::deserialize(&bytes)
            .map_err(|e| VectorError::SerializationError(e.to_string()))?;

        // Check entry count after deserialization
        if let Some(max_entries) = self.config.max_index_entries {
            if index.vectors.len() > max_entries {
                return Err(VectorError::ConfigurationError(format!(
                    "index entry count {} exceeds limit {}",
                    index.vectors.len(),
                    max_entries
                )));
            }
        }

        self.restore_from_index(index)
    }

    /// Restore vectors from a `PersistentVectorIndex`.
    fn restore_from_index(&self, index: PersistentVectorIndex) -> Result<String> {
        let collection = index.collection.clone();

        // Create or update collection config
        if collection != Self::DEFAULT_COLLECTION {
            let mut collections = self.collections.write();
            collections.insert(collection.clone(), index.config.clone());
            drop(collections);
        }

        // Restore vectors
        for entry in index.vectors {
            let metadata: HashMap<String, TensorValue> = entry
                .metadata
                .unwrap_or_default()
                .into_iter()
                .map(|(k, v)| (k, TensorValue::from(v)))
                .collect();

            if collection == Self::DEFAULT_COLLECTION {
                self.store_embedding_with_metadata(&entry.key, entry.vector, metadata)?;
            } else {
                self.store_in_collection_with_metadata(
                    &collection,
                    &entry.key,
                    entry.vector,
                    metadata,
                )?;
            }
        }

        Ok(collection)
    }

    /// Save all collections to a directory.
    ///
    /// Creates one JSON file per collection (including default).
    #[instrument(skip(self, dir))]
    pub fn save_all_indices(&self, dir: impl AsRef<Path>) -> Result<Vec<String>> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;

        let mut saved = Vec::new();

        // Save default collection
        let default_count = self.count();
        if default_count > 0 {
            let path = dir.join("default.json");
            self.save_index(Self::DEFAULT_COLLECTION, &path)?;
            saved.push(Self::DEFAULT_COLLECTION.to_string());
        }

        // Save named collections
        for collection in self.list_collections() {
            let count = self.collection_count(&collection);
            if count > 0 {
                let filename = format!("{}.json", collection);
                let path = dir.join(filename);
                self.save_index(&collection, &path)?;
                saved.push(collection);
            }
        }

        Ok(saved)
    }

    /// Load all indices from a directory.
    ///
    /// Returns the names of collections that were loaded.
    #[instrument(skip(self, dir))]
    pub fn load_all_indices(&self, dir: impl AsRef<Path>) -> Result<Vec<String>> {
        let dir = dir.as_ref();
        let mut loaded = Vec::new();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                match self.load_index(&path) {
                    Ok(collection) => loaded.push(collection),
                    Err(e) => {
                        // Log but continue with other files
                        tracing::warn!("Failed to load index from {:?}: {}", path, e);
                    },
                }
            }
        }

        Ok(loaded)
    }
}

impl Default for VectorEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to insert a vector into an HNSW index using the specified storage strategy.
fn insert_with_strategy(index: &HNSWIndex, vector: Vec<f32>, strategy: HNSWStorageStrategy) {
    match strategy {
        HNSWStorageStrategy::Dense => {
            index.insert(vector);
        },
        HNSWStorageStrategy::Auto => {
            index.insert_auto(vector);
        },
        HNSWStorageStrategy::Quantized => {
            index.insert_quantized(&vector);
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor_store::ScalarValue;

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
    fn cosine_similarity_both_zero_vectors() {
        let a = vec![0.0, 0.0];
        let b = vec![0.0, 0.0];
        let score = VectorEngine::compute_similarity(&a, &b).unwrap();
        // Both zero vectors should return 0.0, not NaN
        assert_eq!(score, 0.0);
        assert!(!score.is_nan());
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

        let cleared = engine.clear().unwrap();
        assert_eq!(cleared, 2);

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
    fn config_validate_invalid_max_index_file_bytes_zero() {
        let config = VectorEngineConfig {
            max_index_file_bytes: Some(0),
            ..Default::default()
        };
        let result = config.validate();
        assert!(matches!(
            result,
            Err(VectorError::ConfigurationError(msg)) if msg.contains("max_index_file_bytes")
        ));
    }

    #[test]
    fn config_validate_invalid_max_index_entries_zero() {
        let config = VectorEngineConfig {
            max_index_entries: Some(0),
            ..Default::default()
        };
        let result = config.validate();
        assert!(matches!(
            result,
            Err(VectorError::ConfigurationError(msg)) if msg.contains("max_index_entries")
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

    #[test]
    fn config_builder_methods() {
        let config = VectorEngineConfig::default()
            .with_default_dimension(128)
            .with_sparse_threshold(0.7)
            .with_parallel_threshold(1000)
            .with_default_metric(DistanceMetric::Euclidean)
            .with_max_dimension(4096)
            .with_max_keys_per_scan(50_000)
            .with_batch_parallel_threshold(200);

        assert_eq!(config.default_dimension, Some(128));
        assert!((config.sparse_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.parallel_threshold, 1000);
        assert_eq!(config.default_metric, DistanceMetric::Euclidean);
        assert_eq!(config.max_dimension, Some(4096));
        assert_eq!(config.max_keys_per_scan, Some(50_000));
        assert_eq!(config.batch_parallel_threshold, 200);
    }

    #[test]
    #[ignore] // Run with: cargo test --release -p vector_engine large_scale -- --ignored
    fn large_scale_million_vectors() {
        use std::time::Instant;

        let config = VectorEngineConfig::high_throughput();
        let engine = VectorEngine::with_config(config).unwrap();

        const VECTOR_COUNT: usize = 1_000_000;
        const DIMENSION: usize = 128;

        // Insert 1M vectors
        let start = Instant::now();
        for i in 0..VECTOR_COUNT {
            let key = format!("vec_{}", i);
            let vector: Vec<f32> = (0..DIMENSION)
                .map(|j| ((i * DIMENSION + j) % 1000) as f32 / 1000.0)
                .collect();
            engine.store_embedding(&key, vector).unwrap();
        }
        let insert_time = start.elapsed();

        assert_eq!(engine.count(), VECTOR_COUNT);

        // Search performance
        let query: Vec<f32> = (0..DIMENSION)
            .map(|i| i as f32 / DIMENSION as f32)
            .collect();
        let start = Instant::now();
        for _ in 0..100 {
            let results = engine.search_similar(&query, 10).unwrap();
            assert_eq!(results.len(), 10);
        }
        let search_time = start.elapsed();

        // Performance assertions (relaxed for CI)
        assert!(
            insert_time.as_secs() < 120,
            "Insert took too long: {:?}",
            insert_time
        );
        assert!(
            search_time.as_millis() < 5000,
            "100 searches took too long: {:?}",
            search_time
        );
    }

    #[test]
    fn error_from_tensor_store_error() {
        let tensor_error = TensorStoreError::NotFound("test_key".to_string());
        let vector_error: VectorError = tensor_error.into();
        assert!(matches!(vector_error, VectorError::StorageError(_)));
        assert!(vector_error.to_string().contains("test_key"));
    }

    #[test]
    fn error_std_error_trait() {
        let error = VectorError::NotFound("test".to_string());
        let _: &dyn std::error::Error = &error;
    }

    #[test]
    fn search_with_metric_parallel_path() {
        // Use a low parallel threshold to trigger parallel search
        let config = VectorEngineConfig {
            parallel_threshold: 5,
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        // Store enough vectors to trigger parallel path
        for i in 0..10 {
            engine
                .store_embedding(&format!("vec_{}", i), vec![i as f32, 0.0, 0.0])
                .unwrap();
        }

        let results = engine
            .search_similar_with_metric(&[5.0, 0.0, 0.0], 3, DistanceMetric::Euclidean)
            .unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn get_embedding_invalid_format() {
        let engine = VectorEngine::new();
        // Directly write a tensor without a vector field to test error path
        let mut tensor = TensorData::new();
        tensor.set("not_vector", TensorValue::Scalar(ScalarValue::Int(42)));
        engine
            .store()
            .put("emb:invalid".to_string(), tensor)
            .unwrap();

        let result = engine.get_embedding("invalid");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    // ========== Metadata Storage Tests ==========

    #[test]
    fn store_embedding_with_metadata_basic() {
        let engine = VectorEngine::new();
        let mut metadata = HashMap::new();
        metadata.insert(
            "category".to_string(),
            TensorValue::Scalar(ScalarValue::String("electronics".to_string())),
        );
        metadata.insert(
            "price".to_string(),
            TensorValue::Scalar(ScalarValue::Float(299.99)),
        );

        engine
            .store_embedding_with_metadata("product1", vec![0.1, 0.2, 0.3], metadata)
            .unwrap();

        // Verify embedding is stored
        let vector = engine.get_embedding("product1").unwrap();
        assert_eq!(vector.len(), 3);

        // Verify metadata is stored
        let retrieved_metadata = engine.get_metadata("product1").unwrap();
        assert_eq!(retrieved_metadata.len(), 2);
        assert!(retrieved_metadata.contains_key("category"));
        assert!(retrieved_metadata.contains_key("price"));
    }

    #[test]
    fn store_embedding_with_metadata_empty_metadata() {
        let engine = VectorEngine::new();
        let metadata = HashMap::new();

        engine
            .store_embedding_with_metadata("key", vec![1.0, 2.0], metadata)
            .unwrap();

        let vector = engine.get_embedding("key").unwrap();
        assert_eq!(vector, vec![1.0, 2.0]);

        let retrieved_metadata = engine.get_metadata("key").unwrap();
        assert!(retrieved_metadata.is_empty());
    }

    #[test]
    fn store_embedding_with_metadata_empty_vector_error() {
        let engine = VectorEngine::new();
        let metadata = HashMap::new();

        let result = engine.store_embedding_with_metadata("key", vec![], metadata);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn store_embedding_with_metadata_dimension_limit() {
        let config = VectorEngineConfig {
            max_dimension: Some(5),
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        let result = engine.store_embedding_with_metadata("key", vec![0.0; 10], HashMap::new());
        assert!(matches!(
            result,
            Err(VectorError::DimensionMismatch {
                expected: 5,
                got: 10
            })
        ));
    }

    #[test]
    fn get_metadata_nonexistent_key() {
        let engine = VectorEngine::new();
        let result = engine.get_metadata("nonexistent");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn update_metadata_basic() {
        let engine = VectorEngine::new();
        let mut initial_metadata = HashMap::new();
        initial_metadata.insert(
            "color".to_string(),
            TensorValue::Scalar(ScalarValue::String("red".to_string())),
        );

        engine
            .store_embedding_with_metadata("item", vec![1.0, 2.0], initial_metadata)
            .unwrap();

        // Update with new metadata
        let mut update = HashMap::new();
        update.insert(
            "size".to_string(),
            TensorValue::Scalar(ScalarValue::String("large".to_string())),
        );
        update.insert(
            "color".to_string(),
            TensorValue::Scalar(ScalarValue::String("blue".to_string())),
        );

        engine.update_metadata("item", update).unwrap();

        let metadata = engine.get_metadata("item").unwrap();
        assert_eq!(metadata.len(), 2);

        // Color should be updated to blue
        match metadata.get("color") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => assert_eq!(s, "blue"),
            _ => panic!("Expected color to be 'blue'"),
        }

        // Size should be added
        assert!(metadata.contains_key("size"));
    }

    #[test]
    fn update_metadata_nonexistent_key() {
        let engine = VectorEngine::new();
        let metadata = HashMap::new();

        let result = engine.update_metadata("nonexistent", metadata);
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn update_metadata_no_vector_error() {
        let engine = VectorEngine::new();
        // Create tensor without vector field
        let mut tensor = TensorData::new();
        tensor.set(
            "meta:test",
            TensorValue::Scalar(ScalarValue::String("value".to_string())),
        );
        engine.store().put("emb:orphan", tensor).unwrap();

        let result = engine.update_metadata("orphan", HashMap::new());
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn remove_metadata_field_basic() {
        let engine = VectorEngine::new();
        let mut metadata = HashMap::new();
        metadata.insert("a".to_string(), TensorValue::Scalar(ScalarValue::Int(1)));
        metadata.insert("b".to_string(), TensorValue::Scalar(ScalarValue::Int(2)));

        engine
            .store_embedding_with_metadata("key", vec![1.0], metadata)
            .unwrap();

        engine.remove_metadata_field("key", "a").unwrap();

        let retrieved = engine.get_metadata("key").unwrap();
        assert!(!retrieved.contains_key("a"));
        assert!(retrieved.contains_key("b"));
    }

    #[test]
    fn remove_metadata_field_nonexistent_key() {
        let engine = VectorEngine::new();
        let result = engine.remove_metadata_field("nonexistent", "field");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn has_metadata_field_true() {
        let engine = VectorEngine::new();
        let mut metadata = HashMap::new();
        metadata.insert(
            "field1".to_string(),
            TensorValue::Scalar(ScalarValue::Int(42)),
        );

        engine
            .store_embedding_with_metadata("key", vec![1.0], metadata)
            .unwrap();

        assert!(engine.has_metadata_field("key", "field1"));
    }

    #[test]
    fn has_metadata_field_false() {
        let engine = VectorEngine::new();
        engine.store_embedding("key", vec![1.0]).unwrap();

        assert!(!engine.has_metadata_field("key", "nonexistent_field"));
    }

    #[test]
    fn has_metadata_field_nonexistent_key() {
        let engine = VectorEngine::new();
        assert!(!engine.has_metadata_field("nonexistent", "field"));
    }

    #[test]
    fn get_metadata_field_basic() {
        let engine = VectorEngine::new();
        let mut metadata = HashMap::new();
        metadata.insert(
            "score".to_string(),
            TensorValue::Scalar(ScalarValue::Float(0.95)),
        );

        engine
            .store_embedding_with_metadata("key", vec![1.0], metadata)
            .unwrap();

        let value = engine.get_metadata_field("key", "score").unwrap();
        match value {
            Some(TensorValue::Scalar(ScalarValue::Float(f))) => {
                assert!((f - 0.95).abs() < f64::EPSILON)
            },
            _ => panic!("Expected float value"),
        }
    }

    #[test]
    fn get_metadata_field_not_present() {
        let engine = VectorEngine::new();
        engine.store_embedding("key", vec![1.0]).unwrap();

        let value = engine.get_metadata_field("key", "nonexistent").unwrap();
        assert!(value.is_none());
    }

    #[test]
    fn get_metadata_field_nonexistent_key() {
        let engine = VectorEngine::new();
        let result = engine.get_metadata_field("nonexistent", "field");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn metadata_with_sparse_vector() {
        let engine = VectorEngine::new();

        // Create sparse vector (>50% zeros)
        let mut sparse = vec![0.0f32; 100];
        sparse[0] = 1.0;
        sparse[50] = 2.0;

        let mut metadata = HashMap::new();
        metadata.insert(
            "type".to_string(),
            TensorValue::Scalar(ScalarValue::String("sparse".to_string())),
        );

        engine
            .store_embedding_with_metadata("sparse_key", sparse.clone(), metadata)
            .unwrap();

        // Verify both vector and metadata work with sparse storage
        let retrieved = engine.get_embedding("sparse_key").unwrap();
        assert_eq!(retrieved.len(), 100);
        assert_eq!(retrieved[0], 1.0);
        assert_eq!(retrieved[50], 2.0);

        let meta = engine.get_metadata("sparse_key").unwrap();
        assert!(meta.contains_key("type"));
    }

    #[test]
    fn metadata_field_key_helper() {
        let key = VectorEngine::metadata_field_key("test_field");
        assert_eq!(key, "meta:test_field");
    }

    #[test]
    fn metadata_multiple_types() {
        let engine = VectorEngine::new();
        let mut metadata = HashMap::new();
        metadata.insert(
            "int_field".to_string(),
            TensorValue::Scalar(ScalarValue::Int(42)),
        );
        metadata.insert(
            "float_field".to_string(),
            TensorValue::Scalar(ScalarValue::Float(3.14)),
        );
        metadata.insert(
            "string_field".to_string(),
            TensorValue::Scalar(ScalarValue::String("hello".to_string())),
        );
        metadata.insert(
            "bool_field".to_string(),
            TensorValue::Scalar(ScalarValue::Bool(true)),
        );

        engine
            .store_embedding_with_metadata("multi_type", vec![1.0, 2.0], metadata)
            .unwrap();

        let retrieved = engine.get_metadata("multi_type").unwrap();
        assert_eq!(retrieved.len(), 4);

        // Verify each type
        match retrieved.get("int_field") {
            Some(TensorValue::Scalar(ScalarValue::Int(i))) => assert_eq!(*i, 42),
            _ => panic!("Expected int"),
        }
        match retrieved.get("float_field") {
            Some(TensorValue::Scalar(ScalarValue::Float(f))) => {
                assert!((*f - 3.14).abs() < f64::EPSILON)
            },
            _ => panic!("Expected float"),
        }
        match retrieved.get("string_field") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => assert_eq!(s, "hello"),
            _ => panic!("Expected string"),
        }
        match retrieved.get("bool_field") {
            Some(TensorValue::Scalar(ScalarValue::Bool(b))) => assert!(*b),
            _ => panic!("Expected bool"),
        }
    }

    #[test]
    fn metadata_overwrites_on_store() {
        let engine = VectorEngine::new();

        // First store
        let mut meta1 = HashMap::new();
        meta1.insert("a".to_string(), TensorValue::Scalar(ScalarValue::Int(1)));
        meta1.insert("b".to_string(), TensorValue::Scalar(ScalarValue::Int(2)));
        engine
            .store_embedding_with_metadata("key", vec![1.0], meta1)
            .unwrap();

        // Second store with different metadata
        let mut meta2 = HashMap::new();
        meta2.insert("c".to_string(), TensorValue::Scalar(ScalarValue::Int(3)));
        engine
            .store_embedding_with_metadata("key", vec![2.0], meta2)
            .unwrap();

        // Only meta2 should exist
        let retrieved = engine.get_metadata("key").unwrap();
        assert_eq!(retrieved.len(), 1);
        assert!(retrieved.contains_key("c"));
        assert!(!retrieved.contains_key("a"));
        assert!(!retrieved.contains_key("b"));
    }

    // ========== Filtered Search Tests ==========

    fn setup_filtered_search_engine() -> VectorEngine {
        let engine = VectorEngine::new();

        // Create test data with various categories
        let categories = ["electronics", "clothing", "food"];
        let prices = [100, 50, 25];

        for (i, (cat, price)) in categories.iter().zip(prices.iter()).enumerate() {
            let mut metadata = HashMap::new();
            metadata.insert(
                "category".to_string(),
                TensorValue::Scalar(ScalarValue::String(cat.to_string())),
            );
            metadata.insert(
                "price".to_string(),
                TensorValue::Scalar(ScalarValue::Int(*price)),
            );
            metadata.insert(
                "active".to_string(),
                TensorValue::Scalar(ScalarValue::Bool(i % 2 == 0)),
            );

            // Use non-zero vectors (i+1) to avoid zero-magnitude issues
            engine
                .store_embedding_with_metadata(
                    &format!("item{}", i),
                    vec![(i + 1) as f32, 1.0, 1.0],
                    metadata,
                )
                .unwrap();
        }

        engine
    }

    #[test]
    fn search_filtered_eq_string() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Eq(
            "category".to_string(),
            FilterValue::String("electronics".to_string()),
        );
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "item0");
    }

    #[test]
    fn search_filtered_eq_int() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Eq("price".to_string(), FilterValue::Int(50));
        let results = engine
            .search_similar_filtered(&[1.0, 0.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "item1");
    }

    #[test]
    fn search_filtered_gt() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Gt("price".to_string(), FilterValue::Int(30));
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 2); // electronics (100) and clothing (50)
    }

    #[test]
    fn search_filtered_lt() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Lt("price".to_string(), FilterValue::Int(60));
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 2); // clothing (50) and food (25)
    }

    #[test]
    fn search_filtered_le() {
        let engine = setup_filtered_search_engine();

        // Less than or equal to 50 should match clothing (50) and food (25)
        let filter = FilterCondition::Le("price".to_string(), FilterValue::Int(50));
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 2); // clothing (50) and food (25)
    }

    #[test]
    fn search_filtered_ge() {
        let engine = setup_filtered_search_engine();

        // Greater than or equal to 50 should match electronics (100) and clothing (50)
        let filter = FilterCondition::Ge("price".to_string(), FilterValue::Int(50));
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 2); // electronics (100) and clothing (50)
    }

    #[test]
    fn search_filtered_and() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Gt("price".to_string(), FilterValue::Int(30)).and(
            FilterCondition::Lt("price".to_string(), FilterValue::Int(80)),
        );
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1); // Only clothing (50)
        assert_eq!(results[0].key, "item1");
    }

    #[test]
    fn search_filtered_or() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Eq(
            "category".to_string(),
            FilterValue::String("electronics".to_string()),
        )
        .or(FilterCondition::Eq(
            "category".to_string(),
            FilterValue::String("food".to_string()),
        ));
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_filtered_true() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::True;
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 3); // All items
    }

    #[test]
    fn search_filtered_exists() {
        let engine = VectorEngine::new();

        let mut meta1 = HashMap::new();
        meta1.insert(
            "tag".to_string(),
            TensorValue::Scalar(ScalarValue::String("a".to_string())),
        );
        engine
            .store_embedding_with_metadata("with_tag", vec![1.0, 0.0], meta1)
            .unwrap();

        engine
            .store_embedding("without_tag", vec![0.0, 1.0])
            .unwrap();

        let filter = FilterCondition::Exists("tag".to_string());
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "with_tag");
    }

    #[test]
    fn search_filtered_contains() {
        let engine = VectorEngine::new();

        let mut meta1 = HashMap::new();
        meta1.insert(
            "description".to_string(),
            TensorValue::Scalar(ScalarValue::String("blue shirt".to_string())),
        );
        engine
            .store_embedding_with_metadata("item1", vec![1.0, 0.0], meta1)
            .unwrap();

        let mut meta2 = HashMap::new();
        meta2.insert(
            "description".to_string(),
            TensorValue::Scalar(ScalarValue::String("red pants".to_string())),
        );
        engine
            .store_embedding_with_metadata("item2", vec![0.0, 1.0], meta2)
            .unwrap();

        let filter = FilterCondition::Contains("description".to_string(), "shirt".to_string());
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "item1");
    }

    #[test]
    fn search_filtered_contains_on_non_string() {
        let engine = VectorEngine::new();

        let mut meta = HashMap::new();
        meta.insert(
            "count".to_string(),
            TensorValue::Scalar(ScalarValue::Int(42)),
        );
        engine
            .store_embedding_with_metadata("item", vec![1.0, 0.0], meta)
            .unwrap();

        // Contains on non-string field should not match
        let filter = FilterCondition::Contains("count".to_string(), "4".to_string());
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn search_filtered_starts_with() {
        let engine = VectorEngine::new();

        let mut meta1 = HashMap::new();
        meta1.insert(
            "sku".to_string(),
            TensorValue::Scalar(ScalarValue::String("ABC123".to_string())),
        );
        engine
            .store_embedding_with_metadata("item1", vec![1.0, 0.0], meta1)
            .unwrap();

        let mut meta2 = HashMap::new();
        meta2.insert(
            "sku".to_string(),
            TensorValue::Scalar(ScalarValue::String("XYZ789".to_string())),
        );
        engine
            .store_embedding_with_metadata("item2", vec![0.0, 1.0], meta2)
            .unwrap();

        let filter = FilterCondition::StartsWith("sku".to_string(), "ABC".to_string());
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "item1");
    }

    #[test]
    fn search_filtered_starts_with_on_non_string() {
        let engine = VectorEngine::new();

        let mut meta = HashMap::new();
        meta.insert(
            "count".to_string(),
            TensorValue::Scalar(ScalarValue::Int(123)),
        );
        engine
            .store_embedding_with_metadata("item", vec![1.0, 0.0], meta)
            .unwrap();

        // StartsWith on non-string field should not match
        let filter = FilterCondition::StartsWith("count".to_string(), "1".to_string());
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn search_filtered_missing_field() {
        let engine = VectorEngine::new();

        // Store without any metadata
        engine.store_embedding("item", vec![1.0, 0.0]).unwrap();

        // Filter on non-existent field should not match
        let filter = FilterCondition::Eq("missing".to_string(), FilterValue::Int(42));
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn search_filtered_in() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::In(
            "category".to_string(),
            vec![
                FilterValue::String("electronics".to_string()),
                FilterValue::String("food".to_string()),
            ],
        );
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_filtered_ne() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Ne(
            "category".to_string(),
            FilterValue::String("electronics".to_string()),
        );
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 2); // clothing and food
    }

    #[test]
    fn search_filtered_bool() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Eq("active".to_string(), FilterValue::Bool(true));
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 2); // item0 and item2
    }

    #[test]
    fn search_filtered_empty_result() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Eq(
            "category".to_string(),
            FilterValue::String("nonexistent".to_string()),
        );
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn search_filtered_pre_filter_strategy() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Eq(
            "category".to_string(),
            FilterValue::String("electronics".to_string()),
        );
        let config = FilteredSearchConfig::pre_filter();
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, Some(config))
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_filtered_post_filter_strategy() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Eq(
            "category".to_string(),
            FilterValue::String("electronics".to_string()),
        );
        let config = FilteredSearchConfig::post_filter();
        let results = engine
            .search_similar_filtered(&[1.0, 1.0, 1.0], 10, &filter, Some(config))
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_filtered_empty_vector_error() {
        let engine = VectorEngine::new();
        let filter = FilterCondition::True;
        let result = engine.search_similar_filtered(&[], 5, &filter, None);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_filtered_zero_top_k_error() {
        let engine = VectorEngine::new();
        let filter = FilterCondition::True;
        let result = engine.search_similar_filtered(&[1.0], 0, &filter, None);
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    #[test]
    fn search_filtered_dimension_limit() {
        let config = VectorEngineConfig {
            max_dimension: Some(5),
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();
        let filter = FilterCondition::True;

        let result = engine.search_similar_filtered(&[0.0; 10], 5, &filter, None);
        assert!(matches!(
            result,
            Err(VectorError::DimensionMismatch {
                expected: 5,
                got: 10
            })
        ));
    }

    #[test]
    fn count_matching_basic() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Gt("price".to_string(), FilterValue::Int(30));
        let count = engine.count_matching(&filter);

        assert_eq!(count, 2);
    }

    #[test]
    fn list_keys_matching_basic() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::Eq(
            "category".to_string(),
            FilterValue::String("electronics".to_string()),
        );
        let keys = engine.list_keys_matching(&filter);

        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&"item0".to_string()));
    }

    #[test]
    fn estimate_filter_selectivity_basic() {
        let engine = setup_filtered_search_engine();

        let filter = FilterCondition::True;
        let selectivity = engine.estimate_filter_selectivity(&filter);
        assert!((selectivity - 1.0).abs() < 0.01);

        let filter_specific = FilterCondition::Eq(
            "category".to_string(),
            FilterValue::String("electronics".to_string()),
        );
        let selectivity_specific = engine.estimate_filter_selectivity(&filter_specific);
        assert!(selectivity_specific > 0.0 && selectivity_specific < 1.0);
    }

    #[test]
    fn filter_condition_and_or_builders() {
        let a = FilterCondition::Eq("x".to_string(), FilterValue::Int(1));
        let b = FilterCondition::Eq("y".to_string(), FilterValue::Int(2));

        let and_cond = a.clone().and(b.clone());
        assert!(matches!(and_cond, FilterCondition::And(_, _)));

        let or_cond = a.or(b);
        assert!(matches!(or_cond, FilterCondition::Or(_, _)));
    }

    #[test]
    fn filter_value_from_traits() {
        let v1: FilterValue = 42_i64.into();
        assert!(matches!(v1, FilterValue::Int(42)));

        let v2: FilterValue = 3.14_f64.into();
        assert!(matches!(v2, FilterValue::Float(f) if (f - 3.14).abs() < f64::EPSILON));

        let v3: FilterValue = "hello".into();
        assert!(matches!(v3, FilterValue::String(s) if s == "hello"));

        let v4: FilterValue = "world".to_string().into();
        assert!(matches!(v4, FilterValue::String(s) if s == "world"));

        let v5: FilterValue = true.into();
        assert!(matches!(v5, FilterValue::Bool(true)));
    }

    #[test]
    fn filter_strategy_default() {
        assert_eq!(FilterStrategy::default(), FilterStrategy::Auto);
    }

    #[test]
    fn filtered_search_config_builders() {
        let pre = FilteredSearchConfig::pre_filter();
        assert_eq!(pre.strategy, FilterStrategy::PreFilter);

        let post = FilteredSearchConfig::post_filter();
        assert_eq!(post.strategy, FilterStrategy::PostFilter);

        let custom = FilteredSearchConfig::default().with_oversample(5);
        assert_eq!(custom.oversample_factor, 5);
    }

    #[test]
    fn search_filtered_float_comparison() {
        let engine = VectorEngine::new();

        let mut meta1 = HashMap::new();
        meta1.insert(
            "score".to_string(),
            TensorValue::Scalar(ScalarValue::Float(0.95)),
        );
        engine
            .store_embedding_with_metadata("high", vec![1.0, 0.0], meta1)
            .unwrap();

        let mut meta2 = HashMap::new();
        meta2.insert(
            "score".to_string(),
            TensorValue::Scalar(ScalarValue::Float(0.5)),
        );
        engine
            .store_embedding_with_metadata("low", vec![0.0, 1.0], meta2)
            .unwrap();

        let filter = FilterCondition::Gt("score".to_string(), FilterValue::Float(0.8));
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "high");
    }

    #[test]
    fn search_filtered_mixed_int_float_comparison() {
        let engine = VectorEngine::new();

        let mut meta = HashMap::new();
        meta.insert(
            "value".to_string(),
            TensorValue::Scalar(ScalarValue::Float(50.5)),
        );
        engine
            .store_embedding_with_metadata("item", vec![1.0, 0.0], meta)
            .unwrap();

        // Compare float field with int filter value
        let filter = FilterCondition::Gt("value".to_string(), FilterValue::Int(50));
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn search_filtered_int_vs_float_filter() {
        let engine = VectorEngine::new();

        // Store an int value
        let mut meta = HashMap::new();
        meta.insert(
            "count".to_string(),
            TensorValue::Scalar(ScalarValue::Int(100)),
        );
        engine
            .store_embedding_with_metadata("item", vec![1.0, 0.0], meta)
            .unwrap();

        // Compare int field with float filter value
        let filter = FilterCondition::Gt("count".to_string(), FilterValue::Float(50.5));
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);

        // Test boundary: 100 should not be > 100.0
        let filter = FilterCondition::Gt("count".to_string(), FilterValue::Float(100.0));
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn search_filtered_null_comparison() {
        let engine = VectorEngine::new();

        // Store a null value
        let mut meta = HashMap::new();
        meta.insert(
            "optional".to_string(),
            TensorValue::Scalar(ScalarValue::Null),
        );
        engine
            .store_embedding_with_metadata("with_null", vec![1.0, 0.0], meta)
            .unwrap();

        // Store without the field
        engine
            .store_embedding("without_field", vec![0.0, 1.0])
            .unwrap();

        // Filter for null should match the one with explicit null
        let filter = FilterCondition::Eq("optional".to_string(), FilterValue::Null);
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "with_null");
    }

    #[test]
    fn search_filtered_string_comparison() {
        let engine = VectorEngine::new();

        let mut meta1 = HashMap::new();
        meta1.insert(
            "name".to_string(),
            TensorValue::Scalar(ScalarValue::String("apple".to_string())),
        );
        engine
            .store_embedding_with_metadata("item1", vec![1.0, 0.0], meta1)
            .unwrap();

        let mut meta2 = HashMap::new();
        meta2.insert(
            "name".to_string(),
            TensorValue::Scalar(ScalarValue::String("banana".to_string())),
        );
        engine
            .store_embedding_with_metadata("item2", vec![0.0, 1.0], meta2)
            .unwrap();

        // String greater than comparison (lexicographic)
        let filter =
            FilterCondition::Gt("name".to_string(), FilterValue::String("app".to_string()));
        let results = engine
            .search_similar_filtered(&[1.0, 1.0], 10, &filter, None)
            .unwrap();

        // Both "apple" and "banana" are > "app"
        assert_eq!(results.len(), 2);

        // Test Le: "apple" <= "apple" should match
        let filter =
            FilterCondition::Le("name".to_string(), FilterValue::String("apple".to_string()));
        let results = engine
            .search_similar_filtered(&[1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "item1");
    }

    #[test]
    fn search_filtered_bool_false() {
        let engine = VectorEngine::new();

        let mut meta1 = HashMap::new();
        meta1.insert(
            "active".to_string(),
            TensorValue::Scalar(ScalarValue::Bool(true)),
        );
        engine
            .store_embedding_with_metadata("active_item", vec![1.0, 0.0], meta1)
            .unwrap();

        let mut meta2 = HashMap::new();
        meta2.insert(
            "active".to_string(),
            TensorValue::Scalar(ScalarValue::Bool(false)),
        );
        engine
            .store_embedding_with_metadata("inactive_item", vec![0.0, 1.0], meta2)
            .unwrap();

        // Filter for false
        let filter = FilterCondition::Eq("active".to_string(), FilterValue::Bool(false));
        let results = engine
            .search_similar_filtered(&[1.0, 1.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "inactive_item");
    }

    #[test]
    fn search_filtered_incompatible_types() {
        let engine = VectorEngine::new();

        let mut meta = HashMap::new();
        meta.insert(
            "value".to_string(),
            TensorValue::Scalar(ScalarValue::String("text".to_string())),
        );
        engine
            .store_embedding_with_metadata("item", vec![1.0, 0.0], meta)
            .unwrap();

        // Try to compare string field with int filter - should not match
        let filter = FilterCondition::Eq("value".to_string(), FilterValue::Int(42));
        let results = engine
            .search_similar_filtered(&[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 0);
    }

    #[test]
    fn search_filtered_respects_top_k() {
        let engine = VectorEngine::new();

        for i in 0..10 {
            let mut meta = HashMap::new();
            meta.insert("idx".to_string(), TensorValue::Scalar(ScalarValue::Int(i)));
            engine
                .store_embedding_with_metadata(&format!("item{}", i), vec![i as f32, 0.0], meta)
                .unwrap();
        }

        let filter = FilterCondition::True;
        let results = engine
            .search_similar_filtered(&[5.0, 0.0], 3, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 3);
    }

    // ========== Collection Tests ==========

    #[test]
    fn create_collection_basic() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default();

        engine.create_collection("test", config).unwrap();

        assert!(engine.collection_exists("test"));
        assert!(engine.get_collection_config("test").is_some());
    }

    #[test]
    fn create_collection_already_exists() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default();

        engine.create_collection("test", config.clone()).unwrap();
        let result = engine.create_collection("test", config);

        assert!(matches!(result, Err(VectorError::CollectionExists(_))));
    }

    #[test]
    fn delete_collection_basic() {
        let engine = VectorEngine::new();
        engine
            .create_collection("test", VectorCollectionConfig::default())
            .unwrap();

        // Add some embeddings
        engine
            .store_in_collection("test", "key1", vec![1.0, 2.0])
            .unwrap();
        engine
            .store_in_collection("test", "key2", vec![3.0, 4.0])
            .unwrap();

        assert_eq!(engine.collection_count("test"), 2);

        // Delete collection
        engine.delete_collection("test").unwrap();

        assert!(!engine.collection_exists("test"));
        assert_eq!(engine.collection_count("test"), 0);
    }

    #[test]
    fn delete_collection_not_found() {
        let engine = VectorEngine::new();
        let result = engine.delete_collection("nonexistent");

        assert!(matches!(result, Err(VectorError::CollectionNotFound(_))));
    }

    #[test]
    fn list_collections_basic() {
        let engine = VectorEngine::new();
        engine
            .create_collection("alpha", VectorCollectionConfig::default())
            .unwrap();
        engine
            .create_collection("beta", VectorCollectionConfig::default())
            .unwrap();

        let mut collections = engine.list_collections();
        collections.sort();

        assert_eq!(collections, vec!["alpha", "beta"]);
    }

    #[test]
    fn store_in_collection_basic() {
        let engine = VectorEngine::new();
        engine
            .create_collection("products", VectorCollectionConfig::default())
            .unwrap();

        engine
            .store_in_collection("products", "item1", vec![1.0, 2.0, 3.0])
            .unwrap();

        let vector = engine.get_from_collection("products", "item1").unwrap();
        assert_eq!(vector, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn store_in_collection_without_prior_create() {
        let engine = VectorEngine::new();

        // Store without creating collection first (should work)
        engine
            .store_in_collection("auto_created", "key", vec![1.0])
            .unwrap();

        let vector = engine.get_from_collection("auto_created", "key").unwrap();
        assert_eq!(vector, vec![1.0]);
    }

    #[test]
    fn store_in_collection_dimension_constraint() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default().with_dimension(3);
        engine.create_collection("fixed_dim", config).unwrap();

        // Correct dimension should work
        engine
            .store_in_collection("fixed_dim", "good", vec![1.0, 2.0, 3.0])
            .unwrap();

        // Wrong dimension should fail
        let result = engine.store_in_collection("fixed_dim", "bad", vec![1.0, 2.0]);
        assert!(matches!(
            result,
            Err(VectorError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn get_from_collection_not_found() {
        let engine = VectorEngine::new();
        let result = engine.get_from_collection("coll", "nonexistent");

        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn delete_from_collection_basic() {
        let engine = VectorEngine::new();
        engine
            .store_in_collection("test", "key", vec![1.0])
            .unwrap();

        assert!(engine.exists_in_collection("test", "key"));

        engine.delete_from_collection("test", "key").unwrap();

        assert!(!engine.exists_in_collection("test", "key"));
    }

    #[test]
    fn delete_from_collection_not_found() {
        let engine = VectorEngine::new();
        let result = engine.delete_from_collection("coll", "nonexistent");

        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn list_collection_keys_basic() {
        let engine = VectorEngine::new();
        engine
            .store_in_collection("test", "alpha", vec![1.0])
            .unwrap();
        engine
            .store_in_collection("test", "beta", vec![2.0])
            .unwrap();

        let mut keys = engine.list_collection_keys("test");
        keys.sort();

        assert_eq!(keys, vec!["alpha", "beta"]);
    }

    #[test]
    fn collection_count_basic() {
        let engine = VectorEngine::new();
        assert_eq!(engine.collection_count("empty"), 0);

        engine.store_in_collection("test", "a", vec![1.0]).unwrap();
        engine.store_in_collection("test", "b", vec![2.0]).unwrap();

        assert_eq!(engine.collection_count("test"), 2);
    }

    #[test]
    fn search_in_collection_basic() {
        let engine = VectorEngine::new();

        engine
            .store_in_collection("products", "p1", vec![1.0, 0.0, 0.0])
            .unwrap();
        engine
            .store_in_collection("products", "p2", vec![0.0, 1.0, 0.0])
            .unwrap();
        engine
            .store_in_collection("products", "p3", vec![0.0, 0.0, 1.0])
            .unwrap();

        let results = engine
            .search_in_collection("products", &[1.0, 0.0, 0.0], 2)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "p1");
    }

    #[test]
    fn search_in_collection_empty() {
        let engine = VectorEngine::new();

        let results = engine
            .search_in_collection("empty", &[1.0, 2.0], 5)
            .unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn search_in_collection_dimension_constraint() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default().with_dimension(3);
        engine.create_collection("fixed", config).unwrap();

        // Wrong query dimension should fail
        let result = engine.search_in_collection("fixed", &[1.0, 2.0], 5);
        assert!(matches!(
            result,
            Err(VectorError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn search_filtered_in_collection_basic() {
        let engine = VectorEngine::new();

        let mut meta1 = HashMap::new();
        meta1.insert(
            "category".to_string(),
            TensorValue::Scalar(ScalarValue::String("A".to_string())),
        );
        engine
            .store_in_collection_with_metadata("test", "item1", vec![1.0, 0.0], meta1)
            .unwrap();

        let mut meta2 = HashMap::new();
        meta2.insert(
            "category".to_string(),
            TensorValue::Scalar(ScalarValue::String("B".to_string())),
        );
        engine
            .store_in_collection_with_metadata("test", "item2", vec![0.0, 1.0], meta2)
            .unwrap();

        let filter =
            FilterCondition::Eq("category".to_string(), FilterValue::String("A".to_string()));
        let results = engine
            .search_filtered_in_collection("test", &[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "item1");
    }

    #[test]
    fn collection_isolation() {
        let engine = VectorEngine::new();

        engine
            .store_in_collection("coll_a", "key1", vec![1.0])
            .unwrap();
        engine
            .store_in_collection("coll_b", "key1", vec![2.0])
            .unwrap();

        // Same key, different collections
        let v_a = engine.get_from_collection("coll_a", "key1").unwrap();
        let v_b = engine.get_from_collection("coll_b", "key1").unwrap();

        assert_eq!(v_a, vec![1.0]);
        assert_eq!(v_b, vec![2.0]);

        // Search should be isolated
        let results_a = engine.search_in_collection("coll_a", &[1.0], 10).unwrap();
        let results_b = engine.search_in_collection("coll_b", &[1.0], 10).unwrap();

        assert_eq!(results_a.len(), 1);
        assert_eq!(results_b.len(), 1);
    }

    #[test]
    fn collection_and_default_isolation() {
        let engine = VectorEngine::new();

        // Store in default (no collection prefix)
        engine.store_embedding("key1", vec![1.0]).unwrap();

        // Store in named collection
        engine
            .store_in_collection("named", "key1", vec![2.0])
            .unwrap();

        // Should be isolated
        let default_v = engine.get_embedding("key1").unwrap();
        let named_v = engine.get_from_collection("named", "key1").unwrap();

        assert_eq!(default_v, vec![1.0]);
        assert_eq!(named_v, vec![2.0]);
    }

    #[test]
    fn collection_config_with_dimension() {
        let config = VectorCollectionConfig::default().with_dimension(128);
        assert_eq!(config.dimension, Some(128));
    }

    #[test]
    fn collection_config_with_metric() {
        let config = VectorCollectionConfig::default().with_metric(DistanceMetric::Euclidean);
        assert_eq!(config.distance_metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn collection_config_with_auto_index() {
        let config = VectorCollectionConfig::default().with_auto_index(500);
        assert!(config.auto_index);
        assert_eq!(config.auto_index_threshold, 500);
    }

    #[test]
    fn collection_config_default() {
        let config = VectorCollectionConfig::default();
        assert_eq!(config.dimension, None);
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(!config.auto_index);
        assert_eq!(config.auto_index_threshold, 1000);
    }

    // ========== Phase 4: Persistence Tests ==========

    #[test]
    fn metadata_value_from_tensor_value() {
        use tensor_store::ScalarValue;

        // Test all supported scalar types
        let null_tv = TensorValue::Scalar(ScalarValue::Null);
        assert!(matches!(
            MetadataValue::from_tensor_value(&null_tv),
            Some(MetadataValue::Null)
        ));

        let bool_tv = TensorValue::Scalar(ScalarValue::Bool(true));
        assert!(matches!(
            MetadataValue::from_tensor_value(&bool_tv),
            Some(MetadataValue::Bool(true))
        ));

        let int_tv = TensorValue::Scalar(ScalarValue::Int(42));
        assert!(matches!(
            MetadataValue::from_tensor_value(&int_tv),
            Some(MetadataValue::Int(42))
        ));

        let float_tv = TensorValue::Scalar(ScalarValue::Float(3.14));
        if let Some(MetadataValue::Float(f)) = MetadataValue::from_tensor_value(&float_tv) {
            assert!((f - 3.14).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }

        let string_tv = TensorValue::Scalar(ScalarValue::String("hello".to_string()));
        assert!(matches!(
            MetadataValue::from_tensor_value(&string_tv),
            Some(MetadataValue::String(s)) if s == "hello"
        ));

        // Vector types should return None
        let vector_tv = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
        assert!(MetadataValue::from_tensor_value(&vector_tv).is_none());
    }

    #[test]
    fn metadata_value_to_tensor_value() {
        use tensor_store::ScalarValue;

        let null_mv = MetadataValue::Null;
        assert!(matches!(
            TensorValue::from(null_mv),
            TensorValue::Scalar(ScalarValue::Null)
        ));

        let bool_mv = MetadataValue::Bool(true);
        assert!(matches!(
            TensorValue::from(bool_mv),
            TensorValue::Scalar(ScalarValue::Bool(true))
        ));

        let int_mv = MetadataValue::Int(42);
        assert!(matches!(
            TensorValue::from(int_mv),
            TensorValue::Scalar(ScalarValue::Int(42))
        ));

        let float_mv = MetadataValue::Float(3.14);
        if let TensorValue::Scalar(ScalarValue::Float(f)) = TensorValue::from(float_mv) {
            assert!((f - 3.14).abs() < 1e-10);
        } else {
            panic!("Expected Float");
        }

        let string_mv = MetadataValue::String("hello".to_string());
        assert!(matches!(
            TensorValue::from(string_mv),
            TensorValue::Scalar(ScalarValue::String(s)) if s == "hello"
        ));
    }

    #[test]
    fn persistent_vector_index_new() {
        let config = VectorCollectionConfig::default();
        let index = PersistentVectorIndex::new("test".to_string(), config);

        assert_eq!(index.collection, "test");
        assert!(index.vectors.is_empty());
        assert_eq!(index.version, PersistentVectorIndex::CURRENT_VERSION);
    }

    #[test]
    fn persistent_vector_index_push() {
        let config = VectorCollectionConfig::default();
        let mut index = PersistentVectorIndex::new("test".to_string(), config);

        assert!(index.is_empty());

        index.push("key1".to_string(), vec![1.0, 2.0], None);
        assert_eq!(index.len(), 1);

        let mut meta = HashMap::new();
        meta.insert(
            "tag".to_string(),
            MetadataValue::String("value".to_string()),
        );
        index.push("key2".to_string(), vec![3.0, 4.0], Some(meta));
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn snapshot_collection_default() {
        let engine = VectorEngine::new();

        engine.store_embedding("vec1", vec![1.0, 2.0]).unwrap();
        engine.store_embedding("vec2", vec![3.0, 4.0]).unwrap();

        let index = engine.snapshot_collection(VectorEngine::DEFAULT_COLLECTION);

        assert_eq!(index.collection, VectorEngine::DEFAULT_COLLECTION);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn snapshot_collection_with_metadata() {
        let engine = VectorEngine::new();

        let mut meta = HashMap::new();
        meta.insert(
            "category".to_string(),
            TensorValue::Scalar(ScalarValue::String("test".to_string())),
        );
        engine
            .store_embedding_with_metadata("vec1", vec![1.0, 2.0], meta)
            .unwrap();

        let index = engine.snapshot_collection(VectorEngine::DEFAULT_COLLECTION);

        assert_eq!(index.len(), 1);
        let entry = &index.vectors[0];
        assert_eq!(entry.key, "vec1");
        assert!(entry.metadata.is_some());

        let meta = entry.metadata.as_ref().unwrap();
        assert!(matches!(
            meta.get("category"),
            Some(MetadataValue::String(s)) if s == "test"
        ));
    }

    #[test]
    fn snapshot_collection_named() {
        let engine = VectorEngine::new();

        engine
            .create_collection("mycoll", VectorCollectionConfig::default())
            .unwrap();
        engine
            .store_in_collection("mycoll", "vec1", vec![1.0, 2.0])
            .unwrap();

        let index = engine.snapshot_collection("mycoll");

        assert_eq!(index.collection, "mycoll");
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn save_and_load_index_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.json");

        let engine = VectorEngine::new();
        engine.store_embedding("vec1", vec![1.0, 2.0, 3.0]).unwrap();
        engine.store_embedding("vec2", vec![4.0, 5.0, 6.0]).unwrap();

        engine
            .save_index(VectorEngine::DEFAULT_COLLECTION, &path)
            .unwrap();

        // Load into new engine
        let engine2 = VectorEngine::new();
        let collection = engine2.load_index(&path).unwrap();

        assert_eq!(collection, VectorEngine::DEFAULT_COLLECTION);
        assert_eq!(engine2.count(), 2);

        let v1 = engine2.get_embedding("vec1").unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0]);

        let v2 = engine2.get_embedding("vec2").unwrap();
        assert_eq!(v2, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn save_and_load_index_binary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.bin");

        let engine = VectorEngine::new();
        engine.store_embedding("vec1", vec![1.0, 2.0, 3.0]).unwrap();

        engine
            .save_index_binary(VectorEngine::DEFAULT_COLLECTION, &path)
            .unwrap();

        let engine2 = VectorEngine::new();
        let collection = engine2.load_index_binary(&path).unwrap();

        assert_eq!(collection, VectorEngine::DEFAULT_COLLECTION);
        let v1 = engine2.get_embedding("vec1").unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn save_and_load_index_with_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.json");

        let engine = VectorEngine::new();
        let mut meta = HashMap::new();
        meta.insert(
            "name".to_string(),
            TensorValue::Scalar(ScalarValue::String("test".to_string())),
        );
        meta.insert(
            "score".to_string(),
            TensorValue::Scalar(ScalarValue::Int(42)),
        );
        engine
            .store_embedding_with_metadata("vec1", vec![1.0, 2.0], meta)
            .unwrap();

        engine
            .save_index(VectorEngine::DEFAULT_COLLECTION, &path)
            .unwrap();

        let engine2 = VectorEngine::new();
        engine2.load_index(&path).unwrap();

        let meta = engine2.get_metadata("vec1").unwrap();
        assert!(matches!(
            meta.get("name"),
            Some(TensorValue::Scalar(ScalarValue::String(s))) if s == "test"
        ));
        assert!(matches!(
            meta.get("score"),
            Some(TensorValue::Scalar(ScalarValue::Int(42)))
        ));
    }

    #[test]
    fn save_and_load_named_collection() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mycoll.json");

        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default().with_dimension(3);
        engine.create_collection("mycoll", config).unwrap();
        engine
            .store_in_collection("mycoll", "vec1", vec![1.0, 2.0, 3.0])
            .unwrap();

        engine.save_index("mycoll", &path).unwrap();

        let engine2 = VectorEngine::new();
        let collection = engine2.load_index(&path).unwrap();

        assert_eq!(collection, "mycoll");
        assert!(engine2.collection_exists("mycoll"));

        let v1 = engine2.get_from_collection("mycoll", "vec1").unwrap();
        assert_eq!(v1, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn save_all_indices_basic() {
        let dir = tempfile::tempdir().unwrap();

        let engine = VectorEngine::new();
        engine
            .store_embedding("default_vec", vec![1.0, 2.0])
            .unwrap();

        engine
            .create_collection("coll_a", VectorCollectionConfig::default())
            .unwrap();
        engine
            .store_in_collection("coll_a", "vec_a", vec![3.0, 4.0])
            .unwrap();

        engine
            .create_collection("coll_b", VectorCollectionConfig::default())
            .unwrap();
        engine
            .store_in_collection("coll_b", "vec_b", vec![5.0, 6.0])
            .unwrap();

        let saved = engine.save_all_indices(dir.path()).unwrap();

        assert_eq!(saved.len(), 3);
        assert!(saved.contains(&VectorEngine::DEFAULT_COLLECTION.to_string()));
        assert!(saved.contains(&"coll_a".to_string()));
        assert!(saved.contains(&"coll_b".to_string()));

        // Check files exist
        assert!(dir.path().join("default.json").exists());
        assert!(dir.path().join("coll_a.json").exists());
        assert!(dir.path().join("coll_b.json").exists());
    }

    #[test]
    fn load_all_indices_basic() {
        let dir = tempfile::tempdir().unwrap();

        // Create and save
        let engine = VectorEngine::new();
        engine
            .store_embedding("default_vec", vec![1.0, 2.0])
            .unwrap();
        engine
            .create_collection("coll_a", VectorCollectionConfig::default())
            .unwrap();
        engine
            .store_in_collection("coll_a", "vec_a", vec![3.0, 4.0])
            .unwrap();

        engine.save_all_indices(dir.path()).unwrap();

        // Load into fresh engine
        let engine2 = VectorEngine::new();
        let loaded = engine2.load_all_indices(dir.path()).unwrap();

        assert_eq!(loaded.len(), 2);

        // Verify data
        let default_vec = engine2.get_embedding("default_vec").unwrap();
        assert_eq!(default_vec, vec![1.0, 2.0]);

        let vec_a = engine2.get_from_collection("coll_a", "vec_a").unwrap();
        assert_eq!(vec_a, vec![3.0, 4.0]);
    }

    #[test]
    fn save_empty_collection_skipped() {
        let dir = tempfile::tempdir().unwrap();

        let engine = VectorEngine::new();
        // Create empty collection
        engine
            .create_collection("empty", VectorCollectionConfig::default())
            .unwrap();

        let saved = engine.save_all_indices(dir.path()).unwrap();

        // Empty collections should not be saved
        assert!(saved.is_empty());
        assert!(!dir.path().join("empty.json").exists());
    }

    #[test]
    fn index_roundtrip_preserves_collection_config() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("coll.json");

        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default()
            .with_dimension(128)
            .with_metric(DistanceMetric::Euclidean)
            .with_auto_index(500);

        engine.create_collection("custom", config.clone()).unwrap();
        engine
            .store_in_collection("custom", "vec1", vec![1.0; 128])
            .unwrap();

        engine.save_index("custom", &path).unwrap();

        let engine2 = VectorEngine::new();
        engine2.load_index(&path).unwrap();

        let loaded_config = engine2.get_collection_config("custom").unwrap();
        assert_eq!(loaded_config.dimension, Some(128));
        assert_eq!(loaded_config.distance_metric, DistanceMetric::Euclidean);
        assert!(loaded_config.auto_index);
        assert_eq!(loaded_config.auto_index_threshold, 500);
    }

    #[test]
    fn load_index_io_error() {
        let engine = VectorEngine::new();
        let result = engine.load_index("/nonexistent/path/index.json");
        assert!(matches!(result, Err(VectorError::IoError(_))));
    }

    #[test]
    fn load_index_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.json");

        fs::write(&path, "not valid json").unwrap();

        let engine = VectorEngine::new();
        let result = engine.load_index(&path);
        assert!(matches!(result, Err(VectorError::SerializationError(_))));
    }

    #[test]
    fn load_index_binary_invalid() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("invalid.bin");

        fs::write(&path, &[0xFF, 0xFF, 0xFF]).unwrap();

        let engine = VectorEngine::new();
        let result = engine.load_index_binary(&path);
        assert!(matches!(result, Err(VectorError::SerializationError(_))));
    }

    #[test]
    fn load_index_file_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.json");

        // Create a file larger than 100 bytes
        fs::write(&path, vec![b'x'; 200]).unwrap();

        let config = VectorEngineConfig::default().with_max_index_file_bytes(100);
        let engine = VectorEngine::with_config(config).unwrap();
        let result = engine.load_index(&path);
        assert!(matches!(
            result,
            Err(VectorError::ConfigurationError(msg)) if msg.contains("exceeds limit")
        ));
    }

    #[test]
    fn load_index_binary_file_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("large.bin");

        // Create a file larger than 100 bytes
        fs::write(&path, vec![0u8; 200]).unwrap();

        let config = VectorEngineConfig::default().with_max_index_file_bytes(100);
        let engine = VectorEngine::with_config(config).unwrap();
        let result = engine.load_index_binary(&path);
        assert!(matches!(
            result,
            Err(VectorError::ConfigurationError(msg)) if msg.contains("exceeds limit")
        ));
    }

    #[test]
    fn load_index_entry_count_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("index.json");

        // Create valid index with 5 entries
        let index = PersistentVectorIndex {
            collection: "test".to_string(),
            config: VectorCollectionConfig::default(),
            vectors: (0..5)
                .map(|i| VectorEntry {
                    key: format!("key{i}"),
                    vector: vec![1.0, 2.0, 3.0],
                    metadata: None,
                })
                .collect(),
            created_at: 0,
            version: 1,
        };
        let json = serde_json::to_string(&index).unwrap();
        fs::write(&path, json).unwrap();

        // Limit to 2 entries
        let config = VectorEngineConfig::default().with_max_index_entries(2);
        let engine = VectorEngine::with_config(config).unwrap();
        let result = engine.load_index(&path);
        assert!(matches!(
            result,
            Err(VectorError::ConfigurationError(msg)) if msg.contains("entry count") && msg.contains("exceeds limit")
        ));
    }

    #[test]
    fn vector_entry_serialization() {
        let entry = VectorEntry {
            key: "test".to_string(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: VectorEntry = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.key, "test");
        assert_eq!(deserialized.vector, vec![1.0, 2.0, 3.0]);
        assert!(deserialized.metadata.is_none());
    }

    #[test]
    fn metadata_value_serialization() {
        let values = vec![
            MetadataValue::Null,
            MetadataValue::Bool(true),
            MetadataValue::Int(42),
            MetadataValue::Float(3.14),
            MetadataValue::String("hello".to_string()),
        ];

        for value in values {
            let json = serde_json::to_string(&value).unwrap();
            let deserialized: MetadataValue = serde_json::from_str(&json).unwrap();

            match (&value, &deserialized) {
                (MetadataValue::Null, MetadataValue::Null) => {},
                (MetadataValue::Bool(a), MetadataValue::Bool(b)) => assert_eq!(a, b),
                (MetadataValue::Int(a), MetadataValue::Int(b)) => assert_eq!(a, b),
                (MetadataValue::Float(a), MetadataValue::Float(b)) => {
                    assert!((a - b).abs() < 1e-10);
                },
                (MetadataValue::String(a), MetadataValue::String(b)) => assert_eq!(a, b),
                _ => panic!("Type mismatch"),
            }
        }
    }

    // ========== Search Timeout Tests ==========

    #[test]
    fn config_with_search_timeout() {
        let config = VectorEngineConfig::default().with_search_timeout(Duration::from_secs(5));
        assert_eq!(config.search_timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn search_timeout_error_display() {
        let err = VectorError::SearchTimeout {
            operation: "search_similar".to_string(),
            timeout_ms: 5000,
        };
        let display = err.to_string();
        assert!(display.contains("search_similar"));
        assert!(display.contains("5000"));
    }

    #[test]
    fn search_similar_respects_timeout() {
        let config = VectorEngineConfig::default().with_search_timeout(Duration::from_nanos(1));
        let engine = VectorEngine::with_config(config).unwrap();

        for i in 0..1000 {
            engine
                .store_embedding(&format!("v{i}"), vec![i as f32; 128])
                .unwrap();
        }

        let result = engine.search_similar(&[0.5f32; 128], 10);
        assert!(matches!(result, Err(VectorError::SearchTimeout { .. })));
    }

    #[test]
    fn search_similar_no_timeout_when_none() {
        let engine = VectorEngine::new();
        for i in 0..100 {
            engine
                .store_embedding(&format!("v{i}"), vec![i as f32, 0.0])
                .unwrap();
        }
        let result = engine.search_similar(&[50.0, 0.0], 10);
        assert!(result.is_ok());
    }

    #[test]
    fn deadline_never_does_not_expire() {
        let deadline = Deadline::never();
        assert!(!deadline.is_expired());
        assert_eq!(deadline.timeout_ms(), 0);
    }

    #[test]
    fn deadline_from_duration_expires() {
        let deadline = Deadline::from_duration(Some(Duration::from_nanos(1)));
        std::thread::sleep(Duration::from_millis(1));
        assert!(deadline.is_expired());
    }

    #[test]
    fn deadline_none_duration_never_expires() {
        let deadline = Deadline::from_duration(None);
        assert!(!deadline.is_expired());
    }

    #[test]
    fn low_memory_config_has_timeout() {
        let config = VectorEngineConfig::low_memory();
        assert_eq!(config.search_timeout, Some(Duration::from_secs(30)));
    }

    #[test]
    fn high_throughput_config_has_no_timeout() {
        let config = VectorEngineConfig::high_throughput();
        assert!(config.search_timeout.is_none());
    }

    #[test]
    fn search_with_metric_respects_timeout() {
        let config = VectorEngineConfig::default().with_search_timeout(Duration::from_nanos(1));
        let engine = VectorEngine::with_config(config).unwrap();

        for i in 0..1000 {
            engine
                .store_embedding(&format!("v{i}"), vec![i as f32; 128])
                .unwrap();
        }

        let result = engine.search_similar_with_metric(&[0.5f32; 128], 10, DistanceMetric::Cosine);
        assert!(matches!(result, Err(VectorError::SearchTimeout { .. })));
    }

    #[test]
    fn search_entities_respects_timeout() {
        let config = VectorEngineConfig::default().with_search_timeout(Duration::from_nanos(1));
        let engine = VectorEngine::with_config(config).unwrap();

        for i in 0..1000 {
            engine
                .set_entity_embedding(&format!("entity:{i}"), vec![i as f32; 128])
                .unwrap();
        }

        let result = engine.search_entities(&[0.5f32; 128], 10);
        assert!(matches!(result, Err(VectorError::SearchTimeout { .. })));
    }

    // ==================== HNSWBuildOptions tests ====================

    #[test]
    fn hnsw_build_options_default() {
        let options = HNSWBuildOptions::default();
        assert_eq!(options.storage, HNSWStorageStrategy::Dense);
    }

    #[test]
    fn hnsw_build_options_new() {
        let options = HNSWBuildOptions::new();
        assert_eq!(options.storage, HNSWStorageStrategy::Dense);
    }

    #[test]
    fn hnsw_build_options_memory_optimized() {
        let options = HNSWBuildOptions::memory_optimized();
        assert_eq!(options.storage, HNSWStorageStrategy::Quantized);
    }

    #[test]
    fn hnsw_build_options_high_recall() {
        let options = HNSWBuildOptions::high_recall();
        assert_eq!(options.storage, HNSWStorageStrategy::Dense);
    }

    #[test]
    fn hnsw_build_options_sparse_optimized() {
        let options = HNSWBuildOptions::sparse_optimized();
        assert_eq!(options.storage, HNSWStorageStrategy::Auto);
    }

    #[test]
    fn hnsw_build_options_builder_methods() {
        let options = HNSWBuildOptions::new()
            .with_storage(HNSWStorageStrategy::Quantized)
            .with_sparsity_threshold(0.7);

        assert_eq!(options.storage, HNSWStorageStrategy::Quantized);
        assert!((options.hnsw_config.sparsity_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn hnsw_build_options_with_hnsw_config() {
        let config = HNSWConfig::high_recall();
        let options = HNSWBuildOptions::new().with_hnsw_config(config.clone());

        assert_eq!(options.hnsw_config.m, config.m);
        assert_eq!(options.hnsw_config.ef_construction, config.ef_construction);
    }

    #[test]
    fn build_hnsw_index_with_options_dense() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![0.0, 0.0, 1.0]).unwrap();

        let options = HNSWBuildOptions::new().with_storage(HNSWStorageStrategy::Dense);
        let (index, keys) = engine.build_hnsw_index_with_options(options).unwrap();

        assert_eq!(keys.len(), 3);
        assert_eq!(index.len(), 3);

        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn build_hnsw_index_with_options_auto() {
        let engine = VectorEngine::new();

        // Store sparse vectors (70% zeros)
        engine
            .store_embedding(
                "sparse1",
                vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            .unwrap();
        engine
            .store_embedding(
                "sparse2",
                vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            )
            .unwrap();

        let options = HNSWBuildOptions::sparse_optimized();
        let (index, keys) = engine.build_hnsw_index_with_options(options).unwrap();

        assert_eq!(keys.len(), 2);
        assert_eq!(index.len(), 2);
    }

    #[test]
    fn build_hnsw_index_with_options_quantized() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![0.0, 0.0, 1.0]).unwrap();

        let options = HNSWBuildOptions::memory_optimized();
        let (index, keys) = engine.build_hnsw_index_with_options(options).unwrap();

        assert_eq!(keys.len(), 3);
        assert_eq!(index.len(), 3);

        let results = index.search(&[1.0, 0.0, 0.0], 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn build_hnsw_index_with_options_empty_store() {
        let engine = VectorEngine::new();

        let options = HNSWBuildOptions::new();
        let (index, keys) = engine.build_hnsw_index_with_options(options).unwrap();

        assert!(keys.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn build_hnsw_index_with_options_dimension_mismatch() {
        let engine = VectorEngine::new();

        // Store vectors with different dimensions
        engine.store_embedding("a", vec![1.0, 2.0, 3.0]).unwrap();
        engine.store_embedding("b", vec![1.0, 2.0]).unwrap();

        let options = HNSWBuildOptions::new();
        let result = engine.build_hnsw_index_with_options(options);

        // Should fail due to dimension mismatch between vectors
        assert!(matches!(
            result,
            Err(VectorError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn quantized_search_recall() {
        let engine = VectorEngine::new();

        // Create 100 random-ish vectors
        for i in 0..100 {
            let v = create_test_vector(64, i);
            engine.store_embedding(&format!("v{i}"), v).unwrap();
        }

        // Build both dense and quantized indexes
        let (dense_index, _) = engine
            .build_hnsw_index_with_options(HNSWBuildOptions::new())
            .unwrap();
        let (quantized_index, _) = engine
            .build_hnsw_index_with_options(HNSWBuildOptions::memory_optimized())
            .unwrap();

        // Query and compare recall
        let query = create_test_vector(64, 999);
        let k = 10;

        let dense_results = dense_index.search(&query, k);
        let quantized_results = quantized_index.search(&query, k);

        // Get the node IDs from dense results
        let dense_ids: std::collections::HashSet<_> =
            dense_results.iter().map(|(id, _)| *id).collect();
        let quantized_ids: std::collections::HashSet<_> =
            quantized_results.iter().map(|(id, _)| *id).collect();

        // Calculate recall: how many of the quantized results match the dense results
        let matching = quantized_ids.intersection(&dense_ids).count();
        let recall = matching as f32 / k as f32;

        // Quantized should achieve at least 70% recall on this dataset
        // (we use 70% instead of 90% because the test vectors may not be ideal for quantization)
        assert!(
            recall >= 0.7,
            "Quantized recall ({recall}) should be at least 70%"
        );
    }

    #[test]
    fn hnsw_storage_strategy_default() {
        let strategy = HNSWStorageStrategy::default();
        assert_eq!(strategy, HNSWStorageStrategy::Dense);
    }

    #[test]
    fn test_open_durable() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("vector.wal");

        let engine = VectorEngine::open_durable(&wal_path, WalConfig::default()).unwrap();
        assert!(engine.is_durable());

        // Create some data to verify the engine works
        engine
            .store_embedding("test_key", vec![1.0, 2.0, 3.0])
            .unwrap();
        assert!(engine.get_embedding("test_key").is_ok());
    }

    #[test]
    fn test_recover_durable() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("vector.wal");

        // First create a durable engine
        {
            let _engine = VectorEngine::open_durable(&wal_path, WalConfig::default()).unwrap();
            // Engine drops, WAL is closed
        }

        // Recover
        let recovered = VectorEngine::recover(&wal_path, &WalConfig::default(), None);
        assert!(recovered.is_ok());
        assert!(recovered.unwrap().is_durable());
    }

    #[test]
    fn test_is_durable_false_for_in_memory() {
        let engine = VectorEngine::new();
        assert!(!engine.is_durable());
    }

    // ==================== IVFBuildOptions tests ====================

    #[test]
    fn ivf_build_options_default() {
        let options = IVFBuildOptions::default();
        assert_eq!(options.config.num_clusters, 100);
    }

    #[test]
    fn ivf_build_options_new() {
        let options = IVFBuildOptions::new();
        assert_eq!(options.config.num_clusters, 100);
    }

    #[test]
    fn ivf_build_options_flat() {
        let options = IVFBuildOptions::flat(50);
        assert_eq!(options.config.num_clusters, 50);
        assert!(matches!(options.config.storage, IVFStorage::Flat));
    }

    #[test]
    fn ivf_build_options_pq() {
        let pq_config = PQConfig::default();
        let options = IVFBuildOptions::pq(50, pq_config);
        assert_eq!(options.config.num_clusters, 50);
        assert!(matches!(options.config.storage, IVFStorage::PQ(_)));
    }

    #[test]
    fn ivf_build_options_binary() {
        let options = IVFBuildOptions::binary(50);
        assert_eq!(options.config.num_clusters, 50);
        assert!(matches!(options.config.storage, IVFStorage::Binary(_)));
    }

    #[test]
    fn ivf_build_options_with_nprobe() {
        let options = IVFBuildOptions::flat(50).with_nprobe(10);
        assert_eq!(options.config.nprobe, 10);
    }

    #[test]
    fn ivf_build_options_with_num_clusters() {
        let options = IVFBuildOptions::new().with_num_clusters(200);
        assert_eq!(options.config.num_clusters, 200);
    }

    #[test]
    fn ivf_build_options_with_storage() {
        let options = IVFBuildOptions::new().with_storage(IVFStorage::Flat);
        assert!(matches!(options.config.storage, IVFStorage::Flat));
    }

    // ==================== VectorCollectionConfig tests ====================

    #[test]
    fn vector_collection_config_with_dimension() {
        let config = VectorCollectionConfig::default().with_dimension(128);
        assert_eq!(config.dimension, Some(128));
    }

    #[test]
    fn vector_collection_config_with_metric() {
        let config = VectorCollectionConfig::default().with_metric(DistanceMetric::Euclidean);
        assert_eq!(config.distance_metric, DistanceMetric::Euclidean);
    }

    #[test]
    fn vector_collection_config_with_auto_index() {
        let config = VectorCollectionConfig::default().with_auto_index(500);
        assert!(config.auto_index);
        assert_eq!(config.auto_index_threshold, 500);
    }

    // ==================== MetadataValue tests ====================

    #[test]
    fn metadata_value_from_tensor_value_bytes() {
        let bytes_val = TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3]));
        let result = MetadataValue::from_tensor_value(&bytes_val);
        assert!(result.is_none()); // Bytes are not supported
    }

    #[test]
    fn metadata_value_from_tensor_value_vector() {
        let vec_val = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
        let result = MetadataValue::from_tensor_value(&vec_val);
        assert!(result.is_none()); // Vector is not supported
    }

    // ==================== Extended metric conversion tests ====================

    #[test]
    fn distance_metric_to_extended_euclidean() {
        let metric = DistanceMetric::Euclidean;
        let extended: ExtendedDistanceMetric = metric.into();
        assert!(matches!(extended, ExtendedDistanceMetric::Euclidean));
    }

    #[test]
    fn distance_metric_to_extended_cosine() {
        let metric = DistanceMetric::Cosine;
        let extended: ExtendedDistanceMetric = metric.into();
        assert!(matches!(extended, ExtendedDistanceMetric::Cosine));
    }

    #[test]
    fn distance_metric_to_extended_dot_product() {
        let metric = DistanceMetric::DotProduct;
        let extended: ExtendedDistanceMetric = metric.into();
        // DotProduct maps to Cosine as the closest equivalent
        assert!(matches!(extended, ExtendedDistanceMetric::Cosine));
    }

    // ==================== VectorError Display tests ====================

    #[test]
    fn vector_error_display_collection_exists() {
        let err = VectorError::CollectionExists("my_collection".to_string());
        assert!(err.to_string().contains("my_collection"));
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn vector_error_display_collection_not_found() {
        let err = VectorError::CollectionNotFound("missing_collection".to_string());
        assert!(err.to_string().contains("missing_collection"));
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn vector_error_display_io_error() {
        let err = VectorError::IoError("disk full".to_string());
        assert!(err.to_string().contains("disk full"));
        assert!(err.to_string().contains("IO error"));
    }

    #[test]
    fn vector_error_display_serialization_error() {
        let err = VectorError::SerializationError("invalid format".to_string());
        assert!(err.to_string().contains("invalid format"));
        assert!(err.to_string().contains("Serialization"));
    }

    // ==================== FilterValue From tests ====================

    #[test]
    fn filter_value_from_i64() {
        let val: FilterValue = 42i64.into();
        assert!(matches!(val, FilterValue::Int(42)));
    }

    #[test]
    fn filter_value_from_f64() {
        let val: FilterValue = 3.14f64.into();
        assert!(matches!(val, FilterValue::Float(f) if (f - 3.14).abs() < f64::EPSILON));
    }

    #[test]
    fn filter_value_from_string() {
        let val: FilterValue = String::from("hello").into();
        assert!(matches!(val, FilterValue::String(s) if s == "hello"));
    }

    #[test]
    fn filter_value_from_str() {
        let val: FilterValue = "world".into();
        assert!(matches!(val, FilterValue::String(s) if s == "world"));
    }

    #[test]
    fn filter_value_from_bool() {
        let val: FilterValue = true.into();
        assert!(matches!(val, FilterValue::Bool(true)));
    }

    // ==================== Deadline tests ====================

    #[test]
    fn deadline_never_not_expired() {
        let deadline = Deadline::never();
        assert!(!deadline.is_expired());
        assert_eq!(deadline.timeout_ms(), 0);
    }

    #[test]
    fn deadline_from_duration_some() {
        let deadline = Deadline::from_duration(Some(Duration::from_millis(100)));
        assert!(!deadline.is_expired());
        assert_eq!(deadline.timeout_ms(), 100);
    }

    #[test]
    fn deadline_from_duration_none() {
        let deadline = Deadline::from_duration(None);
        assert!(!deadline.is_expired());
        assert_eq!(deadline.timeout_ms(), 0);
    }

    // ==================== SearchResult tests ====================

    #[test]
    fn search_result_new() {
        let result = SearchResult::new("test_key".to_string(), 0.95);
        assert_eq!(result.key, "test_key");
        assert!((result.score - 0.95).abs() < f32::EPSILON);
    }

    // ==================== EmbeddingInput tests ====================

    #[test]
    fn embedding_input_constructor() {
        let input = EmbeddingInput::new("my_key", vec![1.0, 2.0, 3.0]);
        assert_eq!(input.key, "my_key");
        assert_eq!(input.vector, vec![1.0, 2.0, 3.0]);
    }

    // ==================== Batch operations tests ====================

    #[test]
    fn batch_delete_multiple_keys() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0]).unwrap();
        engine.store_embedding("c", vec![1.0, 1.0]).unwrap();

        let deleted = engine
            .batch_delete_embeddings(vec!["a".to_string(), "b".to_string()])
            .unwrap();

        assert_eq!(deleted, 2);
        assert!(!engine.exists("a"));
        assert!(!engine.exists("b"));
        assert!(engine.exists("c"));
    }

    #[test]
    fn batch_delete_partial_exists() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();

        // Try to delete one that exists and one that doesn't
        let deleted = engine
            .batch_delete_embeddings(vec!["a".to_string(), "nonexistent".to_string()])
            .unwrap();

        // Only one was actually deleted
        assert_eq!(deleted, 1);
    }

    // ==================== Additional Pagination tests ====================

    #[test]
    fn pagination_constructor_variants() {
        let p1 = Pagination::new(10, 20);
        assert_eq!(p1.skip, 10);
        assert_eq!(p1.limit, Some(20));
        assert!(!p1.count_total);

        let p2 = p1.with_total();
        assert!(p2.count_total);

        let p3 = Pagination::skip_only(5);
        assert_eq!(p3.skip, 5);
        assert!(p3.limit.is_none());
    }

    #[test]
    fn list_keys_paginated_no_limit_variant() {
        let engine = VectorEngine::new();

        for i in 0..5 {
            engine
                .store_embedding(&format!("key{i}"), vec![i as f32, 0.0])
                .unwrap();
        }

        let pagination = Pagination::skip_only(2);
        let result = engine.list_keys_paginated(pagination);

        assert_eq!(result.items.len(), 3);
    }

    #[test]
    fn search_entities_paginated_no_count_variant() {
        let engine = VectorEngine::new();

        for i in 0..5 {
            engine
                .set_entity_embedding(&format!("entity:{i}"), vec![i as f32, 0.0])
                .unwrap();
        }

        let pagination = Pagination::new(0, 3);
        let result = engine
            .search_entities_paginated(&[2.0, 0.0], 5, pagination)
            .unwrap();

        assert_eq!(result.items.len(), 3);
        assert!(result.total_count.is_none());
        assert!(!result.has_more);
    }

    // ==================== Entity embedding tests ====================

    #[test]
    fn scan_entities_with_embeddings_test() {
        let engine = VectorEngine::new();

        engine
            .set_entity_embedding("user:1", vec![1.0, 0.0])
            .unwrap();
        engine
            .set_entity_embedding("user:2", vec![0.0, 1.0])
            .unwrap();

        let entities = engine.scan_entities_with_embeddings();
        assert_eq!(entities.len(), 2);
    }

    #[test]
    fn count_entities_with_embeddings_test() {
        let engine = VectorEngine::new();

        engine
            .set_entity_embedding("user:1", vec![1.0, 0.0])
            .unwrap();
        engine
            .set_entity_embedding("user:2", vec![0.0, 1.0])
            .unwrap();

        let count = engine.count_entities_with_embeddings();
        assert_eq!(count, 2);
    }

    // ==================== IVF memory estimate tests ====================

    #[test]
    fn estimate_ivf_memory_populated() {
        let engine = VectorEngine::new();

        for i in 0..10 {
            engine
                .store_embedding(&format!("key{i}"), vec![1.0, 2.0, 3.0, 4.0])
                .unwrap();
        }

        let options = IVFBuildOptions::flat(5);
        let memory = engine.estimate_ivf_memory(&options).unwrap();
        assert!(memory > 0);
    }

    #[test]
    fn estimate_ivf_memory_no_vectors() {
        let engine = VectorEngine::new();
        let options = IVFBuildOptions::flat(5);
        let memory = engine.estimate_ivf_memory(&options).unwrap();
        assert_eq!(memory, 0);
    }

    // ==================== Extended distance metric search tests ====================

    #[test]
    fn search_with_hnsw_angular_metric() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![0.0, 0.0, 1.0]).unwrap();

        let (index, keys) = engine.build_hnsw_index_default().unwrap();

        let results = engine
            .search_with_hnsw_and_metric(
                &index,
                &keys,
                &[1.0, 0.0, 0.0],
                3,
                ExtendedDistanceMetric::Angular,
            )
            .unwrap();

        assert_eq!(results.len(), 3);
    }

    // ==================== Collection metadata tests ====================

    #[test]
    fn collection_get_metadata_test() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default();
        engine.create_collection("products", config).unwrap();

        let mut meta = HashMap::new();
        meta.insert(
            "price".to_string(),
            TensorValue::Scalar(ScalarValue::Int(100)),
        );

        engine
            .store_in_collection_with_metadata("products", "item1", vec![1.0, 2.0], meta)
            .unwrap();

        let retrieved = engine.get_collection_metadata("products", "item1").unwrap();
        assert!(retrieved.contains_key("price"));
    }

    #[test]
    fn collection_search_filtered_test() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default();
        engine.create_collection("items", config).unwrap();

        let mut meta1 = HashMap::new();
        meta1.insert(
            "category".to_string(),
            TensorValue::Scalar(ScalarValue::String("A".to_string())),
        );
        engine
            .store_in_collection_with_metadata("items", "item1", vec![1.0, 0.0], meta1)
            .unwrap();

        let mut meta2 = HashMap::new();
        meta2.insert(
            "category".to_string(),
            TensorValue::Scalar(ScalarValue::String("B".to_string())),
        );
        engine
            .store_in_collection_with_metadata("items", "item2", vec![0.0, 1.0], meta2)
            .unwrap();

        let filter =
            FilterCondition::Eq("category".to_string(), FilterValue::String("A".to_string()));
        let results = engine
            .search_filtered_in_collection("items", &[1.0, 0.0], 10, &filter, None)
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "item1");
    }

    // ==================== Collection CRUD tests ====================

    #[test]
    fn collection_delete_from_basic() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default();
        engine.create_collection("test", config).unwrap();

        engine
            .store_in_collection("test", "key1", vec![1.0, 2.0])
            .unwrap();
        assert!(engine.exists_in_collection("test", "key1"));

        engine.delete_from_collection("test", "key1").unwrap();
        assert!(!engine.exists_in_collection("test", "key1"));
    }

    #[test]
    fn collection_exists_in_false() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default();
        engine.create_collection("test", config).unwrap();

        assert!(!engine.exists_in_collection("test", "nonexistent"));
    }

    #[test]
    fn collection_list_keys_basic() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default();
        engine.create_collection("test", config).unwrap();

        engine
            .store_in_collection("test", "key1", vec![1.0, 2.0])
            .unwrap();
        engine
            .store_in_collection("test", "key2", vec![2.0, 3.0])
            .unwrap();

        let keys = engine.list_collection_keys("test");
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn collection_list_keys_empty() {
        let engine = VectorEngine::new();
        let config = VectorCollectionConfig::default();
        engine.create_collection("test", config).unwrap();

        let keys = engine.list_collection_keys("test");
        assert!(keys.is_empty());
    }

    // ==================== Entity embedding edge cases ====================

    #[test]
    fn entity_has_embedding_false() {
        let engine = VectorEngine::new();
        assert!(!engine.entity_has_embedding("nonexistent:key"));
    }

    #[test]
    fn remove_entity_embedding_success() {
        let engine = VectorEngine::new();

        engine
            .set_entity_embedding("user:1", vec![1.0, 2.0])
            .unwrap();
        assert!(engine.entity_has_embedding("user:1"));

        engine.remove_entity_embedding("user:1").unwrap();
        assert!(!engine.entity_has_embedding("user:1"));
    }

    #[test]
    fn remove_entity_embedding_not_found() {
        let engine = VectorEngine::new();

        let result = engine.remove_entity_embedding("nonexistent:key");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn get_entity_embedding_not_found() {
        let engine = VectorEngine::new();

        let result = engine.get_entity_embedding("nonexistent:key");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    // ==================== Search edge cases ====================

    #[test]
    fn search_similar_with_metric_euclidean() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![0.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![3.0, 4.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[0.0, 0.0], 3, DistanceMetric::Euclidean)
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "a");
    }

    #[test]
    fn search_similar_with_metric_dot_product() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[1.0, 0.0], 2, DistanceMetric::DotProduct)
            .unwrap();

        assert_eq!(results.len(), 2);
    }

    // ==================== Compute similarity edge cases ====================

    #[test]
    fn compute_similarity_identical_vectors() {
        let score = VectorEngine::compute_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn compute_similarity_orthogonal_vectors() {
        let score = VectorEngine::compute_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(score.abs() < 0.001);
    }

    // ==================== List keys bounded ====================

    #[test]
    fn list_keys_bounded_with_limit() {
        let config = VectorEngineConfig {
            max_keys_per_scan: Some(100),
            ..Default::default()
        };
        let engine = VectorEngine::with_config(config).unwrap();

        for i in 0..10 {
            engine
                .store_embedding(&format!("key{i}"), vec![i as f32])
                .unwrap();
        }

        let keys = engine.list_keys_bounded();
        assert_eq!(keys.len(), 10);
    }

    // ==================== Clear ====================

    #[test]
    fn clear_all_embeddings() {
        let engine = VectorEngine::new();

        for i in 0..5 {
            engine
                .store_embedding(&format!("key{i}"), vec![i as f32])
                .unwrap();
        }

        assert_eq!(engine.count(), 5);

        let removed = engine.clear().unwrap();
        assert_eq!(removed, 5);
        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn clear_empty_engine() {
        let engine = VectorEngine::new();
        let removed = engine.clear().unwrap();
        assert_eq!(removed, 0);
    }

    // ==================== IVF index tests ====================

    #[test]
    fn build_ivf_index_basic() {
        let engine = VectorEngine::new();

        for i in 0..20 {
            engine
                .store_embedding(&format!("key{i}"), vec![i as f32, (i * 2) as f32])
                .unwrap();
        }

        let options = IVFBuildOptions::flat(5);
        let (index, keys) = engine.build_ivf_index(options).unwrap();

        assert_eq!(keys.len(), 20);
        assert!(index.len() > 0);
    }

    #[test]
    fn build_ivf_index_default_test() {
        let engine = VectorEngine::new();

        for i in 0..10 {
            engine
                .store_embedding(&format!("key{i}"), vec![i as f32, (i * 2) as f32])
                .unwrap();
        }

        let (index, keys) = engine.build_ivf_index_default().unwrap();

        assert_eq!(keys.len(), 10);
        assert!(index.len() > 0);
    }

    #[test]
    fn search_with_ivf_basic() {
        let engine = VectorEngine::new();

        for i in 0..20 {
            engine
                .store_embedding(&format!("key{i}"), vec![i as f32, (i * 2) as f32])
                .unwrap();
        }

        let options = IVFBuildOptions::flat(5);
        let (index, keys) = engine.build_ivf_index(options).unwrap();

        let results = engine
            .search_with_ivf(&index, &keys, &[5.0, 10.0], 3)
            .unwrap();

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_with_ivf_nprobe_basic() {
        let engine = VectorEngine::new();

        for i in 0..20 {
            engine
                .store_embedding(&format!("key{i}"), vec![i as f32, (i * 2) as f32])
                .unwrap();
        }

        let options = IVFBuildOptions::flat(5);
        let (index, keys) = engine.build_ivf_index(options).unwrap();

        let results = engine
            .search_with_ivf_nprobe(&index, &keys, &[5.0, 10.0], 3, 2)
            .unwrap();

        assert_eq!(results.len(), 3);
    }

    // ==================== Dimension and store access ====================

    #[test]
    fn dimension_with_vectors() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 2.0, 3.0]).unwrap();

        let dim = engine.dimension();
        assert_eq!(dim, Some(3));
    }

    #[test]
    fn dimension_empty_store() {
        let engine = VectorEngine::new();
        let dim = engine.dimension();
        assert_eq!(dim, None);
    }

    // ==================== Search with HNSW edge cases ====================

    #[test]
    fn search_with_hnsw_simple() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![0.0, 0.0, 1.0]).unwrap();

        let (index, keys) = engine.build_hnsw_index_default().unwrap();

        let results = engine
            .search_with_hnsw(&index, &keys, &[1.0, 0.0, 0.0], 2)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "a");
    }
}
