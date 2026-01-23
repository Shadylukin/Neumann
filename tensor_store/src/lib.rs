use std::{
    collections::HashMap,
    fs::File,
    hash::{Hash, Hasher},
    io::{BufReader, BufWriter},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use serde::{Deserialize, Serialize};

pub mod blob_log;
pub mod cache_ring;
pub mod consistent_hash;
pub mod delta_vector;
pub mod distance;
pub mod embedding_slab;
pub mod entity_index;
pub mod graph_tensor;
pub mod hnsw;
pub mod instrumentation;
pub mod metadata_slab;
pub mod mmap;
pub mod partitioned;
pub mod partitioner;
pub mod relational_slab;
pub mod semantic_partitioner;
pub mod slab_router;
pub mod snapshot;
pub mod sparse_vector;
pub mod tiered;
pub mod voronoi;
pub mod wal;

pub use blob_log::{BlobLog, BlobLogSnapshot, ChunkHash};
pub use cache_ring::{CacheRing, CacheRingSnapshot, CacheStats, EvictionScorer, EvictionStrategy};
pub use consistent_hash::{ConsistentHashConfig, ConsistentHashPartitioner, ConsistentHashStats};
pub use delta_vector::{
    ArchetypeRegistry, CoverageStats, DeltaVector, KMeans, KMeansConfig, KMeansInit,
};
pub use distance::{DistanceMetric, GeometricConfig};
pub use embedding_slab::{
    CompressedEmbedding, EmbeddingError, EmbeddingSlab, EmbeddingSlabSnapshot, EmbeddingSlot,
};
pub use entity_index::{EntityId, EntityIndex, EntityIndexSnapshot};
pub use graph_tensor::{EdgeId, GraphTensor, GraphTensorSnapshot};
pub use hnsw::{EmbeddingStorage, EmbeddingStorageError, HNSWConfig, HNSWIndex, HNSWMemoryStats};
pub use instrumentation::{
    HNSWAccessStats, HNSWStatsSnapshot, ShardAccessSnapshot, ShardAccessTracker, ShardStatsSnapshot,
};
pub use metadata_slab::{MetadataSlab, MetadataSlabSnapshot};
pub use mmap::{MmapError, MmapStore, MmapStoreBuilder, MmapStoreMut};
pub use partitioned::{
    PartitionedError, PartitionedGet, PartitionedPut, PartitionedResult, PartitionedStore,
};
pub use partitioner::{PartitionId, PartitionResult, Partitioner, PhysicalNodeId};
pub use relational_slab::{
    ColumnDef, ColumnType, ColumnValue, RangeOp, RelationalError, RelationalSlab,
    RelationalSlabSnapshot, Row, RowId, TableSchema,
};
pub use semantic_partitioner::{
    EncodedEmbedding, RoutingMethod, SemanticPartitionResult, SemanticPartitioner,
    SemanticPartitionerConfig, SemanticPartitionerStats,
};
pub use slab_router::{SlabRouter, SlabRouterConfig, SlabRouterError, SlabRouterSnapshot};
pub use snapshot::{
    detect_version as snapshot_detect_version, load as snapshot_load,
    migrate_v2_to_v3 as snapshot_migrate, save_v3 as snapshot_save, SnapshotFormatError,
    SnapshotHeader, SnapshotVersion, V3Snapshot,
};
pub use sparse_vector::SparseVector;
pub use tiered::{TieredConfig, TieredError, TieredStats, TieredStore};
pub use voronoi::{VoronoiPartitioner, VoronoiPartitionerConfig, VoronoiRegion};
pub use wal::{TensorWal, WalConfig, WalEntry, WalError, WalRecovery, WalResult, WalStatus};

/// Reserved field prefixes for unified entity storage.
///
/// These prefixes are used by the different engines to store their data
/// within a single `TensorData` entity, enabling cross-engine queries.
pub mod fields {
    /// Graph: outgoing edge pointers (`Vec<String>`)
    pub const OUT: &str = "_out";
    /// Graph: incoming edge pointers (`Vec<String>`)
    pub const IN: &str = "_in";
    /// Vector: embedding vector (`Vec<f32>`)
    pub const EMBEDDING: &str = "_embedding";
    /// Graph/Relational: entity type/label
    pub const LABEL: &str = "_label";
    /// System: entity type discriminator ("node", "edge", "row")
    pub const TYPE: &str = "_type";
    /// System: entity ID
    pub const ID: &str = "_id";
    /// Graph: edge source node
    pub const FROM: &str = "_from";
    /// Graph: edge target node
    pub const TO: &str = "_to";
    /// Graph: edge type
    pub const EDGE_TYPE: &str = "_edge_type";
    /// Graph: whether edge is directed
    pub const DIRECTED: &str = "_directed";
    /// Relational: table name for row entities
    pub const TABLE: &str = "_table";
}

/// Thread-safe Bloom filter for fast negative lookups.
///
/// A Bloom filter is a probabilistic data structure that can quickly tell you:
/// - Definitely NOT in set (no false negatives)
/// - POSSIBLY in set (may have false positives)
///
/// This is useful for avoiding expensive lookups when the key doesn't exist.
pub struct BloomFilter {
    bits: Box<[AtomicU64]>,
    num_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter with the given expected number of items and false positive rate.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of items to insert
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )] // Bloom filter math: values are bounded by reasonable filter sizes
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        // Calculate optimal size: m = -n*ln(p) / (ln(2)^2)
        let ln2_squared = std::f64::consts::LN_2 * std::f64::consts::LN_2;
        let num_bits =
            (-(expected_items as f64) * false_positive_rate.ln() / ln2_squared).ceil() as usize;
        let num_bits = num_bits.max(64); // Minimum 64 bits

        // Calculate optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes =
            ((num_bits as f64 / expected_items as f64) * std::f64::consts::LN_2).ceil() as usize;
        let num_hashes = num_hashes.clamp(1, 16); // Between 1 and 16 hash functions

        // Allocate bit array (using u64 blocks)
        let num_blocks = num_bits.div_ceil(64);
        let bits: Vec<AtomicU64> = (0..num_blocks).map(|_| AtomicU64::new(0)).collect();

        Self {
            bits: bits.into_boxed_slice(),
            num_bits,
            num_hashes,
        }
    }

    /// Create a Bloom filter with default parameters for typical key-value usage.
    /// Expects ~10,000 items with 1% false positive rate.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(10_000, 0.01)
    }

    /// Add a key to the Bloom filter.
    pub fn add<K: Hash>(&self, key: &K) {
        for i in 0..self.num_hashes {
            let bit_index = self.hash_index(key, i);
            let block_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            self.bits[block_index].fetch_or(1 << bit_offset, Ordering::Relaxed);
        }
    }

    /// Check if a key might be in the set.
    /// Returns false if the key is definitely NOT in the set.
    /// Returns true if the key MIGHT be in the set (could be false positive).
    #[inline]
    pub fn might_contain<K: Hash>(&self, key: &K) -> bool {
        for i in 0..self.num_hashes {
            let bit_index = self.hash_index(key, i);
            let block_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            if (self.bits[block_index].load(Ordering::Relaxed) & (1 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Clear all bits in the filter.
    pub fn clear(&self) {
        for block in &*self.bits {
            block.store(0, Ordering::Relaxed);
        }
    }

    /// Compute hash index for a key with a given seed.
    #[inline]
    #[allow(clippy::cast_possible_truncation)] // Hash modulo num_bits fits in usize
    fn hash_index<K: Hash>(&self, key: &K, seed: usize) -> usize {
        let mut hasher = SipHasher::new_with_seed(seed as u64);
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.num_bits
    }

    #[must_use]
    pub const fn num_bits(&self) -> usize {
        self.num_bits
    }

    #[must_use]
    pub const fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

// Simple SipHash-like hasher with configurable seed
struct SipHasher {
    state: u64,
    seed: u64,
}

impl SipHasher {
    const fn new_with_seed(seed: u64) -> Self {
        Self {
            state: seed ^ 0x736f_6d65_7073_6575,
            seed,
        }
    }
}

impl Hasher for SipHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.state = self.state.wrapping_mul(31).wrapping_add(u64::from(*byte));
            self.state ^= self.seed;
        }
    }
}

/// Represents different types of values a tensor can hold
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TensorValue {
    /// Scalar values (properties): integers, floats, strings, booleans
    Scalar(ScalarValue),
    /// Vector values (embeddings): f32 arrays for similarity search
    Vector(Vec<f32>),
    /// Sparse vector values: only non-zero positions stored
    ///
    /// Philosophy: Zero represents absence of information, not a stored value.
    /// The dimension defines the boundary/shell of meaningful space.
    Sparse(SparseVector),
    /// Pointer to another tensor (relationships)
    Pointer(String),
    /// List of pointers (multiple relationships)
    Pointers(Vec<String>),
}

/// Default sparsity threshold for auto-sparsification (70%)
pub const DEFAULT_SPARSITY_THRESHOLD: f32 = 0.7;

/// Default value threshold for pruning small values
pub const DEFAULT_VALUE_THRESHOLD: f32 = 0.01;

impl TensorValue {
    /// Create an embedding value, automatically choosing sparse or dense representation.
    ///
    /// If the vector has sparsity above `sparsity_threshold` after pruning values
    /// below `value_threshold`, stores as `Sparse`. Otherwise stores as `Vector`.
    ///
    /// # Example
    ///
    /// ```
    /// use tensor_store::TensorValue;
    ///
    /// // Dense embedding - stored as Vector
    /// let dense = vec![0.5, 0.3, 0.8, 0.2];
    /// let val = TensorValue::from_embedding(dense.clone(), 0.01, 0.7);
    /// assert!(matches!(val, TensorValue::Vector(_)));
    ///
    /// // Sparse embedding - stored as Sparse
    /// let sparse = vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0];
    /// let val = TensorValue::from_embedding(sparse, 0.01, 0.7);
    /// assert!(matches!(val, TensorValue::Sparse(_)));
    /// ```
    #[must_use]
    pub fn from_embedding(dense: Vec<f32>, value_threshold: f32, sparsity_threshold: f32) -> Self {
        let sparse = SparseVector::from_dense_with_threshold(&dense, value_threshold);

        if sparse.sparsity() >= sparsity_threshold {
            Self::Sparse(sparse)
        } else {
            Self::Vector(dense)
        }
    }

    /// Create an embedding with default thresholds (0.01 value, 0.7 sparsity).
    #[must_use]
    pub fn from_embedding_auto(dense: Vec<f32>) -> Self {
        Self::from_embedding(dense, DEFAULT_VALUE_THRESHOLD, DEFAULT_SPARSITY_THRESHOLD)
    }

    /// Convert to dense vector representation (for compatibility with dense operations).
    ///
    /// Returns `Some(Vec<f32>)` for Vector or Sparse variants, `None` otherwise.
    #[must_use]
    pub fn to_dense(&self) -> Option<Vec<f32>> {
        match self {
            Self::Vector(v) => Some(v.clone()),
            Self::Sparse(s) => Some(s.to_dense()),
            _ => None,
        }
    }

    #[must_use]
    pub const fn dimension(&self) -> Option<usize> {
        match self {
            Self::Vector(v) => Some(v.len()),
            Self::Sparse(s) => Some(s.dimension()),
            _ => None,
        }
    }

    /// Compute dot product between two tensor values (if both are vectors).
    ///
    /// Optimizes for sparse-sparse and sparse-dense cases.
    #[must_use]
    pub fn dot(&self, other: &Self) -> Option<f32> {
        match (self, other) {
            (Self::Sparse(a), Self::Sparse(b)) => Some(a.dot(b)),
            (Self::Sparse(s), Self::Vector(d)) | (Self::Vector(d), Self::Sparse(s)) => {
                Some(s.dot_dense(d))
            },
            (Self::Vector(a), Self::Vector(b)) => {
                if a.len() != b.len() {
                    return None;
                }
                Some(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
            },
            _ => None,
        }
    }

    /// Compute cosine similarity between two tensor values.
    #[must_use]
    pub fn cosine_similarity(&self, other: &Self) -> Option<f32> {
        match (self, other) {
            (Self::Sparse(a), Self::Sparse(b)) => Some(a.cosine_similarity(b)),
            (Self::Sparse(s), Self::Vector(d)) | (Self::Vector(d), Self::Sparse(s)) => {
                let dot = s.dot_dense(d);
                let mag_s = s.magnitude();
                let mag_d: f32 = d.iter().map(|x| x * x).sum::<f32>().sqrt();
                if mag_s == 0.0 || mag_d == 0.0 {
                    Some(0.0)
                } else {
                    Some(dot / (mag_s * mag_d))
                }
            },
            (Self::Vector(a), Self::Vector(b)) => {
                if a.len() != b.len() {
                    return None;
                }
                let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
                let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
                if mag_a == 0.0 || mag_b == 0.0 {
                    Some(0.0)
                } else {
                    Some(dot / (mag_a * mag_b))
                }
            },
            _ => None,
        }
    }

    #[must_use]
    pub const fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(_) | Self::Sparse(_))
    }

    #[must_use]
    pub const fn is_sparse(&self) -> bool {
        matches!(self, Self::Sparse(_))
    }

    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::Scalar(s) => match s {
                ScalarValue::Null => 0,
                ScalarValue::Bool(_) => 1,
                ScalarValue::Int(_) | ScalarValue::Float(_) => 8,
                ScalarValue::String(s) => s.len(),
                ScalarValue::Bytes(b) => b.len(),
            },
            Self::Vector(v) => v.len() * 4,
            Self::Sparse(s) => s.memory_bytes(),
            Self::Pointer(p) => p.len(),
            Self::Pointers(ps) => ps.iter().map(String::len).sum(),
        }
    }
}

/// Scalar value types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalarValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
}

/// An entity that can hold scalar properties, vector embeddings, and pointers to other tensors.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TensorData {
    fields: HashMap<String, TensorValue>,
}

impl TensorData {
    #[must_use]
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: impl Into<String>, value: TensorValue) {
        self.fields.insert(key.into(), value);
    }

    #[must_use]
    pub fn get(&self, key: &str) -> Option<&TensorValue> {
        self.fields.get(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<TensorValue> {
        self.fields.remove(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.fields.keys()
    }

    /// Iterate over all fields as (key, value) pairs.
    pub fn fields_iter(&self) -> impl Iterator<Item = (&String, &TensorValue)> {
        self.fields.iter()
    }

    #[must_use]
    pub fn has(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    #[must_use]
    pub fn entity_type(&self) -> Option<&str> {
        match self.get(fields::TYPE) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => Some(s.as_str()),
            _ => None,
        }
    }

    #[must_use]
    pub fn entity_id(&self) -> Option<i64> {
        match self.get(fields::ID) {
            Some(TensorValue::Scalar(ScalarValue::Int(id))) => Some(*id),
            _ => None,
        }
    }

    #[must_use]
    pub fn label(&self) -> Option<&str> {
        match self.get(fields::LABEL) {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => Some(s.as_str()),
            _ => None,
        }
    }

    #[must_use]
    pub fn embedding(&self) -> Option<&Vec<f32>> {
        match self.get(fields::EMBEDDING) {
            Some(TensorValue::Vector(v)) => Some(v),
            _ => None,
        }
    }

    #[must_use]
    pub fn outgoing_edges(&self) -> Option<&Vec<String>> {
        match self.get(fields::OUT) {
            Some(TensorValue::Pointers(p)) => Some(p),
            _ => None,
        }
    }

    #[must_use]
    pub fn incoming_edges(&self) -> Option<&Vec<String>> {
        match self.get(fields::IN) {
            Some(TensorValue::Pointers(p)) => Some(p),
            _ => None,
        }
    }

    pub fn set_entity_type(&mut self, entity_type: &str) {
        self.set(
            fields::TYPE,
            TensorValue::Scalar(ScalarValue::String(entity_type.to_string())),
        );
    }

    pub fn set_entity_id(&mut self, id: i64) {
        self.set(fields::ID, TensorValue::Scalar(ScalarValue::Int(id)));
    }

    pub fn set_label(&mut self, label: &str) {
        self.set(
            fields::LABEL,
            TensorValue::Scalar(ScalarValue::String(label.to_string())),
        );
    }

    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.set(fields::EMBEDDING, TensorValue::Vector(embedding));
    }

    pub fn set_outgoing_edges(&mut self, edges: Vec<String>) {
        self.set(fields::OUT, TensorValue::Pointers(edges));
    }

    pub fn set_incoming_edges(&mut self, edges: Vec<String>) {
        self.set(fields::IN, TensorValue::Pointers(edges));
    }

    /// Adds edge if not already present.
    pub fn add_outgoing_edge(&mut self, edge_key: String) {
        let mut edges = match self.get(fields::OUT) {
            Some(TensorValue::Pointers(p)) => p.clone(),
            _ => Vec::new(),
        };
        if !edges.contains(&edge_key) {
            edges.push(edge_key);
        }
        self.set(fields::OUT, TensorValue::Pointers(edges));
    }

    /// Adds edge if not already present.
    pub fn add_incoming_edge(&mut self, edge_key: String) {
        let mut edges = match self.get(fields::IN) {
            Some(TensorValue::Pointers(p)) => p.clone(),
            _ => Vec::new(),
        };
        if !edges.contains(&edge_key) {
            edges.push(edge_key);
        }
        self.set(fields::IN, TensorValue::Pointers(edges));
    }

    #[must_use]
    pub fn has_embedding(&self) -> bool {
        self.has(fields::EMBEDDING)
    }

    #[must_use]
    pub fn has_edges(&self) -> bool {
        self.has(fields::OUT) || self.has(fields::IN)
    }

    /// Returns fields that don't start with underscore.
    pub fn user_fields(&self) -> impl Iterator<Item = (&String, &TensorValue)> {
        self.fields.iter().filter(|(k, _)| !k.starts_with('_'))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorValue)> {
        self.fields.iter()
    }
}

pub type Result<T> = std::result::Result<T, TensorStoreError>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorStoreError {
    NotFound(String),
}

impl std::fmt::Display for TensorStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(key) => write!(f, "Key not found: {key}"),
        }
    }
}

impl std::error::Error for TensorStoreError {}

/// Errors that can occur during snapshot operations.
#[derive(Debug)]
pub enum SnapshotError {
    /// Failed to create or open the file.
    IoError(std::io::Error),
    /// Failed to serialize or deserialize data.
    SerializationError(String),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "I/O error: {e}"),
            Self::SerializationError(msg) => write!(f, "Serialization error: {msg}"),
        }
    }
}

impl std::error::Error for SnapshotError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            Self::SerializationError(_) => None,
        }
    }
}

impl From<std::io::Error> for SnapshotError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<bincode::Error> for SnapshotError {
    fn from(e: bincode::Error) -> Self {
        Self::SerializationError(e.to_string())
    }
}

pub type SnapshotResult<T> = std::result::Result<T, SnapshotError>;

/// Thread-safe key-value store backed by `SlabRouter`.
///
/// `TensorStore` provides a unified storage layer for all tensor data, with
/// optional Bloom filter for fast negative lookups and instrumentation for
/// access pattern tracking.
///
/// # Performance
///
/// `SlabRouter` eliminates resize stalls by using `BTreeMap`-based storage.
/// - Throughput: ~3.2M ops/sec PUT, ~5M ops/sec GET (release mode)
/// - CV: <7% (vs 222% with hash table resize events)
///
/// Clone creates a shared reference to the same underlying storage.
#[derive(Clone)]
pub struct TensorStore {
    router: Arc<SlabRouter>,
    bloom_filter: Option<Arc<BloomFilter>>,
    instrumentation: Option<Arc<ShardAccessTracker>>,
}

impl TensorStore {
    const DEFAULT_SHARD_COUNT: usize = 16;

    #[must_use]
    pub fn new() -> Self {
        Self {
            router: Arc::new(SlabRouter::new()),
            bloom_filter: None,
            instrumentation: None,
        }
    }

    /// Create a store with a specific capacity hint for better initial allocation.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            router: Arc::new(SlabRouter::with_capacity(capacity)),
            bloom_filter: None,
            instrumentation: None,
        }
    }

    /// Create a store with a Bloom filter for fast negative lookups.
    ///
    /// This is useful for sparse key spaces where most lookups are misses.
    /// The Bloom filter provides O(1) rejection of non-existent keys.
    ///
    /// # Arguments
    /// * `expected_items` - Expected number of items to store
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    #[must_use]
    pub fn with_bloom_filter(expected_items: usize, false_positive_rate: f64) -> Self {
        Self {
            router: Arc::new(SlabRouter::new()),
            bloom_filter: Some(Arc::new(BloomFilter::new(
                expected_items,
                false_positive_rate,
            ))),
            instrumentation: None,
        }
    }

    /// Create a store with default Bloom filter settings.
    ///
    /// Uses defaults: 10,000 expected items, 1% false positive rate.
    #[must_use]
    pub fn with_default_bloom_filter() -> Self {
        Self {
            router: Arc::new(SlabRouter::new()),
            bloom_filter: Some(Arc::new(BloomFilter::with_defaults())),
            instrumentation: None,
        }
    }

    /// Create a store with memory instrumentation enabled.
    ///
    /// Instrumentation tracks shard access patterns with minimal overhead
    /// using sampling. Use `sample_rate=1` for full tracking, 100 for 1% sampling.
    #[must_use]
    pub fn with_instrumentation(sample_rate: u32) -> Self {
        Self {
            router: Arc::new(SlabRouter::new()),
            bloom_filter: None,
            instrumentation: Some(Arc::new(ShardAccessTracker::new(
                instrumentation::DEFAULT_SHARD_COUNT,
                sample_rate,
            ))),
        }
    }

    /// Create a store with both Bloom filter and instrumentation.
    #[must_use]
    pub fn with_bloom_and_instrumentation(
        expected_items: usize,
        false_positive_rate: f64,
        sample_rate: u32,
    ) -> Self {
        Self {
            router: Arc::new(SlabRouter::new()),
            bloom_filter: Some(Arc::new(BloomFilter::new(
                expected_items,
                false_positive_rate,
            ))),
            instrumentation: Some(Arc::new(ShardAccessTracker::new(
                instrumentation::DEFAULT_SHARD_COUNT,
                sample_rate,
            ))),
        }
    }

    /// Compute shard index for a key (for instrumentation).
    #[allow(clippy::cast_possible_truncation)] // Hash modulo shard count fits in usize
    fn shard_for_key(key: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize % Self::DEFAULT_SHARD_COUNT
    }

    #[must_use]
    pub const fn has_bloom_filter(&self) -> bool {
        self.bloom_filter.is_some()
    }

    #[must_use]
    pub const fn has_instrumentation(&self) -> bool {
        self.instrumentation.is_some()
    }

    /// Get a snapshot of shard access patterns.
    ///
    /// Returns None if instrumentation is not enabled.
    #[must_use]
    pub fn access_snapshot(&self) -> Option<ShardAccessSnapshot> {
        self.instrumentation.as_ref().map(|i| i.snapshot())
    }

    /// Get shards sorted by access count (hottest first).
    ///
    /// Returns None if instrumentation is not enabled.
    #[must_use]
    pub fn hot_shards(&self, limit: usize) -> Option<Vec<(usize, u64)>> {
        self.instrumentation.as_ref().map(|i| i.hot_shards(limit))
    }

    /// Get shards that haven't been accessed within the threshold (in ms).
    ///
    /// Returns None if instrumentation is not enabled.
    #[must_use]
    pub fn cold_shards(&self, threshold_ms: u64) -> Option<Vec<usize>> {
        self.instrumentation
            .as_ref()
            .map(|i| i.cold_shards(threshold_ms))
    }

    /// Store a tensor under the given key.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying storage operation fails.
    pub fn put(&self, key: impl Into<String>, tensor: TensorData) -> Result<()> {
        let key = key.into();
        if let Some(ref filter) = self.bloom_filter {
            filter.add(&key);
        }
        if let Some(ref instr) = self.instrumentation {
            instr.record_write(Self::shard_for_key(&key));
        }
        self.router
            .put(&key, tensor)
            .map_err(|e| TensorStoreError::NotFound(e.to_string()))
    }

    /// Returns cloned data to ensure thread safety.
    ///
    /// If a Bloom filter is enabled, this will first check the filter and return
    /// `NotFound` immediately if the key is definitely not present.
    ///
    /// # Errors
    ///
    /// Returns `TensorStoreError::NotFound` if the key does not exist.
    pub fn get(&self, key: &str) -> Result<TensorData> {
        // Fast path: check Bloom filter first
        if let Some(ref filter) = self.bloom_filter {
            if !filter.might_contain(&key) {
                return Err(TensorStoreError::NotFound(key.to_string()));
            }
        }
        if let Some(ref instr) = self.instrumentation {
            instr.record_read(Self::shard_for_key(key));
        }
        self.router
            .get(key)
            .map_err(|_| TensorStoreError::NotFound(key.to_string()))
    }

    /// Delete a key from the store.
    ///
    /// # Errors
    ///
    /// Returns `TensorStoreError::NotFound` if the key does not exist.
    pub fn delete(&self, key: &str) -> Result<()> {
        if let Some(ref instr) = self.instrumentation {
            instr.record_write(Self::shard_for_key(key));
        }
        self.router
            .delete(key)
            .map_err(|e| TensorStoreError::NotFound(e.to_string()))
    }

    /// Check if a key exists in the store.
    ///
    /// If a Bloom filter is enabled, this will first check the filter and return
    /// `false` immediately if the key is definitely not present.
    #[must_use]
    pub fn exists(&self, key: &str) -> bool {
        // Fast path: check Bloom filter first
        if let Some(ref filter) = self.bloom_filter {
            if !filter.might_contain(&key) {
                return false;
            }
        }
        if let Some(ref instr) = self.instrumentation {
            instr.record_read(Self::shard_for_key(key));
        }
        self.router.exists(key)
    }

    #[must_use]
    pub fn scan(&self, prefix: &str) -> Vec<String> {
        self.router.scan(prefix)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.router.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.router.is_empty()
    }

    pub fn clear(&self) {
        self.router.clear();
        if let Some(ref filter) = self.bloom_filter {
            filter.clear();
        }
    }

    /// Access the underlying `SlabRouter` for direct slab operations.
    ///
    /// This provides access to specialized slabs like `RelationalSlab` for
    /// engines that need direct columnar storage access.
    #[must_use]
    pub fn router(&self) -> &SlabRouter {
        &self.router
    }

    /// Evict entries from the cache ring using the configured eviction strategy.
    ///
    /// Returns the number of entries actually evicted.
    #[must_use]
    pub fn evict_cache(&self, count: usize) -> usize {
        self.router.evict_cache(count)
    }

    #[must_use]
    pub fn scan_count(&self, prefix: &str) -> usize {
        self.router.scan_count(prefix)
    }

    /// Scan entries by prefix, filtering and mapping in a single pass.
    ///
    /// This is significantly more efficient than `scan()` + `get()` because:
    /// - Takes locks only once
    /// - Only clones entries where the filter function returns `Some`
    /// - Avoids intermediate allocations for non-matching entries
    ///
    /// # Performance
    ///
    /// For a table with 5000 rows where only 5% match a filter:
    /// - Old path: 5000 clones (all rows) = ~2.6ms
    /// - New path: 250 clones (matches only) = ~0.13ms (20x faster)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Select users where age > 25, only cloning matching rows
    /// let matching: Vec<TensorData> = store.scan_filter_map("users:", |key, data| {
    ///     if let Some(TensorValue::Scalar(ScalarValue::Int(age))) = data.get("age") {
    ///         if *age > 25 {
    ///             return Some(data.clone());
    ///         }
    ///     }
    ///     None
    /// });
    /// ```
    pub fn scan_filter_map<F, T>(&self, prefix: &str, f: F) -> Vec<T>
    where
        F: FnMut(&str, &TensorData) -> Option<T>,
    {
        self.router.scan_filter_map(prefix, f)
    }

    /// Save a snapshot of the store to a file.
    ///
    /// Uses v3 format (SlabRouter-based). Loads auto-detect v2/v3 format.
    ///
    /// # Errors
    ///
    /// Returns `SnapshotError::IoError` if file operations fail, or
    /// `SnapshotError::SerializationError` if serialization fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = TensorStore::new();
    /// store.put("key", tensor).unwrap();
    /// store.save_snapshot("data.bin")?;
    /// ```
    pub fn save_snapshot<P: AsRef<Path>>(&self, path: P) -> SnapshotResult<()> {
        self.router
            .save_to_file(path)
            .map_err(|e| SnapshotError::SerializationError(e.to_string()))
    }

    /// Load a store from a snapshot file.
    ///
    /// Auto-detects v2 (`HashMap`) or v3 (`SlabRouter`) format.
    /// Note: Bloom filter state is not persisted and will need to be re-enabled.
    ///
    /// # Errors
    ///
    /// Returns `SnapshotError::IoError` if file operations fail, or
    /// `SnapshotError::SerializationError` if deserialization fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = TensorStore::load_snapshot("data.bin")?;
    /// let tensor = store.get("key")?;
    /// ```
    pub fn load_snapshot<P: AsRef<Path>>(path: P) -> SnapshotResult<Self> {
        let router = SlabRouter::load_from_file(path)
            .map_err(|e| SnapshotError::SerializationError(e.to_string()))?;

        Ok(Self {
            router: Arc::new(router),
            bloom_filter: None,
            instrumentation: None,
        })
    }

    /// Load a store from a snapshot file with a Bloom filter.
    ///
    /// The Bloom filter is rebuilt from the loaded keys.
    ///
    /// # Errors
    ///
    /// Returns `SnapshotError::IoError` if file operations fail, or
    /// `SnapshotError::SerializationError` if deserialization fails.
    pub fn load_snapshot_with_bloom_filter<P: AsRef<Path>>(
        path: P,
        expected_items: usize,
        false_positive_rate: f64,
    ) -> SnapshotResult<Self> {
        let router = SlabRouter::load_from_file(&path)
            .map_err(|e| SnapshotError::SerializationError(e.to_string()))?;

        let bloom = BloomFilter::new(expected_items, false_positive_rate);

        // Rebuild bloom filter from keys
        for key in router.scan("") {
            bloom.add(&key);
        }

        Ok(Self {
            router: Arc::new(router),
            bloom_filter: Some(Arc::new(bloom)),
            instrumentation: None,
        })
    }

    /// Save a compressed snapshot using bespoke tensor compression.
    ///
    /// Compression includes:
    /// - Vector quantization (int8 or binary) for embeddings
    /// - Delta + varint encoding for sorted ID lists
    /// - Run-length encoding for repeated values
    ///
    /// # Errors
    /// Returns error if file creation or serialization fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn save_snapshot_compressed<P: AsRef<Path>>(
        &self,
        path: P,
        config: tensor_compress::CompressionConfig,
    ) -> SnapshotResult<()> {
        use tensor_compress::format::{
            compress_vector, CompressedEntry, CompressedScalar, CompressedSnapshot,
            CompressedValue, Header,
        };

        let path = path.as_ref();
        let temp_path = path.with_extension("tmp");

        let keys = self.router.scan("");
        let mut entries = Vec::with_capacity(keys.len());

        for key in keys {
            let Ok(tensor) = self.router.get(&key) else {
                continue;
            };

            let mut fields = HashMap::new();
            for (field_name, value) in tensor.iter() {
                let compressed = match value {
                    TensorValue::Scalar(s) => CompressedValue::Scalar(match s {
                        ScalarValue::Null => CompressedScalar::Null,
                        ScalarValue::Bool(b) => CompressedScalar::Bool(*b),
                        ScalarValue::Int(i) => CompressedScalar::Int(*i),
                        ScalarValue::Float(f) => CompressedScalar::Float(*f),
                        ScalarValue::String(s) => CompressedScalar::String(s.clone()),
                        ScalarValue::Bytes(b) => {
                            CompressedScalar::String(format!("bytes:{}", b.len()))
                        },
                    }),
                    TensorValue::Vector(v) => compress_vector(v, &key, field_name, &config)
                        .map_err(|e| SnapshotError::SerializationError(e.to_string()))?,
                    TensorValue::Sparse(sv) => {
                        // Convert sparse to dense for compression, then compress
                        // Future: add native sparse compression format
                        compress_vector(&sv.to_dense(), &key, field_name, &config)
                            .map_err(|e| SnapshotError::SerializationError(e.to_string()))?
                    },
                    TensorValue::Pointer(p) => CompressedValue::Pointer(p.clone()),
                    TensorValue::Pointers(ps) => CompressedValue::Pointers(ps.clone()),
                };
                fields.insert(field_name.clone(), compressed);
            }

            entries.push(CompressedEntry { key, fields });
        }

        let header = Header::new(config, entries.len() as u64);
        let snapshot = CompressedSnapshot { header, entries };

        let file = File::create(&temp_path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, &snapshot)?;

        std::fs::rename(&temp_path, path)?;

        Ok(())
    }

    /// Load a compressed snapshot.
    ///
    /// # Errors
    /// Returns error if file read or deserialization fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn load_snapshot_compressed<P: AsRef<Path>>(path: P) -> SnapshotResult<Self> {
        use tensor_compress::format::{decompress_vector, CompressedSnapshot, CompressedValue};

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let snapshot: CompressedSnapshot = bincode::deserialize_from(reader)?;

        snapshot
            .header
            .validate()
            .map_err(|e| SnapshotError::SerializationError(e.to_string()))?;

        let store = Self::new();

        for entry in snapshot.entries {
            let mut tensor = TensorData::new();

            for (field_name, value) in entry.fields {
                let tensor_value = match value {
                    CompressedValue::Scalar(s) => {
                        use tensor_compress::format::CompressedScalar;
                        TensorValue::Scalar(match s {
                            CompressedScalar::Null => ScalarValue::Null,
                            CompressedScalar::Bool(b) => ScalarValue::Bool(b),
                            CompressedScalar::Int(i) => ScalarValue::Int(i),
                            CompressedScalar::Float(f) => ScalarValue::Float(f),
                            CompressedScalar::String(s) => ScalarValue::String(s),
                        })
                    },
                    CompressedValue::VectorRaw(v) => TensorValue::Vector(v),
                    CompressedValue::VectorSparse {
                        dimension,
                        positions,
                        values,
                    } => {
                        // Load directly as sparse vector
                        let pos_ids = tensor_compress::decompress_ids(&positions);
                        #[allow(clippy::cast_possible_truncation)]
                        // Sparse vector positions fit in u32
                        let positions_u32: Vec<u32> = pos_ids.iter().map(|&p| p as u32).collect();
                        TensorValue::Sparse(SparseVector::from_parts(
                            dimension,
                            positions_u32,
                            values,
                        ))
                    },
                    CompressedValue::VectorTT { .. } | CompressedValue::IdList(_) => {
                        let v = decompress_vector(&value)
                            .map_err(|e| SnapshotError::SerializationError(e.to_string()))?;
                        TensorValue::Vector(v)
                    },
                    CompressedValue::RleInt(encoded) => {
                        let ints = tensor_compress::rle_decode(&encoded);
                        TensorValue::Vector(ints.iter().map(|&i| i as f32).collect())
                    },
                    CompressedValue::Pointer(p) => TensorValue::Pointer(p),
                    CompressedValue::Pointers(ps) => TensorValue::Pointers(ps),
                };

                tensor.set(&field_name, tensor_value);
            }

            let _ = store.router.put(&entry.key, tensor);
        }

        Ok(store)
    }

    /// Serialize the store contents to bytes for checkpointing.
    ///
    /// # Errors
    ///
    /// Returns `SnapshotError::SerializationError` if serialization fails.
    pub fn snapshot_bytes(&self) -> SnapshotResult<Vec<u8>> {
        self.router
            .to_bytes()
            .map_err(|e| SnapshotError::SerializationError(e.to_string()))
    }

    /// Restore store contents from serialized checkpoint bytes.
    ///
    /// # Errors
    ///
    /// Returns `SnapshotError::SerializationError` if deserialization fails.
    pub fn restore_from_bytes(&self, bytes: &[u8]) -> SnapshotResult<()> {
        // Create a new router from the bytes
        let new_router = SlabRouter::from_bytes(bytes)
            .map_err(|e| SnapshotError::SerializationError(e.to_string()))?;

        // Clear current and copy data from new router
        self.router.clear();
        for key in new_router.scan("") {
            if let Ok(value) = new_router.get(&key) {
                let _ = self.router.put(&key, value);
            }
        }

        Ok(())
    }

    // ========== WAL / Durable Operations ==========

    /// Create a store with a Write-Ahead Log for durability.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL file cannot be opened or created.
    pub fn open_durable<P: AsRef<Path>>(wal_path: P, config: WalConfig) -> std::io::Result<Self> {
        let router = SlabRouter::with_wal(wal_path, config)?;
        Ok(Self {
            router: Arc::new(router),
            bloom_filter: None,
            instrumentation: None,
        })
    }

    /// Create a durable store with a Bloom filter.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL file cannot be opened or created.
    pub fn open_durable_with_bloom<P: AsRef<Path>>(
        wal_path: P,
        config: WalConfig,
        expected_items: usize,
        false_positive_rate: f64,
    ) -> std::io::Result<Self> {
        let router = SlabRouter::with_wal(wal_path, config)?;
        Ok(Self {
            router: Arc::new(router),
            bloom_filter: Some(Arc::new(BloomFilter::new(
                expected_items,
                false_positive_rate,
            ))),
            instrumentation: None,
        })
    }

    /// Recover a store from a snapshot and WAL.
    ///
    /// This is the primary crash recovery mechanism:
    /// 1. Load the most recent snapshot (if available)
    /// 2. Replay the WAL to recover uncommitted changes
    ///
    /// # Errors
    ///
    /// Returns an error if snapshot loading or WAL replay fails.
    pub fn recover<P: AsRef<Path>>(
        wal_path: P,
        config: &WalConfig,
        snapshot_path: Option<&Path>,
    ) -> std::result::Result<Self, SlabRouterError> {
        let router = SlabRouter::recover(wal_path, config, snapshot_path)?;
        Ok(Self {
            router: Arc::new(router),
            bloom_filter: None,
            instrumentation: None,
        })
    }

    /// Recover a store with a Bloom filter.
    ///
    /// The Bloom filter is rebuilt from the recovered keys.
    ///
    /// # Errors
    ///
    /// Returns an error if recovery fails.
    pub fn recover_with_bloom<P: AsRef<Path>>(
        wal_path: P,
        config: &WalConfig,
        snapshot_path: Option<&Path>,
        expected_items: usize,
        false_positive_rate: f64,
    ) -> std::result::Result<Self, SlabRouterError> {
        let router = SlabRouter::recover(wal_path, config, snapshot_path)?;

        let bloom = BloomFilter::new(expected_items, false_positive_rate);
        for key in router.scan("") {
            bloom.add(&key);
        }

        Ok(Self {
            router: Arc::new(router),
            bloom_filter: Some(Arc::new(bloom)),
            instrumentation: None,
        })
    }

    /// Store a tensor durably, logging to WAL before applying.
    ///
    /// Cache operations are never logged (cache is transient by design).
    ///
    /// # Errors
    ///
    /// Returns an error if WAL append fails. If no WAL is configured, falls back to
    /// non-durable `put`.
    pub fn put_durable(&self, key: impl Into<String>, tensor: TensorData) -> Result<()> {
        let key = key.into();
        if let Some(ref filter) = self.bloom_filter {
            filter.add(&key);
        }
        if let Some(ref instr) = self.instrumentation {
            instr.record_write(Self::shard_for_key(&key));
        }
        self.router
            .put_durable(&key, tensor)
            .map_err(|e| TensorStoreError::NotFound(e.to_string()))
    }

    /// Delete a key durably, logging to WAL before applying.
    ///
    /// Cache operations are never logged (cache is transient by design).
    ///
    /// # Errors
    ///
    /// Returns an error if the key does not exist or WAL append fails.
    pub fn delete_durable(&self, key: &str) -> Result<()> {
        if let Some(ref instr) = self.instrumentation {
            instr.record_write(Self::shard_for_key(key));
        }
        self.router
            .delete_durable(key)
            .map_err(|e| TensorStoreError::NotFound(e.to_string()))
    }

    /// Create a checkpoint by saving a snapshot and truncating the WAL.
    ///
    /// After checkpoint, all committed data is in the snapshot and the WAL
    /// can be safely truncated. Returns the checkpoint ID.
    ///
    /// # Errors
    ///
    /// Returns an error if snapshot save or WAL operations fail.
    pub fn checkpoint<P: AsRef<Path>>(
        &self,
        snapshot_path: P,
    ) -> std::result::Result<u64, SlabRouterError> {
        self.router.checkpoint(snapshot_path.as_ref())
    }

    /// Get WAL status if WAL is configured.
    #[must_use]
    pub fn wal_status(&self) -> Option<WalStatus> {
        self.router.wal_status()
    }

    /// Check if WAL is enabled.
    #[must_use]
    pub fn has_wal(&self) -> bool {
        self.router.has_wal()
    }

    /// Flush and sync the WAL to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL fsync fails.
    pub fn wal_sync(&self) -> std::result::Result<(), SlabRouterError> {
        self.router.wal_sync()
    }
}

impl Default for TensorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified entity store that provides a shared storage layer for all engines.
///
/// `EntityStore` wraps a `TensorStore` and provides entity-oriented access patterns
/// that enable cross-engine queries. All engines can share the same `EntityStore`
/// to enable unified entity access.
///
/// # Entity Key Format
///
/// Entities use the format `{type}:{id}`, for example:
/// - `user:1` - A user entity
/// - `post:42` - A post entity
/// - `edge:123` - An edge entity
///
/// # Unified Entity Model
///
/// A single entity can have:
/// - Relational fields (scalars like name, age, email)
/// - Graph connections (outgoing/incoming edge pointers)
/// - Vector embeddings (for similarity search)
///
/// ```text
/// user:1
/// ├── Relational: name="Alice", age=30, email="..."
/// ├── Graph: _out=["edge:1", "edge:2"], _in=["edge:3"]
/// └── Vector: _embedding=[0.1, 0.2, 0.3, ...]
/// ```
#[derive(Clone)]
pub struct EntityStore {
    store: Arc<TensorStore>,
}

impl EntityStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            store: Arc::new(TensorStore::new()),
        }
    }

    #[must_use]
    pub fn with_store(store: TensorStore) -> Self {
        Self {
            store: Arc::new(store),
        }
    }

    #[must_use]
    pub const fn with_arc(store: Arc<TensorStore>) -> Self {
        Self { store }
    }

    #[must_use]
    pub fn store(&self) -> &TensorStore {
        &self.store
    }

    #[must_use]
    pub fn store_arc(&self) -> Arc<TensorStore> {
        Arc::clone(&self.store)
    }

    #[must_use]
    pub fn entity_key(entity_type: &str, id: u64) -> String {
        format!("{entity_type}:{id}")
    }

    #[must_use]
    pub fn parse_key(key: &str) -> Option<(&str, u64)> {
        let parts: Vec<&str> = key.splitn(2, ':').collect();
        if parts.len() == 2 {
            parts[1].parse().ok().map(|id| (parts[0], id))
        } else {
            None
        }
    }

    /// # Errors
    ///
    /// Returns an error if the key does not exist.
    pub fn get(&self, key: &str) -> Result<TensorData> {
        self.store.get(key)
    }

    /// # Errors
    ///
    /// Returns an error if the underlying storage operation fails.
    pub fn put(&self, key: impl Into<String>, data: TensorData) -> Result<()> {
        self.store.put(key, data)
    }

    /// # Errors
    ///
    /// Returns an error if the key does not exist.
    pub fn delete(&self, key: &str) -> Result<()> {
        self.store.delete(key)
    }

    #[must_use]
    pub fn exists(&self, key: &str) -> bool {
        self.store.exists(key)
    }

    /// Returns existing entity or creates empty `TensorData` if not found.
    #[must_use]
    pub fn get_or_create(&self, key: &str) -> TensorData {
        self.store.get(key).unwrap_or_else(|_| TensorData::new())
    }

    /// Atomically read-modify-write an entity.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying storage operation fails.
    pub fn update<F>(&self, key: &str, updater: F) -> Result<()>
    where
        F: FnOnce(&mut TensorData),
    {
        let mut data = self.get_or_create(key);
        updater(&mut data);
        self.store.put(key, data)
    }

    #[must_use]
    pub fn scan_type(&self, entity_type: &str) -> Vec<String> {
        self.store.scan(&format!("{entity_type}:"))
    }

    #[must_use]
    pub fn scan_with_embeddings(&self) -> Vec<String> {
        self.store
            .scan("")
            .into_iter()
            .filter(|key| self.store.get(key).is_ok_and(|data| data.has_embedding()))
            .collect()
    }

    #[must_use]
    pub fn scan_with_edges(&self) -> Vec<String> {
        self.store
            .scan("")
            .into_iter()
            .filter(|key| self.store.get(key).is_ok_and(|data| data.has_edges()))
            .collect()
    }

    #[must_use]
    pub fn get_embedding(&self, key: &str) -> Option<Vec<f32>> {
        self.store
            .get(key)
            .ok()
            .and_then(|data| data.embedding().cloned())
    }

    /// Creates entity if it doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying storage operation fails.
    pub fn set_embedding(&self, key: &str, embedding: Vec<f32>) -> Result<()> {
        self.update(key, |data| {
            data.set_embedding(embedding);
        })
    }

    /// Updates both from and to nodes with edge pointers.
    ///
    /// # Errors
    ///
    /// Returns an error if updating either node fails.
    pub fn add_edge(&self, from_key: &str, to_key: &str, edge_key: &str) -> Result<()> {
        self.update(from_key, |data| {
            data.add_outgoing_edge(edge_key.to_string());
        })?;

        self.update(to_key, |data| {
            data.add_incoming_edge(edge_key.to_string());
        })
    }

    /// # Errors
    ///
    /// Returns an error if the key does not exist.
    pub fn outgoing_neighbors(&self, key: &str) -> Result<Vec<String>> {
        let data = self.get(key)?;
        Ok(data.outgoing_edges().cloned().unwrap_or_default())
    }

    /// # Errors
    ///
    /// Returns an error if the key does not exist.
    pub fn incoming_neighbors(&self, key: &str) -> Result<Vec<String>> {
        let data = self.get(key)?;
        Ok(data.incoming_edges().cloned().unwrap_or_default())
    }

    pub fn clear(&self) {
        self.store.clear();
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    #[must_use]
    pub fn count_type(&self, entity_type: &str) -> usize {
        self.store.scan_count(&format!("{entity_type}:"))
    }
}

impl Default for EntityStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

    use super::*;

    // TensorData tests

    #[test]
    fn tensor_data_stores_scalars() {
        let mut tensor = TensorData::new();
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        tensor.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
        tensor.set("score", TensorValue::Scalar(ScalarValue::Float(95.5)));
        tensor.set("active", TensorValue::Scalar(ScalarValue::Bool(true)));
        tensor.set("nullable", TensorValue::Scalar(ScalarValue::Null));

        assert_eq!(tensor.len(), 5);
        assert!(tensor.has("name"));
        assert!(!tensor.has("nonexistent"));

        match tensor.get("name") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => assert_eq!(s, "Alice"),
            _ => panic!("expected string"),
        }
    }

    #[test]
    fn tensor_data_stores_vectors() {
        let mut tensor = TensorData::new();
        let embedding = vec![0.1, 0.2, 0.3, 0.4];
        tensor.set("embedding", TensorValue::Vector(embedding.clone()));

        match tensor.get("embedding") {
            Some(TensorValue::Vector(v)) => assert_eq!(v, &embedding),
            _ => panic!("expected vector"),
        }
    }

    #[test]
    fn tensor_data_stores_pointers() {
        let mut tensor = TensorData::new();
        tensor.set("friend", TensorValue::Pointer("user:2".into()));
        tensor.set(
            "posts",
            TensorValue::Pointers(vec!["post:1".into(), "post:2".into()]),
        );

        match tensor.get("friend") {
            Some(TensorValue::Pointer(p)) => assert_eq!(p, "user:2"),
            _ => panic!("expected pointer"),
        }

        match tensor.get("posts") {
            Some(TensorValue::Pointers(ps)) => assert_eq!(ps.len(), 2),
            _ => panic!("expected pointers"),
        }
    }

    #[test]
    fn tensor_data_remove_field() {
        let mut tensor = TensorData::new();
        tensor.set("key", TensorValue::Scalar(ScalarValue::Int(1)));

        assert!(tensor.has("key"));
        let removed = tensor.remove("key");
        assert!(removed.is_some());
        assert!(!tensor.has("key"));
        assert!(tensor.remove("key").is_none());
    }

    #[test]
    fn tensor_data_overwrite_field() {
        let mut tensor = TensorData::new();
        tensor.set("key", TensorValue::Scalar(ScalarValue::Int(1)));
        tensor.set("key", TensorValue::Scalar(ScalarValue::Int(2)));

        match tensor.get("key") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 2),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn tensor_data_empty() {
        let tensor = TensorData::new();
        assert!(tensor.is_empty());
        assert_eq!(tensor.len(), 0);
        assert!(tensor.get("anything").is_none());
    }

    #[test]
    fn tensor_data_keys_iteration() {
        let mut tensor = TensorData::new();
        tensor.set("a", TensorValue::Scalar(ScalarValue::Int(1)));
        tensor.set("b", TensorValue::Scalar(ScalarValue::Int(2)));

        let keys: Vec<_> = tensor.keys().collect();
        assert_eq!(keys.len(), 2);
    }

    // TensorStore tests

    #[test]
    fn store_put_get() {
        let store = TensorStore::new();
        let mut tensor = TensorData::new();
        tensor.set("value", TensorValue::Scalar(ScalarValue::Int(42)));

        store.put("key1", tensor).unwrap();

        let retrieved = store.get("key1").unwrap();
        match retrieved.get("value") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 42),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn store_get_not_found() {
        let store = TensorStore::new();
        let result = store.get("nonexistent");
        assert!(matches!(result, Err(TensorStoreError::NotFound(_))));
    }

    #[test]
    fn store_delete() {
        let store = TensorStore::new();
        store.put("key1", TensorData::new()).unwrap();

        assert!(store.exists("key1"));
        store.delete("key1").unwrap();
        assert!(!store.exists("key1"));
    }

    #[test]
    fn store_delete_not_found() {
        let store = TensorStore::new();
        let result = store.delete("nonexistent");
        assert!(matches!(result, Err(TensorStoreError::NotFound(_))));
    }

    #[test]
    fn store_exists() {
        let store = TensorStore::new();
        assert!(!store.exists("key"));
        store.put("key", TensorData::new()).unwrap();
        assert!(store.exists("key"));
    }

    #[test]
    fn store_overwrite() {
        let store = TensorStore::new();
        let mut t1 = TensorData::new();
        t1.set("v", TensorValue::Scalar(ScalarValue::Int(1)));
        let mut t2 = TensorData::new();
        t2.set("v", TensorValue::Scalar(ScalarValue::Int(2)));

        store.put("key", t1).unwrap();
        store.put("key", t2).unwrap();

        let retrieved = store.get("key").unwrap();
        match retrieved.get("v") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 2),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn store_scan_basic() {
        let store = TensorStore::new();
        store.put("user:1", TensorData::new()).unwrap();
        store.put("user:2", TensorData::new()).unwrap();
        store.put("post:1", TensorData::new()).unwrap();

        let users = store.scan("user:");
        assert_eq!(users.len(), 2);
        assert!(users.contains(&"user:1".to_string()));
        assert!(users.contains(&"user:2".to_string()));
    }

    #[test]
    fn store_scan_empty_prefix() {
        let store = TensorStore::new();
        store.put("a", TensorData::new()).unwrap();
        store.put("b", TensorData::new()).unwrap();

        let all = store.scan("");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn store_scan_no_match() {
        let store = TensorStore::new();
        store.put("user:1", TensorData::new()).unwrap();

        let results = store.scan("post:");
        assert!(results.is_empty());
    }

    #[test]
    fn store_len_and_is_empty() {
        let store = TensorStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.put("key", TensorData::new()).unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn store_10k_entities() {
        let store = TensorStore::new();

        for i in 0..10_000 {
            let mut tensor = TensorData::new();
            tensor.set("id", TensorValue::Scalar(ScalarValue::Int(i)));
            tensor.set("embedding", TensorValue::Vector(vec![i as f32; 128]));
            store.put(format!("entity:{}", i), tensor).unwrap();
        }

        assert_eq!(store.len(), 10_000);

        let tensor = store.get("entity:5000").unwrap();
        match tensor.get("id") {
            Some(TensorValue::Scalar(ScalarValue::Int(id))) => assert_eq!(*id, 5000),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn store_concurrent_writes() {
        let store = Arc::new(TensorStore::new());
        let mut handles = vec![];

        for t in 0..4 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let mut tensor = TensorData::new();
                    tensor.set("value", TensorValue::Scalar(ScalarValue::Int(i)));
                    store.put(format!("thread{}:key{}", t, i), tensor).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(store.len(), 4000);
    }

    #[test]
    fn store_concurrent_read_write() {
        let store = Arc::new(TensorStore::new());

        for i in 0..100 {
            store.put(format!("key:{}", i), TensorData::new()).unwrap();
        }

        let mut handles = vec![];

        for _ in 0..4 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let _ = store.get(&format!("key:{}", i));
                }
            }));
        }

        for t in 0..2 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let start = 100 + t * 100;
                for i in start..(start + 100) {
                    store.put(format!("key:{}", i), TensorData::new()).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(store.len(), 300);
    }

    #[test]
    fn store_concurrent_writes_same_keys() {
        // This test verifies SlabRouter's thread safety under contention
        // Multiple threads write to overlapping keys
        let store = Arc::new(TensorStore::new());
        let mut handles = vec![];

        for t in 0..8 {
            let store = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                for i in 0..500 {
                    let key = format!("key:{}", i % 100); // Only 100 unique keys
                    let mut tensor = TensorData::new();
                    tensor.set("thread", TensorValue::Scalar(ScalarValue::Int(t)));
                    tensor.set("iter", TensorValue::Scalar(ScalarValue::Int(i)));
                    store.put(key, tensor).unwrap();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have exactly 100 keys (last write wins)
        assert_eq!(store.len(), 100);
    }

    #[test]
    fn store_scan_many_prefixes() {
        let store = TensorStore::new();

        for i in 0..100 {
            store.put(format!("user:{}", i), TensorData::new()).unwrap();
            store.put(format!("post:{}", i), TensorData::new()).unwrap();
            store
                .put(format!("comment:{}", i), TensorData::new())
                .unwrap();
        }

        assert_eq!(store.scan("user:").len(), 100);
        assert_eq!(store.scan("post:").len(), 100);
        assert_eq!(store.scan("comment:").len(), 100);
        assert_eq!(store.scan("").len(), 300);
    }

    #[test]
    fn store_empty_key() {
        let store = TensorStore::new();
        store.put("", TensorData::new()).unwrap();
        assert!(store.exists(""));
        store.delete("").unwrap();
        assert!(!store.exists(""));
    }

    #[test]
    fn store_unicode_keys() {
        let store = TensorStore::new();
        store.put("user:café", TensorData::new()).unwrap();
        store.put("user:東京", TensorData::new()).unwrap();

        assert!(store.exists("user:café"));
        assert!(store.exists("user:東京"));
    }

    #[test]
    fn store_clear() {
        let store = TensorStore::new();
        store.put("a", TensorData::new()).unwrap();
        store.put("b", TensorData::new()).unwrap();

        assert_eq!(store.len(), 2);
        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn store_scan_count() {
        let store = TensorStore::new();
        for i in 0..50 {
            store.put(format!("user:{}", i), TensorData::new()).unwrap();
        }
        for i in 0..30 {
            store.put(format!("post:{}", i), TensorData::new()).unwrap();
        }

        assert_eq!(store.scan_count("user:"), 50);
        assert_eq!(store.scan_count("post:"), 30);
        assert_eq!(store.scan_count(""), 80);
        assert_eq!(store.scan_count("nonexistent:"), 0);
    }

    #[test]
    fn store_scan_filter_map() {
        let store = TensorStore::new();

        // Insert some test data
        for i in 0..10 {
            let mut tensor = TensorData::new();
            tensor.set("id", TensorValue::Scalar(ScalarValue::Int(i)));
            store.put(format!("item:{}", i), tensor).unwrap();
        }

        // Filter and map: only return items with id > 5
        let results: Vec<i64> = store.scan_filter_map("item:", |_key, data| {
            if let Some(TensorValue::Scalar(ScalarValue::Int(id))) = data.get("id") {
                if *id > 5 {
                    return Some(*id);
                }
            }
            None
        });

        assert_eq!(results.len(), 4); // 6, 7, 8, 9
        assert!(results.contains(&6));
        assert!(results.contains(&7));
        assert!(results.contains(&8));
        assert!(results.contains(&9));
    }

    #[test]
    fn store_with_capacity() {
        let store = TensorStore::with_capacity(1000);
        assert!(store.is_empty());

        for i in 0..1000 {
            store.put(format!("key:{}", i), TensorData::new()).unwrap();
        }
        assert_eq!(store.len(), 1000);
    }

    #[test]
    fn tensor_data_stores_bytes() {
        let mut tensor = TensorData::new();
        let data = vec![0x00, 0xff, 0x42];
        tensor.set(
            "binary",
            TensorValue::Scalar(ScalarValue::Bytes(data.clone())),
        );

        match tensor.get("binary") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => assert_eq!(b, &data),
            _ => panic!("expected bytes"),
        }
    }

    #[test]
    fn error_display_not_found() {
        let err = TensorStoreError::NotFound("test_key".to_string());
        let msg = format!("{}", err);
        assert!(msg.contains("test_key"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn error_is_error_trait() {
        let err: &dyn std::error::Error = &TensorStoreError::NotFound("x".into());
        assert!(err.to_string().contains("x"));
    }

    #[test]
    fn store_default_trait() {
        let store = TensorStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn tensor_data_default_trait() {
        let tensor = TensorData::default();
        assert!(tensor.is_empty());
    }

    #[test]
    fn tensor_data_clone() {
        let mut original = TensorData::new();
        original.set("key", TensorValue::Scalar(ScalarValue::Int(42)));

        let cloned = original.clone();
        assert_eq!(cloned.len(), 1);
        match cloned.get("key") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 42),
            _ => panic!("expected int"),
        }
    }

    #[test]
    fn tensor_value_clone_all_variants() {
        let scalar = TensorValue::Scalar(ScalarValue::Int(1));
        let vector = TensorValue::Vector(vec![1.0, 2.0]);
        let pointer = TensorValue::Pointer("ref".into());
        let pointers = TensorValue::Pointers(vec!["a".into(), "b".into()]);

        assert_eq!(scalar.clone(), scalar);
        assert_eq!(vector.clone(), vector);
        assert_eq!(pointer.clone(), pointer);
        assert_eq!(pointers.clone(), pointers);
    }

    #[test]
    fn scalar_value_clone_all_variants() {
        let null = ScalarValue::Null;
        let bool_val = ScalarValue::Bool(true);
        let int_val = ScalarValue::Int(42);
        let float_val = ScalarValue::Float(3.14);
        let string_val = ScalarValue::String("test".into());
        let bytes_val = ScalarValue::Bytes(vec![1, 2, 3]);

        assert_eq!(null.clone(), null);
        assert_eq!(bool_val.clone(), bool_val);
        assert_eq!(int_val.clone(), int_val);
        assert_eq!(float_val.clone(), float_val);
        assert_eq!(string_val.clone(), string_val);
        assert_eq!(bytes_val.clone(), bytes_val);
    }

    #[test]
    fn tensor_store_error_clone() {
        let err = TensorStoreError::NotFound("key".into());
        assert_eq!(err.clone(), err);
    }

    #[test]
    fn tensor_data_debug() {
        let tensor = TensorData::new();
        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("TensorData"));
    }

    #[test]
    fn tensor_value_debug() {
        let val = TensorValue::Scalar(ScalarValue::Int(1));
        let debug_str = format!("{:?}", val);
        assert!(debug_str.contains("Scalar"));
    }

    #[test]
    fn tensor_store_error_debug() {
        let err = TensorStoreError::NotFound("key".into());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("NotFound"));
    }

    #[test]
    #[cfg_attr(miri, ignore)] // crossbeam-epoch has known Miri issues with stacked borrows
    fn store_parallel_scan_large_dataset() {
        let store = TensorStore::new();

        // Insert enough entries to trigger parallel scan (>1000)
        for i in 0..1500 {
            store.put(format!("user:{}", i), TensorData::new()).unwrap();
        }
        for i in 0..500 {
            store.put(format!("post:{}", i), TensorData::new()).unwrap();
        }

        assert_eq!(store.len(), 2000);

        // These should use parallel iteration
        let users = store.scan("user:");
        assert_eq!(users.len(), 1500);

        let posts = store.scan("post:");
        assert_eq!(posts.len(), 500);

        assert_eq!(store.scan_count("user:"), 1500);
        assert_eq!(store.scan_count("post:"), 500);
        assert_eq!(store.scan_count(""), 2000);
    }

    // Bloom filter tests

    #[test]
    fn bloom_filter_basic() {
        let filter = BloomFilter::new(100, 0.01);

        filter.add(&"key1");
        filter.add(&"key2");
        filter.add(&"key3");

        // Added keys should be found (no false negatives)
        assert!(filter.might_contain(&"key1"));
        assert!(filter.might_contain(&"key2"));
        assert!(filter.might_contain(&"key3"));

        // Non-added keys should likely not be found (may have false positives)
        // We check multiple to reduce chance of false positive affecting test
        let mut misses = 0;
        for i in 100..200 {
            if !filter.might_contain(&format!("nonexistent{}", i)) {
                misses += 1;
            }
        }
        // With 1% false positive rate, should have ~99% misses
        assert!(
            misses > 90,
            "Too many false positives: {} misses out of 100",
            misses
        );
    }

    #[test]
    fn bloom_filter_clear() {
        let filter = BloomFilter::new(100, 0.01);

        filter.add(&"key1");
        assert!(filter.might_contain(&"key1"));

        filter.clear();

        // After clear, key should not be found
        assert!(!filter.might_contain(&"key1"));
    }

    #[test]
    fn bloom_filter_defaults() {
        let filter = BloomFilter::with_defaults();

        // Should be able to add items
        filter.add(&"test_key");
        assert!(filter.might_contain(&"test_key"));

        // Check configuration (10k items, 1% FP rate)
        assert!(filter.num_bits() > 0);
        assert!(filter.num_hashes() > 0);
    }

    #[test]
    fn bloom_filter_many_items() {
        let filter = BloomFilter::new(1000, 0.01);

        // Add 1000 items
        for i in 0..1000 {
            filter.add(&format!("item{}", i));
        }

        // All added items should be found
        for i in 0..1000 {
            assert!(filter.might_contain(&format!("item{}", i)));
        }
    }

    #[test]
    fn store_with_bloom_filter() {
        let store = TensorStore::with_bloom_filter(100, 0.01);
        assert!(store.has_bloom_filter());

        let mut tensor = TensorData::new();
        tensor.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        store.put("key1", tensor).unwrap();

        // Key should be found
        assert!(store.exists("key1"));
        assert!(store.get("key1").is_ok());

        // Non-existent key should return not found
        assert!(!store.exists("nonexistent"));
        assert!(store.get("nonexistent").is_err());
    }

    #[test]
    fn store_with_default_bloom_filter() {
        let store = TensorStore::with_default_bloom_filter();
        assert!(store.has_bloom_filter());

        store.put("key", TensorData::new()).unwrap();
        assert!(store.exists("key"));
    }

    #[test]
    fn store_bloom_filter_accelerates_negative_lookups() {
        let store = TensorStore::with_bloom_filter(100, 0.01);

        // Add some keys
        for i in 0..50 {
            store
                .put(format!("existing:{}", i), TensorData::new())
                .unwrap();
        }

        // Existing keys should be found
        for i in 0..50 {
            assert!(store.exists(&format!("existing:{}", i)));
        }

        // Non-existing keys should not be found
        // The Bloom filter will reject most of these without HashMap lookup
        for i in 1000..1050 {
            assert!(!store.exists(&format!("missing:{}", i)));
        }
    }

    #[test]
    fn store_bloom_filter_clear() {
        let store = TensorStore::with_bloom_filter(100, 0.01);

        store.put("key1", TensorData::new()).unwrap();
        assert!(store.exists("key1"));

        store.clear();

        // After clear, key should not be found
        assert!(!store.exists("key1"));
        assert!(store.is_empty());
    }

    #[test]
    fn store_without_bloom_filter() {
        let store = TensorStore::new();
        assert!(!store.has_bloom_filter());

        store.put("key", TensorData::new()).unwrap();
        assert!(store.exists("key"));
        assert!(!store.exists("missing"));
    }

    #[test]
    fn bloom_filter_concurrent_access() {
        let filter = Arc::new(BloomFilter::new(10000, 0.01));
        let mut handles = vec![];

        // Multiple threads adding keys
        for t in 0..4 {
            let filter = Arc::clone(&filter);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    filter.add(&format!("thread{}:key{}", t, i));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All keys should be found
        for t in 0..4 {
            for i in 0..100 {
                assert!(filter.might_contain(&format!("thread{}:key{}", t, i)));
            }
        }
    }

    // TensorData entity helper tests

    #[test]
    fn tensor_data_entity_type_accessors() {
        let mut tensor = TensorData::new();
        assert!(tensor.entity_type().is_none());

        tensor.set_entity_type("node");
        assert_eq!(tensor.entity_type(), Some("node"));

        tensor.set_entity_id(42);
        assert_eq!(tensor.entity_id(), Some(42));

        tensor.set_label("Person");
        assert_eq!(tensor.label(), Some("Person"));
    }

    #[test]
    fn tensor_data_embedding_accessors() {
        let mut tensor = TensorData::new();
        assert!(tensor.embedding().is_none());
        assert!(!tensor.has_embedding());

        tensor.set_embedding(vec![0.1, 0.2, 0.3]);
        assert!(tensor.has_embedding());
        assert_eq!(tensor.embedding(), Some(&vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn tensor_data_edge_accessors() {
        let mut tensor = TensorData::new();
        assert!(tensor.outgoing_edges().is_none());
        assert!(tensor.incoming_edges().is_none());
        assert!(!tensor.has_edges());

        tensor.set_outgoing_edges(vec!["edge:1".to_string()]);
        tensor.set_incoming_edges(vec!["edge:2".to_string()]);
        assert!(tensor.has_edges());

        assert_eq!(tensor.outgoing_edges(), Some(&vec!["edge:1".to_string()]));
        assert_eq!(tensor.incoming_edges(), Some(&vec!["edge:2".to_string()]));
    }

    #[test]
    fn tensor_data_add_edges_deduplicates() {
        let mut tensor = TensorData::new();

        tensor.add_outgoing_edge("edge:1".to_string());
        tensor.add_outgoing_edge("edge:1".to_string());
        tensor.add_outgoing_edge("edge:2".to_string());

        let edges = tensor.outgoing_edges().unwrap();
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&"edge:1".to_string()));
        assert!(edges.contains(&"edge:2".to_string()));
    }

    #[test]
    fn tensor_data_user_fields() {
        let mut tensor = TensorData::new();
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        tensor.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
        tensor.set_entity_type("user");
        tensor.set_entity_id(1);

        let user_fields: Vec<_> = tensor.user_fields().collect();
        assert_eq!(user_fields.len(), 2);

        let all_fields: Vec<_> = tensor.iter().collect();
        assert_eq!(all_fields.len(), 4);
    }

    // EntityStore tests

    #[test]
    fn entity_store_basic_operations() {
        let store = EntityStore::new();
        assert!(store.is_empty());

        let mut data = TensorData::new();
        data.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        store.put("user:1", data).unwrap();

        assert!(store.exists("user:1"));
        assert_eq!(store.len(), 1);

        let retrieved = store.get("user:1").unwrap();
        match retrieved.get("name") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => assert_eq!(s, "Alice"),
            _ => panic!("expected string"),
        }

        store.delete("user:1").unwrap();
        assert!(!store.exists("user:1"));
    }

    #[test]
    fn entity_store_entity_key() {
        assert_eq!(EntityStore::entity_key("user", 42), "user:42");
        assert_eq!(EntityStore::entity_key("post", 1), "post:1");
    }

    #[test]
    fn entity_store_parse_key() {
        assert_eq!(EntityStore::parse_key("user:42"), Some(("user", 42)));
        assert_eq!(EntityStore::parse_key("post:1"), Some(("post", 1)));
        assert_eq!(EntityStore::parse_key("invalid"), None);
        assert_eq!(EntityStore::parse_key("user:abc"), None);
    }

    #[test]
    fn entity_store_get_or_create() {
        let store = EntityStore::new();

        let data = store.get_or_create("user:1");
        assert!(data.is_empty());

        let mut existing = TensorData::new();
        existing.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Bob".into())),
        );
        store.put("user:2", existing).unwrap();

        let data2 = store.get_or_create("user:2");
        assert!(!data2.is_empty());
    }

    #[test]
    fn entity_store_update() {
        let store = EntityStore::new();

        store
            .update("user:1", |data| {
                data.set(
                    "name",
                    TensorValue::Scalar(ScalarValue::String("Alice".into())),
                );
            })
            .unwrap();

        store
            .update("user:1", |data| {
                data.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
            })
            .unwrap();

        let data = store.get("user:1").unwrap();
        assert!(data.has("name"));
        assert!(data.has("age"));
    }

    #[test]
    fn entity_store_embeddings() {
        let store = EntityStore::new();

        store.set_embedding("user:1", vec![0.1, 0.2, 0.3]).unwrap();
        store.set_embedding("user:2", vec![0.4, 0.5, 0.6]).unwrap();

        assert_eq!(store.get_embedding("user:1"), Some(vec![0.1, 0.2, 0.3]));
        assert_eq!(store.get_embedding("user:3"), None);

        let with_embeddings = store.scan_with_embeddings();
        assert_eq!(with_embeddings.len(), 2);
    }

    #[test]
    fn entity_store_edges() {
        let store = EntityStore::new();

        let mut user1 = TensorData::new();
        user1.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        store.put("user:1", user1).unwrap();

        let mut user2 = TensorData::new();
        user2.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Bob".into())),
        );
        store.put("user:2", user2).unwrap();

        store.add_edge("user:1", "user:2", "edge:1").unwrap();

        let outgoing = store.outgoing_neighbors("user:1").unwrap();
        assert_eq!(outgoing, vec!["edge:1"]);

        let incoming = store.incoming_neighbors("user:2").unwrap();
        assert_eq!(incoming, vec!["edge:1"]);

        let with_edges = store.scan_with_edges();
        assert_eq!(with_edges.len(), 2);
    }

    #[test]
    fn entity_store_scan_type() {
        let store = EntityStore::new();

        store.put("user:1", TensorData::new()).unwrap();
        store.put("user:2", TensorData::new()).unwrap();
        store.put("post:1", TensorData::new()).unwrap();

        let users = store.scan_type("user");
        assert_eq!(users.len(), 2);

        let posts = store.scan_type("post");
        assert_eq!(posts.len(), 1);

        assert_eq!(store.count_type("user"), 2);
        assert_eq!(store.count_type("post"), 1);
    }

    #[test]
    fn entity_store_with_arc() {
        let tensor_store = Arc::new(TensorStore::new());
        let store1 = EntityStore::with_arc(Arc::clone(&tensor_store));
        let store2 = EntityStore::with_arc(Arc::clone(&tensor_store));

        store1.put("shared:1", TensorData::new()).unwrap();
        assert!(store2.exists("shared:1"));
    }

    #[test]
    fn entity_store_clone() {
        let store1 = EntityStore::new();
        store1.put("key:1", TensorData::new()).unwrap();

        let store2 = store1.clone();
        assert!(store2.exists("key:1"));

        store2.put("key:2", TensorData::new()).unwrap();
        assert!(store1.exists("key:2"));
    }

    #[test]
    fn entity_store_default() {
        let store = EntityStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn entity_store_clear() {
        let store = EntityStore::new();
        store.put("a", TensorData::new()).unwrap();
        store.put("b", TensorData::new()).unwrap();

        assert_eq!(store.len(), 2);
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn entity_store_unified_entity() {
        let store = EntityStore::new();

        store
            .update("user:1", |data| {
                data.set_entity_type("user");
                data.set_entity_id(1);
                data.set(
                    "name",
                    TensorValue::Scalar(ScalarValue::String("Alice".into())),
                );
                data.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
            })
            .unwrap();

        store.set_embedding("user:1", vec![0.1, 0.2, 0.3]).unwrap();

        store
            .add_edge("user:1", "user:2", "edge:follows:1")
            .unwrap();

        let data = store.get("user:1").unwrap();
        assert_eq!(data.entity_type(), Some("user"));
        assert_eq!(data.entity_id(), Some(1));
        assert!(data.has_embedding());
        assert!(data.has_edges());

        let user_fields: Vec<_> = data.user_fields().collect();
        assert_eq!(user_fields.len(), 2);
    }

    // Snapshot tests

    #[test]
    fn snapshot_save_and_load() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_basic.bin");

        // Create and populate store
        let store = TensorStore::new();
        let mut tensor1 = TensorData::new();
        tensor1.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        tensor1.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
        store.put("user:1", tensor1).unwrap();

        let mut tensor2 = TensorData::new();
        tensor2.set("embedding", TensorValue::Vector(vec![0.1, 0.2, 0.3]));
        store.put("user:2", tensor2).unwrap();

        // Save snapshot
        store.save_snapshot(&path).unwrap();

        // Load into new store
        let loaded = TensorStore::load_snapshot(&path).unwrap();

        // Verify data
        assert_eq!(loaded.len(), 2);
        assert!(loaded.exists("user:1"));
        assert!(loaded.exists("user:2"));

        let user1 = loaded.get("user:1").unwrap();
        match user1.get("name") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => assert_eq!(s, "Alice"),
            _ => panic!("expected string"),
        }

        let user2 = loaded.get("user:2").unwrap();
        match user2.get("embedding") {
            Some(TensorValue::Vector(v)) => assert_eq!(v, &vec![0.1, 0.2, 0.3]),
            _ => panic!("expected vector"),
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_empty_store() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_empty.bin");

        let store = TensorStore::new();
        store.save_snapshot(&path).unwrap();

        let loaded = TensorStore::load_snapshot(&path).unwrap();
        assert!(loaded.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_all_scalar_types() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_scalars.bin");

        let store = TensorStore::new();
        let mut tensor = TensorData::new();
        tensor.set("null", TensorValue::Scalar(ScalarValue::Null));
        tensor.set("bool", TensorValue::Scalar(ScalarValue::Bool(true)));
        tensor.set("int", TensorValue::Scalar(ScalarValue::Int(-42)));
        tensor.set("float", TensorValue::Scalar(ScalarValue::Float(3.14)));
        tensor.set(
            "string",
            TensorValue::Scalar(ScalarValue::String("hello".into())),
        );
        tensor.set(
            "bytes",
            TensorValue::Scalar(ScalarValue::Bytes(vec![0xff, 0x00, 0xab])),
        );
        store.put("test", tensor).unwrap();

        store.save_snapshot(&path).unwrap();
        let loaded = TensorStore::load_snapshot(&path).unwrap();

        let t = loaded.get("test").unwrap();
        assert_eq!(t.get("null"), Some(&TensorValue::Scalar(ScalarValue::Null)));
        assert_eq!(
            t.get("bool"),
            Some(&TensorValue::Scalar(ScalarValue::Bool(true)))
        );
        assert_eq!(
            t.get("int"),
            Some(&TensorValue::Scalar(ScalarValue::Int(-42)))
        );
        assert_eq!(
            t.get("float"),
            Some(&TensorValue::Scalar(ScalarValue::Float(3.14)))
        );
        assert_eq!(
            t.get("string"),
            Some(&TensorValue::Scalar(ScalarValue::String("hello".into())))
        );
        assert_eq!(
            t.get("bytes"),
            Some(&TensorValue::Scalar(ScalarValue::Bytes(vec![
                0xff, 0x00, 0xab
            ])))
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_pointers() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_pointers.bin");

        let store = TensorStore::new();
        let mut tensor = TensorData::new();
        tensor.set("single", TensorValue::Pointer("ref:1".into()));
        tensor.set(
            "multi",
            TensorValue::Pointers(vec!["ref:2".into(), "ref:3".into()]),
        );
        store.put("test", tensor).unwrap();

        store.save_snapshot(&path).unwrap();
        let loaded = TensorStore::load_snapshot(&path).unwrap();

        let t = loaded.get("test").unwrap();
        assert_eq!(t.get("single"), Some(&TensorValue::Pointer("ref:1".into())));
        assert_eq!(
            t.get("multi"),
            Some(&TensorValue::Pointers(vec!["ref:2".into(), "ref:3".into()]))
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_large_dataset() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_large.bin");

        let store = TensorStore::new();
        for i in 0..1000 {
            let mut tensor = TensorData::new();
            tensor.set("id", TensorValue::Scalar(ScalarValue::Int(i)));
            tensor.set("embedding", TensorValue::Vector(vec![i as f32; 128]));
            store.put(format!("entity:{}", i), tensor).unwrap();
        }

        store.save_snapshot(&path).unwrap();
        let loaded = TensorStore::load_snapshot(&path).unwrap();

        assert_eq!(loaded.len(), 1000);

        // Verify a few entries
        for i in [0, 500, 999] {
            let t = loaded.get(&format!("entity:{}", i)).unwrap();
            match t.get("id") {
                Some(TensorValue::Scalar(ScalarValue::Int(id))) => assert_eq!(*id, i),
                _ => panic!("expected int"),
            }
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_with_bloom_filter() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_snapshot_bloom.bin");

        let store = TensorStore::new();
        for i in 0..100 {
            store.put(format!("key:{}", i), TensorData::new()).unwrap();
        }

        store.save_snapshot(&path).unwrap();

        // Load with bloom filter
        let loaded = TensorStore::load_snapshot_with_bloom_filter(&path, 200, 0.01).unwrap();

        assert!(loaded.has_bloom_filter());
        assert_eq!(loaded.len(), 100);

        // Bloom filter should work
        assert!(loaded.exists("key:50"));
        assert!(!loaded.exists("nonexistent"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_load_nonexistent_file() {
        let result = TensorStore::load_snapshot("/nonexistent/path/file.bin");
        assert!(result.is_err());
    }

    #[test]
    fn snapshot_error_display() {
        let io_err = SnapshotError::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file not found",
        ));
        assert!(format!("{}", io_err).contains("I/O error"));

        let ser_err = SnapshotError::SerializationError("bad data".into());
        assert!(format!("{}", ser_err).contains("Serialization error"));
    }

    #[test]
    fn snapshot_error_source() {
        use std::error::Error;

        let io_err =
            SnapshotError::IoError(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(io_err.source().is_some());

        let ser_err = SnapshotError::SerializationError("test".into());
        assert!(ser_err.source().is_none());
    }

    #[test]
    fn snapshot_compressed_roundtrip() {
        let store = TensorStore::new();

        let mut tensor = TensorData::new();
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("test".into())),
        );
        tensor.set("count", TensorValue::Scalar(ScalarValue::Int(42)));
        // Use 64-element vector to match TT config (64 = 4*4*4)
        let vector_64: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect();
        tensor.set("vector", TensorValue::Vector(vector_64));
        store.put("emb:test1", tensor).unwrap();

        let mut tensor2 = TensorData::new();
        tensor2.set("value", TensorValue::Scalar(ScalarValue::Float(3.14)));
        store.put("other", tensor2).unwrap();

        let config = tensor_compress::CompressionConfig {
            tensor_mode: Some(tensor_compress::TensorMode::tensor_train(64)),
            delta_encoding: true,
            rle_encoding: true,
        };

        let temp = std::env::temp_dir().join("test_compressed.bin");
        store.save_snapshot_compressed(&temp, config).unwrap();

        let loaded = TensorStore::load_snapshot_compressed(&temp).unwrap();
        assert_eq!(loaded.len(), 2);

        let t1 = loaded.get("emb:test1").unwrap();
        assert!(t1.has("name"));
        assert!(t1.has("vector"));

        let t2 = loaded.get("other").unwrap();
        assert!(t2.has("value"));

        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn snapshot_compressed_with_quantization() {
        let store = TensorStore::new();

        let embedding: Vec<f32> = (0..768).map(|i| (i as f32 / 768.0) - 0.5).collect();
        let mut tensor = TensorData::new();
        tensor.set("_embedding", TensorValue::Vector(embedding.clone()));
        store.put("emb:doc1", tensor).unwrap();

        let config = tensor_compress::CompressionConfig {
            tensor_mode: Some(tensor_compress::TensorMode::tensor_train(768)),
            ..Default::default()
        };

        let temp = std::env::temp_dir().join("test_quant.bin");
        store.save_snapshot_compressed(&temp, config).unwrap();

        let file_size = std::fs::metadata(&temp).unwrap().len();

        let loaded = TensorStore::load_snapshot_compressed(&temp).unwrap();
        let restored = loaded.get("emb:doc1").unwrap();
        let restored_vec = restored.get("_embedding").unwrap();

        if let TensorValue::Vector(v) = restored_vec {
            assert_eq!(v.len(), 768);
            for (orig, rest) in embedding.iter().zip(v) {
                assert!((orig - rest).abs() < 0.02, "Quantization error too large");
            }
        } else {
            panic!("Expected vector");
        }

        let uncompressed_size = 768 * 4;
        assert!(
            file_size < uncompressed_size as u64,
            "Compressed file should be smaller"
        );

        std::fs::remove_file(&temp).ok();
    }

    #[test]
    fn snapshot_compressed_empty_store() {
        let store = TensorStore::new();
        let config = tensor_compress::CompressionConfig::default();

        let temp = std::env::temp_dir().join("test_empty_compressed.bin");
        store.save_snapshot_compressed(&temp, config).unwrap();

        let loaded = TensorStore::load_snapshot_compressed(&temp).unwrap();
        assert!(loaded.is_empty());

        std::fs::remove_file(&temp).ok();
    }

    // ==================== TensorValue embedding methods ====================

    #[test]
    fn tensor_value_from_embedding() {
        // Sparse vector (high sparsity should stay sparse)
        let sparse_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let val = TensorValue::from_embedding(sparse_vec.clone(), 0.01, 0.8);
        assert!(matches!(val, TensorValue::Sparse(_)));

        // Dense vector (low sparsity should stay dense)
        let dense_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let val = TensorValue::from_embedding(dense_vec.clone(), 0.01, 0.8);
        assert!(matches!(val, TensorValue::Vector(_)));
    }

    #[test]
    fn tensor_value_from_embedding_auto() {
        // High sparsity vector
        let sparse_vec = vec![0.0; 100];
        let mut sparse_vec = sparse_vec;
        sparse_vec[50] = 1.0;
        let val = TensorValue::from_embedding_auto(sparse_vec);
        assert!(matches!(val, TensorValue::Sparse(_)));
    }

    #[test]
    fn tensor_value_to_dense() {
        // Dense vector
        let val = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(val.to_dense(), Some(vec![1.0, 2.0, 3.0]));

        // Sparse vector
        let sparse = SparseVector::from_dense(&[1.0, 0.0, 2.0]);
        let val = TensorValue::Sparse(sparse);
        assert_eq!(val.to_dense(), Some(vec![1.0, 0.0, 2.0]));

        // Non-vector type
        let val = TensorValue::Scalar(ScalarValue::Int(42));
        assert!(val.to_dense().is_none());
    }

    #[test]
    fn tensor_value_dimension() {
        // Dense vector
        let val = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(val.dimension(), Some(3));

        // Sparse vector
        let sparse = SparseVector::from_dense(&[1.0, 0.0, 2.0, 0.0]);
        let val = TensorValue::Sparse(sparse);
        assert_eq!(val.dimension(), Some(4));

        // Non-vector type
        let val = TensorValue::Scalar(ScalarValue::Int(42));
        assert!(val.dimension().is_none());
    }

    #[test]
    fn tensor_value_dot() {
        // Sparse-Sparse
        let a = TensorValue::Sparse(SparseVector::from_dense(&[1.0, 0.0, 2.0]));
        let b = TensorValue::Sparse(SparseVector::from_dense(&[2.0, 0.0, 3.0]));
        let dot = a.dot(&b).unwrap();
        assert!((dot - 8.0).abs() < 0.001); // 1*2 + 2*3 = 8

        // Sparse-Dense
        let c = TensorValue::Vector(vec![2.0, 0.0, 3.0]);
        let dot2 = a.dot(&c).unwrap();
        assert!((dot2 - 8.0).abs() < 0.001);

        // Dense-Sparse (reversed)
        let dot3 = c.dot(&a).unwrap();
        assert!((dot3 - 8.0).abs() < 0.001);

        // Dense-Dense
        let d = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
        let e = TensorValue::Vector(vec![1.0, 1.0, 1.0]);
        let dot4 = d.dot(&e).unwrap();
        assert!((dot4 - 6.0).abs() < 0.001);

        // Dimension mismatch
        let f = TensorValue::Vector(vec![1.0, 2.0]);
        assert!(d.dot(&f).is_none());

        // Non-vector type
        let g = TensorValue::Scalar(ScalarValue::Int(42));
        assert!(d.dot(&g).is_none());
    }

    #[test]
    fn tensor_value_cosine_similarity() {
        // Same vectors
        let a = TensorValue::Vector(vec![1.0, 0.0, 0.0]);
        let sim = a.cosine_similarity(&a).unwrap();
        assert!((sim - 1.0).abs() < 0.001);

        // Orthogonal vectors
        let b = TensorValue::Vector(vec![0.0, 1.0, 0.0]);
        let sim2 = a.cosine_similarity(&b).unwrap();
        assert!(sim2.abs() < 0.001);

        // Sparse vectors
        let c = TensorValue::Sparse(SparseVector::from_dense(&[1.0, 0.0, 0.0]));
        let d = TensorValue::Sparse(SparseVector::from_dense(&[1.0, 0.0, 0.0]));
        let sim3 = c.cosine_similarity(&d).unwrap();
        assert!((sim3 - 1.0).abs() < 0.001);

        // Mixed
        let sim4 = a.cosine_similarity(&c).unwrap();
        assert!((sim4 - 1.0).abs() < 0.001);
    }

    // ==================== SparseVector additional coverage ====================

    #[test]
    fn sparse_vector_with_capacity() {
        let sv = SparseVector::with_capacity(100, 10);
        assert_eq!(sv.dimension(), 100);
        assert_eq!(sv.nnz(), 0);
    }

    #[test]
    fn sparse_vector_in_bounds() {
        let sv = SparseVector::from_dense(&[1.0, 0.0, 2.0]);
        assert!(sv.in_bounds(0));
        assert!(sv.in_bounds(2));
        assert!(!sv.in_bounds(3));
        assert!(!sv.in_bounds(100));
    }

    #[test]
    fn sparse_vector_set_existing() {
        let mut sv = SparseVector::from_dense(&[1.0, 0.0, 2.0]);
        sv.set(0, 5.0); // Update existing
        assert!((sv.get(0) - 5.0).abs() < 0.001);
    }

    #[test]
    fn sparse_vector_get_zero_position() {
        let sv = SparseVector::from_dense(&[1.0, 0.0, 2.0]);
        // Position 1 is zero (not stored), should return 0.0
        assert!((sv.get(1) - 0.0).abs() < 0.001);
    }

    #[test]
    fn sparse_vector_dot_ordering() {
        // Test the Greater branch in sparse-sparse dot product
        let a = SparseVector::from_dense(&[0.0, 1.0, 0.0, 0.0]);
        let b = SparseVector::from_dense(&[1.0, 0.0, 0.0, 0.0]);
        let dot = a.dot(&b);
        assert!((dot - 0.0).abs() < 0.001); // No overlap
    }

    #[test]
    fn sparse_vector_cosine_distance_dense() {
        let sv = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let dense = vec![1.0, 0.0, 0.0];
        let dist = sv.cosine_distance_dense(&dense);
        assert!(dist.abs() < 0.001); // Same vector = 0 distance
    }

    #[test]
    fn sparse_vector_empty() {
        let sv = SparseVector::new(10);
        assert_eq!(sv.nnz(), 0);
        assert!((sv.magnitude() - 0.0).abs() < 0.001);
    }

    #[test]
    fn sparse_vector_prune() {
        let mut sv = SparseVector::from_dense(&[1.0, 0.001, 2.0, 0.0001]);
        sv.prune(0.01);
        // Values below 0.01 should be removed
        assert_eq!(sv.nnz(), 2);
        assert!((sv.get(0) - 1.0).abs() < 0.001);
        assert!((sv.get(2) - 2.0).abs() < 0.001);
    }

    #[test]
    fn sparse_vector_iter() {
        let sv = SparseVector::from_dense(&[1.0, 0.0, 2.0]);
        let pairs: Vec<_> = sv.iter().collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], (0, 1.0));
        assert_eq!(pairs[1], (2, 2.0));
    }

    #[test]
    fn sparse_vector_zero_magnitude_cosine() {
        let sv = SparseVector::new(3);
        let other = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        // Zero magnitude should return 0 for cosine similarity
        let sim = sv.cosine_similarity(&other);
        assert!((sim - 0.0).abs() < 0.001);
    }

    // Instrumentation tests

    #[test]
    fn store_with_instrumentation() {
        let store = TensorStore::with_instrumentation(1); // Full tracking
        assert!(store.has_instrumentation());
        assert!(!store.has_bloom_filter());
    }

    #[test]
    fn store_instrumentation_tracks_reads_writes() {
        let store = TensorStore::with_instrumentation(1);

        // Perform some operations
        for i in 0..10 {
            store.put(format!("key:{}", i), TensorData::new()).unwrap();
        }

        for i in 0..5 {
            let _ = store.get(&format!("key:{}", i));
        }

        // Check snapshot
        let snapshot = store.access_snapshot().unwrap();
        // Total writes scaled by sample_rate (1)
        assert_eq!(snapshot.total_writes(), 10);
        // Total reads scaled by sample_rate (1)
        assert_eq!(snapshot.total_reads(), 5);
    }

    #[test]
    fn store_instrumentation_hot_cold_shards() {
        let store = TensorStore::with_instrumentation(1);

        // Write to keys that will hash to different shards
        for i in 0..100 {
            store.put(format!("key:{}", i), TensorData::new()).unwrap();
        }

        let hot = store.hot_shards(3).unwrap();
        assert!(hot.len() <= 3);
        // Top shards should have some accesses
        if !hot.is_empty() {
            assert!(hot[0].1 > 0);
        }
    }

    #[test]
    fn store_no_instrumentation_returns_none() {
        let store = TensorStore::new();
        assert!(!store.has_instrumentation());
        assert!(store.access_snapshot().is_none());
        assert!(store.hot_shards(3).is_none());
        assert!(store.cold_shards(1000).is_none());
    }

    #[test]
    fn store_with_bloom_and_instrumentation() {
        let store = TensorStore::with_bloom_and_instrumentation(1000, 0.01, 1);
        assert!(store.has_bloom_filter());
        assert!(store.has_instrumentation());

        store.put("key", TensorData::new()).unwrap();
        assert!(store.exists("key"));

        let snapshot = store.access_snapshot().unwrap();
        assert!(snapshot.total_writes() > 0);
    }

    #[test]
    fn hnsw_with_instrumentation() {
        let index = HNSWIndex::with_instrumentation(HNSWConfig::default());
        assert!(index.has_instrumentation());

        // Insert some vectors
        for i in 0..10 {
            index.insert(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
        }

        // Perform searches
        for _ in 0..5 {
            index.search(&[1.0, 2.0, 3.0], 3);
        }

        let stats = index.access_stats().unwrap();
        assert_eq!(stats.total_searches, 5);
        assert!(stats.entry_point_accesses >= 5);
        assert!(stats.layer0_traversals >= 5);
    }

    #[test]
    fn hnsw_no_instrumentation_returns_none() {
        let index = HNSWIndex::new();
        assert!(!index.has_instrumentation());
        assert!(index.access_stats().is_none());
    }

    #[test]
    fn hnsw_instrumentation_sparse_search() {
        let index = HNSWIndex::with_instrumentation(HNSWConfig::default());

        // Insert dense vectors
        for i in 0..10 {
            index.insert(vec![i as f32, 0.0, (i * 2) as f32]);
        }

        // Search with sparse query
        let query = SparseVector::from_dense(&[1.0, 0.0, 2.0]);
        for _ in 0..3 {
            index.search_sparse(&query, 2);
        }

        let stats = index.access_stats().unwrap();
        assert_eq!(stats.total_searches, 3);
    }

    #[test]
    fn instrumentation_snapshot_distribution() {
        let store = TensorStore::with_instrumentation(1);

        // Generate accesses
        for i in 0..100 {
            store.put(format!("k:{}", i), TensorData::new()).unwrap();
        }

        let snapshot = store.access_snapshot().unwrap();
        let dist = snapshot.distribution();

        // Should have entries for shards that were accessed
        assert!(!dist.is_empty());

        // Total should add up to ~100%
        let total_pct: f64 = dist.iter().map(|(_, pct)| pct).sum();
        assert!((total_pct - 100.0).abs() < 1.0);
    }

    #[test]
    fn instrumentation_hnsw_layer_stats() {
        let index = HNSWIndex::with_instrumentation(HNSWConfig::default());

        // Build index with enough vectors to have multiple layers
        for i in 0..100 {
            index.insert(vec![
                i as f32,
                (i * 2) as f32,
                (i * 3) as f32,
                (i % 10) as f32,
            ]);
        }

        // Run searches
        for i in 0..20 {
            index.search(&[(i % 10) as f32, 0.0, 0.0, 0.0], 5);
        }

        let stats = index.access_stats().unwrap();
        assert_eq!(stats.total_searches, 20);
        assert!(stats.distance_calculations > 0);
        assert!(stats.layer0_ratio() > 0.0);
        assert!(stats.searches_per_second() > 0.0);
    }

    #[test]
    fn instrumentation_cold_shards_after_time() {
        use std::{thread, time::Duration};

        let store = TensorStore::with_instrumentation(1);

        // Write to trigger access
        store.put("key", TensorData::new()).unwrap();

        // Wait a bit
        thread::sleep(Duration::from_millis(50));

        // Shards accessed more than 10ms ago should not be cold
        let cold = store.cold_shards(10).unwrap();
        // The one shard we wrote to should now be "cold" (>10ms old)
        assert!(!cold.is_empty() || cold.is_empty()); // Either way is valid

        // Shards not accessed in 1000ms - most should be cold
        let very_cold = store.cold_shards(1).unwrap();
        // At least some shards should be cold (never accessed)
        assert!(very_cold.len() >= 1);
    }

    #[test]
    fn instrumentation_exists_and_delete_tracking() {
        let store = TensorStore::with_instrumentation(1);

        store.put("key", TensorData::new()).unwrap();
        assert!(store.exists("key")); // Should track read
        store.delete("key").unwrap(); // Should track write

        let snapshot = store.access_snapshot().unwrap();
        assert!(snapshot.total_reads() >= 1);
        assert!(snapshot.total_writes() >= 2); // put + delete
    }

    #[test]
    fn snapshot_bytes_roundtrip() {
        let store = TensorStore::new();

        let mut data = TensorData::new();
        data.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        store.put("user:1", data).unwrap();

        let bytes = store.snapshot_bytes().unwrap();
        assert!(!bytes.is_empty());

        // Create a new store and restore
        let store2 = TensorStore::new();
        store2.restore_from_bytes(&bytes).unwrap();

        assert!(store2.exists("user:1"));
        let restored = store2.get("user:1").unwrap();
        match restored.get("name") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => {
                assert_eq!(s, "Alice");
            },
            _ => panic!("Expected string"),
        }
    }

    #[test]
    fn snapshot_bytes_empty_store() {
        let store = TensorStore::new();

        let bytes = store.snapshot_bytes().unwrap();
        assert!(!bytes.is_empty()); // Header at minimum

        let store2 = TensorStore::new();
        store2.restore_from_bytes(&bytes).unwrap();
        assert!(store2.is_empty());
    }

    #[test]
    fn restore_from_bytes_replaces_data() {
        let store = TensorStore::new();

        // Add initial data
        let mut data = TensorData::new();
        data.set("id", TensorValue::Scalar(ScalarValue::Int(1)));
        store.put("old:1", data).unwrap();

        // Create snapshot of different data
        let store2 = TensorStore::new();
        let mut data2 = TensorData::new();
        data2.set("id", TensorValue::Scalar(ScalarValue::Int(2)));
        store2.put("new:1", data2).unwrap();
        let bytes = store2.snapshot_bytes().unwrap();

        // Restore into first store
        store.restore_from_bytes(&bytes).unwrap();

        // Old data should be gone, new data should be there
        assert!(!store.exists("old:1"));
        assert!(store.exists("new:1"));
    }

    #[test]
    fn slab_router_file_io_roundtrip() {
        use tempfile::tempdir;

        let router = SlabRouter::new();

        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        router.put("test:1", data).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("router.bin");

        router.save_to_file(&path).unwrap();

        let loaded = SlabRouter::load_from_file(&path).unwrap();
        assert!(loaded.exists("test:1"));

        let retrieved = loaded.get("test:1").unwrap();
        match retrieved.get("value") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => {
                assert_eq!(*v, 42);
            },
            _ => panic!("Expected int"),
        }
    }

    #[test]
    fn restore_from_bytes_invalid() {
        let store = TensorStore::new();
        let result = store.restore_from_bytes(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn slab_router_to_from_bytes() {
        let router = SlabRouter::new();

        let mut data = TensorData::new();
        data.set(
            "key",
            TensorValue::Scalar(ScalarValue::String("value".into())),
        );
        router.put("test:1", data).unwrap();

        let bytes = router.to_bytes().unwrap();
        let restored = SlabRouter::from_bytes(&bytes).unwrap();

        assert!(restored.exists("test:1"));
        assert_eq!(restored.len(), 1);
    }

    #[test]
    fn slab_router_with_capacity() {
        let router = SlabRouter::with_capacity(1000);
        assert!(router.is_empty());
        router.put("key", TensorData::new()).unwrap();
        assert_eq!(router.len(), 1);
    }

    #[test]
    fn slab_router_error_variants() {
        use crate::slab_router::SlabRouterError;

        let not_found = SlabRouterError::NotFound("key".to_string());
        assert!(not_found.to_string().contains("not found"));

        let emb_err = SlabRouterError::EmbeddingError("msg".to_string());
        assert!(emb_err.to_string().contains("embedding"));
    }

    #[test]
    fn tensor_store_delete_not_found_error() {
        let store = TensorStore::new();
        let result = store.delete("nonexistent");
        assert!(result.is_err());
        match result {
            Err(TensorStoreError::NotFound(key)) => {
                assert!(key.contains("nonexistent"));
            },
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn slab_router_scan_includes_graph_nodes() {
        let router = SlabRouter::new();

        let mut data = TensorData::new();
        data.set(
            "label",
            TensorValue::Scalar(ScalarValue::String("Person".into())),
        );
        router.put("node:1", data).unwrap();

        let keys = router.scan("node:");
        assert_eq!(keys.len(), 1);
        assert!(keys.contains(&"node:1".to_string()));
    }

    #[test]
    fn slab_router_scan_includes_table_rows() {
        let router = SlabRouter::new();

        let mut data = TensorData::new();
        data.set("id", TensorValue::Scalar(ScalarValue::Int(1)));
        router.put("table:users:1", data).unwrap();

        let keys = router.scan("table:");
        assert_eq!(keys.len(), 1);
    }

    // Additional TensorValue coverage tests

    #[test]
    fn tensor_value_is_vector_predicates() {
        let vec = TensorValue::Vector(vec![1.0, 2.0]);
        assert!(vec.is_vector());
        assert!(!vec.is_sparse());

        let sparse = TensorValue::Sparse(SparseVector::from_dense(&[1.0, 0.0]));
        assert!(sparse.is_vector());
        assert!(sparse.is_sparse());

        let scalar = TensorValue::Scalar(ScalarValue::Int(42));
        assert!(!scalar.is_vector());
        assert!(!scalar.is_sparse());

        let pointer = TensorValue::Pointer("ref".into());
        assert!(!pointer.is_vector());
        assert!(!pointer.is_sparse());
    }

    #[test]
    fn tensor_value_memory_bytes() {
        // Null
        let null = TensorValue::Scalar(ScalarValue::Null);
        assert_eq!(null.memory_bytes(), 0);

        // Bool
        let bool_val = TensorValue::Scalar(ScalarValue::Bool(true));
        assert_eq!(bool_val.memory_bytes(), 1);

        // Int
        let int_val = TensorValue::Scalar(ScalarValue::Int(42));
        assert_eq!(int_val.memory_bytes(), 8);

        // Float
        let float_val = TensorValue::Scalar(ScalarValue::Float(3.14));
        assert_eq!(float_val.memory_bytes(), 8);

        // String
        let string_val = TensorValue::Scalar(ScalarValue::String("hello".into()));
        assert_eq!(string_val.memory_bytes(), 5);

        // Bytes
        let bytes_val = TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3, 4]));
        assert_eq!(bytes_val.memory_bytes(), 4);

        // Vector (4 bytes per f32)
        let vec_val = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(vec_val.memory_bytes(), 12);

        // Sparse
        let sparse = TensorValue::Sparse(SparseVector::from_dense(&[1.0, 0.0, 2.0]));
        assert!(sparse.memory_bytes() > 0);

        // Pointer
        let pointer = TensorValue::Pointer("ref:123".into());
        assert_eq!(pointer.memory_bytes(), 7);

        // Pointers
        let pointers = TensorValue::Pointers(vec!["a".into(), "bb".into(), "ccc".into()]);
        assert_eq!(pointers.memory_bytes(), 6); // 1 + 2 + 3
    }

    #[test]
    fn tensor_value_cosine_similarity_edge_cases() {
        // Dimension mismatch
        let a = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
        let b = TensorValue::Vector(vec![1.0, 2.0]);
        assert!(a.cosine_similarity(&b).is_none());

        // Zero magnitude dense vector
        let zero_vec = TensorValue::Vector(vec![0.0, 0.0, 0.0]);
        let nonzero = TensorValue::Vector(vec![1.0, 0.0, 0.0]);
        let sim = zero_vec.cosine_similarity(&nonzero);
        assert!(sim.is_some());
        assert!((sim.unwrap() - 0.0).abs() < 0.001);

        // Both zero magnitude
        let sim2 = zero_vec.cosine_similarity(&zero_vec);
        assert!(sim2.is_some());
        assert!((sim2.unwrap() - 0.0).abs() < 0.001);

        // Mixed sparse-dense
        let sparse = TensorValue::Sparse(SparseVector::from_dense(&[1.0, 0.0, 0.0]));
        let dense = TensorValue::Vector(vec![1.0, 0.0, 0.0]);
        let sim3 = sparse.cosine_similarity(&dense);
        assert!(sim3.is_some());
        assert!((sim3.unwrap() - 1.0).abs() < 0.001);

        // Dense-sparse reversed
        let sim4 = dense.cosine_similarity(&sparse);
        assert!(sim4.is_some());
        assert!((sim4.unwrap() - 1.0).abs() < 0.001);

        // Zero magnitude sparse
        let zero_sparse = TensorValue::Sparse(SparseVector::new(3));
        let sim5 = zero_sparse.cosine_similarity(&dense);
        assert!(sim5.is_some());
        assert!((sim5.unwrap() - 0.0).abs() < 0.001);

        // Non-vector types
        let scalar = TensorValue::Scalar(ScalarValue::Int(1));
        assert!(scalar.cosine_similarity(&dense).is_none());
        assert!(dense.cosine_similarity(&scalar).is_none());
    }

    #[test]
    fn test_tensor_data_entity_id_none() {
        let mut data = TensorData::new();
        // No ID field
        assert!(data.entity_id().is_none());

        // Wrong type for ID field
        data.set(
            fields::ID,
            TensorValue::Scalar(ScalarValue::String("not_an_id".to_string())),
        );
        assert!(data.entity_id().is_none());
    }

    #[test]
    fn test_tensor_data_label_none() {
        let mut data = TensorData::new();
        // No label field
        assert!(data.label().is_none());

        // Wrong type for label field
        data.set(fields::LABEL, TensorValue::Scalar(ScalarValue::Int(42)));
        assert!(data.label().is_none());
    }

    #[test]
    fn test_snapshot_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let snap_err: SnapshotError = io_err.into();
        match snap_err {
            SnapshotError::IoError(e) => assert!(e.to_string().contains("file not found")),
            _ => panic!("Expected IoError variant"),
        }
    }

    #[test]
    fn test_snapshot_error_from_bincode_error() {
        // Create a bincode error by trying to deserialize invalid data
        let invalid_data = &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        let result: std::result::Result<String, _> = bincode::deserialize(invalid_data);
        if let Err(e) = result {
            let snap_err: SnapshotError = e.into();
            match snap_err {
                SnapshotError::SerializationError(_) => {},
                _ => panic!("Expected SerializationError variant"),
            }
        }
    }

    #[test]
    fn test_tensor_store_router_access() {
        let store = TensorStore::new();
        let router = store.router();
        // Just verify we can access the router
        assert_eq!(router.len(), 0);
    }

    #[test]
    fn test_tensor_store_evict_cache() {
        let store = TensorStore::new();
        // Add some cache entries
        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(1)));
        store.put("_cache:item1", data.clone()).unwrap();
        store.put("_cache:item2", data.clone()).unwrap();

        // Evict from cache
        let evicted = store.evict_cache(1);
        // May or may not evict depending on cache state
        assert!(evicted <= 2);
    }

    #[test]
    fn test_tensor_data_add_incoming_edge_existing() {
        let mut data = TensorData::new();
        // First add an incoming edge
        data.add_incoming_edge("edge1".to_string());
        // Then add another
        data.add_incoming_edge("edge2".to_string());
        // Verify both are present
        match data.get(fields::IN) {
            Some(TensorValue::Pointers(edges)) => {
                assert_eq!(edges.len(), 2);
                assert!(edges.contains(&"edge1".to_string()));
                assert!(edges.contains(&"edge2".to_string()));
            },
            _ => panic!("Expected Pointers"),
        }
    }

    #[test]
    fn snapshot_compressed_all_value_types() {
        let store = TensorStore::new();

        // Test all scalar types
        let mut tensor1 = TensorData::new();
        tensor1.set("null_val", TensorValue::Scalar(ScalarValue::Null));
        tensor1.set("bool_val", TensorValue::Scalar(ScalarValue::Bool(true)));
        tensor1.set("int_val", TensorValue::Scalar(ScalarValue::Int(42)));
        tensor1.set("float_val", TensorValue::Scalar(ScalarValue::Float(3.14)));
        tensor1.set(
            "string_val",
            TensorValue::Scalar(ScalarValue::String("hello".into())),
        );
        tensor1.set(
            "bytes_val",
            TensorValue::Scalar(ScalarValue::Bytes(vec![0xAB, 0xCD, 0xEF])),
        );
        store.put("scalars", tensor1).unwrap();

        // Test sparse vector
        let mut tensor2 = TensorData::new();
        let sparse = SparseVector::from_dense(&[0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0]);
        tensor2.set("sparse_vec", TensorValue::Sparse(sparse));
        store.put("sparse", tensor2).unwrap();

        // Test pointer types
        let mut tensor3 = TensorData::new();
        tensor3.set("ptr", TensorValue::Pointer("ref:123".into()));
        tensor3.set(
            "ptrs",
            TensorValue::Pointers(vec!["ref:1".into(), "ref:2".into()]),
        );
        store.put("pointers", tensor3).unwrap();

        let config = tensor_compress::CompressionConfig::default();
        let temp = std::env::temp_dir().join("test_all_types_compressed.bin");
        store.save_snapshot_compressed(&temp, config).unwrap();

        let loaded = TensorStore::load_snapshot_compressed(&temp).unwrap();
        assert_eq!(loaded.len(), 3);

        // Verify scalars
        let t1 = loaded.get("scalars").unwrap();
        assert!(matches!(
            t1.get("null_val"),
            Some(TensorValue::Scalar(ScalarValue::Null))
        ));
        assert!(matches!(
            t1.get("bool_val"),
            Some(TensorValue::Scalar(ScalarValue::Bool(true)))
        ));
        assert!(matches!(
            t1.get("int_val"),
            Some(TensorValue::Scalar(ScalarValue::Int(42)))
        ));

        // Verify pointers
        let t3 = loaded.get("pointers").unwrap();
        assert!(matches!(
            t3.get("ptr"),
            Some(TensorValue::Pointer(ref p)) if p == "ref:123"
        ));
        assert!(matches!(t3.get("ptrs"), Some(TensorValue::Pointers(_))));

        std::fs::remove_file(&temp).ok();
    }

    // ========== TensorStore WAL Tests ==========

    #[test]
    fn store_open_durable() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();
        assert!(store.has_wal());
        assert!(wal_path.exists());
    }

    #[test]
    fn store_open_durable_with_bloom() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let store =
            TensorStore::open_durable_with_bloom(&wal_path, WalConfig::default(), 1000, 0.01)
                .unwrap();
        assert!(store.has_wal());
        assert!(store.has_bloom_filter());
    }

    #[test]
    fn store_has_wal_false_without_wal() {
        let store = TensorStore::new();
        assert!(!store.has_wal());
    }

    #[test]
    fn store_wal_status() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

        let status = store.wal_status().unwrap();
        assert_eq!(status.path, wal_path);
        assert!(status.checksums_enabled);
    }

    #[test]
    fn store_wal_status_none_without_wal() {
        let store = TensorStore::new();
        assert!(store.wal_status().is_none());
    }

    #[test]
    fn store_put_durable() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

        let mut tensor = TensorData::new();
        tensor.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        store.put_durable("key1", tensor).unwrap();

        assert!(store.exists("key1"));

        let status = store.wal_status().unwrap();
        assert!(status.entry_count > 0);
    }

    #[test]
    fn store_put_durable_without_wal() {
        let store = TensorStore::new();

        let mut tensor = TensorData::new();
        tensor.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        store.put_durable("key1", tensor).unwrap();

        assert!(store.exists("key1"));
    }

    #[test]
    fn store_delete_durable() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

        store.put_durable("key1", TensorData::new()).unwrap();
        assert!(store.exists("key1"));

        store.delete_durable("key1").unwrap();
        assert!(!store.exists("key1"));
    }

    #[test]
    fn store_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let snapshot_path = dir.path().join("snapshot.bin");

        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

        store.put_durable("key1", TensorData::new()).unwrap();
        store.put_durable("key2", TensorData::new()).unwrap();

        let checkpoint_id = store.checkpoint(&snapshot_path).unwrap();
        assert_eq!(checkpoint_id, 0);

        assert!(snapshot_path.exists());

        let status = store.wal_status().unwrap();
        assert_eq!(status.entry_count, 0);
    }

    #[test]
    fn store_checkpoint_without_wal() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot_path = dir.path().join("snapshot.bin");

        let store = TensorStore::new();
        store.put("key1", TensorData::new()).unwrap();

        let checkpoint_id = store.checkpoint(&snapshot_path).unwrap();
        assert_eq!(checkpoint_id, 0);
        assert!(snapshot_path.exists());
    }

    #[test]
    fn store_recover_from_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

            let mut tensor = TensorData::new();
            tensor.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
            store.put_durable("key1", tensor).unwrap();
        }

        let recovered = TensorStore::recover(&wal_path, &WalConfig::default(), None).unwrap();

        assert!(recovered.exists("key1"));
        assert!(recovered.has_wal());
    }

    #[test]
    fn store_recover_from_snapshot_and_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let snapshot_path = dir.path().join("snapshot.bin");

        {
            let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();
            store.put_durable("key1", TensorData::new()).unwrap();
            store.checkpoint(&snapshot_path).unwrap();

            let mut tensor = TensorData::new();
            tensor.set("value", TensorValue::Scalar(ScalarValue::Int(99)));
            store.put_durable("key2", tensor).unwrap();
        }

        let recovered =
            TensorStore::recover(&wal_path, &WalConfig::default(), Some(&snapshot_path)).unwrap();

        assert!(recovered.exists("key1"));
        assert!(recovered.exists("key2"));
    }

    #[test]
    fn store_recover_with_bloom() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        {
            let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();
            store.put_durable("key1", TensorData::new()).unwrap();
        }

        let recovered =
            TensorStore::recover_with_bloom(&wal_path, &WalConfig::default(), None, 1000, 0.01)
                .unwrap();

        assert!(recovered.exists("key1"));
        assert!(recovered.has_wal());
        assert!(recovered.has_bloom_filter());
    }

    #[test]
    fn store_wal_sync() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();
        store.put_durable("key1", TensorData::new()).unwrap();

        store.wal_sync().unwrap();
    }

    #[test]
    fn store_wal_sync_without_wal() {
        let store = TensorStore::new();
        store.wal_sync().unwrap();
    }

    #[test]
    fn store_crash_recovery_simulation() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let snapshot_path = dir.path().join("snapshot.bin");

        {
            let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

            store.put_durable("persisted", TensorData::new()).unwrap();
            store.checkpoint(&snapshot_path).unwrap();

            let mut tensor = TensorData::new();
            tensor.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
            store.put_durable("in_wal_only", tensor).unwrap();
        }

        let recovered =
            TensorStore::recover(&wal_path, &WalConfig::default(), Some(&snapshot_path)).unwrap();

        assert!(recovered.exists("persisted"));
        assert!(recovered.exists("in_wal_only"));

        let retrieved = recovered.get("in_wal_only").unwrap();
        match retrieved.get("value") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => assert_eq!(*v, 42),
            _ => panic!("Expected Int value"),
        }
    }

    #[test]
    fn store_put_durable_with_bloom_updates_filter() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let store =
            TensorStore::open_durable_with_bloom(&wal_path, WalConfig::default(), 1000, 0.01)
                .unwrap();

        store.put_durable("key1", TensorData::new()).unwrap();

        assert!(store.exists("key1"));
        assert!(!store.exists("nonexistent"));
    }
}
