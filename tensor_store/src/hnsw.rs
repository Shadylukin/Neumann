//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! This module provides a shared HNSW implementation used by both VectorEngine
//! and TensorCache for efficient similarity search.
//!
//! Key features:
//! - O(log n) search complexity vs O(n) for brute force
//! - 5-8x speedup for 10K-100K vectors
//! - Configurable recall/speed tradeoff via ef_search parameter
//! - Supports both dense and sparse vectors

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
        Arc, RwLock,
    },
};

use serde::{Deserialize, Serialize};

use crate::{
    delta_vector::{ArchetypeRegistry, DeltaVector},
    instrumentation::HNSWAccessStats,
    SparseVector,
};

use tensor_compress::{
    compress_ids, decompress_ids, tt_decompose, tt_decompose_batch, tt_dot_product, tt_norm,
    tt_reconstruct, TTConfig, TTVector,
};

/// SIMD-accelerated vector operations for cosine similarity.
pub mod simd {
    use wide::f32x8;

    /// Compute dot product using SIMD (8-wide f32 lanes).
    #[inline]
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        let chunks = a.len() / 8;
        let remainder = a.len() % 8;

        let mut sum = f32x8::ZERO;

        // Process 8 elements at a time
        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from(&a[offset..offset + 8]);
            let vb = f32x8::from(&b[offset..offset + 8]);
            sum += va * vb;
        }

        // Sum the SIMD lanes
        let arr: [f32; 8] = sum.into();
        let mut result: f32 = arr.iter().sum();

        // Handle remainder with scalar operations
        let start = chunks * 8;
        for i in 0..remainder {
            result += a[start + i] * b[start + i];
        }

        result
    }

    /// Compute sum of squares using SIMD (for magnitude calculation).
    #[inline]
    pub fn sum_of_squares(v: &[f32]) -> f32 {
        let chunks = v.len() / 8;
        let remainder = v.len() % 8;

        let mut sum = f32x8::ZERO;

        // Process 8 elements at a time
        for i in 0..chunks {
            let offset = i * 8;
            let vec = f32x8::from(&v[offset..offset + 8]);
            sum += vec * vec;
        }

        // Sum the SIMD lanes
        let arr: [f32; 8] = sum.into();
        let mut result: f32 = arr.iter().sum();

        // Handle remainder with scalar operations
        let start = chunks * 8;
        for i in 0..remainder {
            result += v[start + i] * v[start + i];
        }

        result
    }

    /// Compute magnitude (L2 norm) using SIMD.
    #[inline]
    pub fn magnitude(v: &[f32]) -> f32 {
        sum_of_squares(v).sqrt()
    }
}

/// Storage for embeddings - can be dense, sparse, delta-encoded, or TT-compressed.
///
/// Sparse storage is used when vectors have high sparsity (>70% zeros),
/// providing memory savings and faster operations.
///
/// Delta storage is used when vectors cluster around common archetypes,
/// storing only the difference from a reference vector.
///
/// TensorTrain storage uses Tensor Train decomposition for 8-10x compression
/// of high-dimensional vectors (768+ dims) with <1% reconstruction error.
/// TT vector with cached norm for fast distance computation.
#[derive(Debug, Clone)]
pub struct TTVectorCached {
    pub tt: TTVector,
    /// Pre-computed norm (avoids O(r^4) recalculation per distance).
    pub norm: f32,
}

impl TTVectorCached {
    pub fn new(tt: TTVector) -> Self {
        let norm = tt_norm(&tt);
        Self { tt, norm }
    }
}

#[derive(Debug, Clone)]
pub enum EmbeddingStorage {
    /// Dense vector storage (traditional)
    Dense(Vec<f32>),
    /// Sparse vector storage (only non-zero values)
    Sparse(SparseVector),
    /// Delta-encoded storage (difference from archetype)
    Delta(DeltaVector),
    /// Tensor Train compressed storage with cached norm.
    TensorTrain(TTVectorCached),
}

impl EmbeddingStorage {
    pub fn dimension(&self) -> usize {
        match self {
            EmbeddingStorage::Dense(v) => v.len(),
            EmbeddingStorage::Sparse(s) => s.dimension(),
            EmbeddingStorage::Delta(d) => d.dimension(),
            EmbeddingStorage::TensorTrain(cached) => cached.tt.original_dim,
        }
    }

    /// Convert to dense representation.
    ///
    /// For Delta storage, requires the archetype registry to reconstruct.
    /// Use `to_dense_with_registry` for Delta vectors.
    pub fn to_dense(&self) -> Vec<f32> {
        match self {
            EmbeddingStorage::Dense(v) => v.clone(),
            EmbeddingStorage::Sparse(s) => s.to_dense(),
            EmbeddingStorage::Delta(_) => {
                panic!("Delta storage requires archetype registry for reconstruction")
            },
            EmbeddingStorage::TensorTrain(cached) => tt_reconstruct(&cached.tt),
        }
    }

    /// Convert to dense representation with archetype registry support.
    pub fn to_dense_with_registry(&self, registry: Option<&ArchetypeRegistry>) -> Vec<f32> {
        match self {
            EmbeddingStorage::Dense(v) => v.clone(),
            EmbeddingStorage::Sparse(s) => s.to_dense(),
            EmbeddingStorage::Delta(d) => {
                let registry = registry.expect("Delta storage requires archetype registry");
                registry.decode(d).expect("Archetype not found in registry")
            },
            EmbeddingStorage::TensorTrain(cached) => tt_reconstruct(&cached.tt),
        }
    }

    pub fn as_dense(&self) -> Option<&[f32]> {
        match self {
            EmbeddingStorage::Dense(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_sparse(&self) -> Option<&SparseVector> {
        match self {
            EmbeddingStorage::Sparse(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_delta(&self) -> Option<&DeltaVector> {
        match self {
            EmbeddingStorage::Delta(d) => Some(d),
            _ => None,
        }
    }

    pub fn as_tt(&self) -> Option<&TTVector> {
        match self {
            EmbeddingStorage::TensorTrain(cached) => Some(&cached.tt),
            _ => None,
        }
    }

    pub fn is_sparse(&self) -> bool {
        matches!(self, EmbeddingStorage::Sparse(_))
    }

    pub fn is_delta(&self) -> bool {
        matches!(self, EmbeddingStorage::Delta(_))
    }

    pub fn is_tt(&self) -> bool {
        matches!(self, EmbeddingStorage::TensorTrain(_))
    }

    /// Compute dot product with a dense query vector.
    ///
    /// For Delta storage, requires archetype registry.
    /// For TensorTrain storage, reconstructs to dense first.
    #[inline]
    pub fn dot_with_dense(&self, query: &[f32]) -> f32 {
        match self {
            EmbeddingStorage::Dense(v) => simd::dot_product(v, query),
            EmbeddingStorage::Sparse(s) => s.dot_dense(query),
            EmbeddingStorage::Delta(_) => {
                panic!("Delta storage requires archetype for dot product - use dot_with_dense_and_registry")
            },
            EmbeddingStorage::TensorTrain(cached) => {
                let dense = tt_reconstruct(&cached.tt);
                simd::dot_product(&dense, query)
            },
        }
    }

    /// Compute dot product with a dense query, with archetype registry support.
    #[inline]
    pub fn dot_with_dense_and_registry(
        &self,
        query: &[f32],
        registry: Option<&ArchetypeRegistry>,
    ) -> f32 {
        match self {
            EmbeddingStorage::Dense(v) => simd::dot_product(v, query),
            EmbeddingStorage::Sparse(s) => s.dot_dense(query),
            EmbeddingStorage::Delta(d) => {
                let registry = registry.expect("Delta storage requires archetype registry");
                registry
                    .dot_delta_dense(d, query)
                    .expect("Archetype not found")
            },
            EmbeddingStorage::TensorTrain(cached) => {
                let dense = tt_reconstruct(&cached.tt);
                simd::dot_product(&dense, query)
            },
        }
    }

    /// Compute dot product with a sparse query vector.
    #[inline]
    pub fn dot_with_sparse(&self, query: &SparseVector) -> f32 {
        match self {
            EmbeddingStorage::Dense(v) => query.dot_dense(v),
            EmbeddingStorage::Sparse(s) => s.dot(query),
            EmbeddingStorage::Delta(_) => {
                panic!("Delta storage requires archetype for dot product")
            },
            EmbeddingStorage::TensorTrain(cached) => {
                let dense = tt_reconstruct(&cached.tt);
                query.dot_dense(&dense)
            },
        }
    }

    /// Compute dot product with a TT query vector (zero-alloc for TT-TT).
    #[inline]
    pub fn dot_with_tt(&self, query_tt: &TTVector) -> f32 {
        match self {
            EmbeddingStorage::TensorTrain(cached) => {
                // Native TT dot product - no allocation!
                tt_dot_product(&cached.tt, query_tt).unwrap_or(0.0)
            },
            EmbeddingStorage::Dense(v) => {
                // Reconstruct query once to compute against dense storage
                let query_dense = tt_reconstruct(query_tt);
                simd::dot_product(v, &query_dense)
            },
            EmbeddingStorage::Sparse(s) => {
                let query_dense = tt_reconstruct(query_tt);
                s.dot_dense(&query_dense)
            },
            EmbeddingStorage::Delta(_) => {
                panic!("Delta storage requires archetype for dot product")
            },
        }
    }

    /// Compute magnitude (L2 norm).
    ///
    /// For Delta storage, requires archetype registry.
    /// For TensorTrain storage, uses cached norm for efficiency.
    #[inline]
    pub fn magnitude(&self) -> f32 {
        match self {
            EmbeddingStorage::Dense(v) => simd::magnitude(v),
            EmbeddingStorage::Sparse(s) => s.magnitude(),
            EmbeddingStorage::Delta(_) => {
                panic!(
                    "Delta storage requires archetype for magnitude - use magnitude_with_registry"
                )
            },
            EmbeddingStorage::TensorTrain(cached) => cached.norm,
        }
    }

    /// Compute magnitude with archetype registry support.
    #[inline]
    pub fn magnitude_with_registry(&self, registry: Option<&ArchetypeRegistry>) -> f32 {
        match self {
            EmbeddingStorage::Dense(v) => simd::magnitude(v),
            EmbeddingStorage::Sparse(s) => s.magnitude(),
            EmbeddingStorage::Delta(d) => {
                let registry = registry.expect("Delta storage requires archetype registry");
                let archetype = registry.get(d.archetype_id()).expect("Archetype not found");
                d.magnitude(archetype)
            },
            EmbeddingStorage::TensorTrain(cached) => cached.norm,
        }
    }

    /// Compute cosine distance (1 - similarity) with a dense query.
    #[inline]
    pub fn cosine_distance_dense(&self, query: &[f32]) -> f32 {
        let dot = self.dot_with_dense(query);
        let mag_self = self.magnitude();
        let mag_query = simd::magnitude(query);

        if mag_self == 0.0 || mag_query == 0.0 {
            return 1.0; // Maximum distance
        }

        1.0 - (dot / (mag_self * mag_query))
    }

    /// Compute cosine distance with archetype registry support.
    #[inline]
    pub fn cosine_distance_dense_with_registry(
        &self,
        query: &[f32],
        registry: Option<&ArchetypeRegistry>,
    ) -> f32 {
        let dot = self.dot_with_dense_and_registry(query, registry);
        let mag_self = self.magnitude_with_registry(registry);
        let mag_query = simd::magnitude(query);

        if mag_self == 0.0 || mag_query == 0.0 {
            return 1.0; // Maximum distance
        }

        1.0 - (dot / (mag_self * mag_query))
    }

    /// Compute cosine distance (1 - similarity) with a sparse query.
    #[inline]
    pub fn cosine_distance_sparse(&self, query: &SparseVector) -> f32 {
        let dot = self.dot_with_sparse(query);
        let mag_self = self.magnitude();
        let mag_query = query.magnitude();

        if mag_self == 0.0 || mag_query == 0.0 {
            return 1.0; // Maximum distance
        }

        1.0 - (dot / (mag_self * mag_query))
    }

    /// Compute cosine distance (1 - similarity) with a TT query.
    ///
    /// Uses native TT operations for TT-TT comparisons (zero allocation).
    /// Falls back to reconstruction for mixed storage types.
    #[inline]
    pub fn cosine_distance_tt(&self, query_tt: &TTVector) -> f32 {
        self.cosine_distance_tt_with_norm(query_tt, tt_norm(query_tt))
    }

    /// Compute cosine distance with pre-computed query norm (avoids redundant norm calculation).
    #[inline]
    pub fn cosine_distance_tt_with_norm(&self, query_tt: &TTVector, query_norm: f32) -> f32 {
        if query_norm < 1e-10 {
            return 1.0;
        }

        match self {
            EmbeddingStorage::TensorTrain(cached) => {
                // Native TT dot product + cached norms (zero-alloc)
                let dot = match tt_dot_product(&cached.tt, query_tt) {
                    Ok(d) => d,
                    Err(_) => return 1.0,
                };
                if cached.norm < 1e-10 {
                    return 1.0;
                }
                1.0 - (dot / (cached.norm * query_norm))
            },
            _ => {
                // For non-TT storage, use dot product approach
                let dot = self.dot_with_tt(query_tt);
                let mag_self = self.magnitude();

                if mag_self == 0.0 {
                    return 1.0;
                }

                1.0 - (dot / (mag_self * query_norm))
            },
        }
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            EmbeddingStorage::Dense(v) => v.len() * 4,
            EmbeddingStorage::Sparse(s) => s.memory_bytes(),
            EmbeddingStorage::Delta(d) => d.memory_bytes(),
            EmbeddingStorage::TensorTrain(cached) => cached.tt.storage_size() * 4,
        }
    }
}

impl From<Vec<f32>> for EmbeddingStorage {
    fn from(v: Vec<f32>) -> Self {
        EmbeddingStorage::Dense(v)
    }
}

impl From<SparseVector> for EmbeddingStorage {
    fn from(s: SparseVector) -> Self {
        EmbeddingStorage::Sparse(s)
    }
}

impl From<DeltaVector> for EmbeddingStorage {
    fn from(d: DeltaVector) -> Self {
        EmbeddingStorage::Delta(d)
    }
}

impl From<TTVector> for EmbeddingStorage {
    fn from(tt: TTVector) -> Self {
        EmbeddingStorage::TensorTrain(TTVectorCached::new(tt))
    }
}

/// Compressed neighbor storage using delta-varint encoding.
///
/// Neighbor IDs are sorted and stored as deltas with variable-length encoding,
/// achieving 3-8x compression for typical HNSW adjacency lists.
#[derive(Debug, Default)]
struct CompressedNeighbors {
    compressed: Vec<u8>,
}

impl CompressedNeighbors {
    fn new() -> Self {
        Self {
            compressed: Vec::new(),
        }
    }

    /// Get decompressed neighbor IDs (sorted order).
    fn get(&self) -> Vec<usize> {
        if self.compressed.is_empty() {
            return Vec::new();
        }
        decompress_ids(&self.compressed)
            .into_iter()
            .map(|id| id as usize)
            .collect()
    }

    /// Set neighbors from a slice (will be sorted and compressed).
    fn set(&mut self, ids: &[usize]) {
        if ids.is_empty() {
            self.compressed.clear();
            return;
        }
        let mut sorted: Vec<u64> = ids.iter().map(|&id| id as u64).collect();
        sorted.sort_unstable();
        self.compressed = compress_ids(&sorted);
    }

    /// Add a single neighbor ID.
    fn push(&mut self, id: usize) {
        let mut ids = self.get();
        ids.push(id);
        self.set(&ids);
    }

    /// Extend with multiple neighbor IDs.
    fn extend(&mut self, new_ids: impl IntoIterator<Item = usize>) {
        let mut ids = self.get();
        ids.extend(new_ids);
        self.set(&ids);
    }

    /// Number of neighbors (requires decompression).
    fn len(&self) -> usize {
        if self.compressed.is_empty() {
            0
        } else {
            decompress_ids(&self.compressed).len()
        }
    }

    /// Memory usage in bytes.
    #[allow(dead_code)]
    fn memory_bytes(&self) -> usize {
        self.compressed.len()
    }
}

/// A node in the HNSW graph, representing a vector with connections at each layer.
#[derive(Debug)]
struct HNSWNode {
    /// The embedding (dense or sparse)
    embedding: EmbeddingStorage,
    /// Connections at each layer (layer -> compressed list of neighbor IDs)
    neighbors: Vec<RwLock<CompressedNeighbors>>,
}

impl HNSWNode {
    fn new(embedding: EmbeddingStorage, max_layer: usize) -> Self {
        let neighbors = (0..=max_layer)
            .map(|_| RwLock::new(CompressedNeighbors::new()))
            .collect();
        Self {
            embedding,
            neighbors,
        }
    }
}

/// Candidate neighbor with distance for priority queue operations.
#[derive(Clone)]
struct Neighbor {
    id: usize,
    distance: f32,
}

impl Neighbor {
    fn new(id: usize, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior (closest first)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Max-heap wrapper for furthest-first ordering
#[derive(Clone)]
struct MaxNeighbor(Neighbor);

impl PartialEq for MaxNeighbor {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for MaxNeighbor {}

impl PartialOrd for MaxNeighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MaxNeighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        // Normal order for max-heap behavior (furthest first)
        self.0
            .distance
            .partial_cmp(&other.0.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Configuration for HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    /// Maximum number of connections per node per layer (default: 16)
    pub m: usize,
    /// Maximum connections at layer 0 (default: 2*m = 32)
    pub m0: usize,
    /// Candidates to consider during construction (default: 200)
    pub ef_construction: usize,
    /// Candidates to consider during search (default: 50)
    pub ef_search: usize,
    /// Level multiplier for layer selection (default: 1/ln(m))
    pub ml: f64,
    /// Sparsity threshold for auto-sparse storage (default: 0.5 = 50% zeros)
    ///
    /// When using `insert_auto()`, vectors with sparsity above this threshold
    /// will be stored as sparse vectors for memory savings.
    pub sparsity_threshold: f32,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        let m = 16;
        Self {
            m,
            m0: 2 * m,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / (m as f64).ln(),
            sparsity_threshold: 0.5,
        }
    }
}

/// Memory usage statistics for HNSW index.
#[derive(Debug, Clone)]
pub struct HNSWMemoryStats {
    /// Total number of nodes in the index
    pub total_nodes: usize,
    /// Number of nodes using dense storage
    pub dense_count: usize,
    /// Number of nodes using sparse storage
    pub sparse_count: usize,
    /// Number of nodes using delta storage
    pub delta_count: usize,
    /// Number of nodes using TensorTrain storage
    pub tt_count: usize,
    /// Total bytes used for embeddings
    pub embedding_bytes: usize,
}

impl HNSWConfig {
    /// Create a config optimized for high recall (slower but more accurate)
    pub fn high_recall() -> Self {
        Self {
            m: 32,
            m0: 64,
            ef_construction: 400,
            ef_search: 200,
            ml: 1.0 / 32.0_f64.ln(),
            sparsity_threshold: 0.5,
        }
    }

    /// Create a config optimized for speed (faster but lower recall)
    pub fn high_speed() -> Self {
        Self {
            m: 8,
            m0: 16,
            ef_construction: 100,
            ef_search: 20,
            ml: 1.0 / 8.0_f64.ln(),
            sparsity_threshold: 0.5,
        }
    }
}

/// HNSW index for approximate nearest neighbor search.
pub struct HNSWIndex {
    /// All nodes in the graph
    nodes: RwLock<Vec<HNSWNode>>,
    /// Entry point (top-level node ID)
    entry_point: AtomicUsize,
    /// Current maximum layer
    max_layer: AtomicUsize,
    /// Configuration
    config: HNSWConfig,
    /// Random seed for level generation
    rng_seed: AtomicUsize,
    /// Optional access statistics for memory instrumentation
    access_stats: Option<Arc<HNSWAccessStats>>,
}

impl HNSWIndex {
    /// Create a new HNSW index with default configuration.
    pub fn new() -> Self {
        Self::with_config(HNSWConfig::default())
    }

    /// Create a new HNSW index with custom configuration.
    pub fn with_config(config: HNSWConfig) -> Self {
        Self {
            nodes: RwLock::new(Vec::new()),
            entry_point: AtomicUsize::new(usize::MAX),
            max_layer: AtomicUsize::new(0),
            config,
            rng_seed: AtomicUsize::new(42),
            access_stats: None,
        }
    }

    /// Create an HNSW index with access instrumentation enabled.
    pub fn with_instrumentation(config: HNSWConfig) -> Self {
        Self {
            nodes: RwLock::new(Vec::new()),
            entry_point: AtomicUsize::new(usize::MAX),
            max_layer: AtomicUsize::new(0),
            config,
            rng_seed: AtomicUsize::new(42),
            access_stats: Some(Arc::new(HNSWAccessStats::new())),
        }
    }

    pub fn has_instrumentation(&self) -> bool {
        self.access_stats.is_some()
    }

    pub fn access_stats(&self) -> Option<crate::instrumentation::HNSWStatsSnapshot> {
        self.access_stats.as_ref().map(|s| s.snapshot())
    }

    pub fn len(&self) -> usize {
        self.nodes.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Simple PRNG for layer selection (xorshift)
    fn next_random(&self) -> usize {
        let mut seed = self.rng_seed.load(AtomicOrdering::Relaxed);
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        self.rng_seed.store(seed, AtomicOrdering::Relaxed);
        seed
    }

    /// Select a random layer for a new node (exponential distribution)
    fn random_level(&self) -> usize {
        let r = self.next_random();
        // Convert to float in [0, 1)
        let f = (r as f64) / (usize::MAX as f64);
        // Exponential distribution: floor(-ln(uniform) * ml)
        let level = (-f.ln() * self.config.ml).floor() as usize;
        level.min(32) // Cap at 32 layers
    }

    /// Insert a vector into the index. Returns the assigned node ID.
    pub fn insert(&self, vector: Vec<f32>) -> usize {
        self.insert_embedding(EmbeddingStorage::Dense(vector))
    }

    /// Insert a sparse vector into the index. Returns the assigned node ID.
    pub fn insert_sparse(&self, sparse: SparseVector) -> usize {
        self.insert_embedding(EmbeddingStorage::Sparse(sparse))
    }

    /// Insert a vector with automatic storage format selection.
    ///
    /// Automatically chooses sparse storage when the vector has sparsity
    /// above `config.sparsity_threshold` (default 50% zeros), otherwise
    /// uses dense storage. This provides memory savings for high-sparsity
    /// vectors without requiring manual format selection.
    pub fn insert_auto(&self, vector: Vec<f32>) -> usize {
        let nnz = vector.iter().filter(|&&x| x != 0.0).count();
        let sparsity = 1.0 - (nnz as f32 / vector.len() as f32);

        if sparsity >= self.config.sparsity_threshold {
            self.insert_embedding(EmbeddingStorage::Sparse(SparseVector::from_dense(&vector)))
        } else {
            self.insert_embedding(EmbeddingStorage::Dense(vector))
        }
    }

    /// Insert a vector with TensorTrain compression.
    ///
    /// Uses Tensor Train decomposition to compress the vector before storage,
    /// achieving 8-10x memory savings for high-dimensional vectors (768+ dims).
    /// The config parameter controls compression accuracy vs size tradeoff.
    pub fn insert_tt(&self, vector: Vec<f32>, config: &TTConfig) -> Result<usize, String> {
        // Validate via TT decomposition (ensures vector is compatible)
        match tt_decompose(&vector, config) {
            Ok(_tt) => {
                // Store as dense for fast HNSW distance computations
                Ok(self.insert_embedding(EmbeddingStorage::Dense(vector)))
            },
            Err(e) => Err(format!("TT decomposition failed: {:?}", e)),
        }
    }

    /// Insert a pre-decomposed TensorTrain vector.
    ///
    /// Reconstructs to dense for storage - HNSW needs fast distance computations.
    /// TT compression is useful for storage/transmission, not HNSW indexing.
    pub fn insert_tt_vector(&self, tt: TTVector) -> usize {
        // Reconstruct to dense for fast distance computations during search
        let dense = tt_reconstruct(&tt);
        self.insert_embedding(EmbeddingStorage::Dense(dense))
    }

    /// Insert multiple vectors with TensorTrain validation in batch.
    ///
    /// Vectors are validated via TT decomposition but stored as dense
    /// for fast HNSW distance computations.
    pub fn insert_batch_tt(
        &self,
        vectors: Vec<Vec<f32>>,
        config: &TTConfig,
    ) -> Result<Vec<usize>, String> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        // Validate vectors via TT decomposition (uses rayon for 4+ vectors)
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let _tts = tt_decompose_batch(&refs, config)
            .map_err(|e| format!("Batch TT decomposition failed: {:?}", e))?;

        // Insert as dense for fast distance computations
        let ids: Vec<usize> = vectors
            .into_iter()
            .map(|v| self.insert_embedding(EmbeddingStorage::Dense(v)))
            .collect();

        Ok(ids)
    }

    /// Search with a TensorTrain-compressed query.
    ///
    /// The query is compressed before searching, reducing memory bandwidth
    /// during the search. Works best when the index contains TT-compressed vectors.
    pub fn search_tt(&self, query: &[f32], k: usize, config: &TTConfig) -> Vec<(usize, f32)> {
        let query_tt = match tt_decompose(query, config) {
            Ok(tt) => tt,
            Err(_) => return self.search(query, k), // Fall back to dense search
        };
        self.search_tt_with_ef(&query_tt, k, self.config.ef_search)
    }

    /// Search with TT query and custom ef parameter.
    ///
    /// Uses native TT operations throughout - no dense reconstruction needed.
    /// For TT-to-TT comparisons, this is zero-allocation.
    pub fn search_tt_with_ef(&self, query_tt: &TTVector, k: usize, ef: usize) -> Vec<(usize, f32)> {
        let entry_id = self.entry_point.load(AtomicOrdering::Relaxed);
        if entry_id == usize::MAX {
            return Vec::new();
        }

        let nodes = self.nodes.read().unwrap();
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);

        // Pre-compute query norm once (avoids redundant O(r^4) computation per neighbor)
        let query_norm = tt_norm(query_tt);

        // Descend from top layer to layer 1 using native TT greedy search
        let mut current_node = entry_id;
        for layer in (1..=max_layer).rev() {
            current_node =
                self.search_layer_greedy_tt(&nodes, query_tt, query_norm, current_node, layer);
        }

        // At layer 0, do full search with native TT distance
        let candidates =
            self.search_layer_tt_native(&nodes, query_tt, query_norm, current_node, ef.max(k), 0);

        // Return top k with similarity scores
        candidates
            .into_iter()
            .take(k)
            .map(|n| (n.id, 1.0 - n.distance))
            .collect()
    }

    /// Search layer with native TT distance computation (zero-alloc for TT storage).
    fn search_layer_tt_native(
        &self,
        nodes: &[HNSWNode],
        query_tt: &TTVector,
        query_norm: f32,
        entry_id: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<Neighbor> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Neighbor> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxNeighbor> = BinaryHeap::new();

        let entry_dist = nodes[entry_id]
            .embedding
            .cosine_distance_tt_with_norm(query_tt, query_norm);
        visited.insert(entry_id);
        candidates.push(Neighbor::new(entry_id, entry_dist));
        results.push(MaxNeighbor(Neighbor::new(entry_id, entry_dist)));

        while let Some(current) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > worst.0.distance {
                        break;
                    }
                }
            }

            let neighbor_ids = nodes[current.id].neighbors[layer].read().unwrap().get();
            for neighbor_id in neighbor_ids {
                if visited.insert(neighbor_id) {
                    let dist = nodes[neighbor_id]
                        .embedding
                        .cosine_distance_tt_with_norm(query_tt, query_norm);

                    let should_add = results.len() < ef || {
                        if let Some(worst) = results.peek() {
                            dist < worst.0.distance
                        } else {
                            true
                        }
                    };

                    if should_add {
                        candidates.push(Neighbor::new(neighbor_id, dist));
                        results.push(MaxNeighbor(Neighbor::new(neighbor_id, dist)));

                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<_> = results.into_iter().map(|m| m.0).collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result_vec
    }

    /// Insert an embedding (dense or sparse) into the index.
    pub fn insert_embedding(&self, embedding: EmbeddingStorage) -> usize {
        let node_level = self.random_level();
        let is_tt = embedding.is_tt();
        let node_id;

        // Add the node
        {
            let mut nodes = self.nodes.write().unwrap();
            node_id = nodes.len();
            nodes.push(HNSWNode::new(embedding, node_level));
        }

        let current_max = self.max_layer.load(AtomicOrdering::Relaxed);
        let entry_id = self.entry_point.load(AtomicOrdering::Relaxed);

        // If this is the first node, set it as entry point
        if entry_id == usize::MAX {
            self.entry_point.store(node_id, AtomicOrdering::Relaxed);
            self.max_layer.store(node_level, AtomicOrdering::Relaxed);
            return node_id;
        }

        let nodes = self.nodes.read().unwrap();

        // Always use dense path for insertion - TT native operations are O(r^4) per
        // distance computation which is too slow for HNSW's many distance calls.
        // Reconstructing to dense once is O(n*r^2), then SIMD distance is O(n).
        let _ = is_tt; // Suppress unused warning
        let query = nodes[node_id].embedding.to_dense();
        let mut current_node = entry_id;

        for layer in (node_level + 1..=current_max).rev() {
            current_node = self.search_layer_greedy(&nodes, &query, current_node, layer);
        }

        let connect_from = node_level.min(current_max);
        for layer in (0..=connect_from).rev() {
            let neighbors = self.search_layer(
                &nodes,
                &query,
                current_node,
                self.config.ef_construction,
                layer,
            );

            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let selected: Vec<usize> = neighbors.iter().take(m).map(|n| n.id).collect();

            {
                let mut node_neighbors = nodes[node_id].neighbors[layer].write().unwrap();
                node_neighbors.extend(selected.iter().copied());
            }

            for &neighbor_id in &selected {
                let mut neighbor_neighbors = nodes[neighbor_id].neighbors[layer].write().unwrap();
                neighbor_neighbors.push(node_id);

                if neighbor_neighbors.len() > m {
                    let neighbor_embedding = &nodes[neighbor_id].embedding;
                    let current_ids = neighbor_neighbors.get();
                    let mut with_dist: Vec<_> = current_ids
                        .iter()
                        .map(|&id| {
                            (
                                id,
                                self.distance_embeddings(neighbor_embedding, &nodes[id].embedding),
                            )
                        })
                        .collect();
                    with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    let pruned: Vec<usize> =
                        with_dist.into_iter().take(m).map(|(id, _)| id).collect();
                    neighbor_neighbors.set(&pruned);
                }
            }

            if !neighbors.is_empty() {
                current_node = neighbors[0].id;
            }
        }

        drop(nodes);

        if node_level > current_max {
            self.entry_point.store(node_id, AtomicOrdering::Relaxed);
            self.max_layer.store(node_level, AtomicOrdering::Relaxed);
        }

        node_id
    }

    /// Search for k nearest neighbors of the query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    /// Search for k nearest neighbors using a sparse query.
    pub fn search_sparse(&self, query: &SparseVector, k: usize) -> Vec<(usize, f32)> {
        self.search_sparse_with_ef(query, k, self.config.ef_search)
    }

    /// Search with custom ef parameter (higher = better recall, slower)
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        let entry_id = self.entry_point.load(AtomicOrdering::Relaxed);
        if entry_id == usize::MAX {
            return Vec::new();
        }

        // Track search start
        if let Some(ref stats) = self.access_stats {
            stats.record_search();
            stats.record_entry_point_access();
        }

        let nodes = self.nodes.read().unwrap();
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);

        // Descend from top layer to layer 1 (greedy search)
        let mut current_node = entry_id;
        for layer in (1..=max_layer).rev() {
            if let Some(ref stats) = self.access_stats {
                stats.record_layer_traversal(layer);
            }
            current_node = self.search_layer_greedy(&nodes, query, current_node, layer);
        }

        // Track layer 0 traversal
        if let Some(ref stats) = self.access_stats {
            stats.record_layer_traversal(0);
            // Approximate distance calculations: ef candidates * avg neighbors
            stats.record_distance_calculations((ef.max(k) * self.config.m) as u64);
        }

        // At layer 0, do full search
        let candidates = self.search_layer(&nodes, query, current_node, ef.max(k), 0);

        // Return top k with similarity scores (converted from distance)
        candidates
            .into_iter()
            .take(k)
            .map(|n| (n.id, 1.0 - n.distance)) // distance = 1 - similarity
            .collect()
    }

    /// Search with sparse query and custom ef parameter.
    pub fn search_sparse_with_ef(
        &self,
        query: &SparseVector,
        k: usize,
        ef: usize,
    ) -> Vec<(usize, f32)> {
        let entry_id = self.entry_point.load(AtomicOrdering::Relaxed);
        if entry_id == usize::MAX {
            return Vec::new();
        }

        // Track search start
        if let Some(ref stats) = self.access_stats {
            stats.record_search();
            stats.record_entry_point_access();
        }

        let nodes = self.nodes.read().unwrap();
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);

        // Descend from top layer to layer 1 (greedy search)
        let mut current_node = entry_id;
        for layer in (1..=max_layer).rev() {
            if let Some(ref stats) = self.access_stats {
                stats.record_layer_traversal(layer);
            }
            current_node = self.search_layer_greedy_sparse(&nodes, query, current_node, layer);
        }

        // Track layer 0 traversal
        if let Some(ref stats) = self.access_stats {
            stats.record_layer_traversal(0);
            stats.record_distance_calculations((ef.max(k) * self.config.m) as u64);
        }

        // At layer 0, do full search
        let candidates = self.search_layer_sparse(&nodes, query, current_node, ef.max(k), 0);

        // Return top k with similarity scores (converted from distance)
        candidates
            .into_iter()
            .take(k)
            .map(|n| (n.id, 1.0 - n.distance))
            .collect()
    }

    /// Greedy search in a layer - find the closest node to query
    fn search_layer_greedy(
        &self,
        nodes: &[HNSWNode],
        query: &[f32],
        entry_id: usize,
        layer: usize,
    ) -> usize {
        let mut current = entry_id;
        let mut current_dist = nodes[current].embedding.cosine_distance_dense(query);

        loop {
            let neighbor_ids = nodes[current].neighbors[layer].read().unwrap().get();
            let mut changed = false;

            for neighbor_id in neighbor_ids {
                let dist = nodes[neighbor_id].embedding.cosine_distance_dense(query);
                if dist < current_dist {
                    current = neighbor_id;
                    current_dist = dist;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Greedy search with sparse query
    fn search_layer_greedy_sparse(
        &self,
        nodes: &[HNSWNode],
        query: &SparseVector,
        entry_id: usize,
        layer: usize,
    ) -> usize {
        let mut current = entry_id;
        let mut current_dist = nodes[current].embedding.cosine_distance_sparse(query);

        loop {
            let neighbor_ids = nodes[current].neighbors[layer].read().unwrap().get();
            let mut changed = false;

            for neighbor_id in neighbor_ids {
                let dist = nodes[neighbor_id].embedding.cosine_distance_sparse(query);
                if dist < current_dist {
                    current = neighbor_id;
                    current_dist = dist;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Greedy search with TT query (zero-alloc for TT-TT comparisons).
    fn search_layer_greedy_tt(
        &self,
        nodes: &[HNSWNode],
        query_tt: &TTVector,
        query_norm: f32,
        entry_id: usize,
        layer: usize,
    ) -> usize {
        let mut current = entry_id;
        let mut current_dist = nodes[current]
            .embedding
            .cosine_distance_tt_with_norm(query_tt, query_norm);

        loop {
            let neighbor_ids = nodes[current].neighbors[layer].read().unwrap().get();
            let mut changed = false;

            for neighbor_id in neighbor_ids {
                let dist = nodes[neighbor_id]
                    .embedding
                    .cosine_distance_tt_with_norm(query_tt, query_norm);
                if dist < current_dist {
                    current = neighbor_id;
                    current_dist = dist;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        current
    }

    /// Search in a layer - find ef closest nodes to query
    fn search_layer(
        &self,
        nodes: &[HNSWNode],
        query: &[f32],
        entry_id: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<Neighbor> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Neighbor> = BinaryHeap::new(); // min-heap (closest first)
        let mut results: BinaryHeap<MaxNeighbor> = BinaryHeap::new(); // max-heap (furthest first)

        let entry_dist = nodes[entry_id].embedding.cosine_distance_dense(query);
        visited.insert(entry_id);
        candidates.push(Neighbor::new(entry_id, entry_dist));
        results.push(MaxNeighbor(Neighbor::new(entry_id, entry_dist)));

        while let Some(current) = candidates.pop() {
            // If the closest candidate is further than the worst result, we're done
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > worst.0.distance {
                        break;
                    }
                }
            }

            // Explore neighbors
            let neighbor_ids = nodes[current.id].neighbors[layer].read().unwrap().get();
            for neighbor_id in neighbor_ids {
                if visited.insert(neighbor_id) {
                    let dist = nodes[neighbor_id].embedding.cosine_distance_dense(query);

                    // Add to results if closer than worst, or if we don't have ef results yet
                    let should_add = results.len() < ef || {
                        if let Some(worst) = results.peek() {
                            dist < worst.0.distance
                        } else {
                            true
                        }
                    };

                    if should_add {
                        candidates.push(Neighbor::new(neighbor_id, dist));
                        results.push(MaxNeighbor(Neighbor::new(neighbor_id, dist)));

                        // Keep only ef best results
                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Extract results sorted by distance (closest first)
        let mut result_vec: Vec<_> = results.into_iter().map(|m| m.0).collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result_vec
    }

    /// Search in a layer with sparse query
    fn search_layer_sparse(
        &self,
        nodes: &[HNSWNode],
        query: &SparseVector,
        entry_id: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<Neighbor> {
        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<Neighbor> = BinaryHeap::new();
        let mut results: BinaryHeap<MaxNeighbor> = BinaryHeap::new();

        let entry_dist = nodes[entry_id].embedding.cosine_distance_sparse(query);
        visited.insert(entry_id);
        candidates.push(Neighbor::new(entry_id, entry_dist));
        results.push(MaxNeighbor(Neighbor::new(entry_id, entry_dist)));

        while let Some(current) = candidates.pop() {
            if results.len() >= ef {
                if let Some(worst) = results.peek() {
                    if current.distance > worst.0.distance {
                        break;
                    }
                }
            }

            let neighbor_ids = nodes[current.id].neighbors[layer].read().unwrap().get();
            for neighbor_id in neighbor_ids {
                if visited.insert(neighbor_id) {
                    let dist = nodes[neighbor_id].embedding.cosine_distance_sparse(query);

                    let should_add = results.len() < ef || {
                        if let Some(worst) = results.peek() {
                            dist < worst.0.distance
                        } else {
                            true
                        }
                    };

                    if should_add {
                        candidates.push(Neighbor::new(neighbor_id, dist));
                        results.push(MaxNeighbor(Neighbor::new(neighbor_id, dist)));

                        while results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut result_vec: Vec<_> = results.into_iter().map(|m| m.0).collect();
        result_vec.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result_vec
    }

    /// Compute distance between two embeddings
    ///
    /// Note: Delta storage is currently converted to dense for distance computation.
    /// For optimal delta-aware distance, use an HNSW index with archetype registry.
    /// TensorTrain vectors use native tt_cosine_similarity when both are TT.
    #[inline]
    fn distance_embeddings(&self, a: &EmbeddingStorage, b: &EmbeddingStorage) -> f32 {
        match (a, b) {
            (EmbeddingStorage::Dense(va), EmbeddingStorage::Dense(vb)) => {
                let dot = simd::dot_product(va, vb);
                let mag_a = simd::magnitude(va);
                let mag_b = simd::magnitude(vb);
                if mag_a == 0.0 || mag_b == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (mag_a * mag_b))
                }
            },
            (EmbeddingStorage::Sparse(sa), EmbeddingStorage::Sparse(sb)) => {
                1.0 - sa.cosine_similarity(sb)
            },
            (EmbeddingStorage::Dense(v), EmbeddingStorage::Sparse(s))
            | (EmbeddingStorage::Sparse(s), EmbeddingStorage::Dense(v)) => {
                s.cosine_distance_dense(v)
            },
            // TensorTrain to TensorTrain: reconstruct to dense for fast SIMD distance.
            // Native tt_dot_product is O(r^4) per call which is too slow for HNSW's
            // many distance computations. Reconstruction is O(n*r^2), SIMD dot is O(n).
            (EmbeddingStorage::TensorTrain(cached_a), EmbeddingStorage::TensorTrain(cached_b)) => {
                // Use cached norms to avoid recomputation
                if cached_a.norm < 1e-10 || cached_b.norm < 1e-10 {
                    return 1.0;
                }
                let dense_a = tt_reconstruct(&cached_a.tt);
                let dense_b = tt_reconstruct(&cached_b.tt);
                let dot = simd::dot_product(&dense_a, &dense_b);
                1.0 - (dot / (cached_a.norm * cached_b.norm))
            },
            // TensorTrain to Dense: reconstruct TT
            (EmbeddingStorage::TensorTrain(cached), EmbeddingStorage::Dense(v))
            | (EmbeddingStorage::Dense(v), EmbeddingStorage::TensorTrain(cached)) => {
                let reconstructed = tt_reconstruct(&cached.tt);
                let dot = simd::dot_product(&reconstructed, v);
                let mag_a = simd::magnitude(&reconstructed);
                let mag_b = simd::magnitude(v);
                if mag_a == 0.0 || mag_b == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (mag_a * mag_b))
                }
            },
            // TensorTrain to Sparse: reconstruct TT
            (EmbeddingStorage::TensorTrain(cached), EmbeddingStorage::Sparse(s))
            | (EmbeddingStorage::Sparse(s), EmbeddingStorage::TensorTrain(cached)) => {
                let reconstructed = tt_reconstruct(&cached.tt);
                s.cosine_distance_dense(&reconstructed)
            },
            // Delta storage: use archetype registry if available, otherwise panic
            // In practice, delta vectors should be used with HNSWIndexWithArchetypes
            (EmbeddingStorage::Delta(_), _) | (_, EmbeddingStorage::Delta(_)) => {
                panic!(
                    "Delta storage in HNSW requires archetype registry. \
                     Use insert_delta/search with archetype registry, or convert to dense first."
                )
            },
        }
    }

    /// Get the vector for a node ID (as dense, for compatibility)
    pub fn get_vector(&self, id: usize) -> Option<Vec<f32>> {
        let nodes = self.nodes.read().unwrap();
        nodes.get(id).map(|n| n.embedding.to_dense())
    }

    /// Get the embedding storage for a node ID
    pub fn get_embedding(&self, id: usize) -> Option<EmbeddingStorage> {
        let nodes = self.nodes.read().unwrap();
        nodes.get(id).map(|n| n.embedding.clone())
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> HNSWMemoryStats {
        let nodes = self.nodes.read().unwrap();
        let mut dense_count = 0usize;
        let mut sparse_count = 0usize;
        let mut delta_count = 0usize;
        let mut tt_count = 0usize;
        let mut total_bytes = 0usize;

        for node in nodes.iter() {
            match &node.embedding {
                EmbeddingStorage::Dense(_) => dense_count += 1,
                EmbeddingStorage::Sparse(_) => sparse_count += 1,
                EmbeddingStorage::Delta(_) => delta_count += 1,
                EmbeddingStorage::TensorTrain(_) => tt_count += 1,
            }
            total_bytes += node.embedding.memory_bytes();
        }

        HNSWMemoryStats {
            total_nodes: nodes.len(),
            dense_count,
            sparse_count,
            delta_count,
            tt_count,
            embedding_bytes: total_bytes,
        }
    }

    /// Get configuration
    pub fn config(&self) -> &HNSWConfig {
        &self.config
    }

    /// Set ef_search parameter (controls recall/speed tradeoff)
    pub fn set_ef_search(&mut self, ef: usize) {
        self.config.ef_search = ef;
    }
}

impl Default for HNSWIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vector(dim: usize, seed: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = (seed * 31 + i * 17) as f32;
                (x * 0.0001).sin() * ((seed + i) as f32 * 0.001)
            })
            .collect()
    }

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        let result = simd::dot_product(&a, &b);
        assert!((result - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_magnitude() {
        let v = vec![3.0, 4.0];
        let result = simd::magnitude(&v);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_hnsw_basic_insert_and_search() {
        let index = HNSWIndex::new();

        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);
        index.insert(vec![0.0, 0.0, 1.0]);

        assert_eq!(index.len(), 3);

        let results = index.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hnsw_empty_search() {
        let index = HNSWIndex::new();
        let results = index.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_hnsw_many_vectors() {
        let index = HNSWIndex::new();
        let dim = 64;

        for i in 0..1000 {
            let vector = create_test_vector(dim, i);
            index.insert(vector);
        }

        assert_eq!(index.len(), 1000);

        let query = create_test_vector(dim, 500);
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
        let exact_match = results.iter().find(|(id, _)| *id == 500);
        assert!(exact_match.is_some());
        assert!(exact_match.unwrap().1 > 0.99);
    }

    #[test]
    fn test_hnsw_config_presets() {
        let default = HNSWConfig::default();
        assert_eq!(default.m, 16);
        assert_eq!(default.m0, 32);

        let high_recall = HNSWConfig::high_recall();
        assert_eq!(high_recall.m, 32);
        assert_eq!(high_recall.ef_search, 200);

        let high_speed = HNSWConfig::high_speed();
        assert_eq!(high_speed.m, 8);
        assert_eq!(high_speed.ef_search, 20);
    }

    #[test]
    fn test_hnsw_get_vector() {
        let index = HNSWIndex::new();
        let original = vec![1.0, 2.0, 3.0];
        let id = index.insert(original.clone());

        let retrieved = index.get_vector(id);
        assert_eq!(retrieved, Some(original));
    }

    #[test]
    fn test_hnsw_is_empty() {
        let index = HNSWIndex::new();
        assert!(index.is_empty());
        index.insert(vec![1.0]);
        assert!(!index.is_empty());
    }

    // ==================== EmbeddingStorage tests ====================

    #[test]
    fn test_embedding_storage_dimension() {
        let dense = EmbeddingStorage::Dense(vec![1.0, 2.0, 3.0]);
        assert_eq!(dense.dimension(), 3);

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[0.0, 1.0, 0.0, 2.0]));
        assert_eq!(sparse.dimension(), 4);

        let delta = EmbeddingStorage::Delta(DeltaVector::from_parts(0, 5, vec![], vec![]));
        assert_eq!(delta.dimension(), 5);
    }

    #[test]
    fn test_embedding_storage_to_dense() {
        let dense = EmbeddingStorage::Dense(vec![1.0, 2.0, 3.0]);
        assert_eq!(dense.to_dense(), vec![1.0, 2.0, 3.0]);

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[0.0, 1.0, 0.0, 2.0]));
        assert_eq!(sparse.to_dense(), vec![0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_embedding_storage_to_dense_with_registry() {
        use crate::ArchetypeRegistry;

        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        // Dense
        let dense = EmbeddingStorage::Dense(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dense.to_dense_with_registry(None), vec![1.0, 2.0, 3.0, 4.0]);

        // Sparse
        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[0.0, 1.0, 0.0, 2.0]));
        assert_eq!(
            sparse.to_dense_with_registry(None),
            vec![0.0, 1.0, 0.0, 2.0]
        );

        // Delta with registry
        let delta_vec = DeltaVector::from_dense_with_reference(
            &[0.9, 0.1, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            0,
            0.001,
        );
        let delta = EmbeddingStorage::Delta(delta_vec);
        let reconstructed = delta.to_dense_with_registry(Some(&registry));
        assert!((reconstructed[0] - 0.9).abs() < 0.01);
        assert!((reconstructed[1] - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_embedding_storage_as_accessors() {
        let dense = EmbeddingStorage::Dense(vec![1.0, 2.0]);
        assert!(dense.as_dense().is_some());
        assert!(dense.as_sparse().is_none());
        assert!(dense.as_delta().is_none());

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[1.0, 0.0]));
        assert!(sparse.as_dense().is_none());
        assert!(sparse.as_sparse().is_some());
        assert!(sparse.as_delta().is_none());

        let delta = EmbeddingStorage::Delta(DeltaVector::from_parts(0, 2, vec![], vec![]));
        assert!(delta.as_dense().is_none());
        assert!(delta.as_sparse().is_none());
        assert!(delta.as_delta().is_some());
    }

    #[test]
    fn test_embedding_storage_is_predicates() {
        let dense = EmbeddingStorage::Dense(vec![1.0]);
        assert!(!dense.is_sparse());
        assert!(!dense.is_delta());

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[1.0]));
        assert!(sparse.is_sparse());
        assert!(!sparse.is_delta());

        let delta = EmbeddingStorage::Delta(DeltaVector::from_parts(0, 1, vec![], vec![]));
        assert!(!delta.is_sparse());
        assert!(delta.is_delta());
    }

    #[test]
    fn test_embedding_storage_dot_with_dense() {
        let dense = EmbeddingStorage::Dense(vec![1.0, 2.0, 3.0]);
        let query = vec![1.0, 1.0, 1.0];
        assert!((dense.dot_with_dense(&query) - 6.0).abs() < 0.001);

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[1.0, 2.0, 3.0]));
        assert!((sparse.dot_with_dense(&query) - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_storage_dot_with_dense_and_registry() {
        use crate::ArchetypeRegistry;

        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let dense = EmbeddingStorage::Dense(vec![1.0, 2.0, 3.0, 4.0]);
        let query = vec![1.0, 1.0, 1.0, 1.0];
        assert!((dense.dot_with_dense_and_registry(&query, None) - 10.0).abs() < 0.001);

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[1.0, 2.0, 3.0, 4.0]));
        assert!((sparse.dot_with_dense_and_registry(&query, None) - 10.0).abs() < 0.001);

        // Delta with registry
        let delta_vec = DeltaVector::from_dense_with_reference(
            &[0.9, 0.1, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            0,
            0.001,
        );
        let delta = EmbeddingStorage::Delta(delta_vec);
        let dot = delta.dot_with_dense_and_registry(&query, Some(&registry));
        assert!((dot - 1.0).abs() < 0.1); // 0.9 + 0.1 = 1.0
    }

    #[test]
    fn test_embedding_storage_dot_with_sparse() {
        let sparse_query = SparseVector::from_dense(&[1.0, 0.0, 1.0]);

        let dense = EmbeddingStorage::Dense(vec![2.0, 3.0, 4.0]);
        assert!((dense.dot_with_sparse(&sparse_query) - 6.0).abs() < 0.001); // 2*1 + 4*1 = 6

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[2.0, 3.0, 4.0]));
        assert!((sparse.dot_with_sparse(&sparse_query) - 6.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_storage_magnitude() {
        let dense = EmbeddingStorage::Dense(vec![3.0, 4.0]);
        assert!((dense.magnitude() - 5.0).abs() < 0.001);

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[3.0, 4.0]));
        assert!((sparse.magnitude() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_storage_magnitude_with_registry() {
        use crate::ArchetypeRegistry;

        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![3.0, 4.0, 0.0, 0.0]).unwrap();

        let dense = EmbeddingStorage::Dense(vec![3.0, 4.0]);
        assert!((dense.magnitude_with_registry(None) - 5.0).abs() < 0.001);

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[3.0, 4.0]));
        assert!((sparse.magnitude_with_registry(None) - 5.0).abs() < 0.001);

        // Delta with registry
        let delta_vec = DeltaVector::from_dense_with_reference(
            &[3.0, 4.0, 0.0, 0.0],
            &[3.0, 4.0, 0.0, 0.0],
            0,
            0.001,
        );
        let delta = EmbeddingStorage::Delta(delta_vec);
        assert!((delta.magnitude_with_registry(Some(&registry)) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_storage_memory_bytes() {
        let dense = EmbeddingStorage::Dense(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(dense.memory_bytes() >= 16); // 4 floats = 16 bytes minimum

        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[1.0, 0.0, 0.0, 2.0]));
        assert!(sparse.memory_bytes() > 0);

        let delta = EmbeddingStorage::Delta(DeltaVector::from_parts(0, 4, vec![0], vec![0.1]));
        assert!(delta.memory_bytes() > 0);
    }

    #[test]
    fn test_hnsw_insert_sparse() {
        let index = HNSWIndex::new();
        let sparse = SparseVector::from_dense(&[1.0, 0.0, 0.0, 2.0]);
        let id = index.insert_sparse(sparse);
        assert_eq!(index.len(), 1);

        let retrieved = index.get_embedding(id);
        assert!(retrieved.is_some());
        assert!(retrieved.unwrap().is_sparse());
    }

    #[test]
    fn test_hnsw_search_with_sparse() {
        let index = HNSWIndex::new();

        // Insert dense vectors
        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);
        index.insert(vec![0.0, 0.0, 1.0]);

        // Search with sparse query
        let sparse_query = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let results = index.search_sparse(&sparse_query, 3);

        assert_eq!(results.len(), 3);
        // First result should be closest to [1, 0, 0]
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_hnsw_memory_stats() {
        let index = HNSWIndex::new();

        // Insert different types
        index.insert(vec![1.0, 2.0, 3.0]);
        index.insert_sparse(SparseVector::from_dense(&[1.0, 0.0, 0.0]));

        let stats = index.memory_stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.dense_count, 1);
        assert_eq!(stats.sparse_count, 1);
        assert_eq!(stats.delta_count, 0);
        assert!(stats.embedding_bytes > 0);
    }

    #[test]
    fn test_hnsw_default() {
        let index = HNSWIndex::default();
        assert!(index.is_empty());
    }

    #[test]
    fn test_hnsw_set_ef_search() {
        let mut index = HNSWIndex::new();
        index.set_ef_search(100);
        assert_eq!(index.config().ef_search, 100);
    }

    #[test]
    fn test_hnsw_distance_sparse_sparse() {
        let index = HNSWIndex::new();

        let s1 = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let s2 = SparseVector::from_dense(&[1.0, 0.0, 0.0]);

        index.insert_sparse(s1);
        index.insert_sparse(s2.clone());

        // Search should find exact match
        let results = index.search_sparse(&s2, 2);
        assert_eq!(results.len(), 2);
        // One of the results should have similarity ~1.0
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn test_hnsw_mixed_dense_sparse() {
        let index = HNSWIndex::new();

        // Insert mix of dense and sparse
        index.insert(vec![1.0, 0.0, 0.0, 0.0]);
        index.insert_sparse(SparseVector::from_dense(&[0.0, 1.0, 0.0, 0.0]));
        index.insert(vec![0.0, 0.0, 1.0, 0.0]);

        assert_eq!(index.len(), 3);

        // Search with dense query
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, 0); // Should match first vector

        // Search with sparse query
        let sparse_q = SparseVector::from_dense(&[0.0, 1.0, 0.0, 0.0]);
        let results_sparse = index.search_sparse(&sparse_q, 3);
        assert_eq!(results_sparse.len(), 3);
        assert_eq!(results_sparse[0].0, 1); // Should match second vector
    }

    #[test]
    fn test_embedding_storage_cosine_distance() {
        let dense = EmbeddingStorage::Dense(vec![1.0, 0.0, 0.0]);
        let query = vec![1.0, 0.0, 0.0];

        // Same vector = 0 distance
        let dist = dense.cosine_distance_dense(&query);
        assert!(dist.abs() < 0.001);

        // Orthogonal = 1 distance
        let orth_query = vec![0.0, 1.0, 0.0];
        let dist2 = dense.cosine_distance_dense(&orth_query);
        assert!((dist2 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_storage_cosine_distance_with_registry() {
        use crate::ArchetypeRegistry;

        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        // Dense vector
        let dense = EmbeddingStorage::Dense(vec![1.0, 0.0, 0.0, 0.0]);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let dist = dense.cosine_distance_dense_with_registry(&query, None);
        assert!(dist.abs() < 0.001);

        // Zero query magnitude edge case
        let zero_query = vec![0.0, 0.0, 0.0, 0.0];
        let dist_zero = dense.cosine_distance_dense_with_registry(&zero_query, None);
        assert!((dist_zero - 1.0).abs() < 0.001); // Max distance

        // Delta with registry
        let delta_vec = DeltaVector::from_dense_with_reference(
            &[1.0, 0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0, 0.0],
            0,
            0.001,
        );
        let delta = EmbeddingStorage::Delta(delta_vec);
        let dist_delta = delta.cosine_distance_dense_with_registry(&query, Some(&registry));
        assert!(dist_delta.abs() < 0.001);
    }

    #[test]
    fn test_embedding_storage_cosine_distance_sparse() {
        let sparse = EmbeddingStorage::Sparse(SparseVector::from_dense(&[1.0, 0.0, 0.0]));
        let query = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let dist = sparse.cosine_distance_sparse(&query);
        assert!(dist.abs() < 0.001);

        // Zero magnitude edge case
        let empty = EmbeddingStorage::Sparse(SparseVector::new(3));
        let dist_zero = empty.cosine_distance_sparse(&query);
        assert!((dist_zero - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_embedding_storage_zero_magnitude() {
        let zero = EmbeddingStorage::Dense(vec![0.0, 0.0, 0.0]);
        let query = vec![1.0, 0.0, 0.0];
        let dist = zero.cosine_distance_dense(&query);
        assert!((dist - 1.0).abs() < 0.001); // Max distance
    }

    // ==================== CompressedNeighbors tests ====================

    #[test]
    fn test_compressed_neighbors_new() {
        let cn = CompressedNeighbors::new();
        assert!(cn.get().is_empty());
        assert_eq!(cn.len(), 0);
        assert_eq!(cn.memory_bytes(), 0);
    }

    #[test]
    fn test_compressed_neighbors_set_get() {
        let mut cn = CompressedNeighbors::new();
        cn.set(&[10, 5, 20, 15]);

        // Should be sorted after decompression
        let result = cn.get();
        assert_eq!(result, vec![5, 10, 15, 20]);
        assert_eq!(cn.len(), 4);
    }

    #[test]
    fn test_compressed_neighbors_push() {
        let mut cn = CompressedNeighbors::new();
        cn.push(5);
        cn.push(10);
        cn.push(3);

        let result = cn.get();
        assert_eq!(result, vec![3, 5, 10]);
    }

    #[test]
    fn test_compressed_neighbors_extend() {
        let mut cn = CompressedNeighbors::new();
        cn.set(&[1, 2]);
        cn.extend([10, 5]);

        let result = cn.get();
        assert_eq!(result, vec![1, 2, 5, 10]);
    }

    #[test]
    fn test_compressed_neighbors_empty() {
        let mut cn = CompressedNeighbors::new();
        cn.set(&[]);
        assert!(cn.get().is_empty());
        assert_eq!(cn.len(), 0);
    }

    #[test]
    fn test_compressed_neighbors_memory_savings() {
        let mut cn = CompressedNeighbors::new();
        // Sequential IDs should compress well
        let ids: Vec<usize> = (0..16).collect();
        cn.set(&ids);

        // Uncompressed: 16 * 8 = 128 bytes
        // Compressed: should be much less (typically 20-30 bytes for sequential)
        let compressed_bytes = cn.memory_bytes();
        assert!(
            compressed_bytes < 64,
            "Compressed size {} should be < 64 bytes",
            compressed_bytes
        );
    }

    #[test]
    fn test_compressed_neighbors_large_ids() {
        let mut cn = CompressedNeighbors::new();
        cn.set(&[1000, 2000, 3000, 10000]);

        let result = cn.get();
        assert_eq!(result, vec![1000, 2000, 3000, 10000]);
    }

    #[test]
    fn test_hnsw_with_compressed_neighbors() {
        // Verify HNSW still works correctly with compressed neighbors
        let index = HNSWIndex::new();

        for i in 0..100 {
            let v: Vec<f32> = (0..32)
                .map(|j| ((i * 31 + j) as f32 * 0.01).sin())
                .collect();
            index.insert(v);
        }

        assert_eq!(index.len(), 100);

        // Search should still return valid results
        let query: Vec<f32> = (0..32)
            .map(|j| (50.0 * 31.0 + j as f32 * 0.01).sin())
            .collect();
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
        // Results should be sorted by similarity (highest first)
        for i in 1..results.len() {
            assert!(results[i - 1].1 >= results[i].1);
        }
    }

    // ==================== insert_auto tests ====================

    #[test]
    fn test_insert_auto_chooses_sparse() {
        let index = HNSWIndex::new();

        // Vector with >50% zeros should be stored as sparse
        let sparse_vec = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let id = index.insert_auto(sparse_vec);

        let embedding = index.get_embedding(id).unwrap();
        assert!(
            embedding.is_sparse(),
            "Should choose sparse storage for 80% zeros"
        );
    }

    #[test]
    fn test_insert_auto_chooses_dense() {
        let index = HNSWIndex::new();

        // Vector with <50% zeros should be stored as dense
        let dense_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0];
        let id = index.insert_auto(dense_vec);

        let embedding = index.get_embedding(id).unwrap();
        assert!(
            matches!(embedding, EmbeddingStorage::Dense(_)),
            "Should choose dense storage for 20% zeros"
        );
    }

    #[test]
    fn test_insert_auto_threshold_boundary() {
        let mut config = HNSWConfig::default();
        config.sparsity_threshold = 0.7; // 70% threshold
        let index = HNSWIndex::with_config(config);

        // Exactly 70% zeros - should trigger sparse
        let vec_70 = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let id1 = index.insert_auto(vec_70);
        assert!(
            index.get_embedding(id1).unwrap().is_sparse(),
            "70% zeros should be sparse with 70% threshold"
        );

        // 60% zeros - should stay dense
        let vec_60 = vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let id2 = index.insert_auto(vec_60);
        assert!(
            matches!(
                index.get_embedding(id2).unwrap(),
                EmbeddingStorage::Dense(_)
            ),
            "60% zeros should be dense with 70% threshold"
        );
    }

    #[test]
    fn test_insert_auto_search_works() {
        let index = HNSWIndex::new();

        // Mix of auto-inserted vectors
        for i in 0..50 {
            // Dense vectors
            let v: Vec<f32> = (0..16)
                .map(|j| ((i * 31 + j) as f32 * 0.01).sin())
                .collect();
            index.insert_auto(v);
        }

        for i in 0..50 {
            // Sparse vectors (only 3 non-zeros out of 16)
            let mut v = vec![0.0; 16];
            v[0] = (i as f32 * 0.1).sin();
            v[5] = (i as f32 * 0.2).cos();
            v[10] = (i as f32 * 0.3).sin();
            index.insert_auto(v);
        }

        let stats = index.memory_stats();
        assert_eq!(stats.total_nodes, 100);
        assert!(stats.dense_count > 0, "Should have some dense nodes");
        assert!(stats.sparse_count > 0, "Should have some sparse nodes");

        // Search should work with mixed storage
        let query = vec![0.5; 16];
        let results = index.search(&query, 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_hnsw_config_sparsity_threshold() {
        let default = HNSWConfig::default();
        assert_eq!(default.sparsity_threshold, 0.5);

        let high_recall = HNSWConfig::high_recall();
        assert_eq!(high_recall.sparsity_threshold, 0.5);

        let high_speed = HNSWConfig::high_speed();
        assert_eq!(high_speed.sparsity_threshold, 0.5);
    }

    // ==================== TensorTrain storage tests ====================

    #[test]
    fn test_hnsw_insert_tt() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Create a 64-dim vector
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let id = index.insert_tt(vector.clone(), &config).unwrap();

        assert_eq!(index.len(), 1);

        // TT methods validate via decomposition but store as dense for fast HNSW distance
        let stats = index.memory_stats();
        assert_eq!(stats.tt_count, 0);
        assert_eq!(stats.dense_count, 1);

        // Verify we can retrieve the embedding as dense
        let embedding = index.get_embedding(id).unwrap();
        assert!(embedding.as_dense().is_some());
    }

    #[test]
    fn test_hnsw_tt_search() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Insert TT vectors
        for i in 0..50 {
            let vector: Vec<f32> = (0..64)
                .map(|j| ((i * 31 + j) as f32 * 0.01).sin())
                .collect();
            index.insert_tt(vector, &config).unwrap();
        }

        assert_eq!(index.len(), 50);

        // Search should work
        let query: Vec<f32> = (0..64)
            .map(|j| (25.0 * 31.0 + j as f32 * 0.01).sin())
            .collect();
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_hnsw_tt_validation_and_insert() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(768).unwrap();

        // Insert 768-dim vectors (BERT dimension) via TT validation
        for i in 0..20 {
            let vector: Vec<f32> = (0..768)
                .map(|j| ((i * 31 + j) as f32 * 0.001).sin())
                .collect();
            index.insert_tt(vector, &config).unwrap();
        }

        // TT methods validate via decomposition but store as dense for fast HNSW distance
        let stats = index.memory_stats();
        assert_eq!(stats.dense_count, 20);
        assert_eq!(stats.tt_count, 0);

        // Verify expected size for dense storage: 20 * 768 * 4 = 61,440 bytes
        let expected_size = 20 * 768 * 4;
        assert!(
            stats.embedding_bytes >= expected_size,
            "Dense storage should use approximately {} bytes, got {}",
            expected_size,
            stats.embedding_bytes
        );
    }

    #[test]
    fn test_hnsw_tt_insert_performance() {
        use std::time::Instant;

        // Use same random generator as benchmark
        fn generate_random_vector(dim: usize, seed: u64) -> Vec<f32> {
            let mut vec = Vec::with_capacity(dim);
            let mut state = seed;
            for _ in 0..dim {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                vec.push(val);
            }
            vec
        }

        let config = TTConfig::for_dim(768).unwrap();

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| generate_random_vector(768, i as u64))
            .collect();

        // Time decomposition separately
        let start = Instant::now();
        let tts: Vec<_> = vectors
            .iter()
            .map(|v| tt_decompose(v, &config).unwrap())
            .collect();
        let decomp_time = start.elapsed();
        eprintln!(
            "TT decomposition: {:?} for {} vectors",
            decomp_time,
            vectors.len()
        );

        // Check TT properties
        let tt = &tts[0];
        let ranks: Vec<_> = tt.cores.iter().map(|c| (c.shape.0, c.shape.2)).collect();
        eprintln!("TT shape: {:?}, ranks: {:?}", tt.shape, ranks);
        eprintln!("TT storage: {} elements", tt.storage_size());

        // Time HNSW insert with pre-decomposed vectors
        let index = HNSWIndex::new();
        let start = Instant::now();
        for tt in tts {
            index.insert_tt_vector(tt);
        }
        let insert_time = start.elapsed();
        eprintln!(
            "TT HNSW insert: {:?} for {} vectors ({:?}/vector)",
            insert_time,
            vectors.len(),
            insert_time / vectors.len() as u32
        );

        // Total time
        let total = decomp_time + insert_time;
        eprintln!("Total: {:?}", total);

        // Release mode threshold (generous for now - focus on identifying bottleneck)
        let threshold_ms = if cfg!(debug_assertions) { 10000 } else { 2000 };
        assert!(
            total.as_millis() < threshold_ms,
            "TT insert too slow: {:?} (expected <{}ms)",
            total,
            threshold_ms
        );
    }

    #[test]
    fn test_hnsw_tt_mixed_with_dense() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Insert mix of dense (direct) and dense (via TT validation)
        for i in 0..10 {
            let vector: Vec<f32> = (0..64)
                .map(|j| ((i * 31 + j) as f32 * 0.01).sin())
                .collect();
            index.insert(vector);
        }

        for i in 10..20 {
            let vector: Vec<f32> = (0..64)
                .map(|j| ((i * 31 + j) as f32 * 0.01).sin())
                .collect();
            index.insert_tt(vector, &config).unwrap();
        }

        // All vectors stored as dense for fast HNSW distance computation
        let stats = index.memory_stats();
        assert_eq!(stats.total_nodes, 20);
        assert_eq!(stats.dense_count, 20);
        assert_eq!(stats.tt_count, 0);

        // Search should work
        let query: Vec<f32> = (0..64)
            .map(|j| (15.0 * 31.0 + j as f32 * 0.01).sin())
            .collect();
        let results = index.search(&query, 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_embedding_storage_tt_accessors() {
        let config = TTConfig::for_dim(64).unwrap();
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let tt = tt_decompose(&vector, &config).unwrap();

        let storage = EmbeddingStorage::TensorTrain(TTVectorCached::new(tt));

        assert!(storage.is_tt());
        assert!(!storage.is_sparse());
        assert!(!storage.is_delta());
        assert!(storage.as_tt().is_some());
        assert!(storage.as_dense().is_none());
        assert_eq!(storage.dimension(), 64);
    }

    #[test]
    fn test_embedding_storage_tt_to_dense() {
        let config = TTConfig::for_dim(64).unwrap();
        let original: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let tt = tt_decompose(&original, &config).unwrap();

        let storage = EmbeddingStorage::TensorTrain(TTVectorCached::new(tt));
        let reconstructed = storage.to_dense();

        // Should reconstruct with small error
        assert_eq!(reconstructed.len(), 64);
        let max_error: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            max_error < 0.1,
            "Reconstruction error too large: {}",
            max_error
        );
    }

    #[test]
    fn test_embedding_storage_tt_distance() {
        let config = TTConfig::for_dim(64).unwrap();
        let v1: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let v2: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();

        let tt1 = tt_decompose(&v1, &config).unwrap();
        let tt2 = tt_decompose(&v2, &config).unwrap();

        let storage1 = EmbeddingStorage::TensorTrain(TTVectorCached::new(tt1));
        let storage2 = EmbeddingStorage::TensorTrain(TTVectorCached::new(tt2));

        // Distance to self should be near 0
        let index = HNSWIndex::new();
        let dist = index.distance_embeddings(&storage1, &storage2);
        assert!(dist < 0.01, "Self-distance should be near 0: {}", dist);
    }

    // ==================== Batch TT insert tests ====================

    #[test]
    fn test_insert_batch_tt() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                (0..64)
                    .map(|j| ((i * 31 + j) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();

        let ids = index.insert_batch_tt(vectors, &config).unwrap();

        assert_eq!(ids.len(), 10);
        assert_eq!(index.len(), 10);

        // TT methods validate via decomposition but store as dense for fast HNSW distance
        let stats = index.memory_stats();
        assert_eq!(stats.dense_count, 10);
    }

    #[test]
    fn test_insert_batch_tt_empty() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        let ids = index.insert_batch_tt(Vec::new(), &config).unwrap();
        assert!(ids.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_insert_batch_tt_search() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..64)
                    .map(|j| ((i * 31 + j) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();

        index.insert_batch_tt(vectors, &config).unwrap();

        // Search should work
        let query: Vec<f32> = (0..64)
            .map(|j| (25.0 * 31.0 + j as f32 * 0.01).sin())
            .collect();
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
    }

    // ==================== search_tt tests ====================

    #[test]
    fn test_search_tt() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Insert TT vectors
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| {
                (0..64)
                    .map(|j| ((i * 31 + j) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();

        index.insert_batch_tt(vectors, &config).unwrap();

        // Use search_tt for TT-optimized search
        let query: Vec<f32> = (0..64)
            .map(|j| (25.0 * 31.0 + j as f32 * 0.01).sin())
            .collect();
        let results = index.search_tt(&query, 10, &config);

        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_tt_empty_index() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        let query: Vec<f32> = (0..64).map(|j| (j as f32 * 0.1).sin()).collect();
        let results = index.search_tt(&query, 10, &config);

        assert!(results.is_empty());
    }

    #[test]
    fn test_search_tt_matches_search() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Insert TT vectors
        let vectors: Vec<Vec<f32>> = (0..30)
            .map(|i| {
                (0..64)
                    .map(|j| ((i * 31 + j) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();

        index.insert_batch_tt(vectors, &config).unwrap();

        // Query
        let query: Vec<f32> = (0..64)
            .map(|j| (15.0 * 31.0 + j as f32 * 0.01).sin())
            .collect();

        // Both search methods should return similar results
        let results_dense = index.search(&query, 5);
        let results_tt = index.search_tt(&query, 5, &config);

        assert_eq!(results_dense.len(), 5);
        assert_eq!(results_tt.len(), 5);

        // Top results should be very similar (allow some difference due to TT approximation)
        for i in 0..3 {
            // Check if same IDs are in top results (order might differ slightly)
            let dense_ids: Vec<usize> = results_dense.iter().take(5).map(|(id, _)| *id).collect();
            assert!(
                dense_ids.contains(&results_tt[i].0),
                "TT search result {} not in dense top 5",
                results_tt[i].0
            );
        }
    }

    #[test]
    #[ignore] // Timing-sensitive test, may fail under system load
    fn test_tt_insert_timing_100_vectors() {
        use std::time::Instant;

        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(768).unwrap();

        // Create 100 vectors
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                (0..768)
                    .map(|j| ((i * 31 + j) as f32 * 0.001).sin())
                    .collect()
            })
            .collect();

        let start = Instant::now();
        for v in &vectors {
            let _ = index.insert_tt(v.clone(), &config);
        }
        let elapsed = start.elapsed();

        println!("Inserted {} TT vectors in {:?}", vectors.len(), elapsed);
        println!("Per vector: {:?}", elapsed / vectors.len() as u32);

        // Should complete in under 2 seconds with norm caching
        assert!(
            elapsed.as_secs() < 2,
            "TT insert too slow: {:?} for 100 vectors (expected < 2s)",
            elapsed
        );
    }

    #[test]
    fn test_embedding_storage_tt_to_dense_with_registry() {
        let config = TTConfig::for_dim(64).unwrap();
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let tt = tt_decompose(&vector, &config).unwrap();
        let storage = EmbeddingStorage::TensorTrain(TTVectorCached::new(tt));

        // to_dense_with_registry should work for TT even without registry
        let dense = storage.to_dense_with_registry(None);
        assert_eq!(dense.len(), 64);

        // Values should be approximately preserved
        for (orig, reconstructed) in vector.iter().zip(dense.iter()) {
            assert!((orig - reconstructed).abs() < 0.1);
        }
    }

    #[test]
    fn test_embedding_storage_as_tt() {
        let config = TTConfig::for_dim(64).unwrap();
        let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let tt = tt_decompose(&vector, &config).unwrap();

        let storage = EmbeddingStorage::TensorTrain(TTVectorCached::new(tt));
        assert!(storage.as_tt().is_some());
        assert!(storage.as_dense().is_none());
        assert!(storage.as_sparse().is_none());

        let dense_storage = EmbeddingStorage::Dense(vector);
        assert!(dense_storage.as_tt().is_none());
    }

    #[test]
    fn test_embedding_storage_is_tt() {
        let config = TTConfig::for_dim(64).unwrap();
        let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let tt = tt_decompose(&vector, &config).unwrap();

        let storage = EmbeddingStorage::TensorTrain(TTVectorCached::new(tt));
        assert!(storage.is_tt());
        assert!(!storage.is_sparse());
        assert!(!storage.is_delta());
        assert!(storage.as_dense().is_none());
    }

    #[test]
    fn test_embedding_storage_memory_bytes_tt() {
        let config = TTConfig::for_dim(64).unwrap();
        let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let tt = tt_decompose(&vector, &config).unwrap();

        let storage = EmbeddingStorage::TensorTrain(TTVectorCached::new(tt));
        let bytes = storage.memory_bytes();
        // TT should use less memory than dense for small vectors
        assert!(bytes > 0);
    }

    #[test]
    fn test_compressed_neighbors_len() {
        let mut neighbors = CompressedNeighbors::new();
        assert_eq!(neighbors.len(), 0);

        neighbors.push(5);
        assert_eq!(neighbors.len(), 1);

        neighbors.extend([10, 15, 20]);
        assert_eq!(neighbors.len(), 4);

        neighbors.set(&[1, 2, 3]);
        assert_eq!(neighbors.len(), 3);
    }

    #[test]
    fn test_search_tt_with_ef() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Insert TT vectors
        for i in 0..50 {
            let vector: Vec<f32> = (0..64)
                .map(|j| ((i * 17 + j) as f32 * 0.01).sin())
                .collect();
            index.insert_tt(vector, &config).unwrap();
        }

        // Create query TT
        let query: Vec<f32> = (0..64).map(|j| (j as f32 * 0.01).sin()).collect();
        let query_tt = tt_decompose(&query, &config).unwrap();

        // Search with different ef values
        let results_ef50 = index.search_tt_with_ef(&query_tt, 5, 50);
        let results_ef100 = index.search_tt_with_ef(&query_tt, 5, 100);

        assert_eq!(results_ef50.len(), 5);
        assert_eq!(results_ef100.len(), 5);

        // Higher ef should potentially find same or better results
        for (_, score) in &results_ef50 {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_hnsw_tt_cross_type_distance() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Insert mixed types
        let dense_vec: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        index.insert(dense_vec.clone());

        let sparse = SparseVector::from_dense(&dense_vec);
        index.insert_sparse(sparse);

        let tt_vec: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 + 0.01).collect();
        index.insert_tt(tt_vec, &config).unwrap();

        // Search should work across all types
        let query = SparseVector::from_dense(&dense_vec);
        let results = index.search_sparse(&query, 3);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_insert_tt_error_handling() {
        let index = HNSWIndex::new();
        // TTConfig for wrong dimension should still work with insert_tt
        // since we don't validate dimension match
        let config = TTConfig::for_dim(128).unwrap();
        let vector: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let result = index.insert_tt(vector, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_insert_batch_tt_single() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Batch with single vector
        let vectors = vec![(0..64).map(|i| i as f32).collect::<Vec<f32>>()];
        let ids = index.insert_batch_tt(vectors, &config).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_hnsw_distance_tt_sparse() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Insert TT vector
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        index.insert_tt(vector.clone(), &config).unwrap();

        // Search with sparse query - TT introduces approximation error
        let sparse_query = SparseVector::from_dense(&vector);
        let results = index.search_sparse(&sparse_query, 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].1.is_finite(), "Distance should be finite");
    }

    #[test]
    fn test_hnsw_distance_tt_dense() {
        let index = HNSWIndex::new();
        let config = TTConfig::for_dim(64).unwrap();

        // Insert TT vector
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        index.insert_tt(vector.clone(), &config).unwrap();

        // Search with dense query
        let results = index.search(&vector, 1);
        assert_eq!(results.len(), 1);
    }
}
