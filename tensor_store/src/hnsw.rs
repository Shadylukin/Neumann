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

use crate::delta_vector::{ArchetypeRegistry, DeltaVector};
use crate::SparseVector;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::RwLock;

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

/// Storage for embeddings - can be dense, sparse, or delta-encoded.
///
/// Sparse storage is used when vectors have high sparsity (>70% zeros),
/// providing memory savings and faster operations.
///
/// Delta storage is used when vectors cluster around common archetypes,
/// storing only the difference from a reference vector.
#[derive(Debug, Clone)]
pub enum EmbeddingStorage {
    /// Dense vector storage (traditional)
    Dense(Vec<f32>),
    /// Sparse vector storage (only non-zero values)
    Sparse(SparseVector),
    /// Delta-encoded storage (difference from archetype)
    Delta(DeltaVector),
}

impl EmbeddingStorage {
    /// Get the dimension of the embedding.
    pub fn dimension(&self) -> usize {
        match self {
            EmbeddingStorage::Dense(v) => v.len(),
            EmbeddingStorage::Sparse(s) => s.dimension(),
            EmbeddingStorage::Delta(d) => d.dimension(),
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
        }
    }

    /// Get as dense slice (only works for Dense variant).
    pub fn as_dense(&self) -> Option<&[f32]> {
        match self {
            EmbeddingStorage::Dense(v) => Some(v),
            EmbeddingStorage::Sparse(_) | EmbeddingStorage::Delta(_) => None,
        }
    }

    /// Get as sparse reference (only works for Sparse variant).
    pub fn as_sparse(&self) -> Option<&SparseVector> {
        match self {
            EmbeddingStorage::Sparse(s) => Some(s),
            EmbeddingStorage::Dense(_) | EmbeddingStorage::Delta(_) => None,
        }
    }

    /// Get as delta reference (only works for Delta variant).
    pub fn as_delta(&self) -> Option<&DeltaVector> {
        match self {
            EmbeddingStorage::Delta(d) => Some(d),
            EmbeddingStorage::Dense(_) | EmbeddingStorage::Sparse(_) => None,
        }
    }

    /// Check if this is sparse storage.
    pub fn is_sparse(&self) -> bool {
        matches!(self, EmbeddingStorage::Sparse(_))
    }

    /// Check if this is delta storage.
    pub fn is_delta(&self) -> bool {
        matches!(self, EmbeddingStorage::Delta(_))
    }

    /// Compute dot product with a dense query vector.
    ///
    /// For Delta storage, requires archetype registry.
    #[inline]
    pub fn dot_with_dense(&self, query: &[f32]) -> f32 {
        match self {
            EmbeddingStorage::Dense(v) => simd::dot_product(v, query),
            EmbeddingStorage::Sparse(s) => s.dot_dense(query),
            EmbeddingStorage::Delta(_) => {
                panic!("Delta storage requires archetype for dot product - use dot_with_dense_and_registry")
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
        }
    }

    /// Compute magnitude (L2 norm).
    ///
    /// For Delta storage, requires archetype registry.
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

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            EmbeddingStorage::Dense(v) => v.len() * 4,
            EmbeddingStorage::Sparse(s) => s.memory_bytes(),
            EmbeddingStorage::Delta(d) => d.memory_bytes(),
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

/// A node in the HNSW graph, representing a vector with connections at each layer.
#[derive(Debug)]
struct HNSWNode {
    /// The embedding (dense or sparse)
    embedding: EmbeddingStorage,
    /// Connections at each layer (layer -> list of neighbor IDs)
    neighbors: Vec<RwLock<Vec<usize>>>,
}

impl HNSWNode {
    fn new(embedding: EmbeddingStorage, max_layer: usize) -> Self {
        let neighbors = (0..=max_layer).map(|_| RwLock::new(Vec::new())).collect();
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
        }
    }

    /// Get the number of indexed vectors.
    pub fn len(&self) -> usize {
        self.nodes.read().unwrap().len()
    }

    /// Check if the index is empty.
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

    /// Insert an embedding (dense or sparse) into the index.
    pub fn insert_embedding(&self, embedding: EmbeddingStorage) -> usize {
        let node_level = self.random_level();
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

        // Get the embedding we just inserted (as dense for internal operations)
        let nodes = self.nodes.read().unwrap();
        let query = nodes[node_id].embedding.to_dense();

        // Find entry point at the top layer and descend
        let mut current_node = entry_id;

        // Descend from top layer to node_level + 1 (greedy search)
        for layer in (node_level + 1..=current_max).rev() {
            current_node = self.search_layer_greedy(&nodes, &query, current_node, layer);
        }

        // For layers from min(node_level, current_max) down to 0, do full search and connect
        let connect_from = node_level.min(current_max);
        for layer in (0..=connect_from).rev() {
            let neighbors = self.search_layer(
                &nodes,
                &query,
                current_node,
                self.config.ef_construction,
                layer,
            );

            // Connect to the best M neighbors
            let m = if layer == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let selected: Vec<usize> = neighbors.iter().take(m).map(|n| n.id).collect();

            // Add bidirectional connections
            {
                let mut node_neighbors = nodes[node_id].neighbors[layer].write().unwrap();
                node_neighbors.extend(selected.iter().copied());
            }

            for &neighbor_id in &selected {
                let mut neighbor_neighbors = nodes[neighbor_id].neighbors[layer].write().unwrap();
                neighbor_neighbors.push(node_id);

                // Prune if over capacity
                if neighbor_neighbors.len() > m {
                    // Keep the closest M neighbors
                    let neighbor_embedding = &nodes[neighbor_id].embedding;
                    let mut with_dist: Vec<_> = neighbor_neighbors
                        .iter()
                        .map(|&id| {
                            (
                                id,
                                self.distance_embeddings(neighbor_embedding, &nodes[id].embedding),
                            )
                        })
                        .collect();
                    with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    *neighbor_neighbors = with_dist.into_iter().take(m).map(|(id, _)| id).collect();
                }
            }

            if !neighbors.is_empty() {
                current_node = neighbors[0].id;
            }
        }

        drop(nodes);

        // Update entry point if new node has higher level
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

        let nodes = self.nodes.read().unwrap();
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);

        // Descend from top layer to layer 1 (greedy search)
        let mut current_node = entry_id;
        for layer in (1..=max_layer).rev() {
            current_node = self.search_layer_greedy(&nodes, query, current_node, layer);
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

        let nodes = self.nodes.read().unwrap();
        let max_layer = self.max_layer.load(AtomicOrdering::Relaxed);

        // Descend from top layer to layer 1 (greedy search)
        let mut current_node = entry_id;
        for layer in (1..=max_layer).rev() {
            current_node = self.search_layer_greedy_sparse(&nodes, query, current_node, layer);
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
            let neighbors = nodes[current].neighbors[layer].read().unwrap();
            let mut changed = false;

            for &neighbor_id in neighbors.iter() {
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
            let neighbors = nodes[current].neighbors[layer].read().unwrap();
            let mut changed = false;

            for &neighbor_id in neighbors.iter() {
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
            let neighbors = nodes[current.id].neighbors[layer].read().unwrap();
            for &neighbor_id in neighbors.iter() {
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

            let neighbors = nodes[current.id].neighbors[layer].read().unwrap();
            for &neighbor_id in neighbors.iter() {
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
        let mut total_bytes = 0usize;

        for node in nodes.iter() {
            match &node.embedding {
                EmbeddingStorage::Dense(_) => dense_count += 1,
                EmbeddingStorage::Sparse(_) => sparse_count += 1,
                EmbeddingStorage::Delta(_) => delta_count += 1,
            }
            total_bytes += node.embedding.memory_bytes();
        }

        HNSWMemoryStats {
            total_nodes: nodes.len(),
            dense_count,
            sparse_count,
            delta_count,
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
}
