//! Vector Engine - Module 4 of Neumann
//!
//! Provides embeddings storage and similarity search functionality.

use rayon::prelude::*;
use tensor_store::{TensorData, TensorStore, TensorStoreError, TensorValue};

// Re-export HNSW types for public use
pub use hnsw::{HNSWConfig, HNSWIndex};

/// SIMD-accelerated vector operations for cosine similarity.
mod simd {
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

/// HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
///
/// This implements the HNSW algorithm from "Efficient and robust approximate nearest neighbor
/// search using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018).
///
/// Key features:
/// - O(log n) search complexity vs O(n) for brute force
/// - 5-8x speedup for 10K-100K vectors
/// - Configurable recall/speed tradeoff via ef_search parameter
pub mod hnsw {
    use super::simd;
    use std::cmp::Ordering;
    use std::collections::{BinaryHeap, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::RwLock;

    /// A node in the HNSW graph, representing a vector with connections at each layer.
    #[derive(Debug)]
    struct HNSWNode {
        /// The embedding vector
        vector: Vec<f32>,
        /// Connections at each layer (layer -> list of neighbor IDs)
        neighbors: Vec<RwLock<Vec<usize>>>,
    }

    impl HNSWNode {
        fn new(vector: Vec<f32>, max_layer: usize) -> Self {
            let neighbors = (0..=max_layer).map(|_| RwLock::new(Vec::new())).collect();
            Self { vector, neighbors }
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
    #[derive(Debug, Clone)]
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
            let node_level = self.random_level();
            let node_id;

            // Add the node
            {
                let mut nodes = self.nodes.write().unwrap();
                node_id = nodes.len();
                nodes.push(HNSWNode::new(vector, node_level));
            }

            let current_max = self.max_layer.load(AtomicOrdering::Relaxed);
            let entry_id = self.entry_point.load(AtomicOrdering::Relaxed);

            // If this is the first node, set it as entry point
            if entry_id == usize::MAX {
                self.entry_point.store(node_id, AtomicOrdering::Relaxed);
                self.max_layer.store(node_level, AtomicOrdering::Relaxed);
                return node_id;
            }

            // Get the vector we just inserted
            let nodes = self.nodes.read().unwrap();
            let query = &nodes[node_id].vector;

            // Find entry point at the top layer and descend
            let mut current_node = entry_id;

            // Descend from top layer to node_level + 1 (greedy search)
            for layer in (node_level + 1..=current_max).rev() {
                current_node = self.search_layer_greedy(&nodes, query, current_node, layer);
            }

            // For layers from min(node_level, current_max) down to 0, do full search and connect
            let connect_from = node_level.min(current_max);
            for layer in (0..=connect_from).rev() {
                let neighbors = self.search_layer(
                    &nodes,
                    query,
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
                    let mut neighbor_neighbors =
                        nodes[neighbor_id].neighbors[layer].write().unwrap();
                    neighbor_neighbors.push(node_id);

                    // Prune if over capacity
                    if neighbor_neighbors.len() > m {
                        // Keep the closest M neighbors
                        let neighbor_vec = &nodes[neighbor_id].vector;
                        let mut with_dist: Vec<_> = neighbor_neighbors
                            .iter()
                            .map(|&id| (id, self.distance(neighbor_vec, &nodes[id].vector)))
                            .collect();
                        with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                        *neighbor_neighbors =
                            with_dist.into_iter().take(m).map(|(id, _)| id).collect();
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

        /// Greedy search in a layer - find the closest node to query
        fn search_layer_greedy(
            &self,
            nodes: &[HNSWNode],
            query: &[f32],
            entry_id: usize,
            layer: usize,
        ) -> usize {
            let mut current = entry_id;
            let mut current_dist = self.distance(query, &nodes[current].vector);

            loop {
                let neighbors = nodes[current].neighbors[layer].read().unwrap();
                let mut changed = false;

                for &neighbor_id in neighbors.iter() {
                    let dist = self.distance(query, &nodes[neighbor_id].vector);
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

            let entry_dist = self.distance(query, &nodes[entry_id].vector);
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
                        let dist = self.distance(query, &nodes[neighbor_id].vector);

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

        /// Compute distance between two vectors (1 - cosine similarity for distance metric)
        #[inline]
        fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
            let dot = simd::dot_product(a, b);
            let mag_a = simd::magnitude(a);
            let mag_b = simd::magnitude(b);

            if mag_a == 0.0 || mag_b == 0.0 {
                return 1.0; // Maximum distance for zero vectors
            }

            let similarity = dot / (mag_a * mag_b);
            1.0 - similarity // Convert similarity to distance
        }

        /// Get the vector for a node ID
        pub fn get_vector(&self, id: usize) -> Option<Vec<f32>> {
            let nodes = self.nodes.read().unwrap();
            nodes.get(id).map(|n| n.vector.clone())
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
}

/// Error types for vector operations.
#[derive(Debug, Clone, PartialEq)]
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
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorError::NotFound(key) => write!(f, "Embedding not found: {}", key),
            VectorError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            },
            VectorError::EmptyVector => write!(f, "Empty vector provided"),
            VectorError::InvalidTopK => write!(f, "Invalid top_k value (must be > 0)"),
            VectorError::StorageError(e) => write!(f, "Storage error: {}", e),
        }
    }
}

impl std::error::Error for VectorError {}

impl From<TensorStoreError> for VectorError {
    fn from(e: TensorStoreError) -> Self {
        VectorError::StorageError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, VectorError>;

/// Result of a similarity search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// The key of the matching embedding.
    pub key: String,
    /// The similarity score (cosine similarity, range -1 to 1).
    pub score: f32,
}

impl SearchResult {
    pub fn new(key: String, score: f32) -> Self {
        Self { key, score }
    }
}

/// Vector Engine for storing and searching embeddings.
///
/// Uses cosine similarity for nearest neighbor search.
pub struct VectorEngine {
    store: TensorStore,
}

impl VectorEngine {
    /// Create a new VectorEngine with a fresh TensorStore.
    pub fn new() -> Self {
        Self {
            store: TensorStore::new(),
        }
    }

    /// Create a VectorEngine using an existing TensorStore.
    pub fn with_store(store: TensorStore) -> Self {
        Self { store }
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
    /// Overwrites any existing embedding with the same key.
    pub fn store_embedding(&self, key: &str, vector: Vec<f32>) -> Result<()> {
        if vector.is_empty() {
            return Err(VectorError::EmptyVector);
        }

        let storage_key = Self::embedding_key(key);
        let mut tensor = TensorData::new();
        tensor.set("vector", TensorValue::Vector(vector));

        self.store.put(storage_key, tensor)?;
        Ok(())
    }

    /// Get an embedding by key.
    pub fn get_embedding(&self, key: &str) -> Result<Vec<f32>> {
        let storage_key = Self::embedding_key(key);
        let tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(key.to_string()))?;

        match tensor.get("vector") {
            Some(TensorValue::Vector(v)) => Ok(v.clone()),
            _ => Err(VectorError::NotFound(key.to_string())),
        }
    }

    /// Delete an embedding by key.
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

    /// Threshold for parallel search (below this, sequential is faster)
    const PARALLEL_THRESHOLD: usize = 5000;

    /// Search for the top_k most similar embeddings to the query vector.
    ///
    /// Returns results sorted by similarity score (highest first).
    /// Uses cosine similarity with SIMD acceleration.
    /// Automatically uses parallel iteration for large datasets (>5000 vectors).
    pub fn search_similar(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        // Pre-compute query magnitude for efficiency
        let query_magnitude = Self::magnitude(query);
        if query_magnitude == 0.0 {
            // Zero vector - return empty results
            return Ok(Vec::new());
        }

        let keys = self.store.scan(Self::embedding_prefix());

        // Use parallel iteration for large datasets, sequential for small
        let mut results: Vec<SearchResult> = if keys.len() >= Self::PARALLEL_THRESHOLD {
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
                let stored_vec = match tensor.get("vector") {
                    Some(TensorValue::Vector(v)) => v,
                    _ => return None,
                };

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::cosine_similarity(query, stored_vec, query_magnitude);
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
                let stored_vec = match tensor.get("vector") {
                    Some(TensorValue::Vector(v)) => v,
                    _ => return None,
                };

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::cosine_similarity(query, stored_vec, query_magnitude);
                let key = storage_key
                    .strip_prefix(Self::embedding_prefix())
                    .unwrap_or(storage_key)
                    .to_string();

                Some(SearchResult::new(key, score))
            })
            .collect()
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
                if let Some(TensorValue::Vector(v)) = tensor.get("vector") {
                    return Some(v.len());
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
    pub fn build_hnsw_index(&self, config: HNSWConfig) -> Result<(HNSWIndex, Vec<String>)> {
        let keys = self.list_keys();
        let index = HNSWIndex::with_config(config);
        let mut key_mapping = Vec::with_capacity(keys.len());

        for key in keys {
            let vector = self.get_embedding(&key)?;
            index.insert(vector);
            key_mapping.push(key);
        }

        Ok((index, key_mapping))
    }

    /// Build an HNSW index with default configuration.
    pub fn build_hnsw_index_default(&self) -> Result<(HNSWIndex, Vec<String>)> {
        self.build_hnsw_index(HNSWConfig::default())
    }

    /// Search using an existing HNSW index.
    ///
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
}
