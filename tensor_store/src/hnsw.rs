//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! This module provides a shared HNSW implementation used by both VectorEngine
//! and TensorCache for efficient similarity search.
//!
//! Key features:
//! - O(log n) search complexity vs O(n) for brute force
//! - 5-8x speedup for 10K-100K vectors
//! - Configurable recall/speed tradeoff via ef_search parameter

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
                let mut neighbor_neighbors = nodes[neighbor_id].neighbors[layer].write().unwrap();
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
}
