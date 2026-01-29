// SPDX-License-Identifier: MIT OR Apache-2.0
//! Semantic partitioner for embedding-based data distribution.
//!
//! Routes keys to shards based on embedding similarity using archetype centroids.
//! Data with similar embeddings is co-located on the same shard, improving
//! locality for similarity searches and reducing cross-shard queries.

use parking_lot::RwLock;

use crate::{
    consistent_hash::{ConsistentHashConfig, ConsistentHashPartitioner},
    delta_vector::{ArchetypeRegistry, KMeans, KMeansConfig},
    hnsw::simd,
    partitioner::{PartitionId, PartitionResult, Partitioner, PhysicalNodeId},
};

/// Configuration for the semantic partitioner.
#[derive(Debug, Clone)]
pub struct SemanticPartitionerConfig {
    /// Number of shards (must match physical nodes).
    pub num_shards: usize,
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Minimum similarity to route semantically (0.0-1.0).
    /// Below this threshold, falls back to consistent hashing.
    pub similarity_threshold: f32,
    /// Number of virtual nodes per physical node for fallback.
    pub virtual_nodes: usize,
    /// Local node ID.
    pub local_node: PhysicalNodeId,
    /// K-means configuration for centroid discovery.
    pub kmeans_config: KMeansConfig,
}

impl SemanticPartitionerConfig {
    /// Create a new config with the given parameters.
    pub fn new(
        local_node: impl Into<PhysicalNodeId>,
        num_shards: usize,
        embedding_dim: usize,
    ) -> Self {
        Self {
            num_shards,
            embedding_dim,
            similarity_threshold: 0.5,
            virtual_nodes: 256,
            local_node: local_node.into(),
            kmeans_config: KMeansConfig::default(),
        }
    }

    /// Sets the similarity threshold for partition assignment.
    #[must_use]
    pub const fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Sets the number of virtual nodes per shard.
    #[must_use]
    pub const fn with_virtual_nodes(mut self, count: usize) -> Self {
        self.virtual_nodes = count;
        self
    }

    /// Sets the k-means clustering configuration.
    #[must_use]
    pub const fn with_kmeans_config(mut self, config: KMeansConfig) -> Self {
        self.kmeans_config = config;
        self
    }
}

impl Default for SemanticPartitionerConfig {
    fn default() -> Self {
        Self {
            num_shards: 3,
            embedding_dim: 128,
            similarity_threshold: 0.5,
            virtual_nodes: 256,
            local_node: String::new(),
            kmeans_config: KMeansConfig::default(),
        }
    }
}

/// Routing method used for a partition decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingMethod {
    /// Routed based on embedding similarity to shard centroid.
    Semantic,
    /// Fell back to consistent hashing (no embedding or low similarity).
    ConsistentHash,
}

/// Extended partition result with routing method information.
#[derive(Debug, Clone)]
pub struct SemanticPartitionResult {
    /// Base partition result.
    pub result: PartitionResult,
    /// Method used for routing.
    pub method: RoutingMethod,
    /// Similarity to the chosen shard's centroid (for semantic routing).
    pub similarity: Option<f32>,
}

impl SemanticPartitionResult {
    /// Create a semantic routing result.
    #[must_use]
    pub const fn semantic(result: PartitionResult, similarity: f32) -> Self {
        Self {
            result,
            method: RoutingMethod::Semantic,
            similarity: Some(similarity),
        }
    }

    /// Create a consistent hash routing result.
    #[must_use]
    pub const fn consistent_hash(result: PartitionResult) -> Self {
        Self {
            result,
            method: RoutingMethod::ConsistentHash,
            similarity: None,
        }
    }
}

/// Semantic partitioner that routes data based on embedding similarity.
///
/// Uses archetype centroids to route semantically similar data to the same shard.
/// Falls back to consistent hashing for keys without embeddings or when
/// similarity is below the threshold.
#[derive(Debug)]
pub struct SemanticPartitioner {
    /// Configuration.
    config: SemanticPartitionerConfig,
    /// Shard centroids (one per shard).
    centroids: RwLock<Vec<Vec<f32>>>,
    /// Precomputed centroid magnitudes for fast similarity.
    centroid_magnitudes: RwLock<Vec<f32>>,
    /// Fallback partitioner for non-embedded keys.
    fallback: ConsistentHashPartitioner,
    /// Shard-to-node mapping.
    shard_nodes: RwLock<Vec<PhysicalNodeId>>,
}

impl SemanticPartitioner {
    /// Create a new semantic partitioner.
    #[must_use]
    pub fn new(config: SemanticPartitionerConfig) -> Self {
        let hash_config = ConsistentHashConfig::new(config.local_node.clone())
            .with_virtual_nodes(config.virtual_nodes);

        Self {
            centroids: RwLock::new(Vec::new()),
            centroid_magnitudes: RwLock::new(Vec::new()),
            fallback: ConsistentHashPartitioner::new(hash_config),
            shard_nodes: RwLock::new(Vec::new()),
            config,
        }
    }

    /// Create with initial nodes.
    #[must_use]
    pub fn with_nodes(config: SemanticPartitionerConfig, nodes: Vec<PhysicalNodeId>) -> Self {
        let mut partitioner = Self::new(config);
        for node in nodes {
            partitioner.add_node(node);
        }
        partitioner
    }

    /// Partition a key using its embedding.
    ///
    /// Routes to the shard with the most similar centroid if similarity
    /// exceeds the threshold. Falls back to consistent hashing otherwise.
    ///
    #[allow(clippy::significant_drop_tightening)] // Locks held during iteration over centroids
    pub fn partition_with_embedding(
        &self,
        key: &str,
        embedding: &[f32],
    ) -> SemanticPartitionResult {
        let centroids = self.centroids.read();

        // Fall back if no centroids or dimension mismatch
        if centroids.is_empty() || embedding.len() != self.config.embedding_dim {
            return SemanticPartitionResult::consistent_hash(self.partition(key));
        }

        // Find nearest centroid
        let magnitudes = self.centroid_magnitudes.read();
        let query_magnitude = simd::magnitude(embedding);

        if query_magnitude == 0.0 {
            return SemanticPartitionResult::consistent_hash(self.partition(key));
        }

        let mut best_shard = 0;
        let mut best_similarity = f32::NEG_INFINITY;

        for (i, centroid) in centroids.iter().enumerate() {
            let centroid_magnitude = magnitudes[i];
            if centroid_magnitude == 0.0 {
                continue;
            }

            let dot = simd::dot_product(embedding, centroid);
            let similarity = dot / (query_magnitude * centroid_magnitude);

            if similarity > best_similarity {
                best_similarity = similarity;
                best_shard = i;
            }
        }

        // Check if similarity exceeds threshold
        if best_similarity >= self.config.similarity_threshold {
            let shard_nodes = self.shard_nodes.read();
            if best_shard < shard_nodes.len() {
                let node = shard_nodes[best_shard].clone();
                let is_local = node == self.config.local_node;
                let result = PartitionResult::new(node, best_shard as PartitionId, is_local);
                return SemanticPartitionResult::semantic(result, best_similarity);
            }
        }

        // Fall back to consistent hashing
        SemanticPartitionResult::consistent_hash(self.partition(key))
    }

    /// Initialize centroids from sample embeddings using k-means.
    ///
    /// Returns the number of centroids created.
    ///
    #[allow(clippy::significant_drop_tightening)] // Locks updated atomically
    pub fn initialize_centroids(&self, samples: &[Vec<f32>]) -> usize {
        if samples.is_empty() {
            return 0;
        }

        // Filter to correct dimension
        let valid_samples: Vec<_> = samples
            .iter()
            .filter(|s| s.len() == self.config.embedding_dim)
            .cloned()
            .collect();

        if valid_samples.is_empty() {
            return 0;
        }

        // Run k-means to find centroids
        let k = self.config.num_shards.min(valid_samples.len());
        let kmeans = KMeans::new(self.config.kmeans_config.clone());
        let new_centroids = kmeans.fit(&valid_samples, k);

        // Compute magnitudes
        let new_magnitudes: Vec<f32> = new_centroids.iter().map(|c| simd::magnitude(c)).collect();

        // Update centroids
        let mut centroids = self.centroids.write();
        let mut magnitudes = self.centroid_magnitudes.write();
        *centroids = new_centroids;
        *magnitudes = new_magnitudes;

        centroids.len()
    }

    /// Set centroids directly (useful for cluster-wide synchronization).
    ///
    #[allow(clippy::significant_drop_tightening)] // Locks updated atomically
    pub fn set_centroids(&self, new_centroids: Vec<Vec<f32>>) {
        let new_magnitudes: Vec<f32> = new_centroids.iter().map(|c| simd::magnitude(c)).collect();

        let mut centroids = self.centroids.write();
        let mut magnitudes = self.centroid_magnitudes.write();
        *centroids = new_centroids;
        *magnitudes = new_magnitudes;
    }

    /// Returns a clone of the current centroids.
    pub fn centroids(&self) -> Vec<Vec<f32>> {
        self.centroids.read().clone()
    }

    /// Returns the number of centroids.
    pub fn num_centroids(&self) -> usize {
        self.centroids.read().len()
    }

    /// Returns the embedding dimension.
    pub const fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Returns the similarity threshold for partition assignment.
    pub const fn similarity_threshold(&self) -> f32 {
        self.config.similarity_threshold
    }

    /// Get shards that are semantically relevant to an embedding.
    ///
    /// Returns shards with centroid similarity above the threshold,
    /// sorted by similarity (highest first).
    ///
    #[allow(clippy::significant_drop_tightening)] // Locks held during iteration over centroids
    pub fn shards_for_embedding(&self, embedding: &[f32]) -> Vec<(usize, f32)> {
        let centroids = self.centroids.read();

        if centroids.is_empty() || embedding.len() != self.config.embedding_dim {
            return Vec::new();
        }

        let magnitudes = self.centroid_magnitudes.read();
        let query_magnitude = simd::magnitude(embedding);

        if query_magnitude == 0.0 {
            return Vec::new();
        }

        let mut relevant: Vec<(usize, f32)> = centroids
            .iter()
            .enumerate()
            .filter_map(|(i, centroid)| {
                let centroid_magnitude = magnitudes[i];
                if centroid_magnitude == 0.0 {
                    return None;
                }

                let dot = simd::dot_product(embedding, centroid);
                let similarity = dot / (query_magnitude * centroid_magnitude);

                if similarity >= self.config.similarity_threshold {
                    Some((i, similarity))
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity descending
        relevant.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        relevant
    }

    /// Returns all shard indices.
    pub fn all_shards(&self) -> Vec<usize> {
        let shard_nodes = self.shard_nodes.read();
        (0..shard_nodes.len()).collect()
    }

    /// Returns statistics about the partitioner state.
    pub fn stats(&self) -> SemanticPartitionerStats {
        let centroids = self.centroids.read();
        let shard_nodes = self.shard_nodes.read();

        SemanticPartitionerStats {
            num_shards: shard_nodes.len(),
            num_centroids: centroids.len(),
            embedding_dim: self.config.embedding_dim,
            similarity_threshold: self.config.similarity_threshold,
            fallback_stats: self.fallback.stats(),
        }
    }

    /// Encode an embedding for transfer using archetype registry.
    ///
    /// Used for delta replication - encodes embedding as archetype + delta.
    pub fn encode_for_transfer(
        &self,
        embedding: &[f32],
        registry: &ArchetypeRegistry,
        threshold: f32,
    ) -> Option<EncodedEmbedding> {
        let delta = registry.encode(embedding, threshold)?;
        Some(EncodedEmbedding {
            archetype_id: delta.archetype_id(),
            sparse_delta: delta.to_sparse_delta(),
            original_dimension: embedding.len(),
        })
    }
}

impl Partitioner for SemanticPartitioner {
    fn partition(&self, key: &str) -> PartitionResult {
        self.fallback.partition(key)
    }

    fn partitions_for_node(&self, node: &PhysicalNodeId) -> Vec<PartitionId> {
        self.fallback.partitions_for_node(node)
    }

    fn add_node(&mut self, node: PhysicalNodeId) -> Vec<PartitionId> {
        // Add to shard-node mapping
        {
            let mut shard_nodes = self.shard_nodes.write();
            shard_nodes.push(node.clone());
        }

        // Add to fallback partitioner
        self.fallback.add_node(node)
    }

    fn remove_node(&mut self, node: &PhysicalNodeId) -> Vec<PartitionId> {
        // Remove from shard-node mapping
        {
            let mut shard_nodes = self.shard_nodes.write();
            shard_nodes.retain(|n| n != node);
        }

        // Remove from fallback partitioner
        self.fallback.remove_node(node)
    }

    fn nodes(&self) -> Vec<PhysicalNodeId> {
        self.fallback.nodes()
    }

    fn local_node(&self) -> &PhysicalNodeId {
        &self.config.local_node
    }

    fn partitions_per_node(&self) -> usize {
        self.config.virtual_nodes
    }

    fn total_partitions(&self) -> usize {
        self.fallback.total_partitions()
    }
}

/// Statistics about the semantic partitioner.
#[derive(Debug, Clone)]
pub struct SemanticPartitionerStats {
    /// Number of physical shards.
    pub num_shards: usize,
    /// Number of centroids (may differ from shards during rebalancing).
    pub num_centroids: usize,
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Similarity threshold for semantic routing.
    pub similarity_threshold: f32,
    /// Stats from the fallback consistent hash partitioner.
    pub fallback_stats: crate::consistent_hash::ConsistentHashStats,
}

/// Encoded embedding for bandwidth-efficient transfer.
#[derive(Debug, Clone)]
pub struct EncodedEmbedding {
    /// ID of the reference archetype.
    pub archetype_id: usize,
    /// Sparse delta from archetype.
    pub sparse_delta: crate::SparseVector,
    /// Original embedding dimension.
    pub original_dimension: usize,
}

impl EncodedEmbedding {
    /// Decode back to dense embedding using archetype registry.
    #[must_use]
    pub fn decode(&self, registry: &ArchetypeRegistry) -> Option<Vec<f32>> {
        let archetype = registry.get(self.archetype_id)?;
        let mut result = archetype.to_vec();

        // Apply sparse delta
        for (pos, val) in self.sparse_delta.iter() {
            if (pos as usize) < result.len() {
                result[pos as usize] += val;
            }
        }

        Some(result)
    }

    /// Memory bytes used by this encoded embedding.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.sparse_delta.memory_bytes()
    }

    /// Compression ratio compared to dense embedding.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let dense_bytes = self.original_dimension * std::mem::size_of::<f32>();
        if dense_bytes == 0 {
            return 1.0;
        }
        #[allow(clippy::cast_precision_loss)]
        // Compression ratio calculation accepts precision loss
        let ratio = dense_bytes as f32 / self.memory_bytes() as f32;
        ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_config_new() {
        let config = SemanticPartitionerConfig::new("node1", 3, 128);
        assert_eq!(config.num_shards, 3);
        assert_eq!(config.embedding_dim, 128);
        assert_eq!(config.local_node, "node1");
    }

    #[test]
    fn test_config_with_methods() {
        let config = SemanticPartitionerConfig::new("node1", 3, 128)
            .with_similarity_threshold(0.7)
            .with_virtual_nodes(128);

        assert!(approx_eq(config.similarity_threshold, 0.7, 0.001));
        assert_eq!(config.virtual_nodes, 128);
    }

    #[test]
    fn test_config_threshold_clamping() {
        let config = SemanticPartitionerConfig::default().with_similarity_threshold(1.5);
        assert!(approx_eq(config.similarity_threshold, 1.0, 0.001));

        let config = SemanticPartitionerConfig::default().with_similarity_threshold(-0.5);
        assert!(approx_eq(config.similarity_threshold, 0.0, 0.001));
    }

    #[test]
    fn test_config_default() {
        let config = SemanticPartitionerConfig::default();
        assert_eq!(config.num_shards, 3);
        assert_eq!(config.embedding_dim, 128);
        assert!(approx_eq(config.similarity_threshold, 0.5, 0.001));
    }

    #[test]
    fn test_partitioner_empty() {
        let config = SemanticPartitionerConfig::new("local", 3, 4);
        let partitioner = SemanticPartitioner::new(config);

        // No centroids - should fall back to consistent hash
        let result = partitioner.partition_with_embedding("key", &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(result.method, RoutingMethod::ConsistentHash);
    }

    #[test]
    fn test_partitioner_with_nodes() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let partitioner = SemanticPartitioner::with_nodes(
            config,
            vec![
                "node1".to_string(),
                "node2".to_string(),
                "node3".to_string(),
            ],
        );

        assert_eq!(partitioner.nodes().len(), 3);
    }

    #[test]
    fn test_initialize_centroids() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let mut partitioner = SemanticPartitioner::new(config);
        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());
        partitioner.add_node("node3".to_string());

        // Create sample embeddings in 3 clusters
        let samples = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.1, 0.9, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.9, 0.1],
        ];

        let num_centroids = partitioner.initialize_centroids(&samples);
        assert!(num_centroids > 0);
        assert!(num_centroids <= 3);
    }

    #[test]
    fn test_semantic_routing() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4).with_similarity_threshold(0.5);
        let mut partitioner = SemanticPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());
        partitioner.add_node("node3".to_string());

        // Set explicit centroids for each shard
        partitioner.set_centroids(vec![
            vec![1.0, 0.0, 0.0, 0.0], // Shard 0
            vec![0.0, 1.0, 0.0, 0.0], // Shard 1
            vec![0.0, 0.0, 1.0, 0.0], // Shard 2
        ]);

        // Embedding close to shard 0's centroid
        let result = partitioner.partition_with_embedding("key1", &[0.95, 0.05, 0.0, 0.0]);
        assert_eq!(result.method, RoutingMethod::Semantic);
        assert!(result.similarity.unwrap() > 0.9);

        // Embedding close to shard 2's centroid
        let result = partitioner.partition_with_embedding("key2", &[0.0, 0.0, 0.98, 0.02]);
        assert_eq!(result.method, RoutingMethod::Semantic);
    }

    #[test]
    fn test_fallback_on_low_similarity() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4).with_similarity_threshold(0.9); // High threshold
        let mut partitioner = SemanticPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        partitioner.set_centroids(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]);

        // Embedding not very similar to any centroid
        let result = partitioner.partition_with_embedding("key", &[0.5, 0.5, 0.5, 0.5]);
        assert_eq!(result.method, RoutingMethod::ConsistentHash);
    }

    #[test]
    fn test_shards_for_embedding() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4).with_similarity_threshold(0.3);
        let partitioner = SemanticPartitioner::new(config);

        partitioner.set_centroids(vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ]);

        // Embedding similar to shard 0 and slightly to shard 1
        let relevant = partitioner.shards_for_embedding(&[0.8, 0.2, 0.0, 0.0]);

        // Should include shard 0 with highest similarity
        assert!(!relevant.is_empty());
        assert_eq!(relevant[0].0, 0); // First one should be shard 0
    }

    #[test]
    fn test_shards_for_embedding_empty() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let partitioner = SemanticPartitioner::new(config);

        // No centroids
        let relevant = partitioner.shards_for_embedding(&[1.0, 0.0, 0.0, 0.0]);
        assert!(relevant.is_empty());
    }

    #[test]
    fn test_all_shards() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let mut partitioner = SemanticPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());
        partitioner.add_node("node3".to_string());

        let shards = partitioner.all_shards();
        assert_eq!(shards, vec![0, 1, 2]);
    }

    #[test]
    fn test_stats() {
        let config = SemanticPartitionerConfig::new("node1", 3, 128).with_similarity_threshold(0.6);
        let mut partitioner = SemanticPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        let stats = partitioner.stats();
        assert_eq!(stats.num_shards, 2);
        assert_eq!(stats.embedding_dim, 128);
        assert!(approx_eq(stats.similarity_threshold, 0.6, 0.001));
    }

    #[test]
    fn test_deterministic_routing() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let mut partitioner = SemanticPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        partitioner.set_centroids(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]);

        let embedding = vec![0.8, 0.2, 0.0, 0.0];

        let result1 = partitioner.partition_with_embedding("key", &embedding);
        let result2 = partitioner.partition_with_embedding("key", &embedding);

        assert_eq!(result1.result.primary, result2.result.primary);
        assert_eq!(result1.result.partition, result2.result.partition);
    }

    #[test]
    fn test_add_remove_node() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let mut partitioner = SemanticPartitioner::new(config);

        let added = partitioner.add_node("node1".to_string());
        assert!(!added.is_empty());
        assert_eq!(partitioner.nodes().len(), 1);

        partitioner.add_node("node2".to_string());
        assert_eq!(partitioner.nodes().len(), 2);

        let removed = partitioner.remove_node(&"node2".to_string());
        assert!(!removed.is_empty());
        assert_eq!(partitioner.nodes().len(), 1);
    }

    #[test]
    fn test_partitioner_trait_methods() {
        let config = SemanticPartitionerConfig::new("local", 3, 4).with_virtual_nodes(10);
        let mut partitioner = SemanticPartitioner::new(config);

        partitioner.add_node("local".to_string());

        assert_eq!(partitioner.local_node(), "local");
        assert_eq!(partitioner.partitions_per_node(), 10);
        assert_eq!(partitioner.total_partitions(), 10);

        let partitions = partitioner.partitions_for_node(&"local".to_string());
        assert_eq!(partitions.len(), 10);
    }

    #[test]
    fn test_dimension_mismatch_fallback() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let mut partitioner = SemanticPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.set_centroids(vec![vec![1.0, 0.0, 0.0, 0.0]]);

        // Wrong dimension embedding
        let result = partitioner.partition_with_embedding("key", &[1.0, 0.0]);
        assert_eq!(result.method, RoutingMethod::ConsistentHash);
    }

    #[test]
    fn test_zero_magnitude_embedding() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let mut partitioner = SemanticPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.set_centroids(vec![vec![1.0, 0.0, 0.0, 0.0]]);

        // Zero vector
        let result = partitioner.partition_with_embedding("key", &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(result.method, RoutingMethod::ConsistentHash);
    }

    #[test]
    fn test_get_centroids() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let partitioner = SemanticPartitioner::new(config);

        let centroids = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        partitioner.set_centroids(centroids.clone());

        assert_eq!(partitioner.centroids(), centroids);
        assert_eq!(partitioner.num_centroids(), 2);
    }

    #[test]
    fn test_encoded_embedding() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let partitioner = SemanticPartitioner::new(config);

        let embedding = vec![0.9, 0.1, 0.0, 0.0];
        let encoded = partitioner
            .encode_for_transfer(&embedding, &registry, 0.001)
            .unwrap();

        assert_eq!(encoded.archetype_id, 0);
        assert_eq!(encoded.original_dimension, 4);

        // Decode and verify
        let decoded = encoded.decode(&registry).unwrap();
        for (orig, dec) in embedding.iter().zip(decoded.iter()) {
            assert!(approx_eq(*orig, *dec, 0.01));
        }
    }

    #[test]
    fn test_encoded_embedding_compression() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0; 128]).unwrap();

        let config = SemanticPartitionerConfig::new("node1", 3, 128);
        let partitioner = SemanticPartitioner::new(config);

        // Embedding very similar to archetype
        let mut embedding = vec![1.0; 128];
        embedding[0] = 0.9;

        let encoded = partitioner
            .encode_for_transfer(&embedding, &registry, 0.01)
            .unwrap();

        // Should have good compression
        assert!(encoded.compression_ratio() > 1.0);
    }

    #[test]
    fn test_semantic_partition_result() {
        let result = PartitionResult::local("node1", 42);

        let semantic = SemanticPartitionResult::semantic(result.clone(), 0.95);
        assert_eq!(semantic.method, RoutingMethod::Semantic);
        assert!(approx_eq(semantic.similarity.unwrap(), 0.95, 0.001));

        let hash = SemanticPartitionResult::consistent_hash(result);
        assert_eq!(hash.method, RoutingMethod::ConsistentHash);
        assert!(hash.similarity.is_none());
    }

    #[test]
    fn test_initialize_centroids_filters_invalid() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let mut partitioner = SemanticPartitioner::new(config);
        partitioner.add_node("node1".to_string());

        // Mix of valid and invalid dimensions
        let samples = vec![
            vec![1.0, 0.0, 0.0, 0.0], // Valid
            vec![1.0, 0.0],           // Invalid
            vec![0.0, 1.0, 0.0, 0.0], // Valid
        ];

        let num = partitioner.initialize_centroids(&samples);
        assert!(num > 0);
    }

    #[test]
    fn test_initialize_centroids_empty() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let partitioner = SemanticPartitioner::new(config);

        let num = partitioner.initialize_centroids(&[]);
        assert_eq!(num, 0);
    }

    #[test]
    fn test_config_with_kmeans_config() {
        let kmeans_config = KMeansConfig {
            max_iterations: 50,
            ..Default::default()
        };

        let config = SemanticPartitionerConfig::default().with_kmeans_config(kmeans_config);

        assert_eq!(config.kmeans_config.max_iterations, 50);
    }

    #[test]
    fn test_routing_method_equality() {
        assert_eq!(RoutingMethod::Semantic, RoutingMethod::Semantic);
        assert_eq!(RoutingMethod::ConsistentHash, RoutingMethod::ConsistentHash);
        assert_ne!(RoutingMethod::Semantic, RoutingMethod::ConsistentHash);
    }

    #[test]
    fn test_stats_debug() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let partitioner = SemanticPartitioner::new(config);
        let stats = partitioner.stats();

        let debug = format!("{:?}", stats);
        assert!(debug.contains("SemanticPartitionerStats"));
    }

    #[test]
    fn test_zero_centroid_magnitude_skipped() {
        let config = SemanticPartitionerConfig::new("node1", 3, 4);
        let mut partitioner = SemanticPartitioner::new(config);
        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        // One centroid is zero vector
        partitioner.set_centroids(vec![
            vec![0.0, 0.0, 0.0, 0.0], // Zero magnitude
            vec![1.0, 0.0, 0.0, 0.0], // Valid
        ]);

        let result = partitioner.partition_with_embedding("key", &[0.9, 0.1, 0.0, 0.0]);
        // Should route to shard 1 (the valid one)
        if result.method == RoutingMethod::Semantic {
            assert_eq!(result.result.partition, 1);
        }
    }
}
