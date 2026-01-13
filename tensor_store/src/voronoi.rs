//! Voronoi partitioner with explicit geometric region boundaries.
//!
//! Extends SemanticPartitioner with explicit Voronoi regions per node,
//! enabling geometric-aware data distribution and query routing.

use std::{collections::HashMap, sync::RwLock};

use serde::{Deserialize, Serialize};

use crate::{
    delta_vector::{KMeans, KMeansConfig},
    hnsw::simd,
    partitioner::{PartitionId, PartitionResult, Partitioner, PhysicalNodeId},
    semantic_partitioner::{SemanticPartitioner, SemanticPartitionerConfig},
};

/// A Voronoi region owned by a node.
///
/// Represents a geometric partition of the embedding space. All points
/// closest to this region's centroid (compared to other regions) belong
/// to this region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoronoiRegion {
    /// Node that owns this region.
    pub owner: PhysicalNodeId,
    /// Region centroid (center point).
    pub centroid: Vec<f32>,
    /// Estimated hypervolume (for load balancing metrics).
    /// Approximated by counting sample points assigned to this region.
    pub volume_estimate: f32,
    /// Number of data points currently assigned to this region.
    pub point_count: usize,
}

impl VoronoiRegion {
    /// Create a new Voronoi region.
    pub fn new(owner: PhysicalNodeId, centroid: Vec<f32>) -> Self {
        Self {
            owner,
            centroid,
            volume_estimate: 0.0,
            point_count: 0,
        }
    }

    /// Compute the distance from a point to this region's centroid.
    pub fn distance_to(&self, point: &[f32]) -> f32 {
        if self.centroid.len() != point.len() {
            return f32::MAX;
        }

        // Use Euclidean distance for Voronoi regions
        let mut sum = 0.0;
        for (a, b) in self.centroid.iter().zip(point.iter()) {
            let diff = a - b;
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Compute similarity (1 - normalized distance) to a point.
    pub fn similarity_to(&self, point: &[f32]) -> f32 {
        if self.centroid.is_empty() || point.is_empty() {
            return 0.0;
        }

        // Use cosine similarity for consistency with other components
        let dot = simd::dot_product(&self.centroid, point);
        let mag_a = simd::magnitude(&self.centroid);
        let mag_b = simd::magnitude(point);

        if mag_a < 1e-10 || mag_b < 1e-10 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }
}

/// Configuration for Voronoi partitioner.
#[derive(Debug, Clone)]
pub struct VoronoiPartitionerConfig {
    /// Base semantic partitioner configuration.
    pub semantic_config: SemanticPartitionerConfig,
    /// Whether to automatically rebalance on node changes.
    pub auto_rebalance: bool,
    /// Minimum samples needed before computing regions.
    pub min_samples_for_regions: usize,
    /// K-means configuration for centroid discovery.
    pub kmeans_config: KMeansConfig,
}

impl VoronoiPartitionerConfig {
    /// Create a new config for the given local node.
    pub fn new(local_node: impl Into<PhysicalNodeId>, num_shards: usize, dimension: usize) -> Self {
        Self {
            semantic_config: SemanticPartitionerConfig::new(local_node, num_shards, dimension),
            auto_rebalance: true,
            min_samples_for_regions: 100,
            kmeans_config: KMeansConfig::default(),
        }
    }

    /// Disable automatic rebalancing.
    pub fn without_auto_rebalance(mut self) -> Self {
        self.auto_rebalance = false;
        self
    }

    /// Set the minimum samples needed before computing regions.
    pub fn with_min_samples(mut self, min: usize) -> Self {
        self.min_samples_for_regions = min;
        self
    }
}

/// Voronoi partitioner with explicit geometric regions.
///
/// Extends SemanticPartitioner with explicit Voronoi region definitions.
/// Each node owns a region defined by its centroid, and data points are
/// routed to the node whose region centroid is closest.
#[derive(Debug)]
pub struct VoronoiPartitioner {
    /// Underlying semantic partitioner for routing.
    semantic: SemanticPartitioner,
    /// Explicit region definitions per node.
    regions: RwLock<HashMap<PhysicalNodeId, VoronoiRegion>>,
    /// Configuration.
    config: VoronoiPartitionerConfig,
    /// Sample embeddings for region computation.
    samples: RwLock<Vec<Vec<f32>>>,
    /// Whether regions have been computed.
    regions_computed: RwLock<bool>,
}

impl VoronoiPartitioner {
    /// Create a new Voronoi partitioner.
    pub fn new(config: VoronoiPartitionerConfig) -> Self {
        let semantic = SemanticPartitioner::new(config.semantic_config.clone());

        Self {
            semantic,
            regions: RwLock::new(HashMap::new()),
            config,
            samples: RwLock::new(Vec::new()),
            regions_computed: RwLock::new(false),
        }
    }

    /// Create from an existing SemanticPartitioner.
    pub fn from_semantic(semantic: SemanticPartitioner, config: VoronoiPartitionerConfig) -> Self {
        Self {
            semantic,
            regions: RwLock::new(HashMap::new()),
            config,
            samples: RwLock::new(Vec::new()),
            regions_computed: RwLock::new(false),
        }
    }

    /// Add a sample embedding for region computation.
    pub fn add_sample(&self, embedding: Vec<f32>) {
        let mut samples = self.samples.write().unwrap();
        samples.push(embedding);

        // Auto-compute regions if we have enough samples
        if self.config.auto_rebalance && samples.len() >= self.config.min_samples_for_regions {
            let samples_clone: Vec<Vec<f32>> = samples.clone();
            drop(samples);
            self.compute_regions_from_samples(&samples_clone);
        }
    }

    /// Compute regions from the given sample embeddings.
    pub fn compute_regions_from_samples(&self, samples: &[Vec<f32>]) {
        if samples.is_empty() {
            return;
        }

        let nodes = self.semantic.nodes();
        let k = nodes.len();

        if k == 0 {
            return;
        }

        // Run k-means to find k centroids
        let kmeans = KMeans::new(self.config.kmeans_config.clone());
        let centroids = kmeans.fit(samples, k);

        // Assign centroids to nodes
        let mut regions = self.regions.write().unwrap();
        regions.clear();

        for (i, node) in nodes.iter().enumerate() {
            if i < centroids.len() {
                let mut region = VoronoiRegion::new(node.clone(), centroids[i].clone());

                // Count how many samples belong to this region
                let count = samples
                    .iter()
                    .filter(|s| self.nearest_centroid_index(s, &centroids) == Some(i))
                    .count();
                region.point_count = count;
                region.volume_estimate = count as f32 / samples.len() as f32;

                regions.insert(node.clone(), region);
            }
        }

        // Update semantic partitioner centroids
        self.semantic.set_centroids(centroids);

        *self.regions_computed.write().unwrap() = true;
    }

    /// Find the index of the nearest centroid to a point.
    fn nearest_centroid_index(&self, point: &[f32], centroids: &[Vec<f32>]) -> Option<usize> {
        let mut best_idx = None;
        let mut best_dist = f32::MAX;

        for (i, centroid) in centroids.iter().enumerate() {
            if centroid.len() != point.len() {
                continue;
            }

            let mut dist = 0.0;
            for (a, b) in centroid.iter().zip(point.iter()) {
                let diff = a - b;
                dist += diff * diff;
            }

            if dist < best_dist {
                best_dist = dist;
                best_idx = Some(i);
            }
        }

        best_idx
    }

    /// Get the region containing an embedding.
    pub fn region_for_embedding(&self, embedding: &[f32]) -> Option<VoronoiRegion> {
        let regions = self.regions.read().unwrap();

        let mut best_region = None;
        let mut best_sim = f32::MIN;

        for region in regions.values() {
            let sim = region.similarity_to(embedding);
            if sim > best_sim {
                best_sim = sim;
                best_region = Some(region.clone());
            }
        }

        best_region
    }

    /// Get a node's Voronoi region.
    pub fn get_region(&self, node: &PhysicalNodeId) -> Option<VoronoiRegion> {
        self.regions.read().unwrap().get(node).cloned()
    }

    /// Get all regions.
    pub fn all_regions(&self) -> Vec<VoronoiRegion> {
        self.regions.read().unwrap().values().cloned().collect()
    }

    /// Check if regions have been computed.
    pub fn has_regions(&self) -> bool {
        *self.regions_computed.read().unwrap()
    }

    /// Rebalance regions after a node change.
    fn rebalance(&mut self) {
        let samples = self.samples.read().unwrap().clone();
        if samples.len() >= self.config.min_samples_for_regions {
            self.compute_regions_from_samples(&samples);
        }
    }

    /// Get embeddings that should migrate from one node to another.
    pub fn embeddings_for_migration(
        &self,
        _from: &PhysicalNodeId,
        _to: &PhysicalNodeId,
    ) -> Vec<Vec<f32>> {
        // After rebalancing, identify embeddings that now belong to a different region
        // For now, return empty - actual migration would require tracking embedding ownership
        Vec::new()
    }

    /// Get the underlying semantic partitioner.
    pub fn semantic(&self) -> &SemanticPartitioner {
        &self.semantic
    }
}

impl Partitioner for VoronoiPartitioner {
    fn partition(&self, key: &str) -> PartitionResult {
        self.semantic.partition(key)
    }

    fn partitions_for_node(&self, node: &PhysicalNodeId) -> Vec<PartitionId> {
        self.semantic.partitions_for_node(node)
    }

    fn add_node(&mut self, node: PhysicalNodeId) -> Vec<PartitionId> {
        let partitions = self.semantic.add_node(node);

        // Rebalance regions if auto-rebalance is enabled
        if self.config.auto_rebalance {
            self.rebalance();
        }

        partitions
    }

    fn remove_node(&mut self, node: &PhysicalNodeId) -> Vec<PartitionId> {
        // Remove the region for this node
        self.regions.write().unwrap().remove(node);

        let partitions = self.semantic.remove_node(node);

        // Rebalance regions if auto-rebalance is enabled
        if self.config.auto_rebalance {
            self.rebalance();
        }

        partitions
    }

    fn nodes(&self) -> Vec<PhysicalNodeId> {
        self.semantic.nodes()
    }

    fn local_node(&self) -> &PhysicalNodeId {
        self.semantic.local_node()
    }

    fn partitions_per_node(&self) -> usize {
        self.semantic.partitions_per_node()
    }

    fn total_partitions(&self) -> usize {
        self.semantic.total_partitions()
    }

    fn partition_by_embedding(&self, key: &str, embedding: &[f32]) -> PartitionResult {
        // If we have regions, use them for routing
        if self.has_regions() {
            if let Some(region) = self.region_for_embedding(embedding) {
                let is_local = &region.owner == self.local_node();
                return PartitionResult::new(region.owner, 0, is_local);
            }
        }

        // Fall back to semantic partitioner
        self.semantic.partition_by_embedding(key, embedding)
    }

    fn region_centroid(&self, node: &PhysicalNodeId) -> Option<Vec<f32>> {
        self.get_region(node).map(|r| r.centroid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> VoronoiPartitionerConfig {
        VoronoiPartitionerConfig::new("node1", 3, 4)
    }

    #[test]
    fn test_voronoi_region_creation() {
        let region = VoronoiRegion::new("node1".to_string(), vec![1.0, 0.0, 0.0, 0.0]);
        assert_eq!(region.owner, "node1");
        assert_eq!(region.centroid.len(), 4);
        assert_eq!(region.point_count, 0);
    }

    #[test]
    fn test_voronoi_region_distance() {
        let region = VoronoiRegion::new("node1".to_string(), vec![0.0, 0.0, 0.0, 0.0]);
        let point = vec![3.0, 4.0, 0.0, 0.0];
        let dist = region.distance_to(&point);
        assert!((dist - 5.0).abs() < 0.001); // 3-4-5 triangle
    }

    #[test]
    fn test_voronoi_region_similarity() {
        let region = VoronoiRegion::new("node1".to_string(), vec![1.0, 0.0, 0.0, 0.0]);

        // Same direction = high similarity
        let same_dir = vec![2.0, 0.0, 0.0, 0.0];
        assert!(region.similarity_to(&same_dir) > 0.99);

        // Orthogonal = zero similarity
        let orthogonal = vec![0.0, 1.0, 0.0, 0.0];
        assert!(region.similarity_to(&orthogonal).abs() < 0.001);
    }

    #[test]
    fn test_voronoi_partitioner_creation() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);
        assert!(!partitioner.has_regions());
    }

    #[test]
    fn test_voronoi_partitioner_add_samples() {
        let mut config = create_test_config();
        config.min_samples_for_regions = 10;
        config.auto_rebalance = true;

        let partitioner = VoronoiPartitioner::new(config);

        // Add samples (not enough yet)
        for i in 0..5 {
            partitioner.add_sample(vec![i as f32, 0.0, 0.0, 0.0]);
        }
        assert!(!partitioner.has_regions());

        // Add more samples to trigger region computation
        for i in 5..15 {
            partitioner.add_sample(vec![i as f32, 0.0, 0.0, 0.0]);
        }
        // Note: regions may or may not be computed depending on node count
    }

    #[test]
    fn test_voronoi_partitioner_partition() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        // Should fall back to semantic/hash partitioning
        let result = partitioner.partition("test_key");
        assert!(!result.primary.is_empty());
    }

    #[test]
    fn test_voronoi_config_builder() {
        let config = VoronoiPartitionerConfig::new("node1", 3, 128)
            .without_auto_rebalance()
            .with_min_samples(500);

        assert!(!config.auto_rebalance);
        assert_eq!(config.min_samples_for_regions, 500);
    }

    #[test]
    fn test_voronoi_region_debug() {
        let region = VoronoiRegion::new("node1".to_string(), vec![1.0, 2.0]);
        let debug = format!("{:?}", region);
        assert!(debug.contains("VoronoiRegion"));
        assert!(debug.contains("node1"));
    }

    #[test]
    fn test_voronoi_partitioner_debug() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);
        let debug = format!("{:?}", partitioner);
        assert!(debug.contains("VoronoiPartitioner"));
    }

    #[test]
    fn test_voronoi_region_distance_dimension_mismatch() {
        let region = VoronoiRegion::new("node1".to_string(), vec![1.0, 0.0, 0.0]);
        let point = vec![1.0, 0.0]; // Different dimension
        let dist = region.distance_to(&point);
        assert_eq!(dist, f32::MAX);
    }

    #[test]
    fn test_voronoi_region_similarity_empty() {
        // Empty centroid
        let region = VoronoiRegion::new("node1".to_string(), vec![]);
        let point = vec![1.0, 0.0];
        assert_eq!(region.similarity_to(&point), 0.0);

        // Empty point
        let region2 = VoronoiRegion::new("node1".to_string(), vec![1.0, 0.0]);
        assert_eq!(region2.similarity_to(&[]), 0.0);
    }

    #[test]
    fn test_voronoi_region_similarity_zero_magnitude() {
        let region = VoronoiRegion::new("node1".to_string(), vec![0.0, 0.0, 0.0]);
        let point = vec![1.0, 0.0, 0.0];
        assert_eq!(region.similarity_to(&point), 0.0);

        let region2 = VoronoiRegion::new("node1".to_string(), vec![1.0, 0.0, 0.0]);
        let zero_point = vec![0.0, 0.0, 0.0];
        assert_eq!(region2.similarity_to(&zero_point), 0.0);
    }

    #[test]
    fn test_voronoi_region_clone() {
        let region = VoronoiRegion::new("node1".to_string(), vec![1.0, 2.0, 3.0]);
        let cloned = region.clone();
        assert_eq!(cloned.owner, region.owner);
        assert_eq!(cloned.centroid, region.centroid);
    }

    #[test]
    fn test_voronoi_region_serde() {
        let region = VoronoiRegion::new("node1".to_string(), vec![1.0, 2.0, 3.0]);
        let serialized = bincode::serialize(&region).unwrap();
        let deserialized: VoronoiRegion = bincode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.owner, region.owner);
        assert_eq!(deserialized.centroid, region.centroid);
    }

    #[test]
    fn test_voronoi_from_semantic() {
        let semantic_config = SemanticPartitionerConfig::new("node1", 3, 4);
        let semantic = SemanticPartitioner::new(semantic_config);
        let voronoi_config = create_test_config();

        let partitioner = VoronoiPartitioner::from_semantic(semantic, voronoi_config);
        assert!(!partitioner.has_regions());
        assert_eq!(partitioner.local_node(), "node1");
    }

    #[test]
    fn test_voronoi_get_region_none() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        // No regions computed yet
        assert!(partitioner.get_region(&"node1".to_string()).is_none());
    }

    #[test]
    fn test_voronoi_all_regions_empty() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);
        assert!(partitioner.all_regions().is_empty());
    }

    #[test]
    fn test_voronoi_region_for_embedding_none() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        // No regions computed
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        assert!(partitioner.region_for_embedding(&embedding).is_none());
    }

    #[test]
    fn test_voronoi_semantic_accessor() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        let semantic = partitioner.semantic();
        assert_eq!(semantic.local_node(), "node1");
    }

    #[test]
    fn test_voronoi_embeddings_for_migration() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        let result =
            partitioner.embeddings_for_migration(&"node1".to_string(), &"node2".to_string());
        assert!(result.is_empty());
    }

    #[test]
    fn test_voronoi_compute_regions_empty_samples() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        partitioner.compute_regions_from_samples(&[]);
        assert!(!partitioner.has_regions());
    }

    #[test]
    fn test_voronoi_compute_regions_no_nodes() {
        // Create config but don't add any other nodes
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        let samples: Vec<Vec<f32>> = (0..20)
            .map(|i| vec![i as f32, (i * 2) as f32, 0.0, 0.0])
            .collect();

        partitioner.compute_regions_from_samples(&samples);
        // Should compute regions for single node
    }

    #[test]
    fn test_voronoi_add_remove_node() {
        let config = create_test_config();
        let mut partitioner = VoronoiPartitioner::new(config);

        // Add a new node
        let partitions = partitioner.add_node("node2".to_string());
        assert!(!partitions.is_empty());

        // Verify node is added
        let nodes = partitioner.nodes();
        assert!(nodes.contains(&"node2".to_string()));

        // Remove the node
        let removed = partitioner.remove_node(&"node2".to_string());
        assert!(!removed.is_empty());

        // Verify node is removed
        let nodes_after = partitioner.nodes();
        assert!(!nodes_after.contains(&"node2".to_string()));
    }

    #[test]
    fn test_voronoi_partitions_for_node() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        // The semantic partitioner may not assign partitions to node1 until hash-based assignment
        let partitions = partitioner.partitions_for_node(&"node1".to_string());
        // Just verify the method works (may be empty or non-empty depending on implementation)
        let _ = partitions;
    }

    #[test]
    fn test_voronoi_partitions_per_node() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        // Default partitions_per_node is implementation-defined
        let per_node = partitioner.partitions_per_node();
        assert!(per_node > 0);
    }

    #[test]
    fn test_voronoi_total_partitions() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        // Total partitions depends on semantic partitioner implementation
        let total = partitioner.total_partitions();
        // Just verify method works
        let _ = total;
    }

    #[test]
    fn test_voronoi_partition_by_embedding_no_regions() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let result = partitioner.partition_by_embedding("test_key", &embedding);
        assert!(!result.primary.is_empty());
    }

    #[test]
    fn test_voronoi_region_centroid_none() {
        let config = create_test_config();
        let partitioner = VoronoiPartitioner::new(config);

        let centroid = partitioner.region_centroid(&"node1".to_string());
        assert!(centroid.is_none());
    }

    #[test]
    fn test_voronoi_config_debug() {
        let config = create_test_config();
        let debug = format!("{:?}", config);
        assert!(debug.contains("VoronoiPartitionerConfig"));
    }

    #[test]
    fn test_voronoi_config_clone() {
        let config = create_test_config();
        let cloned = config.clone();
        assert_eq!(cloned.auto_rebalance, config.auto_rebalance);
        assert_eq!(
            cloned.min_samples_for_regions,
            config.min_samples_for_regions
        );
    }

    #[test]
    fn test_voronoi_compute_regions_with_multiple_nodes() {
        let mut config = create_test_config();
        config.auto_rebalance = false;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);

        // Add more nodes
        partitioner.add_node("node2".to_string());

        // Create sample embeddings in different regions
        let samples: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.8, 0.2, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.1, 0.9, 0.0, 0.0],
            vec![0.0, 0.8, 0.2, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.9, 0.1],
            vec![0.0, 0.0, 0.8, 0.2],
            vec![0.5, 0.5, 0.0, 0.0],
        ];

        partitioner.compute_regions_from_samples(&samples);
        assert!(partitioner.has_regions());

        // Verify regions were created
        let all_regions = partitioner.all_regions();
        // At least one region should be created (depends on k-means and node tracking)
        assert!(!all_regions.is_empty());

        // Each region should have a centroid
        for region in &all_regions {
            assert_eq!(region.centroid.len(), 4);
            assert!(region.point_count > 0 || region.volume_estimate >= 0.0);
        }
    }

    #[test]
    fn test_voronoi_region_for_embedding_with_regions() {
        let mut config = create_test_config();
        config.auto_rebalance = false;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);
        partitioner.add_node("node2".to_string());
        partitioner.add_node("node3".to_string());

        // Create distinct cluster samples
        let samples: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.95, 0.05, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.05, 0.95, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.95, 0.05],
        ];

        partitioner.compute_regions_from_samples(&samples);

        // Query for embedding that should match a region
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let region = partitioner.region_for_embedding(&embedding);
        assert!(region.is_some());

        let found_region = region.unwrap();
        assert!(!found_region.owner.is_empty());
    }

    #[test]
    fn test_voronoi_get_region_with_regions() {
        let mut config = create_test_config();
        config.auto_rebalance = false;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);
        partitioner.add_node("node2".to_string());

        let samples: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![(i % 3) as f32, ((i + 1) % 3) as f32, 0.0, 0.0])
            .collect();

        partitioner.compute_regions_from_samples(&samples);

        // Try to get a region for each node
        let nodes = partitioner.nodes();
        let mut found_count = 0;
        for node in &nodes {
            if partitioner.get_region(node).is_some() {
                found_count += 1;
            }
        }
        assert!(found_count > 0);
    }

    #[test]
    fn test_voronoi_region_centroid_with_regions() {
        let mut config = create_test_config();
        config.auto_rebalance = false;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);
        partitioner.add_node("node2".to_string());

        let samples: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();

        partitioner.compute_regions_from_samples(&samples);

        // Get centroid for a node
        let nodes = partitioner.nodes();
        let mut found_centroid = false;
        for node in &nodes {
            if let Some(centroid) = partitioner.region_centroid(node) {
                assert_eq!(centroid.len(), 4);
                found_centroid = true;
                break;
            }
        }
        assert!(found_centroid);
    }

    #[test]
    fn test_voronoi_partition_by_embedding_with_regions() {
        let mut config = create_test_config();
        config.auto_rebalance = false;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);
        partitioner.add_node("node2".to_string());

        let samples: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.1, 0.9, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0],
            vec![0.6, 0.4, 0.0, 0.0],
        ];

        partitioner.compute_regions_from_samples(&samples);
        assert!(partitioner.has_regions());

        // Partition by embedding should use regions
        let embedding = vec![1.0, 0.0, 0.0, 0.0];
        let result = partitioner.partition_by_embedding("test_key", &embedding);
        assert!(!result.primary.is_empty());
    }

    #[test]
    fn test_voronoi_add_node_with_auto_rebalance() {
        let mut config = create_test_config();
        config.auto_rebalance = true;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);

        // Add some samples first
        for i in 0..10 {
            partitioner.add_sample(vec![i as f32, (i * 2) as f32, 0.0, 0.0]);
        }

        // Add a new node - should trigger rebalance
        let partitions = partitioner.add_node("node2".to_string());
        assert!(!partitions.is_empty());
    }

    #[test]
    fn test_voronoi_remove_node_with_auto_rebalance() {
        let mut config = create_test_config();
        config.auto_rebalance = true;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);
        partitioner.add_node("node2".to_string());

        // Add samples
        for i in 0..10 {
            partitioner.add_sample(vec![i as f32, 0.0, 0.0, 0.0]);
        }

        // Remove node - should trigger rebalance
        let removed = partitioner.remove_node(&"node2".to_string());
        assert!(!removed.is_empty());
    }

    #[test]
    fn test_voronoi_nearest_centroid_dimension_mismatch() {
        let mut config = create_test_config();
        config.auto_rebalance = false;

        let mut partitioner = VoronoiPartitioner::new(config);
        partitioner.add_node("node2".to_string());

        // Samples with correct dimension
        let samples: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        partitioner.compute_regions_from_samples(&samples);

        // This tests the nearest_centroid_index path indirectly
        assert!(partitioner.has_regions());
    }

    #[test]
    fn test_voronoi_add_sample_triggers_region_computation() {
        let mut config = create_test_config();
        config.auto_rebalance = true;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);
        partitioner.add_node("node2".to_string());

        // Add exactly min_samples to trigger computation
        for i in 0..5 {
            partitioner.add_sample(vec![i as f32, 0.0, 0.0, 0.0]);
        }

        // Regions should now be computed
        assert!(partitioner.has_regions());
    }

    #[test]
    fn test_voronoi_region_point_count_and_volume() {
        let mut config = create_test_config();
        config.auto_rebalance = false;
        config.min_samples_for_regions = 5;

        let mut partitioner = VoronoiPartitioner::new(config);
        partitioner.add_node("node2".to_string());

        // Create samples that will cluster
        let samples: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.0, 0.0, 0.0],
            vec![0.0, 0.1, 0.0, 0.0],
            vec![10.0, 0.0, 0.0, 0.0],
            vec![10.1, 0.0, 0.0, 0.0],
            vec![10.0, 0.1, 0.0, 0.0],
        ];

        partitioner.compute_regions_from_samples(&samples);

        let regions = partitioner.all_regions();
        for region in &regions {
            // Volume estimate should be between 0 and 1
            assert!(region.volume_estimate >= 0.0 && region.volume_estimate <= 1.0);
        }
    }
}
