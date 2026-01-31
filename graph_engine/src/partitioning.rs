//! Graph partitioning strategies for distributed graph operations.
//!
//! Provides different strategies for assigning graph nodes to shards
//! in a distributed graph engine.

#![allow(clippy::module_name_repetitions)]

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// Shard identifier.
pub type ShardId = u32;

/// Graph partitioning strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Simple hash-based partitioning: `hash(node_id) % num_shards`.
    /// Fast and uniform distribution, but may create many cross-shard edges.
    #[default]
    HashBased,
    /// Range-based partitioning: `node_id` ranges assigned to shards.
    /// Good for sequential inserts, may become unbalanced over time.
    RangeBased,
    /// Modular partitioning with custom modulus.
    /// Allows fine-grained control over distribution.
    Modular,
}

/// Configuration for graph partitioning.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Number of shards in the cluster.
    pub num_shards: u32,
    /// Partitioning strategy to use.
    pub strategy: PartitionStrategy,
    /// For range-based: explicit ranges per shard.
    pub ranges: Option<Vec<(u64, u64)>>,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            num_shards: 1,
            strategy: PartitionStrategy::HashBased,
            ranges: None,
        }
    }
}

impl PartitionConfig {
    #[must_use]
    pub const fn new(num_shards: u32) -> Self {
        Self {
            num_shards,
            strategy: PartitionStrategy::HashBased,
            ranges: None,
        }
    }

    #[must_use]
    pub const fn with_strategy(mut self, strategy: PartitionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    #[must_use]
    pub fn with_ranges(mut self, ranges: Vec<(u64, u64)>) -> Self {
        self.ranges = Some(ranges);
        self.strategy = PartitionStrategy::RangeBased;
        self
    }
}

/// Graph partitioner that assigns nodes to shards.
#[derive(Debug, Clone)]
pub struct GraphPartitioner {
    config: PartitionConfig,
    /// Cache of node -> shard assignments for consistency.
    assignments: HashMap<u64, ShardId>,
}

impl GraphPartitioner {
    #[must_use]
    pub fn new(config: PartitionConfig) -> Self {
        Self {
            config,
            assignments: HashMap::new(),
        }
    }

    /// Get the shard for a node ID.
    #[must_use]
    pub fn shard_for_node(&self, node_id: u64) -> ShardId {
        // Check cache first
        if let Some(&shard) = self.assignments.get(&node_id) {
            return shard;
        }

        match self.config.strategy {
            PartitionStrategy::HashBased => self.hash_partition(node_id),
            PartitionStrategy::RangeBased => self.range_partition(node_id),
            PartitionStrategy::Modular => self.modular_partition(node_id),
        }
    }

    /// Assign a node to a specific shard (for rebalancing).
    pub fn assign_node(&mut self, node_id: u64, shard: ShardId) {
        self.assignments.insert(node_id, shard);
    }

    /// Get all shards that might contain nodes connected to a given node.
    /// For cross-shard traversals, we may need to query multiple shards.
    #[must_use]
    pub fn all_shards(&self) -> Vec<ShardId> {
        (0..self.config.num_shards).collect()
    }

    /// Get the number of shards.
    #[must_use]
    pub const fn num_shards(&self) -> u32 {
        self.config.num_shards
    }

    /// Get shards involved in an edge (for cross-shard edge tracking).
    #[must_use]
    pub fn shards_for_edge(&self, from_id: u64, to_id: u64) -> (ShardId, ShardId) {
        (self.shard_for_node(from_id), self.shard_for_node(to_id))
    }

    /// Check if an edge crosses shard boundaries.
    #[must_use]
    pub fn is_cross_shard_edge(&self, from_id: u64, to_id: u64) -> bool {
        self.shard_for_node(from_id) != self.shard_for_node(to_id)
    }

    fn hash_partition(&self, node_id: u64) -> ShardId {
        let mut hasher = DefaultHasher::new();
        node_id.hash(&mut hasher);
        let hash = hasher.finish();
        #[allow(clippy::cast_possible_truncation)]
        let shard = (hash % u64::from(self.config.num_shards)) as u32;
        shard
    }

    fn range_partition(&self, node_id: u64) -> ShardId {
        if let Some(ref ranges) = self.config.ranges {
            for (shard_id, (start, end)) in ranges.iter().enumerate() {
                if node_id >= *start && node_id < *end {
                    #[allow(clippy::cast_possible_truncation)]
                    return shard_id as u32;
                }
            }
        }
        // Fallback to hash-based if no range matches
        self.hash_partition(node_id)
    }

    fn modular_partition(&self, node_id: u64) -> ShardId {
        #[allow(clippy::cast_possible_truncation)]
        let shard = (node_id % u64::from(self.config.num_shards)) as u32;
        shard
    }
}

/// Statistics about partition distribution.
#[derive(Debug, Clone, Default)]
pub struct PartitionStats {
    /// Number of nodes per shard.
    pub nodes_per_shard: HashMap<ShardId, usize>,
    /// Number of edges per shard (source shard).
    pub edges_per_shard: HashMap<ShardId, usize>,
    /// Number of cross-shard edges.
    pub cross_shard_edges: usize,
    /// Total nodes.
    pub total_nodes: usize,
    /// Total edges.
    pub total_edges: usize,
}

impl PartitionStats {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate the imbalance ratio (max/min nodes per shard).
    /// A value of 1.0 means perfectly balanced.
    #[must_use]
    pub fn imbalance_ratio(&self) -> f64 {
        if self.nodes_per_shard.is_empty() {
            return 1.0;
        }

        let max = self.nodes_per_shard.values().max().copied().unwrap_or(0);
        let min = self.nodes_per_shard.values().min().copied().unwrap_or(0);

        if min == 0 {
            return f64::INFINITY;
        }

        #[allow(clippy::cast_precision_loss)]
        let ratio = max as f64 / min as f64;
        ratio
    }

    /// Calculate the cross-shard edge ratio.
    #[must_use]
    pub fn cross_shard_ratio(&self) -> f64 {
        if self.total_edges == 0 {
            return 0.0;
        }

        #[allow(clippy::cast_precision_loss)]
        let ratio = self.cross_shard_edges as f64 / self.total_edges as f64;
        ratio
    }
}

/// Partition assignment for batch operations.
#[derive(Debug, Clone)]
pub struct PartitionAssignment {
    /// Node IDs grouped by shard.
    pub nodes_by_shard: HashMap<ShardId, Vec<u64>>,
}

impl PartitionAssignment {
    /// Create assignments for a batch of node IDs.
    #[must_use]
    pub fn from_nodes(partitioner: &GraphPartitioner, node_ids: &[u64]) -> Self {
        let mut nodes_by_shard: HashMap<ShardId, Vec<u64>> = HashMap::new();
        for &node_id in node_ids {
            let shard = partitioner.shard_for_node(node_id);
            nodes_by_shard.entry(shard).or_default().push(node_id);
        }
        Self { nodes_by_shard }
    }

    /// Get the shards involved in this assignment.
    #[must_use]
    pub fn shards(&self) -> Vec<ShardId> {
        self.nodes_by_shard.keys().copied().collect()
    }

    /// Check if this assignment spans multiple shards.
    #[must_use]
    pub fn is_multi_shard(&self) -> bool {
        self.nodes_by_shard.len() > 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_partition_deterministic() {
        let partitioner = GraphPartitioner::new(PartitionConfig::new(4));

        // Same node ID should always map to same shard
        let shard1 = partitioner.shard_for_node(100);
        let shard2 = partitioner.shard_for_node(100);
        assert_eq!(shard1, shard2);
    }

    #[test]
    fn test_hash_partition_distribution() {
        let partitioner = GraphPartitioner::new(PartitionConfig::new(4));

        let mut counts = [0usize; 4];
        for i in 0..1000 {
            let shard = partitioner.shard_for_node(i);
            counts[shard as usize] += 1;
        }

        // Check roughly even distribution (each shard should have 150-350 nodes)
        for count in counts {
            assert!(
                count > 150 && count < 350,
                "Uneven distribution: {counts:?}"
            );
        }
    }

    #[test]
    fn test_modular_partition() {
        let config = PartitionConfig::new(3).with_strategy(PartitionStrategy::Modular);
        let partitioner = GraphPartitioner::new(config);

        assert_eq!(partitioner.shard_for_node(0), 0);
        assert_eq!(partitioner.shard_for_node(1), 1);
        assert_eq!(partitioner.shard_for_node(2), 2);
        assert_eq!(partitioner.shard_for_node(3), 0);
        assert_eq!(partitioner.shard_for_node(4), 1);
    }

    #[test]
    fn test_range_partition() {
        let config = PartitionConfig::new(3).with_ranges(vec![(0, 100), (100, 200), (200, 300)]);
        let partitioner = GraphPartitioner::new(config);

        assert_eq!(partitioner.shard_for_node(50), 0);
        assert_eq!(partitioner.shard_for_node(150), 1);
        assert_eq!(partitioner.shard_for_node(250), 2);
    }

    #[test]
    fn test_cross_shard_edge_detection() {
        let config = PartitionConfig::new(2).with_strategy(PartitionStrategy::Modular);
        let partitioner = GraphPartitioner::new(config);

        // Node 0 -> shard 0, Node 1 -> shard 1
        assert!(partitioner.is_cross_shard_edge(0, 1));
        // Node 0 -> shard 0, Node 2 -> shard 0
        assert!(!partitioner.is_cross_shard_edge(0, 2));
    }

    #[test]
    fn test_partition_assignment() {
        let config = PartitionConfig::new(2).with_strategy(PartitionStrategy::Modular);
        let partitioner = GraphPartitioner::new(config);

        let assignment = PartitionAssignment::from_nodes(&partitioner, &[0, 1, 2, 3, 4, 5]);

        assert!(assignment.is_multi_shard());
        assert_eq!(assignment.shards().len(), 2);
        // Even nodes to shard 0, odd to shard 1
        assert_eq!(assignment.nodes_by_shard[&0], vec![0, 2, 4]);
        assert_eq!(assignment.nodes_by_shard[&1], vec![1, 3, 5]);
    }

    #[test]
    fn test_manual_assignment() {
        let mut partitioner = GraphPartitioner::new(PartitionConfig::new(4));

        // Override assignment for specific node
        partitioner.assign_node(100, 3);
        assert_eq!(partitioner.shard_for_node(100), 3);
    }

    #[test]
    fn test_partition_stats() {
        let mut stats = PartitionStats::new();
        stats.nodes_per_shard.insert(0, 100);
        stats.nodes_per_shard.insert(1, 100);
        stats.nodes_per_shard.insert(2, 100);
        stats.total_nodes = 300;
        stats.total_edges = 500;
        stats.cross_shard_edges = 100;

        assert!((stats.imbalance_ratio() - 1.0).abs() < f64::EPSILON);
        assert!((stats.cross_shard_ratio() - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_partition_stats_imbalanced() {
        let mut stats = PartitionStats::new();
        stats.nodes_per_shard.insert(0, 100);
        stats.nodes_per_shard.insert(1, 200);
        stats.total_nodes = 300;

        assert!((stats.imbalance_ratio() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_all_shards() {
        let partitioner = GraphPartitioner::new(PartitionConfig::new(5));
        let shards = partitioner.all_shards();
        assert_eq!(shards, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_config_builder() {
        let config = PartitionConfig::new(4).with_strategy(PartitionStrategy::Modular);

        assert_eq!(config.num_shards, 4);
        assert_eq!(config.strategy, PartitionStrategy::Modular);
    }

    #[test]
    fn test_shards_for_edge() {
        let config = PartitionConfig::new(2).with_strategy(PartitionStrategy::Modular);
        let partitioner = GraphPartitioner::new(config);

        let (from_shard, to_shard) = partitioner.shards_for_edge(0, 1);
        assert_eq!(from_shard, 0);
        assert_eq!(to_shard, 1);
    }
}
