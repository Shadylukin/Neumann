//! Data partitioning traits for distributed storage.
//!
//! Provides abstractions for mapping keys to partitions across a cluster.
//! The default implementation uses consistent hashing with virtual nodes.

use std::fmt::Debug;

/// Unique identifier for a physical node in the cluster.
pub type PhysicalNodeId = String;

/// Unique identifier for a partition (virtual node).
pub type PartitionId = u64;

/// Result of partitioning a key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionResult {
    /// Physical node that owns this partition.
    pub primary: PhysicalNodeId,
    /// Partition identifier.
    pub partition: PartitionId,
    /// Whether this partition is owned by the local node.
    pub is_local: bool,
}

impl PartitionResult {
    /// Create a new partition result.
    pub fn new(primary: impl Into<PhysicalNodeId>, partition: PartitionId, is_local: bool) -> Self {
        Self {
            primary: primary.into(),
            partition,
            is_local,
        }
    }

    /// Create a local partition result.
    pub fn local(primary: impl Into<PhysicalNodeId>, partition: PartitionId) -> Self {
        Self::new(primary, partition, true)
    }

    /// Create a remote partition result.
    pub fn remote(primary: impl Into<PhysicalNodeId>, partition: PartitionId) -> Self {
        Self::new(primary, partition, false)
    }
}

/// Trait for partitioning keys across nodes.
pub trait Partitioner: Debug + Send + Sync {
    /// Get the partition for a key.
    fn partition(&self, key: &str) -> PartitionResult;

    /// Get all partitions owned by a physical node.
    fn partitions_for_node(&self, node: &PhysicalNodeId) -> Vec<PartitionId>;

    /// Add a new physical node to the cluster.
    /// Returns the partitions that should be migrated to the new node.
    fn add_node(&mut self, node: PhysicalNodeId) -> Vec<PartitionId>;

    /// Remove a physical node from the cluster.
    /// Returns the partitions that need to be reassigned.
    fn remove_node(&mut self, node: &PhysicalNodeId) -> Vec<PartitionId>;

    /// Get all physical nodes in the cluster.
    fn nodes(&self) -> Vec<PhysicalNodeId>;

    /// Get the local node ID.
    fn local_node(&self) -> &PhysicalNodeId;

    /// Check if a key belongs to the local node.
    fn is_local(&self, key: &str) -> bool {
        self.partition(key).is_local
    }

    /// Get the number of partitions per node.
    fn partitions_per_node(&self) -> usize;

    /// Get the total number of partitions.
    fn total_partitions(&self) -> usize;

    /// Route based on embedding similarity to node centroids.
    ///
    /// For geometric routing, routes to the node whose centroid is most similar
    /// to the given embedding. Default implementation ignores the embedding
    /// and falls back to key-based hashing.
    fn partition_by_embedding(&self, key: &str, _embedding: &[f32]) -> PartitionResult {
        self.partition(key)
    }

    /// Get a node's geometric region centroid.
    ///
    /// Returns the centroid vector for the given node's Voronoi region,
    /// or None if geometric routing is not supported.
    fn region_centroid(&self, _node: &PhysicalNodeId) -> Option<Vec<f32>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_result_new() {
        let result = PartitionResult::new("node1", 42, true);
        assert_eq!(result.primary, "node1");
        assert_eq!(result.partition, 42);
        assert!(result.is_local);
    }

    #[test]
    fn test_partition_result_local() {
        let result = PartitionResult::local("node1", 42);
        assert!(result.is_local);
    }

    #[test]
    fn test_partition_result_remote() {
        let result = PartitionResult::remote("node2", 42);
        assert!(!result.is_local);
    }

    #[test]
    fn test_partition_result_debug() {
        let result = PartitionResult::local("node1", 42);
        let debug = format!("{:?}", result);
        assert!(debug.contains("PartitionResult"));
        assert!(debug.contains("node1"));
    }

    #[test]
    fn test_partition_result_clone() {
        let result = PartitionResult::local("node1", 42);
        let cloned = result.clone();
        assert_eq!(result, cloned);
    }

    #[test]
    fn test_partition_result_equality() {
        let a = PartitionResult::local("node1", 42);
        let b = PartitionResult::local("node1", 42);
        let c = PartitionResult::local("node2", 42);

        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
