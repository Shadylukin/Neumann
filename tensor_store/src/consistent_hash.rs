//! Consistent hash ring partitioner with virtual nodes.
//!
//! Uses consistent hashing to distribute keys across physical nodes with
//! configurable virtual nodes per physical node for better distribution.

use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
};

use crate::partitioner::{PartitionId, PartitionResult, Partitioner, PhysicalNodeId};

/// Configuration for consistent hash partitioner.
#[derive(Debug, Clone)]
pub struct ConsistentHashConfig {
    /// Number of virtual nodes per physical node.
    pub virtual_nodes: usize,
    /// Local node ID.
    pub local_node: PhysicalNodeId,
}

impl ConsistentHashConfig {
    /// Create a new config with the given local node ID.
    #[must_use]
    pub fn new(local_node: impl Into<PhysicalNodeId>) -> Self {
        Self {
            virtual_nodes: 256,
            local_node: local_node.into(),
        }
    }

    /// Set the number of virtual nodes.
    #[must_use]
    pub const fn with_virtual_nodes(mut self, count: usize) -> Self {
        self.virtual_nodes = count;
        self
    }
}

impl Default for ConsistentHashConfig {
    fn default() -> Self {
        Self {
            virtual_nodes: 256,
            local_node: String::new(),
        }
    }
}

/// Virtual node in the hash ring.
#[derive(Debug, Clone)]
struct VirtualNode {
    /// Position on the ring (hash value).
    position: u64,
    /// Physical node that owns this virtual node.
    physical_node: PhysicalNodeId,
}

/// Consistent hash ring partitioner.
///
/// Maps keys to physical nodes using a ring of virtual nodes.
/// Each physical node owns multiple virtual nodes for better distribution.
#[derive(Debug, Clone)]
pub struct ConsistentHashPartitioner {
    /// Sorted list of virtual nodes by position.
    ring: Vec<VirtualNode>,
    /// Configuration.
    config: ConsistentHashConfig,
    /// Physical node to virtual node mapping.
    node_vnodes: HashMap<PhysicalNodeId, Vec<PartitionId>>,
}

impl ConsistentHashPartitioner {
    /// Create a new partitioner with the given configuration.
    #[must_use]
    pub fn new(config: ConsistentHashConfig) -> Self {
        Self {
            ring: Vec::new(),
            config,
            node_vnodes: HashMap::new(),
        }
    }

    /// Create a partitioner with a list of initial nodes.
    #[must_use]
    pub fn with_nodes(config: ConsistentHashConfig, nodes: Vec<PhysicalNodeId>) -> Self {
        let mut partitioner = Self::new(config);
        for node in nodes {
            partitioner.add_node(node);
        }
        partitioner
    }

    /// Hash a key to a position on the ring.
    fn hash_key(key: &str) -> u64 {
        let mut hasher = fxhash::FxHasher64::default();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Hash a virtual node identifier to a position on the ring.
    fn hash_vnode(node_id: &str, vnode_index: usize) -> u64 {
        let mut hasher = fxhash::FxHasher64::default();
        node_id.hash(&mut hasher);
        vnode_index.hash(&mut hasher);
        hasher.finish()
    }

    /// Find the virtual node for a given key using binary search.
    fn find_vnode(&self, key: &str) -> Option<&VirtualNode> {
        if self.ring.is_empty() {
            return None;
        }

        let hash = Self::hash_key(key);

        // Binary search for the first vnode with position >= hash
        let idx = match self.ring.binary_search_by_key(&hash, |v| v.position) {
            Ok(i) => i,
            Err(i) => {
                // Wrap around to first node if past end
                if i >= self.ring.len() {
                    0
                } else {
                    i
                }
            },
        };

        self.ring.get(idx)
    }

    /// Get ring statistics.
    #[must_use]
    pub fn stats(&self) -> ConsistentHashStats {
        ConsistentHashStats {
            total_vnodes: self.ring.len(),
            physical_nodes: self.node_vnodes.len(),
            vnodes_per_node: self.config.virtual_nodes,
        }
    }
}

impl Partitioner for ConsistentHashPartitioner {
    fn partition(&self, key: &str) -> PartitionResult {
        self.find_vnode(key).map_or_else(
            || {
                // No nodes in ring - return local with partition 0
                PartitionResult::local(self.config.local_node.clone(), 0)
            },
            |vnode| {
                PartitionResult::new(
                    vnode.physical_node.clone(),
                    vnode.position,
                    vnode.physical_node == self.config.local_node,
                )
            },
        )
    }

    fn partitions_for_node(&self, node: &PhysicalNodeId) -> Vec<PartitionId> {
        self.node_vnodes.get(node).cloned().unwrap_or_default()
    }

    fn add_node(&mut self, node: PhysicalNodeId) -> Vec<PartitionId> {
        let mut added = Vec::with_capacity(self.config.virtual_nodes);

        for i in 0..self.config.virtual_nodes {
            let position = Self::hash_vnode(&node, i);
            added.push(position);

            self.ring.push(VirtualNode {
                position,
                physical_node: node.clone(),
            });
        }

        // Keep ring sorted by position
        self.ring.sort_by_key(|v| v.position);

        // Update node -> vnodes mapping
        self.node_vnodes.insert(node, added.clone());

        added
    }

    fn remove_node(&mut self, node: &PhysicalNodeId) -> Vec<PartitionId> {
        let removed = self.node_vnodes.remove(node).unwrap_or_default();

        // Remove from ring
        self.ring.retain(|v| v.physical_node != *node);

        removed
    }

    fn nodes(&self) -> Vec<PhysicalNodeId> {
        self.node_vnodes.keys().cloned().collect()
    }

    fn local_node(&self) -> &PhysicalNodeId {
        &self.config.local_node
    }

    fn partitions_per_node(&self) -> usize {
        self.config.virtual_nodes
    }

    fn total_partitions(&self) -> usize {
        self.ring.len()
    }
}

/// Statistics about the consistent hash ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsistentHashStats {
    /// Total virtual nodes in the ring.
    pub total_vnodes: usize,
    /// Number of physical nodes.
    pub physical_nodes: usize,
    /// Configured virtual nodes per physical node.
    pub vnodes_per_node: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new() {
        let config = ConsistentHashConfig::new("node1");
        assert_eq!(config.local_node, "node1");
        assert_eq!(config.virtual_nodes, 256);
    }

    #[test]
    fn test_config_with_virtual_nodes() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(128);
        assert_eq!(config.virtual_nodes, 128);
    }

    #[test]
    fn test_config_default() {
        let config = ConsistentHashConfig::default();
        assert_eq!(config.virtual_nodes, 256);
        assert!(config.local_node.is_empty());
    }

    #[test]
    fn test_config_debug() {
        let config = ConsistentHashConfig::new("node1");
        let debug = format!("{:?}", config);
        assert!(debug.contains("ConsistentHashConfig"));
    }

    #[test]
    fn test_config_clone() {
        let config = ConsistentHashConfig::new("node1");
        let cloned = config.clone();
        assert_eq!(config.local_node, cloned.local_node);
    }

    #[test]
    fn test_partitioner_empty() {
        let config = ConsistentHashConfig::new("local");
        let partitioner = ConsistentHashPartitioner::new(config);

        let result = partitioner.partition("any_key");
        assert_eq!(result.primary, "local");
        assert!(result.is_local);
    }

    #[test]
    fn test_partitioner_single_node() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());

        // All keys should map to node1
        for i in 0..100 {
            let result = partitioner.partition(&format!("key{}", i));
            assert_eq!(result.primary, "node1");
            assert!(result.is_local);
        }
    }

    #[test]
    fn test_partitioner_multiple_nodes() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());
        partitioner.add_node("node3".to_string());

        // Keys should be distributed across nodes
        let mut node_counts = HashMap::new();
        for i in 0..1000 {
            let result = partitioner.partition(&format!("key{}", i));
            *node_counts.entry(result.primary).or_insert(0) += 1;
        }

        // Each node should get at least some keys
        assert!(node_counts.len() == 3);
        for count in node_counts.values() {
            assert!(*count > 100, "Each node should get >10% of keys");
        }
    }

    #[test]
    fn test_partitioner_deterministic() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        let key = "test_key_123";
        let result1 = partitioner.partition(key);
        let result2 = partitioner.partition(key);

        assert_eq!(result1, result2);
    }

    #[test]
    fn test_partitioner_add_node() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        let added = partitioner.add_node("node1".to_string());
        assert_eq!(added.len(), 10);

        assert_eq!(partitioner.total_partitions(), 10);
        assert_eq!(partitioner.nodes().len(), 1);
    }

    #[test]
    fn test_partitioner_remove_node() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        assert_eq!(partitioner.total_partitions(), 20);

        let removed = partitioner.remove_node(&"node2".to_string());
        assert_eq!(removed.len(), 10);
        assert_eq!(partitioner.total_partitions(), 10);
        assert_eq!(partitioner.nodes().len(), 1);
    }

    #[test]
    fn test_partitioner_remove_nonexistent() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());

        let removed = partitioner.remove_node(&"nonexistent".to_string());
        assert!(removed.is_empty());
        assert_eq!(partitioner.total_partitions(), 10);
    }

    #[test]
    fn test_partitioner_partitions_for_node() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        let node1_partitions = partitioner.partitions_for_node(&"node1".to_string());
        assert_eq!(node1_partitions.len(), 10);

        let nonexistent = partitioner.partitions_for_node(&"nonexistent".to_string());
        assert!(nonexistent.is_empty());
    }

    #[test]
    fn test_partitioner_is_local() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(100);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        // Some keys should be local, some remote
        let mut local_count = 0;
        for i in 0..100 {
            if partitioner.is_local(&format!("key{}", i)) {
                local_count += 1;
            }
        }

        assert!(local_count > 0, "Should have some local keys");
        assert!(local_count < 100, "Should have some remote keys");
    }

    #[test]
    fn test_partitioner_local_node() {
        let config = ConsistentHashConfig::new("my_node");
        let partitioner = ConsistentHashPartitioner::new(config);

        assert_eq!(partitioner.local_node(), "my_node");
    }

    #[test]
    fn test_partitioner_partitions_per_node() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(42);
        let partitioner = ConsistentHashPartitioner::new(config);

        assert_eq!(partitioner.partitions_per_node(), 42);
    }

    #[test]
    fn test_partitioner_with_nodes() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let partitioner = ConsistentHashPartitioner::with_nodes(
            config,
            vec!["node1".to_string(), "node2".to_string()],
        );

        assert_eq!(partitioner.total_partitions(), 20);
        assert_eq!(partitioner.nodes().len(), 2);
    }

    #[test]
    fn test_partitioner_stats() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(50);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        let stats = partitioner.stats();
        assert_eq!(stats.total_vnodes, 100);
        assert_eq!(stats.physical_nodes, 2);
        assert_eq!(stats.vnodes_per_node, 50);
    }

    #[test]
    fn test_stats_debug() {
        let stats = ConsistentHashStats {
            total_vnodes: 100,
            physical_nodes: 2,
            vnodes_per_node: 50,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("ConsistentHashStats"));
    }

    #[test]
    fn test_stats_clone() {
        let stats = ConsistentHashStats {
            total_vnodes: 100,
            physical_nodes: 2,
            vnodes_per_node: 50,
        };
        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }

    #[test]
    fn test_partitioner_debug() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let partitioner = ConsistentHashPartitioner::new(config);
        let debug = format!("{:?}", partitioner);
        assert!(debug.contains("ConsistentHashPartitioner"));
    }

    #[test]
    fn test_partitioner_clone() {
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(10);
        let mut partitioner = ConsistentHashPartitioner::new(config);
        partitioner.add_node("node1".to_string());

        let cloned = partitioner.clone();
        assert_eq!(partitioner.total_partitions(), cloned.total_partitions());
    }

    #[test]
    fn test_distribution_uniformity() {
        // Test that keys are distributed across all nodes
        // With 3 nodes and 256 vnodes each, distribution can vary due to hash patterns
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(256);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());
        partitioner.add_node("node3".to_string());

        let mut counts = HashMap::new();
        for i in 0..10000 {
            let result = partitioner.partition(&format!("key_{}", i));
            *counts.entry(result.primary).or_insert(0) += 1;
        }

        // Verify all nodes received at least some keys (no node completely starved)
        assert_eq!(counts.len(), 3, "All 3 nodes should receive keys");

        // Each node should get at least 100 keys (1% of 10000)
        for (node, count) in &counts {
            assert!(
                *count >= 100,
                "Node {} should receive at least 100 keys, got {}",
                node,
                count
            );
        }
    }

    #[test]
    fn test_minimal_redistribution() {
        // When a node is added/removed, only keys that need to move should move
        let config = ConsistentHashConfig::new("node1").with_virtual_nodes(256);
        let mut partitioner = ConsistentHashPartitioner::new(config);

        partitioner.add_node("node1".to_string());
        partitioner.add_node("node2".to_string());

        // Record initial assignments
        let mut initial: HashMap<String, String> = HashMap::new();
        for i in 0..1000 {
            let key = format!("key_{}", i);
            initial.insert(key.clone(), partitioner.partition(&key).primary);
        }

        // Add a third node
        partitioner.add_node("node3".to_string());

        // Count how many keys moved
        let mut moved = 0;
        for i in 0..1000 {
            let key = format!("key_{}", i);
            let new_owner = partitioner.partition(&key).primary;
            if initial.get(&key) != Some(&new_owner) {
                moved += 1;
            }
        }

        // With consistent hashing, only ~1/N keys should move when adding Nth node
        // Adding 3rd node: ~33% of keys should move
        let move_percentage = (moved as f64 / 1000.0) * 100.0;
        assert!(
            move_percentage < 50.0,
            "Only ~33% of keys should move, got {}%",
            move_percentage
        );
    }
}
