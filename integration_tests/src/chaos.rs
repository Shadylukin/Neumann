// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chaos testing utilities for distributed consensus testing.
//!
//! Provides infrastructure for injecting faults and verifying safety
//! properties under adverse conditions.

use std::sync::Arc;
use std::time::Duration;

use tensor_chain::hlc::HybridLogicalClock;
use tensor_chain::network::MemoryTransport;

/// Configuration for chaos testing scenarios.
#[derive(Debug, Clone)]
pub struct ChaosConfig {
    /// Probability of network partition (0.0-1.0).
    pub partition_probability: f32,
    /// Message reorder rate (0.0-1.0).
    pub message_reorder_rate: f32,
    /// Maximum reorder delay in milliseconds.
    pub max_reorder_delay_ms: u64,
    /// Clock drift in milliseconds (can be negative).
    pub clock_drift_ms: i64,
    /// Probability of node crash (0.0-1.0).
    pub crash_probability: f32,
    /// Message corruption rate (0.0-1.0).
    pub corruption_rate: f32,
    /// Link quality drop rate per peer (0.0-1.0).
    pub link_drop_rate: f32,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            partition_probability: 0.0,
            message_reorder_rate: 0.0,
            max_reorder_delay_ms: 0,
            clock_drift_ms: 0,
            crash_probability: 0.0,
            corruption_rate: 0.0,
            link_drop_rate: 0.0,
        }
    }
}

impl ChaosConfig {
    /// Create a mild chaos configuration for basic testing.
    pub fn mild() -> Self {
        Self {
            partition_probability: 0.01,
            message_reorder_rate: 0.05,
            max_reorder_delay_ms: 10,
            clock_drift_ms: 100,
            crash_probability: 0.0,
            corruption_rate: 0.0,
            link_drop_rate: 0.01,
        }
    }

    /// Create a moderate chaos configuration.
    pub fn moderate() -> Self {
        Self {
            partition_probability: 0.05,
            message_reorder_rate: 0.10,
            max_reorder_delay_ms: 50,
            clock_drift_ms: 500,
            crash_probability: 0.01,
            corruption_rate: 0.01,
            link_drop_rate: 0.05,
        }
    }

    /// Create an aggressive chaos configuration for stress testing.
    pub fn aggressive() -> Self {
        Self {
            partition_probability: 0.10,
            message_reorder_rate: 0.20,
            max_reorder_delay_ms: 100,
            clock_drift_ms: 1000,
            crash_probability: 0.05,
            corruption_rate: 0.02,
            link_drop_rate: 0.10,
        }
    }
}

/// A node in the chaos cluster with transport and clock.
pub struct ChaosNode {
    /// Node identifier.
    pub node_id: String,
    /// Network transport.
    pub transport: Arc<MemoryTransport>,
    /// Hybrid logical clock.
    pub clock: Arc<HybridLogicalClock>,
    /// Whether the node is currently crashed.
    crashed: bool,
}

impl ChaosNode {
    /// Create a new chaos node.
    pub fn new(node_id: String) -> Self {
        let transport = Arc::new(MemoryTransport::new(node_id.clone()));
        let clock =
            Arc::new(HybridLogicalClock::from_node_id(&node_id).expect("failed to create HLC"));

        Self {
            node_id,
            transport,
            clock,
            crashed: false,
        }
    }

    /// Check if the node is crashed.
    pub fn is_crashed(&self) -> bool {
        self.crashed
    }

    /// Crash the node (partition from all peers).
    pub fn crash(&mut self) {
        self.crashed = true;
        self.transport.partition_all();
    }

    /// Recover the node (heal all partitions).
    pub fn recover(&mut self) {
        self.crashed = false;
        self.transport.heal_all();
    }
}

/// A cluster of nodes for chaos testing.
pub struct ChaosCluster {
    /// Nodes in the cluster.
    nodes: Vec<ChaosNode>,
    /// Chaos configuration.
    config: ChaosConfig,
}

impl ChaosCluster {
    /// Create a new chaos cluster with the given number of nodes.
    pub fn new(node_count: usize, config: ChaosConfig) -> Self {
        let nodes: Vec<ChaosNode> = (0..node_count)
            .map(|i| ChaosNode::new(format!("node-{}", i)))
            .collect();

        // Connect all nodes to each other
        for i in 0..node_count {
            for j in 0..node_count {
                if i != j {
                    let sender = nodes[j].transport.sender();
                    nodes[i]
                        .transport
                        .connect_to(nodes[j].node_id.clone(), sender);
                }
            }
        }

        // Apply initial chaos configuration
        let cluster = Self { nodes, config };
        cluster.apply_config();
        cluster
    }

    /// Apply the chaos configuration to all nodes.
    fn apply_config(&self) {
        for node in &self.nodes {
            // Apply message reordering
            if self.config.message_reorder_rate > 0.0 {
                node.transport.enable_reordering(
                    self.config.message_reorder_rate,
                    self.config.max_reorder_delay_ms,
                );
            }

            // Apply corruption rate
            if self.config.corruption_rate > 0.0 {
                node.transport
                    .set_corruption_rate(self.config.corruption_rate);
            }

            // Apply clock drift
            if self.config.clock_drift_ms != 0 {
                node.clock.set_drift_offset(self.config.clock_drift_ms);
            }
        }
    }

    /// Get the number of nodes in the cluster.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get a reference to a node by index.
    pub fn node(&self, index: usize) -> Option<&ChaosNode> {
        self.nodes.get(index)
    }

    /// Get a mutable reference to a node by index.
    pub fn node_mut(&mut self, index: usize) -> Option<&mut ChaosNode> {
        self.nodes.get_mut(index)
    }

    /// Get the transport for a node.
    pub fn transport(&self, index: usize) -> Option<Arc<MemoryTransport>> {
        self.nodes.get(index).map(|n| n.transport.clone())
    }

    /// Get the clock for a node.
    pub fn clock(&self, index: usize) -> Option<Arc<HybridLogicalClock>> {
        self.nodes.get(index).map(|n| n.clock.clone())
    }

    /// Inject a network partition between two nodes.
    pub fn inject_partition(&self, from: usize, to: usize) {
        if let (Some(from_node), Some(to_node)) = (self.nodes.get(from), self.nodes.get(to)) {
            from_node.transport.partition(&to_node.node_id);
        }
    }

    /// Heal a network partition between two nodes.
    pub fn heal_partition(&self, from: usize, to: usize) {
        if let (Some(from_node), Some(to_node)) = (self.nodes.get(from), self.nodes.get(to)) {
            from_node.transport.heal(&to_node.node_id);
        }
    }

    /// Inject a symmetric partition (both directions).
    pub fn inject_symmetric_partition(&self, node_a: usize, node_b: usize) {
        self.inject_partition(node_a, node_b);
        self.inject_partition(node_b, node_a);
    }

    /// Heal a symmetric partition.
    pub fn heal_symmetric_partition(&self, node_a: usize, node_b: usize) {
        self.heal_partition(node_a, node_b);
        self.heal_partition(node_b, node_a);
    }

    /// Partition a node from all other nodes (isolate it).
    pub fn isolate_node(&self, node_index: usize) {
        if let Some(node) = self.nodes.get(node_index) {
            node.transport.partition_all();
        }
    }

    /// Heal all partitions for a node.
    pub fn rejoin_node(&self, node_index: usize) {
        if let Some(node) = self.nodes.get(node_index) {
            node.transport.heal_all();
        }
    }

    /// Inject clock drift on a specific node.
    pub fn inject_clock_drift(&self, node_index: usize, drift_ms: i64) {
        if let Some(node) = self.nodes.get(node_index) {
            node.clock.set_drift_offset(drift_ms);
        }
    }

    /// Inject a clock jump on a specific node.
    pub fn inject_clock_jump(&self, node_index: usize, jump_ms: i64) {
        if let Some(node) = self.nodes.get(node_index) {
            node.clock.inject_clock_jump(jump_ms);
        }
    }

    /// Crash a node (mark as crashed and isolate).
    pub fn crash_node(&mut self, node_index: usize) {
        if let Some(node) = self.nodes.get_mut(node_index) {
            node.crash();
        }
    }

    /// Recover a crashed node.
    pub fn recover_node(&mut self, node_index: usize) {
        if let Some(node) = self.nodes.get_mut(node_index) {
            node.recover();
        }
    }

    /// Check if a node is crashed.
    pub fn is_node_crashed(&self, node_index: usize) -> bool {
        self.nodes
            .get(node_index)
            .map(|n| n.is_crashed())
            .unwrap_or(false)
    }

    /// Set link quality between two nodes.
    pub fn set_link_quality(&self, from: usize, to: usize, drop_rate: f32) {
        if let (Some(from_node), Some(to_node)) = (self.nodes.get(from), self.nodes.get(to)) {
            from_node
                .transport
                .set_link_quality(&to_node.node_id, drop_rate);
        }
    }

    /// Get chaos statistics for all nodes.
    pub fn all_chaos_stats(&self) -> Vec<tensor_chain::network::ChaosStats> {
        self.nodes
            .iter()
            .map(|n| n.transport.chaos_stats())
            .collect()
    }

    /// Reset chaos counters for all nodes.
    pub fn reset_all_counters(&self) {
        for node in &self.nodes {
            node.transport.reset_chaos_counters();
        }
    }

    /// Get total dropped messages across all nodes.
    pub fn total_dropped_messages(&self) -> u64 {
        self.nodes
            .iter()
            .map(|n| n.transport.dropped_message_count())
            .sum()
    }

    /// Get total reordered messages across all nodes.
    pub fn total_reordered_messages(&self) -> u64 {
        self.nodes
            .iter()
            .map(|n| n.transport.reordered_message_count())
            .sum()
    }

    /// Get total corrupted messages across all nodes.
    pub fn total_corrupted_messages(&self) -> u64 {
        self.nodes
            .iter()
            .map(|n| n.transport.corrupted_message_count())
            .sum()
    }

    /// Count the number of active (non-crashed) nodes.
    pub fn active_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| !n.is_crashed()).count()
    }

    /// Create a majority partition (isolate minority).
    ///
    /// Returns the indices of nodes in the minority partition.
    pub fn create_majority_partition(&mut self) -> Vec<usize> {
        let n = self.nodes.len();
        let minority_size = n / 2; // For 5 nodes, this is 2

        let minority: Vec<usize> = (0..minority_size).collect();

        // Isolate minority nodes from majority
        for &minority_idx in &minority {
            for majority_idx in minority_size..n {
                self.inject_symmetric_partition(minority_idx, majority_idx);
            }
        }

        minority
    }

    /// Heal majority partition.
    pub fn heal_majority_partition(&mut self) {
        let n = self.nodes.len();
        let minority_size = n / 2;

        for minority_idx in 0..minority_size {
            for majority_idx in minority_size..n {
                self.heal_symmetric_partition(minority_idx, majority_idx);
            }
        }
    }
}

/// Wait for a condition with timeout.
pub async fn wait_for<F>(condition: F, timeout: Duration, check_interval: Duration) -> bool
where
    F: Fn() -> bool,
{
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        if condition() {
            return true;
        }
        tokio::time::sleep(check_interval).await;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chaos_config_default() {
        let config = ChaosConfig::default();
        assert!(config.partition_probability.abs() < f32::EPSILON);
        assert!(config.message_reorder_rate.abs() < f32::EPSILON);
        assert_eq!(config.clock_drift_ms, 0);
    }

    #[test]
    fn test_chaos_config_presets() {
        let mild = ChaosConfig::mild();
        assert!(mild.partition_probability > 0.0);

        let moderate = ChaosConfig::moderate();
        assert!(moderate.partition_probability > mild.partition_probability);

        let aggressive = ChaosConfig::aggressive();
        assert!(aggressive.partition_probability > moderate.partition_probability);
    }

    #[test]
    fn test_chaos_node_creation() {
        let node = ChaosNode::new("test-node".to_string());
        assert_eq!(node.node_id, "test-node");
        assert!(!node.is_crashed());
    }

    #[test]
    fn test_chaos_node_crash_recover() {
        let mut node = ChaosNode::new("test-node".to_string());

        assert!(!node.is_crashed());

        node.crash();
        assert!(node.is_crashed());

        node.recover();
        assert!(!node.is_crashed());
    }

    #[test]
    fn test_chaos_cluster_creation() {
        let config = ChaosConfig::default();
        let cluster = ChaosCluster::new(3, config);

        assert_eq!(cluster.node_count(), 3);
        assert_eq!(cluster.active_node_count(), 3);
    }

    #[test]
    fn test_chaos_cluster_node_access() {
        let config = ChaosConfig::default();
        let cluster = ChaosCluster::new(3, config);

        assert!(cluster.node(0).is_some());
        assert!(cluster.node(1).is_some());
        assert!(cluster.node(2).is_some());
        assert!(cluster.node(3).is_none());
    }

    #[test]
    fn test_chaos_cluster_crash_recover() {
        let config = ChaosConfig::default();
        let mut cluster = ChaosCluster::new(3, config);

        cluster.crash_node(0);
        assert!(cluster.is_node_crashed(0));
        assert_eq!(cluster.active_node_count(), 2);

        cluster.recover_node(0);
        assert!(!cluster.is_node_crashed(0));
        assert_eq!(cluster.active_node_count(), 3);
    }

    #[test]
    fn test_chaos_cluster_inject_clock_drift() {
        let config = ChaosConfig::default();
        let cluster = ChaosCluster::new(3, config);

        cluster.inject_clock_drift(0, 1000);

        let clock = cluster.clock(0).unwrap();
        assert_eq!(clock.drift_offset(), 1000);
    }

    #[test]
    fn test_chaos_cluster_inject_clock_jump() {
        let config = ChaosConfig::default();
        let cluster = ChaosCluster::new(3, config);

        cluster.inject_clock_jump(0, 500);
        cluster.inject_clock_jump(0, 300);

        let clock = cluster.clock(0).unwrap();
        assert_eq!(clock.drift_offset(), 800);
    }

    #[test]
    fn test_chaos_cluster_partition() {
        let config = ChaosConfig::default();
        let cluster = ChaosCluster::new(3, config);

        cluster.inject_partition(0, 1);

        let transport = cluster.transport(0).unwrap();
        assert!(transport.is_partitioned(&"node-1".to_string()));
        assert!(!transport.is_partitioned(&"node-2".to_string()));
    }

    #[test]
    fn test_chaos_cluster_symmetric_partition() {
        let config = ChaosConfig::default();
        let cluster = ChaosCluster::new(3, config);

        cluster.inject_symmetric_partition(0, 1);

        let t0 = cluster.transport(0).unwrap();
        let t1 = cluster.transport(1).unwrap();

        assert!(t0.is_partitioned(&"node-1".to_string()));
        assert!(t1.is_partitioned(&"node-0".to_string()));
    }

    #[test]
    fn test_chaos_cluster_isolate_node() {
        let config = ChaosConfig::default();
        let cluster = ChaosCluster::new(3, config);

        cluster.isolate_node(0);

        let t0 = cluster.transport(0).unwrap();
        assert!(t0.is_partitioned(&"node-1".to_string()));
        assert!(t0.is_partitioned(&"node-2".to_string()));
    }

    #[test]
    fn test_chaos_cluster_stats() {
        let config = ChaosConfig::default();
        let cluster = ChaosCluster::new(3, config);

        assert_eq!(cluster.total_dropped_messages(), 0);
        assert_eq!(cluster.total_reordered_messages(), 0);
        assert_eq!(cluster.total_corrupted_messages(), 0);
    }

    #[test]
    fn test_chaos_cluster_majority_partition() {
        let config = ChaosConfig::default();
        let mut cluster = ChaosCluster::new(5, config);

        let minority = cluster.create_majority_partition();
        assert_eq!(minority.len(), 2);
        assert_eq!(minority, vec![0, 1]);

        // Minority nodes should be partitioned from majority
        let t0 = cluster.transport(0).unwrap();
        assert!(t0.is_partitioned(&"node-2".to_string()));
        assert!(t0.is_partitioned(&"node-3".to_string()));
        assert!(t0.is_partitioned(&"node-4".to_string()));
        // But not from each other
        assert!(!t0.is_partitioned(&"node-1".to_string()));
    }

    #[test]
    fn test_chaos_config_clone_debug() {
        let config = ChaosConfig::mild();
        let cloned = config.clone();
        assert!((config.partition_probability - cloned.partition_probability).abs() < f32::EPSILON);

        let debug = format!("{:?}", config);
        assert!(debug.contains("ChaosConfig"));
    }
}
