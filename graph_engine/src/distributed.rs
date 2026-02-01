//! Distributed graph engine abstractions for consensus and coordination.
//!
//! This module provides traits and types for building distributed graph
//! operations. The actual consensus implementation (via `tensor_chain`) is
//! integrated at a higher level to avoid circular dependencies.

#![allow(clippy::module_name_repetitions)]

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::partitioning::{GraphPartitioner, PartitionAssignment, PartitionConfig, ShardId};
use crate::{Direction, Edge, GraphEngine, Node, Path, PropertyValue, Result};

/// Node identifier in the cluster.
pub type NodeId = String;

/// Raft node state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftState {
    /// Following a leader.
    Follower,
    /// Requesting votes for leadership.
    Candidate,
    /// Leading the cluster.
    Leader,
}

/// Cluster partition status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStatus {
    /// Quorum is available for writes.
    QuorumReachable,
    /// Quorum lost, reject writes.
    QuorumLost,
    /// Exact 50/50 split.
    Stalemate,
    /// Status unknown.
    Unknown,
}

/// Cluster view snapshot.
#[derive(Debug, Clone)]
pub struct ClusterView {
    /// All node IDs in the cluster.
    pub nodes: Vec<NodeId>,
    /// Healthy node IDs.
    pub healthy_nodes: Vec<NodeId>,
    /// Failed node IDs.
    pub failed_nodes: Vec<NodeId>,
    /// Cluster generation number.
    pub generation: u64,
    /// Partition status.
    pub partition_status: PartitionStatus,
}

impl Default for ClusterView {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            healthy_nodes: Vec::new(),
            failed_nodes: Vec::new(),
            generation: 0,
            partition_status: PartitionStatus::Unknown,
        }
    }
}

/// Trait for consensus operations.
pub trait Consensus: Send + Sync {
    /// Get this node's ID.
    fn node_id(&self) -> &str;

    /// Check if this node is the leader.
    fn is_leader(&self) -> bool;

    /// Get the current leader.
    fn current_leader(&self) -> Option<NodeId>;

    /// Get the Raft state.
    fn raft_state(&self) -> RaftState;

    /// Get current term.
    fn current_term(&self) -> u64;

    /// Get commit index.
    fn commit_index(&self) -> u64;
}

/// Trait for cluster membership.
pub trait Membership: Send + Sync {
    /// Get current cluster view.
    fn view(&self) -> ClusterView;

    /// Check if a node is healthy.
    fn is_healthy(&self, node_id: &NodeId) -> bool;

    /// Get count of healthy nodes.
    fn healthy_count(&self) -> usize;

    /// Get partition status.
    fn partition_status(&self) -> PartitionStatus;
}

/// Trait for distributed transactions.
pub trait DistributedTx: Send + Sync {
    /// Begin a transaction.
    ///
    /// # Errors
    ///
    /// Returns error if transaction cannot be started.
    fn begin(&self, participants: Vec<ShardId>) -> std::result::Result<u64, String>;

    /// Prepare phase.
    ///
    /// # Errors
    ///
    /// Returns error if prepare fails.
    fn prepare(&self, tx_id: u64) -> std::result::Result<bool, String>;

    /// Commit phase.
    ///
    /// # Errors
    ///
    /// Returns error if commit fails.
    fn commit(&self, tx_id: u64) -> std::result::Result<(), String>;

    /// Abort phase.
    ///
    /// # Errors
    ///
    /// Returns error if abort fails.
    fn abort(&self, tx_id: u64) -> std::result::Result<(), String>;
}

/// Configuration for distributed graph engine.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// This node's ID.
    pub node_id: NodeId,
    /// Peer node IDs.
    pub peers: Vec<NodeId>,
    /// Partitioning configuration.
    pub partition_config: PartitionConfig,
    /// Enable read replicas (read from any node).
    pub enable_read_replicas: bool,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            node_id: "node1".to_string(),
            peers: Vec::new(),
            partition_config: PartitionConfig::default(),
            enable_read_replicas: true,
        }
    }
}

impl DistributedConfig {
    #[must_use]
    pub fn new(node_id: impl Into<NodeId>) -> Self {
        Self {
            node_id: node_id.into(),
            ..Default::default()
        }
    }

    #[must_use]
    pub fn with_peers(mut self, peers: Vec<NodeId>) -> Self {
        self.peers = peers;
        self
    }

    #[must_use]
    pub fn with_partition_config(mut self, config: PartitionConfig) -> Self {
        self.partition_config = config;
        self
    }

    #[must_use]
    pub const fn with_read_replicas(mut self, enable: bool) -> Self {
        self.enable_read_replicas = enable;
        self
    }
}

/// Result type for distributed operations.
pub type DistributedResult<T> = std::result::Result<T, DistributedError>;

/// Errors specific to distributed operations.
#[derive(Debug, Clone)]
pub enum DistributedError {
    /// Not the leader, cannot perform write.
    NotLeader { leader: Option<NodeId> },
    /// Quorum not available.
    QuorumLost,
    /// Transaction conflict.
    Conflict { tx_id: u64, reason: String },
    /// Timeout waiting for consensus.
    Timeout,
    /// Shard not found.
    ShardNotFound { shard: ShardId },
    /// Cross-shard operation failed.
    CrossShardFailed { reason: String },
    /// Local graph error.
    GraphError(String),
    /// Network error.
    NetworkError(String),
}

impl fmt::Display for DistributedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotLeader { leader } => {
                write!(f, "Not leader. Current leader: {leader:?}")
            },
            Self::QuorumLost => write!(f, "Quorum lost"),
            Self::Conflict { tx_id, reason } => {
                write!(f, "Transaction {tx_id} conflict: {reason}")
            },
            Self::Timeout => write!(f, "Operation timed out"),
            Self::ShardNotFound { shard } => write!(f, "Shard {shard} not found"),
            Self::CrossShardFailed { reason } => {
                write!(f, "Cross-shard operation failed: {reason}")
            },
            Self::GraphError(e) => write!(f, "Graph error: {e}"),
            Self::NetworkError(e) => write!(f, "Network error: {e}"),
        }
    }
}

impl std::error::Error for DistributedError {}

impl From<crate::GraphError> for DistributedError {
    fn from(e: crate::GraphError) -> Self {
        Self::GraphError(e.to_string())
    }
}

/// Operation to be replicated via Raft.
#[derive(Debug, Clone)]
pub enum GraphOperation {
    /// Create a node.
    CreateNode {
        label: String,
        properties: HashMap<String, PropertyValue>,
    },
    /// Create an edge.
    CreateEdge {
        from_id: u64,
        to_id: u64,
        edge_type: String,
        properties: HashMap<String, PropertyValue>,
        directed: bool,
    },
    /// Update node properties.
    UpdateNode {
        node_id: u64,
        properties: HashMap<String, PropertyValue>,
    },
    /// Delete a node.
    DeleteNode { node_id: u64 },
    /// Delete an edge.
    DeleteEdge { edge_id: u64 },
}

/// Statistics for distributed operations.
#[derive(Debug, Default)]
pub struct DistributedStats {
    /// Total write operations.
    pub writes: AtomicU64,
    /// Total read operations.
    pub reads: AtomicU64,
    /// Cross-shard reads.
    pub cross_shard_reads: AtomicU64,
    /// Successful commits.
    pub commits: AtomicU64,
    /// Aborted transactions.
    pub aborts: AtomicU64,
    /// Leader elections.
    pub elections: AtomicU64,
}

impl DistributedStats {
    #[must_use]
    pub fn snapshot(&self) -> DistributedStatsSnapshot {
        DistributedStatsSnapshot {
            writes: self.writes.load(Ordering::Relaxed),
            reads: self.reads.load(Ordering::Relaxed),
            cross_shard_reads: self.cross_shard_reads.load(Ordering::Relaxed),
            commits: self.commits.load(Ordering::Relaxed),
            aborts: self.aborts.load(Ordering::Relaxed),
            elections: self.elections.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of distributed statistics.
#[derive(Debug, Clone)]
pub struct DistributedStatsSnapshot {
    pub writes: u64,
    pub reads: u64,
    pub cross_shard_reads: u64,
    pub commits: u64,
    pub aborts: u64,
    pub elections: u64,
}

/// No-op consensus for single-node mode.
#[derive(Debug)]
pub struct SingleNodeConsensus {
    node_id: NodeId,
}

impl SingleNodeConsensus {
    #[must_use]
    pub fn new(node_id: impl Into<NodeId>) -> Self {
        Self {
            node_id: node_id.into(),
        }
    }
}

impl Consensus for SingleNodeConsensus {
    fn node_id(&self) -> &str {
        &self.node_id
    }

    fn is_leader(&self) -> bool {
        true // Single node is always leader
    }

    fn current_leader(&self) -> Option<NodeId> {
        Some(self.node_id.clone())
    }

    fn raft_state(&self) -> RaftState {
        RaftState::Leader
    }

    fn current_term(&self) -> u64 {
        1
    }

    fn commit_index(&self) -> u64 {
        0
    }
}

/// No-op membership for single-node mode.
#[derive(Debug)]
pub struct SingleNodeMembership {
    node_id: NodeId,
}

impl SingleNodeMembership {
    #[must_use]
    pub fn new(node_id: impl Into<NodeId>) -> Self {
        Self {
            node_id: node_id.into(),
        }
    }
}

impl Membership for SingleNodeMembership {
    fn view(&self) -> ClusterView {
        ClusterView {
            nodes: vec![self.node_id.clone()],
            healthy_nodes: vec![self.node_id.clone()],
            failed_nodes: Vec::new(),
            generation: 1,
            partition_status: PartitionStatus::QuorumReachable,
        }
    }

    fn is_healthy(&self, node_id: &NodeId) -> bool {
        node_id == &self.node_id
    }

    fn healthy_count(&self) -> usize {
        1
    }

    fn partition_status(&self) -> PartitionStatus {
        PartitionStatus::QuorumReachable
    }
}

/// Distributed graph engine that wraps local `GraphEngine` with consensus.
pub struct DistributedGraphEngine {
    /// Configuration (reserved for future distributed features).
    _config: DistributedConfig,
    /// Local graph engine for this shard.
    local: Arc<GraphEngine>,
    /// Consensus implementation.
    consensus: Arc<dyn Consensus>,
    /// Membership implementation.
    membership: Arc<dyn Membership>,
    /// Graph partitioner.
    partitioner: RwLock<GraphPartitioner>,
    /// Statistics.
    pub stats: DistributedStats,
    /// Pending operations log (before commit).
    pending_ops: RwLock<HashMap<u64, Vec<GraphOperation>>>,
}

impl DistributedGraphEngine {
    /// Create a new distributed graph engine in single-node mode.
    #[must_use]
    pub fn new(config: DistributedConfig) -> Self {
        let consensus = Arc::new(SingleNodeConsensus::new(config.node_id.clone()));
        let membership = Arc::new(SingleNodeMembership::new(config.node_id.clone()));
        Self::with_consensus(config, consensus, membership)
    }

    /// Create with custom consensus and membership implementations.
    #[must_use]
    pub fn with_consensus(
        config: DistributedConfig,
        consensus: Arc<dyn Consensus>,
        membership: Arc<dyn Membership>,
    ) -> Self {
        let local = Arc::new(GraphEngine::new());
        let partitioner = GraphPartitioner::new(config.partition_config.clone());

        Self {
            _config: config,
            local,
            consensus,
            membership,
            partitioner: RwLock::new(partitioner),
            stats: DistributedStats::default(),
            pending_ops: RwLock::new(HashMap::new()),
        }
    }

    /// Get this node's ID.
    #[must_use]
    pub fn node_id(&self) -> &str {
        self.consensus.node_id()
    }

    /// Check if this node is the leader.
    #[must_use]
    pub fn is_leader(&self) -> bool {
        self.consensus.is_leader()
    }

    /// Get the current leader.
    #[must_use]
    pub fn current_leader(&self) -> Option<NodeId> {
        self.consensus.current_leader()
    }

    /// Get the Raft state.
    #[must_use]
    pub fn raft_state(&self) -> RaftState {
        self.consensus.raft_state()
    }

    /// Get the local graph engine for direct reads.
    #[must_use]
    pub fn local_engine(&self) -> &GraphEngine {
        &self.local
    }

    /// Get shard for a node ID.
    #[must_use]
    pub fn shard_for_node(&self, node_id: u64) -> ShardId {
        self.partitioner.read().shard_for_node(node_id)
    }

    // --- Read Operations (local, no consensus required) ---

    /// Get a node by ID (local read).
    ///
    /// # Errors
    ///
    /// Returns an error if the node is not found.
    pub fn get_node(&self, node_id: u64) -> Result<Node> {
        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        self.local.get_node(node_id)
    }

    /// Get an edge by ID (local read).
    ///
    /// # Errors
    ///
    /// Returns an error if the edge is not found.
    pub fn get_edge(&self, edge_id: u64) -> Result<Edge> {
        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        self.local.get_edge(edge_id)
    }

    /// Get neighbors of a node (local read).
    ///
    /// # Errors
    ///
    /// Returns an error if the node is not found.
    pub fn neighbors(
        &self,
        node_id: u64,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> Result<Vec<Node>> {
        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        self.local.neighbors(node_id, edge_type, direction, None)
    }

    /// Find a path between nodes (local read).
    ///
    /// # Errors
    ///
    /// Returns an error if nodes are not found.
    pub fn find_path(&self, from: u64, to: u64) -> Result<Path> {
        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        self.local.find_path(from, to, None)
    }

    // --- Write Operations (require consensus) ---

    /// Create a node (replicated via Raft).
    ///
    /// # Errors
    ///
    /// Returns an error if not leader or consensus fails.
    pub fn create_node(
        &self,
        label: &str,
        properties: HashMap<String, PropertyValue>,
    ) -> DistributedResult<u64> {
        self.require_leader()?;

        // Create locally first
        let node_id = self.local.create_node(label, properties.clone())?;

        // Log the operation for replication
        let op = GraphOperation::CreateNode {
            label: label.to_string(),
            properties,
        };
        self.log_operation(op);

        self.stats.writes.fetch_add(1, Ordering::Relaxed);
        self.stats.commits.fetch_add(1, Ordering::Relaxed);

        Ok(node_id)
    }

    /// Create an edge (replicated via Raft).
    ///
    /// # Errors
    ///
    /// Returns an error if not leader, nodes not found, or consensus fails.
    pub fn create_edge(
        &self,
        from_id: u64,
        to_id: u64,
        edge_type: &str,
        properties: HashMap<String, PropertyValue>,
        directed: bool,
    ) -> DistributedResult<u64> {
        self.require_leader()?;

        // Check if this is a cross-shard edge
        let partitioner = self.partitioner.read();
        if partitioner.is_cross_shard_edge(from_id, to_id) {
            self.stats.cross_shard_reads.fetch_add(1, Ordering::Relaxed);
        }
        drop(partitioner);

        let edge_id =
            self.local
                .create_edge(from_id, to_id, edge_type, properties.clone(), directed)?;

        let op = GraphOperation::CreateEdge {
            from_id,
            to_id,
            edge_type: edge_type.to_string(),
            properties,
            directed,
        };
        self.log_operation(op);

        self.stats.writes.fetch_add(1, Ordering::Relaxed);
        self.stats.commits.fetch_add(1, Ordering::Relaxed);

        Ok(edge_id)
    }

    /// Update node properties (replicated via Raft).
    ///
    /// # Errors
    ///
    /// Returns an error if not leader or node not found.
    pub fn update_node(
        &self,
        node_id: u64,
        properties: HashMap<String, PropertyValue>,
    ) -> DistributedResult<()> {
        self.require_leader()?;

        self.local.update_node(node_id, None, properties.clone())?;

        let op = GraphOperation::UpdateNode {
            node_id,
            properties,
        };
        self.log_operation(op);

        self.stats.writes.fetch_add(1, Ordering::Relaxed);
        self.stats.commits.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Delete a node (replicated via Raft).
    ///
    /// # Errors
    ///
    /// Returns an error if not leader or node not found.
    pub fn delete_node(&self, node_id: u64) -> DistributedResult<()> {
        self.require_leader()?;

        self.local.delete_node(node_id)?;

        let op = GraphOperation::DeleteNode { node_id };
        self.log_operation(op);

        self.stats.writes.fetch_add(1, Ordering::Relaxed);
        self.stats.commits.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Delete an edge (replicated via Raft).
    ///
    /// # Errors
    ///
    /// Returns an error if not leader or edge not found.
    pub fn delete_edge(&self, edge_id: u64) -> DistributedResult<()> {
        self.require_leader()?;

        self.local.delete_edge(edge_id)?;

        let op = GraphOperation::DeleteEdge { edge_id };
        self.log_operation(op);

        self.stats.writes.fetch_add(1, Ordering::Relaxed);
        self.stats.commits.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    // --- Distributed Algorithms ---

    /// Compute `PageRank` across all nodes (local computation).
    ///
    /// In a truly distributed setting, this would use Pregel-style message passing.
    /// For now, this operates on the local shard.
    ///
    /// # Errors
    ///
    /// Returns an error if computation fails.
    pub fn pagerank(&self) -> DistributedResult<HashMap<u64, f64>> {
        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        let result = self.local.pagerank(None)?;
        Ok(result.scores)
    }

    /// Find connected components (local computation).
    ///
    /// # Errors
    ///
    /// Returns an error if computation fails.
    pub fn connected_components(&self) -> DistributedResult<HashMap<u64, u64>> {
        self.stats.reads.fetch_add(1, Ordering::Relaxed);
        let result = self.local.connected_components(None)?;
        Ok(result.communities)
    }

    // --- Transaction Support ---

    /// Begin a distributed transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if transaction cannot be started.
    pub fn begin_transaction(&self) -> DistributedResult<DistributedTransaction<'_>> {
        self.require_leader()?;

        let tx_id = self.generate_tx_id();
        self.pending_ops.write().insert(tx_id, Vec::new());

        Ok(DistributedTransaction {
            tx_id,
            engine: self,
            committed: false,
        })
    }

    /// Commit a transaction.
    fn commit_transaction(&self, tx_id: u64) -> DistributedResult<()> {
        let ops = self.pending_ops.write().remove(&tx_id);
        if let Some(operations) = ops {
            // Apply all operations
            for op in operations {
                self.apply_operation(&op)?;
            }
        }
        self.stats.commits.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Abort a transaction.
    fn abort_transaction(&self, tx_id: u64) {
        self.pending_ops.write().remove(&tx_id);
        self.stats.aborts.fetch_add(1, Ordering::Relaxed);
    }

    // --- Internal Methods ---

    fn require_leader(&self) -> DistributedResult<()> {
        if !self.is_leader() {
            return Err(DistributedError::NotLeader {
                leader: self.current_leader(),
            });
        }
        Ok(())
    }

    #[allow(clippy::unused_self)] // Will use self when Raft integration is complete
    fn log_operation(&self, _op: GraphOperation) {
        // In a real implementation, this would append to Raft log
        // For now, operations are applied directly
    }

    fn apply_operation(&self, op: &GraphOperation) -> DistributedResult<()> {
        match op {
            GraphOperation::CreateNode { label, properties } => {
                self.local.create_node(label, properties.clone())?;
            },
            GraphOperation::CreateEdge {
                from_id,
                to_id,
                edge_type,
                properties,
                directed,
            } => {
                self.local.create_edge(
                    *from_id,
                    *to_id,
                    edge_type,
                    properties.clone(),
                    *directed,
                )?;
            },
            GraphOperation::UpdateNode {
                node_id,
                properties,
            } => {
                self.local.update_node(*node_id, None, properties.clone())?;
            },
            GraphOperation::DeleteNode { node_id } => {
                self.local.delete_node(*node_id)?;
            },
            GraphOperation::DeleteEdge { edge_id } => {
                self.local.delete_edge(*edge_id)?;
            },
        }
        Ok(())
    }

    #[allow(clippy::unused_self)] // Will use self for node-specific IDs in distributed mode
    fn generate_tx_id(&self) -> u64 {
        static TX_COUNTER: AtomicU64 = AtomicU64::new(1);
        TX_COUNTER.fetch_add(1, Ordering::SeqCst)
    }

    /// Get cluster view from membership manager.
    #[must_use]
    pub fn cluster_view(&self) -> ClusterView {
        self.membership.view()
    }

    /// Check if quorum is available.
    #[must_use]
    pub fn has_quorum(&self) -> bool {
        let view = self.membership.view();
        matches!(view.partition_status, PartitionStatus::QuorumReachable)
    }

    /// Get partition assignment for a set of nodes.
    #[must_use]
    pub fn partition_assignment(&self, node_ids: &[u64]) -> PartitionAssignment {
        let partitioner = self.partitioner.read();
        PartitionAssignment::from_nodes(&partitioner, node_ids)
    }
}

/// A distributed transaction handle.
pub struct DistributedTransaction<'a> {
    tx_id: u64,
    engine: &'a DistributedGraphEngine,
    committed: bool,
}

impl DistributedTransaction<'_> {
    /// Get the transaction ID.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.tx_id
    }

    /// Commit the transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if commit fails.
    pub fn commit(mut self) -> DistributedResult<()> {
        self.committed = true;
        self.engine.commit_transaction(self.tx_id)
    }

    /// Abort the transaction.
    pub fn abort(mut self) {
        self.committed = true; // Prevent double-abort in drop
        self.engine.abort_transaction(self.tx_id);
    }
}

impl Drop for DistributedTransaction<'_> {
    fn drop(&mut self) {
        if !self.committed {
            self.engine.abort_transaction(self.tx_id);
        }
    }
}

/// Builder for distributed graph operations across multiple shards.
pub struct CrossShardQuery<'a> {
    engine: &'a DistributedGraphEngine,
    target_shards: Vec<ShardId>,
}

impl<'a> CrossShardQuery<'a> {
    /// Create a new cross-shard query targeting specific shards.
    #[must_use]
    pub const fn new(engine: &'a DistributedGraphEngine, shards: Vec<ShardId>) -> Self {
        Self {
            engine,
            target_shards: shards,
        }
    }

    /// Create a query targeting all shards.
    #[must_use]
    pub fn all_shards(engine: &'a DistributedGraphEngine) -> Self {
        let shards = engine.partitioner.read().all_shards();
        Self::new(engine, shards)
    }

    /// Get target shards.
    #[must_use]
    pub fn shards(&self) -> &[ShardId] {
        &self.target_shards
    }

    /// Execute a scatter-gather read across shards.
    /// For now, this just reads from local since we have a single shard.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard query fails.
    pub fn find_nodes_by_label(&self, label: &str) -> DistributedResult<Vec<Node>> {
        self.engine
            .stats
            .cross_shard_reads
            .fetch_add(1, Ordering::Relaxed);
        let nodes = self.engine.local.find_nodes_by_label(label)?;
        Ok(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_config_builder() {
        let config = DistributedConfig::new("node1")
            .with_peers(vec!["node2".to_string(), "node3".to_string()])
            .with_read_replicas(false);

        assert_eq!(config.node_id, "node1");
        assert_eq!(config.peers.len(), 2);
        assert!(!config.enable_read_replicas);
    }

    #[test]
    fn test_distributed_engine_creation() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        assert_eq!(engine.node_id(), "node1");
        assert!(engine.is_leader()); // Single-node is always leader
    }

    #[test]
    fn test_single_node_consensus() {
        let consensus = SingleNodeConsensus::new("node1");

        assert_eq!(consensus.node_id(), "node1");
        assert!(consensus.is_leader());
        assert_eq!(consensus.current_leader(), Some("node1".to_string()));
        assert_eq!(consensus.raft_state(), RaftState::Leader);
    }

    #[test]
    fn test_single_node_membership() {
        let membership = SingleNodeMembership::new("node1");

        let view = membership.view();
        assert_eq!(view.nodes.len(), 1);
        assert_eq!(view.healthy_nodes.len(), 1);
        assert!(view.failed_nodes.is_empty());
        assert_eq!(view.partition_status, PartitionStatus::QuorumReachable);
    }

    #[test]
    fn test_local_reads() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        // Create directly on local engine
        let node_id = engine.local.create_node("Person", HashMap::new()).unwrap();

        // Read via distributed engine
        let node = engine.get_node(node_id).unwrap();
        assert!(node.labels.contains(&"Person".to_string()));

        let stats = engine.stats.snapshot();
        assert_eq!(stats.reads, 1);
    }

    #[test]
    fn test_write_operations() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        // Create node via distributed engine
        let node_id = engine.create_node("Person", HashMap::new()).unwrap();

        // Verify it was created
        let node = engine.get_node(node_id).unwrap();
        assert!(node.labels.contains(&"Person".to_string()));

        let stats = engine.stats.snapshot();
        assert_eq!(stats.writes, 1);
        assert_eq!(stats.commits, 1);
    }

    #[test]
    fn test_create_edge() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();

        let edge_id = engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        let edge = engine.get_edge(edge_id).unwrap();
        assert_eq!(edge.edge_type, "KNOWS");
    }

    #[test]
    fn test_update_node() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let node_id = engine.create_node("Person", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Alice".to_string()),
        );
        engine.update_node(node_id, props).unwrap();

        let node = engine.get_node(node_id).unwrap();
        assert_eq!(
            node.properties.get("name"),
            Some(&PropertyValue::String("Alice".to_string()))
        );
    }

    #[test]
    fn test_delete_node() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let node_id = engine.create_node("Person", HashMap::new()).unwrap();
        engine.delete_node(node_id).unwrap();

        assert!(engine.get_node(node_id).is_err());
    }

    #[test]
    fn test_delete_edge() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        let edge_id = engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        engine.delete_edge(edge_id).unwrap();
        assert!(engine.get_edge(edge_id).is_err());
    }

    #[test]
    fn test_transaction_commit() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let tx = engine.begin_transaction().unwrap();
        let tx_id = tx.id();
        assert!(tx_id > 0);

        tx.commit().unwrap();

        let stats = engine.stats.snapshot();
        assert!(stats.commits >= 1);
    }

    #[test]
    fn test_transaction_abort() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let tx = engine.begin_transaction().unwrap();
        tx.abort();

        let stats = engine.stats.snapshot();
        assert_eq!(stats.aborts, 1);
    }

    #[test]
    fn test_transaction_abort_on_drop() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        {
            let _tx = engine.begin_transaction().unwrap();
            // Transaction drops without commit
        }

        let stats = engine.stats.snapshot();
        assert_eq!(stats.aborts, 1);
    }

    #[test]
    fn test_partition_assignment() {
        let partition_config = PartitionConfig::new(4);
        let config = DistributedConfig::new("node1").with_partition_config(partition_config);
        let engine = DistributedGraphEngine::new(config);

        let assignment = engine.partition_assignment(&[1, 2, 3, 4, 5]);
        assert!(!assignment.shards().is_empty());
    }

    #[test]
    fn test_cross_shard_query() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        // Create some nodes
        engine.create_node("Person", HashMap::new()).unwrap();
        engine.create_node("Person", HashMap::new()).unwrap();

        let query = CrossShardQuery::all_shards(&engine);
        let nodes = query.find_nodes_by_label("Person").unwrap();
        assert_eq!(nodes.len(), 2);

        let stats = engine.stats.snapshot();
        assert_eq!(stats.cross_shard_reads, 1);
    }

    #[test]
    fn test_distributed_error_display() {
        let err = DistributedError::NotLeader {
            leader: Some("node2".to_string()),
        };
        assert!(err.to_string().contains("node2"));

        let err = DistributedError::Conflict {
            tx_id: 42,
            reason: "write conflict".to_string(),
        };
        assert!(err.to_string().contains("42"));
        assert!(err.to_string().contains("write conflict"));
    }

    #[test]
    fn test_cluster_view() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let view = engine.cluster_view();
        assert_eq!(view.nodes.len(), 1);
        assert!(engine.has_quorum());
    }

    #[test]
    fn test_shard_for_node() {
        let partition_config = PartitionConfig::new(4);
        let config = DistributedConfig::new("node1").with_partition_config(partition_config);
        let engine = DistributedGraphEngine::new(config);

        let shard = engine.shard_for_node(100);
        assert!(shard < 4);
    }

    #[test]
    fn test_raft_state() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        // Single node is always leader
        assert_eq!(engine.raft_state(), RaftState::Leader);
    }

    #[test]
    fn test_pagerank_distributed() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let n1 = engine.create_node("Page", HashMap::new()).unwrap();
        let n2 = engine.create_node("Page", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "LINKS", HashMap::new(), true)
            .unwrap();

        let scores = engine.pagerank().unwrap();
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_connected_components_distributed() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        engine.create_node("Node", HashMap::new()).unwrap();
        engine.create_node("Node", HashMap::new()).unwrap();

        let components = engine.connected_components().unwrap();
        assert_eq!(components.len(), 2); // Two disconnected nodes
    }

    #[test]
    fn test_neighbors() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let n1 = engine.create_node("Person", HashMap::new()).unwrap();
        let n2 = engine.create_node("Person", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "KNOWS", HashMap::new(), true)
            .unwrap();

        let neighbors = engine.neighbors(n1, None, Direction::Outgoing).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].id, n2);
    }

    #[test]
    fn test_find_path() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let n1 = engine.create_node("Node", HashMap::new()).unwrap();
        let n2 = engine.create_node("Node", HashMap::new()).unwrap();
        let n3 = engine.create_node("Node", HashMap::new()).unwrap();
        engine
            .create_edge(n1, n2, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(n2, n3, "NEXT", HashMap::new(), true)
            .unwrap();

        let path = engine.find_path(n1, n3).unwrap();
        assert_eq!(path.nodes.len(), 3);
    }

    #[test]
    fn test_cluster_view_default() {
        let view = ClusterView::default();
        assert!(view.nodes.is_empty());
        assert_eq!(view.generation, 0);
        assert_eq!(view.partition_status, PartitionStatus::Unknown);
    }

    #[test]
    fn test_single_node_consensus_current_term() {
        let consensus = SingleNodeConsensus::new("node1");
        assert_eq!(consensus.current_term(), 1);
    }

    #[test]
    fn test_single_node_consensus_commit_index() {
        let consensus = SingleNodeConsensus::new("node1");
        assert_eq!(consensus.commit_index(), 0);
    }

    #[test]
    fn test_single_node_membership_is_healthy() {
        let membership = SingleNodeMembership::new("node1");
        assert!(membership.is_healthy(&"node1".to_string()));
        assert!(!membership.is_healthy(&"node2".to_string()));
    }

    #[test]
    fn test_single_node_membership_healthy_count() {
        let membership = SingleNodeMembership::new("node1");
        assert_eq!(membership.healthy_count(), 1);
    }

    #[test]
    fn test_single_node_membership_partition_status() {
        let membership = SingleNodeMembership::new("node1");
        assert_eq!(
            membership.partition_status(),
            PartitionStatus::QuorumReachable
        );
    }

    #[test]
    fn test_distributed_error_quorum_lost_display() {
        let err = DistributedError::QuorumLost;
        assert_eq!(err.to_string(), "Quorum lost");
    }

    #[test]
    fn test_distributed_error_timeout_display() {
        let err = DistributedError::Timeout;
        assert_eq!(err.to_string(), "Operation timed out");
    }

    #[test]
    fn test_distributed_error_shard_not_found_display() {
        let err = DistributedError::ShardNotFound { shard: 5 };
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn test_distributed_error_cross_shard_failed_display() {
        let err = DistributedError::CrossShardFailed {
            reason: "network partition".to_string(),
        };
        assert!(err.to_string().contains("network partition"));
    }

    #[test]
    fn test_distributed_error_network_error_display() {
        let err = DistributedError::NetworkError("connection refused".to_string());
        assert!(err.to_string().contains("connection refused"));
    }

    #[test]
    fn test_distributed_error_graph_error_display() {
        let err = DistributedError::GraphError("node not found".to_string());
        assert!(err.to_string().contains("node not found"));
    }

    #[test]
    fn test_distributed_error_is_error() {
        let err: Box<dyn std::error::Error> = Box::new(DistributedError::Timeout);
        assert!(err.to_string().contains("timed out"));
    }

    #[test]
    fn test_distributed_config_with_read_replicas_disabled() {
        let config = DistributedConfig::new("node1").with_read_replicas(false);
        assert!(!config.enable_read_replicas);
    }

    #[test]
    fn test_distributed_config_with_read_replicas_enabled() {
        let config = DistributedConfig::new("node1").with_read_replicas(true);
        assert!(config.enable_read_replicas);
    }

    #[test]
    fn test_partition_status_variants() {
        assert_ne!(
            PartitionStatus::QuorumReachable,
            PartitionStatus::QuorumLost
        );
        assert_ne!(PartitionStatus::Stalemate, PartitionStatus::Unknown);
        assert_eq!(
            PartitionStatus::QuorumReachable,
            PartitionStatus::QuorumReachable
        );
    }

    #[test]
    fn test_raft_state_variants() {
        assert_ne!(RaftState::Follower, RaftState::Candidate);
        assert_ne!(RaftState::Candidate, RaftState::Leader);
        assert_eq!(RaftState::Leader, RaftState::Leader);
    }

    #[test]
    fn test_cross_shard_query_new() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let query = CrossShardQuery::new(&engine, vec![0, 1, 2]);
        assert_eq!(query.shards(), &[0, 1, 2]);
    }

    #[test]
    fn test_graph_operation_variants() {
        let op1 = GraphOperation::CreateNode {
            label: "Person".to_string(),
            properties: HashMap::new(),
        };
        let _ = format!("{:?}", op1);

        let op2 = GraphOperation::CreateEdge {
            from_id: 1,
            to_id: 2,
            edge_type: "KNOWS".to_string(),
            properties: HashMap::new(),
            directed: true,
        };
        let _ = format!("{:?}", op2);

        let op3 = GraphOperation::UpdateNode {
            node_id: 1,
            properties: HashMap::new(),
        };
        let _ = format!("{:?}", op3);

        let op4 = GraphOperation::DeleteNode { node_id: 1 };
        let _ = format!("{:?}", op4);

        let op5 = GraphOperation::DeleteEdge { edge_id: 1 };
        let _ = format!("{:?}", op5);
    }

    #[test]
    fn test_distributed_stats_default() {
        let stats = DistributedStats::default();
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.writes, 0);
        assert_eq!(snapshot.reads, 0);
        assert_eq!(snapshot.cross_shard_reads, 0);
        assert_eq!(snapshot.commits, 0);
        assert_eq!(snapshot.aborts, 0);
        assert_eq!(snapshot.elections, 0);
    }

    #[test]
    fn test_distributed_graph_engine_local_engine() {
        let config = DistributedConfig::new("node1");
        let engine = DistributedGraphEngine::new(config);

        let local = engine.local_engine();
        let node_id = local.create_node("Test", HashMap::new()).unwrap();
        assert!(local.get_node(node_id).is_ok());
    }

    #[test]
    fn test_distributed_error_from_graph_error() {
        let graph_error = crate::GraphError::NodeNotFound(123);
        let distributed_error: DistributedError = graph_error.into();
        match distributed_error {
            DistributedError::GraphError(msg) => assert!(msg.contains("123")),
            _ => panic!("Expected GraphError variant"),
        }
    }

    #[test]
    fn test_distributed_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.node_id, "node1");
        assert!(config.peers.is_empty());
        assert!(config.enable_read_replicas);
    }

    #[test]
    fn test_cluster_view_clone() {
        let view = ClusterView {
            nodes: vec!["node1".to_string()],
            healthy_nodes: vec!["node1".to_string()],
            failed_nodes: Vec::new(),
            generation: 5,
            partition_status: PartitionStatus::QuorumReachable,
        };
        let cloned = view.clone();
        assert_eq!(cloned.generation, 5);
        assert_eq!(cloned.nodes.len(), 1);
    }
}
