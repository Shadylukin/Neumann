//! TensorChain - Tensor-Native Blockchain for Neumann
//!
//! A blockchain where transactions have semantic meaning:
//! - Query chain by similarity
//! - Compress history 100x via quantization
//! - Detect semantic conflicts before they happen
//! - Smart contracts as geometric constraints
//!
//! # Architecture
//!
//! ```text
//! TensorChain
//!   ├── Chain (block linking via graph edges)
//!   ├── TransactionManager (workspace isolation)
//!   ├── Codebook (hierarchical VQ for state discretization)
//!   ├── ConsensusManager (semantic conflict detection + merge)
//!   └── Network (distributed consensus via Tensor-Raft)
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use tensor_chain::TensorChain;
//!
//! // Create a new chain
//! let chain = TensorChain::new(store, "node1".to_string());
//! chain.initialize()?;
//!
//! // Begin a transaction
//! let tx = chain.begin()?;
//!
//! // Add operations
//! tx.add_operation(Transaction::Put { key: "users:1".into(), data: vec![...] })?;
//!
//! // Commit (creates new block)
//! let block_hash = chain.commit(tx)?;
//! ```

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

pub mod block;
pub mod chain;
pub mod codebook;
pub mod consensus;
pub mod delta_replication;
pub mod error;
pub mod membership;
pub mod network;
pub mod raft;
pub mod tcp;
pub mod transaction;
pub mod validation;

// Re-exports
pub use block::{Block, BlockHash, BlockHeader, NodeId, Transaction, ValidatorSignature};
pub use chain::{BlockBuilder, Chain, ChainIterator};
pub use codebook::{
    CodebookConfig, CodebookEntry, CodebookManager, GlobalCodebook, HierarchicalQuantization,
    LocalCodebook, LocalCodebookStats, PruningStrategy,
};
pub use consensus::{
    ConflictClass, ConflictResult, ConsensusConfig, ConsensusManager, DeltaVector, MergeAction,
    MergeResult,
};
pub use delta_replication::{
    DeltaBatch, DeltaReplicationConfig, DeltaReplicationManager, DeltaUpdate, ReplicationStats,
};
pub use error::{ChainError, Result};
pub use membership::{
    ClusterConfig, ClusterView, HealthConfig, LocalNodeConfig, MembershipCallback,
    MembershipManager, NodeHealth, NodeStatus, PeerNodeConfig,
};
pub use network::{
    AppendEntries, AppendEntriesResponse, LogEntry, MemoryTransport, Message, NetworkManager,
    PeerConfig, RequestVote, RequestVoteResponse, Transport,
};
pub use raft::{RaftConfig, RaftNode, RaftState};
pub use tcp::{
    Handshake, LengthDelimitedCodec, ReconnectConfig, TcpError, TcpResult, TcpTransport,
    TcpTransportConfig, TlsConfig, TransportStats,
};
pub use transaction::{
    TransactionDelta, TransactionManager, TransactionState, TransactionWorkspace,
};
pub use validation::{
    StateValidation, TransitionValidation, TransitionValidator, ValidationConfig,
};

use std::sync::Arc;

use graph_engine::GraphEngine;
use tensor_store::TensorStore;

/// Configuration for TensorChain.
#[derive(Debug, Clone)]
pub struct ChainConfig {
    /// Node ID for this chain instance.
    pub node_id: NodeId,
    /// Maximum transactions per block.
    pub max_txs_per_block: usize,
    /// Similarity threshold for conflict detection.
    pub conflict_threshold: f32,
    /// Enable auto-merge for orthogonal transactions.
    pub auto_merge: bool,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            max_txs_per_block: 1000,
            conflict_threshold: 0.7,
            auto_merge: true,
        }
    }
}

impl ChainConfig {
    /// Create a new config with the given node ID.
    pub fn new(node_id: impl Into<NodeId>) -> Self {
        Self {
            node_id: node_id.into(),
            ..Default::default()
        }
    }

    /// Set the maximum transactions per block.
    pub fn with_max_txs(mut self, max: usize) -> Self {
        self.max_txs_per_block = max;
        self
    }

    /// Set the conflict threshold.
    pub fn with_conflict_threshold(mut self, threshold: f32) -> Self {
        self.conflict_threshold = threshold;
        self
    }

    /// Enable or disable auto-merge.
    pub fn with_auto_merge(mut self, enabled: bool) -> Self {
        self.auto_merge = enabled;
        self
    }
}

/// The main TensorChain interface.
///
/// Provides a unified API for:
/// - Transaction management (begin/commit/rollback)
/// - Chain operations (append, query, verify)
/// - History queries (by key or similarity)
pub struct TensorChain {
    /// The underlying chain structure.
    chain: Chain,

    /// Transaction manager for workspace isolation.
    tx_manager: TransactionManager,

    /// Graph engine reference.
    graph: Arc<GraphEngine>,

    /// Configuration.
    config: ChainConfig,
}

impl TensorChain {
    /// Create a new TensorChain with the given store.
    pub fn new(store: TensorStore, node_id: impl Into<NodeId>) -> Self {
        let graph = Arc::new(GraphEngine::with_store(store));
        let config = ChainConfig::new(node_id);
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(store: TensorStore, config: ChainConfig) -> Self {
        let graph = Arc::new(GraphEngine::with_store(store));
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
        }
    }

    /// Initialize the chain (creates genesis block if needed).
    pub fn initialize(&self) -> Result<()> {
        self.chain.initialize()
    }

    /// Get the current chain height.
    pub fn height(&self) -> u64 {
        self.chain.height()
    }

    /// Get the tip block hash.
    pub fn tip_hash(&self) -> BlockHash {
        self.chain.tip_hash()
    }

    /// Get the node ID.
    pub fn node_id(&self) -> &NodeId {
        &self.config.node_id
    }

    /// Begin a new transaction.
    pub fn begin(&self) -> Result<Arc<TransactionWorkspace>> {
        self.tx_manager.begin(self.graph.store())
    }

    /// Commit a transaction, creating a new block.
    pub fn commit(&self, workspace: Arc<TransactionWorkspace>) -> Result<BlockHash> {
        // Mark as committing
        workspace.mark_committing()?;

        // Get operations from workspace
        let operations = workspace.operations();

        if operations.is_empty() {
            // No operations - just mark as committed without creating block
            workspace.mark_committed();
            self.tx_manager.remove(workspace.id());
            return Ok(self.chain.tip_hash());
        }

        // Build the block
        let block = self.chain.new_block().add_transactions(operations).build();

        // Append to chain
        match self.chain.append(block) {
            Ok(hash) => {
                workspace.mark_committed();
                self.tx_manager.remove(workspace.id());
                Ok(hash)
            },
            Err(e) => {
                workspace.mark_failed();
                Err(e)
            },
        }
    }

    /// Rollback a transaction.
    pub fn rollback(&self, workspace: Arc<TransactionWorkspace>) -> Result<()> {
        workspace.rollback(self.graph.store())?;
        self.tx_manager.remove(workspace.id());
        Ok(())
    }

    /// Get a block at a specific height.
    pub fn get_block(&self, height: u64) -> Result<Option<Block>> {
        self.chain.get_block_at(height)
    }

    /// Get the tip block.
    pub fn get_tip(&self) -> Result<Option<Block>> {
        self.chain.get_tip()
    }

    /// Get the genesis block.
    pub fn get_genesis(&self) -> Result<Option<Block>> {
        self.chain.get_genesis()
    }

    /// Verify the entire chain.
    pub fn verify(&self) -> Result<()> {
        self.chain.verify_chain()
    }

    /// Get change history for a key.
    pub fn history(&self, key: &str) -> Result<Vec<(u64, Transaction)>> {
        self.chain.history(key)
    }

    /// Get blocks in a height range.
    pub fn get_blocks(&self, start: u64, end: u64) -> Result<Vec<Block>> {
        self.chain.get_blocks_range(start, end)
    }

    /// Iterate over all blocks.
    pub fn iter(&self) -> ChainIterator<'_> {
        self.chain.iter()
    }

    /// Get the number of active transactions.
    pub fn active_transactions(&self) -> usize {
        self.tx_manager.active_count()
    }

    /// Access the underlying store (for advanced operations).
    pub fn store(&self) -> &TensorStore {
        self.graph.store()
    }

    /// Access the graph engine (for advanced operations).
    pub fn graph(&self) -> &GraphEngine {
        &self.graph
    }

    /// Directly append a pre-built block (for consensus/sync).
    pub fn append_block(&self, block: Block) -> Result<BlockHash> {
        self.chain.append(block)
    }

    /// Create a new block builder.
    pub fn new_block(&self) -> BlockBuilder {
        self.chain.new_block()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_chain_basic() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        chain.initialize().unwrap();

        assert_eq!(chain.height(), 0);
        assert!(chain.get_genesis().unwrap().is_some());
    }

    #[test]
    fn test_transaction_commit() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: "test_key".to_string(),
            data: vec![1, 2, 3],
        })
        .unwrap();

        let hash = chain.commit(tx).unwrap();

        assert_eq!(chain.height(), 1);
        assert_eq!(chain.tip_hash(), hash);
    }

    #[test]
    fn test_transaction_rollback() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Store something first
        let tx1 = chain.begin().unwrap();
        tx1.add_operation(Transaction::Put {
            key: "persistent".to_string(),
            data: vec![1],
        })
        .unwrap();
        chain.commit(tx1).unwrap();

        // Begin a new transaction
        let tx2 = chain.begin().unwrap();
        tx2.add_operation(Transaction::Put {
            key: "rollback_me".to_string(),
            data: vec![2],
        })
        .unwrap();

        // Rollback
        chain.rollback(tx2).unwrap();

        // Height should still be 1
        assert_eq!(chain.height(), 1);
    }

    #[test]
    fn test_empty_transaction() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        let tx = chain.begin().unwrap();
        // No operations added

        let hash = chain.commit(tx).unwrap();

        // Should not create a new block
        assert_eq!(chain.height(), 0);
        assert_eq!(chain.tip_hash(), hash);
    }

    #[test]
    fn test_chain_history() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Multiple transactions on same key
        for i in 1..=3 {
            let tx = chain.begin().unwrap();
            tx.add_operation(Transaction::Put {
                key: "tracked_key".to_string(),
                data: vec![i as u8],
            })
            .unwrap();
            chain.commit(tx).unwrap();
        }

        let history = chain.history("tracked_key").unwrap();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_chain_config() {
        let config = ChainConfig::new("my_node")
            .with_max_txs(500)
            .with_conflict_threshold(0.5)
            .with_auto_merge(false);

        assert_eq!(config.node_id, "my_node");
        assert_eq!(config.max_txs_per_block, 500);
        assert_eq!(config.conflict_threshold, 0.5);
        assert!(!config.auto_merge);
    }

    #[test]
    fn test_chain_verification() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Add some blocks
        for i in 1..=5 {
            let tx = chain.begin().unwrap();
            tx.add_operation(Transaction::Put {
                key: format!("key{}", i),
                data: vec![i as u8],
            })
            .unwrap();
            chain.commit(tx).unwrap();
        }

        // Verify should pass
        chain.verify().unwrap();
    }
}
