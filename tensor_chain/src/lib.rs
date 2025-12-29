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
pub mod distributed_tx;
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
    BatchConflict, ConflictClass, ConflictResult, ConsensusConfig, ConsensusManager, DeltaVector,
    MergeAction, MergeResult,
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
    PeerConfig, RequestVote, RequestVoteResponse, Transport, TxAbortMsg, TxAckMsg, TxCommitMsg,
    TxPrepareMsg, TxPrepareResponseMsg, TxVote,
};
pub use raft::{FastPathState, FastPathStats, RaftConfig, RaftNode, RaftState};
pub use tcp::{
    Handshake, LengthDelimitedCodec, ReconnectConfig, TcpError, TcpResult, TcpTransport,
    TcpTransportConfig, TlsConfig, TransportStats,
};
pub use transaction::{
    TransactionDelta, TransactionManager, TransactionState, TransactionWorkspace,
};
pub use validation::{
    FastPathResult, FastPathValidator, StateValidation, TransitionValidation, TransitionValidator,
    ValidationConfig, ValidationMode,
};
pub use distributed_tx::{
    AbortRequest, CommitRequest, DistributedTransaction, DistributedTxConfig,
    DistributedTxCoordinator, DistributedTxStats, KeyLock, LockManager, PrepareRequest,
    PrepareVote, PreparedTx, ShardId, TxParticipant, TxPhase, TxResponse,
};

use std::sync::Arc;

use graph_engine::GraphEngine;
use tensor_store::TensorStore;

/// Configuration for automatic merging of orthogonal transactions.
#[derive(Debug, Clone)]
pub struct AutoMergeConfig {
    /// Whether auto-merge is enabled.
    pub enabled: bool,
    /// Cosine similarity threshold below which transactions are considered orthogonal.
    /// Transactions with |similarity| < threshold can be auto-merged.
    pub orthogonal_threshold: f32,
    /// Maximum number of transactions to merge in a single batch.
    pub max_merge_batch: usize,
    /// Time window in milliseconds to wait for merge candidates.
    pub merge_window_ms: u64,
}

impl Default for AutoMergeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            orthogonal_threshold: 0.1,
            max_merge_batch: 10,
            merge_window_ms: 100,
        }
    }
}

impl AutoMergeConfig {
    /// Create a disabled auto-merge config.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Set the orthogonal threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.orthogonal_threshold = threshold;
        self
    }

    /// Set the maximum batch size.
    pub fn with_max_batch(mut self, max: usize) -> Self {
        self.max_merge_batch = max;
        self
    }

    /// Set the merge window.
    pub fn with_window(mut self, ms: u64) -> Self {
        self.merge_window_ms = ms;
        self
    }
}

/// Configuration for TensorChain.
#[derive(Debug, Clone)]
pub struct ChainConfig {
    /// Node ID for this chain instance.
    pub node_id: NodeId,
    /// Maximum transactions per block.
    pub max_txs_per_block: usize,
    /// Similarity threshold for conflict detection.
    pub conflict_threshold: f32,
    /// Auto-merge configuration for orthogonal transactions.
    pub auto_merge: AutoMergeConfig,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            max_txs_per_block: 1000,
            conflict_threshold: 0.7,
            auto_merge: AutoMergeConfig::default(),
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
        self.auto_merge.enabled = enabled;
        self
    }

    /// Set the full auto-merge configuration.
    pub fn with_auto_merge_config(mut self, config: AutoMergeConfig) -> Self {
        self.auto_merge = config;
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
    ///
    /// If auto-merge is enabled and orthogonal transactions are found,
    /// they will be merged into a single block via vector addition.
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

        // Compute delta embedding for the workspace
        let delta = workspace.to_delta_vector();

        // Check for auto-merge candidates
        let (merged_operations, merged_delta, merged_workspaces) =
            if self.config.auto_merge.enabled {
                self.find_and_merge_orthogonal(&workspace, delta)?
            } else {
                (operations, workspace.delta_embedding(), vec![])
            };

        // Build the block with merged operations and delta embedding
        let block = self
            .chain
            .new_block()
            .add_transactions(merged_operations)
            .with_embedding(merged_delta)
            .build();

        // Append to chain
        match self.chain.append(block) {
            Ok(hash) => {
                // Mark all merged workspaces as committed
                workspace.mark_committed();
                self.tx_manager.remove(workspace.id());

                for merged_ws in merged_workspaces {
                    merged_ws.mark_committed();
                    self.tx_manager.remove(merged_ws.id());
                }

                Ok(hash)
            }
            Err(e) => {
                workspace.mark_failed();
                Err(e)
            }
        }
    }

    /// Find orthogonal transactions and merge them via vector addition.
    #[allow(clippy::type_complexity)]
    fn find_and_merge_orthogonal(
        &self,
        workspace: &Arc<TransactionWorkspace>,
        mut delta: DeltaVector,
    ) -> Result<(Vec<Transaction>, Vec<f32>, Vec<Arc<TransactionWorkspace>>)> {
        let mut all_operations = workspace.operations();
        let mut merged_workspaces = Vec::new();

        // Find merge candidates
        let candidates = self.tx_manager.find_merge_candidates(
            workspace,
            self.config.auto_merge.orthogonal_threshold,
        );

        // Limit to max_merge_batch
        let max_merge = self.config.auto_merge.max_merge_batch;
        let candidates_to_merge: Vec<_> = candidates.into_iter().take(max_merge).collect();

        // Merge each orthogonal candidate
        for candidate in candidates_to_merge {
            // Mark candidate as committing
            if candidate.workspace.mark_committing().is_err() {
                continue; // Skip if can't mark as committing
            }

            // Merge operations
            all_operations.extend(candidate.workspace.operations());

            // Merge delta via vector addition
            delta = delta.add(&candidate.delta);

            // Track merged workspace
            merged_workspaces.push(candidate.workspace);
        }

        Ok((all_operations, delta.vector, merged_workspaces))
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
        assert!(!config.auto_merge.enabled);
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

    #[test]
    fn test_auto_merge_config_default() {
        let config = AutoMergeConfig::default();

        assert!(config.enabled);
        assert_eq!(config.orthogonal_threshold, 0.1);
        assert_eq!(config.max_merge_batch, 10);
        assert_eq!(config.merge_window_ms, 100);
    }

    #[test]
    fn test_auto_merge_config_disabled() {
        let config = AutoMergeConfig::disabled();

        assert!(!config.enabled);
    }

    #[test]
    fn test_auto_merge_config_builder() {
        let config = AutoMergeConfig::default()
            .with_threshold(0.2)
            .with_max_batch(5)
            .with_window(50);

        assert!(config.enabled);
        assert_eq!(config.orthogonal_threshold, 0.2);
        assert_eq!(config.max_merge_batch, 5);
        assert_eq!(config.merge_window_ms, 50);
    }

    #[test]
    fn test_chain_config_with_auto_merge_config() {
        let auto_merge = AutoMergeConfig::default()
            .with_threshold(0.15)
            .with_max_batch(20);

        let config = ChainConfig::new("node1").with_auto_merge_config(auto_merge);

        assert!(config.auto_merge.enabled);
        assert_eq!(config.auto_merge.orthogonal_threshold, 0.15);
        assert_eq!(config.auto_merge.max_merge_batch, 20);
    }

    #[test]
    fn test_commit_with_auto_merge_disabled() {
        let config = ChainConfig::new("node1").with_auto_merge(false);
        let store = TensorStore::new();
        let chain = TensorChain::with_config(store, config);
        chain.initialize().unwrap();

        // Create a transaction
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: "test".to_string(),
            data: vec![1, 2, 3],
        })
        .unwrap();

        let hash = chain.commit(tx).unwrap();

        assert_eq!(chain.height(), 1);
        assert_eq!(chain.tip_hash(), hash);
    }

    #[test]
    fn test_commit_preserves_block_embedding() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Create transaction with explicit delta embedding
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: "key1".to_string(),
            data: vec![1],
        })
        .unwrap();

        // Set delta embedding
        tx.set_before_embedding(vec![0.0; 128]);
        tx.compute_delta(vec![1.0; 128]); // Delta: all 1s

        chain.commit(tx).unwrap();

        // Check the block has a non-zero embedding
        let block = chain.get_tip().unwrap().unwrap();
        let embedding = &block.header.delta_embedding;
        assert!(!embedding.is_empty());
    }

    #[test]
    fn test_tensor_chain_node_id() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "my_node_id");

        assert_eq!(chain.node_id(), "my_node_id");
    }

    #[test]
    fn test_tensor_chain_active_transactions() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        assert_eq!(chain.active_transactions(), 0);

        let tx1 = chain.begin().unwrap();
        assert_eq!(chain.active_transactions(), 1);

        let tx2 = chain.begin().unwrap();
        assert_eq!(chain.active_transactions(), 2);

        chain.rollback(tx1).unwrap();
        assert_eq!(chain.active_transactions(), 1);

        chain.rollback(tx2).unwrap();
        assert_eq!(chain.active_transactions(), 0);
    }

    #[test]
    fn test_tensor_chain_store_accessor() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        // Access the store
        let _store = chain.store();
        // Just verify we can access it
    }

    #[test]
    fn test_tensor_chain_graph_accessor() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        // Access the graph
        let _graph = chain.graph();
        // Just verify we can access it
    }

    #[test]
    fn test_tensor_chain_append_block() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Create a block manually
        let genesis = chain.get_genesis().unwrap().unwrap();
        let block = chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: "key".to_string(),
                data: vec![1],
            })
            .build();

        // Check the block height
        assert_eq!(block.header.height, 1);
        assert_eq!(block.header.prev_hash, genesis.hash());

        let hash = chain.append_block(block).unwrap();
        assert_eq!(chain.height(), 1);
        assert_eq!(chain.tip_hash(), hash);
    }

    #[test]
    fn test_tensor_chain_new_block() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        let builder = chain.new_block();
        let block = builder.build();

        assert_eq!(block.header.height, 1);
    }

    #[test]
    fn test_tensor_chain_iter() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Add a few blocks
        for i in 1..=3 {
            let tx = chain.begin().unwrap();
            tx.add_operation(Transaction::Put {
                key: format!("key{}", i),
                data: vec![i as u8],
            })
            .unwrap();
            chain.commit(tx).unwrap();
        }

        // Iterate over blocks
        let blocks: Vec<_> = chain.iter().collect();
        assert_eq!(blocks.len(), 4); // Genesis + 3 blocks
    }

    #[test]
    fn test_tensor_chain_get_blocks_range() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Add blocks
        for i in 1..=5 {
            let tx = chain.begin().unwrap();
            tx.add_operation(Transaction::Put {
                key: format!("key{}", i),
                data: vec![i as u8],
            })
            .unwrap();
            chain.commit(tx).unwrap();
        }

        // Get a range
        let blocks = chain.get_blocks(1, 3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].header.height, 1);
        assert_eq!(blocks[1].header.height, 2);
        assert_eq!(blocks[2].header.height, 3);
    }

    #[test]
    fn test_tensor_chain_get_block_nonexistent() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Height 1 doesn't exist yet
        let block = chain.get_block(1).unwrap();
        assert!(block.is_none());

        // Height 100 definitely doesn't exist
        let block = chain.get_block(100).unwrap();
        assert!(block.is_none());
    }

    #[test]
    fn test_auto_merge_config_debug_clone() {
        let config = AutoMergeConfig::default();
        let cloned = config.clone();
        assert_eq!(config.enabled, cloned.enabled);
        assert_eq!(config.orthogonal_threshold, cloned.orthogonal_threshold);

        let debug = format!("{:?}", config);
        assert!(debug.contains("AutoMergeConfig"));
    }

    #[test]
    fn test_chain_config_debug_clone() {
        let config = ChainConfig::new("node1");
        let cloned = config.clone();
        assert_eq!(config.node_id, cloned.node_id);
        assert_eq!(config.max_txs_per_block, cloned.max_txs_per_block);

        let debug = format!("{:?}", config);
        assert!(debug.contains("ChainConfig"));
    }

    #[test]
    fn test_chain_config_default() {
        let config = ChainConfig::default();

        // Node ID should be a UUID
        assert!(!config.node_id.is_empty());
        assert_eq!(config.max_txs_per_block, 1000);
        assert_eq!(config.conflict_threshold, 0.7);
        assert!(config.auto_merge.enabled);
    }

    #[test]
    fn test_auto_merge_orthogonal_transactions() {
        let config = ChainConfig::new("node1").with_auto_merge(true);
        let store = TensorStore::new();
        let chain = TensorChain::with_config(store, config);
        chain.initialize().unwrap();

        // Create first transaction with orthogonal delta (X direction)
        let tx1 = chain.begin().unwrap();
        tx1.add_operation(Transaction::Put {
            key: "key1".to_string(),
            data: vec![1],
        })
        .unwrap();
        tx1.set_before_embedding(vec![0.0, 0.0, 0.0, 0.0]);
        tx1.compute_delta(vec![1.0, 0.0, 0.0, 0.0]);

        // Create second transaction with orthogonal delta (Y direction)
        let tx2 = chain.begin().unwrap();
        tx2.add_operation(Transaction::Put {
            key: "key2".to_string(),
            data: vec![2],
        })
        .unwrap();
        tx2.set_before_embedding(vec![0.0, 0.0, 0.0, 0.0]);
        tx2.compute_delta(vec![0.0, 1.0, 0.0, 0.0]);

        // Commit tx1 - should also merge tx2 since they're orthogonal
        let _hash = chain.commit(tx1).unwrap();

        // Both should be merged - only 1 block created, 0 active transactions
        assert_eq!(chain.height(), 1);
        assert_eq!(chain.active_transactions(), 0);
    }

    #[test]
    fn test_tensor_chain_get_tip() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        let tip = chain.get_tip().unwrap().unwrap();
        assert_eq!(tip.header.height, 0);

        // Add a block
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: "key".to_string(),
            data: vec![1],
        })
        .unwrap();
        chain.commit(tx).unwrap();

        let tip = chain.get_tip().unwrap().unwrap();
        assert_eq!(tip.header.height, 1);
    }
}
