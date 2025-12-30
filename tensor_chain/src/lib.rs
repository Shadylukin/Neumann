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
    DeltaBatch, DeltaReplicationConfig, DeltaReplicationManager, DeltaUpdate, QuantizedDeltaUpdate,
    ReplicationStats,
};
pub use distributed_tx::{
    AbortRequest, CommitRequest, DistributedTransaction, DistributedTxConfig,
    DistributedTxCoordinator, DistributedTxStats, KeyLock, LockManager, PrepareRequest,
    PrepareVote, PreparedTx, ShardId, TxParticipant, TxPhase, TxResponse,
};
pub use error::{ChainError, Result};
pub use membership::{
    ClusterConfig, ClusterView, HealthConfig, LocalNodeConfig, MembershipCallback,
    MembershipManager, NodeHealth, NodeStatus, PeerNodeConfig,
};
pub use network::{
    AppendEntries, AppendEntriesResponse, LogEntry, MemoryTransport, Message, MessageHandler,
    NetworkManager, PeerConfig, RequestVote, RequestVoteResponse, Transport, TxAbortMsg, TxAckMsg,
    TxCommitMsg, TxHandler, TxPrepareMsg, TxPrepareResponseMsg, TxVote,
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

use std::sync::Arc;

use graph_engine::GraphEngine;
use tensor_store::TensorStore;
use tokio::sync::broadcast;

/// Handle for a running Raft consensus node.
///
/// Provides methods to interact with and shut down the Raft node.
/// The node runs in a background tokio task.
pub struct RaftHandle {
    /// Shutdown signal sender.
    shutdown_tx: broadcast::Sender<()>,
    /// Join handle for the background task.
    join_handle: tokio::task::JoinHandle<Result<()>>,
    /// Node ID of this Raft instance.
    node_id: NodeId,
}

impl RaftHandle {
    /// Create a new RaftHandle by spawning the RaftNode's run loop.
    pub fn spawn(node: Arc<RaftNode>) -> Self {
        let node_id = node.node_id().to_string();
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);

        let join_handle = tokio::spawn(async move { node.run(shutdown_rx).await });

        Self {
            shutdown_tx,
            join_handle,
            node_id,
        }
    }

    /// Get the node ID.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Signal the Raft node to shut down gracefully.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Check if the Raft node task has finished.
    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
    }

    /// Wait for the Raft node to finish and return its result.
    pub async fn join(self) -> Result<()> {
        match self.join_handle.await {
            Ok(result) => result,
            Err(e) => Err(ChainError::ConsensusError(format!(
                "Raft task panicked: {}",
                e
            ))),
        }
    }

    /// Shut down and wait for the Raft node to finish.
    pub async fn shutdown_and_wait(self) -> Result<()> {
        self.shutdown();
        self.join().await
    }
}

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

    /// Codebook manager for state quantization and validation.
    codebook_manager: CodebookManager,

    /// Transition validator for semantic state validation.
    transition_validator: TransitionValidator,
}

impl TensorChain {
    /// Create a new TensorChain with the given store.
    pub fn new(store: TensorStore, node_id: impl Into<NodeId>) -> Self {
        let graph = Arc::new(GraphEngine::with_store(store));
        let config = ChainConfig::new(node_id);
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        // Initialize codebook with default dimension (matches delta embedding size)
        let global_codebook = GlobalCodebook::new(4);
        let codebook_manager = CodebookManager::with_global(global_codebook.clone());
        let transition_validator = TransitionValidator::with_global(Arc::new(global_codebook));

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
            codebook_manager,
            transition_validator,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(store: TensorStore, config: ChainConfig) -> Self {
        let graph = Arc::new(GraphEngine::with_store(store));
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        // Initialize codebook with default dimension
        let global_codebook = GlobalCodebook::new(4);
        let codebook_manager = CodebookManager::with_global(global_codebook.clone());
        let transition_validator = TransitionValidator::with_global(Arc::new(global_codebook));

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
            codebook_manager,
            transition_validator,
        }
    }

    /// Create with custom codebook configuration.
    pub fn with_codebook(
        store: TensorStore,
        config: ChainConfig,
        global_codebook: GlobalCodebook,
        codebook_config: CodebookConfig,
        validation_config: ValidationConfig,
    ) -> Self {
        let graph = Arc::new(GraphEngine::with_store(store));
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        let codebook_manager = CodebookManager::new(global_codebook.clone(), codebook_config);
        let transition_validator =
            TransitionValidator::new(Arc::new(global_codebook), validation_config);

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
            codebook_manager,
            transition_validator,
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

    /// Get the codebook manager.
    pub fn codebook_manager(&self) -> &CodebookManager {
        &self.codebook_manager
    }

    /// Get the transition validator.
    pub fn transition_validator(&self) -> &TransitionValidator {
        &self.transition_validator
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
        let (merged_operations, merged_delta, merged_workspaces) = if self.config.auto_merge.enabled
        {
            self.find_and_merge_orthogonal(&workspace, delta)?
        } else {
            (operations, workspace.delta_embedding(), vec![])
        };

        // Quantize delta embedding using codebook
        let quantized_codes = if !merged_delta.is_empty() {
            if let Some((code, _similarity)) =
                self.codebook_manager.global().quantize(&merged_delta)
            {
                vec![code as u16]
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // Build the block with merged operations, delta embedding, and quantized codes
        let block = self
            .chain
            .new_block()
            .add_transactions(merged_operations)
            .with_embedding(merged_delta)
            .with_codes(quantized_codes)
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
            },
            Err(e) => {
                workspace.mark_failed();
                Err(e)
            },
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
        let candidates = self
            .tx_manager
            .find_merge_candidates(workspace, self.config.auto_merge.orthogonal_threshold);

        // Limit to max_merge_batch
        let max_merge = self.config.auto_merge.max_merge_batch;
        let candidates_to_merge: Vec<_> = candidates.into_iter().take(max_merge).collect();

        // Track original delta for validation (convert to dense for compatibility)
        let dim = delta.sparse().dimension();
        let original_delta = delta.to_dense(dim);

        // Merge each orthogonal candidate
        for candidate in candidates_to_merge {
            // Mark candidate as committing
            if candidate.workspace.mark_committing().is_err() {
                continue; // Skip if can't mark as committing
            }

            // Tentatively merge delta
            let tentative_delta = delta.add(&candidate.delta);

            // Validate merged state using transition validator (if codebook has entries)
            // Skip validation if codebook is empty (learning mode)
            if !self.codebook_manager.global().is_empty() {
                let tentative_dim = tentative_delta.sparse().dimension();
                let validation = self.transition_validator.validate_transition(
                    "chain",
                    &original_delta,
                    &tentative_delta.to_dense(tentative_dim),
                );

                if !validation.is_valid {
                    // Reject merge - rollback candidate
                    candidate.workspace.mark_failed();
                    continue;
                }
            }

            // Accept merge
            all_operations.extend(candidate.workspace.operations());
            delta = tentative_delta;

            // Track merged workspace
            merged_workspaces.push(candidate.workspace);
        }

        // Convert sparse delta to dense for return (Phase 4 will make this sparse)
        let final_dim = delta.sparse().dimension();
        Ok((all_operations, delta.to_dense(final_dim), merged_workspaces))
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

    /// Save global codebook entries to TensorStore.
    ///
    /// Stores each centroid with key pattern `_codebook:global:{entry_id}`.
    pub fn save_global_codebook(&self) -> Result<usize> {
        use tensor_store::{ScalarValue, TensorData, TensorValue};

        let mut count = 0;
        for entry in self.codebook_manager.global().iter() {
            let key = format!("_codebook:global:{}", entry.id());
            let mut data = TensorData::new();

            // Store centroid as vector
            data.set("_embedding", TensorValue::Vector(entry.centroid().to_vec()));

            // Store metadata
            data.set(
                "id",
                TensorValue::Scalar(ScalarValue::Int(entry.id() as i64)),
            );
            data.set(
                "magnitude",
                TensorValue::Scalar(ScalarValue::Float(entry.magnitude() as f64)),
            );
            data.set(
                "access_count",
                TensorValue::Scalar(ScalarValue::Int(entry.access_count() as i64)),
            );
            if let Some(label) = entry.label() {
                data.set(
                    "label",
                    TensorValue::Scalar(ScalarValue::String(String::from(label))),
                );
            }

            self.graph
                .store()
                .put(&key, data)
                .map_err(|e| ChainError::StorageError(e.to_string()))?;
            count += 1;
        }

        // Store metadata about the codebook
        let meta_key = "_codebook:global:_meta";
        let mut meta = TensorData::new();
        meta.set(
            "entry_count",
            TensorValue::Scalar(ScalarValue::Int(count as i64)),
        );
        meta.set(
            "dimension",
            TensorValue::Scalar(ScalarValue::Int(
                self.codebook_manager.global().dimension() as i64
            )),
        );
        self.graph
            .store()
            .put(meta_key, meta)
            .map_err(|e| ChainError::StorageError(e.to_string()))?;

        Ok(count)
    }

    /// Load global codebook from TensorStore.
    ///
    /// Returns a GlobalCodebook constructed from stored entries.
    pub fn load_global_codebook(&self) -> Result<Option<GlobalCodebook>> {
        use tensor_store::TensorValue;

        // First check if metadata exists
        let meta_key = "_codebook:global:_meta";
        let meta = match self.graph.store().get(meta_key) {
            Ok(m) => m,
            Err(_) => return Ok(None),
        };

        let entry_count = match meta.get("entry_count") {
            Some(TensorValue::Scalar(tensor_store::ScalarValue::Int(n))) => *n as usize,
            _ => return Ok(None),
        };

        if entry_count == 0 {
            return Ok(None);
        }

        // Load all centroids
        let mut centroids = Vec::with_capacity(entry_count);
        for id in 0..entry_count {
            let key = format!("_codebook:global:{}", id);
            if let Ok(data) = self.graph.store().get(&key) {
                if let Some(TensorValue::Vector(centroid)) = data.get("_embedding") {
                    centroids.push(centroid.clone());
                }
            }
        }

        if centroids.is_empty() {
            return Ok(None);
        }

        Ok(Some(GlobalCodebook::from_centroids(centroids)))
    }

    /// Create a TensorChain, loading existing codebook from store if available.
    ///
    /// This is the recommended constructor for production use, as it preserves
    /// learned codebooks across restarts.
    pub fn load_or_create(store: TensorStore, config: ChainConfig) -> Self {
        let graph = Arc::new(GraphEngine::with_store(store));
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        // Try to load existing codebook from store
        let global_codebook = Self::try_load_codebook_from_store(graph.store())
            .unwrap_or_else(|| GlobalCodebook::new(4));

        let codebook_manager = CodebookManager::with_global(global_codebook.clone());
        let transition_validator = TransitionValidator::with_global(Arc::new(global_codebook));

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
            codebook_manager,
            transition_validator,
        }
    }

    /// Helper to load codebook from a TensorStore.
    fn try_load_codebook_from_store(store: &TensorStore) -> Option<GlobalCodebook> {
        use tensor_store::TensorValue;

        let meta_key = "_codebook:global:_meta";
        let meta = store.get(meta_key).ok()?;

        let entry_count = match meta.get("entry_count") {
            Some(TensorValue::Scalar(tensor_store::ScalarValue::Int(n))) => *n as usize,
            _ => return None,
        };

        if entry_count == 0 {
            return None;
        }

        let mut centroids = Vec::with_capacity(entry_count);
        for id in 0..entry_count {
            let key = format!("_codebook:global:{}", id);
            if let Ok(data) = store.get(&key) {
                if let Some(TensorValue::Vector(centroid)) = data.get("_embedding") {
                    centroids.push(centroid.clone());
                }
            }
        }

        if centroids.is_empty() {
            return None;
        }

        Some(GlobalCodebook::from_centroids(centroids))
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

    #[test]
    fn test_codebook_manager_accessor() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        let manager = chain.codebook_manager();
        // Default codebook has dimension 4
        assert_eq!(manager.global().dimension(), 4);
    }

    #[test]
    fn test_transition_validator_accessor() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        let validator = chain.transition_validator();
        // Verify we can use the validator (empty codebook, so no valid centroids)
        let validation = validator.validate_state("test_domain", &[1.0, 0.0, 0.0, 0.0]);
        // With empty codebook, similarity is 0.0 (below threshold)
        assert_eq!(validation.global_similarity, 0.0);
    }

    #[test]
    fn test_save_and_load_global_codebook() {
        let store = TensorStore::new();

        // Create chain with custom codebook
        let centroids = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        let global_codebook = GlobalCodebook::from_centroids(centroids);
        let config = ChainConfig::new("node1");

        let chain = TensorChain::with_codebook(
            store,
            config,
            global_codebook,
            CodebookConfig::default(),
            ValidationConfig::default(),
        );

        // Save codebook
        let count = chain.save_global_codebook().unwrap();
        assert_eq!(count, 3);

        // Load it back
        let loaded = chain.load_global_codebook().unwrap().unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.dimension(), 4);
    }

    #[test]
    fn test_load_global_codebook_empty_store() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        // No codebook saved - should return None
        let loaded = chain.load_global_codebook().unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn test_load_or_create_with_empty_store() {
        let store = TensorStore::new();
        let config = ChainConfig::new("node1");

        let chain = TensorChain::load_or_create(store, config);

        // Should create default codebook (empty, dimension 4)
        assert_eq!(chain.codebook_manager().global().dimension(), 4);
        assert_eq!(chain.codebook_manager().global().len(), 0);
    }

    #[test]
    fn test_load_or_create_with_existing_codebook() {
        let store = TensorStore::new();

        // First create and save a codebook
        let centroids = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        let global_codebook = GlobalCodebook::from_centroids(centroids);
        let config = ChainConfig::new("node1");

        let chain1 = TensorChain::with_codebook(
            store.clone(),
            config.clone(),
            global_codebook,
            CodebookConfig::default(),
            ValidationConfig::default(),
        );
        chain1.save_global_codebook().unwrap();

        // Now load_or_create should find the existing codebook
        let chain2 = TensorChain::load_or_create(store, config);

        assert_eq!(chain2.codebook_manager().global().len(), 2);
        assert_eq!(chain2.codebook_manager().global().dimension(), 4);
    }

    #[test]
    fn test_commit_quantizes_delta() {
        // Create a codebook with known centroids
        let centroids = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let global_codebook = GlobalCodebook::from_centroids(centroids);
        let config = ChainConfig::new("node1").with_auto_merge(false);
        let store = TensorStore::new();

        let chain = TensorChain::with_codebook(
            store,
            config,
            global_codebook,
            CodebookConfig::default(),
            ValidationConfig::default(),
        );
        chain.initialize().unwrap();

        // Create transaction with delta close to centroid 0
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: "test".to_string(),
            data: vec![1],
        })
        .unwrap();
        tx.set_before_embedding(vec![0.0, 0.0, 0.0, 0.0]);
        tx.compute_delta(vec![0.9, 0.1, 0.0, 0.0]); // Close to [1,0,0,0]

        chain.commit(tx).unwrap();

        // Check block has quantized codes
        let block = chain.get_tip().unwrap().unwrap();
        assert!(!block.header.quantized_codes.is_empty());
        assert_eq!(block.header.quantized_codes[0], 0); // Should quantize to centroid 0
    }
}
