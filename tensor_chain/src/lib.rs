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

pub mod atomic_io;
pub mod block;
pub mod chain;
pub mod cluster;
pub mod codebook;
pub mod consensus;
pub mod deadlock;
pub mod delta_replication;
pub mod distributed_tx;
pub mod embedding;
pub mod error;
pub mod geometric_membership;
pub mod gossip;
pub mod membership;
pub mod message_validation;
pub mod metrics;
pub mod network;
pub mod partition_merge;
pub mod raft;
pub mod raft_wal;
pub mod signing;
pub mod snapshot_buffer;
pub mod snapshot_streaming;
pub mod state_machine;
pub mod tcp;
pub mod transaction;
pub mod tx_id;
pub mod tx_wal;
pub mod validation;

// Re-exports
use std::sync::Arc;

pub use atomic_io::{atomic_truncate, atomic_write, AtomicIoError, AtomicWriter};
pub use block::{Block, BlockHash, BlockHeader, NodeId, Transaction, ValidatorSignature};
pub use chain::{BlockBuilder, Chain, ChainIterator};
pub use cluster::{
    ClusterOrchestrator, LocalNodeConfig as ClusterNodeConfig, OrchestratorConfig,
    PeerConfig as ClusterPeerConfig,
};
pub use codebook::{
    CodebookConfig, CodebookEntry, CodebookManager, GlobalCodebook, HierarchicalQuantization,
    LocalCodebook, LocalCodebookStats, PruningStrategy,
};
pub use consensus::{
    BatchConflict, ConflictClass, ConflictResult, ConsensusConfig, ConsensusManager, DeltaVector,
    MergeAction, MergeResult,
};
pub use deadlock::{
    DeadlockDetector, DeadlockDetectorConfig, DeadlockInfo, DeadlockStats, DeadlockStatsSnapshot,
    VictimSelectionPolicy, WaitForGraph, WaitInfo,
};
pub use delta_replication::{
    DeltaBatch, DeltaReplicationConfig, DeltaReplicationManager, DeltaUpdate, ReplicationStats,
    ReplicationStatsSnapshot,
};
pub use distributed_tx::{
    AbortRequest, CommitRequest, CoordinatorState, DistributedTransaction, DistributedTxConfig,
    DistributedTxCoordinator, DistributedTxStats, DistributedTxStatsSnapshot, EpochMillis, KeyLock,
    LockManager, ParticipantState, PrepareRequest, PrepareVote, PreparedTx, RecoveryStats,
    SerializableLockState, ShardId, TxParticipant, TxPhase, TxResponse,
};
pub use embedding::{EmbeddingError, EmbeddingState};
pub use error::{ChainError, Result};
pub use geometric_membership::{GeometricMembershipConfig, GeometricMembershipManager, RankedPeer};
pub use gossip::{
    GossipConfig, GossipMembershipManager, GossipMessage, GossipNodeState, LWWMembershipState,
};
use graph_engine::GraphEngine;
pub use membership::{
    ClusterConfig, ClusterView, HealthConfig, LocalNodeConfig, MembershipCallback,
    MembershipManager, MembershipStats, MembershipStatsSnapshot, NodeHealth, NodeStatus,
    PartitionStatus, PeerNodeConfig,
};
pub use message_validation::{
    CompositeValidator, EmbeddingValidator, MessageValidationConfig, MessageValidator,
};
pub use metrics::{TimingSnapshot, TimingStats};
pub use network::{
    AppendEntries, AppendEntriesResponse, BlockRequest, BlockResponse, ConfigChange,
    DataMergeRequest, DataMergeResponse, GeometricTransport, JointConfig, LogEntry, LogEntryData,
    MemoryTransport, MergeAck, MergeDeltaEntry, MergeFinalize, MergeInit, MergeOpType,
    MergeViewExchange, Message, MessageHandler, NetworkManager, PeerConfig, PreVote,
    PreVoteResponse, QueryExecutor, QueryHandler, QueryRequest, QueryResponse,
    RaftMembershipConfig, RequestVote, RequestVoteResponse, SnapshotRequest, SnapshotResponse,
    TimeoutNow, Transport, TxAbortMsg, TxAckMsg, TxCommitMsg, TxHandler, TxPrepareMsg,
    TxPrepareResponseMsg, TxReconcileRequest, TxReconcileResponse, TxVote,
};
pub use partition_merge::{
    ConflictResolution, ConflictType, DataReconcileResult, DataReconciler, MembershipReconciler,
    MembershipViewSummary, MergeConflict, MergePhase, MergeSession, PartitionMergeConfig,
    PartitionMergeManager, PartitionMergeStats, PartitionMergeStatsSnapshot, PartitionStateSummary,
    PendingTxState, TransactionReconciler, TxReconcileResult,
};
pub use raft::{
    FastPathState, FastPathStats, HeartbeatStats, HeartbeatStatsSnapshot, QuorumTracker,
    RaftConfig, RaftNode, RaftState, RaftStats, RaftStatsSnapshot, SnapshotMetadata, TransferState,
};
pub use raft_wal::{RaftRecoveryState, RaftWal, RaftWalEntry};
pub use signing::{
    Identity, PublicIdentity, SequenceTracker, SequenceTrackerConfig, SignedGossipMessage,
    SignedMessage, ValidatorRegistry,
};
pub use snapshot_buffer::{SnapshotBuffer, SnapshotBufferConfig, SnapshotBufferError};
pub use snapshot_streaming::{
    deserialize_entries, serialize_entries, SnapshotReader, SnapshotWriter, StreamingError,
};
pub use state_machine::TensorStateMachine;
pub use tcp::{
    Handshake, LengthDelimitedCodec, NodeIdVerification, ReconnectConfig, SecurityConfig,
    SecurityMode, TcpError, TcpResult, TcpTransport, TcpTransportConfig, TlsConfig, TransportStats,
};
use tensor_store::TensorStore;
use tokio::sync::broadcast;
pub use transaction::{
    TransactionDelta, TransactionManager, TransactionState, TransactionWorkspace,
};
pub use tx_id::{extract_timestamp_hint, generate_tx_id, is_plausible_tx_id};
pub use tx_wal::{
    PrepareVoteKind, RecoveredPreparedTx, TxOutcome, TxRecoveryState, TxWal, TxWalEntry,
};
pub use validation::{
    FastPathResult, FastPathValidator, StateValidation, TransitionValidation, TransitionValidator,
    ValidationConfig, ValidationMode,
};

/// Aggregated metrics from all tensor_chain components.
///
/// Provides a unified interface to collect and snapshot metrics from:
/// - Raft consensus (elections, heartbeats, quorum)
/// - Distributed transactions (2PC timing, lock contention)
/// - Membership (health checks, partition status)
/// - Replication (bandwidth, compression ratio)
#[derive(Debug)]
pub struct ChainMetrics {
    /// Raft consensus metrics.
    pub raft: Arc<RaftStats>,
    /// Distributed transaction metrics.
    pub dtx: Arc<DistributedTxStats>,
    /// Membership and health check metrics.
    pub membership: Arc<MembershipStats>,
    /// Delta replication metrics.
    pub replication: Arc<ReplicationStats>,
}

impl ChainMetrics {
    pub fn new() -> Self {
        Self {
            raft: Arc::new(RaftStats::new()),
            dtx: Arc::new(DistributedTxStats::new()),
            membership: Arc::new(MembershipStats::new()),
            replication: Arc::new(ReplicationStats::new()),
        }
    }

    pub fn from_components(
        raft: Arc<RaftStats>,
        dtx: Arc<DistributedTxStats>,
        membership: Arc<MembershipStats>,
        replication: Arc<ReplicationStats>,
    ) -> Self {
        Self {
            raft,
            dtx,
            membership,
            replication,
        }
    }

    /// Take a point-in-time snapshot of all metrics.
    pub fn snapshot(&self) -> ChainMetricsSnapshot {
        ChainMetricsSnapshot {
            raft: self.raft.snapshot(),
            dtx: self.dtx.snapshot(),
            membership: self.membership.snapshot(),
            replication: self.replication.snapshot(),
        }
    }

    /// Emit metrics as structured logs.
    ///
    /// This is useful for periodic metrics reporting to log aggregation systems.
    pub fn emit_as_logs(&self) {
        let s = self.snapshot();
        tracing::info!(
            target: "tensor_chain::metrics",
            // Raft metrics
            raft_heartbeat_successes = s.raft.heartbeat_successes,
            raft_heartbeat_failures = s.raft.heartbeat_failures,
            raft_fast_path_accepted = s.raft.fast_path_accepted,
            raft_fast_path_rejected = s.raft.fast_path_rejected,
            raft_quorum_lost_events = s.raft.quorum_lost_events,
            raft_leader_step_downs = s.raft.leader_step_downs,
            // DTX metrics
            dtx_started = s.dtx.started,
            dtx_committed = s.dtx.committed,
            dtx_aborted = s.dtx.aborted,
            dtx_timed_out = s.dtx.timed_out,
            dtx_conflicts = s.dtx.conflicts,
            // Membership metrics
            membership_health_checks = s.membership.health_checks,
            membership_health_check_failures = s.membership.health_check_failures,
            membership_quorum_lost_events = s.membership.quorum_lost_events,
            // Replication metrics
            replication_updates_sent = s.replication.updates_sent,
            replication_bytes_sent = s.replication.bytes_sent,
            replication_bytes_saved = s.replication.bytes_saved,
            "Chain metrics snapshot"
        );
    }
}

impl Default for ChainMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ChainMetrics {
    fn clone(&self) -> Self {
        Self {
            raft: Arc::clone(&self.raft),
            dtx: Arc::clone(&self.dtx),
            membership: Arc::clone(&self.membership),
            replication: Arc::clone(&self.replication),
        }
    }
}

/// Point-in-time snapshot of all chain metrics.
///
/// This struct is serializable and can be exported to monitoring systems.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ChainMetricsSnapshot {
    /// Raft consensus metrics snapshot.
    pub raft: RaftStatsSnapshot,
    /// Distributed transaction metrics snapshot.
    pub dtx: DistributedTxStatsSnapshot,
    /// Membership metrics snapshot.
    pub membership: MembershipStatsSnapshot,
    /// Replication metrics snapshot.
    pub replication: ReplicationStatsSnapshot,
}

impl ChainMetricsSnapshot {
    pub fn is_empty(&self) -> bool {
        self.raft.fast_path_accepted == 0
            && self.raft.heartbeat_successes == 0
            && self.dtx.started == 0
            && self.membership.health_checks == 0
            && self.replication.updates_sent == 0
    }

    pub fn total_heartbeats(&self) -> u64 {
        self.raft.heartbeat_successes + self.raft.heartbeat_failures
    }

    /// Calculate heartbeat success rate (0.0 - 1.0).
    pub fn heartbeat_success_rate(&self) -> f64 {
        let total = self.total_heartbeats();
        if total == 0 {
            1.0 // No heartbeats means no failures
        } else {
            self.raft.heartbeat_successes as f64 / total as f64
        }
    }

    /// Calculate transaction commit rate (0.0 - 1.0).
    pub fn tx_commit_rate(&self) -> f64 {
        if self.dtx.started == 0 {
            1.0
        } else {
            self.dtx.committed as f64 / self.dtx.started as f64
        }
    }

    /// Calculate health check success rate (0.0 - 1.0).
    pub fn health_check_success_rate(&self) -> f64 {
        if self.membership.health_checks == 0 {
            1.0
        } else {
            let failures = self.membership.health_check_failures;
            let successes = self.membership.health_checks.saturating_sub(failures);
            successes as f64 / self.membership.health_checks as f64
        }
    }

    /// Check if the cluster is healthy based on metrics.
    ///
    /// Returns true if:
    /// - Heartbeat success rate > 0.9
    /// - No quorum lost events
    /// - Health check success rate > 0.9
    pub fn is_cluster_healthy(&self) -> bool {
        self.heartbeat_success_rate() > 0.9
            && self.raft.quorum_lost_events == 0
            && self.health_check_success_rate() > 0.9
    }
}

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

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    pub fn is_finished(&self) -> bool {
        self.join_handle.is_finished()
    }

    pub async fn join(self) -> Result<()> {
        match self.join_handle.await {
            Ok(result) => result,
            Err(e) => Err(ChainError::ConsensusError(format!(
                "Raft task panicked: {}",
                e
            ))),
        }
    }

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
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.orthogonal_threshold = threshold;
        self
    }

    pub fn with_max_batch(mut self, max: usize) -> Self {
        self.max_merge_batch = max;
        self
    }

    pub fn with_window(mut self, ms: u64) -> Self {
        self.merge_window_ms = ms;
        self
    }
}

/// Configuration for geometric routing in distributed operations.
#[derive(Debug, Clone)]
pub struct GeometricRoutingConfig {
    /// Enable geometric routing based on embeddings.
    pub enabled: bool,
    /// Minimum similarity threshold for geometric routing (0.0-1.0).
    /// Below this threshold, falls back to hash-based routing.
    pub min_similarity: f32,
    /// Enable fallback to hash-based routing when geometric routing fails.
    pub fallback_to_hash: bool,
}

impl Default for GeometricRoutingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_similarity: 0.5,
            fallback_to_hash: true,
        }
    }
}

impl GeometricRoutingConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    pub fn with_min_similarity(mut self, threshold: f32) -> Self {
        self.min_similarity = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn without_fallback(mut self) -> Self {
        self.fallback_to_hash = false;
        self
    }
}

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
    /// Geometric routing configuration.
    pub geometric_routing: GeometricRoutingConfig,
}

impl Default for ChainConfig {
    fn default() -> Self {
        Self {
            node_id: uuid::Uuid::new_v4().to_string(),
            max_txs_per_block: 1000,
            conflict_threshold: 0.7,
            auto_merge: AutoMergeConfig::default(),
            geometric_routing: GeometricRoutingConfig::default(),
        }
    }
}

impl ChainConfig {
    pub fn new(node_id: impl Into<NodeId>) -> Self {
        Self {
            node_id: node_id.into(),
            ..Default::default()
        }
    }

    pub fn with_max_txs(mut self, max: usize) -> Self {
        self.max_txs_per_block = max;
        self
    }

    pub fn with_conflict_threshold(mut self, threshold: f32) -> Self {
        self.conflict_threshold = threshold;
        self
    }

    pub fn with_auto_merge(mut self, enabled: bool) -> Self {
        self.auto_merge.enabled = enabled;
        self
    }

    pub fn with_auto_merge_config(mut self, config: AutoMergeConfig) -> Self {
        self.auto_merge = config;
        self
    }

    pub fn with_geometric_routing(mut self, config: GeometricRoutingConfig) -> Self {
        self.geometric_routing = config;
        self
    }

    pub fn without_geometric_routing(mut self) -> Self {
        self.geometric_routing.enabled = false;
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

    /// Ed25519 identity for signing blocks.
    identity: signing::Identity,
}

impl TensorChain {
    /// Create a new TensorChain with the given store.
    ///
    /// Generates a new Ed25519 identity for signing blocks. The node_id parameter
    /// is used for chain configuration but the actual cryptographic identity is
    /// derived from the generated key pair.
    pub fn new(store: TensorStore, node_id: impl Into<NodeId>) -> Self {
        use crate::transaction::DEFAULT_EMBEDDING_DIM;

        let graph = Arc::new(GraphEngine::with_store(store));
        let config = ChainConfig::new(node_id);
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        // Initialize codebook with default dimension (matches delta embedding size)
        let global_codebook = GlobalCodebook::new(DEFAULT_EMBEDDING_DIM);
        let codebook_manager = CodebookManager::with_global(global_codebook.clone());
        let transition_validator = TransitionValidator::with_global(Arc::new(global_codebook));

        // Generate a new Ed25519 identity for signing blocks
        let identity = signing::Identity::generate();

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
            codebook_manager,
            transition_validator,
            identity,
        }
    }

    pub fn with_config(store: TensorStore, config: ChainConfig) -> Self {
        use crate::transaction::DEFAULT_EMBEDDING_DIM;

        let graph = Arc::new(GraphEngine::with_store(store));
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        // Initialize codebook with default dimension
        let global_codebook = GlobalCodebook::new(DEFAULT_EMBEDDING_DIM);
        let codebook_manager = CodebookManager::with_global(global_codebook.clone());
        let transition_validator = TransitionValidator::with_global(Arc::new(global_codebook));

        // Generate a new Ed25519 identity for signing blocks
        let identity = signing::Identity::generate();

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
            codebook_manager,
            transition_validator,
            identity,
        }
    }

    pub fn with_identity(
        store: TensorStore,
        config: ChainConfig,
        identity: signing::Identity,
    ) -> Self {
        use crate::transaction::DEFAULT_EMBEDDING_DIM;

        let graph = Arc::new(GraphEngine::with_store(store));
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        // Initialize codebook with default dimension
        let global_codebook = GlobalCodebook::new(DEFAULT_EMBEDDING_DIM);
        let codebook_manager = CodebookManager::with_global(global_codebook.clone());
        let transition_validator = TransitionValidator::with_global(Arc::new(global_codebook));

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
            codebook_manager,
            transition_validator,
            identity,
        }
    }

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

        // Generate a new Ed25519 identity for signing blocks
        let identity = signing::Identity::generate();

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
            codebook_manager,
            transition_validator,
            identity,
        }
    }

    pub fn initialize(&self) -> Result<()> {
        self.chain.initialize()
    }

    pub fn height(&self) -> u64 {
        self.chain.height()
    }

    pub fn tip_hash(&self) -> BlockHash {
        self.chain.tip_hash()
    }

    pub fn node_id(&self) -> &NodeId {
        &self.config.node_id
    }

    pub fn codebook_manager(&self) -> &CodebookManager {
        &self.codebook_manager
    }

    pub fn transition_validator(&self) -> &TransitionValidator {
        &self.transition_validator
    }

    pub fn identity(&self) -> &signing::Identity {
        &self.identity
    }

    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.identity.public_key_bytes()
    }

    pub fn geometric_routing_config(&self) -> &GeometricRoutingConfig {
        &self.config.geometric_routing
    }

    pub fn is_geometric_routing_enabled(&self) -> bool {
        self.config.geometric_routing.enabled
    }

    /// Route a key to a node based on embedding similarity.
    ///
    /// Uses the geometric routing configuration to determine routing.
    /// If geometric routing is disabled or the embedding is empty,
    /// returns the local node as the default route.
    pub fn route_by_embedding(&self, embedding: &tensor_store::SparseVector) -> NodeId {
        if !self.config.geometric_routing.enabled || embedding.dimension() == 0 {
            return self.config.node_id.clone();
        }

        // For now, return local node - actual distributed routing requires
        // integration with SemanticPartitioner or VoronoiPartitioner
        self.config.node_id.clone()
    }

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

        // Build and sign the block with Ed25519 identity
        let block = self
            .chain
            .new_block()
            .add_transactions(merged_operations)
            .with_dense_embedding(&merged_delta)
            .with_codes(quantized_codes)
            .sign_and_build(&self.identity);

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
        let candidates = self.tx_manager.find_merge_candidates(
            workspace,
            self.config.auto_merge.orthogonal_threshold,
            self.config.auto_merge.merge_window_ms,
        );

        // Limit to max_merge_batch
        let max_merge = self.config.auto_merge.max_merge_batch;
        let candidates_to_merge: Vec<_> = candidates.into_iter().take(max_merge).collect();

        // Track original delta for validation (convert to dense for compatibility)
        let dim = delta.dimension();
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
                let tentative_dim = tentative_delta.dimension();
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
        let final_dim = delta.dimension();
        Ok((all_operations, delta.to_dense(final_dim), merged_workspaces))
    }

    pub fn rollback(&self, workspace: Arc<TransactionWorkspace>) -> Result<()> {
        workspace.rollback(self.graph.store())?;
        self.tx_manager.remove(workspace.id());
        Ok(())
    }

    pub fn get_block(&self, height: u64) -> Result<Option<Block>> {
        self.chain.get_block_at(height)
    }

    pub fn get_tip(&self) -> Result<Option<Block>> {
        self.chain.get_tip()
    }

    pub fn get_genesis(&self) -> Result<Option<Block>> {
        self.chain.get_genesis()
    }

    pub fn verify(&self) -> Result<()> {
        self.chain.verify_chain()
    }

    pub fn history(&self, key: &str) -> Result<Vec<(u64, Transaction)>> {
        self.chain.history(key)
    }

    pub fn get_blocks(&self, start: u64, end: u64) -> Result<Vec<Block>> {
        self.chain.get_blocks_range(start, end)
    }

    pub fn iter(&self) -> ChainIterator<'_> {
        self.chain.iter()
    }

    pub fn active_transactions(&self) -> usize {
        self.tx_manager.active_count()
    }

    pub fn store(&self) -> &TensorStore {
        self.graph.store()
    }

    pub fn graph(&self) -> &GraphEngine {
        &self.graph
    }

    pub fn append_block(&self, block: Block) -> Result<BlockHash> {
        self.chain.append(block)
    }

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
        use crate::transaction::DEFAULT_EMBEDDING_DIM;

        let graph = Arc::new(GraphEngine::with_store(store));
        let chain = Chain::new(graph.clone(), config.node_id.clone());

        // Try to load existing codebook from store
        let global_codebook = Self::try_load_codebook_from_store(graph.store())
            .unwrap_or_else(|| GlobalCodebook::new(DEFAULT_EMBEDDING_DIM));

        let codebook_manager = CodebookManager::with_global(global_codebook.clone());
        let transition_validator = TransitionValidator::with_global(Arc::new(global_codebook));

        // Generate a new Ed25519 identity for signing blocks
        let identity = signing::Identity::generate();

        Self {
            chain,
            tx_manager: TransactionManager::new(),
            graph,
            config,
            codebook_manager,
            transition_validator,
            identity,
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
    use tensor_store::SparseVector;

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
        assert!(embedding.nnz() > 0);
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
        use crate::transaction::DEFAULT_EMBEDDING_DIM;

        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        let manager = chain.codebook_manager();
        // Default codebook has dimension matching DEFAULT_EMBEDDING_DIM
        assert_eq!(manager.global().dimension(), DEFAULT_EMBEDDING_DIM);
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
        use crate::transaction::DEFAULT_EMBEDDING_DIM;

        let store = TensorStore::new();
        let config = ChainConfig::new("node1");

        let chain = TensorChain::load_or_create(store, config);

        // Should create default codebook (empty, dimension matches DEFAULT_EMBEDDING_DIM)
        assert_eq!(
            chain.codebook_manager().global().dimension(),
            DEFAULT_EMBEDDING_DIM
        );
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

    #[test]
    fn test_geometric_routing_config_disabled() {
        let config = GeometricRoutingConfig::disabled();
        assert!(!config.enabled);
        assert!((config.min_similarity - 0.5).abs() < 0.001);
        assert!(config.fallback_to_hash);
    }

    #[test]
    fn test_geometric_routing_config_accessors() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        // Access geometric routing config
        let geo_config = chain.geometric_routing_config();
        assert!(geo_config.enabled);
        assert!((geo_config.min_similarity - 0.5).abs() < 0.001);

        // Check is_geometric_routing_enabled
        assert!(chain.is_geometric_routing_enabled());
    }

    #[test]
    fn test_geometric_routing_config_disabled_chain() {
        let config =
            ChainConfig::new("node1").with_geometric_routing(GeometricRoutingConfig::disabled());
        let store = TensorStore::new();
        let chain = TensorChain::with_config(store, config);

        assert!(!chain.is_geometric_routing_enabled());
    }

    #[test]
    fn test_route_by_embedding_disabled() {
        let config =
            ChainConfig::new("node1").with_geometric_routing(GeometricRoutingConfig::disabled());
        let store = TensorStore::new();
        let chain = TensorChain::with_config(store, config);

        let embedding = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let routed = chain.route_by_embedding(&embedding);

        // When disabled, should return local node
        assert_eq!(routed, "node1");
    }

    #[test]
    fn test_route_by_embedding_empty() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        let embedding = SparseVector::from_dense(&[]);
        let routed = chain.route_by_embedding(&embedding);

        // Empty embedding returns local node
        assert_eq!(routed, "node1");
    }

    #[test]
    fn test_route_by_embedding_enabled() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");

        let embedding = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let routed = chain.route_by_embedding(&embedding);

        // Without peers, routes to local node
        assert_eq!(routed, "node1");
    }

    // ChainMetrics tests

    #[test]
    fn test_chain_metrics_new() {
        let metrics = ChainMetrics::new();

        // All stats should be at initial values
        let snapshot = metrics.snapshot();
        assert!(snapshot.is_empty());
    }

    #[test]
    fn test_chain_metrics_default() {
        let metrics = ChainMetrics::default();
        let snapshot = metrics.snapshot();
        assert!(snapshot.is_empty());
    }

    #[test]
    fn test_chain_metrics_clone() {
        let metrics = ChainMetrics::new();

        // Record some data
        metrics
            .raft
            .fast_path_accepted
            .fetch_add(5, std::sync::atomic::Ordering::Relaxed);

        let cloned = metrics.clone();

        // Clone should share the same Arc
        assert_eq!(
            metrics
                .raft
                .fast_path_accepted
                .load(std::sync::atomic::Ordering::Relaxed),
            cloned
                .raft
                .fast_path_accepted
                .load(std::sync::atomic::Ordering::Relaxed)
        );

        // Modifying one should affect both
        metrics
            .raft
            .fast_path_accepted
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            cloned
                .raft
                .fast_path_accepted
                .load(std::sync::atomic::Ordering::Relaxed),
            6
        );
    }

    #[test]
    fn test_chain_metrics_from_components() {
        let raft = Arc::new(RaftStats::new());
        let dtx = Arc::new(DistributedTxStats::new());
        let membership = Arc::new(MembershipStats::new());
        let replication = Arc::new(ReplicationStats::new());

        let metrics = ChainMetrics::from_components(
            Arc::clone(&raft),
            Arc::clone(&dtx),
            Arc::clone(&membership),
            Arc::clone(&replication),
        );

        // Should use the same Arc instances
        raft.fast_path_accepted
            .fetch_add(10, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            metrics
                .raft
                .fast_path_accepted
                .load(std::sync::atomic::Ordering::Relaxed),
            10
        );
    }

    #[test]
    fn test_chain_metrics_snapshot_is_empty() {
        let snapshot = ChainMetricsSnapshot::default();
        assert!(snapshot.is_empty());

        // Create non-empty snapshot
        let metrics = ChainMetrics::new();
        metrics
            .dtx
            .started
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let snapshot = metrics.snapshot();
        assert!(!snapshot.is_empty());
    }

    #[test]
    fn test_chain_metrics_snapshot_total_heartbeats() {
        let metrics = ChainMetrics::new();
        metrics
            .raft
            .heartbeat_successes
            .fetch_add(10, std::sync::atomic::Ordering::Relaxed);
        metrics
            .raft
            .heartbeat_failures
            .fetch_add(2, std::sync::atomic::Ordering::Relaxed);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_heartbeats(), 12);
    }

    #[test]
    fn test_chain_metrics_snapshot_heartbeat_success_rate() {
        let metrics = ChainMetrics::new();

        // No heartbeats - 100% success rate
        let snapshot = metrics.snapshot();
        assert!((snapshot.heartbeat_success_rate() - 1.0).abs() < 0.001);

        // 8 successes, 2 failures = 80%
        metrics
            .raft
            .heartbeat_successes
            .fetch_add(8, std::sync::atomic::Ordering::Relaxed);
        metrics
            .raft
            .heartbeat_failures
            .fetch_add(2, std::sync::atomic::Ordering::Relaxed);
        let snapshot = metrics.snapshot();
        assert!((snapshot.heartbeat_success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_chain_metrics_snapshot_tx_commit_rate() {
        let metrics = ChainMetrics::new();

        // No transactions - 100% commit rate
        let snapshot = metrics.snapshot();
        assert!((snapshot.tx_commit_rate() - 1.0).abs() < 0.001);

        // 7 committed out of 10 started = 70%
        metrics
            .dtx
            .started
            .fetch_add(10, std::sync::atomic::Ordering::Relaxed);
        metrics
            .dtx
            .committed
            .fetch_add(7, std::sync::atomic::Ordering::Relaxed);
        let snapshot = metrics.snapshot();
        assert!((snapshot.tx_commit_rate() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_chain_metrics_snapshot_health_check_success_rate() {
        let metrics = ChainMetrics::new();

        // No health checks - 100% success rate
        let snapshot = metrics.snapshot();
        assert!((snapshot.health_check_success_rate() - 1.0).abs() < 0.001);

        // 9 successes out of 10 (1 failure) = 90%
        metrics
            .membership
            .health_checks
            .fetch_add(10, std::sync::atomic::Ordering::Relaxed);
        metrics
            .membership
            .health_check_failures
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let snapshot = metrics.snapshot();
        assert!((snapshot.health_check_success_rate() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_chain_metrics_snapshot_is_cluster_healthy() {
        let metrics = ChainMetrics::new();

        // Empty metrics - healthy by default
        let snapshot = metrics.snapshot();
        assert!(snapshot.is_cluster_healthy());

        // Add good stats - still healthy
        metrics
            .raft
            .heartbeat_successes
            .fetch_add(100, std::sync::atomic::Ordering::Relaxed);
        metrics
            .raft
            .heartbeat_failures
            .fetch_add(5, std::sync::atomic::Ordering::Relaxed); // 95% success
        metrics
            .membership
            .health_checks
            .fetch_add(100, std::sync::atomic::Ordering::Relaxed);
        metrics
            .membership
            .health_check_failures
            .fetch_add(5, std::sync::atomic::Ordering::Relaxed); // 95% success
        let snapshot = metrics.snapshot();
        assert!(snapshot.is_cluster_healthy());

        // Add quorum lost event - unhealthy
        metrics
            .raft
            .quorum_lost_events
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let snapshot = metrics.snapshot();
        assert!(!snapshot.is_cluster_healthy());
    }

    #[test]
    fn test_chain_metrics_snapshot_unhealthy_heartbeats() {
        let metrics = ChainMetrics::new();

        // 50% heartbeat failure rate - unhealthy
        metrics
            .raft
            .heartbeat_successes
            .fetch_add(50, std::sync::atomic::Ordering::Relaxed);
        metrics
            .raft
            .heartbeat_failures
            .fetch_add(50, std::sync::atomic::Ordering::Relaxed);
        let snapshot = metrics.snapshot();
        assert!(!snapshot.is_cluster_healthy());
    }

    #[test]
    fn test_chain_metrics_snapshot_unhealthy_health_checks() {
        let metrics = ChainMetrics::new();

        // 50% health check failure rate - unhealthy
        metrics
            .membership
            .health_checks
            .fetch_add(100, std::sync::atomic::Ordering::Relaxed);
        metrics
            .membership
            .health_check_failures
            .fetch_add(50, std::sync::atomic::Ordering::Relaxed);
        let snapshot = metrics.snapshot();
        assert!(!snapshot.is_cluster_healthy());
    }

    #[test]
    fn test_chain_metrics_snapshot_serialization() {
        let metrics = ChainMetrics::new();
        metrics
            .raft
            .heartbeat_successes
            .fetch_add(100, std::sync::atomic::Ordering::Relaxed);
        metrics
            .dtx
            .started
            .fetch_add(50, std::sync::atomic::Ordering::Relaxed);

        let snapshot = metrics.snapshot();

        // Serialize with bincode
        let bytes = bincode::serialize(&snapshot).unwrap();
        let restored: ChainMetricsSnapshot = bincode::deserialize(&bytes).unwrap();

        assert_eq!(
            snapshot.raft.heartbeat_successes,
            restored.raft.heartbeat_successes
        );
        assert_eq!(snapshot.dtx.started, restored.dtx.started);
    }

    #[test]
    fn test_chain_metrics_debug() {
        let metrics = ChainMetrics::new();
        let debug = format!("{:?}", metrics);
        assert!(debug.contains("ChainMetrics"));
    }

    #[test]
    fn test_chain_metrics_emit_as_logs() {
        // Initialize tracing subscriber for this test
        let _ = tracing_subscriber::fmt()
            .with_env_filter("tensor_chain=debug")
            .with_test_writer()
            .try_init();

        let metrics = ChainMetrics::new();

        // Simulate some activity
        metrics
            .dtx
            .started
            .fetch_add(5, std::sync::atomic::Ordering::Relaxed);
        metrics
            .dtx
            .committed
            .fetch_add(3, std::sync::atomic::Ordering::Relaxed);
        metrics
            .raft
            .heartbeat_successes
            .fetch_add(100, std::sync::atomic::Ordering::Relaxed);

        // This should emit logs (visible with --nocapture)
        metrics.emit_as_logs();
    }

    #[test]
    fn test_block_ed25519_signing() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Commit a transaction - this creates a signed block
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: "signed_data".to_string(),
            data: vec![1, 2, 3, 4],
        })
        .unwrap();
        chain.commit(tx).unwrap();

        // Get the committed block
        let block = chain.get_block(1).unwrap().unwrap();

        // Verify the signature is present and non-empty
        assert_eq!(
            block.header.signature.len(),
            64,
            "Ed25519 signature should be 64 bytes"
        );
        assert_ne!(
            block.header.signature,
            vec![0u8; 64],
            "Signature should not be all zeros"
        );

        // Verify the signature manually using the identity's public key
        let public_identity = chain.identity().verifying_key();
        let signing_bytes = block.header.signing_bytes();
        let result = public_identity.verify(&signing_bytes, &block.header.signature);
        assert!(
            result.is_ok(),
            "Ed25519 signature should verify: {:?}",
            result
        );
    }

    #[test]
    fn test_block_signature_verification_fails_with_wrong_key() {
        let store = TensorStore::new();
        let chain = TensorChain::new(store, "node1");
        chain.initialize().unwrap();

        // Commit a transaction
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: "test".to_string(),
            data: vec![1],
        })
        .unwrap();
        chain.commit(tx).unwrap();

        // Get the block
        let block = chain.get_block(1).unwrap().unwrap();

        // Verify with a DIFFERENT identity's public key (should fail)
        let wrong_identity = signing::Identity::generate();
        let wrong_public = wrong_identity.verifying_key();
        let signing_bytes = block.header.signing_bytes();
        let result = wrong_public.verify(&signing_bytes, &block.header.signature);
        assert!(result.is_err(), "Verification should fail with wrong key");
    }

    #[test]
    fn test_chain_with_custom_identity() {
        let store = TensorStore::new();
        let identity = signing::Identity::generate();
        let expected_node_id = identity.node_id();
        let config = ChainConfig::new("test_node");
        let chain = TensorChain::with_identity(store, config, identity);
        chain.initialize().unwrap();

        // Verify the identity is accessible
        assert_eq!(chain.identity().node_id(), expected_node_id);

        // Commit a transaction
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: "test".to_string(),
            data: vec![1],
        })
        .unwrap();
        chain.commit(tx).unwrap();

        // Get the block and verify signature using the identity's public key
        let block = chain.get_block(1).unwrap().unwrap();
        let public_identity = chain.identity().verifying_key();
        let signing_bytes = block.header.signing_bytes();
        let result = public_identity.verify(&signing_bytes, &block.header.signature);
        assert!(
            result.is_ok(),
            "Block signature should verify: {:?}",
            result
        );
    }
}
