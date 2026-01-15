//! Tensor-Raft consensus implementation.
//!
//! Modified Raft protocol with tensor-native optimizations:
//! - Similarity fast-path for block validation
//! - State embedding for tie-breaking
//! - Two-phase finality (committed -> finalized)

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tensor_store::SparseVector;

use crate::{
    block::{Block, NodeId},
    error::{ChainError, Result},
    membership::MembershipManager,
    network::{
        AppendEntries, AppendEntriesResponse, LogEntry, Message, PreVote, PreVoteResponse,
        RequestVote, RequestVoteResponse, SnapshotRequest, SnapshotResponse, TimeoutNow, Transport,
    },
    snapshot_buffer::{SnapshotBuffer, SnapshotBufferConfig},
    validation::FastPathValidator,
};

/// Raft node state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaftState {
    /// Follower state - receives log entries from leader.
    Follower,
    /// Candidate state - requesting votes for leadership.
    Candidate,
    /// Leader state - handles client requests and replicates log.
    Leader,
}

/// Configuration for Raft consensus.
#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// Election timeout range (min, max) in milliseconds.
    pub election_timeout: (u64, u64),
    /// Heartbeat interval in milliseconds.
    pub heartbeat_interval: u64,
    /// Similarity threshold for fast-path validation.
    pub similarity_threshold: f32,
    /// Enable similarity fast-path.
    pub enable_fast_path: bool,
    /// Minimum nodes for quorum (default: majority).
    pub quorum_size: Option<usize>,
    /// Enable geometric tie-breaking in leader elections.
    /// When logs are equal, prefer candidates with similar state embeddings.
    pub enable_geometric_tiebreak: bool,
    /// Minimum similarity for geometric tie-breaking to apply (0.0-1.0).
    /// Below this threshold, vote randomly among equal candidates.
    pub geometric_tiebreak_threshold: f32,
    /// Number of log entries before triggering automatic snapshot.
    pub snapshot_threshold: usize,
    /// Number of log entries to keep after snapshot (for followers catching up).
    pub snapshot_trailing_logs: usize,
    /// Chunk size in bytes for snapshot transfer.
    pub snapshot_chunk_size: u64,
    /// Enable pre-vote phase to prevent disruptive elections.
    pub enable_pre_vote: bool,
    /// Leadership transfer timeout in milliseconds.
    pub transfer_timeout_ms: u64,
    /// Interval in ticks between compaction eligibility checks.
    pub compaction_check_interval: u64,
    /// Minimum time between compactions in milliseconds.
    pub compaction_cooldown_ms: u64,
    /// Maximum memory for snapshot buffering before spilling to disk.
    /// Default: 256MB.
    pub snapshot_max_memory: usize,
    /// Directory for temporary snapshot files.
    /// Default: system temp directory.
    pub snapshot_temp_dir: Option<std::path::PathBuf>,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            election_timeout: (150, 300),
            heartbeat_interval: 50,
            similarity_threshold: 0.95,
            enable_fast_path: true,
            quorum_size: None,
            enable_geometric_tiebreak: true,
            geometric_tiebreak_threshold: 0.3,
            snapshot_threshold: 10_000,
            snapshot_trailing_logs: 100,
            snapshot_chunk_size: 1024 * 1024, // 1MB
            enable_pre_vote: true,
            transfer_timeout_ms: 1000,
            compaction_check_interval: 10,
            compaction_cooldown_ms: 60_000,
            snapshot_max_memory: 256 * 1024 * 1024, // 256MB
            snapshot_temp_dir: None,
        }
    }
}

/// Volatile state on all servers.
struct VolatileState {
    /// Index of highest log entry known to be committed.
    commit_index: u64,
    /// Index of highest log entry applied to state machine.
    last_applied: u64,
}

/// Volatile state on leaders (reinitialized after election).
struct LeaderState {
    /// For each server, index of next log entry to send.
    next_index: HashMap<NodeId, u64>,
    /// For each server, index of highest log entry known to be replicated.
    match_index: HashMap<NodeId, u64>,
}

/// Persistent state on all servers.
struct PersistentState {
    /// Latest term server has seen.
    current_term: u64,
    /// Candidate that received vote in current term.
    voted_for: Option<NodeId>,
    /// Log entries.
    log: Vec<LogEntry>,
}

/// State for active leadership transfer.
#[derive(Debug, Clone)]
pub struct TransferState {
    /// Target node for leadership transfer.
    pub target: NodeId,
    /// Term when transfer was initiated.
    pub initiated_term: u64,
    /// Timestamp when transfer started (for timeout).
    pub started_at: Instant,
}

/// Comprehensive Raft statistics including fast-path validation and quorum tracking.
#[derive(Debug, Default)]
pub struct RaftStats {
    // Fast-path validation counters (backward compatible with FastPathStats)
    /// Number of blocks that used fast-path validation.
    pub fast_path_accepted: AtomicU64,
    /// Number of blocks that failed fast-path and required full validation.
    pub fast_path_rejected: AtomicU64,
    /// Number of blocks that required full validation (no embedding or first blocks).
    pub full_validation_required: AtomicU64,

    // Timing metrics
    /// Election duration timing (microseconds).
    pub election_timing: crate::metrics::TimingStats,
    /// Heartbeat round-trip timing (microseconds).
    pub heartbeat_timing: crate::metrics::TimingStats,

    // Quorum tracking
    /// Number of quorum checks performed.
    pub quorum_checks: AtomicU64,
    /// Number of times quorum was lost.
    pub quorum_lost_events: AtomicU64,
    /// Number of times leader stepped down due to quorum loss.
    pub leader_step_downs: AtomicU64,
    /// Number of successful heartbeat responses.
    pub heartbeat_successes: AtomicU64,
    /// Number of failed heartbeat attempts.
    pub heartbeat_failures: AtomicU64,
}

impl RaftStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a fast-path acceptance.
    pub fn record_fast_path(&self) {
        self.fast_path_accepted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a fast-path rejection.
    pub fn record_rejected(&self) {
        self.fast_path_rejected.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a full validation.
    pub fn record_full_validation(&self) {
        self.full_validation_required
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Get the fast-path acceptance rate.
    pub fn acceptance_rate(&self) -> f32 {
        let accepted = self.fast_path_accepted.load(Ordering::Relaxed);
        let rejected = self.fast_path_rejected.load(Ordering::Relaxed);
        let total = accepted + rejected;
        if total == 0 {
            0.0
        } else {
            accepted as f32 / total as f32
        }
    }

    /// Get total blocks validated.
    pub fn total_validated(&self) -> u64 {
        self.fast_path_accepted.load(Ordering::Relaxed)
            + self.fast_path_rejected.load(Ordering::Relaxed)
            + self.full_validation_required.load(Ordering::Relaxed)
    }

    /// Get the heartbeat success rate.
    pub fn heartbeat_success_rate(&self) -> f32 {
        let successes = self.heartbeat_successes.load(Ordering::Relaxed);
        let failures = self.heartbeat_failures.load(Ordering::Relaxed);
        let total = successes + failures;
        if total == 0 {
            1.0 // No heartbeats sent yet, assume success
        } else {
            successes as f32 / total as f32
        }
    }

    /// Take a point-in-time snapshot of all statistics.
    pub fn snapshot(&self) -> RaftStatsSnapshot {
        RaftStatsSnapshot {
            fast_path_accepted: self.fast_path_accepted.load(Ordering::Relaxed),
            fast_path_rejected: self.fast_path_rejected.load(Ordering::Relaxed),
            full_validation_required: self.full_validation_required.load(Ordering::Relaxed),
            election_timing: self.election_timing.snapshot(),
            heartbeat_timing: self.heartbeat_timing.snapshot(),
            quorum_checks: self.quorum_checks.load(Ordering::Relaxed),
            quorum_lost_events: self.quorum_lost_events.load(Ordering::Relaxed),
            leader_step_downs: self.leader_step_downs.load(Ordering::Relaxed),
            heartbeat_successes: self.heartbeat_successes.load(Ordering::Relaxed),
            heartbeat_failures: self.heartbeat_failures.load(Ordering::Relaxed),
            fast_path_rate: self.acceptance_rate(),
            heartbeat_success_rate: self.heartbeat_success_rate(),
        }
    }
}

/// Point-in-time snapshot of Raft statistics.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RaftStatsSnapshot {
    pub fast_path_accepted: u64,
    pub fast_path_rejected: u64,
    pub full_validation_required: u64,
    pub election_timing: crate::metrics::TimingSnapshot,
    pub heartbeat_timing: crate::metrics::TimingSnapshot,
    pub quorum_checks: u64,
    pub quorum_lost_events: u64,
    pub leader_step_downs: u64,
    pub heartbeat_successes: u64,
    pub heartbeat_failures: u64,
    pub fast_path_rate: f32,
    pub heartbeat_success_rate: f32,
}

/// Backward compatibility alias for RaftStats.
pub type FastPathStats = RaftStats;

/// Tracks heartbeat responses to detect quorum loss.
///
/// Used by the Raft leader to verify it can still reach a majority
/// of followers. If quorum is lost, the leader steps down to prevent
/// split-brain scenarios.
pub struct QuorumTracker {
    /// Last successful response time per peer.
    last_response: RwLock<HashMap<NodeId, Instant>>,
    /// Consecutive failures per peer.
    consecutive_failures: RwLock<HashMap<NodeId, u32>>,
    /// Response timeout threshold.
    response_timeout: std::time::Duration,
    /// Max failures before considering peer unreachable.
    max_failures: u32,
}

impl QuorumTracker {
    /// Create a new quorum tracker.
    pub fn new(response_timeout: std::time::Duration, max_failures: u32) -> Self {
        Self {
            last_response: RwLock::new(HashMap::new()),
            consecutive_failures: RwLock::new(HashMap::new()),
            response_timeout,
            max_failures,
        }
    }

    /// Create with default settings (5s timeout, 3 max failures).
    pub fn default_config() -> Self {
        Self::new(std::time::Duration::from_secs(5), 3)
    }

    /// Record a successful response from a peer.
    pub fn record_success(&self, node_id: &NodeId) {
        self.last_response
            .write()
            .insert(node_id.clone(), Instant::now());
        self.consecutive_failures.write().remove(node_id);
    }

    /// Record a failed attempt to reach a peer.
    pub fn record_failure(&self, node_id: &NodeId) {
        let mut failures = self.consecutive_failures.write();
        *failures.entry(node_id.clone()).or_insert(0) += 1;
    }

    /// Check if a specific peer is currently reachable.
    pub fn is_reachable(&self, node_id: &NodeId) -> bool {
        // Check consecutive failures
        let failures = self.consecutive_failures.read();
        if failures.get(node_id).copied().unwrap_or(0) >= self.max_failures {
            return false;
        }

        // Check last response time
        let last = self.last_response.read();
        last.get(node_id)
            .map(|t| t.elapsed() < self.response_timeout)
            .unwrap_or(false)
    }

    /// Count the number of currently reachable peers.
    pub fn reachable_count(&self) -> usize {
        let last = self.last_response.read();
        last.keys().filter(|id| self.is_reachable(id)).count()
    }

    /// Check if we have quorum (including self).
    ///
    /// For a cluster of N nodes, quorum requires (N/2)+1 nodes.
    /// Self is always considered reachable.
    pub fn has_quorum(&self, total_peers: usize) -> bool {
        // Total nodes = peers + self
        let total_nodes = total_peers + 1;
        let quorum_size = (total_nodes / 2) + 1;
        // Reachable = peers responding + self (always reachable)
        self.reachable_count() + 1 >= quorum_size
    }

    /// Reset all tracking state.
    pub fn reset(&self) {
        self.last_response.write().clear();
        self.consecutive_failures.write().clear();
    }

    /// Mark a peer as initially reachable (for new peers).
    pub fn mark_reachable(&self, node_id: &NodeId) {
        self.last_response
            .write()
            .insert(node_id.clone(), Instant::now());
    }

    /// Get the list of unreachable peers.
    pub fn unreachable_peers(&self) -> Vec<NodeId> {
        let last = self.last_response.read();
        last.keys()
            .filter(|id| !self.is_reachable(id))
            .cloned()
            .collect()
    }
}

impl std::fmt::Debug for QuorumTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuorumTracker")
            .field("reachable_count", &self.reachable_count())
            .field("response_timeout", &self.response_timeout)
            .field("max_failures", &self.max_failures)
            .finish()
    }
}

impl Default for QuorumTracker {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Metadata describing a Raft snapshot.
///
/// Snapshots capture the state machine state at a particular log index,
/// allowing log truncation and faster catch-up for lagging followers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    /// Log index included in this snapshot (all entries up to and including).
    pub last_included_index: u64,
    /// Term of the entry at last_included_index.
    pub last_included_term: u64,
    /// Block hash at the snapshot height.
    pub snapshot_hash: [u8; 32],
    /// Cluster configuration at snapshot time (legacy field for backward compatibility).
    #[serde(default)]
    pub config: Vec<NodeId>,
    /// Full membership configuration at snapshot time.
    #[serde(default)]
    pub membership: crate::network::RaftMembershipConfig,
    /// Unix timestamp when snapshot was created.
    pub created_at: u64,
    /// Snapshot data size in bytes.
    pub size: u64,
}

impl SnapshotMetadata {
    /// Create new snapshot metadata.
    pub fn new(
        last_included_index: u64,
        last_included_term: u64,
        snapshot_hash: [u8; 32],
        config: Vec<NodeId>,
        size: u64,
    ) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self {
            last_included_index,
            last_included_term,
            snapshot_hash,
            membership: crate::network::RaftMembershipConfig::new(config.clone()),
            config,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            size,
        }
    }

    /// Create snapshot metadata with full membership configuration.
    pub fn with_membership(
        last_included_index: u64,
        last_included_term: u64,
        snapshot_hash: [u8; 32],
        membership: crate::network::RaftMembershipConfig,
        size: u64,
    ) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        Self {
            last_included_index,
            last_included_term,
            snapshot_hash,
            config: membership.voters.clone(),
            membership,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            size,
        }
    }
}

/// Internal snapshot state tracking.
struct SnapshotState {
    /// Current snapshot metadata (if any).
    last_snapshot: Option<SnapshotMetadata>,
    /// Whether a snapshot operation is in progress.
    in_progress: bool,
    /// Memory-efficient buffer for pending snapshot data during chunked transfer.
    pending_buffer: Option<SnapshotBuffer>,
    /// Expected total size for pending snapshot.
    pending_total_size: u64,
    /// Configuration for snapshot buffer creation.
    buffer_config: SnapshotBufferConfig,
}

impl SnapshotState {
    fn new(config: SnapshotBufferConfig) -> Self {
        Self {
            last_snapshot: None,
            in_progress: false,
            pending_buffer: None,
            pending_total_size: 0,
            buffer_config: config,
        }
    }

    fn from_raft_config(config: &RaftConfig) -> Self {
        let buffer_config = SnapshotBufferConfig {
            max_memory_bytes: config.snapshot_max_memory,
            temp_dir: config
                .snapshot_temp_dir
                .clone()
                .unwrap_or_else(std::env::temp_dir),
            initial_file_capacity: config.snapshot_chunk_size as usize * 16, // 16 chunks worth
        };
        Self::new(buffer_config)
    }

    fn start_receive(
        &mut self,
        total_size: u64,
    ) -> std::result::Result<(), crate::snapshot_buffer::SnapshotBufferError> {
        self.in_progress = true;
        // Clean up any existing buffer
        if let Some(mut buf) = self.pending_buffer.take() {
            let _ = buf.cleanup();
        }
        self.pending_buffer = Some(SnapshotBuffer::new(self.buffer_config.clone())?);
        self.pending_total_size = total_size;
        Ok(())
    }

    fn append_chunk(
        &mut self,
        data: &[u8],
    ) -> std::result::Result<(), crate::snapshot_buffer::SnapshotBufferError> {
        if let Some(ref mut buf) = self.pending_buffer {
            buf.write(data)?;
        }
        Ok(())
    }

    fn finish_receive(&mut self) -> Option<SnapshotBuffer> {
        self.in_progress = false;
        self.pending_total_size = 0;
        if let Some(mut buf) = self.pending_buffer.take() {
            if buf.finalize().is_ok() {
                return Some(buf);
            }
        }
        None
    }

    fn cancel_receive(&mut self) {
        self.in_progress = false;
        self.pending_total_size = 0;
        if let Some(mut buf) = self.pending_buffer.take() {
            let _ = buf.cleanup();
        }
    }
}

/// State for fast-path validation per leader.
#[derive(Debug)]
pub struct FastPathState {
    /// Recent embeddings per leader (sparse for memory efficiency).
    leader_embeddings: RwLock<HashMap<NodeId, Vec<SparseVector>>>,
    /// Maximum number of embeddings to keep per leader.
    max_history: usize,
    /// Statistics.
    pub stats: FastPathStats,
}

impl Default for FastPathState {
    fn default() -> Self {
        Self::new(10)
    }
}

impl FastPathState {
    /// Create new fast-path state.
    pub fn new(max_history: usize) -> Self {
        Self {
            leader_embeddings: RwLock::new(HashMap::new()),
            max_history,
            stats: FastPathStats::new(),
        }
    }

    /// Add an embedding for a leader (sparse).
    pub fn add_embedding(&self, leader: &NodeId, embedding: SparseVector) {
        let mut embeddings = self.leader_embeddings.write();
        let history = embeddings.entry(leader.clone()).or_default();

        history.push(embedding);

        // Keep only recent embeddings
        if history.len() > self.max_history {
            history.remove(0);
        }
    }

    /// Add an embedding for a leader from dense vector.
    pub fn add_dense_embedding(&self, leader: &NodeId, embedding: Vec<f32>) {
        self.add_embedding(leader, SparseVector::from_dense(&embedding));
    }

    /// Get embeddings for a leader (returns dense for fast-path validation compatibility).
    pub fn get_embeddings(&self, leader: &NodeId) -> Vec<Vec<f32>> {
        self.leader_embeddings
            .read()
            .get(leader)
            .map(|sparse_vecs| sparse_vecs.iter().map(|sv| sv.to_dense()).collect())
            .unwrap_or_default()
    }

    /// Get sparse embeddings for a leader.
    pub fn get_sparse_embeddings(&self, leader: &NodeId) -> Vec<SparseVector> {
        self.leader_embeddings
            .read()
            .get(leader)
            .cloned()
            .unwrap_or_default()
    }

    /// Clear embeddings for a leader (e.g., on leader change).
    pub fn clear_leader(&self, leader: &NodeId) {
        self.leader_embeddings.write().remove(leader);
    }

    /// Get the number of embeddings stored for a leader.
    pub fn leader_history_size(&self, leader: &NodeId) -> usize {
        self.leader_embeddings
            .read()
            .get(leader)
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

/// Tensor-Raft node.
pub struct RaftNode {
    /// Local node ID.
    node_id: NodeId,
    /// Current state (follower/candidate/leader).
    state: RwLock<RaftState>,
    /// Persistent state.
    persistent: RwLock<PersistentState>,
    /// Volatile state.
    volatile: RwLock<VolatileState>,
    /// Leader-specific state.
    leader_state: RwLock<Option<LeaderState>>,
    /// Current leader (if known).
    current_leader: RwLock<Option<NodeId>>,
    /// List of peer node IDs.
    peers: RwLock<Vec<NodeId>>,
    /// Transport for network communication.
    transport: Arc<dyn Transport>,
    /// Configuration.
    config: RaftConfig,
    /// Last heartbeat received.
    last_heartbeat: RwLock<Instant>,
    /// Votes received in current election.
    votes_received: RwLock<Vec<NodeId>>,
    /// State embedding for similarity-based tie-breaking (sparse for efficiency).
    state_embedding: RwLock<SparseVector>,
    /// Finalized height (checkpointed).
    finalized_height: AtomicU64,
    /// Fast-path validation state.
    fast_path_state: FastPathState,
    /// Fast-path validator for replication.
    fast_path_validator: FastPathValidator,
    /// Optional membership manager for health-aware voting.
    membership: Option<Arc<MembershipManager>>,
    /// Snapshot state for log compaction.
    snapshot_state: RwLock<SnapshotState>,
    /// Quorum tracker for split-brain prevention.
    quorum_tracker: QuorumTracker,
    /// Statistics.
    pub stats: RaftStats,
    /// Pre-votes received in current pre-election.
    pre_votes_received: RwLock<Vec<NodeId>>,
    /// Whether currently in pre-vote phase.
    in_pre_vote: RwLock<bool>,
    /// Active leadership transfer state.
    transfer_state: RwLock<Option<TransferState>>,
    /// Tick counter for compaction check interval.
    compaction_tick_counter: AtomicU64,
    /// Timestamp of last successful compaction (for cooldown).
    last_compaction: RwLock<Option<Instant>>,
    /// Optional TensorStore reference for snapshot persistence.
    store: Option<Arc<tensor_store::TensorStore>>,
    /// Optional Write-Ahead Log for durable state changes.
    wal: Option<Arc<parking_lot::Mutex<crate::raft_wal::RaftWal>>>,
    /// Membership configuration for dynamic cluster membership.
    membership_config: RwLock<crate::network::RaftMembershipConfig>,
}

impl RaftNode {
    /// Create a new Raft node.
    pub fn new(
        node_id: NodeId,
        peers: Vec<NodeId>,
        transport: Arc<dyn Transport>,
        config: RaftConfig,
    ) -> Self {
        // Build initial membership config with self + all peers as voters
        let initial_voters: Vec<NodeId> = std::iter::once(node_id.clone())
            .chain(peers.iter().cloned())
            .collect();
        let membership_config = crate::network::RaftMembershipConfig::new(initial_voters);

        Self {
            node_id,
            state: RwLock::new(RaftState::Follower),
            persistent: RwLock::new(PersistentState {
                current_term: 0,
                voted_for: None,
                log: Vec::new(),
            }),
            volatile: RwLock::new(VolatileState {
                commit_index: 0,
                last_applied: 0,
            }),
            leader_state: RwLock::new(None),
            current_leader: RwLock::new(None),
            peers: RwLock::new(peers),
            transport,
            last_heartbeat: RwLock::new(Instant::now()),
            votes_received: RwLock::new(Vec::new()),
            state_embedding: RwLock::new(SparseVector::new(0)),
            finalized_height: AtomicU64::new(0),
            fast_path_state: FastPathState::default(),
            fast_path_validator: FastPathValidator::new(config.similarity_threshold, 3),
            snapshot_state: RwLock::new(SnapshotState::from_raft_config(&config)),
            config,
            membership: None,
            quorum_tracker: QuorumTracker::default(),
            stats: RaftStats::new(),
            pre_votes_received: RwLock::new(Vec::new()),
            in_pre_vote: RwLock::new(false),
            transfer_state: RwLock::new(None),
            compaction_tick_counter: AtomicU64::new(0),
            last_compaction: RwLock::new(None),
            store: None,
            wal: None,
            membership_config: RwLock::new(membership_config),
        }
    }

    /// Create a new Raft node with membership manager.
    pub fn with_membership(
        node_id: NodeId,
        peers: Vec<NodeId>,
        transport: Arc<dyn Transport>,
        config: RaftConfig,
        membership: Arc<MembershipManager>,
    ) -> Self {
        let mut node = Self::new(node_id, peers, transport, config);
        node.membership = Some(membership);
        node
    }

    /// Set the membership manager.
    pub fn set_membership(&mut self, membership: Arc<MembershipManager>) {
        self.membership = Some(membership);
    }

    // ========== Persistence Methods ==========

    /// Key for persisting Raft state in TensorStore.
    fn persistence_key(node_id: &str) -> String {
        format!("_raft:state:{}", node_id)
    }

    /// Key for persisting snapshot metadata in TensorStore.
    fn snapshot_meta_key(node_id: &str) -> String {
        format!("_raft:snapshot:meta:{}", node_id)
    }

    /// Key for persisting snapshot data in TensorStore.
    fn snapshot_data_key(node_id: &str) -> String {
        format!("_raft:snapshot:data:{}", node_id)
    }

    /// Save persistent state to TensorStore.
    ///
    /// Stores term, voted_for, log entries, and state embedding.
    /// Use `save_snapshot_compressed()` for tensor-native compression.
    pub fn save_to_store(&self, store: &tensor_store::TensorStore) -> Result<()> {
        use tensor_store::{ScalarValue, TensorData, TensorValue};

        let persistent = self.persistent.read();
        let key = Self::persistence_key(&self.node_id);

        let mut data = TensorData::new();

        // Store current term
        data.set(
            "term",
            TensorValue::Scalar(ScalarValue::Int(persistent.current_term as i64)),
        );

        // Store voted_for if present
        if let Some(ref voted_for) = persistent.voted_for {
            data.set(
                "voted_for",
                TensorValue::Scalar(ScalarValue::String(voted_for.clone())),
            );
        }

        // Serialize log entries as bytes
        let log_bytes = bincode::serialize(&persistent.log)
            .map_err(|e| ChainError::SerializationError(format!("Raft log: {}", e)))?;
        data.set("log", TensorValue::Scalar(ScalarValue::Bytes(log_bytes)));

        // Store state embedding for geometric recovery
        let state_embedding = self.state_embedding.read();
        if state_embedding.nnz() > 0 {
            data.set("_embedding", TensorValue::Sparse(state_embedding.clone()));
        }

        store
            .put(&key, data)
            .map_err(|e| ChainError::StorageError(e.to_string()))
    }

    /// Load persistent state from TensorStore.
    ///
    /// Returns (term, voted_for, log) if state exists.
    pub fn load_from_store(
        node_id: &str,
        store: &tensor_store::TensorStore,
    ) -> Option<(u64, Option<NodeId>, Vec<LogEntry>)> {
        use tensor_store::{ScalarValue, TensorValue};

        let key = Self::persistence_key(node_id);
        let data = store.get(&key).ok()?;

        // Load term
        let term = match data.get("term") {
            Some(TensorValue::Scalar(ScalarValue::Int(t))) => *t as u64,
            _ => return None,
        };

        // Load voted_for
        let voted_for = match data.get("voted_for") {
            Some(TensorValue::Scalar(ScalarValue::String(v))) => Some(v.clone()),
            _ => None,
        };

        // Load log entries
        let log = match data.get("log") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) => {
                bincode::deserialize(bytes).ok()?
            },
            _ => Vec::new(),
        };

        Some((term, voted_for, log))
    }

    /// Persist snapshot metadata and data to TensorStore.
    ///
    /// MUST be called BEFORE truncate_log() to ensure atomicity.
    pub fn save_snapshot(
        &self,
        meta: &SnapshotMetadata,
        data: &[u8],
        store: &tensor_store::TensorStore,
    ) -> Result<()> {
        use tensor_store::{ScalarValue, TensorData, TensorValue};

        // Serialize and store metadata
        let meta_bytes =
            bincode::serialize(meta).map_err(|e| ChainError::SerializationError(e.to_string()))?;
        let mut meta_data = TensorData::new();
        meta_data.set(
            "metadata",
            TensorValue::Scalar(ScalarValue::Bytes(meta_bytes)),
        );
        store
            .put(Self::snapshot_meta_key(&self.node_id), meta_data)
            .map_err(|e| ChainError::StorageError(e.to_string()))?;

        // Store snapshot data
        let mut snap_data = TensorData::new();
        snap_data.set(
            "data",
            TensorValue::Scalar(ScalarValue::Bytes(data.to_vec())),
        );
        store
            .put(Self::snapshot_data_key(&self.node_id), snap_data)
            .map_err(|e| ChainError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// Load snapshot state from TensorStore.
    ///
    /// Returns (metadata, data) if a snapshot exists.
    pub fn load_snapshot(
        node_id: &str,
        store: &tensor_store::TensorStore,
    ) -> Option<(SnapshotMetadata, Vec<u8>)> {
        use tensor_store::{ScalarValue, TensorValue};

        // Load metadata
        let meta_tensor = store.get(&Self::snapshot_meta_key(node_id)).ok()?;
        let meta_bytes = match meta_tensor.get("metadata") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b,
            _ => return None,
        };
        let metadata: SnapshotMetadata = bincode::deserialize(meta_bytes).ok()?;

        // Load data
        let data_tensor = store.get(&Self::snapshot_data_key(node_id)).ok()?;
        let data = match data_tensor.get("data") {
            Some(TensorValue::Scalar(ScalarValue::Bytes(b))) => b.clone(),
            _ => return None,
        };

        Some((metadata, data))
    }

    // ========== Compaction Cooldown Methods ==========

    /// Check if enough time has passed since last compaction.
    fn can_compact(&self) -> bool {
        match *self.last_compaction.read() {
            Some(instant) => {
                instant.elapsed().as_millis() as u64 >= self.config.compaction_cooldown_ms
            },
            None => true,
        }
    }

    /// Update last compaction timestamp.
    fn mark_compacted(&self) {
        *self.last_compaction.write() = Some(Instant::now());
    }

    /// Create a Raft node, loading persisted state from store if available.
    pub fn with_store(
        node_id: NodeId,
        peers: Vec<NodeId>,
        transport: Arc<dyn Transport>,
        config: RaftConfig,
        store: &tensor_store::TensorStore,
    ) -> Self {
        // Load Raft state
        let (term, voted_for, log) =
            Self::load_from_store(&node_id, store).unwrap_or((0, None, Vec::new()));

        // Load and validate snapshot state
        let snapshot_meta = if let Some((meta, data)) = Self::load_snapshot(&node_id, store) {
            // Validate snapshot hash
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            hasher.update(&data);
            let computed_hash: [u8; 32] = hasher.finalize().into();

            if computed_hash == meta.snapshot_hash {
                tracing::info!(
                    "Restored snapshot at index {} with valid hash",
                    meta.last_included_index
                );
                Some(meta)
            } else {
                tracing::warn!("Snapshot hash mismatch on startup, ignoring corrupted snapshot");
                None
            }
        } else {
            None
        };

        // Create node with loaded state
        let mut node = Self::with_state(node_id, peers, transport, config, term, voted_for, log);

        // Restore snapshot metadata if available
        if let Some(meta) = snapshot_meta {
            node.snapshot_state.write().last_snapshot = Some(meta);
        }

        // Store reference for future persistence
        node.store = Some(Arc::new(store.clone()));

        node
    }

    /// Create a Raft node with explicit initial state.
    pub fn with_state(
        node_id: NodeId,
        peers: Vec<NodeId>,
        transport: Arc<dyn Transport>,
        config: RaftConfig,
        term: u64,
        voted_for: Option<NodeId>,
        log: Vec<LogEntry>,
    ) -> Self {
        // Build initial membership config with self + all peers as voters
        let initial_voters: Vec<NodeId> = std::iter::once(node_id.clone())
            .chain(peers.iter().cloned())
            .collect();
        let membership_config = crate::network::RaftMembershipConfig::new(initial_voters);

        Self {
            node_id,
            state: RwLock::new(RaftState::Follower),
            persistent: RwLock::new(PersistentState {
                current_term: term,
                voted_for,
                log,
            }),
            volatile: RwLock::new(VolatileState {
                commit_index: 0,
                last_applied: 0,
            }),
            leader_state: RwLock::new(None),
            current_leader: RwLock::new(None),
            peers: RwLock::new(peers),
            transport,
            last_heartbeat: RwLock::new(Instant::now()),
            votes_received: RwLock::new(Vec::new()),
            state_embedding: RwLock::new(SparseVector::new(0)),
            finalized_height: AtomicU64::new(0),
            fast_path_state: FastPathState::default(),
            fast_path_validator: FastPathValidator::new(config.similarity_threshold, 3),
            snapshot_state: RwLock::new(SnapshotState::from_raft_config(&config)),
            config,
            membership: None,
            quorum_tracker: QuorumTracker::default(),
            pre_votes_received: RwLock::new(Vec::new()),
            in_pre_vote: RwLock::new(false),
            transfer_state: RwLock::new(None),
            stats: RaftStats::new(),
            compaction_tick_counter: AtomicU64::new(0),
            last_compaction: RwLock::new(None),
            store: None,
            wal: None,
            membership_config: RwLock::new(membership_config),
        }
    }

    /// Create a Raft node with WAL for durable state changes.
    ///
    /// The WAL ensures that term and voted_for changes are persisted to disk
    /// before being applied in memory, preventing split-brain scenarios.
    pub fn with_wal(
        node_id: NodeId,
        peers: Vec<NodeId>,
        transport: Arc<dyn Transport>,
        config: RaftConfig,
        wal_path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<Self> {
        use crate::raft_wal::{RaftRecoveryState, RaftWal};

        let wal = RaftWal::open(wal_path)?;
        let recovery = RaftRecoveryState::from_wal(&wal)?;

        let mut node = Self::with_state(
            node_id,
            peers,
            transport,
            config,
            recovery.current_term,
            recovery.voted_for,
            Vec::new(), // Log entries recovered separately
        );

        node.wal = Some(Arc::new(parking_lot::Mutex::new(wal)));

        Ok(node)
    }

    /// Persist term and vote to WAL before applying state change.
    ///
    /// Returns Ok(()) if WAL is disabled or if write succeeds.
    /// Returns Err if WAL write fails (state change should be aborted).
    fn persist_term_and_vote(&self, term: u64, voted_for: Option<&str>) -> Result<()> {
        use crate::raft_wal::RaftWalEntry;

        if let Some(ref wal) = self.wal {
            wal.lock()
                .append(&RaftWalEntry::TermAndVote {
                    term,
                    voted_for: voted_for.map(String::from),
                })
                .map_err(|e| ChainError::StorageError(e.to_string()))?;
        }
        Ok(())
    }

    /// Check if a node is healthy according to membership.
    fn is_peer_healthy(&self, peer_id: &NodeId) -> bool {
        match &self.membership {
            Some(membership) => membership.view().is_healthy(peer_id),
            None => true, // No membership manager = assume all peers healthy
        }
    }

    /// Compute geometric vote bias based on state embedding similarity.
    ///
    /// Returns a score from 0.0 to 1.0 indicating preference for a candidate.
    /// Higher similarity between local and candidate state embeddings = higher score.
    fn geometric_vote_bias(&self, candidate_embedding: &SparseVector) -> f32 {
        if !self.config.enable_geometric_tiebreak {
            return 0.5; // Neutral bias
        }

        let local_embedding = self.state_embedding.read();

        // If either embedding is empty (zero dimension), return neutral bias
        if local_embedding.dimension() == 0 || candidate_embedding.dimension() == 0 {
            return 0.5;
        }

        let similarity = local_embedding.cosine_similarity(candidate_embedding);

        // Normalize to 0-1 range (cosine similarity can be -1 to 1)
        ((similarity + 1.0) / 2.0).clamp(0.0, 1.0)
    }

    /// Get the node ID.
    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    /// Get the current state.
    pub fn state(&self) -> RaftState {
        *self.state.read()
    }

    /// Get the current term.
    pub fn current_term(&self) -> u64 {
        self.persistent.read().current_term
    }

    /// Get the current leader.
    pub fn current_leader(&self) -> Option<NodeId> {
        self.current_leader.read().clone()
    }

    /// Set the current leader (test helper).
    pub fn set_current_leader(&self, leader_id: Option<NodeId>) {
        *self.current_leader.write() = leader_id;
    }

    /// Reset last heartbeat to simulate election timeout elapsed (test helper).
    pub fn reset_heartbeat_for_election(&self) {
        *self.last_heartbeat.write() = Instant::now() - std::time::Duration::from_secs(10);
    }

    /// Check if this node is the leader.
    pub fn is_leader(&self) -> bool {
        *self.state.read() == RaftState::Leader
    }

    /// Get the commit index.
    pub fn commit_index(&self) -> u64 {
        self.volatile.read().commit_index
    }

    /// Get the finalized height.
    pub fn finalized_height(&self) -> u64 {
        self.finalized_height.load(Ordering::SeqCst)
    }

    /// Set the finalized height directly (test helper).
    pub fn set_finalized_height(&self, height: u64) {
        self.finalized_height.store(height, Ordering::SeqCst);
    }

    /// Get the log length.
    pub fn log_length(&self) -> usize {
        self.persistent.read().log.len()
    }

    /// Get fast-path statistics.
    pub fn fast_path_stats(&self) -> &FastPathStats {
        &self.fast_path_state.stats
    }

    /// Get fast-path state.
    pub fn fast_path_state(&self) -> &FastPathState {
        &self.fast_path_state
    }

    /// Get the Raft stats.
    pub fn stats(&self) -> &RaftStats {
        &self.stats
    }

    /// Get the quorum tracker.
    pub fn quorum_tracker(&self) -> &QuorumTracker {
        &self.quorum_tracker
    }

    /// Get a reference to the transport layer.
    pub fn transport(&self) -> &Arc<dyn Transport> {
        &self.transport
    }

    // ========== Dynamic Membership APIs ==========

    /// Get the current membership configuration.
    pub fn membership_config(&self) -> crate::network::RaftMembershipConfig {
        self.membership_config.read().clone()
    }

    /// Set the membership configuration (used by state machine when applying config entries).
    pub fn set_membership_config(&self, config: crate::network::RaftMembershipConfig) {
        *self.membership_config.write() = config;
    }

    /// Add a new node as a learner.
    ///
    /// The node will receive log entries but not participate in voting
    /// until it catches up and is promoted to voter.
    pub fn add_learner(&self, node_id: NodeId) -> crate::error::Result<()> {
        if !self.is_leader() {
            return Err(crate::error::ChainError::NotLeader);
        }

        let mut config = self.membership_config.write();
        if config.voters.contains(&node_id) || config.learners.contains(&node_id) {
            return Err(crate::error::ChainError::InvalidState(
                "Node already in cluster".to_string(),
            ));
        }

        config.add_learner(node_id.clone());

        // Also add to peers list for replication
        let mut peers = self.peers.write();
        if !peers.contains(&node_id) {
            peers.push(node_id);
        }

        Ok(())
    }

    /// Promote a learner to a voting member.
    ///
    /// The learner must have caught up with the leader's log.
    pub fn promote_learner(&self, node_id: &NodeId) -> crate::error::Result<()> {
        if !self.is_leader() {
            return Err(crate::error::ChainError::NotLeader);
        }

        let mut config = self.membership_config.write();
        if !config.is_learner(node_id) {
            return Err(crate::error::ChainError::InvalidState(
                "Node is not a learner".to_string(),
            ));
        }

        if !config.promote_learner(node_id) {
            return Err(crate::error::ChainError::InvalidState(
                "Failed to promote learner".to_string(),
            ));
        }

        Ok(())
    }

    /// Remove a node from the cluster.
    ///
    /// Cannot remove self if we are the leader (transfer leadership first).
    pub fn remove_node(&self, node_id: &NodeId) -> crate::error::Result<()> {
        if !self.is_leader() {
            return Err(crate::error::ChainError::NotLeader);
        }

        if node_id == &self.node_id {
            return Err(crate::error::ChainError::InvalidState(
                "Cannot remove self while leader".to_string(),
            ));
        }

        let mut config = self.membership_config.write();
        if !config.remove_node(node_id) {
            return Err(crate::error::ChainError::InvalidState(
                "Node not in cluster".to_string(),
            ));
        }

        // Also remove from peers list
        let mut peers = self.peers.write();
        peers.retain(|p| p != node_id);

        Ok(())
    }

    /// Check if a learner has caught up with the leader's log.
    pub fn is_learner_caught_up(&self, node_id: &NodeId) -> bool {
        let config = self.membership_config.read();
        if !config.is_learner(node_id) {
            return false;
        }

        // Check if we have leader state and the learner's match index
        if let Some(ref leader_state) = *self.leader_state.read() {
            if let Some(&match_index) = leader_state.match_index.get(node_id) {
                let commit_index = self.commit_index();
                // Consider caught up if within 10 entries of commit index
                return match_index + 10 >= commit_index;
            }
        }

        false
    }

    /// Check if the cluster has quorum for the given set of votes.
    pub fn has_quorum(&self, votes: &std::collections::HashSet<NodeId>) -> bool {
        self.membership_config.read().has_quorum(votes)
    }

    /// Get all nodes that should receive log entries.
    pub fn replication_targets(&self) -> Vec<NodeId> {
        self.membership_config.read().replication_targets()
    }

    /// Check if currently in joint consensus mode.
    pub fn in_joint_consensus(&self) -> bool {
        self.membership_config.read().in_joint_consensus()
    }

    /// Get the last log index.
    pub fn last_log_index(&self) -> u64 {
        let log = &self.persistent.read().log;
        if log.is_empty() {
            0
        } else {
            log[log.len() - 1].index
        }
    }

    /// Calculate quorum size (majority of total nodes).
    fn quorum_size(&self) -> usize {
        self.config
            .quorum_size
            .unwrap_or_else(|| (self.peers.read().len() + 1).div_ceil(2))
    }

    /// Get last log index and term.
    fn last_log_info(&self) -> (u64, u64) {
        let log = &self.persistent.read().log;
        if log.is_empty() {
            (0, 0)
        } else {
            let last = &log[log.len() - 1];
            (last.index, last.term)
        }
    }

    /// Update state embedding (rolling hash of recent state).
    pub fn update_state_embedding(&self, embedding: SparseVector) {
        *self.state_embedding.write() = embedding;
    }

    /// Update state embedding from a dense vector.
    pub fn update_state_embedding_dense(&self, embedding: Vec<f32>) {
        *self.state_embedding.write() = SparseVector::from_dense(&embedding);
    }

    /// Handle a received message.
    pub fn handle_message(&self, from: &NodeId, msg: &Message) -> Option<Message> {
        match msg {
            Message::RequestVote(rv) => self.handle_request_vote(from, rv),
            Message::RequestVoteResponse(rvr) => {
                self.handle_request_vote_response(from, rvr);
                None
            },
            Message::PreVote(pv) => self.handle_pre_vote(from, pv),
            Message::PreVoteResponse(pvr) => {
                self.handle_pre_vote_response(from, pvr);
                None
            },
            Message::TimeoutNow(tn) => self.handle_timeout_now(from, tn),
            Message::AppendEntries(ae) => self.handle_append_entries(from, ae),
            Message::AppendEntriesResponse(aer) => {
                self.handle_append_entries_response(from, aer);
                None
            },
            Message::SnapshotRequest(sr) => self.handle_snapshot_request(from, sr),
            Message::SnapshotResponse(sr) => {
                self.handle_snapshot_response(from, sr);
                None
            },
            Message::Ping { term } => Some(Message::Pong { term: *term }),
            _ => None,
        }
    }

    /// Handle RequestVote RPC.
    fn handle_request_vote(&self, _from: &NodeId, rv: &RequestVote) -> Option<Message> {
        let mut persistent = self.persistent.write();
        let mut vote_granted = false;

        // Update term if needed
        if rv.term > persistent.current_term {
            // CRITICAL: Persist BEFORE applying state change
            if let Err(e) = self.persist_term_and_vote(rv.term, None) {
                tracing::error!("WAL persist failed during term update: {}", e);
                // Return current state - don't update term
                return Some(Message::RequestVoteResponse(RequestVoteResponse {
                    term: persistent.current_term,
                    vote_granted: false,
                    voter_id: self.node_id.clone(),
                }));
            }
            persistent.current_term = rv.term;
            persistent.voted_for = None;
            *self.state.write() = RaftState::Follower;
        }

        // Grant vote if:
        // 1. Term is current
        // 2. Haven't voted or voted for this candidate
        // 3. Candidate's log is at least as up-to-date
        // 4. Candidate is healthy (if membership manager is configured)
        // 5. For equal logs, use geometric tie-breaking based on state embedding similarity
        if rv.term == persistent.current_term {
            let can_vote = persistent.voted_for.is_none()
                || persistent.voted_for.as_ref() == Some(&rv.candidate_id);

            // Check candidate health - don't vote for unhealthy candidates
            let candidate_healthy = self.is_peer_healthy(&rv.candidate_id);

            // Compute last log info from the lock we already hold
            let (last_log_index, last_log_term) = if persistent.log.is_empty() {
                (0, 0)
            } else {
                let last = &persistent.log[persistent.log.len() - 1];
                (last.index, last.term)
            };

            // Check if candidate's log is strictly better or exactly equal
            let log_strictly_better = rv.last_log_term > last_log_term
                || (rv.last_log_term == last_log_term && rv.last_log_index > last_log_index);
            let log_equal =
                rv.last_log_term == last_log_term && rv.last_log_index == last_log_index;

            // For equal logs, use geometric tie-breaking
            let geometric_ok = if log_equal && self.config.enable_geometric_tiebreak {
                let bias = self.geometric_vote_bias(&rv.state_embedding);
                bias >= self.config.geometric_tiebreak_threshold
            } else {
                true // No geometric tie-breaking needed for strictly better logs
            };

            let log_ok = log_strictly_better || (log_equal && geometric_ok);

            if can_vote && log_ok && candidate_healthy {
                // CRITICAL: Persist vote BEFORE granting
                if let Err(e) =
                    self.persist_term_and_vote(persistent.current_term, Some(&rv.candidate_id))
                {
                    tracing::error!("WAL persist failed during vote grant: {}", e);
                    // Don't grant vote if WAL fails
                } else {
                    vote_granted = true;
                    persistent.voted_for = Some(rv.candidate_id.clone());
                    *self.last_heartbeat.write() = Instant::now();
                }
            }
        }

        Some(Message::RequestVoteResponse(RequestVoteResponse {
            term: persistent.current_term,
            vote_granted,
            voter_id: self.node_id.clone(),
        }))
    }

    /// Handle RequestVoteResponse RPC.
    fn handle_request_vote_response(&self, from: &NodeId, rvr: &RequestVoteResponse) {
        if *self.state.read() != RaftState::Candidate {
            return;
        }

        let mut persistent = self.persistent.write();
        if rvr.term > persistent.current_term {
            // CRITICAL: Persist BEFORE applying state change
            if let Err(e) = self.persist_term_and_vote(rvr.term, None) {
                tracing::error!("WAL persist failed during step-down: {}", e);
                return; // Don't step down if WAL fails
            }
            persistent.current_term = rvr.term;
            persistent.voted_for = None;
            *self.state.write() = RaftState::Follower;
            return;
        }

        if rvr.vote_granted && rvr.term == persistent.current_term {
            let mut votes = self.votes_received.write();
            if !votes.contains(from) {
                votes.push(from.clone());

                // Check if we have quorum
                if votes.len() >= self.quorum_size() {
                    drop(votes);
                    drop(persistent);
                    self.become_leader();
                }
            }
        }
    }

    /// Start pre-vote phase (sync version).
    ///
    /// Pre-vote prevents disruptive elections from partitioned nodes by requiring
    /// candidates to confirm they can win before incrementing their term.
    pub fn start_pre_vote(&self) {
        *self.in_pre_vote.write() = true;
        *self.pre_votes_received.write() = vec![self.node_id.clone()]; // Vote for self

        let persistent = self.persistent.read();
        let (last_log_index, last_log_term) = if persistent.log.is_empty() {
            (0, 0)
        } else {
            let last = &persistent.log[persistent.log.len() - 1];
            (last.index, last.term)
        };
        let term = persistent.current_term;
        let state_embedding = self.state_embedding.read().clone();
        drop(persistent);

        let _request = Message::PreVote(PreVote {
            term, // NOT incremented - this is the key difference from RequestVote
            candidate_id: self.node_id.clone(),
            last_log_index,
            last_log_term,
            state_embedding,
        });
        // Note: Broadcast handled by async version in production
    }

    /// Handle PreVote RPC.
    ///
    /// Grants pre-vote if:
    /// 1. Candidate's term >= our term
    /// 2. Election timeout has elapsed (no recent leader heartbeat)
    /// 3. Candidate's log is at least as up-to-date
    /// 4. Candidate is healthy (if membership configured)
    fn handle_pre_vote(&self, _from: &NodeId, pv: &PreVote) -> Option<Message> {
        let persistent = self.persistent.read();
        let mut vote_granted = false;

        // Check if candidate's term is at least as recent
        if pv.term >= persistent.current_term {
            // Check if election timeout has elapsed (no recent heartbeat from leader)
            let elapsed = self.last_heartbeat.read().elapsed().as_millis() as u64;
            let timeout_elapsed = elapsed > self.config.election_timeout.0;

            // Check candidate health
            let candidate_healthy = self.is_peer_healthy(&pv.candidate_id);

            // Check if candidate's log is at least as up-to-date
            let (last_log_index, last_log_term) = if persistent.log.is_empty() {
                (0, 0)
            } else {
                let last = &persistent.log[persistent.log.len() - 1];
                (last.index, last.term)
            };

            let log_ok = pv.last_log_term > last_log_term
                || (pv.last_log_term == last_log_term && pv.last_log_index >= last_log_index);

            // Grant pre-vote if all conditions are met
            // Unlike RequestVote, we don't update voted_for or term
            if timeout_elapsed && log_ok && candidate_healthy {
                vote_granted = true;
            }
        }

        Some(Message::PreVoteResponse(PreVoteResponse {
            term: persistent.current_term,
            vote_granted,
            voter_id: self.node_id.clone(),
        }))
    }

    /// Handle PreVoteResponse RPC.
    fn handle_pre_vote_response(&self, from: &NodeId, pvr: &PreVoteResponse) {
        // Only process if we're in pre-vote phase
        if !*self.in_pre_vote.read() {
            return;
        }

        let persistent = self.persistent.read();

        // If we see a higher term, step down
        if pvr.term > persistent.current_term {
            drop(persistent);

            // CRITICAL: Persist BEFORE applying state change
            if let Err(e) = self.persist_term_and_vote(pvr.term, None) {
                tracing::error!("WAL persist failed during pre-vote step-down: {}", e);
                return; // Don't step down if WAL fails
            }

            let mut persistent = self.persistent.write();
            persistent.current_term = pvr.term;
            persistent.voted_for = None;
            *self.state.write() = RaftState::Follower;
            *self.in_pre_vote.write() = false;
            return;
        }

        if pvr.vote_granted && pvr.term == persistent.current_term {
            let mut votes = self.pre_votes_received.write();
            if !votes.contains(from) {
                votes.push(from.clone());

                // Check if we have quorum
                if votes.len() >= self.quorum_size() {
                    drop(votes);
                    drop(persistent);
                    *self.in_pre_vote.write() = false;
                    // Pre-vote succeeded - now start real election
                    self.start_election();
                }
            }
        }
    }

    /// Check if a leadership transfer is currently in progress.
    pub fn is_transfer_in_progress(&self) -> bool {
        self.transfer_state.read().is_some()
    }

    /// Cancel an in-progress leadership transfer.
    pub fn cancel_transfer(&self) {
        *self.transfer_state.write() = None;
    }

    /// Initiate leadership transfer to target node.
    ///
    /// Returns error if:
    /// - This node is not the leader
    /// - Target is not a known peer
    /// - A transfer is already in progress
    pub fn transfer_leadership(&self, target: &NodeId) -> Result<()> {
        // Verify we're the leader
        if *self.state.read() != RaftState::Leader {
            return Err(ChainError::ConsensusError(
                "cannot transfer leadership: not leader".to_string(),
            ));
        }

        // Verify no transfer already in progress
        if self.is_transfer_in_progress() {
            return Err(ChainError::ConsensusError(
                "leadership transfer already in progress".to_string(),
            ));
        }

        // Verify target is a known peer
        if !self.peers.read().contains(target) {
            return Err(ChainError::ConsensusError(format!(
                "unknown transfer target: {}",
                target
            )));
        }

        // Set transfer state
        let term = self.persistent.read().current_term;
        *self.transfer_state.write() = Some(TransferState {
            target: target.clone(),
            initiated_term: term,
            started_at: Instant::now(),
        });

        Ok(())
    }

    /// Handle TimeoutNow RPC from leader initiating transfer.
    ///
    /// This causes the node to immediately start an election (skipping pre-vote).
    fn handle_timeout_now(&self, from: &NodeId, tn: &TimeoutNow) -> Option<Message> {
        // Verify sender is our believed leader
        let current_leader = self.current_leader.read().clone();
        if current_leader.as_ref() != Some(from) && current_leader.as_ref() != Some(&tn.leader_id) {
            return None;
        }

        // Verify term matches
        let term = self.persistent.read().current_term;
        if tn.term != term {
            return None;
        }

        // Start election immediately (skip pre-vote)
        self.start_election();

        None
    }

    /// Handle AppendEntries RPC.
    fn handle_append_entries(&self, _from: &NodeId, ae: &AppendEntries) -> Option<Message> {
        let mut persistent = self.persistent.write();
        let mut success = false;
        let mut match_index = 0;
        let mut used_fast_path = false;

        // Update term if needed
        if ae.term > persistent.current_term {
            // CRITICAL: Persist BEFORE applying state change
            if let Err(e) = self.persist_term_and_vote(ae.term, None) {
                tracing::error!("WAL persist failed during AppendEntries step-down: {}", e);
                // Return failure response with current term
                return Some(Message::AppendEntriesResponse(AppendEntriesResponse {
                    term: persistent.current_term,
                    success: false,
                    match_index: 0,
                    follower_id: self.node_id.clone(),
                    used_fast_path: false,
                }));
            }
            persistent.current_term = ae.term;
            persistent.voted_for = None;
            *self.state.write() = RaftState::Follower;
        }

        if ae.term == persistent.current_term {
            // Valid leader
            *self.state.write() = RaftState::Follower;

            // Detect leader change and clear stale embeddings
            let old_leader = self.current_leader.read().clone();
            if old_leader.as_ref() != Some(&ae.leader_id) {
                if let Some(ref old) = old_leader {
                    self.fast_path_state.clear_leader(old);
                }
                self.fast_path_validator.reset();
            }

            *self.current_leader.write() = Some(ae.leader_id.clone());
            *self.last_heartbeat.write() = Instant::now();

            // Check if we can use fast-path using the validator
            if self.config.enable_fast_path && ae.block_embedding.is_some() {
                let sparse_embedding = ae.block_embedding.as_ref().unwrap();
                // FastPathValidator still uses dense for similarity computation
                let dense_embedding = sparse_embedding.to_dense();
                let history = self.fast_path_state.get_embeddings(&ae.leader_id);
                let result = self
                    .fast_path_validator
                    .check_fast_path(&dense_embedding, &history);

                used_fast_path = result.can_use_fast_path;

                // Record statistics
                if used_fast_path {
                    self.fast_path_state.stats.record_fast_path();
                } else if result.rejection_reason.as_deref()
                    == Some("periodic full validation required")
                {
                    self.fast_path_state.stats.record_full_validation();
                } else {
                    self.fast_path_state.stats.record_rejected();
                }

                // Record this validation for periodic full validation tracking
                self.fast_path_validator.record_validation(used_fast_path);

                // Track embedding for future fast-path checks (store as sparse)
                self.fast_path_state
                    .add_embedding(&ae.leader_id, sparse_embedding.clone());
            }

            // Check log consistency
            let log_ok = if ae.prev_log_index == 0 {
                true
            } else if ae.prev_log_index <= persistent.log.len() as u64 {
                persistent.log[(ae.prev_log_index - 1) as usize].term == ae.prev_log_term
            } else {
                false
            };

            if log_ok {
                success = true;

                // Append new entries
                for entry in &ae.entries {
                    let idx = entry.index as usize;
                    if idx > persistent.log.len() {
                        persistent.log.push(entry.clone());
                    } else if persistent.log[idx - 1].term != entry.term {
                        // Conflict - remove existing entries
                        persistent.log.truncate(idx - 1);
                        persistent.log.push(entry.clone());
                    }
                }

                match_index = persistent.log.len() as u64;

                // Update commit index
                let mut volatile = self.volatile.write();
                if ae.leader_commit > volatile.commit_index {
                    volatile.commit_index = ae.leader_commit.min(persistent.log.len() as u64);
                }
            }
        }

        Some(Message::AppendEntriesResponse(AppendEntriesResponse {
            term: persistent.current_term,
            success,
            follower_id: self.node_id.clone(),
            match_index,
            used_fast_path,
        }))
    }

    /// Handle AppendEntriesResponse RPC.
    fn handle_append_entries_response(&self, from: &NodeId, aer: &AppendEntriesResponse) {
        if *self.state.read() != RaftState::Leader {
            return;
        }

        let mut persistent = self.persistent.write();
        if aer.term > persistent.current_term {
            // CRITICAL: Persist BEFORE applying state change
            if let Err(e) = self.persist_term_and_vote(aer.term, None) {
                tracing::error!("WAL persist failed during leader step-down: {}", e);
                return; // Don't step down if WAL fails
            }
            persistent.current_term = aer.term;
            persistent.voted_for = None;
            *self.state.write() = RaftState::Follower;
            *self.leader_state.write() = None;
            return;
        }

        let should_advance_commit = {
            let mut leader_state = self.leader_state.write();
            if let Some(ref mut ls) = *leader_state {
                if aer.success {
                    ls.next_index.insert(from.clone(), aer.match_index + 1);
                    ls.match_index.insert(from.clone(), aer.match_index);
                    true
                } else {
                    // Decrement next_index and retry
                    let next = ls.next_index.entry(from.clone()).or_insert(1);
                    if *next > 1 {
                        *next -= 1;
                    }
                    false
                }
            } else {
                false
            }
        };

        // Drop persistent before calling try_advance_commit_index to avoid deadlock
        drop(persistent);

        if should_advance_commit {
            self.try_advance_commit_index();
        }
    }

    /// Handle SnapshotRequest RPC from a follower requesting snapshot chunks.
    fn handle_snapshot_request(&self, _from: &NodeId, sr: &SnapshotRequest) -> Option<Message> {
        // Only leaders should respond to snapshot requests
        if *self.state.read() != RaftState::Leader {
            return None;
        }

        // Get current snapshot metadata and data
        let snapshot_meta = match self.get_snapshot_metadata() {
            Some(meta) => meta,
            None => return None, // No snapshot available
        };

        // Create snapshot data
        let (_, data) = match self.create_snapshot() {
            Ok(result) => result,
            Err(_) => return None,
        };

        // Calculate the chunk to send
        let offset = sr.offset;
        let chunk_size = sr.chunk_size.min(self.config.snapshot_chunk_size);
        let total_size = data.len() as u64;

        if offset >= total_size {
            return None; // Invalid offset
        }

        let end = ((offset + chunk_size) as usize).min(data.len());
        let chunk_data = data[offset as usize..end].to_vec();
        let is_last = end >= data.len();

        Some(Message::SnapshotResponse(SnapshotResponse {
            snapshot_height: snapshot_meta.last_included_index,
            snapshot_hash: snapshot_meta.snapshot_hash,
            data: chunk_data,
            offset,
            total_size,
            is_last,
        }))
    }

    /// Handle SnapshotResponse RPC when receiving snapshot chunks.
    fn handle_snapshot_response(&self, _from: &NodeId, sr: &SnapshotResponse) {
        // Only followers should process snapshot responses
        if *self.state.read() != RaftState::Follower {
            return;
        }

        // Receive the chunk
        let complete =
            match self.receive_snapshot_chunk(sr.offset, &sr.data, sr.total_size, sr.is_last) {
                Ok(complete) => complete,
                Err(_) => return, // Error handling chunk
            };

        if complete {
            // Get the accumulated data
            let data = self.take_pending_snapshot_data();

            // Find the term for this snapshot - it should be in our log or snapshot state
            let term = self
                .get_snapshot_metadata()
                .map(|m| m.last_included_term)
                .unwrap_or(1);

            // Create metadata for installation
            let metadata = SnapshotMetadata::new(
                sr.snapshot_height,
                term,
                sr.snapshot_hash,
                self.peers.read().clone(),
                sr.total_size,
            );

            // Install the snapshot
            if self.install_snapshot(metadata.clone(), &data).is_ok() {
                // Persist snapshot to store for crash recovery
                if let Some(ref store) = self.store {
                    if let Err(e) = self.save_snapshot(&metadata, &data, store) {
                        tracing::warn!("Failed to persist snapshot: {}", e);
                    } else {
                        tracing::info!(
                            "Follower persisted snapshot: index={}, term={}",
                            metadata.last_included_index,
                            metadata.last_included_term
                        );
                    }
                }
            }
        }
    }

    /// Start an election.
    pub fn start_election(&self) {
        let mut persistent = self.persistent.write();
        let new_term = persistent.current_term + 1;

        // CRITICAL: Persist BEFORE applying state change
        if let Err(e) = self.persist_term_and_vote(new_term, Some(&self.node_id)) {
            tracing::error!("WAL persist failed during election start: {}", e);
            return; // Abort election if WAL write fails
        }

        persistent.current_term = new_term;
        persistent.voted_for = Some(self.node_id.clone());

        *self.state.write() = RaftState::Candidate;
        *self.votes_received.write() = vec![self.node_id.clone()]; // Vote for self

        // Compute last log info while holding the write lock to avoid deadlock
        let (last_log_index, last_log_term) = if persistent.log.is_empty() {
            (0, 0)
        } else {
            let last = &persistent.log[persistent.log.len() - 1];
            (last.index, last.term)
        };
        let term = persistent.current_term;
        let state_embedding = self.state_embedding.read().clone();
        drop(persistent);

        // Request votes from all peers
        let request = Message::RequestVote(RequestVote {
            term,
            candidate_id: self.node_id.clone(),
            last_log_index,
            last_log_term,
            state_embedding,
        });

        // Note: In real implementation, this would be async
        // For now, we'll send synchronously in the test
        let _ = request; // TODO: broadcast
    }

    /// Become leader after winning election.
    /// This is public to allow testing scenarios.
    pub fn become_leader(&self) {
        *self.state.write() = RaftState::Leader;
        *self.current_leader.write() = Some(self.node_id.clone());

        // Initialize leader state
        let (last_log_index, _) = self.last_log_info();
        let peers = self.peers.read().clone();
        let mut next_index = HashMap::new();
        let mut match_index = HashMap::new();

        for peer in peers {
            next_index.insert(peer.clone(), last_log_index + 1);
            match_index.insert(peer, 0);
        }

        *self.leader_state.write() = Some(LeaderState {
            next_index,
            match_index,
        });
    }

    /// Try to advance the commit index (leader only).
    fn try_advance_commit_index(&self) {
        if *self.state.read() != RaftState::Leader {
            return;
        }

        let persistent = self.persistent.read();
        let leader_state = self.leader_state.read();
        let mut volatile = self.volatile.write();

        if let Some(ref ls) = *leader_state {
            // Find the highest N such that majority have match_index >= N
            let mut match_indices: Vec<u64> = ls.match_index.values().copied().collect();
            match_indices.push(persistent.log.len() as u64); // Include self
            match_indices.sort_unstable();

            let quorum_idx = match_indices.len() - self.quorum_size();
            let new_commit = match_indices[quorum_idx];

            // Only commit if entry is from current term
            if new_commit > volatile.commit_index {
                let entry_idx = (new_commit - 1) as usize;
                if entry_idx < persistent.log.len()
                    && persistent.log[entry_idx].term == persistent.current_term
                {
                    volatile.commit_index = new_commit;
                }
            }
        }
    }

    /// Propose a block (leader only).
    pub fn propose(&self, block: Block) -> Result<u64> {
        if *self.state.read() != RaftState::Leader {
            return Err(ChainError::ConsensusError("not leader".to_string()));
        }

        // Block proposals during leadership transfer
        if self.is_transfer_in_progress() {
            return Err(ChainError::ConsensusError(
                "leadership transfer in progress".to_string(),
            ));
        }

        // Capture sparse embedding before moving block
        let embedding = block.header.delta_embedding.clone();

        let mut persistent = self.persistent.write();
        let index = persistent.log.len() as u64 + 1;
        let term = persistent.current_term;

        let entry = LogEntry::new(term, index, block);
        persistent.log.push(entry);

        // Track embedding for fast-path validation by followers (already sparse)
        self.fast_path_state.add_embedding(&self.node_id, embedding);

        Ok(index)
    }

    /// Finalize committed entries up to a height.
    pub fn finalize_to(&self, height: u64) -> Result<()> {
        let commit_index = self.volatile.read().commit_index;
        if height > commit_index {
            return Err(ChainError::ConsensusError(format!(
                "cannot finalize {} above commit index {}",
                height, commit_index
            )));
        }

        self.finalized_height.store(height, Ordering::SeqCst);
        Ok(())
    }

    /// Get entries that need to be applied.
    pub fn get_uncommitted_entries(&self) -> Vec<LogEntry> {
        let persistent = self.persistent.read();
        let volatile = self.volatile.read();

        let start = volatile.last_applied as usize;
        let end = volatile.commit_index as usize;

        if end > start && end <= persistent.log.len() {
            persistent.log[start..end].to_vec()
        } else {
            Vec::new()
        }
    }

    /// Mark entries as applied.
    pub fn mark_applied(&self, up_to: u64) {
        let mut volatile = self.volatile.write();
        if up_to <= volatile.commit_index {
            volatile.last_applied = up_to;
        }
    }

    // ========== Snapshot / Log Compaction Methods ==========

    /// Check if log should be compacted based on configured threshold.
    pub fn should_compact(&self) -> bool {
        let persistent = self.persistent.read();
        let snapshot_state = self.snapshot_state.read();

        // Only compact if log exceeds threshold and we have finalized entries
        let log_len = persistent.log.len();
        let finalized = self.finalized_height.load(Ordering::SeqCst);

        if log_len < self.config.snapshot_threshold {
            return false;
        }

        // Check if we have entries to compact (finalized entries)
        match &snapshot_state.last_snapshot {
            Some(meta) => finalized > meta.last_included_index,
            None => finalized > 0,
        }
    }

    /// Create a snapshot using streaming serialization for memory efficiency.
    ///
    /// Uses SnapshotWriter to serialize log entries incrementally, avoiding
    /// the need to hold the entire serialized snapshot in memory.
    pub fn create_snapshot_streaming(&self) -> Result<(SnapshotMetadata, SnapshotBuffer)> {
        use crate::snapshot_streaming::SnapshotWriter;

        let finalized = self.finalized_height.load(Ordering::SeqCst);
        if finalized == 0 {
            return Err(ChainError::SnapshotError(
                "no finalized entries to snapshot".into(),
            ));
        }

        let persistent = self.persistent.read();

        // Find the log entry at finalized height
        let snapshot_idx = finalized.saturating_sub(1) as usize;
        if snapshot_idx >= persistent.log.len() {
            return Err(ChainError::SnapshotError(format!(
                "finalized height {} exceeds log length {}",
                finalized,
                persistent.log.len()
            )));
        }

        // Get buffer config from our snapshot state
        let buffer_config = self.snapshot_state.read().buffer_config.clone();

        // Create streaming writer and write entries incrementally
        let mut writer = SnapshotWriter::new(buffer_config).map_err(|e| {
            ChainError::SnapshotError(format!("failed to create snapshot writer: {}", e))
        })?;

        for entry in &persistent.log[..=snapshot_idx] {
            writer.write_entry(entry).map_err(|e| {
                ChainError::SnapshotError(format!("failed to write snapshot entry: {}", e))
            })?;
        }

        // Finalize the buffer
        let buffer = writer.finish().map_err(|e| {
            ChainError::SnapshotError(format!("failed to finalize snapshot: {}", e))
        })?;

        // Get hash from buffer
        let snapshot_hash = buffer.hash();

        // Get current cluster config
        let config = self.peers.read().clone();

        let entry = &persistent.log[snapshot_idx];
        let metadata = SnapshotMetadata::new(
            entry.index,
            entry.term,
            snapshot_hash,
            config,
            buffer.total_len(),
        );

        Ok((metadata, buffer))
    }

    /// Create a snapshot and return data as Vec<u8>.
    ///
    /// For large snapshots, prefer `create_snapshot_streaming()` which is more
    /// memory-efficient.
    pub fn create_snapshot(&self) -> Result<(SnapshotMetadata, Vec<u8>)> {
        let finalized = self.finalized_height.load(Ordering::SeqCst);
        if finalized == 0 {
            return Err(ChainError::SnapshotError(
                "no finalized entries to snapshot".into(),
            ));
        }

        let persistent = self.persistent.read();

        // Find the log entry at finalized height
        let snapshot_idx = finalized.saturating_sub(1) as usize;
        if snapshot_idx >= persistent.log.len() {
            return Err(ChainError::SnapshotError(format!(
                "finalized height {} exceeds log length {}",
                finalized,
                persistent.log.len()
            )));
        }

        let entry = &persistent.log[snapshot_idx];

        // Serialize the state up to finalized height
        // In a full implementation, this would serialize the state machine state
        // For now, we serialize the log entries themselves as the "state"
        let state_entries: Vec<LogEntry> = persistent.log[..=snapshot_idx].to_vec();
        let data = bincode::serialize(&state_entries)?;

        // Compute SHA-256 hash of snapshot data for integrity validation
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let snapshot_hash: [u8; 32] = hasher.finalize().into();

        // Get current cluster config
        let config = self.peers.read().clone();

        let metadata = SnapshotMetadata::new(
            entry.index,
            entry.term,
            snapshot_hash,
            config,
            data.len() as u64,
        );

        Ok((metadata, data))
    }

    /// Truncate log entries that are covered by a snapshot.
    ///
    /// Keeps `snapshot_trailing_logs` entries after the snapshot point
    /// to help followers catch up without needing a full snapshot transfer.
    pub fn truncate_log(&self, snapshot_meta: &SnapshotMetadata) -> Result<()> {
        let mut persistent = self.persistent.write();
        let mut snapshot_state = self.snapshot_state.write();

        let snapshot_idx = snapshot_meta.last_included_index;
        let trailing = self.config.snapshot_trailing_logs;

        // Find the index in our log array
        // Log entries are 1-indexed, array is 0-indexed
        let cut_point = if snapshot_idx as usize > trailing {
            snapshot_idx as usize - trailing
        } else {
            0
        };

        if cut_point > 0 && cut_point < persistent.log.len() {
            // Remove entries before the cut point
            persistent.log.drain(..cut_point);
        }

        // Update snapshot state
        snapshot_state.last_snapshot = Some(snapshot_meta.clone());

        Ok(())
    }

    // ========== Automatic Compaction Methods ==========

    /// Try to perform automatic log compaction if conditions are met.
    ///
    /// Only leaders should call this. Compaction happens if:
    /// 1. Check interval has been reached
    /// 2. Cooldown period has elapsed
    /// 3. Log exceeds snapshot_threshold
    /// 4. Finalized entries exist beyond last snapshot
    async fn try_auto_compact(&self) -> Result<()> {
        // Check interval counter (only check every N ticks)
        let tick_count = self.compaction_tick_counter.fetch_add(1, Ordering::Relaxed);
        if tick_count % self.config.compaction_check_interval != 0 {
            return Ok(());
        }

        // Check cooldown
        if !self.can_compact() {
            return Ok(());
        }

        // Check if compaction is needed
        if !self.should_compact() {
            return Ok(());
        }

        // Perform compaction
        self.perform_compaction().await
    }

    /// Perform the actual compaction operation.
    ///
    /// Steps:
    /// 1. Create snapshot
    /// 2. Persist snapshot to store (if available)
    /// 3. Truncate log (only after successful persistence)
    /// 4. Update cooldown timestamp
    async fn perform_compaction(&self) -> Result<()> {
        // Create snapshot (reads log)
        let (metadata, data) = self.create_snapshot()?;

        // Persist snapshot BEFORE truncating (critical for atomicity)
        if let Some(ref store) = self.store {
            self.save_snapshot(&metadata, &data, store)?;
        }

        // Safety check: verify snapshot index is still valid before truncating
        // (log only grows, but be defensive against concurrent modifications)
        {
            let persistent = self.persistent.read();
            if metadata.last_included_index as usize > persistent.log.len() {
                return Err(ChainError::SnapshotError(
                    "snapshot index no longer valid".to_string(),
                ));
            }
        }

        // Truncate log (safe now that snapshot is persisted and index verified)
        self.truncate_log(&metadata)?;

        // Update cooldown timestamp
        self.mark_compacted();

        // Persist updated Raft state
        if let Some(ref store) = self.store {
            self.save_to_store(store)?;
        }

        Ok(())
    }

    /// Get current snapshot metadata if available.
    pub fn get_snapshot_metadata(&self) -> Option<SnapshotMetadata> {
        self.snapshot_state.read().last_snapshot.clone()
    }

    /// Install a snapshot from a memory-efficient buffer.
    ///
    /// Uses SnapshotReader for streaming deserialization, reading entries
    /// one at a time to minimize peak memory usage.
    pub fn install_snapshot_streaming(
        &self,
        metadata: SnapshotMetadata,
        buffer: &SnapshotBuffer,
    ) -> Result<()> {
        use crate::snapshot_streaming::SnapshotReader;

        // Validate snapshot hash
        let computed_hash = buffer.hash();
        if computed_hash != metadata.snapshot_hash {
            return Err(ChainError::SnapshotError(format!(
                "snapshot hash mismatch: expected {:?}, got {:?}",
                metadata.snapshot_hash, computed_hash
            )));
        }

        // Create reader for streaming deserialization
        let reader = SnapshotReader::new(buffer).map_err(|e| {
            ChainError::SnapshotError(format!("failed to create snapshot reader: {}", e))
        })?;

        // Read entries incrementally
        let mut entries = Vec::with_capacity(reader.entry_count() as usize);
        for entry_result in reader {
            let entry = entry_result.map_err(|e| {
                ChainError::SnapshotError(format!("failed to read snapshot entry: {}", e))
            })?;
            entries.push(entry);
        }

        // Validate the snapshot
        if entries.is_empty() {
            return Err(ChainError::SnapshotError(
                "snapshot contains no entries".into(),
            ));
        }

        let last_entry = entries.last().unwrap();
        if last_entry.index != metadata.last_included_index {
            return Err(ChainError::SnapshotError(format!(
                "snapshot index mismatch: expected {}, got {}",
                metadata.last_included_index, last_entry.index
            )));
        }
        if last_entry.term != metadata.last_included_term {
            return Err(ChainError::SnapshotError(format!(
                "snapshot term mismatch: expected {}, got {}",
                metadata.last_included_term, last_entry.term
            )));
        }

        // Install the snapshot
        self.install_snapshot_entries(metadata, entries)
    }

    /// Install a snapshot from raw bytes.
    ///
    /// For large snapshots, prefer `install_snapshot_streaming()` with a
    /// SnapshotBuffer for better memory efficiency.
    pub fn install_snapshot(&self, metadata: SnapshotMetadata, data: &[u8]) -> Result<()> {
        use crate::snapshot_streaming::deserialize_entries;

        // Validate snapshot hash before installing
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data);
        let computed_hash: [u8; 32] = hasher.finalize().into();

        if computed_hash != metadata.snapshot_hash {
            return Err(ChainError::SnapshotError(format!(
                "snapshot hash mismatch: expected {:?}, got {:?}",
                metadata.snapshot_hash, computed_hash
            )));
        }

        // Deserialize the log entries from snapshot data (supports both legacy and streaming formats)
        let entries: Vec<LogEntry> = deserialize_entries(data).map_err(|e| {
            ChainError::SnapshotError(format!("failed to deserialize snapshot: {}", e))
        })?;

        // Validate the snapshot
        if entries.is_empty() {
            return Err(ChainError::SnapshotError(
                "snapshot contains no entries".into(),
            ));
        }

        let last_entry = entries.last().unwrap();
        if last_entry.index != metadata.last_included_index {
            return Err(ChainError::SnapshotError(format!(
                "snapshot index mismatch: expected {}, got {}",
                metadata.last_included_index, last_entry.index
            )));
        }
        if last_entry.term != metadata.last_included_term {
            return Err(ChainError::SnapshotError(format!(
                "snapshot term mismatch: expected {}, got {}",
                metadata.last_included_term, last_entry.term
            )));
        }

        // Install the snapshot
        self.install_snapshot_entries(metadata, entries)
    }

    /// Install snapshot entries into the Raft state.
    fn install_snapshot_entries(
        &self,
        metadata: SnapshotMetadata,
        entries: Vec<LogEntry>,
    ) -> Result<()> {
        // Install the snapshot
        {
            let mut persistent = self.persistent.write();
            let mut volatile = self.volatile.write();
            let mut snapshot_state = self.snapshot_state.write();

            // Replace log with entries from snapshot
            persistent.log = entries;

            // Update term if snapshot has higher term
            if metadata.last_included_term > persistent.current_term {
                persistent.current_term = metadata.last_included_term;
                persistent.voted_for = None;
            }

            // Update commit/apply indices
            volatile.commit_index = metadata.last_included_index;
            volatile.last_applied = metadata.last_included_index;

            // Update snapshot metadata
            snapshot_state.last_snapshot = Some(metadata.clone());
            snapshot_state.cancel_receive(); // Clear any pending transfer
        }

        // Update finalized height
        self.finalized_height
            .store(metadata.last_included_index, Ordering::SeqCst);

        // Update peers from snapshot config
        {
            let mut peers = self.peers.write();
            *peers = metadata.config;
        }

        Ok(())
    }

    /// Receive a snapshot chunk during transfer.
    ///
    /// Call this for each chunk received. Returns true when snapshot is complete.
    pub fn receive_snapshot_chunk(
        &self,
        offset: u64,
        data: &[u8],
        total_size: u64,
        is_last: bool,
    ) -> Result<bool> {
        let mut snapshot_state = self.snapshot_state.write();

        // Start new transfer if offset is 0
        if offset == 0 {
            snapshot_state.start_receive(total_size).map_err(|e| {
                ChainError::SnapshotError(format!("failed to start snapshot receive: {}", e))
            })?;
        }

        // Get current buffer size for offset validation
        let current_size = snapshot_state
            .pending_buffer
            .as_ref()
            .map(|b| b.total_len())
            .unwrap_or(0);

        // Validate offset matches current position
        if offset != current_size {
            snapshot_state.cancel_receive();
            return Err(ChainError::SnapshotError(format!(
                "chunk offset mismatch: expected {}, got {}",
                current_size, offset
            )));
        }

        snapshot_state.append_chunk(data).map_err(|e| {
            snapshot_state.cancel_receive();
            ChainError::SnapshotError(format!("failed to append chunk: {}", e))
        })?;

        if is_last {
            // Validate total size
            let actual_size = snapshot_state
                .pending_buffer
                .as_ref()
                .map(|b| b.total_len())
                .unwrap_or(0);
            if actual_size != total_size {
                snapshot_state.cancel_receive();
                return Err(ChainError::SnapshotError(format!(
                    "snapshot size mismatch: expected {}, got {}",
                    total_size, actual_size
                )));
            }
            return Ok(true);
        }

        Ok(false)
    }

    /// Get the accumulated snapshot buffer after receiving all chunks.
    pub fn take_pending_snapshot_buffer(&self) -> Option<SnapshotBuffer> {
        self.snapshot_state.write().finish_receive()
    }

    /// Get the accumulated snapshot data after receiving all chunks.
    ///
    /// Returns a copy of the data as a Vec<u8> for backwards compatibility.
    pub fn take_pending_snapshot_data(&self) -> Vec<u8> {
        self.snapshot_state
            .write()
            .finish_receive()
            .and_then(|buf| buf.as_bytes().ok().map(|b| b.to_vec()))
            .unwrap_or_default()
    }

    /// Check if we need to send a snapshot to a follower.
    ///
    /// Returns true if the follower's next_index is before our first log entry.
    pub fn needs_snapshot_for_follower(&self, follower: &NodeId) -> bool {
        let leader_state = self.leader_state.read();
        let persistent = self.persistent.read();
        let snapshot_state = self.snapshot_state.read();

        let next_idx = leader_state
            .as_ref()
            .and_then(|ls| ls.next_index.get(follower))
            .copied()
            .unwrap_or(1);

        // If we have no log entries, we might need snapshot
        if persistent.log.is_empty() {
            return snapshot_state.last_snapshot.is_some();
        }

        // Check if follower needs entries before our first log entry
        let first_log_index = persistent.log.first().map(|e| e.index).unwrap_or(1);
        next_idx < first_log_index && snapshot_state.last_snapshot.is_some()
    }

    /// Get snapshot chunks for transfer to a follower.
    ///
    /// Returns a vector of (offset, data, is_last) chunks.
    pub fn get_snapshot_chunks(&self, data: &[u8]) -> Vec<(u64, Vec<u8>, bool)> {
        let chunk_size = self.config.snapshot_chunk_size as usize;
        let total_chunks = (data.len() + chunk_size - 1) / chunk_size;

        data.chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let offset = (i * chunk_size) as u64;
                let is_last = i == total_chunks - 1;
                (offset, chunk.to_vec(), is_last)
            })
            .collect()
    }

    /// Get a single chunk from a SnapshotBuffer with zero-copy access.
    ///
    /// Returns (data_slice, is_last) for the chunk at the given offset.
    /// Uses memory-mapped I/O when the buffer is file-backed, avoiding copies.
    pub fn get_snapshot_chunk_streaming<'a>(
        &self,
        buffer: &'a SnapshotBuffer,
        offset: u64,
    ) -> Result<(&'a [u8], bool)> {
        let total_len = buffer.total_len();
        if offset >= total_len {
            return Err(ChainError::SnapshotError(format!(
                "chunk offset {} beyond buffer length {}",
                offset, total_len
            )));
        }

        let chunk_size = self.config.snapshot_chunk_size as usize;
        let remaining = (total_len - offset) as usize;
        let actual_size = chunk_size.min(remaining);
        let is_last = offset + actual_size as u64 >= total_len;

        let data = buffer.as_slice(offset, actual_size).map_err(|e| {
            ChainError::SnapshotError(format!("failed to read chunk at offset {}: {}", offset, e))
        })?;

        Ok((data, is_last))
    }

    /// Iterator over snapshot chunks from a SnapshotBuffer.
    ///
    /// Yields (offset, data_slice, is_last) for each chunk.
    /// Uses zero-copy access when possible.
    pub fn snapshot_chunk_iter<'a>(
        &'a self,
        buffer: &'a SnapshotBuffer,
    ) -> impl Iterator<Item = Result<(u64, &'a [u8], bool)>> + 'a {
        let chunk_size = self.config.snapshot_chunk_size;
        let total_len = buffer.total_len();
        let num_chunks = (total_len + chunk_size - 1) / chunk_size;

        (0..num_chunks).map(move |i| {
            let offset = i * chunk_size;
            let (data, is_last) = self.get_snapshot_chunk_streaming(buffer, offset)?;
            Ok((offset, data, is_last))
        })
    }

    /// Get entries for replication to a specific follower.
    ///
    /// Returns (prev_log_index, prev_log_term, entries, block_embedding).
    /// The block_embedding is from the last entry if any, for fast-path validation.
    pub fn get_entries_for_follower(
        &self,
        follower: &NodeId,
    ) -> (u64, u64, Vec<LogEntry>, Option<SparseVector>) {
        let persistent = self.persistent.read();
        let leader_state = self.leader_state.read();

        let next_idx = leader_state
            .as_ref()
            .and_then(|ls| ls.next_index.get(follower))
            .copied()
            .unwrap_or(1);

        let (prev_log_index, prev_log_term) = if next_idx <= 1 {
            (0, 0)
        } else {
            let idx = (next_idx - 2) as usize;
            if idx < persistent.log.len() {
                (persistent.log[idx].index, persistent.log[idx].term)
            } else {
                (0, 0)
            }
        };

        let start = (next_idx - 1) as usize;
        let entries = if start < persistent.log.len() {
            persistent.log[start..].to_vec()
        } else {
            Vec::new()
        };

        // Extract embedding from last entry for fast-path (already sparse)
        let block_embedding = entries
            .last()
            .map(|e| e.block.header.delta_embedding.clone());

        (prev_log_index, prev_log_term, entries, block_embedding)
    }

    // ========== Async Transport Methods ==========

    /// Send a message to a specific peer via transport.
    pub async fn send_to_peer(&self, peer: &NodeId, msg: Message) -> Result<()> {
        self.transport.send(peer, msg).await
    }

    /// Broadcast a message to all peers via transport.
    pub async fn broadcast_to_peers(&self, msg: Message) -> Result<()> {
        self.transport.broadcast(msg).await
    }

    /// Start an election and broadcast RequestVote to all peers.
    pub async fn start_election_async(&self) -> Result<()> {
        // Build the RequestVote message in a sync block (no await)
        let request = {
            let mut persistent = self.persistent.write();
            persistent.current_term += 1;
            persistent.voted_for = Some(self.node_id.clone());

            *self.state.write() = RaftState::Candidate;
            *self.votes_received.write() = vec![self.node_id.clone()]; // Vote for self

            let (last_log_index, last_log_term) = if persistent.log.is_empty() {
                (0, 0)
            } else {
                let last = &persistent.log[persistent.log.len() - 1];
                (last.index, last.term)
            };
            let term = persistent.current_term;
            let state_embedding = self.state_embedding.read().clone();

            Message::RequestVote(RequestVote {
                term,
                candidate_id: self.node_id.clone(),
                last_log_index,
                last_log_term,
                state_embedding,
            })
        }; // All locks dropped here

        // Now broadcast via transport (async)
        self.transport.broadcast(request).await
    }

    /// Start pre-vote phase and broadcast PreVote to all peers.
    ///
    /// Pre-vote prevents disruptive elections from partitioned nodes by requiring
    /// candidates to confirm they can win before incrementing their term.
    pub async fn start_pre_vote_async(&self) -> Result<()> {
        // Build the PreVote message in a sync block (no await)
        let request = {
            *self.in_pre_vote.write() = true;
            *self.pre_votes_received.write() = vec![self.node_id.clone()]; // Vote for self

            let persistent = self.persistent.read();
            let (last_log_index, last_log_term) = if persistent.log.is_empty() {
                (0, 0)
            } else {
                let last = &persistent.log[persistent.log.len() - 1];
                (last.index, last.term)
            };
            let term = persistent.current_term;
            let state_embedding = self.state_embedding.read().clone();

            Message::PreVote(PreVote {
                term, // NOT incremented - this is the key difference from RequestVote
                candidate_id: self.node_id.clone(),
                last_log_index,
                last_log_term,
                state_embedding,
            })
        }; // All locks dropped here

        // Now broadcast via transport (async)
        self.transport.broadcast(request).await
    }

    /// Receive a message from transport (test helper).
    pub async fn transport_recv(&self) -> Result<(NodeId, Message)> {
        self.transport.recv().await
    }

    /// Initiate leadership transfer to target node (async version).
    ///
    /// This sends a heartbeat to ensure the target is caught up, then sends
    /// TimeoutNow to trigger an immediate election on the target.
    pub async fn transfer_leadership_async(&self, target: &NodeId) -> Result<()> {
        // Do the sync validation and state setup
        self.transfer_leadership(target)?;

        // Send heartbeat to ensure target is caught up
        let term = self.persistent.read().current_term;
        let commit_index = self.volatile.read().commit_index;
        let (prev_log_index, prev_log_term, entries, block_embedding) =
            self.get_entries_for_follower(target);

        let heartbeat = Message::AppendEntries(AppendEntries {
            term,
            leader_id: self.node_id.clone(),
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit: commit_index,
            block_embedding,
        });
        self.transport.send(target, heartbeat).await?;

        // Send TimeoutNow to trigger immediate election
        let timeout_now = Message::TimeoutNow(TimeoutNow {
            term,
            leader_id: self.node_id.clone(),
        });
        self.transport.send(target, timeout_now).await?;

        Ok(())
    }

    /// Send heartbeats (AppendEntries) to all followers.
    pub async fn send_heartbeats(&self) -> Result<()> {
        // Build all messages in a sync block (no await)
        let messages: Vec<(NodeId, Message)> = {
            if *self.state.read() != RaftState::Leader {
                return Ok(());
            }

            let peers = self.peers.read().clone();
            let term = self.persistent.read().current_term;
            let commit_index = self.volatile.read().commit_index;

            peers
                .into_iter()
                // Skip unhealthy peers - they won't respond anyway
                .filter(|peer| self.is_peer_healthy(peer))
                .map(|peer| {
                    let (prev_log_index, prev_log_term, entries, block_embedding) =
                        self.get_entries_for_follower(&peer);

                    let ae = Message::AppendEntries(AppendEntries {
                        term,
                        leader_id: self.node_id.clone(),
                        prev_log_index,
                        prev_log_term,
                        entries,
                        leader_commit: commit_index,
                        block_embedding,
                    });

                    (peer, ae)
                })
                .collect()
        }; // All locks dropped here

        // Now send messages (async)
        for (peer, msg) in messages {
            // Send to peer, ignore failures (peer may be down)
            let _ = self.transport.send(&peer, msg).await;
        }

        Ok(())
    }

    /// Propose a block and replicate to followers (async version).
    pub async fn propose_async(&self, block: Block) -> Result<u64> {
        // First, add to local log (sync part)
        let index = self.propose(block)?;

        // Then replicate to followers
        self.send_heartbeats().await?;

        Ok(index)
    }

    /// Handle incoming message and optionally send response.
    pub async fn handle_message_async(&self, from: &NodeId, msg: Message) -> Result<()> {
        if let Some(response) = self.handle_message(from, &msg) {
            self.transport.send(from, response).await?;
        }
        Ok(())
    }

    /// Tick the Raft node - check for election timeout (async version).
    pub async fn tick_async(&self) -> Result<()> {
        let elapsed = self.last_heartbeat.read().elapsed().as_millis() as u64;
        let state = *self.state.read();

        match state {
            RaftState::Follower | RaftState::Candidate => {
                // Check election timeout
                let timeout = self.config.election_timeout.0
                    + rand::random::<u64>()
                        % (self.config.election_timeout.1 - self.config.election_timeout.0);
                if elapsed > timeout {
                    // Use pre-vote if enabled to prevent disruptive elections
                    if self.config.enable_pre_vote {
                        self.start_pre_vote_async().await?;
                    } else {
                        self.start_election_async().await?;
                    }
                }
            },
            RaftState::Leader => {
                // Check if leadership transfer has timed out
                let should_cancel = {
                    if let Some(ref transfer) = *self.transfer_state.read() {
                        transfer.started_at.elapsed().as_millis() as u64
                            > self.config.transfer_timeout_ms
                    } else {
                        false
                    }
                };
                if should_cancel {
                    self.cancel_transfer();
                }

                // Send heartbeats
                if elapsed > self.config.heartbeat_interval {
                    self.send_heartbeats().await?;
                    *self.last_heartbeat.write() = Instant::now();
                }

                // Automatic log compaction check
                self.try_auto_compact().await?;
            },
        }
        Ok(())
    }

    /// Main event loop - process messages and handle timeouts.
    ///
    /// This runs the Raft protocol, handling:
    /// - Incoming messages from peers
    /// - Election timeouts (start election if no heartbeat)
    /// - Heartbeat sending (if leader)
    pub async fn run(&self, mut shutdown: tokio::sync::broadcast::Receiver<()>) -> Result<()> {
        let tick_interval = std::time::Duration::from_millis(self.config.heartbeat_interval / 2);
        let mut ticker = tokio::time::interval(tick_interval);

        loop {
            tokio::select! {
                // Handle shutdown signal
                _ = shutdown.recv() => {
                    break;
                }

                // Periodic tick for elections/heartbeats
                _ = ticker.tick() => {
                    self.tick_async().await?;
                }

                // Handle incoming messages
                result = self.transport.recv() => {
                    match result {
                        Ok((from, msg)) => {
                            self.handle_message_async(&from, msg).await?;
                        }
                        Err(e) => {
                            // Log error but continue running
                            tracing::warn!("Error receiving message: {}", e);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{block::BlockHeader, network::MemoryTransport};

    fn create_test_node(id: &str, peers: Vec<String>) -> RaftNode {
        let transport = Arc::new(MemoryTransport::new(id.to_string()));
        RaftNode::new(id.to_string(), peers, transport, RaftConfig::default())
    }

    fn create_test_block(height: u64) -> Block {
        let header = BlockHeader::new(
            height,
            [0u8; 32],
            [0u8; 32],
            [0u8; 32],
            "proposer".to_string(),
        );
        Block::new(header, vec![])
    }

    fn create_test_log_entry(index: u64) -> LogEntry {
        LogEntry::new(1, index, create_test_block(index))
    }

    #[test]
    fn test_raft_node_creation() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        assert_eq!(node.node_id(), "node1");
        assert_eq!(node.state(), RaftState::Follower);
        assert_eq!(node.current_term(), 0);
        assert!(!node.is_leader());
    }

    #[test]
    fn test_request_vote_handling() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Initial vote request from a candidate
        let rv = RequestVote {
            term: 1,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
            state_embedding: SparseVector::new(0),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::RequestVote(rv));

        if let Some(Message::RequestVoteResponse(rvr)) = response {
            assert!(rvr.vote_granted);
            assert_eq!(rvr.term, 1);
        } else {
            panic!("expected RequestVoteResponse");
        }

        // Node should have updated term
        assert_eq!(node.current_term(), 1);
    }

    #[test]
    fn test_append_entries_heartbeat() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: None,
        };

        let response = node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(aer.success);
            assert_eq!(aer.term, 1);
        } else {
            panic!("expected AppendEntriesResponse");
        }

        // Node should recognize leader
        assert_eq!(node.current_leader(), Some("node2".to_string()));
    }

    #[test]
    fn test_leader_proposal() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Make node the leader
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 1;
        }
        node.become_leader();

        assert!(node.is_leader());

        // Propose a block
        let block = create_test_block(1);
        let index = node.propose(block).unwrap();

        assert_eq!(index, 1);
        assert_eq!(node.log_length(), 1);
    }

    #[test]
    fn test_finalization() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Setup: become leader and commit an entry
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 1;
            let block = create_test_block(1);
            persistent.log.push(LogEntry::new(1, 1, block));
        }
        {
            let mut volatile = node.volatile.write();
            volatile.commit_index = 1;
        }

        // Finalize
        node.finalize_to(1).unwrap();
        assert_eq!(node.finalized_height(), 1);

        // Cannot finalize above commit
        assert!(node.finalize_to(2).is_err());
    }

    #[test]
    fn test_uncommitted_entries() {
        let node = create_test_node("node1", vec![]);

        // Add some entries
        {
            let mut persistent = node.persistent.write();
            for i in 1..=3 {
                let block = create_test_block(i);
                persistent.log.push(LogEntry::new(1, i, block));
            }
        }
        {
            let mut volatile = node.volatile.write();
            volatile.commit_index = 2;
            volatile.last_applied = 0;
        }

        let uncommitted = node.get_uncommitted_entries();
        assert_eq!(uncommitted.len(), 2);
        assert_eq!(uncommitted[0].index, 1);
        assert_eq!(uncommitted[1].index, 2);

        node.mark_applied(2);
        let uncommitted = node.get_uncommitted_entries();
        assert!(uncommitted.is_empty());
    }

    #[test]
    fn test_quorum_calculation() {
        // 3 nodes (self + 2 peers) -> quorum = 2
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);
        assert_eq!(node.quorum_size(), 2);

        // 5 nodes -> quorum = 3
        let node = create_test_node(
            "node1",
            vec![
                "node2".to_string(),
                "node3".to_string(),
                "node4".to_string(),
                "node5".to_string(),
            ],
        );
        assert_eq!(node.quorum_size(), 3);
    }

    #[test]
    fn test_fast_path_stats_new() {
        let stats = FastPathStats::new();
        assert_eq!(stats.fast_path_accepted.load(Ordering::Relaxed), 0);
        assert_eq!(stats.fast_path_rejected.load(Ordering::Relaxed), 0);
        assert_eq!(stats.full_validation_required.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_fast_path_stats_record() {
        let stats = FastPathStats::new();

        stats.record_fast_path();
        stats.record_fast_path();
        stats.record_rejected();
        stats.record_full_validation();

        assert_eq!(stats.fast_path_accepted.load(Ordering::Relaxed), 2);
        assert_eq!(stats.fast_path_rejected.load(Ordering::Relaxed), 1);
        assert_eq!(stats.full_validation_required.load(Ordering::Relaxed), 1);
        assert_eq!(stats.total_validated(), 4);
    }

    #[test]
    fn test_fast_path_stats_acceptance_rate() {
        let stats = FastPathStats::new();

        // Empty stats should give 0 rate
        assert_eq!(stats.acceptance_rate(), 0.0);

        // Record some stats
        stats.record_fast_path();
        stats.record_fast_path();
        stats.record_fast_path();
        stats.record_rejected();

        // 3 accepted, 1 rejected = 75% acceptance rate
        assert!((stats.acceptance_rate() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_fast_path_state_new() {
        let state = FastPathState::new(5);
        assert_eq!(state.max_history, 5);
        assert_eq!(state.leader_history_size(&"leader1".to_string()), 0);
    }

    #[test]
    fn test_fast_path_state_add_embedding() {
        let state = FastPathState::new(3);
        let leader = "leader1".to_string();

        state.add_dense_embedding(&leader, vec![1.0, 0.0, 0.0]);
        state.add_dense_embedding(&leader, vec![0.0, 1.0, 0.0]);

        assert_eq!(state.leader_history_size(&leader), 2);

        let embeddings = state.get_embeddings(&leader);
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(embeddings[1], vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_fast_path_state_max_history() {
        let state = FastPathState::new(2);
        let leader = "leader1".to_string();

        state.add_dense_embedding(&leader, vec![1.0, 0.0, 0.0]);
        state.add_dense_embedding(&leader, vec![0.0, 1.0, 0.0]);
        state.add_dense_embedding(&leader, vec![0.0, 0.0, 1.0]);

        // Should only keep last 2
        assert_eq!(state.leader_history_size(&leader), 2);

        let embeddings = state.get_embeddings(&leader);
        assert_eq!(embeddings[0], vec![0.0, 1.0, 0.0]);
        assert_eq!(embeddings[1], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_fast_path_state_multiple_leaders() {
        let state = FastPathState::new(5);

        state.add_dense_embedding(&"leader1".to_string(), vec![1.0, 0.0]);
        state.add_dense_embedding(&"leader2".to_string(), vec![0.0, 1.0]);

        assert_eq!(state.leader_history_size(&"leader1".to_string()), 1);
        assert_eq!(state.leader_history_size(&"leader2".to_string()), 1);
    }

    #[test]
    fn test_fast_path_state_clear_leader() {
        let state = FastPathState::new(5);
        let leader = "leader1".to_string();

        state.add_dense_embedding(&leader, vec![1.0, 0.0]);
        assert_eq!(state.leader_history_size(&leader), 1);

        state.clear_leader(&leader);
        assert_eq!(state.leader_history_size(&leader), 0);
    }

    #[test]
    fn test_raft_node_fast_path_stats() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Initially no stats
        let stats = node.fast_path_stats();
        assert_eq!(stats.total_validated(), 0);
    }

    #[test]
    fn test_start_election() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        // Start election
        node.start_election();

        // Should become candidate with incremented term
        assert_eq!(node.state(), RaftState::Candidate);
        assert_eq!(node.current_term(), 1);

        // Should have voted for self
        let votes = node.votes_received.read();
        assert_eq!(votes.len(), 1);
        assert!(votes.contains(&"node1".to_string()));
    }

    #[test]
    fn test_handle_request_vote_response_higher_term() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        // Start election
        node.start_election();
        assert_eq!(node.state(), RaftState::Candidate);
        assert_eq!(node.current_term(), 1);

        // Receive response with higher term
        let rvr = RequestVoteResponse {
            term: 5,
            vote_granted: false,
            voter_id: "node2".to_string(),
        };

        node.handle_message(&"node2".to_string(), &Message::RequestVoteResponse(rvr));

        // Should step down to follower with updated term
        assert_eq!(node.state(), RaftState::Follower);
        assert_eq!(node.current_term(), 5);
    }

    #[test]
    fn test_handle_request_vote_response_gains_quorum() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        // Start election
        node.start_election();

        // Receive vote from node2
        let rvr = RequestVoteResponse {
            term: 1,
            vote_granted: true,
            voter_id: "node2".to_string(),
        };

        node.handle_message(&"node2".to_string(), &Message::RequestVoteResponse(rvr));

        // Should become leader (has 2 votes: self + node2, quorum is 2)
        assert_eq!(node.state(), RaftState::Leader);
        assert!(node.is_leader());
    }

    #[test]
    fn test_handle_request_vote_response_not_candidate() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Node is follower, not candidate
        let rvr = RequestVoteResponse {
            term: 1,
            vote_granted: true,
            voter_id: "node2".to_string(),
        };

        // Should be ignored
        node.handle_message(&"node2".to_string(), &Message::RequestVoteResponse(rvr));
        assert_eq!(node.state(), RaftState::Follower);
    }

    #[test]
    fn test_handle_append_entries_response_not_leader() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Node is follower
        let aer = AppendEntriesResponse {
            term: 1,
            success: true,
            follower_id: "node2".to_string(),
            match_index: 1,
            used_fast_path: false,
        };

        // Should be ignored
        node.handle_message(&"node2".to_string(), &Message::AppendEntriesResponse(aer));
        assert_eq!(node.state(), RaftState::Follower);
    }

    #[test]
    fn test_handle_append_entries_response_higher_term() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Make leader
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 1;
        }
        node.become_leader();
        assert!(node.is_leader());

        // Receive response with higher term
        let aer = AppendEntriesResponse {
            term: 5,
            success: false,
            follower_id: "node2".to_string(),
            match_index: 0,
            used_fast_path: false,
        };

        node.handle_message(&"node2".to_string(), &Message::AppendEntriesResponse(aer));

        // Should step down
        assert_eq!(node.state(), RaftState::Follower);
        assert_eq!(node.current_term(), 5);
    }

    #[test]
    fn test_handle_append_entries_response_failure_decrements_next_index() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Make leader with some log entries
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 1;
            for i in 1..=3 {
                persistent
                    .log
                    .push(LogEntry::new(1, i, create_test_block(i)));
            }
        }
        node.become_leader();

        // Get initial next_index for node2
        let initial_next = {
            let ls = node.leader_state.read();
            ls.as_ref()
                .unwrap()
                .next_index
                .get(&"node2".to_string())
                .copied()
                .unwrap()
        };

        // Receive failure response
        let aer = AppendEntriesResponse {
            term: 1,
            success: false,
            follower_id: "node2".to_string(),
            match_index: 0,
            used_fast_path: false,
        };

        node.handle_message(&"node2".to_string(), &Message::AppendEntriesResponse(aer));

        // next_index should be decremented
        let new_next = {
            let ls = node.leader_state.read();
            ls.as_ref()
                .unwrap()
                .next_index
                .get(&"node2".to_string())
                .copied()
                .unwrap()
        };

        assert_eq!(new_next, initial_next - 1);
    }

    #[test]
    fn test_try_advance_commit_index() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        // Make leader with log entries
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 1;
            for i in 1..=3 {
                persistent
                    .log
                    .push(LogEntry::new(1, i, create_test_block(i)));
            }
        }
        node.become_leader();

        // Simulate successful replication to node2
        {
            let mut ls = node.leader_state.write();
            if let Some(ref mut state) = *ls {
                state.match_index.insert("node2".to_string(), 2);
            }
        }

        node.try_advance_commit_index();

        // With 3 nodes, quorum is 2. Leader has 3, node2 has 2 -> commit = 2
        assert_eq!(node.commit_index(), 2);
    }

    #[test]
    fn test_handle_append_entries_log_conflict() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Add an entry with term 1
        {
            let mut persistent = node.persistent.write();
            persistent
                .log
                .push(LogEntry::new(1, 1, create_test_block(1)));
        }

        // Leader sends entry at same index but different term
        let ae = AppendEntries {
            term: 2,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry::new(2, 1, create_test_block(1))],
            leader_commit: 0,
            block_embedding: None,
        };

        let response = node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(aer.success);
        } else {
            panic!("expected AppendEntriesResponse");
        }

        // Entry should be replaced with new term
        let term = node.persistent.read().log[0].term;
        assert_eq!(term, 2);
    }

    #[test]
    fn test_handle_append_entries_with_embedding() {
        let mut config = RaftConfig::default();
        config.enable_fast_path = true;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
        );

        // First, establish node2 as leader and add sufficient embedding history (min_leader_history
        // = 3)
        {
            *node.current_leader.write() = Some("node2".to_string());
            // Need at least 3 embeddings for fast-path to be considered
            node.fast_path_state
                .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
            node.fast_path_state
                .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
            node.fast_path_state
                .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        }

        // Send append entries with similar embedding
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(SparseVector::from_dense(&[1.0, 0.0, 0.0])),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(aer.success);
            assert!(aer.used_fast_path);
        } else {
            panic!("expected AppendEntriesResponse");
        }
    }

    #[test]
    fn test_fast_path_leader_change_clears_history() {
        let mut config = RaftConfig::default();
        config.enable_fast_path = true;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string(), "node3".to_string()],
            transport,
            config,
        );

        // Build up history for node2
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        *node.current_leader.write() = Some("node2".to_string());

        // Now receive from node3 - should trigger leader change cleanup
        let ae = AppendEntries {
            term: 2,
            leader_id: "node3".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(SparseVector::from_dense(&[1.0, 0.0, 0.0])),
        };

        let response = node.handle_message(&"node3".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(aer.success);
            // Should not use fast-path because node3 has no history yet
            assert!(!aer.used_fast_path);
        } else {
            panic!("expected AppendEntriesResponse");
        }

        // Old leader's history should be cleared
        assert_eq!(
            node.fast_path_state
                .leader_history_size(&"node2".to_string()),
            0
        );
    }

    #[test]
    fn test_fast_path_insufficient_history() {
        let mut config = RaftConfig::default();
        config.enable_fast_path = true;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
        );

        // Only 1 embedding in history (need 3 minimum)
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        *node.current_leader.write() = Some("node2".to_string());

        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(SparseVector::from_dense(&[1.0, 0.0, 0.0])),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(aer.success);
            // Insufficient history - should not use fast-path
            assert!(!aer.used_fast_path);
        } else {
            panic!("expected AppendEntriesResponse");
        }
    }

    #[test]
    fn test_fast_path_low_similarity_rejected() {
        let mut config = RaftConfig::default();
        config.enable_fast_path = true;
        config.similarity_threshold = 0.95;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
        );

        *node.current_leader.write() = Some("node2".to_string());
        // Add sufficient history
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);

        // Orthogonal vector - low similarity
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(SparseVector::from_dense(&[0.0, 1.0, 0.0])),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(aer.success);
            // Low similarity - should not use fast-path
            assert!(!aer.used_fast_path);
        } else {
            panic!("expected AppendEntriesResponse");
        }
    }

    #[test]
    fn test_fast_path_stats_recorded() {
        use std::sync::atomic::Ordering;

        let mut config = RaftConfig::default();
        config.enable_fast_path = true;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
        );

        *node.current_leader.write() = Some("node2".to_string());
        // Add sufficient history for fast-path
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state
            .add_dense_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);

        // Send similar embedding - should use fast-path
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(SparseVector::from_dense(&[1.0, 0.0, 0.0])),
        };

        node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae.clone()));

        // Check stats - should have 1 fast-path accepted
        let stats = node.fast_path_stats();
        assert_eq!(stats.fast_path_accepted.load(Ordering::Relaxed), 1);
        assert_eq!(stats.fast_path_rejected.load(Ordering::Relaxed), 0);

        // Send dissimilar embedding - should reject fast-path
        let ae2 = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(SparseVector::from_dense(&[0.0, 1.0, 0.0])),
        };

        node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae2));

        // Check stats - should have 1 rejected
        assert_eq!(stats.fast_path_accepted.load(Ordering::Relaxed), 1);
        assert_eq!(stats.fast_path_rejected.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_handle_ping_message() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        let response = node.handle_message(&"node2".to_string(), &Message::Ping { term: 5 });

        if let Some(Message::Pong { term }) = response {
            assert_eq!(term, 5);
        } else {
            panic!("expected Pong");
        }
    }

    #[test]
    fn test_handle_unknown_message() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Pong is not handled, should return None
        let response = node.handle_message(&"node2".to_string(), &Message::Pong { term: 1 });
        assert!(response.is_none());
    }

    #[test]
    fn test_update_state_embedding() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        node.update_state_embedding_dense(vec![1.0, 2.0, 3.0]);

        let embedding = node.state_embedding.read();
        assert_eq!(embedding.to_dense(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_last_log_info_empty() {
        let node = create_test_node("node1", vec![]);
        let (index, term) = node.last_log_info();
        assert_eq!(index, 0);
        assert_eq!(term, 0);
    }

    #[test]
    fn test_last_log_info_with_entries() {
        let node = create_test_node("node1", vec![]);

        {
            let mut persistent = node.persistent.write();
            persistent
                .log
                .push(LogEntry::new(2, 1, create_test_block(1)));
            persistent
                .log
                .push(LogEntry::new(3, 2, create_test_block(2)));
        }

        let (index, term) = node.last_log_info();
        assert_eq!(index, 2);
        assert_eq!(term, 3);
    }

    #[test]
    fn test_raft_config_default() {
        let config = RaftConfig::default();

        assert_eq!(config.election_timeout, (150, 300));
        assert_eq!(config.heartbeat_interval, 50);
        assert!((config.similarity_threshold - 0.95).abs() < 0.001);
        assert!(config.enable_fast_path);
        assert!(config.quorum_size.is_none());
    }

    #[test]
    fn test_fast_path_state_default() {
        let state = FastPathState::default();
        assert_eq!(state.max_history, 10);
    }

    #[test]
    fn test_fast_path_state_get_embeddings_unknown_leader() {
        let state = FastPathState::new(5);
        let embeddings = state.get_embeddings(&"unknown".to_string());
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_propose_not_leader() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        let block = create_test_block(1);
        let result = node.propose(block);

        assert!(result.is_err());
    }

    #[test]
    fn test_handle_append_entries_prev_log_mismatch() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Add an entry
        {
            let mut persistent = node.persistent.write();
            persistent
                .log
                .push(LogEntry::new(1, 1, create_test_block(1)));
        }

        // Leader claims prev_log at index 1 with term 5 (but we have term 1)
        let ae = AppendEntries {
            term: 2,
            leader_id: "node2".to_string(),
            prev_log_index: 1,
            prev_log_term: 5, // Wrong term
            entries: vec![LogEntry::new(2, 2, create_test_block(2))],
            leader_commit: 0,
            block_embedding: None,
        };

        let response = node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(!aer.success); // Log mismatch
        } else {
            panic!("expected AppendEntriesResponse");
        }
    }

    #[test]
    fn test_handle_append_entries_prev_log_index_too_high() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // No entries in log
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 5, // We don't have this
            prev_log_term: 1,
            entries: vec![],
            leader_commit: 0,
            block_embedding: None,
        };

        let response = node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(!aer.success); // Log too short
        } else {
            panic!("expected AppendEntriesResponse");
        }
    }

    #[test]
    fn test_handle_append_entries_updates_commit_index() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Add entries first
        {
            let mut persistent = node.persistent.write();
            for i in 1..=3 {
                persistent
                    .log
                    .push(LogEntry::new(1, i, create_test_block(i)));
            }
        }

        // Leader says commit index is 2
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 3,
            prev_log_term: 1,
            entries: vec![],
            leader_commit: 2,
            block_embedding: None,
        };

        node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        assert_eq!(node.commit_index(), 2);
    }

    #[test]
    fn test_quorum_size_custom() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = RaftConfig {
            quorum_size: Some(5),
            ..Default::default()
        };
        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string(), "node3".to_string()],
            transport,
            config,
        );

        assert_eq!(node.quorum_size(), 5);
    }

    #[test]
    fn test_try_advance_commit_index_not_leader() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Not a leader, should do nothing
        node.try_advance_commit_index();

        assert_eq!(node.commit_index(), 0);
    }

    #[test]
    fn test_try_advance_commit_index_no_leader_state() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Set to leader but without proper leader state
        *node.state.write() = RaftState::Leader;

        node.try_advance_commit_index();
        assert_eq!(node.commit_index(), 0);
    }

    #[test]
    fn test_handle_request_vote_stale_term() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Set our term higher
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 5;
        }

        // Request with lower term
        let rv = RequestVote {
            term: 3,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
            state_embedding: SparseVector::new(0),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::RequestVote(rv));

        if let Some(Message::RequestVoteResponse(rvr)) = response {
            assert!(!rvr.vote_granted);
            assert_eq!(rvr.term, 5); // Our term
        } else {
            panic!("expected RequestVoteResponse");
        }
    }

    #[test]
    fn test_handle_request_vote_already_voted() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        // Vote for node2 first
        let rv1 = RequestVote {
            term: 1,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
            state_embedding: SparseVector::new(0),
        };

        let response1 = node.handle_message(&"node2".to_string(), &Message::RequestVote(rv1));
        if let Some(Message::RequestVoteResponse(rvr)) = response1 {
            assert!(rvr.vote_granted);
        }

        // node3 requests vote in same term
        let rv2 = RequestVote {
            term: 1,
            candidate_id: "node3".to_string(),
            last_log_index: 0,
            last_log_term: 0,
            state_embedding: SparseVector::new(0),
        };

        let response2 = node.handle_message(&"node3".to_string(), &Message::RequestVote(rv2));
        if let Some(Message::RequestVoteResponse(rvr)) = response2 {
            assert!(!rvr.vote_granted); // Already voted for node2
        } else {
            panic!("expected RequestVoteResponse");
        }
    }

    #[test]
    fn test_handle_request_vote_log_not_up_to_date() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Add entries to our log
        {
            let mut persistent = node.persistent.write();
            persistent
                .log
                .push(LogEntry::new(3, 1, create_test_block(1)));
        }

        // Candidate with older log
        let rv = RequestVote {
            term: 4,
            candidate_id: "node2".to_string(),
            last_log_index: 1,
            last_log_term: 2, // Our log has term 3
            state_embedding: SparseVector::new(0),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::RequestVote(rv));

        if let Some(Message::RequestVoteResponse(rvr)) = response {
            assert!(!rvr.vote_granted); // Their log is not as up-to-date
        } else {
            panic!("expected RequestVoteResponse");
        }
    }

    #[test]
    fn test_handle_request_vote_response_duplicate_vote() {
        let node = create_test_node(
            "node1",
            vec![
                "node2".to_string(),
                "node3".to_string(),
                "node4".to_string(),
            ],
        );

        node.start_election();

        // First vote from node2
        let rvr1 = RequestVoteResponse {
            term: 1,
            vote_granted: true,
            voter_id: "node2".to_string(),
        };
        node.handle_message(
            &"node2".to_string(),
            &Message::RequestVoteResponse(rvr1.clone()),
        );

        // Duplicate vote from node2 (should be ignored)
        node.handle_message(&"node2".to_string(), &Message::RequestVoteResponse(rvr1));

        // Still just 2 votes (self + node2)
        let votes = node.votes_received.read();
        assert_eq!(votes.len(), 2);
    }

    #[test]
    fn test_handle_append_entries_old_term() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Set our term higher
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 5;
        }

        let ae = AppendEntries {
            term: 3, // Old term
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: None,
        };

        let response = node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(!aer.success);
            assert_eq!(aer.term, 5);
        } else {
            panic!("expected AppendEntriesResponse");
        }
    }

    #[test]
    fn test_mark_applied_beyond_commit() {
        let node = create_test_node("node1", vec![]);

        // Set commit index to 2
        {
            let mut volatile = node.volatile.write();
            volatile.commit_index = 2;
        }

        // Try to mark applied beyond commit (should be ignored)
        node.mark_applied(5);

        let last_applied = node.volatile.read().last_applied;
        assert_eq!(last_applied, 0); // Unchanged
    }

    #[test]
    fn test_get_uncommitted_entries_edge_cases() {
        let node = create_test_node("node1", vec![]);

        // Add entries
        {
            let mut persistent = node.persistent.write();
            for i in 1..=5 {
                persistent
                    .log
                    .push(LogEntry::new(1, i, create_test_block(i)));
            }
        }

        // Case 1: commit_index beyond log length
        {
            let mut volatile = node.volatile.write();
            volatile.commit_index = 10; // Beyond log
            volatile.last_applied = 0;
        }

        let entries = node.get_uncommitted_entries();
        assert!(entries.is_empty()); // end > log.len()

        // Case 2: last_applied == commit_index
        {
            let mut volatile = node.volatile.write();
            volatile.commit_index = 3;
            volatile.last_applied = 3;
        }

        let entries = node.get_uncommitted_entries();
        assert!(entries.is_empty()); // start == end
    }

    #[test]
    fn test_handle_append_entries_response_success_advances_commit() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Make leader with entries
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 1;
            for i in 1..=3 {
                persistent
                    .log
                    .push(LogEntry::new(1, i, create_test_block(i)));
            }
        }
        node.become_leader();

        // Successful replication response
        let aer = AppendEntriesResponse {
            term: 1,
            success: true,
            follower_id: "node2".to_string(),
            match_index: 3,
            used_fast_path: false,
        };

        node.handle_message(&"node2".to_string(), &Message::AppendEntriesResponse(aer));

        // With 2 nodes (quorum = 1), commit should advance
        // Leader has 3, node2 has 3 -> commit = 3
        assert_eq!(node.commit_index(), 3);
    }

    #[test]
    fn test_raft_node_accessors() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Test fast_path_state accessor
        let fps = node.fast_path_state();
        assert_eq!(fps.max_history, 10);

        // Test finalized_height
        assert_eq!(node.finalized_height(), 0);

        // Test log_length
        assert_eq!(node.log_length(), 0);
    }

    #[test]
    fn test_handle_append_entries_extends_log() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Add initial entry
        {
            let mut persistent = node.persistent.write();
            persistent
                .log
                .push(LogEntry::new(1, 1, create_test_block(1)));
        }

        // Leader sends more entries
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 1,
            prev_log_term: 1,
            entries: vec![
                LogEntry::new(1, 2, create_test_block(2)),
                LogEntry::new(1, 3, create_test_block(3)),
            ],
            leader_commit: 0,
            block_embedding: None,
        };

        let response = node.handle_message(&"node2".to_string(), &Message::AppendEntries(ae));

        if let Some(Message::AppendEntriesResponse(aer)) = response {
            assert!(aer.success);
            assert_eq!(aer.match_index, 3);
        } else {
            panic!("expected AppendEntriesResponse");
        }

        assert_eq!(node.log_length(), 3);
    }

    #[test]
    fn test_try_advance_commit_index_entry_from_old_term() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Add entries from old term
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 2;
            persistent
                .log
                .push(LogEntry::new(1, 1, create_test_block(1))); // Old term
        }
        node.become_leader();

        // Simulate replication
        {
            let mut ls = node.leader_state.write();
            if let Some(ref mut state) = *ls {
                state.match_index.insert("node2".to_string(), 1);
            }
        }

        node.try_advance_commit_index();

        // Should NOT advance because entry is from old term (term 1, current is 2)
        assert_eq!(node.commit_index(), 0);
    }

    #[test]
    fn test_is_peer_healthy_no_membership() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Without membership manager, all peers are considered healthy
        assert!(node.is_peer_healthy(&"node2".to_string()));
        assert!(node.is_peer_healthy(&"unknown".to_string()));
    }

    #[test]
    fn test_with_membership_constructor() {
        use crate::membership::{ClusterConfig, LocalNodeConfig, MembershipManager};

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9101".parse().unwrap());

        let membership = Arc::new(MembershipManager::new(config, transport.clone()));

        let node = RaftNode::with_membership(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            RaftConfig::default(),
            membership,
        );

        // Membership should be set
        assert!(node.membership.is_some());
    }

    #[test]
    fn test_set_membership() {
        use crate::membership::{ClusterConfig, LocalNodeConfig, MembershipManager};

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport.clone(),
            RaftConfig::default(),
        );

        assert!(node.membership.is_none());

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        );
        let membership = Arc::new(MembershipManager::new(config, transport));
        node.set_membership(membership);

        assert!(node.membership.is_some());
    }

    #[test]
    fn test_request_vote_rejected_for_unhealthy_candidate() {
        use crate::membership::{ClusterConfig, HealthConfig, LocalNodeConfig, MembershipManager};

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));

        // Create membership with node2 that will be marked as failed
        let health_config = HealthConfig {
            startup_grace_ms: 0, // No grace period
            failure_threshold: 1,
            ..Default::default()
        };
        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
        .with_health(health_config);

        let membership = Arc::new(MembershipManager::new(config, transport.clone()));

        // Mark node2 as failed
        membership.mark_failed(&"node2".to_string());

        let node = RaftNode::with_membership(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            RaftConfig::default(),
            membership,
        );

        // Request vote from unhealthy node2
        let rv = RequestVote {
            term: 1,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
            state_embedding: SparseVector::new(0),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::RequestVote(rv));

        if let Some(Message::RequestVoteResponse(rvr)) = response {
            // Vote should be denied because candidate is unhealthy
            assert!(!rvr.vote_granted);
        } else {
            panic!("expected RequestVoteResponse");
        }
    }

    #[test]
    fn test_request_vote_granted_for_healthy_candidate() {
        use crate::membership::{ClusterConfig, HealthConfig, LocalNodeConfig, MembershipManager};

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));

        let health_config = HealthConfig {
            startup_grace_ms: 0,
            failure_threshold: 3,
            ..Default::default()
        };
        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
        .with_health(health_config);

        let membership = Arc::new(MembershipManager::new(config, transport.clone()));

        // Mark node2 as healthy
        membership.mark_healthy(&"node2".to_string());

        let node = RaftNode::with_membership(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            RaftConfig::default(),
            membership,
        );

        // Request vote from healthy node2
        let rv = RequestVote {
            term: 1,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
            state_embedding: SparseVector::new(0),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::RequestVote(rv));

        if let Some(Message::RequestVoteResponse(rvr)) = response {
            // Vote should be granted because candidate is healthy
            assert!(rvr.vote_granted);
        } else {
            panic!("expected RequestVoteResponse");
        }
    }

    #[test]
    fn test_geometric_vote_bias_enabled() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = RaftConfig {
            enable_geometric_tiebreak: true,
            geometric_tiebreak_threshold: 0.3,
            ..Default::default()
        };

        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
        );

        // Set local embedding
        node.update_state_embedding(SparseVector::from_dense(&[1.0, 0.0, 0.0]));

        // Similar embedding should have high bias
        let similar = SparseVector::from_dense(&[0.9, 0.1, 0.0]);
        let bias = node.geometric_vote_bias(&similar);
        assert!(bias > 0.5, "Similar embeddings should have high bias");

        // Orthogonal embedding should have neutral bias
        let orthogonal = SparseVector::from_dense(&[0.0, 1.0, 0.0]);
        let orthogonal_bias = node.geometric_vote_bias(&orthogonal);
        assert!(
            (orthogonal_bias - 0.5).abs() < 0.1,
            "Orthogonal embeddings should have neutral bias"
        );
    }

    #[test]
    fn test_geometric_vote_bias_disabled() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = RaftConfig {
            enable_geometric_tiebreak: false,
            ..Default::default()
        };

        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
        );

        // Should return neutral bias when disabled
        let embedding = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let bias = node.geometric_vote_bias(&embedding);
        assert!(
            (bias - 0.5).abs() < 0.001,
            "Should return 0.5 when disabled"
        );
    }

    #[test]
    fn test_geometric_vote_bias_empty_embeddings() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = RaftConfig {
            enable_geometric_tiebreak: true,
            ..Default::default()
        };

        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
        );

        // Local embedding is empty by default
        let candidate = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let bias = node.geometric_vote_bias(&candidate);
        assert!(
            (bias - 0.5).abs() < 0.001,
            "Should return 0.5 for empty local embedding"
        );

        // Set local embedding but use empty candidate
        node.update_state_embedding(SparseVector::from_dense(&[1.0, 0.0, 0.0]));
        let empty_candidate = SparseVector::new(0);
        let bias2 = node.geometric_vote_bias(&empty_candidate);
        assert!(
            (bias2 - 0.5).abs() < 0.001,
            "Should return 0.5 for empty candidate embedding"
        );
    }

    #[test]
    fn test_geometric_tiebreak_in_request_vote() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = RaftConfig {
            enable_geometric_tiebreak: true,
            geometric_tiebreak_threshold: 0.3,
            ..Default::default()
        };

        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
        );

        // Set local embedding
        node.update_state_embedding(SparseVector::from_dense(&[1.0, 0.0, 0.0]));

        // Request vote with similar embedding (should pass geometric tiebreak)
        let rv = RequestVote {
            term: 1,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
            state_embedding: SparseVector::from_dense(&[0.9, 0.1, 0.0]),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::RequestVote(rv));

        if let Some(Message::RequestVoteResponse(rvr)) = response {
            // Vote should be granted because logs are equal and geometric bias is high
            assert!(rvr.vote_granted);
        } else {
            panic!("expected RequestVoteResponse");
        }
    }

    #[test]
    fn test_raft_config_geometric_defaults() {
        let config = RaftConfig::default();
        assert!(config.enable_geometric_tiebreak);
        assert!((config.geometric_tiebreak_threshold - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_fast_path_state_get_sparse_embeddings() {
        let state = FastPathState::new(5);
        let leader = "leader1".to_string();

        // Initially empty
        let sparse = state.get_sparse_embeddings(&leader);
        assert!(sparse.is_empty());

        // Add embedding and retrieve
        state.add_embedding(&leader, SparseVector::from_dense(&[1.0, 2.0, 3.0]));
        let sparse = state.get_sparse_embeddings(&leader);
        assert_eq!(sparse.len(), 1);
        assert_eq!(sparse[0].dimension(), 3);
    }

    #[test]
    fn test_raft_node_transport_accessor() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string(), "node3".to_string()],
            transport.clone(),
            RaftConfig::default(),
        );

        let t = node.transport();
        assert_eq!(t.local_id(), "node1");
    }

    #[test]
    fn test_raft_node_last_log_index() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            RaftConfig::default(),
        );

        // Initially empty
        assert_eq!(node.last_log_index(), 0);

        // Append an entry
        {
            let mut persistent = node.persistent.write();
            persistent
                .log
                .push(LogEntry::new(1, 1, Block::genesis("node1".to_string())));
        }

        assert_eq!(node.last_log_index(), 1);
    }

    #[test]
    fn test_raft_node_get_entries_for_follower() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Add some entries and become leader
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 1;
            persistent
                .log
                .push(LogEntry::new(1, 1, Block::genesis("node1".to_string())));
            persistent
                .log
                .push(LogEntry::new(1, 2, Block::genesis("node1".to_string())));
        }
        node.become_leader();

        // After becoming leader, next_index = log.len() + 1, so no entries to send
        let (prev_idx, prev_term, entries, _embedding) =
            node.get_entries_for_follower(&"node2".to_string());
        // prev_log is the last entry (index 2, term 1)
        assert_eq!(prev_idx, 2);
        assert_eq!(prev_term, 1);
        // No new entries since next_index is already at end
        assert!(entries.is_empty());

        // Manually set next_index to 1 to get all entries
        {
            let mut ls = node.leader_state.write();
            if let Some(ref mut leader_state) = *ls {
                leader_state.next_index.insert("node2".to_string(), 1);
            }
        }

        let (prev_idx, prev_term, entries, _embedding) =
            node.get_entries_for_follower(&"node2".to_string());
        assert_eq!(prev_idx, 0);
        assert_eq!(prev_term, 0);
        assert_eq!(entries.len(), 2);
    }

    // ========== Persistence Tests ==========

    #[test]
    fn test_save_load_from_store() {
        let store = tensor_store::TensorStore::new();
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Set up some state
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 5;
            persistent.voted_for = Some("node2".to_string());
            persistent
                .log
                .push(LogEntry::new(1, 1, Block::genesis("node1".to_string())));
        }

        // Save to store
        node.save_to_store(&store).unwrap();

        // Load from store
        let (term, voted_for, log) = RaftNode::load_from_store("node1", &store).unwrap();

        assert_eq!(term, 5);
        assert_eq!(voted_for, Some("node2".to_string()));
        assert_eq!(log.len(), 1);
        assert_eq!(log[0].term, 1);
        assert_eq!(log[0].index, 1);
    }

    #[test]
    fn test_load_from_empty_store() {
        let store = tensor_store::TensorStore::new();

        // No state saved - should return None
        let result = RaftNode::load_from_store("node1", &store);
        assert!(result.is_none());
    }

    #[test]
    fn test_with_store_loads_state() {
        let store = tensor_store::TensorStore::new();

        // First node saves state
        let node1 = create_test_node("node1", vec!["node2".to_string()]);
        {
            let mut persistent = node1.persistent.write();
            persistent.current_term = 10;
            persistent.voted_for = Some("node3".to_string());
            persistent
                .log
                .push(LogEntry::new(5, 1, Block::genesis("node1".to_string())));
            persistent
                .log
                .push(LogEntry::new(8, 2, Block::genesis("node1".to_string())));
        }
        node1.save_to_store(&store).unwrap();

        // Second node loads state via with_store
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node2 = RaftNode::with_store(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            RaftConfig::default(),
            &store,
        );

        // Verify state was loaded
        assert_eq!(node2.current_term(), 10);
        assert_eq!(node2.log_length(), 2);
    }

    #[test]
    fn test_with_store_empty_store() {
        let store = tensor_store::TensorStore::new();
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));

        // Create node from empty store
        let node = RaftNode::with_store(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            RaftConfig::default(),
            &store,
        );

        // Should have default state
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.log_length(), 0);
    }

    #[test]
    fn test_with_state() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let log = vec![LogEntry::new(3, 1, Block::genesis("node1".to_string()))];

        let node = RaftNode::with_state(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            RaftConfig::default(),
            7,
            Some("node3".to_string()),
            log,
        );

        assert_eq!(node.current_term(), 7);
        assert_eq!(node.log_length(), 1);
    }

    #[test]
    fn test_persistence_preserves_embedding() {
        let store = tensor_store::TensorStore::new();
        let node = create_test_node("node1", vec![]);

        // Set state embedding
        node.update_state_embedding_dense(vec![1.0, 0.5, 0.25, 0.125]);

        // Set up term so there's state to save
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 1;
        }

        // Save to store
        node.save_to_store(&store).unwrap();

        // Verify embedding was stored
        let data = store.get("_raft:state:node1").unwrap();
        assert!(data.get("_embedding").is_some());
    }

    #[test]
    fn test_save_without_voted_for() {
        let store = tensor_store::TensorStore::new();
        let node = create_test_node("node1", vec![]);

        // Set term but not voted_for
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 3;
            persistent.voted_for = None;
        }

        node.save_to_store(&store).unwrap();

        // Load and verify voted_for is None
        let (term, voted_for, _log) = RaftNode::load_from_store("node1", &store).unwrap();
        assert_eq!(term, 3);
        assert!(voted_for.is_none());
    }

    #[test]
    fn test_persistence_key() {
        let key = RaftNode::persistence_key("my_node");
        assert_eq!(key, "_raft:state:my_node");
    }

    // ========== Snapshot / Log Compaction Tests ==========

    #[test]
    fn test_snapshot_metadata_new() {
        let meta = SnapshotMetadata::new(
            100,
            5,
            [1u8; 32],
            vec!["node1".to_string(), "node2".to_string()],
            1024,
        );

        assert_eq!(meta.last_included_index, 100);
        assert_eq!(meta.last_included_term, 5);
        assert_eq!(meta.snapshot_hash, [1u8; 32]);
        assert_eq!(meta.config.len(), 2);
        assert_eq!(meta.size, 1024);
        assert!(meta.created_at > 0);
    }

    #[test]
    fn test_snapshot_metadata_serialization() {
        let meta = SnapshotMetadata::new(50, 3, [2u8; 32], vec!["n1".to_string()], 512);

        let bytes = bincode::serialize(&meta).unwrap();
        let decoded: SnapshotMetadata = bincode::deserialize(&bytes).unwrap();

        assert_eq!(decoded.last_included_index, 50);
        assert_eq!(decoded.last_included_term, 3);
        assert_eq!(decoded.snapshot_hash, [2u8; 32]);
        assert_eq!(decoded.size, 512);
    }

    #[test]
    fn test_should_compact_threshold_not_met() {
        let config = RaftConfig {
            snapshot_threshold: 100,
            ..Default::default()
        };
        let node = RaftNode::new(
            "node1".to_string(),
            vec![],
            Arc::new(MemoryTransport::new("test".to_string())),
            config,
        );

        // Add fewer entries than threshold
        {
            let mut persistent = node.persistent.write();
            for i in 1..50 {
                persistent.log.push(create_test_log_entry(i));
            }
        }

        assert!(!node.should_compact());
    }

    #[test]
    fn test_should_compact_no_finalized() {
        let config = RaftConfig {
            snapshot_threshold: 10,
            ..Default::default()
        };
        let node = RaftNode::new(
            "node1".to_string(),
            vec![],
            Arc::new(MemoryTransport::new("test".to_string())),
            config,
        );

        // Add entries but don't finalize any
        {
            let mut persistent = node.persistent.write();
            for i in 1..=20 {
                persistent.log.push(create_test_log_entry(i));
            }
        }

        // No finalized entries, shouldn't compact
        assert!(!node.should_compact());
    }

    #[test]
    fn test_should_compact_ready() {
        let config = RaftConfig {
            snapshot_threshold: 10,
            ..Default::default()
        };
        let node = RaftNode::new(
            "node1".to_string(),
            vec![],
            Arc::new(MemoryTransport::new("test".to_string())),
            config,
        );

        // Add entries and finalize some
        {
            let mut persistent = node.persistent.write();
            for i in 1..=20 {
                persistent.log.push(create_test_log_entry(i));
            }
        }
        node.finalized_height.store(15, Ordering::SeqCst);

        assert!(node.should_compact());
    }

    #[test]
    fn test_create_snapshot_empty_chain() {
        let node = create_test_node("node1", vec![]);

        let result = node.create_snapshot();
        assert!(result.is_err());
        if let Err(ChainError::SnapshotError(msg)) = result {
            assert!(msg.contains("no finalized entries"));
        }
    }

    #[test]
    fn test_create_snapshot_basic() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Add entries
        {
            let mut persistent = node.persistent.write();
            for i in 1..=5 {
                persistent.log.push(create_test_log_entry(i));
            }
        }

        // Finalize and commit
        {
            let mut volatile = node.volatile.write();
            volatile.commit_index = 5;
        }
        node.finalized_height.store(3, Ordering::SeqCst);

        let result = node.create_snapshot();
        assert!(result.is_ok());

        let (meta, data) = result.unwrap();
        assert_eq!(meta.last_included_index, 3);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_truncate_log() {
        let config = RaftConfig {
            snapshot_trailing_logs: 2,
            ..Default::default()
        };
        let node = RaftNode::new(
            "node1".to_string(),
            vec![],
            Arc::new(MemoryTransport::new("test".to_string())),
            config,
        );

        // Add 10 entries
        {
            let mut persistent = node.persistent.write();
            for i in 1..=10 {
                persistent.log.push(create_test_log_entry(i));
            }
        }

        let meta = SnapshotMetadata::new(7, 1, [0u8; 32], vec![], 0);

        node.truncate_log(&meta).unwrap();

        // Should keep entries from index 5 onwards (7 - 2 = 5)
        let persistent = node.persistent.read();
        assert!(persistent.log.len() < 10);

        // Verify snapshot state was updated
        let snapshot_state = node.snapshot_state.read();
        assert!(snapshot_state.last_snapshot.is_some());
        assert_eq!(
            snapshot_state
                .last_snapshot
                .as_ref()
                .unwrap()
                .last_included_index,
            7
        );
    }

    #[test]
    fn test_get_snapshot_metadata() {
        let node = create_test_node("node1", vec![]);

        // Initially no snapshot
        assert!(node.get_snapshot_metadata().is_none());

        // Set snapshot metadata
        {
            let mut snapshot_state = node.snapshot_state.write();
            snapshot_state.last_snapshot = Some(SnapshotMetadata::new(
                10,
                2,
                [5u8; 32],
                vec!["peer1".to_string()],
                100,
            ));
        }

        let meta = node.get_snapshot_metadata();
        assert!(meta.is_some());
        assert_eq!(meta.unwrap().last_included_index, 10);
    }

    #[test]
    fn test_install_snapshot_basic() {
        let node = create_test_node("follower", vec!["leader".to_string()]);

        // Create snapshot data with log entries
        let entries: Vec<LogEntry> = (1..=5).map(|i| create_test_log_entry(i)).collect();
        let data = bincode::serialize(&entries).unwrap();

        // Compute proper hash of the data
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let snapshot_hash: [u8; 32] = hasher.finalize().into();

        let meta = SnapshotMetadata::new(
            5,
            1,
            snapshot_hash,
            vec!["peer".to_string()],
            data.len() as u64,
        );

        node.install_snapshot(meta.clone(), &data).unwrap();

        // Verify state was updated
        assert_eq!(node.finalized_height(), 5);
        assert_eq!(node.volatile.read().commit_index, 5);
        assert!(node.get_snapshot_metadata().is_some());

        let persistent = node.persistent.read();
        assert_eq!(persistent.log.len(), 5);
    }

    #[test]
    fn test_install_snapshot_empty_data() {
        let node = create_test_node("follower", vec![]);

        let entries: Vec<LogEntry> = vec![];
        let data = bincode::serialize(&entries).unwrap();

        // Compute proper hash of empty data
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let snapshot_hash: [u8; 32] = hasher.finalize().into();

        let meta = SnapshotMetadata::new(5, 1, snapshot_hash, vec![], data.len() as u64);

        let result = node.install_snapshot(meta, &data);
        assert!(result.is_err());
        if let Err(ChainError::SnapshotError(msg)) = result {
            assert!(msg.contains("no entries"));
        }
    }

    #[test]
    fn test_install_snapshot_index_mismatch() {
        let node = create_test_node("follower", vec![]);

        let entries: Vec<LogEntry> = (1..=3).map(|i| create_test_log_entry(i)).collect();
        let data = bincode::serialize(&entries).unwrap();

        // Compute proper hash of the data
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let snapshot_hash: [u8; 32] = hasher.finalize().into();

        // Metadata says index 5, but entries only go to 3
        let meta = SnapshotMetadata::new(5, 1, snapshot_hash, vec![], data.len() as u64);

        let result = node.install_snapshot(meta, &data);
        assert!(result.is_err());
        if let Err(ChainError::SnapshotError(msg)) = result {
            assert!(msg.contains("index mismatch"));
        }
    }

    #[test]
    fn test_receive_snapshot_chunks() {
        let node = create_test_node("follower", vec![]);

        // Simulate receiving chunks
        let data = vec![1u8; 100];
        let chunk_size = 30;

        // First chunk
        let result = node.receive_snapshot_chunk(0, &data[0..30], 100, false);
        assert!(result.is_ok());
        assert!(!result.unwrap());

        // Second chunk
        let result = node.receive_snapshot_chunk(30, &data[30..60], 100, false);
        assert!(result.is_ok());
        assert!(!result.unwrap());

        // Third chunk
        let result = node.receive_snapshot_chunk(60, &data[60..90], 100, false);
        assert!(result.is_ok());
        assert!(!result.unwrap());

        // Final chunk
        let result = node.receive_snapshot_chunk(90, &data[90..100], 100, true);
        assert!(result.is_ok());
        assert!(result.unwrap()); // Complete!

        let received = node.take_pending_snapshot_data();
        assert_eq!(received.len(), 100);
    }

    #[test]
    fn test_receive_snapshot_chunk_offset_mismatch() {
        let node = create_test_node("follower", vec![]);

        // Start receiving
        let _ = node.receive_snapshot_chunk(0, &[1, 2, 3], 10, false);

        // Send wrong offset (should be 3)
        let result = node.receive_snapshot_chunk(5, &[4, 5, 6], 10, false);
        assert!(result.is_err());
        if let Err(ChainError::SnapshotError(msg)) = result {
            assert!(msg.contains("offset mismatch"));
        }
    }

    #[test]
    fn test_get_snapshot_chunks() {
        let config = RaftConfig {
            snapshot_chunk_size: 10,
            ..Default::default()
        };
        let node = RaftNode::new(
            "node1".to_string(),
            vec![],
            Arc::new(MemoryTransport::new("test".to_string())),
            config,
        );

        let data = vec![0u8; 25];
        let chunks = node.get_snapshot_chunks(&data);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].0, 0); // offset
        assert_eq!(chunks[0].1.len(), 10); // chunk size
        assert!(!chunks[0].2); // not last

        assert_eq!(chunks[1].0, 10);
        assert_eq!(chunks[1].1.len(), 10);
        assert!(!chunks[1].2);

        assert_eq!(chunks[2].0, 20);
        assert_eq!(chunks[2].1.len(), 5);
        assert!(chunks[2].2); // last chunk
    }

    #[test]
    fn test_needs_snapshot_for_follower() {
        let node = create_test_node("leader", vec!["follower".to_string()]);

        // Make this node a leader
        *node.state.write() = RaftState::Leader;
        node.become_leader();

        // Without snapshot, should not need snapshot transfer
        assert!(!node.needs_snapshot_for_follower(&"follower".to_string()));

        // Add a snapshot and truncate log
        {
            let mut snapshot_state = node.snapshot_state.write();
            snapshot_state.last_snapshot =
                Some(SnapshotMetadata::new(100, 5, [0u8; 32], vec![], 0));
        }

        // Clear the log (simulating truncation)
        node.persistent.write().log.clear();

        // Add some new entries starting from index 101
        {
            let mut persistent = node.persistent.write();
            let mut entry = create_test_log_entry(101);
            entry.index = 101;
            persistent.log.push(entry);
        }

        // Follower needs index 1, but our log starts at 101
        assert!(node.needs_snapshot_for_follower(&"follower".to_string()));
    }

    #[test]
    fn test_snapshot_config_defaults() {
        let config = RaftConfig::default();

        assert_eq!(config.snapshot_threshold, 10_000);
        assert_eq!(config.snapshot_trailing_logs, 100);
        assert_eq!(config.snapshot_chunk_size, 1024 * 1024);
    }

    #[test]
    fn test_handle_snapshot_request_not_leader() {
        let node = create_test_node("follower", vec![]);

        let request = SnapshotRequest {
            requester_id: "other".to_string(),
            offset: 0,
            chunk_size: 1024,
        };

        // Followers shouldn't respond to snapshot requests
        let response = node.handle_snapshot_request(&"other".to_string(), &request);
        assert!(response.is_none());
    }

    #[test]
    fn test_handle_snapshot_response_not_follower() {
        let node = create_test_node("leader", vec![]);
        *node.state.write() = RaftState::Leader;

        let response = SnapshotResponse {
            snapshot_height: 10,
            snapshot_hash: [0u8; 32],
            data: vec![1, 2, 3],
            offset: 0,
            total_size: 3,
            is_last: true,
        };

        // Leaders shouldn't process snapshot responses
        node.handle_snapshot_response(&"other".to_string(), &response);
        // Should not have installed any snapshot
        assert!(node.get_snapshot_metadata().is_none());
    }

    // ========== QuorumTracker Tests ==========

    #[test]
    fn test_quorum_tracker_new() {
        let tracker = QuorumTracker::new(std::time::Duration::from_secs(5), 3);
        assert_eq!(tracker.reachable_count(), 0);
    }

    #[test]
    fn test_quorum_tracker_default() {
        let tracker = QuorumTracker::default();
        assert_eq!(tracker.reachable_count(), 0);
    }

    #[test]
    fn test_quorum_tracker_record_success() {
        let tracker = QuorumTracker::default();
        tracker.record_success(&"node2".to_string());

        assert!(tracker.is_reachable(&"node2".to_string()));
        assert_eq!(tracker.reachable_count(), 1);
    }

    #[test]
    fn test_quorum_tracker_record_failure() {
        let tracker = QuorumTracker::new(std::time::Duration::from_secs(5), 3);

        // First mark as reachable
        tracker.record_success(&"node2".to_string());
        assert!(tracker.is_reachable(&"node2".to_string()));

        // Record failures up to threshold
        tracker.record_failure(&"node2".to_string());
        tracker.record_failure(&"node2".to_string());
        assert!(tracker.is_reachable(&"node2".to_string())); // Still reachable

        tracker.record_failure(&"node2".to_string()); // 3rd failure
        assert!(!tracker.is_reachable(&"node2".to_string())); // Now unreachable
    }

    #[test]
    fn test_quorum_tracker_success_clears_failures() {
        let tracker = QuorumTracker::new(std::time::Duration::from_secs(5), 3);

        tracker.record_success(&"node2".to_string());
        tracker.record_failure(&"node2".to_string());
        tracker.record_failure(&"node2".to_string());

        // Success should clear failure count
        tracker.record_success(&"node2".to_string());

        // Now we can record failures again without being unreachable
        tracker.record_failure(&"node2".to_string());
        tracker.record_failure(&"node2".to_string());
        assert!(tracker.is_reachable(&"node2".to_string()));
    }

    #[test]
    fn test_quorum_tracker_has_quorum_3_nodes() {
        let tracker = QuorumTracker::default();

        // 3 nodes total, need 2 for quorum (including self)
        // Just self = 1, need 1 more peer
        assert!(!tracker.has_quorum(2)); // 2 peers + self = 3 nodes, 0 peers reachable

        tracker.record_success(&"node2".to_string());
        assert!(tracker.has_quorum(2)); // 1 peer + self = 2, quorum achieved
    }

    #[test]
    fn test_quorum_tracker_has_quorum_5_nodes() {
        let tracker = QuorumTracker::default();

        // 5 nodes total, need 3 for quorum
        // Self + 2 peers = 3 needed
        assert!(!tracker.has_quorum(4)); // 0 peers reachable

        tracker.record_success(&"node2".to_string());
        assert!(!tracker.has_quorum(4)); // 1 peer + self = 2, need 3

        tracker.record_success(&"node3".to_string());
        assert!(tracker.has_quorum(4)); // 2 peers + self = 3, quorum achieved
    }

    #[test]
    fn test_quorum_tracker_reset() {
        let tracker = QuorumTracker::default();
        tracker.record_success(&"node2".to_string());
        tracker.record_success(&"node3".to_string());
        assert_eq!(tracker.reachable_count(), 2);

        tracker.reset();
        assert_eq!(tracker.reachable_count(), 0);
    }

    #[test]
    fn test_quorum_tracker_unreachable_peers() {
        let tracker = QuorumTracker::new(std::time::Duration::from_secs(5), 2);

        tracker.record_success(&"node2".to_string());
        tracker.record_success(&"node3".to_string());

        // Make node3 unreachable
        tracker.record_failure(&"node3".to_string());
        tracker.record_failure(&"node3".to_string());

        let unreachable = tracker.unreachable_peers();
        assert_eq!(unreachable.len(), 1);
        assert!(unreachable.contains(&"node3".to_string()));
    }

    #[test]
    fn test_quorum_tracker_mark_reachable() {
        let tracker = QuorumTracker::default();

        tracker.mark_reachable(&"node2".to_string());
        assert!(tracker.is_reachable(&"node2".to_string()));
    }

    #[test]
    fn test_quorum_tracker_debug() {
        let tracker = QuorumTracker::default();
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("QuorumTracker"));
    }

    // ========== RaftStats Tests ==========

    #[test]
    fn test_raft_stats_new() {
        let stats = RaftStats::new();
        assert_eq!(stats.fast_path_accepted.load(Ordering::Relaxed), 0);
        assert_eq!(stats.heartbeat_successes.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_raft_stats_record_fast_path() {
        let stats = RaftStats::new();
        stats.record_fast_path();
        stats.record_fast_path();
        assert_eq!(stats.fast_path_accepted.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_raft_stats_record_rejected() {
        let stats = RaftStats::new();
        stats.record_rejected();
        assert_eq!(stats.fast_path_rejected.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_raft_stats_acceptance_rate() {
        let stats = RaftStats::new();

        // No entries yet
        assert_eq!(stats.acceptance_rate(), 0.0);

        // 3 accepted, 1 rejected = 75%
        stats.record_fast_path();
        stats.record_fast_path();
        stats.record_fast_path();
        stats.record_rejected();

        assert!((stats.acceptance_rate() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_raft_stats_heartbeat_success_rate() {
        let stats = RaftStats::new();

        // No heartbeats yet = 100% (assume success)
        assert_eq!(stats.heartbeat_success_rate(), 1.0);

        // 9 successes, 1 failure = 90%
        for _ in 0..9 {
            stats.heartbeat_successes.fetch_add(1, Ordering::Relaxed);
        }
        stats.heartbeat_failures.fetch_add(1, Ordering::Relaxed);

        assert!((stats.heartbeat_success_rate() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_raft_stats_snapshot() {
        let stats = RaftStats::new();
        stats.record_fast_path();
        stats.record_fast_path();
        stats.record_rejected();
        stats.quorum_checks.fetch_add(5, Ordering::Relaxed);
        stats.heartbeat_successes.fetch_add(10, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.fast_path_accepted, 2);
        assert_eq!(snapshot.fast_path_rejected, 1);
        assert_eq!(snapshot.quorum_checks, 5);
        assert_eq!(snapshot.heartbeat_successes, 10);
        assert!((snapshot.fast_path_rate - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_raft_stats_timing() {
        let stats = RaftStats::new();
        stats.election_timing.record(100);
        stats.election_timing.record(200);
        stats.heartbeat_timing.record(50);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.election_timing.count, 2);
        assert_eq!(snapshot.election_timing.avg_us, 150.0);
        assert_eq!(snapshot.heartbeat_timing.count, 1);
    }

    #[test]
    fn test_raft_node_has_stats() {
        let node = create_test_node("node1", vec!["node2".to_string()]);
        assert_eq!(node.stats.fast_path_accepted.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_fast_path_stats_alias() {
        // Verify FastPathStats is an alias for RaftStats
        let stats: FastPathStats = RaftStats::new();
        stats.record_fast_path();
        assert_eq!(stats.fast_path_accepted.load(Ordering::Relaxed), 1);
    }

    // ==================== Pre-Vote Tests ====================

    #[test]
    fn test_pre_vote_basic() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        // Start pre-vote
        node.start_pre_vote();

        // Should be in pre-vote state
        assert!(*node.in_pre_vote.read());

        // Should have voted for self
        let votes = node.pre_votes_received.read();
        assert_eq!(votes.len(), 1);
        assert!(votes.contains(&"node1".to_string()));
    }

    #[test]
    fn test_handle_pre_vote_grants_when_eligible() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Advance time past election timeout by setting last_heartbeat in the past
        *node.last_heartbeat.write() = Instant::now() - std::time::Duration::from_secs(10);

        let pv = PreVote {
            term: 0, // Same term
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
            state_embedding: SparseVector::new(0),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::PreVote(pv));

        if let Some(Message::PreVoteResponse(pvr)) = response {
            assert!(
                pvr.vote_granted,
                "should grant pre-vote when timeout elapsed"
            );
            assert_eq!(pvr.term, 0);
        } else {
            panic!("expected PreVoteResponse");
        }
    }

    #[test]
    fn test_handle_pre_vote_denies_when_leader_active() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Recent heartbeat means leader is active
        *node.last_heartbeat.write() = Instant::now();

        let pv = PreVote {
            term: 0,
            candidate_id: "node2".to_string(),
            last_log_index: 0,
            last_log_term: 0,
            state_embedding: SparseVector::new(0),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::PreVote(pv));

        if let Some(Message::PreVoteResponse(pvr)) = response {
            assert!(
                !pvr.vote_granted,
                "should deny pre-vote when leader is active"
            );
        } else {
            panic!("expected PreVoteResponse");
        }
    }

    #[test]
    fn test_handle_pre_vote_denies_stale_log() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Add an entry to our log
        {
            let mut persistent = node.persistent.write();
            persistent.log.push(create_test_log_entry(1));
            persistent.log[0].term = 5; // Higher term
        }

        // Advance time past election timeout
        *node.last_heartbeat.write() = Instant::now() - std::time::Duration::from_secs(10);

        let pv = PreVote {
            term: 0,
            candidate_id: "node2".to_string(),
            last_log_index: 0, // Candidate has no log
            last_log_term: 0,
            state_embedding: SparseVector::new(0),
        };

        let response = node.handle_message(&"node2".to_string(), &Message::PreVote(pv));

        if let Some(Message::PreVoteResponse(pvr)) = response {
            assert!(
                !pvr.vote_granted,
                "should deny pre-vote when candidate log is stale"
            );
        } else {
            panic!("expected PreVoteResponse");
        }
    }

    #[test]
    fn test_pre_vote_quorum_triggers_election() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        // Start pre-vote
        node.start_pre_vote();
        assert!(*node.in_pre_vote.read());

        // Receive pre-vote response from node2
        let pvr1 = PreVoteResponse {
            term: 0,
            vote_granted: true,
            voter_id: "node2".to_string(),
        };
        node.handle_message(&"node2".to_string(), &Message::PreVoteResponse(pvr1));

        // Should have reached quorum (2 out of 3) and started real election
        assert!(!*node.in_pre_vote.read(), "should exit pre-vote phase");
        // After start_election, we should be candidate with incremented term
        assert_eq!(node.state(), RaftState::Candidate);
        assert_eq!(node.current_term(), 1);
    }

    #[test]
    fn test_pre_vote_ignores_late_responses() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Not in pre-vote
        assert!(!*node.in_pre_vote.read());

        // Late response should be ignored
        let pvr = PreVoteResponse {
            term: 0,
            vote_granted: true,
            voter_id: "node2".to_string(),
        };
        node.handle_message(&"node2".to_string(), &Message::PreVoteResponse(pvr));

        // State should not change
        assert_eq!(node.state(), RaftState::Follower);
    }

    #[test]
    fn test_pre_vote_resets_on_higher_term() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        // Start pre-vote
        node.start_pre_vote();
        assert!(*node.in_pre_vote.read());

        // Receive pre-vote response with higher term
        let pvr = PreVoteResponse {
            term: 5, // Higher term
            vote_granted: false,
            voter_id: "node2".to_string(),
        };
        node.handle_message(&"node2".to_string(), &Message::PreVoteResponse(pvr));

        // Should reset pre-vote and update term
        assert!(
            !*node.in_pre_vote.read(),
            "should exit pre-vote on higher term"
        );
        assert_eq!(node.current_term(), 5);
        assert_eq!(node.state(), RaftState::Follower);
    }

    #[test]
    fn test_pre_vote_config_disabled() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut config = RaftConfig::default();
        config.enable_pre_vote = false;

        let node = RaftNode::new(
            "node1".to_string(),
            vec!["node2".to_string()],
            transport,
            config,
        );

        // Pre-vote disabled means config flag is false
        assert!(!node.config.enable_pre_vote);
    }

    #[test]
    fn test_pre_vote_serialization_roundtrip() {
        let pv = PreVote {
            term: 42,
            candidate_id: "candidate1".to_string(),
            last_log_index: 100,
            last_log_term: 5,
            state_embedding: SparseVector::from_dense(&[0.1, 0.2, 0.3]),
        };

        let bytes = bincode::serialize(&pv).expect("serialize");
        let restored: PreVote = bincode::deserialize(&bytes).expect("deserialize");

        assert_eq!(restored.term, 42);
        assert_eq!(restored.candidate_id, "candidate1");
        assert_eq!(restored.last_log_index, 100);
        assert_eq!(restored.last_log_term, 5);
    }

    #[test]
    fn test_pre_vote_response_serialization() {
        let pvr = PreVoteResponse {
            term: 10,
            vote_granted: true,
            voter_id: "voter1".to_string(),
        };

        let bytes = bincode::serialize(&pvr).expect("serialize");
        let restored: PreVoteResponse = bincode::deserialize(&bytes).expect("deserialize");

        assert_eq!(restored.term, 10);
        assert!(restored.vote_granted);
        assert_eq!(restored.voter_id, "voter1");
    }

    // ==================== Leadership Transfer Tests ====================

    #[test]
    fn test_transfer_leadership_not_leader() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Not leader, should fail
        let result = node.transfer_leadership(&"node2".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not leader"));
    }

    #[test]
    fn test_transfer_leadership_unknown_target() {
        let node = create_test_node("node1", vec!["node2".to_string()]);
        node.become_leader();

        // Unknown target
        let result = node.transfer_leadership(&"unknown".to_string());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unknown transfer target"));
    }

    #[test]
    fn test_transfer_leadership_already_in_progress() {
        let node = create_test_node("node1", vec!["node2".to_string()]);
        node.become_leader();

        // First transfer
        node.transfer_leadership(&"node2".to_string()).unwrap();
        assert!(node.is_transfer_in_progress());

        // Second transfer should fail
        let result = node.transfer_leadership(&"node2".to_string());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already in progress"));
    }

    #[test]
    fn test_transfer_leadership_blocks_proposals() {
        let node = create_test_node("node1", vec!["node2".to_string()]);
        node.become_leader();

        // Start transfer
        node.transfer_leadership(&"node2".to_string()).unwrap();

        // Proposals should be blocked
        let block = create_test_block(1);
        let result = node.propose(block);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("transfer in progress"));
    }

    #[test]
    fn test_handle_timeout_now_starts_election() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Set node2 as current leader
        *node.current_leader.write() = Some("node2".to_string());
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 1;
        }

        let tn = TimeoutNow {
            term: 1,
            leader_id: "node2".to_string(),
        };

        node.handle_message(&"node2".to_string(), &Message::TimeoutNow(tn));

        // Should have started election
        assert_eq!(node.state(), RaftState::Candidate);
        assert_eq!(node.current_term(), 2); // Term incremented
    }

    #[test]
    fn test_handle_timeout_now_rejects_wrong_leader() {
        let node = create_test_node("node1", vec!["node2".to_string(), "node3".to_string()]);

        // Set node2 as current leader
        *node.current_leader.write() = Some("node2".to_string());

        let tn = TimeoutNow {
            term: 0,
            leader_id: "node3".to_string(), // Wrong leader
        };

        let initial_term = node.current_term();
        node.handle_message(&"node3".to_string(), &Message::TimeoutNow(tn));

        // Should not start election
        assert_eq!(node.state(), RaftState::Follower);
        assert_eq!(node.current_term(), initial_term);
    }

    #[test]
    fn test_handle_timeout_now_rejects_wrong_term() {
        let node = create_test_node("node1", vec!["node2".to_string()]);

        // Set node2 as current leader and set term
        *node.current_leader.write() = Some("node2".to_string());
        {
            let mut persistent = node.persistent.write();
            persistent.current_term = 5;
        }

        let tn = TimeoutNow {
            term: 3, // Wrong term
            leader_id: "node2".to_string(),
        };

        node.handle_message(&"node2".to_string(), &Message::TimeoutNow(tn));

        // Should not start election
        assert_eq!(node.state(), RaftState::Follower);
        assert_eq!(node.current_term(), 5);
    }

    #[test]
    fn test_cancel_transfer() {
        let node = create_test_node("node1", vec!["node2".to_string()]);
        node.become_leader();

        // Start transfer
        node.transfer_leadership(&"node2".to_string()).unwrap();
        assert!(node.is_transfer_in_progress());

        // Cancel transfer
        node.cancel_transfer();
        assert!(!node.is_transfer_in_progress());
    }

    #[test]
    fn test_timeout_now_serialization() {
        let tn = TimeoutNow {
            term: 42,
            leader_id: "leader1".to_string(),
        };

        let bytes = bincode::serialize(&tn).expect("serialize");
        let restored: TimeoutNow = bincode::deserialize(&bytes).expect("deserialize");

        assert_eq!(restored.term, 42);
        assert_eq!(restored.leader_id, "leader1");
    }

    #[test]
    fn test_transfer_state_fields() {
        let state = TransferState {
            target: "target1".to_string(),
            initiated_term: 5,
            started_at: Instant::now(),
        };

        assert_eq!(state.target, "target1");
        assert_eq!(state.initiated_term, 5);
    }

    // ========== Automatic Compaction Tests ==========

    #[test]
    fn test_compaction_config_defaults() {
        let config = RaftConfig::default();
        assert_eq!(config.compaction_check_interval, 10);
        assert_eq!(config.compaction_cooldown_ms, 60_000);
    }

    #[test]
    fn test_can_compact_no_previous() {
        let node = create_test_node("node1", vec![]);
        // No previous compaction, should be able to compact
        assert!(node.can_compact());
    }

    #[test]
    fn test_can_compact_within_cooldown() {
        let node = create_test_node("node1", vec![]);
        // Mark compaction just happened
        node.mark_compacted();
        // Should NOT be able to compact (within cooldown)
        assert!(!node.can_compact());
    }

    #[test]
    fn test_can_compact_after_cooldown() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        // Create config with very short cooldown for testing
        let mut config = RaftConfig::default();
        config.compaction_cooldown_ms = 1; // 1ms cooldown
        let node = RaftNode::new("node1".to_string(), vec![], transport, config);

        // Mark compaction
        node.mark_compacted();

        // Wait for cooldown to elapse
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Should now be able to compact
        assert!(node.can_compact());
    }

    #[test]
    fn test_save_snapshot_to_store() {
        let node = create_test_node("node1", vec![]);
        let store = tensor_store::TensorStore::new();

        let metadata = SnapshotMetadata::new(
            100,
            5,
            [0u8; 32],
            vec!["node1".to_string(), "node2".to_string()],
            1024,
        );
        let data = vec![1, 2, 3, 4, 5];

        // Save snapshot
        node.save_snapshot(&metadata, &data, &store).unwrap();

        // Verify it can be loaded back
        let (loaded_meta, loaded_data) = RaftNode::load_snapshot("node1", &store).unwrap();
        assert_eq!(loaded_meta.last_included_index, 100);
        assert_eq!(loaded_meta.last_included_term, 5);
        assert_eq!(loaded_meta.config, vec!["node1", "node2"]);
        assert_eq!(loaded_data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_load_snapshot_missing() {
        let store = tensor_store::TensorStore::new();
        // No snapshot saved, should return None
        assert!(RaftNode::load_snapshot("node1", &store).is_none());
    }

    #[test]
    fn test_with_store_loads_snapshot() {
        let store = tensor_store::TensorStore::new();

        // First, create a node and save a snapshot
        let transport1 = Arc::new(MemoryTransport::new("node1".to_string()));
        let node1 = RaftNode::new(
            "node1".to_string(),
            vec![],
            transport1,
            RaftConfig::default(),
        );

        let data = vec![10, 20, 30];

        // Compute proper hash of the data
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let snapshot_hash: [u8; 32] = hasher.finalize().into();

        let metadata = SnapshotMetadata::new(
            50,
            3,
            snapshot_hash,
            vec!["node1".to_string()],
            data.len() as u64,
        );
        node1.save_snapshot(&metadata, &data, &store).unwrap();

        // Now create a new node using with_store - it should load the snapshot
        let transport2 = Arc::new(MemoryTransport::new("node1".to_string()));
        let node2 = RaftNode::with_store(
            "node1".to_string(),
            vec![],
            transport2,
            RaftConfig::default(),
            &store,
        );

        // Verify snapshot metadata was loaded
        let loaded_meta = node2.get_snapshot_metadata().unwrap();
        assert_eq!(loaded_meta.last_included_index, 50);
        assert_eq!(loaded_meta.last_included_term, 3);
    }

    #[test]
    fn test_try_auto_compact_interval_skip() {
        let node = create_test_node("node1", vec![]);

        // Reset tick counter
        node.compaction_tick_counter.store(0, Ordering::SeqCst);

        // First tick (0 % 10 == 0), so it would check
        // But should_compact() will return false (no finalized entries)
        // Let's verify interval skipping by checking ticks 1-9 are skipped
        for i in 1..10 {
            node.compaction_tick_counter.store(i, Ordering::SeqCst);
            // Increment will make it i+1, so i=1 becomes 2, which is 2 % 10 != 0
        }
        // At tick 9, after increment it becomes 10, which is 10 % 10 == 0
        // So only multiples of 10 will proceed past the interval check
    }

    #[test]
    fn test_compaction_tick_counter_increments() {
        let node = create_test_node("node1", vec![]);
        assert_eq!(node.compaction_tick_counter.load(Ordering::SeqCst), 0);

        // The counter is incremented in try_auto_compact, but we can verify initialization
        node.compaction_tick_counter.fetch_add(1, Ordering::SeqCst);
        assert_eq!(node.compaction_tick_counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_snapshot_key_patterns() {
        assert_eq!(
            RaftNode::snapshot_meta_key("node1"),
            "_raft:snapshot:meta:node1"
        );
        assert_eq!(
            RaftNode::snapshot_data_key("node1"),
            "_raft:snapshot:data:node1"
        );
    }

    #[test]
    fn test_mark_compacted_sets_timestamp() {
        let node = create_test_node("node1", vec![]);

        // Initially None
        assert!(node.last_compaction.read().is_none());

        // Mark compacted
        node.mark_compacted();

        // Now should have a timestamp
        assert!(node.last_compaction.read().is_some());
    }

    #[test]
    fn test_store_field_none_by_default() {
        let node = create_test_node("node1", vec![]);
        assert!(node.store.is_none());
    }

    #[test]
    fn test_with_store_sets_store_reference() {
        let store = tensor_store::TensorStore::new();
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node = RaftNode::with_store(
            "node1".to_string(),
            vec![],
            transport,
            RaftConfig::default(),
            &store,
        );

        // Store should be set
        assert!(node.store.is_some());
    }

    #[test]
    fn test_create_snapshot_computes_hash() {
        let node = create_test_node("node1", vec![]);
        node.become_leader();

        // Add some log entries
        let block = create_test_block(1);
        node.propose(block).unwrap();
        node.set_finalized_height(1);

        // Create snapshot
        let (metadata, data) = node.create_snapshot().unwrap();

        // Verify hash is computed correctly (not zeroed)
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let expected_hash: [u8; 32] = hasher.finalize().into();

        assert_eq!(metadata.snapshot_hash, expected_hash);
        assert_ne!(metadata.snapshot_hash, [0u8; 32]);
    }

    #[test]
    fn test_install_snapshot_validates_hash() {
        let node = create_test_node("node1", vec![]);

        // Create valid snapshot data
        let entry = LogEntry::new(1, 1, create_test_block(1));
        let data = bincode::serialize(&vec![entry]).unwrap();

        // Compute correct hash
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let correct_hash: [u8; 32] = hasher.finalize().into();

        // Create metadata with WRONG hash
        let wrong_metadata = SnapshotMetadata::new(1, 1, [0u8; 32], vec![], data.len() as u64);

        // Should fail with hash mismatch
        let result = node.install_snapshot(wrong_metadata, &data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hash mismatch"));

        // Create metadata with CORRECT hash - should succeed
        let correct_metadata = SnapshotMetadata::new(1, 1, correct_hash, vec![], data.len() as u64);
        let result = node.install_snapshot(correct_metadata, &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_startup_validates_snapshot_hash() {
        let store = tensor_store::TensorStore::new();

        // Create valid snapshot data
        let entry = LogEntry::new(1, 1, create_test_block(1));
        let data = bincode::serialize(&vec![entry]).unwrap();

        // Compute correct hash
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let correct_hash: [u8; 32] = hasher.finalize().into();

        // Save snapshot with correct hash
        let metadata = SnapshotMetadata::new(1, 1, correct_hash, vec![], data.len() as u64);
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let node1 = RaftNode::with_store(
            "node1".to_string(),
            vec![],
            transport.clone(),
            RaftConfig::default(),
            &store,
        );
        node1.save_snapshot(&metadata, &data, &store).unwrap();
        drop(node1);

        // Load - should restore valid snapshot
        let node2 = RaftNode::with_store(
            "node1".to_string(),
            vec![],
            transport.clone(),
            RaftConfig::default(),
            &store,
        );
        let restored = node2.get_snapshot_metadata();
        assert!(restored.is_some());
        assert_eq!(restored.unwrap().last_included_index, 1);

        // Now corrupt the data and save again
        let corrupted_data = vec![0u8; data.len()];
        // Save with original hash but corrupted data
        let mut snap_data = tensor_store::TensorData::new();
        snap_data.set(
            "data",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Bytes(corrupted_data)),
        );
        store
            .put(&format!("_raft:snapshot:data:{}", "node1"), snap_data)
            .unwrap();
        drop(node2);

        // Load again - should reject corrupted snapshot
        let node3 = RaftNode::with_store(
            "node1".to_string(),
            vec![],
            transport,
            RaftConfig::default(),
            &store,
        );
        // Corrupted snapshot should be ignored
        let restored = node3.get_snapshot_metadata();
        assert!(restored.is_none());
    }
}
