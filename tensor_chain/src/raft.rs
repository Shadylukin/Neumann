//! Tensor-Raft consensus implementation.
//!
//! Modified Raft protocol with tensor-native optimizations:
//! - Similarity fast-path for block validation
//! - State embedding for tie-breaking
//! - Two-phase finality (committed -> finalized)

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tensor_store::SparseVector;

use crate::block::{Block, NodeId};
use crate::error::{ChainError, Result};
use crate::membership::MembershipManager;
use crate::network::{
    AppendEntries, AppendEntriesResponse, LogEntry, Message, RequestVote, RequestVoteResponse,
    Transport,
};
use crate::validation::FastPathValidator;

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

/// Statistics for fast-path validation.
#[derive(Debug, Default)]
pub struct FastPathStats {
    /// Number of blocks that used fast-path validation.
    pub fast_path_accepted: AtomicU64,
    /// Number of blocks that failed fast-path and required full validation.
    pub fast_path_rejected: AtomicU64,
    /// Number of blocks that required full validation (no embedding or first blocks).
    pub full_validation_required: AtomicU64,
}

impl FastPathStats {
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
}

impl RaftNode {
    /// Create a new Raft node.
    pub fn new(
        node_id: NodeId,
        peers: Vec<NodeId>,
        transport: Arc<dyn Transport>,
        config: RaftConfig,
    ) -> Self {
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
            config,
            membership: None,
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

    /// Get a reference to the transport layer.
    pub fn transport(&self) -> &Arc<dyn Transport> {
        &self.transport
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
            Message::AppendEntries(ae) => self.handle_append_entries(from, ae),
            Message::AppendEntriesResponse(aer) => {
                self.handle_append_entries_response(from, aer);
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
                vote_granted = true;
                persistent.voted_for = Some(rv.candidate_id.clone());
                *self.last_heartbeat.write() = Instant::now();
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

    /// Handle AppendEntries RPC.
    fn handle_append_entries(&self, _from: &NodeId, ae: &AppendEntries) -> Option<Message> {
        let mut persistent = self.persistent.write();
        let mut success = false;
        let mut match_index = 0;
        let mut used_fast_path = false;

        // Update term if needed
        if ae.term > persistent.current_term {
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

    /// Start an election.
    pub fn start_election(&self) {
        let mut persistent = self.persistent.write();
        persistent.current_term += 1;
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

        // Capture sparse embedding before moving block
        let embedding = block.header.delta_embedding.clone();

        let mut persistent = self.persistent.write();
        let index = persistent.log.len() as u64 + 1;
        let term = persistent.current_term;

        let entry = LogEntry { term, index, block };
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
                    self.start_election_async().await?;
                }
            },
            RaftState::Leader => {
                // Send heartbeats
                if elapsed > self.config.heartbeat_interval {
                    self.send_heartbeats().await?;
                    *self.last_heartbeat.write() = Instant::now();
                }
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
    use crate::block::BlockHeader;
    use crate::network::MemoryTransport;

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
            persistent.log.push(LogEntry {
                term: 1,
                index: 1,
                block,
            });
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
                persistent.log.push(LogEntry {
                    term: 1,
                    index: i,
                    block,
                });
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
                persistent.log.push(LogEntry {
                    term: 1,
                    index: i,
                    block: create_test_block(i),
                });
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
                persistent.log.push(LogEntry {
                    term: 1,
                    index: i,
                    block: create_test_block(i),
                });
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
            persistent.log.push(LogEntry {
                term: 1,
                index: 1,
                block: create_test_block(1),
            });
        }

        // Leader sends entry at same index but different term
        let ae = AppendEntries {
            term: 2,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 2,
                index: 1,
                block: create_test_block(1),
            }],
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

        // First, establish node2 as leader and add sufficient embedding history (min_leader_history = 3)
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
            persistent.log.push(LogEntry {
                term: 2,
                index: 1,
                block: create_test_block(1),
            });
            persistent.log.push(LogEntry {
                term: 3,
                index: 2,
                block: create_test_block(2),
            });
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
            persistent.log.push(LogEntry {
                term: 1,
                index: 1,
                block: create_test_block(1),
            });
        }

        // Leader claims prev_log at index 1 with term 5 (but we have term 1)
        let ae = AppendEntries {
            term: 2,
            leader_id: "node2".to_string(),
            prev_log_index: 1,
            prev_log_term: 5, // Wrong term
            entries: vec![LogEntry {
                term: 2,
                index: 2,
                block: create_test_block(2),
            }],
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
                persistent.log.push(LogEntry {
                    term: 1,
                    index: i,
                    block: create_test_block(i),
                });
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
            persistent.log.push(LogEntry {
                term: 3,
                index: 1,
                block: create_test_block(1),
            });
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
                persistent.log.push(LogEntry {
                    term: 1,
                    index: i,
                    block: create_test_block(i),
                });
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
                persistent.log.push(LogEntry {
                    term: 1,
                    index: i,
                    block: create_test_block(i),
                });
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
            persistent.log.push(LogEntry {
                term: 1,
                index: 1,
                block: create_test_block(1),
            });
        }

        // Leader sends more entries
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 1,
            prev_log_term: 1,
            entries: vec![
                LogEntry {
                    term: 1,
                    index: 2,
                    block: create_test_block(2),
                },
                LogEntry {
                    term: 1,
                    index: 3,
                    block: create_test_block(3),
                },
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
            persistent.log.push(LogEntry {
                term: 1, // Old term
                index: 1,
                block: create_test_block(1),
            });
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
            persistent.log.push(LogEntry {
                term: 1,
                index: 1,
                block: Block::genesis("node1".to_string()),
            });
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
            persistent.log.push(LogEntry {
                term: 1,
                index: 1,
                block: Block::genesis("node1".to_string()),
            });
            persistent.log.push(LogEntry {
                term: 1,
                index: 2,
                block: Block::genesis("node1".to_string()),
            });
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
}
