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

use crate::block::{Block, NodeId};
use crate::error::{ChainError, Result};
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
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            election_timeout: (150, 300),
            heartbeat_interval: 50,
            similarity_threshold: 0.95,
            enable_fast_path: true,
            quorum_size: None,
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
        self.full_validation_required.fetch_add(1, Ordering::Relaxed);
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
    /// Recent embeddings per leader.
    leader_embeddings: RwLock<HashMap<NodeId, Vec<Vec<f32>>>>,
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

    /// Add an embedding for a leader.
    pub fn add_embedding(&self, leader: &NodeId, embedding: Vec<f32>) {
        let mut embeddings = self.leader_embeddings.write();
        let history = embeddings.entry(leader.clone()).or_default();

        history.push(embedding);

        // Keep only recent embeddings
        if history.len() > self.max_history {
            history.remove(0);
        }
    }

    /// Get embeddings for a leader.
    pub fn get_embeddings(&self, leader: &NodeId) -> Vec<Vec<f32>> {
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
    #[allow(dead_code)] // Will be used for async message sending
    transport: Arc<dyn Transport>,
    /// Configuration.
    config: RaftConfig,
    /// Last heartbeat received.
    last_heartbeat: RwLock<Instant>,
    /// Votes received in current election.
    votes_received: RwLock<Vec<NodeId>>,
    /// State embedding for similarity-based tie-breaking.
    state_embedding: RwLock<Vec<f32>>,
    /// Finalized height (checkpointed).
    finalized_height: AtomicU64,
    /// Fast-path validation state.
    fast_path_state: FastPathState,
    /// Fast-path validator for replication.
    fast_path_validator: FastPathValidator,
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
            state_embedding: RwLock::new(Vec::new()),
            finalized_height: AtomicU64::new(0),
            fast_path_state: FastPathState::default(),
            fast_path_validator: FastPathValidator::new(config.similarity_threshold, 3),
            config,
        }
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
    pub fn update_state_embedding(&self, embedding: Vec<f32>) {
        *self.state_embedding.write() = embedding;
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
        if rv.term == persistent.current_term {
            let can_vote = persistent.voted_for.is_none()
                || persistent.voted_for.as_ref() == Some(&rv.candidate_id);

            // Compute last log info from the lock we already hold
            let (last_log_index, last_log_term) = if persistent.log.is_empty() {
                (0, 0)
            } else {
                let last = &persistent.log[persistent.log.len() - 1];
                (last.index, last.term)
            };
            let log_ok = rv.last_log_term > last_log_term
                || (rv.last_log_term == last_log_term && rv.last_log_index >= last_log_index);

            if can_vote && log_ok {
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
                let embedding = ae.block_embedding.as_ref().unwrap();
                let history = self.fast_path_state.get_embeddings(&ae.leader_id);
                let result = self.fast_path_validator.check_fast_path(embedding, &history);

                used_fast_path = result.can_use_fast_path;

                // Record statistics
                if used_fast_path {
                    self.fast_path_state.stats.record_fast_path();
                } else if result.rejection_reason.as_deref() == Some("periodic full validation required") {
                    self.fast_path_state.stats.record_full_validation();
                } else {
                    self.fast_path_state.stats.record_rejected();
                }

                // Record this validation for periodic full validation tracking
                self.fast_path_validator.record_validation(used_fast_path);

                // Track embedding for future fast-path checks
                self.fast_path_state.add_embedding(&ae.leader_id, embedding.clone());
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
    fn become_leader(&self) {
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

        // Capture embedding before moving block
        let embedding = block.header.delta_embedding.clone();

        let mut persistent = self.persistent.write();
        let index = persistent.log.len() as u64 + 1;
        let term = persistent.current_term;

        let entry = LogEntry { term, index, block };
        persistent.log.push(entry);

        // Track embedding for fast-path validation by followers
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
    pub fn get_entries_for_follower(&self, follower: &NodeId) -> (u64, u64, Vec<LogEntry>, Option<Vec<f32>>) {
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

        // Extract embedding from last entry for fast-path
        let block_embedding = entries.last().map(|e| e.block.header.delta_embedding.clone());

        (prev_log_index, prev_log_term, entries, block_embedding)
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
            state_embedding: vec![],
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

        state.add_embedding(&leader, vec![1.0, 0.0, 0.0]);
        state.add_embedding(&leader, vec![0.0, 1.0, 0.0]);

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

        state.add_embedding(&leader, vec![1.0, 0.0, 0.0]);
        state.add_embedding(&leader, vec![0.0, 1.0, 0.0]);
        state.add_embedding(&leader, vec![0.0, 0.0, 1.0]);

        // Should only keep last 2
        assert_eq!(state.leader_history_size(&leader), 2);

        let embeddings = state.get_embeddings(&leader);
        assert_eq!(embeddings[0], vec![0.0, 1.0, 0.0]);
        assert_eq!(embeddings[1], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_fast_path_state_multiple_leaders() {
        let state = FastPathState::new(5);

        state.add_embedding(&"leader1".to_string(), vec![1.0, 0.0]);
        state.add_embedding(&"leader2".to_string(), vec![0.0, 1.0]);

        assert_eq!(state.leader_history_size(&"leader1".to_string()), 1);
        assert_eq!(state.leader_history_size(&"leader2".to_string()), 1);
    }

    #[test]
    fn test_fast_path_state_clear_leader() {
        let state = FastPathState::new(5);
        let leader = "leader1".to_string();

        state.add_embedding(&leader, vec![1.0, 0.0]);
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
            ls.as_ref().unwrap().next_index.get(&"node2".to_string()).copied().unwrap()
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
            ls.as_ref().unwrap().next_index.get(&"node2".to_string()).copied().unwrap()
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
            node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
            node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
            node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        }

        // Send append entries with similar embedding
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(vec![1.0, 0.0, 0.0]),
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
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        *node.current_leader.write() = Some("node2".to_string());

        // Now receive from node3 - should trigger leader change cleanup
        let ae = AppendEntries {
            term: 2,
            leader_id: "node3".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(vec![1.0, 0.0, 0.0]),
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
        assert_eq!(node.fast_path_state.leader_history_size(&"node2".to_string()), 0);
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
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        *node.current_leader.write() = Some("node2".to_string());

        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(vec![1.0, 0.0, 0.0]),
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
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);

        // Orthogonal vector - low similarity
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(vec![0.0, 1.0, 0.0]),
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
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);
        node.fast_path_state.add_embedding(&"node2".to_string(), vec![1.0, 0.0, 0.0]);

        // Send similar embedding - should use fast-path
        let ae = AppendEntries {
            term: 1,
            leader_id: "node2".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(vec![1.0, 0.0, 0.0]),
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
            block_embedding: Some(vec![0.0, 1.0, 0.0]),
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

        node.update_state_embedding(vec![1.0, 2.0, 3.0]);

        let embedding = node.state_embedding.read();
        assert_eq!(*embedding, vec![1.0, 2.0, 3.0]);
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
            state_embedding: vec![],
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
            state_embedding: vec![],
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
            state_embedding: vec![],
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
            state_embedding: vec![],
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
            vec!["node2".to_string(), "node3".to_string(), "node4".to_string()],
        );

        node.start_election();

        // First vote from node2
        let rvr1 = RequestVoteResponse {
            term: 1,
            vote_granted: true,
            voter_id: "node2".to_string(),
        };
        node.handle_message(&"node2".to_string(), &Message::RequestVoteResponse(rvr1.clone()));

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
}
