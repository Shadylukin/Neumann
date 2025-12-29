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
    /// Recent block embeddings for fast-path (leader only).
    recent_embeddings: RwLock<Vec<Vec<f32>>>,
    /// Finalized height (checkpointed).
    finalized_height: AtomicU64,
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
            config,
            last_heartbeat: RwLock::new(Instant::now()),
            votes_received: RwLock::new(Vec::new()),
            state_embedding: RwLock::new(Vec::new()),
            recent_embeddings: RwLock::new(Vec::new()),
            finalized_height: AtomicU64::new(0),
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
            }
            Message::AppendEntries(ae) => self.handle_append_entries(from, ae),
            Message::AppendEntriesResponse(aer) => {
                self.handle_append_entries_response(from, aer);
                None
            }
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
    fn handle_append_entries(&self, from: &NodeId, ae: &AppendEntries) -> Option<Message> {
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
            *self.current_leader.write() = Some(ae.leader_id.clone());
            *self.last_heartbeat.write() = Instant::now();

            // Check if we can use fast-path
            if self.config.enable_fast_path && ae.block_embedding.is_some() {
                let embedding = ae.block_embedding.as_ref().unwrap();
                if self.can_use_fast_path(from, embedding) {
                    used_fast_path = true;
                }
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
                    volatile.commit_index =
                        ae.leader_commit.min(persistent.log.len() as u64);
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

        let mut leader_state = self.leader_state.write();
        if let Some(ref mut ls) = *leader_state {
            if aer.success {
                ls.next_index.insert(from.clone(), aer.match_index + 1);
                ls.match_index.insert(from.clone(), aer.match_index);

                // Try to advance commit index
                drop(leader_state);
                self.try_advance_commit_index();
            } else {
                // Decrement next_index and retry
                let next = ls.next_index.entry(from.clone()).or_insert(1);
                if *next > 1 {
                    *next -= 1;
                }
            }
        }
    }

    /// Check if fast-path validation can be used.
    fn can_use_fast_path(&self, leader: &NodeId, embedding: &[f32]) -> bool {
        // Only use fast-path if embedding is similar to recent blocks from same leader
        if self.current_leader.read().as_ref() != Some(leader) {
            return false;
        }

        let recent = self.recent_embeddings.read();
        if recent.is_empty() {
            return false;
        }

        // Check similarity with most recent embedding
        let last = &recent[recent.len() - 1];
        cosine_similarity(embedding, last) >= self.config.similarity_threshold
    }

    /// Start an election.
    pub fn start_election(&self) {
        let mut persistent = self.persistent.write();
        persistent.current_term += 1;
        persistent.voted_for = Some(self.node_id.clone());

        *self.state.write() = RaftState::Candidate;
        *self.votes_received.write() = vec![self.node_id.clone()]; // Vote for self

        let (last_log_index, last_log_term) = self.last_log_info();
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

        let mut persistent = self.persistent.write();
        let index = persistent.log.len() as u64 + 1;
        let term = persistent.current_term;

        let entry = LogEntry { term, index, block };
        persistent.log.push(entry);

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
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
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
}
