//! Network transport abstraction for distributed consensus.
//!
//! Provides a pluggable transport layer for node communication:
//! - `Transport` trait defines the interface
//! - `MemoryTransport` for testing
//! - Message types for Raft and sync protocols

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use tracing;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tensor_store::SparseVector;
use tokio::sync::mpsc;

use crate::{
    block::{Block, BlockHash, NodeId},
    error::{ChainError, Result},
};

/// Configuration for a network peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerConfig {
    /// Node identifier.
    pub node_id: NodeId,
    /// Network address (host:port).
    pub address: String,
}

/// Message types for consensus protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Message {
    // Raft messages
    /// Request vote during leader election.
    RequestVote(RequestVote),
    /// Response to vote request.
    RequestVoteResponse(RequestVoteResponse),
    /// Pre-vote request (prevents disruptive elections).
    PreVote(PreVote),
    /// Response to pre-vote request.
    PreVoteResponse(PreVoteResponse),
    /// Immediate election trigger for leadership transfer.
    TimeoutNow(TimeoutNow),
    /// Append entries (log replication).
    AppendEntries(AppendEntries),
    /// Response to append entries.
    AppendEntriesResponse(AppendEntriesResponse),

    // Sync messages
    /// Request blocks from a peer.
    BlockRequest(BlockRequest),
    /// Response with requested blocks.
    BlockResponse(BlockResponse),
    /// Request state snapshot.
    SnapshotRequest(SnapshotRequest),
    /// Response with state snapshot chunk.
    SnapshotResponse(SnapshotResponse),

    // Heartbeat
    /// Ping to check liveness.
    Ping { term: u64 },
    /// Pong response.
    Pong { term: u64 },

    // 2PC Distributed Transaction messages
    /// Prepare request for distributed transaction.
    TxPrepare(TxPrepareMsg),
    /// Vote response for prepare request.
    TxPrepareResponse(TxPrepareResponseMsg),
    /// Commit request for distributed transaction.
    TxCommit(TxCommitMsg),
    /// Abort request for distributed transaction.
    TxAbort(TxAbortMsg),
    /// Response to commit/abort.
    TxAck(TxAckMsg),

    // Distributed Query messages
    /// Request to execute a query on a remote shard.
    QueryRequest(QueryRequest),
    /// Response with query results.
    QueryResponse(QueryResponse),

    // Gossip protocol messages
    /// Gossip message for membership protocol.
    Gossip(crate::gossip::GossipMessage),

    // Partition merge protocol messages
    /// Initialize merge session after partition heal.
    MergeInit(MergeInit),
    /// Acknowledge merge initialization.
    MergeAck(MergeAck),
    /// Exchange membership view summaries.
    ViewExchange(MergeViewExchange),
    /// Request data merge information.
    DataMergeRequest(DataMergeRequest),
    /// Response with data merge information.
    DataMergeResponse(DataMergeResponse),
    /// Request transaction reconciliation.
    TxReconcileRequest(TxReconcileRequest),
    /// Response with transaction reconciliation result.
    TxReconcileResponse(TxReconcileResponse),
    /// Finalize merge session.
    MergeFinalize(MergeFinalize),
}

impl Message {
    /// Extract the routing embedding from a message for geometric routing decisions.
    ///
    /// Returns the most semantically relevant embedding for the message type:
    /// - RequestVote/PreVote: candidate's state embedding
    /// - AppendEntries: block embedding (if present)
    /// - TxPrepare: transaction delta embedding
    pub fn routing_embedding(&self) -> Option<&SparseVector> {
        match self {
            Message::RequestVote(rv) => Some(&rv.state_embedding),
            Message::PreVote(pv) => Some(&pv.state_embedding),
            Message::AppendEntries(ae) => ae.block_embedding.as_ref(),
            Message::TxPrepare(tp) => Some(&tp.delta_embedding),
            _ => None,
        }
    }

    pub fn has_routing_embedding(&self) -> bool {
        self.routing_embedding().is_some()
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Message::RequestVote(_) => "RequestVote",
            Message::RequestVoteResponse(_) => "RequestVoteResponse",
            Message::PreVote(_) => "PreVote",
            Message::PreVoteResponse(_) => "PreVoteResponse",
            Message::TimeoutNow(_) => "TimeoutNow",
            Message::AppendEntries(_) => "AppendEntries",
            Message::AppendEntriesResponse(_) => "AppendEntriesResponse",
            Message::BlockRequest(_) => "BlockRequest",
            Message::BlockResponse(_) => "BlockResponse",
            Message::SnapshotRequest(_) => "SnapshotRequest",
            Message::SnapshotResponse(_) => "SnapshotResponse",
            Message::Ping { .. } => "Ping",
            Message::Pong { .. } => "Pong",
            Message::TxPrepare(_) => "TxPrepare",
            Message::TxPrepareResponse(_) => "TxPrepareResponse",
            Message::TxCommit(_) => "TxCommit",
            Message::TxAbort(_) => "TxAbort",
            Message::TxAck(_) => "TxAck",
            Message::QueryRequest(_) => "QueryRequest",
            Message::QueryResponse(_) => "QueryResponse",
            Message::Gossip(_) => "Gossip",
            Message::MergeInit(_) => "MergeInit",
            Message::MergeAck(_) => "MergeAck",
            Message::ViewExchange(_) => "ViewExchange",
            Message::DataMergeRequest(_) => "DataMergeRequest",
            Message::DataMergeResponse(_) => "DataMergeResponse",
            Message::TxReconcileRequest(_) => "TxReconcileRequest",
            Message::TxReconcileResponse(_) => "TxReconcileResponse",
            Message::MergeFinalize(_) => "MergeFinalize",
        }
    }
}

/// Request vote message for leader election.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVote {
    /// Candidate's term.
    pub term: u64,
    /// Candidate requesting vote.
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry.
    pub last_log_index: u64,
    /// Term of candidate's last log entry.
    pub last_log_term: u64,
    /// Similarity embedding of candidate's state (for tie-breaking).
    pub state_embedding: SparseVector,
}

/// Response to vote request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteResponse {
    /// Current term, for candidate to update itself.
    pub term: u64,
    /// True if candidate received vote.
    pub vote_granted: bool,
    /// Voter's node ID.
    pub voter_id: NodeId,
}

/// Pre-vote request for disruption-free elections.
///
/// Sent before incrementing term to check if election would succeed.
/// Prevents partitioned nodes from disrupting the cluster with high terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreVote {
    /// Candidate's CURRENT term (not incremented).
    pub term: u64,
    /// Candidate requesting pre-vote.
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry.
    pub last_log_index: u64,
    /// Term of candidate's last log entry.
    pub last_log_term: u64,
    /// State embedding for geometric tie-breaking.
    pub state_embedding: SparseVector,
}

/// Response to pre-vote request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreVoteResponse {
    /// Responder's current term.
    pub term: u64,
    /// True if responder would grant vote in a real election.
    pub vote_granted: bool,
    /// Voter's node ID.
    pub voter_id: NodeId,
}

/// Immediate election trigger for leadership transfer.
///
/// Sent by leader to target node to initiate immediate election,
/// bypassing normal election timeout and pre-vote phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutNow {
    /// Leader's current term.
    pub term: u64,
    /// Leader initiating transfer.
    pub leader_id: NodeId,
}

/// Append entries message for log replication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntries {
    /// Leader's term.
    pub term: u64,
    /// Leader's ID for redirects.
    pub leader_id: NodeId,
    /// Index of log entry immediately preceding new ones.
    pub prev_log_index: u64,
    /// Term of prev_log_index entry.
    pub prev_log_term: u64,
    /// Log entries to store (empty for heartbeat).
    pub entries: Vec<LogEntry>,
    /// Leader's commit index.
    pub leader_commit: u64,
    /// Block embedding for similarity fast-path (sparse for bandwidth efficiency).
    pub block_embedding: Option<SparseVector>,
}

/// Response to append entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term.
    pub term: u64,
    /// True if follower contained entry matching prev_log.
    pub success: bool,
    /// Follower's node ID.
    pub follower_id: NodeId,
    /// Last index replicated.
    pub match_index: u64,
    /// Used similarity fast-path.
    pub used_fast_path: bool,
}

/// A log entry in the Raft log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Term when entry was received.
    pub term: u64,
    /// Index of entry in log.
    pub index: u64,
    /// The block being proposed.
    pub block: Block,
    /// Optional config change (for joint consensus membership changes).
    #[serde(default)]
    pub config_change: Option<ConfigChange>,
}

impl LogEntry {
    pub fn new(term: u64, index: u64, block: Block) -> Self {
        Self {
            term,
            index,
            block,
            config_change: None,
        }
    }

    pub fn config(term: u64, index: u64, change: ConfigChange) -> Self {
        Self {
            term,
            index,
            block: Block::default(),
            config_change: Some(change),
        }
    }

    pub fn is_config_change(&self) -> bool {
        self.config_change.is_some()
    }
}

/// Request for blocks from a peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRequest {
    /// Start height.
    pub from_height: u64,
    /// End height (inclusive).
    pub to_height: u64,
    /// Requester's node ID.
    pub requester_id: NodeId,
}

/// Response with blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockResponse {
    /// Requested blocks.
    pub blocks: Vec<Block>,
    /// Responder's current height.
    pub current_height: u64,
}

/// Request for state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotRequest {
    /// Requester's node ID.
    pub requester_id: NodeId,
    /// Chunk offset (for resumable transfer).
    pub offset: u64,
    /// Requested chunk size.
    pub chunk_size: u64,
}

/// Response with snapshot chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotResponse {
    /// Height at which snapshot was taken.
    pub snapshot_height: u64,
    /// Block hash at snapshot height.
    pub snapshot_hash: BlockHash,
    /// Chunk data.
    pub data: Vec<u8>,
    /// Offset in snapshot.
    pub offset: u64,
    /// Total snapshot size.
    pub total_size: u64,
    /// Whether this is the last chunk.
    pub is_last: bool,
}

// 2PC Distributed Transaction Messages

/// Prepare request for distributed transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxPrepareMsg {
    /// Distributed transaction ID.
    pub tx_id: u64,
    /// Coordinator node ID.
    pub coordinator: NodeId,
    /// Shard being addressed.
    pub shard_id: usize,
    /// Operations to execute on this shard.
    pub operations: Vec<crate::block::Transaction>,
    /// Delta embedding for conflict detection (sparse for bandwidth efficiency).
    pub delta_embedding: SparseVector,
    /// Timeout in milliseconds.
    pub timeout_ms: u64,
}

/// Vote response from participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxPrepareResponseMsg {
    /// Distributed transaction ID.
    pub tx_id: u64,
    /// Shard responding.
    pub shard_id: usize,
    /// Vote result.
    pub vote: TxVote,
}

/// Vote from a participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TxVote {
    /// Participant is ready to commit.
    Yes {
        /// Lock handle for the prepared transaction.
        lock_handle: u64,
        /// Delta computed by participant (sparse for bandwidth efficiency).
        delta: SparseVector,
        /// Keys affected by this transaction.
        affected_keys: Vec<String>,
    },
    /// Participant cannot commit.
    No {
        /// Reason for rejection.
        reason: String,
    },
    /// Participant detected a conflict.
    Conflict {
        /// Similarity with conflicting transaction.
        similarity: f32,
        /// ID of the conflicting transaction.
        conflicting_tx: u64,
    },
}

impl From<crate::distributed_tx::PrepareVote> for TxVote {
    fn from(vote: crate::distributed_tx::PrepareVote) -> Self {
        use crate::distributed_tx::PrepareVote;
        match vote {
            PrepareVote::Yes { lock_handle, delta } => TxVote::Yes {
                lock_handle,
                delta: delta.delta.clone(),
                affected_keys: delta.affected_keys.iter().cloned().collect(),
            },
            PrepareVote::No { reason } => TxVote::No { reason },
            PrepareVote::Conflict {
                similarity,
                conflicting_tx,
            } => TxVote::Conflict {
                similarity,
                conflicting_tx,
            },
        }
    }
}

impl From<TxVote> for crate::distributed_tx::PrepareVote {
    fn from(vote: TxVote) -> Self {
        use crate::consensus::DeltaVector;
        use crate::distributed_tx::PrepareVote;
        match vote {
            TxVote::Yes {
                lock_handle,
                delta,
                affected_keys,
            } => PrepareVote::Yes {
                lock_handle,
                delta: DeltaVector::from_sparse(delta, affected_keys.into_iter().collect(), 0),
            },
            TxVote::No { reason } => PrepareVote::No { reason },
            TxVote::Conflict {
                similarity,
                conflicting_tx,
            } => PrepareVote::Conflict {
                similarity,
                conflicting_tx,
            },
        }
    }
}

/// Commit request for distributed transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxCommitMsg {
    /// Distributed transaction ID.
    pub tx_id: u64,
    /// Shards to commit.
    pub shards: Vec<usize>,
}

/// Abort request for distributed transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxAbortMsg {
    /// Distributed transaction ID.
    pub tx_id: u64,
    /// Reason for abort.
    pub reason: String,
    /// Shards to abort.
    pub shards: Vec<usize>,
}

/// Acknowledgment for commit/abort.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxAckMsg {
    /// Distributed transaction ID.
    pub tx_id: u64,
    /// Shard responding.
    pub shard_id: usize,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

// ============================================================================
// Joint Consensus Types for Dynamic Membership
// ============================================================================

/// Configuration change for Raft membership.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConfigChange {
    /// Add a new node as a non-voting learner.
    AddLearner { node_id: NodeId },
    /// Promote a learner to a voting member.
    PromoteLearner { node_id: NodeId },
    /// Remove a node from the cluster.
    RemoveNode { node_id: NodeId },
    /// Batch change: add and remove multiple nodes atomically.
    JointChange {
        additions: Vec<NodeId>,
        removals: Vec<NodeId>,
    },
}

impl ConfigChange {
    pub fn add_learner(node_id: NodeId) -> Self {
        Self::AddLearner { node_id }
    }

    pub fn promote_learner(node_id: NodeId) -> Self {
        Self::PromoteLearner { node_id }
    }

    pub fn remove_node(node_id: NodeId) -> Self {
        Self::RemoveNode { node_id }
    }

    pub fn joint_change(additions: Vec<NodeId>, removals: Vec<NodeId>) -> Self {
        Self::JointChange {
            additions,
            removals,
        }
    }
}

/// Joint consensus configuration during membership transitions.
///
/// Implements Raft's joint consensus algorithm (Section 6) where both
/// old and new configurations must agree during transitions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct JointConfig {
    /// Voters in the old configuration.
    pub old_voters: Vec<NodeId>,
    /// Voters in the new configuration.
    pub new_voters: Vec<NodeId>,
}

impl JointConfig {
    pub fn new(old_voters: Vec<NodeId>, new_voters: Vec<NodeId>) -> Self {
        Self {
            old_voters,
            new_voters,
        }
    }

    /// Check if we have a joint quorum (majority in BOTH old and new configs).
    pub fn has_joint_quorum(&self, votes: &HashSet<NodeId>) -> bool {
        let old_quorum = (self.old_voters.len() / 2) + 1;
        let new_quorum = (self.new_voters.len() / 2) + 1;

        let old_votes = self
            .old_voters
            .iter()
            .filter(|n| votes.contains(*n))
            .count();
        let new_votes = self
            .new_voters
            .iter()
            .filter(|n| votes.contains(*n))
            .count();

        old_votes >= old_quorum && new_votes >= new_quorum
    }

    pub fn all_voters(&self) -> Vec<NodeId> {
        let mut all: Vec<_> = self.old_voters.clone();
        for node in &self.new_voters {
            if !all.contains(node) {
                all.push(node.clone());
            }
        }
        all
    }
}

/// Raft membership configuration.
///
/// Tracks voters, learners, and joint consensus state for dynamic membership.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RaftMembershipConfig {
    /// Current voting members.
    pub voters: Vec<NodeId>,
    /// Non-voting learners (catching up before becoming voters).
    pub learners: Vec<NodeId>,
    /// Joint configuration during membership transitions.
    pub joint: Option<JointConfig>,
    /// Log index where this configuration was committed.
    pub config_index: u64,
}

impl RaftMembershipConfig {
    pub fn new(voters: Vec<NodeId>) -> Self {
        Self {
            voters,
            learners: Vec::new(),
            joint: None,
            config_index: 0,
        }
    }

    /// Check if we have quorum (considering joint consensus if active).
    pub fn has_quorum(&self, votes: &HashSet<NodeId>) -> bool {
        if let Some(ref joint) = self.joint {
            joint.has_joint_quorum(votes)
        } else {
            let quorum = (self.voters.len() / 2) + 1;
            self.voters.iter().filter(|n| votes.contains(*n)).count() >= quorum
        }
    }

    /// Get all nodes that should receive log entries (voters + learners + joint).
    pub fn replication_targets(&self) -> Vec<NodeId> {
        let mut targets = self.voters.clone();
        targets.extend(self.learners.clone());

        // Include new voters during joint consensus
        if let Some(ref joint) = self.joint {
            for node in &joint.new_voters {
                if !targets.contains(node) {
                    targets.push(node.clone());
                }
            }
        }

        targets
    }

    pub fn in_joint_consensus(&self) -> bool {
        self.joint.is_some()
    }

    pub fn add_learner(&mut self, node_id: NodeId) {
        if !self.learners.contains(&node_id) && !self.voters.contains(&node_id) {
            self.learners.push(node_id);
        }
    }

    pub fn promote_learner(&mut self, node_id: &NodeId) -> bool {
        if let Some(pos) = self.learners.iter().position(|n| n == node_id) {
            self.learners.remove(pos);
            self.voters.push(node_id.clone());
            true
        } else {
            false
        }
    }

    pub fn remove_node(&mut self, node_id: &NodeId) -> bool {
        if let Some(pos) = self.voters.iter().position(|n| n == node_id) {
            self.voters.remove(pos);
            return true;
        }
        if let Some(pos) = self.learners.iter().position(|n| n == node_id) {
            self.learners.remove(pos);
            return true;
        }
        false
    }

    pub fn is_voter(&self, node_id: &NodeId) -> bool {
        self.voters.contains(node_id)
    }

    pub fn is_learner(&self, node_id: &NodeId) -> bool {
        self.learners.contains(node_id)
    }
}

/// Log entry data that can be either a block or a configuration change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogEntryData {
    /// Regular block entry (boxed to reduce enum size).
    Block(Box<Block>),
    /// Configuration change entry.
    Config(ConfigChange),
}

/// Request to execute a query on a remote shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// Unique query ID for correlation.
    pub query_id: u64,
    /// The query string to execute.
    pub query: String,
    /// Target shard ID.
    pub shard_id: usize,
    /// Optional embedding for semantic routing.
    pub embedding: Option<SparseVector>,
    /// Timeout in milliseconds.
    pub timeout_ms: u64,
}

/// Response with query results from a shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResponse {
    /// Query ID for correlation.
    pub query_id: u64,
    /// Shard that executed the query.
    pub shard_id: usize,
    /// Serialized query result (bincode-encoded).
    pub result: Vec<u8>,
    /// Execution time in microseconds.
    pub execution_time_us: u64,
    /// Whether the query succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

// ========== Partition Merge Protocol Messages ==========

/// Initialize a merge session after partition heal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeInit {
    /// Unique session identifier.
    pub session_id: u64,
    /// Node initiating the merge.
    pub initiator: NodeId,
    /// Nodes that have healed and are participating.
    pub healed_nodes: Vec<NodeId>,
    /// Initiator's partition state summary.
    pub local_summary: crate::partition_merge::PartitionStateSummary,
}

/// Acknowledgment of merge initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeAck {
    /// Session ID being acknowledged.
    pub session_id: u64,
    /// Node sending the acknowledgment.
    pub responder: NodeId,
    /// Whether the node accepts the merge session.
    pub accepted: bool,
    /// Responder's partition state summary.
    pub local_summary: Option<crate::partition_merge::PartitionStateSummary>,
    /// Reason for rejection (if not accepted).
    pub reject_reason: Option<String>,
}

/// Exchange membership view summaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeViewExchange {
    /// Session ID.
    pub session_id: u64,
    /// Node sending the view.
    pub sender: NodeId,
    /// Membership view summary.
    pub view: crate::partition_merge::MembershipViewSummary,
}

/// Request data merge information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMergeRequest {
    /// Session ID.
    pub session_id: u64,
    /// Requesting node.
    pub requester: NodeId,
    /// Last committed Raft index on requester.
    pub last_committed_index: u64,
    /// Last committed Raft term on requester.
    pub last_committed_term: u64,
    /// Keys or key patterns to reconcile (empty = all).
    pub key_patterns: Vec<String>,
}

/// Response with data merge information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataMergeResponse {
    /// Session ID.
    pub session_id: u64,
    /// Responding node.
    pub responder: NodeId,
    /// Delta entries since common ancestor.
    pub delta_entries: Vec<MergeDeltaEntry>,
    /// State embedding for semantic comparison.
    pub state_embedding: Option<SparseVector>,
    /// Whether more data is available (pagination).
    pub has_more: bool,
}

/// A single delta entry for merge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeDeltaEntry {
    /// Key that was modified.
    pub key: String,
    /// Raft log index of the modification.
    pub log_index: u64,
    /// Raft log term of the modification.
    pub log_term: u64,
    /// Type of operation (put, delete).
    pub op_type: MergeOpType,
    /// Data hash for conflict detection.
    pub data_hash: [u8; 32],
}

/// Type of merge operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeOpType {
    Put,
    Delete,
    Update,
}

/// Request transaction reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxReconcileRequest {
    /// Session ID.
    pub session_id: u64,
    /// Requesting node.
    pub requester: NodeId,
    /// Pending transactions on requester.
    pub pending_txs: Vec<crate::partition_merge::PendingTxState>,
}

/// Response with transaction reconciliation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxReconcileResponse {
    /// Session ID.
    pub session_id: u64,
    /// Responding node.
    pub responder: NodeId,
    /// Pending transactions on responder.
    pub pending_txs: Vec<crate::partition_merge::PendingTxState>,
    /// Transactions to commit (agreed by both).
    pub to_commit: Vec<u64>,
    /// Transactions to abort (timed out or conflicted).
    pub to_abort: Vec<u64>,
}

/// Finalize merge session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeFinalize {
    /// Session ID.
    pub session_id: u64,
    /// Node sending finalization.
    pub sender: NodeId,
    /// Whether the merge completed successfully.
    pub success: bool,
    /// Final merged state hash.
    pub final_state_hash: [u8; 32],
    /// Number of conflicts resolved.
    pub conflicts_resolved: u32,
    /// Merge duration in milliseconds.
    pub duration_ms: u64,
}

/// Transport trait for network communication.
#[async_trait]
pub trait Transport: Send + Sync {
    async fn send(&self, to: &NodeId, msg: Message) -> Result<()>;

    async fn broadcast(&self, msg: Message) -> Result<()>;

    async fn recv(&self) -> Result<(NodeId, Message)>;

    async fn connect(&self, peer: &PeerConfig) -> Result<()>;

    async fn disconnect(&self, peer_id: &NodeId) -> Result<()>;

    fn peers(&self) -> Vec<NodeId>;

    fn local_id(&self) -> &NodeId;
}

/// Extended transport trait with geometric routing capabilities.
///
/// Adds methods for sending messages based on embedding similarity,
/// enabling geometric-aware message routing in distributed systems.
#[allow(async_fn_in_trait)]
pub trait GeometricTransport: Transport {
    /// Send a message to the geometrically nearest peer based on embedding similarity.
    ///
    /// Finds the peer whose cached embedding is most similar to the provided
    /// embedding and sends the message to that peer.
    async fn send_to_nearest(&self, embedding: &SparseVector, msg: Message) -> Result<()>;

    /// Broadcast a message to all peers within a geometric region.
    ///
    /// Sends the message only to peers whose embeddings have similarity
    /// above the threshold to the provided region centroid.
    async fn broadcast_to_region(
        &self,
        region_centroid: &SparseVector,
        similarity_threshold: f32,
        msg: Message,
    ) -> Result<()>;
}

/// In-memory transport for testing with partition injection support.
pub struct MemoryTransport {
    /// Local node ID.
    local_id: NodeId,
    /// Channel for receiving messages.
    receiver: RwLock<mpsc::Receiver<(NodeId, Message)>>,
    /// Senders to other nodes.
    peers: RwLock<HashMap<NodeId, mpsc::Sender<(NodeId, Message)>>>,
    /// Sender for this node (given to peers).
    local_sender: mpsc::Sender<(NodeId, Message)>,
    /// Partitioned peers - messages to/from these nodes are dropped.
    partitioned: RwLock<HashSet<NodeId>>,
    /// Message drop counter for testing assertions.
    dropped_messages: AtomicU64,
    /// Simulated network latency (min, max) in milliseconds.
    latency_ms: RwLock<Option<(u64, u64)>>,
    /// Per-peer latency overrides.
    peer_latency_ms: RwLock<HashMap<NodeId, (u64, u64)>>,
}

impl MemoryTransport {
    pub fn new(local_id: NodeId) -> Self {
        let (tx, rx) = mpsc::channel(1000);
        Self {
            local_id,
            receiver: RwLock::new(rx),
            peers: RwLock::new(HashMap::new()),
            local_sender: tx,
            partitioned: RwLock::new(HashSet::new()),
            dropped_messages: AtomicU64::new(0),
            latency_ms: RwLock::new(None),
            peer_latency_ms: RwLock::new(HashMap::new()),
        }
    }

    /// Set global simulated latency range (min_ms, max_ms).
    ///
    /// Each message will be delayed by a random duration in this range.
    /// Use `None` to disable latency simulation.
    pub fn set_latency(&self, latency: Option<(u64, u64)>) {
        *self.latency_ms.write() = latency;
    }

    /// Set latency for a specific peer (overrides global latency).
    pub fn set_peer_latency(&self, peer_id: &NodeId, latency: (u64, u64)) {
        self.peer_latency_ms
            .write()
            .insert(peer_id.clone(), latency);
    }

    /// Clear peer-specific latency (will use global latency if set).
    pub fn clear_peer_latency(&self, peer_id: &NodeId) {
        self.peer_latency_ms.write().remove(peer_id);
    }

    fn get_latency_for_peer(&self, peer_id: &NodeId) -> Option<(u64, u64)> {
        // Check peer-specific latency first
        if let Some(latency) = self.peer_latency_ms.read().get(peer_id) {
            return Some(*latency);
        }
        // Fall back to global latency
        *self.latency_ms.read()
    }

    /// Apply simulated latency delay.
    async fn apply_latency(&self, peer_id: &NodeId) {
        if let Some((min_ms, max_ms)) = self.get_latency_for_peer(peer_id) {
            let delay_ms = if min_ms == max_ms {
                min_ms
            } else {
                use std::time::{SystemTime, UNIX_EPOCH};
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                min_ms + (seed % (max_ms - min_ms + 1))
            };
            if delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
        }
    }

    pub fn sender(&self) -> mpsc::Sender<(NodeId, Message)> {
        self.local_sender.clone()
    }

    pub fn connect_to(&self, other_id: NodeId, sender: mpsc::Sender<(NodeId, Message)>) {
        tracing::debug!(
            local_id = %self.local_id,
            peer_id = %other_id,
            "Peer connected"
        );
        self.peers.write().insert(other_id, sender);
    }

    /// Simulate network partition - messages to this peer will be dropped.
    pub fn partition(&self, peer_id: &NodeId) {
        tracing::info!(
            local_id = %self.local_id,
            peer_id = %peer_id,
            "Network partition simulated"
        );
        self.partitioned.write().insert(peer_id.clone());
    }

    /// Heal partition - restore communication with peer.
    pub fn heal(&self, peer_id: &NodeId) {
        tracing::info!(
            local_id = %self.local_id,
            peer_id = %peer_id,
            "Network partition healed"
        );
        self.partitioned.write().remove(peer_id);
    }

    pub fn is_partitioned(&self, peer_id: &NodeId) -> bool {
        self.partitioned.read().contains(peer_id)
    }

    pub fn dropped_message_count(&self) -> u64 {
        self.dropped_messages.load(Ordering::Relaxed)
    }

    pub fn partitioned_peers(&self) -> Vec<NodeId> {
        self.partitioned.read().iter().cloned().collect()
    }

    /// Partition from all connected peers (total isolation).
    pub fn partition_all(&self) {
        let peer_ids: Vec<_> = self.peers.read().keys().cloned().collect();
        let mut partitioned = self.partitioned.write();
        for peer_id in peer_ids {
            partitioned.insert(peer_id);
        }
    }

    pub fn heal_all(&self) {
        self.partitioned.write().clear();
    }
}

#[async_trait]
impl Transport for MemoryTransport {
    async fn send(&self, to: &NodeId, msg: Message) -> Result<()> {
        // Check if peer is partitioned - silently drop message
        if self.partitioned.read().contains(to) {
            self.dropped_messages.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                from = %self.local_id,
                to = %to,
                "Message dropped: network partition"
            );
            return Err(ChainError::NetworkError(format!(
                "network partition: {} -> {}",
                self.local_id, to
            )));
        }

        // Apply simulated latency before sending
        self.apply_latency(to).await;

        let sender = {
            let peers = self.peers.read();
            match peers.get(to).cloned() {
                Some(s) => s,
                None => {
                    tracing::debug!(
                        from = %self.local_id,
                        to = %to,
                        "Send failed: peer not found"
                    );
                    return Err(ChainError::NetworkError(format!("peer not found: {}", to)));
                }
            }
        };

        sender
            .send((self.local_id.clone(), msg))
            .await
            .map_err(|e| ChainError::NetworkError(e.to_string()))?;

        Ok(())
    }

    async fn broadcast(&self, msg: Message) -> Result<()> {
        let partitioned = self.partitioned.read().clone();
        let senders: Vec<_> = {
            let peers = self.peers.read();
            peers
                .iter()
                .filter(|(id, _)| !partitioned.contains(*id))
                .map(|(_, sender)| sender.clone())
                .collect()
        };

        for sender in senders {
            let _ = sender.send((self.local_id.clone(), msg.clone())).await;
        }
        Ok(())
    }

    async fn recv(&self) -> Result<(NodeId, Message)> {
        // Try to receive, sleeping briefly if no message
        loop {
            let msg = {
                let mut receiver = self.receiver.write();
                receiver.try_recv().ok()
            };

            if let Some(msg) = msg {
                return Ok(msg);
            }

            // Sleep briefly to avoid busy-waiting and allow shutdown signal
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        }
    }

    async fn connect(&self, peer: &PeerConfig) -> Result<()> {
        // For memory transport, connection is handled externally
        // This is a no-op - use connect_to directly
        let _ = peer;
        Ok(())
    }

    async fn disconnect(&self, peer_id: &NodeId) -> Result<()> {
        tracing::debug!(
            local_id = %self.local_id,
            peer_id = %peer_id,
            "Peer disconnected"
        );
        self.peers.write().remove(peer_id);
        Ok(())
    }

    fn peers(&self) -> Vec<NodeId> {
        self.peers.read().keys().cloned().collect()
    }

    fn local_id(&self) -> &NodeId {
        &self.local_id
    }
}

/// Network manager for handling multiple transports.
pub struct NetworkManager {
    /// Transport implementation.
    transport: Arc<dyn Transport>,
    /// Message handlers.
    handlers: RwLock<Vec<Box<dyn MessageHandler>>>,
}

/// Trait for handling incoming messages.
pub trait MessageHandler: Send + Sync {
    fn handle(&self, from: &NodeId, msg: &Message) -> Option<Message>;
}

impl NetworkManager {
    pub fn new(transport: Arc<dyn Transport>) -> Self {
        Self {
            transport,
            handlers: RwLock::new(Vec::new()),
        }
    }

    pub fn add_handler(&self, handler: Box<dyn MessageHandler>) {
        self.handlers.write().push(handler);
    }

    pub fn transport(&self) -> &Arc<dyn Transport> {
        &self.transport
    }

    pub async fn process_messages(&self) -> Result<()> {
        loop {
            let (from, msg) = self.transport.recv().await?;

            tracing::debug!(
                from = %from,
                msg_type = msg.type_name(),
                "Message received"
            );

            // Collect responses without holding lock across await
            let responses: Vec<(NodeId, Message)> = {
                let handlers = self.handlers.read();
                handlers
                    .iter()
                    .filter_map(|handler| handler.handle(&from, &msg).map(|r| (from.clone(), r)))
                    .collect()
            };

            // Send responses after releasing lock
            for (to, response) in responses {
                self.transport.send(&to, response).await?;
            }
        }
    }
}

/// Trait for executing queries locally.
///
/// Implemented by QueryRouter to enable distributed query execution.
/// This trait lives in tensor_chain to avoid circular dependencies.
pub trait QueryExecutor: Send + Sync {
    /// Execute a query and return serialized result.
    fn execute(&self, query: &str) -> std::result::Result<Vec<u8>, String>;
}

/// Handler for distributed query messages.
///
/// Bridges network messages (QueryRequest, QueryResponse) to a QueryExecutor.
pub struct QueryHandler {
    executor: Arc<dyn QueryExecutor>,
    local_shard_id: usize,
}

impl QueryHandler {
    pub fn new(executor: Arc<dyn QueryExecutor>, local_shard_id: usize) -> Self {
        Self {
            executor,
            local_shard_id,
        }
    }
}

impl MessageHandler for QueryHandler {
    fn handle(&self, _from: &NodeId, msg: &Message) -> Option<Message> {
        match msg {
            Message::QueryRequest(request) => {
                tracing::debug!(
                    query_id = request.query_id,
                    shard_id = request.shard_id,
                    "Executing remote query"
                );

                let start = std::time::Instant::now();

                let (result, success, error) = match self.executor.execute(&request.query) {
                    Ok(bytes) => (bytes, true, None),
                    Err(e) => {
                        tracing::warn!(
                            query_id = request.query_id,
                            shard_id = request.shard_id,
                            error = %e,
                            "Remote query failed"
                        );
                        (Vec::new(), false, Some(e))
                    }
                };

                let execution_time_us = start.elapsed().as_micros() as u64;

                Some(Message::QueryResponse(QueryResponse {
                    query_id: request.query_id,
                    shard_id: self.local_shard_id,
                    result,
                    execution_time_us,
                    success,
                    error,
                }))
            },
            _ => None,
        }
    }
}

/// Handler for distributed transaction messages.
///
/// Bridges network messages (TxPrepare, TxCommit, TxAbort) to TxParticipant.
pub struct TxHandler {
    participant: Arc<crate::distributed_tx::TxParticipant>,
}

impl TxHandler {
    pub fn new(participant: Arc<crate::distributed_tx::TxParticipant>) -> Self {
        Self { participant }
    }
}

impl MessageHandler for TxHandler {
    fn handle(&self, from: &NodeId, msg: &Message) -> Option<Message> {
        match msg {
            Message::TxPrepare(prepare) => {
                tracing::debug!(
                    tx_id = prepare.tx_id,
                    shard_id = prepare.shard_id,
                    coordinator = %from,
                    "Handling TxPrepare"
                );

                // Convert TxPrepareMsg to PrepareRequest
                let request = crate::distributed_tx::PrepareRequest {
                    tx_id: prepare.tx_id,
                    coordinator: from.clone(),
                    operations: prepare.operations.clone(),
                    delta_embedding: prepare.delta_embedding.clone(),
                    timeout_ms: prepare.timeout_ms,
                };

                let vote = self.participant.prepare(request);

                // Convert PrepareVote to TxVote
                let tx_vote = match vote {
                    crate::distributed_tx::PrepareVote::Yes { lock_handle, delta } => TxVote::Yes {
                        lock_handle,
                        delta: delta.delta.clone(),
                        affected_keys: delta.affected_keys.into_iter().collect(),
                    },
                    crate::distributed_tx::PrepareVote::No { reason } => TxVote::No { reason },
                    crate::distributed_tx::PrepareVote::Conflict {
                        similarity,
                        conflicting_tx,
                    } => TxVote::Conflict {
                        similarity,
                        conflicting_tx,
                    },
                };

                Some(Message::TxPrepareResponse(TxPrepareResponseMsg {
                    tx_id: prepare.tx_id,
                    shard_id: prepare.shard_id,
                    vote: tx_vote,
                }))
            },
            Message::TxCommit(commit) => {
                tracing::debug!(
                    tx_id = commit.tx_id,
                    "Handling TxCommit"
                );

                let response = self.participant.commit(commit.tx_id);

                Some(Message::TxAck(TxAckMsg {
                    tx_id: commit.tx_id,
                    shard_id: 0, // Participant doesn't track its shard ID
                    success: response.success,
                    error: response.error,
                }))
            },
            Message::TxAbort(abort) => {
                tracing::debug!(
                    tx_id = abort.tx_id,
                    reason = %abort.reason,
                    "Handling TxAbort"
                );

                let response = self.participant.abort(abort.tx_id);

                Some(Message::TxAck(TxAckMsg {
                    tx_id: abort.tx_id,
                    shard_id: 0,
                    success: response.success,
                    error: response.error,
                }))
            },
            _ => None, // Not a 2PC message, pass to other handlers
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_transport_creation() {
        let node1 = MemoryTransport::new("node1".to_string());
        let node2 = MemoryTransport::new("node2".to_string());

        // Connect nodes
        node1.connect_to("node2".to_string(), node2.sender());
        node2.connect_to("node1".to_string(), node1.sender());

        assert_eq!(node1.peers().len(), 1);
        assert_eq!(node1.local_id(), "node1");
    }

    #[test]
    fn test_memory_transport_peers() {
        let node1 = MemoryTransport::new("node1".to_string());
        let node2 = MemoryTransport::new("node2".to_string());

        assert!(node1.peers().is_empty());

        node1.connect_to("node2".to_string(), node2.sender());
        assert_eq!(node1.peers().len(), 1);
        assert!(node1.peers().contains(&"node2".to_string()));
    }

    #[test]
    fn test_message_serialization() {
        let msg = Message::RequestVote(RequestVote {
            term: 1,
            candidate_id: "candidate".to_string(),
            last_log_index: 10,
            last_log_term: 1,
            state_embedding: SparseVector::from_dense(&[0.1, 0.2, 0.3]),
        });

        let bytes = bincode::serialize(&msg).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::RequestVote(rv) = decoded {
            assert_eq!(rv.term, 1);
            assert_eq!(rv.candidate_id, "candidate");
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_log_entry_serialization() {
        use crate::block::BlockHeader;

        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "proposer".to_string());
        let block = crate::block::Block::new(header, vec![]);

        let entry = LogEntry::new(1, 1, block);

        let bytes = bincode::serialize(&entry).unwrap();
        let decoded: LogEntry = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.term, 1);
        assert_eq!(decoded.index, 1);
    }

    #[test]
    fn test_peer_config() {
        let config = PeerConfig {
            node_id: "peer1".to_string(),
            address: "127.0.0.1:9000".to_string(),
        };
        assert_eq!(config.node_id, "peer1");
        assert_eq!(config.address, "127.0.0.1:9000");

        // Test serialization
        let bytes = bincode::serialize(&config).unwrap();
        let decoded: PeerConfig = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.node_id, "peer1");
    }

    #[test]
    fn test_request_vote_response_serialization() {
        let msg = Message::RequestVoteResponse(RequestVoteResponse {
            term: 5,
            vote_granted: true,
            voter_id: "voter1".to_string(),
        });

        let bytes = bincode::serialize(&msg).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::RequestVoteResponse(rvr) = decoded {
            assert_eq!(rvr.term, 5);
            assert!(rvr.vote_granted);
            assert_eq!(rvr.voter_id, "voter1");
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_append_entries_serialization() {
        let msg = Message::AppendEntries(AppendEntries {
            term: 3,
            leader_id: "leader1".to_string(),
            prev_log_index: 10,
            prev_log_term: 2,
            entries: vec![],
            leader_commit: 8,
            block_embedding: Some(SparseVector::from_dense(&[0.1, 0.2, 0.3])),
        });

        let bytes = bincode::serialize(&msg).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::AppendEntries(ae) = decoded {
            assert_eq!(ae.term, 3);
            assert_eq!(ae.leader_id, "leader1");
            assert!(ae.block_embedding.is_some());
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_append_entries_response_serialization() {
        let msg = Message::AppendEntriesResponse(AppendEntriesResponse {
            term: 3,
            success: true,
            follower_id: "follower1".to_string(),
            match_index: 15,
            used_fast_path: true,
        });

        let bytes = bincode::serialize(&msg).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::AppendEntriesResponse(aer) = decoded {
            assert_eq!(aer.term, 3);
            assert!(aer.success);
            assert!(aer.used_fast_path);
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_block_request_response_serialization() {
        let req = Message::BlockRequest(BlockRequest {
            from_height: 10,
            to_height: 20,
            requester_id: "requester".to_string(),
        });

        let bytes = bincode::serialize(&req).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::BlockRequest(br) = decoded {
            assert_eq!(br.from_height, 10);
            assert_eq!(br.to_height, 20);
        } else {
            panic!("wrong message type");
        }

        let resp = Message::BlockResponse(BlockResponse {
            blocks: vec![],
            current_height: 25,
        });

        let bytes = bincode::serialize(&resp).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::BlockResponse(br) = decoded {
            assert!(br.blocks.is_empty());
            assert_eq!(br.current_height, 25);
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_snapshot_request_response_serialization() {
        let req = Message::SnapshotRequest(SnapshotRequest {
            requester_id: "node1".to_string(),
            offset: 1000,
            chunk_size: 4096,
        });

        let bytes = bincode::serialize(&req).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::SnapshotRequest(sr) = decoded {
            assert_eq!(sr.offset, 1000);
            assert_eq!(sr.chunk_size, 4096);
        } else {
            panic!("wrong message type");
        }

        let resp = Message::SnapshotResponse(SnapshotResponse {
            snapshot_height: 100,
            snapshot_hash: [1u8; 32],
            data: vec![1, 2, 3, 4],
            offset: 0,
            total_size: 4096,
            is_last: false,
        });

        let bytes = bincode::serialize(&resp).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::SnapshotResponse(sr) = decoded {
            assert_eq!(sr.snapshot_height, 100);
            assert!(!sr.is_last);
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_ping_pong_serialization() {
        let ping = Message::Ping { term: 42 };
        let bytes = bincode::serialize(&ping).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();
        if let Message::Ping { term } = decoded {
            assert_eq!(term, 42);
        } else {
            panic!("wrong message type");
        }

        let pong = Message::Pong { term: 42 };
        let bytes = bincode::serialize(&pong).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();
        if let Message::Pong { term } = decoded {
            assert_eq!(term, 42);
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_tx_prepare_msg_serialization() {
        use crate::block::Transaction;

        let msg = Message::TxPrepare(TxPrepareMsg {
            tx_id: 123,
            coordinator: "coord".to_string(),
            shard_id: 1,
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1, 2, 3],
            }],
            delta_embedding: SparseVector::from_dense(&[0.5, 0.5]),
            timeout_ms: 5000,
        });

        let bytes = bincode::serialize(&msg).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::TxPrepare(tp) = decoded {
            assert_eq!(tp.tx_id, 123);
            assert_eq!(tp.shard_id, 1);
            assert_eq!(tp.operations.len(), 1);
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_tx_prepare_response_serialization() {
        let msg = Message::TxPrepareResponse(TxPrepareResponseMsg {
            tx_id: 123,
            shard_id: 1,
            vote: TxVote::Yes {
                lock_handle: 456,
                delta: SparseVector::from_dense(&[0.1, 0.2]),
                affected_keys: vec!["key1".to_string()],
            },
        });

        let bytes = bincode::serialize(&msg).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();

        if let Message::TxPrepareResponse(tpr) = decoded {
            assert_eq!(tpr.tx_id, 123);
            if let TxVote::Yes { lock_handle, .. } = tpr.vote {
                assert_eq!(lock_handle, 456);
            } else {
                panic!("wrong vote type");
            }
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_tx_vote_variants_serialization() {
        let yes = TxVote::Yes {
            lock_handle: 1,
            delta: SparseVector::from_dense(&[0.1]),
            affected_keys: vec!["k1".to_string()],
        };
        let bytes = bincode::serialize(&yes).unwrap();
        let decoded: TxVote = bincode::deserialize(&bytes).unwrap();
        assert!(matches!(decoded, TxVote::Yes { .. }));

        let no = TxVote::No {
            reason: "test reason".to_string(),
        };
        let bytes = bincode::serialize(&no).unwrap();
        let decoded: TxVote = bincode::deserialize(&bytes).unwrap();
        if let TxVote::No { reason } = decoded {
            assert_eq!(reason, "test reason");
        } else {
            panic!("wrong vote type");
        }

        let conflict = TxVote::Conflict {
            similarity: 0.95,
            conflicting_tx: 999,
        };
        let bytes = bincode::serialize(&conflict).unwrap();
        let decoded: TxVote = bincode::deserialize(&bytes).unwrap();
        if let TxVote::Conflict {
            similarity,
            conflicting_tx,
        } = decoded
        {
            assert!((similarity - 0.95).abs() < 0.01);
            assert_eq!(conflicting_tx, 999);
        } else {
            panic!("wrong vote type");
        }
    }

    #[test]
    fn test_tx_commit_abort_ack_serialization() {
        let commit = Message::TxCommit(TxCommitMsg {
            tx_id: 100,
            shards: vec![0, 1, 2],
        });
        let bytes = bincode::serialize(&commit).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();
        if let Message::TxCommit(tc) = decoded {
            assert_eq!(tc.tx_id, 100);
            assert_eq!(tc.shards, vec![0, 1, 2]);
        } else {
            panic!("wrong message type");
        }

        let abort = Message::TxAbort(TxAbortMsg {
            tx_id: 100,
            reason: "conflict".to_string(),
            shards: vec![0, 1],
        });
        let bytes = bincode::serialize(&abort).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();
        if let Message::TxAbort(ta) = decoded {
            assert_eq!(ta.tx_id, 100);
            assert_eq!(ta.reason, "conflict");
        } else {
            panic!("wrong message type");
        }

        let ack = Message::TxAck(TxAckMsg {
            tx_id: 100,
            shard_id: 1,
            success: true,
            error: None,
        });
        let bytes = bincode::serialize(&ack).unwrap();
        let decoded: Message = bincode::deserialize(&bytes).unwrap();
        if let Message::TxAck(ta) = decoded {
            assert_eq!(ta.tx_id, 100);
            assert!(ta.success);
            assert!(ta.error.is_none());
        } else {
            panic!("wrong message type");
        }
    }

    #[test]
    fn test_tx_ack_with_error() {
        let ack = TxAckMsg {
            tx_id: 200,
            shard_id: 2,
            success: false,
            error: Some("transaction not found".to_string()),
        };

        let bytes = bincode::serialize(&ack).unwrap();
        let decoded: TxAckMsg = bincode::deserialize(&bytes).unwrap();
        assert!(!decoded.success);
        assert_eq!(decoded.error, Some("transaction not found".to_string()));
    }

    #[tokio::test]
    async fn test_memory_transport_send_recv() {
        let node1 = MemoryTransport::new("node1".to_string());
        let node2 = MemoryTransport::new("node2".to_string());

        node1.connect_to("node2".to_string(), node2.sender());

        // Send from node1 to node2
        node1
            .send(&"node2".to_string(), Message::Ping { term: 1 })
            .await
            .unwrap();

        // Receive on node2
        let (from, msg) = node2.recv().await.unwrap();
        assert_eq!(from, "node1");
        assert!(matches!(msg, Message::Ping { term: 1 }));
    }

    #[tokio::test]
    async fn test_memory_transport_send_unknown_peer() {
        let node1 = MemoryTransport::new("node1".to_string());

        let result = node1
            .send(&"unknown".to_string(), Message::Ping { term: 1 })
            .await;
        assert!(result.is_err());
    }

    // NOTE: This test is disabled because recv() now blocks indefinitely
    // when no messages are available (required for the async run loop).
    // Use test_memory_transport_send_recv for recv testing.
    #[tokio::test]
    #[ignore = "recv() now blocks until message received"]
    async fn test_memory_transport_recv_empty() {
        let node1 = MemoryTransport::new("node1".to_string());

        // This will now block forever since recv loops until message
        let result = tokio::time::timeout(std::time::Duration::from_millis(10), node1.recv()).await;
        assert!(result.is_err()); // Should timeout
    }

    #[tokio::test]
    async fn test_memory_transport_broadcast() {
        let node1 = MemoryTransport::new("node1".to_string());
        let node2 = MemoryTransport::new("node2".to_string());
        let node3 = MemoryTransport::new("node3".to_string());

        node1.connect_to("node2".to_string(), node2.sender());
        node1.connect_to("node3".to_string(), node3.sender());

        node1.broadcast(Message::Ping { term: 10 }).await.unwrap();

        // Both should receive
        let (from2, _) = node2.recv().await.unwrap();
        let (from3, _) = node3.recv().await.unwrap();
        assert_eq!(from2, "node1");
        assert_eq!(from3, "node1");
    }

    #[tokio::test]
    async fn test_memory_transport_connect_disconnect() {
        let node1 = MemoryTransport::new("node1".to_string());
        let node2 = MemoryTransport::new("node2".to_string());

        node1.connect_to("node2".to_string(), node2.sender());
        assert_eq!(node1.peers().len(), 1);

        node1.disconnect(&"node2".to_string()).await.unwrap();
        assert!(node1.peers().is_empty());
    }

    #[tokio::test]
    async fn test_memory_transport_connect_noop() {
        let node1 = MemoryTransport::new("node1".to_string());

        let peer = PeerConfig {
            node_id: "node2".to_string(),
            address: "127.0.0.1:9000".to_string(),
        };

        // Connect is a no-op for MemoryTransport
        let result = node1.connect(&peer).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_network_manager_creation() {
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager = NetworkManager::new(transport.clone());

        assert_eq!(manager.transport().local_id(), "node1");
    }

    #[test]
    fn test_network_manager_add_handler() {
        struct TestHandler;
        impl MessageHandler for TestHandler {
            fn handle(&self, _from: &NodeId, msg: &Message) -> Option<Message> {
                if let Message::Ping { term } = msg {
                    Some(Message::Pong { term: *term })
                } else {
                    None
                }
            }
        }

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager = NetworkManager::new(transport);

        manager.add_handler(Box::new(TestHandler));

        // Handler should be registered
        assert_eq!(manager.handlers.read().len(), 1);
    }

    #[test]
    fn test_message_debug_format() {
        let msg = Message::Ping { term: 42 };
        let debug = format!("{:?}", msg);
        assert!(debug.contains("Ping"));
        assert!(debug.contains("42"));
    }

    #[test]
    fn test_message_clone() {
        let msg = Message::RequestVote(RequestVote {
            term: 1,
            candidate_id: "c1".to_string(),
            last_log_index: 5,
            last_log_term: 1,
            state_embedding: SparseVector::from_dense(&[0.1, 0.2]),
        });

        let cloned = msg.clone();
        if let Message::RequestVote(rv) = cloned {
            assert_eq!(rv.term, 1);
            assert_eq!(rv.candidate_id, "c1");
        }
    }

    #[test]
    fn test_peer_config_debug_clone() {
        let config = PeerConfig {
            node_id: "node1".to_string(),
            address: "127.0.0.1:8080".to_string(),
        };
        let cloned = config.clone();
        assert_eq!(config.node_id, cloned.node_id);

        let debug = format!("{:?}", config);
        assert!(debug.contains("PeerConfig"));
    }

    #[test]
    fn test_request_vote_debug_clone() {
        let rv = RequestVote {
            term: 5,
            candidate_id: "cand".to_string(),
            last_log_index: 10,
            last_log_term: 4,
            state_embedding: SparseVector::from_dense(&[0.1, 0.2, 0.3]),
        };
        let cloned = rv.clone();
        assert_eq!(rv.term, cloned.term);

        let debug = format!("{:?}", rv);
        assert!(debug.contains("RequestVote"));
    }

    #[test]
    fn test_request_vote_response_debug_clone() {
        let rvr = RequestVoteResponse {
            term: 5,
            vote_granted: true,
            voter_id: "voter".to_string(),
        };
        let cloned = rvr.clone();
        assert_eq!(rvr.vote_granted, cloned.vote_granted);

        let debug = format!("{:?}", rvr);
        assert!(debug.contains("RequestVoteResponse"));
    }

    #[test]
    fn test_append_entries_debug_clone() {
        let ae = AppendEntries {
            term: 3,
            leader_id: "leader".to_string(),
            prev_log_index: 10,
            prev_log_term: 2,
            entries: vec![],
            leader_commit: 8,
            block_embedding: None,
        };
        let cloned = ae.clone();
        assert_eq!(ae.term, cloned.term);

        let debug = format!("{:?}", ae);
        assert!(debug.contains("AppendEntries"));
    }

    #[test]
    fn test_append_entries_response_debug_clone() {
        let aer = AppendEntriesResponse {
            term: 3,
            success: true,
            follower_id: "follower".to_string(),
            match_index: 15,
            used_fast_path: false,
        };
        let cloned = aer.clone();
        assert_eq!(aer.match_index, cloned.match_index);

        let debug = format!("{:?}", aer);
        assert!(debug.contains("AppendEntriesResponse"));
    }

    #[test]
    fn test_log_entry_debug_clone() {
        use crate::block::{Block, BlockHeader};

        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "p".to_string());
        let block = Block::new(header, vec![]);

        let entry = LogEntry::new(1, 1, block);
        let cloned = entry.clone();
        assert_eq!(entry.term, cloned.term);

        let debug = format!("{:?}", entry);
        assert!(debug.contains("LogEntry"));
    }

    #[test]
    fn test_block_request_debug_clone() {
        let br = BlockRequest {
            from_height: 10,
            to_height: 20,
            requester_id: "node1".to_string(),
        };
        let cloned = br.clone();
        assert_eq!(br.from_height, cloned.from_height);

        let debug = format!("{:?}", br);
        assert!(debug.contains("BlockRequest"));
    }

    #[test]
    fn test_block_response_debug_clone() {
        let br = BlockResponse {
            blocks: vec![],
            current_height: 100,
        };
        let cloned = br.clone();
        assert_eq!(br.current_height, cloned.current_height);

        let debug = format!("{:?}", br);
        assert!(debug.contains("BlockResponse"));
    }

    #[test]
    fn test_snapshot_request_debug_clone() {
        let sr = SnapshotRequest {
            requester_id: "node1".to_string(),
            offset: 1000,
            chunk_size: 4096,
        };
        let cloned = sr.clone();
        assert_eq!(sr.offset, cloned.offset);

        let debug = format!("{:?}", sr);
        assert!(debug.contains("SnapshotRequest"));
    }

    #[test]
    fn test_snapshot_response_debug_clone() {
        let sr = SnapshotResponse {
            snapshot_height: 100,
            snapshot_hash: [1u8; 32],
            data: vec![1, 2, 3],
            offset: 0,
            total_size: 1000,
            is_last: false,
        };
        let cloned = sr.clone();
        assert_eq!(sr.snapshot_height, cloned.snapshot_height);

        let debug = format!("{:?}", sr);
        assert!(debug.contains("SnapshotResponse"));
    }

    #[test]
    fn test_tx_prepare_msg_debug_clone() {
        let msg = TxPrepareMsg {
            tx_id: 123,
            coordinator: "coord".to_string(),
            shard_id: 1,
            operations: vec![],
            delta_embedding: SparseVector::from_dense(&[0.1]),
            timeout_ms: 5000,
        };
        let cloned = msg.clone();
        assert_eq!(msg.tx_id, cloned.tx_id);

        let debug = format!("{:?}", msg);
        assert!(debug.contains("TxPrepareMsg"));
    }

    #[test]
    fn test_tx_prepare_response_msg_debug_clone() {
        let msg = TxPrepareResponseMsg {
            tx_id: 123,
            shard_id: 1,
            vote: TxVote::No {
                reason: "test".to_string(),
            },
        };
        let cloned = msg.clone();
        assert_eq!(msg.tx_id, cloned.tx_id);

        let debug = format!("{:?}", msg);
        assert!(debug.contains("TxPrepareResponseMsg"));
    }

    #[test]
    fn test_tx_vote_debug_clone() {
        let yes = TxVote::Yes {
            lock_handle: 1,
            delta: SparseVector::from_dense(&[0.1]),
            affected_keys: vec!["k1".to_string()],
        };
        let cloned = yes.clone();
        if let (
            TxVote::Yes {
                lock_handle: l1, ..
            },
            TxVote::Yes {
                lock_handle: l2, ..
            },
        ) = (&yes, &cloned)
        {
            assert_eq!(l1, l2);
        }

        let debug = format!("{:?}", yes);
        assert!(debug.contains("Yes"));
    }

    #[test]
    fn test_tx_commit_msg_debug_clone() {
        let msg = TxCommitMsg {
            tx_id: 100,
            shards: vec![0, 1, 2],
        };
        let cloned = msg.clone();
        assert_eq!(msg.tx_id, cloned.tx_id);

        let debug = format!("{:?}", msg);
        assert!(debug.contains("TxCommitMsg"));
    }

    #[test]
    fn test_tx_abort_msg_debug_clone() {
        let msg = TxAbortMsg {
            tx_id: 100,
            reason: "conflict".to_string(),
            shards: vec![0, 1],
        };
        let cloned = msg.clone();
        assert_eq!(msg.reason, cloned.reason);

        let debug = format!("{:?}", msg);
        assert!(debug.contains("TxAbortMsg"));
    }

    #[test]
    fn test_tx_ack_msg_debug_clone() {
        let msg = TxAckMsg {
            tx_id: 100,
            shard_id: 1,
            success: true,
            error: None,
        };
        let cloned = msg.clone();
        assert_eq!(msg.success, cloned.success);

        let debug = format!("{:?}", msg);
        assert!(debug.contains("TxAckMsg"));
    }

    #[test]
    fn test_all_message_variants_debug() {
        let messages = vec![
            Message::Ping { term: 1 },
            Message::Pong { term: 1 },
            Message::RequestVote(RequestVote {
                term: 1,
                candidate_id: "c".to_string(),
                last_log_index: 0,
                last_log_term: 0,
                state_embedding: SparseVector::new(0),
            }),
            Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                vote_granted: true,
                voter_id: "v".to_string(),
            }),
            Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: "l".to_string(),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![],
                leader_commit: 0,
                block_embedding: None,
            }),
            Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 1,
                success: true,
                follower_id: "f".to_string(),
                match_index: 0,
                used_fast_path: false,
            }),
            Message::BlockRequest(BlockRequest {
                from_height: 0,
                to_height: 1,
                requester_id: "r".to_string(),
            }),
            Message::BlockResponse(BlockResponse {
                blocks: vec![],
                current_height: 0,
            }),
            Message::SnapshotRequest(SnapshotRequest {
                requester_id: "r".to_string(),
                offset: 0,
                chunk_size: 1,
            }),
            Message::SnapshotResponse(SnapshotResponse {
                snapshot_height: 0,
                snapshot_hash: [0u8; 32],
                data: vec![],
                offset: 0,
                total_size: 0,
                is_last: true,
            }),
            Message::TxPrepare(TxPrepareMsg {
                tx_id: 1,
                coordinator: "c".to_string(),
                shard_id: 0,
                operations: vec![],
                delta_embedding: SparseVector::new(0),
                timeout_ms: 1000,
            }),
            Message::TxPrepareResponse(TxPrepareResponseMsg {
                tx_id: 1,
                shard_id: 0,
                vote: TxVote::No {
                    reason: "n".to_string(),
                },
            }),
            Message::TxCommit(TxCommitMsg {
                tx_id: 1,
                shards: vec![],
            }),
            Message::TxAbort(TxAbortMsg {
                tx_id: 1,
                reason: "r".to_string(),
                shards: vec![],
            }),
            Message::TxAck(TxAckMsg {
                tx_id: 1,
                shard_id: 0,
                success: true,
                error: None,
            }),
        ];

        for msg in messages {
            let debug = format!("{:?}", msg);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_tx_handler_prepare() {
        use std::sync::Arc;

        use crate::{block::Transaction, distributed_tx::TxParticipant};

        let participant = Arc::new(TxParticipant::new());
        let handler = TxHandler::new(participant);

        let prepare_msg = Message::TxPrepare(TxPrepareMsg {
            tx_id: 1,
            coordinator: "coordinator".to_string(),
            shard_id: 0,
            operations: vec![Transaction::Put {
                key: "test_key".to_string(),
                data: vec![1, 2, 3],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            timeout_ms: 5000,
        });

        let response = handler.handle(&"coordinator".to_string(), &prepare_msg);
        assert!(response.is_some());

        if let Some(Message::TxPrepareResponse(resp)) = response {
            assert_eq!(resp.tx_id, 1);
            assert!(matches!(resp.vote, TxVote::Yes { .. }));
        } else {
            panic!("Expected TxPrepareResponse");
        }
    }

    #[test]
    fn test_tx_handler_commit() {
        use std::sync::Arc;

        use crate::{block::Transaction, distributed_tx::TxParticipant};

        let participant = Arc::new(TxParticipant::new());
        let handler = TxHandler::new(participant.clone());

        // First prepare
        let prepare_msg = Message::TxPrepare(TxPrepareMsg {
            tx_id: 1,
            coordinator: "coordinator".to_string(),
            shard_id: 0,
            operations: vec![Transaction::Put {
                key: "test_key".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        });
        handler.handle(&"coordinator".to_string(), &prepare_msg);

        // Then commit
        let commit_msg = Message::TxCommit(TxCommitMsg {
            tx_id: 1,
            shards: vec![0],
        });

        let response = handler.handle(&"coordinator".to_string(), &commit_msg);
        assert!(response.is_some());

        if let Some(Message::TxAck(ack)) = response {
            assert_eq!(ack.tx_id, 1);
            assert!(ack.success);
        } else {
            panic!("Expected TxAck");
        }
    }

    #[test]
    fn test_tx_handler_abort() {
        use std::sync::Arc;

        use crate::{block::Transaction, distributed_tx::TxParticipant};

        let participant = Arc::new(TxParticipant::new());
        let handler = TxHandler::new(participant.clone());

        // First prepare
        let prepare_msg = Message::TxPrepare(TxPrepareMsg {
            tx_id: 1,
            coordinator: "coordinator".to_string(),
            shard_id: 0,
            operations: vec![Transaction::Put {
                key: "test_key".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        });
        handler.handle(&"coordinator".to_string(), &prepare_msg);

        // Then abort
        let abort_msg = Message::TxAbort(TxAbortMsg {
            tx_id: 1,
            reason: "test abort".to_string(),
            shards: vec![0],
        });

        let response = handler.handle(&"coordinator".to_string(), &abort_msg);
        assert!(response.is_some());

        if let Some(Message::TxAck(ack)) = response {
            assert_eq!(ack.tx_id, 1);
            assert!(ack.success);
        } else {
            panic!("Expected TxAck");
        }

        // Participant should have no prepared transactions
        assert_eq!(participant.prepared_count(), 0);
    }

    #[test]
    fn test_tx_handler_ignores_non_2pc_messages() {
        use std::sync::Arc;

        use crate::distributed_tx::TxParticipant;

        let participant = Arc::new(TxParticipant::new());
        let handler = TxHandler::new(participant);

        let ping_msg = Message::Ping { term: 1 };
        let response = handler.handle(&"peer".to_string(), &ping_msg);
        assert!(response.is_none());
    }

    #[test]
    fn test_tx_handler_prepare_conflict() {
        use std::sync::Arc;

        use crate::{block::Transaction, distributed_tx::TxParticipant};

        let participant = Arc::new(TxParticipant::new());
        let handler = TxHandler::new(participant);

        // First prepare succeeds
        let prepare_msg1 = Message::TxPrepare(TxPrepareMsg {
            tx_id: 1,
            coordinator: "coordinator".to_string(),
            shard_id: 0,
            operations: vec![Transaction::Put {
                key: "conflict_key".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            timeout_ms: 5000,
        });

        let response1 = handler.handle(&"coordinator".to_string(), &prepare_msg1);
        assert!(response1.is_some());
        if let Some(Message::TxPrepareResponse(resp)) = response1 {
            assert!(matches!(resp.vote, TxVote::Yes { .. }));
        }

        // Second prepare on same key should return Conflict
        let prepare_msg2 = Message::TxPrepare(TxPrepareMsg {
            tx_id: 2,
            coordinator: "coordinator".to_string(),
            shard_id: 0,
            operations: vec![Transaction::Put {
                key: "conflict_key".to_string(),
                data: vec![2],
            }],
            delta_embedding: SparseVector::from_dense(&[0.0, 1.0, 0.0]),
            timeout_ms: 5000,
        });

        let response2 = handler.handle(&"coordinator".to_string(), &prepare_msg2);
        assert!(response2.is_some());
        if let Some(Message::TxPrepareResponse(resp)) = response2 {
            assert!(matches!(resp.vote, TxVote::Conflict { .. }));
        }
    }

    #[test]
    fn test_message_routing_embedding_request_vote() {
        let msg = Message::RequestVote(RequestVote {
            term: 1,
            candidate_id: "c1".to_string(),
            last_log_index: 5,
            last_log_term: 1,
            state_embedding: SparseVector::from_dense(&[0.1, 0.2, 0.3]),
        });

        let embedding = msg.routing_embedding();
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap().dimension(), 3);
        assert!(msg.has_routing_embedding());
    }

    #[test]
    fn test_message_routing_embedding_append_entries_with() {
        let msg = Message::AppendEntries(AppendEntries {
            term: 1,
            leader_id: "leader".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(SparseVector::from_dense(&[0.5, 0.5])),
        });

        let embedding = msg.routing_embedding();
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap().dimension(), 2);
        assert!(msg.has_routing_embedding());
    }

    #[test]
    fn test_message_routing_embedding_append_entries_without() {
        let msg = Message::AppendEntries(AppendEntries {
            term: 1,
            leader_id: "leader".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: None,
        });

        let embedding = msg.routing_embedding();
        assert!(embedding.is_none());
        assert!(!msg.has_routing_embedding());
    }

    #[test]
    fn test_message_routing_embedding_tx_prepare() {
        let msg = Message::TxPrepare(TxPrepareMsg {
            tx_id: 1,
            coordinator: "coord".to_string(),
            shard_id: 0,
            operations: vec![],
            delta_embedding: SparseVector::from_dense(&[1.0, 2.0, 3.0, 4.0]),
            timeout_ms: 1000,
        });

        let embedding = msg.routing_embedding();
        assert!(embedding.is_some());
        assert_eq!(embedding.unwrap().dimension(), 4);
        assert!(msg.has_routing_embedding());
    }

    #[test]
    fn test_message_routing_embedding_ping() {
        let msg = Message::Ping { term: 1 };
        assert!(msg.routing_embedding().is_none());
        assert!(!msg.has_routing_embedding());
    }

    #[test]
    fn test_message_routing_embedding_pong() {
        let msg = Message::Pong { term: 1 };
        assert!(msg.routing_embedding().is_none());
        assert!(!msg.has_routing_embedding());
    }

    #[test]
    fn test_message_routing_embedding_tx_commit() {
        let msg = Message::TxCommit(TxCommitMsg {
            tx_id: 1,
            shards: vec![0],
        });
        assert!(msg.routing_embedding().is_none());
        assert!(!msg.has_routing_embedding());
    }

    #[test]
    fn test_message_routing_embedding_tx_abort() {
        let msg = Message::TxAbort(TxAbortMsg {
            tx_id: 1,
            reason: "test".to_string(),
            shards: vec![0],
        });
        assert!(msg.routing_embedding().is_none());
        assert!(!msg.has_routing_embedding());
    }

    // =========================================================================
    // Partition Injection Tests
    // =========================================================================

    #[test]
    fn test_partition_basic() {
        let t1 = MemoryTransport::new("node1".to_string());
        let t2 = MemoryTransport::new("node2".to_string());

        t1.connect_to("node2".to_string(), t2.sender());

        // Initially not partitioned
        assert!(!t1.is_partitioned(&"node2".to_string()));
        assert!(t1.partitioned_peers().is_empty());

        // Partition
        t1.partition(&"node2".to_string());
        assert!(t1.is_partitioned(&"node2".to_string()));
        assert_eq!(t1.partitioned_peers().len(), 1);

        // Heal
        t1.heal(&"node2".to_string());
        assert!(!t1.is_partitioned(&"node2".to_string()));
        assert!(t1.partitioned_peers().is_empty());
    }

    #[test]
    fn test_partition_all_and_heal_all() {
        let t1 = MemoryTransport::new("node1".to_string());
        let t2 = MemoryTransport::new("node2".to_string());
        let t3 = MemoryTransport::new("node3".to_string());

        t1.connect_to("node2".to_string(), t2.sender());
        t1.connect_to("node3".to_string(), t3.sender());

        // Partition all
        t1.partition_all();
        assert!(t1.is_partitioned(&"node2".to_string()));
        assert!(t1.is_partitioned(&"node3".to_string()));
        assert_eq!(t1.partitioned_peers().len(), 2);

        // Heal all
        t1.heal_all();
        assert!(!t1.is_partitioned(&"node2".to_string()));
        assert!(!t1.is_partitioned(&"node3".to_string()));
        assert!(t1.partitioned_peers().is_empty());
    }

    #[tokio::test]
    async fn test_partition_blocks_send() {
        let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
        let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

        t1.connect_to("node2".to_string(), t2.sender());
        t2.connect_to("node1".to_string(), t1.sender());

        // Send works before partition
        let result = t1
            .send(&"node2".to_string(), Message::Ping { term: 1 })
            .await;
        assert!(result.is_ok());
        assert_eq!(t1.dropped_message_count(), 0);

        // Partition
        t1.partition(&"node2".to_string());

        // Send fails after partition
        let result = t1
            .send(&"node2".to_string(), Message::Ping { term: 2 })
            .await;
        assert!(result.is_err());
        assert_eq!(t1.dropped_message_count(), 1);

        // Heal and send works again
        t1.heal(&"node2".to_string());
        let result = t1
            .send(&"node2".to_string(), Message::Ping { term: 3 })
            .await;
        assert!(result.is_ok());
        assert_eq!(t1.dropped_message_count(), 1); // Still 1
    }

    #[tokio::test]
    async fn test_partition_is_asymmetric() {
        let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
        let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

        t1.connect_to("node2".to_string(), t2.sender());
        t2.connect_to("node1".to_string(), t1.sender());

        // Partition only on t1 side
        t1.partition(&"node2".to_string());

        // t1 -> t2 fails
        assert!(t1
            .send(&"node2".to_string(), Message::Ping { term: 1 })
            .await
            .is_err());

        // t2 -> t1 still works (asymmetric)
        assert!(t2
            .send(&"node1".to_string(), Message::Ping { term: 1 })
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn test_broadcast_skips_partitioned() {
        let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
        let t2 = Arc::new(MemoryTransport::new("node2".to_string()));
        let t3 = Arc::new(MemoryTransport::new("node3".to_string()));

        t1.connect_to("node2".to_string(), t2.sender());
        t1.connect_to("node3".to_string(), t3.sender());

        // Partition node2 only
        t1.partition(&"node2".to_string());

        // Broadcast from t1
        t1.broadcast(Message::Ping { term: 1 }).await.unwrap();

        // t3 should receive (not partitioned)
        let result = tokio::time::timeout(std::time::Duration::from_millis(100), t3.recv()).await;
        assert!(result.is_ok());

        // t2 should NOT receive (partitioned)
        let result = tokio::time::timeout(std::time::Duration::from_millis(50), t2.recv()).await;
        assert!(result.is_err()); // Timeout
    }

    #[tokio::test]
    async fn test_dropped_message_counter() {
        let t1 = Arc::new(MemoryTransport::new("node1".to_string()));
        let t2 = Arc::new(MemoryTransport::new("node2".to_string()));

        t1.connect_to("node2".to_string(), t2.sender());
        t1.partition(&"node2".to_string());

        // Send 5 messages - all dropped
        for _ in 0..5 {
            let _ = t1
                .send(&"node2".to_string(), Message::Ping { term: 1 })
                .await;
        }
        assert_eq!(t1.dropped_message_count(), 5);

        // Heal and send - should not increment
        t1.heal(&"node2".to_string());
        t1.send(&"node2".to_string(), Message::Ping { term: 1 })
            .await
            .unwrap();
        assert_eq!(t1.dropped_message_count(), 5);
    }

    // ========== Joint Consensus Tests ==========

    #[test]
    fn test_joint_config_has_joint_quorum_both_required() {
        let joint = JointConfig {
            old_voters: vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
            new_voters: vec!["n2".to_string(), "n3".to_string(), "n4".to_string()],
        };

        // Need majority from both: old requires 2/3, new requires 2/3
        let mut votes = HashSet::new();
        votes.insert("n1".to_string());
        votes.insert("n2".to_string());
        // Only old has quorum (n1, n2), new only has 1 (n2)
        assert!(!joint.has_joint_quorum(&votes));

        votes.insert("n3".to_string());
        // Now both have quorum: old has 3/3, new has 2/3 (n2, n3)
        assert!(joint.has_joint_quorum(&votes));
    }

    #[test]
    fn test_joint_config_serialization() {
        let joint = JointConfig {
            old_voters: vec!["n1".to_string(), "n2".to_string()],
            new_voters: vec!["n2".to_string(), "n3".to_string()],
        };

        let bytes = bincode::serialize(&joint).unwrap();
        let decoded: JointConfig = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.old_voters.len(), 2);
        assert_eq!(decoded.new_voters.len(), 2);
    }

    #[test]
    fn test_raft_membership_config_quorum_simple() {
        let config =
            RaftMembershipConfig::new(vec!["n1".to_string(), "n2".to_string(), "n3".to_string()]);

        let mut votes = HashSet::new();
        votes.insert("n1".to_string());
        // 1/3 - not quorum
        assert!(!config.has_quorum(&votes));

        votes.insert("n2".to_string());
        // 2/3 - quorum (majority)
        assert!(config.has_quorum(&votes));
    }

    #[test]
    fn test_raft_membership_config_quorum_joint() {
        let mut config = RaftMembershipConfig::new(vec!["n1".to_string(), "n2".to_string()]);
        config.joint = Some(JointConfig {
            old_voters: vec!["n1".to_string(), "n2".to_string()],
            new_voters: vec!["n2".to_string(), "n3".to_string()],
        });

        let mut votes = HashSet::new();
        votes.insert("n1".to_string());
        // Old has 1/2 (needs 2), not quorum
        assert!(!config.has_quorum(&votes));

        votes.insert("n2".to_string());
        // Old has 2/2, new has 1/2 (needs 2), not joint quorum
        assert!(!config.has_quorum(&votes));

        votes.insert("n3".to_string());
        // Old has 2/2, new has 2/2, joint quorum achieved
        assert!(config.has_quorum(&votes));
    }

    #[test]
    fn test_raft_membership_config_replication_targets() {
        let mut config = RaftMembershipConfig::new(vec!["n1".to_string(), "n2".to_string()]);
        config.learners.push("learner1".to_string());

        let targets = config.replication_targets();
        assert!(targets.contains(&"n1".to_string()));
        assert!(targets.contains(&"n2".to_string()));
        assert!(targets.contains(&"learner1".to_string()));
    }

    #[test]
    fn test_raft_membership_config_replication_targets_joint() {
        let mut config = RaftMembershipConfig::new(vec!["n1".to_string(), "n2".to_string()]);
        config.joint = Some(JointConfig {
            old_voters: vec!["n1".to_string(), "n2".to_string()],
            new_voters: vec!["n2".to_string(), "n3".to_string()],
        });

        let targets = config.replication_targets();
        assert!(targets.contains(&"n1".to_string()));
        assert!(targets.contains(&"n2".to_string()));
        assert!(targets.contains(&"n3".to_string())); // New voter included
    }

    #[test]
    fn test_raft_membership_config_add_learner() {
        let mut config = RaftMembershipConfig::new(vec!["n1".to_string()]);
        assert!(config.learners.is_empty());

        config.add_learner("learner1".to_string());
        assert!(config.learners.contains(&"learner1".to_string()));
    }

    #[test]
    fn test_raft_membership_config_promote_learner() {
        let mut config = RaftMembershipConfig::new(vec!["n1".to_string()]);
        config.add_learner("learner1".to_string());

        config.promote_learner(&"learner1".to_string());
        assert!(!config.learners.contains(&"learner1".to_string()));
        assert!(config.voters.contains(&"learner1".to_string()));
    }

    #[test]
    fn test_raft_membership_config_in_joint_consensus() {
        let mut config = RaftMembershipConfig::new(vec!["n1".to_string()]);
        assert!(!config.in_joint_consensus());

        config.joint = Some(JointConfig {
            old_voters: vec!["n1".to_string()],
            new_voters: vec!["n1".to_string(), "n2".to_string()],
        });
        assert!(config.in_joint_consensus());
    }

    #[test]
    fn test_config_change_serialization() {
        let changes = vec![
            ConfigChange::AddLearner {
                node_id: "new_node".to_string(),
            },
            ConfigChange::PromoteLearner {
                node_id: "learner1".to_string(),
            },
            ConfigChange::RemoveNode {
                node_id: "old_node".to_string(),
            },
            ConfigChange::JointChange {
                additions: vec!["n4".to_string()],
                removals: vec!["n1".to_string()],
            },
        ];

        for change in changes {
            let bytes = bincode::serialize(&change).unwrap();
            let decoded: ConfigChange = bincode::deserialize(&bytes).unwrap();
            assert_eq!(decoded, change);
        }
    }

    #[test]
    fn test_log_entry_new() {
        use crate::block::BlockHeader;

        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "proposer".to_string());
        let block = crate::block::Block::new(header, vec![]);

        let entry = LogEntry::new(1, 1, block);
        assert_eq!(entry.term, 1);
        assert_eq!(entry.index, 1);
        assert!(!entry.is_config_change());
    }

    #[test]
    fn test_log_entry_config() {
        let entry = LogEntry::config(
            2,
            5,
            ConfigChange::AddLearner {
                node_id: "new_node".to_string(),
            },
        );
        assert_eq!(entry.term, 2);
        assert_eq!(entry.index, 5);
        assert!(entry.is_config_change());
        assert!(matches!(
            entry.config_change,
            Some(ConfigChange::AddLearner { .. })
        ));
    }

    #[test]
    fn test_log_entry_with_config_change_serialization() {
        use crate::block::BlockHeader;

        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "proposer".to_string());
        let block = crate::block::Block::new(header, vec![]);

        let mut entry = LogEntry::new(1, 1, block);
        entry.config_change = Some(ConfigChange::AddLearner {
            node_id: "new_node".to_string(),
        });

        let bytes = bincode::serialize(&entry).unwrap();
        let decoded: LogEntry = bincode::deserialize(&bytes).unwrap();
        assert!(decoded.is_config_change());
    }
}
