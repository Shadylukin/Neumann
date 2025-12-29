//! Network transport abstraction for distributed consensus.
//!
//! Provides a pluggable transport layer for node communication:
//! - `Transport` trait defines the interface
//! - `MemoryTransport` for testing
//! - Message types for Raft and sync protocols

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::block::{Block, BlockHash, NodeId};
use crate::error::{ChainError, Result};

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
    pub state_embedding: Vec<f32>,
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
    /// Block embedding for similarity fast-path.
    pub block_embedding: Option<Vec<f32>>,
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

/// Transport trait for network communication.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send a message to a specific peer.
    async fn send(&self, to: &NodeId, msg: Message) -> Result<()>;

    /// Broadcast a message to all known peers.
    async fn broadcast(&self, msg: Message) -> Result<()>;

    /// Receive the next message.
    async fn recv(&self) -> Result<(NodeId, Message)>;

    /// Connect to a peer.
    async fn connect(&self, peer: &PeerConfig) -> Result<()>;

    /// Disconnect from a peer.
    async fn disconnect(&self, peer_id: &NodeId) -> Result<()>;

    /// Get list of connected peers.
    fn peers(&self) -> Vec<NodeId>;

    /// Get local node ID.
    fn local_id(&self) -> &NodeId;
}

/// In-memory transport for testing.
pub struct MemoryTransport {
    /// Local node ID.
    local_id: NodeId,
    /// Channel for receiving messages.
    receiver: RwLock<mpsc::Receiver<(NodeId, Message)>>,
    /// Senders to other nodes.
    peers: RwLock<HashMap<NodeId, mpsc::Sender<(NodeId, Message)>>>,
    /// Sender for this node (given to peers).
    local_sender: mpsc::Sender<(NodeId, Message)>,
}

impl MemoryTransport {
    /// Create a new memory transport.
    pub fn new(local_id: NodeId) -> Self {
        let (tx, rx) = mpsc::channel(1000);
        Self {
            local_id,
            receiver: RwLock::new(rx),
            peers: RwLock::new(HashMap::new()),
            local_sender: tx,
        }
    }

    /// Get sender for this node (used to connect other nodes).
    pub fn sender(&self) -> mpsc::Sender<(NodeId, Message)> {
        self.local_sender.clone()
    }

    /// Connect to another memory transport.
    pub fn connect_to(&self, other_id: NodeId, sender: mpsc::Sender<(NodeId, Message)>) {
        self.peers.write().insert(other_id, sender);
    }
}

#[async_trait]
impl Transport for MemoryTransport {
    async fn send(&self, to: &NodeId, msg: Message) -> Result<()> {
        let sender = {
            let peers = self.peers.read();
            peers
                .get(to)
                .cloned()
                .ok_or_else(|| ChainError::NetworkError(format!("peer not found: {}", to)))?
        };

        sender
            .send((self.local_id.clone(), msg))
            .await
            .map_err(|e| ChainError::NetworkError(e.to_string()))?;

        Ok(())
    }

    async fn broadcast(&self, msg: Message) -> Result<()> {
        let senders: Vec<_> = {
            let peers = self.peers.read();
            peers.values().cloned().collect()
        };

        for sender in senders {
            let _ = sender.send((self.local_id.clone(), msg.clone())).await;
        }
        Ok(())
    }

    async fn recv(&self) -> Result<(NodeId, Message)> {
        // Use tokio mutex instead for async safety
        let msg = {
            let mut receiver = self.receiver.write();
            // Try to receive without blocking
            receiver.try_recv().ok()
        };

        if let Some(msg) = msg {
            return Ok(msg);
        }

        // For blocking receive, we need a different approach
        // This is simplified - in production, use tokio::sync::Mutex
        Err(ChainError::NetworkError("no message available".to_string()))
    }

    async fn connect(&self, peer: &PeerConfig) -> Result<()> {
        // For memory transport, connection is handled externally
        // This is a no-op - use connect_to directly
        let _ = peer;
        Ok(())
    }

    async fn disconnect(&self, peer_id: &NodeId) -> Result<()> {
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
    /// Handle a message from a peer.
    fn handle(&self, from: &NodeId, msg: &Message) -> Option<Message>;
}

impl NetworkManager {
    /// Create a new network manager.
    pub fn new(transport: Arc<dyn Transport>) -> Self {
        Self {
            transport,
            handlers: RwLock::new(Vec::new()),
        }
    }

    /// Register a message handler.
    pub fn add_handler(&self, handler: Box<dyn MessageHandler>) {
        self.handlers.write().push(handler);
    }

    /// Get the transport.
    pub fn transport(&self) -> &Arc<dyn Transport> {
        &self.transport
    }

    /// Process incoming messages.
    pub async fn process_messages(&self) -> Result<()> {
        loop {
            let (from, msg) = self.transport.recv().await?;

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
            state_embedding: vec![0.1, 0.2, 0.3],
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

        let entry = LogEntry {
            term: 1,
            index: 1,
            block,
        };

        let bytes = bincode::serialize(&entry).unwrap();
        let decoded: LogEntry = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.term, 1);
        assert_eq!(decoded.index, 1);
    }
}
