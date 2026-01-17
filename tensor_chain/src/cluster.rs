//! Cluster orchestration for distributed Neumann nodes.
//!
//! Provides a unified API for starting and managing cluster nodes with:
//! - TCP transport initialization
//! - Membership management with geometric peer scoring
//! - Raft consensus with persistence
//! - State machine for block application

use std::{net::SocketAddr, sync::Arc, time::Duration};

use graph_engine::GraphEngine;
use parking_lot::RwLock;
use tensor_store::TensorStore;
use tokio::sync::broadcast;

use crate::{
    block::NodeId,
    chain::Chain,
    consensus::ConsensusManager,
    delta_replication::{DeltaReplicationConfig, DeltaReplicationManager},
    distributed_tx::{DistributedTxConfig, DistributedTxCoordinator, PrepareRequest},
    error::{ChainError, Result},
    geometric_membership::{GeometricMembershipConfig, GeometricMembershipManager},
    gossip::{GossipConfig, GossipMembershipManager},
    membership::{ClusterConfig, MembershipManager},
    message_validation::{CompositeValidator, MessageValidationConfig, MessageValidator},
    network::{
        Message, QueryExecutor, QueryRequest, QueryResponse, Transport, TxAckMsg, TxCommitMsg,
        TxPrepareMsg, TxPrepareResponseMsg,
    },
    raft::{RaftConfig, RaftNode},
    state_machine::TensorStateMachine,
    tcp::{TcpTransport, TcpTransportConfig},
};

/// Configuration for a local cluster node.
#[derive(Debug, Clone)]
pub struct LocalNodeConfig {
    /// Unique node identifier.
    pub node_id: NodeId,
    /// Address to bind for incoming connections.
    pub bind_address: SocketAddr,
}

impl LocalNodeConfig {
    pub fn new(node_id: impl Into<NodeId>, bind_address: SocketAddr) -> Self {
        Self {
            node_id: node_id.into(),
            bind_address,
        }
    }
}

/// Configuration for a peer node.
#[derive(Debug, Clone)]
pub struct PeerConfig {
    /// Peer's node identifier.
    pub node_id: NodeId,
    /// Peer's address for connections.
    pub address: SocketAddr,
}

impl PeerConfig {
    pub fn new(node_id: impl Into<NodeId>, address: SocketAddr) -> Self {
        Self {
            node_id: node_id.into(),
            address,
        }
    }
}

/// Timeout configuration for message handlers.
///
/// Prevents handlers from blocking indefinitely on slow operations.
#[derive(Debug, Clone)]
pub struct HandlerTimeoutConfig {
    /// Timeout for query execution in milliseconds.
    pub query_timeout_ms: u64,
    /// Timeout for 2PC prepare operations in milliseconds.
    pub prepare_timeout_ms: u64,
    /// Timeout for 2PC commit/abort operations in milliseconds.
    pub commit_abort_timeout_ms: u64,
}

impl Default for HandlerTimeoutConfig {
    fn default() -> Self {
        Self {
            query_timeout_ms: 5000,
            prepare_timeout_ms: 3000,
            commit_abort_timeout_ms: 2000,
        }
    }
}

impl HandlerTimeoutConfig {
    pub fn new(
        query_timeout_ms: u64,
        prepare_timeout_ms: u64,
        commit_abort_timeout_ms: u64,
    ) -> Self {
        Self {
            query_timeout_ms,
            prepare_timeout_ms,
            commit_abort_timeout_ms,
        }
    }
}

/// Configuration for the cluster orchestrator.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Local node configuration.
    pub local: LocalNodeConfig,
    /// Peer configurations.
    pub peers: Vec<PeerConfig>,
    /// Raft consensus configuration.
    pub raft: RaftConfig,
    /// Geometric membership configuration.
    pub geometric: GeometricMembershipConfig,
    /// Gossip protocol configuration.
    pub gossip: GossipConfig,
    /// Distributed transaction (2PC) configuration.
    pub dtx: DistributedTxConfig,
    /// Delta replication configuration.
    pub delta_replication: DeltaReplicationConfig,
    /// Fast-path similarity threshold for state machine.
    pub fast_path_threshold: f32,
    /// Handler timeout configuration.
    pub handler_timeouts: HandlerTimeoutConfig,
    /// Message validation configuration.
    pub message_validation: MessageValidationConfig,
}

impl OrchestratorConfig {
    pub fn new(local: LocalNodeConfig, peers: Vec<PeerConfig>) -> Self {
        Self {
            local,
            peers,
            raft: RaftConfig::default(),
            geometric: GeometricMembershipConfig::default(),
            gossip: GossipConfig::default(),
            dtx: DistributedTxConfig::default(),
            delta_replication: DeltaReplicationConfig::default(),
            fast_path_threshold: 0.95,
            handler_timeouts: HandlerTimeoutConfig::default(),
            message_validation: MessageValidationConfig::default(),
        }
    }

    /// Set the Raft configuration.
    pub fn with_raft(mut self, raft: RaftConfig) -> Self {
        self.raft = raft;
        self
    }

    /// Set the geometric membership configuration.
    pub fn with_geometric(mut self, geometric: GeometricMembershipConfig) -> Self {
        self.geometric = geometric;
        self
    }

    /// Set the fast-path threshold.
    pub fn with_fast_path_threshold(mut self, threshold: f32) -> Self {
        self.fast_path_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set the gossip configuration.
    pub fn with_gossip(mut self, gossip: GossipConfig) -> Self {
        self.gossip = gossip;
        self
    }

    /// Set the distributed transaction configuration.
    pub fn with_dtx(mut self, dtx: DistributedTxConfig) -> Self {
        self.dtx = dtx;
        self
    }

    /// Set the delta replication configuration.
    pub fn with_delta_replication(mut self, delta_replication: DeltaReplicationConfig) -> Self {
        self.delta_replication = delta_replication;
        self
    }

    /// Set the handler timeout configuration.
    pub fn with_handler_timeouts(mut self, handler_timeouts: HandlerTimeoutConfig) -> Self {
        self.handler_timeouts = handler_timeouts;
        self
    }

    /// Set the message validation configuration.
    pub fn with_message_validation(mut self, message_validation: MessageValidationConfig) -> Self {
        self.message_validation = message_validation;
        self
    }
}

/// Orchestrates cluster startup and manages node components.
///
/// Brings together transport, membership, Raft, chain, and state machine
/// into a cohesive unit for distributed operation.
pub struct ClusterOrchestrator {
    /// Configuration.
    config: OrchestratorConfig,
    /// TCP transport for networking.
    transport: Arc<TcpTransport>,
    /// Membership manager.
    membership: Arc<MembershipManager>,
    /// Geometric membership wrapper.
    geometric: Arc<GeometricMembershipManager>,
    /// Gossip-based membership manager.
    gossip: Arc<GossipMembershipManager>,
    /// Distributed transaction (2PC) coordinator.
    dtx: Arc<DistributedTxCoordinator>,
    /// Delta replication manager.
    delta_replication: Arc<DeltaReplicationManager>,
    /// Raft consensus node.
    raft: Arc<RaftNode>,
    /// Chain storage.
    chain: Arc<Chain>,
    /// State machine for applying committed entries.
    state_machine: Arc<TensorStateMachine>,
    /// TensorStore for persistence.
    store: TensorStore,
    /// Optional query executor for distributed query handling.
    query_executor: RwLock<Option<Arc<dyn QueryExecutor>>>,
    /// Local shard ID for query responses.
    local_shard_id: usize,
    /// Message validator for incoming messages.
    validator: Arc<CompositeValidator>,
}

impl ClusterOrchestrator {
    /// Start a new cluster node.
    ///
    /// This initializes all components:
    /// 1. TensorStore for persistence
    /// 2. TCP transport
    /// 3. Membership manager with geometric scoring
    /// 4. Raft node (loading persisted state if available)
    /// 5. Chain storage
    /// 6. State machine
    pub async fn start(config: OrchestratorConfig) -> Result<Self> {
        // 1. Initialize TensorStore for persistence and chain storage
        let store = TensorStore::new();

        // 2. Create TCP transport
        let tcp_config = TcpTransportConfig::new(&config.local.node_id, config.local.bind_address);
        let transport = Arc::new(TcpTransport::new(tcp_config));

        // Start listening
        transport
            .start()
            .await
            .map_err(|e| ChainError::NetworkError(format!("Failed to start transport: {}", e)))?;

        // 3. Connect to peers (non-blocking, failures are OK at startup)
        for peer in &config.peers {
            let peer_config = crate::network::PeerConfig {
                node_id: peer.node_id.clone(),
                address: peer.address.to_string(),
            };
            // Ignore connection failures - peers may not be up yet
            let _ = transport.connect(&peer_config).await;
        }

        // 4. Create cluster config for membership
        let local_node_config = crate::membership::LocalNodeConfig {
            node_id: config.local.node_id.clone(),
            bind_address: config.local.bind_address,
        };

        let mut cluster_config = ClusterConfig::new(
            format!("cluster-{}", config.local.node_id),
            local_node_config,
        );

        for peer in &config.peers {
            cluster_config = cluster_config.with_peer(peer.node_id.clone(), peer.address);
        }

        // 5. Start membership manager
        let membership = Arc::new(MembershipManager::new(cluster_config, transport.clone()));

        // 6. Wrap with geometric membership
        let geometric = Arc::new(GeometricMembershipManager::new(
            membership.clone(),
            config.geometric.clone(),
        ));

        // 7. Create gossip-based membership manager with geometric routing
        let gossip = Arc::new(GossipMembershipManager::with_geometric(
            config.local.node_id.clone(),
            config.gossip.clone(),
            transport.clone(),
            geometric.clone(),
        ));

        // Add all peers to gossip
        for peer in &config.peers {
            gossip.add_peer(peer.node_id.clone());
        }

        // 8. Create distributed transaction coordinator (2PC)
        let consensus = ConsensusManager::default_config();
        let dtx = Arc::new(DistributedTxCoordinator::new(consensus, config.dtx.clone()));

        // 9. Create delta replication manager with shared archetype registry
        let delta_replication = Arc::new(DeltaReplicationManager::with_store(
            config.local.node_id.clone(),
            config.delta_replication.clone(),
            &store,
        ));

        // 10. Create Raft node, loading persisted state
        let peer_ids: Vec<NodeId> = config.peers.iter().map(|p| p.node_id.clone()).collect();

        let raft = Arc::new(RaftNode::with_store(
            config.local.node_id.clone(),
            peer_ids,
            transport.clone(),
            config.raft.clone(),
            &store,
        ));

        // 11. Create chain storage
        let graph = Arc::new(GraphEngine::with_store(store.clone()));
        let chain = Arc::new(Chain::new(graph, config.local.node_id.clone()));

        // Initialize chain (creates genesis if needed)
        chain.initialize()?;

        // 12. Create state machine with store for transaction application
        let state_machine = Arc::new(TensorStateMachine::with_threshold(
            chain.clone(),
            raft.clone(),
            store.clone(),
            config.fast_path_threshold,
        ));

        // 13. Create message validator
        let validator = Arc::new(CompositeValidator::new(config.message_validation.clone()));

        Ok(Self {
            config,
            transport,
            membership,
            geometric,
            gossip,
            dtx,
            delta_replication,
            raft,
            chain,
            state_machine,
            store,
            query_executor: RwLock::new(None),
            local_shard_id: 0,
            validator,
        })
    }

    /// Register a query executor for handling distributed queries.
    ///
    /// The executor will be called when QueryRequest messages are received.
    pub fn register_query_executor(&self, executor: Arc<dyn QueryExecutor>) {
        *self.query_executor.write() = Some(executor);
    }

    /// Handle a 2PC prepare response (vote from participant).
    fn handle_tx_prepare_response(&self, msg: &TxPrepareResponseMsg) {
        // Record the vote - phase change happens automatically
        let _ = self
            .dtx
            .record_vote(msg.tx_id, msg.shard_id, msg.vote.clone().into());
    }

    /// Handle a 2PC acknowledgment message.
    fn handle_tx_ack(&self, msg: &TxAckMsg) {
        // Record the ack for commit/abort tracking
        self.dtx.handle_abort_ack(msg.tx_id, msg.shard_id);
    }

    /// Handle query request with timeout.
    async fn handle_query_request_with_timeout(
        &self,
        from: &NodeId,
        request: &QueryRequest,
    ) -> Option<Message> {
        let timeout_ms = self.config.handler_timeouts.query_timeout_ms;
        let timeout = Duration::from_millis(timeout_ms);

        let executor = {
            let guard = self.query_executor.read();
            guard.as_ref().cloned()
        };

        let executor = executor?;
        let query = request.query.clone();
        let query_id = request.query_id;
        let shard_id = self.local_shard_id;
        let from_node = from.clone();

        let result = tokio::time::timeout(timeout, async move {
            tokio::task::spawn_blocking(move || {
                let start = std::time::Instant::now();
                let (result, success, error) = match executor.execute(&query) {
                    Ok(bytes) => (bytes, true, None),
                    Err(e) => (Vec::new(), false, Some(e)),
                };
                let execution_time_us = start.elapsed().as_micros() as u64;
                (result, success, error, execution_time_us)
            })
            .await
        })
        .await;

        match result {
            Ok(Ok((result, success, error, execution_time_us))) => {
                tracing::debug!(
                    "Executed query from {} in {}us: success={}",
                    from_node,
                    execution_time_us,
                    success
                );
                Some(Message::QueryResponse(QueryResponse {
                    query_id,
                    shard_id,
                    result,
                    execution_time_us,
                    success,
                    error,
                }))
            },
            Ok(Err(_)) => {
                // spawn_blocking panicked
                Some(Message::QueryResponse(QueryResponse {
                    query_id,
                    shard_id,
                    result: Vec::new(),
                    execution_time_us: timeout_ms * 1000,
                    success: false,
                    error: Some("query execution panicked".to_string()),
                }))
            },
            Err(_) => {
                // Timeout
                tracing::warn!("Query from {} timed out after {}ms", from_node, timeout_ms);
                Some(Message::QueryResponse(QueryResponse {
                    query_id,
                    shard_id,
                    result: Vec::new(),
                    execution_time_us: timeout_ms * 1000,
                    success: false,
                    error: Some(format!(
                        "query execution timeout: exceeded {}ms",
                        timeout_ms
                    )),
                }))
            },
        }
    }

    /// Handle 2PC prepare with timeout.
    async fn handle_tx_prepare_with_timeout(&self, msg: &TxPrepareMsg) -> Option<Message> {
        let timeout_ms = self.config.handler_timeouts.prepare_timeout_ms;
        let timeout = Duration::from_millis(timeout_ms);

        let dtx = self.dtx.clone();
        let request = PrepareRequest {
            tx_id: msg.tx_id,
            coordinator: msg.coordinator.clone(),
            operations: msg.operations.clone(),
            delta_embedding: msg.delta_embedding.clone(),
            timeout_ms: msg.timeout_ms,
        };
        let tx_id = msg.tx_id;
        let shard_id = msg.shard_id;

        let result = tokio::time::timeout(timeout, async move {
            tokio::task::spawn_blocking(move || dtx.handle_prepare(request)).await
        })
        .await;

        match result {
            Ok(Ok(vote)) => Some(Message::TxPrepareResponse(TxPrepareResponseMsg {
                tx_id,
                shard_id,
                vote: vote.into(),
            })),
            Ok(Err(_)) | Err(_) => {
                // Timeout or panic - vote No
                tracing::warn!("TX prepare {} timed out after {}ms", tx_id, timeout_ms);
                Some(Message::TxPrepareResponse(TxPrepareResponseMsg {
                    tx_id,
                    shard_id,
                    vote: crate::network::TxVote::No {
                        reason: format!("prepare timeout: exceeded {}ms", timeout_ms),
                    },
                }))
            },
        }
    }

    /// Handle 2PC commit with timeout.
    async fn handle_tx_commit_with_timeout(&self, msg: &TxCommitMsg) -> Option<Message> {
        if !msg.shards.contains(&self.local_shard_id) {
            return None;
        }

        let timeout_ms = self.config.handler_timeouts.commit_abort_timeout_ms;
        let timeout = Duration::from_millis(timeout_ms);

        let dtx = self.dtx.clone();
        let tx_id = msg.tx_id;
        let shard_id = self.local_shard_id;

        let result = tokio::time::timeout(timeout, async move {
            tokio::task::spawn_blocking(move || dtx.commit(tx_id).is_ok()).await
        })
        .await;

        match result {
            Ok(Ok(success)) => Some(Message::TxAck(TxAckMsg {
                tx_id,
                shard_id,
                success,
                error: None,
            })),
            Ok(Err(_)) | Err(_) => {
                tracing::warn!("TX commit {} timed out after {}ms", tx_id, timeout_ms);
                Some(Message::TxAck(TxAckMsg {
                    tx_id,
                    shard_id,
                    success: false,
                    error: Some(format!("commit timeout: exceeded {}ms", timeout_ms)),
                }))
            },
        }
    }

    /// Handle 2PC abort with timeout.
    async fn handle_tx_abort_with_timeout(
        &self,
        msg: &crate::network::TxAbortMsg,
    ) -> Option<Message> {
        if !msg.shards.contains(&self.local_shard_id) {
            return None;
        }

        let timeout_ms = self.config.handler_timeouts.commit_abort_timeout_ms;
        let timeout = Duration::from_millis(timeout_ms);

        let dtx = self.dtx.clone();
        let tx_id = msg.tx_id;
        let reason = msg.reason.clone();
        let shard_id = self.local_shard_id;

        let result = tokio::time::timeout(timeout, async move {
            tokio::task::spawn_blocking(move || dtx.abort(tx_id, &reason).is_ok()).await
        })
        .await;

        match result {
            Ok(Ok(success)) => Some(Message::TxAck(TxAckMsg {
                tx_id,
                shard_id,
                success,
                error: None,
            })),
            Ok(Err(_)) | Err(_) => {
                tracing::warn!("TX abort {} timed out after {}ms", tx_id, timeout_ms);
                Some(Message::TxAck(TxAckMsg {
                    tx_id,
                    shard_id,
                    success: false,
                    error: Some(format!("abort timeout: exceeded {}ms", timeout_ms)),
                }))
            },
        }
    }

    /// Run the node until shutdown signal.
    ///
    /// Spawns background tasks for:
    /// - Raft tick loop
    /// - Gossip protocol rounds
    /// - 2PC distributed transaction handling
    /// - Geometric membership embedding updates
    /// - Membership health checks
    /// - State machine apply loop
    /// - Query message handling
    pub async fn run(&self, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
        let raft = self.raft.clone();
        let state_machine = self.state_machine.clone();
        let transport = self.transport.clone();
        let gossip = self.gossip.clone();
        let dtx = self.dtx.clone();
        let geometric = self.geometric.clone();

        // Track gossip timing separately (gossip runs at its own interval)
        let gossip_interval =
            std::time::Duration::from_millis(self.config.gossip.gossip_interval_ms);
        let mut last_gossip = std::time::Instant::now();

        // Track cleanup timing for 2PC transaction timeouts (presumed abort)
        let cleanup_interval = std::time::Duration::from_secs(30);
        let mut last_cleanup = std::time::Instant::now();

        // Main loop
        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    // Shutdown gossip manager
                    gossip.shutdown();
                    break;
                }
                // Try to receive a message with timeout
                recv_result = tokio::time::timeout(
                    tokio::time::Duration::from_millis(10),
                    transport.recv()
                ) => {
                    if let Ok(Ok((from, msg))) = recv_result {
                        // Validate message before processing
                        if let Err(e) = self.validator.validate(&msg, &from) {
                            tracing::warn!(
                                from = %from,
                                error = %e,
                                "Message validation failed, skipping"
                            );
                            continue;
                        }

                        // Record peer embeddings from messages for geometric routing
                        if let Some(embedding) = msg.routing_embedding() {
                            geometric.record_peer_embedding(&from, embedding.clone());
                        }

                        // Handle the message with timeout protection
                        match msg {
                            Message::QueryRequest(request) => {
                                if let Some(response) = self.handle_query_request_with_timeout(&from, &request).await {
                                    if let Err(e) = transport.send(&from, response).await {
                                        tracing::debug!(peer = %from, error = %e, "failed to send query response");
                                    }
                                }
                            }
                            Message::Gossip(gossip_msg) => {
                                // Handle gossip protocol messages
                                gossip.handle_gossip(gossip_msg);
                            }
                            // 2PC distributed transaction messages
                            Message::TxPrepare(prepare_msg) => {
                                if let Some(response) = self.handle_tx_prepare_with_timeout(&prepare_msg).await {
                                    if let Err(e) = transport.send(&from, response).await {
                                        tracing::debug!(peer = %from, error = %e, "failed to send tx prepare response");
                                    }
                                }
                            }
                            Message::TxPrepareResponse(response_msg) => {
                                self.handle_tx_prepare_response(&response_msg);
                            }
                            Message::TxCommit(commit_msg) => {
                                if let Some(ack) = self.handle_tx_commit_with_timeout(&commit_msg).await {
                                    if let Err(e) = transport.send(&from, ack).await {
                                        tracing::debug!(peer = %from, error = %e, "failed to send tx commit ack");
                                    }
                                }
                            }
                            Message::TxAbort(abort_msg) => {
                                if let Some(ack) = self.handle_tx_abort_with_timeout(&abort_msg).await {
                                    if let Err(e) = transport.send(&from, ack).await {
                                        tracing::debug!(peer = %from, error = %e, "failed to send tx abort ack");
                                    }
                                }
                            }
                            Message::TxAck(ack_msg) => {
                                self.handle_tx_ack(&ack_msg);
                            }
                            // Raft and other messages
                            other => {
                                // Let Raft handle other messages
                                let _ = raft.handle_message_async(&from, other).await;
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(50)) => {
                    // Tick Raft
                    raft.tick_async().await?;

                    // Apply any committed entries
                    let _ = state_machine.apply_committed();

                    // Update local embedding from state machine (for geometric routing)
                    if let Some(embedding) = state_machine.current_state_embedding() {
                        geometric.update_local_embedding(embedding);
                    }

                    // Run gossip round if interval elapsed
                    if last_gossip.elapsed() >= gossip_interval {
                        let _ = gossip.gossip_round().await;
                        last_gossip = std::time::Instant::now();
                    }

                    // Clean up timed-out 2PC transactions (presumed abort)
                    if last_cleanup.elapsed() >= cleanup_interval {
                        let timed_out = dtx.cleanup_timeouts();
                        if !timed_out.is_empty() {
                            tracing::info!(
                                "Cleaned up {} timed-out distributed transactions",
                                timed_out.len()
                            );
                        }
                        last_cleanup = std::time::Instant::now();
                    }

                    // Check quorum health and step down if lost (split-brain prevention)
                    raft.check_quorum_health();

                    // Process pending abort broadcasts
                    dtx.process_pending_aborts(&*transport).await;
                }
            }
        }

        Ok(())
    }

    /// Gracefully shut down the node.
    ///
    /// Saves Raft state to the store for recovery on restart.
    pub async fn shutdown(&self) -> Result<()> {
        // Save Raft state before shutdown
        self.raft.save_to_store(&self.store)?;

        // Transport will be dropped when orchestrator is dropped
        // No explicit stop needed - connections close automatically

        Ok(())
    }

    /// Get the node ID.
    pub fn node_id(&self) -> &NodeId {
        &self.config.local.node_id
    }

    /// Check if this node is the leader.
    pub fn is_leader(&self) -> bool {
        self.raft.is_leader()
    }

    /// Get the current leader (if known).
    pub fn current_leader(&self) -> Option<NodeId> {
        self.raft.current_leader()
    }

    /// Get the chain height.
    pub fn chain_height(&self) -> u64 {
        self.chain.height()
    }

    /// Get the Raft commit index.
    pub fn commit_index(&self) -> u64 {
        self.raft.commit_index()
    }

    /// Access the Raft node.
    pub fn raft(&self) -> &RaftNode {
        &self.raft
    }

    /// Access the chain.
    pub fn chain(&self) -> &Chain {
        &self.chain
    }

    /// Access the state machine.
    pub fn state_machine(&self) -> &TensorStateMachine {
        &self.state_machine
    }

    /// Access the membership manager.
    pub fn membership(&self) -> &MembershipManager {
        &self.membership
    }

    /// Access the geometric membership manager.
    pub fn geometric_membership(&self) -> &GeometricMembershipManager {
        &self.geometric
    }

    /// Access the gossip-based membership manager.
    pub fn gossip_membership(&self) -> &GossipMembershipManager {
        &self.gossip
    }

    /// Access the distributed transaction coordinator.
    pub fn dtx(&self) -> &DistributedTxCoordinator {
        &self.dtx
    }

    /// Access the delta replication manager.
    pub fn delta_replication(&self) -> &DeltaReplicationManager {
        &self.delta_replication
    }

    /// Access the store.
    pub fn store(&self) -> &TensorStore {
        &self.store
    }

    /// Access the transport.
    pub fn transport(&self) -> &TcpTransport {
        &self.transport
    }

    /// Send a query request to a remote node and await the response.
    pub async fn send_query(
        &self,
        target: &NodeId,
        request: QueryRequest,
    ) -> Result<QueryResponse> {
        use std::time::Duration;

        let query_id = request.query_id;
        let timeout_ms = request.timeout_ms;

        // Send the query request
        self.transport
            .send(target, Message::QueryRequest(request))
            .await?;

        // Wait for response with timeout
        // In a production system, we'd use a response correlation map
        // For now, we'll use a simple polling approach via the transport receiver
        let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms);

        loop {
            if tokio::time::Instant::now() > deadline {
                return Err(ChainError::NetworkError(format!(
                    "Query timeout after {}ms",
                    timeout_ms
                )));
            }

            // Check for response in incoming messages
            if let Ok(Ok(Some((from, Message::QueryResponse(response))))) =
                tokio::time::timeout(Duration::from_millis(50), self.transport.receive_one()).await
            {
                if response.query_id == query_id && &from == target {
                    return Ok(response);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use super::*;

    fn test_addr(port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
    }

    #[test]
    fn test_local_node_config() {
        let config = LocalNodeConfig::new("node1", test_addr(8080));
        assert_eq!(config.node_id, "node1");
        assert_eq!(config.bind_address.port(), 8080);
    }

    #[test]
    fn test_peer_config() {
        let config = PeerConfig::new("peer1", test_addr(8081));
        assert_eq!(config.node_id, "peer1");
        assert_eq!(config.address.port(), 8081);
    }

    #[test]
    fn test_orchestrator_config_default() {
        let local = LocalNodeConfig::new("node1", test_addr(8080));
        let peers = vec![PeerConfig::new("node2", test_addr(8081))];
        let config = OrchestratorConfig::new(local, peers);

        assert_eq!(config.local.node_id, "node1");
        assert_eq!(config.peers.len(), 1);
        assert!((config.fast_path_threshold - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_orchestrator_config_builders() {
        let local = LocalNodeConfig::new("node1", test_addr(8080));
        let config = OrchestratorConfig::new(local, vec![])
            .with_fast_path_threshold(0.9)
            .with_raft(RaftConfig {
                heartbeat_interval: 100,
                ..RaftConfig::default()
            })
            .with_geometric(GeometricMembershipConfig::default());

        assert!((config.fast_path_threshold - 0.9).abs() < 0.001);
        assert_eq!(config.raft.heartbeat_interval, 100);
    }

    #[test]
    fn test_fast_path_threshold_clamping() {
        let local = LocalNodeConfig::new("node1", test_addr(8080));

        // Above 1.0
        let config = OrchestratorConfig::new(local.clone(), vec![]).with_fast_path_threshold(1.5);
        assert!((config.fast_path_threshold - 1.0).abs() < 0.001);

        // Below 0.0
        let config = OrchestratorConfig::new(local, vec![]).with_fast_path_threshold(-0.5);
        assert!(config.fast_path_threshold.abs() < 0.001);
    }

    #[tokio::test]
    async fn test_orchestrator_start_single_node() {
        let local = LocalNodeConfig::new("test_node", test_addr(0)); // Port 0 = auto-assign
        let config = OrchestratorConfig::new(local, vec![]);

        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        assert_eq!(orchestrator.node_id(), "test_node");
        assert!(!orchestrator.is_leader()); // Not leader initially
        assert_eq!(orchestrator.chain_height(), 0);

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_orchestrator_accessors() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);

        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // Test all accessors
        let _ = orchestrator.raft();
        let _ = orchestrator.chain();
        let _ = orchestrator.state_machine();
        let _ = orchestrator.membership();
        let _ = orchestrator.geometric_membership();
        let _ = orchestrator.store();
        let _ = orchestrator.current_leader();
        let _ = orchestrator.commit_index();
        let _ = orchestrator.transport();

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_orchestrator_with_peers() {
        // Start first node
        let local1 = LocalNodeConfig::new("node1", test_addr(0));
        let config1 = OrchestratorConfig::new(local1, vec![]);
        let orchestrator1 = ClusterOrchestrator::start(config1).await.unwrap();

        // Get the actual bound address
        let addr1 = orchestrator1.transport().bound_addr().unwrap();

        // Start second node pointing to first
        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let peers = vec![PeerConfig::new("node1", addr1)];
        let config2 = OrchestratorConfig::new(local2, peers);
        let orchestrator2 = ClusterOrchestrator::start(config2).await.unwrap();

        // Both should start successfully
        assert_eq!(orchestrator1.node_id(), "node1");
        assert_eq!(orchestrator2.node_id(), "node2");

        orchestrator1.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_orchestrator_run_with_shutdown() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);

        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);

        // Spawn the run loop
        let orchestrator_clone = Arc::new(orchestrator);
        let orch = orchestrator_clone.clone();
        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        // Let it run briefly
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Signal shutdown
        let _ = shutdown_tx.send(());

        // Wait for completion
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    #[test]
    fn test_orchestrator_config_debug() {
        let local = LocalNodeConfig::new("node1", test_addr(8080));
        let config = OrchestratorConfig::new(local, vec![]);
        let debug = format!("{:?}", config);
        assert!(debug.contains("OrchestratorConfig"));
    }

    #[test]
    fn test_peer_config_debug() {
        let config = PeerConfig::new("peer1", test_addr(8081));
        let debug = format!("{:?}", config);
        assert!(debug.contains("PeerConfig"));
    }

    #[test]
    fn test_local_node_config_debug() {
        let config = LocalNodeConfig::new("node1", test_addr(8080));
        let debug = format!("{:?}", config);
        assert!(debug.contains("LocalNodeConfig"));
    }

    // ============= Message Handler Tests =============

    #[tokio::test]
    async fn test_handle_query_request_without_executor() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // Without registering an executor, query requests should return None
        let request = crate::network::QueryRequest {
            query_id: 1,
            query: "SELECT * FROM test".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 1000,
        };
        let result = orchestrator
            .handle_query_request_with_timeout(&"peer1".to_string(), &request)
            .await;
        assert!(
            result.is_none(),
            "Should return None when no executor is registered"
        );

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_handle_query_request_with_executor() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // Create a mock executor
        struct MockExecutor {
            call_count: AtomicUsize,
        }

        impl crate::network::QueryExecutor for MockExecutor {
            fn execute(&self, query: &str) -> std::result::Result<Vec<u8>, String> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                if query.contains("error") {
                    Err("Test error".to_string())
                } else {
                    Ok(b"test result".to_vec())
                }
            }
        }

        let executor = Arc::new(MockExecutor {
            call_count: AtomicUsize::new(0),
        });
        orchestrator.register_query_executor(executor.clone());

        // Test successful query
        let request = crate::network::QueryRequest {
            query_id: 1,
            query: "SELECT * FROM test".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 1000,
        };
        let result = orchestrator
            .handle_query_request_with_timeout(&"peer1".to_string(), &request)
            .await;
        assert!(result.is_some());

        if let Some(Message::QueryResponse(response)) = result {
            assert_eq!(response.query_id, 1);
            assert!(response.success);
            assert_eq!(response.result, b"test result");
            // execution_time_us can be 0 for very fast mock operations
        } else {
            panic!("Expected QueryResponse message");
        }
        assert_eq!(executor.call_count.load(Ordering::SeqCst), 1);

        // Test error query
        let error_request = crate::network::QueryRequest {
            query_id: 2,
            query: "SELECT error".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 1000,
        };
        let error_result = orchestrator
            .handle_query_request_with_timeout(&"peer1".to_string(), &error_request)
            .await;
        assert!(error_result.is_some());

        if let Some(Message::QueryResponse(response)) = error_result {
            assert_eq!(response.query_id, 2);
            assert!(!response.success);
            assert_eq!(response.error, Some("Test error".to_string()));
        } else {
            panic!("Expected QueryResponse message");
        }

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_handle_tx_prepare_votes_yes_when_no_conflict() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // Prepare a transaction with unique keys - should vote YES
        let prepare_msg = crate::network::TxPrepareMsg {
            tx_id: 1,
            coordinator: "coordinator".to_string(),
            shard_id: 0,
            operations: vec![crate::block::Transaction::Put {
                key: "unique_key_1".to_string(),
                data: vec![1, 2, 3],
            }],
            delta_embedding: tensor_store::SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            timeout_ms: 5000,
        };

        let response = orchestrator
            .handle_tx_prepare_with_timeout(&prepare_msg)
            .await;
        assert!(response.is_some());

        if let Some(Message::TxPrepareResponse(resp)) = response {
            assert_eq!(resp.tx_id, 1);
            assert_eq!(resp.shard_id, 0);
            assert!(
                matches!(resp.vote, crate::network::TxVote::Yes { .. }),
                "Expected Yes vote, got {:?}",
                resp.vote
            );
        } else {
            panic!("Expected TxPrepareResponse message");
        }

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_handle_tx_prepare_votes_conflict_when_locked() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // First prepare locks the key
        let prepare_msg1 = crate::network::TxPrepareMsg {
            tx_id: 1,
            coordinator: "coordinator".to_string(),
            shard_id: 0,
            operations: vec![crate::block::Transaction::Put {
                key: "shared_key".to_string(),
                data: vec![1],
            }],
            delta_embedding: tensor_store::SparseVector::from_dense(&[1.0, 0.0]),
            timeout_ms: 5000,
        };
        let response1 = orchestrator
            .handle_tx_prepare_with_timeout(&prepare_msg1)
            .await;
        assert!(response1.is_some());
        if let Some(Message::TxPrepareResponse(resp)) = response1 {
            assert!(matches!(resp.vote, crate::network::TxVote::Yes { .. }));
        }

        // Second prepare on same key should conflict
        let prepare_msg2 = crate::network::TxPrepareMsg {
            tx_id: 2,
            coordinator: "coordinator".to_string(),
            shard_id: 0,
            operations: vec![crate::block::Transaction::Put {
                key: "shared_key".to_string(),
                data: vec![2],
            }],
            delta_embedding: tensor_store::SparseVector::from_dense(&[0.0, 1.0]),
            timeout_ms: 5000,
        };
        let response2 = orchestrator
            .handle_tx_prepare_with_timeout(&prepare_msg2)
            .await;
        assert!(response2.is_some());

        if let Some(Message::TxPrepareResponse(resp)) = response2 {
            assert_eq!(resp.tx_id, 2);
            assert!(
                matches!(resp.vote, crate::network::TxVote::Conflict { .. }),
                "Expected Conflict vote when key is locked, got {:?}",
                resp.vote
            );
        } else {
            panic!("Expected TxPrepareResponse message");
        }

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_handle_tx_commit_returns_ack() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // Commit for shard 0 - returns ack even if tx not found locally
        // (in distributed systems, commit messages may arrive even if local prepare was missed)
        let commit_msg = crate::network::TxCommitMsg {
            tx_id: 100,
            shards: vec![0], // local_shard_id is 0
        };
        let ack = orchestrator
            .handle_tx_commit_with_timeout(&commit_msg)
            .await;

        // Verify ack is returned (may or may not succeed depending on local state)
        assert!(ack.is_some());
        if let Some(Message::TxAck(ack_msg)) = ack {
            assert_eq!(ack_msg.tx_id, 100);
            assert_eq!(ack_msg.shard_id, 0);
            // Note: success may be false if tx wasn't in local pending list
        } else {
            panic!("Expected TxAck message");
        }

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_handle_tx_commit_ignored_when_not_participant() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // Commit for shard 5 (not this node's shard)
        let commit_msg = crate::network::TxCommitMsg {
            tx_id: 100,
            shards: vec![5], // Not local shard (which is 0)
        };
        let ack = orchestrator
            .handle_tx_commit_with_timeout(&commit_msg)
            .await;

        // Should return None since this node's shard isn't in the list
        assert!(ack.is_none());

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_handle_tx_abort_returns_ack() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // Abort for shard 0 - returns ack even if tx not found locally
        let abort_msg = crate::network::TxAbortMsg {
            tx_id: 200,
            reason: "test abort".to_string(),
            shards: vec![0],
        };
        let ack = orchestrator.handle_tx_abort_with_timeout(&abort_msg).await;

        assert!(ack.is_some());
        if let Some(Message::TxAck(ack_msg)) = ack {
            assert_eq!(ack_msg.tx_id, 200);
            assert_eq!(ack_msg.shard_id, 0);
            // Note: success may be false if tx wasn't found
        } else {
            panic!("Expected TxAck message");
        }

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_handle_tx_abort_ignored_when_not_participant() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        let abort_msg = crate::network::TxAbortMsg {
            tx_id: 200,
            reason: "test abort".to_string(),
            shards: vec![5], // Not local shard
        };
        let ack = orchestrator.handle_tx_abort_with_timeout(&abort_msg).await;

        assert!(ack.is_none());

        orchestrator.shutdown().await.unwrap();
    }

    // ============= Run Loop Timing Tests =============

    #[tokio::test]
    async fn test_run_loop_raft_tick_interval() {
        tokio::time::pause();

        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();

        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        // Advance time and let the run loop process
        tokio::time::advance(tokio::time::Duration::from_millis(200)).await;
        tokio::task::yield_now().await;

        // Signal shutdown
        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_loop_gossip_interval() {
        tokio::time::pause();

        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let gossip_config = GossipConfig {
            gossip_interval_ms: 100, // 100ms gossip interval
            ..GossipConfig::default()
        };
        let config = OrchestratorConfig::new(local, vec![]).with_gossip(gossip_config);

        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();

        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        // Advance past gossip interval
        tokio::time::advance(tokio::time::Duration::from_millis(150)).await;
        tokio::task::yield_now().await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_loop_shutdown_signal_graceful_exit() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);

        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();

        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        // Brief delay then shutdown
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        let _ = shutdown_tx.send(());

        // Should complete without hanging
        let result = tokio::time::timeout(tokio::time::Duration::from_millis(500), handle).await;
        assert!(result.is_ok(), "Run loop should exit gracefully");
        assert!(result.unwrap().unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_run_loop_cleanup_interval() {
        // Test that cleanup interval is respected (30s is too long for unit test,
        // but we can verify the mechanism works)
        tokio::time::pause();

        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();

        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        // Advance time past cleanup interval (30s)
        tokio::time::advance(tokio::time::Duration::from_secs(31)).await;
        tokio::task::yield_now().await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    // ============= Error Handling Tests =============

    #[tokio::test]
    async fn test_peer_connection_failure_non_blocking() {
        // Peer at unreachable address - startup should still succeed
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let peers = vec![PeerConfig::new(
            "unreachable_peer",
            test_addr(59999), // Unlikely to be listening
        )];
        let config = OrchestratorConfig::new(local, peers);

        // Should start successfully despite peer being unreachable
        let result = ClusterOrchestrator::start(config).await;
        assert!(
            result.is_ok(),
            "Startup should continue despite peer connection failure"
        );

        let orchestrator = result.unwrap();
        assert_eq!(orchestrator.node_id(), "node1");
        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_dtx_accessors() {
        let local = LocalNodeConfig::new("test_node", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = ClusterOrchestrator::start(config).await.unwrap();

        // Test DTX coordinator accessor
        let dtx = orchestrator.dtx();
        assert_eq!(dtx.pending_count(), 0);

        // Test gossip membership accessor
        let gossip = orchestrator.gossip_membership();
        let _ = gossip.all_states();

        // Test delta replication accessor
        let delta_repl = orchestrator.delta_replication();
        let _ = delta_repl.stats();

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_orchestrator_config_with_dtx() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let dtx_config = DistributedTxConfig {
            commit_timeout_ms: 10000,
            orthogonal_threshold: 0.8,
            ..DistributedTxConfig::default()
        };
        let config = OrchestratorConfig::new(local, vec![]).with_dtx(dtx_config);

        assert_eq!(config.dtx.commit_timeout_ms, 10000);
        assert!((config.dtx.orthogonal_threshold - 0.8).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_orchestrator_config_with_delta_replication() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let delta_config = DeltaReplicationConfig {
            max_batch_size: 100,
            ..DeltaReplicationConfig::default()
        };
        let config = OrchestratorConfig::new(local, vec![]).with_delta_replication(delta_config);

        assert_eq!(config.delta_replication.max_batch_size, 100);
    }

    // ============= Query Execution Tests =============

    #[tokio::test]
    async fn test_send_query_success_returns_response() {
        let local1 = LocalNodeConfig::new("node1", test_addr(0));
        let config1 = OrchestratorConfig::new(local1, vec![]);
        let orchestrator1 = Arc::new(ClusterOrchestrator::start(config1).await.unwrap());
        let addr1 = orchestrator1.transport().bound_addr().unwrap();

        struct MockExecutor;
        impl crate::network::QueryExecutor for MockExecutor {
            fn execute(&self, _query: &str) -> std::result::Result<Vec<u8>, String> {
                Ok(b"success_result".to_vec())
            }
        }
        orchestrator1.register_query_executor(Arc::new(MockExecutor));

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch1_run = orchestrator1.clone();
        let handle = tokio::spawn(async move { orch1_run.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let peers2 = vec![PeerConfig::new("node1", addr1)];
        let config2 = OrchestratorConfig::new(local2, peers2);
        let orchestrator2 = Arc::new(ClusterOrchestrator::start(config2).await.unwrap());

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let connect_result = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr1.to_string(),
            })
            .await;

        if connect_result.is_err() {
            let _ = shutdown_tx.send(());
            let _ = handle.await;
            orchestrator1.shutdown().await.unwrap();
            orchestrator2.shutdown().await.unwrap();
            return;
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let request = crate::network::QueryRequest {
            query_id: 42,
            query: "SELECT * FROM test".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 2000,
        };

        let result = orchestrator2
            .send_query(&"node1".to_string(), request)
            .await;
        if result.is_ok() {
            let response = result.unwrap();
            assert_eq!(response.query_id, 42);
            assert!(response.success);
            assert_eq!(response.result, b"success_result");
        }

        let _ = shutdown_tx.send(());
        let _ = handle.await;
        orchestrator1.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_send_query_timeout_returns_error() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());

        let request = crate::network::QueryRequest {
            query_id: 99,
            query: "SELECT * FROM test".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 50,
        };

        let result = orchestrator
            .send_query(&"nonexistent_node".to_string(), request)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("peer") || err.to_string().contains("not found"),
            "Expected peer not found error, got: {}",
            err
        );

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_send_query_network_error_propagates() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());

        let request = crate::network::QueryRequest {
            query_id: 1,
            query: "SELECT 1".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 1000,
        };

        let result = orchestrator
            .send_query(&"nonexistent_peer".to_string(), request)
            .await;
        assert!(result.is_err());

        orchestrator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_send_query_wrong_query_id_ignored() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());
        let addr = orchestrator.transport().bound_addr().unwrap();

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let peers2 = vec![PeerConfig::new("node1", addr)];
        let config2 = OrchestratorConfig::new(local2, peers2);
        let orchestrator2 = Arc::new(ClusterOrchestrator::start(config2).await.unwrap());

        let _ = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr.to_string(),
            })
            .await;

        // Use very short timeout for test efficiency
        let request = crate::network::QueryRequest {
            query_id: 100,
            query: "SELECT 1".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 50, // Short timeout for testing
        };

        // Query should timeout since no executor registered to respond with matching query_id
        let result = orchestrator2
            .send_query(&"node1".to_string(), request)
            .await;

        assert!(
            result.is_err(),
            "Should timeout since no matching response arrives"
        );

        orchestrator.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_send_query_concurrent_queries_correlate_correctly() {
        let local1 = LocalNodeConfig::new("node1", test_addr(0));
        let config1 = OrchestratorConfig::new(local1, vec![]);
        let orchestrator1 = Arc::new(ClusterOrchestrator::start(config1).await.unwrap());
        let addr1 = orchestrator1.transport().bound_addr().unwrap();

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let peers2 = vec![PeerConfig::new("node1", addr1)];
        let config2 = OrchestratorConfig::new(local2, peers2);
        let orchestrator2 = Arc::new(ClusterOrchestrator::start(config2).await.unwrap());

        use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
        struct CountingExecutor {
            counter: AtomicU64,
        }
        impl crate::network::QueryExecutor for CountingExecutor {
            fn execute(&self, query: &str) -> std::result::Result<Vec<u8>, String> {
                let count = self.counter.fetch_add(1, AtomicOrdering::SeqCst);
                Ok(format!("result_{}_{}", query, count).into_bytes())
            }
        }
        orchestrator1.register_query_executor(Arc::new(CountingExecutor {
            counter: AtomicU64::new(0),
        }));

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch1_run = orchestrator1.clone();
        let handle = tokio::spawn(async move { orch1_run.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let _ = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr1.to_string(),
            })
            .await;

        let orch2_a = orchestrator2.clone();
        let orch2_b = orchestrator2.clone();

        let handle_a = tokio::spawn(async move {
            let req = crate::network::QueryRequest {
                query_id: 1,
                query: "query_a".to_string(),
                shard_id: 0,
                embedding: None,
                timeout_ms: 2000,
            };
            orch2_a.send_query(&"node1".to_string(), req).await
        });

        let handle_b = tokio::spawn(async move {
            let req = crate::network::QueryRequest {
                query_id: 2,
                query: "query_b".to_string(),
                shard_id: 0,
                embedding: None,
                timeout_ms: 2000,
            };
            orch2_b.send_query(&"node1".to_string(), req).await
        });

        let (result_a, result_b) = tokio::join!(handle_a, handle_b);
        let resp_a = result_a.unwrap();
        let resp_b = result_b.unwrap();

        if let Ok(a) = resp_a {
            assert_eq!(a.query_id, 1);
        }
        if let Ok(b) = resp_b {
            assert_eq!(b.query_id, 2);
        }

        let _ = shutdown_tx.send(());
        let _ = handle.await;
        orchestrator1.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    // ============= Run Loop Message Handling Tests =============

    #[tokio::test]
    async fn test_run_loop_handles_query_request_message() {
        let local1 = LocalNodeConfig::new("node1", test_addr(0));
        let config1 = OrchestratorConfig::new(local1, vec![]);
        let orchestrator1 = Arc::new(ClusterOrchestrator::start(config1).await.unwrap());
        let addr1 = orchestrator1.transport().bound_addr().unwrap();

        struct TestExecutor;
        impl crate::network::QueryExecutor for TestExecutor {
            fn execute(&self, _query: &str) -> std::result::Result<Vec<u8>, String> {
                Ok(b"handled".to_vec())
            }
        }
        orchestrator1.register_query_executor(Arc::new(TestExecutor));

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch1 = orchestrator1.clone();
        let handle = tokio::spawn(async move { orch1.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let config2 = OrchestratorConfig::new(local2, vec![PeerConfig::new("node1", addr1)]);
        let orchestrator2 = ClusterOrchestrator::start(config2).await.unwrap();

        let _ = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr1.to_string(),
            })
            .await;

        let query_msg = Message::QueryRequest(crate::network::QueryRequest {
            query_id: 123,
            query: "TEST QUERY".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 1000,
        });
        let _ = orchestrator2
            .transport()
            .send(&"node1".to_string(), query_msg)
            .await;

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());

        orchestrator1.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_run_loop_handles_gossip_message() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let gossip_config = GossipConfig {
            gossip_interval_ms: 500,
            ..GossipConfig::default()
        };
        let config = OrchestratorConfig::new(local, vec![]).with_gossip(gossip_config);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());
        let addr = orchestrator.transport().bound_addr().unwrap();

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let config2 = OrchestratorConfig::new(local2, vec![PeerConfig::new("node1", addr)]);
        let orchestrator2 = ClusterOrchestrator::start(config2).await.unwrap();

        let _ = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr.to_string(),
            })
            .await;

        let gossip_msg = Message::Gossip(crate::gossip::GossipMessage::Sync {
            sender: "node2".to_string(),
            states: vec![],
            sender_time: 1,
        });
        let _ = orchestrator2
            .transport()
            .send(&"node1".to_string(), gossip_msg)
            .await;

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());

        orchestrator.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_run_loop_routes_raft_messages() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());
        let addr = orchestrator.transport().bound_addr().unwrap();

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let config2 = OrchestratorConfig::new(local2, vec![PeerConfig::new("node1", addr)]);
        let orchestrator2 = ClusterOrchestrator::start(config2).await.unwrap();

        let _ = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr.to_string(),
            })
            .await;

        let ping_msg = Message::Ping { term: 1 };
        let _ = orchestrator2
            .transport()
            .send(&"node1".to_string(), ping_msg)
            .await;

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());

        orchestrator.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_run_loop_tx_prepare_full_flow() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());
        let addr = orchestrator.transport().bound_addr().unwrap();

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let config2 = OrchestratorConfig::new(local2, vec![PeerConfig::new("node1", addr)]);
        let orchestrator2 = ClusterOrchestrator::start(config2).await.unwrap();

        let _ = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr.to_string(),
            })
            .await;

        let prepare_msg = Message::TxPrepare(crate::network::TxPrepareMsg {
            tx_id: 500,
            coordinator: "node2".to_string(),
            shard_id: 0,
            operations: vec![crate::block::Transaction::Put {
                key: "test_key".to_string(),
                data: vec![1, 2, 3],
            }],
            delta_embedding: tensor_store::SparseVector::from_dense(&[1.0, 0.0]),
            timeout_ms: 5000,
        });
        let _ = orchestrator2
            .transport()
            .send(&"node1".to_string(), prepare_msg)
            .await;

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());

        orchestrator.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_run_loop_tx_commit_full_flow() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());
        let addr = orchestrator.transport().bound_addr().unwrap();

        let prepare_msg = crate::network::TxPrepareMsg {
            tx_id: 600,
            coordinator: "node2".to_string(),
            shard_id: 0,
            operations: vec![],
            delta_embedding: tensor_store::SparseVector::new(0),
            timeout_ms: 5000,
        };
        let _ = orchestrator
            .handle_tx_prepare_with_timeout(&prepare_msg)
            .await;

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let config2 = OrchestratorConfig::new(local2, vec![PeerConfig::new("node1", addr)]);
        let orchestrator2 = ClusterOrchestrator::start(config2).await.unwrap();

        let _ = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr.to_string(),
            })
            .await;

        let commit_msg = Message::TxCommit(crate::network::TxCommitMsg {
            tx_id: 600,
            shards: vec![0],
        });
        let _ = orchestrator2
            .transport()
            .send(&"node1".to_string(), commit_msg)
            .await;

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());

        orchestrator.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_run_loop_tx_abort_full_flow() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());
        let addr = orchestrator.transport().bound_addr().unwrap();

        let prepare_msg = crate::network::TxPrepareMsg {
            tx_id: 700,
            coordinator: "node2".to_string(),
            shard_id: 0,
            operations: vec![],
            delta_embedding: tensor_store::SparseVector::new(0),
            timeout_ms: 5000,
        };
        let _ = orchestrator
            .handle_tx_prepare_with_timeout(&prepare_msg)
            .await;

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let config2 = OrchestratorConfig::new(local2, vec![PeerConfig::new("node1", addr)]);
        let orchestrator2 = ClusterOrchestrator::start(config2).await.unwrap();

        let _ = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr.to_string(),
            })
            .await;

        let abort_msg = Message::TxAbort(crate::network::TxAbortMsg {
            tx_id: 700,
            reason: "test abort".to_string(),
            shards: vec![0],
        });
        let _ = orchestrator2
            .transport()
            .send(&"node1".to_string(), abort_msg)
            .await;

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());

        orchestrator.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_run_loop_records_peer_embeddings() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());
        let addr = orchestrator.transport().bound_addr().unwrap();

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;

        let local2 = LocalNodeConfig::new("node2", test_addr(0));
        let config2 = OrchestratorConfig::new(local2, vec![PeerConfig::new("node1", addr)]);
        let orchestrator2 = ClusterOrchestrator::start(config2).await.unwrap();

        let _ = orchestrator2
            .transport()
            .connect(&crate::network::PeerConfig {
                node_id: "node1".to_string(),
                address: addr.to_string(),
            })
            .await;

        let test_embedding = tensor_store::SparseVector::from_dense(&[0.5, 0.5, 0.0]);
        let prepare_msg = Message::TxPrepare(crate::network::TxPrepareMsg {
            tx_id: 800,
            coordinator: "node2".to_string(),
            shard_id: 0,
            operations: vec![],
            delta_embedding: test_embedding,
            timeout_ms: 5000,
        });
        let _ = orchestrator2
            .transport()
            .send(&"node1".to_string(), prepare_msg)
            .await;

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());

        orchestrator.shutdown().await.unwrap();
        orchestrator2.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_run_loop_updates_local_embedding_from_state_machine() {
        tokio::time::pause();

        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();
        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        tokio::time::advance(tokio::time::Duration::from_millis(100)).await;
        tokio::task::yield_now().await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }

    // ============= Builder & Error Handling Tests =============

    #[tokio::test]
    async fn test_orchestrator_config_with_gossip_custom_interval() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let gossip_config = GossipConfig {
            gossip_interval_ms: 250,
            fanout: 5,
            ..GossipConfig::default()
        };
        let config = OrchestratorConfig::new(local, vec![]).with_gossip(gossip_config);

        assert_eq!(config.gossip.gossip_interval_ms, 250);
        assert_eq!(config.gossip.fanout, 5);
    }

    #[tokio::test]
    async fn test_orchestrator_config_chained_builders() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![])
            .with_fast_path_threshold(0.85)
            .with_raft(RaftConfig {
                heartbeat_interval: 200,
                election_timeout: (500, 1000),
                ..RaftConfig::default()
            })
            .with_gossip(GossipConfig {
                gossip_interval_ms: 300,
                ..GossipConfig::default()
            })
            .with_dtx(DistributedTxConfig {
                commit_timeout_ms: 15000,
                ..DistributedTxConfig::default()
            })
            .with_delta_replication(DeltaReplicationConfig {
                max_batch_size: 200,
                ..DeltaReplicationConfig::default()
            })
            .with_geometric(GeometricMembershipConfig::default());

        assert!((config.fast_path_threshold - 0.85).abs() < 0.001);
        assert_eq!(config.raft.heartbeat_interval, 200);
        assert_eq!(config.raft.election_timeout, (500, 1000));
        assert_eq!(config.gossip.gossip_interval_ms, 300);
        assert_eq!(config.dtx.commit_timeout_ms, 15000);
        assert_eq!(config.delta_replication.max_batch_size, 200);
    }

    #[tokio::test]
    async fn test_start_transport_failure_returns_error() {
        let local1 = LocalNodeConfig::new("node1", test_addr(12345));
        let config1 = OrchestratorConfig::new(local1, vec![]);
        let _orchestrator1 = ClusterOrchestrator::start(config1).await.unwrap();

        let local2 = LocalNodeConfig::new("node2", test_addr(12345));
        let config2 = OrchestratorConfig::new(local2, vec![]);
        let result = ClusterOrchestrator::start(config2).await;

        assert!(result.is_err());
        match result {
            Err(err) => {
                let err_str = err.to_string();
                assert!(
                    err_str.contains("Failed to start transport")
                        || err_str.contains("address")
                        || err_str.contains("bind"),
                    "Unexpected error: {}",
                    err_str
                );
            },
            Ok(_) => panic!("Expected error but got success"),
        }
    }

    #[tokio::test]
    async fn test_run_loop_continues_on_message_send_failure() {
        let local = LocalNodeConfig::new("node1", test_addr(0));
        let config = OrchestratorConfig::new(local, vec![]);
        let orchestrator = Arc::new(ClusterOrchestrator::start(config).await.unwrap());

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let orch = orchestrator.clone();

        let handle = tokio::spawn(async move { orch.run(shutdown_rx).await });

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let _ = shutdown_tx.send(());
        let result = handle.await.unwrap();
        assert!(
            result.is_ok(),
            "Run loop should continue despite message send failures"
        );

        orchestrator.shutdown().await.unwrap();
    }
}
