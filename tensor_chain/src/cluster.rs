//! Cluster orchestration for distributed Neumann nodes.
//!
//! Provides a unified API for starting and managing cluster nodes with:
//! - TCP transport initialization
//! - Membership management with geometric peer scoring
//! - Raft consensus with persistence
//! - State machine for block application

use std::net::SocketAddr;
use std::sync::Arc;

use tensor_store::TensorStore;
use tokio::sync::broadcast;

use crate::block::NodeId;
use crate::chain::Chain;
use crate::error::{ChainError, Result};
use crate::geometric_membership::{GeometricMembershipConfig, GeometricMembershipManager};
use crate::membership::{ClusterConfig, MembershipManager};
use crate::network::{Message, QueryExecutor, QueryRequest, QueryResponse, Transport};
use crate::raft::{RaftConfig, RaftNode};
use crate::state_machine::TensorStateMachine;
use crate::tcp::{TcpTransport, TcpTransportConfig};

use parking_lot::RwLock;

use graph_engine::GraphEngine;

/// Configuration for a local cluster node.
#[derive(Debug, Clone)]
pub struct LocalNodeConfig {
    /// Unique node identifier.
    pub node_id: NodeId,
    /// Address to bind for incoming connections.
    pub bind_address: SocketAddr,
}

impl LocalNodeConfig {
    /// Create a new local node config.
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
    /// Create a new peer config.
    pub fn new(node_id: impl Into<NodeId>, address: SocketAddr) -> Self {
        Self {
            node_id: node_id.into(),
            address,
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
    /// Fast-path similarity threshold for state machine.
    pub fast_path_threshold: f32,
}

impl OrchestratorConfig {
    /// Create a new orchestrator config.
    pub fn new(local: LocalNodeConfig, peers: Vec<PeerConfig>) -> Self {
        Self {
            local,
            peers,
            raft: RaftConfig::default(),
            geometric: GeometricMembershipConfig::default(),
            fast_path_threshold: 0.95,
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

        // 7. Create Raft node, loading persisted state
        let peer_ids: Vec<NodeId> = config.peers.iter().map(|p| p.node_id.clone()).collect();

        let raft = Arc::new(RaftNode::with_store(
            config.local.node_id.clone(),
            peer_ids,
            transport.clone(),
            config.raft.clone(),
            &store,
        ));

        // 8. Create chain storage
        let graph = Arc::new(GraphEngine::with_store(store.clone()));
        let chain = Arc::new(Chain::new(graph, config.local.node_id.clone()));

        // Initialize chain (creates genesis if needed)
        chain.initialize()?;

        // 9. Create state machine with store for transaction application
        let state_machine = Arc::new(TensorStateMachine::with_threshold(
            chain.clone(),
            raft.clone(),
            store.clone(),
            config.fast_path_threshold,
        ));

        Ok(Self {
            config,
            transport,
            membership,
            geometric,
            raft,
            chain,
            state_machine,
            store,
            query_executor: RwLock::new(None),
            local_shard_id: 0,
        })
    }

    /// Register a query executor for handling distributed queries.
    ///
    /// The executor will be called when QueryRequest messages are received.
    pub fn register_query_executor(&self, executor: Arc<dyn QueryExecutor>) {
        *self.query_executor.write() = Some(executor);
    }

    /// Handle an incoming query request.
    fn handle_query_request(&self, from: &NodeId, request: &QueryRequest) -> Option<Message> {
        let executor = self.query_executor.read();
        let executor = executor.as_ref()?;

        let start = std::time::Instant::now();

        let (result, success, error) = match executor.execute(&request.query) {
            Ok(bytes) => (bytes, true, None),
            Err(e) => (Vec::new(), false, Some(e)),
        };

        let execution_time_us = start.elapsed().as_micros() as u64;

        // Log query execution
        tracing::debug!(
            "Executed query from {} in {}us: success={}",
            from,
            execution_time_us,
            success
        );

        Some(Message::QueryResponse(QueryResponse {
            query_id: request.query_id,
            shard_id: self.local_shard_id,
            result,
            execution_time_us,
            success,
            error,
        }))
    }

    /// Run the node until shutdown signal.
    ///
    /// Spawns background tasks for:
    /// - Raft tick loop
    /// - Membership health checks
    /// - State machine apply loop
    /// - Query message handling
    pub async fn run(&self, mut shutdown: broadcast::Receiver<()>) -> Result<()> {
        let raft = self.raft.clone();
        let state_machine = self.state_machine.clone();
        let transport = self.transport.clone();

        // Main loop
        loop {
            tokio::select! {
                _ = shutdown.recv() => {
                    break;
                }
                // Try to receive a message with timeout
                recv_result = tokio::time::timeout(
                    tokio::time::Duration::from_millis(10),
                    transport.recv()
                ) => {
                    if let Ok(Ok((from, msg))) = recv_result {
                        // Handle the message
                        match &msg {
                            Message::QueryRequest(request) => {
                                if let Some(response) = self.handle_query_request(&from, request) {
                                    let _ = transport.send(&from, response).await;
                                }
                            }
                            // Raft messages are handled by RaftNode internally
                            _ => {
                                // Let Raft handle other messages
                                let _ = raft.handle_message_async(&from, msg).await;
                            }
                        }
                    }
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(50)) => {
                    // Tick Raft
                    raft.tick_async().await?;

                    // Apply any committed entries
                    let _ = state_machine.apply_committed();
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
            if let Ok(Ok(Some((from, Message::QueryResponse(response))))) = tokio::time::timeout(
                Duration::from_millis(50),
                self.transport.receive_one(),
            )
            .await
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
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

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
}
