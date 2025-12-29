//! Cluster membership management with health checking.
//!
//! Provides:
//! - Static cluster configuration
//! - Health checking via ping/pong
//! - Failure detection with configurable thresholds
//! - Membership view with generation tracking

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::block::NodeId;
use crate::error::Result;
use crate::network::{Message, Transport};

/// Health state of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NodeHealth {
    /// Node is responding to health checks.
    Healthy,
    /// Node has missed some health checks but not enough to be failed.
    Degraded,
    /// Node has exceeded failure threshold.
    Failed,
    /// Node health is unknown (e.g., startup grace period).
    #[default]
    Unknown,
}

/// Status information for a single node.
#[derive(Debug, Clone)]
pub struct NodeStatus {
    /// Node identifier.
    pub node_id: NodeId,
    /// Node address.
    pub address: SocketAddr,
    /// Current health state.
    pub health: NodeHealth,
    /// Last successful ping time.
    pub last_ping: Option<Instant>,
    /// Round-trip time of last ping in milliseconds.
    pub rtt_ms: Option<u64>,
    /// Consecutive failure count.
    pub consecutive_failures: usize,
    /// Total successful pings.
    pub total_pings: u64,
    /// Total failed pings.
    pub total_failures: u64,
}

impl NodeStatus {
    fn new(node_id: NodeId, address: SocketAddr) -> Self {
        Self {
            node_id,
            address,
            health: NodeHealth::Unknown,
            last_ping: None,
            rtt_ms: None,
            consecutive_failures: 0,
            total_pings: 0,
            total_failures: 0,
        }
    }

    fn record_success(&mut self, rtt_ms: u64) {
        self.last_ping = Some(Instant::now());
        self.rtt_ms = Some(rtt_ms);
        self.consecutive_failures = 0;
        self.total_pings += 1;
        self.health = NodeHealth::Healthy;
    }

    fn record_failure(&mut self, failure_threshold: usize) {
        self.consecutive_failures += 1;
        self.total_failures += 1;

        if self.consecutive_failures >= failure_threshold {
            self.health = NodeHealth::Failed;
        } else if self.consecutive_failures > 0 {
            self.health = NodeHealth::Degraded;
        }
    }
}

/// View of the cluster membership at a point in time.
#[derive(Debug, Clone)]
pub struct ClusterView {
    /// All known nodes.
    pub nodes: Vec<NodeStatus>,
    /// Healthy node IDs.
    pub healthy_nodes: Vec<NodeId>,
    /// Failed node IDs.
    pub failed_nodes: Vec<NodeId>,
    /// View generation (increments on changes).
    pub generation: u64,
    /// Time this view was created.
    pub timestamp: Instant,
}

impl ClusterView {
    fn new(nodes: Vec<NodeStatus>, generation: u64) -> Self {
        let healthy_nodes: Vec<NodeId> = nodes
            .iter()
            .filter(|n| n.health == NodeHealth::Healthy)
            .map(|n| n.node_id.clone())
            .collect();

        let failed_nodes: Vec<NodeId> = nodes
            .iter()
            .filter(|n| n.health == NodeHealth::Failed)
            .map(|n| n.node_id.clone())
            .collect();

        Self {
            nodes,
            healthy_nodes,
            failed_nodes,
            generation,
            timestamp: Instant::now(),
        }
    }

    /// Check if a node is healthy.
    pub fn is_healthy(&self, node_id: &NodeId) -> bool {
        self.healthy_nodes.contains(node_id)
    }

    /// Get the number of healthy nodes.
    pub fn healthy_count(&self) -> usize {
        self.healthy_nodes.len()
    }

    /// Get the total number of nodes.
    pub fn total_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Configuration for a peer node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerNodeConfig {
    /// Node identifier.
    pub node_id: NodeId,
    /// Node address.
    pub address: SocketAddr,
}

/// Health check configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Interval between health checks in milliseconds.
    #[serde(default = "default_ping_interval")]
    pub ping_interval_ms: u64,

    /// Number of consecutive failures before marking as failed.
    #[serde(default = "default_failure_threshold")]
    pub failure_threshold: usize,

    /// Timeout for ping response in milliseconds.
    #[serde(default = "default_ping_timeout")]
    pub ping_timeout_ms: u64,

    /// Grace period on startup before enforcing failures (milliseconds).
    #[serde(default = "default_startup_grace")]
    pub startup_grace_ms: u64,
}

fn default_ping_interval() -> u64 {
    1000
}
fn default_failure_threshold() -> usize {
    3
}
fn default_ping_timeout() -> u64 {
    500
}
fn default_startup_grace() -> u64 {
    5000
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            ping_interval_ms: default_ping_interval(),
            failure_threshold: default_failure_threshold(),
            ping_timeout_ms: default_ping_timeout(),
            startup_grace_ms: default_startup_grace(),
        }
    }
}

/// Configuration for the local node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalNodeConfig {
    /// Local node identifier.
    pub node_id: NodeId,
    /// Address to bind for incoming connections.
    pub bind_address: SocketAddr,
}

/// Cluster configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Unique cluster identifier.
    pub cluster_id: String,

    /// Local node configuration.
    pub local: LocalNodeConfig,

    /// Peer nodes.
    #[serde(default)]
    pub peers: Vec<PeerNodeConfig>,

    /// Health check configuration.
    #[serde(default)]
    pub health: HealthConfig,
}

impl ClusterConfig {
    /// Create a new cluster config.
    pub fn new(cluster_id: impl Into<String>, local: LocalNodeConfig) -> Self {
        Self {
            cluster_id: cluster_id.into(),
            local,
            peers: Vec::new(),
            health: HealthConfig::default(),
        }
    }

    /// Add a peer node.
    pub fn with_peer(mut self, node_id: impl Into<NodeId>, address: SocketAddr) -> Self {
        self.peers.push(PeerNodeConfig {
            node_id: node_id.into(),
            address,
        });
        self
    }

    /// Set health configuration.
    pub fn with_health(mut self, health: HealthConfig) -> Self {
        self.health = health;
        self
    }
}

/// Callback trait for membership changes.
pub trait MembershipCallback: Send + Sync {
    /// Called when a node's health state changes.
    fn on_health_change(&self, node_id: &NodeId, old_health: NodeHealth, new_health: NodeHealth);

    /// Called when the cluster view changes.
    fn on_view_change(&self, view: &ClusterView);
}

/// Membership manager for cluster health tracking.
pub struct MembershipManager {
    /// Cluster configuration.
    config: ClusterConfig,

    /// Node status map.
    nodes: RwLock<HashMap<NodeId, NodeStatus>>,

    /// Current view generation.
    generation: AtomicU64,

    /// Transport for sending pings.
    transport: Arc<dyn Transport>,

    /// Registered callbacks.
    callbacks: RwLock<Vec<Arc<dyn MembershipCallback>>>,

    /// Shutdown signal.
    shutdown_tx: broadcast::Sender<()>,

    /// Start time for grace period.
    start_time: Instant,

    /// Running flag.
    running: RwLock<bool>,
}

impl MembershipManager {
    /// Create a new membership manager.
    pub fn new(config: ClusterConfig, transport: Arc<dyn Transport>) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);

        let mut nodes = HashMap::new();
        for peer in &config.peers {
            nodes.insert(
                peer.node_id.clone(),
                NodeStatus::new(peer.node_id.clone(), peer.address),
            );
        }

        Self {
            config,
            nodes: RwLock::new(nodes),
            generation: AtomicU64::new(0),
            transport,
            callbacks: RwLock::new(Vec::new()),
            shutdown_tx,
            start_time: Instant::now(),
            running: RwLock::new(false),
        }
    }

    /// Register a callback for membership changes.
    pub fn register_callback(&self, callback: Arc<dyn MembershipCallback>) {
        self.callbacks.write().push(callback);
    }

    /// Get the current cluster view.
    pub fn view(&self) -> ClusterView {
        let nodes = self.nodes.read();
        let statuses: Vec<NodeStatus> = nodes.values().cloned().collect();
        ClusterView::new(statuses, self.generation.load(Ordering::SeqCst))
    }

    /// Get the status of a specific node.
    pub fn node_status(&self, node_id: &NodeId) -> Option<NodeStatus> {
        self.nodes.read().get(node_id).cloned()
    }

    /// Check if we're still in the startup grace period.
    pub fn in_grace_period(&self) -> bool {
        self.start_time.elapsed() < Duration::from_millis(self.config.health.startup_grace_ms)
    }

    /// Get the local node ID.
    pub fn local_id(&self) -> &NodeId {
        &self.config.local.node_id
    }

    /// Get the cluster ID.
    pub fn cluster_id(&self) -> &str {
        &self.config.cluster_id
    }

    /// Get all peer node IDs.
    pub fn peer_ids(&self) -> Vec<NodeId> {
        self.nodes.read().keys().cloned().collect()
    }

    /// Start the health checking loop.
    pub async fn start(&self) -> Result<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Ok(());
            }
            *running = true;
        }

        // Initial connection to all peers
        for peer in &self.config.peers {
            let peer_config = crate::network::PeerConfig {
                node_id: peer.node_id.clone(),
                address: peer.address.to_string(),
            };
            // Ignore connection errors - health check will retry
            let _ = self.transport.connect(&peer_config).await;
        }

        Ok(())
    }

    /// Run a single health check round.
    pub async fn check_health(&self) -> Result<()> {
        let peer_ids = self.peer_ids();

        for peer_id in peer_ids {
            let start = Instant::now();
            let term = self.generation.load(Ordering::SeqCst);

            // Send ping
            let ping_result = tokio::time::timeout(
                Duration::from_millis(self.config.health.ping_timeout_ms),
                self.transport.send(&peer_id, Message::Ping { term }),
            )
            .await;

            let old_health = self
                .nodes
                .read()
                .get(&peer_id)
                .map(|n| n.health)
                .unwrap_or(NodeHealth::Unknown);

            match ping_result {
                Ok(Ok(())) => {
                    let rtt = start.elapsed().as_millis() as u64;
                    let mut nodes = self.nodes.write();
                    if let Some(status) = nodes.get_mut(&peer_id) {
                        status.record_success(rtt);
                    }
                },
                _ => {
                    // Timeout or send error
                    if !self.in_grace_period() {
                        let mut nodes = self.nodes.write();
                        if let Some(status) = nodes.get_mut(&peer_id) {
                            status.record_failure(self.config.health.failure_threshold);
                        }
                    }
                },
            }

            let new_health = self
                .nodes
                .read()
                .get(&peer_id)
                .map(|n| n.health)
                .unwrap_or(NodeHealth::Unknown);

            // Notify callbacks on health change
            if old_health != new_health {
                self.generation.fetch_add(1, Ordering::SeqCst);
                let callbacks = self.callbacks.read();
                for callback in callbacks.iter() {
                    callback.on_health_change(&peer_id, old_health, new_health);
                }
            }
        }

        // Notify callbacks of view change
        let view = self.view();
        let callbacks = self.callbacks.read();
        for callback in callbacks.iter() {
            callback.on_view_change(&view);
        }

        Ok(())
    }

    /// Run the health checking loop until shutdown.
    pub async fn run(&self) -> Result<()> {
        self.start().await?;

        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let interval = Duration::from_millis(self.config.health.ping_interval_ms);

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    break;
                }
                _ = tokio::time::sleep(interval) => {
                    let _ = self.check_health().await;
                }
            }
        }

        Ok(())
    }

    /// Shutdown the membership manager.
    pub fn shutdown(&self) {
        *self.running.write() = false;
        let _ = self.shutdown_tx.send(());
    }

    /// Check if the manager is running.
    pub fn is_running(&self) -> bool {
        *self.running.read()
    }

    /// Manually mark a node as failed.
    pub fn mark_failed(&self, node_id: &NodeId) {
        let old_health;
        {
            let nodes = self.nodes.read();
            old_health = nodes.get(node_id).map(|n| n.health);
        }

        if let Some(old) = old_health {
            let mut nodes = self.nodes.write();
            if let Some(status) = nodes.get_mut(node_id) {
                status.health = NodeHealth::Failed;
            }
            drop(nodes);

            if old != NodeHealth::Failed {
                self.generation.fetch_add(1, Ordering::SeqCst);
                let callbacks = self.callbacks.read();
                for callback in callbacks.iter() {
                    callback.on_health_change(node_id, old, NodeHealth::Failed);
                }
            }
        }
    }

    /// Manually mark a node as healthy.
    pub fn mark_healthy(&self, node_id: &NodeId) {
        let old_health;
        {
            let nodes = self.nodes.read();
            old_health = nodes.get(node_id).map(|n| n.health);
        }

        if let Some(old) = old_health {
            let mut nodes = self.nodes.write();
            if let Some(status) = nodes.get_mut(node_id) {
                status.health = NodeHealth::Healthy;
                status.consecutive_failures = 0;
            }
            drop(nodes);

            if old != NodeHealth::Healthy {
                self.generation.fetch_add(1, Ordering::SeqCst);
                let callbacks = self.callbacks.read();
                for callback in callbacks.iter() {
                    callback.on_health_change(node_id, old, NodeHealth::Healthy);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ChainError;
    use std::sync::atomic::AtomicUsize;

    // Mock transport for testing
    struct MockTransport {
        local_id: NodeId,
        send_count: AtomicUsize,
        fail_sends: RwLock<bool>,
    }

    impl MockTransport {
        fn new(local_id: &str) -> Self {
            Self {
                local_id: local_id.to_string(),
                send_count: AtomicUsize::new(0),
                fail_sends: RwLock::new(false),
            }
        }

        fn set_fail_sends(&self, fail: bool) {
            *self.fail_sends.write() = fail;
        }
    }

    #[async_trait::async_trait]
    impl Transport for MockTransport {
        async fn send(&self, _to: &NodeId, _msg: Message) -> Result<()> {
            self.send_count.fetch_add(1, Ordering::SeqCst);
            if *self.fail_sends.read() {
                Err(ChainError::NetworkError("mock failure".to_string()))
            } else {
                Ok(())
            }
        }

        async fn broadcast(&self, _msg: Message) -> Result<()> {
            Ok(())
        }

        async fn recv(&self) -> Result<(NodeId, Message)> {
            Err(ChainError::NetworkError("not implemented".to_string()))
        }

        async fn connect(&self, _peer: &crate::network::PeerConfig) -> Result<()> {
            Ok(())
        }

        async fn disconnect(&self, _peer_id: &NodeId) -> Result<()> {
            Ok(())
        }

        fn peers(&self) -> Vec<NodeId> {
            Vec::new()
        }

        fn local_id(&self) -> &NodeId {
            &self.local_id
        }
    }

    fn create_test_config() -> ClusterConfig {
        ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9102".parse().unwrap())
    }

    #[test]
    fn test_node_health_default() {
        assert_eq!(NodeHealth::default(), NodeHealth::Unknown);
    }

    #[test]
    fn test_node_status_new() {
        let status = NodeStatus::new("node1".to_string(), "127.0.0.1:9100".parse().unwrap());
        assert_eq!(status.node_id, "node1");
        assert_eq!(status.health, NodeHealth::Unknown);
        assert!(status.last_ping.is_none());
        assert_eq!(status.consecutive_failures, 0);
    }

    #[test]
    fn test_node_status_record_success() {
        let mut status = NodeStatus::new("node1".to_string(), "127.0.0.1:9100".parse().unwrap());
        status.consecutive_failures = 2;
        status.health = NodeHealth::Degraded;

        status.record_success(50);

        assert_eq!(status.health, NodeHealth::Healthy);
        assert_eq!(status.consecutive_failures, 0);
        assert_eq!(status.total_pings, 1);
        assert_eq!(status.rtt_ms, Some(50));
        assert!(status.last_ping.is_some());
    }

    #[test]
    fn test_node_status_record_failure() {
        let mut status = NodeStatus::new("node1".to_string(), "127.0.0.1:9100".parse().unwrap());

        // First failure - degraded
        status.record_failure(3);
        assert_eq!(status.health, NodeHealth::Degraded);
        assert_eq!(status.consecutive_failures, 1);

        // Second failure - still degraded
        status.record_failure(3);
        assert_eq!(status.health, NodeHealth::Degraded);
        assert_eq!(status.consecutive_failures, 2);

        // Third failure - failed
        status.record_failure(3);
        assert_eq!(status.health, NodeHealth::Failed);
        assert_eq!(status.consecutive_failures, 3);
        assert_eq!(status.total_failures, 3);
    }

    #[test]
    fn test_cluster_view() {
        let nodes = vec![
            {
                let mut s = NodeStatus::new("node1".to_string(), "127.0.0.1:9100".parse().unwrap());
                s.health = NodeHealth::Healthy;
                s
            },
            {
                let mut s = NodeStatus::new("node2".to_string(), "127.0.0.1:9101".parse().unwrap());
                s.health = NodeHealth::Failed;
                s
            },
            {
                let mut s = NodeStatus::new("node3".to_string(), "127.0.0.1:9102".parse().unwrap());
                s.health = NodeHealth::Healthy;
                s
            },
        ];

        let view = ClusterView::new(nodes, 5);

        assert_eq!(view.generation, 5);
        assert_eq!(view.total_count(), 3);
        assert_eq!(view.healthy_count(), 2);
        assert!(view.is_healthy(&"node1".to_string()));
        assert!(!view.is_healthy(&"node2".to_string()));
        assert!(view.is_healthy(&"node3".to_string()));
        assert_eq!(view.healthy_nodes.len(), 2);
        assert_eq!(view.failed_nodes.len(), 1);
    }

    #[test]
    fn test_health_config_default() {
        let config = HealthConfig::default();
        assert_eq!(config.ping_interval_ms, 1000);
        assert_eq!(config.failure_threshold, 3);
        assert_eq!(config.ping_timeout_ms, 500);
        assert_eq!(config.startup_grace_ms, 5000);
    }

    #[test]
    fn test_cluster_config_builder() {
        let config = ClusterConfig::new(
            "my-cluster",
            LocalNodeConfig {
                node_id: "local".to_string(),
                bind_address: "0.0.0.0:9100".parse().unwrap(),
            },
        )
        .with_peer("peer1", "10.0.0.1:9100".parse().unwrap())
        .with_peer("peer2", "10.0.0.2:9100".parse().unwrap())
        .with_health(HealthConfig {
            ping_interval_ms: 500,
            ..Default::default()
        });

        assert_eq!(config.cluster_id, "my-cluster");
        assert_eq!(config.local.node_id, "local");
        assert_eq!(config.peers.len(), 2);
        assert_eq!(config.health.ping_interval_ms, 500);
    }

    #[tokio::test]
    async fn test_membership_manager_creation() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config.clone(), transport);

        assert_eq!(manager.local_id(), "node1");
        assert_eq!(manager.cluster_id(), "test-cluster");
        assert_eq!(manager.peer_ids().len(), 2);
    }

    #[tokio::test]
    async fn test_membership_manager_view() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        let view = manager.view();
        assert_eq!(view.total_count(), 2);
        // All nodes start as Unknown
        assert_eq!(view.healthy_count(), 0);
    }

    #[tokio::test]
    async fn test_membership_manager_node_status() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        let status = manager.node_status(&"node2".to_string());
        assert!(status.is_some());
        let status = status.unwrap();
        assert_eq!(status.node_id, "node2");
        assert_eq!(status.health, NodeHealth::Unknown);

        let missing = manager.node_status(&"nonexistent".to_string());
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn test_membership_manager_start() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        assert!(!manager.is_running());
        manager.start().await.unwrap();
        assert!(manager.is_running());

        // Starting again should be idempotent
        manager.start().await.unwrap();
        assert!(manager.is_running());
    }

    #[tokio::test]
    async fn test_membership_manager_shutdown() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        manager.start().await.unwrap();
        assert!(manager.is_running());

        manager.shutdown();
        assert!(!manager.is_running());
    }

    #[tokio::test]
    async fn test_membership_manager_mark_failed() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        // Initially unknown
        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert_eq!(status.health, NodeHealth::Unknown);

        // Mark as failed
        manager.mark_failed(&"node2".to_string());

        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert_eq!(status.health, NodeHealth::Failed);
    }

    #[tokio::test]
    async fn test_membership_manager_mark_healthy() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        // Mark as failed first
        manager.mark_failed(&"node2".to_string());

        // Mark as healthy
        manager.mark_healthy(&"node2".to_string());

        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert_eq!(status.health, NodeHealth::Healthy);
        assert_eq!(status.consecutive_failures, 0);
    }

    #[tokio::test]
    async fn test_membership_manager_check_health_success() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0; // Disable grace period

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport.clone());

        // Run health check (sends will succeed)
        manager.check_health().await.unwrap();

        // Node should be healthy
        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert_eq!(status.health, NodeHealth::Healthy);
        assert!(transport.send_count.load(Ordering::SeqCst) > 0);
    }

    #[tokio::test]
    async fn test_membership_manager_check_health_failure() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;
        health.failure_threshold = 1; // Fail after 1 failure

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        transport.set_fail_sends(true);
        let manager = MembershipManager::new(config, transport);

        // Run health check (sends will fail)
        manager.check_health().await.unwrap();

        // Node should be failed
        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert_eq!(status.health, NodeHealth::Failed);
    }

    #[tokio::test]
    async fn test_membership_manager_grace_period() {
        let health = HealthConfig {
            startup_grace_ms: 60000, // Long grace period
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
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        transport.set_fail_sends(true);
        let manager = MembershipManager::new(config, transport);

        assert!(manager.in_grace_period());

        // Run health check during grace period
        manager.check_health().await.unwrap();

        // Node should still be unknown (grace period protects it)
        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert_eq!(status.health, NodeHealth::Unknown);
    }

    #[tokio::test]
    async fn test_membership_callback() {
        use std::sync::Mutex;

        struct TestCallback {
            health_changes: Mutex<Vec<(NodeId, NodeHealth, NodeHealth)>>,
            view_changes: AtomicUsize,
        }

        impl MembershipCallback for TestCallback {
            fn on_health_change(
                &self,
                node_id: &NodeId,
                old_health: NodeHealth,
                new_health: NodeHealth,
            ) {
                self.health_changes
                    .lock()
                    .unwrap()
                    .push((node_id.clone(), old_health, new_health));
            }

            fn on_view_change(&self, _view: &ClusterView) {
                self.view_changes.fetch_add(1, Ordering::SeqCst);
            }
        }

        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        let callback = Arc::new(TestCallback {
            health_changes: Mutex::new(Vec::new()),
            view_changes: AtomicUsize::new(0),
        });

        manager.register_callback(callback.clone());

        // Trigger a health change
        manager.mark_failed(&"node2".to_string());

        let changes = callback.health_changes.lock().unwrap();
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].0, "node2");
        assert_eq!(changes[0].1, NodeHealth::Unknown);
        assert_eq!(changes[0].2, NodeHealth::Failed);
    }

    #[test]
    fn test_cluster_config_debug() {
        let config = create_test_config();
        let debug = format!("{:?}", config);
        assert!(debug.contains("ClusterConfig"));
        assert!(debug.contains("test-cluster"));
    }

    #[test]
    fn test_node_health_equality() {
        assert_eq!(NodeHealth::Healthy, NodeHealth::Healthy);
        assert_ne!(NodeHealth::Healthy, NodeHealth::Failed);
    }

    #[test]
    fn test_cluster_view_debug() {
        let view = ClusterView::new(vec![], 0);
        let debug = format!("{:?}", view);
        assert!(debug.contains("ClusterView"));
    }

    #[tokio::test]
    async fn test_check_health_invokes_callback_on_health_change() {
        use std::sync::Mutex;

        struct TestCallback {
            health_changes: Mutex<Vec<(NodeId, NodeHealth, NodeHealth)>>,
            view_changes: AtomicUsize,
        }

        impl MembershipCallback for TestCallback {
            fn on_health_change(
                &self,
                node_id: &NodeId,
                old_health: NodeHealth,
                new_health: NodeHealth,
            ) {
                self.health_changes
                    .lock()
                    .unwrap()
                    .push((node_id.clone(), old_health, new_health));
            }

            fn on_view_change(&self, _view: &ClusterView) {
                self.view_changes.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;
        health.failure_threshold = 1;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport.clone());

        let callback = Arc::new(TestCallback {
            health_changes: Mutex::new(Vec::new()),
            view_changes: AtomicUsize::new(0),
        });

        manager.register_callback(callback.clone());

        // Run check_health - should trigger callback on health change (Unknown -> Healthy)
        manager.check_health().await.unwrap();

        // Verify callbacks were invoked
        let changes = callback.health_changes.lock().unwrap();
        assert_eq!(changes.len(), 1, "Should have one health change");
        assert_eq!(changes[0].0, "node2");
        assert_eq!(changes[0].1, NodeHealth::Unknown);
        assert_eq!(changes[0].2, NodeHealth::Healthy);

        // View change should have been called
        assert!(
            callback.view_changes.load(Ordering::SeqCst) > 0,
            "on_view_change should be called"
        );
    }

    #[tokio::test]
    async fn test_run_method_with_shutdown() {
        let mut health = HealthConfig::default();
        health.ping_interval_ms = 50;
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = Arc::new(MembershipManager::new(config, transport.clone()));

        let manager_clone = manager.clone();
        let run_handle = tokio::spawn(async move { manager_clone.run().await });

        // Let it run for a bit
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should have done some health checks
        assert!(transport.send_count.load(Ordering::SeqCst) > 0);

        // Shutdown
        manager.shutdown();

        // run() should complete
        let result = tokio::time::timeout(Duration::from_secs(1), run_handle).await;
        assert!(result.is_ok(), "run() should complete after shutdown");
        assert!(result.unwrap().is_ok());
    }

    #[tokio::test]
    async fn test_check_health_view_change_callback() {
        use std::sync::Mutex;

        struct TestCallback {
            view_snapshots: Mutex<Vec<ClusterView>>,
        }

        impl MembershipCallback for TestCallback {
            fn on_health_change(&self, _: &NodeId, _: NodeHealth, _: NodeHealth) {}

            fn on_view_change(&self, view: &ClusterView) {
                self.view_snapshots.lock().unwrap().push(view.clone());
            }
        }

        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9100".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9101".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        let callback = Arc::new(TestCallback {
            view_snapshots: Mutex::new(Vec::new()),
        });

        manager.register_callback(callback.clone());

        // Run multiple health checks
        manager.check_health().await.unwrap();
        manager.check_health().await.unwrap();

        let snapshots = callback.view_snapshots.lock().unwrap();
        assert_eq!(
            snapshots.len(),
            2,
            "Should have 2 view change notifications"
        );
    }
}
