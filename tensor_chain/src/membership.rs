//! Cluster membership management with health checking.
//!
//! Provides:
//! - Static cluster configuration
//! - Health checking via ping/pong
//! - Failure detection with configurable thresholds
//! - Membership view with generation tracking

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tensor_store::SparseVector;
use tokio::sync::broadcast;

use crate::{
    block::NodeId,
    error::Result,
    network::{Message, Transport},
};

/// Health state of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
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

/// Partition status based on reachable nodes.
///
/// "Split-brain" technically refers to two leaders existing simultaneously.
/// Since quorum tracking prevents that, these states indicate quorum availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum PartitionStatus {
    /// Majority of nodes reachable - safe to accept writes.
    QuorumReachable,
    /// Less than majority reachable - must reject writes.
    QuorumLost,
    /// Exact 50/50 split (even cluster) - neither side has quorum.
    Stalemate,
    /// Status unknown (during startup grace period).
    #[default]
    Unknown,
}

/// Membership health metrics.
#[derive(Debug, Default)]
pub struct MembershipStats {
    /// Total health checks performed.
    pub health_checks: AtomicU64,
    /// Health checks that failed.
    pub health_check_failures: AtomicU64,
    /// Ping round-trip timing (microseconds).
    pub ping_timing: crate::metrics::TimingStats,
    /// Number of partition events detected.
    pub partition_events: AtomicU64,
    /// Number of times quorum was lost.
    pub quorum_lost_events: AtomicU64,
}

impl MembershipStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> MembershipStatsSnapshot {
        MembershipStatsSnapshot {
            health_checks: self.health_checks.load(Ordering::Relaxed),
            health_check_failures: self.health_check_failures.load(Ordering::Relaxed),
            ping_timing: self.ping_timing.snapshot(),
            partition_events: self.partition_events.load(Ordering::Relaxed),
            quorum_lost_events: self.quorum_lost_events.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of membership statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MembershipStatsSnapshot {
    pub health_checks: u64,
    pub health_check_failures: u64,
    pub ping_timing: crate::metrics::TimingSnapshot,
    pub partition_events: u64,
    pub quorum_lost_events: u64,
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
    /// Consecutive success count (for heal detection).
    pub consecutive_successes: u32,
    /// When the node entered Failed state (for partition duration tracking).
    pub partition_start: Option<Instant>,
    /// Total successful pings.
    pub total_pings: u64,
    /// Total failed pings.
    pub total_failures: u64,
    /// Node's state embedding for geometric routing.
    pub state_embedding: Option<SparseVector>,
    /// When the state embedding was last updated.
    pub embedding_updated: Option<Instant>,
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
            consecutive_successes: 0,
            partition_start: None,
            total_pings: 0,
            total_failures: 0,
            state_embedding: None,
            embedding_updated: None,
        }
    }

    pub fn update_embedding(&mut self, embedding: SparseVector) {
        self.state_embedding = Some(embedding);
        self.embedding_updated = Some(Instant::now());
    }

    fn record_success(&mut self, rtt_ms: u64) {
        let was_failed = self.health == NodeHealth::Failed;
        self.last_ping = Some(Instant::now());
        self.rtt_ms = Some(rtt_ms);
        self.consecutive_failures = 0;
        self.consecutive_successes += 1;
        self.total_pings += 1;
        self.health = NodeHealth::Healthy;

        // Clear partition_start when node becomes healthy again
        if was_failed {
            // Keep partition_start until heal is confirmed at threshold
            // (caller will clear it after processing heal callback)
        }
    }

    fn record_failure(&mut self, failure_threshold: usize) {
        let was_not_failed = self.health != NodeHealth::Failed;
        self.consecutive_failures += 1;
        self.consecutive_successes = 0; // Reset heal detection counter
        self.total_failures += 1;

        if self.consecutive_failures >= failure_threshold {
            self.health = NodeHealth::Failed;
            // Track when node first entered Failed state
            if was_not_failed && self.partition_start.is_none() {
                self.partition_start = Some(Instant::now());
            }
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
    /// Current partition status.
    pub partition_status: PartitionStatus,
}

impl ClusterView {
    #[allow(dead_code)]
    fn new(nodes: Vec<NodeStatus>, generation: u64) -> Self {
        Self::with_partition_status(nodes, generation, PartitionStatus::Unknown)
    }

    fn with_partition_status(
        nodes: Vec<NodeStatus>,
        generation: u64,
        partition_status: PartitionStatus,
    ) -> Self {
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
            partition_status,
        }
    }

    pub fn is_healthy(&self, node_id: &NodeId) -> bool {
        self.healthy_nodes.contains(node_id)
    }

    pub fn healthy_count(&self) -> usize {
        self.healthy_nodes.len()
    }

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
    pub fn new(cluster_id: impl Into<String>, local: LocalNodeConfig) -> Self {
        Self {
            cluster_id: cluster_id.into(),
            local,
            peers: Vec::new(),
            health: HealthConfig::default(),
        }
    }

    pub fn with_peer(mut self, node_id: impl Into<NodeId>, address: SocketAddr) -> Self {
        self.peers.push(PeerNodeConfig {
            node_id: node_id.into(),
            address,
        });
        self
    }

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

    /// Called when previously partitioned nodes have healed and are reachable again.
    ///
    /// This callback is invoked when a node that was in Failed state has achieved
    /// the required number of consecutive successful pings (heal confirmation threshold).
    ///
    /// # Arguments
    /// * `healed_nodes` - List of node IDs that have healed
    /// * `partition_duration_ms` - Duration (in milliseconds) the node was partitioned
    fn on_partition_heal(&self, healed_nodes: &[NodeId], partition_duration_ms: u64) {
        // Default no-op implementation for backwards compatibility
        let _ = (healed_nodes, partition_duration_ms);
    }
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

    /// Statistics.
    pub stats: MembershipStats,
}

impl MembershipManager {
    pub fn new(config: ClusterConfig, transport: Arc<dyn Transport>) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);

        let mut nodes = HashMap::new();

        // Add local node (always healthy)
        let mut local_status =
            NodeStatus::new(config.local.node_id.clone(), config.local.bind_address);
        local_status.health = NodeHealth::Healthy;
        nodes.insert(config.local.node_id.clone(), local_status);

        // Add peer nodes
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
            stats: MembershipStats::new(),
        }
    }

    pub fn register_callback(&self, callback: Arc<dyn MembershipCallback>) {
        self.callbacks.write().push(callback);
    }

    pub fn view(&self) -> ClusterView {
        let nodes = self.nodes.read();
        let statuses: Vec<NodeStatus> = nodes.values().cloned().collect();
        let partition_status = self.compute_partition_status_inner(&statuses);
        ClusterView::with_partition_status(
            statuses,
            self.generation.load(Ordering::SeqCst),
            partition_status,
        )
    }

    pub fn partition_status(&self) -> PartitionStatus {
        if self.in_grace_period() {
            return PartitionStatus::Unknown;
        }

        let nodes = self.nodes.read();
        let statuses: Vec<NodeStatus> = nodes.values().cloned().collect();
        self.compute_partition_status_inner(&statuses)
    }

    fn compute_partition_status_inner(&self, statuses: &[NodeStatus]) -> PartitionStatus {
        if self.in_grace_period() {
            return PartitionStatus::Unknown;
        }

        let total = statuses.len();
        if total == 0 {
            return PartitionStatus::Unknown;
        }

        let healthy = statuses
            .iter()
            .filter(|s| s.health == NodeHealth::Healthy)
            .count();

        let quorum = (total / 2) + 1;

        if healthy >= quorum {
            PartitionStatus::QuorumReachable
        } else if total % 2 == 0 && healthy * 2 == total {
            // Exact 50/50 split in even-sized cluster
            PartitionStatus::Stalemate
        } else {
            PartitionStatus::QuorumLost
        }
    }

    /// Check if it's safe to perform write operations.
    ///
    /// Returns false during grace period or when quorum is lost.
    pub fn is_safe_to_write(&self) -> bool {
        self.partition_status() == PartitionStatus::QuorumReachable
    }

    pub fn node_status(&self, node_id: &NodeId) -> Option<NodeStatus> {
        self.nodes.read().get(node_id).cloned()
    }

    pub fn in_grace_period(&self) -> bool {
        self.start_time.elapsed() < Duration::from_millis(self.config.health.startup_grace_ms)
    }

    pub fn local_id(&self) -> &NodeId {
        &self.config.local.node_id
    }

    pub fn cluster_id(&self) -> &str {
        &self.config.cluster_id
    }

    pub fn peer_ids(&self) -> Vec<NodeId> {
        let local_id = &self.config.local.node_id;
        self.nodes
            .read()
            .keys()
            .filter(|k| *k != local_id)
            .cloned()
            .collect()
    }

    /// Detect nodes that have healed from partition.
    ///
    /// Returns a list of (node_id, partition_duration_ms) for nodes that:
    /// 1. Were previously in Failed state (have partition_start set)
    /// 2. Are now Healthy
    /// 3. Have achieved the required consecutive successes threshold
    ///
    /// # Arguments
    /// * `threshold` - Number of consecutive successes required to confirm heal
    pub fn detect_healed_nodes(&self, threshold: u32) -> Vec<(NodeId, u64)> {
        let nodes = self.nodes.read();
        let mut healed = Vec::new();

        for (node_id, status) in nodes.iter() {
            // Check if this node has healed from a partition
            if status.health == NodeHealth::Healthy
                && status.partition_start.is_some()
                && status.consecutive_successes >= threshold
            {
                let partition_duration_ms = status
                    .partition_start
                    .map(|start| start.elapsed().as_millis() as u64)
                    .unwrap_or(0);

                tracing::info!(
                    node_id = %node_id,
                    partition_duration_ms = partition_duration_ms,
                    consecutive_successes = status.consecutive_successes,
                    "Node healed from partition"
                );

                healed.push((node_id.clone(), partition_duration_ms));
            }
        }

        healed
    }

    /// Clear the partition state for a healed node.
    ///
    /// Should be called after processing the heal callback to reset tracking state.
    pub fn clear_partition_state(&self, node_id: &NodeId) {
        let mut nodes = self.nodes.write();
        if let Some(status) = nodes.get_mut(node_id) {
            status.partition_start = None;
            status.consecutive_successes = 0;
        }
    }

    /// Clear partition state for multiple nodes.
    pub fn clear_partition_states(&self, node_ids: &[NodeId]) {
        let mut nodes = self.nodes.write();
        for node_id in node_ids {
            if let Some(status) = nodes.get_mut(node_id) {
                status.partition_start = None;
                status.consecutive_successes = 0;
            }
        }
    }

    pub async fn start(&self) -> Result<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Ok(());
            }
            *running = true;
        }

        tracing::info!(
            node_id = %self.config.local.node_id,
            cluster_id = %self.config.cluster_id,
            peers = self.config.peers.len(),
            "Membership manager started"
        );

        // Initial connection to all peers
        for peer in &self.config.peers {
            let peer_config = crate::network::PeerConfig {
                node_id: peer.node_id.clone(),
                address: peer.address.to_string(),
            };
            // Connection errors are expected - health check will retry
            if let Err(e) = self.transport.connect(&peer_config).await {
                tracing::debug!(peer = %peer.node_id, error = %e, "failed to connect to peer during bootstrap");
            }
        }

        Ok(())
    }

    pub async fn check_health(&self) -> Result<()> {
        let peer_ids = self.peer_ids();
        self.stats.health_checks.fetch_add(1, Ordering::Relaxed);

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
                    self.stats.ping_timing.record(rtt);
                    tracing::debug!(
                        peer_id = %peer_id,
                        rtt_ms = rtt,
                        "Health check succeeded"
                    );
                    let mut nodes = self.nodes.write();
                    if let Some(status) = nodes.get_mut(&peer_id) {
                        status.record_success(rtt);
                    }
                },
                _ => {
                    // Timeout or send error
                    self.stats
                        .health_check_failures
                        .fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(
                        peer_id = %peer_id,
                        "Health check failed"
                    );
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
                tracing::info!(
                    peer_id = %peer_id,
                    old_health = ?old_health,
                    new_health = ?new_health,
                    "Node health state changed"
                );
                self.generation.fetch_add(1, Ordering::SeqCst);
                let callbacks = self.callbacks.read();
                for callback in callbacks.iter() {
                    callback.on_health_change(&peer_id, old_health, new_health);
                }
            }
        }

        // Check partition status and log if changed
        let current_partition_status = self.partition_status();
        let view = self.view();

        if current_partition_status == PartitionStatus::QuorumLost {
            self.stats
                .quorum_lost_events
                .fetch_add(1, Ordering::Relaxed);
            tracing::error!(
                healthy = view.healthy_count(),
                total = view.total_count(),
                "QUORUM LOST - rejecting writes"
            );
        }

        // Notify callbacks of view change
        let callbacks = self.callbacks.read();
        for callback in callbacks.iter() {
            callback.on_view_change(&view);
        }

        Ok(())
    }

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

    pub fn shutdown(&self) {
        tracing::info!(
            node_id = %self.config.local.node_id,
            "Membership manager shutting down"
        );
        *self.running.write() = false;
        let _ = self.shutdown_tx.send(());
    }

    pub fn is_running(&self) -> bool {
        *self.running.read()
    }

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
    use std::sync::atomic::AtomicUsize;

    use super::*;
    use crate::error::ChainError;

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
        // 3 nodes: local + 2 peers
        assert_eq!(view.total_count(), 3);
        // Local node starts as Healthy, peers as Unknown
        assert_eq!(view.healthy_count(), 1);
        assert!(view.is_healthy(&"node1".to_string()));
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

    // ========== PartitionStatus Tests ==========

    #[test]
    fn test_partition_status_default() {
        let status = PartitionStatus::default();
        assert_eq!(status, PartitionStatus::Unknown);
    }

    #[test]
    fn test_partition_status_equality() {
        assert_eq!(
            PartitionStatus::QuorumReachable,
            PartitionStatus::QuorumReachable
        );
        assert_ne!(
            PartitionStatus::QuorumReachable,
            PartitionStatus::QuorumLost
        );
    }

    #[test]
    fn test_partition_status_clone() {
        let status = PartitionStatus::Stalemate;
        let cloned = status;
        assert_eq!(status, cloned);
    }

    #[test]
    fn test_partition_status_debug() {
        let status = PartitionStatus::QuorumLost;
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("QuorumLost"));
    }

    #[tokio::test]
    async fn test_partition_status_during_grace_period() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 60_000; // Long grace period

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9300".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9301".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        // During grace period, status should be Unknown
        assert_eq!(manager.partition_status(), PartitionStatus::Unknown);
        assert!(!manager.is_safe_to_write());
    }

    #[tokio::test]
    async fn test_partition_status_quorum_reachable() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0; // No grace period

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9400".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9401".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9402".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        // Local node is always healthy, need 2/3 for quorum
        // With just local healthy, we have 1/3 - not quorum
        let status = manager.partition_status();

        // Without health checks running, other nodes are Unknown
        // 1 Healthy (self) out of 3 = QuorumLost
        assert_eq!(status, PartitionStatus::QuorumLost);
    }

    #[test]
    fn test_cluster_view_has_partition_status() {
        // Create a view manually to test the field
        let status = NodeStatus::new("node1".to_string(), "127.0.0.1:9000".parse().unwrap());
        let view =
            ClusterView::with_partition_status(vec![status], 1, PartitionStatus::QuorumReachable);

        assert_eq!(view.partition_status, PartitionStatus::QuorumReachable);
    }

    // ========== MembershipStats Tests ==========

    #[test]
    fn test_membership_stats_new() {
        let stats = MembershipStats::new();
        assert_eq!(stats.health_checks.load(Ordering::Relaxed), 0);
        assert_eq!(stats.partition_events.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_membership_stats_snapshot() {
        let stats = MembershipStats::new();
        stats.health_checks.fetch_add(10, Ordering::Relaxed);
        stats.health_check_failures.fetch_add(2, Ordering::Relaxed);
        stats.partition_events.fetch_add(1, Ordering::Relaxed);
        stats.ping_timing.record(100);
        stats.ping_timing.record(200);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.health_checks, 10);
        assert_eq!(snapshot.health_check_failures, 2);
        assert_eq!(snapshot.partition_events, 1);
        assert_eq!(snapshot.ping_timing.count, 2);
        assert_eq!(snapshot.ping_timing.avg_us, 150.0);
    }

    #[test]
    fn test_membership_stats_default() {
        let stats = MembershipStats::default();
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.health_checks, 0);
        assert_eq!(snapshot.ping_timing.count, 0);
    }

    #[tokio::test]
    async fn test_membership_manager_has_stats() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        // Manager should have stats field
        assert_eq!(manager.stats.health_checks.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_is_safe_to_write_false_during_grace() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 60_000;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9500".parse().unwrap(),
            },
        )
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        assert!(!manager.is_safe_to_write());
    }

    #[test]
    fn test_compute_partition_status_stalemate() {
        // Test the stalemate (50/50 split) detection
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        // 4 node cluster - if 2 are healthy, we have stalemate
        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9600".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9601".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9602".parse().unwrap())
        .with_peer("node4", "127.0.0.1:9603".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        // Initially, only local node is healthy (1/4)
        // This is QuorumLost, not Stalemate
        let status = manager.partition_status();
        assert_eq!(status, PartitionStatus::QuorumLost);
    }

    // ========== Priority 1: Partition Healing Tests ==========

    #[test]
    fn test_detect_healed_nodes_returns_healed_node() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9700".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9701".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        {
            let mut nodes = manager.nodes.write();
            let status = nodes.get_mut(&"node2".to_string()).unwrap();
            status.health = NodeHealth::Healthy;
            status.partition_start = Some(Instant::now() - Duration::from_secs(10));
            status.consecutive_successes = 5;
        }

        let healed = manager.detect_healed_nodes(3);
        assert_eq!(healed.len(), 1);
        assert_eq!(healed[0].0, "node2");
        assert!(healed[0].1 >= 10_000);
    }

    #[test]
    fn test_detect_healed_nodes_threshold_not_met() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9702".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9703".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        {
            let mut nodes = manager.nodes.write();
            let status = nodes.get_mut(&"node2".to_string()).unwrap();
            status.health = NodeHealth::Healthy;
            status.partition_start = Some(Instant::now());
            status.consecutive_successes = 2;
        }

        let healed = manager.detect_healed_nodes(5);
        assert!(healed.is_empty());
    }

    #[test]
    fn test_detect_healed_nodes_never_partitioned() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9704".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9705".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        {
            let mut nodes = manager.nodes.write();
            let status = nodes.get_mut(&"node2".to_string()).unwrap();
            status.health = NodeHealth::Healthy;
            status.partition_start = None;
            status.consecutive_successes = 10;
        }

        let healed = manager.detect_healed_nodes(3);
        assert!(healed.is_empty());
    }

    #[test]
    fn test_detect_healed_nodes_multiple_nodes() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9706".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9707".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9708".parse().unwrap())
        .with_peer("node4", "127.0.0.1:9709".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        {
            let mut nodes = manager.nodes.write();
            for node_id in &["node2", "node3", "node4"] {
                let status = nodes.get_mut(&node_id.to_string()).unwrap();
                status.health = NodeHealth::Healthy;
                status.partition_start = Some(Instant::now() - Duration::from_secs(5));
                status.consecutive_successes = 5;
            }
        }

        let healed = manager.detect_healed_nodes(3);
        assert_eq!(healed.len(), 3);
    }

    #[test]
    fn test_detect_healed_nodes_partition_duration_calculation() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9710".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9711".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        let partition_duration = Duration::from_secs(30);
        {
            let mut nodes = manager.nodes.write();
            let status = nodes.get_mut(&"node2".to_string()).unwrap();
            status.health = NodeHealth::Healthy;
            status.partition_start = Some(Instant::now() - partition_duration);
            status.consecutive_successes = 5;
        }

        let healed = manager.detect_healed_nodes(3);
        assert_eq!(healed.len(), 1);
        let duration_ms = healed[0].1;
        assert!(duration_ms >= 30_000);
        assert!(duration_ms < 31_000);
    }

    // ========== Priority 1: Clear Partition State Tests ==========

    #[test]
    fn test_clear_partition_state_resets_node() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9712".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9713".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        {
            let mut nodes = manager.nodes.write();
            let status = nodes.get_mut(&"node2".to_string()).unwrap();
            status.partition_start = Some(Instant::now());
            status.consecutive_successes = 10;
        }

        manager.clear_partition_state(&"node2".to_string());

        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert!(status.partition_start.is_none());
        assert_eq!(status.consecutive_successes, 0);
    }

    #[test]
    fn test_clear_partition_state_nonexistent_node() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        manager.clear_partition_state(&"nonexistent".to_string());
    }

    #[test]
    fn test_clear_partition_states_multiple_nodes() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9714".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9715".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9716".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        {
            let mut nodes = manager.nodes.write();
            for node_id in &["node2", "node3"] {
                let status = nodes.get_mut(&node_id.to_string()).unwrap();
                status.partition_start = Some(Instant::now());
                status.consecutive_successes = 7;
            }
        }

        manager.clear_partition_states(&["node2".to_string(), "node3".to_string()]);

        for node_id in &["node2", "node3"] {
            let status = manager.node_status(&node_id.to_string()).unwrap();
            assert!(status.partition_start.is_none());
            assert_eq!(status.consecutive_successes, 0);
        }
    }

    #[test]
    fn test_clear_partition_states_mixed_existing_nonexisting() {
        let config = create_test_config();
        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        {
            let mut nodes = manager.nodes.write();
            let status = nodes.get_mut(&"node2".to_string()).unwrap();
            status.partition_start = Some(Instant::now());
            status.consecutive_successes = 5;
        }

        manager.clear_partition_states(&[
            "node2".to_string(),
            "nonexistent1".to_string(),
            "nonexistent2".to_string(),
        ]);

        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert!(status.partition_start.is_none());
        assert_eq!(status.consecutive_successes, 0);
    }

    // ========== Priority 1: on_partition_heal Callback Tests ==========

    #[test]
    fn test_on_partition_heal_callback_invoked() {
        use std::sync::Mutex;

        struct HealCallback {
            healed: Mutex<Vec<(Vec<NodeId>, u64)>>,
        }

        impl MembershipCallback for HealCallback {
            fn on_health_change(&self, _: &NodeId, _: NodeHealth, _: NodeHealth) {}
            fn on_view_change(&self, _: &ClusterView) {}
            fn on_partition_heal(&self, healed_nodes: &[NodeId], partition_duration_ms: u64) {
                self.healed
                    .lock()
                    .unwrap()
                    .push((healed_nodes.to_vec(), partition_duration_ms));
            }
        }

        let callback = HealCallback {
            healed: Mutex::new(Vec::new()),
        };

        callback.on_partition_heal(&["node2".to_string(), "node3".to_string()], 5000);

        let healed = callback.healed.lock().unwrap();
        assert_eq!(healed.len(), 1);
        assert_eq!(healed[0].0, vec!["node2".to_string(), "node3".to_string()]);
        assert_eq!(healed[0].1, 5000);
    }

    #[test]
    fn test_on_partition_heal_default_implementation() {
        struct NoOpCallback;

        impl MembershipCallback for NoOpCallback {
            fn on_health_change(&self, _: &NodeId, _: NodeHealth, _: NodeHealth) {}
            fn on_view_change(&self, _: &ClusterView) {}
        }

        let callback = NoOpCallback;
        callback.on_partition_heal(&["node2".to_string()], 1000);
    }

    // ========== Priority 1: Quorum Edge Cases ==========

    #[test]
    fn test_partition_status_single_node_cluster() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9717".parse().unwrap(),
            },
        )
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        assert_eq!(manager.partition_status(), PartitionStatus::QuorumReachable);
        assert!(manager.is_safe_to_write());
    }

    #[test]
    fn test_partition_status_two_node_cluster_split() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9718".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9719".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        manager.mark_failed(&"node2".to_string());

        assert_eq!(manager.partition_status(), PartitionStatus::Stalemate);
    }

    #[test]
    fn test_partition_status_three_node_quorum() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9720".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9721".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9722".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        manager.mark_healthy(&"node2".to_string());

        assert_eq!(manager.partition_status(), PartitionStatus::QuorumReachable);
        assert!(manager.is_safe_to_write());
    }

    #[test]
    fn test_partition_status_stalemate_verified() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9723".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9724".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9725".parse().unwrap())
        .with_peer("node4", "127.0.0.1:9726".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        manager.mark_healthy(&"node2".to_string());
        manager.mark_failed(&"node3".to_string());
        manager.mark_failed(&"node4".to_string());

        assert_eq!(manager.partition_status(), PartitionStatus::Stalemate);
        assert!(!manager.is_safe_to_write());
    }

    #[test]
    fn test_partition_status_five_node_quorum_boundary() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9727".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9728".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9729".parse().unwrap())
        .with_peer("node4", "127.0.0.1:9730".parse().unwrap())
        .with_peer("node5", "127.0.0.1:9731".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        manager.mark_healthy(&"node2".to_string());
        manager.mark_failed(&"node3".to_string());
        manager.mark_failed(&"node4".to_string());
        manager.mark_failed(&"node5".to_string());

        assert_eq!(manager.partition_status(), PartitionStatus::QuorumLost);

        manager.mark_healthy(&"node3".to_string());
        assert_eq!(manager.partition_status(), PartitionStatus::QuorumReachable);
    }

    // ========== Priority 2: Complex Failure Patterns ==========

    #[test]
    fn test_oscillating_node_health() {
        let mut status = NodeStatus::new("node1".to_string(), "127.0.0.1:9732".parse().unwrap());

        status.record_success(50);
        assert_eq!(status.health, NodeHealth::Healthy);
        assert_eq!(status.consecutive_successes, 1);

        status.record_failure(3);
        assert_eq!(status.health, NodeHealth::Degraded);
        assert_eq!(status.consecutive_successes, 0);

        status.record_success(50);
        assert_eq!(status.health, NodeHealth::Healthy);
        assert_eq!(status.consecutive_successes, 1);

        status.record_failure(3);
        status.record_failure(3);
        status.record_failure(3);
        assert_eq!(status.health, NodeHealth::Failed);

        status.record_success(50);
        assert_eq!(status.health, NodeHealth::Healthy);
        assert_eq!(status.consecutive_successes, 1);
    }

    #[tokio::test]
    async fn test_cascading_failures() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;
        health.failure_threshold = 1;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9733".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9734".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9735".parse().unwrap())
        .with_peer("node4", "127.0.0.1:9736".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport);

        manager.mark_healthy(&"node2".to_string());
        manager.mark_healthy(&"node3".to_string());
        manager.mark_healthy(&"node4".to_string());
        assert_eq!(manager.partition_status(), PartitionStatus::QuorumReachable);

        manager.mark_failed(&"node2".to_string());
        assert_eq!(manager.partition_status(), PartitionStatus::QuorumReachable);

        manager.mark_failed(&"node3".to_string());
        assert_eq!(manager.partition_status(), PartitionStatus::Stalemate);

        manager.mark_failed(&"node4".to_string());
        assert_eq!(manager.partition_status(), PartitionStatus::QuorumLost);
    }

    #[tokio::test]
    async fn test_recovery_after_grace_period() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;
        health.failure_threshold = 2;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9737".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9738".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        let manager = MembershipManager::new(config, transport.clone());

        transport.set_fail_sends(true);
        manager.check_health().await.unwrap();
        manager.check_health().await.unwrap();

        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert_eq!(status.health, NodeHealth::Failed);

        transport.set_fail_sends(false);
        manager.check_health().await.unwrap();

        let status = manager.node_status(&"node2".to_string()).unwrap();
        assert_eq!(status.health, NodeHealth::Healthy);
    }

    #[test]
    fn test_health_check_failure_during_degraded() {
        let mut status = NodeStatus::new("node1".to_string(), "127.0.0.1:9739".parse().unwrap());

        status.record_failure(3);
        assert_eq!(status.health, NodeHealth::Degraded);
        assert_eq!(status.consecutive_failures, 1);

        status.record_failure(3);
        assert_eq!(status.health, NodeHealth::Degraded);
        assert_eq!(status.consecutive_failures, 2);

        status.record_failure(3);
        assert_eq!(status.health, NodeHealth::Failed);
        assert_eq!(status.consecutive_failures, 3);
        assert!(status.partition_start.is_some());
    }

    // ========== Priority 2: NodeStatus & Stats ==========

    #[test]
    fn test_node_status_update_embedding() {
        let mut status = NodeStatus::new("node1".to_string(), "127.0.0.1:9740".parse().unwrap());
        assert!(status.state_embedding.is_none());
        assert!(status.embedding_updated.is_none());

        let embedding = SparseVector::from_parts(100, vec![0, 5, 10], vec![1.0, 2.0, 3.0]);
        status.update_embedding(embedding.clone());

        assert!(status.state_embedding.is_some());
        assert!(status.embedding_updated.is_some());

        let stored = status.state_embedding.unwrap();
        assert_eq!(stored.positions(), embedding.positions());
        assert_eq!(stored.values(), embedding.values());
    }

    #[test]
    fn test_node_status_record_failure_tracks_partition_start() {
        let mut status = NodeStatus::new("node1".to_string(), "127.0.0.1:9741".parse().unwrap());
        assert!(status.partition_start.is_none());

        status.record_failure(3);
        assert!(status.partition_start.is_none());

        status.record_failure(3);
        assert!(status.partition_start.is_none());

        status.record_failure(3);
        assert_eq!(status.health, NodeHealth::Failed);
        assert!(status.partition_start.is_some());

        let first_partition_start = status.partition_start;
        status.record_failure(3);
        assert_eq!(status.partition_start, first_partition_start);
    }

    #[tokio::test]
    async fn test_membership_stats_quorum_lost_events_incremented() {
        let mut health = HealthConfig::default();
        health.startup_grace_ms = 0;
        health.failure_threshold = 1;

        let config = ClusterConfig::new(
            "test-cluster",
            LocalNodeConfig {
                node_id: "node1".to_string(),
                bind_address: "127.0.0.1:9742".parse().unwrap(),
            },
        )
        .with_peer("node2", "127.0.0.1:9743".parse().unwrap())
        .with_peer("node3", "127.0.0.1:9744".parse().unwrap())
        .with_health(health);

        let transport = Arc::new(MockTransport::new("node1"));
        transport.set_fail_sends(true);
        let manager = MembershipManager::new(config, transport);

        let initial = manager.stats.quorum_lost_events.load(Ordering::Relaxed);

        manager.check_health().await.unwrap();

        let after = manager.stats.quorum_lost_events.load(Ordering::Relaxed);
        assert!(after > initial);
    }

    #[test]
    fn test_membership_stats_partition_events_tracked() {
        let stats = MembershipStats::new();
        assert_eq!(stats.partition_events.load(Ordering::Relaxed), 0);

        stats.partition_events.fetch_add(1, Ordering::Relaxed);
        assert_eq!(stats.partition_events.load(Ordering::Relaxed), 1);

        stats.partition_events.fetch_add(5, Ordering::Relaxed);
        assert_eq!(stats.partition_events.load(Ordering::Relaxed), 6);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.partition_events, 6);
    }
}
