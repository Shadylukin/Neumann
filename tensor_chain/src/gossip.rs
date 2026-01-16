//! SWIM-style gossip protocol for scalable membership management.
//!
//! Replaces O(N) sequential pings with O(log N) propagation via:
//! - Peer sampling with geometric routing awareness
//! - LWW-CRDT-based membership state
//! - SWIM suspicion/alive protocol for failure detection

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::{
    block::NodeId,
    error::Result,
    geometric_membership::GeometricMembershipManager,
    membership::{MembershipCallback, NodeHealth},
    network::{Message, Transport},
};

/// State of a node in the gossip protocol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GossipNodeState {
    /// Node identifier.
    pub node_id: NodeId,
    /// Health status.
    pub health: NodeHealth,
    /// Lamport timestamp for ordering.
    pub timestamp: u64,
    /// Wall clock time when updated (milliseconds since epoch).
    pub updated_at: u64,
    /// Incarnation number (monotonically increasing per node).
    pub incarnation: u64,
}

impl GossipNodeState {
    pub fn new(node_id: NodeId, health: NodeHealth, timestamp: u64, incarnation: u64) -> Self {
        let updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            node_id,
            health,
            timestamp,
            updated_at,
            incarnation,
        }
    }

    /// Check if this state supersedes another state for the same node.
    /// Uses incarnation first, then timestamp as tiebreaker.
    pub fn supersedes(&self, other: &GossipNodeState) -> bool {
        if self.incarnation != other.incarnation {
            self.incarnation > other.incarnation
        } else {
            self.timestamp > other.timestamp
        }
    }
}

/// Gossip protocol messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GossipMessage {
    /// Sync message with piggy-backed node states.
    Sync {
        sender: NodeId,
        states: Vec<GossipNodeState>,
        sender_time: u64,
    },
    /// Suspect a node of failure.
    Suspect {
        reporter: NodeId,
        suspect: NodeId,
        incarnation: u64,
    },
    /// Refute suspicion by proving aliveness.
    Alive { node_id: NodeId, incarnation: u64 },
    /// Indirect ping request (SWIM protocol).
    PingReq {
        origin: NodeId,
        target: NodeId,
        sequence: u64,
    },
    /// Indirect ping response.
    PingAck {
        origin: NodeId,
        target: NodeId,
        sequence: u64,
        success: bool,
    },
}

/// Last-Writer-Wins CRDT for membership state.
#[derive(Debug, Clone, Default)]
pub struct LWWMembershipState {
    /// Node states indexed by node ID.
    states: HashMap<NodeId, GossipNodeState>,
    /// Lamport time for ordering.
    lamport_time: u64,
}

impl LWWMembershipState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lamport_time(&self) -> u64 {
        self.lamport_time
    }

    /// Advance Lamport time and return new value.
    pub fn tick(&mut self) -> u64 {
        self.lamport_time += 1;
        self.lamport_time
    }

    /// Update Lamport time from incoming message.
    pub fn sync_time(&mut self, incoming_time: u64) {
        self.lamport_time = self.lamport_time.max(incoming_time) + 1;
    }

    pub fn get(&self, node_id: &NodeId) -> Option<&GossipNodeState> {
        self.states.get(node_id)
    }

    pub fn all_states(&self) -> impl Iterator<Item = &GossipNodeState> {
        self.states.values()
    }

    /// Get states for gossip (up to max_count, prioritizing recently updated).
    pub fn states_for_gossip(&self, max_count: usize) -> Vec<GossipNodeState> {
        let mut states: Vec<_> = self.states.values().cloned().collect();
        // Sort by timestamp descending (most recent first)
        states.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        states.truncate(max_count);
        states
    }

    /// Merge incoming states. Returns list of node IDs that changed.
    pub fn merge(&mut self, incoming: &[GossipNodeState]) -> Vec<NodeId> {
        let mut changed = Vec::new();

        for state in incoming {
            let should_update = match self.states.get(&state.node_id) {
                Some(existing) => state.supersedes(existing),
                None => true,
            };

            if should_update {
                changed.push(state.node_id.clone());
                self.states.insert(state.node_id.clone(), state.clone());
            }
        }

        // Update Lamport time from max incoming timestamp
        if let Some(max_ts) = incoming.iter().map(|s| s.timestamp).max() {
            self.sync_time(max_ts);
        }

        changed
    }

    /// Update local node state.
    pub fn update_local(
        &mut self,
        node_id: NodeId,
        health: NodeHealth,
        incarnation: u64,
    ) -> GossipNodeState {
        let timestamp = self.tick();
        let state = GossipNodeState::new(node_id.clone(), health, timestamp, incarnation);
        self.states.insert(node_id, state.clone());
        state
    }

    /// Mark a node as suspected.
    pub fn suspect(&mut self, node_id: &NodeId, incarnation: u64) -> bool {
        // Check if we should suspect (without mutable borrow)
        let should_suspect = self.states.get(node_id).is_some_and(|state| {
            state.incarnation == incarnation && state.health != NodeHealth::Failed
        });

        if should_suspect {
            let timestamp = self.tick();
            if let Some(state) = self.states.get_mut(node_id) {
                state.health = NodeHealth::Degraded;
                state.timestamp = timestamp;
                return true;
            }
        }
        false
    }

    /// Mark a node as failed.
    pub fn fail(&mut self, node_id: &NodeId) -> bool {
        // Check if we should fail (without mutable borrow)
        let should_fail = self
            .states
            .get(node_id)
            .is_some_and(|state| state.health != NodeHealth::Failed);

        if should_fail {
            let timestamp = self.tick();
            if let Some(state) = self.states.get_mut(node_id) {
                state.health = NodeHealth::Failed;
                state.timestamp = timestamp;
                return true;
            }
        }
        false
    }

    /// Refute suspicion with new incarnation.
    pub fn refute(&mut self, node_id: &NodeId, new_incarnation: u64) -> bool {
        // Check if we should refute (without mutable borrow)
        let should_refute = self
            .states
            .get(node_id)
            .is_some_and(|state| new_incarnation > state.incarnation);

        if should_refute {
            let timestamp = self.tick();
            if let Some(state) = self.states.get_mut(node_id) {
                state.incarnation = new_incarnation;
                state.health = NodeHealth::Healthy;
                state.timestamp = timestamp;
                return true;
            }
        }
        false
    }

    /// Count nodes by health status.
    pub fn count_by_health(&self) -> (usize, usize, usize) {
        let mut healthy = 0;
        let mut degraded = 0;
        let mut failed = 0;

        for state in self.states.values() {
            match state.health {
                NodeHealth::Healthy => healthy += 1,
                NodeHealth::Degraded => degraded += 1,
                NodeHealth::Failed | NodeHealth::Unknown => failed += 1,
            }
        }

        (healthy, degraded, failed)
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

/// Configuration for gossip protocol.
#[derive(Debug, Clone)]
pub struct GossipConfig {
    /// Number of peers to gossip with per round.
    pub fanout: usize,
    /// Interval between gossip rounds (milliseconds).
    pub gossip_interval_ms: u64,
    /// Timeout for suspicion before failure (milliseconds).
    pub suspicion_timeout_ms: u64,
    /// Maximum states to include per gossip message.
    pub max_states_per_message: usize,
    /// Whether to use geometric routing for peer selection.
    pub geometric_routing: bool,
    /// Number of indirect pings to attempt on direct failure.
    pub indirect_ping_count: usize,
    /// Timeout for indirect pings (milliseconds).
    pub indirect_ping_timeout_ms: u64,
}

impl Default for GossipConfig {
    fn default() -> Self {
        Self {
            fanout: 3,
            gossip_interval_ms: 200,
            suspicion_timeout_ms: 5000,
            max_states_per_message: 20,
            geometric_routing: true,
            indirect_ping_count: 3,
            indirect_ping_timeout_ms: 500,
        }
    }
}

/// Pending suspicion record.
#[derive(Debug, Clone)]
struct PendingSuspicion {
    /// When suspicion started.
    started_at: Instant,
    /// Incarnation at time of suspicion (for refutation matching).
    #[allow(dead_code)]
    incarnation: u64,
}

/// Heal progress tracking for a node recovering from partition.
#[derive(Debug, Clone)]
struct HealProgress {
    /// When the node first started recovering (kept for future debugging/metrics).
    #[allow(dead_code)]
    started_at: Instant,
    /// When the partition originally started.
    partition_start: Instant,
    /// Number of consecutive successful communications.
    consecutive_successes: u32,
}

/// Gossip-based membership manager.
pub struct GossipMembershipManager {
    /// Local node identifier.
    local_node: NodeId,
    /// Current incarnation number.
    incarnation: AtomicU64,
    /// CRDT membership state.
    state: RwLock<LWWMembershipState>,
    /// Configuration.
    config: GossipConfig,
    /// Network transport.
    transport: Arc<dyn Transport>,
    /// Optional geometric manager for peer selection.
    geometric: Option<Arc<GeometricMembershipManager>>,
    /// Registered callbacks.
    callbacks: RwLock<Vec<Arc<dyn MembershipCallback>>>,
    /// Pending suspicions (node_id -> suspicion record).
    suspicions: RwLock<HashMap<NodeId, PendingSuspicion>>,
    /// Heal progress tracking (node_id -> heal progress).
    heal_progress: RwLock<HashMap<NodeId, HealProgress>>,
    /// Known peers for random selection fallback.
    known_peers: RwLock<Vec<NodeId>>,
    /// Shutdown signal.
    shutdown_tx: broadcast::Sender<()>,
    /// Ping sequence counter for indirect pings.
    ping_sequence: AtomicU64,
    /// Gossip round counter.
    round_counter: AtomicU64,
}

impl GossipMembershipManager {
    pub fn new(local_node: NodeId, config: GossipConfig, transport: Arc<dyn Transport>) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);

        let mut state = LWWMembershipState::new();
        // Register local node as healthy
        state.update_local(local_node.clone(), NodeHealth::Healthy, 0);

        Self {
            local_node,
            incarnation: AtomicU64::new(0),
            state: RwLock::new(state),
            config,
            transport,
            geometric: None,
            callbacks: RwLock::new(Vec::new()),
            suspicions: RwLock::new(HashMap::new()),
            heal_progress: RwLock::new(HashMap::new()),
            known_peers: RwLock::new(Vec::new()),
            shutdown_tx,
            ping_sequence: AtomicU64::new(0),
            round_counter: AtomicU64::new(0),
        }
    }

    /// Create with geometric membership manager.
    pub fn with_geometric(
        local_node: NodeId,
        config: GossipConfig,
        transport: Arc<dyn Transport>,
        geometric: Arc<GeometricMembershipManager>,
    ) -> Self {
        let mut manager = Self::new(local_node, config, transport);
        manager.geometric = Some(geometric);
        manager
    }

    /// Add a known peer.
    pub fn add_peer(&self, peer: NodeId) {
        let mut peers = self.known_peers.write();
        if !peers.contains(&peer) {
            peers.push(peer.clone());
        }

        // Initialize state if not present
        let mut state = self.state.write();
        if state.get(&peer).is_none() {
            let timestamp = state.tick();
            state.merge(&[GossipNodeState::new(
                peer,
                NodeHealth::Unknown,
                timestamp,
                0,
            )]);
        }
    }

    /// Register a callback for membership changes.
    pub fn register_callback(&self, callback: Arc<dyn MembershipCallback>) {
        self.callbacks.write().push(callback);
    }

    /// Get current node state.
    pub fn node_state(&self, node_id: &NodeId) -> Option<GossipNodeState> {
        self.state.read().get(node_id).cloned()
    }

    /// Get all node states.
    pub fn all_states(&self) -> Vec<GossipNodeState> {
        self.state.read().all_states().cloned().collect()
    }

    /// Get number of known nodes.
    pub fn node_count(&self) -> usize {
        self.state.read().len()
    }

    /// Get health counts (healthy, degraded, failed).
    pub fn health_counts(&self) -> (usize, usize, usize) {
        self.state.read().count_by_health()
    }

    /// Get gossip round counter.
    pub fn round_count(&self) -> u64 {
        self.round_counter.load(Ordering::Relaxed)
    }

    /// Select peers for gossip round.
    fn select_gossip_targets(&self, k: usize) -> Vec<NodeId> {
        // Try geometric selection first
        if self.config.geometric_routing {
            if let Some(ref geo) = self.geometric {
                if let Some(local_emb) = geo.local_embedding() {
                    let ranked = geo.ranked_peers(&local_emb);
                    let targets: Vec<_> = ranked
                        .into_iter()
                        .filter(|p| p.is_healthy)
                        .take(k)
                        .map(|p| p.node_id)
                        .collect();

                    if !targets.is_empty() {
                        return targets;
                    }
                }
            }
        }

        // Fallback: random selection from known peers
        let peers = self.known_peers.read();
        if peers.is_empty() {
            return Vec::new();
        }

        // Simple random selection (use round counter as seed)
        let round = self.round_counter.load(Ordering::Relaxed);
        let mut selected = Vec::with_capacity(k.min(peers.len()));

        for i in 0..k.min(peers.len()) {
            let idx = ((round + i as u64) as usize) % peers.len();
            let peer = &peers[idx];
            if peer != &self.local_node && !selected.contains(peer) {
                selected.push(peer.clone());
            }
        }

        selected
    }

    /// Run a single gossip round.
    pub async fn gossip_round(&self) -> Result<()> {
        self.round_counter.fetch_add(1, Ordering::Relaxed);

        let targets = self.select_gossip_targets(self.config.fanout);
        if targets.is_empty() {
            return Ok(());
        }

        let states = self
            .state
            .read()
            .states_for_gossip(self.config.max_states_per_message);
        let sender_time = self.state.read().lamport_time();

        let msg = Message::Gossip(GossipMessage::Sync {
            sender: self.local_node.clone(),
            states,
            sender_time,
        });

        // Send in parallel using tokio spawn (fire-and-forget)
        for target in targets {
            let transport = Arc::clone(&self.transport);
            let msg = msg.clone();
            tokio::spawn(async move {
                let _ = transport.send(&target, msg).await;
            });
        }

        // Check and expire suspicions
        self.expire_suspicions();

        Ok(())
    }

    /// Handle incoming gossip message.
    pub fn handle_gossip(&self, msg: GossipMessage) {
        match msg {
            GossipMessage::Sync {
                sender,
                states,
                sender_time,
            } => {
                self.handle_sync(&sender, states, sender_time);
            },
            GossipMessage::Suspect {
                reporter,
                suspect,
                incarnation,
            } => {
                self.handle_suspect(&reporter, &suspect, incarnation);
            },
            GossipMessage::Alive {
                node_id,
                incarnation,
            } => {
                self.handle_alive(&node_id, incarnation);
            },
            GossipMessage::PingReq {
                origin,
                target,
                sequence,
            } => {
                self.handle_ping_req(&origin, &target, sequence);
            },
            GossipMessage::PingAck {
                origin,
                target,
                sequence,
                success,
            } => {
                self.handle_ping_ack(&origin, &target, sequence, success);
            },
        }
    }

    fn handle_sync(&self, sender: &NodeId, states: Vec<GossipNodeState>, sender_time: u64) {
        let changed = {
            let mut state = self.state.write();
            state.sync_time(sender_time);
            state.merge(&states)
        };

        // Notify callbacks for changed nodes
        for node_id in changed {
            if let Some(new_state) = self.state.read().get(&node_id) {
                let callbacks = self.callbacks.read();
                for callback in callbacks.iter() {
                    callback.on_health_change(
                        &node_id,
                        NodeHealth::Unknown, // Don't track old state for efficiency
                        new_state.health,
                    );
                }
            }
        }

        // Mark sender as healthy (they're clearly alive)
        let sender_incarnation = self
            .state
            .read()
            .get(sender)
            .map(|s| s.incarnation)
            .unwrap_or(0);

        let sender_state = GossipNodeState::new(
            sender.clone(),
            NodeHealth::Healthy,
            self.state.read().lamport_time() + 1,
            sender_incarnation,
        );
        self.state.write().merge(&[sender_state]);

        // Clear any suspicion on sender
        self.suspicions.write().remove(sender);
    }

    fn handle_suspect(&self, _reporter: &NodeId, suspect: &NodeId, incarnation: u64) {
        // If we're being suspected, refute it
        if suspect == &self.local_node {
            let new_incarnation = self.incarnation.fetch_add(1, Ordering::SeqCst) + 1;
            self.broadcast_alive(new_incarnation);
            return;
        }

        // Start suspicion timer
        let mut suspicions = self.suspicions.write();
        if !suspicions.contains_key(suspect) {
            self.state.write().suspect(suspect, incarnation);
            suspicions.insert(
                suspect.clone(),
                PendingSuspicion {
                    started_at: Instant::now(),
                    incarnation,
                },
            );
        }
    }

    fn handle_alive(&self, node_id: &NodeId, incarnation: u64) {
        if self.state.write().refute(node_id, incarnation) {
            self.suspicions.write().remove(node_id);
        }
    }

    fn handle_ping_req(&self, origin: &NodeId, target: &NodeId, sequence: u64) {
        // Try to ping the target
        let transport = Arc::clone(&self.transport);
        let origin = origin.clone();
        let target = target.clone();
        let local = self.local_node.clone();

        tokio::spawn(async move {
            let success = transport
                .send(&target, Message::Ping { term: 0 })
                .await
                .is_ok();

            // Send ack back to origin
            let _ = transport
                .send(
                    &origin,
                    Message::Gossip(GossipMessage::PingAck {
                        origin: local,
                        target,
                        sequence,
                        success,
                    }),
                )
                .await;
        });
    }

    fn handle_ping_ack(&self, _origin: &NodeId, target: &NodeId, _sequence: u64, success: bool) {
        if success {
            // Target is alive - clear suspicion
            if let Some(state) = self.state.read().get(target) {
                self.handle_alive(target, state.incarnation);
            }
        }
    }

    /// Broadcast alive message to refute suspicion.
    fn broadcast_alive(&self, incarnation: u64) {
        let msg = Message::Gossip(GossipMessage::Alive {
            node_id: self.local_node.clone(),
            incarnation,
        });

        let targets = self.select_gossip_targets(self.config.fanout);
        let transport = Arc::clone(&self.transport);

        tokio::spawn(async move {
            for target in targets {
                let _ = transport.send(&target, msg.clone()).await;
            }
        });
    }

    /// Expire suspicions that have timed out.
    fn expire_suspicions(&self) {
        let timeout = Duration::from_millis(self.config.suspicion_timeout_ms);
        let now = Instant::now();

        let mut to_fail = Vec::new();
        {
            let suspicions = self.suspicions.read();
            for (node_id, suspicion) in suspicions.iter() {
                if now.duration_since(suspicion.started_at) >= timeout {
                    to_fail.push(node_id.clone());
                }
            }
        }

        for node_id in to_fail {
            self.suspicions.write().remove(&node_id);
            if self.state.write().fail(&node_id) {
                // Notify callbacks
                let callbacks = self.callbacks.read();
                for callback in callbacks.iter() {
                    callback.on_health_change(&node_id, NodeHealth::Degraded, NodeHealth::Failed);
                }
            }
        }
    }

    /// Initiate suspicion protocol for a node that failed direct ping.
    pub async fn suspect_node(&self, node_id: &NodeId) -> Result<()> {
        let incarnation = self
            .state
            .read()
            .get(node_id)
            .map(|s| s.incarnation)
            .unwrap_or(0);

        // Start local suspicion
        {
            let mut suspicions = self.suspicions.write();
            if !suspicions.contains_key(node_id) {
                self.state.write().suspect(node_id, incarnation);
                suspicions.insert(
                    node_id.clone(),
                    PendingSuspicion {
                        started_at: Instant::now(),
                        incarnation,
                    },
                );
            }
        }

        // Broadcast suspicion
        let msg = Message::Gossip(GossipMessage::Suspect {
            reporter: self.local_node.clone(),
            suspect: node_id.clone(),
            incarnation,
        });

        let targets = self.select_gossip_targets(self.config.fanout);
        for target in targets {
            let _ = self.transport.send(&target, msg.clone()).await;
        }

        // Try indirect pings
        self.try_indirect_ping(node_id).await;

        Ok(())
    }

    /// Try indirect ping through other nodes.
    async fn try_indirect_ping(&self, target: &NodeId) {
        let intermediaries = self.select_gossip_targets(self.config.indirect_ping_count);
        if intermediaries.is_empty() {
            return;
        }

        let sequence = self.ping_sequence.fetch_add(1, Ordering::Relaxed);

        let msg = Message::Gossip(GossipMessage::PingReq {
            origin: self.local_node.clone(),
            target: target.clone(),
            sequence,
        });

        for intermediary in intermediaries {
            let _ = self.transport.send(&intermediary, msg.clone()).await;
        }
    }

    /// Run the gossip protocol loop.
    pub async fn run(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let interval = Duration::from_millis(self.config.gossip_interval_ms);

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    break;
                }
                _ = tokio::time::sleep(interval) => {
                    let _ = self.gossip_round().await;
                }
            }
        }

        Ok(())
    }

    /// Shutdown the gossip manager.
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());
    }

    /// Record successful communication with a previously failed node.
    ///
    /// Call this when you successfully communicate with a node that was in Failed state.
    /// This starts or continues tracking the heal progress for partition reconciliation.
    ///
    /// # Arguments
    /// * `node` - The node ID that successfully communicated
    /// * `partition_start` - When the partition originally started (if known)
    pub fn record_heal_progress(&self, node: &NodeId, partition_start: Option<Instant>) {
        let state = self.state.read();
        let node_state = state.get(node);

        // Only track heal progress for nodes that were Failed or are being monitored
        let was_failed = node_state.is_some_and(|s| s.health == NodeHealth::Failed);
        drop(state);

        let mut progress = self.heal_progress.write();

        if let Some(hp) = progress.get_mut(node) {
            // Already tracking - increment counter
            hp.consecutive_successes += 1;
        } else if was_failed || partition_start.is_some() {
            // Start tracking heal progress
            let start = partition_start.unwrap_or_else(Instant::now);
            progress.insert(
                node.clone(),
                HealProgress {
                    started_at: Instant::now(),
                    partition_start: start,
                    consecutive_successes: 1,
                },
            );
        }
    }

    /// Check if a node has met the heal confirmation threshold.
    ///
    /// # Arguments
    /// * `node` - The node ID to check
    /// * `threshold` - Number of consecutive successes required
    ///
    /// # Returns
    /// * `Some((partition_duration_ms))` if heal is confirmed
    /// * `None` if heal is not confirmed or node is not being tracked
    pub fn is_heal_confirmed(&self, node: &NodeId, threshold: u32) -> Option<u64> {
        let progress = self.heal_progress.read();

        if let Some(hp) = progress.get(node) {
            if hp.consecutive_successes >= threshold {
                let partition_duration_ms = hp.partition_start.elapsed().as_millis() as u64;
                return Some(partition_duration_ms);
            }
        }

        None
    }

    /// Clear heal progress for a node after heal is processed.
    pub fn clear_heal_progress(&self, node: &NodeId) {
        self.heal_progress.write().remove(node);
    }

    /// Clear heal progress for multiple nodes.
    pub fn clear_heal_progress_batch(&self, nodes: &[NodeId]) {
        let mut progress = self.heal_progress.write();
        for node in nodes {
            progress.remove(node);
        }
    }

    /// Reset heal progress counter for a node (e.g., on communication failure).
    pub fn reset_heal_progress(&self, node: &NodeId) {
        let mut progress = self.heal_progress.write();
        if let Some(hp) = progress.get_mut(node) {
            hp.consecutive_successes = 0;
        }
    }

    /// Get all nodes currently being tracked for heal progress.
    pub fn healing_nodes(&self) -> Vec<(NodeId, u32)> {
        self.heal_progress
            .read()
            .iter()
            .map(|(id, hp)| (id.clone(), hp.consecutive_successes))
            .collect()
    }

    /// Get the current Lamport time from the CRDT state.
    pub fn lamport_time(&self) -> u64 {
        self.state.read().lamport_time()
    }

    /// Get all node states for membership view exchange.
    pub fn membership_view(&self) -> Vec<GossipNodeState> {
        self.state.read().all_states().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gossip_node_state_supersedes() {
        let state1 = GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 1, 0);
        let state2 = GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 2, 0);

        assert!(state2.supersedes(&state1));
        assert!(!state1.supersedes(&state2));

        // Higher incarnation wins
        let state3 = GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 1, 1);
        assert!(state3.supersedes(&state1));
        assert!(state3.supersedes(&state2));
    }

    #[test]
    fn test_lww_merge_newer_wins() {
        let mut state = LWWMembershipState::new();

        let old = GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 1, 0);
        let new = GossipNodeState::new("node1".to_string(), NodeHealth::Failed, 2, 0);

        state.merge(&[old]);
        assert_eq!(
            state.get(&"node1".to_string()).unwrap().health,
            NodeHealth::Healthy
        );

        state.merge(&[new]);
        assert_eq!(
            state.get(&"node1".to_string()).unwrap().health,
            NodeHealth::Failed
        );
    }

    #[test]
    fn test_lww_merge_older_ignored() {
        let mut state = LWWMembershipState::new();

        let new = GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 2, 0);
        let old = GossipNodeState::new("node1".to_string(), NodeHealth::Failed, 1, 0);

        state.merge(&[new]);
        state.merge(&[old]); // Should be ignored

        assert_eq!(
            state.get(&"node1".to_string()).unwrap().health,
            NodeHealth::Healthy
        );
    }

    #[test]
    fn test_lamport_time_advances() {
        let mut state = LWWMembershipState::new();
        assert_eq!(state.lamport_time(), 0);

        state.tick();
        assert_eq!(state.lamport_time(), 1);

        state.sync_time(10);
        assert_eq!(state.lamport_time(), 11); // max(1, 10) + 1
    }

    #[test]
    fn test_suspicion_refutation() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Healthy, 0);

        // Suspect the node
        assert!(state.suspect(&"node1".to_string(), 0));
        assert_eq!(
            state.get(&"node1".to_string()).unwrap().health,
            NodeHealth::Degraded
        );

        // Refute with higher incarnation
        assert!(state.refute(&"node1".to_string(), 1));
        assert_eq!(
            state.get(&"node1".to_string()).unwrap().health,
            NodeHealth::Healthy
        );
        assert_eq!(state.get(&"node1".to_string()).unwrap().incarnation, 1);
    }

    #[test]
    fn test_states_for_gossip_limits_count() {
        let mut state = LWWMembershipState::new();

        for i in 0..10 {
            state.update_local(format!("node{}", i), NodeHealth::Healthy, 0);
        }

        let gossip_states = state.states_for_gossip(5);
        assert_eq!(gossip_states.len(), 5);
    }

    #[test]
    fn test_health_counts() {
        let mut state = LWWMembershipState::new();

        state.update_local("healthy1".to_string(), NodeHealth::Healthy, 0);
        state.update_local("healthy2".to_string(), NodeHealth::Healthy, 0);
        state.update_local("degraded".to_string(), NodeHealth::Degraded, 0);
        state.update_local("failed".to_string(), NodeHealth::Failed, 0);

        let (healthy, degraded, failed) = state.count_by_health();
        assert_eq!(healthy, 2);
        assert_eq!(degraded, 1);
        assert_eq!(failed, 1);
    }

    #[test]
    fn test_gossip_message_serialization() {
        let messages = vec![
            GossipMessage::Sync {
                sender: "node1".to_string(),
                states: vec![GossipNodeState::new(
                    "node2".to_string(),
                    NodeHealth::Healthy,
                    1,
                    0,
                )],
                sender_time: 42,
            },
            GossipMessage::Suspect {
                reporter: "node1".to_string(),
                suspect: "node2".to_string(),
                incarnation: 5,
            },
            GossipMessage::Alive {
                node_id: "node1".to_string(),
                incarnation: 6,
            },
            GossipMessage::PingReq {
                origin: "node1".to_string(),
                target: "node2".to_string(),
                sequence: 123,
            },
            GossipMessage::PingAck {
                origin: "node1".to_string(),
                target: "node2".to_string(),
                sequence: 123,
                success: true,
            },
        ];

        for msg in messages {
            let bytes = bincode::serialize(&msg).unwrap();
            let restored: GossipMessage = bincode::deserialize(&bytes).unwrap();
            assert_eq!(msg, restored);
        }
    }

    #[test]
    fn test_gossip_node_state_serialization() {
        let state = GossipNodeState::new("test".to_string(), NodeHealth::Healthy, 100, 5);

        let bytes = bincode::serialize(&state).unwrap();
        let restored: GossipNodeState = bincode::deserialize(&bytes).unwrap();

        assert_eq!(state.node_id, restored.node_id);
        assert_eq!(state.health, restored.health);
        assert_eq!(state.timestamp, restored.timestamp);
        assert_eq!(state.incarnation, restored.incarnation);
    }

    // ========== Heal Progress Tracking Tests ==========

    #[test]
    fn test_heal_progress_tracking() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Mark a node as failed first
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Failed, 0);
        }

        let partition_start = Some(Instant::now());

        // Record heal progress
        manager.record_heal_progress(&"node2".to_string(), partition_start);
        assert_eq!(manager.healing_nodes().len(), 1);

        // Not yet confirmed (threshold 3)
        assert!(manager.is_heal_confirmed(&"node2".to_string(), 3).is_none());

        // Record more progress
        manager.record_heal_progress(&"node2".to_string(), None);
        manager.record_heal_progress(&"node2".to_string(), None);

        // Now should be confirmed
        let result = manager.is_heal_confirmed(&"node2".to_string(), 3);
        assert!(result.is_some());
    }

    #[test]
    fn test_heal_progress_clear() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Mark as failed and track heal
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Failed, 0);
        }
        manager.record_heal_progress(&"node2".to_string(), Some(Instant::now()));
        assert_eq!(manager.healing_nodes().len(), 1);

        // Clear progress
        manager.clear_heal_progress(&"node2".to_string());
        assert_eq!(manager.healing_nodes().len(), 0);
    }

    #[test]
    fn test_heal_progress_reset() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Mark as failed
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Failed, 0);
        }

        // Build up some progress
        manager.record_heal_progress(&"node2".to_string(), Some(Instant::now()));
        manager.record_heal_progress(&"node2".to_string(), None);

        let nodes = manager.healing_nodes();
        assert_eq!(nodes[0].1, 2); // 2 consecutive successes

        // Reset on failure
        manager.reset_heal_progress(&"node2".to_string());

        let nodes = manager.healing_nodes();
        assert_eq!(nodes[0].1, 0); // Reset to 0
    }

    #[test]
    fn test_heal_progress_batch_clear() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Mark multiple nodes as failed
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Failed, 0);
            state.update_local("node3".to_string(), NodeHealth::Failed, 0);
        }

        // Track heal progress for both
        manager.record_heal_progress(&"node2".to_string(), Some(Instant::now()));
        manager.record_heal_progress(&"node3".to_string(), Some(Instant::now()));
        assert_eq!(manager.healing_nodes().len(), 2);

        // Batch clear
        manager.clear_heal_progress_batch(&["node2".to_string(), "node3".to_string()]);
        assert_eq!(manager.healing_nodes().len(), 0);
    }

    #[test]
    fn test_lamport_time_getter() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Initial state has lamport time from local node update
        let initial_time = manager.lamport_time();

        // Manually advance time
        {
            let mut state = manager.state.write();
            state.tick();
        }

        assert!(manager.lamport_time() > initial_time);
    }

    #[test]
    fn test_membership_view() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Add some nodes
        manager.add_peer("node2".to_string());
        manager.add_peer("node3".to_string());

        let view = manager.membership_view();
        assert_eq!(view.len(), 3); // local + 2 peers
    }

    // ========== Gossip Target Selection Tests ==========

    #[test]
    fn test_select_gossip_targets_fallback_random() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut config = GossipConfig::default();
        config.geometric_routing = false; // Disable geometric
        config.fanout = 2;

        let manager =
            GossipMembershipManager::new("node1".to_string(), config, transport);

        manager.add_peer("node2".to_string());
        manager.add_peer("node3".to_string());
        manager.add_peer("node4".to_string());

        let targets = manager.select_gossip_targets(2);
        assert!(targets.len() <= 2);
        // Should not include self
        assert!(!targets.contains(&"node1".to_string()));
    }

    #[test]
    fn test_select_gossip_targets_excludes_self() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut config = GossipConfig::default();
        config.geometric_routing = false;

        let manager =
            GossipMembershipManager::new("node1".to_string(), config, transport);

        // Add self as peer (shouldn't happen but test robustness)
        manager.add_peer("node1".to_string());
        manager.add_peer("node2".to_string());

        let targets = manager.select_gossip_targets(5);
        assert!(!targets.contains(&"node1".to_string()));
    }

    #[test]
    fn test_select_gossip_targets_empty_peers() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut config = GossipConfig::default();
        config.geometric_routing = false;

        let manager =
            GossipMembershipManager::new("node1".to_string(), config, transport);

        let targets = manager.select_gossip_targets(3);
        assert!(targets.is_empty());
    }

    #[test]
    fn test_select_gossip_targets_no_duplicates() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut config = GossipConfig::default();
        config.geometric_routing = false;
        config.fanout = 3;

        let manager =
            GossipMembershipManager::new("node1".to_string(), config, transport);

        manager.add_peer("node2".to_string());
        manager.add_peer("node3".to_string());
        manager.add_peer("node4".to_string());

        // Run multiple rounds to test various selection paths
        for _ in 0..10 {
            let targets = manager.select_gossip_targets(3);
            let unique: std::collections::HashSet<_> = targets.iter().collect();
            assert_eq!(targets.len(), unique.len(), "no duplicates");
        }
    }

    // ========== Message Handler Tests ==========

    #[test]
    fn test_handle_sync_sender_marked_healthy() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Add node2 as peer with unknown health
        manager.add_peer("node2".to_string());

        // Handle sync from node2
        let states = vec![GossipNodeState::new(
            "node3".to_string(),
            NodeHealth::Healthy,
            1,
            0,
        )];
        manager.handle_gossip(GossipMessage::Sync {
            sender: "node2".to_string(),
            states,
            sender_time: 5,
        });

        // node2 should now be healthy
        let node2_state = manager.node_state(&"node2".to_string());
        assert!(node2_state.is_some());
        assert_eq!(node2_state.unwrap().health, NodeHealth::Healthy);
    }

    #[test]
    fn test_handle_sync_callbacks_invoked() {
        use crate::membership::ClusterView;
        use crate::network::MemoryTransport;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Create a callback that counts invocations
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        struct CountingCallback {
            count: Arc<AtomicUsize>,
        }
        impl MembershipCallback for CountingCallback {
            fn on_health_change(
                &self,
                _node_id: &NodeId,
                _old_health: NodeHealth,
                _new_health: NodeHealth,
            ) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            fn on_view_change(&self, _view: &ClusterView) {}
        }

        manager.register_callback(Arc::new(CountingCallback { count: count_clone }));

        // Handle sync with a new node
        let states = vec![GossipNodeState::new(
            "node2".to_string(),
            NodeHealth::Healthy,
            100, // High timestamp to ensure it wins
            0,
        )];

        manager.handle_gossip(GossipMessage::Sync {
            sender: "node3".to_string(),
            states,
            sender_time: 100,
        });

        // Callback should have been invoked for node2
        assert!(count.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn test_handle_suspect_self_refutation() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        let initial_incarnation = manager.incarnation.load(Ordering::SeqCst);

        // When we're suspected, we should refute by incrementing incarnation
        manager.handle_gossip(GossipMessage::Suspect {
            reporter: "node2".to_string(),
            suspect: "node1".to_string(), // Self is suspected
            incarnation: 0,
        });

        // Give time for async broadcast_alive to spawn
        tokio::task::yield_now().await;

        // Incarnation should have increased
        let new_incarnation = manager.incarnation.load(Ordering::SeqCst);
        assert!(new_incarnation > initial_incarnation);
    }

    #[test]
    fn test_handle_suspect_remote_tracking() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Add node2 first
        manager.add_peer("node2".to_string());
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Healthy, 0);
        }

        // Suspect node2
        manager.handle_gossip(GossipMessage::Suspect {
            reporter: "node3".to_string(),
            suspect: "node2".to_string(),
            incarnation: 0,
        });

        // node2 should now be in suspicions
        let suspicions = manager.suspicions.read();
        assert!(suspicions.contains_key(&"node2".to_string()));

        // Health should be degraded
        let state = manager.node_state(&"node2".to_string());
        assert_eq!(state.unwrap().health, NodeHealth::Degraded);
    }

    #[test]
    fn test_handle_alive_clears_suspicion() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Add and suspect node2
        manager.add_peer("node2".to_string());
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Healthy, 0);
        }
        manager.handle_gossip(GossipMessage::Suspect {
            reporter: "node3".to_string(),
            suspect: "node2".to_string(),
            incarnation: 0,
        });

        // Verify suspicion exists
        assert!(manager.suspicions.read().contains_key(&"node2".to_string()));

        // Handle alive with higher incarnation
        manager.handle_gossip(GossipMessage::Alive {
            node_id: "node2".to_string(),
            incarnation: 1,
        });

        // Suspicion should be cleared
        assert!(!manager.suspicions.read().contains_key(&"node2".to_string()));

        // Health should be back to healthy
        let state = manager.node_state(&"node2".to_string());
        assert_eq!(state.unwrap().health, NodeHealth::Healthy);
    }

    #[test]
    fn test_handle_ping_ack_success_for_unknown_target() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Handle successful ping ack for a target we don't know about
        // This exercises the early return path when state.get(target) returns None
        manager.handle_gossip(GossipMessage::PingAck {
            origin: "node3".to_string(),
            target: "unknown_node".to_string(),
            sequence: 1,
            success: true,
        });

        // Should not crash and no suspicions should be modified
        assert!(manager.suspicions.read().is_empty());
    }

    #[test]
    fn test_handle_ping_ack_failure_keeps_suspicion() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Setup node2 with suspicion
        manager.add_peer("node2".to_string());
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Degraded, 0);
        }
        manager.suspicions.write().insert(
            "node2".to_string(),
            PendingSuspicion {
                started_at: Instant::now(),
                incarnation: 0,
            },
        );

        // Handle failed ping ack
        manager.handle_gossip(GossipMessage::PingAck {
            origin: "node3".to_string(),
            target: "node2".to_string(),
            sequence: 1,
            success: false,
        });

        // Suspicion should still exist
        assert!(manager.suspicions.read().contains_key(&"node2".to_string()));
    }

    // ========== Suspicion Protocol Tests ==========

    #[test]
    fn test_expire_suspicions_timeout() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut config = GossipConfig::default();
        config.suspicion_timeout_ms = 1; // 1ms timeout for quick test

        let manager =
            GossipMembershipManager::new("node1".to_string(), config, transport);

        // Add and suspect node2
        manager.add_peer("node2".to_string());
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Degraded, 0);
        }
        manager.suspicions.write().insert(
            "node2".to_string(),
            PendingSuspicion {
                started_at: Instant::now() - Duration::from_millis(10), // Already expired
                incarnation: 0,
            },
        );

        // Expire suspicions
        manager.expire_suspicions();

        // node2 should now be failed
        let state = manager.node_state(&"node2".to_string());
        assert_eq!(state.unwrap().health, NodeHealth::Failed);

        // Suspicion should be removed
        assert!(!manager.suspicions.read().contains_key(&"node2".to_string()));
    }

    #[test]
    fn test_expire_suspicions_not_yet_timed_out() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut config = GossipConfig::default();
        config.suspicion_timeout_ms = 60000; // 60s timeout

        let manager =
            GossipMembershipManager::new("node1".to_string(), config, transport);

        // Add and suspect node2
        manager.add_peer("node2".to_string());
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Degraded, 0);
        }
        manager.suspicions.write().insert(
            "node2".to_string(),
            PendingSuspicion {
                started_at: Instant::now(),
                incarnation: 0,
            },
        );

        // Expire suspicions (nothing should happen)
        manager.expire_suspicions();

        // node2 should still be degraded
        let state = manager.node_state(&"node2".to_string());
        assert_eq!(state.unwrap().health, NodeHealth::Degraded);

        // Suspicion should still exist
        assert!(manager.suspicions.read().contains_key(&"node2".to_string()));
    }

    #[test]
    fn test_expire_suspicions_callback_invoked() {
        use crate::membership::ClusterView;
        use crate::network::MemoryTransport;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut config = GossipConfig::default();
        config.suspicion_timeout_ms = 1;

        let manager =
            GossipMembershipManager::new("node1".to_string(), config, transport);

        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        struct CountingCallback {
            count: Arc<AtomicUsize>,
        }
        impl MembershipCallback for CountingCallback {
            fn on_health_change(
                &self,
                _node_id: &NodeId,
                _old_health: NodeHealth,
                _new_health: NodeHealth,
            ) {
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            fn on_view_change(&self, _view: &ClusterView) {}
        }

        manager.register_callback(Arc::new(CountingCallback { count: count_clone }));

        // Add and suspect node2
        manager.add_peer("node2".to_string());
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Degraded, 0);
        }
        manager.suspicions.write().insert(
            "node2".to_string(),
            PendingSuspicion {
                started_at: Instant::now() - Duration::from_millis(10),
                incarnation: 0,
            },
        );

        // Expire suspicions
        manager.expire_suspicions();

        // Callback should have been invoked
        assert!(count.load(Ordering::SeqCst) >= 1);
    }

    // ========== Gossip Round Tests ==========

    #[tokio::test]
    async fn test_gossip_round_increments_counter() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        let initial = manager.round_count();

        let _ = manager.gossip_round().await;

        assert_eq!(manager.round_count(), initial + 1);
    }

    #[tokio::test]
    async fn test_gossip_round_with_empty_targets() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let mut config = GossipConfig::default();
        config.geometric_routing = false;

        let manager =
            GossipMembershipManager::new("node1".to_string(), config, transport);

        // No peers, so targets will be empty
        let result = manager.gossip_round().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_suspect_node_starts_suspicion() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Add node2
        manager.add_peer("node2".to_string());
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Healthy, 0);
        }

        // Suspect node2
        let _ = manager.suspect_node(&"node2".to_string()).await;

        // Should have suspicion entry
        assert!(manager.suspicions.read().contains_key(&"node2".to_string()));
    }

    // ========== Edge Case Tests ==========

    #[test]
    fn test_add_peer_duplicates_filtered() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        manager.add_peer("node2".to_string());
        manager.add_peer("node2".to_string()); // Duplicate
        manager.add_peer("node2".to_string()); // Another duplicate

        let peers = manager.known_peers.read();
        let count = peers.iter().filter(|p| *p == "node2").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_node_state_nonexistent_returns_none() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        let state = manager.node_state(&"nonexistent".to_string());
        assert!(state.is_none());
    }

    #[test]
    fn test_with_geometric_constructor() {
        use crate::geometric_membership::{
            GeometricMembershipConfig, GeometricMembershipManager,
        };
        use crate::membership::{ClusterConfig, LocalNodeConfig, MembershipManager};
        use crate::network::MemoryTransport;
        use std::net::SocketAddr;

        let transport: Arc<dyn crate::network::Transport> =
            Arc::new(MemoryTransport::new("node1".to_string()));

        // Create proper ClusterConfig
        let local_config = LocalNodeConfig {
            node_id: "node1".to_string(),
            bind_address: "127.0.0.1:8000".parse::<SocketAddr>().unwrap(),
        };
        let cluster_config = ClusterConfig::new("test-cluster", local_config);

        // Create inner membership manager
        let inner = Arc::new(MembershipManager::new(cluster_config, Arc::clone(&transport)));
        let geo_config = GeometricMembershipConfig::default();
        let geometric = Arc::new(GeometricMembershipManager::new(inner, geo_config));

        let manager = GossipMembershipManager::with_geometric(
            "node1".to_string(),
            GossipConfig::default(),
            transport,
            geometric,
        );

        assert!(manager.geometric.is_some());
    }

    #[test]
    fn test_lww_fail_node() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Healthy, 0);

        assert!(state.fail(&"node1".to_string()));
        assert_eq!(
            state.get(&"node1".to_string()).unwrap().health,
            NodeHealth::Failed
        );

        // Fail again should return false (already failed)
        assert!(!state.fail(&"node1".to_string()));
    }

    #[test]
    fn test_lww_fail_nonexistent() {
        let mut state = LWWMembershipState::new();

        // Failing a nonexistent node returns false
        assert!(!state.fail(&"nonexistent".to_string()));
    }

    #[test]
    fn test_lww_suspect_wrong_incarnation() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Healthy, 5);

        // Suspect with wrong incarnation should fail
        assert!(!state.suspect(&"node1".to_string(), 0));
        assert_eq!(
            state.get(&"node1".to_string()).unwrap().health,
            NodeHealth::Healthy
        );
    }

    #[test]
    fn test_lww_refute_lower_incarnation() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Degraded, 5);

        // Refute with lower incarnation should fail
        assert!(!state.refute(&"node1".to_string(), 3));
        assert_eq!(
            state.get(&"node1".to_string()).unwrap().health,
            NodeHealth::Degraded
        );
    }

    #[test]
    fn test_lww_is_empty() {
        let state = LWWMembershipState::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_gossip_config_default() {
        let config = GossipConfig::default();
        assert_eq!(config.fanout, 3);
        assert_eq!(config.gossip_interval_ms, 200);
        assert_eq!(config.suspicion_timeout_ms, 5000);
        assert_eq!(config.max_states_per_message, 20);
        assert!(config.geometric_routing);
        assert_eq!(config.indirect_ping_count, 3);
        assert_eq!(config.indirect_ping_timeout_ms, 500);
    }

    #[test]
    fn test_gossip_node_state_equal_incarnation() {
        let state1 = GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 5, 1);
        let state2 = GossipNodeState::new("node1".to_string(), NodeHealth::Degraded, 5, 1);

        // Same incarnation and timestamp - neither supersedes
        assert!(!state1.supersedes(&state2));
        assert!(!state2.supersedes(&state1));
    }

    #[test]
    fn test_shutdown() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Should not panic
        manager.shutdown();
    }

    #[test]
    fn test_all_states() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        manager.add_peer("node2".to_string());
        manager.add_peer("node3".to_string());

        let states = manager.all_states();
        assert_eq!(states.len(), 3); // node1 + node2 + node3
    }

    #[test]
    fn test_node_count() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        assert_eq!(manager.node_count(), 1); // Just local node

        manager.add_peer("node2".to_string());
        assert_eq!(manager.node_count(), 2);
    }

    #[test]
    fn test_health_counts_from_manager() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Initially just local node (healthy)
        let (healthy, degraded, failed) = manager.health_counts();
        assert_eq!(healthy, 1);
        assert_eq!(degraded, 0);
        assert_eq!(failed, 0);
    }

    #[test]
    fn test_lww_count_by_health_unknown() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Unknown, 0);

        let (healthy, degraded, failed) = state.count_by_health();
        assert_eq!(healthy, 0);
        assert_eq!(degraded, 0);
        assert_eq!(failed, 1); // Unknown counts as failed
    }

    #[test]
    fn test_merge_returns_changed_nodes() {
        let mut state = LWWMembershipState::new();

        let incoming = vec![
            GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 1, 0),
            GossipNodeState::new("node2".to_string(), NodeHealth::Healthy, 1, 0),
        ];

        let changed = state.merge(&incoming);
        assert_eq!(changed.len(), 2);
        assert!(changed.contains(&"node1".to_string()));
        assert!(changed.contains(&"node2".to_string()));
    }

    #[test]
    fn test_merge_empty_incoming() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Healthy, 0);

        let changed = state.merge(&[]);
        assert!(changed.is_empty());
    }
}
