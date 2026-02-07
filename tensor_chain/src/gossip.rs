// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! SWIM gossip protocol for scalable cluster membership and failure detection.
//!
//! # Overview
//!
//! This module implements a SWIM (Scalable Weakly-consistent Infection-style Process
//! Group Membership) protocol variant for cluster membership management. It replaces
//! the traditional O(N) sequential pinging approach with O(log N) epidemic dissemination,
//! enabling efficient scaling to large clusters.
//!
//! Key features:
//! - **Epidemic dissemination**: Membership changes propagate in O(log N) rounds
//! - **CRDT-based state**: Last-Writer-Wins semantics ensure eventual consistency
//! - **Suspicion mechanism**: Indirect probes before marking nodes as failed
//! - **Geometric routing**: Peer sampling weighted by embedding similarity
//!
//! # Architecture
//!
//! ```text
//! +--------------------+
//! |   GossipProtocol   |  Main protocol driver
//! +--------------------+
//!          |
//!          | manages
//!          v
//! +--------------------+     merge()      +--------------------+
//! | LWWMembershipState | <--------------> | LWWMembershipState |
//! | (CRDT)             |                  | (from peer)        |
//! +--------------------+                  +--------------------+
//!          |
//!          | sends
//!          v
//! +--------------------+
//! |   GossipMessage    |
//! +--------------------+
//! | - Sync (states)    |
//! | - Suspect (node)   |
//! | - Alive (refute)   |
//! | - PingReq (indirect)|
//! | - PingAck (response)|
//! +--------------------+
//! ```
//!
//! # SWIM Protocol Overview
//!
//! The protocol operates in rounds, where each round:
//!
//! 1. **Probe Phase**: Select a random peer and send a Sync message
//! 2. **Suspicion Phase**: If no response, send indirect probes via other peers
//! 3. **Failure Phase**: If all probes fail, mark node as suspected/failed
//! 4. **Dissemination**: Piggyback membership changes on all outgoing messages
//!
//! ```text
//! Node A              Node B              Node C
//!   |                   |                   |
//!   |-- Sync ---------->|                   |  (direct probe)
//!   |<-- Sync ----------|                   |  (response with B's state)
//!   |                   |                   |
//!   |-- PingReq(B) -------------------->|  (indirect probe)
//!   |                   |<-- Sync ---------|  (C probes B)
//!   |                   |-- Sync --------->|  (B responds to C)
//!   |<-- PingAck(B,ok) --------------------|  (C reports B alive)
//! ```
//!
//! # LWW-CRDT Membership State
//!
//! Membership state uses a Last-Writer-Wins CRDT with the following ordering:
//!
//! 1. **Incarnation number**: Higher incarnation always wins (for same node)
//! 2. **Lamport timestamp**: Tiebreaker when incarnations are equal
//!
//! This ensures:
//! - A node rejoining the cluster supersedes its old state
//! - Concurrent updates from different observers converge
//! - No coordination required between nodes
//!
//! ```rust
//! use tensor_chain::gossip::{GossipNodeState, LWWMembershipState};
//! use tensor_chain::membership::NodeHealth;
//!
//! let mut state = LWWMembershipState::new();
//!
//! // Update local node
//! state.update_local("node1".to_string(), NodeHealth::Healthy, /*incarnation=*/1);
//!
//! // Merge incoming state from peer
//! let incoming = vec![
//!     GossipNodeState::new("node2".to_string(), NodeHealth::Healthy, 5, 1),
//! ];
//! let changed = state.merge(&incoming);
//! ```
//!
//! # Suspicion and Failure Detection
//!
//! Nodes are not immediately marked as failed when they don't respond:
//!
//! 1. **Direct Probe**: Send Sync, wait for response
//! 2. **Indirect Probe**: If no response, ask K other nodes to probe
//! 3. **Suspect**: If all probes fail, mark as Degraded (suspected)
//! 4. **Refute**: If suspected node responds, it increments incarnation
//! 5. **Fail**: After suspicion timeout, mark as Failed
//!
//! This prevents false positives from temporary network issues.
//!
//! # Message Types
//!
//! | Message | Purpose |
//! |---------|---------|
//! | `Sync` | Exchange membership state (primary dissemination) |
//! | `Suspect` | Report a node as potentially failed |
//! | `Alive` | Refute suspicion (with incremented incarnation) |
//! | `PingReq` | Request indirect probe of a target node |
//! | `PingAck` | Report result of indirect probe |
//!
//! # Configuration
//!
//! ```rust
//! use tensor_chain::gossip::GossipConfig;
//!
//! let config = GossipConfig {
//!     // Gossip interval
//!     gossip_interval_ms: 1000,  // 1 second between rounds
//!
//!     // Failure detection
//!     suspicion_timeout_ms: 5000,  // 5 seconds before marking failed
//!
//!     // Indirect probing
//!     indirect_ping_count: 3,  // Ask 3 peers for indirect probes
//!
//!     // State dissemination
//!     max_states_per_message: 10,  // Max states per message
//!
//!     // Other settings...
//!     ..Default::default()
//! };
//! ```
//!
//! # Security
//!
//! Messages can be signed using [`SignedGossipMessage`]
//! to provide:
//! - Authentication (Ed25519 signatures)
//! - Identity binding (`NodeId` derived from public key)
//! - Replay protection (sequence numbers + timestamps)
//!
//! # See Also
//!
//! - [`crate::membership`]: Cluster membership and health tracking
//! - [`crate::signing`]: Signed gossip message creation
//! - [`crate::geometric_membership`]: Embedding-aware peer selection
//! - [`crate::hlc`]: Hybrid logical clocks for timestamp generation

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
    error::{ChainError, Result},
    geometric_membership::GeometricMembershipManager,
    hlc::HybridLogicalClock,
    membership::{MembershipCallback, NodeHealth},
    network::{Message, Transport},
    signing::{Identity, SequenceTracker, SignedGossipMessage, ValidatorRegistry},
};

/// State of a node in the gossip protocol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    /// Create a new gossip node state.
    ///
    /// Uses the HLC to get a monotonic timestamp. If the HLC fails to initialize
    /// (system time before epoch), falls back to the provided timestamp as the
    /// wall clock time to ensure the state is still usable.
    #[must_use]
    pub fn new(node_id: NodeId, health: NodeHealth, timestamp: u64, incarnation: u64) -> Self {
        // Try to get wall clock time via HLC for monotonic timestamps
        let updated_at =
            HybridLogicalClock::new(0).map_or(timestamp, |hlc| hlc.estimated_wall_ms());

        Self {
            node_id,
            health,
            timestamp,
            updated_at,
            incarnation,
        }
    }

    /// Create a new gossip node state with explicit wall clock time.
    ///
    /// Use this when you have a known-good wall clock time from an HLC.
    #[must_use]
    pub const fn with_wall_time(
        node_id: NodeId,
        health: NodeHealth,
        timestamp: u64,
        incarnation: u64,
        updated_at: u64,
    ) -> Self {
        Self {
            node_id,
            health,
            timestamp,
            updated_at,
            incarnation,
        }
    }

    /// Create a new gossip node state, returning an error if system time is unavailable.
    ///
    /// Prefer this in production code where timestamp accuracy is critical.
    ///
    /// # Errors
    ///
    /// Returns an error if the HLC cannot be initialized (system time unavailable).
    pub fn try_new(
        node_id: NodeId,
        health: NodeHealth,
        timestamp: u64,
        incarnation: u64,
    ) -> Result<Self> {
        let hlc = HybridLogicalClock::new(0)?;
        let updated_at = hlc.estimated_wall_ms();

        Ok(Self {
            node_id,
            health,
            timestamp,
            updated_at,
            incarnation,
        })
    }

    /// Check if this state supersedes another state for the same node.
    /// Uses incarnation first, then timestamp as tiebreaker.
    #[must_use]
    pub const fn supersedes(&self, other: &Self) -> bool {
        if self.incarnation == other.incarnation {
            self.timestamp > other.timestamp
        } else {
            self.incarnation > other.incarnation
        }
    }
}

/// Gossip protocol messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    /// Bidirectional connectivity probe (sent to verify reverse path).
    BidirectionalProbe {
        origin: NodeId,
        probe_id: u64,
        timestamp: u64,
    },
    /// Acknowledgement of a bidirectional probe (proves reverse path works).
    BidirectionalAck {
        origin: NodeId,
        probe_id: u64,
        responder: NodeId,
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn lamport_time(&self) -> u64 {
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

    #[must_use]
    pub fn get(&self, node_id: &NodeId) -> Option<&GossipNodeState> {
        self.states.get(node_id)
    }

    pub fn all_states(&self) -> impl Iterator<Item = &GossipNodeState> {
        self.states.values()
    }

    /// Get states for gossip (up to `max_count`, prioritizing recently updated).
    #[must_use]
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
            let should_update = self.states.get(&state.node_id).map_or(true, |existing| {
                let supersedes = state.supersedes(existing);
                if supersedes {
                    tracing::debug!(
                        node_id = %state.node_id,
                        old_incarnation = existing.incarnation,
                        new_incarnation = state.incarnation,
                        old_timestamp = existing.timestamp,
                        new_timestamp = state.timestamp,
                        "State superseded"
                    );
                }
                supersedes
            });

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
                tracing::debug!(
                    node_id = %node_id,
                    incarnation = incarnation,
                    timestamp = timestamp,
                    "Node suspected"
                );
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
                tracing::warn!(
                    node_id = %node_id,
                    timestamp = timestamp,
                    "Node marked as failed"
                );
                return true;
            }
        }
        false
    }

    /// Refute suspicion with new incarnation.
    pub fn refute(&mut self, node_id: &NodeId, new_incarnation: u64) -> bool {
        // Check if we should refute (without mutable borrow)
        let old_incarnation = self.states.get(node_id).map(|s| s.incarnation);
        let should_refute = old_incarnation.is_some_and(|inc| new_incarnation > inc);

        if should_refute {
            let timestamp = self.tick();
            if let Some(state) = self.states.get_mut(node_id) {
                state.incarnation = new_incarnation;
                state.health = NodeHealth::Healthy;
                state.timestamp = timestamp;
                tracing::debug!(
                    node_id = %node_id,
                    old_incarnation = ?old_incarnation,
                    new_incarnation = new_incarnation,
                    "Suspicion refuted"
                );
                return true;
            }
        }
        false
    }

    /// Mark a node as healthy (used when we have direct evidence the node is alive).
    pub fn mark_healthy(&mut self, node_id: &NodeId) -> bool {
        // Check if we need to update (without holding mutable borrow)
        let needs_update = self
            .states
            .get(node_id)
            .is_some_and(|s| s.health != NodeHealth::Healthy);

        if needs_update {
            let timestamp = self.tick();
            if let Some(state) = self.states.get_mut(node_id) {
                state.health = NodeHealth::Healthy;
                state.timestamp = timestamp;
                tracing::debug!(
                    node_id = %node_id,
                    "Node marked healthy"
                );
                return true;
            }
        }
        false
    }

    /// Count nodes by health status.
    #[must_use]
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

    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    #[must_use]
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
    /// Whether to require Ed25519 signatures on all gossip messages.
    /// When true, unsigned messages are rejected.
    pub require_signatures: bool,
    /// Maximum age of signed messages in milliseconds (default: 5 minutes).
    pub max_message_age_ms: u64,
    /// Timeout for bidirectional probe acknowledgements in milliseconds.
    pub bidirectional_probe_timeout_ms: u64,
    /// Whether to require confirmed bidirectional connectivity before healing.
    pub require_bidirectional: bool,
    /// Maximum allowed incarnation jump per update. Updates exceeding this
    /// delta are rejected to prevent incarnation inflation attacks.
    pub max_incarnation_delta: u64,
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
            require_signatures: false,
            max_message_age_ms: 5 * 60 * 1000, // 5 minutes
            bidirectional_probe_timeout_ms: 1000,
            require_bidirectional: true,
            max_incarnation_delta: 100,
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

/// Tracks node flapping behavior (rapid healthy/failed transitions).
#[derive(Debug, Clone)]
struct FlapRecord {
    /// Number of health state transitions in the tracking window.
    flap_count: u32,
    /// When the last health transition occurred.
    last_transition: Instant,
    /// When the node last became stable (no transitions for the reset window).
    last_stable: Instant,
}

impl FlapRecord {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            flap_count: 0,
            last_transition: now,
            last_stable: now,
        }
    }

    /// Record a health transition and return the new flap count.
    fn record_transition(&mut self) -> u32 {
        self.flap_count += 1;
        self.last_transition = Instant::now();
        self.flap_count
    }

    /// Check if the node has been stable long enough to reset flap count.
    fn maybe_reset(&mut self, stable_window: Duration) -> bool {
        if self.last_transition.elapsed() >= stable_window {
            self.flap_count = 0;
            self.last_stable = Instant::now();
            true
        } else {
            false
        }
    }

    /// Compute backoff duration based on flap count (exponential after threshold).
    fn backoff_duration(&self, threshold: u32) -> Option<Duration> {
        if self.flap_count < threshold {
            return None;
        }
        let excess = self.flap_count.saturating_sub(threshold);
        // 1s, 2s, 4s, 8s, ... capped at 5 minutes
        let secs = 1u64 << excess.min(8);
        Some(Duration::from_secs(secs.min(300)))
    }
}

/// Tracks bidirectional connectivity status for a peer.
#[derive(Debug, Clone)]
pub struct ConnectivityEntry {
    /// Whether we can send to this node (outbound path works).
    pub outbound_ok: bool,
    /// Whether this node can send to us (inbound path works, confirmed by ack).
    pub inbound_ok: bool,
    /// When connectivity was last verified.
    pub last_check: Instant,
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
    /// Pending suspicions (`node_id` -> suspicion record).
    suspicions: RwLock<HashMap<NodeId, PendingSuspicion>>,
    /// Heal progress tracking (`node_id` -> heal progress).
    heal_progress: RwLock<HashMap<NodeId, HealProgress>>,
    /// Known peers for random selection fallback.
    known_peers: RwLock<Vec<NodeId>>,
    /// Shutdown signal.
    shutdown_tx: broadcast::Sender<()>,
    /// Ping sequence counter for indirect pings.
    ping_sequence: AtomicU64,
    /// Number of times the sequence exhaustion threshold was exceeded.
    #[allow(dead_code)]
    sequence_exhaustion_warnings: AtomicU64,
    /// Gossip round counter.
    round_counter: AtomicU64,
    // --- Signing support ---
    /// Optional Ed25519 identity for signing outgoing messages.
    identity: Option<Arc<Identity>>,
    /// Registry of known validators for verifying signatures.
    validator_registry: Option<Arc<ValidatorRegistry>>,
    /// Sequence tracker for replay protection.
    sequence_tracker: Option<Arc<SequenceTracker>>,
    /// Monotonic sequence number for outgoing signed messages.
    outgoing_sequence: AtomicU64,
    /// Bidirectional connectivity matrix.
    connectivity_matrix: RwLock<HashMap<NodeId, ConnectivityEntry>>,
    /// Counter for bidirectional probe IDs.
    probe_id_counter: AtomicU64,
    /// Pending bidirectional probes awaiting acknowledgement.
    pending_probes: RwLock<HashMap<u64, (NodeId, Instant)>>,
    /// Number of asymmetric partitions detected.
    asymmetric_partitions_detected: AtomicU64,
    /// Number of incarnation updates rejected due to exceeding max delta.
    incarnation_rejected: AtomicU64,
    /// Number of signature verification failures.
    signature_verification_failures: AtomicU64,
    /// Per-node flap tracking for detecting rapid health transitions.
    flap_tracker: RwLock<HashMap<NodeId, FlapRecord>>,
    /// Number of times flap backoff was applied.
    flap_backoffs_applied: AtomicU64,
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
            sequence_exhaustion_warnings: AtomicU64::new(0),
            round_counter: AtomicU64::new(0),
            identity: None,
            validator_registry: None,
            sequence_tracker: None,
            outgoing_sequence: AtomicU64::new(0),
            connectivity_matrix: RwLock::new(HashMap::new()),
            probe_id_counter: AtomicU64::new(0),
            pending_probes: RwLock::new(HashMap::new()),
            asymmetric_partitions_detected: AtomicU64::new(0),
            incarnation_rejected: AtomicU64::new(0),
            signature_verification_failures: AtomicU64::new(0),
            flap_tracker: RwLock::new(HashMap::new()),
            flap_backoffs_applied: AtomicU64::new(0),
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

    /// Create with Ed25519 signing support.
    ///
    /// When signing is enabled, all outgoing gossip messages are signed and
    /// incoming messages are verified. Set `config.require_signatures = true`
    /// to reject unsigned messages.
    pub fn with_signing(
        local_node: NodeId,
        config: GossipConfig,
        transport: Arc<dyn Transport>,
        identity: Arc<Identity>,
        validator_registry: Arc<ValidatorRegistry>,
        sequence_tracker: Arc<SequenceTracker>,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);

        let mut state = LWWMembershipState::new();
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
            sequence_exhaustion_warnings: AtomicU64::new(0),
            round_counter: AtomicU64::new(0),
            identity: Some(identity),
            validator_registry: Some(validator_registry),
            sequence_tracker: Some(sequence_tracker),
            outgoing_sequence: AtomicU64::new(0),
            connectivity_matrix: RwLock::new(HashMap::new()),
            probe_id_counter: AtomicU64::new(0),
            pending_probes: RwLock::new(HashMap::new()),
            asymmetric_partitions_detected: AtomicU64::new(0),
            incarnation_rejected: AtomicU64::new(0),
            signature_verification_failures: AtomicU64::new(0),
            flap_tracker: RwLock::new(HashMap::new()),
            flap_backoffs_applied: AtomicU64::new(0),
        }
    }

    /// Check if signatures are required for incoming gossip messages.
    pub const fn require_signatures(&self) -> bool {
        self.config.require_signatures
    }

    /// Returns true if the ping sequence counter has exceeded 90% of `u64::MAX`,
    /// indicating that sequence number exhaustion is approaching.
    pub fn check_sequence_exhaustion(&self) -> bool {
        self.ping_sequence.load(Ordering::Relaxed) > u64::MAX / 10 * 9
    }

    /// Returns the number of times the sequence exhaustion warning was triggered.
    pub fn sequence_exhaustion_warning_count(&self) -> u64 {
        self.sequence_exhaustion_warnings.load(Ordering::Relaxed)
    }

    /// Create a signed or unsigned `Message::Gossip` depending on configuration.
    ///
    /// When `require_signatures` is enabled and signing fails, returns `None`
    /// instead of falling back to unsigned (fail-fast behavior).
    fn create_gossip_message(&self, gossip_msg: GossipMessage) -> Option<Message> {
        if let Some(ref identity) = self.identity {
            let seq = self.outgoing_sequence.fetch_add(1, Ordering::SeqCst);
            match SignedGossipMessage::new(identity, &gossip_msg, seq) {
                Ok(signed) => Some(Message::SignedGossip(signed)),
                Err(e) => {
                    self.signature_verification_failures
                        .fetch_add(1, Ordering::Relaxed);
                    if self.config.require_signatures {
                        tracing::error!(
                            error = %e,
                            "Failed to sign gossip message, dropping (require_signatures=true)"
                        );
                        None
                    } else {
                        tracing::warn!(
                            error = %e,
                            "Failed to sign gossip message, falling back to unsigned"
                        );
                        Some(Message::Gossip(gossip_msg))
                    }
                },
            }
        } else {
            Some(Message::Gossip(gossip_msg))
        }
    }

    /// Handle a signed gossip message.
    ///
    /// Verifies the signature and replay protection, then processes the gossip.
    ///
    /// # Errors
    ///
    /// Returns an error if signing is not configured or signature verification fails.
    pub fn handle_signed_gossip(&self, signed: &SignedGossipMessage) -> Result<()> {
        let (Some(registry), Some(tracker)) = (&self.validator_registry, &self.sequence_tracker)
        else {
            self.signature_verification_failures
                .fetch_add(1, Ordering::Relaxed);
            return Err(ChainError::CryptoError(
                "signing not configured for gossip verification".into(),
            ));
        };

        match signed.verify_with_tracker(registry, tracker) {
            Ok(msg) => {
                self.handle_gossip(msg);
                Ok(())
            },
            Err(e) => {
                self.signature_verification_failures
                    .fetch_add(1, Ordering::Relaxed);
                Err(e)
            },
        }
    }

    /// Add a known peer.
    pub fn add_peer(&self, peer: NodeId) {
        {
            let mut peers = self.known_peers.write();
            if !peers.contains(&peer) {
                peers.push(peer.clone());
            }
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

    pub fn node_state(&self, node_id: &NodeId) -> Option<GossipNodeState> {
        self.state.read().get(node_id).cloned()
    }

    pub fn all_states(&self) -> Vec<GossipNodeState> {
        self.state.read().all_states().cloned().collect()
    }

    pub fn node_count(&self) -> usize {
        self.state.read().len()
    }

    pub fn health_counts(&self) -> (usize, usize, usize) {
        self.state.read().count_by_health()
    }

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
                        tracing::debug!(
                            count = targets.len(),
                            targets = ?targets,
                            "Selected targets via geometric routing"
                        );
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
        let peer_count = peers.len();
        let mut selected = Vec::with_capacity(k.min(peer_count));

        for i in 0..k.min(peer_count) {
            #[allow(clippy::cast_possible_truncation)] // i < peer_count which fits in usize
            let idx = (round.wrapping_add(i as u64) as usize) % peer_count;
            let peer = &peers[idx];
            if peer != &self.local_node && !selected.contains(peer) {
                selected.push(peer.clone());
            }
        }
        drop(peers);

        if !selected.is_empty() {
            tracing::debug!(
                count = selected.len(),
                targets = ?selected,
                "Selected targets via random fallback"
            );
        }

        selected
    }

    /// Run a single gossip round.
    ///
    /// # Errors
    ///
    /// Currently infallible but returns `Result` for future error propagation.
    #[allow(clippy::unused_async)] // Async for API consistency; spawns async tasks internally
    pub async fn gossip_round(&self) -> Result<()> {
        let round = self.round_counter.fetch_add(1, Ordering::Relaxed) + 1;

        let targets = self.select_gossip_targets(self.config.fanout);
        if targets.is_empty() {
            return Ok(());
        }

        let states = self
            .state
            .read()
            .states_for_gossip(self.config.max_states_per_message);
        let sender_time = self.state.read().lamport_time();

        tracing::debug!(
            round = round,
            target_count = targets.len(),
            state_count = states.len(),
            lamport_time = sender_time,
            "Starting gossip round"
        );

        let Some(msg) = self.create_gossip_message(GossipMessage::Sync {
            sender: self.local_node.clone(),
            states,
            sender_time,
        }) else {
            return Ok(());
        };

        // Send in parallel using tokio spawn (fire-and-forget)
        for target in targets {
            let transport = Arc::clone(&self.transport);
            let msg = msg.clone();
            let target_clone = target.clone();
            tokio::spawn(async move {
                if let Err(e) = transport.send(&target_clone, msg).await {
                    tracing::debug!(peer = %target_clone, error = %e, "failed to send gossip sync");
                }
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
                self.handle_sync(&sender, &states, sender_time);
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
            GossipMessage::BidirectionalProbe {
                origin,
                probe_id,
                timestamp,
            } => {
                self.handle_bidirectional_probe(&origin, probe_id, timestamp);
            },
            GossipMessage::BidirectionalAck {
                origin,
                probe_id,
                responder,
            } => {
                self.handle_bidirectional_ack(&origin, probe_id, &responder);
            },
        }
    }

    fn handle_sync(&self, sender: &NodeId, states: &[GossipNodeState], sender_time: u64) {
        tracing::debug!(
            sender = %sender,
            state_count = states.len(),
            sender_time = sender_time,
            "Gossip sync received"
        );

        let changed = {
            let mut state = self.state.write();
            state.sync_time(sender_time);
            // Filter out states with incarnation jumps exceeding the configured delta
            let max_delta = self.config.max_incarnation_delta;
            let filtered: Vec<GossipNodeState> = states
                .iter()
                .filter(|incoming| {
                    let delta = state.get(&incoming.node_id).map_or(
                        incoming.incarnation, // new node: accept if incarnation <= max_delta
                        |existing| incoming.incarnation.saturating_sub(existing.incarnation),
                    );
                    if delta > max_delta {
                        self.incarnation_rejected.fetch_add(1, Ordering::Relaxed);
                        tracing::warn!(
                            node_id = %incoming.node_id,
                            delta,
                            max_delta,
                            "Rejected incarnation update: delta exceeds maximum"
                        );
                        false
                    } else {
                        true
                    }
                })
                .cloned()
                .collect();
            state.merge(&filtered)
        };

        if !changed.is_empty() {
            tracing::debug!(
                updated_count = changed.len(),
                updated_nodes = ?changed,
                "States updated from gossip sync"
            );
        }

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
        let sender_incarnation = self.state.read().get(sender).map_or(0, |s| s.incarnation);

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

    fn handle_suspect(&self, reporter: &NodeId, suspect: &NodeId, incarnation: u64) {
        // If we're being suspected, refute it
        if suspect == &self.local_node {
            let new_incarnation = self.incarnation.fetch_add(1, Ordering::SeqCst) + 1;
            tracing::info!(
                reporter = %reporter,
                old_incarnation = incarnation,
                new_incarnation = new_incarnation,
                "Refuting suspicion against self"
            );
            self.broadcast_alive(new_incarnation);
            return;
        }

        // Start suspicion timer
        let mut suspicions = self.suspicions.write();
        if !suspicions.contains_key(suspect) {
            tracing::debug!(
                suspect = %suspect,
                reporter = %reporter,
                incarnation = incarnation,
                "Starting suspicion timer for node"
            );
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
        // Check incarnation delta before accepting alive message
        let current_incarnation = self.state.read().get(node_id).map_or(0, |s| s.incarnation);
        let delta = incarnation.saturating_sub(current_incarnation);
        if delta > self.config.max_incarnation_delta {
            self.incarnation_rejected.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                node_id = %node_id,
                delta,
                max_delta = self.config.max_incarnation_delta,
                "Rejected alive message: incarnation delta exceeds maximum"
            );
            return;
        }

        if self.state.write().refute(node_id, incarnation) {
            tracing::debug!(
                node_id = %node_id,
                incarnation = incarnation,
                "Alive received, clearing suspicion"
            );
            self.suspicions.write().remove(node_id);
            // Track health transition for flap detection (suspected → alive)
            self.record_flap(node_id);
        }
    }

    fn handle_ping_req(&self, origin: &NodeId, target: &NodeId, sequence: u64) {
        tracing::debug!(
            origin = %origin,
            target = %target,
            sequence = sequence,
            "Handling indirect ping request"
        );

        // Try to ping the target
        let transport = Arc::clone(&self.transport);
        let origin_clone = origin.clone();
        let target_clone = target.clone();
        let local = self.local_node.clone();

        // Pre-create signed message templates for both success and failure
        let Some(ack_success) = self.create_gossip_message(GossipMessage::PingAck {
            origin: local.clone(),
            target: target.clone(),
            sequence,
            success: true,
        }) else {
            return;
        };
        let Some(ack_failure) = self.create_gossip_message(GossipMessage::PingAck {
            origin: local,
            target: target.clone(),
            sequence,
            success: false,
        }) else {
            return;
        };

        tokio::spawn(async move {
            let success = transport
                .send(&target_clone, Message::Ping { term: 0 })
                .await
                .is_ok();

            // Send ack back to origin
            let ack_msg = if success { ack_success } else { ack_failure };
            let _ = transport.send(&origin_clone, ack_msg).await;
        });
    }

    fn handle_ping_ack(&self, origin: &NodeId, target: &NodeId, sequence: u64, success: bool) {
        tracing::debug!(
            origin = %origin,
            target = %target,
            sequence = sequence,
            success = success,
            "Handling indirect ping ack"
        );

        if success {
            // Target is alive - clear suspicion directly
            // We have direct evidence the node is alive from the successful ping
            if self.state.read().get(target).is_some() {
                self.state.write().mark_healthy(target);
                self.suspicions.write().remove(target);
                tracing::debug!(
                    target = %target,
                    "Suspicion cleared via successful ping ack"
                );
            }
        }
    }

    /// Broadcast alive message to refute suspicion.
    fn broadcast_alive(&self, incarnation: u64) {
        let Some(msg) = self.create_gossip_message(GossipMessage::Alive {
            node_id: self.local_node.clone(),
            incarnation,
        }) else {
            return;
        };

        let targets = self.select_gossip_targets(self.config.fanout);
        let transport = Arc::clone(&self.transport);

        tokio::spawn(async move {
            for target in targets {
                if let Err(e) = transport.send(&target, msg.clone()).await {
                    tracing::debug!(peer = %target, error = %e, "failed to broadcast gossip");
                }
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
                    to_fail.push((node_id.clone(), suspicion.started_at));
                }
            }
        }

        for (node_id, started_at) in to_fail {
            #[allow(clippy::cast_possible_truncation)] // Duration in ms always fits u64
            let elapsed_ms = now.duration_since(started_at).as_millis() as u64;
            tracing::warn!(
                node_id = %node_id,
                timeout_ms = self.config.suspicion_timeout_ms,
                elapsed_ms = elapsed_ms,
                "Suspicion timeout, marking node as failed"
            );
            self.suspicions.write().remove(&node_id);
            // Track health transition for flap detection (suspected → failed)
            self.record_flap(&node_id);
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
    ///
    /// # Errors
    ///
    /// Returns an error if gossip messages fail to send to all targets.
    #[allow(clippy::significant_drop_tightening)] // Lock scope is already a minimal block
    pub async fn suspect_node(&self, node_id: &NodeId) -> Result<()> {
        let incarnation = self.state.read().get(node_id).map_or(0, |s| s.incarnation);

        tracing::debug!(
            node_id = %node_id,
            incarnation = incarnation,
            "Initiating suspicion protocol for node"
        );

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
                // Track health transition for flap detection (healthy → suspected)
                self.record_flap(node_id);
            }
        }

        // Broadcast suspicion
        if let Some(msg) = self.create_gossip_message(GossipMessage::Suspect {
            reporter: self.local_node.clone(),
            suspect: node_id.clone(),
            incarnation,
        }) {
            let targets = self.select_gossip_targets(self.config.fanout);
            for target in targets {
                if let Err(e) = self.transport.send(&target, msg.clone()).await {
                    tracing::debug!(peer = %target, error = %e, "failed to send suspect message");
                }
            }
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

        if sequence > u64::MAX / 10 * 9 {
            self.sequence_exhaustion_warnings
                .fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                sequence = sequence,
                "Ping sequence counter approaching u64::MAX (>90% exhausted)"
            );
        }

        let Some(msg) = self.create_gossip_message(GossipMessage::PingReq {
            origin: self.local_node.clone(),
            target: target.clone(),
            sequence,
        }) else {
            return;
        };

        for intermediary in intermediaries {
            if let Err(e) = self.transport.send(&intermediary, msg.clone()).await {
                tracing::debug!(peer = %intermediary, error = %e, "failed to send indirect ping");
            }
        }
    }

    /// Run the gossip protocol loop.
    ///
    /// # Errors
    ///
    /// Currently infallible but returns `Result` for future error propagation.
    pub async fn run(&self) -> Result<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let interval = Duration::from_millis(self.config.gossip_interval_ms);

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    break;
                }
                () = tokio::time::sleep(interval) => {
                    let _ = self.gossip_round().await;
                }
            }
        }

        Ok(())
    }

    /// Shutdown the gossip manager.
    pub fn shutdown(&self) {
        // Receiver may already be dropped during shutdown
        self.shutdown_tx.send(()).ok();
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
            tracing::debug!(
                node_id = %node,
                consecutive_successes = hp.consecutive_successes,
                "Heal progress recorded"
            );
        } else if was_failed || partition_start.is_some() {
            // Start tracking heal progress
            let start = partition_start.unwrap_or_else(Instant::now);
            tracing::debug!(
                node_id = %node,
                was_failed = was_failed,
                "Heal progress tracking started"
            );
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
        let hp = progress.get(node)?;

        if hp.consecutive_successes >= threshold {
            // Check bidirectional connectivity if required
            if !self.is_bidirectional_confirmed(node) {
                tracing::debug!(
                    node_id = %node,
                    "Heal threshold met but bidirectional connectivity not confirmed"
                );
                return None;
            }

            #[allow(clippy::cast_possible_truncation)] // Duration in ms always fits u64
            let partition_duration_ms = hp.partition_start.elapsed().as_millis() as u64;
            let consecutive = hp.consecutive_successes;
            drop(progress);
            tracing::info!(
                node_id = %node,
                threshold = threshold,
                consecutive_successes = consecutive,
                partition_duration_ms = partition_duration_ms,
                "Heal confirmed"
            );
            return Some(partition_duration_ms);
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

    pub fn healing_nodes(&self) -> Vec<(NodeId, u32)> {
        self.heal_progress
            .read()
            .iter()
            .map(|(id, hp)| (id.clone(), hp.consecutive_successes))
            .collect()
    }

    /// Send a bidirectional connectivity probe to a node.
    ///
    /// The probe verifies that the target can receive our messages AND send
    /// messages back to us, detecting asymmetric partitions where A can reach
    /// B but B cannot reach A.
    #[allow(clippy::significant_drop_tightening)] // Lock scope is already a minimal block
    pub fn send_bidirectional_probe(&self, target: &NodeId) {
        let probe_id = self.probe_id_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp = self.state.read().lamport_time();

        self.pending_probes
            .write()
            .insert(probe_id, (target.clone(), Instant::now()));

        // Mark outbound as ok (we're sending)
        {
            let mut matrix = self.connectivity_matrix.write();
            let entry = matrix
                .entry(target.clone())
                .or_insert_with(|| ConnectivityEntry {
                    outbound_ok: false,
                    inbound_ok: false,
                    last_check: Instant::now(),
                });
            entry.outbound_ok = true;
            entry.last_check = Instant::now();
        }

        let Some(msg) = self.create_gossip_message(GossipMessage::BidirectionalProbe {
            origin: self.local_node.clone(),
            probe_id,
            timestamp,
        }) else {
            return;
        };

        let transport = Arc::clone(&self.transport);
        let target_clone = target.clone();
        tokio::spawn(async move {
            if let Err(e) = transport.send(&target_clone, msg).await {
                tracing::debug!(
                    peer = %target_clone,
                    error = %e,
                    "Failed to send bidirectional probe"
                );
            }
        });
    }

    fn handle_bidirectional_probe(&self, origin: &NodeId, probe_id: u64, _timestamp: u64) {
        tracing::debug!(
            origin = %origin,
            probe_id = probe_id,
            "Received bidirectional probe, sending ack"
        );

        let Some(msg) = self.create_gossip_message(GossipMessage::BidirectionalAck {
            origin: origin.clone(),
            probe_id,
            responder: self.local_node.clone(),
        }) else {
            return;
        };

        let transport = Arc::clone(&self.transport);
        let origin_clone = origin.clone();
        tokio::spawn(async move {
            if let Err(e) = transport.send(&origin_clone, msg).await {
                tracing::debug!(
                    peer = %origin_clone,
                    error = %e,
                    "Failed to send bidirectional ack"
                );
            }
        });
    }

    #[allow(clippy::significant_drop_tightening)] // Lock scope matches usage
    fn handle_bidirectional_ack(&self, _origin: &NodeId, probe_id: u64, responder: &NodeId) {
        // Remove from pending
        let pending = self.pending_probes.write().remove(&probe_id);
        if let Some((expected_target, _sent_at)) = pending {
            if &expected_target == responder {
                tracing::debug!(
                    responder = %responder,
                    probe_id = probe_id,
                    "Bidirectional connectivity confirmed"
                );
                let mut matrix = self.connectivity_matrix.write();
                let entry = matrix
                    .entry(responder.clone())
                    .or_insert_with(|| ConnectivityEntry {
                        outbound_ok: true,
                        inbound_ok: false,
                        last_check: Instant::now(),
                    });
                entry.inbound_ok = true;
                entry.last_check = Instant::now();
            }
        }
    }

    /// Expire pending bidirectional probes that have timed out.
    ///
    /// Probes that don't receive an ack within the timeout indicate that the
    /// reverse path (target -> us) is broken, i.e., asymmetric partition.
    #[allow(clippy::significant_drop_tightening)] // Multiple guards needed in cleanup loop
    pub fn expire_bidirectional_probes(&self) {
        let timeout = Duration::from_millis(self.config.bidirectional_probe_timeout_ms);
        let now = Instant::now();

        let mut expired = Vec::new();
        {
            let pending = self.pending_probes.read();
            for (probe_id, (target, sent_at)) in pending.iter() {
                if now.duration_since(*sent_at) >= timeout {
                    expired.push((*probe_id, target.clone()));
                }
            }
        }

        if !expired.is_empty() {
            let mut pending = self.pending_probes.write();
            let mut matrix = self.connectivity_matrix.write();
            for (probe_id, target) in &expired {
                pending.remove(probe_id);
                let entry = matrix
                    .entry(target.clone())
                    .or_insert_with(|| ConnectivityEntry {
                        outbound_ok: true,
                        inbound_ok: false,
                        last_check: Instant::now(),
                    });
                // Outbound worked (we sent it) but inbound failed (no ack)
                entry.inbound_ok = false;
                entry.last_check = now;

                self.asymmetric_partitions_detected
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    node_id = %target,
                    probe_id = probe_id,
                    "Asymmetric partition detected: outbound OK but no inbound ack"
                );
            }
        }
    }

    /// Check whether bidirectional connectivity to a node has been confirmed.
    ///
    /// Returns `true` if both outbound and inbound paths are verified, or if
    /// `require_bidirectional` is disabled in config.
    pub fn is_bidirectional_confirmed(&self, node: &NodeId) -> bool {
        if !self.config.require_bidirectional {
            return true;
        }

        let matrix = self.connectivity_matrix.read();
        matrix
            .get(node)
            .is_some_and(|entry| entry.outbound_ok && entry.inbound_ok)
    }

    /// Get the connectivity status for a node.
    pub fn connectivity_status(&self, node: &NodeId) -> Option<ConnectivityEntry> {
        self.connectivity_matrix.read().get(node).cloned()
    }

    /// Get the number of asymmetric partitions detected.
    pub fn asymmetric_partition_count(&self) -> u64 {
        self.asymmetric_partitions_detected.load(Ordering::Relaxed)
    }

    /// Get the number of incarnation updates rejected due to exceeding max delta.
    pub fn incarnation_rejected_count(&self) -> u64 {
        self.incarnation_rejected.load(Ordering::Relaxed)
    }

    /// Get the number of signature verification failures.
    pub fn signature_verification_failure_count(&self) -> u64 {
        self.signature_verification_failures.load(Ordering::Relaxed)
    }

    /// Record a health transition for flap detection.
    /// Returns the backoff duration if the node is flapping, or `None` if normal.
    #[allow(clippy::significant_drop_tightening)] // Entry borrows from write guard
    fn record_flap(&self, node_id: &NodeId) -> Option<Duration> {
        let mut tracker = self.flap_tracker.write();
        let record = tracker
            .entry(node_id.clone())
            .or_insert_with(FlapRecord::new);

        // Reset if stable for 5 minutes
        record.maybe_reset(Duration::from_secs(300));

        let count = record.record_transition();
        let backoff = record.backoff_duration(5); // threshold: 5 flaps

        if let Some(ref duration) = backoff {
            self.flap_backoffs_applied.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(
                node_id = %node_id,
                flap_count = count,
                backoff_secs = duration.as_secs(),
                "Node flapping detected, applying exponential backoff"
            );
        }
        backoff
    }

    /// Check if a node is currently in flap backoff.
    pub fn is_in_flap_backoff(&self, node_id: &NodeId) -> bool {
        let tracker = self.flap_tracker.read();
        tracker
            .get(node_id)
            .and_then(|r| r.backoff_duration(5))
            .is_some()
    }

    /// Get the flap count for a node.
    pub fn flap_count(&self, node_id: &NodeId) -> u32 {
        self.flap_tracker
            .read()
            .get(node_id)
            .map_or(0, |r| r.flap_count)
    }

    /// Get the number of times flap backoff was applied.
    pub fn flap_backoffs_count(&self) -> u64 {
        self.flap_backoffs_applied.load(Ordering::Relaxed)
    }

    /// Reset flap tracking for all nodes that have been stable.
    pub fn reset_stable_flap_records(&self) {
        let mut tracker = self.flap_tracker.write();
        tracker.retain(|_, record| {
            record.maybe_reset(Duration::from_secs(300));
            record.flap_count > 0
        });
    }

    /// Clear connectivity state for a node (e.g., after partition fully heals).
    pub fn clear_connectivity(&self, node: &NodeId) {
        self.connectivity_matrix.write().remove(node);
    }

    pub fn lamport_time(&self) -> u64 {
        self.state.read().lamport_time()
    }

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
            let bytes = bitcode::serialize(&msg).unwrap();
            let restored: GossipMessage = bitcode::deserialize(&bytes).unwrap();
            assert_eq!(msg, restored);
        }
    }

    #[test]
    fn test_gossip_node_state_serialization() {
        let state = GossipNodeState::new("test".to_string(), NodeHealth::Healthy, 100, 5);

        let bytes = bitcode::serialize(&state).unwrap();
        let restored: GossipNodeState = bitcode::deserialize(&bytes).unwrap();

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
        let config = GossipConfig {
            require_bidirectional: false,
            ..GossipConfig::default()
        };
        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

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
        let config = GossipConfig {
            geometric_routing: false,
            fanout: 2,
            ..Default::default()
        };

        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

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
        let config = GossipConfig {
            geometric_routing: false,
            ..Default::default()
        };

        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

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
        let config = GossipConfig {
            geometric_routing: false,
            ..Default::default()
        };

        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

        let targets = manager.select_gossip_targets(3);
        assert!(targets.is_empty());
    }

    #[test]
    fn test_select_gossip_targets_no_duplicates() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = GossipConfig {
            geometric_routing: false,
            fanout: 3,
            ..Default::default()
        };

        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

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
        let config = GossipConfig {
            suspicion_timeout_ms: 1, // 1ms timeout for quick test
            ..Default::default()
        };

        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

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
        let config = GossipConfig {
            suspicion_timeout_ms: 60000, // 60s timeout
            ..Default::default()
        };

        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

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
        let config = GossipConfig {
            suspicion_timeout_ms: 1,
            ..Default::default()
        };

        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

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
        let config = GossipConfig {
            geometric_routing: false,
            ..Default::default()
        };

        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

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
        use crate::geometric_membership::{GeometricMembershipConfig, GeometricMembershipManager};
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
        let inner = Arc::new(MembershipManager::new(
            cluster_config,
            Arc::clone(&transport),
        ));
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

    #[test]
    fn test_lww_suspect_already_failed() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Failed, 0);

        // Suspecting a failed node should return false
        assert!(!state.suspect(&"node1".to_string(), 0));
    }

    #[test]
    fn test_lww_suspect_nonexistent_node() {
        let mut state = LWWMembershipState::new();
        assert!(!state.suspect(&"nonexistent".to_string(), 0));
    }

    #[test]
    fn test_lww_refute_same_incarnation() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Degraded, 1);

        // Refuting with same incarnation should return false
        assert!(!state.refute(&"node1".to_string(), 1));
    }

    #[test]
    fn test_lww_refute_nonexistent_node() {
        let mut state = LWWMembershipState::new();
        assert!(!state.refute(&"nonexistent".to_string(), 1));
    }

    #[test]
    fn test_gossip_message_debug_clone() {
        let msg = GossipMessage::Sync {
            sender: "node1".to_string(),
            states: vec![],
            sender_time: 0,
        };
        let cloned = msg.clone();
        let debug = format!("{:?}", cloned);
        assert!(debug.contains("Sync"));
    }

    #[test]
    fn test_gossip_node_state_debug_clone() {
        let state = GossipNodeState::new("node1".to_string(), NodeHealth::Healthy, 1, 0);
        let cloned = state.clone();
        assert_eq!(state.node_id, cloned.node_id);
        let debug = format!("{:?}", state);
        assert!(debug.contains("GossipNodeState"));
    }

    #[test]
    fn test_gossip_config_clone_debug() {
        let config = GossipConfig::default();
        let cloned = config.clone();
        assert_eq!(config.fanout, cloned.fanout);
        let debug = format!("{:?}", config);
        assert!(debug.contains("GossipConfig"));
    }

    #[test]
    fn test_lww_membership_state_debug_clone() {
        let state = LWWMembershipState::new();
        let cloned = state.clone();
        assert_eq!(state.len(), cloned.len());
        let debug = format!("{:?}", state);
        assert!(debug.contains("LWWMembershipState"));
    }

    #[test]
    fn test_gossip_manager_register_callback() {
        use crate::membership::ClusterView;
        use crate::network::MemoryTransport;
        use std::sync::atomic::AtomicBool;

        struct TestCallback {
            called: Arc<AtomicBool>,
        }

        impl MembershipCallback for TestCallback {
            fn on_health_change(&self, _node_id: &NodeId, _old: NodeHealth, _new: NodeHealth) {
                self.called.store(true, Ordering::SeqCst);
            }

            fn on_view_change(&self, _view: &ClusterView) {
                // No-op for test
            }
        }

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        let called = Arc::new(AtomicBool::new(false));
        let callback = Arc::new(TestCallback {
            called: called.clone(),
        });

        manager.register_callback(callback);

        // Trigger a health change by handling a sync message
        let states = vec![GossipNodeState::new(
            "node2".to_string(),
            NodeHealth::Healthy,
            1,
            0,
        )];
        let msg = GossipMessage::Sync {
            sender: "node2".to_string(),
            states,
            sender_time: 1,
        };
        manager.handle_gossip(msg);

        assert!(called.load(Ordering::SeqCst));
    }

    #[test]
    fn test_gossip_manager_node_state() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Local node should exist
        let state = manager.node_state(&"node1".to_string());
        assert!(state.is_some());
        assert_eq!(state.unwrap().health, NodeHealth::Healthy);

        // Non-existent node
        assert!(manager.node_state(&"nonexistent".to_string()).is_none());
    }

    #[test]
    fn test_gossip_manager_handle_alive() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Add and suspect a peer
        manager.add_peer("node2".to_string());
        {
            let mut state = manager.state.write();
            state.suspect(&"node2".to_string(), 0);
        }

        // Handle alive message
        let msg = GossipMessage::Alive {
            node_id: "node2".to_string(),
            incarnation: 1,
        };
        manager.handle_gossip(msg);

        // Verify node is healthy again
        let state = manager.node_state(&"node2".to_string());
        assert!(state.is_some());
        assert_eq!(state.unwrap().health, NodeHealth::Healthy);
    }

    #[tokio::test]
    async fn test_gossip_manager_handle_suspect_self() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Handle suspect message targeting self
        let msg = GossipMessage::Suspect {
            reporter: "node2".to_string(),
            suspect: "node1".to_string(),
            incarnation: 0,
        };

        // Should not panic and should increment incarnation
        let old_incarnation = manager.incarnation.load(Ordering::SeqCst);
        manager.handle_gossip(msg);
        let new_incarnation = manager.incarnation.load(Ordering::SeqCst);
        assert!(new_incarnation > old_incarnation);
    }

    #[test]
    fn test_gossip_manager_handle_suspect_other() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Add a peer
        manager.add_peer("node2".to_string());

        // Handle suspect message
        let msg = GossipMessage::Suspect {
            reporter: "node3".to_string(),
            suspect: "node2".to_string(),
            incarnation: 0,
        };
        manager.handle_gossip(msg);

        // Verify suspicion is tracked
        assert!(manager.suspicions.read().contains_key(&"node2".to_string()));
    }

    #[test]
    fn test_gossip_manager_round_count() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        assert_eq!(manager.round_count(), 0);

        // Increment via internal counter
        manager.round_counter.fetch_add(1, Ordering::SeqCst);
        assert_eq!(manager.round_count(), 1);
    }

    #[test]
    fn test_gossip_manager_select_targets_empty() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // No peers, should return empty
        let targets = manager.select_gossip_targets(3);
        assert!(targets.is_empty());
    }

    #[test]
    fn test_gossip_manager_select_targets_with_peers() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = GossipConfig {
            geometric_routing: false, // Disable geometric to test fallback
            ..Default::default()
        };
        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

        manager.add_peer("node2".to_string());
        manager.add_peer("node3".to_string());
        manager.add_peer("node4".to_string());

        let targets = manager.select_gossip_targets(2);
        assert!(targets.len() <= 2);
        assert!(!targets.contains(&"node1".to_string())); // Local node excluded
    }

    #[test]
    fn test_pending_suspicion_struct() {
        let suspicion = PendingSuspicion {
            started_at: Instant::now(),
            incarnation: 5,
        };

        assert_eq!(suspicion.incarnation, 5);
        // started_at should be approximately now
        assert!(suspicion.started_at.elapsed().as_secs() < 1);
    }

    #[test]
    fn test_handle_ping_ack_success() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Add a suspected peer
        manager.add_peer("node2".to_string());
        {
            let mut state = manager.state.write();
            state.suspect(&"node2".to_string(), 0);
        }
        manager.suspicions.write().insert(
            "node2".to_string(),
            PendingSuspicion {
                started_at: Instant::now(),
                incarnation: 0,
            },
        );

        // Handle ping ack with success
        let msg = GossipMessage::PingAck {
            origin: "node1".to_string(),
            target: "node2".to_string(),
            sequence: 1,
            success: true,
        };
        manager.handle_gossip(msg);

        // Suspicion should be cleared
        assert!(!manager.suspicions.read().contains_key(&"node2".to_string()));
    }

    #[test]
    fn test_handle_ping_ack_failure() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Add a suspected peer
        manager.add_peer("node2".to_string());
        manager.suspicions.write().insert(
            "node2".to_string(),
            PendingSuspicion {
                started_at: Instant::now(),
                incarnation: 0,
            },
        );

        // Handle ping ack with failure
        let msg = GossipMessage::PingAck {
            origin: "node1".to_string(),
            target: "node2".to_string(),
            sequence: 1,
            success: false,
        };
        manager.handle_gossip(msg);

        // Suspicion should still be present
        assert!(manager.suspicions.read().contains_key(&"node2".to_string()));
    }

    #[test]
    fn test_add_peer_already_exists() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        manager.add_peer("node2".to_string());
        let count_before = manager.known_peers.read().len();

        manager.add_peer("node2".to_string()); // Add again
        let count_after = manager.known_peers.read().len();

        assert_eq!(count_before, count_after); // No duplicate
    }

    #[test]
    fn test_gossip_manager_create_gossip_message() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        let inner = GossipMessage::Sync {
            sender: "node1".to_string(),
            states: vec![],
            sender_time: 0,
        };

        let msg = manager
            .create_gossip_message(inner)
            .expect("signing not configured");
        if let Message::Gossip(gossip) = msg {
            assert!(matches!(gossip, GossipMessage::Sync { .. }));
        } else {
            panic!("Expected Gossip message");
        }
    }

    #[test]
    fn test_gossip_config_custom() {
        let config = GossipConfig {
            fanout: 5,
            gossip_interval_ms: 100,
            suspicion_timeout_ms: 3000,
            max_states_per_message: 10,
            geometric_routing: false,
            indirect_ping_count: 2,
            indirect_ping_timeout_ms: 250,
            require_signatures: true,
            max_message_age_ms: 60000,
            bidirectional_probe_timeout_ms: 2000,
            require_bidirectional: true,
            max_incarnation_delta: 50,
        };

        assert_eq!(config.fanout, 5);
        assert_eq!(config.gossip_interval_ms, 100);
        assert!(!config.geometric_routing);
        assert!(config.require_signatures);
        assert_eq!(config.bidirectional_probe_timeout_ms, 2000);
        assert!(config.require_bidirectional);
        assert_eq!(config.max_incarnation_delta, 50);
    }

    #[test]
    fn test_lww_all_states_iterator() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Healthy, 0);
        state.update_local("node2".to_string(), NodeHealth::Degraded, 0);

        let all: Vec<_> = state.all_states().collect();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_gossip_node_state_with_wall_time() {
        let state =
            GossipNodeState::with_wall_time("node1".to_string(), NodeHealth::Healthy, 100, 1, 200);

        assert_eq!(state.node_id, "node1");
        assert_eq!(state.health, NodeHealth::Healthy);
        assert_eq!(state.timestamp, 100);
        assert_eq!(state.incarnation, 1);
        assert_eq!(state.updated_at, 200);
    }

    #[test]
    fn test_gossip_node_state_try_new() {
        let result = GossipNodeState::try_new("node1".to_string(), NodeHealth::Healthy, 100, 1);

        assert!(result.is_ok());
        let state = result.unwrap();
        assert_eq!(state.node_id, "node1");
        assert_eq!(state.health, NodeHealth::Healthy);
        assert_eq!(state.timestamp, 100);
        assert_eq!(state.incarnation, 1);
    }

    #[test]
    fn test_gossip_node_state_supersedes_higher_incarnation() {
        let older =
            GossipNodeState::with_wall_time("node1".to_string(), NodeHealth::Healthy, 100, 1, 100);
        let newer_incarnation = GossipNodeState::with_wall_time(
            "node1".to_string(),
            NodeHealth::Failed,
            50, // older timestamp but higher incarnation
            2,
            50,
        );

        assert!(newer_incarnation.supersedes(&older));
        assert!(!older.supersedes(&newer_incarnation));
    }

    #[test]
    fn test_gossip_node_state_supersedes_newer_timestamp() {
        let older =
            GossipNodeState::with_wall_time("node1".to_string(), NodeHealth::Healthy, 100, 1, 100);
        let newer_timestamp = GossipNodeState::with_wall_time(
            "node1".to_string(),
            NodeHealth::Failed,
            200,
            1, // same incarnation
            200,
        );

        assert!(newer_timestamp.supersedes(&older));
        assert!(!older.supersedes(&newer_timestamp));
    }

    #[test]
    fn test_lww_membership_state_get() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Healthy, 0);
        state.update_local("node2".to_string(), NodeHealth::Degraded, 0);

        assert_eq!(state.all_states().count(), 2);

        let node1 = state.get(&"node1".to_string());
        assert!(node1.is_some());
        assert_eq!(node1.unwrap().health, NodeHealth::Healthy);
    }

    #[test]
    fn test_gossip_message_suspect_debug() {
        let msg = GossipMessage::Suspect {
            reporter: "node1".to_string(),
            suspect: "node2".to_string(),
            incarnation: 1,
        };

        let debug = format!("{:?}", msg);
        assert!(debug.contains("Suspect"));
    }

    #[test]
    fn test_gossip_message_alive_debug() {
        let msg = GossipMessage::Alive {
            node_id: "node1".to_string(),
            incarnation: 1,
        };

        let debug = format!("{:?}", msg);
        assert!(debug.contains("Alive"));
    }

    #[test]
    fn test_lww_membership_state_states_for_gossip() {
        let mut state = LWWMembershipState::new();
        state.update_local("node1".to_string(), NodeHealth::Healthy, 0);
        state.update_local("node2".to_string(), NodeHealth::Degraded, 0);
        state.update_local("node3".to_string(), NodeHealth::Failed, 0);

        // Request max 2 states
        let gossip_states = state.states_for_gossip(2);
        assert_eq!(gossip_states.len(), 2);
    }

    #[test]
    fn test_gossip_signing_and_handle_signed() {
        use crate::network::MemoryTransport;

        let identity = Arc::new(Identity::generate());
        let node_id = identity.node_id();
        let transport = Arc::new(MemoryTransport::new(node_id.clone()));
        let registry = Arc::new(ValidatorRegistry::new());
        registry.register(&identity);
        let tracker = Arc::new(SequenceTracker::new());

        let mut config = GossipConfig::default();
        config.require_signatures = true;

        let manager = GossipMembershipManager::with_signing(
            node_id.clone(),
            config,
            transport,
            Arc::clone(&identity),
            Arc::clone(&registry),
            Arc::clone(&tracker),
        );

        assert!(manager.require_signatures());

        let msg = GossipMessage::Alive {
            node_id: node_id.clone(),
            incarnation: 1,
        };
        let signed = SignedGossipMessage::new(&identity, &msg, 1).unwrap();
        manager.handle_signed_gossip(&signed).unwrap();

        let wrapped = manager
            .create_gossip_message(GossipMessage::PingReq {
                origin: node_id.clone(),
                target: "peer".to_string(),
                sequence: 42,
            })
            .expect("signing should succeed");
        assert!(matches!(wrapped, Message::SignedGossip(_)));
    }

    #[test]
    fn test_handle_signed_gossip_without_signing_configured() {
        use crate::network::MemoryTransport;

        let identity = Identity::generate();
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        let msg = GossipMessage::Alive {
            node_id: "node1".to_string(),
            incarnation: 1,
        };
        let signed = SignedGossipMessage::new(&identity, &msg, 1).unwrap();
        let result = manager.handle_signed_gossip(&signed);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_gossip_run_shutdown() {
        use crate::network::MemoryTransport;

        let mut config = GossipConfig::default();
        config.gossip_interval_ms = 10;
        let manager = Arc::new(GossipMembershipManager::new(
            "node1".to_string(),
            config,
            Arc::new(MemoryTransport::new("node1".to_string())),
        ));

        let runner = Arc::clone(&manager);
        let handle = tokio::spawn(async move { runner.run().await });

        tokio::time::sleep(Duration::from_millis(30)).await;
        manager.shutdown();

        let result = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("gossip shutdown timed out");
        assert!(result.unwrap().is_ok());
    }

    #[test]
    fn test_gossip_heal_progress_tracking() {
        use crate::network::MemoryTransport;

        let config = GossipConfig {
            require_bidirectional: false,
            ..GossipConfig::default()
        };
        let manager = GossipMembershipManager::new(
            "node1".to_string(),
            config,
            Arc::new(MemoryTransport::new("node1".to_string())),
        );

        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Failed, 0);
        }

        manager.record_heal_progress(&"node2".to_string(), None);
        manager.record_heal_progress(&"node2".to_string(), None);

        let confirmed = manager.is_heal_confirmed(&"node2".to_string(), 2);
        assert!(confirmed.is_some());

        manager.reset_heal_progress(&"node2".to_string());
        assert!(manager.is_heal_confirmed(&"node2".to_string(), 1).is_none());

        let healing = manager.healing_nodes();
        assert!(healing.iter().any(|(id, _)| id == "node2"));

        manager.clear_heal_progress(&"node2".to_string());
        assert!(manager.is_heal_confirmed(&"node2".to_string(), 1).is_none());

        manager.record_heal_progress(&"node2".to_string(), Some(Instant::now()));
        manager.clear_heal_progress_batch(&["node2".to_string()]);
        assert!(manager.healing_nodes().is_empty());
    }

    #[test]
    fn test_select_gossip_targets_geometric() {
        use crate::network::MemoryTransport;
        use crate::{ClusterConfig, LocalNodeConfig, MembershipManager};
        use std::net::{Ipv4Addr, SocketAddr};
        use tensor_store::SparseVector;

        let local = LocalNodeConfig {
            node_id: "node1".to_string(),
            bind_address: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
        };
        let cluster_config = ClusterConfig::new("cluster", local)
            .with_peer("node2", SocketAddr::from((Ipv4Addr::LOCALHOST, 1)));
        let transport: Arc<dyn Transport> = Arc::new(MemoryTransport::new("node1".to_string()));
        let membership = Arc::new(MembershipManager::new(
            cluster_config,
            Arc::clone(&transport),
        ));
        membership.mark_healthy(&"node2".to_string());

        let geo = Arc::new(GeometricMembershipManager::with_defaults(Arc::clone(
            &membership,
        )));
        geo.update_local_embedding(SparseVector::from_dense(&[1.0, 0.0]));
        geo.record_peer_embedding(&"node2".to_string(), SparseVector::from_dense(&[1.0, 0.0]));

        let mut config = GossipConfig::default();
        config.geometric_routing = true;
        let manager =
            GossipMembershipManager::with_geometric("node1".to_string(), config, transport, geo);

        let targets = manager.select_gossip_targets(1);
        assert_eq!(targets, vec!["node2".to_string()]);
    }

    #[tokio::test]
    async fn test_gossip_round_sends_sync() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let peer_transport = MemoryTransport::new("node2".to_string());
        transport.connect_to("node2".to_string(), peer_transport.sender());

        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);
        manager.add_peer("node2".to_string());

        manager.gossip_round().await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    #[tokio::test]
    async fn test_handle_ping_req_and_ack_flow() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let origin_transport = MemoryTransport::new("origin".to_string());
        let target_transport = MemoryTransport::new("target".to_string());

        transport.connect_to("origin".to_string(), origin_transport.sender());
        transport.connect_to("target".to_string(), target_transport.sender());

        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);
        manager.add_peer("origin".to_string());
        manager.add_peer("target".to_string());

        manager.handle_gossip(GossipMessage::PingReq {
            origin: "origin".to_string(),
            target: "target".to_string(),
            sequence: 7,
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        manager.handle_gossip(GossipMessage::PingAck {
            origin: "origin".to_string(),
            target: "target".to_string(),
            sequence: 7,
            success: true,
        });
    }

    #[test]
    fn test_expire_suspicions_marks_failed() {
        use crate::network::MemoryTransport;

        let mut config = GossipConfig::default();
        config.suspicion_timeout_ms = 0;

        let manager = GossipMembershipManager::new(
            "node1".to_string(),
            config,
            Arc::new(MemoryTransport::new("node1".to_string())),
        );
        manager.add_peer("node2".to_string());

        manager.suspicions.write().insert(
            "node2".to_string(),
            PendingSuspicion {
                started_at: Instant::now(),
                incarnation: 0,
            },
        );

        manager.expire_suspicions();
        assert!(matches!(
            manager.node_state(&"node2".to_string()).map(|s| s.health),
            Some(NodeHealth::Failed | NodeHealth::Degraded)
        ));
    }

    #[test]
    fn test_check_sequence_exhaustion() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Initially not exhausted
        assert!(!manager.check_sequence_exhaustion());
        assert_eq!(manager.sequence_exhaustion_warning_count(), 0);

        // Set ping_sequence just below the threshold
        let threshold = u64::MAX / 10 * 9;
        manager.ping_sequence.store(threshold, Ordering::Relaxed);
        assert!(!manager.check_sequence_exhaustion());

        // Set ping_sequence above the threshold
        manager
            .ping_sequence
            .store(threshold + 1, Ordering::Relaxed);
        assert!(manager.check_sequence_exhaustion());
    }

    // ========== Bidirectional Connectivity Tests ==========

    #[test]
    fn test_bidirectional_probe_and_ack_confirms_connectivity() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Initially no connectivity data
        assert!(!manager.is_bidirectional_confirmed(&"node2".to_string()));
        assert!(manager.connectivity_status(&"node2".to_string()).is_none());

        // Simulate sending a probe (insert pending probe manually)
        let probe_id = manager.probe_id_counter.fetch_add(1, Ordering::Relaxed);
        manager
            .pending_probes
            .write()
            .insert(probe_id, ("node2".to_string(), Instant::now()));
        manager.connectivity_matrix.write().insert(
            "node2".to_string(),
            ConnectivityEntry {
                outbound_ok: true,
                inbound_ok: false,
                last_check: Instant::now(),
            },
        );

        // Not yet confirmed (no ack)
        assert!(!manager.is_bidirectional_confirmed(&"node2".to_string()));

        // Simulate receiving the ack
        manager.handle_bidirectional_ack(&"node2".to_string(), probe_id, &"node2".to_string());

        // Now confirmed
        assert!(manager.is_bidirectional_confirmed(&"node2".to_string()));
        let status = manager.connectivity_status(&"node2".to_string()).unwrap();
        assert!(status.outbound_ok);
        assert!(status.inbound_ok);
    }

    #[test]
    fn test_bidirectional_probe_timeout_detects_asymmetric() {
        use crate::network::MemoryTransport;

        let config = GossipConfig {
            bidirectional_probe_timeout_ms: 50,
            ..GossipConfig::default()
        };
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

        // Insert a probe that will time out
        let sent_at = Instant::now() - Duration::from_millis(100);
        manager
            .pending_probes
            .write()
            .insert(42, ("node2".to_string(), sent_at));

        assert_eq!(manager.asymmetric_partition_count(), 0);

        // Expire probes
        manager.expire_bidirectional_probes();

        assert_eq!(manager.asymmetric_partition_count(), 1);
        assert!(!manager.is_bidirectional_confirmed(&"node2".to_string()));

        // Verify pending probe was removed
        assert!(manager.pending_probes.read().is_empty());

        // Verify connectivity matrix shows asymmetric
        let status = manager.connectivity_status(&"node2".to_string()).unwrap();
        assert!(status.outbound_ok);
        assert!(!status.inbound_ok);
    }

    #[test]
    fn test_heal_blocked_without_bidirectional_confirmation() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = GossipConfig {
            require_bidirectional: true,
            ..GossipConfig::default()
        };
        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

        // Set up a node as failed
        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Failed, 0);
        }

        // Record enough heal progress
        manager.record_heal_progress(&"node2".to_string(), Some(Instant::now()));
        manager.record_heal_progress(&"node2".to_string(), None);
        manager.record_heal_progress(&"node2".to_string(), None);

        // Heal NOT confirmed because bidirectional not verified
        assert!(manager.is_heal_confirmed(&"node2".to_string(), 3).is_none());

        // Now confirm bidirectional connectivity
        manager.connectivity_matrix.write().insert(
            "node2".to_string(),
            ConnectivityEntry {
                outbound_ok: true,
                inbound_ok: true,
                last_check: Instant::now(),
            },
        );

        // Now heal IS confirmed
        assert!(manager.is_heal_confirmed(&"node2".to_string(), 3).is_some());
    }

    #[test]
    fn test_bidirectional_disabled_allows_heal_without_probe() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let config = GossipConfig {
            require_bidirectional: false,
            ..GossipConfig::default()
        };
        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

        {
            let mut state = manager.state.write();
            state.update_local("node2".to_string(), NodeHealth::Failed, 0);
        }

        manager.record_heal_progress(&"node2".to_string(), Some(Instant::now()));
        manager.record_heal_progress(&"node2".to_string(), None);

        // Should confirm without bidirectional check
        assert!(manager.is_heal_confirmed(&"node2".to_string(), 2).is_some());
    }

    #[tokio::test]
    async fn test_handle_bidirectional_probe_creates_ack() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Handle a probe from node2 - this spawns an async ack send
        // Just verify it doesn't panic
        manager.handle_bidirectional_probe(&"node2".to_string(), 99, 100);
    }

    #[test]
    fn test_clear_connectivity() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        manager.connectivity_matrix.write().insert(
            "node2".to_string(),
            ConnectivityEntry {
                outbound_ok: true,
                inbound_ok: true,
                last_check: Instant::now(),
            },
        );

        assert!(manager.connectivity_status(&"node2".to_string()).is_some());

        manager.clear_connectivity(&"node2".to_string());
        assert!(manager.connectivity_status(&"node2".to_string()).is_none());
    }

    #[test]
    fn test_bidirectional_ack_wrong_responder_ignored() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Insert pending probe for node2
        manager
            .pending_probes
            .write()
            .insert(10, ("node2".to_string(), Instant::now()));

        // Ack from wrong responder (node3 instead of node2)
        manager.handle_bidirectional_ack(&"node3".to_string(), 10, &"node3".to_string());

        // Probe was consumed but node2 not confirmed
        assert!(manager.pending_probes.read().is_empty());
        assert!(!manager.is_bidirectional_confirmed(&"node2".to_string()));
    }

    #[test]
    fn test_bidirectional_ack_unknown_probe_ignored() {
        use crate::network::MemoryTransport;

        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager =
            GossipMembershipManager::new("node1".to_string(), GossipConfig::default(), transport);

        // Ack for non-existent probe ID
        manager.handle_bidirectional_ack(&"node2".to_string(), 999, &"node2".to_string());

        // Should be no-op
        assert!(!manager.is_bidirectional_confirmed(&"node2".to_string()));
    }

    #[test]
    fn test_gossip_message_bidirectional_variants_serialize() {
        let probe = GossipMessage::BidirectionalProbe {
            origin: "node1".to_string(),
            probe_id: 42,
            timestamp: 100,
        };
        let ack = GossipMessage::BidirectionalAck {
            origin: "node1".to_string(),
            probe_id: 42,
            responder: "node2".to_string(),
        };

        // Verify serialization roundtrip
        let probe_bytes = bitcode::serialize(&probe).unwrap();
        let probe_deser: GossipMessage = bitcode::deserialize(&probe_bytes).unwrap();
        assert_eq!(probe, probe_deser);

        let ack_bytes = bitcode::serialize(&ack).unwrap();
        let ack_deser: GossipMessage = bitcode::deserialize(&ack_bytes).unwrap();
        assert_eq!(ack, ack_deser);
    }

    #[test]
    fn test_expire_bidirectional_probes_keeps_fresh() {
        use crate::network::MemoryTransport;

        let config = GossipConfig {
            bidirectional_probe_timeout_ms: 5000,
            ..GossipConfig::default()
        };
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        let manager = GossipMembershipManager::new("node1".to_string(), config, transport);

        // Insert a fresh probe (not yet timed out)
        manager
            .pending_probes
            .write()
            .insert(1, ("node2".to_string(), Instant::now()));

        manager.expire_bidirectional_probes();

        // Probe should still be pending
        assert_eq!(manager.pending_probes.read().len(), 1);
        assert_eq!(manager.asymmetric_partition_count(), 0);
    }

    #[test]
    fn test_incarnation_inflation_bound_rejects_large_jump_in_sync() {
        use crate::network::MemoryTransport;
        let transport = Arc::new(MemoryTransport::new("local".to_string()));
        let config = GossipConfig {
            max_incarnation_delta: 10,
            require_bidirectional: false,
            ..GossipConfig::default()
        };
        let manager = GossipMembershipManager::new("local".to_string(), config, transport);

        // Register a peer at incarnation 5
        manager
            .state
            .write()
            .update_local("peer1".to_string(), NodeHealth::Healthy, 5);

        // Sync with incarnation jump of 100 (exceeds max_delta of 10)
        let inflated = vec![GossipNodeState::new(
            "peer1".to_string(),
            NodeHealth::Healthy,
            100,
            105, // delta = 100
        )];
        manager.handle_sync(&"peer1".to_string(), &inflated, 100);

        // State should not have been updated
        let state = manager.state.read();
        assert_eq!(state.get(&"peer1".to_string()).unwrap().incarnation, 5);
        drop(state);

        // Rejected count should be 1
        assert_eq!(manager.incarnation_rejected_count(), 1);
    }

    #[test]
    fn test_incarnation_inflation_bound_allows_small_jump_in_sync() {
        use crate::network::MemoryTransport;
        let transport = Arc::new(MemoryTransport::new("local".to_string()));
        let config = GossipConfig {
            max_incarnation_delta: 10,
            require_bidirectional: false,
            ..GossipConfig::default()
        };
        let manager = GossipMembershipManager::new("local".to_string(), config, transport);

        // Register a peer at incarnation 5
        manager
            .state
            .write()
            .update_local("peer1".to_string(), NodeHealth::Healthy, 5);

        // Sync with incarnation jump of 3 (within max_delta of 10)
        let valid = vec![GossipNodeState::new(
            "peer1".to_string(),
            NodeHealth::Healthy,
            100,
            8, // delta = 3
        )];
        manager.handle_sync(&"peer1".to_string(), &valid, 100);

        // State should have been updated
        let state = manager.state.read();
        assert_eq!(state.get(&"peer1".to_string()).unwrap().incarnation, 8);
        drop(state);

        assert_eq!(manager.incarnation_rejected_count(), 0);
    }

    #[test]
    fn test_incarnation_inflation_bound_rejects_alive_with_large_jump() {
        use crate::network::MemoryTransport;
        let transport = Arc::new(MemoryTransport::new("local".to_string()));
        let config = GossipConfig {
            max_incarnation_delta: 5,
            require_bidirectional: false,
            ..GossipConfig::default()
        };
        let manager = GossipMembershipManager::new("local".to_string(), config, transport);

        // Register a peer at incarnation 2
        manager
            .state
            .write()
            .update_local("peer1".to_string(), NodeHealth::Degraded, 2);

        // Alive message with incarnation 100 (delta = 98, exceeds max of 5)
        manager.handle_alive(&"peer1".to_string(), 100);

        // Should still be at incarnation 2 (Alive rejected)
        let state = manager.state.read();
        assert_eq!(state.get(&"peer1".to_string()).unwrap().incarnation, 2);
        drop(state);

        assert_eq!(manager.incarnation_rejected_count(), 1);
    }

    #[test]
    fn test_incarnation_inflation_bound_allows_alive_within_delta() {
        use crate::network::MemoryTransport;
        let transport = Arc::new(MemoryTransport::new("local".to_string()));
        let config = GossipConfig {
            max_incarnation_delta: 5,
            require_bidirectional: false,
            ..GossipConfig::default()
        };
        let manager = GossipMembershipManager::new("local".to_string(), config, transport);

        // Register a peer at incarnation 2, suspected
        manager
            .state
            .write()
            .update_local("peer1".to_string(), NodeHealth::Degraded, 2);

        // Alive message with incarnation 4 (delta = 2, within max of 5)
        manager.handle_alive(&"peer1".to_string(), 4);

        // Should be updated to incarnation 4
        let state = manager.state.read();
        assert_eq!(state.get(&"peer1".to_string()).unwrap().incarnation, 4);
        drop(state);

        assert_eq!(manager.incarnation_rejected_count(), 0);
    }

    #[test]
    fn test_incarnation_inflation_default_config() {
        let config = GossipConfig::default();
        assert_eq!(config.max_incarnation_delta, 100);
    }

    #[test]
    fn test_signature_verification_failure_counter_starts_at_zero() {
        use crate::network::MemoryTransport;
        let transport = Arc::new(MemoryTransport::new("local".to_string()));
        let manager =
            GossipMembershipManager::new("local".to_string(), GossipConfig::default(), transport);
        assert_eq!(manager.signature_verification_failure_count(), 0);
    }

    #[test]
    fn test_handle_signed_gossip_without_signing_increments_failure() {
        use crate::network::MemoryTransport;
        let transport = Arc::new(MemoryTransport::new("local".to_string()));
        // Manager without signing configured
        let manager =
            GossipMembershipManager::new("local".to_string(), GossipConfig::default(), transport);

        // Create a dummy signed message (it won't matter - verification not configured)
        let dummy_signed = SignedGossipMessage {
            envelope: crate::signing::SignedMessage {
                sender: "attacker".to_string(),
                public_key: [0u8; 32],
                payload: vec![],
                signature: vec![],
                sequence: 0,
                timestamp_ms: 0,
            },
        };

        let result = manager.handle_signed_gossip(&dummy_signed);
        assert!(result.is_err());
        assert_eq!(manager.signature_verification_failure_count(), 1);
    }

    #[test]
    fn test_create_gossip_message_unsigned_returns_some() {
        use crate::network::MemoryTransport;
        let transport = Arc::new(MemoryTransport::new("local".to_string()));
        let manager =
            GossipMembershipManager::new("local".to_string(), GossipConfig::default(), transport);

        let msg = manager.create_gossip_message(GossipMessage::Alive {
            node_id: "local".to_string(),
            incarnation: 1,
        });
        assert!(msg.is_some());
        assert_eq!(manager.signature_verification_failure_count(), 0);
    }

    #[test]
    fn test_flap_record_tracks_transitions() {
        let mut record = FlapRecord::new();
        assert_eq!(record.flap_count, 0);
        assert!(record.backoff_duration(5).is_none());

        // Record 4 transitions - still below threshold
        for _ in 0..4 {
            record.record_transition();
        }
        assert_eq!(record.flap_count, 4);
        assert!(record.backoff_duration(5).is_none());

        // 5th transition triggers backoff
        record.record_transition();
        assert_eq!(record.flap_count, 5);
        let backoff = record.backoff_duration(5);
        assert!(backoff.is_some());
        assert_eq!(backoff.unwrap(), Duration::from_secs(1));

        // 6th transition increases backoff
        record.record_transition();
        let backoff = record.backoff_duration(5);
        assert_eq!(backoff.unwrap(), Duration::from_secs(2));
    }

    #[test]
    fn test_flap_record_backoff_exponential() {
        let mut record = FlapRecord::new();
        // Get to threshold + 3 (backoff = 2^3 = 8 seconds)
        for _ in 0..8 {
            record.record_transition();
        }
        let backoff = record.backoff_duration(5).unwrap();
        assert_eq!(backoff, Duration::from_secs(8));
    }

    #[test]
    fn test_flap_record_backoff_capped() {
        let mut record = FlapRecord::new();
        // Push past the cap
        for _ in 0..20 {
            record.record_transition();
        }
        let backoff = record.backoff_duration(5).unwrap();
        // Capped at 300 seconds (5 minutes)
        assert!(backoff.as_secs() <= 300);
    }

    #[test]
    fn test_flap_detection_via_manager() {
        use crate::network::MemoryTransport;
        let transport = Arc::new(MemoryTransport::new("local".to_string()));
        let config = GossipConfig {
            require_bidirectional: false,
            ..GossipConfig::default()
        };
        let manager = GossipMembershipManager::new("local".to_string(), config, transport);

        assert_eq!(manager.flap_count(&"peer1".to_string()), 0);
        assert!(!manager.is_in_flap_backoff(&"peer1".to_string()));
        assert_eq!(manager.flap_backoffs_count(), 0);

        // Simulate alive messages that refute suspicion (causes flap recording)
        // Register a peer as suspected
        manager
            .state
            .write()
            .update_local("peer1".to_string(), NodeHealth::Degraded, 1);

        // Simulate 6 alive/refute cycles
        for i in 2..8u64 {
            manager.handle_alive(&"peer1".to_string(), i);
            // Re-suspect to allow next refute
            manager
                .state
                .write()
                .update_local("peer1".to_string(), NodeHealth::Degraded, i);
        }

        assert!(manager.flap_count(&"peer1".to_string()) >= 5);
        assert!(manager.is_in_flap_backoff(&"peer1".to_string()));
        assert!(manager.flap_backoffs_count() > 0);
    }
}
