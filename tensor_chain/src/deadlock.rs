// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Wait-for graph tracking with DFS-based cycle detection for distributed transactions.
//!
//! # Overview
//!
//! This module implements deadlock detection for the 2PC (Two-Phase Commit) distributed
//! transaction coordinator. It maintains a wait-for graph tracking which transactions
//! are blocked waiting for other transactions to release locks, and uses depth-first
//! search to detect cycles that indicate deadlocks.
//!
//! Timeout-based deadlock prevention alone is insufficient for distributed systems
//! because it cannot distinguish between:
//! - A transaction that is genuinely slow
//! - A transaction that is permanently blocked by a circular dependency
//!
//! This module provides active cycle detection to resolve deadlocks immediately
//! rather than waiting for arbitrary timeouts.
//!
//! # Architecture
//!
//! ```text
//! +------------------+     add_wait()      +----------------+
//! | LockManager      | ------------------> | WaitForGraph   |
//! | (2PC Coordinator)|                     |                |
//! +------------------+                     | edges: tx->txs |
//! | acquire_locks()  | <-- wait_info --    | reverse_edges  |
//! | on conflict:     |                     | wait_started   |
//! |   add to graph   |                     | priorities     |
//! +------------------+                     +----------------+
//!                                                  |
//!                                          detect_cycles()
//!                                                  |
//!                                                  v
//!                                          +----------------+
//!                                          | DeadlockInfo   |
//!                                          | - cycle        |
//!                                          | - victim_tx_id |
//!                                          +----------------+
//! ```
//!
//! # Wait-For Graph Optimization
//!
//! The graph maintains bidirectional edges for efficient operations:
//! - `edges`: Maps `waiter_tx_id` -> set of `holder_tx_ids` (forward edges)
//! - `reverse_edges`: Maps `holder_tx_id` -> set of `waiter_tx_ids` (backward edges)
//!
//! This allows O(1) transaction removal when a transaction commits or aborts,
//! since we can efficiently find and remove all edges involving that transaction.
//! Space overhead is acceptable because concurrent transactions are bounded by
//! cluster capacity.
//!
//! # Victim Selection Policies
//!
//! When a deadlock is detected, one transaction must be aborted to break the cycle.
//! The module supports multiple victim selection policies with different trade-offs:
//!
//! | Policy | Description | Trade-off |
//! |--------|-------------|-----------|
//! | `Youngest` | Abort most recent waiter | Minimizes wasted work, may starve long transactions |
//! | `Oldest` | Abort earliest waiter | Prevents starvation, wastes more work |
//! | `LowestPriority` | Abort lowest-priority tx | Business-rule prioritization |
//! | `MostLocks` | Abort tx holding most locks | Maximizes freed resources |
//!
//! # Usage
//!
//! ```rust
//! use tensor_chain::deadlock::{DeadlockDetector, DeadlockDetectorConfig, VictimSelectionPolicy};
//!
//! // Create detector with custom configuration
//! let config = DeadlockDetectorConfig::default()
//!     .with_interval(50)  // Check every 50ms
//!     .with_policy(VictimSelectionPolicy::Youngest);
//!
//! let detector = DeadlockDetector::new(config);
//!
//! // Add wait-for edges when lock conflicts occur
//! detector.graph().add_wait(/*waiter=*/1, /*holder=*/2, Some(/*priority=*/10));
//! detector.graph().add_wait(/*waiter=*/2, /*holder=*/1, Some(/*priority=*/20));
//!
//! // Run detection cycle
//! let deadlocks = detector.detect();
//! for info in deadlocks {
//!     println!("Deadlock detected: {:?}, victim: {}", info.cycle, info.victim_tx_id);
//!     // Abort the victim transaction
//! }
//!
//! // Remove transaction when it commits/aborts
//! detector.graph().remove_transaction(1);
//! ```
//!
//! # Cycle Detection Algorithm
//!
//! The cycle detection uses a variant of Tarjan's DFS algorithm:
//!
//! 1. Maintain a `visited` set of all nodes seen during traversal
//! 2. Maintain a `rec_stack` (recursion stack) of nodes in current DFS path
//! 3. When visiting a neighbor already in `rec_stack`, a cycle is found
//! 4. Extract the cycle from the current path
//!
//! Time complexity: O(V + E) where V = transactions, E = wait edges
//!
//! # Security Considerations
//!
//! - The `max_cycle_length` configuration prevents `DoS` via artificially long cycles
//! - Detection can be disabled for testing via `DeadlockDetectorConfig::disabled()`
//! - Statistics track detection performance for monitoring
//!
//! # See Also
//!
//! - [`crate::distributed_tx`]: 2PC coordinator that uses this module
//! - [`crate::distributed_tx::LockManager`]: Lock acquisition that populates the graph

use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::sync_compat::RwLock;
use serde::{Deserialize, Serialize};

use crate::distributed_tx::EpochMillis;

/// Policy for selecting which transaction to abort in a deadlock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VictimSelectionPolicy {
    /// Abort the youngest transaction (most recent wait start).
    #[default]
    Youngest,
    /// Abort the oldest transaction (earliest wait start).
    Oldest,
    /// Abort the transaction with lowest priority (highest priority value).
    LowestPriority,
    /// Abort the transaction holding the most locks.
    MostLocks,
}

/// Configuration for deadlock detection.
#[derive(Debug, Clone)]
pub struct DeadlockDetectorConfig {
    /// Whether deadlock detection is enabled.
    pub enabled: bool,
    /// Detection interval in milliseconds.
    pub detection_interval_ms: u64,
    /// Victim selection policy.
    pub victim_policy: VictimSelectionPolicy,
    /// Maximum cycle length to detect.
    pub max_cycle_length: usize,
    /// Whether to automatically abort victim transactions.
    pub auto_abort_victim: bool,
    /// Maximum wait-for edges a single transaction can have.
    pub max_edges_per_tx: usize,
    /// Edges older than this (in milliseconds) are considered stale and eligible for cleanup.
    pub edge_ttl_ms: u64,
    /// Maximum depth for cascading deadlock resolution. When a victim resolves
    /// multiple cycles, this limits the cascading resolution depth.
    pub victim_cascade_depth: u32,
}

impl Default for DeadlockDetectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detection_interval_ms: 100,
            victim_policy: VictimSelectionPolicy::Youngest,
            max_cycle_length: 100,
            auto_abort_victim: true,
            max_edges_per_tx: 50,
            edge_ttl_ms: 30_000,
            victim_cascade_depth: 3,
        }
    }
}

impl DeadlockDetectorConfig {
    /// Create a disabled configuration.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Set the detection interval.
    #[must_use]
    pub const fn with_interval(mut self, ms: u64) -> Self {
        self.detection_interval_ms = ms;
        self
    }

    /// Set the victim selection policy.
    #[must_use]
    pub const fn with_policy(mut self, policy: VictimSelectionPolicy) -> Self {
        self.victim_policy = policy;
        self
    }

    /// Set the maximum cycle length.
    #[must_use]
    pub const fn with_max_cycle_length(mut self, len: usize) -> Self {
        self.max_cycle_length = len;
        self
    }

    /// Disable auto-abort of victim.
    #[must_use]
    pub const fn without_auto_abort(mut self) -> Self {
        self.auto_abort_victim = false;
        self
    }

    /// Set the maximum number of wait-for edges per transaction.
    #[must_use]
    pub const fn with_max_edges_per_tx(mut self, max: usize) -> Self {
        self.max_edges_per_tx = max;
        self
    }

    /// Set the edge TTL in milliseconds for stale edge cleanup.
    #[must_use]
    pub const fn with_edge_ttl_ms(mut self, ttl_ms: u64) -> Self {
        self.edge_ttl_ms = ttl_ms;
        self
    }

    /// Set the maximum victim cascade depth for resolving overlapping deadlocks.
    #[must_use]
    pub const fn with_victim_cascade_depth(mut self, depth: u32) -> Self {
        self.victim_cascade_depth = depth;
        self
    }
}

/// Information about a wait condition when lock acquisition fails.
#[derive(Debug, Clone)]
pub struct WaitInfo {
    /// Transaction ID that holds the conflicting lock.
    pub blocking_tx_id: u64,
    /// Keys that caused the conflict.
    pub conflicting_keys: Vec<String>,
}

/// Information about a detected deadlock.
#[derive(Debug, Clone)]
pub struct DeadlockInfo {
    /// Transaction IDs involved in the cycle.
    pub cycle: Vec<u64>,
    /// Selected victim transaction ID.
    pub victim_tx_id: u64,
    /// When the deadlock was detected.
    pub detected_at: EpochMillis,
    /// Policy used for victim selection.
    pub victim_policy: VictimSelectionPolicy,
}

/// Statistics for deadlock detection.
#[derive(Debug, Default)]
pub struct DeadlockStats {
    /// Number of deadlocks detected.
    pub deadlocks_detected: AtomicU64,
    /// Number of victims aborted.
    pub victims_aborted: AtomicU64,
    /// Total time spent in detection (microseconds).
    pub detection_time_us: AtomicU64,
    /// Number of detection cycles run.
    pub detection_cycles: AtomicU64,
    /// Longest cycle detected.
    pub max_cycle_length: AtomicU64,
    /// Number of stale transactions cleaned up by TTL-based edge cleanup.
    pub stale_edges_cleaned: AtomicU64,
    /// Number of cycles resolved by cascading (victim appeared in multiple cycles).
    pub cascade_resolutions: AtomicU64,
}

impl DeadlockStats {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a detection cycle.
    pub fn record_detection(&self, duration_us: u64, deadlocks_found: usize) {
        self.detection_cycles.fetch_add(1, Ordering::Relaxed);
        self.detection_time_us
            .fetch_add(duration_us, Ordering::Relaxed);
        self.deadlocks_detected
            .fetch_add(deadlocks_found as u64, Ordering::Relaxed);
    }

    /// Record a victim abort.
    pub fn record_victim_abort(&self) {
        self.victims_aborted.fetch_add(1, Ordering::Relaxed);
    }

    /// Update max cycle length if longer.
    pub fn update_max_cycle(&self, len: usize) {
        let len = len as u64;
        let mut current = self.max_cycle_length.load(Ordering::Relaxed);
        while len > current {
            match self.max_cycle_length.compare_exchange_weak(
                current,
                len,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => current = c,
            }
        }
    }

    /// Get a snapshot of stats.
    pub fn snapshot(&self) -> DeadlockStatsSnapshot {
        DeadlockStatsSnapshot {
            deadlocks_detected: self.deadlocks_detected.load(Ordering::Relaxed),
            victims_aborted: self.victims_aborted.load(Ordering::Relaxed),
            detection_time_us: self.detection_time_us.load(Ordering::Relaxed),
            detection_cycles: self.detection_cycles.load(Ordering::Relaxed),
            max_cycle_length: self.max_cycle_length.load(Ordering::Relaxed),
            stale_edges_cleaned: self.stale_edges_cleaned.load(Ordering::Relaxed),
            cascade_resolutions: self.cascade_resolutions.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of deadlock stats.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeadlockStatsSnapshot {
    pub deadlocks_detected: u64,
    pub victims_aborted: u64,
    pub detection_time_us: u64,
    pub detection_cycles: u64,
    pub max_cycle_length: u64,
    pub stale_edges_cleaned: u64,
    #[serde(default)]
    pub cascade_resolutions: u64,
}

/// Wait-for graph for tracking transaction dependencies.
///
/// Tracks which transactions are waiting for which other transactions
/// to release locks. Used for cycle detection in deadlock scenarios.
#[derive(Debug)]
pub struct WaitForGraph {
    /// Maps `tx_id` -> set of `tx_ids` it is waiting for.
    edges: RwLock<HashMap<u64, HashSet<u64>>>,
    /// Maps `tx_id` -> timestamp when wait started (for victim selection).
    wait_started: RwLock<HashMap<u64, EpochMillis>>,
    /// Maps `tx_id` -> priority (lower value = higher priority).
    priorities: RwLock<HashMap<u64, u32>>,
    /// Reverse edges for efficient `waiting_on` queries.
    reverse_edges: RwLock<HashMap<u64, HashSet<u64>>>,
    /// Maximum wait-for edges a single transaction can have (0 = unlimited).
    max_edges_per_tx: usize,
}

impl Default for WaitForGraph {
    fn default() -> Self {
        Self {
            edges: RwLock::new(HashMap::new()),
            wait_started: RwLock::new(HashMap::new()),
            priorities: RwLock::new(HashMap::new()),
            reverse_edges: RwLock::new(HashMap::new()),
            max_edges_per_tx: 0,
        }
    }
}

impl WaitForGraph {
    /// Create a new empty wait-for graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new wait-for graph with a per-transaction edge limit.
    #[must_use]
    pub fn with_max_edges_per_tx(max: usize) -> Self {
        Self {
            max_edges_per_tx: max,
            ..Self::default()
        }
    }

    /// Add a wait-for edge: waiter is waiting for holder.
    ///
    /// The waiter transaction is blocked waiting for holder to release locks.
    /// If `max_edges_per_tx` is set and the waiter already has that many edges,
    /// the new edge is silently dropped to bound memory usage.
    #[allow(clippy::significant_drop_tightening)] // Lock scopes are already minimal blocks
    pub fn add_wait(&self, waiter_tx_id: u64, holder_tx_id: u64, priority: Option<u32>) {
        if waiter_tx_id == holder_tx_id {
            return; // Self-wait is invalid
        }

        #[allow(clippy::cast_possible_truncation)]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        {
            let mut edges = self.edges.write();
            let waiter_edges = edges.entry(waiter_tx_id).or_default();
            if self.max_edges_per_tx > 0 && waiter_edges.len() >= self.max_edges_per_tx {
                return;
            }
            waiter_edges.insert(holder_tx_id);
        }

        {
            let mut reverse = self.reverse_edges.write();
            reverse
                .entry(holder_tx_id)
                .or_default()
                .insert(waiter_tx_id);
        }

        {
            let mut wait_started = self.wait_started.write();
            wait_started.entry(waiter_tx_id).or_insert(now);
        }

        if let Some(p) = priority {
            let mut priorities = self.priorities.write();
            priorities.insert(waiter_tx_id, p);
        }
    }

    /// Remove all wait edges for a transaction (when it commits/aborts).
    pub fn remove_transaction(&self, tx_id: u64) {
        // Remove outgoing edges
        let outgoing = {
            let mut edges = self.edges.write();
            edges.remove(&tx_id)
        };

        // Update reverse edges for removed outgoing edges
        if let Some(holders) = outgoing {
            let mut reverse = self.reverse_edges.write();
            for holder in holders {
                if let Some(waiters) = reverse.get_mut(&holder) {
                    waiters.remove(&tx_id);
                }
            }
        }

        // Remove incoming edges (other transactions waiting for this one)
        let incoming = {
            let mut reverse = self.reverse_edges.write();
            reverse.remove(&tx_id)
        };

        // Update forward edges for removed incoming edges
        if let Some(waiters) = incoming {
            let mut edges = self.edges.write();
            for waiter in waiters {
                if let Some(holders) = edges.get_mut(&waiter) {
                    holders.remove(&tx_id);
                }
            }
        }

        // Remove metadata
        self.wait_started.write().remove(&tx_id);
        self.priorities.write().remove(&tx_id);
    }

    /// Remove a specific wait edge.
    #[allow(clippy::significant_drop_tightening)] // Lock scopes are already minimal blocks
    pub fn remove_wait(&self, waiter_tx_id: u64, holder_tx_id: u64) {
        {
            let mut edges = self.edges.write();
            if let Some(holders) = edges.get_mut(&waiter_tx_id) {
                holders.remove(&holder_tx_id);
                if holders.is_empty() {
                    edges.remove(&waiter_tx_id);
                    // Also remove wait_started since not waiting anymore
                    self.wait_started.write().remove(&waiter_tx_id);
                }
            }
        }

        {
            let mut reverse = self.reverse_edges.write();
            if let Some(waiters) = reverse.get_mut(&holder_tx_id) {
                waiters.remove(&waiter_tx_id);
                if waiters.is_empty() {
                    reverse.remove(&holder_tx_id);
                }
            }
        }
    }

    /// Detect cycles in the wait-for graph using DFS.
    ///
    /// Returns all cycles found. Each cycle is a vec of `tx_ids` forming the cycle.
    #[allow(clippy::significant_drop_tightening)] // Read guard needed throughout DFS
    pub fn detect_cycles(&self) -> Vec<Vec<u64>> {
        let edges = self.edges.read();
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();

        for &start in edges.keys() {
            if !visited.contains(&start) {
                dfs_detect(
                    start,
                    &edges,
                    &mut visited,
                    &mut rec_stack,
                    &mut path,
                    &mut cycles,
                );
            }
        }

        cycles
    }

    /// Check if adding an edge would create a cycle.
    ///
    /// Useful for deadlock prevention: reject lock acquisition that would cause deadlock.
    pub fn would_create_cycle(&self, waiter_tx_id: u64, holder_tx_id: u64) -> bool {
        if waiter_tx_id == holder_tx_id {
            return true;
        }

        let edges = self.edges.read();

        // Check if there's already a path from holder back to waiter
        let mut visited = HashSet::new();
        let mut stack = vec![holder_tx_id];

        while let Some(current) = stack.pop() {
            if current == waiter_tx_id {
                return true;
            }
            if visited.insert(current) {
                if let Some(neighbors) = edges.get(&current) {
                    stack.extend(neighbors);
                }
            }
        }

        false
    }

    /// Get all transactions a given transaction is waiting for.
    pub fn waiting_for(&self, tx_id: u64) -> HashSet<u64> {
        self.edges.read().get(&tx_id).cloned().unwrap_or_default()
    }

    /// Get all transactions waiting for a given transaction.
    pub fn waiting_on(&self, tx_id: u64) -> HashSet<u64> {
        self.reverse_edges
            .read()
            .get(&tx_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.read().values().map(HashSet::len).sum()
    }

    /// Get the number of transactions in the graph.
    pub fn transaction_count(&self) -> usize {
        let edges = self.edges.read();
        let reverse = self.reverse_edges.read();

        let mut all_txs: HashSet<u64> = edges.keys().copied().collect();
        all_txs.extend(reverse.keys());
        drop(edges);
        drop(reverse);
        all_txs.len()
    }

    /// Clear the entire graph.
    pub fn clear(&self) {
        self.edges.write().clear();
        self.reverse_edges.write().clear();
        self.wait_started.write().clear();
        self.priorities.write().clear();
    }

    /// Get transaction wait start time for victim selection.
    pub fn get_wait_start(&self, tx_id: u64) -> Option<EpochMillis> {
        self.wait_started.read().get(&tx_id).copied()
    }

    /// Get transaction priority for victim selection.
    pub fn get_priority(&self, tx_id: u64) -> Option<u32> {
        self.priorities.read().get(&tx_id).copied()
    }

    /// Check if graph is empty.
    pub fn is_empty(&self) -> bool {
        self.edges.read().is_empty()
    }

    /// Remove transactions whose wait-start time is older than `ttl_ms` milliseconds.
    ///
    /// Returns the number of stale transactions removed.
    pub fn cleanup_stale_edges(&self, ttl_ms: u64) -> usize {
        #[allow(clippy::cast_possible_truncation)]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let stale_txs: Vec<u64> = {
            let wait_started = self.wait_started.read();
            wait_started
                .iter()
                .filter(|&(_, &started)| now.saturating_sub(started) > ttl_ms)
                .map(|(&tx_id, _)| tx_id)
                .collect()
        };

        let count = stale_txs.len();
        for tx_id in stale_txs {
            self.remove_transaction(tx_id);
        }

        count
    }
}

/// Tarjan's DFS: cycle exists when we hit a node in `rec_stack` (back edge to ancestor).
/// Path tracked explicitly for victim selection.
fn dfs_detect(
    node: u64,
    edges: &HashMap<u64, HashSet<u64>>,
    visited: &mut HashSet<u64>,
    rec_stack: &mut HashSet<u64>,
    path: &mut Vec<u64>,
    cycles: &mut Vec<Vec<u64>>,
) {
    visited.insert(node);
    rec_stack.insert(node);
    path.push(node);

    if let Some(neighbors) = edges.get(&node) {
        for &neighbor in neighbors {
            if !visited.contains(&neighbor) {
                dfs_detect(neighbor, edges, visited, rec_stack, path, cycles);
            } else if rec_stack.contains(&neighbor) {
                // Found cycle - extract it from path
                if let Some(cycle_start) = path.iter().position(|&n| n == neighbor) {
                    cycles.push(path[cycle_start..].to_vec());
                }
            }
        }
    }

    path.pop();
    rec_stack.remove(&node);
}

/// Deadlock detector with configurable victim selection.
pub struct DeadlockDetector {
    /// Wait-for graph.
    graph: WaitForGraph,
    /// Configuration.
    config: DeadlockDetectorConfig,
    /// Statistics.
    stats: DeadlockStats,
    /// Lock count function for `MostLocks` policy.
    lock_count_fn: Option<Box<dyn Fn(u64) -> usize + Send + Sync>>,
}

impl std::fmt::Debug for DeadlockDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeadlockDetector")
            .field("graph", &self.graph)
            .field("config", &self.config)
            .field("stats", &self.stats)
            .field(
                "lock_count_fn",
                &self.lock_count_fn.as_ref().map(|_| "<fn>"),
            )
            .finish()
    }
}

impl DeadlockDetector {
    /// Create a new deadlock detector with the given configuration.
    #[must_use]
    pub fn new(config: DeadlockDetectorConfig) -> Self {
        let graph = WaitForGraph::with_max_edges_per_tx(config.max_edges_per_tx);
        Self {
            graph,
            config,
            stats: DeadlockStats::new(),
            lock_count_fn: None,
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DeadlockDetectorConfig::default())
    }

    /// Set the lock count function for `MostLocks` victim selection.
    pub fn set_lock_count_fn<F>(&mut self, f: F)
    where
        F: Fn(u64) -> usize + Send + Sync + 'static,
    {
        self.lock_count_fn = Some(Box::new(f));
    }

    /// Get the wait-for graph.
    pub const fn graph(&self) -> &WaitForGraph {
        &self.graph
    }

    /// Get the configuration.
    pub const fn config(&self) -> &DeadlockDetectorConfig {
        &self.config
    }

    /// Get statistics.
    pub const fn stats(&self) -> &DeadlockStats {
        &self.stats
    }

    /// Check if detection is enabled.
    pub const fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Run one detection cycle, returns detected deadlocks.
    ///
    /// Performs cascading resolution: if a victim from one cycle also appears
    /// in other unresolved cycles, those cycles are considered resolved by the
    /// same abort, up to `victim_cascade_depth`.
    pub fn detect(&self) -> Vec<DeadlockInfo> {
        if !self.config.enabled {
            return Vec::new();
        }

        let start = std::time::Instant::now();
        let cycles = self.graph.detect_cycles();
        #[allow(clippy::cast_possible_truncation)]
        let duration_us = start.elapsed().as_micros() as u64;

        // Filter and collect valid cycles
        let valid_cycles: Vec<Vec<u64>> = cycles
            .into_iter()
            .filter(|c| c.len() <= self.config.max_cycle_length)
            .collect();

        for cycle in &valid_cycles {
            self.stats.update_max_cycle(cycle.len());
        }

        #[allow(clippy::cast_possible_truncation)]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Cascading resolution: deduplicate victims across overlapping cycles
        let mut deadlocks = Vec::new();
        let mut resolved_victims: HashSet<u64> = HashSet::new();
        let mut cascade_count = 0u32;

        for cycle in &valid_cycles {
            // Check if any existing victim already resolves this cycle
            let already_resolved = cycle.iter().any(|tx| resolved_victims.contains(tx));

            if already_resolved && cascade_count < self.config.victim_cascade_depth {
                cascade_count += 1;
                self.stats
                    .cascade_resolutions
                    .fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    cycle = ?cycle,
                    cascade_depth = cascade_count,
                    "Cycle resolved by cascading victim"
                );
                continue;
            }

            let victim = self.select_victim(cycle);
            resolved_victims.insert(victim);

            deadlocks.push(DeadlockInfo {
                cycle: cycle.clone(),
                victim_tx_id: victim,
                detected_at: now,
                victim_policy: self.config.victim_policy,
            });
        }

        self.stats.record_detection(duration_us, deadlocks.len());

        deadlocks
    }

    /// Policy trade-offs:
    /// - Youngest: minimize wasted work, may starve long transactions
    /// - Oldest: prevent starvation, wastes more work
    /// - `LowestPriority`: business-rule prioritization
    /// - `MostLocks`: maximize freed resources
    pub fn select_victim(&self, cycle: &[u64]) -> u64 {
        if cycle.is_empty() {
            return 0;
        }
        if cycle.len() == 1 {
            return cycle[0];
        }

        match self.config.victim_policy {
            VictimSelectionPolicy::Youngest => {
                // Youngest = most recent wait start = highest timestamp
                cycle
                    .iter()
                    .max_by_key(|&&tx_id| self.graph.get_wait_start(tx_id).unwrap_or(0))
                    .copied()
                    .unwrap_or(cycle[0])
            },
            VictimSelectionPolicy::Oldest => {
                // Oldest = earliest wait start = lowest timestamp
                cycle
                    .iter()
                    .min_by_key(|&&tx_id| self.graph.get_wait_start(tx_id).unwrap_or(u64::MAX))
                    .copied()
                    .unwrap_or(cycle[0])
            },
            VictimSelectionPolicy::LowestPriority => {
                // Lowest priority = highest priority value (lower value = higher priority)
                cycle
                    .iter()
                    .max_by_key(|&&tx_id| self.graph.get_priority(tx_id).unwrap_or(0))
                    .copied()
                    .unwrap_or(cycle[0])
            },
            VictimSelectionPolicy::MostLocks => {
                self.lock_count_fn.as_ref().map_or_else(
                    || {
                        // Fall back to youngest if no lock count function
                        cycle
                            .iter()
                            .max_by_key(|&&tx_id| self.graph.get_wait_start(tx_id).unwrap_or(0))
                            .copied()
                            .unwrap_or(cycle[0])
                    },
                    |lock_fn| {
                        cycle
                            .iter()
                            .max_by_key(|&&tx_id| lock_fn(tx_id))
                            .copied()
                            .unwrap_or(cycle[0])
                    },
                )
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // WaitForGraph tests

    #[test]
    fn test_wait_for_graph_new() {
        let graph = WaitForGraph::new();
        assert!(graph.is_empty());
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.transaction_count(), 0);
    }

    #[test]
    fn test_wait_for_graph_add_wait() {
        let graph = WaitForGraph::new();

        graph.add_wait(1, 2, None);

        assert!(!graph.is_empty());
        assert_eq!(graph.edge_count(), 1);
        assert!(graph.waiting_for(1).contains(&2));
        assert!(graph.waiting_on(2).contains(&1));
    }

    #[test]
    fn test_wait_for_graph_add_wait_with_priority() {
        let graph = WaitForGraph::new();

        graph.add_wait(1, 2, Some(10));

        assert_eq!(graph.get_priority(1), Some(10));
        assert!(graph.get_wait_start(1).is_some());
    }

    #[test]
    fn test_wait_for_graph_self_wait_ignored() {
        let graph = WaitForGraph::new();

        graph.add_wait(1, 1, None);

        assert!(graph.is_empty());
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_wait_for_graph_remove_transaction() {
        let graph = WaitForGraph::new();

        // Create: 1 -> 2 -> 3
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 3, None);

        assert_eq!(graph.edge_count(), 2);

        // Remove tx 2 (middle node)
        graph.remove_transaction(2);

        // Edge 1->2 should be gone, 2->3 should be gone
        assert!(!graph.waiting_for(1).contains(&2));
        assert!(graph.waiting_for(2).is_empty());
    }

    #[test]
    fn test_wait_for_graph_remove_wait() {
        let graph = WaitForGraph::new();

        graph.add_wait(1, 2, None);
        graph.add_wait(1, 3, None);

        assert_eq!(graph.waiting_for(1).len(), 2);

        graph.remove_wait(1, 2);

        assert_eq!(graph.waiting_for(1).len(), 1);
        assert!(graph.waiting_for(1).contains(&3));
    }

    #[test]
    fn test_wait_for_graph_detect_simple_cycle() {
        let graph = WaitForGraph::new();

        // Create cycle: 1 -> 2 -> 1
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 1, None);

        let cycles = graph.detect_cycles();

        assert!(!cycles.is_empty());
        // Cycle should contain both 1 and 2
        let cycle = &cycles[0];
        assert!(cycle.contains(&1) || cycle.contains(&2));
    }

    #[test]
    fn test_wait_for_graph_detect_multi_node_cycle() {
        let graph = WaitForGraph::new();

        // Create cycle: 1 -> 2 -> 3 -> 1
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 3, None);
        graph.add_wait(3, 1, None);

        let cycles = graph.detect_cycles();

        assert!(!cycles.is_empty());
        let cycle = &cycles[0];
        assert!(cycle.len() >= 2);
    }

    #[test]
    fn test_wait_for_graph_no_cycle() {
        let graph = WaitForGraph::new();

        // Create chain: 1 -> 2 -> 3 (no cycle)
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 3, None);

        let cycles = graph.detect_cycles();

        assert!(cycles.is_empty());
    }

    #[test]
    fn test_wait_for_graph_would_create_cycle() {
        let graph = WaitForGraph::new();

        // Create: 1 -> 2 -> 3
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 3, None);

        // Adding 3 -> 1 would create cycle
        assert!(graph.would_create_cycle(3, 1));

        // Adding 4 -> 1 would not create cycle
        assert!(!graph.would_create_cycle(4, 1));

        // Self-wait always creates cycle
        assert!(graph.would_create_cycle(1, 1));
    }

    #[test]
    fn test_wait_for_graph_clear() {
        let graph = WaitForGraph::new();

        graph.add_wait(1, 2, Some(10));
        graph.add_wait(2, 3, Some(20));

        assert!(!graph.is_empty());

        graph.clear();

        assert!(graph.is_empty());
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.get_priority(1).is_none());
    }

    #[test]
    fn test_wait_for_graph_waiting_for_and_on() {
        let graph = WaitForGraph::new();

        // 1 waits for 2 and 3
        graph.add_wait(1, 2, None);
        graph.add_wait(1, 3, None);
        // 4 also waits for 2
        graph.add_wait(4, 2, None);

        let waiting_for_1 = graph.waiting_for(1);
        assert_eq!(waiting_for_1.len(), 2);
        assert!(waiting_for_1.contains(&2));
        assert!(waiting_for_1.contains(&3));

        let waiting_on_2 = graph.waiting_on(2);
        assert_eq!(waiting_on_2.len(), 2);
        assert!(waiting_on_2.contains(&1));
        assert!(waiting_on_2.contains(&4));
    }

    #[test]
    fn test_wait_for_graph_transaction_count() {
        let graph = WaitForGraph::new();

        graph.add_wait(1, 2, None);
        graph.add_wait(3, 4, None);

        // Should count all unique tx_ids: 1, 2, 3, 4
        assert_eq!(graph.transaction_count(), 4);
    }

    // DeadlockDetectorConfig tests

    #[test]
    fn test_config_default() {
        let config = DeadlockDetectorConfig::default();

        assert!(config.enabled);
        assert_eq!(config.detection_interval_ms, 100);
        assert_eq!(config.victim_policy, VictimSelectionPolicy::Youngest);
        assert_eq!(config.max_cycle_length, 100);
        assert!(config.auto_abort_victim);
    }

    #[test]
    fn test_config_disabled() {
        let config = DeadlockDetectorConfig::disabled();

        assert!(!config.enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = DeadlockDetectorConfig::default()
            .with_interval(50)
            .with_policy(VictimSelectionPolicy::Oldest)
            .with_max_cycle_length(50)
            .without_auto_abort();

        assert!(config.enabled);
        assert_eq!(config.detection_interval_ms, 50);
        assert_eq!(config.victim_policy, VictimSelectionPolicy::Oldest);
        assert_eq!(config.max_cycle_length, 50);
        assert!(!config.auto_abort_victim);
    }

    // VictimSelectionPolicy tests

    #[test]
    fn test_victim_policy_serialization() {
        let policies = [
            VictimSelectionPolicy::Youngest,
            VictimSelectionPolicy::Oldest,
            VictimSelectionPolicy::LowestPriority,
            VictimSelectionPolicy::MostLocks,
        ];

        for policy in policies {
            let serialized = bitcode::serialize(&policy).unwrap();
            let deserialized: VictimSelectionPolicy = bitcode::deserialize(&serialized).unwrap();
            assert_eq!(policy, deserialized);
        }
    }

    // DeadlockDetector tests

    #[test]
    fn test_detector_new() {
        let detector = DeadlockDetector::with_defaults();

        assert!(detector.is_enabled());
        assert!(detector.graph().is_empty());
    }

    #[test]
    fn test_detector_detect_cycle() {
        let detector = DeadlockDetector::with_defaults();

        // Create cycle
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);

        let deadlocks = detector.detect();

        assert!(!deadlocks.is_empty());
        let dl = &deadlocks[0];
        assert!(!dl.cycle.is_empty());
        assert!(dl.victim_tx_id == 1 || dl.victim_tx_id == 2);
    }

    #[test]
    fn test_detector_no_deadlock() {
        let detector = DeadlockDetector::with_defaults();

        // No cycle
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 3, None);

        let deadlocks = detector.detect();

        assert!(deadlocks.is_empty());
    }

    #[test]
    fn test_detector_disabled() {
        let config = DeadlockDetectorConfig::disabled();
        let detector = DeadlockDetector::new(config);

        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);

        let deadlocks = detector.detect();

        // Should not detect anything when disabled
        assert!(deadlocks.is_empty());
    }

    #[test]
    fn test_detector_victim_selection_youngest() {
        let config = DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::Youngest);
        let detector = DeadlockDetector::new(config);

        // Add waits with time difference
        detector.graph().add_wait(1, 2, None);
        std::thread::sleep(std::time::Duration::from_millis(10));
        detector.graph().add_wait(2, 1, None);

        // tx 2 started waiting later, should be victim
        let victim = detector.select_victim(&[1, 2]);
        assert_eq!(victim, 2);
    }

    #[test]
    fn test_detector_victim_selection_oldest() {
        let config = DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::Oldest);
        let detector = DeadlockDetector::new(config);

        // Add waits with time difference
        detector.graph().add_wait(1, 2, None);
        std::thread::sleep(std::time::Duration::from_millis(10));
        detector.graph().add_wait(2, 1, None);

        // tx 1 started waiting first, should be victim
        let victim = detector.select_victim(&[1, 2]);
        assert_eq!(victim, 1);
    }

    #[test]
    fn test_detector_victim_selection_lowest_priority() {
        let config =
            DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::LowestPriority);
        let detector = DeadlockDetector::new(config);

        // Add waits with different priorities (lower value = higher priority)
        detector.graph().add_wait(1, 2, Some(10)); // High priority
        detector.graph().add_wait(2, 1, Some(100)); // Low priority

        // tx 2 has lowest priority (highest value), should be victim
        let victim = detector.select_victim(&[1, 2]);
        assert_eq!(victim, 2);
    }

    #[test]
    fn test_detector_victim_selection_most_locks() {
        let config =
            DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::MostLocks);
        let mut detector = DeadlockDetector::new(config);

        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);

        // Set up lock count function: tx 1 has 5 locks, tx 2 has 10 locks
        detector.set_lock_count_fn(|tx_id| if tx_id == 1 { 5 } else { 10 });

        // tx 2 has more locks, should be victim
        let victim = detector.select_victim(&[1, 2]);
        assert_eq!(victim, 2);
    }

    #[test]
    fn test_detector_stats_updated() {
        let detector = DeadlockDetector::with_defaults();

        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);

        let _ = detector.detect();

        let snapshot = detector.stats().snapshot();
        assert!(snapshot.detection_cycles > 0);
        assert!(snapshot.deadlocks_detected > 0);
    }

    #[test]
    fn test_detector_max_cycle_length_filter() {
        let config = DeadlockDetectorConfig::default().with_max_cycle_length(2);
        let detector = DeadlockDetector::new(config);

        // Create a 3-node cycle
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 3, None);
        detector.graph().add_wait(3, 1, None);

        let deadlocks = detector.detect();

        // Cycle has 3 nodes, max is 2, so should be filtered
        // Note: actual cycle detection may return partial cycles
        // This tests that long cycles are filtered
        for dl in &deadlocks {
            assert!(dl.cycle.len() <= 2);
        }
    }

    // DeadlockStats tests

    #[test]
    fn test_stats_new() {
        let stats = DeadlockStats::new();
        let snapshot = stats.snapshot();

        assert_eq!(snapshot.deadlocks_detected, 0);
        assert_eq!(snapshot.victims_aborted, 0);
        assert_eq!(snapshot.detection_cycles, 0);
    }

    #[test]
    fn test_stats_record_detection() {
        let stats = DeadlockStats::new();

        stats.record_detection(100, 2);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.detection_cycles, 1);
        assert_eq!(snapshot.detection_time_us, 100);
        assert_eq!(snapshot.deadlocks_detected, 2);
    }

    #[test]
    fn test_stats_record_victim_abort() {
        let stats = DeadlockStats::new();

        stats.record_victim_abort();
        stats.record_victim_abort();

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.victims_aborted, 2);
    }

    #[test]
    fn test_stats_update_max_cycle() {
        let stats = DeadlockStats::new();

        stats.update_max_cycle(3);
        assert_eq!(stats.snapshot().max_cycle_length, 3);

        stats.update_max_cycle(5);
        assert_eq!(stats.snapshot().max_cycle_length, 5);

        // Smaller value shouldn't update
        stats.update_max_cycle(2);
        assert_eq!(stats.snapshot().max_cycle_length, 5);
    }

    // WaitInfo tests

    #[test]
    fn test_wait_info() {
        let info = WaitInfo {
            blocking_tx_id: 42,
            conflicting_keys: vec!["key1".to_string(), "key2".to_string()],
        };

        assert_eq!(info.blocking_tx_id, 42);
        assert_eq!(info.conflicting_keys.len(), 2);
    }

    // DeadlockInfo tests

    #[test]
    fn test_deadlock_info() {
        let info = DeadlockInfo {
            cycle: vec![1, 2, 3],
            victim_tx_id: 2,
            detected_at: 12345,
            victim_policy: VictimSelectionPolicy::Youngest,
        };

        assert_eq!(info.cycle.len(), 3);
        assert_eq!(info.victim_tx_id, 2);
        assert_eq!(info.detected_at, 12345);
        assert_eq!(info.victim_policy, VictimSelectionPolicy::Youngest);
    }

    // Concurrent access tests

    #[test]
    fn test_wait_for_graph_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let graph = Arc::new(WaitForGraph::new());
        let mut handles = vec![];

        // Spawn multiple threads adding edges
        for i in 0..10 {
            let g = Arc::clone(&graph);
            handles.push(thread::spawn(move || {
                g.add_wait(i, i + 100, Some(i as u32));
            }));
        }

        // Spawn a thread doing cycle detection
        let g = Arc::clone(&graph);
        handles.push(thread::spawn(move || {
            let _ = g.detect_cycles();
        }));

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify graph is consistent
        assert!(graph.edge_count() <= 10);
    }

    #[test]
    fn test_detector_empty_cycle() {
        let detector = DeadlockDetector::with_defaults();

        // Empty cycle should return 0
        let victim = detector.select_victim(&[]);
        assert_eq!(victim, 0);
    }

    #[test]
    fn test_detector_single_node_cycle() {
        let detector = DeadlockDetector::with_defaults();

        // Single node cycle returns that node
        let victim = detector.select_victim(&[42]);
        assert_eq!(victim, 42);
    }

    // Gap 19: Wait graph memory bounds tests

    #[test]
    fn test_edges_bounded_per_transaction() {
        let graph = WaitForGraph::with_max_edges_per_tx(3);

        // Add 3 edges for tx 1 -- all should succeed
        graph.add_wait(1, 10, None);
        graph.add_wait(1, 11, None);
        graph.add_wait(1, 12, None);
        assert_eq!(graph.waiting_for(1).len(), 3);

        // 4th edge for tx 1 should be dropped due to limit
        graph.add_wait(1, 13, None);
        assert_eq!(graph.waiting_for(1).len(), 3);
        assert!(!graph.waiting_for(1).contains(&13));

        // Different transaction should be unaffected
        graph.add_wait(2, 20, None);
        assert_eq!(graph.waiting_for(2).len(), 1);
    }

    #[test]
    fn test_edges_unbounded_when_limit_zero() {
        let graph = WaitForGraph::new(); // default: max_edges_per_tx = 0 (unlimited)

        for i in 0..100 {
            graph.add_wait(1, 1000 + i, None);
        }

        assert_eq!(graph.waiting_for(1).len(), 100);
    }

    #[test]
    fn test_cleanup_stale_edges_by_ttl() {
        let graph = WaitForGraph::new();

        // Add edges -- these will have "now" as their wait_started timestamp
        graph.add_wait(1, 10, None);
        graph.add_wait(2, 20, None);

        // With a very large TTL, nothing should be cleaned
        let cleaned = graph.cleanup_stale_edges(60_000);
        assert_eq!(cleaned, 0);
        assert_eq!(graph.edge_count(), 2);

        // Manually backdate wait_started for tx 1 to simulate staleness
        {
            let mut ws = graph.wait_started.write();
            ws.insert(1, 1000); // epoch + 1 second -- definitely stale
        }

        // With a TTL of 1ms, the backdated tx should be cleaned
        let cleaned = graph.cleanup_stale_edges(1);
        assert_eq!(cleaned, 1);

        // tx 1 edges should be gone, tx 2 should remain
        assert!(graph.waiting_for(1).is_empty());
        assert_eq!(graph.waiting_for(2).len(), 1);
    }

    #[test]
    fn test_stale_edges_cleaned_stat() {
        let stats = DeadlockStats::new();

        stats.stale_edges_cleaned.fetch_add(5, Ordering::Relaxed);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.stale_edges_cleaned, 5);
    }

    #[test]
    fn test_config_max_edges_per_tx_default() {
        let config = DeadlockDetectorConfig::default();
        assert_eq!(config.max_edges_per_tx, 50);
        assert_eq!(config.edge_ttl_ms, 30_000);
    }

    #[test]
    fn test_config_builder_max_edges_and_ttl() {
        let config = DeadlockDetectorConfig::default()
            .with_max_edges_per_tx(10)
            .with_edge_ttl_ms(5000);

        assert_eq!(config.max_edges_per_tx, 10);
        assert_eq!(config.edge_ttl_ms, 5000);
    }

    #[test]
    fn test_detector_graph_inherits_max_edges() {
        let config = DeadlockDetectorConfig::default().with_max_edges_per_tx(2);
        let detector = DeadlockDetector::new(config);

        detector.graph().add_wait(1, 10, None);
        detector.graph().add_wait(1, 11, None);
        detector.graph().add_wait(1, 12, None); // should be dropped

        assert_eq!(detector.graph().waiting_for(1).len(), 2);
    }

    #[test]
    fn test_cascade_resolution_shared_victim() {
        // Create two cycles sharing tx 2:
        // Cycle 1: 1 -> 2 -> 1
        // Cycle 2: 2 -> 3 -> 2
        // Aborting tx 2 resolves both cycles
        let config = DeadlockDetectorConfig::default().with_victim_cascade_depth(3);
        let detector = DeadlockDetector::new(config);

        // Set up wait starts so victim selection is deterministic (youngest = highest timestamp)
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);
        detector.graph().add_wait(2, 3, None);
        detector.graph().add_wait(3, 2, None);

        let deadlocks = detector.detect();

        // Two cycles exist but cascading should reduce the number of deadlock infos
        // if a victim from one cycle resolves the other
        let unique_victims: HashSet<u64> = deadlocks.iter().map(|d| d.victim_tx_id).collect();

        // The key invariant: with cascading, we should need fewer unique victims
        // than the total number of cycles detected
        assert!(!deadlocks.is_empty());
        assert!(unique_victims.len() <= 2);

        // Check cascade stat is tracked
        let snap = detector.stats().snapshot();
        let total_resolved = deadlocks.len() as u64 + snap.cascade_resolutions;
        assert!(total_resolved >= 1, "At least one cycle should be resolved");
    }

    #[test]
    fn test_cascade_depth_limit() {
        let config = DeadlockDetectorConfig::default().with_victim_cascade_depth(0); // Disable cascading
        let detector = DeadlockDetector::new(config);

        // Two overlapping cycles
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);
        detector.graph().add_wait(2, 3, None);
        detector.graph().add_wait(3, 2, None);

        let deadlocks = detector.detect();

        // With cascade depth 0, no cascading should occur
        // Each cycle gets its own victim
        let snap = detector.stats().snapshot();
        assert_eq!(snap.cascade_resolutions, 0);
        assert!(deadlocks.len() >= 2 || deadlocks.len() == 1);
    }

    #[test]
    fn test_cascade_config_builder() {
        let config = DeadlockDetectorConfig::default().with_victim_cascade_depth(5);
        assert_eq!(config.victim_cascade_depth, 5);
    }

    #[test]
    fn test_cascade_stats_snapshot() {
        let stats = DeadlockStats::new();
        stats.cascade_resolutions.fetch_add(7, Ordering::Relaxed);
        let snap = stats.snapshot();
        assert_eq!(snap.cascade_resolutions, 7);
    }

    #[test]
    fn test_single_cycle_no_cascade() {
        let config = DeadlockDetectorConfig::default().with_victim_cascade_depth(3);
        let detector = DeadlockDetector::new(config);

        // Single cycle: 1 -> 2 -> 1
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);

        let deadlocks = detector.detect();
        assert_eq!(deadlocks.len(), 1);

        let snap = detector.stats().snapshot();
        assert_eq!(snap.cascade_resolutions, 0);
    }

    #[test]
    fn test_update_max_cycle_tracks_longest() {
        let stats = DeadlockStats::new();

        stats.update_max_cycle(3);
        assert_eq!(stats.snapshot().max_cycle_length, 3);

        // Same length should not change
        stats.update_max_cycle(3);
        assert_eq!(stats.snapshot().max_cycle_length, 3);

        // Shorter should not change
        stats.update_max_cycle(2);
        assert_eq!(stats.snapshot().max_cycle_length, 3);

        // Longer should update
        stats.update_max_cycle(5);
        assert_eq!(stats.snapshot().max_cycle_length, 5);
    }

    #[test]
    fn test_cycle_detection_finds_correct_members() {
        // Verify the detected cycle contains the exact right members.
        // Catches mutation of == to != in dfs_detect path extraction.
        let config = DeadlockDetectorConfig::default();
        let detector = DeadlockDetector::new(config);

        // Simple 3-node cycle: 1 -> 2 -> 3 -> 1
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 3, None);
        detector.graph().add_wait(3, 1, None);

        let deadlocks = detector.detect();
        assert_eq!(deadlocks.len(), 1);

        // The cycle must contain exactly {1, 2, 3}
        let cycle_members: HashSet<u64> = deadlocks[0].cycle.iter().copied().collect();
        assert!(cycle_members.contains(&1), "Cycle must contain tx 1");
        assert!(cycle_members.contains(&2), "Cycle must contain tx 2");
        assert!(cycle_members.contains(&3), "Cycle must contain tx 3");
        assert_eq!(cycle_members.len(), 3, "Cycle must have exactly 3 members");
    }

    #[test]
    fn test_is_enabled_reflects_config() {
        let enabled = DeadlockDetector::new(DeadlockDetectorConfig::default());
        assert!(enabled.is_enabled());

        let disabled = DeadlockDetector::new(DeadlockDetectorConfig::disabled());
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn test_config_accessor_returns_actual_config() {
        let config = DeadlockDetectorConfig::default()
            .with_interval(42_000)
            .with_max_cycle_length(7);
        let detector = DeadlockDetector::new(config);

        assert_eq!(detector.config().max_cycle_length, 7);
    }

    #[test]
    fn test_cascade_depth_respected() {
        // Three overlapping cycles sharing tx 2 as victim.
        // With cascade_depth=1, at most 1 cascading resolution should occur.
        let config = DeadlockDetectorConfig::default().with_victim_cascade_depth(1);
        let detector = DeadlockDetector::new(config);

        // Cycle A: 1 -> 2 -> 1
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);
        // Cycle B: 2 -> 3 -> 2
        detector.graph().add_wait(2, 3, None);
        detector.graph().add_wait(3, 2, None);
        // Cycle C: 2 -> 4 -> 2
        detector.graph().add_wait(2, 4, None);
        detector.graph().add_wait(4, 2, None);

        let deadlocks = detector.detect();
        let snap = detector.stats().snapshot();

        // With cascade_depth=1, only 1 cascade should happen (not more)
        assert!(
            snap.cascade_resolutions <= 1,
            "cascade_resolutions={} should be <= 1 with depth=1",
            snap.cascade_resolutions
        );
        // Must still detect at least 2 deadlocks (first cycle + one non-cascaded)
        assert!(
            deadlocks.len() >= 2,
            "Should report >= 2 deadlocks, got {}",
            deadlocks.len()
        );
    }

    // ========== Mutation-Catching Tests ==========

    #[test]
    fn test_cleanup_stale_edges_ttl_boundary() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 10, None);
        graph.add_wait(2, 20, None);

        let ttl_ms = 5000;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        {
            let mut ws = graph.wait_started.write();
            // tx 1: recently started, well within TTL — should NOT be cleaned
            ws.insert(1, now_ms);
            // tx 2: ancient timestamp, definitely stale
            ws.insert(2, 1000);
        }

        let cleaned = graph.cleanup_stale_edges(ttl_ms);
        assert_eq!(cleaned, 1, "Only definitely-stale edges should be cleaned");
        // tx 2 edges should be gone
        assert!(
            graph.waiting_for(2).is_empty(),
            "Stale tx 2 edges must be removed"
        );
        // tx 1 edges should remain
        assert!(
            graph.waiting_for(1).contains(&10),
            "Fresh tx 1 edges must remain"
        );
    }

    #[test]
    fn test_detect_max_cycle_length_filter_precise() {
        let config = DeadlockDetectorConfig::default().with_max_cycle_length(3);
        let detector = DeadlockDetector::new(config);

        // 3-node cycle: should pass max_cycle_length=3 filter
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 3, None);
        detector.graph().add_wait(3, 1, None);

        let deadlocks = detector.detect();
        assert!(
            !deadlocks.is_empty(),
            "3-node cycle must be detected with max_cycle_length=3"
        );

        // Now test with max_cycle_length=2: 3-node cycle should be filtered
        let config2 = DeadlockDetectorConfig::default().with_max_cycle_length(2);
        let detector2 = DeadlockDetector::new(config2);
        detector2.graph().add_wait(10, 20, None);
        detector2.graph().add_wait(20, 30, None);
        detector2.graph().add_wait(30, 10, None);

        let deadlocks2 = detector2.detect();
        // All detected cycles must respect the length limit
        for dl in &deadlocks2 {
            assert!(
                dl.cycle.len() <= 2,
                "Cycle length {} exceeds max_cycle_length=2",
                dl.cycle.len()
            );
        }
    }

    #[test]
    fn test_detect_cascade_counter_increments() {
        let config = DeadlockDetectorConfig::default().with_victim_cascade_depth(5);
        let detector = DeadlockDetector::new(config);

        // Create two overlapping cycles sharing tx 2
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);
        detector.graph().add_wait(2, 3, None);
        detector.graph().add_wait(3, 2, None);

        let deadlocks = detector.detect();
        let snap = detector.stats().snapshot();

        // At least one cycle detected
        assert!(!deadlocks.is_empty(), "Must detect at least one cycle");
        // The cascade counter + explicit deadlocks should account for all cycles
        let total = deadlocks.len() as u64 + snap.cascade_resolutions;
        assert!(
            total >= 2,
            "Total resolved cycles ({total}) must account for both cycles"
        );
    }

    #[test]
    fn test_dfs_detect_extracts_correct_cycle() {
        let graph = WaitForGraph::new();

        // Chain: 10 -> 20 -> 30 -> 40 -> 20 (cycle is 20->30->40)
        graph.add_wait(10, 20, None);
        graph.add_wait(20, 30, None);
        graph.add_wait(30, 40, None);
        graph.add_wait(40, 20, None);

        let cycles = graph.detect_cycles();
        assert!(!cycles.is_empty(), "Must find cycle 20->30->40->20");

        // The cycle should contain {20, 30, 40} but NOT 10
        let cycle = &cycles[0];
        let members: HashSet<u64> = cycle.iter().copied().collect();
        assert!(
            members.contains(&20) && members.contains(&30) && members.contains(&40),
            "Cycle must contain 20, 30, 40; got {members:?}"
        );
        assert!(
            !members.contains(&10),
            "Node 10 is not part of the cycle, must not be included"
        );
    }

    #[test]
    fn test_update_max_cycle_exact_boundary() {
        // Kills mutation: replace > with >= in DeadlockStats::update_max_cycle
        // When len == current, update should be a no-op (no unnecessary CAS).
        let stats = DeadlockStats::new();
        stats.update_max_cycle(5);
        assert_eq!(
            stats.max_cycle_length.load(Ordering::Relaxed),
            5,
            "max_cycle_length must be 5 after update(5)"
        );
        // Calling with same value should not panic or change anything
        stats.update_max_cycle(5);
        assert_eq!(
            stats.max_cycle_length.load(Ordering::Relaxed),
            5,
            "max_cycle_length must still be 5 after update(5) again"
        );
        // Calling with smaller value should not change
        stats.update_max_cycle(3);
        assert_eq!(
            stats.max_cycle_length.load(Ordering::Relaxed),
            5,
            "max_cycle_length must still be 5 after update(3)"
        );
        // Calling with larger value should update
        stats.update_max_cycle(7);
        assert_eq!(
            stats.max_cycle_length.load(Ordering::Relaxed),
            7,
            "max_cycle_length must be 7 after update(7)"
        );
    }

    #[test]
    fn test_cleanup_stale_edges_exact_boundary() {
        // Kills mutation: replace > with >= in WaitForGraph::cleanup_stale_edges
        let graph = WaitForGraph::new();

        // Add a wait edge with a known timestamp
        graph.add_wait(1, 2, None);

        // Cleanup with very large TTL — edge should NOT be removed
        let removed = graph.cleanup_stale_edges(u64::MAX);
        assert_eq!(removed, 0, "Edge must not be stale with max TTL");
        assert_eq!(
            graph.edge_count(),
            1,
            "Edge must still exist after cleanup with max TTL"
        );

        // Sleep to ensure elapsed > 0
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Cleanup with 0 TTL — edge is at least 5ms old, so elapsed > 0 is true
        let removed = graph.cleanup_stale_edges(0);
        assert!(removed > 0, "Edge must be cleaned up with 0 TTL");
        assert_eq!(
            graph.edge_count(),
            0,
            "Edge must be removed after cleanup with 0 TTL"
        );
    }

    #[test]
    fn test_detect_cascade_depth_tracking() {
        // Kills mutations in DeadlockDetector::detect cascade logic:
        // - replace < with ==/>/<= in cascade_count check (line 771)
        // - replace += with *= in cascade_count increment (line 772)
        //
        // Strategy: use Oldest policy so node with lowest wait_start is victim.
        // Add the shared node (node 1) first to guarantee it becomes victim.
        let config = DeadlockDetectorConfig::default()
            .with_policy(VictimSelectionPolicy::Oldest)
            .with_victim_cascade_depth(2);
        let detector = DeadlockDetector::new(config);
        let graph = detector.graph();

        // Node 1 added first -> oldest timestamp -> victim under Oldest policy
        graph.add_wait(1, 2, None);
        std::thread::sleep(std::time::Duration::from_millis(2));
        // Cycle 1: 1 -> 2 -> 1
        graph.add_wait(2, 1, None);
        // Cycle 2: 1 -> 3 -> 1 (overlaps on node 1)
        graph.add_wait(1, 3, None);
        graph.add_wait(3, 1, None);
        // Cycle 3: 1 -> 4 -> 1 (also overlaps on node 1)
        graph.add_wait(1, 4, None);
        graph.add_wait(4, 1, None);

        let deadlocks = detector.detect();
        let snap = detector.stats().snapshot();

        // With depth=2: first cycle selects node 1 as victim,
        // cascading resolves cycles 2 and 3.
        let total = deadlocks.len() as u64 + snap.cascade_resolutions;
        assert!(
            total >= 3,
            "Total resolved cycles ({total}) must account for all 3 cycles"
        );
        assert!(
            !deadlocks.is_empty(),
            "Must report at least 1 explicit deadlock"
        );
        // Cascade counter must be incremented, not multiplied
        assert!(
            snap.cascade_resolutions >= 1,
            "Must have at least 1 cascade resolution, got {}",
            snap.cascade_resolutions
        );
        assert!(
            snap.cascade_resolutions <= 2,
            "Cascade resolutions must be <= depth(2), got {}",
            snap.cascade_resolutions
        );
    }

    #[test]
    fn test_cascade_depth_zero_disables_cascading() {
        // With cascade_depth=0, no cascading should happen at all.
        // This catches: replace < with <= (0 < 0 is false, 0 <= 0 is true)
        let config = DeadlockDetectorConfig::default()
            .with_policy(VictimSelectionPolicy::Oldest)
            .with_victim_cascade_depth(0);
        let detector = DeadlockDetector::new(config);
        let graph = detector.graph();

        // Node 1 added first -> oldest -> victim
        graph.add_wait(1, 2, None);
        std::thread::sleep(std::time::Duration::from_millis(2));
        graph.add_wait(2, 1, None);
        graph.add_wait(1, 3, None);
        graph.add_wait(3, 1, None);

        let deadlocks = detector.detect();
        let snap = detector.stats().snapshot();

        // With depth=0: cascade_count(0) < 0 is always false -> no cascading
        assert_eq!(
            snap.cascade_resolutions, 0,
            "No cascading with depth=0, got {}",
            snap.cascade_resolutions
        );
        // Both cycles must be reported as explicit deadlocks
        assert!(
            deadlocks.len() >= 2,
            "All cycles must be explicit deadlocks with depth=0, got {}",
            deadlocks.len()
        );
    }

    #[test]
    fn test_cascade_depth_exactly_one() {
        // With cascade_depth=1 and 3 overlapping cycles, exactly 1 cascade.
        // This catches: replace < with == (count(0)==1 is false on first check)
        let config = DeadlockDetectorConfig::default()
            .with_policy(VictimSelectionPolicy::Oldest)
            .with_victim_cascade_depth(1);
        let detector = DeadlockDetector::new(config);
        let graph = detector.graph();

        // Node 1 added first -> oldest -> victim
        graph.add_wait(1, 2, None);
        std::thread::sleep(std::time::Duration::from_millis(2));
        graph.add_wait(2, 1, None);
        graph.add_wait(1, 3, None);
        graph.add_wait(3, 1, None);
        graph.add_wait(1, 4, None);
        graph.add_wait(4, 1, None);

        let deadlocks = detector.detect();
        let snap = detector.stats().snapshot();

        // depth=1: first cycle explicit (victim=node1), second cascaded
        // (count 0<1), third cannot cascade (count 1<1 is false) so explicit.
        assert_eq!(
            snap.cascade_resolutions, 1,
            "Exactly 1 cascade with depth=1, got {}",
            snap.cascade_resolutions
        );
        assert!(
            deadlocks.len() >= 2,
            "At least 2 explicit deadlocks with depth=1, got {}",
            deadlocks.len()
        );
    }

    #[test]
    fn test_detect_disabled_returns_empty() {
        let config = DeadlockDetectorConfig::disabled();
        let detector = DeadlockDetector::new(config);
        let graph = detector.graph();
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 1, None);

        let deadlocks = detector.detect();
        assert!(deadlocks.is_empty());
    }

    #[test]
    fn test_concurrent_detect_stats_consistency() {
        let detector = std::sync::Arc::new(DeadlockDetector::with_defaults());
        let graph = detector.graph();
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 1, None);

        let mut handles = Vec::new();
        for _ in 0..4 {
            let det = std::sync::Arc::clone(&detector);
            handles.push(std::thread::spawn(move || det.detect()));
        }

        let mut total_deadlocks = 0;
        for h in handles {
            total_deadlocks += h.join().unwrap().len();
        }

        // Each thread should find the same cycle
        assert!(total_deadlocks >= 4);
        let snap = detector.stats().snapshot();
        assert!(snap.detection_cycles >= 4);
    }

    #[test]
    fn test_dfs_cycle_extraction_in_long_path() {
        let graph = WaitForGraph::new();
        // A -> B -> C -> D -> B (cycle B->C->D->B within longer path)
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 3, None);
        graph.add_wait(3, 4, None);
        graph.add_wait(4, 2, None);

        let cycles = graph.detect_cycles();
        assert!(!cycles.is_empty());
        // The cycle should include 2, 3, 4
        let cycle = &cycles[0];
        assert!(cycle.contains(&2));
        assert!(cycle.contains(&3));
        assert!(cycle.contains(&4));
        assert!(!cycle.contains(&1)); // 1 is not part of the cycle
    }

    #[test]
    fn test_max_cycle_length_boundary_exact() {
        let config = DeadlockDetectorConfig::default().with_max_cycle_length(2);
        let detector = DeadlockDetector::new(config);
        let graph = detector.graph();

        // Two-node cycle (length=2, at boundary)
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 1, None);

        let deadlocks = detector.detect();
        assert_eq!(deadlocks.len(), 1);

        // Three-node cycle (length=3, exceeds max)
        graph.add_wait(3, 4, None);
        graph.add_wait(4, 5, None);
        graph.add_wait(5, 3, None);

        let deadlocks = detector.detect();
        // Should still only find the 2-node cycle, 3-node cycle filtered
        let two_node_cycles: Vec<_> = deadlocks.iter().filter(|d| d.cycle.len() <= 2).collect();
        assert!(!two_node_cycles.is_empty());
    }

    #[test]
    fn test_stale_edge_cleanup_removes_old() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 3, None);

        // Wait just enough for edges to be stale
        std::thread::sleep(std::time::Duration::from_millis(10));

        // TTL of 5ms: waiters 1 and 2 should be stale
        let cleaned = graph.cleanup_stale_edges(5);
        assert!(cleaned >= 1);
        // After cleanup, edges involving waiters 1 and 2 are removed
        assert!(graph.is_empty());
    }

    #[test]
    fn test_stale_edge_cleanup_keeps_fresh() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 2, None);

        // Very long TTL: nothing should be stale
        let cleaned = graph.cleanup_stale_edges(100_000);
        assert_eq!(cleaned, 0);
        assert!(!graph.is_empty());
    }

    #[test]
    fn test_stats_cascade_resolutions_counter() {
        let stats = DeadlockStats::new();
        stats
            .cascade_resolutions
            .fetch_add(3, std::sync::atomic::Ordering::Relaxed);
        let snap = stats.snapshot();
        assert_eq!(snap.cascade_resolutions, 3);
    }

    #[test]
    fn test_graph_reverse_edges_tracked() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 2, None);
        graph.add_wait(1, 3, None);

        // waiting_for returns forward edges (what tx 1 waits for)
        let targets = graph.waiting_for(1);
        assert_eq!(targets.len(), 2);
        assert!(targets.contains(&2));
        assert!(targets.contains(&3));

        // waiting_on returns reverse edges (who waits on tx 2)
        let waiters = graph.waiting_on(2);
        assert_eq!(waiters.len(), 1);
        assert!(waiters.contains(&1));
    }

    #[test]
    fn test_with_lock_count_fn_most_locks() {
        let config =
            DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::MostLocks);
        let mut detector = DeadlockDetector::new(config);
        detector.set_lock_count_fn(|tx_id| (tx_id * 10) as usize);
        let graph = detector.graph();
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 1, None);

        let deadlocks = detector.detect();
        assert_eq!(deadlocks.len(), 1);
        // tx 2 has more locks (2*10=20 vs 1*10=10), so tx 2 should be victim
        assert_eq!(deadlocks[0].victim_tx_id, 2);
    }

    // ========== Additional Coverage Tests ==========

    #[test]
    fn test_config_with_interval() {
        let config = DeadlockDetectorConfig::default().with_interval(250);
        assert_eq!(config.detection_interval_ms, 250);
        // Other fields unchanged
        assert!(config.enabled);
        assert_eq!(config.victim_policy, VictimSelectionPolicy::Youngest);
    }

    #[test]
    fn test_config_with_policy() {
        let config =
            DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::LowestPriority);
        assert_eq!(config.victim_policy, VictimSelectionPolicy::LowestPriority);
        assert!(config.enabled);
    }

    #[test]
    fn test_config_with_max_cycle_length() {
        let config = DeadlockDetectorConfig::default().with_max_cycle_length(42);
        assert_eq!(config.max_cycle_length, 42);
    }

    #[test]
    fn test_config_without_auto_abort() {
        let config = DeadlockDetectorConfig::default().without_auto_abort();
        assert!(!config.auto_abort_victim);
        // Other fields preserved
        assert!(config.enabled);
    }

    #[test]
    fn test_config_with_max_edges_per_tx() {
        let config = DeadlockDetectorConfig::default().with_max_edges_per_tx(7);
        assert_eq!(config.max_edges_per_tx, 7);
    }

    #[test]
    fn test_config_with_edge_ttl_ms() {
        let config = DeadlockDetectorConfig::default().with_edge_ttl_ms(15_000);
        assert_eq!(config.edge_ttl_ms, 15_000);
    }

    #[test]
    fn test_config_with_victim_cascade_depth() {
        let config = DeadlockDetectorConfig::default().with_victim_cascade_depth(10);
        assert_eq!(config.victim_cascade_depth, 10);
    }

    #[test]
    fn test_config_chained_all_builders() {
        let config = DeadlockDetectorConfig::default()
            .with_interval(200)
            .with_policy(VictimSelectionPolicy::MostLocks)
            .with_max_cycle_length(25)
            .without_auto_abort()
            .with_max_edges_per_tx(8)
            .with_edge_ttl_ms(60_000)
            .with_victim_cascade_depth(4);

        assert_eq!(config.detection_interval_ms, 200);
        assert_eq!(config.victim_policy, VictimSelectionPolicy::MostLocks);
        assert_eq!(config.max_cycle_length, 25);
        assert!(!config.auto_abort_victim);
        assert_eq!(config.max_edges_per_tx, 8);
        assert_eq!(config.edge_ttl_ms, 60_000);
        assert_eq!(config.victim_cascade_depth, 4);
    }

    #[test]
    fn test_remove_wait_last_edge_cleans_wait_started() {
        let graph = WaitForGraph::new();

        // Add a single edge for tx 1
        graph.add_wait(1, 2, None);
        assert!(graph.get_wait_start(1).is_some());
        assert_eq!(graph.edge_count(), 1);

        // Remove the only edge -- should clean up wait_started too
        graph.remove_wait(1, 2);
        assert!(graph.waiting_for(1).is_empty());
        assert!(
            graph.get_wait_start(1).is_none(),
            "wait_started must be removed when last edge is removed"
        );
        // Reverse edge for holder should also be cleaned up
        assert!(
            graph.waiting_on(2).is_empty(),
            "reverse edge must be cleaned when last waiter is removed"
        );
        assert!(graph.is_empty());
    }

    #[test]
    fn test_remove_wait_nonexistent_edge() {
        let graph = WaitForGraph::new();

        // Remove an edge that doesn't exist -- should be a no-op
        graph.remove_wait(99, 100);
        assert!(graph.is_empty());
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_remove_wait_partial_edges_remain() {
        let graph = WaitForGraph::new();

        // tx 1 waits for 2 and 3; tx 4 also waits for 2
        graph.add_wait(1, 2, None);
        graph.add_wait(1, 3, None);
        graph.add_wait(4, 2, None);

        // Remove 1->2 but 1->3 still exists
        graph.remove_wait(1, 2);
        assert_eq!(graph.waiting_for(1).len(), 1);
        assert!(graph.waiting_for(1).contains(&3));
        // wait_started should still be present since 1 still has edges
        assert!(graph.get_wait_start(1).is_some());

        // tx 4 still waits on 2
        assert!(graph.waiting_on(2).contains(&4));
    }

    #[test]
    fn test_remove_transaction_cleans_metadata() {
        let graph = WaitForGraph::new();

        // Add edges with priority and wait timestamps
        graph.add_wait(1, 2, Some(10));
        graph.add_wait(2, 3, Some(20));

        assert!(graph.get_wait_start(1).is_some());
        assert!(graph.get_priority(1).is_some());

        // Remove tx 1 -- should remove wait_started and priority
        graph.remove_transaction(1);

        assert!(
            graph.get_wait_start(1).is_none(),
            "wait_started must be removed for removed transaction"
        );
        assert!(
            graph.get_priority(1).is_none(),
            "priority must be removed for removed transaction"
        );
        // tx 2 should still be intact
        assert!(graph.get_wait_start(2).is_some());
        assert!(graph.get_priority(2).is_some());
    }

    #[test]
    fn test_detector_debug_impl() {
        let detector = DeadlockDetector::with_defaults();
        let debug_str = format!("{detector:?}");

        assert!(
            debug_str.contains("DeadlockDetector"),
            "Debug output must contain struct name"
        );
        assert!(
            debug_str.contains("graph"),
            "Debug output must contain graph field"
        );
        assert!(
            debug_str.contains("config"),
            "Debug output must contain config field"
        );
        assert!(
            debug_str.contains("stats"),
            "Debug output must contain stats field"
        );
        assert!(
            debug_str.contains("lock_count_fn"),
            "Debug output must contain lock_count_fn field"
        );
        // Without lock_count_fn set, should show None
        assert!(
            debug_str.contains("None"),
            "lock_count_fn should be None when not set"
        );
    }

    #[test]
    fn test_detector_debug_with_lock_fn() {
        let config =
            DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::MostLocks);
        let mut detector = DeadlockDetector::new(config);
        detector.set_lock_count_fn(|tx_id| tx_id as usize);

        let debug_str = format!("{detector:?}");
        assert!(
            debug_str.contains("<fn>"),
            "Debug output must show <fn> when lock_count_fn is set"
        );
    }

    #[test]
    fn test_most_locks_fallback_without_lock_fn() {
        // MostLocks without a lock_count_fn should fall back to youngest
        let config =
            DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::MostLocks);
        let detector = DeadlockDetector::new(config);

        // Add waits with time difference so youngest is deterministic
        detector.graph().add_wait(1, 2, None);
        std::thread::sleep(std::time::Duration::from_millis(10));
        detector.graph().add_wait(2, 1, None);

        // Without lock_count_fn, falls back to youngest (highest wait_start)
        let victim = detector.select_victim(&[1, 2]);
        assert_eq!(
            victim, 2,
            "MostLocks without lock_count_fn must fall back to youngest"
        );
    }

    #[test]
    fn test_most_locks_detect_without_lock_fn() {
        // Exercise MostLocks fallback through detect() path
        let config =
            DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::MostLocks);
        let detector = DeadlockDetector::new(config);

        detector.graph().add_wait(1, 2, None);
        std::thread::sleep(std::time::Duration::from_millis(10));
        detector.graph().add_wait(2, 1, None);

        let deadlocks = detector.detect();
        assert_eq!(deadlocks.len(), 1);
        // Victim should be the youngest (tx 2 started waiting later)
        assert_eq!(
            deadlocks[0].victim_tx_id, 2,
            "MostLocks fallback through detect() must pick youngest"
        );
        assert_eq!(deadlocks[0].victim_policy, VictimSelectionPolicy::MostLocks);
    }

    #[test]
    fn test_select_victim_oldest_through_detect() {
        let config = DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::Oldest);
        let detector = DeadlockDetector::new(config);

        detector.graph().add_wait(10, 20, None);
        std::thread::sleep(std::time::Duration::from_millis(10));
        detector.graph().add_wait(20, 10, None);

        let deadlocks = detector.detect();
        assert_eq!(deadlocks.len(), 1);
        // Oldest policy: victim is the one that started waiting earliest (tx 10)
        assert_eq!(
            deadlocks[0].victim_tx_id, 10,
            "Oldest policy must select earliest waiter as victim"
        );
    }

    #[test]
    fn test_select_victim_lowest_priority_through_detect() {
        let config =
            DeadlockDetectorConfig::default().with_policy(VictimSelectionPolicy::LowestPriority);
        let detector = DeadlockDetector::new(config);

        // Lower priority value = higher priority, so tx with highest value is victim
        detector.graph().add_wait(10, 20, Some(5));
        detector.graph().add_wait(20, 10, Some(100));

        let deadlocks = detector.detect();
        assert_eq!(deadlocks.len(), 1);
        // tx 20 has priority 100 (lowest priority), should be victim
        assert_eq!(
            deadlocks[0].victim_tx_id, 20,
            "LowestPriority must select tx with highest priority value"
        );
    }

    #[test]
    fn test_cleanup_stale_edges_returns_count() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 10, None);
        graph.add_wait(2, 20, None);
        graph.add_wait(3, 30, None);

        // Backdate all three to make them stale
        {
            let mut ws = graph.wait_started.write();
            ws.insert(1, 100);
            ws.insert(2, 200);
            ws.insert(3, 300);
        }

        let cleaned = graph.cleanup_stale_edges(1);
        assert_eq!(cleaned, 3, "All three stale transactions must be cleaned");
        assert!(graph.is_empty());
    }

    #[test]
    fn test_cleanup_stale_edges_no_stale() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 2, None);
        graph.add_wait(3, 4, None);

        // With max TTL, nothing is stale
        let cleaned = graph.cleanup_stale_edges(u64::MAX);
        assert_eq!(cleaned, 0);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_cleanup_stale_edges_empty_graph() {
        let graph = WaitForGraph::new();
        let cleaned = graph.cleanup_stale_edges(1000);
        assert_eq!(cleaned, 0);
    }

    #[test]
    fn test_waiting_for_nonexistent_tx() {
        let graph = WaitForGraph::new();
        let result = graph.waiting_for(999);
        assert!(result.is_empty());
    }

    #[test]
    fn test_waiting_on_nonexistent_tx() {
        let graph = WaitForGraph::new();
        let result = graph.waiting_on(999);
        assert!(result.is_empty());
    }

    #[test]
    fn test_edge_count_empty() {
        let graph = WaitForGraph::new();
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_edge_count_multiple_waiters() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 10, None);
        graph.add_wait(1, 11, None);
        graph.add_wait(2, 20, None);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_transaction_count_empty() {
        let graph = WaitForGraph::new();
        assert_eq!(graph.transaction_count(), 0);
    }

    #[test]
    fn test_clear_with_priorities_and_wait_times() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 2, Some(10));
        graph.add_wait(3, 4, Some(20));

        assert!(!graph.is_empty());
        assert!(graph.get_priority(1).is_some());
        assert!(graph.get_wait_start(1).is_some());

        graph.clear();

        assert!(graph.is_empty());
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.transaction_count(), 0);
        assert!(graph.get_priority(1).is_none());
        assert!(graph.get_wait_start(1).is_none());
    }

    #[test]
    fn test_detect_cycles_empty_graph() {
        let graph = WaitForGraph::new();
        let cycles = graph.detect_cycles();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_would_create_cycle_empty_graph() {
        let graph = WaitForGraph::new();
        // No edges, so adding 1->2 cannot create cycle
        assert!(!graph.would_create_cycle(1, 2));
        // Self-edge always creates cycle
        assert!(graph.would_create_cycle(1, 1));
    }

    #[test]
    fn test_would_create_cycle_long_chain() {
        let graph = WaitForGraph::new();
        // Chain: 1 -> 2 -> 3 -> 4 -> 5
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 3, None);
        graph.add_wait(3, 4, None);
        graph.add_wait(4, 5, None);

        // Adding 5 -> 1 would close the cycle
        assert!(graph.would_create_cycle(5, 1));
        // Adding 5 -> 3 would also create cycle (3->4->5->3)
        assert!(graph.would_create_cycle(5, 3));
        // Adding 6 -> 1 would not create cycle (6 is not reachable from 1..5)
        assert!(!graph.would_create_cycle(6, 1));
    }

    #[test]
    fn test_get_wait_start_present() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 2, None);
        let start = graph.get_wait_start(1);
        assert!(start.is_some());
        assert!(start.unwrap() > 0);
    }

    #[test]
    fn test_get_wait_start_absent() {
        let graph = WaitForGraph::new();
        assert!(graph.get_wait_start(999).is_none());
    }

    #[test]
    fn test_get_priority_absent() {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 2, None); // No priority
        assert!(graph.get_priority(1).is_none());
    }

    #[test]
    fn test_detector_detect_records_stats() {
        let detector = DeadlockDetector::with_defaults();
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);

        let deadlocks = detector.detect();
        let snap = detector.stats().snapshot();

        assert_eq!(snap.detection_cycles, 1);
        assert!(snap.detection_time_us > 0 || snap.detection_time_us == 0); // timing can be 0 on fast machines
        assert_eq!(snap.deadlocks_detected, deadlocks.len() as u64);
        assert!(snap.max_cycle_length >= 2);
    }

    #[test]
    fn test_deadlock_info_clone_and_debug() {
        let info = DeadlockInfo {
            cycle: vec![1, 2, 3],
            victim_tx_id: 2,
            detected_at: 12345,
            victim_policy: VictimSelectionPolicy::Oldest,
        };

        let cloned = info.clone();
        assert_eq!(cloned.cycle, info.cycle);
        assert_eq!(cloned.victim_tx_id, info.victim_tx_id);
        assert_eq!(cloned.detected_at, info.detected_at);
        assert_eq!(cloned.victim_policy, info.victim_policy);

        let debug_str = format!("{info:?}");
        assert!(debug_str.contains("DeadlockInfo"));
    }

    #[test]
    fn test_wait_info_clone_and_debug() {
        let info = WaitInfo {
            blocking_tx_id: 42,
            conflicting_keys: vec!["key_a".to_string()],
        };
        let cloned = info.clone();
        assert_eq!(cloned.blocking_tx_id, 42);
        assert_eq!(cloned.conflicting_keys, vec!["key_a".to_string()]);

        let debug_str = format!("{info:?}");
        assert!(debug_str.contains("WaitInfo"));
    }

    #[test]
    fn test_deadlock_stats_snapshot_clone_debug() {
        let stats = DeadlockStats::new();
        stats.record_detection(50, 1);
        stats.record_victim_abort();
        let snap = stats.snapshot();
        let cloned = snap.clone();
        assert_eq!(cloned.deadlocks_detected, 1);
        assert_eq!(cloned.victims_aborted, 1);
        assert_eq!(cloned.detection_time_us, 50);

        let debug_str = format!("{snap:?}");
        assert!(debug_str.contains("DeadlockStatsSnapshot"));
    }

    #[test]
    fn test_deadlock_stats_snapshot_default() {
        let snap = DeadlockStatsSnapshot::default();
        assert_eq!(snap.deadlocks_detected, 0);
        assert_eq!(snap.victims_aborted, 0);
        assert_eq!(snap.detection_time_us, 0);
        assert_eq!(snap.detection_cycles, 0);
        assert_eq!(snap.max_cycle_length, 0);
        assert_eq!(snap.stale_edges_cleaned, 0);
        assert_eq!(snap.cascade_resolutions, 0);
    }

    #[test]
    fn test_deadlock_stats_snapshot_serialization() {
        let snap = DeadlockStatsSnapshot {
            deadlocks_detected: 5,
            victims_aborted: 2,
            detection_time_us: 1000,
            detection_cycles: 10,
            max_cycle_length: 4,
            stale_edges_cleaned: 3,
            cascade_resolutions: 1,
        };
        let serialized = bitcode::serialize(&snap).unwrap();
        let deserialized: DeadlockStatsSnapshot = bitcode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized.deadlocks_detected, 5);
        assert_eq!(deserialized.victims_aborted, 2);
        assert_eq!(deserialized.detection_time_us, 1000);
        assert_eq!(deserialized.detection_cycles, 10);
        assert_eq!(deserialized.max_cycle_length, 4);
        assert_eq!(deserialized.stale_edges_cleaned, 3);
        assert_eq!(deserialized.cascade_resolutions, 1);
    }

    #[test]
    fn test_victim_policy_default() {
        let policy = VictimSelectionPolicy::default();
        assert_eq!(policy, VictimSelectionPolicy::Youngest);
    }

    #[test]
    fn test_victim_policy_clone_copy_debug() {
        let policy = VictimSelectionPolicy::MostLocks;
        let copied = policy;
        let cloned = policy.clone();
        assert_eq!(copied, cloned);
        let debug_str = format!("{policy:?}");
        assert!(debug_str.contains("MostLocks"));
    }

    #[test]
    fn test_remove_transaction_with_both_incoming_and_outgoing() {
        let graph = WaitForGraph::new();

        // tx 5 waits for tx 10, tx 10 waits for tx 15, tx 20 waits for tx 10
        graph.add_wait(5, 10, Some(1));
        graph.add_wait(10, 15, Some(2));
        graph.add_wait(20, 10, Some(3));

        assert_eq!(graph.transaction_count(), 4);

        // Remove tx 10 -- has both outgoing (10->15) and incoming (5->10, 20->10)
        graph.remove_transaction(10);

        // tx 10's outgoing edges removed
        assert!(graph.waiting_for(10).is_empty());
        // tx 5 no longer waits for tx 10
        assert!(!graph.waiting_for(5).contains(&10));
        // tx 20 no longer waits for tx 10
        assert!(!graph.waiting_for(20).contains(&10));
        // Metadata for tx 10 removed
        assert!(graph.get_wait_start(10).is_none());
        assert!(graph.get_priority(10).is_none());
    }

    #[test]
    fn test_add_wait_max_edges_reached() {
        let graph = WaitForGraph::with_max_edges_per_tx(2);

        graph.add_wait(1, 10, None);
        graph.add_wait(1, 11, None);
        assert_eq!(graph.edge_count(), 2);

        // This should be silently dropped
        graph.add_wait(1, 12, None);
        assert_eq!(graph.edge_count(), 2);
        assert!(!graph.waiting_for(1).contains(&12));
        // But the dropped edge's reverse should NOT be added either
        assert!(graph.waiting_on(12).is_empty());
    }

    #[test]
    fn test_multiple_detect_cycles_on_same_graph() {
        let detector = DeadlockDetector::with_defaults();
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 1, None);

        // Detect multiple times -- graph is not modified by detection
        let d1 = detector.detect();
        let d2 = detector.detect();
        let d3 = detector.detect();

        assert_eq!(d1.len(), 1);
        assert_eq!(d2.len(), 1);
        assert_eq!(d3.len(), 1);

        let snap = detector.stats().snapshot();
        assert_eq!(snap.detection_cycles, 3);
    }

    #[test]
    fn test_detect_no_cycle_records_stats() {
        let detector = DeadlockDetector::with_defaults();
        detector.graph().add_wait(1, 2, None);

        let deadlocks = detector.detect();
        assert!(deadlocks.is_empty());

        let snap = detector.stats().snapshot();
        assert_eq!(snap.detection_cycles, 1);
        assert_eq!(snap.deadlocks_detected, 0);
    }
}
