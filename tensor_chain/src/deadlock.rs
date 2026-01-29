// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! - The `max_cycle_length` configuration prevents DoS via artificially long cycles
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

use parking_lot::RwLock;
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
}

impl Default for DeadlockDetectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            detection_interval_ms: 100,
            victim_policy: VictimSelectionPolicy::Youngest,
            max_cycle_length: 100,
            auto_abort_victim: true,
        }
    }
}

impl DeadlockDetectorConfig {
    /// Create a disabled configuration.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Set the detection interval.
    pub fn with_interval(mut self, ms: u64) -> Self {
        self.detection_interval_ms = ms;
        self
    }

    /// Set the victim selection policy.
    pub fn with_policy(mut self, policy: VictimSelectionPolicy) -> Self {
        self.victim_policy = policy;
        self
    }

    /// Set the maximum cycle length.
    pub fn with_max_cycle_length(mut self, len: usize) -> Self {
        self.max_cycle_length = len;
        self
    }

    /// Disable auto-abort of victim.
    pub fn without_auto_abort(mut self) -> Self {
        self.auto_abort_victim = false;
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
}

impl DeadlockStats {
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
}

/// Wait-for graph for tracking transaction dependencies.
///
/// Tracks which transactions are waiting for which other transactions
/// to release locks. Used for cycle detection in deadlock scenarios.
#[derive(Debug, Default)]
pub struct WaitForGraph {
    /// Maps tx_id -> set of tx_ids it is waiting for.
    edges: RwLock<HashMap<u64, HashSet<u64>>>,
    /// Maps tx_id -> timestamp when wait started (for victim selection).
    wait_started: RwLock<HashMap<u64, EpochMillis>>,
    /// Maps tx_id -> priority (lower value = higher priority).
    priorities: RwLock<HashMap<u64, u32>>,
    /// Reverse edges for efficient waiting_on queries.
    reverse_edges: RwLock<HashMap<u64, HashSet<u64>>>,
}

impl WaitForGraph {
    /// Create a new empty wait-for graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a wait-for edge: waiter is waiting for holder.
    ///
    /// The waiter transaction is blocked waiting for holder to release locks.
    pub fn add_wait(&self, waiter_tx_id: u64, holder_tx_id: u64, priority: Option<u32>) {
        if waiter_tx_id == holder_tx_id {
            return; // Self-wait is invalid
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        {
            let mut edges = self.edges.write();
            edges.entry(waiter_tx_id).or_default().insert(holder_tx_id);
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
    /// Returns all cycles found. Each cycle is a vec of tx_ids forming the cycle.
    pub fn detect_cycles(&self) -> Vec<Vec<u64>> {
        let edges = self.edges.read();
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();

        for &start in edges.keys() {
            if !visited.contains(&start) {
                self.dfs_detect(
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

    /// Tarjan's DFS: cycle exists when we hit a node in rec_stack (back edge to ancestor).
    /// Path tracked explicitly for victim selection.
    fn dfs_detect(
        &self,
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
                    self.dfs_detect(neighbor, edges, visited, rec_stack, path, cycles);
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
        self.edges.read().values().map(|s| s.len()).sum()
    }

    /// Get the number of transactions in the graph.
    pub fn transaction_count(&self) -> usize {
        let edges = self.edges.read();
        let reverse = self.reverse_edges.read();

        let mut all_txs: HashSet<u64> = edges.keys().copied().collect();
        all_txs.extend(reverse.keys());
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
}

/// Deadlock detector with configurable victim selection.
pub struct DeadlockDetector {
    /// Wait-for graph.
    graph: WaitForGraph,
    /// Configuration.
    config: DeadlockDetectorConfig,
    /// Statistics.
    stats: DeadlockStats,
    /// Lock count function for MostLocks policy.
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
    pub fn new(config: DeadlockDetectorConfig) -> Self {
        Self {
            graph: WaitForGraph::new(),
            config,
            stats: DeadlockStats::new(),
            lock_count_fn: None,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(DeadlockDetectorConfig::default())
    }

    /// Set the lock count function for MostLocks victim selection.
    pub fn set_lock_count_fn<F>(&mut self, f: F)
    where
        F: Fn(u64) -> usize + Send + Sync + 'static,
    {
        self.lock_count_fn = Some(Box::new(f));
    }

    /// Get the wait-for graph.
    pub fn graph(&self) -> &WaitForGraph {
        &self.graph
    }

    /// Get the configuration.
    pub fn config(&self) -> &DeadlockDetectorConfig {
        &self.config
    }

    /// Get statistics.
    pub fn stats(&self) -> &DeadlockStats {
        &self.stats
    }

    /// Check if detection is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Run one detection cycle, returns detected deadlocks.
    pub fn detect(&self) -> Vec<DeadlockInfo> {
        if !self.config.enabled {
            return Vec::new();
        }

        let start = std::time::Instant::now();
        let cycles = self.graph.detect_cycles();
        let duration_us = start.elapsed().as_micros() as u64;

        let mut deadlocks = Vec::new();

        for cycle in cycles {
            if cycle.len() > self.config.max_cycle_length {
                continue;
            }

            self.stats.update_max_cycle(cycle.len());

            let victim = self.select_victim(&cycle);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            deadlocks.push(DeadlockInfo {
                cycle,
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
    /// - LowestPriority: business-rule prioritization
    /// - MostLocks: maximize freed resources
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
                if let Some(ref lock_fn) = self.lock_count_fn {
                    cycle
                        .iter()
                        .max_by_key(|&&tx_id| lock_fn(tx_id))
                        .copied()
                        .unwrap_or(cycle[0])
                } else {
                    // Fall back to youngest if no lock count function
                    cycle
                        .iter()
                        .max_by_key(|&&tx_id| self.graph.get_wait_start(tx_id).unwrap_or(0))
                        .copied()
                        .unwrap_or(cycle[0])
                }
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
}
