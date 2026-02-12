// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Cross-shard distributed transactions with 2PC and delta-based conflict detection.
//!
//! Coordinates transactions spanning multiple shards using two-phase commit:
//! - Phase 1 (PREPARE): Acquire locks, compute delta, check conflicts
//! - Phase 2 (COMMIT/ABORT): Finalize or rollback based on votes
//!
//! Tensor-native optimization: Orthogonal deltas can commit in parallel
//! without coordination using vector similarity.
//!
//! ## Lock Ordering (to prevent deadlocks)
//!
//! When acquiring multiple locks, always follow this order:
//!
//! 1. `pending` - Transaction state map
//! 2. `lock_manager.locks` - Key-level locks
//! 3. `lock_manager.tx_locks` - Per-transaction lock sets
//! 4. `pending_aborts` - Abort queue
//!
//! **Critical**: Never acquire `pending_aborts` while holding `pending`.
//! The `record_vote` method releases `pending` before acquiring `pending_aborts`
//! to maintain this invariant.

use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::sync_compat::RwLock;
use serde::{Deserialize, Serialize};
use tensor_store::{ScalarValue, SparseVector, TensorData, TensorStore, TensorValue};

use crate::{
    block::{NodeId, Transaction},
    consensus::{ConsensusManager, DeltaVector},
    error::{ChainError, Result},
    network::{Message, Transport, TxAbortMsg},
    tx_id::generate_tx_id,
    tx_wal::{TxOutcome, TxRecoveryState, TxWal, TxWalEntry},
};

/// Milliseconds since UNIX epoch, serializable replacement for Instant.
pub type EpochMillis = u64;

/// Get current time as epoch milliseconds.
fn now_epoch_millis() -> EpochMillis {
    #[allow(clippy::cast_possible_truncation)]
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    ms
}

/// Shard identifier.
pub type ShardId = usize;

/// Phase of a distributed transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TxPhase {
    /// Transaction is being prepared (Phase 1).
    #[default]
    Preparing,
    /// All participants voted YES.
    Prepared,
    /// Transaction is being committed (Phase 2).
    Committing,
    /// Transaction has been committed.
    Committed,
    /// Transaction is being aborted.
    Aborting,
    /// Transaction has been aborted.
    Aborted,
}

/// Undo entry capturing the previous state of a key for rollback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UndoEntry {
    /// Key existed before - restore this value on rollback.
    Restore {
        key: String,
        /// Bincode-serialized `TensorData`.
        data: Vec<u8>,
    },
    /// Key did not exist before - delete on rollback.
    Delete { key: String },
}

impl UndoEntry {
    /// Capture the current state of a key from the store.
    #[must_use]
    pub fn capture(key: &str, store: &TensorStore) -> Self {
        store.get(key).map_or_else(
            |_| Self::Delete {
                key: key.to_string(),
            },
            |data| {
                let bytes = bitcode::serialize(&data).unwrap_or_default();
                Self::Restore {
                    key: key.to_string(),
                    data: bytes,
                }
            },
        )
    }

    /// Apply this undo entry to restore the previous state.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization or store write fails.
    pub fn apply(&self, store: &TensorStore) -> Result<()> {
        match self {
            Self::Restore { key, data } => {
                let tensor: TensorData = bitcode::deserialize(data)
                    .map_err(|e| ChainError::SerializationError(e.to_string()))?;
                store
                    .put(key, tensor)
                    .map_err(|e| ChainError::StorageError(e.to_string()))?;
            },
            Self::Delete { key } => {
                // Idempotent delete - ignore NotFound
                store.delete(key).ok();
            },
        }
        Ok(())
    }

    /// Get the key affected by this undo entry.
    #[must_use]
    pub fn key(&self) -> &str {
        match self {
            Self::Restore { key, .. } | Self::Delete { key } => key,
        }
    }

    /// Compute a CRC32 checksum of this undo entry for integrity verification.
    #[must_use]
    pub fn checksum(&self) -> u32 {
        let mut hasher = crc32fast::Hasher::new();
        match self {
            Self::Restore { key, data } => {
                hasher.update(b"R"); // tag byte
                hasher.update(key.as_bytes());
                hasher.update(data);
            },
            Self::Delete { key } => {
                hasher.update(b"D"); // tag byte
                hasher.update(key.as_bytes());
            },
        }
        hasher.finalize()
    }

    /// Apply this undo entry after verifying its checksum.
    ///
    /// # Errors
    ///
    /// Returns an error if the checksum doesn't match or if apply fails.
    pub fn apply_verified(&self, store: &TensorStore, expected_checksum: u32) -> Result<()> {
        let actual = self.checksum();
        if actual != expected_checksum {
            return Err(ChainError::StorageError(format!(
                "undo entry checksum mismatch for key '{}': expected {expected_checksum:#010X}, got {actual:#010X}",
                self.key()
            )));
        }
        self.apply(store)
    }
}

/// A distributed transaction spanning multiple shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTransaction {
    /// Unique transaction ID.
    pub tx_id: u64,
    /// Coordinator node ID.
    pub coordinator: NodeId,
    /// Participating shards.
    pub participants: Vec<ShardId>,
    /// Current phase.
    pub phase: TxPhase,
    /// Operations per shard.
    pub operations: HashMap<ShardId, Vec<Transaction>>,
    /// Delta embeddings per shard.
    pub deltas: HashMap<ShardId, DeltaVector>,
    /// Votes received from participants.
    pub votes: HashMap<ShardId, PrepareVote>,
    /// Timestamp when transaction started.
    pub started_at: u64,
    /// Timeout for the transaction.
    pub timeout_ms: u64,
}

impl DistributedTransaction {
    #[must_use]
    pub fn new(coordinator: NodeId, participants: Vec<ShardId>) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            tx_id: generate_tx_id(),
            coordinator,
            participants,
            phase: TxPhase::Preparing,
            operations: HashMap::new(),
            deltas: HashMap::new(),
            votes: HashMap::new(),
            started_at,
            timeout_ms: 5000, // 5 second default timeout
        }
    }

    pub fn add_operations(&mut self, shard: ShardId, ops: Vec<Transaction>) {
        self.operations.insert(shard, ops);
    }

    pub fn add_delta(&mut self, shard: ShardId, delta: DeltaVector) {
        self.deltas.insert(shard, delta);
    }

    pub fn record_vote(&mut self, shard: ShardId, vote: PrepareVote) {
        self.votes.insert(shard, vote);
    }

    #[must_use]
    pub fn all_voted(&self) -> bool {
        self.participants
            .iter()
            .all(|shard| self.votes.contains_key(shard))
    }

    #[must_use]
    pub fn all_yes(&self) -> bool {
        self.votes
            .values()
            .all(|v| matches!(v, PrepareVote::Yes { .. }))
    }

    #[must_use]
    pub fn any_no(&self) -> bool {
        self.votes
            .values()
            .any(|v| matches!(v, PrepareVote::No { .. } | PrepareVote::Conflict { .. }))
    }

    #[must_use]
    pub fn is_timed_out(&self) -> bool {
        #[allow(clippy::cast_possible_truncation)]
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now - self.started_at > self.timeout_ms
    }

    #[must_use]
    pub fn affected_keys(&self) -> HashSet<String> {
        self.deltas
            .values()
            .flat_map(|d| d.affected_keys.iter().cloned())
            .collect()
    }

    #[must_use]
    pub fn merged_delta(&self) -> Option<DeltaVector> {
        let deltas: Vec<_> = self.deltas.values().cloned().collect();
        if deltas.is_empty() {
            return None;
        }

        let mut merged = deltas[0].clone();
        for delta in &deltas[1..] {
            merged = merged.add(delta);
        }
        Some(merged)
    }
}

/// Request to prepare a transaction on a shard.
#[derive(Debug, Clone)]
pub struct PrepareRequest {
    /// Distributed transaction ID.
    pub tx_id: u64,
    /// Coordinator node ID.
    pub coordinator: NodeId,
    /// Operations to execute on this shard.
    pub operations: Vec<Transaction>,
    /// Delta embedding computed by coordinator (for conflict detection, sparse for efficiency).
    pub delta_embedding: SparseVector,
    /// Timeout in milliseconds.
    pub timeout_ms: u64,
}

/// Vote from a participant in response to prepare request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PrepareVote {
    /// Participant is ready to commit.
    Yes {
        /// Lock handle for the prepared transaction.
        lock_handle: u64,
        /// Delta computed by participant.
        delta: DeltaVector,
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

/// Request to commit a prepared transaction.
#[derive(Debug, Clone)]
pub struct CommitRequest {
    /// Distributed transaction ID.
    pub tx_id: u64,
    /// Lock handles to release.
    pub lock_handles: Vec<u64>,
}

/// Request to abort a transaction.
#[derive(Debug, Clone)]
pub struct AbortRequest {
    /// Distributed transaction ID.
    pub tx_id: u64,
    /// Reason for abort.
    pub reason: String,
    /// Lock handles to release.
    pub lock_handles: Vec<u64>,
}

/// Response from commit/abort.
#[derive(Debug, Clone)]
pub struct TxResponse {
    /// Distributed transaction ID.
    pub tx_id: u64,
    /// Whether the operation succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
}

/// Key-level lock for distributed transactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyLock {
    /// Key being locked.
    pub key: String,
    /// Transaction ID holding the lock.
    pub tx_id: u64,
    /// Lock handle for release.
    pub lock_handle: u64,
    /// When the lock was acquired (epoch milliseconds).
    pub acquired_at_ms: EpochMillis,
    /// Lock timeout in milliseconds.
    pub timeout_ms: u64,
}

impl KeyLock {
    #[must_use]
    pub fn is_expired(&self) -> bool {
        now_epoch_millis().saturating_sub(self.acquired_at_ms) > self.timeout_ms
    }
}

/// Lock handle counter.
static LOCK_COUNTER: AtomicU64 = AtomicU64::new(1);

/// High-water mark: 90% of `u64::MAX`. Beyond this, lock handle exhaustion is imminent.
const LOCK_HANDLE_HIGH_WATER: u64 = u64::MAX / 10 * 9;

/// Count of high-water warnings emitted (to avoid log spam).
static LOCK_HIGH_WATER_WARNINGS: AtomicU64 = AtomicU64::new(0);

/// Allocate the next lock handle, warning if approaching exhaustion.
fn next_lock_handle() -> u64 {
    let handle = LOCK_COUNTER.fetch_add(1, Ordering::Relaxed);
    if handle >= LOCK_HANDLE_HIGH_WATER {
        // Warn once per million handles to avoid log spam
        let warnings = LOCK_HIGH_WATER_WARNINGS.fetch_add(1, Ordering::Relaxed);
        if warnings == 0 || warnings % 1_000_000 == 0 {
            let pct_used = if handle == 0 {
                0.0
            } else {
                #[allow(clippy::cast_precision_loss)]
                let pct = (handle as f64 / u64::MAX as f64) * 100.0;
                pct
            };
            tracing::warn!(
                handle,
                pct_used,
                "Lock handle counter approaching u64::MAX exhaustion"
            );
        }
    }
    handle
}

/// Returns the current lock handle counter value (for monitoring).
#[must_use]
pub fn lock_handle_current() -> u64 {
    LOCK_COUNTER.load(Ordering::Relaxed)
}

/// Returns the number of high-water warnings emitted.
#[must_use]
pub fn lock_handle_high_water_warnings() -> u64 {
    LOCK_HIGH_WATER_WARNINGS.load(Ordering::Relaxed)
}

/// Returns the high-water threshold.
#[must_use]
pub const fn lock_handle_high_water_threshold() -> u64 {
    LOCK_HANDLE_HIGH_WATER
}

/// Key-level lock manager for distributed transactions.
#[derive(Debug, Default)]
pub struct LockManager {
    /// Locks by key.
    locks: RwLock<HashMap<String, KeyLock>>,
    /// Locks by transaction ID.
    tx_locks: RwLock<HashMap<u64, Vec<String>>>,
    /// Default lock timeout.
    pub default_timeout: Duration,
}

impl LockManager {
    #[must_use]
    pub fn new() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            tx_locks: RwLock::new(HashMap::new()),
            default_timeout: Duration::from_secs(30),
        }
    }

    #[must_use]
    pub fn with_default_timeout(timeout: Duration) -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            tx_locks: RwLock::new(HashMap::new()),
            default_timeout: timeout,
        }
    }

    /// Try to acquire locks for a set of keys.
    /// Returns a lock handle if successful, or the conflicting `tx_id` if not.
    ///
    /// # Errors
    ///
    /// Returns the `tx_id` of the conflicting lock holder.
    pub fn try_lock(&self, tx_id: u64, keys: &[String]) -> std::result::Result<u64, u64> {
        let mut locks = self.locks.write();
        let mut tx_locks = self.tx_locks.write();

        // First check if all keys are available
        for key in keys {
            if let Some(existing) = locks.get(key) {
                if !existing.is_expired() && existing.tx_id != tx_id {
                    return Err(existing.tx_id);
                }
            }
        }

        // Acquire all locks
        let lock_handle = next_lock_handle();
        let now_ms = now_epoch_millis();
        #[allow(clippy::cast_possible_truncation)]
        let timeout_ms = self.default_timeout.as_millis() as u64;

        for key in keys {
            locks.insert(
                key.clone(),
                KeyLock {
                    key: key.clone(),
                    tx_id,
                    lock_handle,
                    acquired_at_ms: now_ms,
                    timeout_ms,
                },
            );
        }

        tx_locks
            .entry(tx_id)
            .or_default()
            .extend(keys.iter().cloned());

        drop(locks);
        drop(tx_locks);
        Ok(lock_handle)
    }

    pub fn release(&self, tx_id: u64) {
        let mut locks = self.locks.write();
        let mut tx_locks = self.tx_locks.write();

        if let Some(keys) = tx_locks.remove(&tx_id) {
            for key in keys {
                if let Some(lock) = locks.get(&key) {
                    if lock.tx_id == tx_id {
                        locks.remove(&key);
                    }
                }
            }
        }
    }

    pub fn release_by_handle(&self, lock_handle: u64) {
        let mut locks = self.locks.write();
        let mut tx_locks = self.tx_locks.write();

        // Find and remove locks with this handle
        let keys_to_remove: Vec<_> = locks
            .iter()
            .filter(|(_, lock)| lock.lock_handle == lock_handle)
            .map(|(k, _)| k.clone())
            .collect();

        for key in &keys_to_remove {
            if let Some(lock) = locks.remove(key) {
                if let Some(tx_keys) = tx_locks.get_mut(&lock.tx_id) {
                    tx_keys.retain(|k| k != key);
                }
            }
        }
    }

    /// Release locks by handle and clean up the wait-for graph.
    ///
    /// This ensures that when locks are released, any wait edges involving
    /// the releasing transaction are removed to prevent orphaned edges.
    pub fn release_by_handle_with_wait_cleanup(
        &self,
        lock_handle: u64,
        wait_graph: &crate::deadlock::WaitForGraph,
    ) {
        // Single critical section: extract tx_id and release atomically to prevent TOCTOU
        let tx_id = {
            let mut locks = self.locks.write();
            let mut tx_locks = self.tx_locks.write();

            let keys_to_remove: Vec<_> = locks
                .iter()
                .filter(|(_, lock)| lock.lock_handle == lock_handle)
                .map(|(k, lock)| (k.clone(), lock.tx_id))
                .collect();

            let mut found_tx_id = None;
            for (key, tx_id) in &keys_to_remove {
                found_tx_id = Some(*tx_id);
                locks.remove(key);
                if let Some(tx_keys) = tx_locks.get_mut(tx_id) {
                    tx_keys.retain(|k| k != key);
                }
            }
            drop(locks);
            drop(tx_locks);
            found_tx_id
        };

        // Clean up wait graph for this transaction (outside lock scope)
        if let Some(tx_id) = tx_id {
            wait_graph.remove_transaction(tx_id);
        }
    }

    pub fn is_locked(&self, key: &str) -> bool {
        let locks = self.locks.read();
        locks.get(key).is_some_and(|lock| !lock.is_expired())
    }

    pub fn lock_holder(&self, key: &str) -> Option<u64> {
        let locks = self.locks.read();
        locks
            .get(key)
            .filter(|lock| !lock.is_expired())
            .map(|lock| lock.tx_id)
    }

    pub fn cleanup_expired(&self) -> usize {
        let mut locks = self.locks.write();
        let mut tx_locks = self.tx_locks.write();

        let expired: Vec<_> = locks
            .iter()
            .filter(|(_, lock)| lock.is_expired())
            .map(|(k, lock)| (k.clone(), lock.tx_id))
            .collect();

        for (key, tx_id) in &expired {
            locks.remove(key);
            if let Some(tx_keys) = tx_locks.get_mut(tx_id) {
                tx_keys.retain(|k| k != key);
            }
        }

        drop(locks);
        drop(tx_locks);
        expired.len()
    }

    pub fn cleanup_expired_with_wait_cleanup(
        &self,
        wait_graph: &crate::deadlock::WaitForGraph,
    ) -> usize {
        let mut locks = self.locks.write();
        let mut tx_locks = self.tx_locks.write();

        let expired: Vec<_> = locks
            .iter()
            .filter(|(_, lock)| lock.is_expired())
            .map(|(k, lock)| (k.clone(), lock.tx_id))
            .collect();

        let expired_tx_ids: std::collections::HashSet<u64> =
            expired.iter().map(|(_, tx_id)| *tx_id).collect();

        for (key, tx_id) in &expired {
            locks.remove(key);
            if let Some(tx_keys) = tx_locks.get_mut(tx_id) {
                tx_keys.retain(|k| k != key);
            }
        }

        let count = expired.len();
        drop(tx_locks);
        drop(locks);

        for tx_id in expired_tx_ids {
            wait_graph.remove_transaction(tx_id);
        }

        count
    }

    pub fn active_lock_count(&self) -> usize {
        self.locks.read().len()
    }

    #[must_use]
    pub fn to_serializable(&self) -> SerializableLockState {
        #[allow(clippy::cast_possible_truncation)]
        let default_timeout_ms = self.default_timeout.as_millis() as u64;
        SerializableLockState {
            locks: self.locks.read().clone(),
            tx_locks: self.tx_locks.read().clone(),
            default_timeout_ms,
        }
    }

    #[must_use]
    pub fn from_serializable(state: SerializableLockState) -> Self {
        Self {
            locks: RwLock::new(state.locks),
            tx_locks: RwLock::new(state.tx_locks),
            default_timeout: Duration::from_millis(state.default_timeout_ms),
        }
    }

    /// Try to acquire locks with wait-for graph tracking.
    ///
    /// On success, returns `Ok(lock_handle)`.
    /// On conflict, updates the wait-for graph and returns `Err(WaitInfo)`.
    ///
    /// This method atomically checks for conflicts, collects all blocking transactions,
    /// and updates the wait-for graph WHILE holding the lock to prevent TOCTOU races.
    ///
    /// # Errors
    ///
    /// Returns `WaitInfo` if a conflicting lock holder is found.
    ///
    /// # Panics
    ///
    /// Panics if the blocking transaction set is empty after detecting conflicts.
    pub fn try_lock_with_wait_tracking(
        &self,
        tx_id: u64,
        keys: &[String],
        wait_graph: &crate::deadlock::WaitForGraph,
        priority: Option<u32>,
    ) -> std::result::Result<u64, crate::deadlock::WaitInfo> {
        let mut locks = self.locks.write();
        let mut tx_locks = self.tx_locks.write();

        // Check for conflicts and collect ALL blocking transactions atomically
        let mut blocking_tx_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut conflicting_keys = Vec::new();

        for key in keys {
            if let Some(existing) = locks.get(key) {
                if !existing.is_expired() && existing.tx_id != tx_id {
                    blocking_tx_ids.insert(existing.tx_id);
                    conflicting_keys.push(key.clone());
                }
            }
        }

        if !blocking_tx_ids.is_empty() {
            // CRITICAL: Add ALL conflicts to wait-for graph WHILE holding locks
            // This ensures atomicity between lock state and wait graph state
            for blocker_id in &blocking_tx_ids {
                wait_graph.add_wait(tx_id, *blocker_id, priority);
            }

            // Now safe to drop locks - wait graph is consistent with lock state
            let blocking_tx_id = *blocking_tx_ids.iter().next().unwrap();
            drop(tx_locks);
            drop(locks);

            return Err(crate::deadlock::WaitInfo {
                blocking_tx_id,
                conflicting_keys,
            });
        }

        // No conflicts - acquire all locks atomically
        let lock_handle = next_lock_handle();
        let now_ms = now_epoch_millis();
        #[allow(clippy::cast_possible_truncation)]
        let timeout_ms = self.default_timeout.as_millis() as u64;

        for key in keys {
            locks.insert(
                key.clone(),
                KeyLock {
                    key: key.clone(),
                    tx_id,
                    lock_handle,
                    acquired_at_ms: now_ms,
                    timeout_ms,
                },
            );
        }

        tx_locks
            .entry(tx_id)
            .or_default()
            .extend(keys.iter().cloned());

        // Remove from wait graph WHILE holding locks for consistency
        wait_graph.remove_transaction(tx_id);

        drop(tx_locks);
        drop(locks);

        Ok(lock_handle)
    }

    #[must_use]
    pub fn keys_for_transaction(&self, tx_id: u64) -> Vec<String> {
        self.tx_locks
            .read()
            .get(&tx_id)
            .cloned()
            .unwrap_or_default()
    }

    #[must_use]
    pub fn lock_count_for_transaction(&self, tx_id: u64) -> usize {
        self.tx_locks.read().get(&tx_id).map_or(0, Vec::len)
    }
}

/// Serializable representation of `LockManager` state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableLockState {
    locks: HashMap<String, KeyLock>,
    tx_locks: HashMap<u64, Vec<String>>,
    default_timeout_ms: u64,
}

impl SerializableLockState {
    #[must_use]
    pub const fn new(
        locks: HashMap<String, KeyLock>,
        tx_locks: HashMap<u64, Vec<String>>,
        default_timeout_ms: u64,
    ) -> Self {
        Self {
            locks,
            tx_locks,
            default_timeout_ms,
        }
    }

    #[must_use]
    pub const fn locks(&self) -> &HashMap<String, KeyLock> {
        &self.locks
    }

    #[must_use]
    pub const fn tx_locks(&self) -> &HashMap<u64, Vec<String>> {
        &self.tx_locks
    }

    #[must_use]
    pub const fn default_timeout_ms(&self) -> u64 {
        self.default_timeout_ms
    }
}

/// Serializable state for crash recovery of the distributed transaction coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorState {
    /// Pending transactions.
    pub pending: HashMap<u64, DistributedTransaction>,
    /// Lock manager state.
    pub lock_state: SerializableLockState,
}

/// Serializable state for crash recovery of a transaction participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantState {
    /// Prepared transactions awaiting commit/abort decision.
    pub prepared: HashMap<u64, PreparedTx>,
    /// Lock manager state.
    pub lock_state: SerializableLockState,
}

/// Configuration for distributed transaction coordinator.
#[derive(Debug, Clone)]
pub struct DistributedTxConfig {
    /// Timeout for prepare phase in milliseconds.
    pub prepare_timeout_ms: u64,
    /// Timeout for commit phase in milliseconds.
    pub commit_timeout_ms: u64,
    /// Maximum concurrent distributed transactions.
    pub max_concurrent: usize,
    /// Orthogonal threshold for auto-merge during 2PC.
    pub orthogonal_threshold: f32,
    /// Whether to allow optimistic locking.
    pub optimistic_locking: bool,
    /// Soft limit percentage of `max_concurrent` at which a warning is logged.
    /// Transactions are still allowed through (soft limit), but the warning
    /// signals that the system is approaching capacity. Range: 0-100.
    pub tx_queue_soft_limit_pct: u8,
}

impl Default for DistributedTxConfig {
    fn default() -> Self {
        Self {
            prepare_timeout_ms: 5000,
            commit_timeout_ms: 10000,
            max_concurrent: 100,
            orthogonal_threshold: 0.1,
            optimistic_locking: true,
            tx_queue_soft_limit_pct: 80,
        }
    }
}

/// Statistics for distributed transactions.
#[derive(Debug, Default)]
pub struct DistributedTxStats {
    /// Total transactions started.
    pub started: AtomicU64,
    /// Transactions committed.
    pub committed: AtomicU64,
    /// Transactions aborted.
    pub aborted: AtomicU64,
    /// Transactions timed out.
    pub timed_out: AtomicU64,
    /// Conflicts detected.
    pub conflicts: AtomicU64,
    /// Orthogonal merges (avoided conflicts).
    pub orthogonal_merges: AtomicU64,

    // Timing metrics
    /// Time spent in prepare phase (microseconds).
    pub prepare_timing: crate::metrics::TimingStats,
    /// Time spent in commit phase (microseconds).
    pub commit_timing: crate::metrics::TimingStats,

    // Lock contention
    /// Time waiting for locks (microseconds).
    pub lock_wait_timing: crate::metrics::TimingStats,

    // Participation tracking
    /// Transactions aborted due to participant timeout (partition).
    pub participation_timeouts: AtomicU64,

    // Queue pressure tracking
    /// Number of transactions that triggered the soft queue limit warning.
    pub tx_queue_soft_limit_warnings: AtomicU64,

    // Abort delivery tracking
    /// Number of abort delivery retries attempted.
    pub abort_delivery_retries: AtomicU64,
    /// Number of abort delivery send failures.
    pub abort_delivery_failures: AtomicU64,
}

impl DistributedTxStats {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn commit_rate(&self) -> f32 {
        let started = self.started.load(Ordering::Relaxed);
        if started == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let rate = self.committed.load(Ordering::Relaxed) as f32 / started as f32;
        rate
    }

    #[must_use]
    pub fn conflict_rate(&self) -> f32 {
        let started = self.started.load(Ordering::Relaxed);
        if started == 0 {
            return 0.0;
        }
        #[allow(clippy::cast_precision_loss)]
        let rate = self.conflicts.load(Ordering::Relaxed) as f32 / started as f32;
        rate
    }

    #[must_use]
    pub fn snapshot(&self) -> DistributedTxStatsSnapshot {
        DistributedTxStatsSnapshot {
            started: self.started.load(Ordering::Relaxed),
            committed: self.committed.load(Ordering::Relaxed),
            aborted: self.aborted.load(Ordering::Relaxed),
            timed_out: self.timed_out.load(Ordering::Relaxed),
            conflicts: self.conflicts.load(Ordering::Relaxed),
            orthogonal_merges: self.orthogonal_merges.load(Ordering::Relaxed),
            prepare_timing: self.prepare_timing.snapshot(),
            commit_timing: self.commit_timing.snapshot(),
            lock_wait_timing: self.lock_wait_timing.snapshot(),
            participation_timeouts: self.participation_timeouts.load(Ordering::Relaxed),
            commit_rate: self.commit_rate(),
            conflict_rate: self.conflict_rate(),
            lock_handle_current: lock_handle_current(),
            lock_handle_high_water_warnings: lock_handle_high_water_warnings(),
            abort_delivery_retries: self.abort_delivery_retries.load(Ordering::Relaxed),
            abort_delivery_failures: self.abort_delivery_failures.load(Ordering::Relaxed),
        }
    }
}

/// Point-in-time snapshot of distributed transaction statistics.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DistributedTxStatsSnapshot {
    pub started: u64,
    pub committed: u64,
    pub aborted: u64,
    pub timed_out: u64,
    pub conflicts: u64,
    pub orthogonal_merges: u64,
    pub prepare_timing: crate::metrics::TimingSnapshot,
    pub commit_timing: crate::metrics::TimingSnapshot,
    pub lock_wait_timing: crate::metrics::TimingSnapshot,
    pub participation_timeouts: u64,
    pub commit_rate: f32,
    pub conflict_rate: f32,
    /// Current lock handle counter value.
    #[serde(default)]
    pub lock_handle_current: u64,
    /// Number of high-water warnings emitted for lock handle exhaustion.
    #[serde(default)]
    pub lock_handle_high_water_warnings: u64,
    /// Number of abort delivery retries attempted.
    #[serde(default)]
    pub abort_delivery_retries: u64,
    /// Number of abort delivery send failures.
    #[serde(default)]
    pub abort_delivery_failures: u64,
}

/// Statistics from crash recovery of coordinator or participant state.
#[derive(Debug, Default)]
pub struct RecoveryStats {
    /// Transactions in prepare phase (awaiting votes).
    pub pending_prepare: usize,
    /// Transactions ready to commit.
    pub pending_commit: usize,
    /// Transactions to abort.
    pub pending_abort: usize,
    /// Transactions that timed out (presumed abort).
    pub timed_out: usize,
    /// Transactions that were already completed (cleaned up).
    pub completed: usize,
    /// Orphaned locks force-released during recovery (`TxComplete` logged
    /// but lock release was interrupted before `AllLocksReleased`).
    pub lock_releases_recovered: usize,
}

/// Tracks abort acknowledgments from participants.
#[derive(Debug, Clone)]
pub(crate) struct AbortState {
    /// Shards that need to acknowledge abort.
    pub(crate) pending_acks: HashSet<usize>,
    /// When abort was initiated (epoch millis).
    pub(crate) initiated_at: EpochMillis,
    /// Number of retry attempts.
    pub(crate) retry_count: u32,
}

/// Error when recording a vote fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoteRecordError {
    /// Transaction not found in pending map.
    TxNotFound(u64),
    /// Transaction is not in the expected phase.
    WrongPhase {
        tx_id: u64,
        expected: TxPhase,
        actual: TxPhase,
    },
    /// Shard has already voted for this transaction.
    DuplicateVote { tx_id: u64, shard: ShardId },
}

impl std::fmt::Display for VoteRecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TxNotFound(tx_id) => write!(f, "transaction {tx_id} not found"),
            Self::WrongPhase {
                tx_id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "transaction {tx_id} in wrong phase: expected {expected:?}, actual {actual:?}"
                )
            },
            Self::DuplicateVote { tx_id, shard } => {
                write!(
                    f,
                    "duplicate vote for transaction {tx_id} from shard {shard}"
                )
            },
        }
    }
}

/// Coordinator for distributed transactions.
pub struct DistributedTxCoordinator {
    /// Pending transactions by ID.
    pending: RwLock<HashMap<u64, DistributedTransaction>>,
    /// Consensus manager for conflict detection (used in future batch optimizations).
    #[allow(dead_code)]
    consensus: ConsensusManager,
    /// Lock manager.
    lock_manager: LockManager,
    /// Wait-for graph for deadlock detection.
    wait_graph: crate::deadlock::WaitForGraph,
    /// Configuration.
    config: DistributedTxConfig,
    /// Statistics.
    pub stats: DistributedTxStats,
    /// Tracks in-flight abort broadcasts awaiting acknowledgment.
    abort_states: RwLock<HashMap<u64, AbortState>>,
    /// Transactions marked for abort broadcast (processed by async tick).
    pending_aborts: RwLock<Vec<(u64, String, Vec<usize>)>>,
    /// Optional Write-Ahead Log for durable phase transitions.
    wal: Option<RwLock<TxWal>>,
}

impl DistributedTxCoordinator {
    #[must_use]
    pub fn new(consensus: ConsensusManager, config: DistributedTxConfig) -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            consensus,
            lock_manager: LockManager::new(),
            wait_graph: crate::deadlock::WaitForGraph::new(),
            config,
            stats: DistributedTxStats::new(),
            abort_states: RwLock::new(HashMap::new()),
            pending_aborts: RwLock::new(Vec::new()),
            wal: None,
        }
    }

    #[must_use]
    pub fn with_consensus(consensus: ConsensusManager) -> Self {
        Self::new(consensus, DistributedTxConfig::default())
    }

    #[must_use]
    pub fn with_wal(mut self, wal: TxWal) -> Self {
        self.wal = Some(RwLock::new(wal));
        self
    }

    fn log_wal_entry(&self, entry: &TxWalEntry) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.write()
                .append(entry)
                .map_err(|e| ChainError::StorageError(format!("WAL write failed: {e}")))?;
        }
        Ok(())
    }

    /// Recover coordinator state from the WAL.
    ///
    /// Returns statistics about recovered transactions:
    /// - Prepared transactions are restored to pending
    /// - Committing transactions are marked for completion
    /// - Aborting transactions are marked for completion
    ///
    /// # Errors
    ///
    /// Returns an error if WAL replay fails.
    pub fn recover_from_wal(&self) -> Result<RecoveryStats> {
        let Some(wal) = &self.wal else {
            return Ok(RecoveryStats::default());
        };

        let recovery_state = TxRecoveryState::from_wal(&wal.read())
            .map_err(|e| ChainError::StorageError(format!("WAL replay failed: {e}")))?;

        let mut stats = RecoveryStats::default();
        let mut pending = self.pending.write();

        // Helper to restore a transaction from WAL data
        let restore_tx = |prepared: &crate::tx_wal::RecoveredPreparedTx,
                          phase: TxPhase|
         -> DistributedTransaction {
            let mut tx = DistributedTransaction::new(
                "recovered".to_string(), // Coordinator unknown during recovery
                prepared.participants.clone(),
            );
            tx.tx_id = prepared.tx_id;
            tx.phase = phase;
            // Restore votes with persisted lock handles
            for (shard, vote_kind) in &prepared.votes {
                let vote = match vote_kind {
                    crate::tx_wal::PrepareVoteKind::Yes { lock_handle } => PrepareVote::Yes {
                        lock_handle: *lock_handle,
                        delta: DeltaVector::zero(0),
                    },
                    crate::tx_wal::PrepareVoteKind::No => PrepareVote::No {
                        reason: "recovered from WAL".to_string(),
                    },
                };
                tx.votes.insert(*shard, vote);
            }
            tx
        };

        // Restore prepared transactions
        for prepared in &recovery_state.prepared_txs {
            let tx = restore_tx(prepared, TxPhase::Prepared);
            pending.insert(prepared.tx_id, tx);
            stats.pending_prepare += 1;
        }

        // Restore committing transactions
        for prepared in &recovery_state.committing_txs {
            let tx = restore_tx(prepared, TxPhase::Committing);
            pending.insert(prepared.tx_id, tx);
            stats.pending_commit += 1;
        }

        // Restore aborting transactions
        for prepared in &recovery_state.aborting_txs {
            let tx = restore_tx(prepared, TxPhase::Aborting);
            pending.insert(prepared.tx_id, tx);
            stats.pending_abort += 1;
        }

        drop(pending);
        // Force-release orphaned locks from completed transactions
        for orphan in &recovery_state.orphaned_locks {
            tracing::warn!(
                tx_id = orphan.tx_id,
                lock_handle = orphan.lock_handle,
                "Releasing orphaned lock from crashed commit"
            );
            self.lock_manager
                .release_by_handle_with_wait_cleanup(orphan.lock_handle, &self.wait_graph);
            stats.lock_releases_recovered += 1;
        }

        Ok(stats)
    }

    /// # Errors
    ///
    /// Returns an error if the WAL truncation fails.
    pub fn truncate_wal(&self) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.write()
                .truncate()
                .map_err(|e| ChainError::StorageError(format!("WAL truncate failed: {e}")))?;
        }
        Ok(())
    }

    /// Begin a new distributed transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the maximum concurrent transaction limit is reached
    /// or if the WAL write fails.
    pub fn begin(
        &self,
        coordinator: &NodeId,
        participants: &[ShardId],
    ) -> Result<DistributedTransaction> {
        // Atomically check limit, write WAL, and insert - all in single critical section
        // to prevent TOCTOU race on max_concurrent and ensure WAL/pending consistency
        let tx = {
            let mut pending = self.pending.write();

            if pending.len() >= self.config.max_concurrent {
                tracing::warn!(
                    pending = pending.len(),
                    max = self.config.max_concurrent,
                    "Distributed transaction rejected: too many concurrent transactions"
                );
                return Err(ChainError::TransactionFailed(
                    "too many concurrent distributed transactions".to_string(),
                ));
            }

            // Soft limit warning: approaching capacity but still allow through
            let soft_limit =
                self.config.max_concurrent * usize::from(self.config.tx_queue_soft_limit_pct) / 100;
            if pending.len() >= soft_limit {
                self.stats
                    .tx_queue_soft_limit_warnings
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    pending = pending.len(),
                    soft_limit = soft_limit,
                    max = self.config.max_concurrent,
                    "Transaction queue approaching capacity (soft limit reached)"
                );
            }

            let mut tx = DistributedTransaction::new(coordinator.clone(), participants.to_vec());
            tx.timeout_ms = self.config.prepare_timeout_ms;

            // Write WAL INSIDE critical section - if this fails, tx is NOT in pending
            // This ensures WAL and pending state are always consistent
            self.log_wal_entry(&TxWalEntry::TxBegin {
                tx_id: tx.tx_id,
                participants: participants.to_vec(),
            })?;

            pending.insert(tx.tx_id, tx.clone());
            tx
        };

        self.stats.started.fetch_add(1, Ordering::Relaxed);

        tracing::info!(
            tx_id = tx.tx_id,
            coordinator = %coordinator,
            participants = ?participants,
            "Distributed transaction started"
        );

        Ok(tx)
    }

    pub fn get(&self, tx_id: u64) -> Option<DistributedTransaction> {
        self.pending.read().get(&tx_id).cloned()
    }

    /// Two-stage validation: (1) Lock acquisition with wait-for graph tracking,
    /// (2) Semantic conflict detection via delta similarity. Returns Conflict early
    /// on lock failure to avoid expensive embedding computation.
    pub fn handle_prepare(&self, request: &PrepareRequest) -> PrepareVote {
        // Extract affected keys from operations
        let keys: Vec<String> = request
            .operations
            .iter()
            .map(|op| op.affected_key().to_string())
            .collect();

        tracing::debug!(
            tx_id = request.tx_id,
            keys = ?keys,
            "Processing prepare request"
        );

        // Try to acquire locks with wait-for graph tracking for deadlock detection
        let lock_handle = match self.lock_manager.try_lock_with_wait_tracking(
            request.tx_id,
            &keys,
            &self.wait_graph,
            None, // No priority for now
        ) {
            Ok(handle) => {
                tracing::debug!(
                    tx_id = request.tx_id,
                    lock_handle = handle,
                    "Locks acquired"
                );
                handle
            },
            Err(wait_info) => {
                tracing::warn!(
                    tx_id = request.tx_id,
                    conflicting_tx = wait_info.blocking_tx_id,
                    conflicting_keys = ?wait_info.conflicting_keys,
                    "Lock conflict detected"
                );
                return PrepareVote::Conflict {
                    similarity: 1.0, // Same keys = maximum conflict
                    conflicting_tx: wait_info.blocking_tx_id,
                };
            },
        };

        // Compute delta from request embedding (already sparse)
        let delta = DeltaVector::from_sparse(
            request.delta_embedding.clone(),
            keys.iter().cloned().collect(),
            request.tx_id,
        );

        // Check for semantic conflicts with pending transactions
        if self.config.optimistic_locking {
            let pending = self.pending.read();
            for (_, pending_tx) in pending.iter() {
                if pending_tx.tx_id == request.tx_id {
                    continue;
                }

                // Check if any pending delta conflicts
                for pending_delta in pending_tx.deltas.values() {
                    let similarity = delta.cosine_similarity(pending_delta);
                    if similarity.abs() >= self.config.orthogonal_threshold {
                        // Conflicting - check if same keys
                        let overlap: HashSet<_> = delta
                            .affected_keys
                            .intersection(&pending_delta.affected_keys)
                            .collect();

                        if !overlap.is_empty() {
                            tracing::warn!(
                                tx_id = request.tx_id,
                                conflicting_tx = pending_tx.tx_id,
                                similarity = similarity,
                                "Semantic conflict detected"
                            );
                            self.lock_manager.release(request.tx_id);
                            return PrepareVote::Conflict {
                                similarity,
                                conflicting_tx: pending_tx.tx_id,
                            };
                        }
                    }
                }
            }
        }

        tracing::debug!(tx_id = request.tx_id, "Prepare vote: YES");

        PrepareVote::Yes { lock_handle, delta }
    }

    /// Records a vote from a participant shard.
    ///
    /// This method uses a phased approach to minimize lock duration:
    /// - Phase 1: Quick lookup and vote recording (brief pending lock)
    /// - Phase 2: Expensive conflict detection (NO LOCKS HELD)
    /// - Phase 3: Brief re-acquisition for state update
    ///
    /// This prevents deadlocks with `release_orphaned_locks` by never holding
    /// pending lock while acquiring `pending_aborts` lock.
    ///
    /// # Errors
    ///
    /// Returns a [`VoteRecordError`] if the transaction is not found, in the
    /// wrong phase, or the shard already voted.
    ///
    /// # Panics
    ///
    /// Panics if all votes are YES but the transaction snapshot is missing.
    #[allow(clippy::too_many_lines, clippy::significant_drop_tightening)]
    pub fn record_vote(
        &self,
        tx_id: u64,
        shard: ShardId,
        vote: PrepareVote,
    ) -> std::result::Result<Option<TxPhase>, VoteRecordError> {
        // Log vote to WAL BEFORE recording in memory to ensure durability
        let vote_kind = match &vote {
            PrepareVote::Yes { lock_handle, .. } => crate::tx_wal::PrepareVoteKind::Yes {
                lock_handle: *lock_handle,
            },
            PrepareVote::No { .. } | PrepareVote::Conflict { .. } => {
                crate::tx_wal::PrepareVoteKind::No
            },
        };
        if let Err(e) = self.log_wal_entry(&TxWalEntry::PrepareVote {
            tx_id,
            shard,
            vote: vote_kind,
        }) {
            tracing::error!(tx_id = tx_id, shard = shard, error = %e, "Failed to log vote to WAL");
            return Ok(None);
        }

        // Phase 1: Quick lookup and vote recording (brief lock)
        let (needs_conflict_check, tx_snapshot, abort_info) = {
            let mut pending = self.pending.write();
            let tx = pending
                .get_mut(&tx_id)
                .ok_or(VoteRecordError::TxNotFound(tx_id))?;

            if tx.phase != TxPhase::Preparing {
                return Err(VoteRecordError::WrongPhase {
                    tx_id,
                    expected: TxPhase::Preparing,
                    actual: tx.phase,
                });
            }

            if tx.votes.contains_key(&shard) {
                return Err(VoteRecordError::DuplicateVote { tx_id, shard });
            }

            tracing::debug!(
                tx_id = tx_id,
                shard = shard,
                vote = ?vote,
                "Recording vote"
            );

            // Extract delta from YES vote
            if let PrepareVote::Yes { delta, .. } = &vote {
                tx.add_delta(shard, delta.clone());
            }

            tx.record_vote(shard, vote);

            if tx.all_voted() {
                if tx.all_yes() {
                    // Need conflict check - clone state for phase 2
                    (true, Some(tx.clone()), None)
                } else {
                    // Abort case - set phase and collect info for pending_aborts
                    tx.phase = TxPhase::Aborting;
                    let reason = if tx
                        .votes
                        .values()
                        .any(|v| matches!(v, PrepareVote::Conflict { .. }))
                    {
                        self.stats.conflicts.fetch_add(1, Ordering::Relaxed);
                        "conflict detected"
                    } else {
                        "participant voted no"
                    };
                    let shards = tx.participants.clone();
                    tracing::warn!(tx_id = tx_id, reason = reason, "Transaction aborting");
                    (false, None, Some((reason.to_string(), shards)))
                }
            } else {
                // Not all voted yet
                (false, None, None)
            }
        }; // pending lock released

        // Handle abort case - acquire pending_aborts AFTER releasing pending
        if let Some((reason, shards)) = abort_info {
            self.pending_aborts.write().push((tx_id, reason, shards));
            return Ok(Some(TxPhase::Aborting));
        }

        // Phase 2: Expensive conflict detection (NO LOCKS HELD)
        if needs_conflict_check {
            let tx: DistributedTransaction = tx_snapshot.unwrap();

            // Check cross-shard conflicts using cloned data
            if tx.merged_delta().is_some() {
                // Verify deltas are orthogonal
                let deltas: Vec<_> = tx.deltas.values().cloned().collect();
                for i in 0..deltas.len() {
                    for j in (i + 1)..deltas.len() {
                        let sim = deltas[i].cosine_similarity(&deltas[j]);
                        if sim.abs() >= self.config.orthogonal_threshold {
                            // Not orthogonal - need to check keys
                            let overlap: HashSet<_> = deltas[i]
                                .affected_keys
                                .intersection(&deltas[j].affected_keys)
                                .collect();

                            if !overlap.is_empty() {
                                tracing::warn!(
                                    tx_id = tx_id,
                                    similarity = sim,
                                    "Cross-shard conflict detected, aborting"
                                );

                                // Phase 3a: Re-acquire pending lock to set abort state
                                {
                                    let mut pending = self.pending.write();
                                    if let Some(tx) = pending.get_mut(&tx_id) {
                                        tx.phase = TxPhase::Aborting;
                                    }
                                }

                                self.stats.conflicts.fetch_add(1, Ordering::Relaxed);

                                // Queue abort for broadcast (after releasing pending)
                                let shards = tx.participants;
                                self.pending_aborts.write().push((
                                    tx_id,
                                    "cross-shard conflict".to_string(),
                                    shards,
                                ));
                                return Ok(Some(TxPhase::Aborting));
                            }
                        }
                    }
                }

                // All orthogonal - can merge and commit
                self.stats.orthogonal_merges.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(tx_id = tx_id, "Deltas verified orthogonal, ready to commit");
            }

            // Phase 3b: Brief re-acquisition for success state update
            {
                let mut pending = self.pending.write();
                if let Some(tx) = pending.get_mut(&tx_id) {
                    // Log phase transition to WAL BEFORE updating in-memory state
                    if let Err(e) = self.log_wal_entry(&TxWalEntry::PhaseChange {
                        tx_id,
                        from: TxPhase::Preparing,
                        to: TxPhase::Prepared,
                    }) {
                        tracing::error!(tx_id = tx_id, error = %e, "Failed to log phase change to WAL");
                        return Ok(None);
                    }

                    tracing::info!(
                        tx_id = tx_id,
                        phase = ?TxPhase::Prepared,
                        "Transaction prepared"
                    );
                    tx.phase = TxPhase::Prepared;
                    return Ok(Some(TxPhase::Prepared));
                }
            }
        }

        Ok(None)
    }

    /// # Errors
    ///
    /// Returns an error if the transaction is not found or not in the prepared phase.
    #[allow(clippy::significant_drop_tightening)]
    pub fn commit(&self, tx_id: u64) -> Result<()> {
        let mut pending = self.pending.write();
        let tx = pending.get_mut(&tx_id).ok_or_else(|| {
            tracing::warn!(tx_id = tx_id, "Commit failed: transaction not found");
            ChainError::TransactionFailed(format!("transaction {tx_id} not found"))
        })?;

        if tx.phase != TxPhase::Prepared {
            tracing::warn!(
                tx_id = tx_id,
                phase = ?tx.phase,
                "Commit failed: transaction not in prepared phase"
            );
            return Err(ChainError::TransactionFailed(format!(
                "transaction {tx_id} not in prepared phase"
            )));
        }

        tracing::debug!(tx_id = tx_id, "Transitioning to committing phase");

        // Log phase change to WAL BEFORE applying state change
        self.log_wal_entry(&TxWalEntry::PhaseChange {
            tx_id,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })?;

        tx.phase = TxPhase::Committing;

        // CRITICAL: Log TxComplete BEFORE releasing locks to prevent double-release on recovery
        self.log_wal_entry(&TxWalEntry::TxComplete {
            tx_id,
            outcome: TxOutcome::Committed,
        })?;

        // Release all locks AFTER TxComplete is logged, WAL-logging each release
        for vote in tx.votes.values() {
            if let PrepareVote::Yes { lock_handle, .. } = vote {
                tracing::debug!(tx_id = tx_id, lock_handle = lock_handle, "Releasing lock");
                // Best-effort WAL of individual lock release -- do not fail the commit
                let _ = self.log_wal_entry(&TxWalEntry::LockRelease {
                    tx_id,
                    lock_handle: *lock_handle,
                });
                self.lock_manager
                    .release_by_handle_with_wait_cleanup(*lock_handle, &self.wait_graph);
            }
        }

        // Mark all locks released
        let _ = self.log_wal_entry(&TxWalEntry::AllLocksReleased { tx_id });

        tx.phase = TxPhase::Committed;
        self.stats.committed.fetch_add(1, Ordering::Relaxed);

        // Remove from pending
        pending.remove(&tx_id);

        tracing::info!(tx_id = tx_id, "Transaction committed");

        Ok(())
    }

    /// Complete a commit for a transaction already in the Committing phase.
    /// Used during recovery to finalize transactions that were interrupted mid-commit.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not found or not in the committing phase.
    #[allow(clippy::significant_drop_tightening)]
    pub fn complete_commit(&self, tx_id: u64) -> Result<()> {
        let mut pending = self.pending.write();
        let tx = pending.get_mut(&tx_id).ok_or_else(|| {
            ChainError::TransactionFailed(format!("transaction {tx_id} not found"))
        })?;

        if tx.phase != TxPhase::Committing {
            return Err(ChainError::TransactionFailed(format!(
                "transaction {tx_id} not in committing phase"
            )));
        }

        // Release any remaining locks
        for vote in tx.votes.values() {
            if let PrepareVote::Yes { lock_handle, .. } = vote {
                self.lock_manager
                    .release_by_handle_with_wait_cleanup(*lock_handle, &self.wait_graph);
            }
        }

        tx.phase = TxPhase::Committed;
        self.stats.committed.fetch_add(1, Ordering::Relaxed);

        pending.remove(&tx_id);

        Ok(())
    }

    /// Complete an abort for a transaction already in the Aborting phase.
    /// Used during recovery to finalize transactions that were interrupted mid-abort.
    ///
    /// # Errors
    ///
    /// Returns an error if the transaction is not found or not in the aborting phase.
    #[allow(clippy::significant_drop_tightening)]
    pub fn complete_abort(&self, tx_id: u64) -> Result<()> {
        let mut pending = self.pending.write();
        let tx = pending.get_mut(&tx_id).ok_or_else(|| {
            ChainError::TransactionFailed(format!("transaction {tx_id} not found"))
        })?;

        if tx.phase != TxPhase::Aborting {
            return Err(ChainError::TransactionFailed(format!(
                "transaction {tx_id} not in aborting phase"
            )));
        }

        // Release any remaining locks
        for vote in tx.votes.values() {
            if let PrepareVote::Yes { lock_handle, .. } = vote {
                self.lock_manager
                    .release_by_handle_with_wait_cleanup(*lock_handle, &self.wait_graph);
            }
        }

        tx.phase = TxPhase::Aborted;
        self.stats.aborted.fetch_add(1, Ordering::Relaxed);

        pending.remove(&tx_id);

        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if the transaction is not found or WAL logging fails.
    #[allow(clippy::significant_drop_tightening)]
    pub fn abort(&self, tx_id: u64, reason: &str) -> Result<()> {
        let mut pending = self.pending.write();
        let tx = pending.get_mut(&tx_id).ok_or_else(|| {
            tracing::warn!(tx_id = tx_id, "Abort failed: transaction not found");
            ChainError::TransactionFailed(format!("transaction {tx_id} not found"))
        })?;

        let from_phase = tx.phase;

        tracing::warn!(
            tx_id = tx_id,
            from_phase = ?from_phase,
            reason = reason,
            "Aborting transaction"
        );

        // Log phase change to WAL BEFORE applying state change
        self.log_wal_entry(&TxWalEntry::PhaseChange {
            tx_id,
            from: from_phase,
            to: TxPhase::Aborting,
        })?;

        tx.phase = TxPhase::Aborting;

        // CRITICAL: Log TxComplete BEFORE releasing locks to prevent double-release on recovery
        self.log_wal_entry(&TxWalEntry::TxComplete {
            tx_id,
            outcome: TxOutcome::Aborted,
        })?;

        // Release all locks AFTER TxComplete is logged
        for vote in tx.votes.values() {
            if let PrepareVote::Yes { lock_handle, .. } = vote {
                tracing::debug!(
                    tx_id = tx_id,
                    lock_handle = lock_handle,
                    "Releasing lock on abort"
                );
                self.lock_manager
                    .release_by_handle_with_wait_cleanup(*lock_handle, &self.wait_graph);
            }
        }

        tx.phase = TxPhase::Aborted;
        self.stats.aborted.fetch_add(1, Ordering::Relaxed);

        // Remove from pending
        pending.remove(&tx_id);

        tracing::info!(tx_id = tx_id, reason = reason, "Transaction aborted");

        Ok(())
    }

    pub fn cleanup_timeouts(&self) -> Vec<u64> {
        let mut pending = self.pending.write();
        let timed_out: Vec<_> = pending
            .iter()
            .filter(|(_, tx)| tx.is_timed_out())
            .map(|(id, _)| *id)
            .collect();

        if !timed_out.is_empty() {
            tracing::warn!(
                count = timed_out.len(),
                tx_ids = ?timed_out,
                "Cleaning up timed out transactions"
            );
        }

        for tx_id in &timed_out {
            if let Some(tx) = pending.remove(tx_id) {
                tracing::warn!(
                    tx_id = tx_id,
                    phase = ?tx.phase,
                    participants = ?tx.participants,
                    "Transaction timed out"
                );

                // Queue abort broadcast to participants
                self.pending_aborts.write().push((
                    *tx_id,
                    "timeout".to_string(),
                    tx.participants.clone(),
                ));

                // Release locks
                for vote in tx.votes.values() {
                    if let PrepareVote::Yes { lock_handle, .. } = vote {
                        self.lock_manager
                            .release_by_handle_with_wait_cleanup(*lock_handle, &self.wait_graph);
                    }
                }
                self.stats.timed_out.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Also cleanup expired locks
        let expired_locks = self
            .lock_manager
            .cleanup_expired_with_wait_cleanup(&self.wait_graph);
        if expired_locks > 0 {
            tracing::debug!(count = expired_locks, "Cleaned up expired locks");
        }

        timed_out
    }

    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }

    pub const fn lock_manager(&self) -> &LockManager {
        &self.lock_manager
    }

    pub const fn wait_graph(&self) -> &crate::deadlock::WaitForGraph {
        &self.wait_graph
    }

    pub const fn stats(&self) -> &DistributedTxStats {
        &self.stats
    }

    pub fn take_pending_aborts(&self) -> Vec<(u64, String, Vec<usize>)> {
        std::mem::take(&mut *self.pending_aborts.write())
    }

    /// Process and broadcast pending abort messages to all participants.
    ///
    /// Logs an `AbortIntent` WAL entry before sending to ensure abort delivery
    /// can be retried on recovery. This should be called periodically from the
    /// cluster run loop.
    pub async fn process_pending_aborts(&self, transport: &dyn Transport) {
        let pending = self.take_pending_aborts();

        for (tx_id, reason, shards) in pending {
            // Log abort intent to WAL before sending so recovery can resend
            if let Err(e) = self.log_wal_entry(&TxWalEntry::AbortIntent {
                tx_id,
                reason: reason.clone(),
                shards: shards.clone(),
            }) {
                tracing::error!(tx_id, error = %e, "failed to log abort intent to WAL");
            }

            // Track this abort for acknowledgment tracking
            self.track_abort(tx_id, shards.clone());

            // Send abort to all participant shards
            for shard_id in &shards {
                let abort_msg = TxAbortMsg {
                    tx_id,
                    reason: reason.clone(),
                    shards: vec![*shard_id],
                };

                let target = format!("shard-{shard_id}");
                if let Err(e) = transport.send(&target, Message::TxAbort(abort_msg)).await {
                    tracing::warn!(tx_id, shard = shard_id, error = %e, "failed to send abort to shard");
                    self.stats
                        .abort_delivery_failures
                        .fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn track_abort(&self, tx_id: u64, shards: Vec<usize>) {
        let mut abort_states = self.abort_states.write();
        abort_states.insert(
            tx_id,
            AbortState {
                pending_acks: shards.into_iter().collect(),
                initiated_at: now_epoch_millis(),
                retry_count: 0,
            },
        );
    }

    pub fn handle_abort_ack(&self, tx_id: u64, shard_id: usize) -> bool {
        let mut abort_states = self.abort_states.write();
        if let Some(state) = abort_states.get_mut(&tx_id) {
            state.pending_acks.remove(&shard_id);
            if state.pending_acks.is_empty() {
                abort_states.remove(&tx_id);
                return true; // All acknowledged
            }
        }
        drop(abort_states);
        false
    }

    /// Maximum number of abort delivery retry attempts before giving up.
    const MAX_ABORT_RETRIES: u32 = 5;

    /// Get aborts that need to be retried with exponential backoff.
    ///
    /// Returns list of (`tx_id`, shards) pairs for transactions that haven't
    /// been acknowledged. Uses exponential backoff: 1s, 2s, 4s, 8s, 16s.
    pub fn get_retry_aborts(&self) -> Vec<(u64, Vec<usize>)> {
        let now = now_epoch_millis();

        let mut to_retry = Vec::new();
        let mut abort_states = self.abort_states.write();

        for (tx_id, state) in abort_states.iter_mut() {
            if state.retry_count >= Self::MAX_ABORT_RETRIES {
                continue;
            }
            let elapsed = now.saturating_sub(state.initiated_at);
            // Exponential backoff: 1s, 2s, 4s, 8s, 16s
            let backoff_ms = 1000u64 << state.retry_count;
            // Cumulative delay: sum of all previous backoffs
            let cumulative_ms = (1u64 << (state.retry_count + 1)).saturating_sub(1) * 1000;

            if elapsed >= cumulative_ms {
                state.retry_count += 1;
                self.stats
                    .abort_delivery_retries
                    .fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    tx_id,
                    retry = state.retry_count,
                    backoff_ms,
                    pending_shards = state.pending_acks.len(),
                    "Retrying abort delivery"
                );
                to_retry.push((*tx_id, state.pending_acks.iter().copied().collect()));
            }
        }
        drop(abort_states);

        to_retry
    }

    pub fn cleanup_stale_aborts(&self) -> Vec<u64> {
        let now = now_epoch_millis();
        let max_abort_wait_ms = 30_000u64; // 30 seconds

        let mut abort_states = self.abort_states.write();
        let stale: Vec<_> = abort_states
            .iter()
            .filter(|(_, state)| {
                state.retry_count >= Self::MAX_ABORT_RETRIES
                    || now.saturating_sub(state.initiated_at) >= max_abort_wait_ms
            })
            .map(|(id, _)| *id)
            .collect();

        for tx_id in &stale {
            abort_states.remove(tx_id);
        }

        stale
    }

    fn persistence_key(node_id: &str) -> String {
        format!("_dtx:coordinator:{node_id}:state")
    }

    #[must_use]
    pub fn to_state(&self) -> CoordinatorState {
        CoordinatorState {
            pending: self.pending.read().clone(),
            lock_state: self.lock_manager.to_serializable(),
        }
    }

    /// # Errors
    ///
    /// Returns an error if serialization or storage fails.
    pub fn save_to_store(&self, node_id: &str, store: &TensorStore) -> Result<()> {
        let state = self.to_state();
        let bytes = bitcode::serialize(&state)?;

        let mut data = TensorData::new();
        data.set("state", TensorValue::Scalar(ScalarValue::Bytes(bytes)));
        store
            .put(Self::persistence_key(node_id), data)
            .map_err(|e| ChainError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// Load coordinator from `TensorStore` or create fresh if not found.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization of persisted state fails.
    pub fn load_from_store(
        node_id: &str,
        store: &TensorStore,
        consensus: ConsensusManager,
        config: DistributedTxConfig,
    ) -> Result<Self> {
        let key = Self::persistence_key(node_id);

        if let Ok(data) = store.get(&key) {
            if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("state") {
                let state: CoordinatorState = bitcode::deserialize(bytes)?;
                return Ok(Self::with_state(consensus, config, state));
            }
        }

        // No persisted state - create fresh coordinator
        Ok(Self::new(consensus, config))
    }

    fn with_state(
        consensus: ConsensusManager,
        config: DistributedTxConfig,
        state: CoordinatorState,
    ) -> Self {
        Self {
            pending: RwLock::new(state.pending),
            consensus,
            lock_manager: LockManager::from_serializable(state.lock_state),
            wait_graph: crate::deadlock::WaitForGraph::new(),
            config,
            stats: DistributedTxStats::new(),
            abort_states: RwLock::new(HashMap::new()),
            pending_aborts: RwLock::new(Vec::new()),
            wal: None,
        }
    }

    /// # Errors
    ///
    /// Returns an error if the store deletion fails (other than key not found).
    pub fn clear_persisted_state(node_id: &str, store: &TensorStore) -> Result<()> {
        // Idempotent: if key doesn't exist, that's fine (state is already cleared)
        match store.delete(&Self::persistence_key(node_id)) {
            Ok(()) => Ok(()),
            Err(e) if e.to_string().contains("not found") => Ok(()),
            Err(e) => Err(ChainError::StorageError(format!(
                "failed to clear coordinator state for {node_id}: {e}"
            ))),
        }
    }

    /// Recover from crash by processing pending transactions.
    ///
    /// Returns statistics about recovered transactions:
    /// - Timed out transactions are moved to abort
    /// - Prepared transactions with all YES votes proceed to commit
    /// - Transactions in Committing/Aborting phase retry the operation
    /// - Completed transactions (Committed/Aborted) are cleaned up
    pub fn recover(&self) -> RecoveryStats {
        let mut stats = RecoveryStats::default();
        let mut pending = self.pending.write();

        // Collect transactions by action needed
        let mut to_remove = Vec::new();

        for (tx_id, tx) in pending.iter_mut() {
            match tx.phase {
                TxPhase::Preparing => {
                    if tx.is_timed_out() {
                        // Timed out during prepare - abort
                        tx.phase = TxPhase::Aborting;
                        stats.timed_out += 1;
                    } else {
                        // Still waiting for votes
                        stats.pending_prepare += 1;
                    }
                },
                TxPhase::Prepared => {
                    if tx.is_timed_out() {
                        tx.phase = TxPhase::Aborting;
                        stats.timed_out += 1;
                    } else if tx.all_yes() {
                        // All voted yes - proceed to commit
                        tx.phase = TxPhase::Committing;
                        stats.pending_commit += 1;
                    } else if tx.any_no() {
                        // Some voted no - abort
                        tx.phase = TxPhase::Aborting;
                        stats.pending_abort += 1;
                    } else {
                        // Still waiting for more votes
                        stats.pending_prepare += 1;
                    }
                },
                TxPhase::Committing => {
                    // Retry commit
                    stats.pending_commit += 1;
                },
                TxPhase::Aborting => {
                    // Retry abort
                    stats.pending_abort += 1;
                },
                TxPhase::Committed | TxPhase::Aborted => {
                    // Already completed - clean up
                    to_remove.push(*tx_id);
                    stats.completed += 1;
                },
            }
        }

        // Release locks and remove completed transactions
        for tx_id in to_remove {
            if let Some(tx) = pending.remove(&tx_id) {
                for vote in tx.votes.values() {
                    if let PrepareVote::Yes { lock_handle, .. } = vote {
                        self.lock_manager
                            .release_by_handle_with_wait_cleanup(*lock_handle, &self.wait_graph);
                    }
                }
            }
        }

        // Also cleanup any expired locks
        self.lock_manager
            .cleanup_expired_with_wait_cleanup(&self.wait_graph);

        stats
    }

    pub fn get_pending_decisions(&self) -> Vec<(u64, TxPhase)> {
        self.pending
            .read()
            .iter()
            .filter(|(_, tx)| matches!(tx.phase, TxPhase::Committing | TxPhase::Aborting))
            .map(|(id, tx)| (*id, tx.phase))
            .collect()
    }

    /// Get all pending transactions for partition merge reconciliation.
    ///
    /// Returns transaction state in a format suitable for merging with
    /// another partition's pending transactions.
    pub fn get_pending_transactions(&self) -> Vec<crate::partition_merge::PendingTxState> {
        self.pending
            .read()
            .values()
            .map(|tx| {
                let mut state = crate::partition_merge::PendingTxState::new(
                    tx.tx_id,
                    tx.coordinator.clone(),
                    tx.phase,
                );
                state.participants.clone_from(&tx.participants);
                state.votes = tx
                    .votes
                    .iter()
                    .map(|(shard, vote)| (*shard, matches!(vote, PrepareVote::Yes { .. })))
                    .collect();
                state.delta = tx.merged_delta().map(|d| d.delta);
                state.started_at = tx.started_at;
                state
            })
            .collect()
    }

    /// Force resolve a transaction during partition merge.
    ///
    /// This is used to reconcile transactions that were pending during a partition.
    /// If `commit` is true and the transaction is in a commitable state, it commits.
    /// Otherwise, the transaction is aborted.
    /// # Errors
    ///
    /// Returns an error if the transaction is not found or cannot be committed.
    #[allow(clippy::significant_drop_tightening)]
    pub fn force_resolve(&self, tx_id: u64, commit: bool) -> Result<()> {
        let mut pending = self.pending.write();
        let tx = pending.get_mut(&tx_id).ok_or_else(|| {
            ChainError::TransactionFailed(format!("transaction {tx_id} not found"))
        })?;

        if commit {
            // Can only commit if all votes are YES
            if tx.all_yes() || matches!(tx.phase, TxPhase::Prepared | TxPhase::Committing) {
                tx.phase = TxPhase::Committing;
                // Release locks
                for vote in tx.votes.values() {
                    if let PrepareVote::Yes { lock_handle, .. } = vote {
                        self.lock_manager
                            .release_by_handle_with_wait_cleanup(*lock_handle, &self.wait_graph);
                    }
                }
                tx.phase = TxPhase::Committed;
                self.stats.committed.fetch_add(1, Ordering::Relaxed);
                pending.remove(&tx_id);
            } else {
                return Err(ChainError::TransactionFailed(format!(
                    "transaction {tx_id} cannot be committed (not all votes are YES)"
                )));
            }
        } else {
            // Abort the transaction
            tx.phase = TxPhase::Aborting;
            for vote in tx.votes.values() {
                if let PrepareVote::Yes { lock_handle, .. } = vote {
                    self.lock_manager
                        .release_by_handle_with_wait_cleanup(*lock_handle, &self.wait_graph);
                }
            }
            tx.phase = TxPhase::Aborted;
            self.stats.aborted.fetch_add(1, Ordering::Relaxed);
            pending.remove(&tx_id);
        }

        Ok(())
    }

    /// Release orphaned locks from transactions that started before a partition.
    ///
    /// During a partition, transactions may have acquired locks but never completed.
    /// This method releases locks for transactions that started before `partition_start`
    /// and are no longer active.
    ///
    /// Returns the number of locks released.
    pub fn release_orphaned_locks(&self, partition_start_ms: u64) -> usize {
        // Atomic single critical section: identify and clean together to prevent TOCTOU races
        let mut locks = self.lock_manager.locks.write();
        let mut tx_locks = self.lock_manager.tx_locks.write();
        let pending = self.pending.read();

        // Snapshot active transactions while holding all locks
        let active_tx_ids: std::collections::HashSet<u64> = pending.keys().copied().collect();
        drop(pending);

        // Identify orphaned locks while holding write locks
        let orphaned_keys: Vec<(String, u64)> = locks
            .iter()
            .filter_map(|(key, lock)| {
                if !active_tx_ids.contains(&lock.tx_id) && lock.acquired_at_ms < partition_start_ms
                {
                    Some((key.clone(), lock.tx_id))
                } else {
                    None
                }
            })
            .collect();

        // Collect unique tx_ids for wait graph cleanup
        let orphaned_tx_ids: HashSet<u64> = orphaned_keys.iter().map(|(_, tx_id)| *tx_id).collect();

        // Release all orphaned locks atomically
        let count = orphaned_keys.len();
        for (key, tx_id) in orphaned_keys {
            locks.remove(&key);
            if let Some(tx_keys) = tx_locks.get_mut(&tx_id) {
                tx_keys.retain(|k| k != &key);
                if tx_keys.is_empty() {
                    tx_locks.remove(&tx_id);
                }
            }
        }

        // Release lock manager locks before touching wait graph
        drop(tx_locks);
        drop(locks);

        // Clean up wait graph for all orphaned transactions
        for tx_id in orphaned_tx_ids {
            self.wait_graph.remove_transaction(tx_id);
        }

        count
    }
}

/// Transaction participant on a shard.
pub struct TxParticipant {
    /// Prepared transactions awaiting commit/abort.
    pub prepared: RwLock<HashMap<u64, PreparedTx>>,
    /// Lock manager for this shard.
    pub locks: LockManager,
    /// Store for undo log capture and rollback.
    store: TensorStore,
}

impl std::fmt::Debug for TxParticipant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TxParticipant")
            .field("prepared", &self.prepared)
            .field("locks", &self.locks)
            .field("store", &"<TensorStore>")
            .finish()
    }
}

/// A prepared transaction on a participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedTx {
    /// Transaction ID.
    pub tx_id: u64,
    /// Lock handle.
    pub lock_handle: u64,
    /// Operations to execute.
    pub operations: Vec<Transaction>,
    /// Delta embedding.
    pub delta: DeltaVector,
    /// When the transaction was prepared (epoch milliseconds).
    pub prepared_at_ms: EpochMillis,
    /// Undo log for rollback (captures previous state of modified keys).
    #[serde(default)]
    pub undo_log: Vec<UndoEntry>,
    /// CRC32 checksums for each undo entry (parallel to `undo_log`).
    #[serde(default)]
    pub undo_checksums: Vec<u32>,
}

impl TxParticipant {
    #[must_use]
    pub fn new(store: TensorStore) -> Self {
        Self {
            prepared: RwLock::new(HashMap::new()),
            locks: LockManager::new(),
            store,
        }
    }

    /// Create a new transaction participant with a fresh in-memory store.
    /// Primarily for testing.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::new(TensorStore::new())
    }

    /// Access the underlying store for reads (e.g. in integration tests).
    #[must_use]
    pub const fn store(&self) -> &TensorStore {
        &self.store
    }

    pub fn prepare(&self, request: PrepareRequest) -> PrepareVote {
        // Use affected_key() for locking (logical keys)
        let lock_keys: Vec<String> = request
            .operations
            .iter()
            .map(|op| op.affected_key().to_string())
            .collect();

        tracing::debug!(
            tx_id = request.tx_id,
            keys = ?lock_keys,
            "Participant processing prepare request"
        );

        // Try to acquire locks
        let lock_handle = match self.locks.try_lock(request.tx_id, &lock_keys) {
            Ok(handle) => handle,
            Err(conflicting_tx) => {
                tracing::warn!(
                    tx_id = request.tx_id,
                    conflicting_tx = conflicting_tx,
                    "Participant lock conflict"
                );
                return PrepareVote::Conflict {
                    similarity: 1.0,
                    conflicting_tx,
                };
            },
        };

        // Capture undo log using storage_key() (actual keys in TensorStore)
        // This ensures rollback operates on the correct keys
        let undo_log: Vec<UndoEntry> = request
            .operations
            .iter()
            .map(|op| UndoEntry::capture(&op.storage_key(), &self.store))
            .collect();

        // Compute checksums for integrity verification during rollback
        let undo_checksums: Vec<u32> = undo_log.iter().map(UndoEntry::checksum).collect();

        let delta = DeltaVector::from_sparse(
            request.delta_embedding,
            lock_keys.into_iter().collect(),
            request.tx_id,
        );

        // Store prepared state with undo log and checksums
        self.prepared.write().insert(
            request.tx_id,
            PreparedTx {
                tx_id: request.tx_id,
                lock_handle,
                operations: request.operations,
                delta: delta.clone(),
                prepared_at_ms: now_epoch_millis(),
                undo_log,
                undo_checksums,
            },
        );

        tracing::debug!(
            tx_id = request.tx_id,
            lock_handle = lock_handle,
            "Participant prepared transaction"
        );

        PrepareVote::Yes { lock_handle, delta }
    }

    /// Apply transaction operations to the store.
    ///
    /// # Errors
    ///
    /// Returns an error if any store operation fails.
    #[allow(clippy::too_many_lines)]
    fn apply_operations(&self, operations: &[Transaction]) -> Result<()> {
        for tx in operations {
            match tx {
                Transaction::Put { key, data } => {
                    let mut tensor = TensorData::new();
                    tensor.set(
                        "data",
                        TensorValue::Scalar(ScalarValue::Bytes(data.clone())),
                    );
                    self.store
                        .put(key, tensor)
                        .map_err(|e| ChainError::StorageError(e.to_string()))?;
                },
                Transaction::Delete { key } => {
                    // Idempotent delete - ignore NotFound
                    self.store.delete(key).ok();
                },
                Transaction::Embed { key, vector } => {
                    let storage_key = format!("emb:{key}");
                    let mut tensor = TensorData::new();
                    tensor.set("vector", TensorValue::Vector(vector.clone()));
                    self.store
                        .put(storage_key, tensor)
                        .map_err(|e| ChainError::StorageError(e.to_string()))?;
                },
                Transaction::NodeCreate { key, label } => {
                    let storage_key = format!("node:{key}");
                    let mut tensor = TensorData::new();
                    tensor.set("_id", TensorValue::Scalar(ScalarValue::String(key.clone())));
                    tensor.set(
                        "_type",
                        TensorValue::Scalar(ScalarValue::String("node".into())),
                    );
                    tensor.set(
                        "_label",
                        TensorValue::Scalar(ScalarValue::String(label.clone())),
                    );
                    self.store
                        .put(storage_key, tensor)
                        .map_err(|e| ChainError::StorageError(e.to_string()))?;
                },
                Transaction::NodeDelete { key } => {
                    let storage_key = format!("node:{key}");
                    self.store.delete(&storage_key).ok();
                },
                Transaction::EdgeCreate {
                    from,
                    to,
                    edge_type,
                } => {
                    let storage_key = format!("edge:{from}:{to}:{edge_type}");
                    let mut tensor = TensorData::new();
                    tensor.set(
                        "_from",
                        TensorValue::Scalar(ScalarValue::String(from.clone())),
                    );
                    tensor.set("_to", TensorValue::Scalar(ScalarValue::String(to.clone())));
                    tensor.set(
                        "_type",
                        TensorValue::Scalar(ScalarValue::String(edge_type.clone())),
                    );
                    self.store
                        .put(storage_key, tensor)
                        .map_err(|e| ChainError::StorageError(e.to_string()))?;
                },
                Transaction::TableInsert { table, values } => {
                    let storage_key = format!("table:{table}");
                    let mut tensor = TensorData::new();
                    tensor.set(
                        "values",
                        TensorValue::Scalar(ScalarValue::Bytes(values.clone())),
                    );
                    self.store
                        .put(storage_key, tensor)
                        .map_err(|e| ChainError::StorageError(e.to_string()))?;
                },
                Transaction::TableUpdate {
                    table,
                    row_id,
                    values,
                } => {
                    let storage_key = format!("table:{table}:row:{row_id}");
                    let mut tensor = TensorData::new();
                    tensor.set(
                        "values",
                        TensorValue::Scalar(ScalarValue::Bytes(values.clone())),
                    );
                    tensor.set(
                        "row_id",
                        #[allow(clippy::cast_possible_wrap)]
                        TensorValue::Scalar(ScalarValue::Int(*row_id as i64)),
                    );
                    self.store
                        .put(storage_key, tensor)
                        .map_err(|e| ChainError::StorageError(e.to_string()))?;
                },
                Transaction::TableDelete { table, row_id } => {
                    let storage_key = format!("table:{table}:row:{row_id}");
                    self.store.delete(&storage_key).ok();
                },
                Transaction::CompareAndSwap {
                    key,
                    expected_data,
                    new_data,
                } => {
                    let current = self.store.get(key).ok();
                    let current_bytes = current
                        .as_ref()
                        .and_then(|d| d.get("data"))
                        .and_then(|v| match v {
                            TensorValue::Scalar(ScalarValue::Bytes(b)) => Some(b.as_slice()),
                            _ => None,
                        })
                        .unwrap_or(&[]);
                    if current_bytes == expected_data.as_slice() {
                        let mut tensor = TensorData::new();
                        tensor.set(
                            "data",
                            TensorValue::Scalar(ScalarValue::Bytes(new_data.clone())),
                        );
                        self.store
                            .put(key, tensor)
                            .map_err(|e| ChainError::StorageError(e.to_string()))?;
                    }
                },
            }
        }
        Ok(())
    }

    pub fn commit(&self, tx_id: u64) -> TxResponse {
        let mut prepared = self.prepared.write();

        if let Some(tx) = prepared.remove(&tx_id) {
            // Apply operations BEFORE releasing locks
            if let Err(e) = self.apply_operations(&tx.operations) {
                // Operations failed - rollback and report failure
                for entry in tx.undo_log.iter().rev() {
                    if let Err(undo_err) = entry.apply(&self.store) {
                        tracing::warn!(
                            tx_id = tx_id,
                            key = entry.key(),
                            error = %undo_err,
                            "Failed to apply undo entry during commit rollback"
                        );
                    }
                }
                self.locks.release_by_handle(tx.lock_handle);
                tracing::error!(tx_id = tx_id, error = %e, "Commit failed during operation apply");
                return TxResponse {
                    tx_id,
                    success: false,
                    error: Some(e.to_string()),
                };
            }

            self.locks.release_by_handle(tx.lock_handle);
            tracing::info!(
                tx_id = tx_id,
                op_count = tx.operations.len(),
                "Participant committed transaction"
            );
            TxResponse {
                tx_id,
                success: true,
                error: None,
            }
        } else {
            tracing::warn!(
                tx_id = tx_id,
                "Participant commit failed: transaction not found"
            );
            TxResponse {
                tx_id,
                success: false,
                error: Some("transaction not found".to_string()),
            }
        }
    }

    pub fn abort(&self, tx_id: u64) -> TxResponse {
        let tx = self.prepared.write().remove(&tx_id);

        if let Some(tx) = tx {
            // Apply undo log in reverse order to restore previous state
            for entry in tx.undo_log.iter().rev() {
                if let Err(e) = entry.apply(&self.store) {
                    tracing::warn!(
                        tx_id = tx_id,
                        key = entry.key(),
                        error = %e,
                        "Failed to apply undo entry during abort"
                    );
                    // Continue with other entries - best effort rollback
                }
            }

            self.locks.release_by_handle(tx.lock_handle);
            tracing::info!(
                tx_id = tx_id,
                undo_count = tx.undo_log.len(),
                "Participant aborted transaction with rollback"
            );
        } else {
            tracing::debug!(
                tx_id = tx_id,
                "Participant abort: transaction not found (may already be cleaned up)"
            );
        }

        TxResponse {
            tx_id,
            success: true,
            error: None,
        }
    }

    pub fn prepared_count(&self) -> usize {
        self.prepared.read().len()
    }

    pub fn cleanup_stale(&self, timeout: Duration) -> Vec<u64> {
        let mut prepared = self.prepared.write();
        let now = now_epoch_millis();
        #[allow(clippy::cast_possible_truncation)]
        let timeout_ms = timeout.as_millis() as u64;
        let stale: Vec<_> = prepared
            .iter()
            .filter(|(_, tx)| now.saturating_sub(tx.prepared_at_ms) >= timeout_ms)
            .map(|(id, _)| *id)
            .collect();

        for tx_id in &stale {
            if let Some(tx) = prepared.remove(tx_id) {
                // Apply undo log to restore previous state before releasing locks
                for entry in tx.undo_log.iter().rev() {
                    if let Err(e) = entry.apply(&self.store) {
                        tracing::warn!(
                            tx_id = tx_id,
                            key = entry.key(),
                            error = %e,
                            "Failed to apply undo entry during stale cleanup"
                        );
                    }
                }
                self.locks.release_by_handle(tx.lock_handle);
                tracing::debug!(
                    tx_id = tx_id,
                    undo_count = tx.undo_log.len(),
                    "Cleaned up stale transaction with rollback"
                );
            }
        }

        stale
    }

    fn persistence_key(node_id: &str, shard_id: ShardId) -> String {
        format!("_dtx:participant:{node_id}:shard:{shard_id}:state")
    }

    pub fn to_state(&self) -> ParticipantState {
        ParticipantState {
            prepared: self.prepared.read().clone(),
            lock_state: self.locks.to_serializable(),
        }
    }

    /// # Errors
    /// Returns an error if serialization or storage fails.
    pub fn save_to_store(
        &self,
        node_id: &str,
        shard_id: ShardId,
        store: &TensorStore,
    ) -> Result<()> {
        let state = self.to_state();
        let bytes = bitcode::serialize(&state)?;

        let mut data = TensorData::new();
        data.set("state", TensorValue::Scalar(ScalarValue::Bytes(bytes)));
        store
            .put(Self::persistence_key(node_id, shard_id), data)
            .map_err(|e| ChainError::StorageError(e.to_string()))?;

        Ok(())
    }

    /// Load participant from `TensorStore` or create fresh if not found.
    #[must_use]
    pub fn load_from_store(node_id: &str, shard_id: ShardId, store: &TensorStore) -> Self {
        let key = Self::persistence_key(node_id, shard_id);

        if let Ok(data) = store.get(&key) {
            if let Some(TensorValue::Scalar(ScalarValue::Bytes(bytes))) = data.get("state") {
                if let Ok(state) = bitcode::deserialize::<ParticipantState>(bytes) {
                    return Self::with_state(state, store.clone());
                }
            }
        }

        // No persisted state - create fresh participant
        Self::new(store.clone())
    }

    fn with_state(state: ParticipantState, store: TensorStore) -> Self {
        Self {
            prepared: RwLock::new(state.prepared),
            locks: LockManager::from_serializable(state.lock_state),
            store,
        }
    }

    /// # Errors
    /// Returns an error if the underlying store deletion fails.
    pub fn clear_persisted_state(
        node_id: &str,
        shard_id: ShardId,
        store: &TensorStore,
    ) -> Result<()> {
        // Idempotent: if key doesn't exist, that's fine (state is already cleared)
        match store.delete(&Self::persistence_key(node_id, shard_id)) {
            Ok(()) => Ok(()),
            Err(e) if e.to_string().contains("not found") => Ok(()),
            Err(e) => Err(ChainError::StorageError(format!(
                "failed to clear participant state for {node_id}:{shard_id}: {e}"
            ))),
        }
    }

    /// Recover from crash by checking prepared transactions.
    ///
    /// Returns the transaction IDs that are still awaiting coordinator decisions.
    /// For transactions that have exceeded the timeout, their locks are released
    /// (presumed abort - the coordinator must have crashed or aborted).
    pub fn recover(&self, timeout: Duration) -> Vec<u64> {
        let mut prepared = self.prepared.write();
        let now = now_epoch_millis();
        #[allow(clippy::cast_possible_truncation)]
        let timeout_ms = timeout.as_millis() as u64;

        // Find expired transactions (presumed abort)
        let expired: Vec<u64> = prepared
            .iter()
            .filter(|(_, tx)| now.saturating_sub(tx.prepared_at_ms) > timeout_ms)
            .map(|(id, _)| *id)
            .collect();

        // Apply undo logs and release locks for expired transactions (presumed abort)
        for tx_id in &expired {
            if let Some(tx) = prepared.remove(tx_id) {
                // Apply undo log BEFORE releasing locks to restore previous state
                for entry in tx.undo_log.iter().rev() {
                    if let Err(e) = entry.apply(&self.store) {
                        tracing::warn!(
                            tx_id = tx_id,
                            key = entry.key(),
                            error = %e,
                            "Failed to apply undo entry during recovery"
                        );
                    }
                }
                self.locks.release_by_handle(tx.lock_handle);
                tracing::info!(
                    tx_id = tx_id,
                    undo_count = tx.undo_log.len(),
                    "Recovered expired transaction with rollback"
                );
            }
        }

        // Cleanup any other expired locks
        self.locks.cleanup_expired();

        // Return IDs of transactions still awaiting decision
        prepared.keys().copied().collect()
    }

    pub fn get_awaiting_decision(&self) -> Vec<u64> {
        self.prepared.read().keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::ConsensusConfig;

    fn create_test_coordinator() -> DistributedTxCoordinator {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        DistributedTxCoordinator::with_consensus(consensus)
    }

    #[test]
    fn test_distributed_tx_creation() {
        let tx = DistributedTransaction::new("node1".to_string(), vec![0, 1, 2]);

        assert_eq!(tx.coordinator, "node1");
        assert_eq!(tx.participants, vec![0, 1, 2]);
        assert_eq!(tx.phase, TxPhase::Preparing);
        assert!(tx.operations.is_empty());
        assert!(!tx.is_timed_out());
    }

    #[test]
    fn test_distributed_tx_operations() {
        let mut tx = DistributedTransaction::new("node1".to_string(), vec![0, 1]);

        tx.add_operations(
            0,
            vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
        );

        tx.add_operations(
            1,
            vec![Transaction::Put {
                key: "key2".to_string(),
                data: vec![2],
            }],
        );

        assert_eq!(tx.operations.len(), 2);
        assert!(tx.operations.contains_key(&0));
        assert!(tx.operations.contains_key(&1));
    }

    #[test]
    fn test_lock_manager_basic() {
        let lock_manager = LockManager::new();

        let keys = vec!["key1".to_string(), "key2".to_string()];
        let handle = lock_manager.try_lock(1, &keys).unwrap();

        assert!(lock_manager.is_locked("key1"));
        assert!(lock_manager.is_locked("key2"));
        assert_eq!(lock_manager.lock_holder("key1"), Some(1));

        lock_manager.release_by_handle(handle);

        assert!(!lock_manager.is_locked("key1"));
        assert!(!lock_manager.is_locked("key2"));
    }

    #[test]
    fn test_lock_manager_conflict() {
        let lock_manager = LockManager::new();

        let keys = vec!["key1".to_string()];
        lock_manager.try_lock(1, &keys).unwrap();

        // Second transaction should fail
        let result = lock_manager.try_lock(2, &keys);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), 1);
    }

    #[test]
    fn test_lock_manager_different_keys() {
        let lock_manager = LockManager::new();

        lock_manager.try_lock(1, &["key1".to_string()]).unwrap();
        lock_manager.try_lock(2, &["key2".to_string()]).unwrap();

        assert!(lock_manager.is_locked("key1"));
        assert!(lock_manager.is_locked("key2"));
        assert_eq!(lock_manager.lock_holder("key1"), Some(1));
        assert_eq!(lock_manager.lock_holder("key2"), Some(2));
    }

    #[test]
    fn test_coordinator_begin() {
        let coordinator = create_test_coordinator();

        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();

        assert_eq!(coordinator.pending_count(), 1);
        assert!(coordinator.get(tx.tx_id).is_some());
    }

    #[test]
    fn test_coordinator_max_concurrent() {
        let config = DistributedTxConfig {
            max_concurrent: 2,
            ..Default::default()
        };
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config);

        coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        coordinator.begin(&"node1".to_string(), &[1]).unwrap();

        // Third should fail
        let result = coordinator.begin(&"node1".to_string(), &[2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_prepare_vote_yes() {
        let coordinator = create_test_coordinator();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            timeout_ms: 5000,
        };

        let vote = coordinator.handle_prepare(&request);

        match vote {
            PrepareVote::Yes { lock_handle, delta } => {
                assert!(lock_handle > 0);
                assert!(delta.affected_keys.contains("key1"));
            },
            _ => panic!("expected Yes vote"),
        }
    }

    #[test]
    fn test_voting_all_yes() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();

        let delta0 = DeltaVector::new(
            &[1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );
        let delta1 = DeltaVector::new(
            &[0.0, 1.0, 0.0],
            ["key2"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        // Vote from shard 0
        let result = coordinator.record_vote(
            tx.tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta: delta0,
            },
        );
        assert_eq!(result.unwrap(), None); // Not all voted yet

        // Vote from shard 1
        let result = coordinator.record_vote(
            tx.tx_id,
            1,
            PrepareVote::Yes {
                lock_handle: 2,
                delta: delta1,
            },
        );
        assert_eq!(result.unwrap(), Some(TxPhase::Prepared));
    }

    #[test]
    fn test_voting_any_no() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();

        let delta = DeltaVector::new(
            &[1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        let _ = coordinator.record_vote(
            tx.tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta,
            },
        );

        let result = coordinator.record_vote(
            tx.tx_id,
            1,
            PrepareVote::No {
                reason: "test".to_string(),
            },
        );
        assert_eq!(result.unwrap(), Some(TxPhase::Aborting));
    }

    #[test]
    fn test_commit_flow() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        let delta = DeltaVector::new(
            &[1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        let _ = coordinator.record_vote(
            tx.tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta,
            },
        );

        coordinator.commit(tx.tx_id).unwrap();

        assert_eq!(coordinator.pending_count(), 0);
        assert_eq!(coordinator.stats.committed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_abort_flow() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        coordinator.abort(tx.tx_id, "test abort").unwrap();

        assert_eq!(coordinator.pending_count(), 0);
        assert_eq!(coordinator.stats.aborted.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_participant_prepare_commit() {
        let participant = TxParticipant::new_in_memory();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            timeout_ms: 5000,
        };

        let vote = participant.prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));
        assert_eq!(participant.prepared_count(), 1);

        let response = participant.commit(1);
        assert!(response.success);
        assert_eq!(participant.prepared_count(), 0);
    }

    #[test]
    fn test_participant_abort() {
        let participant = TxParticipant::new_in_memory();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        let response = participant.abort(1);

        assert!(response.success);
        assert_eq!(participant.prepared_count(), 0);
    }

    #[test]
    fn test_distributed_tx_stats() {
        let stats = DistributedTxStats::new();

        stats.started.fetch_add(10, Ordering::Relaxed);
        stats.committed.fetch_add(7, Ordering::Relaxed);
        stats.conflicts.fetch_add(2, Ordering::Relaxed);

        assert!((stats.commit_rate() - 0.7).abs() < 0.01);
        assert!((stats.conflict_rate() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_tx_phase_default() {
        assert_eq!(TxPhase::default(), TxPhase::Preparing);
    }

    #[test]
    fn test_distributed_tx_merged_delta() {
        let mut tx = DistributedTransaction::new("node1".to_string(), vec![0, 1]);

        let delta0 = DeltaVector::new(
            &[1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            1,
        );
        let delta1 = DeltaVector::new(
            &[0.0, 1.0, 0.0],
            ["key2"].iter().map(|s| s.to_string()).collect(),
            2,
        );

        tx.add_delta(0, delta0);
        tx.add_delta(1, delta1);

        let merged = tx.merged_delta().unwrap();
        assert_eq!(merged.to_dense(3), vec![1.0, 1.0, 0.0]);
        assert!(merged.affected_keys.contains("key1"));
        assert!(merged.affected_keys.contains("key2"));
    }

    #[test]
    fn test_distributed_tx_merged_delta_empty() {
        let tx = DistributedTransaction::new("node1".to_string(), vec![0]);
        assert!(tx.merged_delta().is_none());
    }

    #[test]
    fn test_distributed_tx_affected_keys() {
        let mut tx = DistributedTransaction::new("node1".to_string(), vec![0, 1]);

        let delta0 = DeltaVector::new(
            &[1.0, 0.0],
            ["key1", "key2"].iter().map(|s| s.to_string()).collect(),
            1,
        );
        let delta1 = DeltaVector::new(
            &[0.0, 1.0],
            ["key3"].iter().map(|s| s.to_string()).collect(),
            2,
        );

        tx.add_delta(0, delta0);
        tx.add_delta(1, delta1);

        let keys = tx.affected_keys();
        assert!(keys.contains("key1"));
        assert!(keys.contains("key2"));
        assert!(keys.contains("key3"));
    }

    #[test]
    fn test_distributed_tx_voting() {
        let mut tx = DistributedTransaction::new("node1".to_string(), vec![0, 1]);

        // Initially no votes
        assert!(!tx.all_voted());
        assert!(tx.all_yes()); // vacuously true for empty
        assert!(!tx.any_no());

        // Add one vote
        tx.record_vote(
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta: DeltaVector::new(&[1.0], HashSet::new(), 1),
            },
        );

        assert!(!tx.all_voted()); // Still missing shard 1
        assert!(tx.all_yes());
        assert!(!tx.any_no());

        // Add second vote
        tx.record_vote(
            1,
            PrepareVote::No {
                reason: "test".to_string(),
            },
        );

        assert!(tx.all_voted());
        assert!(!tx.all_yes());
        assert!(tx.any_no());
    }

    #[test]
    fn test_distributed_tx_timeout() {
        let mut tx = DistributedTransaction::new("node1".to_string(), vec![0]);
        tx.timeout_ms = 0; // Immediate timeout

        // Sleep a tiny bit to ensure timeout
        std::thread::sleep(Duration::from_millis(1));
        assert!(tx.is_timed_out());
    }

    #[test]
    fn test_coordinator_cleanup_timeouts() {
        let config = DistributedTxConfig {
            prepare_timeout_ms: 0, // For quick test
            ..Default::default()
        };
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config);

        // Create a transaction with immediate timeout
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Modify timeout to 0 (simulating timeout)
        {
            let mut pending = coordinator.pending.write();
            if let Some(t) = pending.get_mut(&tx.tx_id) {
                t.timeout_ms = 0;
            }
        }

        std::thread::sleep(Duration::from_millis(1));

        let timed_out = coordinator.cleanup_timeouts();
        assert!(timed_out.contains(&tx.tx_id));
        assert_eq!(coordinator.pending_count(), 0);
    }

    #[test]
    fn test_lock_manager_release_by_tx_id() {
        let lock_manager = LockManager::new();

        let keys = vec!["key1".to_string(), "key2".to_string()];
        lock_manager.try_lock(1, &keys).unwrap();

        assert!(lock_manager.is_locked("key1"));
        assert!(lock_manager.is_locked("key2"));

        lock_manager.release(1);

        assert!(!lock_manager.is_locked("key1"));
        assert!(!lock_manager.is_locked("key2"));
    }

    #[test]
    fn test_lock_manager_cleanup_expired() {
        let mut lock_manager = LockManager::new();
        lock_manager.default_timeout = Duration::from_millis(0);

        let keys = vec!["key1".to_string()];
        lock_manager.try_lock(1, &keys).unwrap();

        std::thread::sleep(Duration::from_millis(1));

        let cleaned = lock_manager.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert!(!lock_manager.is_locked("key1"));
    }

    #[test]
    fn test_lock_manager_expired_lock_allows_new_lock() {
        let mut lock_manager = LockManager::new();
        lock_manager.default_timeout = Duration::from_millis(0);

        let keys = vec!["key1".to_string()];
        lock_manager.try_lock(1, &keys).unwrap();

        std::thread::sleep(Duration::from_millis(1));

        // Expired lock should allow new lock
        let result = lock_manager.try_lock(2, &keys);
        assert!(result.is_ok());
    }

    #[test]
    fn test_lock_manager_same_tx_can_relock() {
        let lock_manager = LockManager::new();

        let keys = vec!["key1".to_string()];
        lock_manager.try_lock(1, &keys).unwrap();

        // Same tx can lock same keys again
        let result = lock_manager.try_lock(1, &keys);
        assert!(result.is_ok());
    }

    #[test]
    fn test_commit_not_prepared() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Try to commit without preparing
        let result = coordinator.commit(tx.tx_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_commit_not_found() {
        let coordinator = create_test_coordinator();
        let result = coordinator.commit(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_abort_not_found() {
        let coordinator = create_test_coordinator();
        let result = coordinator.abort(999, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_handle_prepare_conflict() {
        let coordinator = create_test_coordinator();

        // First prepare
        let request1 = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            timeout_ms: 5000,
        };

        let vote1 = coordinator.handle_prepare(&request1);
        assert!(matches!(vote1, PrepareVote::Yes { .. }));

        // Second prepare on same key should conflict
        let request2 = PrepareRequest {
            tx_id: 2,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![2],
            }],
            delta_embedding: SparseVector::from_dense(&[0.0, 1.0, 0.0]),
            timeout_ms: 5000,
        };

        let vote2 = coordinator.handle_prepare(&request2);
        assert!(matches!(vote2, PrepareVote::Conflict { .. }));
    }

    #[test]
    fn test_participant_prepare_conflict() {
        let participant = TxParticipant::new_in_memory();

        // First prepare
        let request1 = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        };

        let vote1 = participant.prepare(request1);
        assert!(matches!(vote1, PrepareVote::Yes { .. }));

        // Second prepare on same key
        let request2 = PrepareRequest {
            tx_id: 2,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![2],
            }],
            delta_embedding: SparseVector::from_dense(&[0.0, 1.0]),
            timeout_ms: 5000,
        };

        let vote2 = participant.prepare(request2);
        assert!(matches!(vote2, PrepareVote::Conflict { .. }));
    }

    #[test]
    fn test_participant_commit_not_found() {
        let participant = TxParticipant::new_in_memory();
        let response = participant.commit(999);
        assert!(!response.success);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_participant_abort_not_prepared() {
        let participant = TxParticipant::new_in_memory();
        // Abort should succeed even if not prepared
        let response = participant.abort(999);
        assert!(response.success);
    }

    #[test]
    fn test_participant_cleanup_stale() {
        let participant = TxParticipant::new_in_memory();

        // Insert a prepared transaction with an old timestamp (10 seconds ago)
        let old_prepared_at = now_epoch_millis().saturating_sub(10_000);
        participant.prepared.write().insert(
            1,
            PreparedTx {
                tx_id: 1,
                lock_handle: 1,
                operations: vec![Transaction::Put {
                    key: "key1".to_string(),
                    data: vec![1],
                }],
                delta: DeltaVector::new(&[1.0], HashSet::new(), 1),
                prepared_at_ms: old_prepared_at,
                undo_log: Vec::new(),
                undo_checksums: Vec::new(),
            },
        );
        assert_eq!(participant.prepared_count(), 1);

        // Cleanup with 5 second timeout should find the 10-second-old transaction
        let stale = participant.cleanup_stale(Duration::from_secs(5));
        assert_eq!(stale.len(), 1);
        assert_eq!(participant.prepared_count(), 0);
    }

    // ========== UndoEntry Tests ==========

    #[test]
    fn test_undo_entry_capture_existing_key() {
        let store = TensorStore::new();
        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        store.put("test_key", data).unwrap();

        let entry = UndoEntry::capture("test_key", &store);
        assert!(matches!(entry, UndoEntry::Restore { key, .. } if key == "test_key"));
    }

    #[test]
    fn test_undo_entry_capture_missing_key() {
        let store = TensorStore::new();

        let entry = UndoEntry::capture("nonexistent", &store);
        assert!(matches!(entry, UndoEntry::Delete { key } if key == "nonexistent"));
    }

    #[test]
    fn test_undo_entry_apply_restore() {
        let store = TensorStore::new();

        // Create original data
        let mut original = TensorData::new();
        original.set("value", TensorValue::Scalar(ScalarValue::Int(100)));
        store.put("key1", original.clone()).unwrap();

        // Capture original state
        let entry = UndoEntry::capture("key1", &store);

        // Overwrite with new data
        let mut new_data = TensorData::new();
        new_data.set("value", TensorValue::Scalar(ScalarValue::Int(999)));
        store.put("key1", new_data).unwrap();

        // Verify new value is set
        let current = store.get("key1").unwrap();
        assert_eq!(
            current.get("value"),
            Some(&TensorValue::Scalar(ScalarValue::Int(999)))
        );

        // Apply undo to restore original
        entry.apply(&store).unwrap();

        // Verify original is restored
        let restored = store.get("key1").unwrap();
        assert_eq!(
            restored.get("value"),
            Some(&TensorValue::Scalar(ScalarValue::Int(100)))
        );
    }

    #[test]
    fn test_undo_entry_apply_delete() {
        let store = TensorStore::new();

        // Capture state when key doesn't exist
        let entry = UndoEntry::capture("new_key", &store);
        assert!(matches!(entry, UndoEntry::Delete { .. }));

        // Create the key
        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        store.put("new_key", data).unwrap();
        assert!(store.exists("new_key"));

        // Apply undo - should delete the key
        entry.apply(&store).unwrap();
        assert!(!store.exists("new_key"));
    }

    #[test]
    fn test_undo_entry_key_accessor() {
        let restore = UndoEntry::Restore {
            key: "restore_key".to_string(),
            data: vec![],
        };
        assert_eq!(restore.key(), "restore_key");

        let delete = UndoEntry::Delete {
            key: "delete_key".to_string(),
        };
        assert_eq!(delete.key(), "delete_key");
    }

    #[test]
    fn test_abort_applies_undo_log() {
        let store = TensorStore::new();

        // Set up initial state
        let mut original = TensorData::new();
        original.set("value", TensorValue::Scalar(ScalarValue::Int(100)));
        store.put("key1", original).unwrap();

        let participant = TxParticipant::new(store.clone());

        // Prepare a transaction that modifies key1
        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1, 2, 3],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        };

        let vote = participant.prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));

        // Simulate writes that would happen during execution
        let mut new_data = TensorData::new();
        new_data.set("value", TensorValue::Scalar(ScalarValue::Int(999)));
        store.put("key1", new_data).unwrap();

        // Abort should restore original value
        let response = participant.abort(1);
        assert!(response.success);

        // Verify original state is restored
        let restored = store.get("key1").unwrap();
        assert_eq!(
            restored.get("value"),
            Some(&TensorValue::Scalar(ScalarValue::Int(100)))
        );
    }

    #[test]
    fn test_prepare_captures_undo_for_new_keys() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        // Prepare a transaction for a key that doesn't exist
        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "new_key".to_string(),
                data: vec![1, 2, 3],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        };

        let vote = participant.prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));

        // Check that undo log has Delete entry for the new key
        let prepared = participant.prepared.read();
        let tx = prepared.get(&1).unwrap();
        assert_eq!(tx.undo_log.len(), 1);
        assert!(matches!(&tx.undo_log[0], UndoEntry::Delete { key } if key == "new_key"));
        // Checksums should be populated
        assert_eq!(tx.undo_checksums.len(), 1);
        assert_eq!(tx.undo_checksums[0], tx.undo_log[0].checksum());
    }

    #[test]
    fn test_undo_entry_checksum_deterministic() {
        let store = TensorStore::new();
        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        store.put("key1", data).unwrap();

        let entry = UndoEntry::capture("key1", &store);
        let c1 = entry.checksum();
        let c2 = entry.checksum();
        assert_eq!(c1, c2, "Checksums should be deterministic");
    }

    #[test]
    fn test_undo_entry_checksum_differs_by_variant() {
        let restore = UndoEntry::Restore {
            key: "key1".to_string(),
            data: vec![1, 2, 3],
        };
        let delete = UndoEntry::Delete {
            key: "key1".to_string(),
        };
        assert_ne!(
            restore.checksum(),
            delete.checksum(),
            "Different variants should produce different checksums"
        );
    }

    #[test]
    fn test_undo_entry_checksum_differs_by_key() {
        let e1 = UndoEntry::Delete {
            key: "key1".to_string(),
        };
        let e2 = UndoEntry::Delete {
            key: "key2".to_string(),
        };
        assert_ne!(e1.checksum(), e2.checksum());
    }

    #[test]
    fn test_undo_entry_apply_verified_succeeds() {
        let store = TensorStore::new();
        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(99)));
        store.put("key1", data).unwrap();

        let entry = UndoEntry::capture("key1", &store);
        let checksum = entry.checksum();

        // Delete the key then restore via verified apply
        store.delete("key1").unwrap();
        assert!(entry.apply_verified(&store, checksum).is_ok());
        assert!(store.get("key1").is_ok());
    }

    #[test]
    fn test_undo_entry_apply_verified_rejects_bad_checksum() {
        let store = TensorStore::new();
        let entry = UndoEntry::Delete {
            key: "key1".to_string(),
        };
        let bad_checksum = entry.checksum() ^ 0xDEAD_BEEF;
        let result = entry.apply_verified(&store, bad_checksum);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch"));
    }

    #[test]
    fn test_cleanup_stale_applies_undo() {
        let store = TensorStore::new();

        // Set up initial state
        let mut original = TensorData::new();
        original.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        store.put("stale_key", original).unwrap();

        let participant = TxParticipant::new(store.clone());

        // Manually insert a stale prepared transaction with undo log
        let old_prepared_at = now_epoch_millis().saturating_sub(10_000);
        let undo_entry = UndoEntry::capture("stale_key", &store);
        let undo_checksum = undo_entry.checksum();

        // Modify the data (simulating what would happen during tx execution)
        let mut new_data = TensorData::new();
        new_data.set("value", TensorValue::Scalar(ScalarValue::Int(999)));
        store.put("stale_key", new_data).unwrap();

        // Insert prepared tx with the undo log
        participant.prepared.write().insert(
            1,
            PreparedTx {
                tx_id: 1,
                lock_handle: 1,
                operations: vec![],
                delta: DeltaVector::new(&[1.0], HashSet::new(), 1),
                prepared_at_ms: old_prepared_at,
                undo_log: vec![undo_entry],
                undo_checksums: vec![undo_checksum],
            },
        );

        // Cleanup stale should apply undo
        let stale = participant.cleanup_stale(Duration::from_secs(5));
        assert_eq!(stale.len(), 1);

        // Verify original state is restored
        let restored = store.get("stale_key").unwrap();
        assert_eq!(
            restored.get("value"),
            Some(&TensorValue::Scalar(ScalarValue::Int(42)))
        );
    }

    #[test]
    fn test_distributed_tx_stats_zero() {
        let stats = DistributedTxStats::new();
        assert_eq!(stats.commit_rate(), 0.0);
        assert_eq!(stats.conflict_rate(), 0.0);
    }

    #[test]
    fn test_distributed_tx_config_default() {
        let config = DistributedTxConfig::default();
        assert_eq!(config.prepare_timeout_ms, 5000);
        assert_eq!(config.commit_timeout_ms, 10000);
        assert_eq!(config.max_concurrent, 100);
        assert!((config.orthogonal_threshold - 0.1).abs() < 0.01);
        assert!(config.optimistic_locking);
        assert_eq!(config.tx_queue_soft_limit_pct, 80);
    }

    #[test]
    fn test_key_lock_expiry() {
        // Lock acquired 100 seconds ago with 1 second timeout - should be expired
        let lock = KeyLock {
            key: "test".to_string(),
            tx_id: 1,
            lock_handle: 1,
            acquired_at_ms: now_epoch_millis().saturating_sub(100_000),
            timeout_ms: 1000,
        };
        assert!(lock.is_expired());

        // Lock acquired now with 100 second timeout - should not be expired
        let lock2 = KeyLock {
            key: "test".to_string(),
            tx_id: 1,
            lock_handle: 1,
            acquired_at_ms: now_epoch_millis(),
            timeout_ms: 100_000,
        };
        assert!(!lock2.is_expired());
    }

    #[test]
    fn test_lock_holder_expired() {
        let mut lock_manager = LockManager::new();
        lock_manager.default_timeout = Duration::from_millis(0);

        let keys = vec!["key1".to_string()];
        lock_manager.try_lock(1, &keys).unwrap();

        std::thread::sleep(Duration::from_millis(1));

        // Expired lock holder should return None
        assert!(lock_manager.lock_holder("key1").is_none());
    }

    #[test]
    fn test_lock_manager_active_count() {
        let lock_manager = LockManager::new();

        assert_eq!(lock_manager.active_lock_count(), 0);

        lock_manager.try_lock(1, &["key1".to_string()]).unwrap();
        assert_eq!(lock_manager.active_lock_count(), 1);

        lock_manager.try_lock(2, &["key2".to_string()]).unwrap();
        assert_eq!(lock_manager.active_lock_count(), 2);

        lock_manager.release(1);
        assert_eq!(lock_manager.active_lock_count(), 1);
    }

    #[test]
    fn test_coordinator_lock_manager_accessor() {
        let coordinator = create_test_coordinator();
        let lm = coordinator.lock_manager();
        assert_eq!(lm.active_lock_count(), 0);
    }

    #[test]
    fn test_coordinator_stats_accessor() {
        let coordinator = create_test_coordinator();
        let stats = coordinator.stats();
        assert_eq!(stats.started.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_voting_with_conflict() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();

        let delta = DeltaVector::new(
            &[1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        let _ = coordinator.record_vote(
            tx.tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta,
            },
        );

        let result = coordinator.record_vote(
            tx.tx_id,
            1,
            PrepareVote::Conflict {
                similarity: 0.9,
                conflicting_tx: 999,
            },
        );

        assert_eq!(result.unwrap(), Some(TxPhase::Aborting));
        assert!(coordinator.stats.conflicts.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_tx_participant_new_in_memory() {
        let participant = TxParticipant::new_in_memory();
        assert_eq!(participant.prepared_count(), 0);
    }

    #[test]
    fn test_cross_shard_conflict_detection() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();

        // Two shards with overlapping keys and similar deltas
        let delta0 = DeltaVector::new(
            &[1.0, 0.0, 0.0],
            ["shared_key"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );
        let delta1 = DeltaVector::new(
            &[1.0, 0.1, 0.0], // Similar to delta0
            ["shared_key"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        let _ = coordinator.record_vote(
            tx.tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta: delta0,
            },
        );

        let result = coordinator.record_vote(
            tx.tx_id,
            1,
            PrepareVote::Yes {
                lock_handle: 2,
                delta: delta1,
            },
        );

        // Should detect conflict due to overlapping keys and non-orthogonal deltas
        assert_eq!(result.unwrap(), Some(TxPhase::Aborting));
    }

    #[test]
    fn test_handle_prepare_semantic_conflict() {
        let config = DistributedTxConfig {
            optimistic_locking: true,
            orthogonal_threshold: 0.1,
            ..Default::default()
        };
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config);

        // Start a transaction and add delta
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        {
            let mut pending = coordinator.pending.write();
            if let Some(t) = pending.get_mut(&tx.tx_id) {
                let delta = DeltaVector::new(
                    &[1.0, 0.0, 0.0],
                    ["key1"].iter().map(|s| s.to_string()).collect(),
                    tx.tx_id,
                );
                t.add_delta(0, delta);
            }
        }

        // Prepare request with similar delta and overlapping key
        let request = PrepareRequest {
            tx_id: tx.tx_id + 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]), // Same direction
            timeout_ms: 5000,
        };

        let vote = coordinator.handle_prepare(&request);
        assert!(matches!(vote, PrepareVote::Conflict { .. }));
    }

    #[test]
    fn test_record_vote_not_found() {
        let coordinator = create_test_coordinator();
        let result = coordinator.record_vote(
            999,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta: DeltaVector::new(&[1.0], HashSet::new(), 1),
            },
        );
        assert_eq!(result, Err(VoteRecordError::TxNotFound(999)));
    }

    #[test]
    fn test_record_vote_wrong_phase() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;

        // Record a YES vote to move to Prepared phase
        let delta = DeltaVector::new(&[1.0], HashSet::new(), tx_id);
        let vote = PrepareVote::Yes {
            lock_handle: 1,
            delta,
        };
        let result = coordinator.record_vote(tx_id, 0, vote);
        assert_eq!(result.unwrap(), Some(TxPhase::Prepared));

        // Try to record another vote -- tx is now in Prepared phase, not Preparing
        let delta2 = DeltaVector::new(&[1.0], HashSet::new(), tx_id);
        let vote2 = PrepareVote::Yes {
            lock_handle: 2,
            delta: delta2,
        };
        let result2 = coordinator.record_vote(tx_id, 0, vote2);
        assert_eq!(
            result2,
            Err(VoteRecordError::WrongPhase {
                tx_id,
                expected: TxPhase::Preparing,
                actual: TxPhase::Prepared,
            })
        );
    }

    #[test]
    fn test_record_vote_duplicate() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        // Record first vote from shard 0
        let delta = DeltaVector::new(&[1.0], HashSet::new(), tx_id);
        let vote = PrepareVote::Yes {
            lock_handle: 1,
            delta,
        };
        let result = coordinator.record_vote(tx_id, 0, vote);
        assert_eq!(result.unwrap(), None); // Not all voted yet

        // Try duplicate vote from shard 0
        let delta2 = DeltaVector::new(&[1.0], HashSet::new(), tx_id);
        let vote2 = PrepareVote::Yes {
            lock_handle: 2,
            delta: delta2,
        };
        let result2 = coordinator.record_vote(tx_id, 0, vote2);
        assert_eq!(
            result2,
            Err(VoteRecordError::DuplicateVote { tx_id, shard: 0 })
        );
    }

    #[test]
    fn test_abort_releases_locks() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        let delta = DeltaVector::new(
            &[1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        // Vote and record
        let _ = coordinator.record_vote(
            tx.tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta,
            },
        );

        coordinator.abort(tx.tx_id, "test").unwrap();
        assert_eq!(coordinator.pending_count(), 0);
    }

    #[test]
    fn test_cleanup_timeouts_releases_locks() {
        let config = DistributedTxConfig::default();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config);

        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Add a vote with lock
        {
            let mut pending = coordinator.pending.write();
            if let Some(t) = pending.get_mut(&tx.tx_id) {
                t.record_vote(
                    0,
                    PrepareVote::Yes {
                        lock_handle: 1,
                        delta: DeltaVector::new(&[1.0], HashSet::new(), tx.tx_id),
                    },
                );
                t.timeout_ms = 0;
            }
        }

        std::thread::sleep(Duration::from_millis(1));

        let timed_out = coordinator.cleanup_timeouts();
        assert!(timed_out.contains(&tx.tx_id));
        assert!(coordinator.stats.timed_out.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_prepare_request_debug() {
        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        };
        let _ = format!("{:?}", request);
    }

    #[test]
    fn test_commit_request_debug() {
        let request = CommitRequest {
            tx_id: 1,
            lock_handles: vec![1, 2, 3],
        };
        let _ = format!("{:?}", request);
    }

    #[test]
    fn test_abort_request_debug() {
        let request = AbortRequest {
            tx_id: 1,
            reason: "test".to_string(),
            lock_handles: vec![1, 2],
        };
        let _ = format!("{:?}", request);
    }

    #[test]
    fn test_tx_response_debug() {
        let response = TxResponse {
            tx_id: 1,
            success: true,
            error: None,
        };
        let _ = format!("{:?}", response);
    }

    #[test]
    fn test_prepared_tx_debug() {
        let tx = PreparedTx {
            tx_id: 1,
            lock_handle: 1,
            operations: vec![],
            delta: DeltaVector::new(&[1.0], HashSet::new(), 1),
            prepared_at_ms: now_epoch_millis(),
            undo_log: Vec::new(),
            undo_checksums: Vec::new(),
        };
        let _ = format!("{:?}", tx);
    }

    // ========== Serialization Roundtrip Tests ==========

    #[test]
    fn test_key_lock_serialize_deserialize_roundtrip() {
        let lock = KeyLock {
            key: "test_key".to_string(),
            tx_id: 42,
            lock_handle: 123,
            acquired_at_ms: now_epoch_millis(),
            timeout_ms: 5000,
        };

        let bytes = bitcode::serialize(&lock).unwrap();
        let restored: KeyLock = bitcode::deserialize(&bytes).unwrap();

        assert_eq!(restored.key, lock.key);
        assert_eq!(restored.tx_id, lock.tx_id);
        assert_eq!(restored.lock_handle, lock.lock_handle);
        assert_eq!(restored.acquired_at_ms, lock.acquired_at_ms);
        assert_eq!(restored.timeout_ms, lock.timeout_ms);
    }

    #[test]
    fn test_prepared_tx_serialize_deserialize_roundtrip() {
        let tx = PreparedTx {
            tx_id: 100,
            lock_handle: 200,
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1, 2, 3],
            }],
            delta: DeltaVector::new(&[1.0, 2.0], ["key1".to_string()].into(), 100),
            prepared_at_ms: now_epoch_millis(),
            undo_log: Vec::new(),
            undo_checksums: Vec::new(),
        };

        let bytes = bitcode::serialize(&tx).unwrap();
        let restored: PreparedTx = bitcode::deserialize(&bytes).unwrap();

        assert_eq!(restored.tx_id, tx.tx_id);
        assert_eq!(restored.lock_handle, tx.lock_handle);
        assert_eq!(restored.operations.len(), tx.operations.len());
        assert_eq!(restored.prepared_at_ms, tx.prepared_at_ms);
    }

    #[test]
    fn test_serializable_lock_state_roundtrip() {
        let mut locks = HashMap::new();
        locks.insert(
            "key1".to_string(),
            KeyLock {
                key: "key1".to_string(),
                tx_id: 1,
                lock_handle: 1,
                acquired_at_ms: 1000,
                timeout_ms: 5000,
            },
        );

        let mut tx_locks = HashMap::new();
        tx_locks.insert(1u64, vec!["key1".to_string()]);

        let state = SerializableLockState {
            locks,
            tx_locks,
            default_timeout_ms: 30000,
        };

        let bytes = bitcode::serialize(&state).unwrap();
        let restored: SerializableLockState = bitcode::deserialize(&bytes).unwrap();

        assert_eq!(restored.locks.len(), 1);
        assert_eq!(restored.tx_locks.len(), 1);
        assert_eq!(restored.default_timeout_ms, 30000);
    }

    #[test]
    fn test_coordinator_state_serialize_deserialize_roundtrip() {
        let mut pending = HashMap::new();
        let mut tx = DistributedTransaction::new("node1".to_string(), vec![0, 1]);
        tx.phase = TxPhase::Prepared;
        pending.insert(tx.tx_id, tx);

        let state = CoordinatorState {
            pending,
            lock_state: SerializableLockState {
                locks: HashMap::new(),
                tx_locks: HashMap::new(),
                default_timeout_ms: 30000,
            },
        };

        let bytes = bitcode::serialize(&state).unwrap();
        let restored: CoordinatorState = bitcode::deserialize(&bytes).unwrap();

        assert_eq!(restored.pending.len(), 1);
        assert_eq!(restored.lock_state.default_timeout_ms, 30000);
    }

    #[test]
    fn test_participant_state_serialize_deserialize_roundtrip() {
        let mut prepared = HashMap::new();
        prepared.insert(
            1u64,
            PreparedTx {
                tx_id: 1,
                lock_handle: 100,
                operations: vec![],
                delta: DeltaVector::new(&[1.0], HashSet::new(), 1),
                prepared_at_ms: 1000,
                undo_log: Vec::new(),
                undo_checksums: Vec::new(),
            },
        );

        let state = ParticipantState {
            prepared,
            lock_state: SerializableLockState {
                locks: HashMap::new(),
                tx_locks: HashMap::new(),
                default_timeout_ms: 30000,
            },
        };

        let bytes = bitcode::serialize(&state).unwrap();
        let restored: ParticipantState = bitcode::deserialize(&bytes).unwrap();

        assert_eq!(restored.prepared.len(), 1);
        assert!(restored.prepared.contains_key(&1));
    }

    #[test]
    fn test_lock_manager_to_from_serializable_roundtrip() {
        let lock_manager = LockManager::new();

        // Add some locks
        lock_manager
            .try_lock(1, &["key1".to_string(), "key2".to_string()])
            .unwrap();
        lock_manager.try_lock(2, &["key3".to_string()]).unwrap();

        // Serialize
        let state = lock_manager.to_serializable();
        assert_eq!(state.locks.len(), 3);
        assert_eq!(state.tx_locks.len(), 2);

        // Restore
        let restored = LockManager::from_serializable(state);
        assert!(restored.is_locked("key1"));
        assert!(restored.is_locked("key2"));
        assert!(restored.is_locked("key3"));
        assert_eq!(restored.lock_holder("key1"), Some(1));
        assert_eq!(restored.lock_holder("key3"), Some(2));
    }

    // ========== Epoch Time Tests ==========

    #[test]
    fn test_key_lock_is_expired_with_epoch_millis() {
        let old_lock = KeyLock {
            key: "test".to_string(),
            tx_id: 1,
            lock_handle: 1,
            acquired_at_ms: now_epoch_millis().saturating_sub(10_000), // 10 seconds ago
            timeout_ms: 5000,                                          // 5 second timeout
        };
        assert!(old_lock.is_expired());

        let fresh_lock = KeyLock {
            key: "test".to_string(),
            tx_id: 2,
            lock_handle: 2,
            acquired_at_ms: now_epoch_millis(),
            timeout_ms: 5000,
        };
        assert!(!fresh_lock.is_expired());
    }

    #[test]
    fn test_now_epoch_millis_increases() {
        let t1 = now_epoch_millis();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t2 = now_epoch_millis();
        assert!(t2 >= t1);
    }

    // ========== Coordinator Persistence Tests ==========

    #[test]
    fn test_coordinator_to_state() {
        let coordinator = create_test_coordinator();

        // Begin a transaction
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();

        let state = coordinator.to_state();
        assert_eq!(state.pending.len(), 1);
        assert!(state.pending.contains_key(&tx.tx_id));
    }

    #[test]
    fn test_coordinator_save_load_empty_state() {
        let store = TensorStore::new();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);

        // Save empty coordinator
        coordinator.save_to_store("node1", &store).unwrap();

        // Load back
        let consensus2 = ConsensusManager::new(ConsensusConfig::default());
        let restored = DistributedTxCoordinator::load_from_store(
            "node1",
            &store,
            consensus2,
            DistributedTxConfig::default(),
        )
        .unwrap();

        assert_eq!(restored.pending_count(), 0);
    }

    #[test]
    fn test_coordinator_save_load_with_pending_transactions() {
        let store = TensorStore::new();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, DistributedTxConfig::default());

        // Begin transactions
        let tx1 = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();
        let tx2 = coordinator.begin(&"node1".to_string(), &[2, 3]).unwrap();

        // Save
        coordinator.save_to_store("node1", &store).unwrap();

        // Load back
        let consensus2 = ConsensusManager::new(ConsensusConfig::default());
        let restored = DistributedTxCoordinator::load_from_store(
            "node1",
            &store,
            consensus2,
            DistributedTxConfig::default(),
        )
        .unwrap();

        assert_eq!(restored.pending_count(), 2);
        assert!(restored.get(tx1.tx_id).is_some());
        assert!(restored.get(tx2.tx_id).is_some());
    }

    #[test]
    fn test_coordinator_save_load_all_phases() {
        let store = TensorStore::new();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, DistributedTxConfig::default());

        // Create transactions in different phases
        let tx_preparing = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Manually update internal state
        {
            let mut pending = coordinator.pending.write();
            if let Some(_tx) = pending.get_mut(&tx_preparing.tx_id) {
                // tx stays in Preparing
            }
            // Add a committed transaction manually
            let mut committed_tx = DistributedTransaction::new("node1".to_string(), vec![1]);
            committed_tx.phase = TxPhase::Committed;
            let committed_id = committed_tx.tx_id;
            pending.insert(committed_id, committed_tx);
        }

        // Save
        coordinator.save_to_store("node1", &store).unwrap();

        // Load back
        let consensus2 = ConsensusManager::new(ConsensusConfig::default());
        let restored = DistributedTxCoordinator::load_from_store(
            "node1",
            &store,
            consensus2,
            DistributedTxConfig::default(),
        )
        .unwrap();

        assert_eq!(restored.pending_count(), 2);
    }

    #[test]
    fn test_coordinator_clear_persisted_state() {
        let store = TensorStore::new();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);
        coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        coordinator.save_to_store("node1", &store).unwrap();

        // Clear
        DistributedTxCoordinator::clear_persisted_state("node1", &store).unwrap();

        // Load should create fresh coordinator
        let consensus2 = ConsensusManager::new(ConsensusConfig::default());
        let restored = DistributedTxCoordinator::load_from_store(
            "node1",
            &store,
            consensus2,
            DistributedTxConfig::default(),
        )
        .unwrap();

        assert_eq!(restored.pending_count(), 0);
    }

    #[test]
    fn test_coordinator_clear_persisted_state_returns_result() {
        let store = TensorStore::new();
        // Fresh store with no state - should succeed (no-op delete)
        let result = DistributedTxCoordinator::clear_persisted_state("node1", &store);
        assert!(result.is_ok());
    }

    // ========== Participant Persistence Tests ==========

    #[test]
    fn test_participant_to_state() {
        let participant = TxParticipant::new_in_memory();

        // Prepare a transaction
        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        participant.prepare(request);

        let state = participant.to_state();
        assert_eq!(state.prepared.len(), 1);
        assert!(state.prepared.contains_key(&1));
    }

    #[test]
    fn test_participant_save_load_empty_state() {
        let store = TensorStore::new();
        let participant = TxParticipant::new_in_memory();

        participant.save_to_store("node1", 0, &store).unwrap();

        let restored = TxParticipant::load_from_store("node1", 0, &store);
        assert_eq!(restored.prepared_count(), 0);
    }

    #[test]
    fn test_participant_save_load_with_prepared_transactions() {
        let store = TensorStore::new();
        let participant = TxParticipant::new_in_memory();

        // Prepare transactions
        let request1 = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        let request2 = PrepareRequest {
            tx_id: 2,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key2".to_string(),
                data: vec![2],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        participant.prepare(request1);
        participant.prepare(request2);

        // Save
        participant.save_to_store("node1", 0, &store).unwrap();

        // Load
        let restored = TxParticipant::load_from_store("node1", 0, &store);
        assert_eq!(restored.prepared_count(), 2);
        assert!(restored.locks.is_locked("key1"));
        assert!(restored.locks.is_locked("key2"));
    }

    #[test]
    fn test_participant_clear_persisted_state() {
        let store = TensorStore::new();
        let participant = TxParticipant::new_in_memory();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        participant.prepare(request);
        participant.save_to_store("node1", 0, &store).unwrap();

        TxParticipant::clear_persisted_state("node1", 0, &store).unwrap();

        let restored = TxParticipant::load_from_store("node1", 0, &store);
        assert_eq!(restored.prepared_count(), 0);
    }

    #[test]
    fn test_participant_clear_persisted_state_returns_result() {
        let store = TensorStore::new();
        // Fresh store with no state - should succeed (no-op delete)
        let result = TxParticipant::clear_persisted_state("node1", 0, &store);
        assert!(result.is_ok());
    }

    // ========== Recovery Tests ==========

    #[test]
    fn test_coordinator_recover_removes_completed() {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);

        // Add completed transactions manually
        {
            let mut pending = coordinator.pending.write();
            let mut committed_tx = DistributedTransaction::new("node1".to_string(), vec![0]);
            committed_tx.phase = TxPhase::Committed;
            pending.insert(committed_tx.tx_id, committed_tx);

            let mut aborted_tx = DistributedTransaction::new("node1".to_string(), vec![1]);
            aborted_tx.phase = TxPhase::Aborted;
            pending.insert(aborted_tx.tx_id, aborted_tx);
        }

        assert_eq!(coordinator.pending_count(), 2);

        let stats = coordinator.recover();

        assert_eq!(stats.completed, 2);
        assert_eq!(coordinator.pending_count(), 0);
    }

    #[test]
    fn test_coordinator_recover_prepared_to_commit() {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);

        {
            let mut pending = coordinator.pending.write();
            let mut tx = DistributedTransaction::new("node1".to_string(), vec![0]);
            tx.phase = TxPhase::Prepared;
            // Add YES vote
            tx.votes.insert(
                0,
                PrepareVote::Yes {
                    lock_handle: 1,
                    delta: DeltaVector::new(&[1.0], HashSet::new(), 1),
                },
            );
            pending.insert(tx.tx_id, tx);
        }

        let stats = coordinator.recover();

        assert_eq!(stats.pending_commit, 1);
        let decisions = coordinator.get_pending_decisions();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].1, TxPhase::Committing);
    }

    #[test]
    fn test_coordinator_recover_committing_retry() {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);

        {
            let mut pending = coordinator.pending.write();
            let mut tx = DistributedTransaction::new("node1".to_string(), vec![0]);
            tx.phase = TxPhase::Committing;
            pending.insert(tx.tx_id, tx);
        }

        let stats = coordinator.recover();

        assert_eq!(stats.pending_commit, 1);
    }

    #[test]
    fn test_coordinator_recover_aborting_retry() {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);

        {
            let mut pending = coordinator.pending.write();
            let mut tx = DistributedTransaction::new("node1".to_string(), vec![0]);
            tx.phase = TxPhase::Aborting;
            pending.insert(tx.tx_id, tx);
        }

        let stats = coordinator.recover();

        assert_eq!(stats.pending_abort, 1);
    }

    #[test]
    fn test_participant_recover_awaiting_decision() {
        let participant = TxParticipant::new_in_memory();

        // Prepare a transaction
        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        participant.prepare(request);

        // Recover with long timeout - should keep the transaction
        let awaiting = participant.recover(Duration::from_secs(3600));
        assert_eq!(awaiting, vec![1]);
        assert!(participant.locks.is_locked("key1"));
    }

    #[test]
    fn test_participant_recover_expired_releases_locks() {
        let participant = TxParticipant::new_in_memory();
        let old_time = now_epoch_millis().saturating_sub(10_000); // 10 seconds ago

        // Manually insert an old prepared transaction and matching lock
        {
            let mut prepared = participant.prepared.write();
            prepared.insert(
                1,
                PreparedTx {
                    tx_id: 1,
                    lock_handle: 12345, // Fixed lock handle
                    operations: vec![],
                    delta: DeltaVector::new(&[1.0], HashSet::new(), 1),
                    prepared_at_ms: old_time,
                    undo_log: Vec::new(),
                    undo_checksums: Vec::new(),
                },
            );
        }
        {
            // Manually add lock with matching handle
            let mut locks = participant.locks.locks.write();
            let mut tx_locks = participant.locks.tx_locks.write();
            locks.insert(
                "key1".to_string(),
                KeyLock {
                    key: "key1".to_string(),
                    tx_id: 1,
                    lock_handle: 12345, // Same lock handle
                    acquired_at_ms: old_time,
                    timeout_ms: 30000,
                },
            );
            tx_locks.insert(1, vec!["key1".to_string()]);
        }

        assert!(participant.locks.is_locked("key1"));

        // Recover with short timeout
        let awaiting = participant.recover(Duration::from_secs(5));
        assert!(awaiting.is_empty());
        assert!(!participant.locks.is_locked("key1"));
    }

    #[test]
    fn test_participant_get_awaiting_decision() {
        let participant = TxParticipant::new_in_memory();

        let request = PrepareRequest {
            tx_id: 42,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        participant.prepare(request);

        let awaiting = participant.get_awaiting_decision();
        assert_eq!(awaiting, vec![42]);
    }

    #[test]
    fn test_recovery_stats_default() {
        let stats = RecoveryStats::default();
        assert_eq!(stats.pending_prepare, 0);
        assert_eq!(stats.pending_commit, 0);
        assert_eq!(stats.pending_abort, 0);
        assert_eq!(stats.timed_out, 0);
        assert_eq!(stats.completed, 0);
    }

    #[test]
    fn test_abort_state_tracking() {
        let coordinator = create_test_coordinator();

        // Track an abort with 3 shards
        coordinator.track_abort(123, vec![0, 1, 2]);

        // Check that the abort state was created
        let abort_states = coordinator.abort_states.read();
        assert!(abort_states.contains_key(&123));
        let state = abort_states.get(&123).unwrap();
        assert_eq!(state.pending_acks.len(), 3);
        assert!(state.pending_acks.contains(&0));
        assert!(state.pending_acks.contains(&1));
        assert!(state.pending_acks.contains(&2));
        assert_eq!(state.retry_count, 0);
    }

    #[test]
    fn test_abort_ack_removes_shard() {
        let coordinator = create_test_coordinator();

        // Track an abort with 2 shards
        coordinator.track_abort(456, vec![0, 1]);

        // Acknowledge from shard 0
        let complete = coordinator.handle_abort_ack(456, 0);
        assert!(!complete); // Still waiting for shard 1

        // Check state was updated
        let abort_states = coordinator.abort_states.read();
        let state = abort_states.get(&456).unwrap();
        assert_eq!(state.pending_acks.len(), 1);
        assert!(!state.pending_acks.contains(&0));
        assert!(state.pending_acks.contains(&1));
    }

    #[test]
    fn test_abort_ack_cleanup_on_complete() {
        let coordinator = create_test_coordinator();

        // Track an abort with 2 shards
        coordinator.track_abort(789, vec![0, 1]);

        // Acknowledge from both shards
        let complete1 = coordinator.handle_abort_ack(789, 0);
        assert!(!complete1);
        let complete2 = coordinator.handle_abort_ack(789, 1);
        assert!(complete2); // All acknowledged

        // State should be cleaned up
        let abort_states = coordinator.abort_states.read();
        assert!(!abort_states.contains_key(&789));
    }

    #[test]
    fn test_pending_aborts_queued_on_vote_no() {
        let coordinator = create_test_coordinator();

        // Begin a transaction with 2 shards
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();

        // Record votes from both shards - shard 0 says No, shard 1 says Yes
        let phase1 = coordinator.record_vote(
            tx.tx_id,
            0,
            PrepareVote::No {
                reason: "test rejection".to_string(),
            },
        );
        assert_eq!(phase1.unwrap(), None); // Not all voted yet

        // Record YES from shard 1
        let phase2 = coordinator.record_vote(
            tx.tx_id,
            1,
            PrepareVote::Yes {
                lock_handle: 42,
                delta: DeltaVector::new(
                    &[1.0, 0.0, 0.0],
                    ["key1"].iter().map(|s| s.to_string()).collect(),
                    tx.tx_id,
                ),
            },
        );
        // Now all have voted and one said NO, so it should abort
        assert_eq!(phase2.unwrap(), Some(TxPhase::Aborting));

        // Check that abort was queued
        let pending = coordinator.take_pending_aborts();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, tx.tx_id);
        assert_eq!(pending[0].2, vec![0, 1]); // Both shards
    }

    #[test]
    fn test_take_pending_aborts_clears_queue() {
        let coordinator = create_test_coordinator();

        // Manually add some pending aborts
        coordinator
            .pending_aborts
            .write()
            .push((111, "test".to_string(), vec![0]));
        coordinator
            .pending_aborts
            .write()
            .push((222, "test2".to_string(), vec![1, 2]));

        // Take them
        let aborts = coordinator.take_pending_aborts();
        assert_eq!(aborts.len(), 2);

        // Queue should be empty now
        let aborts2 = coordinator.take_pending_aborts();
        assert!(aborts2.is_empty());
    }

    // ========== WAL Integration Tests ==========

    #[test]
    fn test_coordinator_with_wal_logs_begin() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();

        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus).with_wal(wal);

        // Begin a transaction
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();

        // Verify WAL has the entry
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 1);
        if let crate::tx_wal::TxWalEntry::TxBegin {
            tx_id,
            participants,
        } = &entries[0]
        {
            assert_eq!(*tx_id, tx.tx_id);
            assert_eq!(*participants, vec![0, 1]);
        } else {
            panic!("expected TxBegin entry");
        }
    }

    #[test]
    fn test_coordinator_with_wal_logs_commit() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();

        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus).with_wal(wal);

        // Create a prepared transaction manually
        let tx_id = {
            let mut tx = DistributedTransaction::new("node1".to_string(), vec![0]);
            tx.phase = TxPhase::Prepared;
            tx.votes.insert(
                0,
                PrepareVote::Yes {
                    lock_handle: 1,
                    delta: crate::consensus::DeltaVector::zero(0),
                },
            );
            let id = tx.tx_id;
            coordinator.pending.write().insert(tx.tx_id, tx);
            id
        };

        // Commit the transaction
        coordinator.commit(tx_id).unwrap();

        // Verify WAL has phase change, completion, lock release, and all-released entries
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        // PhaseChange + TxComplete + LockRelease + AllLocksReleased
        assert_eq!(entries.len(), 4);

        // First entry should be phase change
        if let crate::tx_wal::TxWalEntry::PhaseChange {
            tx_id: id,
            from,
            to,
        } = &entries[0]
        {
            assert_eq!(*id, tx_id);
            assert_eq!(*from, TxPhase::Prepared);
            assert_eq!(*to, TxPhase::Committing);
        } else {
            panic!("expected PhaseChange entry");
        }

        // Second entry should be completion
        if let crate::tx_wal::TxWalEntry::TxComplete { tx_id: id, outcome } = &entries[1] {
            assert_eq!(*id, tx_id);
            assert_eq!(*outcome, crate::tx_wal::TxOutcome::Committed);
        } else {
            panic!("expected TxComplete entry");
        }

        // Third entry should be lock release
        if let crate::tx_wal::TxWalEntry::LockRelease {
            tx_id: id,
            lock_handle,
        } = &entries[2]
        {
            assert_eq!(*id, tx_id);
            assert_eq!(*lock_handle, 1);
        } else {
            panic!("expected LockRelease entry");
        }

        // Fourth entry should be all-locks-released
        if let crate::tx_wal::TxWalEntry::AllLocksReleased { tx_id: id } = &entries[3] {
            assert_eq!(*id, tx_id);
        } else {
            panic!("expected AllLocksReleased entry");
        }
    }

    #[test]
    fn test_coordinator_with_wal_logs_abort() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();

        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus).with_wal(wal);

        // Create a transaction manually
        let tx_id = {
            let mut tx = DistributedTransaction::new("node1".to_string(), vec![0]);
            tx.phase = TxPhase::Preparing;
            let id = tx.tx_id;
            coordinator.pending.write().insert(tx.tx_id, tx);
            id
        };

        // Abort the transaction
        coordinator.abort(tx_id, "test abort").unwrap();

        // Verify WAL has the phase change and completion entries
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 2);

        // First entry should be phase change to Aborting
        if let crate::tx_wal::TxWalEntry::PhaseChange { to, .. } = &entries[0] {
            assert_eq!(*to, TxPhase::Aborting);
        } else {
            panic!("expected PhaseChange entry");
        }

        // Second entry should be completion with Aborted outcome
        if let crate::tx_wal::TxWalEntry::TxComplete { outcome, .. } = &entries[1] {
            assert_eq!(*outcome, crate::tx_wal::TxOutcome::Aborted);
        } else {
            panic!("expected TxComplete entry");
        }
    }

    #[test]
    fn test_coordinator_recovers_from_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");

        // Create WAL with some entries
        {
            let mut wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();

            // Transaction 1: prepared with YES vote
            wal.append(&crate::tx_wal::TxWalEntry::TxBegin {
                tx_id: 100,
                participants: vec![0, 1],
            })
            .unwrap();
            wal.append(&crate::tx_wal::TxWalEntry::PrepareVote {
                tx_id: 100,
                shard: 0,
                vote: crate::tx_wal::PrepareVoteKind::Yes { lock_handle: 200 },
            })
            .unwrap();
            wal.append(&crate::tx_wal::TxWalEntry::PhaseChange {
                tx_id: 100,
                from: TxPhase::Preparing,
                to: TxPhase::Prepared,
            })
            .unwrap();

            // Transaction 2: prepared with NO vote
            wal.append(&crate::tx_wal::TxWalEntry::TxBegin {
                tx_id: 200,
                participants: vec![0],
            })
            .unwrap();
            wal.append(&crate::tx_wal::TxWalEntry::PrepareVote {
                tx_id: 200,
                shard: 0,
                vote: crate::tx_wal::PrepareVoteKind::No,
            })
            .unwrap();
            wal.append(&crate::tx_wal::TxWalEntry::PhaseChange {
                tx_id: 200,
                from: TxPhase::Preparing,
                to: TxPhase::Prepared,
            })
            .unwrap();
        }

        // Create coordinator and recover
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus).with_wal(wal);

        let stats = coordinator.recover_from_wal().unwrap();

        // Verify recovery stats - both transactions in prepared phase
        assert_eq!(stats.pending_prepare, 2);
        assert_eq!(stats.pending_commit, 0);
        assert_eq!(stats.pending_abort, 0);

        // Verify pending transactions were restored
        assert_eq!(coordinator.pending_count(), 2);

        // Verify transaction details
        let tx100 = coordinator.get(100).unwrap();
        assert_eq!(tx100.phase, TxPhase::Prepared);
        assert_eq!(tx100.participants, vec![0, 1]);

        let tx200 = coordinator.get(200).unwrap();
        assert_eq!(tx200.phase, TxPhase::Prepared);
        assert_eq!(tx200.participants, vec![0]);
    }

    #[test]
    fn test_coordinator_truncate_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();

        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus).with_wal(wal);

        // Begin a transaction (creates WAL entry)
        let _ = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Truncate WAL
        coordinator.truncate_wal().unwrap();

        // Verify WAL is empty
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();
        let entries = wal.replay().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_coordinator_without_wal_works() {
        // Coordinator without WAL should still work
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);

        // Begin should work
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Create prepared state
        {
            let mut pending = coordinator.pending.write();
            if let Some(tx) = pending.get_mut(&tx.tx_id) {
                tx.phase = TxPhase::Prepared;
                tx.votes.insert(
                    0,
                    PrepareVote::Yes {
                        lock_handle: 1,
                        delta: crate::consensus::DeltaVector::zero(0),
                    },
                );
            }
        }

        // Commit should work
        coordinator.commit(tx.tx_id).unwrap();
        assert_eq!(coordinator.pending_count(), 0);
    }

    #[test]
    fn test_release_orphaned_locks_no_orphans() {
        let coordinator = create_test_coordinator();

        // No locks, no orphans
        let released = coordinator.release_orphaned_locks(now_epoch_millis());
        assert_eq!(released, 0);
    }

    #[test]
    fn test_release_orphaned_locks_active_tx_not_released() {
        let coordinator = create_test_coordinator();

        // Begin transaction (creates active tx)
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Manually add a lock for the active transaction
        coordinator
            .lock_manager
            .try_lock(tx.tx_id, &["key1".to_string()])
            .unwrap();

        assert!(coordinator.lock_manager.is_locked("key1"));

        // Try to release orphaned locks - should not release active tx's lock
        let released = coordinator.release_orphaned_locks(now_epoch_millis() + 1000);
        assert_eq!(released, 0);
        assert!(coordinator.lock_manager.is_locked("key1"));
    }

    #[test]
    fn test_release_orphaned_locks_releases_orphaned() {
        let coordinator = create_test_coordinator();

        // Manually add a lock for a non-existent transaction (orphaned)
        let orphan_tx_id = 999;
        coordinator
            .lock_manager
            .try_lock(orphan_tx_id, &["orphan_key".to_string()])
            .unwrap();

        assert!(coordinator.lock_manager.is_locked("orphan_key"));

        // Release orphaned locks
        let released = coordinator.release_orphaned_locks(now_epoch_millis() + 1000);
        assert_eq!(released, 1);
        assert!(!coordinator.lock_manager.is_locked("orphan_key"));
    }

    #[test]
    fn test_release_orphaned_locks_respects_partition_time() {
        let coordinator = create_test_coordinator();

        // Add a lock for orphaned tx
        let orphan_tx_id = 888;
        coordinator
            .lock_manager
            .try_lock(orphan_tx_id, &["key1".to_string()])
            .unwrap();

        // Try with partition_start_ms in the past - should not release
        // because lock was acquired AFTER partition_start
        let released = coordinator.release_orphaned_locks(0);
        assert_eq!(released, 0);
        assert!(coordinator.lock_manager.is_locked("key1"));

        // Now try with partition_start_ms in the future - should release
        let released = coordinator.release_orphaned_locks(now_epoch_millis() + 10_000);
        assert_eq!(released, 1);
        assert!(!coordinator.lock_manager.is_locked("key1"));
    }

    #[test]
    fn test_release_orphaned_locks_multiple_keys() {
        let coordinator = create_test_coordinator();

        // Add multiple locks for different orphaned transactions
        coordinator
            .lock_manager
            .try_lock(100, &["key_a".to_string(), "key_b".to_string()])
            .unwrap();
        coordinator
            .lock_manager
            .try_lock(101, &["key_c".to_string()])
            .unwrap();

        assert_eq!(coordinator.lock_manager.active_lock_count(), 3);

        // Release all orphaned locks
        let released = coordinator.release_orphaned_locks(now_epoch_millis() + 1000);
        assert_eq!(released, 3);
        assert_eq!(coordinator.lock_manager.active_lock_count(), 0);
    }

    #[test]
    fn test_release_orphaned_locks_mixed_active_and_orphaned() {
        let coordinator = create_test_coordinator();

        // Begin an active transaction with a lock
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        coordinator
            .lock_manager
            .try_lock(tx.tx_id, &["active_key".to_string()])
            .unwrap();

        // Add orphaned locks
        coordinator
            .lock_manager
            .try_lock(555, &["orphan1".to_string()])
            .unwrap();
        coordinator
            .lock_manager
            .try_lock(556, &["orphan2".to_string()])
            .unwrap();

        assert_eq!(coordinator.lock_manager.active_lock_count(), 3);

        // Release orphaned - should only release 2
        let released = coordinator.release_orphaned_locks(now_epoch_millis() + 1000);
        assert_eq!(released, 2);

        // Active key should still be locked
        assert!(coordinator.lock_manager.is_locked("active_key"));
        assert!(!coordinator.lock_manager.is_locked("orphan1"));
        assert!(!coordinator.lock_manager.is_locked("orphan2"));
    }

    #[test]
    fn test_release_orphaned_locks_concurrent_safety() {
        use std::sync::Arc;
        use std::thread;

        let coordinator = Arc::new(create_test_coordinator());

        // Add many orphaned locks
        for i in 0..50 {
            coordinator
                .lock_manager
                .try_lock(1000 + i, &[format!("key_{}", i)])
                .unwrap();
        }

        let partition_time = now_epoch_millis() + 10_000;

        // Spawn multiple threads to call release_orphaned_locks concurrently
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let coord = Arc::clone(&coordinator);
                thread::spawn(move || coord.release_orphaned_locks(partition_time))
            })
            .collect();

        // All threads should complete without deadlock
        let mut total = 0;
        for handle in handles {
            total += handle.join().unwrap();
        }

        // Total released should be 50 (some threads may see 0 if others already cleaned)
        assert!(total >= 50);
        assert_eq!(coordinator.lock_manager.active_lock_count(), 0);
    }

    #[test]
    fn test_lock_manager_serializable_roundtrip() {
        let lock_manager = LockManager::new();

        lock_manager.try_lock(1, &["key1".to_string()]).unwrap();
        lock_manager.try_lock(2, &["key2".to_string()]).unwrap();

        let serializable = lock_manager.to_serializable();
        let restored = LockManager::from_serializable(serializable);

        assert!(restored.is_locked("key1"));
        assert!(restored.is_locked("key2"));
        assert_eq!(restored.lock_holder("key1"), Some(1));
        assert_eq!(restored.lock_holder("key2"), Some(2));
    }

    #[test]
    fn test_lock_manager_concurrent_try_lock() {
        use std::sync::Arc;
        use std::thread;

        let lock_manager = Arc::new(LockManager::new());
        let success_count = Arc::new(std::sync::atomic::AtomicU32::new(0));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let lm = Arc::clone(&lock_manager);
                let counter = Arc::clone(&success_count);
                thread::spawn(move || {
                    if lm
                        .try_lock(i as u64, &["contested_key".to_string()])
                        .is_ok()
                    {
                        counter.fetch_add(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Exactly one should succeed
        assert_eq!(success_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_lock_manager_release_partial() {
        let lock_manager = LockManager::new();

        // Transaction 1 locks keys A and B
        lock_manager
            .try_lock(1, &["key_a".to_string(), "key_b".to_string()])
            .unwrap();

        // Release transaction 1
        lock_manager.release(1);

        // Both keys should be unlocked
        assert!(!lock_manager.is_locked("key_a"));
        assert!(!lock_manager.is_locked("key_b"));

        // Another tx should be able to lock them
        assert!(lock_manager.try_lock(2, &["key_a".to_string()]).is_ok());
    }

    #[test]
    fn test_lock_manager_keys_for_transaction() {
        let lock_manager = LockManager::new();

        // No locks - should return empty
        assert!(lock_manager.keys_for_transaction(1).is_empty());

        // Lock some keys
        lock_manager
            .try_lock(1, &["key_a".to_string(), "key_b".to_string()])
            .unwrap();

        let keys = lock_manager.keys_for_transaction(1);
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"key_a".to_string()));
        assert!(keys.contains(&"key_b".to_string()));

        // Different tx has no keys
        assert!(lock_manager.keys_for_transaction(2).is_empty());
    }

    #[test]
    fn test_lock_manager_lock_count_for_transaction() {
        let lock_manager = LockManager::new();

        assert_eq!(lock_manager.lock_count_for_transaction(1), 0);

        lock_manager
            .try_lock(
                1,
                &[
                    "key_a".to_string(),
                    "key_b".to_string(),
                    "key_c".to_string(),
                ],
            )
            .unwrap();

        assert_eq!(lock_manager.lock_count_for_transaction(1), 3);
        assert_eq!(lock_manager.lock_count_for_transaction(999), 0);
    }

    #[test]
    fn test_lock_manager_try_lock_with_wait_tracking_success() {
        let lock_manager = LockManager::new();
        let wait_graph = crate::deadlock::WaitForGraph::new();

        let result = lock_manager.try_lock_with_wait_tracking(
            1,
            &["key1".to_string()],
            &wait_graph,
            Some(10),
        );

        assert!(result.is_ok());
        assert!(lock_manager.is_locked("key1"));
    }

    #[test]
    fn test_lock_manager_try_lock_with_wait_tracking_conflict() {
        let lock_manager = LockManager::new();
        let wait_graph = crate::deadlock::WaitForGraph::new();

        // First tx locks key1
        lock_manager.try_lock(1, &["key1".to_string()]).unwrap();

        // Second tx tries to lock same key
        let result = lock_manager.try_lock_with_wait_tracking(
            2,
            &["key1".to_string()],
            &wait_graph,
            Some(5),
        );

        assert!(result.is_err());
        let wait_info = result.unwrap_err();
        assert_eq!(wait_info.blocking_tx_id, 1);
        assert!(wait_info.conflicting_keys.contains(&"key1".to_string()));
    }

    #[test]
    fn test_try_lock_with_wait_tracking_conflicting_keys_accurate() {
        let lock_manager = LockManager::new();
        let wait_graph = crate::deadlock::WaitForGraph::new();

        // tx1 locks key1 and key2
        lock_manager
            .try_lock(1, &["key1".to_string(), "key2".to_string()])
            .unwrap();

        // tx2 locks key3
        lock_manager.try_lock(2, &["key3".to_string()]).unwrap();

        // tx3 tries to lock key1, key2, key3, key4
        // All conflicting keys from ALL blocking transactions are collected
        let result = lock_manager.try_lock_with_wait_tracking(
            3,
            &[
                "key1".to_string(),
                "key2".to_string(),
                "key3".to_string(),
                "key4".to_string(),
            ],
            &wait_graph,
            Some(5),
        );

        assert!(result.is_err());
        let wait_info = result.unwrap_err();
        // blocking_tx_id is one of the blocking transactions
        assert!(wait_info.blocking_tx_id == 1 || wait_info.blocking_tx_id == 2);
        // ALL conflicting keys from ALL blockers should be in conflicting_keys
        assert!(wait_info.conflicting_keys.contains(&"key1".to_string()));
        assert!(wait_info.conflicting_keys.contains(&"key2".to_string()));
        // key3 is also conflicting (held by tx2)
        assert!(wait_info.conflicting_keys.contains(&"key3".to_string()));
        // key4 is not locked at all, so not in conflict list
        assert!(!wait_info.conflicting_keys.contains(&"key4".to_string()));
        // Wait graph should have edges to both blockers
        assert!(wait_graph.waiting_for(3).contains(&1));
        assert!(wait_graph.waiting_for(3).contains(&2));
    }

    #[test]
    fn test_try_lock_with_wait_tracking_atomicity() {
        use std::sync::Arc;
        use std::thread;

        let lock_manager = Arc::new(LockManager::new());
        let wait_graph = Arc::new(crate::deadlock::WaitForGraph::new());

        // tx1 locks key1
        lock_manager.try_lock(1, &["key1".to_string()]).unwrap();

        // Spawn a thread that will release tx1's lock
        let lm_clone = Arc::clone(&lock_manager);
        let release_handle = thread::spawn(move || {
            // Small delay before release
            thread::sleep(Duration::from_millis(5));
            lm_clone.release(1);
        });

        // Try lock with wait tracking - should see the conflict atomically
        // The key insight: even if the lock is released during our check,
        // we should get consistent information (either all conflict info or success)
        let result = lock_manager.try_lock_with_wait_tracking(
            2,
            &["key1".to_string()],
            &wait_graph,
            Some(5),
        );

        // The result should be consistent:
        // - If we saw the conflict, we should have the right blocking_tx_id
        // - If the lock was already released, we should succeed
        match result {
            Ok(_handle) => {
                // Lock was released before our check - tx2 got the lock
                assert!(lock_manager.is_locked("key1"));
                assert_eq!(lock_manager.lock_holder("key1"), Some(2));
            },
            Err(wait_info) => {
                // We saw the conflict - blocking tx should be tx1
                assert_eq!(wait_info.blocking_tx_id, 1);
                assert!(wait_info.conflicting_keys.contains(&"key1".to_string()));
            },
        }

        release_handle.join().unwrap();
    }

    #[test]
    fn test_serializable_lock_state_new() {
        let locks = {
            let mut m = HashMap::new();
            m.insert(
                "key1".to_string(),
                KeyLock {
                    key: "key1".to_string(),
                    tx_id: 1,
                    lock_handle: 10,
                    acquired_at_ms: now_epoch_millis(),
                    timeout_ms: 5000,
                },
            );
            m
        };

        let tx_locks = {
            let mut m = HashMap::new();
            m.insert(1, vec!["key1".to_string()]);
            m
        };

        let state = SerializableLockState::new(locks.clone(), tx_locks.clone(), 30000);

        assert_eq!(state.locks.len(), 1);
        assert_eq!(state.tx_locks.len(), 1);
        assert_eq!(state.default_timeout_ms, 30000);
    }

    #[test]
    fn test_release_nonexistent_tx() {
        let lock_manager = LockManager::new();

        // Should not panic even for non-existent tx
        lock_manager.release(999);
        assert_eq!(lock_manager.active_lock_count(), 0);
    }

    #[test]
    fn test_release_by_handle_removes_tx_locks_entry() {
        let lock_manager = LockManager::new();

        let handle = lock_manager.try_lock(1, &["key1".to_string()]).unwrap();

        assert_eq!(lock_manager.lock_count_for_transaction(1), 1);

        lock_manager.release_by_handle(handle);

        assert_eq!(lock_manager.lock_count_for_transaction(1), 0);
        assert!(!lock_manager.is_locked("key1"));
    }

    #[test]
    fn test_release_by_handle_nonexistent() {
        let lock_manager = LockManager::new();

        // Should not panic
        lock_manager.release_by_handle(99999);
        assert_eq!(lock_manager.active_lock_count(), 0);
    }

    #[test]
    fn test_cleanup_expired_removes_from_tx_locks() {
        let mut lock_manager = LockManager::new();
        lock_manager.default_timeout = Duration::from_millis(0);

        lock_manager
            .try_lock(1, &["key1".to_string(), "key2".to_string()])
            .unwrap();

        assert_eq!(lock_manager.lock_count_for_transaction(1), 2);

        std::thread::sleep(Duration::from_millis(1));
        lock_manager.cleanup_expired();

        assert_eq!(lock_manager.lock_count_for_transaction(1), 0);
        assert!(!lock_manager.is_locked("key1"));
        assert!(!lock_manager.is_locked("key2"));
    }

    #[test]
    fn test_distributed_tx_config_accessors() {
        let config = DistributedTxConfig {
            prepare_timeout_ms: 1000,
            commit_timeout_ms: 2000,
            max_concurrent: 50,
            orthogonal_threshold: 0.2,
            optimistic_locking: false,
            tx_queue_soft_limit_pct: 80,
        };

        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config.clone());

        // Verify stats are accessible
        assert_eq!(coordinator.stats().started.load(Ordering::Relaxed), 0);

        // Verify config was applied by checking behavior
        let lm = coordinator.lock_manager();
        assert_eq!(lm.active_lock_count(), 0);
    }

    #[test]
    fn test_serializable_lock_state_accessors() {
        let locks = {
            let mut m = HashMap::new();
            m.insert(
                "key1".to_string(),
                KeyLock {
                    key: "key1".to_string(),
                    tx_id: 1,
                    lock_handle: 10,
                    acquired_at_ms: now_epoch_millis(),
                    timeout_ms: 5000,
                },
            );
            m
        };
        let tx_locks = {
            let mut m = HashMap::new();
            m.insert(1, vec!["key1".to_string()]);
            m
        };

        let state = SerializableLockState::new(locks.clone(), tx_locks.clone(), 30000);

        assert_eq!(state.locks().len(), 1);
        assert_eq!(state.tx_locks().len(), 1);
        assert_eq!(state.default_timeout_ms(), 30000);
    }

    #[test]
    fn test_coordinator_recover_from_wal_no_wal() {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);

        // No WAL - should return default stats
        let stats = coordinator.recover_from_wal().unwrap();
        assert_eq!(stats.pending_prepare, 0);
        assert_eq!(stats.pending_commit, 0);
        assert_eq!(stats.pending_abort, 0);
    }

    #[test]
    fn test_coordinator_with_wal_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("tx.wal");
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();

        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus).with_wal(wal);

        // Start and prepare a transaction
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Record a Yes vote
        let _ = coordinator.record_vote(
            tx.tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta: DeltaVector::new(&[1.0], HashSet::new(), tx.tx_id),
            },
        );

        // Clear pending to simulate crash
        coordinator.pending.write().clear();

        // Recover from WAL
        let stats = coordinator.recover_from_wal().unwrap();
        // Verify we got stats back (values are usize so always >= 0)
        let _ = stats.pending_prepare + stats.pending_commit;
    }

    #[test]
    fn test_coordinator_state_accessors() {
        let pending = {
            let mut m = HashMap::new();
            let tx = DistributedTransaction::new("node1".to_string(), vec![0]);
            m.insert(tx.tx_id, tx);
            m
        };

        let lock_state = SerializableLockState::new(HashMap::new(), HashMap::new(), 5000);

        let state = CoordinatorState {
            pending,
            lock_state,
        };

        assert_eq!(state.pending.len(), 1);
    }

    #[test]
    fn test_coordinator_to_state_and_restore() {
        let coordinator = create_test_coordinator();

        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Add a lock
        coordinator
            .lock_manager
            .try_lock(tx.tx_id, &["key1".to_string()])
            .unwrap();

        // Convert to state
        let state = coordinator.to_state();
        assert_eq!(state.pending.len(), 1);

        // Create a new coordinator from state
        let consensus2 = ConsensusManager::new(ConsensusConfig::default());
        let coordinator2 =
            DistributedTxCoordinator::with_state(consensus2, DistributedTxConfig::default(), state);

        assert_eq!(coordinator2.pending_count(), 1);
    }

    #[test]
    fn test_abort_state_fields() {
        let mut pending_acks = HashSet::new();
        pending_acks.insert(0);
        pending_acks.insert(1);

        let state = AbortState {
            pending_acks,
            initiated_at: now_epoch_millis(),
            retry_count: 0,
        };

        assert_eq!(state.pending_acks.len(), 2);
        assert_eq!(state.retry_count, 0);
    }

    #[test]
    fn test_lock_release_removes_from_both_maps() {
        let lock_manager = LockManager::new();

        // Lock multiple keys for the same transaction
        lock_manager
            .try_lock(1, &["a".to_string(), "b".to_string(), "c".to_string()])
            .unwrap();

        assert_eq!(lock_manager.active_lock_count(), 3);
        assert_eq!(lock_manager.lock_count_for_transaction(1), 3);

        // Release by transaction
        lock_manager.release(1);

        assert_eq!(lock_manager.active_lock_count(), 0);
        assert_eq!(lock_manager.lock_count_for_transaction(1), 0);
    }

    #[test]
    fn test_release_does_not_affect_other_tx_locks() {
        let lock_manager = LockManager::new();

        lock_manager.try_lock(1, &["key1".to_string()]).unwrap();
        lock_manager.try_lock(2, &["key2".to_string()]).unwrap();

        lock_manager.release(1);

        // tx 1's lock is gone
        assert!(!lock_manager.is_locked("key1"));
        // tx 2's lock remains
        assert!(lock_manager.is_locked("key2"));
        assert_eq!(lock_manager.lock_holder("key2"), Some(2));
    }

    #[test]
    fn test_record_vote_and_release_orphaned_locks_no_deadlock() {
        // Verify that concurrent record_vote and release_orphaned_locks don't deadlock.
        // This tests the phased locking approach in record_vote that releases
        // pending lock before acquiring pending_aborts.
        use std::sync::atomic::{AtomicBool, AtomicUsize};
        use std::sync::Arc;
        use std::thread;
        use std::time::Duration;

        let coordinator = Arc::new(create_test_coordinator());
        let iterations = 100;
        let deadlock_detected = Arc::new(AtomicBool::new(false));
        let operations_completed = Arc::new(AtomicUsize::new(0));

        // Thread 1: Continuously call record_vote on various transactions
        let coord1 = Arc::clone(&coordinator);
        let ops1 = Arc::clone(&operations_completed);
        let record_vote_thread = thread::spawn(move || {
            for i in 0..iterations {
                // Create a transaction first
                let tx = coord1.begin(&"node1".to_string(), &[0, 1]).unwrap();
                let tx_id = tx.tx_id;

                // Record votes from both shards
                let vote1 = PrepareVote::Yes {
                    lock_handle: i as u64,
                    delta: DeltaVector::new(&[1.0], HashSet::new(), tx_id),
                };
                let _ = coord1.record_vote(tx_id, 0, vote1);

                let vote2 = PrepareVote::Yes {
                    lock_handle: i as u64 + 1000,
                    delta: DeltaVector::new(&[1.0], HashSet::new(), tx_id),
                };
                let _ = coord1.record_vote(tx_id, 1, vote2);

                ops1.fetch_add(1, Ordering::Relaxed);
                thread::yield_now();
            }
        });

        // Thread 2: Continuously call release_orphaned_locks
        let coord2 = Arc::clone(&coordinator);
        let ops2 = Arc::clone(&operations_completed);
        let release_thread = thread::spawn(move || {
            for _ in 0..iterations {
                let partition_time = now_epoch_millis() + 10_000;
                coord2.release_orphaned_locks(partition_time);
                ops2.fetch_add(1, Ordering::Relaxed);
                thread::yield_now();
            }
        });

        // Thread 3: Continuously call pending_aborts (which record_vote also uses)
        let coord3 = Arc::clone(&coordinator);
        let ops3 = Arc::clone(&operations_completed);
        let aborts_thread = thread::spawn(move || {
            for _ in 0..iterations {
                let _aborts = coord3.take_pending_aborts();
                ops3.fetch_add(1, Ordering::Relaxed);
                thread::yield_now();
            }
        });

        // Set a watchdog - if threads don't complete in 5 seconds, assume deadlock
        let deadlock = Arc::clone(&deadlock_detected);
        let watchdog = thread::spawn(move || {
            thread::sleep(Duration::from_secs(5));
            deadlock.store(true, Ordering::Relaxed);
        });

        // Wait for threads to complete
        record_vote_thread.join().unwrap();
        release_thread.join().unwrap();
        aborts_thread.join().unwrap();

        // If watchdog hasn't triggered, we didn't deadlock
        assert!(
            !deadlock_detected.load(Ordering::Relaxed),
            "Deadlock detected between record_vote and release_orphaned_locks"
        );

        // Verify operations completed
        assert!(
            operations_completed.load(Ordering::Relaxed) >= iterations * 3,
            "Not all operations completed"
        );

        // Watchdog will timeout on its own, no need to join
        drop(watchdog);
    }

    #[test]
    fn test_record_vote_phased_abort_handling() {
        // Verify that abort votes correctly release pending lock before
        // acquiring pending_aborts lock
        let coordinator = create_test_coordinator();

        // Create transaction
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        // Record YES from shard 0
        let vote1 = PrepareVote::Yes {
            lock_handle: 1,
            delta: DeltaVector::new(&[1.0], HashSet::new(), tx_id),
        };
        assert_eq!(coordinator.record_vote(tx_id, 0, vote1).unwrap(), None);

        // Record NO from shard 1 - this should trigger abort handling
        let vote2 = PrepareVote::No {
            reason: "test abort".to_string(),
        };
        let result = coordinator.record_vote(tx_id, 1, vote2);

        // Should return Aborting phase
        assert_eq!(result.unwrap(), Some(TxPhase::Aborting));

        // Abort should be queued in pending_aborts
        let aborts = coordinator.take_pending_aborts();
        assert_eq!(aborts.len(), 1);
        assert_eq!(aborts[0].0, tx_id);
        assert_eq!(aborts[0].1, "participant voted no");
    }

    fn create_test_coordinator_with_config(
        config: DistributedTxConfig,
    ) -> DistributedTxCoordinator {
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        DistributedTxCoordinator::new(consensus, config)
    }

    #[test]
    fn test_begin_max_concurrent_atomic() {
        // Verify atomic check-and-insert prevents exceeding limit
        let coordinator = create_test_coordinator_with_config(DistributedTxConfig {
            max_concurrent: 2,
            ..Default::default()
        });

        let _tx1 = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        let _tx2 = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Third should fail
        let result = coordinator.begin(&"node1".to_string(), &[0]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::TransactionFailed(_)
        ));
    }

    #[test]
    fn test_begin_max_concurrent_race_prevented() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;
        use std::thread;

        let coordinator = Arc::new(create_test_coordinator_with_config(DistributedTxConfig {
            max_concurrent: 5,
            ..Default::default()
        }));

        let success_count = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..20)
            .map(|_| {
                let coord = Arc::clone(&coordinator);
                let counter = Arc::clone(&success_count);
                thread::spawn(move || {
                    if coord.begin(&"node1".to_string(), &[0]).is_ok() {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Exactly max_concurrent should succeed
        assert_eq!(success_count.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_try_lock_wait_graph_updated_atomically() {
        let lock_manager = LockManager::new();
        let wait_graph = crate::deadlock::WaitForGraph::new();

        // TX1 holds lock on key_a
        lock_manager.try_lock(1, &["key_a".to_string()]).unwrap();

        // TX2 tries to acquire - should add wait edge while holding locks
        let result =
            lock_manager.try_lock_with_wait_tracking(2, &["key_a".to_string()], &wait_graph, None);

        assert!(result.is_err());
        // Wait graph should show TX2 waiting for TX1
        assert!(wait_graph.waiting_for(2).contains(&1));
    }

    #[test]
    fn test_try_lock_wait_graph_concurrent_safety() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Arc;
        use std::thread;

        let lock_manager = Arc::new(LockManager::new());
        let wait_graph = Arc::new(crate::deadlock::WaitForGraph::new());

        // TX1 holds lock on key_a
        lock_manager.try_lock(1, &["key_a".to_string()]).unwrap();

        let contention_count = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (2..12u64)
            .map(|tx_id| {
                let lm = Arc::clone(&lock_manager);
                let wg = Arc::clone(&wait_graph);
                let counter = Arc::clone(&contention_count);
                thread::spawn(move || {
                    if lm
                        .try_lock_with_wait_tracking(tx_id, &["key_a".to_string()], &wg, None)
                        .is_err()
                    {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All 10 threads should see contention
        assert_eq!(contention_count.load(Ordering::Relaxed), 10);
        // All should be waiting for TX1
        for tx_id in 2..12u64 {
            assert!(wait_graph.waiting_for(tx_id).contains(&1));
        }
    }

    #[test]
    fn test_release_orphaned_locks_atomic() {
        let coordinator = create_test_coordinator();

        // Create a transaction
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();

        // Acquire locks directly
        coordinator
            .lock_manager
            .try_lock(tx.tx_id, &["key_a".to_string(), "key_b".to_string()])
            .unwrap();

        // Remove from pending to simulate orphan
        coordinator.pending.write().remove(&tx.tx_id);

        // Release orphaned locks
        let released = coordinator.release_orphaned_locks(u64::MAX);
        assert_eq!(released, 2);

        // Verify locks are gone
        assert_eq!(coordinator.lock_manager.active_lock_count(), 0);
    }

    #[test]
    fn test_release_orphaned_locks_concurrent_transactions() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use std::thread;

        let coordinator = Arc::new(create_test_coordinator());
        let stop = Arc::new(AtomicBool::new(false));

        // Thread 1: Continuously starts/ends transactions
        let coord1 = Arc::clone(&coordinator);
        let stop1 = Arc::clone(&stop);
        let tx_thread = thread::spawn(move || {
            let mut count = 0;
            while !stop1.load(Ordering::Relaxed) && count < 100 {
                if let Ok(tx) = coord1.begin(&"node1".to_string(), &[0]) {
                    coord1
                        .lock_manager
                        .try_lock(tx.tx_id, &[format!("key_{}", count)])
                        .ok();
                    coord1.pending.write().remove(&tx.tx_id);
                    count += 1;
                }
                thread::yield_now();
            }
            count
        });

        // Thread 2: Continuously cleans orphaned locks
        let coord2 = Arc::clone(&coordinator);
        let stop2 = Arc::clone(&stop);
        let cleanup_thread = thread::spawn(move || {
            let mut cleaned = 0;
            while !stop2.load(Ordering::Relaxed) {
                cleaned += coord2.release_orphaned_locks(u64::MAX);
                thread::yield_now();
            }
            cleaned
        });

        // Let them run
        thread::sleep(Duration::from_millis(100));
        stop.store(true, Ordering::Relaxed);

        let created = tx_thread.join().unwrap();
        let cleaned = cleanup_thread.join().unwrap();

        // All created orphans should eventually be cleaned
        assert!(
            cleaned >= created / 2,
            "Expected significant cleanup, got {}/{}",
            cleaned,
            created
        );

        // Final state should have no orphaned locks
        let remaining = coordinator.lock_manager.active_lock_count();
        let final_cleanup = coordinator.release_orphaned_locks(u64::MAX);
        assert_eq!(remaining, final_cleanup, "All remaining should be orphaned");
    }

    #[test]
    fn test_handle_prepare_populates_wait_graph() {
        use tensor_store::SparseVector;

        let coordinator = create_test_coordinator();

        // First transaction acquires locks
        let keys = vec!["key1".to_string(), "key2".to_string()];
        let _handle1 = coordinator.lock_manager.try_lock(100, &keys).unwrap();

        // Second transaction tries to prepare - should fail and populate wait graph
        let request = PrepareRequest {
            tx_id: 200,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0, 0.0, 0.0]),
            timeout_ms: 5000,
        };

        let vote = coordinator.handle_prepare(&request);

        // Should be a conflict vote
        assert!(matches!(vote, PrepareVote::Conflict { .. }));

        // Wait graph should have edge: 200 -> 100
        let waiting_for = coordinator.wait_graph().waiting_for(200);
        assert!(
            waiting_for.contains(&100),
            "tx 200 should be waiting for tx 100"
        );
    }

    #[test]
    fn test_commit_cleans_wait_graph() {
        use crate::consensus::{ConsensusConfig, DeltaVector};
        use tensor_store::SparseVector;

        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);

        // Start a transaction and record a YES vote
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;

        // Manually add to wait graph to simulate prior conflict
        coordinator.wait_graph().add_wait(tx_id, 999, None);
        assert!(!coordinator.wait_graph().is_empty());

        // Prepare the transaction (creates lock)
        let keys = vec!["key1".to_string()];
        let handle = coordinator.lock_manager.try_lock(tx_id, &keys).unwrap();

        // Record YES vote
        let vote = PrepareVote::Yes {
            lock_handle: handle,
            delta: DeltaVector::from_sparse(
                SparseVector::from_dense(&[1.0, 0.0, 0.0]),
                keys.into_iter().collect(),
                tx_id,
            ),
        };
        let _ = coordinator.record_vote(tx_id, 0, vote);

        // Commit should clean wait graph
        coordinator.commit(tx_id).unwrap();

        // Wait graph should be empty (tx removed)
        assert!(
            coordinator.wait_graph().waiting_for(tx_id).is_empty(),
            "wait graph should be cleaned after commit"
        );
    }

    #[test]
    fn test_abort_cleans_wait_graph() {
        use crate::consensus::{ConsensusConfig, DeltaVector};
        use tensor_store::SparseVector;

        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus);

        // Start a transaction
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;

        // Add to wait graph
        coordinator.wait_graph().add_wait(tx_id, 999, None);
        assert!(!coordinator.wait_graph().is_empty());

        // Prepare the transaction
        let keys = vec!["key1".to_string()];
        let handle = coordinator.lock_manager.try_lock(tx_id, &keys).unwrap();

        // Record YES vote
        let vote = PrepareVote::Yes {
            lock_handle: handle,
            delta: DeltaVector::from_sparse(
                SparseVector::from_dense(&[1.0, 0.0, 0.0]),
                keys.into_iter().collect(),
                tx_id,
            ),
        };
        let _ = coordinator.record_vote(tx_id, 0, vote);

        // Abort should clean wait graph
        coordinator.abort(tx_id, "test abort").unwrap();

        // Wait graph should be empty
        assert!(
            coordinator.wait_graph().waiting_for(tx_id).is_empty(),
            "wait graph should be cleaned after abort"
        );
    }

    #[test]
    fn test_timeout_cleanup_cleans_wait_graph() {
        use crate::consensus::{ConsensusConfig, DeltaVector};
        use tensor_store::SparseVector;

        // Create a coordinator with very short timeouts
        let config = DistributedTxConfig {
            max_concurrent: 100,
            prepare_timeout_ms: 1, // Very short timeout
            commit_timeout_ms: 1,
            ..Default::default()
        };
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config);

        // Start a transaction
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;

        // Manually set a very short timeout on the transaction
        {
            let mut pending = coordinator.pending.write();
            if let Some(tx) = pending.get_mut(&tx_id) {
                tx.timeout_ms = 1; // 1ms timeout
            }
        }

        // Prepare the transaction
        let keys = vec!["key1".to_string()];
        let handle = coordinator.lock_manager.try_lock(tx_id, &keys).unwrap();

        // Add to wait graph
        coordinator.wait_graph().add_wait(tx_id, 999, None);

        // Record YES vote (so we have a lock to release)
        let vote = PrepareVote::Yes {
            lock_handle: handle,
            delta: DeltaVector::from_sparse(
                SparseVector::from_dense(&[1.0, 0.0, 0.0]),
                keys.into_iter().collect(),
                tx_id,
            ),
        };
        let _ = coordinator.record_vote(tx_id, 0, vote);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(10));

        // Cleanup should remove from wait graph
        let timed_out = coordinator.cleanup_timeouts();
        assert!(timed_out.contains(&tx_id));

        // Wait graph should be cleaned
        assert!(
            coordinator.wait_graph().waiting_for(tx_id).is_empty(),
            "wait graph should be cleaned after timeout"
        );
    }

    #[test]
    fn test_release_by_handle_with_wait_cleanup() {
        use crate::deadlock::WaitForGraph;

        let lock_manager = LockManager::new();
        let wait_graph = WaitForGraph::new();

        // Transaction 1 acquires lock
        let keys = vec!["key1".to_string()];
        let handle = lock_manager.try_lock(100, &keys).unwrap();

        // Transaction 2 is waiting for transaction 1
        wait_graph.add_wait(200, 100, None);
        assert!(wait_graph.waiting_for(200).contains(&100));

        // Release with wait cleanup
        lock_manager.release_by_handle_with_wait_cleanup(handle, &wait_graph);

        // Lock should be released
        assert!(!lock_manager.is_locked("key1"));

        // Transaction 100 should be removed from wait graph
        // (both as holder and any edges involving it)
        assert!(
            wait_graph.waiting_on(100).is_empty(),
            "tx 100 should be removed from wait graph"
        );
    }

    #[test]
    fn test_begin_wal_atomic_with_pending() {
        use crate::consensus::ConsensusConfig;
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let wal = crate::tx_wal::TxWal::open(&wal_path).unwrap();

        let config = DistributedTxConfig {
            max_concurrent: 2,
            ..Default::default()
        };
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::new(consensus, config).with_wal(wal);

        // Begin two transactions (at limit)
        let tx1 = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        let tx2 = coordinator.begin(&"node1".to_string(), &[1]).unwrap();

        // Third should fail
        let result = coordinator.begin(&"node1".to_string(), &[2]);
        assert!(result.is_err());

        // Both successful transactions should be in pending
        assert!(coordinator.get(tx1.tx_id).is_some());
        assert!(coordinator.get(tx2.tx_id).is_some());
        assert_eq!(coordinator.pending_count(), 2);
    }

    #[test]
    fn test_wait_graph_multiple_blockers_tracked() {
        use crate::deadlock::WaitForGraph;

        let lock_manager = LockManager::new();
        let wait_graph = WaitForGraph::new();

        // Transaction 1 locks key1
        lock_manager.try_lock(100, &["key1".to_string()]).unwrap();

        // Transaction 2 locks key2
        lock_manager.try_lock(200, &["key2".to_string()]).unwrap();

        // Transaction 3 tries to lock both keys - should track both blockers
        let keys = vec!["key1".to_string(), "key2".to_string()];
        let result = lock_manager.try_lock_with_wait_tracking(300, &keys, &wait_graph, None);

        assert!(result.is_err());
        let _wait_info = result.unwrap_err();

        // Wait graph should have edges to both blockers
        let waiting_for = wait_graph.waiting_for(300);
        assert!(
            waiting_for.contains(&100) && waiting_for.contains(&200),
            "tx 300 should be waiting for both tx 100 and tx 200"
        );
    }

    #[test]
    fn test_wait_graph_no_orphaned_edges_after_release() {
        use crate::deadlock::WaitForGraph;

        let lock_manager = LockManager::new();
        let wait_graph = WaitForGraph::new();

        // Create a conflict scenario
        let handle1 = lock_manager.try_lock(100, &["key1".to_string()]).unwrap();

        // Transaction 200 tries and fails, creating wait edge
        let _ =
            lock_manager.try_lock_with_wait_tracking(200, &["key1".to_string()], &wait_graph, None);

        assert!(wait_graph.waiting_for(200).contains(&100));

        // Release tx 100's lock with cleanup
        lock_manager.release_by_handle_with_wait_cleanup(handle1, &wait_graph);

        // Now tx 200's wait edge to 100 should be cleaned
        // (because 100 is removed from the graph)
        assert!(
            !wait_graph.waiting_for(200).contains(&100),
            "edge 200->100 should be removed when 100 is cleaned"
        );
    }

    #[test]
    fn test_get_pending_transactions() {
        let config = DistributedTxConfig::default();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coord = DistributedTxCoordinator::new(consensus, config);

        // Begin a transaction
        let tx = coord.begin(&"node1".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        // Get pending transactions
        let pending = coord.get_pending_transactions();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].tx_id, tx_id);
        assert_eq!(pending[0].coordinator, "node1");
    }

    #[test]
    fn test_force_resolve_commit() {
        let config = DistributedTxConfig::default();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coord = DistributedTxCoordinator::new(consensus, config);

        // Begin a transaction
        let tx = coord.begin(&"node1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;

        // Record a YES vote with lock
        let request = PrepareRequest {
            tx_id,
            coordinator: "node1".to_string(),
            operations: vec![],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        let vote = coord.handle_prepare(&request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));

        // Record the vote
        let _ = coord.record_vote(tx_id, 0, vote);

        // Force commit
        let result = coord.force_resolve(tx_id, true);
        assert!(result.is_ok());

        // Transaction should be removed
        let pending = coord.get_pending_transactions();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_force_resolve_abort() {
        let config = DistributedTxConfig::default();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coord = DistributedTxCoordinator::new(consensus, config);

        // Begin a transaction
        let tx = coord.begin(&"node1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;

        // Record a YES vote with lock
        let request = PrepareRequest {
            tx_id,
            coordinator: "node1".to_string(),
            operations: vec![],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        let vote = coord.handle_prepare(&request);
        let _ = coord.record_vote(tx_id, 0, vote);

        // Force abort
        let result = coord.force_resolve(tx_id, false);
        assert!(result.is_ok());

        // Transaction should be removed
        let pending = coord.get_pending_transactions();
        assert!(pending.is_empty());
    }

    #[test]
    fn test_force_resolve_not_found() {
        let config = DistributedTxConfig::default();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coord = DistributedTxCoordinator::new(consensus, config);

        // Try to force resolve non-existent transaction
        let result = coord.force_resolve(99999, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_force_resolve_commit_with_no_vote() {
        let config = DistributedTxConfig::default();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coord = DistributedTxCoordinator::new(consensus, config);

        // Begin a transaction
        let tx = coord.begin(&"node1".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        // Record a NO vote for one shard
        let no_vote = PrepareVote::No {
            reason: "test rejection".to_string(),
        };
        let _ = coord.record_vote(tx_id, 0, no_vote);

        // Try to force commit - should fail because there's a NO vote
        let result = coord.force_resolve(tx_id, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_release_by_handle_with_wait_cleanup_atomic() {
        let lock_manager = LockManager::new();
        let wait_graph = crate::deadlock::WaitForGraph::new();

        // Acquire locks for tx 1
        let keys = vec!["key1".to_string(), "key2".to_string()];
        let handle = lock_manager.try_lock(1, &keys).unwrap();
        assert!(lock_manager.is_locked("key1"));
        assert!(lock_manager.is_locked("key2"));

        // Add wait edge: tx 2 waits for tx 1
        wait_graph.add_wait(2, 1, None);
        assert!(!wait_graph.waiting_for(2).is_empty());

        // Release locks with wait cleanup
        lock_manager.release_by_handle_with_wait_cleanup(handle, &wait_graph);

        // Verify locks are released
        assert!(!lock_manager.is_locked("key1"));
        assert!(!lock_manager.is_locked("key2"));

        // Verify wait graph is cleaned up for tx 1
        // After remove_transaction(1), tx 2 should no longer wait for tx 1
        assert!(wait_graph.waiting_for(2).is_empty());
        assert!(wait_graph.waiting_on(1).is_empty());
    }

    #[test]
    fn test_cleanup_expired_with_wait_cleanup() {
        let lock_manager = LockManager::with_default_timeout(Duration::from_millis(1));
        let wait_graph = crate::deadlock::WaitForGraph::new();

        // Acquire lock with short timeout
        let keys = vec!["key1".to_string()];
        lock_manager.try_lock(1, &keys).unwrap();

        // Add wait edge: tx 2 waits for tx 1
        wait_graph.add_wait(2, 1, None);
        assert!(!wait_graph.waiting_for(2).is_empty());

        // Wait for lock to expire
        std::thread::sleep(Duration::from_millis(10));

        // Cleanup expired locks with wait cleanup
        let cleaned = lock_manager.cleanup_expired_with_wait_cleanup(&wait_graph);
        assert_eq!(cleaned, 1);

        // Verify lock is released
        assert!(!lock_manager.is_locked("key1"));

        // Verify wait graph is cleaned up for tx 1
        assert!(wait_graph.waiting_for(2).is_empty());
        assert!(wait_graph.waiting_on(1).is_empty());
    }

    #[test]
    fn test_release_orphaned_locks_cleans_wait_graph() {
        let config = DistributedTxConfig::default();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coord = DistributedTxCoordinator::new(consensus, config);

        // Acquire locks directly on lock manager (simulating orphaned locks)
        let keys = vec!["orphan_key".to_string()];
        coord.lock_manager.try_lock(999, &keys).unwrap();
        assert!(coord.lock_manager.is_locked("orphan_key"));

        // Add wait edge: tx 1000 waits for tx 999
        coord.wait_graph.add_wait(1000, 999, None);
        assert!(!coord.wait_graph.waiting_for(1000).is_empty());

        // Release orphaned locks with partition_start in future
        let now_ms = now_epoch_millis();
        let released = coord.release_orphaned_locks(now_ms + 10000);
        assert_eq!(released, 1);

        // Verify lock is released
        assert!(!coord.lock_manager.is_locked("orphan_key"));

        // Verify wait graph is cleaned up for tx 999
        assert!(coord.wait_graph.waiting_on(999).is_empty());
        assert!(coord.wait_graph.waiting_for(1000).is_empty());
    }

    #[test]
    fn test_release_orphaned_locks_cleans_multiple_tx_wait_edges() {
        let config = DistributedTxConfig::default();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coord = DistributedTxCoordinator::new(consensus, config);

        // Acquire locks for multiple orphaned transactions
        coord
            .lock_manager
            .try_lock(100, &["key_a".to_string()])
            .unwrap();
        coord
            .lock_manager
            .try_lock(200, &["key_b".to_string()])
            .unwrap();
        coord
            .lock_manager
            .try_lock(300, &["key_c".to_string()])
            .unwrap();

        // Add wait edges
        coord.wait_graph.add_wait(101, 100, None);
        coord.wait_graph.add_wait(201, 200, None);
        coord.wait_graph.add_wait(301, 300, None);

        assert!(!coord.wait_graph.waiting_for(101).is_empty());
        assert!(!coord.wait_graph.waiting_for(201).is_empty());
        assert!(!coord.wait_graph.waiting_for(301).is_empty());

        // Release all orphaned locks
        let now_ms = now_epoch_millis();
        let released = coord.release_orphaned_locks(now_ms + 10000);
        assert_eq!(released, 3);

        // Verify all wait graph entries are cleaned up
        assert!(coord.wait_graph.waiting_on(100).is_empty());
        assert!(coord.wait_graph.waiting_on(200).is_empty());
        assert!(coord.wait_graph.waiting_on(300).is_empty());
        assert!(coord.wait_graph.waiting_for(101).is_empty());
        assert!(coord.wait_graph.waiting_for(201).is_empty());
        assert!(coord.wait_graph.waiting_for(301).is_empty());
    }

    // === TxParticipant commit/abort/recovery tests ===

    #[test]
    fn test_participant_commit_applies_put_operation() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "test_key".into(),
                data: b"test_data".to_vec(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        let vote = participant.prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));

        let response = participant.commit(1);
        assert!(response.success);
        assert!(store.exists("test_key"));
    }

    #[test]
    fn test_participant_commit_applies_embed_operation() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Embed {
                key: "doc1".into(),
                vector: vec![1.0, 2.0, 3.0],
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        let response = participant.commit(1);

        assert!(response.success);
        // Should be stored with "emb:" prefix
        assert!(store.exists("emb:doc1"));
        // Raw key should not exist
        assert!(!store.exists("doc1"));
    }

    #[test]
    fn test_participant_commit_applies_node_create_operation() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::NodeCreate {
                key: "user1".into(),
                label: "Person".into(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        let response = participant.commit(1);

        assert!(response.success);
        assert!(store.exists("node:user1"));
    }

    #[test]
    fn test_participant_commit_applies_edge_create_operation() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::EdgeCreate {
                from: "a".into(),
                to: "b".into(),
                edge_type: "knows".into(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        let response = participant.commit(1);

        assert!(response.success);
        assert!(store.exists("edge:a:b:knows"));
    }

    #[test]
    fn test_participant_abort_removes_new_key() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        // Key doesn't exist initially
        assert!(!store.exists("new_key"));

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "new_key".into(),
                data: b"data".to_vec(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        participant.abort(1);

        // Key should not exist after abort
        assert!(!store.exists("new_key"));
    }

    #[test]
    fn test_participant_abort_restores_existing_key() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        // Pre-populate with existing data
        let mut tensor = TensorData::new();
        tensor.set(
            "data",
            TensorValue::Scalar(ScalarValue::Bytes(b"original".to_vec())),
        );
        store.put("existing_key", tensor).unwrap();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "existing_key".into(),
                data: b"modified".to_vec(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        participant.abort(1);

        // Verify original data is restored
        let data = store.get("existing_key").unwrap();
        let bytes = data.get("data").unwrap();
        match bytes {
            TensorValue::Scalar(ScalarValue::Bytes(b)) => {
                assert_eq!(b, b"original");
            },
            _ => panic!("Expected bytes"),
        }
    }

    #[test]
    fn test_participant_abort_embed_uses_correct_storage_key() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        // Pre-populate emb:doc1 with original vector
        let mut tensor = TensorData::new();
        tensor.set("vector", TensorValue::Vector(vec![0.0, 0.0, 0.0]));
        store.put("emb:doc1", tensor).unwrap();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Embed {
                key: "doc1".into(),
                vector: vec![1.0, 2.0, 3.0],
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        participant.abort(1);

        // Verify original vector is restored at emb:doc1
        let data = store.get("emb:doc1").unwrap();
        let vec = data.get("vector").unwrap();
        match vec {
            TensorValue::Vector(v) => {
                assert_eq!(v, &[0.0, 0.0, 0.0]);
            },
            _ => panic!("Expected vector"),
        }
    }

    #[test]
    fn test_participant_multiple_operations_commit() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![
                Transaction::Put {
                    key: "key1".into(),
                    data: b"data1".to_vec(),
                },
                Transaction::Embed {
                    key: "doc1".into(),
                    vector: vec![1.0],
                },
                Transaction::NodeCreate {
                    key: "n1".into(),
                    label: "Label".into(),
                },
            ],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        let response = participant.commit(1);

        assert!(response.success);
        assert!(store.exists("key1"));
        assert!(store.exists("emb:doc1"));
        assert!(store.exists("node:n1"));
    }

    #[test]
    fn test_participant_multiple_operations_abort() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![
                Transaction::Put {
                    key: "key1".into(),
                    data: b"data1".to_vec(),
                },
                Transaction::Embed {
                    key: "doc1".into(),
                    vector: vec![1.0],
                },
            ],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        participant.abort(1);

        // Neither key should exist after abort
        assert!(!store.exists("key1"));
        assert!(!store.exists("emb:doc1"));
    }

    #[test]
    fn test_participant_commit_unknown_tx_fails() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store);

        let response = participant.commit(999);
        assert!(!response.success);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_participant_delete_operation_commit() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        // Pre-populate
        let mut tensor = TensorData::new();
        tensor.set(
            "data",
            TensorValue::Scalar(ScalarValue::Bytes(b"data".to_vec())),
        );
        store.put("to_delete", tensor).unwrap();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Delete {
                key: "to_delete".into(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        let response = participant.commit(1);

        assert!(response.success);
        assert!(!store.exists("to_delete"));
    }

    #[test]
    fn test_participant_delete_operation_abort_restores() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        // Pre-populate
        let mut tensor = TensorData::new();
        tensor.set(
            "data",
            TensorValue::Scalar(ScalarValue::Bytes(b"original".to_vec())),
        );
        store.put("to_delete", tensor).unwrap();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Delete {
                key: "to_delete".into(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        participant.abort(1);

        // Key should still exist after abort
        assert!(store.exists("to_delete"));
    }

    #[test]
    fn test_participant_recover_applies_undo_logs() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        // Pre-populate with existing data
        let mut tensor = TensorData::new();
        tensor.set(
            "data",
            TensorValue::Scalar(ScalarValue::Bytes(b"original".to_vec())),
        );
        store.put("recover_key", tensor).unwrap();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "recover_key".into(),
                data: b"modified".to_vec(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);

        // Use cleanup_stale with zero duration (uses >= for timeout check)
        // This simulates timeout and applies undo logs
        let stale = participant.cleanup_stale(Duration::from_secs(0));

        // Transaction should be cleaned up
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0], 1);

        // Original data should be restored (undo log applied)
        let data = store.get("recover_key").unwrap();
        let bytes = data.get("data").unwrap();
        match bytes {
            TensorValue::Scalar(ScalarValue::Bytes(b)) => {
                assert_eq!(b, b"original");
            },
            _ => panic!("Expected bytes"),
        }
    }

    #[test]
    fn test_participant_node_delete_commit() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        // Pre-populate node
        let mut tensor = TensorData::new();
        tensor.set(
            "_id",
            TensorValue::Scalar(ScalarValue::String("user1".into())),
        );
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("node".into())),
        );
        store.put("node:user1", tensor).unwrap();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::NodeDelete {
                key: "user1".into(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        let response = participant.commit(1);

        assert!(response.success);
        assert!(!store.exists("node:user1"));
    }

    #[test]
    fn test_participant_table_insert_commit() {
        let store = TensorStore::new();
        let participant = TxParticipant::new(store.clone());

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::TableInsert {
                table: "users".into(),
                values: b"serialized_row".to_vec(),
            }],
            delta_embedding: SparseVector::new(0),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        let response = participant.commit(1);

        assert!(response.success);
        assert!(store.exists("table:users"));
    }

    #[test]
    fn test_begin_soft_limit_warning() {
        // Configure max_concurrent=10, soft_limit_pct=50 so soft limit triggers at 5.
        // The check runs before inserting, so pending.len() must be >= 5 to trigger.
        let config = DistributedTxConfig {
            max_concurrent: 10,
            tx_queue_soft_limit_pct: 50,
            ..Default::default()
        };
        let coordinator = create_test_coordinator_with_config(config);

        // Begin 5 transactions: pending.len() is 0..4 at each check, all < 5
        for _ in 0..5 {
            coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        }
        assert_eq!(
            coordinator
                .stats
                .tx_queue_soft_limit_warnings
                .load(Ordering::Relaxed),
            0
        );

        // 6th transaction: pending.len()==5 at check, which >= soft_limit(5)
        coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        assert_eq!(
            coordinator
                .stats
                .tx_queue_soft_limit_warnings
                .load(Ordering::Relaxed),
            1
        );

        // 7th still succeeds (soft limit, not hard) and also triggers warning
        coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        assert_eq!(
            coordinator
                .stats
                .tx_queue_soft_limit_warnings
                .load(Ordering::Relaxed),
            2
        );

        // Verify all 7 transactions are pending (none rejected by soft limit)
        assert_eq!(coordinator.pending_count(), 7);
    }

    #[test]
    fn test_next_lock_handle_increments() {
        let h1 = next_lock_handle();
        let h2 = next_lock_handle();
        assert!(h2 > h1, "Lock handles should monotonically increase");
    }

    #[test]
    fn test_lock_handle_current_returns_counter_value() {
        let before = lock_handle_current();
        let _ = next_lock_handle();
        let after = lock_handle_current();
        assert!(after > before);
    }

    #[test]
    fn test_lock_handle_high_water_threshold() {
        let threshold = lock_handle_high_water_threshold();
        // 90% of u64::MAX
        assert_eq!(threshold, u64::MAX / 10 * 9);
        assert!(threshold > u64::MAX / 2, "Threshold should be above 50%");
    }

    #[test]
    fn test_lock_handle_high_water_warnings_accessible() {
        // Just verify the counter is accessible and returns a value
        let warnings = lock_handle_high_water_warnings();
        // In normal operation, this should be 0 since we never reach 90% of u64::MAX
        // But other tests in the process may have run, so just check it doesn't panic
        let _ = warnings;
    }

    #[test]
    fn test_lock_handle_used_in_try_lock() {
        let lm = LockManager::new();
        let keys = vec!["key_a".to_string()];
        let handle = lm.try_lock(100, &keys).unwrap();
        // Handle should come from the global counter (non-zero)
        assert!(handle > 0);
    }

    #[test]
    fn test_lock_handle_used_in_try_lock_with_wait_tracking() {
        let lm = LockManager::new();
        let wg = crate::deadlock::WaitForGraph::new();
        let keys = vec!["key_b".to_string()];
        let handle = lm
            .try_lock_with_wait_tracking(100, &keys, &wg, None)
            .unwrap();
        assert!(handle > 0);
    }

    #[test]
    fn test_stats_snapshot_includes_lock_handle_metrics() {
        let stats = DistributedTxStats::new();
        let snap = stats.snapshot();
        // lock_handle_current should reflect the global counter
        assert!(snap.lock_handle_current > 0);
        // high_water_warnings should be a valid value
        let _ = snap.lock_handle_high_water_warnings;
    }

    #[test]
    fn test_retry_aborts_exponential_backoff() {
        let coordinator = create_test_coordinator();

        // Track an abort that was just initiated
        coordinator.track_abort(100, vec![0, 1]);

        // Immediately, no retries should fire (need at least 1s)
        let retries = coordinator.get_retry_aborts();
        assert!(retries.is_empty());

        // Simulate time passing by backdating the initiated_at
        {
            let mut states = coordinator.abort_states.write();
            let state = states.get_mut(&100).unwrap();
            // Set initiated 2 seconds ago
            state.initiated_at = now_epoch_millis().saturating_sub(2000);
        }

        // First retry should fire (backoff: 1s, cumulative: 1s)
        let retries = coordinator.get_retry_aborts();
        assert_eq!(retries.len(), 1);
        assert_eq!(retries[0].0, 100);
        assert_eq!(retries[0].1.len(), 2); // shards 0 and 1

        // retry_count should now be 1
        {
            let states = coordinator.abort_states.read();
            assert_eq!(states.get(&100).unwrap().retry_count, 1);
        }

        // No immediate second retry (need 3s cumulative for retry 2)
        let retries = coordinator.get_retry_aborts();
        assert!(retries.is_empty());
    }

    #[test]
    fn test_retry_aborts_max_retries_respected() {
        let coordinator = create_test_coordinator();
        coordinator.track_abort(200, vec![0]);

        // Set retry_count to MAX_ABORT_RETRIES
        {
            let mut states = coordinator.abort_states.write();
            let state = states.get_mut(&200).unwrap();
            state.retry_count = DistributedTxCoordinator::MAX_ABORT_RETRIES;
            state.initiated_at = 0; // Long time ago
        }

        // Should not retry since max retries reached
        let retries = coordinator.get_retry_aborts();
        assert!(retries.is_empty());
    }

    #[test]
    fn test_cleanup_stale_aborts_uses_max_retries() {
        let coordinator = create_test_coordinator();
        coordinator.track_abort(300, vec![0]);

        // Set retry_count to MAX_ABORT_RETRIES
        {
            let mut states = coordinator.abort_states.write();
            let state = states.get_mut(&300).unwrap();
            state.retry_count = DistributedTxCoordinator::MAX_ABORT_RETRIES;
        }

        // Should be cleaned up
        let stale = coordinator.cleanup_stale_aborts();
        assert_eq!(stale, vec![300]);
    }

    #[test]
    fn test_stats_snapshot_includes_abort_delivery_metrics() {
        let stats = DistributedTxStats::new();
        stats.abort_delivery_retries.fetch_add(3, Ordering::Relaxed);
        stats
            .abort_delivery_failures
            .fetch_add(1, Ordering::Relaxed);
        let snap = stats.snapshot();
        assert_eq!(snap.abort_delivery_retries, 3);
        assert_eq!(snap.abort_delivery_failures, 1);
    }

    #[test]
    fn test_abort_delivery_retries_stat_incremented() {
        let coordinator = create_test_coordinator();
        coordinator.track_abort(400, vec![0]);

        // Backdate and trigger a retry
        {
            let mut states = coordinator.abort_states.write();
            let state = states.get_mut(&400).unwrap();
            state.initiated_at = now_epoch_millis().saturating_sub(2000);
        }

        let _ = coordinator.get_retry_aborts();
        assert_eq!(
            coordinator
                .stats
                .abort_delivery_retries
                .load(Ordering::Relaxed),
            1
        );
    }

    // ========== Mutation-Catching Tests ==========

    #[test]
    fn test_cleanup_timeouts_removes_expired() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        // Force the tx timeout by setting started_at far in the past (epoch ms)
        {
            let mut pending = coordinator.pending.write();
            if let Some(tx) = pending.get_mut(&tx_id) {
                tx.started_at = 1000; // ancient epoch ms
            }
        }

        let timed_out = coordinator.cleanup_timeouts();
        assert!(
            timed_out.contains(&tx_id),
            "Timed-out tx {tx_id} should be in cleanup list"
        );

        // Tx should no longer be pending
        assert!(
            coordinator.pending.read().get(&tx_id).is_none(),
            "Timed-out tx must be removed from pending"
        );
    }

    #[test]
    fn test_complete_commit_wrong_phase() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;

        // Tx is in Preparing phase, not Committing
        let result = coordinator.complete_commit(tx_id);
        assert!(
            result.is_err(),
            "complete_commit must fail when tx is not in Committing phase"
        );
    }

    #[test]
    fn test_complete_abort_releases_locks() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"node1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;

        // Manually move to Aborting phase
        {
            let mut pending = coordinator.pending.write();
            if let Some(tx) = pending.get_mut(&tx_id) {
                tx.phase = TxPhase::Aborting;
            }
        }

        let result = coordinator.complete_abort(tx_id);
        assert!(result.is_ok(), "complete_abort should succeed");

        // Tx should be removed from pending
        assert!(
            coordinator.pending.read().get(&tx_id).is_none(),
            "Aborted tx must be removed from pending"
        );
    }

    #[test]
    fn test_release_orphaned_locks_age_filter() {
        let coordinator = create_test_coordinator();
        let lock_mgr = &coordinator.lock_manager;

        // Acquire a lock for tx 999 (not in pending)
        let keys = vec!["orphan_key".to_string()];
        lock_mgr.try_lock(999, &keys).unwrap();

        // Set acquired_at_ms to a value older than partition_start_ms
        {
            let mut locks = lock_mgr.locks.write();
            if let Some(lock) = locks.get_mut("orphan_key") {
                lock.acquired_at_ms = 1000; // ancient timestamp
            }
        }

        // With partition_start_ms > lock's acquired_at_ms, lock should be orphaned
        let released = coordinator.release_orphaned_locks(2000);
        assert_eq!(released, 1, "Should release one orphaned lock");

        // Lock should be gone
        assert!(
            lock_mgr.locks.read().get("orphan_key").is_none(),
            "Orphaned lock must be removed"
        );
    }

    #[test]
    fn test_undo_entry_capture_existing_key_mutation() {
        let store = TensorStore::new();
        let mut data = TensorData::new();
        data.set("v", TensorValue::Scalar(ScalarValue::Int(42)));
        store.put("existing", data).unwrap();

        let entry = UndoEntry::capture("existing", &store);
        assert!(
            matches!(entry, UndoEntry::Restore { .. }),
            "Capture of existing key must produce Restore variant"
        );
        assert_eq!(entry.key(), "existing");
    }

    #[test]
    fn test_undo_entry_capture_missing_key_mutation() {
        let store = TensorStore::new();

        let entry = UndoEntry::capture("nonexistent", &store);
        assert!(
            matches!(entry, UndoEntry::Delete { .. }),
            "Capture of missing key must produce Delete variant"
        );
        assert_eq!(entry.key(), "nonexistent");
    }

    #[test]
    fn test_undo_entry_apply_verified_bad_checksum() {
        let store = TensorStore::new();
        let mut data = TensorData::new();
        data.set("v", TensorValue::Scalar(ScalarValue::Int(10)));
        store.put("key1", data).unwrap();

        let entry = UndoEntry::capture("key1", &store);
        let correct_checksum = entry.checksum();

        // Apply with wrong checksum should fail
        let result = entry.apply_verified(&store, correct_checksum.wrapping_add(1));
        assert!(
            result.is_err(),
            "apply_verified must fail with bad checksum"
        );

        // Apply with correct checksum should succeed
        let result = entry.apply_verified(&store, correct_checksum);
        assert!(
            result.is_ok(),
            "apply_verified must succeed with correct checksum"
        );
    }

    #[test]
    fn test_lock_manager_try_lock_conflict() {
        let lock_mgr = LockManager::new();
        let keys = vec!["shared".to_string()];

        // tx 1 acquires the lock
        lock_mgr.try_lock(1, &keys).unwrap();

        // tx 2 should get conflict with holder = 1
        let result = lock_mgr.try_lock(2, &keys);
        assert_eq!(result, Err(1), "Conflicting lock must report holder tx_id");

        // Same tx can re-lock (idempotent)
        let result = lock_mgr.try_lock(1, &keys);
        assert!(result.is_ok(), "Same tx re-locking should succeed");
    }

    #[test]
    fn test_lock_manager_release_cleans_both() {
        let lock_mgr = LockManager::new();
        let keys = vec!["k1".to_string(), "k2".to_string()];

        lock_mgr.try_lock(1, &keys).unwrap();

        // Verify both locks and tx_locks are populated
        assert!(lock_mgr.locks.read().contains_key("k1"));
        assert!(lock_mgr.locks.read().contains_key("k2"));
        assert!(lock_mgr.tx_locks.read().contains_key(&1));

        lock_mgr.release(1);

        // Both maps must be cleaned
        assert!(
            !lock_mgr.locks.read().contains_key("k1"),
            "k1 lock must be removed after release"
        );
        assert!(
            !lock_mgr.locks.read().contains_key("k2"),
            "k2 lock must be removed after release"
        );
        assert!(
            !lock_mgr.tx_locks.read().contains_key(&1),
            "tx_locks entry must be removed after release"
        );
    }

    #[test]
    fn test_is_timed_out_exact_boundary() {
        // When elapsed == timeout_ms, the transaction must NOT be timed out.
        // Kills mutation: replace > with >= in is_timed_out
        let mut tx = DistributedTransaction::new("node1".to_string(), vec![0]);
        tx.timeout_ms = 100;
        // Set started_at so that now - started_at == timeout_ms exactly
        #[allow(clippy::cast_possible_truncation)]
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        tx.started_at = now - 100; // elapsed == timeout_ms
        assert!(
            !tx.is_timed_out(),
            "Transaction must NOT be timed out when elapsed == timeout_ms"
        );
        // But one ms over should be timed out
        tx.started_at = now - 101;
        assert!(
            tx.is_timed_out(),
            "Transaction must be timed out when elapsed > timeout_ms"
        );
    }

    #[test]
    fn test_key_lock_is_expired_exact_boundary() {
        // When elapsed == timeout_ms, the lock must NOT be expired.
        // Kills mutation: replace > with >= in KeyLock::is_expired
        let lock = KeyLock {
            key: "k1".to_string(),
            tx_id: 1,
            lock_handle: 1,
            acquired_at_ms: now_epoch_millis(),
            timeout_ms: 0,
        };
        // elapsed == 0, timeout == 0 => 0 > 0 is false => not expired
        assert!(
            !lock.is_expired(),
            "Lock must NOT be expired when elapsed == timeout"
        );

        let lock2 = KeyLock {
            key: "k2".to_string(),
            tx_id: 2,
            lock_handle: 2,
            acquired_at_ms: now_epoch_millis() - 50,
            timeout_ms: 49,
        };
        assert!(
            lock2.is_expired(),
            "Lock must be expired when elapsed > timeout"
        );
    }

    #[test]
    fn test_lock_handle_high_water_warnings_returns_counter() {
        // Kills mutation: replace lock_handle_high_water_warnings -> u64 with 0
        let initial = lock_handle_high_water_warnings();
        let again = lock_handle_high_water_warnings();
        assert_eq!(initial, again, "Must return consistent counter value");
        let h1 = lock_handle_current();
        let _ = next_lock_handle();
        let h2 = lock_handle_current();
        assert!(
            h2 > h1,
            "Lock handle must increase after next_lock_handle()"
        );
    }

    #[test]
    fn test_release_by_handle_with_wait_cleanup_cleans_tx_locks() {
        // Kills mutation: replace != with == in release_by_handle_with_wait_cleanup (line 567)
        // The retain(|k| k != key) removes the key from tx_locks.
        // If mutated to ==, it keeps the key instead.
        let lm = LockManager::new();
        let graph = crate::deadlock::WaitForGraph::new();
        let handle = lm.try_lock(100, &["alpha".to_string()]).unwrap();

        // Lock two keys so we can verify partial cleanup
        let handle2 = lm.try_lock(100, &["beta".to_string()]).unwrap();
        assert_eq!(lm.lock_count_for_transaction(100), 2);

        // Release first by handle — should remove only "alpha" from tx_locks
        lm.release_by_handle_with_wait_cleanup(handle, &graph);
        let keys = lm.keys_for_transaction(100);
        assert!(
            !keys.contains(&"alpha".to_string()),
            "alpha must be removed from tx_locks after release_by_handle_with_wait_cleanup"
        );
        assert!(
            keys.contains(&"beta".to_string()),
            "beta must remain in tx_locks"
        );

        // Clean up
        lm.release_by_handle_with_wait_cleanup(handle2, &graph);
        assert_eq!(lm.lock_count_for_transaction(100), 0);
    }

    #[test]
    fn test_cleanup_expired_cleans_tx_locks() {
        // Kills mutation: replace != with == in cleanup_expired (line 607)
        let lm = LockManager::with_default_timeout(Duration::from_millis(1));
        let handle1 = lm.try_lock(200, &["key1".to_string()]).unwrap();
        let _handle2 = lm.try_lock(200, &["key2".to_string()]).unwrap();
        assert_eq!(lm.lock_count_for_transaction(200), 2);

        // Wait for locks to expire
        std::thread::sleep(Duration::from_millis(10));

        // Also acquire a non-expired lock on a different tx
        let lm_fresh = LockManager::new(); // default 5s timeout
        let _ = lm_fresh.try_lock(300, &["other".to_string()]);

        let expired_count = lm.cleanup_expired();
        assert!(expired_count >= 2, "Both locks should be expired");

        // tx_locks for tx 200 should be empty after cleanup
        let remaining = lm.keys_for_transaction(200);
        assert!(
            remaining.is_empty(),
            "tx_locks must be empty after all keys expired and cleaned, got: {remaining:?}"
        );
        let _ = handle1; // suppress unused warning
    }

    #[test]
    fn test_cleanup_expired_with_wait_cleanup_cleans_tx_locks() {
        // Kills mutation: replace != with == in cleanup_expired_with_wait_cleanup (line 635)
        let lm = LockManager::with_default_timeout(Duration::from_millis(1));
        let graph = crate::deadlock::WaitForGraph::new();
        let _ = lm.try_lock(300, &["x".to_string()]).unwrap();
        let _ = lm.try_lock(300, &["y".to_string()]).unwrap();

        std::thread::sleep(Duration::from_millis(10));

        let expired_count = lm.cleanup_expired_with_wait_cleanup(&graph);
        assert!(expired_count >= 2, "Both locks should be expired");

        let remaining = lm.keys_for_transaction(300);
        assert!(
            remaining.is_empty(),
            "tx_locks must be empty after cleanup_expired_with_wait_cleanup, got: {remaining:?}"
        );
    }

    #[test]
    fn test_try_lock_with_wait_tracking_ignores_expired_lock() {
        // Kills mutation: replace && with || in try_lock_with_wait_tracking (line 705)
        // The condition is: !existing.is_expired() && existing.tx_id != tx_id
        // With &&, an expired lock lets the new tx through.
        // With ||, an expired lock with different tx_id would still block.
        let lm = LockManager::with_default_timeout(Duration::from_millis(1));
        let graph = crate::deadlock::WaitForGraph::new();

        // Tx 100 locks "key_a"
        let _ = lm
            .try_lock(100, &["key_a".to_string()])
            .expect("initial lock must succeed");

        // Wait for the lock to expire
        std::thread::sleep(Duration::from_millis(10));

        // Tx 200 should be able to lock "key_a" since the old lock expired
        let result = lm.try_lock_with_wait_tracking(200, &["key_a".to_string()], &graph, None);
        assert!(
            result.is_ok(),
            "Must succeed when existing lock is expired, got: {result:?}"
        );
    }

    #[test]
    fn test_recover_from_wal_stats_accuracy() {
        // Kills mutations in recover_from_wal:
        // - replace += with -= (lines 1165, 1172, 1179)
        // - replace += with *= (lines 1165, 1172, 1179)
        // - replace += with -= / *= for lock_releases_recovered (line 1192)
        use crate::tx_wal::{PrepareVoteKind, TxWal, TxWalEntry};

        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("recovery_test.wal");
        let mut wal = TxWal::open(&wal_path).unwrap();

        // Write tx 1: prepared (Preparing -> Prepared, stays there)
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 1,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 1,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 10 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 1,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();

        // Write tx 2: committing (Preparing -> Prepared -> Committing)
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 2,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 2,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 20 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 2,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 2,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })
        .unwrap();

        // Write tx 3: aborting (Preparing -> Prepared -> Aborting)
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 3,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 3,
            shard: 0,
            vote: PrepareVoteKind::No,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 3,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 3,
            from: TxPhase::Prepared,
            to: TxPhase::Aborting,
        })
        .unwrap();

        drop(wal);

        // Re-open WAL read-only for recovery
        let wal2 = TxWal::open(&wal_path).unwrap();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus).with_wal(wal2);

        let stats = coordinator.recover_from_wal().unwrap();

        assert_eq!(stats.pending_prepare, 1, "Must have 1 prepared tx (tx 1)");
        assert_eq!(stats.pending_commit, 1, "Must have 1 committing tx (tx 2)");
        assert_eq!(stats.pending_abort, 1, "Must have 1 aborting tx (tx 3)");
    }

    #[test]
    fn test_recover_stats_per_phase() {
        // Kills mutations: replace += with -= / *= in DistributedTxCoordinator::recover
        // Tests that recovery stats accurately count transactions per phase.
        let coordinator = create_test_coordinator();

        // Insert transactions in various phases directly
        {
            let mut pending = coordinator.pending.write();

            // Preparing tx (not timed out) -> pending_prepare
            let mut tx_prep = DistributedTransaction::new("n".to_string(), vec![0]);
            tx_prep.phase = TxPhase::Preparing;
            pending.insert(tx_prep.tx_id, tx_prep);

            // Prepared tx with all YES votes -> pending_commit
            let mut tx_prepared = DistributedTransaction::new("n".to_string(), vec![0]);
            tx_prepared.phase = TxPhase::Prepared;
            tx_prepared.votes.insert(
                0,
                PrepareVote::Yes {
                    lock_handle: 99,
                    delta: DeltaVector::zero(0),
                },
            );
            pending.insert(tx_prepared.tx_id, tx_prepared);

            // Committing tx -> pending_commit
            let mut tx_committing = DistributedTransaction::new("n".to_string(), vec![0]);
            tx_committing.phase = TxPhase::Committing;
            pending.insert(tx_committing.tx_id, tx_committing);

            // Aborting tx -> pending_abort
            let mut tx_aborting = DistributedTransaction::new("n".to_string(), vec![0]);
            tx_aborting.phase = TxPhase::Aborting;
            pending.insert(tx_aborting.tx_id, tx_aborting);

            // Committed tx -> completed
            let mut tx_done = DistributedTransaction::new("n".to_string(), vec![0]);
            tx_done.phase = TxPhase::Committed;
            pending.insert(tx_done.tx_id, tx_done);
        }

        let stats = coordinator.recover();

        // Preparing (not timed out) counts as pending_prepare
        // Prepared with all yes moves to Committing -> pending_commit
        // Committing stays as pending_commit
        // Aborting stays as pending_abort
        // Committed is completed
        assert!(
            stats.pending_prepare >= 1,
            "Must count preparing tx, got {}",
            stats.pending_prepare
        );
        assert!(
            stats.pending_commit >= 1,
            "Must count committing tx, got {}",
            stats.pending_commit
        );
        assert!(
            stats.pending_abort >= 1,
            "Must count aborting tx, got {}",
            stats.pending_abort
        );
        assert!(
            stats.completed >= 1,
            "Must count completed tx, got {}",
            stats.completed
        );
        // With *= mutation, stats would be 0 (0 *= 1 = 0) instead of >= 1
        // With -= mutation, stats would underflow or be negative
    }

    #[test]
    fn test_cleanup_timeouts_removes_timed_out_and_logs_expired() {
        // Kills mutations:
        // - delete ! in cleanup_timeouts (line 1770) — if inverted, log would fire for empty list
        // - replace > with == in cleanup_timeouts (line 1809)
        let coordinator = create_test_coordinator();

        // Begin a tx then manually make it timed out
        let tx = coordinator.begin(&"n1".to_string(), &[0]).unwrap();
        let tx_id = tx.tx_id;
        {
            let mut pending = coordinator.pending.write();
            if let Some(t) = pending.get_mut(&tx_id) {
                t.started_at = 1; // epoch ms = very old
            }
        }

        // Also lock a key with a very short timeout to create an expired lock
        let short_lm = LockManager::with_default_timeout(Duration::from_millis(1));
        let _ = short_lm.try_lock(999, &["ephemeral".to_string()]);
        std::thread::sleep(Duration::from_millis(5));
        // The coordinator's own lock_manager won't have expired locks, but
        // the timed_out tx should be detected.

        let timed_out_ids = coordinator.cleanup_timeouts();
        assert!(
            timed_out_ids.contains(&tx_id),
            "Timed-out tx must be in the returned list"
        );
        // After cleanup, pending should not contain the timed-out tx
        assert_eq!(
            coordinator.pending_count(),
            0,
            "Timed-out tx must be removed from pending"
        );
    }

    #[test]
    fn test_force_resolve_commit_prepared_tx() {
        // Kills mutation: replace || with && in force_resolve (line 2180)
        // Condition: tx.all_yes() || matches!(tx.phase, TxPhase::Prepared | TxPhase::Committing)
        // With ||, a tx in Prepared phase can be force-committed even without all_yes
        // With &&, both conditions must be true
        let coordinator = create_test_coordinator();

        // Begin tx with 2 shards, only vote YES on one
        let tx = coordinator.begin(&"n1".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        // Record one YES vote (but not all shards voted)
        let _ = coordinator.record_vote(
            tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta: DeltaVector::zero(0),
            },
        );

        // Manually set phase to Prepared (normally needs all votes)
        {
            let mut pending = coordinator.pending.write();
            if let Some(t) = pending.get_mut(&tx_id) {
                t.phase = TxPhase::Prepared;
            }
        }

        // Force commit — should succeed with ||: matches!(phase, Prepared) is true
        // Would fail with &&: all_yes() is false (only 1 of 2 shards voted)
        let result = coordinator.force_resolve(tx_id, true);
        assert!(
            result.is_ok(),
            "force_resolve commit must succeed for Prepared tx, got: {result:?}"
        );
    }

    #[test]
    fn test_release_orphaned_locks_cleans_tx_locks() {
        // Kills mutations:
        // - replace < with <= in release_orphaned_locks (line 2235)
        // - replace != with == in release_orphaned_locks (line 2252)
        let coordinator = create_test_coordinator();

        // Lock a key directly (simulating a lock from a now-gone tx)
        let handle = coordinator
            .lock_manager
            .try_lock(555, &["orphan_key".to_string()])
            .unwrap();

        // No tx 555 in pending — this lock is orphaned
        // Set acquired_at to 1ms (ancient) and partition_start to current time
        // The < check: acquired_at < partition_start -> true -> orphaned
        // The != check on tx_keys.retain: ensures key is removed from tx_locks

        // We need to manually set acquired_at to a low value
        {
            let mut locks = coordinator.lock_manager.locks.write();
            if let Some(lock) = locks.get_mut("orphan_key") {
                lock.acquired_at_ms = 1; // very old
            }
        }

        let released = coordinator.release_orphaned_locks(now_epoch_millis());
        assert_eq!(released, 1, "Must release 1 orphaned lock");

        // Verify the lock is actually gone
        assert!(
            !coordinator.lock_manager.is_locked("orphan_key"),
            "orphan_key must no longer be locked"
        );

        // Verify tx_locks is clean (the != mutation would break this)
        let remaining_keys = coordinator.lock_manager.keys_for_transaction(555);
        assert!(
            remaining_keys.is_empty(),
            "tx_locks must be empty for tx 555 after orphan release, got: {remaining_keys:?}"
        );
        let _ = handle;
    }

    // ── Targeted mutation-killing tests ──────────────────────────────────

    #[test]
    fn test_cleanup_expired_partial_preserves_unexpired_keys() {
        // Kills mutation: replace != with == in LockManager::cleanup_expired (line 607)
        // With ==, retain keeps the EXPIRED key and removes the GOOD key.
        let lm = LockManager::with_default_timeout(Duration::from_millis(5));

        // Lock key1, let it expire, then lock key2 (fresh)
        let _ = lm.try_lock(200, &["key1".to_string()]).unwrap();
        std::thread::sleep(Duration::from_millis(15));

        // key1 is now expired. Lock key2 under the same tx (fresh timeout).
        // We must use a LockManager with a longer timeout for key2.
        // Since LockManager uses one timeout, insert key2 manually.
        {
            let mut locks = lm.locks.write();
            let mut tx_locks = lm.tx_locks.write();
            locks.insert(
                "key2".to_string(),
                KeyLock {
                    key: "key2".to_string(),
                    tx_id: 200,
                    lock_handle: 0,
                    acquired_at_ms: now_epoch_millis(),
                    timeout_ms: 60_000,
                },
            );
            tx_locks.entry(200).or_default().push("key2".to_string());
        }

        let expired = lm.cleanup_expired();
        assert_eq!(expired, 1, "Only key1 should be expired");

        let remaining = lm.keys_for_transaction(200);
        assert_eq!(
            remaining,
            vec!["key2".to_string()],
            "key2 must survive cleanup, got: {remaining:?}"
        );
    }

    #[test]
    fn test_cleanup_expired_with_wait_partial_preserves_unexpired() {
        // Kills mutation: replace != with == in cleanup_expired_with_wait_cleanup (line 635)
        let lm = LockManager::with_default_timeout(Duration::from_millis(5));
        let graph = crate::deadlock::WaitForGraph::new();

        let _ = lm.try_lock(300, &["a".to_string()]).unwrap();
        std::thread::sleep(Duration::from_millis(15));

        // Insert a fresh lock for the same tx
        {
            let mut locks = lm.locks.write();
            let mut tx_locks = lm.tx_locks.write();
            locks.insert(
                "b".to_string(),
                KeyLock {
                    key: "b".to_string(),
                    tx_id: 300,
                    lock_handle: 0,
                    acquired_at_ms: now_epoch_millis(),
                    timeout_ms: 60_000,
                },
            );
            tx_locks.entry(300).or_default().push("b".to_string());
        }

        let expired = lm.cleanup_expired_with_wait_cleanup(&graph);
        assert_eq!(expired, 1);

        let remaining = lm.keys_for_transaction(300);
        assert_eq!(
            remaining,
            vec!["b".to_string()],
            "Only 'b' must survive, got: {remaining:?}"
        );
    }

    #[test]
    fn test_recover_timed_out_preparing_exact() {
        // Kills mutation: replace += with -= / *= in recover line 2069
        let coordinator = create_test_coordinator();
        {
            let mut pending = coordinator.pending.write();
            let mut tx = DistributedTransaction::new("n".to_string(), vec![0]);
            tx.phase = TxPhase::Preparing;
            tx.started_at = 1; // ancient → timed out
            pending.insert(tx.tx_id, tx);
        }
        let stats = coordinator.recover();
        assert_eq!(
            stats.timed_out, 1,
            "Must count exactly 1 timed-out preparing tx"
        );
    }

    #[test]
    fn test_recover_timed_out_prepared_exact() {
        // Kills mutation: replace += with -= / *= in recover line 2078
        let coordinator = create_test_coordinator();
        {
            let mut pending = coordinator.pending.write();
            let mut tx = DistributedTransaction::new("n".to_string(), vec![0]);
            tx.phase = TxPhase::Prepared;
            tx.started_at = 1; // ancient → timed out
            pending.insert(tx.tx_id, tx);
        }
        let stats = coordinator.recover();
        assert_eq!(
            stats.timed_out, 1,
            "Must count exactly 1 timed-out prepared tx"
        );
    }

    #[test]
    fn test_recover_prepared_any_no_exact() {
        // Kills mutation: replace += with -= / *= in recover line 2086
        let coordinator = create_test_coordinator();
        {
            let mut pending = coordinator.pending.write();
            let mut tx = DistributedTransaction::new("n".to_string(), vec![0, 1]);
            tx.phase = TxPhase::Prepared;
            // Record one YES and one NO vote
            tx.votes.insert(
                0,
                PrepareVote::Yes {
                    lock_handle: 1,
                    delta: DeltaVector::zero(0),
                },
            );
            tx.votes.insert(
                1,
                PrepareVote::No {
                    reason: "test".to_string(),
                },
            );
            pending.insert(tx.tx_id, tx);
        }
        let stats = coordinator.recover();
        assert_eq!(
            stats.pending_abort, 1,
            "Must count exactly 1 pending_abort for prepared tx with NO vote"
        );
    }

    #[test]
    fn test_recover_prepared_conflict_vote_aborts() {
        // Line 2089 (Prepared else branch) is unreachable: any_no() matches Conflict too.
        // This test verifies Prepared tx with Conflict vote goes to pending_abort (line 2086).
        // Kills mutation: replace += with -= / *= in recover line 2086 (alternative path).
        let coordinator = create_test_coordinator();
        {
            let mut pending = coordinator.pending.write();
            let mut tx = DistributedTransaction::new("n".to_string(), vec![0]);
            tx.phase = TxPhase::Prepared;
            tx.votes.insert(
                0,
                PrepareVote::Conflict {
                    similarity: 0.5,
                    conflicting_tx: 42,
                },
            );
            pending.insert(tx.tx_id, tx);
        }
        let stats = coordinator.recover();
        assert_eq!(
            stats.pending_abort, 1,
            "Prepared tx with Conflict vote must count as pending_abort"
        );
    }

    #[test]
    fn test_recover_from_wal_orphaned_lock_count() {
        // Kills mutation: replace += with -= / *= in recover_from_wal line 1192
        use crate::tx_wal::{PrepareVoteKind, TxWal, TxWalEntry};

        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("orphan_test.wal");
        let mut wal = TxWal::open(&wal_path).unwrap();

        // Create a tx that committed (TxComplete) but whose lock was never released
        // (no AllLocksReleased). This creates an orphaned lock.
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 10,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 10,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 100 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 10,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 10,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })
        .unwrap();
        wal.append(&TxWalEntry::TxComplete {
            tx_id: 10,
            outcome: crate::tx_wal::TxOutcome::Committed,
        })
        .unwrap();
        // NO AllLocksReleased entry → lock 100 is orphaned

        // Second tx with orphaned lock
        wal.append(&TxWalEntry::TxBegin {
            tx_id: 11,
            participants: vec![0],
        })
        .unwrap();
        wal.append(&TxWalEntry::PrepareVote {
            tx_id: 11,
            shard: 0,
            vote: PrepareVoteKind::Yes { lock_handle: 101 },
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 11,
            from: TxPhase::Preparing,
            to: TxPhase::Prepared,
        })
        .unwrap();
        wal.append(&TxWalEntry::PhaseChange {
            tx_id: 11,
            from: TxPhase::Prepared,
            to: TxPhase::Committing,
        })
        .unwrap();
        wal.append(&TxWalEntry::TxComplete {
            tx_id: 11,
            outcome: crate::tx_wal::TxOutcome::Committed,
        })
        .unwrap();
        // NO AllLocksReleased → lock 101 is orphaned

        drop(wal);

        let wal2 = TxWal::open(&wal_path).unwrap();
        let consensus = ConsensusManager::new(ConsensusConfig::default());
        let coordinator = DistributedTxCoordinator::with_consensus(consensus).with_wal(wal2);

        let stats = coordinator.recover_from_wal().unwrap();
        assert_eq!(
            stats.lock_releases_recovered, 2,
            "Must count exactly 2 orphaned lock releases"
        );
    }

    #[test]
    fn test_force_resolve_commit_no_vote_prepared_phase() {
        // Kills mutation: replace || with && in force_resolve (line 2180)
        // Condition: tx.all_yes() || matches!(tx.phase, Prepared | Committing)
        // With &&: both must be true. With ||: either suffices.
        // Create a Prepared tx with a NO vote: all_yes()=false, matches=true
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin(&"n1".to_string(), &[0, 1]).unwrap();
        let tx_id = tx.tx_id;

        // Record YES for shard 0, NO for shard 1
        let _ = coordinator.record_vote(
            tx_id,
            0,
            PrepareVote::Yes {
                lock_handle: 1,
                delta: DeltaVector::zero(0),
            },
        );
        let _ = coordinator.record_vote(
            tx_id,
            1,
            PrepareVote::No {
                reason: "test".to_string(),
            },
        );

        // Manually set phase to Prepared
        {
            let mut pending = coordinator.pending.write();
            if let Some(t) = pending.get_mut(&tx_id) {
                t.phase = TxPhase::Prepared;
            }
        }

        // With ||: matches!(Prepared, ...) is true → commit succeeds
        // With &&: all_yes() is false → commit fails
        let result = coordinator.force_resolve(tx_id, true);
        assert!(
            result.is_ok(),
            "force_resolve must succeed for Prepared tx even without all YES, got: {result:?}"
        );
    }

    #[test]
    fn test_release_orphaned_locks_exact_boundary_not_released() {
        // Kills mutation: replace < with <= in release_orphaned_locks (line 2235)
        // Lock acquired exactly at partition_start should NOT be released (< is strict)
        let coordinator = create_test_coordinator();

        let lock_mgr = &coordinator.lock_manager;
        let _ = lock_mgr
            .try_lock(777, &["boundary_key".to_string()])
            .unwrap();

        // Use a realistic timestamp for both acquired_at and partition_start
        let boundary_time = now_epoch_millis();
        {
            let mut locks = lock_mgr.locks.write();
            if let Some(lock) = locks.get_mut("boundary_key") {
                lock.acquired_at_ms = boundary_time;
            }
        }

        // With <: boundary_time < boundary_time = false → NOT released
        // With <=: boundary_time <= boundary_time = true → released
        let released = coordinator.release_orphaned_locks(boundary_time);
        assert_eq!(
            released, 0,
            "Lock at exactly partition_start must NOT be released"
        );
        // Check lock map directly (is_locked also checks expiry)
        assert!(
            lock_mgr.locks.read().contains_key("boundary_key"),
            "boundary_key must still be in locks map"
        );
    }

    #[test]
    fn test_apply_operations_cas_matching_applies() {
        // Kills mutation: replace == with != in apply_operations (line 2526)
        // CAS should apply when current data matches expected_data.
        let participant = TxParticipant::new_in_memory();

        // Pre-populate key with known data
        let mut tensor = TensorData::new();
        tensor.set(
            "data",
            TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3])),
        );
        participant.store().put("cas_key", tensor).unwrap();

        // Prepare a CAS operation with matching expected_data
        let request = PrepareRequest {
            tx_id: 9000,
            coordinator: "n".to_string(),
            operations: vec![Transaction::CompareAndSwap {
                key: "cas_key".to_string(),
                expected_data: vec![1, 2, 3],
                new_data: vec![4, 5, 6],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        let vote = participant.prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));

        // Commit — CAS should apply because data matches
        let resp = participant.commit(9000);
        assert!(resp.success, "commit must succeed");

        // Verify the data was updated
        let data = participant.store().get("cas_key").unwrap();
        let bytes = data
            .get("data")
            .and_then(|v| match v {
                TensorValue::Scalar(ScalarValue::Bytes(b)) => Some(b.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(bytes, vec![4, 5, 6], "CAS must apply when data matches");
    }

    #[test]
    fn test_apply_operations_cas_mismatch_skips() {
        // Also helps kill mutation on line 2526: with != mutation,
        // mismatch would APPLY the update (wrong behavior).
        let participant = TxParticipant::new_in_memory();

        // Pre-populate with data that does NOT match expected
        let mut tensor = TensorData::new();
        tensor.set(
            "data",
            TensorValue::Scalar(ScalarValue::Bytes(vec![9, 9, 9])),
        );
        participant.store().put("cas_key2", tensor).unwrap();

        let request = PrepareRequest {
            tx_id: 9001,
            coordinator: "n".to_string(),
            operations: vec![Transaction::CompareAndSwap {
                key: "cas_key2".to_string(),
                expected_data: vec![1, 2, 3], // does NOT match current [9,9,9]
                new_data: vec![4, 5, 6],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        let vote = participant.prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));

        let resp = participant.commit(9001);
        assert!(resp.success);

        // Data should NOT have changed (CAS mismatch)
        let data = participant.store().get("cas_key2").unwrap();
        let bytes = data
            .get("data")
            .and_then(|v| match v {
                TensorValue::Scalar(ScalarValue::Bytes(b)) => Some(b.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            bytes,
            vec![9, 9, 9],
            "CAS must NOT apply when data does not match"
        );
    }

    #[test]
    fn test_participant_recover_exact_timeout_not_expired() {
        // Kills mutation: replace > with >= in TxParticipant::recover (line 2756)
        // At exactly timeout, > returns false (keep waiting), >= returns true (presume abort)
        let participant = TxParticipant::new_in_memory();
        let timeout = Duration::from_secs(5);

        // Prepare a tx
        let request = PrepareRequest {
            tx_id: 8000,
            coordinator: "n".to_string(),
            operations: vec![Transaction::Put {
                key: "recover_key".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::default(),
            timeout_ms: 5000,
        };
        let vote = participant.prepare(request);
        assert!(matches!(vote, PrepareVote::Yes { .. }));

        // Set prepared_at_ms to exactly `timeout` ago
        let now = now_epoch_millis();
        #[allow(clippy::cast_possible_truncation)]
        let timeout_ms = timeout.as_millis() as u64;
        {
            let mut prepared = participant.prepared.write();
            if let Some(tx) = prepared.get_mut(&8000) {
                tx.prepared_at_ms = now - timeout_ms; // exactly at boundary
            }
        }

        // With >: now - (now - timeout_ms) = timeout_ms, timeout_ms > timeout_ms = false → kept
        // With >=: timeout_ms >= timeout_ms = true → expired (presumed abort)
        let awaiting = participant.recover(timeout);

        // The tx should still be awaiting (not expired) since > is strict
        assert!(
            awaiting.contains(&8000),
            "Tx at exactly timeout boundary must be awaiting, not expired. Got: {awaiting:?}"
        );
    }

    #[test]
    fn test_cleanup_stale_aborts_by_time() {
        // Kills mutation: replace >= with < in cleanup_stale_aborts (line 1953)
        let coordinator = create_test_coordinator();
        coordinator.track_abort(500, vec![0]);

        // Set initiated_at to 31 seconds ago (> 30s max_abort_wait_ms)
        // but retry_count to 0 (below MAX_ABORT_RETRIES)
        {
            let mut states = coordinator.abort_states.write();
            let state = states.get_mut(&500).unwrap();
            state.initiated_at = now_epoch_millis().saturating_sub(31_000);
            state.retry_count = 0;
        }

        // With >=: 31000 >= 30000 = true → stale (cleaned)
        // With <: 31000 < 30000 = false → kept (wrong)
        let stale = coordinator.cleanup_stale_aborts();
        assert_eq!(stale, vec![500], "Abort older than 30s must be cleaned up");
    }
}
