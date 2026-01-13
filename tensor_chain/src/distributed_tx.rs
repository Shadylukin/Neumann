//! Cross-shard distributed transactions with 2PC and delta-based conflict detection.
//!
//! Coordinates transactions spanning multiple shards using two-phase commit:
//! - Phase 1 (PREPARE): Acquire locks, compute delta, check conflicts
//! - Phase 2 (COMMIT/ABORT): Finalize or rollback based on votes
//!
//! Tensor-native optimization: Orthogonal deltas can commit in parallel
//! without coordination using vector similarity.

use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use parking_lot::RwLock;
use tensor_store::SparseVector;

use crate::{
    block::{NodeId, Transaction},
    consensus::{ConsensusManager, DeltaVector},
    error::{ChainError, Result},
};

/// Shard identifier.
pub type ShardId = usize;

/// Distributed transaction ID counter.
static DTX_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a new unique distributed transaction ID.
fn next_dtx_id() -> u64 {
    DTX_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Phase of a distributed transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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

/// A distributed transaction spanning multiple shards.
#[derive(Debug, Clone)]
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
    /// Create a new distributed transaction.
    pub fn new(coordinator: NodeId, participants: Vec<ShardId>) -> Self {
        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            tx_id: next_dtx_id(),
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

    /// Add operations for a shard.
    pub fn add_operations(&mut self, shard: ShardId, ops: Vec<Transaction>) {
        self.operations.insert(shard, ops);
    }

    /// Add delta for a shard.
    pub fn add_delta(&mut self, shard: ShardId, delta: DeltaVector) {
        self.deltas.insert(shard, delta);
    }

    /// Record a vote from a participant.
    pub fn record_vote(&mut self, shard: ShardId, vote: PrepareVote) {
        self.votes.insert(shard, vote);
    }

    /// Check if all participants have voted.
    pub fn all_voted(&self) -> bool {
        self.participants
            .iter()
            .all(|shard| self.votes.contains_key(shard))
    }

    /// Check if all votes are YES.
    pub fn all_yes(&self) -> bool {
        self.votes
            .values()
            .all(|v| matches!(v, PrepareVote::Yes { .. }))
    }

    /// Check if any vote is NO or Conflict.
    pub fn any_no(&self) -> bool {
        self.votes
            .values()
            .any(|v| matches!(v, PrepareVote::No { .. } | PrepareVote::Conflict { .. }))
    }

    /// Check if the transaction has timed out.
    pub fn is_timed_out(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now - self.started_at > self.timeout_ms
    }

    /// Get all affected keys across all shards.
    pub fn affected_keys(&self) -> HashSet<String> {
        self.deltas
            .values()
            .flat_map(|d| d.affected_keys.iter().cloned())
            .collect()
    }

    /// Get the merged delta across all shards.
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
#[derive(Debug, Clone)]
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
#[derive(Debug)]
pub struct KeyLock {
    /// Key being locked.
    pub key: String,
    /// Transaction ID holding the lock.
    pub tx_id: u64,
    /// Lock handle for release.
    pub lock_handle: u64,
    /// When the lock was acquired.
    pub acquired_at: Instant,
    /// Lock timeout.
    pub timeout: Duration,
}

impl KeyLock {
    /// Check if the lock has expired.
    pub fn is_expired(&self) -> bool {
        self.acquired_at.elapsed() > self.timeout
    }
}

/// Lock handle counter.
static LOCK_COUNTER: AtomicU64 = AtomicU64::new(1);

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
    /// Create a new lock manager.
    pub fn new() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            tx_locks: RwLock::new(HashMap::new()),
            default_timeout: Duration::from_secs(30),
        }
    }

    /// Try to acquire locks for a set of keys.
    /// Returns a lock handle if successful, or the conflicting tx_id if not.
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
        let lock_handle = LOCK_COUNTER.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();

        for key in keys {
            locks.insert(
                key.clone(),
                KeyLock {
                    key: key.clone(),
                    tx_id,
                    lock_handle,
                    acquired_at: now,
                    timeout: self.default_timeout,
                },
            );
        }

        tx_locks
            .entry(tx_id)
            .or_default()
            .extend(keys.iter().cloned());

        Ok(lock_handle)
    }

    /// Release locks for a transaction.
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

    /// Release locks by handle.
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

    /// Check if a key is locked.
    pub fn is_locked(&self, key: &str) -> bool {
        let locks = self.locks.read();
        locks.get(key).is_some_and(|lock| !lock.is_expired())
    }

    /// Get the transaction holding a lock.
    pub fn lock_holder(&self, key: &str) -> Option<u64> {
        let locks = self.locks.read();
        locks
            .get(key)
            .filter(|lock| !lock.is_expired())
            .map(|lock| lock.tx_id)
    }

    /// Clean up expired locks.
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

        expired.len()
    }

    /// Get the number of active locks.
    pub fn active_lock_count(&self) -> usize {
        self.locks.read().len()
    }
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
}

impl Default for DistributedTxConfig {
    fn default() -> Self {
        Self {
            prepare_timeout_ms: 5000,
            commit_timeout_ms: 10000,
            max_concurrent: 100,
            orthogonal_threshold: 0.1,
            optimistic_locking: true,
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
}

impl DistributedTxStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the commit rate.
    pub fn commit_rate(&self) -> f32 {
        let started = self.started.load(Ordering::Relaxed);
        if started == 0 {
            return 0.0;
        }
        self.committed.load(Ordering::Relaxed) as f32 / started as f32
    }

    /// Get the conflict rate.
    pub fn conflict_rate(&self) -> f32 {
        let started = self.started.load(Ordering::Relaxed);
        if started == 0 {
            return 0.0;
        }
        self.conflicts.load(Ordering::Relaxed) as f32 / started as f32
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
    /// Configuration.
    config: DistributedTxConfig,
    /// Statistics.
    pub stats: DistributedTxStats,
}

impl DistributedTxCoordinator {
    /// Create a new coordinator.
    pub fn new(consensus: ConsensusManager, config: DistributedTxConfig) -> Self {
        Self {
            pending: RwLock::new(HashMap::new()),
            consensus,
            lock_manager: LockManager::new(),
            config,
            stats: DistributedTxStats::new(),
        }
    }

    /// Create with default configuration.
    pub fn with_consensus(consensus: ConsensusManager) -> Self {
        Self::new(consensus, DistributedTxConfig::default())
    }

    /// Begin a new distributed transaction.
    pub fn begin(
        &self,
        coordinator: NodeId,
        participants: Vec<ShardId>,
    ) -> Result<DistributedTransaction> {
        // Check concurrent transaction limit
        if self.pending.read().len() >= self.config.max_concurrent {
            return Err(ChainError::TransactionFailed(
                "too many concurrent distributed transactions".to_string(),
            ));
        }

        let tx = DistributedTransaction::new(coordinator, participants);
        self.pending.write().insert(tx.tx_id, tx.clone());
        self.stats.started.fetch_add(1, Ordering::Relaxed);

        Ok(tx)
    }

    /// Get a pending transaction.
    pub fn get(&self, tx_id: u64) -> Option<DistributedTransaction> {
        self.pending.read().get(&tx_id).cloned()
    }

    /// Process prepare request and return vote.
    pub fn handle_prepare(&self, request: PrepareRequest) -> PrepareVote {
        // Extract affected keys from operations
        let keys: Vec<String> = request
            .operations
            .iter()
            .map(|op| op.affected_key().to_string())
            .collect();

        // Try to acquire locks
        let lock_handle = match self.lock_manager.try_lock(request.tx_id, &keys) {
            Ok(handle) => handle,
            Err(conflicting_tx) => {
                return PrepareVote::Conflict {
                    similarity: 1.0, // Same keys = maximum conflict
                    conflicting_tx,
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

        PrepareVote::Yes { lock_handle, delta }
    }

    /// Record vote and check if ready to commit/abort.
    pub fn record_vote(&self, tx_id: u64, shard: ShardId, vote: PrepareVote) -> Option<TxPhase> {
        let mut pending = self.pending.write();
        let tx = pending.get_mut(&tx_id)?;

        // Extract delta from YES vote
        if let PrepareVote::Yes { delta, .. } = &vote {
            tx.add_delta(shard, delta.clone());
        }

        tx.record_vote(shard, vote);

        if tx.all_voted() {
            if tx.all_yes() {
                // Check cross-shard conflicts
                if let Some(merged) = tx.merged_delta() {
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
                                    tx.phase = TxPhase::Aborting;
                                    self.stats.conflicts.fetch_add(1, Ordering::Relaxed);
                                    return Some(TxPhase::Aborting);
                                }
                            }
                        }
                    }

                    // All orthogonal - can merge and commit
                    self.stats.orthogonal_merges.fetch_add(1, Ordering::Relaxed);
                    drop(merged);
                }

                tx.phase = TxPhase::Prepared;
                Some(TxPhase::Prepared)
            } else {
                tx.phase = TxPhase::Aborting;
                if tx
                    .votes
                    .values()
                    .any(|v| matches!(v, PrepareVote::Conflict { .. }))
                {
                    self.stats.conflicts.fetch_add(1, Ordering::Relaxed);
                }
                Some(TxPhase::Aborting)
            }
        } else {
            None
        }
    }

    /// Commit a prepared transaction.
    pub fn commit(&self, tx_id: u64) -> Result<()> {
        let mut pending = self.pending.write();
        let tx = pending.get_mut(&tx_id).ok_or_else(|| {
            ChainError::TransactionFailed(format!("transaction {} not found", tx_id))
        })?;

        if tx.phase != TxPhase::Prepared {
            return Err(ChainError::TransactionFailed(format!(
                "transaction {} not in prepared phase",
                tx_id
            )));
        }

        tx.phase = TxPhase::Committing;

        // Release all locks
        for vote in tx.votes.values() {
            if let PrepareVote::Yes { lock_handle, .. } = vote {
                self.lock_manager.release_by_handle(*lock_handle);
            }
        }

        tx.phase = TxPhase::Committed;
        self.stats.committed.fetch_add(1, Ordering::Relaxed);

        // Remove from pending
        pending.remove(&tx_id);

        Ok(())
    }

    /// Abort a transaction.
    pub fn abort(&self, tx_id: u64, _reason: &str) -> Result<()> {
        let mut pending = self.pending.write();
        let tx = pending.get_mut(&tx_id).ok_or_else(|| {
            ChainError::TransactionFailed(format!("transaction {} not found", tx_id))
        })?;

        tx.phase = TxPhase::Aborting;

        // Release all locks
        for vote in tx.votes.values() {
            if let PrepareVote::Yes { lock_handle, .. } = vote {
                self.lock_manager.release_by_handle(*lock_handle);
            }
        }

        tx.phase = TxPhase::Aborted;
        self.stats.aborted.fetch_add(1, Ordering::Relaxed);

        // Remove from pending
        pending.remove(&tx_id);

        Ok(())
    }

    /// Clean up timed out transactions.
    pub fn cleanup_timeouts(&self) -> Vec<u64> {
        let mut pending = self.pending.write();
        let timed_out: Vec<_> = pending
            .iter()
            .filter(|(_, tx)| tx.is_timed_out())
            .map(|(id, _)| *id)
            .collect();

        for tx_id in &timed_out {
            if let Some(tx) = pending.remove(tx_id) {
                // Release locks
                for vote in tx.votes.values() {
                    if let PrepareVote::Yes { lock_handle, .. } = vote {
                        self.lock_manager.release_by_handle(*lock_handle);
                    }
                }
                self.stats.timed_out.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Also cleanup expired locks
        self.lock_manager.cleanup_expired();

        timed_out
    }

    /// Get the number of pending transactions.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
    }

    /// Get the lock manager.
    pub fn lock_manager(&self) -> &LockManager {
        &self.lock_manager
    }

    /// Get statistics.
    pub fn stats(&self) -> &DistributedTxStats {
        &self.stats
    }
}

/// Transaction participant on a shard.
#[derive(Debug)]
pub struct TxParticipant {
    /// Prepared transactions awaiting commit/abort.
    pub prepared: RwLock<HashMap<u64, PreparedTx>>,
    /// Lock manager for this shard.
    pub locks: LockManager,
}

/// A prepared transaction on a participant.
#[derive(Debug, Clone)]
pub struct PreparedTx {
    /// Transaction ID.
    pub tx_id: u64,
    /// Lock handle.
    pub lock_handle: u64,
    /// Operations to execute.
    pub operations: Vec<Transaction>,
    /// Delta embedding.
    pub delta: DeltaVector,
    /// When the transaction was prepared.
    pub prepared_at: Instant,
}

impl Default for TxParticipant {
    fn default() -> Self {
        Self::new()
    }
}

impl TxParticipant {
    /// Create a new participant.
    pub fn new() -> Self {
        Self {
            prepared: RwLock::new(HashMap::new()),
            locks: LockManager::new(),
        }
    }

    /// Handle prepare request.
    pub fn prepare(&self, request: PrepareRequest) -> PrepareVote {
        let keys: Vec<String> = request
            .operations
            .iter()
            .map(|op| op.affected_key().to_string())
            .collect();

        // Try to acquire locks
        let lock_handle = match self.locks.try_lock(request.tx_id, &keys) {
            Ok(handle) => handle,
            Err(conflicting_tx) => {
                return PrepareVote::Conflict {
                    similarity: 1.0,
                    conflicting_tx,
                };
            },
        };

        let delta = DeltaVector::from_sparse(
            request.delta_embedding,
            keys.into_iter().collect(),
            request.tx_id,
        );

        // Store prepared state
        self.prepared.write().insert(
            request.tx_id,
            PreparedTx {
                tx_id: request.tx_id,
                lock_handle,
                operations: request.operations,
                delta: delta.clone(),
                prepared_at: Instant::now(),
            },
        );

        PrepareVote::Yes { lock_handle, delta }
    }

    /// Handle commit request.
    pub fn commit(&self, tx_id: u64) -> TxResponse {
        let mut prepared = self.prepared.write();

        if let Some(tx) = prepared.remove(&tx_id) {
            self.locks.release_by_handle(tx.lock_handle);
            TxResponse {
                tx_id,
                success: true,
                error: None,
            }
        } else {
            TxResponse {
                tx_id,
                success: false,
                error: Some("transaction not found".to_string()),
            }
        }
    }

    /// Handle abort request.
    pub fn abort(&self, tx_id: u64) -> TxResponse {
        let mut prepared = self.prepared.write();

        if let Some(tx) = prepared.remove(&tx_id) {
            self.locks.release_by_handle(tx.lock_handle);
        }

        TxResponse {
            tx_id,
            success: true,
            error: None,
        }
    }

    /// Get prepared transaction count.
    pub fn prepared_count(&self) -> usize {
        self.prepared.read().len()
    }

    /// Clean up stale prepared transactions.
    pub fn cleanup_stale(&self, timeout: Duration) -> Vec<u64> {
        let mut prepared = self.prepared.write();
        let stale: Vec<_> = prepared
            .iter()
            .filter(|(_, tx)| tx.prepared_at.elapsed() > timeout)
            .map(|(id, _)| *id)
            .collect();

        for tx_id in &stale {
            if let Some(tx) = prepared.remove(tx_id) {
                self.locks.release_by_handle(tx.lock_handle);
            }
        }

        stale
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

        let tx = coordinator.begin("node1".to_string(), vec![0, 1]).unwrap();

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

        coordinator.begin("node1".to_string(), vec![0]).unwrap();
        coordinator.begin("node1".to_string(), vec![1]).unwrap();

        // Third should fail
        let result = coordinator.begin("node1".to_string(), vec![2]);
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

        let vote = coordinator.handle_prepare(request);

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
        let tx = coordinator.begin("node1".to_string(), vec![0, 1]).unwrap();

        let delta0 = DeltaVector::new(
            vec![1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );
        let delta1 = DeltaVector::new(
            vec![0.0, 1.0, 0.0],
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
        assert!(result.is_none()); // Not all voted yet

        // Vote from shard 1
        let result = coordinator.record_vote(
            tx.tx_id,
            1,
            PrepareVote::Yes {
                lock_handle: 2,
                delta: delta1,
            },
        );
        assert_eq!(result, Some(TxPhase::Prepared));
    }

    #[test]
    fn test_voting_any_no() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin("node1".to_string(), vec![0, 1]).unwrap();

        let delta = DeltaVector::new(
            vec![1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        coordinator.record_vote(
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
        assert_eq!(result, Some(TxPhase::Aborting));
    }

    #[test]
    fn test_commit_flow() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin("node1".to_string(), vec![0]).unwrap();

        let delta = DeltaVector::new(
            vec![1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        coordinator.record_vote(
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
        let tx = coordinator.begin("node1".to_string(), vec![0]).unwrap();

        coordinator.abort(tx.tx_id, "test abort").unwrap();

        assert_eq!(coordinator.pending_count(), 0);
        assert_eq!(coordinator.stats.aborted.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_participant_prepare_commit() {
        let participant = TxParticipant::new();

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
        let participant = TxParticipant::new();

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
            vec![1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            1,
        );
        let delta1 = DeltaVector::new(
            vec![0.0, 1.0, 0.0],
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
            vec![1.0, 0.0],
            ["key1", "key2"].iter().map(|s| s.to_string()).collect(),
            1,
        );
        let delta1 = DeltaVector::new(
            vec![0.0, 1.0],
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
                delta: DeltaVector::new(vec![1.0], HashSet::new(), 1),
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
        let tx = coordinator.begin("node1".to_string(), vec![0]).unwrap();

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
        let tx = coordinator.begin("node1".to_string(), vec![0]).unwrap();

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

        let vote1 = coordinator.handle_prepare(request1);
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

        let vote2 = coordinator.handle_prepare(request2);
        assert!(matches!(vote2, PrepareVote::Conflict { .. }));
    }

    #[test]
    fn test_participant_prepare_conflict() {
        let participant = TxParticipant::new();

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
        let participant = TxParticipant::new();
        let response = participant.commit(999);
        assert!(!response.success);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_participant_abort_not_prepared() {
        let participant = TxParticipant::new();
        // Abort should succeed even if not prepared
        let response = participant.abort(999);
        assert!(response.success);
    }

    #[test]
    fn test_participant_cleanup_stale() {
        let participant = TxParticipant::new();

        let request = PrepareRequest {
            tx_id: 1,
            coordinator: "node1".to_string(),
            operations: vec![Transaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            }],
            delta_embedding: SparseVector::from_dense(&[1.0]),
            timeout_ms: 5000,
        };

        participant.prepare(request);
        assert_eq!(participant.prepared_count(), 1);

        // Cleanup with zero timeout
        let stale = participant.cleanup_stale(Duration::from_secs(0));
        assert_eq!(stale.len(), 1);
        assert_eq!(participant.prepared_count(), 0);
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
    }

    #[test]
    fn test_key_lock_expiry() {
        let lock = KeyLock {
            key: "test".to_string(),
            tx_id: 1,
            lock_handle: 1,
            acquired_at: Instant::now() - Duration::from_secs(100),
            timeout: Duration::from_secs(1),
        };
        assert!(lock.is_expired());

        let lock2 = KeyLock {
            key: "test".to_string(),
            tx_id: 1,
            lock_handle: 1,
            acquired_at: Instant::now(),
            timeout: Duration::from_secs(100),
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
        let tx = coordinator.begin("node1".to_string(), vec![0, 1]).unwrap();

        let delta = DeltaVector::new(
            vec![1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        coordinator.record_vote(
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

        assert_eq!(result, Some(TxPhase::Aborting));
        assert!(coordinator.stats.conflicts.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_tx_participant_default() {
        let participant = TxParticipant::default();
        assert_eq!(participant.prepared_count(), 0);
    }

    #[test]
    fn test_cross_shard_conflict_detection() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin("node1".to_string(), vec![0, 1]).unwrap();

        // Two shards with overlapping keys and similar deltas
        let delta0 = DeltaVector::new(
            vec![1.0, 0.0, 0.0],
            ["shared_key"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );
        let delta1 = DeltaVector::new(
            vec![1.0, 0.1, 0.0], // Similar to delta0
            ["shared_key"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        coordinator.record_vote(
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
        assert_eq!(result, Some(TxPhase::Aborting));
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
        let tx = coordinator.begin("node1".to_string(), vec![0]).unwrap();
        {
            let mut pending = coordinator.pending.write();
            if let Some(t) = pending.get_mut(&tx.tx_id) {
                let delta = DeltaVector::new(
                    vec![1.0, 0.0, 0.0],
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

        let vote = coordinator.handle_prepare(request);
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
                delta: DeltaVector::new(vec![1.0], HashSet::new(), 1),
            },
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_abort_releases_locks() {
        let coordinator = create_test_coordinator();
        let tx = coordinator.begin("node1".to_string(), vec![0]).unwrap();

        let delta = DeltaVector::new(
            vec![1.0, 0.0, 0.0],
            ["key1"].iter().map(|s| s.to_string()).collect(),
            tx.tx_id,
        );

        // Vote and record
        coordinator.record_vote(
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

        let tx = coordinator.begin("node1".to_string(), vec![0]).unwrap();

        // Add a vote with lock
        {
            let mut pending = coordinator.pending.write();
            if let Some(t) = pending.get_mut(&tx.tx_id) {
                t.record_vote(
                    0,
                    PrepareVote::Yes {
                        lock_handle: 1,
                        delta: DeltaVector::new(vec![1.0], HashSet::new(), tx.tx_id),
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
            delta: DeltaVector::new(vec![1.0], HashSet::new(), 1),
            prepared_at: Instant::now(),
        };
        let _ = format!("{:?}", tx);
    }
}
