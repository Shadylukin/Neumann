//! Transaction workspace for isolated execution and atomic commit.
//!
//! Provides ACID transaction semantics:
//! - Isolation via checkpoint-based workspace
//! - Atomic commit via block append
//! - Rollback via checkpoint restore
//!
//! Each transaction creates an isolated workspace where operations
//! are tracked as a delta from the starting state.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tensor_store::TensorStore;

use crate::block::Transaction as ChainTransaction;
use crate::error::{ChainError, Result};

/// Monotonic transaction ID counter.
static TX_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a new unique transaction ID.
fn next_tx_id() -> u64 {
    TX_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Transaction state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Transaction is active (operations can be added).
    Active,
    /// Transaction is being committed.
    Committing,
    /// Transaction has been committed.
    Committed,
    /// Transaction has been rolled back.
    RolledBack,
    /// Transaction has failed.
    Failed,
}

/// A workspace for isolated transaction execution.
#[derive(Debug)]
pub struct TransactionWorkspace {
    /// Unique transaction ID.
    id: u64,

    /// Snapshot of the store at transaction start.
    checkpoint_bytes: Vec<u8>,

    /// Operations executed in this transaction.
    operations: RwLock<Vec<ChainTransaction>>,

    /// Keys affected by this transaction.
    affected_keys: RwLock<HashSet<String>>,

    /// Current state.
    state: RwLock<TransactionState>,

    /// Timestamp when transaction started.
    started_at: u64,
}

impl TransactionWorkspace {
    /// Create a new transaction workspace from the current store state.
    pub fn begin(store: &TensorStore) -> Result<Self> {
        let checkpoint_bytes = store
            .snapshot_bytes()
            .map_err(|e| ChainError::WorkspaceError(e.to_string()))?;

        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Ok(Self {
            id: next_tx_id(),
            checkpoint_bytes,
            operations: RwLock::new(Vec::new()),
            affected_keys: RwLock::new(HashSet::new()),
            state: RwLock::new(TransactionState::Active),
            started_at,
        })
    }

    /// Get the transaction ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the current state.
    pub fn state(&self) -> TransactionState {
        *self.state.read()
    }

    /// Get when the transaction started.
    pub fn started_at(&self) -> u64 {
        self.started_at
    }

    /// Check if the transaction is active.
    pub fn is_active(&self) -> bool {
        *self.state.read() == TransactionState::Active
    }

    /// Add an operation to this transaction.
    pub fn add_operation(&self, op: ChainTransaction) -> Result<()> {
        if !self.is_active() {
            return Err(ChainError::TransactionFailed(
                "transaction is not active".to_string(),
            ));
        }

        // Track affected key
        let key = op.affected_key().to_string();
        self.affected_keys.write().insert(key);

        // Record operation
        self.operations.write().push(op);

        Ok(())
    }

    /// Get all operations in this transaction.
    pub fn operations(&self) -> Vec<ChainTransaction> {
        self.operations.read().clone()
    }

    /// Get all affected keys.
    pub fn affected_keys(&self) -> HashSet<String> {
        self.affected_keys.read().clone()
    }

    /// Get the number of operations.
    pub fn operation_count(&self) -> usize {
        self.operations.read().len()
    }

    /// Mark transaction as committing.
    pub fn mark_committing(&self) -> Result<()> {
        let mut state = self.state.write();
        if *state != TransactionState::Active {
            return Err(ChainError::TransactionFailed(format!(
                "cannot commit transaction in state {:?}",
                *state
            )));
        }
        *state = TransactionState::Committing;
        Ok(())
    }

    /// Mark transaction as committed.
    pub fn mark_committed(&self) {
        *self.state.write() = TransactionState::Committed;
    }

    /// Mark transaction as failed.
    pub fn mark_failed(&self) {
        *self.state.write() = TransactionState::Failed;
    }

    /// Rollback this transaction, restoring the store to its original state.
    pub fn rollback(&self, store: &TensorStore) -> Result<()> {
        let mut state = self.state.write();
        if *state == TransactionState::Committed {
            return Err(ChainError::TransactionFailed(
                "cannot rollback committed transaction".to_string(),
            ));
        }

        store
            .restore_from_bytes(&self.checkpoint_bytes)
            .map_err(|e| ChainError::WorkspaceError(e.to_string()))?;

        *state = TransactionState::RolledBack;
        Ok(())
    }

    /// Get the checkpoint bytes (for delta computation).
    pub fn checkpoint_bytes(&self) -> &[u8] {
        &self.checkpoint_bytes
    }
}

/// Delta between two transaction states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionDelta {
    /// Transaction ID.
    pub tx_id: u64,

    /// Parent transaction ID (for lineage tracking).
    pub parent_tx_id: Option<u64>,

    /// Semantic embedding of the change.
    pub delta_embedding: Vec<f32>,

    /// Keys affected by this transaction.
    pub affected_keys: HashSet<String>,

    /// Timestamp of the delta.
    pub timestamp: u64,

    /// Whether this is a merge of multiple transactions.
    pub is_merge: bool,

    /// Parent transactions if this is a merge.
    pub merge_parents: Vec<u64>,
}

impl TransactionDelta {
    /// Create a new delta from a transaction workspace.
    pub fn from_workspace(workspace: &TransactionWorkspace) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            tx_id: workspace.id(),
            parent_tx_id: None,
            delta_embedding: Vec::new(), // Will be computed by codebook
            affected_keys: workspace.affected_keys(),
            timestamp,
            is_merge: false,
            merge_parents: Vec::new(),
        }
    }

    /// Create an empty delta.
    pub fn empty() -> Self {
        Self {
            tx_id: 0,
            parent_tx_id: None,
            delta_embedding: Vec::new(),
            affected_keys: HashSet::new(),
            timestamp: 0,
            is_merge: false,
            merge_parents: Vec::new(),
        }
    }

    /// Set the delta embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.delta_embedding = embedding;
        self
    }

    /// Mark this as a merge of parent transactions.
    pub fn as_merge(mut self, parent_ids: Vec<u64>) -> Self {
        self.is_merge = true;
        self.merge_parents = parent_ids;
        self
    }

    /// Check if this delta has any affected keys overlapping with another.
    pub fn overlaps_with(&self, other: &TransactionDelta) -> bool {
        self.affected_keys
            .intersection(&other.affected_keys)
            .next()
            .is_some()
    }

    /// Get overlapping keys with another delta.
    pub fn overlapping_keys(&self, other: &TransactionDelta) -> HashSet<String> {
        self.affected_keys
            .intersection(&other.affected_keys)
            .cloned()
            .collect()
    }
}

/// Transaction manager for coordinating workspace lifecycle.
pub struct TransactionManager {
    /// Active transactions by ID.
    active: RwLock<std::collections::HashMap<u64, Arc<TransactionWorkspace>>>,
}

impl TransactionManager {
    /// Create a new transaction manager.
    pub fn new() -> Self {
        Self {
            active: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Begin a new transaction.
    pub fn begin(&self, store: &TensorStore) -> Result<Arc<TransactionWorkspace>> {
        let workspace = Arc::new(TransactionWorkspace::begin(store)?);
        self.active
            .write()
            .insert(workspace.id(), workspace.clone());
        Ok(workspace)
    }

    /// Get an active transaction by ID.
    pub fn get(&self, tx_id: u64) -> Option<Arc<TransactionWorkspace>> {
        self.active.read().get(&tx_id).cloned()
    }

    /// Remove a transaction from active tracking.
    pub fn remove(&self, tx_id: u64) -> Option<Arc<TransactionWorkspace>> {
        self.active.write().remove(&tx_id)
    }

    /// Get count of active transactions.
    pub fn active_count(&self) -> usize {
        self.active.read().len()
    }

    /// Get all active transaction IDs.
    pub fn active_ids(&self) -> Vec<u64> {
        self.active.read().keys().copied().collect()
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_lifecycle() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        assert!(workspace.is_active());
        assert_eq!(workspace.state(), TransactionState::Active);
        assert_eq!(workspace.operation_count(), 0);
    }

    #[test]
    fn test_add_operations() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        workspace
            .add_operation(ChainTransaction::Put {
                key: "key1".to_string(),
                data: vec![1, 2, 3],
            })
            .unwrap();

        workspace
            .add_operation(ChainTransaction::Put {
                key: "key2".to_string(),
                data: vec![4, 5, 6],
            })
            .unwrap();

        assert_eq!(workspace.operation_count(), 2);
        assert!(workspace.affected_keys().contains("key1"));
        assert!(workspace.affected_keys().contains("key2"));
    }

    #[test]
    fn test_transaction_delta_overlap() {
        let delta1 = TransactionDelta {
            tx_id: 1,
            parent_tx_id: None,
            delta_embedding: vec![],
            affected_keys: ["key1", "key2"].iter().map(|s| s.to_string()).collect(),
            timestamp: 0,
            is_merge: false,
            merge_parents: vec![],
        };

        let delta2 = TransactionDelta {
            tx_id: 2,
            parent_tx_id: None,
            delta_embedding: vec![],
            affected_keys: ["key2", "key3"].iter().map(|s| s.to_string()).collect(),
            timestamp: 0,
            is_merge: false,
            merge_parents: vec![],
        };

        let delta3 = TransactionDelta {
            tx_id: 3,
            parent_tx_id: None,
            delta_embedding: vec![],
            affected_keys: ["key4", "key5"].iter().map(|s| s.to_string()).collect(),
            timestamp: 0,
            is_merge: false,
            merge_parents: vec![],
        };

        assert!(delta1.overlaps_with(&delta2));
        assert!(!delta1.overlaps_with(&delta3));
        assert!(!delta2.overlaps_with(&delta3));

        let overlap = delta1.overlapping_keys(&delta2);
        assert_eq!(overlap.len(), 1);
        assert!(overlap.contains("key2"));
    }

    #[test]
    fn test_transaction_manager() {
        let store = TensorStore::new();
        let manager = TransactionManager::new();

        let ws1 = manager.begin(&store).unwrap();
        let ws2 = manager.begin(&store).unwrap();

        assert_eq!(manager.active_count(), 2);
        assert!(manager.get(ws1.id()).is_some());
        assert!(manager.get(ws2.id()).is_some());

        manager.remove(ws1.id());
        assert_eq!(manager.active_count(), 1);
        assert!(manager.get(ws1.id()).is_none());
    }
}
