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
use crate::consensus::DeltaVector;
use crate::embedding::EmbeddingState;
use crate::error::{ChainError, Result};

/// Default embedding dimension for state snapshots.
pub const DEFAULT_EMBEDDING_DIM: usize = 128;

/// Embedding state for a transaction workspace.
///
/// Wraps EmbeddingState to provide a mutable API for transaction lifecycle.
/// Uses type-safe state machine internally, eliminating Option ceremony.
#[derive(Debug, Clone, Default)]
pub struct WorkspaceEmbedding {
    state: EmbeddingState,
}

impl WorkspaceEmbedding {
    /// Create a new workspace embedding from the initial state.
    pub fn new(before: Vec<f32>) -> Self {
        Self {
            state: EmbeddingState::from_dense(&before),
        }
    }

    /// Create an empty embedding (for stores without embeddings).
    pub fn empty(dim: usize) -> Self {
        Self {
            state: EmbeddingState::empty(dim),
        }
    }

    /// Set the after-state embedding and compute delta.
    ///
    /// If dimensions mismatch or delta already computed, this is a no-op.
    pub fn set_after(&mut self, after: Vec<f32>) {
        let current = self.state.clone();
        if let Ok(computed) = current.compute_from_dense(&after) {
            self.state = computed;
        }
        // On error (dimension mismatch or already computed): keep original state
    }

    /// Get the delta embedding, or zeros if not available.
    pub fn delta_or_zero(&self) -> Vec<f32> {
        self.state.delta_or_zero()
    }

    /// Check if the delta has been computed.
    pub fn has_delta(&self) -> bool {
        self.state.is_computed()
    }

    /// Get the before-state embedding as a dense vector.
    pub fn before(&self) -> Vec<f32> {
        self.state.before().to_dense()
    }

    /// Get the after-state embedding as a dense vector, if computed.
    pub fn after(&self) -> Option<Vec<f32>> {
        self.state.after().map(|v| v.to_dense())
    }

    /// Get the delta embedding as a dense vector, if computed.
    pub fn delta(&self) -> Option<Vec<f32>> {
        self.state.delta().map(|v| v.to_dense())
    }

    /// Get the dimension of the embedding space.
    pub fn dimension(&self) -> usize {
        self.state.dimension()
    }

    /// Get the underlying EmbeddingState (for advanced usage).
    pub fn state(&self) -> &EmbeddingState {
        &self.state
    }

    /// Set the before-state embedding, resetting to Initial state.
    ///
    /// This is used when capturing the initial state after workspace creation.
    pub fn set_before(&mut self, before: Vec<f32>) {
        self.state = EmbeddingState::from_dense(&before);
    }
}

/// A candidate for merging with another transaction.
#[derive(Debug, Clone)]
pub struct MergeCandidate {
    /// The transaction workspace.
    pub workspace: Arc<TransactionWorkspace>,
    /// Delta vector for conflict detection.
    pub delta: DeltaVector,
    /// Cosine similarity with the reference transaction.
    pub similarity: f32,
}

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

    /// Embedding state for semantic conflict detection.
    embedding: RwLock<WorkspaceEmbedding>,
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
            embedding: RwLock::new(WorkspaceEmbedding::empty(DEFAULT_EMBEDDING_DIM)),
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

    /// Set the before-state embedding (captured at transaction begin).
    pub fn set_before_embedding(&self, embedding: Vec<f32>) {
        self.embedding.write().set_before(embedding);
    }

    /// Compute the delta embedding from the current state.
    /// Call this at commit time after all operations are recorded.
    pub fn compute_delta(&self, after_embedding: Vec<f32>) {
        self.embedding.write().set_after(after_embedding);
    }

    /// Get a copy of the workspace embedding.
    pub fn embedding(&self) -> WorkspaceEmbedding {
        self.embedding.read().clone()
    }

    /// Get the delta embedding, or zeros if not computed.
    pub fn delta_embedding(&self) -> Vec<f32> {
        self.embedding.read().delta_or_zero()
    }

    /// Check if the delta has been computed.
    pub fn has_delta(&self) -> bool {
        self.embedding.read().has_delta()
    }

    /// Convert the delta embedding to a DeltaVector for conflict detection.
    pub fn to_delta_vector(&self) -> DeltaVector {
        let emb = self.embedding.read();
        let affected = self.affected_keys.read().clone();
        DeltaVector::new(emb.delta_or_zero(), affected, self.id)
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

    /// Find transactions that can be merged with the given workspace.
    ///
    /// Returns candidates where cosine similarity is below the threshold,
    /// indicating orthogonal (non-conflicting) changes.
    pub fn find_merge_candidates(
        &self,
        workspace: &TransactionWorkspace,
        orthogonal_threshold: f32,
    ) -> Vec<MergeCandidate> {
        let target_delta = workspace.to_delta_vector();

        // Skip if target has zero magnitude (no meaningful change)
        if target_delta.magnitude() == 0.0 {
            return Vec::new();
        }

        let active = self.active.read();
        let mut candidates = Vec::new();

        for (id, other) in active.iter() {
            // Skip self
            if *id == workspace.id() {
                continue;
            }

            // Skip non-active transactions
            if !other.is_active() {
                continue;
            }

            let other_delta = other.to_delta_vector();

            // Skip zero-magnitude deltas
            if other_delta.magnitude() == 0.0 {
                continue;
            }

            let similarity = target_delta.cosine_similarity(&other_delta).abs();

            // Orthogonal if similarity is below threshold
            if similarity < orthogonal_threshold {
                candidates.push(MergeCandidate {
                    workspace: other.clone(),
                    delta: other_delta,
                    similarity,
                });
            }
        }

        // Sort by similarity (most orthogonal first)
        candidates.sort_by(|a, b| {
            a.similarity
                .partial_cmp(&b.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates
    }

    /// Get all active transactions.
    pub fn active_transactions(&self) -> Vec<Arc<TransactionWorkspace>> {
        self.active.read().values().cloned().collect()
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

    #[test]
    fn test_workspace_embedding_default() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        let embedding = workspace.embedding();
        assert_eq!(embedding.dimension(), DEFAULT_EMBEDDING_DIM);
        assert!(embedding.after().is_none());
        assert!(embedding.delta().is_none());
        assert!(!workspace.has_delta());
    }

    #[test]
    fn test_workspace_embedding_computation() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        // Set before embedding
        let before = vec![1.0, 0.0, 0.0, 0.0];
        workspace.set_before_embedding(before.clone());

        // Set after embedding and compute delta
        let after = vec![1.0, 1.0, 0.0, 0.0];
        workspace.compute_delta(after.clone());

        let embedding = workspace.embedding();
        assert_eq!(embedding.before(), before);
        assert_eq!(embedding.after(), Some(after));
        assert!(embedding.delta().is_some());

        let delta = embedding.delta().unwrap();
        assert_eq!(delta, vec![0.0, 1.0, 0.0, 0.0]);
        assert!(workspace.has_delta());
    }

    #[test]
    fn test_workspace_to_delta_vector() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        // Add some operations to set affected keys
        workspace
            .add_operation(ChainTransaction::Put {
                key: "test_key".to_string(),
                data: vec![1, 2, 3],
            })
            .unwrap();

        // Set embeddings
        let before = vec![1.0, 0.0, 0.0, 0.0];
        let after = vec![1.0, 0.5, 0.0, 0.0];
        workspace.set_before_embedding(before);
        workspace.compute_delta(after);

        let delta_vector = workspace.to_delta_vector();

        assert_eq!(delta_vector.tx_id, workspace.id());
        assert!(delta_vector.affected_keys.contains("test_key"));
        assert_eq!(delta_vector.dimension(), 4);
    }

    #[test]
    fn test_merge_candidate_finding() {
        let store = TensorStore::new();
        let manager = TransactionManager::new();

        // Create two workspaces with orthogonal delta embeddings
        let ws1 = manager.begin(&store).unwrap();
        ws1.add_operation(ChainTransaction::Put {
            key: "key1".to_string(),
            data: vec![1],
        })
        .unwrap();
        ws1.set_before_embedding(vec![0.0; 4]);
        ws1.compute_delta(vec![1.0, 0.0, 0.0, 0.0]); // Direction: X

        let ws2 = manager.begin(&store).unwrap();
        ws2.add_operation(ChainTransaction::Put {
            key: "key2".to_string(),
            data: vec![2],
        })
        .unwrap();
        ws2.set_before_embedding(vec![0.0; 4]);
        ws2.compute_delta(vec![0.0, 1.0, 0.0, 0.0]); // Direction: Y (orthogonal to X)

        // Find merge candidates for ws1
        let candidates = manager.find_merge_candidates(&ws1, 0.1);

        // ws2 should be a merge candidate (orthogonal)
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].workspace.id(), ws2.id());
        assert!(candidates[0].similarity < 0.1);
    }

    #[test]
    fn test_merge_candidate_excludes_parallel() {
        let store = TensorStore::new();
        let manager = TransactionManager::new();

        // Create two workspaces with parallel (similar) delta embeddings
        let ws1 = manager.begin(&store).unwrap();
        ws1.add_operation(ChainTransaction::Put {
            key: "key1".to_string(),
            data: vec![1],
        })
        .unwrap();
        ws1.set_before_embedding(vec![0.0; 4]);
        ws1.compute_delta(vec![1.0, 0.0, 0.0, 0.0]); // Direction: X

        let ws2 = manager.begin(&store).unwrap();
        ws2.add_operation(ChainTransaction::Put {
            key: "key2".to_string(),
            data: vec![2],
        })
        .unwrap();
        ws2.set_before_embedding(vec![0.0; 4]);
        ws2.compute_delta(vec![0.9, 0.1, 0.0, 0.0]); // Similar to X (not orthogonal)

        // Find merge candidates for ws1 with strict threshold
        let candidates = manager.find_merge_candidates(&ws1, 0.1);

        // ws2 should NOT be a merge candidate (too similar)
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_workspace_embedding_new() {
        let before = vec![1.0, 2.0, 3.0];
        let embedding = WorkspaceEmbedding::new(before.clone());

        assert_eq!(embedding.before(), before);
        assert!(embedding.after().is_none());
        assert!(embedding.delta().is_none());
    }

    #[test]
    fn test_workspace_embedding_empty() {
        let embedding = WorkspaceEmbedding::empty(8);

        assert_eq!(embedding.dimension(), 8);
        assert!(embedding.before().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_workspace_embedding_delta_or_zero() {
        let mut embedding = WorkspaceEmbedding::new(vec![1.0, 2.0, 3.0]);

        // Before computing delta, should return zeros
        let delta = embedding.delta_or_zero();
        assert_eq!(delta, vec![0.0, 0.0, 0.0]);

        // After setting delta, should return actual delta
        embedding.set_after(vec![2.0, 3.0, 4.0]);
        let delta = embedding.delta_or_zero();
        assert_eq!(delta, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_workspace_embedding_set_after_mismatched_dimensions() {
        let mut embedding = WorkspaceEmbedding::new(vec![1.0, 2.0, 3.0]);

        // Set after with different dimension - operation is a no-op
        embedding.set_after(vec![1.0, 2.0]); // Only 2 dimensions vs 3

        // With EmbeddingState, mismatched dimensions are silently ignored
        // The state remains in Initial, so after and delta are both None
        assert!(embedding.after().is_none());
        assert!(embedding.delta().is_none());
        assert!(!embedding.has_delta());
        // Before state is preserved
        assert_eq!(embedding.dimension(), 3);
    }

    #[test]
    fn test_workspace_embedding_has_delta() {
        let mut embedding = WorkspaceEmbedding::new(vec![1.0, 2.0]);
        assert!(!embedding.has_delta());

        embedding.set_after(vec![2.0, 3.0]);
        assert!(embedding.has_delta());
    }

    #[test]
    fn test_workspace_embedding_default_derive() {
        let embedding = WorkspaceEmbedding::default();
        assert_eq!(embedding.dimension(), 0);
        assert!(embedding.after().is_none());
        assert!(embedding.delta().is_none());
    }

    #[test]
    fn test_merge_candidate_debug() {
        let store = TensorStore::new();
        let workspace = Arc::new(TransactionWorkspace::begin(&store).unwrap());
        let delta = DeltaVector::new(vec![1.0], HashSet::new(), 1);

        let candidate = MergeCandidate {
            workspace,
            delta,
            similarity: 0.05,
        };

        let debug = format!("{:?}", candidate);
        assert!(debug.contains("MergeCandidate"));
        assert!(debug.contains("similarity"));
    }

    #[test]
    fn test_transaction_state_display() {
        let states = [
            TransactionState::Active,
            TransactionState::Committing,
            TransactionState::Committed,
            TransactionState::RolledBack,
            TransactionState::Failed,
        ];

        for state in states {
            let debug = format!("{:?}", state);
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn test_workspace_id() {
        let store = TensorStore::new();
        let ws1 = TransactionWorkspace::begin(&store).unwrap();
        let ws2 = TransactionWorkspace::begin(&store).unwrap();

        assert!(ws1.id() > 0);
        assert!(ws2.id() > ws1.id());
    }

    #[test]
    fn test_workspace_started_at() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Should be very close to now (within 1 second)
        assert!(workspace.started_at() <= now);
        assert!(workspace.started_at() > now - 1000);
    }

    #[test]
    fn test_add_operation_to_non_active() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        // Mark as committed
        workspace.mark_committed();

        // Try to add operation - should fail
        let result = workspace.add_operation(ChainTransaction::Put {
            key: "key".to_string(),
            data: vec![1],
        });

        assert!(result.is_err());
        match result.unwrap_err() {
            ChainError::TransactionFailed(msg) => {
                assert!(msg.contains("not active"));
            },
            _ => panic!("Expected TransactionFailed error"),
        }
    }

    #[test]
    fn test_mark_committing_success() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        assert!(workspace.mark_committing().is_ok());
        assert_eq!(workspace.state(), TransactionState::Committing);
    }

    #[test]
    fn test_mark_committing_fails_on_non_active() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        workspace.mark_committed();

        let result = workspace.mark_committing();
        assert!(result.is_err());
        match result.unwrap_err() {
            ChainError::TransactionFailed(msg) => {
                assert!(msg.contains("cannot commit"));
            },
            _ => panic!("Expected TransactionFailed error"),
        }
    }

    #[test]
    fn test_mark_committed() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        workspace.mark_committed();
        assert_eq!(workspace.state(), TransactionState::Committed);
        assert!(!workspace.is_active());
    }

    #[test]
    fn test_mark_failed() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        workspace.mark_failed();
        assert_eq!(workspace.state(), TransactionState::Failed);
        assert!(!workspace.is_active());
    }

    #[test]
    fn test_rollback() {
        let store = TensorStore::new();
        store
            .put("existing", tensor_store::TensorData::new())
            .unwrap();

        let workspace = TransactionWorkspace::begin(&store).unwrap();

        // Add a new key
        store
            .put("new_key", tensor_store::TensorData::new())
            .unwrap();

        // Rollback should restore original state
        workspace.rollback(&store).unwrap();

        assert_eq!(workspace.state(), TransactionState::RolledBack);
        // Check that existing key is still there and new_key is gone
        assert!(store.get("existing").is_ok());
        assert!(
            store
                .get("new_key")
                .unwrap_err()
                .to_string()
                .contains("not found")
                || store.get("new_key").is_err()
        );
    }

    #[test]
    fn test_rollback_fails_on_committed() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        workspace.mark_committed();

        let result = workspace.rollback(&store);
        assert!(result.is_err());
        match result.unwrap_err() {
            ChainError::TransactionFailed(msg) => {
                assert!(msg.contains("committed"));
            },
            _ => panic!("Expected TransactionFailed error"),
        }
    }

    #[test]
    fn test_checkpoint_bytes() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        let bytes = workspace.checkpoint_bytes();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_delta_embedding() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        // Without delta computed, should return zeros
        let delta = workspace.delta_embedding();
        assert!(delta.iter().all(|&x| x == 0.0));

        // After computing delta
        workspace.set_before_embedding(vec![0.0, 0.0, 0.0, 0.0]);
        workspace.compute_delta(vec![1.0, 2.0, 3.0, 4.0]);

        let delta = workspace.delta_embedding();
        assert_eq!(delta, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_transaction_delta_empty() {
        let delta = TransactionDelta::empty();

        assert_eq!(delta.tx_id, 0);
        assert!(delta.parent_tx_id.is_none());
        assert!(delta.delta_embedding.is_empty());
        assert!(delta.affected_keys.is_empty());
        assert_eq!(delta.timestamp, 0);
        assert!(!delta.is_merge);
        assert!(delta.merge_parents.is_empty());
    }

    #[test]
    fn test_transaction_delta_with_embedding() {
        let delta = TransactionDelta::empty().with_embedding(vec![1.0, 2.0, 3.0]);

        assert_eq!(delta.delta_embedding, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_transaction_delta_as_merge() {
        let delta = TransactionDelta::empty().as_merge(vec![1, 2, 3]);

        assert!(delta.is_merge);
        assert_eq!(delta.merge_parents, vec![1, 2, 3]);
    }

    #[test]
    fn test_transaction_delta_from_workspace() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();
        workspace
            .add_operation(ChainTransaction::Put {
                key: "key1".to_string(),
                data: vec![1],
            })
            .unwrap();

        let delta = TransactionDelta::from_workspace(&workspace);

        assert_eq!(delta.tx_id, workspace.id());
        assert!(delta.parent_tx_id.is_none());
        assert!(delta.affected_keys.contains("key1"));
        assert!(!delta.is_merge);
        assert!(delta.merge_parents.is_empty());
        assert!(delta.timestamp > 0);
    }

    #[test]
    fn test_transaction_delta_clone() {
        let delta = TransactionDelta {
            tx_id: 42,
            parent_tx_id: Some(41),
            delta_embedding: vec![1.0, 2.0],
            affected_keys: ["key1"].iter().map(|s| s.to_string()).collect(),
            timestamp: 12345,
            is_merge: true,
            merge_parents: vec![1, 2],
        };

        let cloned = delta.clone();
        assert_eq!(delta.tx_id, cloned.tx_id);
        assert_eq!(delta.parent_tx_id, cloned.parent_tx_id);
        assert_eq!(delta.delta_embedding, cloned.delta_embedding);
        assert_eq!(delta.affected_keys, cloned.affected_keys);
        assert_eq!(delta.is_merge, cloned.is_merge);
        assert_eq!(delta.merge_parents, cloned.merge_parents);
    }

    #[test]
    fn test_transaction_delta_debug() {
        let delta = TransactionDelta::empty();
        let debug = format!("{:?}", delta);
        assert!(debug.contains("TransactionDelta"));
        assert!(debug.contains("tx_id"));
    }

    #[test]
    fn test_transaction_manager_active_ids() {
        let store = TensorStore::new();
        let manager = TransactionManager::new();

        let ws1 = manager.begin(&store).unwrap();
        let ws2 = manager.begin(&store).unwrap();

        let ids = manager.active_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&ws1.id()));
        assert!(ids.contains(&ws2.id()));
    }

    #[test]
    fn test_transaction_manager_active_transactions() {
        let store = TensorStore::new();
        let manager = TransactionManager::new();

        let _ws1 = manager.begin(&store).unwrap();
        let _ws2 = manager.begin(&store).unwrap();

        let txs = manager.active_transactions();
        assert_eq!(txs.len(), 2);
    }

    #[test]
    fn test_transaction_manager_default() {
        let manager = TransactionManager::default();
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_find_merge_candidates_zero_magnitude() {
        let store = TensorStore::new();
        let manager = TransactionManager::new();

        // Create workspace with zero delta (no change)
        let ws = manager.begin(&store).unwrap();
        // Don't compute any delta - it will have zero magnitude

        let candidates = manager.find_merge_candidates(&ws, 0.1);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_find_merge_candidates_skips_inactive() {
        let store = TensorStore::new();
        let manager = TransactionManager::new();

        let ws1 = manager.begin(&store).unwrap();
        ws1.add_operation(ChainTransaction::Put {
            key: "key1".to_string(),
            data: vec![1],
        })
        .unwrap();
        ws1.set_before_embedding(vec![0.0; 4]);
        ws1.compute_delta(vec![1.0, 0.0, 0.0, 0.0]);

        let ws2 = manager.begin(&store).unwrap();
        ws2.add_operation(ChainTransaction::Put {
            key: "key2".to_string(),
            data: vec![2],
        })
        .unwrap();
        ws2.set_before_embedding(vec![0.0; 4]);
        ws2.compute_delta(vec![0.0, 1.0, 0.0, 0.0]);

        // Mark ws2 as committed (inactive)
        ws2.mark_committed();

        // Should not find ws2 as a candidate
        let candidates = manager.find_merge_candidates(&ws1, 0.1);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_find_merge_candidates_skips_zero_other_magnitude() {
        let store = TensorStore::new();
        let manager = TransactionManager::new();

        let ws1 = manager.begin(&store).unwrap();
        ws1.add_operation(ChainTransaction::Put {
            key: "key1".to_string(),
            data: vec![1],
        })
        .unwrap();
        ws1.set_before_embedding(vec![0.0; 4]);
        ws1.compute_delta(vec![1.0, 0.0, 0.0, 0.0]);

        let ws2 = manager.begin(&store).unwrap();
        // ws2 has zero delta (zero magnitude)
        ws2.set_before_embedding(vec![0.0; 4]);
        ws2.compute_delta(vec![0.0, 0.0, 0.0, 0.0]);

        // Should not find ws2 as a candidate
        let candidates = manager.find_merge_candidates(&ws1, 0.1);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_find_merge_candidates_sorting() {
        let store = TensorStore::new();
        let manager = TransactionManager::new();

        // Create reference workspace
        let ws1 = manager.begin(&store).unwrap();
        ws1.add_operation(ChainTransaction::Put {
            key: "key1".to_string(),
            data: vec![1],
        })
        .unwrap();
        ws1.set_before_embedding(vec![0.0; 4]);
        ws1.compute_delta(vec![1.0, 0.0, 0.0, 0.0]); // X direction

        // Create orthogonal workspace (Y direction)
        let ws2 = manager.begin(&store).unwrap();
        ws2.add_operation(ChainTransaction::Put {
            key: "key2".to_string(),
            data: vec![2],
        })
        .unwrap();
        ws2.set_before_embedding(vec![0.0; 4]);
        ws2.compute_delta(vec![0.0, 1.0, 0.0, 0.0]); // Perfectly orthogonal

        // Create almost orthogonal workspace
        let ws3 = manager.begin(&store).unwrap();
        ws3.add_operation(ChainTransaction::Put {
            key: "key3".to_string(),
            data: vec![3],
        })
        .unwrap();
        ws3.set_before_embedding(vec![0.0; 4]);
        ws3.compute_delta(vec![0.05, 0.99, 0.0, 0.0]); // Almost orthogonal

        let candidates = manager.find_merge_candidates(&ws1, 0.1);

        // Should have 2 candidates, sorted by similarity (most orthogonal first)
        assert_eq!(candidates.len(), 2);
        assert!(candidates[0].similarity <= candidates[1].similarity);
    }

    #[test]
    fn test_workspace_debug() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        let debug = format!("{:?}", workspace);
        assert!(debug.contains("TransactionWorkspace"));
    }

    #[test]
    fn test_workspace_operations_returns_clone() {
        let store = TensorStore::new();
        let workspace = TransactionWorkspace::begin(&store).unwrap();

        workspace
            .add_operation(ChainTransaction::Put {
                key: "key".to_string(),
                data: vec![1],
            })
            .unwrap();

        let ops1 = workspace.operations();
        let ops2 = workspace.operations();

        assert_eq!(ops1.len(), 1);
        assert_eq!(ops2.len(), 1);
        assert_eq!(ops1, ops2);
    }

    #[test]
    fn test_transaction_delta_no_overlap() {
        let delta1 = TransactionDelta {
            tx_id: 1,
            parent_tx_id: None,
            delta_embedding: vec![],
            affected_keys: ["key1"].iter().map(|s| s.to_string()).collect(),
            timestamp: 0,
            is_merge: false,
            merge_parents: vec![],
        };

        let delta2 = TransactionDelta {
            tx_id: 2,
            parent_tx_id: None,
            delta_embedding: vec![],
            affected_keys: ["key2"].iter().map(|s| s.to_string()).collect(),
            timestamp: 0,
            is_merge: false,
            merge_parents: vec![],
        };

        assert!(!delta1.overlaps_with(&delta2));
        assert!(delta1.overlapping_keys(&delta2).is_empty());
    }
}
