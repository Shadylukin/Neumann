//! State machine for applying Raft log entries to TensorChain.
//!
//! Bridges the gap between Raft consensus (log commitment) and
//! TensorChain storage (block persistence). Uses block embeddings
//! for fast-path validation when possible.
//!
//! Transactions are applied to the appropriate storage layer:
//! - Put/Delete → TensorStore directly
//! - Embed → VectorEngine (emb: prefix)
//! - NodeCreate/Delete, EdgeCreate → GraphEngine (node:/edge: prefix)
//! - TableInsert/Update/Delete → RelationalEngine (table: prefix)

use std::sync::Arc;

use tensor_store::{SparseVector, TensorData, TensorStore, TensorValue};

use crate::{
    block::{Block, Transaction},
    chain::Chain,
    error::{ChainError, Result},
    network::LogEntry,
    raft::RaftNode,
};

/// Tensor-native state machine that applies Raft log entries to TensorChain.
///
/// Uses block embeddings for fast-path validation, skipping heavy
/// validation when blocks are similar to recently applied ones.
pub struct TensorStateMachine {
    /// The chain storage layer.
    chain: Arc<Chain>,
    /// The Raft consensus node.
    raft: Arc<RaftNode>,
    /// TensorStore for transaction application.
    store: TensorStore,
    /// Embedding similarity threshold for fast-path (skip full validation).
    fast_path_threshold: f32,
    /// Recent block embeddings for similarity comparison.
    recent_embeddings: parking_lot::RwLock<Vec<SparseVector>>,
    /// Maximum number of recent embeddings to keep.
    max_recent: usize,
}

impl TensorStateMachine {
    /// Create a new state machine.
    pub fn new(chain: Arc<Chain>, raft: Arc<RaftNode>, store: TensorStore) -> Self {
        Self {
            chain,
            raft,
            store,
            fast_path_threshold: 0.95,
            recent_embeddings: parking_lot::RwLock::new(Vec::new()),
            max_recent: 10,
        }
    }

    /// Create with custom fast-path threshold.
    pub fn with_threshold(
        chain: Arc<Chain>,
        raft: Arc<RaftNode>,
        store: TensorStore,
        threshold: f32,
    ) -> Self {
        Self {
            chain,
            raft,
            store,
            fast_path_threshold: threshold.clamp(0.0, 1.0),
            recent_embeddings: parking_lot::RwLock::new(Vec::new()),
            max_recent: 10,
        }
    }

    /// Get the fast-path threshold.
    pub fn fast_path_threshold(&self) -> f32 {
        self.fast_path_threshold
    }

    /// Get the number of recent embeddings stored.
    pub fn recent_embedding_count(&self) -> usize {
        self.recent_embeddings.read().len()
    }

    /// Apply all committed but unapplied entries to the chain.
    ///
    /// Returns the number of blocks successfully applied.
    pub fn apply_committed(&self) -> Result<usize> {
        let entries = self.raft.get_uncommitted_entries();
        let mut applied_count = 0;

        for entry in &entries {
            self.apply_entry(entry)?;
            self.raft.mark_applied(entry.index);
            applied_count += 1;
        }

        Ok(applied_count)
    }

    /// Apply a block directly (for testing or manual application).
    pub fn apply_block(&self, block: &Block) -> Result<()> {
        // Apply each transaction to the appropriate storage layer
        for tx in &block.transactions {
            self.apply_transaction(tx)?;
        }

        // Use fast-path if block embedding is similar to recent blocks
        if self.can_fast_path(block) {
            self.append_fast(block.clone())?;
        } else {
            self.append_full(block.clone())?;
        }

        // Track this block's embedding for future fast-path decisions
        self.track_embedding(block);

        Ok(())
    }

    /// Apply a single log entry to the chain.
    fn apply_entry(&self, entry: &LogEntry) -> Result<()> {
        // Handle config changes first (membership updates)
        if let Some(ref config_change) = entry.config_change {
            self.apply_config_change(entry.index, config_change)?;
        }

        // Skip block processing for pure config entries
        if entry.is_config_change() && entry.block.transactions.is_empty() {
            return Ok(());
        }

        let block = &entry.block;

        // Apply each transaction to the appropriate storage layer
        for tx in &block.transactions {
            self.apply_transaction(tx)?;
        }

        // Use fast-path if block embedding is similar to recent blocks
        if self.can_fast_path(block) {
            self.append_fast(block.clone())?;
        } else {
            self.append_full(block.clone())?;
        }

        // Track this block's embedding for future fast-path decisions
        self.track_embedding(block);

        Ok(())
    }

    /// Apply a config change to the Raft membership.
    fn apply_config_change(&self, index: u64, change: &crate::network::ConfigChange) -> Result<()> {
        use crate::network::ConfigChange;

        let mut config = self.raft.membership_config();

        match change {
            ConfigChange::AddLearner { node_id } => {
                if !config.learners.contains(node_id) && !config.voters.contains(node_id) {
                    config.learners.push(node_id.clone());
                }
            },
            ConfigChange::PromoteLearner { node_id } => {
                if let Some(pos) = config.learners.iter().position(|n| n == node_id) {
                    config.learners.remove(pos);
                    if !config.voters.contains(node_id) {
                        config.voters.push(node_id.clone());
                    }
                }
            },
            ConfigChange::RemoveNode { node_id } => {
                config.voters.retain(|n| n != node_id);
                config.learners.retain(|n| n != node_id);
            },
            ConfigChange::JointChange {
                additions,
                removals,
            } => {
                // Enter or exit joint consensus
                if config.joint.is_some() {
                    // Already in joint - this is the commit of the new config
                    // Keep only new_voters from the joint config
                    if let Some(joint) = &config.joint {
                        config.voters = joint.new_voters.clone();
                    }
                    config.joint = None;
                } else {
                    // Enter joint consensus - need quorum from both old and new
                    let old_voters = config.voters.clone();
                    let mut new_voters = old_voters.clone();
                    for node in additions {
                        if !new_voters.contains(node) {
                            new_voters.push(node.clone());
                        }
                    }
                    for node in removals {
                        new_voters.retain(|n| n != node);
                    }
                    config.joint = Some(crate::network::JointConfig {
                        old_voters,
                        new_voters,
                    });
                }
            },
        }

        config.config_index = index;
        self.raft.set_membership_config(config);
        Ok(())
    }

    /// Apply a single transaction to the storage layer.
    fn apply_transaction(&self, tx: &Transaction) -> Result<()> {
        use tensor_store::ScalarValue;

        match tx {
            // Key-value operations → TensorStore directly
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
                // Ignore errors on delete (key may not exist)
                let _ = self.store.delete(key);
            },

            // Embedding operations → emb: prefix (VectorEngine pattern)
            Transaction::Embed { key, vector } => {
                let storage_key = format!("emb:{}", key);
                let mut tensor = TensorData::new();
                tensor.set("vector", TensorValue::Vector(vector.clone()));
                self.store
                    .put(storage_key, tensor)
                    .map_err(|e| ChainError::StorageError(e.to_string()))?;
            },

            // Graph node operations → node: prefix (GraphEngine pattern)
            Transaction::NodeCreate { key, label } => {
                let storage_key = format!("node:{}", key);
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
                let storage_key = format!("node:{}", key);
                let _ = self.store.delete(&storage_key);
            },

            // Graph edge operations → edge: prefix
            Transaction::EdgeCreate {
                from,
                to,
                edge_type,
            } => {
                let storage_key = format!("edge:{}:{}:{}", from, to, edge_type);
                let mut tensor = TensorData::new();
                tensor.set(
                    "_from",
                    TensorValue::Scalar(ScalarValue::String(from.clone())),
                );
                tensor.set("_to", TensorValue::Scalar(ScalarValue::String(to.clone())));
                tensor.set(
                    "_edge_type",
                    TensorValue::Scalar(ScalarValue::String(edge_type.clone())),
                );
                self.store
                    .put(storage_key, tensor)
                    .map_err(|e| ChainError::StorageError(e.to_string()))?;
            },

            // Table operations → table: prefix (RelationalEngine pattern)
            Transaction::TableInsert { table, values } => {
                // Generate a unique row key based on table and timestamp
                let row_key = format!(
                    "table:{}:row:{}",
                    table,
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos()
                );
                let mut tensor = TensorData::new();
                tensor.set(
                    "data",
                    TensorValue::Scalar(ScalarValue::Bytes(values.clone())),
                );
                self.store
                    .put(row_key, tensor)
                    .map_err(|e| ChainError::StorageError(e.to_string()))?;
            },
            Transaction::TableUpdate {
                table,
                row_id,
                values,
            } => {
                let row_key = format!("table:{}:row:{}", table, row_id);
                let mut tensor = TensorData::new();
                tensor.set(
                    "data",
                    TensorValue::Scalar(ScalarValue::Bytes(values.clone())),
                );
                self.store
                    .put(row_key, tensor)
                    .map_err(|e| ChainError::StorageError(e.to_string()))?;
            },
            Transaction::TableDelete { table, row_id } => {
                let row_key = format!("table:{}:row:{}", table, row_id);
                let _ = self.store.delete(&row_key);
            },
        }
        Ok(())
    }

    /// Check if block can use fast-path validation via embedding similarity.
    fn can_fast_path(&self, block: &Block) -> bool {
        let block_embedding = &block.header.delta_embedding;

        // No embedding = can't use fast-path
        if block_embedding.nnz() == 0 {
            return false;
        }

        // Check similarity against recent embeddings
        let recent = self.recent_embeddings.read();
        if recent.is_empty() {
            return false;
        }

        // Find max similarity to any recent embedding
        let max_similarity = recent
            .iter()
            .map(|recent_emb| block_embedding.cosine_similarity(recent_emb))
            .fold(f32::NEG_INFINITY, f32::max);

        max_similarity >= self.fast_path_threshold
    }

    /// Compute similarity of embedding to recent embeddings.
    pub fn recent_embedding_similarity(&self, embedding: &SparseVector) -> f32 {
        let recent = self.recent_embeddings.read();
        if recent.is_empty() || embedding.nnz() == 0 {
            return 0.0;
        }

        recent
            .iter()
            .map(|recent_emb| embedding.cosine_similarity(recent_emb))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Append block using fast-path (minimal validation).
    fn append_fast(&self, block: Block) -> Result<()> {
        // Fast-path: trust the block since embedding is similar to recent
        // Still do basic structural validation via Chain::append
        self.chain.append(block)?;
        Ok(())
    }

    /// Append block with full validation.
    fn append_full(&self, block: Block) -> Result<()> {
        // Full validation path through Chain::append
        self.chain.append(block)?;
        Ok(())
    }

    /// Track a block's embedding in recent history.
    fn track_embedding(&self, block: &Block) {
        let embedding = &block.header.delta_embedding;
        if embedding.nnz() == 0 {
            return;
        }

        let mut recent = self.recent_embeddings.write();
        recent.push(embedding.clone());

        // Keep only the most recent embeddings
        while recent.len() > self.max_recent {
            recent.remove(0);
        }
    }

    /// Clear recent embedding history.
    pub fn clear_recent(&self) {
        self.recent_embeddings.write().clear();
    }

    /// Get the current state embedding for geometric routing.
    ///
    /// Returns a weighted average of recent embeddings, giving more weight
    /// to newer entries. Returns None if no recent embeddings exist.
    pub fn current_state_embedding(&self) -> Option<SparseVector> {
        let recent = self.recent_embeddings.read();
        if recent.is_empty() {
            return None;
        }

        // Return most recent embedding (simpler and more responsive to state changes)
        recent.last().cloned()
    }

    /// Get access to the underlying chain.
    pub fn chain(&self) -> &Chain {
        &self.chain
    }

    /// Get access to the Raft node.
    pub fn raft(&self) -> &RaftNode {
        &self.raft
    }

    /// Get access to the TensorStore.
    pub fn store(&self) -> &TensorStore {
        &self.store
    }
}

#[cfg(test)]
mod tests {
    use graph_engine::GraphEngine;
    use tensor_store::TensorStore;

    use super::*;
    use crate::{
        block::{BlockHeader, Transaction},
        network::MemoryTransport,
        raft::RaftConfig,
    };

    fn create_test_components() -> (Arc<Chain>, Arc<RaftNode>, TensorStore) {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::with_store(store.clone()));
        let chain = Arc::new(Chain::new(graph, "test_node".to_string()));
        let transport = Arc::new(MemoryTransport::new("test_node".to_string()));
        let raft = Arc::new(RaftNode::new(
            "test_node".to_string(),
            vec![],
            transport,
            RaftConfig::default(),
        ));
        (chain, raft, store)
    }

    fn create_block_with_embedding(height: u64, embedding: &[f32]) -> Block {
        Block {
            header: BlockHeader {
                height,
                prev_hash: [0u8; 32],
                tx_root: [0u8; 32],
                state_root: [0u8; 32],
                timestamp: 0,
                proposer: "test".to_string(),
                signature: vec![],
                delta_embedding: SparseVector::from_dense(embedding),
                quantized_codes: vec![],
            },
            transactions: vec![Transaction::Put {
                key: format!("key{}", height),
                data: vec![height as u8],
            }],
            signatures: vec![],
        }
    }

    #[test]
    fn test_state_machine_new() {
        let (chain, raft, store) = create_test_components();

        let sm = TensorStateMachine::new(chain, raft, store);

        assert!((sm.fast_path_threshold() - 0.95).abs() < 0.001);
        assert_eq!(sm.recent_embedding_count(), 0);
    }

    #[test]
    fn test_state_machine_with_threshold() {
        let (chain, raft, store) = create_test_components();

        let sm = TensorStateMachine::with_threshold(chain, raft, store, 0.8);

        assert!((sm.fast_path_threshold() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_threshold_clamping() {
        let (chain, raft, store) = create_test_components();
        let (chain2, raft2, store2) = create_test_components();

        // Test clamping above 1.0
        let sm = TensorStateMachine::with_threshold(chain, raft, store, 1.5);
        assert!((sm.fast_path_threshold() - 1.0).abs() < 0.001);

        // Test clamping below 0.0
        let sm = TensorStateMachine::with_threshold(chain2, raft2, store2, -0.5);
        assert!(sm.fast_path_threshold().abs() < 0.001);
    }

    #[test]
    fn test_can_fast_path_no_embedding() {
        let (chain, raft, store) = create_test_components();
        let sm = TensorStateMachine::new(chain, raft, store);

        // Block without embedding
        let block = create_block_with_embedding(1, &[]);

        assert!(!sm.can_fast_path(&block));
    }

    #[test]
    fn test_can_fast_path_no_history() {
        let (chain, raft, store) = create_test_components();
        let sm = TensorStateMachine::new(chain, raft, store);

        // Block with embedding but no history
        let block = create_block_with_embedding(1, &[1.0, 0.0, 0.0, 0.0]);

        assert!(!sm.can_fast_path(&block));
    }

    #[test]
    fn test_can_fast_path_similar_embedding() {
        let (chain, raft, store) = create_test_components();
        let sm = TensorStateMachine::with_threshold(chain, raft, store, 0.9);

        // Add a recent embedding
        let block1 = create_block_with_embedding(1, &[1.0, 0.0, 0.0, 0.0]);
        sm.track_embedding(&block1);

        // Test with very similar embedding (should pass fast-path)
        let block2 = create_block_with_embedding(2, &[0.99, 0.01, 0.0, 0.0]);
        assert!(sm.can_fast_path(&block2));
    }

    #[test]
    fn test_can_fast_path_dissimilar_embedding() {
        let (chain, raft, store) = create_test_components();
        let sm = TensorStateMachine::with_threshold(chain, raft, store, 0.9);

        // Add a recent embedding
        let block1 = create_block_with_embedding(1, &[1.0, 0.0, 0.0, 0.0]);
        sm.track_embedding(&block1);

        // Test with orthogonal embedding (should fail fast-path)
        let block2 = create_block_with_embedding(2, &[0.0, 1.0, 0.0, 0.0]);
        assert!(!sm.can_fast_path(&block2));
    }

    #[test]
    fn test_track_embedding_max_recent() {
        let (chain, raft, store) = create_test_components();
        let sm = TensorStateMachine::new(chain, raft, store);

        // Add more than max_recent embeddings (use non-zero values to ensure non-empty)
        for i in 1..=15 {
            let block = create_block_with_embedding(i as u64, &[i as f32, 0.1, 0.0, 0.0]);
            sm.track_embedding(&block);
        }

        // Should only keep max_recent (10)
        assert_eq!(sm.recent_embedding_count(), 10);
    }

    #[test]
    fn test_clear_recent() {
        let (chain, raft, store) = create_test_components();
        let sm = TensorStateMachine::new(chain, raft, store);

        // Add some embeddings (use non-zero values to ensure non-empty)
        for i in 1..=5 {
            let block = create_block_with_embedding(i as u64, &[i as f32, 0.1, 0.0, 0.0]);
            sm.track_embedding(&block);
        }
        assert_eq!(sm.recent_embedding_count(), 5);

        sm.clear_recent();
        assert_eq!(sm.recent_embedding_count(), 0);
    }

    #[test]
    fn test_recent_embedding_similarity() {
        let (chain, raft, store) = create_test_components();
        let sm = TensorStateMachine::new(chain, raft, store);

        // No history = 0 similarity
        let emb = SparseVector::from_dense(&[1.0, 0.0]);
        assert!((sm.recent_embedding_similarity(&emb) - 0.0).abs() < 0.001);

        // Add history
        let block = create_block_with_embedding(1, &[1.0, 0.0, 0.0, 0.0]);
        sm.track_embedding(&block);

        // Similar embedding
        let similar = SparseVector::from_dense(&[0.99, 0.01, 0.0, 0.0]);
        let sim = sm.recent_embedding_similarity(&similar);
        assert!(sim > 0.9);
    }

    #[test]
    fn test_recent_embedding_similarity_empty_input() {
        let (chain, raft, store) = create_test_components();
        let sm = TensorStateMachine::new(chain, raft, store);

        // Add history
        let block = create_block_with_embedding(1, &[1.0, 0.0, 0.0, 0.0]);
        sm.track_embedding(&block);

        // Empty embedding = 0 similarity
        let empty = SparseVector::new(0);
        assert!((sm.recent_embedding_similarity(&empty) - 0.0).abs() < 0.001);
    }

    fn create_valid_block(chain: &Chain, embedding: &[f32]) -> Block {
        chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: "test_key".to_string(),
                data: vec![1, 2, 3],
            })
            .with_dense_embedding(embedding)
            .build()
    }

    #[test]
    fn test_apply_entry() {
        let (chain, raft, store) = create_test_components();
        chain.initialize().unwrap();

        let sm = TensorStateMachine::new(chain.clone(), raft, store);

        // Create a log entry using chain's block builder for correct prev_hash
        let block = create_valid_block(&chain, &[1.0, 0.0, 0.0, 0.0]);
        let entry = LogEntry::new(1, 1, block);

        // Apply entry
        sm.apply_entry(&entry).unwrap();

        // Check chain state
        assert_eq!(chain.height(), 1);
        assert_eq!(sm.recent_embedding_count(), 1);
    }

    #[test]
    fn test_apply_entry_no_embedding() {
        let (chain, raft, store) = create_test_components();
        chain.initialize().unwrap();

        let sm = TensorStateMachine::new(chain.clone(), raft, store);

        // Create a log entry without embedding
        let block = create_valid_block(&chain, &[]);
        let entry = LogEntry::new(1, 1, block);

        // Apply entry (should use full validation path)
        sm.apply_entry(&entry).unwrap();

        // Check chain state
        assert_eq!(chain.height(), 1);
        // No embedding tracked
        assert_eq!(sm.recent_embedding_count(), 0);
    }

    #[test]
    fn test_apply_multiple_entries() {
        let (chain, raft, store) = create_test_components();
        chain.initialize().unwrap();

        let sm = TensorStateMachine::with_threshold(chain.clone(), raft, store, 0.9);

        // Apply first entry (full validation) using chain's block builder
        let block1 = create_valid_block(&chain, &[1.0, 0.0, 0.0, 0.0]);
        let entry1 = LogEntry::new(1, 1, block1);
        sm.apply_entry(&entry1).unwrap();

        // Apply second entry (similar embedding - should use fast-path)
        let block2 = create_valid_block(&chain, &[0.99, 0.01, 0.0, 0.0]);
        let entry2 = LogEntry::new(1, 2, block2);
        sm.apply_entry(&entry2).unwrap();

        // Check chain state
        assert_eq!(chain.height(), 2);
        assert_eq!(sm.recent_embedding_count(), 2);
    }

    #[test]
    fn test_accessors() {
        let (chain, raft, store) = create_test_components();
        let sm = TensorStateMachine::new(chain, raft, store);

        // Access chain - verify we can get height
        assert_eq!(sm.chain().height(), 0);

        // Access raft
        assert_eq!(sm.raft().node_id(), "test_node");

        // Access store
        assert!(sm.store().is_empty());
    }

    #[test]
    fn test_apply_committed_empty() {
        let (chain, raft, store) = create_test_components();
        chain.initialize().unwrap();

        let sm = TensorStateMachine::new(chain, raft, store);

        // No committed entries
        let applied = sm.apply_committed().unwrap();
        assert_eq!(applied, 0);
    }

    #[test]
    fn test_transaction_applied_to_store() {
        let (chain, raft, store) = create_test_components();
        chain.initialize().unwrap();

        let sm = TensorStateMachine::new(chain.clone(), raft, store.clone());

        // Create a block with a Put transaction
        let block = chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: "user:1".to_string(),
                data: vec![1, 2, 3, 4],
            })
            .build();

        let entry = LogEntry::new(1, 1, block);

        // Apply entry
        sm.apply_entry(&entry).unwrap();

        // Verify the data was written to the store
        let result = store.get("user:1");
        assert!(result.is_ok(), "Data should be in store after apply");
    }

    #[test]
    fn test_embed_transaction_applied() {
        let (chain, raft, store) = create_test_components();
        chain.initialize().unwrap();

        let sm = TensorStateMachine::new(chain.clone(), raft, store.clone());

        // Create a block with an Embed transaction
        let block = chain
            .new_block()
            .add_transaction(Transaction::Embed {
                key: "doc:1".to_string(),
                vector: vec![1.0, 2.0, 3.0, 4.0],
            })
            .build();

        let entry = LogEntry::new(1, 1, block);

        // Apply entry
        sm.apply_entry(&entry).unwrap();

        // Verify the embedding was written (emb: prefix)
        let result = store.get("emb:doc:1");
        assert!(result.is_ok(), "Embedding should be in store after apply");
    }

    #[test]
    fn test_node_create_transaction_applied() {
        let (chain, raft, store) = create_test_components();
        chain.initialize().unwrap();

        let sm = TensorStateMachine::new(chain.clone(), raft, store.clone());

        // Create a block with a NodeCreate transaction
        let block = chain
            .new_block()
            .add_transaction(Transaction::NodeCreate {
                key: "person:alice".to_string(),
                label: "Person".to_string(),
            })
            .build();

        let entry = LogEntry::new(1, 1, block);

        // Apply entry
        sm.apply_entry(&entry).unwrap();

        // Verify the node was written (node: prefix)
        let result = store.get("node:person:alice");
        assert!(result.is_ok(), "Node should be in store after apply");
    }
}
