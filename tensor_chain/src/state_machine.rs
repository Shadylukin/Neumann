//! State machine for applying Raft log entries to TensorChain.
//!
//! Bridges the gap between Raft consensus (log commitment) and
//! TensorChain storage (block persistence). Uses block embeddings
//! for fast-path validation when possible.

use std::sync::Arc;

use tensor_store::SparseVector;

use crate::block::Block;
use crate::chain::Chain;
use crate::error::Result;
use crate::network::LogEntry;
use crate::raft::RaftNode;

/// Tensor-native state machine that applies Raft log entries to TensorChain.
///
/// Uses block embeddings for fast-path validation, skipping heavy
/// validation when blocks are similar to recently applied ones.
pub struct TensorStateMachine {
    /// The chain storage layer.
    chain: Arc<Chain>,
    /// The Raft consensus node.
    raft: Arc<RaftNode>,
    /// Embedding similarity threshold for fast-path (skip full validation).
    fast_path_threshold: f32,
    /// Recent block embeddings for similarity comparison.
    recent_embeddings: parking_lot::RwLock<Vec<SparseVector>>,
    /// Maximum number of recent embeddings to keep.
    max_recent: usize,
}

impl TensorStateMachine {
    /// Create a new state machine.
    pub fn new(chain: Arc<Chain>, raft: Arc<RaftNode>) -> Self {
        Self {
            chain,
            raft,
            fast_path_threshold: 0.95,
            recent_embeddings: parking_lot::RwLock::new(Vec::new()),
            max_recent: 10,
        }
    }

    /// Create with custom fast-path threshold.
    pub fn with_threshold(chain: Arc<Chain>, raft: Arc<RaftNode>, threshold: f32) -> Self {
        Self {
            chain,
            raft,
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

    /// Apply a single log entry to the chain.
    fn apply_entry(&self, entry: &LogEntry) -> Result<()> {
        let block = &entry.block;

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

    /// Get access to the underlying chain.
    pub fn chain(&self) -> &Chain {
        &self.chain
    }

    /// Get access to the Raft node.
    pub fn raft(&self) -> &RaftNode {
        &self.raft
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::{BlockHeader, Transaction};
    use crate::network::MemoryTransport;
    use crate::raft::RaftConfig;
    use graph_engine::GraphEngine;
    use tensor_store::TensorStore;

    fn create_test_chain() -> Arc<Chain> {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::with_store(store));
        Arc::new(Chain::new(graph, "test_node".to_string()))
    }

    fn create_test_raft() -> Arc<RaftNode> {
        let transport = Arc::new(MemoryTransport::new("test_node".to_string()));
        Arc::new(RaftNode::new(
            "test_node".to_string(),
            vec![],
            transport,
            RaftConfig::default(),
        ))
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
        let chain = create_test_chain();
        let raft = create_test_raft();

        let sm = TensorStateMachine::new(chain, raft);

        assert!((sm.fast_path_threshold() - 0.95).abs() < 0.001);
        assert_eq!(sm.recent_embedding_count(), 0);
    }

    #[test]
    fn test_state_machine_with_threshold() {
        let chain = create_test_chain();
        let raft = create_test_raft();

        let sm = TensorStateMachine::with_threshold(chain, raft, 0.8);

        assert!((sm.fast_path_threshold() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_threshold_clamping() {
        let chain = create_test_chain();
        let raft = create_test_raft();

        // Test clamping above 1.0
        let sm = TensorStateMachine::with_threshold(chain.clone(), raft.clone(), 1.5);
        assert!((sm.fast_path_threshold() - 1.0).abs() < 0.001);

        // Test clamping below 0.0
        let sm = TensorStateMachine::with_threshold(chain, raft, -0.5);
        assert!(sm.fast_path_threshold().abs() < 0.001);
    }

    #[test]
    fn test_can_fast_path_no_embedding() {
        let chain = create_test_chain();
        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain, raft);

        // Block without embedding
        let block = create_block_with_embedding(1, &[]);

        assert!(!sm.can_fast_path(&block));
    }

    #[test]
    fn test_can_fast_path_no_history() {
        let chain = create_test_chain();
        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain, raft);

        // Block with embedding but no history
        let block = create_block_with_embedding(1, &[1.0, 0.0, 0.0, 0.0]);

        assert!(!sm.can_fast_path(&block));
    }

    #[test]
    fn test_can_fast_path_similar_embedding() {
        let chain = create_test_chain();
        let raft = create_test_raft();
        let sm = TensorStateMachine::with_threshold(chain, raft, 0.9);

        // Add a recent embedding
        let block1 = create_block_with_embedding(1, &[1.0, 0.0, 0.0, 0.0]);
        sm.track_embedding(&block1);

        // Test with very similar embedding (should pass fast-path)
        let block2 = create_block_with_embedding(2, &[0.99, 0.01, 0.0, 0.0]);
        assert!(sm.can_fast_path(&block2));
    }

    #[test]
    fn test_can_fast_path_dissimilar_embedding() {
        let chain = create_test_chain();
        let raft = create_test_raft();
        let sm = TensorStateMachine::with_threshold(chain, raft, 0.9);

        // Add a recent embedding
        let block1 = create_block_with_embedding(1, &[1.0, 0.0, 0.0, 0.0]);
        sm.track_embedding(&block1);

        // Test with orthogonal embedding (should fail fast-path)
        let block2 = create_block_with_embedding(2, &[0.0, 1.0, 0.0, 0.0]);
        assert!(!sm.can_fast_path(&block2));
    }

    #[test]
    fn test_track_embedding_max_recent() {
        let chain = create_test_chain();
        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain, raft);

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
        let chain = create_test_chain();
        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain, raft);

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
        let chain = create_test_chain();
        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain, raft);

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
        let chain = create_test_chain();
        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain, raft);

        // Add history
        let block = create_block_with_embedding(1, &[1.0, 0.0, 0.0, 0.0]);
        sm.track_embedding(&block);

        // Empty embedding = 0 similarity
        let empty = SparseVector::new(0);
        assert!((sm.recent_embedding_similarity(&empty) - 0.0).abs() < 0.001);
    }

    fn create_valid_block(chain: &Chain, embedding: &[f32]) -> Block {
        use crate::block::Transaction;
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
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain.clone(), raft);

        // Create a log entry using chain's block builder for correct prev_hash
        let block = create_valid_block(&chain, &[1.0, 0.0, 0.0, 0.0]);
        let entry = LogEntry {
            term: 1,
            index: 1,
            block,
        };

        // Apply entry
        sm.apply_entry(&entry).unwrap();

        // Check chain state
        assert_eq!(chain.height(), 1);
        assert_eq!(sm.recent_embedding_count(), 1);
    }

    #[test]
    fn test_apply_entry_no_embedding() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain.clone(), raft);

        // Create a log entry without embedding
        let block = create_valid_block(&chain, &[]);
        let entry = LogEntry {
            term: 1,
            index: 1,
            block,
        };

        // Apply entry (should use full validation path)
        sm.apply_entry(&entry).unwrap();

        // Check chain state
        assert_eq!(chain.height(), 1);
        // No embedding tracked
        assert_eq!(sm.recent_embedding_count(), 0);
    }

    #[test]
    fn test_apply_multiple_entries() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let raft = create_test_raft();
        let sm = TensorStateMachine::with_threshold(chain.clone(), raft, 0.9);

        // Apply first entry (full validation) using chain's block builder
        let block1 = create_valid_block(&chain, &[1.0, 0.0, 0.0, 0.0]);
        let entry1 = LogEntry {
            term: 1,
            index: 1,
            block: block1,
        };
        sm.apply_entry(&entry1).unwrap();

        // Apply second entry (similar embedding - should use fast-path)
        let block2 = create_valid_block(&chain, &[0.99, 0.01, 0.0, 0.0]);
        let entry2 = LogEntry {
            term: 1,
            index: 2,
            block: block2,
        };
        sm.apply_entry(&entry2).unwrap();

        // Check chain state
        assert_eq!(chain.height(), 2);
        assert_eq!(sm.recent_embedding_count(), 2);
    }

    #[test]
    fn test_accessors() {
        let chain = create_test_chain();
        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain.clone(), raft.clone());

        // Access chain - verify we can get height
        assert_eq!(sm.chain().height(), 0);

        // Access raft
        assert_eq!(sm.raft().node_id(), "test_node");
    }

    #[test]
    fn test_apply_committed_empty() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let raft = create_test_raft();
        let sm = TensorStateMachine::new(chain, raft);

        // No committed entries
        let applied = sm.apply_committed().unwrap();
        assert_eq!(applied, 0);
    }
}
