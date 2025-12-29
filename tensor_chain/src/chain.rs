//! Chain structure linking blocks via graph edges.
//!
//! Blocks are linked as a directed graph where each block points to its predecessor.
//! The chain provides:
//! - Append-only block storage
//! - Chain traversal and validation
//! - Height-indexed block lookup
//! - Similarity search over block embeddings

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use graph_engine::GraphEngine;
use parking_lot::RwLock;

use crate::block::{Block, BlockHash, BlockHeader, NodeId, Transaction};
use crate::error::{ChainError, Result};

/// Edge type for chain links.
const CHAIN_EDGE_TYPE: &str = "chain_next";

/// Prefix for block storage keys.
const BLOCK_PREFIX: &str = "chain:block:";

/// Key for chain metadata.
const CHAIN_META_KEY: &str = "chain:meta";

/// The tensor chain - an append-only linked structure of blocks.
pub struct Chain {
    /// Graph engine for block linking.
    graph: Arc<GraphEngine>,

    /// Current chain height (tip).
    height: AtomicU64,

    /// Hash of the current tip block.
    tip_hash: RwLock<BlockHash>,

    /// Node ID of this chain instance.
    node_id: NodeId,
}

impl Chain {
    /// Create a new chain with the given graph engine.
    pub fn new(graph: Arc<GraphEngine>, node_id: NodeId) -> Self {
        Self {
            graph,
            height: AtomicU64::new(0),
            tip_hash: RwLock::new([0u8; 32]),
            node_id,
        }
    }

    /// Initialize the chain, creating genesis block if needed.
    pub fn initialize(&self) -> Result<()> {
        // Check if chain already exists
        if let Some(height) = self.load_height() {
            self.height.store(height, Ordering::SeqCst);

            // Load tip hash
            if let Some(tip) = self.get_block_at(height)? {
                *self.tip_hash.write() = tip.hash();
            }

            return Ok(());
        }

        // Create genesis block
        let genesis = Block::genesis(self.node_id.clone());
        self.store_block(&genesis)?;

        *self.tip_hash.write() = genesis.hash();
        self.save_height(0)?;

        Ok(())
    }

    /// Get the current chain height.
    pub fn height(&self) -> u64 {
        self.height.load(Ordering::SeqCst)
    }

    /// Get the tip block hash.
    pub fn tip_hash(&self) -> BlockHash {
        *self.tip_hash.read()
    }

    /// Check if the chain is empty (only genesis).
    pub fn is_empty(&self) -> bool {
        self.height() == 0
    }

    /// Append a new block to the chain.
    pub fn append(&self, mut block: Block) -> Result<BlockHash> {
        let current_height = self.height();
        let expected_height = current_height + 1;

        // Validate block height
        if block.header.height != expected_height {
            return Err(ChainError::ValidationFailed(format!(
                "expected height {}, got {}",
                expected_height, block.header.height
            )));
        }

        // Validate prev_hash
        let tip_hash = self.tip_hash();
        if block.header.prev_hash != tip_hash {
            return Err(ChainError::InvalidHash {
                expected: hex::encode(tip_hash),
                actual: hex::encode(block.header.prev_hash),
            });
        }

        // Compute tx_root if not set
        if block.header.tx_root == [0u8; 32] && !block.transactions.is_empty() {
            block.header.tx_root = block.compute_tx_root();
        }

        // Store the block
        let block_hash = block.hash();
        self.store_block(&block)?;

        // Create chain link edge
        let prev_key = block_key(current_height);
        let new_key = block_key(expected_height);
        self.graph
            .add_entity_edge(&prev_key, &new_key, CHAIN_EDGE_TYPE)
            .map_err(|e| ChainError::GraphError(e.to_string()))?;

        // Update chain state
        self.height.store(expected_height, Ordering::SeqCst);
        *self.tip_hash.write() = block_hash;
        self.save_height(expected_height)?;

        Ok(block_hash)
    }

    /// Get a block at a specific height.
    pub fn get_block_at(&self, height: u64) -> Result<Option<Block>> {
        let key = block_key(height);

        let data = match self.graph.store().get(&key) {
            Ok(d) => d,
            Err(_) => return Ok(None),
        };

        // Deserialize block from stored bytes
        let bytes = match data.get("_block") {
            Some(tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Bytes(b))) => b,
            _ => return Err(ChainError::StorageError("missing block data".to_string())),
        };

        let block: Block = bincode::deserialize(bytes)
            .map_err(|e| ChainError::SerializationError(e.to_string()))?;

        Ok(Some(block))
    }

    /// Get the tip block.
    pub fn get_tip(&self) -> Result<Option<Block>> {
        self.get_block_at(self.height())
    }

    /// Get the genesis block.
    pub fn get_genesis(&self) -> Result<Option<Block>> {
        self.get_block_at(0)
    }

    /// Verify the entire chain integrity.
    pub fn verify_chain(&self) -> Result<()> {
        let height = self.height();
        if height == 0 {
            return Ok(()); // Only genesis, nothing to verify
        }

        let mut prev_block = self.get_genesis()?.ok_or(ChainError::EmptyChain)?;

        for h in 1..=height {
            let block = self.get_block_at(h)?.ok_or(ChainError::BlockNotFound(h))?;

            block.verify_chain(&prev_block)?;
            prev_block = block;
        }

        Ok(())
    }

    /// Get blocks in a height range (inclusive).
    pub fn get_blocks_range(&self, start: u64, end: u64) -> Result<Vec<Block>> {
        let mut blocks = Vec::with_capacity((end - start + 1) as usize);

        for h in start..=end {
            if let Some(block) = self.get_block_at(h)? {
                blocks.push(block);
            }
        }

        Ok(blocks)
    }

    /// Iterate over all blocks from genesis to tip.
    pub fn iter(&self) -> ChainIterator<'_> {
        ChainIterator {
            chain: self,
            current: 0,
            end: self.height(),
        }
    }

    /// Get history of changes for a specific key.
    pub fn history(&self, key: &str) -> Result<Vec<(u64, Transaction)>> {
        let mut history = Vec::new();

        for h in 0..=self.height() {
            if let Some(block) = self.get_block_at(h)? {
                for tx in &block.transactions {
                    if tx.affected_key() == key {
                        history.push((h, tx.clone()));
                    }
                }
            }
        }

        Ok(history)
    }

    /// Create a new block builder for the next height.
    pub fn new_block(&self) -> BlockBuilder {
        BlockBuilder {
            height: self.height() + 1,
            prev_hash: self.tip_hash(),
            proposer: self.node_id.clone(),
            transactions: Vec::new(),
            delta_embedding: Vec::new(),
            quantized_codes: Vec::new(),
        }
    }

    /// Store a block in the graph engine.
    fn store_block(&self, block: &Block) -> Result<()> {
        let key = block_key(block.header.height);
        let bytes =
            bincode::serialize(block).map_err(|e| ChainError::SerializationError(e.to_string()))?;

        let mut data = tensor_store::TensorData::new();
        data.set(
            "_block",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Bytes(bytes)),
        );
        data.set(
            "_height",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(
                block.header.height as i64,
            )),
        );
        data.set(
            "_hash",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(hex::encode(
                block.hash(),
            ))),
        );
        data.set(
            "_timestamp",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(
                block.header.timestamp as i64,
            )),
        );

        self.graph
            .store()
            .put(&key, data)
            .map_err(|e| ChainError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Load the current height from storage.
    fn load_height(&self) -> Option<u64> {
        let data = self.graph.store().get(CHAIN_META_KEY).ok()?;
        match data.get("height") {
            Some(tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(h))) => {
                Some(*h as u64)
            },
            _ => None,
        }
    }

    /// Save the current height to storage.
    fn save_height(&self, height: u64) -> Result<()> {
        let mut data = tensor_store::TensorData::new();
        data.set(
            "height",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(height as i64)),
        );
        self.graph
            .store()
            .put(CHAIN_META_KEY, data)
            .map_err(|e| ChainError::StorageError(e.to_string()))?;
        Ok(())
    }
}

/// Generate the storage key for a block at a given height.
fn block_key(height: u64) -> String {
    format!("{}{}", BLOCK_PREFIX, height)
}

/// Iterator over blocks in the chain.
pub struct ChainIterator<'a> {
    chain: &'a Chain,
    current: u64,
    end: u64,
}

impl<'a> Iterator for ChainIterator<'a> {
    type Item = Result<Block>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current > self.end {
            return None;
        }

        let height = self.current;
        self.current += 1;

        match self.chain.get_block_at(height) {
            Ok(Some(block)) => Some(Ok(block)),
            Ok(None) => Some(Err(ChainError::BlockNotFound(height))),
            Err(e) => Some(Err(e)),
        }
    }
}

/// Builder for creating new blocks.
pub struct BlockBuilder {
    height: u64,
    prev_hash: BlockHash,
    proposer: NodeId,
    transactions: Vec<Transaction>,
    delta_embedding: Vec<f32>,
    quantized_codes: Vec<u16>,
}

impl BlockBuilder {
    /// Add a transaction to the block.
    pub fn add_transaction(mut self, tx: Transaction) -> Self {
        self.transactions.push(tx);
        self
    }

    /// Add multiple transactions.
    pub fn add_transactions(mut self, txs: impl IntoIterator<Item = Transaction>) -> Self {
        self.transactions.extend(txs);
        self
    }

    /// Set the delta embedding.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.delta_embedding = embedding;
        self
    }

    /// Set the quantized codes.
    pub fn with_codes(mut self, codes: Vec<u16>) -> Self {
        self.quantized_codes = codes;
        self
    }

    /// Build the block.
    pub fn build(self) -> Block {
        let header = BlockHeader::new(
            self.height,
            self.prev_hash,
            [0u8; 32], // Will be computed
            [0u8; 32], // State root
            self.proposer,
        )
        .with_embedding(self.delta_embedding)
        .with_codes(self.quantized_codes);

        let mut block = Block::new(header, self.transactions);

        // Compute tx_root
        block.header.tx_root = block.compute_tx_root();

        block
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_chain() -> Chain {
        let store = tensor_store::TensorStore::new();
        let graph = Arc::new(GraphEngine::with_store(store));
        Chain::new(graph, "test_node".to_string())
    }

    #[test]
    fn test_chain_initialization() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        assert_eq!(chain.height(), 0);

        let genesis = chain.get_genesis().unwrap().unwrap();
        assert_eq!(genesis.header.height, 0);
        assert_eq!(genesis.header.prev_hash, [0u8; 32]);
    }

    #[test]
    fn test_append_block() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let block = chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: "test".to_string(),
                data: vec![1, 2, 3],
            })
            .build();

        let hash = chain.append(block).unwrap();

        assert_eq!(chain.height(), 1);
        assert_eq!(chain.tip_hash(), hash);

        let stored = chain.get_block_at(1).unwrap().unwrap();
        assert_eq!(stored.header.height, 1);
        assert_eq!(stored.transactions.len(), 1);
    }

    #[test]
    fn test_append_multiple_blocks() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        for i in 1..=5 {
            let block = chain
                .new_block()
                .add_transaction(Transaction::Put {
                    key: format!("key{}", i),
                    data: vec![i as u8],
                })
                .build();

            chain.append(block).unwrap();
        }

        assert_eq!(chain.height(), 5);

        // Verify chain integrity
        chain.verify_chain().unwrap();
    }

    #[test]
    fn test_block_builder() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let block = chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: "k1".to_string(),
                data: vec![1],
            })
            .add_transaction(Transaction::Delete {
                key: "k2".to_string(),
            })
            .with_embedding(vec![0.1, 0.2, 0.3])
            .with_codes(vec![1, 2, 3])
            .build();

        assert_eq!(block.header.height, 1);
        assert_eq!(block.transactions.len(), 2);
        assert_eq!(block.header.delta_embedding.len(), 3);
        assert_eq!(block.header.quantized_codes.len(), 3);
    }

    #[test]
    fn test_chain_history() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Add blocks with transactions on same key
        for i in 1..=3 {
            let block = chain
                .new_block()
                .add_transaction(Transaction::Put {
                    key: "shared_key".to_string(),
                    data: vec![i as u8],
                })
                .build();
            chain.append(block).unwrap();
        }

        let history = chain.history("shared_key").unwrap();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].0, 1); // height
        assert_eq!(history[1].0, 2);
        assert_eq!(history[2].0, 3);
    }

    #[test]
    fn test_chain_iterator() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        for _ in 1..=3 {
            let block = chain.new_block().build();
            chain.append(block).unwrap();
        }

        let blocks: Vec<_> = chain.iter().collect();
        assert_eq!(blocks.len(), 4); // Genesis + 3 blocks

        for (i, result) in blocks.iter().enumerate() {
            let block = result.as_ref().unwrap();
            assert_eq!(block.header.height, i as u64);
        }
    }

    #[test]
    fn test_invalid_block_height() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Try to append block with wrong height
        let mut block = chain.new_block().build();
        block.header.height = 5; // Should be 1

        let result = chain.append(block);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_prev_hash() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let mut block = chain.new_block().build();
        block.header.prev_hash = [99u8; 32]; // Wrong prev_hash

        let result = chain.append(block);
        assert!(result.is_err());
    }
}
