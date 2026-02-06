// SPDX-License-Identifier: MIT OR Apache-2.0
//! Chain structure linking blocks via graph edges.
//!
//! Blocks are linked as a directed graph where each block points to its predecessor.
//! The chain provides:
//! - Append-only block storage
//! - Chain traversal and validation
//! - Height-indexed block lookup
//! - Similarity search over block embeddings

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use graph_engine::{GraphEngine, PropertyValue};
use parking_lot::{Mutex, RwLock};
use tensor_store::SparseVector;

use crate::{
    block::{Block, BlockHash, BlockHeader, NodeId, Transaction},
    error::{ChainError, Result},
    signing::{Identity, ValidatorRegistry},
};

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

    /// Serialize block appends to prevent TOCTOU race conditions.
    append_lock: Mutex<()>,

    /// Optional validator registry for signature verification.
    validator_registry: Option<Arc<ValidatorRegistry>>,
}

impl Chain {
    pub fn new(graph: Arc<GraphEngine>, node_id: NodeId) -> Self {
        Self {
            graph,
            height: AtomicU64::new(0),
            tip_hash: RwLock::new([0u8; 32]),
            node_id,
            append_lock: Mutex::new(()),
            validator_registry: None,
        }
    }

    /// Create a new chain that verifies block signatures with the provided registry.
    pub fn with_registry(
        graph: Arc<GraphEngine>,
        node_id: NodeId,
        registry: Arc<ValidatorRegistry>,
    ) -> Self {
        Self {
            graph,
            height: AtomicU64::new(0),
            tip_hash: RwLock::new([0u8; 32]),
            node_id,
            append_lock: Mutex::new(()),
            validator_registry: Some(registry),
        }
    }

    /// Helper to add a chain edge between two block keys.
    fn add_chain_edge(&self, from_key: &str, to_key: &str, edge_type: &str) -> Result<u64> {
        let get_or_create = |key: &str| -> u64 {
            if let Ok(nodes) = self
                .graph
                .find_nodes_by_property("entity_key", &PropertyValue::String(key.to_string()))
            {
                if let Some(node) = nodes.first() {
                    return node.id;
                }
            }
            let mut props = HashMap::new();
            props.insert(
                "entity_key".to_string(),
                PropertyValue::String(key.to_string()),
            );
            self.graph.create_node("ChainBlock", props).unwrap_or(0)
        };

        let from_node = get_or_create(from_key);
        let to_node = get_or_create(to_key);
        self.graph
            .create_edge(from_node, to_node, edge_type, HashMap::new(), true)
            .map_err(|e| ChainError::GraphError(e.to_string()))
    }

    /// Initialize the chain, creating genesis block if needed.
    ///
    /// Idempotent: safe to call multiple times. If the chain already exists,
    /// loads existing height and tip from storage without creating a new genesis.
    ///
    /// # Errors
    ///
    /// Returns an error if block storage or retrieval fails.
    pub fn initialize(&self) -> Result<()> {
        // Check if chain already exists
        if let Some(mut height) = self.load_height() {
            // If metadata is ahead of stored blocks, walk back to last existing
            while height > 0 && self.get_block_at(height)?.is_none() {
                height = height.saturating_sub(1);
            }

            // If metadata is behind stored blocks, walk forward to last existing
            loop {
                let next = height + 1;
                if self.get_block_at(next)?.is_some() {
                    height = next;
                } else {
                    break;
                }
            }

            self.height.store(height, Ordering::SeqCst);

            // Load tip hash
            if let Some(tip) = self.get_block_at(height)? {
                *self.tip_hash.write() = tip.hash();
            }

            // Persist corrected height if needed
            self.save_height(height)?;

            return Ok(());
        }

        // Create genesis block
        let genesis = Block::genesis(self.node_id.clone());
        self.store_block(&genesis)?;

        *self.tip_hash.write() = genesis.hash();
        self.save_height(0)?;

        Ok(())
    }

    pub fn height(&self) -> u64 {
        self.height.load(Ordering::SeqCst)
    }

    pub fn tip_hash(&self) -> BlockHash {
        *self.tip_hash.read()
    }

    pub fn is_empty(&self) -> bool {
        self.height() == 0
    }

    /// Append a new block to the chain.
    ///
    /// # Errors
    ///
    /// Returns an error if the block height or previous hash is invalid,
    /// the transaction root does not match, the block is unsigned, or storage fails.
    pub fn append(&self, mut block: Block) -> Result<BlockHash> {
        // Serialize appends to prevent TOCTOU race conditions where two
        // concurrent appends read the same height and both attempt to append
        let _guard = self.append_lock.lock();

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

        // Verify tx_root matches transactions
        if !block.verify_tx_root() {
            return Err(ChainError::ValidationFailed(
                "tx_root does not match transactions".to_string(),
            ));
        }

        // Reject unsigned blocks (except genesis)
        if expected_height > 1 {
            if block.header.signature.is_empty() {
                return Err(ChainError::ValidationFailed(
                    "block must be signed by proposer".to_string(),
                ));
            }

            if let Some(ref registry) = self.validator_registry {
                block.header.verify_signature(registry)?;
            }
        }

        // Store the block
        let block_hash = block.hash();
        self.store_block(&block)?;

        // Create chain link edge
        let prev_key = block_key(current_height);
        let new_key = block_key(expected_height);
        self.add_chain_edge(&prev_key, &new_key, CHAIN_EDGE_TYPE)?;

        // Update chain state
        self.height.store(expected_height, Ordering::SeqCst);
        *self.tip_hash.write() = block_hash;
        self.save_height(expected_height)?;

        Ok(block_hash)
    }

    /// # Errors
    ///
    /// Returns an error if the stored block data is missing or cannot be deserialized.
    pub fn get_block_at(&self, height: u64) -> Result<Option<Block>> {
        let key = block_key(height);

        let Ok(data) = self.graph.store().get(&key) else {
            return Ok(None);
        };

        // Deserialize block from stored bytes
        let Some(tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Bytes(bytes))) =
            data.get("_block")
        else {
            return Err(ChainError::StorageError("missing block data".to_string()));
        };

        let block: Block = bitcode::deserialize(bytes)
            .map_err(|e| ChainError::SerializationError(e.to_string()))?;

        Ok(Some(block))
    }

    /// # Errors
    ///
    /// Returns an error if the tip block cannot be retrieved.
    pub fn get_tip(&self) -> Result<Option<Block>> {
        self.get_block_at(self.height())
    }

    /// # Errors
    ///
    /// Returns an error if the genesis block cannot be retrieved.
    pub fn get_genesis(&self) -> Result<Option<Block>> {
        self.get_block_at(0)
    }

    /// Verify the entire chain integrity.
    ///
    /// # Errors
    ///
    /// Returns an error if any block is missing, has an invalid chain link,
    /// or fails signature verification.
    pub fn verify_chain(&self) -> Result<()> {
        let height = self.height();
        if height == 0 {
            return Ok(()); // Only genesis, nothing to verify
        }

        let mut prev_block = self.get_genesis()?.ok_or(ChainError::EmptyChain)?;

        for h in 1..=height {
            let block = self.get_block_at(h)?.ok_or(ChainError::BlockNotFound(h))?;

            block.verify_chain(&prev_block)?;
            if let Some(ref registry) = self.validator_registry {
                block.header.verify_signature(registry)?;
            }
            prev_block = block;
        }

        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if any block in the range cannot be retrieved.
    pub fn get_blocks_range(&self, start: u64, end: u64) -> Result<Vec<Block>> {
        #[allow(clippy::cast_possible_truncation)] // block count fits in usize
        let mut blocks = Vec::with_capacity((end - start + 1) as usize);

        for h in start..=end {
            if let Some(block) = self.get_block_at(h)? {
                blocks.push(block);
            }
        }

        Ok(blocks)
    }

    pub fn iter(&self) -> ChainIterator<'_> {
        ChainIterator {
            chain: self,
            current: 0,
            end: self.height(),
        }
    }

    /// Get history of changes for a specific key.
    ///
    /// # Errors
    ///
    /// Returns an error if any block cannot be retrieved during traversal.
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

    pub fn new_block(&self) -> BlockBuilder {
        BlockBuilder {
            height: self.height() + 1,
            prev_hash: self.tip_hash(),
            proposer: self.node_id.clone(),
            transactions: Vec::new(),
            delta_embedding: SparseVector::new(0),
            quantized_codes: Vec::new(),
            state_root: [0u8; 32],
            signature: Vec::new(),
        }
    }

    /// Store a block in the graph engine.
    fn store_block(&self, block: &Block) -> Result<()> {
        let key = block_key(block.header.height);
        let bytes =
            bitcode::serialize(block).map_err(|e| ChainError::SerializationError(e.to_string()))?;

        let mut data = tensor_store::TensorData::new();
        data.set(
            "_block",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Bytes(bytes)),
        );
        #[allow(clippy::cast_possible_wrap)] // block height won't exceed i64::MAX
        let height_i64 = block.header.height as i64;
        data.set(
            "_height",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(height_i64)),
        );
        data.set(
            "_hash",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(hex::encode(
                block.hash(),
            ))),
        );
        #[allow(clippy::cast_possible_wrap)] // timestamp won't exceed i64::MAX
        let timestamp_i64 = block.header.timestamp as i64;
        data.set(
            "_timestamp",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(timestamp_i64)),
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
                #[allow(clippy::cast_sign_loss)] // height is always non-negative
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
            #[allow(clippy::cast_possible_wrap)] // height won't exceed i64::MAX
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
    format!("{BLOCK_PREFIX}{height}")
}

/// Iterator over blocks in the chain.
pub struct ChainIterator<'a> {
    chain: &'a Chain,
    current: u64,
    end: u64,
}

impl Iterator for ChainIterator<'_> {
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

impl<'a> IntoIterator for &'a Chain {
    type Item = Result<Block>;
    type IntoIter = ChainIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Builder for creating new blocks.
pub struct BlockBuilder {
    height: u64,
    prev_hash: BlockHash,
    proposer: NodeId,
    transactions: Vec<Transaction>,
    delta_embedding: SparseVector,
    quantized_codes: Vec<u16>,
    state_root: BlockHash,
    signature: Vec<u8>,
}

impl BlockBuilder {
    #[must_use]
    pub fn add_transaction(mut self, tx: Transaction) -> Self {
        self.transactions.push(tx);
        self
    }

    #[must_use]
    pub fn add_transactions(mut self, txs: impl IntoIterator<Item = Transaction>) -> Self {
        self.transactions.extend(txs);
        self
    }

    #[must_use]
    pub fn with_embedding(mut self, embedding: SparseVector) -> Self {
        self.delta_embedding = embedding;
        self
    }

    #[must_use]
    pub fn with_dense_embedding(mut self, embedding: &[f32]) -> Self {
        self.delta_embedding = SparseVector::from_dense(embedding);
        self
    }

    #[must_use]
    pub fn with_codes(mut self, codes: Vec<u16>) -> Self {
        self.quantized_codes = codes;
        self
    }

    #[must_use]
    pub fn with_state_root(mut self, state_root: BlockHash) -> Self {
        self.state_root = state_root;
        self
    }

    #[must_use]
    pub fn with_signature(mut self, signature: Vec<u8>) -> Self {
        self.signature = signature;
        self
    }

    #[must_use]
    pub fn build(self) -> Block {
        let header = BlockHeader::new(
            self.height,
            self.prev_hash,
            [0u8; 32], // Will be computed
            self.state_root,
            self.proposer,
        )
        .with_embedding(self.delta_embedding)
        .with_codes(self.quantized_codes)
        .with_signature(self.signature);

        let mut block = Block::new(header, self.transactions);

        // Compute tx_root
        block.header.tx_root = block.compute_tx_root();

        block
    }

    /// Build and sign the block with the given identity.
    ///
    /// This constructs the block header, computes the signing bytes, signs them
    /// with the provided Ed25519 identity, and returns the signed block.
    #[must_use]
    pub fn sign_and_build(self, identity: &Identity) -> Block {
        // Build header without signature first to compute signing bytes
        let header = BlockHeader::new(
            self.height,
            self.prev_hash,
            [0u8; 32], // Will be computed
            self.state_root,
            self.proposer,
        )
        .with_embedding(self.delta_embedding)
        .with_codes(self.quantized_codes);

        let mut block = Block::new(header, self.transactions);

        // Compute tx_root
        block.header.tx_root = block.compute_tx_root();

        // Sign the header's canonical bytes
        let signing_bytes = block.header.signing_bytes();
        let signature = identity.sign(&signing_bytes);

        // Attach the signature
        block.header.signature = signature;

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

    fn test_signature() -> Vec<u8> {
        vec![0u8; 64]
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
                    key: format!("key{i}"),
                    data: vec![i as u8],
                })
                .with_signature(test_signature())
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
            .with_dense_embedding(&[0.1, 0.2, 0.3])
            .with_codes(vec![1, 2, 3])
            .build();

        assert_eq!(block.header.height, 1);
        assert_eq!(block.transactions.len(), 2);
        assert_eq!(block.header.delta_embedding.dimension(), 3);
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
                .with_signature(test_signature())
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
            let block = chain.new_block().with_signature(test_signature()).build();
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

    #[test]
    fn test_chain_is_empty() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // After genesis only
        assert!(chain.is_empty());

        // Add a block
        let block = chain.new_block().build();
        chain.append(block).unwrap();

        assert!(!chain.is_empty());
    }

    #[test]
    fn test_chain_get_tip() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Genesis is the tip initially
        let tip = chain.get_tip().unwrap().unwrap();
        assert_eq!(tip.header.height, 0);

        // Add a block
        let block = chain.new_block().build();
        chain.append(block).unwrap();

        // Now height 1 is the tip
        let tip = chain.get_tip().unwrap().unwrap();
        assert_eq!(tip.header.height, 1);
    }

    #[test]
    fn test_chain_get_blocks_range() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Add 5 blocks
        for _ in 1..=5 {
            let block = chain.new_block().with_signature(test_signature()).build();
            chain.append(block).unwrap();
        }

        // Get range [2, 4]
        let blocks = chain.get_blocks_range(2, 4).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].header.height, 2);
        assert_eq!(blocks[1].header.height, 3);
        assert_eq!(blocks[2].header.height, 4);
    }

    #[test]
    fn test_chain_get_blocks_range_partial() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Add 2 blocks
        for _ in 1..=2 {
            let block = chain.new_block().with_signature(test_signature()).build();
            chain.append(block).unwrap();
        }

        // Try to get range that goes beyond existing blocks
        let blocks = chain.get_blocks_range(1, 5).unwrap();
        // Should only get blocks 1 and 2
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_block_builder_add_transactions() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let txs = vec![
            Transaction::Put {
                key: "k1".to_string(),
                data: vec![1],
            },
            Transaction::Put {
                key: "k2".to_string(),
                data: vec![2],
            },
            Transaction::Delete {
                key: "k3".to_string(),
            },
        ];

        let block = chain.new_block().add_transactions(txs).build();

        assert_eq!(block.transactions.len(), 3);
    }

    #[test]
    fn test_reinitialize_existing_chain() {
        let store = tensor_store::TensorStore::new();
        let graph = Arc::new(GraphEngine::with_store(store.clone()));
        let chain1 = Chain::new(graph.clone(), "node1".to_string());

        // Initialize and add blocks
        chain1.initialize().unwrap();
        for _ in 1..=3 {
            let block = chain1.new_block().with_signature(test_signature()).build();
            chain1.append(block).unwrap();
        }
        assert_eq!(chain1.height(), 3);

        // Create a new chain pointing to the same store
        let chain2 = Chain::new(graph.clone(), "node1".to_string());
        chain2.initialize().unwrap();

        // Should load existing height
        assert_eq!(chain2.height(), 3);
        assert_eq!(chain2.tip_hash(), chain1.tip_hash());
    }

    #[test]
    fn test_verify_chain_empty() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Only genesis - should pass
        chain.verify_chain().unwrap();
    }

    #[test]
    fn test_verify_chain_with_blocks() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Add valid blocks
        for _ in 1..=3 {
            let block = chain.new_block().with_signature(test_signature()).build();
            chain.append(block).unwrap();
        }

        // Verify should pass
        chain.verify_chain().unwrap();
    }

    #[test]
    fn test_get_block_at_nonexistent() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Try to get a block that doesn't exist
        let result = chain.get_block_at(100).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_history_no_matches() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Add blocks with different keys
        for i in 1..=3 {
            let block = chain
                .new_block()
                .add_transaction(Transaction::Put {
                    key: format!("key{i}"),
                    data: vec![i as u8],
                })
                .with_signature(test_signature())
                .build();
            chain.append(block).unwrap();
        }

        // Look for a key that doesn't exist
        let history = chain.history("nonexistent").unwrap();
        assert!(history.is_empty());
    }

    #[test]
    fn test_append_computes_tx_root() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let block = chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: "k".to_string(),
                data: vec![1],
            })
            .build();

        // tx_root should be set by builder already
        assert_ne!(block.header.tx_root, [0u8; 32]);

        chain.append(block).unwrap();

        // Retrieve and verify
        let stored = chain.get_block_at(1).unwrap().unwrap();
        assert_ne!(stored.header.tx_root, [0u8; 32]);
    }

    #[test]
    fn test_chain_iterator_handles_all_heights() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Add blocks
        for _ in 1..=2 {
            let block = chain.new_block().with_signature(test_signature()).build();
            chain.append(block).unwrap();
        }

        // Iterate and collect
        let mut heights = Vec::new();
        for result in chain.iter() {
            let block = result.unwrap();
            heights.push(block.header.height);
        }

        assert_eq!(heights, vec![0, 1, 2]);
    }

    #[test]
    fn test_tip_hash_updates() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let genesis_hash = chain.tip_hash();

        // Add a block (height 1 - no signature required)
        let block = chain.new_block().build();
        let hash1 = chain.append(block).unwrap();

        assert_ne!(chain.tip_hash(), genesis_hash);
        assert_eq!(chain.tip_hash(), hash1);

        // Add another block (height 2+ requires signature)
        let block = chain.new_block().with_signature(test_signature()).build();
        let hash2 = chain.append(block).unwrap();

        assert_ne!(chain.tip_hash(), hash1);
        assert_eq!(chain.tip_hash(), hash2);
    }

    #[test]
    fn test_concurrent_append_serialization() {
        use std::thread;

        let store = tensor_store::TensorStore::new();
        let graph = Arc::new(GraphEngine::with_store(store));
        let chain = Arc::new(Chain::new(graph, "test_node".to_string()));
        chain.initialize().unwrap();

        // Spawn multiple threads trying to append simultaneously
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let chain = Arc::clone(&chain);
                thread::spawn(move || {
                    // Create a block for the next height
                    let block = chain.new_block().with_signature(test_signature()).build();
                    chain.append(block).map(|_| i)
                })
            })
            .collect();

        // Collect results
        let mut successes = 0;
        for handle in handles {
            if handle.join().unwrap().is_ok() {
                successes += 1;
            }
        }

        // With the append_lock, exactly one thread should succeed per height
        // Since all threads try to append at height 1, only one should succeed
        assert!(successes >= 1);
        assert_eq!(chain.height() as usize, successes);
    }

    #[test]
    fn test_chain_append_sequential_heights() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Append multiple blocks sequentially
        for _ in 0..5 {
            let block = chain.new_block().with_signature(test_signature()).build();
            chain.append(block).unwrap();
        }

        // Verify sequential heights
        assert_eq!(chain.height(), 5);
        for i in 0..=5 {
            let block = chain.get_block_at(i).unwrap().unwrap();
            assert_eq!(block.header.height, i);
        }
    }

    #[test]
    fn test_append_invalid_tx_root() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Build a block but manually corrupt the tx_root
        let mut block = chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: "test".to_string(),
                data: vec![1, 2, 3],
            })
            .build();

        // Set a wrong tx_root (not matching the transactions)
        block.header.tx_root = [0xAB; 32];

        let err = chain.append(block).unwrap_err();
        assert!(matches!(err, ChainError::ValidationFailed(msg) if msg.contains("tx_root")));
    }

    #[test]
    fn test_append_unsigned_block_at_height_2() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Append first block (height 1 - no signature required)
        let block1 = chain.new_block().build();
        chain.append(block1).unwrap();

        // Try to append second block without signature (height 2+ requires signature)
        let block2 = chain.new_block().build();
        let err = chain.append(block2).unwrap_err();
        assert!(matches!(err, ChainError::ValidationFailed(msg) if msg.contains("signed")));
    }

    #[test]
    fn test_get_block_at_missing_block_data() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Store malformed data at a block key (missing _block field)
        let key = format!("{}999", BLOCK_PREFIX);
        let mut data = tensor_store::TensorData::new();
        data.set(
            "wrong_field",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(42)),
        );
        chain.graph.store().put(&key, data).unwrap();

        // Try to get this block - should error due to missing _block field
        let err = chain.get_block_at(999).unwrap_err();
        assert!(matches!(err, ChainError::StorageError(msg) if msg.contains("missing block data")));
    }

    #[test]
    fn test_load_height_wrong_type() {
        let store = tensor_store::TensorStore::new();
        let graph = Arc::new(GraphEngine::with_store(store));
        let chain = Chain::new(graph, "test_node".to_string());

        // Store height with wrong type (String instead of Int)
        let mut data = tensor_store::TensorData::new();
        data.set(
            "height",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(
                "not_a_number".to_string(),
            )),
        );
        chain.graph.store().put(CHAIN_META_KEY, data).unwrap();

        // load_height should return None for wrong type
        assert!(chain.load_height().is_none());
    }

    #[test]
    fn test_chain_iterator_block_not_found() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Manually set height to claim we have blocks we don't
        chain.height.store(5, std::sync::atomic::Ordering::SeqCst);

        // Iterate - should get BlockNotFound errors for missing blocks
        let has_not_found = chain
            .iter()
            .any(|r| matches!(r, Err(ChainError::BlockNotFound(_))));
        assert!(has_not_found);
    }

    #[test]
    fn test_chain_iterator_storage_error() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Add a valid block at height 1
        let block = chain.new_block().build();
        chain.append(block).unwrap();

        // Store corrupted data at height 2
        let key = format!("{}2", BLOCK_PREFIX);
        let mut data = tensor_store::TensorData::new();
        data.set(
            "_block",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(999)),
        );
        chain.graph.store().put(&key, data).unwrap();

        // Manually set height to 2
        chain.height.store(2, std::sync::atomic::Ordering::SeqCst);

        // Iterate - height 2 should produce an error (wrong type for _block)
        let mut found_storage_error = false;
        for result in chain.iter() {
            if let Err(ChainError::StorageError(_)) = result {
                found_storage_error = true;
                break;
            }
        }
        assert!(found_storage_error, "Should have found StorageError");
    }

    #[test]
    fn test_block_builder_with_embedding() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let embedding = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let block = chain
            .new_block()
            .with_embedding(embedding.clone())
            .with_signature(test_signature())
            .build();

        assert_eq!(block.header.delta_embedding.dimension(), 3);
        chain.append(block).unwrap();
    }

    #[test]
    fn test_append_zero_tx_root_with_transactions() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Build block and manually zero out the tx_root
        let mut block = chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: "test".to_string(),
                data: vec![1, 2, 3],
            })
            .build();

        // Force tx_root to zero (append should compute it)
        block.header.tx_root = [0u8; 32];

        // Append should succeed - it will compute the tx_root
        let result = chain.append(block);
        assert!(result.is_ok());

        // Verify the stored block has correct tx_root
        let stored = chain.get_block_at(1).unwrap().unwrap();
        assert_ne!(stored.header.tx_root, [0u8; 32]);
    }

    #[test]
    fn test_block_builder_with_dense_embedding() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let block = chain
            .new_block()
            .with_dense_embedding(&[1.0, 2.0, 3.0, 4.0])
            .with_signature(test_signature())
            .build();

        assert_eq!(block.header.delta_embedding.dimension(), 4);
        chain.append(block).unwrap();
    }

    #[test]
    fn test_block_builder_with_codes() {
        let chain = create_test_chain();
        chain.initialize().unwrap();

        let codes = vec![1u16, 2, 3, 4, 5];
        let block = chain
            .new_block()
            .with_codes(codes.clone())
            .with_signature(test_signature())
            .build();

        assert_eq!(block.header.quantized_codes, codes);
        chain.append(block).unwrap();
    }

    #[test]
    fn test_block_builder_sign_and_build() {
        use crate::signing::Identity;

        let chain = create_test_chain();
        chain.initialize().unwrap();

        // Create an identity for signing
        let identity = Identity::generate();

        let block = chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: "test".to_string(),
                data: vec![1, 2, 3],
            })
            .sign_and_build(&identity);

        // Block should be signed with Ed25519 (64-byte signature)
        assert!(!block.header.signature.is_empty());
        assert_eq!(block.header.signature.len(), 64);

        chain.append(block).unwrap();
    }
}
