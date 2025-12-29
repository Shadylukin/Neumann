//! Block structure for the tensor chain.
//!
//! A block contains:
//! - Header with chain metadata and semantic embedding
//! - Transactions (tensor operations)
//! - Validator signatures for consensus

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{ChainError, Result};

/// Unique identifier for a node in the network.
pub type NodeId = String;

/// SHA-256 hash of a block.
pub type BlockHash = [u8; 32];

/// Block header containing chain metadata and semantic embedding.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BlockHeader {
    /// Block height (monotonically increasing from 0).
    pub height: u64,

    /// SHA-256 hash of the previous block header.
    pub prev_hash: BlockHash,

    /// Merkle root of transactions in this block.
    pub tx_root: BlockHash,

    /// Merkle root of state after applying transactions.
    pub state_root: BlockHash,

    /// Semantic embedding of block contents (for similarity consensus).
    pub delta_embedding: Vec<f32>,

    /// Quantized codebook indices (for compact representation).
    pub quantized_codes: Vec<u16>,

    /// Unix timestamp in milliseconds.
    pub timestamp: u64,

    /// Node ID of the block proposer.
    pub proposer: NodeId,

    /// Blake2b HMAC signature of the header.
    pub signature: Vec<u8>,
}

impl BlockHeader {
    /// Create a new block header.
    pub fn new(
        height: u64,
        prev_hash: BlockHash,
        tx_root: BlockHash,
        state_root: BlockHash,
        proposer: NodeId,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            height,
            prev_hash,
            tx_root,
            state_root,
            delta_embedding: Vec::new(),
            quantized_codes: Vec::new(),
            timestamp,
            proposer,
            signature: Vec::new(),
        }
    }

    /// Compute the SHA-256 hash of this header.
    pub fn hash(&self) -> BlockHash {
        let mut hasher = Sha256::new();

        // Hash all fields except signature
        hasher.update(self.height.to_le_bytes());
        hasher.update(self.prev_hash);
        hasher.update(self.tx_root);
        hasher.update(self.state_root);

        // Hash embedding as bytes
        for val in &self.delta_embedding {
            hasher.update(val.to_le_bytes());
        }

        // Hash quantized codes
        for code in &self.quantized_codes {
            hasher.update(code.to_le_bytes());
        }

        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.proposer.as_bytes());

        hasher.finalize().into()
    }

    /// Set the delta embedding for this block.
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.delta_embedding = embedding;
        self
    }

    /// Set the quantized codes for this block.
    pub fn with_codes(mut self, codes: Vec<u16>) -> Self {
        self.quantized_codes = codes;
        self
    }

    /// Sign this header with the given signature.
    pub fn with_signature(mut self, signature: Vec<u8>) -> Self {
        self.signature = signature;
        self
    }
}

/// A transaction representing a tensor operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Transaction {
    /// Store tensor data.
    Put { key: String, data: Vec<u8> },

    /// Delete tensor data.
    Delete { key: String },

    /// Store embedding vector.
    Embed { key: String, vector: Vec<f32> },

    /// Create graph node.
    NodeCreate { key: String, label: String },

    /// Delete graph node.
    NodeDelete { key: String },

    /// Create graph edge.
    EdgeCreate {
        from: String,
        to: String,
        edge_type: String,
    },

    /// Table insert.
    TableInsert { table: String, values: Vec<u8> },

    /// Table update.
    TableUpdate {
        table: String,
        row_id: u64,
        values: Vec<u8>,
    },

    /// Table delete.
    TableDelete { table: String, row_id: u64 },
}

impl Transaction {
    /// Get the primary key affected by this transaction.
    pub fn affected_key(&self) -> &str {
        match self {
            Transaction::Put { key, .. } => key,
            Transaction::Delete { key } => key,
            Transaction::Embed { key, .. } => key,
            Transaction::NodeCreate { key, .. } => key,
            Transaction::NodeDelete { key } => key,
            Transaction::EdgeCreate { from, .. } => from,
            Transaction::TableInsert { table, .. } => table,
            Transaction::TableUpdate { table, .. } => table,
            Transaction::TableDelete { table, .. } => table,
        }
    }

    /// Compute the hash of this transaction.
    pub fn hash(&self) -> [u8; 32] {
        let bytes = bincode::serialize(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hasher.finalize().into()
    }
}

/// Validator signature for consensus.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidatorSignature {
    /// Node ID of the validator.
    pub validator: NodeId,
    /// Signature bytes.
    pub signature: Vec<u8>,
    /// Block hash being signed.
    pub block_hash: BlockHash,
}

/// A complete block in the tensor chain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Block {
    /// Block header with metadata.
    pub header: BlockHeader,

    /// Transactions in this block.
    pub transactions: Vec<Transaction>,

    /// Validator signatures (for multi-node consensus).
    pub signatures: Vec<ValidatorSignature>,
}

impl Block {
    /// Create a new block.
    pub fn new(header: BlockHeader, transactions: Vec<Transaction>) -> Self {
        Self {
            header,
            transactions,
            signatures: Vec::new(),
        }
    }

    /// Create the genesis block (height 0).
    pub fn genesis(proposer: NodeId) -> Self {
        let header = BlockHeader::new(
            0, [0u8; 32], // No previous block
            [0u8; 32], // Empty tx root
            [0u8; 32], // Initial state root
            proposer,
        );

        Self {
            header,
            transactions: Vec::new(),
            signatures: Vec::new(),
        }
    }

    /// Get the hash of this block's header.
    pub fn hash(&self) -> BlockHash {
        self.header.hash()
    }

    /// Compute the merkle root of transactions.
    pub fn compute_tx_root(&self) -> BlockHash {
        if self.transactions.is_empty() {
            return [0u8; 32];
        }

        let leaves: Vec<[u8; 32]> = self.transactions.iter().map(|tx| tx.hash()).collect();

        merkle_root(&leaves)
    }

    /// Verify that the tx_root matches the transactions.
    pub fn verify_tx_root(&self) -> bool {
        self.header.tx_root == self.compute_tx_root()
    }

    /// Verify this block follows the previous block.
    pub fn verify_chain(&self, prev_block: &Block) -> Result<()> {
        // Check height is consecutive
        if self.header.height != prev_block.header.height + 1 {
            return Err(ChainError::ValidationFailed(format!(
                "height {} does not follow {}",
                self.header.height, prev_block.header.height
            )));
        }

        // Check prev_hash matches
        let expected_prev_hash = prev_block.hash();
        if self.header.prev_hash != expected_prev_hash {
            return Err(ChainError::InvalidHash {
                expected: hex::encode(expected_prev_hash),
                actual: hex::encode(self.header.prev_hash),
            });
        }

        // Check timestamp is not before previous block
        if self.header.timestamp < prev_block.header.timestamp {
            return Err(ChainError::ValidationFailed(
                "timestamp before previous block".to_string(),
            ));
        }

        Ok(())
    }

    /// Add a validator signature to this block.
    pub fn add_signature(&mut self, signature: ValidatorSignature) {
        self.signatures.push(signature);
    }

    /// Get all keys affected by transactions in this block.
    pub fn affected_keys(&self) -> Vec<&str> {
        self.transactions
            .iter()
            .map(|tx| tx.affected_key())
            .collect()
    }
}

/// Compute merkle root from a list of leaf hashes.
fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut level: Vec<[u8; 32]> = leaves.to_vec();

    while level.len() > 1 {
        let mut next_level = Vec::with_capacity(level.len().div_ceil(2));

        for chunk in level.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(chunk[0]);
            if chunk.len() > 1 {
                hasher.update(chunk[1]);
            } else {
                // Odd number of leaves - duplicate the last one
                hasher.update(chunk[0]);
            }
            next_level.push(hasher.finalize().into());
        }

        level = next_level;
    }

    level[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genesis_block() {
        let genesis = Block::genesis("node1".to_string());

        assert_eq!(genesis.header.height, 0);
        assert_eq!(genesis.header.prev_hash, [0u8; 32]);
        assert!(genesis.transactions.is_empty());
    }

    #[test]
    fn test_block_hash_deterministic() {
        let header = BlockHeader::new(1, [1u8; 32], [2u8; 32], [3u8; 32], "proposer".to_string());

        let hash1 = header.hash();
        let hash2 = header.hash();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_block_chain_verification() {
        let genesis = Block::genesis("node1".to_string());

        let mut next_header =
            BlockHeader::new(1, genesis.hash(), [0u8; 32], [0u8; 32], "node1".to_string());
        next_header.timestamp = genesis.header.timestamp + 1000;

        let next_block = Block::new(next_header, vec![]);

        assert!(next_block.verify_chain(&genesis).is_ok());
    }

    #[test]
    fn test_block_chain_verification_fails_on_wrong_prev_hash() {
        let genesis = Block::genesis("node1".to_string());

        let mut next_header = BlockHeader::new(
            1,
            [99u8; 32], // Wrong prev_hash
            [0u8; 32],
            [0u8; 32],
            "node1".to_string(),
        );
        next_header.timestamp = genesis.header.timestamp + 1000;

        let next_block = Block::new(next_header, vec![]);

        assert!(next_block.verify_chain(&genesis).is_err());
    }

    #[test]
    fn test_merkle_root() {
        let leaves = vec![[1u8; 32], [2u8; 32], [3u8; 32]];
        let root = merkle_root(&leaves);

        // Should be deterministic
        assert_eq!(root, merkle_root(&leaves));

        // Different leaves should produce different root
        let other_leaves = vec![[4u8; 32], [5u8; 32]];
        assert_ne!(root, merkle_root(&other_leaves));
    }

    #[test]
    fn test_transaction_affected_key() {
        let tx = Transaction::Put {
            key: "test_key".to_string(),
            data: vec![1, 2, 3],
        };
        assert_eq!(tx.affected_key(), "test_key");

        let tx = Transaction::EdgeCreate {
            from: "node_a".to_string(),
            to: "node_b".to_string(),
            edge_type: "link".to_string(),
        };
        assert_eq!(tx.affected_key(), "node_a");
    }
}
