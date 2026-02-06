// SPDX-License-Identifier: MIT OR Apache-2.0
//! Block structure for the tensor chain.
//!
//! A block contains:
//! - Header with chain metadata and semantic embedding
//! - Transactions (tensor operations)
//! - Validator signatures for consensus

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tensor_store::SparseVector;

use crate::error::{ChainError, Result};
use crate::signing::ValidatorRegistry;

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
    pub delta_embedding: SparseVector,

    /// Quantized codebook indices (for compact representation).
    pub quantized_codes: Vec<u16>,

    /// Unix timestamp in milliseconds.
    pub timestamp: u64,

    /// Node ID of the block proposer.
    pub proposer: NodeId,

    pub signature: Vec<u8>,
}

impl Default for BlockHeader {
    fn default() -> Self {
        Self {
            height: 0,
            prev_hash: [0u8; 32],
            tx_root: [0u8; 32],
            state_root: [0u8; 32],
            delta_embedding: SparseVector::new(0),
            quantized_codes: Vec::new(),
            timestamp: 0,
            proposer: String::new(),
            signature: Vec::new(),
        }
    }
}

impl BlockHeader {
    #[must_use]
    pub fn new(
        height: u64,
        prev_hash: BlockHash,
        tx_root: BlockHash,
        state_root: BlockHash,
        proposer: NodeId,
    ) -> Self {
        #[allow(clippy::cast_possible_truncation)] // millis since epoch fits in u64 for centuries
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            height,
            prev_hash,
            tx_root,
            state_root,
            delta_embedding: SparseVector::new(0),
            quantized_codes: Vec::new(),
            timestamp,
            proposer,
            signature: Vec::new(),
        }
    }

    #[must_use]
    pub fn hash(&self) -> BlockHash {
        let mut hasher = Sha256::new();

        // Hash all fields except signature
        hasher.update(self.height.to_le_bytes());
        hasher.update(self.prev_hash);
        hasher.update(self.tx_root);
        hasher.update(self.state_root);

        // Embedding must be serialized for deterministic cross-platform hashing
        let embedding_bytes = match bitcode::serialize(&self.delta_embedding) {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialize delta_embedding for hash");
                Vec::new()
            },
        };
        hasher.update(&embedding_bytes);

        // Hash quantized codes
        for code in &self.quantized_codes {
            hasher.update(code.to_le_bytes());
        }

        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.proposer.as_bytes());

        hasher.finalize().into()
    }

    #[must_use]
    pub fn with_embedding(mut self, embedding: SparseVector) -> Self {
        self.delta_embedding = embedding;
        self
    }

    #[must_use]
    pub fn with_state_root(mut self, state_root: BlockHash) -> Self {
        self.state_root = state_root;
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
    pub fn with_signature(mut self, signature: Vec<u8>) -> Self {
        self.signature = signature;
        self
    }

    /// Get the canonical bytes of this header for signing/verification.
    /// This includes all fields except the signature itself.
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend(self.height.to_le_bytes());
        bytes.extend(self.prev_hash);
        bytes.extend(self.tx_root);
        bytes.extend(self.state_root);

        // Serialize embedding deterministically
        let embedding_bytes = match bitcode::serialize(&self.delta_embedding) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialize delta_embedding for signing");
                Vec::new()
            },
        };
        bytes.extend(&embedding_bytes);

        // Quantized codes
        for code in &self.quantized_codes {
            bytes.extend(code.to_le_bytes());
        }

        bytes.extend(self.timestamp.to_le_bytes());
        bytes.extend(self.proposer.as_bytes());

        bytes
    }

    /// Verify the proposer's signature on this block header.
    ///
    /// # Errors
    ///
    /// Returns an error if the signature is missing, invalid, or the proposer is unknown.
    pub fn verify_signature(&self, registry: &ValidatorRegistry) -> Result<()> {
        // SECURITY: Require non-empty signature
        if self.signature.is_empty() {
            return Err(ChainError::ValidationFailed(
                "missing block signature".to_string(),
            ));
        }

        // Look up proposer's public key
        let public_key = registry.get(&self.proposer).ok_or_else(|| {
            ChainError::ValidationFailed(format!("unknown proposer: {}", self.proposer))
        })?;

        // Get the canonical bytes to verify
        let message = self.signing_bytes();

        // Verify the signature
        public_key
            .verify(&message, &self.signature)
            .map_err(|_| ChainError::ValidationFailed("invalid block signature".to_string()))
    }
}

/// A transaction representing a tensor operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
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
    /// Returns the logical key affected by this transaction (used for locking).
    #[must_use]
    pub fn affected_key(&self) -> &str {
        match self {
            Self::Put { key, .. }
            | Self::Delete { key }
            | Self::Embed { key, .. }
            | Self::NodeCreate { key, .. }
            | Self::NodeDelete { key } => key,
            Self::EdgeCreate { from, .. } => from,
            Self::TableInsert { table, .. }
            | Self::TableUpdate { table, .. }
            | Self::TableDelete { table, .. } => table,
        }
    }

    /// Returns the actual storage key used in `TensorStore` for this transaction.
    ///
    /// This differs from `affected_key()` for transaction types that transform
    /// keys before storage (e.g., `Embed` uses "emb:{key}", `NodeCreate` uses "node:{key}").
    #[must_use]
    pub fn storage_key(&self) -> String {
        match self {
            Self::Put { key, .. } | Self::Delete { key } => key.clone(),
            Self::Embed { key, .. } => format!("emb:{key}"),
            Self::NodeCreate { key, .. } | Self::NodeDelete { key } => {
                format!("node:{key}")
            },
            Self::EdgeCreate {
                from,
                to,
                edge_type,
            } => {
                format!("edge:{from}:{to}:{edge_type}")
            },
            Self::TableInsert { table, .. }
            | Self::TableUpdate { table, .. }
            | Self::TableDelete { table, .. } => {
                format!("table:{table}")
            },
        }
    }

    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let bytes = match bitcode::serialize(self) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(error = %e, "failed to serialize transaction for hash");
                Vec::new()
            },
        };
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        hasher.finalize().into()
    }
}

/// Validator signature for consensus.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidatorSignature {
    /// Node ID of the validator.
    pub validator: NodeId,
    /// Signature bytes.
    pub signature: Vec<u8>,
    /// Block hash being signed.
    pub block_hash: BlockHash,
}

/// A complete block in the tensor chain.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Block {
    /// Block header with metadata.
    pub header: BlockHeader,

    /// Transactions in this block.
    pub transactions: Vec<Transaction>,

    /// Validator signatures (for multi-node consensus).
    pub signatures: Vec<ValidatorSignature>,
}

impl Block {
    #[must_use]
    pub fn new(header: BlockHeader, transactions: Vec<Transaction>) -> Self {
        Self {
            header,
            transactions,
            signatures: Vec::new(),
        }
    }

    #[must_use]
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

    #[must_use]
    pub fn hash(&self) -> BlockHash {
        self.header.hash()
    }

    /// Compute the transaction root using a binary Merkle tree.
    ///
    /// Hashes each transaction to form leaves, then recursively combines pairs
    /// with SHA-256. Odd leaves are duplicated for the final pair.
    #[must_use]
    pub fn compute_tx_root(&self) -> BlockHash {
        if self.transactions.is_empty() {
            return [0u8; 32];
        }

        let leaves: Vec<[u8; 32]> = self.transactions.iter().map(Transaction::hash).collect();

        merkle_root(&leaves)
    }

    #[must_use]
    pub fn verify_tx_root(&self) -> bool {
        self.header.tx_root == self.compute_tx_root()
    }

    /// Verify this block follows the given previous block.
    ///
    /// # Errors
    ///
    /// Returns an error if the height is not consecutive, the previous hash
    /// does not match, the transaction root is invalid, or the timestamp
    /// is before the previous block.
    pub fn verify_chain(&self, prev_block: &Self) -> Result<()> {
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

        // Verify tx_root matches transactions
        if !self.verify_tx_root() {
            return Err(ChainError::ValidationFailed(
                "tx_root does not match transactions".to_string(),
            ));
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
    ///
    /// # Errors
    ///
    /// Returns an error if the validator has already signed this block.
    pub fn add_signature(&mut self, signature: ValidatorSignature) -> crate::Result<()> {
        // Check for duplicate signer
        if self
            .signatures
            .iter()
            .any(|s| s.validator == signature.validator)
        {
            return Err(crate::ChainError::ValidationFailed(format!(
                "duplicate signer: {}",
                signature.validator
            )));
        }
        self.signatures.push(signature);
        Ok(())
    }

    #[must_use]
    pub fn affected_keys(&self) -> Vec<&str> {
        self.transactions
            .iter()
            .map(Transaction::affected_key)
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

    #[test]
    fn test_block_header_with_embedding() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());
        let embedding = SparseVector::from_dense(&[1.0, 2.0, 3.0, 4.0]);
        let header = header.with_embedding(embedding.clone());

        assert_eq!(header.delta_embedding.to_dense(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_block_header_with_dense_embedding() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());
        let header = header.with_dense_embedding(&[1.0, 2.0, 3.0, 4.0]);

        assert_eq!(header.delta_embedding.to_dense(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_block_header_with_codes() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());
        let codes = vec![1u16, 2, 3, 4, 5];
        let header = header.with_codes(codes.clone());

        assert_eq!(header.quantized_codes, codes);
    }

    #[test]
    fn test_block_header_with_signature() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());
        let sig = vec![0xde, 0xad, 0xbe, 0xef];
        let header = header.with_signature(sig.clone());

        assert_eq!(header.signature, sig);
    }

    #[test]
    fn test_block_header_hash_includes_embedding_and_codes() {
        let header1 = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string())
            .with_dense_embedding(&[1.0, 2.0])
            .with_codes(vec![1, 2, 3]);

        let header2 = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string())
            .with_dense_embedding(&[3.0, 4.0])
            .with_codes(vec![4, 5, 6]);

        // Different embeddings/codes should produce different hashes
        assert_ne!(header1.hash(), header2.hash());
    }

    #[test]
    fn test_block_header_builder_chaining() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string())
            .with_dense_embedding(&[1.0])
            .with_codes(vec![1])
            .with_signature(vec![0xff]);

        assert_eq!(header.delta_embedding.to_dense(), vec![1.0]);
        assert_eq!(header.quantized_codes, vec![1]);
        assert_eq!(header.signature, vec![0xff]);
    }

    #[test]
    fn test_transaction_delete_affected_key() {
        let tx = Transaction::Delete {
            key: "delete_key".to_string(),
        };
        assert_eq!(tx.affected_key(), "delete_key");
    }

    #[test]
    fn test_transaction_embed_affected_key() {
        let tx = Transaction::Embed {
            key: "embed_key".to_string(),
            vector: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(tx.affected_key(), "embed_key");
    }

    #[test]
    fn test_transaction_node_create_affected_key() {
        let tx = Transaction::NodeCreate {
            key: "node_key".to_string(),
            label: "Person".to_string(),
        };
        assert_eq!(tx.affected_key(), "node_key");
    }

    #[test]
    fn test_transaction_node_delete_affected_key() {
        let tx = Transaction::NodeDelete {
            key: "node_key".to_string(),
        };
        assert_eq!(tx.affected_key(), "node_key");
    }

    #[test]
    fn test_transaction_table_insert_affected_key() {
        let tx = Transaction::TableInsert {
            table: "users".to_string(),
            values: vec![1, 2, 3],
        };
        assert_eq!(tx.affected_key(), "users");
    }

    #[test]
    fn test_transaction_table_update_affected_key() {
        let tx = Transaction::TableUpdate {
            table: "users".to_string(),
            row_id: 42,
            values: vec![4, 5, 6],
        };
        assert_eq!(tx.affected_key(), "users");
    }

    #[test]
    fn test_transaction_table_delete_affected_key() {
        let tx = Transaction::TableDelete {
            table: "users".to_string(),
            row_id: 42,
        };
        assert_eq!(tx.affected_key(), "users");
    }

    #[test]
    fn test_transaction_hash_deterministic() {
        let tx = Transaction::Put {
            key: "key".to_string(),
            data: vec![1, 2, 3],
        };

        let hash1 = tx.hash();
        let hash2 = tx.hash();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_transaction_hash_different_for_different_tx() {
        let tx1 = Transaction::Put {
            key: "key1".to_string(),
            data: vec![1],
        };
        let tx2 = Transaction::Put {
            key: "key2".to_string(),
            data: vec![2],
        };

        assert_ne!(tx1.hash(), tx2.hash());
    }

    #[test]
    fn test_block_compute_tx_root_empty() {
        let genesis = Block::genesis("node1".to_string());
        assert_eq!(genesis.compute_tx_root(), [0u8; 32]);
    }

    #[test]
    fn test_block_compute_tx_root_with_transactions() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());
        let txs = vec![
            Transaction::Put {
                key: "k1".to_string(),
                data: vec![1],
            },
            Transaction::Put {
                key: "k2".to_string(),
                data: vec![2],
            },
        ];
        let block = Block::new(header, txs);

        let root = block.compute_tx_root();
        assert_ne!(root, [0u8; 32]);

        // Should be deterministic
        assert_eq!(root, block.compute_tx_root());
    }

    #[test]
    fn test_block_verify_tx_root() {
        let txs = vec![
            Transaction::Put {
                key: "k1".to_string(),
                data: vec![1],
            },
            Transaction::Delete {
                key: "k2".to_string(),
            },
        ];

        // Compute the real tx_root
        let leaves: Vec<[u8; 32]> = txs.iter().map(|tx| tx.hash()).collect();
        let tx_root = merkle_root(&leaves);

        let header = BlockHeader::new(1, [0u8; 32], tx_root, [0u8; 32], "node1".to_string());
        let block = Block::new(header, txs);

        assert!(block.verify_tx_root());
    }

    #[test]
    fn test_block_verify_tx_root_fails_on_mismatch() {
        let txs = vec![Transaction::Put {
            key: "k1".to_string(),
            data: vec![1],
        }];

        let header = BlockHeader::new(
            1,
            [0u8; 32],
            [99u8; 32], // Wrong tx_root
            [0u8; 32],
            "node1".to_string(),
        );
        let block = Block::new(header, txs);

        assert!(!block.verify_tx_root());
    }

    #[test]
    fn test_block_add_signature() {
        let mut block = Block::genesis("node1".to_string());
        let sig = ValidatorSignature {
            validator: "validator1".to_string(),
            signature: vec![1, 2, 3],
            block_hash: block.hash(),
        };

        block.add_signature(sig.clone()).unwrap();

        assert_eq!(block.signatures.len(), 1);
        assert_eq!(block.signatures[0], sig);
    }

    #[test]
    fn test_block_reject_duplicate_signer() {
        let mut block = Block::genesis("node1".to_string());
        let hash = block.hash();

        let sig = ValidatorSignature {
            validator: "validator1".to_string(),
            signature: vec![1, 2, 3],
            block_hash: hash,
        };

        // First signature should succeed
        block.add_signature(sig.clone()).unwrap();

        // Second signature from same validator should fail
        let result = block.add_signature(sig);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("duplicate signer"));
    }

    #[test]
    fn test_block_affected_keys() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());
        let txs = vec![
            Transaction::Put {
                key: "k1".to_string(),
                data: vec![1],
            },
            Transaction::Delete {
                key: "k2".to_string(),
            },
            Transaction::NodeCreate {
                key: "k3".to_string(),
                label: "Label".to_string(),
            },
        ];
        let block = Block::new(header, txs);

        let keys = block.affected_keys();
        assert_eq!(keys, vec!["k1", "k2", "k3"]);
    }

    #[test]
    fn test_verify_chain_fails_on_wrong_height() {
        let genesis = Block::genesis("node1".to_string());

        let mut next_header = BlockHeader::new(
            5, // Wrong height (should be 1)
            genesis.hash(),
            [0u8; 32],
            [0u8; 32],
            "node1".to_string(),
        );
        next_header.timestamp = genesis.header.timestamp + 1000;

        let next_block = Block::new(next_header, vec![]);

        let err = next_block.verify_chain(&genesis).unwrap_err();
        match err {
            ChainError::ValidationFailed(msg) => {
                assert!(msg.contains("height"));
            },
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_verify_chain_fails_on_timestamp_before_previous() {
        let genesis = Block::genesis("node1".to_string());

        let mut next_header =
            BlockHeader::new(1, genesis.hash(), [0u8; 32], [0u8; 32], "node1".to_string());
        // Set timestamp BEFORE the genesis block
        next_header.timestamp = genesis.header.timestamp.saturating_sub(1000);

        let next_block = Block::new(next_header, vec![]);

        let err = next_block.verify_chain(&genesis).unwrap_err();
        match err {
            ChainError::ValidationFailed(msg) => {
                assert!(msg.contains("timestamp"));
            },
            _ => panic!("Expected ValidationFailed error"),
        }
    }

    #[test]
    fn test_merkle_root_empty() {
        let leaves: Vec<[u8; 32]> = vec![];
        assert_eq!(merkle_root(&leaves), [0u8; 32]);
    }

    #[test]
    fn test_merkle_root_single_leaf() {
        let leaves = vec![[42u8; 32]];
        assert_eq!(merkle_root(&leaves), [42u8; 32]);
    }

    #[test]
    fn test_merkle_root_two_leaves() {
        let leaves = vec![[1u8; 32], [2u8; 32]];
        let root = merkle_root(&leaves);

        // Should not be either leaf
        assert_ne!(root, [1u8; 32]);
        assert_ne!(root, [2u8; 32]);
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_merkle_root_odd_number() {
        // Odd number of leaves - last one gets duplicated
        let leaves = vec![[1u8; 32], [2u8; 32], [3u8; 32]];
        let root = merkle_root(&leaves);

        // Should produce a valid root
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_merkle_root_power_of_two() {
        let leaves = vec![[1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32]];
        let root = merkle_root(&leaves);

        // Should produce a valid root
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_validator_signature_debug() {
        let sig = ValidatorSignature {
            validator: "node1".to_string(),
            signature: vec![1, 2, 3],
            block_hash: [0u8; 32],
        };

        let debug = format!("{:?}", sig);
        assert!(debug.contains("ValidatorSignature"));
        assert!(debug.contains("node1"));
    }

    #[test]
    fn test_block_header_debug() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());
        let debug = format!("{:?}", header);
        assert!(debug.contains("BlockHeader"));
        assert!(debug.contains("height"));
    }

    #[test]
    fn test_block_debug() {
        let block = Block::genesis("node1".to_string());
        let debug = format!("{:?}", block);
        assert!(debug.contains("Block"));
        assert!(debug.contains("header"));
    }

    #[test]
    fn test_transaction_debug() {
        let tx = Transaction::Put {
            key: "k".to_string(),
            data: vec![1],
        };
        let debug = format!("{:?}", tx);
        assert!(debug.contains("Put"));
    }

    #[test]
    fn test_block_clone_and_eq() {
        let block = Block::genesis("node1".to_string());
        let cloned = block.clone();

        assert_eq!(block, cloned);
    }

    #[test]
    fn test_block_header_clone_and_eq() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string())
            .with_dense_embedding(&[1.0])
            .with_codes(vec![1])
            .with_signature(vec![0xff]);

        let cloned = header.clone();
        assert_eq!(header, cloned);
    }

    #[test]
    fn test_transaction_clone_and_eq() {
        let tx = Transaction::Embed {
            key: "k".to_string(),
            vector: vec![1.0, 2.0],
        };
        let cloned = tx.clone();
        assert_eq!(tx, cloned);
    }

    #[test]
    fn test_validator_signature_clone_and_eq() {
        let sig = ValidatorSignature {
            validator: "v1".to_string(),
            signature: vec![1, 2],
            block_hash: [1u8; 32],
        };
        let cloned = sig.clone();
        assert_eq!(sig, cloned);
    }

    #[test]
    fn test_block_hash_uses_header_hash() {
        let block = Block::genesis("node1".to_string());
        assert_eq!(block.hash(), block.header.hash());
    }

    #[test]
    fn test_verify_chain_error_message_format() {
        let genesis = Block::genesis("node1".to_string());

        // Wrong prev_hash - check error contains expected and actual hashes
        let mut next_header =
            BlockHeader::new(1, [99u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());
        next_header.timestamp = genesis.header.timestamp + 1000;

        let next_block = Block::new(next_header, vec![]);

        let err = next_block.verify_chain(&genesis).unwrap_err();
        match err {
            ChainError::InvalidHash { expected, actual } => {
                // Check that we get proper hex encoding
                assert_eq!(expected.len(), 64); // 32 bytes = 64 hex chars
                assert_eq!(actual.len(), 64);
            },
            _ => panic!("Expected InvalidHash error"),
        }
    }

    #[test]
    fn test_multiple_signatures() {
        let mut block = Block::genesis("node1".to_string());
        let hash = block.hash();

        for i in 0..3 {
            let sig = ValidatorSignature {
                validator: format!("validator{i}"),
                signature: vec![i as u8],
                block_hash: hash,
            };
            block.add_signature(sig).unwrap();
        }

        assert_eq!(block.signatures.len(), 3);
    }

    // === Security Tests for Block Signature Verification ===

    #[test]
    fn test_verify_signature_valid() {
        use crate::signing::Identity;

        // Create an identity and register it
        let identity = Identity::generate();
        let node_id = identity.node_id();

        // Create a registry and register the validator
        let registry = ValidatorRegistry::new();
        registry.register(&identity);

        // Create a header with the identity's node_id as proposer
        let mut header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], node_id);

        // Sign the header
        let signing_bytes = header.signing_bytes();
        let signature = identity.sign(&signing_bytes);
        header = header.with_signature(signature);

        // Verification should succeed
        assert!(header.verify_signature(&registry).is_ok());
    }

    #[test]
    fn test_verify_signature_missing() {
        let registry = ValidatorRegistry::new();

        // Header with empty signature
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());

        let result = header.verify_signature(&registry);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("missing block signature"));
    }

    #[test]
    fn test_verify_signature_unknown_proposer() {
        let registry = ValidatorRegistry::new(); // Empty registry

        // Header with signature but unknown proposer
        let header = BlockHeader::new(
            1,
            [0u8; 32],
            [0u8; 32],
            [0u8; 32],
            "unknown_node".to_string(),
        )
        .with_signature(vec![0u8; 64]);

        let result = header.verify_signature(&registry);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unknown proposer"));
    }

    #[test]
    fn test_verify_signature_invalid_signature() {
        use crate::signing::Identity;

        let identity = Identity::generate();
        let node_id = identity.node_id();

        let registry = ValidatorRegistry::new();
        registry.register(&identity);

        // Create a header with invalid signature (wrong bytes)
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], node_id)
            .with_signature(vec![0u8; 64]); // Invalid signature

        let result = header.verify_signature(&registry);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid block signature"));
    }

    #[test]
    fn test_verify_signature_tampered_header() {
        use crate::signing::Identity;

        let identity = Identity::generate();
        let node_id = identity.node_id();

        let registry = ValidatorRegistry::new();
        registry.register(&identity);

        // Create and sign a header
        let mut header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], node_id);
        let signing_bytes = header.signing_bytes();
        let signature = identity.sign(&signing_bytes);
        header = header.with_signature(signature);

        // Tamper with the header after signing
        header.height = 999;

        // Verification should fail
        let result = header.verify_signature(&registry);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid block signature"));
    }

    #[test]
    fn test_signing_bytes_deterministic() {
        let header = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string())
            .with_dense_embedding(&[1.0, 2.0])
            .with_codes(vec![1, 2, 3]);

        let bytes1 = header.signing_bytes();
        let bytes2 = header.signing_bytes();

        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn test_signing_bytes_different_for_different_headers() {
        let header1 = BlockHeader::new(1, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());
        let header2 = BlockHeader::new(2, [0u8; 32], [0u8; 32], [0u8; 32], "node1".to_string());

        assert_ne!(header1.signing_bytes(), header2.signing_bytes());
    }

    // === storage_key() tests ===

    #[test]
    fn test_storage_key_put() {
        let tx = Transaction::Put {
            key: "mykey".into(),
            data: vec![],
        };
        assert_eq!(tx.storage_key(), "mykey");
    }

    #[test]
    fn test_storage_key_delete() {
        let tx = Transaction::Delete {
            key: "mykey".into(),
        };
        assert_eq!(tx.storage_key(), "mykey");
    }

    #[test]
    fn test_storage_key_embed() {
        let tx = Transaction::Embed {
            key: "doc1".into(),
            vector: vec![1.0],
        };
        assert_eq!(tx.storage_key(), "emb:doc1");
    }

    #[test]
    fn test_storage_key_node_create() {
        let tx = Transaction::NodeCreate {
            key: "user1".into(),
            label: "Person".into(),
        };
        assert_eq!(tx.storage_key(), "node:user1");
    }

    #[test]
    fn test_storage_key_node_delete() {
        let tx = Transaction::NodeDelete {
            key: "user1".into(),
        };
        assert_eq!(tx.storage_key(), "node:user1");
    }

    #[test]
    fn test_storage_key_edge_create() {
        let tx = Transaction::EdgeCreate {
            from: "a".into(),
            to: "b".into(),
            edge_type: "knows".into(),
        };
        assert_eq!(tx.storage_key(), "edge:a:b:knows");
    }

    #[test]
    fn test_storage_key_table_insert() {
        let tx = Transaction::TableInsert {
            table: "users".into(),
            values: vec![],
        };
        assert_eq!(tx.storage_key(), "table:users");
    }

    #[test]
    fn test_storage_key_table_update() {
        let tx = Transaction::TableUpdate {
            table: "users".into(),
            row_id: 42,
            values: vec![],
        };
        assert_eq!(tx.storage_key(), "table:users");
    }

    #[test]
    fn test_storage_key_table_delete() {
        let tx = Transaction::TableDelete {
            table: "users".into(),
            row_id: 42,
        };
        assert_eq!(tx.storage_key(), "table:users");
    }
}
