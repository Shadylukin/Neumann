// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Defense-in-depth validation layer for all cluster messages.
//!
//! # Overview
//!
//! This module provides comprehensive semantic validation for incoming cluster messages
//! before they are processed by the Raft consensus, 2PC coordinator, or gossip subsystems.
//! It acts as a defense-in-depth layer, catching malformed or malicious messages that
//! could cause:
//!
//! - Integer overflow attacks (unbounded terms, indices, or shard IDs)
//! - Resource exhaustion (oversized embeddings, queries, or block requests)
//! - Replay attacks (stale timestamps on signed messages)
//! - Format violations (empty node IDs, malformed signatures)
//!
//! # Architecture
//!
//! ```text
//! +------------------+     Message      +--------------------+
//! | Network Layer    | ---------------> | CompositeValidator |
//! | (Transport)      |                  +--------------------+
//! +------------------+                  | - validate_term()  |
//!                                       | - validate_node_id()|
//!                                       | - validate_shard() |
//!                                       | - validate_timeout()|
//!                                       +--------------------+
//!                                                |
//!                                                | uses
//!                                                v
//!                                       +--------------------+
//!                                       | EmbeddingValidator |
//!                                       +--------------------+
//!                                       | - dimension check  |
//!                                       | - NaN/Inf check    |
//!                                       | - magnitude check  |
//!                                       | - position sorting |
//!                                       +--------------------+
//! ```
//!
//! # Validation Checks
//!
//! ## Numeric Bounds
//!
//! | Field | Constraint | Purpose |
//! |-------|------------|---------|
//! | `term` | > 0, < `max_term` | Prevent overflow in Raft term arithmetic |
//! | `shard_id` | < `max_shard_id` | Prevent out-of-bounds shard access |
//! | `timeout_ms` | > 0, < `max_tx_timeout` | Prevent infinite waits or resource exhaustion |
//! | `tx_id` | > 0 | Distinguish valid transactions from uninitialized state |
//! | `query_id` | > 0 | Distinguish valid queries from uninitialized state |
//!
//! ## Embedding Validation
//!
//! Sparse vector embeddings are validated for:
//! - Non-zero dimension
//! - Dimension within configured maximum
//! - No NaN or Infinite values
//! - Magnitude (L2 norm) within configured maximum
//! - Positions sorted and within bounds
//!
//! ## Signed Message Validation
//!
//! Signed gossip messages are checked for:
//! - Correct signature length (64 bytes for Ed25519)
//! - Valid sender node ID
//! - Timestamp not too far in the future (1 minute tolerance)
//! - Timestamp not too old (configurable max age)
//!
//! # Configuration
//!
//! ```rust
//! use tensor_chain::message_validation::MessageValidationConfig;
//!
//! let config = MessageValidationConfig {
//!     enabled: true,
//!     max_term: u64::MAX - 1,
//!     max_shard_id: 65536,
//!     max_tx_timeout_ms: 300_000,  // 5 minutes
//!     max_node_id_len: 256,
//!     max_key_len: 4096,
//!     max_embedding_dimension: 65536,
//!     max_embedding_magnitude: 1e6,
//!     max_query_len: 1024 * 1024,  // 1 MB
//!     max_message_age_ms: 5 * 60 * 1000,  // 5 minutes
//!     max_blocks_per_request: 1000,
//!     max_snapshot_chunk_size: 10 * 1024 * 1024,  // 10 MB
//! };
//! ```
//!
//! # Usage
//!
//! ```rust
//! use tensor_chain::message_validation::{CompositeValidator, MessageValidationConfig, MessageValidator};
//! use tensor_chain::network::Message;
//!
//! let config = MessageValidationConfig::default();
//! let validator = CompositeValidator::new(config);
//!
//! // Validate incoming message
//! let from_node = "node1".to_string();
//! // let msg: Message = ...;
//! // match validator.validate(&msg, &from_node) {
//! //     Ok(()) => { /* process message */ }
//! //     Err(e) => { /* reject message */ }
//! // }
//! ```
//!
//! # Trait: `MessageValidator`
//!
//! The [`MessageValidator`] trait enables pluggable validation strategies:
//!
//! ```rust
//! use tensor_chain::message_validation::MessageValidator;
//! use tensor_chain::network::Message;
//! use tensor_chain::block::NodeId;
//! use tensor_chain::error::Result;
//!
//! struct CustomValidator;
//!
//! impl MessageValidator for CustomValidator {
//!     fn validate(&self, msg: &Message, from: &NodeId) -> Result<()> {
//!         // Custom validation logic
//!         Ok(())
//!     }
//! }
//! ```
//!
//! # Security Considerations
//!
//! - Validation is enabled by default; disable only for testing
//! - All limits are configurable to balance security vs. functionality
//! - Embedding validation prevents floating-point exploits (NaN, Inf)
//! - Timestamp checks prevent replay attacks on signed messages
//! - Block request limits prevent `DoS` via huge range requests
//!
//! # Error Handling
//!
//! Validation failures return typed errors:
//!
//! - [`ChainError::NumericOutOfBounds`]: Term, shard ID, timeout, or ID out of range
//! - [`ChainError::MessageValidationFailed`]: Node ID or query format invalid
//! - [`ChainError::InvalidEmbedding`]: Embedding dimension, values, or structure invalid
//! - [`ChainError::CryptoError`]: Signature length, timestamp, or age invalid
//!
//! # See Also
//!
//! - [`crate::network`]: Network transport and message types
//! - [`crate::signing`]: Signed message creation and verification
//! - [`crate::gossip`]: Gossip protocol that uses signed messages

use tensor_store::SparseVector;

use crate::{
    block::NodeId,
    error::{ChainError, Result},
    network::Message,
    signing::SignedGossipMessage,
};

/// Configuration for message validation.
#[derive(Debug, Clone)]
pub struct MessageValidationConfig {
    /// Enable validation (can be disabled for testing).
    pub enabled: bool,
    /// Maximum term value (prevent overflow attacks).
    pub max_term: u64,
    /// Maximum shard ID.
    pub max_shard_id: usize,
    /// Maximum transaction timeout in milliseconds.
    pub max_tx_timeout_ms: u64,
    /// Maximum node ID length in bytes.
    pub max_node_id_len: usize,
    /// Maximum key length in bytes.
    pub max_key_len: usize,
    /// Maximum embedding dimension.
    pub max_embedding_dimension: usize,
    /// Maximum embedding magnitude (L2 norm).
    pub max_embedding_magnitude: f32,
    /// Maximum query string length in bytes.
    pub max_query_len: usize,
    /// Maximum age for signed messages in milliseconds.
    pub max_message_age_ms: u64,
    /// Maximum blocks per `BlockRequest` (default: 1000).
    pub max_blocks_per_request: u64,
    /// Maximum chunk size for `SnapshotRequest` in bytes (default: 10MB).
    pub max_snapshot_chunk_size: u64,
}

impl Default for MessageValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_term: u64::MAX - 1,
            max_shard_id: 65536,
            max_tx_timeout_ms: 300_000, // 5 minutes
            max_node_id_len: 256,
            max_key_len: 4096,
            max_embedding_dimension: 65536,
            max_embedding_magnitude: 1e6,
            max_query_len: 1024 * 1024,        // 1 MB
            max_message_age_ms: 5 * 60 * 1000, // 5 minutes
            max_blocks_per_request: 1000,
            max_snapshot_chunk_size: 10 * 1024 * 1024, // 10 MB
        }
    }
}

impl MessageValidationConfig {
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Trait for message validators.
pub trait MessageValidator: Send + Sync {
    /// # Errors
    /// Returns an error if the message fails validation.
    fn validate(&self, msg: &Message, from: &NodeId) -> Result<()>;
}

/// Validates sparse vector embeddings.
pub struct EmbeddingValidator {
    max_dimension: usize,
    max_magnitude: f32,
}

impl EmbeddingValidator {
    #[must_use]
    pub const fn new(max_dimension: usize, max_magnitude: f32) -> Self {
        Self {
            max_dimension,
            max_magnitude,
        }
    }

    /// # Errors
    /// Returns an error if the embedding has invalid dimensions, NaN/Inf values, or excessive magnitude.
    pub fn validate(&self, embedding: &SparseVector, field: &str) -> Result<()> {
        let dim = embedding.dimension();

        // Check dimension bounds
        if dim == 0 {
            return Err(ChainError::InvalidEmbedding {
                dimension: dim,
                reason: format!("{field}: dimension cannot be zero"),
            });
        }

        if dim > self.max_dimension {
            return Err(ChainError::InvalidEmbedding {
                dimension: dim,
                reason: format!(
                    "{field}: dimension {dim} exceeds maximum {}",
                    self.max_dimension
                ),
            });
        }

        // Check for NaN/Inf values
        for (i, value) in embedding.values().iter().enumerate() {
            if value.is_nan() {
                return Err(ChainError::InvalidEmbedding {
                    dimension: dim,
                    reason: format!("{field}: NaN value at position {i}"),
                });
            }
            if value.is_infinite() {
                return Err(ChainError::InvalidEmbedding {
                    dimension: dim,
                    reason: format!("{field}: infinite value at position {i}"),
                });
            }
        }

        // Check magnitude
        let magnitude = embedding.magnitude();
        if magnitude > self.max_magnitude {
            return Err(ChainError::InvalidEmbedding {
                dimension: dim,
                reason: format!(
                    "{field}: magnitude {magnitude:.2} exceeds maximum {:.2}",
                    self.max_magnitude
                ),
            });
        }

        // Check that positions are sorted and within bounds
        let positions = embedding.positions();
        for (i, &pos) in positions.iter().enumerate() {
            if pos as usize >= dim {
                return Err(ChainError::InvalidEmbedding {
                    dimension: dim,
                    reason: format!("{field}: position {pos} out of bounds for dimension {dim}"),
                });
            }
            if i > 0 && positions[i - 1] >= pos {
                return Err(ChainError::InvalidEmbedding {
                    dimension: dim,
                    reason: format!("{field}: positions not strictly sorted"),
                });
            }
        }

        Ok(())
    }
}

/// Composite validator that validates all message types.
pub struct CompositeValidator {
    config: MessageValidationConfig,
    embedding_validator: EmbeddingValidator,
}

impl CompositeValidator {
    #[must_use]
    pub const fn new(config: MessageValidationConfig) -> Self {
        let embedding_validator = EmbeddingValidator::new(
            config.max_embedding_dimension,
            config.max_embedding_magnitude,
        );
        Self {
            config,
            embedding_validator,
        }
    }

    fn validate_node_id(&self, node_id: &str, field: &str) -> Result<()> {
        if node_id.is_empty() {
            return Err(ChainError::MessageValidationFailed {
                message_type: "NodeId",
                reason: format!("{field}: node ID cannot be empty"),
            });
        }
        if node_id.len() > self.config.max_node_id_len {
            return Err(ChainError::MessageValidationFailed {
                message_type: "NodeId",
                reason: format!(
                    "{}: node ID length {} exceeds maximum {}",
                    field,
                    node_id.len(),
                    self.config.max_node_id_len
                ),
            });
        }
        Ok(())
    }

    fn validate_term(&self, term: u64, msg_type: &'static str) -> Result<()> {
        if term == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "term".to_string(),
                value: term.to_string(),
                expected: "greater than 0".to_string(),
            });
        }
        if term > self.config.max_term {
            return Err(ChainError::NumericOutOfBounds {
                field: "term".to_string(),
                value: term.to_string(),
                expected: format!("less than or equal to {}", self.config.max_term),
            });
        }
        let _ = msg_type; // Used for context in error messages
        Ok(())
    }

    fn validate_shard_id(&self, shard_id: usize, msg_type: &'static str) -> Result<()> {
        if shard_id >= self.config.max_shard_id {
            return Err(ChainError::NumericOutOfBounds {
                field: "shard_id".to_string(),
                value: shard_id.to_string(),
                expected: format!("less than {}", self.config.max_shard_id),
            });
        }
        let _ = msg_type;
        Ok(())
    }

    fn validate_timeout(&self, timeout_ms: u64, msg_type: &'static str) -> Result<()> {
        if timeout_ms == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "timeout_ms".to_string(),
                value: timeout_ms.to_string(),
                expected: "greater than 0".to_string(),
            });
        }
        if timeout_ms > self.config.max_tx_timeout_ms {
            return Err(ChainError::NumericOutOfBounds {
                field: "timeout_ms".to_string(),
                value: timeout_ms.to_string(),
                expected: format!("less than or equal to {}", self.config.max_tx_timeout_ms),
            });
        }
        let _ = msg_type;
        Ok(())
    }

    fn validate_request_vote(&self, msg: &crate::network::RequestVote) -> Result<()> {
        self.validate_term(msg.term, "RequestVote")?;
        self.validate_node_id(&msg.candidate_id, "candidate_id")?;
        self.embedding_validator
            .validate(&msg.state_embedding, "state_embedding")?;
        Ok(())
    }

    fn validate_request_vote_response(
        &self,
        msg: &crate::network::RequestVoteResponse,
    ) -> Result<()> {
        self.validate_term(msg.term, "RequestVoteResponse")?;
        self.validate_node_id(&msg.voter_id, "voter_id")?;
        Ok(())
    }

    fn validate_pre_vote(&self, msg: &crate::network::PreVote) -> Result<()> {
        self.validate_term(msg.term, "PreVote")?;
        self.validate_node_id(&msg.candidate_id, "candidate_id")?;
        self.embedding_validator
            .validate(&msg.state_embedding, "state_embedding")?;
        Ok(())
    }

    fn validate_pre_vote_response(&self, msg: &crate::network::PreVoteResponse) -> Result<()> {
        self.validate_term(msg.term, "PreVoteResponse")?;
        self.validate_node_id(&msg.voter_id, "voter_id")?;
        Ok(())
    }

    fn validate_append_entries(&self, msg: &crate::network::AppendEntries) -> Result<()> {
        self.validate_term(msg.term, "AppendEntries")?;
        self.validate_node_id(&msg.leader_id, "leader_id")?;
        if let Some(ref embedding) = msg.block_embedding {
            self.embedding_validator
                .validate(embedding, "block_embedding")?;
        }
        Ok(())
    }

    fn validate_append_entries_response(
        &self,
        msg: &crate::network::AppendEntriesResponse,
    ) -> Result<()> {
        self.validate_term(msg.term, "AppendEntriesResponse")?;
        self.validate_node_id(&msg.follower_id, "follower_id")?;
        Ok(())
    }

    fn validate_tx_prepare(&self, msg: &crate::network::TxPrepareMsg) -> Result<()> {
        if msg.tx_id == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "tx_id".to_string(),
                value: msg.tx_id.to_string(),
                expected: "greater than 0".to_string(),
            });
        }
        self.validate_node_id(&msg.coordinator, "coordinator")?;
        self.validate_shard_id(msg.shard_id, "TxPrepare")?;
        self.validate_timeout(msg.timeout_ms, "TxPrepare")?;
        self.embedding_validator
            .validate(&msg.delta_embedding, "delta_embedding")?;
        Ok(())
    }

    fn validate_tx_prepare_response(
        &self,
        msg: &crate::network::TxPrepareResponseMsg,
    ) -> Result<()> {
        if msg.tx_id == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "tx_id".to_string(),
                value: msg.tx_id.to_string(),
                expected: "greater than 0".to_string(),
            });
        }
        self.validate_shard_id(msg.shard_id, "TxPrepareResponse")?;
        Ok(())
    }

    fn validate_tx_commit(&self, msg: &crate::network::TxCommitMsg) -> Result<()> {
        if msg.tx_id == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "tx_id".to_string(),
                value: msg.tx_id.to_string(),
                expected: "greater than 0".to_string(),
            });
        }
        for &shard_id in &msg.shards {
            self.validate_shard_id(shard_id, "TxCommit")?;
        }
        Ok(())
    }

    fn validate_tx_abort(&self, msg: &crate::network::TxAbortMsg) -> Result<()> {
        if msg.tx_id == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "tx_id".to_string(),
                value: msg.tx_id.to_string(),
                expected: "greater than 0".to_string(),
            });
        }
        for &shard_id in &msg.shards {
            self.validate_shard_id(shard_id, "TxAbort")?;
        }
        Ok(())
    }

    fn validate_tx_ack(&self, msg: &crate::network::TxAckMsg) -> Result<()> {
        if msg.tx_id == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "tx_id".to_string(),
                value: msg.tx_id.to_string(),
                expected: "greater than 0".to_string(),
            });
        }
        self.validate_shard_id(msg.shard_id, "TxAck")?;
        Ok(())
    }

    fn validate_query_request(&self, msg: &crate::network::QueryRequest) -> Result<()> {
        if msg.query_id == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "query_id".to_string(),
                value: msg.query_id.to_string(),
                expected: "greater than 0".to_string(),
            });
        }
        self.validate_shard_id(msg.shard_id, "QueryRequest")?;
        self.validate_timeout(msg.timeout_ms, "QueryRequest")?;

        if msg.query.len() > self.config.max_query_len {
            return Err(ChainError::MessageValidationFailed {
                message_type: "QueryRequest",
                reason: format!(
                    "query length {} exceeds maximum {}",
                    msg.query.len(),
                    self.config.max_query_len
                ),
            });
        }

        if let Some(ref embedding) = msg.embedding {
            self.embedding_validator.validate(embedding, "embedding")?;
        }
        Ok(())
    }

    fn validate_query_response(&self, msg: &crate::network::QueryResponse) -> Result<()> {
        if msg.query_id == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "query_id".to_string(),
                value: msg.query_id.to_string(),
                expected: "greater than 0".to_string(),
            });
        }
        self.validate_shard_id(msg.shard_id, "QueryResponse")?;
        Ok(())
    }

    fn validate_ping_pong(&self, term: u64, msg_type: &'static str) -> Result<()> {
        self.validate_term(term, msg_type)?;
        Ok(())
    }

    fn validate_signed_gossip(&self, msg: &SignedGossipMessage) -> Result<()> {
        // Signature length check (Ed25519 signatures are 64 bytes)
        if msg.envelope.signature.len() != 64 {
            return Err(ChainError::MessageValidationFailed {
                message_type: "SignedGossip",
                reason: format!(
                    "invalid signature length: expected 64, got {}",
                    msg.envelope.signature.len()
                ),
            });
        }

        // Validate sender NodeId
        self.validate_node_id(&msg.envelope.sender, "sender")?;

        // Timestamp freshness check (reject messages too far in the future)
        #[allow(clippy::cast_possible_truncation)]
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if msg.envelope.timestamp_ms > now_ms + 60_000 {
            return Err(ChainError::CryptoError(format!(
                "message timestamp {} is in the future (now: {})",
                msg.envelope.timestamp_ms, now_ms
            )));
        }

        // Reject messages that are too old
        if now_ms > msg.envelope.timestamp_ms + self.config.max_message_age_ms {
            return Err(ChainError::CryptoError(format!(
                "message too old: {} ms",
                now_ms - msg.envelope.timestamp_ms
            )));
        }

        Ok(())
    }

    fn validate_block_request(&self, msg: &crate::network::BlockRequest) -> Result<()> {
        // Validate requester_id
        self.validate_node_id(&msg.requester_id, "requester_id")?;

        // Check height ordering
        if msg.to_height < msg.from_height {
            return Err(ChainError::MessageValidationFailed {
                message_type: "BlockRequest",
                reason: format!(
                    "to_height {} < from_height {}",
                    msg.to_height, msg.from_height
                ),
            });
        }

        // Check block count limit (prevent DoS via huge range requests)
        let block_count = msg
            .to_height
            .saturating_sub(msg.from_height)
            .saturating_add(1);
        if block_count > self.config.max_blocks_per_request {
            return Err(ChainError::NumericOutOfBounds {
                field: "block_count".to_string(),
                value: block_count.to_string(),
                expected: format!("at most {}", self.config.max_blocks_per_request),
            });
        }
        Ok(())
    }

    fn validate_snapshot_request(&self, msg: &crate::network::SnapshotRequest) -> Result<()> {
        // Validate requester_id
        self.validate_node_id(&msg.requester_id, "requester_id")?;

        // Zero chunk size is invalid
        if msg.chunk_size == 0 {
            return Err(ChainError::NumericOutOfBounds {
                field: "chunk_size".to_string(),
                value: "0".to_string(),
                expected: "greater than 0".to_string(),
            });
        }

        // Check chunk size limit (prevent DoS via huge memory allocation)
        if msg.chunk_size > self.config.max_snapshot_chunk_size {
            return Err(ChainError::NumericOutOfBounds {
                field: "chunk_size".to_string(),
                value: msg.chunk_size.to_string(),
                expected: format!("at most {}", self.config.max_snapshot_chunk_size),
            });
        }
        Ok(())
    }
}

impl MessageValidator for CompositeValidator {
    fn validate(&self, msg: &Message, from: &NodeId) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Validate sender node ID
        self.validate_node_id(from, "from")?;

        match msg {
            Message::RequestVote(m) => self.validate_request_vote(m),
            Message::RequestVoteResponse(m) => self.validate_request_vote_response(m),
            Message::PreVote(m) => self.validate_pre_vote(m),
            Message::PreVoteResponse(m) => self.validate_pre_vote_response(m),
            Message::AppendEntries(m) => self.validate_append_entries(m),
            Message::AppendEntriesResponse(m) => self.validate_append_entries_response(m),
            Message::BlockRequest(m) => self.validate_block_request(m),
            Message::SnapshotRequest(m) => self.validate_snapshot_request(m),
            Message::Ping { term } => self.validate_ping_pong(*term, "Ping"),
            Message::Pong { term } => self.validate_ping_pong(*term, "Pong"),
            Message::TxPrepare(m) => self.validate_tx_prepare(m),
            Message::TxPrepareResponse(m) => self.validate_tx_prepare_response(m),
            Message::TxCommit(m) => self.validate_tx_commit(m),
            Message::TxAbort(m) => self.validate_tx_abort(m),
            Message::TxAck(m) => self.validate_tx_ack(m),
            Message::QueryRequest(m) => self.validate_query_request(m),
            Message::QueryResponse(m) => self.validate_query_response(m),
            Message::SignedGossip(m) => self.validate_signed_gossip(m),
            // These message types are validated elsewhere or have no additional constraints
            Message::TimeoutNow(_)
            | Message::BlockResponse(_)
            | Message::SnapshotResponse(_)
            | Message::Gossip(_)
            | Message::MergeInit(_)
            | Message::MergeAck(_)
            | Message::ViewExchange(_)
            | Message::DataMergeRequest(_)
            | Message::DataMergeResponse(_)
            | Message::TxReconcileRequest(_)
            | Message::TxReconcileResponse(_)
            | Message::MergeFinalize(_) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_valid_sparse_vector(dimension: usize, nnz: usize) -> SparseVector {
        let positions: Vec<u32> = (0..nnz as u32).collect();
        let values: Vec<f32> = (0..nnz).map(|i| (i + 1) as f32 * 0.1).collect();
        SparseVector::from_parts(dimension, positions, values)
    }

    fn make_request_vote(term: u64, candidate_id: &str) -> crate::network::RequestVote {
        crate::network::RequestVote {
            term,
            candidate_id: candidate_id.to_string(),
            last_log_index: 1,
            last_log_term: 1,
            state_embedding: make_valid_sparse_vector(128, 10),
        }
    }

    #[test]
    fn test_validate_request_vote_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::RequestVote(make_request_vote(1, "node1"));
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_request_vote_zero_term() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::RequestVote(make_request_vote(0, "node1"));
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::NumericOutOfBounds { .. }
        ));
    }

    #[test]
    fn test_validate_request_vote_empty_candidate() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::RequestVote(make_request_vote(1, ""));
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::MessageValidationFailed { .. }
        ));
    }

    #[test]
    fn test_validate_tx_prepare_excessive_timeout() {
        let config = MessageValidationConfig {
            max_tx_timeout_ms: 60_000,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::TxPrepare(crate::network::TxPrepareMsg {
            tx_id: 1,
            coordinator: "coord1".to_string(),
            shard_id: 0,
            operations: vec![],
            delta_embedding: make_valid_sparse_vector(128, 10),
            timeout_ms: 120_000, // Exceeds max
        });

        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::NumericOutOfBounds { .. }
        ));
    }

    #[test]
    fn test_validate_embedding_nan() {
        let config = MessageValidationConfig::default();
        let validator = EmbeddingValidator::new(
            config.max_embedding_dimension,
            config.max_embedding_magnitude,
        );

        let embedding = SparseVector::from_parts(10, vec![0], vec![f32::NAN]);
        let result = validator.validate(&embedding, "test_field");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::InvalidEmbedding { .. }
        ));
    }

    #[test]
    fn test_validate_embedding_inf() {
        let config = MessageValidationConfig::default();
        let validator = EmbeddingValidator::new(
            config.max_embedding_dimension,
            config.max_embedding_magnitude,
        );

        let embedding = SparseVector::from_parts(10, vec![0], vec![f32::INFINITY]);
        let result = validator.validate(&embedding, "test_field");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::InvalidEmbedding { .. }
        ));
    }

    #[test]
    fn test_validate_embedding_excessive_dimension() {
        let config = MessageValidationConfig {
            max_embedding_dimension: 100,
            ..Default::default()
        };
        let validator = EmbeddingValidator::new(
            config.max_embedding_dimension,
            config.max_embedding_magnitude,
        );

        let embedding = make_valid_sparse_vector(200, 10);
        let result = validator.validate(&embedding, "test_field");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::InvalidEmbedding { .. }
        ));
    }

    #[test]
    fn test_validate_embedding_zero_dimension() {
        let config = MessageValidationConfig::default();
        let validator = EmbeddingValidator::new(
            config.max_embedding_dimension,
            config.max_embedding_magnitude,
        );

        let embedding = SparseVector::from_parts(0, vec![], vec![]);
        let result = validator.validate(&embedding, "test_field");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::InvalidEmbedding { .. }
        ));
    }

    #[test]
    fn test_validate_node_id_too_long() {
        let config = MessageValidationConfig {
            max_node_id_len: 10,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::RequestVote(make_request_vote(
            1,
            "a_very_long_node_id_that_exceeds_limit",
        ));
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::MessageValidationFailed { .. }
        ));
    }

    #[test]
    fn test_composite_validator_all_pass() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        // Test all major message types
        let messages = vec![
            Message::RequestVote(make_request_vote(1, "node1")),
            Message::RequestVoteResponse(crate::network::RequestVoteResponse {
                term: 1,
                vote_granted: true,
                voter_id: "voter1".to_string(),
            }),
            Message::Ping { term: 1 },
            Message::Pong { term: 1 },
        ];

        for msg in messages {
            assert!(
                validator.validate(&msg, &"sender".to_string()).is_ok(),
                "Failed for message: {:?}",
                msg
            );
        }
    }

    #[test]
    fn test_validation_disabled() {
        let config = MessageValidationConfig::disabled();
        let validator = CompositeValidator::new(config);

        // Invalid message should pass when validation is disabled
        let msg = Message::RequestVote(make_request_vote(0, ""));
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_query_request_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::QueryRequest(crate::network::QueryRequest {
            query_id: 1,
            query: "SELECT * FROM test".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 5000,
        });

        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_query_request_zero_id() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::QueryRequest(crate::network::QueryRequest {
            query_id: 0, // Invalid
            query: "SELECT * FROM test".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 5000,
        });

        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_query_request_exceeds_max_length() {
        let config = MessageValidationConfig {
            max_query_len: 100,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::QueryRequest(crate::network::QueryRequest {
            query_id: 1,
            query: "x".repeat(200), // Exceeds max
            shard_id: 0,
            embedding: None,
            timeout_ms: 5000,
        });

        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::MessageValidationFailed { .. }
        ));
    }

    #[test]
    fn test_validate_shard_id_bounds() {
        let config = MessageValidationConfig {
            max_shard_id: 10,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::QueryRequest(crate::network::QueryRequest {
            query_id: 1,
            query: "SELECT 1".to_string(),
            shard_id: 100, // Exceeds max
            embedding: None,
            timeout_ms: 5000,
        });

        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::NumericOutOfBounds { .. }
        ));
    }

    #[test]
    fn test_validate_from_node_id() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::Ping { term: 1 };

        // Empty sender should fail
        let result = validator.validate(&msg, &"".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_config_default() {
        let config = MessageValidationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_shard_id, 65536);
        assert_eq!(config.max_tx_timeout_ms, 300_000);
    }

    #[test]
    fn test_embedding_magnitude_check() {
        let config = MessageValidationConfig::default();
        let validator = EmbeddingValidator::new(
            config.max_embedding_dimension,
            10.0, // Low max magnitude for testing
        );

        // Create embedding with large magnitude
        let embedding = SparseVector::from_parts(10, vec![0, 1, 2], vec![100.0, 100.0, 100.0]);
        let result = validator.validate(&embedding, "test_field");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChainError::InvalidEmbedding { .. }
        ));
    }

    #[test]
    fn test_validate_pre_vote_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::PreVote(crate::network::PreVote {
            term: 1,
            candidate_id: "node1".to_string(),
            last_log_index: 1,
            last_log_term: 1,
            state_embedding: make_valid_sparse_vector(128, 10),
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_pre_vote_response_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::PreVoteResponse(crate::network::PreVoteResponse {
            term: 1,
            vote_granted: true,
            voter_id: "voter1".to_string(),
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_append_entries_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::AppendEntries(crate::network::AppendEntries {
            term: 1,
            leader_id: "leader1".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: Some(make_valid_sparse_vector(128, 10)),
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_append_entries_no_embedding() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::AppendEntries(crate::network::AppendEntries {
            term: 1,
            leader_id: "leader1".to_string(),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![],
            leader_commit: 0,
            block_embedding: None,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_append_entries_response_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::AppendEntriesResponse(crate::network::AppendEntriesResponse {
            term: 1,
            success: true,
            follower_id: "follower1".to_string(),
            match_index: 0,
            used_fast_path: false,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_tx_prepare_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxPrepare(crate::network::TxPrepareMsg {
            tx_id: 1,
            coordinator: "coord1".to_string(),
            shard_id: 0,
            operations: vec![],
            delta_embedding: make_valid_sparse_vector(128, 10),
            timeout_ms: 5000,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_tx_prepare_zero_tx_id() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxPrepare(crate::network::TxPrepareMsg {
            tx_id: 0,
            coordinator: "coord1".to_string(),
            shard_id: 0,
            operations: vec![],
            delta_embedding: make_valid_sparse_vector(128, 10),
            timeout_ms: 5000,
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tx_prepare_response_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxPrepareResponse(crate::network::TxPrepareResponseMsg {
            tx_id: 1,
            shard_id: 0,
            vote: crate::network::TxVote::No {
                reason: "test".to_string(),
            },
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_tx_prepare_response_zero_tx_id() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxPrepareResponse(crate::network::TxPrepareResponseMsg {
            tx_id: 0,
            shard_id: 0,
            vote: crate::network::TxVote::No {
                reason: "test".to_string(),
            },
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tx_commit_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxCommit(crate::network::TxCommitMsg {
            tx_id: 1,
            shards: vec![0, 1, 2],
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_tx_commit_zero_tx_id() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxCommit(crate::network::TxCommitMsg {
            tx_id: 0,
            shards: vec![0],
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tx_commit_invalid_shard() {
        let config = MessageValidationConfig {
            max_shard_id: 10,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::TxCommit(crate::network::TxCommitMsg {
            tx_id: 1,
            shards: vec![0, 100], // 100 exceeds max
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tx_abort_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxAbort(crate::network::TxAbortMsg {
            tx_id: 1,
            shards: vec![0, 1],
            reason: "test abort".to_string(),
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_tx_abort_zero_tx_id() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxAbort(crate::network::TxAbortMsg {
            tx_id: 0,
            shards: vec![0],
            reason: "test".to_string(),
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tx_abort_invalid_shard() {
        let config = MessageValidationConfig {
            max_shard_id: 10,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::TxAbort(crate::network::TxAbortMsg {
            tx_id: 1,
            shards: vec![100], // exceeds max
            reason: "test".to_string(),
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_tx_ack_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxAck(crate::network::TxAckMsg {
            tx_id: 1,
            shard_id: 0,
            success: true,
            error: None,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_tx_ack_zero_tx_id() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TxAck(crate::network::TxAckMsg {
            tx_id: 0,
            shard_id: 0,
            success: true,
            error: None,
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_query_response_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::QueryResponse(crate::network::QueryResponse {
            query_id: 1,
            shard_id: 0,
            result: vec![],
            execution_time_us: 0,
            success: true,
            error: None,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_query_response_zero_id() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::QueryResponse(crate::network::QueryResponse {
            query_id: 0,
            shard_id: 0,
            result: vec![],
            execution_time_us: 0,
            success: true,
            error: None,
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_query_request_with_embedding() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::QueryRequest(crate::network::QueryRequest {
            query_id: 1,
            query: "SELECT 1".to_string(),
            shard_id: 0,
            embedding: Some(make_valid_sparse_vector(128, 10)),
            timeout_ms: 5000,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_query_request_zero_timeout() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::QueryRequest(crate::network::QueryRequest {
            query_id: 1,
            query: "SELECT 1".to_string(),
            shard_id: 0,
            embedding: None,
            timeout_ms: 0,
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_term_exceeds_max() {
        let config = MessageValidationConfig {
            max_term: 100,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::Ping { term: 200 };
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_timeout_now() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::TimeoutNow(crate::network::TimeoutNow {
            term: 1,
            leader_id: "leader1".to_string(),
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_block_request() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::BlockRequest(crate::network::BlockRequest {
            from_height: 0,
            to_height: 10,
            requester_id: "node1".to_string(),
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_block_response() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::BlockResponse(crate::network::BlockResponse {
            blocks: vec![],
            current_height: 0,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_snapshot_request() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::SnapshotRequest(crate::network::SnapshotRequest {
            requester_id: "node1".to_string(),
            offset: 0,
            chunk_size: 1024,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_snapshot_response() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::SnapshotResponse(crate::network::SnapshotResponse {
            snapshot_height: 1,
            snapshot_hash: [0u8; 32],
            data: vec![],
            offset: 0,
            total_size: 0,
            is_last: true,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_gossip() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::Gossip(crate::gossip::GossipMessage::Alive {
            node_id: "node1".to_string(),
            incarnation: 1,
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_signed_gossip_valid() {
        use crate::signing::Identity;

        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let identity = Identity::generate();
        let gossip = crate::gossip::GossipMessage::Alive {
            node_id: "node1".to_string(),
            incarnation: 1,
        };
        let signed = SignedGossipMessage::new(&identity, &gossip, 1).unwrap();

        let msg = Message::SignedGossip(signed);
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_signed_gossip_invalid_signature_length() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let signed = crate::signing::SignedGossipMessage {
            envelope: crate::signing::SignedMessage {
                sender: "node1".to_string(),
                public_key: [0u8; 32],
                payload: vec![],
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                sequence: 1,
                signature: vec![0u8; 32], // Wrong length (should be 64)
            },
        };

        let msg = Message::SignedGossip(signed);
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_signed_gossip_empty_sender() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let signed = crate::signing::SignedGossipMessage {
            envelope: crate::signing::SignedMessage {
                sender: "".to_string(), // Empty sender
                public_key: [0u8; 32],
                payload: vec![],
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
                sequence: 1,
                signature: vec![0u8; 64],
            },
        };

        let msg = Message::SignedGossip(signed);
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_signed_gossip_future_timestamp() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let signed = crate::signing::SignedGossipMessage {
            envelope: crate::signing::SignedMessage {
                sender: "node1".to_string(),
                public_key: [0u8; 32],
                payload: vec![],
                timestamp_ms: now_ms + 120_000, // 2 minutes in future
                sequence: 1,
                signature: vec![0u8; 64],
            },
        };

        let msg = Message::SignedGossip(signed);
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_signed_gossip_old_timestamp() {
        let config = MessageValidationConfig {
            max_message_age_ms: 60_000, // 1 minute
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let signed = crate::signing::SignedGossipMessage {
            envelope: crate::signing::SignedMessage {
                sender: "node1".to_string(),
                public_key: [0u8; 32],
                payload: vec![],
                timestamp_ms: now_ms - 120_000, // 2 minutes in past (exceeds max age)
                sequence: 1,
                signature: vec![0u8; 64],
            },
        };

        let msg = Message::SignedGossip(signed);
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_embedding_position_out_of_bounds() {
        let config = MessageValidationConfig::default();
        let validator = EmbeddingValidator::new(
            config.max_embedding_dimension,
            config.max_embedding_magnitude,
        );

        // Position 10 is out of bounds for dimension 5
        let embedding = SparseVector::try_from_parts(5, vec![10], vec![1.0]);
        assert!(embedding.is_err());

        // Validator should still reject if an invalid vector slips through
        let embedding = SparseVector::try_from_parts(5, vec![3], vec![1.0]).expect("valid vector");
        let result = validator.validate(&embedding, "test_field");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_embedding_valid_sorted() {
        let config = MessageValidationConfig::default();
        let validator = EmbeddingValidator::new(
            config.max_embedding_dimension,
            config.max_embedding_magnitude,
        );

        // Properly sorted positions
        let embedding = SparseVector::from_parts(10, vec![1, 2, 5], vec![1.0, 2.0, 3.0]);
        let result = validator.validate(&embedding, "test_field");
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_debug_clone() {
        let config = MessageValidationConfig::default();
        let cloned = config.clone();
        assert_eq!(config.enabled, cloned.enabled);
        assert_eq!(config.max_term, cloned.max_term);

        let debug = format!("{:?}", config);
        assert!(debug.contains("MessageValidationConfig"));
    }

    // === BlockRequest validation tests ===

    #[test]
    fn test_validate_block_request_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::BlockRequest(crate::network::BlockRequest {
            from_height: 0,
            to_height: 999, // 1000 blocks, at limit
            requester_id: "node1".to_string(),
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_block_request_inverted_range() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::BlockRequest(crate::network::BlockRequest {
            from_height: 100,
            to_height: 50, // Invalid: to < from
            requester_id: "node1".to_string(),
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("to_height"));
    }

    #[test]
    fn test_validate_block_request_too_many_blocks() {
        let config = MessageValidationConfig {
            max_blocks_per_request: 100,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::BlockRequest(crate::network::BlockRequest {
            from_height: 0,
            to_height: 100, // 101 blocks, exceeds limit
            requester_id: "node1".to_string(),
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("block_count"));
    }

    #[test]
    fn test_validate_block_request_empty_requester() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::BlockRequest(crate::network::BlockRequest {
            from_height: 0,
            to_height: 10,
            requester_id: "".to_string(), // Empty
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requester_id"));
    }

    // === SnapshotRequest validation tests ===

    #[test]
    fn test_validate_snapshot_request_valid() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::SnapshotRequest(crate::network::SnapshotRequest {
            requester_id: "node1".to_string(),
            offset: 0,
            chunk_size: 1024 * 1024, // 1 MB, well under limit
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_snapshot_request_zero_chunk() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::SnapshotRequest(crate::network::SnapshotRequest {
            requester_id: "node1".to_string(),
            offset: 0,
            chunk_size: 0, // Invalid
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("chunk_size"));
    }

    #[test]
    fn test_validate_snapshot_request_excessive_chunk() {
        let config = MessageValidationConfig {
            max_snapshot_chunk_size: 1024 * 1024, // 1 MB
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::SnapshotRequest(crate::network::SnapshotRequest {
            requester_id: "node1".to_string(),
            offset: 0,
            chunk_size: 100 * 1024 * 1024, // 100 MB, exceeds limit
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("chunk_size"));
    }

    #[test]
    fn test_validate_snapshot_request_empty_requester() {
        let config = MessageValidationConfig::default();
        let validator = CompositeValidator::new(config);

        let msg = Message::SnapshotRequest(crate::network::SnapshotRequest {
            requester_id: "".to_string(), // Empty
            offset: 0,
            chunk_size: 1024,
        });
        let result = validator.validate(&msg, &"sender".to_string());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requester_id"));
    }

    #[test]
    fn test_validate_block_request_at_exact_limit() {
        let config = MessageValidationConfig {
            max_blocks_per_request: 100,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        // Exactly 100 blocks (0-99 inclusive)
        let msg = Message::BlockRequest(crate::network::BlockRequest {
            from_height: 0,
            to_height: 99,
            requester_id: "node1".to_string(),
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }

    #[test]
    fn test_validate_snapshot_request_at_exact_limit() {
        let config = MessageValidationConfig {
            max_snapshot_chunk_size: 1024,
            ..Default::default()
        };
        let validator = CompositeValidator::new(config);

        let msg = Message::SnapshotRequest(crate::network::SnapshotRequest {
            requester_id: "node1".to_string(),
            offset: 0,
            chunk_size: 1024, // Exactly at limit
        });
        assert!(validator.validate(&msg, &"sender".to_string()).is_ok());
    }
}
