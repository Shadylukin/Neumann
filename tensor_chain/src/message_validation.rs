//! Message validation layer for semantic validation of all message types.
//!
//! Provides bounds checking, format validation, and embedding validation
//! for incoming cluster messages before they are processed.

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
        }
    }
}

impl MessageValidationConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

/// Trait for message validators.
pub trait MessageValidator: Send + Sync {
    fn validate(&self, msg: &Message, from: &NodeId) -> Result<()>;
}

/// Validates sparse vector embeddings.
pub struct EmbeddingValidator {
    max_dimension: usize,
    max_magnitude: f32,
}

impl EmbeddingValidator {
    pub fn new(max_dimension: usize, max_magnitude: f32) -> Self {
        Self {
            max_dimension,
            max_magnitude,
        }
    }

    pub fn validate(&self, embedding: &SparseVector, field: &str) -> Result<()> {
        let dim = embedding.dimension();

        // Check dimension bounds
        if dim == 0 {
            return Err(ChainError::InvalidEmbedding {
                dimension: dim,
                reason: format!("{}: dimension cannot be zero", field),
            });
        }

        if dim > self.max_dimension {
            return Err(ChainError::InvalidEmbedding {
                dimension: dim,
                reason: format!(
                    "{}: dimension {} exceeds maximum {}",
                    field, dim, self.max_dimension
                ),
            });
        }

        // Check for NaN/Inf values
        for (i, value) in embedding.values().iter().enumerate() {
            if value.is_nan() {
                return Err(ChainError::InvalidEmbedding {
                    dimension: dim,
                    reason: format!("{}: NaN value at position {}", field, i),
                });
            }
            if value.is_infinite() {
                return Err(ChainError::InvalidEmbedding {
                    dimension: dim,
                    reason: format!("{}: infinite value at position {}", field, i),
                });
            }
        }

        // Check magnitude
        let magnitude = embedding.magnitude();
        if magnitude > self.max_magnitude {
            return Err(ChainError::InvalidEmbedding {
                dimension: dim,
                reason: format!(
                    "{}: magnitude {:.2} exceeds maximum {:.2}",
                    field, magnitude, self.max_magnitude
                ),
            });
        }

        // Check that positions are sorted and within bounds
        let positions = embedding.positions();
        for (i, &pos) in positions.iter().enumerate() {
            if pos as usize >= dim {
                return Err(ChainError::InvalidEmbedding {
                    dimension: dim,
                    reason: format!(
                        "{}: position {} out of bounds for dimension {}",
                        field, pos, dim
                    ),
                });
            }
            if i > 0 && positions[i - 1] >= pos {
                return Err(ChainError::InvalidEmbedding {
                    dimension: dim,
                    reason: format!("{}: positions not strictly sorted", field),
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
    pub fn new(config: MessageValidationConfig) -> Self {
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
                reason: format!("{}: node ID cannot be empty", field),
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
            Message::TimeoutNow(_) => Ok(()),
            Message::BlockRequest(_) => Ok(()),
            Message::BlockResponse(_) => Ok(()),
            Message::SnapshotRequest(_) => Ok(()),
            Message::SnapshotResponse(_) => Ok(()),
            Message::Ping { term } => self.validate_ping_pong(*term, "Ping"),
            Message::Pong { term } => self.validate_ping_pong(*term, "Pong"),
            Message::TxPrepare(m) => self.validate_tx_prepare(m),
            Message::TxPrepareResponse(m) => self.validate_tx_prepare_response(m),
            Message::TxCommit(m) => self.validate_tx_commit(m),
            Message::TxAbort(m) => self.validate_tx_abort(m),
            Message::TxAck(m) => self.validate_tx_ack(m),
            Message::QueryRequest(m) => self.validate_query_request(m),
            Message::QueryResponse(m) => self.validate_query_response(m),
            Message::Gossip(_) => Ok(()), // Gossip has its own validation
            Message::SignedGossip(m) => self.validate_signed_gossip(m),
            Message::MergeInit(_) => Ok(()),
            Message::MergeAck(_) => Ok(()),
            Message::ViewExchange(_) => Ok(()),
            Message::DataMergeRequest(_) => Ok(()),
            Message::DataMergeResponse(_) => Ok(()),
            Message::TxReconcileRequest(_) => Ok(()),
            Message::TxReconcileResponse(_) => Ok(()),
            Message::MergeFinalize(_) => Ok(()),
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
}
