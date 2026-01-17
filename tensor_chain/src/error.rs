//! Error types for tensor_chain.

use thiserror::Error;

/// Result type for tensor_chain operations.
pub type Result<T> = std::result::Result<T, ChainError>;

/// Errors that can occur in tensor_chain operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ChainError {
    /// Block validation failed.
    #[error("block validation failed: {0}")]
    ValidationFailed(String),

    /// Invalid block hash.
    #[error("invalid block hash: expected {expected}, got {actual}")]
    InvalidHash { expected: String, actual: String },

    /// Block not found.
    #[error("block not found at height {0}")]
    BlockNotFound(u64),

    /// Transaction failed.
    #[error("transaction failed: {0}")]
    TransactionFailed(String),

    /// Workspace isolation error.
    #[error("workspace error: {0}")]
    WorkspaceError(String),

    /// Checkpoint error.
    #[error("checkpoint error: {0}")]
    CheckpointError(String),

    /// Codebook validation error.
    #[error("codebook validation failed: {0}")]
    CodebookError(String),

    /// State transition not valid.
    #[error("invalid state transition: {0}")]
    InvalidTransition(String),

    /// Conflict detected during merge.
    #[error("semantic conflict detected: similarity {similarity:.3}")]
    ConflictDetected { similarity: f32 },

    /// Merge failed.
    #[error("merge failed: {0}")]
    MergeFailed(String),

    /// Consensus error.
    #[error("consensus error: {0}")]
    ConsensusError(String),

    /// Network error.
    #[error("network error: {0}")]
    NetworkError(String),

    /// Serialization error.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// Storage error.
    #[error("storage error: {0}")]
    StorageError(String),

    /// Graph engine error.
    #[error("graph error: {0}")]
    GraphError(String),

    /// Crypto error.
    #[error("crypto error: {0}")]
    CryptoError(String),

    /// Chain is empty.
    #[error("chain is empty")]
    EmptyChain,

    /// Invalid chain state.
    #[error("invalid chain state: {0}")]
    InvalidState(String),

    /// Replication queue is full (backpressure).
    #[error("replication queue full: {pending_count} pending updates")]
    QueueFull { pending_count: usize },

    /// Snapshot error.
    #[error("snapshot error: {0}")]
    SnapshotError(String),

    /// Not the leader node (membership changes require leader).
    #[error("not the leader")]
    NotLeader,

    /// Membership change already in progress.
    #[error("membership change in progress at index {0}")]
    MembershipChangeInProgress(u64),

    /// Node not found in membership.
    #[error("node not found: {0}")]
    NodeNotFound(String),

    /// Deadlock detected in distributed transaction.
    #[error("deadlock detected: cycle {cycle:?}, victim tx {victim}")]
    DeadlockDetected {
        /// Transaction IDs in the deadlock cycle.
        cycle: Vec<u64>,
        /// Selected victim transaction ID.
        victim: u64,
    },

    /// Handler operation timed out.
    #[error("handler timeout: {operation} exceeded {timeout_ms}ms")]
    HandlerTimeout {
        /// The operation that timed out.
        operation: String,
        /// The timeout in milliseconds.
        timeout_ms: u64,
    },

    /// Message validation failed.
    #[error("message validation failed: {message_type}: {reason}")]
    MessageValidationFailed {
        /// Type of message that failed validation.
        message_type: &'static str,
        /// Reason for validation failure.
        reason: String,
    },

    /// Invalid embedding in message.
    #[error("invalid embedding (dimension {dimension}): {reason}")]
    InvalidEmbedding {
        /// Dimension of the invalid embedding.
        dimension: usize,
        /// Reason for invalidity.
        reason: String,
    },

    /// Numeric field out of bounds.
    #[error("numeric out of bounds: {field} = {value} (expected {expected})")]
    NumericOutOfBounds {
        /// Field name.
        field: String,
        /// Actual value.
        value: String,
        /// Expected bounds.
        expected: String,
    },

    /// Delta update integrity check failed.
    #[error("update integrity failed for key '{key}' at index {index}")]
    UpdateIntegrityFailed {
        /// Key of the update.
        key: String,
        /// Index in the batch.
        index: usize,
    },

    /// Delta batch integrity check failed.
    #[error("batch integrity failed: sequence {sequence} from {source_node}")]
    BatchIntegrityFailed {
        /// Batch sequence number.
        sequence: u64,
        /// Source node.
        source_node: String,
    },
}

impl From<bincode::Error> for ChainError {
    fn from(err: bincode::Error) -> Self {
        ChainError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for ChainError {
    fn from(err: std::io::Error) -> Self {
        ChainError::StorageError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_failed() {
        let err = ChainError::ValidationFailed("test reason".to_string());
        assert!(err.to_string().contains("block validation failed"));
        assert!(err.to_string().contains("test reason"));
    }

    #[test]
    fn test_invalid_hash() {
        let err = ChainError::InvalidHash {
            expected: "abc123".to_string(),
            actual: "def456".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("invalid block hash"));
        assert!(msg.contains("abc123"));
        assert!(msg.contains("def456"));
    }

    #[test]
    fn test_block_not_found() {
        let err = ChainError::BlockNotFound(42);
        assert!(err.to_string().contains("block not found at height 42"));
    }

    #[test]
    fn test_transaction_failed() {
        let err = ChainError::TransactionFailed("commit error".to_string());
        assert!(err.to_string().contains("transaction failed"));
        assert!(err.to_string().contains("commit error"));
    }

    #[test]
    fn test_workspace_error() {
        let err = ChainError::WorkspaceError("isolation violated".to_string());
        assert!(err.to_string().contains("workspace error"));
    }

    #[test]
    fn test_checkpoint_error() {
        let err = ChainError::CheckpointError("snapshot failed".to_string());
        assert!(err.to_string().contains("checkpoint error"));
    }

    #[test]
    fn test_codebook_error() {
        let err = ChainError::CodebookError("invalid centroid".to_string());
        assert!(err.to_string().contains("codebook validation failed"));
    }

    #[test]
    fn test_invalid_transition() {
        let err = ChainError::InvalidTransition("state drift".to_string());
        assert!(err.to_string().contains("invalid state transition"));
    }

    #[test]
    fn test_conflict_detected() {
        let err = ChainError::ConflictDetected { similarity: 0.95 };
        assert!(err.to_string().contains("semantic conflict detected"));
        assert!(err.to_string().contains("0.950"));
    }

    #[test]
    fn test_merge_failed() {
        let err = ChainError::MergeFailed("non-orthogonal".to_string());
        assert!(err.to_string().contains("merge failed"));
    }

    #[test]
    fn test_consensus_error() {
        let err = ChainError::ConsensusError("quorum not reached".to_string());
        assert!(err.to_string().contains("consensus error"));
    }

    #[test]
    fn test_network_error() {
        let err = ChainError::NetworkError("connection refused".to_string());
        assert!(err.to_string().contains("network error"));
    }

    #[test]
    fn test_serialization_error() {
        let err = ChainError::SerializationError("invalid format".to_string());
        assert!(err.to_string().contains("serialization error"));
    }

    #[test]
    fn test_storage_error() {
        let err = ChainError::StorageError("disk full".to_string());
        assert!(err.to_string().contains("storage error"));
    }

    #[test]
    fn test_graph_error() {
        let err = ChainError::GraphError("node not found".to_string());
        assert!(err.to_string().contains("graph error"));
    }

    #[test]
    fn test_crypto_error() {
        let err = ChainError::CryptoError("invalid signature".to_string());
        assert!(err.to_string().contains("crypto error"));
    }

    #[test]
    fn test_empty_chain() {
        let err = ChainError::EmptyChain;
        assert!(err.to_string().contains("chain is empty"));
    }

    #[test]
    fn test_invalid_state() {
        let err = ChainError::InvalidState("corrupted".to_string());
        assert!(err.to_string().contains("invalid chain state"));
    }

    #[test]
    fn test_from_bincode_error() {
        let bincode_err = bincode::serialize(&"test")
            .and_then(|_| bincode::deserialize::<u64>(b"invalid"))
            .unwrap_err();
        let chain_err: ChainError = bincode_err.into();
        assert!(matches!(chain_err, ChainError::SerializationError(_)));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let chain_err: ChainError = io_err.into();
        assert!(matches!(chain_err, ChainError::StorageError(_)));
        assert!(chain_err.to_string().contains("file not found"));
    }

    #[test]
    fn test_error_debug() {
        let err = ChainError::BlockNotFound(100);
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("BlockNotFound"));
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_queue_full() {
        let err = ChainError::QueueFull {
            pending_count: 1000,
        };
        let msg = err.to_string();
        assert!(msg.contains("replication queue full"));
        assert!(msg.contains("1000"));
    }

    #[test]
    fn test_snapshot_error() {
        let err = ChainError::SnapshotError("compaction failed".to_string());
        let msg = err.to_string();
        assert!(msg.contains("snapshot error"));
        assert!(msg.contains("compaction failed"));
    }

    #[test]
    fn test_not_leader() {
        let err = ChainError::NotLeader;
        assert!(err.to_string().contains("not the leader"));
    }

    #[test]
    fn test_membership_change_in_progress() {
        let err = ChainError::MembershipChangeInProgress(42);
        let msg = err.to_string();
        assert!(msg.contains("membership change in progress"));
        assert!(msg.contains("42"));
    }

    #[test]
    fn test_node_not_found() {
        let err = ChainError::NodeNotFound("node-1".to_string());
        let msg = err.to_string();
        assert!(msg.contains("node not found"));
        assert!(msg.contains("node-1"));
    }

    #[test]
    fn test_deadlock_detected() {
        let err = ChainError::DeadlockDetected {
            cycle: vec![1, 2, 3],
            victim: 2,
        };
        let msg = err.to_string();
        assert!(msg.contains("deadlock detected"));
        assert!(msg.contains("victim tx 2"));
    }

    #[test]
    fn test_handler_timeout() {
        let err = ChainError::HandlerTimeout {
            operation: "query execution".to_string(),
            timeout_ms: 5000,
        };
        let msg = err.to_string();
        assert!(msg.contains("handler timeout"));
        assert!(msg.contains("query execution"));
        assert!(msg.contains("5000ms"));
    }
}
