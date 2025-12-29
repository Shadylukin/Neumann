//! Error types for tensor_chain.

use thiserror::Error;

/// Result type for tensor_chain operations.
pub type Result<T> = std::result::Result<T, ChainError>;

/// Errors that can occur in tensor_chain operations.
#[derive(Debug, Error)]
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
