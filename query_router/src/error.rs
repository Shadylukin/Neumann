// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error type for the query router and `From` conversions from every engine
//! crate's error type.

use graph_engine::GraphError;
use relational_engine::RelationalError;
use serde::{Deserialize, Serialize};
use tensor_blob::BlobError;
use tensor_cache::CacheError;
use tensor_chain::ChainError;
use tensor_checkpoint::CheckpointError;
use tensor_unified::UnifiedError;
use tensor_vault::VaultError;
use vector_engine::VectorError;

use crate::cursor::CursorError;

/// Error types for query routing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouterError {
    /// Failed to parse the command.
    ParseError(String),
    /// Unknown command or keyword.
    UnknownCommand(String),
    /// Error from relational engine.
    RelationalError(String),
    /// Error from graph engine.
    GraphError(String),
    /// Error from vector engine.
    VectorError(String),
    /// Error from vault.
    VaultError(String),
    /// Error from cache.
    CacheError(String),
    /// Error from blob storage.
    BlobError(String),
    /// Error from checkpoint system.
    CheckpointError(String),
    /// Error from chain system.
    ChainError(String),
    /// Invalid argument provided.
    InvalidArgument(String),
    /// Missing required argument.
    MissingArgument(String),
    /// Type mismatch in query.
    TypeMismatch(String),
    /// Authentication required for vault operations.
    AuthenticationRequired,
    /// Entity or resource not found.
    NotFound(String),
    /// Cursor operation error.
    CursorError(String),
}

impl std::fmt::Display for RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "Parse error: {msg}"),
            Self::UnknownCommand(cmd) => write!(f, "Unknown command: {cmd}"),
            Self::RelationalError(msg) => write!(f, "Relational error: {msg}"),
            Self::GraphError(msg) => write!(f, "Graph error: {msg}"),
            Self::VectorError(msg) => write!(f, "Vector error: {msg}"),
            Self::VaultError(msg) => write!(f, "Vault error: {msg}"),
            Self::CacheError(msg) => write!(f, "Cache error: {msg}"),
            Self::BlobError(msg) => write!(f, "Blob error: {msg}"),
            Self::CheckpointError(msg) => write!(f, "Checkpoint error: {msg}"),
            Self::ChainError(msg) => write!(f, "Chain error: {msg}"),
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {msg}"),
            Self::TypeMismatch(msg) => write!(f, "Type mismatch: {msg}"),
            Self::MissingArgument(msg) => write!(f, "Missing argument: {msg}"),
            Self::AuthenticationRequired => {
                write!(
                    f,
                    "Authentication required: call SET IDENTITY before vault operations"
                )
            },
            Self::NotFound(msg) => write!(f, "Not found: {msg}"),
            Self::CursorError(msg) => write!(f, "Cursor error: {msg}"),
        }
    }
}

impl std::error::Error for RouterError {}

impl From<CursorError> for RouterError {
    fn from(e: CursorError) -> Self {
        Self::CursorError(e.to_string())
    }
}

impl From<RelationalError> for RouterError {
    fn from(e: RelationalError) -> Self {
        Self::RelationalError(e.to_string())
    }
}

impl From<GraphError> for RouterError {
    fn from(e: GraphError) -> Self {
        Self::GraphError(e.to_string())
    }
}

impl From<VectorError> for RouterError {
    fn from(e: VectorError) -> Self {
        Self::VectorError(e.to_string())
    }
}

impl From<VaultError> for RouterError {
    fn from(e: VaultError) -> Self {
        Self::VaultError(e.to_string())
    }
}

impl From<CacheError> for RouterError {
    fn from(e: CacheError) -> Self {
        Self::CacheError(e.to_string())
    }
}

impl From<BlobError> for RouterError {
    fn from(e: BlobError) -> Self {
        Self::BlobError(e.to_string())
    }
}

impl From<CheckpointError> for RouterError {
    fn from(e: CheckpointError) -> Self {
        Self::CheckpointError(e.to_string())
    }
}

impl From<ChainError> for RouterError {
    fn from(e: ChainError) -> Self {
        Self::ChainError(e.to_string())
    }
}

impl From<UnifiedError> for RouterError {
    fn from(e: UnifiedError) -> Self {
        match e {
            UnifiedError::RelationalError(msg) => Self::RelationalError(msg),
            UnifiedError::GraphError(msg) => Self::GraphError(msg),
            UnifiedError::VectorError(msg) => Self::VectorError(msg),
            UnifiedError::NotFound(msg) => Self::VectorError(format!("Not found: {msg}")),
            UnifiedError::InvalidOperation(msg) => Self::InvalidArgument(msg),
            UnifiedError::BatchOperationFailed { index, key, cause } => Self::VectorError(format!(
                "Batch operation failed at index {index} (key: {key}): {cause}"
            )),
            UnifiedError::SpatialError(msg) => {
                Self::InvalidArgument(format!("Spatial error: {msg}"))
            },
        }
    }
}

/// Convenient `Result` alias bound to [`RouterError`].
pub type Result<T> = std::result::Result<T, RouterError>;
