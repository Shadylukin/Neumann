//! Error types for the cache module.

use thiserror::Error;

/// Errors that can occur during cache operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CacheError {
    /// Cache entry not found.
    #[error("cache entry not found: {0}")]
    NotFound(String),

    /// Embedding dimension mismatch.
    #[error("embedding dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// Storage error from underlying tensor store.
    #[error("storage error: {0}")]
    StorageError(String),

    /// Serialization or deserialization error.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// Tokenizer initialization or counting error.
    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    /// Cache is full and cannot accept more entries.
    #[error("cache full: {current} entries >= {capacity} capacity")]
    CacheFull { current: usize, capacity: usize },

    /// Invalid configuration provided.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Operation was cancelled or interrupted.
    #[error("operation cancelled: {0}")]
    Cancelled(String),
}

impl From<tensor_store::TensorStoreError> for CacheError {
    fn from(e: tensor_store::TensorStoreError) -> Self {
        CacheError::StorageError(e.to_string())
    }
}

impl From<bincode::Error> for CacheError {
    fn from(e: bincode::Error) -> Self {
        CacheError::SerializationError(e.to_string())
    }
}

/// Result type for cache operations.
pub type Result<T> = std::result::Result<T, CacheError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = CacheError::NotFound("key123".into());
        assert_eq!(e.to_string(), "cache entry not found: key123");

        let e = CacheError::DimensionMismatch {
            expected: 1536,
            got: 768,
        };
        assert_eq!(
            e.to_string(),
            "embedding dimension mismatch: expected 1536, got 768"
        );

        let e = CacheError::StorageError("disk full".into());
        assert_eq!(e.to_string(), "storage error: disk full");

        let e = CacheError::CacheFull {
            current: 10000,
            capacity: 10000,
        };
        assert_eq!(e.to_string(), "cache full: 10000 entries >= 10000 capacity");
    }

    #[test]
    fn test_error_clone_eq() {
        let e1 = CacheError::NotFound("test".into());
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_from_tensor_store_error() {
        let store_err = tensor_store::TensorStoreError::NotFound("key".into());
        let cache_err: CacheError = store_err.into();
        assert!(matches!(cache_err, CacheError::StorageError(_)));
    }

    #[test]
    fn test_serialization_error_display() {
        let e = CacheError::SerializationError("invalid format".into());
        assert_eq!(e.to_string(), "serialization error: invalid format");
    }

    #[test]
    fn test_tokenizer_error_display() {
        let e = CacheError::TokenizerError("failed to load encoder".into());
        assert_eq!(e.to_string(), "tokenizer error: failed to load encoder");
    }

    #[test]
    fn test_invalid_config_display() {
        let e = CacheError::InvalidConfig("negative capacity".into());
        assert_eq!(e.to_string(), "invalid configuration: negative capacity");
    }

    #[test]
    fn test_cancelled_display() {
        let e = CacheError::Cancelled("timeout".into());
        assert_eq!(e.to_string(), "operation cancelled: timeout");
    }

    #[test]
    fn test_from_bincode_error() {
        // Create a bincode error by trying to deserialize invalid data
        let invalid_data = b"not valid bincode";
        let result: std::result::Result<String, bincode::Error> =
            bincode::deserialize(invalid_data);
        if let Err(bincode_err) = result {
            let cache_err: CacheError = bincode_err.into();
            assert!(matches!(cache_err, CacheError::SerializationError(_)));
        }
    }
}
