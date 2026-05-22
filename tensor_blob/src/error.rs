// SPDX-License-Identifier: MIT OR Apache-2.0
use thiserror::Error;

/// Errors returned by the blob storage layer.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BlobError {
    /// Artifact not found.
    #[error("artifact not found: {0}")]
    NotFound(String),
    /// Chunk missing from storage.
    #[error("chunk missing: {0}")]
    ChunkMissing(String),
    /// Checksum verification failed.
    #[error("checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },
    /// Storage error from `TensorStore`.
    #[error("storage error: {0}")]
    StorageError(String),
    /// Graph engine error.
    #[error("graph error: {0}")]
    GraphError(String),
    /// Vector engine error.
    #[error("vector error: {0}")]
    VectorError(String),
    /// Invalid artifact ID format.
    #[error("invalid artifact id: {0}")]
    InvalidArtifactId(String),
    /// Invalid configuration.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// IO error during streaming.
    #[error("io error: {message}")]
    IoError {
        /// The kind of IO error that occurred.
        kind: std::io::ErrorKind,
        /// Human-readable error message.
        message: String,
    },
    /// GC error.
    #[error("gc error: {0}")]
    GcError(String),
    /// Artifact already exists.
    #[error("artifact already exists: {0}")]
    AlreadyExists(String),
    /// Empty data provided.
    #[error("empty data provided")]
    EmptyData,
    /// Dimension mismatch for embeddings.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
}

impl BlobError {
    /// Returns the `io::ErrorKind` if this is an IO error.
    #[must_use]
    pub const fn io_error_kind(&self) -> Option<std::io::ErrorKind> {
        match self {
            Self::IoError { kind, .. } => Some(*kind),
            _ => None,
        }
    }
}

impl From<tensor_store::TensorStoreError> for BlobError {
    fn from(e: tensor_store::TensorStoreError) -> Self {
        Self::StorageError(e.to_string())
    }
}

#[cfg(feature = "graph")]
impl From<graph_engine::GraphError> for BlobError {
    fn from(e: graph_engine::GraphError) -> Self {
        Self::GraphError(e.to_string())
    }
}

#[cfg(feature = "vector")]
impl From<vector_engine::VectorError> for BlobError {
    fn from(e: vector_engine::VectorError) -> Self {
        Self::VectorError(e.to_string())
    }
}

impl From<std::io::Error> for BlobError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError {
            kind: e.kind(),
            message: e.to_string(),
        }
    }
}

pub type Result<T> = std::result::Result<T, BlobError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = BlobError::NotFound("test.pdf".to_string());
        assert_eq!(err.to_string(), "artifact not found: test.pdf");

        let err = BlobError::ChunkMissing("sha256:abc123".to_string());
        assert_eq!(err.to_string(), "chunk missing: sha256:abc123");

        let err = BlobError::ChecksumMismatch {
            expected: "sha256:aaa".to_string(),
            actual: "sha256:bbb".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "checksum mismatch: expected sha256:aaa, got sha256:bbb"
        );

        let err = BlobError::EmptyData;
        assert_eq!(err.to_string(), "empty data provided");

        let err = BlobError::DimensionMismatch {
            expected: 128,
            got: 256,
        };
        assert_eq!(err.to_string(), "dimension mismatch: expected 128, got 256");
    }

    #[test]
    fn test_error_equality() {
        let err1 = BlobError::NotFound("a".to_string());
        let err2 = BlobError::NotFound("a".to_string());
        let err3 = BlobError::NotFound("b".to_string());

        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }

    #[test]
    fn test_from_tensor_store_error() {
        let store_err = tensor_store::TensorStoreError::NotFound("key".to_string());
        let blob_err: BlobError = store_err.into();
        assert!(matches!(blob_err, BlobError::StorageError(_)));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let blob_err: BlobError = io_err.into();
        assert!(matches!(blob_err, BlobError::IoError { .. }));
    }

    #[test]
    fn test_error_display_all_variants() {
        let cases: Vec<(BlobError, &str)> = vec![
            (
                BlobError::StorageError("disk full".to_string()),
                "storage error: disk full",
            ),
            (
                BlobError::GraphError("cycle detected".to_string()),
                "graph error: cycle detected",
            ),
            (
                BlobError::VectorError("dim mismatch".to_string()),
                "vector error: dim mismatch",
            ),
            (
                BlobError::InvalidArtifactId("!!!".to_string()),
                "invalid artifact id: !!!",
            ),
            (
                BlobError::InvalidConfig("bad chunk size".to_string()),
                "invalid config: bad chunk size",
            ),
            (
                BlobError::IoError {
                    kind: std::io::ErrorKind::Other,
                    message: "permission denied".to_string(),
                },
                "io error: permission denied",
            ),
            (
                BlobError::GcError("gc failed".to_string()),
                "gc error: gc failed",
            ),
            (
                BlobError::AlreadyExists("doc.pdf".to_string()),
                "artifact already exists: doc.pdf",
            ),
        ];

        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }
    }

    #[test]
    fn test_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BlobError::NotFound("test".to_string()));
        assert!(err.to_string().contains("artifact not found"));
    }

    #[test]
    fn test_error_clone() {
        let err = BlobError::ChecksumMismatch {
            expected: "a".to_string(),
            actual: "b".to_string(),
        };
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_debug() {
        let err = BlobError::EmptyData;
        let debug = format!("{err:?}");
        assert!(debug.contains("EmptyData"));
    }

    #[test]
    fn test_io_error_kind_preservation() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let blob_err: BlobError = io_err.into();
        assert_eq!(blob_err.io_error_kind(), Some(std::io::ErrorKind::NotFound));
    }

    #[test]
    fn test_io_error_kind_none_for_other_variants() {
        let err = BlobError::NotFound("test".to_string());
        assert_eq!(err.io_error_kind(), None);
    }
}
