// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlobError {
    /// Artifact not found.
    NotFound(String),
    /// Chunk missing from storage.
    ChunkMissing(String),
    /// Checksum verification failed.
    ChecksumMismatch { expected: String, actual: String },
    /// Storage error from `TensorStore`.
    StorageError(String),
    /// Graph engine error.
    GraphError(String),
    /// Vector engine error.
    VectorError(String),
    /// Invalid artifact ID format.
    InvalidArtifactId(String),
    /// Invalid configuration.
    InvalidConfig(String),
    /// IO error during streaming.
    IoError(String),
    /// GC error.
    GcError(String),
    /// Artifact already exists.
    AlreadyExists(String),
    /// Empty data provided.
    EmptyData,
    /// Dimension mismatch for embeddings.
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for BlobError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "artifact not found: {id}"),
            Self::ChunkMissing(hash) => write!(f, "chunk missing: {hash}"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected}, got {actual}")
            },
            Self::StorageError(msg) => write!(f, "storage error: {msg}"),
            Self::GraphError(msg) => write!(f, "graph error: {msg}"),
            Self::VectorError(msg) => write!(f, "vector error: {msg}"),
            Self::InvalidArtifactId(id) => write!(f, "invalid artifact id: {id}"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::IoError(msg) => write!(f, "io error: {msg}"),
            Self::GcError(msg) => write!(f, "gc error: {msg}"),
            Self::AlreadyExists(id) => write!(f, "artifact already exists: {id}"),
            Self::EmptyData => write!(f, "empty data provided"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            },
        }
    }
}

impl std::error::Error for BlobError {}

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
        Self::IoError(e.to_string())
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
        assert!(matches!(blob_err, BlobError::IoError(_)));
    }
}
