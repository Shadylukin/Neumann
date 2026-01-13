use thiserror::Error;

#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("checkpoint not found: {0}")]
    NotFound(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("deserialization error: {0}")]
    Deserialization(String),

    #[error("blob error: {0}")]
    Blob(#[from] tensor_blob::BlobError),

    #[error("snapshot error: {0}")]
    Snapshot(String),

    #[error("operation cancelled by user")]
    Cancelled,

    #[error("invalid checkpoint id: {0}")]
    InvalidId(String),

    #[error("retention error: {0}")]
    Retention(String),
}

pub type Result<T> = std::result::Result<T, CheckpointError>;

impl From<bincode::Error> for CheckpointError {
    fn from(e: bincode::Error) -> Self {
        CheckpointError::Serialization(e.to_string())
    }
}

impl From<tensor_store::SnapshotError> for CheckpointError {
    fn from(e: tensor_store::SnapshotError) -> Self {
        CheckpointError::Snapshot(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = CheckpointError::NotFound("cp-123".to_string());
        assert!(err.to_string().contains("checkpoint not found"));

        let err = CheckpointError::Storage("disk full".to_string());
        assert!(err.to_string().contains("storage error"));

        let err = CheckpointError::Serialization("invalid data".to_string());
        assert!(err.to_string().contains("serialization error"));

        let err = CheckpointError::Deserialization("corrupt".to_string());
        assert!(err.to_string().contains("deserialization error"));

        let err = CheckpointError::Snapshot("failed".to_string());
        assert!(err.to_string().contains("snapshot error"));

        let err = CheckpointError::Cancelled;
        assert!(err.to_string().contains("cancelled"));

        let err = CheckpointError::InvalidId("bad-id".to_string());
        assert!(err.to_string().contains("invalid checkpoint id"));

        let err = CheckpointError::Retention("too many".to_string());
        assert!(err.to_string().contains("retention error"));
    }

    #[test]
    fn test_from_bincode_error() {
        // Create a bincode error by trying to deserialize invalid data
        let bad_data: &[u8] = &[0xff, 0xff, 0xff];
        let bincode_err: std::result::Result<String, _> = bincode::deserialize(bad_data);

        if let Err(e) = bincode_err {
            let checkpoint_err: CheckpointError = e.into();
            assert!(matches!(checkpoint_err, CheckpointError::Serialization(_)));
        }
    }
}
