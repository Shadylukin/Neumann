//! Error types for the Neumann gRPC server.

use thiserror::Error;
use tonic::Status;

/// Server error type.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ServerError {
    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Transport error.
    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    /// Query execution error.
    #[error("query error: {0}")]
    Query(String),

    /// Authentication error.
    #[error("authentication error: {0}")]
    Auth(String),

    /// Blob storage error.
    #[error("blob error: {0}")]
    Blob(String),

    /// Internal server error.
    #[error("internal error: {0}")]
    Internal(String),

    /// Invalid argument.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Resource not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// Permission denied.
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<ServerError> for Status {
    fn from(err: ServerError) -> Self {
        match &err {
            ServerError::Config(msg) => Status::invalid_argument(msg.clone()),
            ServerError::Transport(e) => Status::unavailable(e.to_string()),
            ServerError::Query(msg) => Status::invalid_argument(msg.clone()),
            ServerError::Auth(msg) => Status::unauthenticated(msg.clone()),
            ServerError::Blob(msg) => Status::internal(msg.clone()),
            ServerError::Internal(msg) => Status::internal(msg.clone()),
            ServerError::InvalidArgument(msg) => Status::invalid_argument(msg.clone()),
            ServerError::NotFound(msg) => Status::not_found(msg.clone()),
            ServerError::PermissionDenied(msg) => Status::permission_denied(msg.clone()),
            ServerError::Io(e) => Status::internal(e.to_string()),
        }
    }
}

impl From<query_router::RouterError> for ServerError {
    fn from(err: query_router::RouterError) -> Self {
        Self::Query(err.to_string())
    }
}

impl From<tensor_blob::BlobError> for ServerError {
    fn from(err: tensor_blob::BlobError) -> Self {
        Self::Blob(err.to_string())
    }
}

/// Result type alias for server operations.
pub type Result<T> = std::result::Result<T, ServerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_error_to_status() {
        let err = ServerError::Config("invalid port".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn test_query_error_to_status() {
        let err = ServerError::Query("syntax error".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn test_auth_error_to_status() {
        let err = ServerError::Auth("invalid token".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[test]
    fn test_not_found_error_to_status() {
        let err = ServerError::NotFound("artifact not found".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::NotFound);
    }

    #[test]
    fn test_permission_denied_to_status() {
        let err = ServerError::PermissionDenied("access denied".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::PermissionDenied);
    }

    #[test]
    fn test_internal_error_to_status() {
        let err = ServerError::Internal("unexpected error".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[test]
    fn test_blob_error_to_status() {
        let err = ServerError::Blob("storage error".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[test]
    fn test_invalid_argument_to_status() {
        let err = ServerError::InvalidArgument("bad input".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn test_error_display() {
        let err = ServerError::Config("test config error".to_string());
        assert_eq!(err.to_string(), "configuration error: test config error");

        let err = ServerError::Query("test query error".to_string());
        assert_eq!(err.to_string(), "query error: test query error");
    }
}
