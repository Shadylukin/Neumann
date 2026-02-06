// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
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

    /// Rate limit exceeded.
    #[error("rate limit exceeded: {0}")]
    RateLimited(String),
}

/// Generic internal error message for client responses.
const INTERNAL_ERROR_MESSAGE: &str = "An internal error occurred. Please try again later.";

impl From<ServerError> for Status {
    fn from(err: ServerError) -> Self {
        match &err {
            ServerError::Config(msg) => Status::invalid_argument(msg.clone()),
            ServerError::Transport(e) => {
                tracing::warn!(error = %e, "Transport error");
                Status::unavailable("Service temporarily unavailable")
            },
            ServerError::Query(msg) => Status::invalid_argument(msg.clone()),
            ServerError::Auth(msg) => Status::unauthenticated(msg.clone()),
            ServerError::Blob(msg) => {
                tracing::error!(error = %msg, "Blob storage error");
                Status::internal(INTERNAL_ERROR_MESSAGE)
            },
            ServerError::Internal(msg) => {
                tracing::error!(error = %msg, "Internal server error");
                Status::internal(INTERNAL_ERROR_MESSAGE)
            },
            ServerError::InvalidArgument(msg) => Status::invalid_argument(msg.clone()),
            ServerError::NotFound(msg) => Status::not_found(msg.clone()),
            ServerError::PermissionDenied(msg) => Status::permission_denied(msg.clone()),
            ServerError::Io(e) => {
                tracing::error!(error = %e, "I/O error");
                Status::internal(INTERNAL_ERROR_MESSAGE)
            },
            ServerError::RateLimited(msg) => Status::resource_exhausted(msg.clone()),
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

/// Sanitize an error message for client responses.
///
/// This function logs the full error details server-side and returns a
/// sanitized message that is safe to expose to clients. Internal errors
/// are replaced with a generic message to avoid leaking implementation details.
pub fn sanitize_internal_error<E: std::fmt::Display>(error: E) -> Status {
    tracing::error!(error = %error, "Internal server error");
    Status::internal(INTERNAL_ERROR_MESSAGE)
}

/// Sanitize any error that should not expose details to clients.
///
/// This logs the full error and returns an appropriate generic response.
pub fn sanitize_error<E: std::fmt::Display>(error: E, code: tonic::Code) -> Status {
    let msg = match code {
        tonic::Code::Internal => {
            tracing::error!(error = %error, "Internal server error");
            INTERNAL_ERROR_MESSAGE.to_string()
        },
        tonic::Code::Unavailable => {
            tracing::warn!(error = %error, "Service unavailable");
            "Service temporarily unavailable".to_string()
        },
        _ => error.to_string(),
    };
    Status::new(code, msg)
}

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

    #[test]
    fn test_rate_limited_to_status() {
        let err = ServerError::RateLimited("too many requests".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn test_rate_limited_error_display() {
        let err = ServerError::RateLimited("too many requests".to_string());
        assert_eq!(err.to_string(), "rate limit exceeded: too many requests");
    }

    #[test]
    fn test_internal_error_sanitization() {
        // Internal errors should not expose their message to clients
        let err = ServerError::Internal("secret database connection string".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(!status.message().contains("secret"));
        assert!(status.message().contains("internal error"));
    }

    #[test]
    fn test_blob_error_sanitization() {
        // Blob errors should not expose internal details
        let err = ServerError::Blob("failed to write to /var/data/secrets".to_string());
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(!status.message().contains("/var/data"));
    }

    #[test]
    fn test_io_error_sanitization() {
        // I/O errors should not expose file paths
        let err = ServerError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file /etc/passwd not found",
        ));
        let status: Status = err.into();
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(!status.message().contains("/etc/passwd"));
    }

    #[test]
    fn test_sanitize_internal_error() {
        let status = sanitize_internal_error("secret data");
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(!status.message().contains("secret"));
    }

    #[test]
    fn test_sanitize_error_internal() {
        let status = sanitize_error("sensitive info", tonic::Code::Internal);
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(!status.message().contains("sensitive"));
    }

    #[test]
    fn test_sanitize_error_unavailable() {
        let status = sanitize_error("connection to db failed", tonic::Code::Unavailable);
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert!(!status.message().contains("db"));
    }

    #[test]
    fn test_sanitize_error_other_codes() {
        // Non-internal codes should pass through
        let status = sanitize_error("invalid input", tonic::Code::InvalidArgument);
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(status.message(), "invalid input");
    }
}
