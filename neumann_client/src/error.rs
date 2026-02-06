// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Error types for the Neumann client SDK.

use thiserror::Error;

/// Result type for client operations.
pub type Result<T> = std::result::Result<T, ClientError>;

/// Error types for client operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ClientError {
    /// Connection error.
    #[error("connection error: {0}")]
    Connection(String),

    /// Query execution error.
    #[error("query error: {0}")]
    Query(String),

    /// Authentication error.
    #[error("authentication failed: {0}")]
    Authentication(String),

    /// Permission denied.
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// Resource not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// Invalid argument.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// Parse error.
    #[error("parse error: {0}")]
    Parse(String),

    /// Internal error.
    #[error("internal error: {0}")]
    Internal(String),

    /// Timeout error.
    #[error("timeout: {0}")]
    Timeout(String),

    /// Server unavailable.
    #[error("server unavailable: {0}")]
    Unavailable(String),
}

impl ClientError {
    /// Get the error code.
    #[must_use]
    pub fn code(&self) -> u32 {
        match self {
            Self::Query(_) => 9,    // QUERY_ERROR
            Self::Parse(_) => 8,    // PARSE_ERROR
            Self::Internal(_) => 7, // INTERNAL
            Self::Connection(_) | Self::Timeout(_) | Self::Unavailable(_) => 6, // UNAVAILABLE
            Self::Authentication(_) => 5, // UNAUTHENTICATED
            Self::PermissionDenied(_) => 3, // PERMISSION_DENIED
            Self::NotFound(_) => 2, // NOT_FOUND
            Self::InvalidArgument(_) => 1, // INVALID_ARGUMENT
        }
    }

    /// Check if this is a retryable error.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Connection(_) | Self::Timeout(_) | Self::Unavailable(_)
        )
    }
}

#[cfg(feature = "remote")]
impl From<tonic::Status> for ClientError {
    fn from(status: tonic::Status) -> Self {
        match status.code() {
            tonic::Code::InvalidArgument => Self::InvalidArgument(status.message().to_string()),
            tonic::Code::NotFound => Self::NotFound(status.message().to_string()),
            tonic::Code::PermissionDenied => Self::PermissionDenied(status.message().to_string()),
            tonic::Code::Unauthenticated => Self::Authentication(status.message().to_string()),
            tonic::Code::Unavailable => Self::Unavailable(status.message().to_string()),
            tonic::Code::DeadlineExceeded => Self::Timeout(status.message().to_string()),
            _ => Self::Internal(status.message().to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(ClientError::Connection("test".to_string()).code(), 6);
        assert_eq!(ClientError::Query("test".to_string()).code(), 9);
        assert_eq!(ClientError::Authentication("test".to_string()).code(), 5);
        assert_eq!(ClientError::NotFound("test".to_string()).code(), 2);
        assert_eq!(ClientError::InvalidArgument("test".to_string()).code(), 1);
        assert_eq!(ClientError::Parse("test".to_string()).code(), 8);
        assert_eq!(ClientError::Internal("test".to_string()).code(), 7);
    }

    #[test]
    fn test_is_retryable() {
        assert!(ClientError::Connection("test".to_string()).is_retryable());
        assert!(ClientError::Timeout("test".to_string()).is_retryable());
        assert!(ClientError::Unavailable("test".to_string()).is_retryable());

        assert!(!ClientError::Query("test".to_string()).is_retryable());
        assert!(!ClientError::NotFound("test".to_string()).is_retryable());
        assert!(!ClientError::InvalidArgument("test".to_string()).is_retryable());
    }

    #[test]
    fn test_error_display() {
        let err = ClientError::Query("table not found".to_string());
        assert_eq!(err.to_string(), "query error: table not found");

        let err = ClientError::Connection("refused".to_string());
        assert_eq!(err.to_string(), "connection error: refused");
    }

    #[test]
    fn test_error_display_all_variants() {
        assert_eq!(
            ClientError::Authentication("bad token".to_string()).to_string(),
            "authentication failed: bad token"
        );
        assert_eq!(
            ClientError::PermissionDenied("access denied".to_string()).to_string(),
            "permission denied: access denied"
        );
        assert_eq!(
            ClientError::NotFound("missing".to_string()).to_string(),
            "not found: missing"
        );
        assert_eq!(
            ClientError::InvalidArgument("bad arg".to_string()).to_string(),
            "invalid argument: bad arg"
        );
        assert_eq!(
            ClientError::Parse("syntax error".to_string()).to_string(),
            "parse error: syntax error"
        );
        assert_eq!(
            ClientError::Internal("unexpected".to_string()).to_string(),
            "internal error: unexpected"
        );
        assert_eq!(
            ClientError::Timeout("exceeded".to_string()).to_string(),
            "timeout: exceeded"
        );
        assert_eq!(
            ClientError::Unavailable("down".to_string()).to_string(),
            "server unavailable: down"
        );
    }

    #[test]
    fn test_error_codes_all() {
        assert_eq!(ClientError::PermissionDenied("test".to_string()).code(), 3);
        assert_eq!(ClientError::Timeout("test".to_string()).code(), 6);
        assert_eq!(ClientError::Unavailable("test".to_string()).code(), 6);
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_from_tonic_status() {
        use tonic::{Code, Status};

        // InvalidArgument
        let status = Status::new(Code::InvalidArgument, "bad argument");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::InvalidArgument(_)));

        // NotFound
        let status = Status::new(Code::NotFound, "not found");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::NotFound(_)));

        // PermissionDenied
        let status = Status::new(Code::PermissionDenied, "denied");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::PermissionDenied(_)));

        // Unauthenticated
        let status = Status::new(Code::Unauthenticated, "auth failed");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Authentication(_)));

        // Unavailable
        let status = Status::new(Code::Unavailable, "service down");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Unavailable(_)));

        // DeadlineExceeded
        let status = Status::new(Code::DeadlineExceeded, "timeout");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Timeout(_)));

        // Unknown codes become Internal
        let status = Status::new(Code::Internal, "internal error");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));

        let status = Status::new(Code::Cancelled, "cancelled");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));

        let status = Status::new(Code::Unknown, "unknown");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));
    }

    #[test]
    fn test_error_debug() {
        let err = ClientError::Query("test".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("Query"));
        assert!(debug.contains("test"));
    }

    #[cfg(feature = "remote")]
    #[test]
    fn test_from_tonic_status_additional_codes() {
        use tonic::{Code, Status};

        // FailedPrecondition -> Internal
        let status = Status::new(Code::FailedPrecondition, "precondition failed");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));

        // OutOfRange -> Internal
        let status = Status::new(Code::OutOfRange, "out of range");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));

        // Aborted -> Internal
        let status = Status::new(Code::Aborted, "aborted");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));

        // DataLoss -> Internal
        let status = Status::new(Code::DataLoss, "data loss");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));

        // ResourceExhausted -> Internal
        let status = Status::new(Code::ResourceExhausted, "exhausted");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));

        // Unimplemented -> Internal
        let status = Status::new(Code::Unimplemented, "unimplemented");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));

        // AlreadyExists -> Internal
        let status = Status::new(Code::AlreadyExists, "exists");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));

        // Ok -> Internal (edge case: should not normally happen)
        let status = Status::new(Code::Ok, "ok");
        let err: ClientError = status.into();
        assert!(matches!(err, ClientError::Internal(_)));
    }
}
