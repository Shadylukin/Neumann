// SPDX-License-Identifier: MIT OR Apache-2.0
//! TCP transport error types.

use std::fmt;

use crate::error::ChainError;
use crate::tcp::config::SecurityMode;

/// Errors specific to TCP transport operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum TcpError {
    /// Connection to peer failed.
    ConnectionFailed { peer: String, reason: String },

    /// Connection was closed by peer.
    ConnectionClosed,

    /// Operation timed out.
    Timeout {
        operation: &'static str,
        timeout_ms: u64,
    },

    /// Message exceeds maximum size.
    MessageTooLarge { size: usize, max_size: usize },

    /// Failed to serialize or deserialize message.
    Serialization(String),

    /// IO error.
    Io(std::io::Error),

    /// Peer not found in connection pool.
    PeerNotFound(String),

    /// Backpressure: outbound queue full.
    BackpressureFull { peer: String, queue_size: usize },

    /// TLS error.
    TlsError(String),

    /// Transport is shutting down.
    Shutdown,

    /// Invalid frame received.
    InvalidFrame(String),

    /// Handshake failed.
    HandshakeFailed(String),

    /// Identity verification failed during TLS handshake.
    IdentityVerificationFailed {
        reason: String,
        claimed_node_id: String,
    },

    /// Certificate NodeId does not match claimed NodeId.
    CertificateNodeIdMismatch {
        cert_node_id: String,
        claimed_node_id: String,
    },

    /// Client certificate required but not provided.
    ClientCertificateRequired,

    /// Compression/decompression error.
    Compression {
        operation: &'static str,
        message: String,
    },

    /// Rate limited: peer is being sent messages too fast.
    RateLimited { peer: String, available: u32 },

    /// TLS is required by security mode but not configured.
    TlsRequired { mode: SecurityMode, reason: String },

    /// Mutual TLS (client auth) is required by security mode.
    MtlsRequired { mode: SecurityMode, reason: String },

    /// NodeId verification is required by security mode.
    NodeIdVerificationRequired { mode: SecurityMode, reason: String },

    /// Connection rejected: plaintext connection when TLS is required.
    PlaintextRejected { remote_addr: String },

    /// Connection rejected: no client certificate when mTLS is required.
    ClientCertMissing { remote_addr: String },
}

impl fmt::Display for TcpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConnectionFailed { peer, reason } => {
                write!(f, "connection to {} failed: {}", peer, reason)
            },
            Self::ConnectionClosed => write!(f, "connection closed by peer"),
            Self::Timeout {
                operation,
                timeout_ms,
            } => {
                write!(f, "{} timed out after {}ms", operation, timeout_ms)
            },
            Self::MessageTooLarge { size, max_size } => {
                write!(f, "message too large: {} bytes (max {})", size, max_size)
            },
            Self::Serialization(msg) => write!(f, "serialization error: {}", msg),
            Self::Io(err) => write!(f, "io error: {}", err),
            Self::PeerNotFound(peer) => write!(f, "peer not found: {}", peer),
            Self::BackpressureFull { peer, queue_size } => {
                write!(
                    f,
                    "backpressure: queue full for {} ({} pending)",
                    peer, queue_size
                )
            },
            Self::TlsError(msg) => write!(f, "TLS error: {}", msg),
            Self::Shutdown => write!(f, "transport is shutting down"),
            Self::InvalidFrame(msg) => write!(f, "invalid frame: {}", msg),
            Self::HandshakeFailed(msg) => write!(f, "handshake failed: {}", msg),
            Self::IdentityVerificationFailed {
                reason,
                claimed_node_id,
            } => {
                write!(
                    f,
                    "identity verification failed for node '{}': {}",
                    claimed_node_id, reason
                )
            },
            Self::CertificateNodeIdMismatch {
                cert_node_id,
                claimed_node_id,
            } => {
                write!(
                    f,
                    "certificate NodeId mismatch: cert='{}', claimed='{}'",
                    cert_node_id, claimed_node_id
                )
            },
            Self::ClientCertificateRequired => {
                write!(f, "client certificate required but not provided")
            },
            Self::Compression { operation, message } => {
                write!(f, "compression {} error: {}", operation, message)
            },
            Self::RateLimited { peer, available } => {
                write!(
                    f,
                    "rate limited: peer {} (available tokens: {})",
                    peer, available
                )
            },
            Self::TlsRequired { mode, reason } => {
                write!(f, "TLS required by security mode {:?}: {}", mode, reason)
            },
            Self::MtlsRequired { mode, reason } => {
                write!(
                    f,
                    "mutual TLS required by security mode {:?}: {}",
                    mode, reason
                )
            },
            Self::NodeIdVerificationRequired { mode, reason } => {
                write!(
                    f,
                    "NodeId verification required by security mode {:?}: {}",
                    mode, reason
                )
            },
            Self::PlaintextRejected { remote_addr } => {
                write!(
                    f,
                    "plaintext connection rejected from {}: TLS is required",
                    remote_addr
                )
            },
            Self::ClientCertMissing { remote_addr } => {
                write!(
                    f,
                    "connection from {} rejected: client certificate required but not provided",
                    remote_addr
                )
            },
        }
    }
}

impl std::error::Error for TcpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TcpError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<bitcode::Error> for TcpError {
    fn from(err: bitcode::Error) -> Self {
        Self::Serialization(err.to_string())
    }
}

impl From<TcpError> for ChainError {
    fn from(err: TcpError) -> Self {
        ChainError::NetworkError(err.to_string())
    }
}

/// Result type for TCP transport operations.
pub type TcpResult<T> = std::result::Result<T, TcpError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_failed_display() {
        let err = TcpError::ConnectionFailed {
            peer: "node1".to_string(),
            reason: "refused".to_string(),
        };
        assert!(err.to_string().contains("node1"));
        assert!(err.to_string().contains("refused"));
    }

    #[test]
    fn test_connection_closed_display() {
        let err = TcpError::ConnectionClosed;
        assert!(err.to_string().contains("closed"));
    }

    #[test]
    fn test_timeout_display() {
        let err = TcpError::Timeout {
            operation: "connect",
            timeout_ms: 5000,
        };
        assert!(err.to_string().contains("connect"));
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn test_message_too_large_display() {
        let err = TcpError::MessageTooLarge {
            size: 100,
            max_size: 50,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_serialization_display() {
        let err = TcpError::Serialization("bad data".to_string());
        assert!(err.to_string().contains("bad data"));
    }

    #[test]
    fn test_io_display() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = TcpError::Io(io_err);
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_peer_not_found_display() {
        let err = TcpError::PeerNotFound("unknown".to_string());
        assert!(err.to_string().contains("unknown"));
    }

    #[test]
    fn test_backpressure_display() {
        let err = TcpError::BackpressureFull {
            peer: "node1".to_string(),
            queue_size: 1000,
        };
        assert!(err.to_string().contains("node1"));
        assert!(err.to_string().contains("1000"));
    }

    #[test]
    fn test_tls_error_display() {
        let err = TcpError::TlsError("certificate error".to_string());
        assert!(err.to_string().contains("certificate"));
    }

    #[test]
    fn test_shutdown_display() {
        let err = TcpError::Shutdown;
        assert!(err.to_string().contains("shutting down"));
    }

    #[test]
    fn test_invalid_frame_display() {
        let err = TcpError::InvalidFrame("bad frame".to_string());
        assert!(err.to_string().contains("bad frame"));
    }

    #[test]
    fn test_handshake_failed_display() {
        let err = TcpError::HandshakeFailed("version mismatch".to_string());
        assert!(err.to_string().contains("version mismatch"));
    }

    #[test]
    fn test_compression_display() {
        let err = TcpError::Compression {
            operation: "decompress",
            message: "invalid data".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("decompress"));
        assert!(display.contains("invalid data"));
    }

    #[test]
    fn test_rate_limited_display() {
        let err = TcpError::RateLimited {
            peer: "node1".to_string(),
            available: 0,
        };
        let display = err.to_string();
        assert!(display.contains("rate limited"));
        assert!(display.contains("node1"));
        assert!(display.contains("0"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused");
        let tcp_err: TcpError = io_err.into();
        assert!(matches!(tcp_err, TcpError::Io(_)));
    }

    #[test]
    fn test_from_bincode_error() {
        // Create a bincode error by deserializing invalid data
        let bad_data = vec![0xff, 0xff, 0xff];
        let result: std::result::Result<String, bitcode::Error> = bitcode::deserialize(&bad_data);
        if let Err(bincode_err) = result {
            let tcp_err: TcpError = bincode_err.into();
            assert!(matches!(tcp_err, TcpError::Serialization(_)));
        }
    }

    #[test]
    fn test_into_chain_error() {
        let tcp_err = TcpError::ConnectionClosed;
        let chain_err: ChainError = tcp_err.into();
        assert!(matches!(chain_err, ChainError::NetworkError(_)));
    }

    #[test]
    fn test_error_source() {
        use std::error::Error;

        // IO error has source
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let tcp_err = TcpError::Io(io_err);
        assert!(tcp_err.source().is_some());

        // Other errors don't have source
        let tcp_err = TcpError::ConnectionClosed;
        assert!(tcp_err.source().is_none());
    }

    #[test]
    fn test_debug_format() {
        let err = TcpError::Timeout {
            operation: "test",
            timeout_ms: 100,
        };
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Timeout"));
    }

    #[test]
    fn test_identity_verification_failed_display() {
        let err = TcpError::IdentityVerificationFailed {
            reason: "public key mismatch".to_string(),
            claimed_node_id: "node1".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("identity verification failed"));
        assert!(display.contains("node1"));
        assert!(display.contains("public key mismatch"));
    }

    #[test]
    fn test_certificate_node_id_mismatch_display() {
        let err = TcpError::CertificateNodeIdMismatch {
            cert_node_id: "node-cert".to_string(),
            claimed_node_id: "node-claimed".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("mismatch"));
        assert!(display.contains("node-cert"));
        assert!(display.contains("node-claimed"));
    }

    #[test]
    fn test_client_certificate_required_display() {
        let err = TcpError::ClientCertificateRequired;
        let display = err.to_string();
        assert!(display.contains("client certificate required"));
    }

    #[test]
    fn test_tls_required_display() {
        let err = TcpError::TlsRequired {
            mode: SecurityMode::Strict,
            reason: "TLS configuration missing".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("TLS required"));
        assert!(display.contains("Strict"));
        assert!(display.contains("TLS configuration missing"));
    }

    #[test]
    fn test_mtls_required_display() {
        let err = TcpError::MtlsRequired {
            mode: SecurityMode::Strict,
            reason: "client auth not enabled".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("mutual TLS required"));
        assert!(display.contains("Strict"));
        assert!(display.contains("client auth not enabled"));
    }

    #[test]
    fn test_node_id_verification_required_display() {
        let err = TcpError::NodeIdVerificationRequired {
            mode: SecurityMode::Strict,
            reason: "verification mode not set".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("NodeId verification required"));
        assert!(display.contains("Strict"));
    }

    #[test]
    fn test_plaintext_rejected_display() {
        let err = TcpError::PlaintextRejected {
            remote_addr: "192.168.1.1:5000".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("plaintext connection rejected"));
        assert!(display.contains("192.168.1.1:5000"));
        assert!(display.contains("TLS is required"));
    }

    #[test]
    fn test_client_cert_missing_display() {
        let err = TcpError::ClientCertMissing {
            remote_addr: "10.0.0.1:8080".to_string(),
        };
        let display = err.to_string();
        assert!(display.contains("client certificate required but not provided"));
        assert!(display.contains("10.0.0.1:8080"));
    }
}
