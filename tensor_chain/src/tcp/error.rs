// SPDX-License-Identifier: MIT OR Apache-2.0
//! TCP transport error types.

use thiserror::Error;

use crate::tcp::config::SecurityMode;

/// Errors specific to TCP transport operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TcpError {
    /// Connection to peer failed.
    #[error("connection to {peer} failed: {reason}")]
    ConnectionFailed { peer: String, reason: String },

    /// Connection was closed by peer.
    #[error("connection closed by peer")]
    ConnectionClosed,

    /// Operation timed out.
    #[error("{operation} timed out after {timeout_ms}ms")]
    Timeout {
        operation: &'static str,
        timeout_ms: u64,
    },

    /// Message exceeds maximum size.
    #[error("message too large: {size} bytes (max {max_size})")]
    MessageTooLarge { size: usize, max_size: usize },

    /// Failed to serialize or deserialize message.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// IO error.
    #[error("io error: {0}")]
    Io(#[source] std::io::Error),

    /// Peer not found in connection pool.
    #[error("peer not found: {0}")]
    PeerNotFound(String),

    /// Backpressure: outbound queue full.
    #[error("backpressure: queue full for {peer} ({queue_size} pending)")]
    BackpressureFull { peer: String, queue_size: usize },

    /// TLS error.
    #[error("TLS error: {0}")]
    TlsError(String),

    /// Transport is shutting down.
    #[error("transport is shutting down")]
    Shutdown,

    /// Invalid frame received.
    #[error("invalid frame: {0}")]
    InvalidFrame(String),

    /// Handshake failed.
    #[error("handshake failed: {0}")]
    HandshakeFailed(String),

    /// Identity verification failed during TLS handshake.
    #[error("identity verification failed for node '{claimed_node_id}': {reason}")]
    IdentityVerificationFailed {
        reason: String,
        claimed_node_id: String,
    },

    /// Certificate `NodeId` does not match claimed `NodeId`.
    #[error("certificate NodeId mismatch: cert='{cert_node_id}', claimed='{claimed_node_id}'")]
    CertificateNodeIdMismatch {
        cert_node_id: String,
        claimed_node_id: String,
    },

    /// Client certificate required but not provided.
    #[error("client certificate required but not provided")]
    ClientCertificateRequired,

    /// Compression/decompression error.
    #[error("compression {operation} error: {message}")]
    Compression {
        operation: &'static str,
        message: String,
    },

    /// Rate limited: peer is being sent messages too fast.
    #[error("rate limited: peer {peer} (available tokens: {available})")]
    RateLimited { peer: String, available: u32 },

    /// Connection pool exhausted: no healthy connections available.
    #[error("pool exhausted for {peer}: {active}/{target} active connections")]
    PoolExhausted {
        peer: String,
        active: usize,
        target: usize,
    },

    /// TLS is required by security mode but not configured.
    #[error("TLS required by security mode {mode:?}: {reason}")]
    TlsRequired { mode: SecurityMode, reason: String },

    /// Mutual TLS (client auth) is required by security mode.
    #[error("mutual TLS required by security mode {mode:?}: {reason}")]
    MtlsRequired { mode: SecurityMode, reason: String },

    /// `NodeId` verification is required by security mode.
    #[error("NodeId verification required by security mode {mode:?}: {reason}")]
    NodeIdVerificationRequired { mode: SecurityMode, reason: String },

    /// Connection rejected: plaintext connection when TLS is required.
    #[error("plaintext connection rejected from {remote_addr}: TLS is required")]
    PlaintextRejected { remote_addr: String },

    /// Connection rejected: no client certificate when mTLS is required.
    #[error(
        "connection from {remote_addr} rejected: client certificate required but not provided"
    )]
    ClientCertMissing { remote_addr: String },
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

/// Result type for TCP transport operations.
pub type TcpResult<T> = std::result::Result<T, TcpError>;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::error::ChainError;

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
        assert!(matches!(chain_err, ChainError::TcpTransportError(_)));
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
