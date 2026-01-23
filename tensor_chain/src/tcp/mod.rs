//! TCP transport implementation for distributed consensus.
//!
//! This module provides a production-ready TCP transport that implements
//! the `Transport` trait for node-to-node communication.
//!
//! # Architecture
//!
//! ```text
//! TcpTransport
//!   ├── ConnectionManager
//!   │     ├── ConnectionPool (per peer)
//!   │     │     └── Connection (persistent TCP)
//!   │     └── ReconnectManager
//!   ├── Listener (accept incoming)
//!   └── Framing (length-delimited codec)
//! ```
//!
//! # Wire Protocol
//!
//! Messages use a simple length-prefixed binary format:
//!
//! ```text
//! +------------------+------------------+
//! | Length (4B BE)   | Payload (bincode)|
//! +------------------+------------------+
//! ```
//!
//! # Example
//!
//! ```rust
//! use tensor_chain::tcp::{TcpTransport, TcpTransportConfig, SecurityMode};
//! use tensor_chain::network::Transport;
//! use std::net::SocketAddr;
//!
//! // Configure transport (sync construction)
//! let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
//! let config = TcpTransportConfig::new("node1", addr)
//!     .with_security_mode(SecurityMode::Development);
//!
//! // Create transport (not yet listening)
//! let transport = TcpTransport::new(config);
//!
//! assert_eq!(transport.local_id(), "node1");
//! assert!(!transport.is_running());
//! ```
//!
//! For async operations (connecting, sending), use within a Tokio runtime:
//!
//! ```rust,no_run
//! use tensor_chain::tcp::{TcpTransport, TcpTransportConfig, SecurityMode};
//! use tensor_chain::network::{Message, PeerConfig, Transport};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = TcpTransportConfig::new("node1", "0.0.0.0:9100".parse()?)
//!     .with_security_mode(SecurityMode::Development);
//! let transport = TcpTransport::new(config);
//!
//! // Start listening
//! transport.start().await?;
//!
//! // Connect to a peer
//! transport.connect(&PeerConfig {
//!     node_id: "node2".to_string(),
//!     address: "192.168.1.2:9100".to_string(),
//! }).await?;
//!
//! // Send a message
//! transport.send(&"node2".to_string(), Message::Ping { term: 1 }).await?;
//! # Ok(())
//! # }
//! ```

pub mod compression;
pub mod config;
pub mod connection;
pub mod error;
pub mod framing;
pub mod rate_limit;
pub mod stream;
#[cfg(feature = "tls")]
pub mod tls;
pub mod transport;

// Re-exports
pub use compression::{CompressionConfig, CompressionMethod, COMPRESSION_CAPABILITY};
pub use config::{
    NodeIdVerification, ReconnectConfig, SecurityConfig, SecurityMode, TcpTransportConfig,
    TlsConfig,
};
pub use connection::{
    Connection, ConnectionManager, ConnectionPool, ConnectionState, ConnectionStats,
};
pub use error::{TcpError, TcpResult};
pub use framing::{Handshake, LengthDelimitedCodec};
pub use rate_limit::{PeerRateLimiter, RateLimitConfig};
pub use stream::{box_stream, split_stream, AsyncStream, DynRead, DynStream, DynWrite};
#[cfg(feature = "tls")]
pub use tls::{
    extract_node_id_from_cert, wrap_client, wrap_server, wrap_server_with_identity,
    ClientTlsStream, NodeIdSource, ServerTlsStream, VerifiedPeerIdentity,
};
pub use transport::{TcpTransport, TransportStats};
