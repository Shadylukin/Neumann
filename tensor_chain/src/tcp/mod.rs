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
//! ```ignore
//! use tensor_chain::tcp::{TcpTransport, TcpTransportConfig};
//!
//! let config = TcpTransportConfig::new("node1", "0.0.0.0:9100".parse()?);
//! let transport = TcpTransport::new(config).await?;
//!
//! // Connect to a peer
//! transport.connect(&PeerConfig {
//!     node_id: "node2".to_string(),
//!     address: "192.168.1.2:9100".to_string(),
//! }).await?;
//!
//! // Send a message
//! transport.send(&"node2".to_string(), Message::Ping { term: 1 }).await?;
//! ```

pub mod config;
pub mod connection;
pub mod error;
pub mod framing;
pub mod transport;

// Re-exports
pub use config::{ReconnectConfig, TcpTransportConfig, TlsConfig};
pub use connection::{
    Connection, ConnectionManager, ConnectionPool, ConnectionState, ConnectionStats,
};
pub use error::{TcpError, TcpResult};
pub use framing::{Handshake, LengthDelimitedCodec};
pub use transport::{TcpTransport, TransportStats};
