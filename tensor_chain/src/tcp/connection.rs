//! Connection management for TCP transport.
//!
//! Provides connection pooling and state management for peer connections.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::RwLock;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc;

use crate::block::NodeId;
use crate::network::Message;

use super::config::TcpTransportConfig;
use super::error::{TcpError, TcpResult};
use super::framing::LengthDelimitedCodec;
use super::stream::{split_stream, DynRead, DynWrite};

/// State of a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is being established.
    Connecting,
    /// Connection is active and healthy.
    Connected,
    /// Connection was closed.
    Disconnected,
    /// Connection failed and is being reconnected.
    Reconnecting { attempt: usize, next_retry: Instant },
}

/// Statistics for a connection.
#[derive(Debug, Default)]
pub struct ConnectionStats {
    /// Total messages sent.
    pub messages_sent: AtomicU64,
    /// Total messages received.
    pub messages_received: AtomicU64,
    /// Total bytes sent.
    pub bytes_sent: AtomicU64,
    /// Total bytes received.
    pub bytes_received: AtomicU64,
    /// Number of reconnections.
    pub reconnections: AtomicU64,
    /// Last activity timestamp (epoch ms).
    pub last_activity: AtomicU64,
}

impl ConnectionStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_send(&self, bytes: usize) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
        self.bytes_sent.fetch_add(bytes as u64, Ordering::Relaxed);
        self.touch();
    }

    pub fn record_recv(&self, bytes: usize) {
        self.messages_received.fetch_add(1, Ordering::Relaxed);
        self.bytes_received
            .fetch_add(bytes as u64, Ordering::Relaxed);
        self.touch();
    }

    pub fn record_reconnect(&self) {
        self.reconnections.fetch_add(1, Ordering::Relaxed);
    }

    fn touch(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.last_activity.store(now, Ordering::Relaxed);
    }
}

/// A single connection to a peer.
pub struct Connection {
    /// Peer node ID.
    pub peer_id: NodeId,
    /// Peer address.
    pub address: SocketAddr,
    /// Current state.
    state: RwLock<ConnectionState>,
    /// Read half of the stream (boxed for TLS support).
    reader: RwLock<Option<DynRead>>,
    /// Write half of the stream (boxed for TLS support).
    writer: RwLock<Option<DynWrite>>,
    /// Message codec.
    codec: LengthDelimitedCodec,
    /// Connection statistics.
    pub stats: ConnectionStats,
    /// Connection ID (unique within pool).
    pub id: usize,
}

impl Connection {
    /// Create a new connection from any async stream (TCP or TLS).
    pub fn new<S>(
        peer_id: NodeId,
        address: SocketAddr,
        stream: S,
        codec: LengthDelimitedCodec,
        id: usize,
    ) -> Self
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + Sync + 'static,
    {
        let (reader, writer) = split_stream(stream);
        Self {
            peer_id,
            address,
            state: RwLock::new(ConnectionState::Connected),
            reader: RwLock::new(Some(reader)),
            writer: RwLock::new(Some(writer)),
            codec,
            stats: ConnectionStats::new(),
            id,
        }
    }

    /// Create a connection in connecting state (before handshake).
    pub fn connecting(
        peer_id: NodeId,
        address: SocketAddr,
        codec: LengthDelimitedCodec,
        id: usize,
    ) -> Self {
        Self {
            peer_id,
            address,
            state: RwLock::new(ConnectionState::Connecting),
            reader: RwLock::new(None),
            writer: RwLock::new(None),
            codec,
            stats: ConnectionStats::new(),
            id,
        }
    }

    /// Get current connection state.
    pub fn state(&self) -> ConnectionState {
        *self.state.read()
    }

    /// Check if connection is active.
    pub fn is_connected(&self) -> bool {
        matches!(self.state(), ConnectionState::Connected)
    }

    /// Set the stream after successful connection.
    pub fn set_stream<S>(&self, stream: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + Sync + 'static,
    {
        let (reader, writer) = split_stream(stream);
        *self.reader.write() = Some(reader);
        *self.writer.write() = Some(writer);
        *self.state.write() = ConnectionState::Connected;
    }

    /// Mark connection as disconnected.
    pub fn mark_disconnected(&self) {
        *self.state.write() = ConnectionState::Disconnected;
        *self.reader.write() = None;
        *self.writer.write() = None;
    }

    /// Mark connection as reconnecting.
    pub fn mark_reconnecting(&self, attempt: usize, next_retry: Instant) {
        *self.state.write() = ConnectionState::Reconnecting {
            attempt,
            next_retry,
        };
        *self.reader.write() = None;
        *self.writer.write() = None;
        self.stats.record_reconnect();
    }

    /// Take the reader (for spawning read task).
    pub fn take_reader(&self) -> Option<DynRead> {
        self.reader.write().take()
    }

    /// Take the writer (for spawning write task).
    pub fn take_writer(&self) -> Option<DynWrite> {
        self.writer.write().take()
    }

    /// Get the codec.
    pub fn codec(&self) -> &LengthDelimitedCodec {
        &self.codec
    }
}

/// Pool of connections to a single peer.
pub struct ConnectionPool {
    /// Peer node ID.
    pub peer_id: NodeId,
    /// Peer address.
    pub address: SocketAddr,
    /// Connections in the pool.
    connections: RwLock<Vec<Arc<Connection>>>,
    /// Target pool size.
    pool_size: usize,
    /// Outbound message queue.
    outbound_tx: mpsc::Sender<Message>,
    /// Outbound message receiver (taken by writer task).
    outbound_rx: RwLock<Option<mpsc::Receiver<Message>>>,
    /// Round-robin index for load balancing.
    next_connection: AtomicUsize,
    /// Connection ID counter.
    connection_counter: AtomicUsize,
    /// Codec for new connections.
    codec: LengthDelimitedCodec,
}

impl ConnectionPool {
    /// Create a new connection pool.
    pub fn new(peer_id: NodeId, address: SocketAddr, config: &TcpTransportConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.max_pending_messages);
        Self {
            peer_id,
            address,
            connections: RwLock::new(Vec::with_capacity(config.pool_size)),
            pool_size: config.pool_size,
            outbound_tx: tx,
            outbound_rx: RwLock::new(Some(rx)),
            next_connection: AtomicUsize::new(0),
            connection_counter: AtomicUsize::new(0),
            codec: LengthDelimitedCodec::new(config.max_message_size),
        }
    }

    /// Add a connection to the pool.
    pub fn add_connection<S>(&self, stream: S) -> Arc<Connection>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + Sync + 'static,
    {
        let id = self.connection_counter.fetch_add(1, Ordering::Relaxed);
        let conn = Arc::new(Connection::new(
            self.peer_id.clone(),
            self.address,
            stream,
            LengthDelimitedCodec::new(self.codec.max_frame_length()),
            id,
        ));
        self.connections.write().push(conn.clone());
        conn
    }

    /// Create a connection in connecting state.
    pub fn add_connecting(&self) -> Arc<Connection> {
        let id = self.connection_counter.fetch_add(1, Ordering::Relaxed);
        let conn = Arc::new(Connection::connecting(
            self.peer_id.clone(),
            self.address,
            LengthDelimitedCodec::new(self.codec.max_frame_length()),
            id,
        ));
        self.connections.write().push(conn.clone());
        conn
    }

    /// Remove a connection from the pool.
    pub fn remove_connection(&self, id: usize) {
        self.connections.write().retain(|c| c.id != id);
    }

    /// Get a connection for sending (round-robin).
    pub fn get_connection(&self) -> Option<Arc<Connection>> {
        let conns = self.connections.read();
        if conns.is_empty() {
            return None;
        }

        // Find connected connections
        let connected: Vec<_> = conns.iter().filter(|c| c.is_connected()).cloned().collect();

        if connected.is_empty() {
            return None;
        }

        let idx = self.next_connection.fetch_add(1, Ordering::Relaxed) % connected.len();
        Some(connected[idx].clone())
    }

    /// Get all connections.
    pub fn connections(&self) -> Vec<Arc<Connection>> {
        self.connections.read().clone()
    }

    /// Get number of active connections.
    pub fn active_count(&self) -> usize {
        self.connections
            .read()
            .iter()
            .filter(|c| c.is_connected())
            .count()
    }

    /// Get total connection count.
    pub fn total_count(&self) -> usize {
        self.connections.read().len()
    }

    /// Queue a message for sending.
    pub fn queue_message(&self, msg: Message) -> TcpResult<()> {
        self.outbound_tx.try_send(msg).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => TcpError::BackpressureFull {
                peer: self.peer_id.clone(),
                queue_size: self.outbound_tx.capacity(),
            },
            mpsc::error::TrySendError::Closed(_) => TcpError::ConnectionClosed,
        })
    }

    /// Take the outbound receiver (for spawning writer task).
    pub fn take_outbound_rx(&self) -> Option<mpsc::Receiver<Message>> {
        self.outbound_rx.write().take()
    }

    /// Check if pool needs more connections.
    pub fn needs_connections(&self) -> bool {
        self.active_count() < self.pool_size
    }

    /// Get target pool size.
    pub fn target_size(&self) -> usize {
        self.pool_size
    }
}

/// Manages all peer connections.
pub struct ConnectionManager {
    /// Local node ID.
    local_id: NodeId,
    /// Connection pools by peer ID.
    pools: RwLock<HashMap<NodeId, Arc<ConnectionPool>>>,
    /// Peer configurations.
    peer_configs: RwLock<HashMap<NodeId, SocketAddr>>,
    /// Transport configuration.
    config: TcpTransportConfig,
}

impl ConnectionManager {
    /// Create a new connection manager.
    pub fn new(config: TcpTransportConfig) -> Self {
        Self {
            local_id: config.node_id.clone(),
            pools: RwLock::new(HashMap::new()),
            peer_configs: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Get or create a connection pool for a peer.
    pub fn get_or_create_pool(&self, peer_id: &NodeId, address: SocketAddr) -> Arc<ConnectionPool> {
        // Check if exists
        if let Some(pool) = self.pools.read().get(peer_id) {
            return pool.clone();
        }

        // Create new pool
        let pool = Arc::new(ConnectionPool::new(peer_id.clone(), address, &self.config));

        self.pools.write().insert(peer_id.clone(), pool.clone());
        self.peer_configs.write().insert(peer_id.clone(), address);

        pool
    }

    /// Get a connection pool for a peer.
    pub fn get_pool(&self, peer_id: &NodeId) -> Option<Arc<ConnectionPool>> {
        self.pools.read().get(peer_id).cloned()
    }

    /// Remove a connection pool.
    pub fn remove_pool(&self, peer_id: &NodeId) {
        self.pools.write().remove(peer_id);
        self.peer_configs.write().remove(peer_id);
    }

    /// Get all peer IDs.
    pub fn peer_ids(&self) -> Vec<NodeId> {
        self.pools.read().keys().cloned().collect()
    }

    /// Get local node ID.
    pub fn local_id(&self) -> &NodeId {
        &self.local_id
    }

    /// Get configuration.
    pub fn config(&self) -> &TcpTransportConfig {
        &self.config
    }

    /// Get peer address.
    pub fn peer_address(&self, peer_id: &NodeId) -> Option<SocketAddr> {
        self.peer_configs.read().get(peer_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_connection_stats() {
        let stats = ConnectionStats::new();

        stats.record_send(100);
        stats.record_send(200);
        stats.record_recv(50);

        assert_eq!(stats.messages_sent.load(Ordering::Relaxed), 2);
        assert_eq!(stats.bytes_sent.load(Ordering::Relaxed), 300);
        assert_eq!(stats.messages_received.load(Ordering::Relaxed), 1);
        assert_eq!(stats.bytes_received.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn test_connection_stats_reconnect() {
        let stats = ConnectionStats::new();
        stats.record_reconnect();
        stats.record_reconnect();
        assert_eq!(stats.reconnections.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_connection_stats_last_activity() {
        let stats = ConnectionStats::new();
        assert_eq!(stats.last_activity.load(Ordering::Relaxed), 0);
        stats.record_send(10);
        assert!(stats.last_activity.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_connection_state_transitions() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());
        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9101".parse().unwrap(),
            &config,
        );

        let conn = pool.add_connecting();
        assert!(matches!(conn.state(), ConnectionState::Connecting));
        assert!(!conn.is_connected());

        conn.mark_disconnected();
        assert!(matches!(conn.state(), ConnectionState::Disconnected));
        assert!(!conn.is_connected());

        conn.mark_reconnecting(1, Instant::now() + Duration::from_secs(1));
        assert!(matches!(conn.state(), ConnectionState::Reconnecting { .. }));
        assert!(!conn.is_connected());
    }

    #[test]
    fn test_connection_pool_basics() {
        let config =
            TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap()).with_pool_size(3);

        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9101".parse().unwrap(),
            &config,
        );

        assert_eq!(pool.target_size(), 3);
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.total_count(), 0);
        assert!(pool.needs_connections());
        assert!(pool.connections().is_empty());
    }

    #[test]
    fn test_connection_pool_add_connecting() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());
        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9101".parse().unwrap(),
            &config,
        );

        let conn1 = pool.add_connecting();
        let conn2 = pool.add_connecting();

        assert_eq!(pool.total_count(), 2);
        assert_eq!(pool.active_count(), 0); // Not yet connected
        assert_ne!(conn1.id, conn2.id);
    }

    #[test]
    fn test_connection_pool_remove() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());
        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9101".parse().unwrap(),
            &config,
        );

        let conn = pool.add_connecting();
        let id = conn.id;
        assert_eq!(pool.total_count(), 1);

        pool.remove_connection(id);
        assert_eq!(pool.total_count(), 0);

        // Removing non-existent is a no-op
        pool.remove_connection(999);
        assert_eq!(pool.total_count(), 0);
    }

    #[test]
    fn test_connection_pool_get_connection_empty() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());
        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9101".parse().unwrap(),
            &config,
        );

        // No connections
        assert!(pool.get_connection().is_none());

        // Add connecting (not connected yet)
        let _conn = pool.add_connecting();
        assert!(pool.get_connection().is_none()); // Still none - not connected
    }

    #[test]
    fn test_connection_pool_take_outbound() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());
        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9101".parse().unwrap(),
            &config,
        );

        // First take succeeds
        let rx = pool.take_outbound_rx();
        assert!(rx.is_some());

        // Second take returns None
        let rx2 = pool.take_outbound_rx();
        assert!(rx2.is_none());
    }

    #[test]
    fn test_connection_manager() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap());
        let manager = ConnectionManager::new(config);

        assert_eq!(manager.local_id(), "node1");
        assert!(manager.peer_ids().is_empty());

        let pool1 =
            manager.get_or_create_pool(&"peer1".to_string(), "127.0.0.1:9101".parse().unwrap());

        let _pool2 =
            manager.get_or_create_pool(&"peer2".to_string(), "127.0.0.1:9102".parse().unwrap());

        assert_eq!(manager.peer_ids().len(), 2);

        // Same peer returns same pool
        let pool1_again =
            manager.get_or_create_pool(&"peer1".to_string(), "127.0.0.1:9101".parse().unwrap());
        assert!(Arc::ptr_eq(&pool1, &pool1_again));

        // Get existing pool
        let pool1_get = manager.get_pool(&"peer1".to_string());
        assert!(pool1_get.is_some());

        // Get non-existent pool
        let pool_none = manager.get_pool(&"unknown".to_string());
        assert!(pool_none.is_none());

        // Get peer address
        let addr = manager.peer_address(&"peer1".to_string());
        assert!(addr.is_some());
        assert_eq!(addr.unwrap().port(), 9101);

        let addr_none = manager.peer_address(&"unknown".to_string());
        assert!(addr_none.is_none());

        manager.remove_pool(&"peer1".to_string());
        assert_eq!(manager.peer_ids().len(), 1);
    }

    #[test]
    fn test_connection_manager_config() {
        let config =
            TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap()).with_pool_size(5);
        let manager = ConnectionManager::new(config);

        assert_eq!(manager.config().pool_size, 5);
    }

    #[test]
    fn test_connection_state_debug() {
        let state = ConnectionState::Connected;
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("Connected"));

        let state = ConnectionState::Reconnecting {
            attempt: 3,
            next_retry: Instant::now(),
        };
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("Reconnecting"));
    }

    #[test]
    fn test_connection_state_equality() {
        assert_eq!(ConnectionState::Connected, ConnectionState::Connected);
        assert_eq!(ConnectionState::Disconnected, ConnectionState::Disconnected);
        assert_ne!(ConnectionState::Connected, ConnectionState::Disconnected);
    }

    #[test]
    fn test_connection_codec() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9100".parse().unwrap())
            .with_max_message_size(1024);
        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9101".parse().unwrap(),
            &config,
        );

        let conn = pool.add_connecting();
        assert_eq!(conn.codec().max_frame_length(), 1024);
    }
}
