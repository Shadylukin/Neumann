// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Connection management for TCP transport.
//!
//! Provides connection pooling and state management for peer connections.

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use parking_lot::RwLock;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    sync::mpsc,
};

use super::{
    config::TcpTransportConfig,
    error::{TcpError, TcpResult},
    framing::LengthDelimitedCodec,
    stream::{split_stream, DynRead, DynWrite},
};
use crate::{block::NodeId, network::Message};

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
    #[must_use]
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
        #[allow(clippy::cast_possible_truncation)]
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
    #[must_use]
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

    pub fn state(&self) -> ConnectionState {
        *self.state.read()
    }

    pub fn is_connected(&self) -> bool {
        matches!(self.state(), ConnectionState::Connected)
    }

    /// Replace the stream after a reconnection handshake.
    ///
    /// Atomicity guarantee: reader, writer, and state are all updated while
    /// holding their write locks simultaneously. This prevents a concurrent
    /// observer from seeing a partially-updated connection (e.g. new reader
    /// but stale writer) during the handshake-to-connected transition.
    pub fn set_stream<S>(&self, stream: S)
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + Sync + 'static,
    {
        let (reader, writer) = split_stream(stream);
        // Acquire all three locks before mutating any field.
        let mut reader_guard = self.reader.write();
        let mut writer_guard = self.writer.write();
        let mut state_guard = self.state.write();
        *reader_guard = Some(reader);
        *writer_guard = Some(writer);
        *state_guard = ConnectionState::Connected;
    }

    pub fn mark_disconnected(&self) {
        let mut state_guard = self.state.write();
        let mut reader_guard = self.reader.write();
        let mut writer_guard = self.writer.write();
        *state_guard = ConnectionState::Disconnected;
        *reader_guard = None;
        *writer_guard = None;
    }

    pub fn mark_reconnecting(&self, attempt: usize, next_retry: Instant) {
        let mut state_guard = self.state.write();
        let mut reader_guard = self.reader.write();
        let mut writer_guard = self.writer.write();
        *state_guard = ConnectionState::Reconnecting {
            attempt,
            next_retry,
        };
        *reader_guard = None;
        *writer_guard = None;
        self.stats.record_reconnect();
    }

    pub fn take_reader(&self) -> Option<DynRead> {
        self.reader.write().take()
    }

    pub fn take_writer(&self) -> Option<DynWrite> {
        self.writer.write().take()
    }

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
    #[must_use]
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

    pub fn remove_connection(&self, id: usize) {
        self.connections.write().retain(|c| c.id != id);
    }

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

    pub fn connections(&self) -> Vec<Arc<Connection>> {
        self.connections.read().clone()
    }

    pub fn active_count(&self) -> usize {
        self.connections
            .read()
            .iter()
            .filter(|c| c.is_connected())
            .count()
    }

    pub fn total_count(&self) -> usize {
        self.connections.read().len()
    }

    /// # Errors
    /// Returns an error if the outbound queue is full or the connection is closed.
    pub fn queue_message(&self, msg: Message) -> TcpResult<()> {
        self.outbound_tx.try_send(msg).map_err(|e| match e {
            mpsc::error::TrySendError::Full(_) => TcpError::BackpressureFull {
                peer: self.peer_id.clone(),
                queue_size: self.outbound_tx.capacity(),
            },
            mpsc::error::TrySendError::Closed(_) => TcpError::ConnectionClosed,
        })
    }

    pub fn take_outbound_rx(&self) -> Option<mpsc::Receiver<Message>> {
        self.outbound_rx.write().take()
    }

    pub fn needs_connections(&self) -> bool {
        self.active_count() < self.pool_size
    }

    pub fn target_size(&self) -> usize {
        self.pool_size
    }

    /// Get a connection or return a `PoolExhausted` error.
    ///
    /// Unlike `get_connection()` which returns `None`, this returns a typed
    /// error with diagnostic information about pool state.
    ///
    /// # Errors
    /// Returns a `PoolExhausted` error if no healthy connections are available.
    pub fn get_connection_or_error(&self) -> TcpResult<Arc<Connection>> {
        self.get_connection()
            .ok_or_else(|| TcpError::PoolExhausted {
                peer: self.peer_id.clone(),
                active: self.active_count(),
                target: self.pool_size,
            })
    }

    /// Sweep the pool for unhealthy connections and remove them.
    ///
    /// Removes connections in `Disconnected` state and stale connections
    /// that have had no activity beyond the given inactivity timeout.
    /// Returns pool health status with reconnection needs.
    pub fn health_sweep(&self, inactivity_timeout: std::time::Duration) -> PoolHealthStatus {
        let mut conns = self.connections.write();
        let before = conns.len();

        #[allow(clippy::cast_possible_truncation)]
        let timeout_ms = inactivity_timeout.as_millis() as u64;
        #[allow(clippy::cast_possible_truncation)]
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        conns.retain(|conn| {
            match conn.state() {
                ConnectionState::Disconnected => false,
                ConnectionState::Connecting | ConnectionState::Reconnecting { .. } => {
                    // Keep connecting/reconnecting unless inactive too long
                    let last = conn.stats.last_activity.load(Ordering::Relaxed);
                    if last == 0 {
                        // Never had activity - keep it (still trying to connect)
                        true
                    } else {
                        // Had activity before - check for staleness
                        now_ms.saturating_sub(last) < timeout_ms
                    }
                },
                ConnectionState::Connected => true,
            }
        });

        let after = conns.len();
        let active = conns.iter().filter(|c| c.is_connected()).count();

        PoolHealthStatus {
            removed: before - after,
            active,
            total: after,
            target: self.pool_size,
            needs_reconnect: active < self.pool_size,
        }
    }
}

/// Result of a connection pool health sweep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolHealthStatus {
    /// Number of dead connections removed.
    pub removed: usize,
    /// Number of currently active (connected) connections.
    pub active: usize,
    /// Total connections remaining after sweep.
    pub total: usize,
    /// Target pool size.
    pub target: usize,
    /// Whether the pool needs new connections to reach target.
    pub needs_reconnect: bool,
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
    #[must_use]
    pub fn new(config: TcpTransportConfig) -> Self {
        Self {
            local_id: config.node_id.clone(),
            pools: RwLock::new(HashMap::new()),
            peer_configs: RwLock::new(HashMap::new()),
            config,
        }
    }

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

    pub fn get_pool(&self, peer_id: &NodeId) -> Option<Arc<ConnectionPool>> {
        self.pools.read().get(peer_id).cloned()
    }

    pub fn remove_pool(&self, peer_id: &NodeId) {
        self.pools.write().remove(peer_id);
        self.peer_configs.write().remove(peer_id);
    }

    pub fn peer_ids(&self) -> Vec<NodeId> {
        self.pools.read().keys().cloned().collect()
    }

    pub fn local_id(&self) -> &NodeId {
        &self.local_id
    }

    pub fn config(&self) -> &TcpTransportConfig {
        &self.config
    }

    pub fn peer_address(&self, peer_id: &NodeId) -> Option<SocketAddr> {
        self.peer_configs.read().get(peer_id).copied()
    }

    /// Run a health sweep across all connection pools.
    ///
    /// Removes dead connections and returns a list of peer IDs whose pools
    /// need reconnection (below target size). Intended to be called
    /// periodically (e.g., every 10 seconds).
    pub fn health_sweep_all(
        &self,
        stale_timeout: std::time::Duration,
    ) -> Vec<(NodeId, PoolHealthStatus)> {
        let pools = self.pools.read();
        let mut results = Vec::new();

        for (peer_id, pool) in pools.iter() {
            let status = pool.health_sweep(stale_timeout);
            if status.removed > 0 || status.needs_reconnect {
                tracing::debug!(
                    peer = %peer_id,
                    removed = status.removed,
                    active = status.active,
                    target = status.target,
                    "Pool health sweep"
                );
            }
            results.push((peer_id.clone(), status));
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

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

    #[test]
    fn test_health_sweep_removes_disconnected() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9200".parse().unwrap());
        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9201".parse().unwrap(),
            &config,
        );

        // Add connections in various states
        let conn1 = pool.add_connecting();
        let conn2 = pool.add_connecting();
        conn2.mark_disconnected();

        assert_eq!(pool.total_count(), 2);

        let status = pool.health_sweep(Duration::from_secs(30));

        // Disconnected connection should be removed
        assert_eq!(status.removed, 1);
        assert_eq!(status.total, 1);
        assert_eq!(pool.total_count(), 1);
        // conn1 is still Connecting, not active
        assert_eq!(status.active, 0);
        assert!(status.needs_reconnect);
        let _ = conn1; // keep alive
    }

    #[test]
    fn test_health_sweep_empty_pool() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9202".parse().unwrap());
        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9203".parse().unwrap(),
            &config,
        );

        let status = pool.health_sweep(Duration::from_secs(10));
        assert_eq!(status.removed, 0);
        assert_eq!(status.active, 0);
        assert!(status.needs_reconnect);
    }

    #[test]
    fn test_get_connection_or_error() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9204".parse().unwrap());
        let pool = ConnectionPool::new(
            "peer1".to_string(),
            "127.0.0.1:9205".parse().unwrap(),
            &config,
        );

        // Empty pool should return PoolExhausted error
        let result = pool.get_connection_or_error();
        assert!(result.is_err());
        let err = result.err().unwrap();
        match err {
            TcpError::PoolExhausted {
                peer,
                active,
                target,
            } => {
                assert_eq!(peer, "peer1");
                assert_eq!(active, 0);
                assert_eq!(target, config.pool_size);
            },
            other => panic!("Expected PoolExhausted, got {other:?}"),
        }
    }

    #[test]
    fn test_pool_health_status_fields() {
        let status = PoolHealthStatus {
            removed: 2,
            active: 3,
            total: 5,
            target: 4,
            needs_reconnect: false,
        };
        assert_eq!(status.removed, 2);
        assert_eq!(status.active, 3);
        assert_eq!(status.total, 5);
        assert_eq!(status.target, 4);
        assert!(!status.needs_reconnect);

        // Debug and Clone
        let debug = format!("{status:?}");
        assert!(debug.contains("PoolHealthStatus"));
        let cloned = status;
        assert_eq!(cloned, status);
    }

    #[test]
    fn test_pool_exhausted_error_display() {
        let err = TcpError::PoolExhausted {
            peer: "node2".to_string(),
            active: 0,
            target: 3,
        };
        let display = err.to_string();
        assert!(display.contains("pool exhausted"));
        assert!(display.contains("node2"));
        assert!(display.contains("0/3"));
    }

    #[test]
    fn test_health_sweep_all_on_manager() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:9206".parse().unwrap());
        let manager = ConnectionManager::new(config);

        let pool =
            manager.get_or_create_pool(&"peer1".to_string(), "127.0.0.1:9207".parse().unwrap());

        // Add and disconnect a connection
        let conn = pool.add_connecting();
        conn.mark_disconnected();

        let results = manager.health_sweep_all(Duration::from_secs(10));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "peer1");
        assert_eq!(results[0].1.removed, 1);
    }
}
