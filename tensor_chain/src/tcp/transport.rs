//! TCP transport implementation.
//!
//! Implements the `Transport` trait for TCP-based node communication.

use std::{
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use parking_lot::RwLock;
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{broadcast, mpsc},
    task::JoinHandle,
    time::timeout,
};

use super::{
    config::TcpTransportConfig,
    connection::{ConnectionManager, ConnectionPool},
    error::{TcpError, TcpResult},
    framing::{Handshake, LengthDelimitedCodec},
    rate_limit::PeerRateLimiter,
    stream::{box_stream, DynRead, DynStream, DynWrite},
};
use crate::{
    block::NodeId,
    error::{ChainError, Result},
    network::{Message, PeerConfig, Transport},
};

/// TCP-based transport implementation.
pub struct TcpTransport {
    /// Local node ID.
    local_id: NodeId,

    /// Connection manager.
    connections: Arc<ConnectionManager>,

    /// Incoming message channel receiver.
    incoming_rx: tokio::sync::Mutex<mpsc::Receiver<(NodeId, Message)>>,

    /// Incoming message channel sender (for reader tasks).
    incoming_tx: mpsc::Sender<(NodeId, Message)>,

    /// Background task handles.
    tasks: RwLock<Vec<JoinHandle<()>>>,

    /// Shutdown signal sender.
    shutdown_tx: broadcast::Sender<()>,

    /// Running flag.
    running: AtomicBool,

    /// Configuration.
    config: TcpTransportConfig,

    /// Codec for message framing.
    codec: LengthDelimitedCodec,

    /// Bound address after start.
    bound_addr: RwLock<Option<SocketAddr>>,

    /// Per-peer rate limiter.
    rate_limiter: Arc<PeerRateLimiter>,
}

impl TcpTransport {
    /// Create a new TCP transport.
    ///
    /// This does not start listening; call `start()` to begin accepting connections.
    pub fn new(config: TcpTransportConfig) -> Self {
        let (incoming_tx, incoming_rx) = mpsc::channel(config.recv_buffer_size);
        let (shutdown_tx, _) = broadcast::channel(1);
        let rate_limiter = Arc::new(PeerRateLimiter::new(config.rate_limit.clone()));

        Self {
            local_id: config.node_id.clone(),
            connections: Arc::new(ConnectionManager::new(config.clone())),
            incoming_rx: tokio::sync::Mutex::new(incoming_rx),
            incoming_tx,
            tasks: RwLock::new(Vec::new()),
            shutdown_tx,
            running: AtomicBool::new(false),
            codec: LengthDelimitedCodec::new(config.max_message_size),
            config,
            bound_addr: RwLock::new(None),
            rate_limiter,
        }
    }

    pub async fn start(&self) -> TcpResult<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Ok(()); // Already running
        }

        // Bind listener
        let listener = TcpListener::bind(self.config.bind_address).await?;

        // Store the actual bound address
        *self.bound_addr.write() = Some(listener.local_addr()?);

        // Spawn accept loop
        let accept_handle = self.spawn_accept_loop(listener);
        self.tasks.write().push(accept_handle);

        Ok(())
    }

    pub fn bound_addr(&self) -> Option<SocketAddr> {
        *self.bound_addr.read()
    }

    fn spawn_accept_loop(&self, listener: TcpListener) -> JoinHandle<()> {
        let connections = self.connections.clone();
        let incoming_tx = self.incoming_tx.clone();
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let config = self.config.clone();
        let local_id = self.local_id.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                    result = listener.accept() => {
                        match result {
                            Ok((stream, addr)) => {
                                let connections = connections.clone();
                                let incoming_tx = incoming_tx.clone();
                                let config = config.clone();
                                let local_id = local_id.clone();

                                // Handle connection in separate task
                                tokio::spawn(async move {
                                    if let Err(e) = Self::handle_incoming_connection(
                                        stream,
                                        addr,
                                        connections,
                                        incoming_tx,
                                        config,
                                        local_id,
                                    ).await {
                                        tracing::warn!("Failed to handle incoming connection from {}: {}", addr, e);
                                    }
                                });
                            }
                            Err(e) => {
                                tracing::error!("Accept error: {}", e);
                            }
                        }
                    }
                }
            }
        })
    }

    async fn handle_incoming_connection(
        stream: TcpStream,
        addr: SocketAddr,
        connections: Arc<ConnectionManager>,
        incoming_tx: mpsc::Sender<(NodeId, Message)>,
        config: TcpTransportConfig,
        local_id: NodeId,
    ) -> TcpResult<()> {
        // Set socket options
        Self::configure_socket(&stream, &config)?;

        // Upgrade to TLS if configured (server-side)
        #[cfg(feature = "tls")]
        let stream: DynStream = {
            if let Some(tls_config) = &config.tls {
                box_stream(super::tls::wrap_server(stream, tls_config).await?)
            } else {
                box_stream(stream)
            }
        };

        #[cfg(not(feature = "tls"))]
        let stream: DynStream = box_stream(stream);

        // Use a helper to work with the boxed stream
        Self::complete_incoming_handshake(stream, addr, connections, incoming_tx, config, local_id)
            .await
    }

    /// Complete incoming connection handshake with an already-wrapped stream.
    async fn complete_incoming_handshake(
        mut stream: DynStream,
        addr: SocketAddr,
        connections: Arc<ConnectionManager>,
        incoming_tx: mpsc::Sender<(NodeId, Message)>,
        config: TcpTransportConfig,
        local_id: NodeId,
    ) -> TcpResult<()> {
        // Read peer's handshake
        let peer_handshake = timeout(
            Duration::from_millis(config.connect_timeout_ms),
            Handshake::read_from(&mut stream, 4096),
        )
        .await
        .map_err(|_| TcpError::Timeout {
            operation: "handshake read",
            timeout_ms: config.connect_timeout_ms,
        })??;

        let peer_id = peer_handshake.node_id.clone();

        // Send our handshake with timeout, including compression capability if enabled
        let our_handshake = if config.compression.enabled {
            Handshake::new(&local_id).with_compression()
        } else {
            Handshake::new(&local_id)
        };
        our_handshake
            .write_to_with_timeout(&mut stream, config.io_timeout())
            .await?;

        // Negotiate compression: use v2 format if both sides support it
        let use_v2 = our_handshake.compression_negotiated(&peer_handshake);

        // Create codec with compression config
        let mut codec = LengthDelimitedCodec::with_compression(
            config.max_message_size,
            config.compression.clone(),
        );
        codec.set_compression_enabled(use_v2);

        if use_v2 {
            tracing::debug!("Compression negotiated with peer {}", peer_id);
        }

        // Get or create pool for this peer
        let pool = connections.get_or_create_pool(&peer_id, addr);

        // Add connection to pool
        let conn = pool.add_connection(stream);

        // Spawn reader task
        if let Some(reader) = conn.take_reader() {
            let peer_id_clone = peer_id.clone();
            let io_timeout = config.io_timeout();

            tokio::spawn(async move {
                Self::reader_loop(
                    reader,
                    peer_id_clone,
                    incoming_tx,
                    codec,
                    io_timeout,
                    use_v2,
                )
                .await;
            });
        }

        Ok(())
    }

    fn configure_socket(stream: &TcpStream, config: &TcpTransportConfig) -> TcpResult<()> {
        stream.set_nodelay(true)?;

        if config.keepalive {
            let socket = socket2::SockRef::from(stream);
            let keepalive = socket2::TcpKeepalive::new()
                .with_time(Duration::from_secs(config.keepalive_interval_secs));
            socket.set_tcp_keepalive(&keepalive)?;
        }

        Ok(())
    }

    async fn reader_loop(
        mut reader: DynRead,
        peer_id: NodeId,
        incoming_tx: mpsc::Sender<(NodeId, Message)>,
        codec: LengthDelimitedCodec,
        io_timeout: Duration,
        use_v2: bool,
    ) {
        loop {
            let result = if use_v2 {
                codec
                    .read_frame_v2_with_timeout(&mut reader, io_timeout)
                    .await
            } else {
                codec.read_frame_with_timeout(&mut reader, io_timeout).await
            };

            match result {
                Ok(Some(msg)) => {
                    if incoming_tx.send((peer_id.clone(), msg)).await.is_err() {
                        // Channel closed, stop reading
                        break;
                    }
                },
                Ok(None) => {
                    // Connection closed gracefully
                    break;
                },
                Err(TcpError::Timeout { operation, .. }) => {
                    tracing::warn!("Read timeout from {}: {}", peer_id, operation);
                    break;
                },
                Err(e) => {
                    tracing::debug!("Read error from {}: {}", peer_id, e);
                    break;
                },
            }
        }
    }

    async fn writer_loop(
        mut writer: DynWrite,
        mut rx: mpsc::Receiver<Message>,
        codec: LengthDelimitedCodec,
        io_timeout: Duration,
        use_v2: bool,
    ) {
        while let Some(msg) = rx.recv().await {
            let result = if use_v2 {
                codec
                    .write_frame_v2_with_timeout(&mut writer, &msg, io_timeout)
                    .await
            } else {
                codec
                    .write_frame_with_timeout(&mut writer, &msg, io_timeout)
                    .await
            };

            if let Err(e) = result {
                tracing::debug!("Write error: {}", e);
                break;
            }
        }
    }

    async fn connect_to_peer(&self, peer_id: &NodeId, address: SocketAddr) -> TcpResult<()> {
        // Connect with timeout
        let stream = timeout(self.config.connect_timeout(), TcpStream::connect(address))
            .await
            .map_err(|_| TcpError::Timeout {
                operation: "connect",
                timeout_ms: self.config.connect_timeout_ms,
            })??;

        // Configure socket
        Self::configure_socket(&stream, &self.config)?;

        // Upgrade to TLS if configured (client-side)
        #[cfg(feature = "tls")]
        let stream: DynStream = {
            if let Some(tls_config) = &self.config.tls {
                // Use peer_id as server name for TLS verification
                box_stream(super::tls::wrap_client(stream, tls_config, peer_id).await?)
            } else {
                box_stream(stream)
            }
        };

        #[cfg(not(feature = "tls"))]
        let stream: DynStream = box_stream(stream);

        // Get or create pool
        let pool = self.connections.get_or_create_pool(peer_id, address);

        // Perform handshake
        self.perform_handshake(stream, peer_id, pool).await
    }

    async fn perform_handshake(
        &self,
        mut stream: DynStream,
        peer_id: &NodeId,
        pool: Arc<ConnectionPool>,
    ) -> TcpResult<()> {
        // Send our handshake with timeout, including compression capability if enabled
        let handshake = if self.config.compression.enabled {
            Handshake::new(&self.local_id).with_compression()
        } else {
            Handshake::new(&self.local_id)
        };
        handshake
            .write_to_with_timeout(&mut stream, self.config.io_timeout())
            .await?;

        // Read peer's handshake
        let peer_handshake = timeout(
            self.config.connect_timeout(),
            Handshake::read_from(&mut stream, 4096),
        )
        .await
        .map_err(|_| TcpError::Timeout {
            operation: "handshake",
            timeout_ms: self.config.connect_timeout_ms,
        })??;

        // Verify peer ID matches
        if &peer_handshake.node_id != peer_id {
            return Err(TcpError::HandshakeFailed(format!(
                "peer ID mismatch: expected {}, got {}",
                peer_id, peer_handshake.node_id
            )));
        }

        // Negotiate compression: use v2 format if both sides support it
        let use_v2 = handshake.compression_negotiated(&peer_handshake);

        // Create codec with compression config
        let mut codec = LengthDelimitedCodec::with_compression(
            self.config.max_message_size,
            self.config.compression.clone(),
        );
        codec.set_compression_enabled(use_v2);

        if use_v2 {
            tracing::debug!("Compression negotiated with peer {}", peer_id);
        }

        // Add connection to pool
        let conn = pool.add_connection(stream);

        // Spawn reader task
        if let Some(reader) = conn.take_reader() {
            let peer_id = peer_id.clone();
            let incoming_tx = self.incoming_tx.clone();
            let reader_codec = codec.clone();
            let io_timeout = self.config.io_timeout();

            tokio::spawn(async move {
                Self::reader_loop(
                    reader,
                    peer_id,
                    incoming_tx,
                    reader_codec,
                    io_timeout,
                    use_v2,
                )
                .await;
            });
        }

        // Spawn writer task if we have outbound queue
        if let Some(rx) = pool.take_outbound_rx() {
            if let Some(writer) = conn.take_writer() {
                let io_timeout = self.config.io_timeout();

                tokio::spawn(async move {
                    Self::writer_loop(writer, rx, codec, io_timeout, use_v2).await;
                });
            }
        }

        Ok(())
    }

    async fn send_direct(&self, pool: &ConnectionPool, msg: &Message) -> TcpResult<()> {
        // Get a connection
        let conn = pool
            .get_connection()
            .ok_or_else(|| TcpError::PeerNotFound(pool.peer_id.clone()))?;

        // Take writer if available, otherwise queue
        if let Some(mut writer) = conn.take_writer() {
            let result = self.codec.write_frame(&mut writer, msg).await;
            // Note: In a real implementation, we'd return the writer to the connection
            // For now, the connection is one-shot after taking writer
            result
        } else {
            // Queue for writer task
            pool.queue_message(msg.clone())
        }
    }

    pub async fn shutdown(&self) {
        if !self.running.swap(false, Ordering::SeqCst) {
            return; // Already stopped
        }

        // Send shutdown signal
        let _ = self.shutdown_tx.send(());

        // Wait for tasks to complete (with timeout)
        let tasks: Vec<_> = self.tasks.write().drain(..).collect();
        for task in tasks {
            let _ = timeout(Duration::from_secs(5), task).await;
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn stats(&self) -> TransportStats {
        let mut total_sent = 0u64;
        let mut total_recv = 0u64;
        let mut total_bytes_sent = 0u64;
        let mut total_bytes_recv = 0u64;
        let mut peer_count = 0usize;
        let mut connection_count = 0usize;

        for peer_id in self.connections.peer_ids() {
            if let Some(pool) = self.connections.get_pool(&peer_id) {
                peer_count += 1;
                for conn in pool.connections() {
                    connection_count += 1;
                    total_sent += conn.stats.messages_sent.load(Ordering::Relaxed);
                    total_recv += conn.stats.messages_received.load(Ordering::Relaxed);
                    total_bytes_sent += conn.stats.bytes_sent.load(Ordering::Relaxed);
                    total_bytes_recv += conn.stats.bytes_received.load(Ordering::Relaxed);
                }
            }
        }

        TransportStats {
            messages_sent: total_sent,
            messages_received: total_recv,
            bytes_sent: total_bytes_sent,
            bytes_received: total_bytes_recv,
            peer_count,
            connection_count,
        }
    }

    /// Try to receive a single message without blocking.
    /// Returns None if no message is immediately available.
    pub async fn receive_one(&self) -> Result<Option<(NodeId, Message)>> {
        let mut rx = self.incoming_rx.lock().await;
        match rx.try_recv() {
            Ok(msg) => Ok(Some(msg)),
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => Ok(None),
            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                Err(ChainError::NetworkError("transport closed".to_string()))
            },
        }
    }
}

/// Transport statistics.
#[derive(Debug, Clone, Default)]
pub struct TransportStats {
    /// Total messages sent.
    pub messages_sent: u64,
    /// Total messages received.
    pub messages_received: u64,
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Total bytes received.
    pub bytes_received: u64,
    /// Number of connected peers.
    pub peer_count: usize,
    /// Total number of connections.
    pub connection_count: usize,
}

#[async_trait]
impl Transport for TcpTransport {
    async fn send(&self, to: &NodeId, msg: Message) -> Result<()> {
        // Check rate limit before sending
        if !self.rate_limiter.check(to) {
            let available = self.rate_limiter.available_tokens(to);
            return Err(ChainError::NetworkError(format!(
                "rate limited: peer {} (available tokens: {})",
                to, available
            )));
        }

        let pool = self
            .connections
            .get_pool(to)
            .ok_or_else(|| ChainError::NetworkError(format!("peer not found: {}", to)))?;

        self.send_direct(&pool, &msg)
            .await
            .map_err(|e| ChainError::NetworkError(e.to_string()))
    }

    async fn broadcast(&self, msg: Message) -> Result<()> {
        let peer_ids = self.connections.peer_ids();
        let mut errors = Vec::new();

        for peer_id in peer_ids {
            if let Err(e) = self.send(&peer_id, msg.clone()).await {
                errors.push((peer_id, e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            // Log errors but don't fail - broadcast is best-effort
            for (peer, err) in &errors {
                tracing::warn!("Broadcast to {} failed: {}", peer, err);
            }
            Ok(())
        }
    }

    async fn recv(&self) -> Result<(NodeId, Message)> {
        let mut rx = self.incoming_rx.lock().await;
        rx.recv()
            .await
            .ok_or_else(|| ChainError::NetworkError("transport closed".to_string()))
    }

    async fn connect(&self, peer: &PeerConfig) -> Result<()> {
        let address: SocketAddr = peer
            .address
            .parse()
            .map_err(|e| ChainError::NetworkError(format!("invalid address: {}", e)))?;

        self.connect_to_peer(&peer.node_id, address)
            .await
            .map_err(|e| ChainError::NetworkError(e.to_string()))
    }

    async fn disconnect(&self, peer_id: &NodeId) -> Result<()> {
        self.connections.remove_pool(peer_id);
        self.rate_limiter.remove_peer(peer_id);
        Ok(())
    }

    fn peers(&self) -> Vec<NodeId> {
        self.connections.peer_ids()
    }

    fn local_id(&self) -> &NodeId {
        &self.local_id
    }
}

impl Drop for TcpTransport {
    fn drop(&mut self) {
        // Signal shutdown
        let _ = self.shutdown_tx.send(());
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use std::time::Duration;

    use tokio::time::sleep;

    use super::*;

    #[test]
    fn test_transport_stats_default() {
        let stats = TransportStats::default();
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.messages_received, 0);
        assert_eq!(stats.peer_count, 0);
    }

    #[test]
    fn test_transport_stats_clone() {
        let stats = TransportStats {
            messages_sent: 10,
            messages_received: 20,
            bytes_sent: 1000,
            bytes_received: 2000,
            peer_count: 2,
            connection_count: 4,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.messages_sent, 10);
        assert_eq!(cloned.messages_received, 20);
        assert_eq!(cloned.bytes_sent, 1000);
        assert_eq!(cloned.bytes_received, 2000);
        assert_eq!(cloned.peer_count, 2);
        assert_eq!(cloned.connection_count, 4);
    }

    #[test]
    fn test_transport_stats_debug() {
        let stats = TransportStats::default();
        let debug = format!("{:?}", stats);
        assert!(debug.contains("TransportStats"));
        assert!(debug.contains("messages_sent"));
    }

    #[tokio::test]
    async fn test_transport_creation() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        assert_eq!(transport.local_id(), "node1");
        assert!(!transport.is_running());
        assert!(transport.peers().is_empty());
    }

    #[tokio::test]
    async fn test_transport_start_stop() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        // Start
        transport.start().await.unwrap();
        assert!(transport.is_running());

        // Starting again should be idempotent
        transport.start().await.unwrap();
        assert!(transport.is_running());

        // Shutdown
        transport.shutdown().await;
        assert!(!transport.is_running());

        // Shutdown again should be idempotent
        transport.shutdown().await;
        assert!(!transport.is_running());
    }

    #[tokio::test]
    async fn test_send_to_unknown_peer() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        let result = transport
            .send(&"unknown".to_string(), Message::Ping { term: 1 })
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("peer not found"));
    }

    #[tokio::test]
    async fn test_disconnect_unknown_peer() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        // Disconnecting unknown peer should succeed (no-op)
        let result = transport.disconnect(&"unknown".to_string()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_transport_stats() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        let stats = transport.stats();
        assert_eq!(stats.peer_count, 0);
        assert_eq!(stats.connection_count, 0);
    }

    #[tokio::test]
    async fn test_connect_invalid_address() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        let result = transport
            .connect(&PeerConfig {
                node_id: "peer1".to_string(),
                address: "not-a-valid-address".to_string(),
            })
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("invalid address"));
    }

    #[tokio::test]
    async fn test_two_node_communication() {
        // Create two transports
        let config1 = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport1 = Arc::new(TcpTransport::new(config1));

        let config2 = TcpTransportConfig::new("node2", "127.0.0.1:0".parse().unwrap());
        let transport2 = Arc::new(TcpTransport::new(config2));

        // Start both
        transport1.start().await.unwrap();
        transport2.start().await.unwrap();

        // Give listeners time to bind
        sleep(Duration::from_millis(50)).await;

        // Note: In a real test, we'd need to get the actual bound addresses
        // For now, we just verify the transports start successfully
        assert!(transport1.is_running());
        assert!(transport2.is_running());

        // Cleanup
        transport1.shutdown().await;
        transport2.shutdown().await;
    }

    #[tokio::test]
    async fn test_broadcast_no_peers() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        // Broadcasting with no peers should succeed (no-op)
        let result = transport.broadcast(Message::Ping { term: 1 }).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_connect_connection_refused() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap())
            .with_connect_timeout(Duration::from_millis(100)); // Short timeout
        let transport = TcpTransport::new(config);

        // Try to connect to a port that's not listening
        let result = transport
            .connect(&PeerConfig {
                node_id: "peer1".to_string(),
                address: "127.0.0.1:59999".to_string(), // Unlikely to be in use
            })
            .await;

        // Should fail with connection refused or timeout
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_configure_socket() {
        // Create a listener to get a valid stream
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Connect to ourselves
        let connect_future = TcpStream::connect(addr);
        let accept_future = listener.accept();

        let (stream, _) = tokio::join!(connect_future, accept_future);
        let stream = stream.unwrap();

        // Test configuration
        let config = TcpTransportConfig::new("test", addr);
        let result = TcpTransport::configure_socket(&stream, &config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_configure_socket_with_keepalive() {
        // Create a listener to get a valid stream
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Connect to ourselves
        let connect_future = TcpStream::connect(addr);
        let accept_future = listener.accept();

        let (stream, _) = tokio::join!(connect_future, accept_future);
        let stream = stream.unwrap();

        // Test configuration with keepalive
        let config = TcpTransportConfig::new("test", addr)
            .with_keepalive(true)
            .with_keepalive_interval_secs(30);
        let result = TcpTransport::configure_socket(&stream, &config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_reader_loop_connection_closed() {
        // Create connected pair
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let connect_future = TcpStream::connect(addr);
        let accept_future = listener.accept();

        let (client_stream, server_result) = tokio::join!(connect_future, accept_future);
        let client_stream = client_stream.unwrap();
        let (server_stream, _) = server_result.unwrap();

        // Split the server stream and box for dynamic dispatch
        let (reader, _writer) = tokio::io::split(server_stream);
        let reader: DynRead = Box::new(reader);

        // Create a channel to receive messages
        let (tx, mut rx) = mpsc::channel::<(NodeId, Message)>(10);
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let io_timeout = Duration::from_secs(30);

        // Spawn reader loop (v1 format)
        tokio::spawn(async move {
            TcpTransport::reader_loop(reader, "peer1".to_string(), tx, codec, io_timeout, false)
                .await;
        });

        // Close the client side
        drop(client_stream);

        // Reader should exit gracefully (channel won't receive anything)
        sleep(Duration::from_millis(50)).await;
        let msg = rx.try_recv();
        assert!(msg.is_err()); // No message received, reader exited
    }

    #[tokio::test]
    async fn test_writer_loop_channel_closed() {
        // Create connected pair
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let connect_future = TcpStream::connect(addr);
        let accept_future = listener.accept();

        let (client_stream, server_result) = tokio::join!(connect_future, accept_future);
        let _client_stream = client_stream.unwrap();
        let (server_stream, _) = server_result.unwrap();

        // Split the server stream and box for dynamic dispatch
        let (_reader, writer) = tokio::io::split(server_stream);
        let writer: DynWrite = Box::new(writer);

        // Create a channel for outgoing messages
        let (tx, rx) = mpsc::channel::<Message>(10);
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let io_timeout = Duration::from_secs(30);

        // Spawn writer loop (v1 format)
        let handle = tokio::spawn(async move {
            TcpTransport::writer_loop(writer, rx, codec, io_timeout, false).await;
        });

        // Drop sender to close channel
        drop(tx);

        // Writer should exit gracefully
        let result = timeout(Duration::from_millis(100), handle).await;
        assert!(result.is_ok()); // Task completed
    }

    #[tokio::test]
    async fn test_writer_loop_sends_message() {
        // Create connected pair
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let connect_future = TcpStream::connect(addr);
        let accept_future = listener.accept();

        let (client_stream, server_result) = tokio::join!(connect_future, accept_future);
        let client_stream = client_stream.unwrap();
        let (server_stream, _) = server_result.unwrap();

        // Split streams and box for dynamic dispatch
        let (client_reader, _) = tokio::io::split(client_stream);
        let (_, server_writer) = tokio::io::split(server_stream);
        let server_writer: DynWrite = Box::new(server_writer);

        // Create channel and spawn writer
        let (tx, rx) = mpsc::channel::<Message>(10);
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let codec_for_read = LengthDelimitedCodec::new(1024 * 1024);
        let io_timeout = Duration::from_secs(30);

        tokio::spawn(async move {
            TcpTransport::writer_loop(server_writer, rx, codec, io_timeout, false).await;
        });

        // Send a message
        let msg = Message::Ping { term: 42 };
        tx.send(msg).await.unwrap();

        // Read it on client side
        let mut client_reader = client_reader;
        let result = timeout(
            Duration::from_millis(100),
            codec_for_read.read_frame(&mut client_reader),
        )
        .await;

        assert!(result.is_ok());
        let frame_result = result.unwrap();
        assert!(frame_result.is_ok());
        let maybe_msg = frame_result.unwrap();
        assert!(maybe_msg.is_some());
        let received = maybe_msg.unwrap();
        assert!(matches!(received, Message::Ping { term: 42 }));
    }

    #[tokio::test]
    async fn test_transport_drop() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        transport.start().await.unwrap();
        assert!(transport.is_running());

        // Drop should send shutdown signal
        drop(transport);
        // If we got here without hanging, the drop worked
    }

    #[tokio::test]
    async fn test_recv_channel_closed() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());

        // Create transport and immediately drop to close internal channel
        let (incoming_tx, incoming_rx) = mpsc::channel::<(NodeId, Message)>(1);
        drop(incoming_tx); // Close the sender

        // Manually construct a minimal transport to test recv
        let (shutdown_tx, _) = broadcast::channel(1);
        let transport = TcpTransport {
            local_id: "node1".to_string(),
            connections: Arc::new(ConnectionManager::new(config.clone())),
            incoming_rx: tokio::sync::Mutex::new(incoming_rx),
            incoming_tx: mpsc::channel(1).0, // dummy
            tasks: RwLock::new(Vec::new()),
            shutdown_tx,
            running: AtomicBool::new(false),
            codec: LengthDelimitedCodec::new(1024),
            rate_limiter: Arc::new(PeerRateLimiter::new(config.rate_limit.clone())),
            config,
            bound_addr: RwLock::new(None),
        };

        let result = transport.recv().await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("transport closed"));
    }

    #[tokio::test]
    async fn test_send_direct_no_connection() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        // Create a pool without any connections
        let pool = transport
            .connections
            .get_or_create_pool(&"peer1".to_string(), "127.0.0.1:12345".parse().unwrap());

        let msg = Message::Ping { term: 1 };
        let result = transport.send_direct(&pool, &msg).await;

        // Should fail because no connections in pool
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_broadcast_with_failing_peer() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        // Create a pool without connections (will fail to send)
        transport
            .connections
            .get_or_create_pool(&"peer1".to_string(), "127.0.0.1:12345".parse().unwrap());

        // Broadcast should still succeed (best-effort)
        let result = transport.broadcast(Message::Ping { term: 1 }).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_peers_after_pool_creation() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        assert!(transport.peers().is_empty());

        // Create pools for peers
        transport
            .connections
            .get_or_create_pool(&"peer1".to_string(), "127.0.0.1:12345".parse().unwrap());
        transport
            .connections
            .get_or_create_pool(&"peer2".to_string(), "127.0.0.1:12346".parse().unwrap());

        let peers = transport.peers();
        assert_eq!(peers.len(), 2);
        assert!(peers.contains(&"peer1".to_string()));
        assert!(peers.contains(&"peer2".to_string()));
    }

    #[tokio::test]
    async fn test_disconnect_removes_pool() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        // Create a pool
        transport
            .connections
            .get_or_create_pool(&"peer1".to_string(), "127.0.0.1:12345".parse().unwrap());
        assert_eq!(transport.peers().len(), 1);

        // Disconnect
        transport.disconnect(&"peer1".to_string()).await.unwrap();
        assert!(transport.peers().is_empty());
    }

    #[tokio::test]
    async fn test_stats_with_pools() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        // Create pools (without connections)
        transport
            .connections
            .get_or_create_pool(&"peer1".to_string(), "127.0.0.1:12345".parse().unwrap());
        transport
            .connections
            .get_or_create_pool(&"peer2".to_string(), "127.0.0.1:12346".parse().unwrap());

        let stats = transport.stats();
        assert_eq!(stats.peer_count, 2);
        assert_eq!(stats.connection_count, 0); // No actual connections
    }

    #[tokio::test]
    async fn test_handle_incoming_connection() {
        // Create a transport that will accept connections
        let config = TcpTransportConfig::new("server", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config.clone());
        transport.start().await.unwrap();

        // Get the actual bound address - we need to connect manually for this test
        // Since we can't get the address easily, we'll create a listener ourselves
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn a task to accept and handle connection
        let connections = transport.connections.clone();
        let incoming_tx = transport.incoming_tx.clone();
        let config_clone = config.clone();
        let local_id = transport.local_id.clone();

        let server_handle = tokio::spawn(async move {
            let (stream, client_addr) = listener.accept().await.unwrap();
            TcpTransport::handle_incoming_connection(
                stream,
                client_addr,
                connections,
                incoming_tx,
                config_clone,
                local_id,
            )
            .await
        });

        // Connect as client
        let mut client = TcpStream::connect(addr).await.unwrap();

        // Send handshake
        let handshake = Handshake::new("client");
        handshake.write_to(&mut client).await.unwrap();

        // Read server's handshake
        let server_handshake = Handshake::read_from(&mut client, 4096).await.unwrap();
        assert_eq!(server_handshake.node_id, "server");

        // Wait for server to complete
        let result = timeout(Duration::from_millis(200), server_handle).await;
        assert!(result.is_ok());
        assert!(result.unwrap().unwrap().is_ok());

        transport.shutdown().await;
    }

    #[tokio::test]
    async fn test_perform_handshake_peer_id_mismatch() {
        // Create listener
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Server task: send wrong node ID
        let server_handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            // Read client's handshake
            let _client_handshake = Handshake::read_from(&mut stream, 4096).await.unwrap();
            // Send handshake with wrong ID
            let handshake = Handshake::new("wrong_peer");
            handshake.write_to(&mut stream).await.unwrap();
        });

        // Client: create transport and try to connect
        let config = TcpTransportConfig::new("client", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        // Create pool expecting "expected_peer"
        let pool = transport
            .connections
            .get_or_create_pool(&"expected_peer".to_string(), addr);

        // Connect and box the stream for dynamic dispatch
        let stream = TcpStream::connect(addr).await.unwrap();
        let stream: DynStream = box_stream(stream);

        // Perform handshake - should fail due to ID mismatch
        let result = transport
            .perform_handshake(stream, &"expected_peer".to_string(), pool)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, TcpError::HandshakeFailed(_)));
        assert!(err.to_string().contains("peer ID mismatch"));

        // Clean up
        let _ = server_handle.await;
    }

    #[tokio::test]
    async fn test_connect_to_peer_timeout() {
        // Use a non-routable IP to trigger timeout
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap())
            .with_connect_timeout(Duration::from_millis(50)); // Very short timeout
        let transport = TcpTransport::new(config);

        // Try to connect to non-routable address
        let result = transport
            .connect_to_peer(
                &"peer1".to_string(),
                "10.255.255.1:12345".parse().unwrap(), // Non-routable
            )
            .await;

        // Should timeout or fail
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_full_connection_flow() {
        // Start server transport
        let server_config = TcpTransportConfig::new("server", "127.0.0.1:0".parse().unwrap());
        let server = Arc::new(TcpTransport::new(server_config));

        // We need to manually set up a listening server for this test
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let server_addr = listener.local_addr().unwrap();

        // Spawn server accept task
        let server_connections = server.connections.clone();
        let server_incoming_tx = server.incoming_tx.clone();
        let server_config_clone = TcpTransportConfig::new("server", server_addr);

        tokio::spawn(async move {
            while let Ok((stream, addr)) = listener.accept().await {
                let connections = server_connections.clone();
                let incoming_tx = server_incoming_tx.clone();
                let config = server_config_clone.clone();

                tokio::spawn(async move {
                    let _ = TcpTransport::handle_incoming_connection(
                        stream,
                        addr,
                        connections,
                        incoming_tx,
                        config,
                        "server".to_string(),
                    )
                    .await;
                });
            }
        });

        // Create client transport
        let client_config = TcpTransportConfig::new("client", "127.0.0.1:0".parse().unwrap());
        let client = TcpTransport::new(client_config);

        // Connect client to server
        let result = client
            .connect(&PeerConfig {
                node_id: "server".to_string(),
                address: server_addr.to_string(),
            })
            .await;

        assert!(result.is_ok());
        assert!(client.peers().contains(&"server".to_string()));

        // Check stats
        let stats = client.stats();
        assert_eq!(stats.peer_count, 1);
    }

    #[tokio::test]
    async fn test_writer_loop_write_error() {
        // Create connected pair
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let connect_future = TcpStream::connect(addr);
        let accept_future = listener.accept();

        let (client_stream, server_result) = tokio::join!(connect_future, accept_future);
        let client_stream = client_stream.unwrap();
        let (server_stream, _) = server_result.unwrap();

        // Split server stream and box for dynamic dispatch
        let (_, server_writer) = tokio::io::split(server_stream);
        let server_writer: DynWrite = Box::new(server_writer);

        // Create channel with larger buffer
        let (tx, rx) = mpsc::channel::<Message>(100);
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let io_timeout = Duration::from_secs(30);

        // Spawn writer loop (v1 format)
        let handle = tokio::spawn(async move {
            TcpTransport::writer_loop(server_writer, rx, codec, io_timeout, false).await;
        });

        // Drop client immediately to cause broken pipe
        drop(client_stream);

        // Send many messages to fill the socket buffer and trigger EPIPE
        // Socket buffers are typically 64KB-256KB, so sending lots of pings should work
        for i in 0..1000 {
            if tx.send(Message::Ping { term: i }).await.is_err() {
                break; // Channel closed, writer loop exited
            }
            // Small yield to let writer loop process
            if i % 100 == 0 {
                tokio::task::yield_now().await;
            }
        }

        // Writer loop should exit due to write error
        let result = timeout(Duration::from_secs(2), handle).await;
        assert!(result.is_ok(), "Writer loop should exit on write error");
    }

    #[tokio::test]
    async fn test_bound_addr() {
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap());
        let transport = TcpTransport::new(config);

        // Before start, bound_addr is None
        assert!(transport.bound_addr().is_none());

        // Start the transport
        transport.start().await.unwrap();

        // After start, bound_addr should be set
        let addr = transport.bound_addr();
        assert!(addr.is_some());
        let addr = addr.unwrap();
        assert_eq!(addr.ip(), std::net::Ipv4Addr::new(127, 0, 0, 1));
        assert!(addr.port() > 0);

        transport.shutdown().await;
    }

    #[tokio::test]
    async fn test_accept_loop_with_real_connection() {
        // Start server transport
        let server_config = TcpTransportConfig::new("server", "127.0.0.1:0".parse().unwrap());
        let server = Arc::new(TcpTransport::new(server_config));
        server.start().await.unwrap();

        // Get the actual bound address
        let server_addr = server.bound_addr().unwrap();

        // Give the accept loop time to start
        sleep(Duration::from_millis(10)).await;

        // Connect as a client and perform handshake
        let mut client = TcpStream::connect(server_addr).await.unwrap();

        // Send handshake
        let handshake = Handshake::new("client");
        handshake.write_to(&mut client).await.unwrap();

        // Read server's handshake response
        let server_handshake = Handshake::read_from(&mut client, 4096).await.unwrap();
        assert_eq!(server_handshake.node_id, "server");

        // Give the server time to process the connection
        sleep(Duration::from_millis(50)).await;

        // Server should now have the client in its peer list
        assert!(server.peers().contains(&"client".to_string()));

        // Cleanup
        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_accept_loop_message_flow() {
        // Start server transport
        let server_config = TcpTransportConfig::new("server", "127.0.0.1:0".parse().unwrap());
        let server = Arc::new(TcpTransport::new(server_config));
        server.start().await.unwrap();

        let server_addr = server.bound_addr().unwrap();
        sleep(Duration::from_millis(10)).await;

        // Connect as client
        let mut client = TcpStream::connect(server_addr).await.unwrap();

        // Perform handshake
        let handshake = Handshake::new("client");
        handshake.write_to(&mut client).await.unwrap();
        let _ = Handshake::read_from(&mut client, 4096).await.unwrap();

        // Give server time to set up reader
        sleep(Duration::from_millis(50)).await;

        // Send a message from client to server
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let msg = Message::Ping { term: 123 };
        codec.write_frame(&mut client, &msg).await.unwrap();

        // Server should receive the message
        let server_clone = server.clone();
        let recv_result = timeout(Duration::from_millis(500), async {
            server_clone.recv().await
        })
        .await;

        assert!(recv_result.is_ok(), "Should receive message");
        let (peer_id, received_msg) = recv_result.unwrap().unwrap();
        assert_eq!(peer_id, "client");
        assert!(matches!(received_msg, Message::Ping { term: 123 }));

        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_reader_loop_with_messages() {
        // Create connected pair
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let connect_future = TcpStream::connect(addr);
        let accept_future = listener.accept();

        let (client_stream, server_result) = tokio::join!(connect_future, accept_future);
        let mut client_stream = client_stream.unwrap();
        let (server_stream, _) = server_result.unwrap();

        // Split the server stream and box for dynamic dispatch
        let (reader, _writer) = tokio::io::split(server_stream);
        let reader: DynRead = Box::new(reader);

        // Create a channel to receive messages
        let (tx, mut rx) = mpsc::channel::<(NodeId, Message)>(10);
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let io_timeout = Duration::from_secs(30);

        // Spawn reader loop (v1 format)
        tokio::spawn(async move {
            TcpTransport::reader_loop(reader, "peer1".to_string(), tx, codec, io_timeout, false)
                .await;
        });

        // Send a message from client
        let send_codec = LengthDelimitedCodec::new(1024 * 1024);
        let msg = Message::Ping { term: 42 };
        send_codec
            .write_frame(&mut client_stream, &msg)
            .await
            .unwrap();

        // Receive it
        let result = timeout(Duration::from_millis(100), rx.recv()).await;
        assert!(result.is_ok());
        let (peer_id, received) = result.unwrap().unwrap();
        assert_eq!(peer_id, "peer1");
        assert!(matches!(received, Message::Ping { term: 42 }));

        // Send another message
        let msg2 = Message::Pong { term: 99 };
        send_codec
            .write_frame(&mut client_stream, &msg2)
            .await
            .unwrap();

        let result2 = timeout(Duration::from_millis(100), rx.recv()).await;
        assert!(result2.is_ok());
        let (_, received2) = result2.unwrap().unwrap();
        assert!(matches!(received2, Message::Pong { term: 99 }));
    }

    #[tokio::test]
    async fn test_reader_loop_read_error() {
        // Create connected pair
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let connect_future = TcpStream::connect(addr);
        let accept_future = listener.accept();

        let (client_stream, server_result) = tokio::join!(connect_future, accept_future);
        let mut client_stream = client_stream.unwrap();
        let (server_stream, _) = server_result.unwrap();

        // Split the server stream and box for dynamic dispatch
        let (reader, _writer) = tokio::io::split(server_stream);
        let reader: DynRead = Box::new(reader);

        let (tx, mut rx) = mpsc::channel::<(NodeId, Message)>(10);
        let codec = LengthDelimitedCodec::new(1024 * 1024);
        let io_timeout = Duration::from_secs(30);

        let handle = tokio::spawn(async move {
            TcpTransport::reader_loop(reader, "peer1".to_string(), tx, codec, io_timeout, false)
                .await;
        });

        // Send invalid data to cause a read error
        // Write a valid length header but invalid payload
        use tokio::io::AsyncWriteExt;
        let bad_length: [u8; 4] = 100u32.to_be_bytes();
        client_stream.write_all(&bad_length).await.unwrap();
        client_stream.write_all(&[0xff; 50]).await.unwrap(); // Invalid bincode data
        client_stream.shutdown().await.unwrap();

        // Reader loop should exit due to error
        let result = timeout(Duration::from_millis(500), handle).await;
        assert!(result.is_ok(), "Reader loop should exit on error");

        // No valid message should have been received
        let recv_result = rx.try_recv();
        assert!(recv_result.is_err());
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        use crate::tcp::rate_limit::RateLimitConfig;

        // Create transport with very aggressive rate limiting
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap())
            .with_rate_limit(
                RateLimitConfig::default()
                    .with_bucket_size(3)
                    .with_refill_rate(0.0),
            );
        let transport = TcpTransport::new(config);

        // Create a pool without connections (will fail to send)
        transport
            .connections
            .get_or_create_pool(&"peer1".to_string(), "127.0.0.1:12345".parse().unwrap());

        // First 3 messages should pass rate limit check (but fail on send)
        for _ in 0..3 {
            let result = transport
                .send(&"peer1".to_string(), Message::Ping { term: 1 })
                .await;
            // Should fail due to no connection, not rate limiting
            assert!(result.is_err());
            assert!(!result.unwrap_err().to_string().contains("rate limited"));
        }

        // 4th message should be rate limited
        let result = transport
            .send(&"peer1".to_string(), Message::Ping { term: 1 })
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("rate limited"));
    }

    #[tokio::test]
    async fn test_rate_limiting_disabled() {
        use crate::tcp::rate_limit::RateLimitConfig;

        // Create transport with rate limiting disabled
        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap())
            .with_rate_limit(RateLimitConfig::disabled());
        let transport = TcpTransport::new(config);

        // Create a pool without connections
        transport
            .connections
            .get_or_create_pool(&"peer1".to_string(), "127.0.0.1:12345".parse().unwrap());

        // Should never be rate limited
        for _ in 0..100 {
            let result = transport
                .send(&"peer1".to_string(), Message::Ping { term: 1 })
                .await;
            // Should fail due to no connection, not rate limiting
            assert!(result.is_err());
            assert!(!result.unwrap_err().to_string().contains("rate limited"));
        }
    }

    #[tokio::test]
    async fn test_disconnect_clears_rate_limit() {
        use crate::tcp::rate_limit::RateLimitConfig;

        let config = TcpTransportConfig::new("node1", "127.0.0.1:0".parse().unwrap())
            .with_rate_limit(
                RateLimitConfig::default()
                    .with_bucket_size(2)
                    .with_refill_rate(0.0),
            );
        let transport = TcpTransport::new(config);

        // Create a pool
        transport
            .connections
            .get_or_create_pool(&"peer1".to_string(), "127.0.0.1:12345".parse().unwrap());

        // Exhaust rate limit
        for _ in 0..2 {
            let _ = transport
                .send(&"peer1".to_string(), Message::Ping { term: 1 })
                .await;
        }

        // Should be rate limited now
        let result = transport
            .send(&"peer1".to_string(), Message::Ping { term: 1 })
            .await;
        assert!(result.unwrap_err().to_string().contains("rate limited"));

        // Disconnect
        transport.disconnect(&"peer1".to_string()).await.unwrap();

        // Re-create pool
        transport
            .connections
            .get_or_create_pool(&"peer1".to_string(), "127.0.0.1:12345".parse().unwrap());

        // Should have fresh rate limit bucket
        let result = transport
            .send(&"peer1".to_string(), Message::Ping { term: 1 })
            .await;
        // Should fail due to no connection, not rate limiting
        assert!(result.is_err());
        assert!(!result.unwrap_err().to_string().contains("rate limited"));
    }
}
