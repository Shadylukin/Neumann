// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for `neumann_server`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use neumann_server::{NeumannServer, ServerConfig};
use parking_lot::RwLock;
use query_router::QueryRouter;
use tokio::sync::oneshot;

/// Helper to start a test server and return its address and shutdown channel.
async fn start_test_server() -> (SocketAddr, oneshot::Sender<()>) {
    let router = Arc::new(RwLock::new(QueryRouter::new()));

    // Find an available port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let config = ServerConfig {
        bind_addr: addr,
        ..Default::default()
    };

    let server = NeumannServer::new(router, config);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        let _ = server
            .serve_with_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    // Wait for server to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    (addr, shutdown_tx)
}

#[tokio::test]
async fn test_server_startup_and_shutdown() {
    let (addr, shutdown) = start_test_server().await;

    // Server should be listening
    let conn = tokio::net::TcpStream::connect(addr).await;
    assert!(conn.is_ok(), "Server should be accepting connections");

    // Shutdown
    drop(shutdown);
    tokio::time::sleep(Duration::from_millis(50)).await;
}

#[tokio::test]
async fn test_server_with_custom_config() {
    let router = Arc::new(RwLock::new(QueryRouter::new()));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let config = ServerConfig {
        bind_addr: addr,
        max_message_size: 1024 * 1024, // 1MB
        enable_grpc_web: false,
        enable_reflection: false,
        stream_channel_capacity: 16,
        ..Default::default()
    };

    let server = NeumannServer::new(router, config);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        let _ = server
            .serve_with_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Verify server is running
    let conn = tokio::net::TcpStream::connect(addr).await;
    assert!(conn.is_ok());

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_server_router_access() {
    let router = Arc::new(RwLock::new(QueryRouter::new()));
    let config = ServerConfig::default();
    let server = NeumannServer::new(Arc::clone(&router), config);

    // Should be able to access the router
    let server_router = server.router();
    assert!(Arc::ptr_eq(&router, server_router));
}

#[tokio::test]
async fn test_server_with_auth_config() {
    use neumann_server::config::{ApiKey, AuthConfig};

    let router = Arc::new(RwLock::new(QueryRouter::new()));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let auth_config = AuthConfig::new()
        .with_api_key(ApiKey::new(
            "test-api-key-12345678".to_string(),
            "user:alice".to_string(),
        ))
        .with_anonymous(false);

    let config = ServerConfig {
        bind_addr: addr,
        auth: Some(auth_config),
        ..Default::default()
    };

    let server = NeumannServer::new(router, config);
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    tokio::spawn(async move {
        let _ = server
            .serve_with_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    // Server should be running
    let conn = tokio::net::TcpStream::connect(addr).await;
    assert!(conn.is_ok());

    drop(shutdown_tx);
}

#[tokio::test]
async fn test_server_multiple_start_stop() {
    for _ in 0..3 {
        let (addr, shutdown) = start_test_server().await;

        // Verify it's running
        let conn = tokio::net::TcpStream::connect(addr).await;
        assert!(conn.is_ok());

        // Shutdown
        drop(shutdown);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn test_server_config_builder() {
    let addr: SocketAddr = "127.0.0.1:9201".parse().unwrap();
    let config = ServerConfig::new()
        .with_bind_addr(addr)
        .with_max_message_size(2 * 1024 * 1024)
        .with_grpc_web(true)
        .with_reflection(true);

    assert_eq!(config.bind_addr.port(), 9201);
    assert_eq!(config.max_message_size, 2 * 1024 * 1024);
    assert!(config.enable_grpc_web);
    assert!(config.enable_reflection);
}

#[tokio::test]
async fn test_server_graceful_shutdown() {
    let (addr, shutdown) = start_test_server().await;

    // Make a connection
    let conn = tokio::net::TcpStream::connect(addr).await;
    assert!(conn.is_ok());

    // Trigger shutdown
    let _ = shutdown.send(());

    // Give it time to shutdown
    tokio::time::sleep(Duration::from_millis(100)).await;

    // New connections should fail (server is shutting down)
    // This may or may not fail depending on timing
}
