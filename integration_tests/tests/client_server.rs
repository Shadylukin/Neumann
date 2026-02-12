// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for neumann_client connecting to neumann_server.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use neumann_client::{ClientError, NeumannClient};
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
    let (shutdown_tx, shutdown_rx) = oneshot::channel();

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
async fn test_client_connect_and_execute() {
    let (addr, shutdown) = start_test_server().await;

    // Connect client
    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    assert!(client.is_connected());

    // Execute query
    let result = client.execute("CREATE TABLE test (x:int)").await;
    assert!(result.is_ok(), "Query should succeed: {:?}", result.err());

    let result = result.unwrap();
    assert!(!result.has_error());
    assert!(result.is_empty());

    // Cleanup
    drop(shutdown);
}

#[tokio::test]
async fn test_client_execute_with_identity() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    let result = client
        .execute_with_identity("CREATE TABLE id_test (x:int)", Some("user:alice"))
        .await;
    assert!(result.is_ok());

    drop(shutdown);
}

#[tokio::test]
async fn test_client_execute_batch() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Create table first
    let _ = client.execute("CREATE TABLE batch (x:int)").await;

    // Execute batch
    let results = client
        .execute_batch(&["INSERT batch x=1", "INSERT batch x=2", "SELECT batch"])
        .await
        .expect("batch should succeed");

    assert_eq!(results.len(), 3);

    drop(shutdown);
}

#[tokio::test]
async fn test_client_execute_batch_with_identity() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    let results = client
        .execute_batch_with_identity(&["CREATE TABLE batch_id (x:int)"], Some("user:bob"))
        .await
        .expect("batch should succeed");

    assert_eq!(results.len(), 1);

    drop(shutdown);
}

#[tokio::test]
async fn test_client_query_error() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Invalid query should return error in result
    let result = client.execute("INVALID QUERY!!!").await;
    assert!(result.is_err());

    drop(shutdown);
}

#[tokio::test]
async fn test_client_select_rows() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Setup
    let _ = client
        .execute("CREATE TABLE users (name:string, age:int)")
        .await;
    let _ = client.execute("INSERT users name=\"Alice\", age=30").await;
    let _ = client.execute("INSERT users name=\"Bob\", age=25").await;

    // Select
    let result = client.execute("SELECT users").await.unwrap();
    assert!(!result.has_error());
    assert!(!result.is_empty());
    assert!(result.rows().is_some());
    assert_eq!(result.rows().unwrap().len(), 2);

    drop(shutdown);
}

#[tokio::test]
async fn test_client_delete() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Create and populate table
    let _ = client.execute("CREATE TABLE del_test (x:int)").await;
    let _ = client.execute("INSERT del_test x=1").await;
    let _ = client.execute("INSERT del_test x=2").await;

    // Verify rows exist
    let result = client.execute("SELECT del_test").await.unwrap();
    assert_eq!(result.rows().unwrap().len(), 2);

    // Delete one row
    let result = client.execute("DELETE del_test WHERE x = 1").await;
    assert!(result.is_ok());

    // Verify only one row remains
    let result = client.execute("SELECT del_test").await.unwrap();
    assert_eq!(result.rows().unwrap().len(), 1);

    drop(shutdown);
}

#[tokio::test]
async fn test_client_connection_refused() {
    // Try to connect to a port that's not listening
    let result: Result<NeumannClient, ClientError> = NeumannClient::connect("127.0.0.1:59999")
        .timeout_ms(1000)
        .build()
        .await;

    assert!(result.is_err());
    match &result {
        Err(ClientError::Connection(_)) => {},
        Err(e) => panic!("Expected Connection error, got {}", e),
        Ok(_) => panic!("Expected error, got Ok"),
    }
}

#[tokio::test]
async fn test_client_with_api_key() {
    let (addr, shutdown) = start_test_server().await;

    // Server doesn't require auth, but client can still send key
    let client: NeumannClient = NeumannClient::connect(addr.to_string())
        .api_key("test-key-12345678")
        .build()
        .await
        .expect("should connect");

    let result = client.execute("CREATE TABLE api_test (x:int)").await;
    assert!(result.is_ok());

    drop(shutdown);
}

#[tokio::test]
async fn test_client_close() {
    let (addr, shutdown) = start_test_server().await;

    let mut client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    assert!(client.is_connected());

    client.close();
    assert!(!client.is_connected());

    // Execute after close should fail
    let result = client.execute("SELECT test").await;
    assert!(result.is_err());

    drop(shutdown);
}

#[tokio::test]
async fn test_embedded_client_sync_operations() {
    let client = NeumannClient::embedded().expect("should create embedded client");

    // Create table
    let result = client.execute_sync("CREATE TABLE embed_test (x:int)");
    assert!(result.is_ok());

    // Insert
    let result = client.execute_sync("INSERT embed_test x=42");
    assert!(result.is_ok());

    // Select
    let result = client.execute_sync("SELECT embed_test");
    assert!(result.is_ok());
}
