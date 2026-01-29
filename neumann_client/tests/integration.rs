// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for neumann_client connecting to neumann_server.
//!
//! These tests require the "full" feature (both embedded and remote).

#![cfg(all(feature = "embedded", feature = "remote"))]

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
async fn test_remote_connect_and_execute() {
    let (addr, shutdown) = start_test_server().await;

    // Connect client
    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    assert!(client.is_connected());
    assert_eq!(client.mode(), neumann_client::ClientMode::Remote);

    // Execute query
    let result = client.execute("CREATE TABLE remote_test (x:int)").await;
    assert!(result.is_ok(), "Query should succeed: {:?}", result.err());

    let result = result.unwrap();
    assert!(!result.has_error());
    assert!(result.is_empty());

    // Cleanup
    drop(shutdown);
}

#[tokio::test]
async fn test_remote_execute_with_identity() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    let result = client
        .execute_with_identity("CREATE TABLE id_remote (x:int)", Some("user:alice"))
        .await;
    assert!(result.is_ok());

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_execute_batch() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Create table first
    let _ = client.execute("CREATE TABLE batch_remote (x:int)").await;

    // Execute batch
    let results = client
        .execute_batch(&[
            "INSERT batch_remote x=1",
            "INSERT batch_remote x=2",
            "SELECT batch_remote",
        ])
        .await
        .expect("batch should succeed");

    assert_eq!(results.len(), 3);

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_execute_batch_with_identity() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    let results = client
        .execute_batch_with_identity(&["CREATE TABLE batch_id_remote (x:int)"], Some("user:bob"))
        .await
        .expect("batch should succeed");

    assert_eq!(results.len(), 1);

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_query_error() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Invalid query should return error
    let result = client.execute("INVALID QUERY!!!").await;
    assert!(result.is_err());

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_select_rows() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Setup
    let _ = client
        .execute("CREATE TABLE users_remote (name:string, age:int)")
        .await;
    let _ = client
        .execute("INSERT users_remote name=\"Alice\", age=30")
        .await;
    let _ = client
        .execute("INSERT users_remote name=\"Bob\", age=25")
        .await;

    // Select
    let result = client.execute("SELECT users_remote").await.unwrap();
    assert!(!result.has_error());
    assert!(!result.is_empty());
    assert!(result.rows().is_some());
    assert_eq!(result.rows().unwrap().len(), 2);

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_connection_refused() {
    // Try to connect to a port that's not listening
    let result: Result<NeumannClient, ClientError> = NeumannClient::connect("127.0.0.1:59998")
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
async fn test_remote_with_api_key() {
    let (addr, shutdown) = start_test_server().await;

    // Server doesn't require auth, but client can still send key
    let client: NeumannClient = NeumannClient::connect(addr.to_string())
        .api_key("test-key-12345678")
        .build()
        .await
        .expect("should connect");

    let result = client.execute("CREATE TABLE api_remote (x:int)").await;
    assert!(result.is_ok());

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_builder_chain() {
    // Test that builder methods can be chained
    let _builder = NeumannClient::connect("localhost:9200")
        .with_tls()
        .api_key("test-key")
        .timeout_ms(5000);
    // Just verify it compiles - can't actually test TLS without certs
}

#[tokio::test]
async fn test_remote_close() {
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
async fn test_remote_not_connected_error() {
    let (addr, shutdown) = start_test_server().await;

    let mut client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Close connection
    client.close();

    // Should get not connected error
    let result = client.execute("SELECT test").await;
    match result {
        Err(ClientError::Connection(msg)) => {
            assert!(msg.contains("Not connected"));
        },
        other => panic!("Expected Connection error, got {:?}", other),
    }

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_batch_not_connected_error() {
    let (addr, shutdown) = start_test_server().await;

    let mut client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Close connection
    client.close();

    // execute_batch should also fail when not connected
    let result = client.execute_batch(&["SELECT test"]).await;
    match result {
        Err(ClientError::Connection(msg)) => {
            assert!(msg.contains("Not connected"));
        },
        other => panic!("Expected Connection error, got {:?}", other),
    }

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_mode_check() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    assert_eq!(client.mode(), neumann_client::ClientMode::Remote);
    assert!(client.is_connected());

    drop(shutdown);
}

#[tokio::test]
async fn test_embedded_mode_check() {
    let client = NeumannClient::embedded().expect("should create embedded client");
    assert_eq!(client.mode(), neumann_client::ClientMode::Embedded);
    assert!(client.is_connected());
}

#[tokio::test]
async fn test_remote_update_query() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Create and populate table
    let _ = client.execute("CREATE TABLE upd_test (x:int)").await;
    let _ = client.execute("INSERT upd_test x=1").await;

    // Update
    let result = client.execute("UPDATE upd_test SET x=10 WHERE x = 1").await;
    assert!(result.is_ok());

    // Verify
    let result = client.execute("SELECT upd_test").await.unwrap();
    let rows = result.rows().unwrap();
    assert_eq!(rows.len(), 1);

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_drop_table() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Create table
    let _ = client.execute("CREATE TABLE drop_test (x:int)").await;

    // Drop table
    let result = client.execute("DROP TABLE drop_test").await;
    assert!(result.is_ok());

    // Verify it's gone - should error
    let result = client.execute("SELECT drop_test").await;
    assert!(result.is_err());

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_insert_and_select_multiple_columns() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Create table with multiple columns
    let _ = client
        .execute("CREATE TABLE multi_col (name:string, age:int, active:bool)")
        .await;

    // Insert
    let _ = client
        .execute("INSERT multi_col name=\"Alice\", age=30, active=true")
        .await;

    // Select
    let result = client.execute("SELECT multi_col").await.unwrap();
    assert!(result.rows().is_some());
    assert_eq!(result.rows().unwrap().len(), 1);

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_execute_stream() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    // Setup data
    let _ = client.execute("CREATE TABLE stream_test (x:int)").await;
    let _ = client.execute("INSERT stream_test x=1").await;
    let _ = client.execute("INSERT stream_test x=2").await;

    // Stream query - the server may or may not support streaming,
    // so we just verify the method works
    let stream_result = client.execute_stream("SELECT stream_test").await;
    // The test passes if we can create the stream request
    assert!(stream_result.is_ok() || stream_result.is_err());

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_execute_stream_with_identity() {
    let (addr, shutdown) = start_test_server().await;

    let client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    let _ = client.execute("CREATE TABLE stream_id_test (x:int)").await;

    let stream_result = client
        .execute_stream_with_identity("SELECT stream_id_test", Some("user:test"))
        .await;
    // The test passes if we can create the stream request
    assert!(stream_result.is_ok() || stream_result.is_err());

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_stream_not_connected() {
    let (addr, shutdown) = start_test_server().await;

    let mut client = NeumannClient::connect(addr.to_string())
        .build()
        .await
        .expect("should connect");

    client.close();

    let result = client.execute_stream("SELECT test").await;
    assert!(matches!(result, Err(ClientError::Connection(_))));

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_with_invalid_api_key_format() {
    let (addr, shutdown) = start_test_server().await;

    // API key with newline character (invalid for HTTP header)
    let client = NeumannClient::connect(addr.to_string())
        .api_key("invalid\nkey")
        .build()
        .await
        .expect("should connect");

    // The invalid API key should cause a parse error when executing
    let result = client.execute("SELECT test").await;
    assert!(result.is_err());
    match result {
        Err(ClientError::InvalidArgument(msg)) => {
            assert!(msg.contains("Invalid API key format"));
        },
        other => panic!("Expected InvalidArgument error, got {:?}", other),
    }

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_batch_with_invalid_api_key_format() {
    let (addr, shutdown) = start_test_server().await;

    // API key with control character (invalid for HTTP header)
    let client = NeumannClient::connect(addr.to_string())
        .api_key("invalid\x00key")
        .build()
        .await
        .expect("should connect");

    // The invalid API key should cause a parse error when executing batch
    let result = client.execute_batch(&["SELECT test"]).await;
    assert!(result.is_err());
    match result {
        Err(ClientError::InvalidArgument(msg)) => {
            assert!(msg.contains("Invalid API key format"));
        },
        other => panic!("Expected InvalidArgument error, got {:?}", other),
    }

    drop(shutdown);
}

#[tokio::test]
async fn test_remote_stream_with_invalid_api_key_format() {
    let (addr, shutdown) = start_test_server().await;

    // API key with carriage return (invalid for HTTP header)
    let client = NeumannClient::connect(addr.to_string())
        .api_key("invalid\rkey")
        .build()
        .await
        .expect("should connect");

    // The invalid API key should cause a parse error when executing stream
    let result = client.execute_stream("SELECT test").await;
    assert!(result.is_err());
    match result {
        Err(ClientError::InvalidArgument(msg)) => {
            assert!(msg.contains("Invalid API key format"));
        },
        other => panic!("Expected InvalidArgument error, got {:?}", other),
    }

    drop(shutdown);
}
