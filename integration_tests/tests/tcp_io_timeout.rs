//! Integration tests for TCP I/O timeout functionality.
//!
//! Tests that the TCP transport properly times out on:
//! - Slow/unresponsive peers during handshake
//! - Blocked read operations
//! - Connection to non-responsive listeners

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use tensor_chain::{
    tcp::{Handshake, LengthDelimitedCodec, TcpTransportConfig},
    Message, TcpError,
};
use tokio::{
    io::AsyncWriteExt,
    net::TcpListener,
    time::{sleep, timeout},
};

fn test_addr(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

#[tokio::test]
async fn test_handshake_read_timeout() {
    // Start a listener that accepts connections but never sends data
    let listener = TcpListener::bind(test_addr(0)).await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn the slow server
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        // Just hold the connection open without sending anything
        sleep(Duration::from_secs(10)).await;
        drop(socket);
    });

    // Connect and try to read handshake with short timeout
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (mut reader, _writer) = tokio::io::split(stream);

    let result =
        Handshake::read_from_with_timeout(&mut reader, 1024, Duration::from_millis(50)).await;

    assert!(matches!(
        result,
        Err(TcpError::Timeout {
            operation: "handshake read length",
            ..
        })
    ));

    server.abort();
}

#[tokio::test]
async fn test_handshake_write_timeout_success() {
    // Start a listener that accepts and reads handshake
    let listener = TcpListener::bind(test_addr(0)).await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn the server that reads the handshake
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let (mut reader, _writer) = tokio::io::split(socket);
        let handshake = Handshake::read_from(&mut reader, 1024).await.unwrap();
        assert_eq!(handshake.node_id, "client");
    });

    // Connect and write handshake with timeout
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (_reader, mut writer) = tokio::io::split(stream);

    let handshake = Handshake::new("client");
    let result = handshake
        .write_to_with_timeout(&mut writer, Duration::from_secs(5))
        .await;

    assert!(result.is_ok());

    // Wait for server to finish
    timeout(Duration::from_secs(1), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn test_frame_read_timeout() {
    // Start a listener that sends partial frame then stalls
    let listener = TcpListener::bind(test_addr(0)).await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn the slow server - sends length prefix but no payload
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        // Send length prefix (100 bytes)
        let length: u32 = 100;
        socket.write_all(&length.to_be_bytes()).await.unwrap();
        socket.flush().await.unwrap();
        // Don't send payload - just sleep
        sleep(Duration::from_secs(10)).await;
    });

    // Connect and try to read frame with short timeout
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (mut reader, _writer) = tokio::io::split(stream);

    let codec = LengthDelimitedCodec::new(1024);
    let result = codec
        .read_frame_with_timeout(&mut reader, Duration::from_millis(100))
        .await;

    assert!(matches!(
        result,
        Err(TcpError::Timeout {
            operation: "read payload",
            ..
        })
    ));

    server.abort();
}

#[tokio::test]
async fn test_frame_roundtrip_with_timeout() {
    // Test successful roundtrip with timeouts enabled
    let listener = TcpListener::bind(test_addr(0)).await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Spawn server that echoes messages
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let (mut reader, mut writer) = tokio::io::split(socket);

        let codec = LengthDelimitedCodec::new(1024 * 1024);

        // Read message with timeout
        let msg = codec
            .read_frame_with_timeout(&mut reader, Duration::from_secs(5))
            .await
            .unwrap()
            .unwrap();

        // Write response with timeout
        codec
            .write_frame_with_timeout(&mut writer, &msg, Duration::from_secs(5))
            .await
            .unwrap();
    });

    // Connect and send message
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (mut reader, mut writer) = tokio::io::split(stream);

    let codec = LengthDelimitedCodec::new(1024 * 1024);
    let msg = Message::Ping { term: 42 };

    // Write with timeout
    codec
        .write_frame_with_timeout(&mut writer, &msg, Duration::from_secs(5))
        .await
        .unwrap();

    // Read response with timeout
    let response = codec
        .read_frame_with_timeout(&mut reader, Duration::from_secs(5))
        .await
        .unwrap()
        .unwrap();

    if let Message::Ping { term } = response {
        assert_eq!(term, 42);
    } else {
        panic!("unexpected message type");
    }

    timeout(Duration::from_secs(1), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn test_config_io_timeout_propagation() {
    // Verify that config correctly stores and returns io_timeout
    let config = TcpTransportConfig::new("node1", test_addr(19500))
        .with_io_timeout(Duration::from_millis(500));

    assert_eq!(config.io_timeout_ms, 500);
    assert_eq!(config.io_timeout(), Duration::from_millis(500));
}

#[tokio::test]
async fn test_connection_to_slow_handshake_peer() {
    // Start a listener that accepts but sends handshake slowly
    let listener = TcpListener::bind(test_addr(0)).await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Server that delays before sending handshake
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let (_reader, mut writer) = tokio::io::split(socket);

        // Wait longer than client timeout before responding
        sleep(Duration::from_millis(200)).await;

        let handshake = Handshake::new("slow_server");
        let _ = handshake.write_to(&mut writer).await;
    });

    // Try to connect with short timeout
    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (mut reader, mut writer) = tokio::io::split(stream);

    // Send our handshake
    let handshake = Handshake::new("client");
    handshake.write_to(&mut writer).await.unwrap();

    // Try to read server handshake with short timeout - should fail
    let result =
        Handshake::read_from_with_timeout(&mut reader, 1024, Duration::from_millis(50)).await;

    assert!(matches!(result, Err(TcpError::Timeout { .. })));

    server.abort();
}

#[tokio::test]
async fn test_multiple_frames_with_timeout() {
    // Test sending multiple frames with timeouts
    let listener = TcpListener::bind(test_addr(0)).await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        let (mut reader, mut writer) = tokio::io::split(socket);
        let codec = LengthDelimitedCodec::new(1024 * 1024);

        // Read and echo 3 messages
        for _ in 0..3 {
            let msg = codec
                .read_frame_with_timeout(&mut reader, Duration::from_secs(5))
                .await
                .unwrap()
                .unwrap();
            codec
                .write_frame_with_timeout(&mut writer, &msg, Duration::from_secs(5))
                .await
                .unwrap();
        }
    });

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (mut reader, mut writer) = tokio::io::split(stream);
    let codec = LengthDelimitedCodec::new(1024 * 1024);

    // Send 3 messages with different terms
    for term in [1, 2, 3] {
        let msg = Message::Ping { term };
        codec
            .write_frame_with_timeout(&mut writer, &msg, Duration::from_secs(5))
            .await
            .unwrap();

        let response = codec
            .read_frame_with_timeout(&mut reader, Duration::from_secs(5))
            .await
            .unwrap()
            .unwrap();

        if let Message::Ping { term: resp_term } = response {
            assert_eq!(resp_term, term);
        } else {
            panic!("unexpected message type");
        }
    }

    timeout(Duration::from_secs(1), server)
        .await
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn test_timeout_error_contains_operation_info() {
    // Verify timeout errors contain useful debugging info
    let listener = TcpListener::bind(test_addr(0)).await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        // Hold connection but don't send anything
        sleep(Duration::from_secs(10)).await;
        drop(socket);
    });

    let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (mut reader, _writer) = tokio::io::split(stream);

    let codec = LengthDelimitedCodec::new(1024);
    let result = codec
        .read_frame_with_timeout(&mut reader, Duration::from_millis(25))
        .await;

    if let Err(TcpError::Timeout {
        operation,
        timeout_ms,
    }) = result
    {
        assert_eq!(operation, "read length");
        assert_eq!(timeout_ms, 25);
    } else {
        panic!("expected Timeout error");
    }

    server.abort();
}
