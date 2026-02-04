// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for TCP per-peer rate limiting.
//!
//! Tests that the TCP transport properly rate limits:
//! - Fast senders don't overwhelm slow peers
//! - Rate limits are per-peer (isolated)
//! - Rate limit state clears on disconnect
//! - Token bucket refills over time

use std::{
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use tensor_chain::{
    tcp::{RateLimitConfig, TcpTransport, TcpTransportConfig},
    Message, SecurityMode, Transport,
};
use tokio::time::{sleep, timeout};

fn test_addr(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

#[tokio::test]
async fn test_rate_limit_blocks_fast_sender() {
    // Start server transport
    let server_config = TcpTransportConfig::new("server", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false)
        .with_rate_limit(RateLimitConfig::default());
    let server = Arc::new(TcpTransport::new(server_config));
    server.start().await.unwrap();

    let server_addr = server.bound_addr().unwrap();
    sleep(Duration::from_millis(10)).await;

    // Create client with aggressive rate limiting (small burst, no refill)
    let client_config = TcpTransportConfig::new("client", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false)
        .with_rate_limit(
            RateLimitConfig::default()
                .with_bucket_size(5)
                .with_refill_rate(0.0),
        );
    let client = Arc::new(TcpTransport::new(client_config));
    client.start().await.unwrap();

    // Connect client to server
    client
        .connect(&tensor_chain::PeerConfig {
            node_id: "server".to_string(),
            address: server_addr.to_string(),
        })
        .await
        .unwrap();

    // Give connection time to establish
    sleep(Duration::from_millis(100)).await;

    // First 5 messages should succeed
    for i in 0..5 {
        let result = client
            .send(&"server".to_string(), Message::Ping { term: i })
            .await;
        assert!(result.is_ok(), "Message {} should succeed", i);
    }

    // 6th message should be rate limited
    let result = client
        .send(&"server".to_string(), Message::Ping { term: 100 })
        .await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("rate limited"));

    // Cleanup
    client.shutdown().await;
    server.shutdown().await;
}

#[tokio::test]
async fn test_rate_limit_allows_sustained_traffic() {
    // Server
    let server_config = TcpTransportConfig::new("server", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let server = Arc::new(TcpTransport::new(server_config));
    server.start().await.unwrap();

    let server_addr = server.bound_addr().unwrap();
    sleep(Duration::from_millis(10)).await;

    // Client with moderate rate limiting that allows sustained traffic
    let client_config = TcpTransportConfig::new("client", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false)
        .with_rate_limit(
            RateLimitConfig::default()
                .with_bucket_size(10)
                .with_refill_rate(100.0), // 100 tokens/sec = 1 token per 10ms
        );
    let client = Arc::new(TcpTransport::new(client_config));
    client.start().await.unwrap();

    client
        .connect(&tensor_chain::PeerConfig {
            node_id: "server".to_string(),
            address: server_addr.to_string(),
        })
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    // Send messages at sustained rate (slower than refill)
    let mut success_count = 0;
    for i in 0..20 {
        // Sleep 15ms between messages (slower than 100/sec refill rate)
        if i > 0 {
            sleep(Duration::from_millis(15)).await;
        }
        let result = client
            .send(&"server".to_string(), Message::Ping { term: i })
            .await;
        if result.is_ok() {
            success_count += 1;
        }
    }

    // Most messages should succeed (some initial burst may vary)
    assert!(
        success_count >= 15,
        "Expected at least 15 successful sends, got {}",
        success_count
    );

    client.shutdown().await;
    server.shutdown().await;
}

#[tokio::test]
async fn test_rate_limit_recovery_after_wait() {
    // Server
    let server_config = TcpTransportConfig::new("server", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let server = Arc::new(TcpTransport::new(server_config));
    server.start().await.unwrap();

    let server_addr = server.bound_addr().unwrap();
    sleep(Duration::from_millis(10)).await;

    // Client with small burst but fast refill
    let client_config = TcpTransportConfig::new("client", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false)
        .with_rate_limit(
            RateLimitConfig::default()
                .with_bucket_size(3)
                .with_refill_rate(100.0), // 100/sec = 1 token per 10ms
        );
    let client = Arc::new(TcpTransport::new(client_config));
    client.start().await.unwrap();

    client
        .connect(&tensor_chain::PeerConfig {
            node_id: "server".to_string(),
            address: server_addr.to_string(),
        })
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    // Exhaust burst
    for i in 0..3 {
        client
            .send(&"server".to_string(), Message::Ping { term: i })
            .await
            .unwrap();
    }

    // Should be rate limited
    let result = client
        .send(&"server".to_string(), Message::Ping { term: 100 })
        .await;
    assert!(result.unwrap_err().to_string().contains("rate limited"));

    // Wait for refill (~50ms should give us ~5 tokens)
    sleep(Duration::from_millis(60)).await;

    // Should be allowed again
    let result = client
        .send(&"server".to_string(), Message::Ping { term: 200 })
        .await;
    assert!(result.is_ok());

    client.shutdown().await;
    server.shutdown().await;
}

#[tokio::test]
async fn test_rate_limit_per_peer_isolation() {
    // Start two servers
    let server1_config = TcpTransportConfig::new("server1", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let server1 = Arc::new(TcpTransport::new(server1_config));
    server1.start().await.unwrap();

    let server2_config = TcpTransportConfig::new("server2", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let server2 = Arc::new(TcpTransport::new(server2_config));
    server2.start().await.unwrap();

    let server1_addr = server1.bound_addr().unwrap();
    let server2_addr = server2.bound_addr().unwrap();
    sleep(Duration::from_millis(10)).await;

    // Client with small per-peer bucket
    let client_config = TcpTransportConfig::new("client", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false)
        .with_rate_limit(
            RateLimitConfig::default()
                .with_bucket_size(3)
                .with_refill_rate(0.0),
        );
    let client = Arc::new(TcpTransport::new(client_config));
    client.start().await.unwrap();

    // Connect to both servers
    client
        .connect(&tensor_chain::PeerConfig {
            node_id: "server1".to_string(),
            address: server1_addr.to_string(),
        })
        .await
        .unwrap();

    client
        .connect(&tensor_chain::PeerConfig {
            node_id: "server2".to_string(),
            address: server2_addr.to_string(),
        })
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    // Exhaust rate limit for server1
    for i in 0..3 {
        client
            .send(&"server1".to_string(), Message::Ping { term: i })
            .await
            .unwrap();
    }

    // server1 should be rate limited
    let result = client
        .send(&"server1".to_string(), Message::Ping { term: 100 })
        .await;
    assert!(result.unwrap_err().to_string().contains("rate limited"));

    // server2 should still have fresh bucket
    for i in 0..3 {
        let result = client
            .send(&"server2".to_string(), Message::Ping { term: i })
            .await;
        assert!(result.is_ok(), "server2 message {} should succeed", i);
    }

    client.shutdown().await;
    server1.shutdown().await;
    server2.shutdown().await;
}

#[tokio::test]
async fn test_rate_limit_cleanup_on_disconnect() {
    // Server
    let server_config = TcpTransportConfig::new("server", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let server = Arc::new(TcpTransport::new(server_config));
    server.start().await.unwrap();

    let server_addr = server.bound_addr().unwrap();
    sleep(Duration::from_millis(10)).await;

    // Client with small burst
    let client_config = TcpTransportConfig::new("client", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false)
        .with_rate_limit(
            RateLimitConfig::default()
                .with_bucket_size(2)
                .with_refill_rate(0.0),
        );
    let client = Arc::new(TcpTransport::new(client_config));
    client.start().await.unwrap();

    // Connect
    client
        .connect(&tensor_chain::PeerConfig {
            node_id: "server".to_string(),
            address: server_addr.to_string(),
        })
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    // Exhaust rate limit
    client
        .send(&"server".to_string(), Message::Ping { term: 1 })
        .await
        .unwrap();
    client
        .send(&"server".to_string(), Message::Ping { term: 2 })
        .await
        .unwrap();

    // Should be rate limited
    let result = client
        .send(&"server".to_string(), Message::Ping { term: 100 })
        .await;
    assert!(result.unwrap_err().to_string().contains("rate limited"));

    // Disconnect
    client.disconnect(&"server".to_string()).await.unwrap();

    // Reconnect
    client
        .connect(&tensor_chain::PeerConfig {
            node_id: "server".to_string(),
            address: server_addr.to_string(),
        })
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    // Should have fresh rate limit bucket
    let result = client
        .send(&"server".to_string(), Message::Ping { term: 300 })
        .await;
    assert!(result.is_ok(), "Should have fresh bucket after reconnect");

    client.shutdown().await;
    server.shutdown().await;
}

#[tokio::test]
async fn test_rate_limit_disabled() {
    // Server
    let server_config = TcpTransportConfig::new("server", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let server = Arc::new(TcpTransport::new(server_config));
    server.start().await.unwrap();

    let server_addr = server.bound_addr().unwrap();
    sleep(Duration::from_millis(10)).await;

    // Client with rate limiting disabled
    let client_config = TcpTransportConfig::new("client", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false)
        .with_rate_limit(RateLimitConfig::disabled());
    let client = Arc::new(TcpTransport::new(client_config));
    client.start().await.unwrap();

    client
        .connect(&tensor_chain::PeerConfig {
            node_id: "server".to_string(),
            address: server_addr.to_string(),
        })
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    // Should be able to send many messages without rate limiting
    for i in 0..100 {
        let result = client
            .send(&"server".to_string(), Message::Ping { term: i })
            .await;
        assert!(result.is_ok(), "Message {} should succeed", i);
    }

    client.shutdown().await;
    server.shutdown().await;
}

#[tokio::test]
async fn test_rate_limit_config_presets() {
    // Test that config presets work correctly
    let aggressive = RateLimitConfig::aggressive();
    assert_eq!(aggressive.bucket_size, 50);
    assert_eq!(aggressive.refill_rate, 25.0);
    assert!(aggressive.enabled);

    let permissive = RateLimitConfig::permissive();
    assert_eq!(permissive.bucket_size, 200);
    assert_eq!(permissive.refill_rate, 100.0);
    assert!(permissive.enabled);

    let disabled = RateLimitConfig::disabled();
    assert!(!disabled.enabled);
}

#[tokio::test]
async fn test_rate_limit_message_delivery() {
    // Verify that rate-limited messages aren't lost - they're just rejected
    let server_config = TcpTransportConfig::new("server", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false);
    let server = Arc::new(TcpTransport::new(server_config));
    server.start().await.unwrap();

    let server_addr = server.bound_addr().unwrap();
    sleep(Duration::from_millis(10)).await;

    // Client with very small bucket
    let client_config = TcpTransportConfig::new("client", test_addr(0))
        .with_security_mode(SecurityMode::Development)
        .with_require_tls(false)
        .with_rate_limit(
            RateLimitConfig::default()
                .with_bucket_size(3)
                .with_refill_rate(0.0),
        );
    let client = Arc::new(TcpTransport::new(client_config));
    client.start().await.unwrap();

    client
        .connect(&tensor_chain::PeerConfig {
            node_id: "server".to_string(),
            address: server_addr.to_string(),
        })
        .await
        .unwrap();

    sleep(Duration::from_millis(100)).await;

    // Spawn server receiver
    let server_clone = server.clone();
    let receiver = tokio::spawn(async move {
        let mut received = Vec::new();
        for _ in 0..3 {
            match timeout(Duration::from_millis(500), server_clone.recv()).await {
                Ok(Ok((_, msg))) => {
                    if let Message::Ping { term } = msg {
                        received.push(term);
                    }
                },
                _ => break,
            }
        }
        received
    });

    // Send messages - some will be rate limited
    let mut sent_terms = Vec::new();
    for i in 0..10 {
        let result = client
            .send(&"server".to_string(), Message::Ping { term: i })
            .await;
        if result.is_ok() {
            sent_terms.push(i);
        }
    }

    // Should have sent exactly 3
    assert_eq!(sent_terms.len(), 3);
    assert_eq!(sent_terms, vec![0, 1, 2]);

    // Server should receive exactly what was sent
    let received = timeout(Duration::from_secs(1), receiver)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(received.len(), 3);
    assert_eq!(received, vec![0, 1, 2]);

    client.shutdown().await;
    server.shutdown().await;
}
