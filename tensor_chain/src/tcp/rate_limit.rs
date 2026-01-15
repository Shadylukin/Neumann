//! Per-peer rate limiting using token bucket algorithm.

use std::time::Instant;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::block::NodeId;

/// Token bucket rate limiter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum tokens (burst capacity).
    pub bucket_size: u32,

    /// Tokens added per second.
    pub refill_rate: f64,

    /// Whether rate limiting is enabled.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            bucket_size: 100,
            refill_rate: 50.0,
            enabled: true,
        }
    }
}

impl RateLimitConfig {
    /// Create a disabled rate limit config.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create an aggressive rate limit config (lower limits).
    pub fn aggressive() -> Self {
        Self {
            bucket_size: 50,
            refill_rate: 25.0,
            enabled: true,
        }
    }

    /// Create a permissive rate limit config (higher limits).
    pub fn permissive() -> Self {
        Self {
            bucket_size: 200,
            refill_rate: 100.0,
            enabled: true,
        }
    }

    /// Set bucket size.
    pub fn with_bucket_size(mut self, size: u32) -> Self {
        self.bucket_size = size;
        self
    }

    /// Set refill rate (tokens per second).
    pub fn with_refill_rate(mut self, rate: f64) -> Self {
        self.refill_rate = rate;
        self
    }

    /// Enable or disable rate limiting.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

/// Token bucket state for a single peer.
struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(bucket_size: u32) -> Self {
        Self {
            tokens: bucket_size as f64,
            last_refill: Instant::now(),
        }
    }

    fn refill(&mut self, config: &RateLimitConfig) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * config.refill_rate).min(config.bucket_size as f64);
        self.last_refill = now;
    }

    fn try_consume(&mut self, config: &RateLimitConfig) -> bool {
        self.refill(config);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn available(&mut self, config: &RateLimitConfig) -> u32 {
        self.refill(config);
        self.tokens as u32
    }
}

/// Per-peer rate limiter using token bucket algorithm.
pub struct PeerRateLimiter {
    config: RateLimitConfig,
    buckets: DashMap<NodeId, TokenBucket>,
}

impl PeerRateLimiter {
    /// Create a new rate limiter with the given config.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: DashMap::new(),
        }
    }

    /// Check if a message can be sent to peer and consume a token if allowed.
    /// Returns true if the message is allowed, false if rate limited.
    pub fn check(&self, peer: &NodeId) -> bool {
        if !self.config.enabled {
            return true;
        }
        self.buckets
            .entry(peer.clone())
            .or_insert_with(|| TokenBucket::new(self.config.bucket_size))
            .try_consume(&self.config)
    }

    /// Get available tokens for a peer (for metrics/debugging).
    pub fn available_tokens(&self, peer: &NodeId) -> u32 {
        if !self.config.enabled {
            return u32::MAX;
        }
        self.buckets
            .entry(peer.clone())
            .or_insert_with(|| TokenBucket::new(self.config.bucket_size))
            .available(&self.config)
    }

    /// Remove rate limit state for a peer (on disconnect).
    pub fn remove_peer(&self, peer: &NodeId) {
        self.buckets.remove(peer);
    }

    /// Clear all rate limit state.
    pub fn clear(&self) {
        self.buckets.clear();
    }

    /// Number of tracked peers.
    pub fn peer_count(&self) -> usize {
        self.buckets.len()
    }

    /// Check if rate limiting is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use std::{thread, time::Duration};

    use super::*;

    #[test]
    fn test_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.bucket_size, 100);
        assert_eq!(config.refill_rate, 50.0);
        assert!(config.enabled);
    }

    #[test]
    fn test_config_disabled() {
        let config = RateLimitConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_config_aggressive() {
        let config = RateLimitConfig::aggressive();
        assert_eq!(config.bucket_size, 50);
        assert_eq!(config.refill_rate, 25.0);
        assert!(config.enabled);
    }

    #[test]
    fn test_config_permissive() {
        let config = RateLimitConfig::permissive();
        assert_eq!(config.bucket_size, 200);
        assert_eq!(config.refill_rate, 100.0);
        assert!(config.enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = RateLimitConfig::default()
            .with_bucket_size(75)
            .with_refill_rate(30.0)
            .with_enabled(false);

        assert_eq!(config.bucket_size, 75);
        assert_eq!(config.refill_rate, 30.0);
        assert!(!config.enabled);
    }

    #[test]
    fn test_config_debug() {
        let config = RateLimitConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("RateLimitConfig"));
        assert!(debug.contains("bucket_size"));
    }

    #[test]
    fn test_config_clone() {
        let config = RateLimitConfig::aggressive();
        let cloned = config.clone();
        assert_eq!(cloned.bucket_size, config.bucket_size);
        assert_eq!(cloned.refill_rate, config.refill_rate);
    }

    #[test]
    fn test_rate_limit_allows_burst() {
        let config = RateLimitConfig::default().with_bucket_size(10);
        let limiter = PeerRateLimiter::new(config);
        let peer = "node1".to_string();

        // Should allow 10 messages in burst
        for i in 0..10 {
            assert!(limiter.check(&peer), "Message {} should be allowed", i);
        }
    }

    #[test]
    fn test_rate_limit_blocks_after_burst() {
        let config = RateLimitConfig::default()
            .with_bucket_size(5)
            .with_refill_rate(0.0); // No refill
        let limiter = PeerRateLimiter::new(config);
        let peer = "node1".to_string();

        // Consume all tokens
        for _ in 0..5 {
            assert!(limiter.check(&peer));
        }

        // Next message should be blocked
        assert!(!limiter.check(&peer));
        assert!(!limiter.check(&peer));
    }

    #[test]
    fn test_rate_limit_refills_over_time() {
        let config = RateLimitConfig::default()
            .with_bucket_size(5)
            .with_refill_rate(100.0); // Fast refill: 100 tokens/sec
        let limiter = PeerRateLimiter::new(config);
        let peer = "node1".to_string();

        // Consume all tokens
        for _ in 0..5 {
            limiter.check(&peer);
        }

        // Should be blocked
        assert!(!limiter.check(&peer));

        // Wait for refill (at 100/sec, 50ms should give us ~5 tokens)
        thread::sleep(Duration::from_millis(60));

        // Should be allowed again
        assert!(limiter.check(&peer));
    }

    #[test]
    fn test_rate_limit_disabled() {
        let config = RateLimitConfig::disabled();
        let limiter = PeerRateLimiter::new(config);
        let peer = "node1".to_string();

        // All messages should be allowed when disabled
        for _ in 0..1000 {
            assert!(limiter.check(&peer));
        }

        assert!(!limiter.is_enabled());
    }

    #[test]
    fn test_rate_limit_remove_peer() {
        let config = RateLimitConfig::default().with_bucket_size(5);
        let limiter = PeerRateLimiter::new(config);
        let peer = "node1".to_string();

        // Consume all tokens
        for _ in 0..5 {
            limiter.check(&peer);
        }
        assert!(!limiter.check(&peer));

        // Remove peer
        limiter.remove_peer(&peer);

        // Should have fresh bucket
        assert!(limiter.check(&peer));
        assert_eq!(limiter.peer_count(), 1);
    }

    #[test]
    fn test_rate_limit_multiple_peers() {
        let config = RateLimitConfig::default()
            .with_bucket_size(3)
            .with_refill_rate(0.0);
        let limiter = PeerRateLimiter::new(config);
        let peer1 = "node1".to_string();
        let peer2 = "node2".to_string();

        // Exhaust peer1's tokens
        for _ in 0..3 {
            assert!(limiter.check(&peer1));
        }
        assert!(!limiter.check(&peer1));

        // peer2 should still have tokens
        assert!(limiter.check(&peer2));
        assert!(limiter.check(&peer2));
        assert!(limiter.check(&peer2));
        assert!(!limiter.check(&peer2));

        assert_eq!(limiter.peer_count(), 2);
    }

    #[test]
    fn test_rate_limit_clear() {
        let config = RateLimitConfig::default().with_bucket_size(3);
        let limiter = PeerRateLimiter::new(config);

        limiter.check(&"node1".to_string());
        limiter.check(&"node2".to_string());
        limiter.check(&"node3".to_string());

        assert_eq!(limiter.peer_count(), 3);

        limiter.clear();

        assert_eq!(limiter.peer_count(), 0);
    }

    #[test]
    fn test_available_tokens() {
        let config = RateLimitConfig::default()
            .with_bucket_size(10)
            .with_refill_rate(0.0);
        let limiter = PeerRateLimiter::new(config);
        let peer = "node1".to_string();

        assert_eq!(limiter.available_tokens(&peer), 10);

        limiter.check(&peer);
        assert_eq!(limiter.available_tokens(&peer), 9);

        for _ in 0..5 {
            limiter.check(&peer);
        }
        assert_eq!(limiter.available_tokens(&peer), 4);
    }

    #[test]
    fn test_available_tokens_disabled() {
        let config = RateLimitConfig::disabled();
        let limiter = PeerRateLimiter::new(config);
        let peer = "node1".to_string();

        assert_eq!(limiter.available_tokens(&peer), u32::MAX);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let config = RateLimitConfig::default().with_bucket_size(1000);
        let limiter = Arc::new(PeerRateLimiter::new(config));

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let limiter = limiter.clone();
                let peer = format!("node{}", i);
                thread::spawn(move || {
                    for _ in 0..100 {
                        limiter.check(&peer);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(limiter.peer_count(), 10);
    }

    #[test]
    fn test_is_enabled() {
        let enabled = PeerRateLimiter::new(RateLimitConfig::default());
        assert!(enabled.is_enabled());

        let disabled = PeerRateLimiter::new(RateLimitConfig::disabled());
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn test_bucket_refill_caps_at_max() {
        let config = RateLimitConfig::default()
            .with_bucket_size(5)
            .with_refill_rate(1000.0); // Very fast refill
        let limiter = PeerRateLimiter::new(config);
        let peer = "node1".to_string();

        // Wait a bit
        thread::sleep(Duration::from_millis(20));

        // Available should cap at bucket_size
        assert_eq!(limiter.available_tokens(&peer), 5);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = RateLimitConfig::aggressive();
        let encoded = bincode::serialize(&config).unwrap();
        let decoded: RateLimitConfig = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.bucket_size, config.bucket_size);
        assert_eq!(decoded.refill_rate, config.refill_rate);
        assert_eq!(decoded.enabled, config.enabled);
    }
}
