// SPDX-License-Identifier: MIT OR Apache-2.0
//! Rate limiting for server operations.
//!
//! Prevents abuse by throttling requests per identity using a sliding window algorithm.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::unchecked_time_subtraction)]

use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

use dashmap::DashMap;

/// Configuration for rate limiting.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per identity per window.
    pub max_requests: u32,
    /// Maximum query requests per window.
    pub max_queries: u32,
    /// Maximum blob operations per window.
    pub max_blob_ops: u32,
    /// Time window for rate limiting.
    pub window: Duration,
    /// Enable/disable rate limiting.
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 1000,
            max_queries: 500,
            max_blob_ops: 100,
            window: Duration::from_secs(60),
            enabled: true,
        }
    }
}

impl RateLimitConfig {
    /// Create a new default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum requests per window.
    #[must_use]
    pub const fn with_max_requests(mut self, max: u32) -> Self {
        self.max_requests = max;
        self
    }

    /// Set maximum queries per window.
    #[must_use]
    pub const fn with_max_queries(mut self, max: u32) -> Self {
        self.max_queries = max;
        self
    }

    /// Set maximum blob operations per window.
    #[must_use]
    pub const fn with_max_blob_ops(mut self, max: u32) -> Self {
        self.max_blob_ops = max;
        self
    }

    /// Set the time window.
    #[must_use]
    pub const fn with_window(mut self, window: Duration) -> Self {
        self.window = window;
        self
    }

    /// Disable rate limiting.
    #[must_use]
    pub const fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Strict rate limiting preset for testing.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            max_requests: 10,
            max_queries: 5,
            max_blob_ops: 3,
            window: Duration::from_secs(60),
            enabled: true,
        }
    }

    /// Permissive rate limiting preset.
    #[must_use]
    pub fn permissive() -> Self {
        Self {
            max_requests: 10_000,
            max_queries: 5_000,
            max_blob_ops: 1_000,
            window: Duration::from_secs(60),
            enabled: true,
        }
    }
}

/// Operation types for rate limiting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Any authenticated request.
    Request,
    /// Query execution.
    Query,
    /// Blob upload/download/delete.
    BlobOp,
}

impl Operation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Query => "query",
            Self::BlobOp => "blob_op",
        }
    }

    fn limit(self, config: &RateLimitConfig) -> u32 {
        match self {
            Self::Request => config.max_requests,
            Self::Query => config.max_queries,
            Self::BlobOp => config.max_blob_ops,
        }
    }
}

/// Rate limiter using sliding window algorithm.
pub struct RateLimiter {
    history: DashMap<(String, Operation), VecDeque<Instant>>,
    config: RateLimitConfig,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration.
    #[must_use]
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            history: DashMap::new(),
            config,
        }
    }

    /// Check and record an operation atomically.
    ///
    /// Returns `Ok(())` if allowed, `Err` with message if rate limited.
    #[allow(clippy::cast_possible_truncation)]
    pub fn check_and_record(&self, identity: &str, op: Operation) -> Result<(), String> {
        if !self.config.enabled {
            return Ok(());
        }

        let limit = op.limit(&self.config);
        let key = (identity.to_string(), op);
        let now = Instant::now();
        let window_start = now - self.config.window;

        let mut entry = self.history.entry(key).or_default();
        let timestamps = entry.value_mut();

        // Remove old entries outside the window
        while let Some(front) = timestamps.front() {
            if *front < window_start {
                timestamps.pop_front();
            } else {
                break;
            }
        }

        let count = timestamps.len() as u32;
        if count >= limit {
            Err(format!(
                "rate limit exceeded for {}: {} {} calls in {:?} (max {})",
                identity,
                count,
                op.as_str(),
                self.config.window,
                limit
            ))
        } else {
            timestamps.push_back(now);
            Ok(())
        }
    }

    /// Get current count for an identity/operation.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn count(&self, identity: &str, op: Operation) -> u32 {
        if !self.config.enabled {
            return 0;
        }

        let key = (identity.to_string(), op);
        let now = Instant::now();
        let window_start = now - self.config.window;

        self.history.get(&key).map_or(0, |entry| {
            entry.iter().filter(|&&ts| ts >= window_start).count() as u32
        })
    }

    /// Clear history for an identity.
    pub fn clear(&self, identity: &str) {
        let keys_to_remove: Vec<_> = self
            .history
            .iter()
            .filter(|entry| entry.key().0 == identity)
            .map(|entry| entry.key().clone())
            .collect();

        for key in keys_to_remove {
            self.history.remove(&key);
        }
    }

    /// Check if rate limiting is enabled.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &RateLimitConfig {
        &self.config
    }
}

impl Default for RateLimiter {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_allows_under_limit() {
        let limiter = RateLimiter::new(RateLimitConfig::strict());

        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
    }

    #[test]
    fn test_check_and_record_enforces_limit() {
        let limiter = RateLimiter::new(
            RateLimitConfig::new()
                .with_max_requests(3)
                .with_window(Duration::from_secs(60)),
        );

        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());

        let result = limiter.check_and_record("user:alice", Operation::Request);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("rate limit exceeded"));
    }

    #[test]
    fn test_different_identities_separate_limits() {
        let limiter = RateLimiter::new(RateLimitConfig::new().with_max_requests(2));

        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_err());

        // Bob still has his quota
        assert!(limiter
            .check_and_record("user:bob", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:bob", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:bob", Operation::Request)
            .is_err());
    }

    #[test]
    fn test_different_operations_separate_limits() {
        let limiter = RateLimiter::new(
            RateLimitConfig::new()
                .with_max_requests(2)
                .with_max_queries(2),
        );

        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_err());

        // Query quota still available
        assert!(limiter
            .check_and_record("user:alice", Operation::Query)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Query)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Query)
            .is_err());
    }

    #[test]
    fn test_window_expiration() {
        let limiter = RateLimiter::new(
            RateLimitConfig::new()
                .with_max_requests(2)
                .with_window(Duration::from_millis(50)),
        );

        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_err());

        std::thread::sleep(Duration::from_millis(60));

        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
    }

    #[test]
    fn test_disabled_allows_all() {
        let limiter = RateLimiter::new(RateLimitConfig::new().with_max_requests(1).disabled());

        for _ in 0..100 {
            assert!(limiter
                .check_and_record("user:alice", Operation::Request)
                .is_ok());
        }
    }

    #[test]
    fn test_count_tracking() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        assert_eq!(limiter.count("user:alice", Operation::Request), 0);

        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());

        assert_eq!(limiter.count("user:alice", Operation::Request), 2);
        assert_eq!(limiter.count("user:alice", Operation::Query), 0);
    }

    #[test]
    fn test_clear_identity() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Query)
            .is_ok());
        assert!(limiter
            .check_and_record("user:bob", Operation::Request)
            .is_ok());

        limiter.clear("user:alice");

        assert_eq!(limiter.count("user:alice", Operation::Request), 0);
        assert_eq!(limiter.count("user:alice", Operation::Query), 0);
        assert_eq!(limiter.count("user:bob", Operation::Request), 1);
    }

    #[test]
    fn test_config_presets() {
        let default = RateLimitConfig::default();
        assert_eq!(default.max_requests, 1000);
        assert!(default.enabled);

        let strict = RateLimitConfig::strict();
        assert_eq!(strict.max_requests, 10);
        assert!(strict.enabled);

        let permissive = RateLimitConfig::permissive();
        assert_eq!(permissive.max_requests, 10_000);
        assert!(permissive.enabled);
    }

    #[test]
    fn test_is_enabled() {
        let enabled = RateLimiter::new(RateLimitConfig::default());
        assert!(enabled.is_enabled());

        let disabled = RateLimiter::new(RateLimitConfig::default().disabled());
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn test_operation_as_str() {
        assert_eq!(Operation::Request.as_str(), "request");
        assert_eq!(Operation::Query.as_str(), "query");
        assert_eq!(Operation::BlobOp.as_str(), "blob_op");
    }

    #[test]
    fn test_count_disabled() {
        let limiter = RateLimiter::new(RateLimitConfig::default().disabled());

        assert!(limiter
            .check_and_record("user:alice", Operation::Request)
            .is_ok());

        // When disabled, count always returns 0
        assert_eq!(limiter.count("user:alice", Operation::Request), 0);
    }
}
