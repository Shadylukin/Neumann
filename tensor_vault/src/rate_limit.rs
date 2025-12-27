//! Rate limiting for vault operations.
//!
//! Prevents brute-force enumeration and throttles aggressive agents.

#![allow(clippy::missing_panics_doc)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::unchecked_time_subtraction)]

use dashmap::DashMap;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for rate limiting.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum get() calls per window.
    pub max_gets: u32,
    /// Maximum list() calls per window.
    pub max_lists: u32,
    /// Maximum set() calls per window.
    pub max_sets: u32,
    /// Maximum grant() calls per window.
    pub max_grants: u32,
    /// Time window for rate limiting.
    pub window: Duration,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_gets: 60,
            max_lists: 10,
            max_sets: 30,
            max_grants: 20,
            window: Duration::from_secs(60),
        }
    }
}

impl RateLimitConfig {
    /// No rate limiting.
    pub fn unlimited() -> Self {
        Self {
            max_gets: u32::MAX,
            max_lists: u32::MAX,
            max_sets: u32::MAX,
            max_grants: u32::MAX,
            window: Duration::from_secs(60),
        }
    }

    /// Strict rate limiting for testing.
    pub fn strict() -> Self {
        Self {
            max_gets: 5,
            max_lists: 2,
            max_sets: 3,
            max_grants: 2,
            window: Duration::from_secs(60),
        }
    }
}

/// Operation types that can be rate limited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    Get,
    List,
    Set,
    Grant,
}

impl Operation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Get => "get",
            Self::List => "list",
            Self::Set => "set",
            Self::Grant => "grant",
        }
    }

    fn limit(self, config: &RateLimitConfig) -> u32 {
        match self {
            Self::Get => config.max_gets,
            Self::List => config.max_lists,
            Self::Set => config.max_sets,
            Self::Grant => config.max_grants,
        }
    }
}

/// Rate limiter using sliding window algorithm.
pub struct RateLimiter {
    /// (entity, operation) -> timestamps of recent requests
    history: DashMap<(String, String), VecDeque<Instant>>,
    config: RateLimitConfig,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            history: DashMap::new(),
            config,
        }
    }

    /// Check if the operation is allowed for the entity.
    ///
    /// Returns Ok(()) if allowed, Err with message if rate limited.
    pub fn check(&self, entity: &str, op: Operation) -> Result<(), String> {
        let limit = op.limit(&self.config);
        if limit == u32::MAX {
            return Ok(());
        }

        let key = (entity.to_string(), op.as_str().to_string());
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
                "Rate limit exceeded for {}: {} {} calls in {:?} (max {})",
                entity,
                count,
                op.as_str(),
                self.config.window,
                limit
            ))
        } else {
            Ok(())
        }
    }

    /// Record an operation for the entity.
    pub fn record(&self, entity: &str, op: Operation) {
        let key = (entity.to_string(), op.as_str().to_string());
        let now = Instant::now();

        self.history.entry(key).or_default().push_back(now);
    }

    /// Check and record an operation atomically.
    ///
    /// If the check passes, the operation is recorded.
    pub fn check_and_record(&self, entity: &str, op: Operation) -> Result<(), String> {
        let limit = op.limit(&self.config);
        if limit == u32::MAX {
            return Ok(());
        }

        let key = (entity.to_string(), op.as_str().to_string());
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
                "Rate limit exceeded for {}: {} {} calls in {:?} (max {})",
                entity,
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

    /// Get the current count for an operation.
    pub fn count(&self, entity: &str, op: Operation) -> u32 {
        let key = (entity.to_string(), op.as_str().to_string());
        let now = Instant::now();
        let window_start = now - self.config.window;

        if let Some(entry) = self.history.get(&key) {
            entry.iter().filter(|&&ts| ts >= window_start).count() as u32
        } else {
            0
        }
    }

    /// Clear rate limit history for an entity.
    pub fn clear(&self, entity: &str) {
        let prefixes: Vec<_> = self
            .history
            .iter()
            .filter(|entry| entry.key().0 == entity)
            .map(|entry| entry.key().clone())
            .collect();

        for key in prefixes {
            self.history.remove(&key);
        }
    }

    /// Clear all rate limit history.
    pub fn clear_all(&self) {
        self.history.clear();
    }

    /// Get the configuration.
    pub fn config(&self) -> &RateLimitConfig {
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

        // First call should be allowed
        assert!(limiter.check("user:alice", Operation::Get).is_ok());
    }

    #[test]
    fn test_check_and_record_enforces_limit() {
        let limiter = RateLimiter::new(RateLimitConfig {
            max_gets: 3,
            ..RateLimitConfig::default()
        });

        // First 3 should succeed
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());

        // 4th should fail
        let result = limiter.check_and_record("user:alice", Operation::Get);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Rate limit exceeded"));
    }

    #[test]
    fn test_different_entities_separate_limits() {
        let limiter = RateLimiter::new(RateLimitConfig {
            max_gets: 2,
            ..RateLimitConfig::default()
        });

        // Alice uses her quota
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_err());

        // Bob still has his quota
        assert!(limiter.check_and_record("user:bob", Operation::Get).is_ok());
        assert!(limiter.check_and_record("user:bob", Operation::Get).is_ok());
        assert!(limiter
            .check_and_record("user:bob", Operation::Get)
            .is_err());
    }

    #[test]
    fn test_different_operations_separate_limits() {
        let limiter = RateLimiter::new(RateLimitConfig {
            max_gets: 2,
            max_sets: 2,
            ..RateLimitConfig::default()
        });

        // Use up get quota
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_err());

        // Set quota still available
        assert!(limiter
            .check_and_record("user:alice", Operation::Set)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Set)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Set)
            .is_err());
    }

    #[test]
    fn test_count() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        assert_eq!(limiter.count("user:alice", Operation::Get), 0);

        limiter.record("user:alice", Operation::Get);
        limiter.record("user:alice", Operation::Get);

        assert_eq!(limiter.count("user:alice", Operation::Get), 2);
        assert_eq!(limiter.count("user:alice", Operation::Set), 0);
    }

    #[test]
    fn test_clear() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        limiter.record("user:alice", Operation::Get);
        limiter.record("user:alice", Operation::Set);
        limiter.record("user:bob", Operation::Get);

        limiter.clear("user:alice");

        assert_eq!(limiter.count("user:alice", Operation::Get), 0);
        assert_eq!(limiter.count("user:alice", Operation::Set), 0);
        assert_eq!(limiter.count("user:bob", Operation::Get), 1);
    }

    #[test]
    fn test_clear_all() {
        let limiter = RateLimiter::new(RateLimitConfig::default());

        limiter.record("user:alice", Operation::Get);
        limiter.record("user:bob", Operation::Get);

        limiter.clear_all();

        assert_eq!(limiter.count("user:alice", Operation::Get), 0);
        assert_eq!(limiter.count("user:bob", Operation::Get), 0);
    }

    #[test]
    fn test_unlimited_config() {
        let limiter = RateLimiter::new(RateLimitConfig::unlimited());

        // Should never fail
        for _ in 0..1000 {
            assert!(limiter
                .check_and_record("user:alice", Operation::Get)
                .is_ok());
        }
    }

    #[test]
    fn test_window_expiration() {
        let limiter = RateLimiter::new(RateLimitConfig {
            max_gets: 2,
            window: Duration::from_millis(50),
            ..RateLimitConfig::default()
        });

        // Use up quota
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_err());

        // Wait for window to expire
        std::thread::sleep(Duration::from_millis(60));

        // Should be allowed again
        assert!(limiter
            .check_and_record("user:alice", Operation::Get)
            .is_ok());
    }

    #[test]
    fn test_operation_as_str() {
        assert_eq!(Operation::Get.as_str(), "get");
        assert_eq!(Operation::List.as_str(), "list");
        assert_eq!(Operation::Set.as_str(), "set");
        assert_eq!(Operation::Grant.as_str(), "grant");
    }
}
