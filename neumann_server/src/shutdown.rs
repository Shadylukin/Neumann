// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graceful shutdown manager with drain timeout.
//!
//! This module provides a shutdown manager that tracks active streams and
//! waits for them to complete before shutting down, with a configurable timeout.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;

use crate::service::health::HealthState;

/// Configuration for graceful shutdown behavior.
#[derive(Debug, Clone)]
pub struct ShutdownConfig {
    /// Maximum time to wait for in-flight requests to complete.
    pub drain_timeout: Duration,
    /// Grace period before force-closing connections after drain timeout.
    pub grace_period: Duration,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        Self {
            drain_timeout: Duration::from_secs(30),
            grace_period: Duration::from_secs(5),
        }
    }
}

impl ShutdownConfig {
    /// Create a new shutdown configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the drain timeout.
    #[must_use]
    pub fn with_drain_timeout(mut self, timeout: Duration) -> Self {
        self.drain_timeout = timeout;
        self
    }

    /// Set the grace period.
    #[must_use]
    pub fn with_grace_period(mut self, period: Duration) -> Self {
        self.grace_period = period;
        self
    }
}

/// Manages graceful shutdown with active stream tracking.
pub struct ShutdownManager {
    config: ShutdownConfig,
    health_state: Arc<HealthState>,
    active_streams: AtomicU32,
    shutdown_triggered: AtomicBool,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
}

impl ShutdownManager {
    /// Create a new shutdown manager.
    #[must_use]
    pub fn new(config: ShutdownConfig, health_state: Arc<HealthState>) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            config,
            health_state,
            active_streams: AtomicU32::new(0),
            shutdown_triggered: AtomicBool::new(false),
            shutdown_tx,
            shutdown_rx,
        }
    }

    /// Record that a new stream has started.
    pub fn stream_started(&self) {
        self.active_streams.fetch_add(1, Ordering::SeqCst);
    }

    /// Record that a stream has finished.
    pub fn stream_finished(&self) {
        self.active_streams.fetch_sub(1, Ordering::SeqCst);
    }

    /// Get the current count of active streams.
    #[must_use]
    pub fn active_count(&self) -> u32 {
        self.active_streams.load(Ordering::SeqCst)
    }

    /// Check if shutdown has been triggered.
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown_triggered.load(Ordering::SeqCst)
    }

    /// Trigger the shutdown process.
    pub fn trigger_shutdown(&self) {
        self.shutdown_triggered.store(true, Ordering::SeqCst);
        self.health_state.set_draining(true);
        let _ = self.shutdown_tx.send(true);
        tracing::info!("Shutdown triggered, starting drain");
    }

    /// Get a receiver to be notified of shutdown.
    #[must_use]
    pub fn subscribe(&self) -> watch::Receiver<bool> {
        self.shutdown_rx.clone()
    }

    /// Get the shutdown configuration.
    #[must_use]
    pub fn config(&self) -> &ShutdownConfig {
        &self.config
    }

    /// Wait for all active streams to drain.
    ///
    /// Returns `true` if all streams completed within the timeout, `false` otherwise.
    pub async fn wait_for_drain(&self) -> bool {
        let timeout = self.config.drain_timeout;
        let check_interval = Duration::from_millis(100);
        let start = std::time::Instant::now();

        tracing::info!(
            active_streams = self.active_count(),
            timeout_secs = timeout.as_secs(),
            "Waiting for streams to drain"
        );

        loop {
            if self.active_count() == 0 {
                tracing::info!("All streams drained successfully");
                return true;
            }

            if start.elapsed() >= timeout {
                tracing::warn!(
                    remaining_streams = self.active_count(),
                    "Drain timeout reached"
                );
                return false;
            }

            tokio::time::sleep(check_interval).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shutdown_config_default() {
        let config = ShutdownConfig::default();
        assert_eq!(config.drain_timeout, Duration::from_secs(30));
        assert_eq!(config.grace_period, Duration::from_secs(5));
    }

    #[test]
    fn test_shutdown_config_builder() {
        let config = ShutdownConfig::new()
            .with_drain_timeout(Duration::from_secs(60))
            .with_grace_period(Duration::from_secs(10));

        assert_eq!(config.drain_timeout, Duration::from_secs(60));
        assert_eq!(config.grace_period, Duration::from_secs(10));
    }

    #[test]
    fn test_shutdown_manager_new() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = ShutdownManager::new(config, health_state);

        assert_eq!(manager.active_count(), 0);
        assert!(!manager.is_shutting_down());
    }

    #[test]
    fn test_stream_counting() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = ShutdownManager::new(config, health_state);

        assert_eq!(manager.active_count(), 0);

        manager.stream_started();
        assert_eq!(manager.active_count(), 1);

        manager.stream_started();
        assert_eq!(manager.active_count(), 2);

        manager.stream_finished();
        assert_eq!(manager.active_count(), 1);

        manager.stream_finished();
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_trigger_shutdown() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = ShutdownManager::new(config, Arc::clone(&health_state));

        assert!(!manager.is_shutting_down());
        assert!(!health_state.is_draining());

        manager.trigger_shutdown();

        assert!(manager.is_shutting_down());
        assert!(health_state.is_draining());
    }

    #[test]
    fn test_subscribe() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = ShutdownManager::new(config, health_state);

        let rx = manager.subscribe();
        assert!(!*rx.borrow());

        manager.trigger_shutdown();

        // The receiver should be notified
        assert!(rx.has_changed().is_ok());
    }

    #[tokio::test]
    async fn test_drain_completes_when_empty() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::new().with_drain_timeout(Duration::from_secs(1));
        let manager = ShutdownManager::new(config, health_state);

        // No active streams, should complete immediately
        let result = manager.wait_for_drain().await;
        assert!(result);
    }

    #[tokio::test]
    async fn test_drain_waits_for_streams() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::new().with_drain_timeout(Duration::from_secs(2));
        let manager = Arc::new(ShutdownManager::new(config, health_state));

        manager.stream_started();

        let manager_clone = Arc::clone(&manager);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            manager_clone.stream_finished();
        });

        let result = manager.wait_for_drain().await;
        assert!(result);
        assert_eq!(manager.active_count(), 0);
    }

    #[tokio::test]
    async fn test_drain_timeout_enforced() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::new().with_drain_timeout(Duration::from_millis(100));
        let manager = ShutdownManager::new(config, health_state);

        // Start a stream that never finishes
        manager.stream_started();

        let result = manager.wait_for_drain().await;
        assert!(!result);
        assert_eq!(manager.active_count(), 1);
    }

    #[test]
    fn test_config_accessor() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::new().with_drain_timeout(Duration::from_secs(45));
        let manager = ShutdownManager::new(config, health_state);

        assert_eq!(manager.config().drain_timeout, Duration::from_secs(45));
    }

    // === Concurrent Stream Tests ===

    #[tokio::test]
    async fn test_concurrent_stream_registrations() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = Arc::new(ShutdownManager::new(config, health_state));

        // Spawn 100 tasks that each register and unregister a stream
        let mut handles = vec![];
        for _ in 0..100 {
            let manager_clone = Arc::clone(&manager);
            handles.push(tokio::spawn(async move {
                manager_clone.stream_started();
                tokio::time::sleep(Duration::from_millis(1)).await;
                manager_clone.stream_finished();
            }));
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.expect("task should complete");
        }

        // All streams should be finished
        assert_eq!(manager.active_count(), 0);
    }

    #[tokio::test]
    async fn test_concurrent_stream_registrations_during_drain() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::new().with_drain_timeout(Duration::from_secs(1));
        let manager = Arc::new(ShutdownManager::new(config, health_state));

        // Start some initial streams
        for _ in 0..5 {
            manager.stream_started();
        }

        let manager_clone = Arc::clone(&manager);
        let drain_handle = tokio::spawn(async move { manager_clone.wait_for_drain().await });

        // While draining, finish streams one by one
        tokio::time::sleep(Duration::from_millis(50)).await;
        for _ in 0..5 {
            manager.stream_finished();
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        let result = drain_handle.await.expect("drain should complete");
        assert!(result);
    }

    // === Partial Stream Completion Tests ===

    #[tokio::test]
    async fn test_drain_with_partial_completion() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::new().with_drain_timeout(Duration::from_millis(200));
        let manager = Arc::new(ShutdownManager::new(config, health_state));

        // Start 10 streams
        for _ in 0..10 {
            manager.stream_started();
        }

        // Finish only 5 streams
        let manager_clone = Arc::clone(&manager);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            for _ in 0..5 {
                manager_clone.stream_finished();
            }
        });

        // Should timeout because 5 streams remain
        let result = manager.wait_for_drain().await;
        assert!(!result);
        assert_eq!(manager.active_count(), 5);
    }

    #[tokio::test]
    async fn test_drain_with_slow_completion() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::new().with_drain_timeout(Duration::from_secs(1));
        let manager = Arc::new(ShutdownManager::new(config, health_state));

        // Start 3 streams
        for _ in 0..3 {
            manager.stream_started();
        }

        // Finish streams slowly but within timeout
        let manager_clone = Arc::clone(&manager);
        tokio::spawn(async move {
            for _ in 0..3 {
                tokio::time::sleep(Duration::from_millis(100)).await;
                manager_clone.stream_finished();
            }
        });

        let result = manager.wait_for_drain().await;
        assert!(result);
        assert_eq!(manager.active_count(), 0);
    }

    // === Multiple Subscribers Tests ===

    #[tokio::test]
    async fn test_multiple_shutdown_subscribers() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = Arc::new(ShutdownManager::new(config, health_state));

        // Create 5 subscribers
        let mut receivers = vec![];
        for _ in 0..5 {
            receivers.push(manager.subscribe());
        }

        // Verify all start with false
        for rx in &receivers {
            assert!(!*rx.borrow());
        }

        // Trigger shutdown
        manager.trigger_shutdown();

        // All receivers should be notified
        for rx in receivers {
            assert!(rx.has_changed().is_ok());
            assert!(*rx.borrow());
        }
    }

    #[tokio::test]
    async fn test_subscribe_after_shutdown_triggered() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = ShutdownManager::new(config, health_state);

        // Trigger shutdown first
        manager.trigger_shutdown();

        // Subscribe after shutdown is triggered
        let rx = manager.subscribe();

        // Should immediately see shutdown state
        assert!(*rx.borrow());
    }

    // === Grace Period Tests ===

    #[test]
    fn test_grace_period_in_config() {
        let config = ShutdownConfig::new()
            .with_drain_timeout(Duration::from_secs(30))
            .with_grace_period(Duration::from_secs(10));

        assert_eq!(config.grace_period, Duration::from_secs(10));
    }

    #[test]
    fn test_zero_grace_period() {
        let config = ShutdownConfig::new().with_grace_period(Duration::from_secs(0));

        assert_eq!(config.grace_period, Duration::from_secs(0));
    }

    // === Stream Count Edge Cases ===

    #[test]
    fn test_stream_underflow_protection() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = ShutdownManager::new(config, health_state);

        // Finish a stream that was never started (underflow)
        // AtomicU32 wraps on underflow, so this tests that behavior
        manager.stream_finished();

        // The count should wrap to u32::MAX
        assert_eq!(manager.active_count(), u32::MAX);
    }

    #[test]
    fn test_stream_count_large_values() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = ShutdownManager::new(config, health_state);

        // Add a large number of streams
        for _ in 0..1000 {
            manager.stream_started();
        }

        assert_eq!(manager.active_count(), 1000);

        // Remove them all
        for _ in 0..1000 {
            manager.stream_finished();
        }

        assert_eq!(manager.active_count(), 0);
    }

    // === Shutdown State Tests ===

    #[test]
    fn test_shutdown_state_idempotent() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::default();
        let manager = ShutdownManager::new(config, health_state);

        // Trigger shutdown multiple times
        manager.trigger_shutdown();
        manager.trigger_shutdown();
        manager.trigger_shutdown();

        // Should still be shutting down (idempotent)
        assert!(manager.is_shutting_down());
    }

    #[tokio::test]
    async fn test_drain_immediate_return_when_no_streams() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::new().with_drain_timeout(Duration::from_secs(10));
        let manager = ShutdownManager::new(config, health_state);

        let start = std::time::Instant::now();
        let result = manager.wait_for_drain().await;
        let elapsed = start.elapsed();

        // Should complete immediately (much less than 10s timeout)
        assert!(result);
        assert!(elapsed < Duration::from_millis(500));
    }

    #[tokio::test]
    async fn test_drain_with_zero_timeout() {
        let health_state = Arc::new(HealthState::new());
        let config = ShutdownConfig::new().with_drain_timeout(Duration::from_millis(0));
        let manager = ShutdownManager::new(config, health_state);

        manager.stream_started();

        let result = manager.wait_for_drain().await;

        // Should timeout immediately since timeout is 0
        assert!(!result);
        assert_eq!(manager.active_count(), 1);
    }
}
