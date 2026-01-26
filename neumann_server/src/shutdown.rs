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
}
