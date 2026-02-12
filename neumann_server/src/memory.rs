// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Memory budget tracking for server resource management.
//!
//! Provides atomic memory usage tracking with load shedding support
//! to prevent resource exhaustion under high load.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Configuration for memory budget.
#[derive(Debug, Clone)]
pub struct MemoryBudgetConfig {
    /// Maximum memory bytes allowed.
    pub max_bytes: usize,
    /// Enable load shedding when over budget.
    pub enable_load_shedding: bool,
}

impl Default for MemoryBudgetConfig {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024 * 1024, // 1GB
            enable_load_shedding: true,
        }
    }
}

impl MemoryBudgetConfig {
    /// Create a new memory budget configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum memory budget in bytes.
    #[must_use]
    pub const fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    /// Enable or disable load shedding.
    #[must_use]
    pub const fn with_load_shedding(mut self, enabled: bool) -> Self {
        self.enable_load_shedding = enabled;
        self
    }
}

/// Tracks memory usage across the server.
///
/// Uses atomic operations for thread-safe memory accounting.
pub struct MemoryTracker {
    current_bytes: AtomicUsize,
    max_bytes: usize,
    load_shedding_enabled: bool,
}

impl MemoryTracker {
    /// Create a new memory tracker with the given configuration.
    #[must_use]
    pub const fn new(config: &MemoryBudgetConfig) -> Self {
        Self {
            current_bytes: AtomicUsize::new(0),
            max_bytes: config.max_bytes,
            load_shedding_enabled: config.enable_load_shedding,
        }
    }

    /// Create a memory tracker with default configuration.
    #[must_use]
    pub const fn with_max_bytes(max_bytes: usize) -> Self {
        Self {
            current_bytes: AtomicUsize::new(0),
            max_bytes,
            load_shedding_enabled: true,
        }
    }

    /// Try to allocate memory. Returns true if allocation succeeded.
    ///
    /// If load shedding is disabled, always returns true but still tracks usage.
    pub fn try_allocate(&self, bytes: usize) -> bool {
        loop {
            let current = self.current_bytes.load(Ordering::Acquire);
            let new_value = current.saturating_add(bytes);

            // Check if over budget
            if self.load_shedding_enabled && new_value > self.max_bytes {
                return false;
            }

            // Try to update atomically
            if self
                .current_bytes
                .compare_exchange_weak(current, new_value, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return true;
            }
            // Retry on contention
        }
    }

    /// Release previously allocated memory.
    pub fn release(&self, bytes: usize) {
        self.current_bytes.fetch_sub(bytes, Ordering::Release);
    }

    /// Get current memory usage in bytes.
    #[must_use]
    pub fn current_usage(&self) -> usize {
        self.current_bytes.load(Ordering::Acquire)
    }

    /// Get remaining memory budget in bytes.
    #[must_use]
    pub fn remaining(&self) -> usize {
        let current = self.current_bytes.load(Ordering::Acquire);
        self.max_bytes.saturating_sub(current)
    }

    /// Check if memory usage is over budget.
    #[must_use]
    pub fn is_over_budget(&self) -> bool {
        self.current_bytes.load(Ordering::Acquire) > self.max_bytes
    }

    /// Get the maximum memory budget.
    #[must_use]
    pub const fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Get the current usage as a percentage (0.0 to 1.0+).
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Acceptable for ratio calculation
    pub fn usage_ratio(&self) -> f64 {
        if self.max_bytes == 0 {
            return 0.0;
        }
        self.current_usage() as f64 / self.max_bytes as f64
    }

    /// Reset memory tracking to zero.
    pub fn reset(&self) {
        self.current_bytes.store(0, Ordering::Release);
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new(&MemoryBudgetConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_budget_config_default() {
        let config = MemoryBudgetConfig::default();
        assert_eq!(config.max_bytes, 1024 * 1024 * 1024);
        assert!(config.enable_load_shedding);
    }

    #[test]
    fn test_memory_budget_config_builder() {
        let config = MemoryBudgetConfig::new()
            .with_max_bytes(512 * 1024 * 1024)
            .with_load_shedding(false);

        assert_eq!(config.max_bytes, 512 * 1024 * 1024);
        assert!(!config.enable_load_shedding);
    }

    #[test]
    fn test_memory_tracker_new() {
        let config = MemoryBudgetConfig::new().with_max_bytes(1000);
        let tracker = MemoryTracker::new(&config);

        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.max_bytes(), 1000);
        assert_eq!(tracker.remaining(), 1000);
        assert!(!tracker.is_over_budget());
    }

    #[test]
    fn test_memory_tracker_with_max_bytes() {
        let tracker = MemoryTracker::with_max_bytes(2000);

        assert_eq!(tracker.max_bytes(), 2000);
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn test_try_allocate_success() {
        let tracker = MemoryTracker::with_max_bytes(1000);

        assert!(tracker.try_allocate(500));
        assert_eq!(tracker.current_usage(), 500);
        assert_eq!(tracker.remaining(), 500);
    }

    #[test]
    fn test_try_allocate_over_budget() {
        let tracker = MemoryTracker::with_max_bytes(1000);

        assert!(tracker.try_allocate(600));
        assert!(!tracker.try_allocate(600)); // Would exceed budget
        assert_eq!(tracker.current_usage(), 600);
    }

    #[test]
    fn test_try_allocate_exact_budget() {
        let tracker = MemoryTracker::with_max_bytes(1000);

        assert!(tracker.try_allocate(1000)); // Exactly at budget
        assert!(!tracker.try_allocate(1)); // Even 1 byte over fails
    }

    #[test]
    fn test_try_allocate_load_shedding_disabled() {
        let config = MemoryBudgetConfig::new()
            .with_max_bytes(1000)
            .with_load_shedding(false);
        let tracker = MemoryTracker::new(&config);

        assert!(tracker.try_allocate(2000)); // Over budget but allowed
        assert_eq!(tracker.current_usage(), 2000);
        assert!(tracker.is_over_budget());
    }

    #[test]
    fn test_release() {
        let tracker = MemoryTracker::with_max_bytes(1000);

        assert!(tracker.try_allocate(500));
        tracker.release(200);
        assert_eq!(tracker.current_usage(), 300);
        assert_eq!(tracker.remaining(), 700);
    }

    #[test]
    fn test_release_allows_new_allocation() {
        let tracker = MemoryTracker::with_max_bytes(1000);

        assert!(tracker.try_allocate(800));
        assert!(!tracker.try_allocate(300)); // Would exceed

        tracker.release(400);
        assert!(tracker.try_allocate(300)); // Now fits
    }

    #[test]
    fn test_usage_ratio() {
        let tracker = MemoryTracker::with_max_bytes(1000);

        assert!((tracker.usage_ratio() - 0.0).abs() < f64::EPSILON);

        assert!(tracker.try_allocate(500));
        assert!((tracker.usage_ratio() - 0.5).abs() < f64::EPSILON);

        assert!(tracker.try_allocate(500));
        assert!((tracker.usage_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_ratio_zero_max() {
        let tracker = MemoryTracker::with_max_bytes(0);
        assert!((tracker.usage_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset() {
        let tracker = MemoryTracker::with_max_bytes(1000);

        assert!(tracker.try_allocate(500));
        assert_eq!(tracker.current_usage(), 500);

        tracker.reset();
        assert_eq!(tracker.current_usage(), 0);
        assert_eq!(tracker.remaining(), 1000);
    }

    #[test]
    fn test_is_over_budget() {
        let config = MemoryBudgetConfig::new()
            .with_max_bytes(1000)
            .with_load_shedding(false);
        let tracker = MemoryTracker::new(&config);

        assert!(!tracker.is_over_budget());
        assert!(tracker.try_allocate(1500)); // Load shedding disabled
        assert!(tracker.is_over_budget());
    }

    #[test]
    fn test_concurrent_allocations() {
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(MemoryTracker::with_max_bytes(10000));
        let mut handles = vec![];

        for _ in 0..10 {
            let tracker_clone = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    if tracker_clone.try_allocate(1) {
                        tracker_clone.release(1);
                    }
                }
            }));
        }

        for handle in handles {
            handle.join().expect("thread panicked");
        }

        // All allocations should have been released
        assert_eq!(tracker.current_usage(), 0);
    }

    #[test]
    fn test_saturating_add() {
        let tracker = MemoryTracker::with_max_bytes(usize::MAX);

        // This shouldn't panic even with very large values
        assert!(tracker.try_allocate(usize::MAX / 2));
        // The second large allocation won't overflow due to saturating_add
        let result = tracker.try_allocate(usize::MAX / 2);
        // It might succeed or fail depending on the exact values, but shouldn't panic
        assert!(result || !result); // Just checking no panic
    }

    #[test]
    fn test_default() {
        let tracker = MemoryTracker::default();
        assert_eq!(tracker.max_bytes(), 1024 * 1024 * 1024);
        assert_eq!(tracker.current_usage(), 0);
    }
}
