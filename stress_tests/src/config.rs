// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Test configuration presets for stress tests.

use std::env;

/// Scale level for stress tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleLevel {
    /// 100K entities, ~2 min, ~1GB RAM
    Quick,
    /// 1M entities, ~10 min, ~4GB RAM
    Full,
    /// Extended duration (1 hour+)
    Endurance,
}

/// Configuration for stress tests.
#[derive(Debug, Clone)]
pub struct StressConfig {
    pub scale: ScaleLevel,
    pub entity_count: usize,
    pub thread_count: usize,
    pub duration_secs: u64,
    pub embedding_dim: usize,
    pub report_interval_secs: u64,
}

impl StressConfig {
    /// Get thread count, respecting `STRESS_THREADS` env var override.
    #[must_use]
    pub fn effective_thread_count(&self) -> usize {
        env::var("STRESS_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.thread_count)
    }

    /// Get entity count, respecting `STRESS_ENTITIES` env var override.
    #[must_use]
    pub fn effective_entity_count(&self) -> usize {
        env::var("STRESS_ENTITIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.entity_count)
    }

    /// Get duration in seconds, respecting `STRESS_DURATION` env var override.
    #[must_use]
    pub fn effective_duration_secs(&self) -> u64 {
        env::var("STRESS_DURATION")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.duration_secs)
    }
}

/// Quick stress config: 100K entities, 8 threads, ~2 min.
#[must_use]
pub const fn quick_config() -> StressConfig {
    StressConfig {
        scale: ScaleLevel::Quick,
        entity_count: 100_000,
        thread_count: 8,
        duration_secs: 120,
        embedding_dim: 128,
        report_interval_secs: 30,
    }
}

/// Full stress config: 1M entities, 16 threads, ~10 min.
#[must_use]
pub const fn full_config() -> StressConfig {
    StressConfig {
        scale: ScaleLevel::Full,
        entity_count: 1_000_000,
        thread_count: 16,
        duration_secs: 600,
        embedding_dim: 128,
        report_interval_secs: 60,
    }
}

/// Endurance stress config: 500K entities, 8 threads, 1 hour.
#[must_use]
pub const fn endurance_config() -> StressConfig {
    StressConfig {
        scale: ScaleLevel::Endurance,
        entity_count: 500_000,
        thread_count: 8,
        duration_secs: 3600,
        embedding_dim: 128,
        report_interval_secs: 300,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_config() {
        let config = quick_config();
        assert_eq!(config.entity_count, 100_000);
        assert_eq!(config.thread_count, 8);
    }

    #[test]
    fn test_full_config() {
        let config = full_config();
        assert_eq!(config.entity_count, 1_000_000);
        assert_eq!(config.thread_count, 16);
    }

    #[test]
    fn test_effective_duration_secs_default() {
        let config = quick_config();
        // Without STRESS_DURATION set, returns the configured default
        assert_eq!(config.effective_duration_secs(), config.duration_secs);
    }
}
