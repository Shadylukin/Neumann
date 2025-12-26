//! Background eviction manager for cache maintenance.

use crate::config::{CacheConfig, EvictionStrategy};
use crate::stats::CacheStats;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::interval;

/// Handle for controlling the background eviction task.
pub struct EvictionHandle {
    shutdown_tx: mpsc::Sender<()>,
    running: Arc<AtomicBool>,
}

impl EvictionHandle {
    /// Signal the eviction task to stop.
    pub async fn shutdown(&self) {
        let _ = self.shutdown_tx.send(()).await;
    }

    /// Check if the eviction task is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

/// Configuration for the eviction manager.
#[derive(Debug, Clone)]
pub struct EvictionConfig {
    /// How often to run eviction.
    pub interval: Duration,
    /// How many entries to consider per eviction cycle.
    pub batch_size: usize,
    /// Eviction strategy to use.
    pub strategy: EvictionStrategy,
}

impl From<&CacheConfig> for EvictionConfig {
    fn from(config: &CacheConfig) -> Self {
        Self {
            interval: config.eviction_interval,
            batch_size: config.eviction_batch_size,
            strategy: config.eviction_strategy,
        }
    }
}

/// Calculate eviction scores based on the configured strategy.
pub struct EvictionScorer {
    strategy: EvictionStrategy,
}

impl EvictionScorer {
    /// Create a new eviction scorer.
    pub fn new(strategy: EvictionStrategy) -> Self {
        Self { strategy }
    }

    /// Calculate the eviction score for an entry.
    ///
    /// Lower scores are evicted first.
    ///
    /// # Arguments
    /// * `last_access_secs` - Seconds since last access
    /// * `access_count` - Number of times accessed
    /// * `cost_per_hit` - Estimated cost saved per hit (in dollars)
    /// * `size_bytes` - Size of the entry in bytes
    pub fn score(
        &self,
        last_access_secs: f64,
        access_count: u64,
        cost_per_hit: f64,
        size_bytes: usize,
    ) -> f64 {
        match self.strategy {
            EvictionStrategy::LRU => {
                // Lower recency = lower score = evict first
                // Negate so older entries have lower scores
                -last_access_secs
            },
            EvictionStrategy::LFU => {
                // Lower frequency = lower score = evict first
                access_count as f64
            },
            EvictionStrategy::CostBased => {
                // Lower cost efficiency = lower score = evict first
                // cost_per_hit / size_bytes
                if size_bytes == 0 {
                    0.0
                } else {
                    cost_per_hit / size_bytes as f64
                }
            },
            EvictionStrategy::Hybrid {
                lru_weight,
                lfu_weight,
                cost_weight,
            } => {
                // Normalize weights
                let total = (lru_weight + lfu_weight + cost_weight) as f64;
                let recency_weight = lru_weight as f64 / total;
                let frequency_weight = lfu_weight as f64 / total;
                let cost_weight_norm = cost_weight as f64 / total;

                // Normalize each component to [0, 1] range approximately
                // For LRU: 1 / (1 + age_in_minutes) - higher is better (more recent)
                let age_minutes = last_access_secs / 60.0;
                let recency_score = 1.0 / (1.0 + age_minutes);

                // For LFU: log2(1 + access_count) - higher is better
                let frequency_score = (1.0 + access_count as f64).log2();

                // For cost: cost_per_hit - higher is better
                let cost_score = cost_per_hit;

                // Combine with weights
                recency_score * recency_weight
                    + frequency_score * frequency_weight
                    + cost_score * cost_weight_norm
            },
        }
    }
}

/// Manages background eviction for the cache.
pub struct EvictionManager {
    config: EvictionConfig,
    stats: Arc<CacheStats>,
}

impl EvictionManager {
    /// Create a new eviction manager.
    pub fn new(config: EvictionConfig, stats: Arc<CacheStats>) -> Self {
        Self { config, stats }
    }

    /// Create from cache config.
    pub fn from_cache_config(config: &CacheConfig, stats: Arc<CacheStats>) -> Self {
        Self::new(EvictionConfig::from(config), stats)
    }

    /// Get the eviction configuration.
    pub fn config(&self) -> &EvictionConfig {
        &self.config
    }

    /// Get the eviction scorer for the configured strategy.
    pub fn scorer(&self) -> EvictionScorer {
        EvictionScorer::new(self.config.strategy)
    }

    /// Start the background eviction task.
    ///
    /// Returns a handle that can be used to stop the task.
    ///
    /// The `evict_fn` is called on each cycle and should return the number
    /// of entries evicted.
    pub fn start<F>(&self, evict_fn: F) -> EvictionHandle
    where
        F: Fn(usize) -> usize + Send + 'static,
    {
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        let interval_duration = self.config.interval;
        let batch_size = self.config.batch_size;
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut ticker = interval(interval_duration);

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        let evicted = evict_fn(batch_size);
                        if evicted > 0 {
                            stats.record_eviction(evicted);
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        running_clone.store(false, Ordering::Relaxed);
                        break;
                    }
                }
            }
        });

        EvictionHandle {
            shutdown_tx,
            running,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_scoring() {
        let scorer = EvictionScorer::new(EvictionStrategy::LRU);

        // Older entries should have lower (more negative) scores
        let old_score = scorer.score(3600.0, 100, 0.5, 1000);
        let new_score = scorer.score(60.0, 100, 0.5, 1000);

        assert!(old_score < new_score);
    }

    #[test]
    fn test_lfu_scoring() {
        let scorer = EvictionScorer::new(EvictionStrategy::LFU);

        // Less accessed entries should have lower scores
        let low_freq = scorer.score(60.0, 1, 0.5, 1000);
        let high_freq = scorer.score(60.0, 100, 0.5, 1000);

        assert!(low_freq < high_freq);
    }

    #[test]
    fn test_cost_scoring() {
        let scorer = EvictionScorer::new(EvictionStrategy::CostBased);

        // Lower cost efficiency should have lower scores
        let low_value = scorer.score(60.0, 10, 0.001, 10000); // Low cost, large size
        let high_value = scorer.score(60.0, 10, 0.1, 100); // High cost, small size

        assert!(low_value < high_value);
    }

    #[test]
    fn test_hybrid_scoring() {
        let scorer = EvictionScorer::new(EvictionStrategy::Hybrid {
            lru_weight: 40,
            lfu_weight: 30,
            cost_weight: 30,
        });

        // Entry that's old, rarely accessed, and low value
        let bad = scorer.score(3600.0, 1, 0.001, 1000);

        // Entry that's recent, frequently accessed, and high value
        let good = scorer.score(60.0, 100, 0.1, 100);

        assert!(bad < good);
    }

    #[test]
    fn test_eviction_config_from_cache_config() {
        let cache_config = CacheConfig::default();
        let eviction_config = EvictionConfig::from(&cache_config);

        assert_eq!(eviction_config.interval, cache_config.eviction_interval);
        assert_eq!(eviction_config.batch_size, cache_config.eviction_batch_size);
    }

    #[tokio::test]
    async fn test_eviction_handle() {
        let stats = Arc::new(CacheStats::new());
        let manager = EvictionManager::new(
            EvictionConfig {
                interval: Duration::from_millis(10),
                batch_size: 10,
                strategy: EvictionStrategy::LRU,
            },
            stats,
        );

        let evict_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let evict_count_clone = Arc::clone(&evict_count);

        let handle = manager.start(move |_batch_size| {
            evict_count_clone.fetch_add(1, Ordering::Relaxed);
            0
        });

        assert!(handle.is_running());

        // Let it run a few cycles
        tokio::time::sleep(Duration::from_millis(50)).await;

        handle.shutdown().await;

        // Give it time to stop
        tokio::time::sleep(Duration::from_millis(20)).await;

        assert!(!handle.is_running());
        assert!(evict_count.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_cost_scoring_zero_size() {
        let scorer = EvictionScorer::new(EvictionStrategy::CostBased);

        // Zero size should return 0
        let score = scorer.score(60.0, 10, 0.5, 0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_from_cache_config() {
        let cache_config = CacheConfig::default();
        let stats = Arc::new(CacheStats::new());
        let manager = EvictionManager::from_cache_config(&cache_config, stats);

        assert_eq!(manager.config().interval, cache_config.eviction_interval);
        assert_eq!(
            manager.config().batch_size,
            cache_config.eviction_batch_size
        );
    }

    #[test]
    fn test_manager_scorer() {
        let stats = Arc::new(CacheStats::new());
        let manager = EvictionManager::new(
            EvictionConfig {
                interval: Duration::from_secs(60),
                batch_size: 100,
                strategy: EvictionStrategy::LFU,
            },
            stats,
        );

        let scorer = manager.scorer();
        // Verify the scorer uses LFU strategy
        let low_freq = scorer.score(60.0, 1, 0.5, 1000);
        let high_freq = scorer.score(60.0, 100, 0.5, 1000);
        assert!(low_freq < high_freq);
    }

    #[test]
    fn test_eviction_config_clone() {
        let config = EvictionConfig {
            interval: Duration::from_secs(30),
            batch_size: 50,
            strategy: EvictionStrategy::LRU,
        };
        let cloned = config.clone();
        assert_eq!(cloned.interval, config.interval);
        assert_eq!(cloned.batch_size, config.batch_size);
    }

    #[tokio::test]
    async fn test_eviction_with_actual_evictions() {
        let stats = Arc::new(CacheStats::new());
        let manager = EvictionManager::new(
            EvictionConfig {
                interval: Duration::from_millis(10),
                batch_size: 5,
                strategy: EvictionStrategy::LRU,
            },
            Arc::clone(&stats),
        );

        let handle = manager.start(move |batch_size| {
            // Simulate evicting 2 entries
            if batch_size > 0 {
                2
            } else {
                0
            }
        });

        // Let it run a few cycles
        tokio::time::sleep(Duration::from_millis(50)).await;

        handle.shutdown().await;
        tokio::time::sleep(Duration::from_millis(20)).await;

        // Eviction count should be recorded in stats
        assert!(stats.evictions() > 0);
    }
}
