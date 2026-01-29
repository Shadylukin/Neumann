// SPDX-License-Identifier: MIT OR Apache-2.0
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use tokio::{sync::mpsc, time::interval};

use crate::{
    config::{CacheConfig, EvictionStrategy},
    stats::CacheStats,
};

/// Handle for controlling the background eviction task.
pub struct EvictionHandle {
    shutdown_tx: mpsc::Sender<()>,
    running: Arc<AtomicBool>,
}

impl EvictionHandle {
    pub async fn shutdown(&self) {
        // Receiver may already be dropped during shutdown
        self.shutdown_tx.send(()).await.ok();
    }

    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
pub struct EvictionConfig {
    pub interval: Duration,
    pub batch_size: usize,
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

/// Calculates eviction priority scores. Lower scores are evicted first.
pub struct EvictionScorer {
    strategy: EvictionStrategy,
}

impl EvictionScorer {
    #[must_use]
    pub const fn new(strategy: EvictionStrategy) -> Self {
        Self { strategy }
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn score(
        &self,
        last_access_secs: f64,
        access_count: u64,
        cost_per_hit: f64,
        size_bytes: usize,
    ) -> f64 {
        match self.strategy {
            EvictionStrategy::LRU => -last_access_secs,
            EvictionStrategy::LFU => access_count as f64,
            EvictionStrategy::CostBased => {
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
                let total = f64::from(lru_weight) + f64::from(lfu_weight) + f64::from(cost_weight);
                let recency_w = f64::from(lru_weight) / total;
                let frequency_w = f64::from(lfu_weight) / total;
                let cost_w = f64::from(cost_weight) / total;

                let age_minutes = last_access_secs / 60.0;
                let recency_score = 1.0 / (1.0 + age_minutes);
                let frequency_score = (1.0 + access_count as f64).log2();
                let cost_score = cost_per_hit;

                recency_score
                    .mul_add(recency_w, frequency_score * frequency_w)
                    .mul_add(1.0, cost_score * cost_w)
            },
        }
    }
}

/// Manages background eviction for cache maintenance.
pub struct EvictionManager {
    config: EvictionConfig,
    stats: Arc<CacheStats>,
}

impl EvictionManager {
    #[must_use]
    pub const fn new(config: EvictionConfig, stats: Arc<CacheStats>) -> Self {
        Self { config, stats }
    }

    #[must_use]
    pub fn from_cache_config(config: &CacheConfig, stats: Arc<CacheStats>) -> Self {
        Self::new(EvictionConfig::from(config), stats)
    }

    #[must_use]
    pub const fn config(&self) -> &EvictionConfig {
        &self.config
    }

    #[must_use]
    pub const fn scorer(&self) -> EvictionScorer {
        EvictionScorer::new(self.config.strategy)
    }

    /// Start background eviction. The `evict_fn` is called each cycle and should
    /// return the number of entries evicted.
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

        let old_score = scorer.score(3600.0, 100, 0.5, 1000);
        let new_score = scorer.score(60.0, 100, 0.5, 1000);

        assert!(old_score < new_score);
    }

    #[test]
    fn test_lfu_scoring() {
        let scorer = EvictionScorer::new(EvictionStrategy::LFU);

        let low_freq = scorer.score(60.0, 1, 0.5, 1000);
        let high_freq = scorer.score(60.0, 100, 0.5, 1000);

        assert!(low_freq < high_freq);
    }

    #[test]
    fn test_cost_scoring() {
        let scorer = EvictionScorer::new(EvictionStrategy::CostBased);

        let low_value = scorer.score(60.0, 10, 0.001, 10000);
        let high_value = scorer.score(60.0, 10, 0.1, 100);

        assert!(low_value < high_value);
    }

    #[test]
    fn test_hybrid_scoring() {
        let scorer = EvictionScorer::new(EvictionStrategy::Hybrid {
            lru_weight: 40,
            lfu_weight: 30,
            cost_weight: 30,
        });

        let bad = scorer.score(3600.0, 1, 0.001, 1000);
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

        tokio::time::sleep(Duration::from_millis(50)).await;

        handle.shutdown().await;

        tokio::time::sleep(Duration::from_millis(20)).await;

        assert!(!handle.is_running());
        assert!(evict_count.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_cost_scoring_zero_size() {
        let scorer = EvictionScorer::new(EvictionStrategy::CostBased);

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

        let handle = manager.start(move |batch_size| if batch_size > 0 { 2 } else { 0 });

        tokio::time::sleep(Duration::from_millis(50)).await;

        handle.shutdown().await;
        tokio::time::sleep(Duration::from_millis(20)).await;

        assert!(stats.evictions() > 0);
    }
}
