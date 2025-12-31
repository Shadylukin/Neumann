use serde::{Deserialize, Serialize};
use std::time::Duration;
use tensor_store::DistanceMetric;

/// Eviction strategy for cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Evicts entries that haven't been accessed recently.
    LRU,
    /// Evicts entries with lowest access count.
    LFU,
    /// Evicts entries with lowest cost savings per byte.
    CostBased,
    /// Combines LRU, LFU, and cost factors with configurable weights.
    Hybrid {
        lru_weight: u8,
        lfu_weight: u8,
        cost_weight: u8,
    },
}

impl Default for EvictionStrategy {
    fn default() -> Self {
        Self::Hybrid {
            lru_weight: 40,
            lfu_weight: 30,
            cost_weight: 30,
        }
    }
}

/// Cache configuration with capacity limits, TTL, and eviction settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub exact_capacity: usize,
    pub semantic_capacity: usize,
    pub embedding_capacity: usize,
    pub default_ttl: Duration,
    pub max_ttl: Duration,
    /// Minimum similarity threshold for semantic cache hits (0.0 to 1.0).
    pub semantic_threshold: f32,
    /// Expected embedding dimension (e.g., 1536 for text-embedding-ada-002).
    pub embedding_dim: usize,
    pub eviction_strategy: EvictionStrategy,
    pub eviction_interval: Duration,
    pub eviction_batch_size: usize,
    /// Cost per 1000 input tokens (in dollars).
    pub input_cost_per_1k: f64,
    /// Cost per 1000 output tokens (in dollars).
    pub output_cost_per_1k: f64,
    pub inline_threshold: usize,
    pub distance_metric: DistanceMetric,
    /// When true, uses Jaccard for sparse embeddings above `sparsity_metric_threshold`.
    pub auto_select_metric: bool,
    pub sparsity_metric_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            exact_capacity: 10_000,
            semantic_capacity: 5_000,
            embedding_capacity: 50_000,
            default_ttl: Duration::from_secs(3600),
            max_ttl: Duration::from_secs(86400),
            semantic_threshold: 0.92,
            embedding_dim: 1536,
            eviction_strategy: EvictionStrategy::default(),
            eviction_interval: Duration::from_secs(60),
            eviction_batch_size: 100,
            input_cost_per_1k: 0.0015,
            output_cost_per_1k: 0.002,
            inline_threshold: 4096,
            distance_metric: DistanceMetric::Cosine,
            auto_select_metric: true,
            sparsity_metric_threshold: 0.7,
        }
    }
}

impl CacheConfig {
    #[must_use]
    pub fn high_throughput() -> Self {
        Self {
            exact_capacity: 50_000,
            semantic_capacity: 20_000,
            embedding_capacity: 100_000,
            eviction_interval: Duration::from_secs(30),
            eviction_batch_size: 500,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn low_memory() -> Self {
        Self {
            exact_capacity: 1_000,
            semantic_capacity: 500,
            embedding_capacity: 5_000,
            eviction_interval: Duration::from_secs(30),
            eviction_batch_size: 50,
            inline_threshold: 1024,
            ..Self::default()
        }
    }

    #[must_use]
    pub fn development() -> Self {
        Self {
            exact_capacity: 100,
            semantic_capacity: 50,
            embedding_capacity: 200,
            default_ttl: Duration::from_secs(60),
            eviction_interval: Duration::from_secs(5),
            eviction_batch_size: 10,
            ..Self::default()
        }
    }

    /// # Errors
    ///
    /// Returns error if thresholds are out of range or TTL constraints are violated.
    pub fn validate(&self) -> Result<(), String> {
        if self.semantic_threshold < 0.0 || self.semantic_threshold > 1.0 {
            return Err(format!(
                "semantic_threshold must be between 0.0 and 1.0, got {}",
                self.semantic_threshold
            ));
        }

        if self.embedding_dim == 0 {
            return Err("embedding_dim must be greater than 0".into());
        }

        if self.eviction_batch_size == 0 {
            return Err("eviction_batch_size must be greater than 0".into());
        }

        if self.default_ttl > self.max_ttl {
            return Err("default_ttl cannot exceed max_ttl".into());
        }

        if self.sparsity_metric_threshold < 0.0 || self.sparsity_metric_threshold > 1.0 {
            return Err(format!(
                "sparsity_metric_threshold must be between 0.0 and 1.0, got {}",
                self.sparsity_metric_threshold
            ));
        }

        Ok(())
    }

    #[must_use]
    pub fn sparse_embeddings() -> Self {
        Self {
            distance_metric: DistanceMetric::Jaccard,
            auto_select_metric: true,
            sparsity_metric_threshold: 0.5,
            semantic_threshold: 0.85,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CacheConfig::default();
        assert_eq!(config.exact_capacity, 10_000);
        assert_eq!(config.semantic_capacity, 5_000);
        assert_eq!(config.embedding_capacity, 50_000);
        assert!((config.semantic_threshold - 0.92).abs() < 0.001);
        assert_eq!(config.embedding_dim, 1536);
    }

    #[test]
    fn test_config_presets() {
        let high = CacheConfig::high_throughput();
        assert_eq!(high.exact_capacity, 50_000);

        let low = CacheConfig::low_memory();
        assert_eq!(low.exact_capacity, 1_000);

        let dev = CacheConfig::development();
        assert_eq!(dev.exact_capacity, 100);
    }

    #[test]
    fn test_config_validation() {
        let mut config = CacheConfig::default();
        assert!(config.validate().is_ok());

        config.semantic_threshold = 1.5;
        assert!(config.validate().is_err());

        config.semantic_threshold = 0.92;
        config.embedding_dim = 0;
        assert!(config.validate().is_err());

        config.embedding_dim = 1536;
        config.default_ttl = Duration::from_secs(100000);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_eviction_strategy_default() {
        let strategy = EvictionStrategy::default();
        assert!(matches!(strategy, EvictionStrategy::Hybrid { .. }));
    }

    #[test]
    fn test_config_serialize() {
        let config = CacheConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: CacheConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.exact_capacity, restored.exact_capacity);
    }

    #[test]
    fn test_config_distance_metric_default() {
        let config = CacheConfig::default();
        assert_eq!(config.distance_metric, DistanceMetric::Cosine);
        assert!(config.auto_select_metric);
        assert!((config.sparsity_metric_threshold - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_config_sparse_embeddings_preset() {
        let config = CacheConfig::sparse_embeddings();
        assert_eq!(config.distance_metric, DistanceMetric::Jaccard);
        assert!(config.auto_select_metric);
        assert!((config.sparsity_metric_threshold - 0.5).abs() < 0.001);
        assert!((config.semantic_threshold - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_config_validation_sparsity_threshold() {
        let mut config = CacheConfig::default();
        assert!(config.validate().is_ok());

        config.sparsity_metric_threshold = 1.5;
        assert!(config.validate().is_err());

        config.sparsity_metric_threshold = -0.1;
        assert!(config.validate().is_err());

        config.sparsity_metric_threshold = 0.0;
        assert!(config.validate().is_ok());

        config.sparsity_metric_threshold = 1.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialize_with_distance_metric() {
        let mut config = CacheConfig::default();
        config.distance_metric = DistanceMetric::Jaccard;
        config.auto_select_metric = false;
        config.sparsity_metric_threshold = 0.8;

        let json = serde_json::to_string(&config).unwrap();
        let restored: CacheConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.distance_metric, DistanceMetric::Jaccard);
        assert!(!restored.auto_select_metric);
        assert!((restored.sparsity_metric_threshold - 0.8).abs() < 0.001);
    }
}
