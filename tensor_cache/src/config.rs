//! Configuration for the cache module.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tensor_store::DistanceMetric;

/// Eviction strategy for cache entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Least Recently Used - evicts entries that haven't been accessed recently.
    LRU,
    /// Least Frequently Used - evicts entries with lowest access count.
    LFU,
    /// Cost-based - evicts entries with lowest cost savings per byte.
    CostBased,
    /// Hybrid strategy combining LRU, LFU, and cost factors.
    Hybrid {
        lru_weight: u8,
        lfu_weight: u8,
        cost_weight: u8,
    },
}

impl Default for EvictionStrategy {
    fn default() -> Self {
        EvictionStrategy::Hybrid {
            lru_weight: 40,
            lfu_weight: 30,
            cost_weight: 30,
        }
    }
}

/// Configuration for the cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in the exact cache layer.
    pub exact_capacity: usize,

    /// Maximum number of entries in the semantic cache layer.
    pub semantic_capacity: usize,

    /// Maximum number of entries in the embedding cache layer.
    pub embedding_capacity: usize,

    /// Default time-to-live for cache entries.
    pub default_ttl: Duration,

    /// Maximum time-to-live for cache entries.
    pub max_ttl: Duration,

    /// Minimum similarity threshold for semantic cache hits (0.0 to 1.0).
    pub semantic_threshold: f32,

    /// Expected embedding dimension (e.g., 1536 for OpenAI).
    pub embedding_dim: usize,

    /// Eviction strategy to use.
    pub eviction_strategy: EvictionStrategy,

    /// How often to run the eviction cycle.
    pub eviction_interval: Duration,

    /// Number of entries to evict per cycle.
    pub eviction_batch_size: usize,

    /// Cost per 1000 input tokens (in dollars).
    pub input_cost_per_1k: f64,

    /// Cost per 1000 output tokens (in dollars).
    pub output_cost_per_1k: f64,

    /// Size threshold for inline vs. pointer storage (bytes).
    pub inline_threshold: usize,

    /// Distance metric for semantic similarity search.
    pub distance_metric: DistanceMetric,

    /// Auto-select metric based on embedding sparsity.
    /// When true, uses Jaccard for sparse embeddings (sparsity >= threshold).
    pub auto_select_metric: bool,

    /// Sparsity threshold for auto-selecting Jaccard over configured metric.
    /// Embeddings with sparsity >= this value use Jaccard, others use distance_metric.
    pub sparsity_metric_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            exact_capacity: 10_000,
            semantic_capacity: 5_000,
            embedding_capacity: 50_000,
            default_ttl: Duration::from_secs(3600), // 1 hour
            max_ttl: Duration::from_secs(86400),    // 24 hours
            semantic_threshold: 0.92,
            embedding_dim: 1536,
            eviction_strategy: EvictionStrategy::default(),
            eviction_interval: Duration::from_secs(60),
            eviction_batch_size: 100,
            input_cost_per_1k: 0.0015, // GPT-4 Turbo input
            output_cost_per_1k: 0.002, // GPT-4 Turbo output
            inline_threshold: 4096,    // 4KB
            distance_metric: DistanceMetric::Cosine,
            auto_select_metric: true,
            sparsity_metric_threshold: 0.7,
        }
    }
}

impl CacheConfig {
    /// Create a configuration optimized for high throughput.
    pub fn high_throughput() -> Self {
        Self {
            exact_capacity: 50_000,
            semantic_capacity: 20_000,
            embedding_capacity: 100_000,
            eviction_interval: Duration::from_secs(30),
            eviction_batch_size: 500,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for low memory usage.
    pub fn low_memory() -> Self {
        Self {
            exact_capacity: 1_000,
            semantic_capacity: 500,
            embedding_capacity: 5_000,
            eviction_interval: Duration::from_secs(30),
            eviction_batch_size: 50,
            inline_threshold: 1024,
            ..Default::default()
        }
    }

    /// Create a configuration for development/testing.
    pub fn development() -> Self {
        Self {
            exact_capacity: 100,
            semantic_capacity: 50,
            embedding_capacity: 200,
            default_ttl: Duration::from_secs(60),
            eviction_interval: Duration::from_secs(5),
            eviction_batch_size: 10,
            ..Default::default()
        }
    }

    /// Validate the configuration.
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

    /// Create a configuration optimized for sparse embeddings.
    /// Uses Jaccard metric by default and aggressive auto-selection.
    pub fn sparse_embeddings() -> Self {
        Self {
            distance_metric: DistanceMetric::Jaccard,
            auto_select_metric: true,
            sparsity_metric_threshold: 0.5, // More aggressive sparse detection
            semantic_threshold: 0.85,       // Lower threshold for structural matching
            ..Default::default()
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
