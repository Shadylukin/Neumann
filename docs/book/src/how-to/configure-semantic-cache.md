# How to: Configure Semantic Cache

Step-by-step guides for configuring tensor_cache: setting up cache layers,
choosing TTLs, tuning similarity thresholds, selecting eviction policies, and
optimizing for different workloads. This guide also covers eviction tuning, which
is tightly coupled with cache configuration.

For the full type reference, see the
[Tensor Cache API Reference](../reference/api/tensor-cache.md). For the design
rationale, see
[Semantic Caching Design](../explanation/semantic-caching.md).

## Basic Setup

```rust
use tensor_cache::{Cache, CacheConfig};

let mut config = CacheConfig::default();
config.embedding_dim = 3;
let cache = Cache::with_config(config).unwrap();
```

Shell:

```text
CACHE INIT
```

## Store and Look Up a Response

```rust
let embedding = vec![0.1, 0.2, 0.3];
cache.put("What is 2+2?", &embedding, "4", "gpt-4", None).unwrap();

// Tries exact first, then semantic
if let Some(hit) = cache.get("What is 2+2?", Some(&embedding)) {
    println!("Cached: {}", hit.response);
}
```

## Use an Explicit Distance Metric

```rust
use tensor_cache::DistanceMetric;

let hit = cache.get_with_metric(
    "query",
    Some(&embedding),
    Some(&DistanceMetric::Euclidean),
);
```

## Cache Embeddings with Compute Fallback

Avoid redundant embedding API calls:

```rust
let embedding = cache.get_or_compute_embedding(
    "openai",                          // source
    "Hello, world!",                   // content
    "text-embedding-3-small",          // model
    || {
        Ok(compute_embedding("Hello, world!"))
    }
)?;
```

## Share a TensorStore with Other Engines

```rust
use tensor_store::TensorStore;

let store = TensorStore::new();
let cache = Cache::with_store(store.clone(), CacheConfig::default())?;
let vault = Vault::with_store(store.clone(), VaultConfig::default())?;
```

## Choose a Configuration Preset

| Preset | Use Case | Exact | Semantic | Embedding | Batch |
| --- | --- | --- | --- | --- | --- |
| `default()` | General purpose | 10,000 | 5,000 | 50,000 | 100 |
| `high_throughput()` | High-traffic server | 50,000 | 20,000 | 100,000 | 500 |
| `low_memory()` | Memory-constrained | 1,000 | 500 | 5,000 | 50 |
| `development()` | Dev/testing | 100 | 50 | 200 | 10 |
| `sparse_embeddings()` | Sparse vectors | 10,000 | 5,000 | 50,000 | 100 |

```rust
let config = CacheConfig::high_throughput();
let cache = Cache::with_config(config)?;
```

## Tune the Semantic Similarity Threshold

The `semantic_threshold` controls how similar a query must be to return a cache
hit. Lower values increase hit rates but risk returning less relevant responses.

| Threshold | Effect |
| --- | --- |
| 0.95 | Very strict -- nearly identical queries only |
| 0.92 | Default -- good balance of precision and recall |
| 0.85 | Fuzzy -- catches more paraphrases, some false positives |

```rust
let mut config = CacheConfig::default();
config.semantic_threshold = 0.85;  // More permissive matching
```

## Configure TTL

Set default and maximum time-to-live for cache entries:

```rust
let mut config = CacheConfig::default();
config.default_ttl = Duration::from_secs(1800);   // 30 minutes
config.max_ttl = Duration::from_secs(86400);       // 24 hours
```

Per-entry TTL can be specified on `put()`:

```rust
cache.put("query", &embedding, "response", "model",
    Some(Duration::from_secs(600))  // 10 minutes
)?;
```

Entries with `expires_at = 0` never expire. Expired entries return `None` on
lookup but remain in storage until `cleanup_expired()` runs (either explicitly
or via background eviction).

## Choose an Eviction Strategy

### LRU (Least Recently Used)

Evicts entries that have not been accessed recently. Best for general-purpose
workloads.

```rust
config.eviction_strategy = EvictionStrategy::LRU;
```

### LFU (Least Frequently Used)

Evicts entries with the lowest access count. Best for stable workloads where
popular queries should persist.

```rust
config.eviction_strategy = EvictionStrategy::LFU;
```

### CostBased

Evicts entries with the lowest cost savings per byte. Best when API costs are the
primary concern.

```rust
config.eviction_strategy = EvictionStrategy::CostBased;
```

### Hybrid (Recommended for Production)

Combines recency, frequency, and cost factors with configurable weights:

```rust
config.eviction_strategy = EvictionStrategy::Hybrid {
    lru_weight: 40,   // Recency importance
    lfu_weight: 30,   // Frequency importance
    cost_weight: 30,  // Cost savings importance
};
```

Tuning tips:
- Increase `cost_weight` if API costs are your primary concern
- Increase `lru_weight` for workloads with strong temporal locality
- Increase `lfu_weight` for workloads with a stable set of popular queries

## Configure Background Eviction

Background eviction runs as an async Tokio task:

```rust
config.eviction_interval = Duration::from_secs(60);  // Check every minute
config.eviction_batch_size = 100;                     // Process 100 entries per cycle
```

Starting and stopping:

```rust
let handle = manager.start(move |batch_size| {
    cache.evict(batch_size)
});

// Later: graceful shutdown
handle.shutdown().await;
```

### Tuning Batch Size

| Batch Size | Best For |
| --- | --- |
| 10-50 | Low-memory or dev environments |
| 100 | General purpose (default) |
| 500+ | High-throughput production systems |

Larger batches reduce eviction overhead per entry but increase per-cycle latency.

## Configure Auto Metric Selection

Enable automatic metric selection based on embedding sparsity:

```rust
config.auto_select_metric = true;
config.sparsity_metric_threshold = 0.7;  // 70% zeros -> Jaccard
config.distance_metric = DistanceMetric::Cosine;  // Default for dense
```

When `auto_select_metric` is true:
- Embeddings with sparsity >= threshold use Jaccard (better for sparse vectors)
- All other embeddings use the configured `distance_metric`

## Monitor Cache Performance

```rust
let stats = cache.stats_snapshot();

// Hit rates by layer
println!("Exact hit rate: {:.2}%", stats.hit_rate(CacheLayer::Exact) * 100.0);
println!("Semantic hit rate: {:.2}%", stats.hit_rate(CacheLayer::Semantic) * 100.0);

// Tokens and cost saved
println!("Input tokens saved: {}", stats.tokens_saved_in);
println!("Output tokens saved: {}", stats.tokens_saved_out);
println!("Cost saved: ${:.2}", stats.cost_saved_dollars);

// Cache utilization
println!("Total entries: {}", stats.total_entries());
println!("Evictions: {}", stats.evictions);
println!("Expirations: {}", stats.expirations);
```

Shell:

```text
CACHE STATS
```

## Track Token Costs

```rust
use tensor_cache::{TokenCounter, ModelPricing};

// Count tokens
let tokens = TokenCounter::count("Hello, world!");

// Count chat messages (includes per-message overhead)
let messages = vec![("user", "Hello"), ("assistant", "Hi!")];
let total = TokenCounter::count_messages(&messages);

// Estimate cost with predefined pricing
let pricing = ModelPricing::GPT4O;
let cost = pricing.estimate(1000, 500);

// Lookup pricing by model name
if let Some(pricing) = ModelPricing::for_model("gpt-4o-mini") {
    println!("Cost: ${:.4}", pricing.estimate(1000, 500));
}
```

Use `estimate_cost_microdollars()` for atomic accumulation to avoid floating
point errors.

## Optimize Hit Rates

1. **Normalize prompts** before caching: lowercase, trim whitespace
2. **Use versioning** for model or prompt template changes
3. **Set appropriate semantic threshold** for your domain (start at 0.92, lower
   if needed)
4. **Consider domain-specific embeddings** for better semantic matching

## Optimize Memory

1. Use the `sparse_embeddings()` preset for sparse data
2. Set `inline_threshold` based on typical response sizes
3. Enable `auto_select_metric` for mixed dense/sparse workloads
4. Monitor `memory_stats()` to track sparse vs dense ratio

## Optimize Cost Tracking

1. Use `estimate_cost_microdollars()` for atomic accumulation
2. Record cost per cache hit for ROI analysis
3. Compare `tokens_saved` against cache capacity costs

## See Also

- [Tensor Cache API Reference](../reference/api/tensor-cache.md) -- complete
  type tables, error types, and method signatures
- [Semantic Caching Design](../explanation/semantic-caching.md) -- multi-layer
  architecture, HNSW internals, and eviction algorithm details
