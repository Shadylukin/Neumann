# Tensor Cache

Module 10 of Neumann. Semantic caching for LLM responses with cost tracking and background eviction.

## Design Principles

1. **Multi-Layer Caching**: Exact O(1), Semantic O(log n), Embedding O(1) lookups
2. **Cost-Aware**: Tracks tokens and estimates savings using tiktoken
3. **Background Eviction**: Async eviction with configurable strategies
4. **TTL Expiration**: Time-based entry expiration with min-heap tracking
5. **Thread-Safe**: All operations are concurrent via DashMap
6. **Zero Allocation Lookup**: Embeddings stored inline, not as pointers

## Quick Start

```rust
use tensor_cache::{Cache, CacheConfig, CacheHit};

// Create cache with default configuration
let cache = Cache::new();

// Store a response (with embedding for semantic matching)
let embedding = vec![0.1, 0.2, 0.3, /* ... */];
cache.put(
    "What is 2+2?",
    &embedding,
    "4",
    "gpt-4",
    0, // params_hash
)?;

// Look up (tries exact first, then semantic if embedding provided)
if let Some(hit) = cache.get("What is 2+2?", Some(&embedding)) {
    println!("Cached: {} (saved ${:.4})", hit.response, hit.cost_saved);
}
```

## Cache Layers

### Exact Cache (O(1))
Hash-based lookup for identical queries. Keys are generated from the prompt text.

### Semantic Cache (O(log n))
HNSW-based similarity search. Finds responses to semantically similar queries.

### Embedding Cache (O(1))
Stores precomputed embeddings to avoid redundant embedding API calls.

## Configuration

```rust
use tensor_cache::{CacheConfig, EvictionStrategy};
use std::time::Duration;

let config = CacheConfig {
    // Capacity limits
    exact_capacity: 10_000,
    semantic_capacity: 5_000,
    embedding_capacity: 50_000,

    // TTL and thresholds
    default_ttl: Duration::from_secs(3600),
    semantic_threshold: 0.92,
    embedding_dim: 1536,

    // Eviction settings
    eviction_strategy: EvictionStrategy::Hybrid {
        lru_weight: 40,
        lfu_weight: 30,
        cost_weight: 30,
    },
    eviction_interval: Duration::from_secs(60),
    eviction_batch_size: 100,

    // Cost tracking (per 1000 tokens)
    input_cost_per_1k: 0.005,
    output_cost_per_1k: 0.015,
};

let cache = Cache::with_config(config)?;
```

### Configuration Presets

```rust
// High-throughput server
let config = CacheConfig::high_throughput();

// Memory-constrained environment
let config = CacheConfig::low_memory();

// Development/testing
let config = CacheConfig::development();

// Sparse embeddings (uses Jaccard by default)
let config = CacheConfig::sparse_embeddings();
```

## Distance Metrics

tensor_cache supports configurable distance metrics for semantic similarity via integration with tensor_store geometric primitives.

### Available Metrics

| Metric | Best For | Range |
|--------|----------|-------|
| Cosine | Dense embeddings (default) | [-1, 1] |
| Angular | Linear angle relationships | [0, PI] |
| Jaccard | Sparse/binary embeddings | [0, 1] |
| Euclidean | Absolute distances | [0, inf) |
| WeightedJaccard | Sparse with magnitudes | [0, 1] |

### Metric Configuration

```rust
use tensor_cache::{CacheConfig, DistanceMetric};

let config = CacheConfig {
    distance_metric: DistanceMetric::Jaccard,
    auto_select_metric: true,
    sparsity_metric_threshold: 0.7,
    ..Default::default()
};
```

### Auto-Selection

When `auto_select_metric` is true (default), the cache automatically
selects the best metric based on embedding sparsity:

- Sparsity >= threshold (default 70%): Uses Jaccard (structural similarity)
- Sparsity < threshold: Uses configured metric (default: Cosine)

```rust
// Dense embedding (10% zeros) -> Uses Cosine
let dense = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

// Sparse embedding (80% zeros) -> Uses Jaccard
let sparse = vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0];
```

### Explicit Metric Queries

```rust
use tensor_cache::DistanceMetric;

// Query with explicit metric
let hit = cache.get_with_metric(
    "query",
    Some(&embedding),
    Some(&DistanceMetric::Euclidean),
);

if let Some(hit) = hit {
    println!("Metric used: {:?}", hit.metric_used);
}
```

## Eviction Strategies

### LRU (Least Recently Used)
Evicts entries that haven't been accessed recently.

### LFU (Least Frequently Used)
Evicts entries with the lowest access count.

### Cost-Based
Evicts entries with the lowest cost efficiency (cost saved per byte).

### Hybrid (Recommended)
Combines all strategies with configurable weights:
```rust
EvictionStrategy::Hybrid {
    lru_weight: 40,   // 40% recency
    lfu_weight: 30,   // 30% frequency
    cost_weight: 30,  // 30% cost efficiency
}
```

## Token Counting

Uses tiktoken for accurate GPT-4 compatible token counting:

```rust
use tensor_cache::{TokenCounter, ModelPricing};

// Count tokens
let tokens = TokenCounter::count("Hello, world!")?;

// Estimate cost
let cost = TokenCounter::estimate_cost(1000, 500, 0.01, 0.03);

// Use predefined pricing
let pricing = ModelPricing::GPT4O;
let cost = pricing.estimate(1000, 500);
```

### Supported Models

| Model | Input/1K | Output/1K |
|-------|----------|-----------|
| GPT-4o | $0.005 | $0.015 |
| GPT-4o mini | $0.00015 | $0.0006 |
| GPT-4 Turbo | $0.01 | $0.03 |
| GPT-3.5 Turbo | $0.0005 | $0.0015 |
| Claude 3 Opus | $0.015 | $0.075 |
| Claude 3 Sonnet | $0.003 | $0.015 |
| Claude 3 Haiku | $0.00025 | $0.00125 |

## Statistics

```rust
let stats = cache.stats_snapshot();

println!("Hit rate: {:.2}%", stats.overall_hit_rate() * 100.0);
println!("Tokens saved: {}", stats.total_tokens_saved());
println!("Cost saved: ${:.4}", stats.cost_saved_dollars());
println!("Entries: {}", stats.total_entries());
```

## Integration with Query Router

```rust
use query_router::QueryRouter;
use tensor_cache::CacheConfig;

let mut router = QueryRouter::new();

// Initialize with default config
router.init_cache();

// Or with custom config
router.init_cache_with_config(CacheConfig::high_throughput());

// Access cache
if let Some(cache) = router.cache() {
    let stats = cache.stats_snapshot();
    println!("Cache hit rate: {:.2}%", stats.overall_hit_rate() * 100.0);
}
```

## Shell Commands

```
CACHE INIT     Initialize semantic cache
CACHE STATS    Show cache statistics
CACHE CLEAR    Clear all cache entries
```

## API Reference

### Cache

| Method | Description |
|--------|-------------|
| `new()` | Create with default config |
| `with_config(config)` | Create with custom config |
| `get(prompt, embedding)` | Look up cached response |
| `get_with_metric(prompt, embedding, metric)` | Look up with explicit metric |
| `put(prompt, embedding, response, model, params_hash)` | Store response |
| `get_embedding(source, content)` | Get cached embedding |
| `put_embedding(source, content, embedding, model)` | Store embedding |
| `invalidate(prompt, model, params_hash)` | Remove exact entry |
| `evict(count)` | Manually evict entries |
| `stats()` | Get statistics reference |
| `len()` | Total cached entries |

### CacheHit

| Field | Type | Description |
|-------|------|-------------|
| `response` | `String` | Cached response text |
| `layer` | `CacheLayer` | Which layer matched |
| `similarity` | `Option<f32>` | Similarity score (semantic only) |
| `input_tokens` | `usize` | Input tokens saved |
| `output_tokens` | `usize` | Output tokens saved |
| `cost_saved` | `f64` | Estimated cost saved (dollars) |
| `metric_used` | `Option<DistanceMetric>` | Metric used (semantic only) |

### CacheLayer

- `Exact` - Matched via hash lookup
- `Semantic` - Matched via similarity search
- `Embedding` - Embedding cache hit

## Performance

### Benchmarks (10,000 entries, 128-dim embeddings)

| Operation | Time |
|-----------|------|
| Exact lookup (hit) | ~50ns |
| Exact lookup (miss) | ~30ns |
| Semantic lookup | ~5us |
| Put (exact + semantic) | ~10us |
| Eviction (100 entries) | ~200us |

### Distance Metric Performance (128-dim, 1000 entries)

| Metric | Search Time | Notes |
|--------|-------------|-------|
| Cosine | 21 us | Default, best for dense |
| Jaccard | 18 us | Best for sparse |
| Angular | 23 us | +acos overhead |
| Euclidean | 19 us | Absolute distance |

### Auto-Selection Overhead

| Operation | Time |
|-----------|------|
| Sparsity check | ~50 ns |
| Metric selection | ~10 ns |

## Error Handling

```rust
use tensor_cache::{CacheError, Result};

match cache.put("prompt", &embedding, "response", "model", 0) {
    Ok(()) => println!("Stored"),
    Err(CacheError::DimensionMismatch { expected, got }) => {
        println!("Expected {expected}-dim, got {got}");
    }
    Err(CacheError::CacheFull { current, capacity }) => {
        println!("Cache full: {current} entries >= {capacity} capacity");
    }
    Err(e) => println!("Error: {e}"),
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `NotFound` | Cache entry not found |
| `DimensionMismatch` | Embedding dimension does not match config |
| `StorageError` | Underlying tensor store error |
| `SerializationError` | Serialization/deserialization failed |
| `TokenizerError` | Token counting failed |
| `CacheFull` | Cache capacity exceeded |
| `InvalidConfig` | Invalid configuration provided |
| `Cancelled` | Operation was cancelled |

## Test Coverage

The tensor_cache module maintains 95%+ line coverage across all source files:

| File | Coverage |
|------|----------|
| config.rs | 99% |
| embedding.rs | 99% |
| error.rs | 98% |
| eviction.rs | 99% |
| exact.rs | 97% |
| index.rs | 99% |
| lib.rs | 95% |
| semantic.rs | 99% |
| stats.rs | 100% |
| tokenizer.rs | 96% |
| ttl.rs | 95% |

## Architecture

```
+--------------------------------------------------+
|                  Cache (Public API)               |
|   - get(prompt, embedding) -> CacheHit           |
|   - put(prompt, embedding, response, ...)        |
|   - stats(), evict(), clear()                    |
+--------------------------------------------------+
            |           |           |
    +-------+    +------+    +------+
    |            |           |
+--------+  +----------+  +-----------+
| Exact  |  | Semantic |  | Embedding |
| Cache  |  |  Cache   |  |   Cache   |
| O(1)   |  | O(log n) |  |   O(1)    |
+--------+  +----------+  +-----------+
    |            |           |
    +-------+----+----+------+
            |
    +------------------+
    |   CacheIndex     |
    |  (HNSW wrapper)  |
    +------------------+
            |
    +------------------+
    |   tensor_store   |
    |     hnsw.rs      |
    +------------------+
```

## Dependencies

- `tensor_store` - HNSW index implementation
- `tiktoken-rs` - GPT-compatible token counting
- `dashmap` - Concurrent hash maps
- `tokio` - Async runtime for background eviction
- `uuid` - Unique ID generation
- `thiserror` - Error type derivation
