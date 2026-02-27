# Tensor Cache API Reference

Complete type reference, configuration tables, storage patterns, error types, and
method signatures for the `tensor_cache` crate.

## Core Types

| Type | Description |
| --- | --- |
| `Cache` | Main API - multi-layer LLM response cache |
| `CacheConfig` | Configuration (capacity, TTL, eviction, metrics) |
| `CacheHit` | Successful cache lookup result |
| `CacheStats` | Thread-safe statistics with atomic counters |
| `StatsSnapshot` | Point-in-time snapshot for reporting |
| `CacheLayer` | Enum: `Exact`, `Semantic`, `Embedding` |
| `CacheError` | Error types for cache operations |

## Configuration Types

| Type | Description |
| --- | --- |
| `EvictionStrategy` | `LRU`, `LFU`, `CostBased`, `Hybrid` |
| `EvictionManager` | Background eviction task controller |
| `EvictionScorer` | Calculates eviction priority scores |
| `EvictionHandle` | Handle for controlling background eviction |
| `EvictionConfig` | Interval, batch size, and strategy settings |

## Token Counting Types

| Type | Description |
| --- | --- |
| `TokenCounter` | GPT-4 compatible token counting via tiktoken |
| `ModelPricing` | Predefined pricing for GPT-4, Claude 3, etc. |

## Index Types (Internal)

| Type | Description |
| --- | --- |
| `CacheIndex` | HNSW wrapper with key-to-node mapping |
| `IndexSearchResult` | Semantic search result with similarity score |

## Error Types

| Error | Description | Recovery |
| --- | --- | --- |
| `NotFound` | Cache entry not found | Check key exists |
| `DimensionMismatch` | Embedding dimension does not match config | Verify embedding size |
| `StorageError` | Underlying tensor store error | Check store health |
| `SerializationError` | Serialization/deserialization failed | Verify data format |
| `TokenizerError` | Token counting failed | Falls back to estimation |
| `CacheFull` | Cache capacity exceeded | Run eviction or increase capacity |
| `InvalidConfig` | Invalid configuration provided | Fix config values |
| `Cancelled` | Operation was cancelled | Retry operation |
| `LockPoisoned` | Internal lock was poisoned | Restart cache |

### Error Conversion

```rust
impl From<tensor_store::TensorStoreError> for CacheError {
    fn from(e: TensorStoreError) -> Self {
        Self::StorageError(e.to_string())
    }
}

impl From<bitcode::Error> for CacheError {
    fn from(e: bitcode::Error) -> Self {
        Self::SerializationError(e.to_string())
    }
}
```

## Cache Methods

| Method | Description |
| --- | --- |
| `new()` | Create with default config |
| `with_config(config)` | Create with custom config |
| `with_store(store, config)` | Create with shared TensorStore |
| `get(prompt, embedding)` | Look up cached response |
| `get_with_metric(prompt, embedding, metric)` | Look up with explicit metric |
| `put(prompt, embedding, response, model, ttl)` | Store response |
| `get_embedding(source, content)` | Get cached embedding |
| `put_embedding(source, content, embedding, model)` | Store embedding |
| `get_or_compute_embedding(source, content, model, compute)` | Get or compute embedding |
| `get_simple(key)` | Simple key-value lookup |
| `put_simple(key, value)` | Simple key-value store |
| `invalidate(prompt)` | Remove exact entry |
| `invalidate_version(version)` | Remove entries by version |
| `invalidate_embeddings(source)` | Remove embeddings by source |
| `evict(count)` | Manually evict entries |
| `cleanup_expired()` | Remove expired entries |
| `clear()` | Clear all entries |
| `stats()` | Get statistics reference |
| `stats_snapshot()` | Get statistics snapshot |
| `config()` | Get configuration reference |
| `len()` | Total cached entries |
| `is_empty()` | Check if cache is empty |

## CacheHit Fields

| Field | Type | Description |
| --- | --- | --- |
| `response` | `String` | Cached response text |
| `layer` | `CacheLayer` | Which layer matched |
| `similarity` | `Option<f32>` | Similarity score (semantic only) |
| `input_tokens` | `usize` | Input tokens saved |
| `output_tokens` | `usize` | Output tokens saved |
| `cost_saved` | `f64` | Estimated cost saved (dollars) |
| `metric_used` | `Option<DistanceMetric>` | Metric used (semantic only) |

## StatsSnapshot Fields

| Field | Type | Description |
| --- | --- | --- |
| `exact_hits` | `u64` | Exact cache hits |
| `exact_misses` | `u64` | Exact cache misses |
| `semantic_hits` | `u64` | Semantic cache hits |
| `semantic_misses` | `u64` | Semantic cache misses |
| `embedding_hits` | `u64` | Embedding cache hits |
| `embedding_misses` | `u64` | Embedding cache misses |
| `tokens_saved_in` | `u64` | Total input tokens saved |
| `tokens_saved_out` | `u64` | Total output tokens saved |
| `cost_saved_dollars` | `f64` | Total cost saved |
| `evictions` | `u64` | Total evictions |
| `expirations` | `u64` | Total expirations |
| `exact_size` | `usize` | Current exact cache size |
| `semantic_size` | `usize` | Current semantic cache size |
| `embedding_size` | `usize` | Current embedding cache size |
| `uptime_secs` | `u64` | Cache uptime in seconds |

## Cache Layers

| Layer | Complexity | Key Prefix | Best For |
| --- | --- | --- | --- |
| Exact | O(1) | `_cache:exact:` | Identical queries (FAQ, canned responses) |
| Semantic | O(log n) | `_cache:sem:` | Natural language variations |
| Embedding | O(1) | `_cache:emb:` | Precomputed embedding reuse |

## Storage Format

Cache entries are stored as `TensorData` with standardized fields:

| Field | Type | Description |
| --- | --- | --- |
| `_response` | String | Cached response text |
| `_embedding` | Vector/Sparse | Embedding (semantic/embedding layers) |
| `_embedding_dim` | Int | Embedding dimension |
| `_input_tokens` | Int | Input token count |
| `_output_tokens` | Int | Output token count |
| `_model` | String | Model identifier |
| `_layer` | String | Cache layer (exact/semantic/embedding) |
| `_created_at` | Int | Creation timestamp (millis) |
| `_expires_at` | Int | Expiration timestamp (millis) |
| `_access_count` | Int | Access count for LFU |
| `_last_access` | Int | Last access timestamp for LRU |
| `_version` | String | Optional version tag |
| `_source` | String | Embedding source identifier |
| `_content_hash` | Int | Content hash for deduplication |

## Distance Metrics

| Metric | Best For | Range | Formula |
| --- | --- | --- | --- |
| Cosine | Dense embeddings (default) | -1 to 1 | `dot(a,b) / (||a|| * ||b||)` |
| Angular | Linear angle relationships | 0 to PI | `acos(cosine_sim)` |
| Jaccard | Sparse/binary embeddings | 0 to 1 | `||A n B|| / ||A u B||` |
| Euclidean | Absolute distances | 0 to inf | `sqrt(sum((a-b)^2))` |
| WeightedJaccard | Sparse with magnitudes | 0 to 1 | Weighted set similarity |

Auto-selection: when `auto_select_metric` is true, the cache uses Jaccard for
sparse embeddings (sparsity >= threshold, default 70%) and the configured metric
otherwise.

## Configuration

### Default Configuration

```rust
CacheConfig {
    exact_capacity: 10_000,
    semantic_capacity: 5_000,
    embedding_capacity: 50_000,
    default_ttl: Duration::from_secs(3600),
    max_ttl: Duration::from_secs(86400),
    semantic_threshold: 0.92,
    embedding_dim: 1536,
    eviction_strategy: EvictionStrategy::Hybrid {
        lru_weight: 40,
        lfu_weight: 30,
        cost_weight: 30
    },
    eviction_interval: Duration::from_secs(60),
    eviction_batch_size: 100,
    input_cost_per_1k: 0.0015,
    output_cost_per_1k: 0.002,
    inline_threshold: 4096,
    distance_metric: DistanceMetric::Cosine,
    auto_select_metric: true,
    sparsity_metric_threshold: 0.7,
}
```

### Configuration Presets

| Preset | Use Case | Exact | Semantic | Embedding | Eviction Batch |
| --- | --- | --- | --- | --- | --- |
| `default()` | General purpose | 10,000 | 5,000 | 50,000 | 100 |
| `high_throughput()` | High-traffic server | 50,000 | 20,000 | 100,000 | 500 |
| `low_memory()` | Memory-constrained | 1,000 | 500 | 5,000 | 50 |
| `development()` | Dev/testing | 100 | 50 | 200 | 10 |
| `sparse_embeddings()` | Sparse vectors | 10,000 | 5,000 | 50,000 | 100 |

### Configuration Validation

```rust
pub fn validate(&self) -> Result<(), String> {
    if self.semantic_threshold < 0.0 || self.semantic_threshold > 1.0 {
        return Err("semantic_threshold must be between 0.0 and 1.0");
    }
    if self.embedding_dim == 0 {
        return Err("embedding_dim must be greater than 0");
    }
    if self.eviction_batch_size == 0 {
        return Err("eviction_batch_size must be greater than 0");
    }
    if self.default_ttl > self.max_ttl {
        return Err("default_ttl cannot exceed max_ttl");
    }
    if self.sparsity_metric_threshold < 0.0 || self.sparsity_metric_threshold > 1.0 {
        return Err("sparsity_metric_threshold must be between 0.0 and 1.0");
    }
    Ok(())
}
```

## Model Pricing

| Model | Input/1K | Output/1K | Notes |
| --- | --- | --- | --- |
| GPT-4o | $0.005 | $0.015 | Best for complex tasks |
| GPT-4o mini | $0.00015 | $0.0006 | Cost-effective |
| GPT-4 Turbo | $0.01 | $0.03 | High capability |
| GPT-3.5 Turbo | $0.0005 | $0.0015 | Budget option |
| Claude 3 Opus | $0.015 | $0.075 | Highest quality |
| Claude 3 Sonnet | $0.003 | $0.015 | Balanced |
| Claude 3 Haiku | $0.00025 | $0.00125 | Fast and cheap |

## CacheIndex Structure

```rust
pub struct CacheIndex {
    index: RwLock<HNSWIndex>,           // HNSW graph
    config: HNSWConfig,                  // For recreation on clear
    key_to_node: DashMap<String, usize>, // Cache key -> HNSW node
    node_to_key: DashMap<usize, String>, // HNSW node -> Cache key
    dimension: usize,                    // Expected embedding dimension
    entry_count: AtomicUsize,            // Entry count
    distance_metric: DistanceMetric,     // Default metric
}
```

### Insert Methods

| Method | Description |
| --- | --- |
| `insert(key, embedding)` | Dense embedding insert |
| `insert_sparse(key, embedding)` | Sparse embedding insert (memory efficient) |
| `insert_auto(key, embedding, threshold)` | Auto-select based on sparsity |

## Performance

### Benchmarks (10,000 entries, 128-dim embeddings)

| Operation | Time | Notes |
| --- | --- | --- |
| Exact lookup (hit) | ~50ns | Hash lookup + TensorStore get |
| Exact lookup (miss) | ~30ns | Hash lookup only |
| Semantic lookup | ~5us | HNSW search + re-scoring |
| Put (exact + semantic) | ~10us | Two stores + HNSW insert |
| Eviction (100 entries) | ~200us | Batch deletion |
| Clear (full index) | ~1ms | HNSW recreation |

### Distance Metric Performance (128-dim, 1000 entries)

| Metric | Search Time | Notes |
| --- | --- | --- |
| Cosine | 21 us | Default, best for dense |
| Jaccard | 18 us | Best for sparse |
| Angular | 23 us | +acos overhead |
| Euclidean | 19 us | Absolute distance |

### Auto-Selection Overhead

| Operation | Time |
| --- | --- |
| Sparsity check | ~50 ns |
| Metric selection | ~10 ns |

### Memory Efficiency

| Storage Type | Memory per Entry | Best For |
| --- | --- | --- |
| Dense Vector | 4 * dim bytes | Low sparsity (<50% zeros) |
| Sparse Vector | 8 * nnz bytes | High sparsity (>50% zeros) |

## Edge Cases and Gotchas

### TTL Behavior

- Entries with `expires_at = 0` never expire
- Expired entries return `None` on lookup but remain in storage until cleanup
- `cleanup_expired()` must be called explicitly or via background eviction

### Capacity Limits

- `put()` fails with `CacheFull` when capacity is reached
- Capacity is checked per-layer (exact, semantic, embedding)
- No automatic eviction on put -- must be explicit

### Hash Collisions

- Extremely unlikely with 64-bit hashes (~1 in 18 quintillion)
- If collision occurs, exact cache will return wrong response
- Semantic cache provides fallback for semantically different queries

### Metric Re-scoring

- HNSW always uses cosine similarity for graph navigation
- Re-scoring with different metrics may change result order
- Retrieves 3x candidates to account for re-ranking

### Sparse Storage Threshold

- Uses sparse format when `nnz * 2 <= len` (50% zeros)
- Different from auto-metric selection threshold (default 70%)
- Both thresholds are configurable

## Shell Commands

```text
CACHE INIT     Initialize semantic cache
CACHE STATS    Show cache statistics
CACHE CLEAR    Clear all cache entries
```

## Dependencies

| Crate | Purpose |
| --- | --- |
| `tensor_store` | HNSW index implementation, TensorStore |
| `tiktoken-rs` | GPT-compatible token counting |
| `dashmap` | Concurrent hash maps |
| `tokio` | Async runtime for background eviction |
| `uuid` | Unique ID generation |
| `thiserror` | Error type derivation |
| `serde` | Configuration serialization |
| `bincode` | Binary serialization |

## Related Modules

- `tensor_store` -- Backing storage and HNSW index
- `query_router` -- Cache integration for query execution
- `neumann_shell` -- CLI commands for cache management

## See Also

- [Semantic Caching Design](../../explanation/semantic-caching.md) -- multi-layer
  cache architecture, how semantic matching works, and eviction strategy rationale
- [How to: Configure Semantic Cache](../../how-to/configure-semantic-cache.md) --
  configure cache layers, set TTLs, tune similarity thresholds, and set up
  eviction policies
