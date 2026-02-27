# Store Embeddings and Run Similarity Searches

Step-by-step guide for storing vector embeddings and running k-NN similarity
searches with the Vector Engine.

> **See Also:**
> [Vector Engine API Reference](../reference/api/vector-engine.md) |
> [Vector Search Modes](../explanation/vector-search-modes.md) |
> [Filtered Search How-To](filtered-search.md) |
> [Vector Collections How-To](vector-collections.md)

## Prerequisites

Add `vector_engine` and `tensor_store` to your `Cargo.toml`:

```toml
[dependencies]
vector_engine = { path = "../vector_engine" }
tensor_store = { path = "../tensor_store" }
```

## Store Embeddings

### Basic Storage

```rust
let engine = VectorEngine::new();

// Store a single embedding
engine.store_embedding("doc1", vec![0.1, 0.2, 0.3])?;

// Check existence
assert!(engine.exists("doc1"));

// Retrieve the embedding
let vector = engine.get_embedding("doc1")?;

// Get the dimension of stored embeddings
let dim = engine.dimension();  // -> Some(3)
```

### Store with Metadata

Attach arbitrary key-value metadata to embeddings for later filtering:

```rust
use tensor_store::TensorValue;
use std::collections::HashMap;

let mut metadata = HashMap::new();
metadata.insert("category".to_string(), TensorValue::from("science"));
metadata.insert("year".to_string(), TensorValue::from(2024i64));
metadata.insert("score".to_string(), TensorValue::from(0.95f64));

engine.store_embedding_with_metadata("doc1", vec![0.1, 0.2, 0.3], metadata)?;
```

### Manage Metadata After Storage

```rust
// Get all metadata for an embedding
let meta = engine.get_metadata("doc1")?;

// Get a specific field
let category = engine.get_metadata_field("doc1", "category")?;

// Update metadata (merges with existing)
let mut updates = HashMap::new();
updates.insert("score".to_string(), TensorValue::from(0.98f64));
engine.update_metadata("doc1", updates)?;

// Check if a metadata field exists
if engine.has_metadata_field("doc1", "category") {
    // Remove a specific metadata field
    engine.remove_metadata_field("doc1", "category")?;
}
```

### Batch Storage

For bulk inserts, use batch operations. Batches larger than
`batch_parallel_threshold` (default 100) are processed in parallel:

```rust
use vector_engine::EmbeddingInput;

let inputs = vec![
    EmbeddingInput::new("doc1", vec![0.1, 0.2, 0.3]),
    EmbeddingInput::new("doc2", vec![0.2, 0.3, 0.4]),
    EmbeddingInput::new("doc3", vec![0.3, 0.4, 0.5]),
];

let result = engine.batch_store_embeddings(inputs)?;
println!("Stored {} embeddings", result.count);  // -> 3
```

### Delete Embeddings

```rust
// Delete a single embedding
engine.delete_embedding("doc1")?;

// Batch delete
let keys = vec!["doc1".to_string(), "doc2".to_string()];
let deleted = engine.batch_delete_embeddings(keys)?;
println!("Deleted {} embeddings", deleted);  // -> 2

// Clear all embeddings
engine.clear()?;
```

## Run Similarity Searches (k-NN)

### Basic Search (Cosine Similarity)

```rust
let query = vec![0.1, 0.2, 0.3];
let results = engine.search_similar(&query, 5)?;

for result in results {
    println!("Key: {}, Score: {}", result.key, result.score);
}
```

### Search with a Specific Distance Metric

```rust
use vector_engine::DistanceMetric;

// Euclidean similarity (transformed to 0.0-1.0 range)
let results = engine.search_similar_with_metric(
    &query,
    5,
    DistanceMetric::Euclidean,
)?;

// Dot product (unbounded score range)
let results = engine.search_similar_with_metric(
    &query,
    5,
    DistanceMetric::DotProduct,
)?;
```

### Direct Similarity Computation

Compute similarity between two specific vectors without searching:

```rust
let similarity = VectorEngine::compute_similarity(&vec_a, &vec_b)?;
```

### HNSW Index Search (Large Datasets)

For datasets with more than ~10,000 vectors, build an HNSW index for
logarithmic-time approximate search:

```rust
// Build index with default configuration
let (index, key_mapping) = engine.build_hnsw_index_default()?;

// Search using the index
let results = engine.search_with_hnsw(&index, &key_mapping, &query, 10)?;

// Build with a tuned configuration
let config = HNSWConfig::high_recall();
let (index, key_mapping) = engine.build_hnsw_index(config)?;
```

### HNSW with Extended Distance Metrics

```rust
use vector_engine::ExtendedDistanceMetric;

let (index, keys) = engine.build_hnsw_index_default()?;

// Search with Jaccard similarity (useful for sparse/binary vectors)
let results = engine.search_with_hnsw_and_metric(
    &index,
    &keys,
    &query,
    10,
    ExtendedDistanceMetric::Jaccard,
)?;
```

### Paginated Search

For memory-efficient iteration over large result sets:

```rust
use vector_engine::Pagination;

// Skip the first 10 results, return the next 5
let page = Pagination::new(10, 5);
let results = engine.search_similar_paginated(&query, 100, page)?;
println!("Items: {}, Has more: {}", results.items.len(), results.has_more);
```

### Paginated Key Listing

```rust
let page = Pagination::new(0, 100).with_total();
let result = engine.list_keys_paginated(page);
println!("Total: {:?}", result.total_count);  // Some(total)
```

Use `list_keys_bounded()` in production to enforce `max_keys_per_scan` limits.

## Unified Entity Mode

Attach embeddings directly to shared entities for cross-engine queries
(e.g., combining graph traversal with similarity search):

```rust
let store = TensorStore::new();
let engine = VectorEngine::with_store(store.clone());

// Set embedding on an entity
engine.set_entity_embedding("user:1", vec![0.1, 0.2, 0.3])?;

// Get embedding from an entity
let embedding = engine.get_entity_embedding("user:1")?;

// Check if entity has embedding
engine.entity_has_embedding("user:1");  // -> bool

// Remove embedding (preserves other entity data)
engine.remove_entity_embedding("user:1")?;

// Search across entities with embeddings
let results = engine.search_entities(&query, 5)?;

// Scan all entities with embeddings
let entity_keys = engine.scan_entities_with_embeddings();
let count = engine.count_entities_with_embeddings();
```

## Configure Memory Bounds

For production deployments, set bounds to prevent resource exhaustion:

```rust
use std::time::Duration;

let config = VectorEngineConfig::default()
    .with_max_dimension(4096)         // Reject oversized embeddings
    .with_max_keys_per_scan(10_000)   // Limit unbounded scans
    .with_search_timeout(Duration::from_secs(5));  // Prevent runaway queries

let engine = VectorEngine::with_config(config)?;
```

### Handle Search Timeouts

```rust
use vector_engine::{VectorEngine, VectorEngineConfig, VectorError};

match engine.search_similar(&query, 10) {
    Ok(results) => { /* process results */ },
    Err(VectorError::SearchTimeout { operation, timeout_ms }) => {
        eprintln!("Search '{}' timed out after {}ms", operation, timeout_ms);
    },
    Err(e) => { /* handle other errors */ },
}
```

## Best Practices

### Memory Optimization

1. **Sparse vectors are automatic**: Vectors with >50% zeros are stored
   efficiently without any action on your part.
2. **Batch insert before building HNSW**: Build the index once after all data is
   loaded rather than rebuilding incrementally.
3. **Choose appropriate HNSW config**: Do not over-provision `m` and `ef` -- see
   the [HNSW presets](../reference/api/vector-engine.md#hnsw-presets).
4. **Monitor memory**: Use `HNSWMemoryStats` to track dense vs sparse counts.

```rust
let stats = index.memory_stats();
println!("Dense: {}, Sparse: {}, Total bytes: {}",
    stats.dense_count, stats.sparse_count, stats.embedding_bytes);
```

### Search Performance

1. **Use HNSW for >10K vectors**: Brute-force for smaller sets.
2. **Tune `ef_search`**: Higher for recall, lower for speed.
3. **Parallel threshold**: Automatic at 5000 vectors (configurable).
4. **Entity key conventions**: Use prefixes like `user:`, `doc:`, `item:` for
   unified entity embeddings.

## Related Pages

- [Filtered Search How-To](filtered-search.md) -- add metadata filters to searches
- [Vector Collections How-To](vector-collections.md) -- organize embeddings into namespaces
- [Vector Search Modes](../explanation/vector-search-modes.md) -- understand brute-force vs HNSW tradeoffs
- [Vector Engine API Reference](../reference/api/vector-engine.md) -- complete type and config tables
