# Vector Engine API Reference

> **See Also:**
> [Vector Search Modes](../../explanation/vector-search-modes.md) (architecture and design rationale) |
> [Filtered Search Design](../../explanation/filtered-search-design.md) (filter strategies) |
> [Embeddings Search How-To](../../how-to/embeddings-search.md) |
> [Filtered Search How-To](../../how-to/filtered-search.md) |
> [Vector Collections How-To](../../how-to/vector-collections.md)

## Key Types

| Type | Description |
| --- | --- |
| `VectorEngine` | Main engine for storing and searching embeddings |
| `VectorEngineConfig` | Configuration for engine behavior and memory bounds |
| `SearchResult` | Result with key and similarity score |
| `DistanceMetric` | Enum: `Cosine`, `Euclidean`, `DotProduct` |
| `ExtendedDistanceMetric` | Extended metrics for HNSW (9+ variants) |
| `VectorError` | Error types for vector operations |
| `EmbeddingInput` | Input for batch store operations |
| `BatchResult` | Result of batch operations |
| `Pagination` | Parameters for paginated queries |
| `PagedResult<T>` | Paginated query result |
| `HNSWIndex` | Hierarchical navigable small world graph (re-exported from tensor_store) |
| `HNSWConfig` | HNSW index configuration (re-exported from tensor_store) |
| `SparseVector` | Memory-efficient sparse embedding storage |
| `FilterCondition` | Filter for metadata-based search (Eq, Ne, Lt, Gt, And, Or, In, etc.) |
| `FilterValue` | Value type for filters (Int, Float, String, Bool, Null) |
| `FilterStrategy` | Strategy selection (Auto, PreFilter, PostFilter) |
| `FilteredSearchConfig` | Configuration for filtered search behavior |
| `VectorCollectionConfig` | Configuration for vector collections |
| `MetadataValue` | Simplified value type for embedding metadata |
| `PersistentVectorIndex` | Serializable index for disk persistence |

## SearchResult Fields

| Field | Type | Description |
| --- | --- | --- |
| `key` | `String` | The key of the matched embedding |
| `score` | `f32` | Similarity score (higher = more similar) |

## VectorError Variants

| Variant | Description | When Triggered |
| --- | --- | --- |
| `NotFound` | Embedding key doesn't exist | `get_embedding`, `delete_embedding` |
| `DimensionMismatch` | Vectors have different dimensions | `compute_similarity`, exceeds `max_dimension` |
| `EmptyVector` | Empty vector provided | Any operation with `vec![]` |
| `InvalidTopK` | top_k is 0 | `search_similar`, `search_with_hnsw` |
| `StorageError` | Underlying Tensor Store error | Storage failures |
| `BatchValidationError` | Invalid input in batch | `batch_store_embeddings` validation |
| `BatchOperationError` | Operation failed in batch | `batch_store_embeddings` execution |
| `ConfigurationError` | Invalid configuration | `VectorEngineConfig::validate()` |
| `CollectionExists` | Collection already exists | `create_collection` with existing name |
| `CollectionNotFound` | Collection not found | Collection operations on missing collection |
| `IoError` | IO error during persistence | `save_to_file`, `load_from_file` |
| `SerializationError` | Serialization error | Index persistence operations |
| `SearchTimeout` | Search operation timed out | Search operations exceeding configured timeout |

## DistanceMetric Variants

| Metric | Formula | Score Range | Use Case | HNSW Support |
| --- | --- | --- | --- | --- |
| Cosine | `a.b / (||a|| * ||b||)` | -1.0 to 1.0 | Semantic similarity | Yes |
| Euclidean | `1 / (1 + sqrt(sum((a-b)^2)))` | 0.0 to 1.0 | Spatial distance | No (brute-force) |
| DotProduct | `sum(a * b)` | unbounded | Magnitude-aware | No (brute-force) |

All metrics return higher scores for better matches. Euclidean distance is
transformed to a similarity score.

### Euclidean Distance-to-Similarity Transformation

| Distance | Similarity Score |
| --- | --- |
| 0.0 | 1.0 (identical) |
| 1.0 | 0.5 |
| 2.0 | 0.333 |
| 9.0 | 0.1 |
| Infinity | 0.0 |

### ExtendedDistanceMetric Variants (HNSW)

Additional metrics available via `search_with_hnsw_and_metric()`:

| Metric | Description | Best For |
| --- | --- | --- |
| `Cosine` | Angle-based similarity | Text embeddings, normalized vectors |
| `Euclidean` | L2 distance | Spatial data, absolute distances |
| `Angular` | Cosine converted to angular | When angle interpretation needed |
| `Manhattan` | L1 norm | Robust to outliers |
| `Chebyshev` | L-infinity (max diff) | When max deviation matters |
| `Jaccard` | Set similarity | Binary/sparse vectors, TF-IDF |
| `Overlap` | Minimum overlap coefficient | Partial matches |
| `Geodesic` | Spherical distance | Geographic coordinates |
| `Composite` | Weighted combination | Custom similarity functions |

## VectorEngineConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `default_dimension` | `Option<usize>` | `None` | Expected embedding dimension |
| `sparse_threshold` | `f32` | `0.5` | Sparsity threshold (0.0-1.0) |
| `parallel_threshold` | `usize` | `5000` | Dataset size for parallel search |
| `default_metric` | `DistanceMetric` | `Cosine` | Default distance metric |
| `max_dimension` | `Option<usize>` | `None` | Maximum allowed dimension |
| `max_keys_per_scan` | `Option<usize>` | `None` | Limit for unbounded scans |
| `batch_parallel_threshold` | `usize` | `100` | Batch size for parallel processing |
| `search_timeout` | `Option<Duration>` | `None` | Search operation timeout |

### Configuration Presets

| Preset | Description | Key Settings |
| --- | --- | --- |
| `default()` | Balanced for most workloads | All defaults |
| `high_throughput()` | Optimized for write-heavy loads | `parallel_threshold: 1000` |
| `low_memory()` | Memory-constrained environments | `max_dimension: 4096`, `max_keys_per_scan: 10000`, `search_timeout: 30s` |

### Builder Methods

All builder methods are `const fn` for compile-time configuration:

```rust
use std::time::Duration;

let config = VectorEngineConfig::default()
    .with_default_dimension(768)
    .with_sparse_threshold(0.7)
    .with_parallel_threshold(1000)
    .with_default_metric(DistanceMetric::Cosine)
    .with_max_dimension(4096)
    .with_max_keys_per_scan(50_000)
    .with_batch_parallel_threshold(200)
    .with_search_timeout(Duration::from_secs(5));

let engine = VectorEngine::with_config(config)?;
```

## VectorCollectionConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `dimension` | `Option<usize>` | `None` | Enforced dimension (rejects mismatches) |
| `distance_metric` | `DistanceMetric` | `Cosine` | Default metric for this collection |
| `auto_index` | `bool` | `false` | Auto-build HNSW on threshold |
| `auto_index_threshold` | `usize` | `1000` | Vector count to trigger auto-index |

## FilterCondition Variants

| Condition | Description | Example |
| --- | --- | --- |
| `Eq(field, value)` | Equality | `category = "science"` |
| `Ne(field, value)` | Not equal | `status != "deleted"` |
| `Lt(field, value)` | Less than | `price < 100` |
| `Le(field, value)` | Less than or equal | `price <= 100` |
| `Gt(field, value)` | Greater than | `year > 2020` |
| `Ge(field, value)` | Greater than or equal | `year >= 2020` |
| `And(a, b)` | Logical AND | Combined conditions |
| `Or(a, b)` | Logical OR | Alternative conditions |
| `In(field, values)` | Value in list | `status IN ["active", "pending"]` |
| `Contains(field, substr)` | String contains | `title CONTAINS "rust"` |
| `StartsWith(field, prefix)` | String prefix | `name STARTS WITH "doc:"` |
| `Exists(field)` | Field exists | `HAS embedding` |
| `True` | Always matches | No filter |

## FilterStrategy

| Strategy | When to Use | Behavior |
| --- | --- | --- |
| `Auto` | Default | Estimates selectivity and chooses |
| `PreFilter` | < 10% matches | Filters first, then searches subset |
| `PostFilter` | > 10% matches | Searches with oversample, then filters |

## HNSW Configuration Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `m` | 16 | Max connections per node per layer |
| `m0` | 32 | Max connections at layer 0 (2*m) |
| `ef_construction` | 200 | Candidates during index building |
| `ef_search` | 50 | Candidates during search |
| `ml` | 1/ln(m) | Level multiplier for layer selection |
| `sparsity_threshold` | 0.5 | Auto-sparse threshold |
| `max_nodes` | 10,000,000 | Capacity limit |

### HNSW Presets

| Preset | m | m0 | ef_construction | ef_search | Use Case |
| --- | --- | --- | --- | --- | --- |
| `default()` | 16 | 32 | 200 | 50 | Balanced |
| `high_recall()` | 32 | 64 | 400 | 200 | Accuracy over speed |
| `high_speed()` | 8 | 16 | 100 | 20 | Speed over accuracy |

### Memory vs Recall Tradeoff

| Config | Memory/Node | Recall@10 | Search Time |
| --- | --- | --- | --- |
| high_speed | ~128 bytes | ~85% | 0.1ms |
| default | ~256 bytes | ~95% | 0.3ms |
| high_recall | ~512 bytes | ~99% | 1.0ms |

## PersistentVectorIndex Format

| Field | Type | Description |
| --- | --- | --- |
| `collection` | `String` | Collection name |
| `config` | `VectorCollectionConfig` | Collection configuration |
| `vectors` | `Vec<VectorEntry>` | All vectors with metadata |
| `created_at` | `u64` | Unix timestamp |
| `version` | `u32` | Format version (currently 1) |

## Storage Key Patterns

| Key Pattern | Content | Use Case |
| --- | --- | --- |
| `emb:{key}` | TensorData with "vector" field | Default collection embeddings |
| `coll:{collection}:emb:{key}` | TensorData with "vector" field | Named collection embeddings |
| `{entity_key}` | TensorData with "_embedding" field | Unified entities |

## Collection Management Methods

| Method | Description |
| --- | --- |
| `create_collection(name, config)` | Create a new named collection |
| `list_collections()` | List all collection names |
| `collection_exists(name)` | Check if collection exists |
| `get_collection_config(name)` | Retrieve collection configuration |
| `delete_collection(name)` | Delete collection and all its vectors |
| `store_in_collection(name, key, vec)` | Store embedding in collection |
| `store_in_collection_with_metadata(name, key, vec, meta)` | Store with metadata |
| `search_in_collection(name, query, k)` | Similarity search within collection |
| `search_filtered_in_collection(name, query, k, filter, config)` | Filtered search within collection |
| `save_all_indices(path)` | Save all collection indices to directory |
| `load_all_indices(path)` | Load all collection indices from directory |
| `save_index(name, path)` | Save single collection index (JSON) |
| `save_index_binary(name, path)` | Save single collection index (binary) |
| `load_index(path)` | Load single collection index (JSON) |
| `load_index_binary(path)` | Load single collection index (binary) |
| `snapshot_collection(name)` | Get serializable snapshot of collection |

## Performance Characteristics

| Operation | Complexity | Notes |
| --- | --- | --- |
| `store_embedding` | O(1) | Single store put |
| `get_embedding` | O(1) | Single store get |
| `delete_embedding` | O(1) | Single store delete |
| `search_similar` | O(n*d) | Brute-force, n=count, d=dimension |
| `search_with_hnsw` | O(log n * ef * m) | Approximate nearest neighbor |
| `build_hnsw_index` | O(n * log n * ef_construction * m) | Index construction |
| `count` | O(n) | Scans all embeddings |
| `list_keys` | O(n) | Scans all embeddings |

### Benchmark Results

| Dataset Size | Brute-Force | With HNSW | Speedup |
| --- | --- | --- | --- |
| 200 vectors | 4.17s | 9.3us | 448,000x |
| 1,000 vectors | ~5ms | ~20us | 250x |
| 10,000 vectors | ~50ms | ~50us | 1000x |
| 100,000 vectors | ~500ms | ~100us | 5000x |

## Supported Embedding Dimensions

| Model | Dimensions | Recommended Config |
| --- | --- | --- |
| OpenAI text-embedding-ada-002 | 1536 | default |
| OpenAI text-embedding-3-small | 1536 | default |
| OpenAI text-embedding-3-large | 3072 | high_recall |
| BERT base | 768 | default |
| Sentence Transformers | 384-768 | default |
| Cohere embed-v3 | 1024 | default |
| Custom/small | <256 | high_speed |

## Sparse Vector Distance Metrics

| Metric | Complexity | Description |
| --- | --- | --- |
| `dot` | O(min(nnz_a, nnz_b)) | Sparse-sparse dot product |
| `dot_dense` | O(nnz) | Sparse-dense dot product |
| `cosine_similarity` | O(min(nnz_a, nnz_b)) | Angle-based similarity |
| `euclidean_distance` | O(nnz_a + nnz_b) | L2 distance |
| `manhattan_distance` | O(nnz_a + nnz_b) | L1 distance |
| `jaccard_index` | O(min(nnz_a, nnz_b)) | Position overlap |
| `angular_distance` | O(min(nnz_a, nnz_b)) | Arc-cosine |

## SIMD Performance Characteristics

| Dimension | SIMD Speedup | Notes |
| --- | --- | --- |
| 8 | 1x | Baseline (single SIMD operation) |
| 64 | 4-6x | Full pipeline utilization |
| 384 | 6-8x | Sentence Transformers size |
| 768 | 6-8x | BERT embedding size |
| 1536 | 6-8x | OpenAI ada-002 size |
| 3072 | 6-8x | OpenAI text-embedding-3-large |

## Edge Cases

### Zero-Magnitude Vectors

| Metric | Behavior | Rationale |
| --- | --- | --- |
| Cosine | Returns empty results | Division by zero undefined |
| DotProduct | Returns empty results | Undefined direction |
| Euclidean | Works correctly | Finds vectors closest to origin |

### HNSW Limitations

| Limitation | Details | Workaround |
| --- | --- | --- |
| Only cosine similarity | HNSW uses cosine distance internally | Use brute-force for other metrics |
| No deletion | Cannot remove vectors | Rebuild index |
| Static after build | Index doesn't update with new vectors | Rebuild periodically |
| Memory overhead | Graph structure adds ~2-4x | Use for large datasets only |

### NaN/Infinity Handling

Sparse vector operations sanitize NaN/Inf results:

- `cosine_similarity` returns `0.0` for NaN/Inf, otherwise clamps to [-1.0, 1.0]
- `cosine_distance_dense` returns `1.0` (max distance) for NaN/Inf

### Dimension Mismatch Handling

Mismatched dimensions are silently skipped during search. A search with a 2D
query vector will only match 2D embeddings, even if 3D embeddings exist in the
same store.

## Dependencies

| Crate | Purpose |
| --- | --- |
| `tensor_store` | Persistence, SparseVector, HNSWIndex, SIMD |
| `rayon` | Parallel iteration for large datasets |
| `serde` | Serialization of types |
| `tracing` | Instrumentation and observability |

Note: `wide` (SIMD f32x8 operations) is a transitive dependency via `tensor_store`.

## Related Modules

- [Tensor Store](./tensor-store.md) - Underlying storage and HNSW implementation
- [Query Router](./query-router.md) - Executes SIMILAR queries using VectorEngine
- [Tensor Cache](./tensor-cache.md) - Uses vector similarity for semantic caching
