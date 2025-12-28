# Vector Engine

Module 4 of Neumann. Provides embeddings storage and similarity search.

## Design Principles

1. **Layered Architecture**: Depends only on Tensor Store for persistence
2. **Multiple Distance Metrics**: Cosine, Euclidean, and Dot Product similarity
3. **SIMD Acceleration**: 8-wide SIMD for dot products and magnitudes
4. **Dual Search Modes**: Brute-force O(n) or HNSW O(log n)
5. **Unified Entities**: Embeddings can be attached to shared entities
6. **Thread Safety**: Inherits from Tensor Store
7. **Serializable Types**: All types implement `serde::Serialize`/`Deserialize`

## API Reference

### Basic Operations

```rust
let engine = VectorEngine::new();

// Store an embedding
engine.store_embedding("doc1", vec![0.1, 0.2, 0.3])?;

// Get an embedding
let vector = engine.get_embedding("doc1")?;

// Delete an embedding
engine.delete_embedding("doc1")?;

// Check existence
engine.exists("doc1");  // -> bool

// Count embeddings
engine.count();  // -> usize
```

### Similarity Search

```rust
// Find top-k most similar embeddings (default: cosine similarity)
let query = vec![0.1, 0.2, 0.3];
let results = engine.search_similar(&query, 5)?;

for result in results {
    println!("Key: {}, Score: {}", result.key, result.score);
}
```

Results are sorted by similarity score (highest first).

### Distance Metrics

```rust
use vector_engine::DistanceMetric;

// Cosine similarity (default) - range: -1.0 to 1.0
let results = engine.search_similar_with_metric(&query, 5, DistanceMetric::Cosine)?;

// Euclidean distance - score: 1/(1+distance), range: 0.0 to 1.0
let results = engine.search_similar_with_metric(&query, 5, DistanceMetric::Euclidean)?;

// Dot product - raw dot product value
let results = engine.search_similar_with_metric(&query, 5, DistanceMetric::DotProduct)?;
```

| Metric | Formula | Score Range | Best For |
|--------|---------|-------------|----------|
| Cosine | `a . b / (\|a\| * \|b\|)` | -1.0 to 1.0 | Semantic similarity |
| Euclidean | `1 / (1 + sqrt(sum((a-b)^2)))` | 0.0 to 1.0 | Spatial distance |
| DotProduct | `sum(a * b)` | unbounded | Magnitude-aware |

**Note**: Euclidean scores are transformed so higher is always better (consistent with other metrics).

### Cosine Similarity

```rust
// Direct similarity computation
let a = vec![1.0, 0.0];
let b = vec![0.707, 0.707];
let score = VectorEngine::compute_similarity(&a, &b)?;
// score = 0.707 (45 degree angle)
```

The cosine similarity formula:
```
cos(a, b) = (a . b) / (|a| * |b|)
```

Where:
- `a . b` is the dot product
- `|a|` and `|b|` are vector magnitudes (L2 norms)

### Utility Methods

```rust
// Get dimension of stored embeddings (from first found)
engine.dimension();  // -> Option<usize>

// List all embedding keys
engine.list_keys();  // -> Vec<String>

// Clear all embeddings
engine.clear()?;
```

### Unified Entity API

Attach embeddings directly to entities for cross-engine queries:

```rust
// Create engine with shared store
let store = TensorStore::new();
let engine = VectorEngine::with_store(store.clone());

// Set embedding on an existing entity (e.g., user:1)
engine.set_entity_embedding("user:1", vec![0.1, 0.2, 0.3])?;

// Get embedding from an entity
let embedding = engine.get_entity_embedding("user:1")?;

// Search for entities with embeddings similar to another entity's neighbors
// (Used by QueryRouter for cross-engine queries)
engine.search_entity_neighbors_by_similarity(neighbor_keys, &query_vec, 5)?;
```

Unified entity embeddings are stored in the `_embedding` field of the entity's TensorData, enabling the same entity key to have relational fields, graph connections, and a vector embedding.

## Error Handling

| Error | Cause |
|-------|-------|
| `NotFound` | Embedding key doesn't exist |
| `DimensionMismatch` | Vectors have different dimensions |
| `EmptyVector` | Empty vector provided |
| `InvalidTopK` | top_k is 0 |
| `StorageError` | Underlying Tensor Store error |

## Storage Model

Embeddings are stored in Tensor Store:

| Key Pattern | Content |
|-------------|---------|
| `emb:{key}` | TensorData with "vector" field containing the embedding |

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `store_embedding` | O(1) | Single store put |
| `get_embedding` | O(1) | Single store get |
| `delete_embedding` | O(1) | Single store delete |
| `search_similar` | O(n*d) | Brute-force scan, n=count, d=dimension |
| `count` | O(n) | Scans all embeddings |
| `list_keys` | O(n) | Scans all embeddings |

## Supported Dimensions

The engine supports any vector dimension. Common embedding sizes:

| Model | Dimensions |
|-------|------------|
| OpenAI text-embedding-ada-002 | 1536 |
| OpenAI text-embedding-3-small | 1536 |
| OpenAI text-embedding-3-large | 3072 |
| BERT base | 768 |
| Sentence Transformers | 384-768 |

## Test Coverage

| Test | What It Verifies |
|------|------------------|
| `store_and_retrieve_embedding` | Basic store/get |
| `store_10000_vectors_search` | Scale + correct nearest neighbor |
| `cosine_similarity_*` | Mathematical correctness |
| `high_dimensional_768` | BERT-size vectors |
| `high_dimensional_1536` | OpenAI-size vectors |
| `search_similar_top_k` | Exactly k results returned |
| `search_similar_fewer_than_k` | Handles fewer than k exist |
| `similarity_scores_mathematically_correct` | Verifies cos(0)=1, cos(90)=0, cos(180)=-1 |

## HNSW Index

For large datasets, build an HNSW index for O(log n) search:

```rust
// Build HNSW index (via QueryRouter)
router.build_vector_index()?;

// Searches now use HNSW automatically
let results = router.execute_parsed("SIMILAR 'doc:1' LIMIT 10")?;
```

| Dataset Size | Brute-Force | With HNSW | Speedup |
|--------------|-------------|-----------|---------|
| 200 vectors | 4.17s | 9.3us | 448,000x |

**Note**: HNSW currently only supports cosine similarity. Other metrics fall back to brute-force.

## Known Limitations

### Zero-Magnitude Vectors
- **Cosine/DotProduct**: Return empty results (undefined for zero vectors)
- **Euclidean**: Correctly handles zero vectors (finds vectors closest to origin)

## Future Considerations

Not implemented (out of scope):

- **Dimensionality Reduction**: PCA, random projection
- **Product Quantization**: Further memory reduction beyond int8
- **Metadata Filtering**: Pre-filter by attributes before similarity search
- **HNSW with non-cosine metrics**: Currently falls back to brute-force
