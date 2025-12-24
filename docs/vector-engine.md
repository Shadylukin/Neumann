# Vector Engine

Module 4 of Neumann. Provides embeddings storage and similarity search.

## Design Principles

1. **Layered Architecture**: Depends only on Tensor Store for persistence
2. **Cosine Similarity**: Standard metric for embedding similarity
3. **Brute-Force Search**: O(n) k-NN for correctness (optimizations deferred)
4. **Thread Safety**: Inherits from Tensor Store

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
// Find top-k most similar embeddings to query
let query = vec![0.1, 0.2, 0.3];
let results = engine.search_similar(&query, 5)?;

for result in results {
    println!("Key: {}, Score: {}", result.key, result.score);
}
```

Results are sorted by similarity score (highest first). The score is cosine similarity, ranging from -1.0 (opposite) to 1.0 (identical).

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

## Future Considerations

Not implemented (out of scope for Module 4):

- **Approximate Nearest Neighbor (ANN)**: HNSW, IVF for faster search
- **Dimensionality Reduction**: PCA, random projection
- **Quantization**: Reduce memory with int8/binary vectors
- **Batched Operations**: Store/search multiple at once
- **Metadata Filtering**: Pre-filter by attributes before similarity search
