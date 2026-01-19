# vector_engine Benchmarks

The vector engine stores embeddings and performs k-nearest neighbor search using cosine similarity.

## Store Embedding

| Dimension | Time | Throughput |
|-----------|------|------------|
| 128 | 366 ns | 2.7M/s |
| 768 | 892 ns | 1.1M/s |
| 1536 | 969 ns | 1.0M/s |

## Get Embedding

| Dimension | Time |
|-----------|------|
| 768 | 287 ns |

## Delete Embedding

| Operation | Time |
|-----------|------|
| delete | 806 ns |

## Similarity Search (top 10, SIMD + adaptive parallel)

| Dataset | Time | Per Vector | Mode |
|---------|------|------------|------|
| 1,000 x 128d | 242 us | 242 ns | Sequential |
| 1,000 x 768d | 367 us | 367 ns | Sequential |
| 10,000 x 128d | 1.93 ms | 193 ns | Parallel |

## Cosine Similarity Computation (SIMD-accelerated)

| Dimension | Time |
|-----------|------|
| 128 | 26 ns |
| 768 | 165 ns |
| 1536 | 369 ns |

## Analysis

- **SIMD acceleration**: 8-wide f32 SIMD (via `wide` crate) provides 3-9x speedup for cosine similarity
- **Adaptive parallelism**: Uses rayon for parallel search when >5000 vectors (1.6x speedup at 10K)
- **Linear scaling with dimension**: Cosine similarity is O(d) where d is vector dimension
- **Linear scaling with dataset size**: Brute-force search is O(n*d) for n vectors
- **Memory bound**: For 768d vectors, ~3 KB per embedding (768 * 4 bytes)
- **Search throughput**: ~4M vector comparisons/second at 128d (with SIMD)
- **Store/Get performance**: Sub-microsecond for typical embedding sizes

## Complexity

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| store_embedding | O(d) | Vector copy + hash insert |
| get_embedding | O(d) | Hash lookup + vector clone |
| delete_embedding | O(1) | Hash removal |
| search_similar | O(n*d) | Brute-force scan |
| compute_similarity | O(d) | Dot product + 2 magnitude calculations |

## HNSW Index (Approximate Nearest Neighbor)

HNSW provides O(log n) search complexity instead of O(n) brute force.

| Configuration | Search Time (5K, 128d) |
|---------------|------------------------|
| high_speed | ~50 us |
| default | ~100 us |
| high_recall | ~200 us |

### HNSW vs Brute Force (10K vectors, 128d)

| Method | Search Time | Speedup |
|--------|-------------|---------|
| Brute force | ~2 ms | 1x |
| HNSW default | ~150 us | ~13x |

### Recommended Approach by Corpus Size

| Corpus Size | Approach | Rationale |
|-------------|----------|-----------|
| < 10K | Brute force | Fast enough, pure tensor |
| 10K - 100K | HNSW | Pragmatic, 5-13x faster |
| > 100K | HNSW | Necessary for latency |

### Scaling Projections (HNSW for >10K vectors)

| Vectors | Dimension | Search Time (est.) |
|---------|-----------|-------------------|
| 10K | 768 | ~200 us |
| 100K | 768 | ~500 us |
| 1M | 768 | ~1 ms |

For production workloads at extreme scale (>1M vectors), consider:
- Sharded HNSW across multiple nodes
- Dimensionality reduction (PCA)
- Quantization (int8, binary)

## Storage Model

vector_engine stores each embedding as a tensor:

```
emb:{key} -> TensorData { vector: [...] }
```

### Trade-offs

- **Pro**: Simple storage model, consistent with tensor abstraction
- **Pro**: Sub-microsecond store/get operations
- **Pro**: HNSW index for O(log n) approximate nearest neighbor search
- **Con**: Brute-force O(n*d) for exact search (use HNSW for approximate)
