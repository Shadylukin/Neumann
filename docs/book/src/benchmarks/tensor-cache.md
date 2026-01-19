# tensor_cache Benchmarks

The tensor_cache crate provides LLM response caching with exact, semantic (HNSW), and embedding caches.

## Exact Cache (Hash-based O(1))

| Operation | Time |
|-----------|------|
| lookup_hit | 208 ns |
| lookup_miss | 102 ns |

## Semantic Cache (HNSW-based O(log n))

| Operation | Time |
|-----------|------|
| lookup_hit | 21 us |

## Put (Exact + Semantic + HNSW insert)

| Entries | Time |
|---------|------|
| 100 | 49 us |
| 1,000 | 47 us |
| 10,000 | 53 us |

## Embedding Cache

| Operation | Time |
|-----------|------|
| lookup_hit | 230 ns |
| lookup_miss | 110 ns |

## Eviction (batch processing)

| Entries in Cache | Time |
|------------------|------|
| 1,000 | 3.3 us |
| 5,000 | 4.0 us |
| 10,000 | 8.4 us |

## Distance Metrics (raw computation, 128d)

| Metric | Time | Notes |
|--------|------|-------|
| Jaccard | 73 ns | Fastest, best for sparse |
| Euclidean | 105 ns | Good for spatial data |
| Cosine | 186 ns | Default, best for dense |
| Angular | 193 ns | Alternative to cosine |

## Semantic Lookup by Metric (1000 entries)

| Metric | Time |
|--------|------|
| Jaccard | 28.6 us |
| Euclidean | 27.8 us |
| Cosine | 28.4 us |

## Sparse vs Dense (80% sparsity)

| Configuration | Time | Improvement |
|---------------|------|-------------|
| Dense lookup | 28.8 us | baseline |
| Sparse lookup | 24.1 us | **16% faster** |

## Auto-Metric Selection

| Operation | Time |
|-----------|------|
| Sparsity check | 0.66 ns |
| Auto-select dense | 13.4 us |
| Auto-select sparse | 16.5 us |

## Redis Comparison

| System | In-Process | Over TCP |
|--------|------------|----------|
| Redis | ~60 ns | ~143 us |
| tensor_cache (exact) | 208 ns | ~143 us* |
| tensor_cache (semantic) | 21 us | N/A |

*Estimated: network latency dominates (99.9% of time).

**Key Insight**: For embedded use (no network), Redis is 3.5x faster for exact lookups. Over TCP (typical deployment), both are network-bound at ~143us. Our differentiator is **semantic search** (21us) which Redis cannot provide.

## Analysis

- **Exact cache**: Hash-based O(1) lookup provides sub-microsecond hit/miss detection
- **Semantic cache**: HNSW index provides O(log n) similarity search (~21us for hit)
- **Embedding cache**: Fast O(1) lookup for precomputed embeddings
- **Put performance**: Consistent ~50us regardless of cache size (HNSW insert is O(log n))
- **Eviction**: Efficient batch eviction with LRU/LFU/Cost/Hybrid strategies
- **Distance metrics**: Auto-selection based on sparsity (>=70% sparse uses Jaccard)
- **Token counting**: tiktoken cl100k_base encoding for accurate GPT-4 token counts
- **Cost tracking**: Estimates cost savings based on model pricing tables

## Cache Layers

| Layer | Complexity | Use Case |
|-------|------------|----------|
| Exact | O(1) | Identical prompts |
| Semantic | O(log n) | Similar prompts |
| Embedding | O(1) | Precomputed embeddings |

## Eviction Strategies

| Strategy | Description |
|----------|-------------|
| LRU | Evict least recently accessed |
| LFU | Evict least frequently accessed |
| CostBased | Evict lowest cost efficiency |
| Hybrid | Weighted combination (recommended) |

## Metric Selection Guide

| Embedding Type | Recommended Metric |
|----------------|-------------------|
| OpenAI/Cohere (dense) | Cosine (default) |
| Sparse (>=70% zeros) | Jaccard (auto-selected) |
| Spatial/geographic | Euclidean |
| Custom binary | Jaccard |
