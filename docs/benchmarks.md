# Neumann Benchmarks

This document describes the benchmark suite for Neumann's core modules and provides analysis of the performance characteristics.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run benchmarks for a specific crate
cargo bench --package tensor_store
cargo bench --package relational_engine
cargo bench --package graph_engine
cargo bench --package vector_engine
cargo bench --package neumann_parser
cargo bench --package query_router
cargo bench --package neumann_shell
cargo bench --package tensor_compress
cargo bench --package tensor_vault
cargo bench --package tensor_cache
```

Benchmark reports are generated in `target/criterion/` with HTML visualizations.

## Benchmark Results Summary

### tensor_store

The tensor store uses DashMap (sharded concurrent HashMap) for thread-safe key-value storage.

| Operation | 100 items | 1,000 items | 10,000 items |
|-----------|-----------|-------------|--------------|
| **put** | 40µs (2.5M/s) | 447µs (2.2M/s) | 7ms (1.4M/s) |
| **get** | 33µs (3.0M/s) | 320µs (3.1M/s) | 3ms (3.3M/s) |

**Scan Operations (10k total items, parallel):**
| Operation | Time |
|-----------|------|
| scan 1k keys | 191µs |
| scan_count 1k keys | 41µs |

**Concurrent Write Performance:**
| Threads | Disjoint Keys | High Contention (100 keys) |
|---------|---------------|----------------------------|
| 2 | 795µs | 974µs |
| 4 | 1.59ms | 1.48ms |
| 8 | 4.6ms | 2.33ms |

**Mixed Workload:**
| Configuration | Time |
|---------------|------|
| 4 readers + 2 writers | 579µs |

#### Analysis

- **Read vs Write**: Reads are ~20% faster than writes due to DashMap's read-optimized design
- **Scaling**: Near-linear scaling up to 10k items; slight degradation at scale due to hash table growth
- **Concurrency**: DashMap's 16-shard design provides excellent concurrent performance
- **Contention**: Under high contention, performance actually improves at 8 threads vs 4 (lock sharding distributes load)
- **Parallel scans**: Uses rayon for >1000 keys (25-53% faster)
- **scan_count vs scan**: Count-only is ~5x faster (avoids string cloning)

**Bloom Filter (optional):**
| Operation | Time |
|-----------|------|
| add | 68 ns |
| might_contain (hit) | 46 ns |
| might_contain (miss) | 63 ns |

**Sparse Lookups (1K keys in store):**
| Query Type | Without Bloom | With Bloom |
|------------|---------------|------------|
| Negative lookup | 52 ns | 68 ns |
| Positive lookup | 45 ns | 60 ns |
| Sparse workload (90% miss) | 52 ns | 67 ns |

Note: Bloom filter adds ~15ns overhead for in-memory DashMap stores. It's designed for scenarios where the backing store is slower (disk, network, remote database), where the early rejection of non-existent keys avoids expensive I/O.

**Snapshot Persistence (bincode):**

| Operation | 100 items | 1,000 items | 10,000 items |
|-----------|-----------|-------------|--------------|
| **save** | 100 µs (1.0M/s) | 927 µs (1.08M/s) | 12.6 ms (791K/s) |
| **load** | 74 µs (1.35M/s) | 826 µs (1.21M/s) | 10.7 ms (936K/s) |
| **load_with_bloom** | 81 µs (1.23M/s) | 840 µs (1.19M/s) | 11.0 ms (908K/s) |

Each item is a TensorData with 3 fields: id (i64), name (String), embedding (128-dim Vec<f32>).

**Snapshot File Sizes:**
| Items | File Size | Per Item |
|-------|-----------|----------|
| 100 | ~60 KB | ~600 bytes |
| 1,000 | ~600 KB | ~600 bytes |
| 10,000 | ~6 MB | ~600 bytes |

#### Snapshot Analysis

- **Throughput**: ~1M items/second for both save and load
- **Atomicity**: Uses temp file + rename for crash-safe writes
- **Bloom filter overhead**: ~3-5% slower to rebuild filter during load
- **Scaling**: Near-linear with dataset size
- **File size**: ~600 bytes per item with 128-dim embeddings (dominated by vector data)

### Sparse Vectors

SparseVector provides memory-efficient storage for high-sparsity embeddings by storing only non-zero values.

**Construction (768d):**
| Sparsity | Time | Throughput |
|----------|------|------------|
| 50% | 1.2 µs | 640K/s |
| 90% | 890 ns | 870K/s |
| 99% | 650 ns | 1.18M/s |

**Dot Product (768d):**
| Sparsity | Sparse-Sparse | Sparse-Dense | Dense-Dense | Sparse Speedup |
|----------|---------------|--------------|-------------|----------------|
| 50% | 2.1 µs | 1.8 µs | 580 ns | 0.3x (slower) |
| 90% | 380 ns | 290 ns | 580 ns | 1.5-2x |
| 99% | 38 ns | 26 ns | 580 ns | **15-22x** |

**Memory Compression:**
| Dimension | Sparsity | Dense Size | Sparse Size | Ratio |
|-----------|----------|------------|-------------|-------|
| 768 | 90% | 3,072 B | 1,024 B | **3x** |
| 768 | 99% | 3,072 B | 96 B | **32x** |
| 1536 | 99% | 6,144 B | 184 B | **33x** |

**Batch Search (1000 vectors, 768d, 90% sparse):**
| Method | Time | Throughput |
|--------|------|------------|
| Sparse corpus | 2.8 ms | 357K/s |
| Dense corpus | 2.1 ms | 476K/s |
| Dense baseline | 1.9 ms | 526K/s |

#### Analysis

- **High sparsity sweet spot**: At 99% sparsity, dot products are 15-22x faster than dense
- **Memory scaling**: Compression ratio = 1 / (1 - sparsity), so 99% sparse = ~100x smaller
- **Construction overhead**: Negligible (~1µs per vector)
- **Use case**: Embeddings from sparse models, one-hot encodings, pruned representations

### Delta Vectors

DeltaVector stores embeddings as differences from reference "archetype" vectors, ideal for clustered embeddings.

**Construction (768d, 5% delta):**
| Dimension | Time | Throughput |
|-----------|------|------------|
| 128 | 1.9 µs | 526K/s |
| 768 | 12.3 µs | 81K/s |
| 1536 | 25.1 µs | 40K/s |

**Dot Product (768d, precomputed archetype dot):**
| Method | Time | vs Dense |
|--------|------|----------|
| Delta precomputed | 89 ns | **6.5x faster** |
| Delta full | 620 ns | ~same |
| Dense baseline | 580 ns | 1x |

**Same-Archetype Dot Product (768d):**
| Method | Time | Speedup |
|--------|------|---------|
| Delta-delta | 145 ns | **4x** |
| Dense baseline | 580 ns | 1x |

**Memory (768d):**
| Delta Fraction | Dense Size | Delta Size | Ratio |
|----------------|------------|------------|-------|
| 1% diff | 3,072 B | 120 B | **25x** |
| 5% diff | 3,072 B | 360 B | **8.5x** |
| 10% diff | 3,072 B | 680 B | **4.5x** |

**Archetype Registry (8 archetypes, 768d):**
| Operation | Time |
|-----------|------|
| find_best_archetype | 4.2 µs |
| encode | 14 µs |
| decode | 1.1 µs |

#### Analysis

- **Precomputed speedup**: With archetype dot products cached, 6.5x faster than dense
- **Cluster-friendly**: Similar vectors share archetypes, deltas are sparse
- **Use case**: Semantic embeddings that cluster (documents, user profiles, products)

### K-means Clustering

K-means discovers archetype vectors automatically from embedding collections.

**K-means fit (128d, k=5):**
| Vectors | Time | Throughput |
|---------|------|------------|
| 100 | 50 µs | 2.0M elem/s |
| 500 | 241 µs | 2.1M elem/s |
| 1000 | 482 µs | 2.1M elem/s |

**Varying k (1000 vectors, 128d):**
| k | Time | Throughput |
|---|------|------------|
| 2 | 183 µs | 5.5M elem/s |
| 5 | 482 µs | 2.1M elem/s |
| 10 | 984 µs | 1.0M elem/s |
| 20 | 14.5 ms | 69K elem/s |

**Varying dimension (500 vectors, k=5):**
| Dimension | Time | Throughput |
|-----------|------|------------|
| 64 | 115 µs | 8.7M elem/s |
| 128 | 240 µs | 4.2M elem/s |
| 384 | 994 µs | 1.0M elem/s |
| 768 | 2.5 ms | 395K elem/s |

**Initialization Methods (1000 vectors, k=10):**
| Method | Time | Quality |
|--------|------|---------|
| Random | 4.2 ms | Variable |
| K-means++ | 1.7 ms | Better convergence |

**Full Pipeline (discover archetypes + encode batch):**
| Vectors | Time | Throughput |
|---------|------|------------|
| 500 | 1.67 ms | 300K elem/s |
| 1000 | 1.33 ms | 750K elem/s |

**Coverage Analysis (1000 vectors, 5 archetypes):**
| Operation | Time |
|-----------|------|
| analyze_coverage | 1.5 ms |

#### Analysis

- **K-means++ is faster**: Better initial centroids mean fewer iterations to converge
- **Linear with n**: Doubling vectors roughly doubles time
- **Quadratic with k at high k**: Each iteration is O(n*k), and more clusters need more iterations
- **Dimension scaling**: Linear with dimension (distance calculations dominate)
- **Use case**: Auto-discover archetypes for delta encoding, cluster analysis, centroid-based search

### tensor_compress

The tensor_compress crate provides compression algorithms optimized for tensor data: vector quantization, delta encoding, and run-length encoding.

**Vector Quantization (768-dim embedding):**
| Operation | Time | Throughput | Peak RAM |
|-----------|------|------------|----------|
| quantize_int8 | 286 ns | 3.5M vectors/s | ~4 KB |
| quantize_binary | 430 ns | 2.3M vectors/s | ~4 KB |

**Delta Encoding (10K sequential IDs):**
| Operation | Time | Throughput | Peak RAM |
|-----------|------|------------|----------|
| compress_ids | 8.0 µs | 1.25M IDs/s | ~100 KB |
| decompress_ids | 33 µs | 303K IDs/s | ~100 KB |

**Run-Length Encoding (100K values):**
| Operation | Time | Throughput | Peak RAM |
|-----------|------|------------|----------|
| rle_encode | 29 µs | 3.4M values/s | ~400 KB |
| rle_decode | 38 µs | 2.6M values/s | ~400 KB |

**Compression Ratios:**
| Data Type | Technique | Ratio | Lossless |
|-----------|-----------|-------|----------|
| f32 embeddings | Int8 quantization | 4x | No (~1% error) |
| f32 embeddings | Binary quantization | 32x | No (lossy) |
| Sequential IDs | Delta + varint | 4-8x | Yes |
| Repeated values | RLE | 2-100x | Yes |

#### Analysis

- **Quantization speed**: Sub-microsecond for typical 768-dim embeddings (GPT-style)
- **Int8 vs Binary**: Int8 is faster (286ns vs 430ns) and more accurate
- **Delta encoding**: Asymmetric - compression is 4x faster than decompression
- **RLE**: Best for highly repeated data (status columns, category IDs)
- **Memory efficiency**: All operations use < 500 KB for typical data sizes
- **Integration**: Use `SAVE COMPRESSED` in shell or `save_snapshot_compressed()` API
- **Trade-off**: Int8 quantization trades ~1% accuracy for 4x size reduction

### tensor_vault

The tensor_vault crate provides AES-256-GCM encrypted secret storage with graph-based access control, permission levels, TTL grants, rate limiting, namespace isolation, audit logging, and secret versioning.

**Key Derivation (Argon2id):**
| Operation | Time | Peak RAM |
|-----------|------|----------|
| argon2id_derivation | 80 ms | ~64 MB |

Note: Argon2id is intentionally slow to resist brute-force attacks. The 64MB memory cost is configurable via `VaultConfig`.

**Encryption/Decryption (AES-256-GCM):**
| Operation | Time | Peak RAM |
|-----------|------|----------|
| set_1kb | 29 µs | ~3 KB |
| get_1kb | 24 µs | ~3 KB |
| set_10kb | 93 µs | ~25 KB |
| get_10kb | 91 µs | ~25 KB |

Note: `set` includes versioning overhead (storing previous version pointers). `get` includes audit logging.

**Access Control (Graph Path Verification):**
| Operation | Time | Peak RAM |
|-----------|------|----------|
| check_shallow (1 hop) | 6 µs | ~2 KB |
| check_deep (10 hops) | 17 µs | ~3 KB |
| grant | 18 µs | ~1 KB |
| revoke | 1.07 ms | ~1 KB |

**Secret Listing:**
| Operation | Time | Peak RAM |
|-----------|------|----------|
| list_100_secrets | 291 µs | ~4 KB |
| list_1000_secrets | 2.7 ms | ~40 KB |

Note: List includes access control checks and key name decryption for pattern matching.

#### Analysis

- **Key derivation**: Argon2id dominates vault initialization (~80ms). This is by design for security.
- **Access check improved**: Path verification is now ~6µs for shallow, ~17µs for deep (85% faster than before).
- **Versioning overhead**: `set` is ~2x slower due to version tracking (stores pointer array).
- **Audit overhead**: Every operation logs to audit store (adds ~5-10µs per operation).
- **Revoke performance**: ~1ms due to edge deletion, TTL tracker cleanup, and audit logging.
- **List scaling**: ~2.7µs per secret at 1000 (includes decryption for pattern matching).

**New Feature Performance:**
| Feature | Overhead |
|---------|----------|
| Permission check | ~1 µs (edge type comparison) |
| Rate limit check | ~100 ns (DashMap lookup) |
| TTL check | ~50 ns (heap peek) |
| Audit log write | ~5 µs (tensor store put) |
| Version tracking | ~10 µs (pointer array update) |

**Security vs Performance Trade-offs:**
| Configuration | Key Derivation | Security |
|---------------|----------------|----------|
| Default (64MB, 3 iter) | ~80 ms | High |
| Fast (16MB, 1 iter) | ~25 ms | Medium |
| Paranoid (256MB, 10 iter) | ~800 ms | Very High |

### graph_engine

The graph engine stores nodes and edges as tensors, using adjacency lists for neighbor lookups.

**Node Creation:**
| Count | Time | Per Node |
|-------|------|----------|
| 100 | 107µs | 1.07µs |
| 1,000 | 1.67ms | 1.67µs |
| 5,000 | 9.4ms | 1.88µs |

**Edge Creation (1,000 edges):**
| Type | Time | Per Edge |
|------|------|----------|
| Directed | 2.4ms | 2.4µs |
| Undirected | 3.6ms | 3.6µs |

**Neighbor Lookup (star graph):**
| Fan-out | Time | Per Neighbor |
|---------|------|--------------|
| 10 | 16µs | 1.6µs |
| 50 | 79µs | 1.6µs |
| 100 | 178µs | 1.8µs |

**BFS Traversal (binary tree):**
| Depth | Nodes | Time | Per Node |
|-------|-------|------|----------|
| 5 | 31 | 110µs | 3.5µs |
| 7 | 127 | 442µs | 3.5µs |
| 9 | 511 | 1.5ms | 2.9µs |

**Shortest Path (BFS):**
| Graph Type | Size | Time |
|------------|------|------|
| Chain | 10 nodes | 8.2µs |
| Chain | 50 nodes | 44µs |
| Chain | 100 nodes | 96µs |
| Grid | 5x5 | 55µs |
| Grid | 10x10 | 265µs |

#### Analysis

- **Undirected edges**: ~50% slower than directed (stores reverse edge internally)
- **Traversal**: Consistent ~3µs per node visited, good BFS implementation
- **Path finding**: Near-linear with path length in chains; grid explores more nodes
- **Parallel delete_node**: Uses rayon for high-degree nodes (>100 edges)
- **Memory overhead**: Each node/edge is a full TensorData (~5-10 allocations)

### relational_engine

The relational engine provides SQL-like operations on top of tensor_store, with optional hash indexes for accelerated equality lookups.

**Row Insertion:**
| Count | Time | Throughput |
|-------|------|------------|
| 100 | 420µs | 238K rows/s |
| 1,000 | 5.9ms | 170K rows/s |
| 5,000 | 77ms | 65K rows/s |

**Select Full Scan:**
| Rows | Time | Throughput |
|------|------|------------|
| 100 | 104µs | 960K rows/s |
| 1,000 | 977µs | 1.02M rows/s |
| 5,000 | 5.8ms | 857K rows/s |

**Select with Index vs Without (5,000 rows):**
| Query Type | With Index | Without Index | Speedup |
|------------|------------|---------------|---------|
| Equality (2% match) | 126µs | 5.96ms | **47x** |
| By _id (single row) | 3.5µs | 5.59ms | **1,597x** |

**Select Filtered - No Index (5,000 rows):**
| Filter Type | Time |
|-------------|------|
| Range (20% match) | 5.1ms |
| Compound AND | 5.7ms |

**Index Creation (parallel):**
| Rows | Time |
|------|------|
| 100 | 200µs |
| 1,000 | 1.3ms |
| 5,000 | 9.6ms |

**Update/Delete (1,000 rows, 10% affected):**
| Operation | Time |
|-----------|------|
| Update | 2.0ms |
| Delete | 1.6ms |

**Join Performance (hash join):**
| Tables | Result Rows | Time |
|--------|-------------|------|
| 50 users × 500 posts | 500 | 630µs |
| 100 users × 1000 posts | 1,000 | 2.1ms |
| 100 users × 5000 posts | 5,000 | 3.6ms |

**Row Count:**
| Rows | Time |
|------|------|
| 100 | 3.7µs |
| 1,000 | 9.9µs |
| 5,000 | 56µs |

#### Analysis

- **Index acceleration**: Hash indexes provide O(1) lookup for equality conditions
  - 47x speedup for equality queries matching 2% of rows
  - 1,597x speedup for single-row _id lookups
- **Full scan cost**: Without index, O(n) for all queries (parallelized for >1000 rows)
- **Parallel operations**: update/delete/create_index use rayon for condition evaluation (28-45% faster)
- **Index maintenance**: Small overhead on insert/update/delete to maintain indexes
- **Join complexity**: O(n+m) hash join (2-5x faster than nested loop)
- **row_count**: Uses scan_count, much faster than full scan (~100x)

### vector_engine

The vector engine stores embeddings and performs k-nearest neighbor search using cosine similarity.

**Store Embedding:**
| Dimension | Time | Throughput |
|-----------|------|------------|
| 128 | 366 ns | 2.7M/s |
| 768 | 892 ns | 1.1M/s |
| 1536 | 969 ns | 1.0M/s |

**Get Embedding:**
| Dimension | Time |
|-----------|------|
| 768 | 287 ns |

**Delete Embedding:**
| Operation | Time |
|-----------|------|
| delete | 806 ns |

**Similarity Search (top 10, SIMD + adaptive parallel):**
| Dataset | Time | Per Vector | Mode |
|---------|------|------------|------|
| 1,000 x 128d | 242 µs | 242 ns | Sequential |
| 1,000 x 768d | 367 µs | 367 ns | Sequential |
| 10,000 x 128d | 1.93 ms | 193 ns | Parallel |

**Cosine Similarity Computation (SIMD-accelerated):**
| Dimension | Time |
|-----------|------|
| 128 | 26 ns |
| 768 | 165 ns |
| 1536 | 369 ns |

#### Analysis

- **SIMD acceleration**: 8-wide f32 SIMD (via `wide` crate) provides 3-9x speedup for cosine similarity
- **Adaptive parallelism**: Uses rayon for parallel search when >5000 vectors (1.6x speedup at 10K)
- **Linear scaling with dimension**: Cosine similarity is O(d) where d is vector dimension
- **Linear scaling with dataset size**: Brute-force search is O(n*d) for n vectors
- **Memory bound**: For 768d vectors, ~3 KB per embedding (768 * 4 bytes)
- **Search throughput**: ~4M vector comparisons/second at 128d (with SIMD)
- **Store/Get performance**: Sub-microsecond for typical embedding sizes

**Complexity:**
| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| store_embedding | O(d) | Vector copy + hash insert |
| get_embedding | O(d) | Hash lookup + vector clone |
| delete_embedding | O(1) | Hash removal |
| search_similar | O(n*d) | Brute-force scan |
| compute_similarity | O(d) | Dot product + 2 magnitude calculations |

**HNSW Index (Approximate Nearest Neighbor):**

HNSW provides O(log n) search complexity instead of O(n) brute force.

| Configuration | Search Time (5K, 128d) |
|---------------|------------------------|
| high_speed | ~50 µs |
| default | ~100 µs |
| high_recall | ~200 µs |

**HNSW vs Brute Force (10K vectors, 128d):**
| Method | Search Time | Speedup |
|--------|-------------|---------|
| Brute force | ~2 ms | 1x |
| HNSW default | ~150 µs | ~13x |

Recommended approach by corpus size:
| Corpus Size | Approach | Rationale |
|-------------|----------|-----------|
| < 10K | Brute force | Fast enough, pure tensor |
| 10K - 100K | HNSW | Pragmatic, 5-13x faster |
| > 100K | HNSW | Necessary for latency |

**Scaling Projections (HNSW for >10K vectors):**
| Vectors | Dimension | Search Time (est.) |
|---------|-----------|-------------------|
| 10K | 768 | ~200 µs |
| 100K | 768 | ~500 µs |
| 1M | 768 | ~1 ms |

For production workloads at extreme scale (>1M vectors), consider:
- Sharded HNSW across multiple nodes
- Dimensionality reduction (PCA)
- Quantization (int8, binary)

### tensor_cache

The tensor_cache crate provides LLM response caching with exact, semantic (HNSW), and embedding caches.

**Exact Cache (Hash-based O(1)):**
| Operation | Time |
|-----------|------|
| lookup_hit | 208 ns |
| lookup_miss | 102 ns |

**Semantic Cache (HNSW-based O(log n)):**
| Operation | Time |
|-----------|------|
| lookup_hit | 21 µs |

**Put (Exact + Semantic + HNSW insert):**
| Entries | Time |
|---------|------|
| 100 | 49 µs |
| 1,000 | 47 µs |
| 10,000 | 53 µs |

**Embedding Cache:**
| Operation | Time |
|-----------|------|
| lookup_hit | 230 ns |
| lookup_miss | 110 ns |

**Eviction (batch processing):**
| Entries in Cache | Time |
|------------------|------|
| 1,000 | 5.1 µs |
| 5,000 | 4.3 µs |
| 10,000 | 8.1 µs |

#### Analysis

- **Exact cache**: Hash-based O(1) lookup provides sub-microsecond hit/miss detection
- **Semantic cache**: HNSW index provides O(log n) similarity search (~21µs for hit)
- **Embedding cache**: Fast O(1) lookup for precomputed embeddings
- **Put performance**: Consistent ~50µs regardless of cache size (HNSW insert is O(log n))
- **Eviction**: Efficient batch eviction with LRU/LFU/Cost/Hybrid strategies
- **Token counting**: tiktoken cl100k_base encoding for accurate GPT-4 token counts
- **Cost tracking**: Estimates cost savings based on model pricing tables

**Cache Layers:**
| Layer | Complexity | Use Case |
|-------|------------|----------|
| Exact | O(1) | Identical prompts |
| Semantic | O(log n) | Similar prompts |
| Embedding | O(1) | Precomputed embeddings |

**Eviction Strategies:**
| Strategy | Description |
|----------|-------------|
| LRU | Evict least recently accessed |
| LFU | Evict least frequently accessed |
| CostBased | Evict lowest cost efficiency |
| Hybrid | Weighted combination (recommended) |

### neumann_parser

The parser is a hand-written recursive descent parser with Pratt expression parsing for operator precedence.

**Tokenization:**
| Query Type | Time | Throughput |
|------------|------|------------|
| simple_select | 182 ns | 99 MiB/s |
| select_where | 640 ns | 88 MiB/s |
| complex_select | 986 ns | 95 MiB/s |
| insert | 493 ns | 120 MiB/s |
| update | 545 ns | 91 MiB/s |
| node | 625 ns | 98 MiB/s |
| edge | 585 ns | 94 MiB/s |
| path | 486 ns | 75 MiB/s |
| embed | 407 ns | 138 MiB/s |
| similar | 185 ns | 118 MiB/s |

**Parsing (tokenize + parse):**
| Query Type | Time | Throughput |
|------------|------|------------|
| simple_select | 235 ns | 77 MiB/s |
| select_where | 1.19 µs | 47 MiB/s |
| complex_select | 1.89 µs | 50 MiB/s |
| insert | 688 ns | 86 MiB/s |
| update | 806 ns | 61 MiB/s |
| delete | 464 ns | 62 MiB/s |
| create_table | 856 ns | 80 MiB/s |
| node | 837 ns | 81 MiB/s |
| edge | 750 ns | 74 MiB/s |
| neighbors | 520 ns | 55 MiB/s |
| path | 380 ns | 58 MiB/s |
| embed_store | 650 ns | 86 MiB/s |
| similar | 290 ns | 76 MiB/s |

**Expression Complexity:**
| Expression Type | Time |
|-----------------|------|
| simple (a = 1) | 350 ns |
| binary_and | 580 ns |
| binary_or | 570 ns |
| nested_and_or | 950 ns |
| deep_nesting | 1.5 µs |
| arithmetic | 720 ns |
| comparison_chain | 1.3 µs |

**Batch Parsing Throughput:**
| Batch Size | Time | Queries/s |
|------------|------|-----------|
| 10 | 5.2 µs | 1.9M/s |
| 100 | 52 µs | 1.9M/s |
| 1,000 | 520 µs | 1.9M/s |

**Large Query Parsing:**
| Query Type | Time |
|------------|------|
| INSERT 100 rows | 45 µs |
| EMBED 768-dim vector | 38 µs |
| WHERE 20 conditions | 8.5 µs |

#### Analysis

- **Zero dependencies**: Hand-written lexer and parser with no external crates
- **Consistent throughput**: ~75-120 MiB/s across query types
- **Expression complexity**: Linear scaling with expression depth
- **Batch performance**: Consistent 1.9M queries/second regardless of batch size
- **Large vectors**: 768-dim embedding parsing in ~38µs (20K dimensions/second)

### query_router

The query router integrates all engines and routes queries based on parsed AST type.

**Relational Operations:**
| Operation | Time |
|-----------|------|
| SELECT * (100 rows) | 17 µs |
| SELECT WHERE | 17 µs |
| INSERT | 290 µs |
| UPDATE | 6.5 ms |

**Graph Operations:**
| Operation | Time |
|-----------|------|
| NODE CREATE | 2.3 µs |
| EDGE CREATE | 3.5 µs |
| NEIGHBORS | 1.8 µs |
| PATH (1 -> 10) | 85 µs |
| FIND NODE | 1.2 µs |

**Vector Operations:**
| Operation | Time |
|-----------|------|
| EMBED STORE (128d) | 28 µs |
| EMBED GET | 1.5 µs |
| SIMILAR LIMIT 5 (100 vectors) | 10 ms |
| SIMILAR LIMIT 10 (100 vectors) | 10 ms |

**Mixed Workload:**
| Configuration | Time | Queries/s |
|---------------|------|-----------|
| 5 mixed queries (SELECT, NEIGHBORS, SIMILAR, INSERT, NODE) | 11 ms | 455/s |

**Insert Throughput:**
| Batch Size | Time | Rows/s |
|------------|------|--------|
| 100 | 29 ms | 3.4K/s |
| 500 | 145 ms | 3.4K/s |
| 1,000 | 290 ms | 3.4K/s |

#### Analysis

- **Parse overhead**: Parser adds ~200ns-2µs per query (negligible vs execution)
- **Routing overhead**: AST-based routing is O(1) pattern matching
- **Relational**: SELECT is fast (17µs); UPDATE scans all rows (6.5ms for 100 rows)
- **Graph**: Node/edge creation ~2-3µs; path finding scales with path length
- **Vector**: Similarity search dominates mixed workloads (~10ms for 100 vectors)
- **Bottleneck identification**: SIMILAR queries are the slowest operation; use HNSW index for large vector stores

### neumann_shell

The shell provides an interactive REPL interface with readline support, routing queries through the query_router.

**Command Execution (with fresh shell per iteration):**
| Operation | Time | Peak RAM |
|-----------|------|----------|
| empty_input | 2.4 µs | ~0.4 KB |
| help | 2.4 µs | ~0.5 KB |
| SELECT * (100 rows) | 129 µs | ~67 KB |
| SELECT WHERE (100 rows) | 86 µs | ~67 KB |

**Output Formatting:**
| Operation | Time | Peak RAM |
|-----------|------|----------|
| format_1000_rows | 3.2 ms | ~1.3 MB |

**Insert Scaling:**
| Rows | Time | Peak RAM |
|------|------|----------|
| 100 | ~30 ms | ~180 KB |
| 500 | ~150 ms | ~600 KB |
| 1000 | ~300 ms | ~1.2 MB |

#### Analysis

- **Shell creation**: ~2.4µs overhead for creating a fresh shell instance
- **Query execution**: Dominated by relational_engine execution time
- **Output formatting**: ASCII table formatting at ~3.2µs per row (3.2ms / 1000 rows)
- **Memory scaling**: ~1.2 KB per row for typical data
- **Readline integration**: rustyline provides history, arrow keys, and Ctrl+C handling with no measurable overhead

## Performance Characteristics

### DashMap Sharding

tensor_store uses DashMap which internally shards data across ~16 RwLocks:

```
Key → hash(key) % 16 → shard[n] → RwLock<HashMap>
```

This means:
- 16 concurrent writes to different shards can proceed in parallel
- Reads never block other reads (RwLock semantics)
- Write contention only occurs when two writes hash to the same shard

### Graph Storage Model

graph_engine stores each node and edge as a separate tensor:

```
node:{id} → TensorData { label, properties... }
edge:{id} → TensorData { from, to, label, directed, properties... }
adj:{node_id}:out → TensorData { edge_ids: [...] }
adj:{node_id}:in → TensorData { edge_ids: [...] }
```

Trade-offs:
- **Pro**: Flexible property storage, consistent with tensor model
- **Con**: More key lookups than traditional adjacency list
- **Pro**: Each component independently updatable

### Relational Storage Model

relational_engine stores each row as a tensor:

```
_meta:table:{name} → TensorData { schema... }
{table}:{row_id} → TensorData { column values... }
```

Hash indexes provide O(1) lookup for equality conditions, with automatic maintenance on insert/update/delete.

### Vector Storage Model

vector_engine stores each embedding as a tensor:

```
emb:{key} → TensorData { vector: [...] }
```

Trade-offs:
- **Pro**: Simple storage model, consistent with tensor abstraction
- **Pro**: Sub-microsecond store/get operations
- **Pro**: HNSW index for O(log n) approximate nearest neighbor search
- **Con**: Brute-force O(n*d) for exact search (use HNSW for approximate)

## Optimization Opportunities

1. ~~**Batch operations**~~: Done - batch_insert provides 59x speedup for bulk inserts (5000 rows: 7.4ms vs 441ms)
2. ~~**B-tree indexes**~~: Done - range query acceleration for relational_engine
3. ~~**Memory pools**~~: Done - HashMap/Vec pre-allocation with with_capacity in hot paths
4. ~~**Parallel scans**~~: Done - adaptive rayon parallelism for tensor_store (25-53%), vector_engine (1.6x), and relational_engine select (2-3x)
5. ~~**Bloom filters**~~: Done - optional thread-safe Bloom filter for sparse key spaces. Note: DashMap's O(1) hash lookup is already ~50ns, so Bloom filters add ~15ns overhead for in-memory stores. Useful when backing store is disk/network.
6. ~~**ANN indexing**~~: Done - HNSW index for vector_engine, provides O(log n) search vs O(n) brute force
7. ~~**SIMD acceleration**~~: Done - 8-wide f32 SIMD provides 3-9x speedup for cosine similarity
8. ~~**LLM caching**~~: Done - tensor_cache provides multi-layer caching (exact O(1), semantic O(log n)) with cost tracking and background eviction
9. ~~**Sparse vectors**~~: Done - SparseVector provides 3-33x memory reduction and 10-22x dot product speedup at high sparsity (90-99%)
10. ~~**Delta encoding**~~: Done - DeltaVector stores only differences from archetypes, 4-25x memory reduction for clustered embeddings
11. ~~**Archetype discovery**~~: Done - K-means clustering with k-means++ initialization for automatic archetype discovery (~500µs for 1000 128d vectors)

## Hardware Notes

Benchmarks run on:
- Apple M-series (ARM64) or Intel x86_64
- Results may vary based on:
  - CPU cache sizes (L1/L2/L3)
  - Memory bandwidth
  - Number of cores (for concurrent benchmarks)

For consistent benchmarking:
```bash
# Disable CPU frequency scaling (Linux)
sudo cpupower frequency-set --governor performance

# Run with minimal background activity
cargo bench -- --noplot  # Skip HTML report generation for faster runs
```
