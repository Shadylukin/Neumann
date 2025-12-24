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

**Scaling Projections (SIMD + parallel for >5K vectors):**
| Vectors | Dimension | Search Time (est.) |
|---------|-----------|-------------------|
| 10K | 768 | ~2 ms |
| 100K | 768 | ~20 ms |
| 1M | 768 | ~200 ms |

For production workloads with >100K vectors, consider:
- Approximate Nearest Neighbor (ANN) algorithms (HNSW, IVF)
- Dimensionality reduction (PCA)
- Quantization (int8, binary)

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
- **Con**: Brute-force O(n*d) similarity search
- **Con**: No indexing for approximate nearest neighbor

## Optimization Opportunities

1. **Batch operations**: Add bulk insert/update APIs to reduce per-operation overhead
2. **B-tree indexes**: Range query acceleration for relational_engine (hash indexes done)
3. **Memory pools**: Reuse TensorData allocations in hot paths
4. ~~**Parallel scans**~~: Done - adaptive rayon parallelism for tensor_store (25-53%), vector_engine (1.6x), and relational_engine select (2-3x)
5. **Bloom filters**: Quick negative lookups for sparse key spaces
6. **ANN indexing**: HNSW or IVF for vector_engine similarity search at scale
7. ~~**SIMD acceleration**~~: Done - 8-wide f32 SIMD provides 3-9x speedup for cosine similarity

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
