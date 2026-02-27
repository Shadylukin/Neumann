# tensor_store Benchmarks

The tensor store uses DashMap (sharded concurrent HashMap) for thread-safe
key-value storage.

<!-- BENCH:START -->
## Core Operations

| Operation | 100 items | 1,000 items | 10,000 items |
| --- | --- | --- | --- |
| **put** | 40us (2.5M/s) | 447us (2.2M/s) | 7ms (1.4M/s) |
| **get** | 33us (3.0M/s) | 320us (3.1M/s) | 3ms (3.3M/s) |

## Scan Operations (10k total items, parallel)

| Operation | Time |
| --- | --- |
| scan 1k keys | 191us |
| scan_count 1k keys | 41us |

## Concurrent Write Performance

| Threads | Disjoint Keys | High Contention (100 keys) |
| --- | --- | --- |
| 2 | 795us | 974us |
| 4 | 1.59ms | 1.48ms |
| 8 | 4.6ms | 2.33ms |

## Mixed Workload

| Configuration | Time |
| --- | --- |
| 4 readers + 2 writers | 579us |
<!-- BENCH:END -->

## Analysis

- **Read vs Write**: Reads are ~20% faster than writes due to DashMap's
  read-optimized design
- **Scaling**: Near-linear scaling up to 10k items; slight degradation at scale
  due to hash table growth
- **Concurrency**: DashMap's 16-shard design provides excellent concurrent
  performance
- **Contention**: Under high contention, performance actually improves at 8
  threads vs 4 (lock sharding distributes load)
- **Parallel scans**: Uses rayon for >1000 keys (25-53% faster)
- **scan_count vs scan**: Count-only is ~5x faster (avoids string cloning)

## Bloom Filter (optional)

| Operation | Time |
| --- | --- |
| add | 68 ns |
| might_contain (hit) | 46 ns |
| might_contain (miss) | 63 ns |

## Sparse Lookups (1K keys in store)

| Query Type | Without Bloom | With Bloom |
| --- | --- | --- |
| Negative lookup | 52 ns | 68 ns |
| Positive lookup | 45 ns | 60 ns |
| Sparse workload (90% miss) | 52 ns | 67 ns |

> **Note**: Bloom filter adds ~15ns overhead for in-memory DashMap stores. It's
designed for scenarios where the backing store is slower (disk, network, remote
database), where the early rejection of non-existent keys avoids expensive I/O.

## Snapshot Persistence (bincode)

| Operation | 100 items | 1,000 items | 10,000 items |
| --- | --- | --- | --- |
| **save** | 100 us (1.0M/s) | 927 us (1.08M/s) | 12.6 ms (791K/s) |
| **load** | 74 us (1.35M/s) | 826 us (1.21M/s) | 10.7 ms (936K/s) |
| **load_with_bloom** | 81 us (1.23M/s) | 840 us (1.19M/s) | 11.0 ms (908K/s) |

Each item is a TensorData with 3 fields: id (i64), name (String), embedding
(128-dim `Vec<f32>`).

## Snapshot File Sizes

| Items | File Size | Per Item |
| --- | --- | --- |
| 100 | ~60 KB | ~600 bytes |
| 1,000 | ~600 KB | ~600 bytes |
| 10,000 | ~6 MB | ~600 bytes |

## Snapshot Analysis

- **Throughput**: ~1M items/second for both save and load
- **Atomicity**: Uses temp file + rename for crash-safe writes
- **Bloom filter overhead**: ~3-5% slower to rebuild filter during load
- **Scaling**: Near-linear with dataset size
- **File size**: ~600 bytes per item with 128-dim embeddings (dominated by
  vector data)

## Write-Ahead Log (WAL)

WAL provides crash-consistent durability with minimal performance overhead.
Benchmarks use same payload as in-memory tests (128-dim embeddings).

### WAL Writes

| Records | Time | Throughput |
| --- | --- | --- |
| 100 | 152 us | 657K ops/s |
| 1,000 | 753 us | 1.33M ops/s |
| 10,000 | 6.95 ms | 1.44M ops/s |

### WAL Recovery

| Records | Time | Throughput |
| --- | --- | --- |
| 100 | 382 us | 261K elem/s |
| 1,000 | 394 us | 2.5M elem/s |
| 10,000 | 391 us | 25.6M elem/s |

### WAL Analysis

- **Near constant recovery time**: Recovery is dominated by file open overhead
  (~400us), not record count
- **Sequential I/O**: WAL replay reads sequentially, hitting 25M records/sec
- **Durable vs in-memory**: WAL writes at 1.4M ops/sec vs 2.0M ops/sec in-memory
  (72% of in-memory speed)
- **Use case**: Production deployments requiring crash consistency

All engines support WAL via `open_durable()`:

```rust
// Durable graph engine
let engine = GraphEngine::open_durable("data/graph.wal", WalConfig::default())?;

// Recovery after crash
let engine = GraphEngine::recover("data/graph.wal", &WalConfig::default(), None)?;
```

## Sparse Vectors

SparseVector provides memory-efficient storage for high-sparsity embeddings by
storing only non-zero values.

### Construction (768d)

| Sparsity | Time | Throughput |
| --- | --- | --- |
| 50% | 1.2 us | 640K/s |
| 90% | 890 ns | 870K/s |
| 99% | 650 ns | 1.18M/s |

### Dot Product (768d)

| Sparsity | Sparse-Sparse | Sparse-Dense | Dense-Dense | Sparse Speedup |
| --- | --- | --- | --- | --- |
| 50% | 2.1 us | 1.8 us | 580 ns | 0.3x (slower) |
| 90% | 380 ns | 290 ns | 580 ns | 1.5-2x |
| 99% | 38 ns | 26 ns | 580 ns | **15-22x** |

### Memory Compression

| Dimension | Sparsity | Dense Size | Sparse Size | Ratio |
| --- | --- | --- | --- | --- |
| 768 | 90% | 3,072 B | 1,024 B | **3x** |
| 768 | 99% | 3,072 B | 96 B | **32x** |
| 1536 | 99% | 6,144 B | 184 B | **33x** |

### Sparse Vector Analysis

- **High sparsity sweet spot**: At 99% sparsity, dot products are 15-22x faster
  than dense
- **Memory scaling**: Compression ratio = 1 / (1 - sparsity), so 99% sparse =
  ~100x smaller
- **Construction overhead**: Negligible (~1us per vector)
- **Use case**: Embeddings from sparse models, one-hot encodings, pruned
  representations

## Delta Vectors

DeltaVector stores embeddings as differences from reference "archetype" vectors,
ideal for clustered embeddings.

### Construction (768d, 5% delta)

| Dimension | Time | Throughput |
| --- | --- | --- |
| 128 | 1.9 us | 526K/s |
| 768 | 12.3 us | 81K/s |
| 1536 | 25.1 us | 40K/s |

### Dot Product (768d, precomputed archetype dot)

| Method | Time | vs Dense |
| --- | --- | --- |
| Delta precomputed | 89 ns | **6.5x faster** |
| Delta full | 620 ns | ~same |
| Dense baseline | 580 ns | 1x |

### Same-Archetype Dot Product (768d)

| Method | Time | Speedup |
| --- | --- | --- |
| Delta-delta | 145 ns | **4x** |
| Dense baseline | 580 ns | 1x |

### Delta Memory (768d)

| Delta Fraction | Dense Size | Delta Size | Ratio |
| --- | --- | --- | --- |
| 1% diff | 3,072 B | 120 B | **25x** |
| 5% diff | 3,072 B | 360 B | **8.5x** |
| 10% diff | 3,072 B | 680 B | **4.5x** |

### Archetype Registry (8 archetypes, 768d)

| Operation | Time |
| --- | --- |
| find_best_archetype | 4.2 us |
| encode | 14 us |
| decode | 1.1 us |

### Delta Vector Analysis

- **Precomputed speedup**: With archetype dot products cached, 6.5x faster than
  dense
- **Cluster-friendly**: Similar vectors share archetypes, deltas are sparse
- **Use case**: Semantic embeddings that cluster (documents, user profiles,
  products)

## K-means Clustering

K-means discovers archetype vectors automatically from embedding collections.

### K-means fit (128d, k=5)

| Vectors | Time | Throughput |
| --- | --- | --- |
| 100 | 50 us | 2.0M elem/s |
| 500 | 241 us | 2.1M elem/s |
| 1000 | 482 us | 2.1M elem/s |

### Varying k (1000 vectors, 128d)

| k | Time | Throughput |
| --- | --- | --- |
| 2 | 183 us | 5.5M elem/s |
| 5 | 482 us | 2.1M elem/s |
| 10 | 984 us | 1.0M elem/s |
| 20 | 14.5 ms | 69K elem/s |

### K-means Analysis

- **K-means++ is faster**: Better initial centroids mean fewer iterations to
  converge
- **Linear with n**: Doubling vectors roughly doubles time
- **Quadratic with k at high k**: Each iteration is O(n*k), and more clusters
  need more iterations
- **Use case**: Auto-discover archetypes for delta encoding, cluster analysis,
  centroid-based search
