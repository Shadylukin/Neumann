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
```

Benchmark reports are generated in `target/criterion/` with HTML visualizations.

## Benchmark Results Summary

### tensor_store

The tensor store uses DashMap (sharded concurrent HashMap) for thread-safe key-value storage.

| Operation | 100 items | 1,000 items | 10,000 items |
|-----------|-----------|-------------|--------------|
| **put** | 40µs (2.5M/s) | 447µs (2.2M/s) | 7ms (1.4M/s) |
| **get** | 33µs (3.0M/s) | 320µs (3.1M/s) | 3ms (3.3M/s) |

**Scan Operations (10k total items):**
| Operation | Time |
|-----------|------|
| scan 1k keys | 256µs |
| scan_count 1k keys | 88µs |

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
- **scan_count vs scan**: Count-only is 3x faster (avoids string cloning)

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

**Index Creation:**
| Rows | Time |
|------|------|
| 100 | 245µs |
| 1,000 | 2.7ms |
| 5,000 | 16ms |

**Update/Delete (1,000 rows, 10% affected):**
| Operation | Time |
|-----------|------|
| Update | 2.0ms |
| Delete | 1.6ms |

**Join Performance (nested loop):**
| Tables | Result Rows | Time |
|--------|-------------|------|
| 50 users × 500 posts | 500 | 1.2ms |
| 100 users × 1000 posts | 1,000 | 4.0ms |
| 100 users × 5000 posts | 5,000 | 20ms |

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
- **Full scan cost**: Without index, O(n) for all queries
- **Index maintenance**: Small overhead on insert/update/delete to maintain indexes
- **Join complexity**: O(n×m) nested loop join
- **row_count**: Uses scan_count, much faster than full scan (~100x)

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

Current implementation uses full table scans for queries. Future optimizations:
- Secondary indexes for filtered queries
- Sorted storage for range queries
- Hash indexes for equality lookups

## Optimization Opportunities

1. **Batch operations**: Add bulk insert/update APIs to reduce per-operation overhead
2. **Index support**: Add B-tree or hash indexes for relational_engine
3. **Memory pools**: Reuse TensorData allocations in hot paths
4. **Parallel scans**: Use rayon for parallel query execution
5. **Bloom filters**: Quick negative lookups for sparse key spaces

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
