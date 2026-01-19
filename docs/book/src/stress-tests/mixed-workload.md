# Mixed Workload Stress Tests

Stress tests that exercise all Neumann engines simultaneously with realistic workload patterns.

## Test Suite

| Test | Scale | Description |
|------|-------|-------------|
| `stress_all_engines_concurrent` | 25K ops/thread, 12 threads | All engines under concurrent load |
| `stress_realistic_workload` | 30s duration | Mixed OLTP + search + traversal |

## Results

| Test | Key Metric | Result |
|------|------------|--------|
| All engines | Combined throughput | **841 ops/sec** |
| All engines | Relational p50 | 12ms |
| All engines | Graph p50 | 5us |
| All engines | Vector p50 | < 1us |
| Realistic workload | Mixed throughput | **232 ops/sec** |
| Realistic workload | Read rate | 91 reads/sec |
| Realistic workload | Write rate | 68 writes/sec |
| Realistic workload | Search rate | 72 searches/sec |

## Running

```bash
# Run all mixed workload stress tests
cargo test --release -p stress_tests --test mixed_workload_stress -- --ignored --nocapture

# Run specific test
cargo test --release -p stress_tests stress_all_engines_concurrent -- --ignored --nocapture
```

## All Engines Concurrent

Tests all engines (relational, graph, vector) under simultaneous heavy load from 12 threads.

**What it validates:**
- Cross-engine concurrency safety
- Shared TensorStore contention handling
- No deadlocks or livelocks
- Correct results under maximum stress

**Workload distribution per thread:**
- Relational: INSERT, SELECT, UPDATE
- Graph: NODE, EDGE, NEIGHBORS
- Vector: EMBED, SIMILAR

**Expected behavior:**
- No panics or assertion failures
- All operations complete (no hangs)
- Data consistency verified post-test

## Realistic Workload

Simulates a production-like mixed workload over 30 seconds.

**What it validates:**
- Sustained throughput over time
- Memory stability (no leaks)
- Latency consistency

**Workload pattern:**
- 40% Reads (SELECT, GET, NEIGHBORS)
- 30% Writes (INSERT, UPDATE, NODE)
- 30% Searches (SIMILAR, PATH)

**Expected behavior:**
- Throughput variance < 20%
- Memory usage stable
- No degradation over time

## Engine Latency Breakdown

| Engine | Operation | Typical p50 | Notes |
|--------|-----------|-------------|-------|
| Relational | SELECT | 1-10ms | Schema lookup overhead |
| Relational | INSERT | 3-15ms | Index maintenance |
| Graph | NEIGHBORS | 5-50us | Adjacency list lookup |
| Graph | PATH | 100us-5ms | Scales with path length |
| Vector | EMBED | 1-5us | Hash insert |
| Vector | SIMILAR | 1-100ms | Scales with corpus size |

## Bottleneck Identification

When mixed workload throughput is lower than expected:

1. **Vector search dominates**: Use HNSW index for SIMILAR queries
2. **Relational scans**: Add hash/B-tree indexes on filter columns
3. **Graph traversals**: Add LIMIT to NEIGHBORS/PATH queries
4. **Contention**: Check hot shards with instrumentation

## Scaling Considerations

| Bottleneck | Solution |
|------------|----------|
| CPU-bound | Add more cores, enable rayon parallelism |
| Memory-bound | Enable tiered storage, use sparse vectors |
| I/O-bound | Use NVMe storage, increase buffer sizes |
| Network-bound | Batch operations, use local cache |
