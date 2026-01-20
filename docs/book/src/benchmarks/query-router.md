# query_router Benchmarks

The query router integrates all engines and routes queries based on parsed AST
type.

## Relational Operations

| Operation | Time |
| --- | --- |
| SELECT * (100 rows) | 17 us |
| SELECT WHERE | 17 us |
| INSERT | 290 us |
| UPDATE | 6.5 ms |

## Graph Operations

| Operation | Time |
| --- | --- |
| NODE CREATE | 2.3 us |
| EDGE CREATE | 3.5 us |
| NEIGHBORS | 1.8 us |
| PATH (1 -> 10) | 85 us |
| FIND NODE | 1.2 us |

## Vector Operations

| Operation | Time |
| --- | --- |
| EMBED STORE (128d) | 28 us |
| EMBED GET | 1.5 us |
| SIMILAR LIMIT 5 (100 vectors) | 10 ms |
| SIMILAR LIMIT 10 (100 vectors) | 10 ms |

## Mixed Workload

| Configuration | Time | Queries/s |
| --- | --- | --- |
| 5 mixed queries (SELECT, NEIGHBORS, SIMILAR, INSERT, NODE) | 11 ms | 455/s |

## Insert Throughput

| Batch Size | Time | Rows/s |
| --- | --- | --- |
| 100 | 29 ms | 3.4K/s |
| 500 | 145 ms | 3.4K/s |
| 1,000 | 290 ms | 3.4K/s |

## Analysis

- **Parse overhead**: Parser adds ~200ns-2us per query (negligible vs execution)
- **Routing overhead**: AST-based routing is O(1) pattern matching
- **Relational**: SELECT is fast (17us); UPDATE scans all rows (6.5ms for 100
  rows)
- **Graph**: Node/edge creation ~2-3us; path finding scales with path length
- **Vector**: Similarity search dominates mixed workloads (~10ms for 100
  vectors)
- **Bottleneck identification**: SIMILAR queries are the slowest operation; use
  HNSW index for large vector stores

## Query Routing Flow

```text
Query String
    │
    ▼
┌─────────┐
│ Parser  │  ~500ns
└────┬────┘
     │
     ▼
┌─────────┐
│   AST   │
└────┬────┘
     │
     ▼
┌─────────────┐
│   Router    │  O(1) dispatch
└──────┬──────┘
       │
       ├──► RelationalEngine
       ├──► GraphEngine
       ├──► VectorEngine
       ├──► Vault
       ├──► Cache
       └──► BlobStore
```

## Performance Recommendations

| Query Type | Optimization |
| --- | --- |
| High SELECT volume | Create hash indexes on filter columns |
| Large vector search | Build HNSW index |
| Graph traversals | Use NEIGHBORS with LIMIT |
| Batch inserts | Use batch_insert() API |
| Mixed workloads | Profile to identify bottlenecks |
