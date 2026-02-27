# Integration Tests

The integration test suite validates cross-engine functionality, data flow, and
system behavior. All tests use a shared `TensorStore` to verify that relational,
graph, and vector engines work correctly together.

**Test Count:** 267+ tests across 22 files

## Running Tests

```bash
# Run all integration tests
cargo test --package integration_tests

# Run specific test file
cargo test --package integration_tests --test persistence

# Run single test
cargo test --package integration_tests test_snapshot_preserves_all_data

# Run with output
cargo test --package integration_tests -- --nocapture
```

## Test Categories

| Category | Tests | Description |
| --- | --- | --- |
| Persistence | 9 | Snapshot/restore across all engines |
| Concurrency | 10 | Multi-threaded and async operations |
| Cross-Engine | 10 | Data flow between engines |
| Error Handling | 10 | Proper error messages |
| Delete Operations | 7 | Cleanup and consistency |
| Cache Invalidation | 7 | Cache behavior on writes |
| FIND Command | 7 | Unified query syntax |
| Blob Lifecycle | 7 | GC, repair, streaming |
| Cache Advanced | 6 | TTL, semantic, eviction |
| Vault Advanced | 8 | Grants, audit, namespacing |
| Edge Cases | 10 | Boundary conditions |
| Tensor Compress | 10 | Quantization, delta, RLE encoding |
| Join Operations | 10 | Hash-based relational JOINs |
| HNSW Index | 13 | Approximate nearest neighbor search |
| Vault Versioning | 17 | Secret history and rollback |
| Index Operations | 18 | Hash and B-tree indexes |
| Columnar Storage | 20 | Columnar scans, batch insert, projection |
| Entity Graph API | 18 | String-keyed entity edge operations |
| Sparse Vectors | 22 | Sparse vector creation and similarity |
| Store Instrumentation | 15 | Access pattern tracking |
| Tiered Storage | 16 | Hot/cold data migration |
| Distance Metrics | 17 | COSINE, EUCLIDEAN, DOT_PRODUCT similarity |

## Test Helpers

Available in `integration_tests/src/lib.rs`:

| Helper Function | Purpose |
| --- | --- |
| `create_shared_router()` | Creates QueryRouter with shared TensorStore |
| `create_router_with_vault(master_key)` | Router with vault initialized |
| `create_router_with_cache()` | Router with cache initialized |
| `create_router_with_blob()` | Router with blob store initialized |
| `create_router_with_all_features(master_key)` | Router with vault, cache, and blob |
| `sample_embeddings(count, dim)` | Generates deterministic test embeddings using sin() |
| `get_store_from_router(router)` | Extracts TensorStore from router |
| `create_shared_engines()` | Creates (store, relational, graph, vector) tuple |
| `create_shared_engines_arc()` | Same as above but wrapped in Arc for concurrency |

## Key Test Suites

### Persistence Tests

Tests snapshot/restore functionality across all engines.

| Test | What It Tests |
| --- | --- |
| `test_snapshot_preserves_all_data` | All engine data survives snapshot/restore |
| `test_snapshot_during_writes` | Concurrent writes don't corrupt snapshot |
| `test_restore_to_fresh_store` | Snapshot loads into new TensorStore |
| `test_compressed_snapshot_roundtrip` | Compression works for vector data |
| `test_snapshot_includes_vault_secrets` | Vault secrets persist in snapshot |

#### Lessons Learned

- Cache is intentionally ephemeral (internal DashMaps)
- Vault secrets ARE persisted (encrypted in TensorStore)
- Bloom filter must be re-initialized with same parameters on restore

### Concurrency Tests

Tests multi-threaded and async access patterns.

| Test | What It Tests |
| --- | --- |
| `test_concurrent_writes_all_engines` | 6 threads write to relational/graph/vector simultaneously |
| `test_shared_store_contention` | 4 threads write same key 1000 times each |
| `test_reader_writer_isolation` | Reads during heavy writes |
| `test_blob_parallel_uploads` | 10 concurrent blob uploads with barrier sync |

#### Lessons Learned

- DashMap provides excellent concurrent write performance
- Node IDs are NOT guaranteed sequential - must capture actual IDs
- Blob operations require `tokio::sync::Mutex` for shared access
- HNSW search is thread-safe during concurrent writes

### Cross-Engine Tests

Tests data flow and operations across multiple engines.

| Test | What It Tests |
| --- | --- |
| `test_unified_entity_across_engines` | Single entity with data in all 3 engines |
| `test_graph_nodes_with_embeddings` | Graph nodes linked to vector embeddings |
| `test_insert_embed_search_cycle` | INSERT -> EMBED -> SIMILAR workflow |
| `test_query_router_cross_engine_operations` | Router executes across all engines |

#### Lessons Learned

- `execute()` uses `col:type` syntax; `execute_parsed()` uses SQL syntax
- `NEIGHBORS` command returns `QueryResult::Ids`, not `QueryResult::Nodes`
- Node IDs must be captured and reused, not assumed to be 0, 1, 2...

### Sparse Vector Tests (22 tests)

Tests sparse vector creation, storage, and similarity operations.

#### Key APIs

- `TensorValue::from_embedding(dense, value_threshold, sparsity_threshold)`
- `TensorValue::from_embedding_auto(dense)` - Auto thresholds (0.01 value, 0.7
  sparsity)
- `TensorValue::dot(other)` - Dot product (sparse-sparse, sparse-dense,
  dense-dense)
- `TensorValue::cosine_similarity(other)` - Cosine similarity
- `TensorValue::to_dense()` - Convert back to dense
- `TensorValue::dimension()` - Get vector dimension

### Distance Metrics Tests (17 tests)

Tests SIMILAR queries with different distance metrics.

#### Key Syntax

```sql
-- Metric goes AFTER LIMIT clause
SIMILAR 'key' LIMIT 10 EUCLIDEAN
SIMILAR [0.1, 0.2] LIMIT 5 DOT_PRODUCT
```

#### Known Issues

- Metric keyword must be AFTER LIMIT (not `METRIC EUCLIDEAN`)
- COSINE/DOT_PRODUCT return empty for zero-magnitude queries
- EUCLIDEAN correctly handles zero vectors

## Coverage Summary

| Category | Files | Tests | Key Validations |
| --- | --- | --- | --- |
| Storage | 4 | 50+ | Persistence, tiering, instrumentation |
| Engines | 5 | 60+ | Relational, graph, vector operations |
| Security | 2 | 25+ | Vault, access control, versioning |
| Caching | 2 | 13 | Exact, semantic, invalidation |
| Advanced | 6 | 80+ | Compression, joins, indexes, sparse |
| **Total** | **17** | **267+** | |
