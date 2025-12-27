# Integration Tests

This document describes the integration test suite for the Neumann runtime.

## Overview

The integration test suite validates cross-engine functionality, data flow, and system behavior. All tests use a shared `TensorStore` to verify that relational, graph, and vector engines work correctly together.

**Test Count:** 250 tests across 21 files

## Test Helpers (`integration_tests/src/lib.rs`)

| Helper Function | Purpose |
|----------------|---------|
| `create_shared_router()` | Creates QueryRouter with shared TensorStore |
| `create_router_with_vault(master_key)` | Router with vault initialized |
| `create_router_with_cache()` | Router with cache initialized |
| `create_router_with_blob()` | Router with blob store initialized |
| `create_router_with_all_features(master_key)` | Router with vault, cache, and blob |
| `sample_embeddings(count, dim)` | Generates deterministic test embeddings using sin() |
| `get_store_from_router(router)` | Extracts TensorStore from router |
| `create_shared_engines()` | Creates (store, relational, graph, vector) tuple |
| `create_shared_engines_arc()` | Same as above but wrapped in Arc for concurrency |

---

## Persistence Tests (`tests/persistence.rs`) - 9 Tests

Tests snapshot/restore functionality across all engines.

| Test Name | What It Tests |
|-----------|---------------|
| `test_snapshot_preserves_all_data` | All engine data survives snapshot/restore |
| `test_snapshot_during_writes` | Concurrent writes don't corrupt snapshot |
| `test_restore_to_fresh_store` | Snapshot loads into new TensorStore |
| `test_data_survives_engine_restart` | Data persists when engines are recreated on same store |
| `test_compressed_snapshot_roundtrip` | Compression works for vector data |
| `test_snapshot_includes_vault_secrets` | Vault secrets persist in snapshot |
| `test_cache_entries_are_ephemeral` | Cache uses DashMap, not TensorStore |
| `test_snapshot_with_bloom_filter` | Bloom filter works with snapshots |
| `test_multiple_snapshots_incremental` | Multiple snapshots capture progressive state |

**Lessons Learned:**
- Cache is intentionally ephemeral (internal DashMaps)
- Vault secrets ARE persisted (encrypted in TensorStore)
- Bloom filter must be re-initialized with same parameters on restore

---

## Concurrency Tests (`tests/concurrency.rs`) - 10 Tests

Tests multi-threaded and async access patterns.

| Test Name | What It Tests |
|-----------|---------------|
| `test_concurrent_writes_all_engines` | 6 threads write to relational/graph/vector simultaneously |
| `test_shared_store_contention` | 4 threads write same key 1000 times each |
| `test_reader_writer_isolation` | Reads during heavy writes |
| `test_vault_concurrent_access_checks` | 4 threads read secrets concurrently |
| `test_cache_concurrent_lookups` | 4 threads lookup cached entries |
| `test_blob_parallel_uploads` | 10 concurrent blob uploads with barrier sync |
| `test_high_cardinality_inserts` | 100k entries across 4 threads |
| `test_deep_graph_traversal_concurrent` | 4 threads query 100-node chain path |
| `test_vector_search_during_index_build` | Search while adding new embeddings |
| `test_async_blob_concurrent_read_write` | Async readers and writers on blob store |

**Lessons Learned:**
- DashMap provides excellent concurrent write performance
- Node IDs are NOT guaranteed sequential - must capture actual IDs from create_node()
- Blob operations require `tokio::sync::Mutex` for shared access
- HNSW search is thread-safe during concurrent writes

---

## Cross-Engine Tests (`tests/cross_engine.rs`) - 10 Tests

Tests data flow and operations across multiple engines.

| Test Name | What It Tests |
|-----------|---------------|
| `test_unified_entity_across_engines` | Single entity with data in all 3 engines |
| `test_graph_nodes_with_embeddings` | Graph nodes linked to vector embeddings |
| `test_vault_with_graph_access_control` | Vault integrated with graph for access |
| `test_cache_with_relational_queries` | Cache stores relational query results |
| `test_insert_embed_search_cycle` | INSERT -> EMBED -> SIMILAR workflow |
| `test_node_edge_neighbor_path_cycle` | Full graph workflow with cycles |
| `test_table_with_embedded_vectors` | Products with embeddings for similarity |
| `test_blob_links_to_graph_entities` | Blob artifacts linked to graph nodes |
| `test_query_router_cross_engine_operations` | Router executes across all engines |
| `test_cross_engine_data_consistency` | Same entities in all engines |

**Lessons Learned:**
- `execute()` uses `col:type` syntax; `execute_parsed()` uses SQL syntax
- `NEIGHBORS` command returns `QueryResult::Ids`, not `QueryResult::Nodes`
- Node IDs must be captured and reused, not assumed to be 0, 1, 2...

---

## Error Handling Tests (`tests/error_handling.rs`) - 10 Tests

Tests error conditions and proper error messages.

| Test Name | What It Tests |
|-----------|---------------|
| `test_insert_into_nonexistent_table` | Returns proper error |
| `test_select_from_nonexistent_table` | Returns proper error |
| `test_delete_nonexistent_node` | Returns NodeNotFound error |
| `test_get_nonexistent_embedding` | Returns proper error |
| `test_vault_access_denied` | Non-ROOT entity blocked |
| `test_blob_get_nonexistent` | Returns NotFound error |
| `test_dimension_mismatch_on_search` | Returns error, not panic |
| `test_invalid_sql_syntax` | Parser error message quality |
| `test_type_mismatch_on_insert` | Schema validation error |
| `test_duplicate_table_create` | Returns TableExists error |

---

## Delete Consistency Tests (`tests/delete_consistency.rs`) - 7 Tests

Tests delete operations and cross-engine cleanup.

| Test Name | What It Tests |
|-----------|---------------|
| `test_delete_relational_row` | Row removed, verified |
| `test_delete_graph_node` | Node and edges cleaned up |
| `test_delete_vector_embedding` | Embedding removed from search |
| `test_delete_cross_engine_entity` | Entity removed from all engines |
| `test_delete_during_query` | Delete while SELECT running |
| `test_delete_node_with_edges` | Edges cleaned up properly |
| `test_cascade_delete_effects` | Related data consistency |

**Key APIs:**
- RelationalEngine: `delete_rows(table, condition)`
- VectorEngine: `delete_embedding(key)`
- GraphEngine: `delete_node(id)`

---

## Cache Invalidation Tests (`tests/cache_invalidation.rs`) - 7 Tests

Tests cache behavior on data modifications.

| Test Name | What It Tests |
|-----------|---------------|
| `test_cache_cleared_on_insert` | Cache behavior after INSERT |
| `test_cache_cleared_on_update` | Cached query invalidated by UPDATE |
| `test_cache_cleared_on_delete` | Cached query invalidated by DELETE |
| `test_cache_persists_without_writes` | Multiple SELECTs hit cache |
| `test_cache_cleared_on_graph_mutation` | Graph write behavior |
| `test_cache_cleared_on_vector_mutation` | Vector write behavior |
| `test_concurrent_write_cache_invalidation` | Race condition handling |

---

## FIND Command Tests (`tests/find_command.rs`) - 7 Tests

Tests the unified FIND command across engines.

| Test Name | What It Tests |
|-----------|---------------|
| `test_find_with_where_clause` | FIND with WHERE filtering |
| `test_find_with_similar_to` | FIND with SIMILAR TO clause |
| `test_find_with_connected_to` | FIND with CONNECTED TO clause |
| `test_find_combined_where_similar` | WHERE + SIMILAR in one query |
| `test_find_combined_all_clauses` | WHERE + SIMILAR + CONNECTED |
| `test_find_with_limit` | TOP/LIMIT enforcement |
| `test_find_empty_results` | No matches returns empty |

---

## Blob Lifecycle Tests (`tests/blob_lifecycle.rs`) - 7 Tests

Tests blob storage lifecycle operations.

| Test Name | What It Tests |
|-----------|---------------|
| `test_blob_gc_cleans_orphaned_chunks` | GC removes unreferenced chunks |
| `test_blob_gc_during_upload` | GC doesn't break active uploads |
| `test_blob_repair_corrupted_metadata` | Repair fixes broken references |
| `test_blob_verify_integrity` | Checksum verification works |
| `test_blob_streaming_read` | Read chunks without full load |
| `test_blob_chunk_deduplication` | Same content = shared chunks |
| `test_blob_graceful_shutdown` | Data persists after operations |

**Key APIs:**
- `BlobStore::gc()` returns `GcStats { deleted, freed_bytes }`
- `BlobReader::next_chunk()` for streaming reads

---

## Cache Advanced Tests (`tests/cache_advanced.rs`) - 6 Tests

Tests advanced cache features.

| Test Name | What It Tests |
|-----------|---------------|
| `test_cache_ttl_expiration` | Entries expire after TTL |
| `test_cache_semantic_similarity` | Semantic cache finds similar queries |
| `test_cache_embedding_lookup` | Embedding cache hit/miss |
| `test_cache_eviction_under_pressure` | LRU eviction when full |
| `test_cache_multi_layer_stats` | Stats accurate across layers |
| `test_cache_background_eviction` | Eviction task works |

**Key APIs:**
- `CacheConfig { embedding_dim, exact_capacity, default_ttl, ... }`
- `Cache::put(prompt, embedding, response, model, params_hash)`
- `Cache::put_embedding(source, content, embedding, model)`
- `CacheStats::size(CacheLayer)`, `total_entries()`

---

## Vault Advanced Tests (`tests/vault_advanced.rs`) - 8 Tests

Tests vault access control and features.

| Test Name | What It Tests |
|-----------|---------------|
| `test_vault_grant_access` | Grant allows access |
| `test_vault_revoke_access` | Revoke blocks access |
| `test_vault_grant_ttl_expiration` | TTL grant expires |
| `test_vault_permission_levels` | Read/Write/Admin hierarchy |
| `test_vault_audit_logging` | Access attempts logged |
| `test_vault_scoped_vault` | Entity-limited vault view |
| `test_vault_namespaced_vault` | Multi-tenant isolation |
| `test_vault_concurrent_grant_revoke` | Race condition handling |

**Key APIs:**
- `grant_with_ttl(requester, entity, key, permission, duration)`
- `vault.namespace(namespace, identity)` returns `NamespacedVault`
- `NamespacedVault::set(key, value)`, `get(key)`
- `vault.audit_recent(limit)`

---

## Edge Case Tests (`tests/edge_cases.rs`) - 10 Tests

Tests boundary conditions and special values.

| Test Name | What It Tests |
|-----------|---------------|
| `test_empty_table_operations` | SELECT/UPDATE/DELETE on empty |
| `test_empty_graph_traversal` | Path find with no nodes |
| `test_empty_vector_search` | Search with no embeddings |
| `test_zero_vector_handling` | Zero vector search/store |
| `test_max_int_boundary` | INT boundary values |
| `test_empty_string_handling` | Empty strings in conditions |
| `test_very_long_keys` | 1000+ char keys |
| `test_special_characters_in_keys` | Unicode, spaces, etc. |
| `test_null_value_conditions` | NULL in WHERE clauses |
| `test_self_loop_graph_handling` | Node connected to itself |

**Key APIs:**
- `Condition::Ne` (not `NotEq`)
- `update(table, condition, values)` - condition before values

---

## Tensor Compress Tests (`tests/tensor_compress.rs`) - 10 Tests

Tests compression algorithms with real engine data.

| Test Name | What It Tests |
|-----------|---------------|
| `test_quantize_int8_with_embeddings` | Int8 quantization roundtrip with embeddings |
| `test_quantize_binary_with_embeddings` | Binary quantization for 32x compression |
| `test_delta_encode_node_ids` | Delta encoding for sequential node IDs |
| `test_compress_ids_with_sparse_graph` | Delta+varint on sparse ID sequences |
| `test_rle_encode_status_column` | RLE on repeated column values |
| `test_rle_encode_large_runs` | RLE with 1000+ element runs |
| `test_compression_config_with_snapshot` | CompressionConfig serialization |
| `test_int8_quantization_preserves_similarity_order` | Similarity ranking preserved after quantization |
| `test_binary_quantization_for_fast_filtering` | Hamming distance on binary vectors |
| `test_compression_end_to_end` | All compression types across all engines |

**Key APIs:**
- `quantize_int8(vector)` -> `QuantizedInt8` (~4x compression)
- `quantize_binary(vector)` -> `QuantizedBinary` (~32x compression)
- `compress_ids(ids)` -> delta + varint encoded bytes
- `rle_encode(data)` -> `RleEncoded<T>` with run lengths

---

## Join Operations Tests (`tests/join_operations.rs`) - 10 Tests

Tests relational JOIN functionality.

| Test Name | What It Tests |
|-----------|---------------|
| `test_basic_inner_join` | Hash-based inner join on int keys |
| `test_join_empty_tables` | Join on empty tables returns empty |
| `test_join_no_matches` | Non-overlapping keys return empty |
| `test_join_one_to_many` | Department-employee join pattern |
| `test_join_many_to_many` | Student-enrollment junction table |
| `test_join_with_string_keys` | Join on string columns (SKUs) |
| `test_join_large_tables` | 1000x500 row join performance |
| `test_join_duplicate_keys_right` | Multiple right-side matches |
| `test_join_nonexistent_table` | Error on missing table |
| `test_join_preserves_all_columns` | All columns from both tables preserved |

**Key APIs:**
- `RelationalEngine::join(table_a, table_b, on_a, on_b)` -> `Vec<(Row, Row)>`

---

## HNSW Index Tests (`tests/hnsw_index.rs`) - 13 Tests

Tests Hierarchical Navigable Small World graph index.

| Test Name | What It Tests |
|-----------|---------------|
| `test_build_hnsw_index_from_engine` | Build index from VectorEngine embeddings |
| `test_hnsw_search_accuracy` | Exact match returns as top result |
| `test_hnsw_high_recall_config` | HNSWConfig::high_recall() accuracy |
| `test_hnsw_high_speed_config` | HNSWConfig::high_speed() still finds matches |
| `test_hnsw_vs_brute_force` | HNSW matches brute-force top result |
| `test_hnsw_empty_index` | Empty index returns empty results |
| `test_hnsw_single_element` | Single-element index works |
| `test_hnsw_duplicate_vectors` | Identical vectors all have score ~1.0 |
| `test_hnsw_large_dataset` | 1000 embeddings indexed correctly |
| `test_hnsw_different_dimensions` | Works for dims 8, 32, 64, 128, 256 |
| `test_hnsw_recall_at_k` | High recall config >= default recall |
| `test_hnsw_index_rebuild` | Rebuild includes newly added embeddings |
| `test_hnsw_concurrent_search` | 4 threads searching simultaneously |

**Key APIs:**
- `VectorEngine::build_hnsw_index(config)` -> `(HNSWIndex, Vec<String>)`
- `VectorEngine::search_with_hnsw(index, keys, query, k)` -> `Vec<SearchResult>`
- `HNSWConfig::default()`, `high_recall()`, `high_speed()`

---

## Vault Versioning Tests (`tests/vault_versioning.rs`) - 17 Tests

Tests secret versioning, rollback, and history management.

| Test Name | What It Tests |
|-----------|---------------|
| `test_version_increments_on_set` | Each set() increments version |
| `test_get_version_specific` | Retrieve specific version by number |
| `test_get_version_not_found` | Error on non-existent version |
| `test_list_versions` | List all versions with timestamps |
| `test_rollback_to_previous_version` | Rollback creates new version with old content |
| `test_max_versions_pruning` | Oldest versions pruned at limit |
| `test_rotate_keeps_versions` | rotate() preserves version history |
| `test_rotate_respects_max_versions` | rotate() also prunes old versions |
| `test_delete_removes_all_versions` | delete() removes all version blobs |
| `test_version_access_control` | Granted users can access versions |
| `test_multiple_secrets_versioned_independently` | Each secret has own version counter |
| `test_version_timestamps_monotonic` | Version timestamps increase |
| `test_rollback_multiple_times` | Multiple rollbacks work correctly |
| `test_concurrent_version_updates` | 4 threads updating same secret |
| `test_version_with_special_characters` | Unicode/special chars in values |
| `test_min_version_is_one` | max_versions=1 keeps at least 1 |
| `test_version_large_values` | 10KB values versioned correctly |

**Key APIs:**
- `vault.get_version(requester, key, version)` - Get specific version (1-based)
- `vault.list_versions(requester, key)` -> `Vec<VersionInfo>`
- `vault.current_version(requester, key)` -> current version number
- `vault.rollback(requester, key, version)` - Restore old version as new
- `VaultConfig::with_max_versions(n)` - Limit stored versions

---

## Index Operations Tests (`tests/index_operations.rs`) - 18 Tests

Tests hash index and B-tree index functionality.

| Test Name | What It Tests |
|-----------|---------------|
| `test_create_hash_index` | Create hash index on column |
| `test_create_btree_index` | Create B-tree index for range queries |
| `test_index_on_id_column` | Index on internal _id column |
| `test_drop_hash_index` | Remove hash index |
| `test_drop_btree_index` | Remove B-tree index |
| `test_get_indexed_columns` | List all indexed columns |
| `test_get_btree_indexed_columns` | List B-tree indexed columns |
| `test_index_duplicate_error` | Error on duplicate index |
| `test_btree_index_duplicate_error` | Error on duplicate B-tree |
| `test_index_nonexistent_column_error` | Error on missing column |
| `test_drop_nonexistent_index_error` | Error dropping missing index |
| `test_index_with_data_insert` | Index updated on insert |
| `test_btree_index_with_range_data` | B-tree accelerates range queries |
| `test_index_survives_updates` | Index updated on UPDATE |
| `test_index_survives_deletes` | Index updated on DELETE |
| `test_multiple_indexes_same_table` | Multiple indexes coexist |
| `test_index_with_null_values` | Nullable columns indexed |
| `test_index_large_table` | 10k row index performance |

**Key APIs:**
- `create_index(table, column)` - Hash index for equality
- `create_btree_index(table, column)` - B-tree for range queries
- `drop_index()` / `drop_btree_index()` - Remove indexes
- `has_index()` / `has_btree_index()` - Check existence
- `get_indexed_columns()` / `get_btree_indexed_columns()` - List indexes

---

## Columnar Storage Tests (`tests/columnar_storage.rs`) - 20 Tests

Tests columnar data materialization, batch operations, and projection.

| Test Name | What It Tests |
|-----------|---------------|
| `test_materialize_single_column` | Convert column to columnar format |
| `test_materialize_multiple_columns` | Materialize multiple columns |
| `test_load_column_data` | Load materialized column data |
| `test_drop_columnar_data` | Remove columnar data |
| `test_select_columnar_with_filter` | SIMD-accelerated filtering |
| `test_select_columnar_fallback` | Fallback to row-based |
| `test_select_with_projection` | Column projection |
| `test_select_with_projection_includes_id` | Project _id column |
| `test_batch_insert` | Bulk insert 100 rows |
| `test_batch_insert_empty` | Empty batch returns empty |
| `test_batch_insert_validates_schema` | Schema validation on batch |
| `test_batch_insert_large` | 10k row batch insert |
| `test_columnar_with_nulls` | Null tracking in columnar |
| `test_columnar_scan_vs_row_scan` | Columnar vs row results match |
| `test_columnar_with_string_column` | String columns in columnar |
| `test_columnar_scan_with_projection` | Columnar + projection |
| `test_row_count` | Count rows in table |
| `test_table_exists` | Check table existence |
| `test_drop_table` | Remove table and cleanup |
| `test_list_tables` | List all tables |

**Key APIs:**
- `materialize_columns(table, columns)` - Convert to columnar
- `load_column_data(table, column)` -> `ColumnData`
- `select_columnar(table, condition, options)` - SIMD scan
- `select_with_projection(table, condition, projection)` - Column selection
- `batch_insert(table, rows)` - Bulk insert
- `ColumnarScanOptions { projection, prefer_columnar }`

---

## Entity Graph API Tests (`tests/entity_graph_api.rs`) - 18 Tests

Tests string-keyed entity operations for graph connectivity.

| Test Name | What It Tests |
|-----------|---------------|
| `test_add_entity_edge_basic` | Add directed edge between entities |
| `test_add_entity_edge_undirected` | Add undirected edge (both directions) |
| `test_get_entity_neighbors_out` | Outgoing neighbor lookup |
| `test_get_entity_neighbors_in` | Incoming neighbor lookup |
| `test_get_entity_neighbors_both_directions` | All neighbors (in + out) |
| `test_delete_entity_edge` | Remove edge by key |
| `test_delete_entity_edge_nonexistent` | Error on missing edge |
| `test_multiple_edge_types_same_entities` | Multiple edge types between nodes |
| `test_entity_graph_traversal` | Multi-hop path traversal |
| `test_entity_self_loop` | Self-referencing edge |
| `test_concurrent_entity_edge_operations` | 4 threads adding edges |
| `test_entity_edge_delete_and_recreate` | Delete then re-add edge |
| `test_entity_has_edges` | Check if entity has connections |
| `test_entity_bidirectional_edges` | A->B and B->A edges |
| `test_entity_neighbors_empty_entity` | Entity with only incoming edges |
| `test_entity_star_topology` | Hub with many spokes |
| `test_entity_ring_topology` | Circular chain of nodes |
| `test_entity_delete_preserves_other_edges` | Delete one edge, keep others |

**Key APIs:**
- `add_entity_edge(from, to, edge_type)` - Create directed edge
- `add_entity_edge_undirected(key1, key2, type)` - Create undirected edge
- `get_entity_neighbors_out(key)` / `get_entity_neighbors_in(key)` - Direction-specific
- `get_entity_neighbors(key)` - All neighbors (both directions)
- `delete_entity_edge(edge_key)` - Remove edge
- `entity_has_edges(key)` - Check for connections

---

## Sparse Vector Tests (`tests/sparse_vectors.rs`) - 22 Tests

Tests sparse vector creation, storage, and similarity operations.

| Test Name | What It Tests |
|-----------|---------------|
| `test_from_embedding_basic` | Create sparse from dense vector |
| `test_from_embedding_becomes_sparse` | High sparsity triggers Sparse variant |
| `test_from_embedding_auto` | Auto-threshold sparse creation |
| `test_from_embedding_all_zeros` | Zero vector handling |
| `test_from_embedding_all_significant` | All values above threshold |
| `test_dot_product_dense_dense` | Dense-dense dot product |
| `test_dot_product_sparse_sparse` | Sparse-sparse dot product |
| `test_dot_product_orthogonal` | Orthogonal vectors = 0 |
| `test_dot_product_parallel` | Parallel vectors = product |
| `test_cosine_similarity_identical` | Same vectors = 1.0 |
| `test_cosine_similarity_opposite` | Opposite vectors = -1.0 |
| `test_cosine_similarity_orthogonal` | Orthogonal = 0 |
| `test_cosine_similarity_scaled` | Scaled vectors = 1.0 |
| `test_store_and_retrieve_sparse_vector` | Store/retrieve roundtrip |
| `test_sparse_vector_in_similarity_search` | Search with sparse embeddings |
| `test_sparse_high_dimensional` | 1024-dim sparse vectors |
| `test_sparse_vs_dense_dot_product_equivalence` | Same result dense vs sparse |
| `test_sparse_vector_concurrent_operations` | 4 threads storing vectors |
| `test_dot_product_dimension_mismatch` | Dimension mismatch returns None |
| `test_sparse_threshold_effects` | Value threshold impact |
| `test_sparse_with_negative_values` | Negative value handling |
| `test_dimension_method` | dimension() on all variants |

**Key APIs:**
- `TensorValue::from_embedding(dense, value_threshold, sparsity_threshold)` - Convert dense to sparse
- `TensorValue::from_embedding_auto(dense)` - Auto thresholds (0.01 value, 0.7 sparsity)
- `TensorValue::dot(other)` - Dot product (sparse-sparse, sparse-dense, dense-dense)
- `TensorValue::cosine_similarity(other)` - Cosine similarity
- `TensorValue::to_dense()` - Convert back to dense
- `TensorValue::dimension()` - Get vector dimension

---

## Store Instrumentation Tests (`tests/store_instrumentation.rs`) - 15 Tests

Tests access pattern tracking and shard instrumentation.

| Test Name | What It Tests |
|-----------|---------------|
| `test_with_instrumentation_basic` | Create instrumented store |
| `test_instrumentation_disabled_by_default` | Default store has no instrumentation |
| `test_hot_shards_tracking` | Identify frequently accessed shards |
| `test_access_distribution` | Track reads/writes per shard |
| `test_sampling_rate_effect` | Sampling rate reduces tracking overhead |
| `test_concurrent_instrumentation` | Thread-safe access counting |
| `test_bloom_and_instrumentation_combined` | Both features together |
| `test_instrumentation_snapshot_isolation` | Snapshot reflects current state |
| `test_cold_shards_detection` | Identify rarely used shards |
| `test_instrumentation_with_deletes` | Deletes count as writes |
| `test_instrumentation_read_write_ratio` | Read-heavy vs write-heavy detection |
| `test_instrumentation_empty_store` | Empty store has zero counts |
| `test_shard_distribution_fairness` | Hash distribution is uniform |
| `test_hot_shards_limit` | hot_shards() respects limit param |
| `test_instrumentation_with_shared_store` | Track access across engines |

**Key APIs:**
- `TensorStore::with_instrumentation(sample_rate)` - Create instrumented store
- `TensorStore::with_bloom_and_instrumentation(items, fpr, sample_rate)` - Both features
- `has_instrumentation()` - Check if enabled
- `access_snapshot()` -> `Option<ShardAccessSnapshot>` - Get point-in-time stats
- `hot_shards(limit)` -> `Option<Vec<(shard_id, count)>>` - Top accessed shards
- `ShardAccessSnapshot.total_reads()` / `total_writes()` - Aggregate counts
- `ShardAccessSnapshot.shard_stats` - Per-shard read/write counts

---

## Tiered Storage Tests (`tests/tiered_storage.rs`) - 16 Tests

Tests hot/cold data migration and access pattern optimization.

| Test Name | What It Tests |
|-----------|---------------|
| `test_tiered_store_hot_only_mode` | Hot-only store without cold tier |
| `test_tiered_store_with_cold_storage` | Store with cold directory configured |
| `test_migrate_cold` | Migrate stale data to cold storage |
| `test_cold_data_promotion` | Access cold data promotes to hot |
| `test_preload_specific_keys` | Preload cold keys to hot tier |
| `test_hot_shards_tracking` | Identify frequently accessed shards |
| `test_cold_shards_tracking` | Identify rarely accessed shards |
| `test_delete_from_both_tiers` | Delete works across hot and cold |
| `test_exists_across_tiers` | Existence check across tiers |
| `test_tiered_stats` | Migration and access statistics |
| `test_flush_cold_storage` | Flush cold tier to disk |
| `test_into_tensor_store` | Convert TieredStore to TensorStore |
| `test_concurrent_tiered_access` | 4 threads with shared mutex |
| `test_tiered_large_tensors` | Large embedding vectors |
| `test_tiered_empty_operations` | Empty store edge cases |
| `test_tiered_overwrite` | Overwrite existing key |

**Key APIs:**
- `TieredStore::new(config)` - Create with cold storage
- `TieredStore::hot_only(sample_rate)` - Hot-only mode
- `TieredConfig { cold_dir, cold_capacity, sample_rate }`
- `migrate_cold(threshold_ms)` - Move stale data to cold
- `preload(keys)` - Load cold keys to hot tier
- `hot_shards(limit)` / `cold_shards(threshold_ms)` - Shard access patterns
- `stats()` -> `TieredStats { hot_count, cold_count, hot_lookups, migrations_to_cold }`
- `into_tensor_store()` - Convert with cold data loaded

---

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

## Coverage

Integration tests cover:
- **Persistence:** 9 tests - snapshot/restore across all engines
- **Concurrency:** 10 tests - multi-threaded and async operations
- **Cross-Engine:** 10 tests - data flow between engines
- **Error Handling:** 10 tests - proper error messages
- **Delete Operations:** 7 tests - cleanup and consistency
- **Cache Invalidation:** 7 tests - cache behavior on writes
- **FIND Command:** 7 tests - unified query syntax
- **Blob Lifecycle:** 7 tests - GC, repair, streaming
- **Cache Advanced:** 6 tests - TTL, semantic, eviction
- **Vault Advanced:** 8 tests - grants, audit, namespacing
- **Edge Cases:** 10 tests - boundary conditions
- **Tensor Compress:** 10 tests - quantization, delta, RLE encoding
- **Join Operations:** 10 tests - hash-based relational JOINs
- **HNSW Index:** 13 tests - approximate nearest neighbor search
- **Vault Versioning:** 17 tests - secret history and rollback
- **Index Operations:** 18 tests - hash and B-tree indexes
- **Columnar Storage:** 20 tests - columnar scans, batch insert, projection
- **Entity Graph API:** 18 tests - string-keyed entity edge operations
- **Sparse Vectors:** 22 tests - sparse vector creation and similarity
- **Store Instrumentation:** 15 tests - access pattern tracking
- **Tiered Storage:** 16 tests - hot/cold data migration

**Total: 250 integration tests**
