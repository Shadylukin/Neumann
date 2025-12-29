# Tensor Store

The foundational storage layer for Neumann. Holds all data in a unified tensor structure. Knows nothing about queries - just stores and retrieves.

## Design Principles

1. **Single Responsibility**: Store and retrieve tensors by key. No query logic.
2. **Concurrent by Design**: Uses SlabRouter with specialized tensor slabs for zero resize stalls.
3. **Shareable Storage**: TensorStore clones share the same underlying data via Arc.
4. **Pure Tensor Architecture**: All storage uses tensor-based structures (BTreeMap, sorted arrays) - no hash tables that resize.
5. **Predictable Performance**: No throughput stalls from hash table resizing.

## Architecture

TensorStore uses SlabRouter internally, which routes operations to specialized slabs:

```
TensorStore
  |
  +-- Arc<SlabRouter>
         |
         +-- MetadataSlab (general key-value, BTreeMap-based)
         +-- EntityIndex (sorted vocabulary + hash index)
         +-- EmbeddingSlab (dense f32 arrays)
         +-- GraphTensor (CSR format for edges)
         +-- RelationalSlab (columnar storage)
         +-- CacheRing (LRU/LFU eviction)
         +-- BlobLog (append-only blob storage)
```

### Key Routing

Operations are routed based on key prefixes:
- `emb:*` -> EmbeddingSlab (embedding vectors)
- `node:*`, `edge:*` -> GraphTensor (graph data)
- `table:*` -> RelationalSlab (relational rows)
- `_cache:*` -> CacheRing (cached data)
- Everything else -> MetadataSlab (general metadata)

## Concurrency Model

The store uses tensor-based structures instead of hash maps:

- **No Resize Stalls**: BTreeMap and sorted arrays grow incrementally, never causing 99%+ throughput drops
- **Lock-free Reads**: RwLock allows many concurrent readers
- **Predictable Writes**: O(log n) inserts, no amortized O(n) resizing
- **Clone on Read**: `get()` returns cloned data to avoid holding references

### Performance

| Metric | SlabRouter | Previous (DashMap) |
|--------|------------|-------------------|
| PUT throughput | 3.1+ M ops/sec | 2.5 M ops/sec |
| GET throughput | 4.9+ M ops/sec | 4.5 M ops/sec |
| Throughput variance (CV) | 12% steady-state | 222% during resize |
| Resize stalls | None | 99.6% throughput drops |

### Stress Test Results

Stress tests validate stability under sustained concurrent load:

| Test | Throughput | CV | Duration |
|------|------------|-----|----------|
| SlabRouter Stability | 653K ops/sec | 12.0% | 60s |
| Mixed Workload | 802K ops/sec | - | 16s |
| Cache-Heavy | 144K ops/sec | - | 8s |
| Embedding-Heavy | 128K ops/sec | - | 8s |

**Note on CV variance**: The duration stress test (which grows the store continuously over 60s) shows higher CV (~50%) as throughput naturally decreases when the store grows from empty to millions of entries. This is expected behavior - BTreeMap operations are O(log n) so performance scales with size. The key metric is that there are no sudden stalls or throughput cliffs, unlike hash table resizing which causes abrupt 99%+ drops.

The SlabRouter stability test maintains a steady working set size, achieving 12% CV which is well within the 20% target for production stability.

## Data Model

### TensorData

An entity that can hold any combination of:

| Field Type | Rust Type | Use Case |
|------------|-----------|----------|
| Scalar | `ScalarValue` | Properties (name, age, active) |
| Vector | `Vec<f32>` | Embeddings for similarity search |
| Pointer | `String` | Single relationship to another tensor |
| Pointers | `Vec<String>` | Multiple relationships |

### ScalarValue

| Variant | Rust Type | Example |
|---------|-----------|---------|
| `Null` | - | Missing/undefined value |
| `Bool` | `bool` | `true`, `false` |
| `Int` | `i64` | `42`, `-1` |
| `Float` | `f64` | `3.14159` |
| `String` | `String` | `"Alice"` |
| `Bytes` | `Vec<u8>` | Raw binary data |

### Key Format Convention

Keys follow the pattern `type:id`:

```
user:1
user:2
post:42
comment:100
```

This enables efficient prefix scanning: `scan("user:")` returns all user keys.

### Reserved Field Names

For unified entity support, certain field names are reserved:

| Field | Purpose | Used By |
|-------|---------|---------|
| `_out` | Outgoing graph edge pointers | GraphEngine |
| `_in` | Incoming graph edge pointers | GraphEngine |
| `_embedding` | Vector embedding | VectorEngine |
| `_label` | Entity type/label | GraphEngine |
| `_type` | Discriminator field | All engines |
| `_from` | Edge source | GraphEngine |
| `_to` | Edge target | GraphEngine |
| `_edge_type` | Edge relationship type | GraphEngine |
| `_directed` | Edge direction flag | GraphEngine |
| `_table` | Table membership | RelationalEngine |

These are defined in `tensor_store::fields` and should not be used for application data.

## API Reference

### TensorStore

```rust
// Construction
let store = TensorStore::new();
let store = TensorStore::with_capacity(10_000);  // Pre-allocate (hint only)

// Core operations (infallible except get/delete)
store.put("user:1", tensor)?;      // Store tensor under key
store.get("user:1")?;              // Retrieve tensor (cloned), returns Result
store.delete("user:1")?;           // Remove tensor, returns Result
store.exists("user:1");            // Check if key exists (bool)

// Scanning (infallible)
store.scan("user:");               // List keys with prefix -> Vec<String>
store.scan_count("user:");         // Count keys with prefix -> usize

// Metadata (infallible)
store.len();                       // Total tensor count -> usize
store.is_empty();                  // Check if store is empty -> bool
store.clear();                     // Remove all tensors

// Persistence (returns Result<_, SnapshotError>)
store.save_snapshot("data.bin")?;  // Save to file (atomic)
TensorStore::load_snapshot("data.bin")?;  // Load from file
TensorStore::load_snapshot_with_bloom_filter("data.bin", 10000, 0.01)?;  // Load with bloom filter
```

### TensorData

```rust
let mut tensor = TensorData::new();

// Setting values
tensor.set("name", TensorValue::Scalar(ScalarValue::String("Alice".into())));
tensor.set("embedding", TensorValue::Vector(vec![0.1, 0.2, 0.3]));
tensor.set("friend", TensorValue::Pointer("user:2".into()));

// Reading values
tensor.get("name");                // Option<&TensorValue>
tensor.has("name");                // bool
tensor.keys();                     // Iterator over field names

// Modification
tensor.remove("name");             // Option<TensorValue>
tensor.len();                      // Number of fields
tensor.is_empty();                 // bool
```

### SlabRouter (Advanced)

Direct access to the slab router for specialized use cases:

```rust
use tensor_store::SlabRouter;

let router = SlabRouter::new();

// Same API as TensorStore
router.put("key", tensor)?;
router.get("key")?;
router.scan("prefix:");

// Persistence
router.save_to_file("data.bin")?;
let router = SlabRouter::load_from_file("data.bin")?;

// In-memory serialization
let bytes = router.to_bytes()?;
let router = SlabRouter::from_bytes(&bytes)?;
```

## Error Handling

### TensorStoreError

Only `get` and `delete` can fail:

| Error | Cause |
|-------|-------|
| `NotFound(key)` | `get` or `delete` on nonexistent key |

### SnapshotError

Snapshot operations can fail:

| Error | Cause |
|-------|-------|
| `IoError(std::io::Error)` | File not found, permission denied, disk full |
| `SerializationError(String)` | Corrupted file, incompatible format |

Both error types implement `std::error::Error` for use with `?` operator.

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| `put` | O(log n) | BTreeMap insert |
| `get` | O(log n) + clone cost | Clone prevents reference issues |
| `delete` | O(log n) | BTreeMap remove |
| `exists` | O(log n) | BTreeMap lookup |
| `scan` | O(k + log n) | BTreeMap range, k = result count |
| `scan_count` | O(k + log n) | No allocation |
| `len` | O(1) | Cached count |
| `clear` | O(n) | Clears all data |

### Why Not Hash Maps?

Hash maps (including concurrent ones like DashMap) suffer from resize stalls:

| Metric | Hash Map | BTreeMap/Sorted Arrays |
|--------|----------|------------------------|
| Insert | O(1) amortized | O(log n) |
| Resize | O(n) all at once | Incremental |
| Throughput stability | Spikes during resize | Consistent |
| Memory fragmentation | Higher | Lower |

For a database runtime, consistent performance is more important than slightly faster average case.

## Memory Layout

```
TensorStore
  |
  +-- Arc<SlabRouter>
              |
              +-- MetadataSlab
              |     +-- RwLock<BTreeMap<String, TensorData>>
              |
              +-- EntityIndex
              |     +-- Vec<String> (vocabulary)
              |     +-- Vec<(u64, u32)> (sorted hash -> slot)
              |
              +-- EmbeddingSlab
              |     +-- Vec<f32> (dense embeddings)
              |     +-- BTreeMap<EntityId, slot>
              |
              +-- GraphTensor (CSR format)
              |     +-- Vec<u64> (row pointers)
              |     +-- Vec<u32> (column indices)
              |
              +-- CacheRing
              |     +-- Ring buffer with LRU/LFU eviction
              |
              +-- BlobLog
                    +-- Append-only segments
```

## Usage Examples

### Storing a User Entity

```rust
let store = TensorStore::new();

let mut user = TensorData::new();
user.set("name", TensorValue::Scalar(ScalarValue::String("Alice".into())));
user.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
user.set("active", TensorValue::Scalar(ScalarValue::Bool(true)));
user.set("embedding", TensorValue::Vector(vec![0.1, 0.2, 0.3, 0.4]));
user.set("friends", TensorValue::Pointers(vec![
    "user:2".into(),
    "user:3".into(),
]));

store.put("user:1", user)?;
```

### Querying by Prefix

```rust
// Get all user keys
let user_keys = store.scan("user:");

// Count posts without allocating
let post_count = store.scan_count("post:");
```

### High-Concurrency Writes

```rust
use std::sync::Arc;
use std::thread;

let store = Arc::new(TensorStore::new());
let mut handles = vec![];

// 8 threads writing to overlapping key space
for t in 0..8 {
    let store = Arc::clone(&store);
    handles.push(thread::spawn(move || {
        for i in 0..1000 {
            let mut tensor = TensorData::new();
            tensor.set("thread", TensorValue::Scalar(ScalarValue::Int(t)));
            store.put(format!("key:{}", i), tensor).unwrap();
        }
    }));
}

for h in handles {
    h.join().unwrap();
}
// SlabRouter handles contention without resize stalls
```

### Shared Storage Across Engines

TensorStore clones share the same underlying data, enabling unified entity storage:

```rust
let store = TensorStore::new();

// Clone shares the same underlying Arc<SlabRouter>
let store_clone = store.clone();

// Writes via one clone are visible to the other
store.put("user:1", user_data)?;
assert!(store_clone.exists("user:1"));

// Use with engines for cross-engine queries
let vector_engine = VectorEngine::with_store(store.clone());
let graph_engine = GraphEngine::with_store(store.clone());

// Both engines operate on the same entity storage
vector_engine.set_entity_embedding("user:1", vec![0.1, 0.2, 0.3])?;
graph_engine.add_entity_edge("user:1", "user:2", "follows")?;
```

## Test Coverage

| Test | What It Verifies |
|------|------------------|
| `tensor_data_stores_scalars` | All scalar types stored correctly |
| `tensor_data_stores_vectors` | Vector embeddings stored correctly |
| `tensor_data_stores_pointers` | Single and multiple pointers work |
| `tensor_data_stores_bytes` | Binary data stored correctly |
| `tensor_data_remove_field` | Field removal works |
| `tensor_data_overwrite_field` | Overwriting replaces value |
| `store_put_get` | Basic store/retrieve cycle |
| `store_get_not_found` | NotFound error on missing key |
| `store_delete` | Deletion removes key |
| `store_delete_not_found` | NotFound error on missing delete |
| `store_exists` | Existence check works |
| `store_overwrite` | Overwriting key replaces tensor |
| `store_scan_*` | Prefix scanning works correctly |
| `store_scan_count` | Count without allocation |
| `store_clear` | Clear removes all data |
| `store_with_capacity` | Pre-allocation works |
| `store_10k_entities` | Handles 10,000 entities |
| `store_concurrent_writes` | Thread safety for separate keys |
| `store_concurrent_writes_same_keys` | Thread safety under contention |
| `store_concurrent_read_write` | Mixed read/write safety |
| `store_empty_key` | Empty string as key works |
| `store_unicode_keys` | Unicode in keys works |
| `snapshot_save_and_load` | Basic snapshot round-trip |
| `snapshot_empty_store` | Empty store snapshot works |
| `snapshot_all_scalar_types` | All scalar types serialize correctly |
| `snapshot_pointers` | Pointer types serialize correctly |
| `snapshot_large_dataset` | 1000 entities with embeddings |
| `snapshot_with_bloom_filter` | Load with bloom filter rebuild |
| `snapshot_load_nonexistent_file` | Error handling for missing file |
| `snapshot_error_*` | Error type coverage |
| `snapshot_compressed_roundtrip` | Compressed snapshot save/load |
| `snapshot_compressed_with_quantization` | Int8 quantization works |
| `snapshot_compressed_empty_store` | Empty store compression works |

## Architectural Decision: Pure Tensor Storage

**Why SlabRouter over DashMap?**

| Aspect | DashMap | SlabRouter |
|--------|---------|------------|
| Throughput | 2.5 M ops/sec | 3.1+ M ops/sec |
| Stability | 222% CV during resize | 12% CV steady-state |
| Resize stalls | 99.6% throughput drops | None |
| Growth | O(n) resize events | O(log n) incremental |
| Memory | Hash table overhead | Compact tree/array |

For a database runtime where consistent performance is critical, tensor-based structures (BTreeMap, sorted arrays) provide predictable throughput without the pathological resize behavior of hash tables.

**Observed behavior under load**:
- Steady-state workloads (fixed working set): 12% CV - excellent stability
- Growing workloads (continuous inserts): ~50% CV - expected O(log n) slowdown as store grows, but no sudden stalls
- The key difference from hash tables: performance degrades gradually and predictably, never cliff-diving during resize

## Serialization

All core types implement `serde::Serialize` and `serde::Deserialize`:

| Type | Serializable |
|------|--------------|
| `TensorData` | Yes |
| `TensorValue` | Yes |
| `ScalarValue` | Yes |
| `TensorStoreError` | Yes |

This enables:
- Snapshot persistence (built-in)
- Custom serialization formats (JSON, MessagePack, etc.)
- Network transfer of tensor data
- Integration with other systems

## Persistence

### Snapshot API

Save and load the entire store atomically:

```rust
// Save entire store to a file
store.save_snapshot("data.bin")?;

// Load from a snapshot file
let store = TensorStore::load_snapshot("data.bin")?;

// Load with Bloom filter (rebuilds filter from keys)
let store = TensorStore::load_snapshot_with_bloom_filter(
    "data.bin",
    10_000,   // expected items
    0.01      // false positive rate
)?;

// Save with compression (see tensor_compress crate)
use tensor_compress::{CompressionConfig, QuantMode};
let config = CompressionConfig {
    vector_quantization: Some(QuantMode::Int8),  // 4x compression
    delta_encoding: true,                        // For sorted IDs
    rle_encoding: true,                          // For repeated values
};
store.save_snapshot_compressed("data.bin", config)?;

// Load compressed snapshot (auto-detects format)
let store = TensorStore::load_snapshot_compressed("data.bin")?;
```

### Snapshot Versions

| Version | Format | Backward Compatible |
|---------|--------|---------------------|
| v2 | HashMap-based | Read-only support |
| v3 | SlabRouter-based | Current format |

Snapshots auto-detect format on load for backward compatibility.

### Implementation Details

| Feature | Description |
|---------|-------------|
| Format | bincode (compact binary) |
| Compressed Format | Magic bytes "NEUM" + version header + compressed data |
| Atomicity | Writes to `.tmp` file, then atomic rename |
| Bloom Filter | Rebuilt on load if requested |
| HNSW Index | Not persisted (rebuild required) |

### Compression Options

| Technique | Compression | Lossless | Best For |
|-----------|-------------|----------|----------|
| Int8 Quantization | 4x | No (~1% error) | Embeddings |
| Binary Quantization | 32x | No (lossy) | High-dimensional similarity |
| Delta Encoding | 4-8x | Yes | Sorted ID lists |
| RLE | 2-100x | Yes | Repeated values |

See [tensor_compress documentation](tensor-compress.md) for details.

### Usage Example

```rust
use tensor_store::{TensorStore, TensorData, TensorValue, ScalarValue};

// Create and populate store
let store = TensorStore::new();
let mut user = TensorData::new();
user.set("name", TensorValue::Scalar(ScalarValue::String("Alice".into())));
user.set("embedding", TensorValue::Vector(vec![0.1, 0.2, 0.3]));
store.put("user:1", user)?;

// Save snapshot
store.save_snapshot("/path/to/data.bin")?;

// Later: restore from snapshot
let restored = TensorStore::load_snapshot("/path/to/data.bin")?;
assert!(restored.exists("user:1"));
```

### Best Practices

1. **Atomic saves**: `save_snapshot` writes to a temp file first, so interrupted saves don't corrupt data
2. **Backup strategy**: Keep multiple snapshots with timestamps for point-in-time recovery
3. **Bloom filter on load**: Use `load_snapshot_with_bloom_filter` for read-heavy workloads with sparse key access

## Sparse and Delta Vectors

tensor_store provides specialized vector types that exploit sparsity and clustering for memory efficiency.

### SparseVector

For vectors with many zeros (e.g., one-hot encodings, pruned embeddings):

```rust
use tensor_store::SparseVector;

// Create from dense vector (zeros are discarded)
let dense = vec![0.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0];
let sparse = SparseVector::from_dense(&dense);

// Or with threshold (values below threshold become zero)
let sparse = SparseVector::from_dense_with_threshold(&dense, 0.1);

// Operations are O(nnz) instead of O(dimension)
let dot = sparse.dot(&other_sparse);      // sparse-sparse
let dot = sparse.dot_dense(&dense_vec);   // sparse-dense
let cosine = sparse.cosine_similarity(&other_sparse);

// Memory: ~6 bytes per non-zero vs 4 bytes per element
assert!(sparse.memory_bytes() < dense.len() * 4);
```

**When to use**: Sparsity > 80% yields memory savings; > 95% yields significant speedups.

### DeltaVector

For clustered embeddings that share common structure:

```rust
use tensor_store::{DeltaVector, ArchetypeRegistry, KMeansConfig};

// Automatic archetype discovery via k-means
let mut registry = ArchetypeRegistry::new(16);  // max 16 archetypes
registry.discover_archetypes(&embeddings, 5, KMeansConfig::default());

// Encode vectors as deltas from nearest archetype
let results = registry.encode_batch(&embeddings, 0.01);  // threshold
for (delta, compression_ratio) in results {
    println!("Compressed {}x", compression_ratio);
}

// Fast similarity with precomputed archetype dot products
let archetype = registry.get(delta.archetype_id()).unwrap();
let arch_dot_query: f32 = archetype.iter().zip(query.iter()).map(|(a, q)| a * q).sum();
let result = delta.dot_dense_with_precomputed(&query, arch_dot_query);
```

**When to use**: Embeddings that cluster (semantic similarity, user profiles, documents).

### K-means Clustering

Discover archetypes automatically:

```rust
use tensor_store::{KMeans, KMeansConfig, KMeansInit};

let config = KMeansConfig {
    max_iterations: 100,
    convergence_threshold: 1e-4,
    seed: 42,
    init_method: KMeansInit::KMeansPlusPlus,  // Better than random
};

let kmeans = KMeans::new(config);
let centroids = kmeans.fit(&vectors, 5);  // Find 5 clusters

// Analyze coverage quality
let stats = registry.analyze_coverage(&vectors, 0.01);
println!("Avg similarity to archetype: {}", stats.avg_similarity);
println!("Avg compression ratio: {}", stats.avg_compression_ratio);
```

### HNSW with Mixed Storage

The HNSW index supports all storage types:

```rust
use tensor_store::{HNSWIndex, HNSWConfig, EmbeddingStorage};

let index = HNSWIndex::new(HNSWConfig::default());

// Insert dense, sparse, or delta vectors
index.insert("doc:1", EmbeddingStorage::Dense(vec![0.1, 0.2, 0.3]));
index.insert("doc:2", EmbeddingStorage::Sparse(sparse_vec));
index.insert("doc:3", EmbeddingStorage::Delta(delta_vec));

// Search works across all types
let results = index.search(&query, 10);
```

## Tiered Storage

For datasets larger than available RAM, TieredStore provides automatic hot/cold tiering with memory-mapped cold storage.

### TieredStore

Two-tier storage combining fast in-memory access with disk-backed cold storage. Uses pure tensor storage (MetadataSlab) for the hot tier, ensuring zero resize stalls:

```rust
use tensor_store::{TieredStore, TieredConfig};

// Configure tiered storage
let config = TieredConfig {
    cold_dir: "/data/cold".into(),      // Directory for cold storage files
    cold_capacity: 64 * 1024 * 1024,    // Initial cold file size (64MB)
    sample_rate: 100,                    // Track 1% of accesses
};

let mut store = TieredStore::new(config)?;

// Use like regular TensorStore
store.put("user:1", tensor);
let data = store.get("user:1")?;

// Check which tier data is in
let stats = store.stats();
println!("Hot: {}, Cold: {}", stats.hot_count, stats.cold_count);
```

### Cold Migration

Move infrequently accessed data to cold storage:

```rust
// Migrate shards not accessed in 30 seconds
let migrated = store.migrate_cold(30_000)?;
println!("Migrated {} entries to cold storage", migrated);

// Data is still accessible - reads from cold tier
let data = store.get("old_key")?;  // Loads from mmap, promotes to hot
```

### Preloading

Warm specific keys before heavy access:

```rust
// Preload keys you know you'll need
let keys = vec!["user:1", "user:2", "user:3"];
let loaded = store.preload(&keys)?;
println!("Preloaded {} entries from cold to hot", loaded);
```

### Access Instrumentation

Track hot/cold access patterns:

```rust
// Find most accessed shards
let hot_shards = store.hot_shards(5);  // Top 5
for (shard_id, access_count) in hot_shards {
    println!("Shard {}: {} accesses", shard_id, access_count);
}

// Find cold shards (not accessed in 60s)
let cold_shards = store.cold_shards(60_000);
println!("Cold shards: {:?}", cold_shards);
```

### MmapStore (Low-Level)

Direct memory-mapped file access for advanced use cases:

```rust
use tensor_store::{MmapStoreBuilder, MmapStore, MmapStoreMut};

// Create a new mmap file
let mut builder = MmapStoreBuilder::create("/data/store.bin")?;
for (key, tensor) in entries {
    builder.add(&key, &tensor)?;
}
let file_size = builder.finish()?;

// Open read-only
let store = MmapStore::open("/data/store.bin")?;
let tensor = store.get("key")?;

// Open mutable (supports updates)
let mut store = MmapStoreMut::open("/data/store.bin")?;
store.insert("new_key", &tensor)?;
store.flush()?;

// Compact to reclaim space from updates
store.compact("/data/store_compacted.bin")?;
```

### Performance Characteristics

| Operation | Throughput | Notes |
|-----------|------------|-------|
| Hot put | 3.0+ M/s | MetadataSlab insert |
| Hot get | 4.5+ M/s | MetadataSlab lookup + clone |
| Cold migration | 1.0-1.4 M/s | Serialize to mmap |
| Cold promotion | 0.5-0.6 M/s | Deserialize + insert |
| Resize stalls | None | Pure tensor architecture |

### When to Use TieredStore

- Dataset exceeds available RAM
- Working set is smaller than total data (e.g., hot 10%, cold 90%)
- Development machines with limited memory
- Need graceful degradation under memory pressure

### File Format

Cold storage uses a simple append-only format:

```
[Header: 16 bytes]
  Magic: "MMAP" (4 bytes)
  Version: 1 (4 bytes, little-endian)
  Entry count: (8 bytes, little-endian)

[Entries: variable]
  Key length: (4 bytes)
  Key: (variable, UTF-8)
  Data length: (4 bytes)
  Data: (variable, bincode-serialized TensorData)
```

Updates append new versions; use `compact()` to reclaim space from old versions.

## Future Considerations

Not yet implemented:

- **Incremental snapshots**: Only save changes since last snapshot
- **Transactions**: Atomic multi-key operations
- **TTL**: Automatic expiration
- **Background snapshots**: Async save without blocking
- **Streaming Compression**: Process without loading full snapshot
- **Distributed tiering**: Cross-node cold storage

These belong in higher layers or future modules.
