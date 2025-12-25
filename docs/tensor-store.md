# Tensor Store

The foundational storage layer for Neumann. Holds all data in a unified tensor structure. Knows nothing about queries - just stores and retrieves.

## Design Principles

1. **Single Responsibility**: Store and retrieve tensors by key. No query logic.
2. **Concurrent by Design**: Uses DashMap for sharded concurrent access.
3. **Shareable Storage**: TensorStore clones share the same underlying data via Arc.
4. **Minimal Dependencies**: Only DashMap for concurrent HashMap.
5. **Predictable Memory**: Standard collections with no hidden allocations.

## Concurrency Model

The store uses [DashMap](https://crates.io/crates/dashmap), a concurrent sharded HashMap:

- **Reads**: Lock-free, can proceed in parallel with other reads and writes to different shards
- **Writes**: Only block other writes to the same shard (~16 shards by default)
- **No poisoning**: Unlike RwLock, panics don't poison the entire store
- **Clone on read**: `get()` returns cloned data to avoid holding references across operations

This provides significantly better performance under write contention compared to a single RwLock.

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
let store = TensorStore::with_capacity(10_000);  // Pre-allocate

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

## Error Handling

### TensorStoreError

Only `get` and `delete` can fail:

| Error | Cause |
|-------|-------|
| `NotFound(key)` | `get` or `delete` on nonexistent key |

Note: Unlike the previous RwLock-based implementation, there is no `LockError`. DashMap's sharded design eliminates lock poisoning concerns.

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
| `put` | O(1) amortized | Sharded HashMap insert |
| `get` | O(1) + clone cost | Clone prevents reference issues |
| `delete` | O(1) | Sharded HashMap remove |
| `exists` | O(1) | Lock-free check |
| `scan` | O(n) | Iterates all shards |
| `scan_count` | O(n) | Iterates all shards, no allocation |
| `len` | O(shards) | Sums shard lengths |
| `clear` | O(n) | Clears all shards |

### Concurrent Write Performance

With the sharded design, concurrent writes to different keys typically proceed without blocking:

```
Thread A writes "user:1" -> Shard 3 (locked briefly)
Thread B writes "post:5" -> Shard 7 (parallel, different shard)
Thread C writes "user:2" -> Shard 3 (waits for A, same shard)
```

This provides roughly 16x better write throughput under contention compared to a single lock.

## Memory Layout

```
TensorStore
  |
  +-- DashMap<String, TensorData> (16 shards by default)
                        |
                        +-- HashMap<String, TensorValue>
                                            |
                                            +-- Scalar(ScalarValue)
                                            +-- Vector(Vec<f32>)
                                            +-- Pointer(String)
                                            +-- Pointers(Vec<String>)
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
// DashMap handles contention efficiently via sharding
```

### Shared Storage Across Engines

TensorStore clones share the same underlying data, enabling unified entity storage:

```rust
let store = TensorStore::new();

// Clone shares the same underlying Arc<DashMap>
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

## Architectural Decision: DashMap

**Why DashMap over RwLock<HashMap>?**

| Aspect | RwLock<HashMap> | DashMap |
|--------|-----------------|---------|
| Read concurrency | Many readers | Many readers |
| Write concurrency | One writer | ~16 writers (different shards) |
| Poisoning | Can poison on panic | No poisoning |
| API complexity | Manual lock handling | Simple API |
| Dependency | std only | dashmap crate |

For infrastructure code where concurrent writes are expected and reliability is critical, DashMap provides better performance and eliminates the poisoning failure mode.

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
```

### Implementation Details

| Feature | Description |
|---------|-------------|
| Format | bincode (compact binary) |
| Atomicity | Writes to `.tmp` file, then atomic rename |
| Bloom Filter | Rebuilt on load if requested |
| HNSW Index | Not persisted (rebuild required) |

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

## Future Considerations

Not yet implemented:

- **Write-Ahead Log (WAL)**: For durability between snapshots
- **Incremental snapshots**: Only save changes since last snapshot
- **Compression**: Reduce snapshot file size
- **Transactions**: Atomic multi-key operations
- **TTL**: Automatic expiration
- **Background snapshots**: Async save without blocking

These belong in higher layers or future modules.
