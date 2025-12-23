# Tensor Store

The foundational storage layer for Neumann. Holds all data in a unified tensor structure. Knows nothing about queries - just stores and retrieves.

## Design Principles

1. **Single Responsibility**: Store and retrieve tensors by key. No query logic.
2. **Thread Safety**: All operations are safe for concurrent access via `RwLock`.
3. **Zero External Dependencies**: Uses only Rust standard library.
4. **Predictable Memory**: Standard collections with no hidden allocations.

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

## API Reference

### TensorStore

```rust
// Construction
let store = TensorStore::new();

// Core operations
store.put("user:1", tensor)?;      // Store tensor under key
store.get("user:1")?;              // Retrieve tensor (cloned)
store.delete("user:1")?;           // Remove tensor
store.exists("user:1")?;           // Check if key exists

// Scanning
store.scan("user:")?;              // List keys with prefix
store.scan_count("user:")?;        // Count keys with prefix (no allocation)

// Metadata
store.len()?;                      // Total tensor count
store.is_empty()?;                 // Check if store is empty
store.clear()?;                    // Remove all tensors
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

All store operations return `Result<T, TensorStoreError>`:

| Error | Cause |
|-------|-------|
| `NotFound(key)` | `get` or `delete` on nonexistent key |
| `LockError` | Failed to acquire read/write lock (poisoned) |

## Thread Safety

The store uses `RwLock<HashMap>`:

- Multiple concurrent reads allowed
- Writes are exclusive
- `get()` returns cloned data to prevent lock contention

For high-contention scenarios, consider sharding by key prefix.

## Performance Characteristics

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| `put` | O(1) amortized | HashMap insert |
| `get` | O(1) + clone cost | Clone prevents lock holding |
| `delete` | O(1) | HashMap remove |
| `exists` | O(1) | HashMap contains |
| `scan` | O(n) | Iterates all keys |
| `scan_count` | O(n) | Iterates all keys, no allocation |
| `len` | O(1) | HashMap len |
| `clear` | O(n) | HashMap clear |

## Memory Layout

```
TensorStore
  |
  +-- RwLock<HashMap<String, TensorData>>
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
let user_keys = store.scan("user:")?;

// Count posts without allocating
let post_count = store.scan_count("post:")?;
```

### Concurrent Access

```rust
use std::sync::Arc;
use std::thread;

let store = Arc::new(TensorStore::new());

// Spawn readers
for _ in 0..4 {
    let store = Arc::clone(&store);
    thread::spawn(move || {
        let _ = store.get("key");
    });
}

// Spawn writers
for _ in 0..2 {
    let store = Arc::clone(&store);
    thread::spawn(move || {
        store.put("key", TensorData::new()).unwrap();
    });
}
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
| `store_10k_entities` | Handles 10,000 entities |
| `store_concurrent_*` | Thread safety verified |
| `store_empty_key` | Empty string as key works |
| `store_unicode_keys` | Unicode in keys works |

## Future Considerations

Not implemented (out of scope for Module 1):

- **Persistence**: File/S3 backend for durability
- **Transactions**: Atomic multi-key operations
- **TTL**: Automatic expiration
- **Compression**: For large vectors/bytes
- **Sharding**: For horizontal scaling
- **Snapshots**: Point-in-time backups

These belong in higher layers or future modules.
