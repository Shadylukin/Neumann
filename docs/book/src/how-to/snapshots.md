# How to Use Snapshots and Shared Storage

This guide covers persistence (snapshot/restore), shared storage across engines,
and compressed snapshots.

**See also**: [API Reference](../reference/api/tensor-store.md) |
[Tiered Storage](../explanation/tiered-storage.md)

---

## 1. Basic Snapshot and Restore

### Save a Snapshot

```rust
store.save_snapshot("data.bin")?;
```

This serializes all data in the store to a single binary file using bincode.

### Load a Snapshot

```rust
let store = TensorStore::load_snapshot("data.bin")?;
```

### Load with Bloom Filter Rebuild

Bloom filter state is not persisted. Use this variant to reconstruct it during
load:

```rust
let store = TensorStore::load_snapshot_with_bloom_filter(
    "data.bin",
    10_000,   // expected items
    0.01      // false positive rate
)?;
```

## 2. Compressed Snapshots

For large stores, use compressed snapshots to reduce file size:

```rust
use tensor_compress::{CompressionConfig, QuantMode};

let config = CompressionConfig {
    vector_quantization: Some(QuantMode::Int8),  // 4x vector compression
    delta_encoding: true,
    rle_encoding: true,
};

store.save_snapshot_compressed("data.bin", config)?;
```

| Compression Option | Effect | Trade-off |
| --- | --- | --- |
| `QuantMode::Int8` | Quantize f32 vectors to i8 (4x smaller) | Minor precision loss |
| `delta_encoding` | Store sequential differences | Better for sorted data |
| `rle_encoding` | Run-length encode repeated values | Better for sparse data |

## 3. Shared Storage Across Engines

TensorStore uses `Arc<SlabRouter>` internally, so clones share the same data:

```rust
let store = TensorStore::new();

// Clone shares the same underlying storage
let store_clone = store.clone();

// Both handles see the same data
store.put("user:1", user_data)?;
assert!(store_clone.exists("user:1"));
```

### Using with Multiple Engines

Pass cloned stores to different engines. All engines operate on the same data:

```rust
let store = TensorStore::new();

let vector_engine = VectorEngine::with_store(store.clone());
let graph_engine = GraphEngine::with_store(store.clone());
let relational_engine = RelationalEngine::with_store(store.clone());

// Data written by one engine is visible to all others
```

This is the standard pattern in Neumann. The `QueryRouter` creates a single
TensorStore and passes clones to each engine.

## 4. Basic Operations Reference

```rust
let store = TensorStore::new();

// Store a tensor
let mut user = TensorData::new();
user.set("name", TensorValue::Scalar(ScalarValue::String("Alice".into())));
user.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
user.set("embedding", TensorValue::Vector(vec![0.1, 0.2, 0.3, 0.4]));
store.put("user:1", user)?;

// Retrieve
let data = store.get("user:1")?;

// Scan by prefix
let user_keys = store.scan("user:");
let count = store.scan_count("user:");

// Check existence
if store.exists("user:1") { /* ... */ }
```

### Automatic Sparsification

Use `TensorValue::from_embedding_auto` to automatically choose between dense
and sparse storage:

```rust
// Automatically uses Sparse if sparsity >= 70%
let val = TensorValue::from_embedding_auto(dense_vec);

// With custom thresholds (value_threshold, sparsity_threshold)
let val = TensorValue::from_embedding(dense_vec, 0.01, 0.8);
```

### Cross-Format Vector Operations

TensorValue supports operations across dense and sparse formats:

```rust
// Dot product works across Dense, Sparse, and mixed
let dot = tensor_a.dot(&tensor_b);

// Cosine similarity with automatic format handling
let sim = tensor_a.cosine_similarity(&tensor_b);
```

### Sparse Arithmetic

```rust
// Create delta from before/after states (only stores differences)
let delta = SparseVector::from_diff(&before, &after, threshold);

// Subtraction: self - other
let diff = a.sub(&b);

// Weighted average: (w1 * a + w2 * b) / (w1 + w2)
let merged = a.weighted_average(&b, 0.7, 0.3);

// Project out conflicting component
let orthogonal = v.project_orthogonal(&conflict_direction);
```

### Memory Efficiency Metrics

```rust
let sparse = SparseVector::from_dense(&dense_vec);

sparse.sparsity()           // Fraction of zeros (0.0 - 1.0)
sparse.memory_bytes()       // Actual memory used
sparse.dense_memory_bytes() // Memory if stored dense
sparse.compression_ratio()  // Dense / Sparse ratio
```

For a 1000-dim vector with 90% zeros:
- Dense: 4000 bytes
- Sparse: ~800 bytes (100 positions * 4 bytes + 100 values * 4 bytes)
- Compression ratio: 5x

## Common Issues

**"IoError" on save**: Check that the target directory exists and has write
permissions. Snapshot files can be large; ensure sufficient disk space.

**"SerializationError" on load**: The file may be corrupted or from an
incompatible snapshot format version. Neumann supports V2 and V3 formats; older
versions are not backward-compatible.

**Bloom filter not working after load**: Use `load_snapshot_with_bloom_filter`
instead of `load_snapshot`. The bloom filter is not persisted and must be
rebuilt.
