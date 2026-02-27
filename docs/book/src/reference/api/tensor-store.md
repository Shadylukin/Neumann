# Tensor Store API Reference

## See Also

- **Explanation**: [SlabRouter Architecture](../../explanation/slab-router.md) |
  [Bloom Filters](../../explanation/bloom-filter.md) |
  [HNSW Algorithm](../../explanation/hnsw-algorithm.md) |
  [Delta Vectors](../../explanation/delta-vectors.md) |
  [Tiered Storage](../../explanation/tiered-storage.md)
- **How-to**: [Configure Tiered Storage](../../how-to/configure-tiered-storage.md) |
  [Bloom Filters](../../how-to/bloom-filters.md) |
  [Tune HNSW](../../how-to/tune-hnsw.md) |
  [Delta Embeddings](../../how-to/delta-embeddings.md) |
  [Snapshots](../../how-to/snapshots.md)

---

## Core Types

### TensorValue

Represents different types of values a tensor can hold.

| Variant | Rust Type | Use Case |
| --- | --- | --- |
| `Scalar(ScalarValue)` | enum | Properties (name, age, active) |
| `Vector(Vec<f32>)` | dense array | Embeddings for similarity search |
| `Sparse(SparseVector)` | compressed | Sparse embeddings (>70% zeros) |
| `Pointer(String)` | single ref | Single relationship to another tensor |
| `Pointers(Vec<String>)` | multi ref | Multiple relationships |

### ScalarValue

| Variant | Rust Type | Example |
| --- | --- | --- |
| `Null` | --- | Missing/undefined value |
| `Bool` | `bool` | `true`, `false` |
| `Int` | `i64` | `42`, `-1` |
| `Float` | `f64` | `3.14159` |
| `String` | `String` | `"Alice"` |
| `Bytes` | `Vec<u8>` | Raw binary data |

### TensorData

An entity that holds scalar properties, vector embeddings, and pointers to other
tensors via a `HashMap<String, TensorValue>` internally.

### SparseVector

Memory-efficient storage for vectors with many zeros.

```rust
pub struct SparseVector {
    dimension: usize,      // Total dimension (shell/boundary)
    positions: Vec<u32>,   // Sorted positions of non-zero values
    values: Vec<f32>,      // Corresponding values
}
```

### DeltaVector

Stores vectors as differences from reference archetype vectors.

```rust
pub struct DeltaVector {
    archetype_id: usize,       // Reference archetype
    dimension: usize,          // For reconstruction
    positions: Vec<u16>,       // Diff positions (u16 for memory)
    deltas: Vec<f32>,          // Delta values
    cached_magnitude: Option<f32>,  // For fast cosine similarity
}
```

### BloomFilter

Probabilistic data structure for O(1) key existence rejection.

```rust
pub struct BloomFilter {
    bits: Box<[AtomicU64]>,  // Atomic u64 blocks for lock-free access
    num_bits: usize,
    num_hashes: usize,
}
```

### ShardAccessTracker

Low-overhead tracking of shard access patterns for intelligent memory tiering.

```rust
pub struct ShardAccessTracker {
    shards: Box<[ShardStats]>,     // Per-shard counters
    shard_count: usize,            // Default: 16
    start_time: Instant,           // For last_access timestamps
    sample_rate: u32,              // 1 = every access, 100 = 1%
    sample_counter: AtomicU64,     // For sampling
}
```

### HNSWAccessStats

Specialized instrumentation for HNSW index.

```rust
pub struct HNSWAccessStats {
    entry_point_accesses: AtomicU64,
    layer0_traversals: AtomicU64,
    upper_layer_traversals: AtomicU64,
    total_searches: AtomicU64,
    distance_calculations: AtomicU64,
}
```

---

## Reserved Field Names

| Field | Purpose | Used By |
| --- | --- | --- |
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
| `_id` | Entity ID | System |

---

## Configuration

### SlabRouterConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `embedding_dim` | `usize` | 384 | Embedding dimension for EmbeddingSlab |
| `cache_capacity` | `usize` | 10,000 | Cache capacity for CacheRing |
| `cache_strategy` | `EvictionStrategy` | Default | Eviction strategy (LRU/LFU) |
| `blob_segment_size` | `usize` | 64MB | Segment size for BlobLog |
| `graph_merge_threshold` | `usize` | 10,000 | Merge threshold for GraphTensor |

### HNSWConfig Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `m` | 16 | Max connections per node per layer |
| `m0` | 32 | Max connections at layer 0 (2*m) |
| `ef_construction` | 200 | Candidates during construction |
| `ef_search` | 50 | Candidates during search |
| `ml` | 1/ln(m) | Level multiplier |
| `sparsity_threshold` | 0.5 | Auto-sparse storage threshold |
| `max_nodes` | 10,000,000 | Capacity limit (prevents memory exhaustion) |

### TieredConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `cold_dir` | `PathBuf` | `/tmp/tensor_cold` | Directory for cold storage files |
| `cold_capacity` | `usize` | 64MB | Initial cold file size |
| `sample_rate` | `u32` | 100 | Access tracking sampling (100 = 1%) |

### BloomFilter Parameter Tuning

| Expected Items | FP Rate | Bits | Hash Functions | Memory |
| --- | --- | --- | --- | --- |
| 10,000 | 1% | 95,851 | 7 | ~12 KB |
| 10,000 | 0.1% | 143,776 | 10 | ~18 KB |
| 100,000 | 1% | 958,506 | 7 | ~117 KB |
| 1,000,000 | 1% | 9,585,059 | 7 | ~1.2 MB |

---

## Performance Characteristics

### Operation Complexity

| Operation | Time Complexity | Notes |
| --- | --- | --- |
| `put` | O(log n) | BTreeMap insert |
| `get` | O(log n) + clone | Clone prevents reference issues |
| `delete` | O(log n) | BTreeMap remove |
| `exists` | O(log n) | BTreeMap lookup |
| `scan` | O(k + log n) | BTreeMap range, k = result count |
| `scan_count` | O(k + log n) | No allocation |
| `scan_filter_map` | O(k + log n) | Single-pass filter with selective cloning |
| `len` | O(1) | Cached count |
| `clear` | O(n) | Clears all data |

### Throughput Comparison

| Metric | SlabRouter | Previous (DashMap) |
| --- | --- | --- |
| PUT throughput | 3.1+ M ops/sec | 2.5 M ops/sec |
| GET throughput | 4.9+ M ops/sec | 4.5 M ops/sec |
| Throughput variance (CV) | 12% steady-state | 222% during resize |
| Resize stalls | None | 99.6% throughput drops |

### SparseVector Operation Complexity

| Operation | Complexity | Notes |
| --- | --- | --- |
| `from_dense` | O(n) | Filters zeros |
| `to_dense` | O(n) | Reconstructs full vector |
| `get(index)` | O(log nnz) | Binary search |
| `set(index, value)` | O(nnz) | Insert/remove maintains sort |
| `dot(sparse)` | O(min(nnz_a, nnz_b)) | Merge-join on positions |
| `dot_dense(dense)` | O(nnz) | Only access stored positions |
| `add(sparse)` | O(nnz_a + nnz_b) | Merge-based |
| `cosine_similarity` | O(nnz) | Using cached magnitudes |

### SparseVector Distance Metrics

| Metric | Range | Use Case |
| --- | --- | --- |
| `cosine_similarity` | -1 to 1 | Directional similarity |
| `angular_distance` | 0 to PI | Linear for small angles |
| `geodesic_distance` | 0 to PI | Arc length on unit sphere |
| `jaccard_index` | 0 to 1 | Structural overlap (positions) |
| `overlap_coefficient` | 0 to 1 | Subset containment |
| `weighted_jaccard` | 0 to 1 | Value-weighted structural overlap |
| `euclidean_distance` | 0 to inf | L2 norm of difference |
| `manhattan_distance` | 0 to inf | L1 norm of difference |

### HNSW Storage Types

| Storage Type | Memory | Use Case | Distance Computation |
| --- | --- | --- | --- |
| Dense | 4 bytes/dim | General purpose | SIMD dot product |
| Sparse | 6 bytes/nnz | >50% zeros | Sparse-sparse O(nnz) |
| Delta | 6 bytes/diff | Clustered embeddings | Via archetype |
| TensorTrain | 8-10x compression | 768+ dimensions | Native TT or reconstruct |

### Specialized Slabs

| Slab | Data Structure | Purpose |
| --- | --- | --- |
| `MetadataSlab` | `RwLock<BTreeMap<String, TensorData>>` | General key-value storage |
| `EntityIndex` | Sorted vocabulary + hash index | Stable ID assignment |
| `EmbeddingSlab` | Dense f32 arrays + BTreeMap | Embedding vectors |
| `GraphTensor` | CSR format (row pointers + column indices) | Graph edges |
| `RelationalSlab` | Columnar storage | Table rows |
| `CacheRing` | Ring buffer with LRU/LFU | Fixed-size cache |
| `BlobLog` | Append-only segments | Large binary data |

---

## Error Types

### TensorStoreError

| Error | Cause |
| --- | --- |
| `NotFound(key)` | `get` or `delete` on nonexistent key |

### SnapshotError

| Error | Cause |
| --- | --- |
| `IoError(std::io::Error)` | File not found, permission denied, disk full |
| `SerializationError(String)` | Corrupted file, incompatible format |

### TieredError

| Error | Cause |
| --- | --- |
| `Store(TensorStoreError)` | Underlying store error |
| `Mmap(MmapError)` | Memory-mapped file error |
| `Io(std::io::Error)` | I/O error |
| `NotConfigured` | Cold storage not configured |

### EmbeddingStorageError

| Error | Cause |
| --- | --- |
| `DeltaRequiresRegistry` | Delta storage used without archetype registry |
| `ArchetypeNotFound(id)` | Referenced archetype not in registry |
| `CapacityExceeded { limit, current }` | HNSW index at max_nodes limit |
| `DeltaNotSupported` | Delta vectors inserted into HNSW (unsupported) |

---

## Dependencies

| Crate | Purpose |
| --- | --- |
| `serde` | Serialization |
| `bincode` | Binary snapshot format |
| `tensor_compress` | Compression algorithms |
| `wide` | SIMD operations (f32x8) |
| `memmap2` | Memory-mapped files |
| `fxhash` | Fast hashing |
| `parking_lot` | Efficient locks |
| `bitvec` | Bit vectors for bloom filter |

---

## Related Modules

| Module | Relationship |
| --- | --- |
| `relational_engine` | Uses TensorStore for table row storage |
| `graph_engine` | Uses TensorStore for node/edge storage |
| `vector_engine` | Uses TensorStore + HNSWIndex for embeddings |
| `tensor_compress` | Provides compression for snapshots |
| `tensor_checkpoint` | Uses TensorStore snapshots for atomic restore |
| `tensor_chain` | Uses TensorStore for blockchain state |
