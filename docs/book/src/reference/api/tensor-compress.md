# Tensor Compress API Reference

> **See also:** [Compression Algorithms](../../explanation/compression-algorithms.md) |
> [Compression How-To](../../how-to/compression.md)

---

## Core Types

| Type | Description |
| --- | --- |
| `TTVector` | Complete TT-decomposition of a vector with cores, shape, and ranks |
| `TTCore` | Single 3D tensor core (left_rank x mode_size x right_rank) |
| `TTConfig` | Configuration for TT decomposition (shape, max_rank, tolerance) |
| `CompressionConfig` | Snapshot compression settings (tensor mode, delta, RLE) |
| `TensorMode` | Compression mode enum (currently TensorTrain variant) |
| `RleEncoded<T>` | Run-length encoded data with values and run lengths |
| `DeltaSnapshot` | Snapshot containing only changes since a base snapshot |
| `DeltaChain` | Chain of deltas with efficient lookup and compaction |
| `StreamingWriter` | Memory-bounded incremental snapshot writer |
| `StreamingReader` | Iterator-based snapshot reader |
| `StreamingTTWriter` | Streaming TT-compressed vector writer |
| `StreamingTTReader` | Streaming TT-compressed vector reader |
| `Matrix` | Row-major matrix for SVD operations |
| `SvdResult` | Truncated SVD result (U, S, Vt matrices) |
| `TensorView` | Zero-copy logical view of tensor data |
| `DeltaBuilder` | Builder for creating delta snapshots |

## Compressed Value Types

```rust
pub enum CompressedValue {
    Scalar(CompressedScalar),           // Int, Float, String, Bool, Null
    VectorRaw(Vec<f32>),                // Uncompressed
    VectorTT { cores, original_dim, shape, ranks },  // TT-compressed
    VectorSparse { dimension, positions, values },   // Sparse
    IdList(Vec<u8>),                    // Delta + varint encoded
    RleInt(RleEncoded<i64>),            // RLE encoded integers
    Pointer(String),                    // Single pointer
    Pointers(Vec<String>),              // Multiple pointers
}
```

## Error Types

### TTError

| Variant | Description |
| --- | --- |
| `ShapeMismatch` | Vector dimension doesn't match reshape target |
| `EmptyVector` | Cannot decompose empty vector |
| `InvalidRank` | TT-rank must be >= 1 |
| `IncompatibleShapes` | TT vectors have different shapes for operation |
| `InvalidShape` | Shape contains zero or is empty |
| `InvalidTolerance` | Tolerance must be 0 < tol <= 1 |
| `Decompose` | SVD decomposition failed |

### FormatError

| Variant | Description |
| --- | --- |
| `InvalidMagic` | File magic bytes don't match expected |
| `UnsupportedVersion` | Format version is newer than supported |
| `Serialization` | Bincode serialization/deserialization error |

### DeltaError

| Variant | Description |
| --- | --- |
| `BaseNotFound` | Referenced base snapshot doesn't exist |
| `SequenceGap` | Delta sequence numbers have gaps |
| `ChainTooLong` | Delta chain exceeds maximum length |

### DecomposeError

| Variant | Description |
| --- | --- |
| `EmptyMatrix` | Cannot decompose empty matrix |
| `DimensionMismatch` | Matrix dimensions don't match for operation |
| `SvdNotConverged` | SVD iteration didn't converge |

## TTConfig

### Presets

| Preset | max_rank | tolerance | Use Case |
| --- | --- | --- | --- |
| `for_dim(d)` | 8 | 1e-4 | Balanced compression/accuracy |
| `high_compression(d)` | 4 | 1e-2 | Maximize compression (2-3x more) |
| `high_accuracy(d)` | 16 | 1e-6 | Maximize accuracy (<0.1% error) |

### Validation

```rust
impl TTConfig {
    pub fn validate(&self) -> Result<(), TTError> {
        if self.shape.is_empty() {
            return Err(TTError::InvalidShape("empty shape".into()));
        }
        if self.shape.contains(&0) {
            return Err(TTError::InvalidShape("shape contains zero".into()));
        }
        if self.max_rank < 1 {
            return Err(TTError::InvalidRank);
        }
        if self.tolerance <= 0.0 || self.tolerance > 1.0 || !self.tolerance.is_finite() {
            return Err(TTError::InvalidTolerance(self.tolerance));
        }
        Ok(())
    }
}
```

## CompressionConfig

```rust
pub struct CompressionConfig {
    pub tensor_mode: Option<TensorMode>,  // TT compression for vectors
    pub delta_encoding: bool,             // For sorted ID lists
    pub rle_encoding: bool,               // For repeated values
}

// Presets
CompressionConfig::high_compression()  // max_rank=4, all encodings enabled
CompressionConfig::balanced(dim)       // max_rank=8, all encodings enabled
CompressionConfig::high_accuracy(dim)  // max_rank=16, all encodings enabled
```

## Dimension Presets

| Constant | Value | Model |
| --- | --- | --- |
| `SMALL` | 64 | MiniLM and small models |
| `MEDIUM` | 384 | all-MiniLM-L6-v2 |
| `STANDARD` | 768 | BERT, sentence-transformers |
| `LARGE` | 1536 | OpenAI text-embedding-ada-002 |
| `XLARGE` | 4096 | LLaMA and large models |

## TT Operations

| Function | Description | Complexity |
| --- | --- | --- |
| `tt_decompose` | Decompose vector to TT format | O(n * d * r^2) |
| `tt_decompose_batch` | Parallel batch decomposition (4+ vectors) | O(batch * n * d * r^2 / threads) |
| `tt_reconstruct` | Reconstruct vector from TT | O(d^n * r^2) |
| `tt_dot_product` | Dot product in TT space | O(n * d * r^4) |
| `tt_dot_product_batch` | Batch dot products | Parallel when >= 4 targets |
| `tt_cosine_similarity` | Cosine similarity in TT space | O(n * d * r^4) |
| `tt_cosine_similarity_batch` | Batch cosine similarities | Parallel when >= 4 targets |
| `tt_euclidean_distance` | Euclidean distance in TT space | O(n * d * r^4) |
| `tt_euclidean_distance_batch` | Batch Euclidean distances | Parallel when >= 4 targets |
| `tt_norm` | L2 norm of TT vector | O(n * d * r^4) |
| `tt_scale` | Scale TT vector by constant | `O(cores[0].size)` |

Where: n = number of modes, d = mode size, r = TT-rank

## Streaming Operations

### StreamingWriter / StreamingReader

| Function | Description |
| --- | --- |
| `StreamingWriter::new` | Create streaming writer with config |
| `StreamingWriter::write_entry` | Write a single entry |
| `StreamingWriter::finish` | Finalize with trailer |
| `StreamingReader::open` | Open streaming file |
| `StreamingReader::entry_count` | Get total entry count |
| `convert_to_streaming` | Convert snapshot to streaming format |
| `read_streaming_to_snapshot` | Read streaming format into snapshot |
| `merge_streaming` | Merge multiple streaming snapshots |

### StreamingTTWriter / StreamingTTReader

| Function | Description |
| --- | --- |
| `StreamingTTWriter::new` | Create TT streaming writer |
| `StreamingTTWriter::write_vector` | Decompose and write vector |
| `StreamingTTWriter::write_tt` | Write pre-decomposed TT |
| `StreamingTTWriter::finish` | Finalize with trailer |
| `StreamingTTReader::open` | Open TT streaming file |
| `streaming_tt_similarity_search` | Search streaming TT file |
| `convert_vectors_to_streaming_tt` | Batch convert vectors |
| `read_streaming_tt_all` | Load all TT vectors |

### File Format

Uses a trailer-based header so entry count is known at the end:

```text
+------------------------+
| Magic (NEUS/NEUT)  4B  |  Identifies streaming snapshot/TT
+------------------------+
| Entry 1 length     4B  |  Little-endian u32
+------------------------+
| Entry 1 data      var  |  Bincode-serialized entry
+------------------------+
| Entry 2 length     4B  |
+------------------------+
| Entry 2 data      var  |
+------------------------+
| ...                    |
+------------------------+
| Trailer           var  |  Bincode-serialized header
+------------------------+
| Trailer size       8B  |  Little-endian u64
+------------------------+
```

**Security limits:**

- Maximum trailer size: 1 MB (`MAX_TRAILER_SIZE`)
- Maximum entry size: 100 MB (`MAX_ENTRY_SIZE`)

## Delta Operations

| Function | Description |
| --- | --- |
| `DeltaBuilder::new` | Create delta builder with base ID and start sequence |
| `DeltaBuilder::put` | Record a put (add/update) change |
| `DeltaBuilder::delete` | Record a delete change |
| `DeltaBuilder::build` | Build the delta snapshot |
| `apply_delta` | Apply delta to base snapshot |
| `merge_deltas` | Merge multiple deltas (keeps latest state per key) |
| `diff_snapshots` | Compute delta between two snapshots |
| `DeltaChain::get` | Get current state of key (checks chain then base) |
| `DeltaChain::compact` | Compact all deltas into new base |
| `DeltaChain::should_compact` | Check if compaction is recommended |

### Delta Entry Types

```rust
pub enum ChangeType {
    Put,    // Entry was added or updated
    Delete, // Entry was deleted
}

pub struct DeltaEntry {
    pub key: String,
    pub change: ChangeType,
    pub value: Option<CompressedEntry>,  // None for Delete
    pub sequence: u64,
}
```

### Delta Format

```text
+------------------------+
| Magic (NEUD)       4B  |
+------------------------+
| Version            2B  |
+------------------------+
| Base ID           var  |  String (length-prefixed)
+------------------------+
| Sequence Range     16B |  (start, end) u64 pair
+------------------------+
| Change Count        8B |
+------------------------+
| Created At          8B |  Unix timestamp
+------------------------+
| Entries           var  |  Bincode-serialized Vec<DeltaEntry>
+------------------------+
```

## Lossless Compression Methods

### Delta + Varint Encoding

For sorted integer sequences (node IDs, timestamps). Delta encoding stores the
first value then differences; varint encoding uses 7 bits per byte with a
continuation high bit.

**Varint byte sizes:**

| Value Range | Bytes |
| --- | --- |
| 0 - 127 | 1 |
| 128 - 16,383 | 2 |
| 16,384 - 2,097,151 | 3 |
| 2,097,152 - 268,435,455 | 4 |
| ... up to u64::MAX | 10 |

### Run-Length Encoding

```rust
pub struct RleEncoded<T: Eq> {
    pub values: Vec<T>,      // Unique values in order
    pub run_lengths: Vec<u32>, // Count for each value
}
```

**Compression scenarios:**

| Data Pattern | Runs | Compression |
| --- | --- | --- |
| `[5, 5, 5, 5, 5]` (1000x) | 1 | 500x |
| `[1, 2, 3, 4, 5]` (all different) | 5 | 0.8x (overhead) |
| `[1, 1, 2, 2, 2, 3, 1, 1, 1, 1]` | 4 | 2.5x |
| Status column (pending/active/done) | ~300 per 10000 | ~33x |

### Sparse Vector Format

For vectors with >50% zeros. Sparse format is beneficial when:

```text
sparse_storage_size = 8 + 8 + nnz*2 + nnz*4 = 16 + nnz*6
Dense storage = dimension * 4

Sparse is better when: 16 + nnz*6 < dimension*4
```

| Dimension | Max NNZ for Sparse | Sparsity Threshold |
| --- | --- | --- |
| 100 | 64 | 64% |
| 1000 | 664 | 66.4% |
| 4096 | 2728 | 66.6% |

### Automatic Format Selection

```rust
pub fn compress_vector(vector: &[f32], key: &str, field_name: &str,
    config: &CompressionConfig) -> Result<CompressedValue, FormatError> {

    // 1. Check for embedding-like keys
    let is_embedding = key.starts_with("emb:") ||
                       field_name == "_embedding" ||
                       field_name == "vector";

    if is_embedding {
        if let Some(TensorMode::TensorTrain(tt_config)) = &config.tensor_mode {
            return Ok(CompressedValue::VectorTT { ... });
        }
    }

    // 2. Check for ID list pattern
    if config.delta_encoding && looks_like_id_list(vector, field_name) {
        return Ok(CompressedValue::IdList(...));
    }

    // 3. Fall back to raw
    Ok(CompressedValue::VectorRaw(vector.to_vec()))
}
```

## Optimal Shape Selection

The module includes hardcoded optimal shapes for common embedding dimensions:

| Dimension | Shape | Why |
| --- | --- | --- |
| 64 | `[4, 4, 4]` | 3 balanced factors |
| 128 | `[4, 4, 8]` | Near-balanced |
| 256 | `[4, 8, 8]` | Near-balanced |
| 384 | `[4, 8, 12]` | all-MiniLM-L6-v2 |
| 512 | `[8, 8, 8]` | Perfect cube |
| 768 | `[8, 8, 12]` | BERT dimension |
| 1024 | `[8, 8, 16]` | Common LLM size |
| 1536 | `[8, 12, 16]` | OpenAI ada-002 |
| 2048 | `[8, 16, 16]` | Near-balanced |
| 3072 | `[8, 16, 24]` | Large models |
| 4096 | `[8, 8, 8, 8]` | 4D balanced |
| 8192 | `[8, 8, 8, 16]` | Extra large |

For non-standard dimensions, `factorize_balanced` finds factors close to the
nth root.

## Performance

Benchmarks on Apple M4 (aarch64, MacBook Air 24GB), release build:

| Dimension | Decompose | Reconstruct | Similarity | Compression |
| --- | --- | --- | --- | --- |
| 64 | 6.2 us | 29.5 us | 1.1 us | 2.0x |
| 256 | 13.4 us | 113.0 us | 1.5 us | 4.6x |
| 768 | 26.9 us | 431.7 us | 2.4 us | 10.7x |
| 1536 | 62.0 us | 709.8 us | 2.0 us | 16.0x |
| 4096 | 464.5 us | 2142.2 us | 2.4 us | 42.7x |

Batch operations (768-dim, 1000 vectors):

| Operation | Time | Per-vector |
| --- | --- | --- |
| `tt_decompose_batch` | 21 ms | 21.0 us |
| `tt_cosine_similarity_batch` | 11.3 ms | 11.4 us |

Throughput: **39,318 vectors/sec** (768-dim decomposition)

### Industry Comparison

| Method | Compression | Recall | Notes |
| --- | --- | --- | --- |
| **Tensor Train (this)** | 10-42x | ~99% | Similarity in compressed space |
| Scalar Quantization | 4x | 99%+ | Industry default |
| Product Quantization | 16-64x | 56-90% | Requires training |
| Binary Quantization | 32x | 80-95% | Speed-optimized |

## Edge Cases and Gotchas

### Vector Content Patterns

| Pattern | Compression | Reconstruction | Notes |
| --- | --- | --- | --- |
| Constant (all same) | Excellent (>5x) | Accurate | Rank-1 structure |
| All zeros | Good | Accurate | Degenerate case |
| Single spike | Poor | Moderate | No low-rank structure |
| Linear ramp | Good (>2x) | Good | Low-rank |
| Alternating +1/-1 | Poor | Moderate | High-frequency needs high rank |
| Random dense | Good | Good (>0.9 cosine) | Typical embeddings |
| 90% zeros | Consider sparse instead | n/a | Use `compress_dense_as_sparse` |

### Streaming Gotchas

1. **Incomplete files**: Magic bytes are written first, but entry count is in
   trailer. If writer crashes before `finish()`, the file is corrupt.

2. **Memory limits**: `MAX_ENTRY_SIZE = 100MB` and `MAX_TRAILER_SIZE = 1MB`
   prevent allocation attacks. Exceeding these returns an error.

3. **Seek requirement**: `StreamingReader::open` requires `Seek` to read the
   trailer. For non-seekable streams, use `read_streaming_to_snapshot` which
   buffers.

### Delta Chain Gotchas

1. **Chain length**: Default `max_chain_len = 100`. After this, `push()` returns
   `ChainTooLong` error. Call `compact()` periodically.

2. **Sequence gaps**: Deltas should have contiguous sequences. The
   `merge_deltas` function only keeps the latest state per key.

3. **Base reference**: Deltas store a `base_id` string but don't validate it
   exists. Your application must track base snapshots.

## Dependencies

- `serde`: Serialization traits
- `bincode`: Binary format
- `thiserror`: Error types
- `rayon`: Parallel batch operations

No external LAPACK/BLAS -- pure Rust SVD implementation.

## Related Modules

| Module | Relationship |
| --- | --- |
| [tensor_store](./tensor-store.md) | Uses compression for snapshot I/O |
| [tensor_chain](./tensor-chain.md) | Delta compression for state replication |
| [tensor_checkpoint](./tensor-checkpoint.md) | Snapshot format integration |
