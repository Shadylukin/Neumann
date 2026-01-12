# Tensor Compress

Module 8 of Neumann. Provides tensor-native compression exploiting the mathematical structure of high-dimensional embeddings.

## Design Principles

1. **Tensor Mathematics**: Uses Tensor Train decomposition to exploit low-rank structure
2. **Higher Dimensions Are Lower**: Decomposes vectors into products of smaller tensors
3. **Streaming I/O**: Process large snapshots without loading entire dataset
4. **Incremental Updates**: Delta snapshots for efficient replication

## Quick Start

```rust
use tensor_compress::{CompressionConfig, TensorMode, TTConfig};

// High compression preset for 4096-dim embeddings
let config = CompressionConfig::high_compression();

// Or customize
let config = CompressionConfig {
    tensor_mode: Some(TensorMode::tensor_train(4096)),
    delta_encoding: true,
    rle_encoding: true,
};

// Use with TensorStore
store.save_snapshot_compressed("data.bin", config)?;
let loaded = TensorStore::load_snapshot_compressed("data.bin")?;
```

## Tensor Train Decomposition

The primary compression method. Decomposes a d-dimensional vector reshaped as a tensor into a chain of smaller 3D cores.

### How It Works

For a 4096-dim embedding reshaped to (8, 8, 8, 8):

```
Original: 4096 floats = 16 KB
TT-cores: 1x8x4 + 4x8x4 + 4x8x4 + 4x8x1 = 320 floats = 1.25 KB
Compression: 12.8x
```

### Usage

```rust
use tensor_compress::{tt_decompose, tt_reconstruct, tt_cosine_similarity, TTConfig};

let embedding: Vec<f32> = get_embedding();  // 4096-dim
let config = TTConfig::for_dim(4096);

// Decompose
let tt = tt_decompose(&embedding, &config)?;
println!("Compression: {:.1}x", tt.compression_ratio());

// Reconstruct
let restored = tt_reconstruct(&tt);

// Compute similarity without reconstruction
let tt2 = tt_decompose(&other_embedding, &config)?;
let sim = tt_cosine_similarity(&tt, &tt2)?;
```

### Batch Operations

Batch operations use rayon for automatic parallelization when processing 4+ vectors:

```rust
use tensor_compress::{tt_decompose_batch, tt_cosine_similarity_batch, TTConfig};

let vectors: Vec<Vec<f32>> = load_embeddings();
let config = TTConfig::for_dim(4096);

// Batch decompose (parallel for 4+ vectors)
let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
let tts = tt_decompose_batch(&refs, &config)?;

// Batch similarity search
let query_tt = &tts[0];
let similarities = tt_cosine_similarity_batch(query_tt, &tts[1..])?;
```

### Configuration Presets

```rust
// Balanced compression/accuracy (default)
let config = TTConfig::for_dim(4096);

// Maximize compression (lower accuracy)
let config = TTConfig::high_compression(4096);

// Maximize accuracy (lower compression)
let config = TTConfig::high_accuracy(4096);

// Custom
let config = TTConfig {
    shape: vec![8, 8, 8, 8],  // Tensor shape
    max_rank: 8,               // TT-rank bound
    tolerance: 0.01,           // SVD truncation
};
```

### Performance

Benchmarks run on Apple M4 (aarch64, MacBook Air 24GB), release build:

| Dimension | Decompose | Reconstruct | Similarity | Compression |
|-----------|-----------|-------------|------------|-------------|
| 64 | 6.2 us | 29.5 us | 1.1 us | 2.0x |
| 256 | 13.4 us | 113.0 us | 1.5 us | 4.6x |
| 768 | 26.9 us | 431.7 us | 2.4 us | 10.7x |
| 1536 | 62.0 us | 709.8 us | 2.0 us | 16.0x |
| 4096 | 464.5 us | 2142.2 us | 2.4 us | 42.7x |

Batch operations (768-dim, 1000 vectors):

| Operation | Time | Per-vector |
|-----------|------|------------|
| `tt_decompose_batch` | 21 ms | 21.0 us |
| `tt_cosine_similarity_batch` | 11.3 ms | 11.4 us |

Throughput: **39,318 vectors/sec** (768-dim decomposition)

### Industry Comparison

| Method | Compression | Recall | Notes |
|--------|-------------|--------|-------|
| **Tensor Train (this)** | 10-42x | ~99% | Similarity in compressed space |
| Scalar Quantization | 4x | 99%+ | Industry default |
| Product Quantization | 16-64x | 56-90% | Requires training |
| Binary Quantization | 32x | 80-95% | Speed-optimized |

TT advantages over alternatives:
- Compute similarity directly in compressed space (2.4 us vs full reconstruction)
- Higher accuracy at high compression ratios
- No training required, works with any vector distribution
- Pure Rust implementation (no BLAS/LAPACK dependency)

## Streaming Compression

For memory-bounded snapshot I/O.

```rust
use tensor_compress::streaming::{StreamingWriter, StreamingReader};

// Write entries one at a time
let mut writer = StreamingWriter::new(file, config)?;
for entry in entries {
    writer.write_entry(&entry)?;
}
writer.finish()?;

// Read entries one at a time (iterator-based)
let reader = StreamingReader::open(file)?;
for entry in reader {
    process(entry?);
}
```

## Streaming TT Decomposition

For processing large vector collections without loading all into memory:

```rust
use tensor_compress::{
    StreamingTTWriter, StreamingTTReader, streaming_tt_similarity_search, TTConfig
};
use std::fs::File;

let config = TTConfig::for_dim(4096);

// Write vectors as TT format incrementally
let file = File::create("embeddings.tt")?;
let mut writer = StreamingTTWriter::new(file, config.clone())?;
for vector in vectors {
    writer.write_vector(&vector)?;
}
writer.finish()?;

// Read back and search
let file = File::open("embeddings.tt")?;
let query_tt = tt_decompose(&query, &config)?;
let results = streaming_tt_similarity_search(file, &query_tt, 10)?;
// Returns top-10 (index, similarity) pairs
```

### Format

Uses a trailer-based header so entry count is known at the end:

```
+------------------+
| Magic (NEUS)     |
+------------------+
| Entry 1 (len+data) |
+------------------+
| Entry 2          |
+------------------+
| ...              |
+------------------+
| Trailer          |  Entry count, config
+------------------+
| Trailer size     |  8 bytes
+------------------+
```

## Incremental Updates

Delta snapshots store only changes since a base snapshot.

```rust
use tensor_compress::incremental::{DeltaBuilder, DeltaChain, apply_delta};

// Create delta
let mut builder = DeltaBuilder::new("base_snapshot_id", sequence);
builder.put("key1", entry1);
builder.delete("key2");
let delta = builder.build();

// Apply delta
let new_snapshot = apply_delta(&base, &delta)?;

// Chain management
let mut chain = DeltaChain::new(base_snapshot);
chain.push(delta1)?;
chain.push(delta2)?;
let value = chain.get("key1");  // Checks chain then base

// Compact when chain grows long
if chain.should_compact(10) {
    let compacted = chain.compact()?;
}
```

## Delta Encoding

For sorted integer sequences (node IDs, timestamps).

```rust
use tensor_compress::{compress_ids, decompress_ids};

let ids: Vec<u64> = (1000..2000).collect();
let compressed = compress_ids(&ids);  // ~100 bytes vs 8000

let restored = decompress_ids(&compressed);
assert_eq!(ids, restored);
```

## Run-Length Encoding

For repeated values.

```rust
use tensor_compress::{rle_encode, rle_decode};

let statuses = vec!["active"; 1000];
let encoded = rle_encode(&statuses);
assert_eq!(encoded.runs(), 1);
```

## API Reference

### Configuration

```rust
pub struct CompressionConfig {
    pub tensor_mode: Option<TensorMode>,
    pub delta_encoding: bool,
    pub rle_encoding: bool,
}

pub enum TensorMode {
    TensorTrain(TTConfig),
}

pub struct TTConfig {
    pub shape: Vec<usize>,   // Tensor shape
    pub max_rank: usize,     // TT-rank bound
    pub tolerance: f32,      // SVD truncation
}
```

### TT Operations

| Function | Description |
|----------|-------------|
| `tt_decompose` | Decompose vector to TT format |
| `tt_decompose_batch` | Parallel batch decomposition |
| `tt_reconstruct` | Reconstruct vector from TT |
| `tt_dot_product` | Dot product in TT space |
| `tt_dot_product_batch` | Batch dot products |
| `tt_cosine_similarity` | Cosine similarity in TT space |
| `tt_cosine_similarity_batch` | Batch cosine similarities |
| `tt_euclidean_distance` | Euclidean distance in TT space |
| `tt_euclidean_distance_batch` | Batch Euclidean distances |
| `tt_norm` | L2 norm of TT vector |
| `tt_scale` | Scale TT vector by constant |

### Streaming Operations

| Function | Description |
|----------|-------------|
| `StreamingWriter::new` | Create streaming writer |
| `StreamingWriter::write_entry` | Write single entry |
| `StreamingWriter::finish` | Finalize with trailer |
| `StreamingReader::open` | Open streaming file |
| `StreamingReader::entry_count` | Total entries |

### Streaming TT Operations

| Function | Description |
|----------|-------------|
| `StreamingTTWriter::new` | Create TT streaming writer |
| `StreamingTTWriter::write_vector` | Decompose and write vector |
| `StreamingTTWriter::write_tt` | Write pre-decomposed TT |
| `StreamingTTWriter::finish` | Finalize with trailer |
| `StreamingTTReader::open` | Open TT streaming file |
| `streaming_tt_similarity_search` | Search streaming TT file |
| `convert_vectors_to_streaming_tt` | Batch convert vectors |
| `read_streaming_tt_all` | Load all TT vectors |

### Delta Operations

| Function | Description |
|----------|-------------|
| `DeltaBuilder::new` | Create delta builder |
| `DeltaBuilder::put/delete` | Record change |
| `apply_delta` | Apply delta to base |
| `merge_deltas` | Merge multiple deltas |
| `diff_snapshots` | Compute delta between snapshots |

## Error Handling

```rust
pub enum TTError {
    ShapeMismatch { dim, shape, product },
    EmptyVector,
    InvalidRank,
    IncompatibleShapes,
    Decomposition(DecomposeError),
}

pub enum FormatError {
    InvalidMagic,
    UnsupportedVersion(u16),
    Serialization(bincode::Error),
    TensorTrain(TTError),
    Io(std::io::Error),
}

pub enum DeltaError {
    BaseNotFound(String),
    SequenceGap { expected, got },
    ChainTooLong { len, max },
    Format(String),
}
```

## Test Coverage

**Unit Tests**: 147 tests in tensor_compress (94.84% line coverage)

| Module | Tests |
|--------|-------|
| tensor_train | 40+ (decompose, reconstruct, similarity, batch ops, edge cases) |
| streaming_tt | 15+ (write, read, roundtrip, similarity search) |
| streaming | 15+ (write, read, roundtrip, large files) |
| incremental | 20+ (delta build, apply, merge, chain) |
| decompose | 15+ (SVD, matrix ops) |
| format | 10+ (serialize, version, sparse) |
| delta/rle | 20+ (varint, runs) |

**Integration Tests**: 14 tests with real engine data

**Fuzz Targets**: 6 targets
- `tt_roundtrip`: TT decomposition/reconstruction
- `tt_config_validation`: Config validation paths
- `tt_metrics`: Property-based metric testing (similarity in [-1,1], distance >= 0)
- `tt_batch`: Batch decomposition consistency
- `tt_serialization`: TTVector serde roundtrip
- `compress_ids`: Delta+varint encoding

**Stress Tests**: 4 tests
- Concurrent TT decomposition (16 threads)
- Large streaming snapshots (10k entries)
- Deep delta chains (100 deltas)
- Compression ratio verification

## Dependencies

- `serde`: Serialization traits
- `bincode`: Binary format
- `thiserror`: Error types
- `rayon`: Parallel batch operations

No external LAPACK/BLAS - pure Rust SVD implementation.
