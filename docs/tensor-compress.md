# Tensor Compress

Module 8 of Neumann. Provides bespoke compression algorithms optimized for tensor data.

## Design Principles

1. **Tensor-Aware**: Compression understands tensor semantics, not generic bytes
2. **Lossless Where It Matters**: RLE and delta encoding are lossless; quantization is configurable
3. **Composable**: Algorithms can be combined via configuration
4. **Zero Dependencies on Engines**: Pure compression library, usable standalone

## Quick Start

```rust
use tensor_compress::{CompressionConfig, QuantMode};

// Configure compression
let config = CompressionConfig {
    vector_quantization: Some(QuantMode::Int8),  // 4x compression
    delta_encoding: true,                        // For sorted IDs
    rle_encoding: true,                          // For repeated values
};

// Use with TensorStore
store.save_snapshot_compressed("data.bin", config)?;
let loaded = TensorStore::load_snapshot_compressed("data.bin")?;
```

## Compression Algorithms

### Vector Quantization

Reduces f32 embedding dimensions to smaller representations.

| Mode | Compression | Error | Use Case |
|------|-------------|-------|----------|
| Int8 | 4x | ~1% max | General embeddings |
| Binary | 32x | Lossy | High-dimensional similarity |

```rust
use tensor_compress::{quantize_int8, dequantize_int8};

let embedding = vec![0.1, 0.2, 0.3, 0.4];
let quantized = quantize_int8(&embedding)?;
let restored = dequantize_int8(&quantized);

// quantized.data is Vec<i8> - 4x smaller
// restored is approximately equal to embedding
```

### Delta Encoding

Compresses sorted integer sequences by storing differences.

```rust
use tensor_compress::{compress_ids, decompress_ids};

// Sequential IDs compress extremely well
let ids: Vec<u64> = (1000..2000).collect();
let compressed = compress_ids(&ids);  // ~100 bytes vs 8000 bytes

let restored = decompress_ids(&compressed);
assert_eq!(ids, restored);
```

| Data Pattern | Compression Ratio |
|--------------|-------------------|
| Sequential IDs | 8x |
| Small gaps | 4-6x |
| Random | 1x (no benefit) |

### Run-Length Encoding

Compresses repeated values by storing (value, count) pairs.

```rust
use tensor_compress::{rle_encode, rle_decode};

// Repeated status values
let statuses = vec!["active"; 1000];
let encoded = rle_encode(&statuses);

assert_eq!(encoded.runs(), 1);  // Just one run
let restored = rle_decode(&encoded);
assert_eq!(statuses, restored);
```

## Snapshot Format

### Version 2 (Compressed)

```
+----------------+
| Header (NEUM)  |  Magic bytes, version, config
+----------------+
| Entry 1        |  Key + compressed fields
+----------------+
| Entry 2        |
+----------------+
| ...            |
+----------------+
```

The header includes magic bytes `NEUM` for format detection.

### Compression Selection

Fields are compressed based on heuristics:

| Field Type | Key Pattern | Compression |
|------------|-------------|-------------|
| Vector | `emb:*` or `_embedding` | Quantization |
| Vector | `*_ids` or sorted integers | Delta + varint |
| Scalar | Repeated values | RLE |
| Other | - | Raw bincode |

## API Reference

### Configuration

```rust
pub struct CompressionConfig {
    /// Vector quantization mode (None = no quantization)
    pub vector_quantization: Option<QuantMode>,
    /// Enable delta encoding for sorted ID lists
    pub delta_encoding: bool,
    /// Enable RLE for repeated values
    pub rle_encoding: bool,
}

pub enum QuantMode {
    Int8,    // 4x reduction, ~1% error
    Binary,  // 32x reduction, lossy
}
```

### Core Functions

| Function | Description |
|----------|-------------|
| `quantize_int8(Vec<f32>)` | Quantize to int8 with min-max scaling |
| `dequantize_int8(QuantizedInt8)` | Restore f32 from int8 |
| `quantize_binary(Vec<f32>)` | Binary quantization (sign bit) |
| `dequantize_binary(QuantizedBinary)` | Restore f32 from binary |
| `compress_ids(Vec<u64>)` | Delta + varint encode |
| `decompress_ids(Vec<u8>)` | Restore u64 sequence |
| `rle_encode<T>(Vec<T>)` | Run-length encode |
| `rle_decode<T>(RleEncoded<T>)` | Restore from RLE |

## Performance Characteristics

| Operation | Time (768-dim vector) |
|-----------|----------------------|
| quantize_int8 | ~500 ns |
| dequantize_int8 | ~300 ns |
| quantize_binary | ~100 ns |
| compress_ids (10k sequential) | ~50 us |
| rle_encode (100k values) | ~2 ms |

## Error Handling

```rust
pub enum QuantizationError {
    EmptyVector,
    InvalidBinaryLength { expected, actual },
}

pub enum FormatError {
    InvalidMagic,
    UnsupportedVersion(u16),
    Serialization(bincode::Error),
    Quantization(QuantizationError),
}
```

## Integration with TensorStore

```rust
use tensor_store::TensorStore;
use tensor_compress::{CompressionConfig, QuantMode};

// Save with compression
let config = CompressionConfig {
    vector_quantization: Some(QuantMode::Int8),
    delta_encoding: true,
    rle_encoding: true,
};
store.save_snapshot_compressed("data.bin", config)?;

// Load (auto-detects format)
let store = TensorStore::load_snapshot_compressed("data.bin")?;
```

## Test Coverage

| Test Category | What It Verifies |
|---------------|------------------|
| Quantization roundtrip | Error bounds maintained |
| Delta encoding | Lossless for all patterns |
| RLE | Lossless for all types |
| Format detection | Magic bytes validation |
| Integration | TensorStore save/load |

Coverage: 97.94%

## Dependencies

- `serde`: Serialization traits
- `bincode`: Binary format
- `thiserror`: Error types

## Future Considerations

Not implemented (out of scope):

- **Product Quantization**: Higher compression for very large embeddings
- **Streaming Compression**: Process without loading full snapshot
- **Incremental Updates**: Append-only compressed format
