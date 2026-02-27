# tensor_compress Benchmarks

The tensor_compress crate provides compression algorithms optimized for tensor
data: Tensor Train decomposition, delta encoding, sparse vectors, and run-length
encoding.

<!-- BENCH:START -->
## Tensor Train Decomposition (primary compression method)

| Operation | Time | Peak RAM |
| --- | --- | --- |
| tt_decompose_256d | ~50 us | 41.8 KB |
| tt_decompose_1024d | ~80 us | 60.9 KB |
| tt_decompose_4096d | ~120 us | 137.5 KB |
| tt_reconstruct_4096d | ~1.2 ms | 67.9 KB |
| tt_dot_product_4096d | ~400 ns | 69.2 KB |
| tt_cosine_similarity_4096d | ~1 us | 69.2 KB |

## Delta Encoding (10K sequential IDs)

| Operation | Time | Throughput | Peak RAM |
| --- | --- | --- | --- |
| compress_ids | 8.0 us | 1.25M IDs/s | ~210 KB |
| decompress_ids | 33 us | 303K IDs/s | ~100 KB |

## Run-Length Encoding (100K values)

| Operation | Time | Throughput | Peak RAM |
| --- | --- | --- | --- |
| rle_encode | 29 us | 3.4M values/s | ~445 KB |
| rle_decode | 38 us | 2.6M values/s | ~833 KB |

## Compression Ratios

| Data Type | Technique | Ratio | Lossless |
| --- | --- | --- | --- |
| 4096-dim embeddings | Tensor Train | 10-20x | No (<1% error) |
| 1024-dim embeddings | Tensor Train | 4-8x | No (<1% error) |
| Sparse vectors | Native sparse | 3-32x | Yes |
| Sequential IDs | Delta + varint | 4-8x | Yes |
| Repeated values | RLE | 2-100x | Yes |
<!-- BENCH:END -->

## Analysis

- **TT decomposition**: Achieves 10-20x compression for high-dimensional
  embeddings (4096+)
- **TT operations in compressed space**: Dot product and cosine similarity
  computed directly in TT format without full reconstruction
- **Delta encoding**: Asymmetric - compression is 4x faster than decompression
- **Sparse format**: Efficient for vectors with >50% zeros, stores only non-zero
  positions/values
- **RLE**: Best for highly repeated data (status columns, category IDs)
- **Memory efficiency**: All operations use < 1 MB for typical data sizes
- **Integration**: Use `SAVE COMPRESSED` in shell or
  `save_snapshot_compressed()` API

## Usage Recommendations

| Data Characteristics | Recommended Compression |
| --- | --- |
| High-dimensional embeddings (1024+) | Tensor Train |
| Sparse embeddings (>50% zeros) | Native sparse format |
| Sequential IDs (node IDs, row IDs) | Delta + varint |
| Categorical columns with repeats | RLE |
| Mixed data snapshots | Composite (auto-detect) |
