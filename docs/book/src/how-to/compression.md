# Configure and Use Compression

## Goal

Choose the right compression algorithm, tune parameters for your workload,
and use streaming and delta operations for efficient I/O.

> **See also:** [Tensor Compress API](../reference/api/tensor-compress.md) |
> [Compression Algorithms](../explanation/compression-algorithms.md) |
> [Architecture](../reference/api/tensor-compress.md)

## Choose a Compression Configuration

### For search and retrieval (similarity queries)

```rust
use tensor_compress::{tt_decompose, TTConfig};

let config = TTConfig::for_dim(768)?;  // max_rank=8, tolerance=1e-4
let tt = tt_decompose(&embedding, &config)?;
```

This is the default balanced mode: good compression (10x at 768-dim) with high
accuracy (~99% cosine similarity preservation).

### For archival and cold storage

```rust
let config = TTConfig::high_compression(768)?;  // max_rank=4, tolerance=1e-2
```

Produces smaller files (2-3x more compression than balanced) at the cost of
slightly reduced query accuracy. Use this for data that is rarely queried.

### For real-time applications

```rust
let config = TTConfig::high_accuracy(768)?;  // max_rank=16, tolerance=1e-6
```

Minimal information loss (<0.1% error). The TT vectors are larger but
similarity computations remain fast.

## Decompose and Reconstruct Vectors

### Single vector

```rust
use tensor_compress::{tt_decompose, tt_reconstruct, tt_cosine_similarity, TTConfig};

let embedding: Vec<f32> = get_embedding();  // 4096-dim
let config = TTConfig::for_dim(4096)?;

// Decompose
let tt = tt_decompose(&embedding, &config)?;
println!("Compression: {:.1}x", tt.compression_ratio());
println!("Storage: {} floats", tt.storage_size());
println!("Max rank: {}", tt.max_rank());

// Reconstruct
let restored = tt_reconstruct(&tt);

// Compute similarity without reconstruction
let tt2 = tt_decompose(&other_embedding, &config)?;
let sim = tt_cosine_similarity(&tt, &tt2)?;
```

### Batch operations

Batch operations use rayon for parallel processing when handling 4+ vectors:

```rust
use tensor_compress::{tt_decompose_batch, tt_cosine_similarity_batch, TTConfig};

let vectors: Vec<Vec<f32>> = load_embeddings();
let config = TTConfig::for_dim(4096)?;

// Batch decompose (parallel for 4+ vectors)
let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
let tts = tt_decompose_batch(&refs, &config)?;

// Batch similarity search
let query_tt = &tts[0];
let similarities = tt_cosine_similarity_batch(query_tt, &tts[1..])?;

// Find top-k
let mut indexed: Vec<_> = similarities.iter().enumerate().collect();
indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
let top_5: Vec<_> = indexed.iter().take(5).collect();
```

### Batch size optimization

Below the parallel threshold (4), sequential execution is faster due to thread
spawn overhead:

```rust
let small_batch = tt_decompose_batch(&vectors[..3], &config);  // Sequential
let large_batch = tt_decompose_batch(&vectors, &config);        // Parallel if >= 4
```

## Use Streaming I/O

### Write entries incrementally

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
println!("Entry count: {}", reader.entry_count());
for entry in reader {
    process(entry?);
}
```

### Stream TT-compressed vectors

```rust
use tensor_compress::streaming_tt::{StreamingTTWriter, StreamingTTReader,
    streaming_tt_similarity_search};

// Create streaming TT file
let config = TTConfig::for_dim(768)?;
let mut writer = StreamingTTWriter::new(file, config.clone())?;

for vector in vectors {
    writer.write_vector(&vector)?;  // Decompose on-the-fly
}
writer.finish()?;

// Similarity search without loading all into memory
let query_tt = tt_decompose(&query, &config)?;
let top_10 = streaming_tt_similarity_search(file, &query_tt, 10)?;
// Returns Vec<(index, similarity)> sorted by descending similarity
```

### Memory-efficient processing

```rust
// Bad: Load all, then process
let all_vectors = read_streaming_tt_all(file)?;  // Loads all into memory

// Good: Stream process
for tt in StreamingTTReader::open(file)? {
    process(tt?);  // One at a time
}

// Best: Use streaming search
let results = streaming_tt_similarity_search(file, &query_tt, 10)?;
```

### Convert and merge streaming files

```rust
use tensor_compress::streaming::{convert_to_streaming, read_streaming_to_snapshot,
    merge_streaming};

// Convert non-streaming snapshot to streaming format
let count = convert_to_streaming(&snapshot, output_file)?;

// Read streaming format into full snapshot (for compatibility)
let snapshot = read_streaming_to_snapshot(file)?;

// Merge multiple streaming snapshots
let count = merge_streaming(vec![file1, file2, file3], output, config)?;
```

## Work with Delta Snapshots

### Create and apply deltas

```rust
use tensor_compress::incremental::{DeltaBuilder, DeltaChain, apply_delta,
    merge_deltas, diff_snapshots};

// Create delta
let mut builder = DeltaBuilder::new("base_snapshot_id", sequence);
builder.put("key1", entry1);
builder.delete("key2");
let delta = builder.build();

// Apply delta
let new_snapshot = apply_delta(&base, &delta)?;
```

### Manage delta chains

```rust
let mut chain = DeltaChain::new(base_snapshot);
chain.push(delta1)?;
chain.push(delta2)?;
let value = chain.get("key1");  // Checks chain then base

// Compact when chain grows long
if chain.should_compact(10) {
    let compacted = chain.compact()?;
}
```

### Compare snapshots

```rust
// Compare two snapshots
let delta = diff_snapshots(&old_snapshot, &new_snapshot, "old_id")?;

// Merge multiple deltas into one
let merged = merge_deltas(&[delta1, delta2, delta3])?;
```

### Delta compaction strategy

```rust
let mut chain = DeltaChain::new(base);

// After N deltas or M total changes
if chain.len() >= 10 || total_changes >= 10000 {
    let new_base = chain.compact()?;
    chain = DeltaChain::new(new_base);
}
```

## Use Lossless Compression for IDs

### Compress sorted integer sequences

```rust
use tensor_compress::{compress_ids, decompress_ids};

let ids: Vec<u64> = (1000..2000).collect();
let compressed = compress_ids(&ids);  // ~100 bytes vs 8000

let restored = decompress_ids(&compressed);
assert_eq!(ids, restored);
```

### Use RLE for repeated values

```rust
use tensor_compress::{rle_encode, rle_decode};

let statuses = vec!["active"; 1000];
let encoded = rle_encode(&statuses);
assert_eq!(encoded.runs(), 1);  // Single run

// Storage: 1 string + 1 u32 = ~12 bytes vs 6000+ bytes
```

### Use sparse format for vectors with many zeros

```rust
use tensor_compress::{compress_sparse, compress_dense_as_sparse,
    should_use_sparse, should_use_sparse_threshold};

// Direct sparse compression
let positions = vec![0, 50, 99];
let values = vec![1.0, 2.0, 3.0];
let compressed = compress_sparse(100, &positions, &values);

// Auto-detect and compress
if should_use_sparse_threshold(&vector, 0.5) {
    let compressed = compress_dense_as_sparse(&vector);
}

// Check if sparse is beneficial
if should_use_sparse(dimension, non_zero_count) {
    // Use sparse format
}
```

## Configure Full CompressionConfig

```rust
use tensor_compress::CompressionConfig;

// All encodings enabled with balanced TT
let config = CompressionConfig::balanced(768);

// High compression mode
let config = CompressionConfig::high_compression();

// High accuracy mode
let config = CompressionConfig::high_accuracy(768);
```

The `CompressionConfig` controls all three compression paths:
- `tensor_mode`: TT compression for embedding vectors
- `delta_encoding`: Delta + varint for sorted ID lists
- `rle_encoding`: Run-length encoding for repeated values

The `compress_vector` function automatically selects the best format based on
key patterns (e.g., keys starting with `emb:`) and field names (e.g.,
`_embedding`, `vector`).
