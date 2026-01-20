# tensor_blob Benchmarks

The tensor_blob crate provides S3-style chunked blob storage with
content-addressable chunks, garbage collection, and integrity verification.

## Overview

tensor_blob focuses on correctness and durability over raw throughput.
Performance characteristics depend heavily on:

- Chunk size configuration
- Storage backend (memory vs disk)
- Network conditions for streaming operations

## Expected Performance Characteristics

| Operation | Complexity | Notes |
| --- | --- | --- |
| Put (upload) | O(size / chunk_size) | Linear with data size |
| Get (download) | O(size / chunk_size) | Linear with data size |
| Delete | O(chunk_count) | Removes metadata + orphan detection |
| GC | O(total_chunks) | Full chunk scan |
| Verify | O(size) | Re-hash entire blob |
| Repair | O(corrupted_chunks) | Only processes damaged chunks |

## Chunk Deduplication

Identical content shares chunks via SHA-256 content addressing:

- **Duplicate blobs**: Store once, reference count tracked
- **Partial overlap**: Shared chunks deduplicated at chunk boundaries
- **Storage savings**: Depends on data redundancy

## Garbage Collection

| Operation | Behavior |
| --- | --- |
| `gc()` | Returns `GcStats { deleted, freed_bytes }` |
| Orphan detection | Marks unreferenced chunks |
| Active upload protection | GC skips in-progress uploads |

## Streaming Operations

| API | Use Case |
| --- | --- |
| `BlobWriter` | Streaming upload, bounded memory |
| `BlobReader::next_chunk()` | Streaming download, chunk-by-chunk |
| `get_full()` | Small blobs (<10MB), loads to memory |

## Configuration Impact

| Setting | Impact |
| --- | --- |
| Larger chunk_size | Fewer chunks, less overhead, less dedup |
| Smaller chunk_size | More chunks, more overhead, better dedup |
| Recommended | 1-4 MB chunks for most workloads |

## Integration Notes

- Blob store persists to TensorStore
- Metadata includes checksum, size, creation time
- Links enable blob-to-graph entity relationships
- Tags support blob categorization and search

## Benchmarking Blob Operations

```bash
# Run blob-specific benchmarks (if available)
cargo bench --package tensor_blob

# For custom benchmarking, use the streaming API:
# - Measure upload throughput with BlobWriter
# - Measure download throughput with BlobReader
# - Test GC performance with various orphan ratios
```
