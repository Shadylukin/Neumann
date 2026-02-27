# Tensor Blob API Reference

Complete type reference, configuration tables, storage key patterns, error types,
and method signatures for the `tensor_blob` crate.

## Core Types

| Type | Description |
| --- | --- |
| `BlobStore` | Main API for storing, retrieving, and managing artifacts |
| `BlobConfig` | Configuration for chunk size, GC intervals, and limits |
| `BlobWriter` | Streaming upload with incremental chunking and hash computation |
| `BlobReader` | Streaming download with chunk-by-chunk reads and verification |
| `Chunk` | Content-addressed data segment with SHA-256 hash |
| `Chunker` | Splits data into fixed-size content-addressable chunks |
| `StreamingHasher` | Incremental SHA-256 computation for large files |
| `GarbageCollector` | Background task for cleaning orphaned chunks |

## Metadata Types

| Type | Description |
| --- | --- |
| `ArtifactMetadata` | Full metadata including filename, size, checksum, links, tags |
| `PutOptions` | Upload options: content type, creator, links, tags, custom metadata, embedding |
| `MetadataUpdates` | Partial updates for filename, content type, custom fields |
| `SimilarArtifact` | Search result with artifact ID, filename, and similarity score |
| `WriteState` | Internal state tracking artifact metadata during streaming upload |

## Statistics Types

| Type | Description |
| --- | --- |
| `BlobStats` | Storage statistics: artifact count, chunk count, dedup ratio, orphaned chunks |
| `GcStats` | GC results: chunks deleted, bytes freed |
| `RepairStats` | Repair results: artifacts checked, chunks verified, refs fixed, orphans deleted |

## Error Types

| Error | Description |
| --- | --- |
| `NotFound` | Artifact does not exist |
| `ChunkMissing` | Referenced chunk not found in storage |
| `ChecksumMismatch` | Data corruption detected during verification |
| `EmptyData` | Cannot store empty artifact |
| `InvalidConfig` | Invalid configuration parameter (e.g., zero chunk size) |
| `InvalidArtifactId` | Malformed artifact ID format |
| `StorageError` | Underlying tensor store error |
| `GraphError` | Graph engine integration error (feature-gated) |
| `VectorError` | Vector engine integration error (feature-gated) |
| `IoError` | I/O error during streaming operations |
| `GcError` | Garbage collection failure |
| `AlreadyExists` | Artifact with given ID already exists |
| `DimensionMismatch` | Embedding dimension mismatch |

## BlobStore Methods

| Method | Description |
| --- | --- |
| `new(store, config)` | Create with configuration (validates config) |
| `start()` | Start background GC task |
| `shutdown()` | Graceful shutdown (sends signal and awaits task) |
| `store()` | Get reference to underlying TensorStore |
| `put(filename, data, options)` | Store bytes, return artifact ID |
| `get(artifact_id)` | Retrieve all bytes |
| `delete(artifact_id)` | Delete artifact and decrement chunk refs |
| `exists(artifact_id)` | Check if artifact exists |
| `writer(filename, options)` | Create streaming upload writer |
| `reader(artifact_id)` | Create streaming download reader |
| `metadata(artifact_id)` | Get artifact metadata |
| `update_metadata(artifact_id, updates)` | Apply metadata updates |
| `set_meta(artifact_id, key, value)` | Set custom metadata field |
| `get_meta(artifact_id, key)` | Get custom metadata field |
| `link(artifact_id, entity)` | Link to entity |
| `unlink(artifact_id, entity)` | Remove link |
| `links(artifact_id)` | Get linked entities |
| `artifacts_for(entity)` | Find artifacts by linked entity |
| `tag(artifact_id, tag)` | Add tag |
| `untag(artifact_id, tag)` | Remove tag |
| `by_tag(tag)` | Find artifacts by tag |
| `list(prefix)` | List artifacts with optional prefix filter |
| `by_content_type(type)` | Find by content type |
| `by_creator(creator)` | Find by creator |
| `verify(artifact_id)` | Verify checksum integrity |
| `repair()` | Repair broken references |
| `gc()` | Run incremental GC |
| `full_gc()` | Run full GC |
| `stats()` | Get storage statistics |
| `set_embedding(id, vec, model)` | Set artifact embedding (feature-gated) |
| `similar(id, k)` | Find k similar artifacts (feature-gated) |
| `search_by_embedding(vec, k)` | Search by embedding vector (feature-gated) |

## BlobWriter Methods

| Method | Description |
| --- | --- |
| `write(data)` | Write chunk of data (buffers until chunk_size reached) |
| `finish()` | Finalize, flush buffer, store metadata, return artifact ID |
| `bytes_written()` | Total bytes written so far |
| `chunks_written()` | Chunks stored so far (not including buffered data) |

## BlobReader Methods

| Method | Description |
| --- | --- |
| `next_chunk()` | Read next chunk, returns `None` when done |
| `read_all()` | Read all remaining data into buffer |
| `read(buf)` | Read into buffer, returns bytes read (for streaming) |
| `verify()` | Verify checksum against stored value (resets read position) |
| `checksum()` | Get expected checksum |
| `total_size()` | Total artifact size |
| `bytes_read()` | Bytes read so far |
| `chunk_count()` | Number of chunks |

## Storage Key Patterns

### Artifact Metadata

Storage key: `_blob:meta:{artifact_id}`

| Field | Type | Description |
| --- | --- | --- |
| `_type` | String | Always `"blob_artifact"` |
| `_id` | String | Unique artifact identifier (UUID v4) |
| `_filename` | String | Original filename |
| `_content_type` | String | MIME type |
| `_size` | Int | Total size in bytes |
| `_checksum` | String | SHA-256 hash of full content (`sha256:{hex}`) |
| `_chunk_size` | Int | Size of each chunk (except possibly last) |
| `_chunk_count` | Int | Number of chunks |
| `_chunks` | Pointers | Ordered list of chunk keys |
| `_created` | Int | Unix timestamp (seconds) |
| `_modified` | Int | Unix timestamp (seconds) |
| `_created_by` | String | Creator identity |
| `_linked_to` | Pointers | Linked entity IDs |
| `_tags` | Pointers | Applied tags (prefixed with `tag:`) |
| `_meta:*` | String | Custom metadata fields |
| `_embedding` | Vector/Sparse | Optional embedding (sparse if >50% zeros) |
| `_embedded_model` | String | Embedding model name |

### Chunk Data

Storage key: `_blob:chunk:sha256:{64_hex_chars}`

| Field | Type | Description |
| --- | --- | --- |
| `_type` | String | Always `"blob_chunk"` |
| `_data` | Bytes | Raw chunk data |
| `_size` | Int | Chunk size in bytes |
| `_refs` | Int | Reference count for deduplication |
| `_created` | Int | Unix timestamp (seconds) |

## Configuration

| Option | Default | Description |
| --- | --- | --- |
| `chunk_size` | 1 MB (1,048,576 bytes) | Size of each chunk in bytes |
| `max_artifact_size` | None (unlimited) | Maximum artifact size limit |
| `max_artifacts` | None (unlimited) | Maximum number of artifacts |
| `gc_interval` | 5 minutes (300s) | Background GC check frequency |
| `gc_batch_size` | 100 | Chunks processed per GC cycle |
| `gc_min_age` | 1 minute (60s) | Minimum age before GC eligible |
| `default_content_type` | `application/octet-stream` | Default MIME type |

```rust
let config = BlobConfig::new()
    .with_chunk_size(1024 * 1024)
    .with_gc_interval(Duration::from_secs(300))
    .with_gc_batch_size(100)
    .with_gc_min_age(Duration::from_secs(3600))
    .with_max_artifact_size(100 * 1024 * 1024);
```

### Configuration Validation

```rust
pub fn validate(&self) -> Result<()> {
    if self.chunk_size == 0 {
        return Err(BlobError::InvalidConfig("chunk_size must be > 0"));
    }
    if self.gc_batch_size == 0 {
        return Err(BlobError::InvalidConfig("gc_batch_size must be > 0"));
    }
    Ok(())
}
```

## Garbage Collection Modes

| Method | Description | Age Requirement | Reference Source |
| --- | --- | --- | --- |
| `gc()` | Incremental: processes `batch_size` chunks per cycle | Respects `min_age` | Uses stored `_refs` field |
| `full_gc()` | Full: recounts all references from artifacts | Ignores age | Rebuilds from artifact metadata |

## Shell Commands

```text
BLOB PUT 'filename' 'data'              Store inline data
BLOB PUT 'filename' FROM 'path'         Store from file path
BLOB GET 'artifact_id'                  Retrieve data
BLOB GET 'artifact_id' TO 'path'        Write to file
BLOB DELETE 'artifact_id'               Delete artifact
BLOB INFO 'artifact_id'                 Show metadata
BLOB VERIFY 'artifact_id'               Verify integrity

BLOB LINK 'artifact_id' TO 'entity'     Link to entity
BLOB UNLINK 'artifact_id' FROM 'entity' Remove link
BLOB TAG 'artifact_id' 'tag'            Add tag
BLOB UNTAG 'artifact_id' 'tag'          Remove tag

BLOB META SET 'artifact_id' 'key' 'value'  Set custom metadata
BLOB META GET 'artifact_id' 'key'          Get custom metadata

BLOB GC                                 Run incremental GC
BLOB GC FULL                            Full garbage collection
BLOB REPAIR                             Repair broken references
BLOB STATS                              Show storage statistics

BLOBS                                   List all artifacts
BLOBS FOR 'entity'                      Find by linked entity
BLOBS BY TAG 'tag'                      Find by tag
BLOBS WHERE TYPE = 'content/type'       Find by content type
BLOBS SIMILAR TO 'artifact_id' LIMIT n  Find similar (requires embeddings)
```

## Edge Cases and Gotchas

| Scenario | Behavior |
| --- | --- |
| Empty data | Rejected with `BlobError::EmptyData` |
| Exceeding `max_artifact_size` | Returns `InvalidConfig` error |
| Concurrent deduplication | Ref count may be off by one; mitigate with periodic `full_gc()` |
| GC during upload | Chunks older than `min_age` may be collected; set `gc_min_age` longer than max upload time |
| Checksum vs chunk hash | `_checksum` is SHA-256 of entire file; chunk keys are per-chunk hashes -- not comparable |
| Sparse embedding | Embeddings with >50% zeros stored in sparse format automatically |

## Dependencies

| Crate | Purpose |
| --- | --- |
| `tensor_store` | Key-value storage layer |
| `tokio` | Async runtime for streaming and background GC |
| `sha2` | SHA-256 hashing for content addressing |
| `uuid` | Artifact ID generation (UUID v4) |

## Related Modules

| Module | Relationship |
| --- | --- |
| `tensor_store` | Underlying key-value storage for chunks and metadata |
| `query_router` | Executes BLOB commands from parsed queries |
| `neumann_shell` | Interactive CLI for blob operations |
| `vector_engine` | Optional semantic search via embeddings |
| `graph_engine` | Optional entity linking via graph edges |

## See Also

- [Blob Storage Design](../../explanation/blob-storage.md) -- content-addressable
  architecture, streaming I/O model, and async design rationale
- [How to: Store and Retrieve Blobs](../../how-to/store-retrieve-blobs.md) --
  practical examples for uploading, downloading, streaming, and managing blobs
