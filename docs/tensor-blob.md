# Tensor Blob

Module 11 of Neumann. S3-style object storage for large artifacts using content-addressable chunked storage with tensor-native metadata.

## Design Principles

1. **Content-Addressable**: Chunks keyed by SHA-256 hash for automatic deduplication
2. **Tensor-Native**: Metadata participates in graph/relational/vector queries
3. **Streaming**: Large files never fully in memory via BlobWriter/BlobReader
4. **Linked Artifacts**: Objects connect to entities via graph edges
5. **Garbage Collected**: Orphaned chunks cleaned up automatically
6. **Async-First**: All I/O operations are async via Tokio

## Quick Start

```rust
use tensor_blob::{BlobStore, BlobConfig, PutOptions};
use tensor_store::TensorStore;

// Create blob store
let store = TensorStore::new();
let blob = BlobStore::new(store, BlobConfig::default()).await?;

// Store an artifact
let artifact_id = blob.put(
    "report.pdf",
    &file_bytes,
    PutOptions::new()
        .with_created_by("user:alice")
        .with_tag("quarterly")
        .with_link("task:123"),
).await?;

// Retrieve it
let data = blob.get(&artifact_id).await?;

// Get metadata
let meta = blob.metadata(&artifact_id).await?;
println!("Size: {} bytes, Chunks: {}", meta.size, meta.chunk_count);
```

## Streaming API

For large files that shouldn't be loaded entirely into memory:

```rust
// Streaming upload
let mut writer = blob.writer("large_file.bin", PutOptions::default()).await?;
for chunk in file_chunks {
    writer.write(&chunk).await?;
}
let artifact_id = writer.finish().await?;

// Streaming download
let mut reader = blob.reader(&artifact_id).await?;
while let Some(chunk) = reader.next_chunk().await? {
    process_chunk(&chunk);
}

// Or read into a buffer
let mut buf = [0u8; 4096];
let bytes_read = reader.read(&mut buf).await?;
```

## Configuration

```rust
use tensor_blob::{BlobConfig, GcConfig};
use std::time::Duration;

let config = BlobConfig::new()
    .with_chunk_size(1024 * 1024)           // 1 MB chunks (default)
    .with_default_content_type("application/octet-stream")
    .with_gc_interval(Duration::from_secs(300))  // GC every 5 minutes
    .with_gc_batch_size(100)                     // Process 100 chunks per cycle
    .with_gc_min_age(Duration::from_secs(3600))  // Only GC chunks > 1 hour old
    .with_max_artifact_size(100 * 1024 * 1024);  // 100 MB max

let blob = BlobStore::new(store, config).await?;
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `chunk_size` | 1 MB | Size of each chunk |
| `default_content_type` | `application/octet-stream` | Default MIME type |
| `gc_interval` | 5 minutes | Background GC frequency |
| `gc_batch_size` | 100 | Chunks processed per GC cycle |
| `gc_min_age` | 1 hour | Minimum age before GC eligible |
| `max_artifact_size` | None | Maximum artifact size limit |

## Metadata Management

### PutOptions

```rust
let options = PutOptions::new()
    .with_content_type("application/pdf")
    .with_created_by("user:alice")
    .with_link("task:123")
    .with_links(vec!["project:alpha".to_string()])
    .with_tag("quarterly")
    .with_tags(vec!["finance".to_string(), "report".to_string()])
    .with_meta("author", "Alice")
    .with_embedding(embedding_vec, "text-embedding-3-small");
```

### Updating Metadata

```rust
use tensor_blob::MetadataUpdates;

// Update specific fields
let updates = MetadataUpdates::new()
    .with_filename("renamed.pdf")
    .with_content_type("application/pdf")
    .set_meta("version", "2")
    .delete_meta("draft");

blob.update_metadata(&artifact_id, updates).await?;

// Or update individual fields
blob.set_meta(&artifact_id, "author", "Bob").await?;
let author = blob.get_meta(&artifact_id, "author").await?;
```

### ArtifactMetadata

```rust
let meta = blob.metadata(&artifact_id).await?;

println!("ID: {}", meta.id);
println!("Filename: {}", meta.filename);
println!("Content-Type: {}", meta.content_type);
println!("Size: {} bytes", meta.size);
println!("Checksum: {}", meta.checksum);
println!("Chunks: {} x {} bytes", meta.chunk_count, meta.chunk_size);
println!("Created by: {}", meta.created_by);
println!("Created: {}", meta.created);
println!("Modified: {}", meta.modified);
println!("Links: {:?}", meta.linked_to);
println!("Tags: {:?}", meta.tags);
println!("Custom: {:?}", meta.custom);
println!("Has embedding: {}", meta.has_embedding);
```

## Entity Linking

Connect artifacts to other entities in the system:

```rust
// Link artifact to entities
blob.link(&artifact_id, "user:alice").await?;
blob.link(&artifact_id, "task:123").await?;

// Get links for an artifact
let links = blob.links(&artifact_id).await?;
// ["user:alice", "task:123"]

// Find artifacts linked to an entity
let artifacts = blob.artifacts_for("user:alice").await?;

// Unlink
blob.unlink(&artifact_id, "user:alice").await?;
```

## Tagging

```rust
// Add tags
blob.tag(&artifact_id, "important").await?;
blob.tag(&artifact_id, "quarterly").await?;

// Find artifacts by tag
let important_files = blob.by_tag("important").await?;

// Remove tag
blob.untag(&artifact_id, "important").await?;
```

## Queries

```rust
// List all artifacts
let all = blob.list(None).await?;

// List with prefix filter
let reports = blob.list(Some("report")).await?;

// Find by content type
let pdfs = blob.by_content_type("application/pdf").await?;

// Find by creator
let alice_files = blob.by_creator("user:alice").await?;
```

## Semantic Search

With the `vector` feature enabled:

```rust
// Set embedding for artifact
blob.set_embedding(&artifact_id, embedding, "text-embedding-3-small").await?;

// Find similar artifacts
let similar = blob.similar(&artifact_id, 10).await?;
for result in similar {
    println!("{}: {} (similarity: {:.3})",
        result.id, result.filename, result.similarity);
}

// Search by embedding
let results = blob.search_by_embedding(&query_embedding, 10).await?;
```

## Integrity and Garbage Collection

### Verification

```rust
// Verify single artifact
let valid = blob.verify(&artifact_id).await?;

// Repair broken references
let stats = blob.repair().await?;
println!("Checked {} artifacts, fixed {} refs, deleted {} orphans",
    stats.artifacts_checked, stats.refs_fixed, stats.orphans_deleted);
```

### Garbage Collection

```rust
// Run incremental GC (respects min_age)
let stats = blob.gc().await?;
println!("Deleted {} chunks, freed {} bytes", stats.deleted, stats.freed_bytes);

// Full GC (recounts all references, ignores min_age)
let stats = blob.full_gc().await?;

// Background GC (runs automatically)
blob.start().await?;
// ... use blob store ...
blob.shutdown().await?;
```

## Statistics

```rust
let stats = blob.stats().await?;

println!("Artifacts: {}", stats.artifact_count);
println!("Chunks: {}", stats.chunk_count);
println!("Total bytes: {}", stats.total_bytes);
println!("Unique bytes: {}", stats.unique_bytes);
println!("Dedup ratio: {:.1}%", stats.dedup_ratio * 100.0);
println!("Orphaned chunks: {}", stats.orphaned_chunks);
```

## Integration with Query Router

```rust
use query_router::QueryRouter;

let mut router = QueryRouter::new();
router.init_blob()?;

// Store artifact
router.execute_parsed("BLOB PUT 'report.pdf' 'data here'")?;

// Get artifact
router.execute_parsed("BLOB GET 'artifact:uuid'")?;

// Metadata operations
router.execute_parsed("BLOB INFO 'artifact:uuid'")?;
router.execute_parsed("BLOB LINK 'artifact:uuid' TO 'task:123'")?;
router.execute_parsed("BLOB TAG 'artifact:uuid' 'important'")?;

// Queries
router.execute_parsed("BLOBS")?;                        // List all
router.execute_parsed("BLOBS FOR 'task:123'")?;         // By entity
router.execute_parsed("BLOBS BY TAG 'important'")?;     // By tag
router.execute_parsed("BLOBS WHERE TYPE = 'text/plain'")?;  // By content type

// Maintenance
router.execute_parsed("BLOB GC")?;
router.execute_parsed("BLOB GC FULL")?;
router.execute_parsed("BLOB REPAIR")?;
router.execute_parsed("BLOB STATS")?;
```

## Shell Commands

```
BLOB PUT 'filename' 'data'              Store inline data
BLOB PUT 'filename' FROM 'path'         Store from file path
BLOB PUT 'filename' 'data' LINK 'entity' TAG 'tag'  With options

BLOB GET 'artifact_id'                  Retrieve data
BLOB GET 'artifact_id' TO 'path'        Write to file

BLOB DELETE 'artifact_id'               Delete artifact
BLOB INFO 'artifact_id'                 Show metadata
BLOB VERIFY 'artifact_id'               Verify integrity

BLOB LINK 'artifact_id' TO 'entity'     Link to entity
BLOB UNLINK 'artifact_id' FROM 'entity' Remove link
BLOB LINKS 'artifact_id'                List links

BLOB TAG 'artifact_id' 'tag'            Add tag
BLOB UNTAG 'artifact_id' 'tag'          Remove tag

BLOB META SET 'artifact_id' 'key' 'value'  Set custom metadata
BLOB META GET 'artifact_id' 'key'          Get custom metadata

BLOB GC                                 Run garbage collection
BLOB GC FULL                            Full garbage collection
BLOB REPAIR                             Repair broken references
BLOB STATS                              Show storage statistics

BLOBS                                   List all artifacts
BLOBS 'prefix'                          List with prefix filter
BLOBS FOR 'entity'                      Find by linked entity
BLOBS BY TAG 'tag'                      Find by tag
BLOBS WHERE TYPE = 'content/type'       Find by content type
BLOBS SIMILAR TO 'artifact_id' LIMIT n  Find similar (requires embeddings)
```

## Content-Addressable Chunking

Artifacts are split into chunks using a configurable chunk size (default 1 MB). Each chunk is stored with a key derived from its SHA-256 hash:

```
_blob:chunk:sha256:a1b2c3d4...
```

This provides automatic deduplication:
- Identical chunks across artifacts are stored once
- Reference counting tracks chunk usage
- Garbage collection removes unreferenced chunks

### Deduplication Example

```rust
let data = vec![0u8; 10_000];

// Store same data twice
blob.put("file1.bin", &data, PutOptions::default()).await?;
blob.put("file2.bin", &data, PutOptions::default()).await?;

let stats = blob.stats().await?;
// stats.chunk_count = 1 (deduplicated)
// stats.dedup_ratio > 0.0
```

## API Reference

### BlobStore

| Method | Description |
|--------|-------------|
| `new(store, config)` | Create with config |
| `start()` | Start background GC |
| `shutdown()` | Graceful shutdown |
| `put(filename, data, options)` | Store bytes |
| `get(artifact_id)` | Get all bytes |
| `delete(artifact_id)` | Delete artifact |
| `exists(artifact_id)` | Check existence |
| `writer(filename, options)` | Streaming upload |
| `reader(artifact_id)` | Streaming download |
| `metadata(artifact_id)` | Get metadata |
| `update_metadata(artifact_id, updates)` | Update metadata |
| `set_meta(artifact_id, key, value)` | Set custom field |
| `get_meta(artifact_id, key)` | Get custom field |
| `link(artifact_id, entity)` | Link to entity |
| `unlink(artifact_id, entity)` | Unlink from entity |
| `links(artifact_id)` | Get links |
| `artifacts_for(entity)` | Find by entity |
| `tag(artifact_id, tag)` | Add tag |
| `untag(artifact_id, tag)` | Remove tag |
| `by_tag(tag)` | Find by tag |
| `list(prefix)` | List artifacts |
| `by_content_type(type)` | Find by type |
| `by_creator(creator)` | Find by creator |
| `verify(artifact_id)` | Verify integrity |
| `repair()` | Repair references |
| `gc()` | Incremental GC |
| `full_gc()` | Full GC |
| `stats()` | Storage statistics |

### BlobWriter

| Method | Description |
|--------|-------------|
| `write(data)` | Write chunk of data |
| `finish()` | Finalize and return artifact ID |
| `bytes_written()` | Total bytes so far |
| `chunks_written()` | Chunks stored so far |

### BlobReader

| Method | Description |
|--------|-------------|
| `next_chunk()` | Read next chunk |
| `read_all()` | Read all remaining |
| `read(buf)` | Read into buffer |
| `verify()` | Verify checksum |
| `checksum()` | Expected checksum |
| `total_size()` | Total artifact size |
| `bytes_read()` | Bytes read so far |
| `chunk_count()` | Number of chunks |

## Error Handling

```rust
use tensor_blob::{BlobError, Result};

match blob.get(&artifact_id).await {
    Ok(data) => process(data),
    Err(BlobError::NotFound(id)) => println!("Artifact not found: {}", id),
    Err(BlobError::ChunkMissing(key)) => println!("Missing chunk: {}", key),
    Err(BlobError::ChecksumMismatch { expected, actual }) => {
        println!("Corruption detected: {} != {}", expected, actual);
    }
    Err(BlobError::EmptyData) => println!("Cannot store empty data"),
    Err(e) => println!("Error: {}", e),
}
```

### Error Types

| Error | Description |
|-------|-------------|
| `NotFound` | Artifact does not exist |
| `ChunkMissing` | Referenced chunk not found |
| `ChecksumMismatch` | Data corruption detected |
| `EmptyData` | Cannot store empty artifact |
| `InvalidConfig` | Invalid configuration |
| `StorageError` | Underlying tensor store error |

## Architecture

```
+--------------------------------------------------+
|                BlobStore (Public API)             |
|   - put, get, delete                             |
|   - metadata, link, tag                          |
|   - gc, verify, repair                           |
+--------------------------------------------------+
            |              |              |
    +-------+      +-------+      +-------+
    |              |              |
+--------+   +-----------+   +----------+
| Writer |   |  Reader   |   |    GC    |
| Stream |   |  Stream   |   | (Tokio)  |
+--------+   +-----------+   +----------+
    |              |              |
    +-------+------+------+-------+
            |
    +------------------+
    |     Chunker      |
    |   SHA-256 hash   |
    +------------------+
            |
    +------------------+
    |   tensor_store   |
    | _blob:meta:*     |
    | _blob:chunk:*    |
    +------------------+
```

## Storage Format

### Artifact Metadata

Stored at `_blob:meta:{artifact_id}`:

```
{
    _type: "blob_artifact",
    _id: "artifact:uuid",
    _filename: "report.pdf",
    _content_type: "application/pdf",
    _size: 1048576,
    _checksum: "sha256:a1b2c3...",
    _chunk_size: 1048576,
    _chunk_count: 1,
    _chunks: ["_blob:chunk:sha256:a1b2c3..."],
    _created: 1703721600,
    _modified: 1703721600,
    _created_by: "user:alice",
    _linked_to: ["task:123"],
    _tags: ["tag:quarterly"],
    _meta:author: "Alice",
    _embedding: [0.1, 0.2, ...],
    _embedded_model: "text-embedding-3-small"
}
```

### Chunk Data

Stored at `_blob:chunk:sha256:{hash}`:

```
{
    _type: "blob_chunk",
    _data: <binary>,
    _size: 1048576,
    _refs: 2,
    _created: 1703721600
}
```

## Dependencies

- `tensor_store` - Underlying storage
- `tokio` - Async runtime for streaming and background GC
- `sha2` - SHA-256 hashing for content addressing
- `uuid` - Artifact ID generation
- `thiserror` - Error type derivation
