# How to: Store and Retrieve Blobs

Step-by-step guides for common `tensor_blob` operations: uploading artifacts,
downloading them, streaming large files, managing metadata, linking entities,
tagging, running garbage collection, and verifying integrity.

For the full type reference, see the
[Tensor Blob API Reference](../reference/api/tensor-blob.md). For the
design rationale, see
[Blob Storage Design](../explanation/blob-storage.md).

## Initialize a BlobStore

```rust
use tensor_blob::{BlobStore, BlobConfig, PutOptions};
use tensor_store::TensorStore;

let store = TensorStore::new();
let blob = BlobStore::new(store, BlobConfig::default()).await?;
```

## Store an Artifact

```rust
let artifact_id = blob.put(
    "report.pdf",
    &file_bytes,
    PutOptions::new()
        .with_created_by("user:alice")
        .with_tag("quarterly")
        .with_link("task:123"),
).await?;
```

Shell:

```text
BLOB PUT 'report.pdf' 'inline data here'
BLOB PUT 'report.pdf' FROM '/path/to/file'
```

Empty data is rejected with `BlobError::EmptyData`.

## Retrieve an Artifact

```rust
let data = blob.get(&artifact_id).await?;
```

Shell:

```text
BLOB GET 'artifact_id'
BLOB GET 'artifact_id' TO '/path/to/output'
```

## Stream a Large File Upload

Use `BlobWriter` to avoid loading the entire file into memory:

```rust
let mut writer = blob.writer("large_file.bin", PutOptions::default()).await?;

let file = std::fs::File::open("large_file.bin")?;
let mut reader = std::io::BufReader::new(file);
let mut buffer = vec![0u8; 64 * 1024];  // 64KB read buffer

loop {
    let n = reader.read(&mut buffer)?;
    if n == 0 { break; }
    writer.write(&buffer[..n]).await?;
}

let artifact_id = writer.finish().await?;
```

The writer buffers data internally until a full chunk (default 1 MB) is
accumulated, then hashes and stores each chunk incrementally.

## Stream a Large File Download

### Chunk-at-a-time (best for batch processing)

```rust
let mut reader = blob.reader(&artifact_id).await?;
while let Some(chunk) = reader.next_chunk().await? {
    process_chunk(&chunk);
}
```

### Read all into memory (small files only)

```rust
let mut reader = blob.reader(&artifact_id).await?;
let data = reader.read_all().await?;
```

### Buffer-based reading (streaming to other APIs)

```rust
let mut reader = blob.reader(&artifact_id).await?;
let mut buf = vec![0u8; 4096];
loop {
    let n = reader.read(&mut buf).await?;
    if n == 0 { break; }
    output.write_all(&buf[..n])?;
}
```

## Get Artifact Metadata

```rust
let meta = blob.metadata(&artifact_id).await?;
```

Shell:

```text
BLOB INFO 'artifact_id'
```

## Set and Get Custom Metadata

```rust
blob.set_meta(&artifact_id, "author", "Alice").await?;
let author = blob.get_meta(&artifact_id, "author").await?;
```

Shell:

```text
BLOB META SET 'artifact_id' 'author' 'Alice'
BLOB META GET 'artifact_id' 'author'
```

## Link Artifacts to Entities

```rust
blob.link(&artifact_id, "user:alice").await?;
blob.link(&artifact_id, "task:123").await?;

// Find all artifacts linked to an entity
let artifacts = blob.artifacts_for("user:alice").await?;
```

Shell:

```text
BLOB LINK 'artifact_id' TO 'user:alice'
BLOB UNLINK 'artifact_id' FROM 'user:alice'
BLOBS FOR 'user:alice'
```

## Tag Artifacts

```rust
blob.tag(&artifact_id, "important").await?;

// Find artifacts by tag
let tagged = blob.by_tag("important").await?;
```

Shell:

```text
BLOB TAG 'artifact_id' 'important'
BLOB UNTAG 'artifact_id' 'important'
BLOBS BY TAG 'important'
```

## Find Similar Artifacts (with vector feature)

```rust
// Set embedding for an artifact
blob.set_embedding(&artifact_id, embedding, "text-embedding-3-small").await?;

// Find similar artifacts
let similar = blob.similar(&artifact_id, 10).await?;
```

Shell:

```text
BLOBS SIMILAR TO 'artifact_id' LIMIT 10
```

## Verify Artifact Integrity

```rust
let valid = blob.verify(&artifact_id).await?;
if !valid {
    println!("Corruption detected in artifact {}", artifact_id);
}
```

For a paranoid read-then-verify workflow:

```rust
let mut reader = blob.reader(&artifact_id).await?;
let data = reader.read_all().await?;
if !reader.verify().await? {
    return Err("Corruption detected");
}
```

Shell:

```text
BLOB VERIFY 'artifact_id'
```

## Run Periodic Verification

```rust
for artifact_id in blob.list(None).await? {
    if !blob.verify(&artifact_id).await? {
        log::warn!("Corruption in artifact: {}", artifact_id);
    }
}
```

## Run Garbage Collection

### Start Background GC

```rust
blob.start().await?;     // Start background GC task
// ... use blob store ...
blob.shutdown().await?;  // Graceful shutdown
```

### Run Manual GC

```rust
// Incremental: processes batch_size chunks, respects min_age
let gc_stats = blob.gc().await?;
println!("Deleted {} chunks, freed {} bytes", gc_stats.deleted, gc_stats.freed_bytes);

// Full: rebuilds reference counts from all artifacts
let gc_stats = blob.full_gc().await?;
```

Shell:

```text
BLOB GC
BLOB GC FULL
```

### Use Full GC After Bulk Deletions

```rust
for artifact_id in to_delete {
    blob.delete(&artifact_id).await?;
}
blob.full_gc().await?;  // Clean up all orphans at once
```

## Repair Broken References

```rust
let repair_stats = blob.repair().await?;
println!(
    "Checked {} artifacts, verified {} chunks, fixed {} refs, deleted {} orphans",
    repair_stats.artifacts_checked,
    repair_stats.chunks_verified,
    repair_stats.refs_fixed,
    repair_stats.orphans_deleted,
);
```

Shell:

```text
BLOB REPAIR
```

## Check Storage Statistics

```rust
let stats = blob.stats().await?;
println!("Artifacts: {}", stats.artifact_count);
println!("Chunks: {}", stats.chunk_count);
println!("Dedup ratio: {:.1}%", stats.dedup_ratio * 100.0);
```

Shell:

```text
BLOB STATS
```

## Choose a Chunk Size

| Chunk Size | Best For | Trade-offs |
| --- | --- | --- |
| 256 KB | Many small files, high dedup potential | More metadata overhead |
| 1 MB (default) | General purpose | Good balance |
| 4 MB | Large media files, sequential access | Less dedup, fewer chunks |

```rust
let config = BlobConfig::new().with_chunk_size(512 * 1024);  // 512 KB
```

## Tune GC Settings

```rust
// High-throughput: aggressive GC
let config = BlobConfig::new()
    .with_gc_interval(Duration::from_secs(60))    // Every minute
    .with_gc_batch_size(500)                       // Large batches
    .with_gc_min_age(Duration::from_secs(300));    // 5 min grace

// Low-priority: gentle GC
let config = BlobConfig::new()
    .with_gc_interval(Duration::from_secs(3600))   // Hourly
    .with_gc_batch_size(50)                         // Small batches
    .with_gc_min_age(Duration::from_secs(86400));   // 24h grace
```

Set `gc_min_age` longer than your maximum expected upload time to avoid
collecting chunks from in-progress uploads.

## See Also

- [Tensor Blob API Reference](../reference/api/tensor-blob.md) -- complete type
  tables, configuration options, and method signatures
- [Blob Storage Design](../explanation/blob-storage.md) -- content-addressable
  architecture, streaming I/O model, and async design rationale
