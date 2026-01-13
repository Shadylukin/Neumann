//! TensorBlob - Module 11 of Neumann
//!
//! S3-style object storage for large artifacts using content-addressable
//! chunked storage with tensor-native metadata.
//!
//! # Features
//!
//! - **Content-Addressable**: Chunks keyed by SHA-256 hash for automatic deduplication
//! - **Tensor-Native**: Metadata participates in graph/relational/vector queries
//! - **Streaming**: Large files never fully in memory
//! - **Linked Artifacts**: Objects connect to entities via graph edges
//! - **Garbage Collected**: Orphaned chunks cleaned up automatically
//! - **Async-First**: All I/O operations are async
//!
//! # Example
//!
//! ```ignore
//! use tensor_blob::{BlobStore, BlobConfig, PutOptions};
//!
//! let store = BlobStore::new(TensorStore::new(), BlobConfig::default()).await?;
//!
//! // Store an artifact
//! let artifact_id = store.put(
//!     "report.pdf",
//!     &file_bytes,
//!     PutOptions::new()
//!         .with_created_by("user:alice")
//!         .with_tag("quarterly"),
//! ).await?;
//!
//! // Retrieve it
//! let data = store.get(&artifact_id).await?;
//! ```

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

mod chunker;
mod config;
mod error;
mod gc;
mod integrity;
mod metadata;
mod streaming;

use std::{collections::HashMap, sync::Arc};

pub use chunker::{compute_hash, compute_hash_streaming, Chunk, Chunker, StreamingHasher};
pub use config::{BlobConfig, GcConfig};
pub use error::{BlobError, Result};
pub use gc::GarbageCollector;
pub use integrity::{check_chunks_exist, find_orphaned_chunks, verify_chunk};
pub use metadata::{
    ArtifactMetadata, BlobStats, GcStats, MetadataUpdates, PutOptions, RepairStats, SimilarArtifact,
};
#[cfg(feature = "vector")]
use streaming::get_vector;
use streaming::{get_int, get_pointers, get_string};
pub use streaming::{BlobReader, BlobWriter};
#[cfg(feature = "vector")]
use tensor_store::SparseVector;
use tensor_store::{ScalarValue, TensorStore, TensorValue};
use tokio::task::JoinHandle;

/// S3-style blob store with content-addressable chunked storage.
pub struct BlobStore {
    store: TensorStore,
    gc: Arc<GarbageCollector>,
    gc_handle: Option<JoinHandle<()>>,
    config: BlobConfig,
}

impl BlobStore {
    /// Create a new blob store.
    pub async fn new(store: TensorStore, config: BlobConfig) -> Result<Self> {
        config.validate()?;

        let gc_config = GcConfig::from(&config);
        let gc = Arc::new(GarbageCollector::new(store.clone(), gc_config));

        Ok(Self {
            store,
            gc,
            gc_handle: None,
            config,
        })
    }

    /// Start background garbage collection.
    pub async fn start(&mut self) -> Result<()> {
        if self.gc_handle.is_none() {
            let handle = self.gc.clone().start();
            self.gc_handle = Some(handle);
        }
        Ok(())
    }

    /// Graceful shutdown.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.gc.shutdown();
        if let Some(handle) = self.gc_handle.take() {
            let _ = handle.await;
        }
        Ok(())
    }

    /// Get the underlying tensor store.
    pub fn store(&self) -> &TensorStore {
        &self.store
    }

    // === Simple API ===

    /// Store bytes and return artifact ID.
    pub async fn put(&self, filename: &str, data: &[u8], options: PutOptions) -> Result<String> {
        if data.is_empty() {
            return Err(BlobError::EmptyData);
        }

        if let Some(max_size) = self.config.max_artifact_size {
            if data.len() > max_size {
                return Err(BlobError::InvalidConfig(format!(
                    "data size {} exceeds max {}",
                    data.len(),
                    max_size
                )));
            }
        }

        let artifact_id = uuid::Uuid::new_v4().to_string();
        let mut writer = self.writer_with_id(&artifact_id, filename, options).await?;
        writer.write(data).await?;
        writer.finish().await
    }

    /// Get all bytes for an artifact.
    pub async fn get(&self, artifact_id: &str) -> Result<Vec<u8>> {
        let mut reader = self.reader(artifact_id).await?;
        reader.read_all().await
    }

    /// Delete an artifact.
    pub async fn delete(&self, artifact_id: &str) -> Result<()> {
        integrity::delete_artifact(&self.store, artifact_id)
    }

    /// Check if an artifact exists.
    pub async fn exists(&self, artifact_id: &str) -> Result<bool> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        Ok(self.store.exists(&meta_key))
    }

    // === Streaming API ===

    /// Create a writer for streaming upload.
    pub async fn writer(&self, filename: &str, options: PutOptions) -> Result<BlobWriter> {
        let artifact_id = uuid::Uuid::new_v4().to_string();
        self.writer_with_id(&artifact_id, filename, options).await
    }

    /// Create a writer with a specific artifact ID.
    async fn writer_with_id(
        &self,
        artifact_id: &str,
        filename: &str,
        options: PutOptions,
    ) -> Result<BlobWriter> {
        Ok(BlobWriter::new(
            self.store.clone(),
            self.config.chunk_size,
            artifact_id.to_string(),
            filename.to_string(),
            options,
            &self.config.default_content_type,
        ))
    }

    /// Create a reader for streaming download.
    pub async fn reader(&self, artifact_id: &str) -> Result<BlobReader> {
        BlobReader::new(self.store.clone(), artifact_id)
    }

    // === Metadata ===

    /// Get artifact metadata.
    pub async fn metadata(&self, artifact_id: &str) -> Result<ArtifactMetadata> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = self
            .store
            .get(&meta_key)
            .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

        let tags: Vec<String> = get_pointers(&tensor, "_tags")
            .unwrap_or_default()
            .into_iter()
            .filter_map(|t| t.strip_prefix("tag:").map(|s| s.to_string()))
            .collect();

        let mut custom = HashMap::new();
        for key in tensor.keys() {
            if let Some(meta_key) = key.strip_prefix("_meta:") {
                if let Some(value) = get_string(&tensor, key) {
                    custom.insert(meta_key.to_string(), value);
                }
            }
        }

        Ok(ArtifactMetadata {
            id: get_string(&tensor, "_id").unwrap_or_default(),
            filename: get_string(&tensor, "_filename").unwrap_or_default(),
            content_type: get_string(&tensor, "_content_type").unwrap_or_default(),
            size: get_int(&tensor, "_size").unwrap_or(0) as usize,
            checksum: get_string(&tensor, "_checksum").unwrap_or_default(),
            chunk_count: get_int(&tensor, "_chunk_count").unwrap_or(0) as usize,
            chunk_size: get_int(&tensor, "_chunk_size").unwrap_or(0) as usize,
            created_by: get_string(&tensor, "_created_by").unwrap_or_default(),
            created: get_int(&tensor, "_created").unwrap_or(0) as u64,
            modified: get_int(&tensor, "_modified").unwrap_or(0) as u64,
            linked_to: get_pointers(&tensor, "_linked_to").unwrap_or_default(),
            tags,
            custom,
            has_embedding: tensor.get("_embedding").is_some(),
            embedding_model: get_string(&tensor, "_embedded_model"),
        })
    }

    /// Update artifact metadata.
    pub async fn update_metadata(&self, artifact_id: &str, updates: MetadataUpdates) -> Result<()> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        let mut tensor = self
            .store
            .get(&meta_key)
            .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

        if let Some(filename) = updates.filename {
            tensor.set(
                "_filename",
                TensorValue::Scalar(ScalarValue::String(filename)),
            );
        }
        if let Some(content_type) = updates.content_type {
            tensor.set(
                "_content_type",
                TensorValue::Scalar(ScalarValue::String(content_type)),
            );
        }

        for (key, value) in updates.custom {
            let field = format!("_meta:{key}");
            match value {
                Some(v) => tensor.set(&field, TensorValue::Scalar(ScalarValue::String(v))),
                None => {
                    tensor.remove(&field);
                },
            }
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        tensor.set("_modified", TensorValue::Scalar(ScalarValue::Int(now)));

        self.store.put(&meta_key, tensor)?;
        Ok(())
    }

    /// Set custom metadata field.
    pub async fn set_meta(&self, artifact_id: &str, key: &str, value: &str) -> Result<()> {
        integrity::update_artifact_field(
            &self.store,
            artifact_id,
            &format!("_meta:{key}"),
            TensorValue::Scalar(ScalarValue::String(value.to_string())),
        )
    }

    /// Get custom metadata field.
    pub async fn get_meta(&self, artifact_id: &str, key: &str) -> Result<Option<String>> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = self
            .store
            .get(&meta_key)
            .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

        Ok(get_string(&tensor, &format!("_meta:{key}")))
    }

    // === Linking ===

    /// Link artifact to an entity.
    pub async fn link(&self, artifact_id: &str, entity: &str) -> Result<()> {
        integrity::add_artifact_link(&self.store, artifact_id, entity)
    }

    /// Unlink artifact from an entity.
    pub async fn unlink(&self, artifact_id: &str, entity: &str) -> Result<()> {
        integrity::remove_artifact_link(&self.store, artifact_id, entity)
    }

    /// Get entities linked to artifact.
    pub async fn links(&self, artifact_id: &str) -> Result<Vec<String>> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = self
            .store
            .get(&meta_key)
            .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

        Ok(get_pointers(&tensor, "_linked_to").unwrap_or_default())
    }

    /// Get artifacts linked to entity.
    pub async fn artifacts_for(&self, entity: &str) -> Result<Vec<String>> {
        let mut result = Vec::new();

        for meta_key in self.store.scan("_blob:meta:") {
            if let Ok(tensor) = self.store.get(&meta_key) {
                if let Some(linked_to) = get_pointers(&tensor, "_linked_to") {
                    if linked_to.contains(&entity.to_string()) {
                        if let Some(id) = get_string(&tensor, "_id") {
                            result.push(id);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    // === Tagging ===

    /// Add tag to artifact.
    pub async fn tag(&self, artifact_id: &str, tag: &str) -> Result<()> {
        integrity::add_artifact_tag(&self.store, artifact_id, tag)
    }

    /// Remove tag from artifact.
    pub async fn untag(&self, artifact_id: &str, tag: &str) -> Result<()> {
        integrity::remove_artifact_tag(&self.store, artifact_id, tag)
    }

    /// Get artifacts by tag.
    pub async fn by_tag(&self, tag: &str) -> Result<Vec<String>> {
        let tag_ref = format!("tag:{tag}");
        let mut result = Vec::new();

        for meta_key in self.store.scan("_blob:meta:") {
            if let Ok(tensor) = self.store.get(&meta_key) {
                if let Some(tags) = get_pointers(&tensor, "_tags") {
                    if tags.contains(&tag_ref) {
                        if let Some(id) = get_string(&tensor, "_id") {
                            result.push(id);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    // === Semantic (if VectorEngine available) ===

    /// Set artifact embedding.
    #[cfg(feature = "vector")]
    pub async fn set_embedding(
        &self,
        artifact_id: &str,
        embedding: Vec<f32>,
        model: &str,
    ) -> Result<()> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        let mut tensor = self
            .store
            .get(&meta_key)
            .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

        // Use sparse format for vectors with >50% zeros
        let storage = if streaming::should_use_sparse(&embedding) {
            TensorValue::Sparse(SparseVector::from_dense(&embedding))
        } else {
            TensorValue::Vector(embedding)
        };
        tensor.set("_embedding", storage);
        tensor.set(
            "_embedded_model",
            TensorValue::Scalar(ScalarValue::String(model.to_string())),
        );

        self.store.put(&meta_key, tensor)?;
        Ok(())
    }

    /// Find similar artifacts by embedding.
    #[cfg(feature = "vector")]
    pub async fn similar(&self, artifact_id: &str, k: usize) -> Result<Vec<SimilarArtifact>> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = self
            .store
            .get(&meta_key)
            .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

        let embedding = get_vector(&tensor, "_embedding")
            .ok_or_else(|| BlobError::NotFound(format!("embedding for {artifact_id}")))?;

        self.search_by_embedding(&embedding, k + 1)
            .await
            .map(|results| {
                results
                    .into_iter()
                    .filter(|r| r.id != artifact_id)
                    .take(k)
                    .collect()
            })
    }

    /// Search by embedding.
    #[cfg(feature = "vector")]
    pub async fn search_by_embedding(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<SimilarArtifact>> {
        let mut results = Vec::new();

        for meta_key in self.store.scan("_blob:meta:") {
            if let Ok(tensor) = self.store.get(&meta_key) {
                if let Some(stored_embedding) = get_vector(&tensor, "_embedding") {
                    if stored_embedding.len() == embedding.len() {
                        let similarity = SparseVector::from_dense(embedding)
                            .cosine_similarity(&SparseVector::from_dense(&stored_embedding));
                        let id = get_string(&tensor, "_id").unwrap_or_default();
                        let filename = get_string(&tensor, "_filename").unwrap_or_default();
                        results.push(SimilarArtifact {
                            id,
                            filename,
                            similarity,
                        });
                    }
                }
            }
        }

        // Sort by similarity descending
        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    // === Queries ===

    /// List artifacts by prefix.
    pub async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let mut result = Vec::new();

        for meta_key in self.store.scan("_blob:meta:") {
            if let Ok(tensor) = self.store.get(&meta_key) {
                if let Some(id) = get_string(&tensor, "_id") {
                    if let Some(p) = prefix {
                        if id.starts_with(p) {
                            result.push(id);
                        }
                    } else {
                        result.push(id);
                    }
                }
            }
        }

        Ok(result)
    }

    /// List artifacts by content type.
    pub async fn by_content_type(&self, content_type: &str) -> Result<Vec<String>> {
        let mut result = Vec::new();

        for meta_key in self.store.scan("_blob:meta:") {
            if let Ok(tensor) = self.store.get(&meta_key) {
                if let Some(ct) = get_string(&tensor, "_content_type") {
                    if ct == content_type {
                        if let Some(id) = get_string(&tensor, "_id") {
                            result.push(id);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// List artifacts by creator.
    pub async fn by_creator(&self, creator: &str) -> Result<Vec<String>> {
        let mut result = Vec::new();

        for meta_key in self.store.scan("_blob:meta:") {
            if let Ok(tensor) = self.store.get(&meta_key) {
                if let Some(cb) = get_string(&tensor, "_created_by") {
                    if cb == creator {
                        if let Some(id) = get_string(&tensor, "_id") {
                            result.push(id);
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    // === Integrity ===

    /// Verify artifact integrity.
    pub async fn verify(&self, artifact_id: &str) -> Result<bool> {
        integrity::verify_artifact(&self.store, artifact_id).await
    }

    /// Repair broken references.
    pub async fn repair(&self) -> Result<RepairStats> {
        integrity::repair(&self.store).await
    }

    // === GC ===

    /// Run garbage collection.
    pub async fn gc(&self) -> Result<GcStats> {
        Ok(self.gc.gc_cycle().await)
    }

    /// Full GC (recount all references).
    pub async fn full_gc(&self) -> Result<GcStats> {
        self.gc.full_gc().await
    }

    // === Stats ===

    /// Get storage statistics.
    pub async fn stats(&self) -> Result<BlobStats> {
        let mut artifact_count = 0;
        let mut total_bytes = 0;

        for meta_key in self.store.scan("_blob:meta:") {
            artifact_count += 1;
            if let Ok(tensor) = self.store.get(&meta_key) {
                total_bytes += get_int(&tensor, "_size").unwrap_or(0) as usize;
            }
        }

        let mut chunk_count = 0;
        let mut unique_bytes = 0;
        let mut orphaned_chunks = 0;

        for chunk_key in self.store.scan("_blob:chunk:") {
            chunk_count += 1;
            if let Ok(tensor) = self.store.get(&chunk_key) {
                unique_bytes += get_int(&tensor, "_size").unwrap_or(0) as usize;
                if get_int(&tensor, "_refs").unwrap_or(0) == 0 {
                    orphaned_chunks += 1;
                }
            }
        }

        let dedup_ratio = if total_bytes > 0 {
            1.0 - (unique_bytes as f64 / total_bytes as f64)
        } else {
            0.0
        };

        Ok(BlobStats {
            artifact_count,
            chunk_count,
            total_bytes,
            unique_bytes,
            dedup_ratio,
            orphaned_chunks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_blob_store_put_get() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        let data = b"hello world";
        let artifact_id = blob_store
            .put("test.txt", data, PutOptions::default())
            .await
            .unwrap();

        let retrieved = blob_store.get(&artifact_id).await.unwrap();
        assert_eq!(retrieved, data);
    }

    #[tokio::test]
    async fn test_blob_store_delete() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        let artifact_id = blob_store
            .put("test.txt", b"data", PutOptions::default())
            .await
            .unwrap();

        assert!(blob_store.exists(&artifact_id).await.unwrap());

        blob_store.delete(&artifact_id).await.unwrap();

        assert!(!blob_store.exists(&artifact_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_blob_store_metadata() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        let options = PutOptions::new()
            .with_content_type("text/plain")
            .with_created_by("user:alice")
            .with_meta("author", "Alice");

        let artifact_id = blob_store.put("test.txt", b"data", options).await.unwrap();

        let metadata = blob_store.metadata(&artifact_id).await.unwrap();
        assert_eq!(metadata.filename, "test.txt");
        assert_eq!(metadata.content_type, "text/plain");
        assert_eq!(metadata.created_by, "user:alice");
        assert_eq!(metadata.custom.get("author"), Some(&"Alice".to_string()));
    }

    #[tokio::test]
    async fn test_blob_store_update_metadata() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        let artifact_id = blob_store
            .put("test.txt", b"data", PutOptions::default())
            .await
            .unwrap();

        let updates = MetadataUpdates::new()
            .with_filename("renamed.txt")
            .set_meta("version", "2");

        blob_store
            .update_metadata(&artifact_id, updates)
            .await
            .unwrap();

        let metadata = blob_store.metadata(&artifact_id).await.unwrap();
        assert_eq!(metadata.filename, "renamed.txt");
        assert_eq!(metadata.custom.get("version"), Some(&"2".to_string()));
    }

    #[tokio::test]
    async fn test_blob_store_linking() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        let artifact_id = blob_store
            .put("test.txt", b"data", PutOptions::default())
            .await
            .unwrap();

        blob_store.link(&artifact_id, "user:alice").await.unwrap();
        blob_store.link(&artifact_id, "task:123").await.unwrap();

        let links = blob_store.links(&artifact_id).await.unwrap();
        assert!(links.contains(&"user:alice".to_string()));
        assert!(links.contains(&"task:123".to_string()));

        let artifacts = blob_store.artifacts_for("user:alice").await.unwrap();
        assert!(artifacts.contains(&artifact_id));

        blob_store.unlink(&artifact_id, "user:alice").await.unwrap();

        let links = blob_store.links(&artifact_id).await.unwrap();
        assert!(!links.contains(&"user:alice".to_string()));
    }

    #[tokio::test]
    async fn test_blob_store_tagging() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        let artifact_id = blob_store
            .put("test.txt", b"data", PutOptions::default())
            .await
            .unwrap();

        blob_store.tag(&artifact_id, "important").await.unwrap();
        blob_store.tag(&artifact_id, "quarterly").await.unwrap();

        let by_tag = blob_store.by_tag("important").await.unwrap();
        assert!(by_tag.contains(&artifact_id));

        blob_store.untag(&artifact_id, "important").await.unwrap();

        let by_tag = blob_store.by_tag("important").await.unwrap();
        assert!(!by_tag.contains(&artifact_id));
    }

    #[tokio::test]
    async fn test_blob_store_verify() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        let artifact_id = blob_store
            .put(
                "test.txt",
                b"test data for verification",
                PutOptions::default(),
            )
            .await
            .unwrap();

        let valid = blob_store.verify(&artifact_id).await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_blob_store_stats() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        blob_store
            .put("file1.txt", b"data 1", PutOptions::default())
            .await
            .unwrap();
        blob_store
            .put("file2.txt", b"data 2", PutOptions::default())
            .await
            .unwrap();

        let stats = blob_store.stats().await.unwrap();
        assert_eq!(stats.artifact_count, 2);
        assert!(stats.chunk_count >= 2);
    }

    #[tokio::test]
    async fn test_blob_store_list() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        blob_store
            .put("file1.txt", b"data 1", PutOptions::default())
            .await
            .unwrap();
        blob_store
            .put("file2.txt", b"data 2", PutOptions::default())
            .await
            .unwrap();

        let all = blob_store.list(None).await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_blob_store_by_content_type() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        blob_store
            .put(
                "file.txt",
                b"text",
                PutOptions::new().with_content_type("text/plain"),
            )
            .await
            .unwrap();
        blob_store
            .put(
                "file.pdf",
                b"pdf",
                PutOptions::new().with_content_type("application/pdf"),
            )
            .await
            .unwrap();

        let text_files = blob_store.by_content_type("text/plain").await.unwrap();
        assert_eq!(text_files.len(), 1);
    }

    #[tokio::test]
    async fn test_blob_store_gc() {
        let store = TensorStore::new();
        let config = BlobConfig::new().with_gc_min_age(std::time::Duration::from_secs(0));
        let blob_store = BlobStore::new(store.clone(), config).await.unwrap();

        // Create and delete an artifact
        let artifact_id = blob_store
            .put("test.txt", b"data", PutOptions::default())
            .await
            .unwrap();
        blob_store.delete(&artifact_id).await.unwrap();

        // Run full GC (doesn't have age restriction)
        let stats = blob_store.full_gc().await.unwrap();

        // Should have cleaned up orphaned chunks
        assert!(stats.deleted > 0 || store.scan("_blob:chunk:").is_empty());
    }

    #[tokio::test]
    async fn test_blob_store_empty_data() {
        let store = TensorStore::new();
        let blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        let result = blob_store
            .put("empty.txt", b"", PutOptions::default())
            .await;

        assert!(matches!(result, Err(BlobError::EmptyData)));
    }

    #[tokio::test]
    async fn test_blob_store_deduplication() {
        let store = TensorStore::new();
        let config = BlobConfig::new().with_chunk_size(10);
        let blob_store = BlobStore::new(store.clone(), config).await.unwrap();

        let data = vec![42u8; 10];

        // Store same data twice
        blob_store
            .put("file1.bin", &data, PutOptions::default())
            .await
            .unwrap();
        blob_store
            .put("file2.bin", &data, PutOptions::default())
            .await
            .unwrap();

        // Should only have 1 chunk due to deduplication
        let stats = blob_store.stats().await.unwrap();
        assert_eq!(stats.chunk_count, 1);
        assert!(stats.dedup_ratio > 0.0);
    }

    #[tokio::test]
    async fn test_blob_store_streaming() {
        let store = TensorStore::new();
        let config = BlobConfig::new().with_chunk_size(10);
        let blob_store = BlobStore::new(store, config).await.unwrap();

        let mut writer = blob_store
            .writer("stream.bin", PutOptions::default())
            .await
            .unwrap();

        // Write in chunks
        writer.write(&[1, 2, 3, 4, 5]).await.unwrap();
        writer.write(&[6, 7, 8, 9, 10]).await.unwrap();
        writer.write(&[11, 12, 13, 14, 15]).await.unwrap();

        let artifact_id = writer.finish().await.unwrap();

        // Read back
        let mut reader = blob_store.reader(&artifact_id).await.unwrap();
        let data = reader.read_all().await.unwrap();

        assert_eq!(
            data,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    #[tokio::test]
    async fn test_blob_store_start_shutdown() {
        let store = TensorStore::new();
        let mut blob_store = BlobStore::new(store, BlobConfig::default()).await.unwrap();

        blob_store.start().await.unwrap();

        // Let it run briefly
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        blob_store.shutdown().await.unwrap();
    }
}
