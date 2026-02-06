// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Durable blob storage with crash-safe chunk persistence.
//!
//! This module provides `DurableBlobStore`, which extends `BlobStore` with
//! durable chunk storage using `DurableBlobLog` from `tensor_store`.
//!
//! # Architecture
//!
//! - **Metadata**: Stored in `TensorStore` (in-memory with optional snapshots)
//! - **Chunk data**: Stored in `DurableBlobLog` (crash-safe, on-disk)
//!
//! This hybrid approach provides:
//! - Fast metadata queries (in-memory)
//! - Durable chunk storage (survives crashes)
//! - Efficient deduplication (hash-based)

use std::path::PathBuf;
use std::sync::Arc;

use tensor_store::{
    DurableBlobLog, DurableBlobLogConfig, DurableChunkHash, ScalarValue, TensorData, TensorStore,
    TensorValue,
};
use tokio::task::JoinHandle;

use crate::{
    chunker::{Chunker, StreamingHasher},
    config::{BlobConfig, GcConfig},
    error::{BlobError, Result},
    gc::GarbageCollector,
    metadata::PutOptions,
    streaming::get_pointers,
};

/// Configuration for durable blob storage.
#[derive(Debug, Clone)]
pub struct DurableBlobConfig {
    /// Base blob configuration.
    pub blob_config: BlobConfig,
    /// Directory for durable chunk storage.
    pub storage_dir: PathBuf,
    /// Segment size for durable storage (default: 64MB).
    pub segment_size: usize,
    /// Enable fsync for durability (default: true).
    pub enable_fsync: bool,
}

impl DurableBlobConfig {
    /// Create a new durable blob config with the given storage directory.
    #[must_use]
    pub fn new(storage_dir: impl Into<PathBuf>) -> Self {
        Self {
            blob_config: BlobConfig::default(),
            storage_dir: storage_dir.into(),
            segment_size: 64 * 1024 * 1024, // 64MB
            enable_fsync: true,
        }
    }

    /// Set the blob configuration.
    #[must_use]
    pub fn with_blob_config(mut self, config: BlobConfig) -> Self {
        self.blob_config = config;
        self
    }

    /// Set the segment size.
    #[must_use]
    pub const fn with_segment_size(mut self, size: usize) -> Self {
        self.segment_size = size;
        self
    }

    /// Disable fsync for testing.
    #[must_use]
    pub const fn without_fsync(mut self) -> Self {
        self.enable_fsync = false;
        self
    }
}

/// Durable blob store with crash-safe chunk persistence.
///
/// Unlike `BlobStore`, this store persists chunk data to disk using
/// `DurableBlobLog`, ensuring data survives process crashes.
pub struct DurableBlobStore {
    /// Metadata store (in-memory).
    store: TensorStore,
    /// Durable chunk storage.
    chunk_store: Arc<DurableBlobLog>,
    /// Garbage collector.
    gc: Arc<GarbageCollector>,
    gc_handle: Option<JoinHandle<()>>,
    /// Configuration.
    config: DurableBlobConfig,
}

impl DurableBlobStore {
    /// Create or open a durable blob store.
    ///
    /// # Errors
    ///
    /// Returns an error if the storage directory cannot be created or opened.
    pub async fn open(store: TensorStore, config: DurableBlobConfig) -> Result<Self> {
        config.blob_config.validate()?;

        let durable_config = DurableBlobLogConfig {
            segment_dir: config.storage_dir.clone(),
            segment_size: config.segment_size,
            enable_fsync: config.enable_fsync,
            cache_size: 10_000,
        };

        let chunk_store = DurableBlobLog::open(durable_config).map_err(|e| {
            BlobError::InvalidConfig(format!("Failed to open durable storage: {e}"))
        })?;

        let gc_config = GcConfig::from(&config.blob_config);
        let gc = Arc::new(GarbageCollector::new(store.clone(), gc_config));

        Ok(Self {
            store,
            chunk_store: Arc::new(chunk_store),
            gc,
            gc_handle: None,
            config,
        })
    }

    /// Start background garbage collection.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for future compatibility.
    pub async fn start(&mut self) -> Result<()> {
        if self.gc_handle.is_none() {
            let handle = self.gc.clone().start();
            self.gc_handle = Some(handle);
        }
        Ok(())
    }

    /// Graceful shutdown.
    ///
    /// # Errors
    ///
    /// Returns an error if fsync fails.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.gc.shutdown();
        if let Some(handle) = self.gc_handle.take() {
            let _ = handle.await;
        }
        // Sync durable storage
        self.chunk_store.sync().map_err(|e| {
            BlobError::InvalidConfig(format!("Failed to sync durable storage: {e}"))
        })?;
        Ok(())
    }

    /// Get the underlying tensor store (metadata only).
    #[must_use]
    pub const fn store(&self) -> &TensorStore {
        &self.store
    }

    /// Get the durable chunk store.
    #[must_use]
    pub fn chunk_store(&self) -> &DurableBlobLog {
        &self.chunk_store
    }

    /// Store bytes and return artifact ID.
    ///
    /// # Errors
    ///
    /// Returns an error if data is empty, exceeds max size, or storage fails.
    pub async fn put(&self, filename: &str, data: &[u8], options: PutOptions) -> Result<String> {
        if data.is_empty() {
            return Err(BlobError::EmptyData);
        }

        if let Some(max_size) = self.config.blob_config.max_artifact_size {
            if data.len() > max_size {
                return Err(BlobError::InvalidConfig(format!(
                    "data size {} exceeds max {}",
                    data.len(),
                    max_size
                )));
            }
        }

        let artifact_id = uuid::Uuid::new_v4().to_string();
        let mut writer = DurableBlobWriter::new(
            self.store.clone(),
            Arc::clone(&self.chunk_store),
            self.config.blob_config.chunk_size,
            artifact_id.clone(),
            filename.to_string(),
            options,
            &self.config.blob_config.default_content_type,
        );
        writer.write(data).await?;
        writer.finish().await
    }

    /// Get all bytes for an artifact.
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact is not found or chunk retrieval fails.
    pub async fn get(&self, artifact_id: &str) -> Result<Vec<u8>> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = self
            .store
            .get(&meta_key)
            .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

        let chunk_keys = get_pointers(&tensor, "_chunks")
            .ok_or_else(|| BlobError::NotFound(artifact_id.to_string()))?;

        let mut result = Vec::new();
        for chunk_key in chunk_keys {
            // Handle durable storage format: "_blob:durable:{blake3_hex}"
            if let Some(hash_hex) = chunk_key.strip_prefix("_blob:durable:") {
                if let Some(durable_hash) = DurableChunkHash::from_hex(hash_hex) {
                    let data = self
                        .chunk_store
                        .get(&durable_hash)
                        .map_err(|_| BlobError::ChunkMissing(chunk_key.clone()))?;
                    result.extend(data);
                    continue;
                }
            }

            // Handle legacy format: "_blob:chunk:sha256:{hex}" (stored in TensorStore)
            let chunk_tensor = self
                .store
                .get(&chunk_key)
                .map_err(|_| BlobError::ChunkMissing(chunk_key.clone()))?;

            if let Some(TensorValue::Scalar(ScalarValue::Bytes(data))) = chunk_tensor.get("_data") {
                result.extend(data);
            } else {
                return Err(BlobError::ChunkMissing(chunk_key));
            }
        }

        Ok(result)
    }

    /// Check if an artifact exists.
    ///
    /// # Errors
    ///
    /// Currently infallible, but returns `Result` for API consistency.
    pub async fn exists(&self, artifact_id: &str) -> Result<bool> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        Ok(self.store.exists(&meta_key))
    }

    /// Delete an artifact.
    ///
    /// # Errors
    ///
    /// Returns an error if the artifact is not found.
    pub async fn delete(&self, artifact_id: &str) -> Result<()> {
        let meta_key = format!("_blob:meta:{artifact_id}");
        let tensor = self
            .store
            .get(&meta_key)
            .map_err(|_| BlobError::NotFound(artifact_id.to_string()))?;

        // Delete chunks from durable storage
        if let Some(chunk_keys) = get_pointers(&tensor, "_chunks") {
            for chunk_key in chunk_keys {
                // Handle durable storage format
                if let Some(hash_hex) = chunk_key.strip_prefix("_blob:durable:") {
                    if let Some(durable_hash) = DurableChunkHash::from_hex(hash_hex) {
                        let _ = self.chunk_store.delete(&durable_hash);
                    }
                }
                // Also try to delete from TensorStore (legacy format)
                let _ = self.store.delete(&chunk_key);
            }
        }

        // Delete metadata
        self.store
            .delete(&meta_key)
            .map_err(|e| BlobError::InvalidConfig(format!("Failed to delete metadata: {e}")))?;

        Ok(())
    }

    /// Get storage statistics.
    #[must_use]
    pub fn stats(&self) -> DurableBlobStats {
        DurableBlobStats {
            chunk_count: self.chunk_store.chunk_count(),
            total_bytes: self.chunk_store.total_bytes(),
            segment_count: self.chunk_store.segment_count(),
        }
    }
}

/// Statistics for durable blob storage.
#[derive(Debug, Clone)]
pub struct DurableBlobStats {
    /// Total number of chunks stored.
    pub chunk_count: u64,
    /// Total bytes stored in chunks.
    pub total_bytes: u64,
    /// Number of storage segments.
    pub segment_count: usize,
}

/// Streaming writer for durable blob storage.
pub struct DurableBlobWriter {
    store: TensorStore,
    chunk_store: Arc<DurableBlobLog>,
    chunker: Chunker,
    artifact_id: String,
    filename: String,
    content_type: String,
    created_by: String,
    linked_to: Vec<String>,
    tags: Vec<String>,
    chunks: Vec<String>,
    total_size: usize,
    hasher: StreamingHasher,
    buffer: Vec<u8>,
}

impl DurableBlobWriter {
    fn new(
        store: TensorStore,
        chunk_store: Arc<DurableBlobLog>,
        chunk_size: usize,
        artifact_id: String,
        filename: String,
        options: PutOptions,
        default_content_type: &str,
    ) -> Self {
        Self {
            store,
            chunk_store,
            chunker: Chunker::new(chunk_size),
            artifact_id,
            filename,
            content_type: options
                .content_type
                .unwrap_or_else(|| default_content_type.to_string()),
            created_by: options.created_by.unwrap_or_default(),
            linked_to: options.linked_to,
            tags: options.tags,
            chunks: Vec::new(),
            total_size: 0,
            hasher: StreamingHasher::new(),
            buffer: Vec::new(),
        }
    }

    /// Write data to the artifact.
    ///
    /// # Errors
    ///
    /// Returns an error if chunk storage fails.
    pub async fn write(&mut self, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        self.hasher.update(data);
        self.total_size += data.len();
        self.buffer.extend_from_slice(data);

        // Process complete chunks
        while self.buffer.len() >= self.chunker.chunk_size() {
            let chunk_data: Vec<u8> = self.buffer.drain(..self.chunker.chunk_size()).collect();
            self.store_chunk(&chunk_data)?;
        }

        Ok(())
    }

    /// Store a chunk to durable storage.
    fn store_chunk(&mut self, data: &[u8]) -> Result<()> {
        // Store in durable blob log and get the BLAKE3 hash
        let hash = self
            .chunk_store
            .append(data)
            .map_err(|e| BlobError::InvalidConfig(format!("Failed to store chunk: {e}")))?;

        // Use BLAKE3 hash (hex) as chunk key
        let chunk_key = format!("_blob:durable:{}", hash.to_hex());

        self.chunks.push(chunk_key);
        Ok(())
    }

    /// Finalize the artifact and return its ID.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata storage fails.
    pub async fn finish(mut self) -> Result<String> {
        // Flush remaining buffer
        if !self.buffer.is_empty() {
            let data = std::mem::take(&mut self.buffer);
            self.store_chunk(&data)?;
        }

        // Sync durable storage
        self.chunk_store
            .sync()
            .map_err(|e| BlobError::InvalidConfig(format!("Failed to sync chunks: {e}")))?;

        let checksum = self.hasher.finalize();
        let now = current_timestamp();

        // Build metadata tensor
        let mut tensor = TensorData::new();
        tensor.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("blob_artifact".to_string())),
        );
        tensor.set(
            "_id",
            TensorValue::Scalar(ScalarValue::String(self.artifact_id.clone())),
        );
        tensor.set(
            "_filename",
            TensorValue::Scalar(ScalarValue::String(self.filename)),
        );
        tensor.set(
            "_content_type",
            TensorValue::Scalar(ScalarValue::String(self.content_type)),
        );
        tensor.set(
            "_size",
            TensorValue::Scalar(ScalarValue::Int(
                i64::try_from(self.total_size).unwrap_or(0),
            )),
        );
        tensor.set(
            "_checksum",
            TensorValue::Scalar(ScalarValue::String(checksum)),
        );
        tensor.set(
            "_chunk_count",
            TensorValue::Scalar(ScalarValue::Int(
                i64::try_from(self.chunks.len()).unwrap_or(0),
            )),
        );
        tensor.set(
            "_chunk_size",
            TensorValue::Scalar(ScalarValue::Int(
                i64::try_from(self.chunker.chunk_size()).unwrap_or(0),
            )),
        );
        tensor.set(
            "_created_by",
            TensorValue::Scalar(ScalarValue::String(self.created_by)),
        );
        tensor.set(
            "_created",
            TensorValue::Scalar(ScalarValue::Int(i64::try_from(now).unwrap_or(0))),
        );
        tensor.set(
            "_modified",
            TensorValue::Scalar(ScalarValue::Int(i64::try_from(now).unwrap_or(0))),
        );
        tensor.set("_chunks", TensorValue::Pointers(self.chunks));

        if !self.linked_to.is_empty() {
            tensor.set("_linked_to", TensorValue::Pointers(self.linked_to));
        }

        if !self.tags.is_empty() {
            let tag_refs: Vec<String> = self.tags.iter().map(|t| format!("tag:{t}")).collect();
            tensor.set("_tags", TensorValue::Pointers(tag_refs));
        }

        // Store metadata
        let meta_key = format!("_blob:meta:{}", self.artifact_id);
        self.store.put(&meta_key, tensor)?;

        Ok(self.artifact_id)
    }
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    fn test_config(dir: &std::path::Path) -> DurableBlobConfig {
        DurableBlobConfig::new(dir)
            .without_fsync()
            .with_blob_config(BlobConfig::default().with_chunk_size(64))
    }

    #[tokio::test]
    async fn test_durable_put_get() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let store = TensorStore::new();
        let blob_store = DurableBlobStore::open(store, config).await.unwrap();

        let data = b"hello durable world";
        let artifact_id = blob_store
            .put("test.txt", data, PutOptions::default())
            .await
            .unwrap();

        let retrieved = blob_store.get(&artifact_id).await.unwrap();
        assert_eq!(retrieved, data);
    }

    #[tokio::test]
    async fn test_durable_exists() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let store = TensorStore::new();
        let blob_store = DurableBlobStore::open(store, config).await.unwrap();

        let artifact_id = blob_store
            .put("test.txt", b"data", PutOptions::default())
            .await
            .unwrap();

        assert!(blob_store.exists(&artifact_id).await.unwrap());
        assert!(!blob_store.exists("nonexistent").await.unwrap());
    }

    #[tokio::test]
    async fn test_durable_delete() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let store = TensorStore::new();
        let blob_store = DurableBlobStore::open(store, config).await.unwrap();

        let artifact_id = blob_store
            .put("test.txt", b"to be deleted", PutOptions::default())
            .await
            .unwrap();

        assert!(blob_store.exists(&artifact_id).await.unwrap());
        blob_store.delete(&artifact_id).await.unwrap();
        assert!(!blob_store.exists(&artifact_id).await.unwrap());
    }

    #[tokio::test]
    async fn test_durable_stats() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let store = TensorStore::new();
        let blob_store = DurableBlobStore::open(store, config).await.unwrap();

        let initial_stats = blob_store.stats();
        assert_eq!(initial_stats.chunk_count, 0);

        blob_store
            .put("test.txt", b"some data for stats", PutOptions::default())
            .await
            .unwrap();

        let stats = blob_store.stats();
        assert!(stats.chunk_count > 0);
        assert!(stats.total_bytes > 0);
    }

    #[tokio::test]
    async fn test_durable_large_file() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());
        let store = TensorStore::new();
        let blob_store = DurableBlobStore::open(store, config).await.unwrap();

        // Create data larger than chunk size
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let artifact_id = blob_store
            .put("large.bin", &data, PutOptions::default())
            .await
            .unwrap();

        let retrieved = blob_store.get(&artifact_id).await.unwrap();
        assert_eq!(retrieved, data);
    }

    #[tokio::test]
    async fn test_durable_recovery() {
        let dir = tempdir().unwrap();
        let config = test_config(dir.path());

        {
            let store = TensorStore::new();
            let mut blob_store = DurableBlobStore::open(store, config.clone()).await.unwrap();
            let _artifact_id = blob_store
                .put("persistent.txt", b"survives restart", PutOptions::default())
                .await
                .unwrap();
            blob_store.shutdown().await.unwrap();
        }

        // Reopen - chunk data should be recovered, but metadata is lost
        // (TensorStore is in-memory; for full durability, need snapshots too)
        {
            let store = TensorStore::new();
            let blob_store = DurableBlobStore::open(store, config).await.unwrap();

            // Stats should show recovered chunks
            let stats = blob_store.stats();
            assert!(stats.chunk_count > 0, "Chunks should be recovered from WAL");
        }
    }
}
