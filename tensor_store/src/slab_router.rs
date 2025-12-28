//! Slab router for directing operations to specialized storage backends.
//!
//! SlabRouter coordinates between the specialized slabs (EntityIndex, EmbeddingSlab,
//! GraphTensor, RelationalSlab, MetadataSlab, CacheRing, BlobLog) and provides
//! a unified interface similar to TensorStore.
//!
//! # Key Routing
//!
//! Operations are routed based on key prefixes:
//! - `emb:*` -> EmbeddingSlab (embedding vectors)
//! - `node:*`, `edge:*` -> GraphTensor (graph data)
//! - `table:*` -> RelationalSlab (relational rows)
//! - `_cache:*` -> CacheRing (cached data)
//! - Everything else (including `_blob:*`) -> MetadataSlab (general metadata)

use crate::blob_log::{BlobLog, BlobLogSnapshot};
use crate::cache_ring::{CacheRing, CacheRingSnapshot, EvictionStrategy};
use crate::embedding_slab::{EmbeddingSlab, EmbeddingSlabSnapshot};
use crate::entity_index::{EntityIndex, EntityIndexSnapshot};
use crate::graph_tensor::{GraphTensor, GraphTensorSnapshot};
use crate::metadata_slab::{MetadataSlab, MetadataSlabSnapshot};
use crate::relational_slab::{RelationalSlab, RelationalSlabSnapshot};
use crate::snapshot::{self, SnapshotFormatError};
use crate::{TensorData, TensorValue};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration for SlabRouter.
#[derive(Debug, Clone)]
pub struct SlabRouterConfig {
    /// Embedding dimension for EmbeddingSlab.
    pub embedding_dim: usize,
    /// Cache capacity for CacheRing.
    pub cache_capacity: usize,
    /// Eviction strategy for CacheRing.
    pub cache_strategy: EvictionStrategy,
    /// Segment size for BlobLog.
    pub blob_segment_size: usize,
    /// Merge threshold for GraphTensor.
    pub graph_merge_threshold: usize,
}

impl Default for SlabRouterConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            cache_capacity: 10_000,
            cache_strategy: EvictionStrategy::default(),
            blob_segment_size: 64 * 1024 * 1024, // 64MB
            graph_merge_threshold: 10_000,
        }
    }
}

/// Unified router coordinating all specialized slabs.
pub struct SlabRouter {
    /// Entity index for stable ID assignment.
    pub index: EntityIndex,
    /// Embedding storage.
    pub embeddings: EmbeddingSlab,
    /// Graph storage.
    pub graph: GraphTensor,
    /// Relational storage.
    pub relations: RelationalSlab,
    /// General metadata storage.
    pub metadata: MetadataSlab,
    /// Fixed-size cache.
    pub cache: CacheRing<TensorData>,
    /// Blob log storage.
    pub blobs: BlobLog,
    /// Operation counter for stats.
    ops_count: AtomicU64,
}

impl SlabRouter {
    /// Create a new slab router with default configuration.
    pub fn new() -> Self {
        Self::with_config(SlabRouterConfig::default())
    }

    /// Create a new slab router with a capacity hint.
    ///
    /// Note: The capacity hint is informational. BTreeMap-based slabs don't
    /// pre-allocate. This method exists for API compatibility with TensorStore.
    pub fn with_capacity(_capacity: usize) -> Self {
        // BTreeMap-based slabs don't benefit from capacity hints
        // (they split nodes on demand rather than resize)
        Self::new()
    }

    /// Create a new slab router with custom configuration.
    pub fn with_config(config: SlabRouterConfig) -> Self {
        Self {
            index: EntityIndex::new(),
            embeddings: EmbeddingSlab::with_dimension(config.embedding_dim),
            graph: GraphTensor::with_merge_threshold(config.graph_merge_threshold),
            relations: RelationalSlab::new(),
            metadata: MetadataSlab::new(),
            cache: CacheRing::new(config.cache_capacity, config.cache_strategy),
            blobs: BlobLog::new(config.blob_segment_size),
            ops_count: AtomicU64::new(0),
        }
    }

    /// Store a value, routing to the appropriate slab.
    pub fn put(&self, key: &str, value: TensorData) -> Result<(), SlabRouterError> {
        self.ops_count.fetch_add(1, Ordering::Relaxed);

        match Self::classify_key(key) {
            KeyClass::Embedding => {
                let entity_id = self.index.get_or_create(key);
                // Extract vector from TensorValue if present
                if let Some(TensorValue::Vector(vec)) = value.get("_embedding") {
                    // Try to store in embedding slab; if dimension mismatch, just use metadata
                    if self.embeddings.set(entity_id, vec).is_err() {
                        // Dimension mismatch - store in metadata only (this is fine)
                    }
                }
                // Also store metadata (always includes the embedding for retrieval)
                self.metadata.set(key, value);
                Ok(())
            },
            KeyClass::Graph => {
                // Graph operations need special handling based on entity type
                self.metadata.set(key, value);
                Ok(())
            },
            KeyClass::Table => {
                // Tables are handled via RelationalSlab API
                self.metadata.set(key, value);
                Ok(())
            },
            KeyClass::Cache => {
                let size = Self::estimate_size(&value);
                self.cache.put(key, value, 1.0, size);
                Ok(())
            },
            KeyClass::Metadata => {
                self.metadata.set(key, value);
                Ok(())
            },
        }
    }

    /// Retrieve a value, routing to the appropriate slab.
    pub fn get(&self, key: &str) -> Result<TensorData, SlabRouterError> {
        self.ops_count.fetch_add(1, Ordering::Relaxed);

        match Self::classify_key(key) {
            KeyClass::Embedding => {
                if let Some(entity_id) = self.index.get(key) {
                    if let Some(vector) = self.embeddings.get(entity_id) {
                        let mut data = self.metadata.get(key).unwrap_or_default();
                        data.set("_embedding", TensorValue::Vector(vector));
                        return Ok(data);
                    }
                }
                self.metadata
                    .get(key)
                    .ok_or_else(|| SlabRouterError::NotFound(key.to_string()))
            },
            KeyClass::Cache => self
                .cache
                .get(key)
                .ok_or_else(|| SlabRouterError::NotFound(key.to_string())),
            _ => self
                .metadata
                .get(key)
                .ok_or_else(|| SlabRouterError::NotFound(key.to_string())),
        }
    }

    /// Delete a value, routing to the appropriate slab.
    ///
    /// Returns an error if the key doesn't exist.
    pub fn delete(&self, key: &str) -> Result<(), SlabRouterError> {
        self.ops_count.fetch_add(1, Ordering::Relaxed);

        // Check if key exists first
        if !self.exists(key) {
            return Err(SlabRouterError::NotFound(key.to_string()));
        }

        match Self::classify_key(key) {
            KeyClass::Embedding => {
                if let Some(entity_id) = self.index.get(key) {
                    self.embeddings.delete(entity_id);
                }
                self.index.remove(key);
                self.metadata.delete(key);
                Ok(())
            },
            KeyClass::Cache => {
                self.cache.delete(key);
                Ok(())
            },
            _ => {
                self.metadata.delete(key);
                Ok(())
            },
        }
    }

    /// Check if a key exists.
    pub fn exists(&self, key: &str) -> bool {
        match Self::classify_key(key) {
            KeyClass::Embedding => self.index.contains(key) || self.metadata.contains(key),
            KeyClass::Cache => self.cache.contains(key),
            _ => self.metadata.contains(key),
        }
    }

    /// Scan keys by prefix.
    pub fn scan(&self, prefix: &str) -> Vec<String> {
        // Collect from metadata slab
        let mut keys: Vec<String> = self
            .metadata
            .scan(prefix)
            .into_iter()
            .map(|(k, _)| k)
            .collect();

        // Add from entity index
        for (key, _) in self.index.scan_prefix(prefix) {
            if !keys.contains(&key) {
                keys.push(key);
            }
        }

        keys
    }

    /// Count keys by prefix.
    pub fn scan_count(&self, prefix: &str) -> usize {
        self.scan(prefix).len()
    }

    /// Get total entity count across all slabs.
    pub fn len(&self) -> usize {
        self.metadata.len() + self.cache.len()
    }

    /// Check if all slabs are empty.
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
            && self.cache.is_empty()
            && self.embeddings.len() == 0
            && self.graph.edge_count() == 0
    }

    /// Clear all slabs.
    pub fn clear(&self) {
        self.index.clear();
        self.embeddings.clear();
        self.graph.clear();
        self.relations.clear();
        self.metadata.clear();
        self.cache.clear();
        self.blobs.clear();
    }

    /// Get operation count for stats.
    pub fn ops_count(&self) -> u64 {
        self.ops_count.load(Ordering::Relaxed)
    }

    /// Create a snapshot for serialization.
    pub fn snapshot(&self) -> SlabRouterSnapshot {
        SlabRouterSnapshot {
            index: self.index.snapshot(),
            embeddings: self.embeddings.snapshot(),
            graph: self.graph.snapshot(),
            relations: self.relations.snapshot(),
            metadata: self.metadata.snapshot(),
            cache: self.cache.snapshot(),
            blobs: self.blobs.snapshot(),
        }
    }

    /// Restore from a snapshot.
    pub fn restore(snapshot: SlabRouterSnapshot) -> Self {
        Self {
            index: EntityIndex::restore(snapshot.index),
            embeddings: EmbeddingSlab::restore(snapshot.embeddings),
            graph: GraphTensor::restore(snapshot.graph),
            relations: RelationalSlab::restore(snapshot.relations),
            metadata: MetadataSlab::restore(snapshot.metadata),
            cache: CacheRing::restore(snapshot.cache),
            blobs: BlobLog::restore(snapshot.blobs),
            ops_count: AtomicU64::new(0),
        }
    }

    /// Save to a file (v3 format with auto-detection on load).
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), SnapshotFormatError> {
        snapshot::save_v3(self, path)
    }

    /// Load from a file (auto-detects v2/v3 format).
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, SnapshotFormatError> {
        snapshot::load(path)
    }

    /// Serialize to bytes (v3 format).
    pub fn to_bytes(&self) -> Result<Vec<u8>, SnapshotFormatError> {
        let router_snapshot = self.snapshot();
        let entry_count = (self.len() + self.index.len()) as u64;

        let v3 = snapshot::V3Snapshot {
            header: snapshot::SnapshotHeader::new(entry_count),
            router: router_snapshot,
        };

        bincode::serialize(&v3).map_err(SnapshotFormatError::from)
    }

    /// Deserialize from bytes (v3 format).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotFormatError> {
        let v3: snapshot::V3Snapshot =
            bincode::deserialize(bytes).map_err(SnapshotFormatError::from)?;
        v3.header.validate()?;
        Ok(Self::restore(v3.router))
    }

    /// Classify a key to determine which slab should handle it.
    fn classify_key(key: &str) -> KeyClass {
        if key.starts_with("emb:") {
            KeyClass::Embedding
        } else if key.starts_with("node:") || key.starts_with("edge:") {
            KeyClass::Graph
        } else if key.starts_with("table:") {
            KeyClass::Table
        } else if key.starts_with("_cache:") {
            KeyClass::Cache
        } else {
            // Note: _blob:meta: and _blob:chunk: keys from tensor_blob module
            // are stored as metadata (they contain regular TensorData).
            // The internal BlobLog slab is accessed directly, not via key routing.
            KeyClass::Metadata
        }
    }

    /// Estimate the byte size of a TensorData for cache scoring.
    fn estimate_size(data: &TensorData) -> usize {
        // Rough estimate: 100 bytes per field
        data.len() * 100
    }
}

impl Default for SlabRouter {
    fn default() -> Self {
        Self::new()
    }
}

/// Key classification for routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyClass {
    Embedding,
    Graph,
    Table,
    Cache,
    Metadata,
}

/// Errors from SlabRouter operations.
#[derive(Debug, Clone)]
pub enum SlabRouterError {
    NotFound(String),
    EmbeddingError(String),
    GraphError(String),
    RelationalError(String),
}

impl std::fmt::Display for SlabRouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(key) => write!(f, "not found: {}", key),
            Self::EmbeddingError(msg) => write!(f, "embedding error: {}", msg),
            Self::GraphError(msg) => write!(f, "graph error: {}", msg),
            Self::RelationalError(msg) => write!(f, "relational error: {}", msg),
        }
    }
}

impl std::error::Error for SlabRouterError {}

/// Serializable snapshot of all slab state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlabRouterSnapshot {
    pub index: EntityIndexSnapshot,
    pub embeddings: EmbeddingSlabSnapshot,
    pub graph: GraphTensorSnapshot,
    pub relations: RelationalSlabSnapshot,
    pub metadata: MetadataSlabSnapshot,
    pub cache: CacheRingSnapshot<TensorData>,
    pub blobs: BlobLogSnapshot,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> TensorData {
        let mut data = TensorData::new();
        data.set(
            "name",
            TensorValue::Scalar(crate::ScalarValue::String("test".to_string())),
        );
        data
    }

    #[test]
    fn test_new() {
        let router = SlabRouter::new();
        assert!(router.is_empty());
        assert_eq!(router.len(), 0);
    }

    #[test]
    fn test_with_config() {
        let config = SlabRouterConfig {
            embedding_dim: 128,
            cache_capacity: 1000,
            ..Default::default()
        };
        let router = SlabRouter::with_config(config);
        assert!(router.is_empty());
    }

    #[test]
    fn test_default() {
        let router = SlabRouter::default();
        assert!(router.is_empty());
    }

    #[test]
    fn test_put_get_metadata() {
        let router = SlabRouter::new();
        let data = create_test_data();

        router.put("user:1", data.clone()).unwrap();

        let retrieved = router.get("user:1").unwrap();
        assert_eq!(retrieved.get("name"), data.get("name"));
    }

    #[test]
    fn test_put_get_embedding() {
        let router = SlabRouter::new();

        // Create 384-dim embedding (default dimension)
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let mut data = TensorData::new();
        data.set("_embedding", TensorValue::Vector(embedding));

        router.put("emb:vec1", data).unwrap();

        let retrieved = router.get("emb:vec1").unwrap();
        assert!(retrieved.get("_embedding").is_some());
    }

    #[test]
    fn test_put_get_cache() {
        let router = SlabRouter::new();
        let data = create_test_data();

        router.put("_cache:item1", data.clone()).unwrap();

        let retrieved = router.get("_cache:item1").unwrap();
        assert_eq!(retrieved.get("name"), data.get("name"));
    }

    #[test]
    fn test_put_get_blob() {
        let router = SlabRouter::new();

        let mut data = TensorData::new();
        data.set(
            "_data",
            TensorValue::Scalar(crate::ScalarValue::Bytes(vec![1, 2, 3, 4])),
        );

        router.put("_blob:chunk1", data).unwrap();

        let retrieved = router.get("_blob:chunk1").unwrap();
        assert!(retrieved.get("_data").is_some());
    }

    #[test]
    fn test_delete() {
        let router = SlabRouter::new();
        let data = create_test_data();

        router.put("key1", data).unwrap();
        assert!(router.exists("key1"));

        router.delete("key1").unwrap();
        assert!(!router.exists("key1"));
    }

    #[test]
    fn test_delete_embedding() {
        let router = SlabRouter::new();

        // Create 384-dim embedding (default dimension)
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let mut data = TensorData::new();
        data.set("_embedding", TensorValue::Vector(embedding));

        router.put("emb:vec1", data).unwrap();
        assert!(router.exists("emb:vec1"));

        router.delete("emb:vec1").unwrap();
        assert!(!router.exists("emb:vec1"));
    }

    #[test]
    fn test_delete_cache() {
        let router = SlabRouter::new();
        let data = create_test_data();

        router.put("_cache:item1", data).unwrap();
        assert!(router.exists("_cache:item1"));

        router.delete("_cache:item1").unwrap();
        assert!(!router.exists("_cache:item1"));
    }

    #[test]
    fn test_exists() {
        let router = SlabRouter::new();

        assert!(!router.exists("nonexistent"));

        router.put("key1", create_test_data()).unwrap();
        assert!(router.exists("key1"));
    }

    #[test]
    fn test_scan() {
        let router = SlabRouter::new();

        router.put("user:1", create_test_data()).unwrap();
        router.put("user:2", create_test_data()).unwrap();
        router.put("item:1", create_test_data()).unwrap();

        let users = router.scan("user:");
        assert_eq!(users.len(), 2);

        let items = router.scan("item:");
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_scan_count() {
        let router = SlabRouter::new();

        router.put("user:1", create_test_data()).unwrap();
        router.put("user:2", create_test_data()).unwrap();

        assert_eq!(router.scan_count("user:"), 2);
        assert_eq!(router.scan_count("other:"), 0);
    }

    #[test]
    fn test_len() {
        let router = SlabRouter::new();

        router.put("key1", create_test_data()).unwrap();
        router.put("key2", create_test_data()).unwrap();

        assert_eq!(router.len(), 2);
    }

    #[test]
    fn test_clear() {
        let router = SlabRouter::new();

        router.put("key1", create_test_data()).unwrap();
        router.put("_cache:item1", create_test_data()).unwrap();

        router.clear();

        assert!(router.is_empty());
        assert!(!router.exists("key1"));
        assert!(!router.exists("_cache:item1"));
    }

    #[test]
    fn test_ops_count() {
        let router = SlabRouter::new();

        router.put("key1", create_test_data()).unwrap();
        let _ = router.get("key1");
        router.delete("key1").unwrap();

        assert_eq!(router.ops_count(), 3);
    }

    #[test]
    fn test_snapshot_restore() {
        let router = SlabRouter::new();

        router.put("key1", create_test_data()).unwrap();
        router.put("_cache:item1", create_test_data()).unwrap();

        let snapshot = router.snapshot();
        let restored = SlabRouter::restore(snapshot);

        assert!(restored.exists("key1"));
        assert!(restored.exists("_cache:item1"));
    }

    #[test]
    fn test_error_display() {
        let err = SlabRouterError::NotFound("key".to_string());
        assert!(err.to_string().contains("not found"));

        let err = SlabRouterError::EmbeddingError("dim".to_string());
        assert!(err.to_string().contains("embedding"));
    }

    #[test]
    fn test_not_found() {
        let router = SlabRouter::new();

        let result = router.get("nonexistent");
        assert!(matches!(result, Err(SlabRouterError::NotFound(_))));
    }

    #[test]
    fn test_key_classification() {
        assert_eq!(SlabRouter::classify_key("emb:vec1"), KeyClass::Embedding);
        assert_eq!(SlabRouter::classify_key("node:1"), KeyClass::Graph);
        assert_eq!(SlabRouter::classify_key("edge:1"), KeyClass::Graph);
        assert_eq!(SlabRouter::classify_key("table:users"), KeyClass::Table);
        assert_eq!(SlabRouter::classify_key("_cache:item"), KeyClass::Cache);
        // _blob: keys go to metadata - tensor_blob uses these for TensorData storage
        assert_eq!(SlabRouter::classify_key("_blob:chunk"), KeyClass::Metadata);
        assert_eq!(
            SlabRouter::classify_key("anything:else"),
            KeyClass::Metadata
        );
    }

    #[test]
    fn test_config_default() {
        let config = SlabRouterConfig::default();
        assert_eq!(config.embedding_dim, 384);
        assert_eq!(config.cache_capacity, 10_000);
    }

    #[test]
    fn test_put_get_graph() {
        let router = SlabRouter::new();
        let data = create_test_data();

        router.put("node:1", data.clone()).unwrap();
        router.put("edge:1", create_test_data()).unwrap();

        let retrieved = router.get("node:1").unwrap();
        assert_eq!(retrieved.get("name"), data.get("name"));

        let edge = router.get("edge:1").unwrap();
        assert!(edge.get("name").is_some());
    }

    #[test]
    fn test_put_get_table() {
        let router = SlabRouter::new();
        let data = create_test_data();

        router.put("table:users", data.clone()).unwrap();

        let retrieved = router.get("table:users").unwrap();
        assert_eq!(retrieved.get("name"), data.get("name"));
    }

    #[test]
    fn test_delete_graph() {
        let router = SlabRouter::new();
        router.put("node:1", create_test_data()).unwrap();

        assert!(router.exists("node:1"));
        router.delete("node:1").unwrap();
        assert!(!router.exists("node:1"));
    }

    #[test]
    fn test_delete_table() {
        let router = SlabRouter::new();
        router.put("table:users", create_test_data()).unwrap();

        assert!(router.exists("table:users"));
        router.delete("table:users").unwrap();
        assert!(!router.exists("table:users"));
    }

    #[test]
    fn test_blob_roundtrip() {
        let router = SlabRouter::new();

        let mut data = TensorData::new();
        data.set(
            "_data",
            TensorValue::Scalar(crate::ScalarValue::Bytes(vec![1, 2, 3, 4, 5])),
        );

        router.put("_blob:chunk1", data).unwrap();

        // Get the blob back
        let retrieved = router.get("_blob:chunk1").unwrap();
        if let Some(TensorValue::Scalar(crate::ScalarValue::Bytes(bytes))) = retrieved.get("_data")
        {
            assert_eq!(bytes, &[1, 2, 3, 4, 5]);
        } else {
            panic!("Expected bytes data");
        }
    }

    #[test]
    fn test_delete_blob() {
        let router = SlabRouter::new();

        let mut data = TensorData::new();
        data.set(
            "_data",
            TensorValue::Scalar(crate::ScalarValue::Bytes(vec![1, 2, 3])),
        );

        router.put("_blob:chunk1", data).unwrap();
        assert!(router.exists("_blob:chunk1"));

        router.delete("_blob:chunk1").unwrap();
        assert!(!router.exists("_blob:chunk1"));
    }

    #[test]
    fn test_is_empty_with_data() {
        let router = SlabRouter::new();
        assert!(router.is_empty());

        router.put("key1", create_test_data()).unwrap();
        assert!(!router.is_empty());
    }

    #[test]
    fn test_is_empty_with_cache() {
        let router = SlabRouter::new();
        assert!(router.is_empty());

        router.put("_cache:item", create_test_data()).unwrap();
        assert!(!router.is_empty());
    }

    #[test]
    fn test_is_empty_with_embedding() {
        let router = SlabRouter::new();
        assert!(router.is_empty());

        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let mut data = TensorData::new();
        data.set("_embedding", TensorValue::Vector(embedding));
        router.put("emb:vec1", data).unwrap();

        assert!(!router.is_empty());
    }

    #[test]
    fn test_error_display_all_variants() {
        let err = SlabRouterError::NotFound("key".to_string());
        assert!(err.to_string().contains("not found"));

        let err = SlabRouterError::EmbeddingError("dim".to_string());
        assert!(err.to_string().contains("embedding"));

        let err = SlabRouterError::GraphError("graph issue".to_string());
        assert!(err.to_string().contains("graph"));

        let err = SlabRouterError::RelationalError("table issue".to_string());
        assert!(err.to_string().contains("relational"));
    }

    #[test]
    fn test_exists_cache() {
        let router = SlabRouter::new();
        assert!(!router.exists("_cache:nonexistent"));

        router.put("_cache:item", create_test_data()).unwrap();
        assert!(router.exists("_cache:item"));
    }

    #[test]
    fn test_exists_blob() {
        let router = SlabRouter::new();
        assert!(!router.exists("_blob:nonexistent"));

        let mut data = TensorData::new();
        data.set(
            "_data",
            TensorValue::Scalar(crate::ScalarValue::Bytes(vec![1, 2, 3])),
        );
        router.put("_blob:chunk", data).unwrap();

        assert!(router.exists("_blob:chunk"));
    }

    #[test]
    fn test_blob_not_found() {
        let router = SlabRouter::new();
        let result = router.get("_blob:nonexistent");
        assert!(matches!(result, Err(SlabRouterError::NotFound(_))));
    }

    #[test]
    fn test_cache_not_found() {
        let router = SlabRouter::new();
        let result = router.get("_cache:nonexistent");
        assert!(matches!(result, Err(SlabRouterError::NotFound(_))));
    }

    #[test]
    fn test_embedding_without_vector() {
        let router = SlabRouter::new();

        // Put metadata without embedding vector
        let data = create_test_data();
        router.put("emb:vec1", data.clone()).unwrap();

        // Should still be retrievable from metadata
        let retrieved = router.get("emb:vec1").unwrap();
        assert_eq!(retrieved.get("name"), data.get("name"));
    }

    #[test]
    fn test_blob_key_stores_as_metadata() {
        let router = SlabRouter::new();

        // _blob: keys are stored as metadata (tensor_blob uses this for its data)
        let data = create_test_data();
        router.put("_blob:chunk1", data.clone()).unwrap();

        // Should be retrievable from metadata
        let retrieved = router.get("_blob:chunk1").unwrap();
        assert_eq!(retrieved.get("name"), data.get("name"));
    }

    #[test]
    fn test_delete_blob_without_hash() {
        let router = SlabRouter::new();

        // Put blob entry without actual bytes (no hash stored)
        let data = create_test_data();
        router.put("_blob:chunk1", data).unwrap();

        // Delete should still succeed (just cleans up metadata)
        router.delete("_blob:chunk1").unwrap();
    }

    #[test]
    fn test_scan_includes_embeddings() {
        let router = SlabRouter::new();

        // Add an embedding
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let mut data = TensorData::new();
        data.set("_embedding", TensorValue::Vector(embedding));
        router.put("emb:vec1", data).unwrap();

        // Add regular metadata
        router.put("emb:meta1", create_test_data()).unwrap();

        let results = router.scan("emb:");
        assert!(results.len() >= 2);
        assert!(results.contains(&"emb:vec1".to_string()));
    }
}
