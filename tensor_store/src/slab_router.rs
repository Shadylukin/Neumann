// SPDX-License-Identifier: MIT OR Apache-2.0
//! Slab router for directing operations to specialized storage backends.
//!
//! `SlabRouter` coordinates between the specialized slabs (`EntityIndex`, `EmbeddingSlab`,
//! `GraphTensor`, `RelationalSlab`, `MetadataSlab`, `CacheRing`, `BlobLog`) and provides
//! a unified interface similar to `TensorStore`.
//!
//! # Key Routing
//!
//! Operations are routed based on key prefixes:
//! - `emb:*` -> `EmbeddingSlab` (embedding vectors)
//! - `node:*`, `edge:*` -> `GraphTensor` (graph data)
//! - `table:*` -> `RelationalSlab` (relational rows)
//! - `_cache:*` -> `CacheRing` (cached data)
//! - Everything else (including `_blob:*`) -> `MetadataSlab` (general metadata)
//!
//! # Durability (WAL)
//!
//! When created with [`SlabRouter::with_wal`], operations can be made durable via
//! [`put_durable`](SlabRouter::put_durable) and [`delete_durable`](SlabRouter::delete_durable).
//! Cache operations are never logged (cache is transient by design).

use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::instrument;

use crate::{
    blob_log::{BlobLog, BlobLogSnapshot},
    cache_ring::{CacheRing, CacheRingSnapshot, EvictionStrategy},
    embedding_slab::{EmbeddingSlab, EmbeddingSlabSnapshot},
    entity_index::{EntityIndex, EntityIndexSnapshot},
    graph_tensor::{GraphTensor, GraphTensorSnapshot},
    metadata_slab::{MetadataSlab, MetadataSlabSnapshot},
    relational_slab::{RelationalSlab, RelationalSlabSnapshot},
    snapshot::{self, SnapshotFormatError},
    wal::{TensorWal, WalConfig, WalEntry, WalError, WalRecovery, WalStatus},
    TensorData, TensorValue,
};

/// Configuration for `SlabRouter`.
#[derive(Debug, Clone)]
pub struct SlabRouterConfig {
    /// Embedding dimension for `EmbeddingSlab`.
    pub embedding_dim: usize,
    /// Cache capacity for `CacheRing`.
    pub cache_capacity: usize,
    /// Eviction strategy for `CacheRing`.
    pub cache_strategy: EvictionStrategy,
    /// Segment size for `BlobLog`.
    pub blob_segment_size: usize,
    /// Merge threshold for `GraphTensor`.
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
    /// Optional Write-Ahead Log for durability.
    wal: Option<Mutex<TensorWal>>,
    /// Checkpoint counter for unique IDs.
    checkpoint_counter: AtomicU64,
}

impl SlabRouter {
    /// Create a new slab router with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(&SlabRouterConfig::default())
    }

    /// Create a new slab router with a capacity hint.
    ///
    /// Note: The capacity hint is informational. `BTreeMap`-based slabs don't
    /// pre-allocate. This method exists for API compatibility with `TensorStore`.
    #[must_use]
    pub fn with_capacity(_capacity: usize) -> Self {
        // BTreeMap-based slabs don't benefit from capacity hints
        // (they split nodes on demand rather than resize)
        Self::new()
    }

    /// Create a new slab router with custom configuration.
    #[must_use]
    pub fn with_config(config: &SlabRouterConfig) -> Self {
        Self {
            index: EntityIndex::new(),
            embeddings: EmbeddingSlab::with_dimension(config.embedding_dim),
            graph: GraphTensor::with_merge_threshold(config.graph_merge_threshold),
            relations: RelationalSlab::new(),
            metadata: MetadataSlab::new(),
            cache: CacheRing::new(config.cache_capacity, config.cache_strategy),
            blobs: BlobLog::new(config.blob_segment_size),
            ops_count: AtomicU64::new(0),
            wal: None,
            checkpoint_counter: AtomicU64::new(0),
        }
    }

    /// Create a new slab router with a Write-Ahead Log for durability.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL file cannot be opened or created.
    pub fn with_wal(wal_path: impl AsRef<Path>, wal_config: WalConfig) -> std::io::Result<Self> {
        Self::with_wal_and_config(wal_path, wal_config, &SlabRouterConfig::default())
    }

    /// Create a new slab router with both WAL and custom configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL file cannot be opened or created.
    pub fn with_wal_and_config(
        wal_path: impl AsRef<Path>,
        wal_config: WalConfig,
        config: &SlabRouterConfig,
    ) -> std::io::Result<Self> {
        let wal = TensorWal::open(wal_path, wal_config)?;
        Ok(Self {
            index: EntityIndex::new(),
            embeddings: EmbeddingSlab::with_dimension(config.embedding_dim),
            graph: GraphTensor::with_merge_threshold(config.graph_merge_threshold),
            relations: RelationalSlab::new(),
            metadata: MetadataSlab::new(),
            cache: CacheRing::new(config.cache_capacity, config.cache_strategy),
            blobs: BlobLog::new(config.blob_segment_size),
            ops_count: AtomicU64::new(0),
            wal: Some(Mutex::new(wal)),
            checkpoint_counter: AtomicU64::new(0),
        })
    }

    /// Store a value, routing to the appropriate slab.
    ///
    /// # Errors
    ///
    /// This function currently always succeeds but returns `Result` for API consistency.
    #[instrument(skip(self, value), fields(key = %key))]
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
    ///
    /// # Errors
    ///
    /// Returns [`SlabRouterError::NotFound`] if the key does not exist.
    #[instrument(skip(self), fields(key = %key))]
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
    /// # Errors
    ///
    /// Returns [`SlabRouterError::NotFound`] if the key does not exist.
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

        // Add from cache ring
        for key in self.cache.scan_prefix(prefix) {
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

    /// Scan entries by prefix, filtering and mapping in a single pass.
    ///
    /// More efficient than `scan()` + `get()` because:
    /// - Takes locks only once per slab
    /// - Only clones entries where the filter function returns `Some`
    /// - Avoids intermediate allocations for non-matching entries
    pub fn scan_filter_map<F, T>(&self, prefix: &str, f: F) -> Vec<T>
    where
        F: FnMut(&str, &TensorData) -> Option<T>,
    {
        // Currently all data goes through metadata slab
        self.metadata.scan_filter_map(prefix, f)
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

    /// Evict entries from the cache ring.
    pub fn evict_cache(&self, count: usize) -> usize {
        self.cache.evict(count)
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
    #[must_use]
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
            wal: None,
            checkpoint_counter: AtomicU64::new(0),
        }
    }

    /// Restore from a snapshot with WAL attached.
    ///
    /// # Errors
    ///
    /// Returns an error if the WAL file cannot be opened.
    pub fn restore_with_wal(
        snapshot: SlabRouterSnapshot,
        wal_path: impl AsRef<Path>,
        wal_config: WalConfig,
    ) -> std::io::Result<Self> {
        let wal = TensorWal::open(wal_path, wal_config)?;
        Ok(Self {
            index: EntityIndex::restore(snapshot.index),
            embeddings: EmbeddingSlab::restore(snapshot.embeddings),
            graph: GraphTensor::restore(snapshot.graph),
            relations: RelationalSlab::restore(snapshot.relations),
            metadata: MetadataSlab::restore(snapshot.metadata),
            cache: CacheRing::restore(snapshot.cache),
            blobs: BlobLog::restore(snapshot.blobs),
            ops_count: AtomicU64::new(0),
            wal: Some(Mutex::new(wal)),
            checkpoint_counter: AtomicU64::new(0),
        })
    }

    /// Save to a file (v3 format with auto-detection on load).
    ///
    /// # Errors
    ///
    /// Returns [`SnapshotFormatError`] if file I/O or serialization fails.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), SnapshotFormatError> {
        snapshot::save_v3(self, path)
    }

    /// Load from a file (auto-detects v2/v3 format).
    ///
    /// # Errors
    ///
    /// Returns [`SnapshotFormatError`] if file I/O, deserialization, or format validation fails.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, SnapshotFormatError> {
        snapshot::load(path)
    }

    /// Serialize to bytes (v3 format).
    ///
    /// # Errors
    ///
    /// Returns [`SnapshotFormatError`] if serialization fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SnapshotFormatError> {
        let router_snapshot = self.snapshot();
        let entry_count = (self.len() + self.index.len()) as u64;

        let v3 = snapshot::V3Snapshot {
            header: snapshot::SnapshotHeader::new(entry_count),
            router: router_snapshot,
        };

        bitcode::serialize(&v3).map_err(SnapshotFormatError::from)
    }

    /// Deserialize from bytes (v3 format).
    ///
    /// # Errors
    ///
    /// Returns [`SnapshotFormatError`] if deserialization or header validation fails.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, SnapshotFormatError> {
        let v3: snapshot::V3Snapshot =
            bitcode::deserialize(bytes).map_err(SnapshotFormatError::from)?;
        v3.header.validate()?;
        Ok(Self::restore(v3.router))
    }

    // ========== WAL / Durable Operations ==========

    /// Store a value durably, logging to WAL before applying.
    ///
    /// Cache operations are never logged (cache is transient by design).
    ///
    /// # Errors
    ///
    /// Returns an error if WAL append fails. If no WAL is configured, falls back to
    /// non-durable `put`.
    pub fn put_durable(&self, key: &str, value: TensorData) -> Result<(), SlabRouterError> {
        // Cache operations are not durable
        if Self::classify_key(key) == KeyClass::Cache {
            return self.put(key, value);
        }

        // Log to WAL first (if configured)
        if let Some(wal_mutex) = &self.wal {
            let mut wal = wal_mutex.lock();

            // Log embedding separately if present
            if let Some(TensorValue::Vector(embedding)) = value.get("_embedding") {
                let entity_id = self.index.get_or_create(key);
                let entry = WalEntry::EmbeddingSet {
                    entity_id,
                    embedding: embedding.clone(),
                };
                wal.append(&entry).map_err(|e| {
                    SlabRouterError::WalError(format!("Failed to log embedding: {e}"))
                })?;
            }

            // Log metadata set
            let entry = WalEntry::MetadataSet {
                key: key.to_string(),
                data: value.clone(),
            };
            wal.append(&entry)
                .map_err(|e| SlabRouterError::WalError(format!("Failed to log put: {e}")))?;
        }

        // Apply to in-memory state
        self.put(key, value)
    }

    /// Delete a value durably, logging to WAL before applying.
    ///
    /// Cache operations are never logged (cache is transient by design).
    ///
    /// # Errors
    ///
    /// Returns an error if the key does not exist or WAL append fails.
    pub fn delete_durable(&self, key: &str) -> Result<(), SlabRouterError> {
        // Cache operations are not durable
        if Self::classify_key(key) == KeyClass::Cache {
            return self.delete(key);
        }

        // Log to WAL first (if configured)
        if let Some(wal_mutex) = &self.wal {
            let mut wal = wal_mutex.lock();

            // Log embedding delete if key is in entity index
            if let Some(entity_id) = self.index.get(key) {
                let entry = WalEntry::EmbeddingDelete { entity_id };
                wal.append(&entry).map_err(|e| {
                    SlabRouterError::WalError(format!("Failed to log embedding delete: {e}"))
                })?;

                let entry = WalEntry::EntityRemove {
                    key: key.to_string(),
                };
                wal.append(&entry).map_err(|e| {
                    SlabRouterError::WalError(format!("Failed to log entity remove: {e}"))
                })?;
            }

            // Log metadata delete
            let entry = WalEntry::MetadataDelete {
                key: key.to_string(),
            };
            wal.append(&entry)
                .map_err(|e| SlabRouterError::WalError(format!("Failed to log delete: {e}")))?;
        }

        // Apply to in-memory state
        self.delete(key)
    }

    /// Create a checkpoint by saving a snapshot and marking WAL position.
    ///
    /// After checkpoint, the WAL is truncated since the snapshot contains all state.
    ///
    /// # Errors
    ///
    /// Returns an error if snapshot save or WAL operations fail.
    pub fn checkpoint(&self, snapshot_path: &Path) -> Result<u64, SlabRouterError> {
        // Save snapshot first
        self.save_to_file(snapshot_path)
            .map_err(|e| SlabRouterError::WalError(format!("Failed to save snapshot: {e}")))?;

        let checkpoint_id = self.checkpoint_counter.fetch_add(1, Ordering::SeqCst);

        // Log checkpoint marker and truncate WAL
        if let Some(wal_mutex) = &self.wal {
            let mut wal = wal_mutex.lock();

            let entry = WalEntry::Checkpoint {
                snapshot_id: checkpoint_id,
            };
            wal.append(&entry)
                .map_err(|e| SlabRouterError::WalError(format!("Failed to log checkpoint: {e}")))?;

            // Truncate WAL after successful checkpoint
            wal.truncate()
                .map_err(|e| SlabRouterError::WalError(format!("Failed to truncate WAL: {e}")))?;
        }

        Ok(checkpoint_id)
    }

    /// Recover from a snapshot and replay WAL.
    ///
    /// # Errors
    ///
    /// Returns an error if snapshot loading or WAL replay fails.
    pub fn recover(
        wal_path: impl AsRef<Path>,
        wal_config: &WalConfig,
        snapshot_path: Option<&Path>,
    ) -> Result<Self, SlabRouterError> {
        // Load snapshot if provided
        let mut router = if let Some(path) = snapshot_path {
            if path.exists() {
                Self::load_from_file(path).map_err(|e| {
                    SlabRouterError::WalError(format!("Failed to load snapshot: {e}"))
                })?
            } else {
                Self::new()
            }
        } else {
            Self::new()
        };

        // Open WAL and replay
        let wal = TensorWal::open(&wal_path, wal_config.clone())
            .map_err(|e| SlabRouterError::WalError(format!("Failed to open WAL: {e}")))?;

        let recovery = WalRecovery::from_wal(&wal)
            .map_err(|e| SlabRouterError::WalError(format!("Failed to replay WAL: {e}")))?;

        // Apply recovered operations
        for entry in recovery.all_operations() {
            router.apply_wal_entry(entry);
        }

        // Attach the WAL
        router.wal = Some(Mutex::new(wal));

        Ok(router)
    }

    /// Apply a single WAL entry to in-memory state.
    fn apply_wal_entry(&self, entry: &WalEntry) {
        match entry {
            WalEntry::MetadataSet { key, data } => {
                self.metadata.set(key, data.clone());
                // Also update embeddings if present
                if let Some(TensorValue::Vector(vec)) = data.get("_embedding") {
                    let entity_id = self.index.get_or_create(key);
                    if let Err(e) = self.embeddings.set(entity_id, vec) {
                        tracing::warn!(
                            entity_id = %entity_id.as_u64(),
                            key = %key,
                            error = %e,
                            "Failed to restore embedding during WAL replay"
                        );
                    }
                }
            },
            WalEntry::MetadataDelete { key } => {
                self.metadata.delete(key);
            },
            WalEntry::EmbeddingSet {
                entity_id,
                embedding,
            } => {
                if let Err(e) = self.embeddings.set(*entity_id, embedding) {
                    tracing::warn!(
                        entity_id = %entity_id.as_u64(),
                        error = %e,
                        "Failed to restore embedding during WAL replay"
                    );
                }
            },
            WalEntry::EmbeddingDelete { entity_id } => {
                self.embeddings.delete(*entity_id);
            },
            WalEntry::EntityCreate { key, .. } => {
                // Re-establish key -> entity_id mapping (ID may differ from original)
                let _ = self.index.get_or_create(key);
            },
            WalEntry::EntityRemove { key } => {
                self.index.remove(key);
            },
            // Transaction markers are handled by WalRecovery
            WalEntry::TxBegin { .. }
            | WalEntry::TxCommit { .. }
            | WalEntry::TxAbort { .. }
            | WalEntry::Checkpoint { .. } => {},
        }
    }

    /// Get WAL status if WAL is configured.
    #[must_use]
    pub fn wal_status(&self) -> Option<WalStatus> {
        self.wal.as_ref().map(|m| m.lock().status())
    }

    /// Check if WAL is enabled.
    #[must_use]
    pub const fn has_wal(&self) -> bool {
        self.wal.is_some()
    }

    /// Flush and sync the WAL to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if WAL fsync fails.
    pub fn wal_sync(&self) -> Result<(), SlabRouterError> {
        if let Some(wal_mutex) = &self.wal {
            let mut wal = wal_mutex.lock();
            wal.fsync()
                .map_err(|e| SlabRouterError::WalError(format!("Failed to sync WAL: {e}")))?;
        }
        Ok(())
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

    /// Estimate the byte size of a `TensorData` for cache scoring.
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

/// Errors from `SlabRouter` operations.
#[derive(Debug, Clone)]
pub enum SlabRouterError {
    /// Key not found in any slab.
    NotFound(String),
    /// Embedding operation failed.
    EmbeddingError(String),
    /// Graph operation failed.
    GraphError(String),
    /// Relational operation failed.
    RelationalError(String),
    /// WAL operation failed.
    WalError(String),
}

impl std::fmt::Display for SlabRouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(key) => write!(f, "not found: {key}"),
            Self::EmbeddingError(msg) => write!(f, "embedding error: {msg}"),
            Self::GraphError(msg) => write!(f, "graph error: {msg}"),
            Self::RelationalError(msg) => write!(f, "relational error: {msg}"),
            Self::WalError(msg) => write!(f, "WAL error: {msg}"),
        }
    }
}

impl From<WalError> for SlabRouterError {
    fn from(err: WalError) -> Self {
        Self::WalError(err.to_string())
    }
}

impl std::error::Error for SlabRouterError {}

/// Serializable snapshot of all slab state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlabRouterSnapshot {
    /// Entity index state.
    pub index: EntityIndexSnapshot,
    /// Embedding slab state.
    pub embeddings: EmbeddingSlabSnapshot,
    /// Graph tensor state.
    pub graph: GraphTensorSnapshot,
    /// Relational tables state.
    pub relations: RelationalSlabSnapshot,
    /// Metadata slab state.
    pub metadata: MetadataSlabSnapshot,
    /// Cache ring state.
    pub cache: CacheRingSnapshot<TensorData>,
    /// Blob log state.
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
        let router = SlabRouter::with_config(&config);
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

    #[test]
    fn test_scan_includes_cache_entries() {
        let router = SlabRouter::new();

        // Add cache entry via cache ring prefix
        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(crate::ScalarValue::Int(42)));
        router.put("_cache:cached_item", data).unwrap();

        // Add regular metadata entry with same prefix
        router.put("_cache:other_item", create_test_data()).unwrap();

        let results = router.scan("_cache:");
        assert!(results.len() >= 2);
        assert!(results.contains(&"_cache:cached_item".to_string()));
        assert!(results.contains(&"_cache:other_item".to_string()));
    }

    #[test]
    fn test_scan_entity_index_deduplication() {
        let router = SlabRouter::new();

        // Add entry that will appear in both metadata and entity index
        let mut data = TensorData::new();
        data.set("id", TensorValue::Scalar(crate::ScalarValue::Int(1)));
        router.put("entity:item1", data).unwrap();

        // Add another entity entry
        let mut data2 = TensorData::new();
        data2.set("id", TensorValue::Scalar(crate::ScalarValue::Int(2)));
        router.put("entity:item2", data2).unwrap();

        let results = router.scan("entity:");
        // Should not have duplicates
        let mut deduped: Vec<_> = results.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(results.len(), deduped.len());
    }

    // ========== WAL Integration Tests ==========

    #[test]
    fn test_with_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();
        assert!(router.has_wal());
        assert!(wal_path.exists());
    }

    #[test]
    fn test_with_wal_and_config() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let config = SlabRouterConfig {
            embedding_dim: 128,
            ..Default::default()
        };

        let router =
            SlabRouter::with_wal_and_config(&wal_path, WalConfig::default(), &config).unwrap();
        assert!(router.has_wal());
    }

    #[test]
    fn test_has_wal_false_without_wal() {
        let router = SlabRouter::new();
        assert!(!router.has_wal());
    }

    #[test]
    fn test_wal_status() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        let status = router.wal_status().unwrap();
        assert_eq!(status.path, wal_path);
        assert!(status.checksums_enabled);
    }

    #[test]
    fn test_wal_status_none_without_wal() {
        let router = SlabRouter::new();
        assert!(router.wal_status().is_none());
    }

    #[test]
    fn test_put_durable() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        router.put_durable("key1", create_test_data()).unwrap();

        assert!(router.exists("key1"));

        // Check WAL has entry
        let status = router.wal_status().unwrap();
        assert!(status.entry_count > 0);
    }

    #[test]
    fn test_put_durable_without_wal() {
        let router = SlabRouter::new();

        // Should still work, just not durable
        router.put_durable("key1", create_test_data()).unwrap();
        assert!(router.exists("key1"));
    }

    #[test]
    fn test_put_durable_with_embedding() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let mut data = TensorData::new();
        data.set("_embedding", TensorValue::Vector(embedding));
        data.set(
            "name",
            TensorValue::Scalar(crate::ScalarValue::String("test".to_string())),
        );

        router.put_durable("emb:vec1", data).unwrap();

        let retrieved = router.get("emb:vec1").unwrap();
        assert!(retrieved.get("_embedding").is_some());
    }

    #[test]
    fn test_put_durable_cache_not_logged() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        // Cache operations should not be logged
        router
            .put_durable("_cache:item", create_test_data())
            .unwrap();

        // Entry count should still be 0 (cache is not durable)
        let status = router.wal_status().unwrap();
        assert_eq!(status.entry_count, 0);
    }

    #[test]
    fn test_delete_durable() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        router.put_durable("key1", create_test_data()).unwrap();
        assert!(router.exists("key1"));

        router.delete_durable("key1").unwrap();
        assert!(!router.exists("key1"));
    }

    #[test]
    fn test_delete_durable_without_wal() {
        let router = SlabRouter::new();

        router.put("key1", create_test_data()).unwrap();
        router.delete_durable("key1").unwrap();
        assert!(!router.exists("key1"));
    }

    #[test]
    fn test_delete_durable_cache_not_logged() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        router.put("_cache:item", create_test_data()).unwrap();

        let status_before = router.wal_status().unwrap();
        router.delete_durable("_cache:item").unwrap();
        let status_after = router.wal_status().unwrap();

        // No additional entries for cache delete
        assert_eq!(status_before.entry_count, status_after.entry_count);
    }

    #[test]
    fn test_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let snapshot_path = dir.path().join("snapshot.bin");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        router.put_durable("key1", create_test_data()).unwrap();
        router.put_durable("key2", create_test_data()).unwrap();

        let checkpoint_id = router.checkpoint(&snapshot_path).unwrap();
        assert_eq!(checkpoint_id, 0);

        // Snapshot should exist
        assert!(snapshot_path.exists());

        // WAL should be truncated
        let status = router.wal_status().unwrap();
        assert_eq!(status.entry_count, 0);
    }

    #[test]
    fn test_checkpoint_without_wal() {
        let dir = tempfile::tempdir().unwrap();
        let snapshot_path = dir.path().join("snapshot.bin");

        let router = SlabRouter::new();
        router.put("key1", create_test_data()).unwrap();

        // Should still create snapshot
        let checkpoint_id = router.checkpoint(&snapshot_path).unwrap();
        assert_eq!(checkpoint_id, 0);
        assert!(snapshot_path.exists());
    }

    #[test]
    fn test_recover_from_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Create and populate router
        {
            let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();
            router.put_durable("key1", create_test_data()).unwrap();

            let mut data2 = TensorData::new();
            data2.set("value", TensorValue::Scalar(crate::ScalarValue::Int(42)));
            router.put_durable("key2", data2).unwrap();
        }

        // Recover from WAL
        let recovered = SlabRouter::recover(&wal_path, &WalConfig::default(), None).unwrap();

        assert!(recovered.exists("key1"));
        assert!(recovered.exists("key2"));
        assert!(recovered.has_wal());
    }

    #[test]
    fn test_recover_from_snapshot_and_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let snapshot_path = dir.path().join("snapshot.bin");

        // Create initial state and checkpoint
        {
            let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();
            router.put_durable("key1", create_test_data()).unwrap();
            router.checkpoint(&snapshot_path).unwrap();

            // Add more data after checkpoint
            let mut data2 = TensorData::new();
            data2.set("value", TensorValue::Scalar(crate::ScalarValue::Int(99)));
            router.put_durable("key2", data2).unwrap();
        }

        // Recover from snapshot + WAL
        let recovered =
            SlabRouter::recover(&wal_path, &WalConfig::default(), Some(&snapshot_path)).unwrap();

        assert!(recovered.exists("key1"));
        assert!(recovered.exists("key2"));
    }

    #[test]
    fn test_recover_missing_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let snapshot_path = dir.path().join("nonexistent.bin");

        // Create WAL with data
        {
            let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();
            router.put_durable("key1", create_test_data()).unwrap();
        }

        // Recover should work with missing snapshot
        let recovered =
            SlabRouter::recover(&wal_path, &WalConfig::default(), Some(&snapshot_path)).unwrap();

        assert!(recovered.exists("key1"));
    }

    #[test]
    fn test_wal_sync() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();
        router.put_durable("key1", create_test_data()).unwrap();

        // Sync should succeed
        router.wal_sync().unwrap();
    }

    #[test]
    fn test_wal_sync_without_wal() {
        let router = SlabRouter::new();
        // Should be no-op, not error
        router.wal_sync().unwrap();
    }

    #[test]
    fn test_restore_with_wal() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        // Create and save a router
        let original = SlabRouter::new();
        original.put("key1", create_test_data()).unwrap();
        let snapshot = original.snapshot();

        // Restore with WAL
        let restored =
            SlabRouter::restore_with_wal(snapshot, &wal_path, WalConfig::default()).unwrap();

        assert!(restored.exists("key1"));
        assert!(restored.has_wal());
    }

    #[test]
    fn test_error_display_wal_variant() {
        let err = SlabRouterError::WalError("test error".to_string());
        let msg = err.to_string();
        assert!(msg.contains("WAL"));
        assert!(msg.contains("test error"));
    }

    #[test]
    fn test_wal_error_from_conversion() {
        use crate::wal::WalError;
        let wal_err = WalError::NoActiveTransaction;
        let router_err: SlabRouterError = wal_err.into();
        assert!(matches!(router_err, SlabRouterError::WalError(_)));
    }

    #[test]
    fn test_apply_wal_entry_metadata_set() {
        let router = SlabRouter::new();

        let entry = WalEntry::MetadataSet {
            key: "test_key".to_string(),
            data: create_test_data(),
        };

        router.apply_wal_entry(&entry);
        assert!(router.exists("test_key"));
    }

    #[test]
    fn test_apply_wal_entry_metadata_delete() {
        let router = SlabRouter::new();
        router.put("test_key", create_test_data()).unwrap();

        let entry = WalEntry::MetadataDelete {
            key: "test_key".to_string(),
        };

        router.apply_wal_entry(&entry);
        assert!(!router.metadata.contains("test_key"));
    }

    #[test]
    fn test_apply_wal_entry_embedding_set() {
        let router = SlabRouter::new();

        let entry = WalEntry::EmbeddingSet {
            entity_id: crate::EntityId(42),
            embedding: vec![1.0, 2.0, 3.0],
        };

        router.apply_wal_entry(&entry);
        // Embedding should be stored (though dimension mismatch)
    }

    #[test]
    fn test_apply_wal_entry_embedding_delete() {
        let router = SlabRouter::new();

        let entry = WalEntry::EmbeddingDelete {
            entity_id: crate::EntityId(42),
        };

        // Should not error even if entity doesn't exist
        router.apply_wal_entry(&entry);
    }

    #[test]
    fn test_apply_wal_entry_entity_create() {
        let router = SlabRouter::new();

        let entry = WalEntry::EntityCreate {
            key: "entity_key".to_string(),
            entity_id: crate::EntityId(123),
        };

        router.apply_wal_entry(&entry);
        assert!(router.index.contains("entity_key"));
    }

    #[test]
    fn test_apply_wal_entry_entity_remove() {
        let router = SlabRouter::new();
        let _ = router.index.get_or_create("entity_key");

        let entry = WalEntry::EntityRemove {
            key: "entity_key".to_string(),
        };

        router.apply_wal_entry(&entry);
        assert!(!router.index.contains("entity_key"));
    }

    #[test]
    fn test_apply_wal_entry_transaction_markers() {
        let router = SlabRouter::new();

        // These should be no-ops (handled by WalRecovery)
        router.apply_wal_entry(&WalEntry::TxBegin { tx_id: 1 });
        router.apply_wal_entry(&WalEntry::TxCommit { tx_id: 1 });
        router.apply_wal_entry(&WalEntry::TxAbort { tx_id: 1 });
        router.apply_wal_entry(&WalEntry::Checkpoint { snapshot_id: 1 });
    }

    #[test]
    fn test_apply_wal_entry_metadata_set_with_embedding() {
        let router = SlabRouter::new();

        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let mut data = TensorData::new();
        data.set("_embedding", TensorValue::Vector(embedding));
        data.set(
            "name",
            TensorValue::Scalar(crate::ScalarValue::String("test".to_string())),
        );

        let entry = WalEntry::MetadataSet {
            key: "emb:test".to_string(),
            data,
        };

        router.apply_wal_entry(&entry);

        // Both metadata and embedding should be stored
        assert!(router.metadata.contains("emb:test"));
        let entity_id = router.index.get("emb:test").unwrap();
        assert!(router.embeddings.get(entity_id).is_some());
    }

    #[test]
    fn test_multiple_checkpoints() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let snapshot_path = dir.path().join("snapshot.bin");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        router.put_durable("key1", create_test_data()).unwrap();
        let id1 = router.checkpoint(&snapshot_path).unwrap();

        router.put_durable("key2", create_test_data()).unwrap();
        let id2 = router.checkpoint(&snapshot_path).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_delete_durable_with_embedding() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");

        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        // Put embedding
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let mut data = TensorData::new();
        data.set("_embedding", TensorValue::Vector(embedding));
        router.put_durable("emb:vec1", data).unwrap();

        // Delete should log embedding delete and entity remove
        router.delete_durable("emb:vec1").unwrap();

        assert!(!router.exists("emb:vec1"));
    }

    #[test]
    fn test_crash_recovery_simulation() {
        let dir = tempfile::tempdir().unwrap();
        let wal_path = dir.path().join("test.wal");
        let snapshot_path = dir.path().join("snapshot.bin");

        // Simulate initial state
        {
            let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

            // Add data and checkpoint
            router.put_durable("persisted", create_test_data()).unwrap();
            router.checkpoint(&snapshot_path).unwrap();

            // Add more data after checkpoint (this will be in WAL only)
            let mut data = TensorData::new();
            data.set("value", TensorValue::Scalar(crate::ScalarValue::Int(42)));
            router.put_durable("in_wal_only", data).unwrap();

            // "Crash" - drop router without clean shutdown
        }

        // Recover
        let recovered =
            SlabRouter::recover(&wal_path, &WalConfig::default(), Some(&snapshot_path)).unwrap();

        // Both should be present
        assert!(recovered.exists("persisted"));
        assert!(recovered.exists("in_wal_only"));

        // Verify data integrity
        let retrieved = recovered.get("in_wal_only").unwrap();
        if let Some(TensorValue::Scalar(crate::ScalarValue::Int(v))) = retrieved.get("value") {
            assert_eq!(*v, 42);
        } else {
            panic!("Expected Int value");
        }
    }

    // ========== Phase 2: Lock Contention Stress Tests ==========

    #[test]
    fn test_slab_router_high_contention() {
        use std::sync::Arc;
        use std::thread;

        // 16 threads: mixed put/get/delete across all slab types
        let router = Arc::new(SlabRouter::new());
        let mut handles = vec![];

        // 4 put threads for metadata
        for t in 0..4u64 {
            let r = Arc::clone(&router);
            handles.push(thread::spawn(move || {
                for i in 0..200u64 {
                    let mut data = TensorData::new();
                    data.set(
                        "value",
                        TensorValue::Scalar(crate::ScalarValue::Int((t * 1000 + i) as i64)),
                    );
                    let _ = r.put(&format!("meta:t{}_i{}", t, i), data);
                }
            }));
        }

        // 4 put threads for embeddings
        for t in 0..4u64 {
            let r = Arc::clone(&router);
            handles.push(thread::spawn(move || {
                for i in 0..50u64 {
                    let embedding: Vec<f32> = (0..384).map(|j| (t * 1000 + i + j) as f32).collect();
                    let mut data = TensorData::new();
                    data.set("_embedding", TensorValue::Vector(embedding));
                    let _ = r.put(&format!("emb:t{}_i{}", t, i), data);
                }
            }));
        }

        // 4 get threads
        for _ in 0..4 {
            let r = Arc::clone(&router);
            handles.push(thread::spawn(move || {
                for i in 0..200u64 {
                    let _ = r.get(&format!("meta:t0_i{}", i % 200));
                    let _ = r.exists(&format!("emb:t0_i{}", i % 50));
                }
            }));
        }

        // 4 scan/delete threads
        for t in 0..4u64 {
            let r = Arc::clone(&router);
            handles.push(thread::spawn(move || {
                for i in 0..50u64 {
                    let _ = r.scan(&format!("meta:t{}_", t));
                    // Try delete (may or may not exist)
                    let _ = r.delete(&format!("meta:t{}_i{}", t, i));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify router is still consistent
        assert!(router.ops_count() > 0);
    }

    // ========== Phase 3: Negative Path Tests ==========

    #[test]
    fn test_slab_router_error_not_found() {
        let router = SlabRouter::new();

        // get() on non-existent key returns proper error
        let result = router.get("nonexistent_key");
        assert!(matches!(result, Err(SlabRouterError::NotFound(_))));

        // Verify error message contains the key
        if let Err(SlabRouterError::NotFound(key)) = result {
            assert_eq!(key, "nonexistent_key");
        }
    }

    #[test]
    fn test_slab_router_error_not_found_embedding() {
        let router = SlabRouter::new();

        // get() on non-existent embedding key
        let result = router.get("emb:nonexistent");
        assert!(matches!(result, Err(SlabRouterError::NotFound(_))));
    }

    #[test]
    fn test_slab_router_delete_not_found() {
        let router = SlabRouter::new();

        // delete() on non-existent key returns proper error
        let result = router.delete("nonexistent_key");
        assert!(matches!(result, Err(SlabRouterError::NotFound(_))));
    }

    #[test]
    fn test_slab_router_error_embedding_dimension_mismatch() {
        let router = SlabRouter::new();

        // First, insert an embedding with correct dimension (384)
        let embedding: Vec<f32> = (0..384).map(|i| i as f32).collect();
        let mut data = TensorData::new();
        data.set("_embedding", TensorValue::Vector(embedding));
        router.put("emb:correct", data).unwrap();

        // Now try to insert an embedding with wrong dimension
        // The router silently handles this by falling back to metadata only
        let wrong_embedding: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let mut wrong_data = TensorData::new();
        wrong_data.set("_embedding", TensorValue::Vector(wrong_embedding));
        // This should succeed (falls back to metadata storage)
        router.put("emb:wrong_dim", wrong_data).unwrap();

        // The key should exist in metadata
        assert!(router.exists("emb:wrong_dim"));
    }

    #[test]
    fn test_slab_router_error_variants_coverage() {
        // Test all error variant Display implementations
        let not_found = SlabRouterError::NotFound("key".to_string());
        assert!(not_found.to_string().contains("not found"));
        assert!(not_found.to_string().contains("key"));

        let embedding_err = SlabRouterError::EmbeddingError("dim mismatch".to_string());
        assert!(embedding_err.to_string().contains("embedding"));
        assert!(embedding_err.to_string().contains("dim mismatch"));

        let graph_err = SlabRouterError::GraphError("cycle detected".to_string());
        assert!(graph_err.to_string().contains("graph"));

        let relational_err = SlabRouterError::RelationalError("schema error".to_string());
        assert!(relational_err.to_string().contains("relational"));

        let wal_err = SlabRouterError::WalError("io failure".to_string());
        assert!(wal_err.to_string().contains("WAL"));
        assert!(wal_err.to_string().contains("io failure"));
    }

    #[test]
    fn test_slab_router_error_is_std_error() {
        // Verify SlabRouterError implements std::error::Error
        let err: Box<dyn std::error::Error> =
            Box::new(SlabRouterError::NotFound("test".to_string()));
        assert!(err.to_string().contains("not found"));

        // Test source() returns None (no nested error)
        assert!(err.source().is_none());
    }
}
