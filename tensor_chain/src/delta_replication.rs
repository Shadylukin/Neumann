//! Delta-compressed replication for bandwidth-efficient state transfer.
//!
//! Uses archetype-based delta encoding to reduce replication bandwidth by 4-6x.
//! Entities are encoded as (archetype_id + sparse_delta) and batched for transfer.
//!
//! ## Backpressure
//!
//! The replication queue has bounded capacity. When full, `queue_update()` returns
//! `Err(ChainError::QueueFull)` to signal backpressure. Callers should either:
//! - Retry after a delay
//! - Use the async `queue_update_async()` which waits for space
//! - Start auto-drain via `start_auto_drain()` for background sending

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tensor_store::ArchetypeRegistry;
use tokio::sync::mpsc;

use crate::{
    block::NodeId,
    error::{ChainError, Result},
    network::{Message, Transport},
};

/// A delta-encoded update for replication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaUpdate {
    /// Key being updated.
    pub key: String,
    /// ID of the reference archetype.
    pub archetype_id: u32,
    /// Sparse delta from archetype.
    pub delta_indices: Vec<u32>,
    /// Delta values at each index.
    pub delta_values: Vec<f32>,
    /// Update version (for ordering).
    pub version: u64,
    /// Original embedding dimension.
    pub dimension: usize,
}

impl DeltaUpdate {
    /// Create a delta update from embedding and archetype.
    pub fn from_embedding(
        key: String,
        embedding: &[f32],
        registry: &ArchetypeRegistry,
        threshold: f32,
        version: u64,
    ) -> Option<Self> {
        let delta = registry.encode(embedding, threshold)?;
        let sparse = delta.to_sparse_delta();

        Some(Self {
            key,
            archetype_id: delta.archetype_id() as u32,
            delta_indices: sparse.positions().to_vec(),
            delta_values: sparse.values().to_vec(),
            version,
            dimension: embedding.len(),
        })
    }

    /// Create a full update (no delta compression).
    pub fn full(key: String, embedding: &[f32], version: u64) -> Self {
        let indices: Vec<u32> = (0..embedding.len() as u32).collect();
        Self {
            key,
            archetype_id: u32::MAX, // Sentinel for full update
            delta_indices: indices,
            delta_values: embedding.to_vec(),
            version,
            dimension: embedding.len(),
        }
    }

    /// Check if this is a full update (not delta-compressed).
    pub fn is_full_update(&self) -> bool {
        self.archetype_id == u32::MAX
    }

    /// Get the number of non-zero delta values.
    pub fn nnz(&self) -> usize {
        self.delta_values.len()
    }

    /// Memory bytes used by this update.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.key.len()
            + self.delta_indices.len() * std::mem::size_of::<u32>()
            + self.delta_values.len() * std::mem::size_of::<f32>()
    }

    /// Compression ratio compared to full embedding.
    pub fn compression_ratio(&self) -> f32 {
        let full_bytes = self.dimension * std::mem::size_of::<f32>();
        if full_bytes == 0 {
            return 1.0;
        }
        full_bytes as f32 / self.memory_bytes() as f32
    }

    /// Decode back to dense embedding using archetype registry.
    pub fn decode(&self, registry: &ArchetypeRegistry) -> Option<Vec<f32>> {
        if self.is_full_update() {
            // Full update - just return the values
            let mut result = vec![0.0; self.dimension];
            for (&idx, &val) in self.delta_indices.iter().zip(self.delta_values.iter()) {
                if (idx as usize) < result.len() {
                    result[idx as usize] = val;
                }
            }
            return Some(result);
        }

        let archetype = registry.get(self.archetype_id as usize)?;
        let mut result = archetype.to_vec();

        // Apply sparse delta
        for (&idx, &val) in self.delta_indices.iter().zip(self.delta_values.iter()) {
            if (idx as usize) < result.len() {
                result[idx as usize] += val;
            }
        }

        Some(result)
    }
}

/// Batch of delta updates for efficient network transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaBatch {
    /// Updates in this batch.
    pub updates: Vec<DeltaUpdate>,
    /// Source node ID.
    pub source: NodeId,
    /// Batch sequence number.
    pub sequence: u64,
    /// Whether this completes a sync operation.
    pub is_final: bool,
}

impl DeltaBatch {
    /// Create a new batch.
    pub fn new(source: NodeId, sequence: u64) -> Self {
        Self {
            updates: Vec::new(),
            source,
            sequence,
            is_final: false,
        }
    }

    /// Add an update to the batch.
    pub fn add(&mut self, update: DeltaUpdate) {
        self.updates.push(update);
    }

    /// Mark as final batch in sync.
    pub fn finalize(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Get total memory bytes.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.source.len()
            + self
                .updates
                .iter()
                .map(DeltaUpdate::memory_bytes)
                .sum::<usize>()
    }

    /// Get average compression ratio.
    pub fn avg_compression_ratio(&self) -> f32 {
        if self.updates.is_empty() {
            return 1.0;
        }
        let total: f32 = self
            .updates
            .iter()
            .map(DeltaUpdate::compression_ratio)
            .sum();
        total / self.updates.len() as f32
    }

    /// Number of updates.
    pub fn len(&self) -> usize {
        self.updates.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }
}

/// Statistics for delta replication.
#[derive(Debug, Default)]
pub struct ReplicationStats {
    /// Total bytes sent.
    bytes_sent: AtomicU64,
    /// Bytes saved vs full replication.
    bytes_saved: AtomicU64,
    /// Number of updates sent.
    updates_sent: AtomicU64,
    /// Number of batches sent.
    batches_sent: AtomicU64,
    /// Average compression ratio (stored as fixed-point: value * 1000).
    avg_compression_ratio_fp: AtomicU64,
    /// Number of full updates (not delta-compressed).
    full_updates: AtomicU64,
    /// Current queue depth.
    queue_depth: AtomicUsize,
    /// Number of backpressure events (queue full rejections).
    backpressure_events: AtomicU64,
    /// Number of auto-drain operations.
    auto_drains: AtomicU64,
    /// Peak queue depth observed.
    peak_queue_depth: AtomicUsize,
}

/// Snapshot of replication statistics for external consumption.
#[derive(Debug, Clone, Default)]
pub struct ReplicationStatsSnapshot {
    /// Total bytes sent.
    pub bytes_sent: u64,
    /// Bytes saved vs full replication.
    pub bytes_saved: u64,
    /// Number of updates sent.
    pub updates_sent: u64,
    /// Number of batches sent.
    pub batches_sent: u64,
    /// Average compression ratio.
    pub avg_compression_ratio: f32,
    /// Number of full updates (not delta-compressed).
    pub full_updates: u64,
    /// Current queue depth.
    pub queue_depth: usize,
    /// Number of backpressure events.
    pub backpressure_events: u64,
    /// Number of auto-drain operations.
    pub auto_drains: u64,
    /// Peak queue depth observed.
    pub peak_queue_depth: usize,
}

impl ReplicationStats {
    /// Create new stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Update stats with a batch (thread-safe).
    pub fn record_batch(&self, batch: &DeltaBatch, full_bytes: usize) {
        let batch_bytes = batch.memory_bytes();
        self.bytes_sent
            .fetch_add(batch_bytes as u64, Ordering::Relaxed);
        self.bytes_saved.fetch_add(
            full_bytes.saturating_sub(batch_bytes) as u64,
            Ordering::Relaxed,
        );
        self.updates_sent
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        let batches = self.batches_sent.fetch_add(1, Ordering::Relaxed) + 1;

        // Update running average (using fixed-point for atomics)
        let batch_ratio_fp = (batch.avg_compression_ratio() * 1000.0) as u64;
        if batches == 1 {
            self.avg_compression_ratio_fp
                .store(batch_ratio_fp, Ordering::Relaxed);
        } else {
            // Approximate running average: new = old * 0.9 + new * 0.1
            let old = self.avg_compression_ratio_fp.load(Ordering::Relaxed);
            let updated = (old * 9 + batch_ratio_fp) / 10;
            self.avg_compression_ratio_fp
                .store(updated, Ordering::Relaxed);
        }

        // Count full updates
        let full_count = batch.updates.iter().filter(|u| u.is_full_update()).count() as u64;
        self.full_updates.fetch_add(full_count, Ordering::Relaxed);
    }

    /// Record a backpressure event.
    pub fn record_backpressure(&self) {
        self.backpressure_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an auto-drain operation.
    pub fn record_auto_drain(&self) {
        self.auto_drains.fetch_add(1, Ordering::Relaxed);
    }

    /// Update queue depth.
    pub fn set_queue_depth(&self, depth: usize) {
        self.queue_depth.store(depth, Ordering::Relaxed);
        // Update peak if needed
        self.peak_queue_depth.fetch_max(depth, Ordering::Relaxed);
    }

    /// Increment queue depth and return new value.
    pub fn increment_queue_depth(&self) -> usize {
        let new_depth = self.queue_depth.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak_queue_depth
            .fetch_max(new_depth, Ordering::Relaxed);
        new_depth
    }

    /// Decrement queue depth.
    pub fn decrement_queue_depth(&self) {
        self.queue_depth.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current queue depth.
    pub fn queue_depth(&self) -> usize {
        self.queue_depth.load(Ordering::Relaxed)
    }

    /// Get effective compression ratio.
    pub fn effective_compression(&self) -> f32 {
        let sent = self.bytes_sent.load(Ordering::Relaxed);
        if sent == 0 {
            return 1.0;
        }
        let saved = self.bytes_saved.load(Ordering::Relaxed);
        (sent + saved) as f32 / sent as f32
    }

    /// Get a snapshot of current stats.
    pub fn snapshot(&self) -> ReplicationStatsSnapshot {
        ReplicationStatsSnapshot {
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_saved: self.bytes_saved.load(Ordering::Relaxed),
            updates_sent: self.updates_sent.load(Ordering::Relaxed),
            batches_sent: self.batches_sent.load(Ordering::Relaxed),
            avg_compression_ratio: self.avg_compression_ratio_fp.load(Ordering::Relaxed) as f32
                / 1000.0,
            full_updates: self.full_updates.load(Ordering::Relaxed),
            queue_depth: self.queue_depth.load(Ordering::Relaxed),
            backpressure_events: self.backpressure_events.load(Ordering::Relaxed),
            auto_drains: self.auto_drains.load(Ordering::Relaxed),
            peak_queue_depth: self.peak_queue_depth.load(Ordering::Relaxed),
        }
    }
}

/// Handle for controlling the background drain task.
pub struct DrainHandle {
    /// Shutdown signal sender.
    shutdown_tx: mpsc::Sender<()>,
    /// Running state flag.
    running: Arc<AtomicBool>,
}

impl DrainHandle {
    /// Signal the background worker to shut down.
    pub async fn shutdown(&self) {
        let _ = self.shutdown_tx.send(()).await;
    }

    /// Check if the worker is still running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

/// Configuration for delta replication.
#[derive(Debug, Clone)]
pub struct DeltaReplicationConfig {
    /// Delta encoding threshold.
    pub delta_threshold: f32,
    /// Maximum updates per batch.
    pub max_batch_size: usize,
    /// Maximum pending updates (queue capacity).
    pub max_pending: usize,
    /// Minimum similarity for delta encoding (else full update).
    pub min_archetype_similarity: f32,
    /// Auto-drain interval in milliseconds (0 = disabled).
    pub auto_drain_interval_ms: u64,
}

impl Default for DeltaReplicationConfig {
    fn default() -> Self {
        Self {
            delta_threshold: 0.01,
            max_batch_size: 100,
            max_pending: 1000,
            min_archetype_similarity: 0.5,
            auto_drain_interval_ms: 100, // 100ms default
        }
    }
}

/// Manager for delta-compressed replication.
pub struct DeltaReplicationManager {
    /// Configuration.
    config: DeltaReplicationConfig,
    /// Archetype registry for delta encoding.
    registry: Arc<RwLock<ArchetypeRegistry>>,
    /// Pending updates sender (bounded channel for backpressure).
    pending_tx: mpsc::Sender<DeltaUpdate>,
    /// Pending updates receiver.
    pending_rx: Mutex<mpsc::Receiver<DeltaUpdate>>,
    /// Local node ID.
    local_node: NodeId,
    /// Batch sequence counter.
    sequence: AtomicU64,
    /// Replication statistics (thread-safe).
    stats: Arc<ReplicationStats>,
}

impl std::fmt::Debug for DeltaReplicationManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeltaReplicationManager")
            .field("config", &self.config)
            .field("local_node", &self.local_node)
            .field("pending_count", &self.stats.queue_depth())
            .finish()
    }
}

impl DeltaReplicationManager {
    /// Create a new delta replication manager.
    pub fn new(local_node: NodeId, config: DeltaReplicationConfig) -> Self {
        let (pending_tx, pending_rx) = mpsc::channel(config.max_pending);
        Self {
            config,
            registry: Arc::new(RwLock::new(ArchetypeRegistry::new(256))),
            pending_tx,
            pending_rx: Mutex::new(pending_rx),
            local_node,
            sequence: AtomicU64::new(0),
            stats: Arc::new(ReplicationStats::new()),
        }
    }

    /// Create with a shared archetype registry.
    pub fn with_registry(
        local_node: NodeId,
        config: DeltaReplicationConfig,
        registry: Arc<RwLock<ArchetypeRegistry>>,
    ) -> Self {
        let (pending_tx, pending_rx) = mpsc::channel(config.max_pending);
        Self {
            config,
            registry,
            pending_tx,
            pending_rx: Mutex::new(pending_rx),
            local_node,
            sequence: AtomicU64::new(0),
            stats: Arc::new(ReplicationStats::new()),
        }
    }

    /// Encode an embedding update (internal helper).
    fn encode_update(&self, key: String, embedding: &[f32], version: u64) -> DeltaUpdate {
        let registry = self.registry.read();

        match registry.find_best_archetype(embedding) {
            Some((_, similarity)) if similarity >= self.config.min_archetype_similarity => {
                DeltaUpdate::from_embedding(
                    key.clone(),
                    embedding,
                    &registry,
                    self.config.delta_threshold,
                    version,
                )
                .unwrap_or_else(|| DeltaUpdate::full(key, embedding, version))
            },
            _ => DeltaUpdate::full(key, embedding, version),
        }
    }

    /// Queue an embedding update for replication.
    ///
    /// Returns `Err(ChainError::QueueFull)` if the queue is at capacity.
    /// Use `queue_update_async()` to wait for space instead.
    pub fn queue_update(&self, key: String, embedding: &[f32], version: u64) -> Result<()> {
        let update = self.encode_update(key, embedding, version);

        match self.pending_tx.try_send(update) {
            Ok(()) => {
                self.stats.increment_queue_depth();
                Ok(())
            },
            Err(mpsc::error::TrySendError::Full(_)) => {
                self.stats.record_backpressure();
                Err(ChainError::QueueFull {
                    pending_count: self.stats.queue_depth(),
                })
            },
            Err(mpsc::error::TrySendError::Closed(_)) => Err(ChainError::NetworkError(
                "replication channel closed".into(),
            )),
        }
    }

    /// Queue an embedding update, waiting for space if queue is full.
    pub async fn queue_update_async(
        &self,
        key: String,
        embedding: &[f32],
        version: u64,
    ) -> Result<()> {
        let update = self.encode_update(key, embedding, version);

        self.pending_tx
            .send(update)
            .await
            .map_err(|_| ChainError::NetworkError("replication channel closed".into()))?;

        self.stats.increment_queue_depth();
        Ok(())
    }

    /// Start a background drain worker that periodically sends batches.
    ///
    /// Returns a handle to control the worker. Call `handle.shutdown().await`
    /// to stop the worker gracefully.
    pub fn start_auto_drain<T: Transport + Send + Sync + 'static>(
        self: &Arc<Self>,
        transport: Arc<T>,
        peer: NodeId,
    ) -> DrainHandle {
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        let manager = Arc::clone(self);
        let interval_ms = self.config.auto_drain_interval_ms;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(Duration::from_millis(interval_ms));

            loop {
                tokio::select! {
                    _ = ticker.tick() => {
                        // Drain and send batches
                        let batches = manager.flush();
                        for batch in batches {
                            if let Err(e) = manager.send_batch(transport.as_ref(), &peer, batch).await {
                                // Log error but continue
                                tracing::warn!("Auto-drain send failed: {}", e);
                            }
                        }
                        manager.stats.record_auto_drain();
                    }
                    _ = shutdown_rx.recv() => {
                        // Final drain before shutdown
                        let batches = manager.flush();
                        for batch in batches {
                            let _ = manager.send_batch(transport.as_ref(), &peer, batch).await;
                        }
                        running_clone.store(false, Ordering::Relaxed);
                        break;
                    }
                }
            }
        });

        DrainHandle {
            shutdown_tx,
            running,
        }
    }

    /// Send a single batch to a peer (internal helper).
    async fn send_batch<T: Transport>(
        &self,
        transport: &T,
        peer: &NodeId,
        batch: DeltaBatch,
    ) -> Result<()> {
        let full_bytes: usize = batch
            .updates
            .iter()
            .map(|u| u.dimension * std::mem::size_of::<f32>())
            .sum();

        self.stats.record_batch(&batch, full_bytes);

        let batch_bytes = bincode::serialize(&batch)
            .map_err(|e| ChainError::NetworkError(format!("Failed to serialize batch: {e}")))?;

        let msg = Message::SnapshotResponse(crate::network::SnapshotResponse {
            snapshot_height: batch.sequence,
            snapshot_hash: [0u8; 32],
            data: batch_bytes,
            offset: batch.sequence * self.config.max_batch_size as u64,
            total_size: if batch.is_final {
                (batch.sequence + 1) * self.config.max_batch_size as u64
            } else {
                u64::MAX
            },
            is_last: batch.is_final,
        });

        transport.send(peer, msg).await
    }

    /// Create a batch from pending updates.
    pub fn create_batch(&self, is_final: bool) -> Option<DeltaBatch> {
        let mut rx = self.pending_rx.lock();

        // Collect updates from channel
        let mut updates = Vec::new();
        while updates.len() < self.config.max_batch_size {
            match rx.try_recv() {
                Ok(update) => {
                    updates.push(update);
                    self.stats.decrement_queue_depth();
                },
                Err(_) => break,
            }
        }

        if updates.is_empty() {
            return None;
        }

        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        let mut batch = DeltaBatch::new(self.local_node.clone(), seq);

        for update in updates {
            batch.add(update);
        }

        // Check if queue is now empty
        let is_empty = rx.is_empty();
        if is_final || is_empty {
            batch = batch.finalize();
        }

        Some(batch)
    }

    /// Flush all pending updates into batches.
    pub fn flush(&self) -> Vec<DeltaBatch> {
        let mut batches = Vec::new();

        loop {
            match self.create_batch(false) {
                Some(mut batch) => {
                    // Check if this is the last batch
                    if self.pending_rx.lock().is_empty() {
                        batch.is_final = true;
                    }
                    batches.push(batch);

                    if self.pending_rx.lock().is_empty() {
                        break;
                    }
                },
                None => break,
            }
        }

        batches
    }

    /// Send batches to a peer.
    pub async fn send_to_peer<T: Transport>(&self, transport: &T, peer: &NodeId) -> Result<usize> {
        let batches = self.flush();
        let mut total_sent = 0;

        for batch in batches {
            let count = batch.len();
            self.send_batch(transport, peer, batch).await?;
            total_sent += count;
        }

        Ok(total_sent)
    }

    /// Apply a received batch to local state.
    pub fn apply_batch<F>(&self, batch: &DeltaBatch, mut apply_fn: F) -> Result<usize>
    where
        F: FnMut(&str, Vec<f32>) -> Result<()>,
    {
        let registry = self.registry.read();
        let mut applied = 0;

        for update in &batch.updates {
            match update.decode(&registry) {
                Some(embedding) => {
                    apply_fn(&update.key, embedding)?;
                    applied += 1;
                },
                None => {
                    return Err(ChainError::ValidationFailed(format!(
                        "Missing archetype {} for key {}",
                        update.archetype_id, update.key
                    )));
                },
            }
        }

        Ok(applied)
    }

    /// Get replication statistics snapshot.
    pub fn stats(&self) -> ReplicationStatsSnapshot {
        self.stats.snapshot()
    }

    /// Get number of pending updates.
    pub fn pending_count(&self) -> usize {
        self.stats.queue_depth()
    }

    /// Get the archetype registry.
    pub fn registry(&self) -> Arc<RwLock<ArchetypeRegistry>> {
        self.registry.clone()
    }

    /// Initialize archetypes from sample embeddings.
    pub fn initialize_archetypes(&self, samples: &[Vec<f32>], k: usize) -> usize {
        let mut registry = self.registry.write();
        registry.discover_archetypes(samples, k, tensor_store::KMeansConfig::default())
    }

    /// Sync archetypes with another node.
    pub fn get_archetype_sync(&self) -> Vec<Vec<f32>> {
        let registry = self.registry.read();
        (0..registry.len())
            .filter_map(|i| registry.get(i).map(|a| a.to_vec()))
            .collect()
    }

    /// Apply synced archetypes from another node.
    pub fn apply_archetype_sync(&self, archetypes: Vec<Vec<f32>>) -> usize {
        let mut registry = self.registry.write();
        let mut added = 0;
        for archetype in archetypes {
            if registry.register(archetype).is_some() {
                added += 1;
            }
        }
        added
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_delta_update_from_embedding() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let embedding = vec![0.9, 0.1, 0.0, 0.0];
        let update =
            DeltaUpdate::from_embedding("key1".to_string(), &embedding, &registry, 0.001, 1);

        assert!(update.is_some());
        let update = update.unwrap();
        assert_eq!(update.key, "key1");
        assert_eq!(update.archetype_id, 0);
        assert!(!update.is_full_update());
    }

    #[test]
    fn test_delta_update_full() {
        let embedding = vec![1.0, 2.0, 3.0, 4.0];
        let update = DeltaUpdate::full("key1".to_string(), &embedding, 1);

        assert!(update.is_full_update());
        assert_eq!(update.dimension, 4);
        assert_eq!(update.nnz(), 4);
    }

    #[test]
    fn test_delta_update_decode() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let embedding = vec![0.9, 0.1, 0.0, 0.0];
        let update =
            DeltaUpdate::from_embedding("key1".to_string(), &embedding, &registry, 0.001, 1)
                .unwrap();

        let decoded = update.decode(&registry).unwrap();
        for (orig, dec) in embedding.iter().zip(decoded.iter()) {
            assert!(approx_eq(*orig, *dec, 0.01));
        }
    }

    #[test]
    fn test_delta_update_decode_full() {
        let registry = ArchetypeRegistry::new(10);
        let embedding = vec![1.0, 2.0, 3.0, 4.0];
        let update = DeltaUpdate::full("key1".to_string(), &embedding, 1);

        let decoded = update.decode(&registry).unwrap();
        assert_eq!(decoded, embedding);
    }

    #[test]
    fn test_delta_update_compression_ratio() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0; 128]).unwrap();

        // Embedding very similar to archetype - high compression
        let mut embedding = vec![1.0; 128];
        embedding[0] = 0.9;

        let update =
            DeltaUpdate::from_embedding("key1".to_string(), &embedding, &registry, 0.01, 1)
                .unwrap();

        assert!(update.compression_ratio() > 1.0);
    }

    #[test]
    fn test_delta_batch() {
        let mut batch = DeltaBatch::new("node1".to_string(), 0);

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);

        batch.add(DeltaUpdate::full("key1".to_string(), &[1.0, 2.0], 1));
        batch.add(DeltaUpdate::full("key2".to_string(), &[3.0, 4.0], 2));

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
        assert!(!batch.is_final);

        let batch = batch.finalize();
        assert!(batch.is_final);
    }

    #[test]
    fn test_delta_batch_compression() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let mut batch = DeltaBatch::new("node1".to_string(), 0);

        for i in 0..5 {
            let embedding = vec![0.9 + 0.02 * i as f32, 0.1, 0.0, 0.0];
            let update = DeltaUpdate::from_embedding(
                format!("key{}", i),
                &embedding,
                &registry,
                0.001,
                i as u64,
            )
            .unwrap();
            batch.add(update);
        }

        assert!(batch.avg_compression_ratio() > 0.0);
    }

    #[test]
    fn test_replication_stats() {
        let stats = ReplicationStats::default();

        let mut batch = DeltaBatch::new("node1".to_string(), 0);
        batch.add(DeltaUpdate::full(
            "key1".to_string(),
            &[1.0, 2.0, 3.0, 4.0],
            1,
        ));

        let full_bytes = 4 * std::mem::size_of::<f32>();
        stats.record_batch(&batch, full_bytes);

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.updates_sent, 1);
        assert_eq!(snapshot.batches_sent, 1);
        assert_eq!(snapshot.full_updates, 1);
    }

    #[test]
    fn test_config_default() {
        let config = DeltaReplicationConfig::default();
        assert!(approx_eq(config.delta_threshold, 0.01, 0.001));
        assert_eq!(config.max_batch_size, 100);
        assert_eq!(config.max_pending, 1000);
    }

    #[test]
    fn test_manager_new() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        assert_eq!(manager.pending_count(), 0);
        assert_eq!(manager.stats().updates_sent, 0);
    }

    #[test]
    fn test_manager_queue_update() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        manager
            .queue_update("key1".to_string(), &[1.0, 2.0, 3.0, 4.0], 1)
            .unwrap();
        assert_eq!(manager.pending_count(), 1);

        manager
            .queue_update("key2".to_string(), &[5.0, 6.0, 7.0, 8.0], 2)
            .unwrap();
        assert_eq!(manager.pending_count(), 2);
    }

    #[test]
    fn test_manager_create_batch() {
        let config = DeltaReplicationConfig {
            max_batch_size: 2,
            ..Default::default()
        };
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        manager
            .queue_update("key1".to_string(), &[1.0, 2.0], 1)
            .unwrap();
        manager
            .queue_update("key2".to_string(), &[3.0, 4.0], 2)
            .unwrap();
        manager
            .queue_update("key3".to_string(), &[5.0, 6.0], 3)
            .unwrap();

        let batch = manager.create_batch(false).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(manager.pending_count(), 1);
    }

    #[test]
    fn test_manager_flush() {
        let config = DeltaReplicationConfig {
            max_batch_size: 2,
            ..Default::default()
        };
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        for i in 0..5 {
            manager
                .queue_update(format!("key{}", i), &[i as f32], i as u64)
                .unwrap();
        }

        let batches = manager.flush();
        assert_eq!(batches.len(), 3); // 2 + 2 + 1
        assert_eq!(manager.pending_count(), 0);
        assert!(batches.last().unwrap().is_final);
    }

    #[test]
    fn test_manager_apply_batch() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        let mut batch = DeltaBatch::new("node2".to_string(), 0);
        batch.add(DeltaUpdate::full("key1".to_string(), &[1.0, 2.0], 1));
        batch.add(DeltaUpdate::full("key2".to_string(), &[3.0, 4.0], 2));

        let mut applied_keys = Vec::new();
        let result = manager.apply_batch(&batch, |key, _embedding| {
            applied_keys.push(key.to_string());
            Ok(())
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);
        assert_eq!(applied_keys, vec!["key1", "key2"]);
    }

    #[test]
    fn test_manager_with_archetypes() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        // Initialize archetypes
        let samples = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.1, 0.9, 0.0, 0.0],
        ];
        let added = manager.initialize_archetypes(&samples, 2);
        assert!(added > 0);

        // Queue update - should use delta encoding
        manager
            .queue_update("key1".to_string(), &[0.95, 0.05, 0.0, 0.0], 1)
            .unwrap();
        assert_eq!(manager.pending_count(), 1);
    }

    #[test]
    fn test_manager_archetype_sync() {
        let config = DeltaReplicationConfig::default();
        let manager1 = DeltaReplicationManager::new("node1".to_string(), config.clone());
        let manager2 = DeltaReplicationManager::new("node2".to_string(), config);

        // Initialize archetypes on node1
        let samples = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
        manager1.initialize_archetypes(&samples, 2);

        // Sync to node2
        let archetypes = manager1.get_archetype_sync();
        let added = manager2.apply_archetype_sync(archetypes);

        assert!(added > 0);
    }

    #[test]
    fn test_manager_shared_registry() {
        let registry = Arc::new(RwLock::new(ArchetypeRegistry::new(10)));
        registry.write().register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let config = DeltaReplicationConfig::default();
        let manager =
            DeltaReplicationManager::with_registry("node1".to_string(), config, registry.clone());

        // Should use shared registry for encoding
        manager
            .queue_update("key1".to_string(), &[0.9, 0.1, 0.0, 0.0], 1)
            .unwrap();

        let batch = manager.create_batch(true).unwrap();
        assert!(!batch.updates[0].is_full_update());
    }

    #[test]
    fn test_effective_compression() {
        let stats = ReplicationStats::default();
        // Simulate recording batches to set bytes_sent and bytes_saved
        let mut batch = DeltaBatch::new("node1".to_string(), 0);
        batch.add(DeltaUpdate::full("key1".to_string(), &[1.0, 2.0], 1));
        stats.record_batch(&batch, 400); // full_bytes = 400, batch is smaller

        // Effective compression depends on actual batch size
        assert!(stats.effective_compression() >= 1.0);
    }

    #[test]
    fn test_effective_compression_zero_sent() {
        let stats = ReplicationStats::default();
        assert!(approx_eq(stats.effective_compression(), 1.0, 0.01));
    }

    #[test]
    fn test_delta_update_memory_bytes() {
        let update = DeltaUpdate::full("key1".to_string(), &[1.0, 2.0, 3.0, 4.0], 1);
        assert!(update.memory_bytes() > 0);
    }

    #[test]
    fn test_batch_memory_bytes() {
        let mut batch = DeltaBatch::new("node1".to_string(), 0);
        batch.add(DeltaUpdate::full("key1".to_string(), &[1.0, 2.0], 1));
        assert!(batch.memory_bytes() > 0);
    }

    #[test]
    fn test_manager_queue_full_returns_error() {
        let config = DeltaReplicationConfig {
            max_pending: 5,
            ..Default::default()
        };
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        // Queue up to capacity
        for i in 0..5 {
            manager
                .queue_update(format!("key{}", i), &[i as f32], i as u64)
                .unwrap();
        }

        // Next queue should fail with QueueFull
        let result = manager.queue_update("key_overflow".to_string(), &[99.0], 99);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ChainError::QueueFull { .. }));

        // Backpressure event should be recorded
        assert_eq!(manager.stats().backpressure_events, 1);
    }

    #[test]
    fn test_batch_sequence_increments() {
        let config = DeltaReplicationConfig {
            max_batch_size: 1,
            ..Default::default()
        };
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        manager.queue_update("key1".to_string(), &[1.0], 1).unwrap();
        manager.queue_update("key2".to_string(), &[2.0], 2).unwrap();

        let batch1 = manager.create_batch(false).unwrap();
        let batch2 = manager.create_batch(true).unwrap();

        assert_eq!(batch1.sequence, 0);
        assert_eq!(batch2.sequence, 1);
    }

    #[test]
    fn test_apply_batch_missing_archetype() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        // Create update with non-existent archetype
        let update = DeltaUpdate {
            key: "key1".to_string(),
            archetype_id: 99,
            delta_indices: vec![0],
            delta_values: vec![0.1],
            version: 1,
            dimension: 4,
        };

        let mut batch = DeltaBatch::new("node2".to_string(), 0);
        batch.add(update);

        let result = manager.apply_batch(&batch, |_, _| Ok(()));
        assert!(result.is_err());
    }

    #[test]
    fn test_compression_ratio_zero_dimension() {
        let update = DeltaUpdate {
            key: "key1".to_string(),
            archetype_id: u32::MAX,
            delta_indices: vec![],
            delta_values: vec![],
            version: 1,
            dimension: 0, // Zero dimension
        };
        // Should return 1.0 when full_bytes == 0
        assert!(approx_eq(update.compression_ratio(), 1.0, 0.01));
    }

    #[test]
    fn test_batch_avg_compression_ratio_empty() {
        let batch = DeltaBatch::new("node1".to_string(), 0);
        // Empty batch should return 1.0
        assert!(approx_eq(batch.avg_compression_ratio(), 1.0, 0.01));
    }

    #[test]
    fn test_replication_stats_multiple_batches() {
        let stats = ReplicationStats::default();

        // Record multiple batches to test running average
        let mut batch1 = DeltaBatch::new("node1".to_string(), 0);
        batch1.add(DeltaUpdate::full(
            "key1".to_string(),
            &[1.0, 2.0, 3.0, 4.0],
            1,
        ));
        stats.record_batch(&batch1, 64);

        let mut batch2 = DeltaBatch::new("node1".to_string(), 1);
        batch2.add(DeltaUpdate::full(
            "key2".to_string(),
            &[5.0, 6.0, 7.0, 8.0],
            2,
        ));
        stats.record_batch(&batch2, 64);

        // Should have running average now
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.batches_sent, 2);
        assert!(snapshot.avg_compression_ratio > 0.0);
    }

    #[test]
    fn test_decode_index_out_of_bounds() {
        let registry = ArchetypeRegistry::new(10);

        // Create update with out-of-bounds index
        let update = DeltaUpdate {
            key: "key1".to_string(),
            archetype_id: u32::MAX,         // Full update
            delta_indices: vec![0, 1, 100], // Index 100 is out of bounds for dim 4
            delta_values: vec![1.0, 2.0, 3.0],
            version: 1,
            dimension: 4,
        };

        // Should handle gracefully without panic
        let decoded = update.decode(&registry);
        assert!(decoded.is_some());
        let result = decoded.unwrap();
        assert_eq!(result.len(), 4);
        // Only valid indices should be set
        assert!(approx_eq(result[0], 1.0, 0.001));
        assert!(approx_eq(result[1], 2.0, 0.001));
    }

    #[test]
    fn test_from_embedding_no_archetype() {
        // Empty registry - encode should return None
        let registry = ArchetypeRegistry::new(10);
        let embedding = vec![1.0, 2.0, 3.0, 4.0];

        let update =
            DeltaUpdate::from_embedding("key1".to_string(), &embedding, &registry, 0.01, 1);

        // Should return None since no archetypes exist
        assert!(update.is_none());
    }

    #[test]
    fn test_decode_delta_with_valid_archetype() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        // Create a delta update manually
        let update = DeltaUpdate {
            key: "key1".to_string(),
            archetype_id: 0,
            delta_indices: vec![0, 1],
            delta_values: vec![0.1, -0.2],
            version: 1,
            dimension: 4,
        };

        let decoded = update.decode(&registry);
        assert!(decoded.is_some());
        let result = decoded.unwrap();
        assert!(approx_eq(result[0], 1.1, 0.001)); // 1.0 + 0.1
        assert!(approx_eq(result[1], 1.8, 0.001)); // 2.0 - 0.2
        assert!(approx_eq(result[2], 3.0, 0.001));
        assert!(approx_eq(result[3], 4.0, 0.001));
    }

    #[test]
    fn test_manager_registry_accessor() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        let registry = manager.registry();
        assert!(registry.read().len() == 0);
    }

    #[test]
    fn test_create_batch_empty() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        // Create batch when empty should return None
        let batch = manager.create_batch(false);
        assert!(batch.is_none());
    }

    #[test]
    fn test_delta_update_debug_clone() {
        let update = DeltaUpdate::full("key1".to_string(), &[1.0, 2.0], 1);
        let cloned = update.clone();
        assert_eq!(update.key, cloned.key);
        assert_eq!(update.version, cloned.version);

        let debug = format!("{:?}", update);
        assert!(debug.contains("DeltaUpdate"));
    }

    #[test]
    fn test_delta_batch_debug_clone() {
        let mut batch = DeltaBatch::new("node1".to_string(), 42);
        batch.add(DeltaUpdate::full("key1".to_string(), &[1.0], 1));

        let cloned = batch.clone();
        assert_eq!(batch.sequence, cloned.sequence);
        assert_eq!(batch.len(), cloned.len());

        let debug = format!("{:?}", batch);
        assert!(debug.contains("DeltaBatch"));
    }

    #[test]
    fn test_replication_stats_snapshot() {
        let stats = ReplicationStats::default();
        // Record a batch to update stats
        let mut batch = DeltaBatch::new("node1".to_string(), 0);
        batch.add(DeltaUpdate::full("key1".to_string(), &[1.0, 2.0], 1));
        stats.record_batch(&batch, 100);

        let snapshot = stats.snapshot();
        assert!(snapshot.bytes_sent > 0);

        let debug = format!("{:?}", stats);
        assert!(debug.contains("ReplicationStats"));
    }

    #[test]
    fn test_delta_replication_config_debug_clone() {
        let config = DeltaReplicationConfig::default();
        let cloned = config.clone();
        assert_eq!(config.max_batch_size, cloned.max_batch_size);

        let debug = format!("{:?}", config);
        assert!(debug.contains("DeltaReplicationConfig"));
    }

    #[test]
    fn test_manager_debug() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        let debug = format!("{:?}", manager);
        assert!(debug.contains("DeltaReplicationManager"));
    }

    #[test]
    fn test_queue_update_low_similarity_fallback() {
        let config = DeltaReplicationConfig {
            min_archetype_similarity: 0.99, // Very high threshold
            ..Default::default()
        };
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        // Initialize with an archetype
        {
            let registry = manager.registry();
            let mut guard = registry.write();
            guard.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        }

        // Queue an embedding that's not similar enough - should fallback to full
        manager
            .queue_update("key1".to_string(), &[0.5, 0.5, 0.0, 0.0], 1)
            .unwrap();

        let batch = manager.create_batch(true).unwrap();
        // When similarity is too low, it should do a full update
        assert!(batch.updates[0].is_full_update());
    }

    #[test]
    fn test_flush_empty() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        let batches = manager.flush();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_decode_delta_index_out_of_bounds() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 2.0]).unwrap();

        // Create a delta update with out-of-bounds index
        let update = DeltaUpdate {
            key: "key1".to_string(),
            archetype_id: 0,
            delta_indices: vec![0, 10], // Index 10 out of bounds
            delta_values: vec![0.1, 0.2],
            version: 1,
            dimension: 2,
        };

        // Should decode without panic, ignoring out-of-bounds
        let decoded = update.decode(&registry);
        assert!(decoded.is_some());
        let result = decoded.unwrap();
        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0], 1.1, 0.001)); // 1.0 + 0.1
        assert!(approx_eq(result[1], 2.0, 0.001)); // Unchanged
    }

    #[test]
    fn test_create_batch_with_is_final_true() {
        let config = DeltaReplicationConfig::default();
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        manager.queue_update("key1".to_string(), &[1.0], 1).unwrap();

        let batch = manager.create_batch(true).unwrap();
        assert!(batch.is_final);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let update = DeltaUpdate::full("key1".to_string(), &[1.0, 2.0, 3.0], 42);
        let bytes = bincode::serialize(&update).unwrap();
        let decoded: DeltaUpdate = bincode::deserialize(&bytes).unwrap();

        assert_eq!(update.key, decoded.key);
        assert_eq!(update.version, decoded.version);
        assert_eq!(update.delta_values, decoded.delta_values);
    }

    #[test]
    fn test_batch_serialization_roundtrip() {
        let mut batch = DeltaBatch::new("node1".to_string(), 99);
        batch.add(DeltaUpdate::full("key1".to_string(), &[1.0], 1));
        batch = batch.finalize();

        let bytes = bincode::serialize(&batch).unwrap();
        let decoded: DeltaBatch = bincode::deserialize(&bytes).unwrap();

        assert_eq!(batch.source, decoded.source);
        assert_eq!(batch.sequence, decoded.sequence);
        assert_eq!(batch.is_final, decoded.is_final);
        assert_eq!(batch.len(), decoded.len());
    }
}
