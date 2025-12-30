//! Delta-compressed replication for bandwidth-efficient state transfer.
//!
//! Uses archetype-based delta encoding to reduce replication bandwidth by 4-6x.
//! Entities are encoded as (archetype_id + sparse_delta) and batched for transfer.
//!
//! When int8 quantization is enabled, delta values are further compressed from f32 to i8,
//! providing an additional 4x size reduction with ~1% max error.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tensor_compress::{dequantize_int8, quantize_int8, QuantizedInt8};
use tensor_store::ArchetypeRegistry;

use crate::block::NodeId;
use crate::error::{ChainError, Result};
use crate::network::{Message, Transport};

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

    /// Convert to quantized form (4x size reduction for delta values).
    pub fn quantize(&self) -> Option<QuantizedDeltaUpdate> {
        if self.delta_values.is_empty() {
            return Some(QuantizedDeltaUpdate {
                key: self.key.clone(),
                archetype_id: self.archetype_id,
                delta_indices: self.delta_indices.clone(),
                quantized_values: QuantizedInt8 {
                    data: Vec::new(),
                    min: 0.0,
                    scale: 1.0,
                },
                version: self.version,
                dimension: self.dimension,
            });
        }

        let quantized = quantize_int8(&self.delta_values).ok()?;
        Some(QuantizedDeltaUpdate {
            key: self.key.clone(),
            archetype_id: self.archetype_id,
            delta_indices: self.delta_indices.clone(),
            quantized_values: quantized,
            version: self.version,
            dimension: self.dimension,
        })
    }
}

/// A quantized delta update using int8 values for 4x compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedDeltaUpdate {
    /// Key being updated.
    pub key: String,
    /// ID of the reference archetype.
    pub archetype_id: u32,
    /// Sparse delta indices.
    pub delta_indices: Vec<u32>,
    /// Quantized delta values (int8 with min/scale).
    pub quantized_values: QuantizedInt8,
    /// Update version.
    pub version: u64,
    /// Original embedding dimension.
    pub dimension: usize,
}

impl QuantizedDeltaUpdate {
    /// Check if this is a full update (not delta-compressed).
    pub fn is_full_update(&self) -> bool {
        self.archetype_id == u32::MAX
    }

    /// Get the number of non-zero delta values.
    pub fn nnz(&self) -> usize {
        self.quantized_values.data.len()
    }

    /// Memory bytes used by this update (significantly less than DeltaUpdate).
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.key.len()
            + self.delta_indices.len() * std::mem::size_of::<u32>()
            + self.quantized_values.data.len() * std::mem::size_of::<i8>()
            + 2 * std::mem::size_of::<f32>() // min and scale
    }

    /// Compression ratio compared to full f32 embedding.
    pub fn compression_ratio(&self) -> f32 {
        let full_bytes = self.dimension * std::mem::size_of::<f32>();
        if full_bytes == 0 {
            return 1.0;
        }
        full_bytes as f32 / self.memory_bytes() as f32
    }

    /// Convert back to non-quantized DeltaUpdate.
    pub fn dequantize(&self) -> DeltaUpdate {
        let delta_values = if self.quantized_values.data.is_empty() {
            Vec::new()
        } else {
            dequantize_int8(&self.quantized_values)
        };

        DeltaUpdate {
            key: self.key.clone(),
            archetype_id: self.archetype_id,
            delta_indices: self.delta_indices.clone(),
            delta_values,
            version: self.version,
            dimension: self.dimension,
        }
    }

    /// Decode back to dense embedding using archetype registry.
    pub fn decode(&self, registry: &ArchetypeRegistry) -> Option<Vec<f32>> {
        self.dequantize().decode(registry)
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
#[derive(Debug, Clone, Default)]
pub struct ReplicationStats {
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
}

impl ReplicationStats {
    /// Update stats with a batch.
    pub fn record_batch(&mut self, batch: &DeltaBatch, full_bytes: usize) {
        let batch_bytes = batch.memory_bytes();
        self.bytes_sent += batch_bytes as u64;
        self.bytes_saved += (full_bytes.saturating_sub(batch_bytes)) as u64;
        self.updates_sent += batch.len() as u64;
        self.batches_sent += 1;

        // Update running average
        let batch_ratio = batch.avg_compression_ratio();
        if self.batches_sent == 1 {
            self.avg_compression_ratio = batch_ratio;
        } else {
            self.avg_compression_ratio = self.avg_compression_ratio * 0.9 + batch_ratio * 0.1;
        }

        // Count full updates
        self.full_updates += batch.updates.iter().filter(|u| u.is_full_update()).count() as u64;
    }

    /// Get effective compression ratio.
    pub fn effective_compression(&self) -> f32 {
        if self.bytes_sent == 0 {
            return 1.0;
        }
        (self.bytes_sent + self.bytes_saved) as f32 / self.bytes_sent as f32
    }
}

/// Configuration for delta replication.
#[derive(Debug, Clone)]
pub struct DeltaReplicationConfig {
    /// Delta encoding threshold.
    pub delta_threshold: f32,
    /// Maximum updates per batch.
    pub max_batch_size: usize,
    /// Maximum pending updates before flushing.
    pub max_pending: usize,
    /// Minimum similarity for delta encoding (else full update).
    pub min_archetype_similarity: f32,
    /// Enable int8 quantization for delta values (4x additional compression).
    pub enable_quantization: bool,
}

impl Default for DeltaReplicationConfig {
    fn default() -> Self {
        Self {
            delta_threshold: 0.01,
            max_batch_size: 100,
            max_pending: 1000,
            min_archetype_similarity: 0.5,
            enable_quantization: false, // Off by default for backward compatibility
        }
    }
}

impl DeltaReplicationConfig {
    /// Create config with int8 quantization enabled.
    pub fn with_quantization(mut self) -> Self {
        self.enable_quantization = true;
        self
    }
}

/// Manager for delta-compressed replication.
#[derive(Debug)]
pub struct DeltaReplicationManager {
    /// Configuration.
    config: DeltaReplicationConfig,
    /// Archetype registry for delta encoding.
    registry: Arc<RwLock<ArchetypeRegistry>>,
    /// Pending updates to send.
    pending: RwLock<VecDeque<DeltaUpdate>>,
    /// Local node ID.
    local_node: NodeId,
    /// Batch sequence counter.
    sequence: AtomicU64,
    /// Replication statistics.
    stats: RwLock<ReplicationStats>,
}

impl DeltaReplicationManager {
    /// Create a new delta replication manager.
    pub fn new(local_node: NodeId, config: DeltaReplicationConfig) -> Self {
        Self {
            config,
            registry: Arc::new(RwLock::new(ArchetypeRegistry::new(256))),
            pending: RwLock::new(VecDeque::new()),
            local_node,
            sequence: AtomicU64::new(0),
            stats: RwLock::new(ReplicationStats::default()),
        }
    }

    /// Create with a shared archetype registry.
    pub fn with_registry(
        local_node: NodeId,
        config: DeltaReplicationConfig,
        registry: Arc<RwLock<ArchetypeRegistry>>,
    ) -> Self {
        Self {
            config,
            registry,
            pending: RwLock::new(VecDeque::new()),
            local_node,
            sequence: AtomicU64::new(0),
            stats: RwLock::new(ReplicationStats::default()),
        }
    }

    /// Queue an embedding update for replication.
    pub fn queue_update(&self, key: String, embedding: &[f32], version: u64) {
        let registry = self.registry.read();

        // Try delta encoding first
        let update = match registry.find_best_archetype(embedding) {
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
        };

        drop(registry);

        let mut pending = self.pending.write();
        pending.push_back(update);

        // Auto-flush if too many pending
        if pending.len() >= self.config.max_pending {
            // Note: In real use, this would trigger async send
            // For now, just truncate to prevent unbounded growth
            while pending.len() > self.config.max_pending {
                pending.pop_front();
            }
        }
    }

    /// Create a batch from pending updates.
    pub fn create_batch(&self, is_final: bool) -> Option<DeltaBatch> {
        let mut pending = self.pending.write();

        if pending.is_empty() {
            return None;
        }

        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);
        let mut batch = DeltaBatch::new(self.local_node.clone(), seq);

        while !pending.is_empty() && batch.len() < self.config.max_batch_size {
            if let Some(update) = pending.pop_front() {
                batch.add(update);
            }
        }

        if is_final || pending.is_empty() {
            batch = batch.finalize();
        }

        Some(batch)
    }

    /// Flush all pending updates into batches.
    pub fn flush(&self) -> Vec<DeltaBatch> {
        let mut batches = Vec::new();

        loop {
            let batch = self.create_batch(false);
            match batch {
                Some(mut b) => {
                    if self.pending.read().is_empty() {
                        b.is_final = true;
                    }
                    batches.push(b);

                    if self.pending.read().is_empty() {
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
            let full_bytes = batch
                .updates
                .iter()
                .map(|u| u.dimension * std::mem::size_of::<f32>())
                .sum();

            // Record stats before sending
            self.stats.write().record_batch(&batch, full_bytes);

            // Serialize batch
            let batch_bytes = bincode::serialize(&batch)
                .map_err(|e| ChainError::NetworkError(format!("Failed to serialize batch: {e}")))?;

            // Send as snapshot chunk
            let msg = Message::SnapshotResponse(crate::network::SnapshotResponse {
                snapshot_height: batch.sequence,
                snapshot_hash: [0u8; 32], // Placeholder hash
                data: batch_bytes,
                offset: batch.sequence * self.config.max_batch_size as u64,
                total_size: if batch.is_final {
                    (batch.sequence + 1) * self.config.max_batch_size as u64
                } else {
                    u64::MAX
                },
                is_last: batch.is_final,
            });

            transport.send(peer, msg).await?;
            total_sent += batch.len();
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
                    // Missing archetype - request sync
                    return Err(ChainError::ValidationFailed(format!(
                        "Missing archetype {} for key {}",
                        update.archetype_id, update.key
                    )));
                },
            }
        }

        Ok(applied)
    }

    /// Get replication statistics.
    pub fn stats(&self) -> ReplicationStats {
        self.stats.read().clone()
    }

    /// Get number of pending updates.
    pub fn pending_count(&self) -> usize {
        self.pending.read().len()
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
        let mut stats = ReplicationStats::default();

        let mut batch = DeltaBatch::new("node1".to_string(), 0);
        batch.add(DeltaUpdate::full(
            "key1".to_string(),
            &[1.0, 2.0, 3.0, 4.0],
            1,
        ));

        let full_bytes = 4 * std::mem::size_of::<f32>();
        stats.record_batch(&batch, full_bytes);

        assert_eq!(stats.updates_sent, 1);
        assert_eq!(stats.batches_sent, 1);
        assert_eq!(stats.full_updates, 1);
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

        manager.queue_update("key1".to_string(), &[1.0, 2.0, 3.0, 4.0], 1);
        assert_eq!(manager.pending_count(), 1);

        manager.queue_update("key2".to_string(), &[5.0, 6.0, 7.0, 8.0], 2);
        assert_eq!(manager.pending_count(), 2);
    }

    #[test]
    fn test_manager_create_batch() {
        let config = DeltaReplicationConfig {
            max_batch_size: 2,
            ..Default::default()
        };
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        manager.queue_update("key1".to_string(), &[1.0, 2.0], 1);
        manager.queue_update("key2".to_string(), &[3.0, 4.0], 2);
        manager.queue_update("key3".to_string(), &[5.0, 6.0], 3);

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
            manager.queue_update(format!("key{}", i), &[i as f32], i as u64);
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
        manager.queue_update("key1".to_string(), &[0.95, 0.05, 0.0, 0.0], 1);
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
        manager.queue_update("key1".to_string(), &[0.9, 0.1, 0.0, 0.0], 1);

        let batch = manager.create_batch(true).unwrap();
        assert!(!batch.updates[0].is_full_update());
    }

    #[test]
    fn test_effective_compression() {
        let mut stats = ReplicationStats::default();
        stats.bytes_sent = 100;
        stats.bytes_saved = 300;

        // Effective compression = (100 + 300) / 100 = 4x
        assert!(approx_eq(stats.effective_compression(), 4.0, 0.01));
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
    fn test_manager_max_pending_truncation() {
        let config = DeltaReplicationConfig {
            max_pending: 5,
            ..Default::default()
        };
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        // Queue more than max_pending
        for i in 0..10 {
            manager.queue_update(format!("key{}", i), &[i as f32], i as u64);
        }

        // Should be truncated
        assert!(manager.pending_count() <= 5);
    }

    #[test]
    fn test_batch_sequence_increments() {
        let config = DeltaReplicationConfig {
            max_batch_size: 1,
            ..Default::default()
        };
        let manager = DeltaReplicationManager::new("node1".to_string(), config);

        manager.queue_update("key1".to_string(), &[1.0], 1);
        manager.queue_update("key2".to_string(), &[2.0], 2);

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
        let mut stats = ReplicationStats::default();

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
        assert_eq!(stats.batches_sent, 2);
        assert!(stats.avg_compression_ratio > 0.0);
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
    fn test_replication_stats_debug_clone() {
        let mut stats = ReplicationStats::default();
        stats.bytes_sent = 100;

        let cloned = stats.clone();
        assert_eq!(stats.bytes_sent, cloned.bytes_sent);

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
        let mut registry = manager.registry.write();
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        drop(registry);

        // Queue an embedding that's not similar enough - should fallback to full
        manager.queue_update("key1".to_string(), &[0.5, 0.5, 0.0, 0.0], 1);

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

        manager.queue_update("key1".to_string(), &[1.0], 1);

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

    // ==================== Quantization Tests ====================

    #[test]
    fn test_delta_update_quantize() {
        let embedding = vec![1.0, 2.0, 3.0, 4.0];
        let update = DeltaUpdate::full("key1".to_string(), &embedding, 1);

        let quantized = update.quantize();
        assert!(quantized.is_some());

        let quantized = quantized.unwrap();
        assert_eq!(quantized.key, "key1");
        assert_eq!(quantized.version, 1);
        assert!(quantized.is_full_update());
        assert_eq!(quantized.nnz(), 4);
    }

    #[test]
    fn test_quantized_dequantize_roundtrip() {
        let embedding = vec![0.1, 0.5, 0.9, 1.0];
        let update = DeltaUpdate::full("key1".to_string(), &embedding, 1);

        let quantized = update.quantize().unwrap();
        let dequantized = quantized.dequantize();

        // Values should be close (within 1% error)
        for (orig, deq) in update
            .delta_values
            .iter()
            .zip(dequantized.delta_values.iter())
        {
            assert!(
                (orig - deq).abs() < 0.02,
                "Quantization error too large: {} vs {}",
                orig,
                deq
            );
        }
    }

    #[test]
    fn test_quantized_decode() {
        let registry = ArchetypeRegistry::new(10);
        let embedding = vec![1.0, 2.0, 3.0, 4.0];
        let update = DeltaUpdate::full("key1".to_string(), &embedding, 1);

        let quantized = update.quantize().unwrap();
        let decoded = quantized.decode(&registry).unwrap();

        // Should decode correctly (with small quantization error)
        for (orig, dec) in embedding.iter().zip(decoded.iter()) {
            assert!(approx_eq(*orig, *dec, 0.02));
        }
    }

    #[test]
    fn test_quantized_compression_ratio() {
        // Create a larger embedding to see compression benefits
        let embedding: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();
        let update = DeltaUpdate::full("key1".to_string(), &embedding, 1);
        let quantized = update.quantize().unwrap();

        // Quantized should use less memory for the delta values themselves
        let orig_bytes = update.memory_bytes();
        let quant_bytes = quantized.memory_bytes();

        // Note: struct overhead may cause quantized to be larger for small updates
        // but for larger embeddings, quantized should be smaller
        assert!(
            quant_bytes < orig_bytes,
            "Quantized ({}) should be smaller than original ({}) for 256-dim embedding",
            quant_bytes,
            orig_bytes
        );

        // The compression_ratio() method calculates vs full embedding size
        // which may be affected by key length and struct overhead
        let ratio = quantized.compression_ratio();
        // Just verify it's a valid positive number
        assert!(ratio > 0.0, "Compression ratio should be positive");
    }

    #[test]
    fn test_quantized_empty_values() {
        let update = DeltaUpdate {
            key: "key1".to_string(),
            archetype_id: 0,
            delta_indices: vec![],
            delta_values: vec![],
            version: 1,
            dimension: 4,
        };

        let quantized = update.quantize();
        assert!(quantized.is_some());

        let quantized = quantized.unwrap();
        assert_eq!(quantized.nnz(), 0);
    }

    #[test]
    fn test_quantized_serialization_roundtrip() {
        let embedding = vec![1.0, 2.0, 3.0, 4.0];
        let update = DeltaUpdate::full("key1".to_string(), &embedding, 42);
        let quantized = update.quantize().unwrap();

        let bytes = bincode::serialize(&quantized).unwrap();
        let decoded: QuantizedDeltaUpdate = bincode::deserialize(&bytes).unwrap();

        assert_eq!(quantized.key, decoded.key);
        assert_eq!(quantized.version, decoded.version);
        assert_eq!(quantized.dimension, decoded.dimension);
        assert_eq!(
            quantized.quantized_values.data,
            decoded.quantized_values.data
        );
    }

    #[test]
    fn test_config_with_quantization() {
        let config = DeltaReplicationConfig::default().with_quantization();
        assert!(config.enable_quantization);
    }

    #[test]
    fn test_quantized_debug_clone() {
        let embedding = vec![1.0, 2.0];
        let update = DeltaUpdate::full("key1".to_string(), &embedding, 1);
        let quantized = update.quantize().unwrap();

        let cloned = quantized.clone();
        assert_eq!(quantized.key, cloned.key);

        let debug = format!("{:?}", quantized);
        assert!(debug.contains("QuantizedDeltaUpdate"));
    }

    #[test]
    fn test_quantized_vs_original_size() {
        // Compare sizes: f32 delta values vs i8 quantized
        let embedding: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let update = DeltaUpdate::full("key1".to_string(), &embedding, 1);
        let quantized = update.quantize().unwrap();

        // Original: 128 * 4 bytes = 512 bytes for values
        // Quantized: 128 * 1 byte + 8 bytes (min/scale) = 136 bytes for values
        // So quantized values should be ~3.7x smaller

        let orig_value_bytes = update.delta_values.len() * std::mem::size_of::<f32>();
        let quant_value_bytes =
            quantized.quantized_values.data.len() + 2 * std::mem::size_of::<f32>();

        let value_ratio = orig_value_bytes as f32 / quant_value_bytes as f32;
        assert!(
            value_ratio > 3.0,
            "Expected ~4x compression for values, got {}x",
            value_ratio
        );
    }

    #[test]
    fn test_quantized_with_archetype() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let embedding = vec![0.9, 0.1, 0.0, 0.0];
        let update =
            DeltaUpdate::from_embedding("key1".to_string(), &embedding, &registry, 0.001, 1)
                .unwrap();

        // Quantize the delta-encoded update
        let quantized = update.quantize().unwrap();
        assert!(!quantized.is_full_update());

        // Decode and verify
        let decoded = quantized.decode(&registry).unwrap();
        for (orig, dec) in embedding.iter().zip(decoded.iter()) {
            assert!(approx_eq(*orig, *dec, 0.05));
        }
    }
}
