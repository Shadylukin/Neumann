// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dense embedding storage with chunked allocation.
//!
//! `EmbeddingSlab` stores dense f32 embeddings in contiguous chunks to avoid
//! large reallocations. Embeddings are stored by `EntityId`, with O(1) lookup
//! and append operations.
//!
//! # Design Philosophy
//!
//! - Append-only growth with chunked allocation
//! - Free slot reuse for deleted embeddings
//! - O(1) lookup by `EntityId`
//! - Zero-copy iteration for HNSW index building

use std::{
    cell::UnsafeCell,
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::entity_index::EntityId;

/// Default chunk size: 16MB of f32s = 4M floats
const DEFAULT_CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Error types for `EmbeddingSlab` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbeddingError {
    /// Embedding dimension mismatch.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension provided.
        actual: usize,
    },
    /// `EntityId` not found.
    NotFound(EntityId),
    /// Slab is full and cannot allocate more chunks.
    OutOfMemory,
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            },
            Self::NotFound(id) => write!(f, "entity {} not found", id.as_u64()),
            Self::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// Location of an embedding within the chunked storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingSlot {
    /// Chunk index.
    chunk: u32,
    /// Offset within the chunk (in number of embeddings, not bytes).
    offset: u32,
}

impl EmbeddingSlot {
    /// Create from chunk index and offset. Chunk count is practically limited.
    #[allow(clippy::cast_possible_truncation)]
    const fn from_position(chunk_idx: usize, offset: usize) -> Self {
        Self {
            chunk: chunk_idx as u32,
            offset: offset as u32,
        }
    }
}

/// Dense embedding storage with chunked allocation.
///
/// # Thread Safety
///
/// Uses `parking_lot` locks for concurrent access without lock poisoning.
/// Reads can proceed concurrently; writes have exclusive access.
///
/// # Performance
///
/// - `set`: O(1) amortized (may allocate new chunk)
/// - `get`: O(1)
/// - `delete`: O(1)
/// - `iter`: O(n)
pub struct EmbeddingSlab {
    /// Fixed embedding dimension.
    dimension: usize,

    /// Embeddings per chunk.
    chunk_capacity: usize,

    /// Chunked storage with interior mutability for embedding updates.
    ///
    /// Uses `UnsafeCell` to allow writing to individual slots while other
    /// threads read different slots. Safety is ensured by:
    /// - `RwLock` on the outer Vec prevents reallocation during access
    /// - Each slot is disjoint (indexed by `EntityId` -> chunk + offset)
    /// - The index `RwLock` prevents concurrent slot assignment
    chunks: RwLock<Vec<UnsafeCell<Box<[f32]>>>>,

    /// `EntityId` -> slot mapping.
    index: RwLock<BTreeMap<EntityId, EmbeddingSlot>>,

    /// Free slots for deleted embeddings.
    free_slots: Mutex<Vec<EmbeddingSlot>>,

    /// Write position in the current (last) chunk.
    write_pos: AtomicUsize,

    /// Total count of embeddings.
    count: AtomicUsize,
}

impl EmbeddingSlab {
    /// Create a new `EmbeddingSlab` with the given dimension and initial capacity.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Fixed embedding dimension (e.g., 768 for BERT)
    /// * `initial_capacity` - Initial number of embeddings to pre-allocate
    #[must_use]
    pub fn new(dimension: usize, initial_capacity: usize) -> Self {
        let chunk_capacity = DEFAULT_CHUNK_SIZE / dimension;
        let chunk_capacity = chunk_capacity.max(1);

        let num_chunks = (initial_capacity / chunk_capacity).max(1);
        let chunks: Vec<UnsafeCell<Box<[f32]>>> = (0..num_chunks)
            .map(|_| UnsafeCell::new(vec![0.0f32; chunk_capacity * dimension].into_boxed_slice()))
            .collect();

        Self {
            dimension,
            chunk_capacity,
            chunks: RwLock::new(chunks),
            index: RwLock::new(BTreeMap::new()),
            free_slots: Mutex::new(Vec::new()),
            write_pos: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
        }
    }

    /// Create an `EmbeddingSlab` with default capacity.
    #[must_use]
    pub fn with_dimension(dimension: usize) -> Self {
        Self::new(dimension, 1000)
    }

    /// Get the embedding dimension.
    #[inline]
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Validate that a slot is within bounds.
    /// Panics in debug builds if the slot is invalid.
    #[inline]
    fn assert_slot_valid(&self, slot: EmbeddingSlot, chunks_len: usize) {
        let chunk_idx = slot.chunk as usize;
        let offset = slot.offset as usize;
        debug_assert!(
            chunk_idx < chunks_len,
            "chunk index {chunk_idx} out of bounds (len {chunks_len})",
        );
        let chunk_capacity = self.chunk_capacity;
        debug_assert!(
            offset < chunk_capacity,
            "offset {offset} exceeds chunk capacity {chunk_capacity}",
        );
        // Validate slice bounds: start + dimension must fit in chunk
        let start = offset * self.dimension;
        let end = start + self.dimension;
        let expected_chunk_size = chunk_capacity * self.dimension;
        debug_assert!(
            end <= expected_chunk_size,
            "slice end {end} exceeds chunk size {expected_chunk_size}",
        );
    }

    /// Store an embedding for an entity.
    ///
    /// If the entity already has an embedding, it is replaced.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if the embedding has the wrong dimension.
    pub fn set(&self, entity: EntityId, embedding: &[f32]) -> Result<(), EmbeddingError> {
        if embedding.len() != self.dimension {
            return Err(EmbeddingError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        // Check if entity already has a slot
        if let Some(&slot) = self.index.read().get(&entity) {
            // Update in-place
            let chunks = self.chunks.read();
            self.assert_slot_valid(slot, chunks.len());
            let chunk = &chunks[slot.chunk as usize];
            let start = slot.offset as usize * self.dimension;
            // SAFETY: RwLock on chunks prevents Vec reallocation. RwLock on index
            // prevents concurrent slot assignment. Each slot is disjoint.
            // UnsafeCell provides interior mutability for embedding data.
            unsafe {
                let slice = &mut *chunk.get();
                slice[start..start + self.dimension].copy_from_slice(embedding);
            }
            drop(chunks);
            return Ok(());
        }

        // Get or allocate a slot
        let slot = self.allocate_slot();

        // Write the embedding
        let chunks = self.chunks.read();
        self.assert_slot_valid(slot, chunks.len());
        let chunk = &chunks[slot.chunk as usize];
        let start = slot.offset as usize * self.dimension;
        // SAFETY: RwLock on chunks prevents Vec reallocation. RwLock on index
        // prevents concurrent slot assignment. Each slot is disjoint.
        // UnsafeCell provides interior mutability for embedding data.
        unsafe {
            let slice = &mut *chunk.get();
            slice[start..start + self.dimension].copy_from_slice(embedding);
        }
        drop(chunks);

        // Update index
        self.index.write().insert(entity, slot);
        self.count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get an embedding by `EntityId` (cloned).
    #[must_use]
    pub fn get(&self, entity: EntityId) -> Option<Vec<f32>> {
        let slot = *self.index.read().get(&entity)?;

        let chunks = self.chunks.read();
        self.assert_slot_valid(slot, chunks.len());
        let start = slot.offset as usize * self.dimension;
        let end = start + self.dimension;
        // SAFETY: RwLock on chunks prevents Vec reallocation. We only read
        // the slot data, and writes to other slots don't affect this read.
        let embedding = unsafe { (&(*chunks[slot.chunk as usize].get()))[start..end].to_vec() };
        drop(chunks);

        Some(embedding)
    }

    /// Check if an entity has an embedding.
    #[inline]
    pub fn contains(&self, entity: EntityId) -> bool {
        self.index.read().contains_key(&entity)
    }

    /// Delete an embedding.
    ///
    /// The slot is added to the free list for reuse.
    pub fn delete(&self, entity: EntityId) -> bool {
        self.index.write().remove(&entity).is_some_and(|slot| {
            self.free_slots.lock().push(slot);
            self.count.fetch_sub(1, Ordering::Relaxed);
            true
        })
    }

    /// Get the number of embeddings.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if the slab is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the total capacity (across all chunks).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.chunks.read().len() * self.chunk_capacity
    }

    /// Get memory usage in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let chunks = self.chunks.read();
        chunks.len() * self.chunk_capacity * self.dimension * std::mem::size_of::<f32>()
    }

    /// Collect all (`EntityId`, embedding) pairs.
    ///
    /// Note: This clones each embedding. For zero-copy access, use `iter_with`.
    #[must_use]
    pub fn entries(&self) -> Vec<(EntityId, Vec<f32>)> {
        let index = self.index.read();
        let chunks = self.chunks.read();

        index
            .iter()
            .map(|(&entity, &slot)| {
                self.assert_slot_valid(slot, chunks.len());
                let start = slot.offset as usize * self.dimension;
                let end = start + self.dimension;
                // SAFETY: RwLock on chunks prevents Vec reallocation. We only read
                // the slot data under the read lock.
                let embedding =
                    unsafe { (&(*chunks[slot.chunk as usize].get()))[start..end].to_vec() };
                (entity, embedding)
            })
            .collect()
    }

    /// Iterate with a callback for zero-copy access.
    ///
    /// The callback receives each (`EntityId`, `&[f32]`) pair.
    pub fn iter_with<F>(&self, mut f: F)
    where
        F: FnMut(EntityId, &[f32]),
    {
        let index = self.index.read();
        let chunks = self.chunks.read();

        for (&entity, &slot) in index.iter() {
            self.assert_slot_valid(slot, chunks.len());
            let start = slot.offset as usize * self.dimension;
            let end = start + self.dimension;
            // SAFETY: RwLock on chunks prevents Vec reallocation. We only read
            // the slot data under the read lock. The reference is valid for
            // the duration of the callback while we hold the lock.
            let slice = unsafe { &(&(*chunks[slot.chunk as usize].get()))[start..end] };
            f(entity, slice);
        }
        drop(index);
        drop(chunks);
    }

    /// Get all `EntityId`s with embeddings.
    #[must_use]
    pub fn entity_ids(&self) -> Vec<EntityId> {
        self.index.read().keys().copied().collect()
    }

    /// Clear all embeddings.
    pub fn clear(&self) {
        self.index.write().clear();
        self.free_slots.lock().clear();
        self.write_pos.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }

    /// Compact the slab by removing gaps from deleted embeddings.
    ///
    /// This is an expensive O(n) operation that rewrites all embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if any embedding fails to be re-inserted.
    pub fn compact(&self) -> Result<(), EmbeddingError> {
        // Build new compact storage
        let old_entries: Vec<(EntityId, Vec<f32>)> = self.entries();

        if old_entries.is_empty() {
            return Ok(());
        }

        // Clear and rebuild
        self.clear();

        for (entity, embedding) in old_entries {
            self.set(entity, &embedding)?;
        }
        Ok(())
    }

    /// Get serializable state for snapshots.
    ///
    /// Automatically compresses sparse vectors for efficient storage.
    pub fn snapshot(&self) -> EmbeddingSlabSnapshot {
        let index = self.index.read();
        let chunks = self.chunks.read();

        // Collect all embeddings with automatic sparse compression
        let embeddings: Vec<(EntityId, CompressedEmbedding)> = index
            .iter()
            .map(|(&entity, &slot)| {
                self.assert_slot_valid(slot, chunks.len());
                let start = slot.offset as usize * self.dimension;
                let end = start + self.dimension;
                // SAFETY: RwLock on chunks prevents Vec reallocation. We only read
                // the slot data under the read lock.
                let dense = unsafe { &(&(*chunks[slot.chunk as usize].get()))[start..end] };
                (entity, CompressedEmbedding::from_dense(dense))
            })
            .collect();
        drop(index);
        drop(chunks);

        EmbeddingSlabSnapshot {
            dimension: self.dimension,
            embeddings,
        }
    }

    /// Restore from a snapshot.
    ///
    /// Decompresses sparse vectors back to dense format for HNSW compatibility.
    #[must_use]
    pub fn restore(snapshot: EmbeddingSlabSnapshot) -> Self {
        let slab = Self::new(snapshot.dimension, snapshot.embeddings.len().max(1));

        for (entity, compressed) in snapshot.embeddings {
            let dense = compressed.to_dense();
            if let Err(e) = slab.set(entity, &dense) {
                tracing::warn!(
                    entity = %entity.as_u64(),
                    error = %e,
                    "Failed to restore embedding"
                );
            }
        }

        slab
    }

    /// Allocate a slot for a new embedding.
    fn allocate_slot(&self) -> EmbeddingSlot {
        // Try to reuse a free slot
        let free_slot = self.free_slots.lock().pop();
        if let Some(slot) = free_slot {
            return slot;
        }

        // Allocate from current write position
        let pos = self.write_pos.fetch_add(1, Ordering::Relaxed);
        let chunk_idx = pos / self.chunk_capacity;
        let offset = pos % self.chunk_capacity;

        // Ensure we have enough chunks
        let mut chunks = self.chunks.write();
        while chunks.len() <= chunk_idx {
            chunks.push(UnsafeCell::new(
                vec![0.0f32; self.chunk_capacity * self.dimension].into_boxed_slice(),
            ));
        }
        drop(chunks);

        EmbeddingSlot::from_position(chunk_idx, offset)
    }
}

// SAFETY: EmbeddingSlab uses RwLock for all mutable access to chunks and index.
// The UnsafeCell is only accessed through the RwLock-protected methods, ensuring
// proper synchronization.
unsafe impl Send for EmbeddingSlab {}
unsafe impl Sync for EmbeddingSlab {}

impl Default for EmbeddingSlab {
    fn default() -> Self {
        Self::with_dimension(768) // Default to BERT dimension
    }
}

/// Compressed embedding for snapshot storage.
///
/// Automatically selects the best compression based on vector characteristics:
/// - `TensorTrain`: For high-dimensional dense vectors (768+), 10-20x compression
/// - Sparse: For vectors with >50% zeros
/// - Dense: Fallback for small or incompatible vectors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressedEmbedding {
    /// Dense vector storage.
    Dense(Vec<f32>),
    /// Sparse vector storage (dimension, positions, values).
    Sparse {
        /// Vector dimension.
        dimension: usize,
        /// Non-zero positions.
        positions: Vec<u32>,
        /// Non-zero values.
        values: Vec<f32>,
    },
    /// Tensor Train compressed storage (10-20x compression for 768+ dims).
    TensorTrain(tensor_compress::TTVector),
}

/// Minimum dimension for TT compression (below this, overhead exceeds savings).
const TT_MIN_DIMENSION: usize = 256;

impl CompressedEmbedding {
    /// Create from a dense vector, selecting optimal compression.
    ///
    /// Selection order:
    /// 1. Sparse if >50% zeros
    /// 2. `TensorTrain` if dimension >= 256 and has valid factorization
    /// 3. Dense otherwise
    #[must_use]
    pub fn from_dense(vector: &[f32]) -> Self {
        if vector.is_empty() {
            return Self::Dense(Vec::new());
        }

        let nnz = vector.iter().filter(|&&v| v.abs() > 1e-6).count();
        // For 0.5 threshold: sparse if nnz <= len/2, i.e., nnz*2 <= len
        let use_sparse = nnz * 2 <= vector.len();

        if use_sparse {
            let mut positions = Vec::with_capacity(nnz);
            let mut values = Vec::with_capacity(nnz);
            for (i, &v) in vector.iter().enumerate() {
                if v.abs() > 1e-6 {
                    if let Ok(pos) = u32::try_from(i) {
                        positions.push(pos);
                        values.push(v);
                    }
                }
            }
            return Self::Sparse {
                dimension: vector.len(),
                positions,
                values,
            };
        }

        // Try TT compression for high-dimensional dense vectors
        if vector.len() >= TT_MIN_DIMENSION {
            if let Ok(config) = tensor_compress::TTConfig::for_dim(vector.len()) {
                if let Ok(tt) = tensor_compress::tt_decompose(vector, &config) {
                    return Self::TensorTrain(tt);
                }
            }
        }

        Self::Dense(vector.to_vec())
    }

    /// Convert to a dense vector.
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        match self {
            Self::Dense(v) => v.clone(),
            Self::Sparse {
                dimension,
                positions,
                values,
            } => {
                let mut dense = vec![0.0f32; *dimension];
                for (&pos, &val) in positions.iter().zip(values.iter()) {
                    dense[pos as usize] = val;
                }
                dense
            },
            Self::TensorTrain(tt) => tensor_compress::tt_reconstruct(tt),
        }
    }

    /// Returns the compression format name.
    #[must_use]
    pub const fn format_name(&self) -> &'static str {
        match self {
            Self::Dense(_) => "dense",
            Self::Sparse { .. } => "sparse",
            Self::TensorTrain(_) => "tensor_train",
        }
    }
}

/// Serializable snapshot of `EmbeddingSlab` state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSlabSnapshot {
    dimension: usize,
    embeddings: Vec<(EntityId, CompressedEmbedding)>,
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc, thread, time::Instant};

    use super::*;

    #[test]
    fn test_new() {
        let slab = EmbeddingSlab::new(128, 100);
        assert_eq!(slab.dimension(), 128);
        assert_eq!(slab.len(), 0);
        assert!(slab.is_empty());
    }

    #[test]
    fn test_with_dimension() {
        let slab = EmbeddingSlab::with_dimension(768);
        assert_eq!(slab.dimension(), 768);
    }

    #[test]
    fn test_default() {
        let slab = EmbeddingSlab::default();
        assert_eq!(slab.dimension(), 768);
    }

    #[test]
    fn test_set_get() {
        let slab = EmbeddingSlab::new(4, 10);
        let entity = EntityId::new(1);
        let embedding = vec![1.0, 2.0, 3.0, 4.0];

        slab.set(entity, &embedding).unwrap();

        let retrieved = slab.get(entity).unwrap();
        assert_eq!(retrieved, embedding);
    }

    #[test]
    fn test_set_dimension_mismatch() {
        let slab = EmbeddingSlab::new(4, 10);
        let entity = EntityId::new(1);
        let wrong_embedding = vec![1.0, 2.0, 3.0]; // Wrong dimension

        let result = slab.set(entity, &wrong_embedding);
        assert!(matches!(
            result,
            Err(EmbeddingError::DimensionMismatch {
                expected: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn test_set_update() {
        let slab = EmbeddingSlab::new(4, 10);
        let entity = EntityId::new(1);

        slab.set(entity, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        slab.set(entity, &[5.0, 6.0, 7.0, 8.0]).unwrap();

        let retrieved = slab.get(entity).unwrap();
        assert_eq!(retrieved, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(slab.len(), 1);
    }

    #[test]
    fn test_contains() {
        let slab = EmbeddingSlab::new(4, 10);
        let entity = EntityId::new(1);

        assert!(!slab.contains(entity));
        slab.set(entity, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(slab.contains(entity));
    }

    #[test]
    fn test_delete() {
        let slab = EmbeddingSlab::new(4, 10);
        let entity = EntityId::new(1);

        slab.set(entity, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(slab.len(), 1);

        let deleted = slab.delete(entity);
        assert!(deleted);
        assert_eq!(slab.len(), 0);
        assert!(!slab.contains(entity));
    }

    #[test]
    fn test_delete_nonexistent() {
        let slab = EmbeddingSlab::new(4, 10);
        let entity = EntityId::new(1);

        let deleted = slab.delete(entity);
        assert!(!deleted);
    }

    #[test]
    fn test_free_slot_reuse() {
        let slab = EmbeddingSlab::new(4, 10);

        // Add and delete
        for i in 0..5 {
            slab.set(EntityId::new(i), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        }
        for i in 0..3 {
            slab.delete(EntityId::new(i));
        }

        // Free slots should be reused
        let before_capacity = slab.capacity();
        for i in 10..13 {
            slab.set(EntityId::new(i), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        }

        // Capacity shouldn't increase (reusing free slots)
        assert_eq!(slab.capacity(), before_capacity);
    }

    #[test]
    fn test_iter() {
        let slab = EmbeddingSlab::new(4, 10);

        for i in 0..3 {
            slab.set(EntityId::new(i), &[(i as f32), 0.0, 0.0, 0.0])
                .unwrap();
        }

        let entries = slab.entries();
        assert_eq!(entries.len(), 3);

        let ids: HashSet<u64> = entries.iter().map(|(e, _)| e.as_u64()).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_iter_with() {
        let slab = EmbeddingSlab::new(4, 10);

        for i in 0..3 {
            slab.set(EntityId::new(i), &[(i as f32), 0.0, 0.0, 0.0])
                .unwrap();
        }

        let mut count = 0;
        slab.iter_with(|_entity, _embedding| {
            count += 1;
        });

        assert_eq!(count, 3);
    }

    #[test]
    fn test_entity_ids() {
        let slab = EmbeddingSlab::new(4, 10);

        for i in 0..3 {
            slab.set(EntityId::new(i * 10), &[0.0, 0.0, 0.0, 0.0])
                .unwrap();
        }

        let ids = slab.entity_ids();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_clear() {
        let slab = EmbeddingSlab::new(4, 10);

        for i in 0..5 {
            slab.set(EntityId::new(i), &[0.0, 0.0, 0.0, 0.0]).unwrap();
        }

        slab.clear();
        assert_eq!(slab.len(), 0);
        assert!(slab.is_empty());
    }

    #[test]
    fn test_compact() {
        let slab = EmbeddingSlab::new(4, 10);

        // Add embeddings
        for i in 0..10 {
            slab.set(EntityId::new(i), &[(i as f32), 0.0, 0.0, 0.0])
                .unwrap();
        }

        // Delete some
        slab.delete(EntityId::new(2));
        slab.delete(EntityId::new(5));
        slab.delete(EntityId::new(7));

        // Compact
        slab.compact().unwrap();

        // Verify data integrity
        assert_eq!(slab.len(), 7);
        assert!(slab.contains(EntityId::new(0)));
        assert!(!slab.contains(EntityId::new(2)));
        assert!(slab.contains(EntityId::new(3)));
    }

    #[test]
    fn test_compact_empty() {
        let slab = EmbeddingSlab::new(4, 10);
        slab.compact().unwrap();
        assert!(slab.is_empty());
    }

    #[test]
    fn test_snapshot_restore() {
        let slab = EmbeddingSlab::new(4, 10);

        for i in 0..5 {
            slab.set(EntityId::new(i), &[(i as f32), 1.0, 2.0, 3.0])
                .unwrap();
        }

        let snapshot = slab.snapshot();
        let restored = EmbeddingSlab::restore(snapshot);

        assert_eq!(restored.dimension(), 4);
        assert_eq!(restored.len(), 5);

        for i in 0..5 {
            let expected = vec![i as f32, 1.0, 2.0, 3.0];
            assert_eq!(restored.get(EntityId::new(i)).unwrap(), expected);
        }
    }

    #[test]
    fn test_memory_bytes() {
        let slab = EmbeddingSlab::new(128, 100);
        let bytes = slab.memory_bytes();
        // Should be at least 1 chunk worth
        assert!(bytes >= 128 * 4 * 100);
    }

    #[test]
    fn test_capacity_grows() {
        let slab = EmbeddingSlab::new(4, 10);
        let initial_capacity = slab.capacity();

        // Add more than initial capacity
        for i in 0..(initial_capacity * 2) as u64 {
            slab.set(EntityId::new(i), &[0.0, 0.0, 0.0, 0.0]).unwrap();
        }

        // Capacity should have grown
        assert!(slab.capacity() > initial_capacity);
    }

    #[test]
    fn test_concurrent_reads_writes() {
        let slab = Arc::new(EmbeddingSlab::new(8, 1000));
        let mut handles = vec![];

        // Writer threads
        for t in 0..4 {
            let s = Arc::clone(&slab);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let entity = EntityId::new(t * 1000 + i);
                    let embedding = vec![t as f32; 8];
                    s.set(entity, &embedding).unwrap();
                }
            }));
        }

        // Reader threads
        for _ in 0..4 {
            let s = Arc::clone(&slab);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let _ = s.get(EntityId::new(0));
                    let _ = s.len();
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(slab.len(), 400);
    }

    #[test]
    fn test_no_resize_stall() {
        let slab = EmbeddingSlab::new(128, 100);
        let count = 10_000;

        let start = Instant::now();
        let mut max_op_time = std::time::Duration::ZERO;

        for i in 0..count {
            let op_start = Instant::now();
            let embedding = vec![i as f32; 128];
            slab.set(EntityId::new(i), &embedding).unwrap();
            let op_time = op_start.elapsed();
            if op_time > max_op_time {
                max_op_time = op_time;
            }
        }

        let total_time = start.elapsed();

        // No single operation should take more than 100ms (accounts for coverage overhead)
        assert!(
            max_op_time.as_millis() < 100,
            "Max operation time {:?} exceeded 100ms threshold",
            max_op_time
        );

        // Verify throughput is reasonable
        let ops_per_sec = count as f64 / total_time.as_secs_f64();
        assert!(
            ops_per_sec > 10_000.0,
            "Throughput {:.0} ops/sec too low",
            ops_per_sec
        );
    }

    #[test]
    fn test_error_display() {
        let err = EmbeddingError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        assert!(err.to_string().contains("128"));
        assert!(err.to_string().contains("64"));

        let err = EmbeddingError::NotFound(EntityId::new(42));
        assert!(err.to_string().contains("42"));

        let err = EmbeddingError::OutOfMemory;
        assert!(err.to_string().contains("memory"));
    }

    #[test]
    fn test_large_embeddings() {
        let slab = EmbeddingSlab::new(1024, 10);

        for i in 0..5 {
            let embedding: Vec<f32> = (0..1024).map(|j| (i * 1000 + j) as f32).collect();
            slab.set(EntityId::new(i), &embedding).unwrap();
        }

        // Verify retrieval
        for i in 0..5 {
            let embedding = slab.get(EntityId::new(i)).unwrap();
            assert_eq!(embedding.len(), 1024);
            assert_eq!(embedding[0], (i * 1000) as f32);
        }
    }

    // Sparse snapshot tests

    #[test]
    fn test_compressed_embedding_dense() {
        let dense = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let compressed = CompressedEmbedding::from_dense(&dense);

        // Less than 50% zeros, should be stored dense
        assert!(matches!(compressed, CompressedEmbedding::Dense(_)));

        let restored = compressed.to_dense();
        assert_eq!(restored, dense);
    }

    #[test]
    fn test_compressed_embedding_sparse() {
        // Create a sparse vector (>50% zeros)
        let mut sparse = vec![0.0f32; 100];
        sparse[0] = 1.0;
        sparse[50] = 2.0;
        sparse[99] = 3.0;

        let compressed = CompressedEmbedding::from_dense(&sparse);

        // Should be stored as sparse
        assert!(matches!(compressed, CompressedEmbedding::Sparse { .. }));

        let restored = compressed.to_dense();
        assert_eq!(restored.len(), 100);
        assert_eq!(restored[0], 1.0);
        assert_eq!(restored[50], 2.0);
        assert_eq!(restored[99], 3.0);
        assert_eq!(restored[1], 0.0);
    }

    #[test]
    fn test_sparse_snapshot_roundtrip() {
        let slab = EmbeddingSlab::new(100, 10);

        // Add a sparse embedding
        let mut sparse = vec![0.0f32; 100];
        sparse[0] = 1.0;
        sparse[50] = 2.0;
        sparse[99] = 3.0;
        slab.set(EntityId::new(1), &sparse).unwrap();

        // Add a dense embedding
        let dense: Vec<f32> = (0..100).map(|i| i as f32).collect();
        slab.set(EntityId::new(2), &dense).unwrap();

        // Snapshot and restore
        let snapshot = slab.snapshot();
        let restored = EmbeddingSlab::restore(snapshot);

        // Verify both embeddings
        assert_eq!(restored.len(), 2);

        let restored_sparse = restored.get(EntityId::new(1)).unwrap();
        assert_eq!(restored_sparse[0], 1.0);
        assert_eq!(restored_sparse[50], 2.0);
        assert_eq!(restored_sparse[99], 3.0);

        let restored_dense = restored.get(EntityId::new(2)).unwrap();
        assert_eq!(restored_dense, dense);
    }

    #[test]
    fn test_snapshot_serialization_size() {
        let slab = EmbeddingSlab::new(1000, 10);

        // Add very sparse vectors (3 non-zeros in 1000 elements = 99.7% sparse)
        for i in 0..5 {
            let mut sparse = vec![0.0f32; 1000];
            sparse[0] = i as f32;
            sparse[500] = i as f32 * 2.0;
            sparse[999] = i as f32 * 3.0;
            slab.set(EntityId::new(i), &sparse).unwrap();
        }

        let snapshot = slab.snapshot();

        // Verify all embeddings use sparse format
        for (_, emb) in &snapshot.embeddings {
            assert!(
                matches!(emb, CompressedEmbedding::Sparse { .. }),
                "Expected sparse format for 99.7% sparse vector"
            );
        }

        // Serialize and check size is much smaller than dense would be
        let serialized = bitcode::serialize(&snapshot).unwrap();
        let dense_size = 5 * 1000 * 4 + 100; // 5 vectors * 1000 floats * 4 bytes + overhead
        assert!(
            serialized.len() < dense_size / 10,
            "Sparse snapshot {} bytes should be much smaller than dense {} bytes",
            serialized.len(),
            dense_size
        );
    }

    #[test]
    fn test_compressed_embedding_from_empty_vector() {
        let empty: Vec<f32> = vec![];
        let compressed = CompressedEmbedding::from_dense(&empty);
        match compressed {
            CompressedEmbedding::Dense(v) => assert!(v.is_empty()),
            _ => panic!("Expected Dense for empty vector"),
        }
    }

    #[test]
    fn test_slot_bounds_validation() {
        let slab = EmbeddingSlab::new(4, 10);

        // Normal operations should not trigger bounds assertions
        for i in 0..100 {
            slab.set(EntityId::new(i), &[1.0, 2.0, 3.0, 4.0]).unwrap();
        }

        // Read back all - exercises bounds checks in get/iter
        for i in 0..100 {
            let _ = slab.get(EntityId::new(i));
        }

        slab.iter_with(|_, _| {});
        let _ = slab.entries();
        let _ = slab.snapshot();
    }

    // ========== Phase 3: Negative Path Tests ==========

    #[test]
    fn test_embedding_error_not_found_explicit() {
        // Verify NotFound error variant can be constructed and formatted
        let err = EmbeddingError::NotFound(EntityId::new(12345));
        let msg = err.to_string();
        assert!(msg.contains("12345"));
        assert!(msg.contains("not found"));
    }

    #[test]
    fn test_embedding_error_out_of_memory() {
        // Verify OutOfMemory error variant Display
        let err = EmbeddingError::OutOfMemory;
        let msg = err.to_string();
        assert!(msg.contains("memory"));
    }

    #[test]
    fn test_embedding_error_is_std_error() {
        // Verify EmbeddingError implements std::error::Error
        let err: Box<dyn std::error::Error> = Box::new(EmbeddingError::DimensionMismatch {
            expected: 128,
            actual: 64,
        });
        assert!(err.to_string().contains("dimension mismatch"));

        // Test source() returns None (no nested error)
        assert!(err.source().is_none());
    }

    #[test]
    fn test_embedding_get_nonexistent() {
        let slab = EmbeddingSlab::new(4, 10);

        // get() on non-existent entity returns None
        let result = slab.get(EntityId::new(999));
        assert!(result.is_none());
    }

    #[test]
    fn test_embedding_error_equality() {
        let a = EmbeddingError::NotFound(EntityId::new(42));
        let b = EmbeddingError::NotFound(EntityId::new(42));
        let c = EmbeddingError::NotFound(EntityId::new(99));

        assert_eq!(a, b);
        assert_ne!(a, c);

        let d = EmbeddingError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        let e = EmbeddingError::DimensionMismatch {
            expected: 128,
            actual: 64,
        };
        assert_eq!(d, e);
    }

    #[test]
    fn test_embedding_error_debug() {
        let err = EmbeddingError::OutOfMemory;
        let debug = format!("{:?}", err);
        assert!(debug.contains("OutOfMemory"));
    }

    #[test]
    fn test_compressed_embedding_format_name() {
        // Test format_name for all compression types
        let dense = CompressedEmbedding::Dense(vec![1.0, 2.0, 3.0]);
        assert_eq!(dense.format_name(), "dense");

        let sparse = CompressedEmbedding::Sparse {
            dimension: 100,
            positions: vec![0, 50],
            values: vec![1.0, 2.0],
        };
        assert_eq!(sparse.format_name(), "sparse");

        // For TensorTrain, we need to create one via from_dense with a large vector
        let large: Vec<f32> = (0..768).map(|i| i as f32).collect();
        let compressed = CompressedEmbedding::from_dense(&large);
        // May be TensorTrain or Dense depending on config
        let name = compressed.format_name();
        assert!(name == "tensor_train" || name == "dense");
    }
}
