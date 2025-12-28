//! Dense embedding storage with chunked allocation.
//!
//! EmbeddingSlab stores dense f32 embeddings in contiguous chunks to avoid
//! large reallocations. Embeddings are stored by EntityId, with O(1) lookup
//! and append operations.
//!
//! # Design Philosophy
//!
//! - Append-only growth with chunked allocation
//! - Free slot reuse for deleted embeddings
//! - O(1) lookup by EntityId
//! - Zero-copy iteration for HNSW index building

use crate::entity_index::EntityId;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Default chunk size: 16MB of f32s = 4M floats
const DEFAULT_CHUNK_SIZE: usize = 4 * 1024 * 1024;

/// Error types for EmbeddingSlab operations.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingError {
    /// Embedding dimension mismatch.
    DimensionMismatch { expected: usize, actual: usize },
    /// EntityId not found.
    NotFound(EntityId),
    /// Slab is full and cannot allocate more chunks.
    OutOfMemory,
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "dimension mismatch: expected {}, got {}",
                    expected, actual
                )
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
    fn new(chunk: u32, offset: u32) -> Self {
        Self { chunk, offset }
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

    /// Chunked storage: each chunk is a fixed-size box.
    chunks: RwLock<Vec<Box<[f32]>>>,

    /// EntityId -> slot mapping.
    index: RwLock<BTreeMap<EntityId, EmbeddingSlot>>,

    /// Free slots for deleted embeddings.
    free_slots: Mutex<Vec<EmbeddingSlot>>,

    /// Write position in the current (last) chunk.
    write_pos: AtomicUsize,

    /// Total count of embeddings.
    count: AtomicUsize,
}

impl EmbeddingSlab {
    /// Create a new EmbeddingSlab with the given dimension and initial capacity.
    ///
    /// # Arguments
    ///
    /// * `dimension` - Fixed embedding dimension (e.g., 768 for BERT)
    /// * `initial_capacity` - Initial number of embeddings to pre-allocate
    pub fn new(dimension: usize, initial_capacity: usize) -> Self {
        let chunk_capacity = DEFAULT_CHUNK_SIZE / dimension;
        let chunk_capacity = chunk_capacity.max(1);

        let num_chunks = (initial_capacity / chunk_capacity).max(1);
        let chunks: Vec<Box<[f32]>> = (0..num_chunks)
            .map(|_| vec![0.0f32; chunk_capacity * dimension].into_boxed_slice())
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

    /// Create an EmbeddingSlab with default capacity.
    pub fn with_dimension(dimension: usize) -> Self {
        Self::new(dimension, 1000)
    }

    /// Get the embedding dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
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
        {
            let index = self.index.read();
            if let Some(&slot) = index.get(&entity) {
                // Update in-place
                let chunks = self.chunks.read();
                let chunk = &chunks[slot.chunk as usize];
                let start = slot.offset as usize * self.dimension;
                // Safety: we know the slot is valid
                unsafe {
                    let ptr = chunk.as_ptr().add(start) as *mut f32;
                    std::ptr::copy_nonoverlapping(embedding.as_ptr(), ptr, self.dimension);
                }
                return Ok(());
            }
        }

        // Get or allocate a slot
        let slot = self.allocate_slot()?;

        // Write the embedding
        {
            let chunks = self.chunks.read();
            let chunk = &chunks[slot.chunk as usize];
            let start = slot.offset as usize * self.dimension;
            unsafe {
                let ptr = chunk.as_ptr().add(start) as *mut f32;
                std::ptr::copy_nonoverlapping(embedding.as_ptr(), ptr, self.dimension);
            }
        }

        // Update index
        self.index.write().insert(entity, slot);
        self.count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Get an embedding by EntityId (cloned).
    pub fn get(&self, entity: EntityId) -> Option<Vec<f32>> {
        let index = self.index.read();
        let slot = index.get(&entity)?;

        let chunks = self.chunks.read();
        let chunk = &chunks[slot.chunk as usize];
        let start = slot.offset as usize * self.dimension;
        let end = start + self.dimension;

        Some(chunk[start..end].to_vec())
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
        let mut index = self.index.write();
        if let Some(slot) = index.remove(&entity) {
            self.free_slots.lock().push(slot);
            self.count.fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Get the number of embeddings.
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if the slab is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the total capacity (across all chunks).
    pub fn capacity(&self) -> usize {
        self.chunks.read().len() * self.chunk_capacity
    }

    /// Get memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let chunks = self.chunks.read();
        chunks.len() * self.chunk_capacity * self.dimension * std::mem::size_of::<f32>()
    }

    /// Iterate over all (EntityId, embedding) pairs.
    ///
    /// Note: This clones each embedding. For zero-copy access, use `iter_with`.
    pub fn iter(&self) -> Vec<(EntityId, Vec<f32>)> {
        let index = self.index.read();
        let chunks = self.chunks.read();

        index
            .iter()
            .map(|(&entity, &slot)| {
                let chunk = &chunks[slot.chunk as usize];
                let start = slot.offset as usize * self.dimension;
                let end = start + self.dimension;
                (entity, chunk[start..end].to_vec())
            })
            .collect()
    }

    /// Iterate with a callback for zero-copy access.
    ///
    /// The callback receives each (EntityId, &[f32]) pair.
    pub fn iter_with<F>(&self, mut f: F)
    where
        F: FnMut(EntityId, &[f32]),
    {
        let index = self.index.read();
        let chunks = self.chunks.read();

        for (&entity, &slot) in index.iter() {
            let chunk = &chunks[slot.chunk as usize];
            let start = slot.offset as usize * self.dimension;
            let end = start + self.dimension;
            f(entity, &chunk[start..end]);
        }
    }

    /// Get all EntityIds with embeddings.
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
    pub fn compact(&self) {
        // Build new compact storage
        let old_entries: Vec<(EntityId, Vec<f32>)> = self.iter();

        if old_entries.is_empty() {
            return;
        }

        // Clear and rebuild
        self.clear();

        for (entity, embedding) in old_entries {
            let _ = self.set(entity, &embedding);
        }
    }

    /// Get serializable state for snapshots.
    pub fn snapshot(&self) -> EmbeddingSlabSnapshot {
        let index = self.index.read();
        let chunks = self.chunks.read();

        // Collect all embeddings
        let embeddings: Vec<(EntityId, Vec<f32>)> = index
            .iter()
            .map(|(&entity, &slot)| {
                let chunk = &chunks[slot.chunk as usize];
                let start = slot.offset as usize * self.dimension;
                let end = start + self.dimension;
                (entity, chunk[start..end].to_vec())
            })
            .collect();

        EmbeddingSlabSnapshot {
            dimension: self.dimension,
            embeddings,
        }
    }

    /// Restore from a snapshot.
    pub fn restore(snapshot: EmbeddingSlabSnapshot) -> Self {
        let slab = Self::new(snapshot.dimension, snapshot.embeddings.len().max(1));

        for (entity, embedding) in snapshot.embeddings {
            let _ = slab.set(entity, &embedding);
        }

        slab
    }

    /// Allocate a slot for a new embedding.
    fn allocate_slot(&self) -> Result<EmbeddingSlot, EmbeddingError> {
        // Try to reuse a free slot
        {
            let mut free = self.free_slots.lock();
            if let Some(slot) = free.pop() {
                return Ok(slot);
            }
        }

        // Allocate from current write position
        let pos = self.write_pos.fetch_add(1, Ordering::Relaxed);
        let chunk_idx = pos / self.chunk_capacity;
        let offset = pos % self.chunk_capacity;

        // Ensure we have enough chunks
        {
            let mut chunks = self.chunks.write();
            while chunks.len() <= chunk_idx {
                chunks.push(vec![0.0f32; self.chunk_capacity * self.dimension].into_boxed_slice());
            }
        }

        Ok(EmbeddingSlot::new(chunk_idx as u32, offset as u32))
    }
}

impl Default for EmbeddingSlab {
    fn default() -> Self {
        Self::with_dimension(768) // Default to BERT dimension
    }
}

/// Serializable snapshot of EmbeddingSlab state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSlabSnapshot {
    dimension: usize,
    embeddings: Vec<(EntityId, Vec<f32>)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

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

        let entries = slab.iter();
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
        slab.compact();

        // Verify data integrity
        assert_eq!(slab.len(), 7);
        assert!(slab.contains(EntityId::new(0)));
        assert!(!slab.contains(EntityId::new(2)));
        assert!(slab.contains(EntityId::new(3)));
    }

    #[test]
    fn test_compact_empty() {
        let slab = EmbeddingSlab::new(4, 10);
        slab.compact(); // Should not panic
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

        // No single operation should take more than 50ms (accounts for coverage overhead)
        assert!(
            max_op_time.as_millis() < 50,
            "Max operation time {:?} exceeded 50ms threshold",
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
}
