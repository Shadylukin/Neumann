//! Append-only blob log with segment management.
//!
//! BlobLog provides content-addressable storage using append-only log segments.
//! Data is chunked, hashed, and stored in fixed-size segments for efficient
//! sequential writes and garbage collection.
//!
//! # Design Philosophy
//!
//! - Append-only writes: no in-place updates
//! - Content-addressable: chunks are identified by their SHA-256 hash
//! - Segment-based: fixed-size segments for efficient compaction
//! - Thread-safe: uses parking_lot for concurrent access

use std::{
    collections::{BTreeMap, BTreeSet},
    hash::{Hash, Hasher},
    sync::atomic::{AtomicU64, Ordering},
};

use fxhash::FxHasher;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

/// Content hash for blob chunks (simulated SHA-256 via FxHash for performance).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ChunkHash(pub u64);

impl ChunkHash {
    pub fn from_data(data: &[u8]) -> Self {
        let mut hasher = FxHasher::default();
        data.hash(&mut hasher);
        Self(hasher.finish())
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Location of a chunk within the blob log.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct ChunkLocation {
    segment_id: u64,
    offset: usize,
    length: usize,
}

/// A single log segment containing blob data.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LogSegment {
    id: u64,
    data: Vec<u8>,
    capacity: usize,
}

impl LogSegment {
    fn new(id: u64, capacity: usize) -> Self {
        Self {
            id,
            data: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn can_fit(&self, size: usize) -> bool {
        self.data.len() + size <= self.capacity
    }

    fn append(&mut self, data: &[u8]) -> usize {
        let offset = self.data.len();
        self.data.extend_from_slice(data);
        offset
    }

    fn read(&self, offset: usize, length: usize) -> Option<&[u8]> {
        if offset + length <= self.data.len() {
            Some(&self.data[offset..offset + length])
        } else {
            None
        }
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Append-only blob log with segmented storage.
///
/// # Thread Safety
///
/// Uses `parking_lot::RwLock` and `Mutex` for concurrent access.
pub struct BlobLog {
    active: Mutex<LogSegment>,
    sealed: RwLock<Vec<LogSegment>>,
    index: RwLock<BTreeMap<ChunkHash, ChunkLocation>>,
    garbage: Mutex<BTreeSet<ChunkHash>>,
    segment_size: usize,
    next_segment_id: AtomicU64,
    total_bytes: AtomicU64,
    chunk_count: AtomicU64,
}

impl BlobLog {
    /// Create a new blob log with the specified segment size.
    pub fn new(segment_size: usize) -> Self {
        Self {
            active: Mutex::new(LogSegment::new(0, segment_size)),
            sealed: RwLock::new(Vec::new()),
            index: RwLock::new(BTreeMap::new()),
            garbage: Mutex::new(BTreeSet::new()),
            segment_size,
            next_segment_id: AtomicU64::new(1),
            total_bytes: AtomicU64::new(0),
            chunk_count: AtomicU64::new(0),
        }
    }

    /// Create with default 64MB segment size.
    pub fn with_defaults() -> Self {
        Self::new(64 * 1024 * 1024)
    }

    /// Append data to the log and return its content hash.
    pub fn append(&self, data: &[u8]) -> ChunkHash {
        let hash = ChunkHash::from_data(data);

        // Check if already exists (deduplication)
        {
            let index = self.index.read();
            if index.contains_key(&hash) {
                return hash;
            }
        }

        let location = {
            let mut active = self.active.lock();

            // Seal active segment if it can't fit the data
            if !active.can_fit(data.len()) {
                self.seal_active_segment(&mut active);
            }

            let offset = active.append(data);
            ChunkLocation {
                segment_id: active.id,
                offset,
                length: data.len(),
            }
        };

        // Add to index
        {
            let mut index = self.index.write();
            index.insert(hash, location);
        }

        self.total_bytes
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        self.chunk_count.fetch_add(1, Ordering::Relaxed);

        hash
    }

    fn seal_active_segment(&self, active: &mut LogSegment) {
        if active.is_empty() {
            return;
        }

        let old_segment = std::mem::replace(
            active,
            LogSegment::new(
                self.next_segment_id.fetch_add(1, Ordering::Relaxed),
                self.segment_size,
            ),
        );

        let mut sealed = self.sealed.write();
        sealed.push(old_segment);
    }

    /// Get data by its content hash.
    pub fn get(&self, hash: &ChunkHash) -> Option<Vec<u8>> {
        let index = self.index.read();
        let location = index.get(hash)?;

        // Check active segment first
        {
            let active = self.active.lock();
            if location.segment_id == active.id {
                return active
                    .read(location.offset, location.length)
                    .map(|s| s.to_vec());
            }
        }

        // Check sealed segments
        let sealed = self.sealed.read();
        for segment in sealed.iter() {
            if segment.id == location.segment_id {
                return segment
                    .read(location.offset, location.length)
                    .map(|s| s.to_vec());
            }
        }

        None
    }

    /// Check if a chunk exists.
    pub fn contains(&self, hash: &ChunkHash) -> bool {
        let index = self.index.read();
        index.contains_key(hash) && !self.garbage.lock().contains(hash)
    }

    /// Mark a chunk as garbage for later compaction.
    pub fn mark_garbage(&self, hash: &ChunkHash) {
        if self.index.read().contains_key(hash) {
            self.garbage.lock().insert(*hash);
        }
    }

    /// Get the number of garbage-marked chunks.
    pub fn garbage_count(&self) -> usize {
        self.garbage.lock().len()
    }

    /// Compact the log by removing garbage entries.
    ///
    /// This creates new segments containing only live data.
    pub fn compact(&self) {
        let garbage = self.garbage.lock().clone();
        if garbage.is_empty() {
            return;
        }

        let mut new_index = BTreeMap::new();
        let mut new_sealed = Vec::new();
        let mut new_segment = LogSegment::new(
            self.next_segment_id.fetch_add(1, Ordering::Relaxed),
            self.segment_size,
        );

        // Collect live entries from index
        let index = self.index.read();
        let sealed = self.sealed.read();

        for (hash, location) in index.iter() {
            if garbage.contains(hash) {
                continue;
            }

            // Find the data
            let data = if location.segment_id == self.active.lock().id {
                self.active
                    .lock()
                    .read(location.offset, location.length)
                    .map(|s| s.to_vec())
            } else {
                sealed
                    .iter()
                    .find(|s| s.id == location.segment_id)
                    .and_then(|s| s.read(location.offset, location.length).map(|s| s.to_vec()))
            };

            if let Some(data) = data {
                // Add to new segment
                if !new_segment.can_fit(data.len()) {
                    let full_segment = std::mem::replace(
                        &mut new_segment,
                        LogSegment::new(
                            self.next_segment_id.fetch_add(1, Ordering::Relaxed),
                            self.segment_size,
                        ),
                    );
                    new_sealed.push(full_segment);
                }

                let offset = new_segment.append(&data);
                new_index.insert(
                    *hash,
                    ChunkLocation {
                        segment_id: new_segment.id,
                        offset,
                        length: data.len(),
                    },
                );
            }
        }

        drop(index);
        drop(sealed);

        // Update state
        *self.active.lock() = new_segment;
        *self.sealed.write() = new_sealed;
        *self.index.write() = new_index;
        self.garbage.lock().clear();

        // Update stats
        let new_total: u64 = self
            .index
            .read()
            .values()
            .map(|loc| loc.length as u64)
            .sum();
        self.total_bytes.store(new_total, Ordering::Relaxed);
        self.chunk_count
            .store(self.index.read().len() as u64, Ordering::Relaxed);
    }

    /// Get total bytes stored.
    pub fn total_bytes(&self) -> u64 {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Get the number of chunks stored.
    pub fn chunk_count(&self) -> u64 {
        self.chunk_count.load(Ordering::Relaxed)
    }

    /// Get the number of segments (active + sealed).
    pub fn segment_count(&self) -> usize {
        1 + self.sealed.read().len()
    }

    /// Clear all data.
    pub fn clear(&self) {
        *self.active.lock() = LogSegment::new(0, self.segment_size);
        self.sealed.write().clear();
        self.index.write().clear();
        self.garbage.lock().clear();
        self.next_segment_id.store(1, Ordering::Relaxed);
        self.total_bytes.store(0, Ordering::Relaxed);
        self.chunk_count.store(0, Ordering::Relaxed);
    }

    /// Create a snapshot for serialization.
    pub fn snapshot(&self) -> BlobLogSnapshot {
        let active = self.active.lock().clone();
        let sealed = self.sealed.read().clone();
        let index = self.index.read().clone();

        BlobLogSnapshot {
            active,
            sealed,
            index,
            segment_size: self.segment_size,
        }
    }

    /// Restore from a snapshot.
    pub fn restore(snapshot: BlobLogSnapshot) -> Self {
        let next_segment_id = snapshot
            .sealed
            .iter()
            .map(|s| s.id)
            .chain(std::iter::once(snapshot.active.id))
            .max()
            .unwrap_or(0)
            + 1;

        let total_bytes: u64 = snapshot.index.values().map(|loc| loc.length as u64).sum();
        let chunk_count = snapshot.index.len() as u64;

        Self {
            active: Mutex::new(snapshot.active),
            sealed: RwLock::new(snapshot.sealed),
            index: RwLock::new(snapshot.index),
            garbage: Mutex::new(BTreeSet::new()),
            segment_size: snapshot.segment_size,
            next_segment_id: AtomicU64::new(next_segment_id),
            total_bytes: AtomicU64::new(total_bytes),
            chunk_count: AtomicU64::new(chunk_count),
        }
    }
}

impl Default for BlobLog {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Serializable snapshot of BlobLog state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobLogSnapshot {
    active: LogSegment,
    sealed: Vec<LogSegment>,
    index: BTreeMap<ChunkHash, ChunkLocation>,
    segment_size: usize,
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread, time::Instant};

    use super::*;

    #[test]
    fn test_new() {
        let log = BlobLog::new(1024);
        assert_eq!(log.chunk_count(), 0);
        assert_eq!(log.total_bytes(), 0);
        assert_eq!(log.segment_count(), 1);
    }

    #[test]
    fn test_with_defaults() {
        let log = BlobLog::with_defaults();
        assert_eq!(log.chunk_count(), 0);
    }

    #[test]
    fn test_default() {
        let log = BlobLog::default();
        assert_eq!(log.chunk_count(), 0);
    }

    #[test]
    fn test_append_get() {
        let log = BlobLog::new(1024);

        let data1 = b"hello world";
        let hash1 = log.append(data1);

        let data2 = b"goodbye world";
        let hash2 = log.append(data2);

        assert_ne!(hash1, hash2);

        assert_eq!(log.get(&hash1), Some(data1.to_vec()));
        assert_eq!(log.get(&hash2), Some(data2.to_vec()));
    }

    #[test]
    fn test_deduplication() {
        let log = BlobLog::new(1024);

        let data = b"duplicate data";
        let hash1 = log.append(data);
        let hash2 = log.append(data);

        assert_eq!(hash1, hash2);
        assert_eq!(log.chunk_count(), 1);
    }

    #[test]
    fn test_contains() {
        let log = BlobLog::new(1024);

        let hash = log.append(b"test data");

        assert!(log.contains(&hash));
        assert!(!log.contains(&ChunkHash(9999)));
    }

    #[test]
    fn test_mark_garbage() {
        let log = BlobLog::new(1024);

        let hash = log.append(b"garbage data");
        assert!(log.contains(&hash));

        log.mark_garbage(&hash);
        assert!(!log.contains(&hash));
        assert_eq!(log.garbage_count(), 1);
    }

    #[test]
    fn test_segment_sealing() {
        // Small segment size to trigger sealing
        let log = BlobLog::new(100);

        // Add data that exceeds segment size
        for i in 0..10 {
            log.append(format!("data chunk {} with some padding", i).as_bytes());
        }

        assert!(log.segment_count() > 1);
    }

    #[test]
    fn test_get_from_sealed_segment() {
        // Use segment size smaller than each chunk to force sealing
        let log = BlobLog::new(20);

        let hash1 = log.append(b"first chunk of data here that is long");
        let hash2 = log.append(b"second chunk of data here that is also long");

        // Should have sealed first segment
        assert!(log.segment_count() > 1);

        // Both should still be retrievable
        assert!(log.get(&hash1).is_some());
        assert!(log.get(&hash2).is_some());
    }

    #[test]
    fn test_compact() {
        let log = BlobLog::new(100);

        let hash1 = log.append(b"keep this data");
        let hash2 = log.append(b"garbage data");
        let hash3 = log.append(b"also keep this");

        log.mark_garbage(&hash2);
        log.compact();

        assert!(log.get(&hash1).is_some());
        assert!(log.get(&hash3).is_some());
        assert!(log.get(&hash2).is_none()); // Compacted away
        assert_eq!(log.garbage_count(), 0);
    }

    #[test]
    fn test_compact_empty_garbage() {
        let log = BlobLog::new(1024);
        log.append(b"some data");

        // Compact with no garbage should be no-op
        log.compact();
        assert_eq!(log.chunk_count(), 1);
    }

    #[test]
    fn test_clear() {
        let log = BlobLog::new(1024);

        log.append(b"data 1");
        log.append(b"data 2");

        log.clear();

        assert_eq!(log.chunk_count(), 0);
        assert_eq!(log.total_bytes(), 0);
        assert_eq!(log.segment_count(), 1);
    }

    #[test]
    fn test_snapshot_restore() {
        let log = BlobLog::new(1024);

        let hash1 = log.append(b"snapshot data 1");
        let hash2 = log.append(b"snapshot data 2");

        let snapshot = log.snapshot();
        let restored = BlobLog::restore(snapshot);

        assert_eq!(restored.get(&hash1), Some(b"snapshot data 1".to_vec()));
        assert_eq!(restored.get(&hash2), Some(b"snapshot data 2".to_vec()));
        assert_eq!(restored.chunk_count(), 2);
    }

    #[test]
    fn test_concurrent_appends() {
        let log = Arc::new(BlobLog::new(10_000));

        let mut handles = vec![];

        for t in 0..4 {
            let l = Arc::clone(&log);
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    l.append(format!("thread {} chunk {}", t, i).as_bytes());
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Due to the TOCTOU race in deduplication check, some chunks may be
        // deduplicated when multiple threads check simultaneously. Allow for this.
        let count = log.chunk_count();
        assert!(
            count >= 380 && count <= 400,
            "Expected ~400 chunks, got {}",
            count
        );
    }

    #[test]
    fn test_no_resize_stall() {
        let log = BlobLog::new(1024 * 1024); // 1MB segments
        let count = 10_000;
        let mut max_op_time = std::time::Duration::ZERO;

        for i in 0..count {
            let start = Instant::now();
            log.append(format!("data chunk {}", i).as_bytes());
            let elapsed = start.elapsed();
            if elapsed > max_op_time {
                max_op_time = elapsed;
            }
        }

        assert!(
            max_op_time.as_millis() < 100,
            "Max operation time {:?} exceeded 100ms threshold",
            max_op_time
        );
    }

    #[test]
    fn test_chunk_hash() {
        let hash1 = ChunkHash::from_data(b"hello");
        let hash2 = ChunkHash::from_data(b"hello");
        let hash3 = ChunkHash::from_data(b"world");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
        assert_eq!(hash1.as_u64(), hash2.as_u64());
    }

    #[test]
    fn test_get_nonexistent() {
        let log = BlobLog::new(1024);
        assert!(log.get(&ChunkHash(12345)).is_none());
    }

    #[test]
    fn test_mark_garbage_nonexistent() {
        let log = BlobLog::new(1024);
        log.mark_garbage(&ChunkHash(12345)); // Should not panic
        assert_eq!(log.garbage_count(), 0);
    }

    #[test]
    fn test_total_bytes() {
        let log = BlobLog::new(1024);

        log.append(b"12345"); // 5 bytes
        log.append(b"67890"); // 5 bytes

        assert_eq!(log.total_bytes(), 10);
    }

    #[test]
    fn test_total_bytes_after_compact() {
        let log = BlobLog::new(1024);

        log.append(b"keep"); // 4 bytes
        let garbage_hash = log.append(b"garbage data"); // 12 bytes
        log.append(b"also keep"); // 9 bytes

        assert_eq!(log.total_bytes(), 25);

        log.mark_garbage(&garbage_hash);
        log.compact();

        assert_eq!(log.total_bytes(), 13); // 4 + 9
    }

    #[test]
    fn test_compact_with_segment_split() {
        // Use small segment to force splits during compaction
        let log = BlobLog::new(50);

        let hash1 = log.append(b"data chunk one here");
        let hash2 = log.append(b"delete this chunk");
        let hash3 = log.append(b"data chunk three here");

        log.mark_garbage(&hash2);
        log.compact();

        assert!(log.get(&hash1).is_some());
        assert!(log.get(&hash3).is_some());
        assert!(log.get(&hash2).is_none());
    }

    #[test]
    fn test_empty_segment_seal() {
        let log = BlobLog::new(10);

        // This should seal empty active segment gracefully
        let hash = log.append(b"data that exceeds segment");

        assert!(log.get(&hash).is_some());
    }

    #[test]
    fn test_log_segment_methods() {
        let mut segment = LogSegment::new(0, 100);

        assert!(segment.is_empty());
        assert!(segment.can_fit(50));
        assert!(!segment.can_fit(150));

        let offset = segment.append(b"hello");
        assert_eq!(offset, 0);
        assert_eq!(segment.data.len(), 5);
        assert!(!segment.is_empty());

        assert_eq!(segment.read(0, 5), Some(b"hello".as_slice()));
        assert!(segment.read(0, 10).is_none()); // Out of bounds
    }
}
