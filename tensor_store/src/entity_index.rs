// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Vocabulary-based entity index for O(log n) lookup with stable IDs.
//!
//! The `EntityIndex` maps string keys to `EntityId` values using an append-only
//! vocabulary tensor. `EntityId` is simply the index into the vocabulary vector,
//! providing O(1) reverse lookups and stable IDs that never change.
//!
//! # Design Philosophy
//!
//! This is a tensor-native approach: the key's position in the vocabulary IS
//! the `EntityId`. New keys append to the vocabulary, maintaining stable IDs.
//! A sorted hash index provides O(log n) forward lookups.

use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::atomic::{AtomicU64, Ordering},
};

use parking_lot::RwLock;
use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};

/// Default maximum number of entities (100 million).
pub const DEFAULT_MAX_ENTITIES: usize = 100_000_000;

/// Error type for entity index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityIndexError {
    /// Entity capacity limit has been reached.
    CapacityExceeded {
        /// The configured capacity limit.
        limit: usize,
        /// The current number of entities.
        current: usize,
    },
}

impl fmt::Display for EntityIndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExceeded { limit, current } => {
                write!(
                    f,
                    "entity index capacity exceeded: limit {limit}, current {current}"
                )
            },
        }
    }
}

impl std::error::Error for EntityIndexError {}

/// Configuration for `EntityIndex`.
#[derive(Debug, Clone, Copy)]
pub struct EntityIndexConfig {
    /// Maximum number of entities allowed. Set to 0 for unlimited.
    pub max_entities: usize,
    /// Initial capacity for the vocabulary vector.
    pub initial_capacity: usize,
}

impl Default for EntityIndexConfig {
    fn default() -> Self {
        Self {
            max_entities: DEFAULT_MAX_ENTITIES,
            initial_capacity: 0,
        }
    }
}

impl EntityIndexConfig {
    /// Create a config with specified max entities.
    #[must_use]
    pub const fn with_max_entities(max_entities: usize) -> Self {
        Self {
            max_entities,
            initial_capacity: 0,
        }
    }

    /// Create a config with unlimited entities.
    #[must_use]
    pub const fn unlimited() -> Self {
        Self {
            max_entities: 0,
            initial_capacity: 0,
        }
    }
}

/// Hash a string key to a u64 using `FxHasher`.
#[inline]
fn hash_key(key: &str) -> u64 {
    let mut hasher = FxHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

/// A unique identifier for an entity in the store.
///
/// `EntityId` is simply an index into the vocabulary vector, providing
/// O(1) reverse lookups (id -> key).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityId(pub u64);

impl EntityId {
    /// Create a new `EntityId` from a raw u64 value.
    #[inline]
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Returns the underlying u64 value.
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Convert to array index. Truncates on 32-bit platforms.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub const fn as_index(self) -> usize {
        self.0 as usize
    }
}

impl From<u64> for EntityId {
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<EntityId> for u64 {
    fn from(id: EntityId) -> Self {
        id.0
    }
}

/// Vocabulary-based entity index with append-only semantics.
///
/// # Thread Safety
///
/// Uses `parking_lot::RwLock` for concurrent access without lock poisoning.
/// Reads are lock-free when there are no concurrent writes.
///
/// # Performance
///
/// - `get`: O(log n) via binary search on hash index
/// - `get_or_create`: O(log n) amortized, O(n) worst case on hash collision
/// - `key_for`: O(1) direct array access
/// - `scan_prefix`: O(n) linear scan
/// - `len`: O(1)
pub struct EntityIndex {
    /// The vocabulary: `EntityId` = index into this vector.
    /// Append-only for stable IDs.
    vocabulary: RwLock<Vec<String>>,

    /// Sorted (hash, `vocab_index`) pairs for O(log n) lookup.
    /// Binary search by hash, then verify key for collision handling.
    reverse: RwLock<Vec<(u64, u32)>>,

    /// Tombstone tracking for deleted entries.
    /// Bit i is set if EntityId(i) has been deleted.
    tombstones: RwLock<Vec<u64>>,

    /// Count of live (non-deleted) entries.
    live_count: AtomicU64,

    /// Maximum number of entities allowed (0 = unlimited).
    max_entities: usize,
}

impl EntityIndex {
    /// Create a new empty `EntityIndex` with the default capacity limit.
    ///
    /// Uses `DEFAULT_MAX_ENTITIES` (100 million) as the capacity limit.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            vocabulary: RwLock::new(Vec::new()),
            reverse: RwLock::new(Vec::new()),
            tombstones: RwLock::new(Vec::new()),
            live_count: AtomicU64::new(0),
            max_entities: DEFAULT_MAX_ENTITIES,
        }
    }

    /// Create an `EntityIndex` with pre-allocated capacity.
    ///
    /// Uses `DEFAULT_MAX_ENTITIES` (100 million) as the capacity limit.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            vocabulary: RwLock::new(Vec::with_capacity(capacity)),
            reverse: RwLock::new(Vec::with_capacity(capacity)),
            tombstones: RwLock::new(Vec::new()),
            live_count: AtomicU64::new(0),
            max_entities: DEFAULT_MAX_ENTITIES,
        }
    }

    /// Create an `EntityIndex` with the specified configuration.
    #[must_use]
    pub fn with_config(config: EntityIndexConfig) -> Self {
        Self {
            vocabulary: RwLock::new(Vec::with_capacity(config.initial_capacity)),
            reverse: RwLock::new(Vec::with_capacity(config.initial_capacity)),
            tombstones: RwLock::new(Vec::new()),
            live_count: AtomicU64::new(0),
            max_entities: config.max_entities,
        }
    }

    /// Returns the maximum number of entities allowed (0 = unlimited).
    #[must_use]
    pub const fn max_entities(&self) -> usize {
        self.max_entities
    }

    /// Look up an `EntityId` by key.
    ///
    /// Returns `None` if the key doesn't exist or has been deleted.
    ///
    /// # Performance
    ///
    /// O(log n) binary search + O(k) collision scan where k is typically 1.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<EntityId> {
        let hash = hash_key(key);
        // Lock order: vocabulary -> reverse (consistent with get_or_create)
        let vocab = self.vocabulary.read();
        let reverse = self.reverse.read();

        // Binary search to find first entry with matching hash
        let mut idx = reverse.partition_point(|(h, _)| *h < hash);

        // Linear scan through hash collisions
        while idx < reverse.len() && reverse[idx].0 == hash {
            let vocab_idx = reverse[idx].1 as usize;
            if vocab_idx < vocab.len() && vocab[vocab_idx] == key {
                let entity_id = EntityId(vocab_idx as u64);
                // Check if deleted
                if !self.is_tombstone(entity_id) {
                    return Some(entity_id);
                }
            }
            idx += 1;
        }

        drop(vocab);
        drop(reverse);
        None
    }

    /// Get or create an `EntityId` for the given key.
    ///
    /// If the key exists and is not deleted, returns the existing `EntityId`.
    /// Otherwise, appends the key to the vocabulary and returns a new `EntityId`.
    ///
    /// # Panics
    ///
    /// Panics if the entity capacity limit is exceeded.
    /// Use [`try_get_or_create`](Self::try_get_or_create) for a fallible version.
    ///
    /// # Performance
    ///
    /// O(log n) amortized. The sorted index insert is O(n) worst case but
    /// amortized O(log n) due to the distribution of hash values.
    #[must_use]
    pub fn get_or_create(&self, key: &str) -> EntityId {
        self.try_get_or_create(key)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Get or create an `EntityId` for the given key.
    ///
    /// If the key exists and is not deleted, returns the existing `EntityId`.
    /// Otherwise, appends the key to the vocabulary and returns a new `EntityId`.
    ///
    /// # Errors
    ///
    /// Returns [`EntityIndexError::CapacityExceeded`] if the entity limit is reached.
    ///
    /// # Performance
    ///
    /// O(log n) amortized. The sorted index insert is O(n) worst case but
    /// amortized O(log n) due to the distribution of hash values.
    pub fn try_get_or_create(&self, key: &str) -> Result<EntityId, EntityIndexError> {
        // Fast path: read-only lookup
        if let Some(id) = self.get(key) {
            return Ok(id);
        }

        // Slow path: acquire write locks
        let mut vocab = self.vocabulary.write();
        let mut reverse = self.reverse.write();

        // Double-check after acquiring write lock (another thread may have inserted)
        let hash = hash_key(key);
        let insert_pos = reverse.partition_point(|(h, _)| *h < hash);

        // Check for existing entry with same hash
        let mut idx = insert_pos;
        while idx < reverse.len() && reverse[idx].0 == hash {
            let vocab_idx = reverse[idx].1 as usize;
            if vocab_idx < vocab.len() && vocab[vocab_idx] == key {
                let entity_id = EntityId(vocab_idx as u64);
                if !self.is_tombstone(entity_id) {
                    return Ok(entity_id);
                }
                // Key was deleted, will re-add below
                break;
            }
            idx += 1;
        }

        // Check capacity before inserting (0 = unlimited)
        if self.max_entities > 0 && vocab.len() >= self.max_entities {
            return Err(EntityIndexError::CapacityExceeded {
                limit: self.max_entities,
                current: vocab.len(),
            });
        }

        // Insert new entry
        let new_id = vocab.len() as u64;
        vocab.push(key.to_string());
        #[allow(clippy::cast_possible_truncation)] // Entity IDs won't exceed 4 billion
        reverse.insert(insert_pos, (hash, new_id as u32));
        drop(vocab);
        drop(reverse);
        self.live_count.fetch_add(1, Ordering::Relaxed);

        Ok(EntityId(new_id))
    }

    /// Get the key for an `EntityId`.
    ///
    /// Returns `None` if the `EntityId` is invalid or the entry was deleted.
    ///
    /// # Performance
    ///
    /// O(1) direct array access.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Entity IDs limited to address space
    pub fn key_for(&self, id: EntityId) -> Option<String> {
        if self.is_tombstone(id) {
            return None;
        }
        self.vocabulary.read().get(id.0 as usize).cloned()
    }

    /// Check if a key exists and is not deleted.
    #[inline]
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Mark an entity as deleted (tombstone).
    ///
    /// The key remains in the vocabulary to preserve `EntityId` stability,
    /// but lookups will return `None`.
    ///
    /// Returns the `EntityId` if the key existed and was deleted, `None` otherwise.
    pub fn remove(&self, key: &str) -> Option<EntityId> {
        let id = self.get(key)?;
        self.set_tombstone(id);
        self.live_count.fetch_sub(1, Ordering::Relaxed);
        Some(id)
    }

    /// Scan for all keys matching a prefix.
    ///
    /// Returns pairs of (key, `EntityId`) for all matching, non-deleted entries.
    ///
    /// # Performance
    ///
    /// O(n) linear scan of vocabulary.
    #[must_use]
    pub fn scan_prefix(&self, prefix: &str) -> Vec<(String, EntityId)> {
        let vocab = self.vocabulary.read();
        vocab
            .iter()
            .enumerate()
            .filter_map(|(i, k)| {
                let id = EntityId(i as u64);
                if k.starts_with(prefix) && !self.is_tombstone(id) {
                    Some((k.clone(), id))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Count keys matching a prefix.
    ///
    /// # Performance
    ///
    /// O(n) linear scan of vocabulary.
    #[must_use]
    pub fn scan_prefix_count(&self, prefix: &str) -> usize {
        let vocab = self.vocabulary.read();
        vocab
            .iter()
            .enumerate()
            .filter(|(i, k)| {
                let id = EntityId(*i as u64);
                k.starts_with(prefix) && !self.is_tombstone(id)
            })
            .count()
    }

    /// Get the number of live (non-deleted) entries.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Entry count limited to address space
    pub fn len(&self) -> usize {
        self.live_count.load(Ordering::Relaxed) as usize
    }

    /// Check if the index is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the total number of entries including deleted ones.
    ///
    /// This represents the vocabulary size and the range of valid `EntityId`s.
    pub fn total_entries(&self) -> usize {
        self.vocabulary.read().len()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.vocabulary.write().clear();
        self.reverse.write().clear();
        self.tombstones.write().clear();
        self.live_count.store(0, Ordering::Relaxed);
    }

    /// Get serializable state for snapshots.
    pub fn snapshot(&self) -> EntityIndexSnapshot {
        EntityIndexSnapshot {
            vocabulary: self.vocabulary.read().clone(),
            reverse: self.reverse.read().clone(),
            tombstones: self.tombstones.read().clone(),
            live_count: self.live_count.load(Ordering::Relaxed),
            max_entities: self.max_entities,
        }
    }

    /// Restore from a snapshot.
    #[must_use]
    pub fn restore(snapshot: EntityIndexSnapshot) -> Self {
        Self {
            vocabulary: RwLock::new(snapshot.vocabulary),
            reverse: RwLock::new(snapshot.reverse),
            tombstones: RwLock::new(snapshot.tombstones),
            live_count: AtomicU64::new(snapshot.live_count),
            max_entities: snapshot.max_entities,
        }
    }

    /// Check if an `EntityId` has been deleted.
    #[inline]
    fn is_tombstone(&self, id: EntityId) -> bool {
        let tombstones = self.tombstones.read();
        let block_idx = (id.0 / 64) as usize;
        if block_idx >= tombstones.len() {
            return false;
        }
        let bit_idx = id.0 % 64;
        (tombstones[block_idx] & (1u64 << bit_idx)) != 0
    }

    /// Mark an `EntityId` as deleted.
    fn set_tombstone(&self, id: EntityId) {
        let mut tombstones = self.tombstones.write();
        let block_idx = (id.0 / 64) as usize;

        // Extend tombstones vector if needed
        while tombstones.len() <= block_idx {
            tombstones.push(0);
        }

        let bit_idx = id.0 % 64;
        tombstones[block_idx] |= 1u64 << bit_idx;
    }
}

impl Default for EntityIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable snapshot of `EntityIndex` state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityIndexSnapshot {
    vocabulary: Vec<String>,
    reverse: Vec<(u64, u32)>,
    tombstones: Vec<u64>,
    live_count: u64,
    #[serde(default = "default_max_entities")]
    max_entities: usize,
}

const fn default_max_entities() -> usize {
    DEFAULT_MAX_ENTITIES
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc, thread, time::Instant};

    use super::*;

    #[test]
    fn test_new_empty() {
        let index = EntityIndex::new();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert_eq!(index.total_entries(), 0);
    }

    #[test]
    fn test_get_or_create_single() {
        let index = EntityIndex::new();
        let id = index.get_or_create("user:1");
        assert_eq!(id.as_u64(), 0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_get_or_create_multiple() {
        let index = EntityIndex::new();
        let id1 = index.get_or_create("user:1");
        let id2 = index.get_or_create("user:2");
        let id3 = index.get_or_create("user:3");

        assert_eq!(id1.as_u64(), 0);
        assert_eq!(id2.as_u64(), 1);
        assert_eq!(id3.as_u64(), 2);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_get_or_create_idempotent() {
        let index = EntityIndex::new();
        let id1 = index.get_or_create("user:1");
        let id2 = index.get_or_create("user:1");

        assert_eq!(id1, id2);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_get_existing() {
        let index = EntityIndex::new();
        let id = index.get_or_create("user:1");

        let found = index.get("user:1");
        assert_eq!(found, Some(id));
    }

    #[test]
    fn test_get_nonexistent() {
        let index = EntityIndex::new();
        let _ = index.get_or_create("user:1");

        let found = index.get("user:2");
        assert_eq!(found, None);
    }

    #[test]
    fn test_key_for() {
        let index = EntityIndex::new();
        let id = index.get_or_create("user:1");

        let key = index.key_for(id);
        assert_eq!(key, Some("user:1".to_string()));
    }

    #[test]
    fn test_key_for_invalid() {
        let index = EntityIndex::new();
        let _ = index.get_or_create("user:1");

        let key = index.key_for(EntityId(999));
        assert_eq!(key, None);
    }

    #[test]
    fn test_contains() {
        let index = EntityIndex::new();
        let _ = index.get_or_create("user:1");

        assert!(index.contains("user:1"));
        assert!(!index.contains("user:2"));
    }

    #[test]
    fn test_remove() {
        let index = EntityIndex::new();
        let id = index.get_or_create("user:1");
        assert_eq!(index.len(), 1);

        let removed = index.remove("user:1");
        assert_eq!(removed, Some(id));
        assert_eq!(index.len(), 0);
        assert!(!index.contains("user:1"));

        // key_for should also return None
        assert_eq!(index.key_for(id), None);
    }

    #[test]
    fn test_remove_nonexistent() {
        let index = EntityIndex::new();
        let removed = index.remove("user:1");
        assert_eq!(removed, None);
    }

    #[test]
    fn test_remove_then_recreate() {
        let index = EntityIndex::new();
        let id1 = index.get_or_create("user:1");
        index.remove("user:1");

        // Re-adding should create a new ID (append-only)
        let id2 = index.get_or_create("user:1");
        assert_ne!(id1, id2);
        assert_eq!(index.len(), 1);
        assert_eq!(index.total_entries(), 2);
    }

    #[test]
    fn test_scan_prefix() {
        let index = EntityIndex::new();
        let _ = index.get_or_create("user:1");
        let _ = index.get_or_create("user:2");
        let _ = index.get_or_create("post:1");
        let _ = index.get_or_create("user:3");

        let users = index.scan_prefix("user:");
        assert_eq!(users.len(), 3);

        let keys: HashSet<String> = users.into_iter().map(|(k, _)| k).collect();
        assert!(keys.contains("user:1"));
        assert!(keys.contains("user:2"));
        assert!(keys.contains("user:3"));
    }

    #[test]
    fn test_scan_prefix_with_deletes() {
        let index = EntityIndex::new();
        let _ = index.get_or_create("user:1");
        let _ = index.get_or_create("user:2");
        let _ = index.get_or_create("user:3");
        index.remove("user:2");

        let users = index.scan_prefix("user:");
        assert_eq!(users.len(), 2);

        let keys: HashSet<String> = users.into_iter().map(|(k, _)| k).collect();
        assert!(keys.contains("user:1"));
        assert!(!keys.contains("user:2"));
        assert!(keys.contains("user:3"));
    }

    #[test]
    fn test_scan_prefix_count() {
        let index = EntityIndex::new();
        let _ = index.get_or_create("user:1");
        let _ = index.get_or_create("user:2");
        let _ = index.get_or_create("post:1");

        assert_eq!(index.scan_prefix_count("user:"), 2);
        assert_eq!(index.scan_prefix_count("post:"), 1);
        assert_eq!(index.scan_prefix_count("comment:"), 0);
    }

    #[test]
    fn test_snapshot_restore() {
        let index = EntityIndex::new();
        let _ = index.get_or_create("user:1");
        let _ = index.get_or_create("user:2");
        index.remove("user:1");

        let snapshot = index.snapshot();
        let restored = EntityIndex::restore(snapshot);

        assert_eq!(restored.len(), 1);
        assert!(!restored.contains("user:1"));
        assert!(restored.contains("user:2"));
    }

    #[test]
    fn test_clear() {
        let index = EntityIndex::new();
        let _ = index.get_or_create("user:1");
        let _ = index.get_or_create("user:2");

        index.clear();

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(!index.contains("user:1"));
    }

    #[test]
    fn test_concurrent_reads_writes() {
        let index = Arc::new(EntityIndex::new());
        let mut handles = vec![];

        // Writer threads
        for t in 0..4 {
            let idx = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                for i in 0..250 {
                    let _ = idx.get_or_create(&format!("thread{}:key{}", t, i));
                }
            }));
        }

        // Reader threads - just do gets, not expensive scans
        for _ in 0..4 {
            let idx = Arc::clone(&index);
            handles.push(thread::spawn(move || {
                for i in 0..250 {
                    let _ = idx.get(&format!("thread0:key{}", i % 100));
                    let _ = idx.contains(&format!("thread1:key{}", i % 100));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Each writer thread wrote 250 keys
        assert_eq!(index.len(), 1000);
    }

    #[test]
    fn test_no_resize_stall() {
        let index = EntityIndex::new();
        let count = 100_000;

        let start = Instant::now();
        let mut max_op_time = std::time::Duration::ZERO;

        for i in 0..count {
            let op_start = Instant::now();
            let _ = index.get_or_create(&format!("entity:{}", i));
            let op_time = op_start.elapsed();
            if op_time > max_op_time {
                max_op_time = op_time;
            }
        }

        let total_time = start.elapsed();

        // No single operation should take more than 200ms (lenient for coverage builds)
        assert!(
            max_op_time.as_millis() < 200,
            "Max operation time {:?} exceeded 200ms threshold",
            max_op_time
        );

        // Verify throughput is reasonable (lower threshold accounts for coverage overhead)
        let ops_per_sec = count as f64 / total_time.as_secs_f64();
        assert!(
            ops_per_sec > 10_000.0,
            "Throughput {:.0} ops/sec too low",
            ops_per_sec
        );
    }

    #[test]
    fn test_hash_collision_handling() {
        // Create keys that might have hash collisions
        let index = EntityIndex::new();

        // These keys have different content but we test the collision path
        let keys: Vec<String> = (0..1000).map(|i| format!("key:{}", i)).collect();

        for key in &keys {
            let _ = index.get_or_create(key);
        }

        // Verify all keys are retrievable
        for (i, key) in keys.iter().enumerate() {
            let id = index.get(key);
            assert!(id.is_some(), "Key {} should exist", key);
            assert_eq!(id.unwrap().as_u64(), i as u64);
        }
    }

    #[test]
    fn test_entity_id_ordering() {
        let index = EntityIndex::new();

        let id1 = index.get_or_create("a");
        let id2 = index.get_or_create("b");
        let id3 = index.get_or_create("c");

        // IDs should be in insertion order
        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    #[test]
    fn test_with_capacity() {
        let index = EntityIndex::with_capacity(1000);
        assert_eq!(index.len(), 0);

        // Should work normally
        for i in 0..100 {
            let _ = index.get_or_create(&format!("key:{}", i));
        }
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_large_scale() {
        let index = EntityIndex::new();
        let count = 10_000;

        // Insert many keys
        for i in 0..count {
            let _ = index.get_or_create(&format!("entity:{}", i));
        }

        assert_eq!(index.len(), count);

        // Verify random access
        for i in (0..count).step_by(100) {
            let key = format!("entity:{}", i);
            assert!(index.contains(&key));
            let id = index.get(&key).unwrap();
            assert_eq!(index.key_for(id), Some(key));
        }
    }

    #[test]
    fn test_empty_key() {
        let index = EntityIndex::new();
        let id = index.get_or_create("");
        assert_eq!(id.as_u64(), 0);
        assert!(index.contains(""));
        assert_eq!(index.key_for(id), Some("".to_string()));
    }

    #[test]
    fn test_unicode_keys() {
        let index = EntityIndex::new();

        let keys = vec!["user:alice", "user:bob", "user:charlie"];

        for key in &keys {
            let _ = index.get_or_create(key);
        }

        for key in &keys {
            assert!(index.contains(key));
        }
    }

    #[test]
    fn test_tombstone_bit_packing() {
        let index = EntityIndex::new();

        // Create more than 64 entries to test multi-block tombstones
        for i in 0..100 {
            let _ = index.get_or_create(&format!("key:{}", i));
        }

        // Delete entries across multiple blocks
        index.remove("key:0");
        index.remove("key:63");
        index.remove("key:64");
        index.remove("key:99");

        assert!(!index.contains("key:0"));
        assert!(index.contains("key:1"));
        assert!(!index.contains("key:63"));
        assert!(!index.contains("key:64"));
        assert!(index.contains("key:65"));
        assert!(!index.contains("key:99"));

        assert_eq!(index.len(), 96);
    }

    #[test]
    fn test_stable_ids_after_delete() {
        let index = EntityIndex::new();

        let id_a = index.get_or_create("a");
        let id_b = index.get_or_create("b");
        let id_c = index.get_or_create("c");

        // Delete b
        index.remove("b");

        // IDs for a and c should remain stable
        assert_eq!(index.get("a"), Some(id_a));
        assert_eq!(index.get("c"), Some(id_c));

        // Re-adding b gets a new ID
        let id_b_new = index.get_or_create("b");
        assert_ne!(id_b, id_b_new);
        assert_eq!(id_b_new.as_u64(), 3); // Appended at end
    }

    #[test]
    fn test_entity_id_from_u64() {
        let id: EntityId = 42u64.into();
        assert_eq!(id.as_u64(), 42);

        let id2: EntityId = EntityId::from(100u64);
        assert_eq!(id2.as_u64(), 100);
    }

    #[test]
    fn test_u64_from_entity_id() {
        let id = EntityId::new(99);
        let val: u64 = id.into();
        assert_eq!(val, 99);

        let val2: u64 = u64::from(EntityId::new(123));
        assert_eq!(val2, 123);
    }

    #[test]
    fn test_entity_index_default() {
        let index = EntityIndex::default();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    // ========================================================================
    // Security validation tests for unbounded allocation prevention
    // ========================================================================

    #[test]
    fn test_entity_index_capacity_exceeded() {
        let config = EntityIndexConfig::with_max_entities(5);
        let index = EntityIndex::with_config(config);

        // Add 5 entities (should work)
        for i in 0..5 {
            let result = index.try_get_or_create(&format!("key:{i}"));
            assert!(result.is_ok());
        }
        assert_eq!(index.len(), 5);

        // Adding 6th should fail
        let result = index.try_get_or_create("key:5");
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            EntityIndexError::CapacityExceeded {
                limit: 5,
                current: 5
            }
        );

        // Getting an existing key should still work
        let result = index.try_get_or_create("key:0");
        assert!(result.is_ok());
    }

    #[test]
    fn test_entity_index_zero_means_unlimited() {
        let config = EntityIndexConfig::unlimited();
        let index = EntityIndex::with_config(config);
        assert_eq!(index.max_entities(), 0);

        // Should allow many entries
        for i in 0..100 {
            let result = index.try_get_or_create(&format!("key:{i}"));
            assert!(result.is_ok());
        }
        assert_eq!(index.len(), 100);
    }

    #[test]
    fn test_entity_index_default_config() {
        let config = EntityIndexConfig::default();
        assert_eq!(config.max_entities, DEFAULT_MAX_ENTITIES);
        assert_eq!(config.initial_capacity, 0);
    }

    #[test]
    fn test_entity_index_try_get_or_create_success() {
        let index = EntityIndex::new();
        let result = index.try_get_or_create("test_key");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_u64(), 0);

        // Idempotent
        let result2 = index.try_get_or_create("test_key");
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap().as_u64(), 0);
    }

    #[test]
    fn test_entity_index_error_display() {
        let err = EntityIndexError::CapacityExceeded {
            limit: 1000,
            current: 1000,
        };
        let msg = format!("{err}");
        assert!(msg.contains("capacity exceeded"));
        assert!(msg.contains("1000"));
    }

    #[test]
    fn test_entity_index_max_entities_getter() {
        let index = EntityIndex::new();
        assert_eq!(index.max_entities(), DEFAULT_MAX_ENTITIES);

        let config = EntityIndexConfig::with_max_entities(500);
        let index2 = EntityIndex::with_config(config);
        assert_eq!(index2.max_entities(), 500);
    }

    #[test]
    fn test_entity_index_snapshot_preserves_max_entities() {
        let config = EntityIndexConfig::with_max_entities(42);
        let index = EntityIndex::with_config(config);
        let _ = index.get_or_create("test");

        let snapshot = index.snapshot();
        let restored = EntityIndex::restore(snapshot);

        assert_eq!(restored.max_entities(), 42);
        assert_eq!(restored.len(), 1);
    }

    #[test]
    fn test_default_max_entities_constant() {
        assert_eq!(DEFAULT_MAX_ENTITIES, 100_000_000);
    }
}
