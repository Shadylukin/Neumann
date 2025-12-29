//! BTreeMap-based metadata storage slab.
//!
//! MetadataSlab provides key-value storage for metadata, vault secrets, and
//! miscellaneous data using a BTreeMap. Unlike hash tables, BTreeMaps split
//! nodes on insertion rather than resizing the entire structure, avoiding
//! the throughput stalls associated with DashMap resizing.
//!
//! # Design Philosophy
//!
//! This slab is optimized for:
//! - Low-volume, high-value data (secrets, configuration)
//! - Prefix-based scanning (via BTreeMap's ordered iteration)
//! - Stable performance without resize stalls
//! - O(log n) operations for all access patterns

use crate::TensorData;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// BTreeMap-based metadata storage with prefix scanning support.
///
/// # Thread Safety
///
/// Uses `parking_lot::RwLock` for concurrent access without lock poisoning.
/// Multiple readers can access concurrently; writers have exclusive access.
///
/// # Performance
///
/// - `get`: O(log n)
/// - `set`: O(log n)
/// - `delete`: O(log n)
/// - `scan`: O(k + log n) where k is the number of matching entries
/// - `len`: O(1)
///
/// BTreeMaps never resize; they split nodes on insertion, which is O(log n).
pub struct MetadataSlab {
    /// The underlying BTreeMap storage.
    data: RwLock<BTreeMap<String, TensorData>>,

    /// Approximate total bytes stored (for memory tracking).
    total_bytes: AtomicUsize,

    /// Number of entries (O(1) len).
    len: AtomicUsize,
}

impl MetadataSlab {
    /// Create a new empty MetadataSlab.
    pub fn new() -> Self {
        Self {
            data: RwLock::new(BTreeMap::new()),
            total_bytes: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
        }
    }

    /// Get a value by key.
    ///
    /// Returns a cloned TensorData to avoid holding the lock.
    pub fn get(&self, key: &str) -> Option<TensorData> {
        self.data.read().get(key).cloned()
    }

    /// Check if a key exists.
    #[inline]
    pub fn contains(&self, key: &str) -> bool {
        self.data.read().contains_key(key)
    }

    /// Set a value for a key.
    ///
    /// If the key already exists, the old value is replaced.
    pub fn set(&self, key: &str, value: TensorData) {
        let value_bytes = estimate_tensor_data_bytes(&value);
        let mut data = self.data.write();

        // Subtract old value bytes if replacing
        if let Some(old) = data.get(key) {
            let old_bytes = estimate_tensor_data_bytes(old);
            self.total_bytes.fetch_sub(old_bytes, Ordering::Relaxed);
        } else {
            // New key - increment count
            self.len.fetch_add(1, Ordering::Relaxed);
        }

        data.insert(key.to_string(), value);
        self.total_bytes
            .fetch_add(key.len() + value_bytes, Ordering::Relaxed);
    }

    /// Delete a key and return the old value.
    pub fn delete(&self, key: &str) -> Option<TensorData> {
        let mut data = self.data.write();
        if let Some(old) = data.remove(key) {
            let old_bytes = key.len() + estimate_tensor_data_bytes(&old);
            self.total_bytes.fetch_sub(old_bytes, Ordering::Relaxed);
            self.len.fetch_sub(1, Ordering::Relaxed);
            Some(old)
        } else {
            None
        }
    }

    /// Scan for all keys matching a prefix.
    ///
    /// Returns pairs of (key, TensorData) for all matching entries.
    /// Uses BTreeMap's range for efficient prefix scanning.
    pub fn scan(&self, prefix: &str) -> Vec<(String, TensorData)> {
        let data = self.data.read();

        // BTreeMap range can efficiently find prefix matches
        let end = next_prefix(prefix);

        match end {
            Some(end_key) => data
                .range(prefix.to_string()..end_key)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            None => {
                // Prefix is all 0xFF bytes or empty, scan to end
                data.range(prefix.to_string()..)
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            },
        }
    }

    /// Count keys matching a prefix.
    pub fn scan_count(&self, prefix: &str) -> usize {
        if prefix.is_empty() {
            return self.len();
        }

        let data = self.data.read();
        let end = next_prefix(prefix);

        match end {
            Some(end_key) => data.range(prefix.to_string()..end_key).count(),
            None => data.range(prefix.to_string()..).count(),
        }
    }

    /// Get all keys in the slab.
    pub fn keys(&self) -> Vec<String> {
        self.data.read().keys().cloned().collect()
    }

    /// Get the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Check if the slab is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the approximate total bytes stored.
    #[inline]
    pub fn bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Relaxed)
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.data.write().clear();
        self.total_bytes.store(0, Ordering::Relaxed);
        self.len.store(0, Ordering::Relaxed);
    }

    /// Get serializable state for snapshots.
    pub fn snapshot(&self) -> MetadataSlabSnapshot {
        MetadataSlabSnapshot {
            data: self.data.read().clone(),
        }
    }

    /// Restore from a snapshot.
    pub fn restore(snapshot: MetadataSlabSnapshot) -> Self {
        let len = snapshot.data.len();
        let total_bytes: usize = snapshot
            .data
            .iter()
            .map(|(k, v)| k.len() + estimate_tensor_data_bytes(v))
            .sum();

        Self {
            data: RwLock::new(snapshot.data),
            total_bytes: AtomicUsize::new(total_bytes),
            len: AtomicUsize::new(len),
        }
    }

    /// Iterate over all entries.
    ///
    /// Note: This clones all entries. For large slabs, consider using scan with a prefix.
    pub fn iter(&self) -> Vec<(String, TensorData)> {
        self.data
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// Update a value in-place using a closure.
    ///
    /// If the key doesn't exist, the closure is not called.
    /// Returns true if the update was applied.
    pub fn update<F>(&self, key: &str, f: F) -> bool
    where
        F: FnOnce(&mut TensorData),
    {
        let mut data = self.data.write();
        if let Some(value) = data.get_mut(key) {
            let old_bytes = estimate_tensor_data_bytes(value);
            f(value);
            let new_bytes = estimate_tensor_data_bytes(value);

            // Update byte tracking
            if new_bytes > old_bytes {
                self.total_bytes
                    .fetch_add(new_bytes - old_bytes, Ordering::Relaxed);
            } else {
                self.total_bytes
                    .fetch_sub(old_bytes - new_bytes, Ordering::Relaxed);
            }
            true
        } else {
            false
        }
    }

    /// Get or insert a value.
    ///
    /// If the key exists, returns a clone of the existing value.
    /// Otherwise, inserts the provided value and returns a clone of it.
    pub fn get_or_insert(&self, key: &str, value: TensorData) -> TensorData {
        // Fast path: check if exists
        {
            let data = self.data.read();
            if let Some(existing) = data.get(key) {
                return existing.clone();
            }
        }

        // Slow path: insert
        let mut data = self.data.write();
        // Double-check after acquiring write lock
        if let Some(existing) = data.get(key) {
            return existing.clone();
        }

        let value_bytes = key.len() + estimate_tensor_data_bytes(&value);
        let result = value.clone();
        data.insert(key.to_string(), value);
        self.total_bytes.fetch_add(value_bytes, Ordering::Relaxed);
        self.len.fetch_add(1, Ordering::Relaxed);
        result
    }
}

impl Default for MetadataSlab {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable snapshot of MetadataSlab state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataSlabSnapshot {
    pub(crate) data: BTreeMap<String, TensorData>,
}

impl MetadataSlabSnapshot {
    /// Iterate over all entries in the snapshot.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorData)> {
        self.data.iter()
    }
}

/// Compute the exclusive end key for prefix scanning.
///
/// Returns the smallest string greater than all strings starting with the prefix.
fn next_prefix(prefix: &str) -> Option<String> {
    if prefix.is_empty() {
        // Empty prefix matches everything, no upper bound needed
        return None;
    }

    let mut bytes = prefix.as_bytes().to_vec();

    // Find the last byte that can be incremented
    while let Some(last) = bytes.pop() {
        if last < 0xFF {
            bytes.push(last + 1);
            return String::from_utf8(bytes).ok();
        }
        // Last byte was 0xFF, continue to previous byte
    }

    // All bytes were 0xFF, no upper bound
    None
}

/// Estimate the memory usage of a TensorData in bytes.
fn estimate_tensor_data_bytes(data: &TensorData) -> usize {
    use crate::{ScalarValue, TensorValue};

    let mut bytes = 0;
    for (key, value) in data.fields_iter() {
        bytes += key.len();
        bytes += match value {
            TensorValue::Scalar(s) => match s {
                ScalarValue::Null => 0,
                ScalarValue::Bool(_) => 1,
                ScalarValue::Int(_) => 8,
                ScalarValue::Float(_) => 8,
                ScalarValue::String(s) => s.len(),
                ScalarValue::Bytes(b) => b.len(),
            },
            TensorValue::Vector(v) => v.len() * 4,
            TensorValue::Sparse(s) => s.memory_bytes(),
            TensorValue::Pointer(p) => p.len(),
            TensorValue::Pointers(ps) => ps.iter().map(|p| p.len()).sum(),
        };
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ScalarValue, TensorValue};
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    fn make_tensor_data(id: i64) -> TensorData {
        let mut data = TensorData::new();
        data.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
        data
    }

    #[test]
    fn test_new_empty() {
        let slab = MetadataSlab::new();
        assert_eq!(slab.len(), 0);
        assert!(slab.is_empty());
        assert_eq!(slab.bytes(), 0);
    }

    #[test]
    fn test_set_get() {
        let slab = MetadataSlab::new();
        slab.set("key1", make_tensor_data(1));

        let result = slab.get("key1");
        assert!(result.is_some());

        let data = result.unwrap();
        let id = data.get("id").unwrap();
        assert!(matches!(id, TensorValue::Scalar(ScalarValue::Int(1))));
    }

    #[test]
    fn test_get_nonexistent() {
        let slab = MetadataSlab::new();
        assert!(slab.get("nonexistent").is_none());
    }

    #[test]
    fn test_contains() {
        let slab = MetadataSlab::new();
        slab.set("key1", make_tensor_data(1));

        assert!(slab.contains("key1"));
        assert!(!slab.contains("key2"));
    }

    #[test]
    fn test_delete() {
        let slab = MetadataSlab::new();
        slab.set("key1", make_tensor_data(1));
        assert_eq!(slab.len(), 1);

        let deleted = slab.delete("key1");
        assert!(deleted.is_some());
        assert_eq!(slab.len(), 0);
        assert!(slab.get("key1").is_none());
    }

    #[test]
    fn test_delete_nonexistent() {
        let slab = MetadataSlab::new();
        let deleted = slab.delete("nonexistent");
        assert!(deleted.is_none());
    }

    #[test]
    fn test_replace() {
        let slab = MetadataSlab::new();
        slab.set("key1", make_tensor_data(1));
        slab.set("key1", make_tensor_data(2));

        assert_eq!(slab.len(), 1);

        let data = slab.get("key1").unwrap();
        let id = data.get("id").unwrap();
        assert!(matches!(id, TensorValue::Scalar(ScalarValue::Int(2))));
    }

    #[test]
    fn test_scan_prefix() {
        let slab = MetadataSlab::new();
        slab.set("user:1", make_tensor_data(1));
        slab.set("user:2", make_tensor_data(2));
        slab.set("user:3", make_tensor_data(3));
        slab.set("post:1", make_tensor_data(100));

        let users = slab.scan("user:");
        assert_eq!(users.len(), 3);

        let keys: Vec<&String> = users.iter().map(|(k, _)| k).collect();
        assert!(keys.contains(&&"user:1".to_string()));
        assert!(keys.contains(&&"user:2".to_string()));
        assert!(keys.contains(&&"user:3".to_string()));
    }

    #[test]
    fn test_scan_prefix_ordered() {
        let slab = MetadataSlab::new();
        slab.set("user:3", make_tensor_data(3));
        slab.set("user:1", make_tensor_data(1));
        slab.set("user:2", make_tensor_data(2));

        let users = slab.scan("user:");

        // BTreeMap maintains order
        let keys: Vec<String> = users.into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec!["user:1", "user:2", "user:3"]);
    }

    #[test]
    fn test_scan_prefix_empty() {
        let slab = MetadataSlab::new();
        slab.set("user:1", make_tensor_data(1));

        let posts = slab.scan("post:");
        assert!(posts.is_empty());
    }

    #[test]
    fn test_scan_count() {
        let slab = MetadataSlab::new();
        slab.set("user:1", make_tensor_data(1));
        slab.set("user:2", make_tensor_data(2));
        slab.set("post:1", make_tensor_data(100));

        assert_eq!(slab.scan_count("user:"), 2);
        assert_eq!(slab.scan_count("post:"), 1);
        assert_eq!(slab.scan_count("comment:"), 0);
    }

    #[test]
    fn test_keys() {
        let slab = MetadataSlab::new();
        slab.set("a", make_tensor_data(1));
        slab.set("b", make_tensor_data(2));
        slab.set("c", make_tensor_data(3));

        let mut keys = slab.keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_clear() {
        let slab = MetadataSlab::new();
        slab.set("key1", make_tensor_data(1));
        slab.set("key2", make_tensor_data(2));
        assert_eq!(slab.len(), 2);

        slab.clear();
        assert_eq!(slab.len(), 0);
        assert!(slab.is_empty());
        assert_eq!(slab.bytes(), 0);
    }

    #[test]
    fn test_snapshot_restore() {
        let slab = MetadataSlab::new();
        slab.set("key1", make_tensor_data(1));
        slab.set("key2", make_tensor_data(2));

        let snapshot = slab.snapshot();
        let restored = MetadataSlab::restore(snapshot);

        assert_eq!(restored.len(), 2);
        assert!(restored.contains("key1"));
        assert!(restored.contains("key2"));
    }

    #[test]
    fn test_update() {
        let slab = MetadataSlab::new();
        slab.set("key1", make_tensor_data(1));

        let updated = slab.update("key1", |data| {
            data.set("id", TensorValue::Scalar(ScalarValue::Int(999)));
        });
        assert!(updated);

        let data = slab.get("key1").unwrap();
        let id = data.get("id").unwrap();
        assert!(matches!(id, TensorValue::Scalar(ScalarValue::Int(999))));
    }

    #[test]
    fn test_update_nonexistent() {
        let slab = MetadataSlab::new();
        let updated = slab.update("nonexistent", |_| {});
        assert!(!updated);
    }

    #[test]
    fn test_get_or_insert() {
        let slab = MetadataSlab::new();

        // Insert new
        let result = slab.get_or_insert("key1", make_tensor_data(1));
        let id = result.get("id").unwrap();
        assert!(matches!(id, TensorValue::Scalar(ScalarValue::Int(1))));

        // Get existing (should not replace)
        let result = slab.get_or_insert("key1", make_tensor_data(999));
        let id = result.get("id").unwrap();
        assert!(matches!(id, TensorValue::Scalar(ScalarValue::Int(1))));

        assert_eq!(slab.len(), 1);
    }

    #[test]
    fn test_iter() {
        let slab = MetadataSlab::new();
        slab.set("a", make_tensor_data(1));
        slab.set("b", make_tensor_data(2));

        let entries = slab.iter();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_bytes_tracking() {
        let slab = MetadataSlab::new();
        let initial_bytes = slab.bytes();
        assert_eq!(initial_bytes, 0);

        slab.set("key1", make_tensor_data(1));
        let after_insert = slab.bytes();
        assert!(after_insert > 0);

        slab.delete("key1");
        let after_delete = slab.bytes();
        assert_eq!(after_delete, 0);
    }

    #[test]
    fn test_concurrent_reads_writes() {
        let slab = Arc::new(MetadataSlab::new());
        let mut handles = vec![];

        // Writer threads
        for t in 0..4 {
            let s = Arc::clone(&slab);
            handles.push(thread::spawn(move || {
                for i in 0..500 {
                    s.set(&format!("thread{}:key{}", t, i), make_tensor_data(i as i64));
                }
            }));
        }

        // Reader threads
        for _ in 0..4 {
            let s = Arc::clone(&slab);
            handles.push(thread::spawn(move || {
                for _ in 0..500 {
                    let _ = s.get("thread0:key0");
                    let _ = s.scan("thread1:");
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(slab.len(), 2000);
    }

    #[test]
    fn test_no_resize_stall() {
        let slab = MetadataSlab::new();
        let count = 50_000;

        let start = Instant::now();
        let mut max_op_time = std::time::Duration::ZERO;

        for i in 0..count {
            let op_start = Instant::now();
            slab.set(&format!("entity:{}", i), make_tensor_data(i as i64));
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

        // Verify throughput is reasonable (lower threshold for coverage builds)
        let ops_per_sec = count as f64 / total_time.as_secs_f64();
        assert!(
            ops_per_sec > 10_000.0,
            "Throughput {:.0} ops/sec too low",
            ops_per_sec
        );
    }

    #[test]
    fn test_next_prefix() {
        assert_eq!(next_prefix("abc"), Some("abd".to_string()));
        assert_eq!(next_prefix("ab"), Some("ac".to_string()));
        assert_eq!(next_prefix("a"), Some("b".to_string()));
        // Empty prefix matches everything, so no upper bound
        assert_eq!(next_prefix(""), None);
    }

    #[test]
    fn test_prefix_boundary() {
        let slab = MetadataSlab::new();
        slab.set("user:1", make_tensor_data(1));
        slab.set("user:2", make_tensor_data(2));
        slab.set("user:", make_tensor_data(0)); // Edge case: key equals prefix
        slab.set("userz", make_tensor_data(999)); // Should NOT match "user:"

        let users = slab.scan("user:");
        assert_eq!(users.len(), 3);
    }

    #[test]
    fn test_large_values() {
        let slab = MetadataSlab::new();

        let mut data = TensorData::new();
        let large_string = "x".repeat(10_000);
        data.set(
            "content",
            TensorValue::Scalar(ScalarValue::String(large_string)),
        );

        slab.set("large", data);
        assert!(slab.bytes() > 10_000);

        let retrieved = slab.get("large").unwrap();
        let content = retrieved.get("content").unwrap();
        if let TensorValue::Scalar(ScalarValue::String(s)) = content {
            assert_eq!(s.len(), 10_000);
        } else {
            panic!("Expected string value");
        }
    }

    #[test]
    fn test_various_tensor_value_types() {
        use crate::sparse_vector::SparseVector;

        let slab = MetadataSlab::new();

        // Test with all TensorValue types for bytes estimation
        let mut data = TensorData::new();
        data.set("null", TensorValue::Scalar(ScalarValue::Null));
        data.set("bool", TensorValue::Scalar(ScalarValue::Bool(true)));
        data.set("int", TensorValue::Scalar(ScalarValue::Int(42)));
        data.set("float", TensorValue::Scalar(ScalarValue::Float(3.14)));
        data.set(
            "string",
            TensorValue::Scalar(ScalarValue::String("hello".to_string())),
        );
        data.set(
            "bytes",
            TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3])),
        );
        data.set("vector", TensorValue::Vector(vec![1.0, 2.0, 3.0]));
        data.set(
            "sparse",
            TensorValue::Sparse(SparseVector::from_dense(&[0.0, 1.0, 0.0, 2.0])),
        );
        data.set("pointer", TensorValue::Pointer("ref:1".to_string()));
        data.set(
            "pointers",
            TensorValue::Pointers(vec!["ref:1".to_string(), "ref:2".to_string()]),
        );

        slab.set("all_types", data);
        assert!(slab.bytes() > 0);
        assert!(slab.contains("all_types"));
    }

    #[test]
    fn test_scan_empty_prefix() {
        let slab = MetadataSlab::new();
        slab.set("a", make_tensor_data(1));
        slab.set("b", make_tensor_data(2));
        slab.set("c", make_tensor_data(3));

        // Empty prefix should return all entries
        let all = slab.scan("");
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_default_trait() {
        let slab = MetadataSlab::default();
        assert!(slab.is_empty());
    }

    #[test]
    fn test_update_increases_bytes() {
        let slab = MetadataSlab::new();

        let mut data = TensorData::new();
        data.set(
            "content",
            TensorValue::Scalar(ScalarValue::String("short".to_string())),
        );
        slab.set("key", data);

        let bytes_before = slab.bytes();

        slab.update("key", |d| {
            d.set(
                "content",
                TensorValue::Scalar(ScalarValue::String("much longer string".to_string())),
            );
        });

        let bytes_after = slab.bytes();
        assert!(bytes_after > bytes_before);
    }
}
