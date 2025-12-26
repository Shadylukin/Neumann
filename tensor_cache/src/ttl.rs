//! TTL (Time-To-Live) tracking for cache entries.

#![allow(dead_code)]

use crate::stats::CacheLayer;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Mutex;
use std::time::Instant;

/// An entry in the TTL tracker.
#[derive(Debug, Clone)]
struct TTLEntry {
    /// When this entry expires.
    expires_at: Instant,
    /// The cache key.
    key: String,
    /// Which cache layer this entry belongs to.
    layer: CacheLayer,
}

impl PartialEq for TTLEntry {
    fn eq(&self, other: &Self) -> bool {
        self.expires_at == other.expires_at && self.key == other.key
    }
}

impl Eq for TTLEntry {}

impl PartialOrd for TTLEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TTLEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (earliest expiration first)
        other.expires_at.cmp(&self.expires_at)
    }
}

/// Tracks entry expiration times for efficient cleanup.
pub struct TTLTracker {
    /// Priority queue of expiration times (min-heap).
    heap: Mutex<BinaryHeap<TTLEntry>>,
}

impl TTLTracker {
    /// Create a new TTL tracker.
    pub fn new() -> Self {
        Self {
            heap: Mutex::new(BinaryHeap::new()),
        }
    }

    /// Add an entry to the tracker.
    pub fn add(&self, key: String, expires_at: Instant, layer: CacheLayer) {
        let entry = TTLEntry {
            expires_at,
            key,
            layer,
        };
        self.heap.lock().unwrap().push(entry);
    }

    /// Get all expired entries.
    ///
    /// Returns a list of (key, layer) pairs for entries that have expired.
    /// These entries are removed from the tracker.
    pub fn get_expired(&self) -> Vec<(String, CacheLayer)> {
        let now = Instant::now();
        let mut expired = Vec::new();
        let mut heap = self.heap.lock().unwrap();

        while let Some(entry) = heap.peek() {
            if entry.expires_at <= now {
                if let Some(entry) = heap.pop() {
                    expired.push((entry.key, entry.layer));
                }
            } else {
                break;
            }
        }

        expired
    }

    /// Get the number of entries being tracked.
    pub fn len(&self) -> usize {
        self.heap.lock().unwrap().len()
    }

    /// Check if the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.lock().unwrap().is_empty()
    }

    /// Clear all entries.
    pub fn clear(&self) {
        self.heap.lock().unwrap().clear();
    }

    /// Get the next expiration time, if any.
    pub fn next_expiration(&self) -> Option<Instant> {
        self.heap.lock().unwrap().peek().map(|e| e.expires_at)
    }
}

impl Default for TTLTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_add_and_get_expired() {
        let tracker = TTLTracker::new();
        let now = Instant::now();

        // Add an already-expired entry
        tracker.add(
            "key1".into(),
            now - Duration::from_secs(1),
            CacheLayer::Exact,
        );

        // Add a not-yet-expired entry
        tracker.add(
            "key2".into(),
            now + Duration::from_secs(60),
            CacheLayer::Semantic,
        );

        let expired = tracker.get_expired();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].0, "key1");
        assert_eq!(expired[0].1, CacheLayer::Exact);

        // Should still have one entry
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = TTLTracker::new();
        assert!(tracker.is_empty());

        let expired = tracker.get_expired();
        assert!(expired.is_empty());
    }

    #[test]
    fn test_ordering() {
        let tracker = TTLTracker::new();
        let now = Instant::now();

        // Add entries in non-chronological order
        tracker.add(
            "later".into(),
            now - Duration::from_secs(1),
            CacheLayer::Exact,
        );
        tracker.add(
            "earliest".into(),
            now - Duration::from_secs(3),
            CacheLayer::Semantic,
        );
        tracker.add(
            "middle".into(),
            now - Duration::from_secs(2),
            CacheLayer::Embedding,
        );

        let expired = tracker.get_expired();

        // Should be ordered by expiration time (earliest first)
        assert_eq!(expired.len(), 3);
        assert_eq!(expired[0].0, "earliest");
        assert_eq!(expired[1].0, "middle");
        assert_eq!(expired[2].0, "later");
    }

    #[test]
    fn test_next_expiration() {
        let tracker = TTLTracker::new();
        assert!(tracker.next_expiration().is_none());

        let future = Instant::now() + Duration::from_secs(60);
        tracker.add("key".into(), future, CacheLayer::Exact);

        let next = tracker.next_expiration().unwrap();
        assert_eq!(next, future);
    }

    #[test]
    fn test_clear() {
        let tracker = TTLTracker::new();
        let now = Instant::now();

        tracker.add(
            "k1".into(),
            now + Duration::from_secs(60),
            CacheLayer::Exact,
        );
        tracker.add(
            "k2".into(),
            now + Duration::from_secs(120),
            CacheLayer::Semantic,
        );

        assert_eq!(tracker.len(), 2);
        tracker.clear();
        assert!(tracker.is_empty());
    }
}
