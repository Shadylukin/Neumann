//! TTL (Time-To-Live) tracking for vault grants.
//!
//! Enables time-limited access grants that automatically expire.

#![allow(clippy::missing_panics_doc)]

use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    sync::Mutex,
    time::{Duration, Instant},
};

/// An entry in the grant TTL tracker.
#[derive(Debug, Clone)]
struct GrantTTLEntry {
    /// When this grant expires.
    expires_at: Instant,
    /// The entity that was granted access.
    entity: String,
    /// The secret key.
    secret_key: String,
}

impl PartialEq for GrantTTLEntry {
    fn eq(&self, other: &Self) -> bool {
        self.expires_at == other.expires_at
            && self.entity == other.entity
            && self.secret_key == other.secret_key
    }
}

impl Eq for GrantTTLEntry {}

impl PartialOrd for GrantTTLEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GrantTTLEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (earliest expiration first)
        other.expires_at.cmp(&self.expires_at)
    }
}

/// Tracks grant expiration times for automatic revocation.
pub struct GrantTTLTracker {
    /// Priority queue of expiration times (min-heap).
    heap: Mutex<BinaryHeap<GrantTTLEntry>>,
}

impl GrantTTLTracker {
    /// Create a new grant TTL tracker.
    pub fn new() -> Self {
        Self {
            heap: Mutex::new(BinaryHeap::new()),
        }
    }

    /// Add a grant to the tracker with the specified TTL.
    pub fn add(&self, entity: &str, secret_key: &str, ttl: Duration) {
        let entry = GrantTTLEntry {
            expires_at: Instant::now() + ttl,
            entity: entity.to_string(),
            secret_key: secret_key.to_string(),
        };
        self.heap.lock().unwrap().push(entry);
    }

    /// Add a grant with an explicit expiration time.
    pub fn add_with_expiration(&self, entity: &str, secret_key: &str, expires_at: Instant) {
        let entry = GrantTTLEntry {
            expires_at,
            entity: entity.to_string(),
            secret_key: secret_key.to_string(),
        };
        self.heap.lock().unwrap().push(entry);
    }

    /// Get all expired grants.
    ///
    /// Returns a list of (entity, secret_key) pairs for grants that have expired.
    /// These entries are removed from the tracker.
    pub fn get_expired(&self) -> Vec<(String, String)> {
        let now = Instant::now();
        let mut expired = Vec::new();
        let mut heap = self.heap.lock().unwrap();

        while let Some(entry) = heap.peek() {
            if entry.expires_at <= now {
                if let Some(entry) = heap.pop() {
                    expired.push((entry.entity, entry.secret_key));
                }
            } else {
                break;
            }
        }

        expired
    }

    /// Get the number of grants being tracked.
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

    /// Check if a specific grant is expired.
    ///
    /// Note: This does a linear search and is O(n). For bulk checks,
    /// use `get_expired()` instead.
    #[allow(clippy::significant_drop_tightening)]
    pub fn is_expired(&self, entity: &str, secret_key: &str) -> bool {
        let now = Instant::now();
        let heap = self.heap.lock().unwrap();

        for entry in heap.iter() {
            if entry.entity == entity && entry.secret_key == secret_key {
                return entry.expires_at <= now;
            }
        }

        // Not found in tracker = not tracked, so not expired by TTL
        false
    }

    /// Remove a specific grant from tracking (e.g., when manually revoked).
    ///
    /// Returns true if the grant was found and removed.
    pub fn remove(&self, entity: &str, secret_key: &str) -> bool {
        let mut heap = self.heap.lock().unwrap();
        let original_len = heap.len();

        let entries: Vec<GrantTTLEntry> = heap
            .drain()
            .filter(|e| !(e.entity == entity && e.secret_key == secret_key))
            .collect();

        let removed = entries.len() < original_len;
        for entry in entries {
            heap.push(entry);
        }

        removed
    }
}

impl Default for GrantTTLTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_expired() {
        let tracker = GrantTTLTracker::new();
        let now = Instant::now();

        // Add an already-expired entry
        tracker.add_with_expiration("user:alice", "api_key", now - Duration::from_secs(1));

        // Add a not-yet-expired entry
        tracker.add_with_expiration("user:bob", "api_key", now + Duration::from_secs(60));

        let expired = tracker.get_expired();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].0, "user:alice");
        assert_eq!(expired[0].1, "api_key");

        // Should still have one entry
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = GrantTTLTracker::new();
        assert!(tracker.is_empty());

        let expired = tracker.get_expired();
        assert!(expired.is_empty());
    }

    #[test]
    fn test_ordering() {
        let tracker = GrantTTLTracker::new();
        let now = Instant::now();

        // Add entries in non-chronological order
        tracker.add_with_expiration("user:later", "key", now - Duration::from_secs(1));
        tracker.add_with_expiration("user:earliest", "key", now - Duration::from_secs(3));
        tracker.add_with_expiration("user:middle", "key", now - Duration::from_secs(2));

        let expired = tracker.get_expired();

        // Should be ordered by expiration time (earliest first)
        assert_eq!(expired.len(), 3);
        assert_eq!(expired[0].0, "user:earliest");
        assert_eq!(expired[1].0, "user:middle");
        assert_eq!(expired[2].0, "user:later");
    }

    #[test]
    fn test_next_expiration() {
        let tracker = GrantTTLTracker::new();
        assert!(tracker.next_expiration().is_none());

        let future = Instant::now() + Duration::from_secs(60);
        tracker.add_with_expiration("user:alice", "key", future);

        let next = tracker.next_expiration().unwrap();
        assert_eq!(next, future);
    }

    #[test]
    fn test_clear() {
        let tracker = GrantTTLTracker::new();
        let now = Instant::now();

        tracker.add_with_expiration("user:a", "k1", now + Duration::from_secs(60));
        tracker.add_with_expiration("user:b", "k2", now + Duration::from_secs(120));

        assert_eq!(tracker.len(), 2);
        tracker.clear();
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_add_with_duration() {
        let tracker = GrantTTLTracker::new();

        tracker.add("user:alice", "api_key", Duration::from_secs(3600));

        assert_eq!(tracker.len(), 1);

        // Should not be expired yet
        let expired = tracker.get_expired();
        assert!(expired.is_empty());
    }

    #[test]
    fn test_is_expired() {
        let tracker = GrantTTLTracker::new();
        let now = Instant::now();

        tracker.add_with_expiration("user:expired", "key", now - Duration::from_secs(1));
        tracker.add_with_expiration("user:valid", "key", now + Duration::from_secs(60));

        assert!(tracker.is_expired("user:expired", "key"));
        assert!(!tracker.is_expired("user:valid", "key"));
        assert!(!tracker.is_expired("user:unknown", "key")); // Not tracked = not expired
    }

    #[test]
    fn test_remove() {
        let tracker = GrantTTLTracker::new();
        let future = Instant::now() + Duration::from_secs(60);

        tracker.add_with_expiration("user:alice", "key1", future);
        tracker.add_with_expiration("user:alice", "key2", future);
        tracker.add_with_expiration("user:bob", "key1", future);

        assert_eq!(tracker.len(), 3);

        // Remove one grant
        assert!(tracker.remove("user:alice", "key1"));
        assert_eq!(tracker.len(), 2);

        // Try to remove again
        assert!(!tracker.remove("user:alice", "key1"));
        assert_eq!(tracker.len(), 2);

        // Remove non-existent
        assert!(!tracker.remove("user:carol", "key1"));
        assert_eq!(tracker.len(), 2);
    }

    #[test]
    fn test_multiple_grants_same_entity() {
        let tracker = GrantTTLTracker::new();
        let now = Instant::now();

        // Same entity, different secrets, different expiration
        tracker.add_with_expiration("user:alice", "key1", now - Duration::from_secs(1));
        tracker.add_with_expiration("user:alice", "key2", now + Duration::from_secs(60));

        let expired = tracker.get_expired();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].0, "user:alice");
        assert_eq!(expired[0].1, "key1");

        // key2 should still be tracked
        assert_eq!(tracker.len(), 1);
    }
}
