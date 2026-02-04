// SPDX-License-Identifier: MIT OR Apache-2.0
//! TTL (Time-To-Live) tracking for vault grants.
//!
//! Enables time-limited access grants that automatically expire.
//! Supports persistence for survival across restarts.

#![allow(clippy::missing_panics_doc)]

use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use tensor_store::TensorStore;

use crate::{Result, VaultError};

/// Storage key for persisted TTL grants.
const TTL_STORAGE_KEY: &str = "_vault_ttl_grants";

/// Serializable grant expiration for persistence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PersistedGrant {
    /// Unix timestamp (milliseconds) when this grant expires.
    pub expires_at_ms: i64,
    /// The entity that was granted access.
    pub entity: String,
    /// The secret key.
    pub secret_key: String,
}

impl PersistedGrant {
    fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    fn is_expired(&self) -> bool {
        self.expires_at_ms <= Self::now_ms()
    }
}

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

    /// Convert an Instant to Unix timestamp milliseconds.
    fn instant_to_unix_ms(instant: Instant) -> i64 {
        let now = Instant::now();
        let now_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        // Calculate the difference and apply to current Unix time
        if instant > now {
            let delta = (instant - now).as_millis() as i64;
            now_unix + delta
        } else {
            let delta = (now - instant).as_millis() as i64;
            now_unix - delta
        }
    }

    /// Convert Unix timestamp milliseconds to an Instant.
    fn unix_ms_to_instant(unix_ms: i64) -> Instant {
        let now = Instant::now();
        let now_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        let delta = unix_ms - now_unix;
        if delta >= 0 {
            // Safe: delta is guaranteed non-negative here
            now + Duration::from_millis(delta.unsigned_abs())
        } else {
            // Safe: we're subtracting a duration that was computed from the clock difference
            now.checked_sub(Duration::from_millis(delta.unsigned_abs()))
                .unwrap_or(now)
        }
    }

    /// Persist all grants to the store.
    pub fn persist(&self, store: &TensorStore) -> Result<()> {
        // Collect grants while holding the lock, then release it
        let grants: Vec<PersistedGrant> = {
            let heap = self.heap.lock().unwrap();
            heap.iter()
                .map(|e| PersistedGrant {
                    expires_at_ms: Self::instant_to_unix_ms(e.expires_at),
                    entity: e.entity.clone(),
                    secret_key: e.secret_key.clone(),
                })
                .collect()
        };

        if grants.is_empty() {
            // Remove the storage key if no grants - may not exist, delete is idempotent
            store.delete(TTL_STORAGE_KEY).ok();
            return Ok(());
        }

        let data = serde_json::to_vec(&grants)
            .map_err(|e| VaultError::CryptoError(format!("Failed to serialize TTL grants: {e}")))?;

        let mut tensor = tensor_store::TensorData::new();
        tensor.set(
            "_data",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Bytes(data)),
        );
        store
            .put(TTL_STORAGE_KEY, tensor)
            .map_err(|e| VaultError::CryptoError(format!("Failed to persist TTL grants: {e}")))?;

        Ok(())
    }

    /// Load grants from the store.
    /// Returns a new tracker with the loaded grants.
    pub fn load(store: &TensorStore) -> Result<Self> {
        let tracker = Self::new();

        let Ok(tensor) = store.get(TTL_STORAGE_KEY) else {
            return Ok(tracker); // Key not found, return empty tracker
        };

        let Some(tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Bytes(data))) =
            tensor.get("_data")
        else {
            return Ok(tracker);
        };

        let grants: Vec<PersistedGrant> = serde_json::from_slice(data).map_err(|e| {
            VaultError::CryptoError(format!("Failed to deserialize TTL grants: {e}"))
        })?;

        {
            let mut heap = tracker.heap.lock().unwrap();
            for grant in grants {
                // Skip already expired grants
                if !grant.is_expired() {
                    heap.push(GrantTTLEntry {
                        expires_at: Self::unix_ms_to_instant(grant.expires_at_ms),
                        entity: grant.entity,
                        secret_key: grant.secret_key,
                    });
                }
            }
        }

        Ok(tracker)
    }

    /// Get all grants as serializable format for persistence.
    pub fn to_persisted(&self) -> Vec<PersistedGrant> {
        let heap = self.heap.lock().unwrap();
        heap.iter()
            .map(|e| PersistedGrant {
                expires_at_ms: Self::instant_to_unix_ms(e.expires_at),
                entity: e.entity.clone(),
                secret_key: e.secret_key.clone(),
            })
            .collect()
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
        tracker.add_with_expiration(
            "user:alice",
            "api_key",
            now.checked_sub(Duration::from_secs(1)).unwrap(),
        );

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
        tracker.add_with_expiration(
            "user:later",
            "key",
            now.checked_sub(Duration::from_secs(1)).unwrap(),
        );
        tracker.add_with_expiration(
            "user:earliest",
            "key",
            now.checked_sub(Duration::from_secs(3)).unwrap(),
        );
        tracker.add_with_expiration(
            "user:middle",
            "key",
            now.checked_sub(Duration::from_secs(2)).unwrap(),
        );

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

        tracker.add_with_expiration(
            "user:expired",
            "key",
            now.checked_sub(Duration::from_secs(1)).unwrap(),
        );
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
        tracker.add_with_expiration(
            "user:alice",
            "key1",
            now.checked_sub(Duration::from_secs(1)).unwrap(),
        );
        tracker.add_with_expiration("user:alice", "key2", now + Duration::from_secs(60));

        let expired = tracker.get_expired();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].0, "user:alice");
        assert_eq!(expired[0].1, "key1");

        // key2 should still be tracked
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_persist_and_load() {
        use std::sync::Arc;

        let store = Arc::new(TensorStore::new());

        // Create tracker with grants
        let tracker = GrantTTLTracker::new();
        tracker.add("user:alice", "key1", Duration::from_secs(3600));
        tracker.add("user:bob", "key2", Duration::from_secs(7200));

        assert_eq!(tracker.len(), 2);

        // Persist
        tracker.persist(&store).unwrap();

        // Load into new tracker
        let loaded = GrantTTLTracker::load(&store).unwrap();

        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_persist_skips_expired() {
        use std::sync::Arc;

        let store = Arc::new(TensorStore::new());

        let tracker = GrantTTLTracker::new();
        let now = Instant::now();

        // Add expired and valid grants
        tracker.add_with_expiration(
            "user:expired",
            "key",
            now.checked_sub(Duration::from_secs(1)).unwrap(),
        );
        tracker.add_with_expiration("user:valid", "key", now + Duration::from_secs(3600));

        assert_eq!(tracker.len(), 2);

        // Persist (includes both)
        tracker.persist(&store).unwrap();

        // Load - should skip expired
        let loaded = GrantTTLTracker::load(&store).unwrap();
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn test_load_empty_store() {
        use std::sync::Arc;

        let store = Arc::new(TensorStore::new());
        let loaded = GrantTTLTracker::load(&store).unwrap();

        assert!(loaded.is_empty());
    }

    #[test]
    fn test_to_persisted() {
        let tracker = GrantTTLTracker::new();
        tracker.add("user:alice", "key1", Duration::from_secs(60));

        let persisted = tracker.to_persisted();
        assert_eq!(persisted.len(), 1);
        assert_eq!(persisted[0].entity, "user:alice");
        assert_eq!(persisted[0].secret_key, "key1");
        assert!(persisted[0].expires_at_ms > 0);
    }
}
