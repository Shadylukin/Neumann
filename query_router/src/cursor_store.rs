// SPDX-License-Identifier: MIT OR Apache-2.0
//! Thread-safe cursor storage with TTL-based expiration.
//!
//! This module provides a cursor store for managing pagination cursors
//! with automatic expiration and LRU eviction when at capacity.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;

use crate::cursor::{CursorError, CursorId, CursorState};

/// Configuration for the cursor store.
#[derive(Debug, Clone)]
pub struct CursorStoreConfig {
    /// Maximum number of active cursors.
    pub max_cursors: usize,
    /// Default TTL for new cursors.
    pub default_ttl: Duration,
    /// Maximum allowed TTL for cursors.
    pub max_ttl: Duration,
    /// Interval between cleanup runs.
    pub cleanup_interval: Duration,
}

impl Default for CursorStoreConfig {
    fn default() -> Self {
        Self {
            max_cursors: 10_000,
            default_ttl: Duration::from_secs(300), // 5 minutes
            max_ttl: Duration::from_secs(1800),    // 30 minutes
            cleanup_interval: Duration::from_secs(30),
        }
    }
}

impl CursorStoreConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum cursors.
    #[must_use]
    pub const fn with_max_cursors(mut self, max: usize) -> Self {
        self.max_cursors = max;
        self
    }

    /// Set default TTL.
    #[must_use]
    pub const fn with_default_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = ttl;
        self
    }

    /// Set maximum TTL.
    #[must_use]
    pub const fn with_max_ttl(mut self, ttl: Duration) -> Self {
        self.max_ttl = ttl;
        self
    }

    /// Set cleanup interval.
    #[must_use]
    pub const fn with_cleanup_interval(mut self, interval: Duration) -> Self {
        self.cleanup_interval = interval;
        self
    }
}

/// Internal entry in the cursor store.
#[derive(Debug, Clone)]
struct CursorEntry {
    state: CursorState,
    /// Unix timestamp of last access for LRU eviction.
    last_access: i64,
}

/// Thread-safe cursor storage with TTL cleanup.
#[derive(Debug)]
pub struct CursorStore {
    cursors: DashMap<CursorId, CursorEntry>,
    config: CursorStoreConfig,
    shutdown: Arc<AtomicBool>,
}

impl CursorStore {
    /// Create a new cursor store with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(CursorStoreConfig::default())
    }

    /// Create a new cursor store with custom configuration.
    #[must_use]
    pub fn with_config(config: CursorStoreConfig) -> Self {
        Self {
            cursors: DashMap::new(),
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get the store configuration.
    #[must_use]
    pub fn config(&self) -> &CursorStoreConfig {
        &self.config
    }

    /// Store a new cursor.
    ///
    /// If at capacity, evicts the least recently used cursor.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot accept new cursors.
    pub fn insert(&self, cursor: CursorState) -> Result<(), CursorError> {
        // Check capacity and evict if necessary
        if self.cursors.len() >= self.config.max_cursors {
            self.evict_lru();
        }

        // Still at capacity after eviction? Check again.
        if self.cursors.len() >= self.config.max_cursors {
            return Err(CursorError::CapacityExceeded);
        }

        let id = cursor.id.clone();
        let entry = CursorEntry {
            last_access: current_timestamp(),
            state: cursor,
        };
        self.cursors.insert(id, entry);
        Ok(())
    }

    /// Get a cursor by ID, updating its last access time.
    ///
    /// # Errors
    ///
    /// Returns an error if the cursor is not found or has expired.
    pub fn get(&self, id: &str) -> Result<CursorState, CursorError> {
        let mut entry = self
            .cursors
            .get_mut(id)
            .ok_or_else(|| CursorError::NotFound(id.to_string()))?;

        if entry.state.is_expired() {
            drop(entry);
            self.cursors.remove(id);
            return Err(CursorError::Expired(id.to_string()));
        }

        entry.last_access = current_timestamp();
        entry.state.touch();
        Ok(entry.state.clone())
    }

    /// Update an existing cursor.
    ///
    /// # Errors
    ///
    /// Returns an error if the cursor is not found.
    pub fn update(&self, cursor: CursorState) -> Result<(), CursorError> {
        let id = cursor.id.clone();
        let mut entry = self
            .cursors
            .get_mut(&id)
            .ok_or_else(|| CursorError::NotFound(id.clone()))?;

        entry.state = cursor;
        entry.last_access = current_timestamp();
        Ok(())
    }

    /// Remove a cursor by ID.
    ///
    /// Returns `true` if the cursor was removed, `false` if not found.
    pub fn remove(&self, id: &str) -> bool {
        self.cursors.remove(id).is_some()
    }

    /// Get the number of active cursors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.cursors.len()
    }

    /// Check if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cursors.is_empty()
    }

    /// Remove all expired cursors.
    ///
    /// Returns the number of cursors removed.
    pub fn cleanup_expired(&self) -> usize {
        let mut removed = 0;
        self.cursors.retain(|_, entry| {
            if entry.state.is_expired() {
                removed += 1;
                false
            } else {
                true
            }
        });
        removed
    }

    /// Evict the least recently used cursor.
    fn evict_lru(&self) {
        if self.cursors.is_empty() {
            return;
        }

        // Find the entry with oldest last_access
        let mut oldest_id: Option<CursorId> = None;
        let mut oldest_time = i64::MAX;

        for entry in self.cursors.iter() {
            if entry.last_access < oldest_time {
                oldest_time = entry.last_access;
                oldest_id = Some(entry.key().clone());
            }
        }

        if let Some(id) = oldest_id {
            self.cursors.remove(&id);
        }
    }

    /// Signal shutdown to stop background tasks.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Check if shutdown has been signaled.
    #[must_use]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    /// Spawn a background cleanup task (requires tokio runtime).
    ///
    /// The task runs until `shutdown()` is called.
    pub fn spawn_cleanup_task(self: &Arc<Self>) {
        let store = Arc::clone(self);
        let interval = self.config.cleanup_interval;

        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                if store.is_shutdown() {
                    break;
                }
                let removed = store.cleanup_expired();
                if removed > 0 {
                    tracing::debug!("Cursor cleanup removed {} expired cursors", removed);
                }
            }
        });
    }
}

impl Default for CursorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current Unix timestamp in seconds.
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::CursorResultType;

    fn create_test_cursor(id: &str) -> CursorState {
        CursorState::new(
            id.to_string(),
            "SELECT users".to_string(),
            CursorResultType::Rows,
            100,
            Some(500),
            300,
        )
    }

    fn create_expired_cursor(id: &str) -> CursorState {
        let mut cursor = CursorState::new(
            id.to_string(),
            "SELECT users".to_string(),
            CursorResultType::Rows,
            100,
            Some(500),
            1, // 1 second TTL
        );
        cursor.last_accessed_at = current_timestamp() - 10; // Expired
        cursor
    }

    #[test]
    fn test_cursor_store_new() {
        let store = CursorStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_cursor_store_with_config() {
        let config = CursorStoreConfig::new()
            .with_max_cursors(100)
            .with_default_ttl(Duration::from_secs(60));
        let store = CursorStore::with_config(config);
        assert_eq!(store.config().max_cursors, 100);
        assert_eq!(store.config().default_ttl, Duration::from_secs(60));
    }

    #[test]
    fn test_cursor_store_insert_get() {
        let store = CursorStore::new();
        let cursor = create_test_cursor("cursor-1");

        store.insert(cursor.clone()).unwrap();
        assert_eq!(store.len(), 1);

        let retrieved = store.get("cursor-1").unwrap();
        assert_eq!(retrieved.id, "cursor-1");
        assert_eq!(retrieved.query, "SELECT users");
    }

    #[test]
    fn test_cursor_store_get_not_found() {
        let store = CursorStore::new();
        let result = store.get("nonexistent");
        assert!(matches!(result, Err(CursorError::NotFound(_))));
    }

    #[test]
    fn test_cursor_store_get_expired() {
        let store = CursorStore::new();
        let cursor = create_expired_cursor("expired-cursor");

        // Insert directly to bypass any validation
        store.cursors.insert(
            "expired-cursor".to_string(),
            CursorEntry {
                state: cursor,
                last_access: current_timestamp() - 100,
            },
        );

        let result = store.get("expired-cursor");
        assert!(matches!(result, Err(CursorError::Expired(_))));

        // Should have been removed
        assert!(store.is_empty());
    }

    #[test]
    fn test_cursor_store_update() {
        let store = CursorStore::new();
        let cursor = create_test_cursor("cursor-1");
        store.insert(cursor).unwrap();

        let mut updated = store.get("cursor-1").unwrap();
        updated.offset = 200;
        store.update(updated).unwrap();

        let retrieved = store.get("cursor-1").unwrap();
        assert_eq!(retrieved.offset, 200);
    }

    #[test]
    fn test_cursor_store_update_not_found() {
        let store = CursorStore::new();
        let cursor = create_test_cursor("nonexistent");
        let result = store.update(cursor);
        assert!(matches!(result, Err(CursorError::NotFound(_))));
    }

    #[test]
    fn test_cursor_store_remove() {
        let store = CursorStore::new();
        let cursor = create_test_cursor("cursor-1");
        store.insert(cursor).unwrap();

        assert!(store.remove("cursor-1"));
        assert!(store.is_empty());
        assert!(!store.remove("cursor-1")); // Already removed
    }

    #[test]
    fn test_cursor_store_cleanup_expired() {
        let store = CursorStore::new();

        // Insert valid cursor
        let valid = create_test_cursor("valid");
        store.insert(valid).unwrap();

        // Insert expired cursor directly
        let expired = create_expired_cursor("expired");
        store.cursors.insert(
            "expired".to_string(),
            CursorEntry {
                state: expired,
                last_access: current_timestamp() - 100,
            },
        );

        assert_eq!(store.len(), 2);

        let removed = store.cleanup_expired();
        assert_eq!(removed, 1);
        assert_eq!(store.len(), 1);
        assert!(store.get("valid").is_ok());
    }

    #[test]
    fn test_cursor_store_evict_lru() {
        let config = CursorStoreConfig::new().with_max_cursors(2);
        let store = CursorStore::with_config(config);

        // Insert cursors with explicit timestamps to control LRU ordering
        let cursor1 = create_test_cursor("cursor-1");
        let cursor2 = create_test_cursor("cursor-2");
        let cursor3 = create_test_cursor("cursor-3");

        let now = current_timestamp();

        // Insert cursor1 with oldest timestamp
        store.cursors.insert(
            "cursor-1".to_string(),
            CursorEntry {
                state: cursor1,
                last_access: now - 100, // Oldest
            },
        );

        // Insert cursor2 with middle timestamp
        store.cursors.insert(
            "cursor-2".to_string(),
            CursorEntry {
                state: cursor2,
                last_access: now - 50, // Middle
            },
        );

        assert_eq!(store.len(), 2);

        // Insert cursor3 - should evict cursor-1 (oldest)
        store.insert(cursor3).unwrap();

        assert_eq!(store.len(), 2);
        assert!(!store.cursors.contains_key("cursor-1")); // Evicted (check directly)
        assert!(store.cursors.contains_key("cursor-2"));
        assert!(store.cursors.contains_key("cursor-3"));
    }

    #[test]
    fn test_cursor_store_shutdown() {
        let store = CursorStore::new();
        assert!(!store.is_shutdown());
        store.shutdown();
        assert!(store.is_shutdown());
    }

    #[test]
    fn test_cursor_store_config_default() {
        let config = CursorStoreConfig::default();
        assert_eq!(config.max_cursors, 10_000);
        assert_eq!(config.default_ttl, Duration::from_secs(300));
        assert_eq!(config.max_ttl, Duration::from_secs(1800));
        assert_eq!(config.cleanup_interval, Duration::from_secs(30));
    }

    #[test]
    fn test_cursor_store_config_builder() {
        let config = CursorStoreConfig::new()
            .with_max_cursors(500)
            .with_default_ttl(Duration::from_secs(120))
            .with_max_ttl(Duration::from_secs(600))
            .with_cleanup_interval(Duration::from_secs(10));

        assert_eq!(config.max_cursors, 500);
        assert_eq!(config.default_ttl, Duration::from_secs(120));
        assert_eq!(config.max_ttl, Duration::from_secs(600));
        assert_eq!(config.cleanup_interval, Duration::from_secs(10));
    }

    #[test]
    fn test_cursor_store_default_trait() {
        let store = CursorStore::default();
        assert!(store.is_empty());
    }

    #[test]
    fn test_cursor_store_len_is_empty() {
        let store = CursorStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.insert(create_test_cursor("c1")).unwrap();
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_evict_lru_empty_store() {
        let store = CursorStore::new();
        store.evict_lru(); // Should not panic
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let store = Arc::new(CursorStore::new());
        let mut handles = vec![];

        for i in 0..10 {
            let store_clone = Arc::clone(&store);
            handles.push(thread::spawn(move || {
                let cursor = create_test_cursor(&format!("cursor-{i}"));
                store_clone.insert(cursor).unwrap();
                store_clone.get(&format!("cursor-{i}")).unwrap();
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(store.len(), 10);
    }

    #[test]
    fn test_get_updates_last_access() {
        let store = CursorStore::new();
        let cursor = create_test_cursor("cursor-1");
        store.insert(cursor).unwrap();

        // First get
        let _ = store.get("cursor-1").unwrap();
        let first_access = store.cursors.get("cursor-1").unwrap().last_access;

        std::thread::sleep(std::time::Duration::from_millis(10));

        // Second get should update last_access
        let _ = store.get("cursor-1").unwrap();
        let second_access = store.cursors.get("cursor-1").unwrap().last_access;

        assert!(second_access >= first_access);
    }

    #[test]
    fn test_capacity_exceeded() {
        let config = CursorStoreConfig::new().with_max_cursors(1);
        let store = CursorStore::with_config(config);

        store.insert(create_test_cursor("c1")).unwrap();

        // Create an expired cursor so eviction finds something to remove
        // But if all cursors are valid, it should still evict LRU
        let result = store.insert(create_test_cursor("c2"));
        // Should succeed because LRU eviction kicks in
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_spawn_cleanup_task() {
        let store = Arc::new(CursorStore::with_config(
            CursorStoreConfig::new().with_cleanup_interval(Duration::from_millis(50)),
        ));

        // Insert an expired cursor directly
        let expired = create_expired_cursor("expired");
        store.cursors.insert(
            "expired".to_string(),
            CursorEntry {
                state: expired,
                last_access: current_timestamp() - 100,
            },
        );

        store.spawn_cleanup_task();

        // Wait for cleanup to run
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert!(store.is_empty());
        store.shutdown();
    }
}
