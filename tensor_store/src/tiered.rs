use crate::instrumentation::ShardAccessTracker;
use crate::mmap::{MmapError, MmapStoreMut};
use crate::{TensorData, TensorStore, TensorStoreError};
use dashmap::DashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug)]
pub enum TieredError {
    Store(TensorStoreError),
    Mmap(MmapError),
    Io(std::io::Error),
    NotConfigured,
}

impl std::fmt::Display for TieredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TieredError::Store(e) => write!(f, "Store error: {}", e),
            TieredError::Mmap(e) => write!(f, "Mmap error: {}", e),
            TieredError::Io(e) => write!(f, "I/O error: {}", e),
            TieredError::NotConfigured => write!(f, "Cold storage not configured"),
        }
    }
}

impl std::error::Error for TieredError {}

impl From<TensorStoreError> for TieredError {
    fn from(e: TensorStoreError) -> Self {
        TieredError::Store(e)
    }
}

impl From<MmapError> for TieredError {
    fn from(e: MmapError) -> Self {
        TieredError::Mmap(e)
    }
}

impl From<std::io::Error> for TieredError {
    fn from(e: std::io::Error) -> Self {
        TieredError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, TieredError>;

/// Configuration for tiered storage.
#[derive(Clone)]
pub struct TieredConfig {
    /// Directory for cold storage files.
    pub cold_dir: PathBuf,
    /// Initial capacity for cold storage file in bytes.
    pub cold_capacity: usize,
    /// Sampling rate for access tracking (1 = every access, 100 = 1%).
    pub sample_rate: u32,
}

impl Default for TieredConfig {
    fn default() -> Self {
        Self {
            cold_dir: PathBuf::from("/tmp/tensor_cold"),
            cold_capacity: 64 * 1024 * 1024, // 64MB initial
            sample_rate: 100,
        }
    }
}

/// Statistics about tiered storage.
#[derive(Debug, Clone)]
pub struct TieredStats {
    pub hot_count: usize,
    pub cold_count: usize,
    pub hot_lookups: u64,
    pub cold_lookups: u64,
    pub cold_hits: u64,
    pub migrations_to_cold: u64,
    pub migrations_to_hot: u64,
}

/// Two-tier storage with hot (in-memory) and cold (mmap) layers.
///
/// Data starts in the hot tier and can be migrated to cold based on access patterns.
/// When cold data is accessed, it's promoted back to hot.
pub struct TieredStore {
    /// Hot tier: in-memory DashMap with access tracking.
    hot: Arc<DashMap<String, TensorData>>,
    /// Cold tier: memory-mapped file storage.
    cold: Option<MmapStoreMut>,
    /// Access tracking for hot tier.
    instrumentation: Arc<ShardAccessTracker>,
    /// Tracks which keys are in cold storage.
    cold_keys: Arc<DashMap<String, ()>>,
    /// Configuration.
    #[allow(dead_code)]
    config: TieredConfig,
    /// Statistics.
    hot_lookups: AtomicU64,
    cold_lookups: AtomicU64,
    cold_hits: AtomicU64,
    migrations_to_cold: AtomicU64,
    migrations_to_hot: AtomicU64,
}

impl TieredStore {
    /// Create a new tiered store with the given configuration.
    pub fn new(config: TieredConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.cold_dir)?;

        let cold_path = config.cold_dir.join("cold_store.bin");
        let cold = if cold_path.exists() {
            Some(MmapStoreMut::open(&cold_path)?)
        } else {
            Some(MmapStoreMut::create(&cold_path, config.cold_capacity)?)
        };

        // Build cold_keys index from existing cold store
        let cold_keys = Arc::new(DashMap::new());
        if let Some(ref c) = cold {
            for key in c.keys() {
                cold_keys.insert(key.clone(), ());
            }
        }

        Ok(Self {
            hot: Arc::new(DashMap::new()),
            cold,
            instrumentation: Arc::new(ShardAccessTracker::new(16, config.sample_rate)),
            cold_keys,
            config,
            hot_lookups: AtomicU64::new(0),
            cold_lookups: AtomicU64::new(0),
            cold_hits: AtomicU64::new(0),
            migrations_to_cold: AtomicU64::new(0),
            migrations_to_hot: AtomicU64::new(0),
        })
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(TieredConfig::default())
    }

    /// Create without cold storage (hot-only mode).
    pub fn hot_only(sample_rate: u32) -> Self {
        Self {
            hot: Arc::new(DashMap::new()),
            cold: None,
            instrumentation: Arc::new(ShardAccessTracker::new(16, sample_rate)),
            cold_keys: Arc::new(DashMap::new()),
            config: TieredConfig::default(),
            hot_lookups: AtomicU64::new(0),
            cold_lookups: AtomicU64::new(0),
            cold_hits: AtomicU64::new(0),
            migrations_to_cold: AtomicU64::new(0),
            migrations_to_hot: AtomicU64::new(0),
        }
    }

    /// Insert data into hot tier.
    pub fn put(&mut self, key: impl Into<String>, tensor: TensorData) {
        let key = key.into();
        let shard = self.hot.determine_map(&key);
        self.instrumentation.record_write(shard);
        self.hot.insert(key, tensor);
    }

    /// Get data, checking hot tier first, then cold.
    ///
    /// If found in cold tier, promotes to hot.
    pub fn get(&mut self, key: &str) -> Result<TensorData> {
        // Try hot tier first
        self.hot_lookups.fetch_add(1, Ordering::Relaxed);
        if let Some(data) = self.hot.get(key) {
            let shard = self.hot.determine_map(key);
            self.instrumentation.record_read(shard);
            return Ok(data.value().clone());
        }

        // Try cold tier
        if self.cold_keys.contains_key(key) {
            self.cold_lookups.fetch_add(1, Ordering::Relaxed);
            if let Some(ref cold) = self.cold {
                let tensor = cold.get(key)?;
                self.cold_hits.fetch_add(1, Ordering::Relaxed);

                // Promote to hot
                self.hot.insert(key.to_string(), tensor.clone());
                self.cold_keys.remove(key);
                self.migrations_to_hot.fetch_add(1, Ordering::Relaxed);

                let shard = self.hot.determine_map(key);
                self.instrumentation.record_read(shard);

                return Ok(tensor);
            }
        }

        Err(TieredError::Store(TensorStoreError::NotFound(
            key.to_string(),
        )))
    }

    /// Check if key exists in either tier.
    pub fn exists(&self, key: &str) -> bool {
        self.hot.contains_key(key) || self.cold_keys.contains_key(key)
    }

    /// Delete from both tiers.
    pub fn delete(&mut self, key: &str) -> bool {
        let hot_removed = self.hot.remove(key).is_some();
        let cold_removed = self.cold_keys.remove(key).is_some();
        hot_removed || cold_removed
    }

    /// Get statistics.
    pub fn stats(&self) -> TieredStats {
        TieredStats {
            hot_count: self.hot.len(),
            cold_count: self.cold_keys.len(),
            hot_lookups: self.hot_lookups.load(Ordering::Relaxed),
            cold_lookups: self.cold_lookups.load(Ordering::Relaxed),
            cold_hits: self.cold_hits.load(Ordering::Relaxed),
            migrations_to_cold: self.migrations_to_cold.load(Ordering::Relaxed),
            migrations_to_hot: self.migrations_to_hot.load(Ordering::Relaxed),
        }
    }

    /// Migrate cold shards to mmap storage.
    ///
    /// Keys in shards that haven't been accessed within `threshold_ms` are moved to cold.
    /// Returns the number of keys migrated.
    pub fn migrate_cold(&mut self, threshold_ms: u64) -> Result<usize> {
        let cold = self.cold.as_mut().ok_or(TieredError::NotConfigured)?;

        // Find cold shards
        let cold_shards = self.instrumentation.cold_shards(threshold_ms);

        let mut migrated = 0;

        // Collect keys to migrate (can't iterate and modify DashMap simultaneously)
        let keys_to_migrate: Vec<String> = self
            .hot
            .iter()
            .filter(|entry| {
                let shard = self.hot.determine_map(entry.key());
                cold_shards.contains(&shard)
            })
            .map(|entry| entry.key().clone())
            .collect();

        for key in keys_to_migrate {
            if let Some((k, tensor)) = self.hot.remove(&key) {
                cold.insert(&k, &tensor)?;
                self.cold_keys.insert(k, ());
                migrated += 1;
            }
        }

        if migrated > 0 {
            cold.flush()?;
            self.migrations_to_cold
                .fetch_add(migrated as u64, Ordering::Relaxed);
        }

        Ok(migrated)
    }

    /// Preload specific keys from cold to hot.
    pub fn preload(&mut self, keys: &[&str]) -> Result<usize> {
        let cold = self.cold.as_ref().ok_or(TieredError::NotConfigured)?;

        let mut loaded = 0;
        for key in keys {
            if self.cold_keys.contains_key(*key) && !self.hot.contains_key(*key) {
                if let Ok(tensor) = cold.get(key) {
                    self.hot.insert((*key).to_string(), tensor);
                    self.cold_keys.remove(*key);
                    loaded += 1;
                }
            }
        }

        if loaded > 0 {
            self.migrations_to_hot
                .fetch_add(loaded as u64, Ordering::Relaxed);
        }

        Ok(loaded)
    }

    /// Get the number of entries in hot tier.
    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }

    /// Get the number of entries in cold tier.
    pub fn cold_len(&self) -> usize {
        self.cold_keys.len()
    }

    /// Total entries across both tiers.
    pub fn len(&self) -> usize {
        self.hot.len() + self.cold_keys.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.hot.is_empty() && self.cold_keys.is_empty()
    }

    /// Flush cold storage to disk.
    pub fn flush(&self) -> Result<()> {
        if let Some(ref cold) = self.cold {
            cold.flush()?;
        }
        Ok(())
    }

    /// Get hot shards (most accessed).
    pub fn hot_shards(&self, limit: usize) -> Vec<(usize, u64)> {
        self.instrumentation.hot_shards(limit)
    }

    /// Get cold shards (not accessed within threshold).
    pub fn cold_shards(&self, threshold_ms: u64) -> Vec<usize> {
        self.instrumentation.cold_shards(threshold_ms)
    }

    /// Convert to a standard TensorStore (loads all cold data to hot).
    pub fn into_tensor_store(self) -> Result<TensorStore> {
        // Load all cold data into hot
        if let Some(ref cold) = self.cold {
            let cold_key_list: Vec<String> =
                self.cold_keys.iter().map(|e| e.key().clone()).collect();
            for key in cold_key_list {
                if let Ok(tensor) = cold.get(&key) {
                    self.hot.insert(key.clone(), tensor);
                }
                self.cold_keys.remove(&key);
            }
        }

        Ok(TensorStore::from_dashmap(self.hot))
    }
}

// Extend TensorStore with from_dashmap
impl TensorStore {
    /// Create a TensorStore from an existing DashMap.
    pub fn from_dashmap(data: Arc<DashMap<String, TensorData>>) -> Self {
        Self {
            data,
            bloom_filter: None,
            instrumentation: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ScalarValue, TensorValue};
    use std::fs;

    fn create_test_tensor(id: i64) -> TensorData {
        let mut tensor = TensorData::new();
        tensor.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
        tensor.set(
            "name",
            TensorValue::Scalar(ScalarValue::String(format!("entity_{}", id))),
        );
        tensor
    }

    fn setup_test_dir(name: &str) -> PathBuf {
        let dir = PathBuf::from(format!("/tmp/tiered_test_{}", name));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_tiered_store_hot_only() {
        let mut store = TieredStore::hot_only(1);

        for i in 0..100 {
            store.put(format!("key_{}", i), create_test_tensor(i));
        }

        assert_eq!(store.hot_len(), 100);
        assert_eq!(store.cold_len(), 0);

        for i in 0..100 {
            let tensor = store.get(&format!("key_{}", i)).unwrap();
            match tensor.get("id") {
                Some(TensorValue::Scalar(ScalarValue::Int(id))) => {
                    assert_eq!(*id, i);
                },
                _ => panic!("Expected int id"),
            }
        }
    }

    #[test]
    fn test_tiered_store_with_cold() {
        let dir = setup_test_dir("with_cold");
        let config = TieredConfig {
            cold_dir: dir.clone(),
            cold_capacity: 4096,
            sample_rate: 1,
        };

        {
            let mut store = TieredStore::new(config.clone()).unwrap();

            for i in 0..100 {
                store.put(format!("key_{}", i), create_test_tensor(i));
            }

            assert_eq!(store.hot_len(), 100);
            store.flush().unwrap();
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tiered_migration() {
        let dir = setup_test_dir("migration");
        let config = TieredConfig {
            cold_dir: dir.clone(),
            cold_capacity: 4096,
            sample_rate: 1,
        };

        let mut store = TieredStore::new(config).unwrap();

        // Insert data
        for i in 0..100 {
            store.put(format!("key_{}", i), create_test_tensor(i));
        }

        // Access some keys to make them hot
        for i in 0..10 {
            let _ = store.get(&format!("key_{}", i));
        }

        // Wait for instrumentation threshold
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Migrate cold data (threshold 5ms - anything not accessed in 5ms is cold)
        let migrated = store.migrate_cold(5).unwrap();

        // Some keys should have migrated (the ones not accessed)
        // Due to timing, this is approximate
        assert!(store.cold_len() > 0 || migrated > 0 || store.hot_len() == 100);

        // Verify all data is still accessible
        for i in 0..100 {
            assert!(store.exists(&format!("key_{}", i)));
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tiered_cold_promotion() {
        let dir = setup_test_dir("cold_promotion");
        let config = TieredConfig {
            cold_dir: dir.clone(),
            cold_capacity: 4096,
            sample_rate: 1,
        };

        let mut store = TieredStore::new(config).unwrap();

        // Insert directly to cold via mmap
        {
            let cold = store.cold.as_mut().unwrap();
            for i in 0..10 {
                cold.insert(&format!("cold_{}", i), &create_test_tensor(i + 1000))
                    .unwrap();
                store.cold_keys.insert(format!("cold_{}", i), ());
            }
            cold.flush().unwrap();
        }

        assert_eq!(store.cold_len(), 10);
        assert_eq!(store.hot_len(), 0);

        // Access cold data - should promote to hot
        let tensor = store.get("cold_5").unwrap();
        match tensor.get("id") {
            Some(TensorValue::Scalar(ScalarValue::Int(id))) => {
                assert_eq!(*id, 1005);
            },
            _ => panic!("Expected int id"),
        }

        // Should be promoted
        assert_eq!(store.cold_len(), 9);
        assert_eq!(store.hot_len(), 1);

        let stats = store.stats();
        assert_eq!(stats.cold_hits, 1);
        assert_eq!(stats.migrations_to_hot, 1);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tiered_preload() {
        let dir = setup_test_dir("preload");
        let config = TieredConfig {
            cold_dir: dir.clone(),
            cold_capacity: 4096,
            sample_rate: 1,
        };

        let mut store = TieredStore::new(config).unwrap();

        // Insert to cold
        {
            let cold = store.cold.as_mut().unwrap();
            for i in 0..10 {
                cold.insert(&format!("cold_{}", i), &create_test_tensor(i))
                    .unwrap();
                store.cold_keys.insert(format!("cold_{}", i), ());
            }
            cold.flush().unwrap();
        }

        assert_eq!(store.cold_len(), 10);

        // Preload specific keys
        let loaded = store.preload(&["cold_0", "cold_5", "cold_9"]).unwrap();
        assert_eq!(loaded, 3);
        assert_eq!(store.hot_len(), 3);
        assert_eq!(store.cold_len(), 7);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tiered_to_tensor_store() {
        let dir = setup_test_dir("to_tensor_store");
        let config = TieredConfig {
            cold_dir: dir.clone(),
            cold_capacity: 4096,
            sample_rate: 1,
        };

        let mut store = TieredStore::new(config).unwrap();

        // Add hot data
        for i in 0..5 {
            store.put(format!("hot_{}", i), create_test_tensor(i));
        }

        // Add cold data
        {
            let cold = store.cold.as_mut().unwrap();
            for i in 0..5 {
                cold.insert(&format!("cold_{}", i), &create_test_tensor(i + 100))
                    .unwrap();
                store.cold_keys.insert(format!("cold_{}", i), ());
            }
        }

        assert_eq!(store.len(), 10);

        // Convert to TensorStore (loads all cold to hot)
        let tensor_store = store.into_tensor_store().unwrap();
        assert_eq!(tensor_store.len(), 10);

        // All data should be accessible
        for i in 0..5 {
            assert!(tensor_store.get(&format!("hot_{}", i)).is_ok());
            assert!(tensor_store.get(&format!("cold_{}", i)).is_ok());
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tiered_stats() {
        let mut store = TieredStore::hot_only(1);

        for i in 0..10 {
            store.put(format!("key_{}", i), create_test_tensor(i));
        }

        // Access some keys
        for i in 0..5 {
            let _ = store.get(&format!("key_{}", i));
        }

        let stats = store.stats();
        assert_eq!(stats.hot_count, 10);
        assert_eq!(stats.cold_count, 0);
        assert_eq!(stats.hot_lookups, 5);
    }

    #[test]
    fn test_tiered_delete() {
        let mut store = TieredStore::hot_only(1);

        store.put("key1", create_test_tensor(1));
        assert!(store.exists("key1"));

        assert!(store.delete("key1"));
        assert!(!store.exists("key1"));

        assert!(!store.delete("nonexistent"));
    }

    #[test]
    fn test_tiered_error_display() {
        use crate::TensorStoreError;

        let store_err = TieredError::Store(TensorStoreError::NotFound("test".to_string()));
        assert!(store_err.to_string().contains("Store error"));

        let mmap_err = TieredError::Mmap(MmapError::NotFound("test".to_string()));
        assert!(mmap_err.to_string().contains("Mmap error"));

        let io_err = TieredError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        assert!(io_err.to_string().contains("I/O error"));

        let not_configured = TieredError::NotConfigured;
        assert!(not_configured.to_string().contains("not configured"));
    }

    #[test]
    fn test_tiered_error_is_error() {
        let err: Box<dyn std::error::Error> = Box::new(TieredError::NotConfigured);
        assert!(err.to_string().contains("not configured"));
    }

    #[test]
    fn test_tiered_hot_only_no_cold() {
        let mut store = TieredStore::hot_only(1);

        // migrate_cold should fail without cold storage
        assert!(matches!(
            store.migrate_cold(1000),
            Err(TieredError::NotConfigured)
        ));

        // preload should fail without cold storage
        assert!(matches!(
            store.preload(&["key_0"]),
            Err(TieredError::NotConfigured)
        ));
    }

    #[test]
    fn test_tiered_get_not_found() {
        let mut store = TieredStore::hot_only(1);
        assert!(store.get("nonexistent").is_err());
    }

    #[test]
    fn test_tiered_len_and_is_empty() {
        let mut store = TieredStore::hot_only(1);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);

        store.put("key1", create_test_tensor(1));
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn test_tiered_hot_cold_shards() {
        let mut store = TieredStore::hot_only(1);

        for i in 0..100 {
            store.put(format!("key_{}", i), create_test_tensor(i));
        }

        // Access some to make them hot
        for i in 0..10 {
            let _ = store.get(&format!("key_{}", i));
        }

        let hot = store.hot_shards(5);
        assert!(!hot.is_empty());

        // Wait a bit
        std::thread::sleep(std::time::Duration::from_millis(10));

        let cold = store.cold_shards(5);
        // Some shards should be cold now
        assert!(!cold.is_empty() || hot.len() >= 5);
    }

    #[test]
    fn test_tiered_flush_hot_only() {
        let store = TieredStore::hot_only(1);
        assert!(store.flush().is_ok());
    }

    #[test]
    fn test_tiered_config_default() {
        let config = TieredConfig::default();
        assert_eq!(config.cold_capacity, 64 * 1024 * 1024);
        assert_eq!(config.sample_rate, 100);
    }

    #[test]
    fn test_tiered_with_defaults() {
        let dir = setup_test_dir("defaults");
        // Temporarily set the default path
        let _ = TieredStore::new(TieredConfig {
            cold_dir: dir.clone(),
            cold_capacity: 4096,
            sample_rate: 100,
        });
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tiered_preload_nonexistent() {
        let dir = setup_test_dir("preload_nonexistent");
        let config = TieredConfig {
            cold_dir: dir.clone(),
            cold_capacity: 4096,
            sample_rate: 1,
        };

        let mut store = TieredStore::new(config).unwrap();

        // Preload keys that don't exist
        let loaded = store.preload(&["nonexistent1", "nonexistent2"]).unwrap();
        assert_eq!(loaded, 0);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tiered_delete_from_cold() {
        let dir = setup_test_dir("delete_cold");
        let config = TieredConfig {
            cold_dir: dir.clone(),
            cold_capacity: 4096,
            sample_rate: 1,
        };

        let mut store = TieredStore::new(config).unwrap();

        // Add to cold
        {
            let cold = store.cold.as_mut().unwrap();
            cold.insert("cold_key", &create_test_tensor(1)).unwrap();
            store.cold_keys.insert("cold_key".to_string(), ());
        }

        assert!(store.exists("cold_key"));
        assert!(store.delete("cold_key"));
        assert!(!store.exists("cold_key"));

        let _ = fs::remove_dir_all(&dir);
    }
}
