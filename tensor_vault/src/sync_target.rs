// SPDX-License-Identifier: MIT OR Apache-2.0
//! Secret sync: push secrets to external targets on change.

use std::{io::Write, path::PathBuf};

use dashmap::DashMap;
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

use crate::{encryption::Cipher, Result, VaultError};

/// Storage prefix for sync subscriptions.
const SYNC_PREFIX: &str = "_vsync:";

/// Trait for external sync targets.
pub trait SyncTarget: Send + Sync {
    /// Target name (must be unique).
    fn name(&self) -> &str;

    /// Push a secret value to the target.
    fn push(&self, key: &str, value: &str) -> Result<()>;

    /// Delete a secret from the target.
    fn delete(&self, key: &str) -> Result<()>;

    /// Check if the target is healthy.
    fn health_check(&self) -> Result<bool>;
}

/// Thread-safe sync manager.
pub struct SyncManager {
    targets: DashMap<String, Box<dyn SyncTarget>>,
    subscriptions: DashMap<String, Vec<String>>, // secret_key -> [target_names]
    cipher: Option<Cipher>,
    geo_router: Option<crate::geo_routing::GeoRouter>,
}

impl Default for SyncManager {
    fn default() -> Self {
        Self {
            targets: DashMap::new(),
            subscriptions: DashMap::new(),
            cipher: None,
            geo_router: None,
        }
    }
}

impl SyncManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a sync manager with encryption enabled.
    pub fn with_cipher(cipher: Cipher) -> Self {
        Self {
            targets: DashMap::new(),
            subscriptions: DashMap::new(),
            cipher: Some(cipher),
            geo_router: None,
        }
    }

    /// Create a sync manager with a geographic router for latency-aware routing.
    pub fn with_geo_router(cipher: Option<Cipher>, router: crate::geo_routing::GeoRouter) -> Self {
        Self {
            targets: DashMap::new(),
            subscriptions: DashMap::new(),
            cipher,
            geo_router: Some(router),
        }
    }

    /// Update the cipher (used after unseal to re-derive keys).
    pub fn update_cipher(&mut self, cipher: Cipher) {
        self.cipher = Some(cipher);
    }

    /// Load subscriptions from storage.
    pub fn load_subscriptions(&self, store: &TensorStore) {
        for key in store.scan(SYNC_PREFIX) {
            if let Some(sk) = key.strip_prefix(SYNC_PREFIX) {
                if let Ok(tensor) = store.get(&key) {
                    if let Some(TensorValue::Scalar(ScalarValue::String(targets_json))) =
                        tensor.get("_targets")
                    {
                        if let Ok(targets) = serde_json::from_str::<Vec<String>>(targets_json) {
                            self.subscriptions.insert(sk.to_string(), targets);
                        }
                    }
                }
            }
        }
    }

    /// Register a sync target.
    pub fn register_target(&self, target: Box<dyn SyncTarget>) -> Result<()> {
        let name = target.name().to_string();
        self.targets.insert(name, target);
        Ok(())
    }

    /// Subscribe a secret to a sync target.
    pub fn subscribe(
        &self,
        store: &TensorStore,
        secret_key: &str,
        target_name: &str,
    ) -> Result<()> {
        if !self.targets.contains_key(target_name) {
            return Err(VaultError::NotFound(format!("sync target: {target_name}")));
        }

        let mut entry = self
            .subscriptions
            .entry(secret_key.to_string())
            .or_default();
        if !entry.contains(&target_name.to_string()) {
            entry.push(target_name.to_string());
        }

        let targets = entry.clone();
        drop(entry);
        Self::persist_subscription(store, secret_key, &targets);
        Ok(())
    }

    /// Unsubscribe a secret from a sync target.
    pub fn unsubscribe(
        &self,
        store: &TensorStore,
        secret_key: &str,
        target_name: &str,
    ) -> Result<()> {
        if let Some(mut entry) = self.subscriptions.get_mut(secret_key) {
            entry.retain(|t| t != target_name);
            let targets = entry.clone();
            drop(entry);
            Self::persist_subscription(store, secret_key, &targets);
        }
        Ok(())
    }

    /// Trigger sync for a specific secret key and value.
    ///
    /// When a cipher is set, the value is encrypted before pushing to targets.
    /// Format: base64(nonce(12) || ciphertext).
    pub fn trigger_sync(&self, key: &str, value: &str) -> Result<usize> {
        let Some(targets) = self.subscriptions.get(key) else {
            return Ok(0);
        };

        let push_value = if let Some(ref cipher) = self.cipher {
            use base64::Engine;
            let (ciphertext, nonce) = cipher.encrypt(value.as_bytes())?;
            let mut buf = Vec::with_capacity(nonce.len() + ciphertext.len());
            buf.extend_from_slice(&nonce);
            buf.extend_from_slice(&ciphertext);
            base64::engine::general_purpose::STANDARD.encode(&buf)
        } else {
            value.to_string()
        };

        let mut count = 0;
        for target_name in targets.iter() {
            if let Some(target) = self.targets.get(target_name) {
                if target.push(key, &push_value).is_ok() {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Trigger delete sync for a key.
    pub fn trigger_delete(&self, key: &str) -> Result<usize> {
        let Some(targets) = self.subscriptions.get(key) else {
            return Ok(0);
        };

        let mut count = 0;
        for target_name in targets.iter() {
            if let Some(target) = self.targets.get(target_name) {
                if target.delete(key).is_ok() {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Trigger sync using the geo router to select optimal targets.
    ///
    /// Falls back to `trigger_sync` if no geo router is configured.
    pub fn trigger_sync_routed(
        &self,
        key: &str,
        value: &str,
        location: Option<&crate::manifold::GeoCoordinate>,
    ) -> Result<usize> {
        let Some(ref router) = self.geo_router else {
            return self.trigger_sync(key, value);
        };

        let available: Vec<String> = self.targets.iter().map(|e| e.key().clone()).collect();
        if available.is_empty() {
            return Ok(0);
        }

        let decision = router.route(key, location, &available);

        let push_value = if let Some(ref cipher) = self.cipher {
            use base64::Engine;
            let (ciphertext, nonce) = cipher.encrypt(value.as_bytes())?;
            let mut buf = Vec::with_capacity(nonce.len() + ciphertext.len());
            buf.extend_from_slice(&nonce);
            buf.extend_from_slice(&ciphertext);
            base64::engine::general_purpose::STANDARD.encode(&buf)
        } else {
            value.to_string()
        };

        let mut count = 0;
        for routed in &decision.selected_targets {
            if let Some(target) = self.targets.get(&routed.target_name) {
                let start = std::time::Instant::now();
                let ok = target.push(key, &push_value).is_ok();
                let latency = start.elapsed().as_secs_f64() * 1000.0;
                router.record_sync_result(&routed.target_name, latency, ok);
                if ok {
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// List all registered sync target names.
    pub fn list_targets(&self) -> Vec<String> {
        self.targets.iter().map(|e| e.key().clone()).collect()
    }

    /// Health check for all targets.
    pub fn health_check(&self) -> Vec<(String, bool)> {
        self.targets
            .iter()
            .map(|e| {
                let healthy = e.value().health_check().unwrap_or(false);
                (e.key().clone(), healthy)
            })
            .collect()
    }

    fn persist_subscription(store: &TensorStore, secret_key: &str, targets: &[String]) {
        let key = format!("{SYNC_PREFIX}{secret_key}");
        let mut tensor = TensorData::new();
        let json = serde_json::to_string(targets).unwrap_or_default();
        tensor.set("_targets", TensorValue::Scalar(ScalarValue::String(json)));
        let _ = store.put(&key, tensor);
    }
}

/// File-based sync target: writes secrets to files.
pub struct FileSyncTarget {
    name: String,
    base_dir: PathBuf,
}

impl FileSyncTarget {
    pub fn new(name: &str, base_dir: PathBuf) -> Self {
        Self {
            name: name.to_string(),
            base_dir,
        }
    }
}

impl SyncTarget for FileSyncTarget {
    fn name(&self) -> &str {
        &self.name
    }

    fn push(&self, key: &str, value: &str) -> Result<()> {
        let sanitized = key.replace(['/', '\\'], "_");
        let path = self.base_dir.join(sanitized);

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| VaultError::StorageError(format!("mkdir failed: {e}")))?;
        }

        let mut file = std::fs::File::create(&path)
            .map_err(|e| VaultError::StorageError(format!("file create failed: {e}")))?;
        file.write_all(value.as_bytes())
            .map_err(|e| VaultError::StorageError(format!("write failed: {e}")))?;

        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let sanitized = key.replace(['/', '\\'], "_");
        let path = self.base_dir.join(sanitized);
        std::fs::remove_file(&path).ok();
        Ok(())
    }

    fn health_check(&self) -> Result<bool> {
        Ok(self.base_dir.exists())
    }
}

/// Environment variable sync target: sets env vars (in-process only).
pub struct EnvSyncTarget {
    name: String,
}

impl EnvSyncTarget {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

impl SyncTarget for EnvSyncTarget {
    fn name(&self) -> &str {
        &self.name
    }

    fn push(&self, key: &str, value: &str) -> Result<()> {
        let env_key = key.replace(['/', '-'], "_").to_uppercase();
        std::env::set_var(&env_key, value);
        Ok(())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let env_key = key.replace(['/', '-'], "_").to_uppercase();
        std::env::remove_var(&env_key);
        Ok(())
    }

    fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    /// In-memory sync target for testing.
    struct MemorySyncTarget {
        name: String,
        entries: Mutex<std::collections::HashMap<String, String>>,
        healthy: bool,
    }

    impl MemorySyncTarget {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                entries: Mutex::new(std::collections::HashMap::new()),
                healthy: true,
            }
        }
    }

    impl SyncTarget for MemorySyncTarget {
        fn name(&self) -> &str {
            &self.name
        }

        fn push(&self, key: &str, value: &str) -> Result<()> {
            self.entries
                .lock()
                .unwrap()
                .insert(key.to_string(), value.to_string());
            Ok(())
        }

        fn delete(&self, key: &str) -> Result<()> {
            self.entries.lock().unwrap().remove(key);
            Ok(())
        }

        fn health_check(&self) -> Result<bool> {
            Ok(self.healthy)
        }
    }

    #[test]
    fn test_register_target() {
        let manager = SyncManager::new();
        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();

        let targets = manager.list_targets();
        assert_eq!(targets.len(), 1);
        assert!(targets.contains(&"mem".to_string()));
    }

    #[test]
    fn test_subscribe_and_sync() {
        let store = TensorStore::new();
        let manager = SyncManager::new();
        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();

        manager.subscribe(&store, "db/password", "mem").unwrap();

        let count = manager.trigger_sync("db/password", "secret123").unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_subscribe_to_nonexistent_target() {
        let store = TensorStore::new();
        let manager = SyncManager::new();

        let result = manager.subscribe(&store, "key", "missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_unsubscribe() {
        let store = TensorStore::new();
        let manager = SyncManager::new();
        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();

        manager.subscribe(&store, "key", "mem").unwrap();
        manager.unsubscribe(&store, "key", "mem").unwrap();

        let count = manager.trigger_sync("key", "value").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_trigger_delete() {
        let store = TensorStore::new();
        let manager = SyncManager::new();
        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();

        manager.subscribe(&store, "key", "mem").unwrap();
        manager.trigger_sync("key", "value").unwrap();
        manager.trigger_delete("key").unwrap();
    }

    #[test]
    fn test_health_check() {
        let manager = SyncManager::new();
        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();

        let health = manager.health_check();
        assert_eq!(health.len(), 1);
        assert!(health[0].1); // healthy
    }

    #[test]
    fn test_no_subscriptions() {
        let manager = SyncManager::new();
        let count = manager.trigger_sync("key", "value").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_multiple_targets() {
        let store = TensorStore::new();
        let manager = SyncManager::new();
        manager
            .register_target(Box::new(MemorySyncTarget::new("t1")))
            .unwrap();
        manager
            .register_target(Box::new(MemorySyncTarget::new("t2")))
            .unwrap();

        manager.subscribe(&store, "key", "t1").unwrap();
        manager.subscribe(&store, "key", "t2").unwrap();

        let count = manager.trigger_sync("key", "value").unwrap();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_file_sync_target() {
        let dir = std::env::temp_dir().join("neumann_sync_test");
        std::fs::create_dir_all(&dir).ok();

        let target = FileSyncTarget::new("file", dir.clone());
        assert_eq!(target.name(), "file");

        target.push("test_key", "test_value").unwrap();
        let content = std::fs::read_to_string(dir.join("test_key")).unwrap();
        assert_eq!(content, "test_value");

        target.delete("test_key").unwrap();
        assert!(!dir.join("test_key").exists());

        assert!(target.health_check().unwrap());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_env_sync_target() {
        let target = EnvSyncTarget::new("env");
        assert_eq!(target.name(), "env");
        assert!(target.health_check().unwrap());

        target.push("test/secret", "hello").unwrap();
        assert_eq!(std::env::var("TEST_SECRET").unwrap(), "hello");

        target.delete("test/secret").unwrap();
        assert!(std::env::var("TEST_SECRET").is_err());
    }

    #[test]
    fn test_encrypted_sync_produces_output() {
        let store = TensorStore::new();
        let cipher = Cipher::from_raw_key([99u8; 32]);
        let manager = SyncManager::with_cipher(cipher);
        manager
            .register_target(Box::new(MemorySyncTarget::new("enc")))
            .unwrap();
        manager.subscribe(&store, "enc_key", "enc").unwrap();

        let count = manager.trigger_sync("enc_key", "my_secret_value").unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_no_cipher_sync_is_plaintext() {
        let store = TensorStore::new();
        let manager = SyncManager::new();
        assert!(manager.cipher.is_none());

        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();
        manager.subscribe(&store, "key", "mem").unwrap();
        let count = manager.trigger_sync("key", "plain_value").unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_with_cipher_constructor() {
        let cipher = Cipher::from_raw_key([42u8; 32]);
        let manager = SyncManager::with_cipher(cipher);
        assert!(manager.cipher.is_some());
    }

    #[test]
    fn test_duplicate_subscribe_idempotent() {
        let store = TensorStore::new();
        let manager = SyncManager::new();
        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();

        manager.subscribe(&store, "key", "mem").unwrap();
        manager.subscribe(&store, "key", "mem").unwrap();

        let count = manager.trigger_sync("key", "value").unwrap();
        assert_eq!(count, 1); // Not double-synced
    }

    #[test]
    fn test_persistence() {
        let store = TensorStore::new();
        let manager = SyncManager::new();
        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();
        manager.subscribe(&store, "persistent_key", "mem").unwrap();

        // New manager loads from store
        let loaded = SyncManager::new();
        loaded
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();
        loaded.load_subscriptions(&store);

        let count = loaded.trigger_sync("persistent_key", "value").unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_with_geo_router_constructor() {
        use crate::geo_routing::{GeoRouter, RoutingConfig};

        let router = GeoRouter::new(RoutingConfig::default());
        let manager = SyncManager::with_geo_router(None, router);
        assert!(manager.geo_router.is_some());
        assert!(manager.cipher.is_none());
    }

    #[test]
    fn test_with_geo_router_constructor_with_cipher() {
        use crate::geo_routing::{GeoRouter, RoutingConfig};

        let cipher = Cipher::from_raw_key([77u8; 32]);
        let router = GeoRouter::new(RoutingConfig::default());
        let manager = SyncManager::with_geo_router(Some(cipher), router);
        assert!(manager.geo_router.is_some());
        assert!(manager.cipher.is_some());
    }

    #[test]
    fn test_trigger_sync_routed_with_router() {
        use crate::geo_routing::{GeoRouter, RoutingConfig, TargetGeometry};
        use crate::manifold::GeoCoordinate;

        let store = TensorStore::new();

        let config = RoutingConfig {
            sync_fanout: 2,
            ..RoutingConfig::default()
        };
        let router = GeoRouter::new(config);

        // Register target geometry
        router.update_geometry(TargetGeometry {
            target_name: "mem".to_string(),
            location: GeoCoordinate {
                x: 0.0,
                y: 0.0,
                z: None,
            },
            avg_latency_ms: 10.0,
            avg_throughput: 100.0,
            failure_rate: 0.0,
            last_health_check_ms: 0,
        });

        let manager = SyncManager::with_geo_router(None, router);
        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();
        manager.subscribe(&store, "routed_key", "mem").unwrap();

        let count = manager
            .trigger_sync_routed("routed_key", "secret_value", None)
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_trigger_sync_routed_no_router() {
        let store = TensorStore::new();
        let manager = SyncManager::new();
        assert!(manager.geo_router.is_none());

        manager
            .register_target(Box::new(MemorySyncTarget::new("mem")))
            .unwrap();
        manager.subscribe(&store, "fallback_key", "mem").unwrap();

        // Falls back to trigger_sync
        let count = manager
            .trigger_sync_routed("fallback_key", "value", None)
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_trigger_sync_routed_with_location() {
        use crate::geo_routing::{GeoRouter, RoutingConfig, TargetGeometry};
        use crate::manifold::GeoCoordinate;

        let store = TensorStore::new();

        let config = RoutingConfig {
            sync_fanout: 2,
            proximity_weight: 1.0,
            latency_weight: 0.0,
            reliability_weight: 0.0,
            ..RoutingConfig::default()
        };
        let router = GeoRouter::new(config);

        router.update_geometry(TargetGeometry {
            target_name: "nearby".to_string(),
            location: GeoCoordinate {
                x: 1.0,
                y: 1.0,
                z: None,
            },
            avg_latency_ms: 20.0,
            avg_throughput: 100.0,
            failure_rate: 0.0,
            last_health_check_ms: 0,
        });

        let manager = SyncManager::with_geo_router(None, router);
        manager
            .register_target(Box::new(MemorySyncTarget::new("nearby")))
            .unwrap();
        manager.subscribe(&store, "geo_key", "nearby").unwrap();

        let location = GeoCoordinate {
            x: 0.0,
            y: 0.0,
            z: None,
        };
        let count = manager
            .trigger_sync_routed("geo_key", "geo_value", Some(&location))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_trigger_sync_routed_empty_targets() {
        use crate::geo_routing::{GeoRouter, RoutingConfig};

        let router = GeoRouter::new(RoutingConfig::default());
        let manager = SyncManager::with_geo_router(None, router);
        // No targets registered at all
        let count = manager
            .trigger_sync_routed("any_key", "value", None)
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_trigger_sync_routed_with_cipher() {
        use crate::geo_routing::{GeoRouter, RoutingConfig, TargetGeometry};
        use crate::manifold::GeoCoordinate;

        let store = TensorStore::new();
        let cipher = Cipher::from_raw_key([55u8; 32]);

        let router = GeoRouter::new(RoutingConfig::default());
        router.update_geometry(TargetGeometry {
            target_name: "enc_target".to_string(),
            location: GeoCoordinate {
                x: 0.0,
                y: 0.0,
                z: None,
            },
            avg_latency_ms: 10.0,
            avg_throughput: 100.0,
            failure_rate: 0.0,
            last_health_check_ms: 0,
        });

        let manager = SyncManager::with_geo_router(Some(cipher), router);
        manager
            .register_target(Box::new(MemorySyncTarget::new("enc_target")))
            .unwrap();
        manager
            .subscribe(&store, "enc_routed_key", "enc_target")
            .unwrap();

        let count = manager
            .trigger_sync_routed("enc_routed_key", "secret_data", None)
            .unwrap();
        assert_eq!(count, 1);
    }
}
