// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Tiered storage integration tests.
//!
//! Tests hot/cold data migration and access pattern optimization.

use std::{thread, time::Duration};

use tensor_store::{
    tiered::{MigrationStrategy, TieredConfig, TieredStore},
    ScalarValue, TensorData, TensorValue,
};

fn create_test_tensor(id: i64) -> TensorData {
    let mut tensor = TensorData::new();
    tensor.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
    tensor.set(
        "name",
        TensorValue::Scalar(ScalarValue::String(format!("entity_{}", id))),
    );
    tensor.set("data", TensorValue::Vector(vec![id as f32 * 0.1; 64]));
    tensor
}

fn setup_test_dir(name: &str) -> std::path::PathBuf {
    let dir = std::path::PathBuf::from(format!("/tmp/tiered_integration_test_{}", name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_tiered_store_hot_only_mode() {
    let mut store = TieredStore::hot_only(1);

    // Insert data
    for i in 0..100 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    assert_eq!(store.hot_len(), 100);
    assert_eq!(store.cold_len(), 0);
    assert_eq!(store.len(), 100);

    // Read data
    for i in 0..100 {
        let tensor = store.get(&format!("key:{}", i)).unwrap();
        match tensor.get("id") {
            Some(TensorValue::Scalar(ScalarValue::Int(id))) => {
                assert_eq!(*id, i);
            },
            _ => panic!("Expected Int scalar for id"),
        }
    }

    // Stats should show hot lookups
    let stats = store.stats();
    assert_eq!(stats.hot_count, 100);
    assert_eq!(stats.cold_count, 0);
    assert!(stats.hot_lookups >= 100);
}

#[test]
fn test_tiered_store_with_cold_storage() {
    let dir = setup_test_dir("cold_storage");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 1024 * 1024, // 1MB
        sample_rate: 1,
        migration_strategy: MigrationStrategy::default(),
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..50 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    assert_eq!(store.hot_len(), 50);
    assert_eq!(store.cold_len(), 0);

    // All data should be accessible
    for i in 0..50 {
        let tensor = store.get(&format!("key:{}", i)).unwrap();
        match tensor.get("id") {
            Some(TensorValue::Scalar(ScalarValue::Int(id))) => {
                assert_eq!(*id, i);
            },
            _ => panic!("Expected Int scalar"),
        }
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_migrate_cold() {
    let dir = setup_test_dir("migrate_cold");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::default(),
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..100 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // Access only some keys to make others "cold"
    for i in 0..10 {
        let _ = store.get(&format!("key:{}", i));
    }

    // Wait a bit for access timestamps to age
    thread::sleep(Duration::from_millis(100));

    // Migrate cold data (threshold in ms)
    let migrated = store.migrate_cold(50).unwrap();

    // Some data should have migrated to cold
    // (exact count depends on shard distribution)
    assert!(store.cold_len() > 0 || migrated == 0);

    // Stats should reflect migration
    let stats = store.stats();
    assert_eq!(stats.hot_count + stats.cold_count, 100);

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_cold_data_promotion() {
    let dir = setup_test_dir("cold_promotion");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::default(),
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..50 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // Access first 10 to keep them hot
    for i in 0..10 {
        let _ = store.get(&format!("key:{}", i));
    }

    // Wait and migrate cold
    thread::sleep(Duration::from_millis(100));
    let _ = store.migrate_cold(50);

    let cold_before = store.cold_len();

    // Access a cold key - should promote to hot
    if cold_before > 0 {
        // Find a cold key by checking if it's not in hot
        for i in 10..50 {
            let key = format!("key:{}", i);
            if !store.exists(&key) {
                continue;
            }
            let tensor = store.get(&key).unwrap();
            assert!(tensor.get("id").is_some());

            // After access, should have one less cold entry
            let cold_after = store.cold_len();
            // Data was promoted from cold to hot
            assert!(cold_after <= cold_before);
            break;
        }
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_preload_specific_keys() {
    let dir = setup_test_dir("preload");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::default(),
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..100 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // Force all data to cold (use very short threshold)
    thread::sleep(Duration::from_millis(10));
    let _ = store.migrate_cold(1);

    let cold_before = store.cold_len();

    // Preload specific keys
    let keys_to_preload: Vec<&str> = vec!["key:0", "key:10", "key:20"];
    let loaded = store.preload(&keys_to_preload).unwrap();

    // Loaded count should match keys that were actually in cold
    assert!(loaded <= keys_to_preload.len());

    // Preloaded keys should now be in hot
    let cold_after = store.cold_len();
    assert!(cold_after <= cold_before);

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_hot_shards_tracking() {
    let mut store = TieredStore::hot_only(1);

    // Insert and access data to create access patterns
    for i in 0..100 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // Create hotspot by repeatedly accessing certain keys
    for _ in 0..50 {
        let _ = store.get("key:0");
        let _ = store.get("key:1");
    }

    // Get hot shards
    let hot_shards = store.hot_shards(5);
    assert!(!hot_shards.is_empty());

    // Hottest shard should have many accesses
    let (_, access_count) = hot_shards[0];
    assert!(access_count > 10);
}

#[test]
fn test_cold_shards_tracking() {
    let mut store = TieredStore::hot_only(1);

    // Insert data
    for i in 0..100 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // Access only a subset
    for i in 0..10 {
        let _ = store.get(&format!("key:{}", i));
    }

    // Wait for access timestamps to age
    thread::sleep(Duration::from_millis(100));

    // Get cold shards
    let cold_shards = store.cold_shards(50);

    // Should have some cold shards (not all were accessed recently)
    // Note: exact count depends on hash distribution
    assert!(!cold_shards.is_empty() || store.hot_len() <= 10);
}

#[test]
fn test_delete_from_both_tiers() {
    let dir = setup_test_dir("delete_tiers");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::default(),
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..20 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // Migrate some to cold
    thread::sleep(Duration::from_millis(10));
    let _ = store.migrate_cold(1);

    // Delete from hot tier
    let deleted_hot = store.delete("key:0");
    assert!(deleted_hot);

    // Delete should work for any tier
    let total_before = store.len();
    let deleted_any = store.delete("key:10");
    if deleted_any {
        assert_eq!(store.len(), total_before - 1);
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_exists_across_tiers() {
    let dir = setup_test_dir("exists_tiers");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::default(),
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..50 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // All keys should exist
    for i in 0..50 {
        assert!(store.exists(&format!("key:{}", i)));
    }

    // Non-existent key
    assert!(!store.exists("nonexistent"));

    // Migrate to cold
    thread::sleep(Duration::from_millis(10));
    let _ = store.migrate_cold(1);

    // Keys should still exist (in either tier)
    for i in 0..50 {
        assert!(store.exists(&format!("key:{}", i)));
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_tiered_stats() {
    let dir = setup_test_dir("stats");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::default(),
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..100 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // Read some data
    for i in 0..50 {
        let _ = store.get(&format!("key:{}", i));
    }

    let stats = store.stats();
    assert_eq!(stats.hot_count, 100);
    assert!(stats.hot_lookups >= 50);

    // Migrate and check cold stats
    thread::sleep(Duration::from_millis(10));
    let migrated = store.migrate_cold(1).unwrap();

    let stats_after = store.stats();
    assert_eq!(stats_after.hot_count + stats_after.cold_count, 100);
    assert_eq!(stats_after.migrations_to_cold, migrated as u64);

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_flush_cold_storage() {
    let dir = setup_test_dir("flush");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::default(),
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..50 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // Migrate to cold
    thread::sleep(Duration::from_millis(10));
    let _ = store.migrate_cold(1);

    // Flush should not error
    store.flush().unwrap();

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_into_tensor_store() {
    let dir = setup_test_dir("into_store");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 1024 * 1024,
        sample_rate: 1,
        migration_strategy: MigrationStrategy::default(),
    };

    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..50 {
        store.put(format!("key:{}", i), create_test_tensor(i));
    }

    // Migrate some to cold
    thread::sleep(Duration::from_millis(10));
    let _ = store.migrate_cold(1);

    let cold_count = store.cold_len();
    let total = store.len();

    // Convert to TensorStore
    let tensor_store = store.into_tensor_store().unwrap();

    // All data should be in the TensorStore now
    assert_eq!(tensor_store.len(), total);

    // Verify data is accessible
    for i in 0..50 {
        let data = tensor_store.get(&format!("key:{}", i)).unwrap();
        assert!(data.get("id").is_some());
    }

    // The cold data should have been loaded into hot
    assert!(cold_count == 0 || tensor_store.len() >= cold_count);

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_concurrent_tiered_access() {
    use std::sync::{Arc, Mutex};

    let store = Arc::new(Mutex::new(TieredStore::hot_only(1)));

    // Insert initial data
    {
        let mut s = store.lock().unwrap();
        for i in 0..100 {
            s.put(format!("key:{}", i), create_test_tensor(i));
        }
    }

    let mut handles = vec![];

    // 4 threads reading concurrently
    for t in 0..4 {
        let store = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for i in 0..25 {
                let key = format!("key:{}", (t * 25 + i) % 100);
                let mut s = store.lock().unwrap();
                let _ = s.get(&key);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All data should still be accessible
    let s = store.lock().unwrap();
    assert_eq!(s.hot_len(), 100);
}

#[test]
fn test_tiered_large_tensors() {
    let mut store = TieredStore::hot_only(1);

    // Store large tensors
    for i in 0..10 {
        let mut tensor = TensorData::new();
        tensor.set("id", TensorValue::Scalar(ScalarValue::Int(i)));
        // Large embedding vector
        tensor.set(
            "embedding",
            TensorValue::Vector(vec![i as f32 * 0.01; 1024]),
        );
        store.put(format!("large:{}", i), tensor);
    }

    assert_eq!(store.hot_len(), 10);

    // Verify large data is retrieved correctly
    for i in 0..10 {
        let tensor = store.get(&format!("large:{}", i)).unwrap();
        match tensor.get("embedding") {
            Some(TensorValue::Vector(v)) => {
                assert_eq!(v.len(), 1024);
            },
            _ => panic!("Expected Vector"),
        }
    }
}

#[test]
fn test_tiered_empty_operations() {
    let mut store = TieredStore::hot_only(1);

    // Empty store operations
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);

    // Get from empty should fail
    let result = store.get("nonexistent");
    assert!(result.is_err());

    // Delete from empty should return false
    assert!(!store.delete("nonexistent"));

    // Hot/cold shards should be empty or minimal
    let hot = store.hot_shards(5);
    assert!(hot.is_empty() || hot[0].1 == 0);
}

#[test]
fn test_tiered_overwrite() {
    let mut store = TieredStore::hot_only(1);

    // Insert initial value
    store.put("key", create_test_tensor(1));

    // Overwrite with new value
    store.put("key", create_test_tensor(2));

    // Should have only 1 entry
    assert_eq!(store.hot_len(), 1);

    // Should return the new value
    let tensor = store.get("key").unwrap();
    match tensor.get("id") {
        Some(TensorValue::Scalar(ScalarValue::Int(id))) => {
            assert_eq!(*id, 2);
        },
        _ => panic!("Expected Int scalar"),
    }
}
