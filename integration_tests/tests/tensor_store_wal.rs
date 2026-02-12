// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for TensorStore WAL (Write-Ahead Log) durability.
//!
//! These tests verify that:
//! 1. WAL correctly logs all durable operations
//! 2. Recovery from WAL restores all committed data
//! 3. Checkpoint + WAL recovery works correctly
//! 4. Transactions are handled properly during recovery
//! 5. No data corruption occurs during crash simulation

use std::fs;
use std::sync::Arc;
use std::thread;

use tempfile::tempdir;
use tensor_store::{ScalarValue, SlabRouter, TensorData, TensorStore, TensorValue, WalConfig};

#[test]
fn test_tensor_store_wal_basic_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Create store, add data, and close
    {
        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

        let mut tensor1 = TensorData::new();
        tensor1.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Alice".into())),
        );
        tensor1.set("age", TensorValue::Scalar(ScalarValue::Int(30)));
        store.put_durable("user:1", tensor1).unwrap();

        let mut tensor2 = TensorData::new();
        tensor2.set(
            "name",
            TensorValue::Scalar(ScalarValue::String("Bob".into())),
        );
        tensor2.set("age", TensorValue::Scalar(ScalarValue::Int(25)));
        store.put_durable("user:2", tensor2).unwrap();

        // Store goes out of scope without explicit close
    }

    // Recover and verify
    let recovered = TensorStore::recover(&wal_path, &WalConfig::default(), None).unwrap();

    assert!(recovered.exists("user:1"));
    assert!(recovered.exists("user:2"));

    let user1 = recovered.get("user:1").unwrap();
    assert_eq!(
        user1.get("name"),
        Some(&TensorValue::Scalar(ScalarValue::String("Alice".into())))
    );
    assert_eq!(
        user1.get("age"),
        Some(&TensorValue::Scalar(ScalarValue::Int(30)))
    );
}

#[test]
fn test_tensor_store_wal_checkpoint_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");
    let snapshot_path = dir.path().join("snapshot.bin");

    // Create store, add data, checkpoint, add more data
    {
        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

        // First batch of data
        for i in 0..10 {
            let mut tensor = TensorData::new();
            tensor.set("value", TensorValue::Scalar(ScalarValue::Int(i)));
            store.put_durable(format!("item:{}", i), tensor).unwrap();
        }

        // Checkpoint
        store.checkpoint(&snapshot_path).unwrap();

        // Second batch of data (after checkpoint)
        for i in 10..20 {
            let mut tensor = TensorData::new();
            tensor.set("value", TensorValue::Scalar(ScalarValue::Int(i)));
            store.put_durable(format!("item:{}", i), tensor).unwrap();
        }
    }

    // Recover from snapshot + WAL
    let recovered =
        TensorStore::recover(&wal_path, &WalConfig::default(), Some(&snapshot_path)).unwrap();

    // All 20 items should be present
    assert_eq!(recovered.len(), 20);
    for i in 0..20 {
        assert!(recovered.exists(&format!("item:{}", i)));
        let item = recovered.get(&format!("item:{}", i)).unwrap();
        assert_eq!(
            item.get("value"),
            Some(&TensorValue::Scalar(ScalarValue::Int(i)))
        );
    }
}

#[test]
fn test_tensor_store_wal_delete_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Create, add, delete, then crash
    {
        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

        for i in 0..10 {
            let mut tensor = TensorData::new();
            tensor.set("value", TensorValue::Scalar(ScalarValue::Int(i)));
            store.put_durable(format!("item:{}", i), tensor).unwrap();
        }

        // Delete some items
        store.delete_durable("item:3").unwrap();
        store.delete_durable("item:5").unwrap();
        store.delete_durable("item:7").unwrap();
    }

    // Recover
    let recovered = TensorStore::recover(&wal_path, &WalConfig::default(), None).unwrap();

    // Should have 7 items (10 - 3 deleted)
    assert_eq!(recovered.len(), 7);
    assert!(!recovered.exists("item:3"));
    assert!(!recovered.exists("item:5"));
    assert!(!recovered.exists("item:7"));
    assert!(recovered.exists("item:0"));
    assert!(recovered.exists("item:4"));
    assert!(recovered.exists("item:9"));
}

#[test]
fn test_tensor_store_wal_no_temp_files_after_truncate() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");
    let snapshot_path = dir.path().join("snapshot.bin");

    let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

    // Add some data
    for i in 0..5 {
        store
            .put_durable(format!("key:{}", i), TensorData::new())
            .unwrap();
    }

    // Checkpoint (which truncates WAL)
    store.checkpoint(&snapshot_path).unwrap();

    // Verify no temp files remain
    for entry in fs::read_dir(dir.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        assert!(
            !name.contains(".tmp."),
            "Temp file found after checkpoint: {}",
            name
        );
    }
}

#[test]
fn test_tensor_store_wal_concurrent_writes() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    let store = Arc::new(TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap());
    let mut handles = vec![];

    // Spawn multiple threads writing concurrently
    for t in 0..4 {
        let store = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for i in 0..25 {
                let mut tensor = TensorData::new();
                tensor.set("thread", TensorValue::Scalar(ScalarValue::Int(t)));
                tensor.set("index", TensorValue::Scalar(ScalarValue::Int(i)));
                store
                    .put_durable(format!("thread{}:item{}", t, i), tensor)
                    .unwrap();
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All 100 items should be present
    assert_eq!(store.len(), 100);

    // Sync and drop
    store.wal_sync().unwrap();
    drop(store);

    // Recover and verify
    let recovered = TensorStore::recover(&wal_path, &WalConfig::default(), None).unwrap();
    assert_eq!(recovered.len(), 100);

    for t in 0..4 {
        for i in 0..25 {
            assert!(recovered.exists(&format!("thread{}:item{}", t, i)));
        }
    }
}

#[test]
fn test_tensor_store_wal_with_embeddings() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Create store with embedding data
    {
        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

        for i in 0..5 {
            let mut tensor = TensorData::new();
            tensor.set(
                "name",
                TensorValue::Scalar(ScalarValue::String(format!("vec{}", i))),
            );
            // Create embedding vector
            let embedding: Vec<f32> = (0..128).map(|j| (i * 128 + j) as f32 * 0.001).collect();
            tensor.set("_embedding", TensorValue::Vector(embedding));
            store.put_durable(format!("emb:{}", i), tensor).unwrap();
        }
    }

    // Recover
    let recovered = TensorStore::recover(&wal_path, &WalConfig::default(), None).unwrap();

    // Verify all embeddings are recovered
    for i in 0..5 {
        let tensor = recovered.get(&format!("emb:{}", i)).unwrap();
        assert!(tensor.get("_embedding").is_some());
        if let Some(TensorValue::Vector(vec)) = tensor.get("_embedding") {
            assert_eq!(vec.len(), 128);
            // Verify first element
            let expected = (i * 128) as f32 * 0.001;
            assert!((vec[0] - expected).abs() < 0.0001);
        }
    }
}

#[test]
fn test_tensor_store_wal_multiple_checkpoints() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");
    let snapshot_path = dir.path().join("snapshot.bin");

    let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

    // Multiple checkpoint cycles
    for cycle in 0..3 {
        // Add data
        for i in 0..10 {
            let mut tensor = TensorData::new();
            tensor.set("cycle", TensorValue::Scalar(ScalarValue::Int(cycle)));
            tensor.set("index", TensorValue::Scalar(ScalarValue::Int(i)));
            store
                .put_durable(format!("cycle{}:item{}", cycle, i), tensor)
                .unwrap();
        }

        // Checkpoint
        let checkpoint_id = store.checkpoint(&snapshot_path).unwrap();
        assert_eq!(checkpoint_id, cycle as u64);

        // Verify WAL is truncated
        let status = store.wal_status().unwrap();
        assert_eq!(status.entry_count, 0);
    }

    // Final count
    assert_eq!(store.len(), 30);
}

#[test]
fn test_tensor_store_wal_recover_empty() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("test.wal");

    // Create empty WAL
    {
        let _store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();
    }

    // Recover from empty WAL
    let recovered = TensorStore::recover(&wal_path, &WalConfig::default(), None).unwrap();
    assert!(recovered.is_empty());
}

#[test]
fn test_slab_router_wal_recovery() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("router.wal");

    // Test at SlabRouter level
    {
        let router = SlabRouter::with_wal(&wal_path, WalConfig::default()).unwrap();

        let mut data = TensorData::new();
        data.set("value", TensorValue::Scalar(ScalarValue::Int(42)));
        router.put_durable("test_key", data).unwrap();
    }

    // Recover
    let recovered = SlabRouter::recover(&wal_path, &WalConfig::default(), None).unwrap();
    assert!(recovered.exists("test_key"));
}

#[test]
fn test_wal_entry_types_roundtrip() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("entries.wal");

    // Test various entry types through the store
    {
        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

        // Metadata with various value types
        let mut tensor = TensorData::new();
        tensor.set("null", TensorValue::Scalar(ScalarValue::Null));
        tensor.set("bool", TensorValue::Scalar(ScalarValue::Bool(true)));
        tensor.set("int", TensorValue::Scalar(ScalarValue::Int(42)));
        tensor.set("float", TensorValue::Scalar(ScalarValue::Float(3.14)));
        tensor.set(
            "string",
            TensorValue::Scalar(ScalarValue::String("hello".into())),
        );
        tensor.set(
            "bytes",
            TensorValue::Scalar(ScalarValue::Bytes(vec![1, 2, 3])),
        );
        tensor.set("vector", TensorValue::Vector(vec![1.0, 2.0, 3.0]));
        tensor.set("pointer", TensorValue::Pointer("ref:123".into()));
        tensor.set(
            "pointers",
            TensorValue::Pointers(vec!["a".into(), "b".into()]),
        );

        store.put_durable("complex", tensor).unwrap();
    }

    // Recover and verify all types
    let recovered = TensorStore::recover(&wal_path, &WalConfig::default(), None).unwrap();
    let tensor = recovered.get("complex").unwrap();

    assert!(matches!(
        tensor.get("null"),
        Some(TensorValue::Scalar(ScalarValue::Null))
    ));
    assert!(matches!(
        tensor.get("bool"),
        Some(TensorValue::Scalar(ScalarValue::Bool(true)))
    ));
    assert!(matches!(
        tensor.get("int"),
        Some(TensorValue::Scalar(ScalarValue::Int(42)))
    ));
    assert!(matches!(
        tensor.get("pointer"),
        Some(TensorValue::Pointer(ref p)) if p == "ref:123"
    ));
}

#[test]
fn test_wal_size_tracking() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("size.wal");

    let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();

    let initial_status = store.wal_status().unwrap();
    assert_eq!(initial_status.entry_count, 0);

    // Add data
    for i in 0..10 {
        store
            .put_durable(format!("key:{}", i), TensorData::new())
            .unwrap();
    }

    let status = store.wal_status().unwrap();
    assert_eq!(status.entry_count, 10);
    assert!(status.size_bytes > 0);
}

#[test]
fn test_wal_recovery_with_bloom_filter() {
    let dir = tempdir().unwrap();
    let wal_path = dir.path().join("bloom.wal");

    // Create store and add data
    {
        let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();
        for i in 0..100 {
            store
                .put_durable(format!("item:{}", i), TensorData::new())
                .unwrap();
        }
    }

    // Recover with bloom filter
    let recovered =
        TensorStore::recover_with_bloom(&wal_path, &WalConfig::default(), None, 1000, 0.01)
            .unwrap();

    assert!(recovered.has_bloom_filter());
    assert_eq!(recovered.len(), 100);

    // Bloom filter should accelerate negative lookups
    assert!(!recovered.exists("nonexistent:1"));
    assert!(!recovered.exists("nonexistent:2"));
}
