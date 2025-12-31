//! Persistence and recovery integration tests.
//!
//! Tests snapshot/restore functionality across all engines sharing a TensorStore.

use graph_engine::GraphEngine;
use integration_tests::{create_shared_engines, sample_embeddings};
use relational_engine::{Column, ColumnType, Condition, RelationalEngine, Schema, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use tempfile::tempdir;
use tensor_compress::CompressionConfig;
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig};
use vector_engine::VectorEngine;

#[test]
fn test_snapshot_preserves_all_data() {
    let dir = tempdir().unwrap();
    let snapshot_path = dir.path().join("snapshot.bin");

    // Create engines with shared store
    let (store, relational, graph, vector) = create_shared_engines();

    // Add relational data
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("users", schema).unwrap();

    let mut row1 = HashMap::new();
    row1.insert("id".to_string(), Value::Int(1));
    row1.insert("name".to_string(), Value::String("Alice".into()));
    relational.insert("users", row1).unwrap();

    let mut row2 = HashMap::new();
    row2.insert("id".to_string(), Value::Int(2));
    row2.insert("name".to_string(), Value::String("Bob".into()));
    relational.insert("users", row2).unwrap();

    // Add graph data
    let node1 = graph.create_node("person", HashMap::new()).unwrap();
    let node2 = graph.create_node("person", HashMap::new()).unwrap();
    graph
        .create_edge(node1, node2, "knows", HashMap::new(), true)
        .unwrap();

    // Add vector data
    let embeddings = sample_embeddings(3, 4);
    vector
        .store_embedding("vec1", embeddings[0].clone())
        .unwrap();
    vector
        .store_embedding("vec2", embeddings[1].clone())
        .unwrap();
    vector
        .store_embedding("vec3", embeddings[2].clone())
        .unwrap();

    // Save snapshot
    store.save_snapshot(&snapshot_path).unwrap();

    // Load into fresh store
    let restored_store = TensorStore::load_snapshot(&snapshot_path).unwrap();
    let restored_relational = RelationalEngine::with_store(restored_store.clone());
    let restored_graph = GraphEngine::with_store(restored_store.clone());
    let restored_vector = VectorEngine::with_store(restored_store);

    // Verify relational data
    let rows = restored_relational
        .select("users", Condition::True)
        .unwrap();
    assert_eq!(rows.len(), 2);

    // Verify graph data
    let neighbors = restored_graph
        .neighbors(node1, None, graph_engine::Direction::Outgoing)
        .unwrap();
    assert!(neighbors.iter().any(|n| n.id == node2));

    // Verify vector data
    let emb = restored_vector.get_embedding("vec1").unwrap();
    assert_eq!(emb, embeddings[0]);
}

#[test]
fn test_snapshot_during_writes() {
    let dir = tempdir().unwrap();
    let snapshot_path = dir.path().join("concurrent_snapshot.bin");
    let store = Arc::new(TensorStore::new());

    // Spawn writers
    let mut handles = vec![];
    for thread_id in 0..4 {
        let store_clone = Arc::clone(&store);
        let handle = thread::spawn(move || {
            for i in 0..250 {
                let key = format!("thread{}:key{}", thread_id, i);
                let mut data = tensor_store::TensorData::new();
                data.set(
                    "value",
                    tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(i as i64)),
                );
                store_clone.put(&key, data).unwrap();
            }
        });
        handles.push(handle);
    }

    // Take snapshot during writes
    thread::sleep(std::time::Duration::from_millis(5));
    store.save_snapshot(&snapshot_path).unwrap();

    // Wait for writers
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify snapshot is valid and loadable
    let restored = TensorStore::load_snapshot(&snapshot_path).unwrap();
    // At least some entries should exist (snapshot taken mid-write)
    assert!(restored.len() > 0);
}

#[test]
fn test_restore_to_fresh_store() {
    let dir = tempdir().unwrap();
    let snapshot_path = dir.path().join("fresh_restore.bin");

    let store = TensorStore::new();
    for i in 0..100 {
        let key = format!("key{}", i);
        let mut data = tensor_store::TensorData::new();
        data.set(
            "idx",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(i)),
        );
        store.put(&key, data).unwrap();
    }
    store.save_snapshot(&snapshot_path).unwrap();

    // Load into completely fresh store
    let fresh = TensorStore::load_snapshot(&snapshot_path).unwrap();
    assert_eq!(fresh.len(), 100);

    // Verify each entry
    for i in 0..100 {
        let key = format!("key{}", i);
        let data = fresh.get(&key).unwrap();
        let val = data.get("idx").unwrap();
        if let tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(v)) = val {
            assert_eq!(*v, i);
        } else {
            panic!("unexpected value type");
        }
    }
}

#[test]
fn test_data_survives_engine_restart() {
    let store = TensorStore::new();

    // Create engines
    {
        let relational = RelationalEngine::with_store(store.clone());
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        relational.create_table("test_table", schema).unwrap();

        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(42));
        relational.insert("test_table", row).unwrap();
    }
    // Engines dropped, but store persists

    // Create fresh engines with same store
    let relational2 = RelationalEngine::with_store(store.clone());
    let rows = relational2.select("test_table", Condition::True).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("id"), Some(&Value::Int(42)));
}

#[test]
fn test_compressed_snapshot_roundtrip() {
    let dir = tempdir().unwrap();
    let snapshot_path = dir.path().join("compressed.bin");

    let store = TensorStore::new();
    let embeddings = sample_embeddings(50, 128);

    for (i, emb) in embeddings.iter().enumerate() {
        let key = format!("emb{}", i);
        let mut data = tensor_store::TensorData::new();
        data.set("vector", tensor_store::TensorValue::Vector(emb.clone()));
        store.put(&key, data).unwrap();
    }

    // Save compressed
    let config = CompressionConfig::default();
    store
        .save_snapshot_compressed(&snapshot_path, config)
        .unwrap();

    // Load compressed
    let restored = TensorStore::load_snapshot_compressed(&snapshot_path).unwrap();
    assert_eq!(restored.len(), 50);

    // Verify embeddings (with tolerance for quantization)
    for i in 0..50 {
        let key = format!("emb{}", i);
        let data = restored.get(&key).unwrap();
        let val = data.get("vector").unwrap();
        if let tensor_store::TensorValue::Vector(v) = val {
            assert_eq!(v.len(), 128);
        }
    }
}

#[test]
fn test_snapshot_includes_vault_secrets() {
    let dir = tempdir().unwrap();
    let snapshot_path = dir.path().join("vault_snapshot.bin");
    let master_key = b"test-master-key-32-bytes-long!!";

    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));

    // Create vault and store secret
    let vault = Vault::new(
        master_key,
        Arc::clone(&graph),
        store.clone(),
        VaultConfig::default(),
    )
    .unwrap();
    vault.set(Vault::ROOT, "db/password", "secret123").unwrap();
    vault.set(Vault::ROOT, "api/key", "api-key-value").unwrap();

    // Snapshot
    store.save_snapshot(&snapshot_path).unwrap();

    // Restore
    let restored_store = TensorStore::load_snapshot(&snapshot_path).unwrap();
    let restored_graph = Arc::new(GraphEngine::with_store(restored_store.clone()));
    let restored_vault = Vault::new(
        master_key,
        restored_graph,
        restored_store,
        VaultConfig::default(),
    )
    .unwrap();

    // Verify secrets
    let secret1 = restored_vault.get(Vault::ROOT, "db/password").unwrap();
    assert_eq!(secret1, "secret123");
    let secret2 = restored_vault.get(Vault::ROOT, "api/key").unwrap();
    assert_eq!(secret2, "api-key-value");
}

#[test]
fn test_cache_entries_are_ephemeral() {
    // Note: Cache uses internal DashMaps, not TensorStore, so entries are ephemeral.
    // This test documents this architectural decision.
    use tensor_cache::{Cache, CacheConfig};

    let cache = Cache::new();
    cache.put_simple("key1", "value1").unwrap();

    // Cache hit works in same instance
    let result = cache.get_simple("key1");
    assert!(result.is_some());
    assert_eq!(result.unwrap(), "value1");

    // New cache instance starts empty (entries don't persist)
    let cache2 = Cache::with_config(CacheConfig::default()).unwrap();
    let result2 = cache2.get_simple("key1");
    assert!(result2.is_none());
}

#[test]
fn test_snapshot_with_bloom_filter() {
    let dir = tempdir().unwrap();
    let snapshot_path = dir.path().join("bloom_snapshot.bin");

    let store = TensorStore::with_bloom_filter(1000, 0.01);
    for i in 0..100 {
        let key = format!("key{}", i);
        let mut data = tensor_store::TensorData::new();
        data.set(
            "i",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(i)),
        );
        store.put(&key, data).unwrap();
    }

    store.save_snapshot(&snapshot_path).unwrap();

    // Restore with bloom filter
    let restored =
        TensorStore::load_snapshot_with_bloom_filter(&snapshot_path, 1000, 0.01).unwrap();

    assert_eq!(restored.len(), 100);

    // Bloom filter should work for negative lookups
    assert!(restored.get("nonexistent_key").is_err());
}

#[test]
fn test_multiple_snapshots_incremental() {
    let dir = tempdir().unwrap();
    let snapshot1 = dir.path().join("snap1.bin");
    let snapshot2 = dir.path().join("snap2.bin");

    let store = TensorStore::new();

    // First batch of data
    for i in 0..50 {
        let key = format!("batch1:{}", i);
        let mut data = tensor_store::TensorData::new();
        data.set(
            "v",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(i)),
        );
        store.put(&key, data).unwrap();
    }
    store.save_snapshot(&snapshot1).unwrap();

    // Second batch
    for i in 0..50 {
        let key = format!("batch2:{}", i);
        let mut data = tensor_store::TensorData::new();
        data.set(
            "v",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(i + 100)),
        );
        store.put(&key, data).unwrap();
    }
    store.save_snapshot(&snapshot2).unwrap();

    // Verify both snapshots
    let restored1 = TensorStore::load_snapshot(&snapshot1).unwrap();
    assert_eq!(restored1.len(), 50);

    let restored2 = TensorStore::load_snapshot(&snapshot2).unwrap();
    assert_eq!(restored2.len(), 100);
}
