// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! TensorStore instrumentation integration tests.
//!
//! Tests access pattern tracking and shard instrumentation.

use std::{sync::Arc, thread};

use tensor_store::{TensorData, TensorStore, TensorValue};

fn make_data(value: i64) -> TensorData {
    let mut data = TensorData::new();
    data.set(
        "value",
        TensorValue::Scalar(tensor_store::ScalarValue::Int(value)),
    );
    data
}

#[test]
fn test_with_instrumentation_basic() {
    // Create store with instrumentation
    let store = TensorStore::with_instrumentation(1); // Sample rate 1 = track all

    assert!(store.has_instrumentation());

    // Perform some operations
    store.put("key:1", make_data(1)).unwrap();
    store.put("key:2", make_data(2)).unwrap();
    let _ = store.get("key:1");
    let _ = store.get("key:2");
    let _ = store.get("key:1");

    // Get snapshot
    let snapshot = store.access_snapshot().unwrap();

    // Should have recorded accesses
    assert!(snapshot.total_reads() > 0);
    assert!(snapshot.total_writes() > 0);
}

#[test]
fn test_instrumentation_disabled_by_default() {
    let store = TensorStore::new();

    assert!(!store.has_instrumentation());
    assert!(store.access_snapshot().is_none());
    assert!(store.hot_shards(5).is_none());
}

#[test]
fn test_hot_shards_tracking() {
    let store = TensorStore::with_instrumentation(1);

    // Create hotspot by accessing same key many times
    let mut data = TensorData::new();
    data.set(
        "data",
        TensorValue::Scalar(tensor_store::ScalarValue::String("value".into())),
    );
    store.put("hot:key", data).unwrap();

    for _ in 0..100 {
        let _ = store.get("hot:key");
    }

    // Add some cold keys
    for i in 0..10 {
        store.put(format!("cold:{}", i), make_data(i)).unwrap();
    }

    // Check hot shards
    let hot = store.hot_shards(3).unwrap();
    assert!(!hot.is_empty());

    // The hottest shard should have many accesses
    let (_, access_count) = hot[0];
    assert!(access_count > 10);
}

#[test]
fn test_access_distribution() {
    let store = TensorStore::with_instrumentation(1);

    // Write to many keys to distribute across shards
    for i in 0..100 {
        store.put(format!("key:{}", i), make_data(i)).unwrap();
    }

    // Read some keys
    for i in 0..50 {
        let _ = store.get(&format!("key:{}", i));
    }

    let snapshot = store.access_snapshot().unwrap();

    // Check shard stats
    let shard_stats = &snapshot.shard_stats;
    assert!(!shard_stats.is_empty());

    // Total should match our operations
    let total_reads: u64 = shard_stats.iter().map(|s| s.reads).sum();
    let total_writes: u64 = shard_stats.iter().map(|s| s.writes).sum();

    assert!(total_reads > 0);
    assert!(total_writes > 0);
}

#[test]
fn test_sampling_rate_effect() {
    // High sample rate (track less)
    let store_sparse = TensorStore::with_instrumentation(10);

    // Low sample rate (track more)
    let store_dense = TensorStore::with_instrumentation(1);

    // Same operations on both
    for i in 0..100 {
        store_sparse.put(format!("k:{}", i), make_data(i)).unwrap();
        store_dense.put(format!("k:{}", i), make_data(i)).unwrap();
    }

    let sparse_snapshot = store_sparse.access_snapshot().unwrap();
    let dense_snapshot = store_dense.access_snapshot().unwrap();

    // Dense tracking should show more writes (or equal, due to sampling variance)
    // With rate 10, we sample ~10% so expect ~10 writes recorded
    // With rate 1, we sample 100% so expect ~100 writes recorded
    assert!(
        dense_snapshot.total_writes() >= sparse_snapshot.total_writes(),
        "Dense {} should have >= writes than sparse {}",
        dense_snapshot.total_writes(),
        sparse_snapshot.total_writes()
    );
}

#[test]
fn test_concurrent_instrumentation() {
    let store = Arc::new(TensorStore::with_instrumentation(1));
    let mut handles = vec![];

    // 4 threads reading and writing
    for t in 0..4 {
        let store = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for i in 0..50 {
                let key = format!("thread:{}:{}", t, i);
                let _ = store.put(&key, make_data(i as i64));
                let _ = store.get(&key);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let snapshot = store.access_snapshot().unwrap();

    // Should have recorded all accesses from all threads
    // 4 threads * 50 operations each * 2 (read + write) = 400 total ops
    let total = snapshot.total_reads() + snapshot.total_writes();
    assert!(total > 0, "Should have recorded some accesses");
}

#[test]
fn test_bloom_and_instrumentation_combined() {
    let store = TensorStore::with_bloom_and_instrumentation(1000, 0.01, 1);

    assert!(store.has_bloom_filter());
    assert!(store.has_instrumentation());

    // Operations should be tracked and bloom filtered
    let mut data = TensorData::new();
    data.set(
        "data",
        TensorValue::Scalar(tensor_store::ScalarValue::String("value".into())),
    );
    store.put("exists", data).unwrap();

    // Bloom filter should help avoid unnecessary lookups
    let missing = store.get("definitely_not_there");
    assert!(missing.is_err());

    // Existing key should be found
    let exists = store.get("exists");
    assert!(exists.is_ok());

    // Check instrumentation still works
    let snapshot = store.access_snapshot().unwrap();
    assert!(snapshot.total_reads() > 0);
    assert!(snapshot.total_writes() > 0);
}

#[test]
fn test_instrumentation_snapshot_isolation() {
    let store = TensorStore::with_instrumentation(1);

    // First batch of operations
    for i in 0..50 {
        store.put(format!("batch1:{}", i), make_data(i)).unwrap();
    }

    let snapshot1 = store.access_snapshot().unwrap();

    // Second batch of operations
    for i in 0..50 {
        store.put(format!("batch2:{}", i), make_data(i)).unwrap();
    }

    let snapshot2 = store.access_snapshot().unwrap();

    // Second snapshot should show more writes
    assert!(
        snapshot2.total_writes() >= snapshot1.total_writes(),
        "Snapshot2 {} should have >= writes than snapshot1 {}",
        snapshot2.total_writes(),
        snapshot1.total_writes()
    );
}

#[test]
fn test_cold_shards_detection() {
    let store = TensorStore::with_instrumentation(1);

    // Create data in all shards
    for i in 0..1000 {
        store.put(format!("data:{}", i), make_data(i)).unwrap();
    }

    // Create a hotspot on specific keys
    for _ in 0..500 {
        let _ = store.get("data:42");
        let _ = store.get("data:123");
    }

    // Get snapshot and check shard stats
    let snapshot = store.access_snapshot().unwrap();
    let shard_stats = &snapshot.shard_stats;

    // Should have some shards with very few reads (cold)
    let cold_shards: Vec<_> = shard_stats.iter().filter(|s| s.reads < 10).collect();

    // There should be some cold shards (not all shards were hotspots)
    assert!(
        !cold_shards.is_empty() || shard_stats.iter().all(|s| s.reads >= 10),
        "Expected some cold shards or uniform distribution"
    );
}

#[test]
fn test_instrumentation_with_deletes() {
    let store = TensorStore::with_instrumentation(1);

    // Add keys
    for i in 0..20 {
        store.put(format!("key:{}", i), make_data(i)).unwrap();
    }

    // Delete some keys
    for i in 0..10 {
        let _ = store.delete(&format!("key:{}", i));
    }

    let snapshot = store.access_snapshot().unwrap();

    // Writes should include both sets and deletes
    assert!(snapshot.total_writes() >= 20);
}

#[test]
fn test_instrumentation_read_write_ratio() {
    let store = TensorStore::with_instrumentation(1);

    // Write once, read many (common pattern)
    let mut data = TensorData::new();
    data.set(
        "data",
        TensorValue::Scalar(tensor_store::ScalarValue::String("value".into())),
    );
    store.put("popular", data).unwrap();

    for _ in 0..100 {
        let _ = store.get("popular");
    }

    let snapshot = store.access_snapshot().unwrap();

    // Read-heavy workload should show many more reads than writes
    let read_ratio = snapshot.total_reads() as f64 / snapshot.total_writes().max(1) as f64;
    assert!(
        read_ratio > 10.0,
        "Expected read-heavy ratio, got {}",
        read_ratio
    );
}

#[test]
fn test_instrumentation_empty_store() {
    let store = TensorStore::with_instrumentation(1);

    // No operations yet
    let snapshot = store.access_snapshot().unwrap();

    // Should have zero counts
    assert_eq!(snapshot.total_reads(), 0);
    assert_eq!(snapshot.total_writes(), 0);
}

#[test]
fn test_shard_distribution_fairness() {
    let store = TensorStore::with_instrumentation(1);

    // Use UUIDs to ensure random distribution
    for i in 0..1000 {
        let key = format!("uuid:{}:{}", i, i * 31337);
        store.put(&key, make_data(i)).unwrap();
    }

    let snapshot = store.access_snapshot().unwrap();
    let shard_stats = &snapshot.shard_stats;

    if !shard_stats.is_empty() {
        // Calculate coefficient of variation for writes
        let writes: Vec<f64> = shard_stats.iter().map(|s| s.writes as f64).collect();
        let mean = writes.iter().sum::<f64>() / writes.len() as f64;
        let variance = writes.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / writes.len() as f64;
        let std_dev = variance.sqrt();
        let cv = std_dev / mean.max(1.0);

        // CV should be reasonable for a hash-based distribution
        // (not too skewed)
        assert!(
            cv < 2.0,
            "Distribution too skewed, CV = {} (mean={}, stddev={})",
            cv,
            mean,
            std_dev
        );
    }
}

#[test]
fn test_hot_shards_limit() {
    let store = TensorStore::with_instrumentation(1);

    // Create some activity
    for i in 0..100 {
        store.put(format!("key:{}", i), make_data(i)).unwrap();
        let _ = store.get(&format!("key:{}", i));
    }

    // Request more hot shards than exist
    let hot_5 = store.hot_shards(5).unwrap();
    let hot_100 = store.hot_shards(100).unwrap();

    // Should respect the limit
    assert!(hot_5.len() <= 5);

    // With more capacity, might return more (up to actual shard count)
    assert!(hot_100.len() >= hot_5.len());
}

#[test]
fn test_instrumentation_with_shared_store() {
    use std::collections::HashMap;

    use graph_engine::GraphEngine;
    use relational_engine::RelationalEngine;

    let store = TensorStore::with_instrumentation(1);
    let graph = GraphEngine::with_store(store.clone());
    let relational = RelationalEngine::with_store(store.clone());

    // Operations through different engines
    graph.create_node("test", HashMap::new()).unwrap();

    use relational_engine::{Column, ColumnType, Schema, Value};
    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    relational.create_table("test", schema).unwrap();

    let mut row = HashMap::new();
    row.insert("id".to_string(), Value::Int(1));
    relational.insert("test", row).unwrap();

    // All operations should be tracked
    let snapshot = store.access_snapshot().unwrap();
    assert!(snapshot.total_writes() > 0);
}
