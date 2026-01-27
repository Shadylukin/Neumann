//! Concurrency and stress integration tests.
//!
//! Tests multi-threaded and async access patterns across all engines.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
};

use futures::future::join_all;
use graph_engine::{GraphEngine, PropertyValue};
use integration_tests::{create_shared_engines_arc, sample_embeddings};
use relational_engine::{Column, ColumnType, Schema, Value};
use tensor_blob::{BlobConfig, BlobStore, PutOptions};
use tensor_cache::Cache;
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig};
use tokio::sync::Barrier;
use vector_engine::VectorEngine;

#[test]
fn test_concurrent_writes_all_engines() {
    let (_store, relational, graph, vector) = create_shared_engines_arc();

    // Create schema before spawning threads
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("users", schema).unwrap();

    let mut handles = vec![];
    let writes_per_thread = 100;

    // Relational writers
    for t in 0..2 {
        let rel = Arc::clone(&relational);
        let handle = thread::spawn(move || {
            for i in 0..writes_per_thread {
                let id = (t * 1000 + i) as i64;
                let mut row = HashMap::new();
                row.insert("id".to_string(), Value::Int(id));
                row.insert("name".to_string(), Value::String(format!("user{}", id)));
                rel.insert("users", row).unwrap();
            }
        });
        handles.push(handle);
    }

    // Graph writers
    for t in 0..2 {
        let g = Arc::clone(&graph);
        let handle = thread::spawn(move || {
            for i in 0..writes_per_thread {
                let mut props = HashMap::new();
                props.insert("thread".to_string(), PropertyValue::Int(t as i64));
                props.insert("idx".to_string(), PropertyValue::Int(i as i64));
                g.create_node("test_node", props).unwrap();
            }
        });
        handles.push(handle);
    }

    // Vector writers
    for t in 0..2 {
        let v = Arc::clone(&vector);
        let embeddings = sample_embeddings(writes_per_thread, 8);
        let handle = thread::spawn(move || {
            for (i, emb) in embeddings.into_iter().enumerate() {
                let key = format!("thread{}:emb{}", t, i);
                v.store_embedding(&key, emb).unwrap();
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify counts
    let rows = relational
        .select("users", relational_engine::Condition::True)
        .unwrap();
    assert_eq!(rows.len(), 2 * writes_per_thread);

    // Count graph nodes
    let mut node_count = 0;
    for id in 0..10000 {
        if graph.get_node(id).is_ok() {
            node_count += 1;
        }
    }
    assert_eq!(node_count, 2 * writes_per_thread);
}

#[test]
fn test_shared_store_contention() {
    let store = Arc::new(TensorStore::new());
    let key = "contended_key";
    let iterations = 1000;
    let num_threads = 4;

    let success_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    for thread_id in 0..num_threads {
        let store_clone = Arc::clone(&store);
        let success_count_clone = Arc::clone(&success_count);
        let handle = thread::spawn(move || {
            for i in 0..iterations {
                let mut data = tensor_store::TensorData::new();
                data.set(
                    "writer",
                    tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(
                        thread_id as i64,
                    )),
                );
                data.set(
                    "iteration",
                    tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(i as i64)),
                );
                store_clone.put(key, data).unwrap();
                success_count_clone.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All writes should succeed
    assert_eq!(
        success_count.load(Ordering::SeqCst),
        num_threads * iterations
    );

    // Final value should be from one of the threads
    let final_data = store.get(key).unwrap();
    let writer = final_data.get("writer").unwrap();
    if let tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(w)) = writer {
        assert!(*w < num_threads as i64);
    }
}

#[test]
fn test_reader_writer_isolation() {
    let store = Arc::new(TensorStore::new());
    let iterations = 500;

    // Pre-populate some data
    for i in 0..100 {
        let key = format!("key{}", i);
        let mut data = tensor_store::TensorData::new();
        data.set(
            "value",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(i)),
        );
        store.put(&key, data).unwrap();
    }

    let read_count = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    // Spawn readers
    for _ in 0..2 {
        let store_clone = Arc::clone(&store);
        let read_count_clone = Arc::clone(&read_count);
        let handle = thread::spawn(move || {
            for i in 0..iterations {
                let key = format!("key{}", i % 100);
                if store_clone.get(&key).is_ok() {
                    read_count_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }

    // Spawn writers
    for t in 0..2 {
        let store_clone = Arc::clone(&store);
        let write_count_clone = Arc::clone(&write_count);
        let handle = thread::spawn(move || {
            for i in 0..iterations {
                let key = format!("key{}", i % 100);
                let mut data = tensor_store::TensorData::new();
                data.set(
                    "value",
                    tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(
                        (t * 1000 + i) as i64,
                    )),
                );
                store_clone.put(&key, data).unwrap();
                write_count_clone.fetch_add(1, Ordering::SeqCst);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All operations should complete
    assert_eq!(read_count.load(Ordering::SeqCst), 2 * iterations);
    assert_eq!(write_count.load(Ordering::SeqCst), 2 * iterations);
}

#[test]
fn test_vault_concurrent_access_checks() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    let vault = Arc::new(
        Vault::new(
            master_key,
            Arc::clone(&graph),
            store.clone(),
            VaultConfig::default(),
        )
        .unwrap(),
    );

    // Store some secrets
    vault.set(Vault::ROOT, "secret1", "value1").unwrap();
    vault.set(Vault::ROOT, "secret2", "value2").unwrap();
    vault.set(Vault::ROOT, "secret3", "value3").unwrap();

    let access_count = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    // Concurrent reads
    for _ in 0..4 {
        let v = Arc::clone(&vault);
        let count = Arc::clone(&access_count);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let key = format!("secret{}", (i % 3) + 1);
                if v.get(Vault::ROOT, &key).is_ok() {
                    count.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All reads should succeed (root has access to all)
    assert_eq!(access_count.load(Ordering::SeqCst), 4 * 100);
}

#[test]
fn test_cache_concurrent_lookups() {
    let cache = Arc::new(Cache::new());

    // Pre-populate cache
    for i in 0..50 {
        let key = format!("key{}", i);
        let value = format!("value{}", i);
        cache.put_simple(&key, &value).unwrap();
    }

    let hit_count = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    // Concurrent lookups
    for _ in 0..4 {
        let c = Arc::clone(&cache);
        let hits = Arc::clone(&hit_count);
        let handle = thread::spawn(move || {
            for i in 0..200 {
                let key = format!("key{}", i % 50);
                if c.get_simple(&key).is_some() {
                    hits.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All lookups should hit
    assert_eq!(hit_count.load(Ordering::SeqCst), 4 * 200);
}

#[tokio::test]
async fn test_blob_parallel_uploads() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = Arc::new(tokio::sync::Mutex::new(
        BlobStore::new(store, config).await.unwrap(),
    ));

    let num_uploads = 10;
    let barrier = Arc::new(Barrier::new(num_uploads));

    let mut handles = vec![];
    for i in 0..num_uploads {
        let b = Arc::clone(&blob);
        let bar = Arc::clone(&barrier);
        let handle = tokio::spawn(async move {
            // Wait for all tasks to be ready
            bar.wait().await;

            let data = format!("content for blob {}", i).into_bytes();
            let filename = format!("file{}.txt", i);
            let blob_guard = b.lock().await;
            blob_guard
                .put(&filename, &data, PutOptions::default())
                .await
                .unwrap()
        });
        handles.push(handle);
    }

    let artifact_ids: Vec<String> = join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();

    // Verify all uploads succeeded with unique IDs
    assert_eq!(artifact_ids.len(), num_uploads);
    let unique_ids: std::collections::HashSet<_> = artifact_ids.iter().collect();
    assert_eq!(unique_ids.len(), num_uploads);
}

#[test]
fn test_high_cardinality_inserts() {
    let store = Arc::new(TensorStore::new());
    let num_entries = 100_000;
    let num_threads = 4;
    let entries_per_thread = num_entries / num_threads;

    let mut handles = vec![];
    for t in 0..num_threads {
        let s = Arc::clone(&store);
        let handle = thread::spawn(move || {
            for i in 0..entries_per_thread {
                let key = format!("entity:{}:{}", t, i);
                let mut data = tensor_store::TensorData::new();
                data.set(
                    "idx",
                    tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(
                        (t * entries_per_thread + i) as i64,
                    )),
                );
                s.put(&key, data).unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(store.len(), num_entries);
}

#[test]
fn test_deep_graph_traversal_concurrent() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store));

    // Create a chain of nodes
    let chain_length = 100;
    let start_id = graph.create_node("start", HashMap::new()).unwrap();
    let mut prev_id = start_id;
    for i in 1..chain_length {
        let mut props = HashMap::new();
        props.insert("depth".to_string(), PropertyValue::Int(i as i64));
        let node_id = graph.create_node("chain", props).unwrap();
        graph
            .create_edge(prev_id, node_id, "next", HashMap::new(), true)
            .unwrap();
        prev_id = node_id;
    }
    let end_id = prev_id;

    let path_found = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    // Concurrent path queries
    for _ in 0..4 {
        let g = Arc::clone(&graph);
        let found = Arc::clone(&path_found);
        let handle = thread::spawn(move || {
            for _ in 0..10 {
                // Find path from start to end
                if let Ok(path) = g.find_path(start_id, end_id, None) {
                    if !path.nodes.is_empty() {
                        found.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All path queries should succeed
    assert_eq!(path_found.load(Ordering::SeqCst), 40);
}

#[test]
fn test_vector_search_during_index_build() {
    let store = TensorStore::new();
    let vector = Arc::new(VectorEngine::with_store(store));

    // Pre-populate embeddings
    let embeddings = sample_embeddings(1000, 32);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("emb{}", i), emb.clone())
            .unwrap();
    }

    let search_count = Arc::new(AtomicUsize::new(0));

    // Spawn search threads
    let mut handles = vec![];
    for _ in 0..2 {
        let v = Arc::clone(&vector);
        let query = embeddings[0].clone();
        let count = Arc::clone(&search_count);
        let handle = thread::spawn(move || {
            for _ in 0..50 {
                let results = v.search_similar(&query, 10).unwrap();
                if !results.is_empty() {
                    count.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }

    // Concurrent writes
    for t in 0..2 {
        let v = Arc::clone(&vector);
        let handle = thread::spawn(move || {
            let new_embeddings = sample_embeddings(100, 32);
            for (i, emb) in new_embeddings.into_iter().enumerate() {
                let key = format!("new_emb_{}_{}", t, i);
                v.store_embedding(&key, emb).unwrap();
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All searches should find results
    assert_eq!(search_count.load(Ordering::SeqCst), 100);

    // Verify search still works after concurrent writes
    let results = vector.search_similar(&embeddings[0], 10).unwrap();
    assert!(!results.is_empty());
}

#[tokio::test]
async fn test_async_blob_concurrent_read_write() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = Arc::new(tokio::sync::Mutex::new(
        BlobStore::new(store, config).await.unwrap(),
    ));

    // Upload initial blobs
    let initial_ids: Vec<String> = {
        let guard = blob.lock().await;
        let mut ids = vec![];
        for i in 0..5 {
            let data = format!("initial content {}", i).into_bytes();
            let id = guard
                .put(&format!("initial{}.txt", i), &data, PutOptions::default())
                .await
                .unwrap();
            ids.push(id);
        }
        ids
    };

    let read_count = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    // Concurrent readers
    for _ in 0..2 {
        let b = Arc::clone(&blob);
        let ids = initial_ids.clone();
        let count = Arc::clone(&read_count);
        let handle = tokio::spawn(async move {
            for _ in 0..10 {
                for id in &ids {
                    let guard = b.lock().await;
                    if guard.get(id).await.is_ok() {
                        count.fetch_add(1, Ordering::SeqCst);
                    }
                }
            }
        });
        handles.push(handle);
    }

    // Concurrent writers
    for t in 0..2 {
        let b = Arc::clone(&blob);
        let count = Arc::clone(&write_count);
        let handle = tokio::spawn(async move {
            for i in 0..10 {
                let data = format!("new content {} from thread {}", i, t).into_bytes();
                let guard = b.lock().await;
                if guard
                    .put(
                        &format!("new_{}_{}.txt", t, i),
                        &data,
                        PutOptions::default(),
                    )
                    .await
                    .is_ok()
                {
                    count.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }

    join_all(handles).await;

    // All operations should succeed
    assert_eq!(read_count.load(Ordering::SeqCst), 2 * 10 * 5);
    assert_eq!(write_count.load(Ordering::SeqCst), 2 * 10);
}

#[test]
fn test_graph_concurrent_batch_nodes_and_edges() {
    use std::sync::Barrier;

    let (_store, _relational, graph, _vector) = create_shared_engines_arc();

    // Pre-create some nodes for edge targets
    let mut base_nodes = Vec::new();
    for i in 0..100 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        base_nodes.push(graph.create_node("Base", props).unwrap());
    }
    let base_nodes = Arc::new(base_nodes);

    let thread_count = 20;
    let barrier = Arc::new(Barrier::new(thread_count));
    let node_success = Arc::new(AtomicUsize::new(0));
    let edge_success = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    // 10 threads batch creating nodes
    for t in 0..10 {
        let g = Arc::clone(&graph);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&node_success);
        handles.push(thread::spawn(move || {
            bar.wait();
            let nodes: Vec<graph_engine::NodeInput> = (0..50)
                .map(|i| graph_engine::NodeInput {
                    labels: vec!["Batch".to_string()],
                    properties: {
                        let mut p = HashMap::new();
                        p.insert(
                            "thread".to_string(),
                            PropertyValue::Int(t as i64),
                        );
                        p.insert("idx".to_string(), PropertyValue::Int(i as i64));
                        p
                    },
                })
                .collect();
            if let Ok(result) = g.batch_create_nodes(nodes) {
                cnt.fetch_add(result.created_ids.len(), Ordering::SeqCst);
            }
        }));
    }

    // 10 threads batch creating edges
    for t in 0..10 {
        let g = Arc::clone(&graph);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&edge_success);
        let bases = Arc::clone(&base_nodes);
        handles.push(thread::spawn(move || {
            bar.wait();
            let edges: Vec<graph_engine::EdgeInput> = (0..10)
                .map(|i| graph_engine::EdgeInput {
                    from: bases[t * 10 % 100],
                    to: bases[(t * 10 + i + 1) % 100],
                    edge_type: "LINK".to_string(),
                    properties: HashMap::new(),
                    directed: true,
                })
                .collect();
            if let Ok(result) = g.batch_create_edges(edges) {
                cnt.fetch_add(result.count, Ordering::SeqCst);
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    // Verify counts
    assert_eq!(node_success.load(Ordering::SeqCst), 10 * 50);
    assert_eq!(edge_success.load(Ordering::SeqCst), 10 * 10);
}

#[test]
fn test_graph_striped_lock_saturation_100_threads() {
    use std::sync::Barrier;

    let graph = Arc::new(GraphEngine::new());
    graph.create_node_property_index("key").unwrap();

    let thread_count = 100;
    let ops_per_thread = 100;
    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let g = Arc::clone(&graph);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            thread::spawn(move || {
                bar.wait();
                for i in 0..ops_per_thread {
                    let mut props = HashMap::new();
                    // Distribute keys to stress all lock stripes
                    let key = format!("{:02x}_thread{t}_op{i}", (t + i) % 256);
                    props.insert("key".to_string(), PropertyValue::String(key));
                    if g.create_node("Saturated", props).is_ok() {
                        cnt.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("thread should not panic");
    }

    let expected = thread_count * ops_per_thread;
    assert_eq!(success_count.load(Ordering::SeqCst), expected);

    // Verify index integrity with a random lookup
    let results = graph
        .find_nodes_by_property(
            "key",
            &PropertyValue::String("00_thread50_op50".into()),
        )
        .unwrap();
    assert_eq!(results.len(), 1);
}

#[test]
fn test_hnsw_concurrent_insert_search() {
    use std::sync::Barrier;

    let vector = Arc::new(VectorEngine::new());

    // Pre-populate with initial embeddings
    let initial = sample_embeddings(500, 128);
    for (i, emb) in initial.iter().enumerate() {
        vector.store_embedding(&format!("init_{i}"), emb.clone()).unwrap();
    }

    let thread_count = 20;
    let barrier = Arc::new(Barrier::new(thread_count));
    let insert_count = Arc::new(AtomicUsize::new(0));
    let search_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];

    // 10 inserter threads
    for t in 0..10 {
        let v = Arc::clone(&vector);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&insert_count);
        let embeddings = sample_embeddings(100, 128);
        handles.push(thread::spawn(move || {
            bar.wait();
            for (i, emb) in embeddings.into_iter().enumerate() {
                let key = format!("thread{t}_emb{i}");
                if v.store_embedding(&key, emb).is_ok() {
                    cnt.fetch_add(1, Ordering::SeqCst);
                }
            }
        }));
    }

    // 10 searcher threads
    for _t in 0..10 {
        let v = Arc::clone(&vector);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&search_count);
        let queries = sample_embeddings(50, 128);
        handles.push(thread::spawn(move || {
            bar.wait();
            for query in queries {
                if v.search_similar(&query, 10).is_ok() {
                    cnt.fetch_add(1, Ordering::SeqCst);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    assert_eq!(insert_count.load(Ordering::SeqCst), 10 * 100);
    assert_eq!(search_count.load(Ordering::SeqCst), 10 * 50);

    // Verify search still works after concurrent operations
    let query = sample_embeddings(1, 128).pop().unwrap();
    let results = vector.search_similar(&query, 10).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_graph_batch_edges_node_deletion_race() {
    use std::sync::Barrier;

    let graph = Arc::new(GraphEngine::new());

    // Create nodes that will be contested
    let mut node_ids = Vec::new();
    for _ in 0..50 {
        node_ids.push(graph.create_node("Contested", HashMap::new()).unwrap());
    }

    let barrier = Arc::new(Barrier::new(2));

    // Thread 1: delete half the nodes
    let g1 = Arc::clone(&graph);
    let bar1 = Arc::clone(&barrier);
    let nodes_to_delete: Vec<_> = node_ids[25..].to_vec();
    let deleter = thread::spawn(move || {
        bar1.wait();
        let mut deleted = 0;
        for id in nodes_to_delete {
            if g1.delete_node(id).is_ok() {
                deleted += 1;
            }
        }
        deleted
    });

    // Thread 2: create edges between all nodes
    let g2 = Arc::clone(&graph);
    let bar2 = Arc::clone(&barrier);
    let edge_creator = thread::spawn(move || {
        bar2.wait();
        let edges: Vec<graph_engine::EdgeInput> = (0..49)
            .map(|i| graph_engine::EdgeInput {
                from: node_ids[i],
                to: node_ids[i + 1],
                edge_type: "CHAIN".to_string(),
                properties: HashMap::new(),
                directed: true,
            })
            .collect();
        g2.batch_create_edges(edges)
    });

    let deleted = deleter.join().expect("deleter should not panic");
    let edge_result = edge_creator.join().expect("edge creator should not panic");

    // Either operation could partially succeed - just verify no panic/corruption
    assert!(deleted <= 25);
    match edge_result {
        Ok(result) => assert!(result.count <= 49),
        Err(_) => { /* Clean error is fine */ }
    }

    // Verify engine is still usable
    let _ = graph.create_node("AfterRace", HashMap::new()).unwrap();
}

#[test]
fn test_graph_update_delete_same_node_race() {
    use std::sync::Barrier;

    let graph = Arc::new(GraphEngine::new());
    let node_id = graph.create_node("RaceTarget", HashMap::new()).unwrap();

    let barrier = Arc::new(Barrier::new(2));

    // Thread 1: try to update the node repeatedly
    let g1 = Arc::clone(&graph);
    let bar1 = Arc::clone(&barrier);
    let updater = thread::spawn(move || {
        bar1.wait();
        let mut updates = 0;
        for i in 0..100 {
            let mut props = HashMap::new();
            props.insert("counter".to_string(), PropertyValue::Int(i));
            if g1.update_node(node_id, None, props).is_ok() {
                updates += 1;
            }
        }
        updates
    });

    // Thread 2: try to delete the node
    let g2 = Arc::clone(&graph);
    let bar2 = Arc::clone(&barrier);
    let deleter = thread::spawn(move || {
        bar2.wait();
        // Small delay to let some updates happen first
        std::thread::sleep(std::time::Duration::from_micros(100));
        g2.delete_node(node_id)
    });

    let updates = updater.join().expect("updater should not panic");
    let delete_result = deleter.join().expect("deleter should not panic");

    // Some updates should succeed, and delete should eventually work
    // or some updates fail after delete - both are valid outcomes
    assert!(updates > 0 || delete_result.is_ok());

    // Verify engine is still usable
    let _ = graph.create_node("AfterRace", HashMap::new()).unwrap();
}
