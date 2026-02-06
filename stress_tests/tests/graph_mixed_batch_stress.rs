// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Graph engine mixed batch operations stress test.
//!
//! Tests concurrent create and delete operations to verify no deadlocks.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
    time::Instant,
};

use graph_engine::{EdgeInput, GraphEngine, NodeInput, PropertyValue};
use stress_tests::{full_config, LatencyHistogram};

#[test]
#[ignore]
fn stress_graph_mixed_batch_100_threads() {
    let _config = full_config();
    let thread_count = 100;
    let ops_per_thread = 20;

    println!("\n=== Graph Mixed Batch 100 Threads ===");
    println!(
        "Threads: {thread_count} (25 each: create_nodes, create_edges, delete_nodes, delete_edges)"
    );
    println!("Operations per thread: {ops_per_thread}");

    let engine = Arc::new(GraphEngine::new());

    // Pre-create a pool of nodes and edges
    let mut node_pool = Vec::new();
    for i in 0..5000 {
        let mut props = HashMap::new();
        props.insert("pool_idx".to_string(), PropertyValue::Int(i));
        node_pool.push(engine.create_node("Pool", props).unwrap());
    }

    let mut edge_pool = Vec::new();
    for i in 0..2000 {
        let from = node_pool[i % 5000];
        let to = node_pool[(i + 1) % 5000];
        edge_pool.push(
            engine
                .create_edge(from, to, "POOL_EDGE", HashMap::new(), true)
                .unwrap(),
        );
    }

    let node_pool = Arc::new(node_pool);
    let edge_pool = Arc::new(edge_pool);

    let barrier = Arc::new(Barrier::new(thread_count));
    let nodes_created = Arc::new(AtomicUsize::new(0));
    let edges_created = Arc::new(AtomicUsize::new(0));
    let nodes_deleted = Arc::new(AtomicUsize::new(0));
    let edges_deleted = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = Vec::new();

    // 25 batch_create_nodes threads
    for t in 0..25 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&nodes_created);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for batch_idx in 0..ops_per_thread {
                let nodes: Vec<NodeInput> = (0..50)
                    .map(|i| NodeInput {
                        labels: vec!["Created".to_string()],
                        properties: {
                            let mut p = HashMap::new();
                            p.insert("t".to_string(), PropertyValue::Int(t as i64));
                            p.insert("b".to_string(), PropertyValue::Int(batch_idx as i64));
                            p.insert("i".to_string(), PropertyValue::Int(i as i64));
                            p
                        },
                    })
                    .collect();

                let op_start = Instant::now();
                if let Ok(result) = eng.batch_create_nodes(nodes) {
                    cnt.fetch_add(result.count, Ordering::Relaxed);
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // 25 batch_create_edges threads
    for t in 0..25 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&edges_created);
        let nodes = Arc::clone(&node_pool);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for batch_idx in 0..ops_per_thread {
                let edges: Vec<EdgeInput> = (0..30)
                    .map(|i| {
                        let from_idx = (t * 200 + batch_idx * 10 + i) % 5000;
                        let to_idx = (from_idx + 100) % 5000;
                        EdgeInput {
                            from: nodes[from_idx],
                            to: nodes[to_idx],
                            edge_type: "CREATED".to_string(),
                            properties: HashMap::new(),
                            directed: true,
                        }
                    })
                    .collect();

                let op_start = Instant::now();
                if let Ok(result) = eng.batch_create_edges(edges) {
                    cnt.fetch_add(result.count, Ordering::Relaxed);
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // 25 batch_delete_nodes threads (from higher indices to minimize conflicts)
    for t in 0..25 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&nodes_deleted);
        let nodes = Arc::clone(&node_pool);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for batch_idx in 0..ops_per_thread {
                // Delete from indices 3000-4999 (dedicated deletion range)
                let start_idx = 3000 + (t * 80 + batch_idx * 4) % 2000;
                let batch: Vec<u64> = (0..10).map(|i| nodes[(start_idx + i) % 5000]).collect();

                let op_start = Instant::now();
                if let Ok(result) = eng.batch_delete_nodes(batch) {
                    cnt.fetch_add(result.count, Ordering::Relaxed);
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // 25 batch_delete_edges threads
    for t in 0..25 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&edges_deleted);
        let edges = Arc::clone(&edge_pool);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for batch_idx in 0..ops_per_thread {
                let start_idx = (t * 80 + batch_idx * 4) % 2000;
                let batch: Vec<u64> = (0..10).map(|i| edges[(start_idx + i) % 2000]).collect();

                let op_start = Instant::now();
                if let Ok(result) = eng.batch_delete_edges(batch) {
                    cnt.fetch_add(result.count, Ordering::Relaxed);
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_nodes_created = nodes_created.load(Ordering::SeqCst);
    let total_edges_created = edges_created.load(Ordering::SeqCst);
    let total_nodes_deleted = nodes_deleted.load(Ordering::SeqCst);
    let total_edges_deleted = edges_deleted.load(Ordering::SeqCst);

    println!("Duration: {:?}", elapsed);
    println!("Nodes created: {total_nodes_created}");
    println!("Edges created: {total_edges_created}");
    println!("Nodes deleted: {total_nodes_deleted}");
    println!("Edges deleted: {total_edges_deleted}");

    // Sample latencies
    for (i, snapshot) in results.iter().take(4).enumerate() {
        println!("  Thread {i}: {snapshot}");
    }

    // Verify no deadlocks occurred (all threads completed)
    assert!(
        total_nodes_created > 0,
        "no nodes created - possible deadlock"
    );
    assert!(
        total_edges_created > 0,
        "no edges created - possible deadlock"
    );

    // Engine should still be responsive
    let test_node = engine.create_node("ResponsivenessTest", HashMap::new());
    assert!(test_node.is_ok(), "engine not responsive after stress");

    println!("PASSED: mixed batch operations completed without deadlock");
}
