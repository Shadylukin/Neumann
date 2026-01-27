//! Graph engine concurrency stress tests.
//!
//! Tests high-thread-count concurrent operations on GraphEngine.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
    time::Instant,
};

use graph_engine::{GraphEngine, NodeInput, PropertyValue};
use stress_tests::{full_config, LatencyHistogram};

/// Stress test: 50 threads updating nodes concurrently.
#[test]
#[ignore]
fn stress_graph_update_node_50_threads() {
    let _config = full_config();
    let thread_count = 50;
    let updates_per_thread = 10_000;

    println!("\n=== Graph Update Node 50 Threads ===");
    println!("Threads: {thread_count}");
    println!("Updates per thread: {updates_per_thread}");

    let engine = Arc::new(GraphEngine::new());

    // Pre-create nodes to update
    let mut node_ids = Vec::new();
    for t in 0..thread_count {
        let mut props = HashMap::new();
        props.insert("thread".to_string(), PropertyValue::Int(t as i64));
        props.insert("counter".to_string(), PropertyValue::Int(0));
        node_ids.push(engine.create_node("UpdateTarget", props).unwrap());
    }
    let node_ids = Arc::new(node_ids);

    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            let ids = Arc::clone(&node_ids);
            thread::spawn(move || {
                let mut latencies = LatencyHistogram::new();
                bar.wait();

                for i in 0..updates_per_thread {
                    let node_id = ids[t];
                    let mut props = HashMap::new();
                    props.insert("counter".to_string(), PropertyValue::Int(i as i64));

                    let op_start = Instant::now();
                    if eng.update_node(node_id, None, props).is_ok() {
                        cnt.fetch_add(1, Ordering::Relaxed);
                    }
                    latencies.record(op_start.elapsed());
                }

                latencies.snapshot()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_updates = success_count.load(Ordering::SeqCst);
    let expected = thread_count * updates_per_thread;

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} updates/sec",
        total_updates as f64 / elapsed.as_secs_f64()
    );

    for (i, snapshot) in results.iter().enumerate() {
        println!("  Thread {i}: {snapshot}");
    }

    // Verify p99 under threshold
    for snapshot in &results {
        assert!(
            snapshot.p99.as_millis() < 10,
            "p99 latency too high: {:?}",
            snapshot.p99
        );
    }

    assert_eq!(total_updates, expected, "some updates failed");
    println!("PASSED: {total_updates} updates completed");
}

/// Stress test: 100 threads with mixed batch operations.
#[test]
#[ignore]
fn stress_graph_batch_operations_100_threads() {
    let _config = full_config();
    let thread_count = 100;
    let batch_size = 50;
    let batches_per_thread = 10;

    println!("\n=== Graph Batch Operations 100 Threads ===");
    println!("Threads: {thread_count}");
    println!("Batch size: {batch_size}");
    println!("Batches per thread: {batches_per_thread}");

    let engine = Arc::new(GraphEngine::new());

    // Pre-create base nodes for edge targets
    let mut base_nodes = Vec::new();
    for i in 0..1000 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        base_nodes.push(engine.create_node("Base", props).unwrap());
    }
    let base_nodes = Arc::new(base_nodes);

    let barrier = Arc::new(Barrier::new(thread_count));
    let node_count = Arc::new(AtomicUsize::new(0));
    let edge_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let n_cnt = Arc::clone(&node_count);
            let e_cnt = Arc::clone(&edge_count);
            let bases = Arc::clone(&base_nodes);
            thread::spawn(move || {
                let mut latencies = LatencyHistogram::new();
                bar.wait();

                for batch_idx in 0..batches_per_thread {
                    // Alternate between node and edge batches
                    if (t + batch_idx) % 2 == 0 {
                        // Batch create nodes
                        let nodes: Vec<NodeInput> = (0..batch_size)
                            .map(|i| NodeInput {
                                labels: vec!["Batch".to_string()],
                                properties: {
                                    let mut p = HashMap::new();
                                    p.insert(
                                        "thread".to_string(),
                                        PropertyValue::Int(t as i64),
                                    );
                                    p.insert(
                                        "batch".to_string(),
                                        PropertyValue::Int(batch_idx as i64),
                                    );
                                    p.insert("idx".to_string(), PropertyValue::Int(i as i64));
                                    p
                                },
                            })
                            .collect();

                        let op_start = Instant::now();
                        if let Ok(result) = eng.batch_create_nodes(nodes) {
                            n_cnt.fetch_add(result.created_ids.len(), Ordering::Relaxed);
                        }
                        latencies.record(op_start.elapsed());
                    } else {
                        // Batch create edges between base nodes
                        let edges: Vec<graph_engine::EdgeInput> = (0..batch_size.min(999))
                            .map(|i| {
                                let from_idx = (t * 10 + i) % 1000;
                                let to_idx = (t * 10 + i + 1) % 1000;
                                graph_engine::EdgeInput {
                                    from: bases[from_idx],
                                    to: bases[to_idx],
                                    edge_type: "BATCH_LINK".to_string(),
                                    properties: HashMap::new(),
                                    directed: true,
                                }
                            })
                            .collect();

                        let op_start = Instant::now();
                        if let Ok(result) = eng.batch_create_edges(edges) {
                            e_cnt.fetch_add(result.count, Ordering::Relaxed);
                        }
                        latencies.record(op_start.elapsed());
                    }
                }

                latencies.snapshot()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_nodes = node_count.load(Ordering::SeqCst);
    let total_edges = edge_count.load(Ordering::SeqCst);
    let total_ops = total_nodes + total_edges;

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} entities/sec",
        total_ops as f64 / elapsed.as_secs_f64()
    );
    println!("Nodes created: {total_nodes}");
    println!("Edges created: {total_edges}");

    // Sample latencies from first few threads
    for (i, snapshot) in results.iter().take(5).enumerate() {
        println!("  Thread {i}: {snapshot}");
    }

    assert!(total_nodes > 0, "no nodes created");
    assert!(total_edges > 0, "no edges created");
    println!("PASSED: {total_ops} total operations");
}

/// Stress test: Striped lock saturation with 100 threads.
#[test]
#[ignore]
fn stress_graph_striped_locks_saturation() {
    let thread_count = 100;
    let ops_per_thread = 1000;

    println!("\n=== Graph Striped Locks Saturation ===");
    println!("Threads: {thread_count}");
    println!("Operations per thread: {ops_per_thread}");

    let engine = Arc::new(GraphEngine::new());
    engine.create_node_property_index("key").unwrap();

    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            thread::spawn(move || {
                let mut latencies = LatencyHistogram::new();
                bar.wait();

                for i in 0..ops_per_thread {
                    let mut props = HashMap::new();
                    // Distribute across all 64 stripes
                    let shard = (t * ops_per_thread + i) % 64;
                    let key = format!("{shard:02x}_t{t}_i{i}");
                    props.insert("key".to_string(), PropertyValue::String(key));

                    let op_start = Instant::now();
                    if eng.create_node("Saturated", props).is_ok() {
                        cnt.fetch_add(1, Ordering::Relaxed);
                    }
                    latencies.record(op_start.elapsed());
                }

                latencies.snapshot()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total = success_count.load(Ordering::SeqCst);
    let expected = thread_count * ops_per_thread;

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} ops/sec",
        total as f64 / elapsed.as_secs_f64()
    );

    // Verify throughput target: >100K ops/sec
    let throughput = total as f64 / elapsed.as_secs_f64();
    println!("Target: >100,000 ops/sec, Actual: {throughput:.0}");

    // Calculate latency statistics
    let p99_max = results
        .iter()
        .map(|s| s.p99.as_millis())
        .max()
        .unwrap_or(0);
    println!("Max p99 across threads: {p99_max}ms");

    assert_eq!(total, expected, "some operations failed");
    assert!(
        throughput > 50_000.0,
        "throughput too low: {throughput:.0} ops/sec"
    );

    // Verify index integrity
    let results = engine
        .find_nodes_by_property("key", &PropertyValue::String("00_t0_i0".into()))
        .unwrap();
    assert_eq!(results.len(), 1, "index integrity check failed");

    println!("PASSED: {total} operations at {throughput:.0} ops/sec");
}

/// Stress test: Concurrent traversals while modifying graph.
#[test]
#[ignore]
fn stress_graph_concurrent_traversal_modification() {
    let thread_count = 20;
    let traversals_per_thread = 100;
    let modifications_per_thread = 500;

    println!("\n=== Graph Concurrent Traversal + Modification ===");
    println!("Threads: {thread_count} (10 traversers + 10 modifiers)");

    let engine = Arc::new(GraphEngine::new());

    // Build initial graph: chain of 100 nodes
    let mut chain_ids = Vec::new();
    for i in 0..100 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        chain_ids.push(engine.create_node("Chain", props).unwrap());
    }
    for i in 0..99 {
        engine
            .create_edge(chain_ids[i], chain_ids[i + 1], "NEXT", HashMap::new(), true)
            .unwrap();
    }
    let chain_ids = Arc::new(chain_ids);

    let barrier = Arc::new(Barrier::new(thread_count));
    let traversal_count = Arc::new(AtomicUsize::new(0));
    let modification_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = vec![];

    // 10 traverser threads
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&traversal_count);
        let ids = Arc::clone(&chain_ids);
        handles.push(thread::spawn(move || {
            bar.wait();
            for _ in 0..traversals_per_thread {
                let start_idx = t % 50;
                if eng
                    .traverse(
                        ids[start_idx],
                        graph_engine::Direction::Outgoing,
                        10,
                        None,
                        None,
                    )
                    .is_ok()
                {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // 10 modifier threads
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&modification_count);
        handles.push(thread::spawn(move || {
            bar.wait();
            for i in 0..modifications_per_thread {
                let mut props = HashMap::new();
                props.insert("modifier".to_string(), PropertyValue::Int(t as i64));
                props.insert("mod_idx".to_string(), PropertyValue::Int(i as i64));
                if eng.create_node("Modified", props).is_ok() {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }
    let elapsed = start.elapsed();

    let traversals = traversal_count.load(Ordering::SeqCst);
    let modifications = modification_count.load(Ordering::SeqCst);

    println!("Duration: {:?}", elapsed);
    println!("Traversals completed: {traversals}");
    println!("Modifications completed: {modifications}");

    assert_eq!(traversals, 10 * traversals_per_thread);
    assert_eq!(modifications, 10 * modifications_per_thread);
    println!("PASSED: concurrent traversal and modification");
}
