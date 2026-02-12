// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Graph engine index operations stress test.
//!
//! Tests concurrent index create/drop with writers.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
    time::Instant,
};

use graph_engine::{GraphEngine, PropertyValue};
use stress_tests::{full_config, LatencyHistogram};

#[test]
#[ignore]
fn stress_graph_index_50_threads() {
    let _config = full_config();
    let thread_count = 50;
    let ops_per_thread = 100;

    println!("\n=== Graph Index Stress 50 Threads ===");
    println!("Threads: {thread_count} (20 index create/drop + 30 writers)");
    println!("Operations per thread: {ops_per_thread}");

    let engine = Arc::new(GraphEngine::new());

    // Pre-create 10 indexes to work with
    let index_properties: Vec<String> = (0..10).map(|i| format!("prop_{i}")).collect();
    for prop in &index_properties {
        engine.create_node_property_index(prop).unwrap();
    }
    let index_properties = Arc::new(index_properties);

    let barrier = Arc::new(Barrier::new(thread_count));
    let index_ops = Arc::new(AtomicUsize::new(0));
    let write_ops = Arc::new(AtomicUsize::new(0));
    let query_ops = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = Vec::new();

    // 10 index creator threads
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&index_ops);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for i in 0..ops_per_thread {
                let prop_name = format!("new_prop_{t}_{i}");
                let op_start = Instant::now();
                if eng.create_node_property_index(&prop_name).is_ok() {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // 10 index dropper threads (rotate through existing indexes)
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&index_ops);
        let props = Arc::clone(&index_properties);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for i in 0..ops_per_thread {
                // Try to drop and recreate indexes
                let prop = &props[(t + i) % 10];
                let op_start = Instant::now();
                // Drop might fail if another thread dropped it
                let _ = eng.drop_node_index(prop);
                // Recreate it
                if eng.create_node_property_index(prop).is_ok() {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // 30 writer threads
    for t in 0..30 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let w_cnt = Arc::clone(&write_ops);
        let q_cnt = Arc::clone(&query_ops);
        let props = Arc::clone(&index_properties);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for i in 0..ops_per_thread {
                let prop = &props[(t + i) % 10];
                let value = format!("value_{t}_{i}");

                // Write a node with indexed property
                let mut node_props = HashMap::new();
                node_props.insert(prop.clone(), PropertyValue::String(value.clone()));

                let op_start = Instant::now();
                if eng.create_node("Indexed", node_props).is_ok() {
                    w_cnt.fetch_add(1, Ordering::Relaxed);
                }
                latencies.record(op_start.elapsed());

                // Query using the index
                let query_start = Instant::now();
                if eng
                    .find_nodes_by_property(prop, &PropertyValue::String(value))
                    .is_ok()
                {
                    q_cnt.fetch_add(1, Ordering::Relaxed);
                }
                latencies.record(query_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_index_ops = index_ops.load(Ordering::SeqCst);
    let total_write_ops = write_ops.load(Ordering::SeqCst);
    let total_query_ops = query_ops.load(Ordering::SeqCst);

    println!("Duration: {:?}", elapsed);
    println!("Index operations: {total_index_ops}");
    println!("Write operations: {total_write_ops}");
    println!("Query operations: {total_query_ops}");

    // Calculate p99
    let p99_max = results.iter().map(|s| s.p99.as_millis()).max().unwrap_or(0);
    println!("Max p99 latency: {p99_max}ms");

    // Sample latencies
    for (i, snapshot) in results.iter().take(5).enumerate() {
        println!("  Thread {i}: {snapshot}");
    }

    // Verify operations completed
    assert!(total_write_ops > 0, "no writes completed");
    assert!(total_query_ops > 0, "no queries completed");

    // Verify engine consistency
    let test_result = engine.create_node("ConsistencyTest", HashMap::new());
    assert!(test_result.is_ok(), "engine inconsistent after stress");

    println!("PASSED: index stress with concurrent create/drop/write/query");
}
