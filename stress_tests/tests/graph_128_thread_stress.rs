// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Graph engine 128-thread stress test.
//!
//! Tests high-thread-count scalability with batch operations.

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

#[test]
#[ignore]
fn stress_graph_128_threads_batch_operations() {
    let _config = full_config();
    let thread_count = 128;
    let batch_size = 50;
    let batches_per_thread = 50;

    println!("\n=== Graph 128 Threads Batch Operations ===");
    println!("Threads: {thread_count}");
    println!("Batch size: {batch_size}");
    println!("Batches per thread: {batches_per_thread}");

    let engine = Arc::new(GraphEngine::new());

    // Pre-create some nodes for edge targets
    let mut base_nodes = Vec::new();
    for i in 0..2000 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        base_nodes.push(engine.create_node("Base", props).unwrap());
    }
    let base_nodes = Arc::new(base_nodes);

    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            let bases = Arc::clone(&base_nodes);
            thread::spawn(move || {
                let mut latencies = LatencyHistogram::new();
                bar.wait();

                for batch_idx in 0..batches_per_thread {
                    let op_start = Instant::now();

                    // Alternate between node and edge batches
                    if (t + batch_idx) % 2 == 0 {
                        let nodes: Vec<NodeInput> = (0..batch_size)
                            .map(|i| NodeInput {
                                labels: vec!["Stress128".to_string()],
                                properties: {
                                    let mut p = HashMap::new();
                                    p.insert("thread".to_string(), PropertyValue::Int(t as i64));
                                    p.insert(
                                        "batch".to_string(),
                                        PropertyValue::Int(batch_idx as i64),
                                    );
                                    p.insert("idx".to_string(), PropertyValue::Int(i as i64));
                                    p
                                },
                            })
                            .collect();

                        if eng.batch_create_nodes(nodes).is_ok() {
                            cnt.fetch_add(batch_size, Ordering::Relaxed);
                        }
                    } else {
                        let edges: Vec<graph_engine::EdgeInput> = (0..batch_size.min(1999))
                            .map(|i| {
                                let from_idx = (t * 15 + i) % 2000;
                                let to_idx = (t * 15 + i + 1) % 2000;
                                graph_engine::EdgeInput {
                                    from: bases[from_idx],
                                    to: bases[to_idx],
                                    edge_type: "STRESS_LINK".to_string(),
                                    properties: HashMap::new(),
                                    directed: true,
                                }
                            })
                            .collect();

                        if eng.batch_create_edges(edges).is_ok() {
                            cnt.fetch_add(batch_size, Ordering::Relaxed);
                        }
                    }

                    latencies.record(op_start.elapsed());
                }

                latencies.snapshot()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_ops = success_count.load(Ordering::SeqCst);
    let throughput = total_ops as f64 / elapsed.as_secs_f64();

    println!("Duration: {:?}", elapsed);
    println!("Total operations: {total_ops}");
    println!("Throughput: {throughput:.0} ops/sec");

    // Calculate p99 across all threads
    let p99_max = results.iter().map(|s| s.p99.as_millis()).max().unwrap_or(0);
    println!("Max p99 latency: {p99_max}ms");

    // Sample latencies from a few threads
    for (i, snapshot) in results.iter().take(5).enumerate() {
        println!("  Thread {i}: {snapshot}");
    }

    // Verify targets
    assert!(
        throughput > 10_000.0,
        "throughput {throughput:.0} ops/sec below 10K target"
    );
    assert!(p99_max < 50, "p99 latency {p99_max}ms exceeds 50ms target");

    println!("PASSED: 128-thread stress at {throughput:.0} ops/sec, p99={p99_max}ms");
}
