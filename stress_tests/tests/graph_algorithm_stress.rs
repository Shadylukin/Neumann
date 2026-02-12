// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Graph engine algorithm stress test.
//!
//! Tests concurrent algorithm execution with writers.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
    time::Instant,
};

use graph_engine::{
    CentralityConfig, CommunityConfig, Direction, GraphEngine, PageRankConfig, PropertyValue,
};
use stress_tests::{full_config, LatencyHistogram};

#[test]
#[ignore]
fn stress_graph_algorithms_40_threads() {
    let _config = full_config();
    let thread_count = 40;
    let ops_per_thread = 10;

    println!("\n=== Graph Algorithm Stress 40 Threads ===");
    println!("Threads: {thread_count} (10 pagerank + 10 components + 10 betweenness + 10 writers)");
    println!("Operations per thread: {ops_per_thread}");

    let engine = Arc::new(GraphEngine::new());

    // Create a 500-node graph with connections
    let mut node_ids = Vec::new();
    for i in 0..500 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        node_ids.push(engine.create_node("AlgoTest", props).unwrap());
    }

    // Create a connected graph: each node connects to several others
    for i in 0..500 {
        for offset in [1, 7, 13, 31] {
            let target = (i + offset) % 500;
            if target != i {
                let _ = engine.create_edge(
                    node_ids[i],
                    node_ids[target],
                    "CONNECTED",
                    HashMap::new(),
                    false, // undirected for better algorithm results
                );
            }
        }
    }

    let node_ids = Arc::new(node_ids);
    let barrier = Arc::new(Barrier::new(thread_count));
    let pagerank_count = Arc::new(AtomicUsize::new(0));
    let components_count = Arc::new(AtomicUsize::new(0));
    let betweenness_count = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = Vec::new();

    // 10 pagerank threads
    for _ in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&pagerank_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for _ in 0..ops_per_thread {
                let op_start = Instant::now();
                if let Ok(result) = eng.pagerank(Some(PageRankConfig {
                    damping: 0.85,
                    tolerance: 0.01,
                    max_iterations: 50,
                    direction: Direction::Both,
                    edge_type: None,
                })) {
                    // Validate results
                    let mut valid = true;
                    for score in result.scores.values() {
                        if !score.is_finite() || *score < 0.0 {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        cnt.fetch_add(1, Ordering::Relaxed);
                    }
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // 10 connected_components threads
    for _ in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&components_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for _ in 0..ops_per_thread {
                let op_start = Instant::now();
                if let Ok(result) = eng.connected_components(Some(CommunityConfig {
                    max_iterations: 100,
                    ..Default::default()
                })) {
                    // Validate: should have at least one component
                    if !result.communities.is_empty() {
                        cnt.fetch_add(1, Ordering::Relaxed);
                    }
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // 10 betweenness_centrality threads
    for _ in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&betweenness_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for _ in 0..ops_per_thread {
                let op_start = Instant::now();
                // Use sampling for performance
                if let Ok(result) = eng.betweenness_centrality(Some(CentralityConfig {
                    sampling_ratio: 0.1,
                    ..Default::default()
                })) {
                    // Validate: all scores should be finite and non-negative
                    let mut valid = true;
                    for score in result.scores.values() {
                        if !score.is_finite() || *score < 0.0 {
                            valid = false;
                            break;
                        }
                    }
                    if valid {
                        cnt.fetch_add(1, Ordering::Relaxed);
                    }
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // 10 writer threads
    for t in 0..10 {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&write_count);
        let ids = Arc::clone(&node_ids);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for i in 0..ops_per_thread * 10 {
                let op_start = Instant::now();

                // Alternate between node updates and edge creation
                if i % 2 == 0 {
                    let node_idx = (t * 50 + i) % 500;
                    let mut props = HashMap::new();
                    props.insert("modified".to_string(), PropertyValue::Int(i as i64));
                    if eng.update_node(ids[node_idx], None, props).is_ok() {
                        cnt.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    let from = ids[(t * 50 + i) % 500];
                    let to = ids[(t * 50 + i + 17) % 500];
                    if eng
                        .create_edge(from, to, "NEW_EDGE", HashMap::new(), true)
                        .is_ok()
                    {
                        cnt.fetch_add(1, Ordering::Relaxed);
                    }
                }

                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_pagerank = pagerank_count.load(Ordering::SeqCst);
    let total_components = components_count.load(Ordering::SeqCst);
    let total_betweenness = betweenness_count.load(Ordering::SeqCst);
    let total_writes = write_count.load(Ordering::SeqCst);

    println!("Duration: {:?}", elapsed);
    println!("PageRank completions: {total_pagerank}");
    println!("Connected components completions: {total_components}");
    println!("Betweenness centrality completions: {total_betweenness}");
    println!("Write operations: {total_writes}");

    // Calculate p99
    let p99_max = results.iter().map(|s| s.p99.as_millis()).max().unwrap_or(0);
    println!("Max p99 latency: {p99_max}ms");

    // Sample latencies
    for (i, snapshot) in results.iter().take(4).enumerate() {
        println!("  Thread {i}: {snapshot}");
    }

    // Verify algorithms completed
    assert!(total_pagerank > 0, "no pagerank completions");
    assert!(total_components > 0, "no connected_components completions");
    assert!(
        total_betweenness > 0,
        "no betweenness_centrality completions"
    );
    assert!(total_writes > 0, "no write operations");

    // Verify engine consistency
    let test_result = engine.create_node("ConsistencyTest", HashMap::new());
    assert!(test_result.is_ok(), "engine inconsistent after stress");

    println!("PASSED: algorithm stress with concurrent reads and writes");
}
