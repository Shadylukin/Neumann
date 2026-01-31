// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph engine aggregation stress tests.
//!
//! Tests degree queries and aggregation operations under concurrent mutations
//! to verify consistency of aggregated values.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
    time::Instant,
};

use graph_engine::{Direction, GraphEngine, PropertyValue};
use stress_tests::{full_config, LatencyHistogram};

/// Stress test: Concurrent degree queries during edge modifications.
///
/// 10 writer threads creating/deleting edges + 10 reader threads querying degree.
/// Verifies that degree() returns consistent values (never negative, matches edge count).
#[test]
#[ignore]
fn stress_concurrent_aggregation_during_writes() {
    let _config = full_config();
    let writer_count = 10;
    let reader_count = 10;
    let ops_per_writer = 500;
    let queries_per_reader = 1000;

    println!("\n=== Graph Aggregation During Writes Stress ===");
    println!("Writers: {writer_count}, Readers: {reader_count}");
    println!("Operations per writer: {ops_per_writer}");
    println!("Queries per reader: {queries_per_reader}");

    let engine = Arc::new(GraphEngine::new());

    // Create a hub-and-spoke graph structure
    // Central hub node with many spoke nodes
    let hub_id = engine
        .create_node("Hub", {
            let mut p = HashMap::new();
            p.insert("name".to_string(), PropertyValue::String("central".into()));
            p
        })
        .unwrap();

    // Create spoke nodes
    let num_spokes = 100;
    let mut spoke_ids = Vec::new();
    for i in 0..num_spokes {
        let mut props = HashMap::new();
        props.insert("spoke_idx".to_string(), PropertyValue::Int(i));
        spoke_ids.push(engine.create_node("Spoke", props).unwrap());
    }

    // Create initial edges from hub to all spokes
    for &spoke_id in &spoke_ids {
        engine
            .create_edge(hub_id, spoke_id, "CONNECTS", HashMap::new(), true)
            .unwrap();
    }
    let spoke_ids = Arc::new(spoke_ids);

    let barrier = Arc::new(Barrier::new(writer_count + reader_count));
    let edge_creates = Arc::new(AtomicUsize::new(0));
    let edge_deletes = Arc::new(AtomicUsize::new(0));
    let degree_queries = Arc::new(AtomicUsize::new(0));
    let consistency_errors = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = vec![];

    // Writer threads: create and delete edges
    for t in 0..writer_count {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let creates = Arc::clone(&edge_creates);
        let deletes = Arc::clone(&edge_deletes);
        let spokes = Arc::clone(&spoke_ids);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for i in 0..ops_per_writer {
                let op_start = Instant::now();

                // Alternate between creating edges between spokes
                let from_idx = (t * ops_per_writer + i) % spokes.len();
                let to_idx = (from_idx + 1) % spokes.len();

                if i % 2 == 0 {
                    // Create edge
                    let mut props = HashMap::new();
                    props.insert("writer".to_string(), PropertyValue::Int(t as i64));
                    if eng
                        .create_edge(spokes[from_idx], spokes[to_idx], "DYNAMIC", props, true)
                        .is_ok()
                    {
                        creates.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    // Find and delete an edge if one exists
                    if let Ok(edges) = eng.edges_of(spokes[from_idx], Direction::Outgoing) {
                        if let Some(edge) = edges
                            .iter()
                            .find(|e| e.edge_type == "DYNAMIC" && e.to == spokes[to_idx])
                        {
                            if eng.delete_edge(edge.id).is_ok() {
                                deletes.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }

                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // Reader threads: query degree and verify consistency
    for _t in 0..reader_count {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let queries = Arc::clone(&degree_queries);
        let errors = Arc::clone(&consistency_errors);
        let spokes = Arc::clone(&spoke_ids);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for i in 0..queries_per_reader {
                let op_start = Instant::now();

                // Query degree of a random spoke
                let spoke_idx = i % spokes.len();
                let spoke_id = spokes[spoke_idx];

                // Get degree
                if let Ok(degree) = eng.degree(spoke_id) {
                    queries.fetch_add(1, Ordering::Relaxed);

                    // Verify degree matches edge count
                    if let Ok(edges) = eng.edges_of(spoke_id, Direction::Both) {
                        // Note: degree might differ from edges.len() due to concurrent modifications
                        // between the two calls, but degree should never be negative
                        // and should be reasonably close to edges.len()

                        // Degree should never be negative (checked by type system)
                        // but edges.len() should match degree within a reasonable window

                        // Allow some slack for concurrent modifications
                        let diff = if degree > edges.len() {
                            degree - edges.len()
                        } else {
                            edges.len() - degree
                        };

                        // If difference is too large, it might indicate a bug
                        // Allow up to 10 edge difference for concurrent modification window
                        if diff > 10 {
                            errors.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }

                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let creates = edge_creates.load(Ordering::SeqCst);
    let deletes_count = edge_deletes.load(Ordering::SeqCst);
    let queries_count = degree_queries.load(Ordering::SeqCst);
    let errors_count = consistency_errors.load(Ordering::SeqCst);

    println!("Duration: {:?}", elapsed);
    println!("Edges created: {creates}");
    println!("Edges deleted: {deletes_count}");
    println!("Degree queries: {queries_count}");
    println!("Consistency errors (large diff): {errors_count}");

    // Sample latencies
    for (i, snapshot) in results.iter().take(5).enumerate() {
        let role = if i < writer_count { "Writer" } else { "Reader" };
        println!("  {role} {i}: {snapshot}");
    }

    // Verify no major consistency errors
    let error_rate = errors_count as f64 / queries_count as f64;
    println!("Error rate: {:.4}%", error_rate * 100.0);

    // Allow up to 1% error rate due to concurrent modification timing windows
    assert!(
        error_rate < 0.01,
        "Too many consistency errors: {errors_count}/{queries_count}"
    );

    // Verify final state consistency
    let final_hub_degree = engine.degree(hub_id).unwrap();
    let final_hub_edges = engine.edges_of(hub_id, Direction::Both).unwrap().len();
    assert_eq!(
        final_hub_degree, final_hub_edges,
        "Hub node degree mismatch after test"
    );

    println!("PASSED: Aggregation consistency maintained under concurrent writes");
}

/// Stress test: Property aggregation during concurrent updates.
///
/// Tests aggregate_node_property() consistency while nodes are being updated.
#[test]
#[ignore]
fn stress_concurrent_property_aggregation() {
    let _config = full_config();
    let writer_count = 10;
    let reader_count = 10;
    let ops_per_writer = 500;
    let queries_per_reader = 200;

    println!("\n=== Graph Property Aggregation Stress ===");
    println!("Writers: {writer_count}, Readers: {reader_count}");

    let engine = Arc::new(GraphEngine::new());

    // Create nodes with numeric properties
    let num_nodes: usize = 100;
    let mut node_ids = Vec::new();
    for i in 0..num_nodes {
        let mut props = HashMap::new();
        props.insert("value".to_string(), PropertyValue::Int(i as i64));
        props.insert("idx".to_string(), PropertyValue::Int(i as i64));
        node_ids.push(engine.create_node("Aggregated", props).unwrap());
    }
    let node_ids = Arc::new(node_ids);

    let barrier = Arc::new(Barrier::new(writer_count + reader_count));
    let updates = Arc::new(AtomicUsize::new(0));
    let aggregations = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = vec![];

    // Writer threads: update property values
    for t in 0..writer_count {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&updates);
        let ids = Arc::clone(&node_ids);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for i in 0..ops_per_writer {
                let op_start = Instant::now();

                let node_idx = (t * ops_per_writer + i) % ids.len();
                let new_value = (t * ops_per_writer + i) as i64;

                let mut props = HashMap::new();
                props.insert("value".to_string(), PropertyValue::Int(new_value));

                if eng.update_node(ids[node_idx], None, props).is_ok() {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }

                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    // Reader threads: perform property aggregations
    for _t in 0..reader_count {
        let eng = Arc::clone(&engine);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&aggregations);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            bar.wait();

            for _ in 0..queries_per_reader {
                let op_start = Instant::now();

                let result = eng.aggregate_node_property("value");

                // Verify aggregation result is valid
                assert!(
                    result.count > 0,
                    "Aggregation should find nodes with 'value' property"
                );

                // Sum should be non-negative (all values are positive)
                if let Some(sum) = result.sum {
                    assert!(sum >= 0.0, "Sum should be non-negative");
                }

                // Average should be reasonable
                if let Some(avg) = result.avg {
                    assert!(!avg.is_nan(), "Average should not be NaN");
                }

                cnt.fetch_add(1, Ordering::Relaxed);
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_updates = updates.load(Ordering::SeqCst);
    let total_aggregations = aggregations.load(Ordering::SeqCst);

    println!("Duration: {:?}", elapsed);
    println!("Updates completed: {total_updates}");
    println!("Aggregations completed: {total_aggregations}");
    println!(
        "Throughput: {:.0} ops/sec",
        (total_updates + total_aggregations) as f64 / elapsed.as_secs_f64()
    );

    // Sample latencies
    for (i, snapshot) in results.iter().take(5).enumerate() {
        let role = if i < writer_count { "Writer" } else { "Reader" };
        println!("  {role} {i}: {snapshot}");
    }

    // Final aggregation check
    let final_result = engine.aggregate_node_property("value");
    println!(
        "Final aggregation: count={}, sum={:?}, avg={:?}",
        final_result.count, final_result.sum, final_result.avg
    );

    assert_eq!(
        final_result.count, num_nodes as u64,
        "All nodes should still exist"
    );

    println!("PASSED: Property aggregation remained consistent under concurrent updates");
}

/// Stress test: Degree-by-type queries under concurrent edge type changes.
#[test]
#[ignore]
fn stress_concurrent_degree_by_type() {
    let _config = full_config();
    let thread_count = 20;
    let ops_per_thread = 500;

    println!("\n=== Graph Degree-By-Type Stress ===");
    println!("Threads: {thread_count}");
    println!("Operations per thread: {ops_per_thread}");

    let engine = Arc::new(GraphEngine::new());

    // Create a star graph
    let center_id = engine.create_node("Center", HashMap::new()).unwrap();
    let num_spokes: usize = 50;
    let mut spoke_ids = Vec::new();
    for i in 0..num_spokes {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i as i64));
        spoke_ids.push(engine.create_node("Spoke", props).unwrap());
    }

    // Create edges with different types
    let edge_types = ["TYPE_A", "TYPE_B", "TYPE_C"];
    for (i, &spoke_id) in spoke_ids.iter().enumerate() {
        let edge_type = edge_types[i % edge_types.len()];
        engine
            .create_edge(center_id, spoke_id, edge_type, HashMap::new(), true)
            .unwrap();
    }
    let spoke_ids = Arc::new(spoke_ids);

    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let eng = Arc::clone(&engine);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            let spokes = Arc::clone(&spoke_ids);
            thread::spawn(move || {
                let mut latencies = LatencyHistogram::new();
                bar.wait();

                for i in 0..ops_per_thread {
                    let op_start = Instant::now();

                    if i % 3 == 0 {
                        // Query degree by type
                        let edge_type = edge_types[i % edge_types.len()];
                        if let Ok(degree) = eng.out_degree_by_type(center_id, edge_type) {
                            // Degree should be non-negative
                            assert!(degree <= num_spokes, "Degree exceeds max possible");
                            cnt.fetch_add(1, Ordering::Relaxed);
                        }
                    } else if i % 3 == 1 {
                        // Create a new edge
                        let spoke_idx = (t * ops_per_thread + i) % spokes.len();
                        let edge_type = edge_types[(t + i) % edge_types.len()];
                        if eng
                            .create_edge(
                                center_id,
                                spokes[spoke_idx],
                                edge_type,
                                HashMap::new(),
                                true,
                            )
                            .is_ok()
                        {
                            cnt.fetch_add(1, Ordering::Relaxed);
                        }
                    } else {
                        // Query total degree
                        if let Ok(degree) = eng.out_degree(center_id) {
                            assert!(degree >= num_spokes, "Degree below initial value");
                            cnt.fetch_add(1, Ordering::Relaxed);
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

    let total = success_count.load(Ordering::SeqCst);

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} ops/sec",
        total as f64 / elapsed.as_secs_f64()
    );
    println!("Successful operations: {total}");

    // Sample latencies
    for (i, snapshot) in results.iter().take(5).enumerate() {
        println!("  Thread {i}: {snapshot}");
    }

    // Verify final state
    let final_degree = engine.out_degree(center_id).unwrap();
    println!("Final center out_degree: {final_degree}");

    for edge_type in &edge_types {
        let type_degree = engine.out_degree_by_type(center_id, edge_type).unwrap();
        println!("  {edge_type}: {type_degree}");
    }

    println!("PASSED: Degree-by-type queries consistent under concurrent modifications");
}
