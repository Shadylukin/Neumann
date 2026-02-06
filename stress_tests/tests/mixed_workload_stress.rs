// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Mixed workload stress tests across all engines.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use graph_engine::{GraphEngine, PropertyValue};
use relational_engine::{Column, ColumnType, RelationalEngine, Schema, Value};
use stress_tests::{full_config, generate_embeddings, LatencyHistogram};
use tensor_store::TensorStore;
use vector_engine::VectorEngine;

/// Stress test: All engines concurrent (4 threads per engine).
#[test]
#[ignore]
fn stress_all_engines_concurrent() {
    let _config = full_config();
    let threads_per_engine = 4;
    let ops_per_thread = 25_000;

    println!("\n=== All Engines Concurrent ===");
    println!("Threads/engine: {}", threads_per_engine);
    println!("Ops/thread: {}", ops_per_thread);

    // Shared store for all engines
    let store = TensorStore::new();
    let relational = Arc::new(RelationalEngine::with_store(store.clone()));
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let vector = Arc::new(VectorEngine::with_store(store.clone()));

    // Create relational schema
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
        Column::new("value", ColumnType::Float),
    ]);
    relational.create_table("stress_test", schema).unwrap();

    // Pre-generate embeddings
    let embeddings = Arc::new(generate_embeddings(
        ops_per_thread * threads_per_engine,
        128,
        42,
    ));

    let mut handles = vec![];
    let start = Instant::now();

    // Relational threads
    for t in 0..threads_per_engine {
        let rel = Arc::clone(&relational);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let base_id = t * ops_per_thread;

            for i in 0..ops_per_thread {
                let op_start = Instant::now();

                let mut row = HashMap::new();
                row.insert("id".to_string(), Value::Int((base_id + i) as i64));
                row.insert(
                    "name".to_string(),
                    Value::String(format!("entity_{}", base_id + i)),
                );
                row.insert("value".to_string(), Value::Float((i as f64) * 0.01));
                rel.insert("stress_test", row).unwrap();

                latencies.record(op_start.elapsed());
            }

            ("relational", latencies.snapshot())
        }));
    }

    // Graph threads
    for t in 0..threads_per_engine {
        let g = Arc::clone(&graph);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();

            for i in 0..ops_per_thread {
                let op_start = Instant::now();

                let mut props = HashMap::new();
                props.insert("thread".to_string(), PropertyValue::Int(t as i64));
                props.insert("index".to_string(), PropertyValue::Int(i as i64));
                let node_id = g.create_node("stress_node", props).unwrap();

                // Create edge to previous node
                if i > 0 {
                    let prev_id = node_id - 1;
                    let mut edge_props = HashMap::new();
                    edge_props.insert("weight".to_string(), PropertyValue::Float(1.0));
                    let _ = g.create_edge(prev_id, node_id, "next", edge_props, true);
                }

                latencies.record(op_start.elapsed());
            }

            ("graph", latencies.snapshot())
        }));
    }

    // Vector threads
    for t in 0..threads_per_engine {
        let v = Arc::clone(&vector);
        let embeddings = Arc::clone(&embeddings);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let base_idx = t * ops_per_thread;

            for i in 0..ops_per_thread {
                let idx = base_idx + i;
                let op_start = Instant::now();
                v.store_embedding(&format!("vec:{}", idx), embeddings[idx].clone())
                    .unwrap();
                latencies.record(op_start.elapsed());
            }

            ("vector", latencies.snapshot())
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_ops = threads_per_engine * ops_per_thread * 3; // 3 engines
    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} ops/sec",
        total_ops as f64 / elapsed.as_secs_f64()
    );

    // Group results by engine
    let mut by_engine: HashMap<&str, Vec<_>> = HashMap::new();
    for (engine, snapshot) in &results {
        by_engine.entry(*engine).or_default().push(snapshot);
    }

    for (engine, snapshots) in &by_engine {
        println!("  {} ({}x):", engine, snapshots.len());
        for (i, s) in snapshots.iter().enumerate() {
            println!("    Thread {}: {}", i, s);
        }
    }

    // Verify counts
    let rel_count = relational
        .select("stress_test", relational_engine::Condition::True)
        .unwrap()
        .len();
    println!("Relational rows: {}", rel_count);
    assert_eq!(rel_count, threads_per_engine * ops_per_thread);

    println!("PASSED: All engines concurrent");
}

/// Stress test: Realistic mixed workload.
#[test]
#[ignore]
fn stress_realistic_workload() {
    let _config = full_config();
    let duration_secs = 30;
    let reader_threads = 4;
    let writer_threads = 2;
    let search_threads = 2;

    println!("\n=== Realistic Mixed Workload ===");
    println!("Duration: {}s", duration_secs);
    println!("Readers: {}", reader_threads);
    println!("Writers: {}", writer_threads);
    println!("Searchers: {}", search_threads);

    // Setup engines
    let store = TensorStore::new();
    let relational = Arc::new(RelationalEngine::with_store(store.clone()));
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let vector = Arc::new(VectorEngine::with_store(store.clone()));

    // Create schema
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("users", schema).unwrap();

    // Pre-populate some data
    let embeddings = Arc::new(generate_embeddings(10_000, 128, 42));
    for (i, emb) in embeddings.iter().enumerate() {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i as i64));
        row.insert("name".to_string(), Value::String(format!("user_{}", i)));
        relational.insert("users", row).unwrap();

        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String(format!("node_{}", i)),
        );
        graph.create_node("user", props).unwrap();

        vector
            .store_embedding(&format!("emb:{}", i), emb.clone())
            .unwrap();
    }

    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let read_count = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::new(AtomicUsize::new(0));
    let search_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    let start = Instant::now();

    // Reader threads (relational selects + graph traversals)
    for _ in 0..reader_threads {
        let rel = Arc::clone(&relational);
        let g = Arc::clone(&graph);
        let done = Arc::clone(&done);
        let reads = Arc::clone(&read_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();

            while !done.load(Ordering::Acquire) {
                let op_start = Instant::now();

                // Mix of operations
                let _ = rel.select("users", relational_engine::Condition::True);
                let _ = g.get_node(reads.load(Ordering::Relaxed) as u64 % 1000);

                latencies.record(op_start.elapsed());
                reads.fetch_add(1, Ordering::Relaxed);
            }

            ("reader", latencies.snapshot())
        }));
    }

    // Writer threads (relational inserts + graph creates)
    for t in 0..writer_threads {
        let rel = Arc::clone(&relational);
        let g = Arc::clone(&graph);
        let done = Arc::clone(&done);
        let writes = Arc::clone(&write_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let base_id = 10_000 + t * 100_000;
            let mut i = 0;

            while !done.load(Ordering::Acquire) {
                let op_start = Instant::now();

                let id = base_id + i;
                let mut row = HashMap::new();
                row.insert("id".to_string(), Value::Int(id as i64));
                row.insert(
                    "name".to_string(),
                    Value::String(format!("new_user_{}", id)),
                );
                let _ = rel.insert("users", row);

                let mut props = HashMap::new();
                props.insert(
                    "name".to_string(),
                    PropertyValue::String(format!("new_node_{}", id)),
                );
                let _ = g.create_node("user", props);

                latencies.record(op_start.elapsed());
                writes.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }

            ("writer", latencies.snapshot())
        }));
    }

    // Search threads (vector similarity)
    for t in 0..search_threads {
        let v = Arc::clone(&vector);
        let embeddings = Arc::clone(&embeddings);
        let done = Arc::clone(&done);
        let searches = Arc::clone(&search_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut i = 0;

            while !done.load(Ordering::Acquire) {
                let op_start = Instant::now();

                let query_idx = (t * 1000 + i) % embeddings.len();
                let _ = v.search_similar(&embeddings[query_idx], 10);

                latencies.record(op_start.elapsed());
                searches.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }

            ("search", latencies.snapshot())
        }));
    }

    // Run for specified duration
    thread::sleep(Duration::from_secs(duration_secs));
    done.store(true, Ordering::Release);

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let reads = read_count.load(Ordering::Relaxed);
    let writes = write_count.load(Ordering::Relaxed);
    let searches = search_count.load(Ordering::Relaxed);
    let total = reads + writes + searches;

    println!("Duration: {:?}", elapsed);
    println!(
        "Reads: {} ({:.0}/sec)",
        reads,
        reads as f64 / elapsed.as_secs_f64()
    );
    println!(
        "Writes: {} ({:.0}/sec)",
        writes,
        writes as f64 / elapsed.as_secs_f64()
    );
    println!(
        "Searches: {} ({:.0}/sec)",
        searches,
        searches as f64 / elapsed.as_secs_f64()
    );
    println!(
        "Total: {} ({:.0} ops/sec)",
        total,
        total as f64 / elapsed.as_secs_f64()
    );

    // Group by role
    let mut by_role: HashMap<&str, Vec<_>> = HashMap::new();
    for (role, snapshot) in &results {
        by_role.entry(*role).or_default().push(snapshot);
    }

    for (role, snapshots) in &by_role {
        println!("  {}:", role);
        for (i, s) in snapshots.iter().enumerate() {
            println!("    {}: {}", i, s);
        }
    }

    println!("PASSED: Realistic mixed workload");
}
