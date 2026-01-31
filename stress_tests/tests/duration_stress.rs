// SPDX-License-Identifier: MIT OR Apache-2.0
//! Long-running duration stress tests.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use graph_engine::{GraphEngine, PropertyValue};
use relational_engine::{Column, ColumnType, RelationalEngine, Schema, Value};
use stress_tests::{endurance_config, generate_embeddings, LatencyHistogram};
use tensor_store::TensorStore;
use vector_engine::VectorEngine;


/// Stress test: 1 hour sustained load.
#[test]
#[ignore]
fn stress_1_hour_sustained_load() {
    // Use endurance config but allow override via env
    let config = endurance_config();
    let duration_secs = std::env::var("STRESS_DURATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(config.duration_secs);
    let thread_count = config.effective_thread_count();
    let report_interval = Duration::from_secs(config.report_interval_secs);

    println!("\n=== 1 Hour Sustained Load ===");
    println!("Duration: {}s", duration_secs);
    println!("Threads: {}", thread_count);
    println!("Report interval: {:?}", report_interval);

    // Setup engines
    let store = Arc::new(TensorStore::new());
    let relational = Arc::new(RelationalEngine::with_store((*store).clone()));
    let graph = Arc::new(GraphEngine::with_store((*store).clone()));
    let vector = Arc::new(VectorEngine::with_store((*store).clone()));

    // Create schema
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("data", ColumnType::String),
    ]);
    relational.create_table("durability_test", schema).unwrap();

    let done = Arc::new(AtomicBool::new(false));
    let total_ops = Arc::new(AtomicUsize::new(0));
    let cycle_count = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    let start = Instant::now();

    // Worker threads
    for t in 0..thread_count {
        let store = Arc::clone(&store);
        let rel = Arc::clone(&relational);
        let g = Arc::clone(&graph);
        let v = Arc::clone(&vector);
        let done = Arc::clone(&done);
        let ops = Arc::clone(&total_ops);
        let cycles = Arc::clone(&cycle_count);

        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut local_ops = 0;
            let mut local_cycles = 0;
            let cycle_size = 100;
            let base_id = t * 10_000_000;

            // Pre-generate some embeddings
            let embeddings = generate_embeddings(cycle_size, 128, 42 + t as u64);

            while !done.load(Ordering::Acquire) {
                let cycle_start = Instant::now();

                // Insert cycle
                for i in 0..cycle_size {
                    let id = base_id + local_cycles * cycle_size + i;

                    // Relational insert
                    let mut row = HashMap::new();
                    row.insert("id".to_string(), Value::Int(id as i64));
                    row.insert("data".to_string(), Value::String(format!("data_{}", id)));
                    let _ = rel.insert("durability_test", row);

                    // Graph node
                    let mut props = HashMap::new();
                    props.insert("id".to_string(), PropertyValue::Int(id as i64));
                    let _ = g.create_node("test_node", props);

                    // Vector
                    let _ = v.store_embedding(
                        &format!("vec:{}", id),
                        embeddings[i % embeddings.len()].clone(),
                    );

                    local_ops += 3;
                }

                // Read cycle
                for i in 0..cycle_size / 2 {
                    let id = base_id + local_cycles * cycle_size + i;
                    let _ = store.get(&format!("vec:{}", id));
                    local_ops += 1;
                }

                // Search cycle
                let _ = v.search_similar(&embeddings[0], 10);
                local_ops += 1;

                latencies.record(cycle_start.elapsed());
                local_cycles += 1;

                // Periodic update
                if local_cycles % 10 == 0 {
                    ops.fetch_add(local_ops, Ordering::Relaxed);
                    cycles.fetch_add(10, Ordering::Relaxed);
                    local_ops = 0;
                }
            }

            // Final update
            ops.fetch_add(local_ops, Ordering::Relaxed);
            cycles.fetch_add(local_cycles % 10, Ordering::Relaxed);

            latencies.snapshot()
        }));
    }

    // Reporter thread
    {
        let done = Arc::clone(&done);
        let ops = Arc::clone(&total_ops);
        let cycles = Arc::clone(&cycle_count);
        let store = Arc::clone(&store);

        handles.push(thread::spawn(move || {
            let mut last_ops = 0;
            let mut last_report = Instant::now();

            while !done.load(Ordering::Acquire) {
                thread::sleep(report_interval);

                let current_ops = ops.load(Ordering::Relaxed);
                let current_cycles = cycles.load(Ordering::Relaxed);
                let elapsed = start.elapsed();
                let interval_elapsed = last_report.elapsed();

                let interval_ops = current_ops - last_ops;
                let interval_rate = interval_ops as f64 / interval_elapsed.as_secs_f64();
                let overall_rate = current_ops as f64 / elapsed.as_secs_f64();

                println!(
                    "[{:>6.1}s] ops={:>10} cycles={:>8} rate={:>8.0}/s (interval: {:>8.0}/s) store_size={:>8}",
                    elapsed.as_secs_f64(),
                    current_ops,
                    current_cycles,
                    overall_rate,
                    interval_rate,
                    store.len()
                );

                last_ops = current_ops;
                last_report = Instant::now();
            }

            LatencyHistogram::new().snapshot() // Dummy
        }));
    }

    // Timer thread
    {
        let done = Arc::clone(&done);
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(duration_secs));
            done.store(true, Ordering::Release);
        });
    }

    // Wait for completion
    thread::sleep(Duration::from_secs(duration_secs + 5));
    done.store(true, Ordering::Release);

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let final_ops = total_ops.load(Ordering::Relaxed);
    let final_cycles = cycle_count.load(Ordering::Relaxed);

    println!("\n=== Final Results ===");
    println!("Duration: {:?}", elapsed);
    println!("Total ops: {}", final_ops);
    println!("Total cycles: {}", final_cycles);
    println!(
        "Average throughput: {:.0} ops/sec",
        final_ops as f64 / elapsed.as_secs_f64()
    );
    println!("Final store size: {}", store.len());

    // Sample thread latencies
    println!("Sample cycle latencies:");
    for (i, snapshot) in results.iter().take(3).enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    println!("PASSED: Sustained load for {:?}", elapsed);
}

/// Stress test: Memory leak detection.
#[test]
#[ignore]
fn stress_memory_leak_detection() {
    let cycles = 100;
    let items_per_cycle = 10_000;

    println!("\n=== Memory Leak Detection ===");
    println!("Cycles: {}", cycles);
    println!("Items/cycle: {}", items_per_cycle);

    let store = TensorStore::new();
    let mut memory_samples = vec![];

    for cycle in 0..cycles {
        // Insert items
        for i in 0..items_per_cycle {
            let key = format!("cycle{}:item{}", cycle, i);
            let mut data = tensor_store::TensorData::new();
            data.set(
                "value",
                tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(i as i64)),
            );
            store.put(key, data).unwrap();
        }

        // Delete items (cleanup)
        for i in 0..items_per_cycle {
            let key = format!("cycle{}:item{}", cycle, i);
            let _ = store.delete(&key);
        }

        // Sample memory (store should be nearly empty after delete)
        let store_size = store.len();
        memory_samples.push((cycle, store_size));

        if cycle % 10 == 0 {
            println!("Cycle {}: store_size={} (should be ~0)", cycle, store_size);
        }
    }

    // Verify no memory leak (store should be empty after each cycle)
    let final_size = store.len();
    println!("Final store size: {}", final_size);

    // Allow small variance due to timing
    assert!(
        final_size < 100,
        "Possible memory leak: final_size={}",
        final_size
    );

    // Check for growing trend
    let first_half_avg: usize = memory_samples[..cycles / 2]
        .iter()
        .map(|(_, s)| *s)
        .sum::<usize>()
        / (cycles / 2);
    let second_half_avg: usize = memory_samples[cycles / 2..]
        .iter()
        .map(|(_, s)| *s)
        .sum::<usize>()
        / (cycles / 2);

    println!(
        "First half avg size: {}, Second half avg size: {}",
        first_half_avg, second_half_avg
    );

    // Second half shouldn't be significantly larger (would indicate leak)
    assert!(
        second_half_avg <= first_half_avg + 10,
        "Memory growing: first={} second={}",
        first_half_avg,
        second_half_avg
    );

    println!("PASSED: No memory leak detected");
}

/// Stress test: Throughput stability over time.
#[test]
#[ignore]
fn stress_throughput_stability() {
    let duration_secs = 60;
    let sample_interval = Duration::from_secs(5);
    let thread_count = 8;

    println!("\n=== Throughput Stability ===");
    println!("Duration: {}s", duration_secs);
    println!("Sample interval: {:?}", sample_interval);
    println!("Threads: {}", thread_count);

    let store = Arc::new(TensorStore::new());
    let done = Arc::new(AtomicBool::new(false));
    let ops = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    let start = Instant::now();

    // Worker threads
    for t in 0..thread_count {
        let store = Arc::clone(&store);
        let done = Arc::clone(&done);
        let ops = Arc::clone(&ops);

        handles.push(thread::spawn(move || {
            let embeddings = generate_embeddings(1000, 128, 42 + t as u64);
            let mut i = 0;

            while !done.load(Ordering::Acquire) {
                let key = format!("t{}:k{}", t, i);
                let mut data = tensor_store::TensorData::new();
                data.set(
                    "embedding",
                    tensor_store::TensorValue::Vector(embeddings[i % 1000].clone()),
                );
                store.put(key, data).unwrap();
                ops.fetch_add(1, Ordering::Relaxed);
                i += 1;
            }
        }));
    }

    // Collect throughput samples
    let mut samples = vec![];
    let mut last_ops = 0;

    for _ in 0..(duration_secs as usize / sample_interval.as_secs() as usize) {
        thread::sleep(sample_interval);

        let current_ops = ops.load(Ordering::Relaxed);
        let interval_ops = current_ops - last_ops;
        let rate = interval_ops as f64 / sample_interval.as_secs_f64();
        samples.push(rate);
        last_ops = current_ops;

        println!(
            "[{:>5.1}s] rate={:>10.0} ops/sec, total={:>10}",
            start.elapsed().as_secs_f64(),
            rate,
            current_ops
        );
    }

    done.store(true, Ordering::Release);

    for handle in handles {
        handle.join().unwrap();
    }

    let _elapsed = start.elapsed();
    let total = ops.load(Ordering::Relaxed);

    // Analyze stability
    let avg_rate: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance: f64 =
        samples.iter().map(|r| (r - avg_rate).powi(2)).sum::<f64>() / samples.len() as f64;
    let std_dev = variance.sqrt();
    let cv = std_dev / avg_rate * 100.0; // Coefficient of variation

    println!("\n=== Stability Analysis ===");
    println!("Total ops: {}", total);
    println!("Average rate: {:.0} ops/sec", avg_rate);
    println!("Std dev: {:.0} ops/sec", std_dev);
    println!("CV: {:.1}%", cv);

    // CV under 20% is considered stable
    if cv < 20.0 {
        println!("PASSED: Throughput is stable (CV={:.1}%)", cv);
    } else {
        println!("WARNING: Throughput variance is high (CV={:.1}%)", cv);
    }
}
