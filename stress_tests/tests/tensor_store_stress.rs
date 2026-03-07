// SPDX-License-Identifier: MIT OR Apache-2.0
//! TensorStore stress tests at 1M entity scale.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Instant,
};

use stress_tests::{full_config, generate_embeddings, LatencyHistogram};
use tensor_store::{BloomFilter, ScalarValue, TensorData, TensorStore, TensorValue};

fn create_tensor(id: i64, embedding: Vec<f32>) -> TensorData {
    let mut data = TensorData::new();
    data.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
    data.set("embedding", TensorValue::Vector(embedding));
    data
}

/// Stress test: 1M concurrent writes across multiple threads.
#[test]
#[ignore]
fn stress_tensor_store_1m_concurrent_writes() {
    let config = full_config();
    let entity_count = config.effective_entity_count();
    let thread_count = config.effective_thread_count();

    println!("\n=== TensorStore 1M Concurrent Writes ===");
    println!("Entities: {}", entity_count);
    println!("Threads: {}", thread_count);

    let store = Arc::new(TensorStore::new());
    let embeddings = Arc::new(generate_embeddings(entity_count, config.embedding_dim, 42));

    let per_thread = entity_count / thread_count;
    let mut handles = vec![];
    let start = Instant::now();

    for t in 0..thread_count {
        let store = Arc::clone(&store);
        let embeddings = Arc::clone(&embeddings);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let start_idx = t * per_thread;

            for i in 0..per_thread {
                let idx = start_idx + i;
                let op_start = Instant::now();
                store
                    .put(
                        format!("key:{}", idx),
                        create_tensor(idx as i64, embeddings[idx].clone()),
                    )
                    .unwrap();
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} entities/sec",
        entity_count as f64 / elapsed.as_secs_f64()
    );

    // Aggregate latencies
    for (i, snapshot) in results.iter().enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    // Verify correctness
    assert_eq!(store.len(), entity_count);

    // Verify p99 under threshold
    for snapshot in &results {
        assert!(
            snapshot.p99.as_millis() < 50,
            "p99 latency too high: {:?}",
            snapshot.p99
        );
    }

    println!("PASSED: All {} entities written correctly", entity_count);
}

/// Stress test: High contention - many threads writing to few keys.
#[test]
#[ignore]
fn stress_tensor_store_high_contention() {
    let config = full_config();
    let thread_count = config.effective_thread_count();
    let iterations_per_thread = 100_000;
    let num_keys = 1000; // High contention: 16 threads writing to 1000 keys

    println!("\n=== TensorStore High Contention ===");
    println!("Threads: {}", thread_count);
    println!("Keys: {}", num_keys);
    println!("Iterations/thread: {}", iterations_per_thread);

    let store = Arc::new(TensorStore::new());
    let success_count = Arc::new(AtomicUsize::new(0));
    let embeddings = Arc::new(generate_embeddings(num_keys, 128, 42));

    let mut handles = vec![];
    let start = Instant::now();

    for t in 0..thread_count {
        let store = Arc::clone(&store);
        let success = Arc::clone(&success_count);
        let embeddings = Arc::clone(&embeddings);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();

            for i in 0..iterations_per_thread {
                let key_idx = i % num_keys;
                let op_start = Instant::now();
                store
                    .put(
                        format!("key:{}", key_idx),
                        create_tensor(
                            (t * iterations_per_thread + i) as i64,
                            embeddings[key_idx].clone(),
                        ),
                    )
                    .unwrap();
                latencies.record(op_start.elapsed());
                success.fetch_add(1, Ordering::Relaxed);
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_ops = thread_count * iterations_per_thread;
    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} ops/sec",
        total_ops as f64 / elapsed.as_secs_f64()
    );

    for (i, snapshot) in results.iter().enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    // Verify all writes succeeded
    assert_eq!(success_count.load(Ordering::Relaxed), total_ops);

    // Final store should have exactly num_keys entries
    assert_eq!(store.len(), num_keys);

    println!(
        "PASSED: {} ops completed, {} unique keys",
        total_ops, num_keys
    );
}

/// Stress test: Concurrent scans during writes.
#[test]
#[ignore]
fn stress_tensor_store_scan_during_writes() {
    let config = full_config();
    let entity_count = config.effective_entity_count() / 10; // Use 100K for this test
    let writer_threads = 4;
    let scanner_threads = 4;

    println!("\n=== TensorStore Scan During Writes ===");
    println!("Entities: {}", entity_count);
    println!("Writers: {}", writer_threads);
    println!("Scanners: {}", scanner_threads);

    let store = Arc::new(TensorStore::new());
    let embeddings = Arc::new(generate_embeddings(entity_count, 128, 42));
    let write_done = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let per_thread = entity_count / writer_threads;
    let mut handles = vec![];
    let start = Instant::now();

    // Writer threads
    for t in 0..writer_threads {
        let store = Arc::clone(&store);
        let embeddings = Arc::clone(&embeddings);
        let write_done = Arc::clone(&write_done);
        handles.push(thread::spawn(move || {
            let start_idx = t * per_thread;
            for i in 0..per_thread {
                let idx = start_idx + i;
                store
                    .put(
                        format!("prefix:key:{}", idx),
                        create_tensor(idx as i64, embeddings[idx].clone()),
                    )
                    .unwrap();
            }
            if t == 0 {
                write_done.store(true, Ordering::Release);
            }
        }));
    }

    // Scanner threads
    let scan_counts = Arc::new(AtomicUsize::new(0));
    for _ in 0..scanner_threads {
        let store = Arc::clone(&store);
        let write_done = Arc::clone(&write_done);
        let scan_counts = Arc::clone(&scan_counts);
        handles.push(thread::spawn(move || {
            let mut scans = 0;
            while !write_done.load(Ordering::Acquire) {
                let results = store.scan("prefix:");
                scans += 1;
                // Just count, don't validate during concurrent writes
                let _ = results.len();
            }
            scan_counts.fetch_add(scans, Ordering::Relaxed);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let elapsed = start.elapsed();

    println!("Duration: {:?}", elapsed);
    println!("Scans completed: {}", scan_counts.load(Ordering::Relaxed));
    assert_eq!(store.len(), entity_count);

    // Final scan should return all entries
    let final_scan = store.scan("prefix:");
    assert_eq!(final_scan.len(), entity_count);

    println!(
        "PASSED: {} entities, concurrent scans succeeded",
        entity_count
    );
}

/// Stress test: BloomFilter at 1M keys.
#[test]
#[ignore]
fn stress_bloom_filter_1m() {
    let config = full_config();
    let entity_count = config.effective_entity_count();

    println!("\n=== BloomFilter 1M Keys ===");
    println!("Keys: {}", entity_count);

    // Size for ~1% false positive rate at 1M items
    let bloom = Arc::new(BloomFilter::new(entity_count, 0.01));
    let thread_count = config.effective_thread_count();
    let per_thread = entity_count / thread_count;

    // Phase 1: Add all keys
    let start = Instant::now();
    let mut handles = vec![];

    for t in 0..thread_count {
        let bloom = Arc::clone(&bloom);
        handles.push(thread::spawn(move || {
            let start_idx = t * per_thread;
            for i in 0..per_thread {
                let key = format!("key:{}", start_idx + i);
                bloom.add(&key);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let add_elapsed = start.elapsed();

    println!(
        "Add phase: {:?} ({:.0} ops/sec)",
        add_elapsed,
        entity_count as f64 / add_elapsed.as_secs_f64()
    );

    // Phase 2: Check all keys exist (should all return true)
    let start = Instant::now();
    let mut handles = vec![];
    let true_positives = Arc::new(AtomicUsize::new(0));

    for t in 0..thread_count {
        let bloom = Arc::clone(&bloom);
        let tp = Arc::clone(&true_positives);
        handles.push(thread::spawn(move || {
            let start_idx = t * per_thread;
            for i in 0..per_thread {
                let key = format!("key:{}", start_idx + i);
                if bloom.might_contain(&key) {
                    tp.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let check_elapsed = start.elapsed();

    let tp_count = true_positives.load(Ordering::Relaxed);
    println!(
        "Check phase (existing): {:?} ({:.0} ops/sec)",
        check_elapsed,
        entity_count as f64 / check_elapsed.as_secs_f64()
    );
    println!(
        "True positives: {} / {} (should be 100%)",
        tp_count, entity_count
    );
    assert_eq!(tp_count, entity_count, "BloomFilter missed existing keys");

    // Phase 3: Check non-existent keys (measure false positive rate)
    let start = Instant::now();
    let mut handles = vec![];
    let false_positives = Arc::new(AtomicUsize::new(0));

    for t in 0..thread_count {
        let bloom = Arc::clone(&bloom);
        let fp = Arc::clone(&false_positives);
        handles.push(thread::spawn(move || {
            let start_idx = t * per_thread;
            for i in 0..per_thread {
                let key = format!("nonexistent:{}", start_idx + i);
                if bloom.might_contain(&key) {
                    fp.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let fp_elapsed = start.elapsed();

    let fp_count = false_positives.load(Ordering::Relaxed);
    let fp_rate = fp_count as f64 / entity_count as f64 * 100.0;
    println!(
        "Check phase (non-existent): {:?} ({:.0} ops/sec)",
        fp_elapsed,
        entity_count as f64 / fp_elapsed.as_secs_f64()
    );
    println!(
        "False positives: {} / {} ({:.2}%)",
        fp_count, entity_count, fp_rate
    );

    // FP rate should be reasonable (< 5% with our sizing)
    assert!(
        fp_rate < 5.0,
        "False positive rate too high: {:.2}%",
        fp_rate
    );

    println!(
        "PASSED: BloomFilter at {} keys, FP rate {:.2}%",
        entity_count, fp_rate
    );
}
