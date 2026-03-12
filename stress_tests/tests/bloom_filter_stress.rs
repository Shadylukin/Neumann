// SPDX-License-Identifier: MIT OR Apache-2.0
//! BloomFilter stress tests.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Instant,
};

use stress_tests::{full_config, LatencyHistogram};
use tensor_store::BloomFilter;

/// Stress test: Concurrent add and query operations.
#[test]
#[ignore]
fn stress_bloom_concurrent_add_query() {
    let _config = full_config();
    let key_count = 100_000;
    let adder_threads = 8;
    let query_threads = 8;
    let queries_per_thread = 50_000;

    println!("\n=== BloomFilter Concurrent Add/Query ===");
    println!("Keys to add: {}", key_count);
    println!("Adder threads: {}", adder_threads);
    println!("Query threads: {}", query_threads);

    // Size for reasonable FP rate (~1%)
    let bloom = Arc::new(BloomFilter::new(key_count, 0.01));
    let add_complete = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let keys_per_thread = key_count / adder_threads;
    let mut handles = vec![];
    let start = Instant::now();

    // Adder threads
    for t in 0..adder_threads {
        let bloom = Arc::clone(&bloom);
        let add_complete = Arc::clone(&add_complete);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let start_idx = t * keys_per_thread;

            for i in 0..keys_per_thread {
                let key = format!("key:{}", start_idx + i);
                let op_start = Instant::now();
                bloom.add(&key);
                latencies.record(op_start.elapsed());
            }

            if t == adder_threads - 1 {
                add_complete.store(true, Ordering::Release);
            }

            ("adder", latencies.snapshot())
        }));
    }

    // Query threads (run concurrently with adds)
    let query_hits = Arc::new(AtomicUsize::new(0));
    for _t in 0..query_threads {
        let bloom = Arc::clone(&bloom);
        let hits = Arc::clone(&query_hits);
        let _add_complete = Arc::clone(&add_complete);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut local_hits = 0;

            for i in 0..queries_per_thread {
                // Query mix of existing and non-existing keys
                let key = if i % 2 == 0 {
                    format!("key:{}", i % key_count)
                } else {
                    format!("nonexistent:{}", i)
                };

                let op_start = Instant::now();
                if bloom.might_contain(&key) {
                    local_hits += 1;
                }
                latencies.record(op_start.elapsed());
            }

            hits.fetch_add(local_hits, Ordering::Relaxed);
            ("query", latencies.snapshot())
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_ops = key_count + query_threads * queries_per_thread;
    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} ops/sec",
        total_ops as f64 / elapsed.as_secs_f64()
    );
    println!("Query hits: {}", query_hits.load(Ordering::Relaxed));

    for (i, (role, snapshot)) in results.iter().enumerate() {
        if i < 3 || i >= results.len() - 2 {
            println!("  {} {}: {}", role, i, snapshot);
        }
    }

    println!("PASSED: Concurrent add/query completed");
}

/// Stress test: High-volume adds measuring FP rate.
#[test]
#[ignore]
fn stress_bloom_fp_rate_at_scale() {
    let config = full_config();
    let key_count = config.effective_entity_count();
    let thread_count = config.effective_thread_count();

    println!("\n=== BloomFilter FP Rate at Scale ===");
    println!("Keys: {}", key_count);
    println!("Threads: {}", thread_count);

    // Size filter for ~1% FP rate at target capacity
    let bloom = Arc::new(BloomFilter::new(key_count, 0.01));

    // Phase 1: Add all keys
    let keys_per_thread = key_count / thread_count;
    let mut handles = vec![];
    let start = Instant::now();

    for t in 0..thread_count {
        let bloom = Arc::clone(&bloom);
        handles.push(thread::spawn(move || {
            let start_idx = t * keys_per_thread;
            for i in 0..keys_per_thread {
                bloom.add(&format!("key:{}", start_idx + i));
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let add_elapsed = start.elapsed();
    println!(
        "Add phase: {:?} ({:.0} keys/sec)",
        add_elapsed,
        key_count as f64 / add_elapsed.as_secs_f64()
    );

    // Phase 2: Verify all keys exist
    let mut handles = vec![];
    let found = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    for t in 0..thread_count {
        let bloom = Arc::clone(&bloom);
        let found = Arc::clone(&found);
        handles.push(thread::spawn(move || {
            let start_idx = t * keys_per_thread;
            let mut local_found = 0;
            for i in 0..keys_per_thread {
                if bloom.might_contain(&format!("key:{}", start_idx + i)) {
                    local_found += 1;
                }
            }
            found.fetch_add(local_found, Ordering::Relaxed);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let verify_elapsed = start.elapsed();

    let found_count = found.load(Ordering::Relaxed);
    println!(
        "Verify phase: {:?} ({:.0} checks/sec)",
        verify_elapsed,
        key_count as f64 / verify_elapsed.as_secs_f64()
    );
    println!("Found: {} / {} (should be 100%)", found_count, key_count);
    assert_eq!(found_count, key_count, "BloomFilter missed existing keys");

    // Phase 3: Measure false positive rate
    let mut handles = vec![];
    let false_positives = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    for t in 0..thread_count {
        let bloom = Arc::clone(&bloom);
        let fp = Arc::clone(&false_positives);
        handles.push(thread::spawn(move || {
            let start_idx = t * keys_per_thread;
            let mut local_fp = 0;
            for i in 0..keys_per_thread {
                if bloom.might_contain(&format!("nonexistent:{}", start_idx + i)) {
                    local_fp += 1;
                }
            }
            fp.fetch_add(local_fp, Ordering::Relaxed);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let fp_elapsed = start.elapsed();

    let fp_count = false_positives.load(Ordering::Relaxed);
    let fp_rate = fp_count as f64 / key_count as f64 * 100.0;
    println!(
        "FP check phase: {:?} ({:.0} checks/sec)",
        fp_elapsed,
        key_count as f64 / fp_elapsed.as_secs_f64()
    );
    println!(
        "False positives: {} / {} ({:.3}%)",
        fp_count, key_count, fp_rate
    );

    // FP rate should be reasonable
    assert!(fp_rate < 5.0, "FP rate too high: {:.3}%", fp_rate);

    println!("PASSED: FP rate at {} keys = {:.3}%", key_count, fp_rate);
}

/// Stress test: Bit-level concurrency safety.
#[test]
#[ignore]
fn stress_bloom_bit_concurrency() {
    let size = 1_000_000;
    let thread_count = 32;
    let ops_per_thread = 100_000;

    println!("\n=== BloomFilter Bit Concurrency ===");
    println!("Filter size: {} bits", size * 64);
    println!("Threads: {}", thread_count);
    println!("Ops/thread: {}", ops_per_thread);

    let bloom = Arc::new(BloomFilter::new(size, 0.01));
    let total_adds = Arc::new(AtomicUsize::new(0));

    let mut handles = vec![];
    let start = Instant::now();

    for _t in 0..thread_count {
        let bloom = Arc::clone(&bloom);
        let adds = Arc::clone(&total_adds);
        handles.push(thread::spawn(move || {
            for i in 0..ops_per_thread {
                // All threads add overlapping keys to maximize bit contention
                let key = format!("shared:{}", i % 10_000);
                bloom.add(&key);
                adds.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
    let elapsed = start.elapsed();

    let total = total_adds.load(Ordering::Relaxed);
    println!("Duration: {:?}", elapsed);
    println!("Total adds: {}", total);
    println!(
        "Throughput: {:.0} ops/sec",
        total as f64 / elapsed.as_secs_f64()
    );

    // Verify all shared keys are present
    let mut missing = 0;
    for i in 0..10_000 {
        if !bloom.might_contain(&format!("shared:{}", i)) {
            missing += 1;
        }
    }
    println!("Missing shared keys: {} / 10000", missing);
    assert_eq!(missing, 0, "BloomFilter lost keys under contention");

    println!("PASSED: Bit-level concurrency safe");
}
