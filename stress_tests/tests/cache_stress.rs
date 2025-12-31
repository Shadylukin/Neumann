//! Cache stress tests at 10K-100K entry scale.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use stress_tests::{full_config, generate_embeddings, LatencyHistogram};
use tensor_cache::{Cache, CacheConfig, EvictionStrategy};

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Stress test: 10K exact cache entries with latency measurement.
#[test]
#[ignore]
fn stress_cache_10k_exact_entries() {
    let config = full_config();
    let entry_count = 10_000;
    let thread_count = config.effective_thread_count().min(8);

    println!("\n=== Cache 10K Exact Entries ===");
    println!("Entries: {}", entry_count);
    println!("Threads: {}", thread_count);

    let cache = Arc::new(Cache::new());
    let per_thread = entry_count / thread_count;

    // Write phase
    let start = Instant::now();
    let mut handles = vec![];

    for t in 0..thread_count {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let start_idx = t * per_thread;

            for i in 0..per_thread {
                let key = format!("key_{}_{}", t, i);
                let value = format!("value for key {} in thread {} with some padding data", i, t);
                let op_start = Instant::now();
                let _ = cache.put_simple(&key, &value);
                latencies.record(op_start.elapsed());
            }

            ("write", latencies.snapshot())
        }));
    }

    let write_results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let write_elapsed = start.elapsed();

    println!("Write phase: {:?}", write_elapsed);
    println!(
        "Write throughput: {:.0} entries/sec",
        entry_count as f64 / write_elapsed.as_secs_f64()
    );

    for (i, (role, snapshot)) in write_results.iter().enumerate() {
        println!("  {} {}: {}", role, i, snapshot);
    }

    // Read phase
    let mut handles = vec![];
    let read_start = Instant::now();

    for t in 0..thread_count {
        let cache = Arc::clone(&cache);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut hits = 0;

            for i in 0..per_thread {
                let key = format!("key_{}_{}", t, i);
                let op_start = Instant::now();
                if cache.get_simple(&key).is_some() {
                    hits += 1;
                }
                latencies.record(op_start.elapsed());
            }

            (hits, latencies.snapshot())
        }));
    }

    let read_results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let read_elapsed = read_start.elapsed();

    let total_hits: usize = read_results.iter().map(|(h, _)| h).sum();

    println!("\nRead phase: {:?}", read_elapsed);
    println!(
        "Read throughput: {:.0} entries/sec",
        entry_count as f64 / read_elapsed.as_secs_f64()
    );
    println!("Hit rate: {:.2}%", total_hits as f64 / entry_count as f64 * 100.0);

    for (i, (_, snapshot)) in read_results.iter().enumerate() {
        println!("  read {}: {}", i, snapshot);
    }

    // Verify cache size
    let stats = cache.stats_snapshot();
    assert!(stats.total_entries() > 0, "Cache should have entries");

    println!("PASSED: {} entries stored and retrieved", entry_count);
}

/// Stress test: Semantic cache throughput with embeddings.
#[test]
#[ignore]
fn stress_cache_semantic_throughput() {
    let config = full_config();
    let entry_count = 5_000;
    let dim = 64; // Smaller dim for faster test
    let thread_count = config.effective_thread_count().min(4);

    println!("\n=== Cache Semantic Throughput ===");
    println!("Entries: {}", entry_count);
    println!("Dimensions: {}", dim);
    println!("Threads: {}", thread_count);

    let cache_config = CacheConfig {
        embedding_dim: dim,
        semantic_capacity: entry_count * 2,
        exact_capacity: entry_count * 2,
        semantic_threshold: 0.85,
        ..Default::default()
    };
    let cache = Arc::new(Cache::with_config(cache_config).unwrap());
    let embeddings = Arc::new(generate_embeddings(entry_count, dim, 42));

    // Normalize embeddings for cosine similarity
    let normalized: Vec<Vec<f32>> = embeddings.iter().map(|e| normalize(e)).collect();
    let normalized = Arc::new(normalized);

    // Insert phase
    let per_thread = entry_count / thread_count;
    let mut handles = vec![];
    let start = Instant::now();

    for t in 0..thread_count {
        let cache = Arc::clone(&cache);
        let normalized = Arc::clone(&normalized);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let start_idx = t * per_thread;

            for i in 0..per_thread {
                let idx = start_idx + i;
                let prompt = format!("semantic_prompt_{}", idx);
                let response = format!("response for semantic query {}", idx);
                let op_start = Instant::now();
                let _ = cache.put(&prompt, &normalized[idx], &response, "test-model", None);
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let insert_results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let insert_elapsed = start.elapsed();

    println!("Insert phase: {:?}", insert_elapsed);
    println!(
        "Insert throughput: {:.0} entries/sec",
        entry_count as f64 / insert_elapsed.as_secs_f64()
    );

    for (i, snapshot) in insert_results.iter().enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    // Search phase - query with stored embeddings
    let mut handles = vec![];
    let search_start = Instant::now();
    let queries_per_thread = 500;

    for t in 0..thread_count {
        let cache = Arc::clone(&cache);
        let normalized = Arc::clone(&normalized);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut hits = 0;

            for i in 0..queries_per_thread {
                let idx = (t * queries_per_thread + i) % entry_count;
                let prompt = format!("semantic_prompt_{}", idx);
                let op_start = Instant::now();
                if cache.get(&prompt, Some(&normalized[idx])).is_some() {
                    hits += 1;
                }
                latencies.record(op_start.elapsed());
            }

            (hits, latencies.snapshot())
        }));
    }

    let search_results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let search_elapsed = search_start.elapsed();

    let total_queries = queries_per_thread * thread_count;
    let total_hits: usize = search_results.iter().map(|(h, _)| h).sum();

    println!("\nSearch phase: {:?}", search_elapsed);
    println!(
        "Search throughput: {:.0} queries/sec",
        total_queries as f64 / search_elapsed.as_secs_f64()
    );
    println!("Hit rate: {:.2}%", total_hits as f64 / total_queries as f64 * 100.0);

    for (i, (_, snapshot)) in search_results.iter().enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    println!("PASSED: Semantic cache with {} entries", entry_count);
}

/// Stress test: Eviction performance at scale.
#[test]
#[ignore]
fn stress_cache_eviction_performance() {
    println!("\n=== Cache Eviction Performance ===");

    for scale in [10_000, 50_000] {
        let config = CacheConfig {
            exact_capacity: scale / 2, // Force eviction at half capacity
            eviction_batch_size: 100,
            eviction_strategy: EvictionStrategy::Hybrid {
                lru_weight: 40,
                lfu_weight: 30,
                cost_weight: 30,
            },
            ..Default::default()
        };
        let cache = Cache::with_config(config).unwrap();

        println!("\n--- Scale: {} entries (capacity: {}) ---", scale, scale / 2);

        // Fill cache beyond capacity
        let start = Instant::now();
        let mut inserted = 0;
        let mut rejected = 0;

        for i in 0..scale {
            let key = format!("evict_key_{}", i);
            let value = format!("value_{}", i);
            match cache.put_simple(&key, &value) {
                Ok(()) => inserted += 1,
                Err(_) => rejected += 1,
            }
        }

        let fill_elapsed = start.elapsed();
        println!("Fill phase: {:?}", fill_elapsed);
        println!("Inserted: {}, Rejected: {}", inserted, rejected);

        // Manual eviction
        let evict_start = Instant::now();
        let evicted = cache.evict(1000);
        let evict_elapsed = evict_start.elapsed();

        println!("Eviction (1000 entries): {:?}", evict_elapsed);
        println!("Actually evicted: {}", evicted);

        let stats = cache.stats_snapshot();
        println!("Final entries: {}", stats.total_entries());
    }

    println!("\nPASSED: Eviction performance test");
}

/// Stress test: Concurrent readers and writers.
#[test]
#[ignore]
fn stress_cache_concurrent_access() {
    let reader_count = 8;
    let writer_count = 4;
    let ops_per_thread = 10_000;

    println!("\n=== Cache Concurrent Access ===");
    println!("Readers: {}", reader_count);
    println!("Writers: {}", writer_count);
    println!("Ops per thread: {}", ops_per_thread);

    let cache = Arc::new(Cache::new());
    let running = Arc::new(AtomicBool::new(true));
    let read_count = Arc::new(AtomicUsize::new(0));
    let write_count = Arc::new(AtomicUsize::new(0));
    let read_hits = Arc::new(AtomicUsize::new(0));

    // Pre-populate with some entries
    for i in 0..1000 {
        let _ = cache.put_simple(&format!("initial_{}", i), &format!("value_{}", i));
    }

    let mut handles = vec![];
    let start = Instant::now();

    // Writer threads
    for t in 0..writer_count {
        let cache = Arc::clone(&cache);
        let running = Arc::clone(&running);
        let write_count = Arc::clone(&write_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut writes = 0;

            while running.load(Ordering::Acquire) && writes < ops_per_thread {
                let key = format!("writer_{}_{}", t, writes);
                let value = format!("value_{}_{}", t, writes);
                let op_start = Instant::now();
                let _ = cache.put_simple(&key, &value);
                latencies.record(op_start.elapsed());
                writes += 1;
            }

            write_count.fetch_add(writes, Ordering::Relaxed);
            ("writer", latencies.snapshot())
        }));
    }

    // Reader threads
    for t in 0..reader_count {
        let cache = Arc::clone(&cache);
        let running = Arc::clone(&running);
        let read_count = Arc::clone(&read_count);
        let read_hits = Arc::clone(&read_hits);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut reads = 0;
            let mut hits = 0;

            while running.load(Ordering::Acquire) && reads < ops_per_thread {
                // Read mix of initial and potentially written keys
                let key = if reads % 2 == 0 {
                    format!("initial_{}", reads % 1000)
                } else {
                    format!("writer_{}_{}", t % writer_count, reads % ops_per_thread)
                };
                let op_start = Instant::now();
                if cache.get_simple(&key).is_some() {
                    hits += 1;
                }
                latencies.record(op_start.elapsed());
                reads += 1;
            }

            read_count.fetch_add(reads, Ordering::Relaxed);
            read_hits.fetch_add(hits, Ordering::Relaxed);
            ("reader", latencies.snapshot())
        }));
    }

    // Wait for completion (or timeout)
    thread::sleep(Duration::from_millis(100));
    running.store(false, Ordering::Release);

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_reads = read_count.load(Ordering::Relaxed);
    let total_writes = write_count.load(Ordering::Relaxed);
    let total_hits = read_hits.load(Ordering::Relaxed);
    let total_ops = total_reads + total_writes;

    println!("Duration: {:?}", elapsed);
    println!("Total operations: {}", total_ops);
    println!(
        "Throughput: {:.0} ops/sec",
        total_ops as f64 / elapsed.as_secs_f64()
    );
    println!("Reads: {}, Writes: {}", total_reads, total_writes);
    println!(
        "Read hit rate: {:.2}%",
        if total_reads > 0 {
            total_hits as f64 / total_reads as f64 * 100.0
        } else {
            0.0
        }
    );

    for (i, (role, snapshot)) in results.iter().enumerate() {
        println!("  {} {}: {}", role, i, snapshot);
    }

    // Verify no crashes, data is accessible
    let stats = cache.stats_snapshot();
    assert!(stats.total_entries() > 0, "Cache should have entries");

    println!("PASSED: Concurrent access with {} threads", reader_count + writer_count);
}

/// Stress test: Memory stability under sustained load.
#[test]
#[ignore]
fn stress_cache_memory_stability() {
    let duration_secs = std::env::var("STRESS_DURATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30); // Default 30 seconds, override with STRESS_DURATION

    println!("\n=== Cache Memory Stability ===");
    println!("Duration: {}s", duration_secs);

    let config = CacheConfig {
        exact_capacity: 10_000,
        eviction_batch_size: 100,
        ..Default::default()
    };
    let cache = Arc::new(Cache::with_config(config).unwrap());
    let running = Arc::new(AtomicBool::new(true));
    let ops = Arc::new(AtomicUsize::new(0));

    let thread_count = 4;
    let mut handles = vec![];
    let start = Instant::now();

    for t in 0..thread_count {
        let cache = Arc::clone(&cache);
        let running = Arc::clone(&running);
        let ops = Arc::clone(&ops);
        handles.push(thread::spawn(move || {
            let mut local_ops = 0u64;
            let mut i = 0u64;

            while running.load(Ordering::Relaxed) {
                let key = format!("stability_{}_{}", t, i % 20_000);
                let value = format!("value with some padding to simulate real data {}", i);

                // Mix of writes and reads
                if i % 3 == 0 {
                    let _ = cache.put_simple(&key, &value);
                } else {
                    let _ = cache.get_simple(&key);
                }

                local_ops += 1;
                i += 1;

                // Periodic stats check
                if i % 10_000 == 0 {
                    let stats = cache.stats_snapshot();
                    let _ = stats.total_entries(); // Access stats
                }
            }

            ops.fetch_add(local_ops as usize, Ordering::Relaxed);
        }));
    }

    // Run for specified duration
    thread::sleep(Duration::from_secs(duration_secs));
    running.store(false, Ordering::Release);

    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start.elapsed();
    let total_ops = ops.load(Ordering::Relaxed);

    println!("Completed: {} operations", total_ops);
    println!(
        "Throughput: {:.0} ops/sec",
        total_ops as f64 / elapsed.as_secs_f64()
    );

    let stats = cache.stats_snapshot();
    println!("Final entries: {}", stats.total_entries());
    println!("Exact hits: {}", stats.exact_hits);
    println!("Exact misses: {}", stats.exact_misses);

    // Verify cache is still functional
    cache.put_simple("final_test", "value").unwrap();
    assert!(cache.get_simple("final_test").is_some());

    println!("PASSED: Memory stability over {}s", duration_secs);
}
