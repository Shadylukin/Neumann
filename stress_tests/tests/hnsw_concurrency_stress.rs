// SPDX-License-Identifier: MIT OR Apache-2.0
//! HNSW index concurrency stress tests.
//!
//! Tests high-thread-count concurrent insert and search operations.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
    time::Instant,
};

use stress_tests::{full_config, generate_embeddings, LatencyHistogram};
use tensor_store::HNSWIndex;

/// Stress test: 50 threads inserting vectors concurrently.
#[test]
#[ignore]
fn stress_hnsw_concurrent_insert_50_threads() {
    let config = full_config();
    let thread_count = 50;
    let vectors_per_thread = 2000;
    let dim = config.embedding_dim;

    println!("\n=== HNSW Concurrent Insert 50 Threads ===");
    println!("Threads: {thread_count}");
    println!("Vectors per thread: {vectors_per_thread}");
    println!("Dimensions: {dim}");
    println!("Total vectors: {}", thread_count * vectors_per_thread);

    let index = Arc::new(HNSWIndex::new());
    let barrier = Arc::new(Barrier::new(thread_count));
    let success_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let idx = Arc::clone(&index);
            let bar = Arc::clone(&barrier);
            let cnt = Arc::clone(&success_count);
            thread::spawn(move || {
                let mut latencies = LatencyHistogram::new();

                // Generate embeddings for this thread
                let embeddings = generate_embeddings(vectors_per_thread, dim, 42 + t as u64);

                bar.wait();

                for emb in embeddings {
                    let op_start = Instant::now();
                    idx.insert(emb);
                    latencies.record(op_start.elapsed());
                    cnt.fetch_add(1, Ordering::Relaxed);
                }

                latencies.snapshot()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total = success_count.load(Ordering::SeqCst);
    let expected = thread_count * vectors_per_thread;

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} inserts/sec",
        total as f64 / elapsed.as_secs_f64()
    );

    // Sample latencies
    for (i, snapshot) in results.iter().take(5).enumerate() {
        println!("  Thread {i}: {snapshot}");
    }

    // Verify index size
    assert_eq!(index.len(), expected, "index size mismatch");
    assert_eq!(total, expected, "some inserts failed");

    // Test search quality
    let query = generate_embeddings(1, dim, 999).pop().unwrap();
    let search_results = index.search(&query, 10);
    assert!(!search_results.is_empty(), "search should return results");

    println!("PASSED: {total} vectors indexed");
}

/// Stress test: 100 threads mixed insert and search.
#[test]
#[ignore]
fn stress_hnsw_mixed_insert_search_100_threads() {
    let config = full_config();
    let inserter_count = 50;
    let searcher_count = 50;
    let thread_count = inserter_count + searcher_count;
    let vectors_per_inserter = 1000;
    let searches_per_searcher = 500;
    let dim = config.embedding_dim;

    println!("\n=== HNSW Mixed Insert/Search 100 Threads ===");
    println!("Inserters: {inserter_count}");
    println!("Searchers: {searcher_count}");
    println!("Vectors per inserter: {vectors_per_inserter}");
    println!("Searches per searcher: {searches_per_searcher}");

    let index = Arc::new(HNSWIndex::new());

    // Pre-populate with some vectors so searches have results
    let initial = generate_embeddings(1000, dim, 0);
    for emb in initial {
        index.insert(emb);
    }

    let barrier = Arc::new(Barrier::new(thread_count));
    let insert_count = Arc::new(AtomicUsize::new(0));
    let search_count = Arc::new(AtomicUsize::new(0));
    let insert_done = Arc::new(AtomicBool::new(false));
    let start = Instant::now();

    let mut handles = vec![];

    // Inserter threads
    for t in 0..inserter_count {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&insert_count);
        let done = Arc::clone(&insert_done);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let embeddings = generate_embeddings(vectors_per_inserter, dim, 100 + t as u64);

            bar.wait();

            for emb in embeddings {
                let op_start = Instant::now();
                idx.insert(emb);
                latencies.record(op_start.elapsed());
                cnt.fetch_add(1, Ordering::Relaxed);
            }

            // Signal completion after all inserters finish
            if t == inserter_count - 1 {
                done.store(true, Ordering::Release);
            }

            ("inserter", latencies.snapshot())
        }));
    }

    // Searcher threads
    for t in 0..searcher_count {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let cnt = Arc::clone(&search_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let queries = generate_embeddings(searches_per_searcher, dim, 1000 + t as u64);

            bar.wait();

            for query in queries {
                let op_start = Instant::now();
                let results = idx.search(&query, 10);
                latencies.record(op_start.elapsed());

                // Count successful searches (those returning results)
                if !results.is_empty() {
                    cnt.fetch_add(1, Ordering::Relaxed);
                }
            }

            ("searcher", latencies.snapshot())
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let inserts = insert_count.load(Ordering::SeqCst);
    let searches = search_count.load(Ordering::SeqCst);

    println!("Duration: {:?}", elapsed);
    println!("Inserts completed: {inserts}");
    println!("Searches with results: {searches}");

    // Show sample latencies
    let inserter_latencies: Vec<_> = results
        .iter()
        .filter(|(role, _)| *role == "inserter")
        .take(3)
        .collect();
    let searcher_latencies: Vec<_> = results
        .iter()
        .filter(|(role, _)| *role == "searcher")
        .take(3)
        .collect();

    println!("Sample inserter latencies:");
    for (i, (_, snapshot)) in inserter_latencies.iter().enumerate() {
        println!("  Inserter {i}: {snapshot}");
    }

    println!("Sample searcher latencies:");
    for (i, (_, snapshot)) in searcher_latencies.iter().enumerate() {
        println!("  Searcher {i}: {snapshot}");
    }

    // Verify counts
    let expected_inserts = inserter_count * vectors_per_inserter;
    assert_eq!(inserts, expected_inserts, "insert count mismatch");

    // Most searches should find results (initial 1000 + concurrent inserts)
    let expected_searches = searcher_count * searches_per_searcher;
    let search_success_rate = searches as f64 / expected_searches as f64;
    println!("Search success rate: {:.1}%", search_success_rate * 100.0);
    assert!(
        search_success_rate > 0.5,
        "too many searches failed: {search_success_rate:.1}%"
    );

    // Verify final index size
    let final_size = index.len();
    assert!(
        final_size >= 1000 + expected_inserts,
        "final index size too small: {final_size}"
    );

    println!("PASSED: concurrent insert/search completed");
}

/// Stress test: Search latency under heavy insert load.
#[test]
#[ignore]
fn stress_hnsw_search_latency_under_load() {
    let config = full_config();
    let dim = config.embedding_dim;
    let inserter_count = 20;
    let vectors_per_inserter = 5000;
    let search_count = 100;

    println!("\n=== HNSW Search Latency Under Load ===");
    println!("Background inserters: {inserter_count}");
    println!("Vectors per inserter: {vectors_per_inserter}");
    println!("Search queries: {search_count}");

    let index = Arc::new(HNSWIndex::new());

    // Pre-populate
    let initial = generate_embeddings(5000, dim, 0);
    for emb in initial {
        index.insert(emb);
    }

    let insert_done = Arc::new(AtomicBool::new(false));
    let barrier = Arc::new(Barrier::new(inserter_count + 1));

    // Start background inserters
    let mut inserter_handles = vec![];
    for t in 0..inserter_count {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let done = Arc::clone(&insert_done);
        inserter_handles.push(thread::spawn(move || {
            let embeddings = generate_embeddings(vectors_per_inserter, dim, 100 + t as u64);
            bar.wait();

            for emb in embeddings {
                if done.load(Ordering::Acquire) {
                    break;
                }
                idx.insert(emb);
            }
        }));
    }

    // Run timed searches
    let queries = generate_embeddings(search_count, dim, 999);
    let mut latencies = LatencyHistogram::new();

    barrier.wait(); // Start inserters

    for query in queries {
        let op_start = Instant::now();
        let results = index.search(&query, 10);
        latencies.record(op_start.elapsed());
        assert!(!results.is_empty(), "search should return results");
    }

    // Signal inserters to stop
    insert_done.store(true, Ordering::Release);

    for h in inserter_handles {
        h.join().expect("inserter should not panic");
    }

    let snapshot = latencies.snapshot();
    println!("Search latency under load:");
    println!("  {snapshot}");

    // p99 should be reasonable even under load
    let p99_ms = snapshot.p99.as_millis();
    println!("p99 latency: {p99_ms}ms (target: <50ms)");
    assert!(p99_ms < 100, "p99 latency too high: {p99_ms}ms");

    println!("PASSED: search latency under load acceptable");
}

/// Stress test: Recall quality under concurrent modifications.
#[test]
#[ignore]
fn stress_hnsw_recall_under_concurrent_load() {
    let dim = 128;
    let initial_count = 10_000;
    let k = 10;
    let query_count = 50;
    let modifier_count = 10;
    let modifications_per_thread = 1000;

    println!("\n=== HNSW Recall Under Concurrent Load ===");
    println!("Initial vectors: {initial_count}");
    println!("k: {k}");
    println!("Queries: {query_count}");
    println!("Modifiers: {modifier_count}");

    let index = Arc::new(HNSWIndex::new());

    // Build initial index
    let embeddings = Arc::new(generate_embeddings(initial_count, dim, 42));
    for emb in embeddings.iter() {
        index.insert(emb.clone());
    }

    // Generate queries
    let queries = generate_embeddings(query_count, dim, 999);

    // Compute ground truth before modifications
    let ground_truths: Vec<std::collections::HashSet<usize>> = queries
        .iter()
        .map(|query| {
            let mut distances: Vec<(usize, f32)> = embeddings
                .iter()
                .enumerate()
                .map(|(i, emb)| {
                    let dist: f32 = query
                        .iter()
                        .zip(emb.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (i, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect();

    // Start concurrent modifiers
    let barrier = Arc::new(Barrier::new(modifier_count + 1));
    let done = Arc::new(AtomicBool::new(false));

    let mut modifier_handles = vec![];
    for t in 0..modifier_count {
        let idx = Arc::clone(&index);
        let bar = Arc::clone(&barrier);
        let d = Arc::clone(&done);
        modifier_handles.push(thread::spawn(move || {
            let new_embeddings =
                generate_embeddings(modifications_per_thread, dim, 1000 + t as u64);
            bar.wait();

            for emb in new_embeddings {
                if d.load(Ordering::Acquire) {
                    break;
                }
                idx.insert(emb);
            }
        }));
    }

    barrier.wait(); // Start modifiers

    // Run queries and measure recall
    let mut recalls = vec![];
    for (query, ground_truth) in queries.iter().zip(ground_truths.iter()) {
        let results = index.search(query, k);
        let hnsw_results: std::collections::HashSet<_> = results.iter().map(|(i, _)| *i).collect();

        let recall = ground_truth.intersection(&hnsw_results).count() as f32 / k as f32;
        recalls.push(recall);
    }

    done.store(true, Ordering::Release);

    for h in modifier_handles {
        h.join().expect("modifier should not panic");
    }

    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
    let min_recall = recalls.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("Average recall@{k}: {:.2}%", avg_recall * 100.0);
    println!("Min recall@{k}: {:.2}%", min_recall * 100.0);

    // Recall may degrade slightly under load, but should remain reasonable
    assert!(
        avg_recall > 0.70,
        "average recall too low: {:.2}%",
        avg_recall * 100.0
    );

    println!(
        "PASSED: recall under concurrent load = {:.2}%",
        avg_recall * 100.0
    );
}
