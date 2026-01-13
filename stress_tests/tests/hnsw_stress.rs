//! HNSW index stress tests at 100K-1M vector scale.

use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Instant,
};

use stress_tests::{full_config, generate_embeddings, LatencyHistogram};
use tensor_store::{HNSWConfig, HNSWIndex};

/// Stress test: Concurrent HNSW build with 100K vectors.
#[test]
#[ignore]
fn stress_hnsw_100k_concurrent_build() {
    let config = full_config();
    let vector_count = config.effective_entity_count() / 10; // 100K vectors
    let thread_count = config.effective_thread_count();
    let dim = config.embedding_dim;

    println!("\n=== HNSW 100K Concurrent Build ===");
    println!("Vectors: {}", vector_count);
    println!("Dimensions: {}", dim);
    println!("Threads: {}", thread_count);

    let index = Arc::new(HNSWIndex::new());
    let embeddings = Arc::new(generate_embeddings(vector_count, dim, 42));

    let per_thread = vector_count / thread_count;
    let mut handles = vec![];
    let start = Instant::now();

    for t in 0..thread_count {
        let index = Arc::clone(&index);
        let embeddings = Arc::clone(&embeddings);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let start_idx = t * per_thread;

            for i in 0..per_thread {
                let idx = start_idx + i;
                let op_start = Instant::now();
                index.insert(embeddings[idx].clone());
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} vectors/sec",
        vector_count as f64 / elapsed.as_secs_f64()
    );

    for (i, snapshot) in results.iter().enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    // Verify index size
    assert_eq!(index.len(), vector_count);

    // Test search quality
    let query = &embeddings[0];
    let results = index.search(query, 10);
    assert!(!results.is_empty());

    println!("PASSED: {} vectors indexed, search working", vector_count);
}

/// Stress test: Concurrent search during insert.
#[test]
#[ignore]
fn stress_hnsw_search_during_insert() {
    let config = full_config();
    let vector_count = 50_000; // 50K vectors for this test
    let dim = config.embedding_dim;
    let writer_threads = 4;
    let searcher_threads = 4;
    let searches_per_thread = 1000;

    println!("\n=== HNSW Search During Insert ===");
    println!("Vectors: {}", vector_count);
    println!("Writers: {}", writer_threads);
    println!("Searchers: {}", searcher_threads);

    let index = Arc::new(HNSWIndex::new());
    let embeddings = Arc::new(generate_embeddings(vector_count, dim, 42));
    let insert_done = Arc::new(AtomicBool::new(false));

    let per_thread = vector_count / writer_threads;
    let mut handles = vec![];
    let start = Instant::now();

    // Writer threads
    for t in 0..writer_threads {
        let index = Arc::clone(&index);
        let embeddings = Arc::clone(&embeddings);
        let insert_done = Arc::clone(&insert_done);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let start_idx = t * per_thread;

            for i in 0..per_thread {
                let idx = start_idx + i;
                let op_start = Instant::now();
                index.insert(embeddings[idx].clone());
                latencies.record(op_start.elapsed());
            }

            if t == writer_threads - 1 {
                insert_done.store(true, Ordering::Release);
            }

            ("writer", latencies.snapshot())
        }));
    }

    // Searcher threads
    let search_count = Arc::new(AtomicUsize::new(0));
    for t in 0..searcher_threads {
        let index = Arc::clone(&index);
        let embeddings = Arc::clone(&embeddings);
        let insert_done = Arc::clone(&insert_done);
        let search_count = Arc::clone(&search_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut searches = 0;

            // Search until inserts complete or we've done enough searches
            while !insert_done.load(Ordering::Acquire) && searches < searches_per_thread {
                // Use a random query from already-inserted range
                let query_idx = (t * 100 + searches) % vector_count.max(1);
                if query_idx < embeddings.len() {
                    let op_start = Instant::now();
                    let _ = index.search(&embeddings[query_idx], 10);
                    latencies.record(op_start.elapsed());
                    searches += 1;
                }
            }

            search_count.fetch_add(searches, Ordering::Relaxed);
            ("searcher", latencies.snapshot())
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    println!("Duration: {:?}", elapsed);
    println!("Total searches: {}", search_count.load(Ordering::Relaxed));

    for (i, (role, snapshot)) in results.iter().enumerate() {
        println!("  {} {}: {}", role, i, snapshot);
    }

    // Verify index completeness
    assert_eq!(index.len(), vector_count);

    println!("PASSED: {} vectors with concurrent searches", vector_count);
}

/// Stress test: 1M vectors at 128 dimensions.
#[test]
#[ignore]
fn stress_hnsw_1m_vectors() {
    let config = full_config();
    let vector_count = config.effective_entity_count();
    let dim = config.embedding_dim;
    let thread_count = config.effective_thread_count();

    println!("\n=== HNSW 1M Vectors ===");
    println!("Vectors: {}", vector_count);
    println!("Dimensions: {}", dim);
    println!("Threads: {}", thread_count);

    // Use high-speed config for large scale
    let hnsw_config = HNSWConfig::high_speed();
    let index = Arc::new(HNSWIndex::with_config(hnsw_config));

    let per_thread = vector_count / thread_count;
    let start = Instant::now();

    let mut handles = vec![];
    for t in 0..thread_count {
        let index = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();

            // Generate embeddings for this thread's range
            let embeddings = generate_embeddings(per_thread, dim, 42 + t as u64);

            for emb in embeddings.into_iter() {
                let op_start = Instant::now();
                index.insert(emb);
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} vectors/sec",
        vector_count as f64 / elapsed.as_secs_f64()
    );

    for (i, snapshot) in results.iter().enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    // Verify index size
    assert_eq!(index.len(), vector_count);

    // Test search performance
    let query_embeddings = generate_embeddings(100, dim, 999);
    let search_start = Instant::now();
    for query in &query_embeddings {
        let _ = index.search(query, 10);
    }
    let search_elapsed = search_start.elapsed();
    println!(
        "Search: 100 queries in {:?} ({:.2}ms/query)",
        search_elapsed,
        search_elapsed.as_secs_f64() * 1000.0 / 100.0
    );

    println!("PASSED: {} vectors indexed", vector_count);
}

/// Stress test: Verify recall under load.
#[test]
#[ignore]
fn stress_hnsw_recall_under_load() {
    let vector_count = 10_000;
    let dim = 128;
    let k = 10;
    let num_queries = 100;

    println!("\n=== HNSW Recall Under Load ===");
    println!("Vectors: {}", vector_count);
    println!("k: {}", k);
    println!("Queries: {}", num_queries);

    // Build index
    let hnsw_config = HNSWConfig::high_recall();
    let index = HNSWIndex::with_config(hnsw_config);
    let embeddings = generate_embeddings(vector_count, dim, 42);

    for emb in embeddings.iter() {
        index.insert(emb.clone());
    }

    // Compute ground truth via brute force
    let query_embeddings = generate_embeddings(num_queries, dim, 999);
    let mut recalls = vec![];

    for query in &query_embeddings {
        // Brute force k-NN
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
        let ground_truth: std::collections::HashSet<_> =
            distances.iter().take(k).map(|(i, _)| *i).collect();

        // HNSW search
        let results = index.search(query, k);
        let hnsw_results: std::collections::HashSet<_> = results.iter().map(|(i, _)| *i).collect();

        let recall = ground_truth.intersection(&hnsw_results).count() as f32 / k as f32;
        recalls.push(recall);
    }

    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
    let min_recall = recalls.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("Average recall@{}: {:.2}%", k, avg_recall * 100.0);
    println!("Min recall@{}: {:.2}%", k, min_recall * 100.0);

    // With high_recall config, we expect >90% recall
    assert!(
        avg_recall > 0.90,
        "Average recall too low: {:.2}%",
        avg_recall * 100.0
    );

    println!("PASSED: Recall@{} = {:.2}%", k, avg_recall * 100.0);
}
