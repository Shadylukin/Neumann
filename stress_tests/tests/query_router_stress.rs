// SPDX-License-Identifier: MIT OR Apache-2.0
//! QueryRouter async stress tests.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

use stress_tests::{generate_embeddings, LatencyHistogram};
use tensor_store::TensorStore;
use tokio::sync::Barrier;

/// Stress test: 100 concurrent query streams.
#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
#[ignore]
async fn stress_router_concurrent_queries() {
    let concurrent_streams = 100;
    let queries_per_stream = 100;

    println!("\n=== QueryRouter Concurrent Queries ===");
    println!("Streams: {}", concurrent_streams);
    println!("Queries/stream: {}", queries_per_stream);

    let store = TensorStore::new();
    let router = Arc::new(query_router::QueryRouter::with_shared_store(store));

    // Create table first (Neumann syntax uses colon for types)
    router
        .execute("CREATE TABLE test (id:INT, name:TEXT)")
        .expect("create table");

    // Pre-populate with some data
    let _embeddings = generate_embeddings(1000, 128, 42);
    let mut node_ids = Vec::new();
    for i in 0..100 {
        // Neumann INSERT syntax: INSERT table col=val, col=val
        router
            .execute(&format!("INSERT test id={}, name='entity{}'", i, i))
            .expect("insert");
        // Neumann NODE syntax: NODE CREATE <label> <prop>=<value>
        // Capture the actual node ID returned
        match router.execute(&format!("NODE CREATE entity idx={}, type='test'", i)) {
            Ok(query_router::QueryResult::Ids(ids)) => node_ids.push(ids[0]),
            _ => panic!("Failed to create node"),
        }
    }

    let node_ids = Arc::new(node_ids);
    let barrier = Arc::new(Barrier::new(concurrent_streams));
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let mut handles = vec![];

    for stream_id in 0..concurrent_streams {
        let router = Arc::clone(&router);
        let barrier = Arc::clone(&barrier);
        let success = Arc::clone(&success_count);
        let errors = Arc::clone(&error_count);
        let node_ids = Arc::clone(&node_ids);

        handles.push(tokio::spawn(async move {
            // Wait for all streams to be ready
            barrier.wait().await;

            let mut latencies = LatencyHistogram::new();

            for i in 0..queries_per_stream {
                // Use Neumann query syntax - mix of SELECTs, INSERTs, and NODE operations
                let query = match i % 4 {
                    0 => "SELECT * FROM test LIMIT 10".to_string(),
                    1 => format!("NODE GET {}", node_ids[stream_id % 100]),
                    2 => format!(
                        "INSERT test id={}, name='stream{}'",
                        stream_id * 1000 + i,
                        stream_id
                    ),
                    _ => "NODE LIST entity".to_string(),
                };

                let op_start = Instant::now();
                match router.execute(&query) {
                    Ok(_) => success.fetch_add(1, Ordering::Relaxed),
                    Err(_) => errors.fetch_add(1, Ordering::Relaxed),
                };
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    let elapsed = start.elapsed();

    let total_queries = concurrent_streams * queries_per_stream;
    let successes = success_count.load(Ordering::Relaxed);
    let errors = error_count.load(Ordering::Relaxed);

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} queries/sec",
        total_queries as f64 / elapsed.as_secs_f64()
    );
    println!("Successes: {}", successes);
    println!("Errors: {}", errors);

    // Sample latency stats
    println!("Sample stream latencies:");
    for i in [0, concurrent_streams / 2, concurrent_streams - 1] {
        println!("  Stream {}: {}", i, results[i]);
    }

    // Most queries should succeed
    let success_rate = successes as f64 / total_queries as f64;
    println!("Success rate: {:.2}%", success_rate * 100.0);

    assert!(
        success_rate > 0.9,
        "Success rate too low: {:.2}%",
        success_rate * 100.0
    );

    println!("PASSED: {} concurrent query streams", concurrent_streams);
}

/// Stress test: Parallel inserts via router.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn stress_router_parallel_inserts() {
    let stream_count = 50;
    let inserts_per_stream = 100;

    println!("\n=== QueryRouter Parallel Inserts ===");
    println!("Streams: {}", stream_count);
    println!("Inserts/stream: {}", inserts_per_stream);

    let store = TensorStore::new();
    let router = Arc::new(query_router::QueryRouter::with_shared_store(store));

    // Create table first (Neumann syntax)
    router
        .execute("CREATE TABLE parallel_test (id:INT, stream:INT, value:TEXT)")
        .expect("create table");

    let barrier = Arc::new(Barrier::new(stream_count));
    let success_count = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();
    let mut handles = vec![];

    for stream_id in 0..stream_count {
        let router = Arc::clone(&router);
        let barrier = Arc::clone(&barrier);
        let success = Arc::clone(&success_count);

        handles.push(tokio::spawn(async move {
            // Wait for all streams to be ready
            barrier.wait().await;

            let mut latencies = LatencyHistogram::new();

            for i in 0..inserts_per_stream {
                let op_start = Instant::now();
                // Neumann INSERT syntax
                let query = format!(
                    "INSERT parallel_test id={}, stream={}, value='data_{}'",
                    stream_id * 1000 + i,
                    stream_id,
                    i
                );

                if router.execute(&query).is_ok() {
                    success.fetch_add(1, Ordering::Relaxed);
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    let elapsed = start.elapsed();

    let total_inserts = stream_count * inserts_per_stream;
    let successes = success_count.load(Ordering::Relaxed);

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} inserts/sec",
        total_inserts as f64 / elapsed.as_secs_f64()
    );
    println!("Successes: {} / {}", successes, total_inserts);

    // Sample latency stats
    println!("Sample stream latencies:");
    for i in [0, stream_count / 2, stream_count - 1] {
        println!("  Stream {}: {}", i, results[i]);
    }

    let success_rate = successes as f64 / total_inserts as f64;
    println!("Success rate: {:.2}%", success_rate * 100.0);

    assert!(
        success_rate > 0.9,
        "Success rate too low: {:.2}%",
        success_rate * 100.0
    );

    println!("PASSED: {} parallel insert streams", stream_count);
}

/// Stress test: Sustained write load via router.
#[test]
#[ignore]
fn stress_router_sustained_writes() {
    use std::{sync::Mutex, thread, time::Duration};

    let test_duration_secs = 10;
    let thread_count = 8;

    println!("\n=== QueryRouter Sustained Write Load ===");
    println!("Threads: {}", thread_count);
    println!("Duration: {}s", test_duration_secs);

    let store = TensorStore::new();
    let router = query_router::QueryRouter::with_shared_store(store);

    // Create table for stress test (Neumann syntax)
    router
        .execute("CREATE TABLE stress_test (id:INT, thread:INT, iter:INT)")
        .expect("create table");

    let router = Arc::new(Mutex::new(router));
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let write_count = Arc::new(AtomicUsize::new(0));
    let success_count = Arc::new(AtomicUsize::new(0));

    let start = Instant::now();

    // Writer threads
    let mut handles = vec![];
    for t in 0..thread_count {
        let router = Arc::clone(&router);
        let done = Arc::clone(&done);
        let writes = Arc::clone(&write_count);
        let successes = Arc::clone(&success_count);

        handles.push(thread::spawn(move || {
            let mut latencies = stress_tests::LatencyHistogram::new();
            let mut i = 0;

            while !done.load(Ordering::Acquire) {
                // Neumann INSERT syntax
                let query = format!(
                    "INSERT stress_test id={}, thread={}, iter={}",
                    t * 1000000 + i,
                    t,
                    i
                );

                let op_start = Instant::now();
                let router = router.lock().unwrap();
                if router.execute(&query).is_ok() {
                    successes.fetch_add(1, Ordering::Relaxed);
                }
                drop(router);
                latencies.record(op_start.elapsed());

                writes.fetch_add(1, Ordering::Relaxed);
                i += 1;

                // Small yield to allow other threads
                if i % 100 == 0 {
                    thread::yield_now();
                }
            }

            latencies.snapshot()
        }));
    }

    // Run for specified duration
    thread::sleep(Duration::from_secs(test_duration_secs));
    done.store(true, Ordering::Release);

    // Wait for threads to finish
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    let elapsed = start.elapsed();
    let writes = write_count.load(Ordering::Relaxed);
    let successes = success_count.load(Ordering::Relaxed);

    println!("Duration: {:?}", elapsed);
    println!("Total writes: {}", writes);
    println!("Successful writes: {}", successes);
    println!(
        "Write throughput: {:.0} writes/sec",
        writes as f64 / elapsed.as_secs_f64()
    );
    println!(
        "Success rate: {:.2}%",
        successes as f64 / writes as f64 * 100.0
    );

    // Sample latency stats
    println!("Sample thread latencies:");
    for (i, snapshot) in results.iter().enumerate().take(4) {
        println!("  Thread {}: {}", i, snapshot);
    }

    // Should have high success rate
    assert!(successes > writes / 2, "Less than 50% success rate");

    println!(
        "PASSED: {} successful writes at {:.0} writes/sec",
        successes,
        writes as f64 / elapsed.as_secs_f64()
    );
}
