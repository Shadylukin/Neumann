//! TieredStore stress tests for hot/cold migration.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Instant;
use stress_tests::{full_config, generate_embeddings, LatencyHistogram};
use tempfile::tempdir;
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue, TieredConfig, TieredStore};

fn create_tensor(id: i64, embedding: Vec<f32>) -> TensorData {
    let mut data = TensorData::new();
    data.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
    data.set("embedding", TensorValue::Vector(embedding));
    data
}

/// Stress test: Hot/cold migration under concurrent load.
#[test]
#[ignore]
fn stress_tiered_migration_under_load() {
    let config = full_config();
    let entity_count = config.effective_entity_count() / 10; // 100K for this test
    let thread_count = 8;

    println!("\n=== TieredStore Migration Under Load ===");
    println!("Entities: {}", entity_count);
    println!("Threads: {}", thread_count);

    let temp_dir = tempdir().expect("create temp dir");

    // Create tiered store with cold storage
    let tiered_config = TieredConfig {
        cold_dir: temp_dir.path().to_path_buf(),
        cold_capacity: 1024 * 1024 * 512, // 512MB
        sample_rate: 10,
        ..Default::default()
    };
    let tiered = Arc::new(Mutex::new(
        TieredStore::new(tiered_config).expect("create tiered store"),
    ));

    // Pre-populate with data
    let embeddings = Arc::new(generate_embeddings(entity_count, 128, 42));
    println!("Populating {} entities...", entity_count);

    {
        let mut store = tiered.lock().unwrap();
        for (i, emb) in embeddings.iter().enumerate() {
            store.put(format!("key:{}", i), create_tensor(i as i64, emb.clone()));
        }
        println!("Population complete. Hot size: {}", store.hot_len());
    }

    // Concurrent access pattern
    let access_count = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();

    let mut handles = vec![];

    // Reader threads
    for t in 0..thread_count / 2 {
        let tiered = Arc::clone(&tiered);
        let access_count = Arc::clone(&access_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();

            for i in 0..10_000 {
                let key_idx = (t * 100 + i) % 1000;
                let op_start = Instant::now();
                {
                    let mut store = tiered.lock().unwrap();
                    let _ = store.get(&format!("key:{}", key_idx));
                }
                latencies.record(op_start.elapsed());
                access_count.fetch_add(1, Ordering::Relaxed);
            }

            ("reader", latencies.snapshot())
        }));
    }

    // Writer threads
    let writes_per_thread = 10_000;
    for t in 0..thread_count / 2 {
        let tiered = Arc::clone(&tiered);
        let embeddings = Arc::clone(&embeddings);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let start_idx = entity_count + t * writes_per_thread;

            for i in 0..writes_per_thread {
                let idx = start_idx + i;
                let emb_idx = i % embeddings.len();
                let op_start = Instant::now();
                {
                    let mut store = tiered.lock().unwrap();
                    store.put(
                        format!("key:{}", idx),
                        create_tensor(idx as i64, embeddings[emb_idx].clone()),
                    );
                }
                latencies.record(op_start.elapsed());
            }

            ("writer", latencies.snapshot())
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    println!("Duration: {:?}", elapsed);
    println!("Hot accesses: {}", access_count.load(Ordering::Relaxed));

    let store = tiered.lock().unwrap();
    println!("Final hot size: {}", store.hot_len());
    drop(store);

    for (i, (role, snapshot)) in results.iter().enumerate() {
        println!("  {} {}: {}", role, i, snapshot);
    }

    // Verify data integrity
    let mut store = tiered.lock().unwrap();
    for i in 0..100 {
        let data = store.get(&format!("key:{}", i)).expect("key should exist");
        if let Some(TensorValue::Scalar(ScalarValue::Int(id))) = data.get("id") {
            assert_eq!(*id, i as i64);
        }
    }

    println!("PASSED: Migration under load completed successfully");
}

/// Stress test: Hot-only mode at scale.
#[test]
#[ignore]
fn stress_tiered_hot_only_scale() {
    let config = full_config();
    let entity_count = config.effective_entity_count();
    let thread_count = config.effective_thread_count();

    println!("\n=== TieredStore Hot-Only 1M Entries ===");
    println!("Entities: {}", entity_count);
    println!("Threads: {}", thread_count);

    // Create hot-only tiered store
    let tiered = Arc::new(Mutex::new(TieredStore::hot_only(10)));

    // Concurrent writes
    let per_thread = entity_count / thread_count;
    let mut handles = vec![];
    let start = Instant::now();

    for t in 0..thread_count {
        let tiered = Arc::clone(&tiered);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let start_idx = t * per_thread;

            // Generate embeddings for this thread
            let embeddings = generate_embeddings(per_thread, 128, 42 + t as u64);

            for (i, emb) in embeddings.into_iter().enumerate() {
                let idx = start_idx + i;
                let op_start = Instant::now();
                {
                    let mut store = tiered.lock().unwrap();
                    store.put(format!("key:{}", idx), create_tensor(idx as i64, emb));
                }
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

    for (i, snapshot) in results.iter().enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    // Verify total count
    let mut store = tiered.lock().unwrap();
    let total = store.hot_len();
    println!("Hot: {}, Total: {}", store.hot_len(), total);

    // Sample verification
    for i in [0, entity_count / 2, entity_count - 1] {
        let data = store.get(&format!("key:{}", i)).expect("key should exist");
        if let Some(TensorValue::Scalar(ScalarValue::Int(id))) = data.get("id") {
            assert_eq!(*id, i as i64);
        }
    }

    println!("PASSED: {} entries in hot-only mode", entity_count);
}

/// Stress test: Read latency from hot tier.
#[test]
#[ignore]
fn stress_tiered_hot_read_latency() {
    let config = full_config();
    let entity_count = 100_000;
    let thread_count = config.effective_thread_count();

    println!("\n=== TieredStore Hot Read Latency ===");
    println!("Entities: {}", entity_count);
    println!("Threads: {}", thread_count);

    // Create hot-only store and populate
    let mut tiered = TieredStore::hot_only(10);
    let embeddings = generate_embeddings(entity_count, 128, 42);
    for (i, emb) in embeddings.iter().enumerate() {
        tiered.put(format!("key:{}", i), create_tensor(i as i64, emb.clone()));
    }

    println!("Hot size: {}", tiered.hot_len());

    // Wrap for concurrent reads
    let tiered = Arc::new(Mutex::new(tiered));

    // Random read benchmark
    let reads_per_thread = 10_000;
    let mut handles = vec![];
    let start = Instant::now();

    for t in 0..thread_count {
        let tiered = Arc::clone(&tiered);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();

            for i in 0..reads_per_thread {
                // Random access pattern
                let key_idx = (t * 7919 + i * 6271) % entity_count;
                let op_start = Instant::now();
                {
                    let mut store = tiered.lock().unwrap();
                    let _ = store.get(&format!("key:{}", key_idx));
                }
                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_reads = thread_count * reads_per_thread;
    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} reads/sec",
        total_reads as f64 / elapsed.as_secs_f64()
    );

    for (i, snapshot) in results.iter().enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    println!("PASSED: Hot read latency benchmark completed");
}
