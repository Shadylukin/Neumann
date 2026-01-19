//! TensorChain stress tests at 10k+ operation scale.
//!
//! Tests:
//! - Chain append with 10k blocks across multiple threads
//! - Concurrent transactions with 10k operations
//! - HNSW search during concurrent inserts

use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use stress_tests::{LatencyHistogram, StressConfig};
use tensor_chain::{
    block::Transaction,
    consensus::{ConsensusConfig, ConsensusManager},
    BlockHeader, DistributedTxConfig, DistributedTxCoordinator, LockManager, PrepareRequest,
};
use tensor_store::{HNSWIndex, SparseVector, TensorStore};

// ============= Configuration =============

fn stress_config() -> StressConfig {
    // Default to medium scale for CI, can override with environment
    StressConfig {
        scale: stress_tests::ScaleLevel::Quick,
        entity_count: std::env::var("STRESS_ENTITY_COUNT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10_000),
        thread_count: std::env::var("STRESS_THREAD_COUNT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10),
        duration_secs: 300,
        embedding_dim: 128,
        report_interval_secs: 30,
    }
}

// ============= Stress Test 1: Chain Append 10k Blocks =============

#[test]
#[ignore]
fn stress_chain_append_10k_blocks() {
    let config = stress_config();
    let block_count = config.entity_count;
    let thread_count = config.thread_count;

    println!("\n=== TensorChain 10k Block Append Stress Test ===");
    println!("Total blocks: {}", block_count);
    println!("Threads: {}", thread_count);

    let store = Arc::new(TensorStore::new());
    let counter = Arc::new(AtomicUsize::new(0));
    let per_thread = block_count / thread_count;

    let mut handles = vec![];
    let start = Instant::now();

    for t in 0..thread_count {
        let store = Arc::clone(&store);
        let counter = Arc::clone(&counter);

        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut local_count = 0;

            for i in 0..per_thread {
                let op_start = Instant::now();

                // Create a unique block key
                let block_height = t * per_thread + i;
                let key = format!("block:{}", block_height);

                // Create block header
                let header = BlockHeader::new(
                    block_height as u64,
                    [0u8; 32],
                    [0u8; 32],
                    [0u8; 32],
                    format!("proposer_{}", t),
                );

                // Store block header data
                let mut data = tensor_store::TensorData::new();
                data.set(
                    "height",
                    tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(
                        block_height as i64,
                    )),
                );
                data.set(
                    "proposer",
                    tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(format!(
                        "proposer_{}",
                        t
                    ))),
                );
                data.set(
                    "timestamp",
                    tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(
                        header.timestamp as i64,
                    )),
                );

                store.put(key, data).unwrap();

                local_count += 1;
                counter.fetch_add(1, Ordering::Relaxed);
                latencies.record(op_start.elapsed());
            }

            (latencies.snapshot(), local_count)
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    // Aggregate results
    let total_blocks: usize = results.iter().map(|(_, count)| count).sum();
    let throughput = total_blocks as f64 / elapsed.as_secs_f64();

    println!("\n=== Results ===");
    println!("Duration: {:?}", elapsed);
    println!("Total blocks written: {}", total_blocks);
    println!("Throughput: {:.0} blocks/sec", throughput);

    for (i, (snapshot, count)) in results.iter().enumerate() {
        println!(
            "  Thread {}: {} blocks, p50={:?}, p99={:?}",
            i, count, snapshot.p50, snapshot.p99
        );
    }

    // Verify
    assert_eq!(
        store.len(),
        block_count,
        "Expected {} blocks, got {}",
        block_count,
        store.len()
    );
    assert!(
        throughput >= 1000.0,
        "Throughput {} too low, expected >= 1000 ops/sec",
        throughput
    );

    // Verify p99 latency
    for (snapshot, _) in &results {
        assert!(
            snapshot.p99.as_millis() < 50,
            "p99 latency too high: {:?}",
            snapshot.p99
        );
    }

    println!(
        "PASSED: {} blocks appended at {:.0} blocks/sec",
        block_count, throughput
    );
}

// ============= Stress Test 2: Concurrent Transactions 10k =============

#[test]
#[ignore]
fn stress_concurrent_transactions_10k() {
    let config = stress_config();
    let tx_count = config.entity_count;
    let thread_count = config.thread_count;

    println!("\n=== TensorChain 10k Concurrent Transactions Stress Test ===");
    println!("Total transactions: {}", tx_count);
    println!("Threads: {}", thread_count);

    // Shared coordinator
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let coordinator = Arc::new(DistributedTxCoordinator::new(
        consensus,
        DistributedTxConfig::default(),
    ));

    let per_thread = tx_count / thread_count;
    let mut handles = vec![];
    let start = Instant::now();
    let committed = Arc::new(AtomicUsize::new(0));
    let aborted = Arc::new(AtomicUsize::new(0));

    for t in 0..thread_count {
        let coordinator = Arc::clone(&coordinator);
        let committed = Arc::clone(&committed);
        let aborted = Arc::clone(&aborted);

        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut local_committed = 0;
            let mut local_aborted = 0;

            for i in 0..per_thread {
                let op_start = Instant::now();

                // Each transaction uses unique keys to avoid conflicts
                let key = format!("thread{}:key{}", t, i);

                // Begin transaction
                let tx = match coordinator.begin(format!("coord_{}", t), vec![0]) {
                    Ok(tx) => tx,
                    Err(_) => {
                        local_aborted += 1;
                        continue;
                    },
                };

                // Prepare
                let request = PrepareRequest {
                    tx_id: tx.tx_id,
                    coordinator: format!("coord_{}", t),
                    operations: vec![Transaction::Put {
                        key: key.clone(),
                        data: vec![t as u8, (i % 256) as u8],
                    }],
                    delta_embedding: SparseVector::from_dense(&[
                        (t as f32) / 10.0,
                        (i as f32) / 1000.0,
                    ]),
                    timeout_ms: 5000,
                };

                let vote = coordinator.handle_prepare(request);
                coordinator.record_vote(tx.tx_id, 0, vote);

                // Commit or abort
                match coordinator.commit(tx.tx_id) {
                    Ok(_) => {
                        local_committed += 1;
                        committed.fetch_add(1, Ordering::Relaxed);
                    },
                    Err(_) => {
                        local_aborted += 1;
                        aborted.fetch_add(1, Ordering::Relaxed);
                    },
                }

                latencies.record(op_start.elapsed());
            }

            (latencies.snapshot(), local_committed, local_aborted)
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_committed = committed.load(Ordering::Relaxed);
    let total_aborted = aborted.load(Ordering::Relaxed);
    let throughput = (total_committed + total_aborted) as f64 / elapsed.as_secs_f64();

    println!("\n=== Results ===");
    println!("Duration: {:?}", elapsed);
    println!("Transactions committed: {}", total_committed);
    println!("Transactions aborted: {}", total_aborted);
    println!("Throughput: {:.0} tx/sec", throughput);

    for (i, (snapshot, committed, aborted)) in results.iter().enumerate() {
        println!(
            "  Thread {}: {} committed, {} aborted, p50={:?}, p99={:?}",
            i, committed, aborted, snapshot.p50, snapshot.p99
        );
    }

    // Verify
    assert!(
        total_committed > tx_count * 90 / 100,
        "Too many aborts: {} committed out of {}",
        total_committed,
        tx_count
    );
    assert!(
        throughput >= 1000.0,
        "Throughput {} too low, expected >= 1000 tx/sec",
        throughput
    );
    assert_eq!(
        coordinator.pending_count(),
        0,
        "All transactions should be resolved"
    );

    println!(
        "PASSED: {} transactions processed at {:.0} tx/sec",
        total_committed + total_aborted,
        throughput
    );
}

// ============= Stress Test 3: HNSW Search During Insert =============

#[test]
#[ignore]
fn stress_hnsw_search_during_insert_10k() {
    let config = stress_config();
    let entity_count = config.entity_count;
    let search_count = 1000;
    let dim = 64;

    println!("\n=== TensorChain HNSW Search During Insert Stress Test ===");
    println!("Total entities: {}", entity_count);
    println!("Concurrent searches: {}", search_count);
    println!("Dimensions: {}", dim);

    let index = Arc::new(HNSWIndex::new());
    let stop_flag = Arc::new(AtomicBool::new(false));
    let insert_count = Arc::new(AtomicUsize::new(0));
    let search_count_done = Arc::new(AtomicUsize::new(0));

    // Insert thread
    let index_clone = Arc::clone(&index);
    let stop_flag_clone = Arc::clone(&stop_flag);
    let insert_count_clone = Arc::clone(&insert_count);

    let insert_handle = thread::spawn(move || {
        let mut latencies = LatencyHistogram::new();
        let mut rng_state = 12345u64;

        for i in 0..entity_count {
            if stop_flag_clone.load(Ordering::Relaxed) {
                break;
            }

            let op_start = Instant::now();

            // Generate pseudo-random vector
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = ((rng_state >> 33) as i32) as f32 / (i32::MAX as f32);
                vec.push(val);
            }

            index_clone.insert(vec);
            insert_count_clone.fetch_add(1, Ordering::Relaxed);

            latencies.record(op_start.elapsed());

            // Brief yield every 1000 inserts
            if i % 1000 == 0 {
                thread::yield_now();
            }
        }

        latencies.snapshot()
    });

    // Search threads
    let search_threads = 4;
    let searches_per_thread = search_count / search_threads;
    let mut search_handles = vec![];

    for t in 0..search_threads {
        let index_clone = Arc::clone(&index);
        let search_count_done = Arc::clone(&search_count_done);
        let insert_count = Arc::clone(&insert_count);

        search_handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut successful = 0;
            let mut rng_state = (t as u64 + 1) * 54321;

            // Wait until some inserts are done
            while insert_count.load(Ordering::Relaxed) < 100 {
                thread::sleep(Duration::from_millis(1));
            }

            for _ in 0..searches_per_thread {
                let op_start = Instant::now();

                // Generate query vector
                let mut vec = Vec::with_capacity(dim);
                for _ in 0..dim {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let val = ((rng_state >> 33) as i32) as f32 / (i32::MAX as f32);
                    vec.push(val);
                }

                let results = index_clone.search(&vec, 10);

                if !results.is_empty() {
                    successful += 1;
                }

                search_count_done.fetch_add(1, Ordering::Relaxed);
                latencies.record(op_start.elapsed());
            }

            (latencies.snapshot(), successful)
        }));
    }

    // Wait for inserts to complete
    let insert_snapshot = insert_handle.join().unwrap();

    // Wait for searches
    let search_results: Vec<_> = search_handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();

    let total_inserts = insert_count.load(Ordering::Relaxed);
    let total_searches = search_count_done.load(Ordering::Relaxed);
    let successful_searches: usize = search_results.iter().map(|(_, s)| s).sum();

    println!("\n=== Results ===");
    println!("Total inserts: {}", total_inserts);
    println!("Total searches: {}", total_searches);
    println!(
        "Successful searches: {} ({:.1}%)",
        successful_searches,
        (successful_searches as f64 / total_searches as f64) * 100.0
    );
    println!(
        "Insert p50={:?}, p99={:?}",
        insert_snapshot.p50, insert_snapshot.p99
    );

    for (i, (snapshot, success)) in search_results.iter().enumerate() {
        println!(
            "  Search thread {}: {} successful, p50={:?}, p99={:?}",
            i, success, snapshot.p50, snapshot.p99
        );
    }

    // Verify
    assert_eq!(total_inserts, entity_count, "All inserts should complete");
    assert!(
        successful_searches > search_count * 80 / 100,
        "Most searches should succeed: {} out of {}",
        successful_searches,
        search_count
    );

    // Verify p99 latency for searches
    for (snapshot, _) in &search_results {
        assert!(
            snapshot.p99.as_millis() < 100,
            "Search p99 latency too high: {:?}",
            snapshot.p99
        );
    }

    println!(
        "PASSED: {} inserts and {} searches completed successfully",
        total_inserts, successful_searches
    );
}

// ============= Stress Test 4: Lock Manager Contention =============

#[test]
#[ignore]
fn stress_lock_manager_high_contention() {
    let config = stress_config();
    let op_count = config.entity_count;
    let thread_count = config.thread_count;
    let key_count = 100; // Small key space for high contention

    println!("\n=== TensorChain Lock Manager High Contention Stress Test ===");
    println!("Total operations: {}", op_count);
    println!("Threads: {}", thread_count);
    println!("Unique keys: {}", key_count);

    let lock_manager = Arc::new(LockManager::new());
    let per_thread = op_count / thread_count;
    let mut handles = vec![];
    let start = Instant::now();
    let acquired = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicUsize::new(0));

    for t in 0..thread_count {
        let lock_manager = Arc::clone(&lock_manager);
        let acquired = Arc::clone(&acquired);
        let failed = Arc::clone(&failed);

        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let mut local_acquired = 0;
            let mut local_failed = 0;
            let mut rng_state = (t as u64 + 1) * 12345;

            for i in 0..per_thread {
                let op_start = Instant::now();

                // Pick a random key (hash-based)
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let key_idx = (rng_state >> 33) as usize % key_count;
                let key = format!("contention_key_{}", key_idx);

                let tx_id = (t * per_thread + i) as u64;

                // Try to acquire lock
                let keys = vec![key.clone()];
                match lock_manager.try_lock(tx_id, &keys) {
                    Ok(handle) => {
                        local_acquired += 1;
                        acquired.fetch_add(1, Ordering::Relaxed);

                        // Simulate brief work
                        thread::yield_now();

                        // Release lock
                        lock_manager.release_by_handle(handle);
                    },
                    Err(_) => {
                        local_failed += 1;
                        failed.fetch_add(1, Ordering::Relaxed);
                    },
                }

                latencies.record(op_start.elapsed());
            }

            (latencies.snapshot(), local_acquired, local_failed)
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    let total_acquired = acquired.load(Ordering::Relaxed);
    let total_failed = failed.load(Ordering::Relaxed);
    let throughput = op_count as f64 / elapsed.as_secs_f64();

    println!("\n=== Results ===");
    println!("Duration: {:?}", elapsed);
    println!("Locks acquired: {}", total_acquired);
    println!("Locks failed: {}", total_failed);
    println!(
        "Contention rate: {:.1}%",
        (total_failed as f64 / op_count as f64) * 100.0
    );
    println!("Throughput: {:.0} ops/sec", throughput);

    for (i, (snapshot, acquired, failed)) in results.iter().enumerate() {
        println!(
            "  Thread {}: {} acquired, {} failed, p50={:?}, p99={:?}",
            i, acquired, failed, snapshot.p50, snapshot.p99
        );
    }

    // Verify all locks were released (no deadlock)
    assert!(
        lock_manager.active_lock_count() == 0,
        "All locks should be released, got {}",
        lock_manager.active_lock_count()
    );

    // Verify reasonable acquisition rate under contention
    let acquisition_rate = total_acquired as f64 / op_count as f64;
    assert!(
        acquisition_rate > 0.3,
        "Lock acquisition rate too low: {:.1}%",
        acquisition_rate * 100.0
    );

    println!(
        "PASSED: {} operations with {:.1}% acquisition rate, no deadlocks",
        op_count,
        acquisition_rate * 100.0
    );
}
