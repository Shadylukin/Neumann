// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for Chain concurrent append serialization.
//!
//! Verifies that the append_lock prevents TOCTOU race conditions
//! when multiple threads attempt to append blocks simultaneously.

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
    time::Duration,
};

use graph_engine::GraphEngine;
use tensor_chain::{Chain, Transaction};
use tensor_store::TensorStore;

fn create_test_chain() -> Arc<Chain> {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store));
    let chain = Arc::new(Chain::new(graph, "test_node".to_string()));
    chain.initialize().unwrap();
    chain
}

fn test_signature() -> Vec<u8> {
    vec![0u8; 64]
}

#[test]
fn test_concurrent_append_race_condition_prevented() {
    let chain = create_test_chain();
    let num_threads = 10;
    let barrier = Arc::new(Barrier::new(num_threads));
    let success_count = Arc::new(AtomicUsize::new(0));

    // Spawn threads that all try to append at the same time
    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let chain = Arc::clone(&chain);
            let barrier = Arc::clone(&barrier);
            let success_count = Arc::clone(&success_count);

            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier.wait();

                // All threads try to append at height 1
                let block = chain
                    .new_block()
                    .add_transaction(Transaction::Put {
                        key: format!("thread_{}", i),
                        data: vec![i as u8],
                    })
                    .with_signature(test_signature())
                    .build();

                if chain.append(block).is_ok() {
                    success_count.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Exactly one thread should have succeeded for height 1
    // (genesis is at height 0)
    let successes = success_count.load(Ordering::SeqCst);
    assert!(
        successes >= 1,
        "At least one append should succeed, got {}",
        successes
    );

    // Chain height should match number of successful appends
    assert_eq!(
        chain.height() as usize,
        successes,
        "Chain height should match successful appends"
    );

    // Verify chain integrity
    chain.verify_chain().unwrap();
}

#[test]
fn test_sequential_appends_after_concurrent_attempts() {
    let chain = create_test_chain();
    let num_threads = 5;
    let barrier = Arc::new(Barrier::new(num_threads));

    // First wave: concurrent attempts
    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let chain = Arc::clone(&chain);
            let barrier = Arc::clone(&barrier);

            thread::spawn(move || {
                barrier.wait();

                let block = chain
                    .new_block()
                    .add_transaction(Transaction::Put {
                        key: format!("wave1_thread_{}", i),
                        data: vec![i as u8],
                    })
                    .with_signature(test_signature())
                    .build();

                chain.append(block).is_ok()
            })
        })
        .collect();

    let wave1_successes: usize = handles
        .into_iter()
        .map(|h| if h.join().unwrap() { 1 } else { 0 })
        .sum();

    assert!(wave1_successes >= 1);

    // Second wave: sequential appends should all succeed
    let current_height = chain.height();
    for i in 0..5 {
        let block = chain
            .new_block()
            .add_transaction(Transaction::Put {
                key: format!("wave2_{}", i),
                data: vec![100 + i as u8],
            })
            .with_signature(test_signature())
            .build();

        chain.append(block).unwrap();
    }

    // All sequential appends should have succeeded
    assert_eq!(chain.height(), current_height + 5);

    // Verify chain integrity
    chain.verify_chain().unwrap();
}

#[test]
fn test_high_contention_append() {
    let chain = create_test_chain();
    let num_threads = 20;
    let iterations = 5;
    let barrier = Arc::new(Barrier::new(num_threads));
    let total_successes = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let chain = Arc::clone(&chain);
            let barrier = Arc::clone(&barrier);
            let total_successes = Arc::clone(&total_successes);

            thread::spawn(move || {
                barrier.wait();

                let mut thread_successes = 0;
                for iter in 0..iterations {
                    let block = chain
                        .new_block()
                        .add_transaction(Transaction::Put {
                            key: format!("t{}i{}", thread_id, iter),
                            data: vec![thread_id as u8, iter as u8],
                        })
                        .with_signature(test_signature())
                        .build();

                    if chain.append(block).is_ok() {
                        thread_successes += 1;
                    }

                    // Small delay to allow other threads to compete
                    thread::sleep(Duration::from_micros(10));
                }

                total_successes.fetch_add(thread_successes, Ordering::SeqCst);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let successes = total_successes.load(Ordering::SeqCst);
    let chain_height = chain.height();

    // All successful appends should be reflected in chain height
    assert_eq!(
        chain_height as usize, successes,
        "Chain height should match total successful appends"
    );

    // Chain should be valid
    chain.verify_chain().unwrap();

    // Verify all blocks have sequential heights
    for h in 0..=chain_height {
        let block = chain.get_block_at(h).unwrap().unwrap();
        assert_eq!(block.header.height, h);
    }
}

#[test]
fn test_append_lock_does_not_block_reads() {
    let chain = create_test_chain();

    // Pre-populate with some blocks
    for _ in 0..5 {
        let block = chain.new_block().with_signature(test_signature()).build();
        chain.append(block).unwrap();
    }

    let barrier = Arc::new(Barrier::new(3));
    let reads_completed = Arc::new(AtomicUsize::new(0));

    // Reader thread
    let chain_r = Arc::clone(&chain);
    let barrier_r = Arc::clone(&barrier);
    let reads_r = Arc::clone(&reads_completed);
    let reader = thread::spawn(move || {
        barrier_r.wait();

        // Perform many reads while writer is active
        for _ in 0..100 {
            let _ = chain_r.height();
            let _ = chain_r.tip_hash();
            let _ = chain_r.get_block_at(1);
            reads_r.fetch_add(1, Ordering::SeqCst);
        }
    });

    // Writer thread
    let chain_w = Arc::clone(&chain);
    let barrier_w = Arc::clone(&barrier);
    let writer = thread::spawn(move || {
        barrier_w.wait();

        // Perform appends
        for _ in 0..10 {
            let block = chain_w.new_block().with_signature(test_signature()).build();
            let _ = chain_w.append(block);
            thread::sleep(Duration::from_micros(100));
        }
    });

    // Start both threads
    barrier.wait();

    reader.join().unwrap();
    writer.join().unwrap();

    // Reader should have completed all reads
    let total_reads = reads_completed.load(Ordering::SeqCst);
    assert_eq!(total_reads, 100, "All reads should complete");

    // Chain should be valid
    chain.verify_chain().unwrap();
}
