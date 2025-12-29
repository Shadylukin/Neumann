//! Benchmarks for tensor_chain blockchain operations.
//!
//! Covers:
//! - Block creation and appending
//! - Transaction commit latency
//! - Batch transaction throughput
//! - Consensus validation (semantic conflict detection)
//! - Codebook operations (GlobalCodebook/LocalCodebook)
//! - Delta vector operations
//! - Chain queries (by height, history)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashSet;
use std::sync::Arc;
use tensor_chain::{
    Chain, CodebookManager, ConsensusConfig, ConsensusManager, DeltaVector, GlobalCodebook,
    LocalCodebook, TensorChain, Transaction,
};
use tensor_store::TensorStore;

// ============================================================================
// Block Creation Benchmarks
// ============================================================================

fn bench_block_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_creation");
    let store = TensorStore::new();
    let graph = Arc::new(graph_engine::GraphEngine::with_store(store));
    let chain = Chain::new(graph.clone(), "bench_node".to_string());
    chain.initialize().unwrap();

    group.bench_function("empty_block", |b| {
        b.iter(|| {
            let block = chain.new_block().build();
            black_box(block)
        })
    });

    group.bench_function("block_10_txns", |b| {
        let txns: Vec<Transaction> = (0..10)
            .map(|i| Transaction::Put {
                key: format!("key_{}", i),
                data: vec![i as u8; 64],
            })
            .collect();

        b.iter(|| {
            let mut builder = chain.new_block();
            for tx in &txns {
                builder = builder.add_transaction(tx.clone());
            }
            black_box(builder.build())
        })
    });

    group.bench_function("block_100_txns", |b| {
        let txns: Vec<Transaction> = (0..100)
            .map(|i| Transaction::Put {
                key: format!("key_{}", i),
                data: vec![i as u8; 64],
            })
            .collect();

        b.iter(|| {
            let mut builder = chain.new_block();
            for tx in &txns {
                builder = builder.add_transaction(tx.clone());
            }
            black_box(builder.build())
        })
    });

    group.finish();
}

// ============================================================================
// Transaction Commit Benchmarks
// ============================================================================

fn bench_transaction_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("transaction_commit");

    group.bench_function("single_put", |b| {
        b.iter_batched(
            || {
                let store = TensorStore::new();
                let chain = TensorChain::new(store, "bench_node");
                chain.initialize().unwrap();
                chain
            },
            |chain| {
                let tx = chain.begin().unwrap();
                tx.add_operation(Transaction::Put {
                    key: "test_key".to_string(),
                    data: vec![1, 2, 3, 4],
                })
                .unwrap();
                black_box(chain.commit(tx).unwrap())
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("multi_put_10", |b| {
        b.iter_batched(
            || {
                let store = TensorStore::new();
                let chain = TensorChain::new(store, "bench_node");
                chain.initialize().unwrap();
                chain
            },
            |chain| {
                let tx = chain.begin().unwrap();
                for i in 0..10 {
                    tx.add_operation(Transaction::Put {
                        key: format!("key_{}", i),
                        data: vec![i as u8; 64],
                    })
                    .unwrap();
                }
                black_box(chain.commit(tx).unwrap())
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Batch Transaction Benchmarks
// ============================================================================

fn bench_batch_transactions(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_transactions");

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched(
                || {
                    let store = TensorStore::new();
                    let chain = TensorChain::new(store, "bench_node");
                    chain.initialize().unwrap();
                    chain
                },
                |chain| {
                    for i in 0..size {
                        let tx = chain.begin().unwrap();
                        tx.add_operation(Transaction::Put {
                            key: format!("batch_key_{}", i),
                            data: vec![i as u8; 32],
                        })
                        .unwrap();
                        chain.commit(tx).unwrap();
                    }
                    black_box(chain.height())
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

// ============================================================================
// Consensus Validation Benchmarks
// ============================================================================

fn bench_consensus_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("consensus_validation");

    let config = ConsensusConfig {
        orthogonal_threshold: 0.1,
        conflict_threshold: 0.7,
        identical_threshold: 0.99,
        opposite_threshold: -0.9,
        allow_key_overlap_merge: false,
    };
    let consensus = ConsensusManager::new(config);

    // Generate random-ish delta vectors
    let deltas: Vec<DeltaVector> = (0..100)
        .map(|i| {
            let vector: Vec<f32> = (0..128)
                .map(|j| ((i * 17 + j) as f32 / 1000.0).sin())
                .collect();
            DeltaVector::new(vector, HashSet::from([format!("key_{}", i)]), i as u64)
        })
        .collect();

    group.bench_function("conflict_detection_pair", |b| {
        let d1 = &deltas[0];
        let d2 = &deltas[1];
        b.iter(|| black_box(consensus.detect_conflict(d1, d2)))
    });

    group.bench_function("cosine_similarity", |b| {
        let d1 = &deltas[0];
        let d2 = &deltas[1];
        b.iter(|| black_box(d1.cosine_similarity(d2)))
    });

    group.bench_function("merge_pair", |b| {
        let d1 = &deltas[0];
        let d2 = &deltas[50]; // Different enough to be orthogonal
        b.iter(|| black_box(consensus.merge(d1, d2)))
    });

    group.bench_function("merge_all_10", |b| {
        let subset: Vec<DeltaVector> = deltas.iter().take(10).cloned().collect();
        b.iter(|| black_box(consensus.merge_all(&subset)))
    });

    group.bench_function("find_merge_order_10", |b| {
        let subset: Vec<DeltaVector> = deltas.iter().take(10).cloned().collect();
        b.iter(|| black_box(consensus.find_merge_order(&subset)))
    });

    group.finish();
}

// ============================================================================
// Codebook Benchmarks
// ============================================================================

fn bench_codebook_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("codebook");

    // Create centroids via simple initialization
    let centroids: Vec<Vec<f32>> = (0..16)
        .map(|i| {
            (0..128)
                .map(|j| ((i * 17 + j) as f32 / 100.0).cos())
                .collect()
        })
        .collect();

    let global = GlobalCodebook::from_centroids(centroids);

    group.bench_function("global_quantize_128d", |b| {
        let query: Vec<f32> = (0..128).map(|i| (i as f32 / 100.0).sin()).collect();
        b.iter(|| black_box(global.quantize(&query)))
    });

    group.bench_function("global_compute_residual_128d", |b| {
        let query: Vec<f32> = (0..128).map(|i| (i as f32 / 100.0).sin()).collect();
        b.iter(|| black_box(global.compute_residual(&query)))
    });

    group.bench_function("global_is_valid_state", |b| {
        let state: Vec<f32> = (0..128).map(|i| (i as f32 / 100.0).sin()).collect();
        b.iter(|| black_box(global.is_valid_state(&state, 0.7)))
    });

    // LocalCodebook benchmarks
    let local = LocalCodebook::new("test_domain", 128, 16, 0.1);
    // Populate with some entries via quantize_and_update
    for i in 0..8 {
        let vector: Vec<f32> = (0..128)
            .map(|j| ((i * 23 + j) as f32 / 100.0).sin())
            .collect();
        local.quantize_and_update(&vector, 0.9);
    }

    group.bench_function("local_quantize_128d", |b| {
        let query: Vec<f32> = (0..128).map(|i| (i as f32 / 100.0).sin()).collect();
        b.iter(|| black_box(local.quantize(&query)))
    });

    group.bench_function("local_quantize_and_update", |b| {
        let query: Vec<f32> = (0..128).map(|i| (i as f32 / 100.0).sin()).collect();
        b.iter(|| black_box(local.quantize_and_update(&query, 0.9)))
    });

    // CodebookManager benchmarks
    let manager = CodebookManager::with_global(global.clone());

    group.bench_function("manager_quantize_128d", |b| {
        let query: Vec<f32> = (0..128).map(|i| (i as f32 / 100.0).sin()).collect();
        b.iter(|| black_box(manager.quantize("bench_domain", &query)))
    });

    group.finish();
}

// ============================================================================
// Delta Vector Benchmarks
// ============================================================================

fn bench_delta_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_vector");

    let v1: Vec<f32> = (0..128).map(|i| (i as f32 / 100.0).sin()).collect();
    let v2: Vec<f32> = (0..128).map(|i| (i as f32 / 100.0).cos()).collect();

    let d1 = DeltaVector::new(v1.clone(), HashSet::from(["key1".to_string()]), 1);
    let d2 = DeltaVector::new(v2.clone(), HashSet::from(["key2".to_string()]), 2);

    group.bench_function("cosine_similarity_128d", |b| {
        b.iter(|| black_box(d1.cosine_similarity(&d2)))
    });

    group.bench_function("add_128d", |b| b.iter(|| black_box(d1.add(&d2))));

    group.bench_function("scale_128d", |b| b.iter(|| black_box(d1.scale(0.5))));

    group.bench_function("weighted_average_128d", |b| {
        b.iter(|| black_box(d1.weighted_average(&d2, 0.6, 0.4)))
    });

    group.bench_function("overlaps_with", |b| {
        b.iter(|| black_box(d1.overlaps_with(&d2)))
    });

    // Larger dimension benchmark
    let v3: Vec<f32> = (0..768).map(|i| (i as f32 / 100.0).sin()).collect();
    let v4: Vec<f32> = (0..768).map(|i| (i as f32 / 100.0).cos()).collect();
    let d3 = DeltaVector::new(v3, HashSet::from(["key3".to_string()]), 3);
    let d4 = DeltaVector::new(v4, HashSet::from(["key4".to_string()]), 4);

    group.bench_function("cosine_similarity_768d", |b| {
        b.iter(|| black_box(d3.cosine_similarity(&d4)))
    });

    group.bench_function("add_768d", |b| b.iter(|| black_box(d3.add(&d4))));

    group.finish();
}

// ============================================================================
// Chain Query Benchmarks
// ============================================================================

fn bench_chain_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_query");

    // Setup: create a chain with some blocks
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "bench_node");
    chain.initialize().unwrap();

    // Add 100 blocks
    for i in 0..100 {
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: format!("key_{}", i % 10), // Reuse keys for history
            data: vec![i as u8; 32],
        })
        .unwrap();
        chain.commit(tx).unwrap();
    }

    group.bench_function("get_block_by_height", |b| {
        b.iter(|| black_box(chain.get_block(50).unwrap()))
    });

    group.bench_function("get_tip", |b| {
        b.iter(|| black_box(chain.get_tip().unwrap()))
    });

    group.bench_function("get_genesis", |b| {
        b.iter(|| black_box(chain.get_genesis().unwrap()))
    });

    group.bench_function("height", |b| b.iter(|| black_box(chain.height())));

    group.bench_function("tip_hash", |b| b.iter(|| black_box(chain.tip_hash())));

    group.bench_function("history_key", |b| {
        b.iter(|| black_box(chain.history("key_0").unwrap()))
    });

    group.bench_function("verify_chain_100_blocks", |b| {
        b.iter(|| black_box(chain.verify().unwrap()))
    });

    group.finish();
}

// ============================================================================
// Chain Iteration Benchmarks
// ============================================================================

fn bench_chain_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_iteration");

    // Setup chain with blocks
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "bench_node");
    chain.initialize().unwrap();

    for i in 0..50 {
        let tx = chain.begin().unwrap();
        tx.add_operation(Transaction::Put {
            key: format!("key_{}", i),
            data: vec![i as u8; 64],
        })
        .unwrap();
        chain.commit(tx).unwrap();
    }

    group.bench_function("iterate_50_blocks", |b| {
        b.iter(|| {
            let count = chain.iter().count();
            black_box(count)
        })
    });

    group.bench_function("get_blocks_range_0_25", |b| {
        b.iter(|| black_box(chain.get_blocks(0, 25).unwrap()))
    });

    group.finish();
}

// ============================================================================
// GlobalCodebook k-means Benchmarks
// ============================================================================

fn bench_codebook_kmeans(c: &mut Criterion) {
    let mut group = c.benchmark_group("codebook_kmeans");
    group.sample_size(20); // Fewer samples for expensive operations

    // Generate sample vectors
    let vectors_100: Vec<Vec<f32>> = (0..100)
        .map(|i| {
            (0..64)
                .map(|j| ((i * 17 + j) as f32 / 100.0).sin())
                .collect()
        })
        .collect();

    let vectors_1000: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            (0..64)
                .map(|j| ((i * 17 + j) as f32 / 100.0).sin())
                .collect()
        })
        .collect();

    group.bench_function("kmeans_100_vectors_8_clusters", |b| {
        b.iter(|| black_box(GlobalCodebook::from_kmeans(&vectors_100, 8, 10)))
    });

    group.bench_function("kmeans_1000_vectors_16_clusters", |b| {
        b.iter(|| black_box(GlobalCodebook::from_kmeans(&vectors_1000, 16, 10)))
    });

    group.finish();
}

// ============================================================================
// Main
// ============================================================================

criterion_group!(
    benches,
    bench_block_creation,
    bench_transaction_commit,
    bench_batch_transactions,
    bench_consensus_validation,
    bench_codebook_operations,
    bench_delta_vector,
    bench_chain_query,
    bench_chain_iteration,
    bench_codebook_kmeans,
);
criterion_main!(benches);
