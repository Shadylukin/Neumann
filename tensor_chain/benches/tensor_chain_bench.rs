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
//! - Raft consensus operations
//! - 2PC distributed transactions
//! - Gossip protocol operations
//! - Snapshot operations
//! - Membership management
//! - Deadlock detection

use std::{collections::HashSet, sync::Arc};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
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
        structural_conflict_threshold: 0.5,
        sparsity_threshold: 0.01,
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
        b.iter(|| {
            chain.verify().unwrap();
            black_box(())
        })
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
// Conflict Detection Loop Benchmarks (O(n²) scaling)
// ============================================================================

fn bench_conflict_detection_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("conflict_detection_scaling");
    group.sample_size(20); // Fewer samples for expensive O(n²) operations

    let config = ConsensusConfig {
        orthogonal_threshold: 0.1,
        conflict_threshold: 0.7,
        identical_threshold: 0.99,
        opposite_threshold: -0.9,
        allow_key_overlap_merge: false,
        structural_conflict_threshold: 0.5,
        sparsity_threshold: 0.01,
    };
    let consensus = ConsensusManager::new(config);

    // Test at different scales to see O(n²) behavior
    for n in [10, 25, 50, 100, 200].iter() {
        let deltas: Vec<DeltaVector> = (0..*n)
            .map(|i| {
                let vector: Vec<f32> = (0..128)
                    .map(|j| ((i * 17 + j) as f32 / 1000.0).sin())
                    .collect();
                DeltaVector::new(vector, HashSet::from([format!("key_{}", i)]), i as u64)
            })
            .collect();

        // n*(n-1)/2 pairs
        let pair_count = (*n * (*n - 1) / 2) as u64;
        group.throughput(Throughput::Elements(pair_count));

        group.bench_with_input(
            BenchmarkId::new("batch_detect_conflicts", n),
            &deltas,
            |b, deltas| b.iter(|| black_box(consensus.batch_detect_conflicts(deltas))),
        );

        group.bench_with_input(
            BenchmarkId::new("find_orthogonal_set", n),
            &deltas,
            |b, deltas| b.iter(|| black_box(consensus.find_orthogonal_set(deltas))),
        );

        group.bench_with_input(
            BenchmarkId::new("find_merge_order", n),
            &deltas,
            |b, deltas| b.iter(|| black_box(consensus.find_merge_order(deltas))),
        );
    }

    group.finish();
}

// ============================================================================
// Sparse vs Dense Delta Comparison
// ============================================================================

fn bench_sparsity_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity_comparison");

    let config = ConsensusConfig::default();
    let consensus = ConsensusManager::new(config);

    let dimension = 128;

    // Create deltas with varying sparsity levels
    // Sparsity = percentage of zeros

    // 10% sparse (90% non-zero) - almost dense
    let dense_deltas: Vec<DeltaVector> = (0..50)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| {
                    if (i + j) % 10 == 0 {
                        0.0
                    } else {
                        ((i * 17 + j) as f32 / 1000.0).sin()
                    }
                })
                .collect();
            DeltaVector::new(vector, HashSet::from([format!("key_{}", i)]), i as u64)
        })
        .collect();

    // 50% sparse
    let medium_deltas: Vec<DeltaVector> = (0..50)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| {
                    if (i + j) % 2 == 0 {
                        0.0
                    } else {
                        ((i * 17 + j) as f32 / 1000.0).sin()
                    }
                })
                .collect();
            DeltaVector::new(vector, HashSet::from([format!("key_{}", i)]), i as u64)
        })
        .collect();

    // 90% sparse (10% non-zero) - very sparse
    let sparse_deltas: Vec<DeltaVector> = (0..50)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| {
                    if (i + j) % 10 == 0 {
                        ((i * 17 + j) as f32 / 1000.0).sin()
                    } else {
                        0.0
                    }
                })
                .collect();
            DeltaVector::new(vector, HashSet::from([format!("key_{}", i)]), i as u64)
        })
        .collect();

    // 99% sparse (1% non-zero) - extremely sparse (realistic for deltas)
    let very_sparse_deltas: Vec<DeltaVector> = (0..50)
        .map(|i| {
            let vector: Vec<f32> = (0..dimension)
                .map(|j| {
                    if j == (i % dimension) || j == ((i * 7) % dimension) {
                        ((i * 17 + j) as f32 / 1000.0).sin()
                    } else {
                        0.0
                    }
                })
                .collect();
            DeltaVector::new(vector, HashSet::from([format!("key_{}", i)]), i as u64)
        })
        .collect();

    // Benchmark batch_detect_conflicts at each sparsity level
    let pair_count = (50 * 49 / 2) as u64;
    group.throughput(Throughput::Elements(pair_count));

    group.bench_function("batch_detect_10pct_sparse", |b| {
        b.iter(|| black_box(consensus.batch_detect_conflicts(&dense_deltas)))
    });

    group.bench_function("batch_detect_50pct_sparse", |b| {
        b.iter(|| black_box(consensus.batch_detect_conflicts(&medium_deltas)))
    });

    group.bench_function("batch_detect_90pct_sparse", |b| {
        b.iter(|| black_box(consensus.batch_detect_conflicts(&sparse_deltas)))
    });

    group.bench_function("batch_detect_99pct_sparse", |b| {
        b.iter(|| black_box(consensus.batch_detect_conflicts(&very_sparse_deltas)))
    });

    // Also measure the actual sparsity achieved
    println!("\n--- Sparsity Analysis ---");
    println!("10% target: actual nnz = {}", dense_deltas[0].nnz());
    println!("50% target: actual nnz = {}", medium_deltas[0].nnz());
    println!("90% target: actual nnz = {}", sparse_deltas[0].nnz());
    println!("99% target: actual nnz = {}", very_sparse_deltas[0].nnz());

    group.finish();
}

// ============================================================================
// Individual Operation Breakdown
// ============================================================================

fn bench_conflict_operation_breakdown(c: &mut Criterion) {
    let mut group = c.benchmark_group("conflict_ops_breakdown");

    // Create realistic delta vectors (90% sparse, 128 dimensions)
    let make_sparse_delta = |i: usize| {
        let vector: Vec<f32> = (0..128)
            .map(|j| {
                if (i + j) % 10 == 0 {
                    ((i * 17 + j) as f32 / 1000.0).sin()
                } else {
                    0.0
                }
            })
            .collect();
        DeltaVector::new(
            vector,
            HashSet::from([format!("key_{}", i), format!("key_{}", i + 1)]),
            i as u64,
        )
    };

    let d1 = make_sparse_delta(0);
    let d2 = make_sparse_delta(50);
    let d3 = make_sparse_delta(1); // Adjacent, might have key overlap

    // Measure individual operations
    group.bench_function("cosine_similarity_sparse", |b| {
        b.iter(|| black_box(d1.cosine_similarity(&d2)))
    });

    group.bench_function("angular_distance_sparse", |b| {
        b.iter(|| black_box(d1.angular_distance(&d2)))
    });

    group.bench_function("jaccard_index_sparse", |b| {
        b.iter(|| black_box(d1.jaccard_index(&d2)))
    });

    group.bench_function("euclidean_distance_sparse", |b| {
        b.iter(|| black_box(d1.euclidean_distance(&d2)))
    });

    group.bench_function("overlapping_keys", |b| {
        b.iter(|| black_box(d1.overlapping_keys(&d3)))
    });

    group.bench_function("overlaps_with", |b| {
        b.iter(|| black_box(d1.overlaps_with(&d3)))
    });

    group.bench_function("add_sparse", |b| b.iter(|| black_box(d1.add(&d2))));

    group.bench_function("weighted_average_sparse", |b| {
        b.iter(|| black_box(d1.weighted_average(&d2, 0.6, 0.4)))
    });

    group.bench_function("project_orthogonal_sparse", |b| {
        b.iter(|| black_box(d1.project_non_conflicting(&d2.delta)))
    });

    // The full detect_conflict call (combines cosine + key overlap + classification)
    let config = ConsensusConfig::default();
    let consensus = ConsensusManager::new(config);

    group.bench_function("detect_conflict_full", |b| {
        b.iter(|| black_box(consensus.detect_conflict(&d1, &d2)))
    });

    group.finish();
}

// ============================================================================
// High-Dimension Sparse Performance
// ============================================================================

fn bench_high_dimension_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_dimension_sparse");

    let config = ConsensusConfig::default();
    let consensus = ConsensusManager::new(config);

    // Test 768-dimensional embeddings (typical LLM embedding size)
    // At 95% sparsity (realistic for delta embeddings)
    for dim in [128, 256, 512, 768].iter() {
        let deltas: Vec<DeltaVector> = (0..20)
            .map(|i| {
                let vector: Vec<f32> = (0..*dim)
                    .map(|j| {
                        // 5% non-zero
                        if (i * 7 + j) % 20 == 0 {
                            ((i * 17 + j) as f32 / 1000.0).sin()
                        } else {
                            0.0
                        }
                    })
                    .collect();
                DeltaVector::new(vector, HashSet::from([format!("key_{}", i)]), i as u64)
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_detect_20_deltas", dim),
            &deltas,
            |b, deltas| b.iter(|| black_box(consensus.batch_detect_conflicts(deltas))),
        );

        // Single pair similarity
        group.bench_with_input(
            BenchmarkId::new("cosine_similarity", dim),
            &deltas,
            |b, deltas| b.iter(|| black_box(deltas[0].cosine_similarity(&deltas[1]))),
        );
    }

    group.finish();
}

// ============================================================================
// Real Transaction Delta Sparsity Measurement
// ============================================================================

fn measure_real_delta_sparsity(c: &mut Criterion) {
    use tensor_store::SparseVector;

    let mut group = c.benchmark_group("real_delta_sparsity");

    // Simulate different transaction patterns and measure resulting sparsity
    let dimension = 128;

    println!("\n========================================");
    println!("REAL TRANSACTION DELTA SPARSITY ANALYSIS");
    println!("========================================\n");

    // Pattern 1: Single key update (most common - updating one field)
    // Simulates: UPDATE users SET name = 'Bob' WHERE id = 1
    // Only a few dimensions change when one field is modified
    println!("--- Pattern 1: Single Key Update ---");
    let single_key_deltas: Vec<SparseVector> = (0..100)
        .map(|i| {
            let mut before = vec![0.0f32; dimension];
            let mut after = vec![0.0f32; dimension];

            // Base state: some random existing values
            for j in 0..dimension {
                before[j] = ((i * 17 + j) as f32 / 1000.0).sin() * 0.1;
                after[j] = before[j]; // Start same
            }

            // Single key change affects ~3-5 dimensions (field hash spread)
            let change_start = (i * 7) % (dimension - 5);
            for j in 0..4 {
                after[change_start + j] += 0.5 * ((i + j) as f32 / 100.0).cos();
            }

            SparseVector::from_diff(&before, &after, 1e-6)
        })
        .collect();

    let avg_nnz: f32 = single_key_deltas
        .iter()
        .map(|d| d.nnz() as f32)
        .sum::<f32>()
        / 100.0;
    let avg_sparsity: f32 = single_key_deltas.iter().map(|d| d.sparsity()).sum::<f32>() / 100.0;
    println!("  Avg NNZ: {:.1} / {}", avg_nnz, dimension);
    println!("  Avg Sparsity: {:.1}%", avg_sparsity * 100.0);

    // Pattern 2: Multi-field update (updating a few fields at once)
    // Simulates: UPDATE users SET name = 'Bob', email = 'bob@x.com', age = 30
    println!("\n--- Pattern 2: Multi-Field Update (3-5 fields) ---");
    let multi_field_deltas: Vec<SparseVector> = (0..100)
        .map(|i| {
            let mut before = vec![0.0f32; dimension];
            let mut after = vec![0.0f32; dimension];

            for j in 0..dimension {
                before[j] = ((i * 17 + j) as f32 / 1000.0).sin() * 0.1;
                after[j] = before[j];
            }

            // 3-5 fields change, each affecting ~3 dimensions
            let num_fields = 3 + (i % 3);
            for f in 0..num_fields {
                let change_start = (i * 7 + f * 31) % (dimension - 3);
                for j in 0..3 {
                    after[change_start + j] += 0.3 * ((i + f + j) as f32 / 100.0).cos();
                }
            }

            SparseVector::from_diff(&before, &after, 1e-6)
        })
        .collect();

    let avg_nnz: f32 = multi_field_deltas
        .iter()
        .map(|d| d.nnz() as f32)
        .sum::<f32>()
        / 100.0;
    let avg_sparsity: f32 = multi_field_deltas.iter().map(|d| d.sparsity()).sum::<f32>() / 100.0;
    println!("  Avg NNZ: {:.1} / {}", avg_nnz, dimension);
    println!("  Avg Sparsity: {:.1}%", avg_sparsity * 100.0);

    // Pattern 3: Batch insert (inserting new record)
    // Simulates: INSERT INTO users VALUES (...)
    // New record = embedding from zero state
    println!("\n--- Pattern 3: New Record Insert ---");
    let insert_deltas: Vec<SparseVector> = (0..100)
        .map(|i| {
            let before = vec![0.0f32; dimension]; // Empty state
            let mut after = vec![0.0f32; dimension];

            // New record has ~20-30% of dimensions populated
            let num_fields = 25 + (i % 10);
            for f in 0..num_fields {
                let pos = (i * 7 + f * 13) % dimension;
                after[pos] = 0.1 + 0.5 * ((i + f) as f32 / 100.0).sin().abs();
            }

            SparseVector::from_diff(&before, &after, 1e-6)
        })
        .collect();

    let avg_nnz: f32 = insert_deltas.iter().map(|d| d.nnz() as f32).sum::<f32>() / 100.0;
    let avg_sparsity: f32 = insert_deltas.iter().map(|d| d.sparsity()).sum::<f32>() / 100.0;
    println!("  Avg NNZ: {:.1} / {}", avg_nnz, dimension);
    println!("  Avg Sparsity: {:.1}%", avg_sparsity * 100.0);

    // Pattern 4: Counter increment (very sparse - one value changes)
    // Simulates: UPDATE counters SET value = value + 1
    println!("\n--- Pattern 4: Counter/Increment (minimal change) ---");
    let counter_deltas: Vec<SparseVector> = (0..100)
        .map(|i| {
            let mut before = vec![0.0f32; dimension];
            let mut after = vec![0.0f32; dimension];

            for j in 0..dimension {
                before[j] = ((i * 17 + j) as f32 / 1000.0).sin() * 0.1;
                after[j] = before[j];
            }

            // Only 1-2 dimensions change (counter field)
            let pos = (i * 7) % dimension;
            after[pos] += 0.01; // Small increment

            SparseVector::from_diff(&before, &after, 1e-6)
        })
        .collect();

    let avg_nnz: f32 = counter_deltas.iter().map(|d| d.nnz() as f32).sum::<f32>() / 100.0;
    let avg_sparsity: f32 = counter_deltas.iter().map(|d| d.sparsity()).sum::<f32>() / 100.0;
    println!("  Avg NNZ: {:.1} / {}", avg_nnz, dimension);
    println!("  Avg Sparsity: {:.1}%", avg_sparsity * 100.0);

    // Pattern 5: Bulk update (updating many records - rare but happens)
    // Simulates: UPDATE users SET status = 'inactive' WHERE last_login < date
    println!("\n--- Pattern 5: Bulk/Migration (many fields) ---");
    let bulk_deltas: Vec<SparseVector> = (0..100)
        .map(|i| {
            let mut before = vec![0.0f32; dimension];
            let mut after = vec![0.0f32; dimension];

            for j in 0..dimension {
                before[j] = ((i * 17 + j) as f32 / 1000.0).sin() * 0.1;
                after[j] = before[j];
            }

            // 40-60% of dimensions change (bulk operation)
            let changes = 50 + (i % 20);
            for f in 0..changes {
                let pos = (i * 7 + f * 3) % dimension;
                after[pos] += 0.2 * ((i + f) as f32 / 100.0).cos();
            }

            SparseVector::from_diff(&before, &after, 1e-6)
        })
        .collect();

    let avg_nnz: f32 = bulk_deltas.iter().map(|d| d.nnz() as f32).sum::<f32>() / 100.0;
    let avg_sparsity: f32 = bulk_deltas.iter().map(|d| d.sparsity()).sum::<f32>() / 100.0;
    println!("  Avg NNZ: {:.1} / {}", avg_nnz, dimension);
    println!("  Avg Sparsity: {:.1}%", avg_sparsity * 100.0);

    // Pattern 6: Graph edge addition (adding relationship)
    // Simulates: CREATE (a)-[:FOLLOWS]->(b)
    println!("\n--- Pattern 6: Graph Edge (relationship change) ---");
    let edge_deltas: Vec<SparseVector> = (0..100)
        .map(|i| {
            let mut before = vec![0.0f32; dimension];
            let mut after = vec![0.0f32; dimension];

            for j in 0..dimension {
                before[j] = ((i * 17 + j) as f32 / 1000.0).sin() * 0.05;
                after[j] = before[j];
            }

            // Edge affects ~6-8 dimensions (from/to/type/properties)
            let edge_dims = 6 + (i % 3);
            for f in 0..edge_dims {
                let pos = (i * 11 + f * 17) % dimension;
                after[pos] += 0.4 * ((i + f) as f32 / 50.0).sin();
            }

            SparseVector::from_diff(&before, &after, 1e-6)
        })
        .collect();

    let avg_nnz: f32 = edge_deltas.iter().map(|d| d.nnz() as f32).sum::<f32>() / 100.0;
    let avg_sparsity: f32 = edge_deltas.iter().map(|d| d.sparsity()).sum::<f32>() / 100.0;
    println!("  Avg NNZ: {:.1} / {}", avg_nnz, dimension);
    println!("  Avg Sparsity: {:.1}%", avg_sparsity * 100.0);

    println!("\n--- Summary ---");
    println!("Pattern                  | Avg NNZ | Sparsity | Speedup vs Dense");
    println!("-------------------------|---------|----------|------------------");

    let patterns = [
        ("Single Key Update", &single_key_deltas),
        ("Multi-Field Update", &multi_field_deltas),
        ("New Record Insert", &insert_deltas),
        ("Counter Increment", &counter_deltas),
        ("Bulk Migration", &bulk_deltas),
        ("Graph Edge", &edge_deltas),
    ];

    for (name, deltas) in patterns.iter() {
        let avg_nnz: f32 = deltas.iter().map(|d| d.nnz() as f32).sum::<f32>() / 100.0;
        let avg_sparsity: f32 = deltas.iter().map(|d| d.sparsity()).sum::<f32>() / 100.0;
        // Estimated speedup based on sparsity (from our earlier benchmarks)
        let speedup = if avg_sparsity > 0.95 {
            "~10x"
        } else if avg_sparsity > 0.90 {
            "~3x"
        } else if avg_sparsity > 0.80 {
            "~2x"
        } else if avg_sparsity > 0.50 {
            "~1x"
        } else {
            "<1x (overhead)"
        };
        println!(
            "{:24} | {:>7.1} | {:>7.1}% | {}",
            name,
            avg_nnz,
            avg_sparsity * 100.0,
            speedup
        );
    }

    println!("\n========================================\n");

    // Now benchmark conflict detection with realistic workload mix
    // 70% single-key, 20% multi-field, 5% insert, 5% other
    let config = ConsensusConfig::default();
    let consensus = ConsensusManager::new(config);

    let realistic_deltas: Vec<DeltaVector> = (0..100)
        .map(|i| {
            let delta = match i % 20 {
                0..=13 => single_key_deltas[i % 100].clone(),   // 70%
                14..=17 => multi_field_deltas[i % 100].clone(), // 20%
                18 => insert_deltas[i % 100].clone(),           // 5%
                _ => edge_deltas[i % 100].clone(),              // 5%
            };
            DeltaVector::from_sparse(delta, HashSet::from([format!("key_{}", i)]), i as u64)
        })
        .collect();

    let mix_avg_nnz: f32 = realistic_deltas.iter().map(|d| d.nnz() as f32).sum::<f32>() / 100.0;
    let mix_avg_sparsity: f32 = realistic_deltas.iter().map(|d| d.sparsity()).sum::<f32>() / 100.0;

    println!("Realistic Workload Mix (70% single-key, 20% multi-field, 10% other):");
    println!("  Avg NNZ: {:.1} / {}", mix_avg_nnz, dimension);
    println!("  Avg Sparsity: {:.1}%\n", mix_avg_sparsity * 100.0);

    // Benchmark with realistic mix
    group.bench_function("realistic_workload_mix_100_deltas", |b| {
        b.iter(|| black_box(consensus.batch_detect_conflicts(&realistic_deltas)))
    });

    // Compare with worst-case (bulk updates)
    let bulk_delta_vectors: Vec<DeltaVector> = bulk_deltas
        .iter()
        .enumerate()
        .map(|(i, d)| {
            DeltaVector::from_sparse(d.clone(), HashSet::from([format!("key_{}", i)]), i as u64)
        })
        .collect();

    group.bench_function("bulk_migration_worst_case_100", |b| {
        b.iter(|| black_box(consensus.batch_detect_conflicts(&bulk_delta_vectors)))
    });

    // Compare with best-case (counter increments)
    let counter_delta_vectors: Vec<DeltaVector> = counter_deltas
        .iter()
        .enumerate()
        .map(|(i, d)| {
            DeltaVector::from_sparse(d.clone(), HashSet::from([format!("key_{}", i)]), i as u64)
        })
        .collect();

    group.bench_function("counter_increment_best_case_100", |b| {
        b.iter(|| black_box(consensus.batch_detect_conflicts(&counter_delta_vectors)))
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
// Raft Consensus Benchmarks
// ============================================================================

fn bench_raft_operations(c: &mut Criterion) {
    use tensor_chain::{MemoryTransport, RaftConfig, RaftNode};

    let mut group = c.benchmark_group("raft_operations");

    // Benchmark RaftNode creation
    group.bench_function("raft_node_create", |b| {
        b.iter(|| {
            let transport = Arc::new(MemoryTransport::new("bench_node".to_string()));
            let node = RaftNode::new(
                "bench_node".to_string(),
                vec!["peer1".to_string(), "peer2".to_string()],
                transport,
                RaftConfig::default(),
            );
            black_box(node)
        })
    });

    // Benchmark become_leader (state transition)
    group.bench_function("raft_become_leader", |b| {
        let transport = Arc::new(MemoryTransport::new("bench_node".to_string()));
        let node = Arc::new(RaftNode::new(
            "bench_node".to_string(),
            vec!["peer1".to_string(), "peer2".to_string()],
            transport,
            RaftConfig::default(),
        ));

        b.iter(|| {
            node.become_leader();
            black_box(node.state())
        })
    });

    // Benchmark heartbeat stats snapshot
    group.bench_function("raft_heartbeat_stats_snapshot", |b| {
        let transport = Arc::new(MemoryTransport::new("bench_node".to_string()));
        let node = Arc::new(RaftNode::new(
            "bench_node".to_string(),
            vec!["peer1".to_string(), "peer2".to_string()],
            transport,
            RaftConfig::default(),
        ));

        b.iter(|| black_box(node.heartbeat_stats_snapshot()))
    });

    // Benchmark log length access
    group.bench_function("raft_log_length", |b| {
        let transport = Arc::new(MemoryTransport::new("bench_node".to_string()));
        let node = Arc::new(RaftNode::new(
            "bench_node".to_string(),
            vec!["peer1".to_string(), "peer2".to_string()],
            transport,
            RaftConfig::default(),
        ));
        node.become_leader();

        b.iter(|| black_box(node.log_length()))
    });

    // Benchmark Raft stats collection
    group.bench_function("raft_stats_snapshot", |b| {
        let transport = Arc::new(MemoryTransport::new("bench_node".to_string()));
        let node = Arc::new(RaftNode::new(
            "bench_node".to_string(),
            vec!["peer1".to_string(), "peer2".to_string()],
            transport,
            RaftConfig::default(),
        ));

        b.iter(|| black_box(node.stats()))
    });

    group.finish();
}

// ============================================================================
// 2PC Distributed Transaction Benchmarks
// ============================================================================

fn bench_2pc_operations(c: &mut Criterion) {
    use tensor_chain::{
        ConsensusManager, DistributedTxConfig, DistributedTxCoordinator, LockManager, TxParticipant,
    };

    let mut group = c.benchmark_group("2pc_operations");

    // Benchmark LockManager operations
    group.bench_function("lock_manager_acquire", |b| {
        b.iter_batched(
            LockManager::new,
            |lm| {
                let handle = lm.try_lock(1, &["key1".to_string(), "key2".to_string()]);
                black_box(handle)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("lock_manager_release", |b| {
        b.iter_batched(
            || {
                let lm = LockManager::new();
                let _ = lm.try_lock(1, &["key1".to_string(), "key2".to_string()]);
                lm
            },
            |lm| {
                lm.release(1);
                black_box(lm.lock_count_for_transaction(1))
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("lock_manager_is_locked", |b| {
        let lm = LockManager::new();
        let _ = lm.try_lock(1, &["key1".to_string()]);

        b.iter(|| black_box(lm.is_locked("key1")))
    });

    // Benchmark coordinator creation
    group.bench_function("coordinator_create", |b| {
        b.iter(|| {
            let consensus = ConsensusManager::default_config();
            let coord = DistributedTxCoordinator::new(consensus, DistributedTxConfig::default());
            black_box(coord)
        })
    });

    // Benchmark coordinator stats
    group.bench_function("coordinator_stats", |b| {
        let consensus = ConsensusManager::default_config();
        let coord = DistributedTxCoordinator::new(consensus, DistributedTxConfig::default());

        b.iter(|| black_box(coord.stats()))
    });

    // Benchmark participant creation
    group.bench_function("participant_create", |b| {
        b.iter(|| {
            let participant = TxParticipant::new_in_memory();
            black_box(participant)
        })
    });

    group.finish();
}

// ============================================================================
// Gossip Protocol Benchmarks
// ============================================================================

fn bench_gossip_operations(c: &mut Criterion) {
    use tensor_chain::{GossipMessage, GossipNodeState, LWWMembershipState, NodeHealth};

    let mut group = c.benchmark_group("gossip_operations");

    // Benchmark LWW state creation
    group.bench_function("lww_state_create", |b| {
        b.iter(|| {
            let state = LWWMembershipState::new();
            black_box(state)
        })
    });

    // Benchmark LWW state merge
    group.bench_function("lww_state_merge", |b| {
        let mut state1 = LWWMembershipState::new();
        let node_state1 = GossipNodeState {
            node_id: "peer1".to_string(),
            health: NodeHealth::Healthy,
            timestamp: 100,
            updated_at: 1000,
            incarnation: 1,
        };
        state1.merge(&[node_state1]);

        let node_state2 = GossipNodeState {
            node_id: "peer2".to_string(),
            health: NodeHealth::Healthy,
            timestamp: 101,
            updated_at: 1001,
            incarnation: 1,
        };

        b.iter(|| {
            let mut merged = state1.clone();
            merged.merge(std::slice::from_ref(&node_state2));
            black_box(merged)
        })
    });

    // Benchmark GossipNodeState creation
    group.bench_function("gossip_node_state_create", |b| {
        b.iter(|| {
            let state = GossipNodeState {
                node_id: "node1".to_string(),
                health: NodeHealth::Healthy,
                timestamp: 100,
                updated_at: 1000,
                incarnation: 1,
            };
            black_box(state)
        })
    });

    // Benchmark gossip message serialization
    group.bench_function("gossip_message_serialize", |b| {
        let states = vec![
            GossipNodeState {
                node_id: "peer1".to_string(),
                health: NodeHealth::Healthy,
                timestamp: 100,
                updated_at: 1000,
                incarnation: 1,
            },
            GossipNodeState {
                node_id: "peer2".to_string(),
                health: NodeHealth::Healthy,
                timestamp: 101,
                updated_at: 1001,
                incarnation: 1,
            },
        ];
        let msg = GossipMessage::Sync {
            sender: "node1".to_string(),
            states,
            sender_time: 100,
        };

        b.iter(|| {
            let bytes = bincode::serialize(&msg).unwrap();
            black_box(bytes)
        })
    });

    // Benchmark gossip message deserialization
    group.bench_function("gossip_message_deserialize", |b| {
        let states = vec![GossipNodeState {
            node_id: "peer1".to_string(),
            health: NodeHealth::Healthy,
            timestamp: 100,
            updated_at: 1000,
            incarnation: 1,
        }];
        let msg = GossipMessage::Sync {
            sender: "node1".to_string(),
            states,
            sender_time: 100,
        };
        let bytes = bincode::serialize(&msg).unwrap();

        b.iter(|| {
            let deserialized: GossipMessage = bincode::deserialize(&bytes).unwrap();
            black_box(deserialized)
        })
    });

    group.finish();
}

// ============================================================================
// Snapshot Operation Benchmarks
// ============================================================================

fn bench_snapshot_operations(c: &mut Criterion) {
    use tensor_chain::{
        MemoryTransport, RaftConfig, RaftMembershipConfig, RaftNode, SnapshotMetadata,
    };

    let mut group = c.benchmark_group("snapshot_operations");

    // Helper to create snapshot metadata
    fn create_snapshot_metadata() -> SnapshotMetadata {
        SnapshotMetadata {
            last_included_index: 100,
            last_included_term: 5,
            snapshot_hash: [0u8; 32],
            config: vec!["node1".to_string(), "node2".to_string()],
            membership: RaftMembershipConfig {
                voters: vec!["node1".to_string(), "node2".to_string()],
                learners: vec![],
                joint: None,
                config_index: 0,
            },
            created_at: 1700000000,
            size: 1024,
        }
    }

    // Benchmark SnapshotMetadata creation
    group.bench_function("snapshot_metadata_create", |b| {
        b.iter(|| {
            let meta = create_snapshot_metadata();
            black_box(meta)
        })
    });

    // Benchmark SnapshotMetadata serialization
    group.bench_function("snapshot_metadata_serialize", |b| {
        let meta = create_snapshot_metadata();

        b.iter(|| {
            let bytes = bincode::serialize(&meta).unwrap();
            black_box(bytes)
        })
    });

    // Benchmark SnapshotMetadata deserialization
    group.bench_function("snapshot_metadata_deserialize", |b| {
        let meta = create_snapshot_metadata();
        let bytes = bincode::serialize(&meta).unwrap();

        b.iter(|| {
            let deserialized: SnapshotMetadata = bincode::deserialize(&bytes).unwrap();
            black_box(deserialized)
        })
    });

    // Benchmark RaftMembershipConfig creation
    group.bench_function("raft_membership_config_create", |b| {
        b.iter(|| {
            let config = RaftMembershipConfig {
                voters: vec![
                    "node1".to_string(),
                    "node2".to_string(),
                    "node3".to_string(),
                ],
                learners: vec!["learner1".to_string()],
                joint: None,
                config_index: 10,
            };
            black_box(config)
        })
    });

    // Benchmark RaftNode with store
    group.bench_function("raft_with_store_create", |b| {
        b.iter_batched(
            || {
                let store = TensorStore::new();
                for i in 0..10 {
                    let mut data = tensor_store::TensorData::new();
                    data.set(
                        "id".to_string(),
                        tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(i)),
                    );
                    let _ = store.put(format!("key_{}", i), data);
                }
                store
            },
            |store| {
                let transport = Arc::new(MemoryTransport::new("bench_node".to_string()));
                let node = RaftNode::with_store(
                    "bench_node".to_string(),
                    vec!["peer1".to_string()],
                    transport,
                    RaftConfig::default(),
                    &store,
                );
                black_box(node)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// ============================================================================
// Membership Health Check Benchmarks
// ============================================================================

fn bench_membership_operations(c: &mut Criterion) {
    use std::net::SocketAddr;
    use tensor_chain::{
        ClusterConfig, LocalNodeConfig, MembershipManager, MembershipStats, MemoryTransport,
    };

    let mut group = c.benchmark_group("membership_operations");

    // Helper to create a manager
    fn create_manager() -> MembershipManager {
        let local = LocalNodeConfig {
            node_id: "node1".to_string(),
            bind_address: "127.0.0.1:8000".parse::<SocketAddr>().unwrap(),
        };
        let config = ClusterConfig::new("test_cluster", local)
            .with_peer("peer1", "127.0.0.1:8001".parse().unwrap())
            .with_peer("peer2", "127.0.0.1:8002".parse().unwrap());
        let transport = Arc::new(MemoryTransport::new("node1".to_string()));
        MembershipManager::new(config, transport)
    }

    // Benchmark membership manager creation
    group.bench_function("membership_manager_create", |b| {
        b.iter(|| {
            let manager = create_manager();
            black_box(manager)
        })
    });

    // Benchmark cluster view
    group.bench_function("membership_view", |b| {
        let manager = create_manager();

        b.iter(|| black_box(manager.view()))
    });

    // Benchmark partition status check
    group.bench_function("membership_partition_status", |b| {
        let manager = create_manager();

        b.iter(|| black_box(manager.partition_status()))
    });

    // Benchmark node status lookup
    group.bench_function("membership_node_status", |b| {
        let manager = create_manager();

        b.iter(|| black_box(manager.node_status(&"peer1".to_string())))
    });

    // Benchmark stats snapshot
    group.bench_function("membership_stats_snapshot", |b| {
        let stats = MembershipStats::new();
        // Add some data
        stats
            .health_checks
            .fetch_add(100, std::sync::atomic::Ordering::Relaxed);
        stats
            .health_check_failures
            .fetch_add(5, std::sync::atomic::Ordering::Relaxed);

        b.iter(|| black_box(stats.snapshot()))
    });

    // Benchmark peer IDs retrieval
    group.bench_function("membership_peer_ids", |b| {
        let manager = create_manager();

        b.iter(|| black_box(manager.peer_ids()))
    });

    group.finish();
}

// ============================================================================
// Deadlock Detection Benchmarks
// ============================================================================

fn bench_deadlock_detection(c: &mut Criterion) {
    use tensor_chain::{DeadlockDetector, DeadlockDetectorConfig, WaitForGraph};

    let mut group = c.benchmark_group("deadlock_detection");

    // Benchmark wait-for graph edge addition
    group.bench_function("wait_graph_add_edge", |b| {
        b.iter_batched(
            WaitForGraph::new,
            |graph| {
                graph.add_wait(1, 2, None);
                graph.add_wait(2, 3, None);
                graph.add_wait(3, 4, None);
                black_box(graph)
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Benchmark cycle detection (no cycle)
    group.bench_function("wait_graph_detect_no_cycle", |b| {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 3, None);
        graph.add_wait(3, 4, None);

        b.iter(|| black_box(graph.detect_cycles()))
    });

    // Benchmark cycle detection (with cycle)
    group.bench_function("wait_graph_detect_with_cycle", |b| {
        let graph = WaitForGraph::new();
        graph.add_wait(1, 2, None);
        graph.add_wait(2, 3, None);
        graph.add_wait(3, 1, None); // Creates cycle

        b.iter(|| black_box(graph.detect_cycles()))
    });

    // Benchmark deadlock detector
    group.bench_function("deadlock_detector_detect", |b| {
        let config = DeadlockDetectorConfig::default();
        let detector = DeadlockDetector::new(config);
        detector.graph().add_wait(1, 2, None);
        detector.graph().add_wait(2, 3, None);
        detector.graph().add_wait(3, 1, None);

        b.iter(|| black_box(detector.detect()))
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
    bench_conflict_detection_scaling,
    bench_sparsity_comparison,
    bench_conflict_operation_breakdown,
    bench_high_dimension_sparse,
    measure_real_delta_sparsity,
);

criterion_group!(
    distributed_benches,
    bench_raft_operations,
    bench_2pc_operations,
    bench_gossip_operations,
    bench_snapshot_operations,
    bench_membership_operations,
    bench_deadlock_detection,
);

criterion_main!(benches, distributed_benches);
