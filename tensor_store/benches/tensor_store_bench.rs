use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use tensor_store::{
    ArchetypeRegistry, BloomFilter, DeltaVector, KMeans, KMeansConfig, KMeansInit, ScalarValue,
    SparseVector, TensorData, TensorStore, TensorValue, TieredConfig, TieredStore,
};

fn create_test_tensor(id: i64) -> TensorData {
    let mut tensor = TensorData::new();
    tensor.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
    tensor.set(
        "name",
        TensorValue::Scalar(ScalarValue::String(format!("entity_{}", id))),
    );
    tensor.set(
        "embedding",
        TensorValue::Vector(vec![id as f32; 128]), // 128-dim embedding
    );
    tensor
}

fn bench_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("put");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let store = TensorStore::new();
                for i in 0..size {
                    store
                        .put(format!("key:{}", i), create_test_tensor(i as i64))
                        .unwrap();
                }
                black_box(&store);
            });
        });
    }

    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("get");

    for size in [100, 1000, 10000].iter() {
        let store = TensorStore::new();
        for i in 0..*size {
            store
                .put(format!("key:{}", i), create_test_tensor(i as i64))
                .unwrap();
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(store.get(&format!("key:{}", i)).unwrap());
                }
            });
        });
    }

    group.finish();
}

fn bench_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("scan");

    // Setup: 10k entities split across prefixes
    let store = TensorStore::new();
    for i in 0..10000 {
        let prefix = match i % 3 {
            0 => "user",
            1 => "post",
            _ => "comment",
        };
        store
            .put(format!("{}:{}", prefix, i), create_test_tensor(i as i64))
            .unwrap();
    }

    group.bench_function("scan_1k_of_10k", |b| {
        b.iter(|| {
            black_box(store.scan("user:"));
        });
    });

    group.bench_function("scan_count_1k_of_10k", |b| {
        b.iter(|| {
            black_box(store.scan_count("user:"));
        });
    });

    group.finish();
}

fn bench_concurrent_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_writes");

    for num_threads in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let store = Arc::new(TensorStore::new());
                    let mut handles = vec![];

                    for t in 0..num_threads {
                        let store = Arc::clone(&store);
                        handles.push(thread::spawn(move || {
                            for i in 0..1000 {
                                store
                                    .put(
                                        format!("thread{}:key{}", t, i),
                                        create_test_tensor((t * 1000 + i) as i64),
                                    )
                                    .unwrap();
                            }
                        }));
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&store);
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_writes_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_writes_contention");

    // High contention: many threads writing to same keys
    for num_threads in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", num_threads),
            num_threads,
            |b, &num_threads| {
                b.iter(|| {
                    let store = Arc::new(TensorStore::new());
                    let mut handles = vec![];

                    for t in 0..num_threads {
                        let store = Arc::clone(&store);
                        handles.push(thread::spawn(move || {
                            for i in 0..500 {
                                // Only 100 unique keys = high contention
                                let key = format!("key:{}", i % 100);
                                store
                                    .put(key, create_test_tensor((t * 500 + i) as i64))
                                    .unwrap();
                            }
                        }));
                    }

                    for h in handles {
                        h.join().unwrap();
                    }

                    black_box(&store);
                });
            },
        );
    }

    group.finish();
}

fn bench_mixed_read_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_read_write");

    // Pre-populate store
    let store = Arc::new(TensorStore::new());
    for i in 0..1000 {
        store
            .put(format!("key:{}", i), create_test_tensor(i as i64))
            .unwrap();
    }

    group.bench_function("4_readers_2_writers", |b| {
        b.iter(|| {
            let store = Arc::clone(&store);
            let mut handles = vec![];

            // 4 reader threads
            for _ in 0..4 {
                let store = Arc::clone(&store);
                handles.push(thread::spawn(move || {
                    for i in 0..250 {
                        let _ = store.get(&format!("key:{}", i));
                    }
                }));
            }

            // 2 writer threads
            for t in 0..2 {
                let store = Arc::clone(&store);
                handles.push(thread::spawn(move || {
                    for i in 0..250 {
                        store
                            .put(
                                format!("new:{}:{}", t, i),
                                create_test_tensor((t * 250 + i) as i64),
                            )
                            .unwrap();
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    group.finish();
}

fn bench_bloom_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter");

    // Benchmark Bloom filter operations directly
    let filter = BloomFilter::new(10000, 0.01);
    for i in 0..10000 {
        filter.add(&format!("key:{}", i));
    }

    group.bench_function("add", |b| {
        let filter = BloomFilter::new(10000, 0.01);
        let mut i = 0u64;
        b.iter(|| {
            filter.add(&format!("key:{}", i));
            i = i.wrapping_add(1);
        });
    });

    group.bench_function("might_contain_hit", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let result = filter.might_contain(&format!("key:{}", i % 10000));
            i = i.wrapping_add(1);
            black_box(result);
        });
    });

    group.bench_function("might_contain_miss", |b| {
        let mut i = 0u64;
        b.iter(|| {
            // Keys 10000+ were never added
            let result = filter.might_contain(&format!("key:{}", 10000 + i));
            i = i.wrapping_add(1);
            black_box(result);
        });
    });

    group.finish();
}

fn bench_sparse_lookups(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_lookups");

    // Sparse key space: only 1000 keys exist out of potential millions
    let num_existing = 1000;

    // Store WITHOUT Bloom filter
    let store_no_bloom = TensorStore::new();
    for i in 0..num_existing {
        store_no_bloom
            .put(format!("key:{}", i), create_test_tensor(i as i64))
            .unwrap();
    }

    // Store WITH Bloom filter
    let store_bloom = TensorStore::with_bloom_filter(num_existing, 0.01);
    for i in 0..num_existing {
        store_bloom
            .put(format!("key:{}", i), create_test_tensor(i as i64))
            .unwrap();
    }

    // Benchmark: Looking up keys that DON'T exist (sparse negative lookups)
    group.bench_function("negative_lookup_no_bloom", |b| {
        let mut i = 0u64;
        b.iter(|| {
            // Keys starting at 1M definitely don't exist
            let result = store_no_bloom.exists(&format!("key:{}", 1_000_000 + i));
            i = i.wrapping_add(1);
            black_box(result);
        });
    });

    group.bench_function("negative_lookup_with_bloom", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let result = store_bloom.exists(&format!("key:{}", 1_000_000 + i));
            i = i.wrapping_add(1);
            black_box(result);
        });
    });

    // Benchmark: Looking up keys that DO exist (positive lookups)
    group.bench_function("positive_lookup_no_bloom", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let result = store_no_bloom.exists(&format!("key:{}", i % num_existing as u64));
            i = i.wrapping_add(1);
            black_box(result);
        });
    });

    group.bench_function("positive_lookup_with_bloom", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let result = store_bloom.exists(&format!("key:{}", i % num_existing as u64));
            i = i.wrapping_add(1);
            black_box(result);
        });
    });

    // Benchmark: 90% misses, 10% hits (typical sparse workload)
    group.bench_function("sparse_workload_no_bloom", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let key = if i % 10 == 0 {
                // 10% hits
                format!("key:{}", i % num_existing as u64)
            } else {
                // 90% misses
                format!("key:{}", 1_000_000 + i)
            };
            let result = store_no_bloom.exists(&key);
            i = i.wrapping_add(1);
            black_box(result);
        });
    });

    group.bench_function("sparse_workload_with_bloom", |b| {
        let mut i = 0u64;
        b.iter(|| {
            let key = if i % 10 == 0 {
                format!("key:{}", i % num_existing as u64)
            } else {
                format!("key:{}", 1_000_000 + i)
            };
            let result = store_bloom.exists(&key);
            i = i.wrapping_add(1);
            black_box(result);
        });
    });

    group.finish();
}

fn bench_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot");

    // Create test stores of different sizes
    for size in [100, 1000, 10000].iter() {
        let store = TensorStore::new();
        for i in 0..*size {
            store
                .put(format!("key:{}", i), create_test_tensor(i as i64))
                .unwrap();
        }

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join(format!("bench_snapshot_{}.bin", size));

        // Benchmark save
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("save", size), size, |b, _| {
            b.iter(|| {
                store.save_snapshot(&path).unwrap();
            });
        });

        // Save once for load benchmark
        store.save_snapshot(&path).unwrap();

        // Benchmark load
        group.bench_with_input(BenchmarkId::new("load", size), size, |b, _| {
            b.iter(|| {
                let loaded = TensorStore::load_snapshot(&path).unwrap();
                black_box(loaded);
            });
        });

        // Benchmark load with bloom filter
        group.bench_with_input(BenchmarkId::new("load_with_bloom", size), size, |b, _| {
            b.iter(|| {
                let loaded =
                    TensorStore::load_snapshot_with_bloom_filter(&path, *size * 2, 0.01).unwrap();
                black_box(loaded);
            });
        });

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    group.finish();
}

// ============================================================================
// SparseVector Benchmarks
// ============================================================================

/// Generate a dense vector with given sparsity (fraction of zeros)
fn generate_dense_vector(dimension: usize, sparsity: f32, seed: u64) -> Vec<f32> {
    let mut rng_state = seed;
    let mut dense = Vec::with_capacity(dimension);

    for _ in 0..dimension {
        // Simple LCG random number generator
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let rand_float = (rng_state >> 33) as f32 / (1u64 << 31) as f32;

        if rand_float < sparsity {
            dense.push(0.0);
        } else {
            // Generate non-zero value
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((rng_state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
            dense.push(if val == 0.0 { 0.001 } else { val });
        }
    }

    dense
}

fn bench_sparse_vector_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vector/construction");

    for dim in [128, 768, 1536].iter() {
        for sparsity in [0.5, 0.9, 0.99].iter() {
            let dense = generate_dense_vector(*dim, *sparsity, 42);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("from_dense_{}d", dim),
                    format!("{:.0}%", sparsity * 100.0),
                ),
                &dense,
                |b, dense| {
                    b.iter(|| {
                        black_box(SparseVector::from_dense(dense));
                    });
                },
            );
        }
    }

    // Benchmark threshold-based construction
    let dense_768 = generate_dense_vector(768, 0.0, 42); // No zeros initially
    for threshold in [0.01, 0.1, 0.3].iter() {
        group.bench_with_input(
            BenchmarkId::new("from_dense_threshold_768d", format!("{}", threshold)),
            threshold,
            |b, &threshold| {
                b.iter(|| {
                    black_box(SparseVector::from_dense_with_threshold(
                        &dense_768, threshold,
                    ));
                });
            },
        );
    }

    group.finish();
}

fn bench_sparse_vector_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vector/lookup");

    for dim in [128, 768, 1536].iter() {
        let dense = generate_dense_vector(*dim, 0.9, 42);
        let sparse = SparseVector::from_dense(&dense);

        // Benchmark sparse lookup
        group.bench_with_input(
            BenchmarkId::new("sparse_get", format!("{}d_90%", dim)),
            &sparse,
            |b, sparse| {
                let mut i = 0usize;
                b.iter(|| {
                    let idx = i % sparse.dimension();
                    black_box(sparse.get(idx));
                    i = i.wrapping_add(1);
                });
            },
        );

        // Benchmark dense lookup for comparison
        group.bench_with_input(
            BenchmarkId::new("dense_get", format!("{}d", dim)),
            &dense,
            |b, dense| {
                let mut i = 0usize;
                b.iter(|| {
                    let idx = i % dense.len();
                    black_box(dense[idx]);
                    i = i.wrapping_add(1);
                });
            },
        );
    }

    group.finish();
}

fn bench_sparse_vector_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vector/dot_product");

    for dim in [128, 768, 1536].iter() {
        for sparsity in [0.5, 0.9, 0.99].iter() {
            let dense_a = generate_dense_vector(*dim, *sparsity, 42);
            let dense_b = generate_dense_vector(*dim, *sparsity, 43);
            let sparse_a = SparseVector::from_dense(&dense_a);
            let sparse_b = SparseVector::from_dense(&dense_b);

            // Sparse-sparse dot product
            group.bench_with_input(
                BenchmarkId::new(
                    format!("sparse_sparse_{}d", dim),
                    format!("{:.0}%", sparsity * 100.0),
                ),
                &(&sparse_a, &sparse_b),
                |bench, (va, vb)| {
                    bench.iter(|| {
                        black_box(va.dot(vb));
                    });
                },
            );

            // Sparse-dense dot product
            group.bench_with_input(
                BenchmarkId::new(
                    format!("sparse_dense_{}d", dim),
                    format!("{:.0}%", sparsity * 100.0),
                ),
                &(&sparse_a, &dense_b),
                |bench, (sparse, dense)| {
                    bench.iter(|| {
                        black_box(sparse.dot_dense(dense));
                    });
                },
            );

            // Dense-dense dot product for comparison
            group.bench_with_input(
                BenchmarkId::new(
                    format!("dense_dense_{}d", dim),
                    format!("{:.0}%", sparsity * 100.0),
                ),
                &(&dense_a, &dense_b),
                |bench, (va, vb)| {
                    bench.iter(|| {
                        let result: f32 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
                        black_box(result);
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_sparse_vector_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vector/cosine_similarity");

    for dim in [128, 768].iter() {
        for sparsity in [0.5, 0.9].iter() {
            let dense_a = generate_dense_vector(*dim, *sparsity, 42);
            let dense_b = generate_dense_vector(*dim, *sparsity, 43);
            let sparse_a = SparseVector::from_dense(&dense_a);
            let sparse_b = SparseVector::from_dense(&dense_b);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("sparse_{}d", dim),
                    format!("{:.0}%", sparsity * 100.0),
                ),
                &(&sparse_a, &sparse_b),
                |bench, (va, vb)| {
                    bench.iter(|| {
                        black_box(va.cosine_similarity(vb));
                    });
                },
            );

            // Dense cosine for comparison
            group.bench_with_input(
                BenchmarkId::new(
                    format!("dense_{}d", dim),
                    format!("{:.0}%", sparsity * 100.0),
                ),
                &(&dense_a, &dense_b),
                |bench, (va, vb)| {
                    bench.iter(|| {
                        let dot: f32 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
                        let mag_a: f32 = va.iter().map(|x| x * x).sum::<f32>().sqrt();
                        let mag_b: f32 = vb.iter().map(|x| x * x).sum::<f32>().sqrt();
                        black_box(dot / (mag_a * mag_b));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_sparse_vector_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vector/addition");

    for dim in [128, 768].iter() {
        for sparsity in [0.5, 0.9].iter() {
            let dense_a = generate_dense_vector(*dim, *sparsity, 42);
            let dense_b = generate_dense_vector(*dim, *sparsity, 43);
            let sparse_a = SparseVector::from_dense(&dense_a);
            let sparse_b = SparseVector::from_dense(&dense_b);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("sparse_{}d", dim),
                    format!("{:.0}%", sparsity * 100.0),
                ),
                &(&sparse_a, &sparse_b),
                |bench, (va, vb)| {
                    bench.iter(|| {
                        black_box(va.add(vb));
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_sparse_vector_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vector/memory");
    group.sample_size(10); // Memory isn't timing-sensitive

    for dim in [128, 768, 1536].iter() {
        for sparsity in [0.5, 0.9, 0.99].iter() {
            let dense = generate_dense_vector(*dim, *sparsity, 42);
            let sparse = SparseVector::from_dense(&dense);

            let dense_bytes = dense.len() * std::mem::size_of::<f32>();
            let sparse_bytes = sparse.memory_bytes();
            let ratio = dense_bytes as f64 / sparse_bytes as f64;

            // This benchmark just reports the memory ratio via throughput
            group.throughput(Throughput::Bytes(sparse_bytes as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}d_{:.0}%_sparse", dim, sparsity * 100.0),
                    format!("ratio_{:.2}x", ratio),
                ),
                &sparse,
                |b, sparse| {
                    b.iter(|| {
                        black_box(sparse.memory_bytes());
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_sparse_vector_prune(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vector/prune");

    // Start with dense vector, prune to various thresholds
    let dense = generate_dense_vector(768, 0.0, 42);
    let sparse = SparseVector::from_dense(&dense);

    for threshold in [0.01, 0.1, 0.3, 0.5].iter() {
        group.bench_with_input(
            BenchmarkId::new("768d", format!("threshold_{}", threshold)),
            &(&sparse, *threshold),
            |b, (sparse, threshold)| {
                b.iter(|| {
                    black_box(sparse.pruned(*threshold));
                });
            },
        );
    }

    group.finish();
}

fn bench_sparse_vector_batch_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vector/batch_dot");

    // Benchmark: query vector against many stored vectors (typical similarity search)
    let dim = 768;
    let num_vectors = 1000;

    for sparsity in [0.5, 0.9].iter() {
        let query_dense = generate_dense_vector(dim, *sparsity, 0);
        let query_sparse = SparseVector::from_dense(&query_dense);

        // Generate corpus
        let corpus_dense: Vec<Vec<f32>> = (0..num_vectors)
            .map(|i| generate_dense_vector(dim, *sparsity, i as u64 + 100))
            .collect();
        let corpus_sparse: Vec<SparseVector> = corpus_dense
            .iter()
            .map(|v| SparseVector::from_dense(v))
            .collect();

        // Sparse query vs sparse corpus
        group.throughput(Throughput::Elements(num_vectors as u64));
        group.bench_with_input(
            BenchmarkId::new("sparse_corpus", format!("{:.0}%", sparsity * 100.0)),
            &(&query_sparse, &corpus_sparse),
            |b, (query, corpus)| {
                b.iter(|| {
                    let scores: Vec<f32> = corpus.iter().map(|v| query.dot(v)).collect();
                    black_box(scores);
                });
            },
        );

        // Sparse query vs dense corpus
        group.bench_with_input(
            BenchmarkId::new("dense_corpus", format!("{:.0}%", sparsity * 100.0)),
            &(&query_sparse, &corpus_dense),
            |b, (query, corpus)| {
                b.iter(|| {
                    let scores: Vec<f32> = corpus.iter().map(|v| query.dot_dense(v)).collect();
                    black_box(scores);
                });
            },
        );

        // Dense query vs dense corpus (baseline)
        group.bench_with_input(
            BenchmarkId::new("dense_baseline", format!("{:.0}%", sparsity * 100.0)),
            &(&query_dense, &corpus_dense),
            |b, (query, corpus)| {
                b.iter(|| {
                    let scores: Vec<f32> = corpus
                        .iter()
                        .map(|v| query.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
                        .collect();
                    black_box(scores);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_put,
    bench_get,
    bench_scan,
    bench_concurrent_writes,
    bench_concurrent_writes_contention,
    bench_mixed_read_write,
    bench_bloom_filter,
    bench_sparse_lookups,
    bench_snapshot,
);

criterion_group!(
    sparse_vector_benches,
    bench_sparse_vector_construction,
    bench_sparse_vector_lookup,
    bench_sparse_vector_dot_product,
    bench_sparse_vector_cosine,
    bench_sparse_vector_add,
    bench_sparse_vector_memory,
    bench_sparse_vector_prune,
    bench_sparse_vector_batch_dot,
);

// ============================================================================
// DeltaVector Benchmarks
// ============================================================================

fn create_archetype(dim: usize) -> Vec<f32> {
    (0..dim).map(|i| (i as f32 * 0.01).sin()).collect()
}

fn create_similar_vector(archetype: &[f32], delta_fraction: f32, seed: usize) -> Vec<f32> {
    archetype
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            let noise = ((seed * 31 + i * 17) as f32 * 0.0001).sin() * delta_fraction;
            val + noise
        })
        .collect()
}

fn bench_delta_vector_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_vector/construction");

    for dim in [128, 768, 1536] {
        let archetype = create_archetype(dim);

        for delta_fraction in [0.01, 0.05, 0.1] {
            let vector = create_similar_vector(&archetype, delta_fraction, 42);

            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}d", dim),
                    format!("{:.0}%_delta", delta_fraction * 100.0),
                ),
                &(&vector, &archetype),
                |bench, (vec, arch)| {
                    bench.iter(|| {
                        black_box(DeltaVector::from_dense_with_reference(vec, arch, 0, 0.001))
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_delta_vector_reconstruction(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_vector/reconstruction");

    for dim in [128, 768, 1536] {
        let archetype = create_archetype(dim);
        let vector = create_similar_vector(&archetype, 0.05, 42);
        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        group.bench_with_input(
            BenchmarkId::new("to_dense", format!("{}d", dim)),
            &(&delta, &archetype),
            |bench, (d, arch)| {
                bench.iter(|| black_box(d.to_dense(arch)));
            },
        );
    }

    group.finish();
}

fn bench_delta_vector_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_vector/dot_product");

    for dim in [128, 768, 1536] {
        let archetype = create_archetype(dim);
        let vector = create_similar_vector(&archetype, 0.05, 42);
        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.02).cos()).collect();

        // Precomputed archetype dot product
        let arch_dot_query: f32 = archetype.iter().zip(query.iter()).map(|(a, q)| a * q).sum();

        // Delta dot with precomputed (optimized path)
        group.bench_with_input(
            BenchmarkId::new("delta_precomputed", format!("{}d", dim)),
            &(&delta, &query, arch_dot_query),
            |bench, (d, q, precomputed)| {
                bench.iter(|| black_box(d.dot_dense_with_precomputed(q, *precomputed)));
            },
        );

        // Delta dot without precomputed
        group.bench_with_input(
            BenchmarkId::new("delta_full", format!("{}d", dim)),
            &(&delta, &query, &archetype),
            |bench, (d, q, arch)| {
                bench.iter(|| black_box(d.dot_dense(q, arch)));
            },
        );

        // Dense baseline
        group.bench_with_input(
            BenchmarkId::new("dense_baseline", format!("{}d", dim)),
            &(&vector, &query),
            |bench, (v, q)| {
                bench.iter(|| black_box(v.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>()));
            },
        );
    }

    group.finish();
}

fn bench_delta_same_archetype_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_vector/same_archetype_dot");

    for dim in [128, 768, 1536] {
        let archetype = create_archetype(dim);
        let arch_mag_sq: f32 = archetype.iter().map(|x| x * x).sum();

        let vec_a = create_similar_vector(&archetype, 0.05, 42);
        let vec_b = create_similar_vector(&archetype, 0.05, 123);

        let delta_a = DeltaVector::from_dense_with_reference(&vec_a, &archetype, 0, 0.001);
        let delta_b = DeltaVector::from_dense_with_reference(&vec_b, &archetype, 0, 0.001);

        // Delta-delta dot (optimized)
        group.bench_with_input(
            BenchmarkId::new("delta_delta", format!("{}d", dim)),
            &(&delta_a, &delta_b, &archetype, arch_mag_sq),
            |bench, (da, db, arch, mag_sq)| {
                bench.iter(|| black_box(da.dot_same_archetype(db, arch, *mag_sq)));
            },
        );

        // Dense baseline
        group.bench_with_input(
            BenchmarkId::new("dense_baseline", format!("{}d", dim)),
            &(&vec_a, &vec_b),
            |bench, (va, vb)| {
                bench.iter(|| black_box(va.iter().zip(vb.iter()).map(|(a, b)| a * b).sum::<f32>()));
            },
        );
    }

    group.finish();
}

fn bench_delta_vector_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_vector/memory");

    for dim in [128, 768, 1536] {
        let archetype = create_archetype(dim);
        let dense_bytes = dim * std::mem::size_of::<f32>();

        for delta_fraction in [0.01, 0.05, 0.1] {
            let vector = create_similar_vector(&archetype, delta_fraction, 42);
            let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);
            let ratio = dense_bytes as f32 / delta.memory_bytes() as f32;

            group.throughput(Throughput::Bytes(delta.memory_bytes() as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{}d_{:.0}%_delta", dim, delta_fraction * 100.0),
                    format!("ratio_{:.2}x", ratio),
                ),
                &delta,
                |bench, d| {
                    bench.iter(|| black_box(d.memory_bytes()));
                },
            );
        }
    }

    group.finish();
}

fn bench_archetype_registry(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_vector/registry");

    let dim = 768;
    let mut registry = ArchetypeRegistry::new(16);

    // Register some archetypes
    for i in 0..8 {
        let archetype: Vec<f32> = (0..dim)
            .map(|j| ((i * 100 + j) as f32 * 0.01).sin())
            .collect();
        registry.register(archetype);
    }

    let test_vector: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin() + 0.01).collect();

    // Find best archetype
    group.bench_function("find_best_archetype", |bench| {
        bench.iter(|| black_box(registry.find_best_archetype(&test_vector)));
    });

    // Encode vector
    group.bench_function("encode", |bench| {
        bench.iter(|| black_box(registry.encode(&test_vector, 0.001)));
    });

    // Decode delta
    let delta = registry.encode(&test_vector, 0.001).unwrap();
    group.bench_function("decode", |bench| {
        bench.iter(|| black_box(registry.decode(&delta)));
    });

    group.finish();
}

criterion_group!(
    delta_vector_benches,
    bench_delta_vector_construction,
    bench_delta_vector_reconstruction,
    bench_delta_vector_dot_product,
    bench_delta_same_archetype_dot,
    bench_delta_vector_memory,
    bench_archetype_registry,
);

// ============================================================================
// K-means Clustering Benchmarks
// ============================================================================

fn generate_clustered_data(
    n_vectors: usize,
    n_clusters: usize,
    dim: usize,
    noise: f32,
) -> Vec<Vec<f32>> {
    let mut vectors = Vec::with_capacity(n_vectors);
    let per_cluster = n_vectors / n_clusters;

    for cluster_id in 0..n_clusters {
        // Generate cluster center
        let center: Vec<f32> = (0..dim)
            .map(|d| ((cluster_id * 100 + d) as f32 * 0.037).sin())
            .collect();

        for vec_id in 0..per_cluster {
            let vec: Vec<f32> = center
                .iter()
                .enumerate()
                .map(|(d, &c)| {
                    let n = ((cluster_id * 1000 + vec_id * 10 + d) as f32 * 0.0013).sin() * noise;
                    c + n
                })
                .collect();
            vectors.push(vec);
        }
    }

    vectors
}

fn bench_kmeans_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans/fit");

    // Vary number of vectors
    for n in [100, 500, 1000] {
        let dim = 128;
        let data = generate_clustered_data(n, 5, dim, 0.1);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("n_vectors", n), &data, |bench, data| {
            bench.iter(|| {
                let config = KMeansConfig::default();
                let kmeans = KMeans::new(config);
                black_box(kmeans.fit(data, 5))
            });
        });
    }

    // Vary k (number of clusters)
    let data_1000 = generate_clustered_data(1000, 10, 128, 0.1);
    for k in [2, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::new("k_clusters", k),
            &(&data_1000, k),
            |bench, (data, k)| {
                bench.iter(|| {
                    let config = KMeansConfig::default();
                    let kmeans = KMeans::new(config);
                    black_box(kmeans.fit(data, *k))
                });
            },
        );
    }

    // Vary dimension
    for dim in [64, 128, 384, 768] {
        let data = generate_clustered_data(500, 5, dim, 0.1);

        group.bench_with_input(BenchmarkId::new("dimension", dim), &data, |bench, data| {
            bench.iter(|| {
                let config = KMeansConfig::default();
                let kmeans = KMeans::new(config);
                black_box(kmeans.fit(data, 5))
            });
        });
    }

    group.finish();
}

fn bench_kmeans_init_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans/initialization");

    let data = generate_clustered_data(1000, 10, 128, 0.1);

    // Random initialization
    group.bench_with_input(
        BenchmarkId::new("method", "random"),
        &data,
        |bench, data| {
            bench.iter(|| {
                let config = KMeansConfig {
                    init_method: KMeansInit::Random,
                    max_iterations: 50,
                    ..Default::default()
                };
                let kmeans = KMeans::new(config);
                black_box(kmeans.fit(data, 10))
            });
        },
    );

    // K-means++ initialization
    group.bench_with_input(
        BenchmarkId::new("method", "kmeans++"),
        &data,
        |bench, data| {
            bench.iter(|| {
                let config = KMeansConfig {
                    init_method: KMeansInit::KMeansPlusPlus,
                    max_iterations: 50,
                    ..Default::default()
                };
                let kmeans = KMeans::new(config);
                black_box(kmeans.fit(data, 10))
            });
        },
    );

    group.finish();
}

fn bench_discover_archetypes(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans/discover_archetypes");

    for n in [100, 500, 1000] {
        let data = generate_clustered_data(n, 5, 128, 0.1);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("n_vectors", n), &data, |bench, data| {
            bench.iter(|| {
                let mut registry = ArchetypeRegistry::new(16);
                let added = registry.discover_archetypes(data, 5, KMeansConfig::default());
                black_box(added)
            });
        });
    }

    group.finish();
}

fn bench_encode_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans/encode_batch");

    // Pre-discover archetypes
    let data = generate_clustered_data(1000, 5, 128, 0.1);
    let mut registry = ArchetypeRegistry::new(16);
    registry.discover_archetypes(&data, 5, KMeansConfig::default());

    for n in [100, 500, 1000] {
        let batch: Vec<Vec<f32>> = data.iter().take(n).cloned().collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_size", n),
            &(&registry, &batch),
            |bench, (reg, batch)| {
                bench.iter(|| black_box(reg.encode_batch(batch, 0.01)));
            },
        );
    }

    group.finish();
}

fn bench_analyze_coverage(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans/analyze_coverage");

    let data = generate_clustered_data(1000, 5, 128, 0.1);
    let mut registry = ArchetypeRegistry::new(16);
    registry.discover_archetypes(&data, 5, KMeansConfig::default());

    group.throughput(Throughput::Elements(1000));
    group.bench_function("1000_vectors", |bench| {
        bench.iter(|| black_box(registry.analyze_coverage(&data, 0.01)));
    });

    group.finish();
}

fn bench_full_clustering_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans/full_pipeline");

    // End-to-end: discover archetypes then encode all vectors
    for n in [500, 1000] {
        let data = generate_clustered_data(n, 5, 128, 0.1);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("discover_and_encode", n),
            &data,
            |bench, data| {
                bench.iter(|| {
                    let mut registry = ArchetypeRegistry::new(16);
                    registry.discover_archetypes(data, 5, KMeansConfig::default());
                    let encoded = registry.encode_batch(data, 0.01);
                    black_box(encoded)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    kmeans_benches,
    bench_kmeans_fit,
    bench_kmeans_init_methods,
    bench_discover_archetypes,
    bench_encode_batch,
    bench_analyze_coverage,
    bench_full_clustering_pipeline,
);

// ============================================================================
// Tiered Store Benchmarks
// ============================================================================

fn setup_tiered_test_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(format!("/tmp/tiered_bench_{}", name));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn bench_tiered_vs_inmemory_put(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered/put");

    for size in [1000, 10000].iter() {
        // Benchmark pure in-memory TensorStore
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("inmemory", size), size, |b, &size| {
            b.iter(|| {
                let store = TensorStore::new();
                for i in 0..size {
                    store
                        .put(format!("key:{}", i), create_test_tensor(i as i64))
                        .unwrap();
                }
                black_box(&store);
            });
        });

        // Benchmark TieredStore (hot tier only)
        let dir = setup_tiered_test_dir(&format!("put_{}", size));
        group.bench_with_input(BenchmarkId::new("tiered_hot", size), size, |b, &size| {
            b.iter(|| {
                let config = TieredConfig {
                    cold_dir: dir.clone(),
                    cold_capacity: 64 * 1024 * 1024,
                    sample_rate: 100,
                };
                let mut store = TieredStore::new(config).unwrap();
                for i in 0..size {
                    store.put(format!("key:{}", i), create_test_tensor(i as i64));
                }
                black_box(&store);
            });
        });
        let _ = std::fs::remove_dir_all(&dir);
    }

    group.finish();
}

fn bench_tiered_vs_inmemory_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered/get");

    for size in [1000, 10000].iter() {
        // Setup in-memory store
        let inmem_store = TensorStore::new();
        for i in 0..*size {
            inmem_store
                .put(format!("key:{}", i), create_test_tensor(i as i64))
                .unwrap();
        }

        // Setup tiered store
        let dir = setup_tiered_test_dir(&format!("get_{}", size));
        let config = TieredConfig {
            cold_dir: dir.clone(),
            cold_capacity: 64 * 1024 * 1024,
            sample_rate: 100,
        };
        let mut tiered_store = TieredStore::new(config).unwrap();
        for i in 0..*size {
            tiered_store.put(format!("key:{}", i), create_test_tensor(i as i64));
        }

        group.throughput(Throughput::Elements(*size as u64));

        // Benchmark in-memory get
        group.bench_with_input(BenchmarkId::new("inmemory", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(inmem_store.get(&format!("key:{}", i)).unwrap());
                }
            });
        });

        // Benchmark tiered get (all hot)
        group.bench_with_input(BenchmarkId::new("tiered_hot", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(tiered_store.get(&format!("key:{}", i)).unwrap());
                }
            });
        });

        let _ = std::fs::remove_dir_all(&dir);
    }

    group.finish();
}

fn bench_tiered_cold_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered/cold_access");
    let size = 1000;

    let dir = setup_tiered_test_dir("cold_access");
    let config = TieredConfig {
        cold_dir: dir.clone(),
        cold_capacity: 64 * 1024 * 1024,
        sample_rate: 1, // Track every access
    };
    let mut store = TieredStore::new(config).unwrap();

    // Insert data
    for i in 0..size {
        store.put(format!("key:{}", i), create_test_tensor(i as i64));
    }

    // Access first 100 to make them hot
    for i in 0..100 {
        let _ = store.get(&format!("key:{}", i));
    }

    // Wait for cold threshold
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Migrate cold data
    let migrated = store.migrate_cold(10).unwrap();

    group.throughput(Throughput::Elements(100));

    // Benchmark cold reads (with promotion)
    if migrated > 0 {
        group.bench_function("cold_read_with_promotion", |b| {
            b.iter(|| {
                // Read some cold keys (they get promoted)
                for i in 100..200 {
                    if store.exists(&format!("key:{}", i)) {
                        black_box(store.get(&format!("key:{}", i)).ok());
                    }
                }
            });
        });
    }

    // Benchmark hot reads (after promotion)
    group.bench_function("hot_read_after_promotion", |b| {
        b.iter(|| {
            for i in 0..100 {
                black_box(store.get(&format!("key:{}", i)).unwrap());
            }
        });
    });

    let _ = std::fs::remove_dir_all(&dir);
    group.finish();
}

fn bench_tiered_migration(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered/migration");

    for size in [1000, 5000].iter() {
        let dir = setup_tiered_test_dir(&format!("migration_{}", size));

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("migrate_all", size), size, |b, &size| {
            b.iter_with_setup(
                || {
                    // Setup: create store with data
                    let config = TieredConfig {
                        cold_dir: dir.clone(),
                        cold_capacity: 64 * 1024 * 1024,
                        sample_rate: 1,
                    };
                    let mut store = TieredStore::new(config).unwrap();
                    for i in 0..size {
                        store.put(format!("key:{}", i), create_test_tensor(i as i64));
                    }
                    // Wait for all to be "cold"
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    store
                },
                |mut store| {
                    // Benchmark: migrate all cold data
                    let migrated = store.migrate_cold(1).unwrap();
                    black_box(migrated);
                },
            );
        });

        let _ = std::fs::remove_dir_all(&dir);
    }

    group.finish();
}

fn bench_tiered_preload(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiered/preload");

    // We can't easily benchmark preload without accessing private fields
    // Instead, benchmark cold-to-hot promotion via get()
    let dir = setup_tiered_test_dir("preload");

    group.throughput(Throughput::Elements(100));
    group.bench_function("cold_promotion_via_get", |b| {
        b.iter_with_setup(
            || {
                // Setup: create store, insert data, migrate to cold
                let config = TieredConfig {
                    cold_dir: dir.clone(),
                    cold_capacity: 64 * 1024 * 1024,
                    sample_rate: 1,
                };
                let mut store = TieredStore::new(config).unwrap();
                for i in 0..100 {
                    store.put(format!("key:{}", i), create_test_tensor(i as i64));
                }
                // Wait then migrate all to cold
                std::thread::sleep(std::time::Duration::from_millis(5));
                let _ = store.migrate_cold(1);
                store
            },
            |mut store| {
                // Benchmark: read all cold keys (promotes them)
                for i in 0..100 {
                    black_box(store.get(&format!("key:{}", i)).ok());
                }
            },
        );
    });

    let _ = std::fs::remove_dir_all(&dir);
    group.finish();
}

criterion_group!(
    tiered_benches,
    bench_tiered_vs_inmemory_put,
    bench_tiered_vs_inmemory_get,
    bench_tiered_cold_access,
    bench_tiered_migration,
    bench_tiered_preload,
);

criterion_main!(
    benches,
    sparse_vector_benches,
    delta_vector_benches,
    kmeans_benches,
    tiered_benches
);
