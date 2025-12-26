use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::thread;
use tensor_store::{BloomFilter, ScalarValue, SparseVector, TensorData, TensorStore, TensorValue};

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

criterion_main!(benches, sparse_vector_benches);
