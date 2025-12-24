use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::thread;
use tensor_store::{BloomFilter, ScalarValue, TensorData, TensorStore, TensorValue};

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
);

criterion_main!(benches);
