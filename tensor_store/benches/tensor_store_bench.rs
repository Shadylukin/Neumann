use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::sync::Arc;
use std::thread;
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

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

criterion_group!(
    benches,
    bench_put,
    bench_get,
    bench_scan,
    bench_concurrent_writes,
    bench_concurrent_writes_contention,
    bench_mixed_read_write,
);

criterion_main!(benches);
