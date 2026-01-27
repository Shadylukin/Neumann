//! HNSW concurrent operation benchmarks.
//!
//! Measures throughput and latency of concurrent insert and search operations.

use std::sync::{Arc, Barrier};
use std::thread;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensor_store::HNSWIndex;

fn generate_random_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut vec = Vec::with_capacity(dim);
    let mut state = seed;
    for _ in 0..dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        vec.push(val);
    }
    vec
}

fn bench_hnsw_concurrent_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_concurrent/insert");
    let dim = 128;
    let vectors_per_thread = 500;

    for thread_count in [2, 4, 8, 16, 32] {
        let total_vectors = thread_count * vectors_per_thread;
        group.throughput(Throughput::Elements(total_vectors as u64));

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for iter in 0..iters {
                        let index = Arc::new(HNSWIndex::new());
                        let barrier = Arc::new(Barrier::new(threads));

                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let idx = Arc::clone(&index);
                                let bar = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    // Pre-generate vectors
                                    let vectors: Vec<Vec<f32>> = (0..vectors_per_thread)
                                        .map(|i| {
                                            generate_random_vector(
                                                dim,
                                                iter * 1000 + t as u64 * 100 + i as u64,
                                            )
                                        })
                                        .collect();

                                    bar.wait();

                                    for v in vectors {
                                        idx.insert(v);
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                        black_box(&index);
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_hnsw_concurrent_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_concurrent/search");
    let dim = 128;
    let index_size = 5000;
    let searches_per_thread = 100;

    // Pre-build index
    let index = Arc::new(HNSWIndex::new());
    for i in 0..index_size {
        index.insert(generate_random_vector(dim, i as u64));
    }

    for thread_count in [2, 4, 8, 16, 32] {
        let total_searches = thread_count * searches_per_thread;
        group.throughput(Throughput::Elements(total_searches as u64));

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for iter in 0..iters {
                        let barrier = Arc::new(Barrier::new(threads));

                        let start = std::time::Instant::now();

                        let handles: Vec<_> = (0..threads)
                            .map(|t| {
                                let idx = Arc::clone(&index);
                                let bar = Arc::clone(&barrier);
                                thread::spawn(move || {
                                    // Pre-generate queries
                                    let queries: Vec<Vec<f32>> = (0..searches_per_thread)
                                        .map(|i| {
                                            generate_random_vector(
                                                dim,
                                                10000 + iter * 1000 + t as u64 * 100 + i as u64,
                                            )
                                        })
                                        .collect();

                                    bar.wait();

                                    for query in &queries {
                                        black_box(idx.search(query, 10));
                                    }
                                })
                            })
                            .collect();

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

fn bench_hnsw_concurrent_mixed(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_concurrent/mixed");
    let dim = 128;
    let initial_size = 2000;
    let inserts_per_thread = 200;
    let searches_per_thread = 200;

    for thread_count in [4, 8, 16, 32] {
        let inserters = thread_count / 2;
        let searchers = thread_count / 2;
        let total_ops = inserters * inserts_per_thread + searchers * searches_per_thread;
        group.throughput(Throughput::Elements(total_ops as u64));

        group.bench_with_input(
            BenchmarkId::new("threads", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_custom(|iters| {
                    let mut total_duration = std::time::Duration::ZERO;

                    for iter in 0..iters {
                        let index = Arc::new(HNSWIndex::new());

                        // Pre-populate
                        for i in 0..initial_size {
                            index.insert(generate_random_vector(dim, i as u64));
                        }

                        let barrier = Arc::new(Barrier::new(threads));
                        let start = std::time::Instant::now();

                        let mut handles = vec![];

                        // Inserter threads (first half)
                        for t in 0..(threads / 2) {
                            let idx = Arc::clone(&index);
                            let bar = Arc::clone(&barrier);
                            handles.push(thread::spawn(move || {
                                let vectors: Vec<Vec<f32>> = (0..inserts_per_thread)
                                    .map(|i| {
                                        generate_random_vector(
                                            dim,
                                            100000 + iter * 10000 + t as u64 * 1000 + i as u64,
                                        )
                                    })
                                    .collect();

                                bar.wait();

                                for v in vectors {
                                    idx.insert(v);
                                }
                            }));
                        }

                        // Searcher threads (second half)
                        for t in 0..(threads / 2) {
                            let idx = Arc::clone(&index);
                            let bar = Arc::clone(&barrier);
                            handles.push(thread::spawn(move || {
                                let queries: Vec<Vec<f32>> = (0..searches_per_thread)
                                    .map(|i| {
                                        generate_random_vector(
                                            dim,
                                            200000 + iter * 10000 + t as u64 * 1000 + i as u64,
                                        )
                                    })
                                    .collect();

                                bar.wait();

                                for query in &queries {
                                    black_box(idx.search(query, 10));
                                }
                            }));
                        }

                        for h in handles {
                            h.join().unwrap();
                        }

                        total_duration += start.elapsed();
                        black_box(&index);
                    }

                    total_duration
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    hnsw_concurrent_benches,
    bench_hnsw_concurrent_insert,
    bench_hnsw_concurrent_search,
    bench_hnsw_concurrent_mixed,
);

criterion_main!(hnsw_concurrent_benches);
