// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Benchmarks for `vector_engine` performance testing.
//!
//! Measures embedding storage, similarity search, and HNSW index operations.

#![allow(missing_docs)]

use std::sync::atomic::{AtomicUsize, Ordering};

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::Rng;
use vector_engine::{HNSWConfig, HNSWIndex, VectorEngine};

fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn bench_store_embedding(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_embedding");

    for dim in [128, 768, 1536] {
        let counter = AtomicUsize::new(0);
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, &dim| {
            let engine = VectorEngine::new();
            let vector = random_vector(dim);
            b.iter(|| {
                let i = counter.fetch_add(1, Ordering::Relaxed);
                engine
                    .store_embedding(&format!("key{i}"), black_box(vector.clone()))
                    .unwrap();
            });
        });
    }

    group.finish();
}

fn bench_search_similar(c: &mut Criterion) {
    let mut group = c.benchmark_group("search_similar");

    for (count, dim) in [(1000, 128), (1000, 768), (10000, 128)] {
        let engine = VectorEngine::new();
        for i in 0..count {
            engine
                .store_embedding(&format!("v{i}"), random_vector(dim))
                .unwrap();
        }

        let query = random_vector(dim);
        let label = format!("{count}x{dim}");

        group.bench_with_input(BenchmarkId::new("top10", &label), &query, |b, query| {
            b.iter(|| engine.search_similar(black_box(query), 10).unwrap());
        });
    }

    group.finish();
}

fn bench_compute_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_similarity");

    for dim in [128, 768, 1536] {
        let a = random_vector(dim);
        let b = random_vector(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &(a, b), |bench, (a, b)| {
            bench.iter(|| VectorEngine::compute_similarity(black_box(a), black_box(b)).unwrap());
        });
    }

    group.finish();
}

fn bench_get_embedding(c: &mut Criterion) {
    let engine = VectorEngine::new();
    for i in 0..1000 {
        engine
            .store_embedding(&format!("v{i}"), random_vector(768))
            .unwrap();
    }

    c.bench_function("get_embedding_768d", |b| {
        b.iter(|| engine.get_embedding(black_box("v500")).unwrap());
    });
}

fn bench_delete_embedding(c: &mut Criterion) {
    c.bench_function("delete_embedding", |b| {
        b.iter_batched(
            || {
                let engine = VectorEngine::new();
                engine.store_embedding("key", random_vector(128)).unwrap();
                engine
            },
            |engine| {
                engine.delete_embedding(black_box("key")).unwrap();
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");

    for dim in [128, 768] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |b, &dim| {
            let index = HNSWIndex::new();
            b.iter(|| {
                index.insert(black_box(random_vector(dim)));
            });
        });
    }

    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    let dim = 128;

    // Benchmark different corpus sizes
    for count in [1000, 10000] {
        let index = HNSWIndex::new();
        for _ in 0..count {
            index.insert(random_vector(dim));
        }

        let query = random_vector(dim);
        let label = format!("{count}x{dim}");

        group.bench_with_input(BenchmarkId::new("top10", &label), &query, |b, query| {
            b.iter(|| index.search(black_box(query), 10));
        });
    }

    group.finish();
}

fn bench_hnsw_vs_brute_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_vs_brute");
    let dim = 128;
    let count = 10000;

    // Setup brute force engine
    let engine = VectorEngine::new();
    for i in 0..count {
        engine
            .store_embedding(&format!("v{i}"), random_vector(dim))
            .unwrap();
    }

    // Setup HNSW index
    let hnsw = HNSWIndex::new();
    for _ in 0..count {
        hnsw.insert(random_vector(dim));
    }

    let query = random_vector(dim);

    group.bench_with_input(
        BenchmarkId::new("brute_force", count),
        &query,
        |b, query| {
            b.iter(|| engine.search_similar(black_box(query), 10).unwrap());
        },
    );

    group.bench_with_input(BenchmarkId::new("hnsw", count), &query, |b, query| {
        b.iter(|| hnsw.search(black_box(query), 10));
    });

    group.finish();
}

fn bench_hnsw_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_configs");
    let dim = 128;
    let count = 5000;

    // Build indices with different configs
    let index_default = HNSWIndex::with_config(HNSWConfig::default());
    let index_speed = HNSWIndex::with_config(HNSWConfig::high_speed());
    let index_recall = HNSWIndex::with_config(HNSWConfig::high_recall());

    for _ in 0..count {
        let v = random_vector(dim);
        index_default.insert(v.clone());
        index_speed.insert(v.clone());
        index_recall.insert(v);
    }

    let query = random_vector(dim);

    group.bench_with_input(BenchmarkId::new("default", count), &query, |b, query| {
        b.iter(|| index_default.search(black_box(query), 10));
    });

    group.bench_with_input(BenchmarkId::new("high_speed", count), &query, |b, query| {
        b.iter(|| index_speed.search(black_box(query), 10));
    });

    group.bench_with_input(
        BenchmarkId::new("high_recall", count),
        &query,
        |b, query| {
            b.iter(|| index_recall.search(black_box(query), 10));
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_store_embedding,
    bench_search_similar,
    bench_compute_similarity,
    bench_get_embedding,
    bench_delete_embedding,
    bench_hnsw_insert,
    bench_hnsw_search,
    bench_hnsw_vs_brute_force,
    bench_hnsw_configs,
);
criterion_main!(benches);
