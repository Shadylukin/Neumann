use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::Rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use vector_engine::VectorEngine;

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
                    .store_embedding(&format!("key{}", i), black_box(vector.clone()))
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
                .store_embedding(&format!("v{}", i), random_vector(dim))
                .unwrap();
        }

        let query = random_vector(dim);
        let label = format!("{}x{}", count, dim);

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
            .store_embedding(&format!("v{}", i), random_vector(768))
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

criterion_group!(
    benches,
    bench_store_embedding,
    bench_search_similar,
    bench_compute_similarity,
    bench_get_embedding,
    bench_delete_embedding,
);
criterion_main!(benches);
