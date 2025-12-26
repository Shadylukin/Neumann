//! Benchmarks for tensor_cache.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use tensor_cache::{Cache, CacheConfig};

fn create_test_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let x = (seed * 31 + i * 17) as f32;
            (x * 0.0001).sin()
        })
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag == 0.0 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / mag).collect()
    }
}

fn bench_exact_lookup(c: &mut Criterion) {
    let mut config = CacheConfig::default();
    config.embedding_dim = 128;
    let cache = Cache::with_config(config);

    // Pre-populate with 1000 entries
    for i in 0..1000 {
        let embedding = normalize(&create_test_vector(128, i));
        cache
            .put(
                &format!("prompt {}", i),
                &embedding,
                &format!("response {}", i),
                "gpt-4",
                i as u64,
            )
            .unwrap();
    }

    c.bench_function("exact_lookup_hit", |b| {
        b.iter(|| black_box(cache.get("prompt 500", None)))
    });

    c.bench_function("exact_lookup_miss", |b| {
        b.iter(|| black_box(cache.get("nonexistent prompt", None)))
    });
}

fn bench_semantic_lookup(c: &mut Criterion) {
    let mut config = CacheConfig::default();
    config.embedding_dim = 128;
    config.semantic_threshold = 0.8;
    let cache = Cache::with_config(config);

    // Pre-populate
    for i in 0..1000 {
        let embedding = normalize(&create_test_vector(128, i));
        cache
            .put(
                &format!("prompt {}", i),
                &embedding,
                &format!("response {}", i),
                "gpt-4",
                i as u64,
            )
            .unwrap();
    }

    let query_embedding = normalize(&create_test_vector(128, 500));

    c.bench_function("semantic_lookup_hit", |b| {
        b.iter(|| black_box(cache.get("different prompt", Some(&query_embedding))))
    });
}

fn bench_put(c: &mut Criterion) {
    let mut config = CacheConfig::default();
    config.embedding_dim = 128;
    config.exact_capacity = 100_000;
    config.semantic_capacity = 100_000;

    let mut group = c.benchmark_group("put");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let cache = Cache::with_config(config.clone());

            // Pre-populate
            for i in 0..size {
                let embedding = normalize(&create_test_vector(128, i));
                cache
                    .put(
                        &format!("prompt {}", i),
                        &embedding,
                        &format!("response {}", i),
                        "gpt-4",
                        i as u64,
                    )
                    .unwrap();
            }

            let new_embedding = normalize(&create_test_vector(128, size + 1));

            b.iter(|| {
                let _ = cache.put(
                    "new prompt",
                    &new_embedding,
                    "new response",
                    "gpt-4",
                    (size + 1) as u64,
                );
            })
        });
    }

    group.finish();
}

fn bench_embedding_cache(c: &mut Criterion) {
    let config = CacheConfig::default();
    let cache = Cache::with_config(config);

    // Pre-populate
    for i in 0..1000 {
        cache
            .put_embedding(
                "doc",
                &format!("content {}", i),
                create_test_vector(1536, i),
                "text-embedding-3-small",
            )
            .unwrap();
    }

    c.bench_function("embedding_lookup_hit", |b| {
        b.iter(|| black_box(cache.get_embedding("doc", "content 500")))
    });

    c.bench_function("embedding_lookup_miss", |b| {
        b.iter(|| black_box(cache.get_embedding("doc", "nonexistent")))
    });
}

fn bench_eviction(c: &mut Criterion) {
    let mut group = c.benchmark_group("eviction");
    group.measurement_time(Duration::from_secs(10));

    for size in [1000, 5000, 10000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut config = CacheConfig::default();
            config.embedding_dim = 64;
            config.exact_capacity = size * 2;
            config.semantic_capacity = size * 2;
            let cache = Cache::with_config(config);

            // Pre-populate
            for i in 0..size {
                let embedding = normalize(&create_test_vector(64, i));
                cache
                    .put(
                        &format!("prompt {}", i),
                        &embedding,
                        &format!("response {}", i),
                        "gpt-4",
                        i as u64,
                    )
                    .unwrap();
            }

            b.iter(|| black_box(cache.evict(100)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_exact_lookup,
    bench_semantic_lookup,
    bench_put,
    bench_embedding_cache,
    bench_eviction,
);
criterion_main!(benches);
