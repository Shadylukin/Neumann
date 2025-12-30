//! Benchmarks for tensor_cache.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use tensor_cache::{Cache, CacheConfig, DistanceMetric, SparseVector};

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

fn create_sparse_vector(dim: usize, sparsity: f32, seed: usize) -> Vec<f32> {
    let non_zero_count = ((1.0 - sparsity) * dim as f32) as usize;
    let mut v = vec![0.0; dim];
    for i in 0..non_zero_count {
        let idx = (seed * 31 + i * 17) % dim;
        v[idx] = ((seed * 13 + i * 7) as f32 * 0.001).sin().abs() + 0.1;
    }
    normalize(&v)
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

fn bench_distance_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_metrics");

    // Create test vectors
    let dim = 128;
    let v1 = normalize(&create_test_vector(dim, 1));
    let v2 = normalize(&create_test_vector(dim, 2));
    let sv1 = SparseVector::from_dense(&v1);
    let sv2 = SparseVector::from_dense(&v2);

    group.bench_function("cosine_similarity", |b| {
        b.iter(|| black_box(sv1.cosine_similarity(&sv2)))
    });

    group.bench_function("jaccard_index", |b| {
        b.iter(|| black_box(sv1.jaccard_index(&sv2)))
    });

    group.bench_function("euclidean_distance", |b| {
        b.iter(|| black_box(sv1.euclidean_distance(&sv2)))
    });

    group.bench_function("angular_distance", |b| {
        b.iter(|| black_box(sv1.angular_distance(&sv2)))
    });

    group.finish();
}

fn bench_semantic_with_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_metrics");

    let metrics = [
        ("cosine", DistanceMetric::Cosine),
        ("jaccard", DistanceMetric::Jaccard),
        ("euclidean", DistanceMetric::Euclidean),
    ];

    for (name, metric) in metrics.iter() {
        group.bench_function(format!("lookup_{}", name), |b| {
            let mut config = CacheConfig::default();
            config.embedding_dim = 128;
            config.semantic_threshold = 0.3; // Lower for non-cosine metrics
            config.distance_metric = metric.clone();
            config.auto_select_metric = false;
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

            let query = normalize(&create_test_vector(128, 500));

            b.iter(|| black_box(cache.get_with_metric("different", Some(&query), Some(metric))))
        });
    }

    group.finish();
}

fn bench_sparse_vs_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vs_dense");

    // Dense embeddings (10% zeros)
    group.bench_function("dense_lookup", |b| {
        let mut config = CacheConfig::default();
        config.embedding_dim = 128;
        config.auto_select_metric = true;
        let cache = Cache::with_config(config);

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

        let query = normalize(&create_test_vector(128, 500));
        b.iter(|| black_box(cache.get("different", Some(&query))))
    });

    // Sparse embeddings (80% zeros)
    group.bench_function("sparse_lookup", |b| {
        let mut config = CacheConfig::sparse_embeddings();
        config.embedding_dim = 128;
        config.semantic_threshold = 0.3;
        let cache = Cache::with_config(config);

        for i in 0..1000 {
            let embedding = create_sparse_vector(128, 0.8, i);
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

        let query = create_sparse_vector(128, 0.8, 500);
        b.iter(|| black_box(cache.get("different", Some(&query))))
    });

    group.finish();
}

fn bench_auto_metric_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_selection");

    group.bench_function("sparsity_check", |b| {
        let v = normalize(&create_test_vector(128, 1));
        let sv = SparseVector::from_dense(&v);
        b.iter(|| black_box(sv.sparsity()))
    });

    group.bench_function("auto_select_dense", |b| {
        let mut config = CacheConfig::default();
        config.embedding_dim = 128;
        config.auto_select_metric = true;
        let cache = Cache::with_config(config);

        for i in 0..100 {
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

        let query = normalize(&create_test_vector(128, 50));
        b.iter(|| black_box(cache.get("different", Some(&query))))
    });

    group.bench_function("auto_select_sparse", |b| {
        let mut config = CacheConfig::default();
        config.embedding_dim = 128;
        config.auto_select_metric = true;
        config.semantic_threshold = 0.3;
        let cache = Cache::with_config(config);

        for i in 0..100 {
            let embedding = create_sparse_vector(128, 0.8, i);
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

        let query = create_sparse_vector(128, 0.8, 50);
        b.iter(|| black_box(cache.get("different", Some(&query))))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_exact_lookup,
    bench_semantic_lookup,
    bench_put,
    bench_embedding_cache,
    bench_eviction,
    bench_distance_metrics,
    bench_semantic_with_metrics,
    bench_sparse_vs_dense,
    bench_auto_metric_selection,
);
criterion_main!(benches);
