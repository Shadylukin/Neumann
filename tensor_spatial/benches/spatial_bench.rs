//! Benchmarks for `tensor_spatial` R-tree operations.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::Rng;
use tensor_spatial::{BoundingBox, SpatialConfig, SpatialEntry, SpatialIndex, SplitStrategy};

fn random_entries(n: usize) -> Vec<SpatialEntry<u32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|i| {
            let x = rng.gen_range(0.0..1000.0_f32);
            let y = rng.gen_range(0.0..1000.0_f32);
            SpatialEntry {
                bounds: BoundingBox::new(x, y, 1.0, 1.0).unwrap(),
                data: i as u32,
            }
        })
        .collect()
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for &n in &[1_000, 10_000, 100_000] {
        let entries = random_entries(n);
        group.bench_with_input(BenchmarkId::new("rstar", n), &entries, |b, entries| {
            b.iter_batched(
                || entries.clone(),
                |entries| {
                    let mut index = SpatialIndex::new();
                    for entry in entries {
                        index.insert(entry);
                    }
                    black_box(index.len());
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn bench_bulk_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_load");

    for &n in &[1_000, 10_000, 100_000] {
        let entries = random_entries(n);
        group.bench_with_input(BenchmarkId::new("str", n), &entries, |b, entries| {
            b.iter_batched(
                || entries.clone(),
                |entries| {
                    let index = SpatialIndex::bulk_load(entries);
                    black_box(index.len());
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

fn bench_region_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("region_query");

    for &n in &[10_000, 100_000] {
        let index = SpatialIndex::bulk_load(random_entries(n));
        // ~5% of the 1000x1000 area
        let region = BoundingBox::new(400.0, 400.0, 224.0, 224.0).unwrap();
        group.bench_with_input(BenchmarkId::new("5pct_area", n), &index, |b, index| {
            b.iter(|| {
                let results = index.query_region(black_box(region));
                black_box(results.len());
            });
        });
    }

    group.finish();
}

fn bench_nearest_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_query");

    for &n in &[10_000, 100_000] {
        let index = SpatialIndex::bulk_load(random_entries(n));
        group.bench_with_input(BenchmarkId::new("k10", n), &index, |b, index| {
            b.iter(|| {
                let results = index.query_nearest(black_box(500.0), black_box(500.0), 10);
                black_box(results.len());
            });
        });
    }

    group.finish();
}

fn bench_radius_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("radius_query");

    for &n in &[10_000, 100_000] {
        let index = SpatialIndex::bulk_load(random_entries(n));
        group.bench_with_input(BenchmarkId::new("r50", n), &index, |b, index| {
            b.iter(|| {
                let results =
                    index.query_within_radius(black_box(500.0), black_box(500.0), black_box(50.0));
                black_box(results.len());
            });
        });
    }

    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");

    let entries = random_entries(10_000);
    let index = SpatialIndex::bulk_load(entries.clone());
    group.bench_function(BenchmarkId::new("from_10k", 100), |b| {
        b.iter_batched(
            || (index.clone(), entries[..100].to_vec()),
            |(mut idx, to_remove)| {
                for entry in &to_remove {
                    let _ = idx.remove(entry.bounds, |e| e.data == entry.data);
                }
                black_box(idx.len());
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_split_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("split_strategies");

    let entries = random_entries(10_000);
    let region = BoundingBox::new(400.0, 400.0, 200.0, 200.0).unwrap();

    for (name, strategy) in [
        ("linear", SplitStrategy::Linear),
        ("rstar", SplitStrategy::RStar),
    ] {
        let cfg = SpatialConfig::with_strategy(9, strategy).unwrap();
        group.bench_function(BenchmarkId::new("insert_query", name), |b| {
            b.iter_batched(
                || entries.clone(),
                |entries| {
                    let mut index = SpatialIndex::with_config(cfg);
                    for entry in entries {
                        index.insert(entry);
                    }
                    let results = index.query_region(black_box(region));
                    black_box(results.len());
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_bulk_load,
    bench_region_query,
    bench_nearest_query,
    bench_radius_query,
    bench_remove,
    bench_split_strategies,
);
criterion_main!(benches);
