//! Benchmarks for `tensor_spatial` R-tree operations.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor_spatial::{
    BoundingBox, BoundingBox3D, SpatialConfig, SpatialEntry, SpatialEntry3D, SpatialIndex,
    SpatialIndex3D, SplitStrategy,
};

fn random_entries(n: usize, seed: u64) -> Vec<SpatialEntry<u32>> {
    let mut rng = StdRng::seed_from_u64(seed);
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

fn random_entries_3d(n: usize) -> Vec<SpatialEntry3D<u32>> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|i| {
            let x = rng.gen_range(0.0..1000.0_f32);
            let y = rng.gen_range(0.0..1000.0_f32);
            let z = rng.gen_range(0.0..1000.0_f32);
            SpatialEntry3D {
                bounds: BoundingBox3D::new(x, y, z, 1.0, 1.0, 1.0).unwrap(),
                data: i as u32,
            }
        })
        .collect()
}

fn build_with_strategy(
    entries: &[SpatialEntry<u32>],
    strategy: SplitStrategy,
) -> SpatialIndex<u32> {
    let cfg = SpatialConfig::with_strategy(9, strategy).unwrap();
    let mut index = SpatialIndex::with_config(cfg);
    for entry in entries {
        index.insert(entry.clone());
    }
    index
}

fn is_heavy() -> bool {
    std::env::var("BENCH_HEAVY").is_ok()
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for &n in &[1_000, 10_000, 100_000] {
        let entries = random_entries(n, 42);
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

    // Linear strategy at 10K for comparison
    let entries_10k = random_entries(10_000, 42);
    let cfg = SpatialConfig::with_strategy(9, SplitStrategy::Linear).unwrap();
    group.bench_with_input(
        BenchmarkId::new("linear", 10_000),
        &entries_10k,
        |b, entries| {
            b.iter_batched(
                || entries.clone(),
                |entries| {
                    let mut index = SpatialIndex::with_config(cfg);
                    for entry in entries {
                        index.insert(entry);
                    }
                    black_box(index.len());
                },
                BatchSize::LargeInput,
            );
        },
    );

    if is_heavy() {
        let entries_1m = random_entries(1_000_000, 42);
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::new("rstar", 1_000_000),
            &entries_1m,
            |b, entries| {
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
            },
        );
    }

    group.finish();
}

fn bench_bulk_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk_load");

    for &n in &[1_000, 10_000, 100_000] {
        let entries = random_entries(n, 42);
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

    if is_heavy() {
        let entries_1m = random_entries(1_000_000, 42);
        group.sample_size(10);
        group.bench_with_input(
            BenchmarkId::new("str", 1_000_000),
            &entries_1m,
            |b, entries| {
                b.iter_batched(
                    || entries.clone(),
                    |entries| {
                        let index = SpatialIndex::bulk_load(entries);
                        black_box(index.len());
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_region_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("region_query");

    for &n in &[10_000, 100_000] {
        let index = SpatialIndex::bulk_load(random_entries(n, 42));
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
        let index = SpatialIndex::bulk_load(random_entries(n, 42));
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
        let index = SpatialIndex::bulk_load(random_entries(n, 42));
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

    let entries = random_entries(10_000, 42);
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

    let entries = random_entries(10_000, 42);
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

fn bench_query_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_strategies");

    let entries = random_entries(100_000, 42);
    let linear_index = build_with_strategy(&entries, SplitStrategy::Linear);
    let rstar_index = build_with_strategy(&entries, SplitStrategy::RStar);
    let region = BoundingBox::new(400.0, 400.0, 224.0, 224.0).unwrap();

    for (name, index) in [("linear", &linear_index), ("rstar", &rstar_index)] {
        group.bench_function(BenchmarkId::new("region_5pct", name), |b| {
            b.iter(|| {
                let results = index.query_region(black_box(region));
                black_box(results.len());
            });
        });

        group.bench_function(BenchmarkId::new("nearest_k10", name), |b| {
            b.iter(|| {
                let results = index.query_nearest(black_box(500.0), black_box(500.0), 10);
                black_box(results.len());
            });
        });

        group.bench_function(BenchmarkId::new("radius_r50", name), |b| {
            b.iter(|| {
                let results =
                    index.query_within_radius(black_box(500.0), black_box(500.0), black_box(50.0));
                black_box(results.len());
            });
        });
    }

    group.finish();
}

fn bench_nearest_k_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_k_sizes");

    let index = build_with_strategy(&random_entries(100_000, 42), SplitStrategy::RStar);

    for &k in &[1, 10, 100] {
        group.bench_function(BenchmarkId::new("rstar_100k", k), |b| {
            b.iter(|| {
                let results = index.query_nearest(black_box(500.0), black_box(500.0), k);
                black_box(results.len());
            });
        });
    }

    group.finish();
}

fn bench_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("3d");

    // 3D insert
    for &n in &[10_000, 100_000] {
        let entries = random_entries_3d(n);
        group.bench_with_input(BenchmarkId::new("insert", n), &entries, |b, entries| {
            b.iter_batched(
                || entries.clone(),
                |entries| {
                    let mut index = SpatialIndex3D::new();
                    for entry in entries {
                        index.insert(entry);
                    }
                    black_box(index.len());
                },
                BatchSize::LargeInput,
            );
        });
    }

    // 3D queries at 100K
    let entries_100k = random_entries_3d(100_000);
    let mut index = SpatialIndex3D::new();
    for entry in &entries_100k {
        index.insert(entry.clone());
    }

    let region = BoundingBox3D::new(400.0, 400.0, 400.0, 224.0, 224.0, 224.0).unwrap();
    group.bench_function(BenchmarkId::new("region", 100_000), |b| {
        b.iter(|| {
            let results = index.query_region(black_box(region));
            black_box(results.len());
        });
    });

    group.bench_function(BenchmarkId::new("nearest_k10", 100_000), |b| {
        b.iter(|| {
            let results = index.query_nearest_nd(black_box([500.0, 500.0, 500.0]), 10);
            black_box(results.len());
        });
    });

    group.bench_function(BenchmarkId::new("radius_r50", 100_000), |b| {
        b.iter(|| {
            let results =
                index.query_within_radius_nd(black_box([500.0, 500.0, 500.0]), black_box(50.0));
            black_box(results.len());
        });
    });

    group.finish();
}

fn bench_nearest_centroid(c: &mut Criterion) {
    let mut group = c.benchmark_group("nearest_centroid");

    for &n in &[10_000, 100_000] {
        let index = SpatialIndex::bulk_load(random_entries(n, 42));
        group.bench_with_input(BenchmarkId::new("k10", n), &index, |b, index| {
            b.iter(|| {
                let results =
                    index.query_nearest_by_centroid(black_box(500.0), black_box(500.0), 10);
                black_box(results.len());
            });
        });
    }

    group.finish();
}

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    group.sample_size(10);

    let entries = random_entries(10_000, 42);
    let new_entries = random_entries(5_000, 99)
        .into_iter()
        .enumerate()
        .map(|(i, mut e)| {
            e.data = (10_000 + i) as u32;
            e
        })
        .collect::<Vec<_>>();
    let region = BoundingBox::new(400.0, 400.0, 224.0, 224.0).unwrap();

    group.bench_function("insert_delete_insert_query", |b| {
        b.iter_batched(
            || (entries.clone(), new_entries.clone()),
            |(entries, new_entries)| {
                // Insert all
                let mut index = SpatialIndex::new();
                for entry in &entries {
                    index.insert(entry.clone());
                }
                // Delete 50%
                for entry in &entries[..5_000] {
                    let _ = index.remove(entry.bounds, |e| e.data == entry.data);
                }
                // Insert new batch
                for entry in new_entries {
                    index.insert(entry);
                }
                // Query
                let results = index.query_region(black_box(region));
                black_box(results.len());
            },
            BatchSize::LargeInput,
        );
    });

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
    bench_query_strategies,
    bench_nearest_k_sizes,
    bench_nearest_centroid,
    bench_3d,
    bench_mixed_workload,
);
criterion_main!(benches);
