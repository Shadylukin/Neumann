// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![allow(missing_docs)]

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor_blob::{BlobConfig, BlobStore};
use tensor_checkpoint::{CheckpointConfig, CheckpointManager};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

async fn setup_manager(max_checkpoints: usize) -> (CheckpointManager, TensorStore) {
    let store = TensorStore::new();
    let blob = BlobStore::new(store.clone(), BlobConfig::default())
        .await
        .unwrap();
    let blob = Arc::new(Mutex::new(blob));
    let config = CheckpointConfig::new().with_max_checkpoints(max_checkpoints);
    let manager = CheckpointManager::new(blob, config);
    (manager, store)
}

fn make_tensor(key: &str, value: &str) -> TensorData {
    let mut t = TensorData::new();
    t.set(
        key,
        TensorValue::Scalar(ScalarValue::String(value.to_string())),
    );
    t
}

fn populate_store(store: &TensorStore, key_count: usize) {
    for i in 0..key_count {
        store
            .put(
                format!("key_{i}"),
                make_tensor("data", &format!("value_{i}")),
            )
            .unwrap();
    }
}

fn bench_checkpoint_create(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("checkpoint_create");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for key_count in [0, 1_000, 5_000] {
        group.bench_with_input(
            BenchmarkId::new("keys", key_count),
            &key_count,
            |b, &key_count| {
                b.iter_batched(
                    || {
                        rt.block_on(async {
                            let (manager, store) = setup_manager(100).await;
                            populate_store(&store, key_count);
                            (manager, store)
                        })
                    },
                    |(manager, store)| {
                        rt.block_on(async {
                            black_box(manager.create(None, &store).await.unwrap());
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_checkpoint_rollback(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("checkpoint_rollback");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for key_count in [1_000, 5_000] {
        group.bench_with_input(
            BenchmarkId::new("keys", key_count),
            &key_count,
            |b, &key_count| {
                b.iter_batched(
                    || {
                        rt.block_on(async {
                            let (manager, store) = setup_manager(100).await;
                            populate_store(&store, key_count);
                            let id = manager.create(None, &store).await.unwrap();
                            // Mutate store after checkpoint
                            for i in 0..100 {
                                store
                                    .put(format!("extra_{i}"), make_tensor("n", &format!("{i}")))
                                    .unwrap();
                            }
                            (manager, store, id)
                        })
                    },
                    |(manager, store, id)| {
                        rt.block_on(async {
                            black_box(manager.rollback(&id, &store).await.unwrap());
                        });
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_checkpoint_list(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("checkpoint_list");

    for checkpoint_count in [5, 50] {
        let (manager, store) = rt.block_on(setup_manager(100));
        populate_store(&store, 100);

        for i in 0..checkpoint_count {
            rt.block_on(async {
                manager
                    .create(Some(&format!("cp_{i}")), &store)
                    .await
                    .unwrap();
            });
        }

        group.bench_with_input(
            BenchmarkId::new("count", checkpoint_count),
            &checkpoint_count,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        black_box(manager.list(None).await.unwrap());
                    });
                });
            },
        );
    }

    group.finish();
}

fn bench_checkpoint_retention(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("checkpoint_retention");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    group.bench_function("enforce_retain_5_of_20", |b| {
        b.iter_batched(
            || {
                rt.block_on(async {
                    let (manager, store) = setup_manager(5).await;
                    populate_store(&store, 100);
                    for i in 0..20 {
                        manager
                            .create(Some(&format!("cp_{i}")), &store)
                            .await
                            .unwrap();
                    }
                    (manager, store)
                })
            },
            |(manager, store)| {
                rt.block_on(async {
                    // Creating one more triggers retention enforcement
                    black_box(manager.create(Some("trigger"), &store).await.unwrap());
                });
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

fn bench_checkpoint_metadata(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("checkpoint_metadata");

    let (manager, store) = rt.block_on(setup_manager(100));
    populate_store(&store, 1_000);
    let id = rt.block_on(async { manager.create(Some("meta_bench"), &store).await.unwrap() });

    group.bench_function("list_single", |b| {
        b.iter(|| {
            rt.block_on(async {
                let list = manager.list(Some(1)).await.unwrap();
                black_box(&list);
                assert_eq!(list[0].id, id);
            });
        });
    });

    let _ = id;
    group.finish();
}

criterion_group!(
    benches,
    bench_checkpoint_create,
    bench_checkpoint_rollback,
    bench_checkpoint_list,
    bench_checkpoint_retention,
    bench_checkpoint_metadata,
);

criterion_main!(benches);
