// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![allow(missing_docs)]

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor_checkpoint::{CheckpointConfig, CheckpointManager, FileCheckpointStore};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

fn setup_manager(max_checkpoints: usize) -> (CheckpointManager, TensorStore, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let store = TensorStore::new();
    let file_store = Arc::new(FileCheckpointStore::new(dir.path()).unwrap());
    let config = CheckpointConfig::new().with_max_checkpoints(max_checkpoints);
    let manager = CheckpointManager::new(file_store, config);
    (manager, store, dir)
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
    let mut group = c.benchmark_group("checkpoint_create");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for key_count in [0, 100, 1_000] {
        group.bench_with_input(
            BenchmarkId::new("keys", key_count),
            &key_count,
            |b, &key_count| {
                b.iter_batched(
                    || {
                        let (manager, store, dir) = setup_manager(100);
                        populate_store(&store, key_count);
                        (manager, store, dir)
                    },
                    |(manager, store, _dir)| {
                        black_box(manager.create(None, &store).unwrap());
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_checkpoint_rollback(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_rollback");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));

    for key_count in [100, 1_000] {
        group.bench_with_input(
            BenchmarkId::new("keys", key_count),
            &key_count,
            |b, &key_count| {
                b.iter_batched(
                    || {
                        let (manager, store, dir) = setup_manager(100);
                        populate_store(&store, key_count);
                        let id = manager.create(None, &store).unwrap();
                        // Mutate store after checkpoint
                        for i in 0..100 {
                            store
                                .put(format!("extra_{i}"), make_tensor("n", &format!("{i}")))
                                .unwrap();
                        }
                        (manager, store, id, dir)
                    },
                    |(manager, store, id, _dir)| {
                        black_box(manager.rollback(&id, &store).unwrap());
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

fn bench_checkpoint_list(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_list");

    for checkpoint_count in [3, 10] {
        let (manager, store, _dir) = setup_manager(100);
        populate_store(&store, 100);
        for i in 0..checkpoint_count {
            manager.create(Some(&format!("cp_{i}")), &store).unwrap();
        }

        group.bench_with_input(
            BenchmarkId::new("count", checkpoint_count),
            &checkpoint_count,
            |b, _| {
                b.iter(|| {
                    black_box(manager.list(None).unwrap());
                });
            },
        );
    }

    group.finish();
}

fn bench_checkpoint_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_metadata");

    let (manager, store, _dir) = setup_manager(100);
    populate_store(&store, 100);
    let id = manager.create(Some("meta_bench"), &store).unwrap();

    group.bench_function("list_single", |b| {
        b.iter(|| {
            let list = manager.list(Some(1)).unwrap();
            black_box(&list);
            assert_eq!(list[0].id, id);
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
    bench_checkpoint_metadata,
);

criterion_main!(benches);
