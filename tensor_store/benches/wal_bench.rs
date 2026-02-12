// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tempfile::tempdir;
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue, WalConfig};

fn create_test_data(id: i64) -> TensorData {
    let mut data = TensorData::new();
    data.set("id", TensorValue::Scalar(ScalarValue::Int(id)));
    data.set(
        "name",
        TensorValue::Scalar(ScalarValue::String(format!("entity_{}", id))),
    );
    data.set(
        "embedding",
        TensorValue::Vector(vec![id as f32; 128]), // 128-dim embedding (same as tensor_store_bench)
    );
    data
}

fn bench_wal_write_immediate(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal_immediate");

    // Only test 100 for immediate mode (it's slow)
    let size = 100;
    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("write_100", |b| {
        b.iter_with_setup(
            || {
                let dir = tempdir().unwrap();
                let wal_path = dir.path().join("bench.wal");
                let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();
                (dir, store)
            },
            |(_dir, store)| {
                for i in 0..size {
                    store
                        .put_durable(format!("key_{}", i), create_test_data(i))
                        .unwrap();
                }
            },
        );
    });
    group.finish();
}

fn bench_wal_write_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal_batched");

    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("write_{}", size), |b| {
            b.iter_with_setup(
                || {
                    let dir = tempdir().unwrap();
                    let wal_path = dir.path().join("bench.wal");
                    // Batched mode: fsync every 100 entries
                    let config = WalConfig::batched(100);
                    let store = TensorStore::open_durable(&wal_path, config).unwrap();
                    (dir, store)
                },
                |(_dir, store)| {
                    for i in 0..size {
                        store
                            .put_durable(format!("key_{}", i), create_test_data(i))
                            .unwrap();
                    }
                    // Final sync for any remaining entries
                    store.sync().unwrap();
                },
            );
        });
    }
    group.finish();
}

fn bench_wal_recovery(c: &mut Criterion) {
    let mut group = c.benchmark_group("wal_recovery");

    for size in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(format!("recover_{}", size), |b| {
            let dir = tempdir().unwrap();
            let wal_path = dir.path().join("bench.wal");
            {
                let store = TensorStore::open_durable(&wal_path, WalConfig::default()).unwrap();
                for i in 0..size {
                    store
                        .put(format!("key_{}", i), create_test_data(i))
                        .unwrap();
                }
            }

            b.iter(|| {
                let recovered = TensorStore::recover(
                    black_box(&wal_path),
                    black_box(&WalConfig::default()),
                    black_box(None::<&std::path::Path>),
                )
                .unwrap();
                black_box(recovered)
            });
        });
    }
    group.finish();
}

criterion_group!(
    wal_benches,
    bench_wal_write_immediate,
    bench_wal_write_batched,
    bench_wal_recovery
);
criterion_main!(wal_benches);
