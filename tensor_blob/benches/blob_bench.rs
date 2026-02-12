// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![allow(missing_docs)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor_blob::{BlobConfig, BlobStore, PutOptions};
use tensor_store::TensorStore;
use tokio::runtime::Runtime;

fn bench_put_small(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let store = TensorStore::new();
    let blob_store =
        rt.block_on(async { BlobStore::new(store, BlobConfig::default()).await.unwrap() });

    let data = vec![0u8; 1024]; // 1KB

    c.bench_function("put_1kb", |b| {
        b.iter(|| {
            rt.block_on(async {
                blob_store
                    .put("bench.bin", black_box(&data), PutOptions::default())
                    .await
                    .unwrap()
            })
        })
    });
}

fn bench_put_medium(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let store = TensorStore::new();
    let blob_store =
        rt.block_on(async { BlobStore::new(store, BlobConfig::default()).await.unwrap() });

    let data = vec![0u8; 1024 * 1024]; // 1MB

    c.bench_function("put_1mb", |b| {
        b.iter(|| {
            rt.block_on(async {
                blob_store
                    .put("bench.bin", black_box(&data), PutOptions::default())
                    .await
                    .unwrap()
            })
        })
    });
}

fn bench_get(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let store = TensorStore::new();
    let blob_store =
        rt.block_on(async { BlobStore::new(store, BlobConfig::default()).await.unwrap() });

    let data = vec![0u8; 1024 * 1024]; // 1MB
    let artifact_id = rt.block_on(async {
        blob_store
            .put("bench.bin", &data, PutOptions::default())
            .await
            .unwrap()
    });

    c.bench_function("get_1mb", |b| {
        b.iter(|| rt.block_on(async { blob_store.get(black_box(&artifact_id)).await.unwrap() }))
    });
}

fn bench_metadata(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let store = TensorStore::new();
    let blob_store =
        rt.block_on(async { BlobStore::new(store, BlobConfig::default()).await.unwrap() });

    let artifact_id = rt.block_on(async {
        blob_store
            .put("bench.bin", b"data", PutOptions::default())
            .await
            .unwrap()
    });

    c.bench_function("metadata_lookup", |b| {
        b.iter(|| {
            rt.block_on(async { blob_store.metadata(black_box(&artifact_id)).await.unwrap() })
        })
    });
}

fn bench_chunk_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let data = vec![0u8; 10 * 1024 * 1024]; // 10MB

    let mut group = c.benchmark_group("chunk_sizes");

    for chunk_size in [256 * 1024, 512 * 1024, 1024 * 1024, 2 * 1024 * 1024] {
        let store = TensorStore::new();
        let config = BlobConfig::new().with_chunk_size(chunk_size);
        let blob_store = rt.block_on(async { BlobStore::new(store, config).await.unwrap() });

        group.bench_with_input(
            BenchmarkId::new("put_10mb", format!("{}kb", chunk_size / 1024)),
            &chunk_size,
            |b, _| {
                b.iter(|| {
                    rt.block_on(async {
                        blob_store
                            .put("bench.bin", black_box(&data), PutOptions::default())
                            .await
                            .unwrap()
                    })
                })
            },
        );
    }

    group.finish();
}

fn bench_deduplication(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let store = TensorStore::new();
    let config = BlobConfig::new().with_chunk_size(1024);
    let blob_store = rt.block_on(async { BlobStore::new(store, config).await.unwrap() });

    // Pre-store the data once
    let data = vec![42u8; 10 * 1024]; // 10KB of repeated data
    rt.block_on(async {
        blob_store
            .put("original.bin", &data, PutOptions::default())
            .await
            .unwrap()
    });

    c.bench_function("put_deduplicated_10kb", |b| {
        b.iter(|| {
            rt.block_on(async {
                blob_store
                    .put("duplicate.bin", black_box(&data), PutOptions::default())
                    .await
                    .unwrap()
            })
        })
    });
}

criterion_group!(
    benches,
    bench_put_small,
    bench_put_medium,
    bench_get,
    bench_metadata,
    bench_chunk_sizes,
    bench_deduplication,
);

criterion_main!(benches);
