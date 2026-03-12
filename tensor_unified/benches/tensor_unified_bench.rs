// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(missing_docs)]
use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tensor_unified::{FindPattern, UnifiedEngine, UnifiedItem};
use tokio::runtime::Runtime;

fn create_test_entity(id: usize) -> (String, HashMap<String, String>, Option<Vec<f32>>) {
    let key = format!("entity:{}", id);
    let mut fields = HashMap::new();
    fields.insert("name".to_string(), format!("Entity {}", id));
    fields.insert("type".to_string(), "test".to_string());
    let embedding = Some(vec![id as f32 / 100.0; 64]);
    (key, fields, embedding)
}

fn bench_create_entity(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_entity");
    let rt = Runtime::new().unwrap();

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let engine = UnifiedEngine::new();
                rt.block_on(async {
                    for i in 0..size {
                        let (key, fields, embedding) = create_test_entity(i);
                        engine.create_entity(&key, fields, embedding).await.ok();
                    }
                });
                black_box(&engine);
            });
        });
    }

    group.finish();
}

fn bench_embed_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("embed_batch");
    let rt = Runtime::new().unwrap();

    for size in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let engine = UnifiedEngine::new();
                let items: Vec<_> = (0..size)
                    .map(|i| (format!("doc:{}", i), vec![i as f32 / 100.0; 64]))
                    .collect();

                rt.block_on(async {
                    engine.embed_batch(items).await.ok();
                });
                black_box(&engine);
            });
        });
    }

    group.finish();
}

fn bench_find_nodes(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_nodes");
    let rt = Runtime::new().unwrap();

    for size in [10, 100, 1000].iter() {
        // Setup engine with nodes
        let engine = UnifiedEngine::new();
        for _ in 0..*size {
            engine.graph().create_node("test", HashMap::new()).ok();
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _size| {
            b.iter(|| {
                rt.block_on(async {
                    let pattern = FindPattern::Nodes {
                        label: Some("test".to_string()),
                    };
                    let result = engine.find(&pattern, None).await;
                    let _ = black_box(result);
                });
            });
        });
    }

    group.finish();
}

fn bench_unified_item_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("unified_item");

    group.bench_function("new", |b| {
        b.iter(|| {
            let item = UnifiedItem::new("test", "id");
            black_box(item);
        });
    });

    group.bench_function("with_data", |b| {
        let data: HashMap<String, String> = (0..10)
            .map(|i| (format!("key{}", i), format!("value{}", i)))
            .collect();

        b.iter(|| {
            let item = UnifiedItem::with_data("test", "id", data.clone());
            black_box(item);
        });
    });

    group.bench_function("with_score_and_embedding", |b| {
        let embedding = vec![0.1f32; 64];
        b.iter(|| {
            let item = UnifiedItem::new("test", "id")
                .with_score(0.95)
                .with_embedding(embedding.clone());
            black_box(item);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_create_entity,
    bench_embed_batch,
    bench_find_nodes,
    bench_unified_item_creation,
);
criterion_main!(benches);
