//! Benchmarks for tensor_vault operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use graph_engine::GraphEngine;
use std::sync::Arc;
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig};

fn create_vault() -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::new());
    Vault::new(b"benchmark_password", graph, store, VaultConfig::default()).unwrap()
}

fn bench_encrypt_1kb(c: &mut Criterion) {
    let vault = create_vault();
    let data = "x".repeat(1024);

    c.bench_function("vault_set_1kb", |b| {
        b.iter(|| {
            vault
                .set(Vault::ROOT, black_box("bench:key"), black_box(&data))
                .unwrap();
        });
    });
}

fn bench_decrypt_1kb(c: &mut Criterion) {
    let vault = create_vault();
    let data = "x".repeat(1024);
    vault.set(Vault::ROOT, "bench:key", &data).unwrap();

    c.bench_function("vault_get_1kb", |b| {
        b.iter(|| {
            let _ = vault.get(black_box(Vault::ROOT), black_box("bench:key"));
        });
    });
}

fn bench_key_derivation(c: &mut Criterion) {
    c.bench_function("argon2id_derivation", |b| {
        b.iter(|| {
            let store = TensorStore::new();
            let graph = Arc::new(GraphEngine::new());
            let _ = Vault::new(
                black_box(b"test_password"),
                graph,
                store,
                VaultConfig::default(),
            );
        });
    });
}

fn bench_access_check_shallow(c: &mut Criterion) {
    let vault = create_vault();
    vault.set(Vault::ROOT, "secret", "value").unwrap();
    vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();

    c.bench_function("access_check_shallow", |b| {
        b.iter(|| {
            let _ = vault.get(black_box("user:alice"), black_box("secret"));
        });
    });
}

fn bench_access_check_deep(c: &mut Criterion) {
    let vault = create_vault();
    vault.set(Vault::ROOT, "secret", "value").unwrap();

    // Create a chain: user:start -> node:1 -> node:2 -> ... -> vault_secret:secret
    for i in 0..10 {
        let from = if i == 0 {
            "user:start".to_string()
        } else {
            format!("node:{}", i - 1)
        };
        let to = format!("node:{i}");
        vault.graph.add_entity_edge(&from, &to, "LINK").unwrap();
    }
    vault
        .graph
        .add_entity_edge("node:9", "vault_secret:secret", "VAULT_ACCESS")
        .unwrap();

    c.bench_function("access_check_deep_10_hops", |b| {
        b.iter(|| {
            let _ = vault.get(black_box("user:start"), black_box("secret"));
        });
    });
}

criterion_group!(
    benches,
    bench_encrypt_1kb,
    bench_decrypt_1kb,
    bench_key_derivation,
    bench_access_check_shallow,
    bench_access_check_deep,
);
criterion_main!(benches);
