// SPDX-License-Identifier: MIT OR Apache-2.0
//! Benchmarks for `tensor_vault` operations.
#![allow(missing_docs)]

use std::{collections::HashMap, sync::Arc};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use graph_engine::{GraphEngine, PropertyValue};
use peak_alloc::PeakAlloc;
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig};

#[global_allocator]
static PEAK_ALLOC: PeakAlloc = PeakAlloc;

fn create_vault() -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::new());
    Vault::new(b"benchmark_password", graph, store, VaultConfig::default()).unwrap()
}

/// Helper to add edges between entity keys using the node-based API.
fn add_bench_edge(graph: &GraphEngine, from_key: &str, to_key: &str, edge_type: &str) {
    let get_or_create = |key: &str| -> u64 {
        if let Ok(nodes) =
            graph.find_nodes_by_property("entity_key", &PropertyValue::String(key.to_string()))
        {
            if let Some(node) = nodes.first() {
                return node.id;
            }
        }
        let mut props = HashMap::new();
        props.insert(
            "entity_key".to_string(),
            PropertyValue::String(key.to_string()),
        );
        graph.create_node("BenchEntity", props).unwrap_or(0)
    };

    let from_node = get_or_create(from_key);
    let to_node = get_or_create(to_key);
    graph
        .create_edge(from_node, to_node, edge_type, HashMap::new(), true)
        .ok();
}

fn bench_key_derivation(c: &mut Criterion) {
    let mut group = c.benchmark_group("vault_crypto");

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("argon2id_derivation", |b| {
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
    println!(
        "\n  argon2id_derivation peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    group.finish();
}

fn bench_encrypt_decrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("vault_encrypt");

    // 1KB data
    let data_1kb = "x".repeat(1024);

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("set_1kb", |b| {
        let vault = create_vault();
        b.iter(|| {
            vault
                .set(Vault::ROOT, black_box("bench:key"), black_box(&data_1kb))
                .unwrap();
        });
    });
    println!(
        "\n  set_1kb peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("get_1kb", |b| {
        let vault = create_vault();
        vault.set(Vault::ROOT, "bench:key", &data_1kb).unwrap();
        b.iter(|| {
            let _ = vault.get(black_box(Vault::ROOT), black_box("bench:key"));
        });
    });
    println!(
        "  get_1kb peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    // 10KB data
    let large_data = "x".repeat(10 * 1024);

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("set_10kb", |b| {
        let vault = create_vault();
        b.iter(|| {
            vault
                .set(Vault::ROOT, black_box("bench:key"), black_box(&large_data))
                .unwrap();
        });
    });
    println!(
        "  set_10kb peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("get_10kb", |b| {
        let vault = create_vault();
        vault.set(Vault::ROOT, "bench:key", &large_data).unwrap();
        b.iter(|| {
            let _ = vault.get(black_box(Vault::ROOT), black_box("bench:key"));
        });
    });
    println!(
        "  get_10kb peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    group.finish();
}

fn bench_access_control(c: &mut Criterion) {
    let mut group = c.benchmark_group("vault_access");

    // Shallow access (direct edge)
    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("check_shallow", |b| {
        let vault = create_vault();
        vault.set(Vault::ROOT, "secret", "value").unwrap();
        vault.grant(Vault::ROOT, "user:alice", "secret").unwrap();
        b.iter(|| {
            let _ = vault.get(black_box("user:alice"), black_box("secret"));
        });
    });
    println!(
        "\n  check_shallow peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    // Deep access (10 hops)
    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("check_deep_10_hops", |b| {
        let vault = create_vault();
        vault.set(Vault::ROOT, "secret", "value").unwrap();

        // Create chain: user:start -> node:0 -> ... -> node:9 -> vault_secret:secret
        for i in 0..10 {
            let from = if i == 0 {
                "user:start".to_string()
            } else {
                format!("node:{}", i - 1)
            };
            let to = format!("node:{i}");
            add_bench_edge(vault.graph(), &from, &to, "LINK");
        }
        add_bench_edge(
            vault.graph(),
            "node:9",
            "vault_secret:secret",
            "VAULT_ACCESS",
        );

        b.iter(|| {
            let _ = vault.get(black_box("user:start"), black_box("secret"));
        });
    });
    println!(
        "  check_deep_10_hops peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    // Grant/revoke
    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("grant", |b| {
        let vault = create_vault();
        vault.set(Vault::ROOT, "secret", "value").unwrap();
        let mut i = 0;
        b.iter(|| {
            let entity = format!("user:{i}");
            vault
                .grant(Vault::ROOT, black_box(&entity), black_box("secret"))
                .unwrap();
            i += 1;
        });
    });
    println!("  grant peak RAM: {:.1} KB", PEAK_ALLOC.peak_usage_as_kb());

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("revoke", |b| {
        let vault = create_vault();
        vault.set(Vault::ROOT, "secret", "value").unwrap();
        // Pre-grant many users
        for i in 0..10000 {
            vault
                .grant(Vault::ROOT, &format!("user:{i}"), "secret")
                .unwrap();
        }
        let mut i = 0;
        b.iter(|| {
            let entity = format!("user:{i}");
            let _ = vault.revoke(Vault::ROOT, black_box(&entity), black_box("secret"));
            i += 1;
        });
    });
    println!("  revoke peak RAM: {:.1} KB", PEAK_ALLOC.peak_usage_as_kb());

    group.finish();
}

fn bench_list(c: &mut Criterion) {
    let mut group = c.benchmark_group("vault_list");

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("list_100_secrets", |b| {
        let vault = create_vault();
        for i in 0..100 {
            vault
                .set(Vault::ROOT, &format!("secret:{i}"), "value")
                .unwrap();
        }
        b.iter(|| {
            let _ = vault.list(black_box(Vault::ROOT), black_box("secret:*"));
        });
    });
    println!(
        "\n  list_100_secrets peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    PEAK_ALLOC.reset_peak_usage();
    group.bench_function("list_1000_secrets", |b| {
        let vault = create_vault();
        for i in 0..1000 {
            vault
                .set(Vault::ROOT, &format!("secret:{i}"), "value")
                .unwrap();
        }
        b.iter(|| {
            let _ = vault.list(black_box(Vault::ROOT), black_box("secret:*"));
        });
    });
    println!(
        "  list_1000_secrets peak RAM: {:.1} KB",
        PEAK_ALLOC.peak_usage_as_kb()
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_key_derivation,
    bench_encrypt_decrypt,
    bench_access_control,
    bench_list,
);
criterion_main!(benches);
