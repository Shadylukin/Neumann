// SPDX-License-Identifier: MIT OR Apache-2.0
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use query_router::QueryRouter;

fn bench_relational_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("relational_execute");

    let router = QueryRouter::new();
    router
        .execute_parsed("CREATE TABLE users (id INT, name TEXT, email TEXT)")
        .unwrap();

    // Insert some data
    for i in 0..100 {
        router
            .execute_parsed(&format!(
                "INSERT INTO users VALUES ({}, 'user{}', 'user{}@example.com')",
                i, i, i
            ))
            .unwrap();
    }

    group.bench_function("select_all", |b| {
        b.iter(|| {
            let result = router
                .execute_parsed(black_box("SELECT * FROM users"))
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("select_where", |b| {
        b.iter(|| {
            let result = router
                .execute_parsed(black_box("SELECT * FROM users WHERE id = 50"))
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("insert", |b| {
        let mut i = 1000;
        b.iter(|| {
            let query = format!(
                "INSERT INTO users VALUES ({}, 'user{}', 'user{}@example.com')",
                i, i, i
            );
            let result = router.execute_parsed(black_box(&query)).unwrap();
            i += 1;
            black_box(result);
        });
    });

    group.bench_function("update", |b| {
        b.iter(|| {
            let result = router
                .execute_parsed(black_box("UPDATE users SET name = 'updated' WHERE id = 50"))
                .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_graph_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_execute");

    let router = QueryRouter::new();

    // Create nodes
    for i in 0..100 {
        router
            .execute_parsed(&format!(
                "NODE CREATE person {{id: {}, name: 'person{}'}}",
                i, i
            ))
            .unwrap();
    }

    // Create edges
    for i in 0..99 {
        router
            .execute_parsed(&format!("EDGE CREATE {} -> {} : friend", i + 1, i + 2))
            .unwrap();
    }

    group.bench_function("node_create", |b| {
        let mut i = 1000;
        b.iter(|| {
            let query = format!("NODE CREATE person {{id: {}, name: 'person{}'}}", i, i);
            let result = router.execute_parsed(black_box(&query)).unwrap();
            i += 1;
            black_box(result);
        });
    });

    group.bench_function("edge_create", |b| {
        let mut i = 1000;
        b.iter(|| {
            let query = format!("EDGE CREATE {} -> {} : knows", i, i + 1);
            let result = router.execute_parsed(black_box(&query)).unwrap();
            i += 1;
            black_box(result);
        });
    });

    group.bench_function("neighbors", |b| {
        b.iter(|| {
            let result = router
                .execute_parsed(black_box("NEIGHBORS 50 OUTGOING"))
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("path", |b| {
        b.iter(|| {
            let result = router.execute_parsed(black_box("PATH 1 -> 10")).unwrap();
            black_box(result);
        });
    });

    group.bench_function("find_node", |b| {
        b.iter(|| {
            let result = router
                .execute_parsed(black_box("FIND NODE WHERE id > 0"))
                .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_vector_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_execute");

    let router = QueryRouter::new();

    // Store embeddings
    for i in 0..100 {
        let vector: Vec<f64> = (0..128).map(|j| (i * 128 + j) as f64 / 10000.0).collect();
        let vector_str = vector
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
            .join(", ");
        router
            .execute_parsed(&format!("EMBED STORE 'doc{}' [{}]", i, vector_str))
            .unwrap();
    }

    group.bench_function("embed_store", |b| {
        let mut i = 1000;
        b.iter(|| {
            let vector: Vec<f64> = (0..128).map(|j| (i * 128 + j) as f64 / 10000.0).collect();
            let vector_str = vector
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
                .join(", ");
            let query = format!("EMBED STORE 'doc{}' [{}]", i, vector_str);
            let result = router.execute_parsed(black_box(&query)).unwrap();
            i += 1;
            black_box(result);
        });
    });

    group.bench_function("embed_get", |b| {
        b.iter(|| {
            let result = router
                .execute_parsed(black_box("EMBED GET 'doc50'"))
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("similar_top5", |b| {
        b.iter(|| {
            let result = router
                .execute_parsed(black_box("SIMILAR 'doc50' LIMIT 5"))
                .unwrap();
            black_box(result);
        });
    });

    group.bench_function("similar_top10", |b| {
        b.iter(|| {
            let result = router
                .execute_parsed(black_box("SIMILAR 'doc50' LIMIT 10"))
                .unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");

    let router = QueryRouter::new();

    // Setup: tables, nodes, embeddings
    router
        .execute_parsed("CREATE TABLE users (id INT, name TEXT)")
        .unwrap();
    for i in 0..50 {
        router
            .execute_parsed(&format!("INSERT INTO users VALUES ({}, 'user{}')", i, i))
            .unwrap();
        router
            .execute_parsed(&format!("NODE CREATE person {{id: {}}}", i))
            .unwrap();
        let vector: Vec<f64> = (0..64).map(|j| (i * 64 + j) as f64 / 1000.0).collect();
        let vector_str = vector
            .iter()
            .map(|v| format!("{:.4}", v))
            .collect::<Vec<_>>()
            .join(", ");
        router
            .execute_parsed(&format!("EMBED STORE 'doc{}' [{}]", i, vector_str))
            .unwrap();
    }

    let queries = [
        "SELECT * FROM users WHERE id = 25",
        "NEIGHBORS 25 OUTGOING",
        "SIMILAR 'doc25' LIMIT 5",
        "INSERT INTO users VALUES (1000, 'newuser')",
        "NODE CREATE person {id: 1000}",
    ];

    group.throughput(Throughput::Elements(queries.len() as u64));
    group.bench_function("mixed_5_queries", |b| {
        b.iter(|| {
            for query in queries.iter() {
                let result = router.execute_parsed(black_box(query)).unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

fn bench_parse_vs_execute(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse_vs_execute");

    let router = QueryRouter::new();
    router
        .execute_parsed("CREATE TABLE users (id INT, name TEXT)")
        .unwrap();
    for i in 0..100 {
        router
            .execute_parsed(&format!("INSERT INTO users VALUES ({}, 'user{}')", i, i))
            .unwrap();
    }

    let query = "SELECT * FROM users WHERE id = 50";

    group.bench_function("select_where", |b| {
        b.iter(|| {
            let result = router.execute_parsed(black_box(query)).unwrap();
            black_box(result);
        });
    });

    let simple_query = "SELECT * FROM users";
    group.bench_function("select_all", |b| {
        b.iter(|| {
            let result = router.execute_parsed(black_box(simple_query)).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    for count in [100, 500, 1000].iter() {
        let router = QueryRouter::new();
        router
            .execute_parsed("CREATE TABLE bench (id INT, value TEXT)")
            .unwrap();

        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::new("insert", count), count, |b, &count| {
            b.iter(|| {
                for i in 0..count {
                    let query = format!("INSERT INTO bench VALUES ({}, 'value{}')", i, i);
                    router.execute_parsed(black_box(&query)).unwrap();
                }
            });
        });
    }

    group.finish();
}

fn bench_cross_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_engine");

    // Setup: shared store with entities having both embeddings and graph connections
    let store = tensor_store::TensorStore::new();
    let mut router = QueryRouter::with_shared_store(store);

    // Create 200 entities with embeddings (reduced from 1000 for faster benchmarks)
    for i in 0..200 {
        let embedding: Vec<f32> = (0..128)
            .map(|j| ((i * 128 + j) as f32) / 100000.0)
            .collect();
        router
            .vector()
            .set_entity_embedding(&format!("user:{}", i), embedding)
            .unwrap();
    }

    // Connect entities in a graph (each entity connects to ~5 neighbors)
    for i in 0..200 {
        for offset in 1..=5 {
            let target = (i + offset) % 200;
            router
                .connect_entities(
                    &format!("user:{}", i),
                    &format!("user:{}", target),
                    "follows",
                )
                .unwrap();
        }
    }

    let query_vec: Vec<f32> = (0..128).map(|j| (50 * 128 + j) as f32 / 100000.0).collect();

    // Build HNSW index for fast similarity search
    router.build_vector_index().unwrap();

    group.bench_function("connect_entities", |b| {
        let mut i = 2000;
        b.iter(|| {
            let from = format!("user:{}", i);
            let to = format!("user:{}", i + 1);
            // First create entities with embeddings
            let embedding: Vec<f32> = (0..128)
                .map(|j| ((i * 128 + j) as f32) / 100000.0)
                .collect();
            router
                .vector()
                .set_entity_embedding(&from, embedding.clone())
                .unwrap();
            router
                .vector()
                .set_entity_embedding(&to, embedding)
                .unwrap();
            // Then connect them
            let result = router.connect_entities(black_box(&from), black_box(&to), "follows");
            i += 2;
            let _ = black_box(result);
        });
    });

    group.bench_function("find_neighbors_by_similarity", |b| {
        b.iter(|| {
            let result = router.find_neighbors_by_similarity(
                black_box("user:50"),
                black_box(&query_vec),
                black_box(10),
            );
            let _ = black_box(result);
        });
    });

    group.bench_function("find_similar_connected", |b| {
        b.iter(|| {
            let result = router.find_similar_connected(
                black_box("user:50"),
                black_box("user:100"),
                black_box(10),
            );
            let _ = black_box(result);
        });
    });

    // Benchmark with varying top_k values
    for k in [5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("find_neighbors_by_similarity_k", k),
            k,
            |b, &k| {
                b.iter(|| {
                    let result = router.find_neighbors_by_similarity(
                        black_box("user:50"),
                        black_box(&query_vec),
                        black_box(k),
                    );
                    let _ = black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_cross_engine_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_engine_scale");
    group.sample_size(50); // Fewer samples for expensive benchmarks

    for entity_count in [100, 500, 1000].iter() {
        let store = tensor_store::TensorStore::new();
        let mut router = QueryRouter::with_shared_store(store);

        // Create entities with embeddings
        for i in 0..*entity_count {
            let embedding: Vec<f32> = (0..128)
                .map(|j| ((i * 128 + j) as f32) / 100000.0)
                .collect();
            router
                .vector()
                .set_entity_embedding(&format!("user:{}", i), embedding)
                .unwrap();
        }

        // Connect entities
        for i in 0..*entity_count {
            for offset in 1..=5 {
                let target = (i + offset) % entity_count;
                router
                    .connect_entities(
                        &format!("user:{}", i),
                        &format!("user:{}", target),
                        "follows",
                    )
                    .unwrap();
            }
        }

        let query_vec: Vec<f32> = (0..128).map(|j| j as f32 / 100000.0).collect();

        // Build HNSW index for fast similarity search
        router.build_vector_index().unwrap();

        group.throughput(Throughput::Elements(*entity_count as u64));
        group.bench_with_input(
            BenchmarkId::new("find_neighbors_by_similarity", entity_count),
            entity_count,
            |b, _| {
                b.iter(|| {
                    let result = router.find_neighbors_by_similarity(
                        black_box("user:0"),
                        black_box(&query_vec),
                        black_box(10),
                    );
                    let _ = black_box(result);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("find_similar_connected", entity_count),
            entity_count,
            |b, _| {
                b.iter(|| {
                    let result = router.find_similar_connected(
                        black_box("user:0"),
                        black_box("user:50"),
                        black_box(10),
                    );
                    let _ = black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_relational_execute,
    bench_graph_execute,
    bench_vector_execute,
    bench_mixed_workload,
    bench_parse_vs_execute,
    bench_throughput,
    bench_cross_engine,
    bench_cross_engine_scale,
);

criterion_main!(benches);
