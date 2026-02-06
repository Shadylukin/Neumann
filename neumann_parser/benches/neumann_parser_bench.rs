// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![allow(missing_docs)]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use neumann_parser::{lexer, parser};

fn bench_tokenize(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenize");

    let queries = [
        ("simple_select", "SELECT * FROM users"),
        ("select_where", "SELECT id, name FROM users WHERE age > 21 AND active = true"),
        (
            "complex_select",
            "SELECT id, name, email FROM users WHERE age >= 18 AND status = 'active' ORDER BY name ASC LIMIT 100",
        ),
        ("insert", "INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', 30)"),
        ("update", "UPDATE users SET name = 'Bob', age = 25 WHERE id = 1"),
        ("node", "NODE person {name: 'Alice', age: 30, email: 'alice@example.com'}"),
        ("edge", "EDGE person:1 friend person:2 {since: 2020, strength: 0.8}"),
        ("path", "PATH person:1 TO person:100 VIA friend"),
        ("embed", "EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]"),
        ("similar", "SIMILAR 'doc1' LIMIT 10"),
    ];

    for (name, query) in queries.iter() {
        group.throughput(Throughput::Bytes(query.len() as u64));
        group.bench_with_input(BenchmarkId::new("query", name), query, |b, query| {
            b.iter(|| {
                let tokens = lexer::tokenize(black_box(query));
                black_box(tokens);
            });
        });
    }

    group.finish();
}

fn bench_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("parse");

    let queries = [
        ("simple_select", "SELECT * FROM users"),
        ("select_where", "SELECT id, name FROM users WHERE age > 21 AND active = true"),
        (
            "complex_select",
            "SELECT id, name, email FROM users WHERE age >= 18 AND status = 'active' ORDER BY name ASC LIMIT 100",
        ),
        ("insert", "INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', 30)"),
        ("update", "UPDATE users SET name = 'Bob', age = 25 WHERE id = 1"),
        ("delete", "DELETE FROM users WHERE id = 1"),
        ("create_table", "CREATE TABLE users (id INT, name TEXT, email TEXT, age INT, active BOOL)"),
        ("node", "NODE CREATE person {name: 'Alice', age: 30, email: 'alice@example.com'}"),
        ("edge", "EDGE CREATE 1 -> 2 : knows {since: 2020, strength: 0.8}"),
        ("neighbors", "NEIGHBORS 1 OUTGOING : follows"),
        ("path", "PATH 1 -> 100 LIMIT 10"),
        ("find_node", "FIND NODE WHERE age > 21"),
        ("find_edge", "FIND EDGE WHERE type = 'friend'"),
        ("embed_store", "EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]"),
        ("embed_get", "EMBED GET 'doc1'"),
        ("similar", "SIMILAR 'doc1' LIMIT 10"),
    ];

    for (name, query) in queries.iter() {
        group.throughput(Throughput::Bytes(query.len() as u64));
        group.bench_with_input(BenchmarkId::new("query", name), query, |b, query| {
            b.iter(|| {
                let stmt = parser::parse(black_box(query)).unwrap();
                black_box(stmt);
            });
        });
    }

    group.finish();
}

fn bench_expression_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("expression_complexity");

    let expressions = [
        ("simple", "SELECT * FROM t WHERE a = 1"),
        ("binary_and", "SELECT * FROM t WHERE a = 1 AND b = 2"),
        ("binary_or", "SELECT * FROM t WHERE a = 1 OR b = 2"),
        (
            "nested_and_or",
            "SELECT * FROM t WHERE (a = 1 AND b = 2) OR (c = 3 AND d = 4)",
        ),
        (
            "deep_nesting",
            "SELECT * FROM t WHERE ((a = 1 AND b = 2) OR (c = 3 AND d = 4)) AND ((e = 5 OR f = 6) AND g = 7)",
        ),
        (
            "arithmetic",
            "SELECT * FROM t WHERE a + b * c - d / e > 100",
        ),
        (
            "comparison_chain",
            "SELECT * FROM t WHERE a > 1 AND b < 2 AND c >= 3 AND d <= 4 AND e != 5",
        ),
    ];

    for (name, query) in expressions.iter() {
        group.bench_with_input(BenchmarkId::new("expr", name), query, |b, query| {
            b.iter(|| {
                let stmt = parser::parse(black_box(query)).unwrap();
                black_box(stmt);
            });
        });
    }

    group.finish();
}

fn bench_batch_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_parse");

    let queries: Vec<&str> = vec![
        "SELECT * FROM users",
        "INSERT INTO users VALUES (1, 'Alice')",
        "UPDATE users SET name = 'Bob' WHERE id = 1",
        "DELETE FROM users WHERE id = 1",
        "NODE CREATE person {name: 'Alice'}",
        "EDGE CREATE 1 -> 2 : friend",
        "PATH 1 -> 2",
        "EMBED STORE 'doc1' [0.1, 0.2, 0.3]",
        "SIMILAR 'doc1' LIMIT 5",
        "NEIGHBORS 1 OUTGOING",
    ];

    for count in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), count, |b, &count| {
            b.iter(|| {
                for i in 0..count {
                    let query = queries[i % queries.len()];
                    let stmt = parser::parse(black_box(query)).unwrap();
                    black_box(stmt);
                }
            });
        });
    }

    group.finish();
}

fn bench_large_queries(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_queries");

    // Large INSERT with many values
    let large_insert = format!(
        "INSERT INTO users VALUES ({})",
        (0..100)
            .map(|i| format!("{}, 'user{}', 'user{}@example.com'", i, i, i))
            .collect::<Vec<_>>()
            .join("), (")
    );

    // Large embedding vector
    let large_embed = format!(
        "EMBED STORE 'doc1' [{}]",
        (0..768)
            .map(|i| format!("{:.4}", i as f64 / 1000.0))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Complex WHERE with many conditions
    let complex_where = format!(
        "SELECT * FROM users WHERE {}",
        (0..20)
            .map(|i| format!("col{} = {}", i, i))
            .collect::<Vec<_>>()
            .join(" AND ")
    );

    group.bench_function("large_insert_100_rows", |b| {
        b.iter(|| {
            let stmt = parser::parse(black_box(&large_insert)).unwrap();
            black_box(stmt);
        });
    });

    group.bench_function("large_embed_768_dim", |b| {
        b.iter(|| {
            let stmt = parser::parse(black_box(&large_embed)).unwrap();
            black_box(stmt);
        });
    });

    group.bench_function("complex_where_20_conditions", |b| {
        b.iter(|| {
            let stmt = parser::parse(black_box(&complex_where)).unwrap();
            black_box(stmt);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tokenize,
    bench_parse,
    bench_expression_complexity,
    bench_batch_parse,
    bench_large_queries,
);

criterion_main!(benches);
