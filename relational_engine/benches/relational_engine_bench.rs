use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use relational_engine::{Column, ColumnType, Condition, RelationalEngine, Schema, Value};
use std::collections::HashMap;

fn create_users_schema() -> Schema {
    Schema::new(vec![
        Column::new("name", ColumnType::String),
        Column::new("age", ColumnType::Int),
        Column::new("email", ColumnType::String).nullable(),
        Column::new("score", ColumnType::Float),
    ])
}

fn create_user_values(id: i64) -> HashMap<String, Value> {
    let mut values = HashMap::new();
    values.insert("name".to_string(), Value::String(format!("User{}", id)));
    values.insert("age".to_string(), Value::Int(20 + (id % 50)));
    values.insert(
        "email".to_string(),
        Value::String(format!("user{}@example.com", id)),
    );
    values.insert("score".to_string(), Value::Float(id as f64 * 0.1));
    values
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");

    for size in [100, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let engine = RelationalEngine::new();
                engine.create_table("users", create_users_schema()).unwrap();

                for i in 0..size {
                    engine
                        .insert("users", create_user_values(i as i64))
                        .unwrap();
                }
                black_box(&engine);
            });
        });
    }

    group.finish();
}

fn bench_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");

    for size in [100, 1000, 5000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Pre-create the rows vector outside the benchmark loop
            let rows: Vec<_> = (0..size).map(|i| create_user_values(i as i64)).collect();

            b.iter(|| {
                let engine = RelationalEngine::new();
                engine.create_table("users", create_users_schema()).unwrap();
                engine.batch_insert("users", rows.clone()).unwrap();
                black_box(&engine);
            });
        });
    }

    group.finish();
}

fn bench_select_full_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("select_full_scan");

    for size in [100, 1000, 5000].iter() {
        let engine = RelationalEngine::new();
        engine.create_table("users", create_users_schema()).unwrap();
        for i in 0..*size {
            engine
                .insert("users", create_user_values(i as i64))
                .unwrap();
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(engine.select("users", Condition::True).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_select_filtered(c: &mut Criterion) {
    let mut group = c.benchmark_group("select_filtered");

    // Setup: 5000 rows, filter matches ~10%
    let engine = RelationalEngine::new();
    engine.create_table("users", create_users_schema()).unwrap();
    for i in 0..5000 {
        engine
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }

    group.bench_function("eq_10_percent", |b| {
        b.iter(|| {
            // age = 25 matches ~2% (100 rows out of 5000)
            black_box(
                engine
                    .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
                    .unwrap(),
            );
        });
    });

    group.bench_function("range_20_percent", |b| {
        b.iter(|| {
            // age >= 60 matches 20% (ages 60-69 out of 20-69)
            black_box(
                engine
                    .select("users", Condition::Ge("age".to_string(), Value::Int(60)))
                    .unwrap(),
            );
        });
    });

    group.bench_function("compound_and", |b| {
        b.iter(|| {
            // age >= 30 AND age < 40
            let condition = Condition::Ge("age".to_string(), Value::Int(30))
                .and(Condition::Lt("age".to_string(), Value::Int(40)));
            black_box(engine.select("users", condition).unwrap());
        });
    });

    group.finish();
}

fn bench_select_by_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("select_by_id");

    let engine = RelationalEngine::new();
    engine.create_table("users", create_users_schema()).unwrap();
    for i in 0..5000 {
        engine
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }

    group.bench_function("single_row", |b| {
        b.iter(|| {
            black_box(
                engine
                    .select("users", Condition::Eq("_id".to_string(), Value::Int(2500)))
                    .unwrap(),
            );
        });
    });

    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update");

    group.bench_function("update_10_percent", |b| {
        b.iter_batched(
            || {
                let engine = RelationalEngine::new();
                engine.create_table("users", create_users_schema()).unwrap();
                for i in 0..1000 {
                    engine
                        .insert("users", create_user_values(i as i64))
                        .unwrap();
                }
                engine
            },
            |engine| {
                let mut updates = HashMap::new();
                updates.insert("score".to_string(), Value::Float(999.0));
                black_box(
                    engine
                        .update(
                            "users",
                            Condition::Lt("age".to_string(), Value::Int(25)),
                            updates,
                        )
                        .unwrap(),
                );
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_delete(c: &mut Criterion) {
    let mut group = c.benchmark_group("delete");

    group.bench_function("delete_10_percent", |b| {
        b.iter_batched(
            || {
                let engine = RelationalEngine::new();
                engine.create_table("users", create_users_schema()).unwrap();
                for i in 0..1000 {
                    engine
                        .insert("users", create_user_values(i as i64))
                        .unwrap();
                }
                engine
            },
            |engine| {
                black_box(
                    engine
                        .delete_rows("users", Condition::Lt("age".to_string(), Value::Int(25)))
                        .unwrap(),
                );
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_join(c: &mut Criterion) {
    let mut group = c.benchmark_group("join");

    // Setup users and posts tables
    fn setup_tables(num_users: i64, posts_per_user: i64) -> RelationalEngine {
        let engine = RelationalEngine::new();

        let users_schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("age", ColumnType::Int),
        ]);
        engine.create_table("users", users_schema).unwrap();

        let posts_schema = Schema::new(vec![
            Column::new("user_id", ColumnType::Int),
            Column::new("title", ColumnType::String),
            Column::new("views", ColumnType::Int),
        ]);
        engine.create_table("posts", posts_schema).unwrap();

        for i in 1..=num_users {
            let mut user_values = HashMap::new();
            user_values.insert("name".to_string(), Value::String(format!("User{}", i)));
            user_values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", user_values).unwrap();

            for j in 0..posts_per_user {
                let mut post_values = HashMap::new();
                post_values.insert("user_id".to_string(), Value::Int(i));
                post_values.insert(
                    "title".to_string(),
                    Value::String(format!("Post {} by User {}", j, i)),
                );
                post_values.insert("views".to_string(), Value::Int(j * 10));
                engine.insert("posts", post_values).unwrap();
            }
        }

        engine
    }

    // 50 users x 10 posts = 500 join results
    let engine_small = setup_tables(50, 10);
    group.bench_function("50x10_500_results", |b| {
        b.iter(|| {
            black_box(
                engine_small
                    .join("users", "posts", "_id", "user_id")
                    .unwrap(),
            );
        });
    });

    // 100 users x 10 posts = 1000 join results
    let engine_medium = setup_tables(100, 10);
    group.bench_function("100x10_1000_results", |b| {
        b.iter(|| {
            black_box(
                engine_medium
                    .join("users", "posts", "_id", "user_id")
                    .unwrap(),
            );
        });
    });

    // 100 users x 50 posts = 5000 join results
    let engine_large = setup_tables(100, 50);
    group.bench_function("100x50_5000_results", |b| {
        b.iter(|| {
            black_box(
                engine_large
                    .join("users", "posts", "_id", "user_id")
                    .unwrap(),
            );
        });
    });

    group.finish();
}

fn bench_row_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("row_count");

    for size in [100, 1000, 5000].iter() {
        let engine = RelationalEngine::new();
        engine.create_table("users", create_users_schema()).unwrap();
        for i in 0..*size {
            engine
                .insert("users", create_user_values(i as i64))
                .unwrap();
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                black_box(engine.row_count("users").unwrap());
            });
        });
    }

    group.finish();
}

fn bench_indexed_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexed_select");

    // Setup: 5000 rows with index on age
    let engine = RelationalEngine::new();
    engine.create_table("users", create_users_schema()).unwrap();
    for i in 0..5000 {
        engine
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }
    engine.create_index("users", "age").unwrap();

    group.bench_function("eq_with_index_2_percent", |b| {
        b.iter(|| {
            // age = 25 matches 2% (100 rows out of 5000)
            black_box(
                engine
                    .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
                    .unwrap(),
            );
        });
    });

    // Compare without index
    let engine_no_idx = RelationalEngine::new();
    engine_no_idx
        .create_table("users", create_users_schema())
        .unwrap();
    for i in 0..5000 {
        engine_no_idx
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }

    group.bench_function("eq_without_index_2_percent", |b| {
        b.iter(|| {
            black_box(
                engine_no_idx
                    .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
                    .unwrap(),
            );
        });
    });

    group.finish();
}

fn bench_indexed_select_by_id(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexed_select_by_id");

    // Setup: 5000 rows
    let engine = RelationalEngine::new();
    engine.create_table("users", create_users_schema()).unwrap();
    for i in 0..5000 {
        engine
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }
    engine.create_index("users", "_id").unwrap();

    group.bench_function("by_id_with_index", |b| {
        b.iter(|| {
            black_box(
                engine
                    .select("users", Condition::Eq("_id".to_string(), Value::Int(2500)))
                    .unwrap(),
            );
        });
    });

    // Compare without index
    let engine_no_idx = RelationalEngine::new();
    engine_no_idx
        .create_table("users", create_users_schema())
        .unwrap();
    for i in 0..5000 {
        engine_no_idx
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }

    group.bench_function("by_id_without_index", |b| {
        b.iter(|| {
            black_box(
                engine_no_idx
                    .select("users", Condition::Eq("_id".to_string(), Value::Int(2500)))
                    .unwrap(),
            );
        });
    });

    group.finish();
}

fn bench_create_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_index");

    for size in [100, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched(
                || {
                    let engine = RelationalEngine::new();
                    engine.create_table("users", create_users_schema()).unwrap();
                    for i in 0..size {
                        engine
                            .insert("users", create_user_values(i as i64))
                            .unwrap();
                    }
                    engine
                },
                |engine| {
                    black_box(engine.create_index("users", "age").unwrap());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_btree_indexed_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_range_query");

    // Setup: 5000 rows with B-tree index on age
    let engine = RelationalEngine::new();
    engine.create_table("users", create_users_schema()).unwrap();
    for i in 0..5000 {
        engine
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }
    engine.create_btree_index("users", "age").unwrap();

    group.bench_function("ge_with_btree_20_percent", |b| {
        b.iter(|| {
            // age >= 60 matches 20% (ages 60-69 out of 20-69)
            black_box(
                engine
                    .select("users", Condition::Ge("age".to_string(), Value::Int(60)))
                    .unwrap(),
            );
        });
    });

    group.bench_function("lt_with_btree_10_percent", |b| {
        b.iter(|| {
            // age < 25 matches 10% (ages 20-24 out of 20-69)
            black_box(
                engine
                    .select("users", Condition::Lt("age".to_string(), Value::Int(25)))
                    .unwrap(),
            );
        });
    });

    // Compare without B-tree index
    let engine_no_idx = RelationalEngine::new();
    engine_no_idx
        .create_table("users", create_users_schema())
        .unwrap();
    for i in 0..5000 {
        engine_no_idx
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }

    group.bench_function("ge_without_btree_20_percent", |b| {
        b.iter(|| {
            black_box(
                engine_no_idx
                    .select("users", Condition::Ge("age".to_string(), Value::Int(60)))
                    .unwrap(),
            );
        });
    });

    group.bench_function("lt_without_btree_10_percent", |b| {
        b.iter(|| {
            black_box(
                engine_no_idx
                    .select("users", Condition::Lt("age".to_string(), Value::Int(25)))
                    .unwrap(),
            );
        });
    });

    group.finish();
}

fn bench_btree_compound_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("btree_compound_range");

    // Setup: 5000 rows with B-tree index on age
    let engine = RelationalEngine::new();
    engine.create_table("users", create_users_schema()).unwrap();
    for i in 0..5000 {
        engine
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }
    engine.create_btree_index("users", "age").unwrap();

    group.bench_function("range_and_with_btree", |b| {
        b.iter(|| {
            // age >= 30 AND age < 40 (20% of rows)
            let condition = Condition::Ge("age".to_string(), Value::Int(30))
                .and(Condition::Lt("age".to_string(), Value::Int(40)));
            black_box(engine.select("users", condition).unwrap());
        });
    });

    // Compare without B-tree index
    let engine_no_idx = RelationalEngine::new();
    engine_no_idx
        .create_table("users", create_users_schema())
        .unwrap();
    for i in 0..5000 {
        engine_no_idx
            .insert("users", create_user_values(i as i64))
            .unwrap();
    }

    group.bench_function("range_and_without_btree", |b| {
        b.iter(|| {
            let condition = Condition::Ge("age".to_string(), Value::Int(30))
                .and(Condition::Lt("age".to_string(), Value::Int(40)));
            black_box(engine_no_idx.select("users", condition).unwrap());
        });
    });

    group.finish();
}

fn bench_create_btree_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("create_btree_index");

    for size in [100, 1000, 5000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter_batched(
                || {
                    let engine = RelationalEngine::new();
                    engine.create_table("users", create_users_schema()).unwrap();
                    for i in 0..size {
                        engine
                            .insert("users", create_user_values(i as i64))
                            .unwrap();
                    }
                    engine
                },
                |engine| {
                    black_box(engine.create_btree_index("users", "age").unwrap());
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_insert,
    bench_batch_insert,
    bench_select_full_scan,
    bench_select_filtered,
    bench_select_by_id,
    bench_update,
    bench_delete,
    bench_join,
    bench_row_count,
    bench_indexed_select,
    bench_indexed_select_by_id,
    bench_create_index,
    bench_btree_indexed_range,
    bench_btree_compound_range,
    bench_create_btree_index,
);

criterion_main!(benches);
