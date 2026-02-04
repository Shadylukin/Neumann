use super::*;
use tensor_store::{ScalarValue, TensorData, TensorValue};

fn make_tensor(id: i64, val: i64) -> TensorData {
    let mut data = TensorData::new();
    data.set("_id".to_string(), TensorValue::Scalar(ScalarValue::Int(id)));
    data.set(
        "val".to_string(),
        TensorValue::Scalar(ScalarValue::Int(val)),
    );
    data
}

fn make_tensor_float(id: i64, val: f64) -> TensorData {
    let mut data = TensorData::new();
    data.set("_id".to_string(), TensorValue::Scalar(ScalarValue::Int(id)));
    data.set(
        "val".to_string(),
        TensorValue::Scalar(ScalarValue::Float(val)),
    );
    data
}

fn make_tensor_str(id: i64, val: &str) -> TensorData {
    let mut data = TensorData::new();
    data.set("_id".to_string(), TensorValue::Scalar(ScalarValue::Int(id)));
    data.set(
        "name".to_string(),
        TensorValue::Scalar(ScalarValue::String(val.to_string())),
    );
    data
}

#[test]
fn test_tensor_eval_true() {
    let tensor = make_tensor(1, 42);
    assert!(Condition::True.evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_eq_int() {
    let tensor = make_tensor(1, 42);
    assert!(Condition::Eq("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
    assert!(!Condition::Eq("val".to_string(), Value::Int(99)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_ne_int() {
    let tensor = make_tensor(1, 42);
    assert!(Condition::Ne("val".to_string(), Value::Int(99)).evaluate_tensor(&tensor));
    assert!(!Condition::Ne("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_lt_int() {
    let tensor = make_tensor(1, 42);
    assert!(Condition::Lt("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
    assert!(!Condition::Lt("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
    assert!(!Condition::Lt("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_le_int() {
    let tensor = make_tensor(1, 42);
    assert!(Condition::Le("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
    assert!(Condition::Le("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
    assert!(!Condition::Le("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_gt_int() {
    let tensor = make_tensor(1, 42);
    assert!(Condition::Gt("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
    assert!(!Condition::Gt("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
    assert!(!Condition::Gt("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_ge_int() {
    let tensor = make_tensor(1, 42);
    assert!(Condition::Ge("val".to_string(), Value::Int(30)).evaluate_tensor(&tensor));
    assert!(Condition::Ge("val".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
    assert!(!Condition::Ge("val".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_and() {
    let tensor = make_tensor(1, 42);
    let cond = Condition::And(
        Box::new(Condition::Gt("val".to_string(), Value::Int(30))),
        Box::new(Condition::Lt("val".to_string(), Value::Int(50))),
    );
    assert!(cond.evaluate_tensor(&tensor));

    let cond2 = Condition::And(
        Box::new(Condition::Gt("val".to_string(), Value::Int(50))),
        Box::new(Condition::Lt("val".to_string(), Value::Int(60))),
    );
    assert!(!cond2.evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_or() {
    let tensor = make_tensor(1, 42);
    let cond = Condition::Or(
        Box::new(Condition::Lt("val".to_string(), Value::Int(30))),
        Box::new(Condition::Gt("val".to_string(), Value::Int(40))),
    );
    assert!(cond.evaluate_tensor(&tensor));

    let cond2 = Condition::Or(
        Box::new(Condition::Lt("val".to_string(), Value::Int(30))),
        Box::new(Condition::Gt("val".to_string(), Value::Int(50))),
    );
    assert!(!cond2.evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_id_field() {
    let tensor = make_tensor(99, 42);
    assert!(Condition::Eq("_id".to_string(), Value::Int(99)).evaluate_tensor(&tensor));
    assert!(!Condition::Eq("_id".to_string(), Value::Int(1)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_float() {
    let tensor = make_tensor_float(1, 3.14);
    assert!(Condition::Lt("val".to_string(), Value::Float(4.0)).evaluate_tensor(&tensor));
    assert!(Condition::Gt("val".to_string(), Value::Float(3.0)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_missing_column() {
    let tensor = make_tensor(1, 42);
    // Missing column should return false for comparisons
    assert!(!Condition::Eq("nonexistent".to_string(), Value::Int(42)).evaluate_tensor(&tensor));
    assert!(!Condition::Lt("nonexistent".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_string() {
    let tensor = make_tensor_str(1, "hello");
    assert!(
        Condition::Eq("name".to_string(), Value::String("hello".to_string()))
            .evaluate_tensor(&tensor)
    );
    assert!(
        !Condition::Eq("name".to_string(), Value::String("world".to_string()))
            .evaluate_tensor(&tensor)
    );
}

#[test]
fn test_tensor_eval_missing_id() {
    // Tensor without _id field
    let mut data = TensorData::new();
    data.set("val".to_string(), TensorValue::Scalar(ScalarValue::Int(42)));

    // Checking _id on tensor without _id should return false
    assert!(!Condition::Eq("_id".to_string(), Value::Int(1)).evaluate_tensor(&data));
}

#[test]
fn test_tensor_eval_non_int_id() {
    // Tensor with _id as string (not int)
    let mut data = TensorData::new();
    data.set(
        "_id".to_string(),
        TensorValue::Scalar(ScalarValue::String("not_an_int".to_string())),
    );

    // Checking _id should return false since it's not an Int
    assert!(!Condition::Eq("_id".to_string(), Value::Int(1)).evaluate_tensor(&data));
}

#[test]
fn test_tensor_eval_le_missing_column() {
    let tensor = make_tensor(1, 42);
    assert!(!Condition::Le("nonexistent".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_ge_missing_column() {
    let tensor = make_tensor(1, 42);
    assert!(!Condition::Ge("nonexistent".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_gt_missing_column() {
    let tensor = make_tensor(1, 42);
    assert!(!Condition::Gt("nonexistent".to_string(), Value::Int(50)).evaluate_tensor(&tensor));
}

#[test]
fn test_tensor_eval_nested_and_or() {
    let tensor = make_tensor(1, 42);
    // Complex condition: (val > 30 AND val < 50) OR val == 100
    let cond = Condition::Or(
        Box::new(Condition::And(
            Box::new(Condition::Gt("val".to_string(), Value::Int(30))),
            Box::new(Condition::Lt("val".to_string(), Value::Int(50))),
        )),
        Box::new(Condition::Eq("val".to_string(), Value::Int(100))),
    );
    assert!(cond.evaluate_tensor(&tensor));
}

#[test]
fn test_concurrent_index_add_same_value() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::thread;

    let engine = Arc::new(RelationalEngine::new());
    let schema = Schema::new(vec![Column::new("category", ColumnType::String)]);
    engine.create_table("items", schema).unwrap();
    engine.create_index("items", "category").unwrap();

    let success = Arc::new(AtomicUsize::new(0));

    // 100 threads all inserting rows with same category value
    let handles: Vec<_> = (0..100)
        .map(|_| {
            let eng = Arc::clone(&engine);
            let s = Arc::clone(&success);
            thread::spawn(move || {
                let row = HashMap::from([(
                    "category".to_string(),
                    Value::String("same_category".to_string()),
                )]);
                if eng.insert("items", row).is_ok() {
                    s.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(success.load(Ordering::SeqCst), 100);

    // Verify ALL 100 rows are findable via index
    let results = engine
        .select(
            "items",
            Condition::Eq(
                "category".to_string(),
                Value::String("same_category".to_string()),
            ),
        )
        .unwrap();
    assert_eq!(results.len(), 100, "All 100 rows should be indexed");
}

#[test]
fn test_concurrent_btree_index_add_same_value() {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::thread;

    let engine = Arc::new(RelationalEngine::new());
    let schema = Schema::new(vec![Column::new("score", ColumnType::Int)]);
    engine.create_table("scores", schema).unwrap();
    engine.create_btree_index("scores", "score").unwrap();

    let success = Arc::new(AtomicUsize::new(0));

    // 100 threads all inserting rows with same score value
    let handles: Vec<_> = (0..100)
        .map(|_| {
            let eng = Arc::clone(&engine);
            let s = Arc::clone(&success);
            thread::spawn(move || {
                let row = HashMap::from([("score".to_string(), Value::Int(42))]);
                if eng.insert("scores", row).is_ok() {
                    s.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(success.load(Ordering::SeqCst), 100);

    // Verify ALL 100 rows are findable via btree index
    let results = engine
        .select("scores", Condition::Eq("score".to_string(), Value::Int(42)))
        .unwrap();
    assert_eq!(results.len(), 100, "All 100 rows should be indexed");
}

#[test]
fn test_concurrent_index_add_remove() {
    use std::sync::Arc;
    use std::thread;

    let engine = Arc::new(RelationalEngine::new());
    let schema = Schema::new(vec![Column::new("val", ColumnType::Int)]);
    engine.create_table("test", schema).unwrap();
    engine.create_index("test", "val").unwrap();

    // Insert initial rows
    for i in 0..50 {
        engine
            .insert("test", HashMap::from([("val".to_string(), Value::Int(i))]))
            .unwrap();
    }

    // Concurrent inserts and deletes
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let eng = Arc::clone(&engine);
            thread::spawn(move || {
                // Insert new rows
                for j in 0..5 {
                    let _ = eng.insert(
                        "test",
                        HashMap::from([("val".to_string(), Value::Int(100 + i * 5 + j))]),
                    );
                }
                // Delete some existing rows
                let _ = eng.delete_rows("test", Condition::Eq("val".to_string(), Value::Int(i)));
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Engine should remain consistent - no panics or lost data
    let count = engine.row_count("test").unwrap();
    assert!(count > 0, "Table should have rows");
}

#[test]
fn test_max_tables_limit() {
    let config = RelationalConfig::new().with_max_tables(2);
    let engine = RelationalEngine::with_config(config);

    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);

    // First two tables should succeed
    engine.create_table("table1", schema.clone()).unwrap();
    engine.create_table("table2", schema.clone()).unwrap();

    // Third table should fail
    let result = engine.create_table("table3", schema);
    assert!(matches!(
        result,
        Err(RelationalError::TooManyTables { current: 2, max: 2 })
    ));

    // After dropping a table, we should be able to create another
    engine.drop_table("table1").unwrap();
    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    engine.create_table("table3", schema).unwrap();
}

#[test]
fn test_max_indexes_per_table_limit() {
    let config = RelationalConfig::new().with_max_indexes_per_table(2);
    let engine = RelationalEngine::with_config(config);

    let schema = Schema::new(vec![
        Column::new("a", ColumnType::Int),
        Column::new("b", ColumnType::Int),
        Column::new("c", ColumnType::Int),
    ]);
    engine.create_table("test", schema).unwrap();

    // First two indexes should succeed
    engine.create_index("test", "a").unwrap();
    engine.create_btree_index("test", "b").unwrap();

    // Third index should fail
    let result = engine.create_index("test", "c");
    assert!(matches!(
        result,
        Err(RelationalError::TooManyIndexes {
            table,
            current: 2,
            max: 2
        }) if table == "test"
    ));
}

#[test]
fn test_unlimited_when_none() {
    let config = RelationalConfig::default();
    assert!(config.max_tables.is_none());
    assert!(config.max_indexes_per_table.is_none());

    let engine = RelationalEngine::with_config(config);

    // Should be able to create many tables
    for i in 0..10 {
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        engine.create_table(&format!("table{i}"), schema).unwrap();
    }

    // Should be able to create many indexes on one table
    let schema = Schema::new(vec![
        Column::new("a", ColumnType::Int),
        Column::new("b", ColumnType::Int),
        Column::new("c", ColumnType::Int),
        Column::new("d", ColumnType::Int),
        Column::new("e", ColumnType::Int),
    ]);
    engine.create_table("multi_idx", schema).unwrap();
    engine.create_index("multi_idx", "a").unwrap();
    engine.create_index("multi_idx", "b").unwrap();
    engine.create_index("multi_idx", "c").unwrap();
    engine.create_index("multi_idx", "d").unwrap();
    engine.create_index("multi_idx", "e").unwrap();
}

#[test]
fn test_select_with_timeout() {
    let config = RelationalConfig::new().with_default_timeout_ms(0);
    let engine = RelationalEngine::with_config(config);

    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    engine.create_table("test", schema).unwrap();

    // With 0ms timeout, the query should immediately timeout
    // We need to use select_with_options with an explicit 0 timeout
    let result = engine.select_with_options(
        "test",
        Condition::True,
        QueryOptions::new().with_timeout_ms(0),
    );

    // Allow a small window - the query might complete before timeout check
    // or it might timeout
    match result {
        Ok(rows) => assert!(rows.is_empty()),
        Err(RelationalError::QueryTimeout { .. }) => (),
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn test_config_validation() {
    // Valid config
    let config = RelationalConfig::new()
        .with_default_timeout_ms(1000)
        .with_max_timeout_ms(5000);
    assert!(config.validate().is_ok());

    // Invalid config: default > max
    let config = RelationalConfig::new()
        .with_default_timeout_ms(10000)
        .with_max_timeout_ms(5000);
    assert!(config.validate().is_err());

    // Valid: no max set
    let config = RelationalConfig::new().with_default_timeout_ms(100000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_presets() {
    let high = RelationalConfig::high_throughput();
    assert!(high.max_tables.is_none());
    assert!(high.max_indexes_per_table.is_none());
    assert_eq!(high.max_btree_entries, 20_000_000);
    assert_eq!(high.default_query_timeout_ms, Some(30_000));

    let low = RelationalConfig::low_memory();
    assert_eq!(low.max_tables, Some(100));
    assert_eq!(low.max_indexes_per_table, Some(5));
    assert_eq!(low.max_btree_entries, 1_000_000);
    assert_eq!(low.default_query_timeout_ms, Some(10_000));
}

#[test]
fn test_new_uses_default_config() {
    let engine = RelationalEngine::new();

    // Default config should allow unlimited tables
    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    for i in 0..20 {
        engine
            .create_table(&format!("table{i}"), schema.clone())
            .unwrap();
    }
}

#[test]
fn test_concurrent_table_creation_with_limit() {
    use std::sync::Arc;
    use std::thread;

    let config = RelationalConfig::new().with_max_tables(5);
    let engine = Arc::new(RelationalEngine::with_config(config));

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let eng = Arc::clone(&engine);
            thread::spawn(move || {
                let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
                eng.create_table(&format!("table{i}"), schema)
            })
        })
        .collect();

    let mut successes = 0;
    let mut too_many = 0;
    for h in handles {
        match h.join().unwrap() {
            Ok(()) => successes += 1,
            Err(RelationalError::TooManyTables { .. }) => too_many += 1,
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    // Exactly 5 should succeed, 5 should fail
    assert_eq!(successes, 5);
    assert_eq!(too_many, 5);
}

#[test]
fn test_query_options_new_and_builder() {
    let opts = QueryOptions::new();
    assert!(opts.timeout_ms.is_none());

    let opts = QueryOptions::new().with_timeout_ms(5000);
    assert_eq!(opts.timeout_ms, Some(5000));

    let opts = QueryOptions::default();
    assert!(opts.timeout_ms.is_none());
}

#[test]
fn test_config_accessor_methods() {
    let config = RelationalConfig::new()
        .with_max_tables(100)
        .with_max_indexes_per_table(10)
        .with_default_timeout_ms(5000)
        .with_max_timeout_ms(60000)
        .with_max_btree_entries(5_000_000);

    assert_eq!(config.max_tables, Some(100));
    assert_eq!(config.max_indexes_per_table, Some(10));
    assert_eq!(config.default_query_timeout_ms, Some(5000));
    assert_eq!(config.max_query_timeout_ms, Some(60000));
    assert_eq!(config.max_btree_entries, 5_000_000);

    let engine = RelationalEngine::with_config(config);
    let engine_config = engine.config();
    assert_eq!(engine_config.max_tables, Some(100));
}

#[test]
fn test_table_count_tracking() {
    let config = RelationalConfig::new().with_max_tables(10);
    let engine = RelationalEngine::with_config(config);

    assert_eq!(engine.table_count(), 0);

    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    engine.create_table("t1", schema.clone()).unwrap();
    assert_eq!(engine.table_count(), 1);

    engine.create_table("t2", schema).unwrap();
    assert_eq!(engine.table_count(), 2);

    engine.drop_table("t1").unwrap();
    assert_eq!(engine.table_count(), 1);

    engine.drop_table("t2").unwrap();
    assert_eq!(engine.table_count(), 0);
}

#[test]
fn test_timeout_clamps_to_max() {
    let config = RelationalConfig::new()
        .with_default_timeout_ms(1000)
        .with_max_timeout_ms(500);
    let engine = RelationalEngine::with_config(config);

    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    engine.create_table("test", schema).unwrap();

    // The effective timeout should be clamped to 500ms even though
    // options request 1000ms
    let resolved = engine.resolve_timeout(QueryOptions::new().with_timeout_ms(1000));
    assert_eq!(resolved, Some(500));
}

// ========== All-NULL Aggregate Semantics Tests ==========

#[test]
fn test_aggregate_sum_all_nulls() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int).nullable(),
    ]);
    engine.create_table("sum_nulls", schema).unwrap();

    // Insert rows with all NULL values in the value column
    engine
        .insert(
            "sum_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "sum_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();

    // sum treats NULL as 0.0, so all NULLs => 0.0
    let result = engine.sum("sum_nulls", "value", Condition::True).unwrap();
    assert!((result - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_aggregate_avg_all_nulls() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Float).nullable(),
    ]);
    engine.create_table("avg_nulls", schema).unwrap();

    engine
        .insert(
            "avg_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "avg_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();

    // avg returns None when count is 0 (all values are NULL)
    let result = engine.avg("avg_nulls", "value", Condition::True).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_aggregate_min_all_nulls() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int).nullable(),
    ]);
    engine.create_table("min_nulls", schema).unwrap();

    engine
        .insert(
            "min_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "min_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();

    // min filters out NULLs, so returns None when all values are NULL
    let result = engine.min("min_nulls", "value", Condition::True).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_aggregate_max_all_nulls() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int).nullable(),
    ]);
    engine.create_table("max_nulls", schema).unwrap();

    engine
        .insert(
            "max_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "max_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();

    // max filters out NULLs, so returns None when all values are NULL
    let result = engine.max("max_nulls", "value", Condition::True).unwrap();
    assert!(result.is_none());
}

#[test]
fn test_aggregate_count_column_all_nulls() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int).nullable(),
    ]);
    engine.create_table("cnt_nulls", schema).unwrap();

    engine
        .insert(
            "cnt_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "cnt_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();

    // count_column excludes NULLs, so returns 0 when all values are NULL
    let result = engine
        .count_column("cnt_nulls", "value", Condition::True)
        .unwrap();
    assert_eq!(result, 0);
}

#[test]
fn test_aggregate_mixed_nulls_and_values() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int).nullable(),
    ]);
    engine.create_table("mixed_nulls", schema).unwrap();

    engine
        .insert(
            "mixed_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Int(10)),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "mixed_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("value".to_string(), Value::Null),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "mixed_nulls",
            HashMap::from([
                ("id".to_string(), Value::Int(3)),
                ("value".to_string(), Value::Int(20)),
            ]),
        )
        .unwrap();

    // sum treats NULL as 0, so 10 + 0 + 20 = 30
    let sum = engine.sum("mixed_nulls", "value", Condition::True).unwrap();
    assert!((sum - 30.0).abs() < f64::EPSILON);

    // avg only counts non-NULL values: (10 + 20) / 2 = 15
    let avg = engine.avg("mixed_nulls", "value", Condition::True).unwrap();
    assert!((avg.unwrap() - 15.0).abs() < f64::EPSILON);

    // min/max skip NULLs
    let min = engine.min("mixed_nulls", "value", Condition::True).unwrap();
    assert_eq!(min, Some(Value::Int(10)));

    let max = engine.max("mixed_nulls", "value", Condition::True).unwrap();
    assert_eq!(max, Some(Value::Int(20)));

    // count_column excludes NULLs
    let cnt = engine
        .count_column("mixed_nulls", "value", Condition::True)
        .unwrap();
    assert_eq!(cnt, 2);
}

// ========== Float Infinity B-tree Index Tests ==========

#[test]
fn test_btree_index_float_infinity() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Float),
    ]);
    engine.create_table("float_inf", schema).unwrap();
    engine.create_btree_index("float_inf", "value").unwrap();

    // Insert special float values
    engine
        .insert(
            "float_inf",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Float(f64::INFINITY)),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "float_inf",
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("value".to_string(), Value::Float(f64::NEG_INFINITY)),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "float_inf",
            HashMap::from([
                ("id".to_string(), Value::Int(3)),
                ("value".to_string(), Value::Float(0.0)),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "float_inf",
            HashMap::from([
                ("id".to_string(), Value::Int(4)),
                ("value".to_string(), Value::Float(100.0)),
            ]),
        )
        .unwrap();

    // Greater than 0.0 should return INFINITY and 100.0
    let rows = engine
        .select(
            "float_inf",
            Condition::Gt("value".to_string(), Value::Float(0.0)),
        )
        .unwrap();
    assert_eq!(rows.len(), 2);

    // Less than 0.0 should return NEG_INFINITY
    let rows = engine
        .select(
            "float_inf",
            Condition::Lt("value".to_string(), Value::Float(0.0)),
        )
        .unwrap();
    assert_eq!(rows.len(), 1);
    let val = rows[0].get("value").unwrap();
    assert!(matches!(val, Value::Float(f) if *f == f64::NEG_INFINITY));

    // INFINITY is greater than any finite value
    let rows = engine
        .select(
            "float_inf",
            Condition::Gt("value".to_string(), Value::Float(1e308)),
        )
        .unwrap();
    assert_eq!(rows.len(), 1);
    let val = rows[0].get("value").unwrap();
    assert!(matches!(val, Value::Float(f) if *f == f64::INFINITY));
}

#[test]
fn test_btree_index_nan_handling() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Float),
    ]);
    engine.create_table("float_nan", schema).unwrap();
    engine.create_btree_index("float_nan", "value").unwrap();

    // Insert NaN and regular values
    engine
        .insert(
            "float_nan",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Float(f64::NAN)),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "float_nan",
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("value".to_string(), Value::Float(1.0)),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "float_nan",
            HashMap::from([
                ("id".to_string(), Value::Int(3)),
                ("value".to_string(), Value::Float(2.0)),
            ]),
        )
        .unwrap();

    // All rows should be retrievable
    let all_rows = engine.select("float_nan", Condition::True).unwrap();
    assert_eq!(all_rows.len(), 3);

    // Range queries should work with finite values
    let rows = engine
        .select(
            "float_nan",
            Condition::Gt("value".to_string(), Value::Float(0.0)),
        )
        .unwrap();
    // NaN comparisons are false, so only 1.0 and 2.0 match
    assert!(rows.len() >= 2);
}

#[test]
fn test_btree_index_infinity_range_queries() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Float),
    ]);
    engine.create_table("inf_range", schema).unwrap();
    engine.create_btree_index("inf_range", "value").unwrap();

    // Insert: -inf, -100, 0, 100, +inf
    let values = [f64::NEG_INFINITY, -100.0, 0.0, 100.0, f64::INFINITY];
    for (i, v) in values.iter().enumerate() {
        engine
            .insert(
                "inf_range",
                HashMap::from([
                    ("id".to_string(), Value::Int(i as i64)),
                    ("value".to_string(), Value::Float(*v)),
                ]),
            )
            .unwrap();
    }

    // Ge -100 should include -100, 0, 100, +inf (4 rows)
    let rows = engine
        .select(
            "inf_range",
            Condition::Ge("value".to_string(), Value::Float(-100.0)),
        )
        .unwrap();
    assert_eq!(rows.len(), 4);

    // Le 100 should include -inf, -100, 0, 100 (4 rows)
    let rows = engine
        .select(
            "inf_range",
            Condition::Le("value".to_string(), Value::Float(100.0)),
        )
        .unwrap();
    assert_eq!(rows.len(), 4);

    // Between -inf and +inf should include all (using And)
    let rows = engine
        .select(
            "inf_range",
            Condition::And(
                Box::new(Condition::Ge(
                    "value".to_string(),
                    Value::Float(f64::NEG_INFINITY),
                )),
                Box::new(Condition::Le(
                    "value".to_string(),
                    Value::Float(f64::INFINITY),
                )),
            ),
        )
        .unwrap();
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_btree_index_negative_zero() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Float),
    ]);
    engine.create_table("neg_zero", schema).unwrap();
    engine.create_btree_index("neg_zero", "value").unwrap();

    // Insert positive zero and negative zero
    engine
        .insert(
            "neg_zero",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Float(0.0)),
            ]),
        )
        .unwrap();
    engine
        .insert(
            "neg_zero",
            HashMap::from([
                ("id".to_string(), Value::Int(2)),
                ("value".to_string(), Value::Float(-0.0)),
            ]),
        )
        .unwrap();

    // IEEE 754: 0.0 == -0.0
    let rows = engine
        .select(
            "neg_zero",
            Condition::Eq("value".to_string(), Value::Float(0.0)),
        )
        .unwrap();
    assert_eq!(rows.len(), 2);
}

// ========== Timeout Enforcement Tests ==========

#[test]
fn test_query_timeout_select_with_options() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    engine.create_table("timeout_sel", schema).unwrap();

    // Insert enough rows to make the query non-trivial
    for i in 0..100 {
        engine
            .insert(
                "timeout_sel",
                HashMap::from([("id".to_string(), Value::Int(i))]),
            )
            .unwrap();
    }

    // Test with a reasonable timeout - should succeed
    let options = QueryOptions::default().with_timeout_ms(10000);
    let result = engine.select_with_options("timeout_sel", Condition::True, options);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 100);
}

#[test]
fn test_query_timeout_update_with_options() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("val", ColumnType::Int),
    ]);
    engine.create_table("timeout_upd", schema).unwrap();

    for i in 0..50 {
        engine
            .insert(
                "timeout_upd",
                HashMap::from([
                    ("id".to_string(), Value::Int(i)),
                    ("val".to_string(), Value::Int(i)),
                ]),
            )
            .unwrap();
    }

    // Update with reasonable timeout - should succeed
    let options = QueryOptions::default().with_timeout_ms(10000);
    let result = engine.update_with_options(
        "timeout_upd",
        Condition::Lt("id".to_string(), Value::Int(25)),
        HashMap::from([("val".to_string(), Value::Int(999))]),
        options,
    );
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 25);
}

#[test]
fn test_query_timeout_delete_with_options() {
    let engine = RelationalEngine::new();
    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    engine.create_table("timeout_del", schema).unwrap();

    for i in 0..50 {
        engine
            .insert(
                "timeout_del",
                HashMap::from([("id".to_string(), Value::Int(i))]),
            )
            .unwrap();
    }

    // Delete with reasonable timeout - should succeed
    let options = QueryOptions::default().with_timeout_ms(10000);
    let result = engine.delete_rows_with_options(
        "timeout_del",
        Condition::Lt("id".to_string(), Value::Int(25)),
        options,
    );
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 25);

    // Verify remaining rows
    let remaining = engine.count("timeout_del", Condition::True).unwrap();
    assert_eq!(remaining, 25);
}

#[test]
fn test_config_validate_passes_valid() {
    let config = RelationalConfig::new()
        .with_default_timeout_ms(1000)
        .with_max_timeout_ms(5000);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validate_fails_invalid() {
    let config = RelationalConfig::new()
        .with_default_timeout_ms(10000)
        .with_max_timeout_ms(5000);
    let result = config.validate();
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .contains("default_query_timeout_ms (10000) exceeds max_query_timeout_ms (5000)"));
}

#[test]
fn test_transaction_timeout_config() {
    let config = RelationalConfig::new()
        .with_transaction_timeout_secs(60)
        .with_lock_timeout_secs(30);
    assert_eq!(config.transaction_timeout_secs, 60);
    assert_eq!(config.lock_timeout_secs, 30);

    let engine = RelationalEngine::with_config(config);
    assert_eq!(engine.config().transaction_timeout_secs, 60);
    assert_eq!(engine.config().lock_timeout_secs, 30);
}

#[test]
fn test_timeout_none_uses_default() {
    let config = RelationalConfig::new().with_default_timeout_ms(2000);
    let engine = RelationalEngine::with_config(config);

    // When no timeout specified in options, should use default
    let resolved = engine.resolve_timeout(QueryOptions::new());
    assert_eq!(resolved, Some(2000));
}

#[test]
fn test_timeout_explicit_overrides_default() {
    let config = RelationalConfig::new().with_default_timeout_ms(2000);
    let engine = RelationalEngine::with_config(config);

    // Explicit timeout should override default
    let resolved = engine.resolve_timeout(QueryOptions::new().with_timeout_ms(5000));
    assert_eq!(resolved, Some(5000));
}
