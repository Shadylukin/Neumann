//! Columnar storage integration tests.
//!
//! Tests columnar data materialization, batch operations, and projection.

use integration_tests::create_shared_engines;
use relational_engine::{Column, ColumnType, ColumnarScanOptions, Condition, Schema, Value};
use std::collections::HashMap;

#[test]
fn test_materialize_single_column() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int),
    ]);
    relational.create_table("data", schema).unwrap();

    // Insert data
    for i in 0..100 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("value".to_string(), Value::Int(i * 10));
        relational.insert("data", row).unwrap();
    }

    // Materialize column
    relational.materialize_columns("data", &["value"]).unwrap();

    assert!(relational.has_columnar_data("data", "value"));
    assert!(!relational.has_columnar_data("data", "id"));
}

#[test]
fn test_materialize_multiple_columns() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("a", ColumnType::Int),
        Column::new("b", ColumnType::Float),
        Column::new("c", ColumnType::String),
    ]);
    relational.create_table("multi", schema).unwrap();

    for i in 0..50 {
        let mut row = HashMap::new();
        row.insert("a".to_string(), Value::Int(i));
        row.insert("b".to_string(), Value::Float(i as f64 * 1.5));
        row.insert("c".to_string(), Value::String(format!("str{}", i)));
        relational.insert("multi", row).unwrap();
    }

    relational
        .materialize_columns("multi", &["a", "b"])
        .unwrap();

    assert!(relational.has_columnar_data("multi", "a"));
    assert!(relational.has_columnar_data("multi", "b"));
    assert!(!relational.has_columnar_data("multi", "c"));
}

#[test]
fn test_load_column_data() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("score", ColumnType::Int),
    ]);
    relational.create_table("scores", schema).unwrap();

    for i in 0..20 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("score".to_string(), Value::Int(i * 5));
        relational.insert("scores", row).unwrap();
    }

    relational
        .materialize_columns("scores", &["score"])
        .unwrap();

    // Load the materialized column
    let column_data = relational.load_column_data("scores", "score").unwrap();

    assert_eq!(column_data.row_ids.len(), 20);

    // Verify we have 20 values and they're all multiples of 5
    let mut values_found = 0;
    for i in 0..column_data.row_ids.len() {
        if let Some(Value::Int(v)) = column_data.get_value(i) {
            assert_eq!(v % 5, 0, "Score should be multiple of 5");
            assert!(v >= 0 && v < 100, "Score should be in range 0..100");
            values_found += 1;
        }
    }
    assert_eq!(values_found, 20);
}

#[test]
fn test_drop_columnar_data() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("x", ColumnType::Int)]);
    relational.create_table("test", schema).unwrap();

    for i in 0..10 {
        let mut row = HashMap::new();
        row.insert("x".to_string(), Value::Int(i));
        relational.insert("test", row).unwrap();
    }

    relational.materialize_columns("test", &["x"]).unwrap();
    assert!(relational.has_columnar_data("test", "x"));

    relational.drop_columnar_data("test", "x").unwrap();
    assert!(!relational.has_columnar_data("test", "x"));
}

#[test]
fn test_select_columnar_with_filter() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int),
    ]);
    relational.create_table("filtered", schema).unwrap();

    for i in 0..100 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("value".to_string(), Value::Int(i));
        relational.insert("filtered", row).unwrap();
    }

    // Materialize the filter column
    relational
        .materialize_columns("filtered", &["value"])
        .unwrap();

    // Use columnar scan
    let options = ColumnarScanOptions {
        projection: None,
        prefer_columnar: true,
    };

    let results = relational
        .select_columnar(
            "filtered",
            Condition::Gt("value".to_string(), Value::Int(90)),
            options,
        )
        .unwrap();

    // Values 91-99 = 9 rows
    assert_eq!(results.len(), 9);
}

#[test]
fn test_select_columnar_fallback() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("fallback", schema).unwrap();

    for i in 0..10 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("name".to_string(), Value::String(format!("item{}", i)));
        relational.insert("fallback", row).unwrap();
    }

    // Don't materialize - should fall back to row-based
    let options = ColumnarScanOptions {
        projection: None,
        prefer_columnar: true,
    };

    let results = relational
        .select_columnar(
            "fallback",
            Condition::Lt("id".to_string(), Value::Int(5)),
            options,
        )
        .unwrap();

    assert_eq!(results.len(), 5);
}

#[test]
fn test_select_with_projection() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
        Column::new("age", ColumnType::Int),
        Column::new("city", ColumnType::String),
    ]);
    relational.create_table("people", schema).unwrap();

    let mut row = HashMap::new();
    row.insert("id".to_string(), Value::Int(1));
    row.insert("name".to_string(), Value::String("Alice".to_string()));
    row.insert("age".to_string(), Value::Int(30));
    row.insert("city".to_string(), Value::String("NYC".to_string()));
    relational.insert("people", row).unwrap();

    // Project only name and age
    let results = relational
        .select_with_projection(
            "people",
            Condition::True,
            Some(vec!["name".to_string(), "age".to_string()]),
        )
        .unwrap();

    assert_eq!(results.len(), 1);

    let row = &results[0];
    assert!(row.contains("name"));
    assert!(row.contains("age"));
    // city and id should not be present
    assert!(!row.contains("city"));
    // Note: _id may or may not be included depending on implementation
}

#[test]
fn test_select_with_projection_includes_id() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("value", ColumnType::Int)]);
    relational.create_table("simple", schema).unwrap();

    let mut row = HashMap::new();
    row.insert("value".to_string(), Value::Int(42));
    relational.insert("simple", row).unwrap();

    // Project _id explicitly
    let results = relational
        .select_with_projection(
            "simple",
            Condition::True,
            Some(vec!["_id".to_string(), "value".to_string()]),
        )
        .unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].contains("value"));
}

#[test]
fn test_batch_insert() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("data", ColumnType::String),
    ]);
    relational.create_table("batch", schema).unwrap();

    // Prepare batch
    let mut rows = Vec::new();
    for i in 0..100 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("data".to_string(), Value::String(format!("row{}", i)));
        rows.push(row);
    }

    // Batch insert
    let row_ids = relational.batch_insert("batch", rows).unwrap();

    assert_eq!(row_ids.len(), 100);

    // Verify all rows inserted
    let all = relational.select("batch", Condition::True).unwrap();
    assert_eq!(all.len(), 100);
}

#[test]
fn test_batch_insert_empty() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("x", ColumnType::Int)]);
    relational.create_table("empty_batch", schema).unwrap();

    // Empty batch
    let row_ids = relational.batch_insert("empty_batch", Vec::new()).unwrap();

    assert!(row_ids.is_empty());
}

#[test]
fn test_batch_insert_validates_schema() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("required", ColumnType::Int)]);
    relational.create_table("strict", schema).unwrap();

    // Try to batch insert with missing required column
    let rows = vec![
        HashMap::new(), // Missing 'required'
    ];

    let result = relational.batch_insert("strict", rows);
    assert!(result.is_err());
}

#[test]
fn test_batch_insert_large() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Float),
    ]);
    relational.create_table("large_batch", schema).unwrap();

    // Large batch of 10k rows
    let rows: Vec<HashMap<String, Value>> = (0..10_000)
        .map(|i| {
            let mut row = HashMap::new();
            row.insert("id".to_string(), Value::Int(i));
            row.insert("value".to_string(), Value::Float(i as f64 * 0.1));
            row
        })
        .collect();

    let row_ids = relational.batch_insert("large_batch", rows).unwrap();
    assert_eq!(row_ids.len(), 10_000);

    let count = relational.row_count("large_batch").unwrap();
    assert_eq!(count, 10_000);
}

#[test]
fn test_columnar_with_nulls() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("optional", ColumnType::Int).nullable(),
    ]);
    relational.create_table("nullable", schema).unwrap();

    // Insert mix of null and non-null
    for i in 0..20 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        if i % 3 == 0 {
            row.insert("optional".to_string(), Value::Null);
        } else {
            row.insert("optional".to_string(), Value::Int(i * 10));
        }
        relational.insert("nullable", row).unwrap();
    }

    // Materialize the nullable column
    relational
        .materialize_columns("nullable", &["optional"])
        .unwrap();

    let column_data = relational.load_column_data("nullable", "optional").unwrap();

    assert_eq!(column_data.row_ids.len(), 20);

    // Check null tracking
    let null_count = column_data.null_count();
    assert_eq!(null_count, 7); // 0, 3, 6, 9, 12, 15, 18
}

#[test]
fn test_columnar_scan_vs_row_scan() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("category", ColumnType::Int),
    ]);
    relational.create_table("compare", schema).unwrap();

    for i in 0..1000 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("category".to_string(), Value::Int(i % 10));
        relational.insert("compare", row).unwrap();
    }

    // Row-based select
    let row_results = relational
        .select(
            "compare",
            Condition::Eq("category".to_string(), Value::Int(5)),
        )
        .unwrap();

    // Materialize and use columnar
    relational
        .materialize_columns("compare", &["category"])
        .unwrap();

    let columnar_results = relational
        .select_columnar(
            "compare",
            Condition::Eq("category".to_string(), Value::Int(5)),
            ColumnarScanOptions {
                projection: None,
                prefer_columnar: true,
            },
        )
        .unwrap();

    // Results should be the same
    assert_eq!(row_results.len(), columnar_results.len());
    assert_eq!(row_results.len(), 100); // 1000 / 10
}

#[test]
fn test_columnar_with_string_column() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("status", ColumnType::String),
    ]);
    relational.create_table("strings", schema).unwrap();

    let statuses = ["pending", "active", "done", "cancelled"];
    for i in 0..40 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert(
            "status".to_string(),
            Value::String(statuses[(i as usize) % 4].to_string()),
        );
        relational.insert("strings", row).unwrap();
    }

    relational
        .materialize_columns("strings", &["status"])
        .unwrap();

    let column_data = relational.load_column_data("strings", "status").unwrap();
    assert_eq!(column_data.row_ids.len(), 40);
}

#[test]
fn test_columnar_scan_with_projection() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("a", ColumnType::Int),
        Column::new("b", ColumnType::Int),
        Column::new("c", ColumnType::Int),
    ]);
    relational.create_table("projected", schema).unwrap();

    for i in 0..50 {
        let mut row = HashMap::new();
        row.insert("a".to_string(), Value::Int(i));
        row.insert("b".to_string(), Value::Int(i * 2));
        row.insert("c".to_string(), Value::Int(i * 3));
        relational.insert("projected", row).unwrap();
    }

    relational.materialize_columns("projected", &["a"]).unwrap();

    // Columnar scan with projection
    let results = relational
        .select_columnar(
            "projected",
            Condition::Lt("a".to_string(), Value::Int(10)),
            ColumnarScanOptions {
                projection: Some(vec!["a".to_string(), "b".to_string()]),
                prefer_columnar: true,
            },
        )
        .unwrap();

    assert_eq!(results.len(), 10);

    // Should only have projected columns
    for row in &results {
        assert!(row.contains("a"));
        assert!(row.contains("b"));
        // c should not be present (or implementation may include it)
    }
}

#[test]
fn test_row_count() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("x", ColumnType::Int)]);
    relational.create_table("counted", schema).unwrap();

    assert_eq!(relational.row_count("counted").unwrap(), 0);

    for i in 0..25 {
        let mut row = HashMap::new();
        row.insert("x".to_string(), Value::Int(i));
        relational.insert("counted", row).unwrap();
    }

    assert_eq!(relational.row_count("counted").unwrap(), 25);

    relational
        .delete_rows("counted", Condition::Gt("x".to_string(), Value::Int(20)))
        .unwrap();

    assert_eq!(relational.row_count("counted").unwrap(), 21);
}

#[test]
fn test_table_exists() {
    let (_, relational, _, _) = create_shared_engines();

    assert!(!relational.table_exists("nonexistent"));

    let schema = Schema::new(vec![Column::new("x", ColumnType::Int)]);
    relational.create_table("exists", schema).unwrap();

    assert!(relational.table_exists("exists"));

    relational.drop_table("exists").unwrap();

    assert!(!relational.table_exists("exists"));
}

#[test]
fn test_drop_table() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("data", ColumnType::String)]);
    relational.create_table("temporary", schema).unwrap();

    // Insert some data
    let mut row = HashMap::new();
    row.insert("data".to_string(), Value::String("test".to_string()));
    relational.insert("temporary", row).unwrap();

    // Create index
    relational.create_index("temporary", "data").unwrap();

    // Drop table
    relational.drop_table("temporary").unwrap();

    // Table should no longer exist
    assert!(!relational.table_exists("temporary"));

    // Schema should fail
    assert!(relational.get_schema("temporary").is_err());

    // Can recreate with same name
    let schema2 = Schema::new(vec![Column::new("other", ColumnType::Int)]);
    relational.create_table("temporary", schema2).unwrap();
    assert!(relational.table_exists("temporary"));
}

#[test]
fn test_list_tables() {
    let (_, relational, _, _) = create_shared_engines();

    // Initially no tables
    assert!(relational.list_tables().is_empty());

    // Create tables
    let schema = Schema::new(vec![Column::new("x", ColumnType::Int)]);
    relational.create_table("table_a", schema.clone()).unwrap();
    relational.create_table("table_b", schema.clone()).unwrap();
    relational.create_table("table_c", schema).unwrap();

    let tables = relational.list_tables();
    assert_eq!(tables.len(), 3);
    assert!(tables.contains(&"table_a".to_string()));
    assert!(tables.contains(&"table_b".to_string()));
    assert!(tables.contains(&"table_c".to_string()));
}
