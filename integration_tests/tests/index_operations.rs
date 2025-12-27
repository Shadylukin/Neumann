//! Index operation integration tests.
//!
//! Tests hash index and B-tree index functionality.

use integration_tests::create_shared_engines;
use relational_engine::{Column, ColumnType, Condition, Schema, Value};
use std::collections::HashMap;

#[test]
fn test_create_hash_index() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("users", schema).unwrap();

    // Create index on name column
    relational.create_index("users", "name").unwrap();

    assert!(relational.has_index("users", "name"));
    assert!(!relational.has_index("users", "id"));
}

#[test]
fn test_create_btree_index() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("score", ColumnType::Int),
    ]);
    relational.create_table("scores", schema).unwrap();

    // Create B-tree index on score column
    relational.create_btree_index("scores", "score").unwrap();

    assert!(relational.has_btree_index("scores", "score"));
    assert!(!relational.has_btree_index("scores", "id"));
}

#[test]
fn test_index_on_id_column() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("data", ColumnType::String),
    ]);
    relational.create_table("items", schema).unwrap();

    // Can create index on _id (internal row ID)
    relational.create_index("items", "_id").unwrap();
    assert!(relational.has_index("items", "_id"));

    // Can also create B-tree on _id
    relational.create_btree_index("items", "_id").unwrap();
    assert!(relational.has_btree_index("items", "_id"));
}

#[test]
fn test_drop_hash_index() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("value", ColumnType::Int)]);
    relational.create_table("data", schema).unwrap();

    relational.create_index("data", "value").unwrap();
    assert!(relational.has_index("data", "value"));

    relational.drop_index("data", "value").unwrap();
    assert!(!relational.has_index("data", "value"));
}

#[test]
fn test_drop_btree_index() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("value", ColumnType::Int)]);
    relational.create_table("data", schema).unwrap();

    relational.create_btree_index("data", "value").unwrap();
    assert!(relational.has_btree_index("data", "value"));

    relational.drop_btree_index("data", "value").unwrap();
    assert!(!relational.has_btree_index("data", "value"));
}

#[test]
fn test_get_indexed_columns() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("a", ColumnType::Int),
        Column::new("b", ColumnType::String),
        Column::new("c", ColumnType::Float),
    ]);
    relational.create_table("multi", schema).unwrap();

    // No indexes initially
    assert!(relational.get_indexed_columns("multi").is_empty());

    // Create indexes on a and c
    relational.create_index("multi", "a").unwrap();
    relational.create_index("multi", "c").unwrap();

    let indexed = relational.get_indexed_columns("multi");
    assert_eq!(indexed.len(), 2);
    assert!(indexed.contains(&"a".to_string()));
    assert!(indexed.contains(&"c".to_string()));
}

#[test]
fn test_get_btree_indexed_columns() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("x", ColumnType::Int),
        Column::new("y", ColumnType::Int),
    ]);
    relational.create_table("coords", schema).unwrap();

    relational.create_btree_index("coords", "x").unwrap();
    relational.create_btree_index("coords", "y").unwrap();

    let indexed = relational.get_btree_indexed_columns("coords");
    assert_eq!(indexed.len(), 2);
    assert!(indexed.contains(&"x".to_string()));
    assert!(indexed.contains(&"y".to_string()));
}

#[test]
fn test_index_duplicate_error() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("col", ColumnType::Int)]);
    relational.create_table("test", schema).unwrap();

    relational.create_index("test", "col").unwrap();

    // Creating same index again should fail
    let result = relational.create_index("test", "col");
    assert!(result.is_err());
}

#[test]
fn test_btree_index_duplicate_error() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("col", ColumnType::Int)]);
    relational.create_table("test", schema).unwrap();

    relational.create_btree_index("test", "col").unwrap();

    // Creating same B-tree index again should fail
    let result = relational.create_btree_index("test", "col");
    assert!(result.is_err());
}

#[test]
fn test_index_nonexistent_column_error() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("exists", ColumnType::Int)]);
    relational.create_table("test", schema).unwrap();

    // Creating index on non-existent column should fail
    let result = relational.create_index("test", "nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_drop_nonexistent_index_error() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("col", ColumnType::Int)]);
    relational.create_table("test", schema).unwrap();

    // Dropping non-existent index should fail
    let result = relational.drop_index("test", "col");
    assert!(result.is_err());
}

#[test]
fn test_index_with_data_insert() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("category", ColumnType::String),
    ]);
    relational.create_table("products", schema).unwrap();

    // Create index before inserting data
    relational.create_index("products", "category").unwrap();

    // Insert data - index should be updated
    for i in 0..100 {
        let category = if i % 3 == 0 {
            "A"
        } else if i % 3 == 1 {
            "B"
        } else {
            "C"
        };
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("category".to_string(), Value::String(category.to_string()));
        relational.insert("products", row).unwrap();
    }

    // Query should work (index should accelerate equality lookups)
    let results = relational
        .select(
            "products",
            Condition::Eq("category".to_string(), Value::String("A".to_string())),
        )
        .unwrap();

    // Category A appears when i % 3 == 0: 0, 3, 6, ... 99 = 34 items
    assert_eq!(results.len(), 34);
}

#[test]
fn test_btree_index_with_range_data() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("score", ColumnType::Int),
    ]);
    relational.create_table("rankings", schema).unwrap();

    // Create B-tree index for range queries
    relational.create_btree_index("rankings", "score").unwrap();

    // Insert data
    for i in 0..100 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("score".to_string(), Value::Int(i * 10));
        relational.insert("rankings", row).unwrap();
    }

    // Range query (B-tree should accelerate this)
    let results = relational
        .select(
            "rankings",
            Condition::Gt("score".to_string(), Value::Int(500)),
        )
        .unwrap();

    // Scores > 500: 510, 520, ..., 990 = 49 items
    assert_eq!(results.len(), 49);
}

#[test]
fn test_index_survives_updates() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("status", ColumnType::String),
    ]);
    relational.create_table("tasks", schema).unwrap();

    relational.create_index("tasks", "status").unwrap();

    // Insert initial data
    for i in 0..10 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("status".to_string(), Value::String("pending".to_string()));
        relational.insert("tasks", row).unwrap();
    }

    // Update some rows
    relational
        .update(
            "tasks",
            Condition::Lt("id".to_string(), Value::Int(5)),
            HashMap::from([("status".to_string(), Value::String("done".to_string()))]),
        )
        .unwrap();

    // Query by status
    let pending = relational
        .select(
            "tasks",
            Condition::Eq("status".to_string(), Value::String("pending".to_string())),
        )
        .unwrap();
    assert_eq!(pending.len(), 5);

    let done = relational
        .select(
            "tasks",
            Condition::Eq("status".to_string(), Value::String("done".to_string())),
        )
        .unwrap();
    assert_eq!(done.len(), 5);
}

#[test]
fn test_index_survives_deletes() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("type", ColumnType::String),
    ]);
    relational.create_table("items", schema).unwrap();

    relational.create_index("items", "type").unwrap();

    // Insert data
    for i in 0..20 {
        let item_type = if i % 2 == 0 { "even" } else { "odd" };
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("type".to_string(), Value::String(item_type.to_string()));
        relational.insert("items", row).unwrap();
    }

    // Delete even items
    relational
        .delete_rows(
            "items",
            Condition::Eq("type".to_string(), Value::String("even".to_string())),
        )
        .unwrap();

    // Query remaining
    let remaining = relational
        .select(
            "items",
            Condition::Eq("type".to_string(), Value::String("odd".to_string())),
        )
        .unwrap();
    assert_eq!(remaining.len(), 10);

    // Even should be empty
    let evens = relational
        .select(
            "items",
            Condition::Eq("type".to_string(), Value::String("even".to_string())),
        )
        .unwrap();
    assert!(evens.is_empty());
}

#[test]
fn test_multiple_indexes_same_table() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
        Column::new("age", ColumnType::Int),
        Column::new("city", ColumnType::String),
    ]);
    relational.create_table("people", schema).unwrap();

    // Create multiple indexes
    relational.create_index("people", "name").unwrap();
    relational.create_btree_index("people", "age").unwrap();
    relational.create_index("people", "city").unwrap();

    assert!(relational.has_index("people", "name"));
    assert!(relational.has_btree_index("people", "age"));
    assert!(relational.has_index("people", "city"));

    // Insert data
    let people = [
        (1, "Alice", 30, "NYC"),
        (2, "Bob", 25, "LA"),
        (3, "Carol", 35, "NYC"),
        (4, "Dave", 28, "Chicago"),
    ];

    for (id, name, age, city) in people {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(id));
        row.insert("name".to_string(), Value::String(name.to_string()));
        row.insert("age".to_string(), Value::Int(age));
        row.insert("city".to_string(), Value::String(city.to_string()));
        relational.insert("people", row).unwrap();
    }

    // Query using different indexed columns
    let nyc = relational
        .select(
            "people",
            Condition::Eq("city".to_string(), Value::String("NYC".to_string())),
        )
        .unwrap();
    assert_eq!(nyc.len(), 2);

    let over_27 = relational
        .select("people", Condition::Gt("age".to_string(), Value::Int(27)))
        .unwrap();
    assert_eq!(over_27.len(), 3); // Alice(30), Carol(35), Dave(28)
}

#[test]
fn test_index_with_null_values() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("optional", ColumnType::String).nullable(),
    ]);
    relational.create_table("nullable", schema).unwrap();

    relational.create_index("nullable", "optional").unwrap();

    // Insert mix of null and non-null
    for i in 0..10 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        if i % 2 == 0 {
            row.insert("optional".to_string(), Value::String("value".to_string()));
        } else {
            row.insert("optional".to_string(), Value::Null);
        }
        relational.insert("nullable", row).unwrap();
    }

    // Query non-null values
    let with_value = relational
        .select(
            "nullable",
            Condition::Eq("optional".to_string(), Value::String("value".to_string())),
        )
        .unwrap();
    assert_eq!(with_value.len(), 5);
}

#[test]
fn test_index_large_table() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("bucket", ColumnType::Int),
    ]);
    relational.create_table("large", schema).unwrap();

    relational.create_index("large", "bucket").unwrap();

    // Insert 10k rows with 100 buckets
    for i in 0..10_000 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("bucket".to_string(), Value::Int(i % 100));
        relational.insert("large", row).unwrap();
    }

    // Query single bucket
    let bucket_42 = relational
        .select("large", Condition::Eq("bucket".to_string(), Value::Int(42)))
        .unwrap();

    // Each bucket has 100 rows (10000 / 100)
    assert_eq!(bucket_42.len(), 100);
}
