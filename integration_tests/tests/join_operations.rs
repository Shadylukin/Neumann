// SPDX-License-Identifier: MIT OR Apache-2.0
//! Relational JOIN operation integration tests.
//!
//! Tests hash-based JOIN functionality across tables.

use std::collections::HashMap;

use integration_tests::create_shared_engines;
use relational_engine::{Column, ColumnType, Schema, Value};

#[test]
fn test_basic_inner_join() {
    let (_, relational, _, _) = create_shared_engines();

    // Create users table
    let users_schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("users", users_schema).unwrap();

    // Create orders table
    let orders_schema = Schema::new(vec![
        Column::new("order_id", ColumnType::Int),
        Column::new("user_id", ColumnType::Int),
        Column::new("amount", ColumnType::Float),
    ]);
    relational.create_table("orders", orders_schema).unwrap();

    // Insert users
    for (id, name) in [(1, "Alice"), (2, "Bob"), (3, "Carol")] {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(id));
        row.insert("name".to_string(), Value::String(name.to_string()));
        relational.insert("users", row).unwrap();
    }

    // Insert orders
    for (order_id, user_id, amount) in [(101, 1, 50.0), (102, 1, 75.0), (103, 2, 100.0)] {
        let mut row = HashMap::new();
        row.insert("order_id".to_string(), Value::Int(order_id));
        row.insert("user_id".to_string(), Value::Int(user_id));
        row.insert("amount".to_string(), Value::Float(amount));
        relational.insert("orders", row).unwrap();
    }

    // Join users with orders on id = user_id
    let results = relational.join("users", "orders", "id", "user_id").unwrap();

    // Should have 3 matches: Alice-101, Alice-102, Bob-103
    assert_eq!(results.len(), 3);

    // Verify Alice has 2 orders
    let alice_orders: Vec<_> = results
        .iter()
        .filter(|(user, _)| match user.get("name") {
            Some(Value::String(s)) => s == "Alice",
            _ => false,
        })
        .collect();
    assert_eq!(alice_orders.len(), 2);

    // Carol should have no orders
    let carol_orders: Vec<_> = results
        .iter()
        .filter(|(user, _)| match user.get("name") {
            Some(Value::String(s)) => s == "Carol",
            _ => false,
        })
        .collect();
    assert_eq!(carol_orders.len(), 0);
}

#[test]
fn test_join_empty_tables() {
    let (_, relational, _, _) = create_shared_engines();

    // Create empty tables
    let schema_a = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    let schema_b = Schema::new(vec![Column::new("ref_id", ColumnType::Int)]);

    relational.create_table("empty_a", schema_a).unwrap();
    relational.create_table("empty_b", schema_b).unwrap();

    let results = relational
        .join("empty_a", "empty_b", "id", "ref_id")
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_join_no_matches() {
    let (_, relational, _, _) = create_shared_engines();

    let schema_a = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::String),
    ]);
    let schema_b = Schema::new(vec![
        Column::new("ref_id", ColumnType::Int),
        Column::new("data", ColumnType::String),
    ]);

    relational.create_table("table_a", schema_a).unwrap();
    relational.create_table("table_b", schema_b).unwrap();

    // Insert non-overlapping IDs
    for id in 1..=5 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(id));
        row.insert("value".to_string(), Value::String(format!("a{}", id)));
        relational.insert("table_a", row).unwrap();
    }

    for id in 10..=15 {
        let mut row = HashMap::new();
        row.insert("ref_id".to_string(), Value::Int(id));
        row.insert("data".to_string(), Value::String(format!("b{}", id)));
        relational.insert("table_b", row).unwrap();
    }

    let results = relational
        .join("table_a", "table_b", "id", "ref_id")
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_join_one_to_many() {
    let (_, relational, _, _) = create_shared_engines();

    // Create departments (one)
    let dept_schema = Schema::new(vec![
        Column::new("dept_id", ColumnType::Int),
        Column::new("dept_name", ColumnType::String),
    ]);
    relational.create_table("departments", dept_schema).unwrap();

    // Create employees (many)
    let emp_schema = Schema::new(vec![
        Column::new("emp_id", ColumnType::Int),
        Column::new("emp_name", ColumnType::String),
        Column::new("dept_id", ColumnType::Int),
    ]);
    relational.create_table("employees", emp_schema).unwrap();

    // Insert departments
    for (id, name) in [(1, "Engineering"), (2, "Sales")] {
        let mut row = HashMap::new();
        row.insert("dept_id".to_string(), Value::Int(id));
        row.insert("dept_name".to_string(), Value::String(name.to_string()));
        relational.insert("departments", row).unwrap();
    }

    // Insert employees
    let employees = [
        (101, "Alice", 1),
        (102, "Bob", 1),
        (103, "Carol", 1),
        (104, "Dave", 2),
        (105, "Eve", 2),
    ];
    for (id, name, dept) in employees {
        let mut row = HashMap::new();
        row.insert("emp_id".to_string(), Value::Int(id));
        row.insert("emp_name".to_string(), Value::String(name.to_string()));
        row.insert("dept_id".to_string(), Value::Int(dept));
        relational.insert("employees", row).unwrap();
    }

    // Join departments to employees
    let results = relational
        .join("departments", "employees", "dept_id", "dept_id")
        .unwrap();

    // Should have 5 results (one per employee)
    assert_eq!(results.len(), 5);

    // Count employees per department
    let eng_count = results
        .iter()
        .filter(|(dept, _)| match dept.get("dept_name") {
            Some(Value::String(s)) => s == "Engineering",
            _ => false,
        })
        .count();
    assert_eq!(eng_count, 3);

    let sales_count = results
        .iter()
        .filter(|(dept, _)| match dept.get("dept_name") {
            Some(Value::String(s)) => s == "Sales",
            _ => false,
        })
        .count();
    assert_eq!(sales_count, 2);
}

#[test]
fn test_join_many_to_many() {
    let (_, relational, _, _) = create_shared_engines();

    // Create students
    let students_schema = Schema::new(vec![
        Column::new("student_id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational
        .create_table("students", students_schema)
        .unwrap();

    // Create enrollments (junction table)
    let enroll_schema = Schema::new(vec![
        Column::new("student_id", ColumnType::Int),
        Column::new("course_id", ColumnType::Int),
    ]);
    relational
        .create_table("enrollments", enroll_schema)
        .unwrap();

    // Insert students
    for (id, name) in [(1, "Alice"), (2, "Bob")] {
        let mut row = HashMap::new();
        row.insert("student_id".to_string(), Value::Int(id));
        row.insert("name".to_string(), Value::String(name.to_string()));
        relational.insert("students", row).unwrap();
    }

    // Insert enrollments (Alice in 2 courses, Bob in 1)
    for (student, course) in [(1, 101), (1, 102), (2, 101)] {
        let mut row = HashMap::new();
        row.insert("student_id".to_string(), Value::Int(student));
        row.insert("course_id".to_string(), Value::Int(course));
        relational.insert("enrollments", row).unwrap();
    }

    let results = relational
        .join("students", "enrollments", "student_id", "student_id")
        .unwrap();

    assert_eq!(results.len(), 3);
}

#[test]
fn test_join_with_string_keys() {
    let (_, relational, _, _) = create_shared_engines();

    let products_schema = Schema::new(vec![
        Column::new("sku", ColumnType::String),
        Column::new("name", ColumnType::String),
    ]);
    relational
        .create_table("products", products_schema)
        .unwrap();

    let inventory_schema = Schema::new(vec![
        Column::new("sku", ColumnType::String),
        Column::new("quantity", ColumnType::Int),
    ]);
    relational
        .create_table("inventory", inventory_schema)
        .unwrap();

    // Insert products
    for (sku, name) in [("SKU-001", "Widget"), ("SKU-002", "Gadget")] {
        let mut row = HashMap::new();
        row.insert("sku".to_string(), Value::String(sku.to_string()));
        row.insert("name".to_string(), Value::String(name.to_string()));
        relational.insert("products", row).unwrap();
    }

    // Insert inventory
    for (sku, qty) in [("SKU-001", 100), ("SKU-002", 50)] {
        let mut row = HashMap::new();
        row.insert("sku".to_string(), Value::String(sku.to_string()));
        row.insert("quantity".to_string(), Value::Int(qty));
        relational.insert("inventory", row).unwrap();
    }

    let results = relational
        .join("products", "inventory", "sku", "sku")
        .unwrap();

    assert_eq!(results.len(), 2);

    // Verify join correctness
    for (product, inv) in &results {
        let prod_sku = match product.get("sku") {
            Some(Value::String(s)) => s.clone(),
            _ => panic!("Expected sku"),
        };
        let inv_sku = match inv.get("sku") {
            Some(Value::String(s)) => s.clone(),
            _ => panic!("Expected sku"),
        };
        assert_eq!(prod_sku, inv_sku);
    }
}

#[test]
fn test_join_large_tables() {
    let (_, relational, _, _) = create_shared_engines();

    let left_schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("data", ColumnType::String),
    ]);
    let right_schema = Schema::new(vec![
        Column::new("ref_id", ColumnType::Int),
        Column::new("value", ColumnType::Int),
    ]);

    relational.create_table("left_table", left_schema).unwrap();
    relational
        .create_table("right_table", right_schema)
        .unwrap();

    // Insert 1000 rows in left table
    for i in 0..1000 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("data".to_string(), Value::String(format!("data{}", i)));
        relational.insert("left_table", row).unwrap();
    }

    // Insert 500 rows in right table (every other ID)
    for i in (0..1000).step_by(2) {
        let mut row = HashMap::new();
        row.insert("ref_id".to_string(), Value::Int(i));
        row.insert("value".to_string(), Value::Int(i * 10));
        relational.insert("right_table", row).unwrap();
    }

    let results = relational
        .join("left_table", "right_table", "id", "ref_id")
        .unwrap();

    // Should have 500 matches
    assert_eq!(results.len(), 500);
}

#[test]
fn test_join_duplicate_keys_right() {
    let (_, relational, _, _) = create_shared_engines();

    let master_schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    let detail_schema = Schema::new(vec![
        Column::new("master_id", ColumnType::Int),
        Column::new("detail", ColumnType::String),
    ]);

    relational.create_table("master", master_schema).unwrap();
    relational.create_table("detail", detail_schema).unwrap();

    // Insert single master row
    let mut row = HashMap::new();
    row.insert("id".to_string(), Value::Int(1));
    row.insert("name".to_string(), Value::String("Master".to_string()));
    relational.insert("master", row).unwrap();

    // Insert multiple detail rows with same master_id
    for i in 0..10 {
        let mut row = HashMap::new();
        row.insert("master_id".to_string(), Value::Int(1));
        row.insert("detail".to_string(), Value::String(format!("Detail{}", i)));
        relational.insert("detail", row).unwrap();
    }

    let results = relational
        .join("master", "detail", "id", "master_id")
        .unwrap();

    // Should have 10 results (1 master x 10 details)
    assert_eq!(results.len(), 10);
}

#[test]
fn test_join_nonexistent_table() {
    let (_, relational, _, _) = create_shared_engines();

    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    relational.create_table("existing", schema).unwrap();

    let result = relational.join("existing", "nonexistent", "id", "id");
    assert!(result.is_err());

    let result = relational.join("nonexistent", "existing", "id", "id");
    assert!(result.is_err());
}

#[test]
fn test_join_preserves_all_columns() {
    let (_, relational, _, _) = create_shared_engines();

    let left_schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("left_col1", ColumnType::String),
        Column::new("left_col2", ColumnType::Float),
    ]);
    let right_schema = Schema::new(vec![
        Column::new("ref_id", ColumnType::Int),
        Column::new("right_col1", ColumnType::String),
        Column::new("right_col2", ColumnType::Bool),
    ]);

    relational.create_table("left", left_schema).unwrap();
    relational.create_table("right", right_schema).unwrap();

    let mut left_row = HashMap::new();
    left_row.insert("id".to_string(), Value::Int(1));
    left_row.insert(
        "left_col1".to_string(),
        Value::String("left_value".to_string()),
    );
    left_row.insert("left_col2".to_string(), Value::Float(3.14));
    relational.insert("left", left_row).unwrap();

    let mut right_row = HashMap::new();
    right_row.insert("ref_id".to_string(), Value::Int(1));
    right_row.insert(
        "right_col1".to_string(),
        Value::String("right_value".to_string()),
    );
    right_row.insert("right_col2".to_string(), Value::Bool(true));
    relational.insert("right", right_row).unwrap();

    let results = relational.join("left", "right", "id", "ref_id").unwrap();

    assert_eq!(results.len(), 1);

    let (left_result, right_result) = &results[0];

    // Verify all columns from left table
    assert!(left_result.contains("id"));
    assert!(left_result.contains("left_col1"));
    assert!(left_result.contains("left_col2"));

    // Verify all columns from right table
    assert!(right_result.contains("ref_id"));
    assert!(right_result.contains("right_col1"));
    assert!(right_result.contains("right_col2"));
}
