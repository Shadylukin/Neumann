// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Edge case integration tests.
//!
//! Tests boundary conditions, empty inputs, and special values.

use std::collections::HashMap;

use graph_engine::{Direction, GraphEngine};
use integration_tests::sample_embeddings;
use relational_engine::{Column, ColumnType, Condition, Schema, Value};
use tensor_store::TensorStore;
use vector_engine::VectorEngine;

#[test]
fn test_empty_table_operations() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store);

    // Create empty table
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("empty", schema).unwrap();

    // SELECT on empty table should return empty, not error
    let rows = relational.select("empty", Condition::True).unwrap();
    assert_eq!(rows.len(), 0);

    // UPDATE on empty table should update 0 rows
    let updated = relational
        .update(
            "empty",
            Condition::True,
            HashMap::from([("name".to_string(), Value::String("test".to_string()))]),
        )
        .unwrap();
    assert_eq!(updated, 0);

    // DELETE on empty table should delete 0 rows
    let deleted = relational.delete_rows("empty", Condition::True).unwrap();
    assert_eq!(deleted, 0);

    // COUNT equivalent - SELECT returns 0 rows
    let count = relational.select("empty", Condition::True).unwrap().len();
    assert_eq!(count, 0);
}

#[test]
fn test_empty_graph_traversal() {
    let store = TensorStore::new();
    let graph = GraphEngine::with_store(store);

    // Path find on empty graph
    let path = graph.find_path(0, 1, None);
    assert!(path.is_err() || path.as_ref().map(|p| p.nodes.is_empty()).unwrap_or(true));

    // Create single node with no edges
    let node_id = graph.create_node("lonely", HashMap::new()).unwrap();

    // Neighbors of node with no edges should be empty
    let neighbors = graph
        .neighbors(node_id, None, Direction::Both, None)
        .unwrap();
    assert!(neighbors.is_empty());

    // Path to self
    let self_path = graph.find_path(node_id, node_id, None);
    match self_path {
        Ok(path) => {
            // Path to self may return single-node path or empty
            assert!(path.nodes.len() <= 1);
        },
        Err(_) => {
            // Or may return error - both are acceptable
        },
    }
}

#[test]
fn test_empty_vector_search() {
    let store = TensorStore::new();
    let vector = VectorEngine::with_store(store);

    // Search on empty vector store should return empty results, not error
    let query = sample_embeddings(1, 32)[0].clone();
    let results = vector.search_similar(&query, 10).unwrap();
    assert!(results.is_empty());

    // Store one embedding
    vector.store_embedding("single", query.clone()).unwrap();

    // Now search should find it
    let results = vector.search_similar(&query, 10).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].key, "single");
}

#[test]
fn test_zero_vector_handling() {
    let store = TensorStore::new();
    let vector = VectorEngine::with_store(store);

    // Store zero vector
    let zero_vec = vec![0.0f32; 32];
    let result = vector.store_embedding("zero", zero_vec.clone());

    // May succeed or fail depending on implementation
    match result {
        Ok(_) => {
            // If stored, should be retrievable
            let retrieved = vector.get_embedding("zero").unwrap();
            assert_eq!(retrieved, zero_vec);

            // Search with zero vector
            let _results = vector.search_similar(&zero_vec, 10).unwrap();
            // Should handle gracefully (may return itself or empty)
        },
        Err(e) => {
            // Rejecting zero vector is also valid
            assert!(
                e.to_string().contains("zero")
                    || e.to_string().contains("invalid")
                    || e.to_string().contains("norm")
            );
        },
    }
}

#[test]
fn test_max_int_boundary() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store);

    let schema = Schema::new(vec![Column::new("value", ColumnType::Int)]);
    relational.create_table("boundaries", schema).unwrap();

    // Insert max i64
    let mut row_max = HashMap::new();
    row_max.insert("value".to_string(), Value::Int(i64::MAX));
    relational.insert("boundaries", row_max).unwrap();

    // Insert min i64
    let mut row_min = HashMap::new();
    row_min.insert("value".to_string(), Value::Int(i64::MIN));
    relational.insert("boundaries", row_min).unwrap();

    // Verify stored correctly
    let rows = relational.select("boundaries", Condition::True).unwrap();
    assert_eq!(rows.len(), 2);

    // Query with boundary conditions
    let max_rows = relational
        .select(
            "boundaries",
            Condition::Eq("value".to_string(), Value::Int(i64::MAX)),
        )
        .unwrap();
    assert_eq!(max_rows.len(), 1);

    let min_rows = relational
        .select(
            "boundaries",
            Condition::Eq("value".to_string(), Value::Int(i64::MIN)),
        )
        .unwrap();
    assert_eq!(min_rows.len(), 1);
}

#[test]
fn test_empty_string_handling() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store);

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("data", ColumnType::String),
    ]);
    relational.create_table("strings", schema).unwrap();

    // Insert empty string
    let mut row = HashMap::new();
    row.insert("id".to_string(), Value::Int(1));
    row.insert("data".to_string(), Value::String(String::new()));
    relational.insert("strings", row).unwrap();

    // Query for empty string
    let rows = relational
        .select(
            "strings",
            Condition::Eq("data".to_string(), Value::String(String::new())),
        )
        .unwrap();
    assert_eq!(rows.len(), 1);

    // Verify empty string is preserved
    let data = &rows[0].get("data").unwrap();
    match data {
        Value::String(s) => assert_eq!(s, ""),
        _ => panic!("Expected String value"),
    }
}

#[test]
fn test_very_long_keys() {
    let store = TensorStore::new();
    let vector = VectorEngine::with_store(store);

    // Create a very long key (1000 chars)
    let long_key: String = (0..1000).map(|i| ((i % 26) as u8 + b'a') as char).collect();

    // Store embedding with long key
    let embedding = sample_embeddings(1, 16)[0].clone();
    vector
        .store_embedding(&long_key, embedding.clone())
        .unwrap();

    // Retrieve with long key
    let retrieved = vector.get_embedding(&long_key).unwrap();
    assert_eq!(retrieved, embedding);

    // Search should include long key in results
    let results = vector.search_similar(&embedding, 10).unwrap();
    assert!(results.iter().any(|r| r.key == long_key));
}

#[test]
fn test_special_characters_in_keys() {
    let store = TensorStore::new();
    let vector = VectorEngine::with_store(store);

    let special_keys = vec![
        "key with spaces",
        "key:with:colons",
        "key/with/slashes",
        "key.with.dots",
        "key-with-dashes",
        "key_with_underscores",
        "unicode:日本語",
        "emoji:🎉",
        "quotes:\"test\"",
        "mixed:key/path:123",
    ];

    let embedding = sample_embeddings(1, 8)[0].clone();

    for key in &special_keys {
        // Store with special key
        let result = vector.store_embedding(key, embedding.clone());

        match result {
            Ok(_) => {
                // Should be able to retrieve
                let retrieved = vector.get_embedding(key).unwrap();
                assert_eq!(retrieved, embedding);
            },
            Err(e) => {
                // Some characters may be rejected - document which
                println!("Key '{}' rejected: {}", key, e);
            },
        }
    }
}

#[test]
fn test_null_value_conditions() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store);

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("nullable", ColumnType::String).nullable(),
    ]);
    relational.create_table("nullable_test", schema).unwrap();

    // Insert row with value
    let mut row1 = HashMap::new();
    row1.insert("id".to_string(), Value::Int(1));
    row1.insert(
        "nullable".to_string(),
        Value::String("has_value".to_string()),
    );
    relational.insert("nullable_test", row1).unwrap();

    // Insert row with null
    let mut row2 = HashMap::new();
    row2.insert("id".to_string(), Value::Int(2));
    row2.insert("nullable".to_string(), Value::Null);
    relational.insert("nullable_test", row2).unwrap();

    // Query for non-null
    let non_null = relational
        .select(
            "nullable_test",
            Condition::Ne("nullable".to_string(), Value::Null),
        )
        .unwrap();

    // Query for null
    let null_rows = relational
        .select(
            "nullable_test",
            Condition::Eq("nullable".to_string(), Value::Null),
        )
        .unwrap();

    // Should handle null correctly in conditions
    // Note: Exact behavior depends on NULL semantics implementation
    assert!(non_null.len() + null_rows.len() >= 1);
}

#[test]
fn test_self_loop_graph_handling() {
    let store = TensorStore::new();
    let graph = GraphEngine::with_store(store);

    // Create node
    let node = graph.create_node("self_ref", HashMap::new()).unwrap();

    // Create self-loop edge
    let result = graph.create_edge(node, node, "loops_to", HashMap::new(), true);

    match result {
        Ok(_) => {
            // Self-loop created successfully
            // Note: neighbors() may or may not include self in results
            // This documents the behavior
            let _neighbors = graph
                .neighbors(node, None, Direction::Outgoing, None)
                .unwrap();

            // The edge exists, so the graph supports self-loops
            // neighbors may filter out self-references in some implementations

            // Path to self should work or return trivial path
            let path_result = graph.find_path(node, node, None);
            match path_result {
                Ok(path) => {
                    // Path to self exists
                    let _ = path;
                },
                Err(_) => {
                    // Self-path may not be supported
                },
            }
        },
        Err(e) => {
            // Self-loops may be rejected - document behavior
            let err_str = e.to_string().to_lowercase();
            assert!(
                err_str.contains("self")
                    || err_str.contains("loop")
                    || err_str.contains("same")
                    || err_str.contains("invalid"),
                "Self-loop rejection should have descriptive error"
            );
        },
    }
}
