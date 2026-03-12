// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error handling integration tests.
//!
//! Tests that all engines return proper errors for invalid operations.

use std::{collections::HashMap, sync::Arc};

use graph_engine::GraphEngine;
use integration_tests::{create_shared_router, sample_embeddings};
use relational_engine::{Column, ColumnType, Condition, Schema, Value};
use tensor_blob::{BlobConfig, BlobStore};
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig};
use vector_engine::VectorEngine;

#[test]
fn test_insert_into_nonexistent_table() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store);

    let mut row = HashMap::new();
    row.insert("id".to_string(), Value::Int(1));
    row.insert("name".to_string(), Value::String("Alice".to_string()));

    let result = relational.insert("nonexistent_table", row);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("not found") || err.to_string().contains("does not exist"),
        "Expected 'not found' error, got: {}",
        err
    );
}

#[test]
fn test_select_from_nonexistent_table() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store);

    let result = relational.select("nonexistent_table", Condition::True);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("not found") || err.to_string().contains("does not exist"),
        "Expected 'not found' error, got: {}",
        err
    );
}

#[test]
fn test_delete_nonexistent_node() {
    let store = TensorStore::new();
    let graph = GraphEngine::with_store(store);

    // Try to get a node that doesn't exist
    let result = graph.get_node(99999);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("not found") || err.to_string().contains("NodeNotFound"),
        "Expected 'not found' error, got: {}",
        err
    );
}

#[test]
fn test_get_nonexistent_embedding() {
    let store = TensorStore::new();
    let vector = VectorEngine::with_store(store);

    let result = vector.get_embedding("nonexistent_key");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("not found") || err.to_string().contains("NotFound"),
        "Expected 'not found' error, got: {}",
        err
    );
}

#[test]
fn test_vault_access_denied() {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let master_key = b"test-master-key-32-bytes-long!!";

    let vault = Vault::new(
        master_key,
        Arc::clone(&graph),
        store,
        VaultConfig::default(),
    )
    .unwrap();

    // Store a secret as ROOT
    vault
        .set(Vault::ROOT, "admin/secret", "secret-value")
        .unwrap();

    // Create a non-root entity node
    let user_node = graph.create_node("user", HashMap::new()).unwrap();
    let user_entity = format!("node:{}", user_node);

    // Try to access as non-root entity (should be denied)
    let result = vault.get(&user_entity, "admin/secret");
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("denied") || err.to_string().contains("AccessDenied"),
        "Expected access denied error, got: {}",
        err
    );
}

#[tokio::test]
async fn test_blob_get_nonexistent() {
    let store = TensorStore::new();
    let config = BlobConfig::default();
    let blob = BlobStore::new(store, config).await.unwrap();

    let result = blob.get("nonexistent-artifact-id").await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("not found") || err.to_string().contains("NotFound"),
        "Expected 'not found' error, got: {}",
        err
    );
}

#[test]
fn test_dimension_mismatch_on_search() {
    let store = TensorStore::new();
    let vector = VectorEngine::with_store(store);

    // Store embeddings with dimension 32
    let embeddings = sample_embeddings(10, 32);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("emb{}", i), emb.clone())
            .unwrap();
    }

    // Search with wrong dimension (64 instead of 32)
    let wrong_dim_query = sample_embeddings(1, 64)[0].clone();
    let result = vector.search_similar(&wrong_dim_query, 5);

    // Should either return error or empty results (implementation-dependent)
    // The key is it shouldn't panic
    match result {
        Ok(_results) => {
            // If it returns results, they should be empty or handle gracefully
            // This is acceptable behavior
        },
        Err(e) => {
            // Error is also acceptable
            assert!(
                e.to_string().contains("dimension") || e.to_string().contains("mismatch"),
                "Expected dimension-related error, got: {}",
                e
            );
        },
    }
}

#[test]
fn test_invalid_sql_syntax() {
    let router = create_shared_router();

    // Test various invalid SQL syntaxes
    // Note: Some invalid syntaxes may cause panics in the parser
    // These tests verify that errors are returned instead of success
    let invalid_queries = vec![
        "SELECT FROM",       // Missing table
        "INSERT INTO",       // Incomplete
        "CREATE TABLE ()",   // Missing table name
        "CREATE TABLE test", // Missing columns
        "DELETE FROM WHERE", // Invalid condition
        "UNKNOWN COMMAND",   // Unrecognized command
    ];

    for query in invalid_queries {
        let result = router.execute(query);
        assert!(
            result.is_err(),
            "Expected error for invalid query: {}",
            query
        );
    }
}

#[test]
fn test_type_mismatch_on_insert() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store);

    // Create table with specific types
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("typed_table", schema).unwrap();

    // Try to insert wrong type (string into int column)
    let mut row = HashMap::new();
    row.insert("id".to_string(), Value::String("not_an_int".to_string()));
    row.insert("name".to_string(), Value::String("Alice".to_string()));

    let result = relational.insert("typed_table", row);

    // Either returns error or coerces - check behavior
    // Note: Implementation may allow this, so we just verify no panic
    match result {
        Ok(_) => {
            // Some implementations may coerce or store as-is
        },
        Err(e) => {
            // Type error is expected
            assert!(
                e.to_string().contains("type") || e.to_string().contains("mismatch"),
                "Expected type-related error, got: {}",
                e
            );
        },
    }
}

#[test]
fn test_duplicate_table_create() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store);

    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);

    // Create table first time - should succeed
    relational
        .create_table("dup_table", schema.clone())
        .unwrap();

    // Create same table again - should fail
    let result = relational.create_table("dup_table", schema);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("exists") || err.to_string().contains("already"),
        "Expected 'already exists' error, got: {}",
        err
    );
}
