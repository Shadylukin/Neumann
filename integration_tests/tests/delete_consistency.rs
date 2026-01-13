//! Delete operation consistency tests.
//!
//! Tests that delete operations work correctly and maintain cross-engine consistency.

use std::{collections::HashMap, sync::Arc, thread};

use graph_engine::{Direction, GraphEngine, PropertyValue};
use integration_tests::{create_shared_engines, sample_embeddings};
use relational_engine::{Column, ColumnType, Condition, Schema, Value};
use tensor_store::TensorStore;
use vector_engine::VectorEngine;

#[test]
fn test_delete_relational_row() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store);

    // Create table and insert rows
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("users", schema).unwrap();

    for i in 0..5 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("name".to_string(), Value::String(format!("User{}", i)));
        relational.insert("users", row).unwrap();
    }

    // Verify 5 rows exist
    let rows = relational.select("users", Condition::True).unwrap();
    assert_eq!(rows.len(), 5);

    // Delete rows with id > 2
    let deleted = relational
        .delete_rows("users", Condition::Gt("id".to_string(), Value::Int(2)))
        .unwrap();
    assert_eq!(deleted, 2); // Should delete id=3 and id=4

    // Verify only 3 rows remain
    let remaining = relational.select("users", Condition::True).unwrap();
    assert_eq!(remaining.len(), 3);

    // Verify deleted rows are gone
    let high_id_rows = relational
        .select("users", Condition::Gt("id".to_string(), Value::Int(2)))
        .unwrap();
    assert_eq!(high_id_rows.len(), 0);
}

#[test]
fn test_delete_graph_node() {
    let store = TensorStore::new();
    let graph = GraphEngine::with_store(store);

    // Create nodes
    let node1 = graph.create_node("person", HashMap::new()).unwrap();
    let node2 = graph.create_node("person", HashMap::new()).unwrap();
    let node3 = graph.create_node("person", HashMap::new()).unwrap();

    // Create edges
    graph
        .create_edge(node1, node2, "knows", HashMap::new(), true)
        .unwrap();
    graph
        .create_edge(node2, node3, "knows", HashMap::new(), true)
        .unwrap();

    // Verify node2 has neighbors
    let neighbors_before = graph.neighbors(node2, None, Direction::Both).unwrap();
    assert!(!neighbors_before.is_empty());

    // Delete node2
    graph.delete_node(node2).unwrap();

    // Verify node2 is gone
    assert!(graph.get_node(node2).is_err());

    // Verify edges involving node2 are cleaned up
    // node1 should no longer have node2 as neighbor
    let node1_neighbors = graph.neighbors(node1, None, Direction::Outgoing).unwrap();
    assert!(!node1_neighbors.iter().any(|n| n.id == node2));

    // node3 should no longer have node2 as incoming neighbor
    let node3_neighbors = graph.neighbors(node3, None, Direction::Incoming).unwrap();
    assert!(!node3_neighbors.iter().any(|n| n.id == node2));
}

#[test]
fn test_delete_vector_embedding() {
    let store = TensorStore::new();
    let vector = VectorEngine::with_store(store);

    // Store embeddings
    let embeddings = sample_embeddings(5, 32);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc{}", i), emb.clone())
            .unwrap();
    }

    // Verify embedding exists and is searchable
    let query = embeddings[2].clone();
    let results_before = vector.search_similar(&query, 5).unwrap();
    assert!(results_before.iter().any(|r| r.key == "doc2"));

    // Remove embedding
    vector.delete_embedding("doc2").unwrap();

    // Verify embedding is gone
    assert!(vector.get_embedding("doc2").is_err());

    // Verify it's no longer in search results
    let results_after = vector.search_similar(&query, 5).unwrap();
    assert!(!results_after.iter().any(|r| r.key == "doc2"));
}

#[test]
fn test_delete_cross_engine_entity() {
    let (store, relational, graph, vector) = create_shared_engines();

    // Create unified entity across all engines
    let entity_id = 42;

    // Relational
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("entities", schema).unwrap();
    let mut row = HashMap::new();
    row.insert("id".to_string(), Value::Int(entity_id));
    row.insert("name".to_string(), Value::String("Entity42".to_string()));
    relational.insert("entities", row).unwrap();

    // Graph
    let mut props = HashMap::new();
    props.insert("entity_id".to_string(), PropertyValue::Int(entity_id));
    let node_id = graph.create_node("entity", props).unwrap();

    // Vector
    let embedding = sample_embeddings(1, 16)[0].clone();
    let emb_key = format!("entity:{}", entity_id);
    vector.store_embedding(&emb_key, embedding.clone()).unwrap();

    // Verify all exist
    assert_eq!(
        relational
            .select("entities", Condition::True)
            .unwrap()
            .len(),
        1
    );
    assert!(graph.get_node(node_id).is_ok());
    assert!(vector.get_embedding(&emb_key).is_ok());

    // Delete from all engines
    relational
        .delete_rows(
            "entities",
            Condition::Eq("id".to_string(), Value::Int(entity_id)),
        )
        .unwrap();
    graph.delete_node(node_id).unwrap();
    vector.delete_embedding(&emb_key).unwrap();

    // Verify all deleted
    assert_eq!(
        relational
            .select("entities", Condition::True)
            .unwrap()
            .len(),
        0
    );
    assert!(graph.get_node(node_id).is_err());
    assert!(vector.get_embedding(&emb_key).is_err());
}

#[test]
fn test_delete_during_query() {
    let store = Arc::new(TensorStore::new());
    let relational = Arc::new(relational_engine::RelationalEngine::with_store(
        (*store).clone(),
    ));

    // Create table with many rows
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("data", ColumnType::String),
    ]);
    relational.create_table("large_table", schema).unwrap();

    for i in 0..1000 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("data".to_string(), Value::String(format!("data{}", i)));
        relational.insert("large_table", row).unwrap();
    }

    let query_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let delete_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let mut handles = vec![];

    // Query thread
    let rel_query = Arc::clone(&relational);
    let qc = Arc::clone(&query_count);
    handles.push(thread::spawn(move || {
        for _ in 0..50 {
            if let Ok(rows) = rel_query.select("large_table", Condition::True) {
                qc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                // Just access the rows, don't fail if count varies
                let _ = rows.len();
            }
        }
    }));

    // Delete thread
    let rel_delete = Arc::clone(&relational);
    let dc = Arc::clone(&delete_count);
    handles.push(thread::spawn(move || {
        for i in 0..50 {
            // Delete rows with id divisible by 20
            if let Ok(n) = rel_delete.delete_rows(
                "large_table",
                Condition::Eq("id".to_string(), Value::Int(i * 20)),
            ) {
                dc.fetch_add(n, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }));

    for handle in handles {
        handle.join().unwrap();
    }

    // Both operations should complete without panic
    assert!(query_count.load(std::sync::atomic::Ordering::SeqCst) > 0);
    // Final state should be consistent
    let final_rows = relational.select("large_table", Condition::True).unwrap();
    assert!(final_rows.len() < 1000); // Some should be deleted
}

#[test]
fn test_delete_node_with_edges() {
    let store = TensorStore::new();
    let graph = GraphEngine::with_store(store);

    // Create hub-and-spoke pattern
    let hub = graph.create_node("hub", HashMap::new()).unwrap();
    let mut spokes = vec![];
    for i in 0..5 {
        let mut props = HashMap::new();
        props.insert("idx".to_string(), PropertyValue::Int(i));
        let spoke = graph.create_node("spoke", props).unwrap();
        graph
            .create_edge(hub, spoke, "connects", HashMap::new(), true)
            .unwrap();
        spokes.push(spoke);
    }

    // Verify hub has 5 outgoing edges
    let hub_neighbors = graph.neighbors(hub, None, Direction::Outgoing).unwrap();
    assert_eq!(hub_neighbors.len(), 5);

    // Delete hub
    graph.delete_node(hub).unwrap();

    // Verify hub is gone
    assert!(graph.get_node(hub).is_err());

    // Verify all spokes still exist and have no incoming edges from hub
    for spoke in &spokes {
        let node = graph.get_node(*spoke).unwrap();
        assert_eq!(node.label, "spoke");

        let incoming = graph.neighbors(*spoke, None, Direction::Incoming).unwrap();
        assert!(!incoming.iter().any(|n| n.id == hub));
    }
}

#[test]
fn test_cascade_delete_effects() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store.clone());
    let vector = VectorEngine::with_store(store);

    // Create parent table
    let parent_schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("parents", parent_schema).unwrap();

    // Create child table (references parent)
    let child_schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("parent_id", ColumnType::Int),
        Column::new("data", ColumnType::String),
    ]);
    relational.create_table("children", child_schema).unwrap();

    // Insert parent
    let mut parent_row = HashMap::new();
    parent_row.insert("id".to_string(), Value::Int(1));
    parent_row.insert("name".to_string(), Value::String("Parent1".to_string()));
    relational.insert("parents", parent_row).unwrap();

    // Insert children referencing parent
    for i in 0..3 {
        let mut child_row = HashMap::new();
        child_row.insert("id".to_string(), Value::Int(i + 10));
        child_row.insert("parent_id".to_string(), Value::Int(1));
        child_row.insert("data".to_string(), Value::String(format!("Child{}", i)));
        relational.insert("children", child_row).unwrap();

        // Also store embedding for each child
        let emb = sample_embeddings(1, 8)[0].clone();
        vector
            .store_embedding(&format!("child:{}", i + 10), emb)
            .unwrap();
    }

    // Delete parent (simulating cascade - manual in this system)
    relational
        .delete_rows("parents", Condition::Eq("id".to_string(), Value::Int(1)))
        .unwrap();

    // Delete orphaned children
    relational
        .delete_rows(
            "children",
            Condition::Eq("parent_id".to_string(), Value::Int(1)),
        )
        .unwrap();

    // Delete associated embeddings
    for i in 0..3 {
        vector
            .delete_embedding(&format!("child:{}", i + 10))
            .unwrap();
    }

    // Verify cascade completed
    assert_eq!(
        relational.select("parents", Condition::True).unwrap().len(),
        0
    );
    assert_eq!(
        relational
            .select("children", Condition::True)
            .unwrap()
            .len(),
        0
    );
    for i in 0..3 {
        assert!(vector.get_embedding(&format!("child:{}", i + 10)).is_err());
    }
}
