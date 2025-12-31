//! Cross-engine integration tests.
//!
//! Tests data flow and operations across multiple engines sharing a TensorStore.

use graph_engine::{Direction, GraphEngine, PropertyValue};
use integration_tests::{create_shared_engines, create_shared_router, sample_embeddings};
use relational_engine::{Column, ColumnType, Condition, Schema, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tensor_blob::{BlobConfig, BlobStore, PutOptions};
use tensor_cache::Cache;
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig};
use vector_engine::VectorEngine;

#[test]
fn test_unified_entity_across_engines() {
    let (_store, relational, graph, vector) = create_shared_engines();

    // Create a unified "user" entity with data in all engines
    let user_key = "user:alice";

    // Store relational data
    let schema = Schema::new(vec![
        Column::new("key", ColumnType::String),
        Column::new("email", ColumnType::String),
        Column::new("age", ColumnType::Int),
    ]);
    relational.create_table("users", schema).unwrap();

    let mut row = HashMap::new();
    row.insert("key".to_string(), Value::String(user_key.to_string()));
    row.insert(
        "email".to_string(),
        Value::String("alice@example.com".to_string()),
    );
    row.insert("age".to_string(), Value::Int(30));
    relational.insert("users", row).unwrap();

    // Store graph relationships
    let alice_node = graph.create_node("user", HashMap::new()).unwrap();
    let bob_node = graph.create_node("user", HashMap::new()).unwrap();
    graph
        .create_edge(alice_node, bob_node, "follows", HashMap::new(), true)
        .unwrap();

    // Store vector embedding
    let embedding = sample_embeddings(1, 64)[0].clone();
    vector.store_embedding(user_key, embedding.clone()).unwrap();

    // Query across all engines
    // 1. Relational lookup
    let rows = relational
        .select(
            "users",
            Condition::Eq("key".to_string(), Value::String(user_key.to_string())),
        )
        .unwrap();
    assert_eq!(rows.len(), 1);

    // 2. Graph neighbors
    let neighbors = graph
        .neighbors(alice_node, None, Direction::Outgoing)
        .unwrap();
    assert!(neighbors.iter().any(|n| n.id == bob_node));

    // 3. Vector similarity (should find itself)
    let results = vector.search_similar(&embedding, 5).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].key, user_key);
}

#[test]
fn test_graph_nodes_with_embeddings() {
    let store = TensorStore::new();
    let graph = GraphEngine::with_store(store.clone());
    let vector = VectorEngine::with_store(store);

    // Create nodes and store embeddings for each
    let embeddings = sample_embeddings(5, 32);
    let mut node_ids = vec![];

    for (i, emb) in embeddings.iter().enumerate() {
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String(format!("node{}", i)),
        );
        let node_id = graph.create_node("entity", props).unwrap();
        node_ids.push(node_id);

        // Store embedding with node key
        let node_key = format!("node:{}", node_id);
        vector.store_embedding(&node_key, emb.clone()).unwrap();
    }

    // Create edges
    for i in 0..4 {
        graph
            .create_edge(
                node_ids[i],
                node_ids[i + 1],
                "connects",
                HashMap::new(),
                true,
            )
            .unwrap();
    }

    // Find similar nodes
    let query = embeddings[0].clone();
    let similar = vector.search_similar(&query, 3).unwrap();

    // Most similar should be the first node (matches query embedding)
    assert!(similar[0].key.contains(&format!("node:{}", node_ids[0])));

    // Verify graph connectivity of similar nodes
    for result in &similar {
        let node_id: u64 = result.key.split(':').nth(1).unwrap().parse().unwrap();
        let node = graph.get_node(node_id).unwrap();
        assert!(node.label == "entity");
    }
}

#[test]
fn test_vault_with_graph_access_control() {
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

    // Create a graph-based access control structure
    // admin -> team -> user
    let _admin_node = graph.create_node("identity", HashMap::new()).unwrap();
    let _team_node = graph.create_node("identity", HashMap::new()).unwrap();
    let _user_node = graph.create_node("identity", HashMap::new()).unwrap();

    // Store secret accessible by admin
    vault
        .set(Vault::ROOT, "admin/secret", "admin-only")
        .unwrap();

    // Verify access
    let secret = vault.get(Vault::ROOT, "admin/secret").unwrap();
    assert_eq!(secret, "admin-only");
}

#[test]
fn test_cache_with_relational_queries() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store.clone());
    let cache = Cache::new();

    // Create and populate table
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::String),
    ]);
    relational.create_table("data", schema).unwrap();
    for i in 0..10 {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("value".to_string(), Value::String(format!("val{}", i)));
        relational.insert("data", row).unwrap();
    }

    // Simulate query caching
    let query = "SELECT * FROM data WHERE id = 5";
    let result = relational
        .select("data", Condition::Eq("id".to_string(), Value::Int(5)))
        .unwrap();

    // Cache the result
    let result_str = format!("{:?}", result);
    cache.put_simple(query, &result_str).unwrap();

    // Verify cache hit
    let cached = cache.get_simple(query);
    assert!(cached.is_some());
    assert!(cached.unwrap().contains("val5"));
}

#[test]
fn test_insert_embed_search_cycle() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store.clone());
    let vector = VectorEngine::with_store(store);

    // Create table for documents
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("content", ColumnType::String),
    ]);
    relational.create_table("documents", schema).unwrap();

    // Insert documents and their embeddings
    let embeddings = sample_embeddings(10, 64);
    for i in 0..10 {
        // Insert relational record
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert(
            "content".to_string(),
            Value::String(format!("Document content {}", i)),
        );
        relational.insert("documents", row).unwrap();

        // Store embedding with document ID as key
        let key = format!("doc:{}", i);
        vector
            .store_embedding(&key, embeddings[i as usize].clone())
            .unwrap();
    }

    // Search for similar documents
    let query_embedding = embeddings[3].clone();
    let similar = vector.search_similar(&query_embedding, 5).unwrap();

    // Most similar should be doc:3
    assert_eq!(similar[0].key, "doc:3");

    // Look up the document in relational store
    let doc_id: i64 = similar[0].key.split(':').nth(1).unwrap().parse().unwrap();
    let rows = relational
        .select(
            "documents",
            Condition::Eq("id".to_string(), Value::Int(doc_id)),
        )
        .unwrap();

    assert_eq!(rows.len(), 1);
    assert!(format!("{:?}", rows[0]).contains("Document content 3"));
}

#[test]
fn test_node_edge_neighbor_path_cycle() {
    let store = TensorStore::new();
    let graph = GraphEngine::with_store(store);

    // Create a social network graph
    let mut users = HashMap::new();
    let names = ["Alice", "Bob", "Carol", "Dave", "Eve"];

    for name in &names {
        let mut props = HashMap::new();
        props.insert("name".to_string(), PropertyValue::String(name.to_string()));
        let id = graph.create_node("user", props).unwrap();
        users.insert(*name, id);
    }

    // Create edges (follows relationships)
    // Alice -> Bob -> Carol -> Dave -> Eve
    graph
        .create_edge(
            users["Alice"],
            users["Bob"],
            "follows",
            HashMap::new(),
            true,
        )
        .unwrap();
    graph
        .create_edge(
            users["Bob"],
            users["Carol"],
            "follows",
            HashMap::new(),
            true,
        )
        .unwrap();
    graph
        .create_edge(
            users["Carol"],
            users["Dave"],
            "follows",
            HashMap::new(),
            true,
        )
        .unwrap();
    graph
        .create_edge(users["Dave"], users["Eve"], "follows", HashMap::new(), true)
        .unwrap();
    // Carol also follows Alice (creates cycle potential)
    graph
        .create_edge(
            users["Carol"],
            users["Alice"],
            "follows",
            HashMap::new(),
            true,
        )
        .unwrap();

    // Test neighbors
    let bob_following = graph
        .neighbors(users["Bob"], None, Direction::Outgoing)
        .unwrap();
    assert!(bob_following.iter().any(|n| n.id == users["Carol"]));

    let bob_followers = graph
        .neighbors(users["Bob"], None, Direction::Incoming)
        .unwrap();
    assert!(bob_followers.iter().any(|n| n.id == users["Alice"]));

    // Test path finding
    let path = graph.find_path(users["Alice"], users["Eve"]).unwrap();
    assert!(!path.nodes.is_empty());
    assert_eq!(path.nodes[0], users["Alice"]);
    assert_eq!(*path.nodes.last().unwrap(), users["Eve"]);
}

#[test]
fn test_table_with_embedded_vectors() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store.clone());
    let vector = VectorEngine::with_store(store);

    // Create products table
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
        Column::new("price", ColumnType::Float),
    ]);
    relational.create_table("products", schema).unwrap();

    // Insert products with embeddings
    let products = [
        (1, "Running Shoes", 89.99),
        (2, "Basketball Sneakers", 129.99),
        (3, "Hiking Boots", 149.99),
        (4, "Sandals", 39.99),
        (5, "Dress Shoes", 199.99),
    ];

    let embeddings = sample_embeddings(5, 32);

    for (i, (id, name, price)) in products.iter().enumerate() {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(*id));
        row.insert("name".to_string(), Value::String(name.to_string()));
        row.insert("price".to_string(), Value::Float(*price));
        relational.insert("products", row).unwrap();

        let key = format!("product:{}", id);
        vector.store_embedding(&key, embeddings[i].clone()).unwrap();
    }

    // Find products similar to "Running Shoes" (embedding 0)
    let similar = vector.search_similar(&embeddings[0], 3).unwrap();

    // Verify we can look up similar products in relational store
    for result in &similar {
        let product_id: i64 = result.key.split(':').nth(1).unwrap().parse().unwrap();
        let rows = relational
            .select(
                "products",
                Condition::Eq("id".to_string(), Value::Int(product_id)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    // Price-based filtering combined with similarity
    let expensive = relational
        .select(
            "products",
            Condition::Gt("price".to_string(), Value::Float(100.0)),
        )
        .unwrap();
    assert_eq!(expensive.len(), 3); // Basketball, Hiking, Dress
}

#[tokio::test]
async fn test_blob_links_to_graph_entities() {
    let store = TensorStore::new();
    let graph = GraphEngine::with_store(store.clone());
    let config = BlobConfig::default();
    let blob = BlobStore::new(store, config).await.unwrap();

    // Create document nodes
    let doc1_node = graph.create_node("document", HashMap::new()).unwrap();
    let doc2_node = graph.create_node("document", HashMap::new()).unwrap();

    // Upload blobs
    let content1 = b"Document 1 content";
    let content2 = b"Document 2 content";

    let _artifact1 = blob
        .put("doc1.txt", content1, PutOptions::default())
        .await
        .unwrap();
    let _artifact2 = blob
        .put("doc2.txt", content2, PutOptions::default())
        .await
        .unwrap();

    // Link blobs to graph nodes using edges
    let blob1_node = graph.create_node("artifact", HashMap::new()).unwrap();
    let blob2_node = graph.create_node("artifact", HashMap::new()).unwrap();

    graph
        .create_edge(doc1_node, blob1_node, "has_artifact", HashMap::new(), true)
        .unwrap();
    graph
        .create_edge(doc2_node, blob2_node, "has_artifact", HashMap::new(), true)
        .unwrap();

    // Create relationship between documents
    graph
        .create_edge(doc1_node, doc2_node, "references", HashMap::new(), true)
        .unwrap();

    // Verify graph structure
    let doc1_neighbors = graph
        .neighbors(doc1_node, None, Direction::Outgoing)
        .unwrap();
    assert!(doc1_neighbors.iter().any(|n| n.id == blob1_node));
    assert!(doc1_neighbors.iter().any(|n| n.id == doc2_node));
}

#[test]
fn test_query_router_cross_engine_operations() {
    let router = create_shared_router();

    // Create table through router (uses col:type syntax for execute())
    router
        .execute("CREATE TABLE employees (id:INT, name:TEXT, dept:TEXT)")
        .unwrap();

    // Insert data
    router
        .execute("INSERT employees id=1, name='Alice', dept='Engineering'")
        .unwrap();
    router
        .execute("INSERT employees id=2, name='Bob', dept='Sales'")
        .unwrap();

    // Create graph nodes and capture IDs
    let node1_id = match router.execute("NODE CREATE employee id=1").unwrap() {
        query_router::QueryResult::Ids(ids) => ids[0],
        _ => panic!("Expected Ids result"),
    };
    let node2_id = match router.execute("NODE CREATE employee id=2").unwrap() {
        query_router::QueryResult::Ids(ids) => ids[0],
        _ => panic!("Expected Ids result"),
    };
    router
        .execute(&format!(
            "EDGE CREATE {} -> {} reports_to",
            node1_id, node2_id
        ))
        .unwrap();

    // Store embeddings
    router.execute("EMBED emp:1 1.0, 0.5, 0.3, 0.1").unwrap();
    router.execute("EMBED emp:2 0.9, 0.6, 0.2, 0.2").unwrap();

    // Query relational
    let result = router.execute("SELECT employees").unwrap();
    match result {
        query_router::QueryResult::Rows(rows) => {
            assert_eq!(rows.len(), 2);
        },
        _ => panic!("Expected Rows result"),
    }

    // Query graph
    let result = router
        .execute(&format!("NEIGHBORS {} OUT", node1_id))
        .unwrap();
    match result {
        query_router::QueryResult::Ids(ids) => {
            assert_eq!(ids.len(), 1);
            assert_eq!(ids[0], node2_id);
        },
        _ => panic!("Expected Ids result"),
    }

    // Query vector
    let result = router.execute("SIMILAR emp:1 TOP 2").unwrap();
    match result {
        query_router::QueryResult::Similar(results) => {
            assert!(!results.is_empty());
        },
        _ => panic!("Expected Similar result"),
    }
}

#[test]
fn test_cross_engine_data_consistency() {
    let store = TensorStore::new();
    let relational = relational_engine::RelationalEngine::with_store(store.clone());
    let graph = GraphEngine::with_store(store.clone());
    let vector = VectorEngine::with_store(store.clone());

    // Create linked entities
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    relational.create_table("entities", schema).unwrap();

    let mut node_ids = vec![];
    for i in 0..5 {
        // Relational record
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("name".to_string(), Value::String(format!("Entity {}", i)));
        relational.insert("entities", row).unwrap();

        // Graph node
        let mut props = HashMap::new();
        props.insert("entity_id".to_string(), PropertyValue::Int(i));
        let node_id = graph.create_node("entity", props).unwrap();
        node_ids.push(node_id);

        // Vector embedding
        let embedding = sample_embeddings(1, 16)[0].clone();
        vector
            .store_embedding(&format!("entity:{}", i), embedding)
            .unwrap();
    }

    // Verify consistency: all engines have 5 entities
    let rows = relational.select("entities", Condition::True).unwrap();
    assert_eq!(rows.len(), 5);

    // Check graph has nodes with correct IDs
    for node_id in &node_ids {
        let node = graph.get_node(*node_id).unwrap();
        assert_eq!(node.label, "entity");
    }

    // Check vectors exist for all entities
    for i in 0..5 {
        let key = format!("entity:{}", i);
        let emb = vector.get_embedding(&key).unwrap();
        assert_eq!(emb.len(), 16);
    }

    // Verify store has all the data (shared store)
    // Each entity contributes: 1 row key + 1 node key + adjacency keys + 1 embedding key
    assert!(store.len() >= 15); // At least 3 entries per entity
}
