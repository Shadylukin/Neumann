use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use graph_engine::{Direction, GraphEngine, PropertyValue};
use relational_engine::{Column, ColumnType, Condition, RelationalEngine, Schema, Value};
use tensor_blob::{BlobConfig, BlobStore, PutOptions};
use tensor_checkpoint::{CheckpointConfig, CheckpointManager};
use tensor_store::TensorStore;
use tensor_unified::UnifiedEngine;
use vector_engine::VectorEngine;

/// The "Grand Unification" Test
#[tokio::test]
async fn test_the_neumann_protocol() {
    // 1. THE FOUNDATION
    let store = TensorStore::new();

    // 2. THE ENGINES
    // Wrap stateful engines in Arc for sharing
    // BlobStore needs Mutex because it has internal mutability (GC state) and we need to share it
    let blob = Arc::new(Mutex::new(
        BlobStore::new(store.clone(), BlobConfig::default())
            .await
            .unwrap(),
    ));

    let relational = Arc::new(RelationalEngine::with_store(store.clone()));
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let vector = Arc::new(VectorEngine::with_store(store.clone()));

    // The Unified Engine sits on top
    let _unified = UnifiedEngine::with_engines(
        store.clone(),
        relational.clone(),
        graph.clone(),
        vector.clone(),
    );

    // The Time Machine
    let checkpoint_mgr = CheckpointManager::new(blob.clone(), CheckpointConfig::default()).await;

    // === STEP 1: TIME ZERO ===
    let genesis_snapshot = checkpoint_mgr
        .create(Some("genesis"), &store)
        .await
        .unwrap();

    // === STEP 2: INGESTION ===

    // A. Store the Raw Artifact
    let paper_content = b"The universe is not made of atoms, but of tensors.";
    let blob_id = blob
        .lock()
        .await
        .put(
            "neumann_theory.pdf",
            paper_content,
            PutOptions::new().with_content_type("application/pdf"),
        )
        .await
        .unwrap();

    // B. Understand the Meaning
    let paper_embedding = vec![0.1, 0.8, 0.3, 0.9];
    vector
        .store_embedding(&blob_id, paper_embedding.clone())
        .unwrap();

    // C. Structure the Metadata
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::String),
        Column::new("title", ColumnType::String),
        Column::new("year", ColumnType::Int),
        Column::new("blob_ref", ColumnType::String),
    ]);
    relational.create_table("papers", schema).unwrap();

    let mut row = HashMap::new();
    row.insert("id".to_string(), Value::String("p1".to_string()));
    row.insert(
        "title".to_string(),
        Value::String("Unified Field Theory".to_string()),
    );
    row.insert("year".to_string(), Value::Int(2026));
    row.insert("blob_ref".to_string(), Value::String(blob_id.clone()));
    relational.insert("papers", row).unwrap();

    // D. Connect the Knowledge
    let author_node = graph
        .create_node(
            "Person",
            HashMap::from([(
                "name".to_string(),
                PropertyValue::String("Lukin".to_string()),
            )]),
        )
        .unwrap();

    let paper_node = graph
        .create_node(
            "Paper",
            HashMap::from([(
                "rel_id".to_string(),
                PropertyValue::String("p1".to_string()),
            )]),
        )
        .unwrap();

    graph
        .create_edge(author_node, paper_node, "WROTE", HashMap::new(), true)
        .unwrap();

    // === STEP 3: THE "IMPOSSIBLE" QUERY ===

    // 1. Graph lookup
    let written_papers = graph
        .neighbors(author_node, Some("WROTE"), Direction::Outgoing, None)
        .unwrap();
    assert_eq!(written_papers.len(), 1);
    let target_paper_node_id = written_papers[0].id;

    // 2. Vector search
    let query_vec = vec![0.15, 0.85, 0.25, 0.85];
    let similar_blobs = vector.search_similar(&query_vec, 5).unwrap();

    // 3. Intersection
    let node_props = graph.get_node(target_paper_node_id).unwrap().properties;
    let rel_id_val = match node_props.get("rel_id").unwrap() {
        PropertyValue::String(s) => s,
        _ => panic!("Invalid type"),
    };

    let meta_rows = relational
        .select(
            "papers",
            Condition::Eq("id".to_string(), Value::String(rel_id_val.clone())),
        )
        .unwrap();
    let stored_blob_ref = match meta_rows[0].get("blob_ref").unwrap() {
        Value::String(s) => s,
        _ => panic!("Invalid type"),
    };

    let is_semantic_match = similar_blobs.iter().any(|res| res.key == *stored_blob_ref);
    assert!(
        is_semantic_match,
        "The paper written by Lukin matches the semantic query!"
    );

    // 4. Retrieve Blob
    let retrieved_content = blob.lock().await.get(stored_blob_ref).await.unwrap();
    assert_eq!(retrieved_content, paper_content);

    // === STEP 4: THE TIME TRAVEL ===
    checkpoint_mgr
        .rollback(&genesis_snapshot, &store)
        .await
        .unwrap();

    // Verify
    assert_eq!(vector.count(), 0);
    assert!(!graph.node_exists(author_node));
}
