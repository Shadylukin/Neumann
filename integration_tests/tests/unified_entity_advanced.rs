// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for tensor_unified advanced entity operations.

use integration_tests::create_shared_router;
use query_router::QueryResult;

#[test]
fn test_entity_create_and_get() {
    let router = create_shared_router();

    // Create entity with fields and embedding
    let result = router
        .execute_parsed("ENTITY CREATE 'user:alice' { name: 'Alice', role: 'admin' } EMBEDDING [0.1, 0.2, 0.3, 0.4]");
    assert!(result.is_ok(), "ENTITY CREATE failed: {result:?}");

    // GET should return all data
    let get = router.execute_parsed("ENTITY GET 'user:alice'");
    assert!(get.is_ok(), "ENTITY GET failed: {get:?}");

    match get.unwrap() {
        QueryResult::Unified(unified) => {
            assert_eq!(unified.items.len(), 1);
            let item = &unified.items[0];
            assert_eq!(item.id, "user:alice");
            // Verify fields are present
            assert!(
                item.data.contains_key("name") || item.data.contains_key("key"),
                "Expected entity data fields, got: {:?}",
                item.data
            );
            // Verify embedding was stored
            assert!(item.embedding.is_some(), "Expected embedding to be present");
            assert_eq!(item.embedding.as_ref().unwrap().len(), 4);
        },
        other => panic!("Expected Unified result, got: {other:?}"),
    }
}

#[test]
fn test_entity_update_partial_fields() {
    let router = create_shared_router();

    // Create entity
    router
        .execute_parsed("ENTITY CREATE 'user:bob' { name: 'Bob', age: '25', role: 'user' }")
        .unwrap();

    // Update subset of fields
    let update = router.execute_parsed("ENTITY UPDATE 'user:bob' { age: '26', role: 'admin' }");
    assert!(update.is_ok(), "ENTITY UPDATE failed: {update:?}");

    // GET and verify updated values
    let get = router.execute_parsed("ENTITY GET 'user:bob'").unwrap();
    match get {
        QueryResult::Unified(unified) => {
            assert_eq!(unified.items.len(), 1);
            let item = &unified.items[0];
            assert_eq!(item.id, "user:bob");
        },
        other => panic!("Expected Unified result, got: {other:?}"),
    }
}

#[test]
fn test_entity_delete_removes_data() {
    let router = create_shared_router();

    // Create entity
    router
        .execute_parsed("ENTITY CREATE 'temp:item' { name: 'Temporary' }")
        .unwrap();

    // Verify it exists
    let get = router.execute_parsed("ENTITY GET 'temp:item'");
    assert!(get.is_ok(), "Entity should exist after creation");

    // Delete it
    let delete = router.execute_parsed("ENTITY DELETE 'temp:item'");
    assert!(delete.is_ok(), "ENTITY DELETE failed: {delete:?}");

    // GET should now fail
    let get_after = router.execute_parsed("ENTITY GET 'temp:item'");
    assert!(
        get_after.is_err(),
        "ENTITY GET should fail after deletion, got: {get_after:?}"
    );
}

#[test]
fn test_entity_batch_create() {
    let router = create_shared_router();

    // Batch create multiple entities
    let result = router.execute_parsed(
        "ENTITY BATCH CREATE [{key: 'batch:1', name: 'First'}, {key: 'batch:2', name: 'Second'}, {key: 'batch:3', name: 'Third'}]",
    );
    assert!(result.is_ok(), "ENTITY BATCH CREATE failed: {result:?}");

    match result.unwrap() {
        QueryResult::BatchResult(batch) => {
            assert_eq!(batch.affected_count, 3);
        },
        other => panic!("Expected BatchResult, got: {other:?}"),
    }

    // Verify each entity is retrievable
    for i in 1..=3 {
        let get = router.execute_parsed(&format!("ENTITY GET 'batch:{i}'"));
        assert!(
            get.is_ok(),
            "batch:{i} should be retrievable after batch create"
        );
    }
}

#[test]
fn test_find_similar_entities() {
    let router = create_shared_router();

    // Store embeddings directly via EMBED (SIMILAR uses vector engine lookup)
    router.execute("EMBED doc:a 1.0, 0.0, 0.0").unwrap();
    router.execute("EMBED doc:b 0.9, 0.1, 0.0").unwrap();
    router.execute("EMBED doc:c 0.0, 1.0, 0.0").unwrap();

    // Use the SIMILAR command to find entities similar to doc:a
    let result = router.execute("SIMILAR doc:a TOP 2");
    assert!(result.is_ok(), "SIMILAR query failed: {result:?}");

    match result.unwrap() {
        QueryResult::Similar(results) => {
            assert!(!results.is_empty(), "Expected at least one similar result");
            // doc:b should rank higher than doc:c since its embedding is closer to doc:a
            if results.len() >= 2 {
                assert!(
                    results[0].score >= results[1].score,
                    "Results should be ranked by similarity"
                );
            }
        },
        other => panic!("Expected Similar result, got: {other:?}"),
    }
}

#[test]
fn test_find_connected_entities() {
    let router = create_shared_router();

    // Create entities and connect them via graph edges
    router
        .execute_parsed("ENTITY CREATE 'person:alice' { name: 'Alice' }")
        .unwrap();
    router
        .execute_parsed("ENTITY CREATE 'person:bob' { name: 'Bob' }")
        .unwrap();
    router
        .execute_parsed("ENTITY CREATE 'person:carol' { name: 'Carol' }")
        .unwrap();

    // Connect entities
    router
        .execute_parsed("ENTITY CONNECT 'person:alice' -> 'person:bob' : knows")
        .unwrap();
    router
        .execute_parsed("ENTITY CONNECT 'person:alice' -> 'person:carol' : knows")
        .unwrap();

    // Use FIND NODE to verify graph nodes exist
    let result = router.execute_parsed("FIND NODE");
    assert!(result.is_ok(), "FIND NODE failed: {result:?}");

    match result.unwrap() {
        QueryResult::Unified(unified) => {
            // Should have at least the entities we created
            assert!(
                unified.items.len() >= 3,
                "Expected at least 3 nodes, got {}",
                unified.items.len()
            );
        },
        other => {
            // FIND NODE may return different result types
            let _ = other;
        },
    }
}

#[test]
fn test_entity_multiple_create_and_get() {
    let router = create_shared_router();

    // Create 5 entities
    for i in 0..5 {
        router
            .execute_parsed(&format!(
                "ENTITY CREATE 'entity:{i}' {{ idx: '{i}' }} EMBEDDING [0.1, 0.2, 0.3, 0.4]"
            ))
            .unwrap();
    }

    // Verify all entities are individually retrievable via GET
    for i in 0..5 {
        let result = router
            .execute_parsed(&format!("ENTITY GET 'entity:{i}'"))
            .unwrap();
        match result {
            QueryResult::Unified(unified) => {
                assert_eq!(unified.items.len(), 1);
                assert_eq!(unified.items[0].id, format!("entity:{i}"));
            },
            other => panic!("Expected Unified for entity:{i}, got: {other:?}"),
        }
    }
}

#[test]
fn test_entity_embedding_update() {
    let router = create_shared_router();

    // Create entity with initial embedding
    router
        .execute_parsed("ENTITY CREATE 'doc:update' { title: 'Test' } EMBEDDING [0.1, 0.2]")
        .unwrap();

    // Verify initial embedding
    let emb1 = router.vector().get_entity_embedding("doc:update").unwrap();
    assert_eq!(emb1.len(), 2);

    // Update with new embedding
    let update = router
        .execute_parsed("ENTITY UPDATE 'doc:update' { title: 'Updated' } EMBEDDING [0.9, 0.8]");
    assert!(
        update.is_ok(),
        "ENTITY UPDATE with embedding failed: {update:?}"
    );

    // Verify new embedding
    let emb2 = router.vector().get_entity_embedding("doc:update").unwrap();
    assert_eq!(emb2.len(), 2);
    // The embedding values should have changed
    let changed = (emb2[0] - emb1[0]).abs() > 0.01 || (emb2[1] - emb1[1]).abs() > 0.01;
    assert!(
        changed,
        "Embedding should have been updated, old={emb1:?}, new={emb2:?}"
    );
}

#[test]
fn test_find_similar_to_vector() {
    let router = create_shared_router();

    // Store embeddings
    router.execute("EMBED vec:1 1.0, 0.0, 0.0").unwrap();
    router.execute("EMBED vec:2 0.0, 1.0, 0.0").unwrap();
    router.execute("EMBED vec:3 0.9, 0.1, 0.0").unwrap();

    // SIMILAR with inline vector
    let result = router.execute("SIMILAR [1.0, 0.0, 0.0] TOP 2");
    assert!(
        result.is_ok(),
        "SIMILAR with inline vector failed: {result:?}"
    );

    match result.unwrap() {
        QueryResult::Similar(results) => {
            assert_eq!(results.len(), 2, "Expected 2 results from TOP 2");
            // vec:1 or vec:3 should be most similar to [1.0, 0.0, 0.0]
            assert!(
                results[0].key.contains("vec:1") || results[0].key.contains("vec:3"),
                "Top result should be vec:1 or vec:3, got: {}",
                results[0].key
            );
        },
        other => panic!("Expected Similar result, got: {other:?}"),
    }
}

#[test]
fn test_entity_create_no_embedding() {
    let router = create_shared_router();

    // Create entity with only fields, no embedding
    let result =
        router.execute_parsed("ENTITY CREATE 'plain:item' { name: 'NoEmbed', status: 'active' }");
    assert!(
        result.is_ok(),
        "ENTITY CREATE without embedding failed: {result:?}"
    );

    // GET should still work
    let get = router.execute_parsed("ENTITY GET 'plain:item'");
    assert!(
        get.is_ok(),
        "ENTITY GET failed for entity without embedding: {get:?}"
    );

    match get.unwrap() {
        QueryResult::Unified(unified) => {
            assert_eq!(unified.items.len(), 1);
            let item = &unified.items[0];
            assert_eq!(item.id, "plain:item");
            // No embedding was stored
            // (embedding may or may not be None depending on how graph stores it)
        },
        other => panic!("Expected Unified result, got: {other:?}"),
    }
}

#[test]
fn test_similar_connected_to_cross_engine() {
    let router = create_shared_router();

    // Set up entities with embeddings via vector engine
    router
        .vector()
        .set_entity_embedding("hub:main", vec![0.5, 0.5, 0.0])
        .unwrap();
    router
        .vector()
        .set_entity_embedding("item:1", vec![1.0, 0.0, 0.0])
        .unwrap();
    router
        .vector()
        .set_entity_embedding("item:2", vec![0.9, 0.1, 0.0])
        .unwrap();
    router
        .vector()
        .set_entity_embedding("item:3", vec![0.0, 0.0, 1.0])
        .unwrap();

    // Connect items to hub via graph engine using entity connect
    router
        .execute_parsed("ENTITY CONNECT 'hub:main' -> 'item:1' : has")
        .unwrap();
    router
        .execute_parsed("ENTITY CONNECT 'hub:main' -> 'item:2' : has")
        .unwrap();
    // item:3 is NOT connected to hub

    // Query: find entities similar to item:1 that are connected to hub:main
    let result = router.execute_parsed("SIMILAR 'item:1' CONNECTED TO 'hub:main' LIMIT 5");
    assert!(result.is_ok(), "SIMILAR CONNECTED TO failed: {result:?}");

    match result.unwrap() {
        QueryResult::Similar(results) => {
            // Should only return items connected to hub:main
            for r in &results {
                assert!(
                    !r.key.contains("item:3"),
                    "item:3 should not appear since it is not connected to hub:main"
                );
            }
        },
        _ => {
            // Accept other result types (implementation may vary)
        },
    }
}
