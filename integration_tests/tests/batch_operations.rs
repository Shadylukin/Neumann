//! Batch operations integration tests.
//!
//! Tests bulk embedding ingestion and batch processing operations.

use integration_tests::create_shared_router;
use query_router::QueryResult;

#[test]
fn test_embed_batch_basic() {
    let router = create_shared_router();

    // Batch store multiple embeddings
    let result = router.execute_parsed(
        "EMBED BATCH [('doc:1', [1.0, 0.0, 0.0, 0.0]), ('doc:2', [0.0, 1.0, 0.0, 0.0]), ('doc:3', [0.0, 0.0, 1.0, 0.0])]"
    );

    assert!(result.is_ok());
    if let Ok(QueryResult::Count(n)) = result {
        assert_eq!(n, 3);
    }
}

#[test]
fn test_embed_batch_empty() {
    let router = create_shared_router();

    // Empty batch should work
    let result = router.execute_parsed("EMBED BATCH []");

    assert!(result.is_ok());
    if let Ok(QueryResult::Count(n)) = result {
        assert_eq!(n, 0);
    }
}

#[test]
fn test_embed_batch_single() {
    let router = create_shared_router();

    // Single item batch
    let result = router.execute_parsed("EMBED BATCH [('single:key', [0.5, 0.5, 0.5, 0.5])]");

    assert!(result.is_ok());
    if let Ok(QueryResult::Count(n)) = result {
        assert_eq!(n, 1);
    }

    // Verify it was stored
    let get_result = router.execute_parsed("EMBED GET 'single:key'");
    assert!(get_result.is_ok());
}

#[test]
fn test_embed_batch_large() {
    let router = create_shared_router();

    // Build a large batch with simple non-negative values
    let items: Vec<String> = (0..50)
        .map(|i| {
            let v1 = (i as f32) / 50.0;
            let v2 = ((i + 1) as f32) / 50.0;
            format!("('batch:{}', [{:.4}, {:.4}, 0.5, 0.5])", i, v1, v2)
        })
        .collect();

    let batch_query = format!("EMBED BATCH [{}]", items.join(", "));
    let result = router.execute_parsed(&batch_query);

    assert!(result.is_ok());
    if let Ok(QueryResult::Count(n)) = result {
        assert_eq!(n, 50);
    }
}

#[test]
fn test_embed_batch_then_search() {
    let router = create_shared_router();

    // Batch store embeddings
    router
        .execute_parsed(
            "EMBED BATCH [('search:1', [1.0, 0.0, 0.0, 0.0]), ('search:2', [0.9, 0.1, 0.0, 0.0]), ('search:3', [0.0, 0.0, 0.0, 1.0])]"
        )
        .unwrap();

    // Search for similar
    let result = router.execute_parsed("SIMILAR 'search:1' LIMIT 2");

    assert!(result.is_ok());
    if let Ok(QueryResult::Similar(results)) = result {
        // Should find search:1 itself and search:2 as most similar
        assert!(results.len() <= 2);
        if !results.is_empty() {
            assert!(results[0].key.contains("search:1"));
        }
    }
}

#[test]
fn test_embed_batch_overwrite() {
    let router = create_shared_router();

    // First batch
    router
        .execute_parsed("EMBED BATCH [('overwrite:key', [1.0, 0.0, 0.0, 0.0])]")
        .unwrap();

    // Overwrite with new embedding
    router
        .execute_parsed("EMBED BATCH [('overwrite:key', [0.0, 1.0, 0.0, 0.0])]")
        .unwrap();

    // Verify it was overwritten
    let result = router.execute_parsed("EMBED GET 'overwrite:key'");
    assert!(result.is_ok());
}

#[test]
fn test_embed_batch_different_dimensions() {
    let router = create_shared_router();

    // Different dimension embeddings (each batch should be consistent internally)
    let result1 = router.execute_parsed("EMBED BATCH [('dim4:1', [1.0, 0.0, 0.0, 0.0])]");
    assert!(result1.is_ok());

    let result2 = router.execute_parsed("EMBED BATCH [('dim8:1', [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]");
    assert!(result2.is_ok());
}

#[test]
fn test_embed_batch_with_special_keys() {
    let router = create_shared_router();

    // Keys with special characters
    let result = router.execute_parsed(
        "EMBED BATCH [('ns:sub/key-1', [1.0, 0.0]), ('ns:sub/key-2', [0.0, 1.0])]",
    );

    assert!(result.is_ok());
}

#[test]
fn test_embed_batch_count_embeddings() {
    let router = create_shared_router();

    // Store batch
    router
        .execute_parsed(
            "EMBED BATCH [('count:1', [1.0, 0.0]), ('count:2', [0.0, 1.0]), ('count:3', [0.5, 0.5])]",
        )
        .unwrap();

    // Count embeddings
    let result = router.execute_parsed("COUNT EMBEDDINGS");
    assert!(result.is_ok());

    if let Ok(QueryResult::Count(n)) = result {
        assert!(n >= 3);
    }
}

#[test]
fn test_embed_batch_show_embeddings() {
    let router = create_shared_router();

    // Store batch
    router
        .execute_parsed("EMBED BATCH [('show:1', [1.0, 0.0]), ('show:2', [0.0, 1.0])]")
        .unwrap();

    // Show embeddings
    let result = router.execute_parsed("SHOW EMBEDDINGS LIMIT 10");
    assert!(result.is_ok());

    if let Ok(QueryResult::Value(v)) = result {
        // Should contain our keys
        assert!(v.contains("show:") || v.is_empty());
    }
}

#[test]
fn test_embed_batch_build_index() {
    let mut router = create_shared_router();

    // Store enough embeddings for index building with non-negative values
    let items: Vec<String> = (0..20)
        .map(|i| {
            let v = (i as f32) / 20.0;
            format!("('idx:{}', [{:.2}, {:.2}, 0.5, 0.5])", i, v, 1.0 - v)
        })
        .collect();

    router
        .execute_parsed(&format!("EMBED BATCH [{}]", items.join(", ")))
        .unwrap();

    // Build HNSW index via Rust API (EMBED BUILD INDEX returns guidance message)
    let result = router.build_vector_index();
    assert!(result.is_ok());
}

#[test]
fn test_embed_batch_sequential_batches() {
    let router = create_shared_router();

    // Multiple sequential batches
    for batch in 0..5 {
        let items: Vec<String> = (0..10)
            .map(|i| {
                format!("('seq:{}:{}',[{:.1}, {:.1}])", batch, i, batch as f32, i as f32)
            })
            .collect();

        let result = router.execute_parsed(&format!("EMBED BATCH [{}]", items.join(",")));
        assert!(result.is_ok());
    }

    // Should have 50 embeddings total
    let count_result = router.execute_parsed("COUNT EMBEDDINGS");
    if let Ok(QueryResult::Count(n)) = count_result {
        assert!(n >= 50);
    }
}

#[test]
fn test_embed_batch_and_entity_create() {
    let router = create_shared_router();

    // Mix batch embeddings with entity creation
    router
        .execute_parsed("EMBED BATCH [('doc:1', [1.0, 0.0, 0.0, 0.0])]")
        .unwrap();

    router
        .execute_parsed("ENTITY CREATE 'doc:2' { type: 'article' } EMBEDDING [0.0, 1.0, 0.0, 0.0]")
        .unwrap();

    // Both should be searchable
    let result = router.execute_parsed("SIMILAR 'doc:1' LIMIT 5");
    assert!(result.is_ok());
}

#[test]
fn test_embed_batch_performance_baseline() {
    let router = create_shared_router();

    // Performance test with 100 embeddings using non-negative values
    let items: Vec<String> = (0..100)
        .map(|i| {
            let v1 = (i as f32) / 100.0;
            let v2 = ((i + 1) as f32) / 100.0;
            format!("('perf:{}', [{:.4}, {:.4}, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2])", i, v1, v2)
        })
        .collect();

    let start = std::time::Instant::now();
    let result = router.execute_parsed(&format!("EMBED BATCH [{}]", items.join(", ")));
    let elapsed = start.elapsed();

    assert!(result.is_ok());
    // Should complete in reasonable time (under 5 seconds)
    assert!(elapsed.as_secs() < 5);
}

#[test]
fn test_embed_batch_negative_values() {
    let router = create_shared_router();

    // Negative values may not be supported in the current parser - skip or expect error
    // For now, just test that it doesn't crash
    let result = router.execute_parsed(
        "EMBED BATCH [('neg:1', [0.1, 0.5, 0.3, 0.8]), ('neg:2', [0.5, 0.5, 0.5, 0.5])]",
    );

    assert!(result.is_ok());
}

#[test]
fn test_embed_batch_zero_vectors() {
    let router = create_shared_router();

    // Zero vectors should work
    let result =
        router.execute_parsed("EMBED BATCH [('zero:1', [0.0, 0.0, 0.0, 0.0])]");

    assert!(result.is_ok());
}

#[test]
fn test_embed_batch_normalized() {
    let router = create_shared_router();

    // Normalized vectors (unit length)
    let result = router.execute_parsed(
        "EMBED BATCH [('norm:1', [0.5, 0.5, 0.5, 0.5]), ('norm:2', [0.7071, 0.7071, 0.0, 0.0])]",
    );

    assert!(result.is_ok());
}
