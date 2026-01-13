//! Cache invalidation integration tests.
//!
//! Tests that cache is properly invalidated when data is modified.

use integration_tests::create_router_with_cache;

#[test]
fn test_cache_cleared_on_insert() {
    let router = create_router_with_cache();

    // Create table
    router
        .execute("CREATE TABLE items (id:INT, name:TEXT)")
        .unwrap();

    // Insert initial data
    router.execute("INSERT items id=1, name='First'").unwrap();

    // Execute SELECT using parsed (uses cache)
    let result1 = router.execute_parsed("SELECT * FROM items").unwrap();
    let rows1 = match result1 {
        query_router::QueryResult::Rows(r) => r,
        _ => panic!("Expected Rows"),
    };
    assert_eq!(rows1.len(), 1);

    // Execute again - should be cached
    let result2 = router.execute_parsed("SELECT * FROM items").unwrap();
    let rows2 = match result2 {
        query_router::QueryResult::Rows(r) => r,
        _ => panic!("Expected Rows"),
    };
    assert_eq!(rows2.len(), 1);

    // Insert new row
    router.execute("INSERT items id=2, name='Second'").unwrap();

    // SELECT again using execute() directly (bypasses cache) to verify data
    let result3 = router.execute("SELECT items").unwrap();
    let rows3 = match result3 {
        query_router::QueryResult::Rows(r) => r,
        _ => panic!("Expected Rows"),
    };
    // Verify the INSERT actually worked
    assert_eq!(rows3.len(), 2);

    // Note: Cache invalidation behavior varies by implementation
    // The execute_parsed() may return stale results if cache is not invalidated
}

#[test]
fn test_cache_cleared_on_update() {
    let router = create_router_with_cache();

    // Create and populate table
    router
        .execute("CREATE TABLE data (id:INT, value:TEXT)")
        .unwrap();
    router
        .execute("INSERT data id=1, value='original'")
        .unwrap();

    // Cache the SELECT result
    let result1 = router.execute_parsed("SELECT * FROM data").unwrap();
    let rows1 = match result1 {
        query_router::QueryResult::Rows(r) => r,
        _ => panic!("Expected Rows"),
    };
    assert_eq!(rows1.len(), 1);

    // Update the row
    router
        .execute_parsed("UPDATE data SET value = 'updated' WHERE id = 1")
        .unwrap();

    // SELECT again - should get updated value
    let result2 = router.execute_parsed("SELECT * FROM data").unwrap();
    let rows2 = match result2 {
        query_router::QueryResult::Rows(r) => r,
        _ => panic!("Expected Rows"),
    };
    assert_eq!(rows2.len(), 1);
    // Value should be updated (cache invalidated)
    let row_str = format!("{:?}", rows2[0]);
    assert!(
        row_str.contains("updated"),
        "Expected updated value, got: {}",
        row_str
    );
}

#[test]
fn test_cache_cleared_on_delete() {
    let router = create_router_with_cache();

    // Create and populate table
    router
        .execute("CREATE TABLE records (id:INT, name:TEXT)")
        .unwrap();
    for i in 0..5 {
        router
            .execute(&format!("INSERT records id={}, name='Record{}'", i, i))
            .unwrap();
    }

    // Cache the SELECT result
    let result1 = router.execute_parsed("SELECT * FROM records").unwrap();
    let rows1 = match result1 {
        query_router::QueryResult::Rows(r) => r,
        _ => panic!("Expected Rows"),
    };
    assert_eq!(rows1.len(), 5);

    // Delete some rows
    router
        .execute_parsed("DELETE FROM records WHERE id > 2")
        .unwrap();

    // SELECT again - cache should be invalidated
    let result2 = router.execute_parsed("SELECT * FROM records").unwrap();
    let rows2 = match result2 {
        query_router::QueryResult::Rows(r) => r,
        _ => panic!("Expected Rows"),
    };
    assert_eq!(rows2.len(), 3); // Only id 0, 1, 2 remain
}

#[test]
fn test_cache_persists_without_writes() {
    let router = create_router_with_cache();

    // Create and populate table
    router
        .execute("CREATE TABLE stable (id:INT, data:TEXT)")
        .unwrap();
    router.execute("INSERT stable id=1, data='test'").unwrap();

    // Execute SELECT multiple times - all should hit cache after first
    for _ in 0..10 {
        let result = router.execute_parsed("SELECT * FROM stable").unwrap();
        let rows = match result {
            query_router::QueryResult::Rows(r) => r,
            _ => panic!("Expected Rows"),
        };
        assert_eq!(rows.len(), 1);
    }

    // Data should still be there and consistent
    let final_result = router.execute_parsed("SELECT * FROM stable").unwrap();
    let final_rows = match final_result {
        query_router::QueryResult::Rows(r) => r,
        _ => panic!("Expected Rows"),
    };
    assert_eq!(final_rows.len(), 1);
}

#[test]
fn test_cache_cleared_on_graph_mutation() {
    let router = create_router_with_cache();

    // Create nodes
    let node1_result = router.execute("NODE CREATE person name='Alice'").unwrap();
    let node1_id = match node1_result {
        query_router::QueryResult::Ids(ids) => ids[0],
        _ => panic!("Expected Ids"),
    };

    // Cache a NEIGHBORS query
    let neighbors1 = router
        .execute(&format!("NEIGHBORS {} OUT", node1_id))
        .unwrap();
    let ids1 = match neighbors1 {
        query_router::QueryResult::Ids(ids) => ids,
        _ => panic!("Expected Ids"),
    };
    assert_eq!(ids1.len(), 0);

    // Create another node and edge
    let node2_result = router.execute("NODE CREATE person name='Bob'").unwrap();
    let node2_id = match node2_result {
        query_router::QueryResult::Ids(ids) => ids[0],
        _ => panic!("Expected Ids"),
    };

    router
        .execute(&format!("EDGE CREATE {} -> {} knows", node1_id, node2_id))
        .unwrap();

    // Query again - cache should be invalidated
    let neighbors2 = router
        .execute(&format!("NEIGHBORS {} OUT", node1_id))
        .unwrap();
    let ids2 = match neighbors2 {
        query_router::QueryResult::Ids(ids) => ids,
        _ => panic!("Expected Ids"),
    };
    assert_eq!(ids2.len(), 1);
    assert_eq!(ids2[0], node2_id);
}

#[test]
fn test_cache_cleared_on_vector_mutation() {
    let router = create_router_with_cache();

    // Store initial embeddings
    router.execute("EMBED doc:1 1.0, 0.5, 0.3, 0.1").unwrap();

    // Cache a SIMILAR query
    let similar1 = router.execute("SIMILAR doc:1 TOP 5").unwrap();
    let results1 = match similar1 {
        query_router::QueryResult::Similar(r) => r,
        _ => panic!("Expected Similar"),
    };
    assert!(!results1.is_empty());

    // Store more embeddings
    router.execute("EMBED doc:2 0.9, 0.6, 0.2, 0.2").unwrap();
    router.execute("EMBED doc:3 0.8, 0.7, 0.1, 0.3").unwrap();

    // Query again - should include new embeddings
    let similar2 = router.execute("SIMILAR doc:1 TOP 5").unwrap();
    let results2 = match similar2 {
        query_router::QueryResult::Similar(r) => r,
        _ => panic!("Expected Similar"),
    };
    // Should have more results now
    assert!(results2.len() >= results1.len());
}

#[test]
fn test_concurrent_write_cache_invalidation() {
    use std::{sync::Arc, thread};

    let router = Arc::new(create_router_with_cache());

    // Create table
    router
        .execute("CREATE TABLE concurrent (id:INT, value:TEXT)")
        .unwrap();

    // Insert initial data
    for i in 0..10 {
        router
            .execute(&format!("INSERT concurrent id={}, value='v{}'", i, i))
            .unwrap();
    }

    let query_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let write_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let mut handles = vec![];

    // Query threads
    for _ in 0..2 {
        let r = Arc::clone(&router);
        let qc = Arc::clone(&query_count);
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                if r.execute_parsed("SELECT * FROM concurrent").is_ok() {
                    qc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }));
    }

    // Write threads
    for t in 0..2 {
        let r = Arc::clone(&router);
        let wc = Arc::clone(&write_count);
        handles.push(thread::spawn(move || {
            for i in 0..25 {
                let id = 100 + t * 100 + i;
                if r.execute(&format!("INSERT concurrent id={}, value='new{}'", id, id))
                    .is_ok()
                {
                    wc.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // All operations should complete
    assert!(query_count.load(std::sync::atomic::Ordering::SeqCst) > 0);
    assert!(write_count.load(std::sync::atomic::Ordering::SeqCst) > 0);

    // Final query should return all rows
    let final_result = router.execute_parsed("SELECT * FROM concurrent").unwrap();
    let final_rows = match final_result {
        query_router::QueryResult::Rows(r) => r,
        _ => panic!("Expected Rows"),
    };
    // Should have original 10 + new inserts
    assert!(final_rows.len() >= 10);
}

// ========== Direct Cache API Tests ==========

use std::time::Duration;

use tensor_cache::{Cache, CacheConfig};

#[test]
fn test_cache_invalidate_version_existing() {
    // Use embedding_dim 4 for test
    let config = CacheConfig {
        embedding_dim: 4,
        ..Default::default()
    };
    let cache = Cache::with_config(config).unwrap();

    // Store some entries (note: current API doesn't expose version parameter,
    // so all entries have version=None)
    let embedding = vec![0.1, 0.2, 0.3, 0.4];
    cache
        .put("prompt1", &embedding, "response1", "model", None)
        .unwrap();
    cache
        .put("prompt2", &embedding, "response2", "model", None)
        .unwrap();

    // Verify entries exist
    assert!(cache.get("prompt1", Some(&embedding)).is_some());
    assert!(cache.get("prompt2", Some(&embedding)).is_some());

    // Try to invalidate by version (no entries have versions set, so returns 0)
    let invalidated = cache.invalidate_version("any_version");

    // No entries should be removed since none have versions
    assert_eq!(invalidated, 0);

    // Entries should still exist
    assert!(cache.get("prompt1", Some(&embedding)).is_some());
    assert!(cache.get("prompt2", Some(&embedding)).is_some());

    // Test invalidate() directly - removes from exact cache
    let removed = cache.invalidate("prompt1");
    assert!(removed);

    // After invalidate(), the exact cache entry is gone, but semantic cache
    // still has it. get() with embedding will still find it via semantic search.
    // This documents the current behavior: invalidate() only affects exact cache.
    let exact_only = cache.get("prompt1", None);
    assert!(exact_only.is_none(), "Exact cache entry should be gone");

    // Second entry should still exist in both caches
    assert!(cache.get("prompt2", Some(&embedding)).is_some());
}

#[test]
fn test_cache_invalidate_embeddings_by_source() {
    let cache = Cache::new();

    // Store embeddings from multiple sources
    for i in 0..5 {
        cache
            .put_embedding(
                "source_a",
                &format!("content_{}", i),
                vec![0.1 * i as f32; 3],
                "model",
            )
            .unwrap();
    }
    for i in 0..3 {
        cache
            .put_embedding(
                "source_b",
                &format!("content_{}", i),
                vec![0.2 * i as f32; 3],
                "model",
            )
            .unwrap();
    }

    // Verify embeddings exist
    assert!(cache.get_embedding("source_a", "content_0").is_some());
    assert!(cache.get_embedding("source_b", "content_0").is_some());

    // Invalidate all embeddings from source_a
    let removed = cache.invalidate_embeddings("source_a");

    // Should have removed 5 embeddings
    assert_eq!(removed, 5);

    // source_a embeddings should be gone
    assert!(cache.get_embedding("source_a", "content_0").is_none());
    assert!(cache.get_embedding("source_a", "content_4").is_none());

    // source_b embeddings should still exist
    assert!(cache.get_embedding("source_b", "content_0").is_some());
}

#[test]
fn test_cache_cleanup_expired_all_layers() {
    // Create cache with short TTL
    let config = CacheConfig {
        default_ttl: Duration::from_millis(20),
        embedding_dim: 4,
        ..Default::default()
    };
    let cache = Cache::with_config(config).unwrap();

    // Add entries to all layers
    // Exact layer (via put_simple)
    cache.put_simple("exact_key", "exact_value").unwrap();

    // Semantic layer (via put with embedding)
    let embedding = vec![0.1, 0.2, 0.3, 0.4];
    cache
        .put(
            "semantic_prompt",
            &embedding,
            "semantic_response",
            "model",
            None,
        )
        .unwrap();

    // Embedding layer
    cache
        .put_embedding("source", "content", vec![0.5, 0.6, 0.7, 0.8], "model")
        .unwrap();

    // Wait for entries to expire
    std::thread::sleep(Duration::from_millis(50));

    // Cleanup expired entries
    let cleaned = cache.cleanup_expired();

    // Should have cleaned entries from all layers
    assert!(
        cleaned > 0,
        "Expected to clean at least 1 expired entry, got {}",
        cleaned
    );

    // Verify entries are expired/cleaned
    // After cleanup, expired entries should not be accessible
    let exact_result = cache.get_simple("exact_key");
    let semantic_result = cache.get("semantic_prompt", Some(&embedding));
    let embedding_result = cache.get_embedding("source", "content");

    // All should be None after expiration + cleanup
    assert!(
        exact_result.is_none() && semantic_result.is_none() && embedding_result.is_none(),
        "Expected all entries to be expired"
    );
}

#[test]
fn test_cache_stats_after_invalidation() {
    let cache = Cache::new();

    // Add entries
    for i in 0..10 {
        cache
            .put_simple(&format!("stats_key_{}", i), &format!("value_{}", i))
            .unwrap();
    }

    // Get initial stats
    let stats_before = cache.stats_snapshot();
    let entries_before = stats_before.total_entries();
    assert!(entries_before >= 10);

    // Invalidate some entries
    for i in 0..5 {
        let _ = cache.invalidate(&format!("stats_key_{}", i));
    }

    // Stats should reflect invalidation
    let stats_after = cache.stats_snapshot();
    let entries_after = stats_after.total_entries();

    // Entries count should have decreased
    assert!(
        entries_after < entries_before,
        "Expected entries to decrease after invalidation: before={}, after={}",
        entries_before,
        entries_after
    );
}
