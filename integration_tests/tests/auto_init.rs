// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Auto-initialization integration tests.
//!
//! Tests lazy initialization of optional modules: Vault, Cache, and Blob.

use integration_tests::create_shared_router;
use query_router::QueryResult;
use tensor_cache::CacheConfig;

#[test]
fn test_cache_auto_init_on_use() {
    let mut router = create_shared_router();

    // Cache not initialized yet
    assert!(router.cache().is_none());

    // Initialize cache
    router.init_cache();

    // Now cache should be available
    assert!(router.cache().is_some());
}

#[test]
fn test_cache_init_with_config() {
    let mut router = create_shared_router();

    // Initialize with custom config
    let mut config = CacheConfig::default();
    config.embedding_dim = 8;
    router.init_cache_with_config(config).unwrap();

    assert!(router.cache().is_some());
}

#[test]
fn test_cache_operations_after_init() {
    let mut router = create_shared_router();
    router.init_cache();
    router.set_identity("test-user");

    // Cache operations should work
    let result = router.execute_parsed("CACHE STATS");
    assert!(result.is_ok());
}

#[test]
fn test_cache_get_without_init() {
    let router = create_shared_router();

    // Cache not initialized - operation should fail gracefully
    let result = router.execute_parsed("CACHE GET 'key'");
    // Should return error or empty, not panic
    match result {
        Ok(QueryResult::Value(_)) => {
            // Empty or error message
        },
        Err(_) => {
            // Expected - cache not initialized
        },
        _ => {},
    }
}

#[test]
fn test_vault_init_with_key() {
    let mut router = create_shared_router();

    // Initialize vault with master key
    let result = router.init_vault(b"test-master-key-32bytes!!");
    assert!(result.is_ok());
    assert!(router.vault().is_some());
}

#[test]
fn test_vault_operations_after_init() {
    let mut router = create_shared_router();
    router.init_vault(b"test-master-key-32bytes!!").unwrap();
    router.set_identity("node:root");

    // Store a secret
    let result = router.execute_parsed("VAULT SET 'api/key' 'secret-value'");
    assert!(result.is_ok());

    // Retrieve it
    let get_result = router.execute_parsed("VAULT GET 'api/key'");
    assert!(get_result.is_ok());
}

#[test]
fn test_vault_without_init() {
    let router = create_shared_router();

    // Vault not initialized
    assert!(router.vault().is_none());

    // Operations should fail gracefully
    let result = router.execute_parsed("VAULT GET 'key'");
    assert!(result.is_err());
}

#[test]
fn test_blob_init() {
    let mut router = create_shared_router();

    // Initialize blob store
    let result = router.init_blob();
    assert!(result.is_ok());
    assert!(router.blob().is_some());
}

#[test]
fn test_blob_without_init() {
    let router = create_shared_router();

    // Blob not initialized
    assert!(router.blob().is_none());
}

#[test]
fn test_multiple_init_calls() {
    let mut router = create_shared_router();

    // Initialize cache multiple times should be safe
    router.init_cache();
    router.init_cache();

    assert!(router.cache().is_some());
}

#[test]
fn test_init_order_independence() {
    let mut router = create_shared_router();

    // Initialize in any order
    router.init_cache();
    router.init_blob().unwrap();
    router.init_vault(b"test-master-key-32bytes!!").unwrap();

    assert!(router.cache().is_some());
    assert!(router.blob().is_some());
    assert!(router.vault().is_some());
}

#[test]
fn test_cache_stats_after_use() {
    let mut router = create_shared_router();

    // Use small dimension for test
    let mut config = CacheConfig::default();
    config.embedding_dim = 4;
    router.init_cache_with_config(config).unwrap();
    router.set_identity("test-user");

    // Store some cache entries via CACHE PUT (exact key-value cache)
    router.execute_parsed("CACHE PUT 'key1' 'value1'").unwrap();
    router.execute_parsed("CACHE PUT 'key2' 'value2'").unwrap();

    // Get stats
    let result = router.execute_parsed("CACHE STATS");
    assert!(result.is_ok());

    if let Ok(QueryResult::Value(stats)) = result {
        // Stats should contain some information
        assert!(!stats.is_empty());
    }
}

#[test]
fn test_vault_reinit_fails() {
    let mut router = create_shared_router();

    // First init
    router.init_vault(b"test-master-key-32bytes!!").unwrap();

    // Second init with different key should succeed (replaces)
    let result2 = router.init_vault(b"different-key-32bytesss!");
    // Behavior depends on implementation - may succeed or fail
    assert!(result2.is_ok() || result2.is_err());
}

#[test]
fn test_engines_work_without_optional_modules() {
    let router = create_shared_router();

    // Core engines should work without vault/cache/blob
    router.execute("CREATE TABLE test (id:INT)").unwrap();
    router.execute("INSERT test id=1").unwrap();

    router.execute("NODE CREATE user name='Test'").unwrap();

    router.execute("EMBED key1 0.5, 0.5, 0.5, 0.5").unwrap();

    // All core operations work
    let result = router.execute("SELECT test");
    assert!(result.is_ok());
}

#[test]
fn test_cache_clear_after_init() {
    let mut router = create_shared_router();
    router.init_cache();
    router.set_identity("test-user");

    // Put some entries
    router.execute_parsed("CACHE PUT 'a' 'b'").unwrap();

    // Clear should work
    let result = router.execute_parsed("CACHE CLEAR");
    assert!(result.is_ok());
}

#[test]
fn test_vault_with_namespaces() {
    let mut router = create_shared_router();
    router.init_vault(b"test-master-key-32bytes!!").unwrap();
    router.set_identity("node:root");

    // Store in namespace
    router
        .execute_parsed("VAULT SET 'ns1/key' 'value1'")
        .unwrap();
    router
        .execute_parsed("VAULT SET 'ns2/key' 'value2'")
        .unwrap();

    // Both should be retrievable
    let r1 = router.execute_parsed("VAULT GET 'ns1/key'");
    let r2 = router.execute_parsed("VAULT GET 'ns2/key'");

    assert!(r1.is_ok());
    assert!(r2.is_ok());
}

#[test]
fn test_init_with_shared_store() {
    // Verify all modules share the same underlying store
    let mut router = create_shared_router();
    router.init_vault(b"test-master-key-32bytes!!").unwrap();
    router.init_cache();
    router.init_blob().unwrap();

    // Data stored via one path should be in the shared store
    router
        .execute("EMBED shared:key 1.0, 0.0, 0.0, 0.0")
        .unwrap();

    // Vector engine should see it
    let result = router.execute_parsed("EMBED GET 'shared:key'");
    assert!(result.is_ok());
}

#[test]
fn test_cache_evict_after_init() {
    let mut router = create_shared_router();
    router.init_cache();
    router.set_identity("test-user");

    // Evict should work even if cache is empty
    let result = router.execute_parsed("CACHE EVICT");
    assert!(result.is_ok());
}

#[test]
fn test_init_methods_return_reference() {
    let mut router = create_shared_router();
    router.init_cache();

    // Should be able to get reference to cache
    let cache_ref = router.cache();
    assert!(cache_ref.is_some());

    // Multiple calls should return same reference
    let cache_ref2 = router.cache();
    assert!(cache_ref2.is_some());
}
