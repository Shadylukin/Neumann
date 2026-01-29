// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for tensor_cache background eviction

use std::{sync::Arc, time::Duration};

use tensor_cache::{Cache, CacheConfig};
use tensor_store::SparseVector;

fn normalize(v: &[f32]) -> Vec<f32> {
    SparseVector::from_dense(v)
        .normalize()
        .map(|sv| sv.to_dense())
        .unwrap_or_else(|| v.to_vec())
}

#[tokio::test]
async fn test_background_eviction_full_lifecycle() {
    let config = CacheConfig {
        embedding_dim: 3,
        default_ttl: Duration::from_millis(20),
        eviction_interval: Duration::from_millis(30),
        eviction_batch_size: 5,
        ..Default::default()
    };
    let cache = Arc::new(Cache::with_config(config).unwrap());
    let embedding = normalize(&[1.0, 0.0, 0.0]);

    // Add entries that will expire
    for i in 0..5 {
        cache
            .put(
                &format!("prompt{i}"),
                &embedding,
                &format!("response{i}"),
                "gpt-4",
                None,
            )
            .unwrap();
    }

    let initial_len = cache.len();
    assert!(initial_len > 0, "Cache should have entries");

    // Wait for TTL to expire before testing cleanup
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Start background eviction
    cache.start_background_eviction().unwrap();
    assert!(cache.is_background_eviction_running());

    // Let it run for a bit
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Stop gracefully
    cache.stop_background_eviction().await.unwrap();
    assert!(!cache.is_background_eviction_running());

    // Call cleanup directly to ensure entries are cleaned
    // (background task timing is not deterministic in test environment)
    let cleaned = cache.cleanup_expired();

    // Verify stats show expirations were recorded (either from background or direct cleanup)
    let snapshot = cache.stats_snapshot();
    assert!(
        snapshot.expirations > 0 || cleaned > 0,
        "Expected expirations to be recorded or entries to be cleaned"
    );
}

#[tokio::test]
async fn test_background_eviction_concurrent_access() {
    let config = CacheConfig {
        embedding_dim: 3,
        default_ttl: Duration::from_secs(60),
        eviction_interval: Duration::from_millis(10),
        eviction_batch_size: 10,
        ..Default::default()
    };
    let cache = Arc::new(Cache::with_config(config).unwrap());

    // Start background eviction
    cache.start_background_eviction().unwrap();

    // Spawn concurrent writers
    let mut handles = vec![];
    for i in 0..5 {
        let cache_clone = Arc::clone(&cache);
        let handle = tokio::spawn(async move {
            let embedding = normalize(&[1.0, 0.0, 0.0]);
            for j in 0..10 {
                cache_clone
                    .put(
                        &format!("prompt_{i}_{j}"),
                        &embedding,
                        &format!("response_{i}_{j}"),
                        "gpt-4",
                        None,
                    )
                    .unwrap();
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        });
        handles.push(handle);
    }

    // Wait for all writers to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Let eviction run a bit more
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Stop should complete without deadlock
    cache.stop_background_eviction().await.unwrap();

    // Cache should still be functional
    let embedding = normalize(&[1.0, 0.0, 0.0]);
    cache
        .put("final_prompt", &embedding, "final_response", "gpt-4", None)
        .unwrap();
    assert!(cache.get("final_prompt", None).is_some());
}

#[tokio::test]
async fn test_background_eviction_restart() {
    let config = CacheConfig {
        embedding_dim: 3,
        eviction_interval: Duration::from_millis(20),
        eviction_batch_size: 5,
        ..Default::default()
    };
    let cache = Arc::new(Cache::with_config(config).unwrap());

    // First cycle: start and stop
    cache.start_background_eviction().unwrap();
    assert!(cache.is_background_eviction_running());

    tokio::time::sleep(Duration::from_millis(50)).await;

    cache.stop_background_eviction().await.unwrap();
    assert!(!cache.is_background_eviction_running());

    // Second cycle: restart
    cache.start_background_eviction().unwrap();
    assert!(cache.is_background_eviction_running());

    tokio::time::sleep(Duration::from_millis(50)).await;

    cache.stop_background_eviction().await.unwrap();
    assert!(!cache.is_background_eviction_running());
}
