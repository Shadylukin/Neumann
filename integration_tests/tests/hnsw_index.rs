// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! HNSW index integration tests.
//!
//! Tests Hierarchical Navigable Small World graph index operations.

use integration_tests::{create_shared_engines, sample_embeddings};
use vector_engine::{HNSWConfig, VectorEngine};

#[test]
fn test_build_hnsw_index_from_engine() {
    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings
    let embeddings = sample_embeddings(100, 64);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    // Build HNSW index
    let (_index, key_mapping) = vector.build_hnsw_index_default().unwrap();

    assert_eq!(key_mapping.len(), 100);
    assert!(key_mapping.iter().all(|k| k.starts_with("doc:")));
}

#[test]
fn test_hnsw_search_accuracy() {
    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings
    let embeddings = sample_embeddings(100, 64);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    // Build HNSW index
    let (index, key_mapping) = vector.build_hnsw_index_default().unwrap();

    // Search for doc:0
    let query = &embeddings[0];
    let results = vector
        .search_with_hnsw(&index, &key_mapping, query, 5)
        .unwrap();

    // doc:0 should be the top result (exact match)
    assert!(!results.is_empty());
    assert_eq!(results[0].key, "doc:0");
    assert!(
        results[0].score > 0.99,
        "Exact match should have score ~1.0"
    );
}

#[test]
fn test_hnsw_high_recall_config() {
    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings
    let embeddings = sample_embeddings(200, 128);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    // Build with high recall config
    let config = HNSWConfig::high_recall();
    let (index, key_mapping) = vector.build_hnsw_index(config).unwrap();

    // Search should find exact match
    let query = &embeddings[50];
    let results = vector
        .search_with_hnsw(&index, &key_mapping, query, 10)
        .unwrap();

    assert_eq!(results[0].key, "doc:50");
}

#[test]
fn test_hnsw_high_speed_config() {
    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings
    let embeddings = sample_embeddings(200, 128);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    // Build with high speed config
    let config = HNSWConfig::high_speed();
    let (index, key_mapping) = vector.build_hnsw_index(config).unwrap();

    // Search should still work (though possibly less accurate)
    let query = &embeddings[50];
    let results = vector
        .search_with_hnsw(&index, &key_mapping, query, 10)
        .unwrap();

    // Should still find doc:50 in top results (exact match is usually found)
    assert!(results.iter().any(|r| r.key == "doc:50"));
}

#[test]
fn test_hnsw_vs_brute_force() {
    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings
    let embeddings = sample_embeddings(100, 32);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    // Build HNSW index
    let (index, key_mapping) = vector.build_hnsw_index_default().unwrap();

    // Compare results for several queries
    for query_idx in [0, 25, 50, 75, 99] {
        let query = &embeddings[query_idx];

        // HNSW search
        let hnsw_results = vector
            .search_with_hnsw(&index, &key_mapping, query, 5)
            .unwrap();

        // Brute force search
        let brute_results = vector.search_similar(query, 5).unwrap();

        // Top result should match (exact queries)
        assert_eq!(
            hnsw_results[0].key, brute_results[0].key,
            "Query {} top result mismatch",
            query_idx
        );
    }
}

#[test]
fn test_hnsw_empty_index() {
    let vector = VectorEngine::new();

    // Build index from empty engine
    let (index, key_mapping) = vector.build_hnsw_index_default().unwrap();

    assert!(key_mapping.is_empty());

    // Search on empty should return empty
    let query = sample_embeddings(1, 32)[0].clone();
    let results = vector
        .search_with_hnsw(&index, &key_mapping, &query, 5)
        .unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_hnsw_single_element() {
    let (_, _, _, vector) = create_shared_engines();

    let emb = sample_embeddings(1, 32)[0].clone();
    vector.store_embedding("only_one", emb.clone()).unwrap();

    let (index, key_mapping) = vector.build_hnsw_index_default().unwrap();

    assert_eq!(key_mapping.len(), 1);

    let results = vector
        .search_with_hnsw(&index, &key_mapping, &emb, 5)
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].key, "only_one");
}

#[test]
fn test_hnsw_duplicate_vectors() {
    let (_, _, _, vector) = create_shared_engines();

    // Store same embedding with different keys
    let emb = sample_embeddings(1, 64)[0].clone();
    for i in 0..10 {
        vector
            .store_embedding(&format!("dup:{}", i), emb.clone())
            .unwrap();
    }

    let (index, key_mapping) = vector.build_hnsw_index_default().unwrap();

    assert_eq!(key_mapping.len(), 10);

    // Search with the same vector
    let results = vector
        .search_with_hnsw(&index, &key_mapping, &emb, 5)
        .unwrap();

    // All results should have perfect similarity
    for result in &results {
        assert!(
            result.score > 0.999,
            "Duplicate vector should have score ~1.0, got {}",
            result.score
        );
    }
}

#[test]
fn test_hnsw_large_dataset() {
    let (_, _, _, vector) = create_shared_engines();

    // Store 1000 embeddings
    let embeddings = sample_embeddings(1000, 64);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("large:{}", i), emb.clone())
            .unwrap();
    }

    // Build index
    let (index, key_mapping) = vector.build_hnsw_index_default().unwrap();

    assert_eq!(key_mapping.len(), 1000);

    // Verify search still works correctly
    let query = &embeddings[500];
    let results = vector
        .search_with_hnsw(&index, &key_mapping, query, 10)
        .unwrap();

    assert_eq!(results[0].key, "large:500");
}

#[test]
fn test_hnsw_different_dimensions() {
    // Test with various embedding dimensions
    for dim in [8, 32, 64, 128, 256] {
        let vector = VectorEngine::new();

        let embeddings = sample_embeddings(50, dim);
        for (i, emb) in embeddings.iter().enumerate() {
            vector
                .store_embedding(&format!("dim{}:{}", dim, i), emb.clone())
                .unwrap();
        }

        let (index, key_mapping) = vector.build_hnsw_index_default().unwrap();

        assert_eq!(key_mapping.len(), 50);

        // Search should work for any dimension
        let results = vector
            .search_with_hnsw(&index, &key_mapping, &embeddings[0], 5)
            .unwrap();

        assert!(!results.is_empty());
        assert!(results[0].key.starts_with(&format!("dim{}:", dim)));
    }
}

#[test]
fn test_hnsw_recall_at_k() {
    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings
    let embeddings = sample_embeddings(500, 64);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    // Build indices with different configs
    let default_config = HNSWConfig::default();
    let high_recall_config = HNSWConfig::high_recall();

    let (default_index, default_keys) = vector.build_hnsw_index(default_config).unwrap();
    let (recall_index, recall_keys) = vector.build_hnsw_index(high_recall_config).unwrap();

    // Test recall@10 for several queries
    let mut default_correct = 0;
    let mut recall_correct = 0;

    for query_idx in 0..50 {
        let query = &embeddings[query_idx];

        // Ground truth from brute force
        let ground_truth = vector.search_similar(query, 10).unwrap();
        let truth_keys: Vec<_> = ground_truth.iter().map(|r| &r.key).collect();

        // Default HNSW
        let default_results = vector
            .search_with_hnsw(&default_index, &default_keys, query, 10)
            .unwrap();
        default_correct += default_results
            .iter()
            .filter(|r| truth_keys.contains(&&r.key))
            .count();

        // High recall HNSW
        let recall_results = vector
            .search_with_hnsw(&recall_index, &recall_keys, query, 10)
            .unwrap();
        recall_correct += recall_results
            .iter()
            .filter(|r| truth_keys.contains(&&r.key))
            .count();
    }

    // Both should have good recall
    let default_recall = default_correct as f64 / (50.0 * 10.0);
    let high_recall = recall_correct as f64 / (50.0 * 10.0);

    assert!(
        default_recall > 0.8,
        "Default config recall too low: {:.2}",
        default_recall
    );
    assert!(
        high_recall >= default_recall,
        "High recall config should be >= default: {:.2} vs {:.2}",
        high_recall,
        default_recall
    );
}

#[test]
fn test_hnsw_index_rebuild() {
    let (_, _, _, vector) = create_shared_engines();

    // Initial embeddings (using offset to make them unique)
    let initial = sample_embeddings(50, 32);
    for (i, emb) in initial.iter().enumerate() {
        vector
            .store_embedding(&format!("initial:{}", i), emb.clone())
            .unwrap();
    }

    // Build first index
    let (_index1, keys1) = vector.build_hnsw_index_default().unwrap();
    assert_eq!(keys1.len(), 50);

    // Add more embeddings with offset to make them distinct from initial
    for i in 0..50 {
        let emb: Vec<f32> = (0..32)
            .map(|j| (((i + 100) * 32 + j) as f32).sin())
            .collect();
        vector
            .store_embedding(&format!("added:{}", i), emb)
            .unwrap();
    }

    // Rebuild index
    let (index2, keys2) = vector.build_hnsw_index_default().unwrap();
    assert_eq!(keys2.len(), 100);

    // Search should find added embedding (not initial)
    let query: Vec<f32> = (0..32)
        .map(|j| (((25 + 100) * 32 + j) as f32).sin())
        .collect();
    let results = vector.search_with_hnsw(&index2, &keys2, &query, 5).unwrap();

    assert_eq!(results[0].key, "added:25");
}

#[test]
fn test_hnsw_concurrent_search() {
    use std::{sync::Arc, thread};

    let vector = Arc::new(VectorEngine::new());

    // Store embeddings
    let embeddings = sample_embeddings(100, 64);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    // Build index
    let (index, key_mapping) = vector.build_hnsw_index_default().unwrap();
    let index = Arc::new(index);
    let key_mapping = Arc::new(key_mapping);
    let embeddings = Arc::new(embeddings);

    // Concurrent searches
    let mut handles = vec![];

    for t in 0..4 {
        let vector = Arc::clone(&vector);
        let index = Arc::clone(&index);
        let keys = Arc::clone(&key_mapping);
        let embs = Arc::clone(&embeddings);

        handles.push(thread::spawn(move || {
            let mut found = 0;
            for i in 0..25 {
                let query_idx = t * 25 + i;
                let query = &embs[query_idx];
                let results = vector.search_with_hnsw(&index, &keys, query, 5).unwrap();
                if results[0].key == format!("doc:{}", query_idx) {
                    found += 1;
                }
            }
            found
        }));
    }

    let total_found: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    // Should find most exact matches
    assert!(
        total_found > 90,
        "Expected >90% exact matches, got {}%",
        total_found
    );
}
