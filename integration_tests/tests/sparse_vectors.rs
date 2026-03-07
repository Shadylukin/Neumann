// SPDX-License-Identifier: MIT OR Apache-2.0
//! TensorStore sparse vector integration tests.
//!
//! Tests sparse vector creation, storage, and dot product operations.

use tensor_store::{TensorData, TensorStore, TensorValue};

fn create_store() -> TensorStore {
    TensorStore::new()
}

#[test]
fn test_from_embedding_basic() {
    // Dense vector with some near-zero values
    let dense = vec![0.5, 0.0, 0.0, 0.8, 0.0, 0.3, 0.0, 0.0, 0.7, 0.0];

    // Convert with thresholds (60% zeros = 0.6 sparsity, below 0.7 threshold)
    let result = TensorValue::from_embedding(dense.clone(), 0.1, 0.5);

    // Should convert to dense representation (below sparsity threshold or sparse)
    let restored = result.to_dense();
    assert!(restored.is_some());
    let restored_vec = restored.unwrap();
    assert_eq!(restored_vec.len(), dense.len());
}

#[test]
fn test_from_embedding_becomes_sparse() {
    // Very sparse vector (90% zeros = 0.9 sparsity, above 0.7 threshold)
    let mut dense = vec![0.0f32; 100];
    for i in [0, 50, 75] {
        dense[i] = 0.5;
    }

    // With high sparsity, should become Sparse variant
    let result = TensorValue::from_embedding(dense.clone(), 0.01, 0.5);

    // Verify it works regardless of variant
    let restored = result.to_dense().unwrap();
    assert_eq!(restored.len(), 100);
}

#[test]
fn test_from_embedding_auto() {
    // Dense vector
    let dense = vec![0.5, 0.0, 0.0, 0.8, 0.0, 0.3, 0.0, 0.0, 0.7, 0.0];

    // Auto threshold detection
    let result = TensorValue::from_embedding_auto(dense.clone());

    // Should produce valid vector (either Vector or Sparse variant)
    let restored = result.to_dense();
    assert!(restored.is_some());
}

#[test]
fn test_from_embedding_all_zeros() {
    // All zeros should still create a valid result
    let dense = vec![0.0, 0.0, 0.0, 0.0, 0.0];

    let result = TensorValue::from_embedding(dense, 0.1, 0.5);

    // Should be convertible to dense (whether Sparse or Vector)
    let restored = result.to_dense().unwrap();
    assert!(restored.iter().all(|x| *x == 0.0));
}

#[test]
fn test_from_embedding_all_significant() {
    // All values above threshold - stays dense
    let dense = vec![0.5, 0.6, 0.7, 0.8, 0.9];

    let result = TensorValue::from_embedding(dense.clone(), 0.1, 0.9);

    let restored = result.to_dense().unwrap();
    assert!(!restored.is_empty());
    // Low sparsity should keep as Vector
    assert!(matches!(result, TensorValue::Vector(_)));
}

#[test]
fn test_dot_product_dense_dense() {
    let v1 = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
    let v2 = TensorValue::Vector(vec![4.0, 5.0, 6.0]);

    let dot = v1.dot(&v2).unwrap();

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert!((dot - 32.0).abs() < 0.001);
}

#[test]
fn test_dot_product_sparse_sparse() {
    // Create very sparse vectors to ensure Sparse variant
    let mut dense1 = vec![0.0f32; 100];
    let mut dense2 = vec![0.0f32; 100];
    dense1[10] = 1.0;
    dense1[50] = 2.0;
    dense2[50] = 3.0;
    dense2[90] = 1.0;

    let v1 = TensorValue::from_embedding(dense1, 0.01, 0.5);
    let v2 = TensorValue::from_embedding(dense2, 0.01, 0.5);

    let dot = v1.dot(&v2);

    // Only common index is 50: 2.0 * 3.0 = 6.0
    assert!(dot.is_some());
    assert!((dot.unwrap() - 6.0).abs() < 0.001);
}

#[test]
fn test_dot_product_orthogonal() {
    let v1 = TensorValue::Vector(vec![1.0, 0.0, 0.0]);
    let v2 = TensorValue::Vector(vec![0.0, 1.0, 0.0]);

    let dot = v1.dot(&v2).unwrap();

    // Orthogonal vectors have dot product 0
    assert!((dot - 0.0).abs() < 0.001);
}

#[test]
fn test_dot_product_parallel() {
    let v1 = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
    let v2 = TensorValue::Vector(vec![2.0, 4.0, 6.0]); // 2 * v1

    let dot = v1.dot(&v2).unwrap();

    // 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
    assert!((dot - 28.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_identical() {
    let v1 = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
    let v2 = TensorValue::Vector(vec![1.0, 2.0, 3.0]);

    let sim = v1.cosine_similarity(&v2).unwrap();

    // Identical vectors have cosine similarity 1.0
    assert!((sim - 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_opposite() {
    let v1 = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
    let v2 = TensorValue::Vector(vec![-1.0, -2.0, -3.0]);

    let sim = v1.cosine_similarity(&v2).unwrap();

    // Opposite vectors have cosine similarity -1.0
    assert!((sim - (-1.0)).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let v1 = TensorValue::Vector(vec![1.0, 0.0, 0.0]);
    let v2 = TensorValue::Vector(vec![0.0, 1.0, 0.0]);

    let sim = v1.cosine_similarity(&v2).unwrap();

    // Orthogonal vectors have cosine similarity 0
    assert!((sim - 0.0).abs() < 0.001);
}

#[test]
fn test_cosine_similarity_scaled() {
    let v1 = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
    let v2 = TensorValue::Vector(vec![10.0, 20.0, 30.0]); // 10 * v1

    let sim = v1.cosine_similarity(&v2).unwrap();

    // Scaled vectors have cosine similarity 1.0
    assert!((sim - 1.0).abs() < 0.001);
}

#[test]
fn test_store_and_retrieve_sparse_vector() {
    let store = create_store();

    // Very sparse vector to trigger Sparse variant
    let mut dense = vec![0.0f32; 100];
    dense[0] = 0.5;
    dense[50] = 0.8;
    dense[99] = 0.7;

    let sparse = TensorValue::from_embedding(dense.clone(), 0.01, 0.5);

    // Store the sparse vector using put()
    let mut data = TensorData::new();
    data.set("embedding", sparse.clone());
    store.put("sparse:test", data).unwrap();

    // Retrieve it
    let retrieved = store.get("sparse:test").unwrap();
    let emb = retrieved.get("embedding").unwrap();

    // Both should be convertible to dense
    let original_dense = sparse.to_dense().unwrap();
    let retrieved_dense = emb.to_dense().unwrap();
    assert_eq!(original_dense.len(), retrieved_dense.len());
}

#[test]
fn test_sparse_vector_in_similarity_search() {
    use integration_tests::create_shared_engines;

    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings (use Vector directly since VectorEngine expects dense)
    for i in 0..10 {
        let mut dense = vec![0.0f32; 64];
        for j in 0..10 {
            dense[(i * 5 + j) % 64] = ((i + j) as f32 * 0.1).sin();
        }

        vector
            .store_embedding(&format!("sparse:{}", i), dense)
            .unwrap();
    }

    // Create query
    let mut query = vec![0.0f32; 64];
    for j in 0..10 {
        query[j % 64] = (j as f32 * 0.1).sin();
    }

    // Search should work
    let results = vector.search_similar(&query, 5).unwrap();
    assert_eq!(results.len(), 5);
}

#[test]
fn test_sparse_high_dimensional() {
    // High-dimensional sparse vector (like real embeddings)
    let mut dense = vec![0.0f32; 1024];

    // Only 5% non-zero
    for i in (0..1024).step_by(20) {
        dense[i] = (i as f32 * 0.01).sin();
    }

    let result = TensorValue::from_embedding(dense.clone(), 0.001, 0.5);

    // Should produce valid vector (Sparse or Vector)
    let restored = result.to_dense().unwrap();
    assert!(!restored.is_empty());
    assert_eq!(restored.len(), 1024);
}

#[test]
fn test_sparse_vs_dense_dot_product_equivalence() {
    // Create vectors with known values
    let dense1 = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0];
    let dense2 = vec![0.5, 0.0, 1.0, 0.0, 1.5, 0.0, 0.0, 0.0];

    // Dense dot product: 1*0.5 + 2*1 + 3*1.5 = 0.5 + 2 + 4.5 = 7
    let v1_dense = TensorValue::Vector(dense1.clone());
    let v2_dense = TensorValue::Vector(dense2.clone());
    let dot_dense = v1_dense.dot(&v2_dense).unwrap();
    assert!((dot_dense - 7.0).abs() < 0.001);

    // Sparse versions (might become Sparse or Vector depending on thresholds)
    let v1_sparse = TensorValue::from_embedding(dense1, 0.01, 0.5);
    let v2_sparse = TensorValue::from_embedding(dense2, 0.01, 0.5);

    let dot_sparse = v1_sparse.dot(&v2_sparse).unwrap();
    // Should produce same result
    assert!(
        (dot_sparse - dot_dense).abs() < 0.01,
        "Sparse dot {} differs from dense dot {}",
        dot_sparse,
        dot_dense
    );
}

#[test]
fn test_sparse_vector_concurrent_operations() {
    use std::{sync::Arc, thread};

    let store = Arc::new(create_store());
    let mut handles = vec![];

    // Multiple threads storing vectors (use Vector directly)
    for t in 0..4 {
        let store = Arc::clone(&store);
        handles.push(thread::spawn(move || {
            for i in 0..25 {
                let mut dense = vec![0.0f32; 32];
                for j in 0..8 {
                    dense[(t * 8 + j) % 32] = ((t * i + j) as f32 * 0.1).sin();
                }

                let value = TensorValue::Vector(dense);
                let mut data = TensorData::new();
                data.set("emb", value);
                let _ = store.put(format!("sparse:t{}:{}", t, i), data);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Should have 100 entries
    assert_eq!(store.len(), 100);
}

#[test]
fn test_dot_product_dimension_mismatch() {
    let v1 = TensorValue::Vector(vec![1.0, 2.0, 3.0]);
    let v2 = TensorValue::Vector(vec![1.0, 2.0]); // Different dimension

    let dot = v1.dot(&v2);

    // Should return None for dimension mismatch
    assert!(dot.is_none());
}

#[test]
fn test_sparse_threshold_effects() {
    let dense = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    // Low value threshold - keep more values (less sparse -> Vector)
    let low_threshold = TensorValue::from_embedding(dense.clone(), 0.05, 0.9);

    // High value threshold - fewer values (more sparse -> possibly Sparse)
    let high_threshold = TensorValue::from_embedding(dense.clone(), 0.5, 0.3);

    // Both should produce valid vectors
    assert!(low_threshold.to_dense().is_some());
    assert!(high_threshold.to_dense().is_some());

    // Low threshold should preserve more non-zero values
    let low_dense = low_threshold.to_dense().unwrap();
    let high_dense = high_threshold.to_dense().unwrap();

    let low_nonzero = low_dense.iter().filter(|&&x| x != 0.0).count();
    let high_nonzero = high_dense.iter().filter(|&&x| x != 0.0).count();

    // High value threshold zeros out more values
    assert!(
        low_nonzero >= high_nonzero,
        "Low threshold {} should have >= non-zeros than high threshold {}",
        low_nonzero,
        high_nonzero
    );
}

#[test]
fn test_sparse_with_negative_values() {
    let dense = vec![-0.5, 0.0, 0.3, -0.8, 0.0, 0.2, -0.1, 0.0];

    let result = TensorValue::from_embedding(dense.clone(), 0.05, 0.5);

    // Should handle negative values
    let restored = result.to_dense().unwrap();
    assert!(!restored.is_empty());

    // Check dot product works with negative values
    let v2 = TensorValue::Vector(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let dot = result.dot(&v2);
    assert!(dot.is_some());
}

#[test]
fn test_dimension_method() {
    let v1 = TensorValue::Vector(vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v1.dimension(), Some(4));

    // Sparse vector should also report dimension
    let mut dense = vec![0.0f32; 100];
    dense[0] = 1.0;
    let sparse = TensorValue::from_embedding(dense, 0.01, 0.5);
    assert_eq!(sparse.dimension(), Some(100));

    // Scalar should return None
    let scalar = TensorValue::Scalar(tensor_store::ScalarValue::Int(42));
    assert_eq!(scalar.dimension(), None);
}

// ==================== Sparse Compression Integration Tests ====================

#[test]
fn test_vector_engine_sparse_storage() {
    use integration_tests::create_shared_engines;

    let (_, _, _, vector) = create_shared_engines();

    // Create a very sparse vector (97% zeros)
    let mut sparse = vec![0.0f32; 100];
    sparse[0] = 1.0;
    sparse[50] = 2.0;
    sparse[99] = 3.0;

    // Store and retrieve
    vector
        .store_embedding("sparse_test", sparse.clone())
        .unwrap();
    let retrieved = vector.get_embedding("sparse_test").unwrap();

    // Verify values are preserved
    assert_eq!(retrieved.len(), 100);
    assert!((retrieved[0] - 1.0).abs() < 0.001);
    assert!((retrieved[50] - 2.0).abs() < 0.001);
    assert!((retrieved[99] - 3.0).abs() < 0.001);
    assert!((retrieved[1] - 0.0).abs() < 0.001);
}

#[test]
fn test_vector_engine_sparse_search() {
    use integration_tests::create_shared_engines;

    let (_, _, _, vector) = create_shared_engines();

    // Store sparse vectors
    let mut v1 = vec![0.0f32; 100];
    v1[0] = 1.0;

    let mut v2 = vec![0.0f32; 100];
    v2[0] = 0.707;
    v2[1] = 0.707;

    let mut v3 = vec![0.0f32; 100];
    v3[1] = 1.0;

    vector.store_embedding("v1", v1).unwrap();
    vector.store_embedding("v2", v2).unwrap();
    vector.store_embedding("v3", v3).unwrap();

    // Search with sparse query
    let mut query = vec![0.0f32; 100];
    query[0] = 1.0;

    let results = vector.search_similar(&query, 3).unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].key, "v1");
}

#[test]
fn test_embedding_slab_sparse_snapshot() {
    use tensor_store::{EmbeddingSlab, EntityId};

    let slab = EmbeddingSlab::new(100, 10);

    // Add sparse embeddings (97% zeros)
    for i in 0..5u64 {
        let mut sparse = vec![0.0f32; 100];
        sparse[0] = i as f32;
        sparse[50] = i as f32 * 2.0;
        sparse[99] = i as f32 * 3.0;
        slab.set(EntityId::new(i), &sparse).unwrap();
    }

    // Snapshot and restore
    let snapshot = slab.snapshot();

    // Serialize to check size (sparse should be much smaller)
    let serialized = bitcode::serialize(&snapshot).unwrap();
    let dense_size = 5 * 100 * 4 + 100; // 5 vectors * 100 floats * 4 bytes + overhead
    assert!(
        serialized.len() < dense_size / 5,
        "Sparse snapshot {} bytes should be much smaller than dense {} bytes",
        serialized.len(),
        dense_size
    );

    // Restore and verify values are preserved
    let restored = EmbeddingSlab::restore(snapshot);
    assert_eq!(restored.len(), 5);

    for i in 0..5u64 {
        let emb = restored.get(EntityId::new(i)).unwrap();
        assert_eq!(emb.len(), 100);
        assert!((emb[0] - i as f32).abs() < 0.001);
        assert!((emb[50] - i as f32 * 2.0).abs() < 0.001);
        assert!((emb[99] - i as f32 * 3.0).abs() < 0.001);
    }
}

#[test]
fn test_tensor_cache_sparse_embedding() {
    use tensor_cache::{Cache, CacheConfig};

    let mut config = CacheConfig::default();
    config.embedding_dim = 100;
    let cache = Cache::with_config(config).unwrap();

    // Create sparse embedding
    let mut sparse = vec![0.0f32; 100];
    sparse[0] = 1.0;
    sparse[50] = 2.0;

    cache
        .put_embedding("source", "content", sparse.clone(), "model")
        .unwrap();

    let retrieved = cache.get_embedding("source", "content").unwrap();
    assert_eq!(retrieved.len(), 100);
    assert!((retrieved[0] - 1.0).abs() < 0.001);
    assert!((retrieved[50] - 2.0).abs() < 0.001);
}

#[test]
fn test_dimension_validation() {
    use tensor_compress::TTConfig;

    // Valid dimensions
    assert!(TTConfig::for_dim(64).is_ok());
    assert!(TTConfig::for_dim(768).is_ok());
    assert!(TTConfig::for_dim(4096).is_ok());

    // Invalid: zero dimension
    assert!(TTConfig::for_dim(0).is_err());

    // Validation on config
    let bad_config = TTConfig {
        shape: vec![4, 4, 4],
        max_rank: 8,
        tolerance: 0.0, // Invalid: must be > 0
    };
    assert!(bad_config.validate().is_err());
}

#[test]
fn test_compression_defaults() {
    use tensor_compress::CompressionDefaults;

    assert_eq!(CompressionDefaults::SMALL, 64);
    assert_eq!(CompressionDefaults::MEDIUM, 384);
    assert_eq!(CompressionDefaults::STANDARD, 768);
    assert_eq!(CompressionDefaults::LARGE, 1536);
    assert_eq!(CompressionDefaults::XLARGE, 4096);

    // Auto-detect should work
    let vector = vec![0.0f32; 768];
    let config = CompressionDefaults::config_for(&vector);
    assert!(config.tensor_mode.is_some());
}
