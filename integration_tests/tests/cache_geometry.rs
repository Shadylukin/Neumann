//! Integration tests for tensor_cache geometric operations.
//!
//! Tests distance metric configuration, auto-selection, and consistency
//! with tensor_store geometric primitives.

use integration_tests::sample_embeddings_normalized;
use tensor_cache::{Cache, CacheConfig, CacheLayer, DistanceMetric, SparseVector};

fn create_sparse_embedding(dim: usize, sparsity: f32) -> Vec<f32> {
    let non_zero_count = ((1.0 - sparsity) * dim as f32) as usize;
    let mut v = vec![0.0; dim];
    for i in 0..non_zero_count {
        v[i] = ((i + 1) as f32).sqrt();
    }
    // Normalize
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        v.iter_mut().for_each(|x| *x /= mag);
    }
    v
}

#[test]
fn test_cache_with_cosine_metric() {
    let config = CacheConfig {
        embedding_dim: 32,
        distance_metric: DistanceMetric::Cosine,
        auto_select_metric: false,
        ..Default::default()
    };
    let cache = Cache::with_config(config);

    let embeddings = sample_embeddings_normalized(3, 32);

    cache
        .put("First query", &embeddings[0], "First response", "gpt-4", 0)
        .unwrap();

    // Similar embedding should get a hit
    let similar = embeddings[0]
        .iter()
        .enumerate()
        .map(|(i, &x)| if i == 0 { x * 0.99 } else { x })
        .collect::<Vec<_>>();

    // Normalize the similar embedding
    let mag: f32 = similar.iter().map(|x| x * x).sum::<f32>().sqrt();
    let similar: Vec<f32> = similar.iter().map(|x| x / mag).collect();

    let hit = cache.get("Different query", Some(&similar)).unwrap();
    assert_eq!(hit.layer, CacheLayer::Semantic);
    assert!(hit.similarity.unwrap() > 0.9);
    assert_eq!(hit.metric_used, Some(DistanceMetric::Cosine));
}

#[test]
fn test_cache_with_jaccard_metric() {
    let config = CacheConfig {
        embedding_dim: 32,
        distance_metric: DistanceMetric::Jaccard,
        auto_select_metric: false,
        semantic_threshold: 0.3, // Lower for Jaccard on dense vectors
        ..Default::default()
    };
    let cache = Cache::with_config(config);

    let embeddings = sample_embeddings_normalized(3, 32);

    cache
        .put("First query", &embeddings[0], "First response", "gpt-4", 0)
        .unwrap();

    let hit = cache.get("Different query", Some(&embeddings[0])).unwrap();
    assert_eq!(hit.layer, CacheLayer::Semantic);
    assert_eq!(hit.metric_used, Some(DistanceMetric::Jaccard));
}

#[test]
fn test_cache_auto_metric_selection_dense() {
    let config = CacheConfig {
        embedding_dim: 32,
        distance_metric: DistanceMetric::Cosine,
        auto_select_metric: true,
        sparsity_metric_threshold: 0.7,
        ..Default::default()
    };
    let cache = Cache::with_config(config);

    // Dense embedding (low sparsity)
    let embedding = sample_embeddings_normalized(1, 32)[0].clone();

    cache
        .put("Dense query", &embedding, "Dense response", "gpt-4", 0)
        .unwrap();

    let hit = cache.get("Different", Some(&embedding)).unwrap();
    // Dense embedding should use Cosine
    assert_eq!(hit.metric_used, Some(DistanceMetric::Cosine));
}

#[test]
fn test_cache_auto_metric_selection_sparse() {
    let config = CacheConfig {
        embedding_dim: 32,
        distance_metric: DistanceMetric::Cosine,
        auto_select_metric: true,
        sparsity_metric_threshold: 0.5,
        semantic_threshold: 0.3,
        ..Default::default()
    };
    let cache = Cache::with_config(config);

    // Sparse embedding (>70% zeros)
    let embedding = create_sparse_embedding(32, 0.8);

    cache
        .put("Sparse query", &embedding, "Sparse response", "gpt-4", 0)
        .unwrap();

    let hit = cache.get("Different", Some(&embedding)).unwrap();
    // Sparse embedding should use Jaccard
    assert_eq!(hit.metric_used, Some(DistanceMetric::Jaccard));
}

#[test]
fn test_cache_get_with_explicit_metric() {
    let config = CacheConfig {
        embedding_dim: 32,
        semantic_threshold: 0.3,
        ..Default::default()
    };
    let cache = Cache::with_config(config);

    let embedding = sample_embeddings_normalized(1, 32)[0].clone();

    cache
        .put("Query", &embedding, "Response", "gpt-4", 0)
        .unwrap();

    // Query with explicit Euclidean metric
    let hit = cache
        .get_with_metric(
            "Different",
            Some(&embedding),
            Some(&DistanceMetric::Euclidean),
        )
        .unwrap();

    assert_eq!(hit.layer, CacheLayer::Semantic);
    assert_eq!(hit.metric_used, Some(DistanceMetric::Euclidean));
}

#[test]
fn test_cache_sparse_embeddings_preset() {
    let mut config = CacheConfig::sparse_embeddings();
    config.embedding_dim = 32;

    assert_eq!(config.distance_metric, DistanceMetric::Jaccard);
    assert!(config.auto_select_metric);
    assert!((config.sparsity_metric_threshold - 0.5).abs() < 0.001);

    let cache = Cache::with_config(config);

    // Should be functional
    let embedding = create_sparse_embedding(32, 0.8);
    cache
        .put("Test", &embedding, "Response", "model", 0)
        .unwrap();
    assert!(cache.get("Test", None).is_some());
}

#[test]
fn test_cache_metric_consistency_with_tensor_store() {
    // Verify that the cache uses the same SparseVector primitives as tensor_store
    let v1 = vec![1.0, 2.0, 0.0, 0.0];
    let v2 = vec![1.0, 2.1, 0.0, 0.0];

    let sv1 = SparseVector::from_dense(&v1);
    let sv2 = SparseVector::from_dense(&v2);

    // Compute similarities using tensor_store primitives
    let cosine = sv1.cosine_similarity(&sv2);
    let jaccard = sv1.jaccard_index(&sv2);

    // Both should return valid values
    assert!(cosine.is_finite());
    assert!(jaccard.is_finite());
    assert!(cosine > 0.9); // Should be very similar
    assert!(jaccard > 0.0);
}

#[test]
fn test_cache_hit_similarity_range() {
    let config = CacheConfig {
        embedding_dim: 32,
        ..Default::default()
    };
    let cache = Cache::with_config(config);

    let embeddings = sample_embeddings_normalized(2, 32);

    cache
        .put("Query", &embeddings[0], "Response", "gpt-4", 0)
        .unwrap();

    // Exact match should have similarity close to 1.0
    let hit = cache.get("Different", Some(&embeddings[0])).unwrap();
    assert_eq!(hit.layer, CacheLayer::Semantic);
    assert!(hit.similarity.unwrap() >= 0.99);
}

#[test]
fn test_cache_metric_used_in_exact_vs_semantic() {
    let config = CacheConfig {
        embedding_dim: 32,
        ..Default::default()
    };
    let cache = Cache::with_config(config);

    let embedding = sample_embeddings_normalized(1, 32)[0].clone();

    cache
        .put("Exact query", &embedding, "Response", "gpt-4", 0)
        .unwrap();

    // Exact match should not have metric_used
    let exact_hit = cache.get("Exact query", None).unwrap();
    assert_eq!(exact_hit.layer, CacheLayer::Exact);
    assert!(exact_hit.metric_used.is_none());

    // Semantic match should have metric_used
    let semantic_hit = cache.get("Different query", Some(&embedding)).unwrap();
    assert_eq!(semantic_hit.layer, CacheLayer::Semantic);
    assert!(semantic_hit.metric_used.is_some());
}

#[test]
fn test_cache_sparse_vector_reexport() {
    // Verify that SparseVector is properly re-exported from tensor_cache
    let v = vec![1.0, 0.0, 2.0, 0.0, 3.0];
    let sv = SparseVector::from_dense(&v);

    // Should be able to use all SparseVector methods
    assert_eq!(sv.nnz(), 3); // 3 non-zero elements
    assert!(sv.sparsity() > 0.3);

    let normalized = sv.normalize().unwrap();
    let mag: f32 = normalized
        .to_dense()
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
    assert!((mag - 1.0).abs() < 0.001);
}

#[test]
fn test_cache_distance_metric_reexport() {
    // Verify that DistanceMetric is properly re-exported from tensor_cache
    let metrics = [
        DistanceMetric::Cosine,
        DistanceMetric::Jaccard,
        DistanceMetric::Euclidean,
        DistanceMetric::Angular,
    ];

    for metric in &metrics {
        // Should be able to use the metric
        let config = CacheConfig {
            embedding_dim: 8,
            distance_metric: metric.clone(),
            ..Default::default()
        };
        let _ = Cache::with_config(config);
    }
}
