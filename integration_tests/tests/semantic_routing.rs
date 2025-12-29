//! Integration tests for semantic partitioner.
//!
//! Tests basic semantic routing functionality.

use tensor_store::{Partitioner, SemanticPartitioner, SemanticPartitionerConfig};

// ============================================================================
// Helper Functions
// ============================================================================

fn make_embedding(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 17 + i) as f32 / 100.0).sin())
        .collect()
}

fn make_clustered_embedding(cluster: usize, variation: usize, dim: usize) -> Vec<f32> {
    let base_angle = (cluster as f32) * std::f32::consts::PI / 4.0;
    let var_offset = (variation as f32) * 0.1;

    (0..dim)
        .map(|i| ((i as f32) * 0.1 + base_angle + var_offset).cos())
        .collect()
}

// ============================================================================
// Basic Semantic Routing Tests
// ============================================================================

#[test]
fn test_semantic_partitioner_creation() {
    let config = SemanticPartitionerConfig::new("node0", 3, 64).with_similarity_threshold(0.5);
    let partitioner = SemanticPartitioner::new(config);

    // Initially no nodes until we add them
    assert_eq!(partitioner.nodes().len(), 0);
}

#[test]
fn test_semantic_partitioner_with_nodes() {
    let nodes = vec![
        "node0".to_string(),
        "node1".to_string(),
        "node2".to_string(),
    ];
    let config = SemanticPartitionerConfig::new("node0", 3, 64).with_similarity_threshold(0.5);
    let partitioner = SemanticPartitioner::with_nodes(config, nodes.clone());

    assert_eq!(partitioner.nodes().len(), 3);
}

#[test]
fn test_semantic_partition_basic() {
    let nodes = vec![
        "node0".to_string(),
        "node1".to_string(),
        "node2".to_string(),
    ];
    let config = SemanticPartitionerConfig::new("node0", 3, 64).with_similarity_threshold(0.5);
    let partitioner = SemanticPartitioner::with_nodes(config, nodes.clone());

    // Partition without embedding (uses consistent hash fallback)
    let result = partitioner.partition("test_key");
    assert!(nodes.contains(&result.primary));
}

#[test]
fn test_semantic_partition_deterministic() {
    let nodes = vec![
        "node0".to_string(),
        "node1".to_string(),
        "node2".to_string(),
    ];
    let config = SemanticPartitionerConfig::new("node0", 3, 64).with_similarity_threshold(0.5);
    let partitioner = SemanticPartitioner::with_nodes(config, nodes.clone());

    // Same key should always go to same node
    let key = "deterministic_test";
    let result1 = partitioner.partition(key);
    let result2 = partitioner.partition(key);

    assert_eq!(result1.primary, result2.primary);
    assert_eq!(result1.partition, result2.partition);
}

#[test]
fn test_semantic_partition_with_embedding() {
    let nodes = vec![
        "node0".to_string(),
        "node1".to_string(),
        "node2".to_string(),
    ];
    let config = SemanticPartitionerConfig::new("node0", 3, 64).with_similarity_threshold(0.5);
    let partitioner = SemanticPartitioner::with_nodes(config, nodes.clone());

    // Set up some centroids
    let centroids: Vec<Vec<f32>> = (0..3)
        .map(|i| make_clustered_embedding(i * 4, 0, 64))
        .collect();
    partitioner.set_centroids(centroids);

    // Partition with embedding
    let embedding = make_clustered_embedding(0, 1, 64);
    let result = partitioner.partition_with_embedding("test_key", &embedding);

    assert!(nodes.contains(&result.result.primary));
}

#[test]
fn test_semantic_partition_clusters_similar() {
    let nodes = vec![
        "node0".to_string(),
        "node1".to_string(),
        "node2".to_string(),
    ];
    let config = SemanticPartitionerConfig::new("node0", 3, 64).with_similarity_threshold(0.3);
    let partitioner = SemanticPartitioner::with_nodes(config, nodes.clone());

    // Set distinct centroids
    let centroids: Vec<Vec<f32>> = (0..3)
        .map(|i| make_clustered_embedding(i * 4, 0, 64))
        .collect();
    partitioner.set_centroids(centroids);

    // Route embeddings from cluster 0
    let mut shard_counts = std::collections::HashMap::new();
    for variation in 0..10 {
        let embedding = make_clustered_embedding(0, variation, 64);
        let result =
            partitioner.partition_with_embedding(&format!("key_{}", variation), &embedding);
        *shard_counts
            .entry(result.result.primary.clone())
            .or_insert(0) += 1;
    }

    // Most should go to same shard
    let max_count = *shard_counts.values().max().unwrap_or(&0);
    assert!(
        max_count >= 5,
        "Expected similar embeddings to co-locate, got max {} in one shard",
        max_count
    );
}

#[test]
fn test_fallback_for_empty_embedding() {
    let nodes = vec!["node0".to_string(), "node1".to_string()];
    let config = SemanticPartitionerConfig::new("node0", 2, 64).with_similarity_threshold(0.5);
    let partitioner = SemanticPartitioner::with_nodes(config, nodes.clone());

    // Empty embedding should fall back to consistent hash
    let result = partitioner.partition_with_embedding("test", &[]);
    assert!(nodes.contains(&result.result.primary));
}

#[test]
fn test_fallback_for_wrong_dimension() {
    let nodes = vec!["node0".to_string(), "node1".to_string()];
    let config = SemanticPartitionerConfig::new("node0", 2, 64).with_similarity_threshold(0.5);
    let partitioner = SemanticPartitioner::with_nodes(config, nodes.clone());

    // Wrong dimension should fall back
    let wrong_dim: Vec<f32> = vec![1.0; 32];
    let result = partitioner.partition_with_embedding("test", &wrong_dim);
    assert!(nodes.contains(&result.result.primary));
}

#[test]
fn test_single_node_cluster() {
    let nodes = vec!["only_node".to_string()];
    let config = SemanticPartitionerConfig::new("only_node", 1, 64).with_similarity_threshold(0.5);
    let partitioner = SemanticPartitioner::with_nodes(config, nodes.clone());

    // All keys go to the only node
    for i in 0..10 {
        let result = partitioner.partition(&format!("key_{}", i));
        assert_eq!(result.primary, "only_node");
    }
}

#[test]
fn test_semantic_partition_performance() {
    let nodes: Vec<String> = (0..8).map(|i| format!("node{}", i)).collect();
    let config = SemanticPartitionerConfig::new("node0", 8, 128).with_similarity_threshold(0.5);
    let partitioner = SemanticPartitioner::with_nodes(config, nodes);

    // Set centroids
    let centroids: Vec<Vec<f32>> = (0..8).map(|i| make_embedding(i * 100, 128)).collect();
    partitioner.set_centroids(centroids);

    // Benchmark routing
    let start = std::time::Instant::now();
    let iterations = 10_000;

    for i in 0..iterations {
        let embedding = make_embedding(i, 128);
        partitioner.partition_with_embedding(&format!("key_{}", i), &embedding);
    }

    let elapsed = start.elapsed();
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    // Should be fast (lower threshold for coverage instrumentation)
    assert!(
        ops_per_sec > 30_000.0,
        "Expected >30k ops/sec, got {:.0}",
        ops_per_sec
    );
}
