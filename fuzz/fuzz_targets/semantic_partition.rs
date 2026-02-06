// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{Partitioner, SemanticPartitioner, SemanticPartitionerConfig};

#[derive(Arbitrary, Debug)]
struct SemanticPartitionInput {
    // Number of shards (1-8)
    num_shards: u8,
    // Embedding dimension (limit to avoid OOM)
    embedding_dim: u8,
    // Similarity threshold
    similarity_threshold: u8,
    // Keys to partition
    keys: Vec<String>,
    // Embeddings (flattened)
    embeddings_flat: Vec<f32>,
    // Centroids (flattened)
    centroids_flat: Vec<f32>,
}

fuzz_target!(|input: SemanticPartitionInput| {
    // Constrain parameters
    let num_shards = (input.num_shards as usize).clamp(1, 8);
    let embedding_dim = (input.embedding_dim as usize).clamp(4, 64);
    let similarity_threshold = (input.similarity_threshold as f32 / 255.0).clamp(0.1, 0.99);

    // Create node list
    let nodes: Vec<String> = (0..num_shards).map(|i| format!("node{}", i)).collect();

    let config = SemanticPartitionerConfig::new("node0", num_shards, embedding_dim)
        .with_similarity_threshold(similarity_threshold);

    let partitioner = SemanticPartitioner::with_nodes(config, nodes.clone());

    // Property 1: All nodes should be present
    assert_eq!(partitioner.nodes().len(), num_shards);

    // Build and set centroids
    let mut centroids = Vec::new();
    let mut flat_idx = 0;
    for _ in 0..num_shards {
        if flat_idx + embedding_dim > input.centroids_flat.len() {
            break;
        }
        let centroid: Vec<f32> = input
            .centroids_flat
            .iter()
            .skip(flat_idx)
            .take(embedding_dim)
            .copied()
            .map(|x| if x.is_finite() { x } else { 0.0 })
            .collect();
        if centroid.iter().any(|x| *x != 0.0) {
            centroids.push(centroid);
        }
        flat_idx += embedding_dim;
    }
    if !centroids.is_empty() {
        partitioner.set_centroids(centroids);
    }

    // Property 2: Partition without embedding is deterministic
    for key in input.keys.iter().take(20) {
        let result1 = partitioner.partition(key);
        let result2 = partitioner.partition(key);
        assert_eq!(
            result1.primary, result2.primary,
            "Partitioning must be deterministic"
        );
        assert!(
            nodes.contains(&result1.primary),
            "Result must be a valid node"
        );
    }

    // Property 3: Partition with embedding works
    let mut emb_idx = 0;
    for key in input.keys.iter().take(10) {
        if emb_idx + embedding_dim <= input.embeddings_flat.len() {
            let embedding: Vec<f32> = input
                .embeddings_flat
                .iter()
                .skip(emb_idx)
                .take(embedding_dim)
                .copied()
                .collect();

            // Skip invalid embeddings
            if embedding.iter().all(|x| x.is_finite()) && embedding.iter().any(|x| *x != 0.0) {
                let result = partitioner.partition_with_embedding(key, &embedding);
                assert!(
                    nodes.contains(&result.result.primary),
                    "Semantic routing must return valid node"
                );

                // Determinism check
                let result2 = partitioner.partition_with_embedding(key, &embedding);
                assert_eq!(
                    result.result.primary, result2.result.primary,
                    "Semantic routing must be deterministic"
                );
            }

            emb_idx += embedding_dim;
        }
    }

    // Property 4: Empty embedding falls back gracefully
    let result = partitioner.partition_with_embedding("test_key", &[]);
    assert!(nodes.contains(&result.result.primary));

    // Property 5: Wrong dimension embedding falls back gracefully
    let wrong_dim: Vec<f32> = vec![1.0; embedding_dim / 2];
    let result = partitioner.partition_with_embedding("test_key", &wrong_dim);
    assert!(nodes.contains(&result.result.primary));

    // Property 6: Zero vector embedding falls back gracefully
    let zero_vec: Vec<f32> = vec![0.0; embedding_dim];
    let result = partitioner.partition_with_embedding("test_key", &zero_vec);
    assert!(nodes.contains(&result.result.primary));
});
