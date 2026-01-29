// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{DistanceMetric, SparseVector};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    metric_type: u8,
    embedding_a: Vec<f32>,
    embedding_b: Vec<f32>,
}

fuzz_target!(|input: FuzzInput| {
    // Filter out invalid inputs
    if input.embedding_a.is_empty() || input.embedding_b.is_empty() {
        return;
    }

    // Limit dimension to avoid OOM
    if input.embedding_a.len() > 256 || input.embedding_b.len() > 256 {
        return;
    }

    // Ensure all values are finite
    if !input.embedding_a.iter().all(|x| x.is_finite())
        || !input.embedding_b.iter().all(|x| x.is_finite())
    {
        return;
    }

    // Create sparse vectors
    let sv_a = SparseVector::from_dense(&input.embedding_a);
    let sv_b = SparseVector::from_dense(&input.embedding_b);

    // Skip if both are zero vectors
    if sv_a.is_zero() && sv_b.is_zero() {
        return;
    }

    // Select metric based on input
    let metric = match input.metric_type % 5 {
        0 => DistanceMetric::Cosine,
        1 => DistanceMetric::Jaccard,
        2 => DistanceMetric::Euclidean,
        3 => DistanceMetric::Angular,
        _ => DistanceMetric::WeightedJaccard,
    };

    // Compute distance using the metric
    let distance = metric.compute(&sv_a, &sv_b);

    // Verify distance is finite
    assert!(
        distance.is_finite(),
        "Distance must be finite for metric {:?}, got {}",
        metric,
        distance
    );

    // Convert to similarity
    let similarity = metric.to_similarity(distance);

    // Verify similarity is finite and in valid range [0, 1]
    assert!(
        similarity.is_finite(),
        "Similarity must be finite, got {} for distance {}",
        similarity,
        distance
    );
    assert!(
        (0.0..=1.0).contains(&similarity),
        "Similarity {} out of range [0, 1] for metric {:?}",
        similarity,
        metric
    );

    // Verify symmetry for symmetric metrics
    let distance_ba = metric.compute(&sv_b, &sv_a);
    assert!(
        distance_ba.is_finite(),
        "Reverse distance must be finite, got {}",
        distance_ba
    );

    // Most metrics should be symmetric
    match metric {
        DistanceMetric::Cosine
        | DistanceMetric::Jaccard
        | DistanceMetric::Euclidean
        | DistanceMetric::Angular
        | DistanceMetric::WeightedJaccard => {
            let diff = (distance - distance_ba).abs();
            assert!(
                diff < 0.0001,
                "Symmetric metric {:?} should have equal distances: {} vs {}, diff={}",
                metric,
                distance,
                distance_ba,
                diff
            );
        },
        _ => {},
    }
});
