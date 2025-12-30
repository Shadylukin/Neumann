#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{DistanceMetric, SparseVector};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    dimension: u8,
    positions_a: Vec<u8>,
    values_a: Vec<f32>,
    positions_b: Vec<u8>,
    values_b: Vec<f32>,
}

fuzz_target!(|input: FuzzInput| {
    // Limit dimension to avoid OOM
    let dimension = (input.dimension as usize).max(1).min(128);

    // Filter positions to be in range
    let positions_a: Vec<u32> = input
        .positions_a
        .iter()
        .map(|&p| (p as u32) % (dimension as u32))
        .collect();
    let positions_b: Vec<u32> = input
        .positions_b
        .iter()
        .map(|&p| (p as u32) % (dimension as u32))
        .collect();

    // Ensure all values are finite
    if !input.values_a.iter().all(|x| x.is_finite())
        || !input.values_b.iter().all(|x| x.is_finite())
    {
        return;
    }

    // Build sparse vectors from positions and values
    let len_a = positions_a.len().min(input.values_a.len());
    let len_b = positions_b.len().min(input.values_b.len());

    if len_a == 0 && len_b == 0 {
        return;
    }

    // Create dense vectors and then convert to sparse
    let mut dense_a = vec![0.0; dimension];
    for i in 0..len_a {
        let pos = positions_a[i] as usize;
        if pos < dimension {
            dense_a[pos] = input.values_a[i];
        }
    }

    let mut dense_b = vec![0.0; dimension];
    for i in 0..len_b {
        let pos = positions_b[i] as usize;
        if pos < dimension {
            dense_b[pos] = input.values_b[i];
        }
    }

    let sv_a = SparseVector::from_dense(&dense_a);
    let sv_b = SparseVector::from_dense(&dense_b);

    // Test all metrics
    let metrics = [
        DistanceMetric::Cosine,
        DistanceMetric::Jaccard,
        DistanceMetric::Euclidean,
        DistanceMetric::Angular,
        DistanceMetric::WeightedJaccard,
    ];

    for metric in &metrics {
        // Compute distance
        let distance = metric.compute(&sv_a, &sv_b);
        assert!(
            distance.is_finite(),
            "Distance must be finite for metric {:?}",
            metric
        );

        // Convert to similarity
        let similarity = metric.to_similarity(distance);
        assert!(
            similarity.is_finite(),
            "Similarity must be finite for metric {:?}",
            metric
        );
        assert!(
            (0.0..=1.0).contains(&similarity),
            "Similarity {} out of range for metric {:?}",
            similarity,
            metric
        );

        // Self-similarity should be maximum
        if !sv_a.is_zero() {
            let self_distance = metric.compute(&sv_a, &sv_a);
            let self_similarity = metric.to_similarity(self_distance);

            // Self-similarity should be close to 1.0 (or at least >= 0.99)
            assert!(
                self_similarity >= 0.99 || self_distance < 0.01,
                "Self-similarity should be ~1.0 for metric {:?}, got {} (distance={})",
                metric,
                self_similarity,
                self_distance
            );
        }
    }

    // Test sparsity calculation
    let sparsity_a = sv_a.sparsity();
    let sparsity_b = sv_b.sparsity();

    assert!(
        (0.0..=1.0).contains(&sparsity_a),
        "Sparsity must be in [0, 1], got {}",
        sparsity_a
    );
    assert!(
        (0.0..=1.0).contains(&sparsity_b),
        "Sparsity must be in [0, 1], got {}",
        sparsity_b
    );

    // Test normalization
    if !sv_a.is_zero() {
        if let Some(normalized) = sv_a.normalize() {
            let mag = normalized.magnitude();
            assert!(
                (mag - 1.0).abs() < 0.001,
                "Normalized vector should have magnitude 1.0, got {}",
                mag
            );
        }
    }
});
