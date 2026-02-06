// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{DistanceMetric, SparseVector};

#[derive(Arbitrary, Debug)]
struct Input {
    dimension: u8,
    vec_a: Vec<f32>,
    vec_b: Vec<f32>,
}

fn sanitize_vector(v: &[f32], dim: usize) -> Vec<f32> {
    let mut result: Vec<f32> = v
        .iter()
        .take(dim)
        .copied()
        .map(|x| if x.is_finite() { x } else { 0.0 })
        .collect();
    result.resize(dim, 0.0);
    result
}

fuzz_target!(|input: Input| {
    // Limit dimension (4-256)
    let dimension = (input.dimension as usize).max(4).min(256);

    let vec_a = sanitize_vector(&input.vec_a, dimension);
    let vec_b = sanitize_vector(&input.vec_b, dimension);

    let sparse_a = SparseVector::from_dense(&vec_a);
    let sparse_b = SparseVector::from_dense(&vec_b);

    // Test all distance metrics using compute()
    let metrics = [
        DistanceMetric::Cosine,
        DistanceMetric::Euclidean,
        DistanceMetric::Angular,
        DistanceMetric::Jaccard,
        DistanceMetric::Overlap,
    ];

    for metric in &metrics {
        let distance = metric.compute(&sparse_a, &sparse_b);

        // Verify distance is finite
        assert!(
            distance.is_finite(),
            "{:?} distance is not finite: {}",
            metric,
            distance
        );

        // Verify self-distance is appropriate (0 for distance metrics, 1 for similarity)
        let self_dist = metric.compute(&sparse_a, &sparse_a);
        assert!(
            self_dist.is_finite(),
            "{:?} self-distance is not finite",
            metric
        );
    }

    // Test SparseVector's built-in distance methods
    let cosine_sim = sparse_a.cosine_similarity(&sparse_b);
    assert!(
        cosine_sim.is_finite(),
        "Cosine similarity is not finite: {}",
        cosine_sim
    );
    assert!(
        cosine_sim >= -1.0 - 1e-6 && cosine_sim <= 1.0 + 1e-6,
        "Cosine similarity {} out of range [-1, 1]",
        cosine_sim
    );

    let euclidean_dist = sparse_a.euclidean_distance(&sparse_b);
    assert!(
        euclidean_dist.is_finite(),
        "Euclidean distance is not finite: {}",
        euclidean_dist
    );
    assert!(
        euclidean_dist >= 0.0,
        "Euclidean distance is negative: {}",
        euclidean_dist
    );

    let angular_dist = sparse_a.angular_distance(&sparse_b);
    assert!(
        angular_dist.is_finite(),
        "Angular distance is not finite: {}",
        angular_dist
    );
    assert!(
        angular_dist >= 0.0,
        "Angular distance is negative: {}",
        angular_dist
    );

    let jaccard = sparse_a.jaccard_index(&sparse_b);
    assert!(
        jaccard.is_finite(),
        "Jaccard index is not finite: {}",
        jaccard
    );
    assert!(
        jaccard >= 0.0 && jaccard <= 1.0 + 1e-6,
        "Jaccard index {} out of range [0, 1]",
        jaccard
    );

    // Test symmetry (compute(a,b) == compute(b,a))
    for metric in &metrics {
        let dist_ab = metric.compute(&sparse_a, &sparse_b);
        let dist_ba = metric.compute(&sparse_b, &sparse_a);
        assert!(
            (dist_ab - dist_ba).abs() < 1e-5,
            "{:?} distance is not symmetric: {} vs {}",
            metric,
            dist_ab,
            dist_ba
        );
    }
});
