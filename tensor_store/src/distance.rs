// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Distance metrics for geometric vector operations.
//!
//! Provides configurable distance/similarity metrics beyond cosine,
//! enabling richer geometric analysis of tensor data.

use serde::{Deserialize, Serialize};

use crate::sparse_vector::SparseVector;

/// Distance metric for vector similarity/distance computation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity (angle-based).
    /// Range: [-1, 1] where 1 = identical direction.
    #[default]
    Cosine,

    /// Angular distance: acos(cosine).
    /// Range: [0, PI] where 0 = identical.
    /// More linear than cosine for small angles.
    Angular,

    /// Geodesic distance on unit sphere.
    /// Equivalent to angular for normalized vectors.
    Geodesic,

    /// Jaccard index on non-zero positions.
    /// Range: [0, 1] where 1 = same structure.
    /// Measures structural overlap independent of values.
    Jaccard,

    /// Overlap coefficient.
    /// Range: [0, 1] where 1 = smaller is subset of larger.
    Overlap,

    /// Weighted Jaccard considering value magnitudes.
    /// Range: [0, 1] where 1 = identical values.
    WeightedJaccard,

    /// Euclidean distance (L2 norm of difference).
    /// Range: [0, inf) where 0 = identical.
    Euclidean,

    /// Manhattan distance (L1 norm of difference).
    /// Range: [0, inf) where 0 = identical.
    Manhattan,

    /// Composite metric with configurable weights.
    /// Combines cosine (angular), structural (jaccard), and magnitude (euclidean).
    Composite(GeometricConfig),
}

impl DistanceMetric {
    /// Whether higher values mean more similar.
    ///
    /// Similarity metrics (cosine, jaccard) have higher = better.
    /// Distance metrics (angular, euclidean) have lower = better.
    #[must_use]
    pub const fn higher_is_better(&self) -> bool {
        matches!(
            self,
            Self::Cosine
                | Self::Jaccard
                | Self::Overlap
                | Self::WeightedJaccard
                | Self::Composite(_)
        )
    }

    /// Compute the metric between two sparse vectors.
    ///
    /// Returns a value where interpretation depends on the metric type.
    /// Use `higher_is_better()` to determine if high or low is "similar".
    #[must_use]
    pub fn compute(&self, a: &SparseVector, b: &SparseVector) -> f32 {
        match self {
            Self::Cosine => a.cosine_similarity(b),
            Self::Angular => a.angular_distance(b),
            Self::Geodesic => a.geodesic_distance(b),
            Self::Jaccard => a.jaccard_index(b),
            Self::Overlap => a.overlap_coefficient(b),
            Self::WeightedJaccard => a.weighted_jaccard(b),
            Self::Euclidean => a.euclidean_distance(b),
            Self::Manhattan => a.manhattan_distance(b),
            Self::Composite(config) => config.compute(a, b),
        }
    }

    /// Convert raw metric value to similarity score (0-1 range, higher = more similar).
    #[must_use]
    pub fn to_similarity(&self, raw: f32) -> f32 {
        match self {
            // Already similarities
            Self::Cosine => f32::midpoint(raw, 1.0), // [-1, 1] -> [0, 1]
            Self::Jaccard | Self::Overlap | Self::WeightedJaccard | Self::Composite(_) => raw,

            // Distances need inversion
            Self::Angular | Self::Geodesic => {
                1.0 - (raw / std::f32::consts::PI) // [0, PI] -> [1, 0]
            },
            Self::Euclidean | Self::Manhattan => {
                1.0 / (1.0 + raw) // [0, inf) -> (0, 1]
            },
        }
    }
}

/// Configuration for composite geometric scoring.
///
/// Combines multiple geometric aspects into a single score:
/// - **`cosine_weight`**: Angular/directional similarity
/// - **`structural_weight`**: Jaccard positional overlap
/// - **`magnitude_weight`**: Euclidean distance (inverted)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeometricConfig {
    /// Weight for cosine similarity component.
    pub cosine_weight: f32,

    /// Weight for Jaccard structural overlap component.
    pub structural_weight: f32,

    /// Weight for magnitude/distance component (euclidean).
    pub magnitude_weight: f32,
}

impl Default for GeometricConfig {
    fn default() -> Self {
        Self {
            cosine_weight: 0.5,
            structural_weight: 0.3,
            magnitude_weight: 0.2,
        }
    }
}

impl GeometricConfig {
    /// Create a config emphasizing angular similarity.
    #[must_use]
    pub const fn angular_heavy() -> Self {
        Self {
            cosine_weight: 0.8,
            structural_weight: 0.1,
            magnitude_weight: 0.1,
        }
    }

    /// Create a config emphasizing structural overlap.
    #[must_use]
    pub const fn structural_heavy() -> Self {
        Self {
            cosine_weight: 0.2,
            structural_weight: 0.7,
            magnitude_weight: 0.1,
        }
    }

    /// Create a config for balanced conflict detection.
    #[must_use]
    pub const fn conflict_detection() -> Self {
        Self {
            cosine_weight: 0.4,
            structural_weight: 0.5, // High structural weight catches same-key conflicts
            magnitude_weight: 0.1,
        }
    }

    /// Compute composite similarity score.
    ///
    /// All components are normalized to [0, 1] where 1 = most similar.
    #[must_use]
    pub fn compute(&self, a: &SparseVector, b: &SparseVector) -> f32 {
        let total_weight = self.cosine_weight + self.structural_weight + self.magnitude_weight;
        if total_weight == 0.0 {
            return 0.0;
        }

        // Cosine: [-1, 1] -> [0, 1]
        let cosine_sim = f32::midpoint(a.cosine_similarity(b), 1.0);

        // Jaccard: already [0, 1]
        let jaccard_sim = a.jaccard_index(b);

        // Euclidean: [0, inf) -> (0, 1]
        let euclidean_dist = a.euclidean_distance(b);
        let euclidean_sim = 1.0 / (1.0 + euclidean_dist);

        self.cosine_weight.mul_add(
            cosine_sim,
            self.structural_weight
                .mul_add(jaccard_sim, self.magnitude_weight * euclidean_sim),
        ) / total_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_higher_is_better() {
        assert!(DistanceMetric::Cosine.higher_is_better());
        assert!(DistanceMetric::Jaccard.higher_is_better());
        assert!(!DistanceMetric::Angular.higher_is_better());
        assert!(!DistanceMetric::Euclidean.higher_is_better());
    }

    #[test]
    fn metric_compute_cosine() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[1.0, 0.0]);

        let sim = DistanceMetric::Cosine.compute(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn metric_compute_jaccard() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 2.0]);
        let b = SparseVector::from_dense(&[3.0, 0.0, 4.0]);

        let sim = DistanceMetric::Jaccard.compute(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6); // Same positions
    }

    #[test]
    fn to_similarity_cosine() {
        let metric = DistanceMetric::Cosine;

        assert!((metric.to_similarity(1.0) - 1.0).abs() < 1e-6);
        assert!((metric.to_similarity(-1.0) - 0.0).abs() < 1e-6);
        assert!((metric.to_similarity(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn to_similarity_angular() {
        let metric = DistanceMetric::Angular;

        assert!((metric.to_similarity(0.0) - 1.0).abs() < 1e-6); // Same direction
        assert!((metric.to_similarity(std::f32::consts::PI) - 0.0).abs() < 1e-6);
        // Opposite
    }

    #[test]
    fn to_similarity_euclidean() {
        let metric = DistanceMetric::Euclidean;

        assert!((metric.to_similarity(0.0) - 1.0).abs() < 1e-6); // Same point
        assert!((metric.to_similarity(1.0) - 0.5).abs() < 1e-6); // Unit distance
    }

    #[test]
    fn composite_identical_vectors() {
        let config = GeometricConfig::default();
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let b = SparseVector::from_dense(&[1.0, 2.0, 3.0]);

        let score = config.compute(&a, &b);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn composite_different_weights() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 1.0]);

        // Orthogonal vectors: cosine = 0, jaccard = 0, euclidean = sqrt(2)
        let angular = GeometricConfig::angular_heavy().compute(&a, &b);
        let structural = GeometricConfig::structural_heavy().compute(&a, &b);

        // Both should be low for orthogonal vectors
        assert!(angular < 0.5);
        assert!(structural < 0.5);
    }

    #[test]
    fn conflict_detection_config() {
        let config = GeometricConfig::conflict_detection();

        // Structural weight is highest
        assert!(config.structural_weight > config.cosine_weight);
        assert!(config.structural_weight > config.magnitude_weight);
    }

    #[test]
    fn metric_higher_is_better_all_variants() {
        // Similarity metrics (higher = better)
        assert!(DistanceMetric::Cosine.higher_is_better());
        assert!(DistanceMetric::Jaccard.higher_is_better());
        assert!(DistanceMetric::Overlap.higher_is_better());
        assert!(DistanceMetric::WeightedJaccard.higher_is_better());
        assert!(DistanceMetric::Composite(GeometricConfig::default()).higher_is_better());

        // Distance metrics (lower = better)
        assert!(!DistanceMetric::Angular.higher_is_better());
        assert!(!DistanceMetric::Geodesic.higher_is_better());
        assert!(!DistanceMetric::Euclidean.higher_is_better());
        assert!(!DistanceMetric::Manhattan.higher_is_better());
    }

    #[test]
    fn metric_compute_angular() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[1.0, 0.0]);
        let dist = DistanceMetric::Angular.compute(&a, &b);
        assert!(dist.abs() < 1e-5); // Same direction = 0 angle
    }

    #[test]
    fn metric_compute_geodesic() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[1.0, 0.0]);
        let dist = DistanceMetric::Geodesic.compute(&a, &b);
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn metric_compute_overlap() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 2.0]);
        let b = SparseVector::from_dense(&[3.0, 0.0, 4.0]);
        let sim = DistanceMetric::Overlap.compute(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6); // Same non-zero positions
    }

    #[test]
    fn metric_compute_weighted_jaccard() {
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let b = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let sim = DistanceMetric::WeightedJaccard.compute(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6); // Identical
    }

    #[test]
    fn metric_compute_euclidean() {
        let a = SparseVector::from_dense(&[0.0, 0.0]);
        let b = SparseVector::from_dense(&[3.0, 4.0]);
        let dist = DistanceMetric::Euclidean.compute(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6); // 3-4-5 triangle
    }

    #[test]
    fn metric_compute_manhattan() {
        let a = SparseVector::from_dense(&[0.0, 0.0]);
        let b = SparseVector::from_dense(&[3.0, 4.0]);
        let dist = DistanceMetric::Manhattan.compute(&a, &b);
        assert!((dist - 7.0).abs() < 1e-6); // 3 + 4 = 7
    }

    #[test]
    fn metric_compute_composite() {
        let config = GeometricConfig::default();
        let a = SparseVector::from_dense(&[1.0, 2.0]);
        let b = SparseVector::from_dense(&[1.0, 2.0]);
        let sim = DistanceMetric::Composite(config).compute(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6); // Identical
    }

    #[test]
    fn to_similarity_jaccard() {
        let metric = DistanceMetric::Jaccard;
        // Jaccard already returns [0, 1], so to_similarity is identity
        assert!((metric.to_similarity(0.5) - 0.5).abs() < 1e-6);
        assert!((metric.to_similarity(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn to_similarity_overlap() {
        let metric = DistanceMetric::Overlap;
        assert!((metric.to_similarity(0.75) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn to_similarity_weighted_jaccard() {
        let metric = DistanceMetric::WeightedJaccard;
        assert!((metric.to_similarity(0.8) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn to_similarity_composite() {
        let metric = DistanceMetric::Composite(GeometricConfig::default());
        assert!((metric.to_similarity(0.6) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn to_similarity_geodesic() {
        let metric = DistanceMetric::Geodesic;
        assert!((metric.to_similarity(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn to_similarity_manhattan() {
        let metric = DistanceMetric::Manhattan;
        assert!((metric.to_similarity(0.0) - 1.0).abs() < 1e-6);
        assert!((metric.to_similarity(1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn composite_zero_weight() {
        let config = GeometricConfig {
            cosine_weight: 0.0,
            structural_weight: 0.0,
            magnitude_weight: 0.0,
        };
        let a = SparseVector::from_dense(&[1.0, 2.0]);
        let b = SparseVector::from_dense(&[1.0, 2.0]);
        let score = config.compute(&a, &b);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn metric_default() {
        let metric = DistanceMetric::default();
        assert_eq!(metric, DistanceMetric::Cosine);
    }

    #[test]
    fn metric_serde() {
        let metric = DistanceMetric::Angular;
        let serialized = bitcode::serialize(&metric).unwrap();
        let deserialized: DistanceMetric = bitcode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, metric);
    }

    #[test]
    fn config_serde() {
        let config = GeometricConfig::default();
        let serialized = bitcode::serialize(&config).unwrap();
        let deserialized: GeometricConfig = bitcode::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, config);
    }

    #[test]
    fn metric_clone() {
        let metric = DistanceMetric::Composite(GeometricConfig::default());
        let cloned = metric.clone();
        assert_eq!(cloned, metric);
    }

    #[test]
    fn metric_debug() {
        let metric = DistanceMetric::Euclidean;
        let debug = format!("{:?}", metric);
        assert!(debug.contains("Euclidean"));
    }
}
