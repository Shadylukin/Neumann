// SPDX-License-Identifier: MIT OR Apache-2.0
//! Product Quantization for memory-efficient vector storage.
//!
//! Product Quantization (PQ) compresses vectors by:
//! 1. Splitting the vector into M subspaces
//! 2. Training K centroids per subspace using k-means
//! 3. Encoding each subvector as its nearest centroid index
//!
//! This achieves compression ratios of 100-1000x with acceptable recall loss.
//!
//! # Example
//!
//! ```rust
//! use tensor_store::pq::{PQConfig, PQCodebook};
//!
//! // Configure PQ: 8 subspaces, 256 centroids each (8 bits per code)
//! let config = PQConfig::default();
//!
//! // Train on a set of vectors
//! let vectors: Vec<Vec<f32>> = (0..100)
//!     .map(|i| (0..64).map(|j| (i * j) as f32 / 1000.0).collect())
//!     .collect();
//! let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
//! let codebook = PQCodebook::train(&refs, &config);
//!
//! // Encode and decode a vector
//! let original = &vectors[0];
//! let encoded = codebook.encode(original);
//! let decoded = codebook.decode(&encoded);
//!
//! // Encoded is 8 bytes vs 256 bytes for the original (32x compression)
//! assert_eq!(encoded.codes.len(), 8);
//! ```

use serde::{Deserialize, Serialize};

use crate::delta_vector::{KMeans, KMeansConfig};

/// Configuration for Product Quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQConfig {
    /// Number of subspaces (M). The vector is split into M parts.
    /// Typical values: 8, 16, 32. Higher values = better recall but more memory.
    pub num_subspaces: usize,
    /// Number of centroids per subspace (K = 2^`bits_per_subspace`).
    /// Typically 256 (8 bits) for good balance of memory and recall.
    pub num_centroids: usize,
    /// K-means configuration for training codebooks.
    pub kmeans_config: KMeansConfig,
}

impl Default for PQConfig {
    fn default() -> Self {
        Self {
            num_subspaces: 8,
            num_centroids: 256, // 8 bits per code
            kmeans_config: KMeansConfig::default(),
        }
    }
}

impl PQConfig {
    /// Create a PQ config optimized for high compression.
    ///
    /// Uses fewer subspaces for maximum compression at some recall cost.
    #[must_use]
    pub fn high_compression() -> Self {
        Self {
            num_subspaces: 4,
            num_centroids: 256,
            kmeans_config: KMeansConfig::default(),
        }
    }

    /// Create a PQ config optimized for high recall.
    ///
    /// Uses more subspaces for better recall at higher memory cost.
    #[must_use]
    pub fn high_recall() -> Self {
        Self {
            num_subspaces: 32,
            num_centroids: 256,
            kmeans_config: KMeansConfig::default(),
        }
    }

    /// Set the number of subspaces.
    #[must_use]
    pub const fn with_num_subspaces(mut self, num_subspaces: usize) -> Self {
        self.num_subspaces = num_subspaces;
        self
    }

    /// Set the number of centroids per subspace.
    #[must_use]
    pub const fn with_num_centroids(mut self, num_centroids: usize) -> Self {
        self.num_centroids = num_centroids;
        self
    }

    /// Set the k-means configuration.
    #[must_use]
    pub const fn with_kmeans_config(mut self, config: KMeansConfig) -> Self {
        self.kmeans_config = config;
        self
    }
}

/// Trained PQ codebook storing M * K centroids.
///
/// The codebook is organized as M subspaces, each with K centroids.
/// Each centroid has dimension `original_dim / M`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCodebook {
    /// Dimension of each subspace.
    subspace_dim: usize,
    /// Number of subspaces (M).
    num_subspaces: usize,
    /// Number of centroids per subspace (K).
    num_centroids: usize,
    /// Flattened centroid storage: `[M * K * subspace_dim]`.
    /// Layout: `centroids[m][k][d] = centroids[(m * K + k) * subspace_dim + d]`
    centroids: Vec<f32>,
    /// Original vector dimension.
    original_dim: usize,
}

impl PQCodebook {
    /// Train a PQ codebook from a set of vectors.
    ///
    /// Each subspace's codebook is trained independently using k-means.
    ///
    /// # Panics
    ///
    /// Panics if vectors have different dimensions or if dimension
    /// is not divisible by `num_subspaces`.
    #[must_use]
    pub fn train(vectors: &[&[f32]], config: &PQConfig) -> Self {
        if vectors.is_empty() {
            return Self {
                subspace_dim: 0,
                num_subspaces: config.num_subspaces,
                num_centroids: config.num_centroids,
                centroids: Vec::new(),
                original_dim: 0,
            };
        }

        let dim = vectors[0].len();
        assert!(
            dim.is_multiple_of(config.num_subspaces),
            "Vector dimension ({dim}) must be divisible by num_subspaces ({})",
            config.num_subspaces
        );

        let subspace_dim = dim / config.num_subspaces;
        let num_centroids = config.num_centroids.min(vectors.len());
        let kmeans = KMeans::new(config.kmeans_config.clone());

        // Train each subspace independently
        let mut centroids = Vec::with_capacity(config.num_subspaces * num_centroids * subspace_dim);

        for m in 0..config.num_subspaces {
            let start = m * subspace_dim;
            let end = start + subspace_dim;

            // Extract subvectors for this subspace
            let subvectors: Vec<Vec<f32>> =
                vectors.iter().map(|v| v[start..end].to_vec()).collect();

            // Train k-means on subvectors
            let subspace_centroids = kmeans.fit(&subvectors, num_centroids);

            // Flatten and append centroids
            for centroid in &subspace_centroids {
                centroids.extend_from_slice(centroid);
            }

            // Pad with zeros if we got fewer centroids than requested
            let actual_centroids = subspace_centroids.len();
            if actual_centroids < num_centroids {
                centroids.resize(
                    centroids.len() + (num_centroids - actual_centroids) * subspace_dim,
                    0.0,
                );
            }
        }

        Self {
            subspace_dim,
            num_subspaces: config.num_subspaces,
            num_centroids,
            centroids,
            original_dim: dim,
        }
    }

    /// Encode a vector to PQ codes.
    ///
    /// Each subvector is assigned to its nearest centroid.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Intentional: centroid index fits in u8
    pub fn encode(&self, vector: &[f32]) -> PQVector {
        if self.subspace_dim == 0 || vector.len() != self.original_dim {
            return PQVector {
                codes: vec![0; self.num_subspaces],
            };
        }

        let mut codes = Vec::with_capacity(self.num_subspaces);

        for m in 0..self.num_subspaces {
            let subvector_start = m * self.subspace_dim;
            let subvector = &vector[subvector_start..subvector_start + self.subspace_dim];

            // Find nearest centroid for this subspace
            let mut best_idx = 0u8;
            let mut best_dist = f32::MAX;

            for k in 0..self.num_centroids {
                let centroid_start = (m * self.num_centroids + k) * self.subspace_dim;
                let centroid = &self.centroids[centroid_start..centroid_start + self.subspace_dim];

                let dist = Self::squared_euclidean(subvector, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = k as u8;
                }
            }

            codes.push(best_idx);
        }

        PQVector { codes }
    }

    /// Decode PQ codes to an approximate vector.
    ///
    /// Reconstructs the vector by concatenating the centroids.
    #[must_use]
    pub fn decode(&self, pq: &PQVector) -> Vec<f32> {
        if self.subspace_dim == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.original_dim);

        for (m, &code) in pq.codes.iter().enumerate() {
            let centroid_start = (m * self.num_centroids + usize::from(code)) * self.subspace_dim;
            let centroid_end = centroid_start + self.subspace_dim;

            if centroid_end <= self.centroids.len() {
                result.extend_from_slice(&self.centroids[centroid_start..centroid_end]);
            } else {
                // Invalid code, fill with zeros
                result.resize(result.len() + self.subspace_dim, 0.0);
            }
        }

        result
    }

    /// Create an ADC (Asymmetric Distance Computation) table for a query.
    ///
    /// Precomputes distances from the query to all centroids, enabling
    /// O(M) distance computation to any PQ-encoded vector.
    #[must_use]
    pub fn compute_adc_table(&self, query: &[f32]) -> ADCTable {
        if self.subspace_dim == 0 || query.len() != self.original_dim {
            return ADCTable {
                distances: Vec::new(),
                num_subspaces: self.num_subspaces,
                num_centroids: self.num_centroids,
            };
        }

        let mut distances = Vec::with_capacity(self.num_subspaces * self.num_centroids);

        for m in 0..self.num_subspaces {
            let query_start = m * self.subspace_dim;
            let query_sub = &query[query_start..query_start + self.subspace_dim];

            for k in 0..self.num_centroids {
                let centroid_start = (m * self.num_centroids + k) * self.subspace_dim;
                let centroid = &self.centroids[centroid_start..centroid_start + self.subspace_dim];

                distances.push(Self::squared_euclidean(query_sub, centroid));
            }
        }

        ADCTable {
            distances,
            num_subspaces: self.num_subspaces,
            num_centroids: self.num_centroids,
        }
    }

    /// Returns the number of subspaces.
    #[must_use]
    pub const fn num_subspaces(&self) -> usize {
        self.num_subspaces
    }

    /// Returns the number of centroids per subspace.
    #[must_use]
    pub const fn num_centroids(&self) -> usize {
        self.num_centroids
    }

    /// Returns the subspace dimension.
    #[must_use]
    pub const fn subspace_dim(&self) -> usize {
        self.subspace_dim
    }

    /// Returns the original vector dimension.
    #[must_use]
    pub const fn original_dim(&self) -> usize {
        self.original_dim
    }

    /// Returns memory usage in bytes.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn memory_bytes(&self) -> usize {
        // centroids + struct overhead
        self.centroids.len() * std::mem::size_of::<f32>() + 32
    }

    #[inline]
    fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }
}

/// PQ-encoded vector (M bytes for 8-bit codes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PQVector {
    /// Centroid indices for each subspace.
    pub codes: Vec<u8>,
}

impl PQVector {
    /// Create a new PQ vector with the given codes.
    #[must_use]
    pub const fn new(codes: Vec<u8>) -> Self {
        Self { codes }
    }

    /// Returns the number of bytes used by this encoded vector.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn memory_bytes(&self) -> usize {
        self.codes.len()
    }

    /// Returns the number of subspaces (codes).
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn num_subspaces(&self) -> usize {
        self.codes.len()
    }
}

/// ADC (Asymmetric Distance Computation) lookup table for fast distance computation.
///
/// Precomputes distances from a query to all centroids, enabling O(M) distance
/// computation to any PQ-encoded vector instead of O(D).
#[derive(Debug, Clone)]
pub struct ADCTable {
    /// Precomputed squared distances: `[M * K]`.
    /// Layout: `distances[m * K + k] = ||query_m - centroid_m_k||^2`
    distances: Vec<f32>,
    /// Number of subspaces.
    num_subspaces: usize,
    /// Number of centroids per subspace.
    num_centroids: usize,
}

impl ADCTable {
    /// Compute squared Euclidean distance to a PQ vector in O(M).
    ///
    /// Uses lookup table for fast computation.
    #[inline]
    #[must_use]
    pub fn squared_distance(&self, pq: &PQVector) -> f32 {
        if self.distances.is_empty() || pq.codes.len() != self.num_subspaces {
            return f32::MAX;
        }

        pq.codes
            .iter()
            .enumerate()
            .map(|(m, &code)| {
                let idx = m * self.num_centroids + usize::from(code);
                self.distances.get(idx).copied().unwrap_or(0.0)
            })
            .sum()
    }

    /// Compute Euclidean distance to a PQ vector.
    #[inline]
    #[must_use]
    pub fn distance(&self, pq: &PQVector) -> f32 {
        self.squared_distance(pq).sqrt()
    }

    /// Returns the number of subspaces.
    #[must_use]
    pub const fn num_subspaces(&self) -> usize {
        self.num_subspaces
    }

    /// Returns memory usage in bytes.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn memory_bytes(&self) -> usize {
        self.distances.len() * std::mem::size_of::<f32>() + 16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn pq_config_default() {
        let config = PQConfig::default();
        assert_eq!(config.num_subspaces, 8);
        assert_eq!(config.num_centroids, 256);
    }

    #[test]
    fn pq_config_high_compression() {
        let config = PQConfig::high_compression();
        assert_eq!(config.num_subspaces, 4);
    }

    #[test]
    fn pq_config_high_recall() {
        let config = PQConfig::high_recall();
        assert_eq!(config.num_subspaces, 32);
    }

    #[test]
    fn pq_config_builder() {
        let config = PQConfig::default()
            .with_num_subspaces(16)
            .with_num_centroids(128);
        assert_eq!(config.num_subspaces, 16);
        assert_eq!(config.num_centroids, 128);
    }

    #[test]
    fn pq_codebook_train_basic() {
        let vectors = create_test_vectors(100, 64);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = PQConfig::default().with_num_subspaces(8);
        let codebook = PQCodebook::train(&refs, &config);

        assert_eq!(codebook.num_subspaces(), 8);
        assert_eq!(codebook.subspace_dim(), 8);
        assert_eq!(codebook.original_dim(), 64);
    }

    #[test]
    fn pq_encode_decode_roundtrip() {
        let vectors = create_test_vectors(100, 64);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = PQConfig::default()
            .with_num_subspaces(8)
            .with_num_centroids(16);
        let codebook = PQCodebook::train(&refs, &config);

        let original = &vectors[0];
        let encoded = codebook.encode(original);
        let decoded = codebook.decode(&encoded);

        // Check dimensions
        assert_eq!(encoded.codes.len(), 8);
        assert_eq!(decoded.len(), 64);

        // Reconstruction should be reasonably close
        let error: f32 = original
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Error should be bounded (not exact, but not completely wrong)
        assert!(error < 5.0, "Reconstruction error too high: {error}");
    }

    #[test]
    fn pq_adc_table_correctness() {
        let vectors = create_test_vectors(50, 32);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = PQConfig::default()
            .with_num_subspaces(4)
            .with_num_centroids(8);
        let codebook = PQCodebook::train(&refs, &config);

        let query = &vectors[0];
        let target = &vectors[1];

        let adc_table = codebook.compute_adc_table(query);
        let encoded_target = codebook.encode(target);

        // ADC distance should approximate true distance
        let adc_dist = adc_table.distance(&encoded_target);

        // Compare with brute force
        let decoded = codebook.decode(&encoded_target);
        let true_dist = PQCodebook::squared_euclidean(query, &decoded).sqrt();

        // ADC should match decoded distance closely
        let diff = (adc_dist - true_dist).abs();
        assert!(
            diff < 0.01,
            "ADC distance mismatch: {adc_dist} vs {true_dist}"
        );
    }

    #[test]
    fn pq_memory_savings() {
        let vectors = create_test_vectors(100, 768);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = PQConfig::default().with_num_subspaces(8);
        let codebook = PQCodebook::train(&refs, &config);

        let encoded = codebook.encode(&vectors[0]);

        let dense_bytes = 768 * std::mem::size_of::<f32>();
        let pq_bytes = encoded.memory_bytes();

        // Should achieve significant compression
        let compression_ratio = dense_bytes as f32 / pq_bytes as f32;
        assert!(
            compression_ratio > 100.0,
            "Compression ratio too low: {compression_ratio}x"
        );
    }

    #[test]
    fn pq_empty_vectors() {
        let refs: Vec<&[f32]> = Vec::new();
        let config = PQConfig::default();
        let codebook = PQCodebook::train(&refs, &config);

        assert_eq!(codebook.original_dim(), 0);
        assert_eq!(codebook.subspace_dim(), 0);
    }

    #[test]
    fn pq_dimension_validation() {
        let vectors = create_test_vectors(10, 64);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        // 8 subspaces divides 64 evenly
        let config = PQConfig::default().with_num_subspaces(8);
        let codebook = PQCodebook::train(&refs, &config);
        assert_eq!(codebook.subspace_dim(), 8);
    }

    #[test]
    #[should_panic(expected = "must be divisible")]
    fn pq_dimension_not_divisible() {
        let vectors = create_test_vectors(10, 65);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = PQConfig::default().with_num_subspaces(8);
        let _ = PQCodebook::train(&refs, &config);
    }

    #[test]
    fn pq_vector_memory_bytes() {
        let pq = PQVector::new(vec![0; 16]);
        assert_eq!(pq.memory_bytes(), 16);
        assert_eq!(pq.num_subspaces(), 16);
    }

    #[test]
    fn pq_adc_empty_query() {
        let _config = PQConfig::default();
        let codebook = PQCodebook {
            subspace_dim: 0,
            num_subspaces: 8,
            num_centroids: 256,
            centroids: Vec::new(),
            original_dim: 0,
        };

        let table = codebook.compute_adc_table(&[]);
        assert!(table.distances.is_empty());
    }

    #[test]
    fn pq_codebook_memory_bytes() {
        let vectors = create_test_vectors(50, 64);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = PQConfig::default()
            .with_num_subspaces(8)
            .with_num_centroids(16);
        let codebook = PQCodebook::train(&refs, &config);

        let memory = codebook.memory_bytes();
        // 8 subspaces * 16 centroids * 8 dims * 4 bytes + overhead
        let expected_min = 8 * 16 * 8 * 4;
        assert!(memory >= expected_min);
    }

    #[test]
    fn pq_squared_distance_vs_distance() {
        let vectors = create_test_vectors(50, 32);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = PQConfig::default()
            .with_num_subspaces(4)
            .with_num_centroids(8);
        let codebook = PQCodebook::train(&refs, &config);

        let query = &vectors[0];
        let target = codebook.encode(&vectors[1]);
        let adc = codebook.compute_adc_table(query);

        let sq_dist = adc.squared_distance(&target);
        let dist = adc.distance(&target);

        assert!((dist * dist - sq_dist).abs() < 0.001);
    }

    #[test]
    fn pq_adc_mismatched_codes() {
        let adc = ADCTable {
            distances: vec![1.0; 32], // 4 subspaces * 8 centroids
            num_subspaces: 4,
            num_centroids: 8,
        };

        // Wrong number of codes
        let pq = PQVector::new(vec![0; 2]);
        let dist = adc.squared_distance(&pq);
        assert_eq!(dist, f32::MAX);
    }
}
