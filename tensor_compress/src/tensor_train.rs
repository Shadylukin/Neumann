// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Tensor Train (TT) decomposition for high-dimensional embedding compression.
//!
//! Implements the TT-SVD algorithm from Oseledets (2011) for decomposing vectors
//! into products of smaller 3D cores, achieving 10-20x compression for 4096+ dimensions.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::manual_is_multiple_of)]

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::decompose::{left_unfold_for_tt, svd_truncated, DecomposeError, Matrix};

/// Threshold for switching to parallel batch processing.
const PARALLEL_THRESHOLD: usize = 4;

/// Errors from TT operations.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum TTError {
    #[error("dimension {dim} cannot be reshaped to {shape:?} (product: {product})")]
    ShapeMismatch {
        dim: usize,
        shape: Vec<usize>,
        product: usize,
    },
    #[error("empty vector")]
    EmptyVector,
    #[error("invalid TT-rank: must be >= 1")]
    InvalidRank,
    #[error("incompatible TT shapes for operation")]
    IncompatibleShapes,
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    #[error("invalid tolerance: {0} (must be 0 < tol <= 1)")]
    InvalidTolerance(f32),
    #[error("decomposition error: {0}")]
    Decompose(#[from] DecomposeError),
}

/// Configuration for TT decomposition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TTConfig {
    /// Shape to reshape vector into (product must equal dimension).
    /// e.g., [8, 8, 8, 8] for 4096-dim vectors.
    pub shape: Vec<usize>,
    /// Maximum TT-rank (controls compression vs accuracy tradeoff).
    pub max_rank: usize,
    /// Relative tolerance for SVD truncation (e.g., 1e-6 for high accuracy).
    pub tolerance: f32,
}

impl TTConfig {
    /// Validate this configuration.
    ///
    /// # Errors
    /// Returns error if shape is empty, rank < 1, or tolerance out of range.
    pub fn validate(&self) -> Result<(), TTError> {
        if self.shape.is_empty() {
            return Err(TTError::InvalidShape("empty shape".into()));
        }
        if self.shape.contains(&0) {
            return Err(TTError::InvalidShape("shape contains zero".into()));
        }
        if self.max_rank < 1 {
            return Err(TTError::InvalidRank);
        }
        if self.tolerance <= 0.0 || self.tolerance > 1.0 || !self.tolerance.is_finite() {
            return Err(TTError::InvalidTolerance(self.tolerance));
        }
        Ok(())
    }

    /// Create a configuration optimized for the given dimension.
    /// Automatically determines a good tensor shape.
    ///
    /// # Errors
    /// Returns `EmptyVector` if dimension is 0.
    pub fn for_dim(dim: usize) -> Result<Self, TTError> {
        if dim == 0 {
            return Err(TTError::EmptyVector);
        }
        let shape = optimal_shape(dim);
        let config = Self {
            shape,
            max_rank: 8,
            tolerance: 1e-4,
        };
        config.validate()?;
        Ok(config)
    }

    /// High compression preset (lower accuracy).
    ///
    /// # Errors
    /// Returns `EmptyVector` if dimension is 0.
    pub fn high_compression(dim: usize) -> Result<Self, TTError> {
        if dim == 0 {
            return Err(TTError::EmptyVector);
        }
        let shape = optimal_shape(dim);
        let config = Self {
            shape,
            max_rank: 4,
            tolerance: 1e-2,
        };
        config.validate()?;
        Ok(config)
    }

    /// High accuracy preset (lower compression).
    ///
    /// # Errors
    /// Returns `EmptyVector` if dimension is 0.
    pub fn high_accuracy(dim: usize) -> Result<Self, TTError> {
        if dim == 0 {
            return Err(TTError::EmptyVector);
        }
        let shape = optimal_shape(dim);
        let config = Self {
            shape,
            max_rank: 16,
            tolerance: 1e-6,
        };
        config.validate()?;
        Ok(config)
    }
}

/// Find an optimal tensor shape for the given dimension.
/// Prefers balanced shapes with small prime factors.
fn optimal_shape(dim: usize) -> Vec<usize> {
    match dim {
        64 => vec![4, 4, 4],
        128 => vec![4, 4, 8],
        256 => vec![4, 8, 8],
        384 => vec![4, 8, 12],
        512 => vec![8, 8, 8],
        768 => vec![8, 8, 12],
        1024 => vec![8, 8, 16],
        1536 => vec![8, 12, 16],
        2048 => vec![8, 16, 16],
        3072 => vec![8, 16, 24],
        4096 => vec![8, 8, 8, 8],
        8192 => vec![8, 8, 8, 16],
        _ => factorize_balanced(dim),
    }
}

/// Factorize a number into balanced factors for tensor shape.
fn factorize_balanced(n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1];
    }

    // Find factors close to cube root for 3D, fourth root for 4D, etc.
    let target_factors = ((n as f64).ln() / 2.0_f64.ln()).ceil() as usize;
    let target_factors = target_factors.clamp(2, 6);

    let mut factors = vec![];
    let mut remaining = n;

    let target_size = (remaining as f64).powf(1.0 / target_factors as f64) as usize;

    for _ in 0..target_factors - 1 {
        let mut best_factor = 1;
        for f in (2..=target_size.max(2)).rev() {
            if remaining % f == 0 {
                best_factor = f;
                break;
            }
        }
        if best_factor == 1 {
            for f in 2..=remaining {
                if remaining % f == 0 {
                    best_factor = f;
                    break;
                }
            }
        }
        factors.push(best_factor);
        remaining /= best_factor;
        if remaining == 1 {
            break;
        }
    }
    if remaining > 1 {
        factors.push(remaining);
    }

    factors.sort_unstable();
    factors
}

/// A single TT-core (3D tensor stored as flat array in row-major order).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TTCore {
    /// Flattened data: r_{k-1} x n_k x r_k elements.
    pub data: Vec<f32>,
    /// Shape: (left_rank, mode_size, right_rank).
    pub shape: (usize, usize, usize),
}

impl TTCore {
    pub fn new(data: Vec<f32>, left_rank: usize, mode_size: usize, right_rank: usize) -> Self {
        debug_assert_eq!(data.len(), left_rank * mode_size * right_rank);
        Self {
            data,
            shape: (left_rank, mode_size, right_rank),
        }
    }

    #[inline]
    pub fn left_rank(&self) -> usize {
        self.shape.0
    }

    #[inline]
    pub fn mode_size(&self) -> usize {
        self.shape.1
    }

    #[inline]
    pub fn right_rank(&self) -> usize {
        self.shape.2
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> f32 {
        let idx = i * self.shape.1 * self.shape.2 + j * self.shape.2 + k;
        self.data[idx]
    }

    /// Get the j-th slice as a left_rank x right_rank matrix.
    pub fn slice(&self, j: usize) -> Matrix {
        let (r1, _, r2) = self.shape;
        let mut data = vec![0.0; r1 * r2];
        for i in 0..r1 {
            for k in 0..r2 {
                data[i * r2 + k] = self.get(i, j, k);
            }
        }
        Matrix::new(data, r1, r2).expect("valid shape")
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Complete TT-decomposition of a vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TTVector {
    /// The TT-cores G_1, G_2, ..., G_n.
    pub cores: Vec<TTCore>,
    /// Original vector dimension.
    pub original_dim: usize,
    /// Tensor shape used for decomposition.
    pub shape: Vec<usize>,
    /// TT-ranks: [1, r_1, r_2, ..., r_{n-1}, 1].
    pub ranks: Vec<usize>,
}

impl TTVector {
    /// Total storage size in floats.
    #[must_use]
    pub fn storage_size(&self) -> usize {
        self.cores.iter().map(TTCore::size).sum()
    }

    /// Compression ratio compared to original dense vector.
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        self.original_dim as f32 / self.storage_size() as f32
    }

    /// Number of TT-cores.
    #[must_use]
    pub fn num_cores(&self) -> usize {
        self.cores.len()
    }

    /// Maximum rank across all cores.
    #[must_use]
    pub fn max_rank(&self) -> usize {
        self.ranks.iter().copied().max().unwrap_or(1)
    }
}

/// Decompose a vector into TT format using TT-SVD algorithm.
///
/// The vector is first reshaped according to the config shape, then decomposed
/// into a train of 3D cores using successive SVD truncations.
pub fn tt_decompose(vector: &[f32], config: &TTConfig) -> Result<TTVector, TTError> {
    if vector.is_empty() {
        return Err(TTError::EmptyVector);
    }

    let product: usize = config.shape.iter().product();
    if product != vector.len() {
        return Err(TTError::ShapeMismatch {
            dim: vector.len(),
            shape: config.shape.clone(),
            product,
        });
    }

    if config.max_rank < 1 {
        return Err(TTError::InvalidRank);
    }

    let n = config.shape.len();
    let mut cores = Vec::with_capacity(n);
    let mut ranks = vec![1];
    let mut current_data = vector.to_vec();
    let mut left_rank = 1;

    // TT-SVD: sweep left to right
    for k in 0..n - 1 {
        let mode_size = config.shape[k];
        let remaining_product: usize = config.shape[k + 1..].iter().product();

        let m = left_unfold_for_tt(&current_data, left_rank, mode_size);
        let svd_result = svd_truncated(&m, config.max_rank, config.tolerance)?;
        let new_rank = svd_result.rank.min(config.max_rank);
        ranks.push(new_rank);

        let mut core_data = vec![0.0; left_rank * mode_size * new_rank];
        for i in 0..left_rank {
            for j in 0..mode_size {
                for r in 0..new_rank {
                    let u_row = i * mode_size + j;
                    if r < svd_result.u.cols {
                        core_data[i * mode_size * new_rank + j * new_rank + r] =
                            svd_result.u.get(u_row, r);
                    }
                }
            }
        }
        cores.push(TTCore::new(core_data, left_rank, mode_size, new_rank));

        let mut next_data = vec![0.0; new_rank * remaining_product];
        for r in 0..new_rank {
            for j in 0..remaining_product {
                if r < svd_result.s.len() && r < svd_result.vt.rows {
                    next_data[r * remaining_product + j] =
                        svd_result.s[r] * svd_result.vt.get(r, j);
                }
            }
        }

        current_data = next_data;
        left_rank = new_rank;
    }

    let last_mode_size = config.shape[n - 1];
    ranks.push(1);

    let mut last_core_data = vec![0.0; left_rank * last_mode_size];
    for i in 0..left_rank.min(current_data.len() / last_mode_size) {
        for j in 0..last_mode_size {
            let idx = i * last_mode_size + j;
            if idx < current_data.len() {
                last_core_data[i * last_mode_size + j] = current_data[idx];
            }
        }
    }
    cores.push(TTCore::new(last_core_data, left_rank, last_mode_size, 1));

    Ok(TTVector {
        cores,
        original_dim: vector.len(),
        shape: config.shape.clone(),
        ranks,
    })
}

/// Reconstruct a dense vector from TT format.
#[must_use]
pub fn tt_reconstruct(tt: &TTVector) -> Vec<f32> {
    if tt.cores.is_empty() {
        return vec![];
    }

    let n = tt.cores.len();
    let total_size: usize = tt.shape.iter().product();
    let mut result = vec![0.0; total_size];

    for flat_idx in 0..total_size {
        let mut remaining = flat_idx;
        let mut multi_idx = vec![0usize; n];
        for k in (0..n).rev() {
            multi_idx[k] = remaining % tt.shape[k];
            remaining /= tt.shape[k];
        }

        let mut left_vec: Vec<f32> = vec![1.0];

        for (k, core) in tt.cores.iter().enumerate() {
            let j = multi_idx[k];
            let slice = core.slice(j);
            let mut new_vec = vec![0.0; core.right_rank()];
            for r in 0..core.right_rank() {
                for l in 0..core.left_rank().min(left_vec.len()) {
                    new_vec[r] += left_vec[l] * slice.get(l, r);
                }
            }
            left_vec = new_vec;
        }

        result[flat_idx] = left_vec.first().copied().unwrap_or(0.0);
    }

    result
}

/// Compute the Frobenius norm of a TT-vector without reconstruction.
#[must_use]
pub fn tt_norm(tt: &TTVector) -> f32 {
    if tt.cores.is_empty() {
        return 0.0;
    }

    let mut gram = vec![1.0f32];

    for core in &tt.cores {
        let (r1, n, r2) = core.shape;
        let mut new_gram = vec![0.0; r2 * r2];

        for a in 0..r2 {
            for b in 0..r2 {
                let mut sum = 0.0;
                for k in 0..n {
                    for i in 0..r1 {
                        for j in 0..r1 {
                            let gram_ij = if r1 == 1 {
                                gram[0]
                            } else {
                                gram.get(i * r1 + j).copied().unwrap_or(0.0)
                            };
                            sum += gram_ij * core.get(i, k, a) * core.get(j, k, b);
                        }
                    }
                }
                new_gram[a * r2 + b] = sum;
            }
        }

        gram = new_gram;
    }

    gram.first().copied().unwrap_or(0.0).sqrt()
}

/// Compute dot product of two TT-vectors without reconstruction.
pub fn tt_dot_product(a: &TTVector, b: &TTVector) -> Result<f32, TTError> {
    if a.shape != b.shape {
        return Err(TTError::IncompatibleShapes);
    }

    if a.cores.is_empty() || b.cores.is_empty() {
        return Ok(0.0);
    }

    let mut gram = vec![1.0f32];

    for (core_a, core_b) in a.cores.iter().zip(b.cores.iter()) {
        let (r1a, n, r2a) = core_a.shape;
        let (r1b, _, r2b) = core_b.shape;
        let mut new_gram = vec![0.0; r2a * r2b];

        for a_idx in 0..r2a {
            for b_idx in 0..r2b {
                let mut sum = 0.0;
                for k in 0..n {
                    for ia in 0..r1a {
                        for ib in 0..r1b {
                            let gram_idx = if gram.len() == 1 { 0 } else { ia * r1b + ib };
                            let g = gram.get(gram_idx).copied().unwrap_or(0.0);
                            sum += g * core_a.get(ia, k, a_idx) * core_b.get(ib, k, b_idx);
                        }
                    }
                }
                new_gram[a_idx * r2b + b_idx] = sum;
            }
        }

        gram = new_gram;
    }

    Ok(gram.first().copied().unwrap_or(0.0))
}

/// Compute cosine similarity between two TT-vectors without reconstruction.
pub fn tt_cosine_similarity(a: &TTVector, b: &TTVector) -> Result<f32, TTError> {
    let dot = tt_dot_product(a, b)?;
    let norm_a = tt_norm(a);
    let norm_b = tt_norm(b);

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a * norm_b))
}

/// Compute approximate Euclidean distance between two TT-vectors.
pub fn tt_euclidean_distance(a: &TTVector, b: &TTVector) -> Result<f32, TTError> {
    let dot_aa = tt_dot_product(a, a)?;
    let dot_bb = tt_dot_product(b, b)?;
    let dot_ab = tt_dot_product(a, b)?;

    let dist_sq = dot_aa + dot_bb - 2.0 * dot_ab;
    Ok(dist_sq.max(0.0).sqrt())
}

/// Scale a TT-vector by a scalar.
#[must_use]
pub fn tt_scale(tt: &TTVector, scalar: f32) -> TTVector {
    if tt.cores.is_empty() {
        return tt.clone();
    }

    let mut new_cores = tt.cores.clone();
    for val in &mut new_cores[0].data {
        *val *= scalar;
    }

    TTVector {
        cores: new_cores,
        original_dim: tt.original_dim,
        shape: tt.shape.clone(),
        ranks: tt.ranks.clone(),
    }
}

/// Batch decompose multiple vectors with the same config.
///
/// Automatically uses parallel processing when there are 4+ vectors.
pub fn tt_decompose_batch(vectors: &[&[f32]], config: &TTConfig) -> Result<Vec<TTVector>, TTError> {
    if vectors.len() >= PARALLEL_THRESHOLD {
        vectors
            .par_iter()
            .map(|v| tt_decompose(v, config))
            .collect()
    } else {
        vectors.iter().map(|v| tt_decompose(v, config)).collect()
    }
}

/// Batch cosine similarity - compute similarity of query against multiple TT vectors.
///
/// Automatically uses parallel processing when there are 4+ targets.
pub fn tt_cosine_similarity_batch(
    query: &TTVector,
    targets: &[TTVector],
) -> Result<Vec<f32>, TTError> {
    if targets.len() >= PARALLEL_THRESHOLD {
        targets
            .par_iter()
            .map(|t| tt_cosine_similarity(query, t))
            .collect()
    } else {
        targets
            .iter()
            .map(|t| tt_cosine_similarity(query, t))
            .collect()
    }
}

/// Batch dot product - compute dot product of query against multiple TT vectors.
///
/// Automatically uses parallel processing when there are 4+ targets.
pub fn tt_dot_product_batch(query: &TTVector, targets: &[TTVector]) -> Result<Vec<f32>, TTError> {
    if targets.len() >= PARALLEL_THRESHOLD {
        targets
            .par_iter()
            .map(|t| tt_dot_product(query, t))
            .collect()
    } else {
        targets.iter().map(|t| tt_dot_product(query, t)).collect()
    }
}

/// Batch Euclidean distance - compute distance of query against multiple TT vectors.
///
/// Automatically uses parallel processing when there are 4+ targets.
pub fn tt_euclidean_distance_batch(
    query: &TTVector,
    targets: &[TTVector],
) -> Result<Vec<f32>, TTError> {
    if targets.len() >= PARALLEL_THRESHOLD {
        targets
            .par_iter()
            .map(|t| tt_euclidean_distance(query, t))
            .collect()
    } else {
        targets
            .iter()
            .map(|t| tt_euclidean_distance(query, t))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tt_config_for_dim() {
        let config = TTConfig::for_dim(4096).unwrap();
        assert_eq!(config.shape, vec![8, 8, 8, 8]);
        assert_eq!(config.max_rank, 8);
    }

    #[test]
    fn test_tt_config_validation() {
        // Valid configs
        assert!(TTConfig::for_dim(64).is_ok());
        assert!(TTConfig::for_dim(4096).is_ok());
        assert!(TTConfig::high_compression(768).is_ok());
        assert!(TTConfig::high_accuracy(768).is_ok());

        // Invalid: zero dimension
        assert!(TTConfig::for_dim(0).is_err());
        assert!(TTConfig::high_compression(0).is_err());

        // Manual validation
        let bad = TTConfig {
            shape: vec![],
            max_rank: 8,
            tolerance: 0.01,
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_optimal_shape() {
        assert_eq!(optimal_shape(64), vec![4, 4, 4]);
        assert_eq!(optimal_shape(768), vec![8, 8, 12]);
        assert_eq!(optimal_shape(4096), vec![8, 8, 8, 8]);
    }

    #[test]
    fn test_factorize_balanced() {
        let factors = factorize_balanced(24);
        let product: usize = factors.iter().product();
        assert_eq!(product, 24);
    }

    #[test]
    fn test_tt_core_new() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let core = TTCore::new(data, 1, 2, 3);
        assert_eq!(core.left_rank(), 1);
        assert_eq!(core.mode_size(), 2);
        assert_eq!(core.right_rank(), 3);
        assert_eq!(core.size(), 6);
    }

    #[test]
    fn test_tt_core_get() {
        // Shape (2, 2, 2) = 8 elements
        let data: Vec<f32> = (0..8_u8).map(|i| f32::from(i)).collect();
        let core = TTCore::new(data, 2, 2, 2);
        assert!(core.get(0, 0, 0).abs() < f32::EPSILON);
        assert!((core.get(0, 0, 1) - 1.0).abs() < f32::EPSILON);
        assert!((core.get(0, 1, 0) - 2.0).abs() < f32::EPSILON);
        assert!((core.get(1, 0, 0) - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tt_decompose_simple() {
        // Simple test: decompose a 64-dim vector
        let vector: Vec<f32> = (0..64_u8).map(|i| (f32::from(i) * 0.1).sin()).collect();
        let config = TTConfig::for_dim(64).unwrap();
        let tt = tt_decompose(&vector, &config).unwrap();

        assert_eq!(tt.original_dim, 64);
        assert_eq!(tt.num_cores(), 3); // shape [4, 4, 4]
        assert!(tt.compression_ratio() > 1.0);
    }

    #[test]
    fn test_tt_decompose_empty() {
        let config = TTConfig::for_dim(64).unwrap();
        let result = tt_decompose(&[], &config);
        assert!(matches!(result, Err(TTError::EmptyVector)));
    }

    #[test]
    fn test_tt_decompose_shape_mismatch() {
        let vector: Vec<f32> = vec![1.0; 100];
        let config = TTConfig::for_dim(64).unwrap(); // expects 64
        let result = tt_decompose(&vector, &config);
        assert!(matches!(result, Err(TTError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_tt_reconstruct_roundtrip() {
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let config = TTConfig {
            shape: vec![4, 4, 4],
            max_rank: 16, // High rank for accuracy
            tolerance: 1e-8,
        };
        let tt = tt_decompose(&vector, &config).unwrap();
        let reconstructed = tt_reconstruct(&tt);

        assert_eq!(reconstructed.len(), vector.len());

        // Check reconstruction error
        let error: f32 = vector
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let orig_norm: f32 = vector.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let rel_error = error / orig_norm;

        assert!(
            rel_error < 0.1,
            "Relative reconstruction error too high: {rel_error}"
        );
    }

    #[test]
    fn test_tt_norm() {
        let vector: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let config = TTConfig::for_dim(64).unwrap();
        let tt = tt_decompose(&vector, &config).unwrap();

        let tt_norm_val = tt_norm(&tt);
        let dense_norm: f32 = vector.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();

        // TT norm should approximate dense norm
        let rel_error = (tt_norm_val - dense_norm).abs() / dense_norm;
        assert!(rel_error < 0.5, "Norm error too high: {rel_error}");
    }

    #[test]
    fn test_tt_dot_product() {
        let v1: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let v2: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).cos()).collect();
        let config = TTConfig::for_dim(64).unwrap();

        let tt1 = tt_decompose(&v1, &config).unwrap();
        let tt2 = tt_decompose(&v2, &config).unwrap();

        let tt_dot = tt_dot_product(&tt1, &tt2).unwrap();
        let dense_dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();

        // Should be reasonably close
        let rel_error = (tt_dot - dense_dot).abs() / dense_dot.abs().max(1.0);
        assert!(rel_error < 0.5, "Dot product error: {rel_error}");
    }

    #[test]
    fn test_tt_cosine_similarity() {
        let v1: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let v2: Vec<f32> = v1.clone(); // Same vector
        let config = TTConfig::for_dim(64).unwrap();

        let tt1 = tt_decompose(&v1, &config).unwrap();
        let tt2 = tt_decompose(&v2, &config).unwrap();

        let sim = tt_cosine_similarity(&tt1, &tt2).unwrap();
        // Same vector should have similarity close to 1.0
        assert!(sim > 0.9, "Self-similarity should be ~1.0, got {sim}");
    }

    #[test]
    fn test_tt_scale() {
        let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let config = TTConfig::for_dim(64).unwrap();
        let tt = tt_decompose(&vector, &config).unwrap();

        let scaled = tt_scale(&tt, 2.0);
        let norm_orig = tt_norm(&tt);
        let norm_scaled = tt_norm(&scaled);

        // Scaling by 2 should double the norm
        let expected = norm_orig * 2.0;
        let rel_error = (norm_scaled - expected).abs() / expected;
        assert!(rel_error < 0.1, "Scale error: {rel_error}");
    }

    #[test]
    fn test_tt_compression_ratio_4096() {
        // Test with realistic 4096-dim vector
        let vector: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin()).collect();
        let config = TTConfig::for_dim(4096).unwrap();
        let tt = tt_decompose(&vector, &config).unwrap();

        let ratio = tt.compression_ratio();
        // With max_rank=8 and shape [8,8,8,8], expect significant compression
        assert!(ratio > 2.0, "Expected compression ratio > 2.0, got {ratio}");
    }

    #[test]
    fn test_tt_vector_storage_size() {
        let vector: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let config = TTConfig::for_dim(64).unwrap();
        let tt = tt_decompose(&vector, &config).unwrap();

        let storage = tt.storage_size();
        assert!(storage > 0);
        assert!(storage <= 64); // Should compress
    }

    #[test]
    fn test_tt_batch_decompose() {
        let v1: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let v2: Vec<f32> = (0..64).map(|i| (i as f32).sqrt()).collect();
        let config = TTConfig::for_dim(64).unwrap();

        let batch = tt_decompose_batch(&[&v1, &v2], &config).unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_tt_euclidean_distance() {
        let v1: Vec<f32> = vec![1.0; 64];
        let v2: Vec<f32> = vec![2.0; 64];
        let config = TTConfig::for_dim(64).unwrap();

        let tt1 = tt_decompose(&v1, &config).unwrap();
        let tt2 = tt_decompose(&v2, &config).unwrap();

        let dist = tt_euclidean_distance(&tt1, &tt2).unwrap();
        let expected = 8.0; // sqrt(64 * 1^2) = 8
        assert!(
            (dist - expected).abs() < 2.0,
            "Distance error: expected ~{expected}, got {dist}"
        );
    }

    #[test]
    fn test_tt_incompatible_shapes() {
        let v1: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let v2: Vec<f32> = (0..128).map(|i| i as f32).collect();

        let tt1 = tt_decompose(&v1, &TTConfig::for_dim(64).unwrap()).unwrap();
        let tt2 = tt_decompose(&v2, &TTConfig::for_dim(128).unwrap()).unwrap();

        let result = tt_dot_product(&tt1, &tt2);
        assert!(matches!(result, Err(TTError::IncompatibleShapes)));
    }

    // === Edge case tests for robustness ===

    #[test]
    fn test_tt_constant_vector() {
        // All-same values should compress extremely well (rank 1)
        let vector = vec![1.0f32; 256];
        let config = TTConfig::for_dim(256).unwrap();
        let tt = tt_decompose(&vector, &config).unwrap();

        // Constant vector has rank-1 structure, should compress well
        let ratio = tt.compression_ratio();
        assert!(
            ratio > 5.0,
            "Constant vector should compress >5x, got {ratio:.2}x"
        );

        // Reconstruction should be accurate
        let reconstructed = tt_reconstruct(&tt);
        let max_error: f32 = reconstructed
            .iter()
            .map(|x| (x - 1.0).abs())
            .fold(0.0, f32::max);
        assert!(
            max_error < 0.1,
            "Constant vector reconstruction error: {max_error}"
        );
    }

    #[test]
    fn test_tt_zero_vector() {
        // All zeros - degenerate case
        let vector = vec![0.0f32; 64];
        let config = TTConfig::for_dim(64).unwrap();
        let tt = tt_decompose(&vector, &config).unwrap();

        let reconstructed = tt_reconstruct(&tt);
        let max_val: f32 = reconstructed.iter().map(|x| x.abs()).fold(0.0, f32::max);
        // Should reconstruct to near-zero
        assert!(
            max_val < 1e-6,
            "Zero vector should reconstruct to zeros, got max {max_val}"
        );
    }

    #[test]
    fn test_tt_single_nonzero() {
        // Single spike - worst case for TT (no low-rank structure)
        let mut vector = vec![0.0f32; 256];
        vector[128] = 1.0;
        let config = TTConfig::for_dim(256).unwrap();

        let tt = tt_decompose(&vector, &config).unwrap();
        let reconstructed = tt_reconstruct(&tt);

        // The spike should be approximately preserved
        let spike_val = reconstructed[128];
        assert!(
            spike_val > 0.5,
            "Spike should be preserved, got {spike_val}"
        );
    }

    #[test]
    fn test_tt_alternating_high_frequency() {
        // Alternating +1/-1 - high frequency signal needs high rank
        let vector: Vec<f32> = (0..256)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let config = TTConfig::high_accuracy(256).unwrap(); // Use high accuracy for this

        let tt = tt_decompose(&vector, &config).unwrap();
        let reconstructed = tt_reconstruct(&tt);

        // Check that the alternating pattern is roughly preserved
        let mut correct_signs = 0;
        for (i, &v) in reconstructed.iter().enumerate() {
            let expected_sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            if v * expected_sign > 0.0 {
                correct_signs += 1;
            }
        }
        // At least 80% of signs should be correct
        assert!(
            correct_signs > 200,
            "Only {correct_signs}/256 signs correct for alternating"
        );
    }

    #[test]
    fn test_tt_linear_ramp() {
        // Linear ramp - should have very low rank
        let vector: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();
        let config = TTConfig::for_dim(256).unwrap();

        let tt = tt_decompose(&vector, &config).unwrap();
        let ratio = tt.compression_ratio();
        // Linear functions should compress well
        assert!(
            ratio > 2.0,
            "Linear ramp should compress >2x, got {ratio:.2}x"
        );

        let reconstructed = tt_reconstruct(&tt);
        let mse: f32 = vector
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / vector.len() as f32;
        assert!(mse < 0.01, "Linear ramp MSE too high: {mse}");
    }

    #[test]
    fn test_tt_random_dense() {
        // Pseudo-random dense vector - typical neural embedding
        let vector: Vec<f32> = (0..256)
            .map(|i| {
                f32::midpoint(
                    (f32::from(i as u8) * 1.619).sin(),
                    (f32::from(i as u8) * 2.719).cos(),
                )
            })
            .collect();
        let config = TTConfig::for_dim(256).unwrap();

        let tt = tt_decompose(&vector, &config).unwrap();
        let reconstructed = tt_reconstruct(&tt);

        // Should maintain reasonable similarity
        let dot: f32 = vector.iter().zip(&reconstructed).map(|(a, b)| a * b).sum();
        let norm_orig: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_recon: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (norm_orig * norm_recon);
        assert!(
            cosine > 0.9,
            "Random vector cosine similarity too low: {cosine}"
        );
    }

    #[test]
    fn test_tt_prime_dimension_fallback() {
        // Prime dimension (127) - uses general factorization
        let vector: Vec<f32> = (0..127).map(|i| (i as f32 * 0.1).sin()).collect();
        let config = TTConfig::for_dim(127).unwrap();

        // Should not panic, may have poor compression
        let result = tt_decompose(&vector, &config);
        assert!(result.is_ok(), "Prime dimension should not crash");
    }

    #[test]
    fn test_tt_very_small_values() {
        // Denormalized floats / very small values
        let vector: Vec<f32> = (0..64).map(|i| (i as f32) * 1e-38).collect();
        let config = TTConfig::for_dim(64).unwrap();

        let tt = tt_decompose(&vector, &config).unwrap();
        let reconstructed = tt_reconstruct(&tt);

        // Should not produce NaN or Inf
        assert!(
            reconstructed.iter().all(|x| x.is_finite()),
            "Tiny values produced non-finite"
        );
    }

    #[test]
    fn test_tt_large_values() {
        // Large but finite values (1e6 range, typical for unnormalized embeddings)
        let vector: Vec<f32> = (0..64).map(|i| (i as f32) * 1e6).collect();
        let config = TTConfig::for_dim(64).unwrap();

        let tt = tt_decompose(&vector, &config).unwrap();
        let reconstructed = tt_reconstruct(&tt);

        // Should not overflow
        assert!(
            reconstructed.iter().all(|x| x.is_finite()),
            "Large values overflowed"
        );

        // Verify reasonable reconstruction
        let max_rel_error: f32 = vector
            .iter()
            .zip(&reconstructed)
            .filter(|(a, _)| a.abs() > 1.0)
            .map(|(a, b)| (a - b).abs() / a.abs())
            .fold(0.0, f32::max);
        assert!(
            max_rel_error < 0.5,
            "Large values reconstruction error: {max_rel_error}"
        );
    }

    #[test]
    fn test_tt_sparse_like_vector() {
        // 90% zeros - simulating what happens when sparse vector is densified
        let mut vector = vec![0.0f32; 256];
        for i in (0..256).step_by(10) {
            vector[i] = (i as f32 * 0.1).sin();
        }
        let config = TTConfig::for_dim(256).unwrap();

        let tt = tt_decompose(&vector, &config).unwrap();
        let reconstructed = tt_reconstruct(&tt);

        // Non-zero positions should be approximately preserved
        for i in (0..256).step_by(10) {
            let error = (vector[i] - reconstructed[i]).abs();
            let rel_error = if vector[i].abs() > 1e-6 {
                error / vector[i].abs()
            } else {
                error
            };
            assert!(
                rel_error < 0.5,
                "Sparse non-zero at {i} has error {rel_error}"
            );
        }
    }

    // === Batch operation tests ===

    #[test]
    fn test_tt_cosine_similarity_batch() {
        let config = TTConfig::for_dim(64).unwrap();
        let query: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let query_tt = tt_decompose(&query, &config).unwrap();

        // Create batch of targets
        let targets: Vec<TTVector> = (0..8)
            .map(|offset| {
                let vec: Vec<f32> = (0..64).map(|i| ((i + offset) as f32 * 0.1).cos()).collect();
                tt_decompose(&vec, &config).unwrap()
            })
            .collect();

        let similarities = tt_cosine_similarity_batch(&query_tt, &targets).unwrap();
        assert_eq!(similarities.len(), 8);

        // Verify results match individual calls
        for (i, &sim) in similarities.iter().enumerate() {
            let individual = tt_cosine_similarity(&query_tt, &targets[i]).unwrap();
            assert!(
                (sim - individual).abs() < 1e-6,
                "batch[{i}]={sim} != individual={individual}"
            );
        }
    }

    #[test]
    fn test_tt_dot_product_batch() {
        let config = TTConfig::for_dim(64).unwrap();
        let query: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let query_tt = tt_decompose(&query, &config).unwrap();

        // Create batch of targets
        let targets: Vec<TTVector> = (0..6)
            .map(|scale| {
                let vec: Vec<f32> = (0..64).map(|i| (i as f32) * (scale as f32 + 1.0)).collect();
                tt_decompose(&vec, &config).unwrap()
            })
            .collect();

        let dots = tt_dot_product_batch(&query_tt, &targets).unwrap();
        assert_eq!(dots.len(), 6);

        // Verify results match individual calls
        for (i, &dot) in dots.iter().enumerate() {
            let individual = tt_dot_product(&query_tt, &targets[i]).unwrap();
            assert!(
                (dot - individual).abs() < 1e-3,
                "batch[{i}]={dot} != individual={individual}"
            );
        }
    }

    #[test]
    fn test_tt_euclidean_distance_batch() {
        let config = TTConfig::for_dim(64).unwrap();
        let query: Vec<f32> = vec![1.0; 64];
        let query_tt = tt_decompose(&query, &config).unwrap();

        // Create batch of targets at different "distances"
        let targets: Vec<TTVector> = (0..5)
            .map(|offset| {
                let vec: Vec<f32> = vec![1.0 + offset as f32; 64];
                tt_decompose(&vec, &config).unwrap()
            })
            .collect();

        let distances = tt_euclidean_distance_batch(&query_tt, &targets).unwrap();
        assert_eq!(distances.len(), 5);

        // Verify results match individual calls
        for (i, &dist) in distances.iter().enumerate() {
            let individual = tt_euclidean_distance(&query_tt, &targets[i]).unwrap();
            assert!(
                (dist - individual).abs() < 1e-3,
                "batch[{i}]={dist} != individual={individual}"
            );
        }
    }

    #[test]
    fn test_tt_batch_decompose_parallel() {
        // Test with enough vectors to trigger parallel path
        let config = TTConfig::for_dim(64).unwrap();
        let vectors: Vec<Vec<f32>> = (0..8)
            .map(|i| (0..64).map(|j| ((i * 10 + j) as f32 * 0.1).sin()).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(std::vec::Vec::as_slice).collect();

        let batch_results = tt_decompose_batch(&refs, &config).unwrap();
        assert_eq!(batch_results.len(), 8);

        // Verify each result matches individual decomposition
        for (i, tt) in batch_results.iter().enumerate() {
            let individual = tt_decompose(&vectors[i], &config).unwrap();
            assert_eq!(tt.shape, individual.shape);
            assert_eq!(tt.original_dim, individual.original_dim);
        }
    }

    #[test]
    fn test_tt_batch_small() {
        // Test with fewer vectors (sequential path)
        let config = TTConfig::for_dim(64).unwrap();
        let vectors: Vec<Vec<f32>> = (0..2)
            .map(|i| (0..64).map(|j| ((i * 10 + j) as f32 * 0.1).cos()).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(std::vec::Vec::as_slice).collect();

        let batch_results = tt_decompose_batch(&refs, &config).unwrap();
        assert_eq!(batch_results.len(), 2);
    }

    #[test]
    fn test_tt_batch_similarity_empty() {
        let config = TTConfig::for_dim(64).unwrap();
        let query: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let query_tt = tt_decompose(&query, &config).unwrap();

        // Empty batch should return empty results
        let empty_targets: Vec<TTVector> = vec![];
        let sims = tt_cosine_similarity_batch(&query_tt, &empty_targets).unwrap();
        assert!(sims.is_empty());

        let dots = tt_dot_product_batch(&query_tt, &empty_targets).unwrap();
        assert!(dots.is_empty());

        let dists = tt_euclidean_distance_batch(&query_tt, &empty_targets).unwrap();
        assert!(dists.is_empty());
    }

    #[test]
    fn test_ttconfig_shape_contains_zero() {
        let config = TTConfig {
            shape: vec![4, 0, 4],
            max_rank: 8,
            tolerance: 0.01,
        };
        assert!(matches!(config.validate(), Err(TTError::InvalidShape(_))));
    }

    #[test]
    fn test_ttconfig_zero_rank() {
        let config = TTConfig {
            shape: vec![4, 4, 4],
            max_rank: 0,
            tolerance: 0.01,
        };
        assert!(matches!(config.validate(), Err(TTError::InvalidRank)));
    }

    #[test]
    fn test_optimal_shape_various_dimensions() {
        for dim in [384, 768, 1024, 1536, 2048, 3072, 8192] {
            let config = TTConfig::for_dim(dim).unwrap();
            let product: usize = config.shape.iter().product();
            assert_eq!(product, dim, "Shape product mismatch for dim {dim}");
        }
    }

    #[test]
    fn test_tt_incompatible_shapes_cosine() {
        let config64 = TTConfig::for_dim(64).unwrap();
        let config128 = TTConfig::for_dim(128).unwrap();

        let v64: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let v128: Vec<f32> = (0..128).map(|i| i as f32).collect();

        let tt64 = tt_decompose(&v64, &config64).unwrap();
        let tt128 = tt_decompose(&v128, &config128).unwrap();

        assert!(tt_cosine_similarity(&tt64, &tt128).is_err());
    }
}
