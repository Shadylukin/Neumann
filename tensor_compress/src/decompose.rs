// SPDX-License-Identifier: MIT OR Apache-2.0
//! Low-level matrix decomposition primitives for tensor operations.
//!
//! Implements pure-Rust SVD and matrix operations required for Tensor Train decomposition.
//! No external LAPACK or BLAS dependencies.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_panics_doc)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors from decomposition operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DecomposeError {
    #[error("empty matrix")]
    EmptyMatrix,
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("SVD failed to converge after {iterations} iterations")]
    SvdNotConverged { iterations: usize },
    #[error("invalid shape: product {product} does not match length {length}")]
    InvalidShape { product: usize, length: usize },
}

/// A matrix stored in row-major order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Result<Self, DecomposeError> {
        if data.len() != rows * cols {
            return Err(DecomposeError::InvalidShape {
                product: rows * cols,
                length: data.len(),
            });
        }
        Ok(Self { data, rows, cols })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        self.data[row * self.cols + col] = value;
    }

    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    pub fn frobenius_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

/// Result of truncated SVD: A ≈ U * diag(S) * Vt
#[derive(Debug, Clone)]
pub struct SvdResult {
    pub u: Matrix,
    pub s: Vec<f32>,
    pub vt: Matrix,
    pub rank: usize,
}

/// Logical view of a tensor (no data copy, just shape information).
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    pub data: &'a [f32],
    pub shape: Vec<usize>,
}

impl<'a> TensorView<'a> {
    pub fn new(data: &'a [f32], shape: Vec<usize>) -> Result<Self, DecomposeError> {
        let product: usize = shape.iter().product();
        if product != data.len() {
            return Err(DecomposeError::InvalidShape {
                product,
                length: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    pub const fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Reshape a 1D vector into a tensor view with the given shape.
pub fn reshape_to_tensor<'a>(
    vector: &'a [f32],
    shape: &[usize],
) -> Result<TensorView<'a>, DecomposeError> {
    TensorView::new(vector, shape.to_vec())
}

/// Compute strides for a tensor shape (row-major order).
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Unfold (matricize) a tensor along a given mode.
///
/// Mode-k unfolding arranges the tensor as a matrix where:
/// - Rows correspond to index `i_k`
/// - Columns correspond to all other indices in order
pub fn unfold(tensor: &TensorView<'_>, mode: usize) -> Result<Matrix, DecomposeError> {
    if mode >= tensor.ndim() {
        return Err(DecomposeError::DimensionMismatch {
            expected: tensor.ndim(),
            got: mode + 1,
        });
    }

    let rows = tensor.shape[mode];
    let cols: usize = tensor
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != mode)
        .map(|(_, &s)| s)
        .product();

    let mut result = Matrix::zeros(rows, cols);
    let strides = compute_strides(&tensor.shape);

    // Iterate through all tensor indices
    let total_elements: usize = tensor.shape.iter().product();
    for flat_idx in 0..total_elements {
        // Convert flat index to multi-index
        let mut remaining = flat_idx;
        let mut multi_idx = vec![0usize; tensor.ndim()];
        for (i, &stride) in strides.iter().enumerate() {
            multi_idx[i] = remaining / stride;
            remaining %= stride;
        }

        // Compute row (mode-k index) and column (all other indices)
        let row = multi_idx[mode];
        let mut col = 0;
        let mut col_stride = 1;
        for i in (0..tensor.ndim()).rev() {
            if i != mode {
                col += multi_idx[i] * col_stride;
                col_stride *= tensor.shape[i];
            }
        }

        result.set(row, col, tensor.data[flat_idx]);
    }

    Ok(result)
}

/// Matrix multiplication: C = A * B
pub fn matmul(a: &Matrix, b: &Matrix) -> Result<Matrix, DecomposeError> {
    if a.cols != b.rows {
        return Err(DecomposeError::DimensionMismatch {
            expected: a.cols,
            got: b.rows,
        });
    }

    let mut c = Matrix::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut sum = 0.0;
            for k in 0..a.cols {
                sum += a.get(i, k) * b.get(k, j);
            }
            c.set(i, j, sum);
        }
    }
    Ok(c)
}

/// Compute the dot product of two vectors.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the L2 norm of a vector.
#[inline]
fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Normalize a vector in place, returns the original norm.
fn normalize(v: &mut [f32]) -> f32 {
    let n = norm(v);
    if n > 1e-10 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    n
}

/// Simple LCG random number generator for deterministic pseudo-random sequences.
struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    #[allow(clippy::missing_const_for_fn)]
    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Generate deterministic pseudo-random Gaussian matrix using Box-Muller transform.
fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Matrix {
    let mut rng = Lcg::new(seed);
    let mut data = Vec::with_capacity(rows * cols);

    // Box-Muller transform: pairs of uniform -> pairs of Gaussian
    let total = rows * cols;
    let pairs = total.div_ceil(2);

    for _ in 0..pairs {
        let u1 = rng.next_f32().max(1e-10); // Avoid log(0)
        let u2 = rng.next_f32();

        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;

        data.push(r * theta.cos());
        if data.len() < total {
            data.push(r * theta.sin());
        }
    }

    data.truncate(total);
    Matrix::new(data, rows, cols).expect("valid gaussian matrix shape")
}

/// QR factorization returning only Q (orthonormal basis for column space).
/// Uses Modified Gram-Schmidt for numerical stability.
fn qr_orthonormalize(y: &Matrix) -> Matrix {
    if y.rows == 0 || y.cols == 0 {
        return Matrix::zeros(y.rows, 0);
    }

    let mut q_cols: Vec<Vec<f32>> = Vec::with_capacity(y.cols);

    for j in 0..y.cols {
        // Extract column j
        let mut col: Vec<f32> = (0..y.rows).map(|i| y.get(i, j)).collect();

        // Subtract projections onto previous Q columns (Modified Gram-Schmidt)
        for q_col in &q_cols {
            let proj = dot(&col, q_col);
            for (i, val) in col.iter_mut().enumerate() {
                *val -= proj * q_col[i];
            }
        }

        // Normalize
        let col_norm = normalize(&mut col);

        // Only keep if not linearly dependent (norm > epsilon)
        if col_norm > 1e-10 {
            q_cols.push(col);
        }
    }

    // Build Q matrix (rows x actual_rank)
    let actual_rank = q_cols.len();
    if actual_rank == 0 {
        return Matrix::zeros(y.rows, 0);
    }

    let mut q_data = vec![0.0; y.rows * actual_rank];
    for (j, col) in q_cols.iter().enumerate() {
        for (i, &val) in col.iter().enumerate() {
            q_data[i * actual_rank + j] = val;
        }
    }

    Matrix::new(q_data, y.rows, actual_rank).expect("valid Q shape")
}

/// Power iteration to find the largest singular value and vectors.
/// Returns (sigma, u, v) where A*v ≈ sigma*u and A^T*u ≈ sigma*v.
fn power_iteration(
    a: &Matrix,
    max_iter: usize,
    tol: f32,
) -> Result<(f32, Vec<f32>, Vec<f32>), DecomposeError> {
    if a.rows == 0 || a.cols == 0 {
        return Err(DecomposeError::EmptyMatrix);
    }

    // Initialize v randomly (deterministic seed for reproducibility)
    let mut v: Vec<f32> = (0..a.cols)
        .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5)
        .collect();
    normalize(&mut v);

    let mut u = vec![0.0f32; a.rows];
    let mut sigma = 0.0f32;

    for _ in 0..max_iter {
        // u = A * v
        for i in 0..a.rows {
            u[i] = 0.0;
            for j in 0..a.cols {
                u[i] += a.get(i, j) * v[j];
            }
        }
        let new_sigma = normalize(&mut u);

        // v = A^T * u
        for j in 0..a.cols {
            v[j] = 0.0;
            for i in 0..a.rows {
                v[j] += a.get(i, j) * u[i];
            }
        }
        normalize(&mut v);

        // Check convergence
        if (new_sigma - sigma).abs() < tol * sigma.max(1.0) {
            return Ok((new_sigma, u, v));
        }
        sigma = new_sigma;
    }

    Ok((sigma, u, v))
}

/// Truncated SVD using Randomized algorithm (Halko-Martinsson-Tropp 2011).
///
/// For matrices where min(m,n) > 2*k, uses random projection to reduce
/// the problem size before computing SVD. Falls back to direct power
/// iteration for small matrices.
#[allow(clippy::many_single_char_names)]
pub fn svd_truncated(
    matrix: &Matrix,
    max_rank: usize,
    tolerance: f32,
) -> Result<SvdResult, DecomposeError> {
    if matrix.data.is_empty() {
        return Err(DecomposeError::EmptyMatrix);
    }

    let (rows, cols) = (matrix.rows, matrix.cols);
    let rank = max_rank.min(rows).min(cols);

    // For small matrices or when rank is close to matrix size, use direct method
    // Randomized SVD overhead isn't worth it for small problems
    if rows <= 32 || cols <= 32 || rank * 2 >= rows.min(cols) {
        return svd_power_iteration(matrix, max_rank, tolerance);
    }

    // Oversampling parameter (typically 5-10)
    let oversample = 5.min(rows.min(cols).saturating_sub(rank));
    let target_rank = rank + oversample;

    // Step 1: Generate random Gaussian test matrix Omega (cols x target_rank)
    // Use matrix dimensions as seed for reproducibility
    let seed = (rows as u64).wrapping_mul(31).wrapping_add(cols as u64);
    let omega = gaussian_matrix(cols, target_rank, seed);

    // Step 2: Sample the range: Y = A * Omega (rows x target_rank)
    let sample = matmul(matrix, &omega)?;

    // Step 3: Orthonormalize Y to get Q (rows x r where r <= target_rank)
    let basis = qr_orthonormalize(&sample);

    if basis.cols == 0 {
        // Matrix is effectively zero
        return Ok(SvdResult {
            u: Matrix::zeros(rows, 0),
            s: vec![],
            vt: Matrix::zeros(0, cols),
            rank: 0,
        });
    }

    // Step 4: Project to small space: B = Q^T * A (r x cols)
    let basis_t = basis.transpose();
    let projected = matmul(&basis_t, matrix)?;

    // Step 5: SVD of small matrix B
    let svd_small = svd_power_iteration(&projected, rank, tolerance)?;

    if svd_small.rank == 0 {
        return Ok(SvdResult {
            u: Matrix::zeros(rows, 0),
            s: vec![],
            vt: Matrix::zeros(0, cols),
            rank: 0,
        });
    }

    // Step 6: Reconstruct U = Q * U_small
    let u_final = matmul(&basis, &svd_small.u)?;

    Ok(SvdResult {
        u: u_final,
        s: svd_small.s,
        vt: svd_small.vt,
        rank: svd_small.rank,
    })
}

/// SVD using power iteration with deflation.
/// Used for small matrices where randomized overhead isn't beneficial.
fn svd_power_iteration(
    matrix: &Matrix,
    max_rank: usize,
    tolerance: f32,
) -> Result<SvdResult, DecomposeError> {
    // Reduced iterations (20 instead of 100) - sufficient for embedding vectors
    const MAX_POWER_ITERATIONS: usize = 20;

    if matrix.data.is_empty() {
        return Err(DecomposeError::EmptyMatrix);
    }

    let rank = max_rank.min(matrix.rows).min(matrix.cols);
    let mut u_cols = Vec::with_capacity(rank);
    let mut s_vals = Vec::with_capacity(rank);
    let mut v_cols = Vec::with_capacity(rank);

    // Work with a copy for deflation
    let mut a_data = matrix.data.clone();
    let mut a = Matrix::new(a_data.clone(), matrix.rows, matrix.cols)?;

    let original_norm = matrix.frobenius_norm();
    let abs_tol = tolerance * original_norm;

    for _ in 0..rank {
        let (sigma, u, v) = power_iteration(&a, MAX_POWER_ITERATIONS, 1e-6)?;

        // Stop if singular value is below tolerance
        if sigma < abs_tol {
            break;
        }

        s_vals.push(sigma);
        u_cols.push(u.clone());
        v_cols.push(v.clone());

        // Deflate: A = A - sigma * u * v^T
        for i in 0..a.rows {
            for j in 0..a.cols {
                let idx = i * a.cols + j;
                a_data[idx] -= sigma * u[i] * v[j];
            }
        }
        a = Matrix::new(a_data.clone(), matrix.rows, matrix.cols)?;
    }

    let actual_rank = s_vals.len();
    if actual_rank == 0 {
        return Ok(SvdResult {
            u: Matrix::zeros(matrix.rows, 0),
            s: vec![],
            vt: Matrix::zeros(0, matrix.cols),
            rank: 0,
        });
    }

    // Build U matrix (rows x rank)
    let mut u_data = vec![0.0; matrix.rows * actual_rank];
    for (j, col) in u_cols.iter().enumerate() {
        for (i, &val) in col.iter().enumerate() {
            u_data[i * actual_rank + j] = val;
        }
    }
    let u_mat = Matrix::new(u_data, matrix.rows, actual_rank)?;

    // Build Vt matrix (rank x cols)
    let mut vt_data = vec![0.0; actual_rank * matrix.cols];
    for (i, col) in v_cols.iter().enumerate() {
        for (j, &val) in col.iter().enumerate() {
            vt_data[i * matrix.cols + j] = val;
        }
    }
    let vt_mat = Matrix::new(vt_data, actual_rank, matrix.cols)?;

    Ok(SvdResult {
        u: u_mat,
        s: s_vals,
        vt: vt_mat,
        rank: actual_rank,
    })
}

/// Left-unfold for TT-SVD: reshape tensor core for left-to-right sweep.
/// Input shape: (r_{k-1}, n_k, remaining...)
/// Output: matrix of shape (r_{k-1} * n_k, remaining_product)
pub fn left_unfold_for_tt(data: &[f32], left_size: usize, mode_size: usize) -> Matrix {
    let rows = left_size * mode_size;
    if rows == 0 || data.is_empty() {
        return Matrix::new(vec![], 0, 0).expect("valid empty shape");
    }
    let cols = data.len() / rows;
    Matrix::new(data.to_vec(), rows, cols).expect("valid shape")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_new() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 2.0);
        assert_eq!(m.get(1, 0), 3.0);
        assert_eq!(m.get(1, 1), 4.0);
    }

    #[test]
    fn test_matrix_new_invalid_shape() {
        let result = Matrix::new(vec![1.0, 2.0, 3.0], 2, 2);
        assert!(matches!(result, Err(DecomposeError::InvalidShape { .. })));
    }

    #[test]
    fn test_matrix_transpose() {
        let m = Matrix::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(0, 1), 4.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_matmul() {
        let a = Matrix::new(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let b = Matrix::new(vec![5.0, 6.0, 7.0, 8.0], 2, 2).unwrap();
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.get(0, 0), 19.0); // 1*5 + 2*7
        assert_eq!(c.get(0, 1), 22.0); // 1*6 + 2*8
        assert_eq!(c.get(1, 0), 43.0); // 3*5 + 4*7
        assert_eq!(c.get(1, 1), 50.0); // 3*6 + 4*8
    }

    #[test]
    fn test_tensor_view_new() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();
        assert_eq!(view.ndim(), 2);
        assert_eq!(view.shape, vec![2, 3]);
    }

    #[test]
    fn test_tensor_view_invalid_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = TensorView::new(&data, vec![2, 3]);
        assert!(matches!(result, Err(DecomposeError::InvalidShape { .. })));
    }

    #[test]
    fn test_unfold_mode_0() {
        // 2x3 matrix unfolded along mode 0 should give 2x3 matrix (unchanged)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = TensorView::new(&data, vec![2, 3]).unwrap();
        let unfolded = unfold(&tensor, 0).unwrap();
        assert_eq!(unfolded.rows, 2);
        assert_eq!(unfolded.cols, 3);
    }

    #[test]
    fn test_unfold_mode_1() {
        // 2x3 matrix unfolded along mode 1 should give 3x2 matrix
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = TensorView::new(&data, vec![2, 3]).unwrap();
        let unfolded = unfold(&tensor, 1).unwrap();
        assert_eq!(unfolded.rows, 3);
        assert_eq!(unfolded.cols, 2);
    }

    #[test]
    fn test_svd_truncated_simple() {
        // Simple rank-1 matrix: outer product of [1,2] and [3,4]
        let data = vec![3.0, 4.0, 6.0, 8.0];
        let m = Matrix::new(data, 2, 2).unwrap();
        let svd = svd_truncated(&m, 2, 1e-6).unwrap();

        // Should have rank 1
        assert_eq!(svd.rank, 1);
        assert!(svd.s[0] > 0.0);

        // Reconstruct and verify
        let mut reconstructed = Matrix::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..svd.rank {
                    sum += svd.u.get(i, k) * svd.s[k] * svd.vt.get(k, j);
                }
                reconstructed.set(i, j, sum);
            }
        }

        // Check reconstruction error
        for i in 0..2 {
            for j in 0..2 {
                let expected = m.get(i, j);
                let actual = reconstructed.get(i, j);
                assert!(
                    (expected - actual).abs() < 0.1,
                    "Reconstruction error at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_svd_truncated_rank_2() {
        // Diagonal matrix with distinct singular values
        let m = Matrix::new(vec![2.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let svd = svd_truncated(&m, 2, 1e-6).unwrap();
        assert_eq!(svd.rank, 2);
        assert!((svd.s[0] - 2.0).abs() < 0.2);
        assert!((svd.s[1] - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_svd_empty_matrix() {
        let m = Matrix::zeros(0, 0);
        let result = svd_truncated(&m, 1, 1e-6);
        assert!(matches!(result, Err(DecomposeError::EmptyMatrix)));
    }

    #[test]
    fn test_frobenius_norm() {
        let m = Matrix::new(vec![3.0, 4.0], 1, 2).unwrap();
        let norm = m.frobenius_norm();
        assert!((norm - 5.0).abs() < 1e-6); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_reshape_to_tensor() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let t = reshape_to_tensor(&v, &[2, 2, 2]).unwrap();
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.shape, vec![2, 2, 2]);
    }

    #[test]
    fn test_left_unfold_for_tt() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let m = left_unfold_for_tt(&data, 2, 2); // (2*2, 2) = (4, 2)
        assert_eq!(m.rows, 4);
        assert_eq!(m.cols, 2);
    }

    // === Randomized SVD primitive tests ===

    #[test]
    fn test_gaussian_matrix_shape() {
        let g = gaussian_matrix(10, 5, 42);
        assert_eq!(g.rows, 10);
        assert_eq!(g.cols, 5);
        assert_eq!(g.data.len(), 50);
    }

    #[test]
    fn test_gaussian_matrix_deterministic() {
        let g1 = gaussian_matrix(8, 4, 123);
        let g2 = gaussian_matrix(8, 4, 123);
        assert_eq!(g1.data, g2.data, "Same seed should produce same output");

        let g3 = gaussian_matrix(8, 4, 456);
        assert_ne!(g1.data, g3.data, "Different seeds should differ");
    }

    #[test]
    fn test_gaussian_matrix_distribution() {
        // Check that values are roughly Gaussian (mean ~0, stddev ~1)
        let g = gaussian_matrix(100, 100, 42);
        let mean: f32 = g.data.iter().sum::<f32>() / g.data.len() as f32;
        let variance: f32 =
            g.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / g.data.len() as f32;
        let stddev = variance.sqrt();

        // Mean should be close to 0, stddev close to 1
        assert!(mean.abs() < 0.2, "Mean {} too far from 0", mean);
        assert!(
            (stddev - 1.0).abs() < 0.3,
            "Stddev {} too far from 1",
            stddev
        );
    }

    #[test]
    fn test_gaussian_matrix_no_nan_inf() {
        let g = gaussian_matrix(50, 50, 99);
        assert!(
            g.data.iter().all(|x| x.is_finite()),
            "Should have no NaN/Inf"
        );
    }

    #[test]
    fn test_qr_orthonormal() {
        // Create a full-rank matrix and verify Q^T * Q ≈ I
        // Using a matrix with linearly independent columns
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let m = Matrix::new(data, 4, 3).unwrap();
        let q = qr_orthonormalize(&m);

        assert_eq!(q.cols, 3, "Should preserve rank");

        // Q should have orthonormal columns
        for i in 0..q.cols {
            for j in 0..q.cols {
                let col_i: Vec<f32> = (0..q.rows).map(|r| q.get(r, i)).collect();
                let col_j: Vec<f32> = (0..q.rows).map(|r| q.get(r, j)).collect();
                let dot_ij = dot(&col_i, &col_j);

                if i == j {
                    assert!(
                        (dot_ij - 1.0).abs() < 1e-5,
                        "Column {} not unit norm: {}",
                        i,
                        dot_ij
                    );
                } else {
                    assert!(
                        dot_ij.abs() < 1e-5,
                        "Columns {} and {} not orthogonal: {}",
                        i,
                        j,
                        dot_ij
                    );
                }
            }
        }
    }

    #[test]
    fn test_qr_empty() {
        let m = Matrix::zeros(5, 0);
        let q = qr_orthonormalize(&m);
        assert_eq!(q.cols, 0);
    }

    #[test]
    fn test_qr_rank_deficient() {
        // Matrix with identical columns (clearly rank 1)
        let data = vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        let m = Matrix::new(data, 3, 3).unwrap();
        let q = qr_orthonormalize(&m);

        // Should have fewer columns than input (rank < 3)
        assert!(
            q.cols < 3,
            "Rank-deficient matrix should have fewer Q columns, got {}",
            q.cols
        );
    }

    #[test]
    fn test_svd_randomized_large_matrix() {
        // Create a larger matrix that triggers randomized path (>32 dimensions)
        let rows = 64;
        let cols = 48;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 + 3) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let m = Matrix::new(data.clone(), rows, cols).unwrap();

        let svd = svd_truncated(&m, 8, 1e-4).unwrap();

        // Verify shapes
        assert_eq!(svd.u.rows, rows);
        assert!(svd.u.cols <= 8);
        assert!(svd.vt.rows <= 8);
        assert_eq!(svd.vt.cols, cols);
        assert!(svd.rank <= 8);

        // Verify singular values are positive and ordered
        for i in 1..svd.s.len() {
            assert!(
                svd.s[i - 1] >= svd.s[i] - 1e-6,
                "Singular values not ordered: {} < {}",
                svd.s[i - 1],
                svd.s[i]
            );
        }

        // Verify reconstruction is reasonable
        let mut reconstructed = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..svd.rank {
                    sum += svd.u.get(i, k) * svd.s[k] * svd.vt.get(k, j);
                }
                reconstructed.set(i, j, sum);
            }
        }

        // Reconstruction error should be small relative to original norm
        let orig_norm = m.frobenius_norm();
        let mut error_sq = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let diff = m.get(i, j) - reconstructed.get(i, j);
                error_sq += diff * diff;
            }
        }
        let rel_error = error_sq.sqrt() / orig_norm;
        assert!(
            rel_error < 0.5,
            "Reconstruction error too high: {}",
            rel_error
        );
    }

    #[test]
    fn test_svd_power_iteration_direct() {
        // Test the internal power iteration function directly
        let data = vec![3.0, 4.0, 6.0, 8.0];
        let m = Matrix::new(data, 2, 2).unwrap();
        let svd = svd_power_iteration(&m, 2, 1e-6).unwrap();

        assert_eq!(svd.rank, 1);
        assert!(svd.s[0] > 0.0);
    }

    #[test]
    fn test_lcg_determinism() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_lcg_f32_range() {
        let mut rng = Lcg::new(12345);
        for _ in 0..1000 {
            let f = rng.next_f32();
            assert!(f >= 0.0 && f < 1.0, "f32 {} out of range [0, 1)", f);
        }
    }
}
