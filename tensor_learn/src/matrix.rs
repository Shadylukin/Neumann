// SPDX-License-Identifier: MIT OR Apache-2.0
//! 8x8 matrix for geometric transformations.
//!
//! Row-major storage backed by a flat `[f64; 64]` array.

use std::ops::Mul;

/// An 8x8 matrix stored in row-major order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix8x8 {
    /// Row-major flat storage.
    data: [f64; 64],
}

impl Matrix8x8 {
    /// Create the 8x8 identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        let mut data = [0.0; 64];
        for i in 0..8 {
            data[i * 8 + i] = 1.0;
        }
        Self { data }
    }

    /// Create a matrix from 8 rows of 8 elements each.
    #[must_use]
    pub fn from_rows(rows: [[f64; 8]; 8]) -> Self {
        let mut data = [0.0; 64];
        for (i, row) in rows.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                data[i * 8 + j] = val;
            }
        }
        Self { data }
    }

    /// Create a matrix from a flat 64-element vector.
    ///
    /// Returns `None` if the slice length is not 64.
    #[must_use]
    pub const fn from_flat(flat: &[f64]) -> Option<Self> {
        if flat.len() != 64 {
            return None;
        }
        let mut data = [0.0; 64];
        data.copy_from_slice(flat);
        Some(Self { data })
    }

    /// Get element at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if row or col >= 8.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < 8 && col < 8, "index out of bounds");
        self.data[row * 8 + col]
    }

    /// Set element at (row, col).
    ///
    /// # Panics
    ///
    /// Panics if row or col >= 8.
    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        assert!(row < 8 && col < 8, "index out of bounds");
        self.data[row * 8 + col] = val;
    }

    /// Compute the Frobenius norm: sqrt(sum of squared elements).
    #[must_use]
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Return the flat row-major representation.
    #[must_use]
    pub fn to_flat(&self) -> Vec<f64> {
        self.data.to_vec()
    }

    /// Transpose the matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut result = [0.0; 64];
        for i in 0..8 {
            for j in 0..8 {
                result[j * 8 + i] = self.data[i * 8 + j];
            }
        }
        Self { data: result }
    }

    /// Matrix multiplication (self * other).
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = [0.0; 64];
        for i in 0..8 {
            for j in 0..8 {
                let mut sum = 0.0;
                for k in 0..8 {
                    sum += self.data[i * 8 + k] * other.data[k * 8 + j];
                }
                result[i * 8 + j] = sum;
            }
        }
        Self { data: result }
    }
}

impl Default for Matrix8x8 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Mul for Matrix8x8 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul(&self, &rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = Matrix8x8::identity();
        for i in 0..8 {
            for j in 0..8 {
                if i == j {
                    assert_eq!(id.get(i, j), 1.0);
                } else {
                    assert_eq!(id.get(i, j), 0.0);
                }
            }
        }
    }

    #[test]
    fn test_identity_mul_identity() {
        let id = Matrix8x8::identity();
        let result = Matrix8x8::mul(&id, &id);
        assert_eq!(result, id);
    }

    #[test]
    fn test_from_rows() {
        let rows = [[1.0; 8]; 8];
        let m = Matrix8x8::from_rows(rows);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(7, 7), 1.0);
    }

    #[test]
    fn test_get_set() {
        let mut m = Matrix8x8::identity();
        m.set(2, 3, 42.0);
        assert_eq!(m.get(2, 3), 42.0);
    }

    #[test]
    fn test_frobenius_norm_identity() {
        let id = Matrix8x8::identity();
        let norm = id.frobenius_norm();
        assert!((norm - 8.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_transpose() {
        let mut m = Matrix8x8::identity();
        m.set(0, 1, 5.0);
        let t = m.transpose();
        assert_eq!(t.get(1, 0), 5.0);
        assert_eq!(t.get(0, 1), 0.0);
    }

    #[test]
    fn test_to_from_flat() {
        let m = Matrix8x8::identity();
        let flat = m.to_flat();
        assert_eq!(flat.len(), 64);
        let m2 = Matrix8x8::from_flat(&flat).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_from_flat_wrong_size() {
        assert!(Matrix8x8::from_flat(&[0.0; 63]).is_none());
        assert!(Matrix8x8::from_flat(&[0.0; 65]).is_none());
    }

    #[test]
    fn test_mul_operator() {
        let id = Matrix8x8::identity();
        let result = id * id;
        assert_eq!(result, Matrix8x8::identity());
    }

    #[test]
    fn test_default_is_identity() {
        assert_eq!(Matrix8x8::default(), Matrix8x8::identity());
    }

    #[test]
    fn test_frobenius_norm_zero() {
        let m = Matrix8x8::from_flat(&[0.0; 64]).unwrap();
        assert_eq!(m.frobenius_norm(), 0.0);
    }

    #[test]
    fn test_mul_non_trivial() {
        let mut a = Matrix8x8::identity();
        a.set(0, 1, 2.0);
        let mut b = Matrix8x8::identity();
        b.set(1, 0, 3.0);
        let c = Matrix8x8::mul(&a, &b);
        // (0,0): 1*1 + 2*3 = 7
        assert_eq!(c.get(0, 0), 7.0);
        // (0,1): 1*0 + 2*1 = 2
        assert_eq!(c.get(0, 1), 2.0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_get_out_of_bounds() {
        let m = Matrix8x8::identity();
        let _ = m.get(8, 0);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_set_out_of_bounds() {
        let mut m = Matrix8x8::identity();
        m.set(0, 8, 1.0);
    }

    #[test]
    fn test_transpose_symmetric() {
        let id = Matrix8x8::identity();
        assert_eq!(id.transpose(), id);
    }
}
