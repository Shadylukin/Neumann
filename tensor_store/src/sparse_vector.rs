// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Sparse Vector - Storage where zero doesn't exist
//!
//! Philosophy: Zero represents absence of information, not a value to store.
//! Only non-zero values are fundamental. The dimension defines the boundary
//! (shell) of meaningful space.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Maximum dimension for sparse vectors (limited by u32 position storage).
pub const MAX_DIMENSION: usize = u32::MAX as usize;

/// Error type for sparse vector operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseVectorError {
    /// Dimension exceeds the maximum supported value.
    DimensionExceeded {
        /// The requested dimension.
        dimension: usize,
        /// The maximum allowed dimension.
        max: usize,
    },
    /// Index is out of bounds for the vector dimension.
    IndexOutOfBounds {
        /// The requested index.
        index: usize,
        /// The vector dimension.
        dimension: usize,
    },
}

impl fmt::Display for SparseVectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionExceeded { dimension, max } => {
                write!(f, "dimension {dimension} exceeds maximum {max} (u32::MAX)")
            },
            Self::IndexOutOfBounds { index, dimension } => {
                write!(f, "index {index} out of bounds for dimension {dimension}")
            },
        }
    }
}

impl std::error::Error for SparseVectorError {}

/// A vector that only stores non-zero values.
///
/// The `dimension` defines the shell/boundary - the total space where
/// information could exist. Positions not stored are implicitly zero
/// (absence of information within the boundary).
///
/// # Example
///
/// ```
/// use tensor_store::sparse_vector::SparseVector;
///
/// // Create from dense - zeros are not stored
/// let dense = vec![0.0, 1.5, 0.0, 0.0, 2.3, 0.0];
/// let sparse = SparseVector::from_dense(&dense);
///
/// assert_eq!(sparse.dimension(), 6); // Shell size
/// assert_eq!(sparse.nnz(), 2); // Only 2 values stored
/// assert_eq!(sparse.get(1), 1.5); // Stored value
/// assert_eq!(sparse.get(0), 0.0); // Contextual zero (in shell, not stored)
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    /// Total dimension - the boundary/shell of the vector space
    dimension: usize,
    /// Positions of non-zero values (sorted, unique)
    positions: Vec<u32>,
    /// Non-zero values (parallel to positions)
    values: Vec<f32>,
}

impl SparseVector {
    /// Create an empty sparse vector with given dimension.
    ///
    /// The dimension defines the shell - all positions within [0, dimension)
    /// are valid, but none contain information yet.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` exceeds `MAX_DIMENSION` (`u32::MAX`).
    /// Use [`try_new`](Self::try_new) for a fallible version.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self::try_new(dimension).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create an empty sparse vector with given dimension.
    ///
    /// Returns an error if `dimension` exceeds `MAX_DIMENSION`.
    ///
    /// # Errors
    ///
    /// Returns [`SparseVectorError::DimensionExceeded`] if dimension > `u32::MAX`.
    pub const fn try_new(dimension: usize) -> Result<Self, SparseVectorError> {
        if dimension > MAX_DIMENSION {
            return Err(SparseVectorError::DimensionExceeded {
                dimension,
                max: MAX_DIMENSION,
            });
        }
        Ok(Self {
            dimension,
            positions: Vec::new(),
            values: Vec::new(),
        })
    }

    /// Create a sparse vector with pre-allocated capacity.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` exceeds `MAX_DIMENSION`.
    #[must_use]
    pub fn with_capacity(dimension: usize, capacity: usize) -> Self {
        assert!(
            dimension <= MAX_DIMENSION,
            "dimension {dimension} exceeds maximum {MAX_DIMENSION}"
        );
        Self {
            dimension,
            positions: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    /// Create from parallel arrays of positions and values.
    ///
    /// Filters out zeros and sorts by position.
    ///
    /// # Panics
    ///
    /// Panics if `dimension` exceeds `MAX_DIMENSION` or any position >= dimension.
    /// Use [`try_from_parts`](Self::try_from_parts) for a fallible version.
    #[must_use]
    pub fn from_parts(dimension: usize, positions: Vec<u32>, values: Vec<f32>) -> Self {
        Self::try_from_parts(dimension, positions, values).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create from parallel arrays of positions and values.
    ///
    /// Filters out zeros and sorts by position.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `dimension` exceeds `MAX_DIMENSION`
    /// - Any position is >= `dimension`
    pub fn try_from_parts(
        dimension: usize,
        positions: Vec<u32>,
        values: Vec<f32>,
    ) -> Result<Self, SparseVectorError> {
        if dimension > MAX_DIMENSION {
            return Err(SparseVectorError::DimensionExceeded {
                dimension,
                max: MAX_DIMENSION,
            });
        }
        debug_assert_eq!(positions.len(), values.len());

        // Validate positions and filter zeros
        let mut pairs: Vec<(u32, f32)> = Vec::with_capacity(positions.len());
        for (pos, val) in positions.into_iter().zip(values) {
            if pos as usize >= dimension {
                return Err(SparseVectorError::IndexOutOfBounds {
                    index: pos as usize,
                    dimension,
                });
            }
            if val != 0.0 {
                pairs.push((pos, val));
            }
        }

        // Sort by position
        pairs.sort_by_key(|(p, _)| *p);

        // Unzip back
        let (positions, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();

        Ok(Self {
            dimension,
            positions,
            values,
        })
    }

    /// Create from a dense vector, storing only non-zero values.
    ///
    /// # Panics
    ///
    /// Panics if `dense.len()` exceeds `MAX_DIMENSION`.
    /// Use [`try_from_dense`](Self::try_from_dense) for a fallible version.
    #[must_use]
    pub fn from_dense(dense: &[f32]) -> Self {
        Self::try_from_dense(dense).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create from a dense vector, storing only non-zero values.
    ///
    /// # Errors
    ///
    /// Returns [`SparseVectorError::DimensionExceeded`] if `dense.len()` > `MAX_DIMENSION`.
    #[allow(clippy::cast_possible_truncation)] // Position indices stored as u32; validated above
    pub fn try_from_dense(dense: &[f32]) -> Result<Self, SparseVectorError> {
        let dimension = dense.len();
        if dimension > MAX_DIMENSION {
            return Err(SparseVectorError::DimensionExceeded {
                dimension,
                max: MAX_DIMENSION,
            });
        }

        let mut positions = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in dense.iter().enumerate() {
            if val != 0.0 {
                positions.push(i as u32);
                values.push(val);
            }
        }

        Ok(Self {
            dimension,
            positions,
            values,
        })
    }

    /// Create from a dense vector with a threshold - values below threshold become zero.
    ///
    /// # Panics
    ///
    /// Panics if `dense.len()` exceeds `MAX_DIMENSION`.
    /// Use [`try_from_dense_with_threshold`](Self::try_from_dense_with_threshold) for a fallible version.
    #[must_use]
    pub fn from_dense_with_threshold(dense: &[f32], threshold: f32) -> Self {
        Self::try_from_dense_with_threshold(dense, threshold).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create from a dense vector with a threshold - values below threshold become zero.
    ///
    /// # Errors
    ///
    /// Returns [`SparseVectorError::DimensionExceeded`] if `dense.len()` > `MAX_DIMENSION`.
    #[allow(clippy::cast_possible_truncation)] // Position indices stored as u32; validated above
    pub fn try_from_dense_with_threshold(
        dense: &[f32],
        threshold: f32,
    ) -> Result<Self, SparseVectorError> {
        let dimension = dense.len();
        if dimension > MAX_DIMENSION {
            return Err(SparseVectorError::DimensionExceeded {
                dimension,
                max: MAX_DIMENSION,
            });
        }

        let mut positions = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in dense.iter().enumerate() {
            if val.abs() >= threshold {
                positions.push(i as u32);
                values.push(val);
            }
        }

        Ok(Self {
            dimension,
            positions,
            values,
        })
    }

    /// The shell/boundary - total dimension of the vector space.
    #[inline]
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of non-zero values stored.
    #[inline]
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio (fraction of zeros).
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for ratio calculation
    pub fn sparsity(&self) -> f32 {
        if self.dimension == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f32 / self.dimension as f32)
        }
    }

    /// Returns true if the index is within the vector dimension.
    #[inline]
    #[must_use]
    pub const fn in_bounds(&self, index: usize) -> bool {
        index < self.dimension
    }

    /// Get value at position.
    ///
    /// Returns 0.0 for positions within shell that have no stored value.
    /// This is a "contextual zero" - absence of information, not stored zero.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Position indices stored as u32
    pub fn get(&self, index: usize) -> f32 {
        debug_assert!(index < self.dimension, "Index out of bounds");

        // Binary search for position - returns 0.0 (contextual zero) if not found
        self.positions
            .binary_search(&(index as u32))
            .ok()
            .map_or(0.0, |i| self.values[i])
    }

    /// Returns true if the position has an explicit non-zero value.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Position indices stored as u32
    pub fn has_value(&self, index: usize) -> bool {
        self.positions.binary_search(&(index as u32)).is_ok()
    }

    /// Set value at position.
    ///
    /// If value is zero, removes from storage (zero doesn't exist).
    /// If value is non-zero, inserts or updates.
    ///
    /// # Panics
    ///
    /// Panics if `index` >= `dimension`.
    /// Use [`try_set`](Self::try_set) for a fallible version.
    pub fn set(&mut self, index: usize, value: f32) {
        self.try_set(index, value).unwrap_or_else(|e| panic!("{e}"));
    }

    /// Set value at position.
    ///
    /// If value is zero, removes from storage (zero doesn't exist).
    /// If value is non-zero, inserts or updates.
    ///
    /// # Errors
    ///
    /// Returns [`SparseVectorError::IndexOutOfBounds`] if `index` >= `dimension`.
    #[allow(clippy::cast_possible_truncation)] // Position indices stored as u32; validated above
    pub fn try_set(&mut self, index: usize, value: f32) -> Result<(), SparseVectorError> {
        if index >= self.dimension {
            return Err(SparseVectorError::IndexOutOfBounds {
                index,
                dimension: self.dimension,
            });
        }

        let idx = index as u32;
        match self.positions.binary_search(&idx) {
            Ok(i) => {
                if value == 0.0 {
                    // Remove - zero doesn't exist
                    self.positions.remove(i);
                    self.values.remove(i);
                } else {
                    // Update
                    self.values[i] = value;
                }
            },
            Err(i) => {
                if value != 0.0 {
                    // Insert new non-zero
                    self.positions.insert(i, idx);
                    self.values.insert(i, value);
                }
                // If value is 0.0 and not found, do nothing (already absent)
            },
        }
        Ok(())
    }

    /// Convert to dense representation.
    ///
    /// This "realizes" all the contextual zeros as actual values.
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.dimension];
        for (&pos, &val) in self.positions.iter().zip(&self.values) {
            dense[pos as usize] = val;
        }
        dense
    }

    /// Dot product with another sparse vector.
    ///
    /// `O(min(nnz_a, nnz_b))` - only overlapping non-zero positions contribute.
    /// This is where sparse shines: zero * anything = zero, and we don't store zeros.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn dot(&self, other: &Self) -> f32 {
        self.dot_f64(other) as f32
    }

    /// Internal f64 dot product to avoid intermediate overflow.
    fn dot_f64(&self, other: &Self) -> f64 {
        debug_assert_eq!(
            self.dimension, other.dimension,
            "Dimension mismatch: {} vs {}",
            self.dimension, other.dimension
        );

        let mut result = 0.0_f64;
        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() && j < other.positions.len() {
            match self.positions[i].cmp(&other.positions[j]) {
                std::cmp::Ordering::Equal => {
                    result += f64::from(self.values[i]) * f64::from(other.values[j]);
                    i += 1;
                    j += 1;
                },
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }

        result
    }

    /// Dot product with a dense vector.
    ///
    /// `O(nnz)` - only iterate over our non-zero positions.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn dot_dense(&self, dense: &[f32]) -> f32 {
        debug_assert_eq!(
            self.dimension,
            dense.len(),
            "Dimension mismatch: {} vs {}",
            self.dimension,
            dense.len()
        );

        let sum: f64 = self
            .positions
            .iter()
            .zip(&self.values)
            .map(|(&pos, &val)| f64::from(val) * f64::from(dense[pos as usize]))
            .sum();
        sum as f32
    }

    /// Add another sparse vector. Returns new sparse vector.
    ///
    /// Positions with zero sum are not stored in result.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.dimension, other.dimension);

        let mut result_positions = Vec::new();
        let mut result_values = Vec::new();

        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() || j < other.positions.len() {
            let (pos, val) = if i >= self.positions.len() {
                let p = other.positions[j];
                let v = other.values[j];
                j += 1;
                (p, v)
            } else if j >= other.positions.len() {
                let p = self.positions[i];
                let v = self.values[i];
                i += 1;
                (p, v)
            } else {
                match self.positions[i].cmp(&other.positions[j]) {
                    std::cmp::Ordering::Equal => {
                        let p = self.positions[i];
                        let v = self.values[i] + other.values[j];
                        i += 1;
                        j += 1;
                        (p, v)
                    },
                    std::cmp::Ordering::Less => {
                        let p = self.positions[i];
                        let v = self.values[i];
                        i += 1;
                        (p, v)
                    },
                    std::cmp::Ordering::Greater => {
                        let p = other.positions[j];
                        let v = other.values[j];
                        j += 1;
                        (p, v)
                    },
                }
            };

            // Only store if non-zero (zero doesn't exist!)
            if val != 0.0 {
                result_positions.push(pos);
                result_values.push(val);
            }
        }

        Self {
            dimension: self.dimension,
            positions: result_positions,
            values: result_values,
        }
    }

    /// Scale by a constant.
    #[must_use]
    pub fn scale(&self, factor: f32) -> Self {
        if factor == 0.0 {
            // Everything becomes zero, which doesn't exist
            return Self::new(self.dimension);
        }

        Self {
            dimension: self.dimension,
            positions: self.positions.clone(),
            values: self.values.iter().map(|&v| v * factor).collect(),
        }
    }

    /// `L2` norm (magnitude).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn magnitude(&self) -> f32 {
        self.magnitude_f64() as f32
    }

    /// Internal f64 magnitude to avoid intermediate overflow.
    fn magnitude_f64(&self) -> f64 {
        self.values
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt()
    }

    /// Normalize to unit length. Returns None if zero vector.
    #[must_use]
    pub fn normalize(&self) -> Option<Self> {
        let mag = self.magnitude();
        if !mag.is_normal() {
            return None;
        }
        let result = self.scale(1.0 / mag);
        let result_mag = result.magnitude();
        // Return None if f32 precision loss makes normalization unreliable
        if !result_mag.is_normal() || (result_mag - 1.0).abs() > 0.01 {
            None
        } else {
            Some(result)
        }
    }

    /// Cosine similarity with another sparse vector.
    /// Returns a value in [-1.0, 1.0], with 0.0 for degenerate cases.
    /// SECURITY: Sanitizes NaN/Inf to prevent consensus ordering issues.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot = self.dot_f64(other);
        let mag_a = self.magnitude_f64();
        let mag_b = other.magnitude_f64();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        let result = dot / (mag_a * mag_b);

        if result.is_nan() || result.is_infinite() {
            0.0
        } else {
            result.clamp(-1.0, 1.0) as f32
        }
    }

    /// Cosine distance with a dense vector (1 - similarity).
    /// Returns a value in [0.0, 2.0], with 1.0 (max distance) for degenerate cases.
    /// SECURITY: Sanitizes NaN/Inf to prevent consensus ordering issues.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn cosine_distance_dense(&self, dense: &[f32]) -> f32 {
        let dot: f64 = self
            .positions
            .iter()
            .zip(&self.values)
            .map(|(&pos, &val)| f64::from(val) * f64::from(dense[pos as usize]))
            .sum();
        let mag_sparse = self.magnitude_f64();
        let mag_dense: f64 = dense
            .iter()
            .map(|x| f64::from(*x) * f64::from(*x))
            .sum::<f64>()
            .sqrt();

        if mag_sparse == 0.0 || mag_dense == 0.0 {
            return 1.0; // Maximum distance
        }

        let similarity = dot / (mag_sparse * mag_dense);

        if similarity.is_nan() || similarity.is_infinite() {
            1.0 // Maximum distance for invalid cases
        } else {
            (1.0 - similarity.clamp(-1.0, 1.0)) as f32
        }
    }

    // ========================================================================
    // Sparse Arithmetic Operations
    // ========================================================================

    /// Create a sparse delta from two dense vectors.
    ///
    /// Only stores positions where |after\[i\] - before\[i\]| > threshold.
    /// This is the core operation for eliminating dense ceremony.
    ///
    /// # Panics
    ///
    /// Panics if `before.len()` exceeds `MAX_DIMENSION`.
    /// Use [`try_from_diff`](Self::try_from_diff) for a fallible version.
    #[must_use]
    pub fn from_diff(before: &[f32], after: &[f32], threshold: f32) -> Self {
        Self::try_from_diff(before, after, threshold).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Create a sparse delta from two dense vectors.
    ///
    /// Only stores positions where |after\[i\] - before\[i\]| > threshold.
    /// This is the core operation for eliminating dense ceremony.
    ///
    /// # Errors
    ///
    /// Returns [`SparseVectorError::DimensionExceeded`] if `before.len()` > `MAX_DIMENSION`.
    #[allow(clippy::cast_possible_truncation)] // Position indices stored as u32; validated above
    pub fn try_from_diff(
        before: &[f32],
        after: &[f32],
        threshold: f32,
    ) -> Result<Self, SparseVectorError> {
        debug_assert_eq!(before.len(), after.len());
        let dimension = before.len();
        if dimension > MAX_DIMENSION {
            return Err(SparseVectorError::DimensionExceeded {
                dimension,
                max: MAX_DIMENSION,
            });
        }

        let mut positions = Vec::new();
        let mut values = Vec::new();

        for (i, (&b, &a)) in before.iter().zip(after).enumerate() {
            let delta = a - b;
            // Use > threshold (not >=) so that exact zeros are never stored
            // When threshold is 0, only non-zero deltas are stored
            if delta.abs() > threshold {
                positions.push(i as u32);
                values.push(delta);
            }
        }

        Ok(Self {
            dimension,
            positions,
            values,
        })
    }

    /// Subtract another sparse vector (self - other).
    ///
    /// Positions with zero difference are not stored.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.scale(-1.0))
    }

    /// Weighted average of two sparse vectors.
    ///
    /// Result = (w1 * self + w2 * other) / (w1 + w2)
    /// Positions with zero result are not stored.
    #[must_use]
    pub fn weighted_average(&self, other: &Self, w1: f32, w2: f32) -> Self {
        debug_assert_eq!(self.dimension, other.dimension);

        let total = w1 + w2;
        if total == 0.0 {
            return Self::new(self.dimension);
        }

        let mut result_positions = Vec::new();
        let mut result_values = Vec::new();

        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() || j < other.positions.len() {
            let (pos, val) = if i >= self.positions.len() {
                let p = other.positions[j];
                let v = w2 * other.values[j] / total;
                j += 1;
                (p, v)
            } else if j >= other.positions.len() {
                let p = self.positions[i];
                let v = w1 * self.values[i] / total;
                i += 1;
                (p, v)
            } else {
                match self.positions[i].cmp(&other.positions[j]) {
                    std::cmp::Ordering::Equal => {
                        let p = self.positions[i];
                        let v = w1.mul_add(self.values[i], w2 * other.values[j]) / total;
                        i += 1;
                        j += 1;
                        (p, v)
                    },
                    std::cmp::Ordering::Less => {
                        let p = self.positions[i];
                        let v = w1 * self.values[i] / total;
                        i += 1;
                        (p, v)
                    },
                    std::cmp::Ordering::Greater => {
                        let p = other.positions[j];
                        let v = w2 * other.values[j] / total;
                        j += 1;
                        (p, v)
                    },
                }
            };

            if val != 0.0 {
                result_positions.push(pos);
                result_values.push(val);
            }
        }

        Self {
            dimension: self.dimension,
            positions: result_positions,
            values: result_values,
        }
    }

    /// Project out the component along a direction (self - proj(self onto direction)).
    ///
    /// Used for conflict resolution: removes the conflicting component.
    #[must_use]
    pub fn project_orthogonal(&self, direction: &Self) -> Self {
        let dir_mag_sq = direction.values.iter().map(|v| v * v).sum::<f32>();
        if dir_mag_sq == 0.0 {
            return self.clone();
        }

        let dot = self.dot(direction);
        let proj_scalar = dot / dir_mag_sq;

        // self - proj_scalar * direction
        self.sub(&direction.scale(proj_scalar))
    }

    // ========================================================================
    // Geometric Distance Metrics
    // ========================================================================

    /// Angular distance: `acos(cosine_similarity)`.
    ///
    /// More linear than cosine for small angles.
    /// Range: [0, PI] where 0 = identical, PI = opposite.
    #[must_use]
    pub fn angular_distance(&self, other: &Self) -> f32 {
        let cos = self.cosine_similarity(other).clamp(-1.0, 1.0);
        cos.acos()
    }

    /// Geodesic distance on unit sphere.
    ///
    /// For normalized vectors, this is the arc length.
    /// Equivalent to `angular_distance` for normalized inputs.
    #[must_use]
    pub fn geodesic_distance(&self, other: &Self) -> f32 {
        // Geodesic on hypersphere is just arc length = angle
        self.angular_distance(other)
    }

    /// Jaccard index on non-zero positions.
    ///
    /// Measures structural overlap independent of values.
    /// |intersection| / |union| of non-zero positions.
    /// Range: [0, 1] where 1 = same positions, 0 = no overlap.
    #[must_use]
    pub fn jaccard_index(&self, other: &Self) -> f32 {
        if self.positions.is_empty() && other.positions.is_empty() {
            return 1.0; // Both empty = identical structure
        }
        if self.positions.is_empty() || other.positions.is_empty() {
            return 0.0; // One empty, one not = no overlap
        }

        let mut intersection = 0usize;
        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() && j < other.positions.len() {
            match self.positions[i].cmp(&other.positions[j]) {
                std::cmp::Ordering::Equal => {
                    intersection += 1;
                    i += 1;
                    j += 1;
                },
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }

        let union = self.positions.len() + other.positions.len() - intersection;
        #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for ratio
        {
            intersection as f32 / union as f32
        }
    }

    /// Overlap coefficient: |intersection| / min(|a|, |b|).
    ///
    /// Measures containment - 1.0 if smaller is subset of larger.
    /// Range: [0, 1].
    #[must_use]
    pub fn overlap_coefficient(&self, other: &Self) -> f32 {
        if self.positions.is_empty() || other.positions.is_empty() {
            return 0.0;
        }

        let mut intersection = 0usize;
        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() && j < other.positions.len() {
            match self.positions[i].cmp(&other.positions[j]) {
                std::cmp::Ordering::Equal => {
                    intersection += 1;
                    i += 1;
                    j += 1;
                },
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }

        let min_size = self.positions.len().min(other.positions.len());
        #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for ratio
        {
            intersection as f32 / min_size as f32
        }
    }

    /// Weighted Jaccard: sum(min(|a|, |b|)) / sum(max(|a|, |b|)).
    ///
    /// Like Jaccard but accounts for value magnitudes.
    /// Range: [0, 1] where 1 = identical values at all positions.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn weighted_jaccard(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.dimension, other.dimension);

        let mut min_sum = 0.0_f64;
        let mut max_sum = 0.0_f64;

        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() || j < other.positions.len() {
            let (a_val, b_val): (f64, f64) = if i >= self.positions.len() {
                let v = f64::from(other.values[j]).abs();
                j += 1;
                (0.0, v)
            } else if j >= other.positions.len() {
                let v = f64::from(self.values[i]).abs();
                i += 1;
                (v, 0.0)
            } else {
                match self.positions[i].cmp(&other.positions[j]) {
                    std::cmp::Ordering::Equal => {
                        let a = f64::from(self.values[i]).abs();
                        let b = f64::from(other.values[j]).abs();
                        i += 1;
                        j += 1;
                        (a, b)
                    },
                    std::cmp::Ordering::Less => {
                        let v = f64::from(self.values[i]).abs();
                        i += 1;
                        (v, 0.0)
                    },
                    std::cmp::Ordering::Greater => {
                        let v = f64::from(other.values[j]).abs();
                        j += 1;
                        (0.0, v)
                    },
                }
            };

            min_sum += a_val.min(b_val);
            max_sum += a_val.max(b_val);
        }

        if max_sum == 0.0 {
            1.0 // Both zero vectors
        } else {
            (min_sum / max_sum) as f32
        }
    }

    /// Euclidean distance (L2 norm of difference).
    ///
    /// sqrt(sum((a\[i\] - b\[i\])^2))
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        let dist = self.euclidean_distance_squared_f64(other).sqrt();
        if dist > f64::from(f32::MAX) {
            f32::MAX
        } else {
            dist as f32
        }
    }

    /// Squared Euclidean distance (avoids sqrt).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn euclidean_distance_squared(&self, other: &Self) -> f32 {
        let sq = self.euclidean_distance_squared_f64(other);
        if sq > f64::from(f32::MAX) {
            f32::MAX
        } else {
            sq as f32
        }
    }

    /// Internal f64 squared Euclidean distance to avoid intermediate overflow.
    fn euclidean_distance_squared_f64(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.dimension, other.dimension);

        let mut sum_sq = 0.0_f64;

        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() || j < other.positions.len() {
            let diff: f64 = if i >= self.positions.len() {
                let v = f64::from(other.values[j]);
                j += 1;
                v
            } else if j >= other.positions.len() {
                let v = f64::from(self.values[i]);
                i += 1;
                v
            } else {
                match self.positions[i].cmp(&other.positions[j]) {
                    std::cmp::Ordering::Equal => {
                        let d = f64::from(self.values[i]) - f64::from(other.values[j]);
                        i += 1;
                        j += 1;
                        d
                    },
                    std::cmp::Ordering::Less => {
                        let v = f64::from(self.values[i]);
                        i += 1;
                        v
                    },
                    std::cmp::Ordering::Greater => {
                        let v = f64::from(other.values[j]);
                        j += 1;
                        -v
                    },
                }
            };

            sum_sq += diff * diff;
        }

        sum_sq
    }

    /// Manhattan distance (L1 norm of difference).
    ///
    /// sum(|a\[i\] - b\[i\]|)
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn manhattan_distance(&self, other: &Self) -> f32 {
        debug_assert_eq!(self.dimension, other.dimension);

        let mut sum = 0.0_f64;

        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() || j < other.positions.len() {
            let diff: f64 = if i >= self.positions.len() {
                let v = f64::from(other.values[j]).abs();
                j += 1;
                v
            } else if j >= other.positions.len() {
                let v = f64::from(self.values[i]).abs();
                i += 1;
                v
            } else {
                match self.positions[i].cmp(&other.positions[j]) {
                    std::cmp::Ordering::Equal => {
                        let d = (f64::from(self.values[i]) - f64::from(other.values[j])).abs();
                        i += 1;
                        j += 1;
                        d
                    },
                    std::cmp::Ordering::Less => {
                        let v = f64::from(self.values[i]).abs();
                        i += 1;
                        v
                    },
                    std::cmp::Ordering::Greater => {
                        let v = f64::from(other.values[j]).abs();
                        j += 1;
                        v
                    },
                }
            };

            sum += diff;
        }

        if sum > f64::from(f32::MAX) {
            f32::MAX
        } else {
            sum as f32
        }
    }

    /// Memory usage in bytes (approximate).
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Vec::capacity() is not const
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.positions.capacity() * std::mem::size_of::<u32>()
            + self.values.capacity() * std::mem::size_of::<f32>()
    }

    /// Memory that a dense representation would use.
    #[must_use]
    pub const fn dense_memory_bytes(&self) -> usize {
        self.dimension * std::mem::size_of::<f32>()
    }

    /// Compression ratio vs dense storage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for ratio
    pub fn compression_ratio(&self) -> f32 {
        self.dense_memory_bytes() as f32 / self.memory_bytes() as f32
    }

    /// Access raw positions slice.
    #[must_use]
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// Access raw values slice.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Iterate over (position, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.positions
            .iter()
            .copied()
            .zip(self.values.iter().copied())
    }

    /// Check if this is effectively a zero vector (no stored values).
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.values.is_empty()
    }

    /// Prune small values below threshold.
    pub fn prune(&mut self, threshold: f32) {
        let mut i = 0;
        while i < self.values.len() {
            if self.values[i].abs() < threshold {
                self.positions.remove(i);
                self.values.remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Create a pruned copy.
    #[must_use]
    pub fn pruned(&self, threshold: f32) -> Self {
        let (positions, values) = self
            .positions
            .iter()
            .zip(&self.values)
            .filter(|(_, &v)| v.abs() >= threshold)
            .map(|(&p, &v)| (p, v))
            .unzip();

        Self {
            dimension: self.dimension,
            positions,
            values,
        }
    }
}

impl Default for SparseVector {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Builder for constructing sparse vectors incrementally.
pub struct SparseVectorBuilder {
    dimension: usize,
    entries: Vec<(u32, f32)>,
}

impl SparseVectorBuilder {
    /// Creates a new builder with the given dimension.
    #[must_use]
    pub const fn new(dimension: usize) -> Self {
        Self {
            dimension,
            entries: Vec::new(),
        }
    }

    /// Creates a new builder with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(dimension: usize, capacity: usize) -> Self {
        Self {
            dimension,
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Add a value at position. Zeros are ignored.
    pub fn push(&mut self, position: u32, value: f32) {
        if value != 0.0 {
            self.entries.push((position, value));
        }
    }

    /// Build the final sparse vector.
    #[must_use]
    pub fn build(mut self) -> SparseVector {
        self.entries.sort_by_key(|(p, _)| *p);

        // Deduplicate positions (keep last value)
        self.entries.dedup_by(|a, b| {
            if a.0 == b.0 {
                b.1 = a.1; // Keep a's value in b
                true
            } else {
                false
            }
        });

        let (positions, values) = self.entries.into_iter().unzip();

        SparseVector {
            dimension: self.dimension,
            positions,
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_dense_filters_zeros() {
        let dense = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let sparse = SparseVector::from_dense(&dense);

        assert_eq!(sparse.dimension(), 6);
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.positions(), &[1, 3, 5]);
        assert_eq!(sparse.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn get_returns_contextual_zero() {
        let sparse = SparseVector::from_dense(&[0.0, 1.0, 0.0, 2.0]);

        assert_eq!(sparse.get(0), 0.0); // Contextual zero
        assert_eq!(sparse.get(1), 1.0); // Stored
        assert_eq!(sparse.get(2), 0.0); // Contextual zero
        assert_eq!(sparse.get(3), 2.0); // Stored
    }

    #[test]
    fn set_removes_zero() {
        let mut sparse = SparseVector::from_dense(&[1.0, 2.0, 3.0]);

        assert_eq!(sparse.nnz(), 3);

        sparse.set(1, 0.0); // Set to zero = remove
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get(1), 0.0);
        assert!(!sparse.has_value(1));
    }

    #[test]
    fn set_inserts_nonzero() {
        let mut sparse = SparseVector::new(5);

        sparse.set(2, 1.5);
        sparse.set(4, 2.5);

        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get(2), 1.5);
        assert_eq!(sparse.get(4), 2.5);
    }

    #[test]
    fn to_dense_roundtrip() {
        let original = vec![0.0, 1.0, 0.0, 2.0, 3.0, 0.0];
        let sparse = SparseVector::from_dense(&original);
        let dense = sparse.to_dense();

        assert_eq!(dense, original);
    }

    #[test]
    fn dot_product_sparse() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 2.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 3.0, 4.0, 0.0]);

        // Only position 2 overlaps: 2.0 * 4.0 = 8.0
        assert_eq!(a.dot(&b), 8.0);
    }

    #[test]
    fn dot_product_no_overlap() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 0.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 0.0, 0.0, 2.0]);

        assert_eq!(a.dot(&b), 0.0);
    }

    #[test]
    fn dot_product_dense() {
        let sparse = SparseVector::from_dense(&[1.0, 0.0, 2.0, 0.0]);
        let dense = vec![1.0, 2.0, 3.0, 4.0];

        // 1*1 + 0*2 + 2*3 + 0*4 = 1 + 6 = 7
        assert_eq!(sparse.dot_dense(&dense), 7.0);
    }

    #[test]
    fn add_sparse_vectors() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 2.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 3.0, -2.0, 4.0]);

        let c = a.add(&b);

        // Position 2: 2.0 + (-2.0) = 0.0, should NOT be stored
        assert_eq!(c.to_dense(), vec![1.0, 3.0, 0.0, 4.0]);
        assert_eq!(c.nnz(), 3); // Not 4!
    }

    #[test]
    fn scale_by_zero_returns_empty() {
        let sparse = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let scaled = sparse.scale(0.0);

        assert!(scaled.is_zero());
        assert_eq!(scaled.nnz(), 0);
    }

    #[test]
    fn magnitude_and_normalize() {
        let sparse = SparseVector::from_dense(&[3.0, 0.0, 4.0]); // 3-4-5 triangle

        assert_eq!(sparse.magnitude(), 5.0);

        let normalized = sparse.normalize().unwrap();
        assert!((normalized.magnitude() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_identical() {
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let b = SparseVector::from_dense(&[1.0, 2.0, 3.0]);

        assert!((a.cosine_similarity(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 1.0]);

        assert!(a.cosine_similarity(&b).abs() < 1e-6);
    }

    // === Security Tests for NaN/Inf Sanitization ===

    #[test]
    fn cosine_similarity_zero_vector_returns_zero() {
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let zero = SparseVector::new(3);

        // Zero vector should return 0.0 (not NaN)
        let result = a.cosine_similarity(&zero);
        assert!(!result.is_nan());
        assert_eq!(result, 0.0);
    }

    #[test]
    fn cosine_similarity_both_zero_returns_zero() {
        let zero1 = SparseVector::new(3);
        let zero2 = SparseVector::new(3);

        let result = zero1.cosine_similarity(&zero2);
        assert!(!result.is_nan());
        assert_eq!(result, 0.0);
    }

    #[test]
    fn cosine_similarity_clamps_to_valid_range() {
        // Identical vectors should give exactly 1.0 (not 1.0000001 due to floating point)
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let result = a.cosine_similarity(&a);
        assert!((-1.0..=1.0).contains(&result));
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite_returns_negative_one() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let b = SparseVector::from_dense(&[-1.0, 0.0, 0.0]);

        let result = a.cosine_similarity(&b);
        assert!((-1.0..=1.0).contains(&result));
        assert!((result - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_distance_dense_zero_sparse_returns_max_distance() {
        let zero = SparseVector::new(3);
        let dense = vec![1.0, 2.0, 3.0];

        let result = zero.cosine_distance_dense(&dense);
        assert!(!result.is_nan());
        assert_eq!(result, 1.0); // Max distance
    }

    #[test]
    fn cosine_distance_dense_zero_dense_returns_max_distance() {
        let sparse = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let zero_dense = vec![0.0, 0.0, 0.0];

        let result = sparse.cosine_distance_dense(&zero_dense);
        assert!(!result.is_nan());
        assert_eq!(result, 1.0); // Max distance
    }

    #[test]
    fn cosine_distance_dense_valid_range() {
        let sparse = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let dense = vec![1.0, 0.0, 0.0];

        let result = sparse.cosine_distance_dense(&dense);
        assert!((0.0..=2.0).contains(&result));
        assert!(result < 1e-6); // Should be ~0 for identical vectors
    }

    #[test]
    fn sparsity_calculation() {
        let sparse = SparseVector::from_dense(&[0.0, 1.0, 0.0, 0.0, 2.0]); // 3/5 = 60% zeros

        assert!((sparse.sparsity() - 0.6).abs() < 1e-6);
    }

    #[test]
    fn prune_small_values() {
        let sparse = SparseVector::from_dense(&[0.001, 1.0, 0.002, 2.0]);
        let pruned = sparse.pruned(0.01);

        assert_eq!(pruned.nnz(), 2);
        assert_eq!(pruned.to_dense(), vec![0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn builder_deduplicates() {
        let mut builder = SparseVectorBuilder::new(5);
        builder.push(1, 1.0);
        builder.push(1, 2.0); // Duplicate position
        builder.push(3, 3.0);

        let sparse = builder.build();

        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get(1), 2.0); // Last value wins
    }

    #[test]
    fn from_dense_with_threshold() {
        let dense = vec![0.01, 1.0, 0.02, 2.0, 0.001];
        let sparse = SparseVector::from_dense_with_threshold(&dense, 0.05);

        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.positions(), &[1, 3]);
    }

    #[test]
    fn memory_compression() {
        // 90% sparse vector
        let mut dense = vec![0.0; 1000];
        for i in (0..100).map(|x| x * 10) {
            dense[i] = 1.0;
        }

        let sparse = SparseVector::from_dense(&dense);

        assert!(sparse.compression_ratio() > 2.0);
    }

    #[test]
    fn iter_pairs() {
        let sparse = SparseVector::from_dense(&[0.0, 1.0, 0.0, 2.0]);
        let pairs: Vec<_> = sparse.iter().collect();

        assert_eq!(pairs, vec![(1, 1.0), (3, 2.0)]);
    }

    #[test]
    fn empty_vector() {
        let sparse = SparseVector::new(10);

        assert!(sparse.is_zero());
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.magnitude(), 0.0);
        assert!(sparse.normalize().is_none());
    }

    #[test]
    fn has_value_check() {
        let sparse = SparseVector::from_dense(&[0.0, 1.0, 0.0]);

        assert!(!sparse.has_value(0));
        assert!(sparse.has_value(1));
        assert!(!sparse.has_value(2));
    }

    // ========================================================================
    // Tests for new sparse arithmetic operations
    // ========================================================================

    #[test]
    fn from_diff_creates_sparse_delta() {
        let before = vec![1.0, 2.0, 3.0, 4.0];
        let after = vec![1.0, 2.5, 3.0, 5.0]; // positions 1 and 3 changed

        let delta = SparseVector::from_diff(&before, &after, 0.0);

        assert_eq!(delta.nnz(), 2);
        assert_eq!(delta.get(1), 0.5); // 2.5 - 2.0
        assert_eq!(delta.get(3), 1.0); // 5.0 - 4.0
    }

    #[test]
    fn from_diff_respects_threshold() {
        let before = vec![1.0, 2.0, 3.0];
        let after = vec![1.01, 2.5, 3.001]; // 0.01, 0.5, 0.001

        let delta = SparseVector::from_diff(&before, &after, 0.1);

        assert_eq!(delta.nnz(), 1); // Only 0.5 exceeds threshold
        assert_eq!(delta.get(1), 0.5);
    }

    #[test]
    fn sub_sparse_vectors() {
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0, 0.0]);
        let b = SparseVector::from_dense(&[0.5, 2.0, 1.0, 1.0]);

        let diff = a.sub(&b);

        assert_eq!(diff.get(0), 0.5); // 1.0 - 0.5
        assert_eq!(diff.get(1), 0.0); // 2.0 - 2.0 = 0, not stored
        assert_eq!(diff.get(2), 2.0); // 3.0 - 1.0
        assert_eq!(diff.get(3), -1.0); // 0.0 - 1.0
    }

    #[test]
    fn weighted_average_sparse() {
        let a = SparseVector::from_dense(&[2.0, 0.0, 4.0]);
        let b = SparseVector::from_dense(&[0.0, 6.0, 0.0]);

        let avg = a.weighted_average(&b, 1.0, 1.0); // Equal weights

        assert_eq!(avg.get(0), 1.0); // (1*2 + 1*0) / 2 = 1
        assert_eq!(avg.get(1), 3.0); // (1*0 + 1*6) / 2 = 3
        assert_eq!(avg.get(2), 2.0); // (1*4 + 1*0) / 2 = 2
    }

    #[test]
    fn weighted_average_with_weights() {
        let a = SparseVector::from_dense(&[4.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 8.0]);

        let avg = a.weighted_average(&b, 3.0, 1.0); // 3:1 ratio

        assert_eq!(avg.get(0), 3.0); // (3*4 + 1*0) / 4 = 3
        assert_eq!(avg.get(1), 2.0); // (3*0 + 1*8) / 4 = 2
    }

    #[test]
    fn project_orthogonal_perpendicular() {
        // Vector at 45 degrees, project out x-axis component
        let v = SparseVector::from_dense(&[1.0, 1.0]);
        let x_axis = SparseVector::from_dense(&[1.0, 0.0]);

        let result = v.project_orthogonal(&x_axis);

        assert!(result.get(0).abs() < 1e-6); // x component removed
        assert!((result.get(1) - 1.0).abs() < 1e-6); // y component preserved
    }

    #[test]
    fn project_orthogonal_zero_direction() {
        let v = SparseVector::from_dense(&[1.0, 2.0]);
        let zero = SparseVector::new(2);

        let result = v.project_orthogonal(&zero);

        assert_eq!(result.to_dense(), v.to_dense());
    }

    // ========================================================================
    // Tests for geometric distance metrics
    // ========================================================================

    #[test]
    fn angular_distance_identical() {
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let b = SparseVector::from_dense(&[1.0, 2.0, 3.0]);

        // Use 1e-3 tolerance due to floating point in cosine -> acos
        assert!(a.angular_distance(&b).abs() < 1e-3);
    }

    #[test]
    fn angular_distance_orthogonal() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 1.0]);

        let dist = a.angular_distance(&b);
        assert!((dist - std::f32::consts::FRAC_PI_2).abs() < 1e-6); // PI/2
    }

    #[test]
    fn angular_distance_opposite() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[-1.0, 0.0]);

        let dist = a.angular_distance(&b);
        assert!((dist - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn jaccard_identical_structure() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 2.0, 0.0]);
        let b = SparseVector::from_dense(&[3.0, 0.0, 4.0, 0.0]); // Same positions

        assert!((a.jaccard_index(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_no_overlap() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 0.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 0.0, 2.0, 0.0]);

        assert!(a.jaccard_index(&b).abs() < 1e-6);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let a = SparseVector::from_dense(&[1.0, 2.0, 0.0, 0.0]); // positions 0, 1
        let b = SparseVector::from_dense(&[0.0, 3.0, 4.0, 0.0]); // positions 1, 2

        // Intersection: {1}, Union: {0, 1, 2} => 1/3
        assert!((a.jaccard_index(&b) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn overlap_coefficient_subset() {
        let small = SparseVector::from_dense(&[0.0, 1.0, 0.0]);
        let large = SparseVector::from_dense(&[1.0, 2.0, 3.0]);

        // Small has 1 position, large has 3, intersection is 1
        // 1 / min(1, 3) = 1.0
        assert!((small.overlap_coefficient(&large) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_jaccard_identical() {
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let b = SparseVector::from_dense(&[1.0, 2.0, 3.0]);

        assert!((a.weighted_jaccard(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn weighted_jaccard_different_magnitudes() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[2.0, 0.0]);

        // min(1, 2) / max(1, 2) = 1/2 = 0.5
        assert!((a.weighted_jaccard(&b) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance_orthogonal_unit() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 1.0]);

        // sqrt(1^2 + 1^2) = sqrt(2)
        assert!((a.euclidean_distance(&b) - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance_squared_avoids_sqrt() {
        let a = SparseVector::from_dense(&[3.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 4.0]);

        // 3^2 + 4^2 = 9 + 16 = 25
        assert!((a.euclidean_distance_squared(&b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn manhattan_distance_sparse() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 3.0]);
        let b = SparseVector::from_dense(&[0.0, 2.0, 1.0]);

        // |1-0| + |0-2| + |3-1| = 1 + 2 + 2 = 5
        assert!((a.manhattan_distance(&b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn geodesic_equals_angular_for_unit_vectors() {
        let a = SparseVector::from_dense(&[1.0, 0.0]);
        let b = SparseVector::from_dense(&[0.707, 0.707]);

        let angular = a.angular_distance(&b);
        let geodesic = a.geodesic_distance(&b);

        assert!((angular - geodesic).abs() < 1e-6);
    }

    // Additional edge case coverage tests

    #[test]
    fn sparsity_zero_dimension() {
        let sv = SparseVector::new(0);
        assert_eq!(sv.sparsity(), 0.0);
    }

    #[test]
    fn weighted_average_zero_weights() {
        let a = SparseVector::from_dense(&[1.0, 2.0]);
        let b = SparseVector::from_dense(&[3.0, 4.0]);
        let result = a.weighted_average(&b, 0.0, 0.0);
        assert_eq!(result.dimension(), 2);
        assert_eq!(result.nnz(), 0);
    }

    #[test]
    fn weighted_average_overlapping_positions() {
        // Both vectors have values at positions 0 and 2
        let a = SparseVector::from_dense(&[1.0, 0.0, 3.0]);
        let b = SparseVector::from_dense(&[2.0, 0.0, 4.0]);
        let result = a.weighted_average(&b, 1.0, 1.0);
        // (1*1 + 1*2)/2 = 1.5, (1*3 + 1*4)/2 = 3.5
        assert!((result.get(0) - 1.5).abs() < 1e-6);
        assert!((result.get(2) - 3.5).abs() < 1e-6);
    }

    #[test]
    fn weighted_jaccard_different_positions() {
        // a has values at 0, 2; b has values at 1, 2
        let a = SparseVector::from_dense(&[1.0, 0.0, 3.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 2.0, 4.0, 0.0]);
        // This exercises all branches in weighted_jaccard merge loop
        let result = a.weighted_jaccard(&b);
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn weighted_jaccard_one_extends_past_other() {
        // a is longer than b after merge
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0, 4.0]);
        let b = SparseVector::from_dense(&[1.0, 0.0, 0.0, 0.0]);
        let result = a.weighted_jaccard(&b);
        assert!((0.0..=1.0).contains(&result));
    }

    #[test]
    fn euclidean_distance_disjoint_positions() {
        // a has values at 0, 2; b has values at 1, 3
        let a = SparseVector::from_dense(&[1.0, 0.0, 2.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 3.0, 0.0, 4.0]);
        let dist = a.euclidean_distance(&b);
        // sqrt(1^2 + 3^2 + 2^2 + 4^2) = sqrt(1+9+4+16) = sqrt(30)
        assert!((dist - 30.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn euclidean_distance_a_extends_past_b() {
        let a = SparseVector::from_dense(&[1.0, 2.0, 3.0, 4.0]);
        let b = SparseVector::from_dense(&[1.0, 0.0, 0.0, 0.0]);
        let dist = a.euclidean_distance(&b);
        // sqrt(0 + 4 + 9 + 16) = sqrt(29)
        assert!((dist - 29.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn euclidean_distance_b_extends_past_a() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 0.0, 0.0]);
        let b = SparseVector::from_dense(&[1.0, 2.0, 3.0, 4.0]);
        let dist = a.euclidean_distance(&b);
        // sqrt(0 + 4 + 9 + 16) = sqrt(29)
        assert!((dist - 29.0_f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn manhattan_distance_disjoint_positions() {
        let a = SparseVector::from_dense(&[1.0, 0.0, 2.0, 0.0]);
        let b = SparseVector::from_dense(&[0.0, 3.0, 0.0, 4.0]);
        let dist = a.manhattan_distance(&b);
        // |1| + |3| + |2| + |4| = 10
        assert!((dist - 10.0).abs() < 1e-6);
    }

    // ========================================================================
    // Security validation tests for integer truncation prevention
    // ========================================================================

    #[test]
    fn test_try_new_at_max_boundary() {
        // MAX_DIMENSION is u32::MAX, which is valid
        let result = SparseVector::try_new(MAX_DIMENSION);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dimension(), MAX_DIMENSION);
    }

    #[test]
    fn test_try_new_exceeds_max() {
        let result = SparseVector::try_new(MAX_DIMENSION + 1);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            SparseVectorError::DimensionExceeded {
                dimension: MAX_DIMENSION + 1,
                max: MAX_DIMENSION
            }
        );
    }

    #[test]
    fn test_try_from_parts_validates_positions() {
        // Position 10 is out of bounds for dimension 5
        let result = SparseVector::try_from_parts(5, vec![0, 2, 10], vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            SparseVectorError::IndexOutOfBounds {
                index: 10,
                dimension: 5
            }
        );
    }

    #[test]
    fn test_try_from_parts_valid() {
        let result = SparseVector::try_from_parts(10, vec![0, 5, 9], vec![1.0, 2.0, 3.0]);
        assert!(result.is_ok());
        let sv = result.unwrap();
        assert_eq!(sv.dimension(), 10);
        assert_eq!(sv.nnz(), 3);
    }

    #[test]
    fn test_try_set_index_out_of_bounds() {
        let mut sv = SparseVector::new(5);
        let result = sv.try_set(5, 1.0);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            SparseVectorError::IndexOutOfBounds {
                index: 5,
                dimension: 5
            }
        );
    }

    #[test]
    fn test_try_set_valid() {
        let mut sv = SparseVector::new(5);
        let result = sv.try_set(4, 1.0);
        assert!(result.is_ok());
        assert_eq!(sv.get(4), 1.0);
    }

    #[test]
    fn test_sparse_vector_error_display() {
        let err = SparseVectorError::DimensionExceeded {
            dimension: 5_000_000_000,
            max: MAX_DIMENSION,
        };
        let msg = format!("{err}");
        assert!(msg.contains("5000000000"));
        assert!(msg.contains("exceeds maximum"));

        let err2 = SparseVectorError::IndexOutOfBounds {
            index: 100,
            dimension: 50,
        };
        let msg2 = format!("{err2}");
        assert!(msg2.contains("100"));
        assert!(msg2.contains("50"));
        assert!(msg2.contains("out of bounds"));
    }

    #[test]
    fn test_max_dimension_constant() {
        // Verify the constant matches u32::MAX
        assert_eq!(MAX_DIMENSION, u32::MAX as usize);
    }
}
