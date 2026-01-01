//! Sparse Vector - Storage where zero doesn't exist
//!
//! Philosophy: Zero represents absence of information, not a value to store.
//! Only non-zero values are fundamental. The dimension defines the boundary
//! (shell) of meaningful space.

use serde::{Deserialize, Serialize};

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
/// assert_eq!(sparse.dimension(), 6);  // Shell size
/// assert_eq!(sparse.nnz(), 2);        // Only 2 values stored
/// assert_eq!(sparse.get(1), 1.5);     // Stored value
/// assert_eq!(sparse.get(0), 0.0);     // Contextual zero (in shell, not stored)
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
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            positions: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create a sparse vector with pre-allocated capacity.
    pub fn with_capacity(dimension: usize, capacity: usize) -> Self {
        Self {
            dimension,
            positions: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    /// Create from parallel arrays of positions and values.
    ///
    /// Filters out zeros and sorts by position.
    pub fn from_parts(dimension: usize, positions: Vec<u32>, values: Vec<f32>) -> Self {
        debug_assert_eq!(positions.len(), values.len());

        // Filter zeros and pair with positions
        let mut pairs: Vec<(u32, f32)> = positions
            .into_iter()
            .zip(values)
            .filter(|(_, v)| *v != 0.0)
            .collect();

        // Sort by position
        pairs.sort_by_key(|(p, _)| *p);

        // Unzip back
        let (positions, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();

        Self {
            dimension,
            positions,
            values,
        }
    }

    /// Create from a dense vector, storing only non-zero values.
    pub fn from_dense(dense: &[f32]) -> Self {
        let dimension = dense.len();
        let mut positions = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in dense.iter().enumerate() {
            if val != 0.0 {
                positions.push(i as u32);
                values.push(val);
            }
        }

        Self {
            dimension,
            positions,
            values,
        }
    }

    /// Create from a dense vector with a threshold - values below threshold become zero.
    pub fn from_dense_with_threshold(dense: &[f32], threshold: f32) -> Self {
        let dimension = dense.len();
        let mut positions = Vec::new();
        let mut values = Vec::new();

        for (i, &val) in dense.iter().enumerate() {
            if val.abs() >= threshold {
                positions.push(i as u32);
                values.push(val);
            }
        }

        Self {
            dimension,
            positions,
            values,
        }
    }

    /// The shell/boundary - total dimension of the vector space.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Number of non-zero values stored.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity ratio (fraction of zeros).
    #[inline]
    pub fn sparsity(&self) -> f32 {
        if self.dimension == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f32 / self.dimension as f32)
        }
    }

    /// Check if a position is within the shell boundary.
    #[inline]
    pub fn in_bounds(&self, index: usize) -> bool {
        index < self.dimension
    }

    /// Get value at position.
    ///
    /// Returns 0.0 for positions within shell that have no stored value.
    /// This is a "contextual zero" - absence of information, not stored zero.
    #[inline]
    pub fn get(&self, index: usize) -> f32 {
        debug_assert!(index < self.dimension, "Index out of bounds");

        // Binary search for position
        match self.positions.binary_search(&(index as u32)) {
            Ok(i) => self.values[i],
            Err(_) => 0.0, // Contextual zero
        }
    }

    /// Check if a position has a stored (non-zero) value.
    #[inline]
    pub fn has_value(&self, index: usize) -> bool {
        self.positions.binary_search(&(index as u32)).is_ok()
    }

    /// Set value at position.
    ///
    /// If value is zero, removes from storage (zero doesn't exist).
    /// If value is non-zero, inserts or updates.
    pub fn set(&mut self, index: usize, value: f32) {
        debug_assert!(index < self.dimension, "Index out of bounds");

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
    }

    /// Convert to dense representation.
    ///
    /// This "realizes" all the contextual zeros as actual values.
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.dimension];
        for (&pos, &val) in self.positions.iter().zip(&self.values) {
            dense[pos as usize] = val;
        }
        dense
    }

    /// Dot product with another sparse vector.
    ///
    /// O(min(nnz_a, nnz_b)) - only overlapping non-zero positions contribute.
    /// This is where sparse shines: zero * anything = zero, and we don't store zeros.
    pub fn dot(&self, other: &SparseVector) -> f32 {
        debug_assert_eq!(
            self.dimension, other.dimension,
            "Dimension mismatch: {} vs {}",
            self.dimension, other.dimension
        );

        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() && j < other.positions.len() {
            match self.positions[i].cmp(&other.positions[j]) {
                std::cmp::Ordering::Equal => {
                    result += self.values[i] * other.values[j];
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
    /// O(nnz) - only iterate over our non-zero positions.
    pub fn dot_dense(&self, dense: &[f32]) -> f32 {
        debug_assert_eq!(
            self.dimension,
            dense.len(),
            "Dimension mismatch: {} vs {}",
            self.dimension,
            dense.len()
        );

        self.positions
            .iter()
            .zip(&self.values)
            .map(|(&pos, &val)| val * dense[pos as usize])
            .sum()
    }

    /// Add another sparse vector. Returns new sparse vector.
    ///
    /// Positions with zero sum are not stored in result.
    pub fn add(&self, other: &SparseVector) -> SparseVector {
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

        SparseVector {
            dimension: self.dimension,
            positions: result_positions,
            values: result_values,
        }
    }

    /// Scale by a constant.
    pub fn scale(&self, factor: f32) -> SparseVector {
        if factor == 0.0 {
            // Everything becomes zero, which doesn't exist
            return SparseVector::new(self.dimension);
        }

        SparseVector {
            dimension: self.dimension,
            positions: self.positions.clone(),
            values: self.values.iter().map(|&v| v * factor).collect(),
        }
    }

    /// L2 norm (magnitude).
    pub fn magnitude(&self) -> f32 {
        self.values.iter().map(|v| v * v).sum::<f32>().sqrt()
    }

    /// Normalize to unit length. Returns None if zero vector.
    pub fn normalize(&self) -> Option<SparseVector> {
        let mag = self.magnitude();
        if mag == 0.0 {
            None
        } else {
            Some(self.scale(1.0 / mag))
        }
    }

    /// Cosine similarity with another sparse vector.
    pub fn cosine_similarity(&self, other: &SparseVector) -> f32 {
        let dot = self.dot(other);
        let mag_a = self.magnitude();
        let mag_b = other.magnitude();

        if mag_a == 0.0 || mag_b == 0.0 {
            0.0
        } else {
            dot / (mag_a * mag_b)
        }
    }

    /// Cosine distance with a dense vector (1 - similarity).
    pub fn cosine_distance_dense(&self, dense: &[f32]) -> f32 {
        let dot = self.dot_dense(dense);
        let mag_sparse = self.magnitude();
        let mag_dense: f32 = dense.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_sparse == 0.0 || mag_dense == 0.0 {
            1.0 // Maximum distance
        } else {
            1.0 - (dot / (mag_sparse * mag_dense))
        }
    }

    // ========================================================================
    // Sparse Arithmetic Operations
    // ========================================================================

    /// Create a sparse delta from two dense vectors.
    ///
    /// Only stores positions where |after\[i\] - before\[i\]| > threshold.
    /// This is the core operation for eliminating dense ceremony.
    pub fn from_diff(before: &[f32], after: &[f32], threshold: f32) -> Self {
        debug_assert_eq!(before.len(), after.len());
        let dimension = before.len();
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

        Self {
            dimension,
            positions,
            values,
        }
    }

    /// Subtract another sparse vector (self - other).
    ///
    /// Positions with zero difference are not stored.
    pub fn sub(&self, other: &SparseVector) -> SparseVector {
        self.add(&other.scale(-1.0))
    }

    /// Weighted average of two sparse vectors.
    ///
    /// Result = (w1 * self + w2 * other) / (w1 + w2)
    /// Positions with zero result are not stored.
    pub fn weighted_average(&self, other: &SparseVector, w1: f32, w2: f32) -> SparseVector {
        debug_assert_eq!(self.dimension, other.dimension);

        let total = w1 + w2;
        if total == 0.0 {
            return SparseVector::new(self.dimension);
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
                        let v = (w1 * self.values[i] + w2 * other.values[j]) / total;
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

        SparseVector {
            dimension: self.dimension,
            positions: result_positions,
            values: result_values,
        }
    }

    /// Project out the component along a direction (self - proj(self onto direction)).
    ///
    /// Used for conflict resolution: removes the conflicting component.
    pub fn project_orthogonal(&self, direction: &SparseVector) -> SparseVector {
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

    /// Angular distance: acos(cosine_similarity).
    ///
    /// More linear than cosine for small angles.
    /// Range: [0, PI] where 0 = identical, PI = opposite.
    pub fn angular_distance(&self, other: &SparseVector) -> f32 {
        let cos = self.cosine_similarity(other).clamp(-1.0, 1.0);
        cos.acos()
    }

    /// Geodesic distance on unit sphere.
    ///
    /// For normalized vectors, this is the arc length.
    /// Equivalent to angular_distance for normalized inputs.
    pub fn geodesic_distance(&self, other: &SparseVector) -> f32 {
        // Geodesic on hypersphere is just arc length = angle
        self.angular_distance(other)
    }

    /// Jaccard index on non-zero positions.
    ///
    /// Measures structural overlap independent of values.
    /// |intersection| / |union| of non-zero positions.
    /// Range: [0, 1] where 1 = same positions, 0 = no overlap.
    pub fn jaccard_index(&self, other: &SparseVector) -> f32 {
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
        intersection as f32 / union as f32
    }

    /// Overlap coefficient: |intersection| / min(|a|, |b|).
    ///
    /// Measures containment - 1.0 if smaller is subset of larger.
    /// Range: [0, 1].
    pub fn overlap_coefficient(&self, other: &SparseVector) -> f32 {
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
        intersection as f32 / min_size as f32
    }

    /// Weighted Jaccard: sum(min(|a|, |b|)) / sum(max(|a|, |b|)).
    ///
    /// Like Jaccard but accounts for value magnitudes.
    /// Range: [0, 1] where 1 = identical values at all positions.
    pub fn weighted_jaccard(&self, other: &SparseVector) -> f32 {
        debug_assert_eq!(self.dimension, other.dimension);

        let mut min_sum = 0.0f32;
        let mut max_sum = 0.0f32;

        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() || j < other.positions.len() {
            let (a_val, b_val) = if i >= self.positions.len() {
                let v = other.values[j].abs();
                j += 1;
                (0.0, v)
            } else if j >= other.positions.len() {
                let v = self.values[i].abs();
                i += 1;
                (v, 0.0)
            } else {
                match self.positions[i].cmp(&other.positions[j]) {
                    std::cmp::Ordering::Equal => {
                        let a = self.values[i].abs();
                        let b = other.values[j].abs();
                        i += 1;
                        j += 1;
                        (a, b)
                    },
                    std::cmp::Ordering::Less => {
                        let v = self.values[i].abs();
                        i += 1;
                        (v, 0.0)
                    },
                    std::cmp::Ordering::Greater => {
                        let v = other.values[j].abs();
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
            min_sum / max_sum
        }
    }

    /// Euclidean distance (L2 norm of difference).
    ///
    /// sqrt(sum((a\[i\] - b\[i\])^2))
    pub fn euclidean_distance(&self, other: &SparseVector) -> f32 {
        self.euclidean_distance_squared(other).sqrt()
    }

    /// Squared Euclidean distance (avoids sqrt).
    pub fn euclidean_distance_squared(&self, other: &SparseVector) -> f32 {
        debug_assert_eq!(self.dimension, other.dimension);

        let mut sum_sq = 0.0f32;

        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() || j < other.positions.len() {
            let diff = if i >= self.positions.len() {
                let v = other.values[j];
                j += 1;
                v // 0 - v = -v, squared = v^2
            } else if j >= other.positions.len() {
                let v = self.values[i];
                i += 1;
                v // v - 0 = v
            } else {
                match self.positions[i].cmp(&other.positions[j]) {
                    std::cmp::Ordering::Equal => {
                        let d = self.values[i] - other.values[j];
                        i += 1;
                        j += 1;
                        d
                    },
                    std::cmp::Ordering::Less => {
                        let v = self.values[i];
                        i += 1;
                        v
                    },
                    std::cmp::Ordering::Greater => {
                        let v = other.values[j];
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
    pub fn manhattan_distance(&self, other: &SparseVector) -> f32 {
        debug_assert_eq!(self.dimension, other.dimension);

        let mut sum = 0.0f32;

        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() || j < other.positions.len() {
            let diff = if i >= self.positions.len() {
                let v = other.values[j].abs();
                j += 1;
                v
            } else if j >= other.positions.len() {
                let v = self.values[i].abs();
                i += 1;
                v
            } else {
                match self.positions[i].cmp(&other.positions[j]) {
                    std::cmp::Ordering::Equal => {
                        let d = (self.values[i] - other.values[j]).abs();
                        i += 1;
                        j += 1;
                        d
                    },
                    std::cmp::Ordering::Less => {
                        let v = self.values[i].abs();
                        i += 1;
                        v
                    },
                    std::cmp::Ordering::Greater => {
                        let v = other.values[j].abs();
                        j += 1;
                        v
                    },
                }
            };

            sum += diff;
        }

        sum
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.positions.capacity() * std::mem::size_of::<u32>()
            + self.values.capacity() * std::mem::size_of::<f32>()
    }

    /// Memory that a dense representation would use.
    pub fn dense_memory_bytes(&self) -> usize {
        self.dimension * std::mem::size_of::<f32>()
    }

    /// Compression ratio vs dense storage.
    pub fn compression_ratio(&self) -> f32 {
        self.dense_memory_bytes() as f32 / self.memory_bytes() as f32
    }

    /// Access raw positions slice.
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// Access raw values slice.
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
    pub fn is_zero(&self) -> bool {
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
    pub fn pruned(&self, threshold: f32) -> SparseVector {
        let pairs: Vec<_> = self
            .positions
            .iter()
            .zip(&self.values)
            .filter(|(_, &v)| v.abs() >= threshold)
            .map(|(&p, &v)| (p, v))
            .collect();

        let (positions, values) = pairs.into_iter().unzip();

        SparseVector {
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
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            entries: Vec::new(),
        }
    }

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
}
