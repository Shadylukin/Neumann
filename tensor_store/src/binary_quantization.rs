// SPDX-License-Identifier: MIT OR Apache-2.0
//! Binary Quantization for extreme vector compression.
//!
//! Binary quantization converts each dimension to a single bit, achieving
//! 32x compression. Distance is computed using Hamming distance via POPCNT.
//!
//! # Example
//!
//! ```rust
//! use tensor_store::binary_quantization::{BinaryVector, BinaryThreshold};
//!
//! let vector = vec![0.1, -0.5, 0.3, -0.2, 0.8, -0.1, 0.0, 0.4];
//! let binary = BinaryVector::from_dense(&vector, BinaryThreshold::Sign);
//!
//! // 8 dimensions -> 1 byte (8 bits)
//! assert_eq!(binary.memory_bytes(), 8); // 1 u64 word = 8 bytes minimum
//!
//! let other = BinaryVector::from_dense(&[0.2, -0.3, 0.1, -0.4, 0.9, -0.2, 0.1, 0.5], BinaryThreshold::Sign);
//! let hamming = binary.hamming_distance(&other);
//! println!("Hamming distance: {hamming}");
//! ```

use serde::{Deserialize, Serialize};

/// Binarization threshold method.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryThreshold {
    /// bit = 1 if value > 0 (default).
    #[default]
    Sign,
    /// bit = 1 if value > mean(vector).
    Mean,
    /// bit = 1 if value > median(vector).
    Median,
}

impl BinaryThreshold {
    /// Compute the threshold value for a given vector.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute(&self, vector: &[f32]) -> f32 {
        match self {
            Self::Sign => 0.0,
            Self::Mean => {
                if vector.is_empty() {
                    0.0
                } else {
                    vector.iter().sum::<f32>() / vector.len() as f32
                }
            },
            Self::Median => {
                if vector.is_empty() {
                    return 0.0;
                }
                let mut sorted: Vec<f32> = vector.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = sorted.len() / 2;
                if sorted.len().is_multiple_of(2) {
                    sorted[mid - 1].midpoint(sorted[mid])
                } else {
                    sorted[mid]
                }
            },
        }
    }
}

/// Binary quantized vector packed into u64 words.
///
/// Each dimension is encoded as a single bit, with bits packed into u64 words.
/// Achieves 32x compression compared to f32 vectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryVector {
    /// Packed bit data. ceil(dimension/64) words.
    data: Vec<u64>,
    /// Original vector dimension.
    dimension: usize,
}

impl BinaryVector {
    /// Create a binary vector from a dense vector using the specified threshold.
    #[must_use]
    pub fn from_dense(vector: &[f32], threshold: BinaryThreshold) -> Self {
        let dimension = vector.len();
        let num_words = dimension.div_ceil(64);
        let mut data = vec![0u64; num_words];

        let threshold_value = threshold.compute(vector);

        for (i, &v) in vector.iter().enumerate() {
            if v > threshold_value {
                let word_idx = i / 64;
                let bit_idx = i % 64;
                data[word_idx] |= 1u64 << bit_idx;
            }
        }

        Self { data, dimension }
    }

    /// Create a binary vector from raw packed data.
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match expected words for dimension.
    #[must_use]
    pub fn from_raw(data: Vec<u64>, dimension: usize) -> Self {
        let expected_words = dimension.div_ceil(64);
        assert_eq!(
            data.len(),
            expected_words,
            "Data length {} doesn't match expected {} words for dimension {}",
            data.len(),
            expected_words,
            dimension
        );
        Self { data, dimension }
    }

    /// Compute Hamming distance (number of differing bits) using POPCNT.
    #[inline]
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Normalized Hamming similarity in [0, 1].
    ///
    /// Returns 1.0 for identical vectors, 0.0 for completely opposite.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn similarity(&self, other: &Self) -> f32 {
        if self.dimension == 0 {
            return 1.0;
        }
        1.0 - (self.hamming_distance(other) as f32 / self.dimension as f32)
    }

    /// Hamming distance normalized to [0, 1].
    #[inline]
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn normalized_distance(&self, other: &Self) -> f32 {
        if self.dimension == 0 {
            return 0.0;
        }
        self.hamming_distance(other) as f32 / self.dimension as f32
    }

    /// Convert to approximate dense vector (0.0 or 1.0 values).
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dimension);
        for i in 0..self.dimension {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            let bit = (self.data[word_idx] >> bit_idx) & 1;
            result.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        result
    }

    /// Returns the original vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns memory usage in bytes.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<u64>()
    }

    /// Returns the raw packed data.
    #[must_use]
    pub fn as_raw(&self) -> &[u64] {
        &self.data
    }

    /// Returns the number of set bits (ones).
    #[must_use]
    pub fn popcount(&self) -> u32 {
        self.data.iter().map(|w| w.count_ones()).sum()
    }

    /// Compute dot product with another binary vector.
    ///
    /// Returns the number of positions where both bits are 1.
    #[must_use]
    pub fn dot(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a & b).count_ones())
            .sum()
    }

    /// Compute Jaccard similarity between binary vectors.
    ///
    /// Jaccard = |A AND B| / |A OR B|
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn jaccard_similarity(&self, other: &Self) -> f32 {
        let intersection: u32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a & b).count_ones())
            .sum();

        let union: u32 = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a | b).count_ones())
            .sum();

        if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_from_dense_sign() {
        let vector = vec![0.1, -0.5, 0.3, -0.2, 0.8, -0.1, 0.0, 0.4];
        let binary = BinaryVector::from_dense(&vector, BinaryThreshold::Sign);

        assert_eq!(binary.dimension(), 8);
        // With threshold > 0: 0.1>0:1, -0.5>0:0, 0.3>0:1, -0.2>0:0, 0.8>0:1, -0.1>0:0, 0.0>0:0, 0.4>0:1
        // Bits (LSB first): 1, 0, 1, 0, 1, 0, 0, 1 = 0b10010101 = 149
        assert_eq!(binary.data[0], 0b1001_0101);
    }

    #[test]
    fn binary_from_dense_mean() {
        let vector = vec![1.0, 2.0, 3.0, 4.0]; // mean = 2.5
        let binary = BinaryVector::from_dense(&vector, BinaryThreshold::Mean);

        // 1.0 < 2.5: 0, 2.0 < 2.5: 0, 3.0 > 2.5: 1, 4.0 > 2.5: 1
        // Bits: 0, 0, 1, 1 = 0b1100 = 12
        assert_eq!(binary.data[0], 0b1100);
    }

    #[test]
    fn binary_from_dense_median() {
        let vector = vec![1.0, 5.0, 2.0, 4.0, 3.0]; // sorted: 1,2,3,4,5 median = 3
        let binary = BinaryVector::from_dense(&vector, BinaryThreshold::Median);

        // 1.0 < 3: 0, 5.0 > 3: 1, 2.0 < 3: 0, 4.0 > 3: 1, 3.0 = 3: 0
        // Bits: 0, 1, 0, 1, 0 = 0b01010 = 10
        assert_eq!(binary.data[0], 0b0_1010);
    }

    #[test]
    fn binary_hamming_basic() {
        let a = BinaryVector::from_raw(vec![0b1010_1010], 8);
        let b = BinaryVector::from_raw(vec![0b0101_0101], 8);

        // All 8 bits differ
        assert_eq!(a.hamming_distance(&b), 8);
    }

    #[test]
    fn binary_hamming_identical() {
        let a = BinaryVector::from_raw(vec![0b1111_0000], 8);
        let b = BinaryVector::from_raw(vec![0b1111_0000], 8);

        assert_eq!(a.hamming_distance(&b), 0);
    }

    #[test]
    fn binary_hamming_opposite() {
        let a = BinaryVector::from_raw(vec![u64::MAX], 64);
        let b = BinaryVector::from_raw(vec![0], 64);

        assert_eq!(a.hamming_distance(&b), 64);
    }

    #[test]
    fn binary_similarity_range() {
        let a = BinaryVector::from_raw(vec![0b1111_0000], 8);
        let b = BinaryVector::from_raw(vec![0b1111_1111], 8);

        let sim = a.similarity(&b);
        assert!((0.0..=1.0).contains(&sim));
        // 4 bits differ out of 8, so similarity = 1 - 4/8 = 0.5
        assert!((sim - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn binary_compression_ratio() {
        let dim = 768;
        let dense_bytes = dim * std::mem::size_of::<f32>();
        let binary_bytes = dim.div_ceil(64) * std::mem::size_of::<u64>();

        let ratio = dense_bytes as f32 / binary_bytes as f32;
        // Should be 32x for perfectly aligned dimensions
        assert!(ratio >= 30.0, "Compression ratio too low: {ratio}x");
    }

    #[test]
    fn binary_to_dense() {
        let binary = BinaryVector::from_raw(vec![0b1010], 4);
        let dense = binary.to_dense();

        // bit 0: 0 -> -1, bit 1: 1 -> 1, bit 2: 0 -> -1, bit 3: 1 -> 1
        assert_eq!(dense, vec![-1.0, 1.0, -1.0, 1.0]);
    }

    #[test]
    fn binary_popcount() {
        let binary = BinaryVector::from_raw(vec![0b1111_0000_1111], 12);
        assert_eq!(binary.popcount(), 8);
    }

    #[test]
    fn binary_dot() {
        let a = BinaryVector::from_raw(vec![0b1111_0000], 8);
        let b = BinaryVector::from_raw(vec![0b1010_1010], 8);

        // a AND b = 0b1010_0000, popcount = 2
        assert_eq!(a.dot(&b), 2);
    }

    #[test]
    fn binary_jaccard() {
        let a = BinaryVector::from_raw(vec![0b1111_0000], 8);
        let b = BinaryVector::from_raw(vec![0b1100_1100], 8);

        // a OR b = 0b1111_1100, popcount = 6
        // a AND b = 0b1100_0000, popcount = 2
        // Jaccard = 2/6 = 0.333...
        let jaccard = a.jaccard_similarity(&b);
        assert!((jaccard - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn binary_jaccard_identical() {
        let a = BinaryVector::from_raw(vec![0b1111_0000], 8);
        let jaccard = a.jaccard_similarity(&a);
        assert!((jaccard - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn binary_jaccard_empty() {
        let a = BinaryVector::from_raw(vec![0], 64);
        let b = BinaryVector::from_raw(vec![0], 64);
        let jaccard = a.jaccard_similarity(&b);
        assert!((jaccard - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn binary_normalized_distance() {
        let a = BinaryVector::from_raw(vec![0b1111_0000], 8);
        let b = BinaryVector::from_raw(vec![0b0000_1111], 8);

        // All 8 bits differ
        let dist = a.normalized_distance(&b);
        assert!((dist - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn binary_large_dimension() {
        let dim = 1536;
        let vector: Vec<f32> = (0..dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let binary = BinaryVector::from_dense(&vector, BinaryThreshold::Sign);

        assert_eq!(binary.dimension(), dim);
        assert_eq!(binary.data.len(), dim.div_ceil(64));
        assert_eq!(binary.popcount(), (dim / 2) as u32);
    }

    #[test]
    fn binary_threshold_compute() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((BinaryThreshold::Sign.compute(&vector)).abs() < f32::EPSILON);
        assert!((BinaryThreshold::Mean.compute(&vector) - 3.0).abs() < f32::EPSILON);
        assert!((BinaryThreshold::Median.compute(&vector) - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn binary_threshold_empty() {
        let vector: Vec<f32> = vec![];

        assert!((BinaryThreshold::Sign.compute(&vector)).abs() < f32::EPSILON);
        assert!((BinaryThreshold::Mean.compute(&vector)).abs() < f32::EPSILON);
        assert!((BinaryThreshold::Median.compute(&vector)).abs() < f32::EPSILON);
    }

    #[test]
    fn binary_empty_similarity() {
        let a = BinaryVector {
            data: vec![],
            dimension: 0,
        };
        let b = BinaryVector {
            data: vec![],
            dimension: 0,
        };

        assert!((a.similarity(&b) - 1.0).abs() < f32::EPSILON);
        assert!(a.normalized_distance(&b).abs() < f32::EPSILON);
    }

    #[test]
    #[should_panic(expected = "doesn't match")]
    fn binary_from_raw_invalid() {
        let _ = BinaryVector::from_raw(vec![0, 0, 0], 64); // Should need only 1 word
    }

    #[test]
    fn binary_memory_bytes() {
        let binary = BinaryVector::from_raw(vec![0; 2], 128);
        assert_eq!(binary.memory_bytes(), 16); // 2 * 8 bytes
    }

    #[test]
    fn binary_as_raw() {
        let binary = BinaryVector::from_raw(vec![42, 84], 128);
        assert_eq!(binary.as_raw(), &[42, 84]);
    }
}
