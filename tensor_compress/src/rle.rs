// SPDX-License-Identifier: MIT OR Apache-2.0
//! Run-length encoding for repeated values.

use serde::{Deserialize, Serialize};

/// Run-length encoded data: pairs of (value, count).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RleEncoded<T: Eq> {
    pub values: Vec<T>,
    pub run_lengths: Vec<u32>,
}

impl<T: Eq> RleEncoded<T> {
    #[must_use]
    pub fn len(&self) -> usize {
        self.run_lengths.iter().map(|&r| r as usize).sum()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    #[must_use]
    pub const fn runs(&self) -> usize {
        self.values.len()
    }
}

/// RLE-encode a slice of values.
#[must_use]
pub fn rle_encode<T: Eq + Clone>(data: &[T]) -> RleEncoded<T> {
    if data.is_empty() {
        return RleEncoded {
            values: Vec::new(),
            run_lengths: Vec::new(),
        };
    }

    let mut values = Vec::new();
    let mut run_lengths = Vec::new();

    let mut current = &data[0];
    let mut count = 1u32;

    for item in &data[1..] {
        if item == current {
            count += 1;
        } else {
            values.push(current.clone());
            run_lengths.push(count);
            current = item;
            count = 1;
        }
    }

    values.push(current.clone());
    run_lengths.push(count);

    RleEncoded {
        values,
        run_lengths,
    }
}

/// Decode RLE back to original data.
#[must_use]
pub fn rle_decode<T: Clone + Eq>(encoded: &RleEncoded<T>) -> Vec<T> {
    let total_len = encoded.len();
    let mut result = Vec::with_capacity(total_len);

    for (value, &count) in encoded.values.iter().zip(&encoded.run_lengths) {
        for _ in 0..count {
            result.push(value.clone());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rle_encode_empty() {
        let encoded: RleEncoded<i32> = rle_encode(&[]);
        assert!(encoded.is_empty());
        assert_eq!(encoded.len(), 0);
        assert_eq!(encoded.runs(), 0);
    }

    #[test]
    fn test_rle_encode_single() {
        let encoded = rle_encode(&[42]);
        assert_eq!(encoded.values, vec![42]);
        assert_eq!(encoded.run_lengths, vec![1]);
        assert_eq!(encoded.len(), 1);
    }

    #[test]
    fn test_rle_encode_all_same() {
        let data = vec![5, 5, 5, 5, 5];
        let encoded = rle_encode(&data);
        assert_eq!(encoded.values, vec![5]);
        assert_eq!(encoded.run_lengths, vec![5]);
        assert_eq!(encoded.runs(), 1);
    }

    #[test]
    fn test_rle_encode_all_different() {
        let data = vec![1, 2, 3, 4, 5];
        let encoded = rle_encode(&data);
        assert_eq!(encoded.values, vec![1, 2, 3, 4, 5]);
        assert_eq!(encoded.run_lengths, vec![1, 1, 1, 1, 1]);
        assert_eq!(encoded.runs(), 5);
    }

    #[test]
    fn test_rle_encode_mixed() {
        let data = vec![1, 1, 2, 2, 2, 3, 1, 1, 1, 1];
        let encoded = rle_encode(&data);
        assert_eq!(encoded.values, vec![1, 2, 3, 1]);
        assert_eq!(encoded.run_lengths, vec![2, 3, 1, 4]);
        assert_eq!(encoded.len(), 10);
    }

    #[test]
    fn test_rle_roundtrip() {
        let original = vec![1, 1, 1, 2, 2, 3, 3, 3, 3, 1];
        let encoded = rle_encode(&original);
        let decoded = rle_decode(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_rle_roundtrip_strings() {
        let original = vec!["active", "active", "pending", "done", "done", "done"];
        let encoded = rle_encode(&original);
        let decoded = rle_decode(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_rle_decode_empty() {
        let encoded: RleEncoded<i32> = RleEncoded {
            values: Vec::new(),
            run_lengths: Vec::new(),
        };
        assert!(rle_decode(&encoded).is_empty());
    }

    #[test]
    fn test_rle_compression_best_case() {
        // 1000 identical values
        let data: Vec<i32> = vec![42; 1000];
        let encoded = rle_encode(&data);

        assert_eq!(encoded.runs(), 1);
        assert_eq!(encoded.values, vec![42]);
        assert_eq!(encoded.run_lengths, vec![1000]);

        // Original: 1000 * 4 = 4000 bytes
        // Encoded: 1 * 4 + 1 * 4 = 8 bytes
        // Ratio: 500x
    }

    #[test]
    fn test_rle_compression_status_column() {
        // Simulates a status column with 3 values
        let mut data = Vec::with_capacity(10000);
        for _ in 0..100 {
            data.extend(vec!["pending"; 30]);
            data.extend(vec!["active"; 50]);
            data.extend(vec!["done"; 20]);
        }

        let encoded = rle_encode(&data);
        assert_eq!(encoded.len(), 10000);
        assert_eq!(encoded.runs(), 300); // 100 iterations * 3 transitions

        let decoded = rle_decode(&encoded);
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_rle_serialize() {
        let original = vec![1, 1, 2, 2, 2];
        let encoded = rle_encode(&original);

        let bytes = bitcode::serialize(&encoded).unwrap();
        let decoded: RleEncoded<i32> = bitcode::deserialize(&bytes).unwrap();

        assert_eq!(encoded, decoded);
    }

    #[test]
    fn test_rle_large_runs() {
        // Test with run lengths that exceed u16
        let data: Vec<i32> = vec![1; 100_000];
        let encoded = rle_encode(&data);
        let decoded = rle_decode(&encoded);
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_rle_i64_values() {
        let data = vec![i64::MIN, i64::MIN, 0, 0, 0, i64::MAX, i64::MAX];
        let encoded = rle_encode(&data);
        let decoded = rle_decode(&encoded);
        assert_eq!(data, decoded);
    }
}
