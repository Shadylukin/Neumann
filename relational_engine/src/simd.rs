//! SIMD-accelerated bitmap filtering operations.

use wide::{f64x4, i64x4, CmpEq, CmpGt, CmpLt};

#[inline]
pub fn filter_lt_i64(values: &[i64], threshold: i64, result: &mut [u64]) {
    let chunks = values.len() / 4;
    let threshold_vec = i64x4::splat(threshold);

    for i in 0..chunks {
        let offset = i * 4;
        let v = i64x4::new([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let cmp = v.cmp_lt(threshold_vec);
        let mask_arr: [i64; 4] = cmp.into();

        for (j, &m) in mask_arr.iter().enumerate() {
            if m != 0 {
                let bit_pos = offset + j;
                result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let start = chunks * 4;
    for i in start..values.len() {
        if values[i] < threshold {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
}

#[inline]
pub fn filter_le_i64(values: &[i64], threshold: i64, result: &mut [u64]) {
    let chunks = values.len() / 4;
    let threshold_vec = i64x4::splat(threshold);

    for i in 0..chunks {
        let offset = i * 4;
        let v = i64x4::new([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let lt: [i64; 4] = v.cmp_lt(threshold_vec).into();
        let eq: [i64; 4] = v.cmp_eq(threshold_vec).into();

        for j in 0..4 {
            if lt[j] != 0 || eq[j] != 0 {
                let bit_pos = offset + j;
                result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let start = chunks * 4;
    for i in start..values.len() {
        if values[i] <= threshold {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
}

#[inline]
pub fn filter_gt_i64(values: &[i64], threshold: i64, result: &mut [u64]) {
    let chunks = values.len() / 4;
    let threshold_vec = i64x4::splat(threshold);

    for i in 0..chunks {
        let offset = i * 4;
        let v = i64x4::new([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let cmp = v.cmp_gt(threshold_vec);
        let mask_arr: [i64; 4] = cmp.into();

        for (j, &m) in mask_arr.iter().enumerate() {
            if m != 0 {
                let bit_pos = offset + j;
                result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let start = chunks * 4;
    for i in start..values.len() {
        if values[i] > threshold {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
}

#[inline]
pub fn filter_ge_i64(values: &[i64], threshold: i64, result: &mut [u64]) {
    let chunks = values.len() / 4;
    let threshold_vec = i64x4::splat(threshold);

    for i in 0..chunks {
        let offset = i * 4;
        let v = i64x4::new([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let gt: [i64; 4] = v.cmp_gt(threshold_vec).into();
        let eq: [i64; 4] = v.cmp_eq(threshold_vec).into();

        for j in 0..4 {
            if gt[j] != 0 || eq[j] != 0 {
                let bit_pos = offset + j;
                result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let start = chunks * 4;
    for i in start..values.len() {
        if values[i] >= threshold {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
}

#[inline]
pub fn filter_eq_i64(values: &[i64], target: i64, result: &mut [u64]) {
    let chunks = values.len() / 4;
    let target_vec = i64x4::splat(target);

    for i in 0..chunks {
        let offset = i * 4;
        let v = i64x4::new([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let cmp = v.cmp_eq(target_vec);
        let mask_arr: [i64; 4] = cmp.into();

        for (j, &m) in mask_arr.iter().enumerate() {
            if m != 0 {
                let bit_pos = offset + j;
                result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let start = chunks * 4;
    for i in start..values.len() {
        if values[i] == target {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
}

#[inline]
pub fn filter_ne_i64(values: &[i64], target: i64, result: &mut [u64]) {
    let chunks = values.len() / 4;
    let target_vec = i64x4::splat(target);

    for i in 0..chunks {
        let offset = i * 4;
        let v = i64x4::new([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let eq: [i64; 4] = v.cmp_eq(target_vec).into();

        for (j, &eq_val) in eq.iter().enumerate() {
            if eq_val == 0 {
                let bit_pos = offset + j;
                result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let start = chunks * 4;
    for i in start..values.len() {
        if values[i] != target {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
}

#[inline]
pub fn filter_lt_f64(values: &[f64], threshold: f64, result: &mut [u64]) {
    let chunks = values.len() / 4;
    let threshold_vec = f64x4::splat(threshold);

    for i in 0..chunks {
        let offset = i * 4;
        let v = f64x4::new([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let cmp = v.cmp_lt(threshold_vec);
        let mask = cmp.move_mask();

        for j in 0..4 {
            if (mask & (1 << j)) != 0 {
                let bit_pos = offset + j;
                result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let start = chunks * 4;
    for i in start..values.len() {
        if values[i] < threshold {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
}

#[inline]
pub fn filter_gt_f64(values: &[f64], threshold: f64, result: &mut [u64]) {
    let chunks = values.len() / 4;
    let threshold_vec = f64x4::splat(threshold);

    for i in 0..chunks {
        let offset = i * 4;
        let v = f64x4::new([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let cmp = v.cmp_gt(threshold_vec);
        let mask = cmp.move_mask();

        for j in 0..4 {
            if (mask & (1 << j)) != 0 {
                let bit_pos = offset + j;
                result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let start = chunks * 4;
    for i in start..values.len() {
        if values[i] > threshold {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
}

#[inline]
pub fn filter_eq_f64(values: &[f64], threshold: f64, result: &mut [u64]) {
    let chunks = values.len() / 4;
    let threshold_vec = f64x4::splat(threshold);

    for i in 0..chunks {
        let offset = i * 4;
        let v = f64x4::new([
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ]);
        let cmp = v.cmp_eq(threshold_vec);
        let mask = cmp.move_mask();

        for j in 0..4 {
            if (mask & (1 << j)) != 0 {
                let bit_pos = offset + j;
                result[bit_pos / 64] |= 1u64 << (bit_pos % 64);
            }
        }
    }

    let start = chunks * 4;
    for i in start..values.len() {
        if (values[i] - threshold).abs() < f64::EPSILON {
            result[i / 64] |= 1u64 << (i % 64);
        }
    }
}

#[inline]
pub fn bitmap_and(a: &[u64], b: &[u64], result: &mut [u64]) {
    for i in 0..a.len().min(b.len()).min(result.len()) {
        result[i] = a[i] & b[i];
    }
}

#[inline]
pub fn bitmap_or(a: &[u64], b: &[u64], result: &mut [u64]) {
    for i in 0..a.len().min(b.len()).min(result.len()) {
        result[i] = a[i] | b[i];
    }
}

#[inline]
pub fn popcount(bitmap: &[u64]) -> usize {
    bitmap.iter().map(|w| w.count_ones() as usize).sum()
}

pub fn selected_indices(bitmap: &[u64], max_count: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(max_count.min(1024));
    for (word_idx, &word) in bitmap.iter().enumerate() {
        let base = word_idx * 64;
        let mut w = word;
        while w != 0 {
            let bit = w.trailing_zeros() as usize;
            indices.push(base + bit);
            w &= w - 1;
        }
    }
    indices
}

#[inline]
pub const fn bitmap_words(n: usize) -> usize {
    n.div_ceil(64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_lt_i64() {
        let values = vec![1, 5, 3, 8, 2, 9, 4, 7];
        let mut bitmap = vec![0u64; 1];
        filter_lt_i64(&values, 5, &mut bitmap);
        // Positions 0,2,4,6 have values < 5: 1,3,2,4
        assert_eq!(bitmap[0] & 0xff, 0b01010101);
    }

    #[test]
    fn test_filter_eq_i64() {
        let values = vec![1, 5, 5, 8, 5, 9, 4, 7];
        let mut bitmap = vec![0u64; 1];
        filter_eq_i64(&values, 5, &mut bitmap);
        // Positions 1,2,4 have value == 5
        assert_eq!(bitmap[0] & 0xff, 0b00010110);
    }

    #[test]
    fn test_filter_gt_i64() {
        let values = vec![1, 5, 3, 8, 2, 9, 4, 7];
        let mut bitmap = vec![0u64; 1];
        filter_gt_i64(&values, 5, &mut bitmap);
        // Positions 3,5,7 have values > 5: 8,9,7
        assert_eq!(bitmap[0] & 0xff, 0b10101000);
    }

    #[test]
    fn test_filter_handles_remainder() {
        let values = vec![1, 2, 3, 4, 5]; // Not divisible by 4
        let mut bitmap = vec![0u64; 1];
        filter_lt_i64(&values, 4, &mut bitmap);
        // Positions 0,1,2 have values < 4
        assert_eq!(bitmap[0] & 0xff, 0b00000111);
    }

    #[test]
    fn test_filter_lt_remainder_match() {
        // Ensure remainder elements that match are included
        let values = vec![10, 10, 10, 10, 1]; // 5 elements, position 4 matches
        let mut bitmap = vec![0u64; 1];
        filter_lt_i64(&values, 5, &mut bitmap);
        // Only position 4 (value 1) is < 5
        assert_eq!(bitmap[0], 0b00010000);
    }

    #[test]
    fn test_filter_le_remainder_match() {
        let values = vec![10, 10, 10, 10, 5]; // 5 elements, position 4 matches <=
        let mut bitmap = vec![0u64; 1];
        filter_le_i64(&values, 5, &mut bitmap);
        assert_eq!(bitmap[0], 0b00010000);
    }

    #[test]
    fn test_filter_gt_remainder_match() {
        let values = vec![1, 1, 1, 1, 10]; // 5 elements, position 4 matches >
        let mut bitmap = vec![0u64; 1];
        filter_gt_i64(&values, 5, &mut bitmap);
        assert_eq!(bitmap[0], 0b00010000);
    }

    #[test]
    fn test_filter_ge_remainder_match() {
        let values = vec![1, 1, 1, 1, 5]; // 5 elements, position 4 matches >=
        let mut bitmap = vec![0u64; 1];
        filter_ge_i64(&values, 5, &mut bitmap);
        assert_eq!(bitmap[0], 0b00010000);
    }

    #[test]
    fn test_bitmap_and() {
        let a = [0b1111_0000u64];
        let b = [0b1010_1010u64];
        let mut result = [0u64];
        bitmap_and(&a, &b, &mut result);
        assert_eq!(result[0], 0b1010_0000);
    }

    #[test]
    fn test_bitmap_or() {
        let a = [0b1111_0000u64];
        let b = [0b0000_1111u64];
        let mut result = [0u64];
        bitmap_or(&a, &b, &mut result);
        assert_eq!(result[0], 0b1111_1111);
    }

    #[test]
    fn test_selected_indices() {
        let bitmap = [0b00010110u64]; // bits 1,2,4 set
        let indices = selected_indices(&bitmap, 10);
        assert_eq!(indices, vec![1, 2, 4]);
    }

    #[test]
    fn test_popcount() {
        let bitmap = [0b11110000u64, 0b00001111u64];
        assert_eq!(popcount(&bitmap), 8);
    }

    #[test]
    fn test_filter_lt_f64_remainder_match() {
        let values = vec![10.0, 10.0, 10.0, 10.0, 1.0]; // 5 elements
        let mut bitmap = vec![0u64; 1];
        filter_lt_f64(&values, 5.0, &mut bitmap);
        assert_eq!(bitmap[0], 0b00010000);
    }

    #[test]
    fn test_filter_gt_f64_remainder_match() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 10.0]; // 5 elements
        let mut bitmap = vec![0u64; 1];
        filter_gt_f64(&values, 5.0, &mut bitmap);
        assert_eq!(bitmap[0], 0b00010000);
    }

    #[test]
    fn test_filter_eq_f64_remainder_match() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 5.0]; // 5 elements
        let mut bitmap = vec![0u64; 1];
        filter_eq_f64(&values, 5.0, &mut bitmap);
        assert_eq!(bitmap[0], 0b00010000);
    }
}
