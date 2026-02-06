// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::SparseVector;

#[derive(Arbitrary, Debug)]
struct SparseVectorInput {
    dimension: u16,
    positions: Vec<u16>,
    values: Vec<f32>,
}

fuzz_target!(|input: SparseVectorInput| {
    let dim = (input.dimension as usize).max(1).min(10000);

    // Constrain positions to valid range and limit count
    let positions: Vec<u32> = input
        .positions
        .iter()
        .take(100)
        .map(|p| (*p as u32) % (dim as u32))
        .collect();

    // Filter values to be finite and match positions length
    let values: Vec<f32> = input
        .values
        .iter()
        .take(positions.len())
        .filter(|v| v.is_finite())
        .copied()
        .collect();

    if values.is_empty() {
        return;
    }

    // Truncate positions to match filtered values
    let positions: Vec<u32> = positions.into_iter().take(values.len()).collect();

    // Try to create sparse vector from parts
    if let Ok(sv) = SparseVector::try_from_parts(dim, positions.clone(), values.clone()) {
        let dense = sv.to_dense();
        assert_eq!(dense.len(), dim, "Dimension mismatch after roundtrip");

        // Verify non-zero positions have correct values
        for (&pos, &val) in positions.iter().zip(values.iter()) {
            if pos < dim as u32 {
                let dense_val = dense[pos as usize];
                // Allow small numerical differences
                assert!(
                    (dense_val - val).abs() < 1e-5 || dense_val.is_nan() && val.is_nan(),
                    "Value mismatch at position {}: expected {}, got {}",
                    pos,
                    val,
                    dense_val
                );
            }
        }
    }
});
