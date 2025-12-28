#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::SparseVector;

#[derive(Arbitrary, Debug)]
struct SparseVectorInput {
    // Limit dimension to avoid OOM (u16 max = 65535)
    dimension: u16,
    dense: Vec<f32>,
}

fuzz_target!(|input: SparseVectorInput| {
    let dimension = input.dimension as usize;
    if dimension == 0 {
        return;
    }

    // Truncate or pad dense to match dimension
    let dense: Vec<f32> = input
        .dense
        .iter()
        .take(dimension)
        .copied()
        .chain(std::iter::repeat(0.0))
        .take(dimension)
        .collect();

    // Create sparse vector and convert back to dense
    let sparse = SparseVector::from_dense(&dense);
    let recovered = sparse.to_dense();

    // Verify roundtrip
    assert_eq!(recovered.len(), dimension, "Dimension mismatch");

    for (i, (a, b)) in dense.iter().zip(recovered.iter()).enumerate() {
        // Skip NaN comparisons (NaN != NaN)
        if a.is_nan() || b.is_nan() {
            continue;
        }
        assert!(
            (a - b).abs() < 1e-6,
            "Sparse vector roundtrip failed at index {}: {} != {}",
            i,
            a,
            b
        );
    }
});
