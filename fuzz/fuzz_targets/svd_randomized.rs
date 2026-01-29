// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_compress::{svd_truncated, Matrix};

#[derive(Arbitrary, Debug)]
struct SvdInput {
    rows: u8,
    cols: u8,
    values: Vec<f32>,
    rank: u8,
}

fuzz_target!(|input: SvdInput| {
    // Limit dimensions to reasonable range for fuzzing
    let rows = (input.rows as usize).clamp(1, 64);
    let cols = (input.cols as usize).clamp(1, 64);
    let rank = (input.rank as usize).clamp(1, rows.min(cols));

    // Filter non-finite values and pad/truncate to required size
    let values: Vec<f32> = input
        .values
        .into_iter()
        .filter(|f| f.is_finite())
        .take(rows * cols)
        .collect();

    // Need exactly rows * cols elements
    if values.len() < rows * cols {
        return;
    }

    let matrix = match Matrix::new(values, rows, cols) {
        Ok(m) => m,
        Err(_) => return,
    };

    if let Ok(svd) = svd_truncated(&matrix, rank, 1e-4) {
        // Verify rank constraint
        assert!(
            svd.rank <= rank,
            "SVD rank {} exceeds requested {}",
            svd.rank,
            rank
        );

        // Verify output shapes
        assert_eq!(svd.u.rows, rows, "U rows mismatch");
        assert_eq!(svd.u.cols, svd.rank, "U cols mismatch");
        assert_eq!(svd.vt.rows, svd.rank, "Vt rows mismatch");
        assert_eq!(svd.vt.cols, cols, "Vt cols mismatch");
        assert_eq!(svd.s.len(), svd.rank, "S length mismatch");

        // Verify singular values are non-negative
        for (i, &s) in svd.s.iter().enumerate() {
            assert!(s >= -1e-6, "Singular value {} is negative: {}", i, s);
        }

        // Verify singular values are in descending order
        for i in 1..svd.s.len() {
            assert!(
                svd.s[i - 1] >= svd.s[i] - 1e-5,
                "Singular values not ordered: {} < {}",
                svd.s[i - 1],
                svd.s[i]
            );
        }

        // Verify all outputs are finite
        assert!(
            svd.u.data.iter().all(|x| x.is_finite()),
            "U contains non-finite values"
        );
        assert!(
            svd.vt.data.iter().all(|x| x.is_finite()),
            "Vt contains non-finite values"
        );
        assert!(
            svd.s.iter().all(|x| x.is_finite()),
            "S contains non-finite values"
        );
    }
});
