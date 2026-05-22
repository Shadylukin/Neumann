// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target for TT batch decomposition.
//!
//! Tests:
//! - Batch decomposition returns correct number of results
//! - Each result can be reconstructed
//! - Results are consistent with individual decomposition

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_compress::{tt_decompose, tt_decompose_batch, tt_reconstruct, TTConfig};

#[derive(Arbitrary, Debug)]
struct BatchInput {
    vectors: Vec<Vec<f32>>,
}

/// Filter to finite values and normalize to target dimension.
fn normalize_vec(values: Vec<f32>, target_dim: usize) -> Vec<f32> {
    values
        .into_iter()
        .filter(|f| f.is_finite())
        .cycle()
        .take(target_dim)
        .collect()
}

fuzz_target!(|input: BatchInput| {
    // Use fixed dimension for batch processing
    let dim = 64;

    // Limit batch size to avoid OOM
    let max_batch = 8;

    // Filter and normalize vectors
    let vecs: Vec<Vec<f32>> = input
        .vectors
        .into_iter()
        .take(max_batch)
        .filter(|v| !v.is_empty())
        .map(|v| normalize_vec(v, dim))
        .collect();

    if vecs.is_empty() {
        return;
    }

    // Create config
    let config = match TTConfig::for_dim(dim) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Convert to slices for batch API
    let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

    // Batch decompose
    let batch_result = tt_decompose_batch(&refs, &config);
    let tts = match batch_result {
        Ok(tts) => tts,
        Err(_) => return,
    };

    // Property: batch size matches input size
    assert_eq!(
        tts.len(),
        vecs.len(),
        "batch result size should match input size"
    );

    // Property: each TT should reconstruct to correct dimension
    for (i, tt) in tts.iter().enumerate() {
        let recon = tt_reconstruct(tt);
        assert_eq!(
            recon.len(),
            dim,
            "reconstructed vector {} should have length {}",
            i,
            dim
        );

        // Note: Reconstruction may have non-finite values with extreme inputs
        // This is expected behavior - we just verify the length is correct
    }

    // Property: batch results should match individual decomposition
    for (i, (tt_batch, vec)) in tts.iter().zip(&vecs).enumerate() {
        let tt_individual = match tt_decompose(vec, &config) {
            Ok(tt) => tt,
            Err(_) => continue,
        };

        // Shapes should match
        assert_eq!(
            tt_batch.shape, tt_individual.shape,
            "batch {} shape should match individual",
            i
        );
        assert_eq!(
            tt_batch.original_dim, tt_individual.original_dim,
            "batch {} original_dim should match individual",
            i
        );

        // Number of cores should match
        assert_eq!(
            tt_batch.cores.len(),
            tt_individual.cores.len(),
            "batch {} core count should match individual",
            i
        );
    }
});
