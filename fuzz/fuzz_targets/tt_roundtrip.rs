// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_compress::{tt_decompose, tt_reconstruct, TTConfig};

#[derive(Arbitrary, Debug)]
struct TtInput {
    // Keep dimensions reasonable for fuzzing
    values: Vec<f32>,
}

fuzz_target!(|input: TtInput| {
    // Filter out NaN and infinite values, require minimum size
    let floats: Vec<f32> = input
        .values
        .into_iter()
        .filter(|f| f.is_finite())
        .take(256) // Limit size for performance
        .collect();

    // Need at least 16 elements to form a reasonable tensor
    if floats.len() < 16 {
        return;
    }

    // Round down to nearest power of 2 for clean factorization
    let len = floats.len();
    let power = (len as f64).log2().floor() as u32;
    let target_len = 2usize.pow(power);
    let floats: Vec<f32> = floats.into_iter().take(target_len).collect();

    // Create config for this dimension
    let config = match TTConfig::for_dim(floats.len()) {
        Ok(c) => c,
        Err(_) => return, // Skip invalid dimensions
    };

    // Attempt decomposition
    if let Ok(tt) = tt_decompose(&floats, &config) {
        // Reconstruction should always succeed
        let reconstructed = tt_reconstruct(&tt);
        assert_eq!(
            floats.len(),
            reconstructed.len(),
            "Reconstructed length mismatch"
        );

        // Verify approximate equality (TT is lossy)
        let max_error: f32 = floats
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        let norm: f32 = floats.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            let relative_error = max_error / norm;
            // TT should maintain reasonable accuracy
            assert!(
                relative_error < 1.0,
                "Excessive reconstruction error: {}",
                relative_error
            );
        }
    }
});
