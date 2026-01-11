#![no_main]

//! Fuzz target for TT metric operations.
//!
//! Tests that TT metric operations don't panic and produce reasonable results
//! for a variety of inputs. Due to numerical precision limits, we focus on
//! verifying operations complete without panicking rather than strict property checks.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_compress::{
    tt_cosine_similarity, tt_decompose, tt_dot_product, tt_euclidean_distance, tt_norm, TTConfig,
};

#[derive(Arbitrary, Debug)]
struct MetricsInput {
    values_a: Vec<f32>,
    values_b: Vec<f32>,
}

/// Filter to finite, reasonable values and limit size.
fn filter_reasonable(values: &[f32], max_len: usize) -> Vec<f32> {
    values
        .iter()
        .filter(|f| f.is_finite() && f.abs() < 1e10)
        .take(max_len)
        .copied()
        .collect()
}

/// Round down to nearest power of 2 for clean factorization.
fn round_to_power_of_2(len: usize) -> usize {
    if len < 16 {
        return 0;
    }
    let power = (len as f64).log2().floor() as u32;
    2usize.pow(power)
}

fuzz_target!(|input: MetricsInput| {
    // Filter and normalize inputs to reasonable values
    let a = filter_reasonable(&input.values_a, 256);
    let target_len = round_to_power_of_2(a.len());
    if target_len < 16 {
        return;
    }
    let a: Vec<f32> = a.into_iter().take(target_len).collect();

    // Make b the same length as a, skip if no valid values
    let b_filtered = filter_reasonable(&input.values_b, 256);
    if b_filtered.is_empty() {
        return;
    }
    let b: Vec<f32> = b_filtered.into_iter().cycle().take(target_len).collect();

    // Create config
    let config = match TTConfig::for_dim(target_len) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Decompose both vectors
    let tt_a = match tt_decompose(&a, &config) {
        Ok(tt) => tt,
        Err(_) => return,
    };
    let tt_b = match tt_decompose(&b, &config) {
        Ok(tt) => tt,
        Err(_) => return,
    };

    // Test: norm should not panic and should be non-negative if finite
    let norm_a = tt_norm(&tt_a);
    if norm_a.is_finite() {
        // Norm can be NaN with degenerate inputs, but if finite should be >= 0
        // Due to floating point, allow small negative values
        assert!(
            norm_a >= -1e-6,
            "norm_a should be approximately >= 0, got {}",
            norm_a
        );
    }

    let norm_b = tt_norm(&tt_b);
    if norm_b.is_finite() {
        assert!(
            norm_b >= -1e-6,
            "norm_b should be approximately >= 0, got {}",
            norm_b
        );
    }

    // Test: cosine similarity should not panic
    let _ = tt_cosine_similarity(&tt_a, &tt_b);

    // Test: self-similarity should not panic
    let _ = tt_cosine_similarity(&tt_a, &tt_a);

    // Test: distance should not panic
    let _ = tt_euclidean_distance(&tt_a, &tt_b);

    // Test: self-distance should not panic
    let _ = tt_euclidean_distance(&tt_a, &tt_a);

    // Test: dot product should not panic
    let _ = tt_dot_product(&tt_a, &tt_a);
    let _ = tt_dot_product(&tt_a, &tt_b);
});
