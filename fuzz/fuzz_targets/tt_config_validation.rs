// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_compress::{TTConfig, TTError};

#[derive(Arbitrary, Debug)]
struct ConfigInput {
    // Limit shape to avoid OOM (max 8 dimensions)
    shape: Vec<u8>,
    max_rank: u16,
    tolerance_bits: u32,
}

fuzz_target!(|input: ConfigInput| {
    // Convert arbitrary tolerance bits to float
    let tolerance = f32::from_bits(input.tolerance_bits);

    // Limit shape to reasonable size
    let shape: Vec<usize> = input.shape.iter().take(8).map(|&s| s as usize).collect();

    // Create config
    let config = TTConfig {
        shape,
        max_rank: input.max_rank as usize,
        tolerance,
    };

    // Validate should never panic
    let result = config.validate();

    // Verify error conditions
    match result {
        Ok(()) => {
            // Valid configs should have non-empty shape
            assert!(!config.shape.is_empty());
            // Valid configs should have no zeros in shape
            assert!(config.shape.iter().all(|&s| s > 0));
            // Valid configs should have valid tolerance
            assert!(config.tolerance > 0.0);
            assert!(config.tolerance <= 1.0);
            // Valid configs should have max_rank >= 1
            assert!(config.max_rank >= 1);
        },
        Err(TTError::InvalidShape(_)) => {
            // Shape error: empty or contains zero
            assert!(config.shape.is_empty() || config.shape.iter().any(|&s| s == 0));
        },
        Err(TTError::InvalidRank) => {
            // Rank error
            assert!(config.max_rank < 1);
        },
        Err(TTError::InvalidTolerance(_)) => {
            // Tolerance must be 0 < tol <= 1 and finite
            assert!(
                config.tolerance <= 0.0 || config.tolerance > 1.0 || !config.tolerance.is_finite()
            );
        },
        Err(_) => {
            // Other errors are also valid responses
        },
    }
});
