// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

//! Fuzz target for TTVector serialization roundtrip.
//!
//! Tests:
//! - Serialize/deserialize roundtrip preserves all data
//! - Reconstruction is identical after deserialization
//! - Core data integrity is maintained

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_compress::{tt_decompose, tt_reconstruct, TTConfig, TTVector};

#[derive(Arbitrary, Debug)]
struct TtInput {
    values: Vec<f32>,
}

fuzz_target!(|input: TtInput| {
    // Filter out NaN and infinite values, require minimum size
    let floats: Vec<f32> = input
        .values
        .into_iter()
        .filter(|f| f.is_finite())
        .take(256)
        .collect();

    if floats.len() < 16 {
        return;
    }

    // Round down to nearest power of 2 for clean factorization
    let len = floats.len();
    let power = (len as f64).log2().floor() as u32;
    let target_len = 2usize.pow(power);
    let floats: Vec<f32> = floats.into_iter().take(target_len).collect();

    // Create config
    let config = match TTConfig::for_dim(floats.len()) {
        Ok(c) => c,
        Err(_) => return,
    };

    // Decompose
    let tt = match tt_decompose(&floats, &config) {
        Ok(tt) => tt,
        Err(_) => return,
    };

    // Serialize with bincode
    let bytes = match bitcode::serialize(&tt) {
        Ok(b) => b,
        Err(_) => return, // Serialization failure is valid for some inputs
    };

    // Deserialize
    let tt2: TTVector = match bitcode::deserialize(&bytes) {
        Ok(tt) => tt,
        Err(e) => panic!(
            "deserialization failed after successful serialization: {}",
            e
        ),
    };

    // Property: core count should match
    assert_eq!(
        tt.cores.len(),
        tt2.cores.len(),
        "core count mismatch after roundtrip"
    );

    // Property: shape should match
    assert_eq!(tt.shape, tt2.shape, "shape mismatch after roundtrip");

    // Property: ranks should match
    assert_eq!(tt.ranks, tt2.ranks, "ranks mismatch after roundtrip");

    // Property: original_dim should match
    assert_eq!(
        tt.original_dim, tt2.original_dim,
        "original_dim mismatch after roundtrip"
    );

    // Property: core data should match exactly
    for (i, (c1, c2)) in tt.cores.iter().zip(tt2.cores.iter()).enumerate() {
        assert_eq!(
            c1.shape, c2.shape,
            "core {} shape mismatch after roundtrip",
            i
        );
        assert_eq!(
            c1.data.len(),
            c2.data.len(),
            "core {} data length mismatch after roundtrip",
            i
        );
        for (j, (v1, v2)) in c1.data.iter().zip(c2.data.iter()).enumerate() {
            // Use bit-exact comparison for serialization roundtrip
            assert_eq!(
                v1.to_bits(),
                v2.to_bits(),
                "core {} data[{}] mismatch: {} vs {}",
                i,
                j,
                v1,
                v2
            );
        }
    }

    // Property: reconstruction should be identical
    let r1 = tt_reconstruct(&tt);
    let r2 = tt_reconstruct(&tt2);
    assert_eq!(
        r1.len(),
        r2.len(),
        "reconstruction length mismatch after roundtrip"
    );
    for (i, (v1, v2)) in r1.iter().zip(r2.iter()).enumerate() {
        assert_eq!(
            v1.to_bits(),
            v2.to_bits(),
            "reconstruction[{}] mismatch: {} vs {}",
            i,
            v1,
            v2
        );
    }
});
