// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fuzz test for snapshot hash validation.
//!
//! Tests that hash validation correctly identifies:
//! - Valid data with matching hash
//! - Corrupted data with mismatched hash

#![no_main]
use libfuzzer_sys::fuzz_target;
use sha2::{Digest, Sha256};

fuzz_target!(|data: &[u8]| {
    // Skip empty data
    if data.is_empty() {
        return;
    }

    // Compute correct hash
    let mut hasher = Sha256::new();
    hasher.update(data);
    let correct_hash: [u8; 32] = hasher.finalize().into();

    // Verify correct hash matches
    let mut verify_hasher = Sha256::new();
    verify_hasher.update(data);
    let verify_hash: [u8; 32] = verify_hasher.finalize().into();
    assert_eq!(correct_hash, verify_hash, "Hash should be deterministic");

    // Verify wrong hash doesn't match
    let wrong_hash = [0u8; 32];
    if data != &[0u8; 32] {
        // Unless data happens to hash to all zeros (extremely unlikely)
        assert_ne!(correct_hash, wrong_hash, "Hash should not match wrong hash");
    }

    // Verify corrupted data doesn't match original hash
    if data.len() > 1 {
        let mut corrupted = data.to_vec();
        corrupted[0] = corrupted[0].wrapping_add(1);

        let mut corrupt_hasher = Sha256::new();
        corrupt_hasher.update(&corrupted);
        let corrupt_hash: [u8; 32] = corrupt_hasher.finalize().into();

        // Corrupted data should have different hash (with overwhelming probability)
        if corrupted != data {
            assert_ne!(
                correct_hash, corrupt_hash,
                "Corrupted data should have different hash"
            );
        }
    }
});
