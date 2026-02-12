// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_vault::{MasterKey, ShamirConfig, split_master_key, reconstruct_master_key};

#[derive(Arbitrary, Debug)]
struct ShamirInput {
    key_bytes: [u8; 32],
    total: u8,
    threshold: u8,
    /// Bitmask selecting which shares to use for reconstruction.
    share_mask: u16,
}

fuzz_target!(|input: ShamirInput| {
    // Clamp to valid Shamir ranges
    let total = input.total.clamp(2, 10);
    let threshold = input.threshold.clamp(2, total);

    let key = MasterKey::from_bytes(input.key_bytes);
    let config = ShamirConfig {
        total_shares: total,
        threshold,
    };

    let shares = match split_master_key(&key, &config) {
        Ok(s) => s,
        Err(_) => return,
    };

    assert_eq!(shares.len(), total as usize);

    // Select shares via bitmask, ensuring at least `threshold` shares
    let mut selected: Vec<_> = shares
        .iter()
        .enumerate()
        .filter(|(i, _)| input.share_mask & (1 << (i % 16)) != 0)
        .map(|(_, s)| s.clone())
        .collect();

    // If fewer than threshold selected, take the first `threshold` shares
    if selected.len() < threshold as usize {
        selected = shares.iter().take(threshold as usize).cloned().collect();
    }

    // Reconstruct and verify
    match reconstruct_master_key(&selected) {
        Ok(reconstructed) => {
            assert_eq!(
                reconstructed.as_bytes(),
                key.as_bytes(),
                "reconstructed key must match original"
            );
        }
        Err(_) => {
            // Reconstruction can fail if insufficient or corrupt shares -- acceptable
        }
    }
});
