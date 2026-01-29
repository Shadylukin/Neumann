// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::DeltaVector;

#[derive(Arbitrary, Debug)]
struct DeltaInput {
    dimension: u16,
    archetype: Vec<f32>,
    dense: Vec<f32>,
    threshold: f32,
}

fuzz_target!(|input: DeltaInput| {
    let dim = (input.dimension as usize).clamp(1, 1000);

    // Build archetype with finite values, padded to dimension
    let mut archetype: Vec<f32> = input
        .archetype
        .iter()
        .take(dim)
        .filter(|v| v.is_finite())
        .copied()
        .collect();
    archetype.resize(dim, 0.0);

    // Build dense with finite values, padded to dimension
    let mut dense: Vec<f32> = input
        .dense
        .iter()
        .take(dim)
        .filter(|v| v.is_finite())
        .copied()
        .collect();
    dense.resize(dim, 0.0);

    // Clamp threshold to valid range
    let threshold = if input.threshold.is_finite() {
        input.threshold.abs().clamp(0.0, 1.0)
    } else {
        0.1
    };

    // Try to create delta vector
    if let Ok(dv) = DeltaVector::try_from_dense_with_reference(&dense, &archetype, 0, threshold) {
        let reconstructed = dv.to_dense(&archetype);
        assert_eq!(
            reconstructed.len(),
            dim,
            "Reconstructed dimension mismatch"
        );

        // With threshold, the reconstructed values should be close
        // The threshold allows some elements to be dropped if delta is small
        for (i, (&orig, &recon)) in dense.iter().zip(reconstructed.iter()).enumerate() {
            let diff = (orig - recon).abs();
            // Delta vector can drop small differences, so allow threshold * archetype magnitude
            let archetype_val = archetype[i].abs().max(1.0);
            assert!(
                diff <= threshold * archetype_val + 1e-5,
                "Value at {} differs too much: orig={}, recon={}, diff={}, threshold={}",
                i,
                orig,
                recon,
                diff,
                threshold
            );
        }
    }
});
