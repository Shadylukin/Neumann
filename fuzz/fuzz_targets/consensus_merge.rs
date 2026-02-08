// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;
use tensor_chain::{ConsensusConfig, ConsensusManager, DeltaVector};

#[derive(Arbitrary, Debug)]
struct ConsensusMergeInput {
    // Thresholds (as u8 to convert to f32)
    orthogonal_threshold: u8,
    conflict_threshold: u8,
    identical_threshold: u8,
    // Delta vectors
    deltas: Vec<DeltaVectorInput>,
}

#[derive(Arbitrary, Debug)]
struct DeltaVectorInput {
    // Vector components (will be normalized to dimension)
    components: Vec<f32>,
    // Affected keys
    keys: Vec<String>,
    // Transaction ID
    tx_id: u64,
}

fuzz_target!(|input: ConsensusMergeInput| {
    // Constrain thresholds to valid ranges
    let orthogonal_threshold = (input.orthogonal_threshold as f32 / 255.0).clamp(0.01, 0.5);
    let conflict_threshold = (input.conflict_threshold as f32 / 255.0).clamp(0.5, 0.99);
    let identical_threshold = (input.identical_threshold as f32 / 255.0).clamp(0.9, 0.999);

    let config = ConsensusConfig {
        orthogonal_threshold,
        conflict_threshold,
        identical_threshold,
        opposite_threshold: -0.9,
        allow_key_overlap_merge: false,
        structural_conflict_threshold: 0.7,
        sparsity_threshold: 0.5,
    };

    let consensus = ConsensusManager::new(config);

    // Convert inputs to DeltaVectors with fixed dimension
    let dimension = 32;
    let deltas: Vec<DeltaVector> = input
        .deltas
        .iter()
        .take(10) // Limit number of deltas
        .filter_map(|d| {
            // Need at least one component
            if d.components.is_empty() {
                return None;
            }

            // Create vector of fixed dimension
            let vector: Vec<f32> = (0..dimension)
                .map(|i| {
                    let val = d.components.get(i % d.components.len()).copied().unwrap_or(0.0);
                    if val.is_finite() {
                        val
                    } else {
                        0.0
                    }
                })
                .collect();

            // Skip zero vectors
            if vector.iter().all(|x| *x == 0.0) {
                return None;
            }

            let keys: HashSet<String> = d.keys.iter().take(5).cloned().collect();
            Some(DeltaVector::new(&vector, keys, d.tx_id))
        })
        .collect();

    if deltas.is_empty() {
        return;
    }

    // Property 1: Conflict detection is symmetric
    if deltas.len() >= 2 {
        let d1 = &deltas[0];
        let d2 = &deltas[1];

        let conflict12 = consensus.detect_conflict(d1, d2);
        let conflict21 = consensus.detect_conflict(d2, d1);

        // Conflict class should be the same (symmetric)
        assert_eq!(
            std::mem::discriminant(&conflict12.class),
            std::mem::discriminant(&conflict21.class),
            "Conflict detection should be symmetric"
        );
    }

    // Property 2: Cosine similarity is symmetric and bounded
    if deltas.len() >= 2 {
        let d1 = &deltas[0];
        let d2 = &deltas[1];

        let sim12 = d1.cosine_similarity(d2);
        let sim21 = d2.cosine_similarity(d1);

        assert!(
            (sim12 - sim21).abs() < 0.0001,
            "Cosine similarity should be symmetric"
        );
        assert!(
            sim12 >= -1.0 && sim12 <= 1.0,
            "Cosine similarity should be in [-1, 1]"
        );
    }

    // Property 3: Self-similarity is 1.0 (skip degenerate vectors where
    // magnitude is near-zero, subnormal, or overflows in f32)
    for delta in deltas.iter().take(5) {
        let mag = delta.delta.magnitude();
        if mag > 1e-10 {
            let self_sim = delta.cosine_similarity(delta);
            assert!(
                (self_sim - 1.0).abs() < 0.001,
                "Self-similarity should be 1.0, got {}",
                self_sim
            );
        }
    }

    // Property 4: Merge is possible for orthogonal vectors
    if deltas.len() >= 2 {
        let d1 = &deltas[0];
        let d2 = &deltas[1];
        let result = consensus.merge(d1, d2);
        // Should not panic, result can be Ok or Err
        let _ = result;
    }

    // Property 5: merge_all handles multiple deltas
    if !deltas.is_empty() {
        let result = consensus.merge_all(&deltas);
        // Should not panic
        let _ = result;
    }

    // Property 6: find_merge_order returns valid indices
    if !deltas.is_empty() {
        let order = consensus.find_merge_order(&deltas);
        assert_eq!(order.len(), deltas.len(), "Order should include all deltas");
        for idx in &order {
            assert!(*idx < deltas.len(), "Index should be valid");
        }
        // All indices should be unique
        let unique: HashSet<_> = order.iter().collect();
        assert_eq!(unique.len(), order.len(), "Indices should be unique");
    }

    // Property 7: Vector operations don't panic
    if deltas.len() >= 2 {
        let d1 = &deltas[0];
        let d2 = &deltas[1];

        // Add
        let sum = d1.add(d2);
        assert_eq!(sum.delta.dimension(), dimension);

        // Scale
        let scaled = d1.scale(0.5);
        assert_eq!(scaled.delta.dimension(), dimension);

        // Weighted average
        let avg = d1.weighted_average(d2, 0.6, 0.4);
        assert_eq!(avg.delta.dimension(), dimension);
    }

    // Property 8: Key overlap detection is symmetric
    if deltas.len() >= 2 {
        let d1 = &deltas[0];
        let d2 = &deltas[1];

        let overlap12 = d1.overlaps_with(d2);
        let overlap21 = d2.overlaps_with(d1);
        assert_eq!(
            overlap12, overlap21,
            "Overlap detection should be symmetric"
        );
    }
});
