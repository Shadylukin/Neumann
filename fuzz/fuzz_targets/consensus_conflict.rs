// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Fuzz semantic conflict detection with arbitrary transaction deltas.
//!
//! Tests that:
//! - ConsensusManager.detect_conflict never panics on any input
//! - ConflictClass is always consistent with can_merge/should_reject
//! - Batch conflict detection handles all pair combinations
//! - Merge operations produce valid results or graceful rejections

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashSet;
use tensor_chain::consensus::{
    ConflictClass, ConsensusConfig, ConsensusManager, DeltaVector, MergeAction,
};

#[derive(Debug, Arbitrary)]
enum TxOp {
    Put { key_idx: u8, value_len: u8 },
    Delete { key_idx: u8 },
    Embed { key_idx: u8, dim: u8 },
}

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    operations: Vec<TxOp>,
    orthogonal_threshold: u8,
    conflict_threshold: u8,
}

fn make_delta(op: &TxOp, keys: &[String], tx_id: u64) -> DeltaVector {
    match op {
        TxOp::Put {
            key_idx,
            value_len,
        } => {
            let key = keys[(*key_idx as usize) % keys.len()].clone();
            let len = ((*value_len as usize) % 32).max(1);
            let vector: Vec<f32> = (0..len).map(|i| (i as f32 + 1.0) / len as f32).collect();
            let mut affected = HashSet::new();
            affected.insert(key);
            DeltaVector::new(&vector, affected, tx_id)
        }
        TxOp::Delete { key_idx } => {
            let key = keys[(*key_idx as usize) % keys.len()].clone();
            let vector = vec![-1.0, 0.0, 0.0, 0.0];
            let mut affected = HashSet::new();
            affected.insert(key);
            DeltaVector::new(&vector, affected, tx_id)
        }
        TxOp::Embed { key_idx, dim } => {
            let key = keys[(*key_idx as usize) % keys.len()].clone();
            let d = ((*dim as usize) % 16).max(1);
            let vector: Vec<f32> = (0..d).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
            let mut affected = HashSet::new();
            affected.insert(key);
            DeltaVector::new(&vector, affected, tx_id)
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = FuzzInput::arbitrary(&mut arbitrary::Unstructured::new(data)) else {
        return;
    };
    if input.operations.len() > 50 {
        return;
    }
    if input.operations.is_empty() {
        return;
    }

    // Derive thresholds from fuzz input (keep within valid ranges)
    let orth = (f32::from(input.orthogonal_threshold) / 255.0) * 0.5; // 0.0..0.5
    let conf = 0.5 + (f32::from(input.conflict_threshold) / 255.0) * 0.5; // 0.5..1.0

    let config = ConsensusConfig {
        orthogonal_threshold: orth,
        conflict_threshold: conf,
        ..ConsensusConfig::default()
    };
    let manager = ConsensusManager::new(config);

    let keys: Vec<String> = (0..8).map(|i| format!("key{i}")).collect();

    // Build deltas from operations
    let deltas: Vec<DeltaVector> = input
        .operations
        .iter()
        .enumerate()
        .map(|(i, op)| make_delta(op, &keys, i as u64))
        .collect();

    // Pairwise conflict detection - should never panic
    for i in 0..deltas.len() {
        for j in (i + 1)..deltas.len() {
            let result = manager.detect_conflict(&deltas[i], &deltas[j]);

            // Invariant: can_merge and should_reject are consistent with class
            assert_eq!(result.can_merge, result.class.can_merge());
            assert_eq!(
                result.class.should_reject(),
                matches!(result.class, ConflictClass::Ambiguous | ConflictClass::Conflicting)
            );

            // Invariant: action is consistent with class
            match result.class {
                ConflictClass::Orthogonal => {
                    assert!(matches!(result.action, MergeAction::VectorAdd));
                }
                ConflictClass::Identical => {
                    assert!(matches!(result.action, MergeAction::Deduplicate));
                }
                ConflictClass::Opposite => {
                    assert!(matches!(result.action, MergeAction::Cancel));
                }
                ConflictClass::Conflicting | ConflictClass::Ambiguous => {
                    assert!(matches!(result.action, MergeAction::Reject));
                }
                ConflictClass::LowConflict => {
                    assert!(matches!(result.action, MergeAction::WeightedAverage { .. }));
                }
            }

            // Invariant: similarity is in valid range [-1.0, 1.0] or NaN for zero vectors
            if !result.similarity.is_nan() {
                assert!(
                    result.similarity >= -1.01 && result.similarity <= 1.01,
                    "Similarity out of range: {}",
                    result.similarity
                );
            }
        }
    }

    // Pairwise merge - should never panic
    for i in 0..deltas.len().min(10) {
        for j in (i + 1)..deltas.len().min(10) {
            let result = manager.merge(&deltas[i], &deltas[j]);
            if result.success {
                assert!(result.merged_delta.is_some());
                assert!(result.error.is_none());
            } else {
                assert!(result.merged_delta.is_none(), "Failed merge must not produce a delta");
                assert!(result.error.is_some(), "Failed merge must have an error reason");
            }
        }
    }

    // Batch conflict detection - should never panic
    if deltas.len() <= 20 {
        let batch = manager.batch_detect_conflicts(&deltas);
        // All batch results should have valid indices
        for conflict in &batch {
            assert!(conflict.index_a < deltas.len());
            assert!(conflict.index_b < deltas.len());
            assert!(conflict.index_a < conflict.index_b);
        }
    }

    // merge_all - should never panic
    if deltas.len() <= 10 {
        let result = manager.merge_all(&deltas);
        if result.success && !deltas.is_empty() {
            assert!(!result.parent_ids.is_empty());
        }
    }

    // find_merge_order - should never panic and return valid permutation
    if deltas.len() <= 10 {
        let order = manager.find_merge_order(&deltas);
        assert_eq!(order.len(), deltas.len());
        let mut sorted = order.clone();
        sorted.sort_unstable();
        let expected: Vec<usize> = (0..deltas.len()).collect();
        assert_eq!(sorted, expected, "find_merge_order must return a permutation");
    }
});
