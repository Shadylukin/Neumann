// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{CodebookManager, GlobalCodebook, LocalCodebook};

#[derive(Arbitrary, Debug)]
struct CodebookInput {
    // Number of centroids (1-16)
    num_centroids: u8,
    // Dimension (4-64)
    dimension: u8,
    // Centroid data (flattened)
    centroids_flat: Vec<f32>,
    // Query vectors (flattened)
    queries_flat: Vec<f32>,
    // Validity threshold
    validity_threshold: u8,
}

fuzz_target!(|input: CodebookInput| {
    // Constrain parameters
    let num_centroids = (input.num_centroids as usize).clamp(1, 16);
    let dimension = (input.dimension as usize).clamp(4, 64);
    let validity_threshold = (input.validity_threshold as f32 / 255.0).clamp(0.1, 0.99);

    // Build centroids
    let mut centroids = Vec::new();
    let mut flat_idx = 0;

    for _ in 0..num_centroids {
        if flat_idx + dimension > input.centroids_flat.len() {
            break;
        }

        let centroid: Vec<f32> = input
            .centroids_flat
            .iter()
            .skip(flat_idx)
            .take(dimension)
            .copied()
            .map(|x| if x.is_finite() { x } else { 0.0 })
            .collect();

        // Skip zero centroids
        if centroid.iter().any(|x| *x != 0.0) {
            centroids.push(centroid);
        }

        flat_idx += dimension;
    }

    if centroids.is_empty() {
        return;
    }

    // Create GlobalCodebook
    let global = GlobalCodebook::from_centroids(centroids.clone());

    // Property 1: Codebook size matches input
    assert_eq!(global.len(), centroids.len());
    assert!(!global.is_empty());

    // Property 2: Dimension is correct
    assert_eq!(global.dimension(), dimension);

    // Property 3: Can retrieve all entries
    for id in 0..centroids.len() {
        let entry = global.get(id as u32);
        assert!(entry.is_some(), "Entry {} should exist", id);
        assert_eq!(entry.unwrap().centroid().len(), dimension);
    }

    // Property 4: Quantization returns valid ID
    flat_idx = 0;
    for _ in 0..10 {
        if flat_idx + dimension > input.queries_flat.len() {
            break;
        }

        let query: Vec<f32> = input
            .queries_flat
            .iter()
            .skip(flat_idx)
            .take(dimension)
            .copied()
            .map(|x| if x.is_finite() { x } else { 0.0 })
            .collect();

        flat_idx += dimension;

        // Skip zero queries
        if query.iter().all(|x| *x == 0.0) {
            continue;
        }

        if let Some((id, similarity)) = global.quantize(&query) {
            assert!(
                (id as usize) < centroids.len(),
                "Quantized ID should be valid"
            );
            assert!(
                similarity >= -1.0 && similarity <= 1.0,
                "Similarity should be in [-1, 1]"
            );
        }
    }

    // Property 5: Residual computation
    flat_idx = 0;
    for _ in 0..5 {
        if flat_idx + dimension > input.queries_flat.len() {
            break;
        }

        let query: Vec<f32> = input
            .queries_flat
            .iter()
            .skip(flat_idx)
            .take(dimension)
            .copied()
            .map(|x| if x.is_finite() { x } else { 0.0 })
            .collect();

        flat_idx += dimension;

        if query.iter().all(|x| *x == 0.0) {
            continue;
        }

        if let Some((id, residual)) = global.compute_residual(&query) {
            assert!(
                (id as usize) < centroids.len(),
                "Residual ID should be valid"
            );
            assert_eq!(
                residual.len(),
                dimension,
                "Residual should have same dimension"
            );
        }
    }

    // Property 6: is_valid_state is deterministic
    flat_idx = 0;
    for _ in 0..5 {
        if flat_idx + dimension > input.queries_flat.len() {
            break;
        }

        let state: Vec<f32> = input
            .queries_flat
            .iter()
            .skip(flat_idx)
            .take(dimension)
            .copied()
            .map(|x| if x.is_finite() { x } else { 0.0 })
            .collect();

        flat_idx += dimension;

        let valid1 = global.is_valid_state(&state, validity_threshold);
        let valid2 = global.is_valid_state(&state, validity_threshold);
        assert_eq!(valid1, valid2, "is_valid_state should be deterministic");
    }

    // Property 7: LocalCodebook operations
    let local = LocalCodebook::new("test_domain", dimension, 8, 0.1);

    flat_idx = 0;
    for _ in 0..10 {
        if flat_idx + dimension > input.queries_flat.len() {
            break;
        }

        let observation: Vec<f32> = input
            .queries_flat
            .iter()
            .skip(flat_idx)
            .take(dimension)
            .copied()
            .map(|x| if x.is_finite() { x } else { 0.0 })
            .collect();

        flat_idx += dimension;

        if observation.iter().all(|x| *x == 0.0) {
            continue;
        }

        // quantize_and_update should not panic
        let (id, sim) = local.quantize_and_update(&observation, 0.9);
        assert!(id < 1000, "ID should be reasonable"); // Just a sanity check
        assert!(sim >= -1.0 && sim <= 1.0, "Similarity should be bounded");
    }

    // Property 8: CodebookManager works with global
    let manager = CodebookManager::with_global(global.clone());

    flat_idx = 0;
    for _ in 0..5 {
        if flat_idx + dimension > input.queries_flat.len() {
            break;
        }

        let vector: Vec<f32> = input
            .queries_flat
            .iter()
            .skip(flat_idx)
            .take(dimension)
            .copied()
            .map(|x| if x.is_finite() { x } else { 0.0 })
            .collect();

        flat_idx += dimension;

        if vector.iter().all(|x| *x == 0.0) {
            continue;
        }

        // Should not panic
        let _ = manager.quantize("test_domain", &vector);
    }
});
