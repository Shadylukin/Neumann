// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{HNSWIndex, SparseVector};

#[derive(Arbitrary, Debug)]
struct Vector {
    // Limit dimension for performance
    values: Vec<f32>,
}

#[derive(Arbitrary, Debug, Clone)]
enum Op {
    Insert { vector_idx: u8 },
    Search { vector_idx: u8, k: u8 },
}

#[derive(Arbitrary, Debug)]
struct Input {
    dimension: u8,
    vectors: Vec<Vector>,
    ops: Vec<Op>,
}

fn normalize_vector(v: &[f32]) -> Vec<f32> {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag == 0.0 || mag.is_nan() || mag.is_infinite() {
        v.to_vec()
    } else {
        v.iter().map(|x| x / mag).collect()
    }
}

fuzz_target!(|input: Input| {
    // Limit dimension (8-128)
    let dimension = (input.dimension as usize).max(8).min(128);

    // Limit vectors and ops
    if input.vectors.len() > 32 || input.ops.len() > 64 {
        return;
    }

    // Prepare vectors
    let vectors: Vec<Vec<f32>> = input
        .vectors
        .iter()
        .map(|v| {
            let mut vec: Vec<f32> = v
                .values
                .iter()
                .take(dimension)
                .copied()
                .filter(|x| x.is_finite())
                .collect();
            // Pad to dimension
            vec.resize(dimension, 0.0);
            normalize_vector(&vec)
        })
        .collect();

    if vectors.is_empty() {
        return;
    }

    // Create HNSW index with default config
    let index = HNSWIndex::new();
    let mut inserted_count = 0;

    for op in input.ops {
        match op {
            Op::Insert { vector_idx } => {
                let idx = vector_idx as usize % vectors.len();
                let vec = &vectors[idx];
                let sparse = SparseVector::from_dense(vec);
                let _ = index.insert_sparse(sparse);
                inserted_count += 1;
            },
            Op::Search { vector_idx, k } => {
                if inserted_count == 0 {
                    continue;
                }
                let idx = vector_idx as usize % vectors.len();
                let vec = &vectors[idx];
                let sparse = SparseVector::from_dense(vec);
                let k = (k as usize).max(1).min(inserted_count);
                let results = index.search_sparse(&sparse, k);

                // Verify results are valid
                assert!(
                    results.len() <= k,
                    "Search returned more results ({}) than k ({})",
                    results.len(),
                    k
                );

                // Verify all scores are finite
                for (_id, score) in &results {
                    assert!(
                        score.is_finite(),
                        "Search returned non-finite score: {}",
                        score
                    );
                }
            },
        }
    }

    // Verify index state
    assert_eq!(
        index.len(),
        inserted_count,
        "Index len {} != inserted count {}",
        index.len(),
        inserted_count
    );
});
