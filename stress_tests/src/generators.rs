// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Reproducible data generation for stress tests.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tensor_store::{ScalarValue, SparseVector, TensorData, TensorValue};

/// Generate normalized embedding vectors.
pub fn generate_embeddings(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.into_iter().map(|x| x / norm).collect()
            } else {
                v
            }
        })
        .collect()
}

/// Generate sparse embedding vectors with given sparsity (0.0-1.0).
pub fn generate_sparse_embeddings(
    count: usize,
    dim: usize,
    sparsity: f32,
    seed: u64,
) -> Vec<SparseVector> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let non_zero_count = ((1.0 - sparsity) * dim as f32) as usize;

    (0..count)
        .map(|_| {
            let mut indices: Vec<usize> = (0..dim).collect();
            for i in 0..non_zero_count {
                let j = rng.random_range(i..dim);
                indices.swap(i, j);
            }
            indices.truncate(non_zero_count);
            indices.sort_unstable();

            let positions: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            let values: Vec<f32> = (0..non_zero_count)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();

            SparseVector::from_parts(dim, positions, values)
        })
        .collect()
}

/// Generate TensorData entries with id and embedding.
pub fn generate_tensor_data(count: usize, dim: usize, seed: u64) -> Vec<TensorData> {
    let embeddings = generate_embeddings(count, dim, seed);
    embeddings
        .into_iter()
        .enumerate()
        .map(|(i, emb)| {
            let mut data = TensorData::new();
            data.set("id", TensorValue::Scalar(ScalarValue::Int(i as i64)));
            data.set("embedding", TensorValue::Vector(emb));
            data
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_embeddings_reproducible() {
        let v1 = generate_embeddings(10, 128, 42);
        let v2 = generate_embeddings(10, 128, 42);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_generate_embeddings_normalized() {
        let embeddings = generate_embeddings(10, 128, 42);
        for emb in embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_generate_sparse_embeddings() {
        let sparse = generate_sparse_embeddings(10, 100, 0.9, 42);
        for sv in sparse {
            assert!(sv.nnz() <= 10); // 10% non-zero
        }
    }

    #[test]
    fn test_generate_tensor_data() {
        let data = generate_tensor_data(10, 128, 42);
        assert_eq!(data.len(), 10);
        for (i, d) in data.iter().enumerate() {
            if let Some(TensorValue::Scalar(ScalarValue::Int(id))) = d.get("id") {
                assert_eq!(*id, i as i64);
            } else {
                panic!("Expected id field");
            }
        }
    }
}
