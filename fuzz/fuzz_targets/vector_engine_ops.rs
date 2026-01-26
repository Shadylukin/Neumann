#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use vector_engine::{DistanceMetric, EmbeddingInput, Pagination, VectorEngine};

#[derive(Arbitrary, Debug, Clone)]
enum FuzzMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

impl FuzzMetric {
    fn to_metric(&self) -> DistanceMetric {
        match self {
            Self::Cosine => DistanceMetric::Cosine,
            Self::Euclidean => DistanceMetric::Euclidean,
            Self::DotProduct => DistanceMetric::DotProduct,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    Store {
        key_idx: u8,
        vector: Vec<f32>,
    },
    Get {
        key_idx: u8,
    },
    Delete {
        key_idx: u8,
    },
    SearchSimilar {
        vector: Vec<f32>,
        top_k: u8,
    },
    SearchSimilarWithMetric {
        vector: Vec<f32>,
        top_k: u8,
        metric: FuzzMetric,
    },
    BatchStore {
        inputs: Vec<(u8, Vec<f32>)>,
    },
    BatchDelete {
        key_indices: Vec<u8>,
    },
    SetEntityEmbedding {
        key_idx: u8,
        vector: Vec<f32>,
    },
    SearchEntities {
        vector: Vec<f32>,
        top_k: u8,
    },
    ListKeysPaginated {
        skip: u8,
        limit: u8,
    },
    Count,
    Exists {
        key_idx: u8,
    },
    Clear,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<FuzzOp>,
}

fn sanitize_key(idx: u8) -> String {
    format!("key_{}", idx % 64)
}

fn sanitize_vector(v: &[f32], max_dim: usize) -> Vec<f32> {
    if v.is_empty() {
        return vec![1.0];
    }
    v.iter()
        .take(max_dim)
        .map(|&x| {
            if x.is_nan() || x.is_infinite() {
                0.0
            } else {
                x.clamp(-1e6, 1e6)
            }
        })
        .collect()
}

fuzz_target!(|input: FuzzInput| {
    let engine = VectorEngine::new();
    let mut stored_keys: Vec<String> = Vec::new();
    const MAX_DIM: usize = 128;

    for op in input.ops.into_iter().take(100) {
        match op {
            FuzzOp::Store { key_idx, vector } => {
                let key = sanitize_key(key_idx);
                let vec = sanitize_vector(&vector, MAX_DIM);
                if engine.store_embedding(&key, vec).is_ok() && !stored_keys.contains(&key) {
                    stored_keys.push(key);
                }
            },
            FuzzOp::Get { key_idx } => {
                if stored_keys.is_empty() {
                    continue;
                }
                let key = &stored_keys[key_idx as usize % stored_keys.len()];
                let _ = engine.get_embedding(key);
            },
            FuzzOp::Delete { key_idx } => {
                if stored_keys.is_empty() {
                    continue;
                }
                let idx = key_idx as usize % stored_keys.len();
                let key = stored_keys[idx].clone();
                if engine.delete_embedding(&key).is_ok() {
                    stored_keys.remove(idx);
                }
            },
            FuzzOp::SearchSimilar { vector, top_k } => {
                let vec = sanitize_vector(&vector, MAX_DIM);
                let k = (top_k % 20).max(1) as usize;
                let _ = engine.search_similar(&vec, k);
            },
            FuzzOp::SearchSimilarWithMetric {
                vector,
                top_k,
                metric,
            } => {
                let vec = sanitize_vector(&vector, MAX_DIM);
                let k = (top_k % 20).max(1) as usize;
                let _ = engine.search_similar_with_metric(&vec, k, metric.to_metric());
            },
            FuzzOp::BatchStore { inputs } => {
                let batch: Vec<EmbeddingInput> = inputs
                    .into_iter()
                    .take(50)
                    .map(|(idx, vec)| {
                        let key = sanitize_key(idx);
                        let v = sanitize_vector(&vec, MAX_DIM);
                        EmbeddingInput::new(key, v)
                    })
                    .collect();
                if let Ok(result) = engine.batch_store_embeddings(batch) {
                    for key in result.stored_keys {
                        if !stored_keys.contains(&key) {
                            stored_keys.push(key);
                        }
                    }
                }
            },
            FuzzOp::BatchDelete { key_indices } => {
                if stored_keys.is_empty() {
                    continue;
                }
                let keys: Vec<String> = key_indices
                    .iter()
                    .take(50)
                    .map(|&idx| stored_keys[idx as usize % stored_keys.len()].clone())
                    .collect();
                if let Ok(count) = engine.batch_delete_embeddings(keys.clone()) {
                    if count > 0 {
                        stored_keys.retain(|k| !keys.contains(k));
                    }
                }
            },
            FuzzOp::SetEntityEmbedding { key_idx, vector } => {
                let key = format!("entity:{}", key_idx % 64);
                let vec = sanitize_vector(&vector, MAX_DIM);
                let _ = engine.set_entity_embedding(&key, vec);
            },
            FuzzOp::SearchEntities { vector, top_k } => {
                let vec = sanitize_vector(&vector, MAX_DIM);
                let k = (top_k % 20).max(1) as usize;
                let _ = engine.search_entities(&vec, k);
            },
            FuzzOp::ListKeysPaginated { skip, limit } => {
                let pagination = Pagination::new(skip as usize, limit.max(1) as usize);
                let _ = engine.list_keys_paginated(pagination);
            },
            FuzzOp::Count => {
                let _ = engine.count();
            },
            FuzzOp::Exists { key_idx } => {
                let key = sanitize_key(key_idx);
                let _ = engine.exists(&key);
            },
            FuzzOp::Clear => {
                let _ = engine.clear();
                stored_keys.clear();
            },
        }
    }
});
