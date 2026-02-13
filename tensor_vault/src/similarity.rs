// SPDX-License-Identifier: MIT OR Apache-2.0
//! Metadata-based similarity search for secrets using HNSW index.

use std::sync::RwLock;

use dashmap::DashMap;
use tensor_store::hnsw::{HNSWConfig, HNSWDistanceMetric, HNSWIndex};

/// Feature vector dimensions for secret metadata embeddings.
#[cfg(test)]
const FEATURE_DIMS: usize = 6;

/// Operational metadata features for a secret.
#[derive(Debug, Clone)]
pub struct SecretFeatures {
    /// Unique identifier for the secret.
    pub key: String,
    /// Age of the secret in days since creation.
    pub creation_age_days: f32,
    /// Number of stored versions.
    pub version_count: f32,
    /// Number of accesses within the observation period.
    pub access_frequency: f32,
    /// Days elapsed since the last key rotation.
    pub days_since_rotation: f32,
    /// Number of distinct entities with access.
    pub entity_count: f32,
    /// Shannon entropy of permission levels across entities.
    pub permission_entropy: f32,
}

impl SecretFeatures {
    /// Convert features to a normalized embedding vector.
    pub fn to_embedding(&self) -> Vec<f32> {
        // Normalize each feature to [0,1] range using reasonable max values
        let max_age = 365.0_f32;
        let max_versions = 100.0_f32;
        let max_frequency = 1000.0_f32;
        let max_rotation = 365.0_f32;
        let max_entities = 100.0_f32;
        let max_entropy = 3.0_f32; // log2(8) ~ 3 for up to 8 distinct permission levels

        vec![
            (self.creation_age_days / max_age).clamp(0.0, 1.0),
            (self.version_count / max_versions).clamp(0.0, 1.0),
            (self.access_frequency / max_frequency).clamp(0.0, 1.0),
            (self.days_since_rotation / max_rotation).clamp(0.0, 1.0),
            (self.entity_count / max_entities).clamp(0.0, 1.0),
            (self.permission_entropy / max_entropy).clamp(0.0, 1.0),
        ]
    }
}

/// A similar secret found by the similarity index.
#[derive(Debug, Clone)]
pub struct SimilarSecret {
    /// Key of the similar secret.
    pub key: String,
    /// Similarity score in (0, 1], where 1 is identical.
    pub similarity: f32,
}

/// HNSW-backed similarity index for secret metadata embeddings.
pub struct SimilarityIndex {
    index: RwLock<HNSWIndex>,
    key_to_node: DashMap<String, usize>,
    node_to_key: DashMap<usize, String>,
}

impl SimilarityIndex {
    /// Create a new similarity index with default HNSW configuration.
    pub fn new() -> Self {
        let config = HNSWConfig {
            m: 8,
            m0: 16,
            ef_construction: 100,
            ef_search: 50,
            ml: 1.0 / 8.0_f64.ln(),
            sparsity_threshold: 0.5,
            max_nodes: 100_000,
            distance_metric: HNSWDistanceMetric::Euclidean,
        };
        Self {
            index: RwLock::new(HNSWIndex::with_config(config)),
            key_to_node: DashMap::new(),
            node_to_key: DashMap::new(),
        }
    }

    /// Insert or update a secret's embedding in the index.
    pub fn insert(&self, key: &str, embedding: Vec<f32>) {
        let node_id = self
            .index
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(embedding);
        // Clean up old mapping if updating
        if let Some((_, old_id)) = self.key_to_node.remove(key) {
            self.node_to_key.remove(&old_id);
        }
        self.key_to_node.insert(key.to_string(), node_id);
        self.node_to_key.insert(node_id, key.to_string());
    }

    /// Search for the k most similar secrets to a query embedding.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<SimilarSecret> {
        let results = self
            .index
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .search(query, k);
        results
            .into_iter()
            .filter_map(|(node_id, distance)| {
                self.node_to_key.get(&node_id).map(|key| SimilarSecret {
                    key: key.clone(),
                    similarity: 1.0 / (1.0 + distance),
                })
            })
            .collect()
    }

    /// Remove a secret from the index.
    pub fn remove(&self, key: &str) {
        // HNSW doesn't support true removal, but we can remove the mapping
        if let Some((_, node_id)) = self.key_to_node.remove(key) {
            self.node_to_key.remove(&node_id);
        }
    }

    /// Return the number of secrets currently indexed.
    pub fn len(&self) -> usize {
        self.key_to_node.len()
    }

    /// Return `true` if no secrets are indexed.
    pub fn is_empty(&self) -> bool {
        self.key_to_node.is_empty()
    }
}

impl Default for SimilarityIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(key: &str, age: f32, versions: f32, freq: f32) -> SecretFeatures {
        SecretFeatures {
            key: key.to_string(),
            creation_age_days: age,
            version_count: versions,
            access_frequency: freq,
            days_since_rotation: age,
            entity_count: 3.0,
            permission_entropy: 1.0,
        }
    }

    #[test]
    fn test_secret_features_to_embedding() {
        let features = SecretFeatures {
            key: "test".to_string(),
            creation_age_days: 365.0,
            version_count: 100.0,
            access_frequency: 1000.0,
            days_since_rotation: 365.0,
            entity_count: 100.0,
            permission_entropy: 3.0,
        };
        let emb = features.to_embedding();
        assert_eq!(emb.len(), FEATURE_DIMS);
        for &v in &emb {
            assert!((0.0..=1.0).contains(&v), "value {v} out of [0,1]");
        }
        // At max values, all should be 1.0
        assert!((emb[0] - 1.0).abs() < f32::EPSILON);

        // Over-max values should clamp to 1.0
        let over = SecretFeatures {
            key: "over".to_string(),
            creation_age_days: 999.0,
            version_count: 999.0,
            access_frequency: 9999.0,
            days_since_rotation: 999.0,
            entity_count: 999.0,
            permission_entropy: 99.0,
        };
        let emb2 = over.to_embedding();
        for &v in &emb2 {
            assert!((v - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_similarity_index_insert_search() {
        let idx = SimilarityIndex::new();

        let f1 = make_features("secret1", 10.0, 3.0, 50.0);
        let f2 = make_features("secret2", 11.0, 3.0, 52.0);
        let f3 = make_features("secret3", 200.0, 20.0, 500.0);

        idx.insert("secret1", f1.to_embedding());
        idx.insert("secret2", f2.to_embedding());
        idx.insert("secret3", f3.to_embedding());

        assert_eq!(idx.len(), 3);

        // Search near secret1 -- secret2 should be closest
        let results = idx.search(&f1.to_embedding(), 2);
        assert!(!results.is_empty());
        // First result should be secret1 itself (exact match) or secret2
        let keys: Vec<&str> = results.iter().map(|r| r.key.as_str()).collect();
        assert!(keys.contains(&"secret1") || keys.contains(&"secret2"));
    }

    #[test]
    fn test_similarity_index_remove() {
        let idx = SimilarityIndex::new();
        let f1 = make_features("s1", 10.0, 1.0, 10.0);
        idx.insert("s1", f1.to_embedding());
        assert_eq!(idx.len(), 1);

        idx.remove("s1");
        assert!(idx.is_empty());

        // Searching should not return removed key
        let results = idx.search(&f1.to_embedding(), 5);
        assert!(results.iter().all(|r| r.key != "s1"));
    }

    #[test]
    fn test_similarity_index_empty_search() {
        let idx = SimilarityIndex::new();
        let results = idx.search(&[0.0; FEATURE_DIMS], 5);
        assert!(results.is_empty());
    }
}
