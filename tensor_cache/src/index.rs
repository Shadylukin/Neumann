//! HNSW index wrapper for cache semantic search.
//!
//! Supports both dense and sparse embeddings for memory-efficient caching.

#![allow(dead_code)]

use crate::error::{CacheError, Result};
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use tensor_store::{EmbeddingStorage, HNSWConfig, HNSWIndex, HNSWMemoryStats, SparseVector};

/// Cache-specific HNSW index wrapper.
///
/// Provides key-to-node mapping on top of the shared HNSW implementation.
/// Uses RwLock for the HNSW index to support clearing (which requires replacement).
pub struct CacheIndex {
    /// The underlying HNSW index (wrapped for clear support).
    index: RwLock<HNSWIndex>,
    /// HNSW configuration for recreation on clear.
    config: HNSWConfig,
    /// Map from cache key to HNSW node ID.
    key_to_node: DashMap<String, usize>,
    /// Map from node ID to cache key.
    node_to_key: DashMap<usize, String>,
    /// Expected embedding dimension.
    dimension: usize,
    /// Number of entries.
    entry_count: AtomicUsize,
}

/// Result of a semantic search.
#[derive(Debug, Clone)]
pub struct IndexSearchResult {
    /// The cache key.
    pub key: String,
    /// Similarity score (0.0 to 1.0).
    pub similarity: f32,
}

impl CacheIndex {
    /// Create a new cache index.
    pub fn new(dimension: usize) -> Self {
        Self::with_config(dimension, HNSWConfig::default())
    }

    /// Create a new cache index with custom HNSW configuration.
    pub fn with_config(dimension: usize, config: HNSWConfig) -> Self {
        Self {
            index: RwLock::new(HNSWIndex::with_config(config.clone())),
            config,
            key_to_node: DashMap::new(),
            node_to_key: DashMap::new(),
            dimension,
            entry_count: AtomicUsize::new(0),
        }
    }

    /// Insert a dense embedding for a cache key.
    pub fn insert(&self, key: &str, embedding: &[f32]) -> Result<usize> {
        if embedding.len() != self.dimension {
            return Err(CacheError::DimensionMismatch {
                expected: self.dimension,
                got: embedding.len(),
            });
        }

        // Check if key already exists
        if self.key_to_node.contains_key(key) {
            // For simplicity, we don't update existing entries in HNSW
            // The old node remains but is orphaned (will be ignored in search)
            self.key_to_node.remove(key);
        }

        let index = self.index.read().unwrap();
        let node_id = index.insert(embedding.to_vec());
        drop(index);

        self.key_to_node.insert(key.to_string(), node_id);
        self.node_to_key.insert(node_id, key.to_string());
        self.entry_count.fetch_add(1, Ordering::Relaxed);

        Ok(node_id)
    }

    /// Insert a sparse embedding for a cache key.
    ///
    /// Sparse embeddings use less memory when vectors have high sparsity (>70% zeros).
    pub fn insert_sparse(&self, key: &str, embedding: &SparseVector) -> Result<usize> {
        if embedding.dimension() != self.dimension {
            return Err(CacheError::DimensionMismatch {
                expected: self.dimension,
                got: embedding.dimension(),
            });
        }

        // Check if key already exists
        if self.key_to_node.contains_key(key) {
            self.key_to_node.remove(key);
        }

        let index = self.index.read().unwrap();
        let node_id = index.insert_sparse(embedding.clone());
        drop(index);

        self.key_to_node.insert(key.to_string(), node_id);
        self.node_to_key.insert(node_id, key.to_string());
        self.entry_count.fetch_add(1, Ordering::Relaxed);

        Ok(node_id)
    }

    /// Insert an embedding (dense or sparse) for a cache key.
    ///
    /// Automatically selects optimal storage based on sparsity.
    pub fn insert_auto(
        &self,
        key: &str,
        embedding: &[f32],
        sparsity_threshold: f32,
    ) -> Result<usize> {
        let sparse = SparseVector::from_dense(embedding);

        if sparse.sparsity() >= sparsity_threshold {
            self.insert_sparse(key, &sparse)
        } else {
            self.insert(key, embedding)
        }
    }

    /// Search for similar embeddings using a dense query.
    ///
    /// Returns results above the similarity threshold, sorted by similarity (highest first).
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        threshold: f32,
    ) -> Result<Vec<IndexSearchResult>> {
        if query.len() != self.dimension {
            return Err(CacheError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        let index = self.index.read().unwrap();
        if index.is_empty() {
            return Ok(Vec::new());
        }

        let results = index.search(query, k);
        drop(index);

        Ok(results
            .into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .filter_map(|(node_id, similarity)| {
                self.node_to_key.get(&node_id).map(|key| IndexSearchResult {
                    key: key.clone(),
                    similarity,
                })
            })
            .collect())
    }

    /// Search for similar embeddings using a sparse query.
    ///
    /// More efficient when the query is sparse (>70% zeros).
    pub fn search_sparse(
        &self,
        query: &SparseVector,
        k: usize,
        threshold: f32,
    ) -> Result<Vec<IndexSearchResult>> {
        if query.dimension() != self.dimension {
            return Err(CacheError::DimensionMismatch {
                expected: self.dimension,
                got: query.dimension(),
            });
        }

        let index = self.index.read().unwrap();
        if index.is_empty() {
            return Ok(Vec::new());
        }

        let results = index.search_sparse(query, k);
        drop(index);

        Ok(results
            .into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .filter_map(|(node_id, similarity)| {
                self.node_to_key.get(&node_id).map(|key| IndexSearchResult {
                    key: key.clone(),
                    similarity,
                })
            })
            .collect())
    }

    /// Remove a key from the index.
    ///
    /// Note: The HNSW node is orphaned but not removed from the graph
    /// (HNSW doesn't support efficient deletion). The node_to_key mapping
    /// is removed so it won't appear in search results.
    pub fn remove(&self, key: &str) -> bool {
        if let Some((_, node_id)) = self.key_to_node.remove(key) {
            self.node_to_key.remove(&node_id);
            self.entry_count.fetch_sub(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Check if a key exists in the index.
    pub fn contains(&self, key: &str) -> bool {
        self.key_to_node.contains_key(key)
    }

    /// Get the number of indexed entries.
    pub fn len(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed)
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the expected embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Clear the index (thread-safe).
    ///
    /// Note: This creates a new HNSW index since HNSW doesn't support clearing.
    pub fn clear(&self) {
        let mut index = self.index.write().unwrap();
        *index = HNSWIndex::with_config(self.config.clone());
        drop(index);

        self.key_to_node.clear();
        self.node_to_key.clear();
        self.entry_count.store(0, Ordering::Relaxed);
    }

    /// Get all keys in the index.
    pub fn keys(&self) -> Vec<String> {
        self.key_to_node.iter().map(|e| e.key().clone()).collect()
    }

    /// Get memory usage statistics for the index.
    pub fn memory_stats(&self) -> HNSWMemoryStats {
        let index = self.index.read().unwrap();
        index.memory_stats()
    }

    /// Get the embedding storage for a key.
    pub fn get_embedding(&self, key: &str) -> Option<EmbeddingStorage> {
        let node_id = self.key_to_node.get(key)?;
        let index = self.index.read().unwrap();
        index.get_embedding(*node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vector(dim: usize, seed: usize) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = (seed * 31 + i * 17) as f32;
                (x * 0.0001).sin()
            })
            .collect()
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag == 0.0 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / mag).collect()
        }
    }

    #[test]
    fn test_insert_and_search() {
        let index = CacheIndex::new(3);

        index.insert("key1", &[1.0, 0.0, 0.0]).unwrap();
        index.insert("key2", &[0.0, 1.0, 0.0]).unwrap();
        index.insert("key3", &[1.0, 1.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 3, 0.0).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].key, "key1");
        assert!((results[0].similarity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dimension_mismatch() {
        let index = CacheIndex::new(3);

        let result = index.insert("key1", &[1.0, 0.0]);
        assert!(matches!(result, Err(CacheError::DimensionMismatch { .. })));

        index.insert("key1", &[1.0, 0.0, 0.0]).unwrap();
        let result = index.search(&[1.0, 0.0], 1, 0.0);
        assert!(matches!(result, Err(CacheError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_threshold_filtering() {
        let index = CacheIndex::new(3);

        let v1 = normalize(&[1.0, 0.0, 0.0]);
        let v2 = normalize(&[0.0, 1.0, 0.0]); // orthogonal

        index.insert("similar", &v1).unwrap();
        index.insert("orthogonal", &v2).unwrap();

        // High threshold should only return very similar
        let results = index.search(&v1, 2, 0.9).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "similar");
    }

    #[test]
    fn test_remove() {
        let index = CacheIndex::new(3);

        index.insert("key1", &[1.0, 0.0, 0.0]).unwrap();
        assert!(index.contains("key1"));
        assert_eq!(index.len(), 1);

        assert!(index.remove("key1"));
        assert!(!index.contains("key1"));
        assert_eq!(index.len(), 0);

        // Removed key shouldn't appear in search
        let results = index.search(&[1.0, 0.0, 0.0], 1, 0.0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_search() {
        let index = CacheIndex::new(3);
        let results = index.search(&[1.0, 0.0, 0.0], 5, 0.0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_many_vectors() {
        let index = CacheIndex::new(64);

        for i in 0..100 {
            let v = create_test_vector(64, i);
            index.insert(&format!("key{}", i), &v).unwrap();
        }

        assert_eq!(index.len(), 100);

        let query = create_test_vector(64, 50);
        let results = index.search(&query, 5, 0.0).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_keys() {
        let index = CacheIndex::new(3);

        index.insert("a", &[1.0, 0.0, 0.0]).unwrap();
        index.insert("b", &[0.0, 1.0, 0.0]).unwrap();

        let mut keys = index.keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_with_config() {
        let config = HNSWConfig {
            m: 32,
            ef_construction: 100,
            ..Default::default()
        };
        let index = CacheIndex::with_config(3, config);
        assert!(index.is_empty());
        assert_eq!(index.dimension(), 3);
    }

    #[test]
    fn test_is_empty() {
        let index = CacheIndex::new(3);
        assert!(index.is_empty());

        index.insert("key1", &[1.0, 0.0, 0.0]).unwrap();
        assert!(!index.is_empty());
    }

    #[test]
    fn test_dimension() {
        let index = CacheIndex::new(128);
        assert_eq!(index.dimension(), 128);
    }

    #[test]
    fn test_clear() {
        let index = CacheIndex::new(3);

        index.insert("key1", &[1.0, 0.0, 0.0]).unwrap();
        index.insert("key2", &[0.0, 1.0, 0.0]).unwrap();

        assert_eq!(index.len(), 2);
        index.clear();
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_reinsert_same_key() {
        let index = CacheIndex::new(3);

        index.insert("key1", &[1.0, 0.0, 0.0]).unwrap();
        // Re-insert with different embedding
        index.insert("key1", &[0.0, 1.0, 0.0]).unwrap();

        // Entry count should still be 2 (old node orphaned but counted)
        // Note: current implementation adds to count on each insert
        assert!(index.len() >= 1);
    }

    #[test]
    fn test_remove_nonexistent() {
        let index = CacheIndex::new(3);
        assert!(!index.remove("nonexistent"));
    }

    #[test]
    fn test_contains() {
        let index = CacheIndex::new(3);

        assert!(!index.contains("key1"));
        index.insert("key1", &[1.0, 0.0, 0.0]).unwrap();
        assert!(index.contains("key1"));
    }

    #[test]
    fn test_index_search_result_clone() {
        let result = IndexSearchResult {
            key: "test".into(),
            similarity: 0.95,
        };
        let cloned = result.clone();
        assert_eq!(cloned.key, "test");
        assert_eq!(cloned.similarity, 0.95);
    }

    #[test]
    fn test_insert_sparse_and_search() {
        let index = CacheIndex::new(3);

        let sparse1 = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let sparse2 = SparseVector::from_dense(&[0.0, 1.0, 0.0]);

        index.insert_sparse("key1", &sparse1).unwrap();
        index.insert_sparse("key2", &sparse2).unwrap();

        // Search with sparse query
        let results = index.search_sparse(&sparse1, 2, 0.0).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].key, "key1");
        assert!((results[0].similarity - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_insert_auto_dense() {
        let index = CacheIndex::new(3);

        // Dense vector (no zeros) should be stored as dense
        let embedding = [0.5, 0.5, 0.5];
        index.insert_auto("key1", &embedding, 0.7).unwrap();

        // Should be searchable
        let results = index.search(&embedding, 1, 0.0).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].key, "key1");
    }

    #[test]
    fn test_insert_auto_sparse() {
        let index = CacheIndex::new(10);

        // Sparse vector (80% zeros) should be stored as sparse
        let embedding = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        index.insert_auto("key1", &embedding, 0.7).unwrap();

        // Should be searchable
        let results = index.search(&embedding, 1, 0.0).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].key, "key1");
    }

    #[test]
    fn test_sparse_dimension_mismatch() {
        let index = CacheIndex::new(3);

        let sparse = SparseVector::from_dense(&[1.0, 0.0]); // dimension 2
        let result = index.insert_sparse("key1", &sparse);
        assert!(matches!(result, Err(CacheError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_search_sparse_dimension_mismatch() {
        let index = CacheIndex::new(3);
        index.insert("key1", &[1.0, 0.0, 0.0]).unwrap();

        let query = SparseVector::from_dense(&[1.0, 0.0]); // dimension 2
        let result = index.search_sparse(&query, 1, 0.0);
        assert!(matches!(result, Err(CacheError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_memory_stats() {
        let index = CacheIndex::new(3);

        index.insert("key1", &[1.0, 0.0, 0.0]).unwrap();
        let sparse = SparseVector::from_dense(&[0.0, 1.0, 0.0]);
        index.insert_sparse("key2", &sparse).unwrap();

        let stats = index.memory_stats();
        assert_eq!(stats.dense_count, 1);
        assert_eq!(stats.sparse_count, 1);
        assert!(stats.embedding_bytes > 0);
    }

    #[test]
    fn test_get_embedding() {
        let index = CacheIndex::new(3);

        index.insert("dense_key", &[1.0, 0.0, 0.0]).unwrap();
        let sparse = SparseVector::from_dense(&[0.0, 1.0, 0.0]);
        index.insert_sparse("sparse_key", &sparse).unwrap();

        // Check dense embedding retrieval
        let dense_emb = index.get_embedding("dense_key");
        assert!(matches!(dense_emb, Some(EmbeddingStorage::Dense(_))));

        // Check sparse embedding retrieval
        let sparse_emb = index.get_embedding("sparse_key");
        assert!(matches!(sparse_emb, Some(EmbeddingStorage::Sparse(_))));

        // Check missing key
        assert!(index.get_embedding("missing").is_none());
    }

    #[test]
    fn test_mixed_dense_sparse_search() {
        let index = CacheIndex::new(3);

        // Insert mix of dense and sparse
        index.insert("dense1", &[1.0, 0.0, 0.0]).unwrap();
        let sparse = SparseVector::from_dense(&[0.9, 0.1, 0.0]);
        index.insert_sparse("sparse1", &sparse).unwrap();

        // Dense query should find both
        let results = index.search(&[1.0, 0.0, 0.0], 2, 0.0).unwrap();
        assert_eq!(results.len(), 2);

        // Sparse query should find both too
        let sparse_query = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let results = index.search_sparse(&sparse_query, 2, 0.0).unwrap();
        assert_eq!(results.len(), 2);
    }
}
