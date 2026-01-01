//! Vector Engine - Module 4 of Neumann
//!
//! Provides embeddings storage and similarity search functionality.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tensor_store::{
    fields, hnsw::simd, SparseVector, TensorData, TensorStore, TensorStoreError, TensorValue,
};

// Re-export HNSW types from tensor_store for backward compatibility
pub use tensor_store::{HNSWConfig, HNSWIndex};

/// Error types for vector operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VectorError {
    /// The requested embedding was not found.
    NotFound(String),
    /// Vector dimension mismatch.
    DimensionMismatch { expected: usize, got: usize },
    /// Empty vector provided.
    EmptyVector,
    /// Invalid top_k value.
    InvalidTopK,
    /// Storage error from underlying tensor store.
    StorageError(String),
}

impl std::fmt::Display for VectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VectorError::NotFound(key) => write!(f, "Embedding not found: {}", key),
            VectorError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            },
            VectorError::EmptyVector => write!(f, "Empty vector provided"),
            VectorError::InvalidTopK => write!(f, "Invalid top_k value (must be > 0)"),
            VectorError::StorageError(e) => write!(f, "Storage error: {}", e),
        }
    }
}

impl std::error::Error for VectorError {}

impl From<TensorStoreError> for VectorError {
    fn from(e: TensorStoreError) -> Self {
        VectorError::StorageError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, VectorError>;

/// Result of a similarity search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResult {
    /// The key of the matching embedding.
    pub key: String,
    /// The similarity score (cosine similarity, range -1 to 1).
    pub score: f32,
}

impl SearchResult {
    pub fn new(key: String, score: f32) -> Self {
        Self { key, score }
    }
}

/// Distance metric for similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Cosine similarity: measures angle between vectors (default).
    #[default]
    Cosine,
    /// Euclidean distance: L2 norm of difference.
    Euclidean,
    /// Dot product: inner product of vectors.
    DotProduct,
}

/// Vector Engine for storing and searching embeddings.
///
/// Uses cosine similarity for nearest neighbor search.
pub struct VectorEngine {
    store: TensorStore,
}

impl VectorEngine {
    /// Create a new VectorEngine with a fresh TensorStore.
    pub fn new() -> Self {
        Self {
            store: TensorStore::new(),
        }
    }

    /// Create a VectorEngine using an existing TensorStore.
    pub fn with_store(store: TensorStore) -> Self {
        Self { store }
    }

    /// Get a reference to the underlying TensorStore.
    pub fn store(&self) -> &TensorStore {
        &self.store
    }

    /// Key prefix for embeddings.
    fn embedding_key(key: &str) -> String {
        format!("emb:{}", key)
    }

    /// Key prefix for all embeddings (for scanning).
    fn embedding_prefix() -> &'static str {
        "emb:"
    }

    /// Store an embedding vector with the given key.
    ///
    /// Automatically uses sparse format for vectors with >50% zeros.
    /// Overwrites any existing embedding with the same key.
    pub fn store_embedding(&self, key: &str, vector: Vec<f32>) -> Result<()> {
        if vector.is_empty() {
            return Err(VectorError::EmptyVector);
        }

        let storage_key = Self::embedding_key(key);
        let mut tensor = TensorData::new();

        // Detect sparse vectors and store in optimal format
        let storage = if Self::should_use_sparse(&vector) {
            TensorValue::Sparse(SparseVector::from_dense(&vector))
        } else {
            TensorValue::Vector(vector)
        };
        tensor.set("vector", storage);

        self.store.put(storage_key, tensor)?;
        Ok(())
    }

    /// Check if a vector should use sparse storage.
    fn should_use_sparse(vector: &[f32]) -> bool {
        if vector.is_empty() {
            return false;
        }
        let nnz = vector.iter().filter(|&&v| v.abs() > 1e-6).count();
        // For 0.5 threshold: sparse if nnz <= len/2, i.e., nnz*2 <= len
        nnz * 2 <= vector.len()
    }

    /// Get an embedding by key.
    ///
    /// Returns the embedding as a dense vector regardless of storage format.
    pub fn get_embedding(&self, key: &str) -> Result<Vec<f32>> {
        let storage_key = Self::embedding_key(key);
        let tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(key.to_string()))?;

        match tensor.get("vector") {
            Some(TensorValue::Vector(v)) => Ok(v.clone()),
            Some(TensorValue::Sparse(s)) => Ok(s.to_dense()),
            _ => Err(VectorError::NotFound(key.to_string())),
        }
    }

    /// Delete an embedding by key.
    pub fn delete_embedding(&self, key: &str) -> Result<()> {
        let storage_key = Self::embedding_key(key);
        if !self.store.exists(&storage_key) {
            return Err(VectorError::NotFound(key.to_string()));
        }
        self.store.delete(&storage_key)?;
        Ok(())
    }

    /// Check if an embedding exists.
    pub fn exists(&self, key: &str) -> bool {
        let storage_key = Self::embedding_key(key);
        self.store.exists(&storage_key)
    }

    /// Get the count of stored embeddings.
    pub fn count(&self) -> usize {
        self.store.scan_count(Self::embedding_prefix())
    }

    /// Threshold for parallel search (below this, sequential is faster)
    const PARALLEL_THRESHOLD: usize = 5000;

    /// Search for the top_k most similar embeddings to the query vector.
    ///
    /// Returns results sorted by similarity score (highest first).
    /// Uses cosine similarity with SIMD acceleration.
    /// Automatically uses parallel iteration for large datasets (>5000 vectors).
    pub fn search_similar(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        // Pre-compute query magnitude for efficiency
        let query_magnitude = Self::magnitude(query);
        if query_magnitude == 0.0 {
            // Zero vector - return empty results
            return Ok(Vec::new());
        }

        let keys = self.store.scan(Self::embedding_prefix());

        // Use parallel iteration for large datasets, sequential for small
        let mut results: Vec<SearchResult> = if keys.len() >= Self::PARALLEL_THRESHOLD {
            Self::search_parallel(&self.store, &keys, query, query_magnitude)
        } else {
            Self::search_sequential(&self.store, &keys, query, query_magnitude)
        };

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top_k
        results.truncate(top_k);

        Ok(results)
    }

    /// Search for the top_k most similar embeddings using the specified distance metric.
    ///
    /// - Cosine: Higher scores are more similar (range -1 to 1)
    /// - DotProduct: Higher scores are more similar (unbounded)
    /// - Euclidean: Lower distances are more similar (converted to score via 1/(1+dist))
    pub fn search_similar_with_metric(
        &self,
        query: &[f32],
        top_k: usize,
        metric: DistanceMetric,
    ) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        let query_magnitude = Self::magnitude(query);
        // Zero-magnitude queries are invalid for cosine/dot product but valid for euclidean
        if query_magnitude == 0.0 && !matches!(metric, DistanceMetric::Euclidean) {
            return Ok(Vec::new());
        }

        let keys = self.store.scan(Self::embedding_prefix());

        let mut results: Vec<SearchResult> = if keys.len() >= Self::PARALLEL_THRESHOLD {
            Self::search_parallel_with_metric(&self.store, &keys, query, query_magnitude, metric)
        } else {
            Self::search_sequential_with_metric(&self.store, &keys, query, query_magnitude, metric)
        };

        // Sort by score descending (higher is better for all metrics after transformation)
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(top_k);
        Ok(results)
    }

    /// Extract vector from TensorValue, handling both dense and sparse formats.
    fn extract_vector(value: &TensorValue) -> Option<Vec<f32>> {
        match value {
            TensorValue::Vector(v) => Some(v.clone()),
            TensorValue::Sparse(s) => Some(s.to_dense()),
            _ => None,
        }
    }

    /// Sequential similarity search (faster for small datasets)
    fn search_sequential(
        store: &TensorStore,
        keys: &[String],
        query: &[f32],
        query_magnitude: f32,
    ) -> Vec<SearchResult> {
        keys.iter()
            .filter_map(|storage_key| {
                let tensor = store.get(storage_key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get("vector")?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::cosine_similarity(query, &stored_vec, query_magnitude);
                let key = storage_key
                    .strip_prefix(Self::embedding_prefix())
                    .unwrap_or(storage_key)
                    .to_string();

                Some(SearchResult::new(key, score))
            })
            .collect()
    }

    /// Parallel similarity search (faster for large datasets)
    fn search_parallel(
        store: &TensorStore,
        keys: &[String],
        query: &[f32],
        query_magnitude: f32,
    ) -> Vec<SearchResult> {
        keys.par_iter()
            .filter_map(|storage_key| {
                let tensor = store.get(storage_key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get("vector")?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::cosine_similarity(query, &stored_vec, query_magnitude);
                let key = storage_key
                    .strip_prefix(Self::embedding_prefix())
                    .unwrap_or(storage_key)
                    .to_string();

                Some(SearchResult::new(key, score))
            })
            .collect()
    }

    /// Sequential similarity search with configurable metric
    fn search_sequential_with_metric(
        store: &TensorStore,
        keys: &[String],
        query: &[f32],
        query_magnitude: f32,
        metric: DistanceMetric,
    ) -> Vec<SearchResult> {
        keys.iter()
            .filter_map(|storage_key| {
                let tensor = store.get(storage_key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get("vector")?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::compute_score(query, &stored_vec, query_magnitude, metric);
                let key = storage_key
                    .strip_prefix(Self::embedding_prefix())
                    .unwrap_or(storage_key)
                    .to_string();

                Some(SearchResult::new(key, score))
            })
            .collect()
    }

    /// Parallel similarity search with configurable metric
    fn search_parallel_with_metric(
        store: &TensorStore,
        keys: &[String],
        query: &[f32],
        query_magnitude: f32,
        metric: DistanceMetric,
    ) -> Vec<SearchResult> {
        keys.par_iter()
            .filter_map(|storage_key| {
                let tensor = store.get(storage_key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get("vector")?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::compute_score(query, &stored_vec, query_magnitude, metric);
                let key = storage_key
                    .strip_prefix(Self::embedding_prefix())
                    .unwrap_or(storage_key)
                    .to_string();

                Some(SearchResult::new(key, score))
            })
            .collect()
    }

    /// Compute score based on the distance metric.
    fn compute_score(
        query: &[f32],
        stored: &[f32],
        query_magnitude: f32,
        metric: DistanceMetric,
    ) -> f32 {
        match metric {
            DistanceMetric::Cosine => Self::cosine_similarity(query, stored, query_magnitude),
            DistanceMetric::DotProduct => simd::dot_product(query, stored),
            DistanceMetric::Euclidean => {
                // Convert distance to similarity: 1 / (1 + distance)
                let dist = Self::euclidean_distance(query, stored);
                1.0 / (1.0 + dist)
            },
        }
    }

    /// Compute Euclidean distance between two vectors.
    fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        // Sum of squared differences
        let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
        sum_sq.sqrt()
    }

    /// Compute cosine similarity between two vectors using SIMD.
    /// Optimized version that accepts pre-computed query magnitude.
    fn cosine_similarity(a: &[f32], b: &[f32], a_magnitude: f32) -> f32 {
        let dot_product = simd::dot_product(a, b);
        let b_magnitude = simd::magnitude(b);

        if b_magnitude == 0.0 {
            return 0.0;
        }

        dot_product / (a_magnitude * b_magnitude)
    }

    /// Compute the magnitude (L2 norm) of a vector using SIMD.
    fn magnitude(v: &[f32]) -> f32 {
        simd::magnitude(v)
    }

    /// Compute cosine similarity between two vectors (public helper).
    pub fn compute_similarity(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.is_empty() || b.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if a.len() != b.len() {
            return Err(VectorError::DimensionMismatch {
                expected: a.len(),
                got: b.len(),
            });
        }

        let a_magnitude = Self::magnitude(a);
        if a_magnitude == 0.0 {
            return Ok(0.0);
        }

        Ok(Self::cosine_similarity(a, b, a_magnitude))
    }

    /// Get the dimension of stored embeddings (from first embedding found).
    pub fn dimension(&self) -> Option<usize> {
        let keys = self.store.scan(Self::embedding_prefix());
        for key in keys {
            if let Ok(tensor) = self.store.get(&key) {
                if let Some(vec) = tensor.get("vector").and_then(Self::extract_vector) {
                    return Some(vec.len());
                }
            }
        }
        None
    }

    /// List all embedding keys.
    pub fn list_keys(&self) -> Vec<String> {
        self.store
            .scan(Self::embedding_prefix())
            .into_iter()
            .filter_map(|k| k.strip_prefix(Self::embedding_prefix()).map(String::from))
            .collect()
    }

    /// Clear all embeddings.
    pub fn clear(&self) -> Result<()> {
        let keys = self.store.scan(Self::embedding_prefix());
        for key in keys {
            self.store.delete(&key)?;
        }
        Ok(())
    }

    /// Build an HNSW index from all stored embeddings.
    ///
    /// Returns a tuple of (index, key_mapping) where key_mapping maps node IDs to keys.
    /// Use this for fast approximate nearest neighbor search on large datasets.
    ///
    /// # Example
    /// ```ignore
    /// let engine = VectorEngine::new();
    /// // ... store embeddings ...
    /// let (index, keys) = engine.build_hnsw_index(HNSWConfig::default())?;
    /// let results = index.search(&query, 10);
    /// for (node_id, score) in results {
    ///     println!("Key: {}, Score: {}", keys[node_id], score);
    /// }
    /// ```
    pub fn build_hnsw_index(&self, config: HNSWConfig) -> Result<(HNSWIndex, Vec<String>)> {
        let keys = self.list_keys();
        let index = HNSWIndex::with_config(config);
        let mut key_mapping = Vec::with_capacity(keys.len());

        for key in keys {
            let vector = self.get_embedding(&key)?;
            index.insert(vector);
            key_mapping.push(key);
        }

        Ok((index, key_mapping))
    }

    /// Build an HNSW index with default configuration.
    pub fn build_hnsw_index_default(&self) -> Result<(HNSWIndex, Vec<String>)> {
        self.build_hnsw_index(HNSWConfig::default())
    }

    /// This is a convenience method that converts node IDs back to SearchResults.
    pub fn search_with_hnsw(
        &self,
        index: &HNSWIndex,
        key_mapping: &[String],
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        let results = index.search(query, top_k);
        Ok(results
            .into_iter()
            .filter_map(|(node_id, score)| {
                key_mapping.get(node_id).map(|key| SearchResult {
                    key: key.clone(),
                    score,
                })
            })
            .collect())
    }

    // ========== Unified Entity Mode ==========
    // These methods work with entity keys directly (e.g., "user:1") and use the
    // _embedding field, enabling cross-engine queries on shared entities.

    /// Store embedding in an entity's _embedding field. Creates entity if needed.
    ///
    /// Automatically uses sparse format for vectors with >50% zeros.
    pub fn set_entity_embedding(&self, entity_key: &str, vector: Vec<f32>) -> Result<()> {
        if vector.is_empty() {
            return Err(VectorError::EmptyVector);
        }

        let mut tensor = self
            .store
            .get(entity_key)
            .unwrap_or_else(|_| TensorData::new());

        // Detect sparse vectors and store in optimal format
        let storage = if Self::should_use_sparse(&vector) {
            TensorValue::Sparse(SparseVector::from_dense(&vector))
        } else {
            TensorValue::Vector(vector)
        };
        tensor.set(fields::EMBEDDING, storage);
        self.store.put(entity_key, tensor)?;
        Ok(())
    }

    /// Get embedding from an entity's _embedding field.
    ///
    /// Returns the embedding as a dense vector regardless of storage format.
    pub fn get_entity_embedding(&self, entity_key: &str) -> Result<Vec<f32>> {
        let tensor = self
            .store
            .get(entity_key)
            .map_err(|_| VectorError::NotFound(entity_key.to_string()))?;

        match tensor.get(fields::EMBEDDING) {
            Some(TensorValue::Vector(v)) => Ok(v.clone()),
            Some(TensorValue::Sparse(s)) => Ok(s.to_dense()),
            _ => Err(VectorError::NotFound(entity_key.to_string())),
        }
    }

    /// Check if an entity has an embedding.
    pub fn entity_has_embedding(&self, entity_key: &str) -> bool {
        self.store
            .get(entity_key)
            .map(|t| t.has(fields::EMBEDDING))
            .unwrap_or(false)
    }

    /// Remove embedding from an entity (keeps other entity data).
    pub fn remove_entity_embedding(&self, entity_key: &str) -> Result<()> {
        let mut tensor = self
            .store
            .get(entity_key)
            .map_err(|_| VectorError::NotFound(entity_key.to_string()))?;

        if tensor.remove(fields::EMBEDDING).is_none() {
            return Err(VectorError::NotFound(entity_key.to_string()));
        }

        self.store.put(entity_key, tensor)?;
        Ok(())
    }

    /// Search for similar entities using _embedding field.
    pub fn search_entities(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if query.is_empty() {
            return Err(VectorError::EmptyVector);
        }
        if top_k == 0 {
            return Err(VectorError::InvalidTopK);
        }

        let query_magnitude = Self::magnitude(query);
        if query_magnitude == 0.0 {
            return Ok(Vec::new());
        }

        let keys = self.store.scan("");

        let mut results: Vec<SearchResult> = keys
            .iter()
            .filter_map(|key| {
                let tensor = self.store.get(key).ok()?;
                let stored_vec = Self::extract_vector(tensor.get(fields::EMBEDDING)?)?;

                if stored_vec.len() != query.len() {
                    return None;
                }

                let score = Self::cosine_similarity(query, &stored_vec, query_magnitude);
                Some(SearchResult::new(key.clone(), score))
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);

        Ok(results)
    }

    /// Scan for all entity keys that have embeddings.
    pub fn scan_entities_with_embeddings(&self) -> Vec<String> {
        self.store
            .scan("")
            .into_iter()
            .filter(|key| self.entity_has_embedding(key))
            .collect()
    }

    /// Count entities with embeddings (unified mode).
    pub fn count_entities_with_embeddings(&self) -> usize {
        self.scan_entities_with_embeddings().len()
    }
}

impl Default for VectorEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vector(dim: usize, seed: usize) -> Vec<f32> {
        // Create vectors where each seed produces a unique direction
        // by using a prime-based hash-like function
        (0..dim)
            .map(|i| {
                let x = (seed * 31 + i * 17) as f32;
                (x * 0.0001).sin() * ((seed + i) as f32 * 0.001)
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
    fn store_and_retrieve_embedding() {
        let engine = VectorEngine::new();
        let vector = vec![1.0, 2.0, 3.0];

        engine.store_embedding("test", vector.clone()).unwrap();

        let retrieved = engine.get_embedding("test").unwrap();
        assert_eq!(retrieved, vector);
    }

    #[test]
    fn store_overwrites_existing() {
        let engine = VectorEngine::new();

        engine.store_embedding("key", vec![1.0, 2.0]).unwrap();
        engine.store_embedding("key", vec![3.0, 4.0]).unwrap();

        let retrieved = engine.get_embedding("key").unwrap();
        assert_eq!(retrieved, vec![3.0, 4.0]);
    }

    #[test]
    fn delete_embedding() {
        let engine = VectorEngine::new();

        engine.store_embedding("key", vec![1.0, 2.0]).unwrap();
        assert!(engine.exists("key"));

        engine.delete_embedding("key").unwrap();
        assert!(!engine.exists("key"));
    }

    #[test]
    fn delete_nonexistent_returns_error() {
        let engine = VectorEngine::new();

        let result = engine.delete_embedding("nonexistent");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn get_nonexistent_returns_error() {
        let engine = VectorEngine::new();

        let result = engine.get_embedding("nonexistent");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn empty_vector_returns_error() {
        let engine = VectorEngine::new();

        let result = engine.store_embedding("key", vec![]);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn count_embeddings() {
        let engine = VectorEngine::new();

        assert_eq!(engine.count(), 0);

        engine.store_embedding("a", vec![1.0]).unwrap();
        engine.store_embedding("b", vec![2.0]).unwrap();
        engine.store_embedding("c", vec![3.0]).unwrap();

        assert_eq!(engine.count(), 3);
    }

    #[test]
    fn search_similar_basic() {
        let engine = VectorEngine::new();

        // Store some vectors
        engine.store_embedding("a", vec![1.0, 0.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![1.0, 1.0, 0.0]).unwrap();

        // Search for similar to [1, 0, 0]
        let results = engine.search_similar(&[1.0, 0.0, 0.0], 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be "a" with score 1.0 (exact match)
        assert_eq!(results[0].key, "a");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_similar_top_k() {
        let engine = VectorEngine::new();

        for i in 0..10 {
            engine
                .store_embedding(&format!("v{}", i), vec![i as f32, 0.0])
                .unwrap();
        }

        let results = engine.search_similar(&[5.0, 0.0], 3).unwrap();

        // Should return exactly 3 results
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn search_similar_fewer_than_k() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.0, 1.0]).unwrap();

        let results = engine.search_similar(&[1.0, 0.0], 10).unwrap();

        // Should return only 2 (all available)
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_similar_empty_query_error() {
        let engine = VectorEngine::new();

        let result = engine.search_similar(&[], 5);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_similar_zero_top_k_error() {
        let engine = VectorEngine::new();

        let result = engine.search_similar(&[1.0, 0.0], 0);
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    #[test]
    fn cosine_similarity_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let score = VectorEngine::compute_similarity(&v, &v).unwrap();
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let score = VectorEngine::compute_similarity(&a, &b).unwrap();
        assert!(score.abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let score = VectorEngine::compute_similarity(&a, &b).unwrap();
        assert!((score - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_normalized_vectors() {
        // cos(45deg) = sqrt(2)/2 ≈ 0.707
        let a = normalize(&[1.0, 0.0]);
        let b = normalize(&[1.0, 1.0]);
        let score = VectorEngine::compute_similarity(&a, &b).unwrap();
        let expected = (2.0_f32).sqrt() / 2.0;
        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_dimension_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = VectorEngine::compute_similarity(&a, &b);
        assert!(matches!(result, Err(VectorError::DimensionMismatch { .. })));
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let score = VectorEngine::compute_similarity(&a, &b).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn search_skips_dimension_mismatch() {
        let engine = VectorEngine::new();

        engine.store_embedding("2d", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("3d", vec![1.0, 0.0, 0.0]).unwrap();

        // Search with 2D query - should only match 2D vector
        let results = engine.search_similar(&[1.0, 0.0], 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "2d");
    }

    #[test]
    fn store_10000_vectors_search() {
        let engine = VectorEngine::new();
        let dim = 128;

        // Store 10,000 vectors
        for i in 0..10000 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        assert_eq!(engine.count(), 10000);

        // Search for similar to a known vector
        let query = create_test_vector(dim, 5000);
        let results = engine.search_similar(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // The most similar should be the exact match
        assert_eq!(results[0].key, "v5000");
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn high_dimensional_768() {
        let engine = VectorEngine::new();
        let dim = 768;

        // Store a few high-dimensional vectors
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        let query = create_test_vector(dim, 50);
        let results = engine.search_similar(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "v50");
    }

    #[test]
    fn high_dimensional_1536() {
        let engine = VectorEngine::new();
        let dim = 1536;

        // Store a few high-dimensional vectors (OpenAI embedding size)
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        let query = create_test_vector(dim, 75);
        let results = engine.search_similar(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].key, "v75");
    }

    #[test]
    fn similarity_scores_mathematically_correct() {
        let engine = VectorEngine::new();

        // Store vectors with known similarity relationships
        // Using normalized vectors for easier verification
        engine
            .store_embedding("unit_x", normalize(&[1.0, 0.0, 0.0]))
            .unwrap();
        engine
            .store_embedding("unit_y", normalize(&[0.0, 1.0, 0.0]))
            .unwrap();
        engine
            .store_embedding("unit_z", normalize(&[0.0, 0.0, 1.0]))
            .unwrap();
        engine
            .store_embedding("diag_xy", normalize(&[1.0, 1.0, 0.0]))
            .unwrap();
        engine
            .store_embedding("neg_x", normalize(&[-1.0, 0.0, 0.0]))
            .unwrap();

        let query = normalize(&[1.0, 0.0, 0.0]);
        let results = engine.search_similar(&query, 5).unwrap();

        // Verify each score mathematically
        for result in &results {
            match result.key.as_str() {
                "unit_x" => assert!((result.score - 1.0).abs() < 1e-6), // cos(0) = 1
                "unit_y" | "unit_z" => assert!(result.score.abs() < 1e-6), // cos(90) = 0
                "diag_xy" => {
                    let expected = (2.0_f32).sqrt() / 2.0; // cos(45) ≈ 0.707
                    assert!((result.score - expected).abs() < 1e-6);
                },
                "neg_x" => assert!((result.score - (-1.0)).abs() < 1e-6), // cos(180) = -1
                _ => panic!("Unexpected key: {}", result.key),
            }
        }
    }

    #[test]
    fn list_keys() {
        let engine = VectorEngine::new();

        engine.store_embedding("alpha", vec![1.0]).unwrap();
        engine.store_embedding("beta", vec![2.0]).unwrap();
        engine.store_embedding("gamma", vec![3.0]).unwrap();

        let mut keys = engine.list_keys();
        keys.sort();

        assert_eq!(keys, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn clear_all() {
        let engine = VectorEngine::new();

        engine.store_embedding("a", vec![1.0]).unwrap();
        engine.store_embedding("b", vec![2.0]).unwrap();

        assert_eq!(engine.count(), 2);

        engine.clear().unwrap();

        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn dimension() {
        let engine = VectorEngine::new();

        assert_eq!(engine.dimension(), None);

        engine.store_embedding("test", vec![1.0, 2.0, 3.0]).unwrap();

        assert_eq!(engine.dimension(), Some(3));
    }

    #[test]
    fn default_trait() {
        let engine = VectorEngine::default();
        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn with_store_constructor() {
        let store = TensorStore::new();
        let engine = VectorEngine::with_store(store);
        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn error_display() {
        assert_eq!(
            format!("{}", VectorError::NotFound("test".into())),
            "Embedding not found: test"
        );
        assert_eq!(
            format!(
                "{}",
                VectorError::DimensionMismatch {
                    expected: 3,
                    got: 5
                }
            ),
            "Dimension mismatch: expected 3, got 5"
        );
        assert_eq!(
            format!("{}", VectorError::EmptyVector),
            "Empty vector provided"
        );
        assert_eq!(
            format!("{}", VectorError::InvalidTopK),
            "Invalid top_k value (must be > 0)"
        );
        assert_eq!(
            format!("{}", VectorError::StorageError("test".into())),
            "Storage error: test"
        );
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = VectorError::NotFound("test".into());
        let e2 = e1.clone();
        assert_eq!(e1, e2);
    }

    #[test]
    fn error_is_error_trait() {
        let error: Box<dyn std::error::Error> = Box::new(VectorError::EmptyVector);
        assert!(error.to_string().contains("Empty"));
    }

    #[test]
    fn search_result_clone_and_eq() {
        let r1 = SearchResult::new("key".into(), 0.5);
        let r2 = r1.clone();
        assert_eq!(r1, r2);
    }

    #[test]
    fn search_zero_query_vector() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();

        // Zero vector query
        let results = engine.search_similar(&[0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_no_embeddings() {
        let engine = VectorEngine::new();
        let results = engine.search_similar(&[1.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn exists_check() {
        let engine = VectorEngine::new();

        assert!(!engine.exists("key"));
        engine.store_embedding("key", vec![1.0]).unwrap();
        assert!(engine.exists("key"));
    }

    // HNSW tests

    #[test]
    fn hnsw_basic_insert_and_search() {
        let index = HNSWIndex::new();

        // Insert a few vectors
        index.insert(vec![1.0, 0.0, 0.0]);
        index.insert(vec![0.0, 1.0, 0.0]);
        index.insert(vec![0.0, 0.0, 1.0]);

        assert_eq!(index.len(), 3);

        // Search for similar to [1, 0, 0]
        let results = index.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);

        // First result should have highest similarity to [1, 0, 0]
        assert_eq!(results[0].0, 0); // node 0 is [1, 0, 0]
        assert!((results[0].1 - 1.0).abs() < 1e-6); // similarity ≈ 1.0
    }

    #[test]
    fn hnsw_empty_search() {
        let index = HNSWIndex::new();
        let results = index.search(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn hnsw_many_vectors() {
        let index = HNSWIndex::new();
        let dim = 64;

        // Insert 1000 vectors
        for i in 0..1000 {
            let vector = create_test_vector(dim, i);
            index.insert(vector);
        }

        assert_eq!(index.len(), 1000);

        // Search for a known vector
        let query = create_test_vector(dim, 500);
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);
        // HNSW is approximate - the exact match should be in top results with high score
        // Find node 500 in results
        let exact_match = results.iter().find(|(id, _)| *id == 500);
        assert!(
            exact_match.is_some(),
            "Expected to find node 500 in top 10 results"
        );
        // The similarity should be very high (close to 1.0)
        assert!(exact_match.unwrap().1 > 0.99);
    }

    #[test]
    fn hnsw_high_recall() {
        let config = HNSWConfig::high_recall();
        let index = HNSWIndex::with_config(config);
        let dim = 32;

        // Insert vectors
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            index.insert(vector);
        }

        // High recall config should find exact match
        let query = create_test_vector(dim, 42);
        let results = index.search(&query, 1);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn hnsw_high_speed() {
        let config = HNSWConfig::high_speed();
        let index = HNSWIndex::with_config(config);
        let dim = 32;

        // Insert vectors
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            index.insert(vector);
        }

        assert_eq!(index.len(), 100);
    }

    #[test]
    fn hnsw_get_vector() {
        let index = HNSWIndex::new();

        let original = vec![1.0, 2.0, 3.0];
        let id = index.insert(original.clone());

        let retrieved = index.get_vector(id);
        assert_eq!(retrieved, Some(original));
    }

    #[test]
    fn hnsw_search_with_ef() {
        let index = HNSWIndex::new();
        let dim = 32;

        for i in 0..100 {
            index.insert(create_test_vector(dim, i));
        }

        let query = create_test_vector(dim, 50);

        // Higher ef should give better recall
        let results_low = index.search_with_ef(&query, 5, 10);
        let results_high = index.search_with_ef(&query, 5, 200);

        assert_eq!(results_low.len(), 5);
        assert_eq!(results_high.len(), 5);

        // Both should find the exact match in top 5
        assert!(results_high.iter().any(|(id, _)| *id == 50));
    }

    #[test]
    fn hnsw_default_trait() {
        let index = HNSWIndex::default();
        assert!(index.is_empty());
    }

    #[test]
    fn hnsw_config_default() {
        let config = HNSWConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m0, 32);
        assert_eq!(config.ef_construction, 200);
        assert_eq!(config.ef_search, 50);
    }

    #[test]
    fn engine_build_hnsw_index() {
        let engine = VectorEngine::new();
        let dim = 32;

        // Store some vectors
        for i in 0..50 {
            let vector = create_test_vector(dim, i);
            engine.store_embedding(&format!("v{}", i), vector).unwrap();
        }

        // Build HNSW index
        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();

        assert_eq!(index.len(), 50);
        assert_eq!(key_mapping.len(), 50);
    }

    #[test]
    fn engine_search_with_hnsw() {
        let engine = VectorEngine::new();
        let dim = 32;

        // Store vectors
        for i in 0..100 {
            let vector = create_test_vector(dim, i);
            engine
                .store_embedding(&format!("vec{}", i), vector)
                .unwrap();
        }

        // Build index
        let (index, key_mapping) = engine.build_hnsw_index_default().unwrap();

        // Search
        let query = create_test_vector(dim, 42);
        let results = engine
            .search_with_hnsw(&index, &key_mapping, &query, 5)
            .unwrap();

        assert_eq!(results.len(), 5);
        // Should find the exact match
        assert!(results.iter().any(|r| r.key.contains("42")));
    }

    #[test]
    fn engine_hnsw_empty_query_error() {
        let engine = VectorEngine::new();
        let index = HNSWIndex::new();
        let key_mapping: Vec<String> = vec![];

        let result = engine.search_with_hnsw(&index, &key_mapping, &[], 5);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn engine_hnsw_zero_top_k_error() {
        let engine = VectorEngine::new();
        let index = HNSWIndex::new();
        let key_mapping: Vec<String> = vec![];

        let result = engine.search_with_hnsw(&index, &key_mapping, &[1.0], 0);
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    // Unified Entity Mode tests

    #[test]
    fn entity_embedding_store_and_retrieve() {
        let engine = VectorEngine::new();
        let vector = vec![1.0, 2.0, 3.0];

        engine
            .set_entity_embedding("user:1", vector.clone())
            .unwrap();

        let retrieved = engine.get_entity_embedding("user:1").unwrap();
        assert_eq!(retrieved, vector);
    }

    #[test]
    fn entity_embedding_preserves_other_fields() {
        let store = TensorStore::new();

        let mut data = TensorData::new();
        data.set(
            "name",
            TensorValue::Scalar(tensor_store::ScalarValue::String("Alice".into())),
        );
        store.put("user:1", data).unwrap();

        let engine = VectorEngine::with_store(store);
        engine
            .set_entity_embedding("user:1", vec![0.1, 0.2])
            .unwrap();

        let tensor = engine.store.get("user:1").unwrap();
        assert!(tensor.has("name"));
        assert!(tensor.has(fields::EMBEDDING));
    }

    #[test]
    fn entity_has_embedding_check() {
        let engine = VectorEngine::new();

        assert!(!engine.entity_has_embedding("user:1"));

        engine
            .set_entity_embedding("user:1", vec![1.0, 2.0])
            .unwrap();
        assert!(engine.entity_has_embedding("user:1"));
    }

    #[test]
    fn entity_embedding_remove() {
        let engine = VectorEngine::new();

        engine
            .set_entity_embedding("user:1", vec![1.0, 2.0])
            .unwrap();
        assert!(engine.entity_has_embedding("user:1"));

        engine.remove_entity_embedding("user:1").unwrap();
        assert!(!engine.entity_has_embedding("user:1"));
    }

    #[test]
    fn entity_embedding_remove_nonexistent_error() {
        let engine = VectorEngine::new();
        let result = engine.remove_entity_embedding("user:999");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn entity_embedding_get_nonexistent_error() {
        let engine = VectorEngine::new();
        let result = engine.get_entity_embedding("user:999");
        assert!(matches!(result, Err(VectorError::NotFound(_))));
    }

    #[test]
    fn entity_embedding_empty_vector_error() {
        let engine = VectorEngine::new();
        let result = engine.set_entity_embedding("user:1", vec![]);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_entities_basic() {
        let engine = VectorEngine::new();

        engine
            .set_entity_embedding("user:1", vec![1.0, 0.0, 0.0])
            .unwrap();
        engine
            .set_entity_embedding("user:2", vec![0.0, 1.0, 0.0])
            .unwrap();
        engine
            .set_entity_embedding("user:3", vec![1.0, 1.0, 0.0])
            .unwrap();

        let results = engine.search_entities(&[1.0, 0.0, 0.0], 3).unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].key, "user:1");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_entities_filters_non_embeddings() {
        let store = TensorStore::new();

        let mut user1 = TensorData::new();
        user1.set(fields::EMBEDDING, TensorValue::Vector(vec![1.0, 0.0]));
        store.put("user:1", user1).unwrap();

        let mut user2 = TensorData::new();
        user2.set(
            "name",
            TensorValue::Scalar(tensor_store::ScalarValue::String("Bob".into())),
        );
        store.put("user:2", user2).unwrap();

        let engine = VectorEngine::with_store(store);
        let results = engine.search_entities(&[1.0, 0.0], 10).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "user:1");
    }

    #[test]
    fn scan_entities_with_embeddings() {
        let engine = VectorEngine::new();

        engine
            .set_entity_embedding("user:1", vec![1.0, 2.0])
            .unwrap();
        engine
            .set_entity_embedding("user:2", vec![3.0, 4.0])
            .unwrap();

        let keys = engine.scan_entities_with_embeddings();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn count_entities_with_embeddings() {
        let engine = VectorEngine::new();

        assert_eq!(engine.count_entities_with_embeddings(), 0);

        engine.set_entity_embedding("user:1", vec![1.0]).unwrap();
        engine.set_entity_embedding("user:2", vec![2.0]).unwrap();

        assert_eq!(engine.count_entities_with_embeddings(), 2);
    }

    #[test]
    fn search_entities_empty_query_error() {
        let engine = VectorEngine::new();
        let result = engine.search_entities(&[], 5);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_entities_zero_top_k_error() {
        let engine = VectorEngine::new();
        let result = engine.search_entities(&[1.0], 0);
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    #[test]
    fn search_entities_zero_query_returns_empty() {
        let engine = VectorEngine::new();
        engine
            .set_entity_embedding("user:1", vec![1.0, 0.0])
            .unwrap();

        let results = engine.search_entities(&[0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn distance_metric_default() {
        assert_eq!(DistanceMetric::default(), DistanceMetric::Cosine);
    }

    #[test]
    fn search_with_metric_cosine() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![0.707, 0.707]).unwrap();
        engine.store_embedding("c", vec![0.0, 1.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[1.0, 0.0], 3, DistanceMetric::Cosine)
            .unwrap();

        assert_eq!(results.len(), 3);
        // Cosine: identical vector should have highest score
        assert_eq!(results[0].key, "a");
        assert!((results[0].score - 1.0).abs() < 0.01);
    }

    #[test]
    fn search_with_metric_dot_product() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![2.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![0.5, 0.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[1.0, 0.0], 3, DistanceMetric::DotProduct)
            .unwrap();

        assert_eq!(results.len(), 3);
        // Dot product: larger magnitude vector has higher score
        assert_eq!(results[0].key, "b");
        assert!((results[0].score - 2.0).abs() < 0.01);
    }

    #[test]
    fn search_with_metric_euclidean() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("b", vec![2.0, 0.0]).unwrap();
        engine.store_embedding("c", vec![10.0, 0.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[1.0, 0.0], 3, DistanceMetric::Euclidean)
            .unwrap();

        assert_eq!(results.len(), 3);
        // Euclidean: closest vector (distance 0) should have highest score
        assert_eq!(results[0].key, "a");
        // Score = 1 / (1 + 0) = 1.0
        assert!((results[0].score - 1.0).abs() < 0.01);
        // "b" is distance 1, score = 1 / (1 + 1) = 0.5
        assert_eq!(results[1].key, "b");
        assert!((results[1].score - 0.5).abs() < 0.01);
    }

    #[test]
    fn search_with_metric_empty_query_error() {
        let engine = VectorEngine::new();
        let result = engine.search_similar_with_metric(&[], 5, DistanceMetric::Cosine);
        assert!(matches!(result, Err(VectorError::EmptyVector)));
    }

    #[test]
    fn search_with_metric_zero_top_k_error() {
        let engine = VectorEngine::new();
        let result = engine.search_similar_with_metric(&[1.0], 0, DistanceMetric::Cosine);
        assert!(matches!(result, Err(VectorError::InvalidTopK)));
    }

    #[test]
    fn search_with_metric_zero_query() {
        let engine = VectorEngine::new();
        engine.store_embedding("a", vec![1.0, 0.0]).unwrap();

        let results = engine
            .search_similar_with_metric(&[0.0, 0.0], 5, DistanceMetric::Cosine)
            .unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_with_metric_zero_query_euclidean() {
        let engine = VectorEngine::new();
        engine.store_embedding("origin", vec![0.0, 0.0]).unwrap();
        engine.store_embedding("unit", vec![1.0, 0.0]).unwrap();
        engine.store_embedding("far", vec![10.0, 0.0]).unwrap();

        // Zero query should work for Euclidean (finds vectors closest to origin)
        let results = engine
            .search_similar_with_metric(&[0.0, 0.0], 3, DistanceMetric::Euclidean)
            .unwrap();

        assert_eq!(results.len(), 3);
        // Origin is closest to [0,0] (distance 0, score 1.0)
        assert_eq!(results[0].key, "origin");
        assert!((results[0].score - 1.0).abs() < 0.01);
        // Unit is next (distance 1, score 0.5)
        assert_eq!(results[1].key, "unit");
        assert!((results[1].score - 0.5).abs() < 0.01);
    }

    #[test]
    fn euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let dist = VectorEngine::euclidean_distance(&a, &a);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance_unit() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let dist = VectorEngine::euclidean_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn euclidean_distance_pythagoras() {
        // 3-4-5 triangle
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = VectorEngine::euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    // Sparse vector tests

    #[test]
    fn sparse_vector_storage_and_retrieval() {
        let engine = VectorEngine::new();

        // Create a sparse vector (>50% zeros)
        let mut sparse = vec![0.0f32; 100];
        sparse[0] = 1.0;
        sparse[50] = 2.0;
        sparse[99] = 3.0;

        engine.store_embedding("sparse", sparse.clone()).unwrap();
        let retrieved = engine.get_embedding("sparse").unwrap();

        assert_eq!(retrieved.len(), sparse.len());
        assert_eq!(retrieved[0], 1.0);
        assert_eq!(retrieved[50], 2.0);
        assert_eq!(retrieved[99], 3.0);
    }

    #[test]
    fn sparse_vector_search() {
        let engine = VectorEngine::new();

        // Create sparse vectors with 97% zeros (3 non-zeros in 100 elements)
        let mut v1 = vec![0.0f32; 100];
        v1[0] = 1.0;
        v1[1] = 0.0;
        v1[2] = 0.0;

        let mut v2 = vec![0.0f32; 100];
        v2[0] = 0.707;
        v2[1] = 0.707;

        let mut v3 = vec![0.0f32; 100];
        v3[0] = 0.0;
        v3[1] = 1.0;

        engine.store_embedding("v1", v1).unwrap();
        engine.store_embedding("v2", v2).unwrap();
        engine.store_embedding("v3", v3).unwrap();

        // Query with [1, 0, 0, ...]
        let mut query = vec![0.0f32; 100];
        query[0] = 1.0;

        let results = engine.search_similar(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        // v1 should be the best match
        assert_eq!(results[0].key, "v1");
        assert!((results[0].score - 1.0).abs() < 0.01);
    }

    #[test]
    fn sparse_entity_embedding() {
        let engine = VectorEngine::new();

        let mut sparse = vec![0.0f32; 100];
        sparse[10] = 5.0;
        sparse[20] = -3.0;

        engine
            .set_entity_embedding("entity:1", sparse.clone())
            .unwrap();
        let retrieved = engine.get_entity_embedding("entity:1").unwrap();

        assert_eq!(retrieved.len(), 100);
        assert_eq!(retrieved[10], 5.0);
        assert_eq!(retrieved[20], -3.0);
        assert_eq!(retrieved[0], 0.0);
    }

    #[test]
    fn sparse_detection_threshold() {
        // Exactly 50% zeros should use sparse
        let half_sparse: Vec<f32> = (0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect();
        assert!(VectorEngine::should_use_sparse(&half_sparse));

        // Less than 50% zeros should use dense
        let mostly_dense: Vec<f32> = (0..100).map(|i| if i < 40 { 0.0 } else { 1.0 }).collect();
        assert!(!VectorEngine::should_use_sparse(&mostly_dense));

        // 97% zeros should use sparse
        let very_sparse: Vec<f32> = (0..100).map(|i| if i < 3 { 1.0 } else { 0.0 }).collect();
        assert!(VectorEngine::should_use_sparse(&very_sparse));
    }

    #[test]
    fn sparse_search_with_metric() {
        let engine = VectorEngine::new();

        // Sparse vectors
        let mut v1 = vec![0.0f32; 100];
        v1[0] = 1.0;

        let mut v2 = vec![0.0f32; 100];
        v2[0] = 2.0;

        engine.store_embedding("v1", v1).unwrap();
        engine.store_embedding("v2", v2).unwrap();

        let mut query = vec![0.0f32; 100];
        query[0] = 1.0;

        let results = engine
            .search_similar_with_metric(&query, 2, DistanceMetric::Euclidean)
            .unwrap();

        assert_eq!(results.len(), 2);
        // v1 is closest (distance 0)
        assert_eq!(results[0].key, "v1");
    }

    #[test]
    fn search_entities_with_sparse() {
        let engine = VectorEngine::new();

        // Create sparse entity embeddings
        let mut e1 = vec![0.0f32; 100];
        e1[0] = 1.0;

        let mut e2 = vec![0.0f32; 100];
        e2[1] = 1.0;

        engine.set_entity_embedding("user:1", e1).unwrap();
        engine.set_entity_embedding("user:2", e2).unwrap();

        let mut query = vec![0.0f32; 100];
        query[0] = 1.0;

        let results = engine.search_entities(&query, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, "user:1");
        assert!((results[0].score - 1.0).abs() < 0.01);
    }
}
