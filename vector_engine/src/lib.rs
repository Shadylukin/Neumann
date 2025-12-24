//! Vector Engine - Module 4 of Neumann
//!
//! Provides embeddings storage and similarity search functionality.

use tensor_store::{TensorData, TensorStore, TensorStoreError, TensorValue};

/// Error types for vector operations.
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
    /// Overwrites any existing embedding with the same key.
    pub fn store_embedding(&self, key: &str, vector: Vec<f32>) -> Result<()> {
        if vector.is_empty() {
            return Err(VectorError::EmptyVector);
        }

        let storage_key = Self::embedding_key(key);
        let mut tensor = TensorData::new();
        tensor.set("vector", TensorValue::Vector(vector));

        self.store.put(storage_key, tensor)?;
        Ok(())
    }

    /// Get an embedding by key.
    pub fn get_embedding(&self, key: &str) -> Result<Vec<f32>> {
        let storage_key = Self::embedding_key(key);
        let tensor = self
            .store
            .get(&storage_key)
            .map_err(|_| VectorError::NotFound(key.to_string()))?;

        match tensor.get("vector") {
            Some(TensorValue::Vector(v)) => Ok(v.clone()),
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

    /// Search for the top_k most similar embeddings to the query vector.
    ///
    /// Returns results sorted by similarity score (highest first).
    /// Uses cosine similarity: score = dot(a,b) / (|a| * |b|)
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
        let mut results: Vec<SearchResult> = Vec::with_capacity(keys.len());

        for storage_key in keys {
            if let Ok(tensor) = self.store.get(&storage_key) {
                if let Some(TensorValue::Vector(stored_vec)) = tensor.get("vector") {
                    // Skip if dimensions don't match
                    if stored_vec.len() != query.len() {
                        continue;
                    }

                    let score = Self::cosine_similarity(query, stored_vec, query_magnitude);

                    // Extract the original key (remove "emb:" prefix)
                    let key = storage_key
                        .strip_prefix(Self::embedding_prefix())
                        .unwrap_or(&storage_key)
                        .to_string();

                    results.push(SearchResult::new(key, score));
                }
            }
        }

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

    /// Compute cosine similarity between two vectors.
    /// Optimized version that accepts pre-computed query magnitude.
    fn cosine_similarity(a: &[f32], b: &[f32], a_magnitude: f32) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let b_magnitude = Self::magnitude(b);

        if b_magnitude == 0.0 {
            return 0.0;
        }

        dot_product / (a_magnitude * b_magnitude)
    }

    /// Compute the magnitude (L2 norm) of a vector.
    fn magnitude(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
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
                if let Some(TensorValue::Vector(v)) = tensor.get("vector") {
                    return Some(v.len());
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
}
