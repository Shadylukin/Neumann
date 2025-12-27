//! Delta-encoded vectors for efficient storage of clustered embeddings.
//!
//! Delta encoding stores vectors as differences from reference "archetype" vectors.
//! When embeddings cluster around common patterns, this provides significant compression
//! beyond sparse encoding alone.
//!
//! # Philosophy
//!
//! If many embeddings are variations of a few archetypes, storing the full vector
//! for each is wasteful. Instead:
//! - Identify archetype vectors (cluster centroids)
//! - Store each embedding as: archetype_id + sparse delta
//! - Reconstruct on demand: archetype + delta = original
//!
//! # Example
//!
//! ```
//! use tensor_store::DeltaVector;
//!
//! // Reference archetype vector
//! let archetype = vec![1.0, 0.0, 0.0, 0.0];
//!
//! // Similar vector with small differences
//! let embedding = vec![0.98, 0.02, 0.01, 0.0];
//!
//! // Create delta (only stores the differences)
//! let delta = DeltaVector::from_dense_with_reference(&embedding, &archetype, 0, 0.001);
//!
//! // Reconstruct original
//! let reconstructed = delta.to_dense(&archetype);
//! ```

use crate::SparseVector;

/// A vector stored as a delta from a reference archetype.
///
/// Memory layout: 8 (archetype_id) + 8 (dimension) + 8 (cached_magnitude) +
///                positions.len() * 2 + deltas.len() * 4 bytes
#[derive(Debug, Clone, PartialEq)]
pub struct DeltaVector {
    /// ID of the reference archetype in the archetype registry.
    archetype_id: usize,
    /// The full dimension of the vector (for reconstruction).
    dimension: usize,
    /// Positions where this vector differs from the archetype.
    positions: Vec<u16>,
    /// Delta values at each position (original - archetype).
    deltas: Vec<f32>,
    /// Cached magnitude of the reconstructed vector (for fast cosine similarity).
    cached_magnitude: Option<f32>,
}

impl DeltaVector {
    /// Create a delta vector from a dense vector and its reference archetype.
    ///
    /// Only stores positions where |original - archetype| > threshold.
    pub fn from_dense_with_reference(
        dense: &[f32],
        archetype: &[f32],
        archetype_id: usize,
        threshold: f32,
    ) -> Self {
        debug_assert_eq!(dense.len(), archetype.len());

        let mut positions = Vec::new();
        let mut deltas = Vec::new();

        for (i, (&val, &ref_val)) in dense.iter().zip(archetype.iter()).enumerate() {
            let delta = val - ref_val;
            if delta.abs() > threshold {
                positions.push(i as u16);
                deltas.push(delta);
            }
        }

        Self {
            archetype_id,
            dimension: dense.len(),
            positions,
            deltas,
            cached_magnitude: None,
        }
    }

    /// Create a delta vector with pre-computed magnitude.
    pub fn from_dense_with_reference_and_magnitude(
        dense: &[f32],
        archetype: &[f32],
        archetype_id: usize,
        threshold: f32,
    ) -> Self {
        let magnitude = dense.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut delta = Self::from_dense_with_reference(dense, archetype, archetype_id, threshold);
        delta.cached_magnitude = Some(magnitude);
        delta
    }

    /// Create from explicit components (for deserialization or testing).
    pub fn from_parts(
        archetype_id: usize,
        dimension: usize,
        positions: Vec<u16>,
        deltas: Vec<f32>,
    ) -> Self {
        debug_assert_eq!(positions.len(), deltas.len());
        Self {
            archetype_id,
            dimension,
            positions,
            deltas,
            cached_magnitude: None,
        }
    }

    /// Reconstruct the full dense vector: archetype + delta.
    pub fn to_dense(&self, archetype: &[f32]) -> Vec<f32> {
        debug_assert_eq!(archetype.len(), self.dimension);

        let mut result = archetype.to_vec();
        for (&pos, &delta) in self.positions.iter().zip(self.deltas.iter()) {
            result[pos as usize] += delta;
        }
        result
    }

    /// Convert delta to a sparse vector (the delta itself, not the full vector).
    pub fn to_sparse_delta(&self) -> SparseVector {
        SparseVector::from_parts(
            self.dimension,
            self.positions.iter().map(|&p| p as u32).collect(),
            self.deltas.clone(),
        )
    }

    /// Get the archetype ID this delta references.
    pub fn archetype_id(&self) -> usize {
        self.archetype_id
    }

    /// Get the full vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of non-zero delta positions.
    pub fn nnz(&self) -> usize {
        self.positions.len()
    }

    /// Calculate sparsity of the delta (fraction of zero deltas).
    pub fn delta_sparsity(&self) -> f32 {
        if self.dimension == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f32 / self.dimension as f32)
    }

    /// Get the cached magnitude, computing it if necessary.
    pub fn magnitude(&self, archetype: &[f32]) -> f32 {
        if let Some(mag) = self.cached_magnitude {
            return mag;
        }
        // Must reconstruct to compute magnitude
        let dense = self.to_dense(archetype);
        dense.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Set the cached magnitude (useful after reconstruction).
    pub fn set_cached_magnitude(&mut self, magnitude: f32) {
        self.cached_magnitude = Some(magnitude);
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.positions.len() * std::mem::size_of::<u16>()
            + self.deltas.len() * std::mem::size_of::<f32>()
    }

    /// Iterate over (position, delta) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u16, f32)> + '_ {
        self.positions
            .iter()
            .copied()
            .zip(self.deltas.iter().copied())
    }

    /// Dot product with a dense query vector.
    ///
    /// dot(archetype + delta, query) = dot(archetype, query) + dot(delta, query)
    ///
    /// If archetype_dot_query is precomputed, this is O(nnz) instead of O(dimension).
    pub fn dot_dense_with_precomputed(&self, query: &[f32], archetype_dot_query: f32) -> f32 {
        let delta_dot: f32 = self
            .positions
            .iter()
            .zip(self.deltas.iter())
            .map(|(&pos, &delta)| delta * query[pos as usize])
            .sum();

        archetype_dot_query + delta_dot
    }

    /// Dot product with a dense query (requires archetype for full computation).
    pub fn dot_dense(&self, query: &[f32], archetype: &[f32]) -> f32 {
        let archetype_dot: f32 = archetype.iter().zip(query.iter()).map(|(a, q)| a * q).sum();
        self.dot_dense_with_precomputed(query, archetype_dot)
    }

    /// Dot product between two delta vectors from the SAME archetype.
    ///
    /// If A = R + delta_a and B = R + delta_b, then:
    /// dot(A, B) = dot(R, R) + dot(R, delta_b) + dot(delta_a, R) + dot(delta_a, delta_b)
    ///
    /// With precomputed values, this is O(nnz_a + nnz_b) instead of O(dimension).
    pub fn dot_same_archetype(
        &self,
        other: &DeltaVector,
        archetype: &[f32],
        archetype_magnitude_sq: f32, // dot(R, R)
    ) -> f32 {
        debug_assert_eq!(self.archetype_id, other.archetype_id);

        // dot(R, delta_b)
        let r_dot_delta_b: f32 = other
            .positions
            .iter()
            .zip(other.deltas.iter())
            .map(|(&pos, &delta)| archetype[pos as usize] * delta)
            .sum();

        // dot(delta_a, R)
        let delta_a_dot_r: f32 = self
            .positions
            .iter()
            .zip(self.deltas.iter())
            .map(|(&pos, &delta)| delta * archetype[pos as usize])
            .sum();

        // dot(delta_a, delta_b) - sparse-sparse dot product
        let delta_a_dot_delta_b = self.sparse_delta_dot(other);

        archetype_magnitude_sq + r_dot_delta_b + delta_a_dot_r + delta_a_dot_delta_b
    }

    /// Sparse dot product between two delta vectors (just the deltas).
    fn sparse_delta_dot(&self, other: &DeltaVector) -> f32 {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < self.positions.len() && j < other.positions.len() {
            match self.positions[i].cmp(&other.positions[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result += self.deltas[i] * other.deltas[j];
                    i += 1;
                    j += 1;
                },
            }
        }

        result
    }

    /// Cosine similarity with a dense query.
    pub fn cosine_similarity_dense(
        &self,
        query: &[f32],
        archetype: &[f32],
        query_magnitude: f32,
    ) -> f32 {
        let dot = self.dot_dense(query, archetype);
        let self_magnitude = self.magnitude(archetype);

        if self_magnitude == 0.0 || query_magnitude == 0.0 {
            return 0.0;
        }

        dot / (self_magnitude * query_magnitude)
    }

    /// Cosine similarity with precomputed archetype dot product.
    pub fn cosine_similarity_dense_precomputed(
        &self,
        query: &[f32],
        archetype_dot_query: f32,
        self_magnitude: f32,
        query_magnitude: f32,
    ) -> f32 {
        if self_magnitude == 0.0 || query_magnitude == 0.0 {
            return 0.0;
        }

        let dot = self.dot_dense_with_precomputed(query, archetype_dot_query);
        dot / (self_magnitude * query_magnitude)
    }

    /// Compression ratio compared to storing the full dense vector.
    pub fn compression_ratio(&self) -> f32 {
        let dense_bytes = self.dimension * std::mem::size_of::<f32>();
        if dense_bytes == 0 {
            return 1.0;
        }
        dense_bytes as f32 / self.memory_bytes() as f32
    }
}

/// Registry for archetype vectors used as references for delta encoding.
#[derive(Debug, Clone)]
pub struct ArchetypeRegistry {
    /// Stored archetype vectors.
    archetypes: Vec<Vec<f32>>,
    /// Precomputed magnitude squared for each archetype.
    magnitude_sq: Vec<f32>,
    /// Maximum number of archetypes to store.
    max_archetypes: usize,
}

impl ArchetypeRegistry {
    /// Create a new archetype registry.
    pub fn new(max_archetypes: usize) -> Self {
        Self {
            archetypes: Vec::new(),
            magnitude_sq: Vec::new(),
            max_archetypes,
        }
    }

    /// Register a new archetype, returning its ID.
    pub fn register(&mut self, archetype: Vec<f32>) -> Option<usize> {
        if self.archetypes.len() >= self.max_archetypes {
            return None;
        }

        let mag_sq: f32 = archetype.iter().map(|x| x * x).sum();
        let id = self.archetypes.len();
        self.archetypes.push(archetype);
        self.magnitude_sq.push(mag_sq);
        Some(id)
    }

    /// Get an archetype by ID.
    pub fn get(&self, id: usize) -> Option<&[f32]> {
        self.archetypes.get(id).map(|v| v.as_slice())
    }

    /// Get the precomputed magnitude squared for an archetype.
    pub fn magnitude_sq(&self, id: usize) -> Option<f32> {
        self.magnitude_sq.get(id).copied()
    }

    /// Number of registered archetypes.
    pub fn len(&self) -> usize {
        self.archetypes.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.archetypes.is_empty()
    }

    /// Find the best archetype for a given vector (highest cosine similarity).
    pub fn find_best_archetype(&self, vector: &[f32]) -> Option<(usize, f32)> {
        if self.archetypes.is_empty() {
            return None;
        }

        let query_mag: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_mag == 0.0 {
            return Some((0, 0.0));
        }

        let mut best_id = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for (id, archetype) in self.archetypes.iter().enumerate() {
            let dot: f32 = archetype
                .iter()
                .zip(vector.iter())
                .map(|(a, v)| a * v)
                .sum();
            let arch_mag = self.magnitude_sq[id].sqrt();
            let sim = if arch_mag == 0.0 {
                0.0
            } else {
                dot / (arch_mag * query_mag)
            };

            if sim > best_sim {
                best_sim = sim;
                best_id = id;
            }
        }

        Some((best_id, best_sim))
    }

    /// Create a delta vector using the best matching archetype.
    pub fn encode(&self, vector: &[f32], threshold: f32) -> Option<DeltaVector> {
        let (archetype_id, _) = self.find_best_archetype(vector)?;
        let archetype = self.get(archetype_id)?;
        Some(DeltaVector::from_dense_with_reference(
            vector,
            archetype,
            archetype_id,
            threshold,
        ))
    }

    /// Decode a delta vector back to dense.
    pub fn decode(&self, delta: &DeltaVector) -> Option<Vec<f32>> {
        let archetype = self.get(delta.archetype_id())?;
        Some(delta.to_dense(archetype))
    }

    /// Compute dot product between a delta and dense query efficiently.
    pub fn dot_delta_dense(&self, delta: &DeltaVector, query: &[f32]) -> Option<f32> {
        let archetype = self.get(delta.archetype_id())?;
        Some(delta.dot_dense(query, archetype))
    }

    /// Compute dot product between two deltas from same archetype efficiently.
    pub fn dot_deltas_same_archetype(&self, a: &DeltaVector, b: &DeltaVector) -> Option<f32> {
        if a.archetype_id() != b.archetype_id() {
            return None;
        }

        let archetype = self.get(a.archetype_id())?;
        let mag_sq = self.magnitude_sq(a.archetype_id())?;
        Some(a.dot_same_archetype(b, archetype, mag_sq))
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.archetypes.iter().map(|a| a.len() * 4).sum::<usize>()
            + self.magnitude_sq.len() * 4
    }

    /// Discover archetypes from a set of vectors using k-means clustering.
    ///
    /// Returns the number of archetypes discovered (may be less than k if
    /// there aren't enough distinct vectors or max_archetypes is reached).
    pub fn discover_archetypes(
        &mut self,
        vectors: &[Vec<f32>],
        k: usize,
        config: KMeansConfig,
    ) -> usize {
        if vectors.is_empty() {
            return 0;
        }

        let k = k.min(vectors.len()).min(self.max_archetypes - self.len());
        if k == 0 {
            return 0;
        }

        let kmeans = KMeans::new(config);
        let centroids = kmeans.fit(vectors, k);

        let mut added = 0;
        for centroid in centroids {
            if self.register(centroid).is_some() {
                added += 1;
            }
        }

        added
    }

    /// Encode multiple vectors using discovered archetypes.
    ///
    /// Returns a vector of (DeltaVector, compression_ratio) pairs.
    pub fn encode_batch(&self, vectors: &[Vec<f32>], threshold: f32) -> Vec<(DeltaVector, f32)> {
        vectors
            .iter()
            .filter_map(|v| {
                let delta = self.encode(v, threshold)?;
                let ratio = delta.compression_ratio();
                Some((delta, ratio))
            })
            .collect()
    }

    /// Analyze how well the current archetypes cover a set of vectors.
    ///
    /// Returns (avg_similarity, avg_compression_ratio, coverage_stats).
    pub fn analyze_coverage(&self, vectors: &[Vec<f32>], threshold: f32) -> CoverageStats {
        if vectors.is_empty() || self.is_empty() {
            return CoverageStats::default();
        }

        let mut total_similarity = 0.0;
        let mut total_compression = 0.0;
        let mut total_delta_nnz = 0;
        let mut archetype_usage = vec![0usize; self.len()];

        for vector in vectors {
            if let Some((arch_id, sim)) = self.find_best_archetype(vector) {
                total_similarity += sim;
                archetype_usage[arch_id] += 1;

                if let Some(delta) = self.encode(vector, threshold) {
                    total_compression += delta.compression_ratio();
                    total_delta_nnz += delta.nnz();
                }
            }
        }

        let n = vectors.len() as f32;
        CoverageStats {
            avg_similarity: total_similarity / n,
            avg_compression_ratio: total_compression / n,
            avg_delta_nnz: total_delta_nnz as f32 / n,
            archetype_usage,
        }
    }
}

/// Statistics about how well archetypes cover a set of vectors.
#[derive(Debug, Clone, Default)]
pub struct CoverageStats {
    /// Average cosine similarity to best archetype.
    pub avg_similarity: f32,
    /// Average compression ratio achieved.
    pub avg_compression_ratio: f32,
    /// Average number of non-zero deltas per vector.
    pub avg_delta_nnz: f32,
    /// Number of vectors assigned to each archetype.
    pub archetype_usage: Vec<usize>,
}

/// Configuration for k-means clustering.
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold (stop if centroids move less than this).
    pub convergence_threshold: f32,
    /// Random seed for initialization.
    pub seed: u64,
    /// Initialization method.
    pub init_method: KMeansInit,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-4,
            seed: 42,
            init_method: KMeansInit::KMeansPlusPlus,
        }
    }
}

/// Initialization method for k-means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansInit {
    /// Random selection of k points.
    Random,
    /// K-means++ initialization (better but slower).
    KMeansPlusPlus,
}

/// K-means clustering implementation.
pub struct KMeans {
    config: KMeansConfig,
}

impl KMeans {
    /// Create a new k-means instance with the given configuration.
    pub fn new(config: KMeansConfig) -> Self {
        Self { config }
    }

    /// Run k-means clustering on the given vectors.
    ///
    /// Returns k centroid vectors.
    pub fn fit(&self, vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        if vectors.is_empty() || k == 0 {
            return Vec::new();
        }

        let k = k.min(vectors.len());
        let dim = vectors[0].len();

        // Initialize centroids
        let mut centroids = match self.config.init_method {
            KMeansInit::Random => self.init_random(vectors, k),
            KMeansInit::KMeansPlusPlus => self.init_kmeans_plusplus(vectors, k),
        };

        let mut assignments = vec![0usize; vectors.len()];

        for _ in 0..self.config.max_iterations {
            // Assign each vector to nearest centroid
            for (i, vector) in vectors.iter().enumerate() {
                assignments[i] = self.nearest_centroid(vector, &centroids);
            }

            // Update centroids
            let new_centroids = self.update_centroids(vectors, &assignments, k, dim);

            // Check convergence
            let max_movement = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| self.euclidean_distance(old, new))
                .fold(0.0f32, f32::max);

            centroids = new_centroids;

            if max_movement < self.config.convergence_threshold {
                break;
            }
        }

        centroids
    }

    /// Random initialization: select k random points.
    fn init_random(&self, vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        // Simple deterministic selection based on seed
        let mut indices: Vec<usize> = (0..vectors.len()).collect();

        // Fisher-Yates shuffle with deterministic seed
        let mut rng_state = self.config.seed;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }

        indices
            .into_iter()
            .take(k)
            .map(|i| vectors[i].clone())
            .collect()
    }

    /// K-means++ initialization: probabilistic selection favoring distant points.
    fn init_kmeans_plusplus(&self, vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        let mut centroids = Vec::with_capacity(k);
        let mut rng_state = self.config.seed;

        // Select first centroid randomly
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let first_idx = (rng_state as usize) % vectors.len();
        centroids.push(vectors[first_idx].clone());

        // Select remaining centroids
        let mut distances = vec![f32::MAX; vectors.len()];

        for _ in 1..k {
            // Update distances to nearest centroid
            for (i, vector) in vectors.iter().enumerate() {
                let dist_to_new = self.euclidean_distance_sq(vector, centroids.last().unwrap());
                distances[i] = distances[i].min(dist_to_new);
            }

            // Select next centroid with probability proportional to distance squared
            let total_dist: f32 = distances.iter().sum();
            if total_dist == 0.0 {
                // All remaining points are at centroids, pick randomly
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let idx = (rng_state as usize) % vectors.len();
                centroids.push(vectors[idx].clone());
            } else {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let threshold = (rng_state as f32 / u64::MAX as f32) * total_dist;

                let mut cumulative = 0.0;
                let mut selected = 0;
                for (i, &dist) in distances.iter().enumerate() {
                    cumulative += dist;
                    if cumulative >= threshold {
                        selected = i;
                        break;
                    }
                }
                centroids.push(vectors[selected].clone());
            }
        }

        centroids
    }

    /// Find the nearest centroid to a vector.
    fn nearest_centroid(&self, vector: &[f32], centroids: &[Vec<f32>]) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, self.euclidean_distance_sq(vector, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update centroids based on assignments.
    fn update_centroids(
        &self,
        vectors: &[Vec<f32>],
        assignments: &[usize],
        k: usize,
        dim: usize,
    ) -> Vec<Vec<f32>> {
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (vector, &cluster) in vectors.iter().zip(assignments.iter()) {
            counts[cluster] += 1;
            for (sum, &val) in sums[cluster].iter_mut().zip(vector.iter()) {
                *sum += val;
            }
        }

        sums.into_iter()
            .zip(counts.iter())
            .map(|(sum, &count)| {
                if count == 0 {
                    vec![0.0; dim]
                } else {
                    sum.into_iter().map(|s| s / count as f32).collect()
                }
            })
            .collect()
    }

    /// Squared Euclidean distance between two vectors.
    #[inline]
    fn euclidean_distance_sq(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y) * (x - y))
            .sum()
    }

    /// Euclidean distance between two vectors.
    #[inline]
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.euclidean_distance_sq(a, b).sqrt()
    }
}

impl Default for ArchetypeRegistry {
    fn default() -> Self {
        Self::new(256) // Default to 256 max archetypes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_delta_from_dense() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.9, 0.1, 0.0, 0.0];

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.01);

        assert_eq!(delta.archetype_id(), 0);
        assert_eq!(delta.dimension(), 4);
        assert_eq!(delta.nnz(), 2); // positions 0 and 1 differ
    }

    #[test]
    fn test_delta_reconstruction() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.9, 0.1, 0.05, 0.0];

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);
        let reconstructed = delta.to_dense(&archetype);

        for (orig, rec) in vector.iter().zip(reconstructed.iter()) {
            assert!(approx_eq(*orig, *rec, 0.01));
        }
    }

    #[test]
    fn test_delta_threshold_filters_small_changes() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![1.001, 0.0005, 0.0, 0.0]; // Very small changes

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.01);

        // Changes below threshold should be filtered
        assert_eq!(delta.nnz(), 0);
    }

    #[test]
    fn test_delta_dot_dense() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.9, 0.1, 0.0, 0.0];
        let query = vec![1.0, 1.0, 0.0, 0.0];

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        // Full dot product
        let expected: f32 = vector.iter().zip(query.iter()).map(|(v, q)| v * q).sum();
        let actual = delta.dot_dense(&query, &archetype);

        assert!(approx_eq(expected, actual, 0.001));
    }

    #[test]
    fn test_delta_dot_with_precomputed() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.9, 0.1, 0.0, 0.0];
        let query = vec![1.0, 1.0, 0.0, 0.0];

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        // Precompute archetype dot query
        let arch_dot_query: f32 = archetype.iter().zip(query.iter()).map(|(a, q)| a * q).sum();

        let expected: f32 = vector.iter().zip(query.iter()).map(|(v, q)| v * q).sum();
        let actual = delta.dot_dense_with_precomputed(&query, arch_dot_query);

        assert!(approx_eq(expected, actual, 0.001));
    }

    #[test]
    fn test_delta_same_archetype_dot() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vec_a = vec![0.9, 0.1, 0.0, 0.0];
        let vec_b = vec![0.8, 0.2, 0.1, 0.0];

        let delta_a = DeltaVector::from_dense_with_reference(&vec_a, &archetype, 0, 0.001);
        let delta_b = DeltaVector::from_dense_with_reference(&vec_b, &archetype, 0, 0.001);

        let arch_mag_sq: f32 = archetype.iter().map(|x| x * x).sum();

        let expected: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();
        let actual = delta_a.dot_same_archetype(&delta_b, &archetype, arch_mag_sq);

        assert!(approx_eq(expected, actual, 0.001));
    }

    #[test]
    fn test_delta_sparsity() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let vector = vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        // 2 non-zero deltas out of 10 = 80% sparsity
        assert!(approx_eq(delta.delta_sparsity(), 0.8, 0.01));
    }

    #[test]
    fn test_compression_ratio() {
        let archetype = vec![1.0; 100];
        let mut vector = vec![1.0; 100];
        vector[0] = 0.9; // Only one difference

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        // Should have high compression ratio
        assert!(delta.compression_ratio() > 5.0);
    }

    #[test]
    fn test_archetype_registry() {
        let mut registry = ArchetypeRegistry::new(10);

        let arch1 = vec![1.0, 0.0, 0.0];
        let arch2 = vec![0.0, 1.0, 0.0];

        let id1 = registry.register(arch1.clone()).unwrap();
        let id2 = registry.register(arch2.clone()).unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(registry.len(), 2);

        assert_eq!(registry.get(id1).unwrap(), &arch1[..]);
        assert_eq!(registry.get(id2).unwrap(), &arch2[..]);
    }

    #[test]
    fn test_find_best_archetype() {
        let mut registry = ArchetypeRegistry::new(10);

        registry.register(vec![1.0, 0.0, 0.0]).unwrap();
        registry.register(vec![0.0, 1.0, 0.0]).unwrap();
        registry.register(vec![0.0, 0.0, 1.0]).unwrap();

        // Query closest to first archetype
        let (id, sim) = registry.find_best_archetype(&[0.9, 0.1, 0.0]).unwrap();
        assert_eq!(id, 0);
        assert!(sim > 0.9);

        // Query closest to second archetype
        let (id, _) = registry.find_best_archetype(&[0.1, 0.9, 0.0]).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn test_registry_encode_decode() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let vector = vec![0.9, 0.1, 0.0, 0.0];
        let delta = registry.encode(&vector, 0.001).unwrap();
        let reconstructed = registry.decode(&delta).unwrap();

        for (orig, rec) in vector.iter().zip(reconstructed.iter()) {
            assert!(approx_eq(*orig, *rec, 0.01));
        }
    }

    #[test]
    fn test_registry_max_capacity() {
        let mut registry = ArchetypeRegistry::new(2);

        assert!(registry.register(vec![1.0, 0.0]).is_some());
        assert!(registry.register(vec![0.0, 1.0]).is_some());
        assert!(registry.register(vec![0.5, 0.5]).is_none()); // Should fail
    }

    #[test]
    fn test_delta_iter() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.8, 0.2, 0.0, 0.0];

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        let pairs: Vec<_> = delta.iter().collect();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, 0); // position 0
        assert!(approx_eq(pairs[0].1, -0.2, 0.01)); // delta at 0
        assert_eq!(pairs[1].0, 1); // position 1
        assert!(approx_eq(pairs[1].1, 0.2, 0.01)); // delta at 1
    }

    #[test]
    fn test_to_sparse_delta() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.8, 0.2, 0.0, 0.0];

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);
        let sparse = delta.to_sparse_delta();

        assert_eq!(sparse.dimension(), 4);
        assert_eq!(sparse.nnz(), 2);
    }

    #[test]
    fn test_cosine_similarity_dense() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.9, 0.1, 0.0, 0.0];
        let query = vec![1.0, 0.0, 0.0, 0.0];

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        let query_mag: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sim = delta.cosine_similarity_dense(&query, &archetype, query_mag);

        // vector and query are very similar
        assert!(sim > 0.9);
    }

    #[test]
    fn test_memory_bytes() {
        let archetype = vec![0.0; 100];
        let mut vector = vec![0.0; 100];
        vector[0] = 1.0;
        vector[1] = 1.0;

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        // Should be much smaller than 100 * 4 = 400 bytes
        assert!(delta.memory_bytes() < 100);
    }

    // ==================== K-means clustering tests ====================

    #[test]
    fn test_kmeans_basic_clustering() {
        // Create two clear clusters
        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![0.05, 0.0],
            vec![10.0, 10.0],
            vec![10.1, 9.9],
            vec![9.9, 10.1],
        ];

        let config = KMeansConfig::default();
        let kmeans = KMeans::new(config);
        let centroids = kmeans.fit(&vectors, 2);

        assert_eq!(centroids.len(), 2);

        // Centroids should be near (0,0) and (10,10)
        let c1 = &centroids[0];
        let c2 = &centroids[1];

        let near_origin = |c: &[f32]| c[0].abs() < 1.0 && c[1].abs() < 1.0;
        let near_ten = |c: &[f32]| (c[0] - 10.0).abs() < 1.0 && (c[1] - 10.0).abs() < 1.0;

        assert!(
            (near_origin(c1) && near_ten(c2)) || (near_origin(c2) && near_ten(c1)),
            "Centroids should be near (0,0) and (10,10), got {:?} and {:?}",
            c1,
            c2
        );
    }

    #[test]
    fn test_kmeans_empty_input() {
        let vectors: Vec<Vec<f32>> = vec![];
        let config = KMeansConfig::default();
        let kmeans = KMeans::new(config);
        let centroids = kmeans.fit(&vectors, 3);

        assert!(centroids.is_empty());
    }

    #[test]
    fn test_kmeans_k_zero() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let config = KMeansConfig::default();
        let kmeans = KMeans::new(config);
        let centroids = kmeans.fit(&vectors, 0);

        assert!(centroids.is_empty());
    }

    #[test]
    fn test_kmeans_k_greater_than_n() {
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let config = KMeansConfig::default();
        let kmeans = KMeans::new(config);
        let centroids = kmeans.fit(&vectors, 10);

        // Should cap at n=2
        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_kmeans_single_point() {
        let vectors = vec![vec![5.0, 5.0]];
        let config = KMeansConfig::default();
        let kmeans = KMeans::new(config);
        let centroids = kmeans.fit(&vectors, 1);

        assert_eq!(centroids.len(), 1);
        assert!(approx_eq(centroids[0][0], 5.0, 0.001));
        assert!(approx_eq(centroids[0][1], 5.0, 0.001));
    }

    #[test]
    fn test_kmeans_random_init() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];

        let config = KMeansConfig {
            init_method: KMeansInit::Random,
            ..Default::default()
        };
        let kmeans = KMeans::new(config);
        let centroids = kmeans.fit(&vectors, 2);

        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_kmeans_plusplus_init() {
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 10.0],
            vec![11.0, 10.0],
        ];

        let config = KMeansConfig {
            init_method: KMeansInit::KMeansPlusPlus,
            ..Default::default()
        };
        let kmeans = KMeans::new(config);
        let centroids = kmeans.fit(&vectors, 2);

        assert_eq!(centroids.len(), 2);
    }

    #[test]
    fn test_kmeans_convergence() {
        // Tight cluster should converge quickly
        let vectors = vec![
            vec![1.0, 1.0],
            vec![1.01, 1.01],
            vec![0.99, 0.99],
            vec![1.0, 1.02],
        ];

        let config = KMeansConfig {
            max_iterations: 10,
            convergence_threshold: 0.01,
            ..Default::default()
        };
        let kmeans = KMeans::new(config);
        let centroids = kmeans.fit(&vectors, 1);

        assert_eq!(centroids.len(), 1);
        // Centroid should be near (1, 1)
        assert!(approx_eq(centroids[0][0], 1.0, 0.1));
        assert!(approx_eq(centroids[0][1], 1.0, 0.1));
    }

    #[test]
    fn test_discover_archetypes() {
        let mut registry = ArchetypeRegistry::new(10);

        // Three clear clusters
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 0.1, 0.9],
        ];

        let added = registry.discover_archetypes(&vectors, 3, KMeansConfig::default());

        assert_eq!(added, 3);
        assert_eq!(registry.len(), 3);
    }

    #[test]
    fn test_discover_archetypes_empty() {
        let mut registry = ArchetypeRegistry::new(10);
        let vectors: Vec<Vec<f32>> = vec![];

        let added = registry.discover_archetypes(&vectors, 3, KMeansConfig::default());

        assert_eq!(added, 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_discover_archetypes_respects_max() {
        let mut registry = ArchetypeRegistry::new(2);

        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ];

        // Request 4 but max is 2
        let added = registry.discover_archetypes(&vectors, 4, KMeansConfig::default());

        assert_eq!(added, 2);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_encode_batch() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0]).unwrap();
        registry.register(vec![0.0, 1.0, 0.0]).unwrap();

        let vectors = vec![
            vec![0.9, 0.1, 0.0],
            vec![0.1, 0.9, 0.0],
            vec![0.95, 0.05, 0.0],
        ];

        let results = registry.encode_batch(&vectors, 0.01);

        assert_eq!(results.len(), 3);

        // First and third should use archetype 0, second should use archetype 1
        assert_eq!(results[0].0.archetype_id(), 0);
        assert_eq!(results[1].0.archetype_id(), 1);
        assert_eq!(results[2].0.archetype_id(), 0);
    }

    #[test]
    fn test_analyze_coverage() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0]).unwrap();
        registry.register(vec![0.0, 1.0, 0.0]).unwrap();

        let vectors = vec![
            vec![0.95, 0.05, 0.0], // Close to arch 0
            vec![0.05, 0.95, 0.0], // Close to arch 1
            vec![0.9, 0.1, 0.0],   // Close to arch 0
        ];

        let stats = registry.analyze_coverage(&vectors, 0.01);

        // Average similarity should be high (vectors are close to archetypes)
        assert!(stats.avg_similarity > 0.9);

        // Archetype 0 used twice, archetype 1 used once
        assert_eq!(stats.archetype_usage[0], 2);
        assert_eq!(stats.archetype_usage[1], 1);
    }

    #[test]
    fn test_analyze_coverage_empty() {
        let registry = ArchetypeRegistry::new(10);
        let vectors: Vec<Vec<f32>> = vec![];

        let stats = registry.analyze_coverage(&vectors, 0.01);

        assert_eq!(stats.avg_similarity, 0.0);
        assert!(stats.archetype_usage.is_empty());
    }

    #[test]
    fn test_kmeans_deterministic() {
        // Same seed should produce same results
        let vectors = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];

        let config1 = KMeansConfig {
            seed: 12345,
            ..Default::default()
        };
        let config2 = KMeansConfig {
            seed: 12345,
            ..Default::default()
        };

        let kmeans1 = KMeans::new(config1);
        let kmeans2 = KMeans::new(config2);

        let centroids1 = kmeans1.fit(&vectors, 2);
        let centroids2 = kmeans2.fit(&vectors, 2);

        // Same seed should give same results
        assert_eq!(centroids1.len(), centroids2.len());
        for (c1, c2) in centroids1.iter().zip(centroids2.iter()) {
            for (v1, v2) in c1.iter().zip(c2.iter()) {
                assert!(approx_eq(*v1, *v2, 0.001));
            }
        }
    }

    #[test]
    fn test_full_pipeline_discover_and_encode() {
        let mut registry = ArchetypeRegistry::new(10);

        // Generate clustered high-dimensional data (128d for realistic embeddings)
        let dim = 128;
        let vectors: Vec<Vec<f32>> = (0..30)
            .map(|i| {
                let cluster = i / 10; // 3 clusters of 10
                let mut v = vec![0.0; dim];
                // Each cluster has a different dominant region
                let start = cluster * 40;
                for j in start..(start + 40) {
                    v[j] = 1.0;
                }
                // Add small noise
                let noise_idx = (i % 10) * 4;
                if noise_idx < dim {
                    v[noise_idx] += 0.1;
                }
                v
            })
            .collect();

        // Discover archetypes
        let added = registry.discover_archetypes(&vectors, 3, KMeansConfig::default());
        assert_eq!(added, 3);

        // Encode with appropriate threshold
        let encoded = registry.encode_batch(&vectors, 0.05);
        assert_eq!(encoded.len(), 30);

        // Verify deltas are sparse (most values match archetype)
        for (delta, _ratio) in &encoded {
            // With 128 dimensions and tight clusters, deltas should be sparse
            assert!(delta.nnz() < delta.dimension());
        }
    }

    // ==================== Additional coverage tests ====================

    #[test]
    fn test_from_dense_with_reference_and_magnitude() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.9, 0.1, 0.0, 0.0];

        let delta =
            DeltaVector::from_dense_with_reference_and_magnitude(&vector, &archetype, 0, 0.001);

        assert_eq!(delta.archetype_id(), 0);
        assert!(delta.cached_magnitude.is_some());

        // Cached magnitude should match actual magnitude
        let expected_mag: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx_eq(
            delta.cached_magnitude.unwrap(),
            expected_mag,
            0.001
        ));
    }

    #[test]
    fn test_from_parts() {
        let delta = DeltaVector::from_parts(5, 100, vec![0, 10, 20], vec![0.1, 0.2, 0.3]);

        assert_eq!(delta.archetype_id(), 5);
        assert_eq!(delta.dimension(), 100);
        assert_eq!(delta.nnz(), 3);

        let pairs: Vec<_> = delta.iter().collect();
        assert_eq!(pairs, vec![(0, 0.1), (10, 0.2), (20, 0.3)]);
    }

    #[test]
    fn test_set_cached_magnitude() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.9, 0.1, 0.0, 0.0];

        let mut delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);
        assert!(delta.cached_magnitude.is_none());

        delta.set_cached_magnitude(0.906);
        assert!(delta.cached_magnitude.is_some());
        assert!(approx_eq(delta.cached_magnitude.unwrap(), 0.906, 0.001));
    }

    #[test]
    fn test_compression_ratio_no_deltas() {
        // When nnz is 0, compression ratio should still be positive
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![1.0, 0.0, 0.0, 0.0]; // Identical to archetype

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        assert_eq!(delta.nnz(), 0);
        // Even with no deltas, there's struct overhead, so ratio is less than dimension
        // but still positive (we're compressing the values, just have fixed overhead)
        assert!(delta.compression_ratio() > 0.0);
    }

    #[test]
    fn test_magnitude_with_cached() {
        let archetype = vec![3.0, 4.0, 0.0]; // magnitude = 5
        let vector = vec![3.0, 4.0, 0.0]; // same, magnitude = 5

        let mut delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        // Without cache, computes from archetype
        let mag = delta.magnitude(&archetype);
        assert!(approx_eq(mag, 5.0, 0.01));

        // Set cached magnitude and verify it's used
        delta.set_cached_magnitude(5.0);
        let cached_mag = delta.magnitude(&archetype);
        assert!(approx_eq(cached_mag, 5.0, 0.01));
    }

    #[test]
    fn test_magnitude_zero_vector() {
        let archetype = vec![0.0, 0.0, 0.0, 0.0];
        let vector = vec![0.0, 0.0, 0.0, 0.0];

        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        // Zero vector magnitude should be 0
        assert!(approx_eq(delta.magnitude(&archetype), 0.0, 0.001));
    }

    #[test]
    fn test_cosine_similarity_dense_edge_cases() {
        let archetype = vec![1.0, 0.0, 0.0, 0.0];
        let vector = vec![0.9, 0.1, 0.0, 0.0];
        let delta = DeltaVector::from_dense_with_reference(&vector, &archetype, 0, 0.001);

        // Zero query vector
        let zero_query = vec![0.0, 0.0, 0.0, 0.0];
        let query_mag = 0.0; // Zero magnitude for zero query
        let sim = delta.cosine_similarity_dense(&zero_query, &archetype, query_mag);
        assert!(approx_eq(sim, 0.0, 0.001));

        // Non-zero query
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let query_mag2: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sim2 = delta.cosine_similarity_dense(&query, &archetype, query_mag2);
        assert!(sim2 > 0.9); // Should be high similarity
    }

    #[test]
    fn test_dot_delta_dense_via_registry() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let vector = vec![0.9, 0.1, 0.0, 0.0];
        let delta = registry.encode(&vector, 0.001).unwrap();

        let query = vec![1.0, 1.0, 0.0, 0.0];
        let dot = registry.dot_delta_dense(&delta, &query).unwrap();

        let expected: f32 = vector.iter().zip(query.iter()).map(|(v, q)| v * q).sum();
        assert!(approx_eq(dot, expected, 0.01));
    }

    #[test]
    fn test_dot_deltas_same_archetype_via_registry() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();

        let vec_a = vec![0.9, 0.1, 0.0, 0.0];
        let vec_b = vec![0.8, 0.2, 0.1, 0.0];

        let delta_a = registry.encode(&vec_a, 0.001).unwrap();
        let delta_b = registry.encode(&vec_b, 0.001).unwrap();

        let dot = registry
            .dot_deltas_same_archetype(&delta_a, &delta_b)
            .unwrap();

        let expected: f32 = vec_a.iter().zip(vec_b.iter()).map(|(a, b)| a * b).sum();
        assert!(approx_eq(dot, expected, 0.01));
    }

    #[test]
    fn test_dot_deltas_different_archetypes_returns_none() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        registry.register(vec![0.0, 1.0, 0.0, 0.0]).unwrap();

        let delta_a = DeltaVector::from_parts(0, 4, vec![0], vec![0.1]);
        let delta_b = DeltaVector::from_parts(1, 4, vec![1], vec![0.2]);

        // Different archetypes should return None
        assert!(registry
            .dot_deltas_same_archetype(&delta_a, &delta_b)
            .is_none());
    }

    #[test]
    fn test_registry_is_empty() {
        let registry = ArchetypeRegistry::new(10);
        assert!(registry.is_empty());

        let mut registry2 = ArchetypeRegistry::new(10);
        registry2.register(vec![1.0, 0.0]).unwrap();
        assert!(!registry2.is_empty());
    }

    #[test]
    fn test_registry_magnitude_sq() {
        let mut registry = ArchetypeRegistry::new(10);
        registry.register(vec![3.0, 4.0]).unwrap(); // magnitude = 5, sq = 25

        let mag_sq = registry.magnitude_sq(0).unwrap();
        assert!(approx_eq(mag_sq, 25.0, 0.01));

        // Non-existent archetype
        assert!(registry.magnitude_sq(99).is_none());
    }

    #[test]
    fn test_find_best_archetype_empty_registry() {
        let registry = ArchetypeRegistry::new(10);
        assert!(registry.find_best_archetype(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn test_encode_empty_registry() {
        let registry = ArchetypeRegistry::new(10);
        assert!(registry.encode(&[1.0, 0.0], 0.01).is_none());
    }

    #[test]
    fn test_decode_invalid_archetype() {
        let registry = ArchetypeRegistry::new(10);
        let delta = DeltaVector::from_parts(99, 4, vec![], vec![]);
        assert!(registry.decode(&delta).is_none());
    }

    #[test]
    fn test_default_registry() {
        let mut registry = ArchetypeRegistry::default();
        assert!(registry.is_empty());
        // Default should allow 256 archetypes
        for i in 0..256 {
            assert!(registry.register(vec![i as f32]).is_some() || i >= 256);
        }
    }

    #[test]
    fn test_default_kmeans_config() {
        let config = KMeansConfig::default();
        assert_eq!(config.max_iterations, 100);
        assert!(approx_eq(config.convergence_threshold, 1e-4, 1e-6));
        assert_eq!(config.seed, 42);
        assert!(matches!(config.init_method, KMeansInit::KMeansPlusPlus));
    }
}
