// SPDX-License-Identifier: MIT OR Apache-2.0
//! IVF (Inverted File Index) for large-scale partitioned search.
//!
//! IVF partitions vectors into clusters using k-means, enabling sublinear search
//! by only scanning the most relevant clusters (nprobe).
//!
//! # Variants
//!
//! - **IVF-Flat**: Full vectors in each list (highest recall)
//! - **IVF-PQ**: PQ-compressed vectors (best memory efficiency)
//! - **IVF-Binary**: Binary-quantized vectors (fastest distance computation)
//!
//! # Example
//!
//! ```rust
//! use tensor_store::ivf::{IVFConfig, IVFIndex, IVFStorage};
//!
//! let config = IVFConfig::default().with_num_clusters(16).with_nprobe(4);
//! let mut index = IVFIndex::new(config);
//!
//! // Add vectors
//! let vectors: Vec<Vec<f32>> = (0..100)
//!     .map(|i| (0..64).map(|j| (i * j) as f32 / 1000.0).collect())
//!     .collect();
//!
//! // Train on the vectors
//! let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();
//! index.train(&refs);
//!
//! // Add vectors to the index
//! for v in &vectors {
//!     index.add(v);
//! }
//!
//! // Search
//! let query = &vectors[0];
//! let results = index.search(query, 10);
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::binary_quantization::{BinaryThreshold, BinaryVector};
use crate::delta_vector::{KMeans, KMeansConfig};
use crate::pq::{PQCodebook, PQConfig, PQVector};

/// Compute default nprobe as `sqrt(num_clusters)`.
#[allow(
    dead_code,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
#[must_use]
fn default_nprobe(num_clusters: usize) -> usize {
    (num_clusters as f32).sqrt().ceil() as usize
}

/// IVF configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFConfig {
    /// Number of clusters (partitions).
    pub num_clusters: usize,
    /// Number of clusters to search (default: `sqrt(num_clusters)`).
    pub nprobe: usize,
    /// K-means configuration for training centroids.
    pub kmeans_config: KMeansConfig,
    /// Storage format for vectors within lists.
    pub storage: IVFStorage,
}

impl Default for IVFConfig {
    fn default() -> Self {
        let num_clusters = 100;
        Self {
            num_clusters,
            nprobe: default_nprobe(num_clusters),
            kmeans_config: KMeansConfig::default(),
            storage: IVFStorage::Flat,
        }
    }
}

impl IVFConfig {
    /// Create a configuration for IVF-Flat (full vectors).
    #[must_use]
    pub fn flat(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            nprobe: default_nprobe(num_clusters),
            kmeans_config: KMeansConfig::default(),
            storage: IVFStorage::Flat,
        }
    }

    /// Create a configuration for IVF-PQ (product quantized).
    #[must_use]
    pub fn pq(num_clusters: usize, pq_config: PQConfig) -> Self {
        Self {
            num_clusters,
            nprobe: default_nprobe(num_clusters),
            kmeans_config: KMeansConfig::default(),
            storage: IVFStorage::PQ(pq_config),
        }
    }

    /// Create a configuration for IVF-Binary (binary quantized).
    #[must_use]
    pub fn binary(num_clusters: usize, threshold: BinaryThreshold) -> Self {
        Self {
            num_clusters,
            nprobe: default_nprobe(num_clusters),
            kmeans_config: KMeansConfig::default(),
            storage: IVFStorage::Binary(threshold),
        }
    }

    /// Set the number of clusters.
    #[must_use]
    pub const fn with_num_clusters(mut self, num_clusters: usize) -> Self {
        self.num_clusters = num_clusters;
        self
    }

    /// Set the number of clusters to probe during search.
    #[must_use]
    pub const fn with_nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set the k-means configuration.
    #[must_use]
    pub const fn with_kmeans_config(mut self, config: KMeansConfig) -> Self {
        self.kmeans_config = config;
        self
    }

    /// Set the storage format.
    #[must_use]
    pub const fn with_storage(mut self, storage: IVFStorage) -> Self {
        self.storage = storage;
        self
    }
}

/// Storage format for vectors within IVF lists.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum IVFStorage {
    /// Full vectors (IVF-Flat). Best recall, highest memory.
    #[default]
    Flat,
    /// PQ-compressed vectors (IVF-PQ). Best compression, good recall.
    PQ(PQConfig),
    /// Binary-quantized vectors (IVF-Binary). Fastest search, lower recall.
    Binary(BinaryThreshold),
}

/// IVF Index with inverted lists.
pub struct IVFIndex {
    /// Cluster centroids.
    centroids: Vec<Vec<f32>>,
    /// Inverted lists, one per cluster.
    inverted_lists: Vec<RwLock<InvertedList>>,
    /// Configuration.
    config: IVFConfig,
    /// PQ codebook (for IVF-PQ).
    pq_codebook: Option<PQCodebook>,
    /// Vector dimension.
    dimension: usize,
    /// Whether the index is trained.
    trained: bool,
    /// Atomic counter for assigning vector IDs.
    next_id: std::sync::atomic::AtomicUsize,
}

struct InvertedList {
    /// Vector IDs in this list.
    ids: Vec<usize>,
    /// Storage for vectors.
    storage: ListStorage,
}

enum ListStorage {
    Flat(Vec<Vec<f32>>),
    PQ(Vec<PQVector>),
    Binary(Vec<BinaryVector>),
}

impl IVFIndex {
    /// Create a new untrained IVF index.
    #[must_use]
    pub fn new(config: IVFConfig) -> Self {
        let num_clusters = config.num_clusters;
        let inverted_lists = (0..num_clusters)
            .map(|_| {
                RwLock::new(InvertedList {
                    ids: Vec::new(),
                    storage: match &config.storage {
                        IVFStorage::Flat => ListStorage::Flat(Vec::new()),
                        IVFStorage::PQ(_) => ListStorage::PQ(Vec::new()),
                        IVFStorage::Binary(_) => ListStorage::Binary(Vec::new()),
                    },
                })
            })
            .collect();

        Self {
            centroids: Vec::new(),
            inverted_lists,
            config,
            pq_codebook: None,
            dimension: 0,
            trained: false,
            next_id: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Train the IVF index on a set of vectors.
    ///
    /// This trains cluster centroids and (for IVF-PQ) the PQ codebook.
    pub fn train(&mut self, vectors: &[&[f32]]) {
        if vectors.is_empty() {
            return;
        }

        self.dimension = vectors[0].len();
        let num_clusters = self.config.num_clusters.min(vectors.len());

        // Train cluster centroids using k-means
        let kmeans = KMeans::new(self.config.kmeans_config.clone());
        let owned_vectors: Vec<Vec<f32>> = vectors.iter().map(|v| v.to_vec()).collect();
        self.centroids = kmeans.fit(&owned_vectors, num_clusters);

        // Train PQ codebook if using IVF-PQ
        if let IVFStorage::PQ(ref pq_config) = self.config.storage {
            // Compute residuals for PQ training
            let residuals: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| {
                    let cluster = self.find_nearest_centroid(v);
                    v.iter()
                        .zip(self.centroids[cluster].iter())
                        .map(|(a, b)| a - b)
                        .collect()
                })
                .collect();
            let residual_refs: Vec<&[f32]> = residuals.iter().map(Vec::as_slice).collect();
            self.pq_codebook = Some(PQCodebook::train(&residual_refs, pq_config));
        }

        // Reinitialize inverted lists with correct storage type
        self.inverted_lists = (0..self.centroids.len())
            .map(|_| {
                RwLock::new(InvertedList {
                    ids: Vec::new(),
                    storage: match &self.config.storage {
                        IVFStorage::Flat => ListStorage::Flat(Vec::new()),
                        IVFStorage::PQ(_) => ListStorage::PQ(Vec::new()),
                        IVFStorage::Binary(_) => ListStorage::Binary(Vec::new()),
                    },
                })
            })
            .collect();

        // Reset ID counter since we cleared all lists
        self.next_id.store(0, std::sync::atomic::Ordering::Relaxed);

        self.trained = true;
    }

    /// Add a vector to the index.
    ///
    /// Returns the assigned ID.
    #[allow(clippy::must_use_candidate)]
    pub fn add(&self, vector: &[f32]) -> usize {
        use std::sync::atomic::Ordering;

        if !self.trained || self.centroids.is_empty() {
            return 0;
        }

        let cluster = self.find_nearest_centroid(vector);
        let mut list = self.inverted_lists[cluster].write();

        // Assign a new ID using atomic counter (avoids deadlock with len())
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        list.ids.push(id);

        match &mut list.storage {
            ListStorage::Flat(vectors) => {
                vectors.push(vector.to_vec());
            },
            ListStorage::PQ(pq_vectors) => {
                if let Some(ref codebook) = self.pq_codebook {
                    // Compute residual and encode
                    let residual: Vec<f32> = vector
                        .iter()
                        .zip(self.centroids[cluster].iter())
                        .map(|(a, b)| a - b)
                        .collect();
                    pq_vectors.push(codebook.encode(&residual));
                }
            },
            ListStorage::Binary(binary_vectors) => {
                if let IVFStorage::Binary(threshold) = self.config.storage {
                    binary_vectors.push(BinaryVector::from_dense(vector, threshold));
                }
            },
        }
        drop(list);

        id
    }

    /// Search for k nearest neighbors.
    #[must_use]
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.search_with_nprobe(query, k, self.config.nprobe)
    }

    /// Search with a custom nprobe value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn search_with_nprobe(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<(usize, f32)> {
        if !self.trained || self.centroids.is_empty() || k == 0 {
            return Vec::new();
        }

        // Find nprobe closest centroids
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, Self::squared_euclidean(query, c)))
            .collect();
        centroid_distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let nprobe = nprobe.min(self.centroids.len());

        // Collect candidates from top nprobe clusters
        let mut candidates: Vec<(usize, f32)> = Vec::new();

        // Precompute ADC table for PQ search
        let adc_table = if let IVFStorage::PQ(_) = self.config.storage {
            self.pq_codebook.as_ref().map(|cb| {
                // For IVF-PQ, we compute ADC tables per centroid
                // For simplicity, we'll recompute residual distances inline
                (cb, query.to_vec())
            })
        } else {
            None
        };

        // Binary query for binary search
        let binary_query = if let IVFStorage::Binary(threshold) = self.config.storage {
            Some(BinaryVector::from_dense(query, threshold))
        } else {
            None
        };

        for &(cluster_idx, _) in centroid_distances.iter().take(nprobe) {
            let list = self.inverted_lists[cluster_idx].read();

            match &list.storage {
                ListStorage::Flat(vectors) => {
                    for (local_idx, vector) in vectors.iter().enumerate() {
                        let dist = Self::squared_euclidean(query, vector).sqrt();
                        candidates.push((list.ids[local_idx], dist));
                    }
                },
                ListStorage::PQ(pq_vectors) => {
                    if let Some((codebook, ref query_vec)) = adc_table {
                        // Compute residual from this cluster's centroid
                        let residual: Vec<f32> = query_vec
                            .iter()
                            .zip(self.centroids[cluster_idx].iter())
                            .map(|(a, b)| a - b)
                            .collect();
                        let adc = codebook.compute_adc_table(&residual);

                        for (local_idx, pq_vec) in pq_vectors.iter().enumerate() {
                            let dist = adc.distance(pq_vec);
                            candidates.push((list.ids[local_idx], dist));
                        }
                    }
                },
                ListStorage::Binary(binary_vectors) => {
                    if let Some(ref bq) = binary_query {
                        for (local_idx, binary_vec) in binary_vectors.iter().enumerate() {
                            // Use normalized Hamming distance
                            let dist = bq.normalized_distance(binary_vec);
                            candidates.push((list.ids[local_idx], dist));
                        }
                    }
                },
            }
        }

        // Sort by distance and return top k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Returns the total number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inverted_lists
            .iter()
            .map(|list| list.read().ids.len())
            .sum()
    }

    /// Returns true if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the index is trained.
    #[must_use]
    pub const fn is_trained(&self) -> bool {
        self.trained
    }

    /// Returns the number of clusters.
    #[must_use]
    pub const fn num_clusters(&self) -> usize {
        self.centroids.len()
    }

    /// Returns the configured nprobe value.
    #[must_use]
    pub const fn nprobe(&self) -> usize {
        self.config.nprobe
    }

    /// Returns the vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns cluster sizes (number of vectors per cluster).
    #[must_use]
    pub fn cluster_sizes(&self) -> Vec<usize> {
        self.inverted_lists
            .iter()
            .map(|list| list.read().ids.len())
            .collect()
    }

    /// Returns memory usage estimate in bytes.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn memory_bytes(&self) -> usize {
        let centroid_bytes = self.centroids.iter().map(|c| c.len() * 4).sum::<usize>();

        let list_bytes: usize = self
            .inverted_lists
            .iter()
            .map(|list| {
                let list = list.read();
                let id_bytes = list.ids.len() * std::mem::size_of::<usize>();
                let storage_bytes: usize = match &list.storage {
                    ListStorage::Flat(vectors) => vectors.iter().map(|v| v.len() * 4).sum(),
                    ListStorage::PQ(pq_vectors) => {
                        pq_vectors.iter().map(PQVector::memory_bytes).sum()
                    },
                    ListStorage::Binary(binary_vectors) => {
                        binary_vectors.iter().map(BinaryVector::memory_bytes).sum()
                    },
                };
                let result = id_bytes + storage_bytes;
                drop(list);
                result
            })
            .sum();

        let codebook_bytes = self
            .pq_codebook
            .as_ref()
            .map_or(0, PQCodebook::memory_bytes);

        centroid_bytes + list_bytes + codebook_bytes
    }

    fn find_nearest_centroid(&self, vector: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, Self::squared_euclidean(vector, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    }

    #[inline]
    fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }
}

// Serialization support for IVFIndex state
impl IVFIndex {
    /// Export the trained state for serialization.
    #[must_use]
    pub fn export_state(&self) -> IVFIndexState {
        IVFIndexState {
            centroids: self.centroids.clone(),
            config: self.config.clone(),
            dimension: self.dimension,
            trained: self.trained,
        }
    }

    /// Import trained state (creates an empty trained index ready for inserts).
    #[must_use]
    pub fn from_state(state: IVFIndexState) -> Self {
        let inverted_lists = (0..state.centroids.len())
            .map(|_| {
                RwLock::new(InvertedList {
                    ids: Vec::new(),
                    storage: match &state.config.storage {
                        IVFStorage::Flat => ListStorage::Flat(Vec::new()),
                        IVFStorage::PQ(_) => ListStorage::PQ(Vec::new()),
                        IVFStorage::Binary(_) => ListStorage::Binary(Vec::new()),
                    },
                })
            })
            .collect();

        Self {
            centroids: state.centroids,
            inverted_lists,
            config: state.config,
            pq_codebook: None, // Would need to be retrained or serialized separately
            dimension: state.dimension,
            trained: state.trained,
            next_id: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

/// Serializable state of an IVF index (excludes vectors).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVFIndexState {
    /// Cluster centroids.
    pub centroids: Vec<Vec<f32>>,
    /// Configuration.
    pub config: IVFConfig,
    /// Vector dimension.
    pub dimension: usize,
    /// Whether trained.
    pub trained: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|j| ((i * 7 + j * 13) % 100) as f32 / 100.0)
                    .collect()
            })
            .collect()
    }

    /// Create a fast IVF config for testing (fewer k-means iterations).
    fn fast_test_config(num_clusters: usize) -> IVFConfig {
        IVFConfig {
            num_clusters,
            nprobe: default_nprobe(num_clusters),
            kmeans_config: fast_kmeans_config(),
            storage: IVFStorage::Flat,
        }
    }

    fn fast_kmeans_config() -> KMeansConfig {
        KMeansConfig {
            max_iterations: 2,
            convergence_threshold: 1.0,
            seed: 42,
            init_method: crate::delta_vector::KMeansInit::Random,
        }
    }

    fn fast_test_config_pq(num_clusters: usize, pq_config: PQConfig) -> IVFConfig {
        IVFConfig {
            num_clusters,
            nprobe: default_nprobe(num_clusters),
            kmeans_config: fast_kmeans_config(),
            storage: IVFStorage::PQ(pq_config),
        }
    }

    fn fast_test_config_binary(num_clusters: usize, threshold: BinaryThreshold) -> IVFConfig {
        IVFConfig {
            num_clusters,
            nprobe: default_nprobe(num_clusters),
            kmeans_config: fast_kmeans_config(),
            storage: IVFStorage::Binary(threshold),
        }
    }

    #[test]
    fn ivf_config_default() {
        let config = IVFConfig::default();
        assert_eq!(config.num_clusters, 100);
        assert!(config.nprobe > 0);
    }

    #[test]
    fn ivf_config_flat() {
        let config = IVFConfig::flat(16);
        assert_eq!(config.num_clusters, 16);
        assert!(matches!(config.storage, IVFStorage::Flat));
    }

    #[test]
    fn ivf_config_pq() {
        let pq_config = PQConfig::default();
        let config = IVFConfig::pq(32, pq_config);
        assert_eq!(config.num_clusters, 32);
        assert!(matches!(config.storage, IVFStorage::PQ(_)));
    }

    #[test]
    fn ivf_config_binary() {
        let config = IVFConfig::binary(16, BinaryThreshold::Sign);
        assert_eq!(config.num_clusters, 16);
        assert!(matches!(config.storage, IVFStorage::Binary(_)));
    }

    #[test]
    fn ivf_config_builder() {
        let config = IVFConfig::default()
            .with_num_clusters(64)
            .with_nprobe(8)
            .with_storage(IVFStorage::Flat);

        assert_eq!(config.num_clusters, 64);
        assert_eq!(config.nprobe, 8);
    }

    #[test]
    fn ivf_train_creates_clusters() {
        let vectors = create_test_vectors(20, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config(4);
        let mut index = IVFIndex::new(config);

        assert!(!index.is_trained());
        index.train(&refs);

        assert!(index.is_trained());
        assert_eq!(index.num_clusters(), 4);
        assert_eq!(index.dimension(), 16);
    }

    #[test]
    fn ivf_add_assigns_correct_cluster() {
        let vectors = create_test_vectors(20, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config(4);
        let mut index = IVFIndex::new(config);
        index.train(&refs);

        for v in &vectors {
            index.add(v);
        }

        assert_eq!(index.len(), 20);

        // Check that all vectors are distributed across clusters
        let sizes = index.cluster_sizes();
        assert_eq!(sizes.iter().sum::<usize>(), 20);
    }

    #[test]
    fn ivf_search_basic() {
        let vectors = create_test_vectors(20, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config(4).with_nprobe(2);
        let mut index = IVFIndex::new(config);
        index.train(&refs);

        for v in &vectors {
            index.add(v);
        }

        let query = &vectors[0];
        let results = index.search(query, 5);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        // Results should be sorted by distance
        for window in results.windows(2) {
            assert!(window[0].1 <= window[1].1);
        }
    }

    #[test]
    fn ivf_search_nprobe_effect() {
        let vectors = create_test_vectors(40, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config(8);
        let mut index = IVFIndex::new(config);
        index.train(&refs);

        for v in &vectors {
            index.add(v);
        }

        let query = &vectors[0];

        // More nprobe should give same or more results
        let results_1 = index.search_with_nprobe(query, 5, 1);
        let results_4 = index.search_with_nprobe(query, 5, 4);
        let results_8 = index.search_with_nprobe(query, 5, 8);

        // With more probes, we might find better results
        // The best result with more probes should be at least as good
        if !results_1.is_empty() && !results_8.is_empty() {
            assert!(results_8[0].1 <= results_1[0].1 + 0.001);
        }

        // Verify we get results
        assert!(!results_4.is_empty());
    }

    #[test]
    fn ivf_pq_integration() {
        let vectors = create_test_vectors(20, 32);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let pq_config = PQConfig {
            num_subspaces: 4,
            num_centroids: 8,
            kmeans_config: fast_kmeans_config(),
        };
        let config = fast_test_config_pq(4, pq_config).with_nprobe(2);

        let mut index = IVFIndex::new(config);
        index.train(&refs);

        for v in &vectors {
            index.add(v);
        }

        let query = &vectors[0];
        let results = index.search(query, 5);

        assert!(!results.is_empty());
    }

    #[test]
    fn ivf_binary_integration() {
        let vectors = create_test_vectors(20, 32);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config_binary(4, BinaryThreshold::Sign).with_nprobe(2);

        let mut index = IVFIndex::new(config);
        index.train(&refs);

        for v in &vectors {
            index.add(v);
        }

        let query = &vectors[0];
        let results = index.search(query, 5);

        assert!(!results.is_empty());
    }

    #[test]
    fn ivf_empty_search() {
        let config = fast_test_config(4);
        let index = IVFIndex::new(config);

        let query = vec![0.0f32; 16];
        let results = index.search(&query, 5);

        assert!(results.is_empty());
    }

    #[test]
    fn ivf_memory_bytes() {
        let vectors = create_test_vectors(20, 32);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config(4);
        let mut index = IVFIndex::new(config);
        index.train(&refs);

        for v in &vectors {
            index.add(v);
        }

        let memory = index.memory_bytes();
        assert!(memory > 0);
    }

    #[test]
    fn ivf_export_import_state() {
        let vectors = create_test_vectors(20, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config(4);
        let mut index = IVFIndex::new(config);
        index.train(&refs);

        let state = index.export_state();

        assert!(state.trained);
        assert_eq!(state.dimension, 16);
        assert_eq!(state.centroids.len(), 4);

        let restored = IVFIndex::from_state(state);
        assert!(restored.is_trained());
        assert_eq!(restored.num_clusters(), 4);
    }

    #[test]
    fn ivf_concurrent_add() {
        use std::sync::Arc;
        use std::thread;

        let vectors = create_test_vectors(20, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config(4);
        let mut index = IVFIndex::new(config);
        index.train(&refs);

        let index = Arc::new(index);
        let vectors = Arc::new(vectors);

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let index = Arc::clone(&index);
                let vectors = Arc::clone(&vectors);
                thread::spawn(move || {
                    for i in (t * 5)..((t + 1) * 5) {
                        index.add(&vectors[i]);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(index.len(), 20);
    }

    #[test]
    fn ivf_concurrent_search() {
        use std::sync::Arc;
        use std::thread;

        let vectors = create_test_vectors(20, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config(4).with_nprobe(2);
        let mut index = IVFIndex::new(config);
        index.train(&refs);

        for v in &vectors {
            index.add(v);
        }

        let index = Arc::new(index);
        let vectors = Arc::new(vectors);

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let index = Arc::clone(&index);
                let vectors = Arc::clone(&vectors);
                thread::spawn(move || {
                    for i in (t * 5)..((t + 1) * 5) {
                        let results = index.search(&vectors[i], 3);
                        assert!(!results.is_empty());
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn ivf_train_empty() {
        let config = fast_test_config(4);
        let mut index = IVFIndex::new(config);

        let refs: Vec<&[f32]> = Vec::new();
        index.train(&refs);

        assert!(!index.is_trained());
    }

    #[test]
    fn ivf_search_k_zero() {
        let vectors = create_test_vectors(20, 16);
        let refs: Vec<&[f32]> = vectors.iter().map(Vec::as_slice).collect();

        let config = fast_test_config(4);
        let mut index = IVFIndex::new(config);
        index.train(&refs);

        for v in &vectors {
            index.add(v);
        }

        let results = index.search(&vectors[0], 0);
        assert!(results.is_empty());
    }
}
