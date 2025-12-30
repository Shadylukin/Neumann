//! Hierarchical codebook system for state discretization.
//!
//! Provides vector quantization for mapping continuous tensor states
//! to a finite vocabulary of valid states:
//!
//! - GlobalCodebook: Static, shared across all nodes for consensus
//! - LocalCodebook: Adaptive per domain, captures residuals via EMA
//!
//! # Architecture
//!
//! ```text
//! Input Vector
//!       |
//!       v
//! [Global Codebook] --> nearest entry + residual
//!       |                      |
//!       v                      v
//!   quantized code      [Local Codebook] --> refined entry
//!                              |
//!                              v
//!                       full quantization
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{ChainError, Result};

/// An entry in a codebook representing a valid state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebookEntry {
    /// Unique identifier within the codebook.
    id: u32,
    /// Centroid vector representing this state.
    centroid: Vec<f32>,
    /// Cached magnitude for fast cosine computation.
    magnitude: f32,
    /// Access count for frequency-based metrics.
    access_count: u64,
    /// Last access timestamp (epoch millis).
    last_access: u64,
    /// Creation timestamp.
    created_at: u64,
    /// Optional semantic label.
    label: Option<String>,
}

impl CodebookEntry {
    /// Create a new codebook entry.
    pub fn new(id: u32, centroid: Vec<f32>) -> Self {
        let magnitude = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        let now = current_timestamp_millis();

        Self {
            id,
            centroid,
            magnitude,
            access_count: 0,
            last_access: now,
            created_at: now,
            label: None,
        }
    }

    /// Create with a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the entry ID.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Get the centroid vector.
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    /// Get the cached magnitude.
    pub fn magnitude(&self) -> f32 {
        self.magnitude
    }

    /// Get access count.
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// Get last access time.
    pub fn last_access(&self) -> u64 {
        self.last_access
    }

    /// Get the label if set.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Record an access.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_access = current_timestamp_millis();
    }

    /// Compute cosine similarity with a query vector.
    pub fn cosine_similarity(&self, query: &[f32]) -> f32 {
        if self.magnitude == 0.0 {
            return 0.0;
        }

        let query_mag: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_mag == 0.0 {
            return 0.0;
        }

        let dot: f32 = self
            .centroid
            .iter()
            .zip(query.iter())
            .map(|(c, q)| c * q)
            .sum();

        dot / (self.magnitude * query_mag)
    }

    /// Update centroid via EMA.
    pub fn ema_update(&mut self, observation: &[f32], alpha: f32) {
        for (c, o) in self.centroid.iter_mut().zip(observation.iter()) {
            *c = alpha * o + (1.0 - alpha) * *c;
        }
        self.magnitude = self.centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.record_access();
    }
}

/// Static global codebook for consensus.
#[derive(Debug, Clone)]
pub struct GlobalCodebook {
    /// Codebook entries (centroids).
    entries: Vec<CodebookEntry>,
    /// Vector dimension.
    dimension: usize,
}

impl GlobalCodebook {
    /// Create an empty global codebook.
    pub fn new(dimension: usize) -> Self {
        Self {
            entries: Vec::new(),
            dimension,
        }
    }

    /// Create from pre-computed centroids.
    pub fn from_centroids(centroids: Vec<Vec<f32>>) -> Self {
        let dimension = centroids.first().map(|v| v.len()).unwrap_or(0);
        let entries = centroids
            .into_iter()
            .enumerate()
            .map(|(id, centroid)| CodebookEntry::new(id as u32, centroid))
            .collect();

        Self { entries, dimension }
    }

    /// Create from centroids with labels.
    pub fn from_centroids_with_labels(centroids: Vec<Vec<f32>>, labels: Vec<String>) -> Self {
        let dimension = centroids.first().map(|v| v.len()).unwrap_or(0);
        let entries = centroids
            .into_iter()
            .zip(labels)
            .enumerate()
            .map(|(id, (centroid, label))| {
                CodebookEntry::new(id as u32, centroid).with_label(label)
            })
            .collect();

        Self { entries, dimension }
    }

    /// Initialize using k-means clustering on training data.
    pub fn from_kmeans(vectors: &[Vec<f32>], k: usize, max_iterations: usize) -> Self {
        if vectors.is_empty() || k == 0 {
            return Self::new(0);
        }

        let dimension = vectors[0].len();
        let k = k.min(vectors.len());

        // K-means++ initialization
        let mut centroids = kmeans_plusplus_init(vectors, k);

        // K-means iterations
        for _ in 0..max_iterations {
            let assignments = assign_to_nearest(vectors, &centroids);
            let new_centroids = update_centroids(vectors, &assignments, k, dimension);

            // Check convergence
            let max_movement = centroids
                .iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| euclidean_distance(old, new))
                .fold(0.0f32, f32::max);

            centroids = new_centroids;

            if max_movement < 1e-4 {
                break;
            }
        }

        Self::from_centroids(centroids)
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by ID.
    pub fn get(&self, id: u32) -> Option<&CodebookEntry> {
        self.entries.get(id as usize)
    }

    /// Find the nearest entry to a query vector.
    pub fn quantize(&self, vector: &[f32]) -> Option<(u32, f32)> {
        if self.entries.is_empty() {
            return None;
        }

        let mut best_id = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for entry in &self.entries {
            let sim = entry.cosine_similarity(vector);
            if sim > best_sim {
                best_sim = sim;
                best_id = entry.id();
            }
        }

        Some((best_id, best_sim))
    }

    /// Compute residual: vector - nearest_centroid.
    pub fn compute_residual(&self, vector: &[f32]) -> Option<(u32, Vec<f32>)> {
        let (id, _) = self.quantize(vector)?;
        let centroid = &self.entries[id as usize].centroid;

        let residual: Vec<f32> = vector
            .iter()
            .zip(centroid.iter())
            .map(|(v, c)| v - c)
            .collect();

        Some((id, residual))
    }

    /// Check if a vector is within distance threshold of any entry.
    pub fn is_valid_state(&self, vector: &[f32], threshold: f32) -> bool {
        if let Some((_, sim)) = self.quantize(vector) {
            sim >= threshold
        } else {
            false
        }
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = &CodebookEntry> {
        self.entries.iter()
    }
}

/// Pruning strategy for local codebooks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PruningStrategy {
    /// Least Recently Used.
    LRU,
    /// Least Frequently Used.
    LFU,
    /// Hybrid: combines recency and frequency.
    Hybrid {
        recency_weight: f32,
        frequency_weight: f32,
    },
}

impl Default for PruningStrategy {
    fn default() -> Self {
        Self::Hybrid {
            recency_weight: 0.5,
            frequency_weight: 0.5,
        }
    }
}

/// Adaptive local codebook for a specific domain.
pub struct LocalCodebook {
    /// Domain identifier.
    domain: String,
    /// Bounded set of entries.
    entries: RwLock<Vec<CodebookEntry>>,
    /// Maximum number of entries.
    max_entries: usize,
    /// Vector dimension.
    dimension: usize,
    /// EMA decay factor.
    ema_alpha: f32,
    /// Minimum usage before pruning.
    min_usage_for_prune: u64,
    /// Pruning strategy.
    pruning_strategy: PruningStrategy,
    /// Entry ID counter.
    next_id: AtomicU64,
    /// Statistics.
    total_updates: AtomicU64,
    total_lookups: AtomicU64,
    total_prunes: AtomicU64,
    total_insertions: AtomicU64,
}

impl LocalCodebook {
    /// Create a new local codebook.
    pub fn new(
        domain: impl Into<String>,
        dimension: usize,
        max_entries: usize,
        ema_alpha: f32,
    ) -> Self {
        Self {
            domain: domain.into(),
            entries: RwLock::new(Vec::new()),
            max_entries,
            dimension,
            ema_alpha,
            min_usage_for_prune: 2,
            pruning_strategy: PruningStrategy::default(),
            next_id: AtomicU64::new(0),
            total_updates: AtomicU64::new(0),
            total_lookups: AtomicU64::new(0),
            total_prunes: AtomicU64::new(0),
            total_insertions: AtomicU64::new(0),
        }
    }

    /// Get the domain name.
    pub fn domain(&self) -> &str {
        &self.domain
    }

    /// Get the dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.read().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.read().is_empty()
    }

    /// Set the pruning strategy.
    pub fn set_pruning_strategy(&mut self, strategy: PruningStrategy) {
        self.pruning_strategy = strategy;
    }

    /// Set minimum usage for pruning.
    pub fn set_min_usage_for_prune(&mut self, min: u64) {
        self.min_usage_for_prune = min;
    }

    /// Find the nearest entry.
    pub fn quantize(&self, vector: &[f32]) -> Option<(u32, f32)> {
        self.total_lookups.fetch_add(1, Ordering::Relaxed);

        let entries = self.entries.read();
        if entries.is_empty() {
            return None;
        }

        let mut best_id = 0;
        let mut best_sim = f32::NEG_INFINITY;

        for entry in entries.iter() {
            let sim = entry.cosine_similarity(vector);
            if sim > best_sim {
                best_sim = sim;
                best_id = entry.id();
            }
        }

        Some((best_id, best_sim))
    }

    /// Quantize and update via EMA, or insert if no good match.
    pub fn quantize_and_update(&self, vector: &[f32], similarity_threshold: f32) -> (u32, f32) {
        // Try to find existing similar entry
        if let Some((id, sim)) = self.quantize(vector) {
            if sim >= similarity_threshold {
                // Update existing entry via EMA
                self.ema_update(id, vector);
                return (id, sim);
            }
        }

        // No good match - insert new entry
        let id = self.insert(vector);
        (id, 1.0) // Perfect match with itself
    }

    /// Update an entry via EMA.
    pub fn ema_update(&self, id: u32, observation: &[f32]) {
        let mut entries = self.entries.write();
        if let Some(entry) = entries.iter_mut().find(|e| e.id() == id) {
            entry.ema_update(observation, self.ema_alpha);
            self.total_updates.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Insert a new entry, pruning if necessary.
    fn insert(&self, vector: &[f32]) -> u32 {
        let mut entries = self.entries.write();

        // Prune if at capacity
        if entries.len() >= self.max_entries {
            if let Some(prune_idx) = self.select_entry_to_prune(&entries) {
                entries.remove(prune_idx);
                self.total_prunes.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Create new entry
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) as u32;
        let entry = CodebookEntry::new(id, vector.to_vec());
        entries.push(entry);

        self.total_insertions.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// Select an entry to prune based on strategy.
    fn select_entry_to_prune(&self, entries: &[CodebookEntry]) -> Option<usize> {
        let now = current_timestamp_millis();

        entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.access_count >= self.min_usage_for_prune)
            .map(|(i, e)| {
                let score = match self.pruning_strategy {
                    PruningStrategy::LRU => e.last_access as f64,
                    PruningStrategy::LFU => e.access_count as f64,
                    PruningStrategy::Hybrid {
                        recency_weight,
                        frequency_weight,
                    } => {
                        let age_secs = (now.saturating_sub(e.last_access)) / 1000;
                        let recency_score = 1.0 / (1.0 + age_secs as f64 / 60.0);
                        let frequency_score = (1.0 + e.access_count as f64).ln();
                        recency_score * recency_weight as f64
                            + frequency_score * frequency_weight as f64
                    },
                };
                (i, score)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }

    /// Check if a vector is within threshold of any entry.
    pub fn is_valid_state(&self, vector: &[f32], threshold: f32) -> bool {
        if let Some((_, sim)) = self.quantize(vector) {
            sim >= threshold
        } else {
            false
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> LocalCodebookStats {
        LocalCodebookStats {
            entry_count: self.len(),
            total_updates: self.total_updates.load(Ordering::Relaxed),
            total_lookups: self.total_lookups.load(Ordering::Relaxed),
            total_prunes: self.total_prunes.load(Ordering::Relaxed),
            total_insertions: self.total_insertions.load(Ordering::Relaxed),
        }
    }
}

/// Statistics for a local codebook.
#[derive(Debug, Clone)]
pub struct LocalCodebookStats {
    pub entry_count: usize,
    pub total_updates: u64,
    pub total_lookups: u64,
    pub total_prunes: u64,
    pub total_insertions: u64,
}

/// Result of hierarchical quantization.
#[derive(Debug, Clone)]
pub struct HierarchicalQuantization {
    /// ID in global codebook.
    pub global_entry_id: u32,
    /// Similarity to global entry.
    pub global_similarity: f32,
    /// ID in local codebook (if residual was significant).
    pub local_entry_id: Option<u32>,
    /// Similarity to local entry.
    pub local_similarity: Option<f32>,
    /// Final quantized codes.
    pub codes: Vec<u16>,
}

/// Hierarchical codebook manager.
pub struct CodebookManager {
    /// Global codebook (shared for consensus).
    global: GlobalCodebook,
    /// Local codebooks per domain.
    locals: RwLock<HashMap<String, LocalCodebook>>,
    /// Configuration.
    config: CodebookConfig,
}

/// Configuration for the codebook system.
#[derive(Debug, Clone)]
pub struct CodebookConfig {
    /// Maximum entries per local codebook.
    pub local_capacity: usize,
    /// EMA alpha for local codebook adaptation.
    pub ema_alpha: f32,
    /// Similarity threshold for considering a match.
    pub similarity_threshold: f32,
    /// Residual threshold for using local codebook.
    pub residual_threshold: f32,
    /// Validity threshold for state validation.
    pub validity_threshold: f32,
}

impl Default for CodebookConfig {
    fn default() -> Self {
        Self {
            local_capacity: 256,
            ema_alpha: 0.1,
            similarity_threshold: 0.9,
            residual_threshold: 0.05,
            validity_threshold: 0.8,
        }
    }
}

impl CodebookManager {
    /// Create a new codebook manager.
    pub fn new(global: GlobalCodebook, config: CodebookConfig) -> Self {
        Self {
            global,
            locals: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_global(global: GlobalCodebook) -> Self {
        Self::new(global, CodebookConfig::default())
    }

    /// Get the global codebook.
    pub fn global(&self) -> &GlobalCodebook {
        &self.global
    }

    /// Get or create a local codebook for a domain.
    pub fn get_or_create_local(&self, domain: &str) -> LocalCodebook {
        // Note: In a production implementation, we'd return a reference to a shared
        // LocalCodebook stored in the map. For now, we create a new one per call.
        // Proper persistence and sharing would be needed for the full implementation.
        LocalCodebook::new(
            domain,
            self.global.dimension(),
            self.config.local_capacity,
            self.config.ema_alpha,
        )
    }

    /// Perform hierarchical quantization.
    pub fn quantize(&self, domain: &str, vector: &[f32]) -> Result<HierarchicalQuantization> {
        // Step 1: Global quantization
        let (global_id, global_sim) = self
            .global
            .quantize(vector)
            .ok_or_else(|| ChainError::CodebookError("empty global codebook".to_string()))?;

        // Step 2: Compute residual
        let global_centroid = &self.global.get(global_id).unwrap().centroid;
        let residual: Vec<f32> = vector
            .iter()
            .zip(global_centroid.iter())
            .map(|(v, c)| v - c)
            .collect();

        let residual_magnitude: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Step 3: If residual is significant, use local codebook
        let (local_entry_id, local_similarity) = if residual_magnitude
            > self.config.residual_threshold
        {
            let local = self.get_or_create_local(domain);
            let (id, sim) = local.quantize_and_update(&residual, self.config.similarity_threshold);
            (Some(id), Some(sim))
        } else {
            (None, None)
        };

        // Build codes vector
        let mut codes = vec![global_id as u16];
        if let Some(local_id) = local_entry_id {
            codes.push(local_id as u16);
        }

        Ok(HierarchicalQuantization {
            global_entry_id: global_id,
            global_similarity: global_sim,
            local_entry_id,
            local_similarity,
            codes,
        })
    }

    /// Check if a state is valid (near a codebook entry).
    pub fn is_valid_state(&self, domain: &str, state: &[f32]) -> bool {
        // First check global
        if self
            .global
            .is_valid_state(state, self.config.validity_threshold)
        {
            return true;
        }

        // Then check local if it exists
        let locals = self.locals.read();
        if let Some(local) = locals.get(domain) {
            return local.is_valid_state(state, self.config.validity_threshold);
        }

        false
    }

    /// Check if a transition is valid.
    pub fn is_valid_transition(
        &self,
        domain: &str,
        from: &[f32],
        to: &[f32],
        max_distance: f32,
    ) -> bool {
        // Both states must be valid
        if !self.is_valid_state(domain, from) || !self.is_valid_state(domain, to) {
            return false;
        }

        // Compute transition magnitude
        let transition_mag: f32 = from
            .iter()
            .zip(to.iter())
            .map(|(f, t)| (t - f).powi(2))
            .sum::<f32>()
            .sqrt();

        transition_mag <= max_distance
    }
}

// ============== Helper functions ==============

fn current_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn kmeans_plusplus_init(vectors: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut centroids = Vec::with_capacity(k);
    let mut rng_state: u64 = 42;

    // Select first centroid randomly
    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
    let first_idx = (rng_state as usize) % vectors.len();
    centroids.push(vectors[first_idx].clone());

    // Select remaining centroids
    let mut distances = vec![f32::MAX; vectors.len()];

    for _ in 1..k {
        // Update distances to nearest centroid
        for (i, vector) in vectors.iter().enumerate() {
            let dist = euclidean_distance(vector, centroids.last().unwrap());
            distances[i] = distances[i].min(dist * dist);
        }

        // Select next centroid with probability proportional to distance squared
        let total_dist: f32 = distances.iter().sum();
        if total_dist == 0.0 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng_state as usize) % vectors.len();
            centroids.push(vectors[idx].clone());
        } else {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let threshold = (rng_state as f32 / u64::MAX as f32) * total_dist;

            let mut cumulative = 0.0;
            let mut selected = false;
            for (i, &d) in distances.iter().enumerate() {
                cumulative += d;
                if cumulative >= threshold {
                    centroids.push(vectors[i].clone());
                    selected = true;
                    break;
                }
            }

            // Fallback if we didn't select (numerical issues)
            if !selected {
                centroids.push(vectors[vectors.len() - 1].clone());
            }
        }
    }

    centroids
}

fn assign_to_nearest(vectors: &[Vec<f32>], centroids: &[Vec<f32>]) -> Vec<usize> {
    vectors
        .iter()
        .map(|v| {
            centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let da = euclidean_distance(v, a);
                    let db = euclidean_distance(v, b);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

fn update_centroids(
    vectors: &[Vec<f32>],
    assignments: &[usize],
    k: usize,
    dim: usize,
) -> Vec<Vec<f32>> {
    let mut centroids = vec![vec![0.0; dim]; k];
    let mut counts = vec![0usize; k];

    for (i, &cluster) in assignments.iter().enumerate() {
        for (j, &val) in vectors[i].iter().enumerate() {
            centroids[cluster][j] += val;
        }
        counts[cluster] += 1;
    }

    for (i, count) in counts.iter().enumerate() {
        if *count > 0 {
            for val in &mut centroids[i] {
                *val /= *count as f32;
            }
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_entry() {
        let entry = CodebookEntry::new(0, vec![1.0, 0.0, 0.0]);

        assert_eq!(entry.id(), 0);
        assert!((entry.magnitude() - 1.0).abs() < 0.001);

        // Cosine similarity with itself
        let sim = entry.cosine_similarity(&[1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 0.001);

        // Cosine similarity with orthogonal vector
        let sim = entry.cosine_similarity(&[0.0, 1.0, 0.0]);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_global_codebook_quantize() {
        let centroids = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let codebook = GlobalCodebook::from_centroids(centroids);

        // Query close to first centroid
        let (id, sim) = codebook.quantize(&[0.9, 0.1, 0.0]).unwrap();
        assert_eq!(id, 0);
        assert!(sim > 0.9);

        // Query close to second centroid
        let (id, sim) = codebook.quantize(&[0.1, 0.9, 0.0]).unwrap();
        assert_eq!(id, 1);
        assert!(sim > 0.9);
    }

    #[test]
    fn test_global_codebook_residual() {
        let centroids = vec![vec![1.0, 0.0]];
        let codebook = GlobalCodebook::from_centroids(centroids);

        let (id, residual) = codebook.compute_residual(&[0.8, 0.2]).unwrap();
        assert_eq!(id, 0);
        assert!((residual[0] - (-0.2)).abs() < 0.001);
        assert!((residual[1] - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_global_codebook_kmeans() {
        // Two clear clusters
        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 9.9],
        ];

        let codebook = GlobalCodebook::from_kmeans(&vectors, 2, 100);
        assert_eq!(codebook.len(), 2);
    }

    #[test]
    fn test_local_codebook() {
        let local = LocalCodebook::new("test_domain", 3, 10, 0.1);

        assert!(local.is_empty());
        assert_eq!(local.domain(), "test_domain");

        // Insert via quantize_and_update
        let (id1, _) = local.quantize_and_update(&[1.0, 0.0, 0.0], 0.9);
        assert_eq!(local.len(), 1);

        // Similar vector should match existing
        let (id2, sim) = local.quantize_and_update(&[0.99, 0.01, 0.0], 0.9);
        assert_eq!(id2, id1);
        assert!(sim > 0.9);

        // Different vector should create new entry
        let (id3, _) = local.quantize_and_update(&[0.0, 1.0, 0.0], 0.9);
        assert_ne!(id3, id1);
        assert_eq!(local.len(), 2);
    }

    #[test]
    fn test_local_codebook_pruning() {
        let mut local = LocalCodebook::new("test", 2, 2, 0.1);
        // Lower min_usage so entries can be pruned after just 1 access
        local.set_min_usage_for_prune(1);

        // Use threshold > 0 so orthogonal vectors don't match
        // Fill to capacity
        local.quantize_and_update(&[1.0, 0.0], 0.5);
        local.quantize_and_update(&[0.0, 1.0], 0.5);
        assert_eq!(local.len(), 2);

        // Access entries to increment access_count (EMA updates them)
        local.quantize_and_update(&[1.0, 0.0], 0.5);
        local.quantize_and_update(&[0.0, 1.0], 0.5);

        // Insert a new entry - should trigger pruning since both entries
        // now have access_count >= min_usage_for_prune
        local.quantize_and_update(&[-1.0, 0.0], 0.5);

        // Should still be at max capacity (pruning should have removed one)
        assert!(local.len() <= 2);
    }

    #[test]
    fn test_codebook_manager() {
        let centroids = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let global = GlobalCodebook::from_centroids(centroids);
        let manager = CodebookManager::with_global(global);

        // Quantize a vector
        let result = manager.quantize("test", &[0.9, 0.1, 0.0]).unwrap();
        assert_eq!(result.global_entry_id, 0);
        assert!(result.global_similarity > 0.9);
    }

    #[test]
    fn test_validity_check() {
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let global = GlobalCodebook::from_centroids(centroids);
        let config = CodebookConfig {
            validity_threshold: 0.9,
            ..Default::default()
        };
        let manager = CodebookManager::new(global, config);

        // Valid state (close to centroid)
        assert!(manager.is_valid_state("test", &[1.0, 0.0]));
        assert!(manager.is_valid_state("test", &[0.95, 0.05]));

        // Invalid state (too far from any centroid)
        assert!(!manager.is_valid_state("test", &[0.5, 0.5]));
    }

    #[test]
    fn test_codebook_entry_with_label() {
        let entry = CodebookEntry::new(0, vec![1.0, 0.0, 0.0]).with_label("test_label");
        assert_eq!(entry.label, Some("test_label".to_string()));
    }

    #[test]
    fn test_codebook_entry_accessors() {
        let mut entry = CodebookEntry::new(0, vec![1.0, 0.0]);
        assert_eq!(entry.access_count(), 0);

        let initial_access = entry.last_access();
        std::thread::sleep(std::time::Duration::from_millis(1));

        entry.record_access();
        assert_eq!(entry.access_count(), 1);
        assert!(entry.last_access() >= initial_access);

        assert_eq!(entry.centroid(), &[1.0, 0.0]);
    }

    #[test]
    fn test_codebook_entry_ema_update() {
        let mut entry = CodebookEntry::new(0, vec![1.0, 0.0]);
        entry.ema_update(&[0.0, 1.0], 0.5);

        // After 50% EMA update: centroid = 0.5 * [0,1] + 0.5 * [1,0] = [0.5, 0.5]
        assert!((entry.centroid[0] - 0.5).abs() < 0.001);
        assert!((entry.centroid[1] - 0.5).abs() < 0.001);
        assert_eq!(entry.access_count(), 1);
    }

    #[test]
    fn test_codebook_entry_cosine_zero_magnitude() {
        let entry = CodebookEntry::new(0, vec![0.0, 0.0, 0.0]);
        assert_eq!(entry.cosine_similarity(&[1.0, 0.0, 0.0]), 0.0);

        let entry2 = CodebookEntry::new(0, vec![1.0, 0.0, 0.0]);
        assert_eq!(entry2.cosine_similarity(&[0.0, 0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_global_codebook_empty() {
        let codebook = GlobalCodebook::new(3);
        assert!(codebook.is_empty());
        assert_eq!(codebook.len(), 0);
        assert_eq!(codebook.dimension(), 3);
        assert!(codebook.quantize(&[1.0, 0.0, 0.0]).is_none());
        assert!(!codebook.is_valid_state(&[1.0, 0.0, 0.0], 0.9));
    }

    #[test]
    fn test_global_codebook_from_centroids_with_labels() {
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let labels = vec!["x_axis".to_string(), "y_axis".to_string()];
        let codebook = GlobalCodebook::from_centroids_with_labels(centroids, labels);

        assert_eq!(codebook.len(), 2);
        assert_eq!(codebook.get(0).unwrap().label, Some("x_axis".to_string()));
        assert_eq!(codebook.get(1).unwrap().label, Some("y_axis".to_string()));
    }

    #[test]
    fn test_global_codebook_iter() {
        let centroids = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 0.0]];
        let codebook = GlobalCodebook::from_centroids(centroids);

        let count = codebook.iter().count();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_global_codebook_get() {
        let centroids = vec![vec![1.0, 0.0]];
        let codebook = GlobalCodebook::from_centroids(centroids);

        assert!(codebook.get(0).is_some());
        assert!(codebook.get(999).is_none());
    }

    #[test]
    fn test_global_codebook_is_valid_state() {
        let centroids = vec![vec![1.0, 0.0]];
        let codebook = GlobalCodebook::from_centroids(centroids);

        assert!(codebook.is_valid_state(&[1.0, 0.0], 0.9));
        assert!(!codebook.is_valid_state(&[0.0, 1.0], 0.9));
    }

    #[test]
    fn test_global_codebook_kmeans_empty() {
        let codebook = GlobalCodebook::from_kmeans(&[], 3, 10);
        assert!(codebook.is_empty());

        let codebook = GlobalCodebook::from_kmeans(&[vec![1.0, 0.0]], 0, 10);
        assert!(codebook.is_empty());
    }

    #[test]
    fn test_global_codebook_kmeans_convergence() {
        // Test early convergence
        let vectors = vec![vec![1.0, 0.0], vec![1.0, 0.0]];
        let codebook = GlobalCodebook::from_kmeans(&vectors, 1, 100);
        assert_eq!(codebook.len(), 1);
    }

    #[test]
    fn test_local_codebook_stats() {
        let local = LocalCodebook::new("test", 2, 10, 0.1);

        local.quantize_and_update(&[1.0, 0.0], 0.9);
        local.quantize_and_update(&[1.0, 0.0], 0.9);

        let stats = local.stats();
        assert_eq!(stats.entry_count, 1);
        assert_eq!(stats.total_insertions, 1);
        assert!(stats.total_updates > 0 || stats.total_lookups > 0);
    }

    #[test]
    fn test_local_codebook_is_valid_state() {
        let local = LocalCodebook::new("test", 2, 10, 0.1);
        local.quantize_and_update(&[1.0, 0.0], 0.9);

        assert!(local.is_valid_state(&[1.0, 0.0], 0.9));
        assert!(!local.is_valid_state(&[0.0, 1.0], 0.99));
    }

    #[test]
    fn test_local_codebook_quantize_empty() {
        let local = LocalCodebook::new("test", 2, 10, 0.1);
        assert!(local.quantize(&[1.0, 0.0]).is_none());
        assert!(!local.is_valid_state(&[1.0, 0.0], 0.5));
    }

    #[test]
    fn test_local_codebook_ema_update_nonexistent() {
        let local = LocalCodebook::new("test", 2, 10, 0.1);
        // Should not panic when updating non-existent entry
        local.ema_update(999, &[1.0, 0.0]);
        assert!(local.is_empty());
    }

    #[test]
    fn test_pruning_strategy_default() {
        let strategy = PruningStrategy::default();
        assert!(matches!(
            strategy,
            PruningStrategy::Hybrid {
                recency_weight: _,
                frequency_weight: _
            }
        ));
    }

    #[test]
    fn test_pruning_strategy_lru() {
        let mut local = LocalCodebook::new("test", 2, 2, 0.1);
        local.set_pruning_strategy(PruningStrategy::LRU);
        local.set_min_usage_for_prune(0);

        local.quantize_and_update(&[1.0, 0.0], 0.5);
        std::thread::sleep(std::time::Duration::from_millis(1));
        local.quantize_and_update(&[0.0, 1.0], 0.5);

        // Insert a new entry to trigger pruning
        local.quantize_and_update(&[-1.0, 0.0], 0.5);
        assert!(local.len() <= 2);
    }

    #[test]
    fn test_pruning_strategy_lfu() {
        let mut local = LocalCodebook::new("test", 2, 2, 0.1);
        local.set_pruning_strategy(PruningStrategy::LFU);
        local.set_min_usage_for_prune(0);

        local.quantize_and_update(&[1.0, 0.0], 0.5);
        local.quantize_and_update(&[1.0, 0.0], 0.5); // Access twice
        local.quantize_and_update(&[0.0, 1.0], 0.5); // Access once

        // Insert a new entry to trigger pruning
        local.quantize_and_update(&[-1.0, 0.0], 0.5);
        assert!(local.len() <= 2);
    }

    #[test]
    fn test_codebook_config_default() {
        let config = CodebookConfig::default();
        assert_eq!(config.local_capacity, 256);
        assert!((config.ema_alpha - 0.1).abs() < 0.01);
        assert!((config.similarity_threshold - 0.9).abs() < 0.01);
        assert!((config.residual_threshold - 0.05).abs() < 0.01);
        assert!((config.validity_threshold - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_codebook_manager_global_accessor() {
        let global = GlobalCodebook::from_centroids(vec![vec![1.0, 0.0]]);
        let manager = CodebookManager::with_global(global);

        assert_eq!(manager.global().len(), 1);
    }

    #[test]
    fn test_codebook_manager_quantize_empty_global() {
        let global = GlobalCodebook::new(3);
        let manager = CodebookManager::with_global(global);

        let result = manager.quantize("test", &[1.0, 0.0, 0.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_codebook_manager_quantize_with_residual() {
        // Create codebook where residual will be significant
        let centroids = vec![vec![1.0, 0.0, 0.0]];
        let global = GlobalCodebook::from_centroids(centroids);
        let config = CodebookConfig {
            residual_threshold: 0.01, // Low threshold to trigger local codebook
            ..Default::default()
        };
        let manager = CodebookManager::new(global, config);

        // Query that produces significant residual
        let result = manager.quantize("test", &[1.0, 0.5, 0.0]).unwrap();
        assert_eq!(result.global_entry_id, 0);
        assert!(result.local_entry_id.is_some());
    }

    #[test]
    fn test_codebook_manager_is_valid_transition() {
        let centroids = vec![vec![1.0, 0.0], vec![0.8, 0.2]];
        let global = GlobalCodebook::from_centroids(centroids);
        let config = CodebookConfig {
            validity_threshold: 0.8,
            ..Default::default()
        };
        let manager = CodebookManager::new(global, config);

        // Valid transition (both states valid, small distance)
        assert!(manager.is_valid_transition("test", &[1.0, 0.0], &[0.8, 0.2], 1.0));

        // Invalid transition (from invalid state)
        assert!(!manager.is_valid_transition("test", &[0.0, 0.0], &[1.0, 0.0], 2.0));

        // Invalid transition (distance too large)
        assert!(!manager.is_valid_transition("test", &[1.0, 0.0], &[0.8, 0.2], 0.01));
    }

    #[test]
    fn test_hierarchical_quantization_debug() {
        let quant = HierarchicalQuantization {
            global_entry_id: 0,
            global_similarity: 0.9,
            local_entry_id: Some(1),
            local_similarity: Some(0.8),
            codes: vec![0, 1],
        };
        let _ = format!("{:?}", quant);
    }

    #[test]
    fn test_local_codebook_stats_debug() {
        let stats = LocalCodebookStats {
            entry_count: 10,
            total_updates: 100,
            total_lookups: 200,
            total_prunes: 5,
            total_insertions: 15,
        };
        let _ = format!("{:?}", stats);
    }

    #[test]
    fn test_kmeans_plusplus_all_same() {
        // All vectors are the same - should handle gracefully
        let vectors = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        let codebook = GlobalCodebook::from_kmeans(&vectors, 2, 10);
        assert!(codebook.len() <= 2);
    }

    #[test]
    fn test_euclidean_distance() {
        let d = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]);
        assert!((d - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_codebook_entry_centroid_accessor() {
        let entry = CodebookEntry::new(0, vec![1.0, 2.0, 3.0]);
        assert_eq!(entry.centroid(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_local_codebook_dimension() {
        let local = LocalCodebook::new("test", 5, 10, 0.1);
        assert_eq!(local.dimension(), 5);
    }

    #[test]
    fn test_get_or_create_local() {
        let global = GlobalCodebook::from_centroids(vec![vec![1.0, 0.0]]);
        let manager = CodebookManager::with_global(global);

        let local = manager.get_or_create_local("domain1");
        assert_eq!(local.domain(), "domain1");
        assert!(local.is_empty());
    }

    #[test]
    fn test_global_codebook_compute_residual_empty() {
        let codebook = GlobalCodebook::new(3);
        assert!(codebook.compute_residual(&[1.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_local_codebook_insert_pruning() {
        let mut local = LocalCodebook::new("test", 2, 3, 0.1);
        local.set_min_usage_for_prune(1);

        // Fill to capacity
        local.quantize_and_update(&[1.0, 0.0], 0.5);
        local.quantize_and_update(&[0.0, 1.0], 0.5);
        local.quantize_and_update(&[-1.0, 0.0], 0.5);

        // Access all to meet min_usage_for_prune
        local.quantize_and_update(&[1.0, 0.0], 0.5);
        local.quantize_and_update(&[0.0, 1.0], 0.5);
        local.quantize_and_update(&[-1.0, 0.0], 0.5);

        let stats = local.stats();
        assert!(stats.total_insertions >= 3);
    }
}
