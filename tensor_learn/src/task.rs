// SPDX-License-Identifier: MIT OR Apache-2.0
//! Synthetic hierarchy tasks for evaluating hyperbolic learning.
//!
//! Provides [`HierarchyTask`] which generates structured graphs (e.g.
//! binary trees) and converts them into ready-to-train sessions.

use std::collections::HashSet;

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::codebook::Codebook;
use crate::error::LearnError;
use crate::poincare::PoincarePoint;
use crate::training::{TrainingConfig, TrainingSession};

/// A synthetic hierarchy with known structure for training evaluation.
///
/// Stores node keys, hierarchy levels, and directed edges as index pairs.
pub struct HierarchyTask {
    /// Node keys in BFS order.
    keys: Vec<String>,
    /// Hierarchy level for each node (root = 0).
    levels: Vec<u32>,
    /// Directed edges as `(parent_idx, child_idx)`.
    edges: Vec<(usize, usize)>,
}

impl HierarchyTask {
    /// Create a complete binary tree of the given depth.
    ///
    /// Depth 0 produces a single root node. Depth 3 produces 15 nodes
    /// across 4 levels. Nodes are named `"n0"`, `"n1"`, ... in BFS order.
    #[must_use]
    pub fn binary_tree(depth: u32) -> Self {
        let node_count = (1_usize << (depth + 1)) - 1;
        let mut keys = Vec::with_capacity(node_count);
        let mut levels = Vec::with_capacity(node_count);
        let mut edges = Vec::with_capacity(node_count.saturating_sub(1));

        for i in 0..node_count {
            keys.push(format!("n{i}"));
            levels.push(Self::level_of(i));
            if i > 0 {
                edges.push(((i - 1) / 2, i));
            }
        }

        Self {
            keys,
            levels,
            edges,
        }
    }

    /// Number of nodes in the task.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.keys.len()
    }

    /// Edge pairs as `(parent_key, child_key)` references.
    #[must_use]
    pub fn edge_pairs(&self) -> Vec<(&str, &str)> {
        self.edges
            .iter()
            .map(|&(p, c)| (self.keys[p].as_str(), self.keys[c].as_str()))
            .collect()
    }

    /// Entries as `(key, level)` references.
    #[must_use]
    pub fn entries(&self) -> Vec<(&str, u32)> {
        self.keys
            .iter()
            .zip(self.levels.iter())
            .map(|(k, &l)| (k.as_str(), l))
            .collect()
    }

    /// Sample non-edge pairs for negative training.
    ///
    /// Precomputes the full pool of unordered non-edge pairs, then samples
    /// `min(count, pool.len())` without replacement. Returns fewer than
    /// `count` when the pool is smaller (depth-0 returns empty).
    pub fn negative_samples<R: Rng>(&self, count: usize, rng: &mut R) -> Vec<(usize, usize)> {
        let edge_set: HashSet<(usize, usize)> = self
            .edges
            .iter()
            .copied()
            .flat_map(|(a, b)| [(a, b), (b, a)])
            .collect();

        let n = self.keys.len();
        let mut pool: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if !edge_set.contains(&(i, j)) {
                    pool.push((i, j));
                }
            }
        }

        let sample_count = count.min(pool.len());
        if sample_count == 0 {
            return Vec::new();
        }

        let (shuffled, _) = pool.partial_shuffle(rng, sample_count);
        shuffled.to_vec()
    }

    /// Convert this task into a ready-to-train session.
    ///
    /// Creates a [`Codebook`] with random initial embeddings (uniform in
    /// `[-0.1, 0.1]` per dimension) and all hierarchy edges. The RNG is
    /// seeded from `config.seed`.
    ///
    /// # Errors
    ///
    /// Returns an error if `config.dimension != 2` (Phase 2 constraint)
    /// or if codebook creation fails.
    pub fn into_session(self, config: TrainingConfig) -> Result<TrainingSession, LearnError> {
        if config.dimension != 2 {
            return Err(LearnError::Config(
                "dimension must be 2 for Phase 2 training".to_string(),
            ));
        }

        let mut codebook = Codebook::new(config.dimension, config.curvature)?;
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        for (i, key) in self.keys.iter().enumerate() {
            let coords: Vec<f64> = (0..config.dimension)
                .map(|_| rng.random_range(-0.1..0.1))
                .collect();
            let point = PoincarePoint::new(coords);
            codebook.add_entry(key, key, &point, self.levels[i])?;
        }

        for &(parent_idx, child_idx) in &self.edges {
            codebook.add_edge(&self.keys[parent_idx], &self.keys[child_idx], "child")?;
        }

        Ok(TrainingSession::new(codebook, config))
    }

    /// Compute BFS level from zero-based index.
    const fn level_of(index: usize) -> u32 {
        usize::BITS - (index + 1).leading_zeros() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_tree_depth_0() {
        let task = HierarchyTask::binary_tree(0);
        assert_eq!(task.node_count(), 1);
        assert!(task.edges.is_empty());
        assert_eq!(task.keys[0], "n0");
    }

    #[test]
    fn binary_tree_depth_3() {
        let task = HierarchyTask::binary_tree(3);
        assert_eq!(task.node_count(), 15);
        assert_eq!(task.edges.len(), 14);
    }

    #[test]
    fn binary_tree_levels() {
        let task = HierarchyTask::binary_tree(3);
        assert_eq!(task.levels[0], 0); // root
        assert_eq!(task.levels[1], 1);
        assert_eq!(task.levels[2], 1);
        assert_eq!(task.levels[3], 2);
        assert_eq!(task.levels[6], 2);
        assert_eq!(task.levels[7], 3);
        assert_eq!(task.levels[14], 3);
    }

    #[test]
    fn negative_samples_excludes_edges() {
        let task = HierarchyTask::binary_tree(2);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let edge_set: HashSet<(usize, usize)> = task
            .edges
            .iter()
            .copied()
            .flat_map(|(a, b)| [(a, b), (b, a)])
            .collect();
        let samples = task.negative_samples(100, &mut rng);
        for (a, b) in &samples {
            assert!(
                !edge_set.contains(&(*a, *b)),
                "negative sample ({a}, {b}) is an edge"
            );
        }
    }

    #[test]
    fn negative_samples_bounded_by_pool() {
        let task = HierarchyTask::binary_tree(1);
        // 3 nodes, 2 edges -> pool = C(3,2) - 2 = 1 pair
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let samples = task.negative_samples(1000, &mut rng);
        assert!(samples.len() <= 1, "pool has at most 1 non-edge pair");
    }

    #[test]
    fn into_session_creates_valid_session() {
        let task = HierarchyTask::binary_tree(3);
        let config = TrainingConfig {
            total_steps: 10,
            dimension: 2,
            ..TrainingConfig::default()
        };
        let session = task.into_session(config).unwrap();
        assert_eq!(session.codebook().len(), 15);
    }

    #[test]
    fn into_session_random_init_inside_disk() {
        let task = HierarchyTask::binary_tree(3);
        let config = TrainingConfig {
            total_steps: 10,
            dimension: 2,
            ..TrainingConfig::default()
        };
        let session = task.into_session(config).unwrap();
        let viz = session.to_viz_data().unwrap();
        for node in &viz.nodes {
            let r = node.x.hypot(node.y);
            assert!(
                r < 1.0,
                "node {} at ({}, {}) outside disk",
                node.id,
                node.x,
                node.y
            );
        }
    }

    #[test]
    fn into_session_rejects_non_2d() {
        let task = HierarchyTask::binary_tree(2);
        let config = TrainingConfig {
            dimension: 3,
            ..TrainingConfig::default()
        };
        let result = task.into_session(config);
        assert!(result.is_err());
    }
}
