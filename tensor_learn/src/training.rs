// SPDX-License-Identifier: MIT OR Apache-2.0
//! Training session for the geometric intelligence model.
//!
//! Manages iterative Riemannian SGD training with step-by-step progress
//! tracking, loss history, and visualization export. Each step computes
//! edge-attraction, negative-repulsion, and level-radial losses, then
//! applies a Riemannian gradient update to move points in the Poincare disk.

use std::collections::{HashMap, HashSet};

use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::codebook::Codebook;
use crate::error::LearnError;
use crate::poincare::PoincarePoint;
use crate::riemannian::{euclidean_grad_distance, riemannian_update};
use crate::viz::VizData;

/// Training status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingStatus {
    /// Training has not started.
    Idle,
    /// Training is actively running.
    Running,
    /// Training is paused.
    Paused,
    /// Training has completed.
    Completed,
}

/// Training statistics.
///
/// Cloning this struct copies the loss history vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStats {
    /// Current training step.
    pub step: usize,
    /// Total number of steps to run.
    pub total_steps: usize,
    /// Current status.
    pub status: TrainingStatus,
    /// Loss at the current step (lower is better).
    pub loss: f64,
    /// Best loss seen so far.
    pub best_loss: f64,
    /// Number of entries in the codebook.
    pub entry_count: usize,
    /// Edge-attraction loss component (last step).
    pub edge_loss: f64,
    /// Level-radial loss component (last step).
    pub radial_loss: f64,
    /// Negative-sampling loss component (last step).
    pub negative_loss: f64,
    /// Loss value at each completed step.
    pub loss_history: Vec<f64>,
}

/// Loss broken down by component, returned from gradient computation.
struct LossComponents {
    /// Total unnormalized loss.
    total: f64,
    /// Edge-attraction loss.
    edge: f64,
    /// Level-radial loss.
    radial: f64,
    /// Negative-sampling hinge loss.
    negative: f64,
}

/// Configuration for a training session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Total number of training steps.
    pub total_steps: usize,
    /// Learning rate for gradient updates.
    pub learning_rate: f64,
    /// Curvature of the Poincare disk.
    pub curvature: f64,
    /// Dimension of the embedding space.
    pub dimension: usize,
    /// RNG seed for reproducible training.
    pub seed: u64,
    /// Weight for edge-attraction loss.
    pub edge_loss_weight: f64,
    /// Weight for level-radial loss.
    pub level_loss_weight: f64,
    /// Number of negative samples per step.
    pub negative_samples: usize,
    /// Margin for negative-sampling hinge loss.
    pub margin: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            total_steps: 100,
            learning_rate: 0.01,
            curvature: 1.0,
            dimension: 2,
            seed: 42,
            edge_loss_weight: 1.0,
            level_loss_weight: 0.5,
            negative_samples: 10,
            margin: 1.5,
        }
    }
}

/// A training session managing codebook learning via Riemannian SGD.
pub struct TrainingSession {
    /// The codebook being trained.
    codebook: Codebook,
    /// Training configuration.
    config: TrainingConfig,
    /// Current step.
    step: usize,
    /// Current status.
    status: TrainingStatus,
    /// Current loss.
    loss: f64,
    /// Best loss seen.
    best_loss: f64,
    /// Stable key order from codebook entries at init time.
    ordered_keys: Vec<String>,
    /// Hierarchy level for each key (parallel to `ordered_keys`).
    ordered_levels: Vec<u32>,
    /// Working-copy embeddings in f64 (parallel to `ordered_keys`).
    embeddings: Vec<PoincarePoint>,
    /// Cached edge pairs as index pairs into `ordered_keys`.
    training_edges: Vec<(usize, usize)>,
    /// Precomputed pool of non-edge index pairs for negative sampling.
    negative_pool: Vec<(usize, usize)>,
    /// Loss value at each completed step.
    loss_history: Vec<f64>,
    /// Last step edge loss (unnormalized).
    edge_loss: f64,
    /// Last step radial loss (unnormalized).
    radial_loss: f64,
    /// Last step negative loss (unnormalized).
    negative_loss: f64,
}

impl TrainingSession {
    /// Create a new training session with the given codebook and config.
    #[must_use]
    pub const fn new(codebook: Codebook, config: TrainingConfig) -> Self {
        Self {
            codebook,
            config,
            step: 0,
            status: TrainingStatus::Idle,
            loss: f64::MAX,
            best_loss: f64::MAX,
            ordered_keys: Vec::new(),
            ordered_levels: Vec::new(),
            embeddings: Vec::new(),
            training_edges: Vec::new(),
            negative_pool: Vec::new(),
            loss_history: Vec::new(),
            edge_loss: 0.0,
            radial_loss: 0.0,
            negative_loss: 0.0,
        }
    }

    /// Start the training session.
    pub const fn start(&mut self) {
        self.status = TrainingStatus::Running;
    }

    /// Pause the training session.
    pub fn pause(&mut self) {
        if self.status == TrainingStatus::Running {
            self.status = TrainingStatus::Paused;
        }
    }

    /// Resume a paused session.
    pub fn resume(&mut self) {
        if self.status == TrainingStatus::Paused {
            self.status = TrainingStatus::Running;
        }
    }

    /// Execute a single Riemannian SGD training step.
    ///
    /// On first call, lazily loads embeddings, edges, and the negative pool
    /// from the codebook. Each step accumulates Euclidean gradients from
    /// edge-attraction, negative-repulsion, and level-radial losses, then
    /// applies the Riemannian update and flushes results back to the
    /// vector engine.
    ///
    /// # Errors
    ///
    /// Returns an error if reading or writing codebook state fails.
    pub fn step(&mut self) -> Result<(), LearnError> {
        if self.status == TrainingStatus::Completed {
            return Ok(());
        }

        if self.status != TrainingStatus::Running {
            self.start();
        }

        // Lazy init on first call
        if self.ordered_keys.is_empty() && !self.codebook.is_empty() {
            self.load_embeddings()?;
            self.load_training_edges()?;
            self.build_negative_pool();
        }

        // Empty codebook fast path
        if self.ordered_keys.is_empty() {
            self.advance_step(0.0);
            return Ok(());
        }

        let lc = self.compute_and_apply_gradients()?;

        // Normalize and record
        #[allow(clippy::cast_precision_loss)]
        let n_f64 = self.ordered_keys.len() as f64;
        self.edge_loss = lc.edge / n_f64;
        self.radial_loss = lc.radial / n_f64;
        self.negative_loss = lc.negative / n_f64;
        self.advance_step(lc.total / n_f64);

        Ok(())
    }

    /// Get current training statistics.
    ///
    /// Clones the loss history; negligible cost for typical run lengths.
    #[must_use]
    pub fn stats(&self) -> TrainingStats {
        TrainingStats {
            step: self.step,
            total_steps: self.config.total_steps,
            status: self.status,
            loss: self.loss,
            best_loss: self.best_loss,
            entry_count: self.codebook.len(),
            edge_loss: self.edge_loss,
            radial_loss: self.radial_loss,
            negative_loss: self.negative_loss,
            loss_history: self.loss_history.clone(),
        }
    }

    /// Get current status.
    #[must_use]
    pub const fn status(&self) -> TrainingStatus {
        self.status
    }

    /// Export current state as visualization data.
    ///
    /// # Errors
    ///
    /// Returns an error if reading codebook state fails.
    pub fn to_viz_data(&self) -> Result<VizData, LearnError> {
        self.codebook.to_viz_data()
    }

    /// Access the codebook.
    #[must_use]
    pub const fn codebook(&self) -> &Codebook {
        &self.codebook
    }

    /// Access the training config.
    #[must_use]
    pub const fn config(&self) -> &TrainingConfig {
        &self.config
    }

    // -- private helpers --

    /// Compute all gradient components and apply a Riemannian update.
    fn compute_and_apply_gradients(&mut self) -> Result<LossComponents, LearnError> {
        let n = self.ordered_keys.len();
        let dim = self.config.dimension;
        let curvature = self.config.curvature;
        let edge_w = self.config.edge_loss_weight;
        let level_w = self.config.level_loss_weight;
        let margin = self.config.margin;
        let lr = self.config.learning_rate;

        let mut grads: Vec<Vec<f64>> = vec![vec![0.0; dim]; n];
        let mut edge_loss = 0.0;
        let mut radial_loss = 0.0;
        let mut negative_loss = 0.0;

        // a. Edge-attraction loss: minimize squared distance between connected nodes
        for &(fi, ti) in &self.training_edges {
            let dist = self.embeddings[fi].distance(&self.embeddings[ti], curvature);
            edge_loss += edge_w * dist * dist;

            let g_from = euclidean_grad_distance(
                self.embeddings[fi].coords(),
                self.embeddings[ti].coords(),
                curvature,
            );
            let g_to = euclidean_grad_distance(
                self.embeddings[ti].coords(),
                self.embeddings[fi].coords(),
                curvature,
            );

            let coeff = edge_w * 2.0 * dist;
            accumulate(&mut grads[fi], &g_from, coeff);
            accumulate(&mut grads[ti], &g_to, coeff);
        }

        // b. Negative-sampling repulsion: push non-edges apart beyond margin
        if !self.negative_pool.is_empty() {
            #[allow(clippy::cast_possible_truncation)]
            let seed = self.config.seed.wrapping_add(self.step as u64);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let k = self.config.negative_samples.min(self.negative_pool.len());

            let mut indices: Vec<usize> = (0..self.negative_pool.len()).collect();
            indices.shuffle(&mut rng);
            indices.truncate(k);

            for &idx in &indices {
                let (ai, bi) = self.negative_pool[idx];
                let dist = self.embeddings[ai].distance(&self.embeddings[bi], curvature);
                let violation = (margin - dist).max(0.0);

                if violation > 0.0 {
                    negative_loss += edge_w * violation * violation;

                    let g_a = euclidean_grad_distance(
                        self.embeddings[ai].coords(),
                        self.embeddings[bi].coords(),
                        curvature,
                    );
                    let g_b = euclidean_grad_distance(
                        self.embeddings[bi].coords(),
                        self.embeddings[ai].coords(),
                        curvature,
                    );

                    let coeff = edge_w * 2.0 * violation;
                    accumulate(&mut grads[ai], &g_a, -coeff);
                    accumulate(&mut grads[bi], &g_b, -coeff);
                }
            }
        }

        // c. Level-radial loss: push nodes to target hyperbolic distance from origin
        let origin = PoincarePoint::origin(dim);
        for (i, &level) in self.ordered_levels.iter().enumerate() {
            let target = f64::from(level) * 0.4;
            let actual = self.embeddings[i].distance(&origin, curvature);
            let diff = actual - target;
            radial_loss += level_w * diff * diff;

            if actual > 1e-10 {
                let g = euclidean_grad_distance(
                    self.embeddings[i].coords(),
                    origin.coords(),
                    curvature,
                );
                accumulate(&mut grads[i], &g, level_w * 2.0 * diff);
            }
        }

        // Riemannian SGD update (stable ordered_keys iteration)
        for (i, grad) in grads.iter().enumerate() {
            let new_point = riemannian_update(&self.embeddings[i], grad, lr, curvature);
            self.embeddings[i] = new_point;
        }

        // Flush updated embeddings to the vector engine
        self.flush_embeddings()?;

        Ok(LossComponents {
            total: edge_loss + radial_loss + negative_loss,
            edge: edge_loss,
            radial: radial_loss,
            negative: negative_loss,
        })
    }

    /// Read f32 embeddings from the vector engine and convert to f64.
    fn load_embeddings(&mut self) -> Result<(), LearnError> {
        let entries = self.codebook.entries();
        self.ordered_keys.reserve(entries.len());
        self.ordered_levels.reserve(entries.len());
        self.embeddings.reserve(entries.len());

        for info in &entries {
            let f32_vec = self.codebook.get_embedding(info.key)?;
            let f64_coords: Vec<f64> = f32_vec.iter().map(|&v| f64::from(v)).collect();
            self.ordered_keys.push(info.key.to_string());
            self.ordered_levels.push(info.level);
            self.embeddings.push(PoincarePoint::new(f64_coords));
        }
        Ok(())
    }

    /// Cache edge pairs from the codebook as index pairs.
    fn load_training_edges(&mut self) -> Result<(), LearnError> {
        let key_to_idx: HashMap<&str, usize> = self
            .ordered_keys
            .iter()
            .enumerate()
            .map(|(i, k)| (k.as_str(), i))
            .collect();

        let pairs = self.codebook.edge_pairs()?;
        for (from_key, to_key) in &pairs {
            if let (Some(&fi), Some(&ti)) = (
                key_to_idx.get(from_key.as_str()),
                key_to_idx.get(to_key.as_str()),
            ) {
                self.training_edges.push((fi, ti));
            }
        }
        Ok(())
    }

    /// Build the pool of non-edge index pairs for negative sampling.
    fn build_negative_pool(&mut self) {
        let edge_set: HashSet<(usize, usize)> = self
            .training_edges
            .iter()
            .copied()
            .flat_map(|(a, b)| [(a, b), (b, a)])
            .collect();

        let n = self.ordered_keys.len();
        for i in 0..n {
            for j in (i + 1)..n {
                if !edge_set.contains(&(i, j)) {
                    self.negative_pool.push((i, j));
                }
            }
        }
    }

    /// Write f64 working-copy embeddings back to the vector engine as f32.
    fn flush_embeddings(&self) -> Result<(), LearnError> {
        for (i, key) in self.ordered_keys.iter().enumerate() {
            self.codebook.update_embedding(key, &self.embeddings[i])?;
        }
        Ok(())
    }

    /// Advance step counter, record loss, and check completion.
    fn advance_step(&mut self, normalized_loss: f64) {
        self.loss = normalized_loss;
        if normalized_loss < self.best_loss {
            self.best_loss = normalized_loss;
        }
        self.loss_history.push(normalized_loss);
        self.step += 1;
        if self.step >= self.config.total_steps {
            self.status = TrainingStatus::Completed;
        }
    }
}

/// Accumulate `coeff * src[i]` into `dst[i]` for all dimensions.
fn accumulate(dst: &mut [f64], src: &[f64], coeff: f64) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += coeff * s;
    }
}

#[cfg(test)]
mod tests {
    use crate::poincare::PoincarePoint;
    use crate::task::HierarchyTask;

    use super::*;

    fn make_session() -> TrainingSession {
        let cb = Codebook::new(2, 1.0).unwrap();
        let config = TrainingConfig {
            total_steps: 10,
            ..TrainingConfig::default()
        };
        TrainingSession::new(cb, config)
    }

    fn make_session_with_entries() -> TrainingSession {
        let mut cb = Codebook::new(2, 1.0).unwrap();

        cb.add_entry("root", "Root", &PoincarePoint::origin(2), 0)
            .unwrap();
        cb.add_entry("child", "Child", &PoincarePoint::new(vec![0.3, 0.0]), 1)
            .unwrap();
        cb.add_edge("root", "child", "contains").unwrap();

        let config = TrainingConfig {
            total_steps: 5,
            ..TrainingConfig::default()
        };
        TrainingSession::new(cb, config)
    }

    fn make_tree_session(total_steps: usize, lr: f64) -> TrainingSession {
        let task = HierarchyTask::binary_tree(3);
        let config = TrainingConfig {
            total_steps,
            learning_rate: lr,
            dimension: 2,
            curvature: 1.0,
            seed: 42,
            ..TrainingConfig::default()
        };
        task.into_session(config).unwrap()
    }

    #[test]
    fn test_new_session() {
        let session = make_session();
        assert_eq!(session.status(), TrainingStatus::Idle);
        assert_eq!(session.stats().step, 0);
    }

    #[test]
    fn test_start_pause_resume() {
        let mut session = make_session();
        session.start();
        assert_eq!(session.status(), TrainingStatus::Running);
        session.pause();
        assert_eq!(session.status(), TrainingStatus::Paused);
        session.resume();
        assert_eq!(session.status(), TrainingStatus::Running);
    }

    #[test]
    fn test_pause_when_idle_does_nothing() {
        let mut session = make_session();
        session.pause();
        assert_eq!(session.status(), TrainingStatus::Idle);
    }

    #[test]
    fn test_resume_when_running_does_nothing() {
        let mut session = make_session();
        session.start();
        session.resume();
        assert_eq!(session.status(), TrainingStatus::Running);
    }

    #[test]
    fn test_step_auto_starts() {
        let mut session = make_session();
        session.step().unwrap();
        assert_eq!(session.stats().step, 1);
    }

    #[test]
    fn test_step_with_entries() {
        let mut session = make_session_with_entries();
        session.step().unwrap();
        let stats = session.stats();
        assert_eq!(stats.step, 1);
        assert_eq!(stats.entry_count, 2);
        assert!(stats.loss.is_finite());
    }

    #[test]
    fn test_training_completes() {
        let mut session = make_session();
        for _ in 0..10 {
            session.step().unwrap();
        }
        assert_eq!(session.status(), TrainingStatus::Completed);
    }

    #[test]
    fn test_stats_serialization() {
        let session = make_session();
        let stats = session.stats();
        let json = serde_json::to_string(&stats).unwrap();
        let recovered: TrainingStats = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.step, 0);
    }

    #[test]
    fn test_viz_data() {
        let session = make_session_with_entries();
        let viz = session.to_viz_data().unwrap();
        assert_eq!(viz.nodes.len(), 2);
        assert_eq!(viz.edges.len(), 1);
    }

    #[test]
    fn test_config_access() {
        let session = make_session();
        assert_eq!(session.config().total_steps, 10);
    }

    #[test]
    fn test_codebook_access() {
        let session = make_session();
        assert!(session.codebook().is_empty());
    }

    #[test]
    fn test_best_loss_tracks_minimum() {
        let mut session = make_session_with_entries();
        session.step().unwrap();
        let first_loss = session.stats().loss;
        session.step().unwrap();
        assert!(session.stats().best_loss <= first_loss);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.total_steps, 100);
        assert_eq!(config.dimension, 2);
        assert_eq!(config.seed, 42);
        assert!((config.curvature - 1.0).abs() < 1e-10);
        assert!((config.edge_loss_weight - 1.0).abs() < 1e-10);
        assert!((config.level_loss_weight - 0.5).abs() < 1e-10);
        assert_eq!(config.negative_samples, 10);
        assert!((config.margin - 1.5).abs() < 1e-10);
    }

    #[test]
    fn step_moves_points() {
        let mut session = make_session_with_entries();
        let viz_before = session.to_viz_data().unwrap();
        session.step().unwrap();
        let viz_after = session.to_viz_data().unwrap();

        // At least one node should have moved
        let moved = viz_before
            .nodes
            .iter()
            .zip(viz_after.nodes.iter())
            .any(|(a, b)| (a.x - b.x).abs() > 1e-10 || (a.y - b.y).abs() > 1e-10);
        assert!(moved, "at least one point should move after a step");
    }

    #[test]
    fn loss_decreases() {
        let mut session = make_tree_session(50, 0.1);
        session.step().unwrap();
        let loss_1 = session.stats().loss;
        for _ in 1..50 {
            session.step().unwrap();
        }
        let loss_50 = session.stats().loss;
        assert!(
            loss_50 < loss_1,
            "loss should decrease: step1={loss_1}, step50={loss_50}"
        );
    }

    #[test]
    fn convergence_root_near_origin() {
        let mut session = make_tree_session(100, 0.1);
        for _ in 0..100 {
            session.step().unwrap();
        }
        let viz = session.to_viz_data().unwrap();
        let root = viz.nodes.iter().find(|n| n.id == "n0").unwrap();
        for node in &viz.nodes {
            if node.level == 3 {
                assert!(
                    root.distance_from_origin < node.distance_from_origin,
                    "root dist ({}) should be less than leaf {} dist ({})",
                    root.distance_from_origin,
                    node.id,
                    node.distance_from_origin,
                );
            }
        }
    }

    #[test]
    fn convergence_hierarchy_ordering() {
        let mut session = make_tree_session(100, 0.1);
        for _ in 0..100 {
            session.step().unwrap();
        }
        let viz = session.to_viz_data().unwrap();

        // For each edge, parent should generally be closer to origin
        let mut correct = 0;
        let mut total = 0;
        for edge in &viz.edges {
            let parent = viz.nodes.iter().find(|n| n.id == edge.from).unwrap();
            let child = viz.nodes.iter().find(|n| n.id == edge.to).unwrap();
            total += 1;
            if parent.distance_from_origin < child.distance_from_origin {
                correct += 1;
            }
        }
        assert!(
            correct * 2 > total,
            "majority of edges should have parent closer: {correct}/{total}"
        );
    }

    #[test]
    fn loss_history_populated() {
        let mut session = make_tree_session(20, 0.1);
        for _ in 0..20 {
            session.step().unwrap();
        }
        assert_eq!(session.stats().loss_history.len(), 20);
    }

    #[test]
    fn training_reproducible() {
        let make = || {
            let task = HierarchyTask::binary_tree(3);
            let config = TrainingConfig {
                total_steps: 30,
                learning_rate: 0.1,
                seed: 42,
                dimension: 2,
                ..TrainingConfig::default()
            };
            let mut s = task.into_session(config).unwrap();
            for _ in 0..30 {
                s.step().unwrap();
            }
            s
        };

        let s1 = make();
        let s2 = make();

        assert!(
            (s1.stats().loss - s2.stats().loss).abs() < 1e-12,
            "same seed must produce identical loss"
        );

        let v1 = s1.to_viz_data().unwrap();
        let v2 = s2.to_viz_data().unwrap();
        for (a, b) in v1.nodes.iter().zip(v2.nodes.iter()) {
            assert!(
                (a.x - b.x).abs() < 1e-10 && (a.y - b.y).abs() < 1e-10,
                "same seed must produce identical positions"
            );
        }
    }

    #[test]
    fn step_with_empty_codebook() {
        let mut session = make_session();
        session.step().unwrap();
        assert_eq!(session.stats().step, 1);
        assert!((session.stats().loss - 0.0).abs() < 1e-12);
    }

    #[test]
    fn completed_status_at_total_steps() {
        let mut session = make_tree_session(5, 0.1);
        for _ in 0..5 {
            session.step().unwrap();
        }
        assert_eq!(session.status(), TrainingStatus::Completed);
        // Extra step should be a no-op
        let loss_before = session.stats().loss;
        session.step().unwrap();
        assert_eq!(session.stats().step, 5);
        assert!((session.stats().loss - loss_before).abs() < 1e-12);
    }
}
