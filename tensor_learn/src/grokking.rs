// SPDX-License-Identifier: MIT OR Apache-2.0
//! Grokking experiment: modular arithmetic on the Poincare disk.
//!
//! Trains a small MLP with hyperbolic embeddings on `a + b mod p`.
//! The model first memorizes the training set, then (after many steps)
//! suddenly generalizes to the held-out test set -- the grokking phenomenon.
//!
//! The embeddings live on the Poincare disk. Weight decay pulls them
//! toward the origin. After grokking, they self-organize into structure
//! that captures the cyclic group Z/pZ.

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::training::TrainingStatus;
use crate::viz::{VizData, VizEdge, VizMetadata, VizNode};

/// Maximum Poincare disk radius for projection.
const MAX_NORM: f64 = 1.0 - 1e-6;

/// Configuration for a grokking experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrokConfig {
    /// Prime modulus for the arithmetic task.
    pub prime: usize,
    /// Embedding dimension per number (2 = pure Poincare, higher = richer).
    pub embed_dim: usize,
    /// Hidden dimension of the MLP.
    pub hidden_dim: usize,
    /// Fraction of pairs used for training (rest is test).
    pub train_fraction: f64,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// `AdamW` weight decay (1.0 is standard for grokking).
    pub weight_decay: f64,
    /// Poincare disk curvature.
    pub curvature: f64,
    /// RNG seed.
    pub seed: u64,
    /// Total training steps.
    pub total_steps: usize,
    /// Steps to advance per API call when running.
    pub steps_per_tick: usize,
}

impl Default for GrokConfig {
    fn default() -> Self {
        Self {
            prime: 31,
            embed_dim: 32,
            hidden_dim: 128,
            train_fraction: 0.5,
            learning_rate: 1e-3,
            weight_decay: 1.0,
            curvature: 1.0,
            seed: 42,
            total_steps: 50_000,
            steps_per_tick: 20,
        }
    }
}

/// Statistics from a grokking experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrokStats {
    /// Current step.
    pub step: usize,
    /// Total steps.
    pub total_steps: usize,
    /// Session status.
    pub status: TrainingStatus,
    /// Training loss (cross-entropy).
    pub train_loss: f64,
    /// Test loss.
    pub test_loss: f64,
    /// Training accuracy (fraction correct).
    pub train_acc: f64,
    /// Test accuracy (fraction correct).
    pub test_acc: f64,
    /// Training loss history (one entry per evaluation).
    pub train_loss_history: Vec<f64>,
    /// Test loss history.
    pub test_loss_history: Vec<f64>,
    /// Training accuracy history.
    pub train_acc_history: Vec<f64>,
    /// Test accuracy history.
    pub test_acc_history: Vec<f64>,
}

// -- MLP weight layout --
// input_dim = 2 * embed_dim
// W1: [0 .. hidden*input_dim)
// b1: [hidden*input_dim .. hidden*input_dim + hidden)
// W2: [.. + hidden .. + hidden + p*hidden)
// b2: [.. + p*hidden .. end)

/// Total number of MLP weight parameters.
const fn weight_count(prime: usize, hidden: usize, input_dim: usize) -> usize {
    hidden * input_dim + hidden + prime * hidden + prime
}

/// Byte range for W1 in the flat weight vector.
const fn w1_slice(hidden: usize, input_dim: usize) -> std::ops::Range<usize> {
    0..hidden * input_dim
}

/// Byte range for b1.
const fn b1_slice(hidden: usize, input_dim: usize) -> std::ops::Range<usize> {
    let off = hidden * input_dim;
    off..off + hidden
}

/// Byte range for W2.
const fn w2_slice(prime: usize, hidden: usize, input_dim: usize) -> std::ops::Range<usize> {
    let off = hidden * input_dim + hidden;
    off..off + prime * hidden
}

/// Byte range for b2.
const fn b2_slice(prime: usize, hidden: usize, input_dim: usize) -> std::ops::Range<usize> {
    let off = hidden * input_dim + hidden + prime * hidden;
    off..off + prime
}

/// A grokking experiment session.
pub struct GrokSession {
    config: GrokConfig,
    /// 2D Poincare embeddings, flat: `[x0, y0, x1, y1, ...]`.
    embeds: Vec<f64>,
    /// MLP weights (W1, b1, W2, b2 concatenated).
    weights: Vec<f64>,
    /// Adam state for embeddings.
    embed_m: Vec<f64>,
    embed_v: Vec<f64>,
    /// Adam state for MLP weights.
    weight_m: Vec<f64>,
    weight_v: Vec<f64>,
    /// Training triples `(a, b, c)` where `c = (a+b) % p`.
    train: Vec<(usize, usize, usize)>,
    /// Test triples.
    test: Vec<(usize, usize, usize)>,
    step: usize,
    status: TrainingStatus,
    train_loss: f64,
    test_loss: f64,
    train_acc: f64,
    test_acc: f64,
    train_loss_hist: Vec<f64>,
    test_loss_hist: Vec<f64>,
    train_acc_hist: Vec<f64>,
    test_acc_hist: Vec<f64>,
}

impl GrokSession {
    /// Create a new grokking experiment.
    #[must_use]
    pub fn new(config: GrokConfig) -> Self {
        let prime = config.prime;
        let edim = config.embed_dim;
        let hidden = config.hidden_dim;
        let input_dim = edim * 2;
        let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);

        // Random embeddings in [-0.05, 0.05]
        let embeds: Vec<f64> = (0..prime * edim)
            .map(|_| rng.random_range(-0.05..0.05))
            .collect();

        // Xavier init for MLP weights
        let wc = weight_count(prime, hidden, input_dim);
        let mut weights = vec![0.0; wc];
        #[allow(clippy::cast_precision_loss)]
        let scale1 = (2.0 / input_dim as f64).sqrt();
        #[allow(clippy::cast_precision_loss)]
        let scale2 = (2.0 / hidden as f64).sqrt();
        for val in &mut weights[w1_slice(hidden, input_dim)] {
            *val = rng.random_range(-scale1..scale1);
        }
        for val in &mut weights[w2_slice(prime, hidden, input_dim)] {
            *val = rng.random_range(-scale2..scale2);
        }

        // Build all pairs and split train/test
        let mut all: Vec<(usize, usize, usize)> = Vec::with_capacity(prime * prime);
        for a_val in 0..prime {
            for b_val in 0..prime {
                all.push((a_val, b_val, (a_val + b_val) % prime));
            }
        }
        all.shuffle(&mut rng);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let split = (all.len() as f64 * config.train_fraction) as usize;
        let test = all.split_off(split);
        let train = all;

        let embed_len = embeds.len();
        Self {
            config,
            embeds,
            weights: weights.clone(),
            embed_m: vec![0.0; embed_len],
            embed_v: vec![0.0; embed_len],
            weight_m: vec![0.0; weights.len()],
            weight_v: vec![0.0; weights.len()],
            train,
            test,
            step: 0,
            status: TrainingStatus::Idle,
            train_loss: f64::MAX,
            test_loss: f64::MAX,
            train_acc: 0.0,
            test_acc: 0.0,
            train_loss_hist: Vec::new(),
            test_loss_hist: Vec::new(),
            train_acc_hist: Vec::new(),
            test_acc_hist: Vec::new(),
        }
    }

    /// Start or resume.
    pub fn start(&mut self) {
        if self.status != TrainingStatus::Completed {
            self.status = TrainingStatus::Running;
        }
    }

    /// Pause.
    pub fn pause(&mut self) {
        if self.status == TrainingStatus::Running {
            self.status = TrainingStatus::Paused;
        }
    }

    /// Current status.
    #[must_use]
    pub const fn status(&self) -> TrainingStatus {
        self.status
    }

    /// Advance multiple steps (used by poll loop).
    pub fn step(&mut self) {
        if self.status == TrainingStatus::Completed {
            return;
        }
        if self.status != TrainingStatus::Running {
            self.status = TrainingStatus::Running;
        }

        let ticks = self.config.steps_per_tick;
        for _ in 0..ticks {
            if self.step >= self.config.total_steps {
                self.status = TrainingStatus::Completed;
                break;
            }
            self.train_one_step();
            self.step += 1;
        }
        self.record_metrics();
        if self.step >= self.config.total_steps {
            self.status = TrainingStatus::Completed;
        }
    }

    /// Advance exactly one step (used by Step button).
    pub fn single_step(&mut self) {
        if self.status == TrainingStatus::Completed {
            return;
        }
        if self.step < self.config.total_steps {
            self.train_one_step();
            self.step += 1;
        }
        self.record_metrics();
        if self.step >= self.config.total_steps {
            self.status = TrainingStatus::Completed;
        }
    }

    /// Current statistics.
    #[must_use]
    pub fn stats(&self) -> GrokStats {
        GrokStats {
            step: self.step,
            total_steps: self.config.total_steps,
            status: self.status,
            train_loss: self.train_loss,
            test_loss: self.test_loss,
            train_acc: self.train_acc,
            test_acc: self.test_acc,
            train_loss_history: self.train_loss_hist.clone(),
            test_loss_history: self.test_loss_hist.clone(),
            train_acc_history: self.train_acc_hist.clone(),
            test_acc_history: self.test_acc_hist.clone(),
        }
    }

    /// Export embeddings as viz data, projected to 2D via PCA.
    #[must_use]
    pub fn to_viz_data(&self) -> VizData {
        let prime = self.config.prime;
        let edim = self.config.embed_dim;
        let projected = pca_project_2d(&self.embeds, prime, edim);

        let nodes: Vec<VizNode> = (0..prime)
            .map(|idx| {
                let (px, py) = projected[idx];
                let norm_sq: f64 = self.embeds[idx * edim..(idx + 1) * edim]
                    .iter()
                    .map(|v| v * v)
                    .sum();
                VizNode {
                    id: format!("{idx}"),
                    label: format!("{idx}"),
                    x: px,
                    y: py,
                    #[allow(clippy::cast_possible_truncation)]
                    level: (idx * 6 / prime) as u32,
                    distance_from_origin: norm_sq.sqrt(),
                }
            })
            .collect();

        // Connect consecutive numbers so circular structure is visible
        let edges: Vec<VizEdge> = (0..prime)
            .map(|idx| {
                let next = (idx + 1) % prime;
                VizEdge {
                    from: format!("{idx}"),
                    to: format!("{next}"),
                    label: String::new(),
                }
            })
            .collect();

        let edge_count = edges.len();
        VizData {
            nodes,
            edges,
            metadata: VizMetadata {
                entry_count: prime,
                edge_count,
                curvature: self.config.curvature,
                dimension: 2,
            },
        }
    }

    // -- Core training --

    /// One full gradient step over the training set.
    fn train_one_step(&mut self) {
        let prime = self.config.prime;
        let lr = self.config.learning_rate;
        let wd = self.config.weight_decay;

        let mut eg = vec![0.0; self.embeds.len()];
        let mut wg = vec![0.0; self.weights.len()];

        let train_copy = self.train.clone();
        for &(av, bv, cv) in &train_copy {
            self.backward(av, bv, cv, &mut eg, &mut wg);
        }

        // Normalize by batch size
        #[allow(clippy::cast_precision_loss)]
        let batch = train_copy.len() as f64;
        for val in &mut eg {
            *val /= batch;
        }
        for val in &mut wg {
            *val /= batch;
        }

        // AdamW updates
        let cur_step = self.step;
        adamw_update(
            &mut self.embeds,
            &eg,
            &mut self.embed_m,
            &mut self.embed_v,
            lr,
            wd,
            cur_step,
        );
        adamw_update(
            &mut self.weights,
            &wg,
            &mut self.weight_m,
            &mut self.weight_v,
            lr,
            wd,
            cur_step,
        );

        // Project full embedding onto the Poincare ball (all dims, norm < 1)
        let edim = self.config.embed_dim;
        for idx in 0..prime {
            let base = idx * edim;
            let norm_sq: f64 = self.embeds[base..base + edim].iter().map(|v| v * v).sum();
            let norm = norm_sq.sqrt();
            if norm >= MAX_NORM {
                let scale = MAX_NORM / norm;
                for dim in 0..edim {
                    self.embeds[base + dim] *= scale;
                }
            }
        }

        // Shuffle training data for next epoch
        if (self.step + 1).is_multiple_of(prime * prime) {
            let mut rng =
                rand::rngs::StdRng::seed_from_u64(self.config.seed.wrapping_add(self.step as u64));
            self.train.shuffle(&mut rng);
        }
    }

    /// Forward pass: returns logits and cross-entropy loss.
    fn forward(&self, av: usize, bv: usize, c_true: usize) -> (Vec<f64>, f64) {
        let hidden = self.config.hidden_dim;
        let prime = self.config.prime;
        let edim = self.config.embed_dim;
        let input_dim = edim * 2;

        // Build input: concat embed(a) and embed(b)
        let mut input = vec![0.0; input_dim];
        input[..edim].copy_from_slice(&self.embeds[av * edim..(av + 1) * edim]);
        input[edim..].copy_from_slice(&self.embeds[bv * edim..(bv + 1) * edim]);

        let w1 = &self.weights[w1_slice(hidden, input_dim)];
        let b1 = &self.weights[b1_slice(hidden, input_dim)];
        let w2 = &self.weights[w2_slice(prime, hidden, input_dim)];
        let b2 = &self.weights[b2_slice(prime, hidden, input_dim)];

        // logits = W2 @ ReLU(W1 @ input + b1) + b2
        let mut h_buf = vec![0.0; hidden];
        for row in 0..hidden {
            let mut sum = b1[row];
            for col in 0..input_dim {
                sum = w1[row * input_dim + col].mul_add(input[col], sum);
            }
            h_buf[row] = sum.max(0.0);
        }
        let mut logits = vec![0.0; prime];
        for row in 0..prime {
            let mut sum = b2[row];
            for col in 0..hidden {
                sum = w2[row * hidden + col].mul_add(h_buf[col], sum);
            }
            logits[row] = sum;
        }

        // Stable cross-entropy
        let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|&lv| (lv - max_l).exp()).sum();
        let loss = -(logits[c_true] - max_l) + exp_sum.ln();

        (logits, loss)
    }

    /// Backward pass for one triple. Accumulates into grad buffers.
    #[allow(clippy::similar_names)]
    fn backward(&self, av: usize, bv: usize, c_true: usize, eg: &mut [f64], wg: &mut [f64]) {
        let hidden = self.config.hidden_dim;
        let prime = self.config.prime;
        let edim = self.config.embed_dim;
        let input_dim = edim * 2;

        let mut input = vec![0.0; input_dim];
        input[..edim].copy_from_slice(&self.embeds[av * edim..(av + 1) * edim]);
        input[edim..].copy_from_slice(&self.embeds[bv * edim..(bv + 1) * edim]);

        // Forward with saved intermediates
        let w1 = &self.weights[w1_slice(hidden, input_dim)];
        let b1 = &self.weights[b1_slice(hidden, input_dim)];
        let w2 = &self.weights[w2_slice(prime, hidden, input_dim)];
        let b2 = &self.weights[b2_slice(prime, hidden, input_dim)];

        let mut pre_h = vec![0.0; hidden];
        for row in 0..hidden {
            let mut sum = b1[row];
            for col in 0..input_dim {
                sum = w1[row * input_dim + col].mul_add(input[col], sum);
            }
            pre_h[row] = sum;
        }
        let h_act: Vec<f64> = pre_h.iter().map(|&val| val.max(0.0)).collect();

        let mut logits = vec![0.0; prime];
        for row in 0..prime {
            let mut sum = b2[row];
            for col in 0..hidden {
                sum = w2[row * hidden + col].mul_add(h_act[col], sum);
            }
            logits[row] = sum;
        }

        // Softmax
        let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&lv| (lv - max_l).exp()).collect();
        let exp_sum: f64 = exps.iter().sum();
        let mut dl = vec![0.0; prime];
        for idx in 0..prime {
            dl[idx] = exps[idx] / exp_sum;
        }
        dl[c_true] -= 1.0;

        // d_W2, d_b2
        let w2_off = w2_slice(prime, hidden, input_dim).start;
        let b2_off = b2_slice(prime, hidden, input_dim).start;
        for row in 0..prime {
            wg[b2_off + row] += dl[row];
            for col in 0..hidden {
                wg[w2_off + row * hidden + col] += dl[row] * h_act[col];
            }
        }

        // d_hidden = W2^T @ d_logits
        let mut dh = vec![0.0; hidden];
        for col in 0..hidden {
            for row in 0..prime {
                dh[col] += w2[row * hidden + col] * dl[row];
            }
        }

        // ReLU backward
        let dph: Vec<f64> = dh
            .iter()
            .zip(pre_h.iter())
            .map(|(&dv, &pv)| if pv > 0.0 { dv } else { 0.0 })
            .collect();

        // d_W1, d_b1
        let w1_off = w1_slice(hidden, input_dim).start;
        let b1_off = b1_slice(hidden, input_dim).start;
        for row in 0..hidden {
            wg[b1_off + row] += dph[row];
            for col in 0..input_dim {
                wg[w1_off + row * input_dim + col] += dph[row] * input[col];
            }
        }

        // d_input = W1^T @ d_pre_h -> embedding gradients
        let mut di = vec![0.0; input_dim];
        for col in 0..input_dim {
            for row in 0..hidden {
                di[col] += w1[row * input_dim + col] * dph[row];
            }
        }

        // Scatter back to embedding gradients
        for dim in 0..edim {
            eg[av * edim + dim] += di[dim];
            eg[bv * edim + dim] += di[edim + dim];
        }
    }

    /// Evaluate loss and accuracy on a set of triples.
    fn evaluate(&self, triples: &[(usize, usize, usize)]) -> (f64, f64) {
        if triples.is_empty() {
            return (0.0, 0.0);
        }
        let mut total_loss = 0.0;
        let mut correct = 0_usize;

        for &(av, bv, c_true) in triples {
            let (logits, loss) = self.forward(av, bv, c_true);
            total_loss += loss;
            let pred = logits
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx);
            if pred == c_true {
                correct += 1;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let count = triples.len() as f64;
        #[allow(clippy::cast_precision_loss)]
        (total_loss / count, correct as f64 / count)
    }

    /// Record train/test metrics into history.
    fn record_metrics(&mut self) {
        let train_copy = self.train.clone();
        let test_copy = self.test.clone();
        let (tl, ta) = self.evaluate(&train_copy);
        let (vl, va) = self.evaluate(&test_copy);
        self.train_loss = tl;
        self.train_acc = ta;
        self.test_loss = vl;
        self.test_acc = va;
        self.train_loss_hist.push(tl);
        self.test_loss_hist.push(vl);
        self.train_acc_hist.push(ta);
        self.test_acc_hist.push(va);
    }
}

/// PCA projection of n points in d dimensions down to 2D.
///
/// Returns (x, y) pairs scaled to fit within [-0.9, 0.9] for disk display.
#[allow(clippy::cast_precision_loss)]
fn pca_project_2d(data: &[f64], n_points: usize, dims: usize) -> Vec<(f64, f64)> {
    if n_points == 0 || dims < 2 {
        return vec![(0.0, 0.0); n_points];
    }

    // Mean-center
    let nf = n_points as f64;
    let mut mean = vec![0.0; dims];
    for pt in 0..n_points {
        for dim in 0..dims {
            mean[dim] += data[pt * dims + dim];
        }
    }
    for val in &mut mean {
        *val /= nf;
    }

    let mut centered = vec![0.0; n_points * dims];
    for pt in 0..n_points {
        for dim in 0..dims {
            centered[pt * dims + dim] = data[pt * dims + dim] - mean[dim];
        }
    }

    // Covariance matrix (dims x dims)
    let mut cov = vec![0.0; dims * dims];
    for pt in 0..n_points {
        for row in 0..dims {
            let cr = centered[pt * dims + row];
            for col in row..dims {
                let val = cr * centered[pt * dims + col];
                cov[row * dims + col] += val;
                if row != col {
                    cov[col * dims + row] += val;
                }
            }
        }
    }
    for val in &mut cov {
        *val /= nf;
    }

    // Power iteration for top eigenvector
    let mut v1 = vec![0.0; dims];
    v1[0] = 1.0;
    for _ in 0..200 {
        let nv = mat_vec_mul(&cov, dims, &v1);
        v1 = nv;
        vec_normalize(&mut v1);
    }

    // Deflate covariance
    let lam1: f64 = {
        let mv = mat_vec_mul(&cov, dims, &v1);
        mv.iter().zip(v1.iter()).map(|(a, b)| a * b).sum()
    };
    for row in 0..dims {
        for col in 0..dims {
            cov[row * dims + col] -= lam1 * v1[row] * v1[col];
        }
    }

    // Second eigenvector
    let mut v2 = vec![0.0; dims];
    v2[dims.min(1)] = 1.0;
    for _ in 0..200 {
        let nv = mat_vec_mul(&cov, dims, &v2);
        v2 = nv;
        vec_normalize(&mut v2);
    }

    // Project each point
    let mut xs = Vec::with_capacity(n_points);
    let mut ys = Vec::with_capacity(n_points);
    for pt in 0..n_points {
        let row = &centered[pt * dims..(pt + 1) * dims];
        let px: f64 = row.iter().zip(v1.iter()).map(|(a, b)| a * b).sum();
        let py: f64 = row.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        xs.push(px);
        ys.push(py);
    }

    // Scale to [-0.9, 0.9]
    let max_abs = xs
        .iter()
        .chain(ys.iter())
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-10);
    let scale = 0.9 / max_abs;

    xs.iter()
        .zip(ys.iter())
        .map(|(&px, &py)| (px * scale, py * scale))
        .collect()
}

/// Matrix-vector multiply for a square matrix.
fn mat_vec_mul(mat: &[f64], dim: usize, vec: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; dim];
    for row in 0..dim {
        for col in 0..dim {
            out[row] += mat[row * dim + col] * vec[col];
        }
    }
    out
}

/// Normalize a vector in-place.
fn vec_normalize(vec: &mut [f64]) {
    let norm: f64 = vec.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for val in vec.iter_mut() {
            *val /= norm;
        }
    }
}

/// `AdamW` parameter update (in-place).
fn adamw_update(
    params: &mut [f64],
    grads: &[f64],
    first_moment: &mut [f64],
    second_moment: &mut [f64],
    lr: f64,
    wd: f64,
    step: usize,
) {
    let beta1: f64 = 0.9;
    let beta2: f64 = 0.98;
    let eps: f64 = 1e-8;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let time = (step + 1) as i32;
    let bc1 = 1.0 - beta1.powi(time);
    let bc2 = 1.0 - beta2.powi(time);

    for idx in 0..params.len() {
        first_moment[idx] = beta1.mul_add(first_moment[idx], (1.0 - beta1) * grads[idx]);
        second_moment[idx] =
            beta2.mul_add(second_moment[idx], (1.0 - beta2) * grads[idx] * grads[idx]);
        let m_hat = first_moment[idx] / bc1;
        let v_hat = second_moment[idx] / bc2;
        let adam_step = m_hat / (v_hat.sqrt() + eps);
        params[idx] -= lr * wd.mul_add(params[idx], adam_step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_creates() {
        let session = GrokSession::new(GrokConfig::default());
        assert_eq!(session.step, 0);
        assert_eq!(session.train.len() + session.test.len(), 31 * 31);
    }

    #[test]
    fn forward_produces_finite_loss() {
        let session = GrokSession::new(GrokConfig::default());
        let (logits, loss) = session.forward(0, 1, 1);
        assert_eq!(logits.len(), 31);
        assert!(loss.is_finite(), "loss should be finite: {loss}");
    }

    #[test]
    fn single_step_advances() {
        let mut session = GrokSession::new(GrokConfig::default());
        session.single_step();
        assert_eq!(session.step, 1);
        assert!(session.train_loss.is_finite());
    }

    #[test]
    fn multi_step_advances() {
        let mut session = GrokSession::new(GrokConfig {
            steps_per_tick: 5,
            total_steps: 100,
            ..GrokConfig::default()
        });
        session.step();
        assert_eq!(session.step, 5);
    }

    #[test]
    fn embeddings_stay_in_disk() {
        let mut session = GrokSession::new(GrokConfig {
            total_steps: 50,
            steps_per_tick: 50,
            ..GrokConfig::default()
        });
        session.step();
        for idx in 0..session.config.prime {
            let ex = session.embeds[idx * 2];
            let ey = session.embeds[idx * 2 + 1];
            assert!(ex.hypot(ey) < 1.0, "embedding {idx} escaped disk");
        }
    }

    #[test]
    fn train_loss_decreases() {
        let mut session = GrokSession::new(GrokConfig {
            total_steps: 200,
            steps_per_tick: 50,
            ..GrokConfig::default()
        });
        session.step();
        let loss1 = session.train_loss;
        for _ in 0..3 {
            session.step();
        }
        let loss2 = session.train_loss;
        assert!(loss2 < loss1, "loss should decrease: {loss1} -> {loss2}");
    }

    #[test]
    fn stats_and_viz() {
        let session = GrokSession::new(GrokConfig::default());
        let stats = session.stats();
        assert_eq!(stats.step, 0);
        let viz = session.to_viz_data();
        assert_eq!(viz.nodes.len(), 31);
    }

    #[test]
    fn reproducible() {
        let run = || {
            let mut session = GrokSession::new(GrokConfig {
                total_steps: 20,
                steps_per_tick: 20,
                ..GrokConfig::default()
            });
            session.step();
            session.train_loss
        };
        let loss_a = run();
        let loss_b = run();
        assert!(
            (loss_a - loss_b).abs() < 1e-12,
            "same seed should give same loss: {loss_a} vs {loss_b}"
        );
    }

    #[test]
    fn completes_at_total_steps() {
        let mut session = GrokSession::new(GrokConfig {
            total_steps: 10,
            steps_per_tick: 10,
            ..GrokConfig::default()
        });
        session.step();
        assert_eq!(session.status(), TrainingStatus::Completed);
    }
}
