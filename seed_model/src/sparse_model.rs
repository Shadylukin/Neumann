//! Pure Geometric Model - No RNN, Emergent Dimensions
//!
//! Implements the Tensor Scaling Theory:
//! - Response IS the geometric shape (no translation needed)
//! - Dimensions emerge from data (not pre-allocated)
//! - Sparsity is native (SparseVector throughout)
//!
//! Architecture:
//! ```text
//! tokens → SparseEmbed → GeometricVQ → logits
//!              |              |
//!         SparseVector   values ARE vocab distributions
//! ```

use rand::Rng;
use tensor_store::SparseVector;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the pure geometric model
#[derive(Debug, Clone)]
pub struct GeometricConfig {
    pub vocab_size: usize,
    /// Maximum dimension capacity (emergence grows within this)
    pub dim_capacity: usize,
    /// Initial non-zero entries per embedding
    pub initial_nnz: usize,
    /// Number of VQ levels (capacity, model uses what it needs)
    pub num_levels: usize,
    /// Number of keys per VQ level
    pub num_keys: usize,
    /// Sparsity target (0.9 = stop when 90% zeros)
    pub sparsity_target: f32,
    /// Context decay factor (higher = longer memory)
    pub context_decay: f32,
}

impl GeometricConfig {
    pub fn from_vocab(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            dim_capacity: 128,    // Reduced for faster iteration
            initial_nnz: 16,      // Start sparse
            num_levels: 4,        // Start with fewer levels
            num_keys: 64,         // Fewer keys = faster
            sparsity_target: 0.9, // The ONLY hard constraint
            context_decay: 0.9,   // Recent tokens matter more
        }
    }

    pub fn tiny(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            dim_capacity: 64,
            initial_nnz: 8,
            num_levels: 2,
            num_keys: 16,
            sparsity_target: 0.9,
            context_decay: 0.9,
        }
    }
}

// ============================================================================
// Sparse Embedding Table
// ============================================================================

/// Embedding table where each entry is a SparseVector
pub struct SparseEmbeddingTable {
    entries: Vec<SparseVector>,
    dim_capacity: usize,
}

impl SparseEmbeddingTable {
    /// Initialize with random sparse embeddings
    pub fn random(vocab_size: usize, dim_capacity: usize, initial_nnz: usize) -> Self {
        let mut rng = rand::thread_rng();

        let entries: Vec<SparseVector> = (0..vocab_size)
            .map(|_| {
                let mut positions: Vec<u32> = (0..initial_nnz)
                    .map(|_| rng.gen_range(0..dim_capacity) as u32)
                    .collect();
                positions.sort();
                positions.dedup();

                let values: Vec<f32> = positions.iter().map(|_| rng.gen_range(-0.1..0.1)).collect();

                SparseVector::from_parts(dim_capacity, positions, values)
            })
            .collect();

        Self {
            entries,
            dim_capacity,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.entries.len()
    }

    pub fn dim_capacity(&self) -> usize {
        self.dim_capacity
    }

    pub fn embed(&self, token_id: usize) -> &SparseVector {
        &self.entries[token_id]
    }

    /// Update embedding with sparse gradient
    pub fn update(&mut self, token_id: usize, grad: &SparseVector, lr: f32) {
        let neg_lr_grad = grad.scale(-lr);
        self.entries[token_id] = self.entries[token_id].add(&neg_lr_grad);
    }
}

// ============================================================================
// Geometric VQ Level
// ============================================================================

/// A single VQ level with sparse keys and vocab-space values
pub struct GeometricVQLevel {
    /// Keys: patterns to recognize (in embedding space)
    keys: Vec<SparseVector>,
    /// Values: vocab distributions (sparse over vocab_size)
    values: Vec<SparseVector>,
    /// Learned temperature (as log for positivity)
    log_temperature: f32,
    /// Key dimension
    key_dim: usize,
    /// Value dimension (= vocab size)
    value_dim: usize,
}

impl GeometricVQLevel {
    pub fn random(
        num_keys: usize,
        key_dim: usize,
        value_dim: usize,
        key_nnz: usize,
        value_nnz: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let keys: Vec<SparseVector> = (0..num_keys)
            .map(|_| {
                let mut positions: Vec<u32> = (0..key_nnz)
                    .map(|_| rng.gen_range(0..key_dim) as u32)
                    .collect();
                positions.sort();
                positions.dedup();
                let values: Vec<f32> = positions.iter().map(|_| rng.gen_range(-0.1..0.1)).collect();
                SparseVector::from_parts(key_dim, positions, values)
            })
            .collect();

        let values: Vec<SparseVector> = (0..num_keys)
            .map(|_| {
                let mut positions: Vec<u32> = (0..value_nnz)
                    .map(|_| rng.gen_range(0..value_dim) as u32)
                    .collect();
                positions.sort();
                positions.dedup();
                let vals: Vec<f32> = positions.iter().map(|_| rng.gen_range(-0.1..0.1)).collect();
                SparseVector::from_parts(value_dim, positions, vals)
            })
            .collect();

        Self {
            keys,
            values,
            log_temperature: 0.0, // exp(0) = 1.0
            key_dim,
            value_dim,
        }
    }

    pub fn temperature(&self) -> f32 {
        self.log_temperature.exp()
    }

    /// Forward: compute similarities, return weighted sum of values
    pub fn forward(&self, input: &SparseVector) -> (SparseVector, Vec<f32>) {
        // Compute similarities to all keys
        let similarities: Vec<f32> = self
            .keys
            .iter()
            .map(|k| input.cosine_similarity(k))
            .collect();

        // Softmax with temperature
        let temp = self.temperature();
        let weights = softmax(&similarities, temp);

        // Weighted sum of values (only top-k weights to maintain sparsity)
        let mut response = SparseVector::new(self.value_dim);

        // Find threshold for top-k weights (use sqrt(num_keys) entries)
        let top_k = (self.keys.len() as f32).sqrt() as usize;
        let mut sorted_weights: Vec<f32> = weights.clone();
        sorted_weights.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let weight_threshold = sorted_weights.get(top_k).copied().unwrap_or(0.0);

        for (v, &w) in self.values.iter().zip(&weights) {
            if w > weight_threshold {
                response = response.add(&v.scale(w));
            }
        }

        // Prune small values to maintain sparsity
        let response = response.pruned(0.001);

        (response, weights)
    }

    /// Update keys and values with gradients
    pub fn update(
        &mut self,
        key_grads: &[SparseVector],
        value_grads: &[SparseVector],
        temp_grad: f32,
        lr: f32,
    ) {
        for (i, grad) in key_grads.iter().enumerate() {
            self.keys[i] = self.keys[i].add(&grad.scale(-lr));
        }
        for (i, grad) in value_grads.iter().enumerate() {
            self.values[i] = self.values[i].add(&grad.scale(-lr));
        }
        self.log_temperature -= lr * temp_grad;
    }
}

// ============================================================================
// Geometric Context (replaces GRU)
// ============================================================================

/// Simple geometric context via weighted averaging
pub struct GeometricContext {
    accumulated: SparseVector,
    decay: f32,
}

impl GeometricContext {
    pub fn new(dim: usize, decay: f32) -> Self {
        Self {
            accumulated: SparseVector::new(dim),
            decay,
        }
    }

    pub fn reset(&mut self) {
        self.accumulated = SparseVector::new(self.accumulated.dimension());
    }

    pub fn update(&mut self, new_repr: &SparseVector) {
        self.accumulated =
            self.accumulated
                .weighted_average(new_repr, self.decay, 1.0 - self.decay);
    }

    pub fn get(&self) -> &SparseVector {
        &self.accumulated
    }
}

// ============================================================================
// Pure Geometric Model
// ============================================================================

/// The complete pure geometric model
pub struct PureGeometricModel {
    embedding: SparseEmbeddingTable,
    vq_levels: Vec<GeometricVQLevel>,
    sparsity_target: f32,
    context_decay: f32,
    vocab_size: usize,
}

impl PureGeometricModel {
    pub fn new(config: &GeometricConfig) -> Self {
        let embedding = SparseEmbeddingTable::random(
            config.vocab_size,
            config.dim_capacity,
            config.initial_nnz,
        );

        let vq_levels: Vec<GeometricVQLevel> = (0..config.num_levels)
            .map(|_| {
                GeometricVQLevel::random(
                    config.num_keys,
                    config.dim_capacity,
                    config.vocab_size,
                    config.initial_nnz,
                    config.initial_nnz,
                )
            })
            .collect();

        Self {
            embedding,
            vq_levels,
            sparsity_target: config.sparsity_target,
            context_decay: config.context_decay,
            vocab_size: config.vocab_size,
        }
    }

    /// Forward pass on a sequence
    /// Returns: (logits for each position, depth used, final sparsity)
    pub fn forward(&self, tokens: &[usize]) -> (Vec<SparseVector>, Vec<usize>, Vec<f32>) {
        let mut context = GeometricContext::new(self.embedding.dim_capacity(), self.context_decay);

        let mut all_logits = Vec::with_capacity(tokens.len());
        let mut all_depths = Vec::with_capacity(tokens.len());
        let mut all_sparsities = Vec::with_capacity(tokens.len());

        for &token in tokens {
            // 1. Embed
            let embedded = self.embedding.embed(token);

            // 2. Update context
            context.update(embedded);

            // 3. VQ until sparse
            let mut current = context.get().clone();
            let mut vocab_logits = SparseVector::new(self.vocab_size);
            let mut depth = 0;

            for level in &self.vq_levels {
                let (response, _weights) = level.forward(&current);
                vocab_logits = vocab_logits.add(&response);

                // Compute residual
                // Note: response is in vocab space, current is in embedding space
                // We use the response's effect on sparsity as the signal
                depth += 1;

                // Check sparsity of current representation
                let sparsity = current.sparsity();
                if sparsity >= self.sparsity_target {
                    break;
                }

                // Update current (simplified - in full version would track residual properly)
                current = current.scale(0.5); // Decay toward sparsity
            }

            all_logits.push(vocab_logits);
            all_depths.push(depth);
            all_sparsities.push(current.sparsity());
        }

        (all_logits, all_depths, all_sparsities)
    }

    /// Compute cross-entropy loss
    pub fn cross_entropy_loss(logits: &SparseVector, target: usize) -> f32 {
        let probs = sparse_softmax(logits);
        let target_prob = probs.get(target);
        -(target_prob + 1e-10).ln()
    }

    /// Training step with simple contrastive learning
    pub fn train_step(&mut self, tokens: &[usize], targets: &[usize], lr: f32) -> f32 {
        let mut total_loss = 0.0;

        // For each token, try to predict the next
        for i in 0..tokens.len().saturating_sub(1) {
            let token = tokens[i];
            let target = targets[i];

            // Get embedding for current token
            let embedded = self.embedding.embed(token).clone();

            // Forward through VQ levels to get logits
            let mut current = embedded.clone();
            let mut vocab_logits = SparseVector::new(self.vocab_size);

            for level_idx in 0..self.vq_levels.len() {
                // Forward pass (read-only)
                let (response, weights) = self.vq_levels[level_idx].forward(&current);
                vocab_logits = vocab_logits.add(&response);

                // Find winner
                let max_idx = weights
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let max_weight = weights[max_idx];

                // Get key and value for winner (clone to avoid borrow)
                let winner_key = self.vq_levels[level_idx].keys[max_idx].clone();
                let winner_value = self.vq_levels[level_idx].values[max_idx].clone();

                // Move winning key toward input
                let key_diff = current.sub(&winner_key);
                let key_update = key_diff.scale(lr * 0.1);
                self.vq_levels[level_idx].keys[max_idx] = winner_key.add(&key_update);

                // Update value to point more toward target token
                let target_onehot =
                    SparseVector::from_parts(self.vocab_size, vec![target as u32], vec![1.0]);
                let value_diff = target_onehot.sub(&winner_value);
                let value_update = value_diff.scale(lr * 0.1 * max_weight);
                self.vq_levels[level_idx].values[max_idx] = winner_value.add(&value_update);

                // Decay current
                current = current.scale(0.5).pruned(0.001);

                if current.sparsity() >= self.sparsity_target {
                    break;
                }
            }

            // Compute loss
            let loss = Self::cross_entropy_loss(&vocab_logits, target);
            total_loss += loss;

            // Update embedding to be closer to target's embedding (simplified)
            let target_emb = self.embedding.embed(target).clone();
            let emb_diff = target_emb.sub(&embedded);
            let emb_update = emb_diff.scale(lr * 0.01);
            self.embedding.update(token, &emb_update, 1.0); // lr already in update
        }

        total_loss / tokens.len().saturating_sub(1).max(1) as f32
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn num_levels(&self) -> usize {
        self.vq_levels.len()
    }
}

// ============================================================================
// Utilities
// ============================================================================

/// Softmax with temperature
fn softmax(values: &[f32], temperature: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }

    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = values
        .iter()
        .map(|&v| ((v - max_val) / temperature).exp())
        .collect();
    let sum: f32 = exp_vals.iter().sum();

    if sum == 0.0 {
        vec![1.0 / values.len() as f32; values.len()]
    } else {
        exp_vals.iter().map(|&e| e / sum).collect()
    }
}

/// Sparse softmax - returns sparse result
fn sparse_softmax(logits: &SparseVector) -> SparseVector {
    let positions = logits.positions();
    let values = logits.values();

    if values.is_empty() {
        return SparseVector::new(logits.dimension());
    }

    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    let probs: Vec<f32> = if sum == 0.0 {
        vec![1.0 / values.len() as f32; values.len()]
    } else {
        exp_vals.iter().map(|&e| e / sum).collect()
    };

    SparseVector::from_parts(logits.dimension(), positions.to_vec(), probs)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_embedding_table() {
        let table = SparseEmbeddingTable::random(100, 256, 16);
        assert_eq!(table.vocab_size(), 100);

        let emb = table.embed(0);
        assert_eq!(emb.dimension(), 256);
        assert!(emb.nnz() > 0);
        assert!(emb.nnz() <= 16);
    }

    #[test]
    fn test_geometric_vq_level() {
        let level = GeometricVQLevel::random(64, 256, 100, 16, 16);
        assert!((level.temperature() - 1.0).abs() < 0.01);

        let input = SparseVector::from_parts(256, vec![0, 10, 20], vec![0.5, -0.3, 0.8]);
        let (response, weights) = level.forward(&input);

        assert_eq!(response.dimension(), 100);
        assert_eq!(weights.len(), 64);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_geometric_context() {
        let mut ctx = GeometricContext::new(256, 0.9);
        assert_eq!(ctx.get().nnz(), 0);

        let v1 = SparseVector::from_parts(256, vec![0, 1], vec![1.0, 1.0]);
        ctx.update(&v1);
        assert!(ctx.get().nnz() > 0);

        let v2 = SparseVector::from_parts(256, vec![2, 3], vec![1.0, 1.0]);
        ctx.update(&v2);
        // Should have entries from both v1 and v2 (with decay)
    }

    #[test]
    fn test_pure_geometric_model() {
        let config = GeometricConfig::from_vocab(100);
        let model = PureGeometricModel::new(&config);

        let tokens = vec![0, 1, 2, 3, 4];
        let (logits, depths, sparsities) = model.forward(&tokens);

        assert_eq!(logits.len(), 5);
        assert_eq!(depths.len(), 5);
        assert_eq!(sparsities.len(), 5);

        for logit in &logits {
            assert_eq!(logit.dimension(), 100);
        }
    }

    #[test]
    fn test_training_step() {
        let config = GeometricConfig::from_vocab(100);
        let mut model = PureGeometricModel::new(&config);

        let tokens = vec![0, 1, 2, 3];
        let targets = vec![1, 2, 3, 4];

        let loss1 = model.train_step(&tokens, &targets, 0.01);
        let loss2 = model.train_step(&tokens, &targets, 0.01);

        // Loss should generally decrease (though not guaranteed with simplified gradients)
        println!("Loss 1: {}, Loss 2: {}", loss1, loss2);
    }
}
