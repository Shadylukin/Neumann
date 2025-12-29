//! Semantic conflict detection and auto-merge for transactions.
//!
//! Uses tensor operations to detect and resolve conflicts:
//! - Delta embeddings capture semantic meaning of changes
//! - Cosine similarity classifies conflict severity
//! - Vector operations enable automatic merging
//!
//! # Conflict Classification
//!
//! | cos(d1, d2) | Key Overlap | Classification | Action |
//! |-------------|-------------|----------------|--------|
//! | < 0.1       | Any         | Orthogonal     | Auto-merge (vector addition) |
//! | 0.1-0.7     | None        | Low conflict   | Merge with validation |
//! | 0.1-0.7     | Some        | Ambiguous      | Reject |
//! | >= 0.7      | Any         | Conflicting    | Reject |
//! | = 1.0       | All         | Identical      | Deduplicate |
//! | <= -0.95    | All         | Opposite       | Cancel (no-op) |

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::transaction::TransactionDelta;

/// Configuration for the consensus manager.
#[derive(Debug, Clone)]
pub struct ConsensusConfig {
    /// Threshold below which transactions are orthogonal.
    pub orthogonal_threshold: f32,
    /// Threshold above which transactions conflict.
    pub conflict_threshold: f32,
    /// Threshold for considering transactions identical.
    pub identical_threshold: f32,
    /// Threshold for considering transactions opposite.
    pub opposite_threshold: f32,
    /// Whether to allow merging with key overlap.
    pub allow_key_overlap_merge: bool,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            orthogonal_threshold: 0.1,
            conflict_threshold: 0.7,
            identical_threshold: 0.99,
            opposite_threshold: -0.95,
            allow_key_overlap_merge: false,
        }
    }
}

/// Classification of conflict between two transaction deltas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictClass {
    /// Transactions are orthogonal (cos < 0.1) - can auto-merge.
    Orthogonal,
    /// Low conflict (0.1-0.7) with no key overlap - can merge with validation.
    LowConflict,
    /// Ambiguous (0.1-0.7) with key overlap - should reject.
    Ambiguous,
    /// High conflict (cos >= 0.7) - must reject.
    Conflicting,
    /// Transactions are identical (cos = 1.0, same keys) - deduplicate.
    Identical,
    /// Transactions are opposite (cos <= -0.95, same keys) - cancel out.
    Opposite,
}

impl ConflictClass {
    /// Whether this classification allows merging.
    pub fn can_merge(&self) -> bool {
        matches!(
            self,
            Self::Orthogonal | Self::LowConflict | Self::Identical | Self::Opposite
        )
    }

    /// Whether this classification should be rejected.
    pub fn should_reject(&self) -> bool {
        matches!(self, Self::Ambiguous | Self::Conflicting)
    }
}

/// Result of conflict detection between two deltas.
#[derive(Debug, Clone)]
pub struct ConflictResult {
    /// Classification of the conflict.
    pub class: ConflictClass,
    /// Cosine similarity between delta embeddings.
    pub similarity: f32,
    /// Keys that overlap between the two deltas.
    pub overlapping_keys: HashSet<String>,
    /// Whether the deltas can be merged.
    pub can_merge: bool,
    /// Recommended action.
    pub action: MergeAction,
}

/// Recommended action for handling conflicting transactions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeAction {
    /// Merge via vector addition.
    VectorAdd,
    /// Merge via weighted average.
    WeightedAverage { weight1: u32, weight2: u32 },
    /// Deduplicate (keep only one).
    Deduplicate,
    /// Cancel out (result is no-op).
    Cancel,
    /// Reject - cannot merge.
    Reject,
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Whether the merge was successful.
    pub success: bool,
    /// The merged delta (if successful).
    pub merged_delta: Option<DeltaVector>,
    /// Parent transaction IDs that were merged.
    pub parent_ids: Vec<u64>,
    /// Action that was taken.
    pub action: MergeAction,
    /// Error message (if failed).
    pub error: Option<String>,
}

/// A delta vector representing a change in state space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaVector {
    /// The embedding delta (after - before).
    pub vector: Vec<f32>,
    /// Magnitude of the delta.
    pub magnitude: f32,
    /// Keys affected by this delta.
    pub affected_keys: HashSet<String>,
    /// Source transaction ID.
    pub tx_id: u64,
}

impl DeltaVector {
    /// Create a new delta vector.
    pub fn new(vector: Vec<f32>, affected_keys: HashSet<String>, tx_id: u64) -> Self {
        let magnitude = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self {
            vector,
            magnitude,
            affected_keys,
            tx_id,
        }
    }

    /// Create from before and after state embeddings.
    pub fn from_states(
        before: &[f32],
        after: &[f32],
        affected_keys: HashSet<String>,
        tx_id: u64,
    ) -> Self {
        let vector: Vec<f32> = after
            .iter()
            .zip(before.iter())
            .map(|(a, b)| a - b)
            .collect();
        Self::new(vector, affected_keys, tx_id)
    }

    /// Create a zero delta.
    pub fn zero(dimension: usize) -> Self {
        Self {
            vector: vec![0.0; dimension],
            magnitude: 0.0,
            affected_keys: HashSet::new(),
            tx_id: 0,
        }
    }

    /// Get the dimension of the delta.
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }

    /// Compute cosine similarity with another delta.
    pub fn cosine_similarity(&self, other: &DeltaVector) -> f32 {
        if self.magnitude == 0.0 || other.magnitude == 0.0 {
            return 0.0;
        }

        let dot: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();

        dot / (self.magnitude * other.magnitude)
    }

    /// Check if this delta overlaps with another in key space.
    pub fn overlaps_with(&self, other: &DeltaVector) -> bool {
        self.affected_keys
            .intersection(&other.affected_keys)
            .next()
            .is_some()
    }

    /// Get overlapping keys with another delta.
    pub fn overlapping_keys(&self, other: &DeltaVector) -> HashSet<String> {
        self.affected_keys
            .intersection(&other.affected_keys)
            .cloned()
            .collect()
    }

    /// Add another delta vector (for orthogonal merge).
    pub fn add(&self, other: &DeltaVector) -> DeltaVector {
        let vector: Vec<f32> = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a + b)
            .collect();
        let keys: HashSet<String> = self
            .affected_keys
            .union(&other.affected_keys)
            .cloned()
            .collect();
        DeltaVector::new(vector, keys, 0) // New tx_id will be assigned
    }

    /// Compute weighted average with another delta.
    pub fn weighted_average(&self, other: &DeltaVector, w1: f32, w2: f32) -> DeltaVector {
        let total = w1 + w2;
        if total == 0.0 {
            return DeltaVector::zero(self.dimension());
        }

        let vector: Vec<f32> = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| (w1 * a + w2 * b) / total)
            .collect();
        let keys: HashSet<String> = self
            .affected_keys
            .union(&other.affected_keys)
            .cloned()
            .collect();
        DeltaVector::new(vector, keys, 0)
    }

    /// Project out the conflicting component (along a direction).
    pub fn project_non_conflicting(&self, conflict_direction: &[f32]) -> DeltaVector {
        let dir_mag: f32 = conflict_direction.iter().map(|x| x * x).sum::<f32>().sqrt();
        if dir_mag == 0.0 {
            return self.clone();
        }

        // Compute projection of self onto conflict direction
        let dot: f32 = self
            .vector
            .iter()
            .zip(conflict_direction.iter())
            .map(|(a, b)| a * b)
            .sum();
        let proj_scalar = dot / (dir_mag * dir_mag);

        // Subtract projection to get orthogonal component
        let vector: Vec<f32> = self
            .vector
            .iter()
            .zip(conflict_direction.iter())
            .map(|(a, b)| a - proj_scalar * b)
            .collect();

        DeltaVector::new(vector, self.affected_keys.clone(), self.tx_id)
    }

    /// Scale the delta by a factor.
    pub fn scale(&self, factor: f32) -> DeltaVector {
        let vector: Vec<f32> = self.vector.iter().map(|x| x * factor).collect();
        DeltaVector::new(vector, self.affected_keys.clone(), self.tx_id)
    }
}

/// Manages semantic conflict detection and transaction merging.
pub struct ConsensusManager {
    config: ConsensusConfig,
}

impl ConsensusManager {
    /// Create a new consensus manager.
    pub fn new(config: ConsensusConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(ConsensusConfig::default())
    }

    /// Detect conflict between two transaction deltas.
    pub fn detect_conflict(&self, d1: &DeltaVector, d2: &DeltaVector) -> ConflictResult {
        let similarity = d1.cosine_similarity(d2);
        let overlapping_keys = d1.overlapping_keys(d2);
        let has_overlap = !overlapping_keys.is_empty();
        let all_keys_overlap = overlapping_keys.len() == d1.affected_keys.len()
            && overlapping_keys.len() == d2.affected_keys.len();

        // Classify based on similarity and key overlap
        let (class, action) = if similarity >= self.config.identical_threshold && all_keys_overlap {
            (ConflictClass::Identical, MergeAction::Deduplicate)
        } else if similarity <= self.config.opposite_threshold && all_keys_overlap {
            (ConflictClass::Opposite, MergeAction::Cancel)
        } else if similarity.abs() < self.config.orthogonal_threshold {
            (ConflictClass::Orthogonal, MergeAction::VectorAdd)
        } else if similarity >= self.config.conflict_threshold {
            (ConflictClass::Conflicting, MergeAction::Reject)
        } else if has_overlap && !self.config.allow_key_overlap_merge {
            (ConflictClass::Ambiguous, MergeAction::Reject)
        } else {
            (
                ConflictClass::LowConflict,
                MergeAction::WeightedAverage {
                    weight1: 50,
                    weight2: 50,
                },
            )
        };

        ConflictResult {
            class,
            similarity,
            overlapping_keys,
            can_merge: class.can_merge(),
            action,
        }
    }

    /// Attempt to merge two transaction deltas.
    pub fn merge(&self, d1: &DeltaVector, d2: &DeltaVector) -> MergeResult {
        let conflict = self.detect_conflict(d1, d2);

        if !conflict.can_merge {
            return MergeResult {
                success: false,
                merged_delta: None,
                parent_ids: vec![d1.tx_id, d2.tx_id],
                action: conflict.action,
                error: Some(format!(
                    "cannot merge: {:?} (similarity: {:.3})",
                    conflict.class, conflict.similarity
                )),
            };
        }

        let merged = match conflict.action {
            MergeAction::VectorAdd => d1.add(d2),
            MergeAction::WeightedAverage { weight1, weight2 } => {
                d1.weighted_average(d2, weight1 as f32, weight2 as f32)
            },
            MergeAction::Deduplicate => d1.clone(),
            MergeAction::Cancel => DeltaVector::zero(d1.dimension()),
            MergeAction::Reject => {
                return MergeResult {
                    success: false,
                    merged_delta: None,
                    parent_ids: vec![d1.tx_id, d2.tx_id],
                    action: MergeAction::Reject,
                    error: Some("merge rejected".to_string()),
                };
            },
        };

        MergeResult {
            success: true,
            merged_delta: Some(merged),
            parent_ids: vec![d1.tx_id, d2.tx_id],
            action: conflict.action,
            error: None,
        }
    }

    /// Merge multiple deltas in order.
    pub fn merge_all(&self, deltas: &[DeltaVector]) -> MergeResult {
        if deltas.is_empty() {
            return MergeResult {
                success: true,
                merged_delta: None,
                parent_ids: vec![],
                action: MergeAction::Cancel,
                error: None,
            };
        }

        if deltas.len() == 1 {
            return MergeResult {
                success: true,
                merged_delta: Some(deltas[0].clone()),
                parent_ids: vec![deltas[0].tx_id],
                action: MergeAction::Deduplicate,
                error: None,
            };
        }

        let mut accumulated = deltas[0].clone();
        let mut parent_ids = vec![deltas[0].tx_id];

        for delta in &deltas[1..] {
            let result = self.merge(&accumulated, delta);
            if !result.success {
                return MergeResult {
                    success: false,
                    merged_delta: None,
                    parent_ids,
                    action: result.action,
                    error: result.error,
                };
            }
            accumulated = result.merged_delta.unwrap();
            parent_ids.push(delta.tx_id);
        }

        MergeResult {
            success: true,
            merged_delta: Some(accumulated),
            parent_ids,
            action: MergeAction::VectorAdd,
            error: None,
        }
    }

    /// Find the best merge order for a set of deltas.
    pub fn find_merge_order(&self, deltas: &[DeltaVector]) -> Vec<usize> {
        if deltas.len() <= 2 {
            return (0..deltas.len()).collect();
        }

        // Greedy: start with most orthogonal pairs first
        let mut order = Vec::with_capacity(deltas.len());
        let mut remaining: Vec<usize> = (0..deltas.len()).collect();

        // Start with first delta
        order.push(remaining.remove(0));

        while !remaining.is_empty() {
            // Find most orthogonal delta to current accumulated
            let current_idx = *order.last().unwrap();
            let best = remaining
                .iter()
                .enumerate()
                .map(|(pos, &idx)| {
                    let sim = deltas[current_idx].cosine_similarity(&deltas[idx]).abs();
                    (pos, idx, sim)
                })
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            order.push(remaining.remove(best.0));
        }

        order
    }

    /// Convert a TransactionDelta to a DeltaVector.
    pub fn delta_to_vector(&self, delta: &TransactionDelta) -> DeltaVector {
        DeltaVector::new(
            delta.delta_embedding.clone(),
            delta.affected_keys.clone(),
            delta.tx_id,
        )
    }
}

impl Default for ConsensusManager {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_vector_creation() {
        let keys: HashSet<String> = ["key1", "key2"].iter().map(|s| s.to_string()).collect();
        let delta = DeltaVector::new(vec![1.0, 0.0, 0.0], keys.clone(), 1);

        assert_eq!(delta.dimension(), 3);
        assert!((delta.magnitude - 1.0).abs() < 0.001);
        assert_eq!(delta.affected_keys, keys);
    }

    #[test]
    fn test_cosine_similarity() {
        let d1 = DeltaVector::new(vec![1.0, 0.0, 0.0], HashSet::new(), 1);
        let d2 = DeltaVector::new(vec![1.0, 0.0, 0.0], HashSet::new(), 2);
        let d3 = DeltaVector::new(vec![0.0, 1.0, 0.0], HashSet::new(), 3);
        let d4 = DeltaVector::new(vec![-1.0, 0.0, 0.0], HashSet::new(), 4);

        // Same direction
        assert!((d1.cosine_similarity(&d2) - 1.0).abs() < 0.001);

        // Orthogonal
        assert!(d1.cosine_similarity(&d3).abs() < 0.001);

        // Opposite
        assert!((d1.cosine_similarity(&d4) - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_vector_add() {
        let keys1: HashSet<String> = ["a"].iter().map(|s| s.to_string()).collect();
        let keys2: HashSet<String> = ["b"].iter().map(|s| s.to_string()).collect();

        let d1 = DeltaVector::new(vec![1.0, 0.0], keys1, 1);
        let d2 = DeltaVector::new(vec![0.0, 1.0], keys2, 2);

        let sum = d1.add(&d2);
        assert_eq!(sum.vector, vec![1.0, 1.0]);
        assert!(sum.affected_keys.contains("a"));
        assert!(sum.affected_keys.contains("b"));
    }

    #[test]
    fn test_weighted_average() {
        let d1 = DeltaVector::new(vec![2.0, 0.0], HashSet::new(), 1);
        let d2 = DeltaVector::new(vec![0.0, 2.0], HashSet::new(), 2);

        let avg = d1.weighted_average(&d2, 0.5, 0.5);
        assert_eq!(avg.vector, vec![1.0, 1.0]);
    }

    #[test]
    fn test_project_non_conflicting() {
        let d = DeltaVector::new(vec![1.0, 1.0], HashSet::new(), 1);
        let conflict_dir = vec![1.0, 0.0];

        let projected = d.project_non_conflicting(&conflict_dir);
        // Should remove the x component, leaving only y
        assert!(projected.vector[0].abs() < 0.001);
        assert!((projected.vector[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_conflict_detection_orthogonal() {
        let manager = ConsensusManager::default_config();

        let d1 = DeltaVector::new(vec![1.0, 0.0, 0.0], HashSet::new(), 1);
        let d2 = DeltaVector::new(vec![0.0, 1.0, 0.0], HashSet::new(), 2);

        let result = manager.detect_conflict(&d1, &d2);
        assert_eq!(result.class, ConflictClass::Orthogonal);
        assert!(result.can_merge);
    }

    #[test]
    fn test_conflict_detection_conflicting() {
        let manager = ConsensusManager::default_config();

        // Use different keys so it's not classified as Identical
        let keys1: HashSet<String> = ["key1"].iter().map(|s| s.to_string()).collect();
        let keys2: HashSet<String> = ["key2"].iter().map(|s| s.to_string()).collect();
        // Similar direction (high cosine) but different keys
        let d1 = DeltaVector::new(vec![1.0, 0.0, 0.0], keys1, 1);
        let d2 = DeltaVector::new(vec![0.9, 0.1, 0.0], keys2, 2);

        let result = manager.detect_conflict(&d1, &d2);
        assert_eq!(result.class, ConflictClass::Conflicting);
        assert!(!result.can_merge);
    }

    #[test]
    fn test_conflict_detection_identical() {
        let manager = ConsensusManager::default_config();

        let keys: HashSet<String> = ["key1"].iter().map(|s| s.to_string()).collect();
        let d1 = DeltaVector::new(vec![1.0, 0.0, 0.0], keys.clone(), 1);
        let d2 = DeltaVector::new(vec![1.0, 0.0, 0.0], keys, 2);

        let result = manager.detect_conflict(&d1, &d2);
        assert_eq!(result.class, ConflictClass::Identical);
        assert!(result.can_merge);
    }

    #[test]
    fn test_conflict_detection_opposite() {
        let manager = ConsensusManager::default_config();

        // Same keys, exactly opposite directions -> Opposite classification
        let keys: HashSet<String> = ["key1"].iter().map(|s| s.to_string()).collect();
        let d1 = DeltaVector::new(vec![1.0, 0.0, 0.0], keys.clone(), 1);
        let d2 = DeltaVector::new(vec![-1.0, 0.0, 0.0], keys.clone(), 2);

        let result = manager.detect_conflict(&d1, &d2);
        // cos = -1.0, same keys -> Opposite
        assert_eq!(result.class, ConflictClass::Opposite);
        assert!(result.can_merge);
        assert_eq!(result.action, MergeAction::Cancel);
    }

    #[test]
    fn test_merge_orthogonal() {
        let manager = ConsensusManager::default_config();

        let d1 = DeltaVector::new(vec![1.0, 0.0, 0.0], HashSet::new(), 1);
        let d2 = DeltaVector::new(vec![0.0, 1.0, 0.0], HashSet::new(), 2);

        let result = manager.merge(&d1, &d2);
        assert!(result.success);
        let merged = result.merged_delta.unwrap();
        assert_eq!(merged.vector, vec![1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_merge_opposite() {
        let manager = ConsensusManager::default_config();

        let keys: HashSet<String> = ["key1"].iter().map(|s| s.to_string()).collect();
        let d1 = DeltaVector::new(vec![1.0, 0.0, 0.0], keys.clone(), 1);
        let d2 = DeltaVector::new(vec![-1.0, 0.0, 0.0], keys, 2);

        let result = manager.merge(&d1, &d2);
        assert!(result.success);
        assert_eq!(result.action, MergeAction::Cancel);
        let merged = result.merged_delta.unwrap();
        assert!(merged.magnitude < 0.001); // Should be zero
    }

    #[test]
    fn test_merge_conflicting_rejected() {
        let manager = ConsensusManager::default_config();

        // Different keys to avoid Identical classification, but similar direction
        let keys1: HashSet<String> = ["key1"].iter().map(|s| s.to_string()).collect();
        let keys2: HashSet<String> = ["key2"].iter().map(|s| s.to_string()).collect();
        let d1 = DeltaVector::new(vec![1.0, 0.0, 0.0], keys1, 1);
        let d2 = DeltaVector::new(vec![0.9, 0.1, 0.0], keys2, 2);

        let result = manager.merge(&d1, &d2);
        assert!(!result.success);
        assert_eq!(result.action, MergeAction::Reject);
    }

    #[test]
    fn test_merge_all() {
        let manager = ConsensusManager::default_config();

        let deltas = vec![
            DeltaVector::new(vec![1.0, 0.0, 0.0], HashSet::new(), 1),
            DeltaVector::new(vec![0.0, 1.0, 0.0], HashSet::new(), 2),
            DeltaVector::new(vec![0.0, 0.0, 1.0], HashSet::new(), 3),
        ];

        let result = manager.merge_all(&deltas);
        assert!(result.success);
        let merged = result.merged_delta.unwrap();
        assert_eq!(merged.vector, vec![1.0, 1.0, 1.0]);
        assert_eq!(result.parent_ids.len(), 3);
    }

    #[test]
    fn test_find_merge_order() {
        let manager = ConsensusManager::default_config();

        let deltas = vec![
            DeltaVector::new(vec![1.0, 0.0, 0.0], HashSet::new(), 1),
            DeltaVector::new(vec![0.9, 0.1, 0.0], HashSet::new(), 2), // Similar to first
            DeltaVector::new(vec![0.0, 1.0, 0.0], HashSet::new(), 3), // Orthogonal to first
        ];

        let order = manager.find_merge_order(&deltas);
        // Should prefer orthogonal (index 2) over similar (index 1)
        assert_eq!(order[0], 0);
        assert_eq!(order[1], 2); // Orthogonal should come before similar
    }

    #[test]
    fn test_ambiguous_with_key_overlap() {
        let manager = ConsensusManager::default_config();

        let keys: HashSet<String> = ["shared"].iter().map(|s| s.to_string()).collect();
        // Mid-range similarity (0.1 to 0.7) with key overlap
        // cos([1,0.5,0], [0.2,0.9,0]) should be in mid-range
        // dot = 0.2 + 0.45 = 0.65
        // mag1 = sqrt(1+0.25) = 1.118
        // mag2 = sqrt(0.04+0.81) = 0.922
        // sim = 0.65/(1.118*0.922) = 0.63 which is in [0.1, 0.7)
        let d1 = DeltaVector::new(vec![1.0, 0.5, 0.0], keys.clone(), 1);
        let d2 = DeltaVector::new(vec![0.2, 0.9, 0.0], keys, 2);

        let result = manager.detect_conflict(&d1, &d2);
        // With default config, key overlap + mid-similarity = Ambiguous
        assert_eq!(result.class, ConflictClass::Ambiguous);
        assert!(!result.can_merge);
    }
}
