// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Embedding state machine for transaction workspaces.
//!
//! Provides type-safe state transitions for embedding lifecycle:
//! - Initial: Only before-state embedding known
//! - Computed: Delta computed from before and after states
//!
//! This eliminates Option ceremony and ensures correct API usage at compile time.

use serde::{Deserialize, Serialize};
use tensor_store::SparseVector;

/// Embedding state machine for type-safe transitions.
///
/// Instead of `Option<Vec<f32>>` for after/delta embeddings, this enum
/// makes the state explicit:
/// - `Initial`: Transaction started, before-state captured
/// - `Computed`: Transaction ready to commit, delta computed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingState {
    /// Initial state - only before embedding known.
    Initial {
        /// Embedding of state at transaction begin.
        before: SparseVector,
    },

    /// Computed state - delta available for conflict detection.
    Computed {
        /// Embedding of state at transaction begin.
        before: SparseVector,
        /// Embedding of state at commit time.
        after: SparseVector,
        /// Delta embedding (after - before).
        delta: SparseVector,
    },
}

/// Error when attempting an invalid embedding state transition.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EmbeddingError {
    /// Attempted to get delta from Initial state.
    NotComputed,
    /// Attempted to compute delta when already computed.
    AlreadyComputed,
    /// Dimension mismatch between before and after embeddings.
    DimensionMismatch { before: usize, after: usize },
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotComputed => write!(f, "delta not yet computed"),
            Self::AlreadyComputed => write!(f, "delta already computed"),
            Self::DimensionMismatch { before, after } => {
                write!(f, "dimension mismatch: before={before}, after={after}")
            },
        }
    }
}

impl std::error::Error for EmbeddingError {}

impl EmbeddingState {
    #[must_use]
    pub fn new(before: SparseVector) -> Self {
        Self::Initial { before }
    }

    #[must_use]
    pub fn from_dense(before: &[f32]) -> Self {
        Self::Initial {
            before: SparseVector::from_dense(before),
        }
    }

    #[must_use]
    pub fn empty(dimension: usize) -> Self {
        Self::Initial {
            before: SparseVector::new(dimension),
        }
    }

    #[must_use]
    pub fn before(&self) -> &SparseVector {
        match self {
            Self::Initial { before } | Self::Computed { before, .. } => before,
        }
    }

    #[must_use]
    pub fn is_computed(&self) -> bool {
        matches!(self, Self::Computed { .. })
    }

    #[must_use]
    pub fn delta(&self) -> Option<&SparseVector> {
        match self {
            Self::Initial { .. } => None,
            Self::Computed { delta, .. } => Some(delta),
        }
    }

    #[must_use]
    pub fn after(&self) -> Option<&SparseVector> {
        match self {
            Self::Initial { .. } => None,
            Self::Computed { after, .. } => Some(after),
        }
    }

    /// Compute the delta from an after-state embedding.
    ///
    /// Transitions from `Initial` to `Computed` state.
    ///
    /// # Errors
    /// Returns error if already computed or dimensions mismatch.
    pub fn compute(self, after: SparseVector) -> Result<Self, EmbeddingError> {
        match self {
            Self::Initial { before } => {
                if before.dimension() != after.dimension() {
                    return Err(EmbeddingError::DimensionMismatch {
                        before: before.dimension(),
                        after: after.dimension(),
                    });
                }

                let delta = after.sub(&before);
                Ok(Self::Computed {
                    before,
                    after,
                    delta,
                })
            },
            Self::Computed { .. } => Err(EmbeddingError::AlreadyComputed),
        }
    }

    /// # Errors
    /// Returns error if already computed or dimensions mismatch.
    pub fn compute_from_dense(self, after: &[f32]) -> Result<Self, EmbeddingError> {
        self.compute(SparseVector::from_dense(after))
    }

    /// Compute delta using sparse diff with threshold.
    ///
    /// More efficient than `compute()` when you have dense vectors,
    /// as it directly creates a sparse delta.
    ///
    /// # Errors
    /// Returns error if already computed or dimensions mismatch.
    pub fn compute_with_threshold(
        self,
        after_dense: &[f32],
        threshold: f32,
    ) -> Result<Self, EmbeddingError> {
        match self {
            Self::Initial { before } => {
                let before_dense = before.to_dense();
                if before_dense.len() != after_dense.len() {
                    return Err(EmbeddingError::DimensionMismatch {
                        before: before_dense.len(),
                        after: after_dense.len(),
                    });
                }

                let delta = SparseVector::from_diff(&before_dense, after_dense, threshold);
                let after = SparseVector::from_dense(after_dense);

                Ok(Self::Computed {
                    before,
                    after,
                    delta,
                })
            },
            Self::Computed { .. } => Err(EmbeddingError::AlreadyComputed),
        }
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.before().dimension()
    }

    /// Get the delta as a dense vector, or zeros if not computed.
    ///
    /// This is a compatibility method for code that expects dense vectors.
    /// Prefer using `delta()` directly when possible.
    #[must_use]
    pub fn delta_or_zero(&self) -> Vec<f32> {
        match self {
            Self::Initial { before } => vec![0.0; before.dimension()],
            Self::Computed { delta, .. } => delta.to_dense(),
        }
    }

    #[must_use]
    pub fn delta_magnitude(&self) -> f32 {
        match self {
            Self::Initial { .. } => 0.0,
            Self::Computed { delta, .. } => delta.magnitude(),
        }
    }
}

impl Default for EmbeddingState {
    fn default() -> Self {
        Self::empty(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_initial_state() {
        let before = SparseVector::from_dense(&[1.0, 2.0, 3.0]);
        let state = EmbeddingState::new(before.clone());

        assert!(!state.is_computed());
        assert!(state.delta().is_none());
        assert!(state.after().is_none());
        assert_eq!(state.before().dimension(), 3);
    }

    #[test]
    fn test_from_dense() {
        let state = EmbeddingState::from_dense(&[1.0, 0.0, 2.0]);

        assert!(!state.is_computed());
        assert_eq!(state.dimension(), 3);
        assert_eq!(state.before().nnz(), 2); // Sparse: only non-zeros
    }

    #[test]
    fn test_empty() {
        let state = EmbeddingState::empty(128);

        assert!(!state.is_computed());
        assert_eq!(state.dimension(), 128);
        assert_eq!(state.before().nnz(), 0);
    }

    #[test]
    fn test_compute_transition() {
        let before = SparseVector::from_dense(&[1.0, 0.0, 0.0]);
        let after = SparseVector::from_dense(&[1.0, 1.0, 0.0]);

        let state = EmbeddingState::new(before);
        let computed = state.compute(after).unwrap();

        assert!(computed.is_computed());
        assert!(computed.delta().is_some());
        assert!(computed.after().is_some());

        // Delta should be [0, 1, 0]
        let delta = computed.delta().unwrap();
        assert_eq!(delta.nnz(), 1);
        assert!((delta.magnitude() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_already_computed_error() {
        let state = EmbeddingState::from_dense(&[1.0, 0.0]);
        let after = SparseVector::from_dense(&[1.0, 1.0]);

        let computed = state.compute(after.clone()).unwrap();
        let result = computed.compute(after);

        assert_eq!(result.unwrap_err(), EmbeddingError::AlreadyComputed);
    }

    #[test]
    fn test_compute_dimension_mismatch_error() {
        let state = EmbeddingState::from_dense(&[1.0, 0.0, 0.0]);
        let after = SparseVector::from_dense(&[1.0, 1.0]); // Wrong dimension

        let result = state.compute(after);

        match result.unwrap_err() {
            EmbeddingError::DimensionMismatch { before, after } => {
                assert_eq!(before, 3);
                assert_eq!(after, 2);
            },
            _ => panic!("expected DimensionMismatch"),
        }
    }

    #[test]
    fn test_compute_from_dense() {
        let state = EmbeddingState::from_dense(&[1.0, 2.0]);
        let computed = state.compute_from_dense(&[3.0, 2.0]).unwrap();

        assert!(computed.is_computed());
        // Delta should be [2, 0]
        let delta = computed.delta().unwrap();
        assert_eq!(delta.nnz(), 1);
    }

    #[test]
    fn test_compute_with_threshold() {
        let state = EmbeddingState::from_dense(&[1.0, 2.0, 3.0]);
        let after = [1.001, 2.0, 5.0]; // Small change at 0, no change at 1, large at 2

        let computed = state.compute_with_threshold(&after, 0.01).unwrap();

        // Only position 2 should be in delta (change of 2.0 > threshold)
        let delta = computed.delta().unwrap();
        assert_eq!(delta.nnz(), 1);
    }

    #[test]
    fn test_delta_or_zero_initial() {
        let state = EmbeddingState::from_dense(&[1.0, 2.0, 3.0]);
        let delta = state.delta_or_zero();

        assert_eq!(delta, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_delta_or_zero_computed() {
        let state = EmbeddingState::from_dense(&[1.0, 0.0]);
        let computed = state.compute_from_dense(&[2.0, 0.0]).unwrap();
        let delta = computed.delta_or_zero();

        assert_eq!(delta, vec![1.0, 0.0]);
    }

    #[test]
    fn test_delta_magnitude() {
        let state = EmbeddingState::from_dense(&[0.0, 0.0]);
        assert_eq!(state.delta_magnitude(), 0.0);

        let computed = state.compute_from_dense(&[3.0, 4.0]).unwrap();
        assert!((computed.delta_magnitude() - 5.0).abs() < 1e-6); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_default() {
        let state = EmbeddingState::default();
        assert!(!state.is_computed());
        assert_eq!(state.dimension(), 0);
    }

    #[test]
    fn test_serde() {
        let state = EmbeddingState::from_dense(&[1.0, 2.0]);
        let computed = state.compute_from_dense(&[3.0, 4.0]).unwrap();

        let serialized = bitcode::serialize(&computed).unwrap();
        let deserialized: EmbeddingState = bitcode::deserialize(&serialized).unwrap();

        assert!(deserialized.is_computed());
        assert_eq!(deserialized.delta_or_zero(), vec![2.0, 2.0]);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(
            format!("{}", EmbeddingError::NotComputed),
            "delta not yet computed"
        );
        assert_eq!(
            format!("{}", EmbeddingError::AlreadyComputed),
            "delta already computed"
        );
        assert_eq!(
            format!(
                "{}",
                EmbeddingError::DimensionMismatch {
                    before: 3,
                    after: 2
                }
            ),
            "dimension mismatch: before=3, after=2"
        );
    }

    #[test]
    fn test_before_always_available() {
        let initial = EmbeddingState::from_dense(&[1.0, 2.0]);
        let _ = initial.before(); // Should not panic

        let computed = EmbeddingState::from_dense(&[1.0, 2.0])
            .compute_from_dense(&[3.0, 4.0])
            .unwrap();
        let _ = computed.before(); // Should not panic
    }
}
