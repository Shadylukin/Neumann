//! Codebook-based transition validation for the tensor chain.
//!
//! Validates state transitions to ensure they stay within the
//! semantic vocabulary defined by the codebook system:
//!
//! - States must be near known codebook entries
//! - Transitions must have bounded magnitude
//! - Domain-specific validation via local codebooks

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::codebook::{CodebookConfig, GlobalCodebook, LocalCodebook};
use crate::error::{ChainError, Result};

/// Validation configuration.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Minimum similarity for a state to be considered valid.
    pub state_threshold: f32,
    /// Maximum allowed transition magnitude.
    pub max_transition_magnitude: f32,
    /// Whether to require both states be valid for a valid transition.
    pub strict_transition: bool,
    /// Codebook configuration.
    pub codebook_config: CodebookConfig,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            state_threshold: 0.8,
            max_transition_magnitude: 1.0,
            strict_transition: true,
            codebook_config: CodebookConfig::default(),
        }
    }
}

/// Result of state validation.
#[derive(Debug, Clone)]
pub struct StateValidation {
    /// Whether the state is valid.
    pub is_valid: bool,
    /// Nearest global codebook entry ID.
    pub global_entry: Option<u32>,
    /// Similarity to global entry.
    pub global_similarity: f32,
    /// Nearest local codebook entry ID (if domain-specific).
    pub local_entry: Option<u32>,
    /// Similarity to local entry.
    pub local_similarity: f32,
    /// Domain that was checked.
    pub domain: String,
}

/// Result of transition validation.
#[derive(Debug, Clone)]
pub struct TransitionValidation {
    /// Whether the transition is valid.
    pub is_valid: bool,
    /// Validation result for the source state.
    pub from_validation: StateValidation,
    /// Validation result for the target state.
    pub to_validation: StateValidation,
    /// Euclidean magnitude of the transition.
    pub magnitude: f32,
    /// Cosine similarity between from and to states.
    pub direction_similarity: f32,
    /// Reason for rejection (if any).
    pub rejection_reason: Option<String>,
}

/// Validates state transitions using the hierarchical codebook system.
pub struct TransitionValidator {
    /// Global codebook (shared across all nodes).
    global: Arc<GlobalCodebook>,
    /// Local codebooks per domain.
    locals: RwLock<HashMap<String, LocalCodebook>>,
    /// Validation configuration.
    config: ValidationConfig,
}

impl TransitionValidator {
    /// Create a new transition validator.
    pub fn new(global: Arc<GlobalCodebook>, config: ValidationConfig) -> Self {
        Self {
            global,
            locals: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Create with default configuration.
    pub fn with_global(global: Arc<GlobalCodebook>) -> Self {
        Self::new(global, ValidationConfig::default())
    }

    /// Get the global codebook.
    pub fn global(&self) -> &GlobalCodebook {
        &self.global
    }

    /// Get or create a local codebook for a domain.
    pub fn get_or_create_local(&self, domain: &str) -> LocalCodebook {
        LocalCodebook::new(
            domain,
            self.global.dimension(),
            self.config.codebook_config.local_capacity,
            self.config.codebook_config.ema_alpha,
        )
    }

    /// Register a local codebook for a domain.
    pub fn register_local(&self, domain: &str, local: LocalCodebook) {
        self.locals.write().insert(domain.to_string(), local);
    }

    /// Check if a state is valid (near a codebook entry).
    pub fn is_valid_state(&self, domain: &str, state: &[f32]) -> bool {
        self.validate_state(domain, state).is_valid
    }

    /// Validate a state and return detailed results.
    pub fn validate_state(&self, domain: &str, state: &[f32]) -> StateValidation {
        // Check global codebook
        let (global_entry, global_similarity) = self
            .global
            .quantize(state)
            .unwrap_or((0, 0.0));

        // Check local codebook if exists
        let (local_entry, local_similarity) = {
            let locals = self.locals.read();
            if let Some(local) = locals.get(domain) {
                local.quantize(state).unwrap_or((0, 0.0))
            } else {
                (0, 0.0)
            }
        };

        // Valid if either global or local similarity meets threshold
        let is_valid = global_similarity >= self.config.state_threshold
            || local_similarity >= self.config.state_threshold;

        StateValidation {
            is_valid,
            global_entry: Some(global_entry),
            global_similarity,
            local_entry: if local_similarity > 0.0 {
                Some(local_entry)
            } else {
                None
            },
            local_similarity,
            domain: domain.to_string(),
        }
    }

    /// Check if a transition is valid.
    pub fn is_valid_transition(&self, domain: &str, from: &[f32], to: &[f32]) -> bool {
        self.validate_transition(domain, from, to).is_valid
    }

    /// Validate a transition and return detailed results.
    pub fn validate_transition(
        &self,
        domain: &str,
        from: &[f32],
        to: &[f32],
    ) -> TransitionValidation {
        let from_validation = self.validate_state(domain, from);
        let to_validation = self.validate_state(domain, to);

        // Compute transition magnitude
        let magnitude: f32 = from
            .iter()
            .zip(to.iter())
            .map(|(f, t)| (t - f).powi(2))
            .sum::<f32>()
            .sqrt();

        // Compute direction similarity
        let direction_similarity = cosine_similarity(from, to);

        // Determine validity
        let (is_valid, rejection_reason) = if self.config.strict_transition {
            if !from_validation.is_valid {
                (false, Some("source state invalid".to_string()))
            } else if !to_validation.is_valid {
                (false, Some("target state invalid".to_string()))
            } else if magnitude > self.config.max_transition_magnitude {
                (
                    false,
                    Some(format!(
                        "transition magnitude {} exceeds max {}",
                        magnitude, self.config.max_transition_magnitude
                    )),
                )
            } else {
                (true, None)
            }
        } else {
            // Non-strict: only check magnitude
            if magnitude > self.config.max_transition_magnitude {
                (
                    false,
                    Some(format!(
                        "transition magnitude {} exceeds max {}",
                        magnitude, self.config.max_transition_magnitude
                    )),
                )
            } else {
                (true, None)
            }
        };

        TransitionValidation {
            is_valid,
            from_validation,
            to_validation,
            magnitude,
            direction_similarity,
            rejection_reason,
        }
    }

    /// Validate a batch of transitions.
    pub fn validate_batch(
        &self,
        domain: &str,
        transitions: &[(Vec<f32>, Vec<f32>)],
    ) -> Vec<TransitionValidation> {
        transitions
            .iter()
            .map(|(from, to)| self.validate_transition(domain, from, to))
            .collect()
    }

    /// Check if a sequence of states forms a valid path.
    pub fn validate_path(&self, domain: &str, states: &[Vec<f32>]) -> Result<()> {
        if states.len() < 2 {
            return Ok(());
        }

        for (i, window) in states.windows(2).enumerate() {
            let validation = self.validate_transition(domain, &window[0], &window[1]);
            if !validation.is_valid {
                return Err(ChainError::ValidationFailed(format!(
                    "transition {} -> {}: {}",
                    i,
                    i + 1,
                    validation.rejection_reason.unwrap_or_default()
                )));
            }
        }

        Ok(())
    }

    /// Compute the accumulated drift over a sequence of states.
    pub fn compute_path_drift(&self, states: &[Vec<f32>]) -> f32 {
        if states.len() < 2 {
            return 0.0;
        }

        states
            .windows(2)
            .map(|window| {
                window[0]
                    .iter()
                    .zip(window[1].iter())
                    .map(|(a, b)| (b - a).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .sum()
    }

    /// Find the state in the path that deviates most from valid states.
    pub fn find_max_deviation(&self, domain: &str, states: &[Vec<f32>]) -> Option<(usize, f32)> {
        states
            .iter()
            .enumerate()
            .map(|(i, state)| {
                let validation = self.validate_state(domain, state);
                let max_sim = validation
                    .global_similarity
                    .max(validation.local_similarity);
                (i, 1.0 - max_sim) // Deviation = 1 - similarity
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Update the local codebook for a domain with observed states.
    pub fn learn_from_states(&self, domain: &str, states: &[Vec<f32>], threshold: f32) {
        let mut locals = self.locals.write();
        let local = locals.entry(domain.to_string()).or_insert_with(|| {
            LocalCodebook::new(
                domain,
                self.global.dimension(),
                self.config.codebook_config.local_capacity,
                self.config.codebook_config.ema_alpha,
            )
        });

        for state in states {
            local.quantize_and_update(state, threshold);
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validator() -> TransitionValidator {
        let centroids = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        TransitionValidator::with_global(global)
    }

    #[test]
    fn test_state_validation() {
        let validator = create_test_validator();

        // Valid state (close to centroid)
        let validation = validator.validate_state("test", &[1.0, 0.0, 0.0]);
        assert!(validation.is_valid);
        assert!(validation.global_similarity > 0.99);

        // Slightly off but still valid
        let validation = validator.validate_state("test", &[0.95, 0.05, 0.0]);
        assert!(validation.is_valid);

        // Invalid state (too far from any centroid)
        let validation = validator.validate_state("test", &[0.5, 0.5, 0.5]);
        // With default threshold of 0.8, this should be invalid
        // cos([0.5,0.5,0.5], [1,0,0]) = 0.5/0.866 = 0.577 < 0.8
        assert!(!validation.is_valid);
    }

    #[test]
    fn test_transition_validation() {
        let config = ValidationConfig {
            state_threshold: 0.9,
            max_transition_magnitude: 0.5,
            strict_transition: true,
            codebook_config: CodebookConfig::default(),
        };
        let centroids = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
        ];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let validator = TransitionValidator::new(global, config);

        // Small valid transition
        let validation = validator.validate_transition(
            "test",
            &[1.0, 0.0, 0.0],
            &[0.95, 0.05, 0.0],
        );
        assert!(validation.is_valid);
        assert!(validation.magnitude < 0.5);

        // Large invalid transition
        let validation = validator.validate_transition(
            "test",
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
        );
        assert!(!validation.is_valid);
        assert!(validation.magnitude > 0.5);
    }

    #[test]
    fn test_path_validation() {
        let config = ValidationConfig {
            state_threshold: 0.9,
            max_transition_magnitude: 0.3,
            strict_transition: true,
            codebook_config: CodebookConfig::default(),
        };
        let centroids = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.8, 0.2, 0.0],
            vec![0.7, 0.3, 0.0],
        ];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let validator = TransitionValidator::new(global, config);

        // Valid path (small steps)
        let path = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.95, 0.05, 0.0],
            vec![0.9, 0.1, 0.0],
        ];
        assert!(validator.validate_path("test", &path).is_ok());
    }

    #[test]
    fn test_path_drift() {
        let validator = create_test_validator();

        let path = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];

        let drift = validator.compute_path_drift(&path);
        // First hop: 1.0, Second hop: 1.0, Total: 2.0
        assert!((drift - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_max_deviation() {
        let validator = create_test_validator();

        let states = vec![
            vec![1.0, 0.0, 0.0],     // Valid (sim = 1.0)
            vec![0.5, 0.5, 0.5],     // Invalid (low similarity)
            vec![0.0, 1.0, 0.0],     // Valid (sim = 1.0)
        ];

        let (idx, deviation) = validator.find_max_deviation("test", &states).unwrap();
        assert_eq!(idx, 1); // Middle state has highest deviation
        assert!(deviation > 0.4); // Deviation should be significant
    }

    #[test]
    fn test_learn_from_states() {
        let validator = create_test_validator();

        // Learn some states
        let states = vec![
            vec![0.5, 0.5, 0.0],
            vec![0.6, 0.4, 0.0],
        ];
        validator.learn_from_states("custom", &states, 0.9);

        // Now validate against learned domain
        // Note: The learned states are stored in the local codebook
        let locals = validator.locals.read();
        assert!(locals.contains_key("custom"));
        assert!(locals.get("custom").unwrap().len() > 0);
    }

    #[test]
    fn test_batch_validation() {
        let validator = create_test_validator();

        let transitions = vec![
            (vec![1.0, 0.0, 0.0], vec![0.9, 0.1, 0.0]),
            (vec![0.0, 1.0, 0.0], vec![0.1, 0.9, 0.0]),
        ];

        let results = validator.validate_batch("test", &transitions);
        assert_eq!(results.len(), 2);
    }
}
