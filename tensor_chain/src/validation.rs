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
        let (global_entry, global_similarity) = self.global.quantize(state).unwrap_or((0, 0.0));

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

/// Mode for validation - controls how thorough validation should be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValidationMode {
    /// Full validation using codebook-based state and transition checks.
    #[default]
    Full,
    /// Fast-path validation using only similarity comparison.
    /// Contains the similarity score that triggered fast-path.
    FastPath,
    /// Skip validation entirely (for trusted sources).
    Trusted,
}

/// Result of fast-path validation check.
#[derive(Debug, Clone)]
pub struct FastPathResult {
    /// Whether fast-path can be used.
    pub can_use_fast_path: bool,
    /// Similarity score with recent embeddings.
    pub similarity: f32,
    /// Number of blocks checked for similarity.
    pub blocks_checked: usize,
    /// Reason for rejection (if any).
    pub rejection_reason: Option<String>,
}

impl FastPathResult {
    /// Create a result indicating fast-path can be used.
    pub fn accept(similarity: f32, blocks_checked: usize) -> Self {
        Self {
            can_use_fast_path: true,
            similarity,
            blocks_checked,
            rejection_reason: None,
        }
    }

    /// Create a result indicating fast-path cannot be used.
    pub fn reject(reason: &str, similarity: f32, blocks_checked: usize) -> Self {
        Self {
            can_use_fast_path: false,
            similarity,
            blocks_checked,
            rejection_reason: Some(reason.to_string()),
        }
    }
}

/// Validates whether fast-path replication can be used.
pub struct FastPathValidator {
    /// Minimum similarity threshold for fast-path.
    pub similarity_threshold: f32,
    /// Minimum number of blocks from leader before allowing fast-path.
    pub min_leader_history: usize,
    /// Interval for forcing full validation (every N blocks).
    pub full_validation_interval: usize,
    /// Blocks since last full validation.
    blocks_since_full: std::sync::atomic::AtomicUsize,
}

impl Default for FastPathValidator {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.95,
            min_leader_history: 3,
            full_validation_interval: 10,
            blocks_since_full: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl FastPathValidator {
    /// Create a new fast-path validator with custom thresholds.
    pub fn new(similarity_threshold: f32, min_leader_history: usize) -> Self {
        Self {
            similarity_threshold,
            min_leader_history,
            full_validation_interval: 10,
            blocks_since_full: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Check if fast-path validation can be used.
    pub fn check_fast_path(
        &self,
        block_embedding: &[f32],
        recent_embeddings: &[Vec<f32>],
    ) -> FastPathResult {
        // Need minimum history from leader
        if recent_embeddings.len() < self.min_leader_history {
            return FastPathResult::reject(
                "insufficient leader history",
                0.0,
                recent_embeddings.len(),
            );
        }

        // Force full validation periodically
        let blocks_since = self
            .blocks_since_full
            .load(std::sync::atomic::Ordering::Relaxed);
        if blocks_since >= self.full_validation_interval {
            return FastPathResult::reject(
                "periodic full validation required",
                0.0,
                recent_embeddings.len(),
            );
        }

        // Check similarity with recent embeddings
        let max_similarity = recent_embeddings
            .iter()
            .map(|emb| cosine_similarity(block_embedding, emb))
            .fold(0.0f32, |max, sim| max.max(sim));

        if max_similarity >= self.similarity_threshold {
            FastPathResult::accept(max_similarity, recent_embeddings.len())
        } else {
            FastPathResult::reject(
                "similarity below threshold",
                max_similarity,
                recent_embeddings.len(),
            )
        }
    }

    /// Record that a block was validated.
    /// Call with `used_fast_path=true` if fast-path was used.
    pub fn record_validation(&self, used_fast_path: bool) {
        if used_fast_path {
            self.blocks_since_full
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            self.blocks_since_full
                .store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Reset the full validation counter.
    pub fn reset(&self) {
        self.blocks_since_full
            .store(0, std::sync::atomic::Ordering::Relaxed);
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
        let centroids = vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1, 0.0]];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let validator = TransitionValidator::new(global, config);

        // Small valid transition
        let validation =
            validator.validate_transition("test", &[1.0, 0.0, 0.0], &[0.95, 0.05, 0.0]);
        assert!(validation.is_valid);
        assert!(validation.magnitude < 0.5);

        // Large invalid transition
        let validation = validator.validate_transition("test", &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
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
            vec![1.0, 0.0, 0.0], // Valid (sim = 1.0)
            vec![0.5, 0.5, 0.5], // Invalid (low similarity)
            vec![0.0, 1.0, 0.0], // Valid (sim = 1.0)
        ];

        let (idx, deviation) = validator.find_max_deviation("test", &states).unwrap();
        assert_eq!(idx, 1); // Middle state has highest deviation
        assert!(deviation > 0.4); // Deviation should be significant
    }

    #[test]
    fn test_learn_from_states() {
        let validator = create_test_validator();

        // Learn some states
        let states = vec![vec![0.5, 0.5, 0.0], vec![0.6, 0.4, 0.0]];
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

    #[test]
    fn test_validation_mode_default() {
        let mode = ValidationMode::default();
        assert_eq!(mode, ValidationMode::Full);
    }

    #[test]
    fn test_fast_path_result_accept() {
        let result = FastPathResult::accept(0.98, 5);
        assert!(result.can_use_fast_path);
        assert_eq!(result.similarity, 0.98);
        assert_eq!(result.blocks_checked, 5);
        assert!(result.rejection_reason.is_none());
    }

    #[test]
    fn test_fast_path_result_reject() {
        let result = FastPathResult::reject("test reason", 0.5, 3);
        assert!(!result.can_use_fast_path);
        assert_eq!(result.similarity, 0.5);
        assert_eq!(result.blocks_checked, 3);
        assert_eq!(result.rejection_reason.as_deref(), Some("test reason"));
    }

    #[test]
    fn test_fast_path_validator_insufficient_history() {
        let validator = FastPathValidator::new(0.95, 3);

        // Only 2 recent embeddings (need 3)
        let recent = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let result = validator.check_fast_path(&[1.0, 0.0, 0.0], &recent);
        assert!(!result.can_use_fast_path);
        assert!(result
            .rejection_reason
            .as_deref()
            .unwrap()
            .contains("insufficient"));
    }

    #[test]
    fn test_fast_path_validator_high_similarity() {
        let validator = FastPathValidator::new(0.95, 2);

        // Similar embeddings
        let recent = vec![vec![1.0, 0.0, 0.0], vec![0.98, 0.02, 0.0]];

        let result = validator.check_fast_path(&[0.99, 0.01, 0.0], &recent);
        assert!(result.can_use_fast_path);
        assert!(result.similarity >= 0.95);
    }

    #[test]
    fn test_fast_path_validator_low_similarity() {
        let validator = FastPathValidator::new(0.95, 2);

        // Dissimilar embeddings
        let recent = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        let result = validator.check_fast_path(&[0.0, 0.0, 1.0], &recent);
        assert!(!result.can_use_fast_path);
        assert!(result.similarity < 0.95);
    }

    #[test]
    fn test_fast_path_validator_periodic_full() {
        let validator = FastPathValidator::new(0.95, 2);

        // Simulate many fast-path validations
        for _ in 0..10 {
            validator.record_validation(true);
        }

        // Should force full validation after interval
        let recent = vec![vec![1.0, 0.0, 0.0], vec![0.99, 0.01, 0.0]];
        let result = validator.check_fast_path(&[1.0, 0.0, 0.0], &recent);
        assert!(!result.can_use_fast_path);
        assert!(result
            .rejection_reason
            .as_deref()
            .unwrap()
            .contains("periodic"));
    }

    #[test]
    fn test_fast_path_validator_reset() {
        let validator = FastPathValidator::new(0.95, 2);

        // Simulate fast-path validations
        for _ in 0..5 {
            validator.record_validation(true);
        }

        // Record full validation
        validator.record_validation(false);

        // Now fast-path should work again
        let recent = vec![vec![1.0, 0.0, 0.0], vec![0.99, 0.01, 0.0]];
        let result = validator.check_fast_path(&[1.0, 0.0, 0.0], &recent);
        assert!(result.can_use_fast_path);
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.state_threshold, 0.8);
        assert_eq!(config.max_transition_magnitude, 1.0);
        assert!(config.strict_transition);
    }

    #[test]
    fn test_validation_config_clone_debug() {
        let config = ValidationConfig::default();
        let cloned = config.clone();
        assert_eq!(config.state_threshold, cloned.state_threshold);

        let debug = format!("{:?}", config);
        assert!(debug.contains("ValidationConfig"));
    }

    #[test]
    fn test_state_validation_debug_clone() {
        let validation = StateValidation {
            is_valid: true,
            global_entry: Some(42),
            global_similarity: 0.95,
            local_entry: Some(10),
            local_similarity: 0.85,
            domain: "test".to_string(),
        };

        let cloned = validation.clone();
        assert_eq!(validation.is_valid, cloned.is_valid);
        assert_eq!(validation.global_entry, cloned.global_entry);

        let debug = format!("{:?}", validation);
        assert!(debug.contains("StateValidation"));
    }

    #[test]
    fn test_transition_validation_debug_clone() {
        let state_val = StateValidation {
            is_valid: true,
            global_entry: Some(0),
            global_similarity: 1.0,
            local_entry: None,
            local_similarity: 0.0,
            domain: "test".to_string(),
        };

        let validation = TransitionValidation {
            is_valid: true,
            from_validation: state_val.clone(),
            to_validation: state_val,
            magnitude: 0.5,
            direction_similarity: 0.9,
            rejection_reason: None,
        };

        let cloned = validation.clone();
        assert_eq!(validation.is_valid, cloned.is_valid);
        assert_eq!(validation.magnitude, cloned.magnitude);

        let debug = format!("{:?}", validation);
        assert!(debug.contains("TransitionValidation"));
    }

    #[test]
    fn test_transition_validator_global_accessor() {
        let validator = create_test_validator();
        let global = validator.global();
        assert_eq!(global.len(), 3);
    }

    #[test]
    fn test_transition_validator_get_or_create_local() {
        let validator = create_test_validator();
        let local = validator.get_or_create_local("new_domain");
        assert_eq!(local.dimension(), 3);
    }

    #[test]
    fn test_transition_validator_register_local() {
        let validator = create_test_validator();

        let local = LocalCodebook::new("my_domain", 3, 10, 0.9);
        validator.register_local("my_domain", local);

        let locals = validator.locals.read();
        assert!(locals.contains_key("my_domain"));
    }

    #[test]
    fn test_is_valid_state() {
        let validator = create_test_validator();

        assert!(validator.is_valid_state("test", &[1.0, 0.0, 0.0]));
        assert!(!validator.is_valid_state("test", &[0.5, 0.5, 0.5]));
    }

    #[test]
    fn test_is_valid_transition() {
        let config = ValidationConfig {
            state_threshold: 0.9,
            max_transition_magnitude: 0.5,
            strict_transition: true,
            codebook_config: CodebookConfig::default(),
        };
        let centroids = vec![vec![1.0, 0.0, 0.0], vec![0.9, 0.1, 0.0]];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let validator = TransitionValidator::new(global, config);

        // Small valid transition
        assert!(validator.is_valid_transition("test", &[1.0, 0.0, 0.0], &[0.95, 0.05, 0.0]));

        // Large invalid transition
        assert!(!validator.is_valid_transition("test", &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_non_strict_transition_validation() {
        let config = ValidationConfig {
            state_threshold: 0.9,
            max_transition_magnitude: 1.5,
            strict_transition: false, // Non-strict
            codebook_config: CodebookConfig::default(),
        };
        let centroids = vec![vec![1.0, 0.0, 0.0]];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let validator = TransitionValidator::new(global, config);

        // Non-strict allows invalid states if magnitude is within limit
        let validation =
            validator.validate_transition("test", &[0.5, 0.5, 0.0], &[0.6, 0.4, 0.0]);
        assert!(validation.is_valid);
    }

    #[test]
    fn test_non_strict_transition_exceeds_magnitude() {
        let config = ValidationConfig {
            state_threshold: 0.9,
            max_transition_magnitude: 0.1, // Very small limit
            strict_transition: false,
            codebook_config: CodebookConfig::default(),
        };
        let centroids = vec![vec![1.0, 0.0, 0.0]];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let validator = TransitionValidator::new(global, config);

        // Even non-strict fails if magnitude exceeds limit
        let validation =
            validator.validate_transition("test", &[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!(!validation.is_valid);
        assert!(validation
            .rejection_reason
            .as_deref()
            .unwrap()
            .contains("magnitude"));
    }

    #[test]
    fn test_transition_invalid_source_state() {
        let config = ValidationConfig {
            state_threshold: 0.99,
            max_transition_magnitude: 10.0,
            strict_transition: true,
            codebook_config: CodebookConfig::default(),
        };
        let centroids = vec![vec![1.0, 0.0, 0.0]];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let validator = TransitionValidator::new(global, config);

        // Invalid source state
        let validation =
            validator.validate_transition("test", &[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!(!validation.is_valid);
        assert!(validation
            .rejection_reason
            .as_deref()
            .unwrap()
            .contains("source"));
    }

    #[test]
    fn test_transition_invalid_target_state() {
        let config = ValidationConfig {
            state_threshold: 0.99,
            max_transition_magnitude: 10.0,
            strict_transition: true,
            codebook_config: CodebookConfig::default(),
        };
        let centroids = vec![vec![1.0, 0.0, 0.0]];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let validator = TransitionValidator::new(global, config);

        // Invalid target state
        let validation =
            validator.validate_transition("test", &[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]);
        assert!(!validation.is_valid);
        assert!(validation
            .rejection_reason
            .as_deref()
            .unwrap()
            .contains("target"));
    }

    #[test]
    fn test_validate_path_short() {
        let validator = create_test_validator();

        // Empty path
        assert!(validator.validate_path("test", &[]).is_ok());

        // Single state
        assert!(validator.validate_path("test", &[vec![1.0, 0.0, 0.0]]).is_ok());
    }

    #[test]
    fn test_validate_path_invalid() {
        let config = ValidationConfig {
            state_threshold: 0.99,
            max_transition_magnitude: 0.1,
            strict_transition: true,
            codebook_config: CodebookConfig::default(),
        };
        let centroids = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let validator = TransitionValidator::new(global, config);

        // Invalid path (big jump)
        let path = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let result = validator.validate_path("test", &path);
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_path_drift_short() {
        let validator = create_test_validator();

        // Empty path
        assert_eq!(validator.compute_path_drift(&[]), 0.0);

        // Single state
        assert_eq!(validator.compute_path_drift(&[vec![1.0, 0.0, 0.0]]), 0.0);
    }

    #[test]
    fn test_find_max_deviation_empty() {
        let validator = create_test_validator();

        let result = validator.find_max_deviation("test", &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_cosine_similarity_zero_magnitude() {
        // Zero magnitude vectors
        let sim = cosine_similarity(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert_eq!(sim, 0.0);

        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert_eq!(sim, 0.0);

        let sim = cosine_similarity(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_validation_mode_debug_copy() {
        let mode = ValidationMode::FastPath;
        let copied = mode;
        assert_eq!(mode, copied);

        let debug = format!("{:?}", mode);
        assert!(debug.contains("FastPath"));

        let debug = format!("{:?}", ValidationMode::Full);
        assert!(debug.contains("Full"));

        let debug = format!("{:?}", ValidationMode::Trusted);
        assert!(debug.contains("Trusted"));
    }

    #[test]
    fn test_fast_path_result_debug_clone() {
        let result = FastPathResult::accept(0.97, 4);
        let cloned = result.clone();
        assert_eq!(result.similarity, cloned.similarity);

        let debug = format!("{:?}", result);
        assert!(debug.contains("FastPathResult"));
    }

    #[test]
    fn test_fast_path_validator_default() {
        let validator = FastPathValidator::default();
        assert_eq!(validator.similarity_threshold, 0.95);
        assert_eq!(validator.min_leader_history, 3);
        assert_eq!(validator.full_validation_interval, 10);
    }

    #[test]
    fn test_fast_path_validator_reset_method() {
        let validator = FastPathValidator::new(0.95, 2);

        // Simulate some validations
        for _ in 0..5 {
            validator.record_validation(true);
        }

        // Verify counter incremented
        let count = validator
            .blocks_since_full
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(count, 5);

        // Reset
        validator.reset();

        // Verify counter is zero
        let count = validator
            .blocks_since_full
            .load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_state_validation_with_local_codebook() {
        let centroids = vec![vec![1.0, 0.0, 0.0]];
        let global = Arc::new(GlobalCodebook::from_centroids(centroids));
        let config = ValidationConfig {
            state_threshold: 0.99, // Very high threshold
            ..Default::default()
        };
        let validator = TransitionValidator::new(global, config);

        // Register a local codebook with a different centroid
        let mut local = LocalCodebook::new("domain", 3, 10, 0.9);
        local.quantize_and_update(&[0.0, 1.0, 0.0], 0.1);
        validator.register_local("domain", local);

        // State should be valid via local codebook
        let validation = validator.validate_state("domain", &[0.0, 1.0, 0.0]);
        assert!(validation.local_entry.is_some());
        assert!(validation.local_similarity > 0.0);
    }

    #[test]
    fn test_validate_state_no_local_codebook() {
        let validator = create_test_validator();

        // State validated against global only (no local codebook for domain)
        let validation = validator.validate_state("unknown_domain", &[1.0, 0.0, 0.0]);
        assert!(validation.is_valid);
        assert!(validation.local_entry.is_none());
        assert_eq!(validation.local_similarity, 0.0);
    }
}
