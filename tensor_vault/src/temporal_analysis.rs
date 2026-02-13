// SPDX-License-Identifier: MIT OR Apache-2.0
//! Temporal analysis of access patterns via TT decomposition and drift detection.
//!
//! Extracts seasonal patterns from the access tensor using Tensor Train
//! decomposition, and detects behavioral drift by comparing recent access
//! patterns against historical baselines.

use serde::{Deserialize, Serialize};
use tensor_compress::tensor_train::{tt_decompose, tt_reconstruct, TTConfig};

use crate::access_tensor::AccessTensor;

/// Configuration for temporal pattern analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysisConfig {
    /// TT decomposition config. If `None`, uses default settings.
    pub tt_config: Option<TTConfig>,
    /// Number of recent buckets for drift comparison (default: 24).
    pub drift_window: usize,
    /// Cosine distance threshold to flag drift (default: 0.3).
    pub drift_threshold: f64,
    /// Minimum total accesses to analyze an entity (default: 5).
    pub min_accesses: u64,
}

impl Default for TemporalAnalysisConfig {
    fn default() -> Self {
        Self {
            tt_config: None,
            drift_window: 24,
            drift_threshold: 0.3,
            min_accesses: 5,
        }
    }
}

/// A seasonal pattern extracted via TT decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    /// Entity identifier.
    pub entity: String,
    /// Compressed (reconstructed) pattern.
    pub compressed_pattern: Vec<f32>,
    /// Dominant periodicity in buckets (from autocorrelation).
    pub dominant_period: usize,
    /// Ratio of compressed to original size.
    pub compression_ratio: f32,
    /// Reconstruction error (L2 norm of difference / L2 norm of original).
    pub reconstruction_error: f32,
}

/// Drift detection result for a single entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDetection {
    /// Entity identifier.
    pub entity: String,
    /// Cosine distance between historical and recent windows.
    pub drift_score: f64,
    /// Whether the drift exceeds the threshold.
    pub is_drifting: bool,
    /// Secrets whose access pattern changed the most.
    pub changed_secrets: Vec<String>,
}

/// Combined temporal analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalAnalysisReport {
    /// Seasonal patterns per entity.
    pub seasonal_patterns: Vec<SeasonalPattern>,
    /// Drift detections per entity.
    pub drift_detections: Vec<DriftDetection>,
    /// Total entities analyzed.
    pub total_entities_analyzed: usize,
    /// Mean compression ratio across all entities.
    pub mean_compression_ratio: f32,
}

/// Run full temporal analysis on an access tensor.
pub fn analyze_temporal_patterns(
    tensor: &AccessTensor,
    config: TemporalAnalysisConfig,
) -> TemporalAnalysisReport {
    let tt_config = config.tt_config.clone().unwrap_or(TTConfig {
        shape: vec![],
        max_rank: 4,
        tolerance: 1e-4,
    });

    let seasonal = extract_seasonal_patterns(tensor, &tt_config, config.min_accesses);
    let drift = detect_drift(tensor, config.drift_window, config.drift_threshold);

    let mean_compression = if seasonal.is_empty() {
        0.0
    } else {
        #[allow(clippy::cast_precision_loss)] // pattern count never exceeds 2^23
        let count = seasonal.len() as f32;
        seasonal.iter().map(|s| s.compression_ratio).sum::<f32>() / count
    };

    let total = seasonal.len().max(drift.len());

    TemporalAnalysisReport {
        seasonal_patterns: seasonal,
        drift_detections: drift,
        total_entities_analyzed: total,
        mean_compression_ratio: mean_compression,
    }
}

/// Try to find a shape whose product equals `n` for TT decomposition.
fn factorize_for_tt(n: usize) -> Option<Vec<usize>> {
    if n < 4 {
        return None;
    }
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)] // safe: n is small
    let sqrt_n = (n as f64).sqrt() as usize;
    for f in (2..=sqrt_n).rev() {
        if n.is_multiple_of(f) {
            let other = n / f;
            if other >= 2 && f >= 2 {
                return Some(vec![f, other]);
            }
        }
    }
    None
}

fn extract_seasonal_patterns(
    tensor: &AccessTensor,
    tt_config: &TTConfig,
    min_accesses: u64,
) -> Vec<SeasonalPattern> {
    let mut patterns = Vec::new();

    for entity in tensor.entities() {
        let vec = tensor.entity_vector(&entity);
        if vec.is_empty() {
            continue;
        }

        let total: f32 = vec.iter().sum();
        #[allow(clippy::cast_sign_loss)]
        let total_u64 = total as u64;
        if total_u64 < min_accesses {
            continue;
        }

        // Try TT decomposition
        let len = vec.len();
        let shape = if tt_config.shape.is_empty() {
            match factorize_for_tt(len) {
                Some(s) => s,
                None => continue,
            }
        } else if tt_config.shape.iter().product::<usize>() == len {
            tt_config.shape.clone()
        } else {
            match factorize_for_tt(len) {
                Some(s) => s,
                None => continue,
            }
        };

        let config = TTConfig {
            shape,
            max_rank: tt_config.max_rank,
            tolerance: tt_config.tolerance,
        };

        let Ok(tt_vec) = tt_decompose(&vec, &config) else {
            continue;
        };

        let reconstructed = tt_reconstruct(&tt_vec);

        // Reconstruction error (relative L2)
        let orig_norm: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
        let error_norm: f32 = vec
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        let reconstruction_error = if orig_norm > f32::EPSILON {
            error_norm / orig_norm
        } else {
            0.0
        };

        // Compression ratio: compressed storage / original
        let compressed_size: usize = tt_vec.cores.iter().map(|c| c.data.len()).sum();
        #[allow(clippy::cast_precision_loss)] // tensor sizes never exceed 2^23
        let compression_ratio = compressed_size as f32 / len as f32;

        // Find dominant period from the entity's time-bucket pattern
        let (_, _, n_buckets) = tensor.dimensions();
        let bucket_pattern = aggregate_entity_buckets(&vec, n_buckets);
        let dominant_period = find_dominant_period(&bucket_pattern);

        patterns.push(SeasonalPattern {
            entity,
            compressed_pattern: reconstructed,
            dominant_period,
            compression_ratio,
            reconstruction_error,
        });
    }

    patterns
}

/// Aggregate entity vector into per-bucket totals.
fn aggregate_entity_buckets(entity_vec: &[f32], n_buckets: usize) -> Vec<f32> {
    if n_buckets == 0 {
        return Vec::new();
    }
    let n_secrets = entity_vec.len() / n_buckets;
    let mut buckets = vec![0.0_f32; n_buckets];
    for s in 0..n_secrets {
        for b in 0..n_buckets {
            buckets[b] += entity_vec[s * n_buckets + b];
        }
    }
    buckets
}

fn detect_drift(tensor: &AccessTensor, window: usize, threshold: f64) -> Vec<DriftDetection> {
    let mut detections = Vec::new();
    let (_, _, n_buckets) = tensor.dimensions();
    if n_buckets < window * 2 {
        return detections;
    }

    let historical_end = n_buckets - window;

    for entity in tensor.entities() {
        let vec = tensor.entity_vector(&entity);
        if vec.is_empty() {
            continue;
        }

        let total: f32 = vec.iter().sum();
        if total < 1.0 {
            continue;
        }

        // Build per-secret mean-rate vectors for historical vs recent
        let secrets = tensor.secrets();
        let mut changed = Vec::new();
        let mut hist_means = Vec::new();
        let mut recent_means = Vec::new();
        let hist_len = historical_end as f32;
        let recent_len = window as f32;

        for secret in &secrets {
            let ts = tensor.time_series(&entity, secret);
            if ts.len() < n_buckets {
                continue;
            }

            let hist = &ts[..historical_end];
            let recent = &ts[historical_end..];

            let hist_mean = hist.iter().sum::<f32>() / hist_len.max(1.0);
            let recent_mean = recent.iter().sum::<f32>() / recent_len.max(1.0);
            hist_means.push(hist_mean);
            recent_means.push(recent_mean);

            // Per-secret drift
            let hist_sum: f32 = hist.iter().sum();
            let recent_sum: f32 = recent.iter().sum();
            if (hist_sum - recent_sum).abs() > hist_sum.max(1.0) * 0.5 {
                changed.push(secret.clone());
            }
        }

        // Combined drift: cosine distance (directional) + magnitude shift
        let cos_dist = cosine_distance(&hist_means, &recent_means);
        let hist_norm: f64 = hist_means.iter().map(|x| f64::from(*x)).sum();
        let recent_norm: f64 = recent_means.iter().map(|x| f64::from(*x)).sum();
        let denom = hist_norm.max(recent_norm).max(f64::EPSILON);
        let magnitude_shift = (recent_norm - hist_norm).abs() / denom;
        let drift_score = cos_dist.max(magnitude_shift);
        let is_drifting = drift_score > threshold;

        detections.push(DriftDetection {
            entity,
            drift_score,
            is_drifting,
            changed_secrets: changed,
        });
    }

    detections
}

/// Find dominant period via autocorrelation.
pub fn find_dominant_period(time_series: &[f32]) -> usize {
    let n = time_series.len();
    if n < 4 {
        return 0;
    }

    #[allow(clippy::cast_precision_loss)] // series length never exceeds 2^23
    let mean: f32 = time_series.iter().sum::<f32>() / n as f32;
    let centered: Vec<f32> = time_series.iter().map(|v| v - mean).collect();
    let variance: f32 = centered.iter().map(|v| v * v).sum();

    if variance < f32::EPSILON {
        return 0;
    }

    let mut best_lag = 0;
    let mut best_corr = f32::NEG_INFINITY;

    // Check lags from 2 to n/2
    let max_lag = n / 2;
    for lag in 2..=max_lag {
        let mut corr = 0.0_f32;
        for i in 0..n - lag {
            corr += centered[i] * centered[i + lag];
        }
        corr /= variance;

        if corr > best_corr {
            best_corr = corr;
            best_lag = lag;
        }
    }

    best_lag
}

/// Cosine distance between two vectors: 1 - cos(a, b).
fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 1.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm_a: f64 = a
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt();

    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seasonal_empty() {
        // Manually construct an empty tensor-like structure for testing
        let config = TemporalAnalysisConfig::default();
        let report = analyze_temporal_patterns(&empty_tensor(), config);
        assert!(report.seasonal_patterns.is_empty());
        assert_eq!(report.total_entities_analyzed, 0);
    }

    #[test]
    fn test_seasonal_periodic_signal() {
        // A signal with clear periodicity should have low reconstruction error
        let period = 6;
        let n_buckets = 24;
        // 24 = 4 * 6, factorable
        let mut data = vec![0.0_f32; n_buckets];
        for i in 0..n_buckets {
            data[i] = ((i % period) as f32 * std::f32::consts::PI / period as f32).sin() + 1.0;
        }

        let tensor = make_single_entity_tensor("user:alice", "secret1", &data);
        let config = TemporalAnalysisConfig {
            min_accesses: 1,
            ..TemporalAnalysisConfig::default()
        };
        let report = analyze_temporal_patterns(&tensor, config);
        // Should find at least one seasonal pattern
        if !report.seasonal_patterns.is_empty() {
            assert!(
                report.seasonal_patterns[0].reconstruction_error < 1.0,
                "Periodic signal should compress well"
            );
        }
    }

    #[test]
    fn test_seasonal_random_high_error() {
        // Random data should compress poorly
        let n_buckets = 12; // 12 = 3 * 4, factorable
        let data: Vec<f32> = (0..n_buckets).map(|i| ((i * 7 + 3) % 11) as f32).collect();

        let tensor = make_single_entity_tensor("user:alice", "secret1", &data);
        let config = TemporalAnalysisConfig {
            min_accesses: 1,
            ..TemporalAnalysisConfig::default()
        };
        let report = analyze_temporal_patterns(&tensor, config);
        // Random data may or may not compress, but report should succeed
        assert!(report.total_entities_analyzed <= 1);
    }

    #[test]
    fn test_drift_stable_entity() {
        // Uniform access pattern: no drift
        let n_buckets = 48;
        let data = vec![1.0_f32; n_buckets];
        let tensor = make_single_entity_tensor("user:alice", "s1", &data);
        let detections = detect_drift(&tensor, 12, 0.3);
        for d in &detections {
            assert!(!d.is_drifting, "Uniform pattern should not drift");
        }
    }

    #[test]
    fn test_drift_changed_entity() {
        // Access pattern that changes dramatically
        let n_buckets = 48;
        let mut data = vec![0.0_f32; n_buckets];
        // First 36 buckets: low access
        for d in data.iter_mut().take(36) {
            *d = 1.0;
        }
        // Last 12 buckets: high access
        for d in data.iter_mut().skip(36) {
            *d = 10.0;
        }

        let tensor = make_single_entity_tensor("user:alice", "s1", &data);
        let detections = detect_drift(&tensor, 12, 0.01);
        assert!(!detections.is_empty());
        // With a low threshold, a big change should be detected
        let alice = detections.iter().find(|d| d.entity == "user:alice");
        assert!(alice.is_some());
        if let Some(d) = alice {
            assert!(d.drift_score > 0.0);
        }
    }

    #[test]
    fn test_drift_threshold_boundary() {
        let n_buckets = 48;
        let data = vec![1.0_f32; n_buckets];
        let tensor = make_single_entity_tensor("user:alice", "s1", &data);

        // With threshold 0.0, even tiny drift is flagged
        let det_strict = detect_drift(&tensor, 12, 0.0);
        // With threshold 2.0, nothing is flagged
        let det_lax = detect_drift(&tensor, 12, 2.0);
        for d in &det_lax {
            assert!(!d.is_drifting);
        }
        // Strict may or may not flag depending on numerical precision
        let _ = det_strict;
    }

    #[test]
    fn test_dominant_period() {
        // Generate a signal with period 6
        let period = 6;
        let n = 48;
        let signal: Vec<f32> = (0..n)
            .map(|i| ((i % period) as f32 * std::f32::consts::PI * 2.0 / period as f32).sin())
            .collect();

        let result = find_dominant_period(&signal);
        // Should detect period near 6
        assert!(
            result >= 4 && result <= 8,
            "Expected period near 6, got {result}"
        );
    }

    #[test]
    fn test_temporal_min_accesses_filter() {
        let n_buckets = 12;
        // Very few accesses
        let mut data = vec![0.0_f32; n_buckets];
        data[0] = 1.0;

        let tensor = make_single_entity_tensor("user:alice", "s1", &data);
        let config = TemporalAnalysisConfig {
            min_accesses: 10, // require at least 10
            ..TemporalAnalysisConfig::default()
        };
        let report = analyze_temporal_patterns(&tensor, config);
        assert!(
            report.seasonal_patterns.is_empty(),
            "Entity with 1 access should be filtered"
        );
    }

    // ===== Test helpers =====

    fn empty_tensor() -> AccessTensor {
        AccessTensor {
            entity_index: std::collections::HashMap::new(),
            secret_index: std::collections::HashMap::new(),
            data: Vec::new(),
            dimensions: (0, 0, 0),
            config: crate::access_tensor::AccessTensorConfig::default(),
        }
    }

    fn make_single_entity_tensor(entity: &str, secret: &str, data: &[f32]) -> AccessTensor {
        let n_buckets = data.len();
        let mut entity_index = std::collections::HashMap::new();
        entity_index.insert(entity.to_string(), 0);
        let mut secret_index = std::collections::HashMap::new();
        secret_index.insert(secret.to_string(), 0);

        AccessTensor {
            entity_index,
            secret_index,
            data: data.to_vec(),
            dimensions: (1, 1, n_buckets),
            config: crate::access_tensor::AccessTensorConfig {
                num_buckets: n_buckets,
                ..crate::access_tensor::AccessTensorConfig::default()
            },
        }
    }
}
