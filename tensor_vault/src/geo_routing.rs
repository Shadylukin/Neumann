// SPDX-License-Identifier: MIT OR Apache-2.0
//! Latency-aware geometric routing for sync targets.
//!
//! `GeoRouter` scores sync targets by proximity, latency, and reliability,
//! and selects the top-k for each sync operation. Latency/failure stats
//! are updated via exponential moving average (EMA).

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::manifold::GeoCoordinate;

/// Health and location metadata for a sync target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetGeometry {
    /// Unique name identifying this sync target.
    pub target_name: String,
    /// Geographic coordinates of the target.
    pub location: GeoCoordinate,
    /// Exponentially weighted average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// Exponentially weighted average throughput.
    pub avg_throughput: f64,
    /// Exponentially weighted average failure rate (0.0 to 1.0).
    pub failure_rate: f64,
    /// Timestamp of the last health check in milliseconds since epoch.
    pub last_health_check_ms: i64,
}

/// Tuning knobs for geometric routing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Maximum acceptable average latency in milliseconds; targets above are excluded.
    pub max_latency_ms: f64,
    /// Maximum acceptable failure rate (0.0 to 1.0); targets above are excluded.
    pub max_failure_rate: f64,
    /// Weight for the latency component in composite scoring.
    pub latency_weight: f64,
    /// Weight for the geographic proximity component in composite scoring.
    pub proximity_weight: f64,
    /// Weight for the reliability component in composite scoring.
    pub reliability_weight: f64,
    /// Smoothing factor for exponential moving average updates (0.0 to 1.0).
    pub ema_alpha: f64,
    /// Maximum number of targets to select per sync operation.
    pub sync_fanout: usize,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 500.0,
            max_failure_rate: 0.1,
            latency_weight: 0.4,
            proximity_weight: 0.3,
            reliability_weight: 0.3,
            ema_alpha: 0.2,
            sync_fanout: 2,
        }
    }
}

/// A target selected for sync with its composite score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutedTarget {
    /// Name of the selected sync target.
    pub target_name: String,
    /// Composite routing score (higher is better).
    pub routing_score: f64,
    /// Expected latency in milliseconds based on EMA statistics.
    pub estimated_latency_ms: f64,
}

/// A target excluded from sync with the reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcludedTarget {
    /// Name of the excluded sync target.
    pub target_name: String,
    /// Why this target was excluded from routing.
    pub reason: ExclusionReason,
}

/// Why a target was excluded from the routing decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExclusionReason {
    /// Average latency exceeds the configured maximum.
    HighLatency {
        /// Observed average latency in milliseconds.
        latency_ms: f64,
        /// Configured maximum latency threshold.
        threshold: f64,
    },
    /// Failure rate exceeds the configured maximum.
    HighFailureRate {
        /// Observed failure rate (0.0 to 1.0).
        rate: f64,
        /// Configured maximum failure rate threshold.
        threshold: f64,
    },
    /// Target has no registered geometry or is otherwise unhealthy.
    Unhealthy,
}

/// Routing outcome for a single secret sync operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    /// Key of the secret being synced.
    pub secret_key: String,
    /// Targets selected for sync, ordered by descending score.
    pub selected_targets: Vec<RoutedTarget>,
    /// Targets excluded from sync with their reasons.
    pub excluded_targets: Vec<ExcludedTarget>,
}

/// Thread-safe geometric router for sync targets.
///
/// Maintains per-target geometry and stats, updates them via EMA on each
/// sync result, and selects the best targets for each sync operation.
pub struct GeoRouter {
    geometries: DashMap<String, TargetGeometry>,
    config: RoutingConfig,
}

impl GeoRouter {
    /// Create a new router with the given configuration.
    pub fn new(config: RoutingConfig) -> Self {
        Self {
            geometries: DashMap::new(),
            config,
        }
    }

    /// Register or update a target's geometry.
    pub fn update_geometry(&self, geometry: TargetGeometry) {
        self.geometries
            .insert(geometry.target_name.clone(), geometry);
    }

    /// Update latency and failure stats after a sync attempt via EMA.
    pub fn record_sync_result(&self, target_name: &str, latency_ms: f64, success: bool) {
        if let Some(mut geo) = self.geometries.get_mut(target_name) {
            let alpha = self.config.ema_alpha;
            geo.avg_latency_ms = alpha.mul_add(latency_ms, (1.0 - alpha) * geo.avg_latency_ms);

            let failure_sample = if success { 0.0 } else { 1.0 };
            geo.failure_rate = alpha.mul_add(failure_sample, (1.0 - alpha) * geo.failure_rate);
        }
    }

    /// Select the best targets for syncing a secret.
    ///
    /// Filters by latency/failure thresholds, scores remaining targets, and
    /// returns the top `sync_fanout` targets.
    pub fn route(
        &self,
        secret_key: &str,
        accessor_location: Option<&GeoCoordinate>,
        available_targets: &[String],
    ) -> RoutingDecision {
        let mut selected = Vec::new();
        let mut excluded = Vec::new();

        let mut candidates: Vec<(String, f64, f64)> = Vec::new(); // (name, score, latency)

        for target_name in available_targets {
            let Some(geo) = self.geometries.get(target_name) else {
                excluded.push(ExcludedTarget {
                    target_name: target_name.clone(),
                    reason: ExclusionReason::Unhealthy,
                });
                continue;
            };

            // Filter by thresholds
            if geo.avg_latency_ms > self.config.max_latency_ms {
                excluded.push(ExcludedTarget {
                    target_name: target_name.clone(),
                    reason: ExclusionReason::HighLatency {
                        latency_ms: geo.avg_latency_ms,
                        threshold: self.config.max_latency_ms,
                    },
                });
                continue;
            }

            if geo.failure_rate > self.config.max_failure_rate {
                excluded.push(ExcludedTarget {
                    target_name: target_name.clone(),
                    reason: ExclusionReason::HighFailureRate {
                        rate: geo.failure_rate,
                        threshold: self.config.max_failure_rate,
                    },
                });
                continue;
            }

            let score = self.compute_score(&geo, accessor_location);
            candidates.push((target_name.clone(), score, geo.avg_latency_ms));
        }

        // Sort by score descending (higher = better)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        for (name, score, latency) in candidates.into_iter().take(self.config.sync_fanout) {
            selected.push(RoutedTarget {
                target_name: name,
                routing_score: score,
                estimated_latency_ms: latency,
            });
        }

        RoutingDecision {
            secret_key: secret_key.to_string(),
            selected_targets: selected,
            excluded_targets: excluded,
        }
    }

    /// Return all registered geometries.
    pub fn geometries(&self) -> Vec<TargetGeometry> {
        self.geometries
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Composite score: higher is better.
    fn compute_score(
        &self,
        geo: &TargetGeometry,
        accessor_location: Option<&GeoCoordinate>,
    ) -> f64 {
        // Latency component: normalized inversely (lower latency = higher score)
        let norm_latency = if self.config.max_latency_ms > 0.0 {
            (geo.avg_latency_ms / self.config.max_latency_ms).min(1.0)
        } else {
            0.0
        };
        let latency_score = 1.0 - norm_latency;

        // Proximity component: distance from accessor to target
        let proximity_score = if let Some(loc) = accessor_location {
            let dist = geodesic_distance_2d(loc, &geo.location);
            // Normalize: assume 1000 as max reasonable distance
            let norm_dist = (dist / 1000.0).min(1.0);
            1.0 - norm_dist
        } else {
            0.5 // neutral when no location known
        };

        // Reliability component: lower failure rate = higher score
        let reliability_score = 1.0 - geo.failure_rate.min(1.0);

        self.config.reliability_weight.mul_add(
            reliability_score,
            self.config.latency_weight.mul_add(
                latency_score,
                self.config.proximity_weight * proximity_score,
            ),
        )
    }
}

/// Euclidean distance in 2D/3D (matching manifold.rs convention).
fn geodesic_distance_2d(a: &GeoCoordinate, b: &GeoCoordinate) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = match (a.z, b.z) {
        (Some(az), Some(bz)) => az - bz,
        _ => 0.0,
    };
    (dx.mul_add(dx, dy.mul_add(dy, dz * dz))).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo(name: &str, x: f64, y: f64, latency: f64, failure: f64) -> TargetGeometry {
        TargetGeometry {
            target_name: name.to_string(),
            location: GeoCoordinate { x, y, z: None },
            avg_latency_ms: latency,
            avg_throughput: 100.0,
            failure_rate: failure,
            last_health_check_ms: 0,
        }
    }

    #[test]
    fn test_router_register() {
        let router = GeoRouter::new(RoutingConfig::default());
        assert!(router.geometries().is_empty());

        router.update_geometry(geo("t1", 0.0, 0.0, 50.0, 0.01));
        assert_eq!(router.geometries().len(), 1);
        assert_eq!(router.geometries()[0].target_name, "t1");
    }

    #[test]
    fn test_router_single_target() {
        let router = GeoRouter::new(RoutingConfig::default());
        router.update_geometry(geo("t1", 0.0, 0.0, 50.0, 0.01));

        let decision = router.route("secret:a", None, &["t1".to_string()]);
        assert_eq!(decision.selected_targets.len(), 1);
        assert_eq!(decision.selected_targets[0].target_name, "t1");
        assert!(decision.excluded_targets.is_empty());
    }

    #[test]
    fn test_router_selects_closest() {
        let config = RoutingConfig {
            proximity_weight: 1.0,
            latency_weight: 0.0,
            reliability_weight: 0.0,
            sync_fanout: 1,
            ..RoutingConfig::default()
        };
        let router = GeoRouter::new(config);

        router.update_geometry(geo("close", 1.0, 1.0, 50.0, 0.0));
        router.update_geometry(geo("far", 100.0, 100.0, 50.0, 0.0));

        let loc = GeoCoordinate {
            x: 0.0,
            y: 0.0,
            z: None,
        };
        let decision = router.route("key", Some(&loc), &["close".to_string(), "far".to_string()]);
        assert_eq!(decision.selected_targets.len(), 1);
        assert_eq!(decision.selected_targets[0].target_name, "close");
    }

    #[test]
    fn test_router_excludes_latency() {
        let config = RoutingConfig {
            max_latency_ms: 100.0,
            ..RoutingConfig::default()
        };
        let router = GeoRouter::new(config);
        router.update_geometry(geo("slow", 0.0, 0.0, 200.0, 0.0));

        let decision = router.route("key", None, &["slow".to_string()]);
        assert!(decision.selected_targets.is_empty());
        assert_eq!(decision.excluded_targets.len(), 1);
        assert!(matches!(
            decision.excluded_targets[0].reason,
            ExclusionReason::HighLatency { .. }
        ));
    }

    #[test]
    fn test_router_excludes_failure() {
        let config = RoutingConfig {
            max_failure_rate: 0.05,
            ..RoutingConfig::default()
        };
        let router = GeoRouter::new(config);
        router.update_geometry(geo("flaky", 0.0, 0.0, 50.0, 0.2));

        let decision = router.route("key", None, &["flaky".to_string()]);
        assert!(decision.selected_targets.is_empty());
        assert_eq!(decision.excluded_targets.len(), 1);
        assert!(matches!(
            decision.excluded_targets[0].reason,
            ExclusionReason::HighFailureRate { .. }
        ));
    }

    #[test]
    fn test_router_ema_update() {
        let router = GeoRouter::new(RoutingConfig::default());
        router.update_geometry(geo("t1", 0.0, 0.0, 100.0, 0.0));

        // Record a sync with 200ms latency (failure)
        router.record_sync_result("t1", 200.0, false);

        let geos = router.geometries();
        let t1 = &geos[0];
        // EMA: new = 0.2 * 200 + 0.8 * 100 = 120
        assert!((t1.avg_latency_ms - 120.0).abs() < f64::EPSILON);
        // failure: 0.2 * 1.0 + 0.8 * 0.0 = 0.2
        assert!((t1.failure_rate - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_router_fanout_limit() {
        let config = RoutingConfig {
            sync_fanout: 2,
            ..RoutingConfig::default()
        };
        let router = GeoRouter::new(config);

        for i in 0..5 {
            #[allow(clippy::cast_precision_loss)]
            let x = i as f64 * 10.0;
            router.update_geometry(geo(&format!("t{i}"), x, 0.0, 50.0, 0.01));
        }

        let targets: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        let decision = router.route("key", None, &targets);
        assert_eq!(decision.selected_targets.len(), 2);
    }

    #[test]
    fn test_router_all_excluded() {
        let config = RoutingConfig {
            max_latency_ms: 10.0,
            ..RoutingConfig::default()
        };
        let router = GeoRouter::new(config);
        router.update_geometry(geo("a", 0.0, 0.0, 100.0, 0.0));
        router.update_geometry(geo("b", 0.0, 0.0, 200.0, 0.0));

        let decision = router.route("key", None, &["a".to_string(), "b".to_string()]);
        assert!(decision.selected_targets.is_empty());
        assert_eq!(decision.excluded_targets.len(), 2);
    }

    #[test]
    fn test_router_unknown_target_excluded() {
        let router = GeoRouter::new(RoutingConfig::default());
        let decision = router.route("key", None, &["unknown".to_string()]);
        assert!(decision.selected_targets.is_empty());
        assert_eq!(decision.excluded_targets.len(), 1);
        assert!(matches!(
            decision.excluded_targets[0].reason,
            ExclusionReason::Unhealthy
        ));
    }

    #[test]
    fn test_routing_config_default() {
        let config = RoutingConfig::default();
        assert!((config.max_latency_ms - 500.0).abs() < f64::EPSILON);
        assert!((config.max_failure_rate - 0.1).abs() < f64::EPSILON);
        assert!((config.ema_alpha - 0.2).abs() < f64::EPSILON);
        assert_eq!(config.sync_fanout, 2);
    }

    #[test]
    fn test_router_ema_success() {
        let router = GeoRouter::new(RoutingConfig::default());
        router.update_geometry(geo("t1", 0.0, 0.0, 100.0, 0.5));

        // Record a successful sync with 50ms
        router.record_sync_result("t1", 50.0, true);

        let geos = router.geometries();
        let t1 = &geos[0];
        // EMA latency: 0.2 * 50 + 0.8 * 100 = 90
        assert!((t1.avg_latency_ms - 90.0).abs() < f64::EPSILON);
        // EMA failure: 0.2 * 0.0 + 0.8 * 0.5 = 0.4
        assert!((t1.failure_rate - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_router_record_unknown_noop() {
        let router = GeoRouter::new(RoutingConfig::default());
        // Should not panic
        router.record_sync_result("nonexistent", 100.0, true);
        assert!(router.geometries().is_empty());
    }
}
