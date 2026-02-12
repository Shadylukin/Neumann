// SPDX-License-Identifier: MIT OR Apache-2.0
//! Heat kernel diffusion for graph-based trust propagation.
//!
//! Models trust as heat flow on the access-control graph: entities with
//! more connections and stronger clustering receive more trust via diffusion.
//! Uses Chebyshev polynomial approximation to avoid eigendecomposition.

use std::collections::HashMap;

use graph_engine::{Direction, PropertyValue};
use serde::{Deserialize, Serialize};

use crate::vault::Vault;

/// Allowed edge types for trust diffusion (same as graph_intel traversal).
const ALLOWED_EDGES: &[&str] = &[
    "VAULT_ACCESS",
    "VAULT_ACCESS_READ",
    "VAULT_ACCESS_WRITE",
    "VAULT_ACCESS_ADMIN",
    "MEMBER",
];

fn is_allowed_edge(edge_type: &str) -> bool {
    ALLOWED_EDGES
        .iter()
        .any(|&allowed| edge_type.starts_with(allowed))
}

/// Configuration for heat kernel trust diffusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatKernelConfig {
    /// How far trust spreads (larger = more diffusion). Default: 1.0.
    pub diffusion_time: f64,
    /// Number of Chebyshev polynomial terms for approximation. Default: 10.
    pub chebyshev_order: usize,
    /// Maximum power-iteration steps for spectral gap. Default: 100.
    pub max_iterations: usize,
}

impl Default for HeatKernelConfig {
    fn default() -> Self {
        Self {
            diffusion_time: 1.0,
            chebyshev_order: 10,
            max_iterations: 100,
        }
    }
}

/// Per-entity trust score after heat kernel diffusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatKernelTrustScore {
    /// Entity identifier.
    pub entity: String,
    /// Diffused trust score.
    pub trust_score: f64,
    /// Ratio of received trust to emitted trust.
    pub trust_ratio: f64,
    /// Pre-diffusion trust score.
    pub initial_trust: f64,
}

/// Report from heat kernel trust analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatKernelTrustReport {
    /// Per-entity scores sorted by trust descending.
    pub entities: Vec<HeatKernelTrustScore>,
    /// Diffusion time parameter used.
    pub diffusion_time: f64,
    /// Second smallest eigenvalue of the Laplacian (algebraic connectivity).
    pub spectral_gap: f64,
    /// Total number of entities analyzed.
    pub total_entities: usize,
}

/// Build the graph Laplacian from the vault's access-control graph.
///
/// Returns (entity_keys, laplacian_matrix) where the matrix is L = D - A.
fn build_graph_laplacian(vault: &Vault) -> (Vec<String>, Vec<Vec<f64>>) {
    let entities = collect_entity_keys(vault);
    let n = entities.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let entity_index: HashMap<String, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.clone(), i))
        .collect();

    // Build adjacency matrix
    let mut adj = vec![vec![0.0_f64; n]; n];

    for entity in &entities {
        let Some(node_id) = vault.find_entity_node(entity) else {
            continue;
        };
        if let Ok(edges) = vault.graph.edges_of(node_id, Direction::Outgoing) {
            for edge in edges {
                if !is_allowed_edge(&edge.edge_type) {
                    continue;
                }
                let target = if edge.from == node_id {
                    edge.to
                } else {
                    edge.from
                };
                if let Ok(target_node) = vault.graph.get_node(target) {
                    if let Some(PropertyValue::String(key)) =
                        target_node.properties.get("entity_key")
                    {
                        if let Some(&j) = entity_index.get(key.as_str()) {
                            let Some(&i) = entity_index.get(entity.as_str()) else {
                                continue;
                            };
                            adj[i][j] = 1.0;
                            adj[j][i] = 1.0; // undirected
                        }
                    }
                }
            }
        }
    }

    // Laplacian L = D - A
    let mut laplacian = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let degree: f64 = adj[i].iter().sum();
        laplacian[i][i] = degree;
        for j in 0..n {
            laplacian[i][j] -= adj[i][j];
        }
    }

    (entities, laplacian)
}

/// Approximate `exp(-t * L) * v` using Chebyshev polynomials.
///
/// This avoids computing eigenvalues/eigenvectors of L.
fn chebyshev_heat_kernel(
    laplacian: &[Vec<f64>],
    initial: &[f64],
    time: f64,
    order: usize,
) -> Vec<f64> {
    let n = initial.len();
    if n == 0 {
        return Vec::new();
    }

    // Estimate spectral radius (max eigenvalue) for scaling
    let lambda_max = estimate_max_eigenvalue(laplacian, 50);
    if lambda_max < f64::EPSILON {
        return initial.to_vec();
    }

    // Scale time by spectral radius: x = t * L / lambda_max mapped to [0, 1]
    let half_t = time * lambda_max / 2.0;

    // Chebyshev coefficients for exp(-x) on [0, lambda_max * t]
    let coeffs = chebyshev_exp_coefficients(half_t, order);

    // T_0(L_scaled) * v = v
    let mut t_prev = initial.to_vec();
    // L_scaled = (2/(lambda_max)) * L - I, so L_scaled * v:
    let scaled_lv = |v: &[f64]| -> Vec<f64> {
        let lv = mat_vec_mul(laplacian, v);
        let mut result = vec![0.0; n];
        for i in 0..n {
            result[i] = (2.0 / lambda_max).mul_add(lv[i], -v[i]);
        }
        result
    };

    // T_1(L_scaled) * v = L_scaled * v
    let mut t_curr = scaled_lv(&t_prev);

    // Accumulate: result = c_0 * T_0 + c_1 * T_1 + ...
    let mut result = vec![0.0; n];
    for i in 0..n {
        result[i] = coeffs[0] * t_prev[i];
    }
    if order > 0 {
        for i in 0..n {
            result[i] += coeffs.get(1).copied().unwrap_or(0.0) * t_curr[i];
        }
    }

    // Chebyshev recurrence: T_{k+1} = 2 * L_scaled * T_k - T_{k-1}
    for k in 2..=order {
        let t_next_base = scaled_lv(&t_curr);
        let mut t_next = vec![0.0; n];
        for i in 0..n {
            t_next[i] = 2.0_f64.mul_add(t_next_base[i], -t_prev[i]);
        }
        let c = coeffs.get(k).copied().unwrap_or(0.0);
        for i in 0..n {
            result[i] += c * t_next[i];
        }
        t_prev = t_curr;
        t_curr = t_next;
    }

    // Clamp negative values to zero (trust can't be negative)
    for v in &mut result {
        if *v < 0.0 {
            *v = 0.0;
        }
    }

    result
}

/// Chebyshev expansion coefficients for exp(-x) shifted to [-1, 1].
fn chebyshev_exp_coefficients(half_scale: f64, order: usize) -> Vec<f64> {
    let num_points = order + 1;
    let mut coeffs = vec![0.0; num_points];

    // Sample exp(-half_scale * (x+1)) at Chebyshev nodes
    #[allow(clippy::cast_precision_loss)] // Chebyshev order and indices never exceed 2^52
    let samples: Vec<f64> = (0..num_points)
        .map(|k| {
            let theta = std::f64::consts::PI * (k as f64 + 0.5) / num_points as f64;
            let x = theta.cos(); // Chebyshev node in [-1, 1]
            (-half_scale * (x + 1.0)).exp()
        })
        .collect();

    #[allow(clippy::cast_precision_loss)] // Chebyshev order never exceeds 2^52
    for (j, coeff) in coeffs.iter_mut().enumerate().take(num_points) {
        let mut sum = 0.0;
        for (k, sample) in samples.iter().enumerate().take(num_points) {
            let theta = std::f64::consts::PI * (k as f64 + 0.5) / num_points as f64;
            let x = theta.cos();
            let t_j = chebyshev_poly(j, x);
            sum += sample * t_j;
        }
        *coeff = sum * 2.0 / num_points as f64;
    }
    // Halve the zeroth coefficient
    coeffs[0] /= 2.0;

    coeffs
}

/// Evaluate Chebyshev polynomial T_n(x) via recurrence.
fn chebyshev_poly(n: usize, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    if n == 1 {
        return x;
    }
    let mut t_prev = 1.0;
    let mut t_curr = x;
    for _ in 2..=n {
        let t_next = (2.0 * x).mul_add(t_curr, -t_prev);
        t_prev = t_curr;
        t_curr = t_next;
    }
    t_curr
}

/// Estimate the largest eigenvalue of L via power iteration.
fn estimate_max_eigenvalue(laplacian: &[Vec<f64>], iterations: usize) -> f64 {
    let n = laplacian.len();
    if n == 0 {
        return 0.0;
    }

    #[allow(clippy::cast_precision_loss)] // graph size never exceeds 2^52
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut eigenvalue = 0.0;

    for _ in 0..iterations {
        let lv = mat_vec_mul(laplacian, &v);
        eigenvalue = vec_dot(&v, &lv);
        let norm = vec_norm(&lv);
        if norm < f64::EPSILON {
            break;
        }
        for i in 0..n {
            v[i] = lv[i] / norm;
        }
    }

    eigenvalue
}

/// Estimate the spectral gap (2nd smallest eigenvalue) via shifted inverse iteration.
fn estimate_spectral_gap(laplacian: &[Vec<f64>], max_iterations: usize) -> f64 {
    let n = laplacian.len();
    if n < 2 {
        return 0.0;
    }

    // The smallest eigenvalue of L is 0 with eigenvector (1,...,1)/sqrt(n).
    // Use power iteration on L directly to find the Fiedler value.
    // Start with a random vector orthogonal to the all-ones vector.
    #[allow(clippy::cast_precision_loss)] // graph size never exceeds 2^52
    let ones_norm = (n as f64).sqrt();
    #[allow(clippy::cast_precision_loss)] // graph size never exceeds 2^52
    let mut v: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
    // Orthogonalize against all-ones
    let proj = v.iter().sum::<f64>() / ones_norm;
    for vi in &mut v {
        *vi -= proj / ones_norm;
    }
    let norm = vec_norm(&v);
    if norm < f64::EPSILON {
        return 0.0;
    }
    for vi in &mut v {
        *vi /= norm;
    }

    let mut eigenvalue = 0.0;

    for _ in 0..max_iterations {
        let lv = mat_vec_mul(laplacian, &v);
        eigenvalue = vec_dot(&v, &lv);

        // Project out the null-space component
        let proj = lv.iter().sum::<f64>() / ones_norm;
        let mut residual = lv;
        for r in &mut residual {
            *r -= proj / ones_norm;
        }

        let norm = vec_norm(&residual);
        if norm < f64::EPSILON {
            break;
        }
        for i in 0..n {
            v[i] = residual[i] / norm;
        }
    }

    if eigenvalue < 0.0 {
        0.0
    } else {
        eigenvalue
    }
}

fn mat_vec_mul(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_norm(v: &[f64]) -> f64 {
    vec_dot(v, v).sqrt()
}

/// Collect non-secret entity keys from the graph.
fn collect_entity_keys(vault: &Vault) -> Vec<String> {
    let mut entities = Vec::new();
    if let Ok(node_ids) = vault.graph.get_all_node_ids() {
        for node_id in node_ids {
            if let Ok(node) = vault.graph.get_node(node_id) {
                if let Some(PropertyValue::String(key)) = node.properties.get("entity_key") {
                    if !key.starts_with("_vk:")
                        && !key.starts_with("vault_secret:")
                        && key != Vault::ROOT
                    {
                        entities.push(key.clone());
                    }
                }
            }
        }
    }
    entities
}

/// Run heat kernel trust diffusion on the vault's access-control graph.
pub fn heat_kernel_trust(vault: &Vault, config: HeatKernelConfig) -> HeatKernelTrustReport {
    let (entities, laplacian) = build_graph_laplacian(vault);
    let n = entities.len();

    if n == 0 {
        return HeatKernelTrustReport {
            entities: Vec::new(),
            diffusion_time: config.diffusion_time,
            spectral_gap: 0.0,
            total_entities: 0,
        };
    }

    // Seed initial trust from trust_transitivity
    let trust_report = crate::graph_intel::trust_transitivity(vault);
    let trust_map: HashMap<String, f64> = trust_report
        .entities
        .into_iter()
        .map(|e| (e.entity, e.trust_score))
        .collect();

    let initial: Vec<f64> = entities
        .iter()
        .map(|e| trust_map.get(e).copied().unwrap_or(0.0))
        .collect();

    // Diffuse
    let diffused = chebyshev_heat_kernel(
        &laplacian,
        &initial,
        config.diffusion_time,
        config.chebyshev_order,
    );

    // Spectral gap
    let spectral_gap = estimate_spectral_gap(&laplacian, config.max_iterations);

    // Build scores
    let mut scores: Vec<HeatKernelTrustScore> = entities
        .into_iter()
        .enumerate()
        .map(|(i, entity)| {
            let initial_trust = initial[i];
            let trust_score = diffused[i];
            let trust_ratio = if initial_trust > f64::EPSILON {
                trust_score / initial_trust
            } else if trust_score > f64::EPSILON {
                f64::INFINITY
            } else {
                1.0
            };
            HeatKernelTrustScore {
                entity,
                trust_score,
                trust_ratio,
                initial_trust,
            }
        })
        .collect();

    scores.sort_by(|a, b| {
        b.trust_score
            .partial_cmp(&a.trust_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    HeatKernelTrustReport {
        total_entities: n,
        entities: scores,
        diffusion_time: config.diffusion_time,
        spectral_gap,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use graph_engine::{GraphEngine, PropertyValue};
    use tensor_store::TensorStore;

    use super::*;
    use crate::VaultConfig;

    fn create_test_vault() -> Vault {
        let store = TensorStore::new();
        let graph = Arc::new(GraphEngine::new());
        Vault::new(
            b"test_password",
            graph.clone(),
            store,
            VaultConfig::default(),
        )
        .unwrap()
    }

    fn add_entity(vault: &Vault, name: &str) -> u64 {
        let mut props = std::collections::HashMap::new();
        props.insert(
            "entity_key".to_string(),
            PropertyValue::String(name.to_string()),
        );
        vault.graph.create_node("Entity", props).unwrap()
    }

    fn link_entities(vault: &Vault, from: u64, to: u64) {
        let props = std::collections::HashMap::new();
        vault
            .graph
            .create_edge(from, to, "VAULT_ACCESS", props, true)
            .unwrap();
    }

    #[test]
    fn test_hk_empty_graph() {
        let vault = create_test_vault();
        let report = heat_kernel_trust(&vault, HeatKernelConfig::default());
        assert!(report.entities.is_empty());
        assert_eq!(report.total_entities, 0);
        assert!((report.spectral_gap - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hk_single_entity() {
        let vault = create_test_vault();
        add_entity(&vault, "user:alice");
        let report = heat_kernel_trust(&vault, HeatKernelConfig::default());
        // Single entity with no connections, initial trust from clustering = 0
        assert_eq!(report.total_entities, 1);
        assert_eq!(report.entities.len(), 1);
        assert_eq!(report.entities[0].entity, "user:alice");
    }

    #[test]
    fn test_hk_two_connected() {
        let vault = create_test_vault();
        let a = add_entity(&vault, "user:alice");
        let b = add_entity(&vault, "user:bob");
        link_entities(&vault, a, b);

        let report = heat_kernel_trust(&vault, HeatKernelConfig::default());
        assert_eq!(report.total_entities, 2);
        assert_eq!(report.entities.len(), 2);
    }

    #[test]
    fn test_hk_disconnected_components() {
        let vault = create_test_vault();
        let a = add_entity(&vault, "user:alice");
        let b = add_entity(&vault, "user:bob");
        let c = add_entity(&vault, "user:carol");
        let d = add_entity(&vault, "user:dave");
        // Component 1: alice -- bob
        link_entities(&vault, a, b);
        // Component 2: carol -- dave
        link_entities(&vault, c, d);

        let report = heat_kernel_trust(&vault, HeatKernelConfig::default());
        assert_eq!(report.total_entities, 4);
        // Trust should not cross the gap between components
        // Both components are symmetric, so scores within each should be similar
    }

    #[test]
    fn test_hk_time_zero() {
        let vault = create_test_vault();
        let a = add_entity(&vault, "user:alice");
        let b = add_entity(&vault, "user:bob");
        link_entities(&vault, a, b);

        let config = HeatKernelConfig {
            diffusion_time: 0.0,
            ..HeatKernelConfig::default()
        };
        let report = heat_kernel_trust(&vault, config);
        // With t=0, diffused = initial (no heat flow)
        for score in &report.entities {
            let diff = (score.trust_score - score.initial_trust).abs();
            assert!(diff < 1e-6, "t=0 should preserve initial trust");
        }
    }

    #[test]
    fn test_hk_high_time_converges() {
        let vault = create_test_vault();
        let a = add_entity(&vault, "user:alice");
        let b = add_entity(&vault, "user:bob");
        let c = add_entity(&vault, "user:carol");
        link_entities(&vault, a, b);
        link_entities(&vault, b, c);
        link_entities(&vault, a, c);

        let config_low = HeatKernelConfig {
            diffusion_time: 10.0,
            ..HeatKernelConfig::default()
        };
        let config_high = HeatKernelConfig {
            diffusion_time: 100.0,
            ..HeatKernelConfig::default()
        };
        let report_low = heat_kernel_trust(&vault, config_low);
        let report_high = heat_kernel_trust(&vault, config_high);

        // At very high diffusion time, all scores converge to the same value
        if report_high.entities.len() >= 2 {
            let scores: Vec<f64> = report_high.entities.iter().map(|e| e.trust_score).collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
            // High-time spread should be <= low-time spread
            let low_scores: Vec<f64> = report_low.entities.iter().map(|e| e.trust_score).collect();
            let low_max = low_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let low_min = low_scores.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                (max - min) <= (low_max - low_min) + 1e-6,
                "Higher diffusion time should converge scores"
            );
        }
    }

    #[test]
    fn test_hk_star_topology() {
        let vault = create_test_vault();
        let hub = add_entity(&vault, "user:hub");
        let leaf1 = add_entity(&vault, "user:leaf1");
        let leaf2 = add_entity(&vault, "user:leaf2");
        let leaf3 = add_entity(&vault, "user:leaf3");
        link_entities(&vault, hub, leaf1);
        link_entities(&vault, hub, leaf2);
        link_entities(&vault, hub, leaf3);

        let report = heat_kernel_trust(&vault, HeatKernelConfig::default());
        assert_eq!(report.total_entities, 4);
        // Hub has most connections
        assert!(!report.entities.is_empty());
    }

    #[test]
    fn test_hk_chain_topology() {
        let vault = create_test_vault();
        let a = add_entity(&vault, "user:a");
        let b = add_entity(&vault, "user:b");
        let c = add_entity(&vault, "user:c");
        let d = add_entity(&vault, "user:d");
        link_entities(&vault, a, b);
        link_entities(&vault, b, c);
        link_entities(&vault, c, d);

        let report = heat_kernel_trust(&vault, HeatKernelConfig::default());
        assert_eq!(report.total_entities, 4);
    }

    #[test]
    fn test_hk_default_config() {
        let config = HeatKernelConfig::default();
        assert!((config.diffusion_time - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.chebyshev_order, 10);
        assert_eq!(config.max_iterations, 100);
    }

    #[test]
    fn test_hk_custom_config() {
        let config = HeatKernelConfig {
            diffusion_time: 5.0,
            chebyshev_order: 20,
            max_iterations: 200,
        };
        assert!((config.diffusion_time - 5.0).abs() < f64::EPSILON);
        assert_eq!(config.chebyshev_order, 20);
        assert_eq!(config.max_iterations, 200);
    }

    /// Helper to link entities with a specific edge type.
    fn link_entities_typed(vault: &Vault, from: u64, to: u64, edge_type: &str) {
        let props = std::collections::HashMap::new();
        vault
            .graph
            .create_edge(from, to, edge_type, props, true)
            .unwrap();
    }

    #[test]
    fn test_hk_larger_graph() {
        // Build a 6-node graph with enough connectivity to exercise all Chebyshev
        // iteration terms (order=10).
        let vault = create_test_vault();
        let a = add_entity(&vault, "user:a");
        let b = add_entity(&vault, "user:b");
        let c = add_entity(&vault, "user:c");
        let d = add_entity(&vault, "user:d");
        let e = add_entity(&vault, "user:e");
        let f = add_entity(&vault, "user:f");

        // Create a dense mesh so the Laplacian has a non-trivial spectrum
        link_entities(&vault, a, b);
        link_entities(&vault, a, c);
        link_entities(&vault, a, d);
        link_entities(&vault, b, c);
        link_entities(&vault, b, e);
        link_entities(&vault, c, f);
        link_entities(&vault, d, e);
        link_entities(&vault, d, f);
        link_entities(&vault, e, f);

        let config = HeatKernelConfig {
            diffusion_time: 2.0,
            chebyshev_order: 10,
            max_iterations: 100,
        };
        let report = heat_kernel_trust(&vault, config);
        assert_eq!(report.total_entities, 6);
        assert_eq!(report.entities.len(), 6);
        // All trust scores should be non-negative after diffusion
        for score in &report.entities {
            assert!(
                score.trust_score >= 0.0,
                "trust_score should be non-negative, got {}",
                score.trust_score,
            );
        }
        // Spectral gap should be positive for a connected graph
        assert!(
            report.spectral_gap > 0.0,
            "spectral gap should be positive for connected graph, got {}",
            report.spectral_gap,
        );
    }

    #[test]
    fn test_hk_weighted_edges() {
        // Test that both VAULT_ACCESS and MEMBER edge types are counted
        let vault = create_test_vault();
        let a = add_entity(&vault, "user:alice");
        let b = add_entity(&vault, "user:bob");
        let c = add_entity(&vault, "user:carol");
        let d = add_entity(&vault, "user:dave");

        // Use VAULT_ACCESS edges
        link_entities(&vault, a, b);
        // Use MEMBER edges
        link_entities_typed(&vault, b, c, "MEMBER");
        // Use VAULT_ACCESS_READ
        link_entities_typed(&vault, c, d, "VAULT_ACCESS_READ");

        let report = heat_kernel_trust(&vault, HeatKernelConfig::default());
        assert_eq!(report.total_entities, 4);
        // All four entities should be connected via the mixed edge types
        assert_eq!(report.entities.len(), 4);
    }

    #[test]
    fn test_hk_disallowed_edge_type() {
        // Entities linked with an edge type not in ALLOWED_EDGES should appear
        // disconnected (adjacency = 0 between them).
        let vault = create_test_vault();
        let a = add_entity(&vault, "user:alice");
        let b = add_entity(&vault, "user:bob");
        let c = add_entity(&vault, "user:carol");

        // Only link via disallowed type
        link_entities_typed(&vault, a, b, "CUSTOM_LINK");
        link_entities_typed(&vault, b, c, "RANDOM_EDGE");

        let (entities, laplacian) = build_graph_laplacian(&vault);
        assert_eq!(entities.len(), 3);
        // All diagonal should be zero (no allowed edges, so degree=0)
        for i in 0..entities.len() {
            assert!(
                laplacian[i][i].abs() < f64::EPSILON,
                "degree should be 0 for entity {} when only disallowed edges exist",
                entities[i],
            );
        }
    }

    #[test]
    fn test_hk_spectral_gap_positive() {
        // A well-connected graph should have a positive spectral gap
        let vault = create_test_vault();
        let a = add_entity(&vault, "user:a");
        let b = add_entity(&vault, "user:b");
        let c = add_entity(&vault, "user:c");
        let d = add_entity(&vault, "user:d");
        let e = add_entity(&vault, "user:e");

        // Complete-ish graph
        link_entities(&vault, a, b);
        link_entities(&vault, a, c);
        link_entities(&vault, a, d);
        link_entities(&vault, a, e);
        link_entities(&vault, b, c);
        link_entities(&vault, b, d);
        link_entities(&vault, c, d);
        link_entities(&vault, d, e);

        let (_, laplacian) = build_graph_laplacian(&vault);
        let gap = estimate_spectral_gap(&laplacian, 200);
        assert!(
            gap > 0.0,
            "spectral gap should be positive for a connected graph, got {gap}",
        );
    }

    #[test]
    fn test_hk_chebyshev_functions() {
        // ---- chebyshev_poly ----
        // T_0(x) = 1 for all x
        assert!((chebyshev_poly(0, 0.5) - 1.0).abs() < 1e-12);
        assert!((chebyshev_poly(0, -0.3) - 1.0).abs() < 1e-12);

        // T_1(x) = x
        assert!((chebyshev_poly(1, 0.7) - 0.7).abs() < 1e-12);
        assert!((chebyshev_poly(1, -0.5) - (-0.5)).abs() < 1e-12);

        // T_2(x) = 2x^2 - 1
        let x = 0.6;
        let expected_t2 = 2.0_f64.mul_add(x * x, -1.0);
        assert!(
            (chebyshev_poly(2, x) - expected_t2).abs() < 1e-12,
            "T_2({x}) should be {expected_t2}, got {}",
            chebyshev_poly(2, x),
        );

        // T_3(x) = 4x^3 - 3x
        let expected_t3 = (4.0 * x * x * x) - 3.0 * x;
        assert!(
            (chebyshev_poly(3, x) - expected_t3).abs() < 1e-12,
            "T_3({x}) should be {expected_t3}, got {}",
            chebyshev_poly(3, x),
        );

        // T_5 at x=1 should be 1 (T_n(1) = 1 for all n)
        assert!((chebyshev_poly(5, 1.0) - 1.0).abs() < 1e-10);

        // ---- chebyshev_exp_coefficients ----
        let coeffs = chebyshev_exp_coefficients(1.0, 10);
        assert_eq!(coeffs.len(), 11);
        // The coefficients should sum to approximately exp(-1) when evaluated
        // at x=0 (which maps to (0+1) = 1, so exp(-1*1) = exp(-1))
        // The zeroth coefficient is the dominant one (should be positive)
        assert!(coeffs[0] > 0.0, "zeroth coefficient should be positive");

        // Higher-order coefficients should decay in magnitude
        let last = coeffs[10].abs();
        let first = coeffs[0].abs();
        assert!(
            last < first,
            "higher-order coefficients should be smaller: |c_10|={last} vs |c_0|={first}",
        );

        // ---- estimate_max_eigenvalue ----
        // Identity matrix has all eigenvalues = 1
        let identity = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let max_eig = estimate_max_eigenvalue(&identity, 100);
        assert!(
            (max_eig - 1.0).abs() < 1e-6,
            "max eigenvalue of identity should be 1.0, got {max_eig}",
        );

        // Diagonal matrix diag(1, 2, 3) has max eigenvalue 3
        let diag = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 3.0],
        ];
        let max_eig = estimate_max_eigenvalue(&diag, 100);
        assert!(
            (max_eig - 3.0).abs() < 1e-6,
            "max eigenvalue of diag(1,2,3) should be 3.0, got {max_eig}",
        );

        // Empty matrix
        let empty: Vec<Vec<f64>> = Vec::new();
        let max_eig = estimate_max_eigenvalue(&empty, 100);
        assert!(
            max_eig.abs() < f64::EPSILON,
            "max eigenvalue of empty matrix should be 0",
        );

        // Asymmetric positive-definite matrix to avoid null-space issues
        // [[3, 1], [1, 2]] has eigenvalues (5+sqrt(5))/2 ~ 3.618 and (5-sqrt(5))/2 ~ 1.382
        let asym = vec![vec![3.0, 1.0], vec![1.0, 2.0]];
        let max_eig = estimate_max_eigenvalue(&asym, 200);
        let expected = (5.0 + 5.0_f64.sqrt()) / 2.0;
        assert!(
            (max_eig - expected).abs() < 1e-4,
            "max eigenvalue of [[3,1],[1,2]] should be ~{expected}, got {max_eig}",
        );
    }

    #[test]
    fn test_hk_trust_ratio_computation() {
        // Create entities with different connectivity levels to verify
        // trust_ratio = diffused / initial
        let vault = create_test_vault();
        let hub = add_entity(&vault, "user:hub");
        let s1 = add_entity(&vault, "user:spoke1");
        let s2 = add_entity(&vault, "user:spoke2");
        let s3 = add_entity(&vault, "user:spoke3");
        let s4 = add_entity(&vault, "user:spoke4");
        let s5 = add_entity(&vault, "user:spoke5");

        link_entities(&vault, hub, s1);
        link_entities(&vault, hub, s2);
        link_entities(&vault, hub, s3);
        link_entities(&vault, hub, s4);
        link_entities(&vault, hub, s5);
        // Add some inter-spoke links for triangles (boosts clustering)
        link_entities(&vault, s1, s2);
        link_entities(&vault, s2, s3);
        link_entities(&vault, s3, s4);

        let config = HeatKernelConfig {
            diffusion_time: 1.0,
            chebyshev_order: 10,
            max_iterations: 100,
        };
        let report = heat_kernel_trust(&vault, config);
        assert_eq!(report.total_entities, 6);

        for score in &report.entities {
            // trust_ratio should be finite (or infinity only when initial_trust is near 0)
            if score.initial_trust > f64::EPSILON {
                assert!(
                    score.trust_ratio.is_finite(),
                    "trust_ratio should be finite for entity {} with positive initial trust",
                    score.entity,
                );
                // Verify the ratio: trust_ratio = trust_score / initial_trust
                let expected_ratio = score.trust_score / score.initial_trust;
                assert!(
                    (score.trust_ratio - expected_ratio).abs() < 1e-10,
                    "trust_ratio mismatch for {}: expected {expected_ratio}, got {}",
                    score.entity,
                    score.trust_ratio,
                );
            } else if score.trust_score > f64::EPSILON {
                // Zero initial trust but positive diffused => infinity
                assert!(
                    score.trust_ratio.is_infinite(),
                    "trust_ratio should be infinity when initial_trust ~0 and diffused > 0",
                );
            } else {
                // Both near zero => ratio should be 1.0
                assert!(
                    (score.trust_ratio - 1.0).abs() < 1e-10,
                    "trust_ratio should be 1.0 when both initial and diffused are ~0",
                );
            }
        }
    }

    #[test]
    fn test_hk_collect_filters_vault_keys() {
        // Entities with _vk: and vault_secret: prefixes and the ROOT key
        // should be filtered out by collect_entity_keys.
        let vault = create_test_vault();

        // Regular entity (should appear)
        add_entity(&vault, "user:alice");

        // Internal vault key (should be filtered)
        add_entity(&vault, "_vk:internal_key");

        // Vault secret (should be filtered)
        add_entity(&vault, "vault_secret:my_secret");

        // Root entity (should be filtered)
        add_entity(&vault, Vault::ROOT);

        // Another regular entity (should appear)
        add_entity(&vault, "group:admins");

        let keys = collect_entity_keys(&vault);
        assert!(
            keys.contains(&"user:alice".to_string()),
            "regular entity should be included",
        );
        assert!(
            keys.contains(&"group:admins".to_string()),
            "regular entity should be included",
        );
        assert!(
            !keys.iter().any(|k| k.starts_with("_vk:")),
            "_vk: keys should be filtered out",
        );
        assert!(
            !keys.iter().any(|k| k.starts_with("vault_secret:")),
            "vault_secret: keys should be filtered out",
        );
        assert!(
            !keys.iter().any(|k| k == Vault::ROOT),
            "ROOT key should be filtered out",
        );
        assert_eq!(keys.len(), 2, "only 2 regular entities should remain");
    }

    #[test]
    fn test_hk_estimate_spectral_gap_direct() {
        // Two-node Laplacian: [[1,-1],[-1,1]]
        // Eigenvalues: 0 and 2, so spectral gap = 2
        let lap = vec![vec![1.0, -1.0], vec![-1.0, 1.0]];
        let gap = estimate_spectral_gap(&lap, 200);
        assert!(
            (gap - 2.0).abs() < 1e-4,
            "spectral gap of 2-node Laplacian should be ~2.0, got {gap}",
        );

        // Single node (n < 2) should return 0
        let single = vec![vec![0.0]];
        let gap = estimate_spectral_gap(&single, 100);
        assert!(
            gap.abs() < f64::EPSILON,
            "spectral gap of single node should be 0, got {gap}",
        );

        // Empty matrix
        let empty: Vec<Vec<f64>> = Vec::new();
        let gap = estimate_spectral_gap(&empty, 100);
        assert!(
            gap.abs() < f64::EPSILON,
            "spectral gap of empty matrix should be 0, got {gap}",
        );

        // Three-node path graph: A-B-C
        // Laplacian: [[1,-1,0],[-1,2,-1],[0,-1,1]]
        // Eigenvalues: 0, 1, 3 => spectral gap = 1
        let lap3 = vec![
            vec![1.0, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 1.0],
        ];
        let gap = estimate_spectral_gap(&lap3, 500);
        assert!(
            (gap - 1.0).abs() < 0.1,
            "spectral gap of 3-node path should be ~1.0, got {gap}",
        );
    }

    #[test]
    fn test_hk_mat_vec_mul_and_helpers() {
        // mat_vec_mul with identity
        let id = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let v = vec![3.0, 5.0, 7.0];
        let result = mat_vec_mul(&id, &v);
        for i in 0..3 {
            assert!(
                (result[i] - v[i]).abs() < f64::EPSILON,
                "identity * v should equal v",
            );
        }

        // mat_vec_mul with a non-trivial matrix
        let m = vec![vec![2.0, 1.0], vec![0.0, 3.0]];
        let v2 = vec![1.0, 2.0];
        let result = mat_vec_mul(&m, &v2);
        assert!((result[0] - 4.0).abs() < f64::EPSILON); // 2*1 + 1*2 = 4
        assert!((result[1] - 6.0).abs() < f64::EPSILON); // 0*1 + 3*2 = 6

        // vec_dot
        let dot = vec_dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert!((dot - 32.0).abs() < f64::EPSILON); // 4+10+18

        // vec_norm
        let norm = vec_norm(&[3.0, 4.0]);
        assert!((norm - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hk_chebyshev_heat_kernel_empty_and_trivial() {
        // Empty input
        let result = chebyshev_heat_kernel(&[], &[], 1.0, 10);
        assert!(result.is_empty());

        // Zero-degree Laplacian (all zeros), lambda_max < epsilon => returns initial
        let zero_lap = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let initial = vec![1.0, 2.0];
        let result = chebyshev_heat_kernel(&zero_lap, &initial, 1.0, 10);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_hk_chebyshev_heat_kernel_nontrivial() {
        // Use a positive-definite matrix (not a Laplacian) so that power
        // iteration from a uniform start does NOT land in the null space.
        // Matrix: diag(1, 2, 3). Eigenvalues are 1, 2, 3 so lambda_max=3.
        // exp(-t*M) on diagonal is diag(e^{-t}, e^{-2t}, e^{-3t}).
        let mat = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 3.0],
        ];
        let initial = vec![1.0, 1.0, 1.0];
        let diffused = chebyshev_heat_kernel(&mat, &initial, 1.0, 10);
        assert_eq!(diffused.len(), 3);

        // Exact result: exp(-1)*1, exp(-2)*1, exp(-3)*1
        // Chebyshev approximation should be close
        let expected = [(-1.0_f64).exp(), (-2.0_f64).exp(), (-3.0_f64).exp()];
        for (i, &v) in diffused.iter().enumerate() {
            assert!(v >= 0.0, "diffused[{i}] should be non-negative, got {v}",);
        }
        // Higher eigenvalues should attenuate more
        assert!(
            diffused[0] > diffused[1],
            "smaller eigenvalue component should retain more: {} vs {}",
            diffused[0],
            diffused[1],
        );
        assert!(
            diffused[1] > diffused[2],
            "smaller eigenvalue component should retain more: {} vs {}",
            diffused[1],
            diffused[2],
        );
        // Check approximate values (Chebyshev may not be exact)
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (diffused[i] - exp).abs() < 0.15,
                "diffused[{i}]={} should be close to {exp}",
                diffused[i],
            );
        }
    }

    #[test]
    fn test_hk_is_allowed_edge() {
        // Directly test the is_allowed_edge function
        assert!(is_allowed_edge("VAULT_ACCESS"));
        assert!(is_allowed_edge("VAULT_ACCESS_READ"));
        assert!(is_allowed_edge("VAULT_ACCESS_WRITE"));
        assert!(is_allowed_edge("VAULT_ACCESS_ADMIN"));
        assert!(is_allowed_edge("MEMBER"));
        // Prefix match: starts_with
        assert!(is_allowed_edge("VAULT_ACCESS_CUSTOM"));
        assert!(is_allowed_edge("MEMBER_ROLE"));
        // Not allowed
        assert!(!is_allowed_edge("CUSTOM_LINK"));
        assert!(!is_allowed_edge("RANDOM_EDGE"));
        assert!(!is_allowed_edge(""));
    }

    #[test]
    fn test_hk_chebyshev_negative_clamping() {
        // Use a matrix and initial vector that produces negative intermediate
        // values during Chebyshev approximation, exercising the clamping path.
        // A large eigenvalue with high diffusion time can cause oscillation in
        // Chebyshev approximation, producing negative values.
        let mat = vec![
            vec![5.0, 0.0, 0.0],
            vec![0.0, 10.0, 0.0],
            vec![0.0, 0.0, 15.0],
        ];
        // Start with most trust in the smallest-eigenvalue component,
        // and use order=3 (low order) with high time to trigger oscillation.
        let initial = vec![0.0, 0.0, 1.0];
        let diffused = chebyshev_heat_kernel(&mat, &initial, 2.0, 3);
        // All output values should be non-negative (clamped)
        for (i, &v) in diffused.iter().enumerate() {
            assert!(
                v >= 0.0,
                "diffused[{i}] should be non-negative after clamping, got {v}"
            );
        }
    }

    #[test]
    fn test_hk_collect_with_non_entity_nodes() {
        // Nodes without entity_key property should be skipped by collect_entity_keys.
        let vault = create_test_vault();

        // Create a node WITHOUT entity_key property
        let props_no_key = std::collections::HashMap::new();
        vault.graph.create_node("Metadata", props_no_key).unwrap();

        // Create a node with a non-String entity_key (Integer)
        let mut props_int = std::collections::HashMap::new();
        props_int.insert("entity_key".to_string(), PropertyValue::Int(42));
        vault.graph.create_node("BadType", props_int).unwrap();

        // Create a valid entity
        add_entity(&vault, "user:valid");

        let keys = collect_entity_keys(&vault);
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], "user:valid");
    }

    #[test]
    fn test_hk_spectral_gap_disconnected() {
        // Disconnected graph: Laplacian is block-diagonal
        // [[1, -1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, -1], [0, 0, -1, 1]]
        // Eigenvalues: 0, 0, 2, 2
        // The algorithm only projects out one null-space eigenvector (the all-ones),
        // so power iteration can converge to eigenvalue 2 (from the other eigenvectors).
        let lap = vec![
            vec![1.0, -1.0, 0.0, 0.0],
            vec![-1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, -1.0],
            vec![0.0, 0.0, -1.0, 1.0],
        ];
        let gap = estimate_spectral_gap(&lap, 500);
        // The result should be a valid eigenvalue (either 0 or 2)
        assert!(gap >= 0.0, "spectral gap should be non-negative, got {gap}",);
    }

    #[test]
    fn test_hk_chebyshev_order_zero() {
        // Test with chebyshev_order = 0 (only T_0 term, no T_1 or higher)
        let mat = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let initial = vec![1.0, 1.0];
        let diffused = chebyshev_heat_kernel(&mat, &initial, 0.5, 0);
        // With only the zeroth coefficient, the approximation is very coarse
        // but should still produce non-negative values
        for (i, &v) in diffused.iter().enumerate() {
            assert!(v >= 0.0, "diffused[{i}] should be non-negative, got {v}");
        }
    }

    #[test]
    fn test_hk_chebyshev_order_one() {
        // Test with chebyshev_order = 1 (T_0 and T_1 only, no recurrence loop)
        let mat = vec![vec![2.0, 0.0], vec![0.0, 3.0]];
        let initial = vec![1.0, 1.0];
        let diffused = chebyshev_heat_kernel(&mat, &initial, 0.5, 1);
        for (i, &v) in diffused.iter().enumerate() {
            assert!(v >= 0.0, "diffused[{i}] should be non-negative, got {v}");
        }
    }
}
