// SPDX-License-Identifier: MIT OR Apache-2.0
//! Secret-to-secret dependency tracking via graph edges.

use std::collections::{HashSet, VecDeque};

use graph_engine::{Direction, GraphEngine, PropertyValue};
use serde::{Deserialize, Serialize};

use crate::{Result, VaultError};

/// Edge type for secret dependencies.
const DEPENDS_ON_EDGE: &str = "SECRET_DEPENDS_ON";

/// Report of secrets and agents affected by a change to a root secret.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactReport {
    /// The root secret that was analyzed.
    pub root_secret: String,
    /// All secrets transitively depending on the root.
    pub affected_secrets: Vec<String>,
    /// Agents with access to any affected secret.
    pub affected_agents: Vec<String>,
    /// Maximum depth of the dependency chain.
    pub depth: usize,
}

/// Information about a single dependency edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyInfo {
    /// The parent secret (depended upon).
    pub parent: String,
    /// The child secret (depends on parent).
    pub child: String,
    /// When the dependency was created (unix millis).
    pub created_at_ms: i64,
}

/// Weight classification for dependency edges.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DependencyWeight {
    Critical,
    High,
    Medium,
    Low,
}

impl DependencyWeight {
    /// Numeric value for scoring.
    #[must_use]
    pub fn value(self) -> f64 {
        match self {
            Self::Critical => 1.0,
            Self::High => 0.7,
            Self::Medium => 0.4,
            Self::Low => 0.1,
        }
    }

    /// Parse from float stored in graph property.
    fn from_float(v: f64) -> Self {
        if v >= 0.9 {
            Self::Critical
        } else if v >= 0.6 {
            Self::High
        } else if v >= 0.3 {
            Self::Medium
        } else {
            Self::Low
        }
    }
}

/// A downstream secret with weight and impact score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedAffectedSecret {
    pub secret: String,
    pub depth: usize,
    pub edge_weight: DependencyWeight,
    pub impact_score: f64,
}

/// Weighted impact analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedImpactReport {
    pub root_secret: String,
    pub affected_secrets: Vec<WeightedAffectedSecret>,
    pub affected_agents: Vec<String>,
    pub max_depth: usize,
    pub total_impact_score: f64,
}

/// A step in a prioritized rotation plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationStep {
    pub secret: String,
    pub depth: usize,
    pub priority: f64,
    pub weight: DependencyWeight,
}

/// Prioritized rotation plan for cascading secret changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationPlan {
    pub root_secret: String,
    pub rotation_order: Vec<RotationStep>,
    pub total_secrets: usize,
}

/// Check for cycles before adding a dependency edge.
fn would_create_cycle(graph: &GraphEngine, parent_node: u64, child_node: u64) -> bool {
    // If adding child -> parent edge for DEPENDS_ON, check if parent can reach child
    // (which would create a cycle)
    if parent_node == child_node {
        return true;
    }

    // BFS from child to see if we can reach parent via DEPENDS_ON edges
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(child_node);
    visited.insert(child_node);

    while let Some(current) = queue.pop_front() {
        if let Ok(edges) = graph.edges_of(current, Direction::Outgoing) {
            for edge in edges {
                if edge.edge_type != DEPENDS_ON_EDGE {
                    continue;
                }
                let target = if edge.from == current {
                    edge.to
                } else {
                    edge.from
                };
                if target == parent_node {
                    return true;
                }
                if visited.insert(target) {
                    queue.push_back(target);
                }
            }
        }
    }

    false
}

/// Add a dependency: child depends on parent.
pub fn add_dependency(
    graph: &GraphEngine,
    parent_node_key: &str,
    child_node_key: &str,
    timestamp: i64,
) -> Result<()> {
    let parent_id = find_or_create_dep_node(graph, parent_node_key);
    let child_id = find_or_create_dep_node(graph, child_node_key);

    if would_create_cycle(graph, parent_id, child_id) {
        return Err(VaultError::CyclicDependency(format!(
            "adding dependency from '{child_node_key}' on '{parent_node_key}' would create a cycle"
        )));
    }

    let mut props = std::collections::HashMap::new();
    props.insert("created_at_ms".to_string(), PropertyValue::Int(timestamp));

    graph
        .create_edge(parent_id, child_id, DEPENDS_ON_EDGE, props, true)
        .map_err(|e| VaultError::GraphError(e.to_string()))?;

    Ok(())
}

/// Remove a dependency edge between parent and child.
pub fn remove_dependency(
    graph: &GraphEngine,
    parent_node_key: &str,
    child_node_key: &str,
) -> Result<()> {
    let Some(parent_id) = find_dep_node(graph, parent_node_key) else {
        return Ok(());
    };
    let Some(child_id) = find_dep_node(graph, child_node_key) else {
        return Ok(());
    };

    if let Ok(edges) = graph.edges_of(parent_id, Direction::Outgoing) {
        for edge in edges {
            if edge.edge_type == DEPENDS_ON_EDGE && edge.to == child_id {
                graph
                    .delete_edge(edge.id)
                    .map_err(|e| VaultError::GraphError(e.to_string()))?;
            }
        }
    }

    Ok(())
}

/// Get direct children (secrets that depend on this one).
pub fn get_dependencies(graph: &GraphEngine, node_key: &str) -> Vec<String> {
    let Some(node_id) = find_dep_node(graph, node_key) else {
        return Vec::new();
    };

    let mut children = Vec::new();
    if let Ok(edges) = graph.edges_of(node_id, Direction::Outgoing) {
        for edge in edges {
            if edge.edge_type != DEPENDS_ON_EDGE {
                continue;
            }
            let target = if edge.from == node_id {
                edge.to
            } else {
                edge.from
            };
            if let Some(key) = node_entity_key(graph, target) {
                children.push(key);
            }
        }
    }

    children
}

/// Get direct parents (secrets this one depends on).
pub fn get_dependents(graph: &GraphEngine, node_key: &str) -> Vec<String> {
    let Some(node_id) = find_dep_node(graph, node_key) else {
        return Vec::new();
    };

    let mut parents = Vec::new();
    if let Ok(edges) = graph.edges_of(node_id, Direction::Incoming) {
        for edge in edges {
            if edge.edge_type != DEPENDS_ON_EDGE {
                continue;
            }
            let source = if edge.to == node_id {
                edge.from
            } else {
                edge.to
            };
            if let Some(key) = node_entity_key(graph, source) {
                parents.push(key);
            }
        }
    }

    parents
}

/// Transitive BFS to find all affected secrets downstream.
pub fn impact_analysis(graph: &GraphEngine, node_key: &str) -> ImpactReport {
    let mut affected = Vec::new();
    let mut max_depth = 0;

    let Some(start_id) = find_dep_node(graph, node_key) else {
        return ImpactReport {
            root_secret: node_key.to_string(),
            affected_secrets: Vec::new(),
            affected_agents: Vec::new(),
            depth: 0,
        };
    };

    let mut visited = HashSet::new();
    visited.insert(start_id);
    let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
    queue.push_back((start_id, 0));

    while let Some((current, depth)) = queue.pop_front() {
        if let Ok(edges) = graph.edges_of(current, Direction::Outgoing) {
            for edge in edges {
                if edge.edge_type != DEPENDS_ON_EDGE {
                    continue;
                }
                let target = if edge.from == current {
                    edge.to
                } else {
                    edge.from
                };
                if visited.insert(target) {
                    let new_depth = depth + 1;
                    if new_depth > max_depth {
                        max_depth = new_depth;
                    }
                    if let Some(key) = node_entity_key(graph, target) {
                        affected.push(key);
                    }
                    queue.push_back((target, new_depth));
                }
            }
        }
    }

    // Collect agents with access to affected secrets (look for VAULT_ACCESS edges)
    let mut agents = HashSet::new();
    for node_id in &visited {
        if let Ok(edges) = graph.edges_of(*node_id, Direction::Incoming) {
            for edge in edges {
                if edge.edge_type.starts_with("VAULT_ACCESS") {
                    let source = if edge.to == *node_id {
                        edge.from
                    } else {
                        edge.to
                    };
                    if let Some(key) = node_entity_key(graph, source) {
                        if !key.starts_with("vault_secret:") {
                            agents.insert(key);
                        }
                    }
                }
            }
        }
    }

    ImpactReport {
        root_secret: node_key.to_string(),
        affected_secrets: affected,
        affected_agents: agents.into_iter().collect(),
        depth: max_depth,
    }
}

/// Add a weighted dependency: child depends on parent with a given weight.
pub fn add_weighted_dependency(
    graph: &GraphEngine,
    parent_key: &str,
    child_key: &str,
    weight: DependencyWeight,
    description: Option<&str>,
    timestamp: i64,
) -> Result<()> {
    let parent_id = find_or_create_dep_node(graph, parent_key);
    let child_id = find_or_create_dep_node(graph, child_key);

    if would_create_cycle(graph, parent_id, child_id) {
        return Err(VaultError::CyclicDependency(format!(
            "adding dependency from '{child_key}' on '{parent_key}' would create a cycle"
        )));
    }

    let mut props = std::collections::HashMap::new();
    props.insert("created_at_ms".to_string(), PropertyValue::Int(timestamp));
    props.insert(
        "dep_weight".to_string(),
        PropertyValue::Float(weight.value()),
    );
    if let Some(desc) = description {
        props.insert(
            "dep_desc".to_string(),
            PropertyValue::String(desc.to_string()),
        );
    }

    graph
        .create_edge(parent_id, child_id, DEPENDS_ON_EDGE, props, true)
        .map_err(|e| VaultError::GraphError(e.to_string()))?;

    Ok(())
}

/// Transitive BFS with weights to find all affected secrets downstream.
#[must_use]
pub fn weighted_impact_analysis(graph: &GraphEngine, node_key: &str) -> WeightedImpactReport {
    let Some(start_id) = find_dep_node(graph, node_key) else {
        return WeightedImpactReport {
            root_secret: node_key.to_string(),
            affected_secrets: Vec::new(),
            affected_agents: Vec::new(),
            max_depth: 0,
            total_impact_score: 0.0,
        };
    };

    let mut affected = Vec::new();
    let mut max_depth = 0;
    let mut total_impact = 0.0;

    let mut visited = HashSet::new();
    visited.insert(start_id);
    let mut queue: VecDeque<(u64, usize)> = VecDeque::new();
    queue.push_back((start_id, 0));

    while let Some((current, depth)) = queue.pop_front() {
        if let Ok(edges) = graph.edges_of(current, Direction::Outgoing) {
            for edge in edges {
                if edge.edge_type != DEPENDS_ON_EDGE {
                    continue;
                }
                let target = if edge.from == current {
                    edge.to
                } else {
                    edge.from
                };
                if visited.insert(target) {
                    let new_depth = depth + 1;
                    if new_depth > max_depth {
                        max_depth = new_depth;
                    }

                    let weight = edge
                        .properties
                        .get("dep_weight")
                        .and_then(|p| match p {
                            PropertyValue::Float(f) => Some(DependencyWeight::from_float(*f)),
                            _ => None,
                        })
                        .unwrap_or(DependencyWeight::Medium);

                    #[allow(clippy::cast_precision_loss)] // depth will never exceed 2^52
                    let impact_score = weight.value() / new_depth as f64;
                    total_impact += impact_score;

                    if let Some(key) = node_entity_key(graph, target) {
                        affected.push(WeightedAffectedSecret {
                            secret: key,
                            depth: new_depth,
                            edge_weight: weight,
                            impact_score,
                        });
                    }
                    queue.push_back((target, new_depth));
                }
            }
        }
    }

    // Collect affected agents
    let mut agents = HashSet::new();
    for node_id in &visited {
        if let Ok(edges) = graph.edges_of(*node_id, Direction::Incoming) {
            for edge in edges {
                if edge.edge_type.starts_with("VAULT_ACCESS") {
                    let source = if edge.to == *node_id {
                        edge.from
                    } else {
                        edge.to
                    };
                    if let Some(key) = node_entity_key(graph, source) {
                        if !key.starts_with("vault_secret:") {
                            agents.insert(key);
                        }
                    }
                }
            }
        }
    }

    WeightedImpactReport {
        root_secret: node_key.to_string(),
        affected_secrets: affected,
        affected_agents: agents.into_iter().collect(),
        max_depth,
        total_impact_score: total_impact,
    }
}

/// Generate a prioritized rotation plan based on weighted dependencies.
#[must_use]
pub fn rotation_plan(graph: &GraphEngine, root_key: &str) -> RotationPlan {
    let report = weighted_impact_analysis(graph, root_key);

    let mut steps: Vec<RotationStep> = report
        .affected_secrets
        .into_iter()
        .map(|s| RotationStep {
            secret: s.secret,
            depth: s.depth,
            priority: s.impact_score,
            weight: s.edge_weight,
        })
        .collect();

    // Sort by priority descending (highest impact first)
    steps.sort_by(|a, b| {
        b.priority
            .partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total = steps.len();
    RotationPlan {
        root_secret: root_key.to_string(),
        rotation_order: steps,
        total_secrets: total,
    }
}

fn find_dep_node(graph: &GraphEngine, key: &str) -> Option<u64> {
    graph
        .find_nodes_by_property("entity_key", &PropertyValue::String(key.to_string()))
        .ok()
        .and_then(|nodes| nodes.first().map(|n| n.id))
}

fn find_or_create_dep_node(graph: &GraphEngine, key: &str) -> u64 {
    if let Some(id) = find_dep_node(graph, key) {
        return id;
    }

    let mut props = std::collections::HashMap::new();
    props.insert(
        "entity_key".to_string(),
        PropertyValue::String(key.to_string()),
    );
    graph.create_node("VaultEntity", props).unwrap_or(0)
}

fn node_entity_key(graph: &GraphEngine, node_id: u64) -> Option<String> {
    graph.get_node(node_id).ok().and_then(|node| {
        if let Some(PropertyValue::String(key)) = node.properties.get("entity_key") {
            Some(key.clone())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn test_graph() -> Arc<GraphEngine> {
        Arc::new(GraphEngine::new())
    }

    #[test]
    fn test_add_and_get_dependencies() {
        let graph = test_graph();
        add_dependency(&graph, "secret:db_password", "secret:app_config", 1000).unwrap();

        let children = get_dependencies(&graph, "secret:db_password");
        assert_eq!(children, vec!["secret:app_config"]);

        let parents = get_dependents(&graph, "secret:app_config");
        assert_eq!(parents, vec!["secret:db_password"]);
    }

    #[test]
    fn test_remove_dependency() {
        let graph = test_graph();
        add_dependency(&graph, "secret:a", "secret:b", 1000).unwrap();
        assert_eq!(get_dependencies(&graph, "secret:a").len(), 1);

        remove_dependency(&graph, "secret:a", "secret:b").unwrap();
        assert!(get_dependencies(&graph, "secret:a").is_empty());
    }

    #[test]
    fn test_cycle_detection_self_reference() {
        let graph = test_graph();
        let result = add_dependency(&graph, "secret:a", "secret:a", 1000);
        assert!(matches!(result, Err(VaultError::CyclicDependency(_))));
    }

    #[test]
    fn test_cycle_detection_two_node() {
        let graph = test_graph();
        add_dependency(&graph, "secret:a", "secret:b", 1000).unwrap();
        let result = add_dependency(&graph, "secret:b", "secret:a", 2000);
        assert!(matches!(result, Err(VaultError::CyclicDependency(_))));
    }

    #[test]
    fn test_cycle_detection_three_node() {
        let graph = test_graph();
        add_dependency(&graph, "secret:a", "secret:b", 1000).unwrap();
        add_dependency(&graph, "secret:b", "secret:c", 2000).unwrap();
        let result = add_dependency(&graph, "secret:c", "secret:a", 3000);
        assert!(matches!(result, Err(VaultError::CyclicDependency(_))));
    }

    #[test]
    fn test_impact_analysis_single_level() {
        let graph = test_graph();
        add_dependency(&graph, "secret:root", "secret:child1", 1000).unwrap();
        add_dependency(&graph, "secret:root", "secret:child2", 2000).unwrap();

        let report = impact_analysis(&graph, "secret:root");
        assert_eq!(report.root_secret, "secret:root");
        assert_eq!(report.affected_secrets.len(), 2);
        assert_eq!(report.depth, 1);
    }

    #[test]
    fn test_impact_analysis_transitive() {
        let graph = test_graph();
        add_dependency(&graph, "secret:a", "secret:b", 1000).unwrap();
        add_dependency(&graph, "secret:b", "secret:c", 2000).unwrap();
        add_dependency(&graph, "secret:c", "secret:d", 3000).unwrap();

        let report = impact_analysis(&graph, "secret:a");
        assert_eq!(report.affected_secrets.len(), 3);
        assert_eq!(report.depth, 3);
    }

    #[test]
    fn test_impact_analysis_diamond() {
        let graph = test_graph();
        // A -> B, A -> C, B -> D, C -> D
        add_dependency(&graph, "secret:a", "secret:b", 1000).unwrap();
        add_dependency(&graph, "secret:a", "secret:c", 2000).unwrap();
        add_dependency(&graph, "secret:b", "secret:d", 3000).unwrap();
        add_dependency(&graph, "secret:c", "secret:d", 4000).unwrap();

        let report = impact_analysis(&graph, "secret:a");
        assert_eq!(report.affected_secrets.len(), 3); // b, c, d (each counted once)
        assert_eq!(report.depth, 2);
    }

    #[test]
    fn test_impact_analysis_no_deps() {
        let graph = test_graph();
        let report = impact_analysis(&graph, "secret:isolated");
        assert!(report.affected_secrets.is_empty());
        assert_eq!(report.depth, 0);
    }

    #[test]
    fn test_multiple_children() {
        let graph = test_graph();
        add_dependency(&graph, "secret:parent", "secret:c1", 1000).unwrap();
        add_dependency(&graph, "secret:parent", "secret:c2", 2000).unwrap();
        add_dependency(&graph, "secret:parent", "secret:c3", 3000).unwrap();

        let children = get_dependencies(&graph, "secret:parent");
        assert_eq!(children.len(), 3);
    }

    #[test]
    fn test_multiple_parents() {
        let graph = test_graph();
        add_dependency(&graph, "secret:p1", "secret:child", 1000).unwrap();
        add_dependency(&graph, "secret:p2", "secret:child", 2000).unwrap();

        let parents = get_dependents(&graph, "secret:child");
        assert_eq!(parents.len(), 2);
    }

    #[test]
    fn test_get_deps_nonexistent() {
        let graph = test_graph();
        let children = get_dependencies(&graph, "secret:nonexistent");
        assert!(children.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_dependency() {
        let graph = test_graph();
        // Should not error
        remove_dependency(&graph, "secret:x", "secret:y").unwrap();
    }

    #[test]
    fn test_no_false_positive_cycles_for_siblings() {
        let graph = test_graph();
        // A -> B and A -> C should not prevent B -> D or C -> D
        add_dependency(&graph, "secret:a", "secret:b", 1000).unwrap();
        add_dependency(&graph, "secret:a", "secret:c", 2000).unwrap();
        add_dependency(&graph, "secret:b", "secret:d", 3000).unwrap();
        add_dependency(&graph, "secret:c", "secret:d", 4000).unwrap();
    }

    #[test]
    fn test_dependency_info_structure() {
        let info = DependencyInfo {
            parent: "secret:parent".to_string(),
            child: "secret:child".to_string(),
            created_at_ms: 12345,
        };
        assert_eq!(info.parent, "secret:parent");
        assert_eq!(info.child, "secret:child");
        assert_eq!(info.created_at_ms, 12345);
    }

    #[test]
    fn test_impact_report_serialization() {
        let report = ImpactReport {
            root_secret: "secret:root".to_string(),
            affected_secrets: vec!["secret:a".to_string()],
            affected_agents: vec!["user:alice".to_string()],
            depth: 1,
        };
        let json = serde_json::to_string(&report).unwrap();
        let deserialized: ImpactReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.root_secret, report.root_secret);
    }

    #[test]
    fn test_add_weighted_dep() {
        let graph = test_graph();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:b",
            DependencyWeight::Critical,
            Some("critical link"),
            1000,
        )
        .unwrap();

        let children = get_dependencies(&graph, "secret:a");
        assert_eq!(children, vec!["secret:b"]);
    }

    #[test]
    fn test_weighted_cycle_detection() {
        let graph = test_graph();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:b",
            DependencyWeight::High,
            None,
            1000,
        )
        .unwrap();
        let result = add_weighted_dependency(
            &graph,
            "secret:b",
            "secret:a",
            DependencyWeight::High,
            None,
            2000,
        );
        assert!(matches!(result, Err(VaultError::CyclicDependency(_))));
    }

    #[test]
    fn test_weighted_impact_single() {
        let graph = test_graph();
        add_weighted_dependency(
            &graph,
            "secret:root",
            "secret:child",
            DependencyWeight::Critical,
            None,
            1000,
        )
        .unwrap();

        let report = weighted_impact_analysis(&graph, "secret:root");
        assert_eq!(report.affected_secrets.len(), 1);
        assert_eq!(report.affected_secrets[0].secret, "secret:child");
        assert_eq!(report.affected_secrets[0].depth, 1);
        assert!((report.affected_secrets[0].edge_weight.value() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weighted_impact_mixed() {
        let graph = test_graph();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:b",
            DependencyWeight::Critical,
            None,
            1000,
        )
        .unwrap();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:c",
            DependencyWeight::Low,
            None,
            2000,
        )
        .unwrap();

        let report = weighted_impact_analysis(&graph, "secret:a");
        assert_eq!(report.affected_secrets.len(), 2);
        assert!(report.total_impact_score > 0.0);
    }

    #[test]
    fn test_impact_score_calculation() {
        let graph = test_graph();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:b",
            DependencyWeight::Critical,
            None,
            1000,
        )
        .unwrap();

        let report = weighted_impact_analysis(&graph, "secret:a");
        // impact_score = weight.value() / depth = 1.0 / 1 = 1.0
        assert!((report.affected_secrets[0].impact_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rotation_plan_chain() {
        let graph = test_graph();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:b",
            DependencyWeight::Critical,
            None,
            1000,
        )
        .unwrap();
        add_weighted_dependency(
            &graph,
            "secret:b",
            "secret:c",
            DependencyWeight::High,
            None,
            2000,
        )
        .unwrap();

        let plan = rotation_plan(&graph, "secret:a");
        assert_eq!(plan.total_secrets, 2);
        assert_eq!(plan.rotation_order.len(), 2);
    }

    #[test]
    fn test_rotation_plan_diamond() {
        let graph = test_graph();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:b",
            DependencyWeight::High,
            None,
            1000,
        )
        .unwrap();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:c",
            DependencyWeight::Medium,
            None,
            2000,
        )
        .unwrap();
        add_weighted_dependency(
            &graph,
            "secret:b",
            "secret:d",
            DependencyWeight::Low,
            None,
            3000,
        )
        .unwrap();
        add_weighted_dependency(
            &graph,
            "secret:c",
            "secret:d",
            DependencyWeight::Low,
            None,
            4000,
        )
        .unwrap();

        let plan = rotation_plan(&graph, "secret:a");
        assert_eq!(plan.total_secrets, 3); // b, c, d
    }

    #[test]
    fn test_rotation_critical_first() {
        let graph = test_graph();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:low",
            DependencyWeight::Low,
            None,
            1000,
        )
        .unwrap();
        add_weighted_dependency(
            &graph,
            "secret:a",
            "secret:critical",
            DependencyWeight::Critical,
            None,
            2000,
        )
        .unwrap();

        let plan = rotation_plan(&graph, "secret:a");
        assert_eq!(plan.rotation_order.len(), 2);
        // Critical should come first (higher priority)
        assert_eq!(plan.rotation_order[0].secret, "secret:critical");
    }
}
