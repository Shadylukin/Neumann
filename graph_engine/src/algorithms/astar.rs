//! A* pathfinding algorithm with pluggable heuristics.
//!
//! Finds the shortest path between two nodes using the A* algorithm,
//! which combines actual distance traveled with a heuristic estimate
//! of remaining distance.

#![allow(clippy::cast_precision_loss)] // Acceptable for graph algorithm metrics

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{Direction, GraphEngine, PropertyValue, Result, WeightedPath};

/// Heuristic function type for A* pathfinding.
///
/// Takes source node ID, target node ID, and the graph engine.
/// Returns an estimated cost from source to target (must never overestimate).
pub type HeuristicFn = Box<dyn Fn(u64, u64, &GraphEngine) -> f64 + Send + Sync>;

/// Configuration for A* pathfinding.
#[derive(Default)]
pub struct AStarConfig {
    /// Property name to use as edge weight.
    pub weight_property: Option<String>,
    /// Default weight for edges without the weight property.
    pub default_weight: f64,
    /// Edge type filter. None means all edge types.
    pub edge_type: Option<String>,
    /// Direction of traversal.
    pub direction: Direction,
    /// Custom heuristic function.
    heuristic: Option<HeuristicFn>,
}

impl std::fmt::Debug for AStarConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AStarConfig")
            .field("weight_property", &self.weight_property)
            .field("default_weight", &self.default_weight)
            .field("edge_type", &self.edge_type)
            .field("direction", &self.direction)
            .field("heuristic", &self.heuristic.is_some())
            .finish()
    }
}

impl AStarConfig {
    #[must_use]
    pub fn new() -> Self {
        Self {
            weight_property: Some("weight".to_string()),
            default_weight: 1.0,
            edge_type: None,
            direction: Direction::Outgoing,
            heuristic: None,
        }
    }

    #[must_use]
    pub fn weight_property(mut self, property: impl Into<String>) -> Self {
        self.weight_property = Some(property.into());
        self
    }

    #[must_use]
    pub fn unweighted(mut self) -> Self {
        self.weight_property = None;
        self
    }

    #[must_use]
    pub const fn default_weight(mut self, weight: f64) -> Self {
        self.default_weight = weight;
        self
    }

    #[must_use]
    pub fn edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.edge_type = Some(edge_type.into());
        self
    }

    #[must_use]
    pub const fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }

    #[must_use]
    pub fn heuristic(mut self, h: HeuristicFn) -> Self {
        self.heuristic = Some(h);
        self
    }
}

/// Result of A* pathfinding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AStarResult {
    /// The shortest path found, if any.
    pub path: Option<WeightedPath>,
    /// Number of nodes explored during search.
    pub nodes_explored: usize,
    /// Number of nodes in the open set at termination.
    pub open_set_size: usize,
}

impl AStarResult {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            path: None,
            nodes_explored: 0,
            open_set_size: 0,
        }
    }

    #[must_use]
    pub const fn found(&self) -> bool {
        self.path.is_some()
    }

    #[must_use]
    pub fn path_length(&self) -> Option<usize> {
        self.path.as_ref().map(|p| p.edges.len())
    }

    #[must_use]
    pub fn total_weight(&self) -> Option<f64> {
        self.path.as_ref().map(|p| p.total_weight)
    }
}

impl Default for AStarResult {
    fn default() -> Self {
        Self::empty()
    }
}

/// Entry in the A* priority queue.
#[derive(Debug, Clone)]
struct AStarEntry {
    node_id: u64,
    /// f(n) = g(n) + h(n)
    f_score: f64,
    /// g(n) = actual cost from start
    g_score: f64,
}

impl PartialEq for AStarEntry {
    fn eq(&self, other: &Self) -> bool {
        self.node_id == other.node_id
    }
}

impl Eq for AStarEntry {}

impl Ord for AStarEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (lower f_score = higher priority)
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for AStarEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Extract coordinate value from property.
fn extract_coord(props: &HashMap<String, PropertyValue>, key: &str) -> Option<f64> {
    match props.get(key) {
        Some(PropertyValue::Float(f)) => Some(*f),
        Some(PropertyValue::Int(i)) => Some(*i as f64),
        _ => None,
    }
}

impl GraphEngine {
    /// Find the shortest path using A* algorithm.
    ///
    /// Time complexity: O((V + E) log V) with a good heuristic.
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn astar_path(&self, from: u64, to: u64, config: &AStarConfig) -> Result<AStarResult> {
        if from == to {
            return Ok(AStarResult {
                path: Some(WeightedPath {
                    nodes: vec![from],
                    edges: Vec::new(),
                    total_weight: 0.0,
                }),
                nodes_explored: 1,
                open_set_size: 0,
            });
        }

        // Check nodes exist
        if !self.node_exists(from) || !self.node_exists(to) {
            return Ok(AStarResult::empty());
        }

        // Default heuristic: zero (degrades to Dijkstra)
        let default_heuristic = |_: u64, _: u64, _: &Self| 0.0;
        let heuristic = config
            .heuristic
            .as_ref()
            .map_or(&default_heuristic as &dyn Fn(u64, u64, &Self) -> f64, |h| {
                h.as_ref()
            });

        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut g_scores: HashMap<u64, f64> = HashMap::new();
        let mut came_from: HashMap<u64, (u64, u64)> = HashMap::new();

        let h_start = heuristic(from, to, self);
        open_set.push(AStarEntry {
            node_id: from,
            f_score: h_start,
            g_score: 0.0,
        });
        g_scores.insert(from, 0.0);

        let mut nodes_explored = 0;

        while let Some(current) = open_set.pop() {
            if closed_set.contains(&current.node_id) {
                continue;
            }

            nodes_explored += 1;

            if current.node_id == to {
                let path = self.reconstruct_astar_path(from, to, &came_from, current.g_score);
                return Ok(AStarResult {
                    path: Some(path),
                    nodes_explored,
                    open_set_size: open_set.len(),
                });
            }

            closed_set.insert(current.node_id);

            let neighbors = self.neighbors(
                current.node_id,
                config.edge_type.as_deref(),
                config.direction,
                None,
            )?;

            for neighbor in neighbors {
                if closed_set.contains(&neighbor.id) {
                    continue;
                }

                let weight = self.get_astar_edge_weight(
                    current.node_id,
                    neighbor.id,
                    config.weight_property.as_deref(),
                    config.default_weight,
                    config.edge_type.as_deref(),
                    config.direction,
                );

                let tentative_g = current.g_score + weight.0;
                let current_g = g_scores.get(&neighbor.id).copied().unwrap_or(f64::INFINITY);

                if tentative_g < current_g {
                    came_from.insert(neighbor.id, (current.node_id, weight.1));
                    g_scores.insert(neighbor.id, tentative_g);

                    let h = heuristic(neighbor.id, to, self);
                    open_set.push(AStarEntry {
                        node_id: neighbor.id,
                        f_score: tentative_g + h,
                        g_score: tentative_g,
                    });
                }
            }
        }

        Ok(AStarResult {
            path: None,
            nodes_explored,
            open_set_size: 0,
        })
    }

    /// A* with Euclidean heuristic for spatial graphs.
    ///
    /// # Errors
    ///
    /// Returns an error if pathfinding fails.
    pub fn astar_path_euclidean(
        &self,
        from: u64,
        to: u64,
        x_property: &str,
        y_property: &str,
    ) -> Result<AStarResult> {
        let x_prop = x_property.to_string();
        let y_prop = y_property.to_string();

        let heuristic: HeuristicFn = Box::new(move |current, target, engine| {
            let Ok(current_node) = engine.get_node(current) else {
                return 0.0;
            };
            let Ok(target_node) = engine.get_node(target) else {
                return 0.0;
            };

            let Some(cx) = extract_coord(&current_node.properties, &x_prop) else {
                return 0.0;
            };
            let Some(cy) = extract_coord(&current_node.properties, &y_prop) else {
                return 0.0;
            };
            let Some(tx) = extract_coord(&target_node.properties, &x_prop) else {
                return 0.0;
            };
            let Some(ty) = extract_coord(&target_node.properties, &y_prop) else {
                return 0.0;
            };

            (tx - cx).hypot(ty - cy)
        });

        self.astar_path(from, to, &AStarConfig::new().heuristic(heuristic))
    }

    /// A* with Manhattan heuristic for grid-based graphs.
    ///
    /// # Errors
    ///
    /// Returns an error if pathfinding fails.
    pub fn astar_path_manhattan(
        &self,
        from: u64,
        to: u64,
        x_property: &str,
        y_property: &str,
    ) -> Result<AStarResult> {
        let x_prop = x_property.to_string();
        let y_prop = y_property.to_string();

        let heuristic: HeuristicFn = Box::new(move |current, target, engine| {
            let Ok(current_node) = engine.get_node(current) else {
                return 0.0;
            };
            let Ok(target_node) = engine.get_node(target) else {
                return 0.0;
            };

            let Some(cx) = extract_coord(&current_node.properties, &x_prop) else {
                return 0.0;
            };
            let Some(cy) = extract_coord(&current_node.properties, &y_prop) else {
                return 0.0;
            };
            let Some(tx) = extract_coord(&target_node.properties, &x_prop) else {
                return 0.0;
            };
            let Some(ty) = extract_coord(&target_node.properties, &y_prop) else {
                return 0.0;
            };

            (tx - cx).abs() + (ty - cy).abs()
        });

        self.astar_path(from, to, &AStarConfig::new().heuristic(heuristic))
    }

    #[allow(clippy::unused_self)] // Keep &self for API consistency
    fn reconstruct_astar_path(
        &self,
        from: u64,
        to: u64,
        came_from: &HashMap<u64, (u64, u64)>,
        total_weight: f64,
    ) -> WeightedPath {
        let mut nodes = vec![to];
        let mut edges = Vec::new();
        let mut current = to;

        while current != from {
            if let Some(&(parent, edge_id)) = came_from.get(&current) {
                nodes.push(parent);
                edges.push(edge_id);
                current = parent;
            } else {
                break;
            }
        }

        nodes.reverse();
        edges.reverse();

        WeightedPath {
            nodes,
            edges,
            total_weight,
        }
    }

    /// Get edge weight between two nodes (returns weight and `edge_id`).
    fn get_astar_edge_weight(
        &self,
        from: u64,
        to: u64,
        weight_property: Option<&str>,
        default_weight: f64,
        edge_type: Option<&str>,
        direction: Direction,
    ) -> (f64, u64) {
        let edges_key = match direction {
            Direction::Outgoing | Direction::Both => Self::outgoing_edges_key(from),
            Direction::Incoming => Self::incoming_edges_key(from),
        };

        for edge_id in self.get_edge_list(&edges_key) {
            let Ok(edge) = self.get_edge(edge_id) else {
                continue;
            };

            let connects = match direction {
                Direction::Outgoing => edge.to == to,
                Direction::Incoming => edge.from == to,
                Direction::Both => edge.to == to || edge.from == to,
            };

            if !connects {
                continue;
            }

            if let Some(et) = edge_type {
                if edge.edge_type != et {
                    continue;
                }
            }

            let weight = match weight_property {
                Some(prop) => match edge.properties.get(prop) {
                    Some(PropertyValue::Float(w)) => *w,
                    Some(PropertyValue::Int(w)) => *w as f64,
                    _ => default_weight,
                },
                None => default_weight,
            };

            return (weight, edge_id);
        }

        (default_weight, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_weighted_edge(engine: &GraphEngine, from: u64, to: u64, weight: f64) -> u64 {
        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(weight));
        engine.create_edge(from, to, "EDGE", props, true).unwrap()
    }

    fn create_spatial_node(engine: &GraphEngine, x: f64, y: f64) -> u64 {
        let mut props = HashMap::new();
        props.insert("x".to_string(), PropertyValue::Float(x));
        props.insert("y".to_string(), PropertyValue::Float(y));
        engine.create_node("Point", props).unwrap()
    }

    #[test]
    fn test_astar_same_node() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.astar_path(a, a, &AStarConfig::new()).unwrap();
        assert!(result.found());
        assert_eq!(result.path_length(), Some(0));
    }

    #[test]
    fn test_astar_direct_path() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 5.0);

        let result = engine.astar_path(a, b, &AStarConfig::new()).unwrap();
        assert!(result.found());
        assert_eq!(result.total_weight(), Some(5.0));
    }

    #[test]
    fn test_astar_chooses_shorter_path() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, c, 10.0);
        create_weighted_edge(&engine, a, b, 1.0);
        create_weighted_edge(&engine, b, c, 2.0);

        let result = engine.astar_path(a, c, &AStarConfig::new()).unwrap();
        assert!(result.found());
        assert!((result.total_weight().unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_astar_no_path() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        let result = engine.astar_path(a, b, &AStarConfig::new()).unwrap();
        assert!(!result.found());
    }

    #[test]
    fn test_astar_euclidean_heuristic() {
        let engine = GraphEngine::new();

        let a = create_spatial_node(&engine, 0.0, 0.0);
        let b = create_spatial_node(&engine, 1.0, 0.0);
        let c = create_spatial_node(&engine, 2.0, 0.0);

        create_weighted_edge(&engine, a, b, 1.0);
        create_weighted_edge(&engine, b, c, 1.0);

        let result = engine.astar_path_euclidean(a, c, "x", "y").unwrap();
        assert!(result.found());
        assert!((result.total_weight().unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_astar_manhattan_heuristic() {
        let engine = GraphEngine::new();

        let a = create_spatial_node(&engine, 0.0, 0.0);
        let b = create_spatial_node(&engine, 1.0, 1.0);

        create_weighted_edge(&engine, a, b, 2.0);

        let result = engine.astar_path_manhattan(a, b, "x", "y").unwrap();
        assert!(result.found());
        assert!((result.total_weight().unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_astar_unweighted() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "EDGE", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), true)
            .unwrap();

        let config = AStarConfig::new().unweighted().default_weight(1.0);
        let result = engine.astar_path(a, c, &config).unwrap();
        assert!(result.found());
        assert!((result.total_weight().unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_astar_node_not_found() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.astar_path(a, 9999, &AStarConfig::new()).unwrap();
        assert!(!result.found());
    }

    #[test]
    fn test_astar_config_debug() {
        let config = AStarConfig::new();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("AStarConfig"));
        assert!(debug_str.contains("weight_property"));
        assert!(debug_str.contains("heuristic"));
    }

    #[test]
    fn test_astar_config_weight_property() {
        let config = AStarConfig::new().weight_property("cost");
        assert_eq!(config.weight_property, Some("cost".to_string()));
    }

    #[test]
    fn test_astar_config_default_weight() {
        let config = AStarConfig::new().default_weight(2.5);
        assert!((config.default_weight - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_astar_config_edge_type() {
        let config = AStarConfig::new().edge_type("ROAD");
        assert_eq!(config.edge_type, Some("ROAD".to_string()));
    }

    #[test]
    fn test_astar_config_direction() {
        let config = AStarConfig::new().direction(Direction::Incoming);
        assert_eq!(config.direction, Direction::Incoming);

        let config2 = AStarConfig::new().direction(Direction::Both);
        assert_eq!(config2.direction, Direction::Both);
    }

    #[test]
    fn test_astar_config_heuristic() {
        let config = AStarConfig::new().heuristic(Box::new(|_src, _dst, _engine| 1.0));
        assert!(config.heuristic.is_some());
    }

    #[test]
    fn test_astar_result_empty() {
        let result = AStarResult::empty();
        assert!(!result.found());
        assert_eq!(result.path_length(), None);
        assert_eq!(result.total_weight(), None);
        assert_eq!(result.nodes_explored, 0);
        assert_eq!(result.open_set_size, 0);
    }

    #[test]
    fn test_astar_result_default() {
        let result = AStarResult::default();
        assert!(!result.found());
        assert_eq!(result.path, None);
    }

    #[test]
    fn test_astar_result_accessors() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 3.0);
        create_weighted_edge(&engine, b, c, 4.0);

        let result = engine.astar_path(a, c, &AStarConfig::new()).unwrap();
        assert!(result.found());
        assert_eq!(result.path_length(), Some(2));
        assert!((result.total_weight().unwrap() - 7.0).abs() < f64::EPSILON);
        assert!(result.nodes_explored > 0);
    }

    #[test]
    fn test_astar_with_integer_weight() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Int(5));
        engine.create_edge(a, b, "EDGE", props, true).unwrap();

        let result = engine.astar_path(a, b, &AStarConfig::new()).unwrap();
        assert!(result.found());
        assert!((result.total_weight().unwrap() - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_astar_missing_weight_uses_default() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "EDGE", HashMap::new(), true)
            .unwrap();

        let config = AStarConfig::new()
            .weight_property("nonexistent")
            .default_weight(3.0);
        let result = engine.astar_path(a, b, &config).unwrap();
        assert!(result.found());
        assert!((result.total_weight().unwrap() - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_astar_edge_type_filter() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 1.0);

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(1.0));
        engine.create_edge(b, c, "ROAD", props, true).unwrap();

        let config = AStarConfig::new().edge_type("EDGE");
        let result = engine.astar_path(a, c, &config).unwrap();
        assert!(!result.found());
    }

    #[test]
    fn test_astar_incoming_direction() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        create_weighted_edge(&engine, b, a, 1.0);
        create_weighted_edge(&engine, c, b, 1.0);

        let config = AStarConfig::new().direction(Direction::Incoming);
        let result = engine.astar_path(a, c, &config).unwrap();
        assert!(result.found());
        assert_eq!(result.path_length(), Some(2));
    }

    #[test]
    fn test_astar_both_direction() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 1.0);
        create_weighted_edge(&engine, c, b, 1.0);

        let config = AStarConfig::new().direction(Direction::Both);
        let result = engine.astar_path(a, c, &config).unwrap();
        assert!(result.found());
        assert_eq!(result.path_length(), Some(2));
    }

    #[test]
    fn test_astar_custom_heuristic() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 1.0);
        create_weighted_edge(&engine, b, c, 1.0);

        let heuristic: HeuristicFn = Box::new(|_src, _dst, _engine| 0.5);
        let config = AStarConfig::new().heuristic(heuristic);
        let result = engine.astar_path(a, c, &config).unwrap();
        assert!(result.found());
        assert!((result.total_weight().unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_astar_euclidean_missing_coords() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 1.0);

        let result = engine.astar_path_euclidean(a, b, "x", "y").unwrap();
        assert!(result.found());
    }

    #[test]
    fn test_astar_manhattan_missing_coords() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 1.0);

        let result = engine.astar_path_manhattan(a, b, "x", "y").unwrap();
        assert!(result.found());
    }

    #[test]
    fn test_astar_euclidean_with_int_coords() {
        let engine = GraphEngine::new();

        let mut props_a = HashMap::new();
        props_a.insert("x".to_string(), PropertyValue::Int(0));
        props_a.insert("y".to_string(), PropertyValue::Int(0));
        let a = engine.create_node("Point", props_a).unwrap();

        let mut props_b = HashMap::new();
        props_b.insert("x".to_string(), PropertyValue::Int(3));
        props_b.insert("y".to_string(), PropertyValue::Int(4));
        let b = engine.create_node("Point", props_b).unwrap();

        create_weighted_edge(&engine, a, b, 5.0);

        let result = engine.astar_path_euclidean(a, b, "x", "y").unwrap();
        assert!(result.found());
    }

    #[test]
    fn test_astar_manhattan_with_int_coords() {
        let engine = GraphEngine::new();

        let mut props_a = HashMap::new();
        props_a.insert("x".to_string(), PropertyValue::Int(0));
        props_a.insert("y".to_string(), PropertyValue::Int(0));
        let a = engine.create_node("Point", props_a).unwrap();

        let mut props_b = HashMap::new();
        props_b.insert("x".to_string(), PropertyValue::Int(3));
        props_b.insert("y".to_string(), PropertyValue::Int(4));
        let b = engine.create_node("Point", props_b).unwrap();

        create_weighted_edge(&engine, a, b, 7.0);

        let result = engine.astar_path_manhattan(a, b, "x", "y").unwrap();
        assert!(result.found());
    }

    #[test]
    fn test_astar_source_not_found() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.astar_path(9999, a, &AStarConfig::new()).unwrap();
        assert!(!result.found());
    }

    #[test]
    fn test_astar_multiple_paths() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 1.0);
        create_weighted_edge(&engine, b, d, 1.0);
        create_weighted_edge(&engine, a, c, 1.0);
        create_weighted_edge(&engine, c, d, 1.0);

        let result = engine.astar_path(a, d, &AStarConfig::new()).unwrap();
        assert!(result.found());
        assert_eq!(result.path_length(), Some(2));
    }

    #[test]
    fn test_astar_dijkstra_fallback() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 3.0);

        let config = AStarConfig::new();
        let result = engine.astar_path(a, b, &config).unwrap();
        assert!(result.found());
    }
}
