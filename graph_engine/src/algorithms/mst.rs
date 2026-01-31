//! Minimum Spanning Tree using Kruskal's algorithm.
//!
//! Computes the minimum spanning tree (or forest) of a graph using edge weights.

#![allow(clippy::cast_precision_loss)] // Acceptable for graph algorithm metrics

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{GraphEngine, PropertyValue, Result};

/// Configuration for MST computation.
#[derive(Debug, Clone)]
pub struct MstConfig {
    /// Property name to use as edge weight.
    pub weight_property: String,
    /// Default weight for edges without the weight property.
    pub default_weight: f64,
    /// Whether to compute MST for each connected component (forest).
    pub compute_forest: bool,
}

impl Default for MstConfig {
    fn default() -> Self {
        Self {
            weight_property: "weight".to_string(),
            default_weight: 1.0,
            compute_forest: true,
        }
    }
}

impl MstConfig {
    #[must_use]
    pub fn new(weight_property: impl Into<String>) -> Self {
        Self {
            weight_property: weight_property.into(),
            ..Self::default()
        }
    }

    #[must_use]
    pub const fn default_weight(mut self, weight: f64) -> Self {
        self.default_weight = weight;
        self
    }

    #[must_use]
    pub const fn compute_forest(mut self, compute: bool) -> Self {
        self.compute_forest = compute;
        self
    }
}

/// An edge in the MST result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MstEdge {
    pub edge_id: u64,
    pub from: u64,
    pub to: u64,
    pub weight: f64,
}

/// Result of MST computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MstResult {
    /// Edges in the minimum spanning tree (or forest).
    pub edges: Vec<MstEdge>,
    /// Total weight of the MST.
    pub total_weight: f64,
    /// Number of trees in the forest (1 for connected graphs).
    pub tree_count: usize,
    /// Nodes included in the MST.
    pub nodes: Vec<u64>,
}

impl MstResult {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            edges: Vec::new(),
            total_weight: 0.0,
            tree_count: 0,
            nodes: Vec::new(),
        }
    }

    #[must_use]
    pub const fn is_connected(&self) -> bool {
        self.tree_count == 1
    }

    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

impl Default for MstResult {
    fn default() -> Self {
        Self::empty()
    }
}

/// Union-Find data structure for Kruskal's algorithm.
struct UnionFind {
    parent: HashMap<u64, u64>,
    rank: HashMap<u64, usize>,
}

impl UnionFind {
    fn new(nodes: &[u64]) -> Self {
        let parent = nodes.iter().map(|&n| (n, n)).collect();
        let rank = nodes.iter().map(|&n| (n, 0)).collect();
        Self { parent, rank }
    }

    fn find(&mut self, x: u64) -> u64 {
        let p = self.parent[&x];
        if p == x {
            x
        } else {
            let root = self.find(p);
            self.parent.insert(x, root);
            root
        }
    }

    fn union(&mut self, x: u64, y: u64) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false; // Already in same set
        }

        let rank_x = self.rank[&rx];
        let rank_y = self.rank[&ry];

        match rank_x.cmp(&rank_y) {
            std::cmp::Ordering::Less => {
                self.parent.insert(rx, ry);
            },
            std::cmp::Ordering::Greater => {
                self.parent.insert(ry, rx);
            },
            std::cmp::Ordering::Equal => {
                self.parent.insert(ry, rx);
                self.rank.insert(rx, rank_x + 1);
            },
        }
        true
    }
}

impl GraphEngine {
    /// Compute the minimum spanning tree (or forest) using Kruskal's algorithm.
    ///
    /// Time complexity: O(E log E) for sorting edges.
    ///
    /// # Errors
    ///
    /// Returns an error if edge retrieval fails.
    pub fn minimum_spanning_tree(&self, config: &MstConfig) -> Result<MstResult> {
        let nodes = self.get_all_node_ids()?;
        if nodes.is_empty() {
            return Ok(MstResult::empty());
        }

        // Collect all edges with weights
        let mut weighted_edges: Vec<(u64, u64, u64, f64)> = Vec::new(); // (from, to, edge_id, weight)

        for key in self.store().scan("edge:") {
            if let Some(id_str) = key.strip_prefix("edge:") {
                if let Ok(edge_id) = id_str.parse::<u64>() {
                    if let Ok(edge) = self.get_edge(edge_id) {
                        let weight = match edge.properties.get(&config.weight_property) {
                            Some(PropertyValue::Float(w)) => *w,
                            Some(PropertyValue::Int(w)) => *w as f64,
                            _ => config.default_weight,
                        };
                        weighted_edges.push((edge.from, edge.to, edge_id, weight));
                    }
                }
            }
        }

        // Sort edges by weight
        weighted_edges.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

        // Kruskal's algorithm
        let mut uf = UnionFind::new(&nodes);
        let mut mst_edges = Vec::new();
        let mut total_weight = 0.0;

        for (from, to, edge_id, weight) in weighted_edges {
            if uf.union(from, to) {
                mst_edges.push(MstEdge {
                    edge_id,
                    from,
                    to,
                    weight,
                });
                total_weight += weight;

                // Early termination if not computing forest
                if !config.compute_forest && mst_edges.len() == nodes.len() - 1 {
                    break;
                }
            }
        }

        // Count number of trees (connected components)
        let mut roots = std::collections::HashSet::new();
        for &node in &nodes {
            roots.insert(uf.find(node));
        }
        let tree_count = roots.len();

        Ok(MstResult {
            edges: mst_edges,
            total_weight,
            tree_count,
            nodes,
        })
    }

    /// Compute minimum spanning forest (MST for each connected component).
    ///
    /// # Errors
    ///
    /// Returns an error if MST computation fails.
    pub fn minimum_spanning_forest(&self, weight_property: &str) -> Result<Vec<MstResult>> {
        let result =
            self.minimum_spanning_tree(&MstConfig::new(weight_property).compute_forest(true))?;

        if result.tree_count <= 1 {
            return Ok(vec![result]);
        }

        // Group edges by component
        let mut uf = UnionFind::new(&result.nodes);
        for edge in &result.edges {
            uf.union(edge.from, edge.to);
        }

        let mut components: HashMap<u64, Vec<MstEdge>> = HashMap::new();
        let mut component_nodes: HashMap<u64, Vec<u64>> = HashMap::new();

        for edge in result.edges {
            let root = uf.find(edge.from);
            components.entry(root).or_default().push(edge);
        }

        for &node in &result.nodes {
            let root = uf.find(node);
            component_nodes.entry(root).or_default().push(node);
        }

        let mut forests = Vec::new();
        for (root, edges) in components {
            let total_weight = edges.iter().map(|e| e.weight).sum();
            let nodes = component_nodes.remove(&root).unwrap_or_default();
            forests.push(MstResult {
                edges,
                total_weight,
                tree_count: 1,
                nodes,
            });
        }

        // Add isolated nodes as separate trees
        for (_, nodes) in component_nodes {
            for node in nodes {
                forests.push(MstResult {
                    edges: Vec::new(),
                    total_weight: 0.0,
                    tree_count: 1,
                    nodes: vec![node],
                });
            }
        }

        Ok(forests)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_weighted_edge(engine: &GraphEngine, from: u64, to: u64, weight: f64) -> u64 {
        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Float(weight));
        engine.create_edge(from, to, "EDGE", props, false).unwrap()
    }

    #[test]
    fn test_mst_empty_graph() {
        let engine = GraphEngine::new();
        let result = engine.minimum_spanning_tree(&MstConfig::default()).unwrap();
        assert!(result.edges.is_empty());
        assert_eq!(result.tree_count, 0);
    }

    #[test]
    fn test_mst_single_node() {
        let engine = GraphEngine::new();
        engine.create_node("A", HashMap::new()).unwrap();

        let result = engine.minimum_spanning_tree(&MstConfig::default()).unwrap();
        assert!(result.edges.is_empty());
        assert_eq!(result.tree_count, 1);
        assert_eq!(result.nodes.len(), 1);
    }

    #[test]
    fn test_mst_simple_triangle() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Triangle with weights: A-B:1, B-C:2, A-C:3
        create_weighted_edge(&engine, a, b, 1.0);
        create_weighted_edge(&engine, b, c, 2.0);
        create_weighted_edge(&engine, a, c, 3.0);

        let result = engine
            .minimum_spanning_tree(&MstConfig::new("weight"))
            .unwrap();

        assert_eq!(result.edge_count(), 2); // MST has n-1 edges
        assert!((result.total_weight - 3.0).abs() < f64::EPSILON); // 1 + 2 = 3
        assert!(result.is_connected());
    }

    #[test]
    fn test_mst_selects_minimum_edges() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // Create a graph where MST should pick specific edges
        create_weighted_edge(&engine, a, b, 1.0);
        create_weighted_edge(&engine, b, c, 2.0);
        create_weighted_edge(&engine, c, d, 3.0);
        create_weighted_edge(&engine, a, d, 10.0); // Should not be selected

        let result = engine
            .minimum_spanning_tree(&MstConfig::new("weight"))
            .unwrap();

        assert_eq!(result.edge_count(), 3);
        assert!((result.total_weight - 6.0).abs() < f64::EPSILON); // 1 + 2 + 3
    }

    #[test]
    fn test_mst_forest() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // Two disconnected components
        create_weighted_edge(&engine, a, b, 1.0);
        create_weighted_edge(&engine, c, d, 2.0);

        let result = engine
            .minimum_spanning_tree(&MstConfig::new("weight"))
            .unwrap();

        assert_eq!(result.edge_count(), 2);
        assert_eq!(result.tree_count, 2);
        assert!(!result.is_connected());
    }

    #[test]
    fn test_mst_forest_split() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        create_weighted_edge(&engine, a, b, 1.0);
        create_weighted_edge(&engine, c, d, 2.0);

        let forests = engine.minimum_spanning_forest("weight").unwrap();
        assert_eq!(forests.len(), 2);
    }

    #[test]
    fn test_mst_default_weight() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        // Edge without weight property
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = MstConfig::new("weight").default_weight(5.0);
        let result = engine.minimum_spanning_tree(&config).unwrap();

        assert_eq!(result.edge_count(), 1);
        assert!((result.total_weight - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mst_integer_weight() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        let mut props = HashMap::new();
        props.insert("weight".to_string(), PropertyValue::Int(42));
        engine.create_edge(a, b, "EDGE", props, false).unwrap();

        let result = engine
            .minimum_spanning_tree(&MstConfig::new("weight"))
            .unwrap();
        assert!((result.total_weight - 42.0).abs() < f64::EPSILON);
    }
}
