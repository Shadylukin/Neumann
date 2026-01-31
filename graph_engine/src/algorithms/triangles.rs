//! Triangle counting and clustering coefficient algorithms.
//!
//! Triangles are cycles of length 3 in a graph. Counting triangles is useful
//! for measuring graph density and computing clustering coefficients.

#![allow(clippy::cast_precision_loss)] // Acceptable for graph algorithm metrics

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{Direction, GraphEngine, Result};

/// Configuration for triangle counting.
#[derive(Debug, Clone, Default)]
pub struct TriangleConfig {
    /// Edge type filter. None means all edge types.
    pub edge_type: Option<String>,
    /// Whether to treat the graph as undirected.
    pub undirected: bool,
}

impl TriangleConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.edge_type = Some(edge_type.into());
        self
    }

    #[must_use]
    pub const fn undirected(mut self) -> Self {
        self.undirected = true;
        self
    }
}

/// Result of triangle counting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TriangleResult {
    /// Total number of triangles in the graph.
    pub triangle_count: usize,
    /// Number of triangles each node participates in.
    pub node_triangles: HashMap<u64, usize>,
    /// Global clustering coefficient.
    pub global_clustering: f64,
    /// Local clustering coefficient for each node.
    pub local_clustering: HashMap<u64, f64>,
}

impl TriangleResult {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            triangle_count: 0,
            node_triangles: HashMap::new(),
            global_clustering: 0.0,
            local_clustering: HashMap::new(),
        }
    }

    #[must_use]
    pub fn average_clustering(&self) -> f64 {
        if self.local_clustering.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.local_clustering.values().sum();
        sum / self.local_clustering.len() as f64
    }

    #[must_use]
    pub fn nodes_in_triangles(&self) -> Vec<u64> {
        self.node_triangles
            .iter()
            .filter(|(_, &count)| count > 0)
            .map(|(&id, _)| id)
            .collect()
    }
}

impl Default for TriangleResult {
    fn default() -> Self {
        Self::empty()
    }
}

impl GraphEngine {
    /// Count triangles in the graph using the forward algorithm.
    ///
    /// Time complexity: O(E^1.5) for sparse graphs.
    ///
    /// # Errors
    ///
    /// Returns an error if node/edge retrieval fails.
    pub fn count_triangles(&self, config: &TriangleConfig) -> Result<TriangleResult> {
        let nodes = self.get_all_node_ids()?;
        if nodes.is_empty() {
            return Ok(TriangleResult::empty());
        }

        // Build adjacency sets for efficient lookup
        let mut adjacency: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut degrees: HashMap<u64, usize> = HashMap::new();

        for &node in &nodes {
            let neighbors = self.get_triangle_neighbor_ids(node, config)?;
            degrees.insert(node, neighbors.len());
            adjacency.insert(node, neighbors.into_iter().collect());
        }

        // Count triangles using forward algorithm
        // For each edge (u, v) where degree(u) < degree(v), check common neighbors
        let mut triangle_count = 0;
        let mut node_triangles: HashMap<u64, usize> = nodes.iter().map(|&n| (n, 0)).collect();
        let mut counted_edges: HashSet<(u64, u64)> = HashSet::new();

        for &u in &nodes {
            let Some(u_neighbors) = adjacency.get(&u) else {
                continue;
            };

            for &v in u_neighbors {
                // Only process each edge once: use degree ordering
                let u_deg = degrees.get(&u).copied().unwrap_or(0);
                let v_deg = degrees.get(&v).copied().unwrap_or(0);

                let edge_key = if u < v { (u, v) } else { (v, u) };
                if counted_edges.contains(&edge_key) {
                    continue;
                }

                // Process edge if u has lower degree (or same degree but lower ID)
                if u_deg > v_deg || (u_deg == v_deg && u > v) {
                    continue;
                }

                counted_edges.insert(edge_key);

                let Some(v_neighbors) = adjacency.get(&v) else {
                    continue;
                };

                // Find common neighbors (complete the triangle)
                // Require w > v to ensure each triangle is counted exactly once
                for &w in u_neighbors {
                    if w > v && v_neighbors.contains(&w) {
                        // Found triangle (u, v, w) where u < v < w
                        triangle_count += 1;
                        *node_triangles.entry(u).or_insert(0) += 1;
                        *node_triangles.entry(v).or_insert(0) += 1;
                        *node_triangles.entry(w).or_insert(0) += 1;
                    }
                }
            }
        }

        // Compute clustering coefficients
        let mut local_clustering: HashMap<u64, f64> = HashMap::new();
        let mut total_triplets = 0usize;

        for &node in &nodes {
            let degree = degrees.get(&node).copied().unwrap_or(0);
            if degree < 2 {
                local_clustering.insert(node, 0.0);
                continue;
            }

            // Number of possible triangles through this node
            let possible_triangles = degree * (degree - 1) / 2;
            total_triplets += possible_triangles;

            let actual_triangles = node_triangles.get(&node).copied().unwrap_or(0);
            let coefficient = if possible_triangles > 0 {
                actual_triangles as f64 / possible_triangles as f64
            } else {
                0.0
            };
            local_clustering.insert(node, coefficient);
        }

        // Global clustering coefficient
        let global_clustering = if total_triplets > 0 {
            (3 * triangle_count) as f64 / total_triplets as f64
        } else {
            0.0
        };

        Ok(TriangleResult {
            triangle_count,
            node_triangles,
            global_clustering,
            local_clustering,
        })
    }

    /// Compute local clustering coefficient for a single node.
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn local_clustering_coefficient(
        &self,
        node_id: u64,
        config: &TriangleConfig,
    ) -> Result<f64> {
        let neighbors = self.get_triangle_neighbor_ids(node_id, config)?;
        let degree = neighbors.len();

        if degree < 2 {
            return Ok(0.0);
        }

        // Count edges among neighbors
        let neighbor_set: HashSet<u64> = neighbors.iter().copied().collect();
        let mut edges_among_neighbors = 0;

        for &n1 in &neighbors {
            let n1_neighbors = self.get_triangle_neighbor_ids(n1, config)?;
            for n2 in n1_neighbors {
                if neighbor_set.contains(&n2) && n1 < n2 {
                    edges_among_neighbors += 1;
                }
            }
        }

        let possible_edges = degree * (degree - 1) / 2;
        #[allow(clippy::cast_precision_loss)] // Acceptable precision loss
        Ok(f64::from(edges_among_neighbors) / possible_edges as f64)
    }

    /// Compute global clustering coefficient.
    ///
    /// # Errors
    ///
    /// Returns an error if triangle counting fails.
    pub fn global_clustering_coefficient(&self, config: &TriangleConfig) -> Result<f64> {
        let result = self.count_triangles(config)?;
        Ok(result.global_clustering)
    }

    /// Get neighbor IDs based on triangle config.
    fn get_triangle_neighbor_ids(&self, node_id: u64, config: &TriangleConfig) -> Result<Vec<u64>> {
        let direction = if config.undirected {
            Direction::Both
        } else {
            Direction::Outgoing
        };

        let neighbors = self.neighbors(node_id, config.edge_type.as_deref(), direction, None)?;
        Ok(neighbors.into_iter().map(|n| n.id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_empty_graph() {
        let engine = GraphEngine::new();
        let result = engine.count_triangles(&TriangleConfig::default()).unwrap();
        assert_eq!(result.triangle_count, 0);
    }

    #[test]
    fn test_triangle_single_triangle() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Create triangle: A -> B -> C -> A
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(c, a, "EDGE", HashMap::new(), false)
            .unwrap();

        let result = engine
            .count_triangles(&TriangleConfig::new().undirected())
            .unwrap();
        assert_eq!(result.triangle_count, 1);

        // Each node participates in 1 triangle
        assert_eq!(result.node_triangles.get(&a), Some(&1));
        assert_eq!(result.node_triangles.get(&b), Some(&1));
        assert_eq!(result.node_triangles.get(&c), Some(&1));
    }

    #[test]
    fn test_triangle_no_triangle() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Line: A -> B -> C (no triangle)
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let result = engine
            .count_triangles(&TriangleConfig::new().undirected())
            .unwrap();
        assert_eq!(result.triangle_count, 0);
    }

    #[test]
    fn test_clustering_coefficient_complete_graph() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Complete graph K3 (fully connected)
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = TriangleConfig::new().undirected();
        let coeff = engine.local_clustering_coefficient(a, &config).unwrap();

        // In a complete graph, clustering coefficient is 1.0
        assert!((coeff - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clustering_coefficient_star_graph() {
        let engine = GraphEngine::new();
        let center = engine.create_node("Center", HashMap::new()).unwrap();
        let leaf1 = engine.create_node("Leaf1", HashMap::new()).unwrap();
        let leaf2 = engine.create_node("Leaf2", HashMap::new()).unwrap();
        let leaf3 = engine.create_node("Leaf3", HashMap::new()).unwrap();

        // Star graph: center connected to all leaves, leaves not connected
        engine
            .create_edge(center, leaf1, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(center, leaf2, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(center, leaf3, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = TriangleConfig::new().undirected();
        let coeff = engine
            .local_clustering_coefficient(center, &config)
            .unwrap();

        // Center has degree 3 but no edges among neighbors
        assert!(coeff.abs() < f64::EPSILON);
    }

    #[test]
    fn test_global_clustering_coefficient() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // Create a graph with one triangle (a,b,c) and one additional edge to d
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(c, d, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = TriangleConfig::new().undirected();
        let coeff = engine.global_clustering_coefficient(&config).unwrap();

        // Global clustering should be > 0 (we have one triangle)
        assert!(coeff > 0.0);
    }

    #[test]
    fn test_nodes_in_triangles() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // Triangle: a, b, c
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();

        // d is isolated from triangle
        let result = engine
            .count_triangles(&TriangleConfig::new().undirected())
            .unwrap();
        let in_triangles = result.nodes_in_triangles();

        assert!(in_triangles.contains(&a));
        assert!(in_triangles.contains(&b));
        assert!(in_triangles.contains(&c));
        assert!(!in_triangles.contains(&d));
    }

    #[test]
    fn test_average_clustering() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let result = engine
            .count_triangles(&TriangleConfig::new().undirected())
            .unwrap();
        let avg = result.average_clustering();

        // Complete graph K3 has average clustering 1.0
        assert!((avg - 1.0).abs() < f64::EPSILON);
    }
}
