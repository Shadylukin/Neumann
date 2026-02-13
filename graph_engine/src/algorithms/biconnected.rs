//! Biconnected components, articulation points, and bridges.
//!
//! - Articulation point: A node whose removal disconnects the graph
//! - Bridge: An edge whose removal disconnects the graph
//! - Biconnected component: A maximal subgraph with no articulation points

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{Direction, GraphEngine, Result};

/// Configuration for biconnected component analysis.
#[derive(Debug, Clone, Default)]
pub struct BiconnectedConfig {
    /// Edge type filter. None means all edge types.
    pub edge_type: Option<String>,
}

impl BiconnectedConfig {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.edge_type = Some(edge_type.into());
        self
    }
}

/// Result of biconnected component analysis.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BiconnectedResult {
    /// Articulation points (cut vertices).
    pub articulation_points: Vec<u64>,
    /// Bridges (cut edges) as (from, to) pairs.
    pub bridges: Vec<(u64, u64)>,
    /// Biconnected components (each as a set of edges).
    pub components: Vec<HashSet<(u64, u64)>>,
    /// Number of biconnected components.
    pub component_count: usize,
}

impl BiconnectedResult {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            articulation_points: Vec::new(),
            bridges: Vec::new(),
            components: Vec::new(),
            component_count: 0,
        }
    }

    #[must_use]
    pub const fn is_biconnected(&self) -> bool {
        self.articulation_points.is_empty() && self.component_count <= 1
    }

    #[must_use]
    pub const fn has_bridges(&self) -> bool {
        !self.bridges.is_empty()
    }
}

impl Default for BiconnectedResult {
    fn default() -> Self {
        Self::empty()
    }
}

/// Internal state for Tarjan's biconnected components algorithm.
struct BiconnectedState {
    time: usize,
    discovery: HashMap<u64, usize>,
    low: HashMap<u64, usize>,
    parent: HashMap<u64, Option<u64>>,
    articulation_points: HashSet<u64>,
    bridges: Vec<(u64, u64)>,
    edge_stack: Vec<(u64, u64)>,
    components: Vec<HashSet<(u64, u64)>>,
}

impl BiconnectedState {
    fn new() -> Self {
        Self {
            time: 0,
            discovery: HashMap::new(),
            low: HashMap::new(),
            parent: HashMap::new(),
            articulation_points: HashSet::new(),
            bridges: Vec::new(),
            edge_stack: Vec::new(),
            components: Vec::new(),
        }
    }
}

impl GraphEngine {
    /// Find articulation points (cut vertices) in the graph.
    ///
    /// An articulation point is a vertex whose removal increases
    /// the number of connected components.
    ///
    /// Time complexity: O(V + E)
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn articulation_points(&self, config: &BiconnectedConfig) -> Result<Vec<u64>> {
        let result = self.biconnected_components(config)?;
        Ok(result.articulation_points)
    }

    /// Find bridges (cut edges) in the graph.
    ///
    /// A bridge is an edge whose removal increases the number
    /// of connected components.
    ///
    /// Time complexity: O(V + E)
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn bridges(&self, config: &BiconnectedConfig) -> Result<Vec<(u64, u64)>> {
        let result = self.biconnected_components(config)?;
        Ok(result.bridges)
    }

    /// Find biconnected components using Tarjan's algorithm.
    ///
    /// Time complexity: O(V + E)
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn biconnected_components(&self, config: &BiconnectedConfig) -> Result<BiconnectedResult> {
        let nodes = self.get_all_node_ids()?;
        if nodes.is_empty() {
            return Ok(BiconnectedResult::empty());
        }

        let mut state = BiconnectedState::new();

        // Run DFS from each unvisited node
        for &node in &nodes {
            if !state.discovery.contains_key(&node) {
                self.biconnected_dfs(node, config, &mut state)?;
            }
        }

        // Pop any remaining edges on stack
        if !state.edge_stack.is_empty() {
            let component: HashSet<(u64, u64)> = state.edge_stack.drain(..).collect();
            if !component.is_empty() {
                state.components.push(component);
            }
        }

        let component_count = state.components.len();

        Ok(BiconnectedResult {
            articulation_points: state.articulation_points.into_iter().collect(),
            bridges: state.bridges,
            components: state.components,
            component_count,
        })
    }

    /// DFS for biconnected components.
    fn biconnected_dfs(
        &self,
        u: u64,
        config: &BiconnectedConfig,
        state: &mut BiconnectedState,
    ) -> Result<()> {
        state.discovery.insert(u, state.time);
        state.low.insert(u, state.time);
        state.time += 1;

        let mut children = 0;
        let mut neighbors = self.get_undirected_neighbors(u, config)?;
        neighbors.sort_unstable();

        for v in neighbors {
            if !state.discovery.contains_key(&v) {
                children += 1;
                state.parent.insert(v, Some(u));
                state.edge_stack.push((u.min(v), u.max(v)));

                self.biconnected_dfs(v, config, state)?;

                // Update low value
                let low_v = state.low.get(&v).copied().unwrap_or(0);
                let low_u = state.low.get(&u).copied().unwrap_or(0);
                state.low.insert(u, low_u.min(low_v));

                // Check for articulation point
                let disc_u = state.discovery.get(&u).copied().unwrap_or(0);
                let parent_u = state.parent.get(&u).copied().flatten();

                // u is an articulation point if:
                // 1. u is root and has multiple children, or
                // 2. u is not root and low[v] >= disc[u]
                let should_pop = if parent_u.is_none() {
                    if children > 1 {
                        state.articulation_points.insert(u);
                        // Root with 2+ children: pop component for each
                        // child after the first (last child's edges stay
                        // on the stack and are collected at cleanup).
                        true
                    } else {
                        false
                    }
                } else if low_v >= disc_u {
                    state.articulation_points.insert(u);
                    true
                } else {
                    false
                };

                if should_pop {
                    let mut component = HashSet::new();
                    let edge = (u.min(v), u.max(v));
                    while let Some(e) = state.edge_stack.pop() {
                        component.insert(e);
                        if e == edge {
                            break;
                        }
                    }
                    if !component.is_empty() {
                        state.components.push(component);
                    }
                }

                // Check for bridge
                if low_v > disc_u {
                    state.bridges.push((u.min(v), u.max(v)));
                }
            } else if state.parent.get(&u).copied().flatten() != Some(v) {
                // Back edge
                let disc_v = state.discovery.get(&v).copied().unwrap_or(0);
                let low_u = state.low.get(&u).copied().unwrap_or(0);
                if disc_v < low_u {
                    state.low.insert(u, disc_v);
                    state.edge_stack.push((u.min(v), u.max(v)));
                }
            }
        }

        Ok(())
    }

    /// Get undirected neighbors (both directions).
    fn get_undirected_neighbors(
        &self,
        node_id: u64,
        config: &BiconnectedConfig,
    ) -> Result<Vec<u64>> {
        let neighbors =
            self.neighbors(node_id, config.edge_type.as_deref(), Direction::Both, None)?;
        Ok(neighbors.into_iter().map(|n| n.id).collect())
    }

    /// Check if the graph is biconnected.
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails.
    pub fn is_biconnected(&self, config: &BiconnectedConfig) -> Result<bool> {
        let result = self.biconnected_components(config)?;
        Ok(result.is_biconnected())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biconnected_empty_graph() {
        let engine = GraphEngine::new();
        let result = engine
            .biconnected_components(&BiconnectedConfig::default())
            .unwrap();
        assert!(result.articulation_points.is_empty());
        assert!(result.bridges.is_empty());
    }

    #[test]
    fn test_biconnected_single_node() {
        let engine = GraphEngine::new();
        engine.create_node("A", HashMap::new()).unwrap();

        let result = engine
            .biconnected_components(&BiconnectedConfig::new())
            .unwrap();
        assert!(result.articulation_points.is_empty());
        assert!(result.is_biconnected());
    }

    #[test]
    fn test_biconnected_simple_path() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Path: A - B - C
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let result = engine
            .biconnected_components(&BiconnectedConfig::new())
            .unwrap();

        // B is an articulation point
        assert!(result.articulation_points.contains(&b));

        // Both edges are bridges
        assert!(result.has_bridges());
        assert_eq!(result.bridges.len(), 2);
    }

    #[test]
    fn test_biconnected_triangle() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Triangle: fully connected
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
            .biconnected_components(&BiconnectedConfig::new())
            .unwrap();

        // No articulation points in a triangle
        assert!(result.articulation_points.is_empty());

        // No bridges in a triangle
        assert!(!result.has_bridges());

        // Triangle is biconnected
        assert!(result.is_biconnected());
    }

    #[test]
    fn test_biconnected_two_triangles_sharing_vertex() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();
        let e = engine.create_node("E", HashMap::new()).unwrap();

        // Triangle 1: a-b-c
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();

        // Triangle 2: c-d-e (sharing c with triangle 1)
        engine
            .create_edge(c, d, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(d, e, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(c, e, "EDGE", HashMap::new(), false)
            .unwrap();

        let result = engine
            .biconnected_components(&BiconnectedConfig::new())
            .unwrap();

        // c is an articulation point
        assert!(result.articulation_points.contains(&c));

        // No bridges (removing c splits into two triangles, but no single edge is a bridge)
        assert!(!result.has_bridges());

        // Two biconnected components
        assert_eq!(result.component_count, 2);
    }

    #[test]
    fn test_bridges() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();

        let bridges = engine.bridges(&BiconnectedConfig::new()).unwrap();

        // Single edge is a bridge
        assert_eq!(bridges.len(), 1);
    }

    #[test]
    fn test_articulation_points_only() {
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

        let aps = engine
            .articulation_points(&BiconnectedConfig::new())
            .unwrap();

        assert!(aps.contains(&b));
        assert!(!aps.contains(&a));
        assert!(!aps.contains(&c));
    }

    #[test]
    fn test_is_biconnected() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Path is not biconnected
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = BiconnectedConfig::new();
        assert!(!engine.is_biconnected(&config).unwrap());

        // Add edge to form a cycle
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();

        assert!(engine.is_biconnected(&config).unwrap());
    }

    #[test]
    fn test_star_graph_articulation() {
        let engine = GraphEngine::new();
        let center = engine.create_node("Center", HashMap::new()).unwrap();
        let leaf1 = engine.create_node("Leaf1", HashMap::new()).unwrap();
        let leaf2 = engine.create_node("Leaf2", HashMap::new()).unwrap();
        let leaf3 = engine.create_node("Leaf3", HashMap::new()).unwrap();

        engine
            .create_edge(center, leaf1, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(center, leaf2, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(center, leaf3, "EDGE", HashMap::new(), false)
            .unwrap();

        let result = engine
            .biconnected_components(&BiconnectedConfig::new())
            .unwrap();

        // Center is articulation point
        assert!(result.articulation_points.contains(&center));

        // All edges are bridges
        assert_eq!(result.bridges.len(), 3);
    }
}
