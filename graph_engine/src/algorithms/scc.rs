//! Strongly Connected Components using Tarjan's algorithm.
//!
//! A strongly connected component (SCC) is a maximal set of vertices such that
//! there is a path from every vertex to every other vertex in the set.

#![allow(clippy::type_complexity)] // Complex return types are acceptable for internal methods

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{Direction, GraphEngine, Result};

/// Configuration for SCC computation.
#[derive(Debug, Clone, Default)]
pub struct SccConfig {
    /// Edge type filter. None means all edge types.
    pub edge_type: Option<String>,
    /// Whether to compute the condensation DAG.
    pub compute_condensation: bool,
}

impl SccConfig {
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
    pub const fn with_condensation(mut self) -> Self {
        self.compute_condensation = true;
        self
    }
}

/// Result of SCC computation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SccResult {
    /// Maps each node ID to its component ID.
    pub components: HashMap<u64, usize>,
    /// List of nodes in each component (indexed by component ID).
    pub members: Vec<Vec<u64>>,
    /// Number of strongly connected components.
    pub component_count: usize,
    /// Edges in the condensation DAG: (`from_component`, `to_component`).
    /// Only populated if `compute_condensation` was set.
    pub condensation_edges: Vec<(usize, usize)>,
    /// Topological order of components (if condensation computed).
    pub topological_order: Vec<usize>,
}

impl SccResult {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            components: HashMap::new(),
            members: Vec::new(),
            component_count: 0,
            condensation_edges: Vec::new(),
            topological_order: Vec::new(),
        }
    }

    #[must_use]
    pub const fn is_strongly_connected(&self) -> bool {
        self.component_count == 1
    }

    #[must_use]
    pub fn largest_component(&self) -> Option<&[u64]> {
        self.members
            .iter()
            .max_by_key(|m| m.len())
            .map(Vec::as_slice)
    }

    #[must_use]
    pub fn components_by_size(&self) -> Vec<(usize, usize)> {
        let mut sizes: Vec<_> = self
            .members
            .iter()
            .enumerate()
            .map(|(i, m)| (i, m.len()))
            .collect();
        sizes.sort_by(|a, b| b.1.cmp(&a.1));
        sizes
    }
}

impl Default for SccResult {
    fn default() -> Self {
        Self::empty()
    }
}

/// Internal state for Tarjan's algorithm.
struct TarjanState {
    index: usize,
    indices: HashMap<u64, usize>,
    low_links: HashMap<u64, usize>,
    on_stack: HashMap<u64, bool>,
    stack: Vec<u64>,
    components: Vec<Vec<u64>>,
}

impl TarjanState {
    fn new() -> Self {
        Self {
            index: 0,
            indices: HashMap::new(),
            low_links: HashMap::new(),
            on_stack: HashMap::new(),
            stack: Vec::new(),
            components: Vec::new(),
        }
    }
}

impl GraphEngine {
    /// Compute strongly connected components using Tarjan's algorithm.
    ///
    /// Time complexity: O(V + E)
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn strongly_connected_components(&self, config: &SccConfig) -> Result<SccResult> {
        let nodes = self.get_all_node_ids()?;
        if nodes.is_empty() {
            return Ok(SccResult::empty());
        }

        let mut state = TarjanState::new();

        // Run Tarjan's algorithm
        for &node in &nodes {
            if !state.indices.contains_key(&node) {
                self.tarjan_strongconnect(node, config, &mut state)?;
            }
        }

        // Build result
        let component_count = state.components.len();
        let mut node_to_component = HashMap::new();
        for (comp_id, members) in state.components.iter().enumerate() {
            for &node in members {
                node_to_component.insert(node, comp_id);
            }
        }

        // Compute condensation DAG if requested
        let (condensation_edges, topological_order) = if config.compute_condensation {
            self.compute_condensation(&node_to_component, component_count, config)?
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(SccResult {
            components: node_to_component,
            members: state.components,
            component_count,
            condensation_edges,
            topological_order,
        })
    }

    /// Tarjan's strongconnect function (recursive).
    fn tarjan_strongconnect(
        &self,
        v: u64,
        config: &SccConfig,
        state: &mut TarjanState,
    ) -> Result<()> {
        // Set the depth index for v
        state.indices.insert(v, state.index);
        state.low_links.insert(v, state.index);
        state.index += 1;
        state.stack.push(v);
        state.on_stack.insert(v, true);

        // Get successors
        let neighbors =
            self.neighbors(v, config.edge_type.as_deref(), Direction::Outgoing, None)?;

        for neighbor in neighbors {
            let w = neighbor.id;
            if !state.indices.contains_key(&w) {
                // Successor w has not yet been visited; recurse on it
                self.tarjan_strongconnect(w, config, state)?;
                let low_w = state.low_links[&w];
                let low_v = state.low_links[&v];
                state.low_links.insert(v, low_v.min(low_w));
            } else if state.on_stack.get(&w).copied().unwrap_or(false) {
                // Successor w is on stack and hence in the current SCC
                let idx_w = state.indices[&w];
                let low_v = state.low_links[&v];
                state.low_links.insert(v, low_v.min(idx_w));
            }
        }

        // If v is a root node, pop the stack and generate an SCC
        if state.low_links[&v] == state.indices[&v] {
            let mut component = Vec::new();
            loop {
                let w = state.stack.pop().expect("Stack should not be empty");
                state.on_stack.insert(w, false);
                component.push(w);
                if w == v {
                    break;
                }
            }
            state.components.push(component);
        }

        Ok(())
    }

    /// Compute the condensation DAG.
    fn compute_condensation(
        &self,
        node_to_component: &HashMap<u64, usize>,
        component_count: usize,
        config: &SccConfig,
    ) -> Result<(Vec<(usize, usize)>, Vec<usize>)> {
        use std::collections::HashSet;

        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        let mut in_degree = vec![0usize; component_count];

        // Find all edges between components
        for (&node, &from_comp) in node_to_component {
            let neighbors =
                self.neighbors(node, config.edge_type.as_deref(), Direction::Outgoing, None)?;
            for neighbor in neighbors {
                if let Some(&to_comp) = node_to_component.get(&neighbor.id) {
                    if from_comp != to_comp && edges.insert((from_comp, to_comp)) {
                        in_degree[to_comp] += 1;
                    }
                }
            }
        }

        // Topological sort using Kahn's algorithm
        let mut queue: Vec<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();
        let mut topo_order = Vec::with_capacity(component_count);
        let mut local_in_degree = in_degree;

        while let Some(comp) = queue.pop() {
            topo_order.push(comp);
            for &(from, to) in &edges {
                if from == comp {
                    local_in_degree[to] -= 1;
                    if local_in_degree[to] == 0 {
                        queue.push(to);
                    }
                }
            }
        }

        Ok((edges.into_iter().collect(), topo_order))
    }

    /// Check if the graph is strongly connected.
    ///
    /// # Errors
    ///
    /// Returns an error if SCC computation fails.
    pub fn is_strongly_connected(&self) -> Result<bool> {
        let result = self.strongly_connected_components(&SccConfig::default())?;
        Ok(result.is_strongly_connected())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_scc_single_node() {
        let engine = GraphEngine::new();
        engine.create_node("A", HashMap::new()).unwrap();

        let result = engine
            .strongly_connected_components(&SccConfig::default())
            .unwrap();
        assert_eq!(result.component_count, 1);
        assert!(result.is_strongly_connected());
    }

    #[test]
    fn test_scc_two_disconnected_nodes() {
        let engine = GraphEngine::new();
        engine.create_node("A", HashMap::new()).unwrap();
        engine.create_node("B", HashMap::new()).unwrap();

        let result = engine
            .strongly_connected_components(&SccConfig::default())
            .unwrap();
        assert_eq!(result.component_count, 2);
        assert!(!result.is_strongly_connected());
    }

    #[test]
    fn test_scc_simple_cycle() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Create cycle: A -> B -> C -> A
        engine
            .create_edge(a, b, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b, c, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(c, a, "NEXT", HashMap::new(), true)
            .unwrap();

        let result = engine
            .strongly_connected_components(&SccConfig::default())
            .unwrap();
        assert_eq!(result.component_count, 1);
        assert!(result.is_strongly_connected());
    }

    #[test]
    fn test_scc_two_cycles() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // Cycle 1: A -> B -> A
        engine
            .create_edge(a, b, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b, a, "NEXT", HashMap::new(), true)
            .unwrap();

        // Cycle 2: C -> D -> C
        engine
            .create_edge(c, d, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(d, c, "NEXT", HashMap::new(), true)
            .unwrap();

        // Connection from cycle 1 to cycle 2
        engine
            .create_edge(b, c, "NEXT", HashMap::new(), true)
            .unwrap();

        let result = engine
            .strongly_connected_components(&SccConfig::default())
            .unwrap();
        assert_eq!(result.component_count, 2);
        assert!(!result.is_strongly_connected());
    }

    #[test]
    fn test_scc_with_condensation() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // A -> B -> C (linear, no cycles)
        engine
            .create_edge(a, b, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b, c, "NEXT", HashMap::new(), true)
            .unwrap();

        let config = SccConfig::new().with_condensation();
        let result = engine.strongly_connected_components(&config).unwrap();

        assert_eq!(result.component_count, 3);
        assert_eq!(result.condensation_edges.len(), 2);
        assert_eq!(result.topological_order.len(), 3);
    }

    #[test]
    fn test_is_strongly_connected() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        // One-way edge
        engine
            .create_edge(a, b, "NEXT", HashMap::new(), true)
            .unwrap();

        assert!(!engine.is_strongly_connected().unwrap());

        // Add return edge to make it strongly connected
        engine
            .create_edge(b, a, "NEXT", HashMap::new(), true)
            .unwrap();

        assert!(engine.is_strongly_connected().unwrap());
    }

    #[test]
    fn test_scc_empty_graph() {
        let engine = GraphEngine::new();
        let result = engine
            .strongly_connected_components(&SccConfig::default())
            .unwrap();
        assert_eq!(result.component_count, 0);
        assert!(result.members.is_empty());
    }

    #[test]
    fn test_scc_largest_component() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let _d = engine.create_node("D", HashMap::new()).unwrap();

        // Large cycle: A -> B -> C -> A
        engine
            .create_edge(a, b, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(b, c, "NEXT", HashMap::new(), true)
            .unwrap();
        engine
            .create_edge(c, a, "NEXT", HashMap::new(), true)
            .unwrap();

        // D is isolated
        let result = engine
            .strongly_connected_components(&SccConfig::default())
            .unwrap();

        let largest = result.largest_component().unwrap();
        assert_eq!(largest.len(), 3);
    }
}
