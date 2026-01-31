//! K-core decomposition algorithm.
//!
//! A k-core is a maximal subgraph where every node has degree at least k.
//! K-core decomposition assigns a core number to each node, indicating
//! the highest k for which that node is part of the k-core.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{Direction, GraphEngine, Result};

/// Configuration for k-core decomposition.
#[derive(Debug, Clone, Default)]
pub struct KCoreConfig {
    /// Edge type filter. None means all edge types.
    pub edge_type: Option<String>,
    /// Whether to treat the graph as undirected.
    pub undirected: bool,
}

impl KCoreConfig {
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

/// Result of k-core decomposition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KCoreResult {
    /// Core number for each node.
    pub core_numbers: HashMap<u64, usize>,
    /// The degeneracy (maximum core number) of the graph.
    pub degeneracy: usize,
    /// Nodes grouped by their core number.
    pub cores: HashMap<usize, Vec<u64>>,
}

impl KCoreResult {
    #[must_use]
    pub fn empty() -> Self {
        Self {
            core_numbers: HashMap::new(),
            degeneracy: 0,
            cores: HashMap::new(),
        }
    }

    #[must_use]
    pub fn get_kcore(&self, k: usize) -> Vec<u64> {
        self.core_numbers
            .iter()
            .filter(|(_, &core)| core >= k)
            .map(|(&id, _)| id)
            .collect()
    }

    #[must_use]
    pub fn core_of(&self, node_id: u64) -> Option<usize> {
        self.core_numbers.get(&node_id).copied()
    }

    #[must_use]
    pub fn shell(&self, k: usize) -> Vec<u64> {
        self.core_numbers
            .iter()
            .filter(|(_, &core)| core == k)
            .map(|(&id, _)| id)
            .collect()
    }
}

impl Default for KCoreResult {
    fn default() -> Self {
        Self::empty()
    }
}

impl GraphEngine {
    /// Compute k-core decomposition using the peeling algorithm.
    ///
    /// Time complexity: O(V + E)
    ///
    /// # Errors
    ///
    /// Returns an error if node/edge retrieval fails.
    pub fn kcore_decomposition(&self, config: &KCoreConfig) -> Result<KCoreResult> {
        let nodes = self.get_all_node_ids()?;
        if nodes.is_empty() {
            return Ok(KCoreResult::empty());
        }

        // Build adjacency lists and compute initial degrees
        let mut adjacency: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut degrees: HashMap<u64, usize> = HashMap::new();

        for &node in &nodes {
            let neighbors = self.get_neighbor_ids_for_kcore(node, config)?;
            let neighbor_set: HashSet<u64> = neighbors.into_iter().collect();
            degrees.insert(node, neighbor_set.len());
            adjacency.insert(node, neighbor_set);
        }

        // Priority queue with (degree, node_id) - process lowest degree first
        let mut pq: BinaryHeap<Reverse<(usize, u64)>> = nodes
            .iter()
            .map(|&n| Reverse((degrees.get(&n).copied().unwrap_or(0), n)))
            .collect();

        let mut core_numbers: HashMap<u64, usize> = HashMap::new();
        let mut processed: HashSet<u64> = HashSet::new();
        let mut current_core = 0;

        while let Some(Reverse((_deg, node))) = pq.pop() {
            if processed.contains(&node) {
                continue;
            }

            // The core number is max(current_core, node's degree at removal time)
            let actual_degree = degrees.get(&node).copied().unwrap_or(0);
            current_core = current_core.max(actual_degree);
            core_numbers.insert(node, current_core);
            processed.insert(node);

            // Update neighbors' degrees
            if let Some(neighbors) = adjacency.get(&node) {
                for &neighbor in neighbors {
                    if !processed.contains(&neighbor) {
                        if let Some(deg) = degrees.get_mut(&neighbor) {
                            if *deg > 0 {
                                *deg -= 1;
                                pq.push(Reverse((*deg, neighbor)));
                            }
                        }
                    }
                }
            }
        }

        // Find degeneracy and group by core
        let degeneracy = core_numbers.values().copied().max().unwrap_or(0);
        let mut cores: HashMap<usize, Vec<u64>> = HashMap::new();
        for (&node, &core) in &core_numbers {
            cores.entry(core).or_default().push(node);
        }

        Ok(KCoreResult {
            core_numbers,
            degeneracy,
            cores,
        })
    }

    /// Extract the k-core subgraph (all nodes with core number >= k).
    ///
    /// # Errors
    ///
    /// Returns an error if decomposition fails.
    pub fn kcore_subgraph(&self, k: usize, config: &KCoreConfig) -> Result<Vec<u64>> {
        let result = self.kcore_decomposition(config)?;
        Ok(result.get_kcore(k))
    }

    /// Get the degeneracy of the graph (maximum core number).
    ///
    /// # Errors
    ///
    /// Returns an error if decomposition fails.
    pub fn degeneracy(&self, config: &KCoreConfig) -> Result<usize> {
        let result = self.kcore_decomposition(config)?;
        Ok(result.degeneracy)
    }

    /// Get neighbor IDs for k-core computation.
    /// K-core always uses undirected degree (both directions).
    fn get_neighbor_ids_for_kcore(&self, node_id: u64, config: &KCoreConfig) -> Result<Vec<u64>> {
        let neighbors =
            self.neighbors(node_id, config.edge_type.as_deref(), Direction::Both, None)?;
        Ok(neighbors.into_iter().map(|n| n.id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kcore_empty_graph() {
        let engine = GraphEngine::new();
        let result = engine.kcore_decomposition(&KCoreConfig::default()).unwrap();
        assert_eq!(result.degeneracy, 0);
        assert!(result.core_numbers.is_empty());
    }

    #[test]
    fn test_kcore_single_node() {
        let engine = GraphEngine::new();
        engine.create_node("A", HashMap::new()).unwrap();

        let result = engine
            .kcore_decomposition(&KCoreConfig::new().undirected())
            .unwrap();
        assert_eq!(result.degeneracy, 0);
    }

    #[test]
    fn test_kcore_simple_path() {
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
            .kcore_decomposition(&KCoreConfig::new().undirected())
            .unwrap();

        // Path graph has degeneracy 1
        assert_eq!(result.degeneracy, 1);

        // All nodes are in 1-core
        assert_eq!(result.core_of(a), Some(1));
        assert_eq!(result.core_of(b), Some(1));
        assert_eq!(result.core_of(c), Some(1));
    }

    #[test]
    fn test_kcore_triangle() {
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
            .kcore_decomposition(&KCoreConfig::new().undirected())
            .unwrap();

        // Triangle (K3) has degeneracy 2
        assert_eq!(result.degeneracy, 2);

        // All nodes are in 2-core
        assert_eq!(result.core_of(a), Some(2));
        assert_eq!(result.core_of(b), Some(2));
        assert_eq!(result.core_of(c), Some(2));
    }

    #[test]
    fn test_kcore_with_pendant() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // Triangle a-b-c with pendant d attached to c
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

        let result = engine
            .kcore_decomposition(&KCoreConfig::new().undirected())
            .unwrap();

        // d is in 1-core (degree 1), triangle is in 2-core
        assert_eq!(result.core_of(d), Some(1));
        assert_eq!(result.core_of(a), Some(2));
        assert_eq!(result.core_of(b), Some(2));
        assert_eq!(result.core_of(c), Some(2));
    }

    #[test]
    fn test_kcore_subgraph() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

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

        let config = KCoreConfig::new().undirected();
        let k2_nodes = engine.kcore_subgraph(2, &config).unwrap();

        // Only triangle nodes are in 2-core
        assert_eq!(k2_nodes.len(), 3);
        assert!(k2_nodes.contains(&a));
        assert!(k2_nodes.contains(&b));
        assert!(k2_nodes.contains(&c));
        assert!(!k2_nodes.contains(&d));
    }

    #[test]
    fn test_kcore_shell() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

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

        let result = engine
            .kcore_decomposition(&KCoreConfig::new().undirected())
            .unwrap();

        // Shell(1) should only contain d
        let shell1 = result.shell(1);
        assert_eq!(shell1.len(), 1);
        assert!(shell1.contains(&d));

        // Shell(2) should contain a, b, c
        let shell2 = result.shell(2);
        assert_eq!(shell2.len(), 3);
    }

    #[test]
    fn test_degeneracy() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // K4 (complete graph on 4 vertices)
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, d, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, d, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(c, d, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = KCoreConfig::new().undirected();
        let deg = engine.degeneracy(&config).unwrap();

        // K4 has degeneracy 3
        assert_eq!(deg, 3);
    }

    #[test]
    fn test_kcore_disconnected() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Two isolated nodes + one isolated node
        engine
            .create_edge(a, b, "EDGE", HashMap::new(), false)
            .unwrap();

        let result = engine
            .kcore_decomposition(&KCoreConfig::new().undirected())
            .unwrap();

        // c is isolated (0-core), a and b are in 1-core
        assert_eq!(result.core_of(c), Some(0));
        assert_eq!(result.core_of(a), Some(1));
        assert_eq!(result.core_of(b), Some(1));
    }
}
