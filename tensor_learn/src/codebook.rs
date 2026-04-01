// SPDX-License-Identifier: MIT OR Apache-2.0
//! Codebook backed by graph and vector engines.
//!
//! Each entry is stored as both a graph node (for hierarchy) and a vector
//! embedding in a Poincare-metric collection (for nearest-neighbor search).

use std::collections::HashMap;

use graph_engine::{Direction, GraphEngine, PropertyValue};
use vector_engine::{DistanceMetric, SearchResult, VectorCollectionConfig, VectorEngine};

use crate::error::LearnError;
use crate::poincare::PoincarePoint;
use crate::viz::{VizData, VizEdge, VizMetadata, VizNode};

/// Collection name used for Poincare embeddings.
const COLLECTION_NAME: &str = "codebook";

/// Public view of a codebook entry.
pub struct EntryInfo<'a> {
    /// User-provided key.
    pub key: &'a str,
    /// Display label.
    pub label: &'a str,
    /// Hierarchy level (0 = root).
    pub level: u32,
}

/// A codebook entry mapping a string key to a graph node ID.
#[derive(Debug, Clone)]
struct Entry {
    /// User-provided key.
    key: String,
    /// Graph node ID.
    node_id: u64,
    /// Label for display.
    label: String,
    /// Hierarchy level.
    level: u32,
}

/// Codebook: a set of entries with graph hierarchy and Poincare embeddings.
pub struct Codebook {
    /// Graph engine for hierarchy relationships.
    graph: GraphEngine,
    /// Vector engine for Poincare nearest-neighbor search.
    vector: VectorEngine,
    /// Dimension of the embedding space.
    dimension: usize,
    /// Curvature parameter.
    curvature: f64,
    /// Entries in insertion order.
    entries: Vec<Entry>,
}

impl Codebook {
    /// Create a new codebook.
    ///
    /// Configures a `"codebook"` collection with Poincare metric.
    ///
    /// # Errors
    ///
    /// Returns an error if collection creation fails.
    #[allow(clippy::cast_possible_truncation)]
    pub fn new(dimension: usize, curvature: f64) -> Result<Self, LearnError> {
        let graph = GraphEngine::new();
        let vector = VectorEngine::new();

        let config = VectorCollectionConfig::default()
            .with_dimension(dimension)
            .with_metric(DistanceMetric::Poincare)
            .with_poincare_curvature(curvature as f32);

        vector.create_collection(COLLECTION_NAME, config)?;

        Ok(Self {
            graph,
            vector,
            dimension,
            curvature,
            entries: Vec::new(),
        })
    }

    /// Add an entry to the codebook.
    ///
    /// Creates a graph node with the given label and stores the Poincare
    /// embedding in the codebook collection.
    ///
    /// # Errors
    ///
    /// Returns an error if the point is invalid or storage fails.
    pub fn add_entry(
        &mut self,
        key: &str,
        label: &str,
        point: &PoincarePoint,
        level: u32,
    ) -> Result<(), LearnError> {
        if point.dim() != self.dimension {
            return Err(LearnError::InvalidPoint(format!(
                "expected dimension {}, got {}",
                self.dimension,
                point.dim()
            )));
        }
        if !point.is_valid() {
            return Err(LearnError::InvalidPoint(
                "point lies outside the unit disk".to_string(),
            ));
        }

        // Create graph node with level property
        let mut props = HashMap::new();
        props.insert("level".to_string(), PropertyValue::Int(i64::from(level)));
        props.insert("key".to_string(), PropertyValue::String(key.to_string()));
        let node_id = self.graph.create_node(label, props)?;

        // Store Poincare embedding
        self.vector
            .store_in_collection(COLLECTION_NAME, key, point.to_f32_vec())?;

        self.entries.push(Entry {
            key: key.to_string(),
            node_id,
            label: label.to_string(),
            level,
        });
        Ok(())
    }

    /// Add a directed edge between two entries by key.
    ///
    /// # Errors
    ///
    /// Returns an error if either entry does not exist.
    pub fn add_edge(
        &mut self,
        from_key: &str,
        to_key: &str,
        label: &str,
    ) -> Result<(), LearnError> {
        let from_id = self
            .find_node_id(from_key)
            .ok_or_else(|| LearnError::Config(format!("entry not found: {from_key}")))?;
        let to_id = self
            .find_node_id(to_key)
            .ok_or_else(|| LearnError::Config(format!("entry not found: {to_key}")))?;

        self.graph
            .create_edge(from_id, to_id, label, HashMap::new(), true)?;
        Ok(())
    }

    /// Find the k nearest entries to a point.
    ///
    /// # Errors
    ///
    /// Returns an error if the search fails.
    pub fn nearest(
        &self,
        point: &PoincarePoint,
        k: usize,
    ) -> Result<Vec<SearchResult>, LearnError> {
        let results = self
            .vector
            .search_in_collection(COLLECTION_NAME, &point.to_f32_vec(), k)?;
        Ok(results)
    }

    /// Get the children of a node (outgoing edges).
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not exist.
    pub fn children(&self, key: &str) -> Result<Vec<String>, LearnError> {
        let node_id = self
            .find_node_id(key)
            .ok_or_else(|| LearnError::Config(format!("entry not found: {key}")))?;

        let edges = self.graph.edges_of(node_id, Direction::Outgoing)?;
        let child_keys: Vec<String> = edges
            .iter()
            .filter_map(|e| self.find_key_by_node_id(e.to))
            .collect();
        Ok(child_keys)
    }

    /// Get the parent of a node (incoming edges).
    ///
    /// # Errors
    ///
    /// Returns an error if the node does not exist.
    pub fn parent(&self, key: &str) -> Result<Option<String>, LearnError> {
        let node_id = self
            .find_node_id(key)
            .ok_or_else(|| LearnError::Config(format!("entry not found: {key}")))?;

        let edges = self.graph.edges_of(node_id, Direction::Incoming)?;
        let parent = edges.first().and_then(|e| self.find_key_by_node_id(e.from));
        Ok(parent)
    }

    /// Iterate all entries in insertion order.
    #[must_use]
    pub fn entries(&self) -> Vec<EntryInfo<'_>> {
        self.entries
            .iter()
            .map(|e| EntryInfo {
                key: &e.key,
                label: &e.label,
                level: e.level,
            })
            .collect()
    }

    /// Directed edge pairs `(from_key, to_key)` for all outgoing edges.
    ///
    /// # Errors
    ///
    /// Returns an error if reading graph state fails.
    pub fn edge_pairs(&self) -> Result<Vec<(String, String)>, LearnError> {
        let mut pairs = Vec::new();
        for entry in &self.entries {
            let edges = self.graph.edges_of(entry.node_id, Direction::Outgoing)?;
            for edge in edges {
                if let Some(to_key) = self.find_key_by_node_id(edge.to) {
                    pairs.push((entry.key.clone(), to_key));
                }
            }
        }
        Ok(pairs)
    }

    /// Retrieve the stored embedding vector for an entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the key does not exist.
    pub fn get_embedding(&self, key: &str) -> Result<Vec<f32>, LearnError> {
        Ok(self.vector.get_from_collection(COLLECTION_NAME, key)?)
    }

    /// Overwrite the embedding for an existing entry.
    ///
    /// # Errors
    ///
    /// Returns an error if storage fails.
    pub fn update_embedding(&self, key: &str, point: &PoincarePoint) -> Result<(), LearnError> {
        self.vector
            .store_in_collection(COLLECTION_NAME, key, point.to_f32_vec())?;
        Ok(())
    }

    /// Curvature of the Poincare disk.
    #[must_use]
    pub const fn curvature(&self) -> f64 {
        self.curvature
    }

    /// Export codebook state as visualization data.
    ///
    /// # Errors
    ///
    /// Returns an error if reading vector state fails.
    pub fn to_viz_data(&self) -> Result<VizData, LearnError> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        for entry in &self.entries {
            // Read full coordinate vector and compute true hyperbolic distance
            let (x, y, distance_from_origin) = self
                .vector
                .get_from_collection(COLLECTION_NAME, &entry.key)
                .map_or((0.0, 0.0, 0.0), |vec| {
                    let px = f64::from(*vec.first().unwrap_or(&0.0));
                    let py = f64::from(*vec.get(1).unwrap_or(&0.0));
                    let f64_coords: Vec<f64> = vec.iter().map(|&v| f64::from(v)).collect();
                    let point = PoincarePoint::new(f64_coords);
                    let origin = PoincarePoint::origin(self.dimension);
                    let dist = point.distance(&origin, self.curvature);
                    (px, py, dist)
                });

            nodes.push(VizNode {
                id: entry.key.clone(),
                label: entry.label.clone(),
                x,
                y,
                level: entry.level,
                distance_from_origin,
            });

            // Collect outgoing edges
            if let Ok(node_edges) = self.graph.edges_of(entry.node_id, Direction::Outgoing) {
                for edge in node_edges {
                    if let Some(to_key) = self.find_key_by_node_id(edge.to) {
                        edges.push(VizEdge {
                            from: entry.key.clone(),
                            to: to_key,
                            label: edge.edge_type.clone(),
                        });
                    }
                }
            }
        }

        Ok(VizData {
            metadata: VizMetadata {
                entry_count: nodes.len(),
                edge_count: edges.len(),
                curvature: self.curvature,
                dimension: self.dimension,
            },
            nodes,
            edges,
        })
    }

    /// Number of entries in the codebook.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the codebook is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Access the graph engine.
    #[must_use]
    pub const fn graph(&self) -> &GraphEngine {
        &self.graph
    }

    /// Access the vector engine.
    #[must_use]
    pub const fn vector(&self) -> &VectorEngine {
        &self.vector
    }

    /// Look up a graph node ID by entry key.
    fn find_node_id(&self, key: &str) -> Option<u64> {
        self.entries
            .iter()
            .find(|e| e.key == key)
            .map(|e| e.node_id)
    }

    /// Look up an entry key by graph node ID.
    fn find_key_by_node_id(&self, node_id: u64) -> Option<String> {
        self.entries
            .iter()
            .find(|e| e.node_id == node_id)
            .map(|e| e.key.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_codebook() {
        let cb = Codebook::new(2, 1.0).unwrap();
        assert_eq!(cb.len(), 0);
        assert!(cb.is_empty());
    }

    #[test]
    fn test_add_entry() {
        let mut cb = Codebook::new(2, 1.0).unwrap();
        let point = PoincarePoint::new(vec![0.0, 0.0]);
        cb.add_entry("root", "Root", &point, 0).unwrap();
        assert_eq!(cb.len(), 1);
        assert!(!cb.is_empty());
    }

    #[test]
    fn test_add_entry_wrong_dimension() {
        let mut cb = Codebook::new(2, 1.0).unwrap();
        let point = PoincarePoint::new(vec![0.0, 0.0, 0.0]);
        let result = cb.add_entry("bad", "Bad", &point, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_edge_and_children() {
        let mut cb = Codebook::new(2, 1.0).unwrap();
        let root = PoincarePoint::origin(2);
        let child = PoincarePoint::new(vec![0.3, 0.0]);

        cb.add_entry("root", "Root", &root, 0).unwrap();
        cb.add_entry("child1", "Child", &child, 1).unwrap();
        cb.add_edge("root", "child1", "contains").unwrap();

        let children = cb.children("root").unwrap();
        assert_eq!(children, vec!["child1"]);

        let parent = cb.parent("child1").unwrap();
        assert_eq!(parent, Some("root".to_string()));
    }

    #[test]
    fn test_nearest() {
        let mut cb = Codebook::new(2, 1.0).unwrap();

        cb.add_entry("a", "A", &PoincarePoint::new(vec![0.0, 0.0]), 0)
            .unwrap();
        cb.add_entry("b", "B", &PoincarePoint::new(vec![0.1, 0.0]), 1)
            .unwrap();
        cb.add_entry("c", "C", &PoincarePoint::new(vec![0.5, 0.5]), 1)
            .unwrap();

        let query = PoincarePoint::origin(2);
        let results = cb.nearest(&query, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "a");
    }

    #[test]
    fn test_to_viz_data() {
        let mut cb = Codebook::new(2, 1.0).unwrap();

        cb.add_entry("root", "Root", &PoincarePoint::origin(2), 0)
            .unwrap();
        cb.add_entry("c1", "Child1", &PoincarePoint::new(vec![0.3, 0.0]), 1)
            .unwrap();
        cb.add_edge("root", "c1", "contains").unwrap();

        let viz = cb.to_viz_data().unwrap();
        assert_eq!(viz.nodes.len(), 2);
        assert_eq!(viz.edges.len(), 1);
        assert_eq!(viz.metadata.entry_count, 2);
        assert_eq!(viz.metadata.curvature, 1.0);
    }

    #[test]
    fn test_no_parent_for_root() {
        let mut cb = Codebook::new(2, 1.0).unwrap();
        cb.add_entry("root", "Root", &PoincarePoint::origin(2), 0)
            .unwrap();
        let parent = cb.parent("root").unwrap();
        assert!(parent.is_none());
    }

    #[test]
    fn test_accessors() {
        let cb = Codebook::new(2, 1.0).unwrap();
        let _ = cb.graph();
        let _ = cb.vector();
    }

    #[test]
    fn test_add_edge_missing_node() {
        let mut cb = Codebook::new(2, 1.0).unwrap();
        cb.add_entry("a", "A", &PoincarePoint::origin(2), 0)
            .unwrap();
        let result = cb.add_edge("a", "missing", "link");
        assert!(result.is_err());
    }

    #[test]
    fn test_children_missing_node() {
        let cb = Codebook::new(2, 1.0).unwrap();
        let result = cb.children("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_parent_missing_node() {
        let cb = Codebook::new(2, 1.0).unwrap();
        let result = cb.parent("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_entries_returns_insertion_order() {
        let mut cb = Codebook::new(2, 1.0).unwrap();
        cb.add_entry("a", "Alpha", &PoincarePoint::origin(2), 0)
            .unwrap();
        cb.add_entry("b", "Beta", &PoincarePoint::new(vec![0.1, 0.0]), 1)
            .unwrap();
        let entries = cb.entries();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].key, "a");
        assert_eq!(entries[0].label, "Alpha");
        assert_eq!(entries[0].level, 0);
        assert_eq!(entries[1].key, "b");
    }

    #[test]
    fn test_edge_pairs() {
        let mut cb = Codebook::new(2, 1.0).unwrap();
        cb.add_entry("r", "Root", &PoincarePoint::origin(2), 0)
            .unwrap();
        cb.add_entry("c1", "C1", &PoincarePoint::new(vec![0.2, 0.0]), 1)
            .unwrap();
        cb.add_entry("c2", "C2", &PoincarePoint::new(vec![0.0, 0.2]), 1)
            .unwrap();
        cb.add_edge("r", "c1", "child").unwrap();
        cb.add_edge("r", "c2", "child").unwrap();
        let pairs = cb.edge_pairs().unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "r");
    }

    #[test]
    fn test_get_and_update_embedding() {
        let mut cb = Codebook::new(2, 1.0).unwrap();
        let original = PoincarePoint::new(vec![0.1, 0.2]);
        cb.add_entry("x", "X", &original, 0).unwrap();

        let stored = cb.get_embedding("x").unwrap();
        assert!((f64::from(stored[0]) - 0.1).abs() < 1e-5);

        let updated = PoincarePoint::new(vec![0.3, 0.4]);
        cb.update_embedding("x", &updated).unwrap();

        let after = cb.get_embedding("x").unwrap();
        assert!((f64::from(after[0]) - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_curvature_accessor() {
        let cb = Codebook::new(2, 2.0).unwrap();
        assert!((cb.curvature() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_to_viz_data_hyperbolic_distance() {
        let mut cb = Codebook::new(2, 1.0).unwrap();
        let point = PoincarePoint::new(vec![0.5, 0.0]);
        cb.add_entry("p", "P", &point, 1).unwrap();
        let viz = cb.to_viz_data().unwrap();
        // Hyperbolic distance should be greater than Euclidean radius
        let node = &viz.nodes[0];
        let euclidean_r = node.x.hypot(node.y);
        assert!(
            node.distance_from_origin > euclidean_r,
            "hyperbolic dist {} should exceed Euclidean radius {}",
            node.distance_from_origin,
            euclidean_r
        );
    }
}
