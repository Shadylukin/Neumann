// SPDX-License-Identifier: MIT OR Apache-2.0
//! Visualization data structures for the Poincare disk dashboard.

use serde::{Deserialize, Serialize};

/// Visualization data for the Poincare disk renderer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizData {
    /// Nodes to render on the Poincare disk.
    pub nodes: Vec<VizNode>,
    /// Edges between nodes (geodesic arcs).
    pub edges: Vec<VizEdge>,
    /// Dashboard metadata.
    pub metadata: VizMetadata,
}

/// A node in the visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizNode {
    /// Unique identifier.
    pub id: String,
    /// Display label.
    pub label: String,
    /// Coordinates in the Poincare disk (2D projection).
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Hierarchy level (0 = root).
    pub level: u32,
    /// Hyperbolic distance from origin.
    pub distance_from_origin: f64,
}

/// An edge in the visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizEdge {
    /// Source node ID.
    pub from: String,
    /// Target node ID.
    pub to: String,
    /// Edge label.
    pub label: String,
}

/// Metadata about the visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizMetadata {
    /// Total number of entries in the codebook.
    pub entry_count: usize,
    /// Total number of edges.
    pub edge_count: usize,
    /// Curvature parameter used.
    pub curvature: f64,
    /// Dimension of the embedding space.
    pub dimension: usize,
}

impl VizData {
    /// Create empty visualization data.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: VizMetadata {
                entry_count: 0,
                edge_count: 0,
                curvature: 1.0,
                dimension: 2,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let viz = VizData::empty();
        assert!(viz.nodes.is_empty());
        assert!(viz.edges.is_empty());
        assert_eq!(viz.metadata.entry_count, 0);
    }

    #[test]
    fn test_serde_roundtrip() {
        let viz = VizData {
            nodes: vec![VizNode {
                id: "n1".to_string(),
                label: "root".to_string(),
                x: 0.0,
                y: 0.0,
                level: 0,
                distance_from_origin: 0.0,
            }],
            edges: vec![VizEdge {
                from: "n1".to_string(),
                to: "n2".to_string(),
                label: "child".to_string(),
            }],
            metadata: VizMetadata {
                entry_count: 1,
                edge_count: 1,
                curvature: 1.0,
                dimension: 2,
            },
        };
        let json = serde_json::to_string(&viz).unwrap();
        let recovered: VizData = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.nodes.len(), 1);
        assert_eq!(recovered.edges.len(), 1);
    }

    #[test]
    fn test_viz_node_fields() {
        let node = VizNode {
            id: "test".to_string(),
            label: "Test".to_string(),
            x: 0.5,
            y: -0.3,
            level: 2,
            distance_from_origin: 0.583,
        };
        assert_eq!(node.id, "test");
        assert_eq!(node.level, 2);
    }
}
