// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target for GraphEngine CRUD operations.
//!
//! Tests create_node, create_edge, update_node, delete_node, delete_edge,
//! get_node, get_edge operations with arbitrary inputs.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use graph_engine::{GraphEngine, PropertyValue};
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzPropertyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl FuzzPropertyValue {
    fn to_property_value(&self) -> PropertyValue {
        match self {
            Self::Null => PropertyValue::Null,
            Self::Bool(b) => PropertyValue::Bool(*b),
            Self::Int(i) => PropertyValue::Int(*i),
            Self::Float(f) => {
                // Sanitize floats to avoid NaN/Inf issues
                let f = if f.is_nan() || f.is_infinite() { 0.0 } else { *f };
                PropertyValue::Float(f)
            }
            Self::String(s) => PropertyValue::String(s.chars().take(100).collect()),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzProperty {
    key: String,
    value: FuzzPropertyValue,
}

#[derive(Arbitrary, Debug)]
enum GraphOp {
    CreateNode {
        label: String,
        properties: Vec<FuzzProperty>,
    },
    CreateEdge {
        from_idx: u8,
        to_idx: u8,
        edge_type: String,
        properties: Vec<FuzzProperty>,
        directed: bool,
    },
    GetNode {
        node_idx: u8,
    },
    GetEdge {
        edge_idx: u8,
    },
    UpdateNode {
        node_idx: u8,
        new_labels: Option<Vec<String>>,
        properties: Vec<FuzzProperty>,
    },
    DeleteNode {
        node_idx: u8,
    },
    DeleteEdge {
        edge_idx: u8,
    },
    GetNeighbors {
        node_idx: u8,
        direction: u8,
    },
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<GraphOp>,
}

fn sanitize_label(s: &str) -> String {
    let sanitized: String = s
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(32)
        .collect();
    if sanitized.is_empty() {
        "Label".to_string()
    } else {
        sanitized
    }
}

fn sanitize_key(s: &str) -> String {
    let sanitized: String = s
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(32)
        .collect();
    if sanitized.is_empty() {
        "key".to_string()
    } else {
        sanitized
    }
}

fn build_properties(props: Vec<FuzzProperty>) -> HashMap<String, PropertyValue> {
    props
        .into_iter()
        .take(10)
        .map(|p| (sanitize_key(&p.key), p.value.to_property_value()))
        .collect()
}

fuzz_target!(|input: FuzzInput| {
    let engine = GraphEngine::new();
    let mut created_nodes: Vec<u64> = Vec::new();
    let mut created_edges: Vec<u64> = Vec::new();

    for op in input.ops.into_iter().take(200) {
        match op {
            GraphOp::CreateNode { label, properties } => {
                let label = sanitize_label(&label);
                let props = build_properties(properties);

                if let Ok(id) = engine.create_node(&label, props) {
                    created_nodes.push(id);
                }
            }
            GraphOp::CreateEdge {
                from_idx,
                to_idx,
                edge_type,
                properties,
                directed,
            } => {
                if created_nodes.len() < 2 {
                    continue;
                }
                let from = created_nodes[from_idx as usize % created_nodes.len()];
                let to = created_nodes[to_idx as usize % created_nodes.len()];
                let edge_type = sanitize_label(&edge_type);
                let props = build_properties(properties);

                if let Ok(id) = engine.create_edge(from, to, &edge_type, props, directed) {
                    created_edges.push(id);
                }
            }
            GraphOp::GetNode { node_idx } => {
                if created_nodes.is_empty() {
                    continue;
                }
                let id = created_nodes[node_idx as usize % created_nodes.len()];
                let _ = engine.get_node(id);
            }
            GraphOp::GetEdge { edge_idx } => {
                if created_edges.is_empty() {
                    continue;
                }
                let id = created_edges[edge_idx as usize % created_edges.len()];
                let _ = engine.get_edge(id);
            }
            GraphOp::UpdateNode {
                node_idx,
                new_labels,
                properties,
            } => {
                if created_nodes.is_empty() {
                    continue;
                }
                let id = created_nodes[node_idx as usize % created_nodes.len()];
                let labels = new_labels.map(|v| {
                    v.into_iter()
                        .take(5)
                        .map(|l| sanitize_label(&l))
                        .collect()
                });
                let props = build_properties(properties);
                let _ = engine.update_node(id, labels, props);
            }
            GraphOp::DeleteNode { node_idx } => {
                if created_nodes.is_empty() {
                    continue;
                }
                let idx = node_idx as usize % created_nodes.len();
                let id = created_nodes[idx];
                if engine.delete_node(id).is_ok() {
                    created_nodes.remove(idx);
                }
            }
            GraphOp::DeleteEdge { edge_idx } => {
                if created_edges.is_empty() {
                    continue;
                }
                let idx = edge_idx as usize % created_edges.len();
                let id = created_edges[idx];
                if engine.delete_edge(id).is_ok() {
                    created_edges.remove(idx);
                }
            }
            GraphOp::GetNeighbors { node_idx, direction } => {
                if created_nodes.is_empty() {
                    continue;
                }
                let id = created_nodes[node_idx as usize % created_nodes.len()];
                let dir = match direction % 3 {
                    0 => graph_engine::Direction::Outgoing,
                    1 => graph_engine::Direction::Incoming,
                    _ => graph_engine::Direction::Both,
                };
                let _ = engine.neighbors(id, None, dir, None);
            }
        }
    }

    // Verify invariants: all remaining nodes should be retrievable
    for &node_id in &created_nodes {
        assert!(
            engine.get_node(node_id).is_ok(),
            "Node {} should exist",
            node_id
        );
    }

    // Verify invariants: all remaining edges should be retrievable
    for &edge_id in &created_edges {
        assert!(
            engine.get_edge(edge_id).is_ok(),
            "Edge {} should exist",
            edge_id
        );
    }
});
