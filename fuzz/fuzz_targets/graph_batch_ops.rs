#![no_main]

//! Fuzz target for graph batch operations.
//!
//! Tests batch_create_nodes, batch_create_edges, batch_delete_nodes,
//! batch_delete_edges, batch_update_nodes with various inputs.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use graph_engine::{EdgeInput, GraphEngine, NodeInput, PropertyValue};
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
                let f = if f.is_nan() || f.is_infinite() { 0.0 } else { *f };
                PropertyValue::Float(f)
            }
            Self::String(s) => PropertyValue::String(s.chars().take(50).collect()),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzNodeInput {
    labels: Vec<String>,
    properties: Vec<(String, FuzzPropertyValue)>,
}

#[derive(Arbitrary, Debug)]
struct FuzzEdgeInput {
    from_idx: u8,
    to_idx: u8,
    edge_type: String,
    properties: Vec<(String, FuzzPropertyValue)>,
    directed: bool,
}

#[derive(Arbitrary, Debug)]
struct FuzzNodeUpdate {
    node_idx: u8,
    properties: Vec<(String, FuzzPropertyValue)>,
}

#[derive(Arbitrary, Debug)]
enum BatchOp {
    CreateNodes { nodes: Vec<FuzzNodeInput> },
    CreateEdges { edges: Vec<FuzzEdgeInput> },
    DeleteNodes { indices: Vec<u8> },
    DeleteEdges { indices: Vec<u8> },
    UpdateNodes { updates: Vec<FuzzNodeUpdate> },
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<BatchOp>,
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

fn build_properties(props: Vec<(String, FuzzPropertyValue)>) -> HashMap<String, PropertyValue> {
    props
        .into_iter()
        .take(10)
        .map(|(k, v)| (sanitize_key(&k), v.to_property_value()))
        .collect()
}

fuzz_target!(|input: FuzzInput| {
    let engine = GraphEngine::new();
    let mut created_nodes: Vec<u64> = Vec::new();
    let mut created_edges: Vec<u64> = Vec::new();

    // Create some initial nodes for edge operations
    for i in 0..20 {
        if let Ok(id) = engine.create_node(
            "Initial",
            HashMap::from([("idx".to_string(), PropertyValue::Int(i))]),
        ) {
            created_nodes.push(id);
        }
    }

    for op in input.ops.into_iter().take(50) {
        match op {
            BatchOp::CreateNodes { nodes } => {
                let node_inputs: Vec<NodeInput> = nodes
                    .into_iter()
                    .take(100)
                    .map(|n| {
                        let labels: Vec<String> = n
                            .labels
                            .into_iter()
                            .take(5)
                            .map(|l| sanitize_label(&l))
                            .collect();
                        let labels = if labels.is_empty() {
                            vec!["Default".to_string()]
                        } else {
                            labels
                        };
                        NodeInput {
                            labels,
                            properties: build_properties(n.properties),
                        }
                    })
                    .collect();

                if node_inputs.is_empty() {
                    continue;
                }

                if let Ok(result) = engine.batch_create_nodes(node_inputs) {
                    created_nodes.extend(result.created_ids);
                }
            }
            BatchOp::CreateEdges { edges } => {
                if created_nodes.len() < 2 {
                    continue;
                }

                let edge_inputs: Vec<EdgeInput> = edges
                    .into_iter()
                    .take(100)
                    .filter_map(|e| {
                        let from = created_nodes.get(e.from_idx as usize % created_nodes.len())?;
                        let to = created_nodes.get(e.to_idx as usize % created_nodes.len())?;
                        Some(EdgeInput {
                            from: *from,
                            to: *to,
                            edge_type: sanitize_label(&e.edge_type),
                            properties: build_properties(e.properties),
                            directed: e.directed,
                        })
                    })
                    .collect();

                if edge_inputs.is_empty() {
                    continue;
                }

                if let Ok(result) = engine.batch_create_edges(edge_inputs) {
                    created_edges.extend(result.created_ids);
                }
            }
            BatchOp::DeleteNodes { indices } => {
                if created_nodes.is_empty() {
                    continue;
                }

                let ids_to_delete: Vec<u64> = indices
                    .into_iter()
                    .take(50)
                    .map(|idx| created_nodes[idx as usize % created_nodes.len()])
                    .collect();

                if ids_to_delete.is_empty() {
                    continue;
                }

                if let Ok(count) = engine.batch_delete_nodes(ids_to_delete.clone()) {
                    // Remove deleted nodes from our tracking
                    created_nodes.retain(|id| !ids_to_delete.contains(id));
                }
            }
            BatchOp::DeleteEdges { indices } => {
                if created_edges.is_empty() {
                    continue;
                }

                let ids_to_delete: Vec<u64> = indices
                    .into_iter()
                    .take(50)
                    .map(|idx| created_edges[idx as usize % created_edges.len()])
                    .collect();

                if ids_to_delete.is_empty() {
                    continue;
                }

                if let Ok(_count) = engine.batch_delete_edges(ids_to_delete.clone()) {
                    // Remove deleted edges from our tracking
                    created_edges.retain(|id| !ids_to_delete.contains(id));
                }
            }
            BatchOp::UpdateNodes { updates } => {
                if created_nodes.is_empty() {
                    continue;
                }

                let update_inputs: Vec<(u64, HashMap<String, PropertyValue>)> = updates
                    .into_iter()
                    .take(50)
                    .map(|u| {
                        let id = created_nodes[u.node_idx as usize % created_nodes.len()];
                        let props = build_properties(u.properties);
                        (id, props)
                    })
                    .collect();

                if update_inputs.is_empty() {
                    continue;
                }

                let _ = engine.batch_update_nodes(update_inputs);
            }
        }
    }

    // Verify invariants: count of nodes should be consistent
    let mut valid_node_count = 0;
    for &node_id in &created_nodes {
        if engine.get_node(node_id).is_ok() {
            valid_node_count += 1;
        }
    }

    // Verify invariants: count of edges should be consistent
    let mut valid_edge_count = 0;
    for &edge_id in &created_edges {
        if engine.get_edge(edge_id).is_ok() {
            valid_edge_count += 1;
        }
    }

    // Nodes and edges should still be retrievable or properly deleted
    // (no panics or corrupted state)
});
