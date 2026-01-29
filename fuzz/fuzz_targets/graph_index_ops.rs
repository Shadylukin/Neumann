// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target for graph index operations.
//!
//! Tests create_node_property_index, create_edge_property_index,
//! drop_node_index, drop_edge_index, has_node_index, has_edge_index,
//! and verifies index consistency.

use arbitrary::Arbitrary;
use graph_engine::{GraphEngine, PropertyValue};
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzPropertyValue {
    Null,
    Bool(bool),
    Int(i64),
    String(String),
}

impl FuzzPropertyValue {
    fn to_property_value(&self) -> PropertyValue {
        match self {
            Self::Null => PropertyValue::Null,
            Self::Bool(b) => PropertyValue::Bool(*b),
            Self::Int(i) => PropertyValue::Int(*i),
            Self::String(s) => PropertyValue::String(s.chars().take(50).collect()),
        }
    }
}

#[derive(Arbitrary, Debug)]
enum IndexOp {
    CreateNodeIndex { property: String },
    CreateEdgeIndex { property: String },
    DropNodeIndex { property: String },
    DropEdgeIndex { property: String },
    CheckNodeIndex { property: String },
    CheckEdgeIndex { property: String },
    CreateNodeWithProp { property: String, value: FuzzPropertyValue },
    CreateEdgeWithProp { property: String, value: FuzzPropertyValue },
    QueryByProperty { property: String, value: FuzzPropertyValue },
    GetIndexedNodeProperties,
    GetIndexedEdgeProperties,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<IndexOp>,
}

fn sanitize_property(s: &str) -> String {
    let sanitized: String = s
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(32)
        .collect();
    if sanitized.is_empty() {
        "prop".to_string()
    } else {
        sanitized
    }
}

fuzz_target!(|input: FuzzInput| {
    let engine = GraphEngine::new();
    let mut created_nodes: Vec<u64> = Vec::new();
    let mut created_edges: Vec<u64> = Vec::new();

    // Create some initial nodes for edges
    for i in 0..5 {
        let id = engine
            .create_node(
                "Initial",
                HashMap::from([("idx".to_string(), PropertyValue::Int(i))]),
            )
            .unwrap();
        created_nodes.push(id);
    }

    for op in input.ops.into_iter().take(100) {
        match op {
            IndexOp::CreateNodeIndex { property } => {
                let prop = sanitize_property(&property);
                let _ = engine.create_node_property_index(&prop);
            }
            IndexOp::CreateEdgeIndex { property } => {
                let prop = sanitize_property(&property);
                let _ = engine.create_edge_property_index(&prop);
            }
            IndexOp::DropNodeIndex { property } => {
                let prop = sanitize_property(&property);
                let _ = engine.drop_node_index(&prop);
            }
            IndexOp::DropEdgeIndex { property } => {
                let prop = sanitize_property(&property);
                let _ = engine.drop_edge_index(&prop);
            }
            IndexOp::CheckNodeIndex { property } => {
                let prop = sanitize_property(&property);
                let _exists = engine.has_node_index(&prop);
            }
            IndexOp::CheckEdgeIndex { property } => {
                let prop = sanitize_property(&property);
                let _exists = engine.has_edge_index(&prop);
            }
            IndexOp::CreateNodeWithProp { property, value } => {
                let prop = sanitize_property(&property);
                let mut props = HashMap::new();
                props.insert(prop, value.to_property_value());
                if let Ok(id) = engine.create_node("Indexed", props) {
                    created_nodes.push(id);
                }
            }
            IndexOp::CreateEdgeWithProp { property, value } => {
                if created_nodes.len() >= 2 {
                    let prop = sanitize_property(&property);
                    let mut props = HashMap::new();
                    props.insert(prop, value.to_property_value());

                    let from = created_nodes[0];
                    let to = created_nodes[created_nodes.len() - 1];
                    if let Ok(id) = engine.create_edge(from, to, "INDEXED", props, true) {
                        created_edges.push(id);
                    }
                }
            }
            IndexOp::QueryByProperty { property, value } => {
                let prop = sanitize_property(&property);
                let pv = value.to_property_value();
                // Query should work regardless of index existence
                let _ = engine.find_nodes_by_property(&prop, &pv);
                let _ = engine.find_edges_by_property(&prop, &pv);
            }
            IndexOp::GetIndexedNodeProperties => {
                let indexed = engine.get_indexed_node_properties();
                // All returned properties should actually have indexes
                for prop in &indexed {
                    assert!(
                        engine.has_node_index(prop),
                        "get_indexed_node_properties returned non-indexed property"
                    );
                }
            }
            IndexOp::GetIndexedEdgeProperties => {
                let indexed = engine.get_indexed_edge_properties();
                // All returned properties should actually have indexes
                for prop in &indexed {
                    assert!(
                        engine.has_edge_index(prop),
                        "get_indexed_edge_properties returned non-indexed property"
                    );
                }
            }
        }
    }

    // Verify index consistency: has_*_index should match get_indexed_*_properties
    let node_indexed = engine.get_indexed_node_properties();
    let edge_indexed = engine.get_indexed_edge_properties();

    for prop in &node_indexed {
        assert!(
            engine.has_node_index(prop),
            "Index consistency: has_node_index should return true for indexed property"
        );
    }

    for prop in &edge_indexed {
        assert!(
            engine.has_edge_index(prop),
            "Index consistency: has_edge_index should return true for indexed property"
        );
    }
});
