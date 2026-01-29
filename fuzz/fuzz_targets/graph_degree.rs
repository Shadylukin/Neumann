// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz target for graph degree calculations.
//!
//! Tests degree(), in_degree(), out_degree(), and their by_type variants.
//! Verifies that degree() == edges_of().len() for each node.

use arbitrary::Arbitrary;
use graph_engine::{Direction, GraphEngine, PropertyValue};
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
enum DegreeOp {
    CreateNode { label: String },
    CreateEdge { from_idx: u8, to_idx: u8, edge_type: String, directed: bool },
    DeleteEdge { edge_idx: u8 },
    CheckDegree { node_idx: u8 },
    CheckInDegree { node_idx: u8 },
    CheckOutDegree { node_idx: u8 },
    CheckDegreeByType { node_idx: u8, edge_type: String },
    CheckInDegreeByType { node_idx: u8, edge_type: String },
    CheckOutDegreeByType { node_idx: u8, edge_type: String },
    VerifyDegreeInvariant { node_idx: u8 },
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<DegreeOp>,
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

fuzz_target!(|input: FuzzInput| {
    let engine = GraphEngine::new();
    let mut node_ids: Vec<u64> = Vec::new();
    let mut edge_ids: Vec<u64> = Vec::new();

    // Create some initial nodes
    for i in 0..5 {
        let id = engine
            .create_node(
                "Initial",
                HashMap::from([("idx".to_string(), PropertyValue::Int(i))]),
            )
            .unwrap();
        node_ids.push(id);
    }

    for op in input.ops.into_iter().take(200) {
        match op {
            DegreeOp::CreateNode { label } => {
                let label = sanitize_label(&label);
                if let Ok(id) = engine.create_node(&label, HashMap::new()) {
                    node_ids.push(id);
                }
            }
            DegreeOp::CreateEdge {
                from_idx,
                to_idx,
                edge_type,
                directed,
            } => {
                if node_ids.len() >= 2 {
                    let from = node_ids[from_idx as usize % node_ids.len()];
                    let to = node_ids[to_idx as usize % node_ids.len()];
                    let edge_type = sanitize_label(&edge_type);
                    if let Ok(id) =
                        engine.create_edge(from, to, &edge_type, HashMap::new(), directed)
                    {
                        edge_ids.push(id);
                    }
                }
            }
            DegreeOp::DeleteEdge { edge_idx } => {
                if !edge_ids.is_empty() {
                    let idx = edge_idx as usize % edge_ids.len();
                    let edge_id = edge_ids[idx];
                    if engine.delete_edge(edge_id).is_ok() {
                        edge_ids.remove(idx);
                    }
                }
            }
            DegreeOp::CheckDegree { node_idx } => {
                if !node_ids.is_empty() {
                    let node_id = node_ids[node_idx as usize % node_ids.len()];
                    if let Ok(degree) = engine.degree(node_id) {
                        // Degree should be non-negative (enforced by type)
                        // and should equal in_degree + out_degree for directed graphs
                        // (but may be different for undirected edges counted once)
                        let _ = degree;
                    }
                }
            }
            DegreeOp::CheckInDegree { node_idx } => {
                if !node_ids.is_empty() {
                    let node_id = node_ids[node_idx as usize % node_ids.len()];
                    let _ = engine.in_degree(node_id);
                }
            }
            DegreeOp::CheckOutDegree { node_idx } => {
                if !node_ids.is_empty() {
                    let node_id = node_ids[node_idx as usize % node_ids.len()];
                    let _ = engine.out_degree(node_id);
                }
            }
            DegreeOp::CheckDegreeByType { node_idx, edge_type } => {
                if !node_ids.is_empty() {
                    let node_id = node_ids[node_idx as usize % node_ids.len()];
                    let edge_type = sanitize_label(&edge_type);
                    let _ = engine.degree_by_type(node_id, &edge_type);
                }
            }
            DegreeOp::CheckInDegreeByType { node_idx, edge_type } => {
                if !node_ids.is_empty() {
                    let node_id = node_ids[node_idx as usize % node_ids.len()];
                    let edge_type = sanitize_label(&edge_type);
                    let _ = engine.in_degree_by_type(node_id, &edge_type);
                }
            }
            DegreeOp::CheckOutDegreeByType { node_idx, edge_type } => {
                if !node_ids.is_empty() {
                    let node_id = node_ids[node_idx as usize % node_ids.len()];
                    let edge_type = sanitize_label(&edge_type);
                    let _ = engine.out_degree_by_type(node_id, &edge_type);
                }
            }
            DegreeOp::VerifyDegreeInvariant { node_idx } => {
                if !node_ids.is_empty() {
                    let node_id = node_ids[node_idx as usize % node_ids.len()];

                    // Verify degree equals edges_of(Both).len()
                    if let (Ok(degree), Ok(edges)) =
                        (engine.degree(node_id), engine.edges_of(node_id, Direction::Both))
                    {
                        assert_eq!(
                            degree,
                            edges.len(),
                            "degree() should equal edges_of(Both).len()"
                        );
                    }

                    // Verify in_degree equals edges_of(Incoming).len()
                    if let (Ok(in_deg), Ok(edges)) = (
                        engine.in_degree(node_id),
                        engine.edges_of(node_id, Direction::Incoming),
                    ) {
                        assert_eq!(
                            in_deg,
                            edges.len(),
                            "in_degree() should equal edges_of(Incoming).len()"
                        );
                    }

                    // Verify out_degree equals edges_of(Outgoing).len()
                    if let (Ok(out_deg), Ok(edges)) = (
                        engine.out_degree(node_id),
                        engine.edges_of(node_id, Direction::Outgoing),
                    ) {
                        assert_eq!(
                            out_deg,
                            edges.len(),
                            "out_degree() should equal edges_of(Outgoing).len()"
                        );
                    }
                }
            }
        }
    }

    // Final verification: check all nodes have consistent degree
    for &node_id in &node_ids {
        if let (Ok(degree), Ok(edges)) =
            (engine.degree(node_id), engine.edges_of(node_id, Direction::Both))
        {
            assert_eq!(
                degree,
                edges.len(),
                "Final check: degree() should equal edges_of(Both).len() for node {}",
                node_id
            );
        }
    }
});
