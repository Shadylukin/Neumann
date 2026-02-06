// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

//! Fuzz target for graph traversal and path algorithms.
//!
//! Tests traverse, find_path, shortest_path, neighbors operations
//! on graphs of various shapes.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use graph_engine::{Direction, GraphEngine, PropertyValue};
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
enum GraphShape {
    Chain { length: u8 },
    Star { spokes: u8 },
    Complete { size: u8 },
    Random { nodes: u8, edges: u8 },
}

#[derive(Arbitrary, Debug)]
enum PathOp {
    Traverse {
        start_idx: u8,
        direction: u8,
        max_depth: u8,
    },
    FindPath {
        from_idx: u8,
        to_idx: u8,
    },
    Neighbors {
        node_idx: u8,
        direction: u8,
        edge_type_filter: Option<u8>,
    },
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    shape: GraphShape,
    ops: Vec<PathOp>,
}

fn direction_from_u8(d: u8) -> Direction {
    match d % 3 {
        0 => Direction::Outgoing,
        1 => Direction::Incoming,
        _ => Direction::Both,
    }
}

fn build_graph(engine: &GraphEngine, shape: &GraphShape) -> Vec<u64> {
    let mut node_ids = Vec::new();

    match shape {
        GraphShape::Chain { length } => {
            let len = (*length as usize).clamp(2, 50);
            for i in 0..len {
                let id = engine
                    .create_node("Chain", HashMap::from([
                        ("idx".to_string(), PropertyValue::Int(i as i64)),
                    ]))
                    .unwrap();
                node_ids.push(id);
            }
            for i in 0..len - 1 {
                let _ = engine.create_edge(
                    node_ids[i],
                    node_ids[i + 1],
                    "NEXT",
                    HashMap::new(),
                    true,
                );
            }
        }
        GraphShape::Star { spokes } => {
            let spoke_count = (*spokes as usize).clamp(3, 30);
            let center = engine
                .create_node("Center", HashMap::new())
                .unwrap();
            node_ids.push(center);

            for i in 0..spoke_count {
                let spoke = engine
                    .create_node("Spoke", HashMap::from([
                        ("idx".to_string(), PropertyValue::Int(i as i64)),
                    ]))
                    .unwrap();
                node_ids.push(spoke);
                let _ = engine.create_edge(center, spoke, "CONNECTS", HashMap::new(), true);
            }
        }
        GraphShape::Complete { size } => {
            let n = (*size as usize).clamp(3, 15);
            for i in 0..n {
                let id = engine
                    .create_node("Complete", HashMap::from([
                        ("idx".to_string(), PropertyValue::Int(i as i64)),
                    ]))
                    .unwrap();
                node_ids.push(id);
            }
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let _ = engine.create_edge(
                            node_ids[i],
                            node_ids[j],
                            "LINK",
                            HashMap::new(),
                            true,
                        );
                    }
                }
            }
        }
        GraphShape::Random { nodes, edges } => {
            let node_count = (*nodes as usize).clamp(5, 50);
            let edge_count = (*edges as usize).clamp(5, 100);

            for i in 0..node_count {
                let id = engine
                    .create_node("Random", HashMap::from([
                        ("idx".to_string(), PropertyValue::Int(i as i64)),
                    ]))
                    .unwrap();
                node_ids.push(id);
            }

            let edge_types = ["A", "B", "C"];
            for i in 0..edge_count {
                let from = node_ids[i % node_count];
                let to = node_ids[(i * 7 + 3) % node_count];
                if from != to {
                    let _ = engine.create_edge(
                        from,
                        to,
                        edge_types[i % 3],
                        HashMap::new(),
                        true,
                    );
                }
            }
        }
    }

    node_ids
}

fuzz_target!(|input: FuzzInput| {
    let engine = GraphEngine::new();
    let node_ids = build_graph(&engine, &input.shape);

    if node_ids.is_empty() {
        return;
    }

    for op in input.ops.into_iter().take(50) {
        match op {
            PathOp::Traverse {
                start_idx,
                direction,
                max_depth,
            } => {
                let start = node_ids[start_idx as usize % node_ids.len()];
                let dir = direction_from_u8(direction);
                let depth = (max_depth as usize).clamp(1, 20);
                let _ = engine.traverse(start, dir, depth, None, None);
            }
            PathOp::FindPath { from_idx, to_idx } => {
                let from = node_ids[from_idx as usize % node_ids.len()];
                let to = node_ids[to_idx as usize % node_ids.len()];
                let _ = engine.find_path(from, to, None);
            }
            PathOp::Neighbors {
                node_idx,
                direction,
                edge_type_filter,
            } => {
                let node = node_ids[node_idx as usize % node_ids.len()];
                let dir = direction_from_u8(direction);
                let edge_type: Option<&str> = match edge_type_filter {
                    Some(t) => match t % 4 {
                        0 => Some("NEXT"),
                        1 => Some("CONNECTS"),
                        2 => Some("LINK"),
                        _ => Some("A"),
                    },
                    None => None,
                };
                let _ = engine.neighbors(node, edge_type, dir, None);
            }
        }
    }

    // Verify graph is still consistent
    for &node_id in &node_ids {
        let node_result = engine.get_node(node_id);
        assert!(
            node_result.is_ok(),
            "Node {} should still exist after path operations",
            node_id
        );
    }
});
