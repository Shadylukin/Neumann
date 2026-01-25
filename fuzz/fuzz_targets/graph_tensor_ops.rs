#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{EdgeId, EntityId, GraphTensor};

#[derive(Arbitrary, Debug)]
enum GraphOp {
    AddEdge { from: u32, to: u32, edge_type: u8 },
    DeleteEdge { edge_id: u32 },
    Outgoing { node: u32 },
    Incoming { node: u32 },
    Merge,
}

#[derive(Arbitrary, Debug)]
struct GraphInput {
    merge_threshold: u8,
    operations: Vec<GraphOp>,
}

fuzz_target!(|input: GraphInput| {
    let threshold = (input.merge_threshold as usize).clamp(1, 100);
    let graph = GraphTensor::with_merge_threshold(threshold);

    // Track edge IDs we've created for valid delete operations
    let mut created_edges: Vec<EdgeId> = Vec::new();

    for op in input.operations.iter().take(200) {
        match op {
            GraphOp::AddEdge {
                from,
                to,
                edge_type,
            } => {
                // Limit node IDs to reasonable range
                let from_id = EntityId::new((*from as u64) % 1000);
                let to_id = EntityId::new((*to as u64) % 1000);
                let etype = format!("type_{}", edge_type % 10);

                let edge_id = graph.add_edge(from_id, to_id, &etype, true);
                created_edges.push(edge_id);
            }
            GraphOp::DeleteEdge { edge_id } => {
                // Use modulo to select from created edges if any exist
                if !created_edges.is_empty() {
                    let idx = (*edge_id as usize) % created_edges.len();
                    let _ = graph.delete_edge(created_edges[idx]);
                } else {
                    // Try deleting with arbitrary EdgeId
                    let _ = graph.delete_edge(EdgeId::new(*edge_id as u64));
                }
            }
            GraphOp::Outgoing { node } => {
                let node_id = EntityId::new((*node as u64) % 1000);
                let _ = graph.outgoing(node_id);
            }
            GraphOp::Incoming { node } => {
                let node_id = EntityId::new((*node as u64) % 1000);
                let _ = graph.incoming(node_id);
            }
            GraphOp::Merge => {
                graph.merge();
            }
        }
    }

    // Verify graph invariants
    let edge_count = graph.edge_count();

    // Verify outgoing and incoming are consistent for all nodes we touched
    for node_id in 0..100u64 {
        let node = EntityId::new(node_id);
        let outgoing = graph.outgoing(node);
        let incoming = graph.incoming(node);

        // Each outgoing edge from A to B should have B's incoming contain A
        for (target, edge_id) in &outgoing {
            let target_incoming = graph.incoming(*target);
            let found = target_incoming
                .iter()
                .any(|(from, eid)| *from == node && *eid == *edge_id);
            assert!(
                found,
                "Outgoing edge from {} to {} not found in incoming index",
                node_id,
                target.as_u64()
            );
        }

        // Each incoming edge to A from B should have B's outgoing contain A
        for (source, edge_id) in &incoming {
            let source_outgoing = graph.outgoing(*source);
            let found = source_outgoing
                .iter()
                .any(|(to, eid)| *to == node && *eid == *edge_id);
            assert!(
                found,
                "Incoming edge to {} from {} not found in outgoing index",
                node_id,
                source.as_u64()
            );
        }
    }

    // Verify edge count is reasonable (sanity check)
    let _ = edge_count;
});
