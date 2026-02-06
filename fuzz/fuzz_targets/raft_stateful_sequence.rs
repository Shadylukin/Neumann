// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Stateful sequence fuzzer for Raft operations.
//!
//! Tests that:
//! - A sequence of operations applied to a Raft cluster never causes a panic
//! - At most one leader exists per term (Raft safety invariant)
//! - State transitions are consistent across operations

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;
use tensor_chain::{
    network::{AppendEntries, RequestVote},
    MemoryTransport, Message, RaftConfig, RaftNode, RaftState,
};
use tensor_store::SparseVector;

#[derive(Debug, Arbitrary)]
enum RaftOp {
    StartElection { node_idx: u8 },
    HandleRequestVote { target_idx: u8, term: u16 },
    HandleAppendEntries { target_idx: u8, term: u16 },
    HandlePing { target_idx: u8, term: u16 },
}

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    node_count: u8,
    operations: Vec<RaftOp>,
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = FuzzInput::arbitrary(&mut arbitrary::Unstructured::new(data)) else {
        return;
    };
    let node_count = (input.node_count % 3 + 3) as usize; // 3, 4, or 5
    if input.operations.len() > 100 {
        return;
    }

    let node_ids: Vec<String> = (0..node_count).map(|i| format!("node{}", i)).collect();
    let transports: Vec<Arc<MemoryTransport>> = node_ids
        .iter()
        .map(|id| Arc::new(MemoryTransport::new(id.clone())))
        .collect();

    // Connect all transports
    for (i, t1) in transports.iter().enumerate() {
        for (j, t2) in transports.iter().enumerate() {
            if i != j {
                t1.connect_to(node_ids[j].clone(), t2.sender());
            }
        }
    }

    let config = RaftConfig {
        election_timeout: (100, 200),
        heartbeat_interval: 50,
        ..RaftConfig::default()
    };

    let nodes: Vec<Arc<RaftNode>> = node_ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let peers: Vec<String> = node_ids.iter().filter(|p| *p != id).cloned().collect();
            Arc::new(RaftNode::new(
                id.clone(),
                peers,
                transports[i].clone(),
                config.clone(),
            ))
        })
        .collect();

    for op in &input.operations {
        match op {
            RaftOp::StartElection { node_idx } => {
                let idx = (*node_idx as usize) % node_count;
                nodes[idx].start_election();
            }
            RaftOp::HandleRequestVote { target_idx, term } => {
                let idx = (*target_idx as usize) % node_count;
                let from_idx = (idx + 1) % node_count;
                let msg = Message::RequestVote(RequestVote {
                    term: *term as u64,
                    candidate_id: node_ids[from_idx].clone(),
                    last_log_index: 0,
                    last_log_term: 0,
                    state_embedding: SparseVector::new(0),
                });
                let _ = nodes[idx].handle_message(&node_ids[from_idx], &msg);
            }
            RaftOp::HandleAppendEntries { target_idx, term } => {
                let idx = (*target_idx as usize) % node_count;
                let from_idx = (idx + 1) % node_count;
                let msg = Message::AppendEntries(AppendEntries {
                    term: *term as u64,
                    leader_id: node_ids[from_idx].clone(),
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: vec![],
                    leader_commit: 0,
                    block_embedding: None,
                });
                let _ = nodes[idx].handle_message(&node_ids[from_idx], &msg);
            }
            RaftOp::HandlePing { target_idx, term } => {
                let idx = (*target_idx as usize) % node_count;
                let from_idx = (idx + 1) % node_count;
                let msg = Message::Ping {
                    term: *term as u64,
                };
                let _ = nodes[idx].handle_message(&node_ids[from_idx], &msg);
            }
        }

        // Invariant: at most one leader per term
        let mut leaders_per_term: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::new();
        for node in &nodes {
            if node.state() == RaftState::Leader {
                *leaders_per_term.entry(node.current_term()).or_insert(0) += 1;
            }
        }
        for (_term, count) in &leaders_per_term {
            assert!(*count <= 1, "Multiple leaders in same term!");
        }
    }
});
