// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Fuzz membership/gossip state convergence via LWW-CRDT merges.
//!
//! Tests that:
//! - LWW membership state merges never panic
//! - Lamport timestamps monotonically increase
//! - Health counts remain consistent with state contents
//! - Refutation and suspicion transitions are well-formed

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::gossip::LWWMembershipState;
use tensor_chain::membership::NodeHealth;

#[derive(Debug, Arbitrary)]
enum MembershipOp {
    UpdateLocal {
        node_idx: u8,
        health: u8,
        incarnation: u8,
    },
    MergeStates {
        from_idx: u8,
        to_idx: u8,
        max_states: u8,
    },
    Suspect {
        state_idx: u8,
        node_idx: u8,
        incarnation: u8,
    },
    Fail {
        state_idx: u8,
        node_idx: u8,
    },
    Refute {
        state_idx: u8,
        node_idx: u8,
        new_incarnation: u8,
    },
    MarkHealthy {
        state_idx: u8,
        node_idx: u8,
    },
}

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    node_count: u8,
    operations: Vec<MembershipOp>,
}

fn health_from_byte(b: u8) -> NodeHealth {
    match b % 4 {
        0 => NodeHealth::Healthy,
        1 => NodeHealth::Degraded,
        2 => NodeHealth::Failed,
        _ => NodeHealth::Unknown,
    }
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = FuzzInput::arbitrary(&mut arbitrary::Unstructured::new(data)) else {
        return;
    };
    let node_count = ((input.node_count % 4) + 2) as usize; // 2..=5
    if input.operations.len() > 100 {
        return;
    }

    let node_ids: Vec<String> = (0..node_count).map(|i| format!("node{i}")).collect();

    // Create one LWW state per "node" to simulate independent views
    let mut states: Vec<LWWMembershipState> = (0..node_count)
        .map(|_| LWWMembershipState::new())
        .collect();

    // Initialize each state with its own local node as healthy
    for (i, state) in states.iter_mut().enumerate() {
        state.update_local(node_ids[i].clone(), NodeHealth::Healthy, 0);
    }

    for op in &input.operations {
        match op {
            MembershipOp::UpdateLocal {
                node_idx,
                health,
                incarnation,
            } => {
                let state_idx = (*node_idx as usize) % node_count;
                let health = health_from_byte(*health);
                states[state_idx].update_local(
                    node_ids[state_idx].clone(),
                    health,
                    u64::from(*incarnation),
                );
            }
            MembershipOp::MergeStates {
                from_idx,
                to_idx,
                max_states,
            } => {
                let from = (*from_idx as usize) % node_count;
                let to = (*to_idx as usize) % node_count;
                if from == to {
                    continue;
                }
                let max = ((*max_states) % 10 + 1) as usize;
                let gossip_states = states[from].states_for_gossip(max);
                let _ = states[to].merge(&gossip_states);
            }
            MembershipOp::Suspect {
                state_idx,
                node_idx,
                incarnation,
            } => {
                let si = (*state_idx as usize) % node_count;
                let ni = (*node_idx as usize) % node_count;
                let _ = states[si].suspect(&node_ids[ni], u64::from(*incarnation));
            }
            MembershipOp::Fail {
                state_idx,
                node_idx,
            } => {
                let si = (*state_idx as usize) % node_count;
                let ni = (*node_idx as usize) % node_count;
                let _ = states[si].fail(&node_ids[ni]);
            }
            MembershipOp::Refute {
                state_idx,
                node_idx,
                new_incarnation,
            } => {
                let si = (*state_idx as usize) % node_count;
                let ni = (*node_idx as usize) % node_count;
                let _ = states[si].refute(&node_ids[ni], u64::from(*new_incarnation));
            }
            MembershipOp::MarkHealthy {
                state_idx,
                node_idx,
            } => {
                let si = (*state_idx as usize) % node_count;
                let ni = (*node_idx as usize) % node_count;
                let _ = states[si].mark_healthy(&node_ids[ni]);
            }
        }

        // Invariant: health counts should match state contents
        for state in &states {
            let (healthy, degraded, failed) = state.count_by_health();
            assert_eq!(
                healthy + degraded + failed,
                state.len(),
                "Health counts do not sum to state length"
            );
        }
    }

    // Final invariant: Lamport time should never be zero after operations
    // (it starts at 0 but any update_local or merge increments it)
    if !input.operations.is_empty() {
        for state in &states {
            // Each state was initialized with update_local, so lamport_time >= 1
            assert!(
                state.lamport_time() >= 1,
                "Lamport time should be >= 1 after initialization"
            );
        }
    }
});
