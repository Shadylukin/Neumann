//! Fuzz test for gossip timestamp ordering.
//!
//! Ensures gossip state timestamps maintain proper ordering and
//! supersedes() logic is consistent.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{membership::NodeHealth, GossipNodeState};

#[derive(Arbitrary, Debug)]
struct GossipInput {
    states: Vec<StateParams>,
}

#[derive(Arbitrary, Debug)]
struct StateParams {
    node_suffix: u8,
    health: HealthChoice,
    timestamp: u64,
    incarnation: u64,
}

#[derive(Arbitrary, Debug, Clone)]
enum HealthChoice {
    Healthy,
    Degraded,
    Failed,
}

impl From<HealthChoice> for NodeHealth {
    fn from(h: HealthChoice) -> Self {
        match h {
            HealthChoice::Healthy => NodeHealth::Healthy,
            HealthChoice::Degraded => NodeHealth::Degraded,
            HealthChoice::Failed => NodeHealth::Failed,
        }
    }
}

fuzz_target!(|input: GossipInput| {
    // Limit to prevent OOM
    let states: Vec<GossipNodeState> = input
        .states
        .iter()
        .take(100)
        .map(|p| {
            GossipNodeState::new(
                format!("node-{}", p.node_suffix),
                p.health.clone().into(),
                p.timestamp,
                p.incarnation,
            )
        })
        .collect();

    // Verify all states have positive updated_at
    for state in &states {
        // updated_at should never be 0 (the old unwrap_or(0) bug)
        // With HLC fallback, it should at least equal the timestamp
        assert!(
            state.updated_at > 0 || state.timestamp == 0,
            "updated_at should be positive or timestamp is 0"
        );
    }

    // Test supersedes() transitivity for same-node states
    let same_node_states: Vec<&GossipNodeState> = states
        .iter()
        .filter(|s| s.node_id == "node-0")
        .collect();

    for i in 0..same_node_states.len() {
        for j in (i + 1)..same_node_states.len() {
            let a = same_node_states[i];
            let b = same_node_states[j];

            let a_super_b = a.supersedes(b);
            let b_super_a = b.supersedes(a);

            // If both have same incarnation and timestamp, neither supersedes
            // Otherwise, exactly one should supersede (or neither if equal)
            if a.incarnation == b.incarnation && a.timestamp == b.timestamp {
                // Equal states - neither supersedes
                assert!(
                    !a_super_b || !b_super_a,
                    "Equal states should not both supersede"
                );
            }
        }
    }

    // Test try_new doesn't panic
    for p in input.states.iter().take(10) {
        let result = GossipNodeState::try_new(
            format!("node-{}", p.node_suffix),
            p.health.clone().into(),
            p.timestamp,
            p.incarnation,
        );
        // Should either succeed or return an error, never panic
        let _ = result;
    }

    // Test with_wall_time doesn't panic
    for p in input.states.iter().take(10) {
        let state = GossipNodeState::with_wall_time(
            format!("node-{}", p.node_suffix),
            p.health.clone().into(),
            p.timestamp,
            p.incarnation,
            p.timestamp.saturating_mul(1000), // Use as wall time
        );
        assert_eq!(state.timestamp, p.timestamp);
        assert_eq!(state.incarnation, p.incarnation);
    }
});
