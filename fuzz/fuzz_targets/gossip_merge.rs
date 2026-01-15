#![no_main]

use libfuzzer_sys::fuzz_target;
use tensor_chain::{GossipNodeState, LWWMembershipState, NodeHealth};

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    let num_states = (data[0] % 16) as usize + 1;
    let mut offset = 1;

    let mut states = Vec::with_capacity(num_states);

    for i in 0..num_states {
        if offset + 4 > data.len() {
            break;
        }

        // Extract fields from data
        let health_idx = data.get(offset).copied().unwrap_or(0) % 4;
        let health = match health_idx {
            0 => NodeHealth::Healthy,
            1 => NodeHealth::Degraded,
            2 => NodeHealth::Failed,
            _ => NodeHealth::Unknown,
        };

        let timestamp = data.get(offset + 1).copied().unwrap_or(0) as u64;
        let incarnation = data.get(offset + 2).copied().unwrap_or(0) as u64;
        let node_id = format!("node{}", i);

        states.push(GossipNodeState::new(node_id, health, timestamp, incarnation));
        offset += 3;
    }

    if states.is_empty() {
        return;
    }

    // Create initial state
    let mut crdt = LWWMembershipState::new();

    // Merge states
    let changed = crdt.merge(&states);

    // Verify invariants
    assert!(changed.len() <= states.len());

    // Verify all changed nodes are in the final state
    for node_id in &changed {
        assert!(crdt.get(node_id).is_some());
    }

    // Verify merge is idempotent
    let changed_again = crdt.merge(&states);
    assert!(
        changed_again.is_empty(),
        "Re-merging same states should not change anything"
    );

    // Verify counts add up
    let (healthy, degraded, failed) = crdt.count_by_health();
    assert_eq!(healthy + degraded + failed, crdt.len());
});
