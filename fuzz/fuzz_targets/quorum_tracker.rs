// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::time::Duration;
use tensor_chain::QuorumTracker;

#[derive(Arbitrary, Debug)]
enum Op {
    RecordSuccess { node: u8 },
    RecordFailure { node: u8 },
    CheckReachable { node: u8 },
    HasQuorum { total_peers: u8 },
    ReachableCount,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    timeout_secs: u8,
    max_failures: u8,
    ops: Vec<Op>,
}

fuzz_target!(|input: FuzzInput| {
    // Create tracker with bounded parameters
    let timeout = Duration::from_secs((input.timeout_secs.max(1) % 60) as u64);
    let max_failures = (input.max_failures.max(1) % 10) as u32;

    let tracker = QuorumTracker::new(timeout, max_failures);

    // Execute operations (limit to 100 to bound execution time)
    for op in input.ops.iter().take(100) {
        match op {
            Op::RecordSuccess { node } => {
                let node_id = format!("node{}", node % 20);
                tracker.record_success(&node_id);
            },
            Op::RecordFailure { node } => {
                let node_id = format!("node{}", node % 20);
                tracker.record_failure(&node_id);
            },
            Op::CheckReachable { node } => {
                let node_id = format!("node{}", node % 20);
                let _ = tracker.is_reachable(&node_id);
            },
            Op::HasQuorum { total_peers } => {
                let total = (*total_peers as usize).clamp(1, 20);
                let has_quorum = tracker.has_quorum(total);

                // Verify invariant: has_quorum depends on reachable_count
                let reachable = tracker.reachable_count();
                let quorum_size = (total + 1).div_ceil(2);
                let expected = reachable + 1 >= quorum_size; // +1 for self

                assert_eq!(
                    has_quorum, expected,
                    "Quorum check inconsistent: reachable={}, total={}, quorum_size={}",
                    reachable, total, quorum_size
                );
            },
            Op::ReachableCount => {
                let count = tracker.reachable_count();
                // Count should never exceed number of unique nodes (20 max in this test)
                assert!(count <= 20, "Reachable count exceeded max nodes");
            },
        }
    }

    // Final invariant checks
    let reachable = tracker.reachable_count();
    assert!(reachable <= 20, "Final reachable count exceeded max nodes");
});
