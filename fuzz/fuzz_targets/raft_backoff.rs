// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz test for Raft quorum tracker and backoff behavior.
//!
//! Exercises the `QuorumTracker` with arbitrary sequences of success/failure
//! recordings, reachability checks, and quorum decisions. Verifies invariants
//! around failure counting, reachability thresholds, and quorum arithmetic.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::QuorumTracker;

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
enum FuzzOp {
    /// Record a successful response from a peer.
    RecordSuccess { peer_idx: u8 },
    /// Record a failed response from a peer.
    RecordFailure { peer_idx: u8 },
    /// Check if a peer is reachable.
    CheckReachable { peer_idx: u8 },
    /// Mark a peer as reachable (manual override).
    MarkReachable { peer_idx: u8 },
    /// Check quorum with a given total peer count.
    CheckQuorum { total_peers: u8 },
    /// Get the count of reachable peers.
    ReachableCount,
    /// Get unreachable peer list.
    UnreachablePeers,
    /// Reset all tracking state.
    Reset,
}

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
struct FuzzInput {
    /// Timeout in milliseconds (clamped).
    timeout_ms: u16,
    /// Max consecutive failures before unreachable.
    max_failures: u8,
    operations: Vec<FuzzOp>,
}

fn peer_name(idx: u8) -> String {
    format!("peer_{}", idx % 10)
}

fuzz_target!(|input: FuzzInput| {
    if input.operations.len() > 200 {
        return;
    }

    let timeout = std::time::Duration::from_millis(u64::from(input.timeout_ms).max(10));
    let max_failures = u32::from(input.max_failures).max(1).min(20);

    let tracker = QuorumTracker::new(timeout, max_failures);

    // Track our own view of failure counts for invariant checking
    let mut failure_counts: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();
    let mut known_peers: std::collections::HashSet<String> = std::collections::HashSet::new();

    for op in &input.operations {
        match op {
            FuzzOp::RecordSuccess { peer_idx } => {
                let name = peer_name(*peer_idx);
                tracker.record_success(&name);
                failure_counts.insert(name.clone(), 0);
                known_peers.insert(name);
            }
            FuzzOp::RecordFailure { peer_idx } => {
                let name = peer_name(*peer_idx);
                tracker.record_failure(&name);
                *failure_counts.entry(name.clone()).or_insert(0) += 1;
                known_peers.insert(name);
            }
            FuzzOp::CheckReachable { peer_idx } => {
                let name = peer_name(*peer_idx);
                let reachable = tracker.is_reachable(&name);

                // If we've recorded >= max_failures consecutive failures,
                // the peer must NOT be reachable
                let our_failures = failure_counts.get(&name).copied().unwrap_or(0);
                if our_failures >= max_failures {
                    assert!(
                        !reachable,
                        "Peer {} should be unreachable after {} failures (max={})",
                        name, our_failures, max_failures
                    );
                }
            }
            FuzzOp::MarkReachable { peer_idx } => {
                let name = peer_name(*peer_idx);
                tracker.mark_reachable(&name);
                known_peers.insert(name);
            }
            FuzzOp::CheckQuorum { total_peers } => {
                let total = (*total_peers as usize) % 20;
                let has_quorum = tracker.has_quorum(total);

                // Quorum requires (total_nodes / 2) + 1 where total_nodes = total + 1 (self)
                let total_nodes = total + 1;
                let required = total_nodes / 2 + 1;
                let reachable_plus_self = tracker.reachable_count() + 1;

                if reachable_plus_self >= required {
                    assert!(
                        has_quorum,
                        "Should have quorum: reachable+self={} >= required={}",
                        reachable_plus_self, required
                    );
                }
            }
            FuzzOp::ReachableCount => {
                let count = tracker.reachable_count();
                // Reachable count cannot exceed known peers
                assert!(
                    count <= known_peers.len(),
                    "Reachable {} > known peers {}",
                    count,
                    known_peers.len()
                );
            }
            FuzzOp::UnreachablePeers => {
                let unreachable = tracker.unreachable_peers();
                // Every unreachable peer must have exceeded the failure threshold
                // or timed out (can't verify timeout precisely in fuzzing)
                for peer in &unreachable {
                    let reachable = tracker.is_reachable(peer);
                    assert!(
                        !reachable,
                        "Peer {} listed as unreachable but is_reachable returns true",
                        peer
                    );
                }
            }
            FuzzOp::Reset => {
                tracker.reset();
                failure_counts.clear();
                // known_peers intentionally not cleared -- reset clears internal
                // state but our tracking remembers what was ever seen
            }
        }
    }

    // Final invariant: reachable_count is bounded
    assert!(tracker.reachable_count() <= known_peers.len());
});
