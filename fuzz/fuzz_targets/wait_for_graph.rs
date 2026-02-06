// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{DeadlockDetector, DeadlockDetectorConfig, VictimSelectionPolicy};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    operations: Vec<GraphOp>,
    config: FuzzConfig,
}

#[derive(Arbitrary, Debug)]
struct FuzzConfig {
    max_cycle_length: u8,
    policy: u8,
}

#[derive(Arbitrary, Debug)]
enum GraphOp {
    /// Add a wait edge (waiter waiting for holder).
    AddWait {
        waiter: u16,
        holder: u16,
        priority: Option<u8>,
    },
    /// Remove a specific wait edge.
    RemoveWait { waiter: u16, holder: u16 },
    /// Remove all edges for a transaction.
    RemoveTransaction { tx_id: u16 },
    /// Detect cycles in the graph.
    DetectCycles,
    /// Check if adding an edge would create a cycle.
    WouldCreateCycle { waiter: u16, holder: u16 },
    /// Query transactions waiting for a given transaction.
    WaitingFor { tx_id: u16 },
    /// Query transactions that a given transaction is waiting on.
    WaitingOn { tx_id: u16 },
    /// Clear the entire graph.
    Clear,
    /// Get edge count.
    EdgeCount,
    /// Get transaction count.
    TransactionCount,
}

fn policy_from_byte(b: u8) -> VictimSelectionPolicy {
    match b % 4 {
        0 => VictimSelectionPolicy::Youngest,
        1 => VictimSelectionPolicy::Oldest,
        2 => VictimSelectionPolicy::LowestPriority,
        _ => VictimSelectionPolicy::MostLocks,
    }
}

fuzz_target!(|input: FuzzInput| {
    // Limit operations to prevent OOM
    if input.operations.len() > 1000 {
        return;
    }

    let config = DeadlockDetectorConfig::default()
        .with_max_cycle_length(input.config.max_cycle_length.max(2) as usize)
        .with_policy(policy_from_byte(input.config.policy));

    let detector = DeadlockDetector::new(config);
    let graph = detector.graph();

    for op in &input.operations {
        match op {
            GraphOp::AddWait {
                waiter,
                holder,
                priority,
            } => {
                let waiter = *waiter as u64;
                let holder = *holder as u64;
                // Skip self-waits (they're rejected anyway)
                if waiter != holder {
                    graph.add_wait(waiter, holder, priority.map(|p| p as u32));
                }
            },

            GraphOp::RemoveWait { waiter, holder } => {
                graph.remove_wait(*waiter as u64, *holder as u64);
            },

            GraphOp::RemoveTransaction { tx_id } => {
                graph.remove_transaction(*tx_id as u64);
            },

            GraphOp::DetectCycles => {
                let cycles = detector.detect();

                // Verify each cycle is valid
                for deadlock in &cycles {
                    // Cycle should have at least 2 elements
                    assert!(
                        deadlock.cycle.len() >= 2 || deadlock.cycle.is_empty(),
                        "invalid cycle length"
                    );

                    // Victim should be in the cycle
                    if !deadlock.cycle.is_empty() {
                        assert!(
                            deadlock.cycle.contains(&deadlock.victim_tx_id),
                            "victim not in cycle"
                        );
                    }
                }
            },

            GraphOp::WouldCreateCycle { waiter, holder } => {
                let waiter = *waiter as u64;
                let holder = *holder as u64;
                let would_cycle = graph.would_create_cycle(waiter, holder);

                // Self-wait should always create cycle
                if waiter == holder {
                    assert!(would_cycle, "self-wait should always create cycle");
                }
            },

            GraphOp::WaitingFor { tx_id } => {
                let waiting = graph.waiting_for(*tx_id as u64);
                // Just ensure it doesn't panic and returns a set
                let _ = waiting.len();
            },

            GraphOp::WaitingOn { tx_id } => {
                let waiting = graph.waiting_on(*tx_id as u64);
                let _ = waiting.len();
            },

            GraphOp::Clear => {
                graph.clear();
                assert!(graph.is_empty(), "graph should be empty after clear");
                assert_eq!(graph.edge_count(), 0, "edge count should be 0 after clear");
            },

            GraphOp::EdgeCount => {
                let count = graph.edge_count();
                // Edge count should be non-negative (always true for usize)
                let _ = count;
            },

            GraphOp::TransactionCount => {
                let count = graph.transaction_count();
                let _ = count;
            },
        }
    }

    // Final consistency checks
    let edge_count = graph.edge_count();
    let tx_count = graph.transaction_count();

    // Edge count should not exceed tx_count^2
    assert!(
        edge_count <= tx_count * tx_count,
        "edge count {} exceeds max possible {} for {} transactions",
        edge_count,
        tx_count * tx_count,
        tx_count
    );
});
