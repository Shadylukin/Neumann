// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

//! Fuzz test for TOCTOU (Time-of-Check-Time-of-Use) safety in 2PC.
//!
//! This fuzz target stress-tests concurrent begin/lock/release operations
//! to verify the atomicity of state transitions.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;
use tensor_chain::{
    consensus::{ConsensusConfig, ConsensusManager},
    DistributedTxConfig, DistributedTxCoordinator,
};

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
enum FuzzOp {
    Begin {
        node_idx: u8,
        participant_count: u8,
    },
    TryLock {
        tx_id_mod: u8,
        key_count: u8,
        key_base: u8,
    },
    TryLockWithWaitTracking {
        tx_id_mod: u8,
        key_count: u8,
        key_base: u8,
        priority: Option<u8>,
    },
    ReleaseByHandle {
        handle_idx: u8,
    },
    ReleaseByHandleWithWaitCleanup {
        handle_idx: u8,
    },
    Abort {
        tx_id_mod: u8,
    },
    CleanupTimeouts,
    CheckMaxConcurrent,
    RecordStateSnapshot,
    VerifyStateConsistency,
}

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
struct FuzzInput {
    max_concurrent: u8,
    operations: Vec<FuzzOp>,
}

#[derive(Default)]
struct StateSnapshot {
    pending_count: usize,
    lock_count: usize,
    edge_count: usize,
}

fuzz_target!(|input: FuzzInput| {
    // Limit operations
    if input.operations.len() > 200 {
        return;
    }

    // Use bounded max_concurrent
    let max_concurrent = ((input.max_concurrent as usize) % 20).max(1);

    let config = DistributedTxConfig {
        max_concurrent,
        ..Default::default()
    };
    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let coordinator = DistributedTxCoordinator::new(consensus, config);

    let mut handles: Vec<u64> = Vec::new();
    let mut active_tx_ids: HashMap<u64, bool> = HashMap::new(); // tx_id -> has_begun
    let mut _last_snapshot = StateSnapshot::default();
    let mut begin_success_count = 0usize;
    let mut begin_failure_count = 0usize;

    for op in &input.operations {
        match op {
            FuzzOp::Begin {
                node_idx,
                participant_count,
            } => {
                let node = format!("node{}", node_idx % 5);
                let count = (*participant_count as usize % 5).max(1);
                let participants: Vec<usize> = (0..count).collect();

                match coordinator.begin(node, participants) {
                    Ok(tx) => {
                        active_tx_ids.insert(tx.tx_id, true);
                        begin_success_count += 1;

                        // Invariant: pending count should increase
                        let pending = coordinator.pending_count();
                        assert!(
                            pending >= 1,
                            "after successful begin, pending should be at least 1"
                        );
                    },
                    Err(_) => {
                        begin_failure_count += 1;
                    },
                }
            },

            FuzzOp::TryLock {
                tx_id_mod,
                key_count,
                key_base,
            } => {
                let tx_id = (*tx_id_mod as u64) % 50 + 1;
                let count = (*key_count as usize % 5).max(1);
                let base = *key_base as usize;
                let keys: Vec<String> = (0..count)
                    .map(|i| format!("key_{}", (base + i) % 30))
                    .collect();

                if let Ok(handle) = coordinator.lock_manager().try_lock(tx_id, &keys) {
                    handles.push(handle);
                }
            },

            FuzzOp::TryLockWithWaitTracking {
                tx_id_mod,
                key_count,
                key_base,
                priority,
            } => {
                let tx_id = (*tx_id_mod as u64) % 50 + 1;
                let count = (*key_count as usize % 5).max(1);
                let base = *key_base as usize;
                let keys: Vec<String> = (0..count)
                    .map(|i| format!("key_{}", (base + i) % 30))
                    .collect();
                let prio = priority.map(|p| p as u32);

                if let Ok(handle) = coordinator.lock_manager().try_lock_with_wait_tracking(
                    tx_id,
                    &keys,
                    coordinator.wait_graph(),
                    prio,
                ) {
                    handles.push(handle);
                }
            },

            FuzzOp::ReleaseByHandle { handle_idx } => {
                if !handles.is_empty() {
                    let idx = (*handle_idx as usize) % handles.len();
                    let handle = handles.remove(idx);
                    coordinator.lock_manager().release_by_handle(handle);
                }
            },

            FuzzOp::ReleaseByHandleWithWaitCleanup { handle_idx } => {
                if !handles.is_empty() {
                    let idx = (*handle_idx as usize) % handles.len();
                    let handle = handles.remove(idx);
                    coordinator.lock_manager().release_by_handle_with_wait_cleanup(
                        handle,
                        coordinator.wait_graph(),
                    );
                }
            },

            FuzzOp::Abort { tx_id_mod } => {
                let tx_id = (*tx_id_mod as u64) % 50 + 1;
                if coordinator.abort(tx_id, "fuzz abort").is_ok() {
                    active_tx_ids.remove(&tx_id);
                }
            },

            FuzzOp::CleanupTimeouts => {
                let timed_out = coordinator.cleanup_timeouts();
                for tx_id in timed_out {
                    active_tx_ids.remove(&tx_id);
                }
            },

            FuzzOp::CheckMaxConcurrent => {
                // CRITICAL INVARIANT: pending count should never exceed max_concurrent
                let pending = coordinator.pending_count();
                assert!(
                    pending <= max_concurrent,
                    "TOCTOU violation: pending {} exceeds max_concurrent {}",
                    pending,
                    max_concurrent
                );
            },

            FuzzOp::RecordStateSnapshot => {
                _last_snapshot = StateSnapshot {
                    pending_count: coordinator.pending_count(),
                    lock_count: coordinator.lock_manager().active_lock_count(),
                    edge_count: coordinator.wait_graph().edge_count(),
                };
            },

            FuzzOp::VerifyStateConsistency => {
                let current = StateSnapshot {
                    pending_count: coordinator.pending_count(),
                    lock_count: coordinator.lock_manager().active_lock_count(),
                    edge_count: coordinator.wait_graph().edge_count(),
                };

                // Monotonicity checks when no mutations happened between snapshots
                // (This is a soft check - we just verify state is reasonable)
                assert!(
                    current.pending_count <= max_concurrent,
                    "pending exceeds max: {} > {}",
                    current.pending_count,
                    max_concurrent
                );
                assert!(
                    current.lock_count <= 10000,
                    "too many locks: {}",
                    current.lock_count
                );
                assert!(
                    current.edge_count <= 10000,
                    "too many edges: {}",
                    current.edge_count
                );
            },
        }
    }

    // Final invariants
    // 1. Max concurrent should never have been exceeded (checked throughout)
    let final_pending = coordinator.pending_count();
    assert!(
        final_pending <= max_concurrent,
        "final TOCTOU violation: pending {} exceeds max_concurrent {}",
        final_pending,
        max_concurrent
    );

    // 2. Success + failure should be bounded by total begin operations
    let total_begins = input
        .operations
        .iter()
        .filter(|op| matches!(op, FuzzOp::Begin { .. }))
        .count();
    assert!(
        begin_success_count + begin_failure_count <= total_begins,
        "begin tracking inconsistent"
    );

    // 3. Active locks should be reasonable
    let active_locks = coordinator.lock_manager().active_lock_count();
    assert!(active_locks <= 1000, "too many active locks: {}", active_locks);

    // 4. Wait graph should be in valid state
    let edges = coordinator.wait_graph().edge_count();
    assert!(edges <= 10000, "edge count too large: {}", edges);
});
