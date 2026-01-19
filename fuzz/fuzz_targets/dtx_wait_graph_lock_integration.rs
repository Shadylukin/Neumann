#![no_main]

//! Fuzz test for LockManager and WaitForGraph integration.
//!
//! This fuzz target tests that the wait-for graph is correctly maintained
//! during lock acquisition and release operations.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{LockManager, WaitForGraph};

#[derive(Debug, Arbitrary)]
enum FuzzOp {
    TryLockWithWaitTracking {
        tx_id_mod: u8,
        key_count: u8,
        key_base: u8,
        priority: Option<u8>,
    },
    TryLockSimple {
        tx_id_mod: u8,
        key_count: u8,
        key_base: u8,
    },
    ReleaseByHandle {
        handle_idx: u8,
    },
    ReleaseByHandleWithWaitCleanup {
        handle_idx: u8,
    },
    ReleaseTransaction {
        tx_id_mod: u8,
    },
    RemoveFromWaitGraph {
        tx_id_mod: u8,
    },
    DetectCycles,
    WouldCreateCycle {
        waiter_mod: u8,
        holder_mod: u8,
    },
    CheckInvariants,
}

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    operations: Vec<FuzzOp>,
}

fuzz_target!(|input: FuzzInput| {
    // Limit operation count
    if input.operations.len() > 200 {
        return;
    }

    let lock_manager = LockManager::new();
    let wait_graph = WaitForGraph::new();

    let mut handles: Vec<u64> = Vec::new();

    for op in &input.operations {
        match op {
            FuzzOp::TryLockWithWaitTracking {
                tx_id_mod,
                key_count,
                key_base,
                priority,
            } => {
                let tx_id = (*tx_id_mod as u64) % 20 + 1;
                let count = (*key_count as usize % 5).max(1);
                let base = *key_base as usize;
                let keys: Vec<String> = (0..count).map(|i| format!("key_{}", (base + i) % 50)).collect();
                let prio = priority.map(|p| p as u32);

                if let Ok(handle) = lock_manager.try_lock_with_wait_tracking(
                    tx_id, &keys, &wait_graph, prio,
                ) {
                    handles.push(handle);
                }
            },

            FuzzOp::TryLockSimple {
                tx_id_mod,
                key_count,
                key_base,
            } => {
                let tx_id = (*tx_id_mod as u64) % 20 + 1;
                let count = (*key_count as usize % 5).max(1);
                let base = *key_base as usize;
                let keys: Vec<String> = (0..count).map(|i| format!("key_{}", (base + i) % 50)).collect();

                if let Ok(handle) = lock_manager.try_lock(tx_id, &keys) {
                    handles.push(handle);
                }
            },

            FuzzOp::ReleaseByHandle { handle_idx } => {
                if !handles.is_empty() {
                    let idx = (*handle_idx as usize) % handles.len();
                    let handle = handles.remove(idx);
                    lock_manager.release_by_handle(handle);
                }
            },

            FuzzOp::ReleaseByHandleWithWaitCleanup { handle_idx } => {
                if !handles.is_empty() {
                    let idx = (*handle_idx as usize) % handles.len();
                    let handle = handles.remove(idx);
                    lock_manager.release_by_handle_with_wait_cleanup(handle, &wait_graph);
                }
            },

            FuzzOp::ReleaseTransaction { tx_id_mod } => {
                let tx_id = (*tx_id_mod as u64) % 20 + 1;
                lock_manager.release(tx_id);
            },

            FuzzOp::RemoveFromWaitGraph { tx_id_mod } => {
                let tx_id = (*tx_id_mod as u64) % 20 + 1;
                wait_graph.remove_transaction(tx_id);
            },

            FuzzOp::DetectCycles => {
                let _ = wait_graph.detect_cycles();
            },

            FuzzOp::WouldCreateCycle { waiter_mod, holder_mod } => {
                let waiter = (*waiter_mod as u64) % 20 + 1;
                let holder = (*holder_mod as u64) % 20 + 1;
                let _ = wait_graph.would_create_cycle(waiter, holder);
            },

            FuzzOp::CheckInvariants => {
                // Invariant 1: Edge count should be bounded
                let edge_count = wait_graph.edge_count();
                assert!(edge_count <= 10000, "edge count too large");

                // Invariant 2: Active lock count should be bounded
                let lock_count = lock_manager.active_lock_count();
                assert!(lock_count <= 10000, "lock count too large");

                // Invariant 3: For any transaction in wait graph that also holds locks,
                // those should be consistent
                for tx_id in 1..=20u64 {
                    let waiting_for = wait_graph.waiting_for(tx_id);
                    let _waiting_on = wait_graph.waiting_on(tx_id);
                    let keys = lock_manager.keys_for_transaction(tx_id);

                    // If a transaction holds locks, it should not be waiting for itself
                    if !keys.is_empty() {
                        assert!(
                            !waiting_for.contains(&tx_id),
                            "transaction {} holds locks but waits for itself",
                            tx_id
                        );
                    }
                }
            },
        }
    }

    // Final invariant check
    let edge_count = wait_graph.edge_count();
    let tx_count = wait_graph.transaction_count();
    let lock_count = lock_manager.active_lock_count();

    // Edge count should be bounded by transaction count squared
    assert!(
        edge_count as usize <= tx_count.saturating_mul(tx_count),
        "edge count {} exceeds theoretical maximum for {} transactions",
        edge_count,
        tx_count
    );

    // Lock count should be reasonable
    assert!(lock_count <= 1000, "too many active locks: {}", lock_count);
});
