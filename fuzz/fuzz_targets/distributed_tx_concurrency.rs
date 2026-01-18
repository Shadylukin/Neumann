//! Fuzz target for distributed transaction concurrency safety.
//!
//! Tests the fixes for TOCTOU race conditions:
//! - Max concurrent transaction limit enforcement
//! - Wait-for graph atomicity with lock acquisition

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::Ordering;
use tensor_chain::{
    deadlock::WaitForGraph, ConsensusConfig, ConsensusManager, DistributedTxConfig,
    DistributedTxCoordinator,
};

#[derive(Debug, Arbitrary)]
enum ConcurrencyOp {
    Begin { participant_count: u8 },
    TryLock { tx_idx: u8, key_count: u8 },
    TryLockWithWait { tx_idx: u8, key_count: u8, priority: Option<u8> },
    Release { tx_idx: u8 },
    ReleaseOrphaned { partition_start_ms: u32 },
    Abort { tx_idx: u8 },
}

fuzz_target!(|ops: Vec<ConcurrencyOp>| {
    // Limit operations to prevent excessive runtime
    if ops.len() > 50 {
        return;
    }

    let consensus = ConsensusManager::new(ConsensusConfig::default());
    let coordinator = DistributedTxCoordinator::new(
        consensus,
        DistributedTxConfig {
            max_concurrent: 10,
            ..Default::default()
        },
    );
    let wait_graph = WaitForGraph::new();

    let mut active_tx_ids: Vec<u64> = Vec::new();

    for op in ops {
        match op {
            ConcurrencyOp::Begin { participant_count } => {
                let participants: Vec<usize> =
                    (0..((participant_count % 4) + 1) as usize).collect();
                if let Ok(tx) = coordinator.begin("fuzz_node".to_string(), participants) {
                    active_tx_ids.push(tx.tx_id);
                    if active_tx_ids.len() > 20 {
                        active_tx_ids.remove(0);
                    }
                }
            }
            ConcurrencyOp::TryLock { tx_idx, key_count } => {
                if !active_tx_ids.is_empty() {
                    let tx = active_tx_ids[tx_idx as usize % active_tx_ids.len()];
                    let keys: Vec<String> = (0..((key_count % 3) + 1))
                        .map(|i| format!("key_{}", i))
                        .collect();
                    let _ = coordinator.lock_manager().try_lock(tx, &keys);
                }
            }
            ConcurrencyOp::TryLockWithWait {
                tx_idx,
                key_count,
                priority,
            } => {
                if !active_tx_ids.is_empty() {
                    let tx = active_tx_ids[tx_idx as usize % active_tx_ids.len()];
                    let keys: Vec<String> = (0..((key_count % 3) + 1))
                        .map(|i| format!("key_{}", i))
                        .collect();
                    let _ = coordinator.lock_manager().try_lock_with_wait_tracking(
                        tx,
                        &keys,
                        &wait_graph,
                        priority.map(|p| p as u32),
                    );
                }
            }
            ConcurrencyOp::Release { tx_idx } => {
                if !active_tx_ids.is_empty() {
                    let tx = active_tx_ids[tx_idx as usize % active_tx_ids.len()];
                    coordinator.lock_manager().release(tx);
                }
            }
            ConcurrencyOp::ReleaseOrphaned { partition_start_ms } => {
                coordinator.release_orphaned_locks(partition_start_ms as u64);
            }
            ConcurrencyOp::Abort { tx_idx } => {
                if !active_tx_ids.is_empty() {
                    let tx = active_tx_ids[tx_idx as usize % active_tx_ids.len()];
                    let _ = coordinator.abort(tx, "fuzz abort");
                }
            }
        }
    }

    // Invariant checks
    let pending = coordinator.pending_count();
    assert!(
        pending <= 10,
        "Pending count {} exceeds max_concurrent 10",
        pending
    );

    let lock_count = coordinator.lock_manager().active_lock_count();
    assert!(
        lock_count < 1000,
        "Lock count {} is suspiciously high",
        lock_count
    );

    // Stats should be consistent
    let stats = coordinator.stats();
    let started = stats.started.load(Ordering::Relaxed);
    let committed = stats.committed.load(Ordering::Relaxed);
    let aborted = stats.aborted.load(Ordering::Relaxed);
    assert!(
        committed + aborted <= started,
        "Committed {} + aborted {} > started {}",
        committed,
        aborted,
        started
    );
});
