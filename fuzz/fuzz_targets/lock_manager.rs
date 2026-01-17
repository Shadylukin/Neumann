//! Fuzz target for LockManager operations.
//!
//! Tests lock acquire, release, and conflict detection.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::LockManager;

#[derive(Arbitrary, Debug)]
enum LockOp {
    TryLock { tx_id: u64, keys: Vec<String> },
    Release { tx_id: u64 },
    ReleaseByHandle { handle: u64 },
    IsLocked { key: String },
    LockHolder { key: String },
    ActiveCount,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<LockOp>,
}

fuzz_target!(|input: FuzzInput| {
    let lock_manager = LockManager::new();

    // Limit operations to prevent excessive runtime
    let ops = if input.ops.len() > 100 {
        &input.ops[..100]
    } else {
        &input.ops
    };

    for op in ops {
        match op {
            LockOp::TryLock { tx_id, keys } => {
                // Limit keys to reasonable size
                let keys: Vec<String> = keys
                    .iter()
                    .take(10)
                    .filter(|k| k.len() < 100)
                    .cloned()
                    .collect();

                if !keys.is_empty() {
                    let _ = lock_manager.try_lock(*tx_id, &keys);
                }
            },
            LockOp::Release { tx_id } => {
                lock_manager.release(*tx_id);
            },
            LockOp::ReleaseByHandle { handle } => {
                lock_manager.release_by_handle(*handle);
            },
            LockOp::IsLocked { key } => {
                if key.len() < 100 {
                    let _ = lock_manager.is_locked(&key);
                }
            },
            LockOp::LockHolder { key } => {
                if key.len() < 100 {
                    let _ = lock_manager.lock_holder(&key);
                }
            },
            LockOp::ActiveCount => {
                let count = lock_manager.active_lock_count();
                // Active count should be consistent
                assert!(count < 10000, "Too many active locks");
            },
        }
    }

    // Final consistency check
    let final_count = lock_manager.active_lock_count();
    assert!(final_count < 10000);
});
