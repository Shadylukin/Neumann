// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

//! Fuzz test for partition merge session deduplication.
//!
//! Exercises `PartitionMergeManager` with arbitrary sequences of session
//! creation, advancement, repartition events, and completion. Verifies
//! invariants around session ID uniqueness, partition generation tracking,
//! and repartition detection during active merges.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{MergePhase, PartitionMergeConfig, PartitionMergeManager};

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
enum FuzzOp {
    /// Start a new merge session with the given healed nodes.
    StartMerge { node_count: u8 },
    /// Advance a session by ID.
    AdvanceSession { session_idx: u8 },
    /// Trigger a partition event (changes generation).
    PartitionEvent,
    /// Check repartition status for a session.
    CheckRepartitioned { session_idx: u8 },
    /// Complete a session.
    CompleteSession { session_idx: u8 },
    /// Fail a session.
    FailSession { session_idx: u8 },
    /// Query active session count.
    ActiveCount,
    /// Get session phase.
    GetPhase { session_idx: u8 },
}

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
struct FuzzInput {
    /// Cooldown in ms (low for fuzzing).
    cooldown_ms: u8,
    /// Max concurrent merges.
    max_concurrent: u8,
    operations: Vec<FuzzOp>,
}

fn node_name(idx: u8) -> String {
    format!("node_{}", idx % 10)
}

fuzz_target!(|input: FuzzInput| {
    if input.operations.len() > 150 {
        return;
    }

    let config = PartitionMergeConfig {
        merge_cooldown_ms: u64::from(input.cooldown_ms).max(1),
        max_concurrent_merges: usize::from(input.max_concurrent).max(1).min(10),
        ..PartitionMergeConfig::default()
    };

    let manager = PartitionMergeManager::new("local_node".to_string(), config.clone());

    // Track created sessions for indexed access
    let mut session_ids: Vec<u64> = Vec::new();
    let mut completed_sessions: std::collections::HashSet<u64> =
        std::collections::HashSet::new();
    let mut failed_sessions: std::collections::HashSet<u64> =
        std::collections::HashSet::new();
    for op in &input.operations {
        match op {
            FuzzOp::StartMerge { node_count } => {
                let count = (usize::from(*node_count) % 5).max(1);
                let nodes: Vec<String> = (0..count).map(|i| node_name(i as u8)).collect();

                if let Some(sid) = manager.start_merge(nodes) {
                    // Session IDs must be unique
                    assert!(
                        !session_ids.contains(&sid),
                        "Duplicate session ID: {}",
                        sid
                    );
                    session_ids.push(sid);
                }
            }

            FuzzOp::AdvanceSession { session_idx } => {
                if session_ids.is_empty() {
                    continue;
                }
                let sid = session_ids[usize::from(*session_idx) % session_ids.len()];
                if completed_sessions.contains(&sid) || failed_sessions.contains(&sid) {
                    continue;
                }

                if let Some(phase) = manager.advance_session(sid) {
                    // Phase must be a valid variant
                    match phase {
                        MergePhase::Failed => {
                            failed_sessions.insert(sid);
                        }
                        MergePhase::Completed => {
                            // advance_session itself doesn't complete --
                            // it transitions through phases
                        }
                        _ => {}
                    }
                }
            }

            FuzzOp::PartitionEvent => {
                manager.notify_partition_event();
            }

            FuzzOp::CheckRepartitioned { session_idx } => {
                if session_ids.is_empty() {
                    continue;
                }
                let sid = session_ids[usize::from(*session_idx) % session_ids.len()];

                let repartitioned = manager.is_repartitioned(sid);

                // If no partition events happened since session creation,
                // it cannot be repartitioned. But we can't track the exact
                // generation per session here, so just verify it doesn't panic.
                let _ = repartitioned;
            }

            FuzzOp::CompleteSession { session_idx } => {
                if session_ids.is_empty() {
                    continue;
                }
                let sid = session_ids[usize::from(*session_idx) % session_ids.len()];
                if completed_sessions.contains(&sid) || failed_sessions.contains(&sid) {
                    continue;
                }
                manager.complete_session(sid);
                completed_sessions.insert(sid);
            }

            FuzzOp::FailSession { session_idx } => {
                if session_ids.is_empty() {
                    continue;
                }
                let sid = session_ids[usize::from(*session_idx) % session_ids.len()];
                if completed_sessions.contains(&sid) || failed_sessions.contains(&sid) {
                    continue;
                }
                manager.fail_session(sid, "fuzz-induced failure");
                failed_sessions.insert(sid);
            }

            FuzzOp::ActiveCount => {
                let count = manager.active_session_count();
                // Active sessions = total started - completed - failed (approximately)
                // Can't be negative
                assert!(
                    count <= session_ids.len(),
                    "Active {} > total created {}",
                    count,
                    session_ids.len()
                );
            }

            FuzzOp::GetPhase { session_idx } => {
                if session_ids.is_empty() {
                    continue;
                }
                let sid = session_ids[usize::from(*session_idx) % session_ids.len()];
                // Just verify it doesn't panic
                let _ = manager.session_phase(sid);
            }
        }
    }

    // Final invariants
    let stats = manager.stats_snapshot();
    assert!(
        stats.sessions_started >= stats.sessions_completed,
        "Completed {} > started {}",
        stats.sessions_completed,
        stats.sessions_started
    );

    // Verify all session IDs are unique
    let unique: std::collections::HashSet<u64> = session_ids.iter().copied().collect();
    assert_eq!(
        unique.len(),
        session_ids.len(),
        "Session IDs must be globally unique"
    );
});
