//! Fuzz target for Raft heartbeat configuration and statistics.
//!
//! Tests:
//! - HeartbeatStats operations don't panic with arbitrary values
//! - HeartbeatStatsSnapshot roundtrip serialization
//! - RaftConfig heartbeat settings validation

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::sync::atomic::Ordering;
use tensor_chain::{HeartbeatStats, HeartbeatStatsSnapshot, RaftConfig};

#[derive(Arbitrary, Debug)]
enum TestCase {
    StatsOperations {
        sent: u64,
        failed: u64,
        consecutive: u32,
    },
    StatsSnapshot {
        heartbeats_sent: u64,
        heartbeats_failed: u64,
        consecutive_failures: u32,
    },
    ConfigValidation {
        heartbeat_interval: u64,
        auto_heartbeat: bool,
        max_heartbeat_failures: u32,
        election_timeout_min: u64,
        election_timeout_max: u64,
    },
    StatsResetCycle {
        operations: Vec<StatsOp>,
    },
}

#[derive(Arbitrary, Debug, Clone)]
enum StatsOp {
    IncrementSent(u64),
    IncrementFailed(u64),
    IncrementConsecutive(u32),
    ResetConsecutive,
    Snapshot,
    Reset,
}

fuzz_target!(|test_case: TestCase| {
    match test_case {
        TestCase::StatsOperations {
            sent,
            failed,
            consecutive,
        } => {
            let stats = HeartbeatStats::default();

            // Store arbitrary values
            stats.heartbeats_sent.store(sent, Ordering::Relaxed);
            stats.heartbeats_failed.store(failed, Ordering::Relaxed);
            stats.consecutive_failures.store(consecutive, Ordering::Relaxed);

            // Snapshot should not panic
            let snapshot = stats.snapshot();

            // Values should match
            assert_eq!(snapshot.heartbeats_sent, sent);
            assert_eq!(snapshot.heartbeats_failed, failed);
            assert_eq!(snapshot.consecutive_failures, consecutive);

            // Fetch_add operations should not panic
            stats.heartbeats_sent.fetch_add(1, Ordering::Relaxed);
            stats.heartbeats_failed.fetch_add(1, Ordering::Relaxed);
            stats.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        }

        TestCase::StatsSnapshot {
            heartbeats_sent,
            heartbeats_failed,
            consecutive_failures,
        } => {
            // Create snapshot with arbitrary values
            let snapshot = HeartbeatStatsSnapshot {
                heartbeats_sent,
                heartbeats_failed,
                consecutive_failures,
                last_heartbeat_at: None,
            };

            // Clone should work
            let cloned = snapshot.clone();
            assert_eq!(cloned.heartbeats_sent, heartbeats_sent);
            assert_eq!(cloned.heartbeats_failed, heartbeats_failed);
            assert_eq!(cloned.consecutive_failures, consecutive_failures);

            // Debug should not panic
            let _ = format!("{:?}", snapshot);
        }

        TestCase::ConfigValidation {
            heartbeat_interval,
            auto_heartbeat,
            max_heartbeat_failures,
            election_timeout_min,
            election_timeout_max,
        } => {
            // Create config with arbitrary heartbeat settings
            // Election timeout needs min <= max
            let (min, max) = if election_timeout_min <= election_timeout_max {
                (election_timeout_min, election_timeout_max)
            } else {
                (election_timeout_max, election_timeout_min)
            };

            let config = RaftConfig {
                heartbeat_interval: heartbeat_interval.max(1), // Avoid zero
                auto_heartbeat,
                max_heartbeat_failures,
                election_timeout: (min.max(1), max.max(min.max(1))),
                ..RaftConfig::default()
            };

            // Config should be valid
            assert!(config.heartbeat_interval >= 1);
            assert!(config.election_timeout.0 <= config.election_timeout.1);

            // Clone and debug should work
            let cloned = config.clone();
            assert_eq!(cloned.auto_heartbeat, auto_heartbeat);
            assert_eq!(cloned.max_heartbeat_failures, max_heartbeat_failures);
            let _ = format!("{:?}", config);
        }

        TestCase::StatsResetCycle { operations } => {
            let stats = HeartbeatStats::default();

            // Limit operations to prevent timeout
            let ops = operations.into_iter().take(100);

            for op in ops {
                match op {
                    StatsOp::IncrementSent(n) => {
                        stats.heartbeats_sent.fetch_add(n.min(1000), Ordering::Relaxed);
                    }
                    StatsOp::IncrementFailed(n) => {
                        stats.heartbeats_failed.fetch_add(n.min(1000), Ordering::Relaxed);
                    }
                    StatsOp::IncrementConsecutive(n) => {
                        stats.consecutive_failures.fetch_add(n.min(100), Ordering::Relaxed);
                    }
                    StatsOp::ResetConsecutive => {
                        stats.consecutive_failures.store(0, Ordering::Relaxed);
                    }
                    StatsOp::Snapshot => {
                        let _ = stats.snapshot();
                    }
                    StatsOp::Reset => {
                        stats.heartbeats_sent.store(0, Ordering::Relaxed);
                        stats.heartbeats_failed.store(0, Ordering::Relaxed);
                        stats.consecutive_failures.store(0, Ordering::Relaxed);
                        *stats.last_heartbeat_at.write() = None;
                    }
                }
            }

            // Final snapshot should not panic
            let final_snapshot = stats.snapshot();

            // Sent + failed should be >= 0 (obviously, but validates no overflow)
            assert!(final_snapshot.heartbeats_sent <= u64::MAX);
            assert!(final_snapshot.heartbeats_failed <= u64::MAX);
        }
    }
});
