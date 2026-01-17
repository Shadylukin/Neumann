#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::ChainMetricsSnapshot;

#[derive(Arbitrary, Debug)]
enum TestCase {
    DeserializeArbitrary { bytes: Vec<u8> },
    RoundtripValues { snapshot: SnapshotInput },
    ComputeRates { snapshot: SnapshotInput },
}

#[derive(Arbitrary, Debug)]
struct SnapshotInput {
    // Raft stats
    fast_path_accepted: u64,
    fast_path_rejected: u64,
    heartbeat_successes: u64,
    heartbeat_failures: u64,
    quorum_checks: u64,
    quorum_lost_events: u64,
    leader_step_downs: u64,

    // DTX stats
    dtx_started: u64,
    dtx_committed: u64,
    dtx_aborted: u64,

    // Membership stats
    health_checks: u64,
    health_check_failures: u64,
    partition_events: u64,

    // Replication stats
    bytes_sent: u64,
    updates_sent: u64,
}

fuzz_target!(|test_case: TestCase| {
    match test_case {
        TestCase::DeserializeArbitrary { bytes } => {
            // Test that arbitrary bytes don't panic during deserialization
            // (may or may not succeed, but shouldn't crash)
            if let Ok(snapshot) = bincode::deserialize::<ChainMetricsSnapshot>(&bytes) {
                // If successful, verify roundtrip works
                if let Ok(roundtrip_bytes) = bincode::serialize(&snapshot) {
                    let restored: Result<ChainMetricsSnapshot, _> =
                        bincode::deserialize(&roundtrip_bytes);
                    assert!(
                        restored.is_ok(),
                        "Roundtrip should succeed for valid snapshot"
                    );
                }

                // Verify computed rates don't panic or return NaN
                let heartbeat_rate = snapshot.heartbeat_success_rate();
                assert!(!heartbeat_rate.is_nan(), "Heartbeat rate should not be NaN");
                assert!(
                    (0.0..=1.0).contains(&heartbeat_rate),
                    "Heartbeat rate should be 0-1"
                );

                let tx_rate = snapshot.tx_commit_rate();
                assert!(!tx_rate.is_nan(), "Tx commit rate should not be NaN");
                assert!(
                    (0.0..=1.0).contains(&tx_rate),
                    "Tx commit rate should be 0-1"
                );

                let health_rate = snapshot.health_check_success_rate();
                assert!(!health_rate.is_nan(), "Health check rate should not be NaN");
                assert!(
                    (0.0..=1.0).contains(&health_rate),
                    "Health check rate should be 0-1"
                );
            }
        },

        TestCase::RoundtripValues { snapshot: input } => {
            // Create a valid snapshot from fuzzed input
            let snapshot = create_snapshot_from_input(&input);

            // Serialize and deserialize
            let bytes = bincode::serialize(&snapshot).expect("Serialization should succeed");
            let restored: ChainMetricsSnapshot =
                bincode::deserialize(&bytes).expect("Deserialization should succeed");

            // Verify values match
            assert_eq!(
                snapshot.raft.fast_path_accepted,
                restored.raft.fast_path_accepted
            );
            assert_eq!(
                snapshot.raft.heartbeat_successes,
                restored.raft.heartbeat_successes
            );
            assert_eq!(
                snapshot.raft.heartbeat_failures,
                restored.raft.heartbeat_failures
            );
            assert_eq!(snapshot.dtx.started, restored.dtx.started);
            assert_eq!(snapshot.dtx.committed, restored.dtx.committed);
            assert_eq!(
                snapshot.membership.health_checks,
                restored.membership.health_checks
            );
        },

        TestCase::ComputeRates { snapshot: input } => {
            let snapshot = create_snapshot_from_input(&input);

            // Test rate calculations
            let heartbeat_rate = snapshot.heartbeat_success_rate();
            let total_heartbeats = snapshot.total_heartbeats();

            if total_heartbeats == 0 {
                // No heartbeats means 100% success (no failures)
                assert_eq!(heartbeat_rate, 1.0);
            } else {
                // Rate should be successes / total
                let expected = snapshot.raft.heartbeat_successes as f64 / total_heartbeats as f64;
                assert!(
                    (heartbeat_rate - expected).abs() < 0.0001,
                    "Heartbeat rate mismatch"
                );
            }

            let tx_rate = snapshot.tx_commit_rate();
            if snapshot.dtx.started == 0 {
                assert_eq!(tx_rate, 1.0);
            } else {
                let expected = snapshot.dtx.committed as f64 / snapshot.dtx.started as f64;
                assert!((tx_rate - expected).abs() < 0.0001, "Tx rate mismatch");
            }

            // Test is_cluster_healthy
            let healthy = snapshot.is_cluster_healthy();

            // If quorum was lost, cluster should be unhealthy
            if snapshot.raft.quorum_lost_events > 0 {
                assert!(!healthy, "Cluster should be unhealthy if quorum was lost");
            }

            // Test is_empty
            let empty = snapshot.is_empty();
            if snapshot.raft.fast_path_accepted == 0
                && snapshot.raft.heartbeat_successes == 0
                && snapshot.dtx.started == 0
                && snapshot.membership.health_checks == 0
                && snapshot.replication.updates_sent == 0
            {
                assert!(empty, "Should be empty when no operations recorded");
            }
        },
    }
});

fn create_snapshot_from_input(input: &SnapshotInput) -> ChainMetricsSnapshot {
    use tensor_chain::{
        DistributedTxStatsSnapshot, MembershipStatsSnapshot, RaftStatsSnapshot,
        ReplicationStatsSnapshot, TimingSnapshot,
    };

    let timing = TimingSnapshot::default();

    // Calculate rates
    let total_fast_path = input.fast_path_accepted + input.fast_path_rejected;
    let fast_path_rate = if total_fast_path > 0 {
        input.fast_path_accepted as f32 / total_fast_path as f32
    } else {
        0.0
    };

    let total_heartbeats = input.heartbeat_successes + input.heartbeat_failures;
    let heartbeat_success_rate = if total_heartbeats > 0 {
        input.heartbeat_successes as f32 / total_heartbeats as f32
    } else {
        1.0
    };

    let commit_rate = if input.dtx_started > 0 {
        input.dtx_committed as f32 / input.dtx_started as f32
    } else {
        1.0
    };

    let conflict_rate = if input.dtx_started > 0 {
        0.0 // No conflicts in fuzz input
    } else {
        0.0
    };

    ChainMetricsSnapshot {
        raft: RaftStatsSnapshot {
            fast_path_accepted: input.fast_path_accepted,
            fast_path_rejected: input.fast_path_rejected,
            full_validation_required: 0,
            election_timing: timing.clone(),
            heartbeat_timing: timing.clone(),
            quorum_checks: input.quorum_checks,
            quorum_lost_events: input.quorum_lost_events,
            leader_step_downs: input.leader_step_downs,
            heartbeat_successes: input.heartbeat_successes,
            heartbeat_failures: input.heartbeat_failures,
            fast_path_rate,
            heartbeat_success_rate,
        },
        dtx: DistributedTxStatsSnapshot {
            started: input.dtx_started,
            committed: input.dtx_committed,
            aborted: input.dtx_aborted,
            timed_out: 0,
            conflicts: 0,
            orthogonal_merges: 0,
            prepare_timing: timing.clone(),
            commit_timing: timing.clone(),
            lock_wait_timing: timing.clone(),
            participation_timeouts: 0,
            commit_rate,
            conflict_rate,
        },
        membership: MembershipStatsSnapshot {
            health_checks: input.health_checks,
            health_check_failures: input.health_check_failures,
            ping_timing: timing,
            partition_events: input.partition_events,
            quorum_lost_events: 0,
        },
        replication: ReplicationStatsSnapshot {
            bytes_sent: input.bytes_sent,
            bytes_saved: 0,
            updates_sent: input.updates_sent,
            batches_sent: 0,
            avg_compression_ratio: 0.0,
            full_updates: 0,
            queue_depth: 0,
            backpressure_events: 0,
            auto_drains: 0,
            peak_queue_depth: 0,
        },
    }
}
