// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Deterministic simulation tests for Raft consensus.
//!
//! Each test runs a DST scenario with controlled fault injection and
//! verifies Raft invariants (election safety, term monotonicity).
//! Tests are parameterized by seed for reproducibility.

use integration_tests::dst::{DSTHarness, FaultAction};

/// Run a scenario across multiple seeds for confidence.
fn run_scenario(scenario: fn(&mut DSTHarness), node_count: usize, max_ticks: u64, seed_count: u64) {
    for seed in 0..seed_count {
        let mut harness = DSTHarness::new(seed, node_count, max_ticks);
        scenario(&mut harness);
        let result = harness.run();
        assert!(
            result.election_safety_held,
            "Election safety violated with seed {seed}: {result:?}"
        );
        assert!(
            result.linearizability_held,
            "Linearizability violated with seed {seed}: {result:?}"
        );
    }
}

fn scenario_no_faults(harness: &mut DSTHarness) {
    // node0 triggers election at tick 5
    harness.schedule_election(5, 0);
    // Heartbeats every 10 ticks to maintain leadership
    for tick in (15..100).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }
    // Propose entries once leader established
    for tick in (20..90).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_no_faults_5_nodes() {
    run_scenario(scenario_no_faults, 5, 100, 1_000);
}

#[test]
fn dst_raft_no_faults_3_nodes() {
    run_scenario(scenario_no_faults, 3, 100, 1_000);
}

fn scenario_follower_crash(harness: &mut DSTHarness) {
    // node0 wins election
    harness.schedule_election(5, 0);
    for tick in (15..200).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Crash node4 at tick 50 (if 5 nodes, still have quorum)
    let crash_idx = harness.node_count() - 1;
    harness.schedule_fault(50, FaultAction::PartitionNode(crash_idx));

    // Continue proposing - should still work with majority
    for tick in (60..180).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_follower_crash() {
    run_scenario(scenario_follower_crash, 5, 200, 1_000);
}

fn scenario_leader_crash_new_election(harness: &mut DSTHarness) {
    // node0 wins initial election
    harness.schedule_election(5, 0);
    for tick in (15..50).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Crash leader at tick 50
    harness.schedule_fault(50, FaultAction::PartitionNode(0));

    // node1 starts election at tick 60 (election timeout)
    harness.schedule_election(60, 1);
    for tick in (70..200).step_by(10) {
        harness.schedule_heartbeat(tick, 1);
    }

    // Proposals to new leader
    for tick in (80..180).step_by(5) {
        harness.schedule_proposal(tick, 1);
    }
}

#[test]
fn dst_raft_leader_crash() {
    run_scenario(scenario_leader_crash_new_election, 5, 200, 1_000);
}

fn scenario_partition_heal(harness: &mut DSTHarness) {
    // node0 wins election
    harness.schedule_election(5, 0);
    for tick in (15..40).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Partition node0 at tick 40
    harness.schedule_fault(40, FaultAction::PartitionNode(0));

    // node1 starts new election at tick 55
    harness.schedule_election(55, 1);
    for tick in (65..100).step_by(10) {
        harness.schedule_heartbeat(tick, 1);
    }

    // Heal partition at tick 100
    harness.schedule_fault(100, FaultAction::HealNode(0));

    // Continue with heartbeats from new leader
    for tick in (110..200).step_by(10) {
        harness.schedule_heartbeat(tick, 1);
    }
}

#[test]
fn dst_raft_partition_heal() {
    run_scenario(scenario_partition_heal, 5, 200, 1_000);
}

fn scenario_competing_elections(harness: &mut DSTHarness) {
    // Two nodes start elections simultaneously
    harness.schedule_election(5, 0);
    harness.schedule_election(5, 1);

    // Only one should win - heartbeat from whoever wins
    for tick in (15..200).step_by(10) {
        // Schedule heartbeats from both; only the actual leader's will be effective
        harness.schedule_heartbeat(tick, 0);
        harness.schedule_heartbeat(tick, 1);
    }
}

#[test]
fn dst_raft_competing_elections() {
    run_scenario(scenario_competing_elections, 5, 200, 1_000);
}

fn scenario_cascading_crashes(harness: &mut DSTHarness) {
    // node0 wins election
    harness.schedule_election(5, 0);
    for tick in (15..30).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Crash node0
    harness.schedule_fault(30, FaultAction::PartitionNode(0));

    // node1 takes over
    harness.schedule_election(40, 1);
    for tick in (50..70).step_by(10) {
        harness.schedule_heartbeat(tick, 1);
    }

    // Crash node1 too
    harness.schedule_fault(70, FaultAction::PartitionNode(1));

    // node2 takes over - still have 3 of 5 active
    harness.schedule_election(80, 2);
    for tick in (90..200).step_by(10) {
        harness.schedule_heartbeat(tick, 2);
    }
}

#[test]
fn dst_raft_cascading_crashes() {
    run_scenario(scenario_cascading_crashes, 5, 200, 1_000);
}

fn scenario_majority_partition(harness: &mut DSTHarness) {
    // node0 wins election
    harness.schedule_election(5, 0);
    for tick in (15..40).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Partition nodes 3 and 4 (minority)
    harness.schedule_fault(40, FaultAction::PartitionNode(3));
    harness.schedule_fault(40, FaultAction::PartitionNode(4));

    // Majority (0, 1, 2) continues
    for tick in (50..200).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }
    for tick in (55..180).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_majority_partition() {
    run_scenario(scenario_majority_partition, 5, 200, 1_000);
}

// ---------------------------------------------------------------------------
// New scenarios
// ---------------------------------------------------------------------------

/// Rolling restart: sequential crash+heal of each follower while leader stays up.
fn scenario_rolling_restart(harness: &mut DSTHarness) {
    harness.schedule_election(5, 0);
    for tick in (15..300).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }
    // Crash and heal each follower in turn
    for i in 1..harness.node_count() {
        #[allow(clippy::cast_possible_truncation)]
        let crash = 50 + (i as u64 * 40);
        harness.schedule_fault(crash, FaultAction::PartitionNode(i));
        harness.schedule_fault(crash + 20, FaultAction::HealNode(i));
    }
    // Propose throughout
    for tick in (60..280).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_rolling_restart() {
    run_scenario(scenario_rolling_restart, 5, 300, 1_000);
}

/// Rapid leader churn: multiple forced elections in quick succession.
fn scenario_rapid_leader_churn(harness: &mut DSTHarness) {
    let n = harness.node_count();
    for (i, tick) in [5u64, 30, 55, 80, 105].iter().enumerate() {
        let node = i % n;
        harness.schedule_election(*tick, node);
        for hb in (*tick + 5..*tick + 25).step_by(5) {
            harness.schedule_heartbeat(hb, node);
        }
    }
    for tick in (20..150).step_by(7) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_rapid_leader_churn() {
    run_scenario(scenario_rapid_leader_churn, 5, 200, 1_000);
}

/// Asymmetric partition: node can send but not receive (half-open).
fn scenario_asymmetric_partition(harness: &mut DSTHarness) {
    harness.schedule_election(5, 0);
    for tick in (15..40).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Node 2 can send but cannot receive (inbound blocked, outbound open).
    // It will miss heartbeats and may start spurious elections, but its votes
    // can still reach other nodes.
    harness.schedule_fault(
        40,
        FaultAction::AsymmetricPartition {
            node: 2,
            inbound_blocked: true,
            outbound_blocked: false,
        },
    );

    // Leader continues heartbeats
    for tick in (50..150).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Heal at tick 150
    harness.schedule_fault(150, FaultAction::HealNode(2));

    for tick in (160..250).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }
    for tick in (60..240).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_asymmetric_partition() {
    run_scenario(scenario_asymmetric_partition, 5, 250, 1_000);
}

/// Outbound-only block: node can receive but not send.
fn scenario_outbound_block(harness: &mut DSTHarness) {
    harness.schedule_election(5, 0);
    for tick in (15..40).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Node 1 can receive but cannot send (outbound blocked).
    // It will receive heartbeats (staying follower) but its vote responses
    // won't reach candidates.
    harness.schedule_fault(
        40,
        FaultAction::AsymmetricPartition {
            node: 1,
            inbound_blocked: false,
            outbound_blocked: true,
        },
    );

    for tick in (50..150).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Heal
    harness.schedule_fault(150, FaultAction::HealNode(1));

    for tick in (160..250).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }
    for tick in (60..240).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_outbound_block() {
    run_scenario(scenario_outbound_block, 5, 250, 1_000);
}

// ---------------------------------------------------------------------------
// Network non-determinism scenarios
// ---------------------------------------------------------------------------

/// Message reordering from tick 0: election + heartbeats + proposals under
/// non-FIFO delivery. Safety properties must hold despite reordering.
fn scenario_message_reordering(harness: &mut DSTHarness) {
    harness.schedule_fault(0, FaultAction::MessageReorder { enabled: true });

    harness.schedule_election(5, 0);
    for tick in (15..200).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }
    for tick in (20..180).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_message_reordering() {
    run_scenario(scenario_message_reordering, 5, 200, 1_000);
}

/// Latency spike on one node: node 2 gets +10 tick delivery latency at tick 50,
/// healed at tick 150. The slow node may miss elections but safety must hold.
fn scenario_latency_spike(harness: &mut DSTHarness) {
    harness.schedule_election(5, 0);
    for tick in (15..300).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Node 2 gets +10 tick latency at tick 50
    harness.schedule_fault(
        50,
        FaultAction::LatencySpike {
            node: 2,
            extra_ticks: 10,
        },
    );
    // Heal at tick 150
    harness.schedule_fault(150, FaultAction::HealNode(2));

    for tick in (60..280).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_latency_spike() {
    run_scenario(scenario_latency_spike, 5, 300, 1_000);
}

/// 10% message drops from tick 30. Multiple election attempts may be needed.
/// Some proposals may fail under drops, which is correct Raft behavior.
fn scenario_message_drops(harness: &mut DSTHarness) {
    harness.schedule_election(5, 0);
    for tick in (15..300).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Enable 10% drops at tick 30
    harness.schedule_fault(30, FaultAction::MessageDrop { rate: 0.1 });

    // Schedule backup elections in case leader loses quorum
    harness.schedule_election(100, 1);
    harness.schedule_election(200, 2);
    for tick in (110..300).step_by(10) {
        harness.schedule_heartbeat(tick, 1);
        harness.schedule_heartbeat(tick, 2);
    }

    // Proposals on all potential leaders
    for tick in (40..280).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_message_drops() {
    run_scenario(scenario_message_drops, 5, 300, 1_000);
}

/// Combined network faults: reorder + 5% drop + latency spike + partition/heal.
fn scenario_combined_network_faults(harness: &mut DSTHarness) {
    // Enable reordering from the start
    harness.schedule_fault(0, FaultAction::MessageReorder { enabled: true });

    harness.schedule_election(5, 0);
    for tick in (15..400).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Enable 5% drops at tick 20
    harness.schedule_fault(20, FaultAction::MessageDrop { rate: 0.05 });

    // Latency spike on node 3 at tick 60
    harness.schedule_fault(
        60,
        FaultAction::LatencySpike {
            node: 3,
            extra_ticks: 8,
        },
    );

    // Partition node 4 at tick 100, heal at 200
    harness.schedule_fault(100, FaultAction::PartitionNode(4));
    harness.schedule_fault(200, FaultAction::HealNode(4));

    // Heal latency at tick 180
    harness.schedule_fault(
        180,
        FaultAction::LatencySpike {
            node: 3,
            extra_ticks: 0,
        },
    );

    // Disable drops at tick 250
    harness.schedule_fault(250, FaultAction::MessageDrop { rate: 0.0 });

    // Backup elections
    harness.schedule_election(150, 1);
    harness.schedule_election(250, 2);
    for tick in (160..400).step_by(10) {
        harness.schedule_heartbeat(tick, 1);
        harness.schedule_heartbeat(tick, 2);
    }

    // Proposals throughout
    for tick in (30..380).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_combined_network_faults() {
    run_scenario(scenario_combined_network_faults, 5, 400, 1_000);
}

// ---------------------------------------------------------------------------
// Heartbeat log consistency
// ---------------------------------------------------------------------------

/// After proposals build a log, partition+heal a follower. Heartbeats after
/// heal must carry correct prev_log_index/prev_log_term so the follower with
/// a non-empty log accepts them. Would fail with old hardcoded 0,0 values.
fn scenario_heartbeat_log_consistency(harness: &mut DSTHarness) {
    // node0 wins election
    harness.schedule_election(5, 0);
    for tick in (15..50).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // Propose several entries to build up the log
    for tick in (20..50).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }

    // Partition node 2 at tick 50
    harness.schedule_fault(50, FaultAction::PartitionNode(2));

    // Continue heartbeats and proposals while node 2 is partitioned
    for tick in (55..100).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }
    for tick in (55..100).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }

    // Heal node 2 at tick 100
    harness.schedule_fault(100, FaultAction::HealNode(2));

    // Heartbeats after heal must have correct prev_log_index/prev_log_term
    // for node 2 (which has a non-empty log from before partition) to accept
    for tick in (105..200).step_by(10) {
        harness.schedule_heartbeat(tick, 0);
    }

    // More proposals to verify the healed follower participates in quorum
    for tick in (110..190).step_by(5) {
        harness.schedule_proposal(tick, 0);
    }
}

#[test]
fn dst_raft_heartbeat_log_consistency() {
    run_scenario(scenario_heartbeat_log_consistency, 5, 200, 1_000);
}

// ---------------------------------------------------------------------------
// Stress tests (10,000 seeds, run with --ignored)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn dst_stress_no_faults_10k_seeds() {
    run_scenario(scenario_no_faults, 5, 100, 10_000);
}

#[test]
#[ignore]
fn dst_stress_leader_crash_10k_seeds() {
    run_scenario(scenario_leader_crash_new_election, 5, 200, 10_000);
}

#[test]
#[ignore]
fn dst_stress_partition_heal_10k_seeds() {
    run_scenario(scenario_partition_heal, 5, 200, 10_000);
}

#[test]
#[ignore]
fn dst_stress_rolling_restart_10k_seeds() {
    run_scenario(scenario_rolling_restart, 5, 300, 10_000);
}

#[test]
#[ignore]
fn dst_stress_rapid_churn_10k_seeds() {
    run_scenario(scenario_rapid_leader_churn, 5, 200, 10_000);
}

#[test]
#[ignore]
fn dst_stress_message_reordering_10k_seeds() {
    run_scenario(scenario_message_reordering, 5, 200, 10_000);
}

#[test]
#[ignore]
fn dst_stress_combined_faults_10k_seeds() {
    run_scenario(scenario_combined_network_faults, 5, 400, 10_000);
}
