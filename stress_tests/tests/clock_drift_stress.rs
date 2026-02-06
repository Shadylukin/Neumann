// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Stress tests for sustained clock drift with concurrent operations.
//!
//! Tests HLC behavior under extreme and rapidly changing drift scenarios:
//! - Sustained drift with rapid offset changes
//! - Drift with cross-clock receive convergence
//! - Extreme drift recovery

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor_chain::hlc::HybridLogicalClock;

#[test]
#[ignore]
fn test_sustained_drift_monotonicity() {
    let clocks: Vec<HybridLogicalClock> = (0..5)
        .map(|i| HybridLogicalClock::new(i).expect("failed to create HLC"))
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // For each clock, track the last timestamp to verify monotonicity
    let mut last_timestamps: Vec<Option<tensor_chain::HLCTimestamp>> = vec![None; 5];

    for iteration in 0..10_000 {
        // Rapidly change drift offsets on all clocks
        for clock in &clocks {
            let drift = rng.random_range(-5000_i64..=5000);
            clock.set_drift_offset(drift);
        }

        // Generate a timestamp from each clock and verify monotonicity
        for (idx, clock) in clocks.iter().enumerate() {
            let ts = clock.now().expect("failed to generate timestamp");

            if let Some(prev) = &last_timestamps[idx] {
                assert!(
                    ts > *prev,
                    "Monotonicity violated on clock {} at iteration {}: {:?} <= {:?}",
                    idx,
                    iteration,
                    ts,
                    prev
                );
            }

            last_timestamps[idx] = Some(ts);
        }
    }

    // Verify all clocks produced timestamps
    for (idx, ts) in last_timestamps.iter().enumerate() {
        assert!(ts.is_some(), "Clock {} never produced a timestamp", idx);
    }

    println!("All 5 clocks maintained monotonicity across 10000 iterations with random drift");
}

#[test]
#[ignore]
fn test_drift_with_receive_convergence() {
    let clocks: Vec<HybridLogicalClock> = (0..3)
        .map(|i| HybridLogicalClock::new(i).expect("failed to create HLC"))
        .collect();

    // Set different drift offsets
    clocks[0].set_drift_offset(3000);
    clocks[1].set_drift_offset(0);
    clocks[2].set_drift_offset(-3000);

    let mut rng = ChaCha8Rng::seed_from_u64(123);

    // Track per-clock timestamps for monotonicity verification
    let mut last_timestamps: Vec<Option<tensor_chain::HLCTimestamp>> = vec![None; 3];

    for round in 0..5000 {
        // Pick a random clock to generate a timestamp
        let sender_idx = rng.random_range(0..3_usize);
        let ts = clocks[sender_idx]
            .now()
            .expect("failed to generate timestamp");

        // Verify monotonicity for sender
        if let Some(prev) = &last_timestamps[sender_idx] {
            assert!(
                ts > *prev,
                "Monotonicity violated on sender clock {} at round {}: {:?} <= {:?}",
                sender_idx,
                round,
                ts,
                prev
            );
        }
        last_timestamps[sender_idx] = Some(ts);

        // Pick a different random clock to receive
        let mut receiver_idx = rng.random_range(0..3_usize);
        while receiver_idx == sender_idx {
            receiver_idx = rng.random_range(0..3_usize);
        }

        let received_ts = clocks[receiver_idx]
            .receive(&ts)
            .expect("failed to receive timestamp");

        // Verify monotonicity for receiver
        if let Some(prev) = &last_timestamps[receiver_idx] {
            assert!(
                received_ts > *prev,
                "Monotonicity violated on receiver clock {} at round {}: {:?} <= {:?}",
                receiver_idx,
                round,
                received_ts,
                prev
            );
        }
        last_timestamps[receiver_idx] = Some(received_ts);
    }

    // Generate final timestamps from all clocks and check convergence
    let final_timestamps: Vec<tensor_chain::HLCTimestamp> = clocks
        .iter()
        .map(|c| c.now().expect("failed to generate final timestamp"))
        .collect();

    let wall_times: Vec<u64> = final_timestamps.iter().map(|ts| ts.wall_ms()).collect();
    let max_wall = wall_times.iter().copied().max().unwrap();
    let min_wall = wall_times.iter().copied().min().unwrap();
    let spread = max_wall - min_wall;

    println!("Final wall_ms values: {:?}", wall_times);
    println!("Wall time spread: {} ms", spread);

    assert!(
        spread <= 1000,
        "Clock wall times did not converge: spread {} ms exceeds 1000 ms threshold. \
         Wall times: {:?}",
        spread,
        wall_times
    );
}

#[test]
#[ignore]
fn test_extreme_drift_recovery() {
    let clock = HybridLogicalClock::new(1).expect("failed to create HLC");

    // Generate a baseline timestamp
    let baseline = clock.now().expect("failed to generate baseline");

    // Set extreme drift
    let extreme_drift = i64::MAX / 4;
    clock.set_drift_offset(extreme_drift);

    // Generate timestamps under extreme drift
    let mut prev = clock
        .now()
        .expect("failed to generate timestamp under extreme drift");
    assert!(
        prev > baseline,
        "Timestamp under extreme drift should be after baseline"
    );

    for i in 0..100 {
        let ts = clock
            .now()
            .expect("failed to generate timestamp under extreme drift");
        assert!(
            ts > prev,
            "Monotonicity violated under extreme drift at iteration {}: {:?} <= {:?}",
            i,
            ts,
            prev
        );
        prev = ts;
    }

    // Reset drift to zero
    clock.set_drift_offset(0);

    // Generate timestamps after recovery and verify monotonicity
    let mut prev_after_reset = clock
        .now()
        .expect("failed to generate timestamp after drift reset");

    // The timestamp after reset should still be monotonically increasing from
    // the last timestamp generated under extreme drift, because HLC maintains
    // the high-water mark.
    assert!(
        prev_after_reset > prev,
        "Timestamp after drift reset should be after last extreme-drift timestamp: \
         {:?} <= {:?}",
        prev_after_reset,
        prev
    );

    for i in 0..1000 {
        let ts = clock
            .now()
            .expect("failed to generate timestamp after recovery");
        assert!(
            ts > prev_after_reset,
            "Monotonicity violated after drift recovery at iteration {}: {:?} <= {:?}",
            i,
            ts,
            prev_after_reset
        );
        prev_after_reset = ts;
    }

    println!(
        "HLC successfully recovered from extreme drift ({}) and maintained monotonicity",
        extreme_drift
    );
}
