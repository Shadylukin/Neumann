// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for HLC clock drift behavior.
//!
//! Tests that the hybrid logical clock correctly handles
//! simulated clock drift, time jumps, and cross-node time skew.

use tensor_chain::hlc::HybridLogicalClock;

#[test]
fn test_clock_drift_positive_offset() {
    let hlc = HybridLogicalClock::new(1).unwrap();

    let ts1 = hlc.now().unwrap();

    // Inject positive drift (clock appears 5 seconds ahead)
    hlc.set_drift_offset(5000);

    let ts2 = hlc.now().unwrap();

    // Wall time should jump forward by approximately the drift amount
    assert!(
        ts2.wall_ms() >= ts1.wall_ms() + 4000,
        "Expected wall_ms to advance by ~5000ms, got delta={}",
        ts2.wall_ms() - ts1.wall_ms()
    );
}

#[test]
fn test_clock_drift_negative_offset_maintains_monotonicity() {
    let hlc = HybridLogicalClock::new(1).unwrap();

    // Generate several timestamps to establish a high-water mark
    let mut last = hlc.now().unwrap();
    for _ in 0..10 {
        last = hlc.now().unwrap();
    }

    // Inject negative drift (clock appears behind)
    hlc.set_drift_offset(-10000);

    // HLC monotonicity guarantee: timestamps never go backwards
    let after_drift = hlc.now().unwrap();
    assert!(
        after_drift > last,
        "HLC monotonicity violated: {:?} should be > {:?}",
        after_drift,
        last
    );
}

#[test]
fn test_clock_drift_between_two_nodes() {
    let hlc1 = HybridLogicalClock::new(1).unwrap();
    let hlc2 = HybridLogicalClock::new(2).unwrap();

    // Node 1 is 2 seconds ahead
    hlc1.set_drift_offset(2000);
    // Node 2 is 2 seconds behind
    hlc2.set_drift_offset(-2000);

    let ts1 = hlc1.now().unwrap();
    let ts2 = hlc2.now().unwrap();

    // Node 1 (ahead) should have higher wall time than node 2 (behind).
    // Node 2's negative drift may saturate to 0, so the gap is at least the
    // positive drift amount.
    assert!(
        ts1.wall_ms() > ts2.wall_ms(),
        "Node with positive drift should have higher wall_ms: {} vs {}",
        ts1.wall_ms(),
        ts2.wall_ms()
    );

    let diff = ts1.wall_ms() - ts2.wall_ms();
    assert!(
        diff >= 1500,
        "Expected significant wall_ms difference, got {}",
        diff
    );
}

#[test]
fn test_clock_drift_receive_corrects_skew() {
    let hlc1 = HybridLogicalClock::new(1).unwrap();
    let hlc2 = HybridLogicalClock::new(2).unwrap();

    // Node 1 is far ahead
    hlc1.set_drift_offset(10000);

    let ts_from_node1 = hlc1.now().unwrap();

    // Node 2 receives from node 1 - should advance past node 1's timestamp
    let ts_node2_after = hlc2.receive(&ts_from_node1).unwrap();

    assert!(
        ts_node2_after > ts_from_node1,
        "receive() should produce timestamp after the received one"
    );
}

#[test]
fn test_clock_jump_simulates_ntp_correction() {
    let hlc = HybridLogicalClock::new(1).unwrap();

    let before_jump = hlc.now().unwrap();

    // Simulate NTP correction: clock jumps forward 3 seconds
    hlc.inject_clock_jump(3000);

    let after_jump = hlc.now().unwrap();

    assert!(
        after_jump.wall_ms() >= before_jump.wall_ms() + 2500,
        "Expected wall_ms to advance by ~3000ms after clock jump"
    );
}

#[test]
fn test_clock_jump_backward_preserves_monotonicity() {
    let hlc = HybridLogicalClock::new(1).unwrap();

    // First advance the clock
    hlc.inject_clock_jump(5000);
    let high_ts = hlc.now().unwrap();

    // Now jump backward (NTP correction)
    hlc.inject_clock_jump(-8000);

    // Timestamps must still be monotonic
    let after_backward_jump = hlc.now().unwrap();
    assert!(
        after_backward_jump > high_ts,
        "Monotonicity violated after backward clock jump"
    );
}

#[test]
fn test_rapid_drift_changes() {
    let hlc = HybridLogicalClock::new(1).unwrap();

    let mut prev = hlc.now().unwrap();

    // Rapidly change drift and verify monotonicity holds throughout
    for drift in [5000i64, -3000, 10000, -8000, 0, 2000, -1000] {
        hlc.set_drift_offset(drift);

        for _ in 0..5 {
            let current = hlc.now().unwrap();
            assert!(
                current > prev,
                "Monotonicity violated with drift={}: {:?} <= {:?}",
                drift,
                current,
                prev
            );
            prev = current;
        }
    }
}

#[test]
fn test_multiple_nodes_with_varying_drift() {
    let clocks: Vec<_> = (0..5)
        .map(|i| HybridLogicalClock::new(i).unwrap())
        .collect();

    // Apply different drift to each node
    let drifts = [0i64, 1000, -1000, 3000, -2000];
    for (clock, &drift) in clocks.iter().zip(&drifts) {
        clock.set_drift_offset(drift);
    }

    // Generate timestamps from all nodes
    let timestamps: Vec<_> = clocks.iter().map(|c| c.now().unwrap()).collect();

    // All timestamps should be valid (non-zero wall_ms)
    for ts in &timestamps {
        assert!(ts.wall_ms() > 0);
    }

    // Simulate gossip: each node receives from all others
    for (i, clock) in clocks.iter().enumerate() {
        for (j, ts) in timestamps.iter().enumerate() {
            if i != j {
                let result = clock.receive(ts);
                assert!(result.is_ok());
            }
        }
    }
}

#[test]
fn test_drift_offset_getter() {
    let hlc = HybridLogicalClock::new(1).unwrap();

    assert_eq!(hlc.drift_offset(), 0);

    hlc.set_drift_offset(42);
    assert_eq!(hlc.drift_offset(), 42);

    hlc.set_drift_offset(-99);
    assert_eq!(hlc.drift_offset(), -99);
}

#[test]
fn test_clock_jump_accumulation() {
    let hlc = HybridLogicalClock::new(1).unwrap();

    hlc.inject_clock_jump(100);
    hlc.inject_clock_jump(200);
    hlc.inject_clock_jump(-50);

    assert_eq!(hlc.drift_offset(), 250);
}

#[test]
fn test_extreme_drift_no_panic() {
    let hlc = HybridLogicalClock::new(1).unwrap();

    // Very large positive drift
    hlc.set_drift_offset(i64::MAX / 2);
    let ts = hlc.now().unwrap();
    assert!(ts.wall_ms() > 0);

    // Very large negative drift (saturating_sub prevents underflow)
    hlc.set_drift_offset(i64::MIN / 2);
    let ts2 = hlc.now().unwrap();
    assert!(ts2 > ts); // Monotonicity still holds
}
