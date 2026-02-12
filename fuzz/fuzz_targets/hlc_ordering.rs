// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Fuzz the HLC under drift conditions to verify monotonicity.
//!
//! Tests that:
//! - Timestamps from now() are always monotonically increasing
//! - Timestamps from receive() are always after both local and remote
//! - Drift injection and clock jumps never violate monotonicity
//! - No panics from any combination of HLC operations

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::hlc::{HLCTimestamp, HybridLogicalClock};

#[derive(Debug, Arbitrary)]
enum HlcOp {
    Now { clock_idx: u8 },
    SetDrift { clock_idx: u8, drift: i16 },
    InjectJump { clock_idx: u8, jump: i16 },
    Receive { to_idx: u8, from_idx: u8 },
}

#[derive(Debug, Arbitrary)]
struct FuzzInput {
    node_count: u8,
    operations: Vec<HlcOp>,
}

fuzz_target!(|data: &[u8]| {
    let Ok(input) = FuzzInput::arbitrary(&mut arbitrary::Unstructured::new(data)) else {
        return;
    };
    let node_count = ((input.node_count % 4) + 2) as usize; // 2..=5
    if input.operations.len() > 200 {
        return;
    }

    let clocks: Vec<HybridLogicalClock> = (0..node_count)
        .filter_map(|i| HybridLogicalClock::new(i as u64).ok())
        .collect();

    // If any clock creation failed, bail
    if clocks.len() != node_count {
        return;
    }

    // Track last timestamp per clock for monotonicity check
    let mut last_ts: Vec<Option<HLCTimestamp>> = vec![None; node_count];

    for op in &input.operations {
        match op {
            HlcOp::Now { clock_idx } => {
                let idx = (*clock_idx as usize) % node_count;
                if let Ok(ts) = clocks[idx].now() {
                    if let Some(ref prev) = last_ts[idx] {
                        assert!(
                            ts > *prev,
                            "Monotonicity violation on now(): {:?} <= {:?}",
                            ts,
                            prev
                        );
                    }
                    last_ts[idx] = Some(ts);
                }
            }
            HlcOp::SetDrift { clock_idx, drift } => {
                let idx = (*clock_idx as usize) % node_count;
                // Scale drift to reasonable range (up to ~3.2 seconds)
                clocks[idx].set_drift_offset(i64::from(*drift) * 100);
            }
            HlcOp::InjectJump { clock_idx, jump } => {
                let idx = (*clock_idx as usize) % node_count;
                clocks[idx].inject_clock_jump(i64::from(*jump) * 100);
            }
            HlcOp::Receive { to_idx, from_idx } => {
                let to = (*to_idx as usize) % node_count;
                let from = (*from_idx as usize) % node_count;
                if to == from {
                    continue;
                }
                let Ok(from_ts) = clocks[from].now() else {
                    continue;
                };
                // Update from clock's last_ts
                if let Some(ref prev) = last_ts[from] {
                    assert!(
                        from_ts > *prev,
                        "Monotonicity violation on sender now(): {:?} <= {:?}",
                        from_ts,
                        prev
                    );
                }
                last_ts[from] = Some(from_ts);

                if let Ok(ts) = clocks[to].receive(&from_ts) {
                    if let Some(ref prev) = last_ts[to] {
                        assert!(
                            ts > *prev,
                            "Monotonicity violation on receive(): {:?} <= {:?}",
                            ts,
                            prev
                        );
                    }
                    assert!(
                        ts > from_ts,
                        "receive() must produce timestamp after input: {:?} <= {:?}",
                        ts,
                        from_ts
                    );
                    last_ts[to] = Some(ts);
                }
            }
        }
    }
});
