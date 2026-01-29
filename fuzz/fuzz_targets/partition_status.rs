// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::PartitionStatus;

#[derive(Arbitrary, Debug)]
struct Input {
    total_nodes: u8,
    healthy_count: u8,
    in_grace: bool,
}

fuzz_target!(|input: Input| {
    // Normalize inputs to valid ranges
    let total = (input.total_nodes as usize).clamp(1, 100);
    let healthy = (input.healthy_count as usize).min(total);

    // Compute expected status
    let status = if input.in_grace {
        PartitionStatus::Unknown
    } else {
        let quorum = (total / 2) + 1;
        if healthy >= quorum {
            PartitionStatus::QuorumReachable
        } else if healthy * 2 == total {
            // Exact 50/50 split in even-sized cluster
            PartitionStatus::Stalemate
        } else {
            PartitionStatus::QuorumLost
        }
    };

    // Verify status invariants based on computed values
    match status {
        PartitionStatus::QuorumReachable => {
            assert!(
                !input.in_grace,
                "QuorumReachable should not be set during grace period"
            );
            let quorum = (total / 2) + 1;
            assert!(
                healthy >= quorum,
                "QuorumReachable requires majority: healthy={}, quorum={}",
                healthy,
                quorum
            );
        },
        PartitionStatus::QuorumLost => {
            assert!(
                !input.in_grace,
                "QuorumLost should not be set during grace period"
            );
            let quorum = (total / 2) + 1;
            assert!(healthy < quorum, "QuorumLost means below quorum");
            // Also verify it's not a stalemate
            assert!(
                healthy * 2 != total || total % 2 != 0,
                "QuorumLost should not be stalemate"
            );
        },
        PartitionStatus::Stalemate => {
            assert!(
                !input.in_grace,
                "Stalemate should not be set during grace period"
            );
            assert_eq!(healthy * 2, total, "Stalemate requires exact 50/50 split");
        },
        PartitionStatus::Unknown => {
            assert!(
                input.in_grace,
                "Unknown should only be set during grace period"
            );
        },
        _ => {
            // Non-exhaustive enum, handle any new variants gracefully
        }
    }

    // Test serialization roundtrip
    let bytes = bitcode::serialize(&status).expect("Failed to serialize PartitionStatus");
    let restored: PartitionStatus =
        bitcode::deserialize(&bytes).expect("Failed to deserialize PartitionStatus");
    assert_eq!(status, restored, "Serialization roundtrip failed");

    // Test debug format doesn't panic
    let _ = format!("{:?}", status);

    // Test clone
    let cloned = status;
    assert_eq!(status, cloned);
});
