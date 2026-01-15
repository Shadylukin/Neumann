#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::tcp::{PeerRateLimiter, RateLimitConfig};

#[derive(Arbitrary, Debug)]
struct RateLimitInput {
    bucket_size: u8,
    refill_rate: u8,
    enabled: bool,
    operations: Vec<RateLimitOp>,
}

#[derive(Arbitrary, Debug, Clone)]
enum RateLimitOp {
    Check { peer_id: u8 },
    AvailableTokens { peer_id: u8 },
    Remove { peer_id: u8 },
    Clear,
    PeerCount,
    IsEnabled,
}

fuzz_target!(|input: RateLimitInput| {
    // Limit operations to prevent timeout
    let operations: Vec<_> = input.operations.into_iter().take(100).collect();

    // Build config with reasonable bounds
    let bucket_size = input.bucket_size.max(1) as u32;
    let refill_rate = input.refill_rate as f64; // 0-255 tokens/sec

    let config = RateLimitConfig::default()
        .with_bucket_size(bucket_size)
        .with_refill_rate(refill_rate)
        .with_enabled(input.enabled);

    let limiter = PeerRateLimiter::new(config);

    // Execute operations
    for op in operations {
        let peer = format!("peer-{}", op.peer_id());
        match op {
            RateLimitOp::Check { .. } => {
                let result = limiter.check(&peer);
                if !input.enabled {
                    assert!(result, "Disabled limiter should always allow");
                }
            },
            RateLimitOp::AvailableTokens { .. } => {
                let tokens = limiter.available_tokens(&peer);
                if !input.enabled {
                    assert_eq!(tokens, u32::MAX, "Disabled limiter should return MAX");
                }
            },
            RateLimitOp::Remove { .. } => {
                limiter.remove_peer(&peer);
            },
            RateLimitOp::Clear => {
                limiter.clear();
                assert_eq!(limiter.peer_count(), 0, "Clear should remove all peers");
            },
            RateLimitOp::PeerCount => {
                let _ = limiter.peer_count();
            },
            RateLimitOp::IsEnabled => {
                let enabled = limiter.is_enabled();
                assert_eq!(enabled, input.enabled);
            },
        }
    }

    // Test that bucket exhaustion works correctly
    if input.enabled && bucket_size > 0 {
        let test_peer = "exhaustion-test".to_string();
        limiter.remove_peer(&test_peer); // Start fresh

        // Check bucket_size times - all should succeed
        for _ in 0..bucket_size {
            assert!(limiter.check(&test_peer));
        }

        // With refill_rate == 0, next check should fail
        // (unless refill_rate is high enough that some time passed)
        if refill_rate == 0.0 {
            assert!(
                !limiter.check(&test_peer),
                "Should be rate limited after exhausting bucket"
            );
        }
    }
});

impl RateLimitOp {
    fn peer_id(&self) -> u8 {
        match self {
            RateLimitOp::Check { peer_id } => *peer_id,
            RateLimitOp::AvailableTokens { peer_id } => *peer_id,
            RateLimitOp::Remove { peer_id } => *peer_id,
            RateLimitOp::Clear => 0,
            RateLimitOp::PeerCount => 0,
            RateLimitOp::IsEnabled => 0,
        }
    }
}
