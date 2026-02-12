// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_cache::{EvictionScorer, EvictionStrategy};

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    strategy_type: u8,
    lru_weight: u8,
    lfu_weight: u8,
    cost_weight: u8,
    last_access_secs: f64,
    access_count: u64,
    cost_per_hit: f64,
    size_bytes: usize,
}

fuzz_target!(|input: FuzzInput| {
    // Ensure we don't have NaN or infinity
    if !input.last_access_secs.is_finite() || !input.cost_per_hit.is_finite() {
        return;
    }

    // Ensure non-negative values
    let last_access_secs = input.last_access_secs.abs();
    let cost_per_hit = input.cost_per_hit.abs();

    // Select strategy based on input
    let strategy = match input.strategy_type % 4 {
        0 => EvictionStrategy::LRU,
        1 => EvictionStrategy::LFU,
        2 => EvictionStrategy::CostBased,
        _ => EvictionStrategy::Hybrid {
            lru_weight: input.lru_weight.max(1),
            lfu_weight: input.lfu_weight.max(1),
            cost_weight: input.cost_weight.max(1),
        },
    };

    let scorer = EvictionScorer::new(strategy);
    let score = scorer.score(
        last_access_secs,
        input.access_count,
        cost_per_hit,
        input.size_bytes,
    );

    // Verify the score is finite (no NaN or infinity)
    assert!(
        score.is_finite(),
        "Score must be finite, got {} for strategy {:?}, inputs: last_access={}, count={}, cost={}, size={}",
        score,
        strategy,
        last_access_secs,
        input.access_count,
        cost_per_hit,
        input.size_bytes
    );
});
