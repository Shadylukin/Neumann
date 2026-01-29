// SPDX-License-Identifier: MIT OR Apache-2.0
use tensor_compress::{tt_decompose, TTConfig};

fn main() {
    let dim = 1024;
    let config = TTConfig::for_dim(dim).unwrap();

    // Sine
    let v_sine: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let tt_sine = tt_decompose(&v_sine, &config).unwrap();
    println!(
        "Sine vector [{:?}] Max Rank: {}",
        config.shape,
        tt_sine.max_rank()
    );

    // Random (LCG-like pseudo random)
    let mut state: u64 = 42;
    let v_rand: Vec<f32> = (0..dim)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state as f32) / (u64::MAX as f32)
        })
        .collect();
    let tt_rand = tt_decompose(&v_rand, &config).unwrap();
    println!(
        "Random vector [{:?}] Max Rank: {}",
        config.shape,
        tt_rand.max_rank()
    );
}
