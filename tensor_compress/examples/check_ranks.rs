// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
use tensor_compress::{tt_decompose, TTConfig};

fn main() {
    // LCG constant from Knuth's MMIX
    const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

    let dim: u16 = 1024;
    let config = TTConfig::for_dim(usize::from(dim)).unwrap();

    // Sine
    let v_sine: Vec<f32> = (0..dim).map(|i| (f32::from(i) * 0.1).sin()).collect();
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
            state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(1);
            // Use upper 16 bits for lossless conversion to f32
            let upper_bits = (state >> 48) as u16;
            f32::from(upper_bits) / f32::from(u16::MAX)
        })
        .collect();
    let tt_rand = tt_decompose(&v_rand, &config).unwrap();
    println!(
        "Random vector [{:?}] Max Rank: {}",
        config.shape,
        tt_rand.max_rank()
    );
}
