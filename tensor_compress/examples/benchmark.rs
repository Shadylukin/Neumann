// SPDX-License-Identifier: MIT OR Apache-2.0
//! Quick TT benchmark
use std::time::Instant;

use tensor_compress::{
    tt_cosine_similarity, tt_cosine_similarity_batch, tt_decompose, tt_decompose_batch,
    tt_reconstruct, TTConfig,
};

fn main() {
    let dims = [64, 256, 768, 1536, 4096];

    println!("=== TT Decomposition Benchmark (Release Build) ===\n");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10}",
        "Dim", "Decompose", "Reconstruct", "Similarity", "Ratio"
    );
    println!("{}", "-".repeat(60));

    for dim in dims {
        let vector: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();
        let config = TTConfig::for_dim(dim).unwrap();

        // Warm up
        let _ = tt_decompose(&vector, &config);

        // Benchmark decompose
        let iterations = 1000;
        let start = Instant::now();
        let mut tt = None;
        for _ in 0..iterations {
            tt = Some(tt_decompose(&vector, &config).unwrap());
        }
        let decompose_us = start.elapsed().as_micros() as f64 / iterations as f64;
        let tt = tt.unwrap();

        // Benchmark reconstruct
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tt_reconstruct(&tt);
        }
        let reconstruct_us = start.elapsed().as_micros() as f64 / iterations as f64;

        // Benchmark similarity
        let tt2 = tt_decompose(&vector, &config).unwrap();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = tt_cosine_similarity(&tt, &tt2);
        }
        let similarity_us = start.elapsed().as_micros() as f64 / iterations as f64;

        let ratio = tt.compression_ratio();

        println!(
            "{:>8} {:>10.1} µs {:>10.1} µs {:>10.1} µs {:>9.1}x",
            dim, decompose_us, reconstruct_us, similarity_us, ratio
        );
    }

    // Batch benchmark
    println!("\n=== Batch Operations (1000 vectors, 768-dim) ===\n");
    let dim = 768;
    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|i| (0..dim).map(|j| ((i * j) as f32 * 0.01).sin()).collect())
        .collect();
    let config = TTConfig::for_dim(dim).unwrap();
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    let start = Instant::now();
    let tts = tt_decompose_batch(&refs, &config).unwrap();
    let batch_ms = start.elapsed().as_millis();
    println!(
        "Batch decompose 1000x{}: {} ms ({:.1} µs/vector)",
        dim,
        batch_ms,
        batch_ms as f64 * 1000.0 / 1000.0
    );

    let start = Instant::now();
    let _ = tt_cosine_similarity_batch(&tts[0], &tts[1..]).unwrap();
    let batch_sim_us = start.elapsed().as_micros();
    println!(
        "Batch similarity 1 vs 999: {} µs ({:.2} µs/comparison)",
        batch_sim_us,
        batch_sim_us as f64 / 999.0
    );

    // Throughput
    println!("\n=== Throughput ===\n");
    let dim = 768;
    let config = TTConfig::for_dim(dim).unwrap();
    let vector: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.01).sin()).collect();

    let start = Instant::now();
    let mut count = 0;
    while start.elapsed().as_secs() < 2 {
        let _ = tt_decompose(&vector, &config).unwrap();
        count += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "Decomposition throughput (768-dim): {:.0} vectors/sec",
        count as f64 / elapsed
    );
}
