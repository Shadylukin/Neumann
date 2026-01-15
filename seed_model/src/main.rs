//! Pure Geometric Training - No Candle, No RNN
//!
//! Trains a model where structure emerges from data:
//! - Depth emerges from sparsity convergence
//! - Temperature is learned per level
//! - Response IS the geometric shape

use std::{collections::HashMap, path::Path, time::Instant};

use seed_model::{
    corpus::{CorpusConfig, TextCorpus},
    GeometricConfig, PureGeometricModel,
};

const SEQ_LEN: usize = 32;
const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 50;
const LEARNING_RATE: f32 = 0.01;
const TIME_LIMIT_SECS: u64 = 600;
const MAX_SEQUENCES: usize = 50000; // Limit training data for faster iteration

fn main() -> anyhow::Result<()> {
    println!("============================================================");
    println!("PURE GEOMETRIC INTELLIGENCE");
    println!("============================================================");
    println!();
    println!("Tensor Scaling Theory - Pure Implementation:");
    println!("  - No RNN (response IS the shape)");
    println!("  - No forced dimensions (sparsity is native)");
    println!("  - No Candle (pure Rust sparse ops)");
    println!("  - Adaptive depth (stop when 90% sparse)");
    println!();

    // Load corpus
    let corpus_path = Path::new("micro_corpus.txt");
    let corpus_config = CorpusConfig {
        max_chars: 2_000_000,
        dolly_limit: 5000,
        stories_chars: 1_000_000,
    };

    let corpus = TextCorpus::load_or_build(corpus_path, &corpus_config)?;
    println!(
        "Corpus: {} chars, {} vocab",
        corpus.text.len(),
        corpus.vocab_size
    );

    // Model config - capacities, not forced structure
    let config = GeometricConfig::from_vocab(corpus.vocab_size);

    println!();
    println!("Model config (capacities, not forced):");
    println!("  Vocab: {}", config.vocab_size);
    println!("  Dim capacity: {}", config.dim_capacity);
    println!("  Level capacity: {}", config.num_levels);
    println!("  Keys per level: {}", config.num_keys);
    println!("  Sparsity target: {:.0}%", config.sparsity_target * 100.0);
    println!();

    // Create model
    let mut model = PureGeometricModel::new(&config);

    // Estimate parameters (sparse, so actual memory is less)
    let embed_params = config.vocab_size * config.initial_nnz;
    let vq_params = config.num_levels * config.num_keys * config.initial_nnz * 2;
    println!(
        "Approximate non-zero parameters: {}",
        embed_params + vq_params
    );
    println!();

    // Training data (limited for faster iteration)
    println!("Creating training sequences...");
    let all_sequences = corpus.create_sequences(SEQ_LEN);
    let sequences: Vec<_> = all_sequences.into_iter().take(MAX_SEQUENCES).collect();
    let num_batches = sequences.len() / BATCH_SIZE;
    println!(
        "Sequences: {} ({} batches of {})",
        sequences.len(),
        num_batches,
        BATCH_SIZE
    );

    // Training loop
    println!();
    println!("============================================================");
    println!("TRAINING - Watching for emergence");
    println!("============================================================");
    println!();

    let start_time = Instant::now();
    let mut best_loss = f32::MAX;
    let mut depth_histogram: HashMap<usize, usize> = HashMap::new();

    for epoch in 1..=EPOCHS {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut epoch_depth_sum = 0usize;
        let mut epoch_sparsity = 0.0;
        let mut epoch_samples = 0usize;

        // Shuffle indices
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..sequences.len()).collect();
        indices.shuffle(&mut rand::thread_rng());

        for batch_idx in 0..num_batches {
            if start_time.elapsed().as_secs() > TIME_LIMIT_SECS {
                println!("\n>>> TIME LIMIT REACHED");
                print_emergence_summary(&depth_histogram);
                return Ok(());
            }

            let batch_start = batch_idx * BATCH_SIZE;
            let batch_indices = &indices[batch_start..batch_start + BATCH_SIZE];

            let mut batch_loss = 0.0;
            let mut batch_depth = 0usize;
            let mut batch_sparsity = 0.0;

            for &idx in batch_indices {
                let (input, target) = &sequences[idx];

                // Forward pass
                let (logits, depths, sparsities) = model.forward(input);

                // Track emergence
                for &d in &depths {
                    *depth_histogram.entry(d).or_insert(0) += 1;
                    batch_depth += d;
                }
                batch_sparsity += sparsities.iter().sum::<f32>() / sparsities.len() as f32;

                // Compute loss and train
                let loss = model.train_step(input, target, LEARNING_RATE);
                batch_loss += loss;
            }

            let avg_loss = batch_loss / BATCH_SIZE as f32;
            let avg_depth = batch_depth as f32 / (BATCH_SIZE * SEQ_LEN) as f32;
            let avg_sparsity = batch_sparsity / BATCH_SIZE as f32;

            epoch_loss += avg_loss;
            epoch_depth_sum += batch_depth;
            epoch_sparsity += avg_sparsity;
            epoch_samples += BATCH_SIZE;

            // Progress
            if batch_idx % 50 == 0 {
                let elapsed = start_time.elapsed().as_secs_f32();
                let batches_per_sec =
                    (epoch_samples as f32 / BATCH_SIZE as f32) / elapsed.max(0.001);
                println!(
                    "Ep {:2} | B {:4}/{} | Loss {:.3} | Depth {:.1} | Sparse {:.0}% | {:.1} batch/s",
                    epoch, batch_idx, num_batches,
                    avg_loss, avg_depth, avg_sparsity * 100.0, batches_per_sec
                );
            }
        }

        // Epoch summary
        let n = num_batches as f32;
        let avg_loss = epoch_loss / n;
        let avg_depth = epoch_depth_sum as f32 / (epoch_samples * SEQ_LEN) as f32;
        let avg_sparsity = epoch_sparsity / n;
        let epoch_time = epoch_start.elapsed();

        println!();
        println!(">> EPOCH {} ({:.1}s)", epoch, epoch_time.as_secs_f32());
        println!("   Loss: {:.4}", avg_loss);
        println!(
            "   Emergent depth: {:.2} | Sparsity: {:.1}%",
            avg_depth,
            avg_sparsity * 100.0
        );

        if avg_loss < best_loss {
            best_loss = avg_loss;
            println!("   [NEW BEST]");
        }
        println!();

        // Early stopping if consistently achieving sparsity target
        if avg_sparsity >= config.sparsity_target && epoch > 5 {
            println!("Sparsity target achieved! Training complete.");
            break;
        }
    }

    print_emergence_summary(&depth_histogram);

    // Test generation
    println!();
    println!("============================================================");
    println!("GENERATION TEST");
    println!("============================================================");
    test_generation(&model, &corpus)?;

    Ok(())
}

fn print_emergence_summary(depth_hist: &HashMap<usize, usize>) {
    println!();
    println!("============================================================");
    println!("EMERGENCE SUMMARY");
    println!("============================================================");
    println!();

    // Depth distribution
    println!("Depth distribution (emergent):");
    let total: usize = depth_hist.values().sum();
    if total > 0 {
        let mut depths: Vec<_> = depth_hist.iter().collect();
        depths.sort_by_key(|(d, _)| *d);
        for (depth, count) in depths {
            let pct = (*count as f32 / total as f32) * 100.0;
            let bar = "#".repeat((pct / 5.0) as usize);
            println!("  Depth {:2}: {:5} ({:4.1}%) {}", depth, count, pct, bar);
        }
    } else {
        println!("  No depth data collected");
    }
}

fn test_generation(model: &PureGeometricModel, corpus: &TextCorpus) -> anyhow::Result<()> {
    let prompts = ["QUESTION:\nWhat is", "Once upon a time", "The quick brown"];

    for prompt in &prompts {
        println!();
        println!("Prompt: \"{}\"", prompt);

        let mut tokens: Vec<usize> = corpus.encode(prompt);

        // Pad to sequence length
        while tokens.len() < SEQ_LEN {
            tokens.insert(0, 0);
        }

        let mut generated = prompt.to_string();
        let mut total_depth = 0usize;
        let mut gen_count = 0usize;

        for _ in 0..100 {
            // Get the last SEQ_LEN tokens
            let input: Vec<usize> = tokens.iter().rev().take(SEQ_LEN).rev().copied().collect();

            let (logits, depths, _) = model.forward(&input);

            // Get last position's logits
            let last_logits = &logits[logits.len() - 1];
            total_depth += depths.iter().sum::<usize>();
            gen_count += depths.len();

            // Sample from sparse logits (greedy for now)
            let next_token = sample_from_sparse(last_logits, 0.8);

            tokens.push(next_token);
            let char_str = corpus.decode(&[next_token]);
            generated.push_str(&char_str);

            // Stop on double newline
            if generated.ends_with("\n\n") {
                break;
            }
        }

        let avg_depth = if gen_count > 0 {
            total_depth as f32 / gen_count as f32
        } else {
            0.0
        };

        println!(
            "Generated ({} chars, avg depth {:.1}):",
            generated.len(),
            avg_depth
        );
        println!("{}", generated);
    }

    Ok(())
}

fn sample_from_sparse(logits: &tensor_store::SparseVector, temperature: f32) -> usize {
    use rand::Rng;

    let positions = logits.positions();
    let values = logits.values();

    if positions.is_empty() {
        return 0; // Fallback
    }

    // Apply temperature and softmax
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = values
        .iter()
        .map(|&v| ((v - max_val) / temperature).exp())
        .collect();
    let sum: f32 = exp_vals.iter().sum();

    if sum == 0.0 {
        return positions[0] as usize;
    }

    // Sample
    let mut rng = rand::thread_rng();
    let threshold: f32 = rng.gen();
    let mut cumsum = 0.0;

    for (i, &e) in exp_vals.iter().enumerate() {
        cumsum += e / sum;
        if cumsum >= threshold {
            return positions[i] as usize;
        }
    }

    positions[positions.len() - 1] as usize
}
