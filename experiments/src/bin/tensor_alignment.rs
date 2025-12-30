use std::time::Instant;
use tensor_store::SparseVector;

fn main() {
    println!("=== Tensor Alignment Experiment ===\n");
    println!("Proving silicon thinks in tensors, not scalars.\n");

    // Part 1: SIMD alignment
    let size = 1_000_000usize;
    let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..size).map(|i| i as f32 * 0.002).collect();

    // Method 1: Scalar (how we "think" computers work)
    let start = Instant::now();
    let mut scalar_result = 0.0f32;
    for i in 0..size {
        scalar_result += a[i] * b[i];
    }
    let scalar_time = start.elapsed();

    // Method 2: Chunks of 4 (how silicon actually works - SIMD width)
    let start = Instant::now();
    let mut chunk_result = 0.0f32;
    for chunk in a.chunks(4).zip(b.chunks(4)) {
        let (ca, cb) = chunk;
        chunk_result += ca[0] * cb[0] + ca[1] * cb[1] + ca[2] * cb[2] + ca[3] * cb[3];
    }
    let chunk_time = start.elapsed();

    // Method 3: Let LLVM see the tensor (iterator fusion)
    let start = Instant::now();
    let iter_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let iter_time = start.elapsed();

    println!("--- Dense Vector Dot Product (n={}) ---", size);
    println!("Scalar:   {:?} = {:.2}", scalar_time, scalar_result);
    println!("Chunk-4:  {:?} = {:.2}", chunk_time, chunk_result);
    println!("Iterator: {:?} = {:.2}", iter_time, iter_result);
    println!();
    println!(
        "Ratio (scalar/iter): {:.1}x",
        scalar_time.as_nanos() as f64 / iter_time.as_nanos() as f64
    );

    println!("\n--- Neumann SparseVector (native tensor representation) ---\n");

    // Now demonstrate with Neumann's SparseVector
    // Sparse vectors only store non-zero elements - geometry, not coordinates
    let dimension = 1_000_000;
    let sparse_indices: Vec<u32> = (0..1000).map(|i| i * 1000).collect();
    let sparse_values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();

    let sv1 = SparseVector::from_parts(dimension, sparse_indices.clone(), sparse_values.clone());
    let sv2 = SparseVector::from_parts(dimension, sparse_indices.clone(), sparse_values.clone());

    // Sparse dot product - O(k) where k is number of non-zeros
    let start = Instant::now();
    let sparse_dot = sv1.dot(&sv2);
    let sparse_time = start.elapsed();

    // Compare to dense representation of the same data
    let mut dense1 = vec![0.0f32; 1_000_000];
    let mut dense2 = vec![0.0f32; 1_000_000];
    for (idx, val) in sparse_indices.iter().zip(sparse_values.iter()) {
        dense1[*idx as usize] = *val;
        dense2[*idx as usize] = *val;
    }

    let start = Instant::now();
    let dense_dot: f32 = dense1.iter().zip(dense2.iter()).map(|(x, y)| x * y).sum();
    let dense_time = start.elapsed();

    println!(
        "Sparse (1000 non-zeros): {:?} = {:.2}",
        sparse_time, sparse_dot
    );
    println!(
        "Dense  (1M elements):    {:?} = {:.2}",
        dense_time, dense_dot
    );
    println!();
    println!(
        "Ratio (dense/sparse): {:.0}x",
        dense_time.as_nanos() as f64 / sparse_time.as_nanos().max(1) as f64
    );

    // Part 3: Ceremony analysis
    println!("\n--- Ceremony vs Real Work ---\n");

    let sparse_ops = 1000u64; // actual multiplications in sparse dot
    let dense_ops = 1_000_000u64; // iterations performed in dense
    let real_work_ratio = sparse_ops as f64 / dense_ops as f64;

    println!("Sparse operations:  {:>10}", sparse_ops);
    println!("Dense operations:   {:>10}", dense_ops);
    println!();
    println!("Real work: {:>6.2}%", real_work_ratio * 100.0);
    println!("Ceremony:  {:>6.2}%", (1.0 - real_work_ratio) * 100.0);

    // Part 4: TensorData field sparsity (simulated)
    println!("\n--- TensorData Field Sparsity ---\n");

    // Typical entity: 5 fields set out of 50 possible schema fields
    let fields_set = 5u64;
    let schema_fields = 50u64;
    let field_ceremony = 1.0 - (fields_set as f64 / schema_fields as f64);

    println!("Fields actually set:   {:>3}", fields_set);
    println!("Possible schema fields: {:>3}", schema_fields);
    println!("Field ceremony:       {:>6.1}%", field_ceremony * 100.0);
    println!();
    println!("Traditional DB scans all 50 columns.");
    println!("TensorData HashMap: only 5 lookups exist.");

    // Part 5: Delta transmission savings
    println!("\n--- Delta Transmission (tensor_chain) ---\n");

    let full_embedding_bytes = 768 * 4; // 768-dim f32 embedding
    let archetype_id_bytes = 4; // u32
    let avg_delta_indices = 50; // typical sparse delta
    let delta_bytes = archetype_id_bytes + avg_delta_indices * 4 + avg_delta_indices * 4;

    let transmission_savings = 1.0 - (delta_bytes as f64 / full_embedding_bytes as f64);

    println!("Full embedding:     {:>5} bytes", full_embedding_bytes);
    println!("Delta transmission: {:>5} bytes", delta_bytes);
    println!("Bandwidth saved:    {:>5.1}%", transmission_savings * 100.0);
    println!();
    println!("tensor_chain sends (archetype_id, sparse_delta),");
    println!("not full embeddings. Only excitations travel the wire.");

    // Part 6: Codebook quantization
    println!("\n--- Codebook Quantization ---\n");

    let vector_bytes = 768 * 4;
    let codebook_index_bytes = 4; // u32 global + optional u32 local
    let quantization_ratio = codebook_index_bytes as f64 / vector_bytes as f64;

    println!("Full vector:    {:>5} bytes", vector_bytes);
    println!("Codebook index: {:>5} bytes", codebook_index_bytes);
    println!("Compression:    {:>5.0}x", 1.0 / quantization_ratio);
    println!();
    println!("GlobalCodebook maps continuous states to discrete codes.");
    println!("Consensus validates indices, not byte arrays.");

    println!("\n=== Insight ===\n");
    println!("The iterator version isn't 'optimized' - it just stops lying");
    println!("to the hardware about what we want. The CPU has tensor units");
    println!("(SIMD/NEON) waiting for patterns it recognizes.");
    println!();
    println!("SparseVector goes further: it encodes geometry, not coordinates.");
    println!("1000x speedup because we stopped storing zeros that don't exist.");
    println!();
    println!("Neumann's architecture is sparse at every layer:");
    println!("  - TensorData:  HashMap fields (no schema iteration)");
    println!("  - SparseVector: positions + values (no zeros stored)");
    println!("  - DeltaVector:  archetype + sparse residual");
    println!("  - tensor_chain: delta-only transmission (excitations only)");
    println!("  - Codebook:     indices into vocabulary (not full vectors)");
    println!("  - Consensus:    cosine similarity (shape, not bytes)");
    println!();
    println!("99.9% of traditional computation is ceremony -");
    println!("visiting positions that contain no information.");
    println!();
    println!("Neumann is trying to be just the physics.");
}
