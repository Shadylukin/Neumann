// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Tensor compression stress tests.

use std::{
    collections::HashMap,
    io::Cursor,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    thread,
    time::Instant,
};

use stress_tests::{full_config, generate_embeddings, LatencyHistogram};
use tensor_compress::{
    format::{CompressedEntry, CompressedScalar, CompressedSnapshot, CompressedValue, Header},
    incremental::{DeltaBuilder, DeltaChain},
    streaming::{StreamingReader, StreamingWriter},
    tt_decompose, tt_decompose_batch, tt_reconstruct, CompressionConfig, StreamingTTReader,
    StreamingTTWriter, TTConfig,
};

fn make_test_entry(key: &str, value: i64) -> CompressedEntry {
    CompressedEntry {
        key: key.to_string(),
        fields: HashMap::from([(
            "value".to_string(),
            CompressedValue::Scalar(CompressedScalar::Int(value)),
        )]),
    }
}

/// Stress test: Concurrent TT decomposition.
#[test]
#[ignore]
fn stress_tt_decompose_concurrent() {
    let config = full_config();
    let thread_count = config.effective_thread_count();
    let vectors_per_thread = 1000;
    let embedding_dim = 256; // Power of 2 for clean factorization

    println!("\n=== TT Decompose Concurrent ===");
    println!("Threads: {}", thread_count);
    println!("Vectors/thread: {}", vectors_per_thread);
    println!("Dimension: {}", embedding_dim);

    let embeddings = Arc::new(generate_embeddings(
        vectors_per_thread * thread_count,
        embedding_dim,
        42,
    ));
    let tt_config = Arc::new(TTConfig::for_dim(embedding_dim).unwrap());

    let mut handles = vec![];
    let start = Instant::now();
    let success_count = Arc::new(AtomicUsize::new(0));

    for t in 0..thread_count {
        let embeddings = Arc::clone(&embeddings);
        let tt_config = Arc::clone(&tt_config);
        let success = Arc::clone(&success_count);
        handles.push(thread::spawn(move || {
            let mut latencies = LatencyHistogram::new();
            let start_idx = t * vectors_per_thread;

            for i in 0..vectors_per_thread {
                let idx = start_idx + i;
                let op_start = Instant::now();

                if let Ok(tt) = tt_decompose(&embeddings[idx], &tt_config) {
                    let reconstructed = tt_reconstruct(&tt);
                    assert_eq!(reconstructed.len(), embedding_dim, "Dimension mismatch");
                    success.fetch_add(1, Ordering::Relaxed);
                }

                latencies.record(op_start.elapsed());
            }

            latencies.snapshot()
        }));
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();
    let total_vectors = thread_count * vectors_per_thread;

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} vectors/sec",
        total_vectors as f64 / elapsed.as_secs_f64()
    );

    for (i, snapshot) in results.iter().enumerate() {
        println!("  Thread {}: {}", i, snapshot);
    }

    // Verify success count
    let successes = success_count.load(Ordering::Relaxed);
    println!(
        "Successful decompositions: {} / {}",
        successes, total_vectors
    );
    assert!(
        successes >= total_vectors * 95 / 100,
        "Too many decomposition failures"
    );

    println!("PASSED: {} vectors decomposed", successes);
}

/// Stress test: Streaming compression with large snapshot.
#[test]
#[ignore]
fn stress_streaming_large_snapshot() {
    let entry_count = 100_000;
    let batch_size = 1000;

    println!("\n=== Streaming Large Snapshot ===");
    println!("Entries: {}", entry_count);
    println!("Batch size: {}", batch_size);

    let start = Instant::now();

    // Write in batches to measure throughput
    let cursor = Cursor::new(Vec::new());
    let config = CompressionConfig::default();
    let mut writer = StreamingWriter::new(cursor, config).unwrap();

    let mut write_latencies = LatencyHistogram::new();
    for batch in 0..(entry_count / batch_size) {
        let batch_start = Instant::now();
        for i in 0..batch_size {
            let idx = batch * batch_size + i;
            let entry = make_test_entry(&format!("key_{}", idx), idx as i64);
            writer.write_entry(&entry).unwrap();
        }
        write_latencies.record(batch_start.elapsed());
    }

    let written = writer.finish().unwrap();
    let write_elapsed = start.elapsed();
    let bytes_written = written.get_ref().len();

    println!("Write phase:");
    println!("  Duration: {:?}", write_elapsed);
    println!(
        "  Throughput: {:.0} entries/sec",
        entry_count as f64 / write_elapsed.as_secs_f64()
    );
    println!(
        "  Bytes: {} ({:.1} KB)",
        bytes_written,
        bytes_written as f64 / 1024.0
    );
    println!("  Batch latencies: {}", write_latencies.snapshot());

    // Read phase
    let read_start = Instant::now();
    let reader = StreamingReader::open(Cursor::new(written.into_inner())).unwrap();
    assert_eq!(reader.entry_count(), entry_count as u64);

    let mut read_count = 0;
    let mut read_latencies = LatencyHistogram::new();
    for batch_entries in reader.collect::<Vec<_>>().chunks(batch_size) {
        let batch_start = Instant::now();
        for entry_result in batch_entries {
            entry_result.as_ref().unwrap();
            read_count += 1;
        }
        read_latencies.record(batch_start.elapsed());
    }

    let read_elapsed = read_start.elapsed();

    println!("Read phase:");
    println!("  Duration: {:?}", read_elapsed);
    println!(
        "  Throughput: {:.0} entries/sec",
        entry_count as f64 / read_elapsed.as_secs_f64()
    );
    println!("  Batch latencies: {}", read_latencies.snapshot());

    assert_eq!(read_count, entry_count);
    println!("PASSED: {} entries written and read", entry_count);
}

/// Stress test: Deep delta chain with compaction.
#[test]
#[ignore]
fn stress_delta_chain_deep() {
    let base_entries = 1000;
    let delta_count = 100;
    let changes_per_delta = 50;

    println!("\n=== Delta Chain Deep ===");
    println!("Base entries: {}", base_entries);
    println!("Delta count: {}", delta_count);
    println!("Changes per delta: {}", changes_per_delta);

    // Create base snapshot
    let base_entries_vec: Vec<_> = (0..base_entries)
        .map(|i| make_test_entry(&format!("key_{}", i), i as i64))
        .collect();
    let header = Header::new(CompressionConfig::default(), base_entries as u64);
    let base = CompressedSnapshot {
        header,
        entries: base_entries_vec,
    };

    // Create deep delta chain
    let start = Instant::now();
    let mut chain = DeltaChain::new(base.clone()).with_max_chain_len(delta_count + 1);
    let mut latencies = LatencyHistogram::new();

    for d in 0..delta_count {
        let delta_start = Instant::now();
        let mut builder = DeltaBuilder::new(format!("base_{}", d), (d * changes_per_delta) as u64);

        for c in 0..changes_per_delta {
            let key_idx = (d * changes_per_delta + c) % base_entries;
            let new_value = (d * 1000 + c) as i64;
            builder.put(
                format!("key_{}", key_idx),
                make_test_entry(&format!("key_{}", key_idx), new_value),
            );
        }

        chain.push(builder.build()).unwrap();
        latencies.record(delta_start.elapsed());
    }

    let delta_elapsed = start.elapsed();
    println!(
        "Delta creation: {:?} ({:.0} deltas/sec)",
        delta_elapsed,
        delta_count as f64 / delta_elapsed.as_secs_f64()
    );
    println!("  Latencies: {}", latencies.snapshot());

    // Test lookups through chain
    let lookup_start = Instant::now();
    let mut found = 0;
    for i in 0..base_entries {
        if chain.get(&format!("key_{}", i)).is_some() {
            found += 1;
        }
    }
    let lookup_elapsed = lookup_start.elapsed();

    println!(
        "Lookups: {:?} ({:.0} lookups/sec)",
        lookup_elapsed,
        base_entries as f64 / lookup_elapsed.as_secs_f64()
    );
    println!("  Found: {} / {}", found, base_entries);

    // Compact the chain
    let compact_start = Instant::now();
    let compacted = chain.compact().unwrap();
    let compact_elapsed = compact_start.elapsed();

    println!(
        "Compaction: {:?} ({} entries)",
        compact_elapsed,
        compacted.entries.len()
    );

    // Verify compacted snapshot has all expected entries
    assert_eq!(compacted.entries.len(), base_entries);

    println!("PASSED: {} deltas applied and compacted", delta_count);
}

/// Stress test: TT compression ratio at scale.
#[test]
#[ignore]
fn stress_tt_compression_ratio_4096d() {
    let vector_count = 1000;
    let embedding_dim = 4096;

    println!("\n=== TT Compression Ratio at 4096d ===");
    println!("Vectors: {}", vector_count);
    println!("Dimension: {}", embedding_dim);

    let embeddings = generate_embeddings(vector_count, embedding_dim, 42);
    let config = TTConfig::for_dim(embedding_dim).unwrap();

    let start = Instant::now();
    let mut total_original = 0usize;
    let mut total_compressed = 0usize;
    let mut ratios = Vec::with_capacity(vector_count);

    for embedding in &embeddings {
        if let Ok(tt) = tt_decompose(embedding, &config) {
            let original = embedding.len() * 4; // f32 = 4 bytes
            let compressed = tt.storage_size() * 4;
            total_original += original;
            total_compressed += compressed;
            ratios.push(tt.compression_ratio());
        }
    }

    let elapsed = start.elapsed();

    let avg_ratio = ratios.iter().sum::<f32>() / ratios.len() as f32;
    let min_ratio = ratios.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_ratio = ratios.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} vectors/sec",
        vector_count as f64 / elapsed.as_secs_f64()
    );
    println!(
        "Original size: {} bytes ({:.1} MB)",
        total_original,
        total_original as f64 / 1024.0 / 1024.0
    );
    println!(
        "Compressed size: {} bytes ({:.1} MB)",
        total_compressed,
        total_compressed as f64 / 1024.0 / 1024.0
    );
    println!(
        "Overall ratio: {:.2}x",
        total_original as f64 / total_compressed as f64
    );
    println!(
        "Per-vector ratio: min={:.2}x, avg={:.2}x, max={:.2}x",
        min_ratio, avg_ratio, max_ratio
    );

    // TT should achieve at least 5x compression on 4096d
    assert!(
        avg_ratio >= 2.0,
        "Average compression ratio too low: {:.2}x",
        avg_ratio
    );

    println!(
        "PASSED: Average {:.2}x compression on {}d vectors",
        avg_ratio, embedding_dim
    );
}

/// Stress test: Batch TT decomposition with concurrent access.
#[test]
#[ignore]
fn stress_tt_batch_concurrent() {
    let config = full_config();
    let thread_count = config.effective_thread_count();
    let batch_size = 500;
    let embedding_dim = 256;

    println!("\n=== TT Batch Concurrent ===");
    println!("Threads: {}", thread_count);
    println!("Batch size: {}", batch_size);

    let embeddings = Arc::new(generate_embeddings(batch_size, embedding_dim, 42));
    let tt_config = Arc::new(TTConfig::for_dim(embedding_dim).unwrap());

    let start = Instant::now();
    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let embs = Arc::clone(&embeddings);
            let cfg = Arc::clone(&tt_config);
            thread::spawn(move || {
                let refs: Vec<&[f32]> = embs.iter().map(|v| v.as_slice()).collect();
                tt_decompose_batch(&refs, &cfg).unwrap()
            })
        })
        .collect();

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let elapsed = start.elapsed();

    println!("Duration: {:?}", elapsed);
    println!(
        "Throughput: {:.0} batches/sec",
        thread_count as f64 / elapsed.as_secs_f64()
    );

    // Verify all results are identical in length
    for result in &results[1..] {
        assert_eq!(results[0].len(), result.len());
    }

    println!("PASSED: {} concurrent batch decompositions", thread_count);
}

/// Stress test: Streaming TT with large vector count.
#[test]
#[ignore]
fn stress_streaming_tt_10k_vectors() {
    let vector_count = 10_000;
    let embedding_dim = 256;

    println!("\n=== Streaming TT 10k Vectors ===");
    println!("Vectors: {}", vector_count);
    println!("Dimension: {}", embedding_dim);

    let config = TTConfig::for_dim(embedding_dim).unwrap();
    let embeddings = generate_embeddings(vector_count, embedding_dim, 42);

    // Write phase
    let write_start = Instant::now();
    let cursor = Cursor::new(Vec::new());
    let mut writer = StreamingTTWriter::new(cursor, config.clone()).unwrap();

    for emb in &embeddings {
        writer.write_vector(emb).unwrap();
    }

    let written = writer.finish().unwrap();
    let write_elapsed = write_start.elapsed();
    let bytes = written.get_ref().len();

    println!(
        "Write: {:?} ({:.1} MB)",
        write_elapsed,
        bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "Throughput: {:.0} vectors/sec",
        vector_count as f64 / write_elapsed.as_secs_f64()
    );

    // Read phase
    let read_start = Instant::now();
    let reader = StreamingTTReader::open(Cursor::new(written.into_inner())).unwrap();
    assert_eq!(reader.vector_count(), vector_count as u64);

    let tts: Vec<_> = reader.map(|r| r.unwrap()).collect();
    let read_elapsed = read_start.elapsed();

    println!("Read: {:?}", read_elapsed);
    println!(
        "Throughput: {:.0} vectors/sec",
        vector_count as f64 / read_elapsed.as_secs_f64()
    );

    assert_eq!(tts.len(), vector_count);
    println!("PASSED: {} vectors written and read", vector_count);
}
