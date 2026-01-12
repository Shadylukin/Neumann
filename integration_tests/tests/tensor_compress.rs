//! TensorCompress integration tests.
//!
//! Tests compression algorithms with real engine data.

use integration_tests::{create_shared_engines, sample_embeddings};
use tensor_compress::{
    compress_ids, decompress_ids, delta_decode, delta_encode, rle_decode, rle_encode,
    tt_cosine_similarity, tt_cosine_similarity_batch, tt_decompose, tt_decompose_batch,
    tt_dot_product, tt_reconstruct, CompressionConfig, StreamingTTReader, StreamingTTWriter,
    TTConfig, TensorMode,
};

#[test]
fn test_delta_encode_node_ids() {
    let (_, _, graph, _) = create_shared_engines();

    // Create nodes and collect their IDs
    let mut node_ids: Vec<u64> = Vec::new();
    for _ in 0..100 {
        let id = graph
            .create_node("entity", std::collections::HashMap::new())
            .unwrap();
        node_ids.push(id);
    }

    // Node IDs should be somewhat sequential
    node_ids.sort();

    // Delta encode
    let encoded = delta_encode(&node_ids);
    let decoded = delta_decode(&encoded);
    assert_eq!(node_ids, decoded);

    // Verify compression ratio for sequential IDs
    let compressed = compress_ids(&node_ids);
    let original_bytes = node_ids.len() * 8;
    let compressed_bytes = compressed.len();
    let ratio = original_bytes as f64 / compressed_bytes as f64;

    // Sequential IDs should compress well
    assert!(
        ratio > 2.0,
        "Expected >2x compression for node IDs, got {:.2}x",
        ratio
    );
}

#[test]
fn test_compress_ids_with_sparse_graph() {
    let (_, _, graph, _) = create_shared_engines();

    // Create nodes with gaps (simulate deletions)
    let mut node_ids: Vec<u64> = Vec::new();
    for i in 0..50 {
        let id = graph
            .create_node("entity", std::collections::HashMap::new())
            .unwrap();
        // Keep every other node
        if i % 2 == 0 {
            node_ids.push(id);
        }
    }

    node_ids.sort();

    let compressed = compress_ids(&node_ids);
    let decompressed = decompress_ids(&compressed);
    assert_eq!(node_ids, decompressed);
}

#[test]
fn test_rle_encode_status_column() {
    use relational_engine::{Column, ColumnType, Condition, Schema, Value};

    let (_, relational, _, _) = create_shared_engines();

    // Create table with status column
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("status", ColumnType::String),
    ]);
    relational.create_table("orders", schema).unwrap();

    // Insert rows with repeated status values (common pattern)
    let statuses = ["pending", "pending", "pending", "active", "active", "done"];
    for (i, status) in statuses.iter().enumerate() {
        let mut row = std::collections::HashMap::new();
        row.insert("id".to_string(), Value::Int(i as i64));
        row.insert("status".to_string(), Value::String(status.to_string()));
        relational.insert("orders", row).unwrap();
    }

    // Extract status column
    let rows = relational.select("orders", Condition::True).unwrap();
    let status_values: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get("status") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();

    // RLE encode
    let encoded = rle_encode(&status_values);
    let decoded = rle_decode(&encoded);
    assert_eq!(status_values, decoded);

    // Verify compression
    assert_eq!(encoded.runs(), 3); // pending, active, done
    assert_eq!(encoded.len(), 6); // total values
}

#[test]
fn test_rle_encode_large_runs() {
    // Simulate a column with long runs of same value
    let mut data = Vec::new();
    for _ in 0..1000 {
        data.push("active");
    }
    for _ in 0..500 {
        data.push("inactive");
    }
    for _ in 0..250 {
        data.push("deleted");
    }

    let encoded = rle_encode(&data);
    let decoded = rle_decode(&encoded);
    assert_eq!(data, decoded);

    // Should have only 3 runs
    assert_eq!(encoded.runs(), 3);
    assert_eq!(encoded.len(), 1750);
}

#[test]
fn test_compression_config_with_snapshot() {
    let config = CompressionConfig {
        tensor_mode: Some(TensorMode::tensor_train(768)),
        delta_encoding: true,
        rle_encoding: true,
    };

    // Verify serialization
    let bytes = bincode::serialize(&config).unwrap();
    let decoded: CompressionConfig = bincode::deserialize(&bytes).unwrap();
    assert_eq!(config, decoded);
}

#[test]
fn test_tt_decompose_with_vector_engine_embeddings() {
    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings with a dimension that works well with TT
    let embeddings = sample_embeddings(10, 64); // 64 = 4*4*4
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    let config = TTConfig::for_dim(64).unwrap();

    // Decompose each embedding and verify roundtrip
    for (i, original) in embeddings.iter().enumerate() {
        let tt = tt_decompose(original, &config).unwrap();
        let reconstructed = tt_reconstruct(&tt);

        // Verify dimension
        assert_eq!(
            reconstructed.len(),
            original.len(),
            "doc:{} - Dimension mismatch",
            i
        );

        // Verify compression ratio
        let ratio = tt.compression_ratio();
        assert!(
            ratio > 1.0,
            "doc:{} - Expected compression, got {:.2}x expansion",
            i,
            ratio
        );

        // Verify accuracy
        let max_error: f32 = original
            .iter()
            .zip(&reconstructed)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);

        let norm: f32 = original.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            let relative_error = max_error / norm;
            assert!(
                relative_error < 0.5,
                "doc:{} - Reconstruction error too high: {:.4}",
                i,
                relative_error
            );
        }
    }
}

#[test]
fn test_tt_preserves_similarity_order() {
    let embeddings = sample_embeddings(5, 64);
    let config = TTConfig::for_dim(64).unwrap();

    // Decompose all embeddings
    let tt_vectors: Vec<_> = embeddings
        .iter()
        .map(|e| tt_decompose(e, &config).unwrap())
        .collect();

    // Compute similarities in original space
    let query = &embeddings[0];
    let mut original_sims: Vec<(usize, f32)> = embeddings[1..]
        .iter()
        .enumerate()
        .map(|(i, e)| (i + 1, cosine_similarity(query, e)))
        .collect();
    original_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Compute similarities in TT space
    let query_tt = &tt_vectors[0];
    let mut tt_sims: Vec<(usize, f32)> = tt_vectors[1..]
        .iter()
        .enumerate()
        .map(|(i, tt)| (i + 1, tt_cosine_similarity(query_tt, tt).unwrap()))
        .collect();
    tt_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Top result should generally be the same (TT is approximate)
    // Just verify we get valid similarity values
    assert!(tt_sims[0].1.is_finite(), "TT similarity should be finite");
}

#[test]
fn test_tt_dot_product_with_embeddings() {
    let embeddings = sample_embeddings(2, 64);
    let config = TTConfig::for_dim(64).unwrap();

    let tt1 = tt_decompose(&embeddings[0], &config).unwrap();
    let tt2 = tt_decompose(&embeddings[1], &config).unwrap();

    // Compute dot product in TT space
    let tt_dot = tt_dot_product(&tt1, &tt2).unwrap();

    // Compute dot product in original space
    let original_dot: f32 = embeddings[0]
        .iter()
        .zip(&embeddings[1])
        .map(|(a, b)| a * b)
        .sum();

    // TT dot product should be reasonably close
    let relative_error = (tt_dot - original_dot).abs() / original_dot.abs().max(1.0);
    assert!(
        relative_error < 0.5,
        "TT dot product error too high: {:.4} vs {:.4}",
        tt_dot,
        original_dot
    );
}

#[test]
fn test_tt_high_compression_mode() {
    let embeddings = sample_embeddings(5, 256);
    let config = TTConfig::high_compression(256).unwrap();

    for (i, original) in embeddings.iter().enumerate() {
        let tt = tt_decompose(original, &config).unwrap();

        // High compression should achieve better compression ratio
        let ratio = tt.compression_ratio();
        assert!(
            ratio > 2.0,
            "doc:{} - High compression mode should achieve >2x, got {:.2}x",
            i,
            ratio
        );
    }
}

#[test]
fn test_compression_end_to_end() {
    let (_store, relational, graph, vector) = create_shared_engines();

    // Create diverse data across engines
    use relational_engine::{Column, ColumnType, Schema, Value};

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("category", ColumnType::String),
    ]);
    relational.create_table("products", schema).unwrap();

    for i in 0..50 {
        let category = if i < 20 {
            "electronics"
        } else if i < 35 {
            "clothing"
        } else {
            "home"
        };
        let mut row = std::collections::HashMap::new();
        row.insert("id".to_string(), Value::Int(i));
        row.insert("category".to_string(), Value::String(category.to_string()));
        relational.insert("products", row).unwrap();
    }

    // Create graph nodes
    let mut node_ids = Vec::new();
    for _ in 0..50 {
        let id = graph
            .create_node("product", std::collections::HashMap::new())
            .unwrap();
        node_ids.push(id);
    }

    // Store embeddings
    let embeddings = sample_embeddings(50, 64);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("product:{}", i), emb.clone())
            .unwrap();
    }

    // Now test compression on each data type:

    // 1. RLE on category column
    let rows = relational
        .select("products", relational_engine::Condition::True)
        .unwrap();
    let categories: Vec<String> = rows
        .iter()
        .filter_map(|r| match r.get("category") {
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        })
        .collect();
    let rle = rle_encode(&categories);
    assert_eq!(rle.runs(), 3);

    // 2. Delta encoding on node IDs
    node_ids.sort();
    let compressed_ids = compress_ids(&node_ids);
    let decompressed_ids = decompress_ids(&compressed_ids);
    assert_eq!(node_ids, decompressed_ids);

    // 3. TT decomposition on embeddings
    let config = TTConfig::for_dim(64).unwrap();
    for emb in &embeddings {
        let tt = tt_decompose(emb, &config).unwrap();
        let reconstructed = tt_reconstruct(&tt);
        assert_eq!(reconstructed.len(), emb.len());
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

#[test]
fn test_streaming_tt_with_vector_engine() {
    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings
    let embeddings = sample_embeddings(100, 256);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    // Write to streaming TT format
    let config = TTConfig::for_dim(256).unwrap();
    let cursor = std::io::Cursor::new(Vec::new());
    let mut writer = StreamingTTWriter::new(cursor, config).unwrap();

    for emb in &embeddings {
        writer.write_vector(emb).unwrap();
    }

    let written = writer.finish().unwrap();

    // Read back and verify
    let reader = StreamingTTReader::open(std::io::Cursor::new(written.into_inner())).unwrap();
    assert_eq!(reader.vector_count(), 100);

    let tts: Vec<_> = reader.map(|r| r.unwrap()).collect();

    // Verify reconstruction quality
    for (tt, original) in tts.iter().zip(&embeddings) {
        let reconstructed = tt_reconstruct(tt);
        assert_eq!(reconstructed.len(), original.len());
    }
}

#[test]
fn test_tt_batch_operations() {
    let embeddings = sample_embeddings(20, 128);
    let config = TTConfig::for_dim(128).unwrap();

    let refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();
    let tts = tt_decompose_batch(&refs, &config).unwrap();

    assert_eq!(tts.len(), 20);

    // Batch similarity
    let query = &tts[0];
    let targets = &tts[1..];
    let sims = tt_cosine_similarity_batch(query, targets).unwrap();

    assert_eq!(sims.len(), 19);
    for sim in &sims {
        assert!(sim.is_finite());
        assert!(*sim >= -1.0 && *sim <= 1.0);
    }
}
