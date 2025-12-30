//! TensorCompress integration tests.
//!
//! Tests compression algorithms with real engine data.

use integration_tests::{create_shared_engines, sample_embeddings};
use tensor_compress::{
    compress_ids, decompress_ids, delta_decode, delta_encode, dequantize_binary, dequantize_int8,
    quantize_binary, quantize_int8, rle_decode, rle_encode, CompressionConfig, QuantMode,
};

#[test]
fn test_quantize_int8_with_embeddings() {
    let (_, _, _, vector) = create_shared_engines();

    // Store embeddings
    let embeddings = sample_embeddings(10, 128);
    for (i, emb) in embeddings.iter().enumerate() {
        vector
            .store_embedding(&format!("doc:{}", i), emb.clone())
            .unwrap();
    }

    // Quantize each embedding and verify roundtrip
    for (i, original) in embeddings.iter().enumerate() {
        let quantized = quantize_int8(original).unwrap();
        let restored = dequantize_int8(&quantized);

        // Verify compression ratio (~4x for int8)
        let original_bytes = original.len() * 4;
        let quantized_bytes = quantized.data.len() + 8; // data + min + scale
        let ratio = original_bytes as f64 / quantized_bytes as f64;
        assert!(
            ratio > 3.5,
            "doc:{} - Expected >3.5x compression, got {:.2}x",
            i,
            ratio
        );

        // Verify accuracy (max ~1% error for int8)
        for (orig, rest) in original.iter().zip(&restored) {
            let range = original.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let error = (orig - rest).abs() / range.max(1.0);
            assert!(
                error < 0.02,
                "doc:{} - Error too large: {:.4} vs {:.4}",
                i,
                orig,
                rest
            );
        }
    }
}

#[test]
fn test_quantize_binary_with_embeddings() {
    let embeddings = sample_embeddings(10, 128);

    for (i, original) in embeddings.iter().enumerate() {
        let quantized = quantize_binary(original);
        let restored = dequantize_binary(&quantized).unwrap();

        // Verify compression ratio (~32x for binary)
        let original_bytes = original.len() * 4;
        let quantized_bytes = quantized.data.len() + 8;
        let ratio = original_bytes as f64 / quantized_bytes as f64;
        assert!(
            ratio > 20.0,
            "doc:{} - Expected >20x compression, got {:.2}x",
            i,
            ratio
        );

        // Binary only preserves sign
        for (orig, rest) in original.iter().zip(&restored) {
            let expected = if *orig > 0.0 { 1.0 } else { -1.0 };
            assert_eq!(*rest, expected, "doc:{} - Sign mismatch", i);
        }
    }
}

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
        vector_quantization: Some(QuantMode::Int8),
        delta_encoding: true,
        rle_encoding: true,
    };

    // Verify serialization
    let bytes = bincode::serialize(&config).unwrap();
    let decoded: CompressionConfig = bincode::deserialize(&bytes).unwrap();
    assert_eq!(config, decoded);
}

#[test]
fn test_int8_quantization_preserves_similarity_order() {
    let embeddings = sample_embeddings(5, 64);

    // Quantize all embeddings
    let quantized: Vec<_> = embeddings
        .iter()
        .map(|e| quantize_int8(e).unwrap())
        .collect();
    let restored: Vec<Vec<f32>> = quantized.iter().map(dequantize_int8).collect();

    // Compute cosine similarities in original space
    let query = &embeddings[0];
    let mut original_sims: Vec<(usize, f32)> = embeddings[1..]
        .iter()
        .enumerate()
        .map(|(i, e)| (i + 1, cosine_similarity(query, e)))
        .collect();
    original_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Compute cosine similarities in quantized space
    let restored_query = &restored[0];
    let mut restored_sims: Vec<(usize, f32)> = restored[1..]
        .iter()
        .enumerate()
        .map(|(i, e)| (i + 1, cosine_similarity(restored_query, e)))
        .collect();
    restored_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Order should be preserved (most similar should still be most similar)
    assert_eq!(
        original_sims[0].0, restored_sims[0].0,
        "Most similar embedding changed after quantization"
    );
}

#[test]
fn test_binary_quantization_for_fast_filtering() {
    let embeddings = sample_embeddings(100, 128);

    // Quantize to binary
    let binary: Vec<_> = embeddings.iter().map(|e| quantize_binary(e)).collect();

    // Binary vectors can be used for fast hamming distance filtering
    let query_binary = &binary[0];

    // Count matching bits (simple hamming similarity)
    let mut similarities: Vec<(usize, usize)> = binary[1..]
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let matching = query_binary
                .data
                .iter()
                .zip(&b.data)
                .map(|(a, b)| (a ^ b).count_zeros() as usize)
                .sum();
            (i + 1, matching)
        })
        .collect();

    similarities.sort_by(|a, b| b.1.cmp(&a.1));

    // Just verify we get valid results
    assert!(!similarities.is_empty());
    assert!(similarities[0].1 > 0);
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

    // 3. Int8 quantization on embeddings
    for emb in &embeddings {
        let quantized = quantize_int8(emb).unwrap();
        let restored = dequantize_int8(&quantized);
        assert_eq!(restored.len(), emb.len());
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}
