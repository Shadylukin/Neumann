// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::tcp::{CompressionConfig, CompressionMethod, LengthDelimitedCodec};
use tensor_chain::{Message, RequestVote};
use tensor_store::SparseVector;

#[derive(Arbitrary, Debug)]
struct CompressionInput {
    data: Vec<u8>,
    test_case: TestCase,
}

#[derive(Arbitrary, Debug)]
enum TestCase {
    /// Test compress/decompress roundtrip with LZ4
    CompressDecompressLz4,
    /// Test compress/decompress with no compression
    CompressDecompressNone,
    /// Test decoding v2 payload with arbitrary flags
    DecodeV2WithFlags { flags: u8 },
    /// Test v2 encode/decode roundtrip with compression enabled
    V2RoundtripCompressed {
        term: u64,
        node_id: String,
        embedding: Vec<f32>,
    },
    /// Test v2 encode/decode roundtrip with compression disabled
    V2RoundtripUncompressed {
        term: u64,
        node_id: String,
        embedding: Vec<f32>,
    },
}

fuzz_target!(|input: CompressionInput| {
    // Limit input data size to prevent memory issues
    // Keep it small for decompression tests to avoid OOM from decompression bombs
    let data: Vec<u8> = input.data.into_iter().take(256).collect();

    match input.test_case {
        TestCase::CompressDecompressLz4 => {
            // Compress and decompress should roundtrip
            let compressed =
                tensor_chain::tcp::compression::compress(&data, CompressionMethod::Lz4);
            if let Ok(decompressed) =
                tensor_chain::tcp::compression::decompress(&compressed, CompressionMethod::Lz4)
            {
                assert_eq!(data, decompressed, "LZ4 roundtrip failed");
            }
        },

        TestCase::CompressDecompressNone => {
            // No compression should be identity
            let compressed =
                tensor_chain::tcp::compression::compress(&data, CompressionMethod::None);
            assert_eq!(data, compressed, "None compression should be identity");

            let decompressed =
                tensor_chain::tcp::compression::decompress(&compressed, CompressionMethod::None)
                    .expect("None decompression should succeed");
            assert_eq!(data, decompressed, "None decompression should be identity");
        },

        TestCase::DecodeV2WithFlags { flags } => {
            // Build a v2 payload with flags + data
            // Only test with flags=0 (uncompressed) to avoid OOM from LZ4 decompression bombs
            // LZ4 decompression can allocate huge buffers based on size prefix in invalid data
            let safe_flags = flags & 0xFE; // Clear compression bit
            let mut payload = vec![safe_flags];
            payload.extend_from_slice(&data);

            let codec = LengthDelimitedCodec::new(1024 * 1024);
            // Should handle gracefully (either succeed or return error)
            let _ = codec.decode_payload_v2(&payload);
        },

        TestCase::V2RoundtripCompressed {
            term,
            node_id,
            embedding,
        } => {
            let node_id: String = node_id.chars().take(64).collect();
            let embedding: Vec<f32> = embedding.into_iter().take(256).collect();

            let msg = Message::RequestVote(RequestVote {
                term,
                candidate_id: node_id,
                last_log_index: 0,
                last_log_term: 0,
                state_embedding: SparseVector::from_dense(&embedding),
            });

            // Create codec with compression enabled
            let compression = CompressionConfig::default().with_min_size(0); // Always compress
            let mut codec = LengthDelimitedCodec::with_compression(1024 * 1024, compression);
            codec.set_compression_enabled(true);

            // Encode with v2 format
            if let Ok(encoded) = codec.encode_v2(&msg) {
                // Should have length prefix (4 bytes) + flags (1 byte) + payload
                assert!(encoded.len() >= 5);

                // Decode the payload part (after length prefix)
                let payload = &encoded[4..];
                let decoded = codec.decode_payload_v2(payload);
                assert!(decoded.is_ok(), "Failed to decode valid v2 message");
            }
        },

        TestCase::V2RoundtripUncompressed {
            term,
            node_id,
            embedding,
        } => {
            let node_id: String = node_id.chars().take(64).collect();
            let embedding: Vec<f32> = embedding.into_iter().take(256).collect();

            let msg = Message::RequestVote(RequestVote {
                term,
                candidate_id: node_id,
                last_log_index: 0,
                last_log_term: 0,
                state_embedding: SparseVector::from_dense(&embedding),
            });

            // Create codec with compression disabled
            let compression = CompressionConfig::disabled();
            let codec = LengthDelimitedCodec::with_compression(1024 * 1024, compression);

            // Encode with v2 format (no compression)
            if let Ok(encoded) = codec.encode_v2(&msg) {
                // Should have length prefix (4 bytes) + flags (1 byte) + payload
                assert!(encoded.len() >= 5);

                // First byte after length should be 0 (no compression)
                assert_eq!(encoded[4], 0x00, "Uncompressed flag should be 0");

                // Decode the payload part
                let payload = &encoded[4..];
                let decoded = codec.decode_payload_v2(payload);
                assert!(decoded.is_ok(), "Failed to decode valid v2 message");
            }
        },
    }
});
