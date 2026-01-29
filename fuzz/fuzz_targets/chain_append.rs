// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use graph_engine::GraphEngine;
use libfuzzer_sys::fuzz_target;
use std::sync::Arc;
use tensor_chain::{Block, BlockHeader, Chain, Transaction};

#[derive(Arbitrary, Debug)]
struct ChainAppendInput {
    // Number of blocks to append (limited to reasonable range)
    block_count: u8,
    // Transaction data for each block
    transactions: Vec<TransactionInput>,
    // Embeddings (as components)
    embedding_components: Vec<f32>,
}

#[derive(Arbitrary, Debug)]
struct TransactionInput {
    key: String,
    data: Vec<u8>,
}

fuzz_target!(|input: ChainAppendInput| {
    // Create a fresh chain for each test
    let store = tensor_store::TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store));
    let chain = Chain::new(graph, "fuzz_node".to_string());

    // Initialize with genesis block
    if chain.initialize().is_err() {
        return;
    }

    // Limit block count to avoid long-running tests
    let block_count = (input.block_count % 10).max(1) as usize;

    // Build embedding from components (fixed dimension)
    let dimension = 32;
    let embedding: Vec<f32> = (0..dimension)
        .map(|i| {
            let val = input
                .embedding_components
                .get(i % input.embedding_components.len().max(1))
                .copied()
                .unwrap_or(0.0);
            if val.is_finite() {
                val
            } else {
                0.0
            }
        })
        .collect();

    // Test: Sequential appends should succeed
    for i in 0..block_count {
        // Create transactions from input
        let txs: Vec<Transaction> = input
            .transactions
            .iter()
            .take(3) // Limit transactions per block
            .map(|t| {
                let key = if t.key.is_empty() {
                    format!("key_{}", i)
                } else {
                    t.key.chars().take(64).collect() // Limit key length
                };
                let data = if t.data.len() > 1024 {
                    t.data[..1024].to_vec()
                } else {
                    t.data.clone()
                };
                Transaction::Put { key, data }
            })
            .collect();

        // Build block using the chain's builder
        let mut builder = chain.new_block();
        for tx in txs {
            builder = builder.add_transaction(tx);
        }

        // Add embedding if non-zero
        if embedding.iter().any(|x| *x != 0.0) {
            builder = builder.with_dense_embedding(&embedding);
        }

        // Add signature (required for height > 1)
        let block = builder.with_signature(vec![0u8; 64]).build();

        // Append should succeed
        match chain.append(block) {
            Ok(hash) => {
                // Verify block was stored
                assert_eq!(chain.height(), (i + 1) as u64);
                assert_eq!(chain.tip_hash(), hash);

                // Verify block can be retrieved
                let stored = chain.get_block_at((i + 1) as u64).unwrap();
                assert!(stored.is_some());
            },
            Err(e) => {
                // Validation errors are acceptable for malformed input
                let _ = e;
            },
        }
    }

    // Property 1: Chain height matches number of successful appends
    let height = chain.height();
    assert!(height <= block_count as u64);

    // Property 2: All blocks in chain are retrievable
    for h in 0..=height {
        let block = chain.get_block_at(h).unwrap();
        assert!(block.is_some(), "Block at height {} should exist", h);
    }

    // Property 3: Block heights are sequential
    for h in 1..=height {
        let block = chain.get_block_at(h).unwrap().unwrap();
        assert_eq!(block.header.height, h);
    }

    // Property 4: Each block points to its predecessor
    for h in 1..=height {
        let block = chain.get_block_at(h).unwrap().unwrap();
        let prev = chain.get_block_at(h - 1).unwrap().unwrap();
        assert_eq!(block.header.prev_hash, prev.hash());
    }

    // Property 5: verify_chain passes
    assert!(chain.verify_chain().is_ok());

    // Property 6: Appending with wrong height fails
    if height > 0 {
        let wrong_height_block = Block {
            header: BlockHeader {
                height: height + 5, // Skip heights
                prev_hash: chain.tip_hash(),
                signature: vec![0u8; 64],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(chain.append(wrong_height_block).is_err());
    }

    // Property 7: Appending with wrong prev_hash fails
    if height > 0 {
        let wrong_hash_block = Block {
            header: BlockHeader {
                height: height + 1,
                prev_hash: [0xAB; 32], // Wrong hash
                signature: vec![0u8; 64],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(chain.append(wrong_hash_block).is_err());
    }
});
