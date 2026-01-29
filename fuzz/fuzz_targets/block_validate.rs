// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{TensorChain, Transaction};
use tensor_store::TensorStore;

#[derive(Arbitrary, Debug)]
struct BlockValidateInput {
    // Transactions to include
    transactions: Vec<TransactionInput>,
    // Whether to initialize chain first
    initialize: bool,
    // Block height manipulation attempts
    #[allow(dead_code)]
    height_offset: i8,
    // Previous hash manipulation
    #[allow(dead_code)]
    corrupt_prev_hash: bool,
}

#[derive(Arbitrary, Debug)]
struct TransactionInput {
    tx_type: u8,
    key: String,
    data: Vec<u8>,
    vector: Vec<f32>,
    label: String,
}

fuzz_target!(|input: BlockValidateInput| {
    // Create a fresh store and chain
    let store = TensorStore::new();
    let chain = TensorChain::new(store, "fuzz_node");

    // Initialize if requested
    if input.initialize {
        if chain.initialize().is_err() {
            return; // Skip if initialization fails
        }
    }

    // Convert input transactions
    let transactions: Vec<Transaction> = input
        .transactions
        .iter()
        .take(100) // Limit transactions
        .filter_map(|tx| {
            if tx.key.is_empty() || tx.key.len() > 256 {
                return None;
            }
            if tx.data.len() > 1024 {
                return None;
            }

            match tx.tx_type % 5 {
                0 => Some(Transaction::Put {
                    key: tx.key.clone(),
                    data: tx.data.clone(),
                }),
                1 => Some(Transaction::Delete {
                    key: tx.key.clone(),
                }),
                2 => {
                    // Embed - limit vector size
                    let vector: Vec<f32> = tx.vector.iter().take(128).copied().collect();
                    if vector.is_empty() {
                        return None;
                    }
                    Some(Transaction::Embed {
                        key: tx.key.clone(),
                        vector,
                    })
                }
                3 => Some(Transaction::NodeCreate {
                    key: tx.key.clone(),
                    label: tx.label.clone(),
                }),
                4 => Some(Transaction::NodeDelete {
                    key: tx.key.clone(),
                }),
                _ => None,
            }
        })
        .collect();

    if transactions.is_empty() && input.initialize {
        // Property 1: Empty transaction commits don't create blocks
        let tx = chain.begin();
        if let Ok(tx) = tx {
            let result = chain.commit(tx);
            // Should succeed but not create a new block
            if result.is_ok() {
                // Height should still be 0 (only genesis)
                assert_eq!(chain.height(), 0);
            }
        }
        return;
    }

    if !input.initialize {
        // Property 2: Operations before initialization should handle gracefully
        let tx = chain.begin();
        if let Ok(tx) = tx {
            for transaction in transactions.iter().take(5) {
                let _ = tx.add_operation(transaction.clone());
            }
            // Commit might fail or succeed depending on initialization state
            let _ = chain.commit(tx);
        }
        return;
    }

    // Property 3: Normal transaction flow
    if !transactions.is_empty() {
        let tx = chain.begin();
        if let Ok(tx) = tx {
            for transaction in &transactions {
                let _ = tx.add_operation(transaction.clone());
            }
            let commit_result = chain.commit(tx);
            if commit_result.is_ok() {
                // Height should have increased
                assert!(chain.height() >= 1);
            }
        }
    }

    // Property 4: Chain verification should not panic
    let _ = chain.verify();

    // Property 5: Block retrieval should be safe
    let height = chain.height();
    for h in 0..=height {
        let block = chain.get_block(h);
        // Should not panic
        let _ = block;
    }

    // Property 6: History queries should be safe
    for tx in transactions.iter().take(10) {
        match tx {
            Transaction::Put { key, .. }
            | Transaction::Delete { key }
            | Transaction::Embed { key, .. }
            | Transaction::NodeCreate { key, .. }
            | Transaction::NodeDelete { key } => {
                let _ = chain.history(key);
            },
            _ => {},
        }
    }

    // Property 7: Iteration should be safe
    let mut count = 0;
    for _block in chain.iter() {
        count += 1;
        if count > 1000 {
            break; // Safety limit
        }
    }

    // Property 8: Multiple transactions in sequence
    for batch in 0..3 {
        if let Ok(tx) = chain.begin() {
            tx.add_operation(Transaction::Put {
                key: format!("batch_{}", batch),
                data: vec![batch as u8],
            })
            .ok();
            let _ = chain.commit(tx);
        }
    }

    // Property 9: Rollback should be safe
    if let Ok(tx) = chain.begin() {
        tx.add_operation(Transaction::Put {
            key: "rollback_test".to_string(),
            data: vec![1, 2, 3],
        })
        .ok();
        let _ = chain.rollback(tx);
    }

    // Property 10: Concurrent begin should be safe
    let tx1 = chain.begin();
    let tx2 = chain.begin();
    if let (Ok(tx1), Ok(tx2)) = (tx1, tx2) {
        tx1.add_operation(Transaction::Put {
            key: "concurrent1".to_string(),
            data: vec![1],
        })
        .ok();
        tx2.add_operation(Transaction::Put {
            key: "concurrent2".to_string(),
            data: vec![2],
        })
        .ok();
        let _ = chain.commit(tx1);
        let _ = chain.commit(tx2);
    }

    // Property 11: Store access should be safe
    let _ = chain.store();
    let _ = chain.graph();
});
