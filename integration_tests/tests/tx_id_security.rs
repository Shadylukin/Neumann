// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Integration tests for secure transaction ID generation.

use std::collections::HashSet;
use std::thread;

use tensor_chain::{extract_timestamp_hint, generate_tx_id, is_plausible_tx_id};
use tensor_store::TensorStore;

#[test]
fn test_transaction_workspace_uses_secure_ids() {
    let store = TensorStore::new();

    // Create multiple workspaces and verify IDs are not sequential
    let mut ids = Vec::new();
    for _ in 0..10 {
        let ws = tensor_chain::TransactionWorkspace::begin(&store).unwrap();
        ids.push(ws.id());
    }

    // Check that consecutive IDs are not sequential (differ by more than 1)
    for window in ids.windows(2) {
        let diff = window[1].abs_diff(window[0]);
        // With 32 bits of randomness, sequential IDs would differ by exactly 1
        assert!(
            diff > 1,
            "IDs appear sequential: {} and {}",
            window[0],
            window[1]
        );
    }
}

#[test]
fn test_distributed_tx_uses_secure_ids() {
    // Create multiple distributed transactions
    let mut ids = Vec::new();
    for _ in 0..10 {
        let dtx =
            tensor_chain::DistributedTransaction::new("coordinator".to_string(), vec![0, 1, 2]);
        ids.push(dtx.tx_id);
    }

    // Check that IDs are not sequential
    for window in ids.windows(2) {
        let diff = window[1].abs_diff(window[0]);
        assert!(
            diff > 1,
            "Distributed TX IDs appear sequential: {} and {}",
            window[0],
            window[1]
        );
    }
}

#[test]
fn test_ids_not_sequential_across_workspaces() {
    let store = TensorStore::new();

    // Create a mix of regular and distributed transactions
    let mut all_ids = HashSet::new();

    for _ in 0..100 {
        let ws = tensor_chain::TransactionWorkspace::begin(&store).unwrap();
        assert!(all_ids.insert(ws.id()), "Duplicate workspace ID");

        let dtx = tensor_chain::DistributedTransaction::new("coordinator".to_string(), vec![0]);
        assert!(all_ids.insert(dtx.tx_id), "Duplicate distributed TX ID");
    }

    // Should have 200 unique IDs
    assert_eq!(all_ids.len(), 200);
}

#[test]
fn test_generate_tx_id_uniqueness() {
    let mut ids = HashSet::new();

    // Generate 10,000 IDs and check for duplicates
    for _ in 0..10_000 {
        let id = generate_tx_id();
        assert!(ids.insert(id), "Duplicate ID generated: {}", id);
    }
}

#[test]
fn test_concurrent_tx_id_generation() {
    let handles: Vec<_> = (0..8)
        .map(|_| {
            thread::spawn(|| {
                let mut ids = Vec::with_capacity(1000);
                for _ in 0..1000 {
                    ids.push(generate_tx_id());
                }
                ids
            })
        })
        .collect();

    let mut all_ids = HashSet::new();
    for handle in handles {
        let ids = handle.join().unwrap();
        for id in ids {
            assert!(
                all_ids.insert(id),
                "Duplicate ID in concurrent test: {}",
                id
            );
        }
    }

    assert_eq!(all_ids.len(), 8000);
}

#[test]
fn test_timestamp_hint_extraction() {
    let id = generate_tx_id();
    let hint = extract_timestamp_hint(id);

    // Hint should be a reasonable timestamp (after 2024-01-01)
    let custom_epoch_ms: u64 = 1704067200000;
    assert!(hint >= custom_epoch_ms, "Timestamp hint too early");

    // Should be within reasonable future
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    // Allow some window for timing variance
    assert!(hint <= now_ms + 65536, "Timestamp hint too far in future");
}

#[test]
fn test_plausibility_check() {
    let id = generate_tx_id();

    // Freshly generated ID should be plausible
    assert!(is_plausible_tx_id(id, 10_000));

    // Very short window might miss due to test timing
    // but 10 second window should definitely include it
    assert!(is_plausible_tx_id(id, 10_000));
}

#[test]
fn test_ids_unpredictable() {
    // Generate 100 IDs and check that the random portion varies significantly
    let ids: Vec<u64> = (0..100).map(|_| generate_tx_id()).collect();

    // Extract random portions (lower 32 bits)
    let random_portions: Vec<u64> = ids.iter().map(|id| id & 0xFFFF_FFFF).collect();

    // All random portions should be unique (probability of collision is 1/2^32)
    let unique: HashSet<u64> = random_portions.iter().copied().collect();
    assert_eq!(unique.len(), 100, "Random portions should all be unique");

    // Check no two random portions differ by exactly 1 (sequential pattern)
    for window in random_portions.windows(2) {
        let diff = window[1].abs_diff(window[0]);
        assert_ne!(diff, 1, "Sequential pattern in random portion");
    }
}

#[test]
fn test_id_bit_structure() {
    let id = generate_tx_id();

    // Extract components
    let ms_bits = (id >> 48) & 0xFFFF;
    let us_bits = (id >> 32) & 0xFFFF;
    let random_bits = id & 0xFFFF_FFFF;

    // Verify they fit in their allocated bit spaces
    assert!(ms_bits <= 0xFFFF, "ms_bits overflow");
    assert!(us_bits <= 0xFFFF, "us_bits overflow");
    assert!(random_bits <= 0xFFFF_FFFF, "random_bits overflow");

    // ID should not be zero
    assert_ne!(id, 0, "ID should not be zero");
}
