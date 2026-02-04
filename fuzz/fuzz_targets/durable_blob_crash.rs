// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;
use tensor_store::{DurableBlobLog, DurableBlobLogConfig, DurableChunkHash};
use tempfile::tempdir;

#[derive(Arbitrary, Debug, Clone)]
enum Op {
    Put { data: Vec<u8> },
    Get { chunk_idx: u8 },
    Delete { chunk_idx: u8 },
    Reopen,
}

#[derive(Arbitrary, Debug)]
struct Input {
    ops: Vec<Op>,
}

fuzz_target!(|input: Input| {
    // Limit operations
    if input.ops.is_empty() || input.ops.len() > 50 {
        return;
    }

    // Create temp directory
    let dir = match tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };

    let config = DurableBlobLogConfig {
        segment_dir: dir.path().to_path_buf(),
        segment_size: 1024 * 1024, // 1MB for faster testing
        enable_fsync: false,       // Disable fsync for speed in fuzzing
        cache_size: 100,
    };

    let mut log = match DurableBlobLog::open(config.clone()) {
        Ok(l) => l,
        Err(_) => return,
    };

    // Track expected state (hash -> data)
    let mut expected: HashMap<DurableChunkHash, Vec<u8>> = HashMap::new();
    let mut chunk_order: Vec<DurableChunkHash> = Vec::new();

    for op in input.ops {
        match op {
            Op::Put { data } => {
                // Skip empty or too large data
                if data.is_empty() || data.len() > 64 * 1024 {
                    continue;
                }

                let hash = DurableChunkHash::from_data(&data);

                match log.append(&data) {
                    Ok(returned_hash) => {
                        // Verify hash matches expected
                        assert_eq!(
                            returned_hash, hash,
                            "Put returned different hash: expected {:?}, got {:?}",
                            hash, returned_hash
                        );

                        // Track expected data
                        if !expected.contains_key(&hash) {
                            chunk_order.push(hash);
                        }
                        expected.insert(hash, data);
                    }
                    Err(_) => {
                        // Put failures are acceptable (e.g., I/O errors)
                    }
                }
            }
            Op::Get { chunk_idx } => {
                if chunk_order.is_empty() {
                    continue;
                }

                let hash = &chunk_order[chunk_idx as usize % chunk_order.len()];

                match log.get(hash) {
                    Ok(data) => {
                        // Verify data matches expected
                        if let Some(expected_data) = expected.get(hash) {
                            assert_eq!(
                                &data, expected_data,
                                "Get returned different data for hash {:?}",
                                hash
                            );
                        }
                    }
                    Err(_) => {
                        // Get failures are acceptable for deleted chunks
                    }
                }
            }
            Op::Delete { chunk_idx } => {
                if chunk_order.is_empty() {
                    continue;
                }

                let hash = chunk_order[chunk_idx as usize % chunk_order.len()];

                match log.delete(&hash) {
                    Ok(()) => {
                        // Remove from expected
                        expected.remove(&hash);
                    }
                    Err(_) => {
                        // Delete failures are acceptable
                    }
                }
            }
            Op::Reopen => {
                // Simulate crash recovery by reopening the log
                drop(log);

                log = match DurableBlobLog::open(config.clone()) {
                    Ok(l) => l,
                    Err(_) => return,
                };

                // Verify all expected data survived
                for (hash, expected_data) in &expected {
                    match log.get(hash) {
                        Ok(data) => {
                            assert_eq!(
                                &data, expected_data,
                                "Data mismatch after reopen for hash {:?}",
                                hash
                            );
                        }
                        Err(e) => {
                            // This is a critical failure - data should survive reopen
                            panic!(
                                "Expected chunk {:?} missing after reopen: {:?}",
                                hash, e
                            );
                        }
                    }
                }
            }
        }
    }

    // Final verification: reopen and check all data
    drop(log);

    let log = match DurableBlobLog::open(config) {
        Ok(l) => l,
        Err(_) => return,
    };

    for (hash, expected_data) in &expected {
        match log.get(hash) {
            Ok(data) => {
                assert_eq!(
                    &data, expected_data,
                    "Final verification: data mismatch for hash {:?}",
                    hash
                );
            }
            Err(e) => {
                panic!(
                    "Final verification: expected chunk {:?} missing: {:?}",
                    hash, e
                );
            }
        }
    }
});
