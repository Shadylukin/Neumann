// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::{EntityIndex, EntityIndexConfig};

#[derive(Arbitrary, Debug)]
enum Operation {
    GetOrCreate(String),
    Get(String),
    Remove(String),
    Contains(String),
    ScanPrefix(String),
}

#[derive(Arbitrary, Debug)]
struct EntityIndexInput {
    max_entities: u8,
    operations: Vec<Operation>,
}

fuzz_target!(|input: EntityIndexInput| {
    let max = (input.max_entities as usize).clamp(10, 1000);
    let config = EntityIndexConfig::with_max_entities(max);
    let index = EntityIndex::with_config(config);

    for op in input.operations.iter().take(100) {
        match op {
            Operation::GetOrCreate(key) => {
                // Limit key length to avoid excessive memory usage
                let key = if key.len() > 100 { &key[..100] } else { key };
                let _ = index.try_get_or_create(key);
            }
            Operation::Get(key) => {
                let key = if key.len() > 100 { &key[..100] } else { key };
                let _ = index.get(key);
            }
            Operation::Remove(key) => {
                let key = if key.len() > 100 { &key[..100] } else { key };
                let _ = index.remove(key);
            }
            Operation::Contains(key) => {
                let key = if key.len() > 100 { &key[..100] } else { key };
                let _ = index.contains(key);
            }
            Operation::ScanPrefix(prefix) => {
                let prefix = if prefix.len() > 50 { &prefix[..50] } else { prefix };
                let _ = index.scan_prefix(prefix);
            }
        }
    }

    // Verify invariants after all operations
    let len = index.len();
    assert!(len <= max, "Index exceeded max_entities");

    // Verify scan returns consistent results
    let all_entities = index.scan_prefix("");
    assert!(
        all_entities.len() <= len,
        "Scan returned more entities than len()"
    );
});
