#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_store::EntityIndex;

#[derive(Arbitrary, Debug)]
enum EntityIndexOp {
    GetOrCreate(String),
    Get(String),
    Contains(String),
    Remove(String),
    KeyFor(u32),
    ScanPrefix(String),
}

#[derive(Arbitrary, Debug)]
struct EntityIndexInput {
    ops: Vec<EntityIndexOp>,
}

fuzz_target!(|input: EntityIndexInput| {
    let index = EntityIndex::new();

    for op in input.ops.into_iter().take(1000) {
        match op {
            EntityIndexOp::GetOrCreate(key) => {
                if !key.is_empty() && key.len() < 256 {
                    let _ = index.get_or_create(&key);
                }
            }
            EntityIndexOp::Get(key) => {
                if !key.is_empty() && key.len() < 256 {
                    let _ = index.get(&key);
                }
            }
            EntityIndexOp::Contains(key) => {
                if !key.is_empty() && key.len() < 256 {
                    let _ = index.contains(&key);
                }
            }
            EntityIndexOp::Remove(key) => {
                if !key.is_empty() && key.len() < 256 {
                    let _ = index.remove(&key);
                }
            }
            EntityIndexOp::KeyFor(id) => {
                let entity_id = tensor_store::EntityId(u64::from(id));
                let _ = index.key_for(entity_id);
            }
            EntityIndexOp::ScanPrefix(prefix) => {
                if prefix.len() < 64 {
                    let _ = index.scan_prefix(&prefix);
                }
            }
        }
    }

    // Verify consistency
    let count = index.len();
    let total = index.total_entries();
    assert!(count <= total, "Live count {} > total {}", count, total);
});
