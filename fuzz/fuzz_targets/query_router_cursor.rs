#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use query_router::{CursorStore, CursorStoreConfig};
use std::time::Duration;

#[derive(Arbitrary, Debug)]
enum CursorOp {
    Get { id: String },
    Remove { id: String },
    CleanupExpired,
    Len,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<CursorOp>,
}

fuzz_target!(|input: FuzzInput| {
    if input.ops.len() > 100 {
        return;
    }

    let config = CursorStoreConfig::new()
        .with_max_cursors(64)
        .with_default_ttl(Duration::from_secs(10))
        .with_max_ttl(Duration::from_secs(30));
    let store = CursorStore::with_config(config);

    for op in &input.ops {
        match op {
            CursorOp::Get { id } => {
                if id.len() <= 128 {
                    let _ = store.get(id);
                }
            },
            CursorOp::Remove { id } => {
                if id.len() <= 128 {
                    let _ = store.remove(id);
                }
            },
            CursorOp::CleanupExpired => {
                let _ = store.cleanup_expired();
            },
            CursorOp::Len => {
                let _ = store.len();
                let _ = store.is_empty();
            },
        }
    }
});
