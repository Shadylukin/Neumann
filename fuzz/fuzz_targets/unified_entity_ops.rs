#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use query_router::QueryRouter;
use tensor_store::TensorStore;

#[derive(Arbitrary, Debug)]
enum EntityOp {
    Create {
        key: String,
        field_name: String,
        field_value: String,
        embedding_dim: u8,
    },
    Get {
        key_idx: u8,
    },
    Delete {
        key_idx: u8,
    },
    Update {
        key_idx: u8,
        field_name: String,
        field_value: String,
    },
    FindSimilar {
        key_idx: u8,
        top_k: u8,
    },
    List,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<EntityOp>,
}

fuzz_target!(|input: FuzzInput| {
    if input.ops.len() > 50 {
        return;
    }

    let store = TensorStore::new();
    let router = QueryRouter::with_shared_store(store);
    let mut created_keys = Vec::new();

    for op in &input.ops {
        match op {
            EntityOp::Create {
                key,
                field_name,
                field_value,
                embedding_dim,
            } => {
                if key.is_empty() || key.len() > 64 || field_name.len() > 64 {
                    continue;
                }
                // Skip keys/fields with characters that break query parsing
                if key.contains('\'')
                    || key.contains('{')
                    || key.contains('}')
                    || field_name.contains(':')
                    || field_name.contains('{')
                {
                    continue;
                }

                let dim = (*embedding_dim as usize).clamp(1, 16);
                let emb: Vec<String> = (0..dim).map(|i| format!("{:.4}", (i as f32).sin())).collect();
                let emb_str = emb.join(", ");

                let query = format!(
                    "ENTITY CREATE '{}' {{ {}: '{}' }} EMBEDDING [{}]",
                    key, field_name, field_value, emb_str
                );
                if router.execute_parsed(&query).is_ok() {
                    created_keys.push(key.clone());
                }
            },
            EntityOp::Get { key_idx } => {
                if !created_keys.is_empty() {
                    let idx = (*key_idx as usize) % created_keys.len();
                    let _ = router.execute_parsed(&format!("ENTITY GET '{}'", created_keys[idx]));
                }
            },
            EntityOp::Delete { key_idx } => {
                if !created_keys.is_empty() {
                    let idx = (*key_idx as usize) % created_keys.len();
                    let _ = router.execute_parsed(&format!("ENTITY DELETE '{}'", created_keys[idx]));
                }
            },
            EntityOp::Update {
                key_idx,
                field_name,
                field_value,
            } => {
                if !created_keys.is_empty() && !field_name.is_empty() && field_name.len() <= 64 {
                    if field_name.contains(':') || field_name.contains('{') {
                        continue;
                    }
                    let idx = (*key_idx as usize) % created_keys.len();
                    let _ = router.execute_parsed(&format!(
                        "ENTITY UPDATE '{}' {{ {}: '{}' }}",
                        created_keys[idx], field_name, field_value
                    ));
                }
            },
            EntityOp::FindSimilar { key_idx, top_k } => {
                if !created_keys.is_empty() {
                    let idx = (*key_idx as usize) % created_keys.len();
                    let k = (*top_k as usize).clamp(1, 20);
                    let _ = router.execute_parsed(&format!(
                        "FIND SIMILAR TO '{}' TOP {}",
                        created_keys[idx], k
                    ));
                }
            },
            EntityOp::List => {
                let _ = router.execute_parsed("ENTITY LIST");
            },
        }
    }
});
