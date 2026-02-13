// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use std::sync::Arc;

use arbitrary::Arbitrary;
use graph_engine::GraphEngine;
use libfuzzer_sys::fuzz_target;
use tensor_store::TensorStore;
use tensor_vault::{AccessTensorConfig, Vault, VaultConfig};

#[derive(Arbitrary, Debug)]
struct AccessTensorInput {
    /// Operations to perform before building tensor.
    ops: Vec<TensorOp>,
    /// Bucket size in ms (clamped).
    bucket_size_ms: i64,
    /// Number of buckets (clamped).
    num_buckets: u8,
}

#[derive(Arbitrary, Debug)]
enum TensorOp {
    Set(String, String),
    Get(String),
    Delete(String),
    Rotate(String, String),
}

fn sanitize_key(s: &str) -> String {
    let filtered: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .take(16)
        .collect();
    if filtered.is_empty() {
        "k".to_string()
    } else {
        filtered
    }
}

fn sanitize_value(s: &str) -> String {
    s.chars().take(64).collect()
}

fn make_vault() -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let mut config = VaultConfig::default();
    config.argon2_memory_cost = 19;
    config.argon2_time_cost = 1;
    config.argon2_parallelism = 1;
    Vault::new(b"fuzz-key-32-bytes-long!!!!!!!!", graph, store, config).unwrap()
}

fuzz_target!(|input: AccessTensorInput| {
    if input.ops.len() > 32 {
        return;
    }

    let vault = make_vault();
    let root = Vault::ROOT;

    // Perform operations to populate audit log
    for op in &input.ops {
        match op {
            TensorOp::Set(k, v) => {
                let _ = vault.set(root, &sanitize_key(k), &sanitize_value(v));
            }
            TensorOp::Get(k) => {
                let _ = vault.get(root, &sanitize_key(k));
            }
            TensorOp::Delete(k) => {
                let _ = vault.delete(root, &sanitize_key(k));
            }
            TensorOp::Rotate(k, v) => {
                let _ = vault.rotate(root, &sanitize_key(k), &sanitize_value(v));
            }
        }
    }

    // Clamp config
    let bucket_size = input.bucket_size_ms.clamp(1_000, 3_600_000);
    let num_buckets = (input.num_buckets as usize).clamp(1, 24);

    let config = AccessTensorConfig {
        bucket_size_ms: bucket_size,
        num_buckets,
        start_time_ms: None,
        operations: None,
    };

    if let Ok(tensor) = vault.build_access_tensor(config) {
        let (_n_entities, _n_secrets, n_buckets) = tensor.dimensions();
        // Invariant: dimensions must be consistent
        assert!(n_buckets <= 24, "bucket count within configured range");
        // Profiles should not panic
        let _ = tensor.entity_profiles();
        let _ = tensor.secret_profiles();
    }
});
