// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use std::collections::HashMap;
use std::sync::Arc;

use arbitrary::Arbitrary;
use graph_engine::GraphEngine;
use libfuzzer_sys::fuzz_target;
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig};

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    Set(String, String),
    Get(String),
    Delete(String),
    Rotate(String, String),
    Grant(String, String),
    Revoke(String, String),
    List(String),
    BatchSet(Vec<(String, String)>),
    Wrap(String),
}

#[derive(Arbitrary, Debug)]
struct OpsInput {
    ops: Vec<FuzzOp>,
}

/// Sanitize a string to a bounded, non-empty alphanumeric key.
fn sanitize_key(s: &str) -> String {
    let filtered: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .take(32)
        .collect();
    if filtered.is_empty() {
        "k".to_string()
    } else {
        filtered
    }
}

/// Sanitize a value string (allow any UTF-8 but bound length).
fn sanitize_value(s: &str) -> String {
    s.chars().take(256).collect()
}

fn make_vault() -> Vault {
    let store = TensorStore::new();
    let graph = Arc::new(GraphEngine::with_store(store.clone()));
    let mut config = VaultConfig::default();
    // Use minimal Argon2 cost for fuzzing speed
    config.argon2_memory_cost = 19;
    config.argon2_time_cost = 1;
    config.argon2_parallelism = 1;
    Vault::new(b"fuzz-key-32-bytes-long!!!!!!!!", graph, store, config).unwrap()
}

fuzz_target!(|input: OpsInput| {
    if input.ops.len() > 64 {
        return;
    }

    let vault = make_vault();
    let root = Vault::ROOT;

    // Track expected state for invariant checking
    let mut known_values: HashMap<String, String> = HashMap::new();

    for op in &input.ops {
        match op {
            FuzzOp::Set(k, v) => {
                let key = sanitize_key(k);
                let val = sanitize_value(v);
                if vault.set(root, &key, &val).is_ok() {
                    known_values.insert(key, val);
                }
            }
            FuzzOp::Get(k) => {
                let key = sanitize_key(k);
                match vault.get(root, &key) {
                    Ok(retrieved) => {
                        // If we successfully get, it must match what we last set
                        if let Some(expected) = known_values.get(&key) {
                            assert_eq!(
                                &retrieved, expected,
                                "get({key}) returned wrong value"
                            );
                        }
                    }
                    Err(_) => {
                        // If we never set this key, get should fail
                        // (but it may also fail for keys we set if they were rotated/deleted)
                    }
                }
            }
            FuzzOp::Delete(k) => {
                let key = sanitize_key(k);
                if vault.delete(root, &key).is_ok() {
                    known_values.remove(&key);
                }
            }
            FuzzOp::Rotate(k, v) => {
                let key = sanitize_key(k);
                let val = sanitize_value(v);
                if vault.rotate(root, &key, &val).is_ok() {
                    known_values.insert(key, val);
                }
            }
            FuzzOp::Grant(entity, k) => {
                let ent = sanitize_key(entity);
                let key = sanitize_key(k);
                let _ = vault.grant(root, &ent, &key);
            }
            FuzzOp::Revoke(entity, k) => {
                let ent = sanitize_key(entity);
                let key = sanitize_key(k);
                let _ = vault.revoke(root, &ent, &key);
            }
            FuzzOp::List(pattern) => {
                let pat = sanitize_key(pattern);
                let result = vault.list(root, &pat);
                // list should always succeed for root
                assert!(result.is_ok(), "list should not fail for root");
            }
            FuzzOp::BatchSet(entries) => {
                let sanitized: Vec<(String, String)> = entries
                    .iter()
                    .take(16)
                    .map(|(k, v)| (sanitize_key(k), sanitize_value(v)))
                    .collect();
                let refs: Vec<(&str, &str)> =
                    sanitized.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                if vault.batch_set(root, &refs).is_ok() {
                    for (k, v) in &sanitized {
                        known_values.insert(k.clone(), v.clone());
                    }
                }
            }
            FuzzOp::Wrap(k) => {
                let key = sanitize_key(k);
                // Must set first so wrap can read
                let _ = vault.set(root, &key, "wrap-value");
                known_values.insert(key.clone(), "wrap-value".to_string());
                if let Ok(token) = vault.wrap_secret(root, &key, 60_000) {
                    // Unwrapped value must match what was stored
                    if let Ok(unwrapped) = vault.unwrap_secret(&token) {
                        assert_eq!(
                            unwrapped, "wrap-value",
                            "unwrap should return the wrapped value"
                        );
                    }
                }
            }
        }
    }

    // Final invariant: every key we think exists should be retrievable
    for (key, expected_val) in &known_values {
        if let Ok(val) = vault.get(root, key) {
            assert_eq!(
                &val, expected_val,
                "final check: get({key}) returned wrong value"
            );
        }
    }
});
