// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use std::sync::Arc;

use arbitrary::Arbitrary;
use graph_engine::GraphEngine;
use libfuzzer_sys::fuzz_target;
use tensor_store::TensorStore;
use tensor_vault::Vault;
use tensor_vault::VaultConfig;

#[derive(Arbitrary, Debug)]
enum AuditOp {
    Set(String, String),
    Get(String),
    Delete(String),
    Rotate(String, String),
    Grant(String, String),
}

#[derive(Arbitrary, Debug)]
struct AuditInput {
    ops: Vec<AuditOp>,
    query_key: String,
    query_entity: String,
}

fn sanitize_key(s: &str) -> String {
    let filtered: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .take(16)
        .collect();
    if filtered.is_empty() {
        "a".to_string()
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

fuzz_target!(|input: AuditInput| {
    if input.ops.len() > 32 {
        return;
    }

    let vault = make_vault();
    let root = Vault::ROOT;
    let mut set_keys: Vec<String> = Vec::new();

    // Perform operations to generate audit entries
    for op in &input.ops {
        match op {
            AuditOp::Set(k, v) => {
                let key = sanitize_key(k);
                let val = sanitize_value(v);
                if vault.set(root, &key, &val).is_ok() {
                    set_keys.push(key);
                }
            }
            AuditOp::Get(k) => {
                let key = sanitize_key(k);
                let _ = vault.get(root, &key);
            }
            AuditOp::Delete(k) => {
                let key = sanitize_key(k);
                let _ = vault.delete(root, &key);
            }
            AuditOp::Rotate(k, v) => {
                let key = sanitize_key(k);
                let val = sanitize_value(v);
                let _ = vault.rotate(root, &key, &val);
            }
            AuditOp::Grant(entity, k) => {
                let ent = sanitize_key(entity);
                let key = sanitize_key(k);
                let _ = vault.grant(root, &ent, &key);
            }
        }
    }

    // Query audit log by key
    let query_k = sanitize_key(&input.query_key);
    if let Ok(entries) = vault.audit_log(&query_k) {
        // Invariants on audit entries
        for entry in &entries {
            assert!(entry.timestamp > 0, "timestamp must be positive");
        }
    }

    // Query audit log by entity
    let query_e = sanitize_key(&input.query_entity);
    if let Ok(entries) = vault.audit_by_entity(&query_e) {
        for entry in &entries {
            assert!(entry.timestamp > 0, "timestamp must be positive");
        }
    }

    // Query audit since epoch -- should return all entries
    if let Ok(all_entries) = vault.audit_since(0) {
        // At least as many entries as successful set operations
        // (each set generates one audit entry)
        assert!(
            all_entries.len() >= set_keys.len(),
            "audit log must contain at least one entry per successful set"
        );
    }
});
