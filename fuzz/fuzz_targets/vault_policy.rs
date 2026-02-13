// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use std::sync::Arc;

use arbitrary::Arbitrary;
use graph_engine::GraphEngine;
use libfuzzer_sys::fuzz_target;
use tensor_store::TensorStore;
use tensor_vault::{Permission, PolicyTemplate, Vault, VaultConfig};

#[derive(Arbitrary, Debug)]
struct PolicyInput {
    templates: Vec<FuzzTemplate>,
    entities_to_evaluate: Vec<String>,
}

#[derive(Arbitrary, Debug)]
struct FuzzTemplate {
    name: String,
    match_pattern: String,
    secret_pattern: String,
    permission: u8,
    ttl_ms: Option<i64>,
}

fn sanitize(s: &str, max_len: usize) -> String {
    let filtered: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '*' || *c == '/' || *c == ':')
        .take(max_len)
        .collect();
    if filtered.is_empty() {
        "x".to_string()
    } else {
        filtered
    }
}

fn permission_from_u8(v: u8) -> Permission {
    match v % 3 {
        0 => Permission::Read,
        1 => Permission::Write,
        _ => Permission::Admin,
    }
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

fuzz_target!(|input: PolicyInput| {
    if input.templates.len() > 16 || input.entities_to_evaluate.len() > 16 {
        return;
    }

    let vault = make_vault();
    let root = Vault::ROOT;

    // Add policy templates
    let mut added_names = Vec::new();
    for ft in &input.templates {
        let name = sanitize(&ft.name, 32);
        let template = PolicyTemplate {
            name: name.clone(),
            match_pattern: sanitize(&ft.match_pattern, 64),
            secret_pattern: sanitize(&ft.secret_pattern, 64),
            permission: permission_from_u8(ft.permission),
            ttl_ms: ft.ttl_ms.map(|t| t.clamp(0, 86_400_000)),
        };
        if vault.add_policy(root, template).is_ok() {
            added_names.push(name);
        }
    }

    // List policies
    let policies = vault.list_policies();
    assert!(
        policies.len() <= input.templates.len(),
        "cannot have more policies than templates added"
    );

    // Evaluate policies for entities
    for entity in &input.entities_to_evaluate {
        let ent = sanitize(entity, 64);
        let matches = vault.evaluate_policies(&ent);
        // Each match must reference a known policy name
        for m in &matches {
            assert!(
                added_names.contains(&m.policy_name),
                "matched policy must exist"
            );
        }
    }

    // Remove some policies
    for name in &added_names {
        let _ = vault.remove_policy(root, name);
    }

    // After removal, list should be empty
    let remaining = vault.list_policies();
    assert!(remaining.is_empty(), "all policies should be removed");
});
