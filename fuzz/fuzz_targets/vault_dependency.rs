// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use std::sync::Arc;

use arbitrary::Arbitrary;
use graph_engine::GraphEngine;
use libfuzzer_sys::fuzz_target;
use tensor_store::TensorStore;
use tensor_vault::{Vault, VaultConfig};

#[derive(Arbitrary, Debug)]
enum DepOp {
    AddDependency(String, String),
    RemoveDependency(String, String),
    GetDependencies(String),
    GetDependents(String),
    ImpactAnalysis(String),
}

#[derive(Arbitrary, Debug)]
struct DependencyInput {
    /// Secrets to pre-create.
    secrets: Vec<String>,
    /// Operations to perform.
    ops: Vec<DepOp>,
}

fn sanitize_key(s: &str) -> String {
    let filtered: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .take(16)
        .collect();
    if filtered.is_empty() {
        "d".to_string()
    } else {
        filtered
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

fuzz_target!(|input: DependencyInput| {
    if input.secrets.len() > 16 || input.ops.len() > 32 {
        return;
    }

    let vault = make_vault();
    let root = Vault::ROOT;

    // Pre-create secrets
    let secrets: Vec<String> = input.secrets.iter().map(|s| sanitize_key(s)).collect();
    for secret in &secrets {
        let _ = vault.set(root, secret, "dep-value");
    }

    for op in &input.ops {
        match op {
            DepOp::AddDependency(parent, child) => {
                let p = sanitize_key(parent);
                let c = sanitize_key(child);
                let _ = vault.add_dependency(root, &p, &c);
            }
            DepOp::RemoveDependency(parent, child) => {
                let p = sanitize_key(parent);
                let c = sanitize_key(child);
                let _ = vault.remove_dependency(root, &p, &c);
            }
            DepOp::GetDependencies(key) => {
                let k = sanitize_key(key);
                let _ = vault.get_dependencies(root, &k);
            }
            DepOp::GetDependents(key) => {
                let k = sanitize_key(key);
                let _ = vault.get_dependents(root, &k);
            }
            DepOp::ImpactAnalysis(key) => {
                let k = sanitize_key(key);
                if let Ok(report) = vault.impact_analysis(root, &k) {
                    // Invariant: root secret must match
                    assert!(
                        !report.root_secret.is_empty(),
                        "impact report must have a root secret"
                    );
                }
            }
        }
    }
});
