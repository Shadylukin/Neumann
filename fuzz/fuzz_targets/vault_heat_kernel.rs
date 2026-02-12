// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use std::sync::Arc;

use arbitrary::Arbitrary;
use graph_engine::GraphEngine;
use libfuzzer_sys::fuzz_target;
use tensor_store::TensorStore;
use tensor_vault::{HeatKernelConfig, Vault, VaultConfig};

#[derive(Arbitrary, Debug)]
struct HeatKernelInput {
    /// Entities to create (clamped to a small set).
    entities: Vec<String>,
    /// Edges as (from_idx, to_idx) into the entities vec.
    edges: Vec<(u8, u8)>,
    /// Diffusion time parameter (clamped to positive finite).
    diffusion_time: f64,
    /// Chebyshev order (clamped to 1..=20).
    chebyshev_order: u8,
}

fn sanitize_entity(s: &str) -> String {
    let filtered: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .take(16)
        .collect();
    if filtered.is_empty() {
        "e".to_string()
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

fuzz_target!(|input: HeatKernelInput| {
    if input.entities.len() > 32 || input.edges.len() > 64 {
        return;
    }

    let vault = make_vault();
    let root = Vault::ROOT;

    // Create entities with secrets and grant access
    let entities: Vec<String> = input.entities.iter().map(|e| sanitize_entity(e)).collect();
    for (i, entity) in entities.iter().enumerate() {
        let secret_key = format!("s{i}");
        let _ = vault.set(root, &secret_key, "heat-value");
        let _ = vault.grant(root, entity, &secret_key);
    }

    // Create edges by granting cross-entity access
    for &(from_idx, to_idx) in &input.edges {
        let from = from_idx as usize % entities.len().max(1);
        let to = to_idx as usize % entities.len().max(1);
        if from != to && !entities.is_empty() {
            let secret_key = format!("s{to}");
            let _ = vault.grant(root, &entities[from], &secret_key);
        }
    }

    // Clamp config params
    let diffusion_time = if input.diffusion_time.is_finite() && input.diffusion_time > 0.0 {
        input.diffusion_time.min(100.0)
    } else {
        1.0
    };
    let chebyshev_order = (input.chebyshev_order as usize).clamp(1, 20);

    let config = HeatKernelConfig {
        diffusion_time,
        chebyshev_order,
        max_iterations: 50,
    };

    let report = vault.heat_kernel_trust(config);

    // Invariant: trust scores should be finite
    for score in &report.entities {
        assert!(score.trust_score.is_finite(), "trust score must be finite");
    }
});
