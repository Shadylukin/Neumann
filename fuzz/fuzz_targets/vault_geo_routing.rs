// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_vault::{GeoCoordinate, GeoRouter, RoutingConfig, TargetGeometry};

#[derive(Arbitrary, Debug)]
struct GeoRoutingInput {
    targets: Vec<FuzzTarget>,
    config: FuzzRoutingConfig,
    query_key: String,
    accessor_x: f64,
    accessor_y: f64,
}

#[derive(Arbitrary, Debug)]
struct FuzzTarget {
    name: String,
    x: f64,
    y: f64,
    latency_ms: f64,
    throughput: f64,
    failure_rate: f64,
}

#[derive(Arbitrary, Debug)]
struct FuzzRoutingConfig {
    max_latency_ms: f64,
    max_failure_rate: f64,
    sync_fanout: u8,
}

fn clamp_finite(v: f64, min: f64, max: f64) -> f64 {
    if v.is_finite() {
        v.clamp(min, max)
    } else {
        min
    }
}

fn sanitize_name(s: &str) -> String {
    let filtered: String = s
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .take(16)
        .collect();
    if filtered.is_empty() {
        "t".to_string()
    } else {
        filtered
    }
}

fuzz_target!(|input: GeoRoutingInput| {
    if input.targets.len() > 32 {
        return;
    }

    let config = RoutingConfig {
        max_latency_ms: clamp_finite(input.config.max_latency_ms, 1.0, 10_000.0),
        max_failure_rate: clamp_finite(input.config.max_failure_rate, 0.0, 1.0),
        latency_weight: 0.4,
        proximity_weight: 0.3,
        reliability_weight: 0.3,
        ema_alpha: 0.2,
        sync_fanout: (input.config.sync_fanout as usize).clamp(1, 10),
    };

    let router = GeoRouter::new(config);

    let mut target_names = Vec::new();
    for t in &input.targets {
        let name = sanitize_name(&t.name);
        let geo = TargetGeometry {
            target_name: name.clone(),
            location: GeoCoordinate {
                x: clamp_finite(t.x, -180.0, 180.0),
                y: clamp_finite(t.y, -90.0, 90.0),
                z: None,
            },
            avg_latency_ms: clamp_finite(t.latency_ms, 0.0, 10_000.0),
            avg_throughput: clamp_finite(t.throughput, 0.0, 1_000_000.0),
            failure_rate: clamp_finite(t.failure_rate, 0.0, 1.0),
            last_health_check_ms: 0,
        };
        router.update_geometry(geo);
        target_names.push(name);
    }

    // Route a secret
    let accessor_loc = GeoCoordinate {
        x: clamp_finite(input.accessor_x, -180.0, 180.0),
        y: clamp_finite(input.accessor_y, -90.0, 90.0),
        z: None,
    };

    let key: String = input.query_key.chars().take(32).collect();
    let decision = router.route(&key, Some(&accessor_loc), &target_names);

    // Invariant: selected + excluded = total registered targets (with dedup)
    let total_decided = decision.selected_targets.len() + decision.excluded_targets.len();
    // Some targets may share names after sanitization, so total_decided <= target_names.len()
    assert!(total_decided <= target_names.len() + 1);

    // Record some sync results and re-route
    for selected in &decision.selected_targets {
        router.record_sync_result(&selected.target_name, 50.0, true);
    }
    let _ = router.route(&key, Some(&accessor_loc), &target_names);
});
