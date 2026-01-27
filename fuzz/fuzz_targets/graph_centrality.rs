#![no_main]

//! Fuzz target for graph centrality algorithms.
//!
//! Tests pagerank, betweenness_centrality, closeness_centrality,
//! eigenvector_centrality with various configurations.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use graph_engine::{
    CentralityConfig, CommunityConfig, Direction, GraphEngine, PageRankConfig, PropertyValue,
};
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
struct FuzzPageRankConfig {
    damping: u8,
    max_iterations: u8,
    tolerance_exp: u8,
    direction: u8,
}

#[derive(Arbitrary, Debug)]
struct FuzzCentralityConfig {
    direction: u8,
    sampling_ratio: u8,
    max_iterations: u8,
}

#[derive(Arbitrary, Debug)]
struct FuzzCommunityConfig {
    resolution: u8,
    max_passes: u8,
    max_iterations: u8,
    direction: u8,
    seed: Option<u64>,
}

#[derive(Arbitrary, Debug)]
enum CentralityOp {
    PageRank(FuzzPageRankConfig),
    BetweennessCentrality(FuzzCentralityConfig),
    ClosenessCentrality(FuzzCentralityConfig),
    EigenvectorCentrality(FuzzCentralityConfig),
    LouvainCommunities(FuzzCommunityConfig),
    LabelPropagation(FuzzCommunityConfig),
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    node_count: u8,
    edge_density: u8,
    ops: Vec<CentralityOp>,
}

fn direction_from_u8(d: u8) -> Direction {
    match d % 3 {
        0 => Direction::Outgoing,
        1 => Direction::Incoming,
        _ => Direction::Both,
    }
}

fn build_random_graph(engine: &GraphEngine, node_count: u8, edge_density: u8) -> Vec<u64> {
    let n = (node_count as usize).clamp(5, 50);
    let density = (edge_density as usize).clamp(10, 150);

    let mut node_ids = Vec::with_capacity(n);
    for i in 0..n {
        let id = engine
            .create_node(
                "Node",
                HashMap::from([("idx".to_string(), PropertyValue::Int(i as i64))]),
            )
            .unwrap();
        node_ids.push(id);
    }

    // Create edges with some structure for interesting centrality results
    for i in 0..density {
        let from_idx = i % n;
        let to_idx = (i * 7 + 3) % n;
        if from_idx != to_idx {
            let _ = engine.create_edge(
                node_ids[from_idx],
                node_ids[to_idx],
                "LINK",
                HashMap::new(),
                true,
            );
        }
    }

    // Add some hub structure
    if n > 3 {
        let hub = node_ids[0];
        for &spoke in node_ids.iter().take(n.min(10)).skip(1) {
            let _ = engine.create_edge(hub, spoke, "HUB", HashMap::new(), true);
        }
    }

    node_ids
}

fuzz_target!(|input: FuzzInput| {
    let engine = GraphEngine::new();
    let _node_ids = build_random_graph(&engine, input.node_count, input.edge_density);

    for op in input.ops.into_iter().take(20) {
        match op {
            CentralityOp::PageRank(config) => {
                let damping = 0.5 + (config.damping as f64 / 255.0) * 0.4;
                let max_iter = (config.max_iterations as usize).clamp(10, 100);
                let tolerance = 10f64.powi(-(config.tolerance_exp as i32).clamp(3, 10));

                let pr_config = PageRankConfig {
                    damping,
                    max_iterations: max_iter,
                    tolerance,
                    direction: direction_from_u8(config.direction),
                    edge_type: None,
                };

                let result = engine.pagerank(Some(pr_config));
                if let Ok(pr_result) = result {
                    // Verify scores are non-negative
                    for score in pr_result.scores.values() {
                        assert!(
                            *score >= 0.0,
                            "PageRank scores should be non-negative"
                        );
                    }
                }
            }
            CentralityOp::BetweennessCentrality(config) => {
                let sampling = (config.sampling_ratio as f64 / 255.0).clamp(0.1, 1.0);
                let max_iter = (config.max_iterations as usize).clamp(10, 100);
                let c_config = CentralityConfig {
                    direction: direction_from_u8(config.direction),
                    sampling_ratio: sampling,
                    edge_type: None,
                    max_iterations: max_iter,
                    tolerance: 1e-6,
                };

                let result = engine.betweenness_centrality(Some(c_config));
                if let Ok(centrality_result) = result {
                    for score in centrality_result.scores.values() {
                        assert!(
                            *score >= 0.0,
                            "Betweenness centrality should be non-negative"
                        );
                    }
                }
            }
            CentralityOp::ClosenessCentrality(config) => {
                let max_iter = (config.max_iterations as usize).clamp(10, 100);
                let c_config = CentralityConfig {
                    direction: direction_from_u8(config.direction),
                    sampling_ratio: 1.0,
                    edge_type: None,
                    max_iterations: max_iter,
                    tolerance: 1e-6,
                };

                let result = engine.closeness_centrality(Some(c_config));
                if let Ok(centrality_result) = result {
                    for score in centrality_result.scores.values() {
                        assert!(
                            *score >= 0.0 && *score <= 1.0,
                            "Closeness centrality should be in [0, 1]"
                        );
                    }
                }
            }
            CentralityOp::EigenvectorCentrality(config) => {
                let max_iter = (config.max_iterations as usize).clamp(10, 100);
                let c_config = CentralityConfig {
                    direction: direction_from_u8(config.direction),
                    sampling_ratio: 1.0,
                    edge_type: None,
                    max_iterations: max_iter,
                    tolerance: 1e-6,
                };

                let result = engine.eigenvector_centrality(Some(c_config));
                if let Ok(centrality_result) = result {
                    for score in centrality_result.scores.values() {
                        assert!(
                            *score >= 0.0,
                            "Eigenvector centrality should be non-negative"
                        );
                    }
                }
            }
            CentralityOp::LouvainCommunities(config) => {
                let resolution = 0.5 + (config.resolution as f64 / 255.0) * 1.5;
                let max_passes = (config.max_passes as usize).clamp(1, 20);
                let max_iter = (config.max_iterations as usize).clamp(1, 50);

                let c_config = CommunityConfig {
                    resolution,
                    max_passes,
                    max_iterations: max_iter,
                    direction: direction_from_u8(config.direction),
                    edge_type: None,
                    seed: config.seed,
                };

                let _ = engine.louvain_communities(Some(c_config));
            }
            CentralityOp::LabelPropagation(config) => {
                let max_iter = (config.max_iterations as usize).clamp(1, 50);
                let max_passes = (config.max_passes as usize).clamp(1, 20);

                let c_config = CommunityConfig {
                    resolution: 1.0,
                    max_passes,
                    max_iterations: max_iter,
                    direction: direction_from_u8(config.direction),
                    edge_type: None,
                    seed: config.seed,
                };

                let _ = engine.label_propagation(Some(c_config));
            }
        }
    }
});
