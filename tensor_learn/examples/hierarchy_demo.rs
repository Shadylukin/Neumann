// SPDX-License-Identifier: MIT OR Apache-2.0
//! Depth-3 binary tree hierarchy learning demo.
//!
//! Generates a 15-node binary tree, trains for 100 Riemannian SGD steps,
//! and verifies that the root drifts to the center while leaves move to
//! the boundary of the Poincare disk.

use tensor_learn::{HierarchyTask, TrainingConfig};

fn main() {
    let task = HierarchyTask::binary_tree(3);
    println!(
        "Binary tree: {} nodes, {} edges",
        task.node_count(),
        task.edge_pairs().len()
    );

    let config = TrainingConfig {
        total_steps: 100,
        learning_rate: 0.1,
        seed: 42,
        dimension: 2,
        curvature: 1.0,
        ..TrainingConfig::default()
    };

    let mut session = task.into_session(config).expect("session creation");

    for step in 1..=100 {
        session.step().expect("training step");
        if step % 10 == 0 {
            let stats = session.stats();
            println!("Step {:>3}/100: loss = {:.6}", stats.step, stats.loss);
        }
    }

    let viz = session.to_viz_data().expect("viz");

    let root = viz.nodes.iter().find(|n| n.id == "n0").expect("root node");
    let max_leaf_dist = viz
        .nodes
        .iter()
        .filter(|n| n.level == 3)
        .map(|n| n.distance_from_origin)
        .fold(f64::NEG_INFINITY, f64::max);

    println!(
        "\nRoot distance from origin: {:.4}",
        root.distance_from_origin
    );
    println!("Max leaf distance from origin: {:.4}", max_leaf_dist);

    assert!(
        root.distance_from_origin < max_leaf_dist,
        "root ({:.4}) should be closer to origin than furthest leaf ({:.4})",
        root.distance_from_origin,
        max_leaf_dist,
    );
    println!("\nHierarchy verified: root closer to center than leaves.");

    let json = serde_json::to_string_pretty(&viz).expect("json");
    println!("\nFinal VizData:\n{json}");
}
