// SPDX-License-Identifier: MIT OR Apache-2.0
//! Five-node codebook demo.
//!
//! Creates a small hierarchy with 5 entries at levels 0-2 in the Poincare disk,
//! runs a nearest-neighbor query, and prints the visualization JSON.

use tensor_learn::{Codebook, PoincarePoint, TrainingConfig, TrainingSession};

fn main() {
    let mut cb = Codebook::new(2, 1.0).expect("codebook creation");

    // Level 0: root at the origin (most general)
    cb.add_entry("root", "Knowledge", &PoincarePoint::origin(2), 0)
        .expect("add root");

    // Level 1: two broad categories
    cb.add_entry(
        "math",
        "Mathematics",
        &PoincarePoint::new(vec![0.3, 0.1]),
        1,
    )
    .expect("add math");

    cb.add_entry(
        "science",
        "Science",
        &PoincarePoint::new(vec![-0.2, 0.3]),
        1,
    )
    .expect("add science");

    // Level 2: specific topics (further from origin)
    cb.add_entry("algebra", "Algebra", &PoincarePoint::new(vec![0.6, 0.2]), 2)
        .expect("add algebra");

    cb.add_entry(
        "physics",
        "Physics",
        &PoincarePoint::new(vec![-0.3, 0.6]),
        2,
    )
    .expect("add physics");

    // Create hierarchy edges
    cb.add_edge("root", "math", "contains").expect("edge");
    cb.add_edge("root", "science", "contains").expect("edge");
    cb.add_edge("math", "algebra", "contains").expect("edge");
    cb.add_edge("science", "physics", "contains").expect("edge");

    // Query: nearest to origin should be root
    let query = PoincarePoint::origin(2);
    let results = cb.nearest(&query, 1).expect("search");
    assert_eq!(results[0].key, "root", "nearest to origin should be root");
    println!(
        "Nearest to origin: {} (score: {:.4})",
        results[0].key, results[0].score
    );

    // Run a training step
    let config = TrainingConfig {
        total_steps: 5,
        dimension: 2,
        ..TrainingConfig::default()
    };
    let mut session = TrainingSession::new(cb, config);
    session.step().expect("training step");
    let stats = session.stats();
    println!(
        "Step {}/{}: loss = {:.6}",
        stats.step, stats.total_steps, stats.loss
    );

    // Print visualization data
    let viz = session.to_viz_data().expect("viz");
    let json = serde_json::to_string_pretty(&viz).expect("json");
    println!("\nVisualization data:\n{json}");
}
