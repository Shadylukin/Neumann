// SPDX-License-Identifier: MIT OR Apache-2.0
//! Graph traversal example.
//!
//! This example demonstrates:
//! - Creating nodes with labels and properties
//! - Creating edges between nodes
//! - Shortest path finding
//!
//! Run with: `cargo run --example graph_traversal`

use graph_engine::{Direction, GraphEngine, PropertyValue};
use std::collections::HashMap;

fn main() {
    println!("Neumann Graph Engine Example\n");

    // Create a new graph engine
    let engine = GraphEngine::new();

    // Create nodes representing people
    let mut alice_props = HashMap::new();
    alice_props.insert(
        "name".to_string(),
        PropertyValue::String("Alice".to_string()),
    );
    alice_props.insert(
        "role".to_string(),
        PropertyValue::String("engineer".to_string()),
    );
    let alice = engine
        .create_node("Person", alice_props)
        .expect("Failed to create Alice");
    println!("Created Alice (node {})", alice);

    let mut bob_props = HashMap::new();
    bob_props.insert("name".to_string(), PropertyValue::String("Bob".to_string()));
    bob_props.insert(
        "role".to_string(),
        PropertyValue::String("manager".to_string()),
    );
    let bob = engine
        .create_node("Person", bob_props)
        .expect("Failed to create Bob");
    println!("Created Bob (node {})", bob);

    let mut charlie_props = HashMap::new();
    charlie_props.insert(
        "name".to_string(),
        PropertyValue::String("Charlie".to_string()),
    );
    charlie_props.insert(
        "role".to_string(),
        PropertyValue::String("engineer".to_string()),
    );
    let charlie = engine
        .create_node("Person", charlie_props)
        .expect("Failed to create Charlie");
    println!("Created Charlie (node {})", charlie);

    let mut diana_props = HashMap::new();
    diana_props.insert(
        "name".to_string(),
        PropertyValue::String("Diana".to_string()),
    );
    diana_props.insert(
        "role".to_string(),
        PropertyValue::String("designer".to_string()),
    );
    let diana = engine
        .create_node("Person", diana_props)
        .expect("Failed to create Diana");
    println!("Created Diana (node {})\n", diana);

    // Create a project node
    let mut project_props = HashMap::new();
    project_props.insert(
        "name".to_string(),
        PropertyValue::String("Neumann".to_string()),
    );
    project_props.insert(
        "status".to_string(),
        PropertyValue::String("active".to_string()),
    );
    let project = engine
        .create_node("Project", project_props)
        .expect("Failed to create Project");
    println!("Created Project Neumann (node {})\n", project);

    // Create edges
    engine
        .create_edge(alice, bob, "KNOWS", HashMap::new(), true)
        .expect("Failed to create edge");
    engine
        .create_edge(alice, charlie, "KNOWS", HashMap::new(), true)
        .expect("Failed to create edge");
    engine
        .create_edge(bob, diana, "MANAGES", HashMap::new(), true)
        .expect("Failed to create edge");
    engine
        .create_edge(charlie, diana, "KNOWS", HashMap::new(), true)
        .expect("Failed to create edge");
    engine
        .create_edge(alice, project, "WORKS_ON", HashMap::new(), true)
        .expect("Failed to create edge");
    engine
        .create_edge(charlie, project, "WORKS_ON", HashMap::new(), true)
        .expect("Failed to create edge");
    println!("Created relationship edges\n");

    // Get Alice's outgoing connections
    println!("Alice's outgoing connections:");
    let alice_edges = engine
        .edges_of(alice, Direction::Outgoing)
        .expect("Failed to get edges");
    for edge in &alice_edges {
        let target = engine.get_node(edge.to).expect("Failed to get node");
        println!(
            "  --[{}]--> {} (node {})",
            edge.edge_type,
            target
                .properties
                .get("name")
                .map(|v| format!("{:?}", v))
                .unwrap_or_else(|| "unknown".to_string()),
            edge.to
        );
    }
    println!();

    // Find shortest path from Alice to Diana
    println!("Shortest path from Alice to Diana:");
    match engine.find_path(alice, diana, None) {
        Ok(path) => {
            print!("  ");
            for (i, node_id) in path.nodes.iter().enumerate() {
                if let Ok(node) = engine.get_node(*node_id) {
                    let name = node
                        .properties
                        .get("name")
                        .map(|v| format!("{:?}", v))
                        .unwrap_or_else(|| "unknown".to_string());
                    if i > 0 {
                        print!(" -> ");
                    }
                    print!("{}", name);
                }
            }
            println!();
            println!("  Path length: {} hops", path.nodes.len() - 1);
        },
        Err(e) => println!("  No path found: {}", e),
    }
    println!();

    // Get all nodes by label
    println!("All Person nodes:");
    let people = engine
        .find_nodes_by_label("Person")
        .expect("Failed to get nodes");
    for node in &people {
        let name = node
            .properties
            .get("name")
            .map(|v| format!("{:?}", v))
            .unwrap_or_else(|| "unknown".to_string());
        let role = node
            .properties
            .get("role")
            .map(|v| format!("{:?}", v))
            .unwrap_or_else(|| "unknown".to_string());
        println!("  {} - {}", name, role);
    }

    // Statistics
    println!("\nGraph statistics:");
    println!("  Total nodes: {}", engine.node_count());
    println!("  Total edges: {}", engine.edge_count());
}
