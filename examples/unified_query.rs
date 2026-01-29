// SPDX-License-Identifier: MIT OR Apache-2.0
//! Unified cross-engine query example.
//!
//! This example demonstrates:
//! - Using QueryRouter for unified queries
//! - Cross-engine operations
//! - Combining relational, graph, and vector queries
//!
//! Run with: `cargo run --example unified_query`

use query_router::QueryRouter;

fn main() {
    println!("Neumann Unified Query Example\n");

    // Create a query router with all engines
    let router = QueryRouter::new();

    // Execute relational queries
    println!("=== Relational Operations ===\n");

    // Create a table
    let result = router.execute("CREATE TABLE products (id INT, name TEXT, category TEXT, price FLOAT)");
    println!("Create table: {:?}\n", result);

    // Insert products
    let products = [
        "INSERT products id=1, name='Laptop', category='Electronics', price=999.99",
        "INSERT products id=2, name='Keyboard', category='Electronics', price=79.99",
        "INSERT products id=3, name='Desk Chair', category='Furniture', price=249.99",
        "INSERT products id=4, name='Monitor', category='Electronics', price=399.99",
        "INSERT products id=5, name='Desk', category='Furniture', price=349.99",
    ];

    for query in &products {
        router.execute(query).expect("Failed to insert");
    }
    println!("Inserted 5 products");

    // Query products
    println!("\nAll products:");
    if let Ok(result) = router.execute("SELECT * FROM products") {
        println!("{:?}", result);
    }

    println!("\nElectronics products:");
    if let Ok(result) = router.execute("SELECT * FROM products WHERE category = 'Electronics'") {
        println!("{:?}", result);
    }

    // Graph operations
    println!("\n=== Graph Operations ===\n");

    // Create nodes
    router.execute("NODE CREATE Person name='Alice', role='developer'").expect("Failed to create node");
    router.execute("NODE CREATE Person name='Bob', role='manager'").expect("Failed to create node");
    router.execute("NODE CREATE Project name='Neumann', status='active'").expect("Failed to create node");
    println!("Created Person and Project nodes");

    // Create edges
    router.execute("EDGE CREATE node:1 -> node:2 REPORTS_TO").expect("Failed to create edge");
    router.execute("EDGE CREATE node:1 -> node:3 WORKS_ON").expect("Failed to create edge");
    router.execute("EDGE CREATE node:2 -> node:3 MANAGES").expect("Failed to create edge");
    println!("Created relationship edges");

    // Query nodes
    println!("\nAll Person nodes:");
    if let Ok(result) = router.execute("NODE QUERY Person") {
        println!("{:?}", result);
    }

    // Traverse path
    println!("\nPath from node 1 to node 3:");
    if let Ok(result) = router.execute("PATH node:1 -> node:3") {
        println!("{:?}", result);
    }

    // Vector operations
    println!("\n=== Vector Operations ===\n");

    // Store embeddings
    router.execute("EMBED 'doc:intro' [0.1, 0.2, 0.3, 0.4, 0.5]").expect("Failed to embed");
    router.execute("EMBED 'doc:advanced' [0.15, 0.25, 0.35, 0.45, 0.55]").expect("Failed to embed");
    router.execute("EMBED 'doc:reference' [0.9, 0.8, 0.7, 0.6, 0.5]").expect("Failed to embed");
    println!("Stored 3 document embeddings");

    // Similarity search
    println!("\nSimilar to 'doc:intro':");
    if let Ok(result) = router.execute("SIMILAR 'doc:intro' TOP 2") {
        println!("{:?}", result);
    }

    // Statistics
    println!("\n=== Statistics ===\n");

    // List tables
    if let Ok(result) = router.execute("LIST TABLES") {
        println!("Tables: {:?}", result);
    }

    // Count products
    if let Ok(result) = router.execute("COUNT products") {
        println!("Product count: {:?}", result);
    }

    println!("\nUnified query example complete!");
}
