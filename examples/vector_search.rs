// SPDX-License-Identifier: MIT OR Apache-2.0
//! Vector similarity search example.
//!
//! This example demonstrates:
//! - Storing vector embeddings
//! - Performing k-NN similarity search
//! - Using different distance metrics
//!
//! Run with: `cargo run --example vector_search`

use vector_engine::{DistanceMetric, VectorEngine, VectorEngineConfig};

fn main() {
    println!("Neumann Vector Engine Example\n");

    // Create a vector engine with 8 dimensions
    let config = VectorEngineConfig {
        default_dimension: Some(8),
        default_metric: DistanceMetric::Cosine,
        ..Default::default()
    };
    let engine = VectorEngine::with_config(config).expect("Failed to create engine");

    // Sample embeddings representing different document categories
    // In practice, these would come from an embedding model
    let documents = vec![
        (
            "doc:ml_intro",
            "Introduction to Machine Learning",
            vec![0.8, 0.7, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1],
        ),
        (
            "doc:deep_learning",
            "Deep Learning Fundamentals",
            vec![0.9, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
        ),
        (
            "doc:neural_nets",
            "Neural Networks Explained",
            vec![0.85, 0.75, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1],
        ),
        (
            "doc:databases",
            "Database Design Patterns",
            vec![0.1, 0.1, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1],
        ),
        (
            "doc:sql_basics",
            "SQL Query Optimization",
            vec![0.1, 0.1, 0.75, 0.8, 0.25, 0.1, 0.1, 0.1],
        ),
        (
            "doc:web_dev",
            "Modern Web Development",
            vec![0.1, 0.1, 0.2, 0.2, 0.8, 0.7, 0.1, 0.1],
        ),
        (
            "doc:rust_lang",
            "The Rust Programming Language",
            vec![0.2, 0.1, 0.3, 0.3, 0.3, 0.3, 0.8, 0.7],
        ),
        (
            "doc:systems",
            "Systems Programming",
            vec![0.15, 0.1, 0.25, 0.25, 0.25, 0.25, 0.75, 0.8],
        ),
    ];

    // Store all embeddings
    println!("Storing document embeddings:");
    for (key, title, embedding) in &documents {
        engine
            .store_embedding(key, embedding.clone())
            .expect("Failed to store embedding");
        println!("  {} - {}", key, title);
    }
    println!();

    // Search for documents similar to a machine learning query
    let ml_query = vec![0.85, 0.75, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
    println!("Searching for documents similar to 'machine learning' query:");
    let ml_results = engine.search_similar(&ml_query, 3).expect("Failed to search");
    for result in &ml_results {
        let title = documents
            .iter()
            .find(|(k, _, _)| *k == result.key)
            .map(|(_, t, _)| *t)
            .unwrap_or("Unknown");
        println!("  {} (score: {:.4}) - {}", result.key, result.score, title);
    }
    println!();

    // Search for database-related documents
    let db_query = vec![0.1, 0.1, 0.8, 0.75, 0.2, 0.1, 0.1, 0.1];
    println!("Searching for documents similar to 'database' query:");
    let db_results = engine.search_similar(&db_query, 3).expect("Failed to search");
    for result in &db_results {
        let title = documents
            .iter()
            .find(|(k, _, _)| *k == result.key)
            .map(|(_, t, _)| *t)
            .unwrap_or("Unknown");
        println!("  {} (score: {:.4}) - {}", result.key, result.score, title);
    }
    println!();

    // Search for systems programming documents
    let systems_query = vec![0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8];
    println!("Searching for documents similar to 'systems programming' query:");
    let systems_results = engine.search_similar(&systems_query, 3).expect("Failed to search");
    for result in &systems_results {
        let title = documents
            .iter()
            .find(|(k, _, _)| *k == result.key)
            .map(|(_, t, _)| *t)
            .unwrap_or("Unknown");
        println!("  {} (score: {:.4}) - {}", result.key, result.score, title);
    }
    println!();

    // Retrieve a specific embedding
    println!("Retrieving embedding for 'doc:rust_lang':");
    if let Ok(embedding) = engine.get_embedding("doc:rust_lang") {
        print!("  [");
        for (i, val) in embedding.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.2}", val);
        }
        println!("]");
    }
    println!();

    // Statistics
    println!("Vector store statistics:");
    println!("  Total embeddings: {}", engine.count());
    if let Some(dim) = engine.dimension() {
        println!("  Dimension: {}", dim);
    }
}
