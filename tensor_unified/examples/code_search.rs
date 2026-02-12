// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
use std::collections::HashMap;
use tensor_unified::UnifiedEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Neumann Code Intelligence Demo...");

    // 1. Initialize the Unified Engine (Replaces Postgres + Neo4j + Qdrant)
    let engine = UnifiedEngine::new();
    println!("✅ Unified Engine Initialized (One System, No Glue Code)");

    // 2. Simulate "Parsing" Code
    // We parsed 3 functions. We have their metadata, their relationships, and their "embeddings" (semantic meaning).

    // Function 1: process_data (Calls validate)
    // Embedding represents: "data processing logic"
    let mut props_1 = HashMap::new();
    props_1.insert("type".to_string(), "function".to_string());
    props_1.insert("language".to_string(), "rust".to_string());
    props_1.insert("line_count".to_string(), "50".to_string());

    // Function 2: validate_input (Called by process_data)
    // Embedding represents: "validation security logic"
    let mut props_2 = HashMap::new();
    props_2.insert("type".to_string(), "function".to_string());
    props_2.insert("is_safe".to_string(), "true".to_string());

    // Function 3: irrelevant_func
    // Embedding represents: "UI rendering"
    let mut props_3 = HashMap::new();
    props_3.insert("type".to_string(), "function".to_string());

    // 3. Store in Unified Engine (Relational + Vector)
    println!("\n📥 Ingesting Parsed Code...");

    // Simulating 4D embeddings for the demo
    // process_data: [1.0, 0.9, 0.0, 0.0]
    engine
        .create_entity("func:process_data", props_1, Some(vec![1.0, 0.9, 0.0, 0.0]))
        .await?;

    // validate_input: [0.0, 0.1, 0.9, 0.9]
    engine
        .create_entity(
            "func:validate_input",
            props_2,
            Some(vec![0.0, 0.1, 0.9, 0.9]),
        )
        .await?;

    // irrelevant_func: [0.5, 0.5, 0.5, 0.5]
    engine
        .create_entity(
            "func:irrelevant_func",
            props_3,
            Some(vec![0.5, 0.5, 0.5, 0.5]),
        )
        .await?;

    // 4. Create Graph Relationships (The "Glue" is now Native)
    // process_data CAUSES/CALLS validate_input
    engine
        .connect_entities("func:process_data", "func:validate_input", "CALLS")
        .await?;

    println!("✅ Codebase Ingested.");

    // 5. Execute a Complex Unified Query
    // "Find functions that are SEMANTICALLY SIMILAR to 'data logic' ([1.0, 1.0, 0.0, 0.0])
    //  AND are connected to 'func:validate_input' in the call graph."

    println!("\n🔍 Query: Find functions similar to 'data logic' that CALL 'validate_input'...");

    // In a pure vector DB, this would be a KNN search (returning process_data and others).
    // In a pure graph DB, this would be a neighbor lookup.
    // Neumann does both in one pass.

    let _query_vec = [1.0, 1.0, 0.0, 0.0];

    // Note: create_entity internally updates the HNSW index.
    // We perform the search.
    // We want to find X where X is similar to _query_vec AND X -> validates_input
    // The current API `find_similar_connected` does:
    // "Find nodes similar to QUERY_KEY that are connected to CONNECTED_TO"
    // Wait, let's check the API in lib.rs for exact semantics.
    // It is: pub async fn find_similar_connected(query_key: &str, connected_to: &str, top_k: usize)

    // So we use "func:process_data" as the query key (finding things similar to itself/its embedding
    // that are also connected to validate_input).

    let results = engine
        .find_similar_connected(
            "func:process_data",   // find things similar to this
            "func:validate_input", // that are neighbors of this
            5,
        )
        .await?;

    for item in results {
        println!(
            "   Found: {} (Score: {:.4})",
            item.id,
            item.score.unwrap_or(0.0)
        );
    }

    println!("\n✅ Demonstration Complete. The Ferrari works.");

    Ok(())
}
