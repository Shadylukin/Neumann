// SPDX-License-Identifier: MIT OR Apache-2.0
//! Example: Start the Neumann Web Admin UI with real documentation.
//!
//! This demonstrates Neumann's unified storage by loading the actual project
//! documentation from `docs/book/src/` and showing it through three views:
//!
//! - **Relational**: Query docs by section, category, difficulty
//! - **Vector**: Semantic search across documentation content
//! - **Graph**: Navigate doc relationships and hierarchy
//!
//! Run with: cargo run --example web_admin

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use graph_engine::GraphEngine;
use parking_lot::RwLock;
use query_router::QueryRouter;
use relational_engine::{Column, ColumnType, RelationalEngine, Schema, Value};
use tensor_store::{ScalarValue, TensorValue};
use vector_engine::VectorEngine;

use neumann_server::{NeumannServer, ServerConfig};

/// A documentation page loaded from markdown.
#[derive(Debug, Clone)]
struct DocPage {
    /// Unique identifier (filename without extension).
    id: String,
    /// Document title (from first H1).
    title: String,
    /// Section (from parent directory).
    section: String,
    /// Full markdown content.
    content: String,
    /// File path relative to docs root.
    path: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ==========================================================================
    // Load real documentation from docs/book/src/
    // ==========================================================================

    let docs_path = find_docs_path()?;
    println!("Loading documentation from: {}", docs_path.display());

    let docs = load_documentation(&docs_path)?;
    println!("Loaded {} documentation pages", docs.len());

    if docs.is_empty() {
        eprintln!("No documentation found. Make sure docs/book/src/ exists.");
        std::process::exit(1);
    }

    // ==========================================================================
    // Create the three engines
    // ==========================================================================

    let relational = Arc::new(RelationalEngine::new());
    let vector = Arc::new(VectorEngine::new());
    let graph = Arc::new(GraphEngine::new());

    // ==========================================================================
    // VIEW 1: RELATIONAL - Query docs by metadata
    // ==========================================================================

    let docs_schema = Schema::new(vec![
        Column::new("id", ColumnType::String),
        Column::new("title", ColumnType::String),
        Column::new("section", ColumnType::String),
        Column::new("path", ColumnType::String),
        Column::new("word_count", ColumnType::Int),
        Column::new("content", ColumnType::String),
    ]);
    relational.create_table("documents", docs_schema)?;

    for doc in &docs {
        let word_count = doc.content.split_whitespace().count() as i64;
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::String(doc.id.clone()));
        row.insert("title".to_string(), Value::String(doc.title.clone()));
        row.insert("section".to_string(), Value::String(doc.section.clone()));
        row.insert("path".to_string(), Value::String(doc.path.clone()));
        row.insert("word_count".to_string(), Value::Int(word_count));
        row.insert("content".to_string(), Value::String(doc.content.clone()));
        relational.insert("documents", row)?;
    }

    // Create a sections summary table
    let sections: Vec<_> = {
        let mut section_counts: HashMap<String, i64> = HashMap::new();
        for doc in &docs {
            *section_counts.entry(doc.section.clone()).or_insert(0) += 1;
        }
        let mut sections: Vec<_> = section_counts.into_iter().collect();
        sections.sort_by(|a, b| b.1.cmp(&a.1));
        sections
    };

    let sections_schema = Schema::new(vec![
        Column::new("name", ColumnType::String),
        Column::new("doc_count", ColumnType::Int),
    ]);
    relational.create_table("sections", sections_schema)?;

    for (name, count) in &sections {
        let mut row = HashMap::new();
        row.insert("name".to_string(), Value::String(name.clone()));
        row.insert("doc_count".to_string(), Value::Int(*count));
        relational.insert("sections", row)?;
    }

    // ==========================================================================
    // VIEW 2: VECTOR - Semantic search across content
    // ==========================================================================

    vector.create_collection(
        "docs",
        vector_engine::VectorCollectionConfig::default()
            .with_dimension(128)
            .with_metric(vector_engine::DistanceMetric::Cosine),
    )?;

    for (i, doc) in docs.iter().enumerate() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "title".to_string(),
            TensorValue::Scalar(ScalarValue::String(doc.title.clone())),
        );
        metadata.insert(
            "section".to_string(),
            TensorValue::Scalar(ScalarValue::String(doc.section.clone())),
        );
        metadata.insert(
            "path".to_string(),
            TensorValue::Scalar(ScalarValue::String(doc.path.clone())),
        );
        // Store full content
        metadata.insert(
            "content".to_string(),
            TensorValue::Scalar(ScalarValue::String(doc.content.clone())),
        );

        // Generate embedding based on content characteristics
        let embedding = generate_content_embedding(i, doc);
        vector.store_in_collection_with_metadata("docs", &doc.id, embedding, metadata)?;
    }

    // ==========================================================================
    // VIEW 3: GRAPH - Navigate doc relationships
    // ==========================================================================

    // Create document nodes
    let mut doc_nodes: HashMap<String, u64> = HashMap::new();
    for doc in &docs {
        let mut props = HashMap::new();
        props.insert(
            "title".to_string(),
            graph_engine::PropertyValue::String(doc.title.clone()),
        );
        props.insert(
            "section".to_string(),
            graph_engine::PropertyValue::String(doc.section.clone()),
        );
        props.insert(
            "path".to_string(),
            graph_engine::PropertyValue::String(doc.path.clone()),
        );
        let node_id = graph.create_node("Document", props)?;
        doc_nodes.insert(doc.id.clone(), node_id);
    }

    // Create section nodes
    let mut section_nodes: HashMap<String, u64> = HashMap::new();
    for (name, count) in &sections {
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            graph_engine::PropertyValue::String(name.clone()),
        );
        props.insert(
            "doc_count".to_string(),
            graph_engine::PropertyValue::Int(*count),
        );
        let node_id = graph.create_node("Section", props)?;
        section_nodes.insert(name.clone(), node_id);
    }

    // Connect documents to their sections (IN_SECTION)
    for doc in &docs {
        if let (Some(&doc_id), Some(&section_id)) =
            (doc_nodes.get(&doc.id), section_nodes.get(&doc.section))
        {
            graph.create_edge(doc_id, section_id, "IN_SECTION", HashMap::new(), true)?;
        }
    }

    // Create RELATED_TO edges based on content similarity and cross-references
    create_doc_relationships(&graph, &docs, &doc_nodes)?;

    // ==========================================================================
    // Start the server
    // ==========================================================================

    let router = Arc::new(RwLock::new(QueryRouter::new()));
    let config = ServerConfig::default().with_web_addr("127.0.0.1:9201".parse()?);

    let doc_count = docs.len();
    let section_count = sections.len();
    let edge_count = graph.edge_count();
    let node_count = graph.node_count();

    println!();
    println!("  Neumann Web Admin - Real Documentation Demo");
    println!("  ============================================");
    println!();
    println!("  Web UI:  http://127.0.0.1:9201/");
    println!("  gRPC:    127.0.0.1:50051");
    println!();
    println!(
        "  Loaded {} documentation pages from docs/book/src/",
        doc_count
    );
    println!();
    println!("  RELATIONAL  /relational");
    println!("    - 'documents' table: {} rows", doc_count);
    println!("    - 'sections' table: {} rows", section_count);
    println!();
    println!("  VECTOR  /vector");
    println!("    - 'docs' collection: {} embeddings", doc_count);
    println!("    - Semantic search across all documentation");
    println!();
    println!("  GRAPH  /graph");
    println!("    - {} nodes (Document + Section)", node_count);
    println!(
        "    - {} edges (IN_SECTION, RELATED_TO, LINKS_TO)",
        edge_count
    );
    println!();
    println!("  Press Ctrl+C to stop the server.");
    println!();

    let server = NeumannServer::new(router, config)
        .with_relational_engine(relational)
        .with_vector_engine(vector)
        .with_graph_engine(graph);

    server.serve().await?;

    Ok(())
}

/// Find the docs/book/src/ directory.
fn find_docs_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Try relative to current directory
    let paths = [
        PathBuf::from("docs/book/src"),
        PathBuf::from("../docs/book/src"),
        PathBuf::from("../../docs/book/src"),
    ];

    for path in &paths {
        if path.exists() && path.is_dir() {
            return Ok(path.clone());
        }
    }

    // Try from CARGO_MANIFEST_DIR
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        let path = PathBuf::from(manifest_dir)
            .parent()
            .map(|p| p.join("docs/book/src"))
            .unwrap_or_default();
        if path.exists() && path.is_dir() {
            return Ok(path);
        }
    }

    Err("Could not find docs/book/src/ directory".into())
}

/// Load all markdown documentation files.
fn load_documentation(docs_path: &Path) -> Result<Vec<DocPage>, Box<dyn std::error::Error>> {
    let mut docs = Vec::new();
    load_docs_recursive(docs_path, docs_path, &mut docs)?;
    Ok(docs)
}

/// Recursively load markdown files from a directory.
fn load_docs_recursive(
    base_path: &Path,
    current_path: &Path,
    docs: &mut Vec<DocPage>,
) -> Result<(), Box<dyn std::error::Error>> {
    if !current_path.is_dir() {
        return Ok(());
    }

    let entries: Vec<_> = fs::read_dir(current_path)?.filter_map(|e| e.ok()).collect();

    for entry in entries {
        let path = entry.path();

        if path.is_dir() {
            load_docs_recursive(base_path, &path, docs)?;
        } else if path.extension().map_or(false, |ext| ext == "md") {
            if let Some(doc) = load_markdown_file(base_path, &path)? {
                docs.push(doc);
            }
        }
    }

    Ok(())
}

/// Load a single markdown file and extract metadata.
fn load_markdown_file(
    base_path: &Path,
    file_path: &Path,
) -> Result<Option<DocPage>, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(file_path)?;

    // Skip empty files
    if content.trim().is_empty() {
        return Ok(None);
    }

    // Extract ID from filename
    let id = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Skip SUMMARY.md (it's just a table of contents)
    if id == "SUMMARY" {
        return Ok(None);
    }

    // Extract title from first H1 heading
    let title = content
        .lines()
        .find(|line| line.starts_with("# "))
        .map(|line| line.trim_start_matches("# ").trim().to_string())
        .unwrap_or_else(|| id.replace('-', " ").to_string());

    // Extract section from parent directory
    let section = file_path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .map(|s| {
            // Convert directory name to title case
            s.replace('-', " ")
                .split_whitespace()
                .map(|word| {
                    let mut chars = word.chars();
                    match chars.next() {
                        None => String::new(),
                        Some(first) => first.to_uppercase().chain(chars).collect(),
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        })
        .unwrap_or_else(|| "General".to_string());

    // Relative path from docs root
    let rel_path = file_path
        .strip_prefix(base_path)
        .unwrap_or(file_path)
        .to_string_lossy()
        .to_string();

    Ok(Some(DocPage {
        id,
        title,
        section,
        content,
        path: rel_path,
    }))
}

/// Generate an embedding for a document based on content characteristics.
/// In production, you would use a real embedding model.
fn generate_content_embedding(index: usize, doc: &DocPage) -> Vec<f32> {
    let mut embedding = vec![0.0f32; 128];

    // Base embedding from content hash
    let content_hash = doc.content.bytes().fold(0u64, |acc, b| {
        acc.wrapping_mul(31).wrapping_add(u64::from(b))
    });

    for (j, val) in embedding.iter_mut().enumerate() {
        let seed = content_hash.wrapping_add(j as u64);
        *val = ((seed as f32 * 0.0000001).sin() * 0.3).clamp(-1.0, 1.0);
    }

    // Section clustering (dims 0-31)
    let section_offset = match doc.section.as_str() {
        "Getting Started" => 0.5,
        "Architecture" => 0.3,
        "Concepts" => 0.2,
        "Operations" => -0.1,
        "Benchmarks" => -0.2,
        "Stress Tests" => -0.3,
        "Contributing" => -0.4,
        "Tutorials" => 0.4,
        "Runbooks" => -0.2,
        "Integration Tests" => -0.3,
        _ => 0.0,
    };
    for val in embedding.iter_mut().take(32) {
        *val += section_offset;
    }

    // Content length clustering (dims 32-63)
    let word_count = doc.content.split_whitespace().count();
    let length_offset = if word_count < 200 {
        0.3 // Short
    } else if word_count < 500 {
        0.0 // Medium
    } else {
        -0.3 // Long
    };
    for val in embedding.iter_mut().skip(32).take(32) {
        *val += length_offset;
    }

    // Code presence clustering (dims 64-95)
    let has_code = doc.content.contains("```");
    let code_offset = if has_code { 0.3 } else { -0.3 };
    for val in embedding.iter_mut().skip(64).take(32) {
        *val += code_offset;
    }

    // Index-based uniqueness (dims 96-127)
    for (j, val) in embedding.iter_mut().skip(96).enumerate() {
        *val += ((index * 32 + j) as f32 * 0.1).sin() * 0.2;
    }

    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }

    embedding
}

/// Create RELATED_TO edges between documents based on content analysis.
fn create_doc_relationships(
    graph: &GraphEngine,
    docs: &[DocPage],
    doc_nodes: &HashMap<String, u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create relationships based on markdown links
    for doc in docs {
        let from_id = match doc_nodes.get(&doc.id) {
            Some(id) => *id,
            None => continue,
        };

        // Find markdown links to other docs: [text](path.md) or [text](../path.md)
        for line in doc.content.lines() {
            // Simple link extraction - look for .md links
            let mut chars = line.chars().peekable();
            while let Some(c) = chars.next() {
                if c == '(' {
                    let link: String = chars.by_ref().take_while(|&c| c != ')').collect();
                    if link.ends_with(".md") {
                        // Extract the filename
                        let linked_file = link
                            .trim_start_matches("../")
                            .trim_start_matches("./")
                            .trim_end_matches(".md");
                        let linked_id = linked_file
                            .rsplit('/')
                            .next()
                            .unwrap_or(linked_file)
                            .to_string();

                        if let Some(&to_id) = doc_nodes.get(&linked_id) {
                            if from_id != to_id {
                                // Create LINKS_TO edge
                                let _ = graph.create_edge(
                                    from_id,
                                    to_id,
                                    "LINKS_TO",
                                    HashMap::new(),
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    // Create RELATED_TO edges for docs in the same section
    let mut section_docs: HashMap<&str, Vec<(&str, u64)>> = HashMap::new();
    for doc in docs {
        if let Some(&node_id) = doc_nodes.get(&doc.id) {
            section_docs
                .entry(&doc.section)
                .or_default()
                .push((&doc.id, node_id));
        }
    }

    for docs_in_section in section_docs.values() {
        // Connect adjacent docs in the same section
        for window in docs_in_section.windows(2) {
            if let [(_id1, node1), (_id2, node2)] = window {
                let _ = graph.create_edge(*node1, *node2, "RELATED_TO", HashMap::new(), true);
            }
        }
    }

    Ok(())
}
