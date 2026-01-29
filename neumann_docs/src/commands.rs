// SPDX-License-Identifier: MIT OR Apache-2.0
//! CLI command handlers.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use crate::indexer::DocIndexer;

/// Neumann documentation storage and search CLI.
#[derive(Parser, Debug)]
#[command(name = "neumann-docs")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// Available commands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Index markdown documentation files.
    Index {
        /// Path to documentation directory.
        #[arg(short, long, default_value = "docs/book/src")]
        path: PathBuf,
    },

    /// List indexed documents.
    List {
        /// Filter by category.
        #[arg(short, long)]
        category: Option<String>,
    },

    /// Semantic search for documents.
    Search {
        /// Search query.
        query: String,

        /// Number of results to return.
        #[arg(short = 'k', long, default_value = "5")]
        top: usize,
    },

    /// Show document details and links.
    Show {
        /// Document path (relative to docs root).
        path: String,
    },

    /// Find path between two documents.
    Path {
        /// Source document path.
        from: String,

        /// Target document path.
        to: String,
    },

    /// Run a full demo showing persistence within a session.
    Demo {
        /// Path to documentation directory.
        #[arg(short, long, default_value = "docs/book/src")]
        path: PathBuf,
    },
}

/// Execute a CLI command.
pub fn execute_command(cmd: &Command) -> Result<()> {
    match cmd {
        Command::Index { path } => cmd_index(path),
        Command::List { category } => cmd_list(category.as_deref()),
        Command::Search { query, top } => cmd_search(query, *top),
        Command::Show { path } => cmd_show(path),
        Command::Path { from, to } => cmd_path(from, to),
        Command::Demo { path } => cmd_demo(path),
    }
}

fn cmd_index(path: &Path) -> Result<()> {
    println!("Indexing documentation from: {}", path.display());

    let mut indexer = DocIndexer::new().context("Failed to create indexer")?;
    let stats = indexer
        .index_directory(path)
        .context("Failed to index directory")?;

    println!("\nIndexing complete:");
    println!("  Documents: {}", stats.documents);
    println!("  Relational rows: {}", stats.rows);
    println!("  Graph nodes: {}", stats.nodes);
    println!("  Graph edges: {}", stats.edges);
    println!("  Vector embeddings: {}", stats.embeddings);

    Ok(())
}

fn cmd_list(category: Option<&str>) -> Result<()> {
    let indexer = DocIndexer::new().context("Failed to create indexer")?;
    let docs = indexer
        .list_documents(category)
        .context("Failed to list documents")?;

    if docs.is_empty() {
        println!("No documents found. Run 'neumann-docs index' first.");
        return Ok(());
    }

    println!(
        "{:<50} {:<15} {:>8}",
        "PATH", "CATEGORY", "WORDS"
    );
    println!("{}", "-".repeat(75));

    for doc in docs {
        println!(
            "{:<50} {:<15} {:>8}",
            truncate(&doc.path, 48),
            doc.category,
            doc.word_count
        );
    }

    Ok(())
}

fn cmd_search(query: &str, top_k: usize) -> Result<()> {
    let indexer = DocIndexer::new().context("Failed to create indexer")?;
    let results = indexer
        .search(query, top_k)
        .context("Failed to search")?;

    if results.is_empty() {
        println!("No results found. Run 'neumann-docs index' first.");
        return Ok(());
    }

    println!("Search results for: \"{query}\"\n");
    println!("{:<60} {:>8}", "PATH", "SCORE");
    println!("{}", "-".repeat(70));

    for result in results {
        println!(
            "{:<60} {:>8.4}",
            truncate(&result.path, 58),
            result.score
        );
    }

    Ok(())
}

fn cmd_show(path: &str) -> Result<()> {
    let indexer = DocIndexer::new().context("Failed to create indexer")?;
    let detail = indexer
        .get_document(path)
        .context("Failed to get document")?;

    if let Some(doc) = detail {
        println!("Document: {}", doc.info.path);
        println!();
        println!("  Title:      {}", doc.info.title);
        println!("  Category:   {}", doc.info.category);
        println!("  Size:       {} bytes", doc.info.size);
        println!("  Words:      {}", doc.info.word_count);

        if !doc.linked_docs.is_empty() {
            println!();
            println!("  Links to:");
            for link in &doc.linked_docs {
                println!("    - {link}");
            }
        }
    } else {
        println!("Document not found: {path}");
        println!("Run 'neumann-docs index' first.");
    }

    Ok(())
}

fn cmd_path(from: &str, to: &str) -> Result<()> {
    let indexer = DocIndexer::new().context("Failed to create indexer")?;
    let path = indexer
        .find_path(from, to)
        .context("Failed to find path")?;

    match path {
        Some(nodes) if !nodes.is_empty() => {
            println!("Path from {from} to {to}:\n");
            for (i, node) in nodes.iter().enumerate() {
                if i > 0 {
                    println!("    |");
                    println!("    v");
                }
                println!("  [{i}] {node}");
            }
        }
        _ => {
            println!("No path found between {from} and {to}");
        }
    }

    Ok(())
}

fn cmd_demo(path: &Path) -> Result<()> {
    println!("=== Neumann Documentation Storage Demo ===\n");
    println!("This demo shows data persistence within a single session.\n");

    // Create a single indexer that persists throughout the demo
    let mut indexer = DocIndexer::new().context("Failed to create indexer")?;

    // Step 1: Index
    println!("Step 1: Indexing documentation from {}", path.display());
    println!("{}", "-".repeat(50));
    let stats = indexer
        .index_directory(path)
        .context("Failed to index directory")?;
    println!("  Documents indexed: {}", stats.documents);
    println!("  Relational rows:   {}", stats.rows);
    println!("  Graph nodes:       {}", stats.nodes);
    println!("  Graph edges:       {}", stats.edges);
    println!("  Embeddings:        {}", stats.embeddings);
    println!();

    // Step 2: List documents
    println!("Step 2: Listing documents (proving relational persistence)");
    println!("{}", "-".repeat(50));
    let docs = indexer.list_documents(None)?;
    println!("  Found {} documents in database", docs.len());
    for doc in docs.iter().take(5) {
        println!("    - {} ({})", doc.title, doc.category);
    }
    if docs.len() > 5 {
        println!("    ... and {} more", docs.len() - 5);
    }
    println!();

    // Step 3: Search
    println!("Step 3: Semantic search (proving vector persistence)");
    println!("{}", "-".repeat(50));
    let query = "distributed consensus";
    println!("  Query: \"{query}\"");
    let results = indexer.search(query, 3)?;
    for (i, r) in results.iter().enumerate() {
        println!("    {}. {} (score: {:.4})", i + 1, r.path, r.score);
    }
    println!();

    // Step 4: Show document with links
    println!("Step 4: Document details (proving graph persistence)");
    println!("{}", "-".repeat(50));
    if let Some(first_doc) = docs.first() {
        if let Some(detail) = indexer.get_document(&first_doc.path)? {
            println!("  Document: {}", detail.info.title);
            println!("  Category: {}", detail.info.category);
            println!("  Links to {} other documents:", detail.linked_docs.len());
            for link in detail.linked_docs.iter().take(3) {
                println!("    -> {link}");
            }
        }
    }
    println!();

    println!("=== Demo Complete ===");
    println!("\nAll three engines (relational, graph, vector) successfully");
    println!("stored and retrieved data within this session.");
    println!("\nNote: Data is stored in-memory. For persistence across");
    println!("sessions, run neumann_server and use remote mode.");

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("a very long string", 10), "a very ...");
        assert_eq!(truncate("exactly10c", 10), "exactly10c");
    }

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::parse_from(["neumann-docs", "index", "-p", "/tmp/docs"]);
        match cli.command {
            Command::Index { path } => {
                assert_eq!(path, PathBuf::from("/tmp/docs"));
            }
            _ => panic!("Expected Index command"),
        }
    }

    #[test]
    fn test_cli_list_parsing() {
        let cli = Cli::parse_from(["neumann-docs", "list", "-c", "architecture"]);
        match cli.command {
            Command::List { category } => {
                assert_eq!(category, Some("architecture".to_string()));
            }
            _ => panic!("Expected List command"),
        }
    }

    #[test]
    fn test_cli_search_parsing() {
        let cli = Cli::parse_from(["neumann-docs", "search", "raft consensus", "-k", "10"]);
        match cli.command {
            Command::Search { query, top } => {
                assert_eq!(query, "raft consensus");
                assert_eq!(top, 10);
            }
            _ => panic!("Expected Search command"),
        }
    }

    #[test]
    fn test_cli_show_parsing() {
        let cli = Cli::parse_from(["neumann-docs", "show", "architecture/overview.md"]);
        match cli.command {
            Command::Show { path } => {
                assert_eq!(path, "architecture/overview.md");
            }
            _ => panic!("Expected Show command"),
        }
    }

    #[test]
    fn test_cli_path_parsing() {
        let cli = Cli::parse_from([
            "neumann-docs",
            "path",
            "introduction.md",
            "tensor-chain.md",
        ]);
        match cli.command {
            Command::Path { from, to } => {
                assert_eq!(from, "introduction.md");
                assert_eq!(to, "tensor-chain.md");
            }
            _ => panic!("Expected Path command"),
        }
    }
}
