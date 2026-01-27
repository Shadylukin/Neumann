//! Document indexer that stores documents across all three engines.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use neumann_client::NeumannClient;
use query_router::QueryResult;
use walkdir::WalkDir;

use crate::embeddings::TfIdfEmbedder;
use crate::markdown::{parse_markdown, ParsedDoc};

/// Helper to extract a string value from a Row.
fn get_string(row: &relational_engine::Row, col: &str) -> String {
    match row.get(col) {
        Some(relational_engine::Value::String(s)) => s.clone(),
        _ => String::new(),
    }
}

/// Helper to extract an integer value from a Row.
fn get_int(row: &relational_engine::Row, col: &str) -> i64 {
    match row.get(col) {
        Some(relational_engine::Value::Int(i)) => *i,
        _ => 0,
    }
}

/// Statistics from indexing operation.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Number of documents indexed.
    pub documents: usize,
    /// Number of relational rows created.
    pub rows: usize,
    /// Number of graph nodes created.
    pub nodes: usize,
    /// Number of graph edges created.
    pub edges: usize,
    /// Number of embeddings stored.
    pub embeddings: usize,
}

/// Document indexer using Neumann's three engines.
pub struct DocIndexer {
    client: NeumannClient,
}

impl DocIndexer {
    /// Create a new indexer with an embedded Neumann client.
    pub fn new() -> Result<Self> {
        let client = NeumannClient::embedded()
            .context("Failed to create embedded Neumann client")?;
        Ok(Self { client })
    }

    /// Create an indexer with an existing client.
    #[must_use]
    pub fn with_client(client: NeumannClient) -> Self {
        Self { client }
    }

    /// Index all markdown files in the given directory.
    pub fn index_directory(&mut self, path: &Path) -> Result<IndexStats> {
        let mut stats = IndexStats::default();

        // Scan for markdown files
        let docs = Self::scan_directory(path);
        stats.documents = docs.len();

        if docs.is_empty() {
            return Ok(stats);
        }

        // Build embedder from corpus
        let contents: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let mut embedder = TfIdfEmbedder::default();
        embedder.fit(&contents);

        // Create relational table
        self.create_table()?;
        stats.rows = docs.len();

        // Build path -> node_id mapping for edges
        let mut path_to_node: HashMap<String, u64> = HashMap::new();

        // Insert documents
        for doc in &docs {
            // Insert into relational table
            self.insert_row(doc)?;

            // Create graph node
            let node_id = self.create_node(doc)?;
            path_to_node.insert(doc.path.clone(), node_id);
            stats.nodes += 1;

            // Store embedding
            let embedding = embedder.embed(&doc.content);
            self.store_embedding(&doc.path, &embedding)?;
            stats.embeddings += 1;
        }

        // Create edges for document links
        for doc in &docs {
            if let Some(&from_id) = path_to_node.get(&doc.path) {
                for link in &doc.links {
                    // Try to find target node
                    if let Some(&to_id) = path_to_node.get(link) {
                        self.create_edge(from_id, to_id)?;
                        stats.edges += 1;
                    }
                }
            }
        }

        Ok(stats)
    }

    /// Scan directory for markdown files.
    fn scan_directory(path: &Path) -> Vec<ParsedDoc> {
        let mut docs = Vec::new();

        for entry in WalkDir::new(path)
            .follow_links(true)
            .into_iter()
            .filter_map(Result::ok)
        {
            let entry_path = entry.path();
            if entry_path.extension().is_some_and(|ext| ext == "md") {
                match parse_markdown(entry_path, path) {
                    Ok(doc) => docs.push(doc),
                    Err(e) => {
                        eprintln!("Warning: Failed to parse {}: {e}", entry_path.display());
                    }
                }
            }
        }

        docs
    }

    /// Create the docs table.
    fn create_table(&self) -> Result<()> {
        let query = "CREATE TABLE docs (id:int, path:string, title:string, category:string, size:int, word_count:int)";
        self.client
            .execute_sync(query)
            .context("Failed to create docs table")?;
        Ok(())
    }

    /// Insert a document row.
    fn insert_row(&self, doc: &ParsedDoc) -> Result<()> {
        // Escape single quotes in title
        let escaped_title = doc.title.replace('\'', "''");
        let query = format!(
            "INSERT docs id={}, path='{}', title='{}', category='{}', size={}, word_count={}",
            Self::path_hash(&doc.path),
            doc.path,
            escaped_title,
            doc.category,
            doc.size,
            doc.word_count
        );
        self.client
            .execute_sync(&query)
            .context("Failed to insert document row")?;
        Ok(())
    }

    /// Create a graph node for a document.
    fn create_node(&self, doc: &ParsedDoc) -> Result<u64> {
        // Escape single quotes in title
        let escaped_title = doc.title.replace('\'', "''");
        let query = format!(
            "NODE CREATE Doc path='{}', title='{}', category='{}'",
            doc.path, escaped_title, doc.category
        );
        let result = self
            .client
            .execute_sync(&query)
            .context("Failed to create graph node")?;

        // Extract the node ID from the result
        match result {
            QueryResult::Ids(ids) => {
                ids.first().copied().context("No node ID returned")
            }
            QueryResult::Count(n) => Ok(n as u64),
            _ => anyhow::bail!("Unexpected result from NODE CREATE: {result:?}"),
        }
    }

    /// Create an edge between two document nodes.
    fn create_edge(&self, from: u64, to: u64) -> Result<()> {
        let query = format!("EDGE CREATE {from} -> {to} links_to");
        self.client
            .execute_sync(&query)
            .context("Failed to create edge")?;
        Ok(())
    }

    /// Store an embedding for a document.
    fn store_embedding(&self, path: &str, embedding: &[f32]) -> Result<()> {
        let key = format!("doc:{path}");
        let values = TfIdfEmbedder::format_embedding(embedding);
        let query = format!("EMBED {key} [{values}]");
        self.client
            .execute_sync(&query)
            .context("Failed to store embedding")?;
        Ok(())
    }

    /// Generate a stable hash for a path to use as row ID (fits in i64).
    fn path_hash(path: &str) -> i64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in path.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        // Mask to ensure it fits in i64 positive range
        (hash & 0x7FFF_FFFF_FFFF_FFFF) as i64
    }

    /// List all indexed documents.
    pub fn list_documents(&self, category: Option<&str>) -> Result<Vec<DocumentInfo>> {
        let query = match category {
            Some(cat) => format!("SELECT docs WHERE category = '{cat}'"),
            None => "SELECT docs".to_string(),
        };

        let result = self
            .client
            .execute_sync(&query)
            .context("Failed to query documents")?;

        match result {
            QueryResult::Rows(rows) => {
                let mut docs = Vec::new();
                for row in rows {
                    docs.push(DocumentInfo {
                        id: row.id,
                        path: get_string(&row, "path"),
                        title: get_string(&row, "title"),
                        category: get_string(&row, "category"),
                        size: get_int(&row, "size") as usize,
                        word_count: get_int(&row, "word_count") as usize,
                    });
                }
                Ok(docs)
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Search for documents similar to a query string.
    pub fn search(&self, query_text: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        // Get all document contents to build embedder
        let docs = self.list_documents(None)?;
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        // Build embedder and generate query embedding
        let mut embedder = TfIdfEmbedder::default();

        // We need the actual content, but for search we'll reuse the stored embeddings
        // Just build a minimal embedder for the query
        let doc_texts: Vec<String> = docs.iter().map(|d| d.title.clone()).collect();
        let doc_refs: Vec<&str> = doc_texts.iter().map(String::as_str).collect();
        embedder.fit(&doc_refs);

        let query_embedding = embedder.embed(query_text);
        let values = TfIdfEmbedder::format_embedding(&query_embedding);

        let query = format!("SIMILAR [{values}] TOP {top_k}");
        let result = self
            .client
            .execute_sync(&query)
            .context("Failed to search")?;

        match result {
            QueryResult::Similar(items) => {
                let results = items
                    .into_iter()
                    .filter_map(|item| {
                        let path = item.key.strip_prefix("doc:")?;
                        Some(SearchResult {
                            path: path.to_string(),
                            score: item.score,
                        })
                    })
                    .collect();
                Ok(results)
            }
            _ => Ok(Vec::new()),
        }
    }

    /// Get document details including graph neighbors.
    pub fn get_document(&self, path: &str) -> Result<Option<DocumentDetail>> {
        // Get relational data
        let query = format!("SELECT docs WHERE path = '{path}'");
        let result = self.client.execute_sync(&query)?;

        let info = match result {
            QueryResult::Rows(rows) => {
                let row = rows.first().context("Document not found")?;
                DocumentInfo {
                    id: row.id,
                    path: get_string(row, "path"),
                    title: get_string(row, "title"),
                    category: get_string(row, "category"),
                    size: get_int(row, "size") as usize,
                    word_count: get_int(row, "word_count") as usize,
                }
            }
            _ => return Ok(None),
        };

        // Find the node ID for this document
        let node_query = format!("NODE FIND Doc WHERE path = '{path}'");
        let node_result = self.client.execute_sync(&node_query);

        let mut linked_docs = Vec::new();

        if let Ok(QueryResult::Nodes(nodes)) = node_result {
            if let Some(node) = nodes.first() {
                // Get neighbors
                let neighbors_query = format!("NEIGHBORS {} OUT", node.id);
                if let Ok(QueryResult::Nodes(neighbors)) =
                    self.client.execute_sync(&neighbors_query)
                {
                    for neighbor in neighbors {
                        if let Some(path) = neighbor.properties.get("path") {
                            linked_docs.push(path.clone());
                        }
                    }
                }
            }
        }

        Ok(Some(DocumentDetail {
            info,
            linked_docs,
        }))
    }

    /// Find path between two documents.
    pub fn find_path(&self, from_path: &str, to_path: &str) -> Result<Option<Vec<String>>> {
        // Find node IDs
        let from_query = format!("NODE FIND Doc WHERE path = '{from_path}'");
        let to_query = format!("NODE FIND Doc WHERE path = '{to_path}'");

        let from_result = self.client.execute_sync(&from_query)?;
        let to_result = self.client.execute_sync(&to_query)?;

        let from_id = match from_result {
            QueryResult::Nodes(nodes) => {
                nodes.first().map(|n| n.id)
            }
            _ => None,
        };

        let to_id = match to_result {
            QueryResult::Nodes(nodes) => {
                nodes.first().map(|n| n.id)
            }
            _ => None,
        };

        if let (Some(from), Some(to)) = (from_id, to_id) {
            let path_query = format!("PATH {from} -> {to}");
            let result = self.client.execute_sync(&path_query)?;

            match result {
                QueryResult::Path(node_ids) => {
                    let mut paths = Vec::new();
                    for id in node_ids {
                        let get_query = format!("NODE GET {id}");
                        if let Ok(QueryResult::Nodes(nodes)) =
                            self.client.execute_sync(&get_query)
                        {
                            if let Some(node) = nodes.first() {
                                if let Some(path) = node.properties.get("path") {
                                    paths.push(path.clone());
                                }
                            }
                        }
                    }
                    Ok(Some(paths))
                }
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    /// Get direct access to the client for advanced queries.
    #[must_use]
    pub fn client(&self) -> &NeumannClient {
        &self.client
    }
}

/// Basic document information from relational storage.
#[derive(Debug, Clone)]
pub struct DocumentInfo {
    pub id: u64,
    pub path: String,
    pub title: String,
    pub category: String,
    pub size: usize,
    pub word_count: usize,
}

/// Document detail with graph relationships.
#[derive(Debug, Clone)]
pub struct DocumentDetail {
    pub info: DocumentInfo,
    pub linked_docs: Vec<String>,
}

/// Search result with similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub path: String,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_docs(dir: &Path) {
        let doc1 = r#"# Introduction

This is the introduction to Neumann.

See [overview](architecture/overview.md) for more details.
"#;

        let doc2 = r#"# Architecture Overview

The architecture of Neumann is based on tensors.
"#;

        fs::write(dir.join("introduction.md"), doc1).unwrap();
        fs::create_dir_all(dir.join("architecture")).unwrap();
        fs::write(dir.join("architecture/overview.md"), doc2).unwrap();
    }

    #[test]
    fn test_indexer_creation() {
        let indexer = DocIndexer::new();
        assert!(indexer.is_ok());
    }

    #[test]
    fn test_index_directory() {
        let temp = TempDir::new().unwrap();
        create_test_docs(temp.path());

        let mut indexer = DocIndexer::new().unwrap();
        let stats = indexer.index_directory(temp.path()).unwrap();

        assert_eq!(stats.documents, 2);
        assert_eq!(stats.rows, 2);
        assert_eq!(stats.nodes, 2);
        assert_eq!(stats.embeddings, 2);
        // One link from introduction to overview
        assert_eq!(stats.edges, 1);
    }

    #[test]
    fn test_list_documents() {
        let temp = TempDir::new().unwrap();
        create_test_docs(temp.path());

        let mut indexer = DocIndexer::new().unwrap();
        indexer.index_directory(temp.path()).unwrap();

        let docs = indexer.list_documents(None).unwrap();
        assert_eq!(docs.len(), 2);

        let arch_docs = indexer.list_documents(Some("architecture")).unwrap();
        assert_eq!(arch_docs.len(), 1);
        assert_eq!(arch_docs[0].title, "Architecture Overview");
    }

    #[test]
    fn test_empty_directory() {
        let temp = TempDir::new().unwrap();
        let mut indexer = DocIndexer::new().unwrap();
        let stats = indexer.index_directory(temp.path()).unwrap();

        assert_eq!(stats.documents, 0);
    }
}
