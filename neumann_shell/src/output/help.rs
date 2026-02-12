// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Styled help text formatting.

use crate::style::{styled, Theme};
use std::fmt::Write;

/// Formats the help text with styling.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn format_help(theme: &Theme) -> String {
    let mut output = String::new();

    // Header
    let _ = writeln!(output, "{}", styled("Neumann Database Shell", theme.header));
    output.push('\n');

    // Built-in Commands
    section(&mut output, "Commands", theme);
    command(
        &mut output,
        "help, \\h, \\?",
        "Show this help message",
        theme,
    );
    command(&mut output, "exit, quit, \\q", "Exit the shell", theme);
    command(&mut output, "tables, \\dt", "List all tables", theme);
    command(&mut output, "clear, \\c", "Clear the screen", theme);
    output.push('\n');

    // Persistence
    section(&mut output, "Persistence", theme);
    command(&mut output, "save 'path'", "Save database snapshot", theme);
    command(
        &mut output,
        "save compressed 'path'",
        "Save compressed snapshot (int8 quantization)",
        theme,
    );
    command(
        &mut output,
        "load 'path'",
        "Load snapshot with strict WAL replay",
        theme,
    );
    command(
        &mut output,
        "load 'path' recover",
        "Load snapshot, skip corrupted WAL entries",
        theme,
    );
    command(
        &mut output,
        "wal status",
        "Show write-ahead log status",
        theme,
    );
    command(
        &mut output,
        "wal truncate",
        "Clear the write-ahead log",
        theme,
    );
    output.push('\n');

    // Relational
    section(&mut output, "Relational (SQL)", theme);
    command(
        &mut output,
        "CREATE TABLE name (col TYPE, ...)",
        "Create a new table",
        theme,
    );
    command(
        &mut output,
        "INSERT INTO table VALUES (...)",
        "Insert a row",
        theme,
    );
    command(
        &mut output,
        "SELECT cols FROM table [WHERE ...]",
        "Query rows",
        theme,
    );
    command(
        &mut output,
        "UPDATE table SET col = val [WHERE ...]",
        "Update rows",
        theme,
    );
    command(
        &mut output,
        "DELETE FROM table [WHERE ...]",
        "Delete rows",
        theme,
    );
    command(&mut output, "DROP TABLE name", "Delete a table", theme);
    command(
        &mut output,
        "DESCRIBE TABLE name",
        "Show table schema",
        theme,
    );
    command(
        &mut output,
        "DESCRIBE NODE label",
        "Show node label schema",
        theme,
    );
    command(
        &mut output,
        "DESCRIBE EDGE type",
        "Show edge type schema",
        theme,
    );
    command(
        &mut output,
        "CREATE INDEX name ON table (cols)",
        "Create an index",
        theme,
    );
    command(&mut output, "DROP INDEX name", "Drop an index", theme);
    output.push('\n');

    // Graph
    section(&mut output, "Graph", theme);
    command(
        &mut output,
        "NODE CREATE label {props}",
        "Create a node",
        theme,
    );
    command(&mut output, "NODE LIST [label]", "List nodes", theme);
    command(&mut output, "NODE GET id", "Get node by ID", theme);
    command(&mut output, "NODE DELETE id", "Delete node", theme);
    command(
        &mut output,
        "EDGE CREATE n1 -> n2 : label",
        "Create an edge",
        theme,
    );
    command(&mut output, "EDGE GET id", "Get edge by ID", theme);
    command(&mut output, "EDGE DELETE id", "Delete edge", theme);
    command(&mut output, "EDGE LIST [type]", "List edges", theme);
    command(
        &mut output,
        "NEIGHBORS id OUTGOING|INCOMING|BOTH",
        "Get neighbors",
        theme,
    );
    command(
        &mut output,
        "PATH SHORTEST|ALL|WEIGHTED n1 TO n2",
        "Find path",
        theme,
    );
    output.push('\n');

    // Graph Algorithms
    section(&mut output, "Graph Algorithms", theme);
    command(
        &mut output,
        "GRAPH ALGORITHM PAGERANK",
        "Compute PageRank scores",
        theme,
    );
    command(
        &mut output,
        "GRAPH ALGORITHM BETWEENNESS",
        "Compute betweenness centrality",
        theme,
    );
    command(
        &mut output,
        "GRAPH ALGORITHM CLOSENESS",
        "Closeness centrality",
        theme,
    );
    command(
        &mut output,
        "GRAPH ALGORITHM EIGENVECTOR",
        "Eigenvector centrality",
        theme,
    );
    command(
        &mut output,
        "GRAPH ALGORITHM LOUVAIN",
        "Detect communities",
        theme,
    );
    command(
        &mut output,
        "GRAPH ALGORITHM LABEL_PROPAGATION",
        "Label propagation communities",
        theme,
    );
    command(
        &mut output,
        "GRAPH CONSTRAINT CREATE|DROP|LIST|GET",
        "Manage graph constraints",
        theme,
    );
    command(
        &mut output,
        "GRAPH INDEX CREATE|DROP|SHOW",
        "Manage graph indexes",
        theme,
    );
    command(
        &mut output,
        "GRAPH AGGREGATE COUNT|SUM|AVG NODES|EDGES",
        "Graph aggregation",
        theme,
    );
    command(
        &mut output,
        "GRAPH PATTERN MATCH (a)-[r]->(b)",
        "Pattern matching",
        theme,
    );
    command(
        &mut output,
        "GRAPH BATCH CREATE|DELETE|UPDATE",
        "Batch graph operations",
        theme,
    );
    output.push('\n');

    // Vector
    section(&mut output, "Vector", theme);
    command(
        &mut output,
        "EMBED STORE 'key' [vector]",
        "Store embedding",
        theme,
    );
    command(&mut output, "EMBED GET 'key'", "Get embedding", theme);
    command(&mut output, "EMBED DELETE 'key'", "Delete embedding", theme);
    command(&mut output, "EMBED BUILD INDEX", "Build HNSW index", theme);
    command(
        &mut output,
        "SIMILAR 'key' [metric] LIMIT n",
        "Find similar vectors",
        theme,
    );
    command(
        &mut output,
        "EMBED BATCH [('key', [vec]), ...]",
        "Batch store embeddings",
        theme,
    );
    command(&mut output, "SHOW EMBEDDINGS", "List embeddings", theme);
    command(
        &mut output,
        "SHOW VECTOR INDEX",
        "Show HNSW index info",
        theme,
    );
    command(
        &mut output,
        "COUNT EMBEDDINGS",
        "Count stored embeddings",
        theme,
    );
    output.push('\n');

    // Entity
    section(&mut output, "Unified Entity", theme);
    command(
        &mut output,
        "ENTITY CREATE 'key' {props} [EMBEDDING [...]]",
        "Create entity",
        theme,
    );
    command(&mut output, "ENTITY GET 'key'", "Get entity", theme);
    command(
        &mut output,
        "ENTITY UPDATE 'key' {props}",
        "Update entity",
        theme,
    );
    command(&mut output, "ENTITY DELETE 'key'", "Delete entity", theme);
    command(
        &mut output,
        "ENTITY CONNECT 'a' -> 'b' : type",
        "Connect entities",
        theme,
    );
    command(
        &mut output,
        "ENTITY BATCH CREATE [{...}, ...]",
        "Batch create entities",
        theme,
    );
    command(
        &mut output,
        "FIND NODE|EDGE|ROWS|PATH [WHERE ...]",
        "Cross-engine search",
        theme,
    );
    output.push('\n');

    // Blob
    section(&mut output, "Blob Storage", theme);
    command(&mut output, "BLOB INIT", "Initialize blob store", theme);
    command(
        &mut output,
        "BLOB PUT 'path' [TAGS ...]",
        "Upload file",
        theme,
    );
    command(
        &mut output,
        "BLOB GET 'id' TO 'path'",
        "Download file",
        theme,
    );
    command(&mut output, "BLOB DELETE 'id'", "Delete blob", theme);
    command(&mut output, "BLOB INFO 'id'", "Show blob metadata", theme);
    command(
        &mut output,
        "BLOB LINK 'id' TO entity",
        "Link blob to entity",
        theme,
    );
    command(
        &mut output,
        "BLOB UNLINK 'id' FROM entity",
        "Unlink blob from entity",
        theme,
    );
    command(
        &mut output,
        "BLOB LINKS 'id'",
        "Show linked entities",
        theme,
    );
    command(
        &mut output,
        "BLOB TAG|UNTAG 'id' 'tag'",
        "Add or remove tag",
        theme,
    );
    command(
        &mut output,
        "BLOB VERIFY 'id'",
        "Verify blob integrity",
        theme,
    );
    command(&mut output, "BLOB GC [FULL]", "Garbage collection", theme);
    command(&mut output, "BLOB REPAIR", "Repair blob storage", theme);
    command(&mut output, "BLOB STATS", "Show blob statistics", theme);
    command(
        &mut output,
        "BLOB META SET|GET 'id' 'key' ['val']",
        "Set or get metadata",
        theme,
    );
    command(&mut output, "BLOBS", "List all blobs", theme);
    command(
        &mut output,
        "BLOBS FOR entity",
        "List blobs for entity",
        theme,
    );
    command(
        &mut output,
        "BLOBS BY TAG 'tag'",
        "List blobs by tag",
        theme,
    );
    output.push('\n');

    // Vault
    section(&mut output, "Vault (Secrets)", theme);
    command(&mut output, "VAULT INIT", "Initialize from env var", theme);
    command(
        &mut output,
        "VAULT IDENTITY 'name'",
        "Set access identity",
        theme,
    );
    command(
        &mut output,
        "VAULT SET 'key' 'value'",
        "Store secret",
        theme,
    );
    command(&mut output, "VAULT GET 'key'", "Retrieve secret", theme);
    command(&mut output, "VAULT DELETE 'key'", "Delete secret", theme);
    command(&mut output, "VAULT LIST [pattern]", "List secrets", theme);
    command(
        &mut output,
        "VAULT ROTATE 'key' 'new_value'",
        "Rotate secret",
        theme,
    );
    command(
        &mut output,
        "VAULT GRANT 'entity' ON 'key'",
        "Grant access",
        theme,
    );
    command(
        &mut output,
        "VAULT REVOKE 'entity' ON 'key'",
        "Revoke access",
        theme,
    );
    output.push('\n');

    // Cache
    section(&mut output, "Cache (LLM)", theme);
    command(&mut output, "CACHE INIT", "Initialize cache", theme);
    command(&mut output, "CACHE STATS", "Show statistics", theme);
    command(&mut output, "CACHE CLEAR", "Clear all entries", theme);
    command(&mut output, "CACHE EVICT [n]", "Evict LRU entries", theme);
    command(&mut output, "CACHE GET 'key'", "Get cached response", theme);
    command(
        &mut output,
        "CACHE PUT 'key' 'value'",
        "Store cache entry",
        theme,
    );
    command(
        &mut output,
        "CACHE SEMANTIC GET 'query' [THRESHOLD n]",
        "Semantic lookup",
        theme,
    );
    command(
        &mut output,
        "CACHE SEMANTIC PUT 'q' 'r' EMBEDDING [v]",
        "Store semantic entry",
        theme,
    );
    output.push('\n');

    // Checkpoints
    section(&mut output, "Checkpoints", theme);
    command(
        &mut output,
        "CHECKPOINT ['name']",
        "Create checkpoint",
        theme,
    );
    command(&mut output, "CHECKPOINTS", "List checkpoints", theme);
    command(
        &mut output,
        "ROLLBACK TO 'name'",
        "Restore checkpoint",
        theme,
    );
    output.push('\n');

    // Chain
    section(&mut output, "Chain (Distributed)", theme);
    command(
        &mut output,
        "BEGIN CHAIN TRANSACTION",
        "Start chain transaction",
        theme,
    );
    command(&mut output, "COMMIT CHAIN", "Commit transaction", theme);
    command(
        &mut output,
        "ROLLBACK CHAIN TO height",
        "Rollback chain",
        theme,
    );
    command(&mut output, "CHAIN HEIGHT", "Get chain height", theme);
    command(&mut output, "CHAIN TIP", "Get latest block", theme);
    command(
        &mut output,
        "CHAIN BLOCK height",
        "Get block at height",
        theme,
    );
    command(&mut output, "CHAIN VERIFY", "Verify chain integrity", theme);
    command(
        &mut output,
        "CHAIN HISTORY 'key'",
        "Key history across blocks",
        theme,
    );
    command(
        &mut output,
        "CHAIN SIMILAR [vec] LIMIT n",
        "Search chain by similarity",
        theme,
    );
    command(
        &mut output,
        "CHAIN DRIFT FROM h1 TO h2",
        "Drift metrics between heights",
        theme,
    );
    command(
        &mut output,
        "SHOW CODEBOOK GLOBAL|LOCAL 'domain'",
        "Show codebook",
        theme,
    );
    command(
        &mut output,
        "ANALYZE CODEBOOK TRANSITIONS",
        "Analyze codebook transitions",
        theme,
    );
    output.push('\n');

    // Cluster
    section(&mut output, "Cluster", theme);
    command(
        &mut output,
        "CLUSTER CONNECT 'node@addr'",
        "Join cluster",
        theme,
    );
    command(&mut output, "CLUSTER DISCONNECT", "Leave cluster", theme);
    command(&mut output, "CLUSTER STATUS", "Show cluster status", theme);
    command(&mut output, "CLUSTER NODES", "List cluster nodes", theme);
    command(&mut output, "CLUSTER LEADER", "Show current leader", theme);
    output.push('\n');

    // Cypher (experimental)
    section(&mut output, "Cypher (Experimental)", theme);
    command(
        &mut output,
        "MATCH (a:Label)-[:TYPE]->(b) RETURN ...",
        "Pattern matching query",
        theme,
    );
    command(
        &mut output,
        "CREATE (n:Label {props})",
        "Create nodes/edges",
        theme,
    );
    command(
        &mut output,
        "MERGE (n:Label {props})",
        "Upsert pattern",
        theme,
    );
    output.push('\n');

    // Examples
    section(&mut output, "Examples", theme);
    example(&mut output, "CREATE TABLE users (id INT, name TEXT)", theme);
    example(&mut output, "INSERT INTO users VALUES (1, 'Alice')", theme);
    example(&mut output, "SELECT * FROM users", theme);
    example(
        &mut output,
        "NODE CREATE person {name: 'Bob', age: 30}",
        theme,
    );
    example(&mut output, "EMBED STORE 'doc1' [0.1, 0.2, 0.3]", theme);
    example(&mut output, "SAVE 'backup.bin'", theme);

    output.trim_end().to_string()
}

/// Adds a section header.
fn section(output: &mut String, title: &str, theme: &Theme) {
    let _ = writeln!(output, "{}", styled(format!("{title}:"), theme.header));
}

/// Adds a command with description.
fn command(output: &mut String, cmd: &str, desc: &str, theme: &Theme) {
    let _ = writeln!(
        output,
        "  {:<40} {}",
        styled(cmd, theme.keyword),
        styled(desc, theme.muted)
    );
}

/// Adds an example command.
fn example(output: &mut String, cmd: &str, theme: &Theme) {
    let _ = writeln!(output, "  > {}", styled(cmd, theme.string));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_help_not_empty() {
        let theme = Theme::plain();
        let result = format_help(&theme);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_format_help_contains_sections() {
        let theme = Theme::plain();
        let result = format_help(&theme);
        assert!(result.contains("Commands"));
        assert!(result.contains("Persistence"));
        assert!(result.contains("Relational"));
        assert!(result.contains("Graph"));
        assert!(result.contains("Vector"));
        assert!(result.contains("Blob Storage"));
        assert!(result.contains("Vault"));
        assert!(result.contains("Cache"));
        assert!(result.contains("Chain"));
        assert!(result.contains("Cluster"));
        assert!(result.contains("Cypher"));
        assert!(result.contains("Examples"));
    }

    #[test]
    fn test_format_help_contains_commands() {
        let theme = Theme::plain();
        let result = format_help(&theme);
        assert!(result.contains("help"));
        assert!(result.contains("exit"));
        assert!(result.contains("SELECT"));
        assert!(result.contains("NODE CREATE"));
        assert!(result.contains("EMBED STORE"));
        assert!(result.contains("VAULT DELETE"));
        assert!(result.contains("VAULT ROTATE"));
        assert!(result.contains("VAULT REVOKE"));
        assert!(result.contains("CACHE EVICT"));
        assert!(result.contains("CACHE GET"));
        assert!(result.contains("CACHE PUT"));
        assert!(result.contains("EMBED BATCH"));
        assert!(result.contains("SHOW VECTOR INDEX"));
        assert!(result.contains("COUNT EMBEDDINGS"));
        assert!(result.contains("BLOB LINK"));
        assert!(result.contains("BLOB TAG"));
        assert!(result.contains("BLOB STATS"));
        assert!(result.contains("BLOB META"));
        assert!(result.contains("BLOBS FOR"));
        assert!(result.contains("BLOBS BY TAG"));
        assert!(result.contains("CHAIN HISTORY"));
        assert!(result.contains("CHAIN TIP"));
        assert!(result.contains("CHAIN DRIFT"));
        assert!(result.contains("SHOW CODEBOOK"));
        assert!(result.contains("CLUSTER NODES"));
        assert!(result.contains("CLUSTER LEADER"));
        assert!(result.contains("ENTITY UPDATE"));
        assert!(result.contains("ENTITY DELETE"));
        assert!(result.contains("ENTITY BATCH"));
        assert!(result.contains("DESCRIBE NODE"));
        assert!(result.contains("CREATE INDEX"));
        assert!(result.contains("GRAPH CONSTRAINT"));
        assert!(result.contains("GRAPH INDEX"));
        assert!(result.contains("GRAPH AGGREGATE"));
        assert!(result.contains("GRAPH BATCH"));
        assert!(result.contains("MATCH"));
        assert!(result.contains("MERGE"));
    }

    #[test]
    fn test_format_help_contains_examples() {
        let theme = Theme::plain();
        let result = format_help(&theme);
        assert!(result.contains("CREATE TABLE users"));
        assert!(result.contains("SAVE 'backup.bin'"));
    }
}
