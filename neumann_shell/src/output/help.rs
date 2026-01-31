// SPDX-License-Identifier: MIT OR Apache-2.0
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
    command(&mut output, "EDGE LIST [type]", "List edges", theme);
    command(
        &mut output,
        "NEIGHBORS id OUTGOING|INCOMING|BOTH",
        "Get neighbors",
        theme,
    );
    command(&mut output, "PATH n1 -> n2 [LIMIT n]", "Find path", theme);
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
        "GRAPH ALGORITHM LOUVAIN",
        "Detect communities",
        theme,
    );
    command(
        &mut output,
        "GRAPH PATTERN MATCH (a)-[r]->(b)",
        "Pattern matching",
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
    command(&mut output, "SHOW EMBEDDINGS", "List embeddings", theme);
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
        "ENTITY CONNECT 'a' -> 'b' : type",
        "Connect entities",
        theme,
    );
    command(
        &mut output,
        "FIND NODE [label] [WHERE ...] [LIMIT n]",
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
    command(&mut output, "BLOB GC", "Garbage collection", theme);
    command(&mut output, "BLOBS", "List all blobs", theme);
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
    command(
        &mut output,
        "VAULT GRANT 'entity' ON 'key'",
        "Grant access",
        theme,
    );
    output.push('\n');

    // Cache
    section(&mut output, "Cache (LLM)", theme);
    command(&mut output, "CACHE INIT", "Initialize cache", theme);
    command(&mut output, "CACHE STATS", "Show statistics", theme);
    command(&mut output, "CACHE CLEAR", "Clear all entries", theme);
    command(
        &mut output,
        "CACHE SEMANTIC GET 'query'",
        "Semantic lookup",
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
    command(&mut output, "CHAIN HEIGHT", "Get chain height", theme);
    command(&mut output, "CHAIN VERIFY", "Verify chain integrity", theme);
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
    }

    #[test]
    fn test_format_help_contains_examples() {
        let theme = Theme::plain();
        let result = format_help(&theme);
        assert!(result.contains("CREATE TABLE users"));
        assert!(result.contains("SAVE 'backup.bin'"));
    }
}
