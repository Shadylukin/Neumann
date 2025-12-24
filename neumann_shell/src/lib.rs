//! Neumann Shell - Interactive CLI for Neumann database
//!
//! Provides a readline-based interface for executing queries against the
//! Neumann unified query engine.

use query_router::{QueryResult, QueryRouter};
use relational_engine::Row;
use rustyline::error::ReadlineError;
use rustyline::history::{DefaultHistory, History};
use rustyline::{DefaultEditor, Editor};
use std::fmt::Write as _;
use std::path::PathBuf;

/// Shell configuration options.
#[derive(Debug, Clone)]
pub struct ShellConfig {
    /// Path to history file (None disables persistence).
    pub history_file: Option<PathBuf>,
    /// Maximum number of history entries to keep.
    pub history_size: usize,
    /// Prompt string displayed before each input.
    pub prompt: String,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            history_file: dirs_home().map(|h| h.join(".neumann_history")),
            history_size: 1000,
            prompt: "> ".to_string(),
        }
    }
}

/// Returns the user's home directory if available.
fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

/// Result of executing a shell command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandResult {
    /// Query executed successfully with output.
    Output(String),
    /// Shell should exit.
    Exit,
    /// Help text to display.
    Help(String),
    /// Empty input (no-op).
    Empty,
    /// Error occurred.
    Error(String),
}

/// Interactive shell for Neumann database.
pub struct Shell {
    router: QueryRouter,
    config: ShellConfig,
}

impl Shell {
    /// Creates a new shell with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            router: QueryRouter::new(),
            config: ShellConfig::default(),
        }
    }

    /// Creates a new shell with custom configuration.
    #[must_use]
    pub fn with_config(config: ShellConfig) -> Self {
        Self {
            router: QueryRouter::new(),
            config,
        }
    }

    /// Returns the query router for direct access.
    #[must_use]
    pub const fn router(&self) -> &QueryRouter {
        &self.router
    }

    /// Executes a single command and returns the result.
    #[must_use]
    pub fn execute(&self, input: &str) -> CommandResult {
        let trimmed = input.trim();

        if trimmed.is_empty() {
            return CommandResult::Empty;
        }

        // Handle built-in commands
        let lower = trimmed.to_lowercase();
        match lower.as_str() {
            "exit" | "quit" | "\\q" => return CommandResult::Exit,
            "help" | "\\h" | "\\?" => return CommandResult::Help(Self::help_text()),
            "tables" | "\\dt" => return self.list_tables(),
            "clear" | "\\c" => return CommandResult::Output("\x1B[2J\x1B[H".to_string()),
            _ => {},
        }

        // Execute as query
        self.router.execute_parsed(trimmed).map_or_else(
            |e| CommandResult::Error(format!("Error: {e}")),
            |result| CommandResult::Output(format_result(&result)),
        )
    }

    /// Lists all tables in the database.
    fn list_tables(&self) -> CommandResult {
        self.router.execute_parsed("SHOW TABLES").map_or_else(
            |_| CommandResult::Output("No tables found.".to_string()),
            |result| CommandResult::Output(format_result(&result)),
        )
    }

    /// Returns the help text.
    #[must_use]
    pub fn help_text() -> String {
        "\
Neumann Database Shell

Commands:
  help, \\h, \\?    Show this help message
  exit, quit, \\q  Exit the shell
  tables, \\dt     List all tables
  clear, \\c       Clear the screen

Query Types:
  Relational (SQL):
    CREATE TABLE name (col1 TYPE, col2 TYPE, ...)
    INSERT INTO table VALUES (val1, val2, ...)
    SELECT cols FROM table [WHERE condition]
    UPDATE table SET col = val [WHERE condition]
    DELETE FROM table [WHERE condition]
    DROP TABLE name

  Graph:
    NODE CREATE label {prop: value, ...}
    EDGE CREATE node1 -> node2 : label [{props}]
    NEIGHBORS node_id OUTGOING|INCOMING|BOTH [: label]
    PATH node1 -> node2 [LIMIT n]
    FIND NODE|EDGE WHERE condition

  Vector:
    EMBED STORE 'key' [vector values]
    EMBED GET 'key'
    EMBED DELETE 'key'
    SIMILAR 'key' LIMIT n

Examples:
  > CREATE TABLE users (id INT, name TEXT)
  > INSERT INTO users VALUES (1, 'Alice')
  > SELECT * FROM users
  > NODE CREATE person {name: 'Bob', age: 30}
  > EMBED STORE 'doc1' [0.1, 0.2, 0.3]
"
        .to_string()
    }

    /// Processes a command result and returns whether to continue the loop.
    #[must_use]
    pub fn process_result(result: &CommandResult) -> LoopAction {
        match result {
            CommandResult::Output(text) | CommandResult::Help(text) => {
                println!("{text}");
                LoopAction::Continue
            },
            CommandResult::Error(text) => {
                eprintln!("{text}");
                LoopAction::Continue
            },
            CommandResult::Exit => {
                println!("Goodbye!");
                LoopAction::Exit
            },
            CommandResult::Empty => LoopAction::Continue,
        }
    }

    /// Returns the shell version string.
    #[must_use]
    pub const fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    /// Runs the interactive shell loop.
    ///
    /// # Errors
    ///
    /// Returns an error if readline initialization fails.
    pub fn run(&self) -> Result<(), ShellError> {
        let mut editor: Editor<(), DefaultHistory> =
            DefaultEditor::new().map_err(|e| ShellError::Init(e.to_string()))?;
        if let Some(ref path) = self.config.history_file {
            let _ = editor.load_history(path);
        }
        editor
            .history_mut()
            .set_max_len(self.config.history_size)
            .map_err(|e| ShellError::Init(e.to_string()))?;

        println!("Neumann Database Shell v{}", Self::version());
        println!("Type 'help' for available commands.\n");

        loop {
            match editor.readline(&self.config.prompt) {
                Ok(line) => {
                    if !line.trim().is_empty() {
                        let _ = editor.add_history_entry(line.trim());
                    }
                    if Self::process_result(&self.execute(&line)) == LoopAction::Exit {
                        break;
                    }
                },
                Err(ReadlineError::Interrupted) => println!("^C"),
                Err(ReadlineError::Eof) => {
                    println!("Goodbye!");
                    break;
                },
                Err(err) => {
                    eprintln!("Error: {err}");
                    break;
                },
            }
        }
        if let Some(ref path) = self.config.history_file {
            let _ = editor.save_history(path);
        }
        Ok(())
    }
}

/// Action to take after processing a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopAction {
    /// Continue the shell loop.
    Continue,
    /// Exit the shell.
    Exit,
}

impl Default for Shell {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur in the shell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellError {
    /// Failed to initialize readline.
    Init(String),
}

impl std::fmt::Display for ShellError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init(msg) => write!(f, "Shell initialization failed: {msg}"),
        }
    }
}

impl std::error::Error for ShellError {}

/// Formats a query result for display.
fn format_result(result: &QueryResult) -> String {
    match result {
        QueryResult::Empty => "OK".to_string(),
        QueryResult::Value(s) => s.clone(),
        QueryResult::Count(n) => format_count(*n),
        QueryResult::Ids(ids) => format_ids(ids),
        QueryResult::Rows(rows) => format_rows(rows),
        QueryResult::Nodes(nodes) => format_nodes(nodes),
        QueryResult::Edges(edges) => format_edges(edges),
        QueryResult::Path(path) => format_path(path),
        QueryResult::Similar(results) => format_similar(results),
        QueryResult::Unified(unified) => unified.description.clone(),
        QueryResult::TableList(tables) => format_table_list(tables),
    }
}

/// Formats a count result.
fn format_count(n: usize) -> String {
    if n == 1 {
        "1 row affected".to_string()
    } else {
        format!("{n} rows affected")
    }
}

/// Formats a list of IDs.
fn format_ids(ids: &[u64]) -> String {
    if ids.is_empty() {
        "(no results)".to_string()
    } else if ids.len() == 1 {
        format!("ID: {}", ids[0])
    } else {
        format!(
            "IDs: {}",
            ids.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Formats graph nodes.
fn format_nodes(nodes: &[query_router::NodeResult]) -> String {
    if nodes.is_empty() {
        "(0 nodes)".to_string()
    } else {
        let lines: Vec<String> = nodes
            .iter()
            .map(|n| {
                let props: Vec<String> = n
                    .properties
                    .iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .collect();
                if props.is_empty() {
                    format!("  [{}] {} {{}}", n.id, n.label)
                } else {
                    format!("  [{}] {} {{{}}}", n.id, n.label, props.join(", "))
                }
            })
            .collect();
        format!("Nodes:\n{}\n({} nodes)", lines.join("\n"), nodes.len())
    }
}

/// Formats graph edges.
fn format_edges(edges: &[query_router::EdgeResult]) -> String {
    if edges.is_empty() {
        "(0 edges)".to_string()
    } else {
        let lines: Vec<String> = edges
            .iter()
            .map(|e| format!("  [{}] {} -> {} : {}", e.id, e.from, e.to, e.label))
            .collect();
        format!("Edges:\n{}\n({} edges)", lines.join("\n"), edges.len())
    }
}

/// Formats a graph path.
fn format_path(path: &[u64]) -> String {
    if path.is_empty() {
        "(no path found)".to_string()
    } else {
        format!(
            "Path: {}",
            path.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(" -> ")
        )
    }
}

/// Formats similar embedding results.
fn format_similar(results: &[query_router::SimilarResult]) -> String {
    if results.is_empty() {
        "(no similar embeddings)".to_string()
    } else {
        let lines: Vec<String> = results
            .iter()
            .enumerate()
            .map(|(i, r)| format!("  {}. {} (similarity: {:.4})", i + 1, r.key, r.score))
            .collect();
        format!("Similar:\n{}", lines.join("\n"))
    }
}

/// Formats a list of table names.
fn format_table_list(tables: &[String]) -> String {
    if tables.is_empty() {
        "No tables found.".to_string()
    } else {
        format!(
            "Tables:\n{}",
            tables
                .iter()
                .map(|t| format!("  {t}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

/// Formats rows as an ASCII table.
fn format_rows(rows: &[Row]) -> String {
    if rows.is_empty() {
        return "(0 rows)".to_string();
    }

    // Get column names from first row
    let columns: Vec<&String> = rows[0].values.keys().collect();
    if columns.is_empty() {
        return "(0 rows)".to_string();
    }

    // Convert rows to string values
    let string_rows: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            columns
                .iter()
                .map(|col| {
                    row.values
                        .get(*col)
                        .map(|v| format!("{v:?}"))
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();

    // Calculate column widths
    let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
    for row in &string_rows {
        for (i, cell) in row.iter().enumerate() {
            if i < widths.len() {
                widths[i] = widths[i].max(cell.len());
            }
        }
    }

    let mut output = String::new();

    // Header
    let header: Vec<String> = columns
        .iter()
        .zip(&widths)
        .map(|(col, &w)| format!("{col:w$}"))
        .collect();
    output.push_str(&header.join(" | "));
    output.push('\n');

    // Separator
    let sep: Vec<String> = widths.iter().map(|&w| "-".repeat(w)).collect();
    output.push_str(&sep.join("-+-"));
    output.push('\n');

    // Rows
    for row in &string_rows {
        let formatted: Vec<String> = row
            .iter()
            .zip(&widths)
            .map(|(cell, &w)| format!("{cell:w$}"))
            .collect();
        output.push_str(&formatted.join(" | "));
        output.push('\n');
    }

    let _ = write!(output, "({} rows)", rows.len());
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_creation() {
        let shell = Shell::new();
        assert!(!shell.config.prompt.is_empty());
    }

    #[test]
    fn test_shell_with_config() {
        let config = ShellConfig {
            history_file: None,
            history_size: 500,
            prompt: "neumann> ".to_string(),
        };
        let shell = Shell::with_config(config);
        assert_eq!(shell.config.prompt, "neumann> ");
        assert_eq!(shell.config.history_size, 500);
    }

    #[test]
    fn test_empty_input() {
        let shell = Shell::new();
        assert_eq!(shell.execute(""), CommandResult::Empty);
        assert_eq!(shell.execute("   "), CommandResult::Empty);
        assert_eq!(shell.execute("\t\n"), CommandResult::Empty);
    }

    #[test]
    fn test_exit_commands() {
        let shell = Shell::new();
        assert_eq!(shell.execute("exit"), CommandResult::Exit);
        assert_eq!(shell.execute("quit"), CommandResult::Exit);
        assert_eq!(shell.execute("\\q"), CommandResult::Exit);
        assert_eq!(shell.execute("EXIT"), CommandResult::Exit);
        assert_eq!(shell.execute("QUIT"), CommandResult::Exit);
    }

    #[test]
    fn test_help_commands() {
        let shell = Shell::new();

        let result = shell.execute("help");
        assert!(matches!(result, CommandResult::Help(_)));

        let result = shell.execute("\\h");
        assert!(matches!(result, CommandResult::Help(_)));

        let result = shell.execute("\\?");
        assert!(matches!(result, CommandResult::Help(_)));
    }

    #[test]
    fn test_help_content() {
        let help = Shell::help_text();
        assert!(help.contains("CREATE TABLE"));
        assert!(help.contains("SELECT"));
        assert!(help.contains("NODE CREATE"));
        assert!(help.contains("EMBED STORE"));
        assert!(help.contains("SIMILAR"));
    }

    #[test]
    fn test_clear_command() {
        let shell = Shell::new();
        let result = shell.execute("clear");
        assert!(matches!(result, CommandResult::Output(_)));

        let result = shell.execute("\\c");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_create_table() {
        let shell = Shell::new();
        let result = shell.execute("CREATE TABLE users (id INT, name TEXT)");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_insert_and_select() {
        let shell = Shell::new();

        let _ = shell.execute("CREATE TABLE test (id INT, value TEXT)");
        let _ = shell.execute("INSERT INTO test VALUES (1, 'hello')");
        let _ = shell.execute("INSERT INTO test VALUES (2, 'world')");

        let result = shell.execute("SELECT * FROM test");
        // Just check we get an Output result
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_select_empty_table() {
        let shell = Shell::new();
        let _ = shell.execute("CREATE TABLE empty (id INT)");

        let result = shell.execute("SELECT * FROM empty");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("0 rows"));
        } else {
            panic!("Expected Output");
        }
    }

    #[test]
    fn test_node_create() {
        let shell = Shell::new();
        let result = shell.execute("NODE CREATE person {name: 'Alice', age: 30}");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_edge_create() {
        let shell = Shell::new();
        let _ = shell.execute("NODE CREATE person {name: 'Alice'}");
        let _ = shell.execute("NODE CREATE person {name: 'Bob'}");

        let result = shell.execute("EDGE CREATE 1 -> 2 : knows");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_neighbors() {
        let shell = Shell::new();
        let _ = shell.execute("NODE CREATE person {name: 'Alice'}");
        let _ = shell.execute("NODE CREATE person {name: 'Bob'}");
        let _ = shell.execute("EDGE CREATE 1 -> 2 : knows");

        let result = shell.execute("NEIGHBORS 1 OUTGOING");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_embed_store_and_get() {
        let shell = Shell::new();

        let result = shell.execute("EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4]");
        assert!(matches!(result, CommandResult::Output(_)));

        let result = shell.execute("EMBED GET 'doc1'");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("0.1") || output.contains("OK"));
        } else {
            panic!("Expected Output");
        }
    }

    #[test]
    fn test_similar_search() {
        let shell = Shell::new();

        let _ = shell.execute("EMBED STORE 'a' [1.0, 0.0, 0.0]");
        let _ = shell.execute("EMBED STORE 'b' [0.9, 0.1, 0.0]");
        let _ = shell.execute("EMBED STORE 'c' [0.0, 1.0, 0.0]");

        let result = shell.execute("SIMILAR 'a' LIMIT 2");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("Similar") || output.contains("similarity"));
        } else {
            panic!("Expected Output");
        }
    }

    #[test]
    fn test_invalid_query() {
        let shell = Shell::new();
        let result = shell.execute("INVALID QUERY SYNTAX");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_shell_error_display() {
        let err = ShellError::Init("test error".to_string());
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_default_config() {
        let config = ShellConfig::default();
        assert_eq!(config.prompt, "> ");
        assert_eq!(config.history_size, 1000);
    }

    #[test]
    fn test_tables_command() {
        let shell = Shell::new();
        let _ = shell.execute("CREATE TABLE foo (id INT)");
        let _ = shell.execute("CREATE TABLE bar (id INT)");

        let result = shell.execute("tables");
        // Either shows tables or "No tables" - both are valid outputs
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_router_access() {
        let shell = Shell::new();
        let _ = shell.router();
    }

    #[test]
    fn test_format_empty_result() {
        let result = QueryResult::Empty;
        assert_eq!(format_result(&result), "OK");
    }

    #[test]
    fn test_format_count() {
        assert_eq!(format_result(&QueryResult::Count(1)), "1 row affected");
        assert_eq!(format_result(&QueryResult::Count(5)), "5 rows affected");
    }

    #[test]
    fn test_format_ids() {
        assert_eq!(format_result(&QueryResult::Ids(vec![])), "(no results)");
        assert_eq!(format_result(&QueryResult::Ids(vec![42])), "ID: 42");
        assert_eq!(
            format_result(&QueryResult::Ids(vec![1, 2, 3])),
            "IDs: 1, 2, 3"
        );
    }

    #[test]
    fn test_format_path() {
        assert_eq!(format_result(&QueryResult::Path(vec![])), "(no path found)");
        assert_eq!(
            format_result(&QueryResult::Path(vec![1, 2, 3])),
            "Path: 1 -> 2 -> 3"
        );
    }

    #[test]
    fn test_format_value() {
        assert_eq!(
            format_result(&QueryResult::Value("hello".to_string())),
            "hello"
        );
    }

    #[test]
    fn test_format_nodes_empty() {
        assert_eq!(format_result(&QueryResult::Nodes(vec![])), "(0 nodes)");
    }

    #[test]
    fn test_format_nodes_with_data() {
        use query_router::NodeResult;
        use std::collections::HashMap;

        let nodes = vec![
            NodeResult {
                id: 1,
                label: "person".to_string(),
                properties: HashMap::new(),
            },
            NodeResult {
                id: 2,
                label: "person".to_string(),
                properties: {
                    let mut m = HashMap::new();
                    m.insert("name".to_string(), "Alice".to_string());
                    m
                },
            },
        ];
        let output = format_result(&QueryResult::Nodes(nodes));
        assert!(output.contains("Nodes:"));
        assert!(output.contains("[1] person"));
        assert!(output.contains("[2] person"));
        assert!(output.contains("2 nodes"));
    }

    #[test]
    fn test_format_edges_empty() {
        assert_eq!(format_result(&QueryResult::Edges(vec![])), "(0 edges)");
    }

    #[test]
    fn test_format_edges_with_data() {
        use query_router::EdgeResult;

        let edges = vec![EdgeResult {
            id: 1,
            from: 1,
            to: 2,
            label: "knows".to_string(),
        }];
        let output = format_result(&QueryResult::Edges(edges));
        assert!(output.contains("Edges:"));
        assert!(output.contains("[1] 1 -> 2 : knows"));
        assert!(output.contains("1 edges"));
    }

    #[test]
    fn test_format_similar_empty() {
        assert_eq!(
            format_result(&QueryResult::Similar(vec![])),
            "(no similar embeddings)"
        );
    }

    #[test]
    fn test_format_similar_with_data() {
        use query_router::SimilarResult;

        let results = vec![
            SimilarResult {
                key: "doc1".to_string(),
                score: 0.95,
            },
            SimilarResult {
                key: "doc2".to_string(),
                score: 0.85,
            },
        ];
        let output = format_result(&QueryResult::Similar(results));
        assert!(output.contains("Similar:"));
        assert!(output.contains("1. doc1"));
        assert!(output.contains("2. doc2"));
        assert!(output.contains("similarity"));
    }

    #[test]
    fn test_format_unified() {
        use query_router::UnifiedResult;

        let unified = UnifiedResult {
            description: "Combined result".to_string(),
            items: vec![],
        };
        assert_eq!(
            format_result(&QueryResult::Unified(unified)),
            "Combined result"
        );
    }

    #[test]
    fn test_format_rows_empty() {
        assert_eq!(format_result(&QueryResult::Rows(vec![])), "(0 rows)");
    }

    #[test]
    fn test_format_rows_with_data() {
        use relational_engine::{Row, Value};
        use std::collections::HashMap;

        let rows = vec![
            Row {
                id: 1,
                values: {
                    let mut m = HashMap::new();
                    m.insert("name".to_string(), Value::String("Alice".into()));
                    m.insert("age".to_string(), Value::Int(30));
                    m
                },
            },
            Row {
                id: 2,
                values: {
                    let mut m = HashMap::new();
                    m.insert("name".to_string(), Value::String("Bob".into()));
                    m.insert("age".to_string(), Value::Int(25));
                    m
                },
            },
        ];
        let output = format_result(&QueryResult::Rows(rows));
        assert!(output.contains("2 rows"));
    }

    #[test]
    fn test_default_shell() {
        let shell = Shell::default();
        assert_eq!(shell.config.prompt, "> ");
    }

    #[test]
    fn test_shell_config_no_history() {
        let config = ShellConfig {
            history_file: None,
            history_size: 100,
            prompt: "$ ".to_string(),
        };
        let shell = Shell::with_config(config);
        assert!(shell.config.history_file.is_none());
    }

    #[test]
    fn test_count_zero() {
        assert_eq!(format_result(&QueryResult::Count(0)), "0 rows affected");
    }

    #[test]
    fn test_format_rows_empty_columns() {
        use relational_engine::Row;
        use std::collections::HashMap;

        // Row with empty values HashMap
        let rows = vec![Row {
            id: 1,
            values: HashMap::new(),
        }];
        assert_eq!(format_rows(&rows), "(0 rows)");
    }

    #[test]
    fn test_shell_error_is_error() {
        use std::error::Error;
        let err = ShellError::Init("test".to_string());
        // Verify Error trait is implemented
        let _: &dyn Error = &err;
    }

    #[test]
    fn test_format_rows_missing_column() {
        use relational_engine::{Row, Value};
        use std::collections::HashMap;

        // Create rows where second row is missing a column
        let rows = vec![
            Row {
                id: 1,
                values: {
                    let mut m = HashMap::new();
                    m.insert("a".to_string(), Value::Int(1));
                    m.insert("b".to_string(), Value::Int(2));
                    m
                },
            },
            Row {
                id: 2,
                values: {
                    let mut m = HashMap::new();
                    m.insert("a".to_string(), Value::Int(3));
                    // 'b' is missing - should use default
                    m
                },
            },
        ];
        let output = format_rows(&rows);
        assert!(output.contains("2 rows"));
    }

    #[test]
    fn test_format_rows_single_row() {
        use relational_engine::{Row, Value};
        use std::collections::HashMap;

        let rows = vec![Row {
            id: 1,
            values: {
                let mut m = HashMap::new();
                m.insert("x".to_string(), Value::Int(42));
                m
            },
        }];
        let output = format_rows(&rows);
        assert!(output.contains("(1 rows)"));
        assert!(output.contains("x"));
        assert!(output.contains("Int(42)"));
    }

    #[test]
    fn test_dirs_home() {
        // dirs_home should return Some when HOME is set
        let result = dirs_home();
        // HOME is typically set in test environment
        assert!(result.is_some() || std::env::var_os("HOME").is_none());
    }

    #[test]
    fn test_list_tables_empty() {
        let shell = Shell::new();
        // Without any tables created, should still return output
        let result = shell.execute("\\dt");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_format_nodes_with_properties() {
        use query_router::NodeResult;
        use std::collections::HashMap;

        let nodes = vec![NodeResult {
            id: 1,
            label: "person".to_string(),
            properties: {
                let mut m = HashMap::new();
                m.insert("name".to_string(), "Alice".to_string());
                m.insert("age".to_string(), "30".to_string());
                m
            },
        }];
        let output = format_result(&QueryResult::Nodes(nodes));
        assert!(output.contains("[1] person"));
        assert!(output.contains("name:"));
        assert!(output.contains("(1 nodes)"));
    }

    #[test]
    fn test_format_single_id() {
        assert_eq!(format_result(&QueryResult::Ids(vec![1])), "ID: 1");
    }

    #[test]
    fn test_tables_alias() {
        let shell = Shell::new();
        // Both "tables" and "\dt" should work
        let result1 = shell.execute("tables");
        let result2 = shell.execute("\\dt");
        assert!(matches!(result1, CommandResult::Output(_)));
        assert!(matches!(result2, CommandResult::Output(_)));
    }

    #[test]
    fn test_case_insensitive_commands() {
        let shell = Shell::new();
        assert_eq!(
            shell.execute("HELP"),
            CommandResult::Help(Shell::help_text())
        );
        assert_eq!(
            shell.execute("Help"),
            CommandResult::Help(Shell::help_text())
        );
        assert_eq!(
            shell.execute("CLEAR"),
            CommandResult::Output("\x1B[2J\x1B[H".to_string())
        );
        assert_eq!(
            shell.execute("TABLES"),
            CommandResult::Output("No tables found.".to_string())
        );
    }

    #[test]
    fn test_process_result_output() {
        assert_eq!(
            Shell::process_result(&CommandResult::Output("test".to_string())),
            LoopAction::Continue
        );
    }

    #[test]
    fn test_process_result_help() {
        assert_eq!(
            Shell::process_result(&CommandResult::Help("help text".to_string())),
            LoopAction::Continue
        );
    }

    #[test]
    fn test_process_result_error() {
        assert_eq!(
            Shell::process_result(&CommandResult::Error("error".to_string())),
            LoopAction::Continue
        );
    }

    #[test]
    fn test_process_result_exit() {
        assert_eq!(
            Shell::process_result(&CommandResult::Exit),
            LoopAction::Exit
        );
    }

    #[test]
    fn test_process_result_empty() {
        assert_eq!(
            Shell::process_result(&CommandResult::Empty),
            LoopAction::Continue
        );
    }

    #[test]
    fn test_shell_version() {
        let version = Shell::version();
        assert!(!version.is_empty());
        assert!(version.contains('.'));
    }

    #[test]
    fn test_loop_action_eq() {
        assert_eq!(LoopAction::Continue, LoopAction::Continue);
        assert_eq!(LoopAction::Exit, LoopAction::Exit);
        assert_ne!(LoopAction::Continue, LoopAction::Exit);
    }

    #[test]
    fn test_loop_action_clone() {
        let action = LoopAction::Continue;
        let cloned = action.clone();
        assert_eq!(action, cloned);
    }

    #[test]
    fn test_loop_action_debug() {
        let action = LoopAction::Exit;
        let debug_str = format!("{action:?}");
        assert!(debug_str.contains("Exit"));
    }

    #[test]
    fn test_loop_action_copy() {
        let action = LoopAction::Continue;
        let copied: LoopAction = action;
        assert_eq!(action, copied);
    }

    #[test]
    fn test_command_result_clone() {
        let result = CommandResult::Output("test".to_string());
        let cloned = result.clone();
        assert_eq!(result, cloned);
    }

    #[test]
    fn test_command_result_debug() {
        let result = CommandResult::Exit;
        let debug = format!("{result:?}");
        assert!(debug.contains("Exit"));
    }

    #[test]
    fn test_shell_config_clone() {
        let config = ShellConfig::default();
        let cloned = config.clone();
        assert_eq!(config.prompt, cloned.prompt);
    }

    #[test]
    fn test_shell_error_clone() {
        let err = ShellError::Init("test".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_format_path_single() {
        assert_eq!(format_result(&QueryResult::Path(vec![1])), "Path: 1");
    }

    #[test]
    fn test_format_rows_header_formatting() {
        use relational_engine::{Row, Value};
        use std::collections::HashMap;

        let rows = vec![Row {
            id: 1,
            values: {
                let mut m = HashMap::new();
                m.insert(
                    "long_column_name".to_string(),
                    Value::String("short".into()),
                );
                m
            },
        }];
        let output = format_rows(&rows);
        assert!(output.contains("long_column_name"));
        assert!(output.contains("-+-") || output.contains("---"));
    }

    #[test]
    fn test_format_count_large() {
        assert_eq!(
            format_result(&QueryResult::Count(1000000)),
            "1000000 rows affected"
        );
    }

    #[test]
    fn test_format_ids_many() {
        let ids: Vec<u64> = (1..=10).collect();
        let output = format_result(&QueryResult::Ids(ids));
        assert!(output.starts_with("IDs:"));
        assert!(output.contains("10"));
    }

    #[test]
    fn test_format_table_list_empty() {
        let output = format_result(&QueryResult::TableList(vec![]));
        assert_eq!(output, "No tables found.");
    }

    #[test]
    fn test_format_table_list_with_tables() {
        let tables = vec!["users".to_string(), "products".to_string()];
        let output = format_result(&QueryResult::TableList(tables));
        assert!(output.starts_with("Tables:"));
        assert!(output.contains("users"));
        assert!(output.contains("products"));
    }
}
