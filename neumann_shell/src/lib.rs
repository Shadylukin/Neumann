//! Neumann Shell - Interactive CLI for Neumann database
//!
//! Provides a readline-based interface for executing queries against the
//! Neumann unified query engine.

use query_router::{
    ChainBlockInfo, ChainCodebookInfo, ChainDriftResult, ChainHistoryEntry, ChainResult,
    ChainSimilarResult, ChainTransitionAnalysis, CheckpointInfo, QueryResult, QueryRouter,
};
use relational_engine::Row;
use rustyline::error::ReadlineError;
use rustyline::history::{DefaultHistory, History};
use rustyline::{DefaultEditor, Editor};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use tensor_store::TensorStore;

/// Write-Ahead Log for crash recovery.
///
/// Logs mutating commands to a file so they can be replayed after loading a snapshot.
/// The WAL is activated after LOAD and truncated after SAVE.
struct Wal {
    file: File,
    path: PathBuf,
}

impl Wal {
    fn open_append(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    fn append(&mut self, cmd: &str) -> std::io::Result<()> {
        writeln!(self.file, "{cmd}")?;
        self.file.flush()
    }

    fn truncate(&mut self) -> std::io::Result<()> {
        self.file = File::create(&self.path)?;
        Ok(())
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn size(&self) -> std::io::Result<u64> {
        std::fs::metadata(&self.path).map(|m| m.len())
    }
}

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
    wal: Option<Wal>,
}

impl Shell {
    /// Creates a new shell with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            router: QueryRouter::new(),
            config: ShellConfig::default(),
            wal: None,
        }
    }

    /// Creates a new shell with custom configuration.
    #[must_use]
    pub fn with_config(config: ShellConfig) -> Self {
        Self {
            router: QueryRouter::new(),
            config,
            wal: None,
        }
    }

    /// Returns the query router for direct access.
    #[must_use]
    pub const fn router(&self) -> &QueryRouter {
        &self.router
    }

    /// Returns a mutable reference to the query router.
    #[must_use]
    pub const fn router_mut(&mut self) -> &mut QueryRouter {
        &mut self.router
    }

    /// Check if a command is a write operation that should be logged to WAL.
    fn is_write_command(cmd: &str) -> bool {
        let upper = cmd.to_uppercase();
        let first_word = upper.split_whitespace().next().unwrap_or("");

        match first_word {
            "INSERT" | "UPDATE" | "DELETE" | "CREATE" | "DROP" => true,
            "NODE" => !upper.contains("NODE GET"),
            "EDGE" => !upper.contains("EDGE GET"),
            "EMBED" => upper.contains("EMBED STORE") || upper.contains("EMBED DELETE"),
            "VAULT" => {
                upper.contains("VAULT SET")
                    || upper.contains("VAULT DELETE")
                    || upper.contains("VAULT ROTATE")
                    || upper.contains("VAULT GRANT")
                    || upper.contains("VAULT REVOKE")
            },
            "CACHE" => upper.contains("CACHE CLEAR"),
            "BLOB" => {
                upper.contains("BLOB PUT")
                    || upper.contains("BLOB DELETE")
                    || upper.contains("BLOB LINK")
                    || upper.contains("BLOB UNLINK")
                    || upper.contains("BLOB TAG")
                    || upper.contains("BLOB UNTAG")
                    || upper.contains("BLOB GC")
                    || upper.contains("BLOB REPAIR")
                    || upper.contains("BLOB META SET")
            },
            _ => false,
        }
    }

    /// Replay commands from a WAL file.
    fn replay_wal(&self, wal_path: &Path) -> Result<usize, String> {
        let file = File::open(wal_path).map_err(|e| format!("Failed to open WAL: {e}"))?;
        let reader = BufReader::new(file);

        let mut count = 0;
        for (line_num, line) in reader.lines().enumerate() {
            let cmd = line.map_err(|e| format!("Failed to read WAL line {}: {e}", line_num + 1))?;
            let cmd = cmd.trim();

            if cmd.is_empty() {
                continue;
            }

            if let Err(e) = self.router.execute_parsed(cmd) {
                return Err(format!("WAL replay failed at line {}: {e}", line_num + 1));
            }
            count += 1;
        }

        Ok(count)
    }

    /// Executes a single command and returns the result.
    pub fn execute(&mut self, input: &str) -> CommandResult {
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
            "wal status" => return self.handle_wal_status(),
            "wal truncate" => return self.handle_wal_truncate(),
            _ => {},
        }

        // Handle SAVE COMPRESSED command
        if lower.starts_with("save compressed") {
            return self.handle_save_compressed(trimmed);
        }

        // Handle SAVE command
        if lower.starts_with("save ") {
            return self.handle_save(trimmed);
        }

        // Handle LOAD command
        if lower.starts_with("load ") {
            return self.handle_load(trimmed);
        }

        // Handle VAULT INIT command
        if lower == "vault init" {
            return self.handle_vault_init();
        }

        // Handle VAULT IDENTITY command
        if lower.starts_with("vault identity") {
            return self.handle_vault_identity(trimmed);
        }

        // Handle CACHE INIT command
        if lower == "cache init" {
            return self.handle_cache_init();
        }

        // Execute as query
        match self.router.execute_parsed(trimmed) {
            Ok(result) => {
                // Log write commands to WAL after successful execution
                if Self::is_write_command(trimmed) {
                    if let Some(ref mut wal) = self.wal {
                        if let Err(e) = wal.append(trimmed) {
                            return CommandResult::Error(format!(
                                "Command succeeded but WAL write failed: {e}"
                            ));
                        }
                    }
                }
                CommandResult::Output(format_result(&result))
            },
            Err(e) => CommandResult::Error(format!("Error: {e}")),
        }
    }

    /// Handles the SAVE command.
    fn handle_save(&mut self, input: &str) -> CommandResult {
        let Some(p) = Self::extract_path(input, "save") else {
            return CommandResult::Error(
                "Usage: SAVE 'path/to/file.bin' or SAVE path/to/file.bin".to_string(),
            );
        };

        let store = self.router.vector().store();
        if let Err(e) = store.save_snapshot(&p) {
            return CommandResult::Error(format!("Failed to save: {e}"));
        }

        // Truncate WAL after successful save (snapshot now contains all data)
        if let Some(ref mut wal) = self.wal {
            if let Err(e) = wal.truncate() {
                return CommandResult::Error(format!(
                    "Saved snapshot but WAL truncate failed: {e}"
                ));
            }
        }

        CommandResult::Output(format!("Saved snapshot to: {p}"))
    }

    /// Handles the SAVE COMPRESSED command.
    fn handle_save_compressed(&mut self, input: &str) -> CommandResult {
        let Some(p) = Self::extract_path(input, "save compressed") else {
            return CommandResult::Error("Usage: SAVE COMPRESSED 'path/to/file.bin'".to_string());
        };

        let store = self.router.vector().store();
        let config = tensor_compress::CompressionConfig {
            vector_quantization: Some(tensor_compress::QuantMode::Int8),
            delta_encoding: true,
            rle_encoding: true,
        };

        if let Err(e) = store.save_snapshot_compressed(&p, config) {
            return CommandResult::Error(format!("Failed to save compressed: {e}"));
        }

        // Truncate WAL after successful save (snapshot now contains all data)
        if let Some(ref mut wal) = self.wal {
            if let Err(e) = wal.truncate() {
                return CommandResult::Error(format!(
                    "Saved snapshot but WAL truncate failed: {e}"
                ));
            }
        }

        CommandResult::Output(format!("Saved compressed snapshot to: {p}"))
    }

    /// Handles the LOAD command.
    fn handle_load(&mut self, input: &str) -> CommandResult {
        let Some(p) = Self::extract_path(input, "load") else {
            return CommandResult::Error(
                "Usage: LOAD 'path/to/file.bin' or LOAD path/to/file.bin".to_string(),
            );
        };

        // Try compressed format first (auto-detects via magic bytes), fall back to legacy
        let result =
            TensorStore::load_snapshot_compressed(&p).or_else(|_| TensorStore::load_snapshot(&p));

        match result {
            Ok(store) => {
                self.router = QueryRouter::with_shared_store(store);

                // Derive WAL path from snapshot path (e.g., data.bin -> data.log)
                let wal_path = Path::new(&p).with_extension("log");

                // Replay WAL if it exists
                let replay_msg = if wal_path.exists() {
                    match self.replay_wal(&wal_path) {
                        Ok(count) if count > 0 => {
                            format!("\nReplayed {count} commands from WAL")
                        },
                        Ok(_) => String::new(),
                        Err(e) => {
                            return CommandResult::Error(format!(
                                "Loaded snapshot but WAL replay failed: {e}"
                            ));
                        },
                    }
                } else {
                    String::new()
                };

                // Initialize WAL for new writes
                match Wal::open_append(&wal_path) {
                    Ok(wal) => {
                        self.wal = Some(wal);
                        CommandResult::Output(format!("Loaded snapshot from: {p}{replay_msg}"))
                    },
                    Err(e) => CommandResult::Error(format!(
                        "Loaded snapshot but failed to initialize WAL: {e}"
                    )),
                }
            },
            Err(e) => CommandResult::Error(format!("Failed to load: {e}")),
        }
    }

    /// Extracts the path from a SAVE or LOAD command.
    fn extract_path(input: &str, command: &str) -> Option<String> {
        let rest = input[command.len()..].trim();
        if rest.is_empty() {
            return None;
        }

        // Handle quoted path
        if (rest.starts_with('\'') && rest.ends_with('\''))
            || (rest.starts_with('"') && rest.ends_with('"'))
        {
            if rest.len() > 2 {
                return Some(rest[1..rest.len() - 1].to_string());
            }
            return None;
        }

        // Handle unquoted path
        Some(rest.to_string())
    }

    /// Handles the WAL STATUS command.
    fn handle_wal_status(&self) -> CommandResult {
        self.wal.as_ref().map_or_else(
            || {
                CommandResult::Output(
                    "WAL not active (use LOAD to enable WAL for a snapshot)".to_string(),
                )
            },
            |wal| {
                let size = wal.size().unwrap_or(0);
                CommandResult::Output(format!(
                    "WAL enabled\n  Path: {}\n  Size: {} bytes",
                    wal.path().display(),
                    size
                ))
            },
        )
    }

    /// Handles the WAL TRUNCATE command.
    fn handle_wal_truncate(&mut self) -> CommandResult {
        self.wal.as_mut().map_or_else(
            || {
                CommandResult::Error(
                    "WAL not active (use LOAD to enable WAL for a snapshot)".to_string(),
                )
            },
            |wal| match wal.truncate() {
                Ok(()) => CommandResult::Output("WAL truncated".to_string()),
                Err(e) => CommandResult::Error(format!("Failed to truncate WAL: {e}")),
            },
        )
    }

    /// Initialize vault from environment variable.
    fn handle_vault_init(&mut self) -> CommandResult {
        match std::env::var("NEUMANN_VAULT_KEY") {
            Ok(key) => {
                let decoded = match base64::Engine::decode(
                    &base64::engine::general_purpose::STANDARD,
                    &key,
                ) {
                    Ok(d) => d,
                    Err(e) => {
                        return CommandResult::Error(format!(
                            "Invalid base64 in NEUMANN_VAULT_KEY: {e}"
                        ))
                    },
                };

                match self.router.init_vault(&decoded) {
                    Ok(()) => CommandResult::Output("Vault initialized".to_string()),
                    Err(e) => CommandResult::Error(format!("Failed to initialize vault: {e}")),
                }
            },
            Err(_) => CommandResult::Error(
                "Set NEUMANN_VAULT_KEY environment variable (base64 encoded 32-byte key)"
                    .to_string(),
            ),
        }
    }

    /// Set current identity for vault access control.
    fn handle_vault_identity(&mut self, input: &str) -> CommandResult {
        let rest = input
            .to_lowercase()
            .strip_prefix("vault identity")
            .unwrap_or("")
            .trim()
            .to_string();

        // Extract identity from quoted string
        let identity = if rest.starts_with('\'') && rest.ends_with('\'') && rest.len() > 2 {
            &rest[1..rest.len() - 1]
        } else if !rest.is_empty() {
            &rest
        } else {
            return CommandResult::Output(format!(
                "Current identity: {}",
                self.router.current_identity()
            ));
        };

        self.router.set_identity(identity);
        CommandResult::Output(format!("Identity set to: {identity}"))
    }

    /// Initialize the cache with default configuration.
    fn handle_cache_init(&mut self) -> CommandResult {
        match self.router.init_cache_default() {
            Ok(()) => CommandResult::Output("Cache initialized".to_string()),
            Err(e) => CommandResult::Error(format!("Failed to initialize cache: {e}")),
        }
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

Persistence:
  save 'path'            Save database snapshot to file
  save compressed 'path' Save compressed snapshot (int8 quantization)
  load 'path'            Load snapshot and enable WAL
  wal status             Show write-ahead log status
  wal truncate           Clear the write-ahead log

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
    NODE LIST [label]              List all nodes or filter by label
    NODE GET id                    Get node by ID
    EDGE CREATE node1 -> node2 : label [{props}]
    EDGE LIST [type]               List all edges or filter by type
    EDGE GET id                    Get edge by ID
    NEIGHBORS node_id OUTGOING|INCOMING|BOTH [: label]
    PATH node1 -> node2 [LIMIT n]

  Vector:
    EMBED STORE 'key' [vector values]
    EMBED GET 'key'
    EMBED DELETE 'key'
    SIMILAR 'key' [COSINE|EUCLIDEAN|DOT_PRODUCT] LIMIT n
    SIMILAR [vector] [metric] LIMIT n

  Unified (Cross-Engine):
    FIND NODE [label] [WHERE condition] [LIMIT n]
    FIND EDGE [type] [WHERE condition] [LIMIT n]

  Blob Storage:
    BLOB PUT 'path' [CHUNK size] [TAGS 'a','b'] [FOR 'entity']
    BLOB GET 'id' TO 'path'        Download blob to file
    BLOB DELETE 'id'               Delete blob
    BLOB INFO 'id'                 Show blob metadata
    BLOB LINK 'id' TO 'entity'     Link blob to entity
    BLOB UNLINK 'id' FROM 'entity' Unlink blob from entity
    BLOB TAG 'id' 'tag'            Add tag to blob
    BLOB UNTAG 'id' 'tag'          Remove tag from blob
    BLOBS                          List all blobs
    BLOBS FOR 'entity'             List blobs linked to entity
    BLOBS BY TAG 'tag'             Find blobs by tag

  Vault (Secrets):
    VAULT INIT                     Initialize vault from NEUMANN_VAULT_KEY
    VAULT IDENTITY 'node:name'     Set current identity for access control
    VAULT SET 'key' 'value'        Store encrypted secret
    VAULT GET 'key'                Retrieve secret (requires access)
    VAULT DELETE 'key'             Delete secret
    VAULT LIST 'pattern'           List accessible secrets
    VAULT ROTATE 'key' 'new'       Rotate secret value
    VAULT GRANT 'entity' ON 'key'  Grant access to entity
    VAULT REVOKE 'entity' ON 'key' Revoke access from entity

  Cache (LLM Responses):
    CACHE INIT                     Initialize semantic cache
    CACHE STATS                    Show cache statistics
    CACHE CLEAR                    Clear all cache entries
    CACHE EVICT [n]                Evict n entries (default: 100)
    CACHE GET 'key'                Get cached response
    CACHE PUT 'key' 'value'        Store cache entry

  Checkpoints (Rollback):
    CHECKPOINT                     Create checkpoint with auto-generated name
    CHECKPOINT 'name'              Create named checkpoint
    CHECKPOINTS                    List all checkpoints
    CHECKPOINTS LIMIT n            List last n checkpoints
    ROLLBACK TO 'name-or-id'       Restore database to checkpoint

Examples:
  > CREATE TABLE users (id INT, name TEXT)
  > INSERT INTO users VALUES (1, 'Alice')
  > SELECT * FROM users
  > NODE CREATE person {name: 'Bob', age: 30}
  > EMBED STORE 'doc1' [0.1, 0.2, 0.3]
  > SAVE 'backup.bin'
  > SAVE COMPRESSED 'backup.bin'
  > LOAD 'backup.bin'
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
    pub fn run(&mut self) -> Result<(), ShellError> {
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
        QueryResult::Blob(data) => format_blob(data),
        QueryResult::ArtifactInfo(info) => format_artifact_info(info),
        QueryResult::ArtifactList(ids) => format_artifact_list(ids),
        QueryResult::BlobStats(stats) => format_blob_stats(stats),
        QueryResult::CheckpointList(checkpoints) => format_checkpoint_list(checkpoints),
        QueryResult::Chain(chain) => format_chain_result(chain),
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

/// Formats blob data for display.
fn format_blob(data: &[u8]) -> String {
    let size = data.len();
    if size <= 256 {
        // Try to display as UTF-8 if valid, otherwise show hex
        if let Ok(s) = std::str::from_utf8(data) {
            if s.chars().all(|c| !c.is_control() || c == '\n' || c == '\t') {
                return s.to_string();
            }
        }
    }
    // Show summary for binary/large data
    format!("<binary data: {size} bytes>")
}

/// Formats artifact info for display.
fn format_artifact_info(info: &query_router::ArtifactInfoResult) -> String {
    let mut lines = vec![
        format!("Artifact: {}", info.id),
        format!("  Filename: {}", info.filename),
        format!("  Type: {}", info.content_type),
        format!("  Size: {} bytes", info.size),
        format!("  Checksum: {}", info.checksum),
        format!("  Chunks: {}", info.chunk_count),
        format!("  Created: {}", info.created),
        format!("  Modified: {}", info.modified),
        format!("  Creator: {}", info.created_by),
    ];

    if !info.tags.is_empty() {
        lines.push(format!("  Tags: {}", info.tags.join(", ")));
    }

    if !info.linked_to.is_empty() {
        lines.push(format!("  Links: {}", info.linked_to.join(", ")));
    }

    if !info.custom.is_empty() {
        lines.push("  Metadata:".to_string());
        for (k, v) in &info.custom {
            lines.push(format!("    {k}: {v}"));
        }
    }

    lines.join("\n")
}

/// Formats artifact list for display.
fn format_artifact_list(ids: &[String]) -> String {
    if ids.is_empty() {
        "(no artifacts)".to_string()
    } else {
        ids.join("\n")
    }
}

/// Formats blob statistics for display.
fn format_blob_stats(stats: &query_router::BlobStatsResult) -> String {
    format!(
        "Blob Storage Statistics:\n\
         Artifacts: {}\n\
         Chunks: {}\n\
         Total bytes: {}\n\
         Unique bytes: {}\n\
         Dedup ratio: {:.1}%\n\
         Orphaned chunks: {}",
        stats.artifact_count,
        stats.chunk_count,
        stats.total_bytes,
        stats.unique_bytes,
        stats.dedup_ratio * 100.0,
        stats.orphaned_chunks
    )
}

fn format_checkpoint_list(checkpoints: &[CheckpointInfo]) -> String {
    if checkpoints.is_empty() {
        return "No checkpoints found".to_string();
    }

    let mut output = String::new();
    let _ = writeln!(output, "Checkpoints:");
    let _ = writeln!(output, "{:<40} {:<30} {:<20} Type", "ID", "Name", "Created");
    let _ = writeln!(output, "{}", "-".repeat(100));

    for cp in checkpoints {
        let created = format_timestamp(cp.created_at);
        let cp_type = if cp.is_auto { "auto" } else { "manual" };
        let _ = writeln!(
            output,
            "{:<40} {:<30} {:<20} {}",
            &cp.id[..cp.id.len().min(36)],
            &cp.name[..cp.name.len().min(28)],
            created,
            cp_type
        );
    }

    output.trim_end().to_string()
}

fn format_chain_result(result: &ChainResult) -> String {
    match result {
        ChainResult::TransactionBegun { tx_id } => {
            format!("Chain transaction started: {tx_id}")
        },
        ChainResult::Committed { block_hash, height } => {
            format!("Committed block {block_hash} at height {height}")
        },
        ChainResult::RolledBack { to_height } => {
            format!("Chain rolled back to height {to_height}")
        },
        ChainResult::History(entries) => format_chain_history(entries),
        ChainResult::Similar(results) => format_chain_similar(results),
        ChainResult::Drift(drift) => format_chain_drift(drift),
        ChainResult::Height(h) => format!("Chain height: {h}"),
        ChainResult::Tip { hash, height } => {
            format!("Chain tip: {hash} at height {height}")
        },
        ChainResult::Block(info) => format_chain_block(info),
        ChainResult::Codebook(info) => format_chain_codebook(info),
        ChainResult::Verified { ok, errors } => {
            if *ok {
                "Chain verified: OK".to_string()
            } else {
                let mut output = "Chain verification failed:\n".to_string();
                for err in errors {
                    let _ = writeln!(output, "  - {err}");
                }
                output.trim_end().to_string()
            }
        },
        ChainResult::TransitionAnalysis(analysis) => format_chain_transitions(analysis),
    }
}

fn format_chain_history(entries: &[ChainHistoryEntry]) -> String {
    if entries.is_empty() {
        return "No history found for key".to_string();
    }

    let mut output = String::new();
    let _ = writeln!(output, "Chain History:");
    let _ = writeln!(output, "{:<10} {:<30}", "Height", "Transaction");
    let _ = writeln!(output, "{}", "-".repeat(50));

    for entry in entries {
        let _ = writeln!(output, "{:<10} {}", entry.height, entry.transaction_type);
    }

    output.trim_end().to_string()
}

fn format_chain_similar(results: &[ChainSimilarResult]) -> String {
    if results.is_empty() {
        return "No similar blocks found".to_string();
    }

    let mut output = String::new();
    let _ = writeln!(output, "Similar Blocks:");
    let _ = writeln!(
        output,
        "{:<10} {:<66} {:<10}",
        "Height", "Hash", "Similarity"
    );
    let _ = writeln!(output, "{}", "-".repeat(90));

    for r in results {
        let _ = writeln!(
            output,
            "{:<10} {:<66} {:.4}",
            r.height, r.block_hash, r.similarity
        );
    }

    output.trim_end().to_string()
}

fn format_chain_drift(drift: &ChainDriftResult) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Chain Drift Analysis:");
    let _ = writeln!(output, "  From height:        {}", drift.from_height);
    let _ = writeln!(output, "  To height:          {}", drift.to_height);
    let _ = writeln!(output, "  Total drift:        {:.4}", drift.total_drift);
    let _ = writeln!(
        output,
        "  Avg drift/block:    {:.4}",
        drift.avg_drift_per_block
    );
    let _ = writeln!(output, "  Max drift:          {:.4}", drift.max_drift);
    output.trim_end().to_string()
}

fn format_chain_block(info: &ChainBlockInfo) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Block Info:");
    let _ = writeln!(output, "  Height:       {}", info.height);
    let _ = writeln!(output, "  Hash:         {}", info.hash);
    let _ = writeln!(output, "  Prev Hash:    {}", info.prev_hash);
    let _ = writeln!(output, "  Timestamp:    {}", info.timestamp);
    let _ = writeln!(output, "  Transactions: {}", info.transaction_count);
    let _ = writeln!(output, "  Proposer:     {}", info.proposer);
    output.trim_end().to_string()
}

fn format_chain_codebook(info: &ChainCodebookInfo) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Codebook Info:");
    let _ = writeln!(output, "  Scope:      {}", info.scope);
    let _ = writeln!(output, "  Entries:    {}", info.entry_count);
    let _ = writeln!(output, "  Dimension:  {}", info.dimension);
    if let Some(domain) = &info.domain {
        let _ = writeln!(output, "  Domain:     {domain}");
    }
    output.trim_end().to_string()
}

fn format_chain_transitions(analysis: &ChainTransitionAnalysis) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "Transition Analysis:");
    let _ = writeln!(
        output,
        "  Total transitions:   {}",
        analysis.total_transitions
    );
    let _ = writeln!(
        output,
        "  Valid transitions:   {}",
        analysis.valid_transitions
    );
    let _ = writeln!(
        output,
        "  Invalid transitions: {}",
        analysis.invalid_transitions
    );
    let _ = writeln!(
        output,
        "  Avg validity score:  {:.4}",
        analysis.avg_validity_score
    );
    output.trim_end().to_string()
}

fn format_timestamp(unix_secs: u64) -> String {
    // Format as relative time for better readability
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    if unix_secs == 0 {
        return "unknown".to_string();
    }

    let diff = now.saturating_sub(unix_secs);

    if diff < 60 {
        format!("{diff}s ago")
    } else if diff < 3600 {
        let mins = diff / 60;
        format!("{mins}m ago")
    } else if diff < 86400 {
        let hours = diff / 3600;
        format!("{hours}h ago")
    } else {
        let days = diff / 86400;
        format!("{days}d ago")
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
        let mut shell = Shell::new();
        assert_eq!(shell.execute(""), CommandResult::Empty);
        assert_eq!(shell.execute("   "), CommandResult::Empty);
        assert_eq!(shell.execute("\t\n"), CommandResult::Empty);
    }

    #[test]
    fn test_exit_commands() {
        let mut shell = Shell::new();
        assert_eq!(shell.execute("exit"), CommandResult::Exit);
        assert_eq!(shell.execute("quit"), CommandResult::Exit);
        assert_eq!(shell.execute("\\q"), CommandResult::Exit);
        assert_eq!(shell.execute("EXIT"), CommandResult::Exit);
        assert_eq!(shell.execute("QUIT"), CommandResult::Exit);
    }

    #[test]
    fn test_help_commands() {
        let mut shell = Shell::new();

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
        let mut shell = Shell::new();
        let result = shell.execute("clear");
        assert!(matches!(result, CommandResult::Output(_)));

        let result = shell.execute("\\c");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_create_table() {
        let mut shell = Shell::new();
        let result = shell.execute("CREATE TABLE users (id INT, name TEXT)");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_insert_and_select() {
        let mut shell = Shell::new();

        let _ = shell.execute("CREATE TABLE test (id INT, value TEXT)");
        let _ = shell.execute("INSERT INTO test VALUES (1, 'hello')");
        let _ = shell.execute("INSERT INTO test VALUES (2, 'world')");

        let result = shell.execute("SELECT * FROM test");
        // Just check we get an Output result
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_select_empty_table() {
        let mut shell = Shell::new();
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
        let mut shell = Shell::new();
        let result = shell.execute("NODE CREATE person {name: 'Alice', age: 30}");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_edge_create() {
        let mut shell = Shell::new();
        let _ = shell.execute("NODE CREATE person {name: 'Alice'}");
        let _ = shell.execute("NODE CREATE person {name: 'Bob'}");

        let result = shell.execute("EDGE CREATE 1 -> 2 : knows");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_neighbors() {
        let mut shell = Shell::new();
        let _ = shell.execute("NODE CREATE person {name: 'Alice'}");
        let _ = shell.execute("NODE CREATE person {name: 'Bob'}");
        let _ = shell.execute("EDGE CREATE 1 -> 2 : knows");

        let result = shell.execute("NEIGHBORS 1 OUTGOING");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_embed_store_and_get() {
        let mut shell = Shell::new();

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
        let mut shell = Shell::new();

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
        let mut shell = Shell::new();
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
        let mut shell = Shell::new();
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
        let mut shell = Shell::new();
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
        let mut shell = Shell::new();
        // Both "tables" and "\dt" should work
        let result1 = shell.execute("tables");
        let result2 = shell.execute("\\dt");
        assert!(matches!(result1, CommandResult::Output(_)));
        assert!(matches!(result2, CommandResult::Output(_)));
    }

    #[test]
    fn test_case_insensitive_commands() {
        let mut shell = Shell::new();
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

    // Save and Load command tests

    #[test]
    fn test_save_command() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'test_key' [1.0, 2.0, 3.0]");

        let path = std::env::temp_dir().join("test_shell_save.bin");
        let result = shell.execute(&format!("SAVE '{}'", path.display()));

        if let CommandResult::Output(output) = result {
            assert!(output.contains("Saved snapshot"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_command() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'test_key' [1.0, 2.0, 3.0]");

        let path = std::env::temp_dir().join("test_shell_load.bin");
        let _ = shell.execute(&format!("SAVE '{}'", path.display()));

        // Load into a fresh shell
        let mut shell2 = Shell::new();
        let result = shell2.execute(&format!("LOAD '{}'", path.display()));

        if let CommandResult::Output(output) = result {
            assert!(output.contains("Loaded snapshot"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }

        // Verify the data is accessible
        let result = shell2.execute("EMBED GET 'test_key'");
        assert!(matches!(result, CommandResult::Output(_)));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_without_path() {
        let mut shell = Shell::new();
        let result = shell.execute("SAVE");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_load_without_path() {
        let mut shell = Shell::new();
        let result = shell.execute("LOAD");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let mut shell = Shell::new();
        let result = shell.execute("LOAD '/nonexistent/path/file.bin'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_save_unquoted_path() {
        let mut shell = Shell::new();
        let path = std::env::temp_dir().join("test_shell_unquoted.bin");
        let result = shell.execute(&format!("SAVE {}", path.display()));

        if let CommandResult::Output(output) = result {
            assert!(output.contains("Saved snapshot"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_double_quoted_path() {
        let mut shell = Shell::new();
        let path = std::env::temp_dir().join("test_shell_dblquote.bin");
        let result = shell.execute(&format!("SAVE \"{}\"", path.display()));

        if let CommandResult::Output(output) = result {
            assert!(output.contains("Saved snapshot"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_extract_path_quoted() {
        assert_eq!(
            Shell::extract_path("save 'foo.bin'", "save"),
            Some("foo.bin".to_string())
        );
        assert_eq!(
            Shell::extract_path("LOAD \"bar.bin\"", "LOAD"),
            Some("bar.bin".to_string())
        );
    }

    #[test]
    fn test_extract_path_unquoted() {
        assert_eq!(
            Shell::extract_path("save /path/to/file.bin", "save"),
            Some("/path/to/file.bin".to_string())
        );
    }

    #[test]
    fn test_extract_path_empty() {
        assert_eq!(Shell::extract_path("save ", "save"), None);
        assert_eq!(Shell::extract_path("save", "save"), None);
    }

    #[test]
    fn test_extract_path_empty_quotes() {
        assert_eq!(Shell::extract_path("save ''", "save"), None);
        assert_eq!(Shell::extract_path("save \"\"", "save"), None);
    }

    #[test]
    fn test_help_contains_save_load() {
        let help = Shell::help_text();
        assert!(help.contains("save"));
        assert!(help.contains("load"));
        assert!(help.contains("snapshot"));
    }

    #[test]
    fn test_save_load_case_insensitive() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'key' [1.0]");

        let path = std::env::temp_dir().join("test_shell_case.bin");

        // Test uppercase SAVE
        let result = shell.execute(&format!("SAVE '{}'", path.display()));
        assert!(matches!(result, CommandResult::Output(_)));

        // Test lowercase load
        let mut shell2 = Shell::new();
        let result = shell2.execute(&format!("load '{}'", path.display()));
        assert!(matches!(result, CommandResult::Output(_)));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_compressed_command() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'test_key' [1.0, 2.0, 3.0, 4.0]");

        let path = std::env::temp_dir().join("test_shell_save_compressed.bin");
        let result = shell.execute(&format!("SAVE COMPRESSED '{}'", path.display()));

        if let CommandResult::Output(output) = result {
            assert!(output.contains("Saved compressed snapshot"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_compressed_without_path() {
        let mut shell = Shell::new();
        let result = shell.execute("SAVE COMPRESSED");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_save_compressed_and_load() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'compressed_key' [0.1, 0.2, 0.3, 0.4]");

        let path = std::env::temp_dir().join("test_shell_compressed_load.bin");
        let _ = shell.execute(&format!("SAVE COMPRESSED '{}'", path.display()));

        // Load into a fresh shell
        let mut shell2 = Shell::new();
        let result = shell2.execute(&format!("LOAD '{}'", path.display()));
        assert!(matches!(result, CommandResult::Output(_)));

        // Verify the data is accessible (embedding restored from int8 quantization)
        let result = shell2.execute("EMBED GET 'compressed_key'");
        assert!(matches!(result, CommandResult::Output(_)));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_compressed_case_insensitive() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'key' [1.0, 2.0]");

        let path = std::env::temp_dir().join("test_shell_compressed_case.bin");

        // Test lowercase
        let result = shell.execute(&format!("save compressed '{}'", path.display()));
        assert!(matches!(result, CommandResult::Output(_)));

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_help_contains_save_compressed() {
        let help = Shell::help_text();
        assert!(help.contains("save compressed"));
        assert!(help.contains("SAVE COMPRESSED"));
    }

    // WAL tests

    #[test]
    fn test_is_write_command() {
        // Write commands
        assert!(Shell::is_write_command(
            "INSERT INTO users VALUES (1, 'Alice')"
        ));
        assert!(Shell::is_write_command("UPDATE users SET name = 'Bob'"));
        assert!(Shell::is_write_command("DELETE FROM users"));
        assert!(Shell::is_write_command("CREATE TABLE test (id INT)"));
        assert!(Shell::is_write_command("DROP TABLE test"));
        assert!(Shell::is_write_command(
            "NODE CREATE person {name: 'Alice'}"
        ));
        assert!(Shell::is_write_command("NODE DELETE 1"));
        assert!(Shell::is_write_command("EDGE CREATE 1 -> 2 : knows"));
        assert!(Shell::is_write_command("EMBED STORE 'key' [1.0, 2.0]"));
        assert!(Shell::is_write_command("EMBED DELETE 'key'"));

        // Read-only commands
        assert!(!Shell::is_write_command("SELECT * FROM users"));
        assert!(!Shell::is_write_command("NODE GET 1"));
        assert!(!Shell::is_write_command("EDGE GET 1"));
        assert!(!Shell::is_write_command("EMBED GET 'key'"));
        assert!(!Shell::is_write_command("SIMILAR 'key' LIMIT 5"));
        assert!(!Shell::is_write_command("NEIGHBORS 1 OUTGOING"));
        assert!(!Shell::is_write_command("SHOW TABLES"));
    }

    #[test]
    fn test_wal_status_no_wal() {
        let mut shell = Shell::new();
        let result = shell.execute("wal status");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("WAL not active"));
        } else {
            panic!("Expected Output");
        }
    }

    #[test]
    fn test_wal_truncate_no_wal() {
        let mut shell = Shell::new();
        let result = shell.execute("wal truncate");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_wal_enabled_after_load() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'key' [1.0, 2.0, 3.0]");

        let path = std::env::temp_dir().join("test_wal_enabled.bin");
        let _ = shell.execute(&format!("SAVE '{}'", path.display()));

        // Load into fresh shell
        let mut shell2 = Shell::new();
        let _ = shell2.execute(&format!("LOAD '{}'", path.display()));

        // WAL should now be active
        let result = shell2.execute("wal status");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("WAL enabled"));
            assert!(output.contains(".log"));
        } else {
            panic!("Expected Output");
        }

        // Cleanup
        let wal_path = path.with_extension("log");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_logs_write_commands() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'key' [1.0]");

        let path = std::env::temp_dir().join("test_wal_logs.bin");
        let _ = shell.execute(&format!("SAVE '{}'", path.display()));

        // Load to enable WAL
        let mut shell2 = Shell::new();
        let _ = shell2.execute(&format!("LOAD '{}'", path.display()));

        // Execute some write commands
        let _ = shell2.execute("EMBED STORE 'key2' [2.0, 3.0]");
        let _ = shell2.execute("CREATE TABLE test (id INT)");

        // Check WAL status shows non-zero size
        let result = shell2.execute("wal status");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("WAL enabled"));
        } else {
            panic!("Expected Output");
        }

        // Cleanup
        let wal_path = path.with_extension("log");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_replay_on_load() {
        use std::time::{SystemTime, UNIX_EPOCH};

        // Use unique path with timestamp to avoid test interference
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("test_wal_replay_{ts}.bin"));
        let wal_path = path.with_extension("log");

        // Clean up any leftover files
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);

        // Step 1: Create initial data and save (using EMBED which is known to work)
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'key1' [1.0, 2.0, 3.0]");
        let _ = shell.execute(&format!("SAVE '{}'", path.display()));

        // Step 2: Load to enable WAL
        let mut shell2 = Shell::new();
        let result = shell2.execute(&format!("LOAD '{}'", path.display()));
        assert!(matches!(result, CommandResult::Output(_)), "Load failed");

        // WAL file should now exist
        assert!(wal_path.exists(), "WAL file should exist after LOAD");

        // Step 3: Add more data (will be written to WAL)
        let result = shell2.execute("EMBED STORE 'key2' [4.0, 5.0, 6.0]");
        assert!(
            matches!(result, CommandResult::Output(_)),
            "EMBED STORE failed: {:?}",
            result
        );

        // Verify WAL has content
        let wal_size = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
        assert!(wal_size > 0, "WAL should have content after EMBED STORE");

        // Step 4: Simulate crash by creating new shell and loading
        let mut shell3 = Shell::new();
        let result = shell3.execute(&format!("LOAD '{}'", path.display()));

        if let CommandResult::Output(output) = result {
            assert!(
                output.contains("Loaded snapshot"),
                "Expected 'Loaded snapshot'"
            );
            assert!(
                output.contains("Replayed"),
                "Expected 'Replayed' in output: {output}"
            );
        } else {
            panic!("Expected Output, got {:?}", result);
        }

        // Verify key2 is present (recovered from WAL)
        let result = shell3.execute("EMBED GET 'key2'");
        assert!(
            matches!(result, CommandResult::Output(_)),
            "key2 should exist after WAL replay"
        );

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_truncate_after_save() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'key' [1.0]");

        let path = std::env::temp_dir().join("test_wal_truncate_save.bin");
        let _ = shell.execute(&format!("SAVE '{}'", path.display()));

        // Load to enable WAL
        let mut shell2 = Shell::new();
        let _ = shell2.execute(&format!("LOAD '{}'", path.display()));

        // Add data to WAL
        let _ = shell2.execute("EMBED STORE 'key2' [2.0]");

        // Save again - should truncate WAL
        let _ = shell2.execute(&format!("SAVE '{}'", path.display()));

        // WAL should be empty (0 bytes or just created)
        let result = shell2.execute("wal status");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("0 bytes") || output.contains("WAL enabled"));
        }

        // Cleanup
        let wal_path = path.with_extension("log");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_truncate_command() {
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'key' [1.0]");

        let path = std::env::temp_dir().join("test_wal_truncate_cmd.bin");
        let _ = shell.execute(&format!("SAVE '{}'", path.display()));

        // Load to enable WAL
        let mut shell2 = Shell::new();
        let _ = shell2.execute(&format!("LOAD '{}'", path.display()));

        // Add data to WAL
        let _ = shell2.execute("EMBED STORE 'key2' [2.0]");

        // Manually truncate WAL
        let result = shell2.execute("wal truncate");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("WAL truncated"));
        } else {
            panic!("Expected Output");
        }

        // Cleanup
        let wal_path = path.with_extension("log");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_help_contains_wal() {
        let help = Shell::help_text();
        assert!(help.contains("wal status"));
        assert!(help.contains("wal truncate"));
        assert!(help.contains("write-ahead log"));
    }

    #[test]
    fn test_wal_does_not_log_reads() {
        let mut shell = Shell::new();
        let _ = shell.execute("CREATE TABLE test (id INT)");
        let _ = shell.execute("INSERT INTO test VALUES (1)");

        let path = std::env::temp_dir().join("test_wal_no_reads.bin");
        let _ = shell.execute(&format!("SAVE '{}'", path.display()));

        // Load to enable WAL
        let mut shell2 = Shell::new();
        let _ = shell2.execute(&format!("LOAD '{}'", path.display()));

        // Execute read-only commands
        let _ = shell2.execute("SELECT * FROM test");
        let _ = shell2.execute("SHOW TABLES");

        // WAL should be empty (0 bytes)
        let wal_path = path.with_extension("log");
        let size = std::fs::metadata(&wal_path).map(|m| m.len()).unwrap_or(0);
        assert_eq!(size, 0);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_case_insensitive_commands() {
        let mut shell = Shell::new();
        assert_eq!(
            shell.execute("WAL STATUS"),
            CommandResult::Output(
                "WAL not active (use LOAD to enable WAL for a snapshot)".to_string()
            )
        );
    }

    #[test]
    fn test_wal_replay_with_empty_lines() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("test_wal_empty_lines_{ts}.bin"));
        let wal_path = path.with_extension("log");

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);

        // Create initial snapshot
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'key1' [1.0]");
        let _ = shell.execute(&format!("SAVE '{}'", path.display()));

        // Write WAL with empty lines manually
        std::fs::write(
            &wal_path,
            "EMBED STORE 'key2' [2.0]\n\n  \nEMBED STORE 'key3' [3.0]\n",
        )
        .unwrap();

        // Load and replay (should skip empty lines)
        let mut shell2 = Shell::new();
        let result = shell2.execute(&format!("LOAD '{}'", path.display()));
        if let CommandResult::Output(output) = result {
            assert!(output.contains("Replayed 2 commands"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_replay_with_invalid_command() {
        use std::time::{SystemTime, UNIX_EPOCH};

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("test_wal_invalid_{ts}.bin"));
        let wal_path = path.with_extension("log");

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);

        // Create initial snapshot
        let mut shell = Shell::new();
        let _ = shell.execute("EMBED STORE 'key1' [1.0]");
        let _ = shell.execute(&format!("SAVE '{}'", path.display()));

        // Write WAL with invalid command
        std::fs::write(&wal_path, "INVALID COMMAND SYNTAX\n").unwrap();

        // Load should fail during WAL replay
        let mut shell2 = Shell::new();
        let result = shell2.execute(&format!("LOAD '{}'", path.display()));
        assert!(
            matches!(result, CommandResult::Error(_)),
            "Expected Error, got {:?}",
            result
        );

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&wal_path);
    }

    // Vault tests

    #[test]
    fn test_help_contains_vault() {
        let help = Shell::help_text();
        assert!(help.contains("VAULT INIT"));
        assert!(help.contains("VAULT IDENTITY"));
        assert!(help.contains("VAULT SET"));
        assert!(help.contains("VAULT GET"));
        assert!(help.contains("VAULT DELETE"));
        assert!(help.contains("VAULT LIST"));
        assert!(help.contains("VAULT ROTATE"));
        assert!(help.contains("VAULT GRANT"));
        assert!(help.contains("VAULT REVOKE"));
    }

    #[test]
    fn test_vault_init_without_env() {
        let mut shell = Shell::new();
        // Ensure NEUMANN_VAULT_KEY is not set
        std::env::remove_var("NEUMANN_VAULT_KEY");

        let result = shell.execute("VAULT INIT");
        if let CommandResult::Error(msg) = result {
            assert!(msg.contains("NEUMANN_VAULT_KEY"));
        } else {
            panic!("Expected Error, got {:?}", result);
        }
    }

    #[test]
    fn test_vault_identity_show_current() {
        let mut shell = Shell::new();
        let result = shell.execute("VAULT IDENTITY");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("Current identity:"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }
    }

    #[test]
    fn test_vault_identity_set() {
        let mut shell = Shell::new();
        let result = shell.execute("VAULT IDENTITY 'node:alice'");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("Identity set to:"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }
    }

    #[test]
    fn test_is_write_command_vault() {
        // Write commands for vault
        assert!(Shell::is_write_command("VAULT SET 'key' 'value'"));
        assert!(Shell::is_write_command("VAULT DELETE 'key'"));
        assert!(Shell::is_write_command("VAULT ROTATE 'key' 'new'"));
        assert!(Shell::is_write_command("VAULT GRANT 'entity' ON 'key'"));
        assert!(Shell::is_write_command("VAULT REVOKE 'entity' ON 'key'"));

        // Read-only commands for vault
        assert!(!Shell::is_write_command("VAULT GET 'key'"));
        assert!(!Shell::is_write_command("VAULT LIST '*'"));
        assert!(!Shell::is_write_command("VAULT INIT"));
        assert!(!Shell::is_write_command("VAULT IDENTITY 'node:alice'"));
    }

    #[test]
    fn test_is_write_command_cache() {
        // Write command for cache
        assert!(Shell::is_write_command("CACHE CLEAR"));

        // Read-only commands for cache
        assert!(!Shell::is_write_command("CACHE INIT"));
        assert!(!Shell::is_write_command("CACHE STATS"));
    }

    #[test]
    fn test_cache_init() {
        let mut shell = Shell::new();
        let result = shell.execute("CACHE INIT");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("Cache initialized"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }
    }

    #[test]
    fn test_cache_stats() {
        let mut shell = Shell::new();
        // Initialize cache first
        shell.execute("CACHE INIT");
        let result = shell.execute("CACHE STATS");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("Cache Statistics"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }
    }

    #[test]
    fn test_help_contains_cache() {
        let help = Shell::help_text();
        assert!(help.contains("CACHE INIT"));
        assert!(help.contains("CACHE STATS"));
        assert!(help.contains("CACHE CLEAR"));
    }

    // ========== Checkpoint Tests ==========

    #[test]
    fn test_help_contains_checkpoint() {
        let help = Shell::help_text();
        assert!(help.contains("CHECKPOINT"));
        assert!(help.contains("CHECKPOINTS"));
        assert!(help.contains("ROLLBACK TO"));
        assert!(help.contains("Checkpoints (Rollback)"));
    }

    #[test]
    fn test_format_checkpoint_list_empty() {
        let checkpoints: Vec<CheckpointInfo> = vec![];
        let output = format_result(&QueryResult::CheckpointList(checkpoints));
        assert!(output.contains("No checkpoints found"));
    }

    #[test]
    fn test_format_checkpoint_list_with_data() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let checkpoints = vec![
            CheckpointInfo {
                id: "cp-123".to_string(),
                name: "manual-checkpoint".to_string(),
                created_at: now,
                is_auto: false,
            },
            CheckpointInfo {
                id: "cp-456".to_string(),
                name: "auto-before-DELETE".to_string(),
                created_at: now - 60,
                is_auto: true,
            },
        ];
        let output = format_result(&QueryResult::CheckpointList(checkpoints));
        assert!(output.contains("Checkpoints:"));
        assert!(output.contains("cp-123"));
        assert!(output.contains("manual-checkpoint"));
        assert!(output.contains("manual"));
        assert!(output.contains("auto"));
    }

    #[test]
    fn test_checkpoint_create() {
        let mut shell = Shell::new();
        // Initialize blob and checkpoint manager
        shell.router_mut().init_blob().unwrap();
        shell.router_mut().init_checkpoint().unwrap();

        let result = shell.execute("CHECKPOINT 'test-checkpoint'");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("Checkpoint created"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }
    }

    #[test]
    fn test_checkpoint_list() {
        let mut shell = Shell::new();
        shell.router_mut().init_blob().unwrap();
        shell.router_mut().init_checkpoint().unwrap();

        shell.execute("CHECKPOINT 'first'");
        shell.execute("CHECKPOINT 'second'");

        let result = shell.execute("CHECKPOINTS");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("Checkpoints:"));
            assert!(output.contains("first"));
            assert!(output.contains("second"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }
    }

    #[test]
    fn test_checkpoint_rollback() {
        let mut shell = Shell::new();
        shell.router_mut().init_blob().unwrap();
        shell.router_mut().init_checkpoint().unwrap();

        // Store some data
        shell.execute("EMBED STORE 'rollback-test' [1.0, 2.0, 3.0]");

        // Create checkpoint
        shell.execute("CHECKPOINT 'before-delete'");

        // Delete the data
        shell.execute("EMBED DELETE 'rollback-test'");

        // Rollback
        let result = shell.execute("ROLLBACK TO 'before-delete'");
        if let CommandResult::Output(output) = result {
            assert!(output.contains("Rolled back"));
        } else {
            panic!("Expected Output, got {:?}", result);
        }

        // Verify data is restored
        let result = shell.execute("EMBED GET 'rollback-test'");
        assert!(matches!(result, CommandResult::Output(_)));
    }
}
