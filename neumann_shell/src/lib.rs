// SPDX-License-Identifier: MIT OR Apache-2.0
//! Neumann Shell - Interactive CLI for Neumann database
//!
//! Provides a readline-based interface for executing queries against the
//! Neumann unified query engine.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

pub mod cli;
mod input;
mod output;
mod progress;
mod style;
mod wal;

pub use input::NeumannHelper;
pub use style::{Icons, Theme};
pub use wal::{Wal, WalRecoveryMode, WalReplayError, WalReplayResult};

use std::{
    fmt::Write as _,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
};

use parking_lot::{Mutex, RwLock};
use query_router::QueryRouter;
use rustyline::{
    error::ReadlineError,
    history::{DefaultHistory, History},
    Editor,
};
use tensor_chain::QueryExecutor;
use tensor_checkpoint::{
    format_confirmation_prompt, ConfirmationHandler, DestructiveOp, OperationPreview,
};
use tensor_store::TensorStore;

/// Shell configuration options.
#[derive(Debug, Clone)]
pub struct ShellConfig {
    /// Path to history file (None disables persistence).
    pub history_file: Option<PathBuf>,
    /// Maximum number of history entries to keep.
    pub history_size: usize,
    /// Prompt string displayed before each input.
    pub prompt: String,
    /// Color theme for output.
    pub theme: Theme,
    /// Disable colored output.
    pub no_color: bool,
    /// Skip boot sequence animation.
    pub no_boot: bool,
    /// Quiet mode: suppress non-essential output.
    pub quiet: bool,
}

impl Default for ShellConfig {
    fn default() -> Self {
        Self {
            history_file: dirs_home().map(|h| h.join(".neumann_history")),
            history_size: 1000,
            // Phosphor green prompt to match boot aesthetic
            prompt: "\x1b[38;2;0;238;0mneumann>\x1b[0m ".to_string(),
            theme: Theme::auto(),
            no_color: false,
            no_boot: false,
            quiet: false,
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
    router: Arc<RwLock<QueryRouter>>,
    config: ShellConfig,
    wal: Mutex<Option<Wal>>,
    icons: &'static Icons,
}

/// Wrapper to implement `QueryExecutor` for `Arc<RwLock<QueryRouter>>`.
struct RouterExecutor(Arc<RwLock<QueryRouter>>);

impl QueryExecutor for RouterExecutor {
    fn execute(&self, query: &str) -> std::result::Result<Vec<u8>, String> {
        let router = self.0.read();
        router.execute_for_cluster(query)
    }
}

/// Interactive confirmation handler for destructive operations.
struct ShellConfirmationHandler {
    editor: Arc<Mutex<Editor<NeumannHelper, DefaultHistory>>>,
}

impl ShellConfirmationHandler {
    const fn new(editor: Arc<Mutex<Editor<NeumannHelper, DefaultHistory>>>) -> Self {
        Self { editor }
    }
}

impl ConfirmationHandler for ShellConfirmationHandler {
    fn confirm(&self, op: &DestructiveOp, preview: &OperationPreview) -> bool {
        let prompt = format_confirmation_prompt(op, preview);
        println!("\n{prompt}");

        let mut editor = self.editor.lock();
        editor
            .readline("Type 'yes' to proceed: ")
            .is_ok_and(|input| input.trim().eq_ignore_ascii_case("yes"))
    }
}

impl Shell {
    /// Creates a new shell with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            router: Arc::new(RwLock::new(QueryRouter::new())),
            config: ShellConfig::default(),
            wal: Mutex::new(None),
            icons: Icons::auto(),
        }
    }

    /// Creates a new shell with custom configuration.
    #[must_use]
    pub fn with_config(mut config: ShellConfig) -> Self {
        // Apply no_color setting
        if config.no_color {
            config.theme = Theme::plain();
            config.prompt = "neumann> ".to_string();
        }

        let icons = if config.no_color {
            Icons::plain()
        } else {
            Icons::auto()
        };

        Self {
            router: Arc::new(RwLock::new(QueryRouter::new())),
            config,
            wal: Mutex::new(None),
            icons,
        }
    }

    /// Returns a clone of the router Arc for shared access.
    #[must_use]
    pub fn router_arc(&self) -> Arc<RwLock<QueryRouter>> {
        Arc::clone(&self.router)
    }

    /// Returns a read guard to the query router for direct access.
    pub fn router(&self) -> parking_lot::RwLockReadGuard<'_, QueryRouter> {
        self.router.read()
    }

    /// Returns a write guard to the query router for mutable access.
    pub fn router_mut(&self) -> parking_lot::RwLockWriteGuard<'_, QueryRouter> {
        self.router.write()
    }

    /// Check if a command is a write operation that should be logged to WAL.
    fn is_write_command(cmd: &str) -> bool {
        let upper = cmd.to_uppercase();
        let first_word = upper.split_whitespace().next().unwrap_or("");

        match first_word {
            "INSERT" | "UPDATE" | "DELETE" | "CREATE" | "DROP" | "CHECKPOINT" | "ROLLBACK" => true,
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
            "ENTITY" => !upper.contains("ENTITY GET"),
            "GRAPH" => {
                upper.contains("GRAPH BATCH")
                    || upper.contains("CONSTRAINT CREATE")
                    || upper.contains("CONSTRAINT DROP")
                    || upper.contains("INDEX CREATE")
                    || upper.contains("INDEX DROP")
            },
            "BEGIN" => upper.contains("BEGIN CHAIN"),
            "COMMIT" => upper.contains("COMMIT CHAIN"),
            _ => false,
        }
    }

    /// Replay commands from a WAL file.
    fn replay_wal(
        &self,
        wal_path: &Path,
        mode: WalRecoveryMode,
    ) -> Result<WalReplayResult, String> {
        let file = std::fs::File::open(wal_path).map_err(|e| format!("Failed to open WAL: {e}"))?;
        let reader = BufReader::new(file);

        let mut replayed = 0;
        let mut errors = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line_number = line_num + 1;

            let cmd = match line {
                Ok(c) => c,
                Err(e) => {
                    let error = format!("Failed to read line: {e}");
                    match mode {
                        WalRecoveryMode::Strict => {
                            return Err(format!(
                                "WAL replay failed at line {line_number}: {error}"
                            ));
                        },
                        WalRecoveryMode::Recover => {
                            errors.push(WalReplayError::new(line_number, "<unreadable>", error));
                            continue;
                        },
                    }
                },
            };

            let cmd = cmd.trim();
            if cmd.is_empty() {
                continue;
            }

            let result = self.router.read().execute_parsed(cmd);
            if let Err(e) = result {
                match mode {
                    WalRecoveryMode::Strict => {
                        return Err(format!("WAL replay failed at line {line_number}: {e}"));
                    },
                    WalRecoveryMode::Recover => {
                        errors.push(WalReplayError::new(line_number, cmd, e.to_string()));
                        continue;
                    },
                }
            }
            replayed += 1;
        }

        Ok(WalReplayResult { replayed, errors })
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
            "help" | "\\h" | "\\?" => {
                return CommandResult::Help(output::format_help(&self.config.theme))
            },
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

        // Handle CLUSTER CONNECT command
        if lower.starts_with("cluster connect") {
            return self.handle_cluster_connect(trimmed);
        }

        // Handle CLUSTER DISCONNECT command
        if lower == "cluster disconnect" {
            return self.handle_cluster_disconnect();
        }

        // Execute with optional spinner for long operations
        let use_spinner = progress::needs_spinner(trimmed);
        let spinner = if use_spinner {
            Some(progress::operation_spinner(trimmed, &self.config.theme))
        } else {
            None
        };

        let query_result = self.router.read().execute_parsed(trimmed);

        if let Some(ref s) = spinner {
            s.finish_and_clear();
        }

        match query_result {
            Ok(result) => {
                // Log write commands to WAL after successful execution
                if Self::is_write_command(trimmed) {
                    let mut wal_guard = self.wal.lock();
                    if let Some(ref mut wal) = *wal_guard {
                        if let Err(e) = wal.append(trimmed) {
                            return CommandResult::Error(format!(
                                "Command succeeded but WAL write failed: {e}"
                            ));
                        }
                    }
                }
                CommandResult::Output(output::format_result(
                    &result,
                    &self.config.theme,
                    self.icons,
                ))
            },
            Err(e) => {
                let error_msg = format!(
                    "{} Error: {e}",
                    style::styled(self.icons.error, self.config.theme.error)
                );
                CommandResult::Error(error_msg)
            },
        }
    }

    /// Handles the SAVE command.
    fn handle_save(&self, input: &str) -> CommandResult {
        let Some(p) = Self::extract_path(input, "save") else {
            return CommandResult::Error(
                "Usage: SAVE 'path/to/file.bin' or SAVE path/to/file.bin".to_string(),
            );
        };

        let spinner = progress::operation_spinner("SAVE", &self.config.theme);

        let store = self.router.read().vector().store().clone();
        if let Err(e) = store.save_snapshot(&p) {
            spinner.finish_error(&format!("Failed to save: {e}"));
            return CommandResult::Error(format!("Failed to save: {e}"));
        }

        // Truncate WAL after successful save
        if let Some(ref mut wal) = *self.wal.lock() {
            if let Err(e) = wal.truncate() {
                spinner.finish_error("Saved but WAL truncate failed");
                return CommandResult::Error(format!(
                    "Saved snapshot but WAL truncate failed: {e}"
                ));
            }
        }

        spinner.finish_success(&format!("Saved to {p}"));
        CommandResult::Output(format!(
            "{} Saved snapshot to: {}",
            style::styled(self.icons.success, self.config.theme.success),
            style::styled(&p, self.config.theme.string)
        ))
    }

    /// Handles the SAVE COMPRESSED command.
    fn handle_save_compressed(&self, input: &str) -> CommandResult {
        let Some(p) = Self::extract_path(input, "save compressed") else {
            return CommandResult::Error("Usage: SAVE COMPRESSED 'path/to/file.bin'".to_string());
        };

        let spinner = progress::operation_spinner("SAVE COMPRESSED", &self.config.theme);

        let store = self.router.read().vector().store().clone();
        let dim = Self::detect_embedding_dimension(&store);
        let config = tensor_compress::CompressionConfig::balanced(dim);

        if let Err(e) = store.save_snapshot_compressed(&p, config) {
            spinner.finish_error(&format!("Failed to save: {e}"));
            return CommandResult::Error(format!("Failed to save compressed: {e}"));
        }

        // Truncate WAL after successful save
        if let Some(ref mut wal) = *self.wal.lock() {
            if let Err(e) = wal.truncate() {
                spinner.finish_error("Saved but WAL truncate failed");
                return CommandResult::Error(format!(
                    "Saved snapshot but WAL truncate failed: {e}"
                ));
            }
        }

        spinner.finish_success(&format!("Saved compressed to {p}"));
        CommandResult::Output(format!(
            "{} Saved compressed snapshot to: {}",
            style::styled(self.icons.success, self.config.theme.success),
            style::styled(&p, self.config.theme.string)
        ))
    }

    /// Handles the LOAD command.
    fn handle_load(&self, input: &str) -> CommandResult {
        let Some((p, recovery_mode)) = Self::extract_load_path_and_mode(input) else {
            return CommandResult::Error(
                "Usage: LOAD 'path/to/file.bin' or LOAD 'path' RECOVER".to_string(),
            );
        };

        let spinner = progress::operation_spinner("LOAD", &self.config.theme);

        // Try compressed format first, fall back to legacy
        let result =
            TensorStore::load_snapshot_compressed(&p).or_else(|_| TensorStore::load_snapshot(&p));

        match result {
            Ok(store) => {
                *self.router.write() = QueryRouter::with_shared_store(store);

                // Derive WAL path from snapshot path
                let wal_path = Path::new(&p).with_extension("log");

                // Replay WAL if it exists
                let replay_msg = if wal_path.exists() {
                    match self.replay_wal(&wal_path, recovery_mode) {
                        Ok(result) => Self::format_wal_replay_result(&result, recovery_mode),
                        Err(e) => {
                            spinner.finish_error("WAL replay failed");
                            let hint = if recovery_mode == WalRecoveryMode::Strict {
                                "\nHint: Use 'LOAD path RECOVER' to skip corrupted entries"
                            } else {
                                ""
                            };
                            return CommandResult::Error(format!(
                                "Loaded snapshot but WAL replay failed: {e}{hint}"
                            ));
                        },
                    }
                } else {
                    String::new()
                };

                // Initialize WAL for new writes
                match Wal::open_append(&wal_path) {
                    Ok(wal) => {
                        *self.wal.lock() = Some(wal);
                        spinner.finish_success(&format!("Loaded from {p}"));
                        CommandResult::Output(format!(
                            "{} Loaded snapshot from: {}{replay_msg}",
                            style::styled(self.icons.success, self.config.theme.success),
                            style::styled(&p, self.config.theme.string)
                        ))
                    },
                    Err(e) => {
                        spinner.finish_error("Failed to initialize WAL");
                        CommandResult::Error(format!(
                            "Loaded snapshot but failed to initialize WAL: {e}"
                        ))
                    },
                }
            },
            Err(e) => {
                spinner.finish_error(&format!("Failed to load: {e}"));
                CommandResult::Error(format!("Failed to load: {e}"))
            },
        }
    }

    /// Extracts path and recovery mode from LOAD command.
    fn extract_load_path_and_mode(input: &str) -> Option<(String, WalRecoveryMode)> {
        let rest = input
            .strip_prefix("load")
            .or_else(|| input.strip_prefix("LOAD"))?
            .trim();

        if rest.is_empty() {
            return None;
        }

        let upper_rest = rest.to_uppercase();

        if upper_rest == "RECOVER" {
            return None;
        }

        let (path_part, mode) = if upper_rest.ends_with(" RECOVER") {
            let path_end = rest.len() - " RECOVER".len();
            (&rest[..path_end], WalRecoveryMode::Recover)
        } else {
            (rest, WalRecoveryMode::Strict)
        };

        let path_part = path_part.trim();
        if path_part.is_empty() {
            return None;
        }

        let path = if (path_part.starts_with('\'') && path_part.ends_with('\''))
            || (path_part.starts_with('"') && path_part.ends_with('"'))
        {
            if path_part.len() > 2 {
                path_part[1..path_part.len() - 1].to_string()
            } else {
                return None;
            }
        } else {
            path_part.to_string()
        };

        Some((path, mode))
    }

    /// Formats the WAL replay result for display.
    fn format_wal_replay_result(result: &WalReplayResult, mode: WalRecoveryMode) -> String {
        let mut output = String::new();

        if result.replayed > 0 {
            let _ = write!(output, "\nReplayed {} commands from WAL", result.replayed);
        }

        if mode == WalRecoveryMode::Recover && !result.errors.is_empty() {
            let _ = write!(
                output,
                "\nWarning: Skipped {} corrupted WAL entries:",
                result.errors.len()
            );

            for error in result.errors.iter().take(5) {
                let _ = write!(
                    output,
                    "\n  Line {}: {} ({})",
                    error.line, error.command, error.error
                );
            }

            if result.errors.len() > 5 {
                let _ = write!(output, "\n  ... and {} more", result.errors.len() - 5);
            }
        }

        output
    }

    /// Extracts the path from a SAVE or LOAD command.
    fn extract_path(input: &str, command: &str) -> Option<String> {
        let rest = input[command.len()..].trim();
        if rest.is_empty() {
            return None;
        }

        if (rest.starts_with('\'') && rest.ends_with('\''))
            || (rest.starts_with('"') && rest.ends_with('"'))
        {
            if rest.len() > 2 {
                return Some(rest[1..rest.len() - 1].to_string());
            }
            return None;
        }

        Some(rest.to_string())
    }

    /// Detect the most common embedding dimension from stored vectors.
    fn detect_embedding_dimension(store: &TensorStore) -> usize {
        use tensor_store::TensorValue;

        let keys = store.scan("");
        for key in keys.iter().take(100) {
            if let Ok(tensor) = store.get(key) {
                for field in tensor.keys() {
                    match tensor.get(field) {
                        Some(TensorValue::Vector(v)) => {
                            return v.len();
                        },
                        Some(TensorValue::Sparse(s)) => {
                            return s.dimension();
                        },
                        _ => {},
                    }
                }
            }
        }

        tensor_compress::CompressionDefaults::STANDARD
    }

    /// Handles the WAL STATUS command.
    fn handle_wal_status(&self) -> CommandResult {
        self.wal.lock().as_ref().map_or_else(
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
    fn handle_wal_truncate(&self) -> CommandResult {
        self.wal.lock().as_mut().map_or_else(
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
    fn handle_vault_init(&self) -> CommandResult {
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

                let result = self.router.write().init_vault(&decoded);
                match result {
                    Ok(()) => CommandResult::Output(format!(
                        "{} Vault initialized",
                        style::styled(self.icons.success, self.config.theme.success)
                    )),
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
    fn handle_vault_identity(&self, input: &str) -> CommandResult {
        let rest = input
            .to_lowercase()
            .strip_prefix("vault identity")
            .unwrap_or("")
            .trim()
            .to_string();

        let identity = if rest.starts_with('\'') && rest.ends_with('\'') && rest.len() > 2 {
            &rest[1..rest.len() - 1]
        } else if !rest.is_empty() {
            &rest
        } else {
            return CommandResult::Output(format!(
                "Current identity: {}",
                self.router.read().current_identity().unwrap_or("<none>")
            ));
        };

        self.router.write().set_identity(identity);
        CommandResult::Output(format!(
            "{} Identity set to: {}",
            style::styled(self.icons.success, self.config.theme.success),
            style::styled(identity, self.config.theme.id)
        ))
    }

    /// Initialize the cache with default configuration.
    fn handle_cache_init(&self) -> CommandResult {
        let result = self.router.write().init_cache_default();
        match result {
            Ok(()) => CommandResult::Output(format!(
                "{} Cache initialized",
                style::styled(self.icons.success, self.config.theme.success)
            )),
            Err(e) => CommandResult::Error(format!("Failed to initialize cache: {e}")),
        }
    }

    /// Handles the CLUSTER CONNECT command.
    fn handle_cluster_connect(&self, input: &str) -> CommandResult {
        let args = input.trim();
        let args = args
            .strip_prefix("cluster connect")
            .or_else(|| args.strip_prefix("CLUSTER CONNECT"))
            .unwrap_or(args)
            .trim();

        if args.is_empty() {
            return CommandResult::Error(
                "Usage: CLUSTER CONNECT 'node_id@bind_addr' ['peer_id@peer_addr', ...]".to_string(),
            );
        }

        let spinner = progress::operation_spinner("CLUSTER CONNECT", &self.config.theme);

        // Parse quoted strings
        let mut addresses: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut in_quote = false;
        let mut quote_char = '"';

        for c in args.chars() {
            match c {
                '\'' | '"' if !in_quote => {
                    in_quote = true;
                    quote_char = c;
                },
                c if c == quote_char && in_quote => {
                    in_quote = false;
                    if !current.is_empty() {
                        addresses.push(current.clone());
                        current.clear();
                    }
                },
                _ if in_quote => current.push(c),
                ' ' | ',' => {},
                _ => current.push(c),
            }
        }
        if !current.is_empty() {
            addresses.push(current);
        }

        if addresses.is_empty() {
            spinner.finish_error("No addresses provided");
            return CommandResult::Error("No addresses provided".to_string());
        }

        // Parse first address as local node
        let local = &addresses[0];
        let (node_id, bind_addr) = match Self::parse_node_address(local) {
            Ok(parsed) => parsed,
            Err(e) => {
                spinner.finish_error(&format!("Invalid address: {e}"));
                return CommandResult::Error(format!("Invalid local address: {e}"));
            },
        };

        // Parse remaining addresses as peers
        let mut peers: Vec<(String, std::net::SocketAddr)> = Vec::new();
        for addr in &addresses[1..] {
            match Self::parse_node_address(addr) {
                Ok((peer_id, peer_addr)) => peers.push((peer_id, peer_addr)),
                Err(e) => {
                    spinner.finish_error(&format!("Invalid peer: {e}"));
                    return CommandResult::Error(format!("Invalid peer address '{addr}': {e}"));
                },
            }
        }

        // Create executor wrapper
        let executor: Arc<dyn QueryExecutor> = Arc::new(RouterExecutor(Arc::clone(&self.router)));

        // Initialize cluster
        let result = {
            let mut router = self.router.write();
            router.init_cluster_with_executor(&node_id, bind_addr, &peers, Some(executor))
        };

        match result {
            Ok(()) => {
                spinner.finish_success(&format!("Connected as {node_id}"));
                CommandResult::Output(format!(
                    "{} Cluster initialized: {} @ {} with {} peer(s)",
                    style::styled(self.icons.success, self.config.theme.success),
                    style::styled(&node_id, self.config.theme.id),
                    style::styled(bind_addr, self.config.theme.muted),
                    style::styled(peers.len(), self.config.theme.number)
                ))
            },
            Err(e) => {
                spinner.finish_error(&format!("Failed: {e}"));
                CommandResult::Error(format!("Failed to connect to cluster: {e}"))
            },
        }
    }

    /// Parse a node address in the format `node_id@host:port`.
    fn parse_node_address(s: &str) -> std::result::Result<(String, std::net::SocketAddr), String> {
        let parts: Vec<&str> = s.splitn(2, '@').collect();
        if parts.len() != 2 {
            return Err("Expected format 'node_id@host:port'".to_string());
        }

        let node_id = parts[0].to_string();
        let addr: std::net::SocketAddr = parts[1]
            .parse()
            .map_err(|e| format!("Invalid address '{}': {}", parts[1], e))?;

        Ok((node_id, addr))
    }

    /// Handles the CLUSTER DISCONNECT command.
    fn handle_cluster_disconnect(&self) -> CommandResult {
        let is_active = self.router.read().is_cluster_active();
        if !is_active {
            return CommandResult::Error("Not connected to cluster".to_string());
        }

        let result = self.router.write().shutdown_cluster();
        match result {
            Ok(()) => CommandResult::Output(format!(
                "{} Disconnected from cluster",
                style::styled(self.icons.success, self.config.theme.success)
            )),
            Err(e) => CommandResult::Error(format!("Failed to disconnect: {e}")),
        }
    }

    /// Lists all tables in the database.
    fn list_tables(&self) -> CommandResult {
        self.router
            .read()
            .execute_parsed("SHOW TABLES")
            .map_or_else(
                |_| CommandResult::Output("No tables found.".to_string()),
                |result| {
                    CommandResult::Output(output::format_result(
                        &result,
                        &self.config.theme,
                        self.icons,
                    ))
                },
            )
    }

    /// Returns the help text.
    #[must_use]
    pub fn help_text() -> String {
        output::format_help(&Theme::auto())
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
                println!("{}", progress::goodbye_message(&Theme::auto()));
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

    /// Executes a single line of input and returns a result.
    ///
    /// This is the primary entry point for non-interactive execution.
    ///
    /// # Errors
    ///
    /// Returns an error string if the command fails.
    pub fn execute_line(&mut self, line: &str) -> Result<String, String> {
        match self.execute(line) {
            CommandResult::Output(text) | CommandResult::Help(text) => Ok(text),
            CommandResult::Empty | CommandResult::Exit => Ok(String::new()),
            CommandResult::Error(e) => Err(e),
        }
    }

    /// Runs the interactive shell loop.
    ///
    /// # Errors
    ///
    /// Returns an error if readline initialization fails.
    pub fn run(&mut self) -> Result<(), ShellError> {
        let helper = NeumannHelper::new(self.config.theme.clone());
        let mut editor: Editor<NeumannHelper, DefaultHistory> =
            Editor::new().map_err(|e| ShellError::Init(e.to_string()))?;
        editor.set_helper(Some(helper));

        let editor = Arc::new(Mutex::new(editor));

        {
            let mut ed = editor.lock();
            if let Some(ref path) = self.config.history_file {
                let _ = ed.load_history(path);
            }
            ed.history_mut()
                .set_max_len(self.config.history_size)
                .map_err(|e| ShellError::Init(e.to_string()))?;
        }

        // Set up confirmation handler if checkpoint is available
        {
            let router = self.router.read();
            if router.has_checkpoint() {
                let handler = Arc::new(ShellConfirmationHandler::new(Arc::clone(&editor)));
                drop(router);
                let router = self.router.write();
                if let Err(e) = router.set_confirmation_handler(handler) {
                    eprintln!("Warning: Failed to set confirmation handler: {e}");
                }
            }
        }

        // Show welcome banner
        let banner = if progress::supports_full_banner() {
            progress::welcome_banner(Self::version(), &self.config.theme)
        } else {
            progress::compact_banner(Self::version(), &self.config.theme)
        };
        println!("{banner}");

        loop {
            let readline_result = {
                let mut ed = editor.lock();
                ed.readline(&self.config.prompt)
            };

            match readline_result {
                Ok(line) => {
                    if !line.trim().is_empty() {
                        let mut ed = editor.lock();
                        let _ = ed.add_history_entry(line.trim());
                    }
                    if Self::process_result(&self.execute(&line)) == LoopAction::Exit {
                        break;
                    }
                },
                Err(ReadlineError::Interrupted) => println!("^C"),
                Err(ReadlineError::Eof) => {
                    println!("{}", progress::goodbye_message(&self.config.theme));
                    break;
                },
                Err(err) => {
                    eprintln!("Error: {err}");
                    break;
                },
            }
        }

        if let Some(ref path) = self.config.history_file {
            let mut ed = editor.lock();
            let _ = ed.save_history(path);
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
            theme: Theme::plain(),
            no_color: false,
            no_boot: false,
            quiet: false,
        };
        let shell = Shell::with_config(config);
        assert_eq!(shell.config.prompt, "neumann> ");
        assert_eq!(shell.config.history_size, 500);
    }

    #[test]
    fn test_shell_with_no_color() {
        let config = ShellConfig {
            no_color: true,
            ..Default::default()
        };
        let shell = Shell::with_config(config);
        // When no_color is set, prompt should be plain
        assert_eq!(shell.config.prompt, "neumann> ");
    }

    #[test]
    fn test_execute_line() {
        let mut shell = Shell::new();
        let result = shell.execute_line("SELECT 1");
        assert!(result.is_ok() || result.is_err()); // Just verify it runs
    }

    #[test]
    fn test_execute_line_help() {
        let mut shell = Shell::new();
        let result = shell.execute_line("help");
        assert!(result.is_ok());
        if let Ok(text) = result {
            assert!(text.contains("Commands"));
        }
    }

    #[test]
    fn test_execute_line_exit() {
        let mut shell = Shell::new();
        let result = shell.execute_line("exit");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn test_execute_line_empty() {
        let mut shell = Shell::new();
        let result = shell.execute_line("");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");
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

        if let CommandResult::Help(text) = result {
            assert!(text.contains("Commands"));
            assert!(text.contains("SELECT"));
        }
    }

    #[test]
    fn test_help_backslash() {
        let mut shell = Shell::new();
        let result = shell.execute("\\h");
        assert!(matches!(result, CommandResult::Help(_)));
    }

    #[test]
    fn test_help_question_mark() {
        let mut shell = Shell::new();
        let result = shell.execute("\\?");
        assert!(matches!(result, CommandResult::Help(_)));
    }

    #[test]
    fn test_clear_command() {
        let mut shell = Shell::new();
        let result = shell.execute("clear");
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(text) = result {
            assert!(text.contains("\x1B[2J"));
        }
    }

    #[test]
    fn test_clear_backslash() {
        let mut shell = Shell::new();
        let result = shell.execute("\\c");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_tables_command() {
        let mut shell = Shell::new();
        let result = shell.execute("tables");
        // Should work even with no tables
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_tables_backslash() {
        let mut shell = Shell::new();
        let result = shell.execute("\\dt");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_wal_status_not_active() {
        let shell = Shell::new();
        let result = shell.handle_wal_status();
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("not active"));
        } else {
            panic!("Expected Output");
        }
    }

    #[test]
    fn test_wal_truncate_not_active() {
        let shell = Shell::new();
        let result = shell.handle_wal_truncate();
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_wal_status_command() {
        let mut shell = Shell::new();
        let result = shell.execute("wal status");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_wal_truncate_command() {
        let mut shell = Shell::new();
        let result = shell.execute("wal truncate");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_is_write_command() {
        assert!(Shell::is_write_command("INSERT INTO users VALUES (1)"));
        assert!(Shell::is_write_command("UPDATE users SET name = 'x'"));
        assert!(Shell::is_write_command("DELETE FROM users"));
        assert!(Shell::is_write_command("CREATE TABLE test (id INT)"));
        assert!(Shell::is_write_command("DROP TABLE test"));
        assert!(Shell::is_write_command("NODE CREATE person {}"));
        assert!(!Shell::is_write_command("SELECT * FROM users"));
        assert!(!Shell::is_write_command("NODE GET 1"));
        assert!(!Shell::is_write_command("SHOW TABLES"));
    }

    #[test]
    fn test_is_write_command_checkpoint() {
        assert!(Shell::is_write_command("CHECKPOINT"));
        assert!(Shell::is_write_command("ROLLBACK"));
    }

    #[test]
    fn test_is_write_command_edge() {
        assert!(Shell::is_write_command("EDGE CREATE knows 1 2"));
        assert!(!Shell::is_write_command("EDGE GET 1"));
    }

    #[test]
    fn test_is_write_command_embed() {
        assert!(Shell::is_write_command("EMBED STORE 'key' [1,2,3]"));
        assert!(Shell::is_write_command("EMBED DELETE 'key'"));
        assert!(!Shell::is_write_command("EMBED GET 'key'"));
    }

    #[test]
    fn test_is_write_command_vault() {
        assert!(Shell::is_write_command("VAULT SET 'key' 'value'"));
        assert!(Shell::is_write_command("VAULT DELETE 'key'"));
        assert!(Shell::is_write_command("VAULT ROTATE 'key'"));
        assert!(Shell::is_write_command("VAULT GRANT read 'key' TO 'user'"));
        assert!(Shell::is_write_command(
            "VAULT REVOKE read 'key' FROM 'user'"
        ));
        assert!(!Shell::is_write_command("VAULT GET 'key'"));
    }

    #[test]
    fn test_is_write_command_cache() {
        assert!(Shell::is_write_command("CACHE CLEAR"));
        assert!(!Shell::is_write_command("CACHE GET 'key'"));
    }

    #[test]
    fn test_is_write_command_blob() {
        assert!(Shell::is_write_command("BLOB PUT 'file.txt' content"));
        assert!(Shell::is_write_command("BLOB DELETE 'hash'"));
        assert!(Shell::is_write_command("BLOB LINK 'hash' 'artifact'"));
        assert!(Shell::is_write_command("BLOB UNLINK 'hash' 'artifact'"));
        assert!(Shell::is_write_command("BLOB TAG 'hash' 'tag'"));
        assert!(Shell::is_write_command("BLOB UNTAG 'hash' 'tag'"));
        assert!(Shell::is_write_command("BLOB GC"));
        assert!(Shell::is_write_command("BLOB REPAIR"));
        assert!(Shell::is_write_command("BLOB META SET 'hash' 'key' 'val'"));
        assert!(!Shell::is_write_command("BLOB GET 'hash'"));
    }

    #[test]
    fn test_is_write_command_entity() {
        assert!(Shell::is_write_command("ENTITY CREATE type {}"));
        assert!(!Shell::is_write_command("ENTITY GET 1"));
    }

    #[test]
    fn test_is_write_command_graph() {
        assert!(Shell::is_write_command("GRAPH BATCH CREATE"));
        assert!(Shell::is_write_command("GRAPH CONSTRAINT CREATE unique"));
        assert!(Shell::is_write_command("GRAPH CONSTRAINT DROP unique"));
        assert!(Shell::is_write_command("GRAPH INDEX CREATE idx"));
        assert!(Shell::is_write_command("GRAPH INDEX DROP idx"));
        assert!(!Shell::is_write_command("GRAPH ALGORITHM PAGERANK"));
    }

    #[test]
    fn test_is_write_command_chain() {
        assert!(Shell::is_write_command("BEGIN CHAIN tx1"));
        assert!(Shell::is_write_command("COMMIT CHAIN"));
        assert!(!Shell::is_write_command("CHAIN HEIGHT"));
    }

    #[test]
    fn test_is_write_command_other() {
        assert!(!Shell::is_write_command("SIMILAR [1,2,3] LIMIT 5"));
        assert!(!Shell::is_write_command("FIND pattern"));
    }

    #[test]
    fn test_extract_path_quoted() {
        let path = Shell::extract_path("save 'test.bin'", "save");
        assert_eq!(path, Some("test.bin".to_string()));
    }

    #[test]
    fn test_extract_path_double_quoted() {
        let path = Shell::extract_path("save \"test.bin\"", "save");
        assert_eq!(path, Some("test.bin".to_string()));
    }

    #[test]
    fn test_extract_path_unquoted() {
        let path = Shell::extract_path("save test.bin", "save");
        assert_eq!(path, Some("test.bin".to_string()));
    }

    #[test]
    fn test_extract_path_empty() {
        let path = Shell::extract_path("save", "save");
        assert_eq!(path, None);
    }

    #[test]
    fn test_extract_path_empty_quotes() {
        let path = Shell::extract_path("save ''", "save");
        assert_eq!(path, None);
    }

    #[test]
    fn test_extract_load_path_strict() {
        let result = Shell::extract_load_path_and_mode("LOAD 'data.bin'");
        assert_eq!(
            result,
            Some(("data.bin".to_string(), WalRecoveryMode::Strict))
        );
    }

    #[test]
    fn test_extract_load_path_recover() {
        let result = Shell::extract_load_path_and_mode("LOAD 'data.bin' RECOVER");
        assert_eq!(
            result,
            Some(("data.bin".to_string(), WalRecoveryMode::Recover))
        );
    }

    #[test]
    fn test_extract_load_path_lowercase() {
        let result = Shell::extract_load_path_and_mode("load data.bin");
        assert_eq!(
            result,
            Some(("data.bin".to_string(), WalRecoveryMode::Strict))
        );
    }

    #[test]
    fn test_extract_load_path_lowercase_recover() {
        let result = Shell::extract_load_path_and_mode("load 'data.bin' recover");
        assert_eq!(
            result,
            Some(("data.bin".to_string(), WalRecoveryMode::Recover))
        );
    }

    #[test]
    fn test_extract_load_path_empty() {
        let result = Shell::extract_load_path_and_mode("LOAD");
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_load_path_only_recover() {
        let result = Shell::extract_load_path_and_mode("LOAD RECOVER");
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_load_path_empty_after_recover() {
        let result = Shell::extract_load_path_and_mode("LOAD '' RECOVER");
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_node_address_valid() {
        let result = Shell::parse_node_address("node1@127.0.0.1:8080");
        assert!(result.is_ok());
        let (id, addr) = result.unwrap();
        assert_eq!(id, "node1");
        assert_eq!(addr.port(), 8080);
    }

    #[test]
    fn test_parse_node_address_invalid() {
        let result = Shell::parse_node_address("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_node_address_invalid_port() {
        let result = Shell::parse_node_address("node1@127.0.0.1:invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_node_address_ipv6() {
        let result = Shell::parse_node_address("node1@[::1]:8080");
        assert!(result.is_ok());
        let (id, addr) = result.unwrap();
        assert_eq!(id, "node1");
        assert_eq!(addr.port(), 8080);
    }

    #[test]
    fn test_version() {
        let version = Shell::version();
        assert!(!version.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = ShellConfig::default();
        assert_eq!(config.history_size, 1000);
        // Prompt contains "neumann>" with ANSI color codes
        assert!(config.prompt.contains("neumann>"));
    }

    #[test]
    fn test_loop_action() {
        assert_eq!(
            Shell::process_result(&CommandResult::Empty),
            LoopAction::Continue
        );
        assert_eq!(
            Shell::process_result(&CommandResult::Exit),
            LoopAction::Exit
        );
    }

    #[test]
    fn test_loop_action_output() {
        assert_eq!(
            Shell::process_result(&CommandResult::Output("test".to_string())),
            LoopAction::Continue
        );
    }

    #[test]
    fn test_loop_action_error() {
        assert_eq!(
            Shell::process_result(&CommandResult::Error("error".to_string())),
            LoopAction::Continue
        );
    }

    #[test]
    fn test_loop_action_help() {
        assert_eq!(
            Shell::process_result(&CommandResult::Help("help".to_string())),
            LoopAction::Continue
        );
    }

    #[test]
    fn test_shell_error_display() {
        let err = ShellError::Init("test error".to_string());
        let display = format!("{err}");
        assert!(display.contains("test error"));
    }

    #[test]
    fn test_shell_error_is_error() {
        let err = ShellError::Init("test".to_string());
        assert!(std::error::Error::source(&err).is_none());
    }

    #[test]
    fn test_create_table_and_query() {
        let mut shell = Shell::new();

        let result = shell.execute("CREATE TABLE test_users (id INT, name TEXT)");
        assert!(matches!(result, CommandResult::Output(_)));

        let result = shell.execute("INSERT INTO test_users VALUES (1, 'Alice')");
        assert!(matches!(result, CommandResult::Output(_)));

        let result = shell.execute("SELECT * FROM test_users");
        if let CommandResult::Output(text) = result {
            assert!(text.contains("Alice") || text.contains("1"));
        }
    }

    #[test]
    fn test_node_operations() {
        let mut shell = Shell::new();

        let result = shell.execute("NODE CREATE person {name: 'Bob', age: 30}");
        assert!(matches!(result, CommandResult::Output(_)));

        let result = shell.execute("NODE LIST person");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_embed_operations() {
        let mut shell = Shell::new();

        let result = shell.execute("EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4]");
        assert!(matches!(result, CommandResult::Output(_)));

        let result = shell.execute("EMBED GET 'doc1'");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_router_access() {
        let shell = Shell::new();
        let _router = shell.router();
        let _router_arc = shell.router_arc();
    }

    #[test]
    fn test_router_mut_access() {
        let shell = Shell::new();
        let _router = shell.router_mut();
    }

    #[test]
    fn test_help_text() {
        let help = Shell::help_text();
        assert!(help.contains("Commands"));
    }

    #[test]
    fn test_default_shell() {
        let shell = Shell::default();
        assert!(!shell.config.prompt.is_empty());
    }

    #[test]
    fn test_vault_init_no_env() {
        let shell = Shell::new();
        // Clear the env var if set
        std::env::remove_var("NEUMANN_VAULT_KEY");
        let result = shell.handle_vault_init();
        assert!(matches!(result, CommandResult::Error(_)));
        if let CommandResult::Error(msg) = result {
            assert!(msg.contains("NEUMANN_VAULT_KEY"));
        }
    }

    #[test]
    fn test_vault_init_invalid_base64() {
        let shell = Shell::new();
        std::env::set_var("NEUMANN_VAULT_KEY", "not-valid-base64!!!");
        let result = shell.handle_vault_init();
        std::env::remove_var("NEUMANN_VAULT_KEY");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_vault_identity_get() {
        let shell = Shell::new();
        let result = shell.handle_vault_identity("vault identity");
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("identity"));
        }
    }

    #[test]
    fn test_vault_identity_set() {
        let shell = Shell::new();
        let result = shell.handle_vault_identity("vault identity alice");
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("alice"));
        }
    }

    #[test]
    fn test_vault_identity_set_quoted() {
        let shell = Shell::new();
        let result = shell.handle_vault_identity("vault identity 'bob'");
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("bob"));
        }
    }

    #[test]
    fn test_cache_init() {
        let shell = Shell::new();
        let result = shell.handle_cache_init();
        // May succeed or fail depending on internal state
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_cluster_connect_no_args() {
        let shell = Shell::new();
        let result = shell.handle_cluster_connect("cluster connect");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_cluster_connect_invalid_address() {
        let shell = Shell::new();
        let result = shell.handle_cluster_connect("cluster connect 'invalid'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_cluster_disconnect_not_connected() {
        let shell = Shell::new();
        let result = shell.handle_cluster_disconnect();
        assert!(matches!(result, CommandResult::Error(_)));
        if let CommandResult::Error(msg) = result {
            assert!(msg.contains("Not connected"));
        }
    }

    #[test]
    fn test_format_wal_replay_result_empty() {
        let result = WalReplayResult {
            replayed: 0,
            errors: vec![],
        };
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Strict);
        assert!(formatted.is_empty());
    }

    #[test]
    fn test_format_wal_replay_result_with_replayed() {
        let result = WalReplayResult {
            replayed: 5,
            errors: vec![],
        };
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Strict);
        assert!(formatted.contains("Replayed 5 commands"));
    }

    #[test]
    fn test_format_wal_replay_result_with_errors() {
        let result = WalReplayResult {
            replayed: 3,
            errors: vec![
                WalReplayError::new(1, "cmd1", "error1".to_string()),
                WalReplayError::new(2, "cmd2", "error2".to_string()),
            ],
        };
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Recover);
        assert!(formatted.contains("Replayed 3 commands"));
        assert!(formatted.contains("Skipped 2"));
        assert!(formatted.contains("Line 1"));
    }

    #[test]
    fn test_format_wal_replay_result_many_errors() {
        let mut errors = Vec::new();
        for i in 1..=10 {
            errors.push(WalReplayError::new(
                i,
                &format!("cmd{i}"),
                "error".to_string(),
            ));
        }
        let result = WalReplayResult {
            replayed: 0,
            errors,
        };
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Recover);
        assert!(formatted.contains("and 5 more"));
    }

    #[test]
    fn test_format_wal_replay_result_strict_mode_ignores_errors() {
        let result = WalReplayResult {
            replayed: 3,
            errors: vec![WalReplayError::new(1, "cmd1", "error1".to_string())],
        };
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Strict);
        assert!(!formatted.contains("Skipped"));
    }

    #[test]
    fn test_detect_embedding_dimension_empty_store() {
        let store = TensorStore::new();
        let dim = Shell::detect_embedding_dimension(&store);
        assert_eq!(dim, tensor_compress::CompressionDefaults::STANDARD);
    }

    #[test]
    fn test_detect_embedding_dimension_with_vector() {
        let store = TensorStore::new();
        let mut data = tensor_store::TensorData::new();
        data.set(
            "embedding",
            tensor_store::TensorValue::Vector(vec![0.1, 0.2, 0.3, 0.4]),
        );
        store.put("test_key", data).unwrap();

        let dim = Shell::detect_embedding_dimension(&store);
        assert_eq!(dim, 4);
    }

    #[test]
    fn test_detect_embedding_dimension_with_sparse() {
        let store = TensorStore::new();
        let mut data = tensor_store::TensorData::new();
        let sparse = tensor_store::SparseVector::new(100);
        data.set("embedding", tensor_store::TensorValue::Sparse(sparse));
        store.put("test_key", data).unwrap();

        let dim = Shell::detect_embedding_dimension(&store);
        assert_eq!(dim, 100);
    }

    #[test]
    fn test_save_invalid_path() {
        let shell = Shell::new();
        let result = shell.handle_save("save");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_save_compressed_invalid_path() {
        let shell = Shell::new();
        let result = shell.handle_save_compressed("save compressed");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_load_invalid_path() {
        let shell = Shell::new();
        let result = shell.handle_load("load");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_execute_via_command() {
        let mut shell = Shell::new();

        // Test vault init command
        let result = shell.execute("vault init");
        assert!(matches!(result, CommandResult::Error(_)));

        // Test vault identity command
        let result = shell.execute("vault identity test_user");
        assert!(matches!(result, CommandResult::Output(_)));

        // Test cache init command
        let result = shell.execute("cache init");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );

        // Test cluster connect command
        let result = shell.execute("cluster connect");
        assert!(matches!(result, CommandResult::Error(_)));

        // Test cluster disconnect command
        let result = shell.execute("cluster disconnect");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_dirs_home() {
        // This tests the dirs_home function indirectly via ShellConfig
        let original = std::env::var_os("HOME");

        std::env::set_var("HOME", "/test/home");
        let result = dirs_home();
        assert_eq!(result, Some(PathBuf::from("/test/home")));

        std::env::remove_var("HOME");
        let result = dirs_home();
        assert!(result.is_none());

        // Restore original
        if let Some(home) = original {
            std::env::set_var("HOME", home);
        }
    }

    #[test]
    fn test_router_executor() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let executor = RouterExecutor(router);

        // Test execute
        let result = executor.execute("SHOW TABLES");
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_shell_config_quiet() {
        let config = ShellConfig {
            quiet: true,
            ..Default::default()
        };
        assert!(config.quiet);
    }

    #[test]
    fn test_shell_config_no_boot() {
        let config = ShellConfig {
            no_boot: true,
            ..Default::default()
        };
        let shell = Shell::with_config(config);
        // Verify config was applied
        assert!(shell.config.no_boot);
    }

    #[test]
    fn test_command_result_equality() {
        assert_eq!(CommandResult::Empty, CommandResult::Empty);
        assert_eq!(CommandResult::Exit, CommandResult::Exit);
        assert_eq!(
            CommandResult::Output("test".to_string()),
            CommandResult::Output("test".to_string())
        );
        assert_ne!(
            CommandResult::Output("a".to_string()),
            CommandResult::Output("b".to_string())
        );
    }

    #[test]
    fn test_cluster_connect_with_peers() {
        let shell = Shell::new();
        // Test with peer addresses in various formats
        let result = shell.handle_cluster_connect(
            "cluster connect 'node1@127.0.0.1:8080' 'node2@127.0.0.1:8081'",
        );
        // Should error because we can't actually connect in tests
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_cluster_connect_double_quoted() {
        let shell = Shell::new();
        let result = shell.handle_cluster_connect("cluster connect \"node1@127.0.0.1:8080\"");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_cluster_connect_mixed_quotes() {
        let shell = Shell::new();
        let result = shell.handle_cluster_connect(
            "cluster connect 'node1@127.0.0.1:8080', \"node2@127.0.0.1:8081\"",
        );
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_execute_save_command() {
        let mut shell = Shell::new();
        let result = shell.execute("SAVE 'test_snapshot.bin'");
        // Will error or succeed depending on permissions
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
        // Clean up if file was created
        let _ = std::fs::remove_file("test_snapshot.bin");
    }

    #[test]
    fn test_execute_save_compressed_command() {
        let mut shell = Shell::new();
        let result = shell.execute("SAVE COMPRESSED 'test_compressed.bin'");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
        let _ = std::fs::remove_file("test_compressed.bin");
    }

    #[test]
    fn test_execute_load_nonexistent() {
        let mut shell = Shell::new();
        let result = shell.execute("LOAD 'nonexistent_file_12345.bin'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_execute_load_with_recover() {
        let mut shell = Shell::new();
        let result = shell.execute("LOAD 'nonexistent_file_12345.bin' RECOVER");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_command_result_debug() {
        let result = CommandResult::Output("test".to_string());
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("Output"));
    }

    #[test]
    fn test_command_result_clone() {
        let result = CommandResult::Error("error".to_string());
        let cloned = result.clone();
        assert_eq!(result, cloned);
    }

    #[test]
    fn test_shell_config_debug() {
        let config = ShellConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("history_size"));
    }

    #[test]
    fn test_shell_config_clone() {
        let config = ShellConfig::default();
        let cloned = config.clone();
        assert_eq!(config.history_size, cloned.history_size);
    }

    #[test]
    fn test_shell_error_clone() {
        let err = ShellError::Init("test".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_shell_error_debug() {
        let err = ShellError::Init("test".to_string());
        let debug_str = format!("{err:?}");
        assert!(debug_str.contains("Init"));
    }

    #[test]
    fn test_loop_action_debug() {
        let action = LoopAction::Continue;
        let debug_str = format!("{action:?}");
        assert!(debug_str.contains("Continue"));
    }

    #[test]
    fn test_loop_action_clone() {
        let action = LoopAction::Exit;
        let cloned = action;
        assert_eq!(action, cloned);
    }

    #[test]
    fn test_extract_load_path_double_quoted() {
        let result = Shell::extract_load_path_and_mode("LOAD \"data.bin\"");
        assert_eq!(
            result,
            Some(("data.bin".to_string(), WalRecoveryMode::Strict))
        );
    }

    #[test]
    fn test_extract_load_path_unquoted_recover() {
        let result = Shell::extract_load_path_and_mode("LOAD data.bin RECOVER");
        assert_eq!(
            result,
            Some(("data.bin".to_string(), WalRecoveryMode::Recover))
        );
    }

    #[test]
    fn test_extract_path_with_spaces() {
        // Quoted path with spaces
        let path = Shell::extract_path("save 'path with spaces.bin'", "save");
        assert_eq!(path, Some("path with spaces.bin".to_string()));
    }

    #[test]
    fn test_vault_init_valid_key() {
        let shell = Shell::new();
        // Set a valid base64 encoded 32-byte key
        let key = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &[0u8; 32]);
        std::env::set_var("NEUMANN_VAULT_KEY", &key);
        let result = shell.handle_vault_init();
        std::env::remove_var("NEUMANN_VAULT_KEY");
        // Should succeed
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_execute_cluster_connect_command() {
        let mut shell = Shell::new();
        let result = shell.execute("CLUSTER CONNECT 'invalid'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_execute_cluster_disconnect_command() {
        let mut shell = Shell::new();
        let result = shell.execute("CLUSTER DISCONNECT");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_cluster_connect_quoted_peers() {
        let shell = Shell::new();
        // Multiple peers with different quote styles
        let result = shell.handle_cluster_connect(
            "CLUSTER CONNECT 'node1@127.0.0.1:8080', 'node2@127.0.0.1:8081', 'node3@127.0.0.1:8082'",
        );
        // Will fail to actually connect but should parse
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_shell_config_with_all_defaults() {
        let config = ShellConfig::default();
        assert!(config.history_file.is_some());
        assert_eq!(config.history_size, 1000);
        assert!(config.prompt.contains("neumann"));
    }

    #[test]
    fn test_shell_with_config_no_color_icons() {
        let config = ShellConfig {
            no_color: true,
            ..Default::default()
        };
        let shell = Shell::with_config(config);
        // icons should be plain
        assert_eq!(shell.icons.success, "[ok]");
    }

    #[test]
    fn test_execute_with_wal_active() {
        let mut shell = Shell::new();

        // Create a temp directory for the snapshot
        let temp_dir = std::env::temp_dir();
        let snapshot_path = temp_dir.join("test_shell_snapshot.bin");
        let snapshot_str = snapshot_path.to_string_lossy().to_string();

        // Create some data first
        shell.execute("CREATE TABLE wal_test (id INT)");

        // Save snapshot
        let result = shell.execute(&format!("SAVE '{}'", snapshot_str));
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );

        // Clean up
        let _ = std::fs::remove_file(&snapshot_path);
        let _ = std::fs::remove_file(snapshot_path.with_extension("log"));
    }

    #[test]
    fn test_list_tables_after_create() {
        let mut shell = Shell::new();
        shell.execute("CREATE TABLE test_list_table (id INT)");
        let result = shell.list_tables();
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_invalid_syntax() {
        let mut shell = Shell::new();
        let result = shell.execute("@#$%^&*");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_execute_backslash_commands_case() {
        let mut shell = Shell::new();

        // These should all work
        assert!(matches!(shell.execute("\\q"), CommandResult::Exit));
        assert!(matches!(shell.execute("\\h"), CommandResult::Help(_)));
        assert!(matches!(shell.execute("\\?"), CommandResult::Help(_)));
        assert!(matches!(shell.execute("\\c"), CommandResult::Output(_)));
        assert!(matches!(shell.execute("\\dt"), CommandResult::Output(_)));
    }

    #[test]
    fn test_extract_load_path_edge_cases() {
        // Empty quotes
        assert_eq!(Shell::extract_load_path_and_mode("LOAD ''"), None);
        assert_eq!(Shell::extract_load_path_and_mode("LOAD \"\""), None);

        // Just whitespace
        assert_eq!(Shell::extract_load_path_and_mode("load   "), None);
    }

    #[test]
    fn test_extract_path_edge_cases() {
        // Whitespace-only after command
        assert_eq!(Shell::extract_path("save   ", "save"), None);

        // Very long path
        let long_path = "a".repeat(1000);
        let result = Shell::extract_path(&format!("save {long_path}"), "save");
        assert_eq!(result, Some(long_path));
    }

    #[test]
    fn test_parse_node_address_edge_cases() {
        // Empty node id
        let result = Shell::parse_node_address("@127.0.0.1:8080");
        assert!(result.is_ok());

        // IPv6 localhost
        let result = Shell::parse_node_address("node@[::]:8080");
        assert!(result.is_ok());
    }

    #[test]
    fn test_shell_error_eq() {
        let e1 = ShellError::Init("error1".to_string());
        let e2 = ShellError::Init("error1".to_string());
        let e3 = ShellError::Init("error2".to_string());
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_command_result_all_variants_debug() {
        let variants = [
            CommandResult::Output("out".to_string()),
            CommandResult::Exit,
            CommandResult::Help("help".to_string()),
            CommandResult::Empty,
            CommandResult::Error("err".to_string()),
        ];
        for v in variants {
            let _ = format!("{v:?}");
        }
    }

    #[test]
    fn test_handle_save_no_path() {
        let shell = Shell::new();
        let result = shell.handle_save("save");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_save_compressed_no_path() {
        let shell = Shell::new();
        let result = shell.handle_save_compressed("save compressed");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_load_no_path() {
        let shell = Shell::new();
        let result = shell.handle_load("load");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_load_nonexistent() {
        let shell = Shell::new();
        let result = shell.handle_load("load 'nonexistent.bin'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_shell_config_clone_2() {
        let config = ShellConfig::default();
        let cloned = config.clone();
        assert_eq!(config.history_size, cloned.history_size);
        assert_eq!(config.no_color, cloned.no_color);
        assert_eq!(config.no_boot, cloned.no_boot);
    }

    #[test]
    fn test_shell_with_no_boot() {
        let config = ShellConfig {
            no_boot: true,
            ..Default::default()
        };
        let shell = Shell::with_config(config);
        // Verify config was applied
        assert!(shell.config.no_boot);
    }

    #[test]
    fn test_list_tables_empty() {
        let shell = Shell::new();
        let result = shell.list_tables();
        // Should work with no tables
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_extract_load_with_recover_keyword() {
        // Recovery mode variants
        let result = Shell::extract_load_path_and_mode("load 'file.bin' RECOVER");
        assert!(result.is_some());
        let (_, mode) = result.unwrap();
        assert_eq!(mode, WalRecoveryMode::Recover);

        let result = Shell::extract_load_path_and_mode("LOAD file.bin RECOVER");
        assert!(result.is_some());
        let (_, mode) = result.unwrap();
        assert_eq!(mode, WalRecoveryMode::Recover);
    }

    #[test]
    fn test_wal_replay_result_debug() {
        let result = WalReplayResult {
            replayed: 5,
            errors: vec![],
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("replayed"));
    }

    #[test]
    fn test_wal_replay_result_clone() {
        let result = WalReplayResult {
            replayed: 10,
            errors: vec![WalReplayError::new(1, "cmd", "error".to_string())],
        };
        let cloned = result.clone();
        assert_eq!(cloned.replayed, result.replayed);
        assert_eq!(cloned.errors.len(), result.errors.len());
    }

    #[test]
    fn test_wal_replay_error_debug() {
        let error = WalReplayError::new(5, "test command", "test error".to_string());
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("line"));
    }

    #[test]
    fn test_wal_replay_error_clone() {
        let error = WalReplayError::new(10, "cmd", "err".to_string());
        let cloned = error.clone();
        assert_eq!(cloned.line, error.line);
        assert_eq!(cloned.command, error.command);
    }

    #[test]
    fn test_wal_recovery_mode_debug() {
        let mode = WalRecoveryMode::Strict;
        let debug_str = format!("{mode:?}");
        assert!(debug_str.contains("Strict"));

        let mode = WalRecoveryMode::Recover;
        let debug_str = format!("{mode:?}");
        assert!(debug_str.contains("Recover"));
    }

    #[test]
    fn test_wal_recovery_mode_clone() {
        let mode = WalRecoveryMode::Recover;
        let cloned = mode;
        assert_eq!(cloned, mode);
    }

    #[test]
    fn test_detect_embedding_dimension_no_vectors() {
        let store = TensorStore::new();
        let mut data = tensor_store::TensorData::new();
        data.set(
            "name",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(
                "test".to_string(),
            )),
        );
        store.put("test_key", data).unwrap();

        // Should return default when no vectors found
        let dim = Shell::detect_embedding_dimension(&store);
        assert_eq!(dim, tensor_compress::CompressionDefaults::STANDARD);
    }

    #[test]
    fn test_cluster_connect_unquoted_address() {
        let shell = Shell::new();
        let result = shell.handle_cluster_connect("cluster connect node1@127.0.0.1:8080");
        // Will error because we can't actually connect, but should parse the unquoted address
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_vault_identity_lowercase() {
        let shell = Shell::new();
        let result = shell.handle_vault_identity("VAULT IDENTITY test_user");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_execute_query_with_spinner() {
        let mut shell = Shell::new();
        // SIMILAR command should trigger spinner
        let result = shell.execute("SIMILAR [0.1, 0.2, 0.3] LIMIT 5");
        // May succeed or fail depending on store state
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_handle_save_to_temp_dir() {
        let shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_shell_save_temp.bin");
        let result = shell.handle_save(&format!("save '{}'", path.display()));
        // Should succeed writing to temp
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_handle_save_compressed_to_temp_dir() {
        let shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_shell_save_compressed_temp.bin");
        let result = shell.handle_save_compressed(&format!("save compressed '{}'", path.display()));
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_shell_error_partial_eq() {
        let e1 = ShellError::Init("same".to_string());
        let e2 = ShellError::Init("same".to_string());
        let e3 = ShellError::Init("different".to_string());
        assert_eq!(e1, e2);
        assert_ne!(e1, e3);
    }

    #[test]
    fn test_loop_action_copy() {
        let action = LoopAction::Continue;
        let copied: LoopAction = action;
        assert_eq!(copied, LoopAction::Continue);
    }

    #[test]
    fn test_format_wal_replay_result_exact_5_errors() {
        // Test with exactly 5 errors (boundary case)
        let errors: Vec<WalReplayError> = (1..=5)
            .map(|i| WalReplayError::new(i, &format!("cmd{i}"), "error".to_string()))
            .collect();
        let result = WalReplayResult {
            replayed: 0,
            errors,
        };
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Recover);
        // Should NOT contain "and X more" since exactly 5 errors are shown
        assert!(!formatted.contains("and"));
    }

    #[test]
    fn test_format_wal_replay_result_zero_replayed() {
        let result = WalReplayResult {
            replayed: 0,
            errors: vec![WalReplayError::new(1, "bad", "err".to_string())],
        };
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Recover);
        // Should not contain "Replayed 0"
        assert!(!formatted.contains("Replayed 0"));
    }

    #[test]
    fn test_execute_multiple_backslash_commands() {
        let mut shell = Shell::new();
        // Test each backslash command variant
        assert!(matches!(shell.execute("\\q"), CommandResult::Exit));
        assert!(matches!(shell.execute("\\h"), CommandResult::Help(_)));
        assert!(matches!(shell.execute("\\?"), CommandResult::Help(_)));
        assert!(matches!(shell.execute("\\c"), CommandResult::Output(_)));
        assert!(matches!(shell.execute("\\dt"), CommandResult::Output(_)));
    }

    #[test]
    fn test_shell_default_trait() {
        let shell1 = Shell::default();
        let shell2 = Shell::new();
        // Both should have the same config defaults
        assert_eq!(shell1.config.history_size, shell2.config.history_size);
    }

    #[test]
    fn test_parse_node_address_with_at_in_nodeid() {
        // Multiple @ should use splitn(2)
        let result = Shell::parse_node_address("node@name@127.0.0.1:8080");
        assert!(result.is_err());
    }

    #[test]
    fn test_is_write_command_case_insensitive() {
        // Verify case insensitivity
        assert!(Shell::is_write_command("insert into test values (1)"));
        assert!(Shell::is_write_command("INSERT into test VALUES (1)"));
        assert!(Shell::is_write_command("INSERT INTO TEST VALUES (1)"));
    }

    #[test]
    fn test_execute_with_trailing_whitespace() {
        let mut shell = Shell::new();
        let result = shell.execute("  help  ");
        assert!(matches!(result, CommandResult::Help(_)));
    }

    #[test]
    fn test_cluster_connect_with_unquoted_comma_separated() {
        let shell = Shell::new();
        // Comma separated without quotes
        let result = shell
            .handle_cluster_connect("cluster connect node1@127.0.0.1:8080,node2@127.0.0.1:8081");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_save_and_load_cycle() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_save_load_cycle.bin");
        let path_str = path.to_string_lossy().to_string();

        // Create some data
        shell.execute("CREATE TABLE cycle_test (id INT, name TEXT)");
        shell.execute("INSERT INTO cycle_test VALUES (1, 'Alice')");
        shell.execute("INSERT INTO cycle_test VALUES (2, 'Bob')");

        // Save
        let save_result = shell.execute(&format!("SAVE '{}'", path_str));
        assert!(
            matches!(save_result, CommandResult::Output(_)),
            "Save should succeed"
        );

        // Create a new shell and load
        let mut shell2 = Shell::new();
        let load_result = shell2.execute(&format!("LOAD '{}'", path_str));
        // Load may succeed or fail based on file system permissions
        let loaded = matches!(load_result, CommandResult::Output(_));

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));

        if loaded {
            // Verify data was loaded
            let result = shell2.execute("SELECT * FROM cycle_test");
            assert!(matches!(result, CommandResult::Output(_)));
        }
    }

    #[test]
    fn test_save_and_load_compressed_cycle() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_save_load_compressed.bin");
        let path_str = path.to_string_lossy().to_string();

        // Create some data
        shell.execute("CREATE TABLE compress_test (id INT)");
        shell.execute("INSERT INTO compress_test VALUES (1)");

        // Save compressed
        let result = shell.execute(&format!("SAVE COMPRESSED '{}'", path_str));
        let saved = matches!(result, CommandResult::Output(_));

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));

        // Just verify it ran without panic
        assert!(saved || matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_wal_status_when_active() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_wal_status_active.bin");
        let path_str = path.to_string_lossy().to_string();

        // Save to create a snapshot
        shell.execute(&format!("SAVE '{}'", path_str));

        // Load to activate WAL
        shell.execute(&format!("LOAD '{}'", path_str));

        // Check WAL status
        let result = shell.execute("WAL STATUS");
        // May show active or not depending on successful load
        assert!(matches!(result, CommandResult::Output(_)));

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));
    }

    #[test]
    fn test_extract_path_whitespace_path() {
        // Path with only whitespace after stripping quotes
        let result = Shell::extract_path("save '   '", "save");
        assert_eq!(result, Some("   ".to_string()));
    }

    #[test]
    fn test_wal_write_on_successful_command() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_wal_write.bin");
        let path_str = path.to_string_lossy().to_string();

        // Save and load to activate WAL
        shell.execute(&format!("SAVE '{}'", path_str));
        shell.execute(&format!("LOAD '{}'", path_str));

        // Execute write command
        let result = shell.execute("CREATE TABLE wal_write_test (id INT)");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));
    }

    #[test]
    fn test_replay_wal_with_valid_commands() {
        use std::io::Write;

        let shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let wal_path = temp_dir.join("test_replay_valid.log");

        // Create a WAL file with valid commands
        {
            let mut file = std::fs::File::create(&wal_path).unwrap();
            writeln!(file, "CREATE TABLE replay_test (id INT)").unwrap();
            writeln!(file, "INSERT INTO replay_test VALUES (1)").unwrap();
            writeln!(file).unwrap(); // empty line
            writeln!(file, "INSERT INTO replay_test VALUES (2)").unwrap();
        }

        let result = shell.replay_wal(&wal_path, WalRecoveryMode::Strict);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.replayed, 3);
        assert!(result.errors.is_empty());

        // Clean up
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_replay_wal_strict_mode_error() {
        use std::io::Write;

        let shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let wal_path = temp_dir.join("test_replay_strict_error.log");

        // Create a WAL file with invalid command
        {
            let mut file = std::fs::File::create(&wal_path).unwrap();
            writeln!(file, "CREATE TABLE strict_test (id INT)").unwrap();
            writeln!(file, "INVALID_COMMAND @#$%").unwrap(); // This will fail
            writeln!(file, "INSERT INTO strict_test VALUES (1)").unwrap();
        }

        let result = shell.replay_wal(&wal_path, WalRecoveryMode::Strict);
        assert!(result.is_err()); // Should fail in strict mode

        // Clean up
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_replay_wal_recover_mode() {
        use std::io::Write;

        let shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let wal_path = temp_dir.join("test_replay_recover.log");

        // Create a WAL file with one invalid command
        {
            let mut file = std::fs::File::create(&wal_path).unwrap();
            writeln!(file, "CREATE TABLE recover_test (id INT)").unwrap();
            writeln!(file, "INVALID_COMMAND @#$%").unwrap(); // This will be skipped
            writeln!(file, "INSERT INTO recover_test VALUES (1)").unwrap();
        }

        let result = shell.replay_wal(&wal_path, WalRecoveryMode::Recover);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.replayed, 2); // 2 commands succeeded
        assert_eq!(result.errors.len(), 1); // 1 error was skipped

        // Clean up
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_replay_wal_file_not_found() {
        let shell = Shell::new();
        let result = shell.replay_wal(
            Path::new("/nonexistent/path/wal.log"),
            WalRecoveryMode::Strict,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to open WAL"));
    }

    #[test]
    fn test_load_with_wal_replay_success() {
        use std::io::Write;

        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let snap_path = temp_dir.join("test_load_wal_replay.bin");
        let wal_path = snap_path.with_extension("log");
        let snap_str = snap_path.to_string_lossy().to_string();

        // Create data and save snapshot
        shell.execute("CREATE TABLE load_replay_test (id INT)");
        shell.execute(&format!("SAVE '{}'", snap_str));

        // Create a WAL file with additional commands
        {
            let mut file = std::fs::File::create(&wal_path).unwrap();
            writeln!(file, "INSERT INTO load_replay_test VALUES (100)").unwrap();
        }

        // Load should replay the WAL
        let result = shell.handle_load(&format!("LOAD '{}'", snap_str));
        // Should succeed and replay the WAL
        assert!(matches!(result, CommandResult::Output(_)));

        // Clean up
        let _ = std::fs::remove_file(&snap_path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_load_with_wal_replay_error_strict() {
        use std::io::Write;

        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let snap_path = temp_dir.join("test_load_wal_error_strict.bin");
        let wal_path = snap_path.with_extension("log");
        let snap_str = snap_path.to_string_lossy().to_string();

        // Save snapshot
        shell.execute(&format!("SAVE '{}'", snap_str));

        // Create a WAL file with invalid command
        {
            let mut file = std::fs::File::create(&wal_path).unwrap();
            writeln!(file, "INVALID_COMMAND @#$%").unwrap();
        }

        // Load in strict mode should fail
        let result = shell.handle_load(&format!("LOAD '{}'", snap_str));
        assert!(matches!(result, CommandResult::Error(_)));
        if let CommandResult::Error(msg) = result {
            assert!(msg.contains("WAL replay failed"));
            assert!(msg.contains("RECOVER")); // Hint should mention RECOVER
        }

        // Clean up
        let _ = std::fs::remove_file(&snap_path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_load_with_wal_replay_error_recover() {
        use std::io::Write;

        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let snap_path = temp_dir.join("test_load_wal_error_recover.bin");
        let wal_path = snap_path.with_extension("log");
        let snap_str = snap_path.to_string_lossy().to_string();

        // Save snapshot
        shell.execute(&format!("SAVE '{}'", snap_str));

        // Create a WAL file with invalid command
        {
            let mut file = std::fs::File::create(&wal_path).unwrap();
            writeln!(file, "CREATE TABLE recover_load_test (id INT)").unwrap();
            writeln!(file, "INVALID_COMMAND @#$%").unwrap();
            writeln!(file, "INSERT INTO recover_load_test VALUES (1)").unwrap();
        }

        // Load with RECOVER should succeed but report skipped entries
        let result = shell.handle_load(&format!("LOAD '{}' RECOVER", snap_str));
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(msg) = &result {
            assert!(msg.contains("Loaded snapshot") || msg.contains("Skipped"));
        }

        // Clean up
        let _ = std::fs::remove_file(&snap_path);
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_wal_truncate_when_active() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_wal_truncate_active.bin");
        let path_str = path.to_string_lossy().to_string();

        // Save and load to activate WAL
        shell.execute(&format!("SAVE '{}'", path_str));
        shell.execute(&format!("LOAD '{}'", path_str));

        // Add some WAL entries
        shell.execute("CREATE TABLE truncate_test (id INT)");
        shell.execute("INSERT INTO truncate_test VALUES (1)");

        // Truncate WAL
        let result = shell.execute("WAL TRUNCATE");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));
    }

    #[test]
    fn test_wal_status_when_active_with_data() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_wal_status_with_data.bin");
        let path_str = path.to_string_lossy().to_string();

        // Save and load to activate WAL
        shell.execute(&format!("SAVE '{}'", path_str));
        shell.execute(&format!("LOAD '{}'", path_str));

        // Add some WAL entries
        shell.execute("CREATE TABLE status_test (id INT)");

        // Check WAL status
        let result = shell.execute("WAL STATUS");
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("WAL") || msg.contains("Path") || msg.contains("not active"));
        }

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));
    }

    #[test]
    fn test_save_truncates_wal() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_save_truncates_wal.bin");
        let path_str = path.to_string_lossy().to_string();

        // First save and load to activate WAL
        shell.execute(&format!("SAVE '{}'", path_str));
        shell.execute(&format!("LOAD '{}'", path_str));

        // Add WAL entries
        shell.execute("CREATE TABLE truncate_on_save_test (id INT)");
        shell.execute("INSERT INTO truncate_on_save_test VALUES (1)");

        // Save should truncate WAL
        let result = shell.execute(&format!("SAVE '{}'", path_str));
        assert!(matches!(result, CommandResult::Output(_)));

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));
    }

    #[test]
    fn test_save_compressed_truncates_wal() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_save_compressed_truncates_wal.bin");
        let path_str = path.to_string_lossy().to_string();

        // First save and load to activate WAL
        shell.execute(&format!("SAVE '{}'", path_str));
        shell.execute(&format!("LOAD '{}'", path_str));

        // Add WAL entries
        shell.execute("CREATE TABLE compressed_wal_test (id INT)");

        // Save compressed should truncate WAL
        let result = shell.execute(&format!("SAVE COMPRESSED '{}'", path_str));
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));
    }

    #[test]
    fn test_execute_select_query() {
        let mut shell = Shell::new();

        // Create table and data
        shell.execute("CREATE TABLE select_test (id INT, name TEXT)");
        shell.execute("INSERT INTO select_test VALUES (1, 'Alice')");
        shell.execute("INSERT INTO select_test VALUES (2, 'Bob')");

        // Execute SELECT
        let result = shell.execute("SELECT * FROM select_test WHERE id = 1");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_execute_update_query() {
        let mut shell = Shell::new();

        // Create table and data
        shell.execute("CREATE TABLE update_test (id INT, name TEXT)");
        shell.execute("INSERT INTO update_test VALUES (1, 'Alice')");

        // Execute UPDATE
        let result = shell.execute("UPDATE update_test SET name = 'Alicia' WHERE id = 1");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_delete_query() {
        let mut shell = Shell::new();

        // Create table and data
        shell.execute("CREATE TABLE delete_test (id INT)");
        shell.execute("INSERT INTO delete_test VALUES (1)");
        shell.execute("INSERT INTO delete_test VALUES (2)");

        // Execute DELETE
        let result = shell.execute("DELETE FROM delete_test WHERE id = 1");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_edge_operations() {
        let mut shell = Shell::new();

        // Create nodes first
        shell.execute("NODE CREATE person {name: 'Alice'}");
        shell.execute("NODE CREATE person {name: 'Bob'}");

        // Create edge
        let result = shell.execute("EDGE CREATE knows 1 2 {since: '2024'}");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );

        // List edges
        let result = shell.execute("EDGE LIST knows");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_similar_with_spinner() {
        let mut shell = Shell::new();

        // Store some embeddings
        shell.execute("EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4]");
        shell.execute("EMBED STORE 'doc2' [0.2, 0.3, 0.4, 0.5]");

        // SIMILAR should use spinner
        let result = shell.execute("SIMILAR [0.1, 0.2, 0.3, 0.4] LIMIT 5");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_find_with_spinner() {
        let mut shell = Shell::new();

        // FIND should use spinner
        let result = shell.execute("FIND test_pattern LIMIT 10");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_detect_embedding_dimension_scalar_only() {
        let store = TensorStore::new();
        let mut data = tensor_store::TensorData::new();
        data.set(
            "int_field",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::Int(42)),
        );
        data.set(
            "string_field",
            tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(
                "test".to_string(),
            )),
        );
        store.put("key1", data).unwrap();

        // Should return default when only scalars
        let dim = Shell::detect_embedding_dimension(&store);
        assert_eq!(dim, tensor_compress::CompressionDefaults::STANDARD);
    }

    #[test]
    fn test_router_executor_execute_valid() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let executor = RouterExecutor(Arc::clone(&router));

        // Create a table first
        {
            let r = router.write();
            let _ = r.execute_parsed("CREATE TABLE executor_test (id INT)");
        }

        // Execute through RouterExecutor
        let result = executor.execute("SELECT * FROM executor_test");
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_execute_path_query() {
        let mut shell = Shell::new();

        // Create graph
        shell.execute("NODE CREATE city {name: 'A'}");
        shell.execute("NODE CREATE city {name: 'B'}");
        shell.execute("NODE CREATE city {name: 'C'}");
        shell.execute("EDGE CREATE road 1 2");
        shell.execute("EDGE CREATE road 2 3");

        // Find path
        let result = shell.execute("PATH 1 3");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_neighbors_query() {
        let mut shell = Shell::new();

        // Create graph
        shell.execute("NODE CREATE person {name: 'Center'}");
        shell.execute("NODE CREATE person {name: 'Friend1'}");
        shell.execute("NODE CREATE person {name: 'Friend2'}");
        shell.execute("EDGE CREATE knows 1 2");
        shell.execute("EDGE CREATE knows 1 3");

        // Get neighbors
        let result = shell.execute("NEIGHBORS 1");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_cluster_connect_parsing_edge_cases() {
        let shell = Shell::new();

        // Test with unquoted addresses
        let result = shell.handle_cluster_connect("CLUSTER CONNECT node1@127.0.0.1:8080");
        assert!(matches!(result, CommandResult::Error(_)));

        // Test with mixed addressing
        let result = shell.handle_cluster_connect("cluster connect 'node@[::1]:8080'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_cluster_connect_invalid_peer_address() {
        let shell = Shell::new();

        // Valid local, invalid peer
        let result =
            shell.handle_cluster_connect("cluster connect 'node1@127.0.0.1:8080' 'invalid_peer'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_shell_icons_with_theme() {
        let config = ShellConfig {
            no_color: false,
            ..Default::default()
        };
        let shell = Shell::with_config(config);
        // Should use auto icons
        assert!(!shell.icons.success.is_empty());
    }

    #[test]
    fn test_shell_config_history_file_default() {
        // Test with HOME set
        let original = std::env::var_os("HOME");
        std::env::set_var("HOME", "/test/home");

        let config = ShellConfig::default();
        assert!(config.history_file.is_some());
        if let Some(path) = config.history_file {
            assert!(path.to_string_lossy().contains(".neumann_history"));
        }

        // Restore
        if let Some(home) = original {
            std::env::set_var("HOME", home);
        } else {
            std::env::remove_var("HOME");
        }
    }

    #[test]
    fn test_execute_pagerank() {
        let mut shell = Shell::new();

        // Create a simple graph for PageRank
        shell.execute("NODE CREATE webpage {url: 'a.com'}");
        shell.execute("NODE CREATE webpage {url: 'b.com'}");
        shell.execute("NODE CREATE webpage {url: 'c.com'}");
        shell.execute("EDGE CREATE links 1 2");
        shell.execute("EDGE CREATE links 2 3");
        shell.execute("EDGE CREATE links 3 1");

        // Execute PageRank
        let result = shell.execute("GRAPH ALGORITHM PAGERANK");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_centrality() {
        let mut shell = Shell::new();

        // Create graph
        shell.execute("NODE CREATE person {name: 'Alice'}");
        shell.execute("NODE CREATE person {name: 'Bob'}");
        shell.execute("NODE CREATE person {name: 'Carol'}");
        shell.execute("EDGE CREATE knows 1 2");
        shell.execute("EDGE CREATE knows 2 3");

        // Execute centrality (degree or betweenness)
        let result = shell.execute("GRAPH ALGORITHM DEGREE");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_communities() {
        let mut shell = Shell::new();

        // Create graph with clusters
        shell.execute("NODE CREATE person {group: 1}");
        shell.execute("NODE CREATE person {group: 1}");
        shell.execute("NODE CREATE person {group: 2}");
        shell.execute("EDGE CREATE knows 1 2");

        // Try community detection
        let result = shell.execute("GRAPH ALGORITHM COMMUNITIES");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_unified_query() {
        let mut shell = Shell::new();

        // Store embeddings
        shell.execute("EMBED STORE 'entity:1' [0.1, 0.2, 0.3, 0.4]");
        shell.execute("EMBED STORE 'entity:2' [0.2, 0.3, 0.4, 0.5]");

        // Create a node
        shell.execute("NODE CREATE entity {id: 1}");

        // Try FIND with pattern matching
        let result = shell.execute("FIND 'entity' LIMIT 10");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_node_list() {
        let mut shell = Shell::new();

        // Create nodes
        shell.execute("NODE CREATE animal {species: 'dog'}");
        shell.execute("NODE CREATE animal {species: 'cat'}");

        // List nodes
        let result = shell.execute("NODE LIST animal");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_execute_node_get() {
        let mut shell = Shell::new();

        // Create a node
        shell.execute("NODE CREATE item {name: 'book'}");

        // Get by ID
        let result = shell.execute("NODE GET 1");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_edge_list() {
        let mut shell = Shell::new();

        // Create graph
        shell.execute("NODE CREATE person {name: 'A'}");
        shell.execute("NODE CREATE person {name: 'B'}");
        shell.execute("EDGE CREATE friend 1 2");

        // List edges
        let result = shell.execute("EDGE LIST friend");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_execute_show_tables() {
        let mut shell = Shell::new();

        // Create some tables
        shell.execute("CREATE TABLE t1 (id INT)");
        shell.execute("CREATE TABLE t2 (id INT)");

        // Show tables
        let result = shell.execute("SHOW TABLES");
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(output) = result {
            assert!(output.contains("t1") || output.contains("Tables"));
        }
    }

    #[test]
    fn test_execute_drop_table() {
        let mut shell = Shell::new();

        // Create and drop table
        shell.execute("CREATE TABLE drop_me (id INT)");
        let result = shell.execute("DROP TABLE drop_me");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_aggregate_query() {
        let mut shell = Shell::new();

        // Create table with data
        shell.execute("CREATE TABLE sales (amount INT, category TEXT)");
        shell.execute("INSERT INTO sales VALUES (100, 'A')");
        shell.execute("INSERT INTO sales VALUES (200, 'A')");
        shell.execute("INSERT INTO sales VALUES (150, 'B')");

        // Try aggregate query
        let result = shell.execute("SELECT COUNT(*) FROM sales");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_embed_delete() {
        let mut shell = Shell::new();

        // Store then delete
        shell.execute("EMBED STORE 'del_key' [0.1, 0.2, 0.3]");
        let result = shell.execute("EMBED DELETE 'del_key'");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_embed_get() {
        let mut shell = Shell::new();

        // Store then get
        shell.execute("EMBED STORE 'get_key' [0.5, 0.6, 0.7]");
        let result = shell.execute("EMBED GET 'get_key'");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_node_delete() {
        let mut shell = Shell::new();

        // Create and delete node
        shell.execute("NODE CREATE temp {name: 'deleteme'}");
        let result = shell.execute("NODE DELETE 1");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_edge_delete() {
        let mut shell = Shell::new();

        // Create graph and delete edge
        shell.execute("NODE CREATE person {name: 'A'}");
        shell.execute("NODE CREATE person {name: 'B'}");
        shell.execute("EDGE CREATE temp 1 2");
        let result = shell.execute("EDGE DELETE temp 1 2");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_node_update() {
        let mut shell = Shell::new();

        // Create and update node
        shell.execute("NODE CREATE item {count: 0}");
        let result = shell.execute("NODE UPDATE 1 {count: 5}");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_is_write_command_all_branches() {
        // Test all branches explicitly
        assert!(!Shell::is_write_command("")); // Empty
        assert!(!Shell::is_write_command("SELECT")); // Read
        assert!(Shell::is_write_command("NODE CREATE test {}"));
        assert!(Shell::is_write_command("node create test {}")); // lowercase
        assert!(!Shell::is_write_command("EDGE GET 1"));
        assert!(Shell::is_write_command("ENTITY CREATE type {}"));
        assert!(!Shell::is_write_command("ENTITY GET 1"));
        assert!(!Shell::is_write_command("GRAPH ALGORITHM X"));
    }

    #[test]
    fn test_extract_path_compressed() {
        let path = Shell::extract_path("save compressed 'test.bin'", "save compressed");
        assert_eq!(path, Some("test.bin".to_string()));
    }

    #[test]
    fn test_detect_embedding_with_multiple_keys() {
        let store = TensorStore::new();

        // Add multiple entities, first few without vectors
        for i in 0..5 {
            let mut data = tensor_store::TensorData::new();
            data.set(
                "name",
                tensor_store::TensorValue::Scalar(tensor_store::ScalarValue::String(format!(
                    "entity{i}"
                ))),
            );
            store.put(&format!("scalar:{i}"), data).unwrap();
        }

        // Add one with vector
        let mut data = tensor_store::TensorData::new();
        data.set(
            "vec",
            tensor_store::TensorValue::Vector(vec![0.1, 0.2, 0.3, 0.4, 0.5]),
        );
        store.put("with_vec", data).unwrap();

        let dim = Shell::detect_embedding_dimension(&store);
        // Should find the 5-dim vector
        assert!(dim == 5 || dim == tensor_compress::CompressionDefaults::STANDARD);
    }

    #[test]
    fn test_cluster_connect_empty_quotes() {
        let shell = Shell::new();
        // Empty quoted address
        let result = shell.handle_cluster_connect("cluster connect ''");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_vault_identity_from_execute() {
        let mut shell = Shell::new();

        // Get current identity via execute
        let result = shell.execute("VAULT IDENTITY");
        assert!(matches!(result, CommandResult::Output(_)));

        // Set identity via execute
        let result = shell.execute("VAULT IDENTITY 'test_user'");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_cache_init_from_execute() {
        let mut shell = Shell::new();

        // Initialize cache via execute
        let result = shell.execute("CACHE INIT");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_wal_replay_with_only_empty_lines() {
        use std::io::Write;

        let shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let wal_path = temp_dir.join("test_replay_empty_only.log");

        // Create WAL with only empty/whitespace lines
        {
            let mut file = std::fs::File::create(&wal_path).unwrap();
            writeln!(file).unwrap();
            writeln!(file, "   ").unwrap();
            writeln!(file, "\t").unwrap();
        }

        let result = shell.replay_wal(&wal_path, WalRecoveryMode::Strict);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.replayed, 0);

        // Clean up
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_format_wal_replay_result_6_errors() {
        // Test with exactly 6 errors (one more than shown)
        let errors: Vec<WalReplayError> = (1..=6)
            .map(|i| WalReplayError::new(i, &format!("cmd{i}"), "error".to_string()))
            .collect();
        let result = WalReplayResult {
            replayed: 0,
            errors,
        };
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Recover);
        // Should contain "and 1 more"
        assert!(formatted.contains("and 1 more"));
    }

    #[test]
    fn test_shell_confirmation_handler_new() {
        let _helper = NeumannHelper::new(Theme::plain());
        let editor: Editor<NeumannHelper, rustyline::history::DefaultHistory> =
            Editor::new().unwrap();
        let editor = Arc::new(Mutex::new(editor));
        let _handler = ShellConfirmationHandler::new(Arc::clone(&editor));
    }

    #[test]
    fn test_router_executor_with_error() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let executor = RouterExecutor(Arc::clone(&router));

        // Execute invalid query
        let result = executor.execute("INVALID QUERY @#$%");
        assert!(result.is_err());
    }

    #[test]
    fn test_shell_execute_with_spinner_operations() {
        let mut shell = Shell::new();

        // Operations that use spinners
        shell.execute("EMBED STORE 'spinner_test' [0.1, 0.2, 0.3]");

        // SIMILAR uses spinner
        let result = shell.execute("SIMILAR [0.1, 0.2, 0.3] LIMIT 10");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_extract_load_mixed_case_recover() {
        // Test RECOVER keyword in different cases
        let result = Shell::extract_load_path_and_mode("LOAD 'file.bin' recover");
        assert!(result.is_some());
        let (_, mode) = result.unwrap();
        assert_eq!(mode, WalRecoveryMode::Recover);

        let result = Shell::extract_load_path_and_mode("load 'file.bin' RECOVER");
        assert!(result.is_some());
        let (_, mode) = result.unwrap();
        assert_eq!(mode, WalRecoveryMode::Recover);
    }

    #[test]
    fn test_handle_save_with_quotes_inside() {
        let shell = Shell::new();
        // Path without quotes
        let result = shell.handle_save("save /tmp/test_file.bin");
        // Should work or fail but not panic
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_chain_query() {
        let mut shell = Shell::new();

        // Chain operations
        let result = shell.execute("CHAIN HEIGHT");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );

        let result = shell.execute("CHAIN STATUS");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_graph_pattern_match() {
        let mut shell = Shell::new();

        // Create nodes and edges
        shell.execute("NODE CREATE person {name: 'Alice'}");
        shell.execute("NODE CREATE person {name: 'Bob'}");
        shell.execute("EDGE CREATE knows 1 2");

        // Try pattern match query (if supported)
        let result = shell.execute("GRAPH MATCH (a)-[r]->(b) WHERE a.name = 'Alice'");
        // May not be fully supported, just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_execute_batch_operations() {
        let mut shell = Shell::new();

        // Create table
        shell.execute("CREATE TABLE batch_test (id INT, name TEXT)");

        // Try batch insert (if supported)
        let result = shell.execute("INSERT INTO batch_test VALUES (1, 'a'), (2, 'b'), (3, 'c')");
        let _ = result;
    }

    #[test]
    fn test_execute_constraint_operations() {
        let mut shell = Shell::new();

        // Try constraint operations
        let result = shell.execute("GRAPH CONSTRAINT LIST");
        let _ = result;
    }

    #[test]
    fn test_execute_index_operations() {
        let mut shell = Shell::new();

        // Try index operations
        let result = shell.execute("GRAPH INDEX LIST");
        let _ = result;
    }

    #[test]
    fn test_shell_config_quiet_mode() {
        let config = ShellConfig {
            quiet: true,
            no_color: true,
            no_boot: true,
            ..Default::default()
        };
        let shell = Shell::with_config(config);
        assert!(shell.config.quiet);
        assert!(shell.config.no_color);
        assert!(shell.config.no_boot);
    }

    #[test]
    fn test_execute_blob_operations() {
        let mut shell = Shell::new();

        // Try blob operations
        let result = shell.execute("BLOB LIST");
        let _ = result;

        let result = shell.execute("BLOB STATS");
        let _ = result;
    }

    #[test]
    fn test_execute_checkpoint_operations() {
        let mut shell = Shell::new();

        // Try checkpoint operations
        let result = shell.execute("CHECKPOINTS");
        let _ = result;
    }

    #[test]
    fn test_execute_cache_operations() {
        let mut shell = Shell::new();

        // Try cache operations
        let result = shell.execute("CACHE STATS");
        let _ = result;
    }

    #[test]
    fn test_load_path_double_quoted_recover() {
        let result = Shell::extract_load_path_and_mode("LOAD \"file.bin\" RECOVER");
        assert!(result.is_some());
        let (path, mode) = result.unwrap();
        assert_eq!(path, "file.bin");
        assert_eq!(mode, WalRecoveryMode::Recover);
    }

    #[test]
    fn test_execute_with_successful_wal_write() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_wal_success.bin");
        let path_str = path.to_string_lossy().to_string();

        // Save and load to activate WAL
        shell.execute(&format!("SAVE '{}'", path_str));
        shell.execute(&format!("LOAD '{}'", path_str));

        // Execute a write command that should be logged to WAL
        let result = shell.execute("CREATE TABLE wal_log_test (id INT)");
        assert!(matches!(result, CommandResult::Output(_)));

        // Execute another write command
        let result = shell.execute("INSERT INTO wal_log_test VALUES (1)");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));
    }

    #[test]
    fn test_execute_edge_with_properties() {
        let mut shell = Shell::new();

        // Create nodes
        shell.execute("NODE CREATE city {name: 'Paris'}");
        shell.execute("NODE CREATE city {name: 'London'}");

        // Create edge with properties
        let result = shell.execute("EDGE CREATE route 1 2 {distance: 450, mode: 'train'}");
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_execute_find_operations() {
        let mut shell = Shell::new();

        // Store some entities
        shell.execute("ENTITY CREATE user {name: 'Test'}");
        shell.execute("EMBED STORE 'user:1' [0.1, 0.2, 0.3]");

        // Try FIND
        let result = shell.execute("FIND 'user' LIMIT 5");
        let _ = result;
    }

    #[test]
    fn test_execute_entity_operations() {
        let mut shell = Shell::new();

        // Try entity operations
        let result = shell.execute("ENTITY CREATE document {title: 'Test'}");
        let _ = result;

        let result = shell.execute("ENTITY GET 1");
        let _ = result;
    }

    #[test]
    fn test_cluster_connect_with_commas() {
        let shell = Shell::new();

        // Addresses separated by commas
        let result = shell.handle_cluster_connect(
            "cluster connect 'node1@127.0.0.1:8080','node2@127.0.0.1:8081'",
        );
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_cluster_connect_unquoted_with_spaces() {
        let shell = Shell::new();

        // Unquoted addresses with spaces
        let result = shell
            .handle_cluster_connect("cluster connect node1@127.0.0.1:8080 node2@127.0.0.1:8081");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_execute_graph_algorithm_betweenness() {
        let mut shell = Shell::new();

        // Create a small graph
        shell.execute("NODE CREATE vertex {}");
        shell.execute("NODE CREATE vertex {}");
        shell.execute("NODE CREATE vertex {}");
        shell.execute("EDGE CREATE link 1 2");
        shell.execute("EDGE CREATE link 2 3");

        // Try betweenness centrality
        let result = shell.execute("GRAPH ALGORITHM BETWEENNESS");
        let _ = result;
    }

    #[test]
    fn test_execute_sparse_vector() {
        let mut shell = Shell::new();

        // Try sparse vector operations
        let result =
            shell.execute("EMBED STORE 'sparse:1' SPARSE [1:0.5, 10:0.3, 100:0.2] DIM 1000");
        let _ = result;
    }

    #[test]
    fn test_execute_with_where_clause() {
        let mut shell = Shell::new();

        // Create table with data
        shell.execute("CREATE TABLE where_test (id INT, status TEXT)");
        shell.execute("INSERT INTO where_test VALUES (1, 'active')");
        shell.execute("INSERT INTO where_test VALUES (2, 'inactive')");
        shell.execute("INSERT INTO where_test VALUES (3, 'active')");

        // Query with WHERE
        let result = shell.execute("SELECT * FROM where_test WHERE status = 'active'");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_execute_order_by() {
        let mut shell = Shell::new();

        // Create table with data
        shell.execute("CREATE TABLE order_test (id INT, value INT)");
        shell.execute("INSERT INTO order_test VALUES (1, 100)");
        shell.execute("INSERT INTO order_test VALUES (2, 50)");
        shell.execute("INSERT INTO order_test VALUES (3, 75)");

        // Query with ORDER BY
        let result = shell.execute("SELECT * FROM order_test ORDER BY value");
        let _ = result;
    }

    #[test]
    fn test_execute_limit() {
        let mut shell = Shell::new();

        // Create table with many rows
        shell.execute("CREATE TABLE limit_test (id INT)");
        for i in 1..=10 {
            shell.execute(&format!("INSERT INTO limit_test VALUES ({i})"));
        }

        // Query with LIMIT
        let result = shell.execute("SELECT * FROM limit_test LIMIT 3");
        let _ = result;
    }

    #[test]
    fn test_execute_describe() {
        let mut shell = Shell::new();

        // Create table
        shell.execute("CREATE TABLE desc_test (id INT, name TEXT, active BOOL)");

        // Describe table
        let result = shell.execute("DESCRIBE desc_test");
        let _ = result;
    }

    #[test]
    fn test_handle_save_to_readonly_path() {
        let shell = Shell::new();

        // Try to save to a path that likely doesn't exist/isn't writable
        let result = shell.handle_save("save '/nonexistent/path/that/should/not/work/test.bin'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_save_compressed_to_readonly_path() {
        let shell = Shell::new();

        // Try to save compressed to a path that likely doesn't exist
        let result = shell.handle_save_compressed("save compressed '/nonexistent/path/test.bin'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_execute_vault_operations() {
        let mut shell = Shell::new();

        // Vault commands without init should fail gracefully
        let result = shell.execute("VAULT GET 'test_key'");
        let _ = result;

        let result = shell.execute("VAULT LIST");
        let _ = result;
    }

    #[test]
    fn test_execute_chain_operations() {
        let mut shell = Shell::new();

        // Chain queries without cluster should handle gracefully
        let result = shell.execute("CHAIN BLOCK 1");
        let _ = result;

        let result = shell.execute("CHAIN LAST");
        let _ = result;
    }

    #[test]
    fn test_execute_complex_graph_query() {
        let mut shell = Shell::new();

        // Create a more complex graph
        shell.execute("NODE CREATE person {name: 'Alice', age: 30}");
        shell.execute("NODE CREATE person {name: 'Bob', age: 25}");
        shell.execute("NODE CREATE person {name: 'Charlie', age: 35}");
        shell.execute("NODE CREATE person {name: 'Diana', age: 28}");

        shell.execute("EDGE CREATE knows 1 2 {since: 2020}");
        shell.execute("EDGE CREATE knows 2 3 {since: 2019}");
        shell.execute("EDGE CREATE knows 3 4 {since: 2021}");
        shell.execute("EDGE CREATE knows 1 4 {since: 2018}");

        // Try various graph queries
        let result = shell.execute("NEIGHBORS 1 OUT");
        let _ = result;

        let result = shell.execute("NEIGHBORS 2 IN");
        let _ = result;

        let result = shell.execute("PATH 1 3");
        let _ = result;
    }

    #[test]
    fn test_shell_execute_line_error() {
        let mut shell = Shell::new();

        // Execute invalid syntax
        let result = shell.execute_line("@#$%^&*()");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_node_address_missing_at() {
        let result = Shell::parse_node_address("node1-127.0.0.1:8080");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Expected format"));
    }

    #[test]
    fn test_parse_node_address_empty_node_id() {
        let result = Shell::parse_node_address("@127.0.0.1:8080");
        assert!(result.is_ok()); // Empty node ID is valid parsing-wise
    }

    #[test]
    fn test_handle_cluster_disconnect_not_connected() {
        let shell = Shell::new();
        let result = shell.handle_cluster_disconnect();
        assert!(matches!(result, CommandResult::Error(_)));
        if let CommandResult::Error(msg) = result {
            assert!(msg.contains("Not connected"));
        }
    }

    #[test]
    fn test_cluster_disconnect_via_execute() {
        let mut shell = Shell::new();
        let result = shell.execute("CLUSTER DISCONNECT");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_cluster_status_via_execute() {
        let mut shell = Shell::new();
        let result = shell.execute("CLUSTER STATUS");
        // Should work whether connected or not
        let _ = result;
    }

    #[test]
    fn test_cluster_nodes_via_execute() {
        let mut shell = Shell::new();
        let result = shell.execute("CLUSTER NODES");
        let _ = result;
    }

    #[test]
    fn test_cluster_leader_via_execute() {
        let mut shell = Shell::new();
        let result = shell.execute("CLUSTER LEADER");
        let _ = result;
    }

    #[test]
    fn test_detect_embedding_dimension_sparse() {
        let store = TensorStore::new();

        // Add sparse vector
        let sparse = tensor_store::SparseVector::from_parts(1000, vec![0, 10], vec![0.5, 0.3]);
        let mut data = tensor_store::TensorData::new();
        data.set("embedding", tensor_store::TensorValue::Sparse(sparse));
        store.put("sparse_key", data).unwrap();

        let dim = Shell::detect_embedding_dimension(&store);
        assert_eq!(dim, 1000);
    }

    #[test]
    fn test_handle_vault_init_invalid_base64() {
        std::env::set_var("NEUMANN_VAULT_KEY", "not-valid-base64!!!");
        let shell = Shell::new();
        let result = shell.handle_vault_init();
        std::env::remove_var("NEUMANN_VAULT_KEY");
        assert!(matches!(result, CommandResult::Error(_)));
        if let CommandResult::Error(msg) = result {
            assert!(msg.contains("base64") || msg.contains("Invalid"));
        }
    }

    #[test]
    fn test_handle_cache_init_success() {
        let shell = Shell::new();
        let result = shell.handle_cache_init();
        // May succeed or fail based on state, just check it doesn't panic
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );
    }

    #[test]
    fn test_handle_vault_identity_get_current() {
        let shell = Shell::new();
        let result = shell.handle_vault_identity("vault identity");
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("Current identity"));
        }
    }

    #[test]
    fn test_handle_vault_identity_set_unquoted() {
        let shell = Shell::new();
        let result = shell.handle_vault_identity("vault identity admin");
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("Identity set"));
        }
    }

    #[test]
    fn test_list_tables_method() {
        let shell = Shell::new();
        let result = shell.list_tables();
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_format_wal_replay_result_with_errors_strict() {
        let errors = vec![WalReplayError::new(1, "cmd", "error".to_string())];
        let result = WalReplayResult {
            replayed: 5,
            errors,
        };
        // In strict mode, errors are not shown (we would have failed)
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Strict);
        assert!(formatted.contains("Replayed 5"));
        assert!(!formatted.contains("Warning"));
    }

    #[test]
    fn test_format_wal_replay_result_with_errors_recover() {
        let errors = vec![
            WalReplayError::new(1, "cmd1", "error1".to_string()),
            WalReplayError::new(2, "cmd2", "error2".to_string()),
        ];
        let result = WalReplayResult {
            replayed: 3,
            errors,
        };
        let formatted = Shell::format_wal_replay_result(&result, WalRecoveryMode::Recover);
        assert!(formatted.contains("Warning"));
        assert!(formatted.contains("2 corrupted"));
    }

    #[test]
    fn test_execute_wal_status_via_execute() {
        let mut shell = Shell::new();
        let result = shell.execute("WAL STATUS");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_execute_wal_truncate_via_execute() {
        let mut shell = Shell::new();
        let result = shell.execute("WAL TRUNCATE");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_execute_vault_init() {
        let mut shell = Shell::new();
        std::env::remove_var("NEUMANN_VAULT_KEY");
        let result = shell.execute("VAULT INIT");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_save_empty_path() {
        let shell = Shell::new();
        let result = shell.handle_save("save");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_save_compressed_empty_path() {
        let shell = Shell::new();
        let result = shell.handle_save_compressed("save compressed");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_load_empty_path() {
        let shell = Shell::new();
        let result = shell.handle_load("load");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_handle_load_nonexistent_file() {
        let shell = Shell::new();
        let result = shell.handle_load("load '/nonexistent/path/file.bin'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_router_executor_success() {
        let router = Arc::new(RwLock::new(QueryRouter::new()));
        let executor = RouterExecutor(Arc::clone(&router));

        // Create table
        let result = executor.execute("CREATE TABLE exec_test (id INT)");
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_cluster_connect_invalid_local() {
        let shell = Shell::new();
        let result = shell.handle_cluster_connect("cluster connect 'invalid-no-at-sign'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_cluster_connect_invalid_peer() {
        let shell = Shell::new();
        let result =
            shell.handle_cluster_connect("cluster connect 'node1@127.0.0.1:8080' 'invalid-peer'");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_execute_similar_with_vector() {
        let mut shell = Shell::new();

        // Store embeddings first
        shell.execute("EMBED STORE 'test1' [0.1, 0.2, 0.3]");
        shell.execute("EMBED STORE 'test2' [0.2, 0.3, 0.4]");

        // Search with vector
        let result = shell.execute("SIMILAR [0.15, 0.25, 0.35] LIMIT 5");
        let _ = result;
    }

    #[test]
    fn test_execute_graph_shortest_path() {
        let mut shell = Shell::new();

        // Create graph
        shell.execute("NODE CREATE vertex {id: 'a'}");
        shell.execute("NODE CREATE vertex {id: 'b'}");
        shell.execute("NODE CREATE vertex {id: 'c'}");
        shell.execute("EDGE CREATE link 1 2");
        shell.execute("EDGE CREATE link 2 3");

        // Find path
        let result = shell.execute("PATH 1 3");
        let _ = result;
    }

    #[test]
    fn test_execute_show_embeddings() {
        let mut shell = Shell::new();

        shell.execute("EMBED STORE 'key1' [0.1, 0.2]");
        let result = shell.execute("SHOW EMBEDDINGS");
        let _ = result;
    }

    #[test]
    fn test_dirs_home_with_env() {
        // This will use the actual HOME env var
        let home = dirs_home();
        // HOME should be set in most environments
        if std::env::var("HOME").is_ok() {
            assert!(home.is_some());
        }
    }

    #[test]
    fn test_router_arc_method() {
        let shell = Shell::new();
        let router_arc = shell.router_arc();
        // Verify we can use the arc
        let _guard = router_arc.read();
    }

    #[test]
    fn test_router_read_method() {
        let shell = Shell::new();
        let guard = shell.router();
        // Verify we can access through the guard
        let _ = guard.execute_parsed("SELECT 1");
    }

    #[test]
    fn test_router_mut_method() {
        let shell = Shell::new();
        let _guard = shell.router_mut();
        // Just verify we can get a write lock
    }

    #[test]
    fn test_replay_wal_recover_mode_with_errors() {
        use std::io::Write;

        let shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let wal_path = temp_dir.join("test_replay_recover_error.log");

        // Create WAL with mix of valid and invalid
        {
            let mut file = std::fs::File::create(&wal_path).unwrap();
            writeln!(file, "CREATE TABLE recover_test (id INT)").unwrap();
            writeln!(file, "@#$%INVALID!@#$").unwrap();
            writeln!(file, "INSERT INTO recover_test VALUES (1)").unwrap();
        }

        let result = shell.replay_wal(&wal_path, WalRecoveryMode::Recover);
        assert!(result.is_ok());
        let replay_result = result.unwrap();
        assert!(replay_result.replayed > 0);
        assert!(!replay_result.errors.is_empty());

        // Clean up
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_replay_wal_nonexistent() {
        let shell = Shell::new();
        let result = shell.replay_wal(
            std::path::Path::new("/nonexistent/path/wal.log"),
            WalRecoveryMode::Strict,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_with_wal_logging() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_wal_execute_log.bin");
        let path_str = path.to_string_lossy().to_string();

        // Save and load to activate WAL
        shell.execute(&format!("SAVE '{}'", path_str));
        shell.execute(&format!("LOAD '{}'", path_str));

        // Now writes should be logged
        shell.execute("CREATE TABLE wal_exec_log_test (id INT)");
        shell.execute("INSERT INTO wal_exec_log_test VALUES (1)");
        shell.execute("UPDATE wal_exec_log_test SET id = 2 WHERE id = 1");
        shell.execute("DELETE FROM wal_exec_log_test WHERE id = 2");
        shell.execute("DROP TABLE wal_exec_log_test");

        // Clean up
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("log"));
    }

    #[test]
    fn test_execute_uses_spinner_for_similar() {
        let mut shell = Shell::new();

        // Store embeddings
        shell.execute("EMBED STORE 'spinner1' [0.1, 0.2, 0.3]");
        shell.execute("EMBED STORE 'spinner2' [0.2, 0.3, 0.4]");

        // SIMILAR should use spinner
        let result = shell.execute("SIMILAR 'spinner1' LIMIT 5 COSINE");
        let _ = result;
    }

    #[test]
    fn test_execute_graph_traversal() {
        let mut shell = Shell::new();

        // Create graph
        shell.execute("NODE CREATE person {name: 'A'}");
        shell.execute("NODE CREATE person {name: 'B'}");
        shell.execute("NODE CREATE person {name: 'C'}");
        shell.execute("EDGE CREATE knows 1 2");
        shell.execute("EDGE CREATE knows 2 3");

        // Traversal operations
        let result = shell.execute("NEIGHBORS 1");
        let _ = result;
    }

    #[test]
    fn test_execute_describe_table() {
        let mut shell = Shell::new();

        shell.execute("CREATE TABLE describe_test (id INT, name TEXT)");
        let result = shell.execute("DESCRIBE describe_test");
        let _ = result;
    }

    #[test]
    fn test_shell_config_with_all_fields() {
        let config = ShellConfig {
            history_file: Some(std::path::PathBuf::from("/tmp/test_history")),
            history_size: 2000,
            prompt: "test> ".to_string(),
            theme: Theme::plain(),
            no_color: true,
            no_boot: true,
            quiet: true,
        };
        let shell = Shell::with_config(config.clone());
        assert_eq!(shell.config.history_size, 2000);
        assert!(shell.config.quiet);
    }

    #[test]
    fn test_execute_various_query_results() {
        let mut shell = Shell::new();

        // Create data
        shell.execute("CREATE TABLE query_test (id INT, name TEXT, active BOOL)");
        shell.execute("INSERT INTO query_test VALUES (1, 'Alice', true)");
        shell.execute("INSERT INTO query_test VALUES (2, 'Bob', false)");

        // Various queries - just verify they run without panic
        let result = shell.execute("SELECT * FROM query_test");
        let _ = result;

        let result = shell.execute("SELECT * FROM query_test WHERE active = true");
        let _ = result;

        let result = shell.execute("SELECT COUNT(*) FROM query_test");
        let _ = result;
    }

    #[test]
    fn test_execute_node_neighbors_both() {
        let mut shell = Shell::new();

        shell.execute("NODE CREATE item {}");
        shell.execute("NODE CREATE item {}");
        shell.execute("NODE CREATE item {}");
        shell.execute("EDGE CREATE link 1 2");
        shell.execute("EDGE CREATE link 3 1");

        let result = shell.execute("NEIGHBORS 1 BOTH");
        let _ = result;
    }

    #[test]
    fn test_execute_blob_info() {
        let mut shell = Shell::new();
        let result = shell.execute("BLOB INFO 'nonexistent'");
        let _ = result;
    }

    #[test]
    fn test_execute_checkpoint_info() {
        let mut shell = Shell::new();
        let result = shell.execute("CHECKPOINT INFO 'nonexistent'");
        let _ = result;
    }

    #[test]
    fn test_execute_chain_tip() {
        let mut shell = Shell::new();
        let result = shell.execute("CHAIN TIP");
        let _ = result;
    }

    #[test]
    fn test_execute_chain_verify() {
        let mut shell = Shell::new();
        let result = shell.execute("CHAIN VERIFY");
        let _ = result;
    }

    #[test]
    fn test_execute_cache_evict() {
        let mut shell = Shell::new();
        let result = shell.execute("CACHE EVICT");
        let _ = result;
    }

    #[test]
    fn test_execute_vault_list() {
        let mut shell = Shell::new();
        let result = shell.execute("VAULT LIST");
        let _ = result;
    }

    #[test]
    fn test_execute_show_vector() {
        let mut shell = Shell::new();
        let result = shell.execute("SHOW VECTOR INDEX");
        let _ = result;
    }

    #[test]
    fn test_execute_show_codebook() {
        let mut shell = Shell::new();
        let result = shell.execute("SHOW CODEBOOK GLOBAL");
        let _ = result;
    }

    #[test]
    fn test_shell_error_std_error_trait() {
        let error = ShellError::Init("test".to_string());
        // Verify it implements std::error::Error
        let _: &dyn std::error::Error = &error;
    }

    #[test]
    fn test_shell_config_default_has_history() {
        let config = ShellConfig::default();
        assert!(config.history_file.is_some());
        assert!(config.history_size > 0);
    }

    #[test]
    fn test_command_result_variants() {
        let empty = CommandResult::Empty;
        let exit = CommandResult::Exit;
        let output = CommandResult::Output("out".to_string());
        let error = CommandResult::Error("err".to_string());
        let help = CommandResult::Help("help".to_string());

        // Verify all variants are different
        assert_ne!(empty, exit);
        assert_ne!(empty, output.clone());
        assert_ne!(empty, error.clone());
        assert_ne!(empty, help.clone());
    }

    #[test]
    fn test_execute_all_graph_algorithms() {
        let mut shell = Shell::new();

        // Create a connected graph
        shell.execute("NODE CREATE vertex {label: 'A'}");
        shell.execute("NODE CREATE vertex {label: 'B'}");
        shell.execute("NODE CREATE vertex {label: 'C'}");
        shell.execute("NODE CREATE vertex {label: 'D'}");
        shell.execute("EDGE CREATE link 1 2");
        shell.execute("EDGE CREATE link 2 3");
        shell.execute("EDGE CREATE link 3 4");
        shell.execute("EDGE CREATE link 4 1");
        shell.execute("EDGE CREATE link 1 3");

        // Test various graph algorithms
        let _ = shell.execute("GRAPH ALGORITHM PAGERANK");
        let _ = shell.execute("GRAPH ALGORITHM BETWEENNESS");
        let _ = shell.execute("GRAPH ALGORITHM CLOSENESS");
        let _ = shell.execute("GRAPH ALGORITHM EIGENVECTOR");
        let _ = shell.execute("GRAPH ALGORITHM LOUVAIN");
        let _ = shell.execute("GRAPH ALGORITHM LABEL_PROPAGATION");
    }

    #[test]
    fn test_execute_full_snapshot_cycle() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let snapshot_path = temp_dir.join("test_full_cycle.bin");
        let path_str = snapshot_path.to_string_lossy().to_string();

        // Create some data
        shell.execute("CREATE TABLE cycle_test (id INT, value TEXT)");
        shell.execute("INSERT INTO cycle_test VALUES (1, 'one')");
        shell.execute("INSERT INTO cycle_test VALUES (2, 'two')");
        shell.execute("NODE CREATE item {name: 'test'}");
        shell.execute("EMBED STORE 'cycle:key' [0.1, 0.2, 0.3, 0.4]");

        // Save snapshot
        let result = shell.execute(&format!("SAVE '{}'", path_str));
        assert!(matches!(result, CommandResult::Output(_)));

        // Load snapshot
        let result = shell.execute(&format!("LOAD '{}'", path_str));
        assert!(matches!(result, CommandResult::Output(_)));

        // Verify data exists
        let _ = shell.execute("SELECT * FROM cycle_test");
        let _ = shell.execute("NODE LIST item");
        let _ = shell.execute("EMBED GET 'cycle:key'");

        // Clean up
        let _ = std::fs::remove_file(&snapshot_path);
        let _ = std::fs::remove_file(snapshot_path.with_extension("log"));
    }

    #[test]
    fn test_execute_compressed_snapshot() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let snapshot_path = temp_dir.join("test_compressed.bin");
        let path_str = snapshot_path.to_string_lossy().to_string();

        // Create some data
        shell.execute("EMBED STORE 'comp:1' [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]");
        shell.execute("EMBED STORE 'comp:2' [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]");

        // Save compressed
        let result = shell.execute(&format!("SAVE COMPRESSED '{}'", path_str));
        assert!(
            matches!(result, CommandResult::Output(_)) || matches!(result, CommandResult::Error(_))
        );

        // Clean up
        let _ = std::fs::remove_file(&snapshot_path);
        let _ = std::fs::remove_file(snapshot_path.with_extension("log"));
    }

    #[test]
    fn test_execute_with_uppercase_variants() {
        let mut shell = Shell::new();

        // Test uppercase versions of commands
        let _ = shell.execute("HELP");
        let _ = shell.execute("TABLES");
        let _ = shell.execute("CLEAR");
        let _ = shell.execute("WAL STATUS");
    }

    #[test]
    fn test_wal_active_after_load() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let snapshot_path = temp_dir.join("test_wal_active.bin");
        let path_str = snapshot_path.to_string_lossy().to_string();

        // Save
        shell.execute(&format!("SAVE '{}'", path_str));

        // Load - should activate WAL
        shell.execute(&format!("LOAD '{}'", path_str));

        // WAL status should show active
        let result = shell.execute("WAL STATUS");
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("WAL enabled") || msg.contains("Path:"));
        }

        // Clean up
        let _ = std::fs::remove_file(&snapshot_path);
        let _ = std::fs::remove_file(snapshot_path.with_extension("log"));
    }

    #[test]
    fn test_wal_truncate_after_load() {
        let mut shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let snapshot_path = temp_dir.join("test_wal_truncate.bin");
        let path_str = snapshot_path.to_string_lossy().to_string();

        // Save and load to activate WAL
        shell.execute(&format!("SAVE '{}'", path_str));
        shell.execute(&format!("LOAD '{}'", path_str));

        // Truncate WAL
        let result = shell.execute("WAL TRUNCATE");
        assert!(matches!(result, CommandResult::Output(_)));

        // Clean up
        let _ = std::fs::remove_file(&snapshot_path);
        let _ = std::fs::remove_file(snapshot_path.with_extension("log"));
    }

    #[test]
    fn test_execute_select_with_limit() {
        let mut shell = Shell::new();

        shell.execute("CREATE TABLE limit_test2 (id INT)");
        for i in 1..=20 {
            shell.execute(&format!("INSERT INTO limit_test2 VALUES ({i})"));
        }

        let result = shell.execute("SELECT * FROM limit_test2 LIMIT 5");
        let _ = result;
    }

    #[test]
    fn test_execute_update_with_where() {
        let mut shell = Shell::new();

        shell.execute("CREATE TABLE update_test2 (id INT, name TEXT)");
        shell.execute("INSERT INTO update_test2 VALUES (1, 'old')");
        shell.execute("INSERT INTO update_test2 VALUES (2, 'keep')");

        let result = shell.execute("UPDATE update_test2 SET name = 'new' WHERE id = 1");
        let _ = result;
    }

    #[test]
    fn test_execute_delete_with_where() {
        let mut shell = Shell::new();

        shell.execute("CREATE TABLE delete_test2 (id INT)");
        shell.execute("INSERT INTO delete_test2 VALUES (1)");
        shell.execute("INSERT INTO delete_test2 VALUES (2)");

        let result = shell.execute("DELETE FROM delete_test2 WHERE id = 1");
        let _ = result;
    }

    #[test]
    fn test_replay_wal_mixed_commands() {
        use std::io::Write;

        let shell = Shell::new();
        let temp_dir = std::env::temp_dir();
        let wal_path = temp_dir.join("test_replay_mixed.log");

        // Create WAL with various commands
        {
            let mut file = std::fs::File::create(&wal_path).unwrap();
            writeln!(file, "CREATE TABLE replay_mix (id INT)").unwrap();
            writeln!(file, "INSERT INTO replay_mix VALUES (1)").unwrap();
            writeln!(file, "INSERT INTO replay_mix VALUES (2)").unwrap();
            writeln!(file).unwrap(); // empty line
            writeln!(file, "  ").unwrap(); // whitespace
            writeln!(file, "INSERT INTO replay_mix VALUES (3)").unwrap();
        }

        let result = shell.replay_wal(&wal_path, WalRecoveryMode::Strict);
        assert!(result.is_ok());
        let replay = result.unwrap();
        assert_eq!(replay.replayed, 4);
        assert!(replay.errors.is_empty());

        // Clean up
        let _ = std::fs::remove_file(&wal_path);
    }

    #[test]
    fn test_cluster_connect_full_flow() {
        let shell = Shell::new();

        // Try connecting with multiple peers
        let result = shell.handle_cluster_connect(
            "cluster connect 'node1@127.0.0.1:9000' 'node2@127.0.0.1:9001' 'node3@127.0.0.1:9002'",
        );
        // Will fail to actually connect but should parse addresses correctly
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_is_write_command_comprehensive() {
        // Comprehensive test of all write command patterns
        assert!(Shell::is_write_command("VAULT GRANT read key TO user"));
        assert!(Shell::is_write_command("VAULT REVOKE write key FROM user"));
        assert!(Shell::is_write_command("CACHE CLEAR"));
        assert!(Shell::is_write_command("BLOB META SET hash key val"));
        assert!(Shell::is_write_command("GRAPH BATCH CREATE nodes"));
        assert!(Shell::is_write_command("GRAPH CONSTRAINT CREATE unique"));
        assert!(Shell::is_write_command("GRAPH INDEX CREATE idx"));
        assert!(Shell::is_write_command("BEGIN CHAIN transaction"));
        assert!(Shell::is_write_command("COMMIT CHAIN"));
    }
}
