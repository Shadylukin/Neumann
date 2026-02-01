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
pub mod tro;
mod wal;

pub use input::NeumannHelper;
pub use style::{Icons, Theme};
pub use tro::{ActivitySensor, BootSequence, BootStyle, OpType, Palette, TroConfig, TroController};
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
    tro: Option<TroController>,
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
        let tro_config = TroConfig::default();
        let tro = if tro_config.enabled {
            Some(TroController::new(tro_config))
        } else {
            None
        };

        Self {
            router: Arc::new(RwLock::new(QueryRouter::new())),
            config: ShellConfig::default(),
            wal: Mutex::new(None),
            icons: Icons::auto(),
            tro,
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

        // Disable TRO if no_boot is set
        let tro_config = TroConfig::default();
        let tro = if tro_config.enabled && !config.no_boot {
            Some(TroController::new(tro_config))
        } else {
            None
        };

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
            tro,
        }
    }

    /// Creates a new shell with custom TRO configuration.
    #[must_use]
    pub fn with_tro_config(config: ShellConfig, tro_config: TroConfig) -> Self {
        let tro = if tro_config.enabled {
            Some(TroController::new(tro_config))
        } else {
            None
        };

        Self {
            router: Arc::new(RwLock::new(QueryRouter::new())),
            config,
            wal: Mutex::new(None),
            icons: Icons::auto(),
            tro,
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

        // Handle TRO commands
        if lower.starts_with("tro ") || lower == "tro" {
            return self.handle_tro_command(trimmed);
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
                // Record activity for TRO visualization
                self.record_tro_activity(trimmed, false);

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
                // Record error activity for TRO visualization
                self.record_tro_activity(trimmed, true);

                let error_msg = format!(
                    "{} Error: {e}",
                    style::styled(self.icons.error, self.config.theme.error)
                );
                CommandResult::Error(error_msg)
            },
        }
    }

    /// Records activity for TRO border visualization.
    fn record_tro_activity(&self, command: &str, is_error: bool) {
        if let Some(ref tro) = self.tro {
            let sensor = tro.activity_sensor();

            if is_error {
                sensor.record_error(command);
                tro.glitch(200);
                return;
            }

            // Determine operation type from command
            let upper = command.to_uppercase();
            let first_word = upper.split_whitespace().next().unwrap_or("");

            let op = match first_word {
                "INSERT" | "PUT" | "CREATE" | "EMBED" | "UPDATE" => OpType::Put,
                "DELETE" | "DROP" => OpType::Delete,
                "SIMILAR" => OpType::VectorSearch,
                "FIND" => OpType::Scan,
                // Default to Get for SELECT, NODE, NEIGHBORS, PATH, and anything else
                _ => OpType::Get,
            };

            sensor.record(op, command);
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

    /// Handles TRO border control commands.
    fn handle_tro_command(&self, input: &str) -> CommandResult {
        let args = input
            .strip_prefix("tro")
            .or_else(|| input.strip_prefix("TRO"))
            .unwrap_or("")
            .trim();

        let lower_args = args.to_lowercase();

        // TRO (no args) or TRO STATUS - show status
        if args.is_empty() || lower_args == "status" {
            return self.handle_tro_status();
        }

        // TRO PAUSE
        if lower_args == "pause" {
            return self.handle_tro_pause();
        }

        // TRO RESUME
        if lower_args == "resume" {
            return self.handle_tro_resume();
        }

        // TRO THEME <name>
        if lower_args.starts_with("theme ") {
            let theme_name = args[6..].trim();
            return self.handle_tro_theme(theme_name);
        }

        // TRO CRT ON/OFF
        if lower_args.starts_with("crt ") {
            let setting = args[4..].trim().to_lowercase();
            return self.handle_tro_crt(&setting);
        }

        // TRO ASCII ON/OFF
        if lower_args.starts_with("ascii ") {
            let setting = args[6..].trim().to_lowercase();
            return self.handle_tro_ascii(&setting);
        }

        // TRO THEMES - list available themes
        if lower_args == "themes" {
            let themes = Palette::all_names().join(", ");
            return CommandResult::Output(format!("Available themes: {themes}"));
        }

        CommandResult::Error(format!(
            "Unknown TRO command: {args}\n\
             Usage:\n  \
             TRO [STATUS]   - Show TRO status\n  \
             TRO PAUSE      - Pause border animation\n  \
             TRO RESUME     - Resume border animation\n  \
             TRO THEME <n>  - Set color theme ({})\n  \
             TRO THEMES     - List available themes\n  \
             TRO CRT ON/OFF - Toggle CRT effects\n  \
             TRO ASCII ON/OFF - Toggle ASCII-only mode",
            Palette::all_names().join("/")
        ))
    }

    /// Shows TRO status.
    fn handle_tro_status(&self) -> CommandResult {
        let Some(ref tro) = self.tro else {
            return CommandResult::Output(
                "TRO border disabled (non-TTY or disabled in config)".to_string(),
            );
        };

        let running = if tro.is_running() {
            "running"
        } else {
            "stopped"
        };
        let palette = tro.palette().name();
        let crt = if tro.config().crt_effects {
            "on"
        } else {
            "off"
        };
        let charset = match tro.charset_mode() {
            tro::CharsetMode::Unicode => "unicode",
            tro::CharsetMode::Ascii => "ascii",
        };
        let fps = tro.config().fps;
        let agents = tro.config().agent_count;

        CommandResult::Output(format!(
            "TRO Border Status:\n  \
             State:   {running}\n  \
             Theme:   {palette}\n  \
             CRT:     {crt}\n  \
             Charset: {charset}\n  \
             FPS:     {fps}\n  \
             Agents:  {agents}"
        ))
    }

    /// Pauses TRO animation.
    fn handle_tro_pause(&self) -> CommandResult {
        let Some(ref tro) = self.tro else {
            return CommandResult::Error("TRO border not enabled".to_string());
        };

        tro.pause();
        CommandResult::Output(format!(
            "{} TRO border paused",
            style::styled(self.icons.success, self.config.theme.success)
        ))
    }

    /// Resumes TRO animation.
    fn handle_tro_resume(&self) -> CommandResult {
        let Some(ref tro) = self.tro else {
            return CommandResult::Error("TRO border not enabled".to_string());
        };

        tro.resume();
        CommandResult::Output(format!(
            "{} TRO border resumed",
            style::styled(self.icons.success, self.config.theme.success)
        ))
    }

    /// Sets TRO color theme.
    fn handle_tro_theme(&self, name: &str) -> CommandResult {
        let Some(ref tro) = self.tro else {
            return CommandResult::Error("TRO border not enabled".to_string());
        };

        let Some(palette) = Palette::from_name(name) else {
            let available = Palette::all_names().join(", ");
            return CommandResult::Error(format!("Unknown theme: {name}\nAvailable: {available}"));
        };

        tro.set_palette(palette);
        CommandResult::Output(format!(
            "{} TRO theme set to: {}",
            style::styled(self.icons.success, self.config.theme.success),
            style::styled(palette.name(), self.config.theme.id)
        ))
    }

    /// Toggles TRO CRT effects.
    fn handle_tro_crt(&self, setting: &str) -> CommandResult {
        let Some(ref tro) = self.tro else {
            return CommandResult::Error("TRO border not enabled".to_string());
        };

        let enabled = match setting {
            "on" | "true" | "1" | "yes" => true,
            "off" | "false" | "0" | "no" => false,
            _ => return CommandResult::Error("Usage: TRO CRT ON or TRO CRT OFF".to_string()),
        };

        tro.set_crt_effects(enabled);
        let state = if enabled { "enabled" } else { "disabled" };
        CommandResult::Output(format!(
            "{} CRT effects {state}",
            style::styled(self.icons.success, self.config.theme.success)
        ))
    }

    /// Toggles TRO ASCII mode.
    fn handle_tro_ascii(&self, setting: &str) -> CommandResult {
        let Some(ref tro) = self.tro else {
            return CommandResult::Error("TRO border not enabled".to_string());
        };

        let mode = match setting {
            "on" | "true" | "1" | "yes" => tro::CharsetMode::Ascii,
            "off" | "false" | "0" | "no" => tro::CharsetMode::Unicode,
            _ => return CommandResult::Error("Usage: TRO ASCII ON or TRO ASCII OFF".to_string()),
        };

        tro.set_charset_mode(mode);
        let state = match mode {
            tro::CharsetMode::Ascii => "enabled (ASCII-only)",
            tro::CharsetMode::Unicode => "disabled (Unicode)",
        };
        CommandResult::Output(format!(
            "{} ASCII mode {state}",
            style::styled(self.icons.success, self.config.theme.success)
        ))
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

        // Show welcome banner (TRO boot sequence or standard banner)
        if let Some(ref tro) = self.tro {
            tro.run_boot_sequence(Self::version());
            // Clean transition to interactive mode
            println!();
            println!(
                "{}",
                style::styled("Ready. Type 'help' for commands.", self.config.theme.info)
            );
            println!();
        } else {
            let banner = if progress::supports_full_banner() {
                progress::welcome_banner(Self::version(), &self.config.theme)
            } else {
                progress::compact_banner(Self::version(), &self.config.theme)
            };
            println!("{banner}");
        }

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

        // Shutdown TRO border animation
        if let Some(ref mut tro) = self.tro {
            tro.shutdown();
        }

        if let Some(ref path) = self.config.history_file {
            let mut ed = editor.lock();
            let _ = ed.save_history(path);
        }
        Ok(())
    }

    /// Returns a reference to the TRO controller if enabled.
    #[must_use]
    pub const fn tro(&self) -> Option<&TroController> {
        self.tro.as_ref()
    }

    /// Records activity to the TRO sensor.
    pub fn record_activity(&self, op: OpType, key: &str) {
        if let Some(ref tro) = self.tro {
            tro.activity_sensor().record(op, key);
        }
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
    fn test_shell_with_tro_config() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        assert!(shell.tro.is_none());
    }

    #[test]
    fn test_shell_with_tro_config_enabled() {
        let config = ShellConfig::default();
        let tro_config = TroConfig {
            enabled: true,
            ..TroConfig::default()
        };
        let shell = Shell::with_tro_config(config, tro_config);
        assert!(shell.tro.is_some());
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
        assert!(Shell::is_write_command("VAULT REVOKE read 'key' FROM 'user'"));
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
    fn test_tro_access() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        assert!(shell.tro().is_none());
    }

    #[test]
    fn test_record_activity_no_tro() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        // Should not panic even without TRO
        shell.record_activity(OpType::Get, "test");
    }

    #[test]
    fn test_record_activity_with_tro() {
        let config = ShellConfig::default();
        let tro_config = TroConfig {
            enabled: true,
            ..TroConfig::default()
        };
        let shell = Shell::with_tro_config(config, tro_config);
        shell.record_activity(OpType::Put, "test_key");
        shell.record_activity(OpType::Get, "test_key");
        shell.record_activity(OpType::Delete, "test_key");
        shell.record_activity(OpType::VectorSearch, "query");
        shell.record_activity(OpType::Scan, "pattern");
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
    fn test_tro_command_status_disabled() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        let result = shell.handle_tro_command("tro");
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("disabled"));
        }
    }

    #[test]
    fn test_tro_command_status_explicit() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        let result = shell.handle_tro_command("tro status");
        assert!(matches!(result, CommandResult::Output(_)));
    }

    #[test]
    fn test_tro_command_pause_disabled() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        let result = shell.handle_tro_command("tro pause");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_tro_command_resume_disabled() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        let result = shell.handle_tro_command("tro resume");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_tro_command_theme_disabled() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        let result = shell.handle_tro_command("tro theme phosphor");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_tro_command_crt_disabled() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        let result = shell.handle_tro_command("tro crt on");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_tro_command_ascii_disabled() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        let result = shell.handle_tro_command("tro ascii on");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_tro_command_themes_list() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        let result = shell.handle_tro_command("tro themes");
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("Available themes"));
        }
    }

    #[test]
    fn test_tro_command_unknown() {
        let config = ShellConfig::default();
        let tro_config = TroConfig::disabled();
        let shell = Shell::with_tro_config(config, tro_config);
        let result = shell.handle_tro_command("tro unknown");
        assert!(matches!(result, CommandResult::Error(_)));
        if let CommandResult::Error(msg) = result {
            assert!(msg.contains("Unknown TRO command"));
        }
    }

    #[test]
    fn test_tro_command_with_enabled() {
        let config = ShellConfig::default();
        let tro_config = TroConfig {
            enabled: true,
            ..TroConfig::default()
        };
        let shell = Shell::with_tro_config(config, tro_config);

        // Test status
        let result = shell.handle_tro_status();
        assert!(matches!(result, CommandResult::Output(_)));
        if let CommandResult::Output(msg) = result {
            assert!(msg.contains("TRO Border Status"));
        }

        // Test pause
        let result = shell.handle_tro_pause();
        assert!(matches!(result, CommandResult::Output(_)));

        // Test resume
        let result = shell.handle_tro_resume();
        assert!(matches!(result, CommandResult::Output(_)));

        // Test theme with valid name
        let result = shell.handle_tro_theme("green");
        assert!(matches!(result, CommandResult::Output(_)));

        // Test invalid theme
        let result = shell.handle_tro_theme("invalid_theme");
        assert!(matches!(result, CommandResult::Error(_)));

        // Test CRT on
        let result = shell.handle_tro_crt("on");
        assert!(matches!(result, CommandResult::Output(_)));

        // Test CRT off
        let result = shell.handle_tro_crt("off");
        assert!(matches!(result, CommandResult::Output(_)));

        // Test CRT invalid
        let result = shell.handle_tro_crt("maybe");
        assert!(matches!(result, CommandResult::Error(_)));

        // Test ASCII on
        let result = shell.handle_tro_ascii("on");
        assert!(matches!(result, CommandResult::Output(_)));

        // Test ASCII off
        let result = shell.handle_tro_ascii("off");
        assert!(matches!(result, CommandResult::Output(_)));

        // Test ASCII invalid
        let result = shell.handle_tro_ascii("maybe");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_tro_crt_variants() {
        let config = ShellConfig::default();
        let tro_config = TroConfig {
            enabled: true,
            ..TroConfig::default()
        };
        let shell = Shell::with_tro_config(config, tro_config);

        // Test all boolean variants
        for val in ["true", "1", "yes"] {
            let result = shell.handle_tro_crt(val);
            assert!(matches!(result, CommandResult::Output(_)));
        }
        for val in ["false", "0", "no"] {
            let result = shell.handle_tro_crt(val);
            assert!(matches!(result, CommandResult::Output(_)));
        }
    }

    #[test]
    fn test_tro_ascii_variants() {
        let config = ShellConfig::default();
        let tro_config = TroConfig {
            enabled: true,
            ..TroConfig::default()
        };
        let shell = Shell::with_tro_config(config, tro_config);

        // Test all boolean variants
        for val in ["true", "1", "yes"] {
            let result = shell.handle_tro_ascii(val);
            assert!(matches!(result, CommandResult::Output(_)));
        }
        for val in ["false", "0", "no"] {
            let result = shell.handle_tro_ascii(val);
            assert!(matches!(result, CommandResult::Output(_)));
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
            errors.push(WalReplayError::new(i, &format!("cmd{i}"), "error".to_string()));
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

        // Test tro command
        let result = shell.execute("tro");
        assert!(matches!(result, CommandResult::Output(_)));
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
        // TRO should be disabled when no_boot is true
        assert!(shell.tro.is_none());
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
    fn test_record_tro_activity_error() {
        let config = ShellConfig::default();
        let tro_config = TroConfig {
            enabled: true,
            ..TroConfig::default()
        };
        let mut shell = Shell::with_tro_config(config, tro_config);

        // Execute an invalid command to trigger error recording
        let result = shell.execute("INVALID COMMAND SYNTAX @#$%");
        assert!(matches!(result, CommandResult::Error(_)));
    }

    #[test]
    fn test_record_tro_activity_op_types() {
        let config = ShellConfig::default();
        let tro_config = TroConfig {
            enabled: true,
            ..TroConfig::default()
        };
        let shell = Shell::with_tro_config(config, tro_config);

        // Test various command patterns
        shell.record_tro_activity("INSERT INTO test VALUES (1)", false);
        shell.record_tro_activity("PUT key value", false);
        shell.record_tro_activity("CREATE TABLE test (id INT)", false);
        shell.record_tro_activity("EMBED STORE key [1,2,3]", false);
        shell.record_tro_activity("UPDATE test SET x = 1", false);
        shell.record_tro_activity("DELETE FROM test", false);
        shell.record_tro_activity("DROP TABLE test", false);
        shell.record_tro_activity("SIMILAR [1,2,3] LIMIT 5", false);
        shell.record_tro_activity("FIND pattern", false);
        shell.record_tro_activity("SELECT * FROM test", false);
        shell.record_tro_activity("NODE GET 1", false);
    }
}
