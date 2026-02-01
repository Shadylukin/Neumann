// SPDX-License-Identifier: MIT OR Apache-2.0
//! Write-Ahead Log for crash recovery.

use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

/// Write-Ahead Log for crash recovery.
///
/// Logs mutating commands to a file so they can be replayed after loading a snapshot.
/// The WAL is activated after LOAD and truncated after SAVE.
pub struct Wal {
    file: File,
    path: PathBuf,
}

impl Wal {
    /// Opens or creates a WAL file for appending.
    pub fn open_append(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    /// Appends a command to the WAL.
    pub fn append(&mut self, cmd: &str) -> std::io::Result<()> {
        writeln!(self.file, "{cmd}")?;
        self.file.flush()
    }

    /// Truncates the WAL (after a successful save).
    pub fn truncate(&mut self) -> std::io::Result<()> {
        self.file = File::create(&self.path)?;
        Ok(())
    }

    /// Returns the WAL file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the current WAL file size in bytes.
    pub fn size(&self) -> std::io::Result<u64> {
        std::fs::metadata(&self.path).map(|m| m.len())
    }

    /// Reads all commands from the WAL file.
    pub fn read_commands(path: &Path) -> std::io::Result<Vec<String>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut commands = Vec::new();

        for line in reader.lines() {
            let cmd = line?;
            let trimmed = cmd.trim();
            if !trimmed.is_empty() {
                commands.push(trimmed.to_string());
            }
        }

        Ok(commands)
    }
}

/// Recovery mode for WAL replay.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WalRecoveryMode {
    /// Fail-fast on any error (default, preserves consistency).
    #[default]
    Strict,
    /// Skip corrupted lines and continue replay, report warnings at end.
    Recover,
}

/// Result of WAL replay operation.
#[derive(Debug, Clone)]
pub struct WalReplayResult {
    /// Number of commands successfully replayed.
    pub replayed: usize,
    /// Errors encountered during replay (only populated in Recover mode).
    pub errors: Vec<WalReplayError>,
}

/// Error encountered during WAL replay.
#[derive(Debug, Clone)]
pub struct WalReplayError {
    /// Line number in the WAL file (1-indexed).
    pub line: usize,
    /// The command that failed (truncated if >80 chars).
    pub command: String,
    /// The error message.
    pub error: String,
}

impl WalReplayError {
    /// Creates a new WAL replay error.
    #[must_use]
    pub fn new(line: usize, command: &str, error: String) -> Self {
        let command = if command.len() > 80 {
            format!("{}...", &command[..77])
        } else {
            command.to_string()
        };
        Self {
            line,
            command,
            error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn test_wal_append_and_read() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_wal_append.wal");

        // Clean up any existing file
        let _ = std::fs::remove_file(&path);

        // Write commands
        {
            let mut wal = Wal::open_append(&path).unwrap();
            wal.append("INSERT INTO users VALUES (1, 'Alice')").unwrap();
            wal.append("INSERT INTO users VALUES (2, 'Bob')").unwrap();
        }

        // Read commands
        let commands = Wal::read_commands(&path).unwrap();
        assert_eq!(commands.len(), 2);
        assert!(commands[0].contains("Alice"));
        assert!(commands[1].contains("Bob"));

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_wal_truncate() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_wal_truncate.wal");

        // Clean up any existing file
        let _ = std::fs::remove_file(&path);

        // Write and truncate
        {
            let mut wal = Wal::open_append(&path).unwrap();
            wal.append("INSERT INTO users VALUES (1, 'Alice')").unwrap();
            wal.truncate().unwrap();
        }

        // Verify empty
        let mut content = String::new();
        File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        assert!(content.is_empty());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_wal_size() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_wal_size.wal");

        // Clean up any existing file
        let _ = std::fs::remove_file(&path);

        let mut wal = Wal::open_append(&path).unwrap();
        wal.append("test").unwrap();
        let size = wal.size().unwrap();
        assert!(size > 0);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_wal_path() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_wal_path.wal");

        // Clean up any existing file
        let _ = std::fs::remove_file(&path);

        let wal = Wal::open_append(&path).unwrap();
        assert_eq!(wal.path(), path);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_wal_replay_error() {
        let error = WalReplayError::new(1, "short command", "error msg".to_string());
        assert_eq!(error.line, 1);
        assert_eq!(error.command, "short command");
        assert_eq!(error.error, "error msg");
    }

    #[test]
    fn test_wal_replay_error_truncates_long_command() {
        let long_command = "x".repeat(100);
        let error = WalReplayError::new(1, &long_command, "error".to_string());
        assert!(error.command.len() < 100);
        assert!(error.command.ends_with("..."));
    }

    #[test]
    fn test_recovery_mode_default() {
        let mode = WalRecoveryMode::default();
        assert_eq!(mode, WalRecoveryMode::Strict);
    }

    #[test]
    fn test_wal_replay_result_debug() {
        let result = WalReplayResult {
            replayed: 5,
            errors: vec![],
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("WalReplayResult"));
    }

    #[test]
    fn test_wal_replay_result_clone() {
        let result = WalReplayResult {
            replayed: 10,
            errors: vec![WalReplayError::new(1, "cmd", "err".to_string())],
        };
        let cloned = result.clone();
        assert_eq!(cloned.replayed, 10);
        assert_eq!(cloned.errors.len(), 1);
    }

    #[test]
    fn test_wal_replay_error_debug() {
        let error = WalReplayError::new(3, "test", "msg".to_string());
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("WalReplayError"));
    }

    #[test]
    fn test_wal_replay_error_clone() {
        let error = WalReplayError::new(7, "cmd", "error".to_string());
        let cloned = error.clone();
        assert_eq!(cloned.line, 7);
        assert_eq!(cloned.command, "cmd");
    }

    #[test]
    fn test_wal_recovery_mode_eq() {
        assert_eq!(WalRecoveryMode::Strict, WalRecoveryMode::Strict);
        assert_eq!(WalRecoveryMode::Recover, WalRecoveryMode::Recover);
        assert_ne!(WalRecoveryMode::Strict, WalRecoveryMode::Recover);
    }

    #[test]
    fn test_wal_recovery_mode_debug() {
        let strict = format!("{:?}", WalRecoveryMode::Strict);
        assert!(strict.contains("Strict"));
        let recover = format!("{:?}", WalRecoveryMode::Recover);
        assert!(recover.contains("Recover"));
    }

    #[test]
    fn test_wal_recovery_mode_copy() {
        let mode = WalRecoveryMode::Recover;
        let copied: WalRecoveryMode = mode;
        assert_eq!(copied, WalRecoveryMode::Recover);
    }

    #[test]
    fn test_read_commands_empty_lines() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_wal_empty_lines.wal");

        // Clean up any existing file
        let _ = std::fs::remove_file(&path);

        // Write commands with empty lines
        {
            let mut file = File::create(&path).unwrap();
            writeln!(file, "cmd1").unwrap();
            writeln!(file).unwrap(); // empty line
            writeln!(file, "   ").unwrap(); // whitespace only
            writeln!(file, "cmd2").unwrap();
        }

        // Read commands - empty lines should be skipped
        let commands = Wal::read_commands(&path).unwrap();
        assert_eq!(commands.len(), 2);
        assert_eq!(commands[0], "cmd1");
        assert_eq!(commands[1], "cmd2");

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_wal_replay_error_exact_80_chars() {
        // Test with exactly 80 characters - should NOT truncate
        let cmd = "x".repeat(80);
        let error = WalReplayError::new(1, &cmd, "err".to_string());
        assert_eq!(error.command.len(), 80);
        assert!(!error.command.ends_with("..."));
    }

    #[test]
    fn test_wal_replay_error_81_chars() {
        // Test with 81 characters - should truncate
        let cmd = "x".repeat(81);
        let error = WalReplayError::new(1, &cmd, "err".to_string());
        assert!(error.command.ends_with("..."));
        assert_eq!(error.command.len(), 80); // 77 chars + "..."
    }
}
