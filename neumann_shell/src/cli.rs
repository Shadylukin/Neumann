// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
// Command-line argument parsing for Neumann shell.

use clap::{Parser, ValueEnum};
#[allow(unused_imports)]
use std::path::PathBuf;

/// Neumann: Unified tensor database for relational, graph, and vector data.
#[derive(Parser, Debug)]
#[command(name = "neumann")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Execute a single query and exit
    #[arg(short = 'c', long = "command")]
    pub command: Option<String>,

    /// Execute queries from file
    #[arg(short = 'f', long = "file")]
    pub file: Option<PathBuf>,

    /// Output format for non-interactive mode
    #[arg(short = 'o', long = "output", value_enum, default_value_t = OutputFormat::Table)]
    pub output_format: OutputFormat,

    /// Disable colored output
    #[arg(long = "no-color", env = "NO_COLOR")]
    pub no_color: bool,

    /// Skip boot sequence animation
    #[arg(long = "no-boot")]
    pub no_boot: bool,

    /// Quiet mode: suppress non-essential output
    #[arg(short = 'q', long = "quiet")]
    pub quiet: bool,
}

/// Output format for non-interactive query results.
#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable table format (default)
    #[default]
    Table,
    /// JSON format
    Json,
    /// CSV format
    Csv,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_verify() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_cli_default() {
        let cli = Cli::parse_from(["neumann"]);
        assert!(cli.command.is_none());
        assert!(cli.file.is_none());
        assert_eq!(cli.output_format, OutputFormat::Table);
        assert!(!cli.no_color);
        assert!(!cli.no_boot);
        assert!(!cli.quiet);
    }

    #[test]
    fn test_cli_command() {
        let cli = Cli::parse_from(["neumann", "-c", "SELECT 1"]);
        assert_eq!(cli.command, Some("SELECT 1".to_string()));
    }

    #[test]
    fn test_cli_file() {
        let cli = Cli::parse_from(["neumann", "-f", "queries.sql"]);
        assert_eq!(cli.file, Some(PathBuf::from("queries.sql")));
    }

    #[test]
    fn test_cli_output_format() {
        let cli = Cli::parse_from(["neumann", "-o", "json", "-c", "SELECT 1"]);
        assert_eq!(cli.output_format, OutputFormat::Json);

        let cli = Cli::parse_from(["neumann", "--output", "csv", "-c", "SELECT 1"]);
        assert_eq!(cli.output_format, OutputFormat::Csv);
    }

    #[test]
    fn test_cli_flags() {
        let cli = Cli::parse_from(["neumann", "--no-color", "--no-boot", "-q"]);
        assert!(cli.no_color);
        assert!(cli.no_boot);
        assert!(cli.quiet);
    }
}
