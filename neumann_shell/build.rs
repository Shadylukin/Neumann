// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
// Build script for generating shell completions and man pages.

use clap::CommandFactory;
use clap_complete::{generate_to, Shell};
use clap_mangen::Man;
use std::{env, fs, io::Error};

// Include the CLI definition from src/cli.rs
// This allows us to generate completions at build time
mod cli_gen {
    use clap::{Parser, ValueEnum};
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
}

fn main() -> Result<(), Error> {
    let out_dir = std::path::PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let completions_dir = out_dir.join("completions");
    let man_dir = out_dir.join("man");

    fs::create_dir_all(&completions_dir)?;
    fs::create_dir_all(&man_dir)?;

    let mut cmd = cli_gen::Cli::command();
    cmd = cmd.name("neumann");

    // Generate shell completions
    for shell in [Shell::Bash, Shell::Zsh, Shell::Fish, Shell::PowerShell] {
        generate_to(shell, &mut cmd, "neumann", &completions_dir)?;
    }

    // Generate man page
    let man = Man::new(cmd);
    let mut buffer = Vec::new();
    man.render(&mut buffer)?;
    fs::write(man_dir.join("neumann.1"), buffer)?;

    println!("cargo:rerun-if-changed=src/cli.rs");
    Ok(())
}
