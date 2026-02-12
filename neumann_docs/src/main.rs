// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Neumann Documentation CLI
//!
//! Index and search Neumann documentation using all three engines.

use anyhow::Result;
use clap::Parser;

use neumann_docs::commands::{execute_command, Cli};

fn main() -> Result<()> {
    let cli = Cli::parse();
    execute_command(&cli.command)
}
