// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Neumann CLI entry point.

use clap::Parser;
use neumann_shell::{cli::Cli, Shell, ShellConfig};
use std::fs;

fn main() {
    let cli = Cli::parse();

    let config = ShellConfig {
        no_color: cli.no_color,
        no_boot: cli.no_boot,
        quiet: cli.quiet,
        ..Default::default()
    };

    let mut shell = Shell::with_config(config);

    // Execute file mode
    if let Some(file) = cli.file {
        let content = match fs::read_to_string(&file) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error reading {}: {e}", file.display());
                std::process::exit(1);
            },
        };

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("--") || line.starts_with('#') {
                continue;
            }
            match shell.execute_line(line) {
                Ok(output) => {
                    if !output.is_empty() && !cli.quiet {
                        println!("{output}");
                    }
                },
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                },
            }
        }
        return;
    }

    // Execute command mode
    if let Some(cmd) = cli.command {
        match shell.execute_line(&cmd) {
            Ok(output) => {
                if !output.is_empty() {
                    println!("{output}");
                }
            },
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            },
        }
        return;
    }

    // Interactive mode
    if let Err(e) = shell.run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
