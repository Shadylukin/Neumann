// SPDX-License-Identifier: MIT OR Apache-2.0
//! Neumann CLI entry point.

use neumann_shell::Shell;

fn main() {
    let mut shell = Shell::new();

    if let Err(e) = shell.run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
