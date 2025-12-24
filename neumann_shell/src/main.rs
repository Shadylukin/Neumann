//! Neumann CLI entry point.

use neumann_shell::Shell;

fn main() {
    let shell = Shell::new();

    if let Err(e) = shell.run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
