# Installation

## Requirements

- **Rust**: 1.75 or later
- **Operating Systems**: macOS, Linux, Windows
- **Memory**: 4GB RAM minimum (more for large datasets)
- **Disk**: Depends on data size

## Quick Install

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann

# Build in release mode
cargo build --release

# The binary is at:
./target/release/neumann_shell
```

### Run the Shell

```bash
# Start interactive shell
./target/release/neumann_shell

# You should see:
# Neumann Shell v0.1.0
# Type 'help' for commands, 'exit' to quit.
# >
```

### Verify Installation

```bash
# In the shell, try:
> help
> CREATE TABLE test (id INT, name TEXT)
> INSERT INTO test VALUES (1, 'hello')
> SELECT * FROM test
> exit
```

## Install Rust

If you don't have Rust installed:

### macOS / Linux

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustc --version  # Should show 1.75.0 or later
```

### Windows

Download and run [rustup-init.exe](https://rustup.rs/)

## Development Setup

For contributors:

```bash
# Clone
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann

# Install pre-commit hook
./scripts/setup-hooks.sh

# Install coverage tool (optional)
cargo install cargo-llvm-cov

# Build and test
cargo build
cargo test

# Run all checks
./scripts/pre-commit
```

## Using as a Library

Add to your `Cargo.toml`:

```toml
[dependencies]
tensor_store = { git = "https://github.com/Shadylukin/Neumann" }
query_router = { git = "https://github.com/Shadylukin/Neumann" }
```

Example usage:

```rust
use query_router::QueryRouter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut router = QueryRouter::new();

    // Create a table
    router.execute_parsed("CREATE TABLE users (id INT, name TEXT)")?;

    // Insert data
    router.execute_parsed("INSERT INTO users VALUES (1, 'Alice')")?;

    // Query
    let result = router.execute_parsed("SELECT * FROM users")?;
    println!("{:?}", result);

    Ok(())
}
```

## Troubleshooting

### Build Errors

**"error: linker not found"**
```bash
# macOS: Install Xcode command line tools
xcode-select --install

# Ubuntu/Debian
sudo apt install build-essential

# Fedora
sudo dnf install gcc
```

**"rustc version too old"**
```bash
rustup update stable
```

### Runtime Issues

**"permission denied"**
```bash
chmod +x ./target/release/neumann_shell
```

**Out of memory with large datasets**
- Neumann supports tiered storage for datasets larger than RAM
- See [Tiered Storage](architecture.md#tiered-memory-storage-complete) in architecture docs

## Next Steps

- [Getting Started](getting-started.md) - Your first queries
- [Architecture](architecture.md) - System design
- [Query Reference](neumann-parser.md) - Full syntax reference
