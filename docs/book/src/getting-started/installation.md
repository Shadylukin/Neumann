# Installation

## Requirements

- Rust 1.75 or later
- Cargo (included with Rust)
- Git

## From crates.io (Coming Soon)

```bash
cargo install neumann
```

## From Source

```bash
# Clone the repository
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann

# Build in release mode
cargo build --release

# Run tests to verify
cargo test

# Install locally
cargo install --path neumann_shell
```

## Verify Installation

```bash
neumann --version
```

## Docker (Coming Soon)

```bash
docker pull neumann/neumann:latest
docker run -it neumann/neumann
```

## Platform Support

| Platform | Status |
|----------|--------|
| Linux x86_64 | Supported |
| macOS x86_64 | Supported |
| macOS ARM64 | Supported |
| Windows x86_64 | Experimental |

## Next Steps

- [Quick Start](quick-start.md) - Run your first queries
- [Building from Source](building-from-source.md) - Development setup
