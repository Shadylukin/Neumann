# Installation

Multiple installation methods are available depending on your needs.

## Quick Install (Recommended)

The easiest way to install Neumann is using the install script:

```bash
curl -sSfL https://raw.githubusercontent.com/Shadylukin/Neumann/main/install.sh | bash
```

This script will:

- Detect your platform (Linux x86_64, macOS x86_64, macOS ARM64)
- Download a pre-built binary if available
- Fall back to building from source if needed
- Install to `/usr/local/bin` or `~/.local/bin`
- Install shell completions and man pages

### Environment Variables

| Variable | Description |
| --- | --- |
| `NEUMANN_INSTALL_DIR` | Custom installation directory |
| `NEUMANN_VERSION` | Install a specific version (e.g., `v0.1.0`) |
| `NEUMANN_NO_MODIFY_PATH` | Set to `1` to skip PATH modification |
| `NEUMANN_SKIP_EXTRAS` | Set to `1` to skip completions and man page installation |

## Homebrew (macOS/Linux)

```bash
brew tap Shadylukin/tap
brew install neumann
```

## Cargo (crates.io)

If you have Rust installed:

```bash
cargo install neumann_shell
```

To install the gRPC server:

```bash
cargo install neumann_server
```

## Docker

### Interactive CLI

```bash
docker run -it shadylukinack/neumann:latest
```

### Server Mode

```bash
docker run -d -p 9200:9200 -v neumann-data:/var/lib/neumann shadylukinack/neumann:server
```

### Docker Compose

```bash
# Clone the repository
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann

# Start the server
docker compose up -d neumann-server

# Run the CLI
docker compose run --rm neumann-cli
```

## From Source

### Requirements

- Rust 1.75 or later
- Cargo (included with Rust)
- Git
- protobuf compiler (for gRPC)

### Build Steps

```bash
# Clone the repository
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann

# Build in release mode
cargo build --release --package neumann_shell

# Install locally
cargo install --path neumann_shell
```

### Run Tests

```bash
cargo test
```

## SDK Installation

### Python

```bash
pip install neumann-db
```

For embedded mode (in-process database):

```bash
pip install neumann-db[native]
```

### TypeScript / JavaScript

```bash
npm install neumann-db
```

Or with yarn:

```bash
yarn add neumann-db
```

## Verify Installation

```bash
neumann --version
```

## Platform Support

| Platform | Binary | Homebrew | Docker | Source |
| --- | --- | --- | --- | --- |
| Linux x86_64 | Yes | Yes | Yes | Yes |
| macOS x86_64 | Yes | Yes | Yes | Yes |
| macOS ARM64 (Apple Silicon) | Yes | Yes | Yes | Yes |
| Windows x86_64 | No | No | Yes | Experimental |

## Troubleshooting

### "command not found: neumann"

The binary may not be in your PATH. Try:

```bash
# Check where it was installed
which neumann || ls ~/.local/bin/neumann

# Add to PATH if needed
export PATH="$HOME/.local/bin:$PATH"
```

### Build fails with protobuf errors

Install the protobuf compiler:

```bash
# macOS
brew install protobuf

# Ubuntu/Debian
sudo apt-get install protobuf-compiler

# Fedora
sudo dnf install protobuf-compiler
```

### Permission denied during install

The installer tries `/usr/local/bin` first (requires sudo) then falls back
to `~/.local/bin`. You can specify a custom directory:

```bash
NEUMANN_INSTALL_DIR=~/bin \
  curl -sSfL https://raw.githubusercontent.com/Shadylukin/Neumann/main/install.sh | bash
```

### Python SDK native module errors

If you get errors about the native module, ensure you have a Rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install neumann-db[native]
```

## Updating

### Quick Install

Re-run the install script to get the latest version:

```bash
curl -sSfL https://raw.githubusercontent.com/Shadylukin/Neumann/main/install.sh | bash
```

### Homebrew

```bash
brew upgrade neumann
```

### Cargo

```bash
cargo install neumann_shell --force
```

### Python SDK

```bash
pip install --upgrade neumann-db
```

### TypeScript SDK

```bash
npm update neumann-db
```

## Uninstalling

### Quick Install / Cargo

```bash
rm $(which neumann)
```

### Homebrew

```bash
brew uninstall neumann
```

### Docker

```bash
docker rmi shadylukinack/neumann:latest shadylukinack/neumann:server
docker volume rm neumann-data
```

### Python SDK

```bash
pip uninstall neumann-db
```

### TypeScript SDK

```bash
npm uninstall neumann-db
```

## Next Steps

- [Quick Start](quick-start.md) - Run your first queries
- [Building from Source](building-from-source.md) - Development setup
