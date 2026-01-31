#!/bin/bash
# Neumann CLI Installer
# Usage: curl -sSfL https://raw.githubusercontent.com/Shadylukin/Neumann/main/install.sh | bash
#
# Environment variables:
#   NEUMANN_INSTALL_DIR - Installation directory (default: /usr/local/bin or ~/.local/bin)
#   NEUMANN_VERSION     - Specific version to install (default: latest)
#   NEUMANN_NO_MODIFY_PATH - Set to 1 to skip PATH modification
#   NEUMANN_SKIP_EXTRAS - Set to 1 to skip completions and man page installation

set -euo pipefail

REPO="Shadylukin/Neumann"
BINARY_NAME="neumann"
GITHUB_API="https://api.github.com"
GITHUB_RELEASES="https://github.com/${REPO}/releases"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}==>${NC} $1"
}

success() {
    echo -e "${GREEN}==>${NC} $1"
}

warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

error() {
    echo -e "${RED}Error:${NC} $1" >&2
    exit 1
}

detect_platform() {
    local os arch

    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)

    case "$os" in
        linux)
            os="linux"
            ;;
        darwin)
            os="darwin"
            ;;
        mingw*|msys*|cygwin*)
            error "Windows detected. Please download from GitHub releases or use: winget install neumann"
            ;;
        *)
            error "Unsupported operating system: $os"
            ;;
    esac

    case "$arch" in
        x86_64|amd64)
            arch="x86_64"
            ;;
        arm64|aarch64)
            arch="aarch64"
            ;;
        *)
            error "Unsupported architecture: $arch"
            ;;
    esac

    echo "${os}-${arch}"
}

get_latest_version() {
    local version
    version=$(curl -sSfL "${GITHUB_API}/repos/${REPO}/releases/latest" 2>/dev/null | \
        grep '"tag_name"' | \
        sed -E 's/.*"tag_name": *"([^"]+)".*/\1/')

    if [ -z "$version" ]; then
        return 1
    fi
    echo "$version"
}

download_binary() {
    local version="$1"
    local platform="$2"
    local tmpdir="$3"

    local archive_name="neumann-${version}-${platform}.tar.gz"
    local url="${GITHUB_RELEASES}/download/${version}/${archive_name}"

    info "Downloading ${archive_name}..."

    if curl -sSfL -o "${tmpdir}/${archive_name}" "$url" 2>/dev/null; then
        info "Extracting archive..."
        tar -xzf "${tmpdir}/${archive_name}" -C "$tmpdir"

        if [ -f "${tmpdir}/${BINARY_NAME}" ]; then
            return 0
        fi
    fi

    return 1
}

build_from_source() {
    local tmpdir="$1"

    info "Pre-built binary not available for this platform"
    info "Building from source..."

    # Check for Rust
    if ! command -v cargo &>/dev/null; then
        error "Cargo not found. Please install Rust from https://rustup.rs"
    fi

    # Check Rust version
    local rust_version
    rust_version=$(rustc --version | sed -E 's/rustc ([0-9]+\.[0-9]+).*/\1/')
    local required="1.75"

    if [ "$(printf '%s\n' "$required" "$rust_version" | sort -V | head -n1)" != "$required" ]; then
        error "Rust $required or later is required (found $rust_version)"
    fi

    # Clone and build
    info "Cloning repository..."
    git clone --depth 1 "https://github.com/${REPO}.git" "${tmpdir}/neumann" || \
        error "Failed to clone repository"

    cd "${tmpdir}/neumann"

    info "Building release binary (this may take a few minutes)..."
    cargo build --release --package neumann_shell || \
        error "Build failed"

    cp "target/release/${BINARY_NAME}" "${tmpdir}/${BINARY_NAME}"

    # Copy completions if available
    local completions_dir
    completions_dir=$(find target/release/build -path "*neumann_shell*/out/completions" -type d 2>/dev/null | head -1)
    if [ -n "$completions_dir" ] && [ -d "$completions_dir" ]; then
        mkdir -p "${tmpdir}/completions"
        cp "$completions_dir"/* "${tmpdir}/completions/" 2>/dev/null || true
    fi

    # Copy man page if available
    local man_dir
    man_dir=$(find target/release/build -path "*neumann_shell*/out/man" -type d 2>/dev/null | head -1)
    if [ -n "$man_dir" ] && [ -d "$man_dir" ]; then
        mkdir -p "${tmpdir}/man"
        cp "$man_dir"/* "${tmpdir}/man/" 2>/dev/null || true
    fi
}

get_install_dir() {
    # Use environment variable if set
    if [ -n "${NEUMANN_INSTALL_DIR:-}" ]; then
        echo "$NEUMANN_INSTALL_DIR"
        return
    fi

    # Try /usr/local/bin first (requires sudo)
    if [ -w "/usr/local/bin" ]; then
        echo "/usr/local/bin"
        return
    fi

    # Fall back to ~/.local/bin
    local local_bin="${HOME}/.local/bin"
    mkdir -p "$local_bin"
    echo "$local_bin"
}

install_binary() {
    local tmpdir="$1"
    local install_dir="$2"
    local binary_path="${tmpdir}/${BINARY_NAME}"

    if [ ! -f "$binary_path" ]; then
        error "Binary not found at ${binary_path}"
    fi

    # Check if we need sudo
    if [ ! -w "$install_dir" ]; then
        info "Installing to ${install_dir} (requires sudo)..."
        sudo install -m 755 "$binary_path" "${install_dir}/${BINARY_NAME}"
    else
        info "Installing to ${install_dir}..."
        install -m 755 "$binary_path" "${install_dir}/${BINARY_NAME}"
    fi
}

install_extras() {
    local tmpdir="$1"

    # Skip if disabled
    if [ "${NEUMANN_SKIP_EXTRAS:-0}" = "1" ]; then
        return
    fi

    # Install shell completions
    if [ -d "${tmpdir}/completions" ]; then
        # Bash completions
        local bash_completions="/usr/local/share/bash-completion/completions"
        if [ -d "$bash_completions" ] || sudo mkdir -p "$bash_completions" 2>/dev/null; then
            if [ -f "${tmpdir}/completions/neumann.bash" ]; then
                sudo cp "${tmpdir}/completions/neumann.bash" "$bash_completions/neumann" 2>/dev/null || true
            fi
        fi

        # Zsh completions
        local zsh_completions="/usr/local/share/zsh/site-functions"
        if [ -d "$zsh_completions" ] || sudo mkdir -p "$zsh_completions" 2>/dev/null; then
            if [ -f "${tmpdir}/completions/_neumann" ]; then
                sudo cp "${tmpdir}/completions/_neumann" "$zsh_completions/_neumann" 2>/dev/null || true
            fi
        fi

        # Fish completions
        local fish_completions="/usr/local/share/fish/vendor_completions.d"
        if [ -d "$fish_completions" ] || sudo mkdir -p "$fish_completions" 2>/dev/null; then
            if [ -f "${tmpdir}/completions/neumann.fish" ]; then
                sudo cp "${tmpdir}/completions/neumann.fish" "$fish_completions/neumann.fish" 2>/dev/null || true
            fi
        fi

        info "Shell completions installed (restart shell to activate)"
    fi

    # Install man page
    if [ -f "${tmpdir}/man/neumann.1" ]; then
        local man_dir="/usr/local/share/man/man1"
        if [ -d "$man_dir" ] || sudo mkdir -p "$man_dir" 2>/dev/null; then
            sudo cp "${tmpdir}/man/neumann.1" "$man_dir/" 2>/dev/null && \
                info "Man page installed (try: man neumann)" || true
        fi
    fi
}

add_to_path() {
    local install_dir="$1"

    # Skip if disabled
    if [ "${NEUMANN_NO_MODIFY_PATH:-0}" = "1" ]; then
        return
    fi

    # Skip if already in PATH
    if echo "$PATH" | tr ':' '\n' | grep -qx "$install_dir"; then
        return
    fi

    # Skip if it's a standard directory
    if [ "$install_dir" = "/usr/local/bin" ] || [ "$install_dir" = "/usr/bin" ]; then
        return
    fi

    local shell_config
    if [ -n "${ZSH_VERSION:-}" ] || [ -f "${HOME}/.zshrc" ]; then
        shell_config="${HOME}/.zshrc"
    elif [ -n "${BASH_VERSION:-}" ] || [ -f "${HOME}/.bashrc" ]; then
        shell_config="${HOME}/.bashrc"
    else
        warn "Could not determine shell config file"
        warn "Please add ${install_dir} to your PATH manually"
        return
    fi

    local path_line="export PATH=\"${install_dir}:\$PATH\""

    if ! grep -qF "$install_dir" "$shell_config" 2>/dev/null; then
        echo "" >> "$shell_config"
        echo "# Added by Neumann installer" >> "$shell_config"
        echo "$path_line" >> "$shell_config"
        info "Added ${install_dir} to PATH in ${shell_config}"
        warn "Please restart your shell or run: source ${shell_config}"
    fi
}

check_existing_installation() {
    if command -v "$BINARY_NAME" &>/dev/null; then
        local current_version
        current_version=$("$BINARY_NAME" --version 2>/dev/null | head -n1 || echo "unknown")
        info "Found existing installation: ${current_version}"
    fi
}

main() {
    echo ""
    echo "  Neumann CLI Installer"
    echo "  ====================="
    echo ""

    # Detect platform
    local platform
    platform=$(detect_platform)
    info "Detected platform: ${platform}"

    # Map platform to target triple
    local target
    case "$platform" in
        linux-x86_64)
            target="x86_64-unknown-linux-gnu"
            ;;
        darwin-x86_64)
            target="x86_64-apple-darwin"
            ;;
        darwin-aarch64)
            target="aarch64-apple-darwin"
            ;;
        *)
            error "Unknown platform: ${platform}"
            ;;
    esac

    # Check for existing installation
    check_existing_installation

    # Get version to install
    local version="${NEUMANN_VERSION:-}"
    if [ -z "$version" ]; then
        info "Fetching latest version..."
        version=$(get_latest_version) || true
    fi

    # Create temporary directory
    local tmpdir
    tmpdir=$(mktemp -d)
    trap 'rm -rf "$tmpdir"' EXIT

    # Try to download pre-built binary
    local binary_available=false
    if [ -n "$version" ]; then
        info "Installing version: ${version}"
        if download_binary "$version" "$target" "$tmpdir"; then
            binary_available=true
        fi
    fi

    # Fall back to building from source
    if [ "$binary_available" = false ]; then
        build_from_source "$tmpdir"
    fi

    # Install
    local install_dir
    install_dir=$(get_install_dir)
    install_binary "$tmpdir" "$install_dir"

    # Install extras (completions, man page)
    install_extras "$tmpdir"

    # Update PATH if needed
    add_to_path "$install_dir"

    # Verify installation
    echo ""
    success "Neumann CLI installed successfully!"
    echo ""

    if [ -x "${install_dir}/${BINARY_NAME}" ]; then
        info "Installed to: ${install_dir}/${BINARY_NAME}"
        info "Version: $("${install_dir}/${BINARY_NAME}" --version 2>/dev/null || echo 'unknown')"
    fi

    echo ""
    echo "  Get started:"
    echo "    neumann              # Start interactive shell"
    echo "    neumann --help       # Show help"
    echo "    neumann -c 'query'   # Execute single query"
    echo ""
    echo "  Documentation:"
    echo "    https://github.com/${REPO}"
    echo ""
}

main "$@"
