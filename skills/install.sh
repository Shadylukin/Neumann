#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Neumann Skills Pack -- Universal Installer
#
# Installs Neumann AI coding skills into Claude Code, Codex, and/or Gemini CLI.
#
# Environment variables:
#   NEUMANN_SKILLS_SCOPE  project|global   (default: project)
#   NEUMANN_SKILLS_TOOLS  claude,codex,gemini,all  (default: auto-detect)
#   NEUMANN_SKILLS_REF    git tag or SHA   (default: latest release)
#
# Flags:
#   --force   Overwrite unmanaged skill directories without prompting
#   --help    Print usage and exit
# ---------------------------------------------------------------------------

REPO="Shadylukin/Neumann"
MANIFEST_NAME=".neumann-skills.json"

# ---- Color helpers --------------------------------------------------------

if [[ -n "${NO_COLOR:-}" ]]; then
    RED="" GREEN="" YELLOW="" BLUE="" NC=""
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
fi

info()    { printf "${BLUE}[info]${NC} %s\n" "$*"; }
success() { printf "${GREEN}[ok]${NC}   %s\n" "$*"; }
warn()    { printf "${YELLOW}[warn]${NC} %s\n" "$*" >&2; }
error()   { printf "${RED}[error]${NC} %s\n" "$*" >&2; exit 1; }

# ---- Parse flags ----------------------------------------------------------

FORCE=false

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=true ;;
        --help|-h)
            cat <<'USAGE'
Usage: install.sh [--force] [--help]

Environment variables:
  NEUMANN_SKILLS_SCOPE   project | global      (default: project)
  NEUMANN_SKILLS_TOOLS   claude,codex,gemini,all (default: auto-detect)
  NEUMANN_SKILLS_REF     git tag or commit SHA  (default: latest release)

Flags:
  --force   Overwrite skill directories even if they lack a Neumann manifest
  --help    Show this message

Examples:
  # Install into current project for auto-detected tools
  bash install.sh

  # Install globally for all tools
  NEUMANN_SKILLS_SCOPE=global NEUMANN_SKILLS_TOOLS=all bash install.sh

  # Pin to a specific release
  NEUMANN_SKILLS_REF=v1.0.0 bash install.sh
USAGE
            exit 0
            ;;
        *) error "Unknown flag: $arg (try --help)" ;;
    esac
done

# ---- Temp directory with cleanup ------------------------------------------

TMPDIR_INSTALL=""
cleanup() {
    if [[ -n "$TMPDIR_INSTALL" && -d "$TMPDIR_INSTALL" ]]; then
        rm -rf "$TMPDIR_INSTALL"
    fi
}
trap cleanup EXIT

# ---- Locate skills source -------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_MODE=false
SKILLS_SRC=""

if [[ -d "$SCRIPT_DIR/neumann-query" && -d "$SCRIPT_DIR/neumann-schema" ]]; then
    LOCAL_MODE=true
    SKILLS_SRC="$SCRIPT_DIR"
    info "Local mode: using skills from $SKILLS_SRC"
else
    info "Remote mode: downloading skills from GitHub"

    REF="${NEUMANN_SKILLS_REF:-}"

    # Resolve latest release tag if no ref specified
    if [[ -z "$REF" ]]; then
        info "Resolving latest release tag..."
        REF=$(curl -sSfL "https://api.github.com/repos/${REPO}/releases/latest" \
            | grep '"tag_name"' \
            | sed -E 's/.*"tag_name":\s*"([^"]+)".*/\1/')
        if [[ -z "$REF" ]]; then
            error "Could not resolve latest release tag. Set NEUMANN_SKILLS_REF explicitly."
        fi
        info "Resolved ref: $REF"
    fi

    TMPDIR_INSTALL="$(mktemp -d)"
    TARBALL="$TMPDIR_INSTALL/neumann.tar.gz"

    info "Downloading tarball for ref $REF..."
    curl -sSfL -o "$TARBALL" "https://api.github.com/repos/${REPO}/tarball/${REF}"

    info "Extracting skills directories..."
    # GitHub tarballs have a top-level directory like Shadylukin-Neumann-<sha>/
    # We extract only the skills/neumann-*/ subtrees.
    TAR_PREFIX=$(tar -tzf "$TARBALL" | head -1 | cut -d/ -f1)
    tar -xzf "$TARBALL" -C "$TMPDIR_INSTALL" --include="${TAR_PREFIX}/skills/neumann-*/*"

    SKILLS_SRC="$TMPDIR_INSTALL/${TAR_PREFIX}/skills"

    # Verify at least one SKILL.md was extracted
    SKILL_COUNT=$(find "$SKILLS_SRC" -name "SKILL.md" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$SKILL_COUNT" -eq 0 ]]; then
        error "No SKILL.md files found in extracted tarball. Archive may be malformed."
    fi
    info "Found $SKILL_COUNT skills in archive"
fi

# Read version from VERSION file if available
VERSION="unknown"
if [[ -f "$SKILLS_SRC/VERSION" ]]; then
    VERSION="$(cat "$SKILLS_SRC/VERSION" | tr -d '[:space:]')"
fi

# ---- Determine scope ------------------------------------------------------

SCOPE="${NEUMANN_SKILLS_SCOPE:-project}"

case "$SCOPE" in
    project|global) ;;
    *) error "Invalid NEUMANN_SKILLS_SCOPE: '$SCOPE' (must be 'project' or 'global')" ;;
esac

# ---- Find project root (for project scope) --------------------------------

PROJECT_ROOT=""

find_project_root() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/.git" ]]; then
            PROJECT_ROOT="$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

if [[ "$SCOPE" == "project" ]]; then
    if ! find_project_root; then
        error "No .git/ directory found above $PWD. Cannot determine project root.
       Use NEUMANN_SKILLS_SCOPE=global to install to your home directory instead."
    fi
    info "Project root: $PROJECT_ROOT"
fi

# ---- Detect target tools --------------------------------------------------

TOOLS_INPUT="${NEUMANN_SKILLS_TOOLS:-}"
INSTALL_CLAUDE=false
INSTALL_CODEX=false
INSTALL_GEMINI=false

if [[ -z "$TOOLS_INPUT" ]]; then
    # Auto-detect
    info "Auto-detecting installed tools..."
    detected=0

    if [[ -d "${PROJECT_ROOT:-.}/.claude" ]] || command -v claude &>/dev/null; then
        INSTALL_CLAUDE=true
        ((detected++)) || true
        info "  Detected: Claude Code"
    fi
    if command -v codex &>/dev/null; then
        INSTALL_CODEX=true
        ((detected++)) || true
        info "  Detected: Codex"
    fi
    if [[ -d "${PROJECT_ROOT:-.}/.gemini" ]] || command -v gemini &>/dev/null; then
        INSTALL_GEMINI=true
        ((detected++)) || true
        info "  Detected: Gemini CLI"
    fi

    if [[ "$detected" -eq 0 ]]; then
        info "  No tools detected -- installing for all three"
        INSTALL_CLAUDE=true
        INSTALL_CODEX=true
        INSTALL_GEMINI=true
    fi
else
    # Explicit tool list
    IFS=',' read -ra TOOL_LIST <<< "$TOOLS_INPUT"
    for tool in "${TOOL_LIST[@]}"; do
        tool="$(echo "$tool" | tr -d '[:space:]')"
        case "$tool" in
            claude) INSTALL_CLAUDE=true ;;
            codex)  INSTALL_CODEX=true ;;
            gemini) INSTALL_GEMINI=true ;;
            all)    INSTALL_CLAUDE=true; INSTALL_CODEX=true; INSTALL_GEMINI=true ;;
            *)      error "Unknown tool: '$tool' (expected: claude, codex, gemini, all)" ;;
        esac
    done
fi

# Codex only supports global scope
if [[ "$INSTALL_CODEX" == true && "$SCOPE" == "project" ]]; then
    # If user explicitly requested codex, let it install to global location
    if [[ -n "$TOOLS_INPUT" ]] && echo "$TOOLS_INPUT" | grep -qw "codex"; then
        warn "Codex does not support project-level skills. Installing to ~/.codex/skills/ instead."
    else
        warn "Codex does not support project-level skills. Skipping Codex."
        warn "  Use NEUMANN_SKILLS_SCOPE=global or NEUMANN_SKILLS_TOOLS=codex to force."
        INSTALL_CODEX=false
    fi
fi

# ---- Build list of target directories -------------------------------------

declare -a TARGET_DIRS=()
declare -a TARGET_TOOLS=()

if [[ "$INSTALL_CLAUDE" == true ]]; then
    if [[ "$SCOPE" == "project" ]]; then
        TARGET_DIRS+=("${PROJECT_ROOT}/.claude/skills")
    else
        TARGET_DIRS+=("${HOME}/.claude/skills")
    fi
    TARGET_TOOLS+=("claude")
fi

if [[ "$INSTALL_CODEX" == true ]]; then
    # Codex is always user-level
    TARGET_DIRS+=("${HOME}/.codex/skills")
    TARGET_TOOLS+=("codex")
fi

if [[ "$INSTALL_GEMINI" == true ]]; then
    if [[ "$SCOPE" == "project" ]]; then
        TARGET_DIRS+=("${PROJECT_ROOT}/.gemini/skills")
    else
        TARGET_DIRS+=("${HOME}/.gemini/skills")
    fi
    TARGET_TOOLS+=("gemini")
fi

if [[ ${#TARGET_DIRS[@]} -eq 0 ]]; then
    error "No target tools selected. Nothing to install."
fi

# ---- Enumerate source skills ----------------------------------------------

declare -a SKILL_NAMES=()
for skill_dir in "$SKILLS_SRC"/neumann-*/; do
    if [[ -d "$skill_dir" ]]; then
        name="$(basename "$skill_dir")"
        SKILL_NAMES+=("$name")
    fi
done

if [[ ${#SKILL_NAMES[@]} -eq 0 ]]; then
    error "No neumann-* skill directories found in $SKILLS_SRC"
fi

info "Skills to install: ${SKILL_NAMES[*]}"

# ---- Resolve ref SHA for manifest -----------------------------------------

REF_DISPLAY="${REF:-local}"
REF_SHA="unknown"

if [[ "$LOCAL_MODE" == true ]]; then
    if command -v git &>/dev/null && git -C "$SCRIPT_DIR" rev-parse HEAD &>/dev/null; then
        REF_SHA="$(git -C "$SCRIPT_DIR" rev-parse HEAD)"
        REF_DISPLAY="$(git -C "$SCRIPT_DIR" describe --tags --always 2>/dev/null || echo "HEAD")"
    fi
else
    # For remote mode, try to extract SHA from the tarball prefix
    if [[ -n "${TAR_PREFIX:-}" ]]; then
        # Format is typically Shadylukin-Neumann-<short-sha>
        REF_SHA="$(echo "$TAR_PREFIX" | sed -E 's/.*-([a-f0-9]+)$/\1/')"
    fi
fi

# ---- Install to each target -----------------------------------------------

INSTALLED_COUNT=0

for i in "${!TARGET_DIRS[@]}"; do
    target_dir="${TARGET_DIRS[$i]}"
    tool_name="${TARGET_TOOLS[$i]}"

    info "Installing to $target_dir (${tool_name})..."

    # Safety check: does the skills dir already exist?
    if [[ -d "$target_dir" ]]; then
        if [[ -f "$target_dir/$MANIFEST_NAME" ]]; then
            info "  Found existing Neumann manifest -- safe to overwrite"
        else
            # Check if any neumann-* dirs exist (unmanaged)
            has_unmanaged=false
            for skill_name in "${SKILL_NAMES[@]}"; do
                if [[ -d "$target_dir/$skill_name" ]]; then
                    has_unmanaged=true
                    break
                fi
            done

            if [[ "$has_unmanaged" == true ]]; then
                if [[ "$FORCE" == true ]]; then
                    warn "  Unmanaged skill directories found in $target_dir -- overwriting (--force)"
                else
                    warn "  Unmanaged skill directories found in $target_dir"
                    warn "  Skipping to avoid overwriting files not managed by this installer."
                    warn "  Use --force to overwrite, or remove the directory first."
                    continue
                fi
            fi
        fi
    fi

    # Create target directory
    mkdir -p "$target_dir"

    # Copy each skill
    declare -a skill_entries=()
    for skill_name in "${SKILL_NAMES[@]}"; do
        src="$SKILLS_SRC/$skill_name"
        dst="$target_dir/$skill_name"

        # Remove existing skill dir if present
        if [[ -d "$dst" ]]; then
            rm -rf "$dst"
        fi

        cp -R "$src" "$dst"
        success "  Installed $skill_name"

        # Build JSON entry for manifest (absolute path)
        abs_dst="$(cd "$dst" && pwd)"
        skill_entries+=("{\"name\":\"${skill_name}\",\"path\":\"${abs_dst}\"}")
    done

    # Build skills JSON array
    skills_json="["
    for j in "${!skill_entries[@]}"; do
        if [[ "$j" -gt 0 ]]; then
            skills_json+=","
        fi
        skills_json+="${skill_entries[$j]}"
    done
    skills_json+="]"

    # Resolve absolute root path
    abs_target="$(cd "$target_dir" && pwd)"

    # Write manifest
    installed_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    cat > "$target_dir/$MANIFEST_NAME" <<MANIFEST
{
  "version": "${VERSION}",
  "ref": "${REF_DISPLAY}",
  "ref_sha": "${REF_SHA}",
  "installed_at": "${installed_at}",
  "tool": "${tool_name}",
  "root": "${abs_target}",
  "skills": ${skills_json}
}
MANIFEST

    success "  Wrote manifest: $target_dir/$MANIFEST_NAME"
    ((INSTALLED_COUNT++)) || true
    unset skill_entries
done

# ---- Summary --------------------------------------------------------------

echo ""
if [[ "$INSTALLED_COUNT" -eq 0 ]]; then
    warn "No skills were installed."
    exit 1
fi

success "Installed ${#SKILL_NAMES[@]} skills to $INSTALLED_COUNT tool location(s)."
echo ""

# Tool-specific reload instructions
if [[ "$INSTALL_CLAUDE" == true ]]; then
    info "Claude Code: Skills are loaded automatically. Restart your session to pick up changes."
fi
if [[ "$INSTALL_CODEX" == true ]]; then
    info "Codex: Skills are loaded automatically from ~/.codex/skills/."
fi
if [[ "$INSTALL_GEMINI" == true ]]; then
    info "Gemini CLI: Skills are loaded automatically. Restart your session to pick up changes."
fi

echo ""
info "To uninstall, run: bash skills/uninstall.sh"
