#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Neumann Skills Pack -- Uninstaller
#
# Finds all Neumann skill installations via their manifest files and removes
# them cleanly. Idempotent -- safe to run multiple times.
# ---------------------------------------------------------------------------

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

# ---- Find project root ----------------------------------------------------

PROJECT_ROOT=""

find_project_root() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/.git" || -f "$dir/.git" ]]; then
            PROJECT_ROOT="$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

find_project_root || true

# ---- Compute allowed roots ------------------------------------------------
# These are the only directories we will delete from. Paths from the manifest
# are verified against these roots before removal.

declare -a ALLOWED_ROOTS=()

# Home-directory roots (always allowed)
ALLOWED_ROOTS+=("${HOME}/.claude/skills")
ALLOWED_ROOTS+=("${HOME}/.codex/skills")
ALLOWED_ROOTS+=("${HOME}/.gemini/skills")

# Project-level roots (if we found a project)
if [[ -n "$PROJECT_ROOT" ]]; then
    ALLOWED_ROOTS+=("${PROJECT_ROOT}/.claude/skills")
    ALLOWED_ROOTS+=("${PROJECT_ROOT}/.gemini/skills")
fi

# ---- Collect manifest locations -------------------------------------------

declare -a MANIFEST_PATHS=()

# Check all known locations
check_manifest() {
    local dir="$1"
    if [[ -f "$dir/$MANIFEST_NAME" ]]; then
        MANIFEST_PATHS+=("$dir/$MANIFEST_NAME")
    fi
}

# Home-directory locations
check_manifest "${HOME}/.claude/skills"
check_manifest "${HOME}/.codex/skills"
check_manifest "${HOME}/.gemini/skills"

# Project-level locations
if [[ -n "$PROJECT_ROOT" ]]; then
    check_manifest "${PROJECT_ROOT}/.claude/skills"
    check_manifest "${PROJECT_ROOT}/.gemini/skills"
fi

# ---- Process manifests ----------------------------------------------------

if [[ ${#MANIFEST_PATHS[@]} -eq 0 ]]; then
    info "No Neumann skills installation found."
    exit 0
fi

TOTAL_SKILLS_REMOVED=0
TOTAL_LOCATIONS=0

for manifest in "${MANIFEST_PATHS[@]}"; do
    info "Processing manifest: $manifest"

    # Parse skills array from manifest using basic tools (no jq dependency)
    # Extract paths from JSON entries like {"name":"neumann-query","path":"/abs/path"}
    skill_paths=()
    while IFS= read -r line; do
        # Extract path values from JSON
        path_value="$(echo "$line" | sed -n 's/.*"path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
        if [[ -n "$path_value" ]]; then
            skill_paths+=("$path_value")
        fi
    done < "$manifest"

    skills_removed=0

    for skill_path in "${skill_paths[@]}"; do
        # Canonicalize skill_path (resolve symlinks + .. segments).
        # If the path does not exist, there is nothing to delete (idempotent).
        if [[ -d "$skill_path" ]]; then
            resolved="$(cd "$skill_path" 2>/dev/null && pwd -P)" || true
            if [[ -z "$resolved" ]]; then
                warn "  Skipping $skill_path -- directory not accessible"
                continue
            fi
            skill_path="$resolved"
        else
            continue
        fi

        # Verify leaf directory name starts with neumann-
        leaf="$(basename "$skill_path")"
        if [[ "$leaf" != neumann-* ]]; then
            warn "  Skipping $skill_path -- not a neumann-* directory"
            continue
        fi

        # Check against allowed roots (strict child, never the root itself)
        path_allowed=false
        for root in "${ALLOWED_ROOTS[@]}"; do
            root="${root%/}"
            if [[ -d "$root" ]]; then
                resolved_root="$(cd "$root" 2>/dev/null && pwd -P)" || true
                [[ -n "$resolved_root" ]] && root="$resolved_root"
            fi
            if [[ "$skill_path" == "$root"/* ]]; then
                path_allowed=true
                break
            fi
        done

        if [[ "$path_allowed" == false ]]; then
            warn "  Skipping $skill_path -- not under any allowed root directory"
            continue
        fi

        rm -rf "$skill_path"
        success "  Removed $leaf"
        ((skills_removed++)) || true
    done

    # Remove the manifest file itself
    if [[ -f "$manifest" ]]; then
        rm -f "$manifest"
        success "  Removed manifest: $manifest"
    fi

    TOTAL_SKILLS_REMOVED=$((TOTAL_SKILLS_REMOVED + skills_removed))
    ((TOTAL_LOCATIONS++)) || true
done

# ---- Summary --------------------------------------------------------------

echo ""
success "Removed $TOTAL_SKILLS_REMOVED skills from $TOTAL_LOCATIONS tool location(s)."
