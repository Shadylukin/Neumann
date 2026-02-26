#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Neumann Skills Pack -- Validator
#
# Checks the structure and frontmatter of all skills/neumann-*/SKILL.md files.
# Exits 0 if all pass, 1 if any fail.
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_BODY_LINES=500
WARN_BODY_LINES=300
MAX_DESC_CHARS=1024

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
success() { printf "${GREEN}[PASS]${NC} %s\n" "$*"; }
warn()    { printf "${YELLOW}[warn]${NC} %s\n" "$*" >&2; }
fail()    { printf "${RED}[FAIL]${NC} %s\n" "$*" >&2; }

# ---- Find all SKILL.md files ----------------------------------------------

shopt -s nullglob
SKILL_FILES=("$SCRIPT_DIR"/neumann-*/SKILL.md)
shopt -u nullglob

if [[ ${#SKILL_FILES[@]} -eq 0 ]]; then
    fail "No skills/neumann-*/SKILL.md files found in $SCRIPT_DIR"
    exit 1
fi

info "Found ${#SKILL_FILES[@]} skill(s) to validate"
echo ""

# ---- Validate each skill ---------------------------------------------------

FAILURES=0

for skill_file in "${SKILL_FILES[@]}"; do
    skill_dir="$(dirname "$skill_file")"
    dir_name="$(basename "$skill_dir")"
    errors=()
    warnings=()

    # -- 1. Parse YAML frontmatter -------------------------------------------
    # Frontmatter is between the first and second --- lines.

    in_frontmatter=false
    frontmatter_started=false
    frontmatter_ended=false
    frontmatter_lines=()
    body_lines=0
    line_num=0

    while IFS= read -r line || [[ -n "$line" ]]; do
        ((line_num++)) || true

        if [[ "$frontmatter_ended" == true ]]; then
            ((body_lines++)) || true
            continue
        fi

        if [[ "$line" == "---" ]]; then
            if [[ "$frontmatter_started" == false ]]; then
                frontmatter_started=true
                in_frontmatter=true
                continue
            else
                in_frontmatter=false
                frontmatter_ended=true
                continue
            fi
        fi

        if [[ "$in_frontmatter" == true ]]; then
            frontmatter_lines+=("$line")
        fi
    done < "$skill_file"

    if [[ "$frontmatter_started" == false ]]; then
        errors+=("Missing YAML frontmatter (no opening ---)")
    elif [[ "$frontmatter_ended" == false ]]; then
        errors+=("Incomplete YAML frontmatter (no closing ---)")
    fi

    # -- Parse frontmatter fields -------------------------------------------
    fm_name=""
    fm_description=""
    fm_keys=()

    for fm_line in "${frontmatter_lines[@]}"; do
        # Skip empty lines and comments
        [[ -z "$fm_line" || "$fm_line" == \#* ]] && continue

        # Extract key: value pairs
        if [[ "$fm_line" =~ ^([a-zA-Z_][a-zA-Z0-9_]*):[[:space:]]*(.*) ]]; then
            key="${BASH_REMATCH[1]}"
            value="${BASH_REMATCH[2]}"
            # Strip surrounding quotes from value
            value="$(echo "$value" | sed -E "s/^['\"]|['\"]$//g")"
            fm_keys+=("$key")

            case "$key" in
                name)        fm_name="$value" ;;
                description) fm_description="$value" ;;
            esac
        fi
    done

    # -- 1a. Verify only name and description present -----------------------
    for key in "${fm_keys[@]}"; do
        if [[ "$key" != "name" && "$key" != "description" ]]; then
            errors+=("Unexpected frontmatter key: '$key' (only 'name' and 'description' allowed)")
        fi
    done

    if [[ -z "$fm_name" ]]; then
        errors+=("Missing 'name' in frontmatter")
    fi
    if [[ -z "$fm_description" ]]; then
        errors+=("Missing 'description' in frontmatter")
    fi

    # -- 2. Verify name matches directory name ------------------------------
    if [[ -n "$fm_name" && "$fm_name" != "$dir_name" ]]; then
        errors+=("Frontmatter name '$fm_name' does not match directory name '$dir_name'")
    fi

    # -- 3. Verify description length ---------------------------------------
    if [[ -n "$fm_description" ]]; then
        desc_len=${#fm_description}
        if [[ "$desc_len" -gt "$MAX_DESC_CHARS" ]]; then
            errors+=("Description is $desc_len chars (max $MAX_DESC_CHARS)")
        fi
    fi

    # -- 4. Check body line count -------------------------------------------
    if [[ "$body_lines" -ge "$MAX_BODY_LINES" ]]; then
        errors+=("Body is $body_lines lines (max $((MAX_BODY_LINES - 1)))")
    elif [[ "$body_lines" -gt "$WARN_BODY_LINES" ]]; then
        warnings+=("Body is $body_lines lines (consider keeping under $WARN_BODY_LINES)")
    fi

    # -- 5. Check agents/openai.yaml ----------------------------------------
    openai_yaml="$skill_dir/agents/openai.yaml"
    if [[ ! -f "$openai_yaml" ]]; then
        errors+=("Missing agents/openai.yaml")
    else
        # Verify required keys exist (may be nested under interface:)
        for required_key in display_name short_description default_prompt; do
            if ! grep -qE "^[[:space:]]*${required_key}:" "$openai_yaml" 2>/dev/null; then
                errors+=("agents/openai.yaml missing required key: $required_key")
            fi
        done
    fi

    # -- Report results -----------------------------------------------------
    for w in "${warnings[@]+"${warnings[@]}"}"; do
        warn "  $dir_name: $w"
    done

    if [[ ${#errors[@]} -eq 0 ]]; then
        success "$dir_name"
    else
        fail "$dir_name"
        for e in "${errors[@]}"; do
            printf "       %s\n" "$e" >&2
        done
        ((FAILURES++)) || true
    fi
done

echo ""

# ---- Optional: run OpenAI's quick_validate.py if available ----------------

if command -v quick_validate.py &>/dev/null; then
    info "Running OpenAI quick_validate.py as secondary check..."
    quick_validate.py "$SCRIPT_DIR" || warn "quick_validate.py reported issues"
    echo ""
fi

# ---- Final result ----------------------------------------------------------

if [[ "$FAILURES" -gt 0 ]]; then
    fail "$FAILURES of ${#SKILL_FILES[@]} skill(s) failed validation"
    exit 1
else
    success "All ${#SKILL_FILES[@]} skill(s) passed validation"
    exit 0
fi
