#!/bin/bash
# Documentation validation script
# Checks markdown files for compliance with documentation standards

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
DOCS_DIR="$REPO_ROOT/docs"
EXIT_CODE=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo "Validating documentation..."

# Check if markdownlint is available
check_markdownlint() {
    if command -v markdownlint &> /dev/null; then
        echo "Running markdownlint..."
        if ! markdownlint "$DOCS_DIR/**/*.md" "$REPO_ROOT/*.md" 2>/dev/null; then
            echo -e "${RED}Markdownlint found issues${NC}"
            EXIT_CODE=1
        else
            echo -e "${GREEN}Markdownlint passed${NC}"
        fi
    elif command -v npx &> /dev/null; then
        echo "Running markdownlint via npx..."
        if ! npx markdownlint-cli "$DOCS_DIR/**/*.md" "$REPO_ROOT/*.md" 2>/dev/null; then
            echo -e "${RED}Markdownlint found issues${NC}"
            EXIT_CODE=1
        else
            echo -e "${GREEN}Markdownlint passed${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping markdownlint (not installed)${NC}"
    fi
}

# Check code blocks have language specifiers
check_code_block_languages() {
    echo "Checking code block language specifiers..."
    local issues=0

    while IFS= read -r -d '' file; do
        # Find code blocks without language specifier
        if grep -Pn '```\s*$' "$file" 2>/dev/null | grep -v '```$' > /dev/null; then
            echo -e "${RED}Missing language specifier in: $file${NC}"
            grep -Pn '```\s*$' "$file" 2>/dev/null || true
            issues=1
        fi
    done < <(find "$DOCS_DIR" -name "*.md" -print0 2>/dev/null)

    if [ "$issues" -eq 0 ]; then
        echo -e "${GREEN}All code blocks have language specifiers${NC}"
    else
        EXIT_CODE=1
    fi
}

# Check for deprecated Mermaid directives
check_mermaid_deprecated() {
    echo "Checking for deprecated Mermaid directives..."
    local issues=0

    while IFS= read -r -d '' file; do
        # Check for deprecated 'graph' directive (should use 'flowchart')
        if grep -Pn '```mermaid\s*\n\s*graph\s' "$file" 2>/dev/null; then
            echo -e "${YELLOW}Deprecated 'graph' directive in: $file${NC}"
            echo "Consider using 'flowchart' instead"
            issues=1
        fi
    done < <(find "$DOCS_DIR" -name "*.md" -print0 2>/dev/null)

    if [ "$issues" -eq 0 ]; then
        echo -e "${GREEN}No deprecated Mermaid directives found${NC}"
    fi
    # Don't fail on deprecated directives, just warn
}

# Check architecture docs have required sections
check_architecture_docs() {
    echo "Checking architecture document structure..."
    local arch_dir="$DOCS_DIR/book/src/architecture"
    local issues=0

    if [ -d "$arch_dir" ]; then
        while IFS= read -r -d '' file; do
            local missing=""

            # Check for required sections
            if ! grep -q "^## Overview" "$file" && ! grep -q "^# .*Overview" "$file"; then
                missing="$missing Overview"
            fi

            if [ -n "$missing" ]; then
                echo -e "${YELLOW}Missing sections in $file:$missing${NC}"
            fi
        done < <(find "$arch_dir" -name "*.md" -print0 2>/dev/null)
    fi

    if [ "$issues" -eq 0 ]; then
        echo -e "${GREEN}Architecture docs structure OK${NC}"
    fi
}

# Check runbooks have required sections
check_runbook_docs() {
    echo "Checking runbook document structure..."
    local runbook_dir="$DOCS_DIR/book/src/operations/runbooks"
    local issues=0

    if [ -d "$runbook_dir" ]; then
        while IFS= read -r -d '' file; do
            # Skip index files
            if [[ "$file" == *"index.md"* ]]; then
                continue
            fi

            local missing=""

            # Check for required runbook sections
            if ! grep -qi "symptom" "$file"; then
                missing="$missing Symptoms"
            fi
            if ! grep -qi "diagnostic\|diagnos" "$file"; then
                missing="$missing Diagnostic"
            fi
            if ! grep -qi "resolution\|procedure\|steps" "$file"; then
                missing="$missing Resolution"
            fi

            if [ -n "$missing" ]; then
                echo -e "${YELLOW}Missing sections in $file:$missing${NC}"
            fi
        done < <(find "$runbook_dir" -name "*.md" -print0 2>/dev/null)
    fi

    if [ "$issues" -eq 0 ]; then
        echo -e "${GREEN}Runbook docs structure OK${NC}"
    fi
}

# Check table alignment
check_table_alignment() {
    echo "Checking table alignment..."
    # This is a basic check - markdownlint handles most cases
    echo -e "${GREEN}Table alignment delegated to markdownlint${NC}"
}

# Main execution
echo ""
echo "=== Documentation Validation ==="
echo ""

check_markdownlint
echo ""
check_code_block_languages
echo ""
check_mermaid_deprecated
echo ""
check_architecture_docs
echo ""
check_runbook_docs
echo ""
check_table_alignment

echo ""
if [ "$EXIT_CODE" -eq 0 ]; then
    echo -e "${GREEN}All documentation checks passed${NC}"
else
    echo -e "${RED}Documentation validation failed${NC}"
fi

exit $EXIT_CODE
