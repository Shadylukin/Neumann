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

# Check reference/api docs have required sections
check_reference_docs() {
    echo "Checking reference API document structure..."
    local ref_dir="$DOCS_DIR/book/src/reference/api"
    local issues=0

    if [ -d "$ref_dir" ]; then
        while IFS= read -r -d '' file; do
            local missing=""

            if ! grep -qi "see also" "$file"; then
                missing="$missing SeeAlso"
            fi

            if [ -n "$missing" ]; then
                echo -e "${YELLOW}Missing sections in $file:$missing${NC}"
            fi
        done < <(find "$ref_dir" -name "*.md" -print0 2>/dev/null)
    fi

    if [ "$issues" -eq 0 ]; then
        echo -e "${GREEN}Reference API docs structure OK${NC}"
    fi
}

# Check tutorial docs have required sections
check_tutorial_docs() {
    echo "Checking tutorial document structure..."
    local tut_dir="$DOCS_DIR/book/src/tutorials"
    local issues=0

    if [ -d "$tut_dir" ]; then
        while IFS= read -r -d '' file; do
            local missing=""

            if ! grep -qi "prerequisite\|step 1\|## step" "$file"; then
                missing="$missing Steps"
            fi
            if ! grep -qi "verification\|verify\|you should" "$file"; then
                missing="$missing Verification"
            fi

            if [ -n "$missing" ]; then
                echo -e "${YELLOW}Missing sections in $file:$missing${NC}"
            fi
        done < <(find "$tut_dir" -name "*.md" -not -name "worked-examples.md" -print0 2>/dev/null)
    fi

    if [ "$issues" -eq 0 ]; then
        echo -e "${GREEN}Tutorial docs structure OK${NC}"
    fi
}

# Check how-to docs have required sections
check_howto_docs() {
    echo "Checking how-to document structure..."
    local howto_dir="$DOCS_DIR/book/src/how-to"
    local issues=0

    if [ -d "$howto_dir" ]; then
        while IFS= read -r -d '' file; do
            # Skip runbooks (checked separately) and index files
            if [[ "$file" == *"runbooks"* ]] || [[ "$file" == *"index.md"* ]]; then
                continue
            fi

            local missing=""

            # How-to guides should have code examples or steps
            if ! grep -q '```' "$file"; then
                missing="$missing CodeExamples"
            fi

            if [ -n "$missing" ]; then
                echo -e "${YELLOW}Missing sections in $file:$missing${NC}"
            fi
        done < <(find "$howto_dir" -name "*.md" -print0 2>/dev/null)
    fi

    if [ "$issues" -eq 0 ]; then
        echo -e "${GREEN}How-to docs structure OK${NC}"
    fi
}

# Check explanation docs have required sections
check_explanation_docs() {
    echo "Checking explanation document structure..."
    local exp_dir="$DOCS_DIR/book/src/explanation"
    local issues=0

    if [ -d "$exp_dir" ]; then
        while IFS= read -r -d '' file; do
            local missing=""

            if ! grep -q "^## " "$file"; then
                missing="$missing Sections"
            fi

            if [ -n "$missing" ]; then
                echo -e "${YELLOW}Missing sections in $file:$missing${NC}"
            fi
        done < <(find "$exp_dir" -name "*.md" -print0 2>/dev/null)
    fi

    if [ "$issues" -eq 0 ]; then
        echo -e "${GREEN}Explanation docs structure OK${NC}"
    fi
}

# Check runbooks have required sections
check_runbook_docs() {
    echo "Checking runbook document structure..."
    local runbook_dir="$DOCS_DIR/book/src/how-to/runbooks"
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
check_reference_docs
echo ""
check_tutorial_docs
echo ""
check_howto_docs
echo ""
check_explanation_docs
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
