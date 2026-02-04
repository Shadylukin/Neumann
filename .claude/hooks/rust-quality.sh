#!/bin/bash
# rust-quality.sh - Clippy Pedantic + Testing Reminder Hook for Claude Code
#
# This hook runs after Write/Edit operations on Rust files and provides:
# 1. Clippy pedantic lint results as additionalContext
# 2. Testing reminders with project-specific coverage requirements

set -e

# Read JSON input from stdin
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Exit early if no file path or not a Rust file
if [[ -z "$FILE_PATH" ]] || [[ ! "$FILE_PATH" =~ \.rs$ ]]; then
  exit 0
fi

# Find the crate directory (walk up to find Cargo.toml)
CRATE_DIR=$(dirname "$FILE_PATH")
while [[ "$CRATE_DIR" != "/" && ! -f "$CRATE_DIR/Cargo.toml" ]]; do
  CRATE_DIR=$(dirname "$CRATE_DIR")
done

# If no Cargo.toml found, try the project root
if [[ ! -f "$CRATE_DIR/Cargo.toml" && -n "$CLAUDE_PROJECT_DIR" ]]; then
  CRATE_DIR="$CLAUDE_PROJECT_DIR"
fi

# Exit if still no Cargo.toml
if [[ ! -f "$CRATE_DIR/Cargo.toml" ]]; then
  exit 0
fi

# Extract crate name from Cargo.toml (handle various formatting)
CRATE_NAME=$(grep -m1 'name\s*=' "$CRATE_DIR/Cargo.toml" | head -1 | sed 's/.*"\([^"]*\)".*/\1/' || echo "")

# Run clippy with pedantic lints, capturing both stdout and stderr
CLIPPY_OUTPUT=""
cd "$CRATE_DIR"
if [[ -n "$CRATE_NAME" ]]; then
  CLIPPY_OUTPUT=$(cargo clippy --package "$CRATE_NAME" --message-format short -- -D warnings -D clippy::pedantic 2>&1 | head -100) || true
else
  CLIPPY_OUTPUT=$(cargo clippy --message-format short -- -D warnings -D clippy::pedantic 2>&1 | head -100) || true
fi

# Build the context message
CONTEXT=""

# Check for clippy issues (errors or warnings)
if [[ -n "$CLIPPY_OUTPUT" ]] && echo "$CLIPPY_OUTPUT" | grep -qE "(error\[|warning\[|error:|warning:)"; then
  # Filter to show only error/warning lines and relevant context
  FILTERED_OUTPUT=$(echo "$CLIPPY_OUTPUT" | grep -E "(error|warning|-->|help:|note:)" | head -50)

  CONTEXT="## Clippy Pedantic Results

The following issues were found after editing \`$FILE_PATH\`:

\`\`\`
$FILTERED_OUTPUT
\`\`\`

Please fix these clippy warnings/errors before continuing.

"
fi

# Get relative path for cleaner display
REL_PATH="${FILE_PATH#$CLAUDE_PROJECT_DIR/}"
if [[ "$REL_PATH" == "$FILE_PATH" ]]; then
  REL_PATH=$(basename "$FILE_PATH")
fi

# Add testing reminder
CONTEXT="${CONTEXT}## Testing Reminder

You modified \`$REL_PATH\`. Remember the project's quality standards:

**Coverage Requirements:**
- Default: 95% minimum line coverage
- neumann_shell: 88%
- tensor_blob: 91%
- query_router: 92%

**Action Items:**
- Add or update unit tests for new/changed functionality
- Test edge cases and error conditions
- Run \`cargo test --package $CRATE_NAME\` to verify tests pass
- Check coverage with \`cargo llvm-cov --package $CRATE_NAME --summary-only\`

**Clean Code Principles:**
- Keep functions small and focused (single responsibility)
- Use descriptive names that reveal intent
- Handle errors explicitly with Result and ? propagation
- Prefer iterators over loops
- No commented-out code - delete it"

# Output JSON with additionalContext
jq -n --arg ctx "$CONTEXT" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: $ctx
  }
}'

exit 0
