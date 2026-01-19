#!/bin/bash
# Clean up old documentation files after migration to mdBook
#
# This script removes old docs/*.md files that have been migrated to
# docs/book/src/. A backup is created first.
#
# Usage: ./scripts/cleanup-old-docs.sh [--dry-run]
#
# Files preserved:
#   - docs/architecture.md (canonical system design reference)
#   - docs/book/ (mdBook source)
#
# Files removed (after backup):
#   - All other docs/*.md files

set -e

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "DRY RUN MODE - No files will be deleted"
    echo ""
fi

DOCS_DIR="docs"
BACKUP_FILE="docs-backup-$(date +%Y%m%d-%H%M%S).tar.gz"

# Files to remove (migrated to docs/book/src/)
FILES_TO_REMOVE=(
    "benchmarks.md"
    "integration-tests.md"
    "stress-tests.md"
    "tensor-store.md"
    "relational-engine.md"
    "graph-engine.md"
    "vector-engine.md"
    "tensor-compress.md"
    "tensor-vault.md"
    "tensor-cache.md"
    "tensor-blob.md"
    "tensor-chain.md"
    "tensor-checkpoint.md"
    "tensor-unified.md"
    "neumann-parser.md"
    "query-router.md"
    "neumann-shell.md"
    "getting-started.md"
    "installation.md"
    "fuzz-testing.md"
    "DOCUMENTATION-ROADMAP.md"
)

# Files to keep
FILES_TO_KEEP=(
    "architecture.md"
)

echo "Documentation Cleanup Script"
echo "============================"
echo ""

# Check which files exist
echo "Checking files to remove..."
existing_files=()
for file in "${FILES_TO_REMOVE[@]}"; do
    filepath="$DOCS_DIR/$file"
    if [ -f "$filepath" ]; then
        existing_files+=("$filepath")
        echo "  Found: $filepath"
    fi
done

if [ ${#existing_files[@]} -eq 0 ]; then
    echo "  No files to remove found."
    echo ""
    echo "Done! Nothing to clean up."
    exit 0
fi

echo ""
echo "Files to keep:"
for file in "${FILES_TO_KEEP[@]}"; do
    filepath="$DOCS_DIR/$file"
    if [ -f "$filepath" ]; then
        echo "  Keep: $filepath"
    fi
done

echo ""
echo "Summary: ${#existing_files[@]} files will be removed"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "DRY RUN - Would create backup: $BACKUP_FILE"
    echo "DRY RUN - Would remove ${#existing_files[@]} files"
    exit 0
fi

# Create backup
echo ""
echo "Creating backup: $BACKUP_FILE"
tar -czf "$BACKUP_FILE" "${existing_files[@]}"
echo "Backup created successfully"

# Remove files
echo ""
echo "Removing old documentation files..."
for filepath in "${existing_files[@]}"; do
    rm "$filepath"
    echo "  Removed: $filepath"
done

echo ""
echo "Cleanup complete!"
echo ""
echo "Backup saved to: $BACKUP_FILE"
echo "To restore: tar -xzf $BACKUP_FILE"
echo ""
echo "Remaining files in docs/:"
ls -la "$DOCS_DIR"/*.md 2>/dev/null || echo "  No .md files remaining"
