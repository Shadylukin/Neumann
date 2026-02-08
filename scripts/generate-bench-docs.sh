#!/bin/bash
# Generate benchmark documentation from Criterion JSON output
#
# Replaces content between <!-- BENCH:START --> and <!-- BENCH:END --> markers
# in docs/book/src/benchmarks/*.md with fresh Criterion data.
#
# Usage: ./scripts/generate-bench-docs.sh [criterion_dir]
#
# Prerequisites:
#   - Run `cargo bench` first to generate Criterion output
#   - jq must be installed for JSON parsing

set -euo pipefail

CRITERION_DIR="${1:-target/criterion}"
DOCS_DIR="docs/book/src/benchmarks"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

# Map Criterion group directories to markdown file basenames.
# Criterion creates dirs named after the criterion_group! identifier or
# the benchmark function group. Multiple groups can map to the same file.
declare -A GROUP_TO_FILE=(
    # tensor_store bench groups
    [tensor_store_bench]="tensor-store"
    [metadata_slab]="tensor-store"
    [sparse_vector]="tensor-store"
    [delta_vector]="tensor-store"
    [kmeans]="tensor-store"
    [hnsw_concurrent]="tensor-store"
    [wal]="tensor-store"
    # engine bench groups
    [relational_engine]="relational-engine"
    [graph_engine]="graph-engine"
    [vector_engine]="vector-engine"
    # specialized storage
    [tensor_compress]="tensor-compress"
    [tensor_vault]="tensor-vault"
    [tensor_cache]="tensor-cache"
    [tensor_blob]="tensor-blob"
    [blob]="tensor-blob"
    # distributed
    [tensor_chain]="tensor-chain"
    [distributed]="tensor-chain"
    # query layer
    [neumann_parser]="neumann-parser"
    [query_router]="query-router"
    [neumann_shell]="neumann-shell"
)

# Check prerequisites
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed."
    echo "Install with: brew install jq (macOS) or apt-get install jq (Linux)"
    exit 1
fi

if [ ! -d "$CRITERION_DIR" ]; then
    echo "Error: Criterion output directory not found at $CRITERION_DIR"
    echo "Run 'cargo bench' first to generate benchmark data."
    exit 1
fi

# Format nanoseconds to human-readable time
format_time() {
    local ns=$1
    if (( $(echo "$ns >= 1000000000" | bc -l) )); then
        printf "%.2f s" "$(echo "$ns / 1000000000" | bc -l)"
    elif (( $(echo "$ns >= 1000000" | bc -l) )); then
        printf "%.2f ms" "$(echo "$ns / 1000000" | bc -l)"
    elif (( $(echo "$ns >= 1000" | bc -l) )); then
        printf "%.2f us" "$(echo "$ns / 1000" | bc -l)"
    else
        printf "%.2f ns" "$ns"
    fi
}

# Collect benchmark rows for a single Criterion group directory.
# Each sub-directory with estimates.json becomes a row.
collect_group_rows() {
    local group_path=$1
    local rows=""

    for bench_dir in "$group_path"/*/; do
        [ -d "$bench_dir" ] || continue
        local bench_name
        bench_name=$(basename "$bench_dir")
        [ "$bench_name" = "report" ] && continue

        local estimates="$bench_dir/new/estimates.json"
        [ -f "$estimates" ] || continue

        local mean median
        mean=$(jq -r '.mean.point_estimate' "$estimates" 2>/dev/null)
        median=$(jq -r '.median.point_estimate' "$estimates" 2>/dev/null)

        if [ "$mean" != "null" ] && [ -n "$mean" ]; then
            local fmt_mean fmt_median
            fmt_mean=$(format_time "$mean")
            fmt_median=$(format_time "$median")
            rows+="| $bench_name | $fmt_mean | $fmt_median |"$'\n'
        fi
    done
    echo -n "$rows"
}

# Replace content between BENCH markers in a markdown file.
# $1 = file path, $2 = new content (without markers themselves)
replace_markers() {
    local file=$1
    local content=$2

    if [ ! -f "$file" ]; then
        echo "  Warning: $file not found, skipping"
        return
    fi

    if ! grep -q '<!-- BENCH:START -->' "$file"; then
        echo "  Warning: no BENCH:START marker in $file, skipping"
        return
    fi

    # Build the replacement block
    local block
    block=$(cat <<ENDBLOCK
<!-- BENCH:START -->
$content
*Auto-generated from Criterion benchmarks on $TIMESTAMP.*
<!-- BENCH:END -->
ENDBLOCK
)

    # Use awk to replace everything between markers (inclusive)
    awk -v replacement="$block" '
        /<!-- BENCH:START -->/ { print replacement; skip=1; next }
        /<!-- BENCH:END -->/   { skip=0; next }
        !skip { print }
    ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
}

echo "Generating benchmark documentation..."
echo "Source: $CRITERION_DIR"
echo "Output: $DOCS_DIR"
echo ""

# Track which markdown files got data, and collect summary rows for index.md
declare -A FILE_CONTENT
declare -a SUMMARY_ROWS

for crate_dir in "$CRITERION_DIR"/*/; do
    [ -d "$crate_dir" ] || continue
    crate_name=$(basename "$crate_dir")
    [ "$crate_name" = "report" ] && continue

    # Try to find the markdown file for this group
    target_file=""
    for prefix in "${!GROUP_TO_FILE[@]}"; do
        # Match if the crate_name starts with the prefix
        if [[ "$crate_name" == "$prefix"* ]]; then
            target_file="${GROUP_TO_FILE[$prefix]}"
            break
        fi
    done

    if [ -z "$target_file" ]; then
        echo "  No mapping for Criterion group '$crate_name', skipping"
        continue
    fi

    # Check if there are any results
    has_results=false
    for bench_dir in "$crate_dir"*/; do
        if [ -f "$bench_dir/new/estimates.json" ]; then
            has_results=true
            break
        fi
    done
    [ "$has_results" = true ] || continue

    echo "Processing: $crate_name -> $target_file.md"

    # Build table for this group
    local_content=""
    local_content+="## $crate_name"$'\n'
    local_content+=""$'\n'
    local_content+="| Benchmark | Mean | Median |"$'\n'
    local_content+="|-----------|------|--------|"$'\n'

    rows=$(collect_group_rows "$crate_dir")
    local_content+="$rows"$'\n'

    # Append to any existing content for this file (multiple groups -> same file)
    FILE_CONTENT[$target_file]+="$local_content"

    # Pick the first benchmark row as a representative for the summary
    first_row=$(echo "$rows" | head -1)
    if [ -n "$first_row" ]; then
        first_bench=$(echo "$first_row" | cut -d'|' -f2 | xargs)
        first_mean=$(echo "$first_row" | cut -d'|' -f3 | xargs)
        SUMMARY_ROWS+=("| [$target_file]($target_file.md) | $first_bench | $first_mean |")
    fi
done

echo ""

# Write collected content to each markdown file
for file_base in "${!FILE_CONTENT[@]}"; do
    md_path="$DOCS_DIR/$file_base.md"
    echo "Updating: $md_path"
    replace_markers "$md_path" "${FILE_CONTENT[$file_base]}"
done

# Update index.md summary table
if [ ${#SUMMARY_ROWS[@]} -gt 0 ]; then
    echo "Updating: $DOCS_DIR/index.md"
    summary_content=""
    summary_content+="### Performance Summary"$'\n'
    summary_content+=""$'\n'
    summary_content+="| Component | Key Benchmark | Mean |"$'\n'
    summary_content+="|-----------|--------------|------|"$'\n'
    for row in "${SUMMARY_ROWS[@]}"; do
        summary_content+="$row"$'\n'
    done
    replace_markers "$DOCS_DIR/index.md" "$summary_content"
fi

echo ""
echo "Done! Benchmark docs updated in-place."
echo "Updated files have fresh data between <!-- BENCH:START/END --> markers."
echo ""
echo "For full HTML reports, open: target/criterion/report/index.html"
