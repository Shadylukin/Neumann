#!/bin/bash
# Generate benchmark documentation from Criterion JSON output
#
# Usage: ./scripts/generate-bench-docs.sh
#
# Prerequisites:
#   - Run `cargo bench` first to generate Criterion output
#   - jq must be installed for JSON parsing

set -e

CRITERION_DIR="target/criterion"
OUTPUT_DIR="docs/book/src/benchmarks"
TIMESTAMP=$(date -u +"%Y-%m-%d %H:%M:%S UTC")

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

echo "Generating benchmark documentation..."
echo "Source: $CRITERION_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Function to format time with appropriate unit
format_time() {
    local ns=$1
    if (( $(echo "$ns >= 1000000000" | bc -l) )); then
        printf "%.2f s" $(echo "$ns / 1000000000" | bc -l)
    elif (( $(echo "$ns >= 1000000" | bc -l) )); then
        printf "%.2f ms" $(echo "$ns / 1000000" | bc -l)
    elif (( $(echo "$ns >= 1000" | bc -l) )); then
        printf "%.2f us" $(echo "$ns / 1000" | bc -l)
    else
        printf "%.2f ns" $ns
    fi
}

# Function to process a benchmark group
process_group() {
    local group_path=$1
    local group_name=$(basename "$group_path")

    local estimates_file="$group_path/new/estimates.json"
    if [ ! -f "$estimates_file" ]; then
        return
    fi

    local mean=$(jq -r '.mean.point_estimate' "$estimates_file" 2>/dev/null)
    local median=$(jq -r '.median.point_estimate' "$estimates_file" 2>/dev/null)
    local std_dev=$(jq -r '.std_dev.point_estimate' "$estimates_file" 2>/dev/null)

    if [ "$mean" != "null" ] && [ -n "$mean" ]; then
        local formatted_mean=$(format_time $mean)
        local formatted_median=$(format_time $median)
        echo "| $group_name | $formatted_mean | $formatted_median |"
    fi
}

# Generate summary for each crate
for crate_dir in "$CRITERION_DIR"/*/; do
    if [ ! -d "$crate_dir" ]; then
        continue
    fi

    crate_name=$(basename "$crate_dir")

    # Skip report directory
    if [ "$crate_name" = "report" ]; then
        continue
    fi

    echo "Processing: $crate_name"

    # Check if there are benchmark results
    has_results=false
    for bench_dir in "$crate_dir"*/; do
        if [ -f "$bench_dir/new/estimates.json" ]; then
            has_results=true
            break
        fi
    done

    if [ "$has_results" = false ]; then
        echo "  No results found, skipping..."
        continue
    fi

    # Generate markdown table header
    echo ""
    echo "### $crate_name"
    echo ""
    echo "| Benchmark | Mean | Median |"
    echo "|-----------|------|--------|"

    # Process each benchmark in the group
    for bench_dir in "$crate_dir"*/; do
        if [ -d "$bench_dir" ]; then
            process_group "$bench_dir"
        fi
    done

    echo ""
done

echo ""
echo "Done! Benchmark data generated."
echo ""
echo "To update documentation:"
echo "1. Review the output above"
echo "2. Copy relevant tables to docs/book/src/benchmarks/*.md"
echo "3. Update timestamps in files"
echo ""
echo "For full HTML reports, open: target/criterion/report/index.html"
