#!/bin/bash
# Scans a crate for per-file coverage and outputs JSON
# Usage: ./scripts/scan-coverage.sh <crate_name> [threshold]
#
# Example:
#   ./scripts/scan-coverage.sh tensor_chain 95
#
# Output: JSON with files below threshold

set -euo pipefail

CRATE=${1:?Usage: scan-coverage.sh <crate_name> [threshold]}
THRESHOLD=${2:-95}

# Get crate-specific thresholds
get_threshold() {
    case "$1" in
        neumann_shell)   echo 88 ;;
        neumann_parser)  echo 91 ;;
        tensor_blob)     echo 91 ;;
        query_router)    echo 92 ;;
        *)               echo "${THRESHOLD}" ;;
    esac
}

CRATE_THRESHOLD=$(get_threshold "$CRATE")

# Run coverage and capture output
COVERAGE_OUTPUT=$(cargo llvm-cov --package "$CRATE" --all-features --summary-only 2>&1 || true)

# Parse output and generate JSON
# Format: Filename  Regions  Missed  Cover%  Functions  Missed  Cover%  Lines  Missed  Cover%
echo "$COVERAGE_OUTPUT" | awk -v threshold="$CRATE_THRESHOLD" -v crate="$CRATE" '
BEGIN {
    printf "{\n"
    printf "  \"crate\": \"%s\",\n", crate
    printf "  \"threshold\": %d,\n", threshold
    printf "  \"files_needing_work\": [\n"
    first = 1
    total_files = 0
    files_below = 0
}

# Match lines that look like .rs file coverage (start with filename.rs)
/^[a-z_]+\.rs/ {
    # Extract filename
    file = $1
    total_files++

    # Line coverage is the 10th field (last Cover column)
    # Format: Lines  Missed  Cover%
    # Fields: $8=Lines, $9=Missed Lines, $10=Line Cover%
    pct = $10
    gsub(/%/, "", pct)
    pct = pct + 0

    missed = $9 + 0

    if (pct < threshold) {
        files_below++

        # Calculate priority
        if (missed > 200 || pct < 85) {
            priority = "high"
        } else if (missed > 50 || pct < 92) {
            priority = "medium"
        } else {
            priority = "low"
        }

        # Construct full path
        full_path = crate "/src/" file

        if (!first) printf ",\n"
        first = 0
        printf "    {\n"
        printf "      \"path\": \"%s\",\n", full_path
        printf "      \"current_coverage\": %.2f,\n", pct
        printf "      \"lines_uncovered\": %d,\n", missed
        printf "      \"priority\": \"%s\"\n", priority
        printf "    }"
    }
}

END {
    printf "\n  ],\n"
    printf "  \"total_files_analyzed\": %d,\n", total_files
    printf "  \"files_below_threshold\": %d\n", files_below
    printf "}\n"
}
'
