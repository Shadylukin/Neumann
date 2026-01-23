#!/bin/bash
# Scans a crate for clippy warnings and outputs JSON
# Usage: ./scripts/scan-clippy.sh <crate_name> [lint_level]
#
# Example:
#   ./scripts/scan-clippy.sh tensor_chain pedantic
#   ./scripts/scan-clippy.sh tensor_chain "pedantic nursery"
#
# Output: JSON with files containing warnings

set -euo pipefail

CRATE=${1:?Usage: scan-clippy.sh <crate_name> [lint_level]}
LINT_LEVEL=${2:-pedantic}

# Build clippy flags from lint levels
CLIPPY_FLAGS=""
for level in $LINT_LEVEL; do
    CLIPPY_FLAGS="$CLIPPY_FLAGS -W clippy::$level"
done

# Run clippy and capture stderr (where warnings go)
CLIPPY_OUTPUT=$(cargo clippy --package "$CRATE" -- $CLIPPY_FLAGS 2>&1 || true)

# Parse output and generate JSON
echo "$CLIPPY_OUTPUT" | awk -v crate="$CRATE" -v levels="$LINT_LEVEL" '
BEGIN {
    # Initialize arrays
    split("", file_warnings)
    split("", file_types)
    split("", type_totals)
    total_warnings = 0
}

# Match warning lines: "warning: lint_name"
/^warning:/ {
    current_lint = $2
    gsub(/`/, "", current_lint)
    getline  # Get the location line

    # Match location: "  --> path/to/file.rs:line:col"
    if ($0 ~ /-->.*\.rs:[0-9]+:[0-9]+/) {
        match($0, /[^ ]+\.rs/)
        file = substr($0, RSTART, RLENGTH)

        match($0, /:[0-9]+:/)
        line_num = substr($0, RSTART+1, RLENGTH-2)

        # Only process files from our crate
        if (index(file, crate "/src/") > 0 || index(file, crate "/") == 1) {
            total_warnings++
            file_warnings[file]++

            # Track warning types per file
            key = file ":" current_lint
            if (!(key in file_types)) {
                file_types[key] = 0
            }
            file_types[key]++

            # Track overall type totals
            type_totals[current_lint]++
        }
    }
}

END {
    # Count unique files
    file_count = 0
    for (f in file_warnings) file_count++

    # Output JSON
    printf "{\n"
    printf "  \"crate\": \"%s\",\n", crate
    printf "  \"lint_levels\": [\"%s\"],\n", levels
    printf "  \"files_with_warnings\": [\n"

    # Sort files by warning count (simple bubble sort)
    n = 0
    for (f in file_warnings) {
        files[n++] = f
    }
    for (i = 0; i < n-1; i++) {
        for (j = i+1; j < n; j++) {
            if (file_warnings[files[i]] < file_warnings[files[j]]) {
                tmp = files[i]
                files[i] = files[j]
                files[j] = tmp
            }
        }
    }

    # Output files
    for (i = 0; i < n; i++) {
        f = files[i]
        count = file_warnings[f]

        # Determine priority
        if (count > 30) priority = "high"
        else if (count > 10) priority = "medium"
        else priority = "low"

        if (i > 0) printf ",\n"
        printf "    {\n"
        printf "      \"path\": \"%s\",\n", f
        printf "      \"warning_count\": %d,\n", count
        printf "      \"priority\": \"%s\",\n", priority
        printf "      \"warnings_by_type\": {\n"

        # Output warning types for this file
        first_type = 1
        for (key in file_types) {
            split(key, parts, ":")
            if (parts[1] == f) {
                lint = parts[2]
                if (!first_type) printf ",\n"
                first_type = 0
                printf "        \"%s\": %d", lint, file_types[key]
            }
        }
        printf "\n      }\n"
        printf "    }"
    }

    printf "\n  ],\n"
    printf "  \"total_warnings\": %d,\n", total_warnings
    printf "  \"files_affected\": %d,\n", file_count
    printf "  \"warnings_by_type_summary\": {\n"

    # Output type summary
    first_type = 1
    for (t in type_totals) {
        if (!first_type) printf ",\n"
        first_type = 0
        printf "    \"%s\": %d", t, type_totals[t]
    }
    printf "\n  }\n"
    printf "}\n"
}
'
