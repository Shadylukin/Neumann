#!/bin/bash
# Validate that every crate marked required=true in bench-thresholds.toml
# has at least one [[bench]] section in its Cargo.toml and at least one
# .rs file in benches/.
#
# This is a fast check (~5 seconds, no compilation).
#
# Usage: ./scripts/validate-bench-inventory.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$REPO_ROOT/bench-thresholds.toml"

if [ ! -f "$CONFIG" ]; then
    echo "Error: bench-thresholds.toml not found at $CONFIG"
    exit 1
fi

FAILED=0
CHECKED=0

# Parse required crates from the TOML config.
# Looks for lines like [crates.foo] followed by required = true.
current_crate=""
while IFS= read -r line; do
    # Match [crates.<name>]
    if [[ "$line" =~ ^\[crates\.([a-zA-Z0-9_]+)\] ]]; then
        current_crate="${BASH_REMATCH[1]}"
        continue
    fi

    # Match required = true for the current crate
    if [[ -n "$current_crate" && "$line" =~ ^required[[:space:]]*=[[:space:]]*true ]]; then
        CHECKED=$((CHECKED + 1))
        crate_dir="$REPO_ROOT/$current_crate"

        if [ ! -d "$crate_dir" ]; then
            echo "FAIL: $current_crate -- crate directory not found"
            FAILED=$((FAILED + 1))
            current_crate=""
            continue
        fi

        # Check for [[bench]] in Cargo.toml
        cargo_toml="$crate_dir/Cargo.toml"
        if ! grep -q '^\[\[bench\]\]' "$cargo_toml" 2>/dev/null; then
            echo "FAIL: $current_crate -- no [[bench]] section in Cargo.toml"
            FAILED=$((FAILED + 1))
            current_crate=""
            continue
        fi

        # Check for at least one .rs file in benches/
        bench_dir="$crate_dir/benches"
        if [ ! -d "$bench_dir" ] || ! ls "$bench_dir"/*.rs >/dev/null 2>&1; then
            echo "FAIL: $current_crate -- no .rs files in benches/"
            FAILED=$((FAILED + 1))
            current_crate=""
            continue
        fi

        echo "  OK: $current_crate"
        current_crate=""
    fi

    # Reset current_crate on next section header
    if [[ "$line" =~ ^\[ && ! "$line" =~ ^\[crates\. ]]; then
        current_crate=""
    fi
done < "$CONFIG"

echo ""
echo "Checked $CHECKED crates, $FAILED failed."

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "Add benchmarks to the failing crates or update bench-thresholds.toml."
    exit 1
fi

echo "All required crates have benchmarks."
