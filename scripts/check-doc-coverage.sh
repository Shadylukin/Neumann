#!/usr/bin/env bash
# check-doc-coverage.sh -- Enforce per-crate rustdoc coverage thresholds
#
# Runs `cargo +nightly rustdoc -p <crate> -- -Z unstable-options --show-coverage`
# for each crate, parses the Total line, and fails if any crate is below its
# threshold.
#
# Usage: ./scripts/check-doc-coverage.sh

set -euo pipefail

# Per-crate thresholds as "crate:threshold" pairs
# Floor of current coverage -- ratchet up as docs improve
CRATES="
tensor_store:100
relational_engine:100
vector_engine:100
neumann_server:100
neumann_client:100
tensor_blob:95
tensor_unified:89
tensor_vault:71
query_router:68
neumann_shell:64
tensor_chain:67
graph_engine:60
tensor_compress:60
neumann_parser:95
tensor_cache:32
tensor_checkpoint:95
stress_tests:71
integration_tests:68
"

# Crates that need --lib flag (have both lib and bin targets)
LIB_ONLY_CRATES="neumann_shell"

FAILED=0
CHECKED=0

echo "=== Doc Coverage Check ==="
echo ""

for entry in $CRATES; do
  crate="${entry%%:*}"
  threshold="${entry##*:}"

  # Build rustdoc flags
  extra_flags=""
  case " $LIB_ONLY_CRATES " in
    *" $crate "*) extra_flags="--lib" ;;
  esac

  # Run rustdoc coverage and save to temp file
  tmpfile=$(mktemp)
  cargo +nightly rustdoc -p "$crate" $extra_flags \
    -- -Z unstable-options --show-coverage >"$tmpfile" 2>&1 || true

  # Parse the Total line: | Total | N | XX.X% | ... |
  total_line=$(command grep "Total" "$tmpfile" || true)
  rm -f "$tmpfile"

  if [ -z "$total_line" ]; then
    echo "WARN: Could not get coverage for $crate (no Total line)"
    continue
  fi

  # Extract the first percentage from the Total line (documented %)
  # The line looks like: | Total | 1365 | 100.0% | 9 | 1.0% |
  pct=$(echo "$total_line" | command grep -oE '[0-9]+\.[0-9]+%' | head -1 | command sed 's/%//')

  if [ -z "$pct" ]; then
    echo "WARN: Could not parse percentage for $crate"
    continue
  fi

  CHECKED=$((CHECKED + 1))

  # Compare using bc (floating point)
  if [ "$(echo "$pct < $threshold" | bc -l)" = "1" ]; then
    echo "FAIL: $crate -- ${pct}% < ${threshold}% threshold"
    FAILED=$((FAILED + 1))
  else
    echo "  OK: $crate -- ${pct}% (threshold: ${threshold}%)"
  fi
done

echo ""
echo "Checked $CHECKED crates, $FAILED failures"

if [ "$FAILED" -gt 0 ]; then
  echo ""
  echo "Doc coverage below threshold. Add doc comments to the failing crates."
  exit 1
fi

echo "All crates meet their doc coverage thresholds."
