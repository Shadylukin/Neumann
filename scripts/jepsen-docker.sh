#!/usr/bin/env bash
set -euo pipefail

echo "=== Neumann Docker Jepsen Tests ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# 1. Build the jepsen Docker image
echo "--- Building jepsen Docker image ---"
docker build --target jepsen -t neumann:jepsen .

# 2. Run the Docker-based Jepsen tests
echo "--- Running Docker Jepsen tests ---"
cargo nextest run --package integration_tests --test jepsen_docker -- --ignored "$@"

echo "=== Done ==="
