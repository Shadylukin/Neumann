# neumann_shell Benchmarks

The neumann_shell crate provides the interactive CLI with readline, WAL-backed
durability, and snapshot support.

## Overview

Shell benchmarks measure command parsing, dispatch overhead, and WAL replay
performance.

<!-- BENCH:START -->
## Expected Performance Characteristics

| Operation | Complexity | Notes |
| --- | --- | --- |
| Command parse + dispatch | O(query) | Dominated by query router |
| WAL append | O(1) | Single append per command |
| WAL replay | O(n) | Linear in log entry count |
| Snapshot save | O(keys) | Serializes full store state |
| Snapshot load | O(keys) | Deserializes store state |
<!-- BENCH:END -->

## Benchmarking Shell Operations

```bash
cargo bench --package neumann_shell
```
