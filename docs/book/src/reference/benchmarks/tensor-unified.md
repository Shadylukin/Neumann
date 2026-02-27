# tensor_unified Benchmarks

The tensor_unified crate provides cross-engine unified entity operations,
allowing a single entity to span relational, graph, and vector storage.

## Overview

Unified entity benchmarks measure the overhead of coordinating operations
across multiple engines compared to single-engine operations.

<!-- BENCH:START -->
## Expected Performance Characteristics

| Operation | Complexity | Notes |
| --- | --- | --- |
| Create unified entity | O(engines) | Writes to each backing engine |
| Read unified entity | O(engines) | Merges results from engines |
| Update unified entity | O(engines) | Partial update per engine |
| Delete unified entity | O(engines) | Cascades across engines |
<!-- BENCH:END -->

## Benchmarking Unified Operations

```bash
cargo bench --package tensor_unified
```
