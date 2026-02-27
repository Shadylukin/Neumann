# neumann_server Benchmarks

The neumann_server crate exposes the QueryRouter over gRPC, handling
serialization, connection management, and request routing.

## Overview

Server benchmarks measure gRPC request/response overhead, serialization
costs, and concurrent connection handling.

<!-- BENCH:START -->
## Expected Performance Characteristics

| Operation | Complexity | Notes |
| --- | --- | --- |
| gRPC unary call | O(query) | Includes ser/de overhead |
| Streaming response | O(rows) | Per-chunk serialization |
| Connection setup | O(1) | TLS handshake if enabled |
<!-- BENCH:END -->

## Benchmarking Server Operations

```bash
cargo bench --package neumann_server
```
