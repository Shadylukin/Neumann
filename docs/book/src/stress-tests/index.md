# Stress Tests

Comprehensive stress testing infrastructure for Neumann targeting 1M entity
scale with extensive coverage of concurrency, data volume, and sustained load.

## Quick Start

```bash
# Run all stress tests (45+ min total)
cargo test --release -p stress_tests -- --ignored --nocapture

# Run specific test suite
cargo test --release -p stress_tests --test hnsw_stress -- --ignored --nocapture

# Run with custom duration (30s instead of default)
STRESS_DURATION=30 cargo test --release -p stress_tests -- --ignored --nocapture
```

## Test Suites

| Suite | Tests | Description |
| --- | --- | --- |
| [HNSW Stress](hnsw.md) | 4 | 1M vector indexing, concurrent builds |
| [TieredStore Stress](tiered-store.md) | 3 | Hot/cold migration under load |
| [Mixed Workload](mixed-workload.md) | 2 | All engines concurrent, realistic patterns |
| TensorStore Stress | 4 | 1M entities, high contention |
| BloomFilter Stress | 3 | 1M keys, bit-level concurrency |
| QueryRouter Stress | 3 | Concurrent queries, sustained writes |
| Duration Stress | 3 | Long-running stability, memory leaks |

## Key Performance Findings

### TensorStore (DashMap)

- **7.5M writes/sec** at 1M entities
- Sub-microsecond median latency
- Handles 16:1 contention ratio with 2.5M ops/sec

### HNSW Index

- **3,372 vectors/sec** insert rate at 1M scale
- **0.11ms** search latency (p50)
- **99.8%** recall@10 under concurrent load

### BloomFilter

- **0.88% FP rate** at 1M keys (target 1%)
- **15M+ ops/sec** bit-level operations
- Thread-safe with AtomicU64

### Mixed Workloads

- All engines can operate concurrently
- Graph operations (5us p50) and vector ops (< 1us p50) are fastest
- Relational engine adds ~12ms p50 overhead due to schema operations

## Configuration

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `STRESS_DURATION` | 30 (quick) / 600 (full) | Test duration in seconds |
| `STRESS_THREADS` | 16 | Thread count for tests |

### Config Presets

```rust
// Quick mode (CI): 100K entities, 4 threads, 30s
let config = quick_config();

// Full mode (local): 1M entities, 16 threads, 10min
let config = full_config();

// Endurance mode: 500K entities, 8 threads, 1 hour
let config = endurance_config();
```

## Latency Metrics

All tests report percentile latencies using HdrHistogram:

| Metric | Description |
| --- | --- |
| p50 | Median latency |
| p99 | 99th percentile |
| p999 | 99.9th percentile |
| max | Maximum observed |

## Running in CI

For CI pipelines, use quick_config with limited duration:

```yaml
- name: Run stress tests
  run: |
    STRESS_DURATION=30 cargo test --release -p stress_tests -- --ignored --nocapture
  timeout-minutes: 15
```
