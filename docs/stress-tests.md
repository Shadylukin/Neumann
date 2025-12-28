# Stress Tests

Comprehensive stress testing infrastructure for Neumann targeting 1M entity scale with extensive coverage of concurrency, data volume, and sustained load.

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

### TensorStore Stress (`tensor_store_stress.rs`)

| Test | Scale | Description |
|------|-------|-------------|
| `stress_tensor_store_1m_concurrent_writes` | 1M entities, 16 threads | Concurrent writes to verify DashMap scalability |
| `stress_tensor_store_high_contention` | 1K keys, 16 threads, 100K ops/thread | High contention workload (16:1 ratio) |
| `stress_tensor_store_scan_during_writes` | 100K entities | Concurrent scanning while writing |
| `stress_bloom_filter_1m` | 1M keys | BloomFilter false positive rate at scale |

**Results:**
- 1M writes: **7.5M entities/sec** throughput, p50 < 1us, p99 < 10us
- High contention: **2.5M ops/sec** with 16:1 contention ratio
- BloomFilter: **0.88% FP rate** at 1M keys (target 1%)

### HNSW Stress (`hnsw_stress.rs`)

| Test | Scale | Description |
|------|-------|-------------|
| `stress_hnsw_1m_vectors` | 1M 128d vectors | Build 1M vector index |
| `stress_hnsw_100k_concurrent_build` | 100K vectors, 16 threads | Concurrent index construction |
| `stress_hnsw_search_during_insert` | 50K vectors, 4+4 threads | Concurrent search during insert |
| `stress_hnsw_recall_under_load` | 10K vectors | Verify recall@10 under load |

**Results:**
- 1M vectors: **3,372 vectors/sec** insert, 0.11ms search latency
- 100K concurrent: **1,155 vectors/sec** with 16 threads
- Recall@10: **99.8%** average (min 90%)

### TieredStore Stress (`tiered_store_stress.rs`)

| Test | Scale | Description |
|------|-------|-------------|
| `stress_tiered_hot_only_scale` | 1M entities | Hot-only tier at scale |
| `stress_tiered_migration_under_load` | 100K entities | Hot/cold migration with concurrent load |
| `stress_tiered_hot_read_latency` | 100K entities | Random access read latency |

**Results:**
- Hot-only 1M: **689K entities/sec** throughput
- Migration: Concurrent access during migration works correctly
- Read latency: p50 < 3us, p99 < 500us

### BloomFilter Stress (`bloom_filter_stress.rs`)

| Test | Scale | Description |
|------|-------|-------------|
| `stress_bloom_fp_rate_at_scale` | 1M keys | False positive rate verification |
| `stress_bloom_concurrent_add_query` | 100K keys, 8+8 threads | Concurrent add/query operations |
| `stress_bloom_bit_concurrency` | 64M bits, 32 threads | Bit-level concurrency safety |

**Results:**
- FP rate: **0.88%** at 1M keys (8.2M adds/sec)
- Concurrent: **4M ops/sec** with 16 threads
- Bit concurrency: **15M+ ops/sec**, 0 missing keys

### Mixed Workload Stress (`mixed_workload_stress.rs`)

| Test | Scale | Description |
|------|-------|-------------|
| `stress_all_engines_concurrent` | 25K ops/thread, 12 threads | All engines under concurrent load |
| `stress_realistic_workload` | 30s duration | Mixed OLTP + search + traversal |

**Results:**
- All engines: **841 ops/sec** combined (relational: 12ms p50, graph: 5us p50, vector: < 1us p50)
- Realistic workload: 232 ops/sec mixed (91 reads/sec, 68 writes/sec, 72 searches/sec)

### QueryRouter Stress (`query_router_stress.rs`)

| Test | Scale | Description |
|------|-------|-------------|
| `stress_router_concurrent_queries` | 100 streams, 100 queries each | Concurrent query execution |
| `stress_router_parallel_inserts` | 50 streams, 100 inserts each | Parallel insert operations |
| `stress_router_sustained_writes` | 8 threads, 10s duration | Sustained write throughput |

**Results:**
- Concurrent queries: **2.4M queries/sec** throughput (ultra-low latency for parse path)
- Parallel inserts: **1.5M inserts/sec** throughput

### Duration Stress (`duration_stress.rs`)

| Test | Scale | Description |
|------|-------|-------------|
| `stress_1_hour_sustained_load` | 5min (configurable to 1hr) | Long-running stability test |
| `stress_memory_leak_detection` | 100 cycles, 10K items each | Memory leak detection |
| `stress_throughput_stability` | 60s duration | Throughput variance analysis |

**Results:**
- Sustained load: **461 ops/sec** average over 5 minutes
- Memory leak: **No leak detected** (store size 0 after cycles)
- Throughput: CV warning (high variance due to background processes)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
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
|--------|-------------|
| p50 | Median latency |
| p99 | 99th percentile |
| p999 | 99.9th percentile |
| max | Maximum observed |

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

## Running in CI

For CI pipelines, use quick_config with limited duration:

```yaml
- name: Run stress tests
  run: |
    STRESS_DURATION=30 cargo test --release -p stress_tests -- --ignored --nocapture
  timeout-minutes: 15
```
