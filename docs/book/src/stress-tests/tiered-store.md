# TieredStore Stress Tests

Stress tests for the two-tier hot/cold storage system with automatic data migration.

## Test Suite

| Test | Scale | Description |
|------|-------|-------------|
| `stress_tiered_hot_only_scale` | 1M entities | Hot-only tier at scale |
| `stress_tiered_migration_under_load` | 100K entities | Hot/cold migration with concurrent load |
| `stress_tiered_hot_read_latency` | 100K entities | Random access read latency |

## Results

| Test | Key Metric | Result |
|------|------------|--------|
| Hot-only 1M | Throughput | **689K entities/sec** |
| Migration | Concurrent access | Works correctly |
| Read latency | p50 | < 3us |
| Read latency | p99 | < 500us |

## Running

```bash
# Run all TieredStore stress tests
cargo test --release -p stress_tests --test tiered_store_stress -- --ignored --nocapture

# Run specific test
cargo test --release -p stress_tests stress_tiered_hot_only_scale -- --ignored --nocapture
```

## Hot-Only Scale Test

Tests TieredStore performance with only hot tier active (no cold storage).

**What it validates:**
- In-memory performance at scale
- DashMap + instrumentation overhead
- Memory usage patterns

**Expected behavior:**
- Throughput > 500K entities/sec
- Linear memory growth
- Consistent latency distribution

## Migration Under Load

Tests hot-to-cold data migration while concurrent reads/writes continue.

**What it validates:**
- Migration correctness during active use
- No data loss during tier transitions
- Read consistency during migration

**Expected behavior:**
- All data accessible before and after migration
- Reads don't block on migration
- Writes to migrated keys work correctly

## Hot Read Latency

Tests random access read latency for hot tier data.

**What it validates:**
- Read latency distribution
- Hot path optimization
- Cache efficiency

**Expected behavior:**
- p50 latency < 3us
- p99 latency < 500us
- No extreme outliers (p999 < 10ms)

## Architecture

```
TieredStore
    │
    ├── Hot Tier (DashMap)
    │   ├── Fast in-memory access
    │   ├── Access instrumentation
    │   └── Automatic hot shard tracking
    │
    └── Cold Tier (mmap)
        ├── Disk-backed storage
        ├── Memory-efficient for large datasets
        └── Transparent promotion on access
```

## Migration Strategies

| Strategy | Trigger | Use Case |
|----------|---------|----------|
| Time-based | Entries older than threshold | Aging data |
| Access-based | Cold shards (low access) | Infrequent data |
| Memory-based | Hot tier size limit | Memory pressure |

## Configuration

```rust
let config = TieredConfig {
    cold_dir: PathBuf::from("/var/lib/neumann/cold"),
    cold_capacity: 1_000_000,  // Max cold entries
    sample_rate: 0.01,         // 1% access sampling
};
```
