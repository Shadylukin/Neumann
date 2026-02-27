# How to Configure Tiered Storage

This guide walks through setting up and operating two-tier (hot/cold) storage
in `tensor_store`.

**See also**: [Tiered Storage Explanation](../explanation/tiered-storage.md) |
[API Reference](../reference/api/tensor-store.md)

---

## 1. Create a TieredStore

```rust
use tensor_store::{TieredStore, TieredConfig};

let config = TieredConfig {
    cold_dir: "/data/cold".into(),
    cold_capacity: 64 * 1024 * 1024,  // 64 MB initial cold file
    sample_rate: 100,                   // 1% access sampling
};

let mut store = TieredStore::new(config)?;
```

| Parameter | Recommended Value | Notes |
| --- | --- | --- |
| `cold_dir` | A dedicated directory on fast SSD | Avoid tmpfs for persistence |
| `cold_capacity` | 2x expected cold data size | File grows if needed, but pre-allocation avoids fragmentation |
| `sample_rate` | 100 (1%) for production, 1 (100%) for debugging | Higher rates increase tracking overhead |

## 2. Write Data Normally

All writes go to the hot tier. There is no special API for cold writes.

```rust
store.put("user:1", tensor_data);
store.put("user:2", tensor_data);
```

## 3. Migrate Cold Data

Call `migrate_cold()` with a staleness threshold in milliseconds. Keys in shards
not accessed within the threshold move to cold storage.

```rust
// Migrate shards not accessed in the last 30 seconds
let migrated_count = store.migrate_cold(30_000)?;
println!("Migrated {} keys to cold storage", migrated_count);
```

**When to call**: during maintenance windows, low-traffic periods, or on a
scheduled interval. Avoid calling during peak load, as migration involves
serialization and disk I/O.

## 4. Read Data (Automatic Promotion)

Reads check hot first, then cold. Accessing cold data automatically promotes it
back to hot.

```rust
// This works regardless of whether the key is hot or cold
let data = store.get("user:1")?;
```

No application-level code change is needed. The promotion is transparent.

## 5. Monitor Statistics

```rust
let stats = store.stats();
println!("Hot: {}, Cold: {}", stats.hot_count, stats.cold_count);
println!("Cold lookups: {}, Cold hits: {}", stats.cold_lookups, stats.cold_hits);
println!("Migrations to cold: {}", stats.migrations_to_cold);
println!("Promotions to hot: {}", stats.migrations_to_hot);
```

Key metrics to watch:

- **cold_hits / cold_lookups**: If this ratio is high, data is being promoted
  frequently, which suggests the staleness threshold is too aggressive.
- **migrations_to_hot**: A high rate of promotions indicates churn. Consider
  increasing the threshold or reviewing access patterns.
- **hot_count vs cold_count**: Gives a picture of the working set size.

## 6. Enable Access Tracking (Optional)

For more detailed shard-level access analysis:

```rust
// Enable instrumentation with 1% sampling
let store = TensorStore::with_instrumentation(100);

// After operations, inspect access patterns
let snapshot = store.access_snapshot()?;
println!("Hot shards: {:?}", store.hot_shards(5)?);
println!("Cold shards: {:?}", store.cold_shards(30_000)?);
```

This is useful for tuning the `threshold_ms` parameter: if `cold_shards(30_000)`
returns most shards, the threshold may be too short for your workload.
