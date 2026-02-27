# How to Use Bloom Filters

This guide shows how to configure and use the BloomFilter in `tensor_store` for
fast probabilistic key rejection.

**See also**: [Bloom Filter Explanation](../explanation/bloom-filter.md) |
[API Reference](../reference/api/tensor-store.md)

---

## When to Use

Bloom filters are most effective when:

- The key space is sparse (most lookups are misses).
- You want to avoid the cost of a full BTreeMap lookup for non-existent keys.
- False positives (rare, controlled by configuration) are acceptable.

## 1. Create a TensorStore with Bloom Filter

```rust
// expected_items: how many keys you expect to store
// false_positive_rate: probability of a false "exists" (0.01 = 1%)
let store = TensorStore::with_bloom_filter(10_000, 0.01);
```

Choose parameters based on your workload:

| Scenario | Expected Items | FP Rate | Memory |
| --- | --- | --- | --- |
| Small cache | 10,000 | 1% | ~12 KB |
| Medium dataset | 100,000 | 1% | ~117 KB |
| Large collection | 1,000,000 | 1% | ~1.2 MB |
| High-precision | 10,000 | 0.1% | ~18 KB |

## 2. Use Normally

The bloom filter is transparent. `put` automatically inserts into the filter,
and `exists` checks the filter before the BTreeMap.

```rust
store.put("key:1", tensor)?;
store.put("key:2", tensor)?;

// O(1) rejection if key definitely doesn't exist
if store.exists("key:999") {
    // Might be a false positive -- do the full lookup
    let data = store.get("key:999")?;
}

// Definite misses are fast
assert!(!store.exists("key:nonexistent"));  // O(1), no BTreeMap lookup
```

## 3. Rebuild After Snapshot Load

Bloom filter state is not persisted in snapshots. Use the bloom-aware load
function to rebuild it:

```rust
let store = TensorStore::load_snapshot_with_bloom_filter(
    "data.bin",
    10_000,   // expected items
    0.01      // false positive rate
)?;
```

This scans all keys once during load to populate the filter.

## 4. Choosing Parameters

**Expected items**: estimate the steady-state key count. Overestimating wastes
a small amount of memory. Underestimating increases the false positive rate.

**False positive rate**: 1% (0.01) is a good default. Lower rates use more
memory but reduce unnecessary BTreeMap lookups. For workloads where misses are
rare, a higher rate (5%) saves memory with minimal impact.

**Rule of thumb**: the bloom filter uses approximately `9.6 * n` bits for a 1%
false positive rate, where `n` is the expected item count. This is independent
of key size.
