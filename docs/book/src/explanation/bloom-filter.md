# Bloom Filter

The BloomFilter in `tensor_store` provides O(1) probabilistic rejection of
non-existent keys. It is most useful for sparse key spaces where the majority of
lookups are misses -- a common pattern in graph traversal and cache probing.

**See also**: [How-to: Bloom Filters](../how-to/bloom-filters.md) |
[API Reference](../reference/api/tensor-store.md)

---

## Mathematical Foundation

A Bloom filter is a bit array of size `m` with `k` independent hash functions.
Inserting an element sets `k` bits; querying checks whether all `k` bits are
set. If any bit is zero, the element is definitely absent. If all bits are set,
the element is *probably* present (a false positive is possible).

### Optimal Parameters

Given `n` expected items and a target false positive rate `p`:

**Bit array size**:

```
m = -n * ln(p) / (ln(2)^2)
```

**Number of hash functions**:

```
k = (m / n) * ln(2)
```

The implementation clamps `k` to the range 1..=16 to bound per-query cost.

### False Positive Analysis

The false positive probability after inserting `n` items into a filter of size
`m` with `k` hash functions is approximately:

```
p_fp = (1 - e^(-kn/m))^k
```

This formula assumes ideal hash functions (independent, uniform). In practice,
SipHash with different seeds provides sufficient independence for non-adversarial
workloads.

| Expected Items | FP Rate | Bits | Hash Functions | Memory |
| --- | --- | --- | --- | --- |
| 10,000 | 1% | 95,851 | 7 | ~12 KB |
| 10,000 | 0.1% | 143,776 | 10 | ~18 KB |
| 100,000 | 1% | 958,506 | 7 | ~117 KB |
| 1,000,000 | 1% | 9,585,059 | 7 | ~1.2 MB |

The memory cost is modest: 1% false positive rate costs roughly 9.6 bits per
element, regardless of element size.

---

## Implementation Design

```rust
pub struct BloomFilter {
    bits: Box<[AtomicU64]>,  // Atomic u64 blocks for lock-free access
    num_bits: usize,
    num_hashes: usize,
}
```

### Hash Function

The filter uses SipHash with per-function seed variation:

```rust
fn hash_index<K: Hash>(&self, key: &K, seed: usize) -> usize {
    let mut hasher = SipHasher::new_with_seed(seed as u64);
    key.hash(&mut hasher);
    (hasher.finish() as usize) % self.num_bits
}
```

Each of the `k` hash functions uses a different `seed` value (0 through k-1).
SipHash is not cryptographic, but it provides good distribution and resistance
to hash flooding for this use case.

### Thread Safety

Bit operations use `AtomicU64` with `Relaxed` ordering. This means:

- **Insertions are eventually visible** to concurrent readers. There is no
  happens-before guarantee between an insert and a subsequent query on a
  different thread.
- **False negatives are transiently possible** during concurrent insert+query,
  but they resolve once the atomic store becomes visible.
- **No locks are needed**, keeping the filter on the fast path for `exists()`
  checks.

Relaxed ordering is acceptable because a transient false negative (missing a
just-inserted key) is harmless -- the caller falls through to the authoritative
BTreeMap lookup.

---

## Design Rationale

**Why not a counting Bloom filter?** The tensor_store BloomFilter does not
support deletion. A counting variant would quadruple memory usage (4-bit
counters instead of 1-bit flags) for a feature that is rarely needed: keys in
TensorStore are typically long-lived, and the filter is rebuilt from scratch
after snapshot restore.

**Why not persisted in snapshots?** The Bloom filter state is ephemeral. Since
it can be rebuilt in O(n) by scanning all keys, persisting it would add
complexity to the snapshot format without meaningful benefit. After loading a
snapshot, call `TensorStore::load_snapshot_with_bloom_filter()` to reconstruct
it.

**Why AtomicU64 blocks instead of a BitVec?** AtomicU64 allows lock-free
concurrent access. A `BitVec` behind a `RwLock` would serialize all insertions
and create contention on the hot path.

---

## Gotchas

- Bloom filter state is **not persisted** in snapshots; rebuild after load.
- Thread-safe via AtomicU64 with Relaxed ordering (eventual consistency).
- Cannot remove items (use a counting bloom filter for that case).
- False positive rate increases if more items than `n` are inserted. If the
  actual item count significantly exceeds the configured expected count,
  consider rebuilding with a larger `n`.
