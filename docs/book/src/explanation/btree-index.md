# B-Tree Index Design

The relational engine's B-tree index accelerates range queries (`Lt`, `Le`,
`Gt`, `Ge`) with O(log n + m) complexity, where n is the number of indexed
values and m is the number of matching rows. This page explains the
implementation details, the key encoding strategy for persistent storage, and
the design rationale behind the dual-storage approach.

For the API reference, see the
[Relational Engine API Reference](../reference/api/relational-engine.md). For
step-by-step index usage, see [Indexes](../how-to/indexes.md).

## Dual-Storage Architecture

The B-tree index uses a dual-storage approach to balance query performance with
durability:

1. **In-memory `BTreeMap`**: Provides O(log n) range operations using Rust's
   standard library `BTreeMap`. This is the primary query path.
2. **Persistent TensorStore entries**: Provides durability and recovery. Index
   entries are written to the underlying store using sortable key encoding.

```rust
// Internal B-tree index structure
btree_indexes: RwLock<HashMap<
    (String, String),               // (table, column)
    BTreeMap<OrderedKey, Vec<u64>>  // value -> row_ids
>>
```

On restart, the in-memory `BTreeMap` is rebuilt lazily on first access by
scanning the persistent index entries from the TensorStore. This avoids a
potentially expensive startup cost for tables with many indexes.

## OrderedKey: Total Ordering for Values

A fundamental challenge in indexing heterogeneous values is establishing a total
ordering. Rust's `f64` type does not implement `Ord` because `NaN` values
break the required properties. The `OrderedKey` enum solves this:

```rust
pub enum OrderedKey {
    Null,                    // Sorts first
    Bool(bool),              // false < true
    Int(i64),                // Standard integer ordering
    Float(OrderedFloat),     // NaN < all other values
    String(String),          // Lexicographic ordering
}
```

Key ordering decisions:

- **Null sorts first**: This matches SQL semantics for `NULLS FIRST` ordering
  and ensures deterministic behavior in range scans.
- **NaN sorts before all finite floats**: This avoids the `NaN != NaN` problem
  by giving NaN a defined position in the ordering.
- **`OrderedFloat`** wraps `f64` with a total ordering implementation, enabling
  use as a `BTreeMap` key.

## Sortable Key Encoding

For persistent storage, values must be encoded as strings that maintain the
correct lexicographic ordering when sorted as raw bytes. This is necessary
because the TensorStore's key-value operations use string keys and B-tree
range scans rely on key ordering.

| Type | Encoding | Example |
| --- | --- | --- |
| `Null` | `"0"` | `"0"` |
| `Int(i)` | `"i{hex(i + 2^63)}"` | `"i8000000000000000"` for 0 |
| `Float(f)` | `"f{sortable_bits}"` | IEEE 754 with sign handling |
| `String(s)` | `"s{s}"` | `"sAlice"` |
| `Bool(b)` | `"b0"` or `"b1"` | `"b1"` for true |

### Integer Encoding Rationale

Signed integers in two's complement have negative values with the high bit set.
When encoded directly as hexadecimal, `-1` (all bits set) would sort after `0`
(all bits clear), which is incorrect.

The solution is to shift the range from `[-2^63, 2^63-1]` to `[0, 2^64-1]` by
adding `2^63` (equivalently, flipping the sign bit). After this
transformation:

- `i64::MIN` (-2^63) maps to `0x0000000000000000` -- sorts first
- `0` maps to `0x8000000000000000` -- sorts in the middle
- `i64::MAX` (2^63-1) maps to `0xFFFFFFFFFFFFFFFF` -- sorts last

The hexadecimal encoding with zero-padding ensures that the string comparison
matches the numeric ordering.

### Float Encoding Rationale

IEEE 754 floating-point numbers have a property where positive floats in their
bit representation already sort correctly as unsigned integers. For negative
floats, the bits must be inverted. The encoding handles three cases:

1. **Positive floats**: Use the raw bits directly (already sort correctly).
2. **Negative floats**: Invert all bits (makes larger negatives sort first).
3. **NaN**: Encoded as the smallest possible value to sort before all other
   floats.

## Range Operations

The in-memory `BTreeMap` supports efficient range queries through Rust's
`BTreeMap::range` method:

```rust
fn btree_range_lookup(&self, table: &str, column: &str,
                      value: &Value, op: RangeOp) -> Option<Vec<u64>> {
    match op {
        RangeOp::Lt => btree.range(..target),
        RangeOp::Le => btree.range(..=target),
        RangeOp::Gt => btree.range((Excluded(target), Unbounded)),
        RangeOp::Ge => btree.range(target..),
    }
}
```

Each range query returns an iterator over `(OrderedKey, Vec<u64>)` pairs. The
row IDs from all matching entries are collected into a single vector, which is
then used to fetch the actual rows from the store.

## Index Maintenance

B-tree indexes are maintained automatically on data mutations:

- **INSERT**: The new value is inserted into both the in-memory `BTreeMap` and
  the persistent store.
- **UPDATE**: The old value is removed and the new value is inserted in both
  stores. This is tracked via `IndexChange` entries in the transaction undo log.
- **DELETE**: The value is removed from both stores.
- **CREATE INDEX on existing data**: All rows are scanned once to populate both
  the in-memory and persistent index.

## Comparison with Hash Indexes

| Aspect | Hash Index | B-Tree Index |
| --- | --- | --- |
| Equality queries | O(1) | O(log n) |
| Range queries | Not supported | O(log n + m) |
| Ordered iteration | Not supported | Supported |
| Memory overhead | Hash buckets | Tree nodes |
| Persistent encoding | Value hash | Sortable key encoding |

Use hash indexes for equality-only workloads (e.g., lookups by ID or status).
Use B-tree indexes when range queries or ordered scans are needed (e.g.,
filtering by timestamp, price range, or age bracket).
