# Indexes

This guide covers creating, using, and dropping hash and B-tree indexes in the
relational engine, including guidance on which index type to choose for
different query patterns.

For the design rationale behind B-tree indexes, see
[B-Tree Index Design](../explanation/btree-index.md). For the full API
reference, see the
[Relational Engine API Reference](../reference/api/relational-engine.md).

## Hash Indexes

Hash indexes provide O(1) equality lookups for `Condition::Eq` queries.

### Create a Hash Index

```rust
engine.create_index("users", "age")?;
```

### Check and List Hash Indexes

```rust
// Check if a hash index exists
engine.has_index("users", "age");  // -> bool

// Get all hash-indexed columns for a table
engine.get_indexed_columns("users");  // -> Vec<String>
```

### Drop a Hash Index

```rust
engine.drop_index("users", "age")?;
```

### Performance

| Query Type | Without Index | With Index | Speedup |
| --- | --- | --- | --- |
| Equality (2% match on 5K rows) | 5.96ms | 126us | 47x |
| Single row by _id (5K rows) | 5.59ms | 3.5us | 1,597x |

## B-Tree Indexes

B-tree indexes accelerate range queries (`Lt`, `Le`, `Gt`, `Ge`) with O(log n
+ m) complexity, where n is the number of indexed values and m is the number
of matching rows.

### Create a B-Tree Index

```rust
engine.create_btree_index("users", "age")?;
```

### Check and List B-Tree Indexes

```rust
// Check if a B-tree index exists
engine.has_btree_index("users", "age");  // -> bool

// Get all B-tree indexed columns for a table
engine.get_btree_indexed_columns("users");  // -> Vec<String>
```

### Drop a B-Tree Index

```rust
engine.drop_btree_index("users", "age")?;
```

### Use a B-Tree Index for Range Queries

Once a B-tree index exists on a column, range conditions automatically use it:

```rust
// This query uses the B-tree index on "age"
engine.select("users", Condition::Ge("age".into(), Value::Int(18)))?;
```

## When to Use Which Index

| Query Pattern | Recommended Index |
| --- | --- |
| `WHERE col = value` | Hash Index |
| `WHERE col > value` | B-Tree Index |
| `WHERE col BETWEEN a AND b` | B-Tree Index |
| `WHERE col IN (...)` | Hash Index |
| Unique lookups by ID | Hash Index on `_id` |
| Ordered scans | B-Tree Index |

**Rules of thumb:**

1. If you only ever test equality on a column, use a hash index -- it is O(1)
   per lookup versus O(log n) for a B-tree.
2. If you need range queries, ordered iteration, or a mix of equality and range,
   use a B-tree index.
3. You can have both a hash index and a B-tree index on the same column. The
   engine will choose the best one for each query.

## Index Maintenance

Indexes are maintained automatically on INSERT, UPDATE, and DELETE. You do not
need to rebuild indexes after data changes.

```rust
// Creating index AFTER data exists scans all rows once
engine.insert("users", values)?;
engine.create_index("users", "age")?;  // Scans all existing rows

// Creating index BEFORE data exists starts empty
engine.create_index("users", "age")?;
engine.insert("users", values)?;  // Index updated on insert
```

## Index Tracking

Use `IndexTracker` to identify columns that would benefit from indexing:

```rust
use relational_engine::observability::IndexTracker;

let tracker = IndexTracker::new();
tracker.record_hit("users", "id");
tracker.record_miss("users", "email");

let reports = tracker.report_misses();
for report in reports {
    println!(
        "Table {}, column {}: {} misses, {} hits",
        report.table, report.column, report.miss_count, report.hit_count
    );
}
```
