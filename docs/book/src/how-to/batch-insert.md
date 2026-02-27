# Batch Insert

This guide covers how to use `batch_insert` for efficient bulk loading in the
relational engine.

For the full API reference, see the
[Relational Engine API Reference](../reference/api/relational-engine.md). For
single-row inserts and other CRUD operations, see
[Create and Manage Tables](create-manage-tables.md).

## Why Use Batch Insert?

`batch_insert` performs a single schema lookup for all rows, compared to one
lookup per row with individual `insert` calls. This makes it 59x faster for
bulk loading.

## Usage

1. Build a `Vec<HashMap<String, Value>>` with the rows to insert.
2. Call `batch_insert` with the table name and rows.

```rust
let rows: Vec<HashMap<String, Value>> = (0..1000)
    .map(|i| {
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String(format!("User{}", i)));
        values.insert("age".to_string(), Value::Int(20 + i));
        values
    })
    .collect();

let row_ids = engine.batch_insert("users", rows)?;
```

The return value is a `Vec<u64>` of the assigned row IDs.

## Atomicity

`batch_insert` validates ALL rows upfront before inserting any. If any row
fails validation (type mismatch, null in non-nullable column, constraint
violation), the entire batch is rejected and no rows are inserted.

```rust
let rows = vec![valid_row, invalid_row];
// Fails on validation -- NO rows inserted (not a partial insert)
engine.batch_insert("users", rows);
```

## Comparison with Individual Inserts

```rust
// Slow: 1000 individual inserts (1000 schema lookups)
for row in rows {
    engine.insert("table", row)?;
}

// Fast: Single batch insert (1 schema lookup, 59x faster)
engine.batch_insert("table", rows)?;
```
