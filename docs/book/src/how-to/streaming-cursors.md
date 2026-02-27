# Streaming Cursors

This guide covers how to use streaming cursors for memory-efficient iteration
over large result sets in the relational engine.

For the full API reference, see the
[Relational Engine API Reference](../reference/api/relational-engine.md).

## When to Use Streaming Cursors

Use streaming cursors instead of `select` when:

- The result set is large and loading all rows into memory at once is
  impractical.
- You want to process rows incrementally (e.g., streaming to a client).
- You want to limit the total number of rows processed without fetching all
  matches first.

## Basic Usage

1. Create a streaming cursor with `select_streaming`.
2. Iterate over results using the standard `Iterator` trait.

```rust
use relational_engine::{StreamingCursor, Condition};

// Create streaming cursor with default batch size (1000)
let cursor = engine.select_streaming("users", Condition::True);

// Iterate over results
for row_result in cursor {
    let row = row_result?;
    println!("User: {:?}", row);
}
```

## Customize Batch Size and Row Limit

### Using Method Chaining

```rust
let cursor = engine.select_streaming("users", Condition::True)
    .with_batch_size(100)
    .with_max_rows(5000);
```

### Using the Builder

```rust
let cursor = engine.select_streaming_builder("users", Condition::True)
    .batch_size(100)
    .max_rows(5000)
    .build();
```

## Monitor Cursor Progress

```rust
let mut cursor = engine.select_streaming("users", Condition::True);
while let Some(row) = cursor.next() {
    println!("Yielded so far: {}", cursor.rows_yielded());
}
println!("Exhausted: {}", cursor.is_exhausted());
```

## Cursor Lifecycle

1. **Open**: The cursor is created via `select_streaming` or
   `select_streaming_builder`. No rows are fetched yet.
2. **Fetch**: On each call to `next()`, the cursor returns the next row from
   the current batch. When the batch is exhausted, the cursor fetches the
   next batch of rows from the engine.
3. **Close**: The cursor is exhausted when there are no more matching rows or
   the `max_rows` limit is reached. The cursor can also be dropped at any
   time to release resources.

## Cursor Methods

| Method | Description |
| --- | --- |
| `with_batch_size(n)` | Set rows fetched per batch (default: 1000) |
| `with_max_rows(n)` | Limit total rows returned |
| `rows_yielded()` | Number of rows returned so far |
| `is_exhausted()` | Whether cursor has no more rows |
