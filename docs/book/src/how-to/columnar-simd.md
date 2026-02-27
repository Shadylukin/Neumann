# Columnar SIMD Queries

This guide covers how to use the columnar storage and SIMD-accelerated filtering
in the relational engine, including materializing columns, running columnar
selects, and managing columnar data.

For the design rationale, see
[Columnar Architecture and SIMD Filtering](../explanation/columnar-simd.md).
For the full API reference, see the
[Relational Engine API Reference](../reference/api/relational-engine.md).

## When to Use Columnar Queries

Materialize columns when:

- Performing many range scans on large tables (thousands of rows or more).
- Query selectivity is low (scanning most rows rather than a small subset).
- The column data fits in memory (columnar storage is an additional copy).
- The column type is `Int` or `Float` (these types support SIMD filtering).

For small tables or highly selective queries (few matching rows), hash or B-tree
indexes are more efficient. See [Indexes](indexes.md).

## Step 1: Materialize Columns

Before columnar queries can use SIMD filtering, the target columns must be
materialized. This extracts column data from row storage into contiguous arrays.

```rust
engine.materialize_columns("events", &["timestamp", "user_id"])?;
```

You can materialize multiple columns at once. Each column is stored
independently.

## Step 2: Verify Materialization

Check whether columnar data exists for a column:

```rust
engine.has_columnar_data("events", "timestamp");  // -> bool
```

## Step 3: Run a Columnar Select

Use `select_columnar` with `ColumnarScanOptions` to query with SIMD
acceleration:

```rust
let options = ColumnarScanOptions {
    projection: Some(vec!["timestamp".into(), "user_id".into()]),
    prefer_columnar: true,
};

let rows = engine.select_columnar(
    "events",
    Condition::Gt("timestamp".into(), Value::Int(cutoff)),
    options
)?;
```

### Options

- **`projection`**: When set, only the listed columns are returned in the
  result rows. This reduces data transfer and materialization cost.
- **`prefer_columnar`**: When `true`, the engine uses SIMD filtering on
  materialized columns when available. When `false` or when the column is not
  materialized, it falls back to row-by-row evaluation.

## Step 4: Drop Columnar Data

When columnar data is no longer needed (e.g., after a schema change or to free
memory), drop it:

```rust
engine.drop_columnar_data("events", "timestamp")?;
```

## Complete Example

```rust
let engine = RelationalEngine::new();

// Create table and insert data
let schema = Schema::new(vec![
    Column::new("timestamp", ColumnType::Int),
    Column::new("user_id", ColumnType::Int),
    Column::new("action", ColumnType::String),
]);
engine.create_table("events", schema)?;

// ... insert many rows ...

// Materialize the columns used in filters
engine.materialize_columns("events", &["timestamp", "user_id"])?;

// Query with SIMD acceleration
let cutoff = 1_700_000_000;
let rows = engine.select_columnar(
    "events",
    Condition::Gt("timestamp".into(), Value::Int(cutoff)),
    ColumnarScanOptions {
        projection: Some(vec!["action".into()]),
        prefer_columnar: true,
    }
)?;

// Clean up when done
engine.drop_columnar_data("events", "timestamp")?;
engine.drop_columnar_data("events", "user_id")?;
```

## How the Engine Chooses an Evaluation Path

The engine automatically selects the best evaluation strategy for each query:

1. If the filtered column has materialized columnar data and the column type is
   `Int` or `Float`, the engine uses SIMD vectorized filtering.
2. Otherwise, the engine evaluates the condition row-by-row using
   `evaluate_tensor`, which is 31% faster than the legacy `evaluate` method.

You do not need to change your query logic -- setting `prefer_columnar: true`
is sufficient to enable SIMD when the data is available.
