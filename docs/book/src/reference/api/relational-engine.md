# Relational Engine API Reference

> **See Also:**
> - [Columnar Architecture and SIMD Filtering](../../explanation/columnar-simd.md) -- design rationale
> - [Hash Join Algorithm](../../explanation/hash-join.md) -- join internals
> - [B-Tree Index Design](../../explanation/btree-index.md) -- index implementation details
> - [Create and Manage Tables](../../how-to/create-manage-tables.md) -- table CRUD how-to
> - [Indexes](../../how-to/indexes.md) -- index creation and usage
> - [Columnar SIMD Queries](../../how-to/columnar-simd.md) -- columnar select how-to
> - [Streaming Cursors](../../how-to/streaming-cursors.md) -- cursor lifecycle how-to
> - [Constraints](../../how-to/constraints.md) -- constraint configuration how-to
> - [Batch Insert](../../how-to/batch-insert.md) -- bulk loading how-to

## Key Types

| Type | Description |
| --- | --- |
| `RelationalEngine` | Main engine struct with TensorStore backend |
| `RelationalConfig` | Configuration for limits, timeouts, thresholds |
| `Schema` | Table schema with column definitions and constraints |
| `Column` | Column name, type, and nullability |
| `ColumnType` | `Int`, `Float`, `String`, `Bool`, `Bytes`, `Json` |
| `Value` | Typed value: `Null`, `Int(i64)`, `Float(f64)`, `String(String)`, `Bool(bool)`, `Bytes(Vec<u8>)`, `Json(Value)` |
| `Row` | Row with ID and ordered column values |
| `Condition` | Composable filter predicate tree |
| `Constraint` | Table constraint: `PrimaryKey`, `Unique`, `ForeignKey`, `NotNull` |
| `ForeignKeyConstraint` | Foreign key definition with referential actions |
| `ReferentialAction` | `Restrict`, `Cascade`, `SetNull`, `SetDefault`, `NoAction` |
| `RelationalError` | Error variants for table/column/index/constraint operations |
| `ColumnData` | Columnar storage for a single column with null bitmap |
| `SelectionVector` | Bitmap-based row selection for SIMD operations |
| `OrderedKey` | B-tree index key with total ordering semantics |
| `StreamingCursor` | Iterator for batch-based query result streaming |
| `CursorBuilder` | Builder for customizing streaming cursor options |
| `QueryMetrics` | Query execution metrics for observability |
| `IndexTracker` | Tracks index hits/misses to detect missing indexes |

## Column Types

| Type | Rust Type | Storage Format | Description |
| --- | --- | --- | --- |
| `Int` | `i64` | 8-byte little-endian | 64-bit signed integer |
| `Float` | `f64` | 8-byte IEEE 754 | 64-bit floating point |
| `String` | `String` | Dictionary-encoded | UTF-8 string with deduplication |
| `Bool` | `bool` | Packed bitmap (64 values per u64) | Boolean |
| `Bytes` | `Vec<u8>` | Raw bytes | Binary data |
| `Json` | `serde_json::Value` | JSON string | JSON value |

## Conditions

| Condition | Description | Index Support |
| --- | --- | --- |
| `Condition::True` | Matches all rows | N/A |
| `Condition::Eq(col, val)` | Column equals value | Hash Index |
| `Condition::Ne(col, val)` | Column not equals value | None |
| `Condition::Lt(col, val)` | Column less than value | B-Tree Index |
| `Condition::Le(col, val)` | Column less than or equal | B-Tree Index |
| `Condition::Gt(col, val)` | Column greater than value | B-Tree Index |
| `Condition::Ge(col, val)` | Column greater than or equal | B-Tree Index |
| `Condition::And(a, b)` | Logical AND of two conditions | Partial (first indexable) |
| `Condition::Or(a, b)` | Logical OR of two conditions | None |

Conditions can be combined using `.and()` and `.or()` methods:

```rust
// age >= 18 AND age < 65
let condition = Condition::Ge("age".into(), Value::Int(18))
    .and(Condition::Lt("age".into(), Value::Int(65)));

// status = 'active' OR priority > 5
let condition = Condition::Eq("status".into(), Value::String("active".into()))
    .or(Condition::Gt("priority".into(), Value::Int(5)));
```

The special column `_id` filters by row ID and can be indexed.

## Error Types

| Error | Cause |
| --- | --- |
| `TableNotFound` | Table does not exist |
| `TableAlreadyExists` | Creating duplicate table |
| `ColumnNotFound` | Update references unknown column |
| `ColumnAlreadyExists` | Column already exists in table |
| `TypeMismatch` | Value type does not match column type |
| `NullNotAllowed` | NULL in non-nullable column |
| `IndexAlreadyExists` | Creating duplicate index |
| `IndexNotFound` | Dropping non-existent index |
| `IndexCorrupted` | Index data is corrupted |
| `StorageError` | Underlying Tensor Store error |
| `InvalidName` | Invalid table or column name |
| `SchemaCorrupted` | Schema metadata is corrupted |
| `TransactionNotFound` | Transaction ID not found |
| `TransactionInactive` | Transaction already committed/aborted |
| `LockConflict` | Lock conflict with another transaction |
| `LockTimeout` | Lock acquisition timed out |
| `RollbackFailed` | Rollback operation failed |
| `ResultTooLarge` | Result set exceeds maximum size |
| `TooManyTables` | Maximum table count exceeded |
| `TooManyIndexes` | Maximum index count exceeded |
| `QueryTimeout` | Query execution timed out |
| `PrimaryKeyViolation` | Primary key constraint violated |
| `UniqueViolation` | Unique constraint violated |
| `ForeignKeyViolation` | Foreign key constraint violated on insert/update |
| `ForeignKeyRestrict` | Foreign key prevents delete/update |
| `ConstraintNotFound` | Constraint does not exist |
| `ConstraintAlreadyExists` | Constraint already exists |
| `ColumnHasConstraint` | Column has constraint preventing operation |
| `CannotAddColumn` | Cannot add column due to constraint |

## Storage Key Patterns

Tables, rows, and indexes are stored in Tensor Store with specific key patterns:

| Key Pattern | Content |
| --- | --- |
| `_meta:table:{name}` | Schema metadata |
| `{table}:{row_id}` | Row data |
| `_idx:{table}:{column}` | Hash index metadata |
| `_idx:{table}:{column}:{hash}` | Hash index entries (list of row IDs) |
| `_btree:{table}:{column}` | B-tree index metadata |
| `_btree:{table}:{column}:{sortable_key}` | B-tree index entries |
| `_col:{table}:{column}:data` | Columnar data storage |
| `_col:{table}:{column}:ids` | Columnar row ID mapping |
| `_col:{table}:{column}:nulls` | Columnar null bitmap |
| `_col:{table}:{column}:meta` | Columnar metadata |

Schema metadata encodes:

- `_columns`: Comma-separated column names
- `_col:{name}`: Type and nullability for each column

### Row Storage Format

Each row is stored as a `TensorData` object:

```rust
// Internal row structure
{
    "_id": Scalar(Int(row_id)),
    "name": Scalar(String("Alice")),
    "age": Scalar(Int(30)),
    "email": Scalar(String("alice@example.com"))
}
```

## Constraint Types

| Constraint | Description |
| --- | --- |
| `PrimaryKey` | Unique + not null, identifies rows uniquely |
| `Unique` | Values must be unique (NULLs allowed) |
| `ForeignKey` | References rows in another table |
| `NotNull` | Column cannot contain NULL values |

### Referential Actions

Foreign keys support these actions on delete/update of referenced rows:

| Action | Description |
| --- | --- |
| `Restrict` (default) | Prevent the operation |
| `Cascade` | Cascade to referencing rows |
| `SetNull` | Set referencing columns to NULL |
| `SetDefault` | Set referencing columns to default |
| `NoAction` | Same as Restrict, checked at commit |

## Condition Evaluation Methods

| Method | Input | Performance | Use Case |
| --- | --- | --- | --- |
| `evaluate(&row)` | Row struct | Legacy, creates intermediate objects | Row-by-row filtering |
| `evaluate_tensor(&tensor)` | TensorData | 31% faster, no intermediate allocation | Direct tensor filtering |

## SIMD Filter Functions

| Function | Operation | Types |
| --- | --- | --- |
| `filter_lt_i64` | Less than | i64 |
| `filter_le_i64` | Less than or equal | i64 |
| `filter_gt_i64` | Greater than | i64 |
| `filter_ge_i64` | Greater than or equal | i64 |
| `filter_eq_i64` | Equal | i64 |
| `filter_ne_i64` | Not equal | i64 |
| `filter_lt_f64` | Less than | f64 |
| `filter_gt_f64` | Greater than | f64 |
| `filter_eq_f64` | Equal (with epsilon) | f64 |

### Bitmap Operations

```rust
// AND two selection bitmaps
pub fn bitmap_and(a: &[u64], b: &[u64], result: &mut [u64])

// OR two selection bitmaps
pub fn bitmap_or(a: &[u64], b: &[u64], result: &mut [u64])

// Count set bits
pub fn popcount(bitmap: &[u64]) -> usize

// Extract selected indices
pub fn selected_indices(bitmap: &[u64], max_count: usize) -> Vec<usize>
```

## Cursor Methods

| Method | Description |
| --- | --- |
| `with_batch_size(n)` | Set rows fetched per batch (default: 1000) |
| `with_max_rows(n)` | Limit total rows returned |
| `rows_yielded()` | Number of rows returned so far |
| `is_exhausted()` | Whether cursor has no more rows |

## Hash Index Value Encoding

| Value Type | Hash Format | Example |
| --- | --- | --- |
| `Null` | `"null"` | `"null"` |
| `Int(i)` | `"i:{value}"` | `"i:42"` |
| `Float(f)` | `"f:{bits}"` | `"f:4614253070214989087"` |
| `String(s)` | `"s:{hash}"` | `"s:a1b2c3d4"` |
| `Bool(b)` | `"b:{value}"` | `"b:true"` |

## B-Tree Sortable Key Encoding

| Type | Encoding | Example |
| --- | --- | --- |
| `Null` | `"0"` | `"0"` |
| `Int(i)` | `"i{hex(i + 2^63)}"` | `"i8000000000000000"` for 0 |
| `Float(f)` | `"f{sortable_bits}"` | IEEE 754 with sign handling |
| `String(s)` | `"s{s}"` | `"sAlice"` |
| `Bool(b)` | `"b0"` or `"b1"` | `"b1"` for true |

## Performance Characteristics

| Operation | Complexity | Notes |
| --- | --- | --- |
| `insert` | O(1) + O(k) | Schema validation + store put + k index updates |
| `batch_insert` | O(n) + O(n*k) | Single schema lookup, 59x faster than n inserts |
| `select` (no index) | O(n) | Full table scan with SIMD filter |
| `select` (hash index) | O(1) | Direct lookup via hash index |
| `select` (btree range) | O(log n + m) | B-tree lookup + m matching rows |
| `update` | O(n) + O(k) | Scan + conditional update + index maintenance |
| `delete_rows` | O(n) + O(k) | Scan + conditional delete + index removal |
| `join` | O(n+m) | Hash join for all 6 join types |
| `cross_join` | O(n*m) | Cartesian product |
| `count/sum/avg/min/max` | O(n) | Single pass over matching rows |
| `create_index` | O(n) | Scan all rows to build index |
| `materialize_columns` | O(n) | Extract column to contiguous array |

Where k = number of indexes on the table, n = rows in left table, m = rows in
right table.

## Configuration

### RelationalConfig

| Option | Default | Description |
| --- | --- | --- |
| `max_tables` | None (unlimited) | Maximum number of tables |
| `max_indexes_per_table` | None (unlimited) | Maximum indexes per table |
| `max_btree_entries` | 10,000,000 | Maximum B-tree index entries total |
| `default_query_timeout_ms` | None | Default timeout for queries |
| `max_query_timeout_ms` | 300,000 (5 min) | Maximum allowed query timeout |
| `slow_query_threshold_ms` | 100 | Threshold for slow query warnings |
| `max_query_result_rows` | None (unlimited) | Maximum rows returned per query |
| `transaction_timeout_secs` | 60 | Transaction timeout |
| `lock_timeout_secs` | 30 | Lock acquisition timeout |

### Internal Constants

| Constant | Value | Description |
| --- | --- | --- |
| `PARALLEL_THRESHOLD` | 1000 | Minimum rows for parallel operations |
| Null bitmap sparse threshold | 10% | Use sparse bitmap when nulls < 10% |
| SIMD vector width | 4 | i64x4/f64x4 operations |

## Observability

### Query Metrics

```rust
use relational_engine::observability::{QueryMetrics, check_slow_query};
use std::time::Duration;

let metrics = QueryMetrics::new("users", "select")
    .with_rows_scanned(10000)
    .with_rows_returned(50)
    .with_index("idx_user_id")
    .with_duration(Duration::from_millis(25));

// Log warning if query exceeds threshold
check_slow_query(&metrics, 100); // threshold in ms
```

### Index Tracking

```rust
use relational_engine::observability::IndexTracker;

let tracker = IndexTracker::new();

// Record when index is used
tracker.record_hit("users", "id");

// Record when index could have been used but wasn't
tracker.record_miss("users", "email");

// Get reports of columns needing indexes
let reports = tracker.report_misses();
for report in reports {
    println!(
        "Table {}, column {}: {} misses, {} hits",
        report.table, report.column, report.miss_count, report.hit_count
    );
}

// Aggregate statistics
let total_hits = tracker.total_hits();
let total_misses = tracker.total_misses();
```

### Slow Query Warnings

```rust
use relational_engine::observability::{check_slow_query, warn_full_table_scan};

// Warn if query took > 100ms
check_slow_query(&metrics, 100);

// Warn about full table scans on large tables (> 1000 rows)
warn_full_table_scan("users", "select", 5000);
```

## Feature Summary

### Implemented

| Feature | Description |
| --- | --- |
| Hash indexes | O(1) equality lookups |
| B-tree indexes | O(log n) range query acceleration |
| All 6 JOIN types | INNER, LEFT, RIGHT, FULL, CROSS, NATURAL |
| Aggregate functions | COUNT, SUM, AVG, MIN, MAX |
| ORDER BY | Multi-column sorting with ASC/DESC, NULLS FIRST/LAST |
| LIMIT/OFFSET | Pagination support |
| GROUP BY + HAVING | Row grouping with aggregate filtering |
| Columnar storage | SIMD-accelerated filtering with selection vectors |
| Batch operations | 59x faster bulk inserts |
| Parallel operations | Rayon-based parallelism for large tables |
| Dictionary encoding | String column compression |
| Transactions | Row-level ACID with undo log -- see [Transactions](./relational-engine.md) |
| Constraints | PRIMARY KEY, UNIQUE, FOREIGN KEY, NOT NULL |
| Foreign Keys | Full referential integrity with CASCADE/SET NULL/RESTRICT |
| ALTER TABLE | add_column, drop_column, rename_column |
| Streaming cursors | Memory-efficient iteration over large result sets |
| Observability | Query metrics, slow query detection, index tracking |

### Future Considerations

| Feature | Status |
| --- | --- |
| Query Optimization | Not implemented |
| Subqueries | Not implemented |
| Window Functions | Not implemented |
| Composite Indexes | Not implemented |

## Related Modules

| Module | Relationship |
| --- | --- |
| `tensor_store` | Storage backend for tables, rows, and indexes |
| `query_router` | Executes SQL queries using RelationalEngine |
| `neumann_parser` | Parses SQL statements into AST |
| `tensor_unified` | Multi-engine unified storage layer |

## Edge Cases and Gotchas

### NULL Handling

1. **NULL in conditions**: Comparisons with NULL columns return `false`:

   ```rust
   // If email is NULL, this returns false (not true!)
   Condition::Lt("email".into(), Value::String("z".into()))
   ```

2. **NULL in joins**: NULL values never match in join conditions:

   ```rust
   // Post with user_id = NULL will not join with any user
   engine.join("users", "posts", "_id", "user_id")
   ```

3. **COUNT vs COUNT(column)**:
   - `count()` counts all rows
   - `count_column()` counts non-null values only

### Type Mismatches

Comparisons between incompatible types return `false` rather than error:

```rust
// Age is Int, comparing with String returns 0 matches (not error)
engine.select("users", Condition::Lt("age".into(), Value::String("30".into())));
```

### Index Maintenance

Indexes are automatically maintained on INSERT, UPDATE, and DELETE:

```rust
// Creating index AFTER data exists
engine.insert("users", values)?;  // No index update
engine.create_index("users", "age")?;  // Scans all rows

// Creating index BEFORE data exists
engine.create_index("users", "age")?;  // Empty index
engine.insert("users", values)?;  // Updates index
```

### Batch Insert Atomicity

`batch_insert` validates ALL rows upfront before inserting any:

```rust
let rows = vec![valid_row, invalid_row];
// Fails on validation - NO rows inserted (not partial insert)
engine.batch_insert("users", rows);
```

### B-Tree Index Recovery

B-tree indexes maintain both in-memory and persistent state. The in-memory
BTreeMap is rebuilt lazily on first access after restart.
