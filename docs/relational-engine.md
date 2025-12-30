# Relational Engine

Module 2 of Neumann. Provides SQL-like operations on top of the Tensor Store.

## Design Principles

1. **Layered Architecture**: Depends only on Tensor Store for persistence
2. **Schema Enforcement**: Type checking on insert/update
3. **Condition Evaluation**: Composable predicates for filtering
4. **Thread Safety**: Inherits from Tensor Store
5. **Serializable Types**: All types implement `serde::Serialize`/`Deserialize`

## Data Model

### Schema

Tables are defined by a schema specifying columns and their types:

```rust
let schema = Schema::new(vec![
    Column::new("name", ColumnType::String),
    Column::new("age", ColumnType::Int),
    Column::new("email", ColumnType::String).nullable(),
]);

engine.create_table("users", schema)?;
```

### Column Types

| Type | Rust Type | Description |
|------|-----------|-------------|
| `Int` | `i64` | 64-bit signed integer |
| `Float` | `f64` | 64-bit floating point |
| `String` | `String` | UTF-8 string |
| `Bool` | `bool` | Boolean |

### Values

```rust
Value::Null           // NULL value
Value::Int(42)        // Integer
Value::Float(3.14)    // Float
Value::String("x")    // String
Value::Bool(true)     // Boolean
```

## API Reference

### Table Operations

```rust
let engine = RelationalEngine::new();

// Create table
engine.create_table("users", schema)?;

// Check existence
engine.table_exists("users")?;  // -> bool

// Drop table (deletes all rows)
engine.drop_table("users")?;

// Row count
engine.row_count("users")?;  // -> usize
```

### CRUD Operations

```rust
// INSERT
let mut values = HashMap::new();
values.insert("name".to_string(), Value::String("Alice".into()));
values.insert("age".to_string(), Value::Int(30));
let row_id = engine.insert("users", values)?;

// BATCH INSERT (59x faster for bulk inserts)
let rows: Vec<HashMap<String, Value>> = (0..1000)
    .map(|i| {
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String(format!("User{}", i)));
        values.insert("age".to_string(), Value::Int(20 + i));
        values
    })
    .collect();
let row_ids = engine.batch_insert("users", rows)?;

// SELECT
let rows = engine.select("users", Condition::Eq("age".into(), Value::Int(30)))?;

// UPDATE
let mut updates = HashMap::new();
updates.insert("age".to_string(), Value::Int(31));
let count = engine.update("users", Condition::Eq("name".into(), Value::String("Alice".into())), updates)?;

// DELETE
let count = engine.delete_rows("users", Condition::Lt("age".into(), Value::Int(18)))?;
```

### Conditions

Conditions are composable predicates for filtering rows:

| Condition | Description |
|-----------|-------------|
| `Condition::True` | Matches all rows |
| `Condition::Eq(col, val)` | Column equals value |
| `Condition::Ne(col, val)` | Column not equals value |
| `Condition::Lt(col, val)` | Column less than value |
| `Condition::Le(col, val)` | Column less than or equal |
| `Condition::Gt(col, val)` | Column greater than value |
| `Condition::Ge(col, val)` | Column greater than or equal |

#### Evaluation Methods

Conditions support two evaluation methods:

```rust
// Row-based evaluation (legacy)
let result = condition.evaluate(&row);

// TensorData-based evaluation (tensor-native, 31% faster)
let result = condition.evaluate_tensor(&tensor);
```

The `evaluate_tensor()` method evaluates conditions directly on `TensorData` without creating intermediate `Row` objects. This provides ~31% speedup on select operations by avoiding HashMap allocation for rows that don't match the condition.

Conditions can be combined:

```rust
// age >= 18 AND age < 65
let condition = Condition::Ge("age".into(), Value::Int(18))
    .and(Condition::Lt("age".into(), Value::Int(65)));

// status = "active" OR status = "pending"
let condition = Condition::Eq("status".into(), Value::String("active".into()))
    .or(Condition::Eq("status".into(), Value::String("pending".into())));
```

The special column `_id` can be used to filter by row ID:

```rust
engine.select("users", Condition::Eq("_id".into(), Value::Int(5)))?;
```

### Joins

Join two tables on matching column values. All 6 SQL join types are supported:

```rust
// INNER JOIN - rows that match in both tables
let joined = engine.join("users", "posts", "_id", "user_id")?;
for (user, post) in joined {
    println!("User {} wrote: {}", user.id, post.get("title"));
}

// LEFT JOIN - all left rows, matched right rows (or None)
let joined = engine.left_join("users", "posts", "_id", "user_id")?;
for (user, post) in joined {
    match post {
        Some(p) => println!("User {} wrote: {}", user.id, p.get("title")),
        None => println!("User {} has no posts", user.id),
    }
}

// RIGHT JOIN - matched left rows (or None), all right rows
let joined = engine.right_join("users", "posts", "_id", "user_id")?;

// FULL JOIN - all rows from both tables, matched where possible
let joined = engine.full_join("users", "posts", "_id", "user_id")?;

// CROSS JOIN - Cartesian product of both tables
let joined = engine.cross_join("users", "posts")?;

// NATURAL JOIN - automatic join on common column names
let joined = engine.natural_join("users", "user_profiles")?;
```

All joins use hash join algorithm: O(n+m) instead of O(n*m) nested loop.

### Aggregate Functions

Aggregate functions compute summary values over rows:

```rust
// COUNT(*) - count all rows
let count = engine.count("users", Condition::True)?;

// COUNT(column) - count non-null values
let count = engine.count_column("users", "email", Condition::True)?;

// SUM - sum numeric column
let total = engine.sum("orders", "amount", Condition::Eq("status".into(), Value::String("paid".into())))?;

// AVG - average of numeric column
let avg = engine.avg("orders", "amount", Condition::True)?;  // -> Option<f64>

// MIN/MAX - minimum/maximum value
let min = engine.min("products", "price", Condition::True)?;  // -> Option<Value>
let max = engine.max("products", "price", Condition::True)?;
```

Aggregates support any condition for filtering rows before computation.

### Indexes

Hash indexes accelerate equality lookups (`Condition::Eq`) by providing O(1) access instead of O(n) table scans.

```rust
// Create an index on a column
engine.create_index("users", "age")?;

// Create an index on _id for fast primary key lookups
engine.create_index("users", "_id")?;

// Check if index exists
engine.has_index("users", "age");  // -> bool

// Get all indexed columns for a table
engine.get_indexed_columns("users");  // -> Vec<String>

// Drop an index
engine.drop_index("users", "age")?;
```

**Hash index behavior:**
- Hash indexes are automatically maintained on insert, update, and delete
- Hash indexes are automatically cleaned up when a table is dropped
- Only `Condition::Eq` queries benefit from hash indexes

**Hash index performance:**
| Query Type | Without Index | With Index | Speedup |
|------------|---------------|------------|---------|
| Equality (2% match on 5K rows) | 5.96ms | 126µs | 47x |
| Single row by _id (5K rows) | 5.59ms | 3.5µs | 1,597x |

### B-Tree Indexes

B-tree indexes accelerate range queries (`Lt`, `Le`, `Gt`, `Ge`) by storing values in sorted order.

```rust
// Create a B-tree index on a column
engine.create_btree_index("users", "age")?;

// Check if B-tree index exists
engine.has_btree_index("users", "age");  // -> bool

// Drop a B-tree index
engine.drop_btree_index("users", "age")?;

// Range queries now use the index
engine.select("users", Condition::Ge("age".into(), Value::Int(18)))?;
engine.select("users", Condition::Lt("age".into(), Value::Int(65)))?;
```

**B-tree index behavior:**
- B-tree indexes are automatically maintained on insert, update, and delete
- B-tree indexes are automatically cleaned up when a table is dropped
- Range conditions (`Lt`, `Le`, `Gt`, `Ge`) benefit from B-tree indexes
- Uses sortable key encoding for correct ordering of integers, floats, and strings

## Columnar Architecture

The relational engine uses a columnar storage model optimized for analytical queries:

### SIMD-Accelerated Filtering

Column data is stored in contiguous arrays enabling SIMD vectorized comparisons:

```rust
// Internal: 8-wide SIMD comparison for i64 columns
fn filter_gt_i64(values: &[i64], threshold: i64, bitmap: &mut [u64])
fn filter_ge_i64(values: &[i64], threshold: i64, bitmap: &mut [u64])
fn filter_lt_i64(values: &[i64], threshold: i64, bitmap: &mut [u64])
fn filter_le_i64(values: &[i64], threshold: i64, bitmap: &mut [u64])
fn filter_eq_i64(values: &[i64], threshold: i64, bitmap: &mut [u64])
fn filter_ne_i64(values: &[i64], threshold: i64, bitmap: &mut [u64])
```

### Selection Vectors

Query results use selection vectors to avoid copying data:

```rust
// SelectionVector tracks which rows match without materializing
let selection = engine.select_columnar("users", condition, options)?;

// Only selected rows are returned
for row in selection {
    // Row data is lazily extracted
}
```

### Column Data Types

```rust
pub enum ColumnData {
    Int(Vec<i64>),
    Float(Vec<f64>),
    String(Vec<String>),
    Bool(Vec<bool>),
}
```

Each column type supports null bitmaps for efficient NULL handling.

## Storage Model

Tables, rows, and indexes are stored in Tensor Store:

| Key Pattern | Content |
|-------------|---------|
| `_meta:table:{name}` | Schema metadata |
| `{table}:{row_id}` | Row data |
| `_idx:{table}:{column}` | Hash index metadata |
| `_idx:{table}:{column}:{hash}` | Hash index entries (list of row IDs) |
| `_btree:{table}:{column}` | B-tree index metadata |
| `_btree:{table}:{column}:{sortable_key}` | B-tree index entries (list of row IDs) |

Schema is encoded in the metadata tensor:
- `_columns`: Comma-separated column names
- `_col:{name}`: Type and nullability for each column

Index entries map value hashes to lists of row IDs, enabling O(1) lookup.

## Error Handling

| Error | Cause |
|-------|-------|
| `TableNotFound` | Table doesn't exist |
| `TableAlreadyExists` | Creating duplicate table |
| `ColumnNotFound` | Update references unknown column |
| `TypeMismatch` | Value type doesn't match column type |
| `NullNotAllowed` | NULL in non-nullable column |
| `IndexAlreadyExists` | Creating duplicate index |
| `IndexNotFound` | Dropping non-existent index |
| `StorageError` | Underlying Tensor Store error |

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `insert` | O(1) + O(k) | Schema validation + store put + k index updates |
| `batch_insert` | O(n) + O(n*k) | Single schema lookup, 59x faster than n inserts |
| `select` (no index) | O(n) | Full table scan with SIMD filter |
| `select` (hash index) | O(1) | Direct lookup via hash index |
| `select` (btree range) | O(log n + m) | B-tree lookup + m matching rows |
| `update` | O(n) + O(k) | Scan + conditional update + index maintenance |
| `delete_rows` | O(n) + O(k) | Scan + conditional delete + index removal |
| `join` | O(n+m) | Hash join for all 6 join types |
| `left/right/full_join` | O(n+m) | Hash join with null padding |
| `cross_join` | O(n*m) | Cartesian product |
| `count/sum/avg/min/max` | O(n) | Single pass over matching rows |
| `create_index` | O(n) | Scan all rows to build index |

Where k = number of indexes on the table, n = rows in left table, m = rows in right table.

## Test Coverage

| Test | What It Verifies |
|------|------------------|
| `create_table_and_insert` | Basic table/row creation |
| `insert_1000_rows_select_with_condition` | Scale + filtering |
| `select_with_range_condition` | Range queries (>=, <) |
| `select_with_compound_condition` | AND/OR conditions |
| `join_two_tables` | Inner join correctness |
| `update_modifies_correct_rows` | Conditional updates |
| `delete_removes_correct_rows` | Conditional deletes |
| `create_and_use_index` | Index creation and query acceleration |
| `index_maintained_on_insert` | Index updated on new rows |
| `index_maintained_on_update` | Index updated on row changes |
| `index_maintained_on_delete` | Index entries removed on delete |
| `drop_table_cleans_up_indexes` | Indexes cleaned up with table |
| `btree_index_accelerates_range_queries` | B-tree index speeds up Lt/Le/Gt/Ge |
| `btree_index_maintained_on_insert` | B-tree updated on new rows |
| `btree_index_maintained_on_update` | B-tree updated on row changes |
| `btree_index_maintained_on_delete` | B-tree entries removed on delete |
| `btree_index_handles_negative_numbers` | Correct ordering of negative ints |
| `btree_index_handles_floats` | Correct ordering of floats |
| `btree_index_handles_strings` | Correct ordering of strings |
| `drop_btree_index` | B-tree index removal |
| `batch_insert_multiple_rows` | Bulk insert functionality |
| `batch_insert_empty` | Empty batch handling |
| `batch_insert_validates_all_rows_upfront` | Fail-fast validation |
| `batch_insert_with_indexes` | Index maintenance during batch |
| `batch_insert_with_btree_index` | B-tree index maintenance during batch |
| `batch_insert_null_not_allowed` | Null validation in batch |
| `evaluate_tensor_*` | TensorData-direct evaluation consistency |

### Integration Tests

| Test | What It Verifies |
|------|------------------|
| `test_evaluate_consistency_eq` | Eq condition consistency between evaluate() and evaluate_tensor() |
| `test_evaluate_consistency_ne` | Ne condition consistency |
| `test_evaluate_consistency_comparisons` | Lt/Le/Gt/Ge condition consistency |
| `test_evaluate_consistency_logical` | And/Or condition consistency |
| `test_evaluate_consistency_nested` | Complex nested condition consistency |
| `test_evaluate_consistency_missing_fields` | Missing field handling |
| `test_evaluate_consistency_null_values` | Null value handling |
| `test_evaluate_consistency_id_field` | _id field handling |
| `test_engine_select_correctness` | End-to-end select correctness |
| `test_evaluate_performance_improvement` | Performance verification |

### Fuzz Targets

| Target | What It Tests |
|--------|---------------|
| `relational_condition` | evaluate() vs evaluate_tensor() consistency with arbitrary inputs |
| `relational_engine_ops` | Engine CRUD operations with arbitrary inputs |

## SQL Features via Query Router

When using the relational engine through `query_router`, additional SQL features are available:

### ORDER BY and OFFSET

Sort results and skip rows for pagination:

```sql
-- Sort by single column
SELECT * FROM users ORDER BY age ASC;

-- Sort by multiple columns
SELECT * FROM users ORDER BY department DESC, name ASC;

-- Null handling
SELECT * FROM users ORDER BY email NULLS FIRST;

-- Pagination with LIMIT and OFFSET
SELECT * FROM users ORDER BY created_at DESC LIMIT 10 OFFSET 20;
```

### GROUP BY and HAVING

Group rows and filter groups:

```sql
-- Group by single column with aggregates
SELECT department, COUNT(*), AVG(salary) FROM employees GROUP BY department;

-- Filter groups with HAVING
SELECT product, SUM(quantity) as total
FROM orders
GROUP BY product
HAVING SUM(quantity) > 100;

-- Combine WHERE (row filter) with HAVING (group filter)
SELECT category, COUNT(*)
FROM products
WHERE active = true
GROUP BY category
HAVING COUNT(*) >= 5;
```

### All JOIN Types

```sql
-- INNER JOIN
SELECT u.name, o.amount FROM users u INNER JOIN orders o ON u.id = o.user_id;

-- LEFT/RIGHT/FULL OUTER JOIN
SELECT u.name, o.amount FROM users u LEFT JOIN orders o ON u.id = o.user_id;

-- CROSS JOIN (Cartesian product)
SELECT * FROM sizes CROSS JOIN colors;

-- NATURAL JOIN (on common columns)
SELECT * FROM users NATURAL JOIN profiles;
```

## Feature Summary

### Implemented

| Feature | Description |
|---------|-------------|
| Hash indexes | O(1) equality lookups |
| B-tree indexes | O(log n) range query acceleration |
| All 6 JOIN types | INNER, LEFT, RIGHT, FULL, CROSS, NATURAL |
| Aggregate functions | COUNT, SUM, AVG, MIN, MAX |
| ORDER BY | Multi-column sorting with ASC/DESC, NULLS FIRST/LAST |
| LIMIT/OFFSET | Pagination support |
| GROUP BY + HAVING | Row grouping with aggregate filtering |
| Columnar storage | SIMD-accelerated filtering with selection vectors |
| Batch operations | 59x faster bulk inserts |

### Future Considerations

Not yet implemented:

- **Query Optimization**: Cost-based query planning
- **Transactions**: ACID guarantees with rollback
- **Foreign Keys**: Referential integrity constraints
- **Subqueries**: Nested SELECT statements
- **Window Functions**: OVER(), PARTITION BY
