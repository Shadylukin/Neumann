# Relational Engine

Module 2 of Neumann. Provides SQL-like operations on top of the Tensor Store.

## Design Principles

1. **Layered Architecture**: Depends only on Tensor Store for persistence
2. **Schema Enforcement**: Type checking on insert/update
3. **Condition Evaluation**: Composable predicates for filtering
4. **Thread Safety**: Inherits from Tensor Store

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

Join two tables on matching column values:

```rust
// SELECT * FROM users JOIN posts ON users._id = posts.user_id
let joined = engine.join("users", "posts", "_id", "user_id")?;

for (user, post) in joined {
    println!("User {} wrote: {}", user.id, post.get("title"));
}
```

Currently implements inner join (rows must match in both tables).

## Storage Model

Tables and rows are stored in Tensor Store:

| Key Pattern | Content |
|-------------|---------|
| `_meta:table:{name}` | Schema metadata |
| `{table}:{row_id}` | Row data |

Schema is encoded in the metadata tensor:
- `_columns`: Comma-separated column names
- `_col:{name}`: Type and nullability for each column

## Error Handling

| Error | Cause |
|-------|-------|
| `TableNotFound` | Table doesn't exist |
| `TableAlreadyExists` | Creating duplicate table |
| `ColumnNotFound` | Update references unknown column |
| `TypeMismatch` | Value type doesn't match column type |
| `NullNotAllowed` | NULL in non-nullable column |
| `StorageError` | Underlying Tensor Store error |

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `insert` | O(1) | Schema validation + store put |
| `select` | O(n) | Full table scan with filter |
| `update` | O(n) | Scan + conditional update |
| `delete_rows` | O(n) | Scan + conditional delete |
| `join` | O(n*m) | Nested loop join |

For large datasets, consider:
- Adding indexes (future work)
- Hash join for equality joins (future work)
- Query planning and optimization (future work)

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
| `delete_data_is_gone` | Data actually removed |
| `table_not_found_error` | Error handling |
| `type_mismatch_error` | Schema enforcement |
| `null_not_allowed_error` | Nullability enforcement |

## Future Considerations

Not implemented (out of scope for Module 2):

- **Indexes**: B-tree or hash indexes for faster lookups
- **Query Optimization**: Cost-based query planning
- **Transactions**: ACID guarantees
- **Foreign Keys**: Referential integrity
- **Aggregations**: COUNT, SUM, AVG, etc.
- **Sorting**: ORDER BY
- **Pagination**: LIMIT, OFFSET
