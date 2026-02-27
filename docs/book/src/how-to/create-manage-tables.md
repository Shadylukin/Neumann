# Create and Manage Tables

This guide covers table lifecycle operations in the relational engine: creating
tables, inserting and querying rows, updating and deleting data, altering
schemas, and using joins and aggregates.

For the full API reference, see the
[Relational Engine API Reference](../reference/api/relational-engine.md). For
bulk loading, see [Batch Insert](batch-insert.md). For constraint management,
see [Constraints](constraints.md).

## Create a Table

1. Define a schema with column names, types, and nullability.
2. Call `create_table` with the table name and schema.

```rust
let engine = RelationalEngine::new();

let schema = Schema::new(vec![
    Column::new("name", ColumnType::String),
    Column::new("age", ColumnType::Int),
    Column::new("email", ColumnType::String).nullable(),
]);
engine.create_table("users", schema)?;
```

## Check Table Existence and List Tables

```rust
// Check if a table exists
engine.table_exists("users")?;  // -> bool

// List all tables
let tables = engine.list_tables();  // -> Vec<String>

// Get the schema of a table
let schema = engine.get_schema("users")?;

// Get the row count
engine.row_count("users")?;  // -> usize
```

## Insert Rows

1. Build a `HashMap<String, Value>` with column names and values.
2. Call `insert` to add a single row, or `batch_insert` for many rows.

```rust
let mut values = HashMap::new();
values.insert("name".to_string(), Value::String("Alice".into()));
values.insert("age".to_string(), Value::Int(30));
let row_id = engine.insert("users", values)?;
```

For inserting many rows at once, see [Batch Insert](batch-insert.md).

## Select Rows

Use `select` with a `Condition` to filter rows:

```rust
// Select all users with age = 30
let rows = engine.select("users", Condition::Eq("age".into(), Value::Int(30)))?;

// Select all rows
let all_rows = engine.select("users", Condition::True)?;
```

Conditions can be composed with `.and()` and `.or()`:

```rust
// age >= 18 AND age < 65
let condition = Condition::Ge("age".into(), Value::Int(18))
    .and(Condition::Lt("age".into(), Value::Int(65)));
let rows = engine.select("users", condition)?;
```

## Update Rows

1. Build a `HashMap` of columns to update.
2. Call `update` with a condition to match target rows.

```rust
let mut updates = HashMap::new();
updates.insert("age".to_string(), Value::Int(31));
let count = engine.update(
    "users",
    Condition::Eq("name".into(), Value::String("Alice".into())),
    updates
)?;
```

The return value is the number of rows updated.

## Delete Rows

```rust
let count = engine.delete_rows("users", Condition::Lt("age".into(), Value::Int(18)))?;
```

The return value is the number of rows deleted.

## Drop a Table

Dropping a table removes all rows, indexes, and metadata:

```rust
engine.drop_table("users")?;
```

## ALTER TABLE Operations

### Add a Column

New columns must be nullable or have a default value:

```rust
engine.add_column("users", Column::new("phone", ColumnType::String).nullable())?;
```

### Drop a Column

Fails if the column has constraints that prevent removal:

```rust
engine.drop_column("users", "phone")?;
```

### Rename a Column

Automatically updates any constraints referencing the column:

```rust
engine.rename_column("users", "email", "email_address")?;
```

## Joins

All six SQL join types are supported. Each join takes two table names and the
column names to join on.

```rust
// INNER JOIN - Only matching rows from both tables
let joined = engine.join("users", "posts", "_id", "user_id")?;
// Returns: Vec<(Row, Row)>

// LEFT JOIN - All rows from left, matching from right (or None)
let joined = engine.left_join("users", "posts", "_id", "user_id")?;
// Returns: Vec<(Row, Option<Row>)>

// RIGHT JOIN - All rows from right, matching from left (or None)
let joined = engine.right_join("users", "posts", "_id", "user_id")?;
// Returns: Vec<(Option<Row>, Row)>

// FULL JOIN - All rows from both tables
let joined = engine.full_join("users", "posts", "_id", "user_id")?;
// Returns: Vec<(Option<Row>, Option<Row>)>

// CROSS JOIN (Cartesian product)
let joined = engine.cross_join("users", "posts")?;
// Returns: Vec<(Row, Row)> with n*m rows

// NATURAL JOIN (on common column names)
let joined = engine.natural_join("users", "user_profiles")?;
// Returns: Vec<(Row, Row)> matching on all common columns
```

For how joins work internally, see [Hash Join Algorithm](../explanation/hash-join.md).

## Aggregate Functions

```rust
// COUNT(*) - count all rows
let count = engine.count("users", Condition::True)?;

// COUNT(column) - count non-null values
let count = engine.count_column("users", "email", Condition::True)?;

// SUM - returns f64
let total = engine.sum("orders", "amount", Condition::True)?;

// AVG - returns Option<f64> (None if no matching rows)
let avg = engine.avg("orders", "amount", Condition::True)?;

// MIN/MAX - returns Option<Value>
let min = engine.min("products", "price", Condition::True)?;
let max = engine.max("products", "price", Condition::True)?;
```

## SQL Features via Query Router

When using the relational engine through `query_router`, additional SQL features
are available:

### ORDER BY and OFFSET

```sql
SELECT * FROM users ORDER BY age ASC;
SELECT * FROM users ORDER BY department DESC, name ASC;
SELECT * FROM users ORDER BY email NULLS FIRST;
SELECT * FROM users ORDER BY created_at DESC LIMIT 10 OFFSET 20;
```

### GROUP BY and HAVING

```sql
SELECT department, COUNT(*), AVG(salary) FROM employees GROUP BY department;

SELECT product, SUM(quantity) as total
FROM orders
GROUP BY product
HAVING SUM(quantity) > 100;
```
