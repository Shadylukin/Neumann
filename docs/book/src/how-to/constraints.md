# Constraints

This guide covers how to configure and manage data integrity constraints in the
relational engine, including primary keys, unique constraints, foreign keys,
and not-null constraints.

For the full API reference, see the
[Relational Engine API Reference](../reference/api/relational-engine.md).

## Constraint Types

| Constraint | Description |
| --- | --- |
| `PrimaryKey` | Unique + not null, identifies rows uniquely |
| `Unique` | Values must be unique (NULLs allowed) |
| `ForeignKey` | References rows in another table |
| `NotNull` | Column cannot contain NULL values |

## Create a Table with Constraints

Use `Schema::with_constraints` to define constraints at table creation time:

```rust
use relational_engine::{Constraint, ForeignKeyConstraint, ReferentialAction};

let schema = Schema::with_constraints(
    vec![
        Column::new("id", ColumnType::Int),
        Column::new("email", ColumnType::String),
        Column::new("dept_id", ColumnType::Int).nullable(),
    ],
    vec![
        Constraint::primary_key("pk_users", vec!["id".to_string()]),
        Constraint::unique("uq_email", vec!["email".to_string()]),
    ],
);
engine.create_table("users", schema)?;
```

## Add Constraints After Table Creation

### NOT NULL Constraint

```rust
engine.add_constraint("users", Constraint::not_null("nn_email", "email"))?;
```

### UNIQUE Constraint

```rust
engine.add_constraint("users", Constraint::unique("uq_name", vec!["name".to_string()]))?;
```

### Foreign Key Constraint

1. Define the foreign key with source columns, target table, and target columns.
2. Optionally configure referential actions for DELETE and UPDATE.
3. Add the constraint to the table.

```rust
let fk = ForeignKeyConstraint::new(
    "fk_users_dept",
    vec!["dept_id".to_string()],
    "departments",
    vec!["id".to_string()],
)
.on_delete(ReferentialAction::SetNull)
.on_update(ReferentialAction::Cascade);
engine.add_constraint("users", Constraint::foreign_key(fk))?;
```

### Referential Actions

Foreign keys support these actions on delete/update of referenced rows:

| Action | Description |
| --- | --- |
| `Restrict` (default) | Prevent the operation |
| `Cascade` | Cascade to referencing rows |
| `SetNull` | Set referencing columns to NULL |
| `SetDefault` | Set referencing columns to default |
| `NoAction` | Same as Restrict, checked at commit |

## List Constraints

```rust
let constraints = engine.get_constraints("users")?;
for c in &constraints {
    println!("{:?}", c);
}
```

## Drop a Constraint

```rust
engine.drop_constraint("users", "uq_email")?;
```
