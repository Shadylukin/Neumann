# Unified Engine -- Quick Reference

The unified engine provides cross-engine entity operations (ENTITY) and a
flexible query interface (FIND) that spans relational, graph, and vector data.

## ENTITY Commands

```sql
-- Create an entity with properties and optional embedding
ENTITY CREATE 'user-1' { name: 'Alice', role: 'admin' }
ENTITY CREATE 'user-1' { name: 'Alice' } EMBEDDING [0.1, 0.2, 0.3]

-- Get an entity by key
ENTITY GET 'user-1'

-- Update entity properties and/or embedding
ENTITY UPDATE 'user-1' { role: 'superadmin' }
ENTITY UPDATE 'user-1' { role: 'superadmin' } EMBEDDING [0.4, 0.5, 0.6]

-- Delete an entity
ENTITY DELETE 'user-1'

-- Connect two entities with a typed edge
ENTITY CONNECT 'user-1' -> 'team-1' : member_of

-- Batch create multiple entities
ENTITY BATCH CREATE [
    { key: 'u1', name: 'Alice' },
    { key: 'u2', name: 'Bob' }
]
```

### Entity key rules

- Keys must be quoted strings: `'my-entity'`
- Keys with colons must be quoted: `'doc:123'`
- Properties use `{ key: value }` syntax (colons, not equals)

### ENTITY CONNECT syntax

Same arrow syntax as EDGE CREATE:

```sql
ENTITY CONNECT 'from-key' -> 'to-key' : relationship_type
```

## FIND Commands

FIND provides a unified query interface across engines.

### FIND NODE

```sql
FIND NODE [label] [WHERE condition] RETURN items [LIMIT n]

-- Examples
FIND NODE RETURN *
FIND NODE person WHERE age > 25 RETURN name, age LIMIT 10
```

### FIND EDGE

```sql
FIND EDGE [type] [WHERE condition] RETURN items [LIMIT n]

-- Examples
FIND EDGE RETURN *
FIND EDGE knows WHERE since > 2020 RETURN * LIMIT 5
```

### FIND ROWS

```sql
FIND ROWS FROM table [WHERE condition] RETURN items [LIMIT n]

-- Examples
FIND ROWS FROM users RETURN *
FIND ROWS FROM orders WHERE total > 100 RETURN id, total LIMIT 20
```

### FIND PATH

```sql
FIND PATH [from_label]-[:edge_type]->[to_label] [WHERE condition] RETURN items

-- Examples
FIND PATH [person]-[:knows]->[person] RETURN *
FIND PATH -[:works_at]-> RETURN *
```

### RETURN clause

- `RETURN *` -- return all fields
- `RETURN col1, col2` -- specific columns
- `RETURN col1 AS alias` -- aliased columns

### WHERE clause

Standard SQL-like conditions apply: `=`, `!=`, `<`, `>`, `<=`, `>=`, `AND`, `OR`, `NOT`, `LIKE`, `IN`, `BETWEEN`, `IS NULL`.

## Cross-Engine Patterns

The unified engine is designed for workflows that span multiple engines:

```sql
-- 1. Create entity with relational data + vector embedding
ENTITY CREATE 'product-1' { name: 'Widget', price: 29.99 } EMBEDDING [0.1, 0.2]

-- 2. Connect entities (creates graph edge)
ENTITY CONNECT 'product-1' -> 'category-electronics' : belongs_to

-- 3. Find similar products that are connected to a category
SIMILAR [0.1, 0.2] LIMIT 5 CONNECTED TO 'category-electronics'
```
