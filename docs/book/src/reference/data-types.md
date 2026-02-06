# Data Types

Neumann has two layers of types: scalar values for individual fields and tensor
values for composite storage.

---

## Scalar Types

`ScalarValue` represents a single value in the system.

| Type | Description | Examples |
|------|-------------|---------|
| `Null` | Absence of a value | `NULL` |
| `Bool` | Boolean | `TRUE`, `FALSE` |
| `Int` | 64-bit signed integer | `42`, `-1`, `0` |
| `Float` | 64-bit floating point | `3.14`, `-0.5`, `1e10` |
| `String` | UTF-8 text | `'hello'`, `'Alice'` |
| `Bytes` | Raw binary data | (used internally for blob content) |

### Literals

- **Strings**: Single-quoted: `'hello world'`
- **Integers**: Unquoted numbers: `42`, `-7`
- **Floats**: Numbers with decimal or exponent: `3.14`, `1e-5`
- **Booleans**: `TRUE` or `FALSE` (case-insensitive)
- **Null**: `NULL`
- **Arrays**: Square brackets: `[1, 2, 3]` or `[0.1, 0.2, 0.3]`

---

## Tensor Types

`TensorValue` wraps scalar values with vector and pointer types for the unified
data model.

| Type | Description | Use Case |
|------|-------------|----------|
| `Scalar(ScalarValue)` | Single scalar value | Table columns, node properties |
| `Vector(Vec<f32>)` | Dense float vector | Embeddings for similarity search |
| `Sparse(SparseVector)` | Sparse vector | Memory-efficient high-dimensional embeddings |
| `Pointer(String)` | Reference to another entity | Graph edges, foreign keys |
| `Pointers(Vec<String>)` | Multiple references | Multi-valued relationships |

---

## Column Types in CREATE TABLE

When creating relational tables, use SQL-style type names. These map to internal
scalar types.

| SQL Type | Internal Type | Notes |
|----------|---------------|-------|
| `INT`, `INTEGER` | Int | 64-bit signed integer |
| `BIGINT` | Int | Same as INT (64-bit) |
| `SMALLINT` | Int | Same as INT (64-bit) |
| `FLOAT` | Float | 64-bit floating point |
| `DOUBLE` | Float | Same as FLOAT |
| `REAL` | Float | Same as FLOAT |
| `DECIMAL(p,s)` | Float | Precision and scale are advisory |
| `NUMERIC(p,s)` | Float | Same as DECIMAL |
| `VARCHAR(n)` | String | Max length is advisory |
| `CHAR(n)` | String | Fixed-width (padded) |
| `TEXT` | String | Unlimited length |
| `BOOLEAN` | Bool | TRUE or FALSE |
| `DATE` | String | Stored as ISO-8601 string |
| `TIME` | String | Stored as ISO-8601 string |
| `TIMESTAMP` | String | Stored as ISO-8601 string |
| `BLOB` | Bytes | Raw binary data |
| custom name | String | Any unrecognized type stores as String |

---

## Type Coercion

Neumann performs implicit type coercion in comparisons:

- `Int` and `Float` in arithmetic: Int is promoted to Float
- `String` comparisons: lexicographic ordering
- `Null` propagation: any operation with NULL yields NULL
- Boolean context: only Bool values are truthy/falsy (no implicit conversion from Int)

---

## Vector Representation

Dense vectors are stored as `Vec<f32>` and used for similarity search via HNSW
indexes. All vectors in a collection must have the same dimensionality.

```sql
EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4]
```

Sparse vectors use a compact representation storing only non-zero indices and values,
making them efficient for high-dimensional data (e.g., 30,000+ dimensions for
bag-of-words models).

---

## Identifiers

Identifiers (table names, column names, labels) follow these rules:

- Start with a letter or underscore
- Contain letters, digits, and underscores
- Case-insensitive for keywords, case-preserving for identifiers
- No quoting required for simple names
- Use single quotes for string values: `'value'`
