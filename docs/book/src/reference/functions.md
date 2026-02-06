# Functions Reference

---

## Aggregate Functions

These functions operate on groups of rows in SELECT queries with GROUP BY.

| Function | Description | Example |
|----------|-------------|---------|
| `COUNT(*)` | Count all rows | `SELECT COUNT(*) FROM users` |
| `COUNT(column)` | Count non-null values | `SELECT COUNT(name) FROM users` |
| `SUM(column)` | Sum numeric values | `SELECT SUM(total) FROM orders` |
| `AVG(column)` | Average numeric values | `SELECT AVG(age) FROM users` |
| `MIN(column)` | Minimum value | `SELECT MIN(created) FROM orders` |
| `MAX(column)` | Maximum value | `SELECT MAX(total) FROM orders` |

```sql
SELECT team, COUNT(*) AS headcount, AVG(age) AS avg_age
FROM employees
GROUP BY team
HAVING COUNT(*) > 5
ORDER BY headcount DESC
```

---

## Graph Algorithm Functions

These are invoked as top-level commands, not as SQL functions. See the
[Query Language Reference](query-language.md#graph-algorithms) for full syntax.

| Algorithm | Description | Returns |
|-----------|-------------|---------|
| `PAGERANK` | Link analysis ranking | Node scores (0.0-1.0) |
| `BETWEENNESS` | Bridge node importance | Node scores |
| `CLOSENESS` | Average distance to all nodes | Node scores |
| `EIGENVECTOR` | Influence-based ranking | Node scores |
| `LOUVAIN` | Community detection | Community assignments |
| `LABEL_PROPAGATION` | Community detection | Community assignments |

---

## Graph Aggregate Functions

Used with the `GRAPH AGGREGATE` command on node/edge properties.

| Function | Description |
|----------|-------------|
| `COUNT` | Count nodes/edges matching criteria |
| `SUM` | Sum of property values |
| `AVG` | Average of property values |
| `MIN` | Minimum property value |
| `MAX` | Maximum property value |

```sql
GRAPH AGGREGATE COUNT NODES person
GRAPH AGGREGATE AVG NODE age person WHERE age > 20
GRAPH AGGREGATE SUM EDGE weight collaborates
```

---

## Distance Metrics

Used with `SIMILAR` and `EMBED` commands for vector similarity search.

| Metric | Keyword | Range | Best For |
|--------|---------|-------|----------|
| Cosine similarity | `COSINE` | -1.0 to 1.0 | Text embeddings, normalized vectors |
| Euclidean distance | `EUCLIDEAN` | 0.0 to infinity | Spatial data, image features |
| Dot product | `DOT_PRODUCT` | -infinity to infinity | Pre-normalized vectors, recommendation |

```sql
SIMILAR [0.1, 0.2, 0.3] LIMIT 10 METRIC COSINE
SIMILAR 'doc1' LIMIT 5 METRIC EUCLIDEAN
```

The default metric is `COSINE` when not specified.

---

## Expression Operators

### Arithmetic

| Operator | Description |
|----------|-------------|
| `+` | Addition |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division |
| `%` | Modulo |

### Comparison

| Operator | Description |
|----------|-------------|
| `=` | Equal |
| `!=` or `<>` | Not equal |
| `<` | Less than |
| `<=` | Less than or equal |
| `>` | Greater than |
| `>=` | Greater than or equal |

### Logical

| Operator | Description |
|----------|-------------|
| `AND` | Logical AND |
| `OR` | Logical OR |
| `NOT` | Logical NOT |

### Special Predicates

| Predicate | Description | Example |
|-----------|-------------|---------|
| `IS NULL` | Test for null | `WHERE name IS NULL` |
| `IS NOT NULL` | Test for non-null | `WHERE name IS NOT NULL` |
| `IN (list)` | Set membership | `WHERE id IN (1, 2, 3)` |
| `NOT IN (list)` | Set non-membership | `WHERE id NOT IN (1, 2)` |
| `BETWEEN a AND b` | Range check | `WHERE age BETWEEN 18 AND 65` |
| `LIKE pattern` | Pattern matching | `WHERE name LIKE 'A%'` |
| `NOT LIKE pattern` | Negative pattern | `WHERE name NOT LIKE '%test%'` |
| `EXISTS (subquery)` | Subquery existence | `WHERE EXISTS (SELECT ...)` |

### CASE Expression

```sql
CASE
    WHEN condition THEN result
    [WHEN condition THEN result ...]
    [ELSE default]
END
```

```sql
SELECT name,
    CASE
        WHEN age < 18 THEN 'minor'
        WHEN age < 65 THEN 'adult'
        ELSE 'senior'
    END AS category
FROM users
```

### CAST

```sql
CAST(expression AS type)
```

```sql
SELECT CAST(age AS FLOAT) / 10 AS decade FROM users
```
