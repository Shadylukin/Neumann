# Relational Engine -- Quick Reference

## SELECT

```sql
SELECT [DISTINCT] columns
FROM table [AS alias]
[JOIN type table ON condition | USING (cols)]
[WHERE condition]
[GROUP BY col1, col2]
[HAVING condition]
[ORDER BY col [ASC|DESC] [NULLS FIRST|LAST]]
[LIMIT n]
[OFFSET m]
```

**Join types:** `INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL JOIN`, `CROSS JOIN`, `NATURAL JOIN`

**Set operations:** `SELECT ... UNION [ALL] SELECT ...`, `INTERSECT`, `EXCEPT`

**Subqueries in FROM:** `SELECT * FROM (SELECT col FROM t) AS sub`

**Expressions:**

- `CASE WHEN cond THEN val [WHEN ...] [ELSE val] END`
- `CAST(expr AS type)` -- types: INT, INTEGER, BIGINT, SMALLINT, FLOAT, DOUBLE,
  REAL, DECIMAL(p,s), NUMERIC(p,s), VARCHAR(n), CHAR(n), TEXT, BOOLEAN, DATE,
  TIME, TIMESTAMP, BLOB
- `EXISTS (SELECT ...)` -- correlated or uncorrelated subquery
- `expr IN (val1, val2)` or `expr IN (SELECT ...)`
- `expr NOT IN (...)`
- `expr BETWEEN low AND high` / `NOT BETWEEN`
- `expr LIKE 'pattern'` / `NOT LIKE` -- `%` and `_` wildcards
- `expr IS NULL` / `IS NOT NULL`
- `||` -- string concatenation

**Aggregates:** `COUNT(*)`, `COUNT(DISTINCT col)`, `SUM(col)`, `AVG(col)`, `MIN(col)`, `MAX(col)`

## INSERT

```sql
-- Single/multi row
INSERT INTO table [(col1, col2)] VALUES (val1, val2) [, (val3, val4)]

-- From query
INSERT INTO table [(col1, col2)] SELECT col1, col2 FROM other_table
```

## UPDATE

```sql
UPDATE table SET col1 = val1 [, col2 = val2] [WHERE condition]
```

## DELETE

```sql
DELETE FROM table [WHERE condition]
```

## CREATE TABLE

```sql
CREATE TABLE [IF NOT EXISTS] name (
    col1 INT PRIMARY KEY,
    col2 VARCHAR(255) NOT NULL,
    col3 FLOAT DEFAULT 0.0,
    col4 INT UNIQUE,
    col5 INT REFERENCES other_table(col),
    col6 INT CHECK (col6 > 0),
    CONSTRAINT pk PRIMARY KEY (col1),
    CONSTRAINT uq UNIQUE (col2, col3),
    CONSTRAINT fk FOREIGN KEY (col4) REFERENCES other(id) [ON DELETE CASCADE|RESTRICT] [ON UPDATE CASCADE|RESTRICT],
    CONSTRAINT ck CHECK (col5 > 0)
)
```

**Column constraints:** `PRIMARY KEY`, `NOT NULL`, `UNIQUE`, `DEFAULT expr`, `REFERENCES table(col)`, `CHECK (expr)`

**Referential actions:** `ON DELETE CASCADE`, `ON DELETE RESTRICT`, `ON UPDATE CASCADE`, `ON UPDATE RESTRICT`

## DDL

```sql
DROP TABLE [IF EXISTS] name
ALTER TABLE name ADD COLUMN col_name col_type [constraints]
ALTER TABLE name DROP COLUMN col_name
CREATE INDEX [IF NOT EXISTS] idx_name ON table (col1 [, col2])
DROP INDEX idx_name
SHOW TABLES
DESCRIBE table_name
```

## Data Types

| Type | Description |
|------|-------------|
| `INT` / `INTEGER` | 64-bit integer |
| `BIGINT` | 64-bit integer |
| `SMALLINT` | Small integer |
| `FLOAT` / `DOUBLE` / `REAL` | 64-bit floating point |
| `DECIMAL(p,s)` / `NUMERIC(p,s)` | Fixed precision |
| `VARCHAR(n)` / `CHAR(n)` / `TEXT` | String types |
| `BOOLEAN` | Boolean (TRUE/FALSE) |
| `DATE` / `TIME` / `TIMESTAMP` | Temporal types |
| `BLOB` | Binary data |
