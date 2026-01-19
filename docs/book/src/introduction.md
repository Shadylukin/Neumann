# Introduction

Neumann is a unified tensor-based runtime that stores relational data, graph relationships, and vector embeddings in a single mathematical structure.

## Why Neumann?

Traditional databases force you to choose: SQL for structured data, graph databases for relationships, or vector stores for embeddings. Neumann unifies all three into a single system built on sparse tensor mathematics.

### Key Benefits

- **Unified Data Model**: Store tables, graphs, and embeddings in one system
- **Semantic Operations**: Query across modalities using tensor operations
- **Distributed by Design**: Built-in Raft consensus with 2PC transactions
- **Memory Efficient**: Sparse vector representations minimize storage
- **Consistent**: Strong consistency guarantees with hybrid logical clocks

## Architecture Overview

```
+-------------------+
|   neumann_shell   |  <- Interactive CLI
+-------------------+
         |
+-------------------+
|   query_router    |  <- Unified query execution
+-------------------+
         |
+--------+--------+--------+--------+
|        |        |        |        |
v        v        v        v        v
relational graph  vector  vault   cache   blob   chain
_engine  _engine _engine _vault  _cache  _blob  _chain
         |        |        |        |        |        |
         +--------+--------+--------+--------+--------+
                           |
                  +--------+--------+
                  |                 |
              tensor_store    tensor_compress
```

## Quick Example

```sql
-- Create a table with vector embeddings
CREATE TABLE documents (
    id INT PRIMARY KEY,
    title STRING,
    embedding VECTOR(128)
);

-- Insert with semantic content
INSERT INTO documents VALUES (1, 'Introduction to ML', [0.1, 0.2, ...]);

-- Find similar documents
SELECT title FROM documents
WHERE SIMILAR(embedding, [0.15, 0.18, ...], 0.9);
```

## Getting Started

- [Installation](getting-started/installation.md) - Set up Neumann on your system
- [Quick Start](getting-started/quick-start.md) - Your first queries in minutes
- [Building from Source](getting-started/building-from-source.md) - Compile from source

## Documentation Structure

- **Getting Started**: Installation and first steps
- **Architecture**: Deep dives into each module
- **Concepts**: Cross-cutting ideas like sparse vectors and consensus
- **Operations**: Deployment, monitoring, and troubleshooting
- **Contributing**: How to contribute to Neumann

## API Reference

Full API documentation is available in the [rustdoc output](api-reference.md).
