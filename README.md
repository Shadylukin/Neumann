# Neumann

<p align="center">
  <img src="images/neumann_logo.png" alt="Neumann" height="80" />
  <img src="images/neumann_text.png" alt="Neumann" height="60" />
</p>

Stop juggling five databases for one AI app.

Neumann stores your tables, graphs, and vectors in one place.
Query across all three in a single statement.

```sql
-- Find engineers similar to Alice who report to Bob
FIND NODE person
  WHERE role = 'engineer'
  SIMILAR TO 'user:alice'
  CONNECTED TO 'user:bob'
```

One query. Relational filter + vector similarity + graph traversal.

[![CI](https://github.com/Shadylukin/Neumann/actions/workflows/ci.yml/badge.svg)](https://github.com/Shadylukin/Neumann/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Shadylukin/Neumann/graph/badge.svg)](https://codecov.io/gh/Shadylukin/Neumann)
[![License](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Discord](https://img.shields.io/badge/Discord-Join-7289da.svg)](https://discord.gg/uN3KbAyKvw)
[![Sponsor](https://img.shields.io/badge/Sponsor-EA4AAA?logo=githubsponsors&logoColor=fff)](https://github.com/sponsors/Shadylukin)

## Why Neumann

**Three engines, one system.** Store a table, connect entities in a graph,
and search by vector similarity -- without moving data between systems.
No ETL, no sync, no glue code.

**Semantic consensus.** Concurrent writes to different fields auto-merge.
The consensus layer classifies conflicts geometrically rather than treating
all concurrent writes as errors.

```text
Alice updates email, Bob updates photo (same user, same time).
Traditional DB: conflict, manual resolution.
Neumann: auto-merges (different fields = orthogonal changes).
```

**AI-native by design.** Built-in embedding storage, semantic caching for
LLM responses, and encrypted vault for secrets. The query language
understands similarity, not just equality.

## Use Cases

**RAG** -- Store documents with embeddings and relationships. Semantic
search follows graph links automatically.

**Agent memory** -- Conversation history with vector recall across
sessions. Cache repeated LLM calls to cut API costs.

**Knowledge graphs** -- Combine structured data with semantic similarity.
Find entities by what they *mean*, not just what they match.

**Access control** -- Graph-based permissions. Query results respect
who's asking.

See [Use Cases](docs/book/src/tutorials/use-cases.md) for worked examples
with Python, TypeScript, and the CLI.

## Quick Start

```bash
curl -sSfL https://raw.githubusercontent.com/Shadylukin/Neumann/main/install.sh | bash
neumann
```

```sql
-- Relational
CREATE TABLE users (id INT, name TEXT, role TEXT);
INSERT users id=1, name='Alice', role='engineer';
SELECT * FROM users WHERE role = 'engineer';

-- Graph
NODE CREATE person {name: 'Alice'};
NODE CREATE person {name: 'Bob'};
EDGE CREATE 1 -> 2 : reports_to;

-- Vector
EMBED STORE 'user:alice' [0.1, 0.2, 0.3];
SIMILAR 'user:alice' TOP 5;
```

More install methods (Homebrew, Cargo, Docker, source) in the
[Installation Guide](docs/book/src/how-to/installation.md).
Full walkthrough in the
[Quick Start Tutorial](docs/book/src/tutorials/quick-start.md).

## SDKs

**Python**

```bash
pip install neumann-db
```

```python
from neumann import NeumannClient
client = NeumannClient.connect("localhost:9200")
result = client.execute("SELECT * FROM users")
```

**TypeScript**

```bash
npm install @scrunchee/client
```

```typescript
import { NeumannClient } from '@scrunchee/client';
const client = await NeumannClient.connect("localhost:9200");
const result = await client.execute("SELECT * FROM users");
```

Full tutorials:
[Python](docs/book/src/tutorials/python-sdk.md) |
[TypeScript](docs/book/src/tutorials/typescript-sdk.md)

## Dashboard

![Neumann Dashboard](images/dash.png)
*Web dashboard with system status and query terminal*

![Graph Visualization](images/graph.png)
*Interactive graph visualization with force-directed layout*

## Documentation

| I want to... | Go to |
|---|---|
| Follow a tutorial | [Quick Start](docs/book/src/tutorials/quick-start.md), [RAG in 5 Minutes](docs/book/src/tutorials/five-minute-rag.md), [Knowledge Graph](docs/book/src/tutorials/knowledge-graph.md) |
| Solve a specific problem | [How-to Guides](docs/book/src/how-to/installation.md) (30+ guides covering storage, graphs, vectors, security, deployment) |
| Look up syntax or config | [Query Language](docs/book/src/reference/query-language.md), [Configuration](docs/book/src/reference/configuration.md), [Error Types](docs/book/src/reference/error-types.md) |
| Understand the architecture | [Design Overview](docs/book/src/explanation/architecture-overview.md), [Consensus](docs/book/src/explanation/consensus-protocols.md), [HNSW](docs/book/src/explanation/hnsw-algorithm.md) |

[Full table of contents](docs/book/src/SUMMARY.md) |
[Rustdoc API Reference](https://shadylukin.github.io/Neumann/)

## Performance

Benchmarked on Apple M-series silicon:

- **3.8M reads/sec, 2.0M writes/sec** (in-memory, no durability)
- **22K durable writes/sec** with group commit, 260/sec with fsync
- **150us vector similarity** (HNSW, 10K embeddings, 128-dim)
- **52M conflict checks/sec** via sparse delta comparison

[Full benchmarks](docs/book/src/reference/benchmarks/index.md)

## Status

Neumann is pre-1.0 (v0.3.1). Core engines have 95%+ test coverage and
139 fuzz targets. Single-node is production-ready. Multi-node consensus
has comprehensive testing (loom, proptest, deterministic simulation) but
needs more real-world validation.

See the [roadmap](ROADMAP.md) for where we're headed -- native
embeddings, natural language queries, and AI-native analytics.

## The Name

John von Neumann unified code and data in the stored-program architecture.
Neumann unifies structure, relationships, and semantics.

## License

Licensed under the [Business Source License 1.1](LICENSE).

## Author

Built by [Lukin Ackroyd](https://scrunchee.ai) in Auckland, New Zealand.

Neumann is the infrastructure layer for [Scrunchee](https://scrunchee.ai).
