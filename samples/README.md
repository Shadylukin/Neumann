# Neumann Sample Datasets

## Knowledge Base (`knowledge-base.nql`)

A sample engineering team knowledge base that exercises all three engines plus vault,
cache, and checkpoint features.

### What it contains

- **Relational**: `people` table with 10 employees (id, name, role, team, joined, level)
- **Graph**: Person nodes with REPORTS_TO, COLLABORATES_WITH, and MENTORS edges
- **Vector**: 8-dimensional skill embeddings for each person
- **Unified Entities**: Cross-engine project entities
- **Cache**: Pre-warmed LLM responses about the team
- **Checkpoint**: A saved checkpoint of the initial state

### Loading the dataset

Start the shell with persistence enabled:

```bash
neumann --wal-dir ./data
```

Then load the sample file:

```text
neumann> \i samples/knowledge-base.nql
```

Or paste sections individually to follow along with the
[Quick Start](../docs/book/src/getting-started/quick-start.md).

### Queries to try

**Relational:**

```sql
SELECT * FROM people WHERE team = 'Platform'
SELECT team, COUNT(*) AS headcount FROM people GROUP BY team
```

**Graph:**

```sql
NODE LIST person LIMIT 5
PAGERANK EDGE_TYPE reports_to
```

**Vector:**

```sql
SIMILAR 'alice' LIMIT 3
SIMILAR [0.9, 0.8, 0.1, 0.5, 0.7, 0.2, 0.3, 0.5] LIMIT 5 METRIC COSINE
```

**Cross-engine:**

```sql
FIND NODE person WHERE name = 'Alice Chen'
ENTITY GET 'project_atlas'
```
