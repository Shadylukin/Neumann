# Neumann Shell

Interactive CLI shell for the Neumann unified tensor database.

## Installation

### Quick Install (Recommended)

```bash
curl -sSfL https://raw.githubusercontent.com/Shadylukin/Neumann/main/install.sh | bash
```

### From crates.io

```bash
cargo install neumann_shell
```

### From Source

```bash
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann
cargo install --path neumann_shell
```

## Usage

```bash
# Start interactive shell
neumann

# Execute a single query
neumann -c "SELECT * FROM users"

# Run queries from a file
neumann -f queries.sql
```

## Features

- Interactive REPL with readline support
- Syntax highlighting and auto-completion
- Write-ahead logging for crash recovery
- Multi-engine query support (relational, graph, vector)
- Table-formatted output with color support

## Query Examples

```sql
-- Relational queries
CREATE TABLE users (id INT, name STRING, email STRING);
INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com');
SELECT * FROM users WHERE id = 1;

-- Graph queries
NODE CREATE person { name: 'Bob', age: 30 };
EDGE CREATE 1 -> 2 : knows;
NEIGHBORS 1 knows OUTGOING;
PATH 1 -> 2;

-- Vector queries
EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4];
SIMILAR 'doc1' LIMIT 10;
```

## Configuration

The shell reads configuration from `~/.neumann/config.toml`:

```toml
[shell]
history_file = "~/.neumann/history"
history_size = 10000
prompt = "neumann> "

[output]
format = "table"  # table, json, csv
color = true
```

## License

MIT OR Apache-2.0
