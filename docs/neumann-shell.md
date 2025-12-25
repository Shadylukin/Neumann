# Neumann Shell

Module 7 of Neumann. Provides an interactive CLI interface for query execution.

## Design Principles

1. **Human-First Interface**: Readable prompts, formatted output, command history
2. **Thin Layer**: Minimal logic, delegates to Query Router for execution
3. **Graceful Handling**: Ctrl+C doesn't exit, errors are displayed cleanly
4. **Zero Configuration**: Works out of the box with sensible defaults

## Quick Start

```bash
# Run the shell
cargo run --package neumann_shell

# Or after installation
neumann
```

## API Reference

### Shell Creation

```rust
use neumann_shell::{Shell, ShellConfig};

// Default configuration
let shell = Shell::new();

// Custom configuration
let config = ShellConfig {
    history_file: Some("/custom/path/.neumann_history".into()),
    history_size: 500,
    prompt: "neumann> ".to_string(),
};
let shell = Shell::with_config(config);
```

### Running the REPL

```rust
// Start interactive loop
shell.run()?;
```

### Programmatic Execution

```rust
use neumann_shell::CommandResult;

// Execute a single command
match shell.execute("SELECT * FROM users") {
    CommandResult::Output(text) => println!("{}", text),
    CommandResult::Error(err) => eprintln!("Error: {}", err),
    CommandResult::Exit => println!("Goodbye!"),
    CommandResult::Help(text) => println!("{}", text),
    CommandResult::Empty => {}, // No-op for empty input
}
```

### Command Result Types

```rust
pub enum CommandResult {
    Output(String),  // Query result or command output
    Exit,            // Shell should exit
    Help(String),    // Help text
    Empty,           // Empty input (no-op)
    Error(String),   // Error message
}
```

## Built-in Commands

| Command | Aliases | Description |
|---------|---------|-------------|
| `help` | `\h`, `\?` | Show help message |
| `exit` | `quit`, `\q` | Exit the shell |
| `tables` | `\dt` | List all tables |
| `clear` | `\c` | Clear the screen |
| `save 'path'` | - | Save database snapshot to file |
| `load 'path'` | - | Load database snapshot from file |

## Query Support

The shell supports all query types from the Query Router:

### Relational Queries

```sql
> CREATE TABLE users (id INT, name TEXT, email TEXT)
OK

> INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')
1 row affected

> SELECT * FROM users
id | name  | email
---+-------+------------------
1  | Alice | alice@example.com
(1 rows)

> SELECT * FROM users WHERE id = 1
id | name  | email
---+-------+------------------
1  | Alice | alice@example.com
(1 rows)
```

### Graph Queries

```sql
> NODE CREATE person {name: 'Alice', age: 30}
ID: 1

> NODE CREATE person {name: 'Bob', age: 25}
ID: 2

> EDGE CREATE 1 -> 2 : knows
ID: 1

> NEIGHBORS 1 OUTGOING
Nodes:
  [2] person {name: Bob, age: 25}
(1 nodes)

> PATH 1 -> 2
Path: 1 -> 2
```

### Vector Queries

```sql
> EMBED STORE 'doc1' [0.1, 0.2, 0.3, 0.4]
OK

> EMBED GET 'doc1'
[0.1, 0.2, 0.3, 0.4]

> SIMILAR 'doc1' LIMIT 3
Similar:
  1. doc1 (similarity: 1.0000)
  2. doc2 (similarity: 0.9500)
  3. doc3 (similarity: 0.8700)
```

### Persistence Commands

Save and load database snapshots:

```sql
> CREATE TABLE users (id INT, name TEXT)
OK

> INSERT INTO users VALUES (1, 'Alice')
1 row affected

> EMBED STORE 'doc1' [0.1, 0.2, 0.3]
OK

> SAVE 'backup.bin'
Saved snapshot to: backup.bin

> -- Later, or in a new session:
> LOAD 'backup.bin'
Loaded snapshot from: backup.bin

> SELECT * FROM users
id | name
---+------
1  | Alice
(1 rows)
```

The snapshot includes all data from the TensorStore:
- Relational tables and rows
- Graph nodes and edges
- Vector embeddings

Path can be quoted (`'path'` or `"path"`) or unquoted.

## Configuration

### ShellConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `history_file` | `Option<PathBuf>` | `~/.neumann_history` | Path for persistent history |
| `history_size` | `usize` | `1000` | Maximum history entries |
| `prompt` | `String` | `"> "` | Input prompt string |

### Disabling History

```rust
let config = ShellConfig {
    history_file: None,  // No persistent history
    history_size: 100,
    prompt: "> ".to_string(),
};
```

## Output Formatting

### Tables (ASCII)

```
name  | age | email
------+-----+------------------
Alice | 30  | alice@example.com
Bob   | 25  | bob@example.com
(2 rows)
```

### Nodes

```
Nodes:
  [1] person {name: Alice, age: 30}
  [2] person {name: Bob, age: 25}
(2 nodes)
```

### Edges

```
Edges:
  [1] 1 -> 2 : knows
(1 edges)
```

### Paths

```
Path: 1 -> 3 -> 5 -> 7
```

### Similar Embeddings

```
Similar:
  1. doc1 (similarity: 0.9800)
  2. doc2 (similarity: 0.9500)
```

## Error Handling

| Error Type | Example |
|------------|---------|
| Parse error | `Error: unexpected token 'FORM' at position 12` |
| Table not found | `Error: table 'users' not found` |
| Invalid query | `Error: unsupported operation` |

Errors are printed to stderr and don't exit the shell.

## Keyboard Shortcuts

Provided by rustyline:

| Shortcut | Action |
|----------|--------|
| Up/Down | Navigate history |
| Ctrl+C | Cancel current input |
| Ctrl+D | Exit shell (EOF) |
| Ctrl+L | Clear screen |
| Ctrl+A | Move to start of line |
| Ctrl+E | Move to end of line |
| Ctrl+W | Delete word backward |
| Ctrl+U | Delete to start of line |

## Performance Characteristics

| Operation | Time |
|-----------|------|
| Empty input | 2.3 ns |
| Help command | 43 ns |
| SELECT (100 rows) | 17.8 us |
| Format 1000 rows | 267 us |

The shell adds negligible overhead to query execution.

## Test Coverage

| Test Category | What It Verifies |
|---------------|------------------|
| Command parsing | Built-in commands recognized |
| Query execution | Queries routed correctly |
| Output formatting | All result types formatted |
| Error handling | Errors displayed, shell continues |
| Configuration | Custom config applied |

Coverage: 94.81% (94% minimum due to untestable interactive REPL code)

## Dependencies

- `query_router`: Query execution
- `relational_engine`: Row type for formatting
- `tensor_store`: Snapshot persistence (save/load)
- `rustyline`: Readline functionality (history, shortcuts, Ctrl+C)

## Future Considerations

Not implemented (out of scope):

- **Syntax highlighting**: Colorized query input
- **Tab completion**: Auto-complete table/column names
- **Multi-line input**: Queries spanning multiple lines
- **Script mode**: Execute queries from file
- **Output formats**: JSON, CSV export
