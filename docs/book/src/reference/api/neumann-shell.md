# Neumann Shell API Reference

> **See Also:**
> - [Shell Design](../../explanation/shell-design.md) -- REPL architecture, WAL integration, and output formatting
> - [Query Router API](neumann-server.md) -- server-side query execution
> - [Neumann Client API](neumann-client.md) -- programmatic client alternative

## Key Types

| Type | Description |
| --- | --- |
| `Shell` | Main shell struct holding router, config, and WAL state |
| `ShellConfig` | Configuration for history file, history size, and prompt |
| `CommandResult` | Result enum: `Output`, `Exit`, `Help`, `Empty`, `Error` |
| `LoopAction` | Action after command: `Continue` or `Exit` |
| `ShellError` | Error type for initialization failures |
| `Wal` | Internal write-ahead log for crash recovery |
| `RouterExecutor` | Wrapper implementing `QueryExecutor` trait for cluster operations |
| `ShellConfirmationHandler` | Interactive confirmation handler for destructive operations |

## Shell Configuration

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `history_file` | `Option<PathBuf>` | `~/.neumann_history` | Path for persistent history |
| `history_size` | `usize` | `1000` | Maximum history entries |
| `prompt` | `String` | `"> "` | Input prompt string |

The default history file location is determined by reading the `HOME`
environment variable:

```rust
fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
```

## Command Result Types

| Variant | Description | REPL Behavior |
| --- | --- | --- |
| `Output(String)` | Query executed successfully with output | Print to stdout, continue loop |
| `Exit` | Shell should exit | Print "Goodbye!", break loop |
| `Help(String)` | Help text to display | Print to stdout, continue loop |
| `Empty` | Empty input (no-op) | Continue loop silently |
| `Error(String)` | Error occurred | Print to stderr, continue loop |

## Shell Creation

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

## Running the REPL

```rust
shell.run()?;
```

## Programmatic Execution

```rust
use neumann_shell::CommandResult;

match shell.execute("SELECT * FROM users") {
    CommandResult::Output(text) => println!("{}", text),
    CommandResult::Error(err) => eprintln!("Error: {}", err),
    CommandResult::Exit => println!("Goodbye!"),
    CommandResult::Help(text) => println!("{}", text),
    CommandResult::Empty => {},
}
```

## Direct Router Access

The shell provides thread-safe access to the underlying Query Router:

```rust
// Read-only access
let router_guard = shell.router();
let tables = router_guard.list_tables();

// Mutable access
let mut router_guard = shell.router_mut();
router_guard.init_vault(&key)?;

// Get Arc clone for shared ownership
let router_arc = shell.router_arc();
```

## Built-in Commands

| Command | Aliases | Description |
| --- | --- | --- |
| `help` | `\h`, `\?` | Show help message |
| `exit` | `quit`, `\q` | Exit the shell |
| `tables` | `\dt` | List all tables |
| `clear` | `\c` | Clear the screen (ANSI escape: `\x1B[2J\x1B[H`) |
| `save 'path'` | --- | Save database snapshot to file |
| `save compressed 'path'` | --- | Save compressed snapshot (int8 quantization) |
| `load 'path'` | --- | Load database snapshot from file (auto-detects format) |
| `wal status` | --- | Show write-ahead log status |
| `wal truncate` | --- | Clear the write-ahead log |
| `vault init` | --- | Initialize vault from `NEUMANN_VAULT_KEY` environment variable |
| `vault identity 'name'` | --- | Set current identity for vault access control |
| `cache init` | --- | Initialize semantic cache with default configuration |
| `cluster connect` | --- | Connect to cluster with specified node addresses |
| `cluster disconnect` | --- | Disconnect from cluster |

All built-in commands are case-insensitive.

## Environment Variables

| Variable | Description |
| --- | --- |
| `HOME` | Used to locate the default history file (`~/.neumann_history`) |
| `NEUMANN_VAULT_KEY` | Base64-encoded vault encryption key for `vault init` |

## Query Support

The shell supports all query types from the Query Router.

### Relational (SQL)

```sql
CREATE TABLE users (id INT, name TEXT, email TEXT)
INSERT INTO users VALUES (1, 'Alice', 'alice@example.com')
SELECT * FROM users WHERE id = 1
UPDATE users SET name = 'Bob' WHERE id = 1
DELETE FROM users WHERE id = 1
DROP TABLE users
```

### Graph

```sql
NODE CREATE person {name: 'Alice', age: 30}
NODE LIST [label]
NODE GET id
EDGE CREATE node1 -> node2 : label [{props}]
EDGE LIST [type]
EDGE GET id
NEIGHBORS node_id OUTGOING|INCOMING|BOTH [: label]
PATH node1 -> node2 [LIMIT n]
```

### Vector

```sql
EMBED STORE 'key' [vector values]
EMBED GET 'key'
EMBED DELETE 'key'
SIMILAR 'key' [COSINE|EUCLIDEAN|DOT_PRODUCT] LIMIT n
```

### Unified (Cross-Engine)

```sql
FIND NODE [label] [WHERE condition] [LIMIT n]
FIND EDGE [type] [WHERE condition] [LIMIT n]
```

### Blob Storage

```sql
BLOB PUT 'path' [CHUNK size] [TAGS 'a','b'] [FOR 'entity']
BLOB GET 'id' TO 'path'
BLOB DELETE 'id'
BLOB INFO 'id'
BLOB LINK 'id' TO 'entity'
BLOB UNLINK 'id' FROM 'entity'
BLOBS
BLOBS FOR 'entity'
BLOBS BY TAG 'tag'
```

### Vault (Secrets)

```sql
VAULT INIT
VAULT IDENTITY 'node:name'
VAULT SET 'key' 'value'
VAULT GET 'key'
VAULT DELETE 'key'
VAULT LIST 'pattern'
VAULT ROTATE 'key' 'new'
VAULT GRANT 'entity' ON 'key'
VAULT REVOKE 'entity' ON 'key'
```

### Cache (LLM Responses)

```sql
CACHE INIT
CACHE STATS
CACHE CLEAR
CACHE EVICT [n]
CACHE GET 'key'
CACHE PUT 'key' 'value'
```

### Checkpoints (Rollback)

```sql
CHECKPOINT
CHECKPOINT 'name'
CHECKPOINTS
CHECKPOINTS LIMIT n
ROLLBACK TO 'name-or-id'
```

## WAL Commands

### Write Commands Logged to WAL

| Category | Commands |
| --- | --- |
| Relational | `INSERT`, `UPDATE`, `DELETE`, `CREATE`, `DROP` |
| Graph | `NODE CREATE`, `NODE DELETE`, `EDGE CREATE`, `EDGE DELETE` |
| Vector | `EMBED STORE`, `EMBED DELETE` |
| Vault | `VAULT SET`, `VAULT DELETE`, `VAULT ROTATE`, `VAULT GRANT`, `VAULT REVOKE` |
| Cache | `CACHE CLEAR` |
| Blob | `BLOB PUT`, `BLOB DELETE`, `BLOB LINK`, `BLOB UNLINK`, `BLOB TAG`, `BLOB UNTAG`, `BLOB GC`, `BLOB REPAIR`, `BLOB META SET` |

### WAL Data Structure

```rust
struct Wal {
    file: File,    // Open file handle for appending
    path: PathBuf, // Path to WAL file (derived from snapshot: data.bin -> data.log)
}

impl Wal {
    fn open_append(path: &Path) -> std::io::Result<Self>;
    fn append(&mut self, cmd: &str) -> std::io::Result<()>;  // Writes line + flush
    fn truncate(&mut self) -> std::io::Result<()>;           // Recreates empty file
    fn path(&self) -> &Path;
    fn size(&self) -> std::io::Result<u64>;
}
```

### WAL Session Example

```sql
> LOAD 'data.bin'
Loaded snapshot from: data.bin

> INSERT INTO users VALUES (1, 'Alice')
1 row affected

> -- If the shell crashes here, the INSERT is saved in data.log

> -- On next load, the WAL is automatically replayed:
> LOAD 'data.bin'
Loaded snapshot from: data.bin
Replayed 1 commands from WAL

> WAL STATUS
WAL enabled
  Path: data.log
  Size: 42 bytes

> SAVE 'data.bin'
Saved snapshot to: data.bin

> WAL STATUS
WAL enabled
  Path: data.log
  Size: 0 bytes
```

## Persistence Commands

### Save and Load

```sql
> SAVE 'backup.bin'
Saved snapshot to: backup.bin

> SAVE COMPRESSED 'backup_compressed.bin'
Saved compressed snapshot to: backup_compressed.bin

> LOAD 'backup.bin'
Loaded snapshot from: backup.bin
```

**Compression options:**

- `SAVE`: Uncompressed bincode format
- `SAVE COMPRESSED`: Uses int8 quantization (4x smaller), delta encoding, and
  RLE
- `LOAD`: Auto-detects format (works with both compressed and uncompressed)

### Path Extraction

The `extract_path` function handles both quoted and unquoted paths:

- `save 'foo.bin'` yields `Some("foo.bin")`
- `LOAD "bar.bin"` yields `Some("bar.bin")`
- `save /path/to/file.bin` yields `Some("/path/to/file.bin")`
- `save ''` yields `None`
- `save` yields `None`

## Cluster Connectivity

### Connect Command Syntax

```text
CLUSTER CONNECT 'node_id@bind_addr' ['peer_id@peer_addr', ...]
```

**Example:**

```sql
> CLUSTER CONNECT 'node1@127.0.0.1:8001' 'node2@127.0.0.1:8002'
Cluster initialized: node1 @ 127.0.0.1:8001 with 1 peer(s)
```

### Cluster Query Execution

The shell wraps the router for distributed query execution:

```rust
struct RouterExecutor(Arc<RwLock<QueryRouter>>);

impl QueryExecutor for RouterExecutor {
    fn execute(&self, query: &str) -> Result<Vec<u8>, String> {
        let router = self.0.read();
        router.execute_for_cluster(query)
    }
}
```

## Destructive Operation Confirmation

When the checkpoint module is available, the shell presents interactive
confirmation prompts for destructive operations:

| Operation | Warning Message |
| --- | --- |
| `Delete` | `WARNING: About to delete N row(s) from table 'name'` |
| `DropTable` | `WARNING: About to drop table 'name' with N row(s)` |
| `DropIndex` | `WARNING: About to drop index on 'column' in table 'name'` |
| `NodeDelete` | `WARNING: About to delete node N and M connected edge(s)` |
| `EmbedDelete` | `WARNING: About to delete embedding 'key'` |
| `VaultDelete` | `WARNING: About to delete vault secret 'key'` |
| `BlobDelete` | `WARNING: About to delete blob 'id' (size)` |
| `CacheClear` | `WARNING: About to clear cache with N entries` |

## Keyboard Shortcuts

Provided by rustyline:

| Shortcut | Action |
| --- | --- |
| Up/Down | Navigate history |
| Ctrl+C | Cancel current input (prints `^C`, continues loop) |
| Ctrl+D | Exit shell (EOF) |
| Ctrl+L | Clear screen |
| Ctrl+A | Move to start of line |
| Ctrl+E | Move to end of line |
| Ctrl+W | Delete word backward |
| Ctrl+U | Delete to start of line |

## Error Handling

| Error Type | Example | Output Stream |
| --- | --- | --- |
| Parse error | `Error: unexpected token 'FORM' at position 12` | stderr |
| Table not found | `Error: table 'users' not found` | stderr |
| Invalid query | `Error: unsupported operation` | stderr |
| WAL write failure | `Command succeeded but WAL write failed: ...` | Returned as Error |
| WAL replay failure | `WAL replay failed at line N: ...` | Returned as Error |

Errors are printed to stderr and do not exit the shell.

## Performance Characteristics

| Operation | Time |
| --- | --- |
| Empty input | 2.3 ns |
| Help command | 43 ns |
| SELECT (100 rows) | 17.8 us |
| Format 1000 rows | 267 us |

## Dependencies

| Crate | Purpose |
| --- | --- |
| `query_router` | Query execution |
| `relational_engine` | Row type for formatting |
| `tensor_store` | Snapshot persistence (save/load) |
| `tensor_compress` | Compressed snapshot support |
| `tensor_checkpoint` | Checkpoint confirmation handling |
| `tensor_chain` | Cluster query executor trait |
| `rustyline` | Readline functionality (history, shortcuts, Ctrl+C) |
| `parking_lot` | Mutex and RwLock for thread-safe router access |
| `base64` | Vault key decoding |

## Edge Cases and Gotchas

1. **Empty quoted paths**: `save ''` returns an error, not an empty path.

2. **WAL not active by default**: The WAL only becomes active after `LOAD`. New
   shells have no WAL.

3. **Case sensitivity**: Built-in commands are case-insensitive, but query
   strings preserve case for data.

4. **History persistence**: History is only saved when the shell exits normally
   (not on crash).

5. **ANSI codes**: The `clear` command outputs ANSI escape sequences
   (`\x1B[2J\x1B[H`), which may not work on all terminals.

6. **Confirmation handler**: Only active if checkpoint module is available when
   shell starts.

7. **WAL replay stops on first error**: If any command fails during replay, the
   entire replay stops.

8. **Missing columns**: When formatting rows with inconsistent columns, missing
   values show as empty strings.

9. **Binary blob display**: Blobs over 256 bytes or with control characters show
   as `<binary data: N bytes>`.

10. **Timestamp overflow**: Very old timestamps (before 1970) or 0 display as
    "unknown".
