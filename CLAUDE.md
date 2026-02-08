# CLAUDE.md

Neumann is a unified tensor-based runtime that stores relational data, graph
relationships, and vector embeddings in a single mathematical structure.
19 crates, ~400K+ lines of Rust (with tests), built on macOS (MacBook Air M4).

## Quality Gates -- YOU MUST FOLLOW THESE EVERY TIME

All code MUST pass these checks. Do not commit, do not mark work as done, do not move on until all four pass cleanly:

```bash
cargo fmt --check
cargo clippy --package <crate> -- -D warnings -D clippy::pedantic
cargo nextest run --package <crate>
cargo doc --no-deps
```

IMPORTANT: When clippy or pedantic reports warnings, you MUST fix them
properly -- not suppress them with `#[allow(...)]` unless there is a
genuine, documented reason (e.g., a false positive on a trait impl).
Fixing means rewriting the code to satisfy the lint. If you are unsure
whether an allow is justified, ask.

Pre-commit hook auto-runs `cargo fmt` on staged files.

### Coverage -- 95% Minimum

After making changes, YOU MUST check coverage and ensure it meets the threshold:

```bash
cargo llvm-cov --package <crate> --lcov --output-path /tmp/coverage.lcov
lcov --extract /tmp/coverage.lcov '*/<crate>/src/*' --output-file /tmp/crate_only.lcov
lcov --summary /tmp/crate_only.lcov
```

Thresholds: 95% default, neumann_shell 88%, tensor_blob 91%, query_router 92%.

If coverage drops below the threshold, you MUST add tests to bring it
back up before finishing. This is non-negotiable. Write targeted tests
for uncovered paths -- do not write dummy tests that inflate numbers
without testing real behavior.

## IMPORTANT: Things That Go Wrong

- NEVER use `unsafe` code unless absolutely necessary and well-justified
- tensor_chain's `raft.rs` is 7,684 lines -- always use targeted edits, never rewrite large sections
- When adding new engine types or query kinds, you MUST update `query_router`'s match arms
- When modifying `TensorValue` or `ScalarValue` variants, update ALL match arms across crates -- these are used everywhere
- The `tensor_store` slab types (relational_slab, graph_tensor, embedding_slab, etc.)
  each have their own locking -- do not wrap them in additional locks
- Snapshot format has versions (V2/V3) in `tensor_store/src/snapshot.rs` -- new formats must be backward-compatible
- `tensor_chain` TCP transport uses length-prefixed framing -- any protocol changes must update both encode and decode paths
- `BlobStore` is async-first -- do not use blocking I/O in tensor_blob

## Workflow -- ALWAYS FOLLOW THIS SEQUENCE

After EVERY set of changes, run these steps in order. Do not skip any step:

1. `cargo fmt` -- format the code
2. `cargo clippy --package <crate> -- -D warnings -D clippy::pedantic` -- fix ALL warnings properly (no blanket allows)
3. `cargo nextest run --package <crate>` -- all tests must pass
4. Check coverage (see above) -- must meet threshold, add tests if it doesn't
5. For cross-crate changes, also run `cargo nextest run --package integration_tests`
6. Commit with clear imperative messages, no emoji, reference issues when applicable

Do NOT tell me "all checks pass" unless you have actually run them and
they passed. If a clippy lint is hard to fix, show me the lint and your
proposed fix before applying an allow attribute.

## Workspace Structure

19 crates organized in dependency tiers:

### Foundation Layer (no workspace dependencies)

| Crate | Purpose |
| ----- | ------- |
| `tensor_store` | Core key-value storage with HNSW, sparse vectors, tiered storage |
| `tensor_compress` | Tensor Train decomposition, delta encoding, RLE compression |
| `neumann_parser` | Hand-written recursive descent parser for query language |

### Engine Layer (depends on tensor_store)

| Crate | Purpose |
| ----- | ------- |
| `relational_engine` | SQL-like tables with SIMD filtering, indexes, columnar scans |
| `graph_engine` | Directed graphs with BFS traversal, shortest path, properties |
| `vector_engine` | k-NN similarity search via HNSW with multiple distance metrics |

### Specialized Storage Layer

| Crate | Purpose |
| ----- | ------- |
| `tensor_vault` | AES-256-GCM encrypted secrets with graph-based access control |
| `tensor_cache` | Multi-layer LLM response cache (exact + semantic + embedding) |
| `tensor_blob` | S3-style content-addressable blob storage with streaming (async) |
| `tensor_checkpoint` | Atomic snapshot/restore with retention and confirmation |
| `tensor_unified` | Cross-engine unified entity operations |

### Distributed Layer

| Crate | Purpose |
| ----- | ------- |
| `tensor_chain` | Tensor-native blockchain with Raft consensus and 2PC |

### Query Execution Layer

| Crate | Purpose |
| ----- | ------- |
| `query_router` | Unified query routing across all engines |
| `neumann_shell` | Interactive CLI with readline, WAL, snapshots |
| `neumann_server` | gRPC server exposing QueryRouter |

### Testing

| Crate | Purpose |
| ----- | ------- |
| `integration_tests` | Cross-crate integration tests (267+ tests) |
| `stress_tests` | Performance and concurrency stress tests |
| `experiments` | Research and experimental features |
| `seed_model` | Geometric intelligence model implementation |

## Core Types

These are the load-bearing types across the workspace:

- `TensorValue`: Scalar | Vector | Sparse | Pointer | Pointers -- the fundamental value enum used everywhere
- `ScalarValue`: Null | Bool | Int | Float | String | Bytes
- `TensorData`: HashMap-based entity with field accessors
- `TensorStore`: Thread-safe key-value store with ShardedBTreeMap via SlabRouter
- `EntityStore`: High-level entity abstraction with type-aware scanning
- `SparseVector`: Memory-efficient sparse embedding (15+ distance metrics)
- `HNSWIndex`: Hierarchical navigable small world graph (Dense/Sparse/Delta/TT variants)
- `QueryRouter`: Unified query execution -- dispatches to all engines
- `QueryResult`: Result variants (Rows, Nodes, Edges, Similar, Chain, etc.)
- `RaftNode`: Tensor-Raft consensus state machine in tensor_chain
- `DistributedTxCoordinator`: 2PC with lock manager and deadlock detection

For full type reference per crate, inspect the crate's `src/lib.rs` public API.

## Concurrency Design

All engines inherit thread safety from TensorStore's SlabRouter:

- SlabRouter uses sharded BTreeMaps -- writes only block same-shard writes
- Uses dashmap/parking_lot instead of std locks (no lock poisoning)
- When adding new concurrent data structures: prefer sharded/partitioned designs, always add concurrent tests (`store_concurrent_*`)

## Comments Policy

Doc comments (`///`) are required on all public items: structs, enums,
enum variants, traits, functions, methods, constants, and type aliases.
Keep them concise -- one line for simple items, a short paragraph for
complex algorithms or non-obvious behavior. Use `# Errors` sections
for fallible methods.

Exceptions: trivial `Default` impls and re-exports documented at their
origin crate.

Inline comments (`//`) explain "why", never "what".

The workspace enforces per-crate doc coverage thresholds via CI.
Run `scripts/check-doc-coverage.sh` locally to verify.

## Testing Philosophy

- Unit tests in same file (`#[cfg(test)]` module), test public API not internals
- Test names: `test_<function>_<scenario>_<expected>`
- Include edge cases: empty inputs, boundaries, error conditions
- Performance tests for operations that must scale (10k+ entities)
- Concurrent tests for thread-safe code
- Fuzz targets exist for parser, compression, crypto, storage -- see `docs/fuzzing.md`

## Code Style

- No emojis in code, comments, or commit messages
- Proper error handling using `Result` and `?` propagation
- Keep functions small and focused
- Prefer composition over inheritance patterns

## Git Identity

All commits attributed to: ShadyL <lukinainz@gmail.com> (GitHub: Shadylukin)

IMPORTANT: Do NOT add "Co-Authored-By" lines, AI attribution, or any
reference to AI tools (Claude, Copilot, ChatGPT, etc.) in commit
messages, PR descriptions, code comments, or any other artifact checked
into the repository. Code should be judged on its own merit.

## Further Reference

- Architecture: `docs/architecture.md`
- Fuzzing: `docs/fuzzing.md`
- Dev tools setup: `docs/dev-tools.md`

## Development Tools

This project uses modern CLI tools for development, testing, and code quality. All tools are installed via Homebrew or Cargo.

### Quick Install

```bash
# Tier 1: Essential tools (install first)
brew install eza zoxide fzf git-delta hyperfine starship tldr lcov tokei jq ripgrep fd bat
cargo install cargo-watch cargo-outdated cargo-expand cargo-nextest cargo-audit cargo-tarpaulin

# Tier 2: High-value tools
brew install bottom dust duf procs lazygit difftastic sd act actionlint yamllint

# Tier 3: Specialized tools
brew install ast-grep grex onefetch glow xh
```

### Tool Categories

#### Code Quality & Coverage

| Tool | Purpose | Usage |
|------|---------|-------|
| `lcov` | Coverage report analysis | `lcov --summary file.lcov` |
| `cargo-tarpaulin` | Rust code coverage | `cargo tarpaulin --out Html` |
| `cargo-audit` | Security vulnerability scanner | `cargo audit` |
| `cargo-clippy` | Rust linter | `cargo clippy -- -D warnings` |

#### Code Search & Navigation

| Tool | Purpose | Speed | Usage |
|------|---------|-------|-------|
| `ripgrep` (rg) | Fast code search | 88x faster than grep | `rg "pattern"` |
| `fd` | Fast file finder | 5x faster than find | `fd "pattern"` |
| `ast-grep` | AST-based code search | Structure-aware | `ast-grep -p 'fn $NAME($$$)'` |
| `fzf` | Fuzzy finder | Interactive | `Ctrl+R` for history |

#### File Operations

| Tool | Purpose | Replaces | Usage |
|------|---------|----------|-------|
| `eza` | Modern file listing | `ls` | `eza -la --git` |
| `bat` | Syntax-highlighted viewer | `cat` | `bat file.rs` |
| `sd` | Better find/replace | `sed` | `sd 'old' 'new' file` |
| `dust` | Disk usage analyzer | `du` | `dust -d 3` |
| `duf` | Disk free viewer | `df` | `duf` |

#### System Monitoring

| Tool | Purpose | Replaces | Usage |
|------|---------|----------|-------|
| `bottom` (btm) | Interactive system monitor | `top`/`htop` | `btm` |
| `procs` | Process viewer | `ps` | `procs` |

#### Git Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `git-delta` | Syntax-aware diffs | `git diff \| delta` |
| `lazygit` | Terminal Git UI | `lazygit` |
| `difftastic` | Structural diffs | `git difftool` |
| `gh` | GitHub CLI | `gh pr create` |

#### Rust Development

| Tool | Purpose | Usage |
|------|---------|-------|
| `cargo-watch` | Auto-rebuild on changes | `cargo watch -x test` |
| `cargo-nextest` | Next-gen test runner | `cargo nextest run` |
| `cargo-expand` | Macro expansion | `cargo expand` |
| `cargo-outdated` | Dependency updates | `cargo outdated` |
| `hyperfine` | Benchmarking | `hyperfine 'cargo test'` |

#### Developer Experience

| Tool | Purpose | Usage |
|------|---------|-------|
| `starship` | Beautiful shell prompt | `eval "$(starship init zsh)"` |
| `zoxide` | Smart directory jumping | `z neumann` |
| `tldr` | Quick command examples | `tldr cargo` |
| `glow` | Markdown renderer | `glow README.md` |
| `tokei` | Code statistics | `tokei src/` |
| `onefetch` | Git repo stats | `onefetch` |

#### Data Processing

| Tool | Purpose | Usage |
|------|---------|-------|
| `jq` | JSON parsing | `jq '.field' file.json` |
| `yq` | YAML/JSON processing | `yq '.field' file.yaml` |
| `xh` | HTTP client | `xh GET api.example.com` |
| `grex` | Regex generator | `grex "example1" "example2"` |

#### CI/CD Testing

| Tool | Purpose | Usage |
|------|---------|-------|
| `act` | Run GitHub Actions locally | `act -l` (list), `act -j jobname` (run job) |
| `actionlint` | Lint GitHub Actions workflows | `actionlint .github/workflows/*.yml` |
| `yamllint` | YAML quality checker | `yamllint .github/` |

### AI-Powered Development Tools

For AI-assisted coding, these tools are available (optional):

#### Local LLM Runtime

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run deepseek-r1:7b          # Fast reasoning model
ollama run qwen2.5-coder:7b        # Code-specialized
ollama run llama3.2:3b             # General purpose
```

- [Ollama GitHub](https://github.com/ollama/ollama) | [Tutorial](https://dev.to/proflead/complete-ollama-tutorial-2026-llms-via-cli-cloud-python-3m97)

#### CLI AI Assistants

- **[AIChat](https://github.com/sigoden/aichat)** - All-in-one LLM CLI (OpenAI, Claude, Gemini, Ollama, Groq)
- **[Aider](https://aider.chat/)** - AI pair programming in terminal
- **[Cline](https://research.aimultiple.com/agentic-cli/)** - Autonomous coding agent (open-source)
- **Claude Code** - Anthropic's official CLI

#### References

- [Best AI Coding Tools 2026](https://www.builder.io/blog/best-ai-tools-2026)
- [CLI AI Coding Agents Comparison](https://www.scriptbyai.com/best-cli-ai-coding-agents/)
- [AI Code Review Tools](https://www.qodo.ai/blog/best-ai-code-review-tools-2026/)

### Coverage Workflow

```bash
# Generate coverage report for a crate
cargo llvm-cov --package <crate_name> --lcov --output-path /tmp/coverage.lcov

# Extract only the crate's source files
lcov --extract /tmp/coverage.lcov '*/crate_name/src/*' --output-file /tmp/crate_only.lcov

# View summary
lcov --summary /tmp/crate_only.lcov

# Generate HTML report
genhtml /tmp/crate_only.lcov -o /tmp/coverage_html
```

### Recommended Shell Aliases

```bash
# Modern CLI replacements
alias ls='eza'
alias ll='eza -l'
alias la='eza -la'
alias tree='eza --tree'
alias cat='bat'
alias find='fd'
alias grep='rg'
alias sed='sd'
alias du='dust'
alias df='duf'
alias ps='procs'
alias top='btm'

# Cargo shortcuts
alias cw='cargo watch'
alias ct='cargo nextest run'
alias cc='cargo clippy -- -D warnings'
alias cf='cargo fmt'
alias cb='cargo build --release'
alias cov='cargo llvm-cov --package'

# Git shortcuts
alias lg='lazygit'
alias gd='git diff | delta'
```
