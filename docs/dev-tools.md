# Development Tools

This project uses modern CLI tools for development, testing, and code quality. All tools are installed via Homebrew or Cargo.

## Quick Install

```bash
# Tier 1: Essential tools (install first)
brew install eza zoxide fzf git-delta hyperfine starship tldr lcov tokei jq ripgrep fd bat
cargo install cargo-watch cargo-outdated cargo-expand cargo-nextest cargo-audit cargo-tarpaulin

# Tier 2: High-value tools
brew install bottom dust duf procs lazygit difftastic sd act actionlint yamllint

# Tier 3: Specialized tools
brew install ast-grep grex onefetch glow xh
```

## Tool Categories

### Code Quality & Coverage

| Tool | Purpose | Usage |
|------|---------|-------|
| `lcov` | Coverage report analysis | `lcov --summary file.lcov` |
| `cargo-tarpaulin` | Rust code coverage | `cargo tarpaulin --out Html` |
| `cargo-audit` | Security vulnerability scanner | `cargo audit` |
| `cargo-clippy` | Rust linter | `cargo clippy -- -D warnings` |

### Code Search & Navigation

| Tool | Purpose | Speed | Usage |
|------|---------|-------|-------|
| `ripgrep` (rg) | Fast code search | 88x faster than grep | `rg "pattern"` |
| `fd` | Fast file finder | 5x faster than find | `fd "pattern"` |
| `ast-grep` | AST-based code search | Structure-aware | `ast-grep -p 'fn $NAME($$$)'` |
| `fzf` | Fuzzy finder | Interactive | `Ctrl+R` for history |

### File Operations

| Tool | Purpose | Replaces | Usage |
|------|---------|----------|-------|
| `eza` | Modern file listing | `ls` | `eza -la --git` |
| `bat` | Syntax-highlighted viewer | `cat` | `bat file.rs` |
| `sd` | Better find/replace | `sed` | `sd 'old' 'new' file` |
| `dust` | Disk usage analyzer | `du` | `dust -d 3` |
| `duf` | Disk free viewer | `df` | `duf` |

### System Monitoring

| Tool | Purpose | Replaces | Usage |
|------|---------|----------|-------|
| `bottom` (btm) | Interactive system monitor | `top`/`htop` | `btm` |
| `procs` | Process viewer | `ps` | `procs` |

### Git Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `git-delta` | Syntax-aware diffs | `git diff \| delta` |
| `lazygit` | Terminal Git UI | `lazygit` |
| `difftastic` | Structural diffs | `git difftool` |
| `gh` | GitHub CLI | `gh pr create` |

### Rust Development

| Tool | Purpose | Usage |
|------|---------|-------|
| `cargo-watch` | Auto-rebuild on changes | `cargo watch -x test` |
| `cargo-nextest` | Next-gen test runner | `cargo nextest run` |
| `cargo-expand` | Macro expansion | `cargo expand` |
| `cargo-outdated` | Dependency updates | `cargo outdated` |
| `hyperfine` | Benchmarking | `hyperfine 'cargo test'` |

### Developer Experience

| Tool | Purpose | Usage |
|------|---------|-------|
| `starship` | Beautiful shell prompt | `eval "$(starship init zsh)"` |
| `zoxide` | Smart directory jumping | `z neumann` |
| `tldr` | Quick command examples | `tldr cargo` |
| `glow` | Markdown renderer | `glow README.md` |
| `tokei` | Code statistics | `tokei src/` |
| `onefetch` | Git repo stats | `onefetch` |

### Data Processing

| Tool | Purpose | Usage |
|------|---------|-------|
| `jq` | JSON parsing | `jq '.field' file.json` |
| `yq` | YAML/JSON processing | `yq '.field' file.yaml` |
| `xh` | HTTP client | `xh GET api.example.com` |
| `grex` | Regex generator | `grex "example1" "example2"` |

### CI/CD Testing

| Tool | Purpose | Usage |
|------|---------|-------|
| `act` | Run GitHub Actions locally | `act -l` (list), `act -j jobname` (run job) |
| `actionlint` | Lint GitHub Actions workflows | `actionlint .github/workflows/*.yml` |
| `yamllint` | YAML quality checker | `yamllint .github/` |

## AI-Powered Development Tools

For AI-assisted coding, these tools are available (optional):

### Local LLM Runtime

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run deepseek-r1:7b          # Fast reasoning model
ollama run qwen2.5-coder:7b        # Code-specialized
ollama run llama3.2:3b             # General purpose
```

- [Ollama GitHub](https://github.com/ollama/ollama) | [Tutorial](https://dev.to/proflead/complete-ollama-tutorial-2026-llms-via-cli-cloud-python-3m97)

### CLI AI Assistants

- **[AIChat](https://github.com/sigoden/aichat)** - All-in-one LLM CLI (OpenAI, Claude, Gemini, Ollama, Groq)
- **[Aider](https://aider.chat/)** - AI pair programming in terminal
- **[Cline](https://research.aimultiple.com/agentic-cli/)** - Autonomous coding agent (open-source)
- **Claude Code** - Anthropic's official CLI

### References

- [Best AI Coding Tools 2026](https://www.builder.io/blog/best-ai-tools-2026)
- [CLI AI Coding Agents Comparison](https://www.scriptbyai.com/best-cli-ai-coding-agents/)
- [AI Code Review Tools](https://www.qodo.ai/blog/best-ai-code-review-tools-2026/)

## Coverage Workflow

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

## Recommended Shell Aliases

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
