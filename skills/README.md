# Neumann Skills Pack

AI coding assistant skills for Claude Code, Codex, and Gemini CLI.

The Neumann Skills Pack teaches AI assistants how to work with the Neumann
tensor runtime -- its query language, graph and vector engines, schema
conventions, deployment patterns, and troubleshooting procedures.

## Skills

| Skill | Description |
|-------|-------------|
| `neumann-query` | Write and optimize Neumann queries across relational, graph, and vector engines |
| `neumann-schema` | Design tensor schemas, indexes, and storage layouts for Neumann workloads |
| `neumann-graph` | Build and traverse graph structures using Neumann's directed graph engine |
| `neumann-vector` | Configure HNSW indexes and perform similarity search with multiple distance metrics |
| `neumann-client` | Integrate applications with Neumann using the gRPC client SDK |
| `neumann-deploy` | Deploy, configure, and scale Neumann clusters with Raft consensus |
| `neumann-migrate` | Plan and execute schema migrations, snapshots, and data transformations |
| `neumann-troubleshoot` | Diagnose performance issues, consensus failures, and storage anomalies |

## Install

### Remote (recommended)

```bash
curl -sSfL https://raw.githubusercontent.com/Shadylukin/Neumann/main/skills/install.sh | bash
```

### Local (from a clone)

```bash
git clone https://github.com/Shadylukin/Neumann.git
cd Neumann
bash skills/install.sh
```

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `NEUMANN_SKILLS_SCOPE` | `project`, `global` | `project` | Install into the current project or the user home directory |
| `NEUMANN_SKILLS_TOOLS` | `claude`, `codex`, `gemini`, `all` | auto-detect | Comma-separated list of target tools |
| `NEUMANN_SKILLS_REF` | tag or SHA | latest release | Pin installation to a specific Git ref |

Examples:

```bash
# Install globally for all tools
NEUMANN_SKILLS_SCOPE=global NEUMANN_SKILLS_TOOLS=all bash skills/install.sh

# Install only Claude Code skills at a pinned version
NEUMANN_SKILLS_TOOLS=claude NEUMANN_SKILLS_REF=v1.0.0 bash skills/install.sh
```

## Uninstall

```bash
bash skills/uninstall.sh
```

The uninstaller finds all Neumann skill installations via their manifest files
and removes them cleanly. It is idempotent -- safe to run multiple times.

## Compatibility

- **Claude Code** -- project-level (`.claude/skills/`) and global (`~/.claude/skills/`)
- **Codex** -- global only (`~/.codex/skills/`)
- **Gemini CLI** -- project-level (`.gemini/skills/`) and global (`~/.gemini/skills/`)
