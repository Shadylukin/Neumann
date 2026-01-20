# Documentation

## Documentation Structure

Neumann documentation consists of:

1. **mdBook** (`docs/book/`) - Conceptual docs, tutorials, operations
2. **rustdoc** - API reference generated from source
3. **README.md** per crate - Quick overview

## Writing mdBook Pages

### File Location

```text
docs/book/src/
├── SUMMARY.md          # Table of contents
├── introduction.md     # Landing page
├── getting-started/    # Tutorials
├── architecture/       # Module deep dives
├── concepts/           # Cross-cutting concepts
├── operations/         # Deployment, monitoring
└── contributing/       # Contribution guides
```

### Page Structure

```markdown
# Page Title

Brief introduction (1-2 paragraphs).

## Section 1

Content with examples.

### Subsection

More detail.

## Section 2

Use tables for structured data:

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |

Use mermaid for diagrams:

\`\`\`mermaid
flowchart LR
    A --> B --> C
\`\`\`
```

### Admonitions

Use mdbook-admonish syntax:

```markdown
```admonish note
This is a note.
```

```admonish warning
This is a warning.
```

```admonish danger
This is dangerous.
```

```bash

## Writing Rustdoc

### Module Documentation

```rust
//! # Module Name
//!
//! Brief description (one line).
//!
//! ## Overview
//!
//! Longer explanation of purpose and design decisions.
//!
//! ## Example
//!
//! ```rust
//! // Example code
//! ```
```

### Type Documentation

```rust
/// Brief description of the type.
///
/// Longer explanation if needed.
///
/// # Example
///
/// ```rust
/// let value = MyType::new();
/// ```
pub struct MyType { ... }
```

### When to Document

Document:

- All public types
- Non-obvious behavior
- Complex algorithms

Don't document:

- Self-explanatory methods (`get`, `set`, `new`)
- Trivial implementations

## Building Documentation

### mdBook

```bash
cd docs/book
mdbook build
mdbook serve  # Local preview at localhost:3000
```

### rustdoc

```bash
cargo doc --workspace --no-deps --open
```

### Full Build

```bash
# mdBook
cd docs/book && mdbook build

# rustdoc
cargo doc --workspace --no-deps

# Combine
cp -r target/doc docs/book-output/api/
```

## Link Checking

```bash
cd docs/book
mdbook-linkcheck --standalone
```

## Adding Mermaid Diagrams

Supported diagram types:

- `flowchart` - Flow diagrams
- `sequenceDiagram` - Sequence diagrams
- `stateDiagram-v2` - State machines
- `classDiagram` - Class diagrams
- `gantt` - Gantt charts

Example:

```markdown
\`\`\`mermaid
sequenceDiagram
    participant C as Client
    participant S as Server
    C->>S: Request
    S->>C: Response
\`\`\`
```
