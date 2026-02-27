# Documentation Standards

This document defines the rigid formatting rules for all documentation in the
Neumann project. All contributions must adhere to these standards.

## General Formatting

### Line Length

- **Maximum line length**: 80 characters
- **Exceptions**: Code blocks, tables, and headings are exempt
- **Rationale**: Ensures readability in terminals and side-by-side diffs

### Headings

- Use ATX-style headings (`#`, `##`, `###`)
- Increment heading levels by one (no skipping from `#` to `###`)
- Leave one blank line before and after headings
- Do not end headings with punctuation

```markdown
# Top Level

## Second Level

### Third Level
```

### Lists

- Use dashes (`-`) for unordered lists
- Use two-space indentation for nested lists
- Leave one blank line before and after list blocks

```markdown
- First item
  - Nested item
  - Another nested item
- Second item
```

### Emphasis

- Use asterisks for *italic* (`*italic*`)
- Use double asterisks for **bold** (`**bold**`)
- Do not use underscores for emphasis

### Code Blocks

- Use fenced code blocks with triple backticks
- **Always specify the language** for syntax highlighting
- Supported languages: `rust`, `bash`, `toml`, `json`, `sql`, `text`

````markdown
```rust
fn example() -> Result<()> {
    Ok(())
}
```
````

### Tables

- Align columns with pipes and padding
- Use header separators with at least three dashes
- Tables are exempt from line length limits

```markdown
| Column A | Column B | Column C |
| -------- | -------- | -------- |
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |
```

## Diagrams

### Mermaid

- Use `flowchart` directive (not deprecated `graph`)
- Keep diagrams simple and focused
- Add descriptive node labels

```markdown
```mermaid
flowchart TD
    A[Start] --> B{Decision}
    B -->|Yes| C[Action]
    B -->|No| D[End]
```

```bash

### ASCII Diagrams

Use ASCII diagrams for wire formats and data structures:

```text
+--------+--------+--------+
| Header | Length | Data   |
| 4B     | 4B     | N bytes|
+--------+--------+--------+
```

## Document Types (Divio System)

Documentation follows the [Divio documentation system](https://docs.divio.com/documentation-system/)
with four quadrants. Each page belongs to exactly one quadrant. Do not mix
content types within a single page.

### Tutorials (learning-oriented)

Location: `docs/book/src/tutorials/`

Purpose: Teach a beginner by walking them through a concrete project.

Required sections:

1. **Prerequisites** - What the reader needs before starting
2. **Step N** - Numbered steps with copy-pasteable commands
3. **Verification** - How to confirm success
4. **Next Steps** - Links to related tutorials and how-to guides

Rules: Concrete actions, visible results after each step, minimal
explanation. Link to Explanation pages for "why" content.

### How-to Guides (goal-oriented)

Location: `docs/book/src/how-to/`

Purpose: Help an experienced user accomplish a specific task.

Required sections:

1. **Goal** or clear title stating what the guide achieves
2. **Steps** or **code examples** showing how to accomplish it

Rules: Assume the reader knows the basics. Link to Reference for
config tables and to Explanation for design rationale.

### Runbooks

Location: `docs/book/src/how-to/runbooks/`

Required sections:

1. **Symptoms** - How to identify the issue
2. **Diagnostic Steps** - Commands to diagnose the problem
3. **Resolution** - Step-by-step fix procedure
4. **Prevention** - How to avoid the issue

### Reference (information-oriented)

Location: `docs/book/src/reference/` and `docs/book/src/reference/api/`

Purpose: Describe the machinery accurately and completely.

Required sections (for API references):

1. **See Also** - Links to related explanation and how-to pages
2. **Types** - Tables of types, fields, variants
3. **Error Types** - Error variants with causes

Rules: Tables, not prose. Accurate, complete, up to date. Do not
include tutorials or explanations. Link to Explanation for "why".

### Explanation (understanding-oriented)

Location: `docs/book/src/explanation/`

Purpose: Help the reader understand design decisions and internals.

Required sections:

1. **Overview** or introduction paragraph
2. **How It Works** - Technical explanation with diagrams

Rules: Discuss alternatives, trade-offs, and design rationale. Use
diagrams. Link to Reference for precise specifications and to How-to
for practical usage.

### Cross-Reference Policy

- Tutorials link to Explanation (for "why") and How-to (for next steps)
- How-to guides link to Reference (for config tables) and Explanation
  (for design rationale)
- Reference pages link to Explanation and How-to via "See Also"
- Explanation pages link to Reference (for precise specs) and How-to
  (for practical usage)

## Validation

### Pre-commit Hook

The pre-commit hook validates:

- Markdownlint compliance
- Code block language specifiers
- Mermaid deprecated directive warnings

### CI Checks

The documentation workflow validates:

- All markdownlint rules
- Required sections by document type
- Link integrity with mdbook-linkcheck

### Running Locally

```bash
# Install markdownlint-cli
npm install -g markdownlint-cli

# Check all documentation
npx markdownlint-cli "docs/**/*.md" "*.md"

# Run validation script
./scripts/validate-docs.sh

# Build and preview book
cd docs/book && mdbook serve
```

## Style Guide

### Tone

- Use active voice
- Be concise and direct
- Avoid jargon without explanation
- No emojis in documentation

### Terminology

Use consistent terminology throughout:

| Term | Definition |
| --- | --- |
| Node | A single instance in the cluster |
| Leader | The node handling write requests |
| Follower | A node replicating from the leader |
| Quorum | Majority of nodes required for consensus |
| Transaction | A unit of work with ACID properties |
| Workspace | Isolated transaction state container |
| Delta | Difference between before/after states |
| Embedding | Vector representation of data |

### Cross-References

- Use relative links for internal references
- Include file extensions in links

```markdown
See [Transaction Workspace](../explanation/transaction-workspace.md) for
details on workspace lifecycle.
```

## Checklist

Before submitting documentation:

- [ ] Line length under 80 characters (except code/tables)
- [ ] All code blocks have language specifiers
- [ ] Headings use ATX style and increment properly
- [ ] Lists use dashes with two-space nesting
- [ ] Tables are aligned with pipe padding
- [ ] Mermaid diagrams use `flowchart` not `graph`
- [ ] Required sections present for document type
- [ ] All links resolve correctly
- [ ] No emojis or excessive formatting
