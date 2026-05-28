# Neumann Intelligence Delivery Board

Neumann should feel like a merge of AI and a modern database, not a database
with a chatbot bolted on.

The target state is an intelligent data runtime that:

- understands structure, semantics, and relationships
- plans from natural language into safe executable queries
- executes deterministically through the parser and router
- explains why an answer was returned
- remembers and reuses insights across similar requests

This document turns that direction into a hierarchical kanban board.

## North Star

Neumann should not stop at "natural-language to query translation".

It should become a database with deep intelligence:

- native semantics: embeddings, schema meaning, entity meaning
- cross-engine reasoning: tables, graphs, vectors, blobs, cache, vault
- safe planning: natural language compiled into validated statements
- provenance: every answer grounded in data and query plan
- reusable intelligence: insight cache, saved semantic views, follow-up prompts
- model optionality: local models, OpenAI, Claude, or future adapters

## Product Principles

- The database owns truth; models own planning.
- Parser and router stay authoritative for validation and execution.
- Intelligence should be native to read and write paths, not an afterthought.
- Cross-engine reasoning should be the default path, not a premium feature.
- Every intelligent action should be auditable.
- Hosted and local model support should share one tool contract.

## Existing Foundation

These are already present and should be treated as shipped dependencies, not
new work.

| Ticket | Status | Asset | Why it matters |
| --- | --- | --- | --- |
| `BASE-001` | Done | `neumann_parser::parse(...)` | Gives a parser-first validation boundary for generated queries. |
| `BASE-002` | Done | `QueryRouter` | Provides one execution layer across relational, graph, vector, unified, blob, vault, cache, checkpoint, and chain workloads. |
| `BASE-003` | Done | `QueryService.Execute`, `ExecuteBatch`, `ExecutePaginated`, `CloseCursor`, `ExecuteStream` | Gives a complete gRPC execution surface for intelligent clients. |
| `BASE-004` | Done | `POST /api/galaxy` and `POST /api/execute` | Gives a simple JSON bridge for read-only and full execution flows. |
| `BASE-005` | Done | `FIND`, `ENTITY`, `SIMILAR`, graph traversal, vector search | Gives Neumann the cross-engine primitives an intelligent planner needs. |

## Program Hierarchy

Program: `INT-0 Neumann As Intelligent Database Runtime`

Epics:

- `INT-1 Semantic Substrate`
  - Make semantics native inside Neumann, not external glue.
- `INT-2 Natural-Language Query Runtime`
  - Build the planner, compiler, and execution loop for ChatGPT and Claude.
- `INT-3 Insight And Reasoning Engine`
  - Move from "query answers" to "grounded insights".
- `INT-4 Trust, Safety, And Governance`
  - Make intelligent execution safe, explainable, and auditable.
- `INT-5 Developer Experience And Productization`
  - Make the system usable from APIs, SDKs, demos, and admin tooling.

## Kanban Board

### Done

| Ticket | Parent | Outcome |
| --- | --- | --- |
| `BASE-001` | Foundation | Parser-first validation exists. |
| `BASE-002` | Foundation | Unified router exists. |
| `BASE-003` | Foundation | gRPC execution surface exists. |
| `BASE-004` | Foundation | JSON execution endpoints exist. |
| `BASE-005` | Foundation | Cross-engine query primitives exist. |

### Ready

These can start immediately without waiting on major architecture changes.

| Ticket | Parent | Outcome | Dependencies |
| --- | --- | --- | --- |
| `INT-201` | `INT-2` | Reuse one parsed-statement safety classifier across router, web handlers, and gateway. | `BASE-001`, `BASE-002` |
| `INT-202` | `INT-2` | Add schema inspection endpoints/tools for tables, node labels, edge types, and engine capabilities. | `BASE-001`, `BASE-004` |
| `INT-203` | `INT-2` | Build a thin `nlq-gateway` that accepts natural language and returns validated Neumann queries plus results. | `INT-201`, `INT-202` |
| `INT-204` | `INT-2` | Define one shared tool contract for OpenAI and Claude integrations. | `INT-203` |
| `INT-404` | `INT-4` | Build an evaluation corpus and benchmark harness for NL request -> query -> result correctness. | `BASE-001`, `BASE-002` |

### Next

These should start after the first read-only NLQ slice is working.

| Ticket | Parent | Outcome | Dependencies |
| --- | --- | --- | --- |
| `INT-101` | `INT-1` | Introduce a pluggable embedding provider abstraction with local and hosted adapters. | `BASE-005` |
| `INT-205` | `INT-2` | Add clarification and repair loops when schema names, labels, or IDs are ambiguous. | `INT-202`, `INT-203` |
| `INT-206` | `INT-2` | Add cursor-aware orchestration so models can paginate and continue large result sets safely. | `INT-203`, `BASE-003` |
| `INT-301` | `INT-3` | Return provenance-bearing answer envelopes, not just raw rows or summaries. | `INT-203` |
| `INT-401` | `INT-4` | Build a policy engine for read-only, write, destructive, and sensitive operations. | `INT-201` |
| `INT-402` | `INT-4` | Add confirmation workflow for writes, rollback, vault, chain, and cluster operations. | `INT-401` |
| `INT-403` | `INT-4` | Capture NL request, generated query, execution result, and final answer in an audit trail. | `INT-203`, `INT-401` |

### Later

These are the features that make Neumann feel deeply intelligent rather than
merely "LLM-accessible".

| Ticket | Parent | Outcome | Dependencies |
| --- | --- | --- | --- |
| `INT-102` | `INT-1` | Auto-generate embeddings on insert/update for configured text fields and entities. | `INT-101` |
| `INT-103` | `INT-1` | Build a semantic catalog and schema knowledge graph with descriptions, embeddings, and aliases. | `INT-101`, `INT-202` |
| `INT-104` | `INT-1` | Add persistent semantic profiles for entities, segments, and exemplars such as "best accounts". | `INT-102`, `INT-103` |
| `INT-302` | `INT-3` | Add insight cache with invalidation and provenance, not just response caching. | `INT-301`, `INT-403` |
| `INT-303` | `INT-3` | Add follow-up suggestion and refinement engine based on result structure and prior questions. | `INT-301`, `INT-302` |
| `INT-304` | `INT-3` | Add semantic views and saved insight definitions that mix relational, graph, and vector logic. | `INT-103`, `INT-302` |
| `INT-501` | `INT-5` | Publish OpenAI and Claude sample apps and SDK helpers. | `INT-204`, `INT-206` |
| `INT-502` | `INT-5` | Build a conversational admin console for Neumann. | `INT-203`, `INT-301`, `INT-402` |
| `INT-503` | `INT-5` | Add observability dashboards for NLQ latency, failure reasons, query repair rate, and model cost. | `INT-403`, `INT-404` |

### Open Decisions

These need explicit product calls to avoid rework.

| Decision | Why it matters |
| --- | --- |
| `DEC-001` Default model strategy: local-first vs hosted-first | Changes delivery order for embedding, planner, and privacy work. |
| `DEC-002` Gateway placement: standalone service vs integrated into `neumann_server` | Changes ownership boundaries, deployment model, and SDK surface. |
| `DEC-003` Semantic catalog source: generated only vs generated + curated annotations | Changes how much schema understanding can be trusted in production. |

## Detailed Ticket Breakdown

## `INT-1 Semantic Substrate`

Goal:
Make semantics a first-class part of Neumann's storage and query model.

### Epic Summary

| Ticket | Stage | Summary | Acceptance Criteria |
| --- | --- | --- | --- |
| `INT-101` | Next | Introduce an embedding provider interface that supports local and hosted models behind one Neumann abstraction. | Providers can be swapped by config; model/version metadata is stored; failures surface deterministically. |
| `INT-102` | Later | Auto-embed configured text fields during insert, update, and entity writes. | `INSERT` and `ENTITY CREATE/UPDATE` can generate embeddings automatically for configured fields; write path remains auditable; retries do not duplicate vectors. |
| `INT-103` | Later | Build a semantic catalog for tables, columns, labels, edges, collections, and aliases. | Catalog stores names, descriptions, embeddings, examples, and aliases; fuzzy schema lookup works through one API. |
| `INT-104` | Later | Support semantic profiles for entities and segments such as "best accounts" or "high-risk customers". | Profiles can be stored, updated, and referenced in planning; planner can ground semantic prompts in explicit saved entities or segments. |

## `INT-2 Natural-Language Query Runtime`

Goal:
Turn natural language into safe, validated, executable Neumann workflows.

### Epic Summary

| Ticket | Stage | Summary | Acceptance Criteria |
| --- | --- | --- | --- |
| `INT-201` | Ready | Centralize parsed-statement classification for read, write, destructive, and sensitive operations. | One shared classifier is used by gateway and server paths; tests cover `SELECT`, `FIND`, `SIMILAR`, `ENTITY CONNECT`, `VAULT`, `ROLLBACK`, and `CHAIN`. |
| `INT-202` | Ready | Add schema inspection tools and APIs for intelligent planning. | Gateway can inspect tables, labels, edge types, vector collections, and capabilities; responses are structured and compact enough for model context. |
| `INT-203` | Ready | Build the first `nlq-gateway` with request intake, planning, validation, execution, and summarization. | Gateway accepts NL input, calls inspection/execution tools, validates through parser, and returns result plus final generated query. |
| `INT-204` | Ready | Define one shared tool contract for ChatGPT and Claude. | OpenAI and Anthropic use the same logical tools: inspect schema, execute query, continue cursor, close cursor; only adapter code differs. |
| `INT-205` | Next | Add clarification and repair loops for ambiguous or invalid requests. | Gateway asks targeted follow-ups when names are unclear; parse failures trigger bounded repair attempts before asking the user. |
| `INT-206` | Next | Add cursor-aware orchestration for large results. | Gateway can request first page, continue with cursor, and close cursor; model outputs mention when results are partial or paginated. |

## `INT-3 Insight And Reasoning Engine`

Goal:
Move Neumann from query execution to grounded reasoning and reusable insights.

### Epic Summary

| Ticket | Stage | Summary | Acceptance Criteria |
| --- | --- | --- | --- |
| `INT-301` | Next | Return a provenance-bearing answer envelope instead of only raw query output. | Every intelligent answer includes executed query, result type, source engine hints, and optional confidence/provenance metadata. |
| `INT-302` | Later | Build an insight cache with invalidation and provenance. | Similar questions can reuse prior derived insights; invalidation is tied to source data changes; cached insights remain inspectable. |
| `INT-303` | Later | Add a follow-up suggestion engine. | Responses can propose grounded next questions such as exploring a segment, path, or anomaly; suggestions are based on result structure, not generic text. |
| `INT-304` | Later | Add semantic views and saved insights. | Users can persist an intelligent query pattern as a named asset; semantic views can mix relational filters, graph relations, and vector similarity. |

## `INT-4 Trust, Safety, And Governance`

Goal:
Make intelligence safe enough for production systems.

### Epic Summary

| Ticket | Stage | Summary | Acceptance Criteria |
| --- | --- | --- | --- |
| `INT-401` | Next | Build a policy engine that classifies operations by risk and required controls. | Policies can mark statements as read-only, write, destructive, or identity-required; gateway enforces policy before execution. |
| `INT-402` | Next | Add confirmation workflows for writes and sensitive operations. | Destructive or privileged operations require explicit confirmation; confirmation state is recorded and tied to the executed query. |
| `INT-403` | Next | Log the full NLQ lifecycle. | System stores the original request, planning steps, generated query, execution result, and final answer for audit and debugging. |
| `INT-404` | Ready | Build an evaluation harness and benchmark corpus. | There is a repeatable suite for schema lookup accuracy, query validity, safety classification, answer grounding, and repair rate. |

## `INT-5 Developer Experience And Productization`

Goal:
Make the intelligent database usable from real products and teams.

### Epic Summary

| Ticket | Stage | Summary | Acceptance Criteria |
| --- | --- | --- | --- |
| `INT-501` | Later | Publish OpenAI and Claude sample integrations. | Repo contains minimal working examples for both vendors using the same tool contract and Neumann execution loop. |
| `INT-502` | Later | Build a conversational admin console. | Admin UI supports asking questions, viewing generated queries, stepping through pagination, and approving writes. |
| `INT-503` | Later | Add NLQ observability dashboards. | Product exposes latency, token/model usage, parse failure rate, repair rate, approval rate, and cache reuse metrics. |

## Milestones

These milestones create a sane build order.

### `M1 Ask And Read`

Scope:

- `INT-201`
- `INT-202`
- `INT-203`
- `INT-204`
- `INT-404`

Outcome:

- ChatGPT and Claude can ask read-only questions against Neumann and get
  grounded answers back through one tool contract.

### `M2 Safe Mutations`

Scope:

- `INT-205`
- `INT-206`
- `INT-401`
- `INT-402`
- `INT-403`

Outcome:

- Natural language can safely drive writes and sensitive operations with
  confirmation, audit, and pagination control.

### `M3 Native Semantics`

Scope:

- `INT-101`
- `INT-102`
- `INT-103`
- `INT-104`

Outcome:

- Neumann gains internal semantic understanding rather than depending on
  external embedding pipelines and weak schema hints.

### `M4 Insight Runtime`

Scope:

- `INT-301`
- `INT-302`
- `INT-303`
- `INT-304`
- `INT-501`
- `INT-502`
- `INT-503`

Outcome:

- Neumann becomes an intelligent analytical runtime, not just an NLQ front end.

## Implementation Anchors

The first implementation should stay close to what already exists:

- parser remains the validation boundary
- `QueryRouter` remains the execution engine
- read-only execution should prefer `/api/galaxy` or `ExecutePaginated`
- write execution should prefer `/api/execute` or `Execute`
- hosted and local models should share one logical tool surface

## Bottom Line

If the product goal is "Neumann with natural-language queries", the current
plan is enough.

If the product goal is "Neumann as a fusion of AI and modern databases", the
real work is broader:

- native semantics
- safe planning
- provenance
- reusable insights
- policy and audit
- product surfaces that make intelligence operational

That is the backlog above.
