# LUKIN ACKROYD

**Auckland, NZ** | 027 588 7298 | lukinack@gmail.com | [LinkedIn](https://linkedin.com/in/lukinackroyd) | [GitHub](https://github.com/lukinackroyd)

---

## Professional Summary

Technology professional with 7+ years delivering enterprise solutions across banking,
government, defense, and energy sectors. I design and build systems that solve real
operational problems—from military asset management to nationwide access control systems.

Currently building Neumann, a 460K-line Rust distributed database with a novel geometric
consensus layer — transactions encoded as vectors, conflicts classified by cosine similarity,
enabling automatic merging of safe concurrent edits. 21 crates, 11,700+ tests, 139 fuzz
targets, 95%+ enforced coverage.

Strong track record in ITSM platforms (ServiceNow, Cherwell, Ivanti, Jira), cloud migrations,
and security implementations. Calm communicator who delivers in both small agile teams
and large multi-vendor programmes.

---

## Core Skills

**Systems Programming:** Rust (async/await, SIMD, zero-copy) | Python | TypeScript/JavaScript | SQL

**Distributed Systems:** Raft consensus | 2PC | CRDT | Gossip protocols | Consistent hashing | Replication

**Data & AI:** HNSW/ANN search | Vector embeddings | Graph algorithms | LLM orchestration | RAG pipelines

**Platforms & Tools:** ServiceNow | Cherwell | Ivanti | Jira | ManageEngine (ITSM/ITOM/ITAM)

**Cloud & Infrastructure:** AWS | Azure | Docker | Linux | CI/CD | gRPC/Protobuf

**Security & Compliance:** ISO 27001 | ASD Essential Eight | IAM | SIEM | Five Eyes | AES-256-GCM | Argon2

**Methods:** ITIL | Agile | Waterfall | SAFe | Solution design | Process automation

**AI-Augmented Development:** Multi-agent orchestration | Modular architecture for
parallelism | Automated quality gates | Comment-based logic auditing

---

## Highlight Project

### Neumann — Unified Tensor-Native Distributed Database (Dec 2025–Present)

Distributed database in Rust (460K lines, 21 crates, 11,700+ tests) unifying relational
tables, graph traversal, and vector similarity search in a single query engine. Novel
contribution: a geometric consensus layer that encodes transactions as vectors and classifies
conflicts by cosine similarity, enabling automatic merging of non-conflicting concurrent edits.

**Distributed Consensus** *(tensor_chain — 79K lines):*

- Designed Tensor-Raft: transactions encoded as vectors; semantically similar operations
  (>95% cosine similarity) take a fast-path that bypasses full conflict resolution
- 6-way conflict classification — Orthogonal (auto-merge), LowConflict (merge with validation),
  Identical (deduplicate), Opposite (cancel), Ambiguous/Conflicting (reject) — replacing
  coarse-grained locking with fine-grained semantic analysis
- 2-Phase Commit coordinator with DFS-based deadlock detection on a wait-for graph
- Delta-compressed replication using Tensor Train decomposition for bandwidth reduction
- SWIM gossip protocol with LWW-CRDT semantics for cluster membership

**Storage & Indexing:**

- HNSW approximate nearest neighbor with 10 distance metrics including Poincare hyperbolic
  distance; product quantization and IVF for memory-efficient search at scale
- Sparse vector support with delta encoding and k-means clustering
- Tiered hot/cold storage with access-pattern-driven migration; Voronoi-based semantic
  partitioning for vector-aware sharding

**Query Engines:**

- Relational engine with SIMD-accelerated filtering (wide crate), B-tree and hash indexes
- Graph engine: BFS/DFS, Dijkstra shortest path, A*, PageRank, betweenness centrality
- Hand-written recursive descent parser benchmarked at 1.9M queries/sec (Criterion);
  supports SQL, Cypher-like graph syntax, and unified multi-engine queries

**Security & Caching:**

- AES-256-GCM encrypted vault with Argon2id key derivation and graph-based access control
- Three-layer LLM response cache: exact match, semantic similarity, embedding proximity

**Quality:** 95%+ line coverage enforced per crate | 139 fuzz targets | clippy pedantic |
mandatory doc coverage thresholds

**Technologies:** Rust | async/await | SIMD | gRPC (tonic) | DashMap/parking_lot | PyO3 Python bindings

---

## Professional Experience

### Solution Architect | Bluechip IT, Auckland

**October 2023 – January 2025**

- Led cloud migrations for banking and government clients, improving performance and
  compliance while reducing operational risk
- Implemented enterprise security solutions: IAM, SIEM, and automated threat detection
  aligned to ASD Essential Eight and ISO 27001
- Deployed ManageEngine ITSM/ITOM/ITAM solutions, streamlining asset governance and security workflows
- **Z Energy (largest client acquisition):** Designed and delivered a distributed Asset
  Management and Just-In-Time access system for nationwide network of pumps, gas stations,
  and trucking fleet. Enabled temporary, role-based contractor access across sites. Secured
  the deal and led delivery end-to-end
- Trained client IT teams on CI/CD, cloud security, and automation strategies
- Delivered within small 5–6 person teams; owned outcomes from scoping through deployment

### Business Solutions Designer | Service Dynamics, Auckland

**April 2021 – October 2023**

*NZ Defence Force project*

- Engineered a full-scale Military Asset Management system in Cherwell, including a custom
  risk matrix for system-of-systems assets (aircraft, naval vessels, facilities)
- Integrated geo-mapped security tracking—buildings, floors, clearance zones—to dynamically assess and visualize risk
- Implemented Five Eyes security protocols ensuring asset tracking met intelligence-sharing cybersecurity standards
- Developed innovative onboarding methods for complex military assets with multi-dimensional risk profiles
- Created training materials and documentation; delivered workshops to onboard military teams into new workflows
- Collaborated directly with NZDF stakeholders in a small, focused team

**Responsibilities across projects:**

- Client consultation and needs assessment through workshops and discovery sessions
- Solution design including security, API integration, automation, reporting, dashboards, and portals
- Hands-on implementation: configuration, coding, testing, deployment
- Ongoing support and client communication through regular stand-ups

### Senior Consultant | Aurec, Melbourne

**July 2018 – March 2020**

Managed professional services consultancy delivering software and technical solutions
across government, banking, insurance, shipping, and FMCG.

- **Bureau of Meteorology:** Led security hardening after a state-sponsored cyberattack.
  Implemented ASD Essential Eight and ISO 27001 controls. Built SIEM monitoring, threat
  intelligence, and post-breach response frameworks
- Contributed to an **$800M government asset sale**, assisting with business negotiations,
  IT transition planning, and AWS migration execution
- Worked directly with incoming CEOs and executive teams to ensure IT and business
  integration
- Facilitated large-scale cloud migration projects, leading scalability and risk discussions
- Built lasting client relationships across diverse industries

### Senior Consultant | Finite920, Auckland

**March 2020 – August 2020**

Recruitment and technical consulting role cut short by COVID-19 pandemic.

- Engaged with clients to assess hiring needs and workforce challenges
- Contributed to solution design for security, cloud, and data projects
- Worked across government, insurance, banking, and Telco sectors

### Earlier Experience

**2007 – 2018** — Various technical and consulting roles. Details available on request.

---

## Education

**Dev Academy Aotearoa** — Full-Stack Web Development Bootcamp
*September 2020 – December 2020*
JavaScript/TypeScript, React, Node.js, SQL, Flask exposure

**Massey University, Auckland** — Computer Science (Incomplete)
*2012 – 2015*

---

## Certifications

- **ITIL Specialist: Create, Deliver & Support (CDS)** — PeopleCert — April 2023
- **Ivanti Certified Administrator** — Ivanti — March 2022
- **Cherwell Certified Professional – Engineer (Associate)** — May 2021
- **Cherwell Certified Professional – Support** — May 2021
- **ITIL Foundation** — AXELOS — April 2021

---

## Ongoing Learning

Self-directed study in systems design and data engineering since 2009. Current deep dives:
distributed consensus (Raft, CRDTs), approximate nearest neighbor algorithms (HNSW, IVF,
PQ), and high-performance Rust (SIMD, zero-copy, async runtimes).

---

## Speaking & Media

- Regular speaker and trainer for technical and executive audiences
- Featured guest — Endpoint Pulse "Partner Stories" #33 (ManageEngine)
- Bluechip Infotech NZ Newsletter (May 2024): Cloud Security Summit Auckland; Shield 2024 IAM & Cybersecurity Seminar
