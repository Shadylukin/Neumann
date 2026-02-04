# LUKIN ACKROYD

**Auckland, NZ** | 027 588 7298 | lukinack@gmail.com | [LinkedIn](https://linkedin.com/in/lukinackroyd) | [GitHub](https://github.com/lukinackroyd)

---

## Professional Summary

Technology professional with 7+ years delivering enterprise solutions across banking,
government, defense, and energy sectors. I design and build systems that solve real
operational problems—from military asset management to nationwide access control systems.

Currently building Neumann, a 353K-line Rust distributed database with semantic consensus—
developed in 5 weeks by orchestrating multiple AI coding agents under strict quality gates.
This demonstrates both systems architecture depth and practical AI-augmented development
at scale.

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

### Neumann — Unified Tensor-Native Distributed Database (Dec 2024–Present)

A production-grade distributed database runtime in Rust (353K lines, 25 crates, 9,500+ tests)
unifying relational, graph, and vector storage with semantic consensus. Built in 5 weeks
using AI-augmented development: architecting modular systems, directing 4-5 coding agents
in parallel sprints, and integrating outputs under strict automated quality gates (clippy
pedantic, 95%+ coverage, 108 fuzz targets).

**Distributed Consensus (tensor_chain — 63K lines):**

- Implemented Tensor-Raft: modified Raft consensus with geometric optimization where
  blocks with >95% cosine similarity bypass full validation (40-60% latency reduction)
- Built 2-Phase Commit coordinator with DFS-based deadlock detection
- Designed 6-way semantic conflict classification (Orthogonal, Identical, Opposite,
  Conflicting, etc.) enabling automatic merge of safe concurrent edits
- Created delta-compressed replication achieving 4-6x bandwidth savings using Tensor Train decomposition
- Implemented SWIM-based gossip protocol with LWW-CRDT semantics for cluster membership

**Storage & Indexing:**

- Built HNSW approximate nearest neighbor search with 9 distance metrics (Cosine, Angular,
  Geodesic, Jaccard, Euclidean, etc.)
- Implemented sparse vector support, delta encoding with k-means clustering, and product quantization
- Designed tiered hot/cold storage with automatic migration based on access patterns
- Created Voronoi and k-means semantic partitioning for vector-aware sharding

**Query Engines:**

- Relational engine with SIMD-accelerated filtering, B-tree/hash indexing, and MVCC
  transactions
- Graph engine with BFS/DFS traversal, shortest path, PageRank, betweenness centrality, A-star
- Hand-written recursive descent parser (1.9M queries/sec) supporting SQL, Cypher-like, and unified syntax

**Security & Caching:**

- AES-256-GCM encrypted vault with Argon2id key derivation and graph-based access control
- Three-layer LLM response cache (exact + semantic + embedding) with tiktoken token counting

**Performance:** 3.2M PUT ops/sec | 5M GET ops/sec | 7.5M concurrent writes/sec @ 8 threads

**Quality Gates:** 95%+ coverage enforced | 108 fuzz targets | Clippy pedantic + nursery |
Mandatory doc comments for logic auditing

**Methodology:** Modular architecture enabling parallel agent development. Kanban per module,
1-2 day sprints, real-time observation. Comments as human-readable audit trail; linters
and coverage as automated verification. Integration once modules pass gates.

**Technologies:** Rust | async/await | SIMD (wide crate) | gRPC/tonic | DashMap/parking_lot | PyO3 bindings

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
