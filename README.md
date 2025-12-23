Neumann
What is Neumann?
Neumann is a unified runtime that stores relational data, graph relationships, and vector embeddings in a single mathematical structure — the tensor. Instead of spinning up Postgres, Neo4j, Qdrant, and Redis separately, you spin up Neumann.
Code and data live together. The same system that stores your tables also understands how your functions call each other, what depends on what, and what's semantically similar to what.
Files are exports. The tensor is truth.
What problem does it solve?
Modern development requires too many moving pieces. A simple project might need:

A relational database for structured data
A graph database for relationships
A vector database for semantic search
A cache layer for performance
Version control for code

Each has its own query language, its own mental model, its own operational overhead. Teams spend more time on infrastructure than on their actual problem.
Neumann collapses this. One runtime. One shell. One way of thinking about information.
For vibe coders and small teams, this means getting started in minutes instead of hours. For AI-native applications, it means code, data, and embeddings are naturally co-located — no glue code, no sync problems.
Core concepts
Tensor: The underlying mathematical structure. Just as von Neumann showed that code and data are both patterns in memory, a tensor can represent a scalar, a vector, a matrix, a table, a graph, or a higher-dimensional structure. The storage format is unified even when the query patterns differ.
Shell: The interface. Simple CLI commands to create, query, and manipulate data. Designed for humans first.
Three query patterns, one substrate:

Relational: Tables, rows, columns, joins. Postgres-style.
Graph: Nodes, edges, traversals, path-finding. Neo4j-style.
Vector: Embeddings, similarity search, nearest neighbors. Qdrant-style.

All three operate on the same underlying tensor. A node in your graph can have relational properties and a vector embedding. A row in your table can have graph relationships. The boundaries dissolve.
Code as data: When you point Neumann at a codebase, it doesn't just store files — it understands structure. Functions, dependencies, call graphs. This enables queries like "what would break if I changed this?" without leaving the same system that holds your application data.
What does a user do with it?
$ neumann init myproject
$ neumann ingest ./src              # code becomes queryable
$ neumann create table users        # relational
$ neumann link user:1 -> post:5     # graph  
$ neumann embed user:1 "semantic description"  # vector
$ neumann query "users connected to posts similar to X"  # unified
The shell is the primary interface. Everything is a command. State lives in the tensor.
What Neumann is not (for now)

Not a distributed system. Single-node, in-memory first. Durability and clustering come later.
Not a replacement for production Postgres at scale. It's for development, prototyping, small-to-medium workloads, and AI-native applications.
Not a full IDE or code editor. It stores and queries code structure, but you still write code elsewhere.

Why "Neumann"?
John von Neumann unified code and data in the stored-program architecture. Sixty years later, we've re-fragmented them into separate systems. Neumann finishes the thought.