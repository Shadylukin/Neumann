# Neumann Database Docker Image
# Multi-stage build with separate CLI and server targets

# ==============================================================================
# Builder Stage
# ==============================================================================
FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace configuration first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Copy all crate Cargo.toml files
COPY tensor_store/Cargo.toml tensor_store/
COPY tensor_compress/Cargo.toml tensor_compress/
COPY tensor_blob/Cargo.toml tensor_blob/
COPY tensor_cache/Cargo.toml tensor_cache/
COPY tensor_vault/Cargo.toml tensor_vault/
COPY tensor_checkpoint/Cargo.toml tensor_checkpoint/
COPY tensor_chain/Cargo.toml tensor_chain/
COPY tensor_unified/Cargo.toml tensor_unified/
COPY relational_engine/Cargo.toml relational_engine/
COPY graph_engine/Cargo.toml graph_engine/
COPY vector_engine/Cargo.toml vector_engine/
COPY query_router/Cargo.toml query_router/
COPY neumann_parser/Cargo.toml neumann_parser/
COPY neumann_shell/Cargo.toml neumann_shell/
COPY neumann_server/Cargo.toml neumann_server/
COPY neumann_client/Cargo.toml neumann_client/
COPY neumann_docs/Cargo.toml neumann_docs/
COPY integration_tests/Cargo.toml integration_tests/
COPY stress_tests/Cargo.toml stress_tests/
COPY examples/Cargo.toml examples/

# Create dummy lib.rs files for dependency caching
RUN mkdir -p tensor_store/src && echo "pub fn dummy() {}" > tensor_store/src/lib.rs && \
    mkdir -p tensor_compress/src && echo "pub fn dummy() {}" > tensor_compress/src/lib.rs && \
    mkdir -p tensor_blob/src && echo "pub fn dummy() {}" > tensor_blob/src/lib.rs && \
    mkdir -p tensor_cache/src && echo "pub fn dummy() {}" > tensor_cache/src/lib.rs && \
    mkdir -p tensor_vault/src && echo "pub fn dummy() {}" > tensor_vault/src/lib.rs && \
    mkdir -p tensor_checkpoint/src && echo "pub fn dummy() {}" > tensor_checkpoint/src/lib.rs && \
    mkdir -p tensor_chain/src && echo "pub fn dummy() {}" > tensor_chain/src/lib.rs && \
    mkdir -p tensor_unified/src && echo "pub fn dummy() {}" > tensor_unified/src/lib.rs && \
    mkdir -p relational_engine/src && echo "pub fn dummy() {}" > relational_engine/src/lib.rs && \
    mkdir -p graph_engine/src && echo "pub fn dummy() {}" > graph_engine/src/lib.rs && \
    mkdir -p vector_engine/src && echo "pub fn dummy() {}" > vector_engine/src/lib.rs && \
    mkdir -p query_router/src && echo "pub fn dummy() {}" > query_router/src/lib.rs && \
    mkdir -p neumann_parser/src && echo "pub fn dummy() {}" > neumann_parser/src/lib.rs && \
    mkdir -p neumann_shell/src && echo "fn main() {}" > neumann_shell/src/main.rs && \
    mkdir -p neumann_server/src && echo "fn main() {}" > neumann_server/src/main.rs && \
    mkdir -p neumann_client/src && echo "pub fn dummy() {}" > neumann_client/src/lib.rs && \
    mkdir -p neumann_docs/src && echo "pub fn dummy() {}" > neumann_docs/src/lib.rs && \
    mkdir -p integration_tests/src && echo "pub fn dummy() {}" > integration_tests/src/lib.rs && \
    mkdir -p stress_tests/src && echo "pub fn dummy() {}" > stress_tests/src/lib.rs && \
    mkdir -p examples/src && echo "pub fn dummy() {}" > examples/src/lib.rs

# Build dependencies only (this layer is cached)
RUN cargo build --release --package neumann_shell 2>/dev/null || true
RUN cargo build --release --package neumann_server 2>/dev/null || true

# Remove dummy files
RUN find . -name "*.rs" -path "*/src/*" -delete

# Copy actual source code
COPY tensor_store/src tensor_store/src
COPY tensor_compress/src tensor_compress/src
COPY tensor_blob/src tensor_blob/src
COPY tensor_cache/src tensor_cache/src
COPY tensor_vault/src tensor_vault/src
COPY tensor_checkpoint/src tensor_checkpoint/src
COPY tensor_chain/src tensor_chain/src
COPY tensor_unified/src tensor_unified/src
COPY relational_engine/src relational_engine/src
COPY graph_engine/src graph_engine/src
COPY vector_engine/src vector_engine/src
COPY query_router/src query_router/src
COPY neumann_parser/src neumann_parser/src
COPY neumann_shell/src neumann_shell/src
COPY neumann_server/src neumann_server/src
COPY neumann_server/proto neumann_server/proto
COPY neumann_server/build.rs neumann_server/
COPY neumann_client/src neumann_client/src
COPY neumann_docs/src neumann_docs/src

# Touch files to ensure rebuild
RUN find . -name "*.rs" -exec touch {} \;

# Build release binaries
RUN cargo build --release --package neumann_shell
RUN cargo build --release --package neumann_server

# ==============================================================================
# CLI Runtime Stage
# ==============================================================================
FROM debian:bookworm-slim AS cli

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash neumann

COPY --from=builder /build/target/release/neumann /usr/local/bin/neumann

USER neumann
WORKDIR /home/neumann

ENTRYPOINT ["neumann"]

# ==============================================================================
# Server Runtime Stage
# ==============================================================================
FROM debian:bookworm-slim AS server

RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash neumann

# Create data directory
RUN mkdir -p /var/lib/neumann && chown neumann:neumann /var/lib/neumann

COPY --from=builder /build/target/release/neumann_server /usr/local/bin/neumann_server

USER neumann
WORKDIR /var/lib/neumann

# gRPC port
EXPOSE 9200

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9200/health || exit 1

ENTRYPOINT ["neumann_server"]
