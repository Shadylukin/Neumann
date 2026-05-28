// SPDX-License-Identifier: MIT OR Apache-2.0
//! Statement classification policy used by the dispatcher and the NLQ tool surface.
//!
//! [`classify_statement`] is the security-oriented classification used to decide
//! whether a parsed [`Statement`] may be exposed through a read-only tool
//! surface. [`is_write_statement`] / [`is_cacheable_statement`] are the
//! narrower predicates the cache layer uses to decide invalidation vs.
//! memoization.

use neumann_parser::{
    BlobOp, CacheOp, ChainOp, ClusterOp, EdgeOp, EmbedOp, EntityOp, GraphConstraintOp,
    GraphIndexOp, NodeOp, SpatialOp, Statement, StatementKind, VaultOp,
};
use serde::{Deserialize, Serialize};

/// Safety classification for a parsed statement.
///
/// Used by the NLQ API and web handlers to decide whether a statement may be
/// exposed through a read-only tool surface. This is a **separate policy** from
/// `is_write_statement`, which answers "should this invalidate the query
/// cache?" and intentionally exempts cache ops, checkpoints, and vault
/// operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatementSafety {
    /// Pure read: SELECT, SHOW, DESCRIBE, FIND, SIMILAR, NEIGHBORS, PATH, etc.
    ReadOnly,
    /// Data mutation: INSERT, UPDATE, DELETE, NODE CREATE, EMBED STORE, etc.
    Write,
    /// Structural destruction: DROP TABLE, DROP INDEX, ROLLBACK.
    Destructive,
    /// Identity-sensitive: VAULT, CACHE GET, BLOB content access, CHAIN txn, CLUSTER topology.
    Sensitive,
}

/// Result of checking whether a destructive operation should proceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectedOpResult {
    /// Operation should proceed (confirmed or auto-checkpoint disabled).
    Proceed,
    /// Operation was cancelled by user.
    Cancelled,
}

/// Classify a parsed statement by safety level for the read-only tool surface.
///
/// Every [`StatementKind`] variant and every sub-operation enum is matched
/// explicitly -- no wildcard fallthrough on types with sub-variants.
///
/// # Relationship to `is_write_statement`
///
/// `is_write_statement` answers "should this invalidate the query cache?"
/// and intentionally treats cache ops, checkpoints, and vault reads as
/// non-writes. This function answers "may this be exposed through a read-only
/// tool?" which is a stricter, security-oriented classification.
#[must_use]
#[allow(clippy::match_same_arms)]
pub const fn classify_statement(stmt: &Statement) -> StatementSafety {
    match &stmt.kind {
        // SQL reads
        StatementKind::Select(_)
        | StatementKind::ShowTables
        | StatementKind::ShowEmbeddings { .. }
        | StatementKind::ShowVectorIndex
        | StatementKind::CountEmbeddings
        | StatementKind::Describe(_) => StatementSafety::ReadOnly,
        // SQL writes
        StatementKind::Insert(_)
        | StatementKind::Update(_)
        | StatementKind::Delete(_)
        | StatementKind::CreateTable(_)
        | StatementKind::CreateIndex(_) => StatementSafety::Write,
        // SQL destructive
        StatementKind::DropTable(_) | StatementKind::DropIndex(_) => StatementSafety::Destructive,

        // Graph: simple read-only
        StatementKind::Neighbors(_)
        | StatementKind::Path(_)
        | StatementKind::GraphAlgorithm(_)
        | StatementKind::GraphAggregate(_)
        | StatementKind::GraphPattern(_)
        | StatementKind::CypherMatch(_) => StatementSafety::ReadOnly,
        // Graph: batch/cypher writes
        StatementKind::GraphBatch(_)
        | StatementKind::CypherCreate(_)
        | StatementKind::CypherDelete(_)
        | StatementKind::CypherMerge(_) => StatementSafety::Write,
        // Graph: per-op dispatch
        StatementKind::Node(n) => classify_node_op(&n.operation),
        StatementKind::Edge(e) => classify_edge_op(&e.operation),
        StatementKind::GraphConstraint(gc) => classify_graph_constraint_op(&gc.operation),
        StatementKind::GraphIndex(gi) => classify_graph_index_op(&gi.operation),

        // Vector / spatial / unified
        StatementKind::Similar(_) | StatementKind::Find(_) => StatementSafety::ReadOnly,
        StatementKind::Embed(e) => classify_embed_op(&e.operation),
        StatementKind::Spatial(s) => classify_spatial_op(&s.op),
        StatementKind::Entity(e) => classify_entity_op(&e.operation),

        // Sensitive subsystems
        StatementKind::Vault(_) | StatementKind::Blobs(_) => StatementSafety::Sensitive,
        StatementKind::Cache(c) => classify_cache_op(&c.operation),
        StatementKind::Blob(b) => classify_blob_op(&b.operation),
        StatementKind::Chain(c) => classify_chain_op(&c.operation),
        StatementKind::Cluster(c) => classify_cluster_op(&c.operation),

        // Checkpoint / rollback
        StatementKind::Checkpoints(_) => StatementSafety::ReadOnly,
        StatementKind::Checkpoint(_) => StatementSafety::Write,
        StatementKind::Rollback(_) => StatementSafety::Destructive,

        StatementKind::Empty => StatementSafety::ReadOnly,
    }
}

const fn classify_node_op(op: &NodeOp) -> StatementSafety {
    match op {
        NodeOp::Get { .. } | NodeOp::List { .. } => StatementSafety::ReadOnly,
        NodeOp::Create { .. } | NodeOp::Delete { .. } => StatementSafety::Write,
    }
}

const fn classify_edge_op(op: &EdgeOp) -> StatementSafety {
    match op {
        EdgeOp::Get { .. } | EdgeOp::List { .. } => StatementSafety::ReadOnly,
        EdgeOp::Create { .. } | EdgeOp::Delete { .. } => StatementSafety::Write,
    }
}

const fn classify_graph_constraint_op(op: &GraphConstraintOp) -> StatementSafety {
    match op {
        GraphConstraintOp::List | GraphConstraintOp::Get { .. } => StatementSafety::ReadOnly,
        GraphConstraintOp::Create { .. } => StatementSafety::Write,
        GraphConstraintOp::Drop { .. } => StatementSafety::Destructive,
    }
}

#[allow(clippy::match_same_arms)]
const fn classify_graph_index_op(op: &GraphIndexOp) -> StatementSafety {
    match op {
        GraphIndexOp::ShowNodeIndexes | GraphIndexOp::ShowEdgeIndexes => StatementSafety::ReadOnly,
        GraphIndexOp::CreateNodeProperty { .. }
        | GraphIndexOp::CreateEdgeProperty { .. }
        | GraphIndexOp::CreateLabel
        | GraphIndexOp::CreateEdgeType => StatementSafety::Write,
        GraphIndexOp::DropNode { .. } | GraphIndexOp::DropEdge { .. } => {
            StatementSafety::Destructive
        },
    }
}

const fn classify_embed_op(op: &EmbedOp) -> StatementSafety {
    match op {
        EmbedOp::Get { .. } => StatementSafety::ReadOnly,
        EmbedOp::Store { .. }
        | EmbedOp::Delete { .. }
        | EmbedOp::BuildIndex
        | EmbedOp::Batch { .. } => StatementSafety::Write,
    }
}

const fn classify_spatial_op(op: &SpatialOp) -> StatementSafety {
    match op {
        SpatialOp::WithinRadius { .. } | SpatialOp::Nearest { .. } | SpatialOp::Count => {
            StatementSafety::ReadOnly
        },
        SpatialOp::Insert { .. } | SpatialOp::Delete { .. } => StatementSafety::Write,
    }
}

const fn classify_entity_op(op: &EntityOp) -> StatementSafety {
    match op {
        EntityOp::Get { .. } => StatementSafety::ReadOnly,
        EntityOp::Create { .. }
        | EntityOp::Update { .. }
        | EntityOp::Delete { .. }
        | EntityOp::Connect { .. }
        | EntityOp::Batch { .. } => StatementSafety::Write,
    }
}

/// Cache reads expose LLM prompt/response data (identity-scoped but globally stored).
const fn classify_cache_op(op: &CacheOp) -> StatementSafety {
    match op {
        CacheOp::Stats => StatementSafety::ReadOnly,
        CacheOp::Get { .. } | CacheOp::SemanticGet { .. } => StatementSafety::Sensitive,
        CacheOp::Init
        | CacheOp::Put { .. }
        | CacheOp::SemanticPut { .. }
        | CacheOp::Clear
        | CacheOp::Evict { .. } => StatementSafety::Write,
    }
}

/// Blob content access and artifact enumeration are sensitive.
#[allow(clippy::match_same_arms)]
const fn classify_blob_op(op: &BlobOp) -> StatementSafety {
    match op {
        BlobOp::Stats | BlobOp::Verify { .. } => StatementSafety::ReadOnly,
        BlobOp::Get { .. }
        | BlobOp::Info { .. }
        | BlobOp::Links { .. }
        | BlobOp::MetaGet { .. } => StatementSafety::Sensitive,
        BlobOp::Init
        | BlobOp::Put { .. }
        | BlobOp::Delete { .. }
        | BlobOp::Link { .. }
        | BlobOp::Unlink { .. }
        | BlobOp::Tag { .. }
        | BlobOp::Untag { .. }
        | BlobOp::Gc { .. }
        | BlobOp::Repair
        | BlobOp::MetaSet { .. } => StatementSafety::Write,
    }
}

/// Chain transaction ops are sensitive; introspection is read-only.
#[allow(clippy::match_same_arms)]
const fn classify_chain_op(op: &ChainOp) -> StatementSafety {
    match op {
        ChainOp::History { .. }
        | ChainOp::Similar { .. }
        | ChainOp::Drift { .. }
        | ChainOp::ShowCodebookGlobal
        | ChainOp::ShowCodebookLocal { .. }
        | ChainOp::AnalyzeTransitions
        | ChainOp::Height
        | ChainOp::Tip
        | ChainOp::Block { .. }
        | ChainOp::Verify => StatementSafety::ReadOnly,
        ChainOp::Begin | ChainOp::Commit => StatementSafety::Sensitive,
        ChainOp::Rollback { .. } => StatementSafety::Destructive,
    }
}

const fn classify_cluster_op(op: &ClusterOp) -> StatementSafety {
    match op {
        ClusterOp::Status | ClusterOp::Nodes | ClusterOp::Leader => StatementSafety::ReadOnly,
        ClusterOp::Connect { .. } | ClusterOp::Disconnect => StatementSafety::Sensitive,
    }
}

/// Whether a statement's result may be cached.
///
/// Used by the query cache layer to decide memoization; this is intentionally
/// narrower than [`classify_statement`].
#[must_use]
pub const fn is_cacheable_statement(stmt: &Statement) -> bool {
    matches!(
        &stmt.kind,
        StatementKind::Select(_)
            | StatementKind::Similar(_)
            | StatementKind::Neighbors(_)
            | StatementKind::Path(_)
    )
}

/// Whether a statement mutates state and should invalidate the query cache.
///
/// This intentionally treats cache ops, checkpoint ops, and vault reads as
/// non-writes -- they do not invalidate cached query results. Use
/// [`classify_statement`] for the broader security-oriented classification.
#[must_use]
#[allow(clippy::match_same_arms)] // Arms kept separate for clarity of write-vs-read intent
pub const fn is_write_statement(stmt: &Statement) -> bool {
    match &stmt.kind {
        // SQL writes
        StatementKind::Insert(_)
        | StatementKind::Update(_)
        | StatementKind::Delete(_)
        | StatementKind::CreateTable(_)
        | StatementKind::DropTable(_)
        | StatementKind::CreateIndex(_)
        | StatementKind::DropIndex(_) => true,

        // Graph writes (structural mutations)
        StatementKind::GraphBatch(_)
        | StatementKind::GraphConstraint(_)
        | StatementKind::GraphIndex(_)
        | StatementKind::CypherCreate(_)
        | StatementKind::CypherDelete(_)
        | StatementKind::CypherMerge(_)
        | StatementKind::Rollback(_) => true,

        // Graph node/edge: Create and Delete are writes, Get/List are reads
        StatementKind::Node(n) => {
            matches!(&n.operation, NodeOp::Create { .. } | NodeOp::Delete { .. })
        },
        StatementKind::Edge(e) => {
            matches!(&e.operation, EdgeOp::Create { .. } | EdgeOp::Delete { .. })
        },

        // Vector writes: Store/Delete/Batch mutate, Get/BuildIndex are reads
        StatementKind::Embed(e) => matches!(
            &e.operation,
            EmbedOp::Store { .. } | EmbedOp::Delete { .. } | EmbedOp::Batch { .. }
        ),

        // Spatial writes: Insert/Delete mutate
        StatementKind::Spatial(s) => {
            matches!(&s.op, SpatialOp::Insert { .. } | SpatialOp::Delete { .. })
        },

        // Entity writes: Create/Update/Delete/Connect/Batch mutate
        StatementKind::Entity(e) => matches!(
            &e.operation,
            EntityOp::Create { .. }
                | EntityOp::Update { .. }
                | EntityOp::Delete { .. }
                | EntityOp::Connect { .. }
                | EntityOp::Batch { .. }
        ),

        // Vault: Set/Delete/Rotate mutate; Get/List/Grant/Revoke are reads or ACL
        StatementKind::Vault(v) => matches!(
            &v.operation,
            VaultOp::Set { .. } | VaultOp::Delete { .. } | VaultOp::Rotate { .. }
        ),

        // Blob: Put/Delete mutate content; Link/Unlink/Tag/Untag/MetaSet mutate metadata
        StatementKind::Blob(b) => matches!(
            &b.operation,
            BlobOp::Put { .. }
                | BlobOp::Delete { .. }
                | BlobOp::Link { .. }
                | BlobOp::Unlink { .. }
                | BlobOp::Tag { .. }
                | BlobOp::Untag { .. }
                | BlobOp::MetaSet { .. }
        ),

        // Everything else: reads (SELECT, SHOW, DESCRIBE, etc.), Cache ops
        // (never invalidate -- would break CACHE PUT/GET), and Checkpoint
        _ => false,
    }
}
