// SPDX-License-Identifier: MIT OR Apache-2.0
//! Result types produced by the query router and pagination support.
//!
//! [`QueryResult`] is the unified return type for every dispatcher path.
//! Each variant carries a domain-specific result struct (also defined here)
//! that external crates depend on for JSON/gRPC serialization.

use std::collections::HashMap;
use std::time::Duration;

use relational_engine::Row;
use serde::{Deserialize, Serialize};
use tensor_unified::{UnifiedItem, UnifiedResult as TensorUnifiedResult};

/// Unified query result type used by every dispatcher path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryResult {
    /// No result (e.g., CREATE, INSERT)
    Empty,
    /// Single value result
    Value(String),
    /// Count of affected rows/nodes/edges
    Count(usize),
    /// List of IDs
    Ids(Vec<u64>),
    /// Rows from relational query
    Rows(Vec<Row>),
    /// Node data from graph query
    Nodes(Vec<NodeResult>),
    /// Edge data from graph query
    Edges(Vec<EdgeResult>),
    /// Path from graph traversal
    Path(Vec<u64>),
    /// Vector similarity results
    Similar(Vec<SimilarResult>),
    /// Combined results from unified query
    Unified(UnifiedResult),
    /// List of table names
    TableList(Vec<String>),
    /// Blob data (bytes)
    Blob(Vec<u8>),
    /// Artifact metadata
    ArtifactInfo(ArtifactInfoResult),
    /// List of artifact IDs
    ArtifactList(Vec<String>),
    /// Blob storage statistics
    BlobStats(BlobStatsResult),
    /// List of checkpoints
    CheckpointList(Vec<CheckpointInfo>),
    /// Chain operation result
    Chain(ChainResult),
    /// `PageRank` algorithm results with metadata
    PageRank(PageRankResult),
    /// Centrality algorithm results with metadata
    Centrality(CentralityResult),
    /// Community detection results with metadata
    Communities(CommunityResult),
    /// Constraint list results
    Constraints(Vec<ConstraintInfo>),
    /// Graph index list results
    GraphIndexes(Vec<String>),
    /// Aggregate result (numeric)
    Aggregate(AggregateResultValue),
    /// Batch operation result
    BatchResult(BatchOperationResult),
    /// Pattern match results
    PatternMatch(PatternMatchResultValue),
    /// Spatial range query results
    Spatial(Vec<SpatialResult>),
}

/// Result of a paginated query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedQueryResult {
    /// The query result for the current page.
    pub result: QueryResult,
    /// Cursor token for the next page (None if this is the last page).
    pub next_cursor: Option<String>,
    /// Cursor token for the previous page (None if this is the first page).
    pub prev_cursor: Option<String>,
    /// Total count of results (if known/requested).
    pub total_count: Option<usize>,
    /// Whether there are more results after this page.
    pub has_more: bool,
    /// Number of items in this page.
    pub page_size: usize,
}

/// Options for paginated query execution.
#[derive(Debug, Clone, Default)]
pub struct PaginationOptions {
    /// Cursor token to resume from (None for first page).
    pub cursor: Option<String>,
    /// Number of items per page (default: 100).
    pub page_size: Option<usize>,
    /// Whether to count total results (may be expensive).
    pub count_total: bool,
    /// Custom TTL for the cursor (default: 5 minutes).
    pub cursor_ttl: Option<Duration>,
}

impl PaginationOptions {
    /// Create new pagination options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Resume from a cursor.
    #[must_use]
    pub fn with_cursor(mut self, cursor: String) -> Self {
        self.cursor = Some(cursor);
        self
    }

    /// Set page size.
    #[must_use]
    pub const fn with_page_size(mut self, size: usize) -> Self {
        self.page_size = Some(size);
        self
    }

    /// Enable total count.
    #[must_use]
    pub const fn with_count_total(mut self, count: bool) -> Self {
        self.count_total = count;
        self
    }

    /// Set cursor TTL.
    #[must_use]
    pub const fn with_cursor_ttl(mut self, ttl: Duration) -> Self {
        self.cursor_ttl = Some(ttl);
        self
    }
}

/// Node result from graph query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResult {
    /// Node identifier.
    pub id: u64,
    /// Node label.
    pub label: String,
    /// Node properties as a string-valued map.
    pub properties: HashMap<String, String>,
}

/// Edge result from graph query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeResult {
    /// Edge identifier.
    pub id: u64,
    /// Source node id.
    pub from: u64,
    /// Destination node id.
    pub to: u64,
    /// Edge label / type.
    pub label: String,
}

/// Similarity search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarResult {
    /// Key of the matching embedding.
    pub key: String,
    /// Similarity score.
    pub score: f32,
}

/// Result from a spatial range query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialResult {
    /// Key of the spatial entry.
    pub key: String,
    /// Distance from query point to the entry (edge distance for WITHIN, centroid distance for NEAREST).
    pub distance: f32,
    /// Bounding box x coordinate.
    pub x: f32,
    /// Bounding box y coordinate.
    pub y: f32,
    /// Bounding box width.
    pub width: f32,
    /// Bounding box height.
    pub height: f32,
}

/// Result from unified cross-engine query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedResult {
    /// Human-readable description of the cross-engine result set.
    pub description: String,
    /// Items returned by the unified query.
    pub items: Vec<UnifiedItem>,
}

impl From<TensorUnifiedResult> for UnifiedResult {
    fn from(r: TensorUnifiedResult) -> Self {
        Self {
            description: r.description,
            items: r.items,
        }
    }
}

/// Artifact info result from blob query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactInfoResult {
    /// Artifact identifier.
    pub id: String,
    /// Original filename.
    pub filename: String,
    /// MIME content type.
    pub content_type: String,
    /// Total artifact size in bytes.
    pub size: usize,
    /// Content-addressed checksum.
    pub checksum: String,
    /// Number of chunks the artifact is split into.
    pub chunk_count: usize,
    /// Unix timestamp of creation.
    pub created: u64,
    /// Unix timestamp of last modification.
    pub modified: u64,
    /// Identity that created the artifact.
    pub created_by: String,
    /// Tags applied to this artifact.
    pub tags: Vec<String>,
    /// Entities linked to this artifact.
    pub linked_to: Vec<String>,
    /// Custom user-defined metadata key/value pairs.
    pub custom: HashMap<String, String>,
}

/// Blob storage statistics result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobStatsResult {
    /// Number of artifacts stored.
    pub artifact_count: usize,
    /// Total number of chunks.
    pub chunk_count: usize,
    /// Total bytes referenced by all artifacts.
    pub total_bytes: usize,
    /// Total unique bytes after deduplication.
    pub unique_bytes: usize,
    /// Ratio of total bytes to unique bytes.
    pub dedup_ratio: f64,
    /// Chunks not referenced by any artifact.
    pub orphaned_chunks: usize,
}

/// Checkpoint information for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    /// Unique checkpoint identifier.
    pub id: String,
    /// Human-readable checkpoint name.
    pub name: String,
    /// Unix timestamp of checkpoint creation.
    pub created_at: u64,
    /// Whether this checkpoint was created automatically before a destructive op.
    pub is_auto: bool,
}

/// Chain operation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChainResult {
    /// Transaction begun
    TransactionBegun {
        /// Transaction identifier returned by the chain.
        tx_id: String,
    },
    /// Transaction committed
    Committed {
        /// Block hash containing the committed transaction.
        block_hash: String,
        /// Block height of the commit.
        height: u64,
    },
    /// Chain rolled back
    RolledBack {
        /// Height the chain was rolled back to.
        to_height: u64,
    },
    /// Chain history for a key
    History(Vec<ChainHistoryEntry>),
    /// Similar blocks/transactions
    Similar(Vec<ChainSimilarResult>),
    /// Chain drift metrics
    Drift(ChainDriftResult),
    /// Chain height
    Height(u64),
    /// Chain tip
    Tip {
        /// Hash of the tip block.
        hash: String,
        /// Height of the tip block.
        height: u64,
    },
    /// Block info
    Block(ChainBlockInfo),
    /// Codebook info
    Codebook(ChainCodebookInfo),
    /// Verification result
    Verified {
        /// Whether verification succeeded.
        ok: bool,
        /// Per-block validation errors encountered.
        errors: Vec<String>,
    },
    /// Transition analysis
    TransitionAnalysis(ChainTransitionAnalysis),
}

/// Entry in chain history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainHistoryEntry {
    /// Block height of this entry.
    pub height: u64,
    /// Transaction type recorded in the block.
    pub transaction_type: String,
    /// Optional raw transaction payload bytes.
    pub data: Option<Vec<u8>>,
}

/// Similar result from chain query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainSimilarResult {
    /// Hash of the matching block.
    pub block_hash: String,
    /// Height of the matching block.
    pub height: u64,
    /// Similarity score in `[0.0, 1.0]`.
    pub similarity: f32,
}

/// Chain drift metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainDriftResult {
    /// Starting block height of the measurement window.
    pub from_height: u64,
    /// Ending block height of the measurement window.
    pub to_height: u64,
    /// Sum of per-block drift across the window.
    pub total_drift: f32,
    /// Mean drift per block across the window.
    pub avg_drift_per_block: f32,
    /// Maximum single-block drift observed in the window.
    pub max_drift: f32,
}

/// Block info from chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainBlockInfo {
    /// Block height.
    pub height: u64,
    /// Block hash.
    pub hash: String,
    /// Hash of the previous block.
    pub prev_hash: String,
    /// Block timestamp (Unix epoch).
    pub timestamp: u64,
    /// Number of transactions in the block.
    pub transaction_count: usize,
    /// Identity of the block proposer.
    pub proposer: String,
}

/// Codebook info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainCodebookInfo {
    /// Scope of the codebook (global or per-domain).
    pub scope: String,
    /// Number of codebook entries.
    pub entry_count: usize,
    /// Embedding dimensionality.
    pub dimension: usize,
    /// Optional domain identifier when the codebook is per-domain.
    pub domain: Option<String>,
}

/// Transition analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainTransitionAnalysis {
    /// Total number of block transitions inspected.
    pub total_transitions: usize,
    /// Transitions classified as valid.
    pub valid_transitions: usize,
    /// Transitions classified as invalid.
    pub invalid_transitions: usize,
    /// Average validity score across all transitions.
    pub avg_validity_score: f32,
}

/// `PageRank` score for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankItem {
    /// Identifier of the scored node.
    pub node_id: u64,
    /// `PageRank` score for the node.
    pub score: f64,
}

/// `PageRank` result with algorithm metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankResult {
    /// Per-node scores.
    pub items: Vec<PageRankItem>,
    /// Number of iterations executed.
    pub iterations: usize,
    /// Final convergence delta.
    pub convergence: f64,
    /// Whether the algorithm reached its convergence threshold.
    pub converged: bool,
}

/// Centrality algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CentralityType {
    /// Betweenness centrality.
    Betweenness,
    /// Closeness centrality.
    Closeness,
    /// Eigenvector centrality.
    Eigenvector,
}

/// Centrality score for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityItem {
    /// Identifier of the scored node.
    pub node_id: u64,
    /// Centrality score for the node.
    pub score: f64,
}

/// Centrality result with algorithm metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralityResult {
    /// Per-node scores.
    pub items: Vec<CentralityItem>,
    /// Centrality algorithm used.
    pub centrality_type: CentralityType,
    /// Iterations executed (only for iterative variants such as eigenvector).
    pub iterations: Option<usize>,
    /// Whether the algorithm converged (only for iterative variants).
    pub converged: Option<bool>,
    /// Sample count used (only for sampled variants).
    pub sample_count: Option<usize>,
}

/// Community assignment for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityItem {
    /// Identifier of the node.
    pub node_id: u64,
    /// Community identifier the node belongs to.
    pub community_id: u64,
}

/// Community detection result with algorithm metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityResult {
    /// Per-node community assignments.
    pub items: Vec<CommunityItem>,
    /// Members grouped by community id.
    pub members: HashMap<u64, Vec<u64>>,
    /// Total number of detected communities.
    pub community_count: usize,
    /// Modularity score (only for modularity-based algorithms).
    pub modularity: Option<f64>,
    /// Number of Louvain passes executed (only for Louvain).
    pub passes: Option<usize>,
    /// Iterations executed (only for iterative variants).
    pub iterations: Option<usize>,
}

/// Constraint information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintInfo {
    /// Constraint name.
    pub name: String,
    /// Target element type (node or edge label).
    pub target: String,
    /// Property name the constraint applies to.
    pub property: String,
    /// Constraint type (uniqueness, existence, range, etc.).
    pub constraint_type: String,
}

/// Aggregate result value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregateResultValue {
    /// `COUNT` result.
    Count(u64),
    /// `SUM` result.
    Sum(f64),
    /// `AVG` result.
    Avg(f64),
    /// `MIN` result.
    Min(f64),
    /// `MAX` result.
    Max(f64),
}

/// Batch operation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOperationResult {
    /// Name of the batch operation that produced this result.
    pub operation: String,
    /// Number of rows/nodes/edges affected.
    pub affected_count: usize,
    /// IDs of newly created entities, when applicable.
    pub created_ids: Option<Vec<u64>>,
}

/// Pattern match result value for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatchResultValue {
    /// Matches with per-binding details.
    pub matches: Vec<PatternMatchBinding>,
    /// Statistics about the matching run.
    pub stats: PatternMatchStatsValue,
}

/// A single match with variable bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatchBinding {
    /// Variable -> bound graph element.
    pub bindings: HashMap<String, BindingValue>,
}

/// A binding to a graph element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BindingValue {
    /// Variable bound to a node.
    Node {
        /// Node identifier.
        id: u64,
        /// Node label.
        label: String,
    },
    /// Variable bound to an edge.
    Edge {
        /// Edge identifier.
        id: u64,
        /// Edge type.
        edge_type: String,
        /// Source node id.
        from: u64,
        /// Destination node id.
        to: u64,
    },
    /// Variable bound to a path.
    Path {
        /// Node identifiers along the path, in order.
        nodes: Vec<u64>,
        /// Edge identifiers along the path, in order.
        edges: Vec<u64>,
        /// Number of edges in the path.
        length: usize,
    },
}

/// Statistics from pattern matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMatchStatsValue {
    /// Number of matches returned.
    pub matches_found: usize,
    /// Number of nodes evaluated during search.
    pub nodes_evaluated: usize,
    /// Number of edges evaluated during search.
    pub edges_evaluated: usize,
    /// Whether the match set was truncated due to limits.
    pub truncated: bool,
}

impl QueryResult {
    /// Convert the result to JSON string.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Convert the result to pretty-printed JSON string.
    #[must_use]
    pub fn to_pretty_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Check if the result is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Get the count if this is a Count result.
    #[must_use]
    pub const fn as_count(&self) -> Option<usize> {
        if let Self::Count(n) = self {
            Some(*n)
        } else {
            None
        }
    }

    /// Get the value if this is a Value result.
    #[must_use]
    pub fn as_value(&self) -> Option<&str> {
        if let Self::Value(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Get the rows if this is a Rows result.
    #[must_use]
    pub fn as_rows(&self) -> Option<&[Row]> {
        if let Self::Rows(rows) = self {
            Some(rows)
        } else {
            None
        }
    }
}
