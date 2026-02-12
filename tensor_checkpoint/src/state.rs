// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Full checkpoint state including the serialized store snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    /// Unique identifier for this checkpoint.
    pub id: String,
    /// Human-readable name (user-provided or auto-generated).
    pub name: String,
    /// Unix timestamp (seconds) when the checkpoint was created.
    pub created_at: u64,
    /// The destructive operation that triggered this checkpoint, if any.
    pub trigger: Option<CheckpointTrigger>,
    /// Serialized `TensorStore` snapshot bytes.
    pub store_snapshot: Vec<u8>,
    /// Summary of what the store contained at checkpoint time.
    pub metadata: CheckpointMetadata,
}

impl CheckpointState {
    /// Create a new checkpoint state with the current timestamp.
    pub fn new(
        id: String,
        name: String,
        store_snapshot: Vec<u8>,
        metadata: CheckpointMetadata,
    ) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            id,
            name,
            created_at,
            trigger: None,
            store_snapshot,
            metadata,
        }
    }

    /// Attach a trigger describing the destructive operation that caused this checkpoint.
    pub fn with_trigger(mut self, trigger: CheckpointTrigger) -> Self {
        self.trigger = Some(trigger);
        self
    }
}

/// Records which destructive operation triggered an auto-checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointTrigger {
    /// The original command text that triggered the checkpoint.
    pub command: String,
    /// The classified destructive operation.
    pub operation: DestructiveOp,
    /// Preview of what the operation would affect.
    pub preview: OperationPreview,
}

impl CheckpointTrigger {
    /// Create a new checkpoint trigger.
    pub fn new(command: String, operation: DestructiveOp, preview: OperationPreview) -> Self {
        Self {
            command,
            operation,
            preview,
        }
    }
}

/// Classifies the type of destructive operation for checkpoint and confirmation purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DestructiveOp {
    /// Delete rows from a relational table.
    Delete {
        /// Target table name.
        table: String,
        /// Number of rows to delete.
        row_count: usize,
    },
    /// Drop an entire relational table.
    DropTable {
        /// Target table name.
        table: String,
        /// Number of rows in the table.
        row_count: usize,
    },
    /// Drop an index on a table column.
    DropIndex {
        /// Target table name.
        table: String,
        /// Indexed column name.
        column: String,
    },
    /// Delete a graph node and its connected edges.
    NodeDelete {
        /// ID of the node to delete.
        node_id: u64,
        /// Number of edges that will be removed.
        edge_count: usize,
    },
    /// Delete a graph edge.
    EdgeDelete {
        /// ID of the edge to delete.
        edge_id: u64,
    },
    /// Delete a vector embedding.
    EmbedDelete {
        /// Key of the embedding to delete.
        key: String,
    },
    /// Delete a vault secret.
    VaultDelete {
        /// Key of the secret to delete.
        key: String,
    },
    /// Delete a blob artifact.
    BlobDelete {
        /// Content-addressable artifact ID.
        artifact_id: String,
        /// Size of the blob in bytes.
        size: usize,
    },
    /// Clear all entries from the cache.
    CacheClear {
        /// Number of cache entries to be cleared.
        entry_count: usize,
    },
}

impl DestructiveOp {
    /// Returns a short uppercase label for this operation type (e.g. `"DELETE"`, `"DROP TABLE"`).
    pub fn operation_name(&self) -> &'static str {
        match self {
            Self::Delete { .. } => "DELETE",
            Self::DropTable { .. } => "DROP TABLE",
            Self::DropIndex { .. } => "DROP INDEX",
            Self::NodeDelete { .. } => "NODE DELETE",
            Self::EdgeDelete { .. } => "EDGE DELETE",
            Self::EmbedDelete { .. } => "EMBED DELETE",
            Self::VaultDelete { .. } => "VAULT DELETE",
            Self::BlobDelete { .. } => "BLOB DELETE",
            Self::CacheClear { .. } => "CACHE CLEAR",
        }
    }

    /// Returns the total number of entities affected by this operation.
    pub fn affected_count(&self) -> usize {
        match self {
            Self::Delete { row_count, .. } => *row_count,
            Self::DropTable { row_count, .. } => *row_count,
            Self::DropIndex { .. } => 1,
            Self::NodeDelete { edge_count, .. } => 1 + edge_count,
            Self::EdgeDelete { .. } => 1,
            Self::EmbedDelete { .. } => 1,
            Self::VaultDelete { .. } => 1,
            Self::BlobDelete { .. } => 1,
            Self::CacheClear { entry_count } => *entry_count,
        }
    }
}

/// Preview of a destructive operation shown to the user before confirmation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationPreview {
    /// Human-readable summary of what will happen.
    pub summary: String,
    /// Sample of affected data keys/rows (truncated to configured sample size).
    pub sample_data: Vec<String>,
    /// Total number of entities that will be affected.
    pub affected_count: usize,
}

impl OperationPreview {
    /// Create a preview with a summary, sample data, and affected count.
    pub fn new(summary: String, sample_data: Vec<String>, affected_count: usize) -> Self {
        Self {
            summary,
            sample_data,
            affected_count,
        }
    }

    /// Create an empty preview with just a message and zero affected items.
    pub fn empty(message: &str) -> Self {
        Self {
            summary: message.to_string(),
            sample_data: Vec::new(),
            affected_count: 0,
        }
    }
}

/// Metadata captured at checkpoint time summarizing what the store contained.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointMetadata {
    /// Relational engine statistics.
    pub relational: RelationalMeta,
    /// Graph engine statistics.
    pub graph: GraphMeta,
    /// Vector engine statistics.
    pub vector: VectorMeta,
    /// Total number of keys in the underlying `TensorStore`.
    pub store_key_count: usize,
}

impl CheckpointMetadata {
    /// Create metadata from per-engine statistics and a total key count.
    pub fn new(
        relational: RelationalMeta,
        graph: GraphMeta,
        vector: VectorMeta,
        store_key_count: usize,
    ) -> Self {
        Self {
            relational,
            graph,
            vector,
            store_key_count,
        }
    }
}

/// Relational engine statistics at checkpoint time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RelationalMeta {
    /// Number of relational tables.
    pub table_count: usize,
    /// Total row count across all tables.
    pub total_rows: usize,
}

impl RelationalMeta {
    /// Create relational metadata with the given table and row counts.
    pub fn new(table_count: usize, total_rows: usize) -> Self {
        Self {
            table_count,
            total_rows,
        }
    }
}

/// Graph engine statistics at checkpoint time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphMeta {
    /// Number of graph nodes.
    pub node_count: usize,
    /// Number of graph edges.
    pub edge_count: usize,
}

impl GraphMeta {
    /// Create graph metadata with the given node and edge counts.
    pub fn new(node_count: usize, edge_count: usize) -> Self {
        Self {
            node_count,
            edge_count,
        }
    }
}

/// Vector engine statistics at checkpoint time.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorMeta {
    /// Number of stored vector embeddings.
    pub embedding_count: usize,
}

impl VectorMeta {
    /// Create vector metadata with the given embedding count.
    pub fn new(embedding_count: usize) -> Self {
        Self { embedding_count }
    }
}

/// Lightweight checkpoint descriptor returned by list operations.
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    /// Unique checkpoint identifier.
    pub id: String,
    /// Human-readable checkpoint name.
    pub name: String,
    /// Unix timestamp (seconds) when created.
    pub created_at: u64,
    /// Blob store artifact ID for the serialized state.
    pub artifact_id: String,
    /// Size of the serialized checkpoint in bytes.
    pub size: usize,
    /// Short label of the triggering operation, if auto-created.
    pub trigger: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_names() {
        assert_eq!(
            DestructiveOp::Delete {
                table: "t".into(),
                row_count: 0
            }
            .operation_name(),
            "DELETE"
        );
        assert_eq!(
            DestructiveOp::DropTable {
                table: "t".into(),
                row_count: 0
            }
            .operation_name(),
            "DROP TABLE"
        );
        assert_eq!(
            DestructiveOp::DropIndex {
                table: "t".into(),
                column: "c".into()
            }
            .operation_name(),
            "DROP INDEX"
        );
        assert_eq!(
            DestructiveOp::NodeDelete {
                node_id: 1,
                edge_count: 0
            }
            .operation_name(),
            "NODE DELETE"
        );
        assert_eq!(
            DestructiveOp::EdgeDelete { edge_id: 1 }.operation_name(),
            "EDGE DELETE"
        );
        assert_eq!(
            DestructiveOp::EmbedDelete { key: "k".into() }.operation_name(),
            "EMBED DELETE"
        );
        assert_eq!(
            DestructiveOp::VaultDelete { key: "k".into() }.operation_name(),
            "VAULT DELETE"
        );
        assert_eq!(
            DestructiveOp::BlobDelete {
                artifact_id: "a".into(),
                size: 0
            }
            .operation_name(),
            "BLOB DELETE"
        );
        assert_eq!(
            DestructiveOp::CacheClear { entry_count: 0 }.operation_name(),
            "CACHE CLEAR"
        );
    }

    #[test]
    fn test_affected_counts() {
        assert_eq!(
            DestructiveOp::Delete {
                table: "t".into(),
                row_count: 10
            }
            .affected_count(),
            10
        );
        assert_eq!(
            DestructiveOp::DropTable {
                table: "t".into(),
                row_count: 100
            }
            .affected_count(),
            100
        );
        assert_eq!(
            DestructiveOp::DropIndex {
                table: "t".into(),
                column: "c".into()
            }
            .affected_count(),
            1
        );
        assert_eq!(
            DestructiveOp::NodeDelete {
                node_id: 1,
                edge_count: 5
            }
            .affected_count(),
            6
        );
        assert_eq!(DestructiveOp::EdgeDelete { edge_id: 1 }.affected_count(), 1);
        assert_eq!(
            DestructiveOp::EmbedDelete { key: "k".into() }.affected_count(),
            1
        );
        assert_eq!(
            DestructiveOp::VaultDelete { key: "k".into() }.affected_count(),
            1
        );
        assert_eq!(
            DestructiveOp::BlobDelete {
                artifact_id: "a".into(),
                size: 1024
            }
            .affected_count(),
            1
        );
        assert_eq!(
            DestructiveOp::CacheClear { entry_count: 50 }.affected_count(),
            50
        );
    }

    #[test]
    fn test_checkpoint_state_with_trigger() {
        let meta = CheckpointMetadata::default();
        let state = CheckpointState::new("id".into(), "name".into(), vec![], meta);

        assert!(state.trigger.is_none());

        let trigger = CheckpointTrigger::new(
            "DELETE FROM users".into(),
            DestructiveOp::Delete {
                table: "users".into(),
                row_count: 5,
            },
            OperationPreview::empty("test"),
        );

        let state_with_trigger = state.with_trigger(trigger);
        assert!(state_with_trigger.trigger.is_some());
    }

    #[test]
    fn test_operation_preview_empty() {
        let preview = OperationPreview::empty("No data");
        assert_eq!(preview.summary, "No data");
        assert!(preview.sample_data.is_empty());
        assert_eq!(preview.affected_count, 0);
    }

    #[test]
    fn test_metadata_constructors() {
        let rel = RelationalMeta::new(5, 100);
        assert_eq!(rel.table_count, 5);
        assert_eq!(rel.total_rows, 100);

        let graph = GraphMeta::new(10, 20);
        assert_eq!(graph.node_count, 10);
        assert_eq!(graph.edge_count, 20);

        let vec = VectorMeta::new(50);
        assert_eq!(vec.embedding_count, 50);

        let meta = CheckpointMetadata::new(rel, graph, vec, 1000);
        assert_eq!(meta.store_key_count, 1000);
    }

    #[test]
    fn test_checkpoint_trigger() {
        let trigger = CheckpointTrigger::new(
            "DROP TABLE users".into(),
            DestructiveOp::DropTable {
                table: "users".into(),
                row_count: 100,
            },
            OperationPreview::new("Dropping table".into(), vec!["row1".into()], 100),
        );

        assert_eq!(trigger.command, "DROP TABLE users");
        assert_eq!(trigger.operation.operation_name(), "DROP TABLE");
        assert_eq!(trigger.preview.affected_count, 100);
    }
}
