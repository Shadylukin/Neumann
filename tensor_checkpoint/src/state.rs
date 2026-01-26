use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointState {
    pub id: String,
    pub name: String,
    pub created_at: u64,
    pub trigger: Option<CheckpointTrigger>,
    pub store_snapshot: Vec<u8>,
    pub metadata: CheckpointMetadata,
}

impl CheckpointState {
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

    pub fn with_trigger(mut self, trigger: CheckpointTrigger) -> Self {
        self.trigger = Some(trigger);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointTrigger {
    pub command: String,
    pub operation: DestructiveOp,
    pub preview: OperationPreview,
}

impl CheckpointTrigger {
    pub fn new(command: String, operation: DestructiveOp, preview: OperationPreview) -> Self {
        Self {
            command,
            operation,
            preview,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DestructiveOp {
    Delete { table: String, row_count: usize },
    DropTable { table: String, row_count: usize },
    DropIndex { table: String, column: String },
    NodeDelete { node_id: u64, edge_count: usize },
    EdgeDelete { edge_id: u64 },
    EmbedDelete { key: String },
    VaultDelete { key: String },
    BlobDelete { artifact_id: String, size: usize },
    CacheClear { entry_count: usize },
}

impl DestructiveOp {
    pub fn operation_name(&self) -> &'static str {
        match self {
            DestructiveOp::Delete { .. } => "DELETE",
            DestructiveOp::DropTable { .. } => "DROP TABLE",
            DestructiveOp::DropIndex { .. } => "DROP INDEX",
            DestructiveOp::NodeDelete { .. } => "NODE DELETE",
            DestructiveOp::EdgeDelete { .. } => "EDGE DELETE",
            DestructiveOp::EmbedDelete { .. } => "EMBED DELETE",
            DestructiveOp::VaultDelete { .. } => "VAULT DELETE",
            DestructiveOp::BlobDelete { .. } => "BLOB DELETE",
            DestructiveOp::CacheClear { .. } => "CACHE CLEAR",
        }
    }

    pub fn affected_count(&self) -> usize {
        match self {
            DestructiveOp::Delete { row_count, .. } => *row_count,
            DestructiveOp::DropTable { row_count, .. } => *row_count,
            DestructiveOp::DropIndex { .. } => 1,
            DestructiveOp::NodeDelete { edge_count, .. } => 1 + edge_count,
            DestructiveOp::EdgeDelete { .. } => 1,
            DestructiveOp::EmbedDelete { .. } => 1,
            DestructiveOp::VaultDelete { .. } => 1,
            DestructiveOp::BlobDelete { .. } => 1,
            DestructiveOp::CacheClear { entry_count } => *entry_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationPreview {
    pub summary: String,
    pub sample_data: Vec<String>,
    pub affected_count: usize,
}

impl OperationPreview {
    pub fn new(summary: String, sample_data: Vec<String>, affected_count: usize) -> Self {
        Self {
            summary,
            sample_data,
            affected_count,
        }
    }

    pub fn empty(message: &str) -> Self {
        Self {
            summary: message.to_string(),
            sample_data: Vec::new(),
            affected_count: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointMetadata {
    pub relational: RelationalMeta,
    pub graph: GraphMeta,
    pub vector: VectorMeta,
    pub store_key_count: usize,
}

impl CheckpointMetadata {
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RelationalMeta {
    pub table_count: usize,
    pub total_rows: usize,
}

impl RelationalMeta {
    pub fn new(table_count: usize, total_rows: usize) -> Self {
        Self {
            table_count,
            total_rows,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphMeta {
    pub node_count: usize,
    pub edge_count: usize,
}

impl GraphMeta {
    pub fn new(node_count: usize, edge_count: usize) -> Self {
        Self {
            node_count,
            edge_count,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorMeta {
    pub embedding_count: usize,
}

impl VectorMeta {
    pub fn new(embedding_count: usize) -> Self {
        Self { embedding_count }
    }
}

#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub id: String,
    pub name: String,
    pub created_at: u64,
    pub artifact_id: String,
    pub size: usize,
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
