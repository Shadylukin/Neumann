//! Error types for the graph engine.

use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error type for graph operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Error)]
pub enum GraphError {
    /// Node with the given ID was not found.
    #[error("Node not found: {0}")]
    NodeNotFound(u64),
    /// Edge with the given ID was not found.
    #[error("Edge not found: {0}")]
    EdgeNotFound(u64),
    /// Underlying storage operation failed.
    #[error("Storage error: {0}")]
    StorageError(String),
    /// No path exists between the specified nodes.
    #[error("No path found between nodes")]
    PathNotFound,
    /// Index already exists.
    #[error("Index already exists: {target}.{property}")]
    IndexAlreadyExists { target: String, property: String },
    /// Index not found.
    #[error("Index not found: {target}.{property}")]
    IndexNotFound { target: String, property: String },
    /// Negative weight found during weighted path search.
    #[error("Edge {edge_id} has negative weight: {weight}")]
    NegativeWeight { edge_id: u64, weight: f64 },
    /// Operation would violate a constraint.
    #[error("Constraint '{constraint_name}' violated: {message}")]
    ConstraintViolation {
        constraint_name: String,
        message: String,
    },
    /// Constraint with the given name already exists.
    #[error("Constraint already exists: {0}")]
    ConstraintAlreadyExists(String),
    /// Constraint with the given name was not found.
    #[error("Constraint not found: {0}")]
    ConstraintNotFound(String),
    /// Batch validation failed at the given index.
    #[error("Batch validation failed at index {index}: {cause}")]
    BatchValidationError { index: usize, cause: Box<Self> },
    /// Batch creation failed at the given index.
    #[error("Batch creation failed at index {index}: {cause}")]
    BatchCreationError { index: usize, cause: Box<Self> },
    /// Node deletion partially failed (some edges could not be deleted).
    #[error("Partial deletion of node {node_id}: {} edges failed to delete", failed_edges.len())]
    PartialDeletionError {
        node_id: u64,
        failed_edges: Vec<u64>,
    },
    /// ID counter would overflow.
    #[error("ID space exhausted for {entity_type}")]
    IdSpaceExhausted { entity_type: &'static str },
    /// Invalid property name (contains reserved characters).
    #[error("Invalid property name '{name}': contains reserved character ':'")]
    InvalidPropertyName { name: String },
    /// Corrupted edge data.
    #[error("Corrupted edge {edge_id}: missing or invalid field '{field}'")]
    CorruptedEdge { edge_id: u64, field: String },
}

impl Eq for GraphError {}

impl Hash for GraphError {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::NodeNotFound(id) | Self::EdgeNotFound(id) => id.hash(state),
            Self::StorageError(s)
            | Self::ConstraintAlreadyExists(s)
            | Self::ConstraintNotFound(s) => s.hash(state),
            Self::PathNotFound => {},
            Self::IndexAlreadyExists { target, property }
            | Self::IndexNotFound { target, property } => {
                target.hash(state);
                property.hash(state);
            },
            Self::NegativeWeight { edge_id, weight } => {
                edge_id.hash(state);
                weight.to_bits().hash(state);
            },
            Self::ConstraintViolation {
                constraint_name,
                message,
            } => {
                constraint_name.hash(state);
                message.hash(state);
            },
            Self::BatchValidationError { index, cause }
            | Self::BatchCreationError { index, cause } => {
                index.hash(state);
                cause.hash(state);
            },
            Self::PartialDeletionError {
                node_id,
                failed_edges,
            } => {
                node_id.hash(state);
                failed_edges.hash(state);
            },
            Self::IdSpaceExhausted { entity_type } => {
                entity_type.hash(state);
            },
            Self::InvalidPropertyName { name } => {
                name.hash(state);
            },
            Self::CorruptedEdge { edge_id, field } => {
                edge_id.hash(state);
                field.hash(state);
            },
        }
    }
}

impl From<tensor_store::TensorStoreError> for GraphError {
    fn from(e: tensor_store::TensorStoreError) -> Self {
        Self::StorageError(e.to_string())
    }
}

/// Result type alias for graph operations.
pub type Result<T> = std::result::Result<T, GraphError>;

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;

    use super::*;

    #[test]
    fn test_partial_deletion_error_display() {
        let err = GraphError::PartialDeletionError {
            node_id: 42,
            failed_edges: vec![1, 2, 3],
        };
        assert_eq!(
            err.to_string(),
            "Partial deletion of node 42: 3 edges failed to delete"
        );
    }

    #[test]
    fn test_id_space_exhausted_display() {
        let err = GraphError::IdSpaceExhausted {
            entity_type: "node",
        };
        assert_eq!(err.to_string(), "ID space exhausted for node");
    }

    #[test]
    fn test_invalid_property_name_display() {
        let err = GraphError::InvalidPropertyName {
            name: "bad:prop".to_owned(),
        };
        assert_eq!(
            err.to_string(),
            "Invalid property name 'bad:prop': contains reserved character ':'"
        );
    }

    #[test]
    fn test_corrupted_edge_display() {
        let err = GraphError::CorruptedEdge {
            edge_id: 99,
            field: "source".to_owned(),
        };
        assert_eq!(
            err.to_string(),
            "Corrupted edge 99: missing or invalid field 'source'"
        );
    }

    #[test]
    fn test_partial_deletion_error_hash() {
        let err = GraphError::PartialDeletionError {
            node_id: 10,
            failed_edges: vec![1, 2],
        };
        let mut hasher = DefaultHasher::new();
        err.hash(&mut hasher);
        let h1 = hasher.finish();

        let err2 = GraphError::PartialDeletionError {
            node_id: 10,
            failed_edges: vec![1, 2],
        };
        let mut hasher2 = DefaultHasher::new();
        err2.hash(&mut hasher2);
        assert_eq!(h1, hasher2.finish());
    }

    #[test]
    fn test_id_space_exhausted_hash() {
        let err = GraphError::IdSpaceExhausted {
            entity_type: "edge",
        };
        let mut hasher = DefaultHasher::new();
        err.hash(&mut hasher);
        // Just ensure hashing does not panic.
        let _ = hasher.finish();
    }

    #[test]
    fn test_invalid_property_name_hash() {
        let err = GraphError::InvalidPropertyName {
            name: "x:y".to_owned(),
        };
        let mut hasher = DefaultHasher::new();
        err.hash(&mut hasher);
        let _ = hasher.finish();
    }

    #[test]
    fn test_corrupted_edge_hash() {
        let err = GraphError::CorruptedEdge {
            edge_id: 5,
            field: "target".to_owned(),
        };
        let mut hasher = DefaultHasher::new();
        err.hash(&mut hasher);
        let _ = hasher.finish();
    }

    #[test]
    fn test_error_is_std_error() {
        let err = GraphError::NodeNotFound(1);
        // Verify the thiserror derive implements std::error::Error.
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn test_partial_deletion_error_eq() {
        let a = GraphError::PartialDeletionError {
            node_id: 1,
            failed_edges: vec![10, 20],
        };
        let b = GraphError::PartialDeletionError {
            node_id: 1,
            failed_edges: vec![10, 20],
        };
        assert_eq!(a, b);
    }
}
