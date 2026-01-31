//! Error types for the graph engine.

use std::{
    fmt,
    hash::{Hash, Hasher},
};

use serde::{Deserialize, Serialize};

/// Error type for graph operations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GraphError {
    /// Node with the given ID was not found.
    NodeNotFound(u64),
    /// Edge with the given ID was not found.
    EdgeNotFound(u64),
    /// Underlying storage operation failed.
    StorageError(String),
    /// No path exists between the specified nodes.
    PathNotFound,
    /// Index already exists.
    IndexAlreadyExists { target: String, property: String },
    /// Index not found.
    IndexNotFound { target: String, property: String },
    /// Negative weight found during weighted path search.
    NegativeWeight { edge_id: u64, weight: f64 },
    /// Operation would violate a constraint.
    ConstraintViolation {
        constraint_name: String,
        message: String,
    },
    /// Constraint with the given name already exists.
    ConstraintAlreadyExists(String),
    /// Constraint with the given name was not found.
    ConstraintNotFound(String),
    /// Batch validation failed at the given index.
    BatchValidationError { index: usize, cause: Box<Self> },
    /// Batch creation failed at the given index.
    BatchCreationError { index: usize, cause: Box<Self> },
    /// Node deletion partially failed (some edges could not be deleted).
    PartialDeletionError {
        node_id: u64,
        failed_edges: Vec<u64>,
    },
    /// ID counter would overflow.
    IdSpaceExhausted { entity_type: &'static str },
    /// Invalid property name (contains reserved characters).
    InvalidPropertyName { name: String },
    /// Corrupted edge data.
    CorruptedEdge { edge_id: u64, field: String },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "Node not found: {id}"),
            Self::EdgeNotFound(id) => write!(f, "Edge not found: {id}"),
            Self::StorageError(e) => write!(f, "Storage error: {e}"),
            Self::PathNotFound => write!(f, "No path found between nodes"),
            Self::IndexAlreadyExists { target, property } => {
                write!(f, "Index already exists: {target}.{property}")
            },
            Self::IndexNotFound { target, property } => {
                write!(f, "Index not found: {target}.{property}")
            },
            Self::NegativeWeight { edge_id, weight } => {
                write!(f, "Edge {edge_id} has negative weight: {weight}")
            },
            Self::ConstraintViolation {
                constraint_name,
                message,
            } => {
                write!(f, "Constraint '{constraint_name}' violated: {message}")
            },
            Self::ConstraintAlreadyExists(name) => {
                write!(f, "Constraint already exists: {name}")
            },
            Self::ConstraintNotFound(name) => write!(f, "Constraint not found: {name}"),
            Self::BatchValidationError { index, cause } => {
                write!(f, "Batch validation failed at index {index}: {cause}")
            },
            Self::BatchCreationError { index, cause } => {
                write!(f, "Batch creation failed at index {index}: {cause}")
            },
            Self::PartialDeletionError {
                node_id,
                failed_edges,
            } => {
                write!(
                    f,
                    "Partial deletion of node {node_id}: {} edges failed to delete",
                    failed_edges.len()
                )
            },
            Self::IdSpaceExhausted { entity_type } => {
                write!(f, "ID space exhausted for {entity_type}")
            },
            Self::InvalidPropertyName { name } => {
                write!(
                    f,
                    "Invalid property name '{name}': contains reserved character ':'"
                )
            },
            Self::CorruptedEdge { edge_id, field } => {
                write!(
                    f,
                    "Corrupted edge {edge_id}: missing or invalid field '{field}'"
                )
            },
        }
    }
}

impl std::error::Error for GraphError {}

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
