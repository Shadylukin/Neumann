// SPDX-License-Identifier: MIT OR Apache-2.0
//! Error types for the `tensor_learn` crate.

use graph_engine::GraphError;
use vector_engine::VectorError;

/// Errors that can occur during learning operations.
#[derive(Debug, thiserror::Error)]
pub enum LearnError {
    /// Graph engine error.
    #[error("graph error: {0}")]
    Graph(#[from] GraphError),

    /// Vector engine error.
    #[error("vector error: {0}")]
    Vector(#[from] VectorError),

    /// Invalid point (outside the Poincare disk).
    #[error("invalid Poincare point: {0}")]
    InvalidPoint(String),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Serialization error.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
