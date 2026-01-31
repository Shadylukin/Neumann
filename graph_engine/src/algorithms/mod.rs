//! Graph algorithms module.
//!
//! This module provides advanced graph algorithms including:
//! - Strongly Connected Components (Tarjan's algorithm)
//! - Minimum Spanning Tree (Kruskal's algorithm)
//! - A* pathfinding with pluggable heuristics
//! - Triangle counting and clustering coefficients
//! - K-core decomposition
//! - Node similarity metrics (Jaccard, Cosine, Adamic-Adar)
//! - Biconnected components, articulation points, and bridges

mod astar;
mod biconnected;
mod kcore;
mod mst;
mod scc;
mod similarity;
mod triangles;

pub use astar::{AStarConfig, AStarResult, HeuristicFn};
pub use biconnected::{BiconnectedConfig, BiconnectedResult};
pub use kcore::{KCoreConfig, KCoreResult};
pub use mst::{MstConfig, MstEdge, MstResult};
pub use scc::{SccConfig, SccResult};
pub use similarity::{SimilarityConfig, SimilarityMetric, SimilarityResult};
pub use triangles::{TriangleConfig, TriangleResult};
