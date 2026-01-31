//! Node similarity metrics for link prediction and community detection.
//!
//! Provides neighborhood-based similarity measures including:
//! - Jaccard similarity
//! - Cosine similarity
//! - Adamic-Adar index
//! - Common neighbors
//! - Preferential attachment

#![allow(clippy::cast_precision_loss)] // Acceptable for graph algorithm metrics

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{Direction, GraphEngine, Result};

/// Similarity metric type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimilarityMetric {
    /// Jaccard similarity: |A ∩ B| / |A ∪ B|
    Jaccard,
    /// Cosine similarity: |A ∩ B| / sqrt(|A| * |B|)
    Cosine,
    /// Adamic-Adar: sum of 1/log(degree) for common neighbors
    AdamicAdar,
    /// Resource allocation: sum of 1/degree for common neighbors
    ResourceAllocation,
    /// Preferential attachment: |A| * |B|
    PreferentialAttachment,
    /// Common neighbors count
    CommonNeighbors,
}

/// Configuration for similarity computation.
#[derive(Debug, Clone, Default)]
pub struct SimilarityConfig {
    /// Edge type filter. None means all edge types.
    pub edge_type: Option<String>,
    /// Direction for neighbor computation.
    pub direction: Direction,
}

impl SimilarityConfig {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            edge_type: None,
            direction: Direction::Both,
        }
    }

    #[must_use]
    pub fn edge_type(mut self, edge_type: impl Into<String>) -> Self {
        self.edge_type = Some(edge_type.into());
        self
    }

    #[must_use]
    pub const fn direction(mut self, direction: Direction) -> Self {
        self.direction = direction;
        self
    }
}

/// Result of similarity computation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// Node A.
    pub node_a: u64,
    /// Node B.
    pub node_b: u64,
    /// Similarity score.
    pub score: f64,
    /// Metric used.
    pub metric: SimilarityMetric,
    /// Common neighbors (if computed).
    pub common_neighbors: Option<Vec<u64>>,
}

impl SimilarityResult {
    #[must_use]
    pub const fn new(node_a: u64, node_b: u64, score: f64, metric: SimilarityMetric) -> Self {
        Self {
            node_a,
            node_b,
            score,
            metric,
            common_neighbors: None,
        }
    }

    #[must_use]
    pub fn with_common_neighbors(mut self, neighbors: Vec<u64>) -> Self {
        self.common_neighbors = Some(neighbors);
        self
    }
}

impl GraphEngine {
    /// Compute Jaccard similarity between two nodes.
    ///
    /// Jaccard(A, B) = |N(A) ∩ N(B)| / |N(A) ∪ N(B)|
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn jaccard_similarity(&self, a: u64, b: u64, config: &SimilarityConfig) -> Result<f64> {
        let neighbors_a = self.get_neighbor_set(a, config)?;
        let neighbors_b = self.get_neighbor_set(b, config)?;

        if neighbors_a.is_empty() && neighbors_b.is_empty() {
            return Ok(0.0);
        }

        let intersection = neighbors_a.intersection(&neighbors_b).count();
        let union = neighbors_a.union(&neighbors_b).count();

        Ok(intersection as f64 / union as f64)
    }

    /// Compute cosine similarity between two nodes.
    ///
    /// Cosine(A, B) = |N(A) ∩ N(B)| / sqrt(|N(A)| * |N(B)|)
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn cosine_similarity(&self, a: u64, b: u64, config: &SimilarityConfig) -> Result<f64> {
        let neighbors_a = self.get_neighbor_set(a, config)?;
        let neighbors_b = self.get_neighbor_set(b, config)?;

        if neighbors_a.is_empty() || neighbors_b.is_empty() {
            return Ok(0.0);
        }

        let intersection = neighbors_a.intersection(&neighbors_b).count();
        let denominator = ((neighbors_a.len() * neighbors_b.len()) as f64).sqrt();

        Ok(intersection as f64 / denominator)
    }

    /// Find common neighbors between two nodes.
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn common_neighbors(&self, a: u64, b: u64, config: &SimilarityConfig) -> Result<Vec<u64>> {
        let neighbors_a = self.get_neighbor_set(a, config)?;
        let neighbors_b = self.get_neighbor_set(b, config)?;

        Ok(neighbors_a.intersection(&neighbors_b).copied().collect())
    }

    /// Compute Adamic-Adar index between two nodes.
    ///
    /// AA(A, B) = sum over common neighbors z of 1/log(|N(z)|)
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn adamic_adar(&self, a: u64, b: u64, config: &SimilarityConfig) -> Result<f64> {
        let common = self.common_neighbors(a, b, config)?;

        let mut score = 0.0;
        for neighbor in common {
            let degree = self.get_neighbor_set(neighbor, config)?.len();
            if degree > 1 {
                score += 1.0 / (degree as f64).ln();
            }
        }

        Ok(score)
    }

    /// Compute resource allocation index between two nodes.
    ///
    /// RA(A, B) = sum over common neighbors z of 1/|N(z)|
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn resource_allocation(&self, a: u64, b: u64, config: &SimilarityConfig) -> Result<f64> {
        let common = self.common_neighbors(a, b, config)?;

        let mut score = 0.0;
        for neighbor in common {
            let degree = self.get_neighbor_set(neighbor, config)?.len();
            if degree > 0 {
                score += 1.0 / degree as f64;
            }
        }

        Ok(score)
    }

    /// Compute preferential attachment score between two nodes.
    ///
    /// PA(A, B) = |N(A)| * |N(B)|
    ///
    /// # Errors
    ///
    /// Returns an error if node retrieval fails.
    pub fn preferential_attachment(
        &self,
        a: u64,
        b: u64,
        config: &SimilarityConfig,
    ) -> Result<f64> {
        let neighbors_a = self.get_neighbor_set(a, config)?;
        let neighbors_b = self.get_neighbor_set(b, config)?;

        Ok((neighbors_a.len() * neighbors_b.len()) as f64)
    }

    /// Compute similarity using the specified metric.
    ///
    /// # Errors
    ///
    /// Returns an error if computation fails.
    pub fn node_similarity(
        &self,
        a: u64,
        b: u64,
        metric: SimilarityMetric,
        config: &SimilarityConfig,
    ) -> Result<SimilarityResult> {
        let score = match metric {
            SimilarityMetric::Jaccard => self.jaccard_similarity(a, b, config)?,
            SimilarityMetric::Cosine => self.cosine_similarity(a, b, config)?,
            SimilarityMetric::AdamicAdar => self.adamic_adar(a, b, config)?,
            SimilarityMetric::ResourceAllocation => self.resource_allocation(a, b, config)?,
            SimilarityMetric::PreferentialAttachment => {
                self.preferential_attachment(a, b, config)?
            },
            SimilarityMetric::CommonNeighbors => self.common_neighbors(a, b, config)?.len() as f64,
        };

        let common = self.common_neighbors(a, b, config)?;
        Ok(SimilarityResult::new(a, b, score, metric).with_common_neighbors(common))
    }

    /// Find most similar nodes to a given node.
    ///
    /// # Errors
    ///
    /// Returns an error if computation fails.
    pub fn most_similar(
        &self,
        node: u64,
        metric: SimilarityMetric,
        config: &SimilarityConfig,
        limit: usize,
    ) -> Result<Vec<SimilarityResult>> {
        let all_nodes = self.get_all_node_ids()?;
        let mut results = Vec::new();

        for &other in &all_nodes {
            if other == node {
                continue;
            }

            let result = self.node_similarity(node, other, metric, config)?;
            if result.score > 0.0 {
                results.push(result);
            }
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(limit);
        Ok(results)
    }

    /// Get neighbor set for similarity computation.
    fn get_neighbor_set(&self, node_id: u64, config: &SimilarityConfig) -> Result<HashSet<u64>> {
        let neighbors =
            self.neighbors(node_id, config.edge_type.as_deref(), config.direction, None)?;
        Ok(neighbors.into_iter().map(|n| n.id).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_jaccard_identical_neighborhoods() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // Both a and b connect to c
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = SimilarityConfig::new();
        let jaccard = engine.jaccard_similarity(a, b, &config).unwrap();

        // Both have same single neighbor
        assert!((jaccard - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_no_overlap() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // a connects to c, b connects to d (no overlap)
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, d, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = SimilarityConfig::new();
        let jaccard = engine.jaccard_similarity(a, b, &config).unwrap();

        assert!(jaccard.abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // a -> c, d; b -> c
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, d, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = SimilarityConfig::new();
        let jaccard = engine.jaccard_similarity(a, b, &config).unwrap();

        // intersection = 1 (c), union = 2 (c, d)
        assert!((jaccard - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cosine_similarity() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = SimilarityConfig::new();
        let cosine = engine.cosine_similarity(a, b, &config).unwrap();

        // Both have 1 neighbor, 1 common
        // cosine = 1 / sqrt(1 * 1) = 1.0
        assert!((cosine - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_common_neighbors() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, d, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, d, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = SimilarityConfig::new();
        let common = engine.common_neighbors(a, b, &config).unwrap();

        assert_eq!(common.len(), 2);
        assert!(common.contains(&c));
        assert!(common.contains(&d));
    }

    #[test]
    fn test_adamic_adar() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        // a and b both connect to c
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = SimilarityConfig::new();
        let aa = engine.adamic_adar(a, b, &config).unwrap();

        // c has degree 2, so AA = 1/ln(2)
        let expected = 1.0 / 2.0_f64.ln();
        assert!((aa - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_allocation() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = SimilarityConfig::new();
        let ra = engine.resource_allocation(a, b, &config).unwrap();

        // c has degree 2, so RA = 1/2
        assert!((ra - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_preferential_attachment() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let d = engine.create_node("D", HashMap::new()).unwrap();

        // a has 2 neighbors, b has 1 neighbor
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(a, d, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = SimilarityConfig::new();
        let pa = engine.preferential_attachment(a, b, &config).unwrap();

        // PA = 2 * 1 = 2
        assert!((pa - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_most_similar() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();
        let _d = engine.create_node("D", HashMap::new()).unwrap();

        // a and b share c as neighbor
        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();
        // d is isolated from a

        let config = SimilarityConfig::new();
        let similar = engine
            .most_similar(a, SimilarityMetric::Jaccard, &config, 5)
            .unwrap();

        // b should be most similar to a (they share c)
        assert!(!similar.is_empty());
        assert_eq!(similar[0].node_b, b);
    }

    #[test]
    fn test_node_similarity_result() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();
        let c = engine.create_node("C", HashMap::new()).unwrap();

        engine
            .create_edge(a, c, "EDGE", HashMap::new(), false)
            .unwrap();
        engine
            .create_edge(b, c, "EDGE", HashMap::new(), false)
            .unwrap();

        let config = SimilarityConfig::new();
        let result = engine
            .node_similarity(a, b, SimilarityMetric::Jaccard, &config)
            .unwrap();

        assert_eq!(result.node_a, a);
        assert_eq!(result.node_b, b);
        assert_eq!(result.metric, SimilarityMetric::Jaccard);
        assert!(result.common_neighbors.is_some());
        assert!(result.common_neighbors.unwrap().contains(&c));
    }

    #[test]
    fn test_similarity_empty_neighborhoods() {
        let engine = GraphEngine::new();
        let a = engine.create_node("A", HashMap::new()).unwrap();
        let b = engine.create_node("B", HashMap::new()).unwrap();

        let config = SimilarityConfig::new();

        // Both nodes have no neighbors
        let jaccard = engine.jaccard_similarity(a, b, &config).unwrap();
        assert!(jaccard.abs() < f64::EPSILON);

        let cosine = engine.cosine_similarity(a, b, &config).unwrap();
        assert!(cosine.abs() < f64::EPSILON);
    }
}
