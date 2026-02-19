//! Cross-modal tensor contraction for unified scoring.
//!
//! Fuses graph adjacency, vector similarity, and relational interactions into a
//! single algebraic scoring expression:
//!
//! ```text
//! score = (G[x,:] ⊙ s)^T R        (shape-safe: (1×n ⊙ 1×n)^T · n×m → 1×m)
//! ```
//!
//! where **G** is the graph adjacency row for source entity *x*, **s** is the
//! cosine-similarity vector between *x* and its neighbors, **⊙** is the
//! element-wise (Hadamard) product, and **R** is the neighbor-to-item
//! interaction matrix from a relational table.
//!
//! # Algorithm
//!
//! Given a source entity (e.g. a user):
//!
//! 1. **Adjacency** — gather the source's graph neighbors with edge weights.
//! 2. **Similarity** — compute cosine similarity between the source's embedding
//!    and each neighbor's embedding.
//! 3. **Hadamard product** — fuse adjacency and similarity into a single weight
//!    per neighbor: `w[i] = adj[i] * sim[i]`.
//! 4. **Normalization** (optional) — L1-normalize the weight vector to control
//!    magnitude.
//! 5. **Contraction** — multiply the weight vector by the interaction matrix:
//!    for each neighbor, distribute its weight to every item it interacted with.
//! 6. **Post-processing** — exclude already-owned items, apply a category mask,
//!    per-item normalization, non-finite filtering, and top-k truncation.
//!
//! # Example
//!
//! ```text
//! Graph: alice --FRIEND--> bob, carol, dave
//!
//! Embeddings:
//!   alice = [1, 0, 0]
//!   bob   = [0.9, 0.1, 0]    cos(alice, bob)   ≈ 0.994
//!   carol = [0.5, 0.5, 0]    cos(alice, carol)  ≈ 0.707
//!   dave  = [0, 1, 0]        cos(alice, dave)   = 0.000
//!
//! Purchases table:
//!   bob   -> {book, pen}
//!   carol -> {pen, laptop}
//!   dave  -> {phone}
//!
//! Contraction scores:
//!   pen    = 0.994 + 0.707 = 1.701   (from bob + carol)
//!   book   = 0.994                    (from bob)
//!   laptop = 0.707                    (from carol)
//!   phone  = 0.000                    (from dave — orthogonal embedding)
//! ```
//!
//! # Usage
//!
//! The pure [`contract`] function operates on sparse hash-map representations
//! and is engine-agnostic. The [`UnifiedEngine::cross_modal_contraction`]
//! adapter gathers data from the graph, vector, and relational engines, then
//! delegates to `contract`.

use std::collections::{HashMap, HashSet};

use graph_engine::{Direction, PropertyValue};
use relational_engine::{ColumnType, Value};
use vector_engine::VectorEngine;

use crate::{Result, UnifiedEngine, UnifiedError};

// ---------------------------------------------------------------------------
// Sparse types
// ---------------------------------------------------------------------------

/// Sparse adjacency: neighbor key to edge weight (1.0 for unweighted).
pub type AdjacencyVec = HashMap<String, f64>;

/// Similarity: entity key to cosine similarity to source.
pub type SimilarityVec = HashMap<String, f64>;

/// Interaction matrix: intermediary key to set of item keys they interacted with.
pub type InteractionMap = HashMap<String, HashSet<String>>;

/// Category mask: set of item keys in target category.
pub type CategoryMask = HashSet<String>;

/// Owned items: set of item keys the source entity already has.
pub type OwnedSet = HashSet<String>;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Graph traversal direction for adjacency gathering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphDirection {
    /// Union of outgoing and incoming edges.
    Symmetric,
    /// Source to neighbors only.
    Outgoing,
    /// Neighbors to source only.
    Incoming,
}

impl From<GraphDirection> for Direction {
    fn from(d: GraphDirection) -> Self {
        match d {
            GraphDirection::Symmetric => Self::Both,
            GraphDirection::Outgoing => Self::Outgoing,
            GraphDirection::Incoming => Self::Incoming,
        }
    }
}

/// Normalization strategy for weight and score vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// No normalization — raw scores.
    None,
    /// Divide each weight by the L1 norm of the weight vector.
    TotalWeight,
    /// Divide each item score by its contributor count.
    PerItem,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for cross-modal tensor contraction.
#[derive(Debug, Clone)]
pub struct ContractionConfig {
    /// Graph traversal direction.
    pub direction: GraphDirection,
    /// How to normalize weight/score vectors.
    pub normalization: Normalization,
    /// Edge type filter (e.g., `"FRIEND"`). `None` means all edge types.
    pub edge_type: Option<String>,
    /// Whether to exclude items the source entity already owns.
    pub exclude_owned: bool,
    /// Maximum results to return.
    pub top_k: usize,
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A single scored item from tensor contraction.
#[derive(Debug, Clone)]
pub struct ScoredItem {
    /// Item key in the unified key space.
    pub item_key: String,
    /// Contraction score (higher = more recommended).
    pub score: f64,
    /// Number of distinct intermediaries that contributed to this score.
    pub contributors: usize,
}

/// Result of a cross-modal tensor contraction.
#[derive(Debug, Clone)]
pub struct ContractionResult {
    /// Scored items, sorted descending by score then ascending by key.
    pub items: Vec<ScoredItem>,
    /// L1 norm of the weight vector, 0.0 if no weights computed.
    pub weight_norm: f64,
    /// Number of items excluded because source already owns them.
    pub excluded_count: usize,
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

/// Element-wise product of adjacency and similarity vectors.
///
/// Only keys present in **both** maps are included. Non-finite values in
/// either map are skipped.
fn hadamard_product(adj: &AdjacencyVec, sim: &SimilarityVec) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for (key, &a) in adj {
        if let Some(&s) = sim.get(key) {
            if a.is_finite() && s.is_finite() {
                out.insert(key.clone(), a * s);
            }
        }
    }
    out
}

/// Normalize weights in place and return the L1 norm used.
///
/// For [`Normalization::TotalWeight`], divides each weight by `Σ|w_i|`.
/// If the norm is below `f64::EPSILON`, weights pass through unchanged.
/// Other strategies are no-ops at this stage.
fn normalize_weights(weights: &mut HashMap<String, f64>, strategy: Normalization) -> f64 {
    if strategy != Normalization::TotalWeight {
        return 0.0;
    }
    let norm: f64 = weights.values().map(|w| w.abs()).sum();
    if norm > f64::EPSILON {
        for w in weights.values_mut() {
            *w /= norm;
        }
    }
    norm
}

// ---------------------------------------------------------------------------
// Pure contraction
// ---------------------------------------------------------------------------

/// Pure tensor contraction over sparse hash-map representations.
///
/// # Steps
///
/// 1. Hadamard product of adjacency and similarity vectors.
/// 2. Optional `TotalWeight` normalization.
/// 3. Matrix multiplication against the interaction map.
/// 4. Owned-item exclusion.
/// 5. Category masking.
/// 6. Optional `PerItem` normalization.
/// 7. Non-finite score removal.
/// 8. Sort and truncate to `top_k`.
#[must_use]
#[allow(clippy::implicit_hasher)]
pub fn contract(
    adjacency: &AdjacencyVec,
    similarity: &SimilarityVec,
    interactions: &InteractionMap,
    owned: Option<&OwnedSet>,
    category: Option<&CategoryMask>,
    config: &ContractionConfig,
) -> ContractionResult {
    // 1. Hadamard
    let mut weights = hadamard_product(adjacency, similarity);

    // 2. TotalWeight normalization
    let weight_norm = normalize_weights(&mut weights, config.normalization);

    // 3. Contraction (matrix multiply)
    let mut scores: HashMap<String, (f64, usize)> = HashMap::new();
    for (key, &w) in &weights {
        if let Some(items) = interactions.get(key) {
            for item in items {
                let entry = scores.entry(item.clone()).or_insert((0.0, 0));
                entry.0 += w;
                entry.1 += 1;
            }
        }
    }

    // 4. Exclude owned
    let mut excluded_count = 0;
    if let Some(owned_set) = owned {
        for key in owned_set {
            if scores.remove(key).is_some() {
                excluded_count += 1;
            }
        }
    }

    // 5. Category mask
    if let Some(cat) = category {
        scores.retain(|k, _| cat.contains(k));
    }

    // 6. PerItem normalization
    if config.normalization == Normalization::PerItem {
        for (score, contributors) in scores.values_mut() {
            if *contributors > 0 {
                #[allow(clippy::cast_precision_loss)]
                let divisor = *contributors as f64;
                *score /= divisor;
            }
        }
    }

    // 7. Non-finite filter
    scores.retain(|_, (s, _)| s.is_finite());

    // 8. Sort + truncate
    let mut items: Vec<ScoredItem> = scores
        .into_iter()
        .map(|(item_key, (score, contributors))| ScoredItem {
            item_key,
            score,
            contributors,
        })
        .collect();

    items.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.item_key.cmp(&b.item_key))
    });

    items.truncate(config.top_k);

    ContractionResult {
        items,
        weight_norm,
        excluded_count,
    }
}

// ---------------------------------------------------------------------------
// Value codec + weight extraction
// ---------------------------------------------------------------------------

/// Convert a relational [`Value`] to a string key.
///
/// Only `String` and `Int` values are converted. `Float` columns are rejected
/// at schema validation, so they never reach this function. Returns `None` for
/// `Null` (skip this row) and any other type.
fn value_to_key(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::Int(i) => Some(i.to_string()),
        _ => None,
    }
}

/// Extract a numeric edge weight from an [`graph_engine::Edge`].
///
/// Returns the `"weight"` property if it exists and is numeric and finite,
/// otherwise 1.0.
fn extract_edge_weight(edge: &graph_engine::Edge) -> f64 {
    edge.properties.get("weight").map_or(1.0, |pv| match pv {
        PropertyValue::Float(f) if f.is_finite() => *f,
        PropertyValue::Int(i) => {
            #[allow(clippy::cast_precision_loss)]
            let w = *i as f64;
            w
        },
        _ => 1.0,
    })
}

/// Validate that a column type is usable as an entity key.
///
/// Only `String` and `Int` are accepted. `Float` is rejected because
/// `f64::to_string()` is not a stable key format (`1.0` vs `"1"` vs
/// `"0.30000000000000004"`), which silently breaks cross-modal joins.
///
/// # Errors
///
/// Returns `InvalidOperation` for `Float`, `Bool`, `Bytes`, or `Json` columns.
fn validate_key_column_type(
    column_name: &str,
    column_type: &ColumnType,
    table: &str,
) -> Result<()> {
    match column_type {
        ColumnType::String | ColumnType::Int => Ok(()),
        other => Err(UnifiedError::InvalidOperation(format!(
            "column type {other:?} not usable as entity key in column '{column_name}' of table '{table}'"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Engine adapter
// ---------------------------------------------------------------------------

impl UnifiedEngine {
    /// Gathers cross-modal data and computes tensor contraction scores.
    ///
    /// Collects graph adjacency, vector similarity, and relational interactions
    /// for `source_key`, then fuses them via [`contract`].
    ///
    /// # Arguments
    ///
    /// * `source_key` — Entity key to compute scores for.
    /// * `interaction_table` — Relational table containing interaction rows.
    /// * `source_column` — Column in `interaction_table` identifying the intermediary.
    /// * `target_column` — Column in `interaction_table` identifying the item.
    /// * `config` — Contraction configuration.
    /// * `category` — Optional category mask to filter results.
    ///
    /// # Errors
    ///
    /// Returns errors for missing tables, missing columns, incompatible
    /// column types, or query timeout. The interaction table is scanned in
    /// streaming fashion (one row at a time), so peak memory is proportional
    /// to the accumulated result maps, not the full table.
    #[allow(clippy::unused_async)]
    pub async fn cross_modal_contraction(
        &self,
        source_key: &str,
        interaction_table: &str,
        source_column: &str,
        target_column: &str,
        config: &ContractionConfig,
        category: Option<&CategoryMask>,
    ) -> Result<ContractionResult> {
        // Fast path: no results requested
        if config.top_k == 0 {
            return Ok(ContractionResult {
                items: Vec::new(),
                weight_norm: 0.0,
                excluded_count: 0,
            });
        }

        // Step 0: Schema validation
        let schema = self
            .relational()
            .get_schema(interaction_table)
            .map_err(|e| UnifiedError::RelationalError(e.to_string()))?;

        let src_col = schema.get_column(source_column).ok_or_else(|| {
            UnifiedError::InvalidOperation(format!(
                "column '{source_column}' not found in table '{interaction_table}'"
            ))
        })?;
        let tgt_col = schema.get_column(target_column).ok_or_else(|| {
            UnifiedError::InvalidOperation(format!(
                "column '{target_column}' not found in table '{interaction_table}'"
            ))
        })?;

        validate_key_column_type(source_column, &src_col.column_type, interaction_table)?;
        validate_key_column_type(target_column, &tgt_col.column_type, interaction_table)?;

        // Step 1: Adjacency
        let direction: Direction = config.direction.into();
        let adjacency = self.gather_adjacency(source_key, direction, config.edge_type.as_deref());

        if adjacency.is_empty() {
            return Ok(ContractionResult {
                items: Vec::new(),
                weight_norm: 0.0,
                excluded_count: 0,
            });
        }

        // Step 2: Similarity (only for neighbors in adjacency)
        let similarity = self.gather_similarity(source_key, &adjacency);

        if similarity.is_empty() {
            return Ok(ContractionResult {
                items: Vec::new(),
                weight_norm: 0.0,
                excluded_count: 0,
            });
        }

        // Step 3+4: Interactions and owned items (streaming table scan)
        let intermediary_keys: HashSet<&str> = adjacency.keys().map(String::as_str).collect();
        let (interactions, owned) = self.gather_interactions(
            interaction_table,
            source_column,
            target_column,
            &intermediary_keys,
            if config.exclude_owned {
                Some(source_key)
            } else {
                None
            },
        )?;

        // Step 5: Contract
        let owned_ref = if config.exclude_owned {
            Some(&owned)
        } else {
            None
        };

        Ok(contract(
            &adjacency,
            &similarity,
            &interactions,
            owned_ref,
            category,
            config,
        ))
    }

    /// Build adjacency vector from graph edges.
    ///
    /// When multiple edges connect the source to the same neighbor (e.g.
    /// parallel edges of the same type), their weights are summed.
    fn gather_adjacency(
        &self,
        source_key: &str,
        direction: Direction,
        edge_type_filter: Option<&str>,
    ) -> AdjacencyVec {
        let Some(node_id) = self.find_entity_node(source_key) else {
            return AdjacencyVec::new();
        };

        let Ok(edges) = self.graph().edges_of(node_id, direction) else {
            return AdjacencyVec::new();
        };

        let mut adj = AdjacencyVec::new();
        for edge in &edges {
            // Edge type filter
            if let Some(et) = edge_type_filter {
                if edge.edge_type != et {
                    continue;
                }
            }

            // Determine the "other" node, skipping self-loops
            let other_id = if edge.from == node_id {
                edge.to
            } else {
                edge.from
            };

            if other_id == node_id {
                continue;
            }

            if let Some(key) = self.get_entity_key(other_id) {
                let weight = extract_edge_weight(edge);
                *adj.entry(key).or_insert(0.0) += weight;
            }
        }
        adj
    }

    /// Build similarity vector for neighbors that have embeddings.
    fn gather_similarity(&self, source_key: &str, adjacency: &AdjacencyVec) -> SimilarityVec {
        let Ok(source_emb) = self.vector().get_entity_embedding(source_key) else {
            return SimilarityVec::new();
        };

        let mut sim = SimilarityVec::new();
        for key in adjacency.keys() {
            if let Ok(neighbor_emb) = self.vector().get_entity_embedding(key) {
                if let Ok(score) = VectorEngine::compute_similarity(&source_emb, &neighbor_emb) {
                    sim.insert(key.clone(), f64::from(score));
                }
            }
        }
        sim
    }

    /// Scan interaction table and build both `InteractionMap` and `OwnedSet`.
    ///
    /// Uses `for_each_row` for a streaming O(n) scan that processes rows
    /// one at a time through a callback. Peak memory is proportional to
    /// a single row plus the accumulated maps, not the full table.
    ///
    /// Inherits the relational engine's default query timeout.
    fn gather_interactions(
        &self,
        table: &str,
        source_column: &str,
        target_column: &str,
        intermediary_keys: &HashSet<&str>,
        source_key_for_owned: Option<&str>,
    ) -> Result<(InteractionMap, OwnedSet)> {
        let mut interactions = InteractionMap::new();
        let mut owned = OwnedSet::new();

        self.relational()
            .for_each_row(table, None, None, &mut |row| {
                let Some(src_val) = row.get(source_column) else {
                    return;
                };
                let Some(src_key) = value_to_key(src_val) else {
                    return;
                };
                let Some(tgt_val) = row.get(target_column) else {
                    return;
                };
                let Some(tgt_key) = value_to_key(tgt_val) else {
                    return;
                };

                // Intermediary interactions
                if intermediary_keys.contains(src_key.as_str()) {
                    interactions
                        .entry(src_key.clone())
                        .or_default()
                        .insert(tgt_key.clone());
                }

                // Owned items (source's own rows)
                if let Some(sk) = source_key_for_owned {
                    if src_key == sk {
                        owned.insert(tgt_key);
                    }
                }
            })
            .map_err(|e| UnifiedError::RelationalError(e.to_string()))?;

        Ok((interactions, owned))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for floating-point comparisons in tests.
    const TOL: f64 = 1e-10;

    // -----------------------------------------------------------------------
    // Config / Enum tests
    // -----------------------------------------------------------------------

    #[test]
    fn graph_direction_to_direction_symmetric() {
        let d: Direction = GraphDirection::Symmetric.into();
        assert_eq!(d, Direction::Both);
    }

    #[test]
    fn graph_direction_to_direction_outgoing() {
        let d: Direction = GraphDirection::Outgoing.into();
        assert_eq!(d, Direction::Outgoing);
    }

    #[test]
    fn graph_direction_to_direction_incoming() {
        let d: Direction = GraphDirection::Incoming.into();
        assert_eq!(d, Direction::Incoming);
    }

    #[test]
    fn normalization_variants_are_distinct() {
        assert_ne!(Normalization::None, Normalization::TotalWeight);
        assert_ne!(Normalization::None, Normalization::PerItem);
        assert_ne!(Normalization::TotalWeight, Normalization::PerItem);
    }

    // -----------------------------------------------------------------------
    // hadamard_product tests
    // -----------------------------------------------------------------------

    #[test]
    fn hadamard_both_empty() {
        let adj = AdjacencyVec::new();
        let sim = SimilarityVec::new();
        assert!(hadamard_product(&adj, &sim).is_empty());
    }

    #[test]
    fn hadamard_adjacency_empty() {
        let adj = AdjacencyVec::new();
        let sim = SimilarityVec::from([("a".into(), 0.5)]);
        assert!(hadamard_product(&adj, &sim).is_empty());
    }

    #[test]
    fn hadamard_similarity_empty() {
        let adj = AdjacencyVec::from([("a".into(), 1.0)]);
        let sim = SimilarityVec::new();
        assert!(hadamard_product(&adj, &sim).is_empty());
    }

    #[test]
    fn hadamard_disjoint_keys() {
        let adj = AdjacencyVec::from([("a".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.5)]);
        assert!(hadamard_product(&adj, &sim).is_empty());
    }

    #[test]
    fn hadamard_partial_overlap() {
        let adj = AdjacencyVec::from([("a".into(), 2.0), ("b".into(), 3.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.5), ("c".into(), 1.0)]);
        let result = hadamard_product(&adj, &sim);
        assert_eq!(result.len(), 1);
        assert!((result["b"] - 1.5).abs() < TOL);
    }

    #[test]
    fn hadamard_full_overlap() {
        let adj = AdjacencyVec::from([("a".into(), 2.0), ("b".into(), 3.0)]);
        let sim = SimilarityVec::from([("a".into(), 0.5), ("b".into(), 0.4)]);
        let result = hadamard_product(&adj, &sim);
        assert_eq!(result.len(), 2);
        assert!((result["a"] - 1.0).abs() < TOL);
        assert!((result["b"] - 1.2).abs() < TOL);
    }

    #[test]
    fn hadamard_nan_in_adjacency_skipped() {
        let adj = AdjacencyVec::from([("a".into(), f64::NAN)]);
        let sim = SimilarityVec::from([("a".into(), 0.5)]);
        assert!(hadamard_product(&adj, &sim).is_empty());
    }

    #[test]
    fn hadamard_inf_in_similarity_skipped() {
        let adj = AdjacencyVec::from([("a".into(), 1.0)]);
        let sim = SimilarityVec::from([("a".into(), f64::INFINITY)]);
        assert!(hadamard_product(&adj, &sim).is_empty());
    }

    #[test]
    fn hadamard_negative_values() {
        let adj = AdjacencyVec::from([("a".into(), -2.0)]);
        let sim = SimilarityVec::from([("a".into(), 0.5)]);
        let result = hadamard_product(&adj, &sim);
        assert!((result["a"] - (-1.0)).abs() < TOL);
    }

    // -----------------------------------------------------------------------
    // normalize_weights tests
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_empty_map() {
        let mut w = HashMap::new();
        let norm = normalize_weights(&mut w, Normalization::TotalWeight);
        assert!(norm.abs() < TOL);
    }

    #[test]
    fn normalize_single_positive() {
        let mut w = HashMap::from([("a".into(), 4.0)]);
        let norm = normalize_weights(&mut w, Normalization::TotalWeight);
        assert!((norm - 4.0).abs() < TOL);
        assert!((w["a"] - 1.0).abs() < TOL);
    }

    #[test]
    fn normalize_mixed_signs_uses_l1() {
        let mut w = HashMap::from([("a".into(), 3.0), ("b".into(), -1.0)]);
        let norm = normalize_weights(&mut w, Normalization::TotalWeight);
        assert!((norm - 4.0).abs() < TOL);
        assert!((w["a"] - 0.75).abs() < TOL);
        assert!((w["b"] - (-0.25)).abs() < TOL);
    }

    #[test]
    fn normalize_all_zero_passes_through() {
        let mut w = HashMap::from([("a".into(), 0.0), ("b".into(), 0.0)]);
        normalize_weights(&mut w, Normalization::TotalWeight);
        assert!(w["a"].abs() < TOL);
        assert!(w["b"].abs() < TOL);
    }

    #[test]
    fn normalize_none_is_noop() {
        let mut w = HashMap::from([("a".into(), 5.0)]);
        let norm = normalize_weights(&mut w, Normalization::None);
        assert!(norm.abs() < TOL);
        assert!((w["a"] - 5.0).abs() < TOL);
    }

    #[test]
    fn normalize_per_item_is_noop_at_weight_stage() {
        let mut w = HashMap::from([("a".into(), 5.0)]);
        let norm = normalize_weights(&mut w, Normalization::PerItem);
        assert!(norm.abs() < TOL);
        assert!((w["a"] - 5.0).abs() < TOL);
    }

    #[test]
    fn normalize_large_values_no_overflow() {
        let mut w = HashMap::from([("a".into(), 1e300), ("b".into(), 1e300)]);
        let norm = normalize_weights(&mut w, Normalization::TotalWeight);
        assert!((norm - 2e300).abs() < 1e285);
        assert!((w["a"] - 0.5).abs() < TOL);
    }

    // -----------------------------------------------------------------------
    // contract() tests
    // -----------------------------------------------------------------------

    fn default_config() -> ContractionConfig {
        ContractionConfig {
            direction: GraphDirection::Symmetric,
            normalization: Normalization::None,
            edge_type: None,
            exclude_owned: false,
            top_k: 100,
        }
    }

    #[test]
    fn contract_all_empty() {
        let result = contract(
            &AdjacencyVec::new(),
            &SimilarityVec::new(),
            &InteractionMap::new(),
            None,
            None,
            &default_config(),
        );
        assert!(result.items.is_empty());
        assert!(result.weight_norm.abs() < TOL);
        assert_eq!(result.excluded_count, 0);
    }

    #[test]
    fn contract_adjacency_only_no_similarity() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let result = contract(
            &adj,
            &SimilarityVec::new(),
            &InteractionMap::new(),
            None,
            None,
            &default_config(),
        );
        assert!(result.items.is_empty());
    }

    #[test]
    fn contract_similarity_only_no_adjacency() {
        let sim = SimilarityVec::from([("b".into(), 0.9)]);
        let result = contract(
            &AdjacencyVec::new(),
            &sim,
            &InteractionMap::new(),
            None,
            None,
            &default_config(),
        );
        assert!(result.items.is_empty());
    }

    #[test]
    fn contract_no_interactions() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.9)]);
        let result = contract(
            &adj,
            &sim,
            &InteractionMap::new(),
            None,
            None,
            &default_config(),
        );
        assert!(result.items.is_empty());
    }

    #[test]
    fn contract_minimal_one_intermediary_one_item() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions = InteractionMap::from([("b".into(), HashSet::from(["item1".into()]))]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].item_key, "item1");
        assert!((result.items[0].score - 0.8).abs() < TOL);
        assert_eq!(result.items[0].contributors, 1);
    }

    #[test]
    fn contract_multiple_contributors_same_item() {
        let adj = AdjacencyVec::from([("b".into(), 1.0), ("c".into(), 1.0), ("d".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.3), ("c".into(), 0.5), ("d".into(), 0.2)]);
        let interactions = InteractionMap::from([
            ("b".into(), HashSet::from(["item1".into()])),
            ("c".into(), HashSet::from(["item1".into()])),
            ("d".into(), HashSet::from(["item1".into()])),
        ]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 1);
        assert!((result.items[0].score - 1.0).abs() < TOL);
        assert_eq!(result.items[0].contributors, 3);
    }

    #[test]
    fn contract_disjoint_intermediaries_different_items() {
        let adj = AdjacencyVec::from([("b".into(), 1.0), ("c".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.6), ("c".into(), 0.4)]);
        let interactions = InteractionMap::from([
            ("b".into(), HashSet::from(["item1".into()])),
            ("c".into(), HashSet::from(["item2".into()])),
        ]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 2);
        // Sorted descending by score
        assert_eq!(result.items[0].item_key, "item1");
        assert!((result.items[0].score - 0.6).abs() < TOL);
        assert_eq!(result.items[1].item_key, "item2");
        assert!((result.items[1].score - 0.4).abs() < TOL);
    }

    #[test]
    fn contract_normalization_total_weight() {
        let adj = AdjacencyVec::from([("b".into(), 1.0), ("c".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.6), ("c".into(), 0.4)]);
        let interactions = InteractionMap::from([
            ("b".into(), HashSet::from(["item1".into()])),
            ("c".into(), HashSet::from(["item2".into()])),
        ]);

        let mut config = default_config();
        config.normalization = Normalization::TotalWeight;

        let result = contract(&adj, &sim, &interactions, None, None, &config);
        assert!((result.weight_norm - 1.0).abs() < TOL);
        assert_eq!(result.items.len(), 2);
        assert!((result.items[0].score - 0.6).abs() < TOL);
        assert!((result.items[1].score - 0.4).abs() < TOL);
    }

    #[test]
    fn contract_normalization_per_item() {
        let adj = AdjacencyVec::from([("b".into(), 1.0), ("c".into(), 1.0), ("d".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.9), ("c".into(), 0.6), ("d".into(), 0.3)]);
        // item1 contributed by all 3, item2 by only d
        let interactions = InteractionMap::from([
            ("b".into(), HashSet::from(["item1".into()])),
            ("c".into(), HashSet::from(["item1".into()])),
            ("d".into(), HashSet::from(["item1".into(), "item2".into()])),
        ]);

        let mut config = default_config();
        config.normalization = Normalization::PerItem;

        let result = contract(&adj, &sim, &interactions, None, None, &config);
        // item1 raw score = 0.9+0.6+0.3 = 1.8, contributors=3, per-item = 0.6
        // item2 raw score = 0.3, contributors=1, per-item = 0.3
        assert_eq!(result.items.len(), 2);
        assert!((result.items[0].score - 0.6).abs() < TOL);
        assert!((result.items[1].score - 0.3).abs() < TOL);
    }

    #[test]
    fn contract_exclude_owned_some() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions =
            InteractionMap::from([("b".into(), HashSet::from(["item1".into(), "item2".into()]))]);
        let owned = OwnedSet::from(["item1".into()]);

        let mut config = default_config();
        config.exclude_owned = true;

        let result = contract(&adj, &sim, &interactions, Some(&owned), None, &config);
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].item_key, "item2");
        assert_eq!(result.excluded_count, 1);
    }

    #[test]
    fn contract_exclude_owned_none() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions =
            InteractionMap::from([("b".into(), HashSet::from(["item1".into(), "item2".into()]))]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 2);
        assert_eq!(result.excluded_count, 0);
    }

    #[test]
    fn contract_category_mask_some() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions = InteractionMap::from([(
            "b".into(),
            HashSet::from(["item1".into(), "item2".into(), "item3".into()]),
        )]);
        let cat = CategoryMask::from(["item1".into(), "item3".into()]);

        let result = contract(
            &adj,
            &sim,
            &interactions,
            None,
            Some(&cat),
            &default_config(),
        );
        assert_eq!(result.items.len(), 2);
        let keys: HashSet<_> = result.items.iter().map(|i| i.item_key.as_str()).collect();
        assert!(keys.contains("item1"));
        assert!(keys.contains("item3"));
    }

    #[test]
    fn contract_category_mask_empty_set() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions = InteractionMap::from([("b".into(), HashSet::from(["item1".into()]))]);
        let cat = CategoryMask::new();

        let result = contract(
            &adj,
            &sim,
            &interactions,
            None,
            Some(&cat),
            &default_config(),
        );
        assert!(result.items.is_empty());
    }

    #[test]
    fn contract_category_mask_none_passes_all() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions =
            InteractionMap::from([("b".into(), HashSet::from(["item1".into(), "item2".into()]))]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 2);
    }

    #[test]
    fn contract_top_k_truncates() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions = InteractionMap::from([(
            "b".into(),
            HashSet::from(["i1".into(), "i2".into(), "i3".into()]),
        )]);

        let mut config = default_config();
        config.top_k = 2;

        let result = contract(&adj, &sim, &interactions, None, None, &config);
        assert_eq!(result.items.len(), 2);
    }

    #[test]
    fn contract_top_k_larger_than_results() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions = InteractionMap::from([("b".into(), HashSet::from(["item1".into()]))]);

        let mut config = default_config();
        config.top_k = 100;

        let result = contract(&adj, &sim, &interactions, None, None, &config);
        assert_eq!(result.items.len(), 1);
    }

    #[test]
    fn contract_top_k_zero() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions = InteractionMap::from([("b".into(), HashSet::from(["item1".into()]))]);

        let mut config = default_config();
        config.top_k = 0;

        let result = contract(&adj, &sim, &interactions, None, None, &config);
        assert!(result.items.is_empty());
    }

    #[test]
    fn contract_score_ordering_descending() {
        let adj = AdjacencyVec::from([("b".into(), 1.0), ("c".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.3), ("c".into(), 0.9)]);
        let interactions = InteractionMap::from([
            ("b".into(), HashSet::from(["low".into()])),
            ("c".into(), HashSet::from(["high".into()])),
        ]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items[0].item_key, "high");
        assert_eq!(result.items[1].item_key, "low");
    }

    #[test]
    fn contract_deterministic_tie_break_by_key() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 1.0)]);
        let interactions = InteractionMap::from([(
            "b".into(),
            HashSet::from(["zebra".into(), "apple".into(), "mango".into()]),
        )]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 3);
        // All have same score, so sorted ascending by key
        assert_eq!(result.items[0].item_key, "apple");
        assert_eq!(result.items[1].item_key, "mango");
        assert_eq!(result.items[2].item_key, "zebra");
    }

    #[test]
    fn contract_negative_similarities() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), -0.5)]);
        let interactions = InteractionMap::from([("b".into(), HashSet::from(["item1".into()]))]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 1);
        assert!((result.items[0].score - (-0.5)).abs() < TOL);
    }

    #[test]
    fn contract_all_zero_similarities() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.0)]);
        let interactions = InteractionMap::from([("b".into(), HashSet::from(["item1".into()]))]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 1);
        assert!(result.items[0].score.abs() < TOL);
    }

    #[test]
    fn contract_weight_norm_populated() {
        let adj = AdjacencyVec::from([("b".into(), 1.0), ("c".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.6), ("c".into(), 0.4)]);
        let interactions = InteractionMap::from([
            ("b".into(), HashSet::from(["item1".into()])),
            ("c".into(), HashSet::from(["item2".into()])),
        ]);

        let mut config = default_config();
        config.normalization = Normalization::TotalWeight;

        let result = contract(&adj, &sim, &interactions, None, None, &config);
        assert!((result.weight_norm - 1.0).abs() < TOL);
    }

    #[test]
    fn contract_weight_norm_zero_when_no_normalization() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions = InteractionMap::from([("b".into(), HashSet::from(["item1".into()]))]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert!(result.weight_norm.abs() < TOL);
    }

    #[test]
    fn contract_non_finite_output_filtered() {
        // If somehow a NaN score makes it through contraction (e.g., 0*inf
        // slipping past hadamard guard), verify it's filtered.
        let mut adj = AdjacencyVec::from([("b".into(), 1.0), ("c".into(), 1.0)]);
        let mut sim = SimilarityVec::from([("b".into(), 0.5), ("c".into(), 0.5)]);

        // Manually inject a NaN weight after hadamard by using a custom
        // interactions map that produces NaN via accumulation.
        // Since our hadamard filters non-finite, we test the output filter
        // indirectly: all products are finite here.
        let interactions = InteractionMap::from([
            ("b".into(), HashSet::from(["item1".into()])),
            ("c".into(), HashSet::from(["item1".into()])),
        ]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        // All items should have finite scores
        for item in &result.items {
            assert!(item.score.is_finite());
        }

        // Now test with NaN in adjacency - should produce no output
        adj.insert("d".into(), f64::NAN);
        sim.insert("d".into(), 0.5);
        let interactions2 =
            InteractionMap::from([("d".into(), HashSet::from(["item_nan".into()]))]);
        let result2 = contract(&adj, &sim, &interactions2, None, None, &default_config());
        // item_nan should not appear (NaN adjacency skipped in hadamard)
        assert!(!result2.items.iter().any(|i| i.item_key == "item_nan"));
    }

    #[test]
    fn contract_single_intermediary_many_items() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.7)]);
        let items: HashSet<String> = (0..20).map(|i| format!("item{i}")).collect();
        let interactions = InteractionMap::from([("b".into(), items)]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 20);
        for item in &result.items {
            assert!((item.score - 0.7).abs() < TOL);
            assert_eq!(item.contributors, 1);
        }
    }

    #[test]
    fn contract_many_intermediaries_single_item() {
        let n = 50;
        let adj: AdjacencyVec = (0..n).map(|i| (format!("n{i}"), 1.0)).collect();
        let sim: SimilarityVec = (0..n).map(|i| (format!("n{i}"), 0.1)).collect();
        let interactions: InteractionMap = (0..n)
            .map(|i| (format!("n{i}"), HashSet::from(["item".into()])))
            .collect();

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        assert_eq!(result.items.len(), 1);
        assert!((result.items[0].score - 5.0).abs() < TOL);
        assert_eq!(result.items[0].contributors, 50);
    }

    #[test]
    fn contract_scale_500_intermediaries_5000_items() {
        let adj: AdjacencyVec = (0..500).map(|i| (format!("n{i}"), 1.0)).collect();
        let sim: SimilarityVec = (0..500).map(|i| (format!("n{i}"), 0.5)).collect();
        let interactions: InteractionMap = (0..500)
            .map(|i| {
                let items: HashSet<String> =
                    (0..10).map(|j| format!("item{}", i * 10 + j)).collect();
                (format!("n{i}"), items)
            })
            .collect();

        let mut config = default_config();
        config.top_k = 10;

        let result = contract(&adj, &sim, &interactions, None, None, &config);
        assert_eq!(result.items.len(), 10);
        // All scores should be 0.5 (each item from exactly one intermediary)
        for item in &result.items {
            assert!((item.score - 0.5).abs() < TOL);
        }
    }

    #[test]
    fn contract_exclude_owned_and_category_combined() {
        let adj = AdjacencyVec::from([("b".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8)]);
        let interactions = InteractionMap::from([(
            "b".into(),
            HashSet::from(["owned1".into(), "cat_ok".into(), "no_cat".into()]),
        )]);
        let owned = OwnedSet::from(["owned1".into()]);
        let cat = CategoryMask::from(["cat_ok".into(), "owned1".into()]);

        let mut config = default_config();
        config.exclude_owned = true;

        let result = contract(&adj, &sim, &interactions, Some(&owned), Some(&cat), &config);
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].item_key, "cat_ok");
        assert_eq!(result.excluded_count, 1);
    }

    #[test]
    fn contract_total_weight_with_negative_similarity() {
        let adj = AdjacencyVec::from([("b".into(), 1.0), ("c".into(), 1.0)]);
        let sim = SimilarityVec::from([("b".into(), 0.8), ("c".into(), -0.2)]);
        let interactions = InteractionMap::from([
            ("b".into(), HashSet::from(["item1".into()])),
            ("c".into(), HashSet::from(["item2".into()])),
        ]);

        let mut config = default_config();
        config.normalization = Normalization::TotalWeight;

        let result = contract(&adj, &sim, &interactions, None, None, &config);
        // L1 norm = |0.8| + |-0.2| = 1.0
        assert!((result.weight_norm - 1.0).abs() < TOL);
        assert!((result.items[0].score - 0.8).abs() < TOL);
        assert!((result.items[1].score - (-0.2)).abs() < TOL);
    }

    #[test]
    fn contract_weighted_edges() {
        let adj = AdjacencyVec::from([("b".into(), 2.5), ("c".into(), 0.5)]);
        let sim = SimilarityVec::from([("b".into(), 0.8), ("c".into(), 0.8)]);
        let interactions = InteractionMap::from([
            ("b".into(), HashSet::from(["item1".into()])),
            ("c".into(), HashSet::from(["item1".into()])),
        ]);

        let result = contract(&adj, &sim, &interactions, None, None, &default_config());
        // item1 score = 2.5*0.8 + 0.5*0.8 = 2.0 + 0.4 = 2.4
        assert_eq!(result.items.len(), 1);
        assert!((result.items[0].score - 2.4).abs() < TOL);
        assert_eq!(result.items[0].contributors, 2);
    }

    // -----------------------------------------------------------------------
    // value_to_key tests
    // -----------------------------------------------------------------------

    #[test]
    fn value_to_key_string() {
        assert_eq!(
            value_to_key(&Value::String("hello".into())),
            Some("hello".into())
        );
    }

    #[test]
    fn value_to_key_int() {
        assert_eq!(value_to_key(&Value::Int(42)), Some("42".into()));
    }

    #[test]
    fn value_to_key_float_returns_none() {
        assert_eq!(value_to_key(&Value::Float(3.14)), None);
    }

    #[test]
    fn value_to_key_null() {
        assert_eq!(value_to_key(&Value::Null), None);
    }

    #[test]
    fn value_to_key_bool_returns_none() {
        assert_eq!(value_to_key(&Value::Bool(true)), None);
    }

    #[test]
    fn value_to_key_bytes_returns_none() {
        assert_eq!(value_to_key(&Value::Bytes(vec![1, 2, 3])), None);
    }

    // -----------------------------------------------------------------------
    // extract_edge_weight tests
    // -----------------------------------------------------------------------

    #[test]
    fn extract_weight_float_property() {
        let edge = graph_engine::Edge {
            id: 1,
            from: 0,
            to: 1,
            edge_type: "TEST".into(),
            properties: HashMap::from([("weight".into(), PropertyValue::Float(2.5))]),
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!((extract_edge_weight(&edge) - 2.5).abs() < TOL);
    }

    #[test]
    fn extract_weight_int_property() {
        let edge = graph_engine::Edge {
            id: 1,
            from: 0,
            to: 1,
            edge_type: "TEST".into(),
            properties: HashMap::from([("weight".into(), PropertyValue::Int(3))]),
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!((extract_edge_weight(&edge) - 3.0).abs() < TOL);
    }

    #[test]
    fn extract_weight_non_numeric_defaults() {
        let edge = graph_engine::Edge {
            id: 1,
            from: 0,
            to: 1,
            edge_type: "TEST".into(),
            properties: HashMap::from([("weight".into(), PropertyValue::String("heavy".into()))]),
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!((extract_edge_weight(&edge) - 1.0).abs() < TOL);
    }

    #[test]
    fn extract_weight_no_weight_property() {
        let edge = graph_engine::Edge {
            id: 1,
            from: 0,
            to: 1,
            edge_type: "TEST".into(),
            properties: HashMap::new(),
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!((extract_edge_weight(&edge) - 1.0).abs() < TOL);
    }

    #[test]
    fn extract_weight_nan_defaults() {
        let edge = graph_engine::Edge {
            id: 1,
            from: 0,
            to: 1,
            edge_type: "TEST".into(),
            properties: HashMap::from([("weight".into(), PropertyValue::Float(f64::NAN))]),
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!((extract_edge_weight(&edge) - 1.0).abs() < TOL);
    }

    #[test]
    fn extract_weight_inf_defaults() {
        let edge = graph_engine::Edge {
            id: 1,
            from: 0,
            to: 1,
            edge_type: "TEST".into(),
            properties: HashMap::from([("weight".into(), PropertyValue::Float(f64::INFINITY))]),
            directed: true,
            created_at: None,
            updated_at: None,
        };
        assert!((extract_edge_weight(&edge) - 1.0).abs() < TOL);
    }

    // -----------------------------------------------------------------------
    // validate_key_column_type tests
    // -----------------------------------------------------------------------

    #[test]
    fn validate_column_type_string_ok() {
        assert!(validate_key_column_type("col", &ColumnType::String, "t").is_ok());
    }

    #[test]
    fn validate_column_type_int_ok() {
        assert!(validate_key_column_type("col", &ColumnType::Int, "t").is_ok());
    }

    #[test]
    fn validate_column_type_float_err() {
        assert!(validate_key_column_type("col", &ColumnType::Float, "t").is_err());
    }

    #[test]
    fn validate_column_type_bool_err() {
        assert!(validate_key_column_type("col", &ColumnType::Bool, "t").is_err());
    }

    #[test]
    fn validate_column_type_bytes_err() {
        assert!(validate_key_column_type("col", &ColumnType::Bytes, "t").is_err());
    }

    #[test]
    fn validate_column_type_json_err() {
        assert!(validate_key_column_type("col", &ColumnType::Json, "t").is_err());
    }

    // -----------------------------------------------------------------------
    // Engine adapter tests (using real engines)
    // -----------------------------------------------------------------------

    /// Helper to create a [`UnifiedEngine`] with a populated graph, vector, and
    /// relational store suitable for contraction tests.
    ///
    /// Setup:
    /// - Graph: alice -> {bob, carol, dave} via "FRIEND" edges, eve via "COWORKER"
    /// - Embeddings: alice=[1,0,0], bob=[0.9,0.1,0], carol=[0.5,0.5,0], dave=[0,1,0]
    ///   (eve has no embedding)
    /// - Table "purchases": columns (buyer: String, item: String)
    ///   Rows: bob->book, bob->pen, carol->pen, carol->laptop,
    ///         dave->phone, alice->book (alice owns book)
    async fn setup_engine() -> UnifiedEngine {
        let engine = UnifiedEngine::new();

        // Create entities (creates graph nodes with entity_key property)
        for key in &["alice", "bob", "carol", "dave", "eve"] {
            engine
                .create_entity(key, HashMap::new(), None)
                .await
                .unwrap();
        }

        // Connect via FRIEND edges
        engine
            .connect_entities("alice", "bob", "FRIEND")
            .await
            .unwrap();
        engine
            .connect_entities("alice", "carol", "FRIEND")
            .await
            .unwrap();
        engine
            .connect_entities("alice", "dave", "FRIEND")
            .await
            .unwrap();
        // eve is a COWORKER, not FRIEND
        engine
            .connect_entities("alice", "eve", "COWORKER")
            .await
            .unwrap();

        // Store embeddings
        engine
            .vector()
            .set_entity_embedding("alice", vec![1.0, 0.0, 0.0])
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("bob", vec![0.9, 0.1, 0.0])
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("carol", vec![0.5, 0.5, 0.0])
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("dave", vec![0.0, 1.0, 0.0])
            .unwrap();
        // eve has no embedding

        // Create purchases table
        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "buyer".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "item".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine
            .relational()
            .create_table("purchases", schema)
            .unwrap();

        // Insert purchase rows
        let purchases = [
            ("bob", "book"),
            ("bob", "pen"),
            ("carol", "pen"),
            ("carol", "laptop"),
            ("dave", "phone"),
            ("alice", "book"),
        ];
        for (buyer, item) in &purchases {
            engine
                .relational()
                .insert(
                    "purchases",
                    HashMap::from([
                        ("buyer".into(), Value::String((*buyer).into())),
                        ("item".into(), Value::String((*item).into())),
                    ]),
                )
                .unwrap();
        }

        engine
    }

    fn friend_config() -> ContractionConfig {
        ContractionConfig {
            direction: GraphDirection::Symmetric,
            normalization: Normalization::None,
            edge_type: Some("FRIEND".into()),
            exclude_owned: false,
            top_k: 10,
        }
    }

    #[tokio::test]
    async fn adapter_basic_end_to_end() {
        let engine = setup_engine().await;
        let config = friend_config();

        let result = engine
            .cross_modal_contraction("alice", "purchases", "buyer", "item", &config, None)
            .await
            .unwrap();

        // Expected similarities: cos(alice=[1,0,0], bob=[0.9,0.1,0])≈0.9939,
        //   cos(alice, carol=[0.5,0.5,0])≈0.7071, cos(alice, dave=[0,1,0])=0.0
        // Scores: pen=0.9939+0.7071≈1.701, book=0.9939, laptop=0.7071, phone=0.0
        assert_eq!(result.items.len(), 4);
        assert_eq!(result.items[0].item_key, "pen");
        assert_eq!(result.items[0].contributors, 2);
        assert!((result.items[0].score - 1.701).abs() < 0.001);
        assert_eq!(result.items[1].item_key, "book");
        assert_eq!(result.items[1].contributors, 1);
        assert!((result.items[1].score - 0.994).abs() < 0.001);
        assert_eq!(result.items[2].item_key, "laptop");
        assert_eq!(result.items[2].contributors, 1);
        assert!((result.items[2].score - 0.707).abs() < 0.001);
        assert_eq!(result.items[3].item_key, "phone");
        assert_eq!(result.items[3].contributors, 1);
        assert!(result.items[3].score.abs() < TOL);
    }

    #[tokio::test]
    async fn adapter_empty_graph_no_neighbors() {
        let engine = UnifiedEngine::new();
        engine
            .create_entity("lonely", HashMap::new(), None)
            .await
            .unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("lonely", "t", "src", "tgt", &config, None)
            .await
            .unwrap();

        assert!(result.items.is_empty());
    }

    #[tokio::test]
    async fn adapter_no_embeddings() {
        let engine = UnifiedEngine::new();
        for key in &["a", "b"] {
            engine
                .create_entity(key, HashMap::new(), None)
                .await
                .unwrap();
        }
        engine.connect_entities("a", "b", "FRIEND").await.unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();
        engine
            .relational()
            .insert(
                "t",
                HashMap::from([
                    ("src".into(), Value::String("b".into())),
                    ("tgt".into(), Value::String("item1".into())),
                ]),
            )
            .unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("a", "t", "src", "tgt", &config, None)
            .await
            .unwrap();

        // No embeddings means empty similarity -> empty results
        assert!(result.items.is_empty());
    }

    #[tokio::test]
    async fn adapter_edge_type_filtering() {
        let engine = setup_engine().await;

        // With FRIEND filter: eve excluded
        let config = friend_config();
        let result = engine
            .cross_modal_contraction("alice", "purchases", "buyer", "item", &config, None)
            .await
            .unwrap();

        // eve has no embedding and no purchases anyway, but verify edge type works
        // by checking that only bob/carol/dave items appear
        let keys: HashSet<_> = result.items.iter().map(|i| i.item_key.as_str()).collect();
        for key in &keys {
            assert!(["book", "pen", "laptop", "phone"].contains(key));
        }

        // With no edge type filter: all neighbors included
        let mut config_all = friend_config();
        config_all.edge_type = None;
        let result_all = engine
            .cross_modal_contraction("alice", "purchases", "buyer", "item", &config_all, None)
            .await
            .unwrap();
        // Should produce same items since eve has no embedding anyway
        assert_eq!(result.items.len(), result_all.items.len());
    }

    #[tokio::test]
    async fn adapter_directed_edges() {
        let engine = UnifiedEngine::new();
        for key in &["a", "b", "c"] {
            engine
                .create_entity(key, HashMap::new(), None)
                .await
                .unwrap();
        }
        // a -> b (outgoing from a), c -> a (incoming to a)
        engine.connect_entities("a", "b", "FRIEND").await.unwrap();
        engine.connect_entities("c", "a", "FRIEND").await.unwrap();

        engine
            .vector()
            .set_entity_embedding("a", vec![1.0, 0.0])
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("b", vec![0.9, 0.1])
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("c", vec![0.8, 0.2])
            .unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();
        engine
            .relational()
            .insert(
                "t",
                HashMap::from([
                    ("src".into(), Value::String("b".into())),
                    ("tgt".into(), Value::String("item_b".into())),
                ]),
            )
            .unwrap();
        engine
            .relational()
            .insert(
                "t",
                HashMap::from([
                    ("src".into(), Value::String("c".into())),
                    ("tgt".into(), Value::String("item_c".into())),
                ]),
            )
            .unwrap();

        // Outgoing: only b
        let mut config = friend_config();
        config.direction = GraphDirection::Outgoing;
        let result_out = engine
            .cross_modal_contraction("a", "t", "src", "tgt", &config, None)
            .await
            .unwrap();
        assert_eq!(result_out.items.len(), 1);
        assert_eq!(result_out.items[0].item_key, "item_b");

        // Incoming: only c
        config.direction = GraphDirection::Incoming;
        let result_in = engine
            .cross_modal_contraction("a", "t", "src", "tgt", &config, None)
            .await
            .unwrap();
        assert_eq!(result_in.items.len(), 1);
        assert_eq!(result_in.items[0].item_key, "item_c");

        // Symmetric: both b and c
        config.direction = GraphDirection::Symmetric;
        let result_sym = engine
            .cross_modal_contraction("a", "t", "src", "tgt", &config, None)
            .await
            .unwrap();
        assert_eq!(result_sym.items.len(), 2);
    }

    #[tokio::test]
    async fn adapter_edge_weights_propagate() {
        let engine = UnifiedEngine::new();
        for key in &["a", "b"] {
            engine
                .create_entity(key, HashMap::new(), None)
                .await
                .unwrap();
        }
        // Connect first (creates graph nodes)
        engine.connect_entities("a", "b", "FRIEND").await.unwrap();

        // Update the edge with a weight property
        let a_node = engine.find_entity_node("a").unwrap();
        let edges = engine
            .graph()
            .edges_of(a_node, Direction::Outgoing)
            .unwrap();
        let edge_id = edges[0].id;
        engine
            .graph()
            .update_edge(
                edge_id,
                HashMap::from([("weight".into(), PropertyValue::Float(3.0))]),
            )
            .unwrap();

        engine
            .vector()
            .set_entity_embedding("a", vec![1.0, 0.0])
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("b", vec![1.0, 0.0])
            .unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();
        engine
            .relational()
            .insert(
                "t",
                HashMap::from([
                    ("src".into(), Value::String("b".into())),
                    ("tgt".into(), Value::String("item1".into())),
                ]),
            )
            .unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("a", "t", "src", "tgt", &config, None)
            .await
            .unwrap();

        // Score = weight(3.0) * similarity(1.0) = 3.0
        assert_eq!(result.items.len(), 1);
        assert!((result.items[0].score - 3.0).abs() < TOL);
    }

    #[tokio::test]
    async fn adapter_exclude_owned() {
        let engine = setup_engine().await;

        let mut config = friend_config();
        config.exclude_owned = true;

        let result = engine
            .cross_modal_contraction("alice", "purchases", "buyer", "item", &config, None)
            .await
            .unwrap();

        // alice owns "book", so it should be excluded
        assert!(!result.items.iter().any(|i| i.item_key == "book"));
        assert!(result.excluded_count > 0);
    }

    #[tokio::test]
    async fn adapter_category_mask() {
        let engine = setup_engine().await;
        let config = friend_config();
        let cat = CategoryMask::from(["pen".into(), "phone".into()]);

        let result = engine
            .cross_modal_contraction("alice", "purchases", "buyer", "item", &config, Some(&cat))
            .await
            .unwrap();

        let keys: HashSet<_> = result.items.iter().map(|i| i.item_key.as_str()).collect();
        assert!(keys.is_subset(&HashSet::from(["pen", "phone"])));
    }

    #[tokio::test]
    async fn adapter_source_not_found_empty_result() {
        let engine = setup_engine().await;
        let config = friend_config();

        let result = engine
            .cross_modal_contraction("nonexistent", "purchases", "buyer", "item", &config, None)
            .await
            .unwrap();

        assert!(result.items.is_empty());
    }

    #[tokio::test]
    async fn adapter_table_not_found_error() {
        let engine = setup_engine().await;
        let config = friend_config();

        let result = engine
            .cross_modal_contraction("alice", "nonexistent_table", "buyer", "item", &config, None)
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn adapter_missing_source_column_error() {
        let engine = setup_engine().await;
        let config = friend_config();

        let result = engine
            .cross_modal_contraction(
                "alice",
                "purchases",
                "no_such_column",
                "item",
                &config,
                None,
            )
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, UnifiedError::InvalidOperation(_)));
    }

    #[tokio::test]
    async fn adapter_missing_target_column_error() {
        let engine = setup_engine().await;
        let config = friend_config();

        let result = engine
            .cross_modal_contraction(
                "alice",
                "purchases",
                "buyer",
                "no_such_column",
                &config,
                None,
            )
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, UnifiedError::InvalidOperation(_)));
    }

    #[tokio::test]
    async fn adapter_empty_table() {
        let engine = UnifiedEngine::new();
        engine
            .create_entity("alice", HashMap::new(), None)
            .await
            .unwrap();
        engine
            .create_entity("bob", HashMap::new(), None)
            .await
            .unwrap();
        engine
            .connect_entities("alice", "bob", "FRIEND")
            .await
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("alice", vec![1.0, 0.0])
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("bob", vec![0.9, 0.1])
            .unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "buyer".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "item".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine
            .relational()
            .create_table("purchases", schema)
            .unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("alice", "purchases", "buyer", "item", &config, None)
            .await
            .unwrap();

        assert!(result.items.is_empty());
    }

    #[tokio::test]
    async fn adapter_large_fan_out() {
        let engine = UnifiedEngine::new();
        engine
            .create_entity("center", HashMap::new(), None)
            .await
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("center", vec![1.0, 0.0, 0.0])
            .unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("actions", schema).unwrap();

        // Create 50 friends with embeddings and purchases
        for i in 0..50 {
            let key = format!("friend{i}");
            engine
                .create_entity(&key, HashMap::new(), None)
                .await
                .unwrap();
            engine
                .connect_entities("center", &key, "FRIEND")
                .await
                .unwrap();
            engine
                .vector()
                .set_entity_embedding(&key, vec![0.9, 0.1, 0.0])
                .unwrap();

            // Each friend bought 3 unique items
            for j in 0..3 {
                engine
                    .relational()
                    .insert(
                        "actions",
                        HashMap::from([
                            ("src".into(), Value::String(key.clone())),
                            ("tgt".into(), Value::String(format!("item_{i}_{j}"))),
                        ]),
                    )
                    .unwrap();
            }
        }

        let mut config = friend_config();
        config.top_k = 5;

        let result = engine
            .cross_modal_contraction("center", "actions", "src", "tgt", &config, None)
            .await
            .unwrap();

        assert_eq!(result.items.len(), 5);
    }

    #[tokio::test]
    async fn adapter_no_graph_node_for_source() {
        let engine = UnifiedEngine::new();
        // Don't create entity in graph, just have a table
        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("no_node", "t", "src", "tgt", &config, None)
            .await
            .unwrap();

        assert!(result.items.is_empty());
    }

    #[tokio::test]
    async fn adapter_mixed_some_have_embeddings() {
        let engine = UnifiedEngine::new();
        for key in &["a", "b", "c"] {
            engine
                .create_entity(key, HashMap::new(), None)
                .await
                .unwrap();
        }
        engine.connect_entities("a", "b", "FRIEND").await.unwrap();
        engine.connect_entities("a", "c", "FRIEND").await.unwrap();

        // Only a and b have embeddings; c does not
        engine
            .vector()
            .set_entity_embedding("a", vec![1.0, 0.0])
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("b", vec![0.9, 0.1])
            .unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();
        engine
            .relational()
            .insert(
                "t",
                HashMap::from([
                    ("src".into(), Value::String("b".into())),
                    ("tgt".into(), Value::String("item_b".into())),
                ]),
            )
            .unwrap();
        engine
            .relational()
            .insert(
                "t",
                HashMap::from([
                    ("src".into(), Value::String("c".into())),
                    ("tgt".into(), Value::String("item_c".into())),
                ]),
            )
            .unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("a", "t", "src", "tgt", &config, None)
            .await
            .unwrap();

        // Only b has embedding, so only item_b should appear
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.items[0].item_key, "item_b");
    }

    #[tokio::test]
    async fn adapter_column_type_validation_bool_rejected() {
        let engine = UnifiedEngine::new();
        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "flag".into(),
                    column_type: ColumnType::Bool,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "item".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("x", "t", "flag", "item", &config, None)
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            UnifiedError::InvalidOperation(_)
        ));
    }

    #[tokio::test]
    async fn adapter_column_type_validation_float_rejected() {
        let engine = UnifiedEngine::new();
        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "score".into(),
                    column_type: ColumnType::Float,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "item".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("x", "t", "score", "item", &config, None)
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            UnifiedError::InvalidOperation(_)
        ));
    }

    #[tokio::test]
    async fn adapter_multi_edge_weights_summed() {
        let engine = UnifiedEngine::new();
        for key in &["a", "b"] {
            engine
                .create_entity(key, HashMap::new(), None)
                .await
                .unwrap();
        }
        // Create two FRIEND edges between a and b
        engine.connect_entities("a", "b", "FRIEND").await.unwrap();
        let a_node = engine.find_entity_node("a").unwrap();
        let b_node = engine.find_entity_node("b").unwrap();
        engine
            .graph()
            .create_edge(
                a_node,
                b_node,
                "FRIEND",
                HashMap::from([("weight".into(), PropertyValue::Float(2.0))]),
                true,
            )
            .unwrap();

        // Identical embeddings -> similarity = 1.0
        engine
            .vector()
            .set_entity_embedding("a", vec![1.0, 0.0])
            .unwrap();
        engine
            .vector()
            .set_entity_embedding("b", vec![1.0, 0.0])
            .unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();
        engine
            .relational()
            .insert(
                "t",
                HashMap::from([
                    ("src".into(), Value::String("b".into())),
                    ("tgt".into(), Value::String("item1".into())),
                ]),
            )
            .unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("a", "t", "src", "tgt", &config, None)
            .await
            .unwrap();

        // First edge weight=1.0, second edge weight=2.0 -> summed adj=3.0
        // similarity=1.0, score = 3.0 * 1.0 = 3.0
        assert_eq!(result.items.len(), 1);
        assert!((result.items[0].score - 3.0).abs() < TOL);
    }

    #[tokio::test]
    async fn adapter_early_return_empty_adjacency_skips_scan() {
        let engine = UnifiedEngine::new();
        // Entity with no graph edges
        engine
            .create_entity("lonely", HashMap::new(), None)
            .await
            .unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("lonely", "t", "src", "tgt", &config, None)
            .await
            .unwrap();

        assert!(result.items.is_empty());
        assert!(result.weight_norm.abs() < TOL);
    }

    #[tokio::test]
    async fn adapter_early_return_empty_similarity_skips_scan() {
        let engine = UnifiedEngine::new();
        // Two entities with edges but no embeddings
        for key in &["a", "b"] {
            engine
                .create_entity(key, HashMap::new(), None)
                .await
                .unwrap();
        }
        engine.connect_entities("a", "b", "FRIEND").await.unwrap();

        let schema = relational_engine::Schema {
            columns: vec![
                relational_engine::Column {
                    name: "src".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
                relational_engine::Column {
                    name: "tgt".into(),
                    column_type: ColumnType::String,
                    nullable: false,
                },
            ],
            constraints: vec![],
        };
        engine.relational().create_table("t", schema).unwrap();

        let config = friend_config();
        let result = engine
            .cross_modal_contraction("a", "t", "src", "tgt", &config, None)
            .await
            .unwrap();

        assert!(result.items.is_empty());
        assert!(result.weight_norm.abs() < TOL);
    }

    #[tokio::test]
    async fn adapter_self_loop_excluded() {
        let engine = setup_engine().await;

        // Add a self-loop on alice
        engine
            .connect_entities("alice", "alice", "FRIEND")
            .await
            .unwrap();

        // alice bought "book" — if the self-loop leaks through, "book" would
        // get an extra contribution from alice's own purchases.
        let config = friend_config();
        let result = engine
            .cross_modal_contraction("alice", "purchases", "buyer", "item", &config, None)
            .await
            .unwrap();

        // alice should NOT appear in the adjacency vec, so her purchases
        // shouldn't contribute to scoring via the self-loop path.
        // Without the fix, "book" would have 3 contributors (bob + alice-self-loop + alice-owned)
        // instead of 1 (bob only).
        for item in &result.items {
            if item.item_key == "book" {
                assert_eq!(
                    item.contributors, 1,
                    "book should have 1 contributor (bob), not extra from self-loop"
                );
            }
        }
    }

    #[tokio::test]
    async fn adapter_top_k_zero_returns_empty() {
        let engine = setup_engine().await;

        let mut config = friend_config();
        config.top_k = 0;

        let result = engine
            .cross_modal_contraction("alice", "purchases", "buyer", "item", &config, None)
            .await
            .unwrap();

        assert!(result.items.is_empty());
        assert!(result.weight_norm.abs() < TOL);
        assert_eq!(result.excluded_count, 0);
    }

    #[tokio::test]
    async fn adapter_top_k_zero_skips_validation() {
        // top_k == 0 should return immediately without even checking the table.
        let engine = UnifiedEngine::new();
        let mut config = friend_config();
        config.top_k = 0;

        // No table exists, but we should still get Ok because of early return.
        let result = engine
            .cross_modal_contraction("nobody", "nonexistent", "a", "b", &config, None)
            .await
            .unwrap();

        assert!(result.items.is_empty());
    }
}
