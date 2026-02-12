// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0

/// Graph aggregate result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphAggregateResult {
    pub operation: String,
    pub value: GraphAggregateValue,
}

/// Graph aggregate value types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphAggregateValue {
    Count(u64),
    Float(f64),
}

/// Batch operation result for graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphBatchResult {
    pub operation: String,
    pub affected_count: usize,
    pub created_ids: Option<Vec<u64>>,
}

/// Constraint info for display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintInfoResult {
    pub name: String,
    pub target: String,
    pub property: String,
    pub constraint_type: String,
}
