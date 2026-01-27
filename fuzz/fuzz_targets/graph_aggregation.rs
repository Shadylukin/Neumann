#![no_main]

//! Fuzz target for graph aggregation operations.
//!
//! Tests aggregate_node_property, aggregate_edge_property,
//! and their variants with labels/types and conditions.

use arbitrary::Arbitrary;
use graph_engine::{GraphEngine, PropertyValue, RangeOp};
use libfuzzer_sys::fuzz_target;
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzPropertyValue {
    Null,
    Int(i64),
    Float(f64),
    String(String),
}

impl FuzzPropertyValue {
    fn to_property_value(&self) -> PropertyValue {
        match self {
            Self::Null => PropertyValue::Null,
            Self::Int(i) => PropertyValue::Int(*i),
            Self::Float(f) => {
                let f = if f.is_nan() || f.is_infinite() { 0.0 } else { *f };
                PropertyValue::Float(f)
            }
            Self::String(s) => PropertyValue::String(s.chars().take(50).collect()),
        }
    }

    fn to_numeric(&self) -> PropertyValue {
        match self {
            Self::Int(i) => PropertyValue::Int(*i),
            Self::Float(f) => {
                let f = if f.is_nan() || f.is_infinite() { 0.0 } else { *f };
                PropertyValue::Float(f)
            }
            _ => PropertyValue::Int(0),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzRangeOp {
    op: u8,
}

impl FuzzRangeOp {
    fn to_range_op(&self) -> RangeOp {
        match self.op % 4 {
            0 => RangeOp::Lt,
            1 => RangeOp::Le,
            2 => RangeOp::Gt,
            _ => RangeOp::Ge,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum AggregateOp {
    AggregateNodeProperty { property: String },
    AggregateNodePropertyByLabel { label: String, property: String },
    AggregateNodePropertyWhere { filter_prop: String, op: FuzzRangeOp, value: FuzzPropertyValue, agg_prop: String },
    AggregateEdgeProperty { property: String },
    AggregateEdgePropertyByType { edge_type: String, property: String },
    AggregateEdgePropertyWhere { filter_prop: String, op: FuzzRangeOp, value: FuzzPropertyValue, agg_prop: String },
    SumNodeProperty { property: String },
    AvgNodeProperty { property: String },
    SumEdgeProperty { property: String },
    AvgEdgeProperty { property: String },
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    nodes: Vec<(String, Vec<(String, FuzzPropertyValue)>)>,
    edges: Vec<(usize, usize, String, Vec<(String, FuzzPropertyValue)>)>,
    ops: Vec<AggregateOp>,
}

fn sanitize_name(s: &str) -> String {
    let sanitized: String = s
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(32)
        .collect();
    if sanitized.is_empty() {
        "name".to_string()
    } else {
        sanitized
    }
}

fuzz_target!(|input: FuzzInput| {
    let engine = GraphEngine::new();
    let mut node_ids: Vec<u64> = Vec::new();

    // Create nodes with properties
    for (label, props) in input.nodes.into_iter().take(50) {
        let label = sanitize_name(&label);
        let mut properties = HashMap::new();
        for (key, value) in props.into_iter().take(10) {
            let key = sanitize_name(&key);
            properties.insert(key, value.to_property_value());
        }
        if let Ok(id) = engine.create_node(&label, properties) {
            node_ids.push(id);
        }
    }

    // Create edges with properties
    for (from_idx, to_idx, edge_type, props) in input.edges.into_iter().take(50) {
        if node_ids.len() < 2 {
            break;
        }
        let from = node_ids[from_idx % node_ids.len()];
        let to = node_ids[to_idx % node_ids.len()];
        let edge_type = sanitize_name(&edge_type);
        let mut properties = HashMap::new();
        for (key, value) in props.into_iter().take(10) {
            let key = sanitize_name(&key);
            properties.insert(key, value.to_property_value());
        }
        let _ = engine.create_edge(from, to, &edge_type, properties, true);
    }

    // Execute aggregation operations
    for op in input.ops.into_iter().take(50) {
        match op {
            AggregateOp::AggregateNodeProperty { property } => {
                let prop = sanitize_name(&property);
                let result = engine.aggregate_node_property(&prop);
                // Verify result invariants
                assert!(result.count <= node_ids.len() as u64);
                if let Some(sum) = result.sum {
                    assert!(!sum.is_nan(), "Sum should not be NaN");
                }
                if let Some(avg) = result.avg {
                    assert!(!avg.is_nan(), "Average should not be NaN");
                }
            }
            AggregateOp::AggregateNodePropertyByLabel { label, property } => {
                let label = sanitize_name(&label);
                let prop = sanitize_name(&property);
                let result = engine.aggregate_node_property_by_label(&label, &prop);
                if let Some(sum) = result.sum {
                    assert!(!sum.is_nan(), "Sum should not be NaN");
                }
            }
            AggregateOp::AggregateNodePropertyWhere { filter_prop, op, value, agg_prop } => {
                let filter = sanitize_name(&filter_prop);
                let agg = sanitize_name(&agg_prop);
                let range_op = op.to_range_op();
                let pv = value.to_numeric();
                if let Ok(result) = engine.aggregate_node_property_where(&filter, range_op, &pv, &agg) {
                    if let Some(sum) = result.sum {
                        assert!(!sum.is_nan(), "Sum should not be NaN");
                    }
                }
            }
            AggregateOp::AggregateEdgeProperty { property } => {
                let prop = sanitize_name(&property);
                let result = engine.aggregate_edge_property(&prop);
                if let Some(sum) = result.sum {
                    assert!(!sum.is_nan(), "Sum should not be NaN");
                }
            }
            AggregateOp::AggregateEdgePropertyByType { edge_type, property } => {
                let edge_type = sanitize_name(&edge_type);
                let prop = sanitize_name(&property);
                if let Ok(result) = engine.aggregate_edge_property_by_type(&edge_type, &prop) {
                    if let Some(sum) = result.sum {
                        assert!(!sum.is_nan(), "Sum should not be NaN");
                    }
                }
            }
            AggregateOp::AggregateEdgePropertyWhere { filter_prop, op, value, agg_prop } => {
                let filter = sanitize_name(&filter_prop);
                let agg = sanitize_name(&agg_prop);
                let range_op = op.to_range_op();
                let pv = value.to_numeric();
                if let Ok(result) = engine.aggregate_edge_property_where(&filter, range_op, &pv, &agg) {
                    if let Some(sum) = result.sum {
                        assert!(!sum.is_nan(), "Sum should not be NaN");
                    }
                }
            }
            AggregateOp::SumNodeProperty { property } => {
                let prop = sanitize_name(&property);
                if let Some(sum) = engine.sum_node_property(&prop) {
                    assert!(!sum.is_nan(), "Sum should not be NaN");
                }
            }
            AggregateOp::AvgNodeProperty { property } => {
                let prop = sanitize_name(&property);
                if let Some(avg) = engine.avg_node_property(&prop) {
                    assert!(!avg.is_nan(), "Average should not be NaN");
                }
            }
            AggregateOp::SumEdgeProperty { property } => {
                let prop = sanitize_name(&property);
                if let Some(sum) = engine.sum_edge_property(&prop) {
                    assert!(!sum.is_nan(), "Sum should not be NaN");
                }
            }
            AggregateOp::AvgEdgeProperty { property } => {
                let prop = sanitize_name(&property);
                if let Some(avg) = engine.avg_edge_property(&prop) {
                    assert!(!avg.is_nan(), "Average should not be NaN");
                }
            }
        }
    }
});
