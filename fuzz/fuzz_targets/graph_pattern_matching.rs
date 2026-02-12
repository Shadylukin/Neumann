// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

//! Fuzz target for graph pattern matching operations.
//!
//! Tests match_pattern with various node patterns, edge patterns,
//! and path patterns.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use graph_engine::{
    CompareOp, Direction, EdgePattern, GraphEngine, NodePattern, PathPattern, Pattern,
    PatternElement, PropertyValue,
};
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzPropertyValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl FuzzPropertyValue {
    fn to_property_value(&self) -> PropertyValue {
        match self {
            Self::Bool(b) => PropertyValue::Bool(*b),
            Self::Int(i) => PropertyValue::Int(*i),
            Self::Float(f) => {
                let f = if f.is_nan() || f.is_infinite() { 0.0 } else { *f };
                PropertyValue::Float(f)
            }
            Self::String(s) => PropertyValue::String(s.chars().take(32).collect()),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzCondition {
    property: String,
    op: u8,
    value: FuzzPropertyValue,
}

#[derive(Arbitrary, Debug)]
struct FuzzNodePattern {
    variable: Option<String>,
    label: Option<String>,
    conditions: Vec<FuzzCondition>,
}

#[derive(Arbitrary, Debug)]
struct FuzzEdgePattern {
    variable: Option<String>,
    edge_type: Option<String>,
    direction: u8,
    min_length: Option<u8>,
    max_length: Option<u8>,
}

#[derive(Arbitrary, Debug)]
enum FuzzPatternElement {
    Node(FuzzNodePattern),
    Edge(FuzzEdgePattern),
}

#[derive(Arbitrary, Debug)]
struct FuzzPattern {
    elements: Vec<FuzzPatternElement>,
    limit: Option<u16>,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    node_count: u8,
    labels: Vec<String>,
    edge_types: Vec<String>,
    patterns: Vec<FuzzPattern>,
}

fn sanitize_name(s: &str) -> String {
    let sanitized: String = s
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(16)
        .collect();
    if sanitized.is_empty() {
        "name".to_string()
    } else {
        sanitized
    }
}

fn compare_op_from_u8(op: u8) -> CompareOp {
    match op % 6 {
        0 => CompareOp::Eq,
        1 => CompareOp::Ne,
        2 => CompareOp::Lt,
        3 => CompareOp::Le,
        4 => CompareOp::Gt,
        _ => CompareOp::Ge,
    }
}

fn direction_from_u8(d: u8) -> Direction {
    match d % 3 {
        0 => Direction::Outgoing,
        1 => Direction::Incoming,
        _ => Direction::Both,
    }
}

fn build_graph(
    engine: &GraphEngine,
    node_count: u8,
    labels: &[String],
    edge_types: &[String],
) -> Vec<u64> {
    let n = (node_count as usize).clamp(5, 30);
    let labels: Vec<String> = if labels.is_empty() {
        vec!["Person".to_string(), "Company".to_string(), "Product".to_string()]
    } else {
        labels.iter().take(5).map(|s| sanitize_name(s)).collect()
    };

    let edge_types: Vec<String> = if edge_types.is_empty() {
        vec!["KNOWS".to_string(), "WORKS_AT".to_string(), "OWNS".to_string()]
    } else {
        edge_types.iter().take(5).map(|s| sanitize_name(s)).collect()
    };

    let mut node_ids = Vec::with_capacity(n);
    for i in 0..n {
        let label = &labels[i % labels.len()];
        let id = engine
            .create_node(
                label,
                HashMap::from([
                    ("idx".to_string(), PropertyValue::Int(i as i64)),
                    ("name".to_string(), PropertyValue::String(format!("Node{i}"))),
                    ("active".to_string(), PropertyValue::Bool(i % 2 == 0)),
                ]),
            )
            .unwrap();
        node_ids.push(id);
    }

    // Create edges with various types
    for i in 0..n * 2 {
        let from_idx = i % n;
        let to_idx = (i * 3 + 1) % n;
        if from_idx != to_idx {
            let edge_type = &edge_types[i % edge_types.len()];
            let _ = engine.create_edge(
                node_ids[from_idx],
                node_ids[to_idx],
                edge_type,
                HashMap::from([
                    ("weight".to_string(), PropertyValue::Float((i as f64) * 0.1)),
                ]),
                true,
            );
        }
    }

    node_ids
}

fn build_node_pattern(fuzz: &FuzzNodePattern) -> NodePattern {
    let mut pattern = NodePattern::new();

    if let Some(ref var) = fuzz.variable {
        let var = sanitize_name(var);
        if !var.is_empty() {
            pattern = pattern.variable(&var);
        }
    }

    if let Some(ref label) = fuzz.label {
        let label = sanitize_name(label);
        if !label.is_empty() {
            pattern = pattern.label(&label);
        }
    }

    for cond in fuzz.conditions.iter().take(3) {
        let prop = sanitize_name(&cond.property);
        if !prop.is_empty() {
            let op = compare_op_from_u8(cond.op);
            pattern = pattern.where_cond(&prop, op, cond.value.to_property_value());
        }
    }

    pattern
}

fn build_edge_pattern(fuzz: &FuzzEdgePattern) -> EdgePattern {
    let mut pattern = EdgePattern::new();

    if let Some(ref var) = fuzz.variable {
        let var = sanitize_name(var);
        if !var.is_empty() {
            pattern = pattern.variable(&var);
        }
    }

    if let Some(ref edge_type) = fuzz.edge_type {
        let edge_type = sanitize_name(edge_type);
        if !edge_type.is_empty() {
            pattern = pattern.edge_type(&edge_type);
        }
    }

    pattern = pattern.direction(direction_from_u8(fuzz.direction));

    if let (Some(min), Some(max)) = (fuzz.min_length, fuzz.max_length) {
        let min = (min as usize).clamp(1, 5);
        let max = (max as usize).clamp(min, 10);
        pattern = pattern.variable_length(min, max);
    }

    pattern
}

fn build_pattern(fuzz: &FuzzPattern) -> Pattern {
    let mut elements = Vec::new();

    for elem in fuzz.elements.iter().take(5) {
        match elem {
            FuzzPatternElement::Node(np) => {
                elements.push(PatternElement::Node(build_node_pattern(np)));
            }
            FuzzPatternElement::Edge(ep) => {
                elements.push(PatternElement::Edge(build_edge_pattern(ep)));
            }
        }
    }

    // Ensure pattern starts with a node
    if elements.is_empty() || !matches!(elements[0], PatternElement::Node(_)) {
        elements.insert(0, PatternElement::Node(NodePattern::new()));
    }

    let limit = fuzz.limit.map(|l| (l as usize).clamp(1, 100));

    let path = PathPattern { elements };
    Pattern { path, limit }
}

fuzz_target!(|input: FuzzInput| {
    let engine = GraphEngine::new();
    let _node_ids = build_graph(&engine, input.node_count, &input.labels, &input.edge_types);

    for fuzz_pattern in input.patterns.into_iter().take(20) {
        let pattern = build_pattern(&fuzz_pattern);

        // Skip empty patterns
        if pattern.path.elements.is_empty() {
            continue;
        }

        let result = engine.match_pattern(&pattern);
        if let Ok(match_result) = result {
            // Verify match result consistency
            assert!(
                match_result.stats.matches_found <= match_result.stats.nodes_evaluated,
                "Matches found should not exceed nodes evaluated"
            );

            // If limit was set, verify it's respected
            if let Some(limit) = pattern.limit {
                assert!(
                    match_result.matches.len() <= limit,
                    "Result count {} should not exceed limit {}",
                    match_result.matches.len(),
                    limit
                );
            }
        }
    }
});
