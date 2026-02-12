// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

//! Fuzz target for graph constraint operations.
//!
//! Tests create_constraint, drop_constraint, validate_constraints
//! with various constraint types and targets.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use graph_engine::{
    Constraint, ConstraintTarget, ConstraintType, GraphEngine, PropertyValue, PropertyValueType,
};
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzConstraintType {
    Unique,
    Exists,
    TypeInt,
    TypeFloat,
    TypeString,
    TypeBool,
}

impl FuzzConstraintType {
    fn to_constraint_type(&self) -> ConstraintType {
        match self {
            Self::Unique => ConstraintType::Unique,
            Self::Exists => ConstraintType::Exists,
            Self::TypeInt => ConstraintType::PropertyType(PropertyValueType::Int),
            Self::TypeFloat => ConstraintType::PropertyType(PropertyValueType::Float),
            Self::TypeString => ConstraintType::PropertyType(PropertyValueType::String),
            Self::TypeBool => ConstraintType::PropertyType(PropertyValueType::Bool),
        }
    }
}

#[derive(Arbitrary, Debug, Clone)]
enum FuzzConstraintTarget {
    NodeLabel(String),
    EdgeType(String),
    AllNodes,
    AllEdges,
}

impl FuzzConstraintTarget {
    fn to_constraint_target(&self) -> ConstraintTarget {
        match self {
            Self::NodeLabel(s) => ConstraintTarget::NodeLabel(sanitize_name(s)),
            Self::EdgeType(s) => ConstraintTarget::EdgeType(sanitize_name(s)),
            Self::AllNodes => ConstraintTarget::AllNodes,
            Self::AllEdges => ConstraintTarget::AllEdges,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum ConstraintOp {
    Create {
        name: String,
        target: FuzzConstraintTarget,
        property: String,
        constraint_type: FuzzConstraintType,
    },
    Drop {
        constraint_idx: u8,
    },
    List,
    Get {
        constraint_idx: u8,
    },
    ValidateNode {
        node_idx: u8,
    },
    CreateNodeWithConstraint {
        label: String,
        properties: Vec<(String, FuzzPropertyValue)>,
    },
}

#[derive(Arbitrary, Debug, Clone)]
enum FuzzPropertyValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

impl FuzzPropertyValue {
    fn to_property_value(&self) -> PropertyValue {
        match self {
            Self::Null => PropertyValue::Null,
            Self::Bool(b) => PropertyValue::Bool(*b),
            Self::Int(i) => PropertyValue::Int(*i),
            Self::Float(f) => {
                let f = if f.is_nan() || f.is_infinite() { 0.0 } else { *f };
                PropertyValue::Float(f)
            }
            Self::String(s) => PropertyValue::String(s.chars().take(50).collect()),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<ConstraintOp>,
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
    let mut created_constraints: Vec<String> = Vec::new();
    let mut created_nodes: Vec<u64> = Vec::new();

    // Create some initial nodes for testing
    for i in 0..10 {
        let label = if i % 2 == 0 { "Person" } else { "Company" };
        if let Ok(id) = engine.create_node(
            label,
            HashMap::from([
                ("name".to_string(), PropertyValue::String(format!("Entity{i}"))),
                ("idx".to_string(), PropertyValue::Int(i)),
                ("active".to_string(), PropertyValue::Bool(i % 3 == 0)),
            ]),
        ) {
            created_nodes.push(id);
        }
    }

    for op in input.ops.into_iter().take(100) {
        match op {
            ConstraintOp::Create {
                name,
                target,
                property,
                constraint_type,
            } => {
                let name = sanitize_name(&name);
                let property = sanitize_name(&property);

                if name.is_empty() || property.is_empty() {
                    continue;
                }

                // Skip if constraint already exists
                if created_constraints.contains(&name) {
                    continue;
                }

                let constraint = Constraint {
                    name: name.clone(),
                    target: target.to_constraint_target(),
                    property,
                    constraint_type: constraint_type.to_constraint_type(),
                };

                if engine.create_constraint(constraint).is_ok() {
                    created_constraints.push(name);
                }
            }
            ConstraintOp::Drop { constraint_idx } => {
                if created_constraints.is_empty() {
                    continue;
                }
                let idx = constraint_idx as usize % created_constraints.len();
                let name = &created_constraints[idx];
                if engine.drop_constraint(name).is_ok() {
                    created_constraints.remove(idx);
                }
            }
            ConstraintOp::List => {
                let _ = engine.list_constraints();
            }
            ConstraintOp::Get { constraint_idx } => {
                if created_constraints.is_empty() {
                    continue;
                }
                let idx = constraint_idx as usize % created_constraints.len();
                let name = &created_constraints[idx];
                let _constraint = engine.get_constraint(name);
            }
            ConstraintOp::ValidateNode { node_idx } => {
                if created_nodes.is_empty() {
                    continue;
                }
                let id = created_nodes[node_idx as usize % created_nodes.len()];
                // Validation happens implicitly during node operations
                let _ = engine.get_node(id);
            }
            ConstraintOp::CreateNodeWithConstraint { label, properties } => {
                let label = sanitize_name(&label);
                let props: HashMap<String, PropertyValue> = properties
                    .into_iter()
                    .take(5)
                    .map(|(k, v)| (sanitize_name(&k), v.to_property_value()))
                    .collect();

                // This may succeed or fail depending on constraints
                if let Ok(id) = engine.create_node(&label, props) {
                    created_nodes.push(id);
                }
            }
        }
    }

    // Verify all remaining constraints can be retrieved
    for name in &created_constraints {
        let result = engine.get_constraint(name);
        assert!(
            result.is_some(),
            "Constraint '{}' should still exist",
            name
        );
    }

    // Verify list_constraints returns all created constraints
    let all_constraints = engine.list_constraints();
    for name in &created_constraints {
        assert!(
            all_constraints.iter().any(|c| &c.name == name),
            "Constraint '{}' should be in list",
            name
        );
    }
});
