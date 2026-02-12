// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use relational_engine::{Column, ColumnType, Condition, RelationalEngine, Schema, Value};
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl FuzzValue {
    fn to_value(&self) -> Value {
        match self {
            FuzzValue::Int(i) => Value::Int(*i),
            FuzzValue::Float(f) => Value::Float(*f),
            FuzzValue::Bool(b) => Value::Bool(*b),
        }
    }

    fn column_type(&self) -> ColumnType {
        match self {
            FuzzValue::Int(_) => ColumnType::Int,
            FuzzValue::Float(_) => ColumnType::Float,
            FuzzValue::Bool(_) => ColumnType::Bool,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum FuzzCondition {
    Eq(FuzzValue),
    Ne(FuzzValue),
    Lt(FuzzValue),
    Le(FuzzValue),
    Gt(FuzzValue),
    Ge(FuzzValue),
    And(Box<FuzzCondition>, Box<FuzzCondition>),
    Or(Box<FuzzCondition>, Box<FuzzCondition>),
}

impl FuzzCondition {
    fn to_condition(&self, col_name: &str, depth: usize) -> Option<Condition> {
        // Limit nesting depth to avoid stack overflow
        if depth > 8 {
            return None;
        }
        Some(match self {
            FuzzCondition::Eq(v) => Condition::Eq(col_name.to_string(), v.to_value()),
            FuzzCondition::Ne(v) => Condition::Ne(col_name.to_string(), v.to_value()),
            FuzzCondition::Lt(v) => Condition::Lt(col_name.to_string(), v.to_value()),
            FuzzCondition::Le(v) => Condition::Le(col_name.to_string(), v.to_value()),
            FuzzCondition::Gt(v) => Condition::Gt(col_name.to_string(), v.to_value()),
            FuzzCondition::Ge(v) => Condition::Ge(col_name.to_string(), v.to_value()),
            FuzzCondition::And(a, b) => {
                let a_cond = a.to_condition(col_name, depth + 1)?;
                let b_cond = b.to_condition(col_name, depth + 1)?;
                a_cond.and(b_cond)
            },
            FuzzCondition::Or(a, b) => {
                let a_cond = a.to_condition(col_name, depth + 1)?;
                let b_cond = b.to_condition(col_name, depth + 1)?;
                a_cond.or(b_cond)
            },
        })
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    col_type: FuzzValue,
    values: Vec<FuzzValue>,
    condition: FuzzCondition,
}

fuzz_target!(|input: FuzzInput| {
    // Limit row count to avoid OOM
    let values: Vec<_> = input.values.into_iter().take(100).collect();
    if values.is_empty() {
        return;
    }

    let engine = RelationalEngine::new();
    let col_type = input.col_type.column_type();

    // Create table with column matching the type
    let schema = Schema::new(vec![Column::new("val", col_type)]);
    if engine.create_table("test", schema).is_err() {
        return;
    }

    // Insert rows with type-compatible values
    for val in &values {
        if val.column_type() == input.col_type.column_type() {
            let mut row = HashMap::new();
            row.insert("val".to_string(), val.to_value());
            let _ = engine.insert("test", row);
        }
    }

    // Build and execute condition
    let Some(condition) = input.condition.to_condition("val", 0) else {
        return;
    };

    // Execute query - should not panic
    let result = engine.select("test", condition.clone());
    assert!(result.is_ok(), "select should not fail: {:?}", result.err());

    // Verify results are consistent with Condition::True
    let all_rows = engine.select("test", Condition::True);
    if let (Ok(filtered), Ok(all)) = (result, all_rows) {
        // Filtered count should not exceed total
        assert!(
            filtered.len() <= all.len(),
            "Filtered rows exceed total rows"
        );
    }
});
