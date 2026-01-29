// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use relational_engine::{Column, ColumnType, Condition, RelationalEngine, Schema, Value};
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzColumnType {
    Int,
    Float,
    String,
    Bool,
}

impl FuzzColumnType {
    fn to_column_type(&self) -> ColumnType {
        match self {
            FuzzColumnType::Int => ColumnType::Int,
            FuzzColumnType::Float => ColumnType::Float,
            FuzzColumnType::String => ColumnType::String,
            FuzzColumnType::Bool => ColumnType::Bool,
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzColumn {
    name: String,
    col_type: FuzzColumnType,
    nullable: bool,
}

#[derive(Arbitrary, Debug, Clone)]
enum FuzzValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Null,
}

impl FuzzValue {
    fn to_value(&self) -> Value {
        match self {
            FuzzValue::Int(i) => Value::Int(*i),
            FuzzValue::Float(f) => Value::Float(*f),
            FuzzValue::String(s) => Value::String(s.clone()),
            FuzzValue::Bool(b) => Value::Bool(*b),
            FuzzValue::Null => Value::Null,
        }
    }

    fn compatible_with(&self, col_type: &FuzzColumnType) -> bool {
        match (self, col_type) {
            (FuzzValue::Int(_), FuzzColumnType::Int) => true,
            (FuzzValue::Float(_), FuzzColumnType::Float) => true,
            (FuzzValue::String(_), FuzzColumnType::String) => true,
            (FuzzValue::Bool(_), FuzzColumnType::Bool) => true,
            (FuzzValue::Null, _) => true,
            _ => false,
        }
    }
}

#[derive(Arbitrary, Debug)]
enum FuzzCondition {
    Eq(usize, FuzzValue),
    Ne(usize, FuzzValue),
    Lt(usize, FuzzValue),
    Le(usize, FuzzValue),
    Gt(usize, FuzzValue),
    Ge(usize, FuzzValue),
    True,
}

#[derive(Arbitrary, Debug)]
enum FuzzOp {
    CreateTable {
        table_name: String,
        columns: Vec<FuzzColumn>,
    },
    Insert {
        table_idx: u8,
        values: Vec<(String, FuzzValue)>,
    },
    Select {
        table_idx: u8,
        condition: FuzzCondition,
    },
    Update {
        table_idx: u8,
        updates: Vec<(String, FuzzValue)>,
        condition: FuzzCondition,
    },
    Delete {
        table_idx: u8,
        condition: FuzzCondition,
    },
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<FuzzOp>,
}

fn sanitize_name(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(32)
        .collect()
}

fuzz_target!(|input: FuzzInput| {
    let engine = RelationalEngine::new();
    let mut tables: Vec<(String, Vec<FuzzColumn>)> = Vec::new();

    for op in input.ops.into_iter().take(100) {
        match op {
            FuzzOp::CreateTable {
                table_name,
                columns,
            } => {
                let name = sanitize_name(&table_name);
                if name.is_empty() || columns.is_empty() {
                    continue;
                }

                let cols: Vec<_> = columns
                    .into_iter()
                    .take(10)
                    .filter(|c| {
                        let cname = sanitize_name(&c.name);
                        !cname.is_empty()
                    })
                    .collect();

                if cols.is_empty() {
                    continue;
                }

                let schema_cols: Vec<Column> = cols
                    .iter()
                    .map(|c| Column::new(sanitize_name(&c.name), c.col_type.to_column_type()))
                    .collect();

                let schema = Schema::new(schema_cols);

                if engine.create_table(&name, schema).is_ok() {
                    tables.push((name, cols));
                }
            },
            FuzzOp::Insert { table_idx, values } => {
                if tables.is_empty() {
                    continue;
                }
                let idx = table_idx as usize % tables.len();
                let (table_name, cols) = &tables[idx];

                // Build a HashMap of compatible values
                let mut row_values: HashMap<String, Value> = HashMap::new();
                for (name, val) in values.into_iter().take(cols.len()) {
                    let col_name = sanitize_name(&name);
                    // Find matching column
                    if let Some(col) = cols.iter().find(|c| sanitize_name(&c.name) == col_name) {
                        if val.compatible_with(&col.col_type) {
                            row_values.insert(col_name, val.to_value());
                        }
                    }
                }

                // If no valid values, add at least one NULL for nullable columns
                if row_values.is_empty() {
                    if let Some(col) = cols.iter().find(|c| c.nullable) {
                        row_values.insert(sanitize_name(&col.name), Value::Null);
                    }
                }

                let _ = engine.insert(table_name, row_values);
            },
            FuzzOp::Select {
                table_idx,
                condition,
            } => {
                if tables.is_empty() {
                    continue;
                }
                let idx = table_idx as usize % tables.len();
                let (table_name, cols) = &tables[idx];

                let cond = build_condition(&condition, cols);
                let _ = engine.select(table_name, cond);
            },
            FuzzOp::Update {
                table_idx,
                updates,
                condition,
            } => {
                if tables.is_empty() {
                    continue;
                }
                let idx = table_idx as usize % tables.len();
                let (table_name, cols) = &tables[idx];

                let cond = build_condition(&condition, cols);

                // Build updates HashMap
                let mut update_values: HashMap<String, Value> = HashMap::new();
                for (name, val) in updates.into_iter().take(cols.len()) {
                    let col_name = sanitize_name(&name);
                    if let Some(col) = cols.iter().find(|c| sanitize_name(&c.name) == col_name) {
                        if val.compatible_with(&col.col_type) {
                            update_values.insert(col_name, val.to_value());
                        }
                    }
                }

                if !update_values.is_empty() {
                    let _ = engine.update(table_name, cond, update_values);
                }
            },
            FuzzOp::Delete {
                table_idx,
                condition,
            } => {
                if tables.is_empty() {
                    continue;
                }
                let idx = table_idx as usize % tables.len();
                let (table_name, cols) = &tables[idx];

                let cond = build_condition(&condition, cols);
                let _ = engine.delete_rows(table_name, cond);
            },
        }
    }
});

fn build_condition(fuzz_cond: &FuzzCondition, cols: &[FuzzColumn]) -> Condition {
    if cols.is_empty() {
        return Condition::True;
    }

    match fuzz_cond {
        FuzzCondition::True => Condition::True,
        FuzzCondition::Eq(col_idx, val) => {
            let col = &cols[*col_idx % cols.len()];
            Condition::Eq(sanitize_name(&col.name), val.to_value())
        },
        FuzzCondition::Ne(col_idx, val) => {
            let col = &cols[*col_idx % cols.len()];
            Condition::Ne(sanitize_name(&col.name), val.to_value())
        },
        FuzzCondition::Lt(col_idx, val) => {
            let col = &cols[*col_idx % cols.len()];
            Condition::Lt(sanitize_name(&col.name), val.to_value())
        },
        FuzzCondition::Le(col_idx, val) => {
            let col = &cols[*col_idx % cols.len()];
            Condition::Le(sanitize_name(&col.name), val.to_value())
        },
        FuzzCondition::Gt(col_idx, val) => {
            let col = &cols[*col_idx % cols.len()];
            Condition::Gt(sanitize_name(&col.name), val.to_value())
        },
        FuzzCondition::Ge(col_idx, val) => {
            let col = &cols[*col_idx % cols.len()];
            Condition::Ge(sanitize_name(&col.name), val.to_value())
        },
    }
}
