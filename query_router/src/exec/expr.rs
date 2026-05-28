// SPDX-License-Identifier: MIT OR Apache-2.0
//! Expression-conversion helpers shared across every exec module.
//!
//! These free functions convert parsed [`Expr`] values into engine-specific
//! types: [`Value`] (relational), [`Condition`] (relational filter),
//! [`FilterCondition`]/[`FilterValue`] (vector), [`PropertyValue`] (graph),
//! and primitives (`u64`/`f32`/`f64`/`usize`/`String`/`Vec<u8>`). They also
//! cover row-level helpers shared by SQL execution (sorting, join column
//! extraction, value comparison, row merging).
//!
//! `eval_string_expr` belongs here rather than `vault.rs` because every
//! subsystem with string arguments (vault, blob, checkpoint, chain, cluster)
//! evaluates them through the same logic.

use std::collections::HashMap;

use graph_engine::PropertyValue;
use neumann_parser::{
    self as parser, BinaryOp, Expr, ExprKind, JoinCondition, Literal, NullsOrder, Property,
    SortDirection,
};
use relational_engine::{Condition, Row, Value};
use vector_engine::{FilterCondition, FilterValue};

use crate::{QueryRouter, Result, RouterError};

/// Convert an expression to a relational `Condition`.
#[allow(
    clippy::only_used_in_recursion,
    reason = "router reserved for later subsystem hooks"
)]
pub fn expr_to_condition(router: &QueryRouter, expr: &Expr) -> Result<Condition> {
    match &expr.kind {
        ExprKind::Binary(left, op, right) => match op {
            BinaryOp::And => {
                let l = expr_to_condition(router, left)?;
                let r = expr_to_condition(router, right)?;
                Ok(l.and(r))
            },
            BinaryOp::Or => {
                let l = expr_to_condition(router, left)?;
                let r = expr_to_condition(router, right)?;
                Ok(l.or(r))
            },
            BinaryOp::Eq => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_value(right)?;
                Ok(Condition::Eq(col, val))
            },
            BinaryOp::Ne => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_value(right)?;
                Ok(Condition::Ne(col, val))
            },
            BinaryOp::Lt => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_value(right)?;
                Ok(Condition::Lt(col, val))
            },
            BinaryOp::Le => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_value(right)?;
                Ok(Condition::Le(col, val))
            },
            BinaryOp::Gt => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_value(right)?;
                Ok(Condition::Gt(col, val))
            },
            BinaryOp::Ge => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_value(right)?;
                Ok(Condition::Ge(col, val))
            },
            _ => Err(RouterError::ParseError(format!(
                "Unsupported operator in condition: {op:?}"
            ))),
        },
        _ => Err(RouterError::ParseError(
            "Expected binary expression in condition".to_string(),
        )),
    }
}

/// Convert an expression to a vector engine [`FilterCondition`].
#[allow(
    clippy::only_used_in_recursion,
    reason = "router reserved for later subsystem hooks"
)]
pub fn expr_to_filter_condition(router: &QueryRouter, expr: &Expr) -> Result<FilterCondition> {
    match &expr.kind {
        ExprKind::Binary(left, op, right) => match op {
            BinaryOp::And => {
                let l = expr_to_filter_condition(router, left)?;
                let r = expr_to_filter_condition(router, right)?;
                Ok(l.and(r))
            },
            BinaryOp::Or => {
                let l = expr_to_filter_condition(router, left)?;
                let r = expr_to_filter_condition(router, right)?;
                Ok(l.or(r))
            },
            BinaryOp::Eq => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_filter_value(right)?;
                Ok(FilterCondition::Eq(col, val))
            },
            BinaryOp::Ne => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_filter_value(right)?;
                Ok(FilterCondition::Ne(col, val))
            },
            BinaryOp::Lt => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_filter_value(right)?;
                Ok(FilterCondition::Lt(col, val))
            },
            BinaryOp::Le => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_filter_value(right)?;
                Ok(FilterCondition::Le(col, val))
            },
            BinaryOp::Gt => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_filter_value(right)?;
                Ok(FilterCondition::Gt(col, val))
            },
            BinaryOp::Ge => {
                let col = expr_to_column_name(left)?;
                let val = expr_to_filter_value(right)?;
                Ok(FilterCondition::Ge(col, val))
            },
            _ => Err(RouterError::ParseError(format!(
                "Unsupported operator in filter condition: {op:?}"
            ))),
        },
        _ => Err(RouterError::ParseError(
            "Expected binary expression in filter condition".to_string(),
        )),
    }
}

/// Convert an expression to a vector engine [`FilterValue`].
pub fn expr_to_filter_value(expr: &Expr) -> Result<FilterValue> {
    match &expr.kind {
        ExprKind::Literal(lit) => match lit {
            Literal::Null => Ok(FilterValue::String("null".to_string())),
            Literal::Boolean(b) => Ok(FilterValue::Bool(*b)),
            Literal::Integer(i) => Ok(FilterValue::Int(*i)),
            Literal::Float(f) => Ok(FilterValue::Float(*f)),
            Literal::String(s) => Ok(FilterValue::String(s.clone())),
        },
        ExprKind::Ident(ident) => Ok(FilterValue::String(ident.name.clone())),
        _ => Err(RouterError::ParseError(format!(
            "Cannot convert expression to filter value: {:?}",
            expr.kind
        ))),
    }
}

/// Convert an expression to a relational [`Value`].
pub fn expr_to_value(expr: &Expr) -> Result<Value> {
    match &expr.kind {
        ExprKind::Literal(lit) => match lit {
            Literal::Null => Ok(Value::Null),
            Literal::Boolean(b) => Ok(Value::Bool(*b)),
            Literal::Integer(i) => Ok(Value::Int(*i)),
            Literal::Float(f) => Ok(Value::Float(*f)),
            Literal::String(s) => Ok(Value::String(s.clone())),
        },
        ExprKind::Ident(ident) => Ok(Value::String(ident.name.clone())),
        ExprKind::Unary(parser::UnaryOp::Neg, inner) => match &inner.kind {
            ExprKind::Literal(Literal::Integer(i)) => Ok(Value::Int(-*i)),
            ExprKind::Literal(Literal::Float(f)) => Ok(Value::Float(-*f)),
            _ => Err(RouterError::ParseError(format!(
                "Cannot negate expression: {:?}",
                inner.kind
            ))),
        },
        _ => Err(RouterError::ParseError(format!(
            "Cannot convert expression to value: {:?}",
            expr.kind
        ))),
    }
}

/// Extract a column name from an expression.
pub fn expr_to_column_name(expr: &Expr) -> Result<String> {
    match &expr.kind {
        ExprKind::Ident(ident) => Ok(ident.name.clone()),
        ExprKind::Qualified(_, name) => Ok(name.name.clone()),
        _ => Err(RouterError::ParseError("Expected column name".to_string())),
    }
}

/// Convert an expression to a non-negative `u64`.
#[allow(clippy::cast_sign_loss, reason = "value checked >= 0")]
pub fn expr_to_u64(expr: &Expr) -> Result<u64> {
    match &expr.kind {
        ExprKind::Literal(Literal::Integer(i)) if *i >= 0 => Ok(*i as u64),
        _ => Err(RouterError::InvalidArgument(
            "Expected positive integer".to_string(),
        )),
    }
}

/// Convert an expression to `f32`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "downcast acceptable for vector coords"
)]
pub fn expr_to_f32(expr: &Expr) -> Result<f32> {
    match &expr.kind {
        ExprKind::Literal(Literal::Float(f)) => Ok(*f as f32),
        ExprKind::Literal(Literal::Integer(i)) => Ok(*i as f32),
        ExprKind::Unary(parser::UnaryOp::Neg, inner) => match &inner.kind {
            ExprKind::Literal(Literal::Float(f)) => Ok(-(*f as f32)),
            ExprKind::Literal(Literal::Integer(i)) => Ok(-(*i as f32)),
            _ => Err(RouterError::InvalidArgument("Expected number".to_string())),
        },
        _ => Err(RouterError::InvalidArgument("Expected number".to_string())),
    }
}

/// Convert an expression to `f64`.
#[allow(
    clippy::cast_precision_loss,
    reason = "integer -> f64 precision acceptable"
)]
pub fn expr_to_f64(expr: &Expr) -> Result<f64> {
    match &expr.kind {
        ExprKind::Literal(Literal::Float(f)) => Ok(*f),
        ExprKind::Literal(Literal::Integer(i)) => Ok(*i as f64),
        ExprKind::Unary(parser::UnaryOp::Neg, inner) => match &inner.kind {
            ExprKind::Literal(Literal::Float(f)) => Ok(-*f),
            ExprKind::Literal(Literal::Integer(i)) => Ok(-(*i as f64)),
            _ => Err(RouterError::InvalidArgument("Expected number".to_string())),
        },
        _ => Err(RouterError::InvalidArgument("Expected number".to_string())),
    }
}

/// Convert an expression to a non-negative `usize`.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "value checked >= 0; truncation acceptable on 32-bit systems"
)]
pub fn expr_to_usize(expr: &Expr) -> Result<usize> {
    match &expr.kind {
        ExprKind::Literal(Literal::Integer(i)) if *i >= 0 => Ok(*i as usize),
        _ => Err(RouterError::InvalidArgument(
            "Expected positive integer".to_string(),
        )),
    }
}

/// Convert an expression to a `String` (literal or identifier).
pub fn expr_to_string(expr: &Expr) -> Result<String> {
    match &expr.kind {
        ExprKind::Literal(Literal::String(s)) => Ok(s.clone()),
        ExprKind::Ident(ident) => Ok(ident.name.clone()),
        _ => Err(RouterError::InvalidArgument("Expected string".to_string())),
    }
}

/// Evaluate an expression to a string for vault/blob/checkpoint/chain/cluster ops.
pub fn eval_string_expr(expr: &Expr) -> Result<String> {
    match &expr.kind {
        ExprKind::Literal(Literal::String(s)) => Ok(s.clone()),
        ExprKind::Ident(ident) => Ok(ident.name.clone()),
        _ => Err(RouterError::InvalidArgument(
            "Expected string literal or identifier".to_string(),
        )),
    }
}

/// Convert an expression to a byte vector for inline blob data.
pub fn expr_to_bytes(expr: &Expr) -> Result<Vec<u8>> {
    match &expr.kind {
        ExprKind::Literal(Literal::String(s)) => Ok(s.as_bytes().to_vec()),
        _ => Err(RouterError::InvalidArgument(
            "Expected string literal for blob data".to_string(),
        )),
    }
}

/// Convert an expression to a graph `PropertyValue`.
pub fn expr_to_property_value(expr: &Expr) -> Result<PropertyValue> {
    match &expr.kind {
        ExprKind::Literal(lit) => match lit {
            Literal::Null => Ok(PropertyValue::Null),
            Literal::Boolean(b) => Ok(PropertyValue::Bool(*b)),
            Literal::Integer(i) => Ok(PropertyValue::Int(*i)),
            Literal::Float(f) => Ok(PropertyValue::Float(*f)),
            Literal::String(s) => Ok(PropertyValue::String(s.clone())),
        },
        ExprKind::Ident(ident) => Ok(PropertyValue::String(ident.name.clone())),
        ExprKind::Unary(parser::UnaryOp::Neg, inner) => match &inner.kind {
            ExprKind::Literal(Literal::Integer(i)) => Ok(PropertyValue::Int(-*i)),
            ExprKind::Literal(Literal::Float(f)) => Ok(PropertyValue::Float(-*f)),
            _ => Err(RouterError::InvalidArgument(format!(
                "Cannot negate expression: {:?}",
                inner.kind
            ))),
        },
        _ => Err(RouterError::InvalidArgument(format!(
            "Invalid property value: {:?}",
            expr.kind
        ))),
    }
}

/// Extract a numeric `f64` from an optional `PropertyValue`.
#[allow(
    clippy::cast_precision_loss,
    reason = "integer -> f64 precision acceptable for aggregates"
)]
#[allow(
    clippy::needless_pass_by_value,
    reason = "matches QueryRouter wrapper signature"
)]
pub fn property_value_to_f64(value: Option<PropertyValue>) -> Option<f64> {
    match value {
        Some(PropertyValue::Int(i)) => Some(i as f64),
        Some(PropertyValue::Float(f)) => Some(f),
        _ => None,
    }
}

/// Extract `(left_column, right_column)` from a JOIN ON / USING condition.
pub fn extract_join_columns(condition: &JoinCondition) -> Result<(String, String)> {
    match condition {
        JoinCondition::On(expr) => match &expr.kind {
            ExprKind::Binary(left, BinaryOp::Eq, right) => {
                let left_col = extract_column_from_expr(left)?;
                let right_col = extract_column_from_expr(right)?;
                Ok((left_col, right_col))
            },
            _ => Err(RouterError::ParseError(
                "JOIN ON condition must be an equality comparison (a.col = b.col)".to_string(),
            )),
        },
        JoinCondition::Using(cols) => {
            if cols.len() == 1 {
                let col = cols[0].name.clone();
                Ok((col.clone(), col))
            } else {
                Err(RouterError::ParseError(
                    "JOIN USING with multiple columns not yet supported".to_string(),
                ))
            }
        },
    }
}

/// Extract a column name from an expression (used inside JOIN conditions).
pub fn extract_column_from_expr(expr: &Expr) -> Result<String> {
    match &expr.kind {
        ExprKind::Ident(ident) => Ok(ident.name.clone()),
        ExprKind::Qualified(_, col) => Ok(col.name.clone()),
        _ => Err(RouterError::ParseError(
            "Expected column reference in JOIN condition".to_string(),
        )),
    }
}

/// Resolve `(left_col, right_col)` for a JOIN, with a clearer error when the
/// condition is missing.
pub fn get_join_columns(
    condition: Option<&JoinCondition>,
    _left_table: &str,
    _right_table: &str,
) -> Result<(String, String)> {
    condition.map_or_else(
        || {
            Err(RouterError::ParseError(
                "JOIN requires ON or USING clause (except CROSS/NATURAL)".to_string(),
            ))
        },
        extract_join_columns,
    )
}

/// Merge two rows from a JOIN, prefixing every column with its table name.
#[allow(
    clippy::cast_possible_wrap,
    reason = "row ids fit in i64 for any sane row count"
)]
pub fn merge_rows(row_a: Option<&Row>, row_b: Option<&Row>, table_a: &str, table_b: &str) -> Row {
    let mut values = Vec::new();

    if let Some(r) = row_a {
        values.push((format!("{table_a}._id"), Value::Int(r.id as i64)));
        for (col, val) in &r.values {
            values.push((format!("{table_a}.{col}"), val.clone()));
        }
    }

    if let Some(r) = row_b {
        values.push((format!("{table_b}._id"), Value::Int(r.id as i64)));
        for (col, val) in &r.values {
            values.push((format!("{table_b}.{col}"), val.clone()));
        }
    }

    Row {
        id: row_a
            .map(|r| r.id)
            .or_else(|| row_b.map(|r| r.id))
            .unwrap_or(0),
        values,
    }
}

/// Convert a list of parsed [`Property`] entries into a `PropertyValue` map.
pub fn properties_to_map(properties: &[Property]) -> Result<HashMap<String, PropertyValue>> {
    let mut map = HashMap::new();
    for prop in properties {
        let value = expr_to_property_value(&prop.value)?;
        map.insert(prop.key.name.clone(), value);
    }
    Ok(map)
}

/// Translate a parser `DataType` to a relational `ColumnType`.
pub fn data_type_to_column_type(dt: &parser::DataType) -> Result<relational_engine::ColumnType> {
    use parser::DataType;
    match dt {
        DataType::Int | DataType::Integer | DataType::Bigint | DataType::Smallint => {
            Ok(relational_engine::ColumnType::Int)
        },
        DataType::Float
        | DataType::Double
        | DataType::Real
        | DataType::Decimal(_, _)
        | DataType::Numeric(_, _) => Ok(relational_engine::ColumnType::Float),
        DataType::Varchar(_)
        | DataType::Char(_)
        | DataType::Text
        | DataType::Date
        | DataType::Time
        | DataType::Timestamp
        | DataType::Blob => Ok(relational_engine::ColumnType::String),
        DataType::Boolean => Ok(relational_engine::ColumnType::Bool),
        DataType::Custom(name) => match name.to_uppercase().as_str() {
            "STRING" => Ok(relational_engine::ColumnType::String),
            "BOOL" => Ok(relational_engine::ColumnType::Bool),
            _ => Err(RouterError::ParseError(format!(
                "Unsupported data type: {name}"
            ))),
        },
    }
}

/// Render a `PropertyValue` as a flat string (recurses into list/map).
pub fn property_to_string(prop: &PropertyValue) -> String {
    match prop {
        PropertyValue::Null => "null".to_string(),
        PropertyValue::Int(i) => i.to_string(),
        PropertyValue::Float(f) => f.to_string(),
        PropertyValue::String(s) => s.clone(),
        PropertyValue::Bool(b) => b.to_string(),
        PropertyValue::DateTime(ts) => ts.to_string(),
        PropertyValue::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(property_to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        PropertyValue::Map(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", k, property_to_string(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        PropertyValue::Bytes(bytes) => format!("<{} bytes>", bytes.len()),
        PropertyValue::Point { lat, lon } => format!("POINT({lat}, {lon})"),
    }
}

/// Compare two relational `Value`s for SQL ORDER BY / JOIN evaluation.
#[allow(
    clippy::cast_precision_loss,
    reason = "cross-type numeric comparison accepts int->f64 precision loss"
)]
pub fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y),
        (Value::Int(x), Value::Float(y)) => (*x as f64).partial_cmp(y),
        (Value::Float(x), Value::Int(y)) => x.partial_cmp(&(*y as f64)),
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        (Value::Bool(x), Value::Bool(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// Compare two values, honoring NULLS FIRST/LAST.
pub fn compare_values_with_nulls(
    a: Option<&Value>,
    b: Option<&Value>,
    nulls_order: Option<NullsOrder>,
) -> std::cmp::Ordering {
    use std::cmp::Ordering;

    match (a, b) {
        (None, None) | (Some(Value::Null), Some(Value::Null)) => Ordering::Equal,
        (None | Some(Value::Null), _) => match nulls_order.unwrap_or(NullsOrder::Last) {
            NullsOrder::First => Ordering::Less,
            NullsOrder::Last => Ordering::Greater,
        },
        (_, None | Some(Value::Null)) => match nulls_order.unwrap_or(NullsOrder::Last) {
            NullsOrder::First => Ordering::Greater,
            NullsOrder::Last => Ordering::Less,
        },
        (Some(va), Some(vb)) => compare_values(va, vb).unwrap_or(Ordering::Equal),
    }
}

/// Resolve the value of an expression against a row (for ORDER BY).
pub fn get_sort_value(expr: &Expr, row: &Row) -> Option<Value> {
    match &expr.kind {
        ExprKind::Ident(ident) => row
            .values
            .iter()
            .find(|(col, _)| col == &ident.name)
            .or_else(|| {
                row.values
                    .iter()
                    .find(|(col, _)| col.ends_with(&format!(".{}", ident.name)))
            })
            .map(|(_, v)| v.clone()),
        ExprKind::Qualified(table_expr, col) => {
            if let ExprKind::Ident(table) = &table_expr.kind {
                let full_name = format!("{}.{}", table.name, col.name);
                row.values
                    .iter()
                    .find(|(c, _)| c == &full_name)
                    .map(|(_, v)| v.clone())
            } else {
                None
            }
        },
        _ => None,
    }
}

/// Sort rows in place by a SQL ORDER BY clause.
pub fn sort_rows(rows: &mut [Row], order_by: &[neumann_parser::OrderByItem]) {
    rows.sort_by(|a, b| {
        for item in order_by {
            let val_a = get_sort_value(&item.expr, a);
            let val_b = get_sort_value(&item.expr, b);

            let cmp = compare_values_with_nulls(val_a.as_ref(), val_b.as_ref(), item.nulls);
            let cmp = match item.direction {
                SortDirection::Asc => cmp,
                SortDirection::Desc => cmp.reverse(),
            };

            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    });
}

/// Render an aggregate `Call` expression as its default column name
/// (e.g. `SUM(amount)`, `COUNT(*)`). Mirrors the SELECT projection naming.
pub fn aggregate_default_name(expr: &Expr) -> String {
    if let ExprKind::Call(call) = &expr.kind {
        let name = call.name.name.to_uppercase();
        if call.args.is_empty() || matches!(&call.args[0].kind, ExprKind::Wildcard) {
            format!("{name}(*)")
        } else if let Ok(col) = expr_to_column_name(&call.args[0]) {
            format!("{name}({col})")
        } else {
            format!("{name}(?)")
        }
    } else {
        "?".to_string()
    }
}

/// Resolve an expression to a row value (for JOIN / WHERE / HAVING evaluation).
pub fn get_row_value(expr: &Expr, row: &Row) -> Option<Value> {
    match &expr.kind {
        ExprKind::Literal(lit) => Some(match lit {
            Literal::Null => Value::Null,
            Literal::Boolean(b) => Value::Bool(*b),
            Literal::Integer(i) => Value::Int(*i),
            Literal::Float(f) => Value::Float(*f),
            Literal::String(s) => Value::String(s.clone()),
        }),
        ExprKind::Ident(ident) => row
            .values
            .iter()
            .find(|(col, _)| col == &ident.name || col.ends_with(&format!(".{}", ident.name)))
            .map(|(_, v)| v.clone()),
        ExprKind::Qualified(table_expr, col) => {
            if let ExprKind::Ident(table) = &table_expr.kind {
                let full_name = format!("{}.{}", table.name, col.name);
                row.values
                    .iter()
                    .find(|(c, _)| c == &full_name)
                    .map(|(_, v)| v.clone())
            } else {
                None
            }
        },
        ExprKind::Call(_) => {
            // Aggregate calls (SUM(col), COUNT(*), etc.) get stored under
            // their default name during GROUP BY computation.
            let col_name = aggregate_default_name(expr);
            row.values
                .iter()
                .find(|(c, _)| c == &col_name)
                .map(|(_, v)| v.clone())
        },
        _ => None,
    }
}

/// Evaluate a JOIN ON condition against a (merged) row.
pub fn evaluate_join_condition(expr: &Expr, row: &Row) -> bool {
    match &expr.kind {
        ExprKind::Binary(left, op, right) => {
            let left_val = get_row_value(left, row);
            let right_val = get_row_value(right, row);
            match (left_val, right_val) {
                (Some(l), Some(r)) => match op {
                    BinaryOp::Eq => l == r,
                    BinaryOp::Ne => l != r,
                    BinaryOp::Lt => compare_values(&l, &r) == Some(std::cmp::Ordering::Less),
                    BinaryOp::Le => matches!(
                        compare_values(&l, &r),
                        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                    ),
                    BinaryOp::Gt => compare_values(&l, &r) == Some(std::cmp::Ordering::Greater),
                    BinaryOp::Ge => matches!(
                        compare_values(&l, &r),
                        Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
                    ),
                    BinaryOp::And => l.is_truthy() && r.is_truthy(),
                    BinaryOp::Or => l.is_truthy() || r.is_truthy(),
                    _ => false,
                },
                _ => false,
            }
        },
        ExprKind::Ident(_) | ExprKind::Qualified(_, _) => {
            get_row_value(expr, row).is_some_and(|v| v.is_truthy())
        },
        _ => true,
    }
}
