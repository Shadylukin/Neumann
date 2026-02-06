// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use relational_engine::{
    AggregateExpr, AggregateRef, Column, ColumnType, Condition, HavingCondition,
    RelationalEngine, Schema, Value,
};
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzAggregate {
    CountAll,
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

impl FuzzAggregate {
    fn to_aggregate(&self, col_name: &str) -> AggregateExpr {
        match self {
            FuzzAggregate::CountAll => AggregateExpr::CountAll,
            FuzzAggregate::Count => AggregateExpr::Count(col_name.to_string()),
            FuzzAggregate::Sum => AggregateExpr::Sum(col_name.to_string()),
            FuzzAggregate::Avg => AggregateExpr::Avg(col_name.to_string()),
            FuzzAggregate::Min => AggregateExpr::Min(col_name.to_string()),
            FuzzAggregate::Max => AggregateExpr::Max(col_name.to_string()),
        }
    }
}

#[derive(Arbitrary, Debug)]
enum FuzzHavingOp {
    Gt(f64),
    Ge(f64),
    Lt(f64),
    Le(f64),
    Eq(i64),
    Ne(i64),
}

impl FuzzHavingOp {
    fn to_having(&self, agg: &FuzzAggregate) -> HavingCondition {
        let agg_ref = match agg {
            FuzzAggregate::CountAll => AggregateRef::CountAll,
            FuzzAggregate::Count => AggregateRef::Count("value".to_string()),
            FuzzAggregate::Sum => AggregateRef::Sum("value".to_string()),
            FuzzAggregate::Avg => AggregateRef::Avg("value".to_string()),
            FuzzAggregate::Min => AggregateRef::Min("value".to_string()),
            FuzzAggregate::Max => AggregateRef::Max("value".to_string()),
        };

        match self {
            FuzzHavingOp::Gt(v) => HavingCondition::Gt(agg_ref, Value::Float(*v)),
            FuzzHavingOp::Ge(v) => HavingCondition::Ge(agg_ref, Value::Float(*v)),
            FuzzHavingOp::Lt(v) => HavingCondition::Lt(agg_ref, Value::Float(*v)),
            FuzzHavingOp::Le(v) => HavingCondition::Le(agg_ref, Value::Float(*v)),
            FuzzHavingOp::Eq(v) => HavingCondition::Eq(agg_ref, Value::Int(*v)),
            FuzzHavingOp::Ne(v) => HavingCondition::Ne(agg_ref, Value::Int(*v)),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    values: Vec<(i64, i64)>,
    aggregates: Vec<FuzzAggregate>,
    having: Option<(FuzzAggregate, FuzzHavingOp)>,
}

fuzz_target!(|input: FuzzInput| {
    // Limit values to avoid OOM
    let values: Vec<_> = input.values.into_iter().take(100).collect();
    let aggregates: Vec<_> = input.aggregates.into_iter().take(5).collect();

    if values.is_empty() || aggregates.is_empty() {
        return;
    }

    let engine = RelationalEngine::new();

    // Create table with group_key and value columns
    let schema = Schema::new(vec![
        Column::new("group_key", ColumnType::Int),
        Column::new("value", ColumnType::Int),
    ]);
    if engine.create_table("test", schema).is_err() {
        return;
    }

    // Insert rows
    for (group_key, value) in &values {
        let mut row = HashMap::new();
        row.insert("group_key".to_string(), Value::Int(*group_key));
        row.insert("value".to_string(), Value::Int(*value));
        let _ = engine.insert("test", row);
    }

    // Build aggregates
    let agg_exprs: Vec<_> = aggregates.iter().map(|a| a.to_aggregate("value")).collect();

    // Build HAVING condition if specified
    let having = input.having.as_ref().map(|(agg, op)| op.to_having(agg));

    // Execute grouped query
    let result = engine.select_grouped(
        "test",
        Condition::True,
        &["group_key".to_string()],
        &agg_exprs,
        having,
    );

    // Verify query executed without panic
    match result {
        Ok(groups) => {
            // Count distinct group keys in input
            let mut unique_keys = std::collections::HashSet::new();
            for (key, _) in &values {
                unique_keys.insert(*key);
            }

            // Without HAVING, group count should equal unique key count
            if input.having.is_none() {
                assert_eq!(
                    groups.len(),
                    unique_keys.len(),
                    "Group count mismatch without HAVING"
                );
            }

            // Verify each group has the expected aggregates
            for group in &groups {
                assert_eq!(
                    group.aggregates.len(),
                    agg_exprs.len(),
                    "Aggregate count mismatch"
                );
            }
        },
        Err(_) => {
            // Some queries may fail for valid reasons
        },
    }
});
