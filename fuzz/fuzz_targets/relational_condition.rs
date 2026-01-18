#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use relational_engine::{Condition, Row, Value};
use tensor_store::{ScalarValue, TensorData, TensorValue};

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

    fn to_tensor_value(&self) -> TensorValue {
        match self {
            FuzzValue::Int(i) => TensorValue::Scalar(ScalarValue::Int(*i)),
            FuzzValue::Float(f) => TensorValue::Scalar(ScalarValue::Float(*f)),
            FuzzValue::String(s) => TensorValue::Scalar(ScalarValue::String(s.clone())),
            FuzzValue::Bool(b) => TensorValue::Scalar(ScalarValue::Bool(*b)),
            FuzzValue::Null => TensorValue::Scalar(ScalarValue::Null),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzField {
    name: String,
    value: FuzzValue,
}

#[derive(Arbitrary, Debug)]
enum FuzzCondition {
    Eq(String, FuzzValue),
    Ne(String, FuzzValue),
    Lt(String, FuzzValue),
    Le(String, FuzzValue),
    Gt(String, FuzzValue),
    Ge(String, FuzzValue),
    True,
}

impl FuzzCondition {
    fn to_condition(&self) -> Condition {
        match self {
            FuzzCondition::Eq(col, val) => Condition::Eq(col.clone(), val.to_value()),
            FuzzCondition::Ne(col, val) => Condition::Ne(col.clone(), val.to_value()),
            FuzzCondition::Lt(col, val) => Condition::Lt(col.clone(), val.to_value()),
            FuzzCondition::Le(col, val) => Condition::Le(col.clone(), val.to_value()),
            FuzzCondition::Gt(col, val) => Condition::Gt(col.clone(), val.to_value()),
            FuzzCondition::Ge(col, val) => Condition::Ge(col.clone(), val.to_value()),
            FuzzCondition::True => Condition::True,
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    id: u64,
    fields: Vec<FuzzField>,
    conditions: Vec<FuzzCondition>,
    combine_with_and: bool,
}

fuzz_target!(|input: FuzzInput| {
    // Limit field count to avoid OOM
    let fields: Vec<_> = input
        .fields
        .into_iter()
        .take(20)
        .filter(|f| !f.name.is_empty() && f.name.len() < 64)
        .collect();

    if fields.is_empty() {
        return;
    }

    // Build equivalent Row and TensorData
    let mut row_values: Vec<(String, Value)> = Vec::new();
    let mut tensor = TensorData::new();

    // Add _id field
    row_values.push(("_id".to_string(), Value::Int(input.id as i64)));
    tensor.set(
        "_id".to_string(),
        TensorValue::Scalar(ScalarValue::Int(input.id as i64)),
    );

    for field in &fields {
        row_values.push((field.name.clone(), field.value.to_value()));
        tensor.set(field.name.clone(), field.value.to_tensor_value());
    }

    let row = Row {
        id: input.id,
        values: row_values,
    };

    // Build condition (limit depth to avoid stack overflow)
    let conditions: Vec<_> = input.conditions.into_iter().take(10).collect();

    if conditions.is_empty() {
        return;
    }

    // Build a combined condition
    let mut iter = conditions.iter();
    let first = iter.next().unwrap().to_condition();
    let combined = if input.combine_with_and {
        iter.fold(first, |acc, c| acc.and(c.to_condition()))
    } else {
        iter.fold(first, |acc, c| acc.or(c.to_condition()))
    };

    // Evaluate both ways and verify consistency
    let row_result = combined.evaluate(&row);
    let tensor_result = combined.evaluate_tensor(&tensor);

    // They should produce the same result for equivalent data
    assert_eq!(
        row_result, tensor_result,
        "Mismatch: evaluate()={} vs evaluate_tensor()={} for condition {:?}",
        row_result, tensor_result, combined
    );
});
