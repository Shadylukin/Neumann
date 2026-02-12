// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use relational_engine::{Condition, Row, Value};

#[derive(Arbitrary, Debug, Clone)]
enum FuzzValue {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
    Bytes(Vec<u8>),
}

impl FuzzValue {
    fn to_value(&self) -> Value {
        match self {
            FuzzValue::Null => Value::Null,
            FuzzValue::Int(i) => Value::Int(*i),
            FuzzValue::Float(f) => Value::Float(*f),
            FuzzValue::String(s) => Value::String(s.clone()),
            FuzzValue::Bool(b) => Value::Bool(*b),
            FuzzValue::Bytes(b) => Value::Bytes(b.clone()),
        }
    }
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    left: FuzzValue,
    right: FuzzValue,
}

fuzz_target!(|input: FuzzInput| {
    let left = input.left.to_value();
    let right = input.right.to_value();

    // Check for NaN values which have special comparison semantics (NaN != NaN)
    let left_is_nan = matches!(&left, Value::Float(f) if f.is_nan());
    let right_is_nan = matches!(&right, Value::Float(f) if f.is_nan());

    // Test equality is reflexive (skip for NaN since NaN != NaN by IEEE754)
    if !left_is_nan {
        let left_clone = left.clone();
        assert!(left == left_clone, "Equality should be reflexive");
    }

    // Test equality is symmetric (skip NaN comparisons)
    if !left_is_nan && !right_is_nan {
        let eq_lr = left == right;
        let eq_rl = right == left;
        assert_eq!(eq_lr, eq_rl, "Equality should be symmetric");
    }

    // Test comparison via Condition evaluation (public API)
    let row = Row {
        id: 1,
        values: vec![("val".to_string(), left.clone())],
    };

    // Test Eq condition
    let eq_cond = Condition::Eq("val".to_string(), right.clone());
    let eq_result = eq_cond.evaluate(&row);

    // Test consistency: if PartialEq says equal, Condition::Eq should match
    // Skip NaN comparisons since NaN != NaN by IEEE754
    if !left_is_nan && !right_is_nan && left == right {
        assert!(eq_result, "Eq condition should match equal values");
    }

    // Test Lt condition
    let lt_cond = Condition::Lt("val".to_string(), right.clone());
    let lt_result = lt_cond.evaluate(&row);

    // Test Gt condition
    let gt_cond = Condition::Gt("val".to_string(), right.clone());
    let gt_result = gt_cond.evaluate(&row);

    // Test Le condition
    let le_cond = Condition::Le("val".to_string(), right.clone());
    let le_result = le_cond.evaluate(&row);

    // Test Ge condition
    let ge_cond = Condition::Ge("val".to_string(), right.clone());
    let ge_result = ge_cond.evaluate(&row);

    // Check if values are comparable via partial_cmp_value
    // Null, Bool, and cross-type comparisons return None from partial_cmp_value
    // so Lt/Le/Gt/Ge return false even when Eq returns true
    let values_are_comparable = matches!(
        (&left, &right),
        (Value::Int(_), Value::Int(_))
            | (Value::Float(_), Value::Float(_))
            | (Value::String(_), Value::String(_))
            | (Value::Bytes(_), Value::Bytes(_))
            | (Value::Json(_), Value::Json(_))
    ) && !left_is_nan
        && !right_is_nan;

    // Verify ordering consistency (only for comparable types)
    if values_are_comparable {
        // If left < right: Lt should be true, Gt should be false
        // If left > right: Lt should be false, Gt should be true
        // If left == right: both Lt and Gt should be false
        if eq_result {
            assert!(
                !lt_result || !gt_result,
                "Equal values can't be both < and >"
            );
        }

        // Le should be Lt OR Eq
        if lt_result || eq_result {
            assert!(le_result, "Le should be true if Lt or Eq is true");
        }

        // Ge should be Gt OR Eq
        if gt_result || eq_result {
            assert!(ge_result, "Ge should be true if Gt or Eq is true");
        }
    }

    // Test hash consistency for Bytes
    if let (FuzzValue::Bytes(_), FuzzValue::Bytes(_)) = (&input.left, &input.right) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        if left == right {
            let mut h1 = DefaultHasher::new();
            let mut h2 = DefaultHasher::new();

            if let (Value::Bytes(b1), Value::Bytes(b2)) = (&left, &right) {
                b1.hash(&mut h1);
                b2.hash(&mut h2);
                assert_eq!(h1.finish(), h2.finish(), "Equal values should have equal hashes");
            }
        }
    }
});
