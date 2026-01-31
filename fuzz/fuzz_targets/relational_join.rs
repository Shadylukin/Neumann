// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use relational_engine::{Column, ColumnType, RelationalEngine, Schema, Value};
use std::collections::HashMap;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    left_values: Vec<i64>,
    right_values: Vec<i64>,
}

fuzz_target!(|input: FuzzInput| {
    // Limit row counts to avoid OOM
    let left_values: Vec<_> = input.left_values.into_iter().take(20).collect();
    let right_values: Vec<_> = input.right_values.into_iter().take(20).collect();

    let engine = RelationalEngine::new();

    // Create left table
    let left_schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int),
    ]);
    if engine.create_table("left", left_schema).is_err() {
        return;
    }

    // Create right table
    let right_schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("data", ColumnType::Int),
    ]);
    if engine.create_table("right", right_schema).is_err() {
        return;
    }

    // Insert left rows
    for (i, val) in left_values.iter().enumerate() {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i as i64));
        row.insert("value".to_string(), Value::Int(*val));
        let _ = engine.insert("left", row);
    }

    // Insert right rows - use overlapping IDs to test join matching
    for (i, val) in right_values.iter().enumerate() {
        let mut row = HashMap::new();
        row.insert("id".to_string(), Value::Int(i as i64));
        row.insert("data".to_string(), Value::Int(*val));
        let _ = engine.insert("right", row);
    }

    // Execute hash join
    let result = engine.join("left", "right", "id", "id");

    // Verify join executed without panic
    match result {
        Ok(pairs) => {
            // For equi-join on id, we expect matching pairs
            let min_count = left_values.len().min(right_values.len());
            assert!(
                pairs.len() <= min_count,
                "Join produced more pairs than expected"
            );

            // Verify each pair has matching join keys
            for (left_row, right_row) in &pairs {
                let left_id = left_row.get("id");
                let right_id = right_row.get("id");
                assert_eq!(left_id, right_id, "Join pairs should have matching keys");
            }
        },
        Err(_) => {
            // Some joins may fail for valid reasons (e.g., schema mismatch)
        },
    }
});
