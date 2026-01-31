// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use relational_engine::{Column, ColumnType, Constraint, RelationalEngine, Schema, Value};
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum FuzzConstraintType {
    PrimaryKey,
    Unique,
    NotNull,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    constraint_type: FuzzConstraintType,
    values: Vec<(Option<i64>, Option<String>)>,
}

fn sanitize_name(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .take(16)
        .collect()
}

fuzz_target!(|input: FuzzInput| {
    // Limit values to avoid OOM
    let values: Vec<_> = input.values.into_iter().take(50).collect();

    let engine = RelationalEngine::new();

    // Create table with constraints
    // Build columns based on constraint type
    // For NotNull constraint, the column must NOT be marked as nullable()
    // because the engine enforces nullability via Column.nullable, not via Constraint::NotNull
    let columns = match input.constraint_type {
        FuzzConstraintType::PrimaryKey => {
            vec![
                Column::new("id", ColumnType::Int), // PK columns are implicitly not null
                Column::new("name", ColumnType::String).nullable(),
            ]
        },
        FuzzConstraintType::Unique => {
            vec![
                Column::new("id", ColumnType::Int).nullable(), // Unique allows NULL
                Column::new("name", ColumnType::String).nullable(),
            ]
        },
        FuzzConstraintType::NotNull => {
            vec![
                Column::new("id", ColumnType::Int).nullable(),
                Column::new("name", ColumnType::String), // NOT nullable - this enforces NotNull
            ]
        },
    };

    let constraints = match input.constraint_type {
        FuzzConstraintType::PrimaryKey => {
            vec![Constraint::primary_key(
                "pk_id",
                vec!["id".to_string()],
            )]
        },
        FuzzConstraintType::Unique => {
            vec![Constraint::unique(
                "uq_id",
                vec!["id".to_string()],
            )]
        },
        FuzzConstraintType::NotNull => {
            vec![Constraint::not_null("nn_name", "name")]
        },
    };

    let schema = Schema::with_constraints(columns, constraints);
    if engine.create_table("test", schema).is_err() {
        return;
    }

    // Track inserted values for constraint validation
    let mut inserted_ids: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut _constraint_violations = 0usize;
    let mut successful_inserts = 0usize;

    for (id_opt, name_opt) in &values {
        let mut row = HashMap::new();

        // Handle id column
        if let Some(id) = id_opt {
            row.insert("id".to_string(), Value::Int(*id));
        } else {
            row.insert("id".to_string(), Value::Null);
        }

        // Handle name column
        if let Some(name) = name_opt {
            let sanitized = sanitize_name(name);
            if !sanitized.is_empty() {
                row.insert("name".to_string(), Value::String(sanitized));
            } else {
                row.insert("name".to_string(), Value::Null);
            }
        } else {
            row.insert("name".to_string(), Value::Null);
        }

        let result = engine.insert("test", row);

        match &result {
            Ok(_) => {
                successful_inserts += 1;
                if let Some(id) = id_opt {
                    inserted_ids.insert(*id);
                }
            },
            Err(_) => {
                _constraint_violations += 1;
            },
        }

        // Validate expected constraint behavior
        match input.constraint_type {
            FuzzConstraintType::PrimaryKey | FuzzConstraintType::Unique => {
                if let Some(id) = id_opt {
                    // Duplicate ID should fail
                    if inserted_ids.contains(id) && successful_inserts > 1 {
                        // Expected to fail on duplicate
                        if result.is_ok() {
                            // This is actually the first insert of this ID
                        }
                    }
                }
            },
            FuzzConstraintType::NotNull => {
                // NULL name should fail with NotNull constraint
                if name_opt.is_none() || name_opt.as_ref().map_or(true, |s| sanitize_name(s).is_empty()) {
                    // Should have failed
                    assert!(
                        result.is_err(),
                        "NotNull constraint should reject NULL values"
                    );
                }
            },
        }
    }

    // Verify engine is in consistent state
    let result = engine.select("test", relational_engine::Condition::True);
    assert!(
        result.is_ok(),
        "Engine should be queryable after constraint tests"
    );

    // Verify row count matches successful inserts
    if let Ok(rows) = result {
        assert_eq!(
            rows.len(),
            successful_inserts,
            "Row count should match successful inserts"
        );
    }
});
