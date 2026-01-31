// SPDX-License-Identifier: MIT OR Apache-2.0
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use relational_engine::{Column, ColumnType, Condition, RelationalEngine, Schema, Value};
use std::collections::HashMap;

#[derive(Arbitrary, Debug, Clone)]
enum TxOp {
    Insert { id: i64 },
    Update { old_id: i64, new_id: i64 },
    Delete { id: i64 },
    Commit,
    Rollback,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<TxOp>,
}

fuzz_target!(|input: FuzzInput| {
    let engine = RelationalEngine::new();

    // Create a simple table
    let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
    if engine.create_table("test", schema).is_err() {
        return;
    }

    // Limit operations to avoid unbounded growth
    let ops: Vec<_> = input.ops.into_iter().take(50).collect();
    let mut current_tx: Option<u64> = None;

    for op in ops {
        match op {
            TxOp::Insert { id } => {
                let tx_id = current_tx.unwrap_or_else(|| {
                    let tx = engine.begin_transaction();
                    current_tx = Some(tx);
                    tx
                });

                let mut values = HashMap::new();
                values.insert("id".to_string(), Value::Int(id));
                let _ = engine.tx_insert(tx_id, "test", values);
            },
            TxOp::Update { old_id, new_id } => {
                let tx_id = current_tx.unwrap_or_else(|| {
                    let tx = engine.begin_transaction();
                    current_tx = Some(tx);
                    tx
                });

                let mut updates = HashMap::new();
                updates.insert("id".to_string(), Value::Int(new_id));
                let _ = engine.tx_update(
                    tx_id,
                    "test",
                    Condition::Eq("id".to_string(), Value::Int(old_id)),
                    updates,
                );
            },
            TxOp::Delete { id } => {
                let tx_id = current_tx.unwrap_or_else(|| {
                    let tx = engine.begin_transaction();
                    current_tx = Some(tx);
                    tx
                });

                let _ = engine.tx_delete(
                    tx_id,
                    "test",
                    Condition::Eq("id".to_string(), Value::Int(id)),
                );
            },
            TxOp::Commit => {
                if let Some(tx_id) = current_tx.take() {
                    let _ = engine.commit(tx_id);
                }
            },
            TxOp::Rollback => {
                if let Some(tx_id) = current_tx.take() {
                    let _ = engine.rollback(tx_id);
                }
            },
        }
    }

    // Clean up any uncommitted transaction
    if let Some(tx_id) = current_tx {
        let _ = engine.rollback(tx_id);
    }

    // Verify engine is in consistent state
    let result = engine.select("test", Condition::True);
    assert!(result.is_ok(), "Engine should be queryable after transaction operations");
});
