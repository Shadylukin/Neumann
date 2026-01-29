// SPDX-License-Identifier: MIT OR Apache-2.0
//! Integration tests for RelationalEngine transaction API.

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
    thread,
};

use integration_tests::create_shared_engines;
use relational_engine::{Column, ColumnType, Condition, RelationalError, Schema, Value};

// ============================================================================
// 1. Transaction Lifecycle
// ============================================================================

#[test]
fn test_transaction_begin_returns_unique_id() {
    let (_, engine, _, _) = create_shared_engines();

    let tx1 = engine.begin_transaction();
    let tx2 = engine.begin_transaction();
    let tx3 = engine.begin_transaction();

    assert!(tx2 > tx1);
    assert!(tx3 > tx2);
    assert!(engine.is_transaction_active(tx1));
    assert!(engine.is_transaction_active(tx2));
    assert!(engine.is_transaction_active(tx3));

    // Cleanup
    engine.rollback(tx1).unwrap();
    engine.rollback(tx2).unwrap();
    engine.rollback(tx3).unwrap();
}

#[test]
fn test_transaction_commit_releases_resources() {
    let (_, engine, _, _) = create_shared_engines();

    let initial_count = engine.active_transaction_count();
    let tx = engine.begin_transaction();

    assert!(engine.is_transaction_active(tx));
    assert_eq!(engine.active_transaction_count(), initial_count + 1);

    engine.commit(tx).unwrap();

    assert!(!engine.is_transaction_active(tx));
    assert_eq!(engine.active_transaction_count(), initial_count);
}

#[test]
fn test_transaction_rollback_releases_resources() {
    let (_, engine, _, _) = create_shared_engines();

    let initial_count = engine.active_transaction_count();
    let tx = engine.begin_transaction();

    assert!(engine.is_transaction_active(tx));

    engine.rollback(tx).unwrap();

    assert!(!engine.is_transaction_active(tx));
    assert_eq!(engine.active_transaction_count(), initial_count);
}

// ============================================================================
// 2. tx_insert
// ============================================================================

#[test]
fn test_tx_insert_commit_persists_data() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
    ]);
    engine.create_table("users", schema).unwrap();

    let tx = engine.begin_transaction();

    for i in 0..5 {
        let row = HashMap::from([
            ("id".to_string(), Value::Int(i)),
            ("name".to_string(), Value::String(format!("User{i}"))),
        ]);
        engine.tx_insert(tx, "users", row).unwrap();
    }

    engine.commit(tx).unwrap();

    let rows = engine.select("users", Condition::True).unwrap();
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_tx_insert_rollback_discards_data() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int),
    ]);
    engine.create_table("data", schema).unwrap();

    let tx = engine.begin_transaction();

    for i in 0..10 {
        let row = HashMap::from([
            ("id".to_string(), Value::Int(i)),
            ("value".to_string(), Value::Int(i * 100)),
        ]);
        engine.tx_insert(tx, "data", row).unwrap();
    }

    engine.rollback(tx).unwrap();

    let rows = engine.select("data", Condition::True).unwrap();
    assert!(rows.is_empty());
}

#[test]
fn test_tx_insert_validates_schema() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("count", ColumnType::Int),
    ]);
    engine.create_table("items", schema).unwrap();

    let tx = engine.begin_transaction();

    // Wrong type: String instead of Int
    let row = HashMap::from([
        ("id".to_string(), Value::Int(1)),
        ("count".to_string(), Value::String("wrong".to_string())),
    ]);
    let result = engine.tx_insert(tx, "items", row);

    assert!(matches!(result, Err(RelationalError::TypeMismatch { .. })));
    assert!(engine.is_transaction_active(tx));

    engine.rollback(tx).unwrap();
}

// ============================================================================
// 3. tx_update
// ============================================================================

#[test]
fn test_tx_update_commit_persists_changes() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("balance", ColumnType::Int),
    ]);
    engine.create_table("accounts", schema).unwrap();

    // Insert initial data
    engine
        .insert(
            "accounts",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("balance".to_string(), Value::Int(1000)),
            ]),
        )
        .unwrap();

    let tx = engine.begin_transaction();
    let updates = HashMap::from([("balance".to_string(), Value::Int(500))]);
    let count = engine
        .tx_update(
            tx,
            "accounts",
            Condition::Eq("id".to_string(), Value::Int(1)),
            updates,
        )
        .unwrap();
    assert_eq!(count, 1);

    engine.commit(tx).unwrap();

    let rows = engine
        .select("accounts", Condition::Eq("id".to_string(), Value::Int(1)))
        .unwrap();
    assert_eq!(rows[0].get("balance"), Some(&Value::Int(500)));
}

#[test]
fn test_tx_update_rollback_restores_original() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("status", ColumnType::String),
    ]);
    engine.create_table("orders", schema).unwrap();

    engine
        .insert(
            "orders",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("status".to_string(), Value::String("pending".to_string())),
            ]),
        )
        .unwrap();

    let tx = engine.begin_transaction();
    engine
        .tx_update(
            tx,
            "orders",
            Condition::Eq("id".to_string(), Value::Int(1)),
            HashMap::from([("status".to_string(), Value::String("shipped".to_string()))]),
        )
        .unwrap();

    engine.rollback(tx).unwrap();

    let rows = engine
        .select("orders", Condition::Eq("id".to_string(), Value::Int(1)))
        .unwrap();
    assert_eq!(
        rows[0].get("status"),
        Some(&Value::String("pending".to_string()))
    );
}

#[test]
fn test_tx_update_multiple_rows_rollback() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("active", ColumnType::Bool),
    ]);
    engine.create_table("flags", schema).unwrap();

    for i in 0..5 {
        engine
            .insert(
                "flags",
                HashMap::from([
                    ("id".to_string(), Value::Int(i)),
                    ("active".to_string(), Value::Bool(true)),
                ]),
            )
            .unwrap();
    }

    let tx = engine.begin_transaction();
    let count = engine
        .tx_update(
            tx,
            "flags",
            Condition::True,
            HashMap::from([("active".to_string(), Value::Bool(false))]),
        )
        .unwrap();
    assert_eq!(count, 5);

    engine.rollback(tx).unwrap();

    let rows = engine
        .select(
            "flags",
            Condition::Eq("active".to_string(), Value::Bool(true)),
        )
        .unwrap();
    assert_eq!(rows.len(), 5);
}

// ============================================================================
// 4. tx_delete
// ============================================================================

#[test]
fn test_tx_delete_commit_removes_rows() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("temp", ColumnType::Bool),
    ]);
    engine.create_table("records", schema).unwrap();

    for i in 0..10 {
        engine
            .insert(
                "records",
                HashMap::from([
                    ("id".to_string(), Value::Int(i)),
                    ("temp".to_string(), Value::Bool(i < 5)),
                ]),
            )
            .unwrap();
    }

    let tx = engine.begin_transaction();
    let count = engine
        .tx_delete(
            tx,
            "records",
            Condition::Eq("temp".to_string(), Value::Bool(true)),
        )
        .unwrap();
    assert_eq!(count, 5);

    engine.commit(tx).unwrap();

    let rows = engine.select("records", Condition::True).unwrap();
    assert_eq!(rows.len(), 5);
}

#[test]
fn test_tx_delete_rollback_restores_rows() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("data", ColumnType::String),
    ]);
    engine.create_table("logs", schema).unwrap();

    for i in 0..3 {
        engine
            .insert(
                "logs",
                HashMap::from([
                    ("id".to_string(), Value::Int(i)),
                    ("data".to_string(), Value::String(format!("log{i}"))),
                ]),
            )
            .unwrap();
    }

    let tx = engine.begin_transaction();
    engine.tx_delete(tx, "logs", Condition::True).unwrap();

    engine.rollback(tx).unwrap();

    let rows = engine.select("logs", Condition::True).unwrap();
    assert_eq!(rows.len(), 3);
}

// ============================================================================
// 5. Lock Conflict
// ============================================================================

#[test]
fn test_lock_conflict_between_transactions() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int),
    ]);
    engine.create_table("shared", schema).unwrap();

    engine
        .insert(
            "shared",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Int(100)),
            ]),
        )
        .unwrap();

    let tx1 = engine.begin_transaction();
    engine
        .tx_update(
            tx1,
            "shared",
            Condition::Eq("id".to_string(), Value::Int(1)),
            HashMap::from([("value".to_string(), Value::Int(200))]),
        )
        .unwrap();

    let tx2 = engine.begin_transaction();
    let result = engine.tx_update(
        tx2,
        "shared",
        Condition::Eq("id".to_string(), Value::Int(1)),
        HashMap::from([("value".to_string(), Value::Int(300))]),
    );

    assert!(matches!(result, Err(RelationalError::LockConflict { .. })));

    engine.commit(tx1).unwrap();
    engine.rollback(tx2).unwrap();
}

#[test]
fn test_lock_release_on_commit() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("seq", ColumnType::Int),
    ]);
    engine.create_table("sequence", schema).unwrap();

    engine
        .insert(
            "sequence",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("seq".to_string(), Value::Int(0)),
            ]),
        )
        .unwrap();

    // tx1 updates and commits
    let tx1 = engine.begin_transaction();
    engine
        .tx_update(
            tx1,
            "sequence",
            Condition::True,
            HashMap::from([("seq".to_string(), Value::Int(1))]),
        )
        .unwrap();
    engine.commit(tx1).unwrap();

    // tx2 can now update same row
    let tx2 = engine.begin_transaction();
    let result = engine.tx_update(
        tx2,
        "sequence",
        Condition::True,
        HashMap::from([("seq".to_string(), Value::Int(2))]),
    );
    assert!(result.is_ok());
    engine.commit(tx2).unwrap();

    let rows = engine.select("sequence", Condition::True).unwrap();
    assert_eq!(rows[0].get("seq"), Some(&Value::Int(2)));
}

#[test]
fn test_lock_release_on_rollback() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("locked", ColumnType::Bool),
    ]);
    engine.create_table("resource", schema).unwrap();

    engine
        .insert(
            "resource",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("locked".to_string(), Value::Bool(false)),
            ]),
        )
        .unwrap();

    // tx1 updates but rolls back
    let tx1 = engine.begin_transaction();
    engine
        .tx_update(
            tx1,
            "resource",
            Condition::True,
            HashMap::from([("locked".to_string(), Value::Bool(true))]),
        )
        .unwrap();
    engine.rollback(tx1).unwrap();

    // tx2 can now acquire lock
    let tx2 = engine.begin_transaction();
    let result = engine.tx_update(
        tx2,
        "resource",
        Condition::True,
        HashMap::from([("locked".to_string(), Value::Bool(true))]),
    );
    assert!(result.is_ok());
    engine.commit(tx2).unwrap();
}

// ============================================================================
// 6. Index Consistency
// ============================================================================

#[test]
fn test_index_consistency_through_tx_commit() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("category", ColumnType::String),
    ]);
    engine.create_table("products", schema).unwrap();
    engine.create_index("products", "category").unwrap();

    let tx = engine.begin_transaction();
    for i in 0..10 {
        let cat = if i < 5 { "A" } else { "B" };
        engine
            .tx_insert(
                tx,
                "products",
                HashMap::from([
                    ("id".to_string(), Value::Int(i)),
                    ("category".to_string(), Value::String(cat.to_string())),
                ]),
            )
            .unwrap();
    }
    engine.commit(tx).unwrap();

    // Verify index lookup works
    let cat_a = engine
        .select(
            "products",
            Condition::Eq("category".to_string(), Value::String("A".to_string())),
        )
        .unwrap();
    assert_eq!(cat_a.len(), 5);

    let cat_b = engine
        .select(
            "products",
            Condition::Eq("category".to_string(), Value::String("B".to_string())),
        )
        .unwrap();
    assert_eq!(cat_b.len(), 5);
}

#[test]
fn test_index_rollback_restores_entries() {
    let (_, engine, _, _) = create_shared_engines();

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("tag", ColumnType::String),
    ]);
    engine.create_table("tagged", schema).unwrap();
    engine.create_index("tagged", "tag").unwrap();

    // Insert committed data
    engine
        .insert(
            "tagged",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("tag".to_string(), Value::String("important".to_string())),
            ]),
        )
        .unwrap();

    // Delete in transaction then rollback
    let tx = engine.begin_transaction();
    engine
        .tx_delete(
            tx,
            "tagged",
            Condition::Eq("tag".to_string(), Value::String("important".to_string())),
        )
        .unwrap();
    engine.rollback(tx).unwrap();

    // Verify index entry restored
    let rows = engine
        .select(
            "tagged",
            Condition::Eq("tag".to_string(), Value::String("important".to_string())),
        )
        .unwrap();
    assert_eq!(rows.len(), 1);
}

// ============================================================================
// 7. Concurrent Transactions
// ============================================================================

#[test]
fn test_concurrent_transactions_different_rows() {
    let (_, engine, _, _) = create_shared_engines();
    let engine = Arc::new(engine);

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("counter", ColumnType::Int),
    ]);
    engine.create_table("counters", schema).unwrap();

    // Insert 10 rows
    for i in 0..10 {
        engine
            .insert(
                "counters",
                HashMap::from([
                    ("id".to_string(), Value::Int(i)),
                    ("counter".to_string(), Value::Int(0)),
                ]),
            )
            .unwrap();
    }

    let success = Arc::new(AtomicUsize::new(0));
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let eng = Arc::clone(&engine);
            let s = Arc::clone(&success);
            thread::spawn(move || {
                let tx = eng.begin_transaction();
                let result = eng.tx_update(
                    tx,
                    "counters",
                    Condition::Eq("id".to_string(), Value::Int(i)),
                    HashMap::from([("counter".to_string(), Value::Int(i + 1))]),
                );
                if result.is_ok() && eng.commit(tx).is_ok() {
                    s.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(success.load(Ordering::SeqCst), 10);
}

#[test]
fn test_concurrent_transactions_same_row_conflict() {
    let (_, engine, _, _) = create_shared_engines();
    let engine = Arc::new(engine);

    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("value", ColumnType::Int),
    ]);
    engine.create_table("contested", schema).unwrap();

    engine
        .insert(
            "contested",
            HashMap::from([
                ("id".to_string(), Value::Int(1)),
                ("value".to_string(), Value::Int(0)),
            ]),
        )
        .unwrap();

    let success = Arc::new(AtomicUsize::new(0));
    let conflicts = Arc::new(AtomicUsize::new(0));

    // Use barrier to synchronize thread starts for guaranteed overlap
    let num_threads = 5;
    let barrier = Arc::new(Barrier::new(num_threads));

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let eng = Arc::clone(&engine);
            let s = Arc::clone(&success);
            let c = Arc::clone(&conflicts);
            let b = Arc::clone(&barrier);
            thread::spawn(move || {
                // Wait for all threads before starting transactions
                b.wait();
                let tx = eng.begin_transaction();
                let result = eng.tx_update(
                    tx,
                    "contested",
                    Condition::True,
                    HashMap::from([("value".to_string(), Value::Int(i as i64 + 1))]),
                );
                match result {
                    Ok(_) => {
                        if eng.commit(tx).is_ok() {
                            s.fetch_add(1, Ordering::SeqCst);
                        }
                    },
                    Err(RelationalError::LockConflict { .. }) => {
                        c.fetch_add(1, Ordering::SeqCst);
                        let _ = eng.rollback(tx);
                    },
                    _ => {},
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // At least one succeeded, rest got conflicts (total should equal num_threads)
    let total = success.load(Ordering::SeqCst) + conflicts.load(Ordering::SeqCst);
    assert_eq!(total, num_threads);
    assert!(success.load(Ordering::SeqCst) >= 1);
}
