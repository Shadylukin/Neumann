// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! End-to-end linearizability tests through `QueryRouter` and `TensorStore`.
//!
//! Tests two levels:
//! 1. SQL through `QueryRouter::execute_parsed` (sequential + per-key-partitioned)
//! 2. Concurrent `TensorStore::put/get` via `get_store_from_router` (concurrent
//!    single-key and multi-key linearizability)

use std::sync::Arc;
use std::time::Duration;

use integration_tests::linearizability::{
    ConcurrentHistoryRecorder, HistoryRecorder, LinearizabilityChecker, LinearizabilityResult,
    OpType, RegisterModel, Value,
};
use integration_tests::{create_shared_router, get_store_from_router};
use query_router::QueryResult;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorValue};

// ---------------------------------------------------------------------------
// Helpers — SQL layer
// ---------------------------------------------------------------------------

fn exec(router: &query_router::QueryRouter, sql: &str) -> QueryResult {
    router
        .execute_parsed(sql)
        .unwrap_or_else(|e| panic!("SQL failed: {sql}: {e}"))
}

fn read_register_sql(router: &query_router::QueryRouter, kid: i64) -> Value {
    let result = exec(router, &format!("SELECT val FROM reg WHERE kid = {kid}"));
    match result {
        QueryResult::Rows(rows) if rows.is_empty() => Value::None,
        QueryResult::Rows(rows) => {
            let row = &rows[0];
            match row.get("val") {
                Some(relational_engine::Value::Int(v)) => Value::Int(*v),
                Some(relational_engine::Value::Null) => Value::None,
                None => Value::None,
                other => panic!("unexpected val type: {other:?}"),
            }
        },
        other => panic!("SELECT returned unexpected result: {other:?}"),
    }
}

fn write_register_sql(router: &query_router::QueryRouter, kid: i64, val: i64) {
    exec(
        router,
        &format!("UPDATE reg SET val = {val} WHERE kid = {kid}"),
    );
}

/// Set up the register table with `n` keys initialized to 0 and record
/// these initial writes in the history so the model is consistent.
fn setup_registers_with_history(
    router: &query_router::QueryRouter,
    n: i64,
    history: &mut HistoryRecorder,
) {
    exec(router, "CREATE TABLE reg (kid INT, val INT)");
    for k in 1..=n {
        exec(router, &format!("INSERT INTO reg VALUES ({k}, 0)"));
        let wid = history.invoke(0, OpType::Write, k.to_string(), Value::Int(0));
        history.complete(wid, Value::None);
    }
}

fn setup_registers_with_concurrent_history(
    router: &query_router::QueryRouter,
    n: i64,
    history: &ConcurrentHistoryRecorder,
) {
    exec(router, "CREATE TABLE reg (kid INT, val INT)");
    for k in 1..=n {
        exec(router, &format!("INSERT INTO reg VALUES ({k}, 0)"));
        let wid = history.invoke(0, OpType::Write, k.to_string(), Value::Int(0));
        history.complete(wid, Value::None);
    }
}

// ---------------------------------------------------------------------------
// Helpers — TensorStore layer
// ---------------------------------------------------------------------------

fn make_val(v: i64) -> TensorData {
    let mut td = TensorData::new();
    td.set("v", TensorValue::Scalar(ScalarValue::Int(v)));
    td
}

fn read_register_store(store: &TensorStore, key: &str) -> Value {
    match store.get(key) {
        Ok(data) => match data.get("v") {
            Some(TensorValue::Scalar(ScalarValue::Int(v))) => Value::Int(*v),
            _ => Value::None,
        },
        Err(_) => Value::None,
    }
}

fn write_register_store(store: &TensorStore, key: &str, val: i64) {
    store.put(key, make_val(val)).expect("store put failed");
}

// ---------------------------------------------------------------------------
// Part 1: SQL through QueryRouter
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_sequential_single_key_linearizable() {
    let router = create_shared_router();
    let mut history = HistoryRecorder::new();
    setup_registers_with_history(&router, 1, &mut history);

    for i in 1..=10_i64 {
        let wid = history.invoke(0, OpType::Write, "1".to_string(), Value::Int(i));
        write_register_sql(&router, 1, i);
        history.complete(wid, Value::None);

        let rid = history.invoke(0, OpType::Read, "1".to_string(), Value::None);
        let val = read_register_sql(&router, 1);
        history.complete(rid, val);
    }

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(history.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "sequential ops should be linearizable: {result:?}"
    );
}

#[test]
fn test_e2e_sql_partitioned_keys_4_threads() {
    let router = Arc::new(create_shared_router());
    let keys_per_thread = 5;
    let total_keys = 4 * keys_per_thread;

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    setup_registers_with_concurrent_history(&router, total_keys, &history);

    let ops_per_thread = 40;

    std::thread::scope(|s| {
        for tid in 0..4u64 {
            let router = Arc::clone(&router);
            let history = Arc::clone(&history);
            s.spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(tid * 1000);
                let key_base = (tid as i64) * keys_per_thread + 1;
                for op_idx in 0..ops_per_thread {
                    let key = key_base + rng.gen_range(0..keys_per_thread);
                    let key_str = key.to_string();
                    let is_write = rng.gen_ratio(1, 2);
                    if is_write {
                        let val = (tid as i64) * 1000 + op_idx;
                        let oid = history.invoke(tid, OpType::Write, key_str, Value::Int(val));
                        write_register_sql(&router, key, val);
                        history.complete(oid, Value::None);
                    } else {
                        let oid = history.invoke(tid, OpType::Read, key_str, Value::None);
                        let val = read_register_sql(&router, key);
                        history.complete(oid, val);
                    }
                }
            });
        }
    });

    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all refs should be dropped"))
        .into_inner();
    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "partitioned SQL should be linearizable: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Part 2: Concurrent TensorStore through QueryRouter's shared store
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_concurrent_single_key_4_threads() {
    let router = create_shared_router();
    let store = Arc::new(get_store_from_router(&router));

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let ops_per_thread = 50;

    std::thread::scope(|s| {
        for tid in 0..4u64 {
            let store = Arc::clone(&store);
            let history = Arc::clone(&history);
            s.spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(tid * 1000);
                for op_idx in 0..ops_per_thread {
                    let is_write = rng.gen_ratio(3, 5);
                    if is_write {
                        let val = (tid as i64) * 1000 + op_idx;
                        let oid =
                            history.invoke(tid, OpType::Write, "r:1".to_string(), Value::Int(val));
                        write_register_store(&store, "r:1", val);
                        history.complete(oid, Value::None);
                    } else {
                        let oid = history.invoke(tid, OpType::Read, "r:1".to_string(), Value::None);
                        let val = read_register_store(&store, "r:1");
                        history.complete(oid, val);
                    }
                }
            });
        }
    });

    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all refs should be dropped"))
        .into_inner();
    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "concurrent single-key should be linearizable: {result:?}"
    );
}

#[test]
fn test_e2e_concurrent_multi_key_4_threads() {
    let router = create_shared_router();
    let store = Arc::new(get_store_from_router(&router));
    let num_keys = 4;

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let ops_per_thread = 40;

    std::thread::scope(|s| {
        for tid in 0..4u64 {
            let store = Arc::clone(&store);
            let history = Arc::clone(&history);
            s.spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(tid * 7919);
                for op_idx in 0..ops_per_thread {
                    let key = rng.gen_range(1..=num_keys);
                    let key_str = format!("r:{key}");
                    let is_write = rng.gen_ratio(1, 2);
                    if is_write {
                        let val = (tid as i64) * 1000 + op_idx;
                        let oid =
                            history.invoke(tid, OpType::Write, key_str.clone(), Value::Int(val));
                        write_register_store(&store, &key_str, val);
                        history.complete(oid, Value::None);
                    } else {
                        let oid = history.invoke(tid, OpType::Read, key_str.clone(), Value::None);
                        let val = read_register_store(&store, &key_str);
                        history.complete(oid, val);
                    }
                }
            });
        }
    });

    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all refs should be dropped"))
        .into_inner();
    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "concurrent multi-key should be linearizable: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Stress
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_high_contention_8_threads() {
    let router = create_shared_router();
    let store = Arc::new(get_store_from_router(&router));

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    // WGL checker complexity is exponential in concurrent overlap width.
    // 8 threads on 1 key creates up to 8 overlapping ops at any point,
    // so keep total ops modest to avoid search timeout.
    let ops_per_thread = 15;

    std::thread::scope(|s| {
        for tid in 0..8u64 {
            let store = Arc::clone(&store);
            let history = Arc::clone(&history);
            s.spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(tid * 3571);
                for op_idx in 0..ops_per_thread {
                    let is_write = rng.gen_ratio(1, 2);
                    if is_write {
                        let val = (tid as i64) * 1000 + op_idx;
                        let oid =
                            history.invoke(tid, OpType::Write, "r:1".to_string(), Value::Int(val));
                        write_register_store(&store, "r:1", val);
                        history.complete(oid, Value::None);
                    } else {
                        let oid = history.invoke(tid, OpType::Read, "r:1".to_string(), Value::None);
                        let val = read_register_store(&store, "r:1");
                        history.complete(oid, val);
                    }
                }
            });
        }
    });

    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all refs should be dropped"))
        .into_inner();
    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(60));
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "high-contention should be linearizable: {result:?}"
    );
}

#[test]
fn test_e2e_many_keys_low_contention() {
    let router = create_shared_router();
    let store = Arc::new(get_store_from_router(&router));
    let num_keys: i64 = 20;

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let ops_per_thread = 100;

    std::thread::scope(|s| {
        for tid in 0..4u64 {
            let store = Arc::clone(&store);
            let history = Arc::clone(&history);
            s.spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(tid * 6151);
                for op_idx in 0..ops_per_thread {
                    let key = rng.gen_range(1..=num_keys);
                    let key_str = format!("r:{key}");
                    let is_write = rng.gen_ratio(1, 2);
                    if is_write {
                        let val = (tid as i64) * 1000 + op_idx;
                        let oid =
                            history.invoke(tid, OpType::Write, key_str.clone(), Value::Int(val));
                        write_register_store(&store, &key_str, val);
                        history.complete(oid, Value::None);
                    } else {
                        let oid = history.invoke(tid, OpType::Read, key_str.clone(), Value::None);
                        let val = read_register_store(&store, &key_str);
                        history.complete(oid, val);
                    }
                }
            });
        }
    });

    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all refs should be dropped"))
        .into_inner();
    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "many-keys low-contention should be linearizable: {result:?}"
    );
}

#[test]
fn test_e2e_write_heavy_workload() {
    let router = create_shared_router();
    let store = Arc::new(get_store_from_router(&router));
    let num_keys: i64 = 4;

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let ops_per_thread = 60;

    std::thread::scope(|s| {
        for tid in 0..4u64 {
            let store = Arc::clone(&store);
            let history = Arc::clone(&history);
            s.spawn(move || {
                let mut rng = ChaCha8Rng::seed_from_u64(tid * 4219);
                for op_idx in 0..ops_per_thread {
                    let key = rng.gen_range(1..=num_keys);
                    let key_str = format!("r:{key}");
                    let is_write = rng.gen_ratio(9, 10); // 90% writes
                    if is_write {
                        let val = (tid as i64) * 1000 + op_idx;
                        let oid =
                            history.invoke(tid, OpType::Write, key_str.clone(), Value::Int(val));
                        write_register_store(&store, &key_str, val);
                        history.complete(oid, Value::None);
                    } else {
                        let oid = history.invoke(tid, OpType::Read, key_str.clone(), Value::None);
                        let val = read_register_store(&store, &key_str);
                        history.complete(oid, val);
                    }
                }
            });
        }
    });

    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all refs should be dropped"))
        .into_inner();
    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "write-heavy should be linearizable: {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

#[test]
fn test_e2e_read_uninitialized_key() {
    let router = create_shared_router();
    let store = Arc::new(get_store_from_router(&router));

    let history = Arc::new(ConcurrentHistoryRecorder::new());

    std::thread::scope(|s| {
        for tid in 0..4u64 {
            let store = Arc::clone(&store);
            let history = Arc::clone(&history);
            s.spawn(move || {
                for _ in 0..5 {
                    let oid = history.invoke(tid, OpType::Read, "r:999".to_string(), Value::None);
                    let val = read_register_store(&store, "r:999");
                    assert_eq!(val, Value::None, "uninitialized key should return None");
                    history.complete(oid, val);
                }
            });
        }
    });

    let recorder = Arc::try_unwrap(history)
        .unwrap_or_else(|_| panic!("all refs should be dropped"))
        .into_inner();
    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(recorder.operations());
    assert!(
        matches!(result, LinearizabilityResult::Ok),
        "uninitialized reads should be linearizable: {result:?}"
    );
}

#[test]
fn test_e2e_deterministic_seed_reproducible() {
    // Returns per-client operation sequences (deterministic given same RNG seed).
    // Thread interleaving is non-deterministic, so we compare per-client order
    // rather than global order.
    let num_threads = 4u64;
    let run = |seed_base: u64| -> Vec<Vec<(OpType, String, Value)>> {
        let router = create_shared_router();
        let store = Arc::new(get_store_from_router(&router));
        let num_keys: i64 = 4;

        let history = Arc::new(ConcurrentHistoryRecorder::new());
        let ops_per_thread = 30;

        std::thread::scope(|s| {
            for tid in 0..num_threads {
                let store = Arc::clone(&store);
                let history = Arc::clone(&history);
                s.spawn(move || {
                    let mut rng = ChaCha8Rng::seed_from_u64(seed_base + tid);
                    for op_idx in 0..ops_per_thread {
                        let key = rng.gen_range(1..=num_keys);
                        let key_str = format!("r:{key}");
                        let is_write = rng.gen_ratio(1, 2);
                        if is_write {
                            let val = (tid as i64) * 1000 + op_idx;
                            let oid = history.invoke(
                                tid,
                                OpType::Write,
                                key_str.clone(),
                                Value::Int(val),
                            );
                            write_register_store(&store, &key_str, val);
                            history.complete(oid, Value::None);
                        } else {
                            let oid =
                                history.invoke(tid, OpType::Read, key_str.clone(), Value::None);
                            let val = read_register_store(&store, &key_str);
                            history.complete(oid, val);
                        }
                    }
                });
            }
        });

        let recorder = Arc::try_unwrap(history)
            .unwrap_or_else(|_| panic!("all refs should be dropped"))
            .into_inner();
        let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
        let result = checker.check(recorder.operations());
        assert!(
            matches!(result, LinearizabilityResult::Ok),
            "deterministic run should be linearizable: {result:?}"
        );

        // Group operations by client_id, preserving per-client order
        let ops = recorder.operations();
        let mut per_client: Vec<Vec<(OpType, String, Value)>> =
            (0..num_threads).map(|_| Vec::new()).collect();
        for op in ops {
            per_client[op.client_id as usize].push((
                op.op_type.clone(),
                op.key.clone(),
                op.input.clone(),
            ));
        }
        per_client
    };

    let seed = 42_u64;
    let run1 = run(seed);
    let run2 = run(seed);

    // Each client's operation sequence (op_type, key, input) must match
    // because the per-thread RNG seed is deterministic.
    for tid in 0..num_threads as usize {
        assert_eq!(
            run1[tid].len(),
            run2[tid].len(),
            "client {tid} op count mismatch"
        );
        for (i, (a, b)) in run1[tid].iter().zip(run2[tid].iter()).enumerate() {
            assert_eq!(a.0, b.0, "client {tid} op type mismatch at {i}");
            assert_eq!(a.1, b.1, "client {tid} key mismatch at {i}");
            assert_eq!(a.2, b.2, "client {tid} input mismatch at {i}");
        }
    }
}
