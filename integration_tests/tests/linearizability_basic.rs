// SPDX-License-Identifier: MIT OR Apache-2.0
//! Basic linearizability tests using the WGL checker.
//!
//! Verifies that the linearizability checker correctly identifies both
//! valid and invalid operation histories without any fault injection.

use std::time::{Duration, Instant};

use integration_tests::linearizability::{
    HistoryRecorder, LinearizabilityChecker, LinearizabilityResult, OpType, RegisterModel, Value,
};

fn make_op(
    id: u64,
    op_type: OpType,
    key: &str,
    input: Value,
    output: Value,
    invoke_offset_ms: u64,
    duration_ms: u64,
    client_id: u64,
) -> integration_tests::linearizability::Operation {
    let base = Instant::now();
    let invoke_time = base + Duration::from_millis(invoke_offset_ms);
    let complete_time = invoke_time + Duration::from_millis(duration_ms);
    integration_tests::linearizability::Operation {
        id,
        op_type,
        key: key.to_string(),
        input,
        output: Some(output),
        invoke_time,
        complete_time: Some(complete_time),
        client_id,
    }
}

#[test]
fn test_single_key_write_then_read() {
    let ops = vec![
        make_op(0, OpType::Write, "k", Value::Int(1), Value::None, 0, 10, 1),
        make_op(1, OpType::Read, "k", Value::None, Value::Int(1), 20, 10, 2),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_multiple_keys_independent() {
    let ops = vec![
        make_op(0, OpType::Write, "a", Value::Int(1), Value::None, 0, 10, 1),
        make_op(1, OpType::Write, "b", Value::Int(2), Value::None, 0, 10, 2),
        make_op(2, OpType::Read, "a", Value::None, Value::Int(1), 20, 10, 1),
        make_op(3, OpType::Read, "b", Value::None, Value::Int(2), 20, 10, 2),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_overwrite_and_read_latest() {
    let ops = vec![
        make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 10, 1),
        make_op(1, OpType::Write, "x", Value::Int(2), Value::None, 20, 10, 1),
        make_op(2, OpType::Read, "x", Value::None, Value::Int(2), 40, 10, 2),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_stale_read_is_violation() {
    // Read returns old value after overwrite completes
    let ops = vec![
        make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 10, 1),
        make_op(1, OpType::Write, "x", Value::Int(2), Value::None, 20, 10, 1),
        make_op(2, OpType::Read, "x", Value::None, Value::Int(1), 40, 10, 2),
    ];
    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(5));
    let result = checker.check(&ops);
    assert!(
        matches!(result, LinearizabilityResult::Violation(_)),
        "expected violation, got {result:?}"
    );
}

#[test]
fn test_concurrent_writes_either_order_valid() {
    // Two concurrent writes -- read can return either final value depending
    // on linearization order.
    let ops = vec![
        make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 50, 1),
        make_op(1, OpType::Write, "x", Value::Int(2), Value::None, 10, 50, 2),
        make_op(2, OpType::Read, "x", Value::None, Value::Int(2), 100, 10, 3),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_read_absent_key_returns_none() {
    let ops = vec![make_op(
        0,
        OpType::Read,
        "missing",
        Value::None,
        Value::None,
        0,
        10,
        1,
    )];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_cas_success_linearizable() {
    let ops = vec![
        make_op(0, OpType::Write, "x", Value::Int(10), Value::None, 0, 10, 1),
        make_op(
            1,
            OpType::Cas,
            "x",
            Value::Int(10),
            Value::Int(10),
            20,
            10,
            2,
        ),
        make_op(2, OpType::Read, "x", Value::None, Value::Int(11), 40, 10, 3),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_cas_failure_linearizable() {
    // CAS with wrong expected value returns current value, no state change
    let ops = vec![
        make_op(0, OpType::Write, "x", Value::Int(10), Value::None, 0, 10, 1),
        make_op(
            1,
            OpType::Cas,
            "x",
            Value::Int(5),
            Value::Int(10),
            20,
            10,
            2,
        ),
        make_op(2, OpType::Read, "x", Value::None, Value::Int(10), 40, 10, 3),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_recorder_invoke_complete_flow() {
    let mut recorder = HistoryRecorder::new();

    let w = recorder.invoke(1, OpType::Write, "key".to_string(), Value::Int(42));
    recorder.complete(w, Value::None);

    let r = recorder.invoke(2, OpType::Read, "key".to_string(), Value::None);
    recorder.complete(r, Value::Int(42));

    assert_eq!(recorder.len(), 2);
    assert_eq!(recorder.completed_operations().len(), 2);

    let checker = LinearizabilityChecker::new(RegisterModel);
    let result = checker.check(recorder.operations());
    assert_eq!(result, LinearizabilityResult::Ok);
}

#[test]
fn test_recorder_incomplete_ops_excluded() {
    let mut recorder = HistoryRecorder::new();

    let w = recorder.invoke(1, OpType::Write, "k".to_string(), Value::Int(1));
    recorder.complete(w, Value::None);

    // Invoke a read but never complete it
    let _r = recorder.invoke(2, OpType::Read, "k".to_string(), Value::None);

    assert_eq!(recorder.len(), 2);
    assert_eq!(recorder.completed_operations().len(), 1);
}

#[test]
fn test_string_values_linearizable() {
    let ops = vec![
        make_op(
            0,
            OpType::Write,
            "name",
            Value::Str("alice".to_string()),
            Value::None,
            0,
            10,
            1,
        ),
        make_op(
            1,
            OpType::Read,
            "name",
            Value::None,
            Value::Str("alice".to_string()),
            20,
            10,
            2,
        ),
    ];
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}

#[test]
fn test_many_sequential_writes_read_latest() {
    let mut ops = Vec::new();
    for i in 0..10 {
        ops.push(make_op(
            i,
            OpType::Write,
            "counter",
            Value::Int(i as i64),
            Value::None,
            u64::from(i) * 20,
            10,
            1,
        ));
    }
    // Read should return the last written value
    ops.push(make_op(
        10,
        OpType::Read,
        "counter",
        Value::None,
        Value::Int(9),
        200,
        10,
        2,
    ));
    let checker = LinearizabilityChecker::new(RegisterModel);
    assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
}
