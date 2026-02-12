// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Linearizability checking module using the Wing-Gong-Liu (WGL) algorithm.
//!
//! Verifies that concurrent operations on a distributed key-value store appear
//! to execute atomically. The checker explores all possible linearization
//! orderings of concurrent operations, pruning branches whose outputs do not
//! match a sequential specification model.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Type of operation performed on the key-value store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpType {
    Read,
    Write,
    /// Compare-and-swap: atomically update if the current value matches the
    /// expected value.
    Cas,
}

/// A value in the key-value store.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    None,
    Int(i64),
    Str(String),
}

/// A single operation in the history, bounded by its invocation and completion
/// timestamps to define the real-time interval during which the operation
/// could have taken effect.
#[derive(Debug, Clone)]
pub struct Operation {
    pub id: u64,
    pub op_type: OpType,
    pub key: String,
    pub input: Value,
    pub output: Option<Value>,
    pub invoke_time: Instant,
    pub complete_time: Option<Instant>,
    pub client_id: u64,
}

/// Records operation history for linearizability checking.
///
/// Operations are first invoked (recording the start time), then completed
/// (recording the output and end time). Only completed operations are
/// candidates for linearizability verification.
pub struct HistoryRecorder {
    operations: Vec<Operation>,
    next_id: u64,
}

impl HistoryRecorder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            operations: Vec::new(),
            next_id: 0,
        }
    }

    /// Record the invocation of an operation. Returns the operation id.
    pub fn invoke(&mut self, client_id: u64, op_type: OpType, key: String, input: Value) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.operations.push(Operation {
            id,
            op_type,
            key,
            input,
            output: None,
            invoke_time: Instant::now(),
            complete_time: None,
            client_id,
        });
        id
    }

    /// Record the completion of an operation, attaching its output value.
    pub fn complete(&mut self, op_id: u64, output: Value) {
        for op in &mut self.operations {
            if op.id == op_id {
                op.output = Some(output);
                op.complete_time = Some(Instant::now());
                return;
            }
        }
    }

    #[must_use]
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// Return only operations that have been completed (have both output and
    /// completion time).
    #[must_use]
    pub fn completed_operations(&self) -> Vec<&Operation> {
        self.operations
            .iter()
            .filter(|op| op.complete_time.is_some() && op.output.is_some())
            .collect()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.operations.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

impl Default for HistoryRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe wrapper around `HistoryRecorder` for concurrent operation
/// recording from multiple tasks.
pub struct ConcurrentHistoryRecorder {
    inner: std::sync::Mutex<HistoryRecorder>,
}

impl ConcurrentHistoryRecorder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(HistoryRecorder::new()),
        }
    }

    /// Record the invocation of an operation. Returns the operation id.
    pub fn invoke(&self, client_id: u64, op_type: OpType, key: String, input: Value) -> u64 {
        self.inner
            .lock()
            .expect("recorder lock poisoned")
            .invoke(client_id, op_type, key, input)
    }

    /// Record the completion of an operation, attaching its output value.
    pub fn complete(&self, op_id: u64, output: Value) {
        self.inner
            .lock()
            .expect("recorder lock poisoned")
            .complete(op_id, output);
    }

    /// Consume the wrapper and return the inner `HistoryRecorder`.
    pub fn into_inner(self) -> HistoryRecorder {
        self.inner.into_inner().expect("recorder lock poisoned")
    }
}

impl Default for ConcurrentHistoryRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of linearizability checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearizabilityResult {
    /// The history is linearizable -- a valid linearization ordering exists.
    Ok,
    /// The history is not linearizable, with an explanation of why.
    Violation(String),
    /// The check was inconclusive, for example due to a timeout.
    Unknown(String),
}

/// Sequential specification for a key-value register.
///
/// Defines the valid behavior of the system when operations are applied one at
/// a time in some sequential order.
pub trait SequentialModel: Send + Sync {
    type State: Clone;

    /// Return the initial state of the model.
    fn init(&self) -> Self::State;

    /// Apply an operation to the model state, returning the new state and the
    /// expected output value.
    fn apply(&self, state: &Self::State, op: &Operation) -> (Self::State, Value);
}

/// Simple read/write/CAS register model backed by an in-memory hash map.
///
/// - **Read**: returns the current value for the key, or `Value::None` if
///   absent.
/// - **Write**: sets the key to the input value; returns `Value::None`.
/// - **CAS**: if the current value equals the input, updates to
///   `Value::Int(input_value + 1)` and returns the old value. Otherwise
///   returns the current value without modifying state.
pub struct RegisterModel;

impl SequentialModel for RegisterModel {
    type State = HashMap<String, Value>;

    fn init(&self) -> Self::State {
        HashMap::new()
    }

    fn apply(&self, state: &Self::State, op: &Operation) -> (Self::State, Value) {
        let mut next = state.clone();
        let current = state.get(&op.key).cloned().unwrap_or(Value::None);

        match op.op_type {
            OpType::Read => (next, current),
            OpType::Write => {
                next.insert(op.key.clone(), op.input.clone());
                (next, Value::None)
            },
            OpType::Cas => {
                if current == op.input {
                    let new_value = match &op.input {
                        Value::Int(v) => Value::Int(v + 1),
                        other => other.clone(),
                    };
                    next.insert(op.key.clone(), new_value);
                }
                (next, current)
            },
        }
    }
}

/// Sequential model for Raft log proposals.
///
/// Proposals are modeled as writes to a single "key" register. Successful
/// proposals return `Value::Int(1)` and failed proposals return `Value::None`.
/// This matches the DST harness convention (dst.rs:373-383).
pub struct RaftLogModel;

impl SequentialModel for RaftLogModel {
    type State = HashMap<String, Value>;

    fn init(&self) -> Self::State {
        HashMap::new()
    }

    fn apply(&self, state: &Self::State, op: &Operation) -> (Self::State, Value) {
        match op.op_type {
            OpType::Write => {
                let mut s = state.clone();
                s.insert(op.key.clone(), op.input.clone());
                (s, Value::Int(1)) // success
            },
            OpType::Read => {
                let val = state.get(&op.key).cloned().unwrap_or(Value::None);
                (state.clone(), val)
            },
            OpType::Cas => (state.clone(), Value::None),
        }
    }
}

/// Wing-Gong-Liu linearizability checker.
///
/// Explores all valid linearization orderings of a concurrent history using
/// recursive backtracking search. At each step the checker picks a candidate
/// operation that could be linearized next (its invocation precedes, or is
/// concurrent with, uncompleted operations), applies it to the sequential
/// model, and checks whether the observed output matches. If so the search
/// recurses; if not it backtracks.
pub struct LinearizabilityChecker<M: SequentialModel> {
    model: M,
    timeout: Duration,
}

impl<M: SequentialModel> LinearizabilityChecker<M> {
    #[must_use]
    pub const fn new(model: M) -> Self {
        Self {
            model,
            timeout: Duration::from_secs(30),
        }
    }

    #[must_use]
    pub const fn with_timeout(model: M, timeout: Duration) -> Self {
        Self { model, timeout }
    }

    /// Check linearizability by partitioning the history by key.
    ///
    /// For independent keys (where each key's state is independent of others),
    /// a history is linearizable iff each key's projection is independently
    /// linearizable. This is dramatically faster than checking the full history
    /// because the WGL search space is exponential in the number of concurrent
    /// operations per key, not the total number of operations.
    #[must_use]
    pub fn check_per_key(&self, history: &[Operation]) -> LinearizabilityResult {
        let completed: Vec<&Operation> = history
            .iter()
            .filter(|op| op.complete_time.is_some() && op.output.is_some())
            .collect();

        if completed.is_empty() {
            return LinearizabilityResult::Ok;
        }

        // Group operations by key
        let mut by_key: HashMap<&str, Vec<&Operation>> = HashMap::new();
        for op in &completed {
            by_key.entry(&op.key).or_default().push(op);
        }

        for (key, ops) in &by_key {
            let start = Instant::now();
            let init_state = self.model.init();
            let remaining: Vec<usize> = (0..ops.len()).collect();

            match self.search(ops, &init_state, &remaining, start) {
                SearchResult::Found => {},
                SearchResult::NotFound => {
                    return LinearizabilityResult::Violation(format!(
                        "no valid linearization ordering exists for key '{key}'"
                    ));
                },
                SearchResult::Timeout => {
                    return LinearizabilityResult::Unknown(format!(
                        "search timed out for key '{key}' ({} ops)",
                        ops.len()
                    ));
                },
            }
        }

        LinearizabilityResult::Ok
    }

    /// Check whether the given history is linearizable with respect to the
    /// sequential model.
    #[must_use]
    pub fn check(&self, history: &[Operation]) -> LinearizabilityResult {
        let completed: Vec<&Operation> = history
            .iter()
            .filter(|op| op.complete_time.is_some() && op.output.is_some())
            .collect();

        if completed.is_empty() {
            return LinearizabilityResult::Ok;
        }

        let start = Instant::now();
        let init_state = self.model.init();
        let remaining: Vec<usize> = (0..completed.len()).collect();

        match self.search(&completed, &init_state, &remaining, start) {
            SearchResult::Found => LinearizabilityResult::Ok,
            SearchResult::NotFound => LinearizabilityResult::Violation(
                "no valid linearization ordering exists".to_string(),
            ),
            SearchResult::Timeout => {
                LinearizabilityResult::Unknown("search timed out before completing".to_string())
            },
        }
    }

    fn search(
        &self,
        ops: &[&Operation],
        state: &M::State,
        remaining: &[usize],
        start: Instant,
    ) -> SearchResult {
        if remaining.is_empty() {
            return SearchResult::Found;
        }

        if start.elapsed() > self.timeout {
            return SearchResult::Timeout;
        }

        let candidates = find_candidates(ops, remaining);

        for &idx in &candidates {
            let op = ops[idx];
            let (next_state, expected_output) = self.model.apply(state, op);

            if let Some(ref actual_output) = op.output {
                if *actual_output != expected_output {
                    continue;
                }
            }

            let next_remaining: Vec<usize> =
                remaining.iter().copied().filter(|&i| i != idx).collect();

            match self.search(ops, &next_state, &next_remaining, start) {
                SearchResult::Found => return SearchResult::Found,
                SearchResult::Timeout => return SearchResult::Timeout,
                SearchResult::NotFound => {},
            }
        }

        SearchResult::NotFound
    }
}

/// Identify operations that could be linearized next.
///
/// An operation is a candidate if no other remaining operation completed
/// strictly before it was invoked. In other words, a candidate is one
/// whose invocation time is not strictly after every other remaining
/// operation's completion time (i.e. it is concurrent with or precedes
/// them).
fn find_candidates(ops: &[&Operation], remaining: &[usize]) -> Vec<usize> {
    let mut candidates = Vec::new();

    for &idx in remaining {
        let op = ops[idx];
        let can_go_next = remaining.iter().all(|&other_idx| {
            if other_idx == idx {
                return true;
            }
            let other = ops[other_idx];
            // `op` can be next if no other operation completed strictly
            // before `op` was invoked (meaning `op` is not forced to come
            // after `other`).
            other.complete_time.is_none_or(|ct| op.invoke_time <= ct)
        });
        if can_go_next {
            candidates.push(idx);
        }
    }

    candidates
}

enum SearchResult {
    Found,
    NotFound,
    Timeout,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[allow(clippy::too_many_arguments)]
    fn make_op(
        id: u64,
        op_type: OpType,
        key: &str,
        input: Value,
        output: Value,
        invoke_offset_ms: u64,
        duration_ms: u64,
        client_id: u64,
    ) -> Operation {
        let base = Instant::now();
        // Stagger invocations by sleeping briefly so Instant values differ.
        // For deterministic tests we rely on the ordering guarantees rather
        // than exact timing.
        let invoke_time = base + Duration::from_millis(invoke_offset_ms);
        let complete_time = invoke_time + Duration::from_millis(duration_ms);
        Operation {
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
    fn test_empty_history_is_linearizable() {
        let checker = LinearizabilityChecker::new(RegisterModel);
        let history: Vec<Operation> = vec![];
        assert_eq!(checker.check(&history), LinearizabilityResult::Ok);
    }

    #[test]
    fn test_single_write_read_linearizable() {
        // Write x=1, then read x -> 1
        let ops = vec![
            make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 10, 1),
            make_op(1, OpType::Read, "x", Value::None, Value::Int(1), 20, 10, 2),
        ];
        let checker = LinearizabilityChecker::new(RegisterModel);
        assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
    }

    #[test]
    fn test_sequential_operations_linearizable() {
        // Sequential: write x=1, write x=2, read x -> 2
        let ops = vec![
            make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 10, 1),
            make_op(1, OpType::Write, "x", Value::Int(2), Value::None, 20, 10, 1),
            make_op(2, OpType::Read, "x", Value::None, Value::Int(2), 40, 10, 2),
        ];
        let checker = LinearizabilityChecker::new(RegisterModel);
        assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
    }

    #[test]
    fn test_concurrent_reads_linearizable() {
        // Write x=1, then two concurrent reads both returning 1.
        let ops = vec![
            make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 10, 1),
            make_op(1, OpType::Read, "x", Value::None, Value::Int(1), 20, 50, 2),
            make_op(2, OpType::Read, "x", Value::None, Value::Int(1), 25, 40, 3),
        ];
        let checker = LinearizabilityChecker::new(RegisterModel);
        assert_eq!(checker.check(&ops), LinearizabilityResult::Ok);
    }

    #[test]
    fn test_stale_read_not_linearizable() {
        // Write x=1, then write x=2, then read x -> 1 (stale).
        // The read starts after both writes complete, so the only valid
        // linearization must place both writes before the read, yielding 2.
        let ops = vec![
            make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 10, 1),
            make_op(1, OpType::Write, "x", Value::Int(2), Value::None, 20, 10, 1),
            make_op(2, OpType::Read, "x", Value::None, Value::Int(1), 40, 10, 2),
        ];
        let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(5));
        let result = checker.check(&ops);
        assert!(
            matches!(result, LinearizabilityResult::Violation(_)),
            "expected violation but got {:?}",
            result
        );
    }

    #[test]
    fn test_history_recorder_basic() {
        let mut recorder = HistoryRecorder::new();
        assert!(recorder.is_empty());
        assert_eq!(recorder.len(), 0);

        let id1 = recorder.invoke(1, OpType::Write, "x".to_string(), Value::Int(42));
        assert_eq!(id1, 0);
        assert_eq!(recorder.len(), 1);
        assert!(!recorder.is_empty());

        let id2 = recorder.invoke(2, OpType::Read, "x".to_string(), Value::None);
        assert_eq!(id2, 1);
        assert_eq!(recorder.len(), 2);

        // Neither operation is completed yet.
        assert!(recorder.completed_operations().is_empty());
    }

    #[test]
    fn test_history_recorder_complete() {
        let mut recorder = HistoryRecorder::new();
        let id1 = recorder.invoke(1, OpType::Write, "x".to_string(), Value::Int(1));
        let id2 = recorder.invoke(2, OpType::Read, "x".to_string(), Value::None);

        recorder.complete(id1, Value::None);
        assert_eq!(recorder.completed_operations().len(), 1);

        recorder.complete(id2, Value::Int(1));
        assert_eq!(recorder.completed_operations().len(), 2);

        let ops = recorder.operations();
        assert_eq!(ops[0].output, Some(Value::None));
        assert!(ops[0].complete_time.is_some());
        assert_eq!(ops[1].output, Some(Value::Int(1)));
        assert!(ops[1].complete_time.is_some());
    }

    #[test]
    fn test_register_model_read_write() {
        let model = RegisterModel;
        let state = model.init();
        assert!(state.is_empty());

        // Write x=5
        let write_op = make_op(0, OpType::Write, "x", Value::Int(5), Value::None, 0, 10, 1);
        let (state, output) = model.apply(&state, &write_op);
        assert_eq!(output, Value::None);
        assert_eq!(state.get("x"), Some(&Value::Int(5)));

        // Read x -> 5
        let read_op = make_op(1, OpType::Read, "x", Value::None, Value::Int(5), 20, 10, 2);
        let (state2, output) = model.apply(&state, &read_op);
        assert_eq!(output, Value::Int(5));
        // State unchanged after read.
        assert_eq!(state2, state);

        // Read absent key -> None
        let read_absent = make_op(2, OpType::Read, "y", Value::None, Value::None, 30, 10, 2);
        let (_state3, output) = model.apply(&state, &read_absent);
        assert_eq!(output, Value::None);
    }

    #[test]
    fn test_register_model_cas() {
        let model = RegisterModel;
        let state = model.init();

        // Write x=10
        let write_op = make_op(0, OpType::Write, "x", Value::Int(10), Value::None, 0, 10, 1);
        let (state, _) = model.apply(&state, &write_op);

        // CAS x: expect 10, should succeed -> old value 10, new value 11
        let cas_op = make_op(
            1,
            OpType::Cas,
            "x",
            Value::Int(10),
            Value::Int(10),
            20,
            10,
            2,
        );
        let (state, output) = model.apply(&state, &cas_op);
        assert_eq!(output, Value::Int(10));
        assert_eq!(state.get("x"), Some(&Value::Int(11)));

        // CAS x: expect 5, should fail (current is 11) -> returns 11
        let cas_fail = make_op(
            2,
            OpType::Cas,
            "x",
            Value::Int(5),
            Value::Int(11),
            30,
            10,
            3,
        );
        let (state, output) = model.apply(&state, &cas_fail);
        assert_eq!(output, Value::Int(11));
        // State unchanged.
        assert_eq!(state.get("x"), Some(&Value::Int(11)));
    }

    #[test]
    fn test_linearizability_result_debug() {
        let ok = LinearizabilityResult::Ok;
        let violation = LinearizabilityResult::Violation("stale read".to_string());
        let unknown = LinearizabilityResult::Unknown("timed out".to_string());

        let ok_debug = format!("{:?}", ok);
        let violation_debug = format!("{:?}", violation);
        let unknown_debug = format!("{:?}", unknown);

        assert!(ok_debug.contains("Ok"));
        assert!(violation_debug.contains("stale read"));
        assert!(unknown_debug.contains("timed out"));

        // Verify equality
        assert_eq!(ok, LinearizabilityResult::Ok);
        assert_ne!(ok, violation);
        assert_ne!(violation, unknown);
    }

    #[test]
    fn test_checker_no_silent_fallback() {
        // Craft a history where timing forces a violation that the old fallback
        // would have masked. Write x=1 (completes), write x=2 (completes after
        // first), read x -> 1 (starts after both writes complete).
        // Without the fallback, the checker correctly returns Violation.
        let ops = vec![
            make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 10, 1),
            make_op(1, OpType::Write, "x", Value::Int(2), Value::None, 20, 10, 1),
            make_op(2, OpType::Read, "x", Value::None, Value::Int(1), 40, 10, 2),
        ];
        let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(5));
        let result = checker.check(&ops);
        assert!(
            matches!(result, LinearizabilityResult::Violation(_)),
            "stale read after sequential writes must be detected as violation, got {result:?}"
        );
    }

    #[test]
    fn test_checker_concurrent_ops_linearizable() {
        // Overlapping concurrent ops that ARE linearizable: two writes overlap,
        // then a concurrent read returns one valid value. The checker must not
        // produce false negatives after the fallback removal.
        //
        // Timeline:
        //   W(x=1): [0ms, 50ms]
        //   W(x=2): [10ms, 40ms]   (overlaps with W(x=1))
        //   R(x):   [20ms, 60ms]   (overlaps with both writes)
        //
        // Valid linearization: W(x=1) -> W(x=2) -> R(x)=2
        // or: W(x=2) -> W(x=1) -> R(x)=1, etc.
        let ops = vec![
            make_op(0, OpType::Write, "x", Value::Int(1), Value::None, 0, 50, 1),
            make_op(1, OpType::Write, "x", Value::Int(2), Value::None, 10, 30, 2),
            make_op(2, OpType::Read, "x", Value::None, Value::Int(2), 20, 40, 3),
        ];
        let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(5));
        let result = checker.check(&ops);
        assert_eq!(
            result,
            LinearizabilityResult::Ok,
            "concurrent ops with valid linearization must pass: {result:?}"
        );
    }
}
