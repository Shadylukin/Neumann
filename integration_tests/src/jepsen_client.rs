// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Jepsen client for multi-process tests.
//!
//! Wraps `NeumannClient` gRPC connections with leader discovery, retry logic,
//! and linearizability history recording.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use neumann_client::NeumannClient;

use crate::linearizability::{ConcurrentHistoryRecorder, OpType, Value};

/// Client for driving Jepsen workloads against a multi-process cluster.
///
/// Maintains connections to all nodes and routes operations to the leader.
/// Records all operations (invoke + complete) in the shared history recorder
/// for post-hoc linearizability checking.
pub struct JepsenClient {
    addrs: Vec<String>,
    leader_idx: AtomicUsize,
    history: Arc<ConcurrentHistoryRecorder>,
    client_id: u64,
    timeout: Duration,
}

impl JepsenClient {
    /// Create a new client connected to the given node addresses.
    #[must_use]
    pub const fn new(
        addrs: Vec<String>,
        history: Arc<ConcurrentHistoryRecorder>,
        client_id: u64,
    ) -> Self {
        Self {
            addrs,
            leader_idx: AtomicUsize::new(0),
            history,
            client_id,
            timeout: Duration::from_secs(5),
        }
    }

    /// Write a key-value pair. Records the operation in history.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation's outcome is indeterminate (timeout).
    pub async fn write(&self, key: &str, value: i64) -> Result<bool, String> {
        let op_id = self.history.invoke(
            self.client_id,
            OpType::Write,
            key.to_string(),
            Value::Int(value),
        );

        let query = format!("INSERT jepsen_register key='{key}', val={value}");

        match self.execute_with_retry(&query).await {
            Ok(_) => {
                self.history.complete(op_id, Value::None);
                Ok(true)
            },
            Err(e) if is_definite_failure(&e) => {
                self.history.complete(op_id, Value::None);
                Ok(false)
            },
            Err(e) => Err(e),
        }
    }

    /// Read a key. Records the operation in history.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation's outcome is indeterminate.
    pub async fn read(&self, key: &str) -> Result<Option<i64>, String> {
        let op_id = self
            .history
            .invoke(self.client_id, OpType::Read, key.to_string(), Value::None);

        let query = format!("SELECT jepsen_register WHERE key = '{key}'");

        match self.execute_with_retry(&query).await {
            Ok(result) => {
                let val = extract_value_from_result(&result);
                let history_val = val.map_or(Value::None, Value::Int);
                self.history.complete(op_id, history_val);
                Ok(val)
            },
            Err(e) if is_definite_failure(&e) => {
                self.history.complete(op_id, Value::None);
                Ok(None)
            },
            Err(e) => Err(e),
        }
    }

    /// Compare-and-swap operation. Records the operation in history.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the operation's outcome is indeterminate.
    pub async fn cas(&self, key: &str, expected: i64, new_val: i64) -> Result<bool, String> {
        let op_id = self.history.invoke(
            self.client_id,
            OpType::Cas,
            key.to_string(),
            Value::Int(expected),
        );

        let query = format!(
            "UPDATE jepsen_register SET val={new_val} WHERE key = '{key}' AND val = {expected}"
        );

        match self.execute_with_retry(&query).await {
            Ok(result) => {
                let success = !result.has_error();
                let output = if success {
                    Value::Int(expected)
                } else {
                    Value::None
                };
                self.history.complete(op_id, output);
                Ok(success)
            },
            Err(e) if is_definite_failure(&e) => {
                self.history.complete(op_id, Value::None);
                Ok(false)
            },
            Err(e) => Err(e),
        }
    }

    /// Execute a query with leader retry logic.
    async fn execute_with_retry(
        &self,
        query: &str,
    ) -> Result<neumann_client::RemoteQueryResult, String> {
        let mut last_err = String::new();

        let leader = self.leader_idx.load(Ordering::Relaxed);
        let n = self.addrs.len();

        for attempt in 0..n {
            let idx = (leader + attempt) % n;
            let addr = &self.addrs[idx];

            let client = match tokio::time::timeout(
                self.timeout,
                NeumannClient::connect(addr).build(),
            )
            .await
            {
                Ok(Ok(c)) => c,
                Ok(Err(e)) => {
                    last_err = format!("connect to {addr}: {e}");
                    continue;
                },
                Err(_) => {
                    last_err = format!("connect to {addr}: timeout");
                    continue;
                },
            };

            match tokio::time::timeout(self.timeout, client.execute(query)).await {
                Ok(Ok(result)) => {
                    self.leader_idx.store(idx, Ordering::Relaxed);
                    return Ok(result);
                },
                Ok(Err(e)) => {
                    last_err = e.to_string();
                },
                Err(_) => {
                    last_err = format!("query to {addr}: timeout");
                },
            }
        }

        Err(last_err)
    }
}

/// Check if an error represents a definite failure (not indeterminate).
fn is_definite_failure(err: &str) -> bool {
    err.contains("not leader")
        || err.contains("NotLeader")
        || err.contains("connection refused")
        || err.contains("table not found")
        || err.contains("Table not found")
}

/// Extract an integer value from a query result.
fn extract_value_from_result(result: &neumann_client::RemoteQueryResult) -> Option<i64> {
    if result.has_error() {
        return None;
    }
    let rows = result.rows()?;
    let row = rows.first()?;
    for col in &row.values {
        if col.name == "val" {
            let value = col.value.as_ref()?;
            return match value.kind {
                Some(neumann_client::proto::value::Kind::IntValue(v)) => Some(v),
                Some(neumann_client::proto::value::Kind::FloatValue(v)) =>
                {
                    #[allow(clippy::cast_possible_truncation)]
                    Some(v as i64)
                },
                Some(neumann_client::proto::value::Kind::StringValue(ref s)) => s.parse().ok(),
                _ => None,
            };
        }
    }
    None
}
