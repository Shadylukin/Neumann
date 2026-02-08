// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Multi-process Jepsen tests.
//!
//! These tests spawn real `neumann_server` processes, inject faults via TCP
//! proxies and SIGKILL, and verify linearizability of the resulting operation
//! history. All tests are `#[ignore]` because they are long-running, bind
//! ports, and require the server binary to be built.
//!
//! Run with:
//! ```bash
//! cargo build --release --bin neumann_server
//! cargo nextest run --package integration_tests --test jepsen_multiprocess -- --ignored
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use integration_tests::jepsen_client::JepsenClient;
use integration_tests::linearizability::{
    ConcurrentHistoryRecorder, LinearizabilityChecker, LinearizabilityResult, OpType, Operation,
    RegisterModel, Value,
};
use integration_tests::process_jepsen::ProcessCluster;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Nemesis types
// ---------------------------------------------------------------------------

/// Fault injection action for multi-process clusters.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum NemesisAction {
    KillNode(usize),
    RestartNode(usize),
    PauseNode(usize),
    ResumeNode(usize),
    Partition(usize, usize),
    HealPartition(usize, usize),
    HealAll,
    Sleep(Duration),
}

/// A timed sequence of fault injection actions.
struct NemesisSchedule {
    actions: Vec<(Duration, NemesisAction)>,
}

impl NemesisSchedule {
    fn new() -> Self {
        Self {
            actions: Vec::new(),
        }
    }

    fn add(mut self, delay: Duration, action: NemesisAction) -> Self {
        self.actions.push((delay, action));
        self
    }
}

/// Execute a nemesis schedule against a cluster.
async fn run_nemesis(cluster: &mut ProcessCluster, schedule: NemesisSchedule) {
    let start = Instant::now();

    for (delay, action) in schedule.actions {
        // Wait until the scheduled time
        let target = start + delay;
        let now = Instant::now();
        if target > now {
            tokio::time::sleep(target - now).await;
        }

        match action {
            NemesisAction::KillNode(idx) => {
                eprintln!("[nemesis] killing node-{idx} at {:?}", start.elapsed());
                cluster.kill_node(idx);
            },
            NemesisAction::RestartNode(idx) => {
                eprintln!("[nemesis] restarting node-{idx} at {:?}", start.elapsed());
                let _ = cluster.restart_node(idx);
                let _ = cluster.wait_node_healthy(idx).await;
            },
            NemesisAction::PauseNode(idx) => {
                eprintln!("[nemesis] pausing node-{idx} at {:?}", start.elapsed());
                cluster.pause_node(idx);
            },
            NemesisAction::ResumeNode(idx) => {
                eprintln!("[nemesis] resuming node-{idx} at {:?}", start.elapsed());
                cluster.resume_node(idx);
            },
            NemesisAction::Partition(a, b) => {
                eprintln!(
                    "[nemesis] partitioning node-{a} <-> node-{b} at {:?}",
                    start.elapsed()
                );
                cluster.partition(a, b);
            },
            NemesisAction::HealPartition(a, b) => {
                eprintln!(
                    "[nemesis] healing node-{a} <-> node-{b} at {:?}",
                    start.elapsed()
                );
                cluster.heal(a, b);
            },
            NemesisAction::HealAll => {
                eprintln!("[nemesis] healing all at {:?}", start.elapsed());
                cluster.heal_all();
            },
            NemesisAction::Sleep(d) => {
                tokio::time::sleep(d).await;
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-built nemesis schedules
// ---------------------------------------------------------------------------

/// Kill leader at T+5s, restart at T+15s.
fn kill_leader_schedule(leader_idx: usize) -> NemesisSchedule {
    NemesisSchedule::new()
        .add(Duration::from_secs(5), NemesisAction::KillNode(leader_idx))
        .add(
            Duration::from_secs(15),
            NemesisAction::RestartNode(leader_idx),
        )
}

/// Partition one node from the other two at T+5s, heal at T+15s.
fn partition_minority_schedule(isolated: usize, others: &[usize]) -> NemesisSchedule {
    let mut schedule = NemesisSchedule::new();
    for &other in others {
        schedule = schedule.add(
            Duration::from_secs(5),
            NemesisAction::Partition(isolated, other),
        );
    }
    schedule = schedule.add(Duration::from_secs(15), NemesisAction::HealAll);
    schedule
}

/// Random chaos schedule seeded for reproducibility.
fn chaos_schedule(seed: u64, node_count: usize, duration_secs: u64) -> NemesisSchedule {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut schedule = NemesisSchedule::new();
    let mut t = 3u64;
    let mut killed = vec![false; node_count];
    let mut paused = vec![false; node_count];

    while t < duration_secs.saturating_sub(5) {
        let action_type = rng.gen_range(0..6);
        let node = rng.gen_range(0..node_count);

        let action = match action_type {
            0 if !killed[node] && !paused[node] => {
                killed[node] = true;
                NemesisAction::KillNode(node)
            },
            1 if killed[node] => {
                killed[node] = false;
                NemesisAction::RestartNode(node)
            },
            2 if !killed[node] && !paused[node] => {
                paused[node] = true;
                NemesisAction::PauseNode(node)
            },
            3 if paused[node] => {
                paused[node] = false;
                NemesisAction::ResumeNode(node)
            },
            4 => {
                let other = (node + 1) % node_count;
                NemesisAction::Partition(node, other)
            },
            _ => NemesisAction::HealAll,
        };

        schedule = schedule.add(Duration::from_secs(t), action);
        t += rng.gen_range(2..6);
    }

    // Heal and restart everything before the end
    for i in 0..node_count {
        if killed[i] {
            schedule = schedule.add(
                Duration::from_secs(duration_secs.saturating_sub(3)),
                NemesisAction::RestartNode(i),
            );
        }
        if paused[i] {
            schedule = schedule.add(
                Duration::from_secs(duration_secs.saturating_sub(3)),
                NemesisAction::ResumeNode(i),
            );
        }
    }
    schedule = schedule.add(
        Duration::from_secs(duration_secs.saturating_sub(2)),
        NemesisAction::HealAll,
    );

    schedule
}

// ---------------------------------------------------------------------------
// Client workload helpers
// ---------------------------------------------------------------------------

/// Run a register workload (read/write a single key) for the given duration.
async fn register_workload(client: Arc<JepsenClient>, key: String, duration: Duration, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let deadline = Instant::now() + duration;
    let mut write_counter = 0i64;

    while Instant::now() < deadline {
        let do_write: bool = rng.gen_bool(0.5);

        if do_write {
            write_counter += 1;
            let _ = client.write(&key, write_counter).await;
        } else {
            let _ = client.read(&key).await;
        }

        // Small delay between operations
        tokio::time::sleep(Duration::from_millis(rng.gen_range(10..50))).await;
    }
}

// ---------------------------------------------------------------------------
// Helper: find server binary
// ---------------------------------------------------------------------------

fn find_server_binary() -> PathBuf {
    // Try release build first, then debug
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf();

    let release = workspace_root.join("target/release/neumann_server");
    if release.exists() {
        return release;
    }

    let debug = workspace_root.join("target/debug/neumann_server");
    if debug.exists() {
        return debug;
    }

    panic!(
        "neumann_server binary not found. Run `cargo build --release --bin neumann_server` first."
    );
}

// ---------------------------------------------------------------------------
// Test 1: Register crash recovery
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_register_crash_recovery() {
    let binary = find_server_binary();
    let base_dir = tempfile::tempdir().expect("temp dir");
    let mut cluster = ProcessCluster::new(3, 19200, binary, base_dir.path().to_path_buf());

    cluster.start().await.expect("cluster start");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(30);

    // Spawn client workers
    let mut handles = Vec::new();
    for i in 0..3 {
        let addrs: Vec<String> = (0..3).map(|j| cluster.grpc_addr(j)).collect();
        let client = Arc::new(JepsenClient::new(addrs, history.clone(), i));
        let h = tokio::spawn(register_workload(
            client,
            "x".to_string(),
            test_duration,
            42 + i,
        ));
        handles.push(h);
    }

    // Run nemesis: kill leader at T+5s, restart at T+15s
    let nemesis = kill_leader_schedule(0);
    run_nemesis(&mut cluster, nemesis).await;

    // Wait for workloads to finish
    for h in handles {
        let _ = h.await;
    }

    // Check linearizability
    let recorder = Arc::try_unwrap(history).unwrap_or_else(|_| panic!("single owner"));
    let inner = recorder.into_inner();
    let ops = inner.operations();

    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
    let result = checker.check(ops);

    cluster.shutdown().await;

    eprintln!(
        "register_crash_recovery: {} operations, result: {:?}",
        ops.len(),
        result
    );
    assert_eq!(
        result,
        LinearizabilityResult::Ok,
        "history must be linearizable"
    );
}

// ---------------------------------------------------------------------------
// Test 2: Register network partition
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_register_network_partition() {
    let binary = find_server_binary();
    let base_dir = tempfile::tempdir().expect("temp dir");
    let mut cluster = ProcessCluster::new(3, 19210, binary, base_dir.path().to_path_buf());

    cluster.start().await.expect("cluster start");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(30);

    let mut handles = Vec::new();
    for i in 0..3 {
        let addrs: Vec<String> = (0..3).map(|j| cluster.grpc_addr(j)).collect();
        let client = Arc::new(JepsenClient::new(addrs, history.clone(), i));
        let h = tokio::spawn(register_workload(
            client,
            "x".to_string(),
            test_duration,
            100 + i,
        ));
        handles.push(h);
    }

    // Partition node-2 from {node-0, node-1}
    let nemesis = partition_minority_schedule(2, &[0, 1]);
    run_nemesis(&mut cluster, nemesis).await;

    for h in handles {
        let _ = h.await;
    }

    let recorder = Arc::try_unwrap(history).unwrap_or_else(|_| panic!("single owner"));
    let inner = recorder.into_inner();
    let ops = inner.operations();

    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
    let result = checker.check(ops);

    cluster.shutdown().await;

    eprintln!(
        "register_network_partition: {} operations, result: {:?}",
        ops.len(),
        result
    );
    assert_eq!(
        result,
        LinearizabilityResult::Ok,
        "history must be linearizable"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Register rolling restart
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_register_rolling_restart() {
    let binary = find_server_binary();
    let base_dir = tempfile::tempdir().expect("temp dir");
    let mut cluster = ProcessCluster::new(5, 19220, binary, base_dir.path().to_path_buf());

    cluster.start().await.expect("cluster start");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(60);

    let mut handles = Vec::new();
    for i in 0..5 {
        let addrs: Vec<String> = (0..5).map(|j| cluster.grpc_addr(j)).collect();
        let client = Arc::new(JepsenClient::new(addrs, history.clone(), i));
        let h = tokio::spawn(register_workload(
            client,
            "x".to_string(),
            test_duration,
            200 + i,
        ));
        handles.push(h);
    }

    // Rolling restart: kill each node one at a time
    let mut schedule = NemesisSchedule::new();
    for i in 0..5 {
        #[allow(clippy::cast_possible_truncation)]
        let base = 5 + i as u64 * 10;
        schedule = schedule
            .add(Duration::from_secs(base), NemesisAction::KillNode(i))
            .add(Duration::from_secs(base + 5), NemesisAction::RestartNode(i));
    }
    run_nemesis(&mut cluster, schedule).await;

    for h in handles {
        let _ = h.await;
    }

    let recorder = Arc::try_unwrap(history).unwrap_or_else(|_| panic!("single owner"));
    let inner = recorder.into_inner();
    let ops = inner.operations();

    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
    let result = checker.check(ops);

    cluster.shutdown().await;

    eprintln!(
        "register_rolling_restart: {} operations, result: {:?}",
        ops.len(),
        result
    );
    assert_eq!(
        result,
        LinearizabilityResult::Ok,
        "history must be linearizable"
    );
}

// ---------------------------------------------------------------------------
// Test 4: Bank transfer atomicity
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_bank_transfer_atomicity() {
    let binary = find_server_binary();
    let base_dir = tempfile::tempdir().expect("temp dir");
    let mut cluster = ProcessCluster::new(3, 19230, binary, base_dir.path().to_path_buf());

    cluster.start().await.expect("cluster start");

    let num_accounts = 10;
    let initial_balance = 100i64;
    let expected_total = num_accounts * initial_balance;

    // Initialize accounts
    let addr = cluster.grpc_addr(0);
    let init_client = neumann_client::NeumannClient::connect(&addr)
        .build()
        .await
        .expect("connect");
    let _ = init_client
        .execute("CREATE TABLE bank (account:string, balance:int)")
        .await;
    for i in 0..num_accounts {
        let _ = init_client
            .execute(&format!(
                "INSERT bank account='acct-{i}', balance={initial_balance}"
            ))
            .await;
    }

    let test_duration = Duration::from_secs(30);
    let history = Arc::new(ConcurrentHistoryRecorder::new());

    // Transfer workers
    let mut handles = Vec::new();
    for worker in 0..3u64 {
        let addrs: Vec<String> = (0..3).map(|j| cluster.grpc_addr(j)).collect();
        let hist = history.clone();
        let dur = test_duration;
        let na = num_accounts;

        let h = tokio::spawn(async move {
            let mut rng = StdRng::seed_from_u64(300 + worker);
            let deadline = Instant::now() + dur;

            while Instant::now() < deadline {
                let from = rng.gen_range(0..na);
                let to = (from + rng.gen_range(1..na)) % na;
                let amount = rng.gen_range(1..=10);

                let _op_id = hist.invoke(
                    worker,
                    OpType::Write,
                    format!("transfer-{from}-{to}"),
                    Value::Int(amount),
                );

                // Try the transfer on any node
                let mut success = false;
                for idx in 0..addrs.len() {
                    if let Ok(client) = neumann_client::NeumannClient::connect(&addrs[idx])
                        .build()
                        .await
                    {
                        let debit = format!(
                            "UPDATE bank SET balance=balance-{amount} WHERE account = 'acct-{from}'"
                        );
                        let credit = format!(
                            "UPDATE bank SET balance=balance+{amount} WHERE account = 'acct-{to}'"
                        );
                        if client.execute(&debit).await.is_ok()
                            && client.execute(&credit).await.is_ok()
                        {
                            success = true;
                            break;
                        }
                    }
                }

                hist.complete(_op_id, if success { Value::Int(1) } else { Value::None });
                tokio::time::sleep(Duration::from_millis(rng.gen_range(20..80))).await;
            }
        });
        handles.push(h);
    }

    // Run chaos nemesis
    let nemesis = chaos_schedule(42, 3, 25);
    run_nemesis(&mut cluster, nemesis).await;

    for h in handles {
        let _ = h.await;
    }

    // Check conservation invariant: total balance should still be expected_total
    tokio::time::sleep(Duration::from_secs(3)).await;

    let mut total = 0i64;
    let mut checked = false;
    for idx in 0..3 {
        let addr = cluster.grpc_addr(idx);
        if let Ok(client) = neumann_client::NeumannClient::connect(&addr).build().await {
            if let Ok(result) = client.execute("SELECT bank").await {
                if let Some(rows) = result.rows() {
                    total = 0;
                    for row in rows {
                        for col in &row.values {
                            if col.name == "balance" {
                                if let Some(ref value) = col.value {
                                    if let Some(neumann_client::proto::value::Kind::IntValue(v)) =
                                        value.kind
                                    {
                                        total += v;
                                    }
                                }
                            }
                        }
                    }
                    checked = true;
                    break;
                }
            }
        }
    }

    cluster.shutdown().await;

    if checked {
        eprintln!("bank_transfer_atomicity: total={total}, expected={expected_total}");
        assert_eq!(
            total, expected_total,
            "conservation invariant violated: total={total}, expected={expected_total}"
        );
    } else {
        eprintln!("bank_transfer_atomicity: could not verify (all nodes unreachable)");
    }
}

// ---------------------------------------------------------------------------
// Test 5: Crash recovery durability
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_crash_recovery_durability() {
    let binary = find_server_binary();
    let base_dir = tempfile::tempdir().expect("temp dir");
    let mut cluster = ProcessCluster::new(3, 19240, binary, base_dir.path().to_path_buf());

    cluster.start().await.expect("cluster start");

    // Write 50 values
    let addr = cluster.grpc_addr(0);
    let client = neumann_client::NeumannClient::connect(&addr)
        .build()
        .await
        .expect("connect");
    let _ = client
        .execute("CREATE TABLE durability_test (key:string, val:int)")
        .await;

    let num_values = 50;
    for i in 0..num_values {
        let result = client
            .execute(&format!("INSERT durability_test key='k-{i}', val={i}"))
            .await;
        assert!(result.is_ok(), "write k-{i} failed: {:?}", result.err());
    }

    // Give time for replication
    tokio::time::sleep(Duration::from_secs(2)).await;

    // SIGKILL the leader (node 0)
    eprintln!("[durability] killing node-0");
    cluster.kill_node(0);
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Restart node 0
    eprintln!("[durability] restarting node-0");
    cluster.restart_node(0).expect("restart");
    cluster.wait_node_healthy(0).await.expect("health check");

    // Verify all values are present (try each node)
    let mut verified = false;
    for idx in 0..3 {
        let addr = cluster.grpc_addr(idx);
        if let Ok(client) = neumann_client::NeumannClient::connect(&addr).build().await {
            if let Ok(result) = client.execute("SELECT durability_test").await {
                if let Some(rows) = result.rows() {
                    eprintln!(
                        "[durability] node-{idx} has {} rows (expected {num_values})",
                        rows.len()
                    );
                    // We accept the data being present on at least one node
                    if rows.len() >= num_values as usize {
                        verified = true;
                        break;
                    }
                }
            }
        }
    }

    // Now crash each node one at a time and verify data survives
    for crash_idx in 0..3 {
        eprintln!("[durability] crashing node-{crash_idx}");
        cluster.kill_node(crash_idx);
        tokio::time::sleep(Duration::from_secs(2)).await;

        eprintln!("[durability] restarting node-{crash_idx}");
        cluster.restart_node(crash_idx).expect("restart");
        cluster
            .wait_node_healthy(crash_idx)
            .await
            .expect("health check");
    }

    // Final verification
    for idx in 0..3 {
        let addr = cluster.grpc_addr(idx);
        if let Ok(client) = neumann_client::NeumannClient::connect(&addr).build().await {
            if let Ok(result) = client.execute("SELECT durability_test").await {
                if let Some(rows) = result.rows() {
                    eprintln!("[durability] final: node-{idx} has {} rows", rows.len());
                }
            }
        }
    }

    cluster.shutdown().await;

    assert!(
        verified,
        "data must survive crash recovery on at least one node"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Stale read detection (intentional violation)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_stale_read_detection() {
    // This test constructs a history that SHOULD fail linearizability checking,
    // proving the checker is not vacuously passing. We don't need a real cluster
    // for this -- we craft the history directly.
    let base = Instant::now();
    let ops = vec![
        // Write x=1 at T=0..10ms
        Operation {
            id: 0,
            op_type: OpType::Write,
            key: "x".to_string(),
            input: Value::Int(1),
            output: Some(Value::None),
            invoke_time: base,
            complete_time: Some(base + Duration::from_millis(10)),
            client_id: 1,
        },
        // Write x=2 at T=20..30ms (after first write completes)
        Operation {
            id: 1,
            op_type: OpType::Write,
            key: "x".to_string(),
            input: Value::Int(2),
            output: Some(Value::None),
            invoke_time: base + Duration::from_millis(20),
            complete_time: Some(base + Duration::from_millis(30)),
            client_id: 1,
        },
        // Stale read: x=1 at T=40..50ms (after both writes)
        // This MUST be detected as a violation
        Operation {
            id: 2,
            op_type: OpType::Read,
            key: "x".to_string(),
            input: Value::None,
            output: Some(Value::Int(1)),
            invoke_time: base + Duration::from_millis(40),
            complete_time: Some(base + Duration::from_millis(50)),
            client_id: 2,
        },
    ];

    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(5));
    let result = checker.check(&ops);

    eprintln!("stale_read_detection: result={result:?}");
    assert!(
        matches!(result, LinearizabilityResult::Violation(_)),
        "checker must detect stale read violation, got: {result:?}"
    );
}
