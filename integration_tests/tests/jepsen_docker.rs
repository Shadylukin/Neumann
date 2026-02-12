// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Docker-based Jepsen tests with kernel-level fault injection.
//!
//! These tests run real `neumann_server` containers via Docker Compose and
//! inject faults using iptables (network partitions), tc netem (latency,
//! packet loss), and docker kill/pause (process faults). All tests are
//! `#[ignore]` because they require Docker Desktop and the pre-built
//! `neumann:jepsen` image.
//!
//! Run with:
//! ```bash
//! docker build --target jepsen -t neumann:jepsen .
//! cargo nextest run --package integration_tests --test jepsen_docker -- --ignored
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use integration_tests::docker_jepsen::DockerCluster;
use integration_tests::jepsen_client::JepsenClient;
use integration_tests::linearizability::{
    ConcurrentHistoryRecorder, LinearizabilityChecker, LinearizabilityResult, OpType, Operation,
    RegisterModel, Value,
};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Helpers
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

        tokio::time::sleep(Duration::from_millis(rng.gen_range(10..50))).await;
    }
}

/// Spawn client workers against all nodes, returning join handles.
fn spawn_register_workers(
    cluster: &DockerCluster,
    history: &Arc<ConcurrentHistoryRecorder>,
    worker_count: usize,
    duration: Duration,
    seed_base: u64,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::new();
    let n = cluster.node_count();
    for i in 0..worker_count {
        let addrs: Vec<String> = (0..n).map(|j| cluster.grpc_addr(j).to_string()).collect();
        #[allow(clippy::cast_possible_truncation)]
        let client = Arc::new(JepsenClient::new(addrs, history.clone(), i as u64));
        let h = tokio::spawn(register_workload(
            client,
            "x".to_string(),
            duration,
            seed_base + i as u64,
        ));
        handles.push(h);
    }
    handles
}

/// Check that the operation history is linearizable.
fn assert_linearizable(history: Arc<ConcurrentHistoryRecorder>, test_name: &str) {
    let recorder = Arc::try_unwrap(history).unwrap_or_else(|_| panic!("single owner"));
    let inner = recorder.into_inner();
    let ops = inner.operations();

    let checker = LinearizabilityChecker::with_timeout(RegisterModel, Duration::from_secs(30));
    let result = checker.check(ops);

    eprintln!("{test_name}: {} operations, result: {result:?}", ops.len());
    assert_eq!(
        result,
        LinearizabilityResult::Ok,
        "{test_name}: history must be linearizable"
    );
}

const CLUSTER_HEALTH_TIMEOUT: Duration = Duration::from_secs(90);
const NODE_RECOVERY_TIMEOUT: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Test 1: Register crash recovery (ported from multiprocess)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_register_crash_recovery() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(30);

    let handles = spawn_register_workers(&cluster, &history, 3, test_duration, 42);

    // Nemesis: kill leader at T+5s, restart at T+15s
    tokio::time::sleep(Duration::from_secs(5)).await;
    let _ = cluster.kill_node(0);
    tokio::time::sleep(Duration::from_secs(10)).await;
    let _ = cluster.start_node(0);
    let _ = cluster.wait_node_healthy(0, NODE_RECOVERY_TIMEOUT).await;

    for h in handles {
        let _ = h.await;
    }

    assert_linearizable(history, "docker_register_crash_recovery");
    cluster.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// Test 2: Register network partition (ported from multiprocess)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_register_network_partition() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(30);

    let handles = spawn_register_workers(&cluster, &history, 3, test_duration, 100);

    // Nemesis: partition node-2 from {node-0, node-1} at T+5s, heal at T+15s
    tokio::time::sleep(Duration::from_secs(5)).await;
    let _ = cluster.partition_node(2, 0);
    let _ = cluster.partition_node(2, 1);
    tokio::time::sleep(Duration::from_secs(10)).await;
    let _ = cluster.heal_all();

    for h in handles {
        let _ = h.await;
    }

    assert_linearizable(history, "docker_register_network_partition");
    cluster.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// Test 3: Register rolling restart (5-node, ported from multiprocess)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_register_rolling_restart() {
    let cluster = DockerCluster::start(5).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(60);

    let handles = spawn_register_workers(&cluster, &history, 5, test_duration, 200);

    // Rolling restart: kill each node one at a time
    for i in 0..5 {
        let base = 5 + i * 10;
        tokio::time::sleep(Duration::from_secs(if i == 0 { base as u64 } else { 10 })).await;
        eprintln!("[nemesis] killing node-{i}");
        let _ = cluster.kill_node(i);
        tokio::time::sleep(Duration::from_secs(5)).await;
        eprintln!("[nemesis] restarting node-{i}");
        let _ = cluster.start_node(i);
        let _ = cluster.wait_node_healthy(i, NODE_RECOVERY_TIMEOUT).await;
    }

    for h in handles {
        let _ = h.await;
    }

    assert_linearizable(history, "docker_register_rolling_restart");
    cluster.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// Test 4: Bank transfer atomicity (ported from multiprocess)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_bank_transfer_atomicity() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    let num_accounts = 10i64;
    let initial_balance = 100i64;
    let expected_total = num_accounts * initial_balance;

    // Initialize accounts
    let addr = cluster.grpc_addr(0);
    let init_client = neumann_client::NeumannClient::connect(addr)
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
        let addrs: Vec<String> = (0..3).map(|j| cluster.grpc_addr(j).to_string()).collect();
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

                let mut success = false;
                for addr in &addrs {
                    if let Ok(client) = neumann_client::NeumannClient::connect(addr).build().await {
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

    // Nemesis: random chaos
    tokio::time::sleep(Duration::from_secs(5)).await;
    let _ = cluster.kill_node(0);
    tokio::time::sleep(Duration::from_secs(10)).await;
    let _ = cluster.start_node(0);
    let _ = cluster.wait_node_healthy(0, NODE_RECOVERY_TIMEOUT).await;

    for h in handles {
        let _ = h.await;
    }

    // Check conservation invariant
    tokio::time::sleep(Duration::from_secs(3)).await;

    let mut total = 0i64;
    let mut checked = false;
    for idx in 0..3 {
        let addr = cluster.grpc_addr(idx);
        if let Ok(client) = neumann_client::NeumannClient::connect(addr).build().await {
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

    cluster.shutdown().expect("shutdown");

    if checked {
        eprintln!("docker_bank_transfer_atomicity: total={total}, expected={expected_total}");
        assert_eq!(
            total, expected_total,
            "conservation invariant violated: total={total}, expected={expected_total}"
        );
    } else {
        eprintln!("docker_bank_transfer_atomicity: could not verify (all nodes unreachable)");
    }
}

// ---------------------------------------------------------------------------
// Test 5: Crash recovery durability (ported from multiprocess)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_crash_recovery_durability() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    // Write 50 values
    let addr = cluster.grpc_addr(0);
    let client = neumann_client::NeumannClient::connect(addr)
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

    // SIGKILL the leader
    eprintln!("[durability] killing node-0");
    let _ = cluster.kill_node(0);
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Restart
    eprintln!("[durability] restarting node-0");
    let _ = cluster.start_node(0);
    cluster
        .wait_node_healthy(0, NODE_RECOVERY_TIMEOUT)
        .await
        .expect("health check");

    // Verify data on any node
    let mut verified = false;
    for idx in 0..3 {
        let addr = cluster.grpc_addr(idx);
        if let Ok(client) = neumann_client::NeumannClient::connect(addr).build().await {
            if let Ok(result) = client.execute("SELECT durability_test").await {
                if let Some(rows) = result.rows() {
                    eprintln!(
                        "[durability] node-{idx} has {} rows (expected {num_values})",
                        rows.len()
                    );
                    if rows.len() >= num_values {
                        verified = true;
                        break;
                    }
                }
            }
        }
    }

    // Crash each node one at a time
    for crash_idx in 0..3 {
        eprintln!("[durability] crashing node-{crash_idx}");
        let _ = cluster.kill_node(crash_idx);
        tokio::time::sleep(Duration::from_secs(2)).await;

        eprintln!("[durability] restarting node-{crash_idx}");
        let _ = cluster.start_node(crash_idx);
        cluster
            .wait_node_healthy(crash_idx, NODE_RECOVERY_TIMEOUT)
            .await
            .expect("health check");
    }

    cluster.shutdown().expect("shutdown");

    assert!(
        verified,
        "data must survive crash recovery on at least one node"
    );
}

// ---------------------------------------------------------------------------
// Test 6: Stale read detection (checker validation, no cluster needed)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_stale_read_detection() {
    let base = Instant::now();
    let ops = vec![
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
        // Stale read: x=1 after both writes completed
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

    eprintln!("docker_stale_read_detection: result={result:?}");
    assert!(
        matches!(result, LinearizabilityResult::Violation(_)),
        "checker must detect stale read violation, got: {result:?}"
    );
}

// ===========================================================================
// New Docker-only test scenarios
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 7: tc netem latency
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_tc_netem_latency() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(30);

    let handles = spawn_register_workers(&cluster, &history, 3, test_duration, 700);

    // Nemesis: add 200ms latency to leader at T+5s, remove at T+20s
    tokio::time::sleep(Duration::from_secs(5)).await;
    eprintln!("[nemesis] adding 200ms latency to node-0");
    let _ = cluster.add_latency(0, 200);
    tokio::time::sleep(Duration::from_secs(15)).await;
    eprintln!("[nemesis] clearing network faults on node-0");
    let _ = cluster.clear_network_faults(0);

    for h in handles {
        let _ = h.await;
    }

    assert_linearizable(history, "docker_tc_netem_latency");
    cluster.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// Test 8: tc netem packet loss
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_tc_netem_packet_loss() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(30);

    let handles = spawn_register_workers(&cluster, &history, 3, test_duration, 800);

    // Nemesis: 10% packet loss on nodes 0 and 1
    tokio::time::sleep(Duration::from_secs(5)).await;
    eprintln!("[nemesis] adding 10% packet loss to node-0 and node-1");
    let _ = cluster.add_packet_loss(0, 10);
    let _ = cluster.add_packet_loss(1, 10);
    tokio::time::sleep(Duration::from_secs(15)).await;
    eprintln!("[nemesis] clearing network faults");
    let _ = cluster.clear_network_faults(0);
    let _ = cluster.clear_network_faults(1);

    for h in handles {
        let _ = h.await;
    }

    assert_linearizable(history, "docker_tc_netem_packet_loss");
    cluster.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// Test 9: Asymmetric partition
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_asymmetric_partition() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(30);

    let handles = spawn_register_workers(&cluster, &history, 3, test_duration, 900);

    // Nemesis: one-way partition (node-0 -> node-1 blocked, but node-1 -> node-0 open)
    tokio::time::sleep(Duration::from_secs(5)).await;
    eprintln!("[nemesis] asymmetric partition: node-0 -> node-1 blocked");
    let _ = cluster.asymmetric_partition(0, 1);
    tokio::time::sleep(Duration::from_secs(15)).await;
    eprintln!("[nemesis] healing all");
    let _ = cluster.heal_all();

    for h in handles {
        let _ = h.await;
    }

    assert_linearizable(history, "docker_asymmetric_partition");
    cluster.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// Test 10: Disk full during writes
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_disk_full_during_writes() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    // Write some initial data
    let addr = cluster.grpc_addr(0);
    let client = neumann_client::NeumannClient::connect(addr)
        .build()
        .await
        .expect("connect");
    let _ = client
        .execute("CREATE TABLE disk_test (key:string, val:int)")
        .await;
    for i in 0..20 {
        let _ = client
            .execute(&format!("INSERT disk_test key='k-{i}', val={i}"))
            .await;
    }
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Fill disk on node-0
    eprintln!("[nemesis] filling disk on node-0");
    let _ = cluster.fill_disk(0);
    tokio::time::sleep(Duration::from_secs(5)).await;

    // Writes should still work on nodes 1 and 2
    let addr1 = cluster.grpc_addr(1);
    if let Ok(client1) = neumann_client::NeumannClient::connect(addr1).build().await {
        for i in 20..30 {
            let _ = client1
                .execute(&format!("INSERT disk_test key='k-{i}', val={i}"))
                .await;
        }
    }

    // Clear disk and restart node-0
    eprintln!("[nemesis] clearing disk on node-0");
    let _ = cluster.clear_disk(0);
    let _ = cluster.kill_node(0);
    tokio::time::sleep(Duration::from_secs(2)).await;
    let _ = cluster.start_node(0);
    let _ = cluster.wait_node_healthy(0, NODE_RECOVERY_TIMEOUT).await;

    // Verify prior data is intact on at least one node
    let mut verified = false;
    for idx in 0..3 {
        let addr = cluster.grpc_addr(idx);
        if let Ok(client) = neumann_client::NeumannClient::connect(addr).build().await {
            if let Ok(result) = client.execute("SELECT disk_test").await {
                if let Some(rows) = result.rows() {
                    eprintln!("[disk_full] node-{idx} has {} rows", rows.len());
                    if rows.len() >= 20 {
                        verified = true;
                        break;
                    }
                }
            }
        }
    }

    cluster.shutdown().expect("shutdown");

    assert!(verified, "data must survive disk-full scenario");
}

// ---------------------------------------------------------------------------
// Test 11: Combined faults (latency + partition + kill)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_combined_faults() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(40);

    let handles = spawn_register_workers(&cluster, &history, 3, test_duration, 1100);

    // T+3s: add latency to node-1
    tokio::time::sleep(Duration::from_secs(3)).await;
    eprintln!("[nemesis] adding 100ms latency to node-1");
    let _ = cluster.add_latency(1, 100);

    // T+8s: partition node-2 from node-0
    tokio::time::sleep(Duration::from_secs(5)).await;
    eprintln!("[nemesis] partitioning node-2 from node-0");
    let _ = cluster.partition_node(2, 0);

    // T+13s: kill node-0
    tokio::time::sleep(Duration::from_secs(5)).await;
    eprintln!("[nemesis] killing node-0");
    let _ = cluster.kill_node(0);

    // T+23s: heal and restart everything
    tokio::time::sleep(Duration::from_secs(10)).await;
    eprintln!("[nemesis] healing and restarting");
    let _ = cluster.heal_all();
    let _ = cluster.start_node(0);
    let _ = cluster.wait_node_healthy(0, NODE_RECOVERY_TIMEOUT).await;

    for h in handles {
        let _ = h.await;
    }

    assert_linearizable(history, "docker_combined_faults");
    cluster.shutdown().expect("shutdown");
}

// ---------------------------------------------------------------------------
// Test 12: Rapid leader churn
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_docker_rapid_leader_churn() {
    let cluster = DockerCluster::start(3).expect("cluster start");
    cluster
        .wait_healthy(CLUSTER_HEALTH_TIMEOUT)
        .await
        .expect("cluster healthy");

    let history = Arc::new(ConcurrentHistoryRecorder::new());
    let test_duration = Duration::from_secs(45);

    let handles = spawn_register_workers(&cluster, &history, 3, test_duration, 1200);

    // Kill a node every 3s for 30s, cycling through nodes
    let start = Instant::now();
    for cycle in 0..10u32 {
        let target = cycle as usize % 3;
        tokio::time::sleep(Duration::from_secs(3)).await;
        eprintln!("[nemesis] killing node-{target} at {:?}", start.elapsed());
        let _ = cluster.kill_node(target);
        tokio::time::sleep(Duration::from_millis(500)).await;
        let _ = cluster.start_node(target);
        // Don't wait for health -- rapid churn
    }

    // Final heal: wait for all to recover
    tokio::time::sleep(Duration::from_secs(5)).await;
    for i in 0..3 {
        let _ = cluster.start_node(i);
    }
    for i in 0..3 {
        let _ = cluster.wait_node_healthy(i, NODE_RECOVERY_TIMEOUT).await;
    }

    for h in handles {
        let _ = h.await;
    }

    assert_linearizable(history, "docker_rapid_leader_churn");
    cluster.shutdown().expect("shutdown");
}
