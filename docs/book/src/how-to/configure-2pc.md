# Configure Two-Phase Commit

This guide covers tuning the distributed transaction coordinator (2PC),
lock manager, and deadlock detection. For background on the 2PC protocol
and semantic conflict detection, see
[Distributed Transactions](../explanation/distributed-transactions.md). For the
full parameter reference, see
[Tensor Chain API Reference](../reference/api/tensor-chain.md).

## Basic Configuration

Create a `DistributedTxConfig` with default settings:

```rust
let config = DistributedTxConfig::default();
// max_concurrent: 100
// prepare_timeout_ms: 5,000
// commit_timeout_ms: 10,000
// orthogonal_threshold: 0.1
// optimistic_locking: true
```

## Transaction Timeout Settings

### Prepare Phase Timeout

Controls how long the coordinator waits for all participants to vote:

```rust
DistributedTxConfig {
    prepare_timeout_ms: 5_000,  // default
    ..Default::default()
}
```

- **Lower (1,000-3,000)**: Fast failure detection but may abort transactions
  prematurely on slow networks. Use for same-datacenter deployments.
- **Higher (10,000-30,000)**: More tolerant of network delays but holds locks
  longer during prepare. Use for geo-distributed deployments.

### Commit Phase Timeout

Controls how long the coordinator waits for commit/abort acknowledgments:

```rust
DistributedTxConfig {
    commit_timeout_ms: 10_000,  // default
    ..Default::default()
}
```

The commit timeout should be at least 2x the prepare timeout. If a commit
times out, the coordinator uses presumed-commit semantics: it assumes the
commit succeeded and participants will eventually apply it.

### Relationship Between Timeouts

```text
lock_timeout > commit_timeout > prepare_timeout

Example:
  prepare_timeout_ms: 5,000
  commit_timeout_ms: 10,000
  lock_timeout_ms: 30,000     (in LockManager)
```

If lock timeout is shorter than transaction timeout, locks may expire before
the transaction completes, causing spurious aborts.

## Concurrency Limits

### Maximum Concurrent Transactions

```rust
DistributedTxConfig {
    max_concurrent: 100,  // default
    ..Default::default()
}
```

- **Lower (10-50)**: Reduces lock contention and memory usage. Use when
  transactions are long-running or touch many keys.
- **Higher (200-500)**: Allows more parallelism. Use when transactions are
  short and touch few keys.
- **Very high (1,000+)**: May cause excessive lock contention and deadlocks.
  Monitor `deadlocks_detected` and `conflict_rate` metrics.

## Semantic Conflict Detection

### Orthogonal Threshold

Controls when transactions are considered independent enough to auto-merge:

```rust
DistributedTxConfig {
    orthogonal_threshold: 0.1,  // default cosine similarity
    optimistic_locking: true,   // enable semantic detection
    ..Default::default()
}
```

Two transactions are orthogonal (can merge without conflict) when their delta
embedding cosine similarity is below this threshold AND their Jaccard
structural overlap is below 0.5.

- **Lower (0.01-0.05)**: Very strict, only truly independent transactions
  auto-merge. Use when data integrity is critical.
- **Higher (0.1-0.3)**: More permissive, allows loosely related transactions
  to auto-merge. Use when throughput is more important than strict isolation.
- **Disable**: Set `optimistic_locking: false` to fall back to pure
  lock-based conflict detection.

### Conflict Classification Summary

| Cosine | Jaccard | Result | Action |
| --- | --- | --- | --- |
| < 0.1 | < 0.5 | Orthogonal | Auto-merge (vector add) |
| 0.1-0.7 | < 0.5 | Low conflict | Weighted merge |
| >= 0.7 | any | Conflicting | Reject |
| any | >= 0.5 | Conflicting | Reject (structural overlap) |
| >= 0.99 | all keys | Identical | Deduplicate |
| <= -0.95 | all keys | Opposite | Cancel (no-op) |

## Deadlock Detection Configuration

### Basic Settings

```rust
let config = DeadlockDetectorConfig {
    enabled: true,                            // default
    detection_interval_ms: 100,               // default
    victim_policy: VictimSelectionPolicy::Youngest,  // default
    max_cycle_length: 100,                    // default
    auto_abort_victim: true,                  // default
};
```

### Detection Interval

How often the detector scans the wait-for graph for cycles:

- **Lower (10-50ms)**: Faster deadlock resolution but more CPU overhead.
  Use under heavy contention.
- **Higher (200-1000ms)**: Less CPU overhead but deadlocked transactions
  hold locks longer. Use when deadlocks are rare.

### Victim Selection Policy

Choose based on your workload characteristics:

| Policy | Best For | Trade-off |
| --- | --- | --- |
| `Youngest` | General purpose | Minimizes wasted work; long transactions always complete |
| `Oldest` | Fairness-sensitive workloads | Prevents starvation; wastes more completed work |
| `LowestPriority` | Priority-based systems | Requires priority assignment; business-rule driven |
| `MostLocks` | High-contention workloads | Frees most resources; may abort complex transactions |

```rust
// Priority-based victim selection
DeadlockDetectorConfig {
    victim_policy: VictimSelectionPolicy::LowestPriority,
    ..Default::default()
}
```

### Maximum Cycle Length

Limits the DFS depth during cycle detection. Cycles longer than this are
not detected:

```rust
DeadlockDetectorConfig {
    max_cycle_length: 100,  // default
    ..Default::default()
}
```

In practice, deadlock cycles are short (2-4 transactions). Increase only if
you have evidence of long dependency chains.

### Auto-Abort

When enabled, the detector automatically aborts the selected victim:

```rust
DeadlockDetectorConfig {
    auto_abort_victim: true,  // default
    ..Default::default()
}
```

Set to `false` if you want to handle deadlocks manually (for example, to log
the cycle and let the application retry).

## Lock Manager Tuning

The lock manager is embedded in `DistributedTxCoordinator` and uses the
`LockManager` type:

```rust
// Acquire lock with timeout
let lock = lock_manager.acquire(
    tx_id,
    key,
    LockMode::Exclusive,
    Duration::from_secs(5),
)?;
```

### Lock Compatibility

| Requested \ Held | Shared (S) | Exclusive (X) |
| --- | --- | --- |
| **Shared (S)** | Compatible | Blocked |
| **Exclusive (X)** | Blocked | Blocked |

### Lock Timeout

Set lock timeout to exceed the expected transaction duration:

```rust
// Lock timeout should be > commit_timeout_ms
let lock_timeout = Duration::from_secs(30);
```

### Orphaned Lock Cleanup

Locks from crashed transactions are cleaned up by:

1. The deadlock detector (detects stuck transactions)
2. Periodic `cleanup_expired()` calls
3. WAL recovery on coordinator restart

## Message Validation

For production, enable message validation to prevent DoS attacks:

```rust
let validation_config = MessageValidationConfig {
    enabled: true,
    max_term: u64::MAX - 1,
    max_shard_id: 65_536,
    max_tx_timeout_ms: 300_000,
    max_node_id_len: 256,
    max_key_len: 4_096,
    max_embedding_dimension: 65_536,
    max_embedding_magnitude: 1_000_000.0,
    max_query_len: 1_048_576,
    max_message_age_ms: 300_000,
    max_blocks_per_request: 1_000,
    max_snapshot_chunk_size: 10_485_760,
};
```

Tighten these bounds based on your actual data dimensions and key sizes.

## Complete Production Example

A production-ready 2PC configuration for a cross-datacenter deployment:

```rust
let tx_config = DistributedTxConfig {
    max_concurrent: 200,
    prepare_timeout_ms: 10_000,
    commit_timeout_ms: 20_000,
    orthogonal_threshold: 0.1,
    optimistic_locking: true,
};

let deadlock_config = DeadlockDetectorConfig {
    enabled: true,
    detection_interval_ms: 100,
    victim_policy: VictimSelectionPolicy::Youngest,
    max_cycle_length: 100,
    auto_abort_victim: true,
};

let validation_config = MessageValidationConfig {
    enabled: true,
    max_embedding_dimension: 1_024,  // tighten to your actual dimensions
    max_key_len: 512,                // tighten to your actual key sizes
    ..Default::default()
};
```

## Monitoring

Key metrics to watch for 2PC health:

| Metric | Warning | Critical | Action |
| --- | --- | --- | --- |
| `conflict_rate` | > 0.10 | > 0.30 | Reduce contention, shard data better |
| `deadlocks_detected` | > 0/min | > 10/min | Review lock ordering, increase concurrency limits |
| `commit_rate` | < 0.80 | < 0.50 | Investigate participant failures |

See [Monitoring Setup](monitoring-setup.md) for dashboard configuration.
