# Deadlock Resolution

## Overview

tensor_chain automatically detects and resolves deadlocks in distributed transactions using wait-for graph analysis. This runbook covers monitoring, tuning, and manual intervention.

## Automatic Detection

Deadlocks are detected within `detection_interval_ms` (default: 100ms) and resolved by aborting a victim transaction based on configured policy.

## Monitoring

### Metrics

```promql
# Deadlock rate
rate(tensor_chain_deadlocks_total[5m])

# Detection latency
histogram_quantile(0.99, tensor_chain_deadlock_detection_seconds_bucket)

# Victim aborts by policy
tensor_chain_deadlock_victims_total{policy="youngest"}
```

### Logs

```bash
grep "deadlock" /var/log/neumann/tensor_chain.log

# Example output:
# [WARN] Deadlock detected: cycle=[tx_123, tx_456, tx_789], victim=tx_789
# [INFO] Aborted transaction tx_789 (youngest in cycle)
```

## Tuning

### Detection Interval

```toml
[deadlock]
detection_interval_ms = 100  # Lower = faster detection, higher CPU
```

Trade-off:
- Lower interval: Faster detection, but more CPU overhead
- Higher interval: Less overhead, but longer deadlock duration

### Victim Selection Policy

```toml
[deadlock]
victim_policy = "youngest"  # Options: youngest, oldest, lowest_priority, most_locks
```

| Policy | Use Case |
|--------|----------|
| `youngest` | Minimize wasted work (default) |
| `oldest` | Prevent starvation of long transactions |
| `lowest_priority` | Business-critical transactions survive |
| `most_locks` | Maximize system throughput |

### Transaction Priorities

```rust
// Set priority when starting transaction
let tx = coordinator.begin_with_priority(Priority::High)?;
```

## Manual Intervention

### Force Abort Specific Transaction

```bash
neumann-admin tx abort --tx-id 12345 --reason "manual deadlock resolution"
```

### Clear All Pending Transactions

```bash
# Emergency only - will lose in-flight work
neumann-admin tx clear-pending --confirm
```

### Disable Auto-Resolution

```toml
[deadlock]
auto_abort_victim = false  # Require manual intervention
```

Then manually resolve:
```bash
# List detected deadlocks
neumann-admin deadlock list

# Resolve specific deadlock
neumann-admin deadlock resolve --cycle-id abc123 --abort tx_789
```

## Prevention

### Lock Ordering

Acquire locks in consistent order across all transactions:

```rust
// Good: always lock in sorted key order
let mut keys = vec!["key_b", "key_a", "key_c"];
keys.sort();
for key in keys {
    tx.lock(key)?;
}
```

### Timeout-Based Prevention

```toml
[transaction]
lock_timeout_ms = 5000  # Abort if can't acquire lock within 5s
```

### Reduce Lock Scope

```rust
// Bad: lock entire table
tx.lock("users/*")?;

// Good: lock specific keys
tx.lock("users/123")?;
tx.lock("users/456")?;
```

## Troubleshooting

### High Deadlock Rate

**Cause**: Hot keys with many concurrent transactions

**Solution**:
1. Identify hot keys: `neumann-admin lock-stats --top 10`
2. Consider sharding hot keys
3. Batch operations to reduce lock duration

### Detection Latency Spikes

**Cause**: Large wait-for graph from many concurrent transactions

**Solution**:
1. Increase `max_concurrent_transactions`
2. Reduce transaction duration
3. Consider optimistic concurrency for read-heavy workloads

### False Positives

**Cause**: Network delays causing timeout-based false waits

**Solution**:
1. Increase `lock_wait_threshold_ms`
2. Verify network latency between nodes
3. Check for GC pauses
