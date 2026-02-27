# Configure Raft Consensus

This guide covers tuning Raft consensus parameters for different deployment
environments. For background on the Raft protocol and Tensor-Raft extensions,
see [Consensus Protocols](../explanation/consensus-protocols.md). For the full
parameter reference, see [Tensor Chain API Reference](../reference/api/tensor-chain.md).

## Basic Configuration

Create a `RaftConfig` with default settings:

```rust
let config = RaftConfig::default();
// election_timeout: (150, 300) ms
// heartbeat_interval: 50 ms
// enable_fast_path: true
// enable_pre_vote: true
// enable_geometric_tiebreak: true
```

## Tuning Election Timeouts

Election timeouts must exceed the round-trip network latency by a comfortable
margin. Too low and you get unnecessary elections; too high and leader failure
takes longer to detect.

### Same Datacenter (< 10ms latency)

```rust
RaftConfig {
    election_timeout: (150, 300),
    heartbeat_interval: 50,
    ..Default::default()
}
```

This is the default configuration. The 150-300ms range provides fast failover
while tolerating typical intra-datacenter jitter.

### Cross-Datacenter (10-50ms latency)

```rust
RaftConfig {
    election_timeout: (500, 1000),
    heartbeat_interval: 150,
    ..Default::default()
}
```

The wider timeout window accounts for inter-datacenter latency spikes. The
heartbeat interval is set to roughly 3x the expected round-trip time.

### Geo-Distributed (> 50ms latency)

```rust
RaftConfig {
    election_timeout: (2000, 4000),
    heartbeat_interval: 500,
    ..Default::default()
}
```

Geo-distributed deployments need generous timeouts to avoid false leader
failures due to transient network delays.

### General Rule

The heartbeat interval should be at least 3x the maximum expected one-way
network latency. The minimum election timeout should be at least 3x the
heartbeat interval:

```text
heartbeat_interval >= 3 * max_one_way_latency
election_timeout_min >= 3 * heartbeat_interval
election_timeout_max >= 2 * election_timeout_min
```

## Cluster Sizing

Choose an odd number of nodes to avoid split-brain scenarios:

| Nodes | Quorum | Fault Tolerance | Recommendation |
| --- | --- | --- | --- |
| 3 | 2 | 1 failure | Minimum for production |
| 5 | 3 | 2 failures | Recommended |
| 7 | 4 | 3 failures | Large or critical deployments |

Avoid even numbers. A 4-node cluster tolerates only 1 failure (same as 3
nodes) but requires more resources.

## Fast-Path Validation

Fast-path lets followers skip expensive block validation when the block
embedding is similar to recent blocks from the same leader.

### Enable or Disable

```rust
RaftConfig {
    enable_fast_path: true,           // default
    similarity_threshold: 0.95,       // default
    ..Default::default()
}
```

### When to Adjust the Threshold

- **Higher threshold (0.98-0.99)**: More conservative, validates more blocks
  fully. Use when data integrity is paramount or workload patterns change
  frequently.
- **Lower threshold (0.85-0.90)**: More aggressive, skips validation more
  often. Use for homogeneous workloads where blocks are predictably similar.
- **Disable entirely**: Set `enable_fast_path: false` for maximum validation
  rigor. Use during testing or when debugging validation issues.

### Monitoring Fast-Path

Monitor `fast_path_rate` in `RaftStats`:

| Rate | Interpretation |
| --- | --- |
| > 0.80 | Normal, fast-path working well |
| 0.50 - 0.80 | Workload is moderately diverse |
| 0.20 - 0.50 | Workload is very diverse, consider disabling fast-path |
| < 0.20 | Fast-path rarely helps, consider disabling |

## Pre-Vote Configuration

Pre-vote prevents disruptive elections from partitioned nodes. It is enabled
by default and should remain enabled in production:

```rust
RaftConfig {
    enable_pre_vote: true,  // default, strongly recommended
    ..Default::default()
}
```

Disable only for testing or debugging election behavior:

```rust
RaftConfig {
    enable_pre_vote: false,  // testing only
    ..Default::default()
}
```

## Geometric Tie-Breaking

When candidates have equally up-to-date logs, geometric tie-breaking prefers
the candidate whose state embedding is closest to the cluster centroid:

```rust
RaftConfig {
    enable_geometric_tiebreak: true,        // default
    geometric_tiebreak_threshold: 0.3,      // default
    ..Default::default()
}
```

A higher threshold (0.5-0.7) means the tie-breaking only activates when
candidates are very geometrically similar, reducing its effect. A lower
threshold (0.1-0.2) makes it activate more often.

## Snapshot Configuration

Snapshots compact the Raft log to prevent unbounded growth.

### Snapshot Threshold

The number of log entries to accumulate before creating a snapshot:

```rust
RaftConfig {
    snapshot_threshold: 10_000,        // default
    snapshot_trailing_logs: 100,       // entries kept after snapshot
    snapshot_chunk_size: 1_048_576,    // 1MB per chunk during transfer
    snapshot_max_memory: 268_435_456,  // 256MB max for snapshot buffering
    ..Default::default()
}
```

### Tuning Snapshot Frequency

- **Lower threshold (1,000-5,000)**: More frequent snapshots, less WAL
  storage, but more CPU overhead. Use when disk space is limited.
- **Higher threshold (50,000-100,000)**: Less frequent snapshots, more WAL
  storage, faster steady-state. Use when disk is plentiful and followers
  rarely fall far behind.

### Compaction Cooldown

Prevents excessive snapshot creation during high-throughput bursts:

```rust
RaftConfig {
    compaction_check_interval: 10,     // ticks between checks
    compaction_cooldown_ms: 60_000,    // 60s minimum between compactions
    ..Default::default()
}
```

## Leadership Transfer

Graceful leadership transfer allows an operator to move the leader role to
a specific node (for maintenance or load balancing):

```rust
RaftConfig {
    transfer_timeout_ms: 1_000,  // 1s default, abort if transfer stalls
    ..Default::default()
}
```

Increase this timeout for geo-distributed deployments where the target node
may need time to catch up on log entries.

## Heartbeat Monitoring

The `QuorumTracker` monitors heartbeat responses:

```rust
RaftConfig {
    auto_heartbeat: true,             // spawn heartbeat task on election
    max_heartbeat_failures: 3,        // warn after 3 consecutive failures
    ..Default::default()
}
```

If `heartbeat_success_rate` drops below 0.95, investigate network health.
Below 0.80 is critical and may indicate an imminent quorum loss.

## Security: Message Signing

For production clusters, enable Ed25519 message signing in `GossipConfig`
(gossip messages are the primary vector for Byzantine attacks):

```rust
GossipConfig {
    require_signatures: true,
    max_message_age_ms: 300_000,  // 5 min, prevents replay attacks
    ..Default::default()
}
```

## Complete Production Example

A production-ready configuration for a 5-node cross-datacenter cluster:

```rust
let raft_config = RaftConfig {
    election_timeout: (500, 1000),
    heartbeat_interval: 150,
    enable_fast_path: true,
    similarity_threshold: 0.95,
    enable_pre_vote: true,
    enable_geometric_tiebreak: true,
    geometric_tiebreak_threshold: 0.3,
    snapshot_threshold: 10_000,
    snapshot_trailing_logs: 100,
    snapshot_chunk_size: 1_048_576,
    compaction_cooldown_ms: 60_000,
    snapshot_max_memory: 268_435_456,
    auto_heartbeat: true,
    max_heartbeat_failures: 3,
    transfer_timeout_ms: 2_000,
    ..Default::default()
};
```
