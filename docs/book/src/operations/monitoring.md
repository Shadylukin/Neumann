# Monitoring

## Metrics Endpoint

Prometheus metrics are exposed at `http://node:9090/metrics`.

## Key Metrics

### Raft Consensus

| Metric | Type | Description |
|--------|------|-------------|
| `tensor_chain_raft_state` | Gauge | Current state (follower=0, candidate=1, leader=2) |
| `tensor_chain_term` | Gauge | Current Raft term |
| `tensor_chain_commit_index` | Gauge | Highest committed log index |
| `tensor_chain_applied_index` | Gauge | Highest applied log index |
| `tensor_chain_elections_total` | Counter | Total elections started |
| `tensor_chain_append_entries_total` | Counter | Total AppendEntries RPCs |

### Transactions

| Metric | Type | Description |
|--------|------|-------------|
| `tensor_chain_tx_active` | Gauge | Currently active transactions |
| `tensor_chain_tx_commits_total` | Counter | Total committed transactions |
| `tensor_chain_tx_aborts_total` | Counter | Total aborted transactions |
| `tensor_chain_tx_latency_seconds` | Histogram | Transaction latency |

### Deadlock Detection

| Metric | Type | Description |
|--------|------|-------------|
| `tensor_chain_deadlocks_total` | Counter | Total deadlocks detected |
| `tensor_chain_deadlock_victims_total` | Counter | Transactions aborted as victims |
| `tensor_chain_wait_graph_size` | Gauge | Current wait-for graph size |

### Gossip

| Metric | Type | Description |
|--------|------|-------------|
| `tensor_chain_gossip_members` | Gauge | Known cluster members |
| `tensor_chain_gossip_healthy` | Gauge | Healthy members |
| `tensor_chain_gossip_suspect` | Gauge | Suspect members |
| `tensor_chain_gossip_failed` | Gauge | Failed members |

### Storage

| Metric | Type | Description |
|--------|------|-------------|
| `tensor_chain_entries_total` | Gauge | Total stored entries |
| `tensor_chain_memory_bytes` | Gauge | Memory usage |
| `tensor_chain_disk_bytes` | Gauge | Disk usage |
| `tensor_chain_wal_size_bytes` | Gauge | WAL file size |

## Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'neumann'
    static_configs:
      - targets:
        - 'node1:9090'
        - 'node2:9090'
        - 'node3:9090'
```

## Grafana Dashboard

Import the dashboard from `deploy/grafana/neumann-dashboard.json`.

Panels include:
- Cluster overview (leader, term, members)
- Transaction throughput and latency
- Replication lag
- Memory and disk usage
- Deadlock rate

## Alerting Rules

See `docs/book/src/operations/runbooks/` for alert definitions.

```yaml
groups:
  - name: neumann
    rules:
      - alert: NoLeader
        expr: sum(tensor_chain_raft_state{state="leader"}) == 0
        for: 30s
        labels:
          severity: critical

      - alert: HighReplicationLag
        expr: tensor_chain_commit_index - tensor_chain_applied_index > 1000
        for: 1m
        labels:
          severity: warning

      - alert: HighDeadlockRate
        expr: rate(tensor_chain_deadlocks_total[5m]) > 1
        for: 5m
        labels:
          severity: warning
```

## Health Endpoint

```bash
curl http://node:9090/health
```

Response:
```json
{
  "status": "healthy",
  "raft_state": "leader",
  "term": 42,
  "commit_index": 12345,
  "members": 3,
  "healthy_members": 3
}
```

## Logging

Configure log level:

```bash
RUST_LOG=tensor_chain=debug neumann
```

Log levels: `error`, `warn`, `info`, `debug`, `trace`
