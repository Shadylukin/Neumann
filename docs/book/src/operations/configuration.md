# Configuration

## Configuration Sources

Configuration is loaded in order (later overrides earlier):

1. Default values
2. Config file (`/etc/neumann/config.toml`)
3. Environment variables (`NEUMANN_*`)
4. Command-line flags

## Config File Format

```toml
[node]
id = "node1"
data_dir = "/var/lib/neumann"
bind_address = "0.0.0.0:7878"

[cluster]
peers = ["node2:7878", "node3:7878"]

[raft]
election_timeout_min_ms = 150
election_timeout_max_ms = 300
heartbeat_interval_ms = 50
max_entries_per_append = 100
snapshot_interval = 10000

[gossip]
bind_address = "0.0.0.0:7879"
ping_interval_ms = 1000
ping_timeout_ms = 500
suspect_timeout_ms = 3000
indirect_ping_count = 3

[transaction]
prepare_timeout_ms = 5000
commit_timeout_ms = 5000
lock_timeout_ms = 5000
max_concurrent_tx = 1000

[deadlock]
enabled = true
detection_interval_ms = 100
victim_policy = "youngest"
auto_abort_victim = true

[storage]
max_memory_mb = 1024
wal_sync_mode = "fsync"
compression = "lz4"

[metrics]
enabled = true
bind_address = "0.0.0.0:9090"
```

## Environment Variables

| Variable | Config Path | Example |
| --- | --- | --- |
| `NEUMANN_NODE_ID` | `node.id` | `node1` |
| `NEUMANN_DATA_DIR` | `node.data_dir` | `/var/lib/neumann` |
| `NEUMANN_PEERS` | `cluster.peers` | `node2:7878,node3:7878` |
| `NEUMANN_LOG_LEVEL` | --- | `info` |

## Command-Line Flags

```bash
neumann \
  --config /etc/neumann/config.toml \
  --node-id node1 \
  --data-dir /var/lib/neumann \
  --bind 0.0.0.0:7878 \
  --bootstrap \
  --log-level debug
```

## Key Parameters

### Raft Tuning

| Parameter | Default | Tuning |
| --- | --- | --- |
| `election_timeout_min_ms` | 150 | Increase for high-latency networks |
| `election_timeout_max_ms` | 300 | Should be 2x min |
| `heartbeat_interval_ms` | 50 | Lower for faster failure detection |
| `snapshot_interval` | 10000 | Higher for less I/O, slower recovery |

### Transaction Tuning

| Parameter | Default | Tuning |
| --- | --- | --- |
| `prepare_timeout_ms` | 5000 | Increase for slow networks |
| `lock_timeout_ms` | 5000 | Lower to fail fast on contention |
| `max_concurrent_tx` | 1000 | Based on memory and CPU |

### Storage Tuning

| Parameter | Default | Tuning |
| --- | --- | --- |
| `max_memory_mb` | 1024 | Based on available RAM |
| `wal_sync_mode` | `fsync` | `none` for speed (data loss risk) |
| `compression` | `lz4` | `none` for speed, `zstd` for ratio |
