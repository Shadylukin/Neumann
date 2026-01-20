# Example Configurations

This page provides complete configuration examples for different deployment
scenarios.

## Development (Single Node)

Minimal configuration for local development and testing.

```toml
[node]
id = "dev-node"
data_dir = "./data"

[cluster]
# Single node cluster - no seeds needed
seeds = []
port = 9100

# Disable TLS for development
[tls]
enabled = false

# Minimal rate limiting
[rate_limit]
enabled = false

# No compression for easier debugging
[compression]
enabled = false

# Shorter timeouts for faster feedback
[transactions]
timeout_ms = 1000
lock_timeout_ms = 500

# Verbose logging
[logging]
level = "debug"
format = "pretty"
```

## Production (3-Node Cluster)

Standard production configuration with TLS, rate limiting, and tuned
timeouts.

```toml
# === Node Configuration ===
[node]
id = "node1"
data_dir = "/var/lib/neumann/data"
# Bind to all interfaces
bind_address = "0.0.0.0"

# === Cluster Configuration ===
[cluster]
seeds = ["node1.example.com:9100", "node2.example.com:9100", "node3.example.com:9100"]
port = 9100
# Cluster name for identification
name = "production"

# === TLS Configuration ===
[tls]
enabled = true
cert_path = "/etc/neumann/node1.crt"
key_path = "/etc/neumann/node1.key"
ca_cert_path = "/etc/neumann/ca.crt"
# Require mutual TLS
require_client_auth = true
# Verify node identity matches certificate
node_id_verification = "CommonName"

# === TCP Transport ===
[tcp]
# Connections per peer
pool_size = 4
# Connection timeout
connect_timeout_ms = 5000
# Read/write timeout
io_timeout_ms = 30000
# Enable keepalive
keepalive = true
keepalive_interval_secs = 30
# Maximum message size (16 MB)
max_message_size = 16777216
# Outbound queue size
max_pending_messages = 1000

# === Rate Limiting ===
[rate_limit]
enabled = true
# Burst capacity
bucket_size = 100
# Tokens per second
refill_rate = 50.0

# === Compression ===
[compression]
enabled = true
method = "Lz4"
# Only compress messages > 256 bytes
min_size = 256

# === Transactions ===
[transactions]
# Transaction timeout
timeout_ms = 5000
# Lock timeout
lock_timeout_ms = 30000
# Default embedding dimension
embedding_dimension = 128

# === Conflict Detection ===
[consensus]
# Similarity threshold for conflict
conflict_threshold = 0.5
# Threshold for orthogonal merge
orthogonal_threshold = 0.1
# Merge window
merge_window_ms = 60000

# === Deadlock Detection ===
[deadlock]
enabled = true
detection_interval_ms = 100
max_cycle_length = 10

# === Snapshots ===
[snapshots]
# Memory threshold before disk spill
max_memory_bytes = 268435456  # 256 MB
# Snapshot interval
interval_secs = 3600
# Retention count
retain_count = 3

# === Metrics ===
[metrics]
enabled = true
# Prometheus endpoint
endpoint = "0.0.0.0:9090"
# Include detailed histograms
detailed = true

# === Logging ===
[logging]
level = "info"
format = "json"
# Log to file
file = "/var/log/neumann/neumann.log"
# Rotate logs
max_size_mb = 100
max_files = 10
```

## High-Throughput (5-Node)

Optimized configuration for maximum write throughput.

```toml
[node]
id = "node1"
data_dir = "/var/lib/neumann/data"

[cluster]
seeds = [
    "node1.example.com:9100",
    "node2.example.com:9100",
    "node3.example.com:9100",
    "node4.example.com:9100",
    "node5.example.com:9100",
]
port = 9100
name = "high-throughput"

# === TLS (same as production) ===
[tls]
enabled = true
cert_path = "/etc/neumann/node1.crt"
key_path = "/etc/neumann/node1.key"
ca_cert_path = "/etc/neumann/ca.crt"
require_client_auth = true

# === TCP - Optimized for throughput ===
[tcp]
# More connections for parallelism
pool_size = 8
# Shorter timeouts for faster failover
connect_timeout_ms = 2000
io_timeout_ms = 10000
keepalive = true
keepalive_interval_secs = 15
# Larger message size for batching
max_message_size = 67108864  # 64 MB
# Larger queues for buffering
max_pending_messages = 5000
recv_buffer_size = 5000

# === Rate Limiting - Permissive ===
[rate_limit]
enabled = true
bucket_size = 500
refill_rate = 250.0

# === Compression - Aggressive ===
[compression]
enabled = true
method = "Lz4"
# Compress even small messages
min_size = 64

# === Transactions - Fast ===
[transactions]
timeout_ms = 2000
lock_timeout_ms = 5000
embedding_dimension = 64  # Smaller for speed

# === Consensus - Optimized ===
[consensus]
# Lower thresholds for more merging
conflict_threshold = 0.7
orthogonal_threshold = 0.2
merge_window_ms = 30000

# === Deadlock - Frequent checks ===
[deadlock]
enabled = true
detection_interval_ms = 50

# === Raft - Tuned for throughput ===
[raft]
# Batch more entries
max_entries_per_append = 1000
# Shorter election timeout
election_timeout_ms = 500
# Faster heartbeats
heartbeat_interval_ms = 100

# === Snapshots - Less frequent ===
[snapshots]
max_memory_bytes = 536870912  # 512 MB
interval_secs = 7200
retain_count = 2
```

## Geo-Distributed (Multi-Region)

Configuration for clusters spanning multiple geographic regions with higher
latency tolerance.

```toml
[node]
id = "node1-us-east"
data_dir = "/var/lib/neumann/data"
region = "us-east-1"

[cluster]
seeds = [
    "node1-us-east.example.com:9100",
    "node2-us-west.example.com:9100",
    "node3-eu-west.example.com:9100",
]
port = 9100
name = "geo-distributed"

# === TLS (same as production) ===
[tls]
enabled = true
cert_path = "/etc/neumann/node1-us-east.crt"
key_path = "/etc/neumann/node1-us-east.key"
ca_cert_path = "/etc/neumann/ca.crt"
require_client_auth = true

# === TCP - WAN optimized ===
[tcp]
pool_size = 4
# Longer timeouts for cross-region latency
connect_timeout_ms = 10000
io_timeout_ms = 60000
keepalive = true
# More frequent keepalives to detect failures
keepalive_interval_secs = 10
max_message_size = 16777216

# === Rate Limiting - Standard ===
[rate_limit]
enabled = true
bucket_size = 100
refill_rate = 50.0

# === Compression - Always on for WAN ===
[compression]
enabled = true
method = "Lz4"
min_size = 128

# === Transactions - Longer timeouts ===
[transactions]
# Higher timeout for cross-region coordination
timeout_ms = 15000
lock_timeout_ms = 60000
embedding_dimension = 128

# === Consensus - Relaxed for latency ===
[consensus]
conflict_threshold = 0.5
orthogonal_threshold = 0.1
# Longer merge window for slow convergence
merge_window_ms = 120000

# === Deadlock - Less frequent for WAN ===
[deadlock]
enabled = true
detection_interval_ms = 500

# === Raft - WAN tuned ===
[raft]
max_entries_per_append = 100
# Longer election timeout for WAN latency
election_timeout_ms = 3000
heartbeat_interval_ms = 500
# Enable pre-vote to prevent disruption during partitions
pre_vote = true

# === Snapshots ===
[snapshots]
max_memory_bytes = 268435456
interval_secs = 3600
retain_count = 5

# === Reconnection - Aggressive ===
[reconnection]
enabled = true
initial_backoff_ms = 500
max_backoff_ms = 60000
multiplier = 2.0
jitter = 0.2

# === Region awareness ===
[region]
# Prefer local reads
local_read_preference = true
# Region priority for leader election
priority = 1
```

## Configuration Reference

### Environment Variables

All configuration values can be overridden with environment variables:

```bash
NEUMANN_NODE_ID=node1
NEUMANN_CLUSTER_PORT=9100
NEUMANN_TLS_ENABLED=true
NEUMANN_LOGGING_LEVEL=debug
```

### Configuration Precedence

1. Environment variables (highest)
2. Command-line arguments
3. Configuration file
4. Default values (lowest)

## See Also

- [Configuration Reference](configuration.md)
- [Monitoring](monitoring.md)
- [Troubleshooting](troubleshooting.md)
