# Capacity Planning

## Resource Requirements

### Memory

| Component | Formula | Example (1M entries) |
|-----------|---------|----------------------|
| Raft log (in-memory) | `entries * avg_size * 2` | 1M * 1KB * 2 = 2 GB |
| Tensor store index | `entries * 64 bytes` | 1M * 64 = 64 MB |
| HNSW index | `vectors * dim * 4 * ef` | 1M * 128 * 4 * 16 = 8 GB |
| Codebook | `centroids * dim * 4` | 1024 * 128 * 4 = 512 KB |
| Connection buffers | `peers * buffer_size` | 10 * 64KB = 640 KB |

**Recommended minimum**: 16 GB for production

### Disk

| Component | Formula | Example |
|-----------|---------|---------|
| WAL | `entries * avg_size` | 10M * 1KB = 10 GB |
| Snapshots | `state_size * 2` | 5 GB * 2 = 10 GB |
| Mmap cold storage | `cold_entries * avg_size` | 100M * 1KB = 100 GB |

**Recommended**: 3x expected data size for growth

### Network

| Traffic Type | Formula | Example (100 TPS) |
|--------------|---------|-------------------|
| Replication | `TPS * entry_size * (replicas-1)` | 100 * 1KB * 2 = 200 KB/s |
| Gossip | `nodes * fanout * state_size / interval` | 5 * 3 * 1KB / 1s = 15 KB/s |
| Client | `TPS * (request + response)` | 100 * 2KB = 200 KB/s |

**Recommended**: 1 Gbps minimum, 10 Gbps for high throughput

### CPU

| Operation | Complexity | Cores Needed |
|-----------|------------|--------------|
| Consensus | O(1) per entry | 1 core |
| Embedding computation | O(dim) | 1-2 cores |
| HNSW search | O(log N * ef) | 2-4 cores |
| Conflict detection | O(concurrent_txs^2) | 1 core |

**Recommended**: 8+ cores for production

## Sizing Examples

### Small (Dev/Test)
- 3 nodes
- 4 cores, 8 GB RAM, 100 GB SSD each
- Up to 1M entries, 10 TPS

### Medium (Production)
- 5 nodes
- 8 cores, 32 GB RAM, 500 GB NVMe each
- Up to 100M entries, 1000 TPS

### Large (High-Scale)
- 7+ nodes
- 16+ cores, 64+ GB RAM, 2 TB NVMe each
- 1B+ entries, 10k+ TPS
- Consider sharding

## Scaling Strategies

### Vertical Scaling

When to use:
- Single-node bottleneck (CPU, memory)
- Read latency requirements

Actions:
- Add RAM for larger in-memory log
- Add cores for parallel embedding computation
- Upgrade to NVMe for faster snapshots

### Horizontal Scaling

When to use:
- Throughput limited by consensus
- Fault tolerance requirements

Actions:
- Add read replicas (don't participate in consensus)
- Add consensus members (odd numbers only)
- Implement sharding by key range

## Monitoring for Capacity

```yaml
# Prometheus alerts
- alert: HighMemoryUsage
  expr: tensor_chain_memory_usage_bytes / tensor_chain_memory_limit_bytes > 0.85
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Memory usage above 85%"

- alert: DiskSpaceLow
  expr: tensor_chain_disk_free_bytes < 10737418240  # 10 GB
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Less than 10 GB disk space remaining"

- alert: HighCPUUsage
  expr: rate(tensor_chain_cpu_seconds_total[5m]) > 0.9
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "CPU usage above 90%"
```

## Growth Projections

```bash
# Calculate daily growth
neumann-admin stats --since "7d ago" --format json | jq '.entries_per_day'

# Project storage needs
DAILY_GROWTH=100000  # entries
ENTRY_SIZE=1024      # bytes
DAYS=365
GROWTH=$((DAILY_GROWTH * ENTRY_SIZE * DAYS / 1024 / 1024 / 1024))
echo "Projected annual growth: ${GROWTH} GB"
```
