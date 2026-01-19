# Troubleshooting

## Common Issues

### Node Won't Start

**Symptom**: Node exits immediately or fails to bind

**Check**:
```bash
# Port already in use
lsof -i :7878
lsof -i :9090

# Permissions
ls -la /var/lib/neumann

# Config syntax
neumann --config /etc/neumann/config.toml --validate
```

**Solutions**:
- Kill conflicting process
- Fix directory permissions: `chown -R neumann:neumann /var/lib/neumann`
- Fix config syntax errors

### Can't Connect to Cluster

**Symptom**: Client connections timeout

**Check**:
```bash
# Network connectivity
nc -zv node1 7878

# Firewall rules
iptables -L -n | grep 7878

# Node health
curl http://node1:9090/health
```

**Solutions**:
- Open firewall ports 7878, 7879, 9090
- Check DNS resolution
- Verify node is running

### Slow Performance

**Symptom**: High latency, low throughput

**Check**:
```bash
# Metrics
curl http://node1:9090/metrics | grep -E "(latency|throughput)"

# Disk I/O
iostat -x 1

# Memory
free -h

# CPU
top -p $(pgrep neumann)
```

**Solutions**:
- Increase memory allocation
- Use faster storage (NVMe)
- Tune Raft parameters
- Add more nodes for read scaling

### Data Inconsistency

**Symptom**: Different nodes return different data

**Check**:
```bash
# Compare commit indices
for node in node1 node2 node3; do
  curl -s http://$node:9090/metrics | grep commit_index
done

# Check for partitions
neumann-admin cluster-status
```

**Solutions**:
- Wait for replication to catch up
- Check network connectivity
- Follow [split-brain runbook](runbooks/split-brain.md) if partitioned

### High Memory Usage

**Symptom**: OOM kills, swap usage

**Check**:
```bash
# Memory breakdown
curl http://node1:9090/metrics | grep memory

# Process memory
ps aux | grep neumann
```

**Solutions**:
- Increase `max_memory_mb` config
- Trigger snapshot to reduce log size
- Add more nodes to distribute load

### WAL Growing Too Large

**Symptom**: Disk filling up

**Check**:
```bash
# WAL size
du -sh /var/lib/neumann/wal/

# Snapshot status
ls -la /var/lib/neumann/snapshots/
```

**Solutions**:
- Trigger manual snapshot: `curl -X POST http://node:9090/admin/snapshot`
- Reduce `snapshot_interval`
- Add more disk space

## Debug Logging

Enable detailed logging:

```bash
RUST_LOG=tensor_chain=debug,tower=warn neumann
```

For specific modules:

```bash
RUST_LOG=tensor_chain::raft=trace,tensor_chain::gossip=debug neumann
```

## Getting Help

1. Check the [runbooks](runbooks/index.md) for specific scenarios
2. Search [GitHub issues](https://github.com/Shadylukin/Neumann/issues)
3. Open a new issue with:
   - Neumann version
   - Configuration (redact secrets)
   - Relevant logs
   - Steps to reproduce
