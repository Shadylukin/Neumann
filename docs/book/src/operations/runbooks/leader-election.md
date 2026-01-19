# Leader Election Failures

## Symptoms

- `NoLeader` alert firing
- Continuous election attempts in logs
- Client requests timing out with "no leader" errors
- `tensor_chain_elections_total` metric increasing rapidly

## Diagnostic Commands

### Check Raft State

```bash
# Query each node's state
for node in node1 node2 node3; do
  curl -s http://$node:9090/metrics | grep tensor_chain_raft_state
done
```

### Inspect Logs

```bash
# Look for election-related entries
grep -E "(election|vote|term)" /var/log/neumann/tensor_chain.log | tail -100
```

### Verify Network Connectivity

```bash
# From each node, verify connectivity to peers
for peer in node1 node2 node3; do
  nc -zv $peer 7878 2>&1 | grep -v "Connection refused" || echo "FAIL: $peer"
done
```

## Root Causes

### 1. Network Partition

**Diagnosis**: Nodes can't reach each other

**Solution**:
- Check firewall rules for port 7878 (Raft) and 7879 (gossip)
- Verify network routes between nodes
- Check for packet loss: `ping -c 100 peer_node`

### 2. Clock Skew

**Diagnosis**: Election timeouts inconsistent across nodes

**Solution**:
- Ensure NTP is running: `timedatectl status`
- Max recommended skew: 500ms
- Sync clocks: `chronyc makestep`

### 3. Quorum Loss

**Diagnosis**: Fewer than `(n/2)+1` nodes available

**Solution**:
- For 3-node cluster: need 2 nodes
- For 5-node cluster: need 3 nodes
- Bring failed nodes back online or add new nodes

### 4. Election Timeout Too Aggressive

**Diagnosis**: Frequent elections even with healthy network

**Solution**:
```toml
[raft]
election_timeout_min_ms = 300   # Increase from default 150
election_timeout_max_ms = 600   # Increase from default 300
```

## Resolution Steps

1. **Identify partitioned nodes** using gossip membership view
2. **Restore connectivity** if network issue
3. **If quorum lost**, follow disaster recovery procedure
4. **Monitor** `tensor_chain_raft_state{state="leader"}` for leader emergence

## Alerting Rule

```yaml
- alert: NoLeader
  expr: sum(tensor_chain_raft_state{state="leader"}) == 0
  for: 30s
  labels:
    severity: critical
  annotations:
    summary: "No Raft leader elected in cluster"
    runbook_url: "https://docs.neumann.io/operations/runbooks/leader-election"
```

## Prevention

- Deploy odd number of nodes (3, 5, 7)
- Use separate availability zones
- Monitor `tensor_chain_elections_total` rate
- Set up network monitoring between nodes
