# Split-Brain Recovery

## What is Split-Brain?

A network partition where multiple nodes believe they are the leader,
potentially accepting conflicting writes.

## Symptoms

- Multiple nodes reporting `raft_state="leader"` in metrics
- Clients seeing different data depending on which node they connect to
- `tensor_chain_partition_detected` metric > 0
- Gossip reporting different membership views

## How tensor_chain Prevents Split-Brain

Raft consensus requires majority quorum:

- 3 nodes: 2 required (only 1 partition can have leader)
- 5 nodes: 3 required

Split-brain can only occur with **symmetric partition** where old leader is
isolated but doesn't realize it.

## Automatic Recovery (Partition Merge Protocol)

When partitions heal, tensor_chain automatically reconciles:

### Phase 1: Detection

- Gossip detects new reachable nodes
- Compare Raft terms and log lengths

### Phase 2: Leader Resolution

- Higher term wins
- If same term, longer log wins
- Losing leader steps down

### Phase 3: State Reconciliation

- Semantic conflict detection on divergent entries
- Orthogonal changes: vector-add merge
- Conflicting changes: reject newer (requires manual resolution)

### Phase 4: Log Synchronization

- Follower truncates divergent suffix
- Leader replicates correct entries

### Phase 5: Membership Merge

- Gossip merges LWW membership states
- Higher incarnation wins for each node

### Phase 6: Checkpoint

- Create snapshot post-merge for fast recovery

## Manual Intervention (When Automatic Fails)

### Scenario: Conflicting Writes

```bash
# 1. Identify conflicts
neumann-admin conflicts list --since "2h ago"

# 2. Export conflicting transactions
neumann-admin conflicts export --tx-id 12345 --output conflict.json

# 3. Choose resolution
neumann-admin conflicts resolve --tx-id 12345 --keep-version A

# 4. Or merge manually
neumann-admin conflicts resolve --tx-id 12345 --merge-custom merge.json
```

### Scenario: Completely Diverged State

```bash
# 1. Stop all nodes
systemctl stop neumann

# 2. Identify authoritative node (longest log, highest term)
for node in node1 node2 node3; do
  ssh $node "neumann-admin raft-info"
done

# 3. On non-authoritative nodes, clear state
rm -rf /var/lib/neumann/raft/*

# 4. Restart authoritative node first
systemctl start neumann

# 5. Restart other nodes (will sync from leader)
systemctl start neumann
```

## Post-Recovery Verification

```bash
# Verify single leader
curl -s http://node1:9090/metrics | grep 'raft_state{state="leader"}'

# Verify all nodes in sync
neumann-admin cluster-status

# Check for unresolved conflicts
neumann-admin conflicts list

# Verify recent transactions
neumann-admin tx-log --last 100
```

## Prevention

1. **Network design**: Avoid symmetric partitions
2. **Monitoring**: Alert on partition detection
3. **Testing**: Regularly run chaos engineering tests
4. **Backups**: Regular snapshots enable point-in-time recovery
