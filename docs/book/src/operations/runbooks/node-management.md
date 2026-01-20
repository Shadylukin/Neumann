# Node Management

This runbook covers adding and removing nodes from a tensor_chain cluster.

## Adding a Node

### Prerequisites Checklist

- [ ] New node has network connectivity to existing cluster members
- [ ] TLS certificates are configured (if using TLS)
- [ ] Node has sufficient disk space for snapshot transfer
- [ ] Firewall rules allow traffic on cluster port (default: 9100)
- [ ] DNS/hostname resolution configured for the new node

### Symptoms (Why Add a Node)

- Cluster capacity insufficient for workload
- Need additional replicas for fault tolerance
- Geographic distribution requirements
- Performance scaling requirements

### Procedure

#### Step 1: Prepare the new node

```bash
# Install Neumann on the new node
cargo install neumann --version X.Y.Z

# Create configuration directory
mkdir -p /etc/neumann
mkdir -p /var/lib/neumann/data

# Copy TLS certificates (if using TLS)
scp admin@existing-node:/etc/neumann/ca.crt /etc/neumann/
# Generate node-specific certificates
./scripts/generate-node-cert.sh node4
```

#### Step 2: Configure the new node

Create `/etc/neumann/config.toml`:

```toml
[node]
id = "node4"
data_dir = "/var/lib/neumann/data"

[cluster]
# Existing cluster members for initial discovery
seeds = ["node1:9100", "node2:9100", "node3:9100"]
port = 9100

[tls]
cert_path = "/etc/neumann/node4.crt"
key_path = "/etc/neumann/node4.key"
ca_cert_path = "/etc/neumann/ca.crt"
```

#### Step 3: Join the cluster

```bash
# Start the node in join mode
neumann start --join

# Monitor the join process
neumann status --watch
```

#### Step 4: Verify cluster membership

```bash
# On any existing node
neumann cluster members

# Expected output:
# ID     ADDRESS       STATE     ROLE
# node1  10.0.1.1:9100 healthy   leader
# node2  10.0.1.2:9100 healthy   follower
# node3  10.0.1.3:9100 healthy   follower
# node4  10.0.1.4:9100 healthy   follower  <-- new node
```

### Post-Addition Verification

```bash
# Verify snapshot transfer completed
neumann status node4 --verbose

# Check replication lag
neumann metrics node4 | grep replication_lag

# Verify the node participates in consensus
neumann raft status
```

## Removing a Node

### Prerequisites Checklist

- [ ] Cluster will maintain quorum after removal
- [ ] Node is not the current leader (trigger election first)
- [ ] Data has been replicated to other nodes
- [ ] No in-flight transactions involving this node

### Symptoms (Why Remove a Node)

- Hardware failure requiring decommission
- Cluster right-sizing
- Node relocation to different region
- Maintenance requiring extended downtime

### Pre-Removal Verification

```bash
# Check current cluster state
neumann cluster members

# Verify quorum will be maintained
# For N nodes, quorum = (N/2) + 1
# 5 nodes -> quorum = 3, can remove 2
# 3 nodes -> quorum = 2, can remove 1
```

### Procedure

#### Step 1: Drain the node (graceful removal)

```bash
# Mark node as draining (stops accepting new requests)
neumann node drain node3

# Wait for in-flight transactions to complete
neumann node wait-drain node3 --timeout 300
```

#### Step 2: Transfer leadership if necessary

```bash
# Check if node is leader
neumann raft status

# If leader, trigger election
neumann raft transfer-leadership --to node1
```

#### Step 3: Remove from cluster

```bash
# Remove the node from cluster configuration
neumann cluster remove node3

# Verify removal
neumann cluster members
```

#### Step 4: Stop the node

```bash
# On the removed node
neumann stop

# Clean up data (optional)
rm -rf /var/lib/neumann/data/*
```

### Post-Removal Verification

```bash
# Verify cluster health
neumann cluster health

# Check that remaining nodes have correct membership
neumann cluster members

# Verify no pending transactions for removed node
neumann transactions pending
```

## Emergency Removal

Use emergency removal only when a node is unresponsive and cannot be drained
gracefully.

### Symptoms

- Node is unreachable (network partition, hardware failure)
- Node is unresponsive (hung process, resource exhaustion)
- Need to restore quorum quickly

### Procedure

```bash
# Force remove unresponsive node
neumann cluster remove node3 --force

# The cluster will:
# 1. Remove node from membership
# 2. Abort any transactions involving the node
# 3. Re-elect leader if necessary
```

### Resolution

After emergency removal:

1. Investigate root cause of node failure
2. Repair or replace hardware if needed
3. Re-add node using the addition procedure above

### Prevention

- Monitor node health with alerting
- Configure appropriate timeouts
- Maintain sufficient cluster size for fault tolerance

## Quorum Considerations

| Cluster Size | Quorum | Fault Tolerance | Notes |
| --- | --- | --- | --- |
| 1 | 1 | 0 | Development only |
| 2 | 2 | 0 | Not recommended |
| 3 | 2 | 1 | Minimum for production |
| 5 | 3 | 2 | Recommended for HA |
| 7 | 4 | 3 | Maximum practical size |

### Quorum Formula

```text
quorum = (cluster_size / 2) + 1
fault_tolerance = cluster_size - quorum
```

### Best Practices

- Always maintain odd number of nodes
- Never remove nodes if it would violate quorum
- Plan node additions/removals during low-traffic periods
- Test failover scenarios regularly

## See Also

- [Cluster Upgrade](cluster-upgrade.md)
- [Leader Election](leader-election.md)
- [Node Recovery](node-recovery.md)
