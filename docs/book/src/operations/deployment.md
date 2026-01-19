# Deployment

## Single Node

For development and testing:

```bash
neumann --data-dir ./data
```

## Cluster Deployment

### Prerequisites

- 3, 5, or 7 nodes (odd number for quorum)
- Network connectivity between nodes
- Synchronized clocks (NTP)

### Configuration

Each node needs a config file:

```toml
# /etc/neumann/config.toml

[node]
id = "node1"
data_dir = "/var/lib/neumann"
bind_address = "0.0.0.0:7878"

[cluster]
peers = [
    "node2:7878",
    "node3:7878",
]

[raft]
election_timeout_min_ms = 150
election_timeout_max_ms = 300
heartbeat_interval_ms = 50

[gossip]
bind_address = "0.0.0.0:7879"
ping_interval_ms = 1000
```

### Starting the Cluster

```bash
# Start first node (will become leader)
neumann --config /etc/neumann/config.toml --bootstrap

# Start remaining nodes
neumann --config /etc/neumann/config.toml
```

### Verify Cluster Health

```bash
# Check cluster status
curl http://node1:9090/health

# View membership
neumann-admin cluster-status
```

## Docker Compose

```yaml
version: '3.8'
services:
  node1:
    image: neumann/neumann:latest
    environment:
      - NEUMANN_NODE_ID=node1
      - NEUMANN_PEERS=node2:7878,node3:7878
    ports:
      - "7878:7878"
      - "9090:9090"
    volumes:
      - node1-data:/var/lib/neumann

  node2:
    image: neumann/neumann:latest
    environment:
      - NEUMANN_NODE_ID=node2
      - NEUMANN_PEERS=node1:7878,node3:7878
    volumes:
      - node2-data:/var/lib/neumann

  node3:
    image: neumann/neumann:latest
    environment:
      - NEUMANN_NODE_ID=node3
      - NEUMANN_PEERS=node1:7878,node2:7878
    volumes:
      - node3-data:/var/lib/neumann

volumes:
  node1-data:
  node2-data:
  node3-data:
```

## Kubernetes

See the Helm chart in `deploy/helm/neumann/`.

```bash
helm install neumann ./deploy/helm/neumann \
  --set replicas=3 \
  --set persistence.size=100Gi
```

## Production Checklist

- [ ] Odd number of nodes (3, 5, or 7)
- [ ] Nodes in separate availability zones
- [ ] NTP configured and synchronized
- [ ] Firewall rules for ports 7878, 7879, 9090
- [ ] Monitoring and alerting configured
- [ ] Backup strategy in place
- [ ] Resource limits set appropriately
