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

Each node is configured via environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `NEUMANN_CLUSTER_NODE_ID` | Unique node identifier | `node1` |
| `NEUMANN_CLUSTER_BIND_ADDR` | Raft/cluster listen address | `0.0.0.0:9300` |
| `NEUMANN_CLUSTER_PEERS` | Comma-separated peer list | `node2=10.0.0.2:9300,node3=10.0.0.3:9300` |
| `NEUMANN_DATA_DIR` | Data directory path | `/var/lib/neumann` |
| `NEUMANN_BIND_ADDR` | gRPC API listen address | `0.0.0.0:9200` |
| `RUST_LOG` | Log level filter | `info` |

Peer format is `node_id=SocketAddr` where `SocketAddr` is an IP:port pair
(DNS hostnames are not supported in the peer list).

### Starting the Cluster

```bash
# On each node, set environment and start
export NEUMANN_CLUSTER_NODE_ID=node1
export NEUMANN_CLUSTER_BIND_ADDR=0.0.0.0:9300
export NEUMANN_CLUSTER_PEERS=node2=10.0.0.2:9300,node3=10.0.0.3:9300
export NEUMANN_DATA_DIR=/var/lib/neumann
neumann_server
```

### Verify Cluster Health

The server exposes a gRPC `Health.Check` service on port 9200:

```bash
grpcurl -plaintext localhost:9200 grpc.health.v1.Health/Check
```

## Docker Compose

```yaml
version: '3.8'
services:
  node1:
    image: shadylukinack/neumann:latest
    environment:
      - NEUMANN_CLUSTER_NODE_ID=node1
      - NEUMANN_CLUSTER_BIND_ADDR=0.0.0.0:9300
      - NEUMANN_CLUSTER_PEERS=node2=node2:9300,node3=node3:9300
      - NEUMANN_BIND_ADDR=0.0.0.0:9200
      - NEUMANN_DATA_DIR=/var/lib/neumann
    ports:
      - "9200:9200"
    volumes:
      - node1-data:/var/lib/neumann

  node2:
    image: shadylukinack/neumann:latest
    environment:
      - NEUMANN_CLUSTER_NODE_ID=node2
      - NEUMANN_CLUSTER_BIND_ADDR=0.0.0.0:9300
      - NEUMANN_CLUSTER_PEERS=node1=node1:9300,node3=node3:9300
      - NEUMANN_DATA_DIR=/var/lib/neumann
    volumes:
      - node2-data:/var/lib/neumann

  node3:
    image: shadylukinack/neumann:latest
    environment:
      - NEUMANN_CLUSTER_NODE_ID=node3
      - NEUMANN_CLUSTER_BIND_ADDR=0.0.0.0:9300
      - NEUMANN_CLUSTER_PEERS=node1=node1:9300,node2=node2:9300
      - NEUMANN_DATA_DIR=/var/lib/neumann
    volumes:
      - node3-data:/var/lib/neumann

volumes:
  node1-data:
  node2-data:
  node3-data:
```

## Kubernetes

Kustomize-based manifests are in `deploy/k8s/`. See `deploy/k8s/README.md`
for full documentation.

### Dev Deployment

```bash
kubectl apply -k deploy/k8s/overlays/dev/
kubectl -n neumann get pods -w
```

### Production Deployment

```bash
kubectl apply -k deploy/k8s/overlays/production/
```

The production overlay adds server-side TLS (via cert-manager), higher
resource limits, and 50Gi persistent volumes.

### Connecting

```bash
kubectl -n neumann port-forward svc/neumann 9200:9200
grpcurl -plaintext localhost:9200 grpc.health.v1.Health/Check
```

## Production Checklist

- [ ] Odd number of nodes (3, 5, or 7)
- [ ] Nodes in separate availability zones
- [ ] NTP configured and synchronized
- [ ] Firewall rules for ports 9200 (gRPC API) and 9300 (Raft)
- [ ] Monitoring and alerting configured
- [ ] Backup strategy in place
- [ ] Resource limits set appropriately
