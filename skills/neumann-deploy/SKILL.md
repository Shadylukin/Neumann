---
name: neumann-deploy
description: Deploy and operate Neumann in production. Use when setting up single-node or cluster deployments, configuring Docker Compose or Kubernetes, or planning production infrastructure.
---

# Neumann Deployment Guide

## Single Node

The simplest deployment runs a single Neumann instance.

### Development

```bash
neumann --data-dir ./data
```

### With WAL for durability

```bash
neumann --data-dir ./data --wal-dir ./data/wal
```

The write-ahead log ensures durability across restarts. Without WAL, data
persists only through explicit checkpoints.

### Production single-node

```bash
NEUMANN_BIND_ADDR=0.0.0.0:9200 \
NEUMANN_DATA_DIR=/var/lib/neumann/data \
neumann
```

Bind to `0.0.0.0` to accept external connections. Use a dedicated data directory
with sufficient disk space.

## Cluster Setup

Neumann uses Raft consensus for distributed coordination. Clusters require an
odd number of nodes for proper quorum.

### Prerequisites

- Odd number of nodes: 3, 5, or 7 (not 2, 4, or 6).
- All nodes can reach each other on the Raft port (default 9300).
- NTP or equivalent time synchronization across all nodes.
- Same Neumann version on all nodes.

### Environment variables

| Variable | Description | Example |
|----------|-------------|---------|
| `NEUMANN_CLUSTER_NODE_ID` | Unique node identifier | `node-1` |
| `NEUMANN_CLUSTER_BIND_ADDR` | Raft transport bind address | `0.0.0.0:9300` |
| `NEUMANN_CLUSTER_PEERS` | Comma-separated peer list | `node-2=10.0.1.2:9300,node-3=10.0.1.3:9300` |
| `NEUMANN_BIND_ADDR` | gRPC API bind address | `0.0.0.0:9200` |
| `NEUMANN_DATA_DIR` | Persistent data directory | `/var/lib/neumann/data` |

### Example: 3-node cluster

**Node 1** (10.0.1.1):

```bash
NEUMANN_CLUSTER_NODE_ID=node-1 \
NEUMANN_CLUSTER_BIND_ADDR=0.0.0.0:9300 \
NEUMANN_CLUSTER_PEERS=node-2=10.0.1.2:9300,node-3=10.0.1.3:9300 \
NEUMANN_BIND_ADDR=0.0.0.0:9200 \
NEUMANN_DATA_DIR=/var/lib/neumann/data \
neumann
```

**Node 2** (10.0.1.2):

```bash
NEUMANN_CLUSTER_NODE_ID=node-2 \
NEUMANN_CLUSTER_BIND_ADDR=0.0.0.0:9300 \
NEUMANN_CLUSTER_PEERS=node-1=10.0.1.1:9300,node-3=10.0.1.3:9300 \
NEUMANN_BIND_ADDR=0.0.0.0:9200 \
NEUMANN_DATA_DIR=/var/lib/neumann/data \
neumann
```

**Node 3** (10.0.1.3):

```bash
NEUMANN_CLUSTER_NODE_ID=node-3 \
NEUMANN_CLUSTER_BIND_ADDR=0.0.0.0:9300 \
NEUMANN_CLUSTER_PEERS=node-1=10.0.1.1:9300,node-2=10.0.1.2:9300 \
NEUMANN_BIND_ADDR=0.0.0.0:9200 \
NEUMANN_DATA_DIR=/var/lib/neumann/data \
neumann
```

### Verify cluster health

```sql
CLUSTER STATUS
CLUSTER NODES
CLUSTER LEADER
```

## Docker Compose

### 3-node cluster

```yaml
version: "3.8"

services:
  neumann-1:
    image: neumann:latest
    environment:
      NEUMANN_CLUSTER_NODE_ID: node-1
      NEUMANN_CLUSTER_BIND_ADDR: "0.0.0.0:9300"
      NEUMANN_CLUSTER_PEERS: "node-2=neumann-2:9300,node-3=neumann-3:9300"
      NEUMANN_BIND_ADDR: "0.0.0.0:9200"
      NEUMANN_DATA_DIR: /data
    ports:
      - "9200:9200"
      - "9300:9300"
    volumes:
      - neumann-1-data:/data

  neumann-2:
    image: neumann:latest
    environment:
      NEUMANN_CLUSTER_NODE_ID: node-2
      NEUMANN_CLUSTER_BIND_ADDR: "0.0.0.0:9300"
      NEUMANN_CLUSTER_PEERS: "node-1=neumann-1:9300,node-3=neumann-3:9300"
      NEUMANN_BIND_ADDR: "0.0.0.0:9200"
      NEUMANN_DATA_DIR: /data
    ports:
      - "9201:9200"
      - "9301:9300"
    volumes:
      - neumann-2-data:/data

  neumann-3:
    image: neumann:latest
    environment:
      NEUMANN_CLUSTER_NODE_ID: node-3
      NEUMANN_CLUSTER_BIND_ADDR: "0.0.0.0:9300"
      NEUMANN_CLUSTER_PEERS: "node-1=neumann-1:9300,node-2=neumann-2:9300"
      NEUMANN_BIND_ADDR: "0.0.0.0:9200"
      NEUMANN_DATA_DIR: /data
    ports:
      - "9202:9200"
      - "9302:9300"
    volumes:
      - neumann-3-data:/data

volumes:
  neumann-1-data:
  neumann-2-data:
  neumann-3-data:
```

Start with:

```bash
docker compose up -d
```

Verify:

```bash
grpcurl -plaintext localhost:9200 grpc.health.v1.Health/Check
grpcurl -plaintext localhost:9201 grpc.health.v1.Health/Check
grpcurl -plaintext localhost:9202 grpc.health.v1.Health/Check
```

## Kubernetes

### Development overlay (Kustomize)

```yaml
# kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - statefulset.yaml
  - service.yaml
```

```yaml
# statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: neumann
spec:
  serviceName: neumann
  replicas: 3
  selector:
    matchLabels:
      app: neumann
  template:
    metadata:
      labels:
        app: neumann
    spec:
      containers:
        - name: neumann
          image: neumann:latest
          ports:
            - containerPort: 9200
              name: grpc
            - containerPort: 9300
              name: raft
          env:
            - name: POD_NAME
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: NEUMANN_CLUSTER_NODE_ID
              value: "$(POD_NAME)"
            - name: NEUMANN_CLUSTER_BIND_ADDR
              value: "0.0.0.0:9300"
            - name: NEUMANN_BIND_ADDR
              value: "0.0.0.0:9200"
            - name: NEUMANN_DATA_DIR
              value: /data
            - name: NEUMANN_CLUSTER_PEERS
              value: "neumann-0=neumann-0.neumann:9300,neumann-1=neumann-1.neumann:9300,neumann-2=neumann-2.neumann:9300"
          volumeMounts:
            - name: data
              mountPath: /data
          readinessProbe:
            exec:
              command: ["grpcurl", "-plaintext", "localhost:9200", "grpc.health.v1.Health/Check"]
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            exec:
              command: ["grpcurl", "-plaintext", "localhost:9200", "grpc.health.v1.Health/Check"]
            initialDelaySeconds: 15
            periodSeconds: 20
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "2Gi"
              cpu: "1000m"
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
```

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: neumann
spec:
  clusterIP: None
  selector:
    app: neumann
  ports:
    - port: 9200
      name: grpc
    - port: 9300
      name: raft
```

**Production overlay** adds TLS via cert-manager, higher resource limits
(4Gi memory, 2 CPU), and 50Gi PersistentVolumeClaims.

### Access via port-forward

```bash
kubectl port-forward svc/neumann 9200:9200
grpcurl -plaintext localhost:9200 grpc.health.v1.Health/Check
```

## Production Checklist

1. **Odd node count.** Use 3, 5, or 7 nodes. Even counts risk split-brain.
2. **Separate availability zones.** Spread nodes across AZs for fault tolerance.
3. **NTP synchronization.** Hybrid logical clocks depend on loosely synchronized wall clocks.
4. **Firewall rules:**
   - Port 9200 (gRPC) -- open to application clients.
   - Port 9300 (Raft) -- open only between cluster nodes.
5. **TLS in production.** Enable TLS for both gRPC and Raft transport.
6. **Resource limits.** Set memory and CPU limits appropriate to your workload. Vector-heavy workloads need more memory.
7. **Persistent storage.** Use SSD-backed volumes for data directories. Size based on expected data volume.
8. **Monitoring.** Expose gRPC health endpoint to load balancers and monitoring systems.
9. **Regular backups.** Schedule periodic `CHECKPOINT` commands.
10. **Test failover.** Verify the cluster survives single-node failure before going live.

## Health Checks

Neumann exposes a standard gRPC health service on the API port (default 9200).

### Manual check

```bash
grpcurl -plaintext localhost:9200 grpc.health.v1.Health/Check
```

### Load balancer integration

Point HTTP health checks at the gRPC health endpoint. Most load balancers
support gRPC health checking natively.

### Kubernetes probes

Use `exec` probes with `grpcurl` as shown in the StatefulSet example above,
or use the native Kubernetes gRPC probe (requires Kubernetes 1.24+):

```yaml
readinessProbe:
  grpc:
    port: 9200
  initialDelaySeconds: 5
  periodSeconds: 10
```

## Backup and Recovery

### Create a checkpoint

```sql
CHECKPOINT 'daily-backup-2024-01-15'
CHECKPOINT
```

Without a name, an auto-generated timestamp-based name is used.

### List checkpoints

```sql
CHECKPOINTS
CHECKPOINTS LIMIT 10
```

### Restore from a checkpoint

```sql
ROLLBACK TO 'daily-backup-2024-01-15'
```

This restores the database state to the point when the checkpoint was created.
All data written after the checkpoint is lost.

### WAL-based durability

With WAL enabled (`--wal-dir`), committed writes survive process crashes without
requiring explicit checkpoints. Checkpoints are still recommended for
point-in-time recovery and long-term backup.
