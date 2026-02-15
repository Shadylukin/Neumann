# Neumann Kubernetes Deployment

Kustomize-based manifests for deploying a 3-node Neumann cluster on Kubernetes.

## Prerequisites

- kubectl 1.24+
- Kustomize (built into kubectl) or standalone kustomize
- A Kubernetes cluster with:
  - PersistentVolume provisioner (for StatefulSet PVCs)
  - gRPC health check support (Kubernetes 1.24+)
- For production TLS: [cert-manager](https://cert-manager.io/) installed

## Quick Start (Dev)

```bash
# Deploy
kubectl apply -k deploy/k8s/overlays/dev/

# Wait for pods
kubectl -n neumann get pods -w

# Connect via port-forward
kubectl -n neumann port-forward svc/neumann 9200:9200
```

## Production Deployment

```bash
# Review rendered manifests
kubectl kustomize deploy/k8s/overlays/production/

# Deploy
kubectl apply -k deploy/k8s/overlays/production/

# Verify
kubectl -n neumann get pods,svc,pvc
```

The production overlay includes:

- Server-side TLS via cert-manager (self-signed by default, replace the
  `ClusterIssuer` with your CA)
- Higher resource limits (4Gi/4CPU)
- 50Gi persistent volumes
- Pinned image tag

## Architecture

```text
StatefulSet (3 replicas, Parallel pod management)
  neumann-0  neumann-1  neumann-2
      |          |          |
      +--- Headless Service (DNS, inter-node Raft on :9300) ---+
      |          |          |
      +--- Client Service (gRPC API on :9200, ready pods only) ---+
```

- **Headless Service** (`neumann-headless`): Provides stable DNS for peer
  discovery. `publishNotReadyAddresses: true` so pods can find each other
  during bootstrap.
- **Client Service** (`neumann`): Routes gRPC traffic to ready pods only.
- **NetworkPolicy**: Raft port (9300) restricted to neumann pods. gRPC port
  (9200) open to all namespaces.
- **PodDisruptionBudget**: `minAvailable: 2` to maintain Raft quorum during
  voluntary disruptions.

## DNS-to-IP Resolution

`NEUMANN_CLUSTER_PEERS` requires `SocketAddr` (IP:port), not DNS hostnames.
The init script resolves pod DNS names to IPs at startup with a retry loop
(up to 120s) to handle bootstrap races.

**Limitation**: Peers are resolved to IPs at boot. If a pod's IP changes
(e.g., node eviction), the cluster needs a rolling restart. This is mitigated
by:

- StatefulSet pods retain identity and PVCs across rescheduling
- PodDisruptionBudget prevents concurrent evictions
- Pod anti-affinity spreads pods across nodes

## Scaling

Cluster size must be an odd number (3, 5, 7) for Raft quorum.

Membership is static -- `NEUMANN_CLUSTER_PEERS` is set at boot from
`NEUMANN_REPLICAS`. The consistent hash partitioner is also built at init
time. To change cluster size:

1. Update `NEUMANN_REPLICAS` env var and StatefulSet `replicas`
2. Rolling restart all pods: `kubectl -n neumann rollout restart sts/neumann`

There is no automatic re-partitioning or dynamic membership.

## Monitoring

Set the `NEUMANN_OTLP_ENDPOINT` environment variable to export OpenTelemetry
traces and metrics:

```yaml
# Add to configmap patch
NEUMANN_OTLP_ENDPOINT: "http://otel-collector.monitoring:4317"
```

## TLS

### Server-side TLS (production overlay)

The production overlay enables server-side TLS using cert-manager. Certificates
are issued by a self-signed `ClusterIssuer` -- replace `cluster-issuer.yaml`
with your organization's CA for real deployments.

gRPC health probes work with server-side TLS (Kubernetes handles this natively).

### mTLS

Client certificate verification (`NEUMANN_TLS_REQUIRE_CLIENT_CERT`) is not
enabled by default because:

- Kubernetes native gRPC probes cannot present client certificates
- Current client tooling lacks client cert flags

For mTLS, use a service mesh (Istio, Linkerd) which handles certificate
distribution transparently.

## Troubleshooting

```bash
# Check pod logs
kubectl -n neumann logs neumann-0

# Check if DNS resolution is working
kubectl -n neumann exec neumann-0 -- getent hosts neumann-1.neumann-headless.neumann.svc.cluster.local

# Check cluster connectivity
kubectl -n neumann exec neumann-0 -- sh -c 'echo $NEUMANN_CLUSTER_PEERS'

# Describe pod for events
kubectl -n neumann describe pod neumann-0
```
