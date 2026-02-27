# Deploying a Three-Node Cluster

Deploy a three-node Neumann cluster using Docker Compose and verify it
reaches consensus. By the end you will have a working distributed cluster
with automatic leader election.

## Prerequisites

- Docker and Docker Compose installed
- Neumann Docker image available (`shadylukinack/neumann:latest`)

## Step 1: Create the Compose File

Create a file named `docker-compose.yml`:

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

## Step 2: Start the Cluster

```bash
docker compose up -d
```

Wait a few seconds for the nodes to start and elect a leader.

## Step 3: Check Cluster Health

Query the health endpoint on node1:

```bash
grpcurl -plaintext localhost:9200 grpc.health.v1.Health/Check
```

You should see:

```json
{
  "status": "SERVING"
}
```

## Step 4: View Logs

Check that leader election succeeded:

```bash
docker compose logs node1 | grep -i "leader\|elected\|term"
```

You should see log entries showing the node transitioning through Raft
states and a leader being elected.

## Step 5: Run Queries

Connect to the cluster through node1 and run a query:

```bash
docker compose exec node1 neumann_client --addr localhost:9200
```

```sql
CREATE TABLE test (id INT PRIMARY KEY, value TEXT);
INSERT INTO test VALUES (1, 'hello cluster');
SELECT * FROM test;
```

## Step 6: Verify Replication

Stop node1 and verify the cluster continues operating through node2:

```bash
docker compose stop node1
```

A new leader should be elected among node2 and node3. Reconnect through
a different node to verify data is still available.

```bash
docker compose start node1
```

The node rejoins the cluster and catches up via log replication.

## Step 7: Clean Up

```bash
docker compose down -v
```

## Verification

You should have seen:

- Three containers started successfully
- Health check returning SERVING
- Leader election in the logs
- Queries executing through the cluster
- Cluster surviving a node failure

## Next Steps

- [Deploy a Cluster](../how-to/deploy-cluster.md) -- production deployment
  guide
- [Configure Raft](../how-to/configure-raft.md) -- tune consensus
  parameters
- [Configuration Reference](../reference/configuration.md) -- all config
  options
