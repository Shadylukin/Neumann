# Set Up Monitoring

## Prometheus Configuration

```yaml
scrape_configs:
  - job_name: 'neumann'
    static_configs:
      - targets:
        - 'node1:9090'
        - 'node2:9090'
        - 'node3:9090'
```

## Grafana Dashboard

Import the dashboard from `deploy/grafana/neumann-dashboard.json`.

Panels include:

- Cluster overview (leader, term, members)
- Transaction throughput and latency
- Replication lag
- Memory and disk usage
- Deadlock rate

## Alerting Rules

See [Metrics reference](../reference/metrics.md#alerting-rules) for the
full list of alerting rules. Deploy them to your Prometheus instance:

```bash
# Reload Prometheus after adding rules
curl -X POST http://prometheus:9090/-/reload
```

## Logging

Configure log level:

```bash
RUST_LOG=tensor_chain=debug neumann
```

Log levels: `error`, `warn`, `info`, `debug`, `trace`
