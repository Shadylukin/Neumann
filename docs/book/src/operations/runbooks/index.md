# Runbooks

Operational runbooks for managing Neumann clusters, focusing on tensor_chain
distributed operations.

## Available Runbooks

| Runbook | Scenario | Severity |
| --- | --- | --- |
| [Leader Election](leader-election.md) | Cluster has no leader | Critical |
| [Split-Brain Recovery](split-brain.md) | Network partition healed | Critical |
| [Node Recovery](node-recovery.md) | Node crash or disk failure | High |
| [Backup and Restore](backup-restore.md) | Data backup and disaster recovery | High |
| [Capacity Planning](capacity-planning.md) | Resource sizing and scaling | Medium |
| [Deadlock Resolution](deadlock-resolution.md) | Transaction deadlocks | Medium |

## How to Use These Runbooks

1. **Identify the symptom** from alerts or monitoring
2. **Find the matching runbook** in the table above
3. **Follow the diagnostic steps** to confirm root cause
4. **Execute the resolution steps** in order
5. **Verify recovery** using the provided checks

## Alerting Rules

Each runbook includes Prometheus alerting rules. Deploy them to your monitoring
stack:

```bash
# Copy alerting rules
cp docs/book/src/operations/alerting-rules.yml /etc/prometheus/rules/neumann.yml

# Reload Prometheus
curl -X POST http://prometheus:9090/-/reload
```

## Emergency Contacts

For production incidents:

1. Page the on-call engineer
2. Start an incident channel
3. Follow the relevant runbook
4. Document actions taken
5. Schedule post-incident review
