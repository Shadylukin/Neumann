# Backup and Restore

## Backup Strategy

| Type | Frequency | Retention | RPO | RTO |
| --- | --- | --- | --- | --- |
| Snapshots | Every 10k entries | 7 days | Minutes | Minutes |
| Full backup | Daily | 30 days | 24 hours | Hours |
| Off-site | Weekly | 1 year | 1 week | Hours |

## Creating Backups

### Snapshot Backup (Hot)

```bash
# Trigger snapshot on leader
curl -X POST http://leader:9090/admin/snapshot

# Wait for completion
watch 'curl -s http://leader:9090/metrics | grep snapshot'

# Copy snapshot files
rsync -av /var/lib/neumann/raft/snapshots/ backup:/backups/neumann/snapshots/

# Include metadata
neumann-admin cluster-info > backup:/backups/neumann/metadata.json
```

### Full Backup (Recommended: Cold)

```bash
# 1. Stop writes (or accept slightly inconsistent backup)
neumann-admin pause-writes

# 2. Create snapshot
curl -X POST http://leader:9090/admin/snapshot
sleep 10

# 3. Backup all state
tar -czf neumann-backup-$(date +%Y%m%d).tar.gz \
  /var/lib/neumann/raft/snapshots/ \
  /var/lib/neumann/store/ \
  /etc/neumann/

# 4. Resume writes
neumann-admin resume-writes

# 5. Verify backup integrity
tar -tzf neumann-backup-*.tar.gz | head
```

### Continuous WAL Archiving

```bash
# In config.toml
[wal]
archive_command = "aws s3 cp %p s3://backups/neumann/wal/%f"
archive_timeout = 60  # seconds

# Or to local storage
archive_command = "cp %p /mnt/backup/wal/%f"
```

## Restore Procedures

### Point-in-Time Recovery

```bash
# 1. Stop all nodes
ansible all -m systemd -a "name=neumann state=stopped"

# 2. Clear current state
ansible all -m shell -a "rm -rf /var/lib/neumann/raft/*"

# 3. Restore snapshot to one node
scp backup:/backups/neumann/snapshots/latest/* node1:/var/lib/neumann/raft/snapshots/

# 4. Replay WAL up to desired point
neumann-admin wal-replay \
  --wal-dir backup:/backups/neumann/wal/ \
  --until "2024-01-15T10:30:00Z"

# 5. Start first node
ssh node1 systemctl start neumann

# 6. Start remaining nodes (will sync from node1)
ansible "node2,node3" -m systemd -a "name=neumann state=started"
```

### Full Cluster Restore

```bash
# 1. Extract backup
tar -xzf neumann-backup-20240115.tar.gz -C /tmp/restore/

# 2. Stop cluster
systemctl stop neumann

# 3. Restore files
rsync -av /tmp/restore/raft/ /var/lib/neumann/raft/
rsync -av /tmp/restore/store/ /var/lib/neumann/store/

# 4. Fix permissions
chown -R neumann:neumann /var/lib/neumann/

# 5. Start cluster
systemctl start neumann
```

### Disaster Recovery (Complete Loss)

```bash
# 1. Provision new infrastructure

# 2. Install Neumann on all nodes

# 3. Restore from off-site backup
aws s3 cp s3://backups/neumann/latest.tar.gz /tmp/
tar -xzf /tmp/latest.tar.gz -C /var/lib/neumann/

# 4. Update config with new node addresses
vim /etc/neumann/config.toml

# 5. Initialize cluster
neumann-admin init-cluster --bootstrap

# 6. Verify
neumann-admin cluster-status
```

## Verification

```bash
# Check data integrity
neumann-admin verify-checksums

# Compare entry counts
neumann-admin stats | grep total_entries

# Spot check recent data
neumann-admin query "SELECT COUNT(*) FROM ..."
```

## Retention Policy

```bash
# Cron job for cleanup
0 2 * * * find /var/lib/neumann/raft/snapshots -mtime +7 -delete
0 3 * * 0 aws s3 rm s3://backups/neumann/wal/ --recursive --exclude "*.wal" --older-than 30d
```
