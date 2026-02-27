# How to: Vault Access Control

Step-by-step guides for setting up graph-based access control in Tensor Vault:
granting and revoking permissions, configuring delegation chains, running
security audits, and using graph intelligence features.

For the full type reference, see the
[Tensor Vault API Reference](../reference/api/tensor-vault.md). For the
underlying security design, see
[Vault Cryptography Design](../explanation/vault-crypto.md).

## Understand the Permission Model

Access is determined by graph topology. Entities reach secrets through
`VAULT_ACCESS_*` edges, optionally traversing `MEMBER` edges through groups:

```text
node:root --VAULT_ACCESS_ADMIN--> vault_secret:api_key
                                          ^
user:alice --VAULT_ACCESS_READ-----------+
                                          ^
team:devs --VAULT_ACCESS_WRITE----------+
      ^
user:bob --MEMBER----------------------------+
```

| Requester | Path | Result |
| --- | --- | --- |
| `node:root` | Always | Granted (Admin) |
| `user:alice` | Direct edge | Granted (Read only) |
| `team:devs` | Direct edge | Granted (Write) |
| `user:bob` | bob -> team:devs -> secret | Granted (Write via team) |
| `user:carol` | No path | Denied |

`MEMBER` edges allow traversal but never grant permissions directly.

## Grant Permissions at Different Levels

```rust
// Read-only access
vault.grant_with_permission(
    Vault::ROOT, "user:reader", "secret", Permission::Read,
)?;

// Read + Write
vault.grant_with_permission(
    Vault::ROOT, "user:writer", "secret", Permission::Write,
)?;

// Full access (includes grant/revoke)
vault.grant_with_permission(
    Vault::ROOT, "user:admin", "secret", Permission::Admin,
)?;
```

What each level allows:

```rust
// Reader can only get/list
vault.get("user:reader", "secret")?;              // OK
vault.set("user:reader", "secret", "new")?;       // InsufficientPermission

// Writer can update
vault.rotate("user:writer", "secret", "new")?;    // OK
vault.delete("user:writer", "secret")?;            // InsufficientPermission

// Admin can do everything
vault.grant_with_permission(
    "user:admin", "user:new", "secret", Permission::Read,
)?;                                                // OK
vault.delete("user:admin", "secret")?;             // OK
```

## Set Up Agent Delegation

Agents can delegate subsets of their own access to child agents. The ceiling
model ensures a child never exceeds the parent's permission.

```rust
// Team lead delegates read-only access to deploy agent
vault.delegate(
    "user:lead",
    "agent:deploy",
    &["prod/db", "prod/api_key"],
    Permission::Read,
    None,  // no TTL
)?;

// Deploy agent delegates to canary (ceiling: Read)
vault.delegate(
    "agent:deploy",
    "agent:canary",
    &["prod/db"],
    Permission::Read,          // cannot exceed parent's Read
    Some(Duration::from_secs(600)),  // 10-minute window
)?;
```

### Revoke Delegation (Cascading)

Revoking a parent revokes all descendants:

```rust
vault.revoke_delegation_cascading("user:lead", "agent:deploy")?;
// Both agent:deploy and agent:canary lose access
```

### Delegation Properties

- **Ceiling model**: `effective = min(parent_permission, requested)`
- **Depth limits**: Configurable maximum chain depth (default 3)
- **Cycle prevention**: A child cannot delegate back to an ancestor
- **TTL support**: Delegations can expire automatically

Delegation composes with attenuation: the effective permission is the minimum of
the delegation ceiling and the attenuated permission at the child's graph
distance.

## Configure Distance Attenuation

Permissions degrade as graph distance increases:

```rust
let config = VaultConfig::default()
    .with_attenuation(AttenuationPolicy {
        admin_limit: 1,   // Admin only at 1 hop
        write_limit: 2,   // Write degrades to Read after 2 hops
        horizon: 5,       // Denied beyond 5 hops
    });
```

| Hops | Effective Permission |
| --- | --- |
| 1 | Admin (if granted Admin) |
| 2 | Write (Admin attenuates to Write) |
| 3-5 | Read (Write attenuates to Read) |
| >5 | Denied (beyond horizon) |

Use `AttenuationPolicy::none()` to disable attenuation for backward
compatibility.

## Configure Rate Limiting

Prevent brute-force attacks with sliding window rate limits:

```rust
let config = VaultConfig::default()
    .with_rate_limit(RateLimitConfig {
        max_gets: 60,     // 60 get() calls per minute
        max_lists: 10,    // 10 list() calls per minute
        max_sets: 30,     // 30 set() calls per minute
        max_grants: 20,   // 20 grant() calls per minute
        window: Duration::from_secs(60),
    });
```

Presets:
- `RateLimitConfig::default()` -- standard limits
- `RateLimitConfig::strict()` -- tight limits for testing
- `RateLimitConfig::unlimited()` -- no limits

`node:root` is exempt from rate limiting.

## Enable Anomaly Detection

Configure real-time behavioral monitoring:

```rust
let config = VaultConfig::default()
    .with_anomaly_thresholds(AnomalyThresholds {
        frequency_spike_limit: 100,    // ops per window
        frequency_window_ms: 60_000,   // 1-minute window
        bulk_operation_threshold: 10,  // burst size
        inactive_threshold_ms: 86_400_000,  // 24h inactivity
    });
```

The monitor flags events but does not deny access:
- `FirstSecretAccess` -- entity accesses a secret it has never accessed before
- `FrequencySpike` -- operations exceed `frequency_spike_limit` in window
- `BulkOperation` -- burst exceeds `bulk_operation_threshold`
- `InactiveAgentResumed` -- entity resumes after long inactivity

## Debug Permissions with explain_access

When access is denied unexpectedly, trace the exact paths:

```rust
let explanation = vault.explain_access("user:bob", "db/password");
if explanation.granted {
    for path in &explanation.paths {
        for hop in path {
            println!("{} --{}-->", hop.entity, hop.edge_type);
        }
    }
} else if let Some(reason) = &explanation.denial_reason {
    println!("Denied: {reason:?}");
    // Possible reasons: NoPath, InsufficientPermission, Attenuation, TamperedEdge
}
```

## Analyze Blast Radius

See all secrets reachable by an entity:

```rust
let radius = vault.blast_radius("user:alice");
println!("{} can reach {} secrets", radius.entity, radius.total_secrets);
for secret in &radius.secrets {
    println!(
        "  {} ({:?}, {} hops)",
        secret.secret_name, secret.permission, secret.hop_count
    );
}
```

## Simulate a Grant Before Applying

Dry-run a hypothetical grant to see its impact without modifying the graph:

```rust
let sim = vault.simulate_grant("user:bob", "db/password", Permission::Write);
println!("{} entities would gain new access", sim.total_affected);
for access in &sim.new_accesses {
    println!("  {} gains {:?} on {}", access.entity, access.permission, access.secret);
}
```

## Run a Security Audit

Detect structural issues in the permission graph:

```rust
let report = vault.security_audit();
println!("Cycles: {}", report.cycles.len());
println!("SPOFs: {}", report.single_points_of_failure.len());
println!("Over-privileged: {}", report.over_privileged.len());
```

## Find Critical Entities

Identify articulation points whose removal would isolate secrets:

```rust
let critical = vault.find_critical_entities();
for entity in &critical {
    println!(
        "{}: SPOF={}, {} secrets solely dependent, PageRank={:.4}",
        entity.entity,
        entity.is_single_point_of_failure,
        entity.secrets_solely_dependent,
        entity.pagerank_score,
    );
}
```

## Run Graph Analytics

### Privilege Analysis

Rank entities by influence using PageRank combined with reachability:

```rust
let report = vault.privilege_analysis();
for entity in &report.entities {
    println!(
        "{}: PageRank={:.4}, reachable={}, privilege={:.4}",
        entity.entity, entity.pagerank_score,
        entity.reachable_secrets, entity.privilege_score,
    );
}
```

### Detect Unusual Delegations

Compute Jaccard similarity and Adamic-Adar scores for delegation edges:

```rust
let scores = vault.delegation_anomaly_scores();
for score in scores.iter().filter(|s| s.anomaly_score > 0.8) {
    println!("Unusual delegation: {} -> {} (anomaly {:.2})",
        score.entity, score.secret, score.anomaly_score);
}
```

### Infer Roles

Discover implicit role groupings via Louvain community detection:

```rust
let roles = vault.infer_roles();
for role in &roles.roles {
    println!(
        "Role {}: {} members, {} common secrets",
        role.role_id, role.members.len(), role.common_secrets.len(),
    );
}
```

### Measure Trust Transitivity

Count triangles and compute clustering coefficients:

```rust
let trust = vault.trust_transitivity();
println!("Global clustering: {:.4}", trust.global_clustering);
for entity in &trust.entities {
    println!(
        "{}: triangles={}, clustering={:.4}",
        entity.entity, entity.triangle_count, entity.clustering_coefficient,
    );
}
```

### Assess Risk Propagation

Model risk propagation using eigenvector centrality:

```rust
let risk = vault.risk_propagation();
for entity in &risk.entities {
    println!(
        "{}: eigenvector={:.4}, admin_secrets={}, risk={:.4}",
        entity.entity, entity.eigenvector_score,
        entity.reachable_admin_secrets, entity.risk_score,
    );
}
```

## Multi-Agent Secret Sharing Example

A deployment pipeline with three agents needing different access levels:

```rust
// CI runner needs read-only access to run migrations
vault.grant_with_permission(
    Vault::ROOT, "agent:ci", "db/password", Permission::Read,
)?;

// Deploy agent needs write to rotate credentials
vault.grant_with_permission(
    Vault::ROOT, "agent:deploy", "db/password", Permission::Write,
)?;

// Monitoring agent gets temporary read access
vault.grant_with_ttl(
    Vault::ROOT, "agent:monitor", "db/password",
    Permission::Read, Duration::from_secs(3600),
)?;
```

## See Also

- [Tensor Vault API Reference](../reference/api/tensor-vault.md) -- complete
  type tables, error types, and configuration options
- [Vault Cryptography Design](../explanation/vault-crypto.md) -- encryption
  architecture and security model rationale
- [How to: Store and Retrieve Secrets](vault-secrets.md) -- basic vault
  operations and secret management
