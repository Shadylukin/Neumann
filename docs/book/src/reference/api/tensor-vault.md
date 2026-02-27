# Tensor Vault API Reference

Complete type reference, configuration tables, storage key patterns, and error
types for the `tensor_vault` crate.

## Core Types

| Type | Description |
| --- | --- |
| `Vault` | Main API for encrypted secret storage with graph-based access control |
| `VaultConfig` | Configuration for key derivation, rate limiting, and versioning |
| `VaultError` | Error types (AccessDenied, NotFound, CryptoError, etc.) |
| `Permission` | Access levels: Read, Write, Admin |
| `VersionInfo` | Metadata about a secret version (version number, timestamp) |
| `ScopedVault` | Entity-bound view for simplified API usage |
| `NamespacedVault` | Namespace-prefixed view for multi-tenant isolation |

## Cryptographic Types

| Type | Description |
| --- | --- |
| `MasterKey` | Derived encryption key with zeroize-on-drop (32 bytes) |
| `Cipher` | AES-256-GCM encryption wrapper |
| `Obfuscator` | HMAC-based key obfuscation and AEAD metadata encryption |
| `PaddingSize` | Padding buckets for length hiding (256B to 64KB) |

## Access Control Types

| Type | Description |
| --- | --- |
| `AccessController` | BFS-based graph path verification |
| `GrantTTLTracker` | Min-heap tracking grant expirations with persistence |
| `RateLimiter` | Sliding window rate limiting per entity |
| `RateLimitConfig` | Configurable limits per operation type |

## Edge Signing Types

| Type | Description |
| --- | --- |
| `EdgeSigner` | HMAC-BLAKE2b signer for graph edge integrity |

## Attenuation Types

| Type | Description |
| --- | --- |
| `AttenuationPolicy` | Distance-based permission degradation policy |

## Delegation Types

| Type | Description |
| --- | --- |
| `DelegationManager` | Agent-to-agent delegation with depth limits and ceiling model |
| `DelegationRecord` | Single delegation grant (parent, child, secrets, ceiling, TTL) |

## Anomaly Detection Types

| Type | Description |
| --- | --- |
| `AnomalyMonitor` | Real-time per-agent behavioral anomaly tracker |
| `AnomalyThresholds` | Configurable limits for spike, bulk, and inactivity detection |
| `AgentProfile` | Per-agent access history (known secrets, counts, timestamps) |
| `AnomalyEvent` | Event variants: FrequencySpike, BulkOperation, InactiveAgentResumed, FirstSecretAccess |

## Audit Types

| Type | Description |
| --- | --- |
| `AuditLog` | Query interface for audit entries with optional HMAC/AEAD protection |
| `AuditEntry` | Single operation record (entity, key, operation, timestamp) |
| `AuditOperation` | Operation types: Get, Set, Delete, Rotate, Grant, Revoke, List |

## Graph Intelligence Types

| Type | Description |
| --- | --- |
| `AccessExplanation` | Path-level explanation of why access was granted or denied |
| `BlastRadius` | All secrets reachable by an entity with permission and hop detail |
| `SimulationResult` | Dry-run impact analysis of a hypothetical grant |
| `SecurityAuditReport` | SCC cycles, single points of failure, over-privileged entities |
| `CriticalEntity` | Articulation-point analysis with PageRank and dependency counts |
| `PrivilegeAnalysisReport` | PageRank-weighted reachability scores per entity |
| `DelegationAnomalyScore` | Jaccard and Adamic-Adar similarity for delegation edges |
| `RoleInferenceResult` | Louvain community detection mapped to inferred roles |
| `TrustTransitivityReport` | Triangle counting and clustering coefficients |
| `RiskPropagationReport` | Eigenvector centrality weighted by admin reachability |

## Error Types

| Error | Description |
| --- | --- |
| `AccessDenied` | Entity has no path to the secret in the access graph |
| `InsufficientPermission` | Entity has a path but the permission level is too low |
| `NotFound` | Secret does not exist |
| `SecretExpired` | Secret has passed its TTL |
| `CryptoError` | Encryption or decryption failure (invalid nonce, plaintext > 64KB, bad UTF-8) |
| `RateLimited` | Entity exceeded rate limit for the operation type |

## Permission Levels

| Level | Capabilities |
| --- | --- |
| Read | `get()`, `list()`, `get_version()`, `list_versions()` |
| Write | Read + `set()` (update), `rotate()`, `rollback()` |
| Admin | Write + `delete()`, `grant()`, `revoke()` |

## Cryptographic Parameters

### Key Derivation (Argon2id)

| Parameter | Default | Description |
| --- | --- | --- |
| Salt size | 16 bytes (128-bit) | Random, persisted in TensorStore at `_vault:salt` |
| Key size | 32 bytes (256-bit) | Output size for AES-256 |
| Memory cost | 65536 KiB (64 MiB) | Argon2id memory parameter |
| Time cost | 3 iterations | Argon2id iteration count |
| Parallelism | 4 threads | Argon2id lane count |

### HKDF Subkey Domains

| Domain | Purpose |
| --- | --- |
| `neumann_vault_encryption_v1` | AES-256-GCM key for secret data |
| `neumann_vault_obfuscation_v1` | HMAC-BLAKE2b key for key name obfuscation |
| `neumann_vault_metadata_v1` | AES-256-GCM key for metadata fields |
| `neumann_vault_audit_v1` | HMAC + AES-256-GCM key for audit entries |
| `neumann_vault_transit_v1` | AES-256-GCM key for transit encryption |

### AES-256-GCM

| Parameter | Value |
| --- | --- |
| Nonce size | 12 bytes (96-bit) |
| Auth tag size | 16 bytes (128-bit) |
| Ciphertext expansion | +16 bytes over plaintext |

### Padding Bucket Sizes

| Variant | Size | Typical Use |
| --- | --- | --- |
| `Small` | 256 bytes | API keys, tokens |
| `Medium` | 1,024 bytes | Certificates, small configs |
| `Large` | 4,096 bytes | Private keys, large configs |
| `ExtraLarge` | 16,384 bytes | Very large secrets |
| `Huge` | 32,768 bytes | Oversized secrets |
| `Maximum` | 65,536 bytes | Maximum supported |

Bucket selection accounts for a 4-byte length prefix and 1 byte minimum
padding. Plaintexts larger than 65,531 bytes (65,536 - 5) return `CryptoError`.

## Configuration

### VaultConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `salt` | `Option<[u8; 16]>` | None | Salt for key derivation (random if not provided, persisted) |
| `argon2_memory_cost` | `u32` | 65536 | Memory cost in KiB (64MB) |
| `argon2_time_cost` | `u32` | 3 | Iteration count |
| `argon2_parallelism` | `u32` | 4 | Thread count |
| `rate_limit` | `Option<RateLimitConfig>` | None | Rate limiting (disabled if None) |
| `max_versions` | `usize` | 5 | Maximum versions to retain per secret |
| `attenuation` | `AttenuationPolicy` | `default()` | Distance-based permission degradation |
| `anomaly_thresholds` | `Option<AnomalyThresholds>` | None | Anomaly detection config (default thresholds if None) |
| `max_delegation_depth` | `Option<u32>` | None | Maximum delegation chain depth (default 3) |

Builder pattern:

```rust
let config = VaultConfig::default()
    .with_rate_limit(RateLimitConfig::default())
    .with_attenuation(AttenuationPolicy {
        admin_limit: 1,
        write_limit: 2,
        horizon: 5,
    })
    .with_anomaly_thresholds(AnomalyThresholds {
        frequency_spike_limit: 100,
        ..AnomalyThresholds::default()
    })
    .with_max_delegation_depth(4);
```

### RateLimitConfig

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `max_gets` | `u32` | 60 | Maximum get() calls per window |
| `max_lists` | `u32` | 10 | Maximum list() calls per window |
| `max_sets` | `u32` | 30 | Maximum set() calls per window |
| `max_grants` | `u32` | 20 | Maximum grant() calls per window |
| `window` | `Duration` | 60s | Sliding window duration |

Presets: `RateLimitConfig::default()`, `RateLimitConfig::strict()`,
`RateLimitConfig::unlimited()`.

`node:root` is exempt from rate limiting.

### AttenuationPolicy

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `admin_limit` | `usize` | 1 | Maximum hops that preserve Admin |
| `write_limit` | `usize` | 2 | Maximum hops that preserve Write |
| `horizon` | `usize` | 10 | BFS cutoff; access denied beyond this |

Use `AttenuationPolicy::none()` to disable attenuation.

### AnomalyThresholds

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `frequency_spike_limit` | `u64` | 50 | Ops per window that trigger a spike |
| `frequency_window_ms` | `i64` | 60,000 | Sliding window size (1 minute) |
| `bulk_operation_threshold` | `u64` | 10 | Burst size that triggers a bulk event |
| `inactive_threshold_ms` | `i64` | 86,400,000 | Inactivity before resumption is flagged (24h) |

### DelegationRecord Fields

| Field | Type | Description |
| --- | --- | --- |
| `parent` | `String` | Delegating (parent) agent |
| `child` | `String` | Receiving (child) agent |
| `secrets` | `Vec<String>` | Secret names delegated |
| `max_permission` | `Permission` | Ceiling permission level |
| `ttl_ms` | `Option<i64>` | Optional TTL in milliseconds |
| `created_at_ms` | `i64` | Creation timestamp (unix ms) |
| `delegation_depth` | `u32` | Hops from root to this child |

### Environment Variables

| Variable | Description |
| --- | --- |
| `NEUMANN_VAULT_KEY` | Base64-encoded 32-byte master key |

## Storage Key Patterns

```text
_vault:salt                          Persisted 16-byte salt for key derivation
_vk:<32-hex-chars>                   Metadata tensor (HMAC of secret key)
_vs:<24-hex-chars>                   Ciphertext blob (HMAC of key + nonce)
_va:<timestamp>:<counter>            Audit log entries
_vault_ttl_grants                    Persisted TTL grants (JSON)
vault_secret:<32-hex-chars>          Secret node for graph access control
```

### Metadata Tensor Fields

Storage key: `_vk:{HMAC(key)}`

| Field | Type | Description |
| --- | --- | --- |
| `_blob` | Pointer | Reference to current version ciphertext blob |
| `_nonce` | Bytes | 12-byte encryption nonce for current version |
| `_versions` | Pointers | List of all version blob keys (oldest first) |
| `_key_enc` | Bytes | AES-GCM encrypted original key name |
| `_key_nonce` | Bytes | Nonce for key encryption |
| `_creator_obf` | Bytes | AEAD-encrypted creator (nonce prepended) |
| `_created_obf` | Bytes | AEAD-encrypted timestamp (nonce prepended) |
| `_rotator_obf` | Bytes | AEAD-encrypted last rotator (optional) |
| `_rotated_obf` | Bytes | AEAD-encrypted last rotation timestamp (optional) |

### Ciphertext Blob Fields

Storage key: `_vs:{HMAC(key, nonce)}`

| Field | Type | Description |
| --- | --- | --- |
| `_data` | Bytes | Padded + encrypted secret |
| `_nonce` | Bytes | 12-byte encryption nonce |
| `_ts` | Int | Unix timestamp (seconds) when version was created |

## Audit Query Methods

| Method | Description | Time Complexity |
| --- | --- | --- |
| `by_secret(key)` | All entries for a secret | O(n) scan + filter |
| `by_entity(entity)` | All entries by requester | O(n) scan + filter |
| `since(timestamp)` | Entries since timestamp | O(n) scan + filter |
| `between(start, end)` | Entries in time range | O(n) scan + filter |
| `recent(limit)` | Last N entries | O(n log n) sort + truncate |

## Graph Intelligence Methods

### Tier 1 -- Graph-Native Features

| Method | Description |
| --- | --- |
| `explain_access(entity, secret)` | Path-level explanation of why access was granted or denied |
| `blast_radius(entity)` | All secrets reachable by entity with permission and hop detail |
| `simulate_grant(entity, secret, perm)` | Dry-run impact analysis of a hypothetical grant |
| `security_audit()` | SCC cycles, single points of failure, over-privileged entities |
| `find_critical_entities()` | Articulation-point analysis with PageRank |

### Tier 3 -- Graph Analytics

| Method | Description |
| --- | --- |
| `privilege_analysis()` | PageRank-weighted reachability scores per entity |
| `delegation_anomaly_scores()` | Jaccard and Adamic-Adar similarity for delegation edges |
| `infer_roles()` | Louvain community detection mapped to inferred roles |
| `trust_transitivity()` | Triangle counting and clustering coefficients |
| `risk_propagation()` | Eigenvector centrality weighted by admin reachability |

## Rust-Only API Features

The following features are available through the Rust API but do not yet have
shell command equivalents:

| Feature | API Method |
| --- | --- |
| Delegation | `vault.delegate()`, `vault.revoke_delegation()` |
| Transit encryption | `vault.encrypt_for()`, `vault.decrypt_as()` |
| Secret expiration | `vault.set_with_ttl()`, `vault.get_expiration()` |
| Break-glass | `vault.emergency_access()` |
| Batch operations | `vault.batch_get()`, `vault.batch_set()` |
| Key rotation | `vault.rotate_master_key()` |

## Shell Commands

### Core Operations

```text
VAULT INIT                              Initialize vault from NEUMANN_VAULT_KEY
VAULT IDENTITY 'node:alice'             Set current identity
VAULT NAMESPACE 'team:backend'          Set current namespace

VAULT SET 'api_key' 'sk-123'            Store encrypted secret
VAULT GET 'api_key'                     Retrieve secret
VAULT GET 'api_key' VERSION 2           Get specific version
VAULT DELETE 'api_key'                  Delete secret
VAULT LIST 'prefix:*'                   List accessible secrets
VAULT ROTATE 'api_key' 'new'            Rotate secret value
VAULT VERSIONS 'api_key'                List version history
VAULT ROLLBACK 'api_key' VERSION 2      Rollback to version
```

### Access Control

```text
VAULT GRANT 'user:bob' ON 'api_key'              Grant admin access
VAULT GRANT 'user:bob' ON 'api_key' READ         Grant read-only access
VAULT GRANT 'user:bob' ON 'api_key' WRITE        Grant write access
VAULT GRANT 'user:bob' ON 'api_key' TTL 3600     Grant with 1-hour expiry
VAULT REVOKE 'user:bob' ON 'api_key'             Revoke access
```

### Audit

```text
VAULT AUDIT 'api_key'                   View audit log for secret
VAULT AUDIT BY 'user:alice'             View audit log for entity
VAULT AUDIT RECENT 10                   View last 10 operations
```

## Edge Cases and Gotchas

| Scenario | Behavior |
| --- | --- |
| Grant to non-existent entity | Succeeds (edge created, entity may exist later) |
| Revoke non-existent grant | Succeeds silently (idempotent) |
| Get non-existent secret | Returns `NotFound` error |
| Get expired secret | Returns `SecretExpired` error |
| Set by non-root without Write | Returns `AccessDenied` or `InsufficientPermission` |
| TTL grant cleanup | Opportunistic on `get()` -- may not be immediate |
| Version limit exceeded | Oldest versions automatically deleted |
| Plaintext > 64KB | Returns `CryptoError` |
| Invalid UTF-8 in secret | `get()` returns `CryptoError` |
| Concurrent modifications | Thread-safe via DashMap sharding |
| MEMBER edge to secret | Path exists but NO permission granted |
| Tampered graph edge | BFS skips the edge; `explain_access` reports `TamperedEdge` |
| Delegation depth exceeded | Returns error; configurable via `max_delegation_depth` |
| Delegation cycle attempt | Returns error; ancestors cannot be children |
| Break-glass rate limited | Only 1 emergency access per rate-limit window |
| Transit ciphertext after key rotation | Undecryptable (forward secrecy) |

## Performance

| Operation | Time | Notes |
| --- | --- | --- |
| Key derivation (Argon2id) | ~80ms | 64MB memory cost |
| set (1KB) | ~29us | Includes encryption + versioning |
| get (1KB) | ~24us | Includes decryption + audit |
| set (10KB) | ~93us | Scales with data size |
| get (10KB) | ~91us | Scales with data size |
| Access check (shallow) | ~6us | Direct edge |
| Access check (deep, 10 hops) | ~17us | BFS traversal |
| grant | ~18us | Creates graph edge |
| revoke | ~1.1ms | Edge deletion + TTL cleanup |
| list (100 secrets) | ~291us | Pattern matching + access check |
| list (1000 secrets) | ~2.7ms | Scales linearly |

## Dependencies

| Crate | Purpose |
| --- | --- |
| `aes-gcm` | AES-256-GCM authenticated encryption |
| `argon2` | Argon2id key derivation |
| `hkdf` | HKDF-SHA256 subkey derivation |
| `blake2` | HMAC-BLAKE2b for obfuscation, edge signing, audit integrity |
| `rand` | Nonce and salt generation |
| `zeroize` | Secure memory cleanup on drop |
| `dashmap` | Concurrent rate limit and anomaly tracking |
| `serde` | TTL grant and delegation persistence |
| `graph_engine` | BFS traversal, PageRank, Louvain, eigenvector centrality |
| `tensor_store` | Underlying key-value storage for all vault data |

## Related Modules

| Module | Relationship |
| --- | --- |
| [Tensor Store](tensor-store.md) | Underlying key-value storage for encrypted secrets |
| [Graph Engine](graph-engine.md) | Access control edges and audit trail |

## See Also

- [Vault Cryptography Design](../../explanation/vault-crypto.md) -- encryption
  architecture, key derivation rationale, and security model
- [How to: Store and Retrieve Secrets](../../how-to/vault-secrets.md) --
  step-by-step examples for common vault operations
- [How to: Vault Access Control](../../how-to/vault-access-control.md) --
  set up graph-based access control, delegation, and security audits
