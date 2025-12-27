# Tensor Vault

Module 9 of Neumann. Secure secret storage with AES-256-GCM encryption and graph-based access control, designed for multi-agent environments.

## Design Principles

1. **Encryption at Rest**: All secrets encrypted with AES-256-GCM
2. **Topological Access Control**: Access determined by graph path, not ACLs
3. **Zero Trust**: No bypass mode; `node:root` is the only universal accessor
4. **Memory Safety**: Keys zeroized on drop via `zeroize` crate
5. **Permanent Audit Trail**: All operations logged with queryable API
6. **Defense in Depth**: Multiple obfuscation layers hide patterns
7. **Multi-Tenant Ready**: Namespace isolation and rate limiting for agent systems

## Quick Start

```rust
use tensor_vault::{Vault, VaultConfig};
use graph_engine::GraphEngine;
use tensor_store::TensorStore;
use std::sync::Arc;

// Initialize vault with master key
let graph = Arc::new(GraphEngine::new());
let store = TensorStore::new();
let vault = Vault::new(b"master_password", graph, store, VaultConfig::default())?;

// Store a secret (only root can create)
vault.set(Vault::ROOT, "api_key", "sk-secret123")?;

// Grant access with permission level
vault.grant_with_permission(Vault::ROOT, "user:alice", "api_key", Permission::Read)?;

// Grant temporary access (expires in 1 hour)
vault.grant_with_ttl(Vault::ROOT, "user:bob", "api_key", Duration::from_secs(3600))?;

// Alice can now retrieve the secret
let value = vault.get("user:alice", "api_key")?;

// Use namespaced vault for multi-tenant isolation
let ns = vault.namespace("team:backend", "user:alice");
ns.set("db_password", "secret123")?;  // Stored as "team:backend:db_password"
```

## Access Control Model

Access is determined by graph topology, not explicit permissions:

```
node:root ──VAULT_ACCESS──> vault_secret:api_key
                                    ^
user:alice ──VAULT_ACCESS──────────┘
                                    ^
team:devs ──VAULT_ACCESS───────────┘
      ^
user:bob ──MEMBER──────────────────┘
```

**Path Resolution**: BFS traversal from requester to secret node, following outgoing edges only.

| Requester | Path | Access |
|-----------|------|--------|
| `node:root` | Always | Granted |
| `user:alice` | Direct edge | Granted |
| `user:bob` | bob -> team:devs -> secret | Granted |
| `user:carol` | No path | Denied |

### Revocation

Delete the edge to revoke access:

```rust
vault.revoke(Vault::ROOT, "user:alice", "api_key")?;
// Alice can no longer access the secret
```

Transitive access is automatically revoked when any edge in the path is removed.

## Permission Levels

Grants support three permission levels:

| Level | Capabilities |
|-------|-------------|
| **Read** | `get()`, `list()` |
| **Write** | Read + `set()`, `rotate()` |
| **Admin** | Write + `delete()`, `grant()`, `revoke()` |

```rust
use tensor_vault::Permission;

// Grant read-only access
vault.grant_with_permission(Vault::ROOT, "user:reader", "api_key", Permission::Read)?;

// Grant write access (can update but not delete/grant)
vault.grant_with_permission(Vault::ROOT, "user:writer", "api_key", Permission::Write)?;

// Grant admin access (full control)
vault.grant_with_permission(Vault::ROOT, "user:admin", "api_key", Permission::Admin)?;

// Check effective permission
let perm = vault.get_permission("user:writer", "api_key")?;
assert_eq!(perm, Permission::Write);
```

**Permission Propagation**: Permissions follow graph paths. The effective permission is the minimum along any path:

```
user:alice ──VAULT_ACCESS_WRITE──> vault_secret:key   = Write
user:alice ──MEMBER──> team ──VAULT_ACCESS_READ──>    = Read (min of path)
```

## TTL Grants

Grants can have automatic expiration:

```rust
use std::time::Duration;

// Grant access for 1 hour
vault.grant_with_ttl(Vault::ROOT, "agent:temp", "api_key", Duration::from_secs(3600))?;

// Check remaining time (returns None if no TTL)
// Expired grants are automatically cleaned up on next vault access
```

The vault uses a min-heap to track expirations and cleans up expired grants lazily during operations.

## Rate Limiting

Prevents brute-force enumeration and throttles aggressive agents:

```rust
use tensor_vault::{VaultConfig, RateLimitConfig};

let config = VaultConfig {
    rate_limit: Some(RateLimitConfig {
        max_gets: 60,      // per minute
        max_lists: 10,
        max_sets: 30,
        max_grants: 20,
        window: Duration::from_secs(60),
    }),
    ..Default::default()
};

let vault = Vault::new(b"password", graph, store, config)?;

// After 60 gets in a minute, further gets return RateLimited error
// Root is exempt from rate limiting
```

**Per-Entity Limits**: Each entity has independent quotas. Rate limits use a sliding window algorithm.

## Namespace Isolation

For multi-tenant agent systems:

```rust
// Create namespaced vault
let backend = vault.namespace("team:backend", "user:alice");
let frontend = vault.namespace("team:frontend", "user:bob");

// Keys are automatically prefixed
backend.set("db_password", "secret1")?;   // Stored as "team:backend:db_password"
frontend.set("api_key", "secret2")?;      // Stored as "team:frontend:api_key"

// Cross-namespace access blocked
frontend.get("db_password")?;  // AccessDenied - can't see backend namespace

// List only shows namespace keys
backend.list("*")?;  // Returns ["db_password"], not frontend keys
```

## Audit Query API

All operations are logged and queryable:

```rust
// Query audit log by secret
let entries = vault.audit_log("api_key");

// Query by entity (who did what)
let alice_actions = vault.audit_by_entity("user:alice");

// Query by time range
let recent = vault.audit_since(timestamp_millis);
let last_10 = vault.audit_recent(10);

// AuditEntry contains:
// - entity: who performed the operation
// - secret_key: which secret was accessed
// - operation: Get, Set, Delete, Rotate, Grant, Revoke, List
// - timestamp: unix milliseconds
```

Operations logged: `Get`, `Set`, `Delete`, `Rotate`, `Grant { to, permission }`, `Revoke { from }`, `List`.

## Secret Versioning

Secrets maintain version history with configurable retention:

```rust
let config = VaultConfig {
    max_versions: 5,  // Keep last 5 versions (default)
    ..Default::default()
};

// Each set/rotate creates a new version
vault.set(Vault::ROOT, "api_key", "v1")?;
vault.rotate(Vault::ROOT, "api_key", "v2")?;
vault.rotate(Vault::ROOT, "api_key", "v3")?;

// Get current version number
let version = vault.current_version(Vault::ROOT, "api_key")?;  // 3

// List all versions
let versions = vault.list_versions(Vault::ROOT, "api_key")?;
// Returns Vec<VersionInfo> with version number and timestamp

// Get specific version
let old_value = vault.get_version(Vault::ROOT, "api_key", 1)?;  // "v1"

// Rollback to previous version
vault.rollback(Vault::ROOT, "api_key", 2)?;  // Now current is "v2"
```

Old versions are automatically pruned when `max_versions` is exceeded.

## Shell Commands

```
> VAULT INIT                                  Initialize vault from NEUMANN_VAULT_KEY env
> VAULT IDENTITY 'node:alice'                 Set current identity
> VAULT NAMESPACE 'team:backend'              Set current namespace

> VAULT SET 'api_key' 'sk-123'                Store encrypted secret
> VAULT GET 'api_key'                         Retrieve secret
> VAULT GET 'api_key' VERSION 2               Get specific version
> VAULT DELETE 'api_key'                      Delete secret
> VAULT LIST 'prefix:*'                       List accessible secrets
> VAULT ROTATE 'api_key' 'new'                Rotate secret value
> VAULT VERSIONS 'api_key'                    List version history
> VAULT ROLLBACK 'api_key' VERSION 2          Rollback to version

> VAULT GRANT 'user:bob' ON 'api_key'         Grant admin access (default)
> VAULT GRANT 'user:bob' ON 'api_key' READ    Grant read-only access
> VAULT GRANT 'user:bob' ON 'api_key' WRITE   Grant write access
> VAULT GRANT 'user:bob' ON 'api_key' TTL 3600   Grant with 1-hour expiry
> VAULT REVOKE 'user:bob' ON 'api_key'        Revoke access

> VAULT AUDIT 'api_key'                       View audit log for secret
> VAULT AUDIT BY 'user:alice'                 View audit log for entity
> VAULT AUDIT RECENT 10                       View last 10 operations
```

## API Reference

### Vault

```rust
pub struct Vault {
    // Encrypted storage with graph-based access control
}

impl Vault {
    pub const ROOT: &'static str = "node:root";

    // Construction
    pub fn new(master_key: &[u8], graph: Arc<GraphEngine>,
               store: TensorStore, config: VaultConfig) -> Result<Self>;
    pub fn from_env(graph: Arc<GraphEngine>, store: TensorStore) -> Result<Self>;

    // Core Operations
    pub fn set(&self, requester: &str, key: &str, value: &str) -> Result<()>;
    pub fn get(&self, requester: &str, key: &str) -> Result<String>;
    pub fn delete(&self, requester: &str, key: &str) -> Result<()>;
    pub fn rotate(&self, requester: &str, key: &str, new_value: &str) -> Result<()>;
    pub fn list(&self, requester: &str, pattern: &str) -> Result<Vec<String>>;

    // Access Control
    pub fn grant(&self, requester: &str, entity: &str, key: &str) -> Result<()>;
    pub fn grant_with_permission(&self, requester: &str, entity: &str,
                                  key: &str, permission: Permission) -> Result<()>;
    pub fn grant_with_ttl(&self, requester: &str, entity: &str,
                          key: &str, ttl: Duration) -> Result<()>;
    pub fn revoke(&self, requester: &str, entity: &str, key: &str) -> Result<()>;
    pub fn get_permission(&self, entity: &str, key: &str) -> Result<Permission>;

    // Versioning
    pub fn get_version(&self, requester: &str, key: &str, version: u32) -> Result<String>;
    pub fn list_versions(&self, requester: &str, key: &str) -> Result<Vec<VersionInfo>>;
    pub fn current_version(&self, requester: &str, key: &str) -> Result<u32>;
    pub fn rollback(&self, requester: &str, key: &str, version: u32) -> Result<()>;

    // Audit
    pub fn audit_log(&self, key: &str) -> Vec<AuditEntry>;
    pub fn audit_by_entity(&self, entity: &str) -> Vec<AuditEntry>;
    pub fn audit_since(&self, since_millis: i64) -> Vec<AuditEntry>;
    pub fn audit_recent(&self, limit: usize) -> Vec<AuditEntry>;

    // Scoped Access
    pub fn scope(&self, entity: &str) -> ScopedVault;
    pub fn namespace(&self, namespace: &str, identity: &str) -> NamespacedVault;
}

// Permission levels
pub enum Permission {
    Read,   // get, list
    Write,  // + set, rotate
    Admin,  // + delete, grant, revoke
}

// Audit entry
pub struct AuditEntry {
    pub entity: String,
    pub secret_key: String,
    pub operation: AuditOperation,
    pub timestamp: i64,
}

// Version info
pub struct VersionInfo {
    pub version: u32,
    pub created_at: i64,
}
```

### VaultConfig

```rust
pub struct VaultConfig {
    // Key derivation
    pub salt: Option<[u8; 16]>,      // Random if not provided
    pub argon2_memory_cost: u32,     // Default: 65536 (64MB)
    pub argon2_time_cost: u32,       // Default: 3 iterations
    pub argon2_parallelism: u32,     // Default: 4 threads

    // Rate limiting
    pub rate_limit: Option<RateLimitConfig>,  // None = disabled

    // Versioning
    pub max_versions: usize,         // Default: 5
}

pub struct RateLimitConfig {
    pub max_gets: u32,               // Default: 60 per window
    pub max_lists: u32,              // Default: 10
    pub max_sets: u32,               // Default: 30
    pub max_grants: u32,             // Default: 20
    pub window: Duration,            // Default: 60 seconds
}
```

### VaultError

```rust
pub enum VaultError {
    AccessDenied(String),           // No path to secret
    InsufficientPermission(String), // Permission level too low
    NotFound(String),               // Secret doesn't exist
    RateLimited(String),            // Rate limit exceeded
    CryptoError(String),            // Encryption/decryption failed
    KeyDerivationError(String),     // Argon2 failed
    StorageError(String),           // TensorStore error
    GraphError(String),             // GraphEngine error
    InvalidKey(String),             // Malformed secret key
}
```

## Encryption

### Key Derivation

Master key derived using Argon2id:

```rust
use tensor_vault::{MasterKey, VaultConfig};

let config = VaultConfig {
    argon2_memory_cost: 65536,  // 64MB
    argon2_time_cost: 3,        // 3 iterations
    argon2_parallelism: 4,      // 4 threads
    ..Default::default()
};

let key = MasterKey::derive(b"password", &config)?;
```

### Encryption

AES-256-GCM with random 12-byte nonce per encryption:

```rust
use tensor_vault::Cipher;

let cipher = Cipher::new(master_key);
let (ciphertext, nonce) = cipher.encrypt(b"secret data")?;
let plaintext = cipher.decrypt(&ciphertext, &nonce)?;
```

## Storage Format

Secrets use a two-tier storage model for security:

### Metadata Tensor (pointer indirection)

Storage key: `_vk:{HMAC(key)}` - key name is obfuscated via HMAC

```
_vk:{obfuscated} -> {
    _blob: Pointer -> ciphertext blob storage key
    _nonce: Bytes (12 bytes)
    _key_enc: Bytes (encrypted original key name)
    _key_nonce: Bytes (nonce for key encryption)
    _creator_obf: Bytes (XOR-obfuscated creator)
    _created_obf: Bytes (XOR-obfuscated timestamp)
    _rotator_obf: Bytes (optional, obfuscated)
    _rotated_obf: Bytes (optional, obfuscated)
}
```

### Ciphertext Blob (indirection target)

Storage key: `_vs:{HMAC(key, nonce)}` - random-looking storage ID

```
_vs:{storage_id} -> {
    _data: Bytes (padded + encrypted secret)
}
```

### Access Control Nodes

```
vault_secret:{key} -> {
    _type: "vault_secret",
    _secret_key: String,
}
```

## Audit Trail

Every successful `get()` creates a permanent audit edge:

```
vault_secret:api_key ──ACCESSED──> user:alice
```

Edge direction is reversed (secret -> accessor) so audit edges don't grant access.

## Environment Variable

Set master key via environment:

```bash
# Generate a 32-byte key and base64 encode it
export NEUMANN_VAULT_KEY=$(openssl rand -base64 32)
```

```rust
let vault = Vault::from_env(graph, store)?;
```

## Security Considerations

### Core Security

1. **Master Key**: Only held in RAM, never persisted
2. **Nonce**: Random 12 bytes per encryption (never reused)
3. **Zeroize**: Keys overwritten on drop
4. **No Bypass**: Every operation checks graph path (except `node:root`)
5. **Audit**: Access logged permanently in graph

### Obfuscation Layers

The vault implements multiple obfuscation layers to hide patterns:

| Layer | Purpose | Implementation |
|-------|---------|----------------|
| **Key Obfuscation** | Hide secret names | HMAC-BLAKE2b hash of key name |
| **Pointer Indirection** | Hide storage patterns | Ciphertext in separate blob |
| **Length Padding** | Hide plaintext size | Pad to fixed sizes (256B/1K/4K/16K) |
| **Metadata Encryption** | Hide creator/timestamps | XOR with BLAKE2b-derived keystream |

### Padding Sizes

| Plaintext Size | Padded Size |
|----------------|-------------|
| 0-240 bytes | 256 bytes |
| 241-1000 bytes | 1 KB |
| 1001-4000 bytes | 4 KB |
| 4001+ bytes | 16 KB |

### Tensor Structure Usage

The vault leverages tensor store features for security:

- **TensorValue::Pointer**: Indirection to ciphertext blobs
- **TensorValue::Bytes**: Binary storage for ciphertext/nonces
- **Separate tensors**: Metadata and ciphertext stored separately

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Key derivation (Argon2id) | ~80ms | 64MB memory cost |
| set (1KB) | ~29µs | Includes encryption + versioning |
| get (1KB) | ~24µs | Includes decryption + audit |
| set (10KB) | ~93µs | Scales with data size |
| get (10KB) | ~91µs | |
| Access check (shallow) | ~6µs | Direct edge |
| Access check (deep, 10 hops) | ~17µs | BFS traversal |
| grant | ~18µs | Creates graph edge |
| revoke | ~1.1ms | Edge deletion + TTL cleanup |
| list (100 secrets) | ~291µs | Pattern matching + access check |
| list (1000 secrets) | ~2.7ms | Scales linearly |

**Bottlenecks**:
- `revoke` is slow due to graph edge deletion and TTL tracker cleanup
- `list` scales linearly with secret count (requires decrypting each key name)
- `set` overhead increased due to versioning (stores previous versions)

## Test Coverage

| Module | Coverage |
|--------|----------|
| lib.rs | 96.59% |
| access.rs | 97.99% |
| audit.rs | 95.86% |
| rate_limit.rs | 91.77% |
| ttl.rs | 95.27% |
| obfuscation.rs | 97.81% |
| key.rs | 97.26% |
| encryption.rs | 96.59% |
| **Total** | **96.34%** |

## Dependencies

- `aes-gcm`: AES-256-GCM encryption
- `argon2`: Key derivation
- `blake2`: HMAC and obfuscation hashing
- `rand`: Nonce generation
- `zeroize`: Secure memory cleanup
- `base64`: Environment variable encoding
- `dashmap`: Concurrent rate limit tracking
- `serde`: Audit log serialization
