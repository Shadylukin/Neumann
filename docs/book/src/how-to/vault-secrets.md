# How to: Store and Retrieve Vault Secrets

Step-by-step guides for common Tensor Vault operations: storing secrets,
retrieving them, rotating values, managing versions, and using advanced features
like transit encryption and batch operations.

For the full type reference, see the
[Tensor Vault API Reference](../reference/api/tensor-vault.md). For the
cryptographic design rationale, see
[Vault Cryptography Design](../explanation/vault-crypto.md).

## Initialize a Vault

```rust
use tensor_vault::{Vault, VaultConfig, Permission};
use graph_engine::GraphEngine;
use tensor_store::TensorStore;
use std::sync::Arc;

let graph = Arc::new(GraphEngine::new());
let store = TensorStore::new();
let vault = Vault::new(b"master_password", graph, store, VaultConfig::default())?;
```

Via the shell:

```text
VAULT INIT
```

The shell reads the master key from the `NEUMANN_VAULT_KEY` environment
variable (base64-encoded, 32 bytes).

## Store a Secret

Only `node:root` can create new secrets:

```rust
vault.set(Vault::ROOT, "api_key", "sk-secret123")?;
```

Shell:

```text
VAULT SET 'api_key' 'sk-secret123'
```

## Retrieve a Secret

The requester must have at least Read permission:

```rust
let value = vault.get("user:alice", "api_key")?;
```

Shell:

```text
VAULT GET 'api_key'
```

## Grant Access

Grant different permission levels to entities:

```rust
// Read-only
vault.grant_with_permission(Vault::ROOT, "user:alice", "api_key", Permission::Read)?;

// Read + Write
vault.grant_with_permission(Vault::ROOT, "user:writer", "api_key", Permission::Write)?;

// Full access (includes grant/revoke)
vault.grant_with_permission(Vault::ROOT, "user:admin", "api_key", Permission::Admin)?;
```

Shell:

```text
VAULT GRANT 'user:alice' ON 'api_key' READ
VAULT GRANT 'user:writer' ON 'api_key' WRITE
VAULT GRANT 'user:admin' ON 'api_key'
```

## Grant Temporary Access (TTL)

TTL grants auto-revoke after the specified duration:

```rust
use std::time::Duration;

vault.grant_with_ttl(
    Vault::ROOT,
    "agent:temp",
    "api_key",
    Permission::Read,
    Duration::from_secs(3600),  // 1 hour
)?;
```

Shell:

```text
VAULT GRANT 'agent:temp' ON 'api_key' TTL 3600
```

Cleanup happens opportunistically on the next `get()` call.

## Revoke Access

```rust
vault.revoke(Vault::ROOT, "user:alice", "api_key")?;
```

Shell:

```text
VAULT REVOKE 'user:alice' ON 'api_key'
```

## Rotate a Secret

Rotation creates a new version while preserving history:

```rust
vault.rotate("user:writer", "api_key", "new_value")?;
```

Shell:

```text
VAULT ROTATE 'api_key' 'new_value'
```

## Manage Secret Versions

```rust
// Check current version
let version = vault.current_version(Vault::ROOT, "api_key")?;

// List all versions
let versions = vault.list_versions(Vault::ROOT, "api_key")?;

// Get a specific version
let old_value = vault.get_version(Vault::ROOT, "api_key", 1)?;

// Rollback (creates a new version with old content)
vault.rollback(Vault::ROOT, "api_key", 1)?;
```

Shell:

```text
VAULT VERSIONS 'api_key'
VAULT GET 'api_key' VERSION 2
VAULT ROLLBACK 'api_key' VERSION 1
```

## Store a Secret with Expiration

Secrets can have their own TTL, distinct from grant TTLs:

```rust
vault.set_with_ttl(
    Vault::ROOT,
    "temp/token",
    "abc123",
    Duration::from_secs(86400),  // 24 hours
)?;

// Check remaining lifetime
let expires_at = vault.get_expiration(Vault::ROOT, "temp/token")?;

// Remove expiration (make permanent)
vault.clear_expiration(Vault::ROOT, "temp/token")?;
```

After expiration, `get()` returns `SecretExpired`. The ciphertext remains in
storage until explicitly deleted.

## Use Transit Encryption

Transit encryption lets agents encrypt data using a vault-managed key without
ever seeing the key material:

```rust
// Encrypt data for storage outside the vault
let sealed = vault.encrypt_for("app:backend", "encryption/key", b"user PII data")?;
// Store `sealed` in your application database

// Later, decrypt it
let plaintext = vault.decrypt_as("app:backend", "encryption/key", &sealed)?;
```

The caller must have at least Read permission on the referenced secret. Transit
encryption uses a dedicated subkey, separate from the secret-encryption key.

After a master key rotation, previously issued transit ciphertexts become
undecryptable (forward secrecy).

## Use Break-Glass Emergency Access

For time-critical scenarios, bypass normal access control:

```rust
let value = vault.emergency_access(
    "ops:oncall",
    "prod/db_root",
    "P1 incident INC-4521",
    Duration::from_secs(1800),  // 30-minute window
)?;
```

Constraints:
- Rate-limited to 1 emergency access per rate-limit window
- The justification string is recorded in the audit log
- Access auto-expires after the specified duration

## Batch Operations

Read or write multiple secrets in one call:

```rust
// Read multiple secrets (partial failures visible per-key)
let results = vault.batch_get("user:alice", &["db/pass", "api/key"])?;
for (key, result) in &results {
    match result {
        Ok(value) => println!("{key}: {value}"),
        Err(e) => println!("{key}: {e}"),
    }
}

// Write multiple secrets atomically (all-or-nothing)
vault.batch_set(Vault::ROOT, &[
    ("db/pass", "new_pass"),
    ("api/key", "new_key"),
])?;
```

Both methods acquire locks in sorted key order to prevent deadlocks.

## Rotate the Master Key

Re-encrypt all secrets with a new master password:

```rust
let secrets_rotated = vault.rotate_master_key(b"new_password")?;
println!("Re-encrypted {secrets_rotated} secrets");
```

The rotation is atomic: if any re-encryption fails, the vault reverts to the old
key.

## Use Namespace Isolation

Create namespace-prefixed views for multi-tenant isolation:

```rust
let backend = vault.namespace("team:backend", "user:alice");
let frontend = vault.namespace("team:frontend", "user:bob");

// Keys are automatically prefixed
backend.set("db_password", "secret1")?;   // Stored as "team:backend:db_password"
frontend.set("api_key", "secret2")?;      // Stored as "team:frontend:api_key"

// Cross-namespace access is blocked
frontend.get("db_password")?;  // AccessDenied
```

## Use Scoped Vault

Create an entity-bound view to avoid repeating the requester:

```rust
let alice = vault.scope("user:alice");

alice.get("api_key")?;  // Same as vault.get("user:alice", "api_key")
alice.list("*")?;       // Same as vault.list("user:alice", "*")
```

## Query the Audit Log

```rust
// By secret
let entries = vault.audit_log("api_key");

// By entity
let alice_actions = vault.audit_by_entity("user:alice");

// By time
let recent = vault.audit_since(timestamp_millis);
let last_10 = vault.audit_recent(10);
```

Shell:

```text
VAULT AUDIT 'api_key'
VAULT AUDIT BY 'user:alice'
VAULT AUDIT RECENT 10
```

## See Also

- [Tensor Vault API Reference](../reference/api/tensor-vault.md) -- complete
  type tables, error types, and configuration options
- [Vault Cryptography Design](../explanation/vault-crypto.md) -- encryption
  architecture and security model rationale
- [How to: Vault Access Control](vault-access-control.md) -- graph-based
  permissions, delegation, and security audits
