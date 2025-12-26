# Tensor Vault

Module 9 of Neumann. Secure secret storage with AES-256-GCM encryption and graph-based access control.

## Design Principles

1. **Encryption at Rest**: All secrets encrypted with AES-256-GCM
2. **Topological Access Control**: Access determined by graph path, not ACLs
3. **Zero Trust**: No bypass mode; `node:root` is the only universal accessor
4. **Memory Safety**: Keys zeroized on drop via `zeroize` crate
5. **Permanent Audit Trail**: Access logged as graph edges

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

// Grant access to an entity
vault.grant(Vault::ROOT, "user:alice", "api_key")?;

// Alice can now retrieve the secret
let value = vault.get("user:alice", "api_key")?;
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

## Shell Commands

```
> VAULT INIT                      Initialize vault from NEUMANN_VAULT_KEY env
> VAULT IDENTITY 'node:alice'     Set current identity

> VAULT SET 'api_key' 'sk-123'    Store encrypted secret
> VAULT GET 'api_key'             Retrieve secret
> VAULT DELETE 'api_key'          Delete secret
> VAULT LIST 'prefix:*'           List accessible secrets
> VAULT ROTATE 'api_key' 'new'    Rotate secret value

> VAULT GRANT 'user:bob' ON 'api_key'   Grant access
> VAULT REVOKE 'user:bob' ON 'api_key'  Revoke access
```

## API Reference

### Vault

```rust
pub struct Vault {
    // Encrypted storage with graph-based access control
}

impl Vault {
    pub const ROOT: &'static str = "node:root";

    pub fn new(
        master_key: &[u8],
        graph: Arc<GraphEngine>,
        store: TensorStore,
        config: VaultConfig,
    ) -> Result<Self>;

    pub fn from_env(graph: Arc<GraphEngine>, store: TensorStore) -> Result<Self>;

    pub fn set(&self, requester: &str, key: &str, value: &str) -> Result<()>;
    pub fn get(&self, requester: &str, key: &str) -> Result<String>;
    pub fn delete(&self, requester: &str, key: &str) -> Result<()>;
    pub fn rotate(&self, requester: &str, key: &str, new_value: &str) -> Result<()>;
    pub fn list(&self, requester: &str, pattern: &str) -> Result<Vec<String>>;

    pub fn grant(&self, requester: &str, entity: &str, key: &str) -> Result<()>;
    pub fn revoke(&self, requester: &str, entity: &str, key: &str) -> Result<()>;

    pub fn scope(&self, entity: &str) -> ScopedVault;
}
```

### VaultConfig

```rust
pub struct VaultConfig {
    pub salt: Option<[u8; 16]>,      // Random if not provided
    pub argon2_memory_cost: u32,     // Default: 65536 (64MB)
    pub argon2_time_cost: u32,       // Default: 3 iterations
    pub argon2_parallelism: u32,     // Default: 4 threads
}
```

### VaultError

```rust
pub enum VaultError {
    AccessDenied(String),       // No path to secret
    NotFound(String),           // Secret doesn't exist
    CryptoError(String),        // Encryption/decryption failed
    KeyDerivationError(String), // Argon2 failed
    StorageError(String),       // TensorStore error
    GraphError(String),         // GraphEngine error
    InvalidKey(String),         // Malformed secret key
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

Secrets stored as `TensorData`:

```
_vault:{key} -> {
    _ciphertext: Bytes,
    _nonce: Bytes,
    _created_by: String,
    _created_at: Int (timestamp),
    _rotated_by: String (optional),
    _rotated_at: Int (optional),
}
```

Access control nodes:

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

1. **Master Key**: Only held in RAM, never persisted
2. **Nonce**: Random 12 bytes per encryption (never reused)
3. **Zeroize**: Keys overwritten on drop
4. **No Bypass**: Every operation checks graph path (except `node:root`)
5. **Audit**: Access logged permanently in graph

## Performance

| Operation | Time |
|-----------|------|
| Key derivation (Argon2id) | ~300ms |
| Encrypt 1KB | ~1us |
| Decrypt 1KB | ~1us |
| Access check (shallow) | ~1us |
| Access check (deep, 10 hops) | ~10us |

## Test Coverage

| Module | Coverage |
|--------|----------|
| lib.rs | 97.31% |
| encryption.rs | 96.59% |
| key.rs | 97.14% |
| access.rs | 98.66% |

## Dependencies

- `aes-gcm`: AES-256-GCM encryption
- `argon2`: Key derivation
- `rand`: Nonce generation
- `zeroize`: Secure memory cleanup
- `base64`: Environment variable encoding
