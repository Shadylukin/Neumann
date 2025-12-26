# Tensor Vault

Module 9 of Neumann. Secure secret storage with AES-256-GCM encryption and graph-based access control.

## Design Principles

1. **Encryption at Rest**: All secrets encrypted with AES-256-GCM
2. **Topological Access Control**: Access determined by graph path, not ACLs
3. **Zero Trust**: No bypass mode; `node:root` is the only universal accessor
4. **Memory Safety**: Keys zeroized on drop via `zeroize` crate
5. **Permanent Audit Trail**: Access logged as graph edges
6. **Defense in Depth**: Multiple obfuscation layers hide patterns

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
| lib.rs | 95.88% |
| obfuscation.rs | 98.94% |
| access.rs | 98.68% |
| key.rs | 97.26% |
| encryption.rs | 96.59% |

## Dependencies

- `aes-gcm`: AES-256-GCM encryption
- `argon2`: Key derivation
- `blake2`: HMAC and obfuscation hashing
- `rand`: Nonce generation
- `zeroize`: Secure memory cleanup
- `base64`: Environment variable encoding
