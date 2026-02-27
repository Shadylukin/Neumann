# Securing Secrets with Vault

Store and manage secrets using Neumann's encrypted vault. By the end you
will have secrets stored with AES-256-GCM encryption and graph-based access
control.

## Prerequisites

- Neumann installed ([Installation](../how-to/installation.md))
- A running Neumann shell

## Step 1: Start the Shell

```bash
neumann --wal-dir ./vault-data
```

## Step 2: Store a Secret

Store an API key as an encrypted secret:

```sql
VAULT PUT 'api-keys/openai' 'sk-proj-abc123def456'
```

Verify it was stored:

```sql
VAULT GET 'api-keys/openai'
```

You should see the decrypted value `sk-proj-abc123def456`.

## Step 3: Store More Secrets

```sql
VAULT PUT 'api-keys/stripe' 'sk_live_xyz789'
VAULT PUT 'database/production/password' 'super-secret-db-pass'
VAULT PUT 'certificates/tls-key' '-----BEGIN PRIVATE KEY-----...'
```

## Step 4: List Secrets

```sql
VAULT LIST 'api-keys/'
```

You should see `api-keys/openai` and `api-keys/stripe`.

## Step 5: Set Up Access Control

Create access policies using the graph engine. Define roles and
permissions:

```sql
NODE CREATE role { name: 'developer' }
NODE CREATE role { name: 'ops' }
NODE CREATE secret_group { name: 'api-keys' }
NODE CREATE secret_group { name: 'database' }
```

Connect roles to secret groups:

```sql
EDGE CREATE 'node:1' -> 'node:3' : can_read
EDGE CREATE 'node:2' -> 'node:3' : can_read
EDGE CREATE 'node:2' -> 'node:4' : can_read
EDGE CREATE 'node:2' -> 'node:4' : can_write
```

This gives:

- Developers: read access to API keys
- Ops: read access to API keys, read+write access to database secrets

## Step 6: Update a Secret

```sql
VAULT PUT 'api-keys/openai' 'sk-proj-new-key-789'
```

The old value is replaced. Retrieve to confirm:

```sql
VAULT GET 'api-keys/openai'
```

## Step 7: Delete a Secret

```sql
VAULT DELETE 'api-keys/stripe'
```

Verify it was removed:

```sql
VAULT GET 'api-keys/stripe'
```

This should return an error indicating the secret was not found.

## Verification

You should have:

- Secrets stored with encryption (VAULT PUT/GET working)
- Hierarchical secret paths (`api-keys/`, `database/`)
- Graph-based access control nodes and edges
- Secret update and deletion working

## Next Steps

- [Vault Access Control](../how-to/vault-access-control.md) -- advanced
  access policies
- [Configuration Reference](../reference/configuration.md) -- vault config
  options
- [Building a Knowledge Graph](knowledge-graph.md) -- more graph patterns
