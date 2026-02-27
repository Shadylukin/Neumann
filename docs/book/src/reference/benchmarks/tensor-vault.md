# tensor_vault Benchmarks

The tensor_vault crate provides AES-256-GCM encrypted secret storage with
graph-based access control, permission levels, TTL grants, rate limiting,
namespace isolation, audit logging, and secret versioning.

<!-- BENCH:START -->
## Key Derivation (Argon2id)

| Operation | Time | Peak RAM |
| --- | --- | --- |
| argon2id_derivation | 80 ms | ~64 MB |

> **Note**: Argon2id is intentionally slow to resist brute-force attacks. The
64MB memory cost is configurable via `VaultConfig`.

## Encryption/Decryption (AES-256-GCM)

| Operation | Time | Peak RAM |
| --- | --- | --- |
| set_1kb | 29 us | ~3 KB |
| get_1kb | 24 us | ~3 KB |
| set_10kb | 93 us | ~25 KB |
| get_10kb | 91 us | ~25 KB |

> **Note**: `set` includes versioning overhead (storing previous version
pointers). `get` includes audit logging.

## Access Control (Graph Path Verification)

| Operation | Time | Peak RAM |
| --- | --- | --- |
| check_shallow (1 hop) | 6 us | ~2 KB |
| check_deep (10 hops) | 17 us | ~3 KB |
| grant | 18 us | ~1 KB |
| revoke | 1.07 ms | ~1 KB |

## Secret Listing

| Operation | Time | Peak RAM |
| --- | --- | --- |
| list_100_secrets | 291 us | ~4 KB |
| list_1000_secrets | 2.7 ms | ~40 KB |

> **Note**: List includes access control checks and key name decryption for
pattern matching.
<!-- BENCH:END -->

## Analysis

- **Key derivation**: Argon2id dominates vault initialization (~80ms). This is
  by design for security.
- **Access check improved**: Path verification is now ~6us for shallow, ~17us
  for deep (85% faster than before).
- **Versioning overhead**: `set` is ~2x slower due to version tracking (stores
  pointer array).
- **Audit overhead**: Every operation logs to audit store (adds ~5-10us per
  operation).
- **Revoke performance**: ~1ms due to edge deletion, TTL tracker cleanup, and
  audit logging.
- **List scaling**: ~2.7us per secret at 1000 (includes decryption for pattern
  matching).

## Feature Performance Overhead

| Feature | Overhead |
| --- | --- |
| Permission check | ~1 us (edge type comparison) |
| Rate limit check | ~100 ns (DashMap lookup) |
| TTL check | ~50 ns (heap peek) |
| Audit log write | ~5 us (tensor store put) |
| Version tracking | ~10 us (pointer array update) |

## Security vs Performance Trade-offs

| Configuration | Key Derivation | Security |
| --- | --- | --- |
| Default (64MB, 3 iter) | ~80 ms | High |
| Fast (16MB, 1 iter) | ~25 ms | Medium |
| Paranoid (256MB, 10 iter) | ~800 ms | Very High |

## Recommendations

- **Development**: Use `Fast` configuration for quicker iteration
- **Production**: Use `Default` or `Paranoid` based on threat model
- **High-throughput**: Cache access decisions where possible
- **Audit compliance**: Accept ~5us overhead for complete audit trail
