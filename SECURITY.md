# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously, especially given that Neumann includes:

- `tensor_vault`: AES-256-GCM encrypted secret storage
- `tensor_blob`: Content-addressable storage with integrity verification
- Cryptographic key derivation (Argon2id)

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them via:

1. **Email**: [lukin@scrunchee.ai](mailto:lukin@scrunchee.ai)
   - Use subject line: `[SECURITY] Neumann: Brief description`
   - Include "SECURITY" in the subject

2. **GitHub Security Advisories**:
   - Go to the [Security tab](https://github.com/Shadylukin/Neumann/security/advisories)
   - Click "Report a vulnerability"

### What to Include

- Type of vulnerability (e.g., cryptographic weakness, injection, memory safety)
- Affected component (e.g., `tensor_vault`, `tensor_blob`)
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 7 days
- **Fix timeline**: Depends on severity
  - Critical: 24-72 hours
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release

### After Reporting

1. We'll confirm receipt and begin investigation
2. We'll keep you updated on progress
3. Once fixed, we'll coordinate disclosure timing with you
4. We'll credit you in the security advisory (unless you prefer anonymity)

## Security Measures

Neumann implements several security measures:

### Cryptography (tensor_vault)

- AES-256-GCM for secret encryption
- Argon2id for key derivation (memory-hard, side-channel resistant)
- Secure key zeroization on drop
- Graph-based access control

### Code Quality

- `forbid(unsafe_code)` in most crates
- Miri testing for undefined behavior detection
- `cargo audit` in CI for dependency vulnerabilities
- Strict Clippy lints

### Integrity (tensor_blob)

- SHA-256 content addressing
- Checksum verification on read
- Integrity repair tools

## Scope

In scope:

- All Rust code in this repository
- Cryptographic implementations
- Access control logic
- Data integrity mechanisms

Out of scope:

- Denial of service (unless caused by small inputs)
- Issues in dependencies (report upstream, but let us know)
- Social engineering

## Recognition

We appreciate security researchers who help keep Neumann secure. With your
permission, we'll acknowledge your contribution in:

- Security advisories
- Release notes
- CONTRIBUTORS.md (if created)

Thank you for helping keep Neumann and its users safe.
