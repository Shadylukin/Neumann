# Neumann Roadmap

This document outlines the planned development milestones for Neumann.

## Version Policy

Neumann follows [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.x.0): New features, backward compatible
- **PATCH** (0.0.x): Bug fixes, backward compatible

## Current Status: Pre-1.0

Neumann is currently in active development. APIs may change between minor versions
until 1.0.0 is released.

## Milestones

### v0.9.0 - Production Hardening (Current)

- [x] Error message sanitization for production
- [x] SPDX license headers on all source files
- [x] API documentation published
- [x] Comprehensive test coverage (>95% per crate)
- [x] Fuzz testing for all parsers and serialization
- [ ] Performance benchmarks published
- [ ] Security audit complete

### v0.10.0 - API Stabilization

- [ ] Public API review and documentation
- [ ] Deprecation warnings for unstable APIs
- [ ] Migration guides for breaking changes
- [ ] Client SDK stabilization (Python, TypeScript)
- [ ] gRPC API versioning (v1 namespace)

### v1.0.0 - Stable Release

Target: Q3 2026

**Stability guarantees:**
- No breaking changes to public APIs without major version bump
- Minimum 12 months support for each major version
- Security patches backported to supported versions

**Features for 1.0:**
- Stable query language syntax
- Stable gRPC service definitions
- Stable storage format with migration support
- Production-ready distributed consensus
- Comprehensive monitoring and observability

### v1.1.0 - Enterprise Features

- Multi-tenancy support
- Advanced access control (RBAC)
- Audit log retention policies
- Backup and restore improvements
- Cluster management UI

### v2.0.0 - Next Generation

Planning phase. Potential features:
- GPU-accelerated tensor operations
- Native Python/TypeScript embedding
- Federated learning support
- Real-time streaming queries

## Breaking Change Policy

### Before 1.0.0

- Breaking changes may occur in minor versions
- Changes are documented in release notes
- Migration guides provided when feasible

### After 1.0.0

1. **Deprecation**: Feature marked deprecated with warning
2. **Grace Period**: Minimum 2 minor versions before removal
3. **Removal**: Only in next major version
4. **Migration**: Guide and tooling provided

### Exceptions

Security vulnerabilities may require immediate breaking changes without
the standard deprecation period.

## Deprecation Policy

Deprecated features:

1. Emit compile-time warnings
2. Are documented in CHANGELOG.md
3. Include migration instructions
4. Remain functional for the grace period

Example deprecation:

```rust
#[deprecated(since = "0.10.0", note = "Use `new_method` instead")]
pub fn old_method() { ... }
```

## Support Matrix

| Version | Status       | Security Fixes | Bug Fixes |
|---------|--------------|----------------|-----------|
| 1.x     | (planned)    | Yes            | Yes       |
| 0.x     | Active       | Yes            | Yes       |

## Release Schedule

- **Major releases**: As needed for breaking changes
- **Minor releases**: Monthly during active development
- **Patch releases**: As needed for critical fixes

## Feature Requests

Feature requests are tracked in [GitHub Issues](https://github.com/Shadylukin/Neumann/issues).

For major features, we use an RFC process:

1. Open an issue with the `rfc` label
2. Community discussion period (minimum 2 weeks)
3. Core team review and decision
4. Implementation (if accepted)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to Neumann's development.
